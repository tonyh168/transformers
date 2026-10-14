# Mamba2 / SSM 精读：不用注意力的序列建模

> 目标：搞清楚「SSM（状态空间模型）」这条**完全不用注意力**的路线。两个收获：(1) 知道注意力的 `O(n²)` 为什么是长序列的死结、SSM 用什么代价换掉它；
> (2) 拿到一把读别的线性模型（Zamba2 / Falcon-H1 / Bamba / Jamba）的钥匙。
> **本篇概念最陌生**：状态、离散化、选择性扫描、chunk 都是新词，所以**不推导任何数学**，全程用类比；只有确认过行号的代码才贴出来。
>
> 源文件：`models/mamba2/modeling_mamba2.py`（921 行）、`configuration_mamba2.py`（110 行）、
> `cache_utils.py`（`LinearAttentionLayer` 系列）、`masking_utils.py`（`create_recurrent_attention_mask`）、`integrations/hub_kernels.py`（kernel 分发）。
> **本模型没有 `modular_` 文件**（实测 `grep -c modular` = 0），按 [2A5](./2A5_modular_code_generation.md) 规则直读 `modeling_mamba2.py`；
> 里面仅存的两处 `# Copied from`（`:678` / `:699`）都指向 `models/mamba/modeling_mamba.py`，是 Mamba1 的输出 dataclass，与结构无关。

---

## 速读区（TL;DR）

**一句话结论**：SSM 用**一个固定大小的状态**替代无限增长的 KV cache，代价是**有损压缩**（精确召回变弱）。
让这条路可行的关键是 **dual form**——同一组权重，训练/prefill 走「卷积形式」（可并行），解码走「递推形式」（`O(1)` 内存）；
在 `modeling_mamba2.py` 里两条路各对应一个函数：`mamba2_chunk_scan`（`:254`）与 `mamba2_selective_state_update`（`:192`）。

### 必须记住的断言

| # | 断言 | 出处（文件:行号） |
|---|------|------------------|
| 1 | mamba2 **没有 `modular_` 文件**，直接读 `modeling_mamba2.py`（921 行）；文件里仅存的两处 `# Copied from`（`:678` / `:699`）是 Mamba1 的输出 dataclass，与结构无关 | `:17-22`；验证 `:963-966` |
| 2 | 架构硬约束：`hidden_size × expand` **必须等于** `num_heads × head_dim`（默认两边都是 8192） | `configuration_mamba2.py:96` |
| 3 | `layer_types` 恒为 `["linear_attention"] * num_hidden_layers`（全模型没有 attention 层） | `configuration_mamba2.py:105` |
| 4 | 默认值：`conv_dim = 8192 + 2×8×128 = 10240`，`projection_size = 8192 + 10240 + 128 = 18560`，一次 `in_proj` 切成 `[8192(gate), 10240(B/C/x), 128(dt)]` | `modeling_mamba2.py:392`、`:403`、`:486` |
| 5 | 形式切换的开关就一句：`if use_precomputed_states and seq_len == 1:`（有历史状态 + 只来一个 token → 递推） | `:532`（`# Recurrent form` `:531` / `# Chunk form` `:556`） |
| 6 | `A_log` 是**每头一个标量** `[num_heads]`，恢复成 `A = -torch.exp(A_log.float())`——取负保证 `A < 0` ⇒ `Ā ∈ (0,1)` ⇒ 状态只衰减、不爆炸 | `:416`、`:459`；初始化 `:429-430` |
| 7 | `dt` 宽度是 **`num_heads`**（每头一个标量，广播到整个 `head_dim`）；Mamba1 是 `intermediate_size` 且另有 `dt_proj` | `:412`、`:534`；第八节表（`:904-905`） |
| 8 | **Mamba2 没有 `is_mambapy` / `use_mambapy` 开关**（那是 Mamba1 的 config 字段）；换实现完全交给 kernel 分发机制 | 4.5 节 `:558-566`；验证 `:1003-1007` |
| 9 | kernel 分发三层优先级：**Hf kernels → 原包 → torch only**；纯 torch 下融合函数直接 `return None`，代码继续走正常流程 | `hub_kernels.py:828-831`、`modeling_mamba2.py:188`、`:490-498` |
| 10 | **HF 侧函数名 ≠ kernel 名**：`mamba2_chunk_scan` 对应的 kernel 名是 `mamba_chunk_scan_combined` | 4.5 节表（`:545-551`） |
| 11 | `time_step_rank`（`:375`）是**只被赋值、从未被使用**的死字段（Mamba1 遗留，Mamba2 已无 `dt_proj`） | `:272-275`；验证 `:983-984` |
| 12 | cache 存**两样常量大小的东西**：`conv_states`（`[B,10240,4]`，80 KiB/层）+ `recurrent_states`（`[B,128,64,128]`，2 MiB/层），且是**原地 `copy_` 而非 append** | `:784-787`、`cache_utils.py:1089` |

### 最容易踩的坑 / 反直觉发现

- **「scan 到底是谁在算」最容易想当然**：答案是三层优先级，而且 **Mamba2 里没有 `is_mambapy` 开关**（`grep` 在 mamba2 目录零命中）——所以 `Mamba2Config` 里看不到任何「用哪个实现」的字段。
- **`Mamba2Block` 只有一条子层**：attention 与 MLP 被揉进了同一个 `Mamba2Mixer`，而 Llama 是两条；所以同宽度下两者参数量不可直接比较。
- **Mamba2 完全没有位置编码**：位置信息隐含在「递推的顺序」里，卷积核 `kernel_size=4` 只带一点局部顺序感——这是两种架构最本质的哲学差异之一。
- **门控顺序反了会更差**：gated RMSNorm 是「先乘 `silu(gate)` 再算均方」，门控**参与统计量**，被关掉的通道不会白占一份分母。
- **名字陷阱**：gated 版叫 `MambaRMSNormGated`，**不带 `2`**（历史遗留，`falcon_h1` 也复用它）；block 里的普通版才叫 `Mamba2RMSNorm`（`:591`）。
- **混合架构拿不到全图编译红利**：`falcon_h1` 写着 `_can_compile_fullgraph = False`，而 `Mamba2PreTrainedModel` 是 `True`（`:640`）。

### 读完应该能回答的问题

- 为什么 RNN 省内存却输给了 Transformer？SSM 用什么绕开了这个死穴（dual form 的两条路分别对应哪个函数）？
- Mamba2 的 `in_proj` 一次投影出哪三块？`conv_dim` / `projection_size` 的默认值各是多少？
- `recurrent_states` 与 2A3 的 `DynamicLayer` 在写入方式上有什么本质差异？这个差异换来了什么能力？
- 纯 SSM 的弱点是什么？仓库里哪几个模型用「SSM + 若干层标准 attention」的混法，比例各是多少？
- 为什么 Mamba2 里看不到任何「用哪个实现」的 config 字段？（想一想 `grep is_mambapy` 的结果）

---

## 前置知识

- [1M1_llama_structure.md](./1M1_llama_structure.md)：Transformer 的 DecoderLayer / RMSNorm / 残差（本篇全程的对照面）
- [2A5_modular_code_generation.md](./2A5_modular_code_generation.md)：为什么本篇能直读 `modeling_mamba2.py`
- （可选）[2A3_kv_cache.md](./2A3_kv_cache.md)：KV cache 的常规形态；本篇讲它的「替代品」

---

## 一、为什么要有「不用注意力」这条路线

### 1.1 注意力的账本

Llama 的注意力每步做「当前 token 和**所有过去的 token** 比一比」（`softmax(QK^T)V`），计算量随序列长度**平方**增长；推理内存上 KV cache 每生成一个 token 就多存一份 K/V，**长度与序列长度成正比**。用 Llama-3-8B（32 层 / 8 KV 头 / head_dim 128 / bf16）手算：

| 上下文长度 | KV cache 大小 |
|-----------|--------------|
| 1 | 128 KiB |
| 8K | 1 GiB |
| 128K | **16 GiB** |

算式：每 token 每层 `2 × 8 × 128 = 2048` 个元素 = 4 KiB（bf16），× 32 层 = 128 KiB/token；128K tokens × 128 KiB = 16 GiB。**这部分内存纯粹是「记住历史」的开销。**

### 1.2 一个很自然的想法：像 RNN 一样只维护一个状态

RNN 维护一个**固定大小的向量** `h`，每读一个 token 更新一次：`h_t = f(h_{t-1}, x_t)`。好处：计算量 `O(n)`、推理内存 `O(1)`。坏处：**训练不能并行**（要算第 100 步必须先算完第 99 步），还有梯度消失/爆炸。这就是「RNN 明明省内存，为什么 Transformer 赢了」的答案：**训练可并行性**比推理省内存更重要。

### 1.3 SSM 的核心贡献：dual form（两种形态）

Mamba/SSM 的全部价值，在于找到一个**特殊的递推结构**，让同一模型有**两种等价形态**：

| 形态 | 公式长相 | 什么时候用 | 复杂度 |
|------|----------|-----------|--------|
| **卷积形式**（convolution） | 整段序列一次性算完，像一次大卷积 | **训练 / prefill** | `O(n log n)` 级别，**完全可并行** |
| **递推形式**（recurrent） | 逐 token 更新状态 `h` | **自回归解码（一次一个 token）** | `O(1)` 内存，`O(1)` 计算/步 |

```
  训练 / prefill：x[0..n] ──► 卷积形式：整段一起算（kernel 里叫 chunk scan）──► y[0..n]（可并行 · 不存历史 · 算完就丢）
  推理 / decode： x_t ──► h_t = Ā·h_{t-1} + B̄·x_t ──► y_t = C·h_t ──► y_t
                        ▲─── h（固定大小的状态，就是全部历史）───┘
```

**这张图就是本篇的核心**。代码上的对照在 `Mamba2Mixer.forward`：`modeling_mamba2.py:556` 的注释是 `# Chunk form`、`:559` 调 `mamba2_chunk_scan`；`:531` 的注释是 `# Recurrent form`、`:541` 调 `mamba2_selective_state_update`。入口判断在 `:532`：`if use_precomputed_states and seq_len == 1:`——**「有历史状态 + 只来了一个 token」= 解码步 → 走递推；否则 → 走 chunk scan。**

---

## 二、核心概念：状态空间模型（直觉版）

### 2.1 连续形式

SSM 借用了控制论描述物理系统的方程：

```
h'(t) = A · h(t) + B · x(t)      ← 状态怎么随时间演化
y(t)  = C · h(t)                  ← 从状态读出输出
```

把 `h` 想成一个**水库**：`A` 决定水自己渗漏多少（衰减，**和输入无关**，是固定参数），`B` 决定这次进水多少，`C` 决定放多少水出来。**水位就是「历史」**——你没法从水位还原每一场雨，但水位足够决定要不要开闸。

### 2.2 离散化

token 一个一个来，所以要引入**步长** `Δ`（代码里叫 `dt`）离散化：`h_t = Ā·h_{t-1} + B̄·x_t`，`y_t = C·h_t`。只要记住一件事：

> `Δ`（`dt`）是这一步「新信息写入多少」的旋钮。`Δ` 大 → 更关注**当前** token（像「遗忘门开大」）；`Δ` 小 → 更依赖**历史**。

**最主要的一点**：`Δ`、`B`、`C` 都是**由输入算出来的**（input-dependent），模型会**根据当前 token 的内容决定要不要记住它**——这就是 Mamba 论文里「**selective**（选择性）」一词的来历（`modeling_mamba2.py:362-365` 的类 docstring 原话：`∆, B, C are input-dependent ... is why Mamba is called selective state spaces`）。而 `A`、`D` 是**输入无关**的固定参数（同一 docstring：`A, D are input independent`）。

### 2.3 本篇最重要的一张图：KV cache vs SSM 状态

```
        Transformer（1M1 的 Llama）                        SSM（本篇的 Mamba2）
   layer 0..N:  K V K V K V K V ...  ← 每 token 加两格     layer 0..N:  [ h ] 固定 2 MiB ← 每 token 原地改写
   长度 ∝ seq_len：Llama-3-8B @8K→1 GiB / @128K→16 GiB    长度 = 常数：Mamba2-2.8B→约 133 MiB（128K 仍是 133 MiB）
```

133 MiB 怎么来的（**按 `configuration_mamba2.py` 默认值手算**）：

| 项 | 算式 | 大小 |
|----|------|------|
| 递推状态 `h`（每层 / 64 层） | `num_heads × head_dim × state_size = 128 × 64 × 128` = 1,048,576 元素，bf16 / ×64 | 2 MiB / 128 MiB |
| 卷积状态（每层 / 64 层） | `conv_dim × conv_kernel = 10240 × 4` = 40,960 元素，bf16 / ×64 | 80 KiB / 约 5 MiB |
| **合计** | | **约 133 MiB，恒定** |

直观换算：Mamba2 单层状态 1,048,576 个元素 ÷ Llama-3-8B「每 token 每层 2048 个元素」**≈ 512 个 token 的 KV**。也就是说 **SSM 把无限长的历史压进一个「约 512 token 容量」的固定容器**——它不是「记得更多」，而是「记得更省」，这就是它的代价，也是它的全部价值。

---

## 三、Mamba2 的模块清单（全在这一个文件里）

| 类 / 函数 | 行号 | 职责 |
|-----------|------|------|
| `pad_tensor_by_size` / `reshape_into_chunks` / `segment_sum` / `apply_mask_to_padding_states` | `:42` / `:53` / `:73` / `:93` | chunk scan 的辅助：补到 `chunk_size` 整数倍 / 重排成块 / chunk 内分段累加 / padding 位置 state 乘成 0 |
| `MambaRMSNormGated` | `:105` | **带门控的 RMSNorm**（Mamba2 特色，见第六节） |
| `causal_conv1d_update` / `causal_conv1d_fn` | `:124` / `:144` | 单步卷积（解码用）/ 整段卷积（prefill 用） |
| `mamba2_split_conv1d_scan_combined` | `:167` | 训练时的**融合**大 kernel（回退实现直接 `return None`） |
| `mamba2_selective_state_update` | `:192` | **递推形式**（解码一步） |
| `mamba2_chunk_scan` | `:254` | **卷积/chunk 形式**（训练 + prefill） |
| `Mamba2Mixer` | `:360` | 核心，SSM 层本体（第四节） |
| `Mamba2RMSNorm` | `:591` | 普通 RMSNorm（对照 `LlamaRMSNorm`） |
| `Mamba2Block` | `:608` | 一层（第五节） |
| `Mamba2PreTrainedModel` | `:635` | 基类，注意 `_is_stateful = True`（`:641`） |
| `Mamba2Model` | `:720` | 堆叠 + 更新 cache（`:782` 建 mask） |
| `Mamba2ForCausalLM` | `:822` | 加 `lm_head` |

**注意模块命名的两套体系**：**类名**都带 `2`（`Mamba2Mixer`），但 **kernel 名**沿用上游 `mamba_ssm` 包（`mamba_chunk_scan_combined`），而 **HF 侧的 torch 回退函数**又叫 `mamba2_chunk_scan`。这个不一致在 4.5 掰清楚。

---

## 四、`Mamba2Mixer` 逐行精读

### 4.0 先把数字钉死：`Mamba2Config` 的默认值

全部来自 `configuration_mamba2.py`（实测行号）：

| 字段 | 行号 | 默认值 | 含义 |
|------|------|--------|------|
| `hidden_size` / `num_hidden_layers` | `:65` / `:67` | 4096 / 64 | 宽度 / 层数 |
| `num_heads` / `head_dim` / `state_size` | `:62` / `:63` / `:66` | 128 / 64 / 128 | SSM 头数 / 每头维度 / **状态 `h` 的宽度**（「压缩后的容量」） |
| `n_groups` / `expand` / `conv_kernel` / `chunk_size` | `:74` / `:72` / `:73` / `:87` | 8 / 2 / 4 / **256** | B/C 组间共享（类似 GQA）/ 中间维度 = `hidden_size × expand` = 8192 / 短卷积核 / chunk scan 块大小 |
| `time_step_min` / `max` / `floor` / `limit` | `:81` / `:82` / `:83` / `:84` | 0.001 / 0.1 / 1e-4 / `(0.0, inf)` | `dt` 初始化范围 / 裁剪区间 |
| `use_bias` / `use_conv_bias` | `:75` / `:76` | `False` / `True` | 投影 / 卷积是否带 bias |
| `hidden_act` / `residual_in_fp32` | `:77` / `:79` | `"silu"` / `True` | 卷积后的激活 / 残差用 fp32 累加 |

两个必须放在一起看的东西：

```python
# configuration_mamba2.py:96   validate_architecture：hidden_size * expand 必须等于 num_heads * head_dim（默认两边 8192）
# configuration_mamba2.py:105  @property layer_types → ["linear_attention"] * self.num_hidden_layers
```

- 第一条约束直接决定 `in_proj` 怎么切（4.1）。
- 第二条：**所有层都是 `"linear_attention"`**，这个字符串被 `cache_utils.py` 用来选 cache 层类型（`DYNAMIC_LAYER_TYPE_MAPPING`，`cache_utils.py:1231`）、被 `masking_utils.py` 用来选 mask 函数（`LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING`，`masking_utils.py:1516`）。第七节详说。

> 实测发现：`time_step_rank` 字段（`:80`，默认 `"auto"` → `:91-93` 算成 `ceil(hidden_size/16) = 256`）在 `Mamba2Mixer.__init__` 里**只被赋值、从未被使用**：`grep -n "dt_proj\|time_step_rank" modeling_mamba2.py` 只有 `:375` 一行命中。这是 Mamba1 的遗留字段（Mamba1 的 `dt_proj` 需要它），Mamba2 已无 `dt_proj`。

### 4.1 `__init__`：`in_proj` 到底投影出了什么

```python
# modeling_mamba2.py:392
self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
# modeling_mamba2.py:403-404
projection_size = self.intermediate_size + self.conv_dim + self.num_heads
self.in_proj = nn.Linear(self.hidden_size, projection_size, bias=config.use_bias)
# modeling_mamba2.py:486   一次 in_proj 投影出三块
gate, hidden_states_B_C, dt = projected_states.split(
    [self.intermediate_size, self.conv_dim, self.num_heads], dim=-1)
```

代入默认值（`intermediate_size = 4096 × 2 = 8192`）：`conv_dim = 8192 + 2×8×128 = 10240`、`projection_size = 8192 + 10240 + 128 = 18560`。切成 `[8192 | 10240 | 128]`：

| 名字 | 宽度 | 干什么用 |
|------|------|----------|
| `gate` | 8192 | **门控**（对应 SSM 里的 `z`），最后和扫描输出相乘 |
| `hidden_states_B_C` | 10240 | 再切：`x`(8192) + `B`(1024) + `C`(1024)。**B、C、x 都要过卷积** |
| `dt` | 128 | 每头一个**离散化步长**（128 = `num_heads`） |

二次切分在 `:525`：`hidden_states, B, C = torch.split(hidden_states_B_C, [intermediate_size, n_groups*ssm_state_size, n_groups*ssm_state_size], dim=-1)`，即 `x = 8192`、`B = 8×128 = 1024`、`C = 1024`。

> **和 Mamba1 的最大差别就在这个 `in_proj`**（Mamba1 在 `models/mamba/modeling_mamba.py:318-323`）：Mamba1 的 `in_proj(hidden → intermediate×2)` 只出 `x` 和门控，再用独立的 `x_proj` 算 B/C/dt，最后还有 `dt_proj`。Mamba2 **把 B/C/dt 全折进 `in_proj`**，一次投影全出，少两次矩阵乘法、少一次 kernel 启动。

### 4.2 参数初始化：`A`、`D`、`dt_bias`

```python
# modeling_mamba2.py:414-418   self.A_log = nn.Parameter(torch.empty(self.num_heads))  # S4D real init, not discretized
self.norm = MambaRMSNormGated(self.intermediate_size, eps=self.layer_norm_epsilon); self.D = nn.Parameter(torch.empty(self.num_heads))
# modeling_mamba2.py:429-431  init：A = arange(1, num_heads+1) → copy_(A_log, log(A))；init.ones_(D)
# :399-407  dt = exp(rand*(log(max)-log(min)) + log(min)).clamp(min=floor)；inv_dt = dt + log(-expm1(-dt))；copy_(dt_bias, inv_dt)
```

三件事：① **`A_log` 是对数形态的 `A`**，初始化为 `log(1..128)`（`:429-430`），恢复时取负指数（`:459`）。② **`D` 是 skip connection，初始化为全 1**（`:431`）——「这一层输入直通到输出的比例」，状态 `h` 什么都没学到时输入也能原样流过。③ **`dt_bias` 用 softplus 的逆函数初始化**（`:406-407`）：forward 里 `dt` 会过 `softplus`（保证 `dt > 0`），所以初始化先用逆函数算 raw 值，经 `softplus` 后正好落在 `[0.001, 0.1]`。这可以类比成**遗忘门的初始化**：`dt` 被夹住，既不太接近 0（什么都记不住）也不太（只看当前 token）。

### 4.3 `forward` 全景：一张数据流图

`forward` 在 `:443-588`。把默认值代进去画出来：

```
  hidden_states  [B, S, 4096]
    │ ① in_proj（:457）→ projected_states [B, S, 18560]   ② split（:486）[8192 | 10240 | 128]
    ├─► gate(z) [B,S,8192] ──────────────────────────────────────────────────────┐
    ├─► hidden_states_B_C [B,S,10240] ③ 转置 [B,10240,S] 过 depthwise 短卷积（causal_conv1d_fn，kernel=4，:495-518）
    │                                ▼ ②' split（:525）→ x[B,S,8192] / B[B,S,1024] / C[B,S,1024]
    ├─► dt [B,S,128] ──► ④ SCAN（状态空间的核心运算）：
    │        训练/prefill = mamba2_chunk_scan（卷积形式，可并行，:559）；解码 = mamba2_selective_state_update（递推，:541）
    │        → scan_output [B,S,8192] ───────────────────────────────────────────┤
    │ ⑤ 门控 + 归一化（:584）self.norm(scan_output, gate)：先乘 silu(gate)，再做 RMSNorm
    └─► ⑥ out_proj（:587）Linear(8192 → 4096) → contextualized_states [B, S, 4096]
```

**四个必须记住的点**：

1. **`dt` 是每头一个标量**（宽度 128 = `num_heads`），不是每个通道一个（Mamba1 的 `dt` 宽度是 `intermediate_size`）。解码分支里显式广播：`dt = dt.transpose(1, 2).expand(-1, -1, self.head_dim)`（`:534`）——**一个 `dt` 管整个 `head_dim` 宽度的状态**。
2. **`A` 是每头一个标量**：`self.A_log` 形状 `[num_heads]`（`:416`），`A = -torch.exp(self.A_log.float())`（`:459`）得 `[128]`，用到时广播成 `[num_heads, head_dim, state_size]`（`:535`）。对比 Mamba1：`A_log` 形状 `[intermediate_size, state_size]`（Mamba1 `:327`）。
3. **`A` 为什么写成 `-exp(A_log)`**（`:459`）：`exp` 恒正，取负后 `A` **恒为负**。递推式 `h_t = Ā·h_{t-1} + ...` 里 `Ā = exp(Δ·A)`，`A < 0` ⇒ `Ā ∈ (0,1)` ⇒ **状态只衰减、不爆炸**；用 `A_log` 而非直接存 `A` 是纯数值稳定性技巧。
4. **`z` 的门控**（`:584`）：gate 先过 SiLU 再逐元素乘回主干，和 1M1 的 `LlamaMLP`/SwiGLU 同一思路——**用一条支路去「闸」另一条支路**，只是 Llama 在 MLP 里闸、Mamba2 在 SSM 输出上闸。

### 4.4 四条真实分支：融合 / prefill / 解码 / 无 cache

| # | 触发条件（真实行号） | 走什么 | 什么时候发生 |
|---|---------------------|--------|-------------|
| A | `:461` `if self.training and cache_params is None:` 且 `:483` `fused_output is not None` | `mamba2_split_conv1d_scan_combined`（**卷积+扫描+norm+out_proj 全融合成一个 kernel**） | 训练，且装了 kernel |
| B | `:496` `use_precomputed_states and seq_len == 1 and not ...record_past` | `causal_conv1d_update` + `:541` `mamba2_selective_state_update` | **自回归解码一步一个 token**（O(1)） |
| C | `:557` `else`（有 cache 但 seq_len > 1，或没装 kernel 的训练） | `causal_conv1d_fn` + `:559` `mamba2_chunk_scan` | **prefill**（整段 prompt）/ 训练回退 |
| D | 同上但 `cache_params is None` | 同上，只是不存状态（`:558` `output_final_state = cache_params is not None`） | 训练时 `use_cache=False` |

分支 A（`:461-484`）值得单独说：它调 `mamba2_split_conv1d_scan_combined(...)`，`:482` 的注释写着「Only kernels can use this shortcircuit, fallback to normal torch otherwise」，`:483` 判断 `if fused_output is not None: return fused_output`。这个函数在纯 PyTorch 下返回 `None`（`:188` 的 `return None` 就是回退实现），代码继续往下走正常流程。**装了 kernel 训练时只用 1 个 kernel 干完 5 件事**（卷积→扫描→norm→门控→out_proj），没装就按 ①→⑥ 走一遍。它传的 `norm_before_gate=False`（`:477`）不是随手写的：告诉 kernel「**先乘门控、再做 norm**」，和纯 PyTorch 路径 `MambaRMSNormGated.forward` 行为一致（第六节）。

### 4.5 关键问题：scan 到底是谁在算？（实测结论）

**答案：三层优先级，没有 `is_mambapy` 开关。**

三行装饰器（`:166` / `:191` / `:253`），如 `@use_kernel_func_from_hub_with_fallback("mamba_chunk_scan_combined", "mamba_ssm")`（`:253`）；该装饰器定义在 `integrations/hub_kernels.py:822`，docstring（`hub_kernels.py:828-831`）直接写了优先级：**`1. Hf kernels (if requested)` → `2. Original package` → `3. Torch only path`**。第 2 条的完整 import 路径由 `_KERNELS_INTERNAL_PATH_MAPPINGS`（`hub_kernels.py:66-76`，`"mamba_chunk_scan_combined": "ops.triton.ssd_combined"` 等）指定；`hub_kernels.py:836-837` 用它拼 `full_path`，`:842` `importlib.import_module(package)` 导入。

**完整对照表（HF 侧函数名 ≠ kernel 名，务必注意）**：

| HF 侧 torch 回退函数 | 行号 | 第 1 优先级：kernel 名 | 第 2 优先级：原包里的完整路径 |
|---------------------|------|----------------------|---------------------------|
| `mamba2_split_conv1d_scan_combined` | `:167` | `mamba_split_conv1d_scan_combined` | `mamba_ssm.ops.triton.ssd_combined.mamba_split_conv1d_scan_combined` |
| `mamba2_selective_state_update` | `:192` | `selective_state_update` | `mamba_ssm.ops.triton.selective_state_update.selective_state_update` |
| `mamba2_chunk_scan` | `:254` | `mamba_chunk_scan_combined` | `mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined` |
| `causal_conv1d_fn` | `:144` | `causal_conv1d_fn` | `causal_conv1d.causal_conv1d_fn`（装饰器第二参 `"causal_conv1d"`，`:143`） |
| `causal_conv1d_update` | `:124` | `causal_conv1d_update` | `causal_conv1d.causal_conv1d_update`（`:123`） |

第 1 优先级只在**用户显式 `use_kernels=True` 且装了 `kernels` 包**时才生效（`hub_kernels.py:225-237` 把 `mamba_chunk_scan_combined` 注册到 `repo_id="kernels-community/mamba-ssm"`，CUDA 专用）；第 3 优先级就是上表最左列函数**自己的函数体**（`modeling_mamba2.py:254-348` 那段纯 PyTorch）。

**「有没有 mambapy 开关？」——实测：Mamba2 里没有。** `grep -rn "is_mambapy\|use_mambapy" src/transformers/models/mamba2/` **零命中**；那个开关属于 **Mamba1**（`models/mamba/configuration_mamba.py:40` 文档、`:90` 字段，`models/mamba/modeling_mamba.py:196` 用 `is_mambapy_available()`、`:218` 判断 `use_mambapy and pscan is not None`；`is_mambapy_available` 定义在 `utils/import_utils.py:984`）。Mamba2 把「换实现」整个交给 kernel 分发机制，**这就是为什么 `Mamba2Config` 里看不到任何「用哪个实现」的字段。**

> `@use_kernelized_func([...])`（`:351-359`）是**类级**装饰器，把上面 5 个函数整体注册进 kernel 化系统；`@use_kernel_func_from_hub_with_fallback` 是**函数级**的。支持整模型的 `model.kernelize()`（`hub_kernels.py:782`），也能单独替换某个函数；环境变量 `USE_HUB_KERNELS=NO` 可整体关掉（`hub_kernels.py:61-62`）。

### 4.6 `mamba2_chunk_scan` 的纯 PyTorch 实现（`return None` 的那位不读，读这个）

第 1、2 优先级都不成立时跑 `:254-348`。**看懂「它分了三步」就够**：

| 步骤 | 行号 | 干什么 | 类比 |
|------|------|--------|------|
| 0 | `:269-292` | 加 `dt_bias` → `softplus` → `clamp`；乘 `dt` 离散化；切成 `chunk_size=256` 的块 | 把长序列切成一段段 |
| 1 | `:297-308` | **块内**：算 `L = exp(segment_sum(A))`（衰减矩阵）、`Y_diag` | **块内**相当于一个小 causal attention（`:298` 注释原话 `This is the analog of a causal mask`） |
| 2 | `:310-326` | 算每块的**块末状态**，再沿块序列递推传播（`new_states`） | **块间**串行递推，但只有 `n/256` 次 |
| 3 | `:328-335` | 用块状态 + `C` 读出块间贡献 `Y_off`，`output = Y_diag + Y_off` | 块内结果 + 块间结果相加 |

**这就是 dual form 在代码里的样子**：看起来很像注意力（`:301` 算 `G` 的注释写着 `attention-weights like`），但没有 `softmax`、没有 `O(n²)` 全局矩阵——**块内是 `chunk_size × chunk_size` 的小矩阵，块间是 `n/chunk_size` 次递推**。`:326` 的 `states, final_state = new_states[:, :-1], new_states[:, -1]` 把**最后一个状态**交出去，就是要写进 cache 的 `recurrent_state`。

而递推形式 `mamba2_selective_state_update`（`:192-250`）只有几行，**是整篇最好懂的扫描代码**：

```python
# modeling_mamba2.py:227-228  State calculation
ssm_states = state * dA + dBx
state.copy_(ssm_states.to(state.dtype))
# modeling_mamba2.py:240
out = torch.bmm(ssm_states_reshaped, C_reshaped)
# modeling_mamba2.py:244-248   D skip connection 和 z 门控
if D is not None: out = (out + hidden_states * D).to(out.dtype)
if z is not None: out = out * F.silu(z)
```

`state * dA + dBx` —— 就是 `h_t = Ā·h_{t-1} + B̄·x_t` 的逐字翻译：`dA = exp(dt·A)`（`:216`）、`dBx = dt·B·x`（`:222-225`）。**没有别的魔法。**

---

## 五、`Mamba2Block` 对照 `LlamaDecoderLayer`

结论先说：**骨架一模一样，只是「把 attention 换成 mixer、把两条子层压成一条」**。

```python
# modeling_mamba2.py:608  class Mamba2Block(GradientCheckpointingLayer)；:613 self.residual_in_fp32 = config.residual_in_fp32
        self.norm = Mamba2RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)          # :614
        self.mixer = Mamba2Mixer(config, layer_idx=layer_idx, initialize_mixer_weights=False)  # :615
    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))              # :625
        if self.residual_in_fp32: residual = residual.to(torch.float32)                        # :626-627
        hidden_states = self.mixer(hidden_states, cache_params=cache_params, attention_mask=attention_mask)  # :629
        return residual + hidden_states                                                        # :630
```

| 维度 | `LlamaDecoderLayer`（1M1） | `Mamba2Block` |
|------|---------------------------|---------------|
| 基类 | `GradientCheckpointingLayer`（`modeling_llama.py:284`） | 同（`:608`）— **相同** |
| 归一化位置 / 残差写法 | Pre-Norm（`:306` / `:321`）、`residual + f(norm(x))` | Pre-Norm（`:625`）、`residual + mixer(norm(x))`（`:630`）— **相同** |
| 子层数量 | **2 个**：attention + MLP | **1 个**：只有 mixer |
| 归一化模块 / 核心模块 | `LlamaRMSNorm` × 2（`:292-293`）/ `LlamaAttention` | `Mamba2RMSNorm` × 1（`:614`）/ `Mamba2Mixer` |
| 残差精度 | 原精度 | `residual_in_fp32=True` 时升 fp32（`:626-627`）— Mamba2 特有 |
| 传给核心模块的东西 | `attention_mask` / `position_ids` / `past_key_values` / `position_embeddings` | 只有 `cache_params` + `attention_mask`（`:629`）— **没有位置编码** |

**两个值得停一下的差异**：

1. **为什么只有一条子层？** `Mamba2Mixer` 内部自己做了「升维 → 处理 → 门控 → 降维」（`intermediate_size = 2 × hidden_size` 的升维在 `in_proj`、降维在 `out_proj`），相当于**把 attention 和 MLP 揉成一个模块**。所以参数量与同宽度 Llama 不完全可比：Llama 一层有 attention 4 个投影 + MLP 3 个投影，Mamba2 一层只有 `in_proj` / `out_proj` / `conv1d` 三个。
2. **`Mamba2Block` 里没有位置编码。** Llama 要显式算 RoPE 的 `cos/sin` 传进每层；Mamba2 完全没有 position embedding——**位置信息隐含在「递推的顺序」里**，卷积核 `kernel_size=4` 也带一点点局部顺序感。

`Mamba2Model.forward`（`:745-813`）的堆叠方式也和 `LlamaModel` 几乎一样，唯一差别在 `:782`：

```python
# modeling_mamba2.py:782
attention_mask = create_recurrent_attention_mask(config=self.config, inputs_embeds=inputs_embeds,
                                                 attention_mask=attention_mask, past_key_values=cache_params)
```

`create_recurrent_attention_mask`（`masking_utils.py:1474`）的 docstring：`Return the 2D padding mask for mamba / linear-attention layers`。它**只处理 padding**，不生成任何 causal mask（递推天然因果），且在单 token 解码时直接返回 `None`（`:1499-1500`）省开销。

---

## 六、两个 norm：`Mamba2RMSNorm` 与 `MambaRMSNormGated`

**两个都存在，但分工完全不同**（实测）：

| | `Mamba2RMSNorm` | `MambaRMSNormGated` |
|---|---|---|
| 行号 | `:591` | **`:105`** |
| 名字里的 `2` | 有 | **没有**（历史遗留，`falcon_h1` 也复用它，见 `falcon_h1/modular_falcon_h1.py:50`） |
| 用在哪 | `Mamba2Block.norm`（`:614`）、`Mamba2Model.norm_f`（`:728`） | `Mamba2Mixer.norm`（`:417`） |
| 数量/层 | 2 个 | 1 个 |
| 有没有门控 | 无 | **有** |

普通版（`:600-605`）和 `LlamaRMSNorm`（`modeling_llama.py:62`）逐行等价：转 fp32 → 求均方 → `rsqrt` → 乘权重。gated 版（`:105`）与其差异只有一行（`:115-116`）：

```python
        if gate is not None:
            hidden_states = hidden_states * nn.functional.silu(gate.to(torch.float32))
```

**顺序很关键：先乘门控，再算均方、再归一化。** 即 `y = (x · silu(gate)) / rms(x · silu(gate)) * w`（普通 RMSNorm 是 `y = x / rms(x) * w`）。门控**参与统计量**，所以被门控关掉的通道不会拖累其它通道的归一化尺度；反过来被关掉的通道会白占一份分母，数值上更差。这解释了 4.4 节融合 kernel 为什么必须收到 `norm_before_gate=False`（`:477`）——它在告诉 kernel「按纯 PyTorch 路径那样**先 gate 后 norm**」，否则两条路径输出会不同。

**为什么 Mamba2 要加这个门控？** Scan 的输出 `y = C·h` 是一个**线性**读出，表达能力不如 attention 的加权求和；加一条 `silu(gate)` 的乘性支路（与 Llama 的 SwiGLU 同一手法），等于给这个线性读出补上一个数据相关的非线性「音量旋钮」。

---

## 七、cache 怎么办？（呼应 2A3）

### 7.1 SSM 推理要存的东西

Mamba2 没有 K、V，要存**两样**（都在 `Mamba2Mixer.forward` 里能看到，`:490-492`）：

| 存的东西 | 形状 | 大小（默认值，bf16） | 干什么 |
|---------|------|-------------------|--------|
| `conv_states` | `[B, conv_dim, conv_kernel]` = `[B, 10240, 4]` | 80 KiB/层 | 短卷积的左侧上下文（**只有 3 个 token**，常量） |
| `recurrent_states` | `[B, num_heads, head_dim, state_size]` = `[B, 128, 64, 128]` | 2 MiB/层 | **状态 `h`** —— 全部历史压缩成的固定容器 |

写入点（**都被显式标了「in-place copy」**）：卷积状态 `:506` `cache_params.update_conv_state(hidden_states_B_C, self.layer_idx, conv_kernel_size=self.conv_kernel_size)`；递推状态 `:578` `cache_params.update_recurrent_state(final_state, layer_idx=self.layer_idx)`，`final_state` 来自 `:577` `scan_output, final_state = scan_result`（对应 4.6 节 `mamba2_chunk_scan:345-346` 的 `return_final_states`）。

**关键差异（一定要和 2A3 的 `DynamicLayer` 对照看）**：`DynamicLayer.update()` 每步是**往一个越来越长的 list 里 append** K/V；而 `LinearAttentionLayer` 每步是**原地 `copy_`**——形状从一开始就是静态的（`cache_utils.py:1077-1093` 的 `update_recurrent_state`）：

```python
# cache_utils.py:1089  # Note that we copy instead of assigning, to preserve the static address for cudagraphs
self.recurrent_states[state_idx].copy_(recurrent_states)
# cache_utils.py:1024  # Mark as static address to be able to use cudagraphs
if not is_torchdynamo_compiling() and not self.record_past: torch._dynamo.mark_static_address(self.conv_states[state_idx])
```

**「形状静态 + 地址静态」= 可以用 CUDA Graph 录制解码**——这是 SSM 推理的隐形大红利：KV cache 会随长度增长而改形状、每个新 token 都可能触发重编译/重放约束，SSM 没这问题。`LinearAttentionCacheLayerMixin.is_compileable = True`（`cache_utils.py:895`）就是这件事的声明。

### 7.2 `cache_utils.py` 里的 `LinearAttentionLayer` 家族（实测行号）

| 类 | 行号 | 存什么 | 用在哪 |
|----|------|--------|--------|
| `LinearAttentionCacheLayerMixin` | `:891` | 抽象基类；定义 `conv_states` / `recurrent_states` 两个 dict（`:902-903`） | — |
| `LinearAttentionLayer` | `:1003` | **只有 SSM 状态** | Mamba2（`layer_types` 全是 `"linear_attention"`） |
| `LinearAttentionAndFullAttentionLayer` | `:1094` | **SSM 状态 + 完整 KV cache**（同时继承 `LinearAttentionLayer` 和 `DynamicLayer`） | 混合架构 |
| `LinearAttentionAndSlidingWindowAttentionLayer` / `LinearAttentionAndStaticFullAttentionLayer` / `LinearAttentionAndStaticSlidingWindowAttentionLayer` | `:1134` / `:1166` / `:1198` | SSM 状态 + 滑动窗口 KV / 静态 KV / 静态滑动 KV（混合 + SWA / `StaticCache`） | 混合架构 |

类型映射（`cache_utils.py:1223` 的 `DYNAMIC_LAYER_TYPE_MAPPING`）里 `:1231` 是 `"linear_attention": LinearAttentionLayer`。所以 `Mamba2Model.forward:780` 的 `cache_params = DynamicCache(config=self.config)` 会为 64 层各建一个 `LinearAttentionLayer`（走 `cache_utils.py:1791-1793` 的 `get_layer_types_and_kwargs` + `[DYNAMIC_LAYER_TYPE_MAPPING[layer_type](**layer_kwargs) for ...]`，`layer_kwargs` 里的 `number_of_states` 由 `cache_utils.py:1701` 给出）。

两个「窗口回滚」钩子（Mamba2 的 `:496` 用到了）：`activate_past_recording()`（`cache_utils.py:963`，置 `record_past = True`）、`crop(tokens_to_remove)`（`:971`，回滚若干 token，**只对线性层有效**）。用途是投机解码 / 拒绝采样这类「生成了又反悔」的场景；`:496` 的 `not cache_params.layers[self.layer_idx].record_past` 是保护：记录模式下不能用快速单步路径。

### 7.3 混合架构：为什么纯 SSM 不够

`LinearAttentionAndFullAttentionLayer`（`:1094`）**存在本身就是证据**：有人需要「一层里既有 SSM 状态、又有 KV cache」（在 `DYNAMIC_LAYER_TYPE_MAPPING` 里叫 `"hybrid"`，`:1233`）。

**为什么混？** SSM 的核心动作是「把历史压进固定容器」——一次**有损压缩**。它对「整体的、模糊的、语言模型的」信息处理得很好，但**精确召回（recall）天生弱**：要它一字不差复述前文里的某个随机字符串、或做 in-context 的「复制粘贴」，它做不到，因为那需要**无损保留某一个具体位置的信息**。注意力恰好相反：KV cache 是**无损**的。所以现代实践是：**大部分层用 SSM（省内存省算力），隔几层插一个标准 attention 层（补召回）。** 实测证据（都不是猜的）：

| 模型 | 证据 | 混合比例 |
|------|------|---------|
| **Zamba2** | `configuration_zamba2.py:126-137`，`layers_block_type` 里 `"hybrid"` 与 `"linear_attention"` 交错；`:140` 定义 `hybrid_layer_ids` | 默认 54 层中 9 层是 hybrid（`7 + 1 + 1`），其余纯 Mamba2 |
| **Falcon-H1** | `configuration_falcon_h1.py:134` `return ["hybrid" for i in range(self.num_hidden_layers)]`；mixer **直接复用 Mamba2 内核函数**（`modular_falcon_h1.py:49-57` 从 `models.mamba2.modeling_mamba2` import 了 7 个符号） | **所有层都是 hybrid**（每层自带 SSM + attention） |
| **Bamba** | `modular_bamba.py:42-43` `from ..mamba2.modeling_mamba2 import Mamba2Mixer`；`configuration_bamba.py:97-101` 按 `attn_layer_indices` 决定哪几层是 `"full_attention"` | 绝大多数层 linear，少数几层 full attention |
| **Jamba** | `configuration_jamba.py:108-110` `layer_types` = mamba / full_attention 交替；`:81-82` `attn_layer_period=8` / `attn_layer_offset=4` | 每 8 层 1 层 attention（1:7），其余是 Mamba1 + MoE |

`Cache` 类还专门提供了 `is_linear` 属性（`cache_utils.py:1671-1677`），docstring 明确说**同时含两种状态的层不算 linear**（`:1672`）。

> 副作用：`falcon_h1/modeling_falcon_h1.py:993` 写着 `_can_compile_fullgraph = False  # StaticCache has no entry for "hybrid" layers (KV + SSM state).`——混合架构拿不到 Mamba2 那种「全图可编译」的红利。对照 `Mamba2PreTrainedModel`：`_can_compile_fullgraph = True`（`modeling_mamba2.py:640`）。

**另外别忘了**：`Mamba2ForCausalLM.prepare_inputs_for_generation`（`:838-861`）在非首轮时把 `attention_mask` 置 `None`（`:858-859`），因为递推不需要 mask；生成侧的命名约定是 `cache_name = "past_key_values" if not is_linear_attn_cache else "cache_params"`（`generation/utils.py:1944`）——**Mamba 系模型用 `cache_params` 这个名字**，这正是 `Mamba2Model.forward` 参数叫 `cache_params` 的原因；`generation/utils.py:131` 也专门列了 `"cache_params",  # mamba-based models`。

---

## 八、Mamba2 相对 Mamba1 的改进（以真实代码为准）

| 维度 | Mamba1（`models/mamba/`） | Mamba2（本篇） | 代码证据 |
|------|--------------------------|---------------|---------|
| **`A` / `D` 的粒度** | 每通道 × 每状态 `[intermediate_size, state_size]` / `[intermediate_size]` | **每头一个标量** `[num_heads]` / `[num_heads]` | Mamba1 `:327`/`:328` vs Mamba2 `:416`/`:418` |
| **`dt` 的宽度 / 怎么来** | `intermediate_size`（每通道一个）；独立 `dt_proj` 线性层（`:323`） | **`num_heads`**（每头一个）；直接是 `in_proj` 的一片，只有 `dt_bias`（`:412`） | `grep -n "dt_proj" modeling_mamba2.py` → **无命中** |
| **B/C 怎么来 / 是否共享** | 独立 `x_proj`（`:321`）；不共享（每通道一份） | 折进 `in_proj`（`:403`）；**`n_groups=8` 组间共享**，类似 GQA | `:392` / `:527` |
| **卷积范围** | depthwise 卷积只覆盖 `hidden_states`（`intermediate_size` 通道） | **覆盖 `x` + `B` + `C`**：`conv_dim = intermediate + 2·n_groups·state_size` | Mamba1 `:303-310` vs Mamba2 `:392-400` |
| **状态宽度** | `state_size = 16`（默认） | **`state_size = 128`**（默认，大 8 倍） | Mamba1 `configuration_mamba.py:69` vs Mamba2 `:66` |
| **扫描方式** | 朴素序列扫描 / `associative_scan` | **chunk scan（`chunk_size=256`）**：块内并行 + 块间递推 | Mamba1 `configuration_mamba.py:91` vs Mamba2 `:254` |
| **训练融合 kernel** | `mamba_inner_fn`（Mamba1 `:104`） | `mamba_split_conv1d_scan_combined`（Mamba2 `:166`） | 见 4.5 |
| **实现回退开关** | **有**：config 的 `use_mambapy`（`configuration_mamba.py:90`） | **无**，交给 kernel 分发 | 见 4.5 |

**一句话**：Mamba1 里「每通道一套参数」带来的是**参数量和 kernel 启动次数的爆炸**；Mamba2 通过「**降粒度（`A`/`D`/`dt` 每头一个）+ 升状态维度（`state_size` 16→128）+ 分块扫描**」，把同样的状态容量用更少的参数和更适合 GPU 的形状表达出来。论文里的「multi-head / multi-input multi-output」在代码里对应的就是 `num_heads` / `head_dim` / `n_groups` 这组 config 字段和 `:96-103` 的 `validate_architecture` 约束。

---

## Transformer vs SSM 对照速查

| 维度 | Transformer（Llama，1M1） | SSM（Mamba2） | 代码证据 |
|------|--------------------------|---------------|---------|
| **序列混合机制** | 注意力 `softmax(QK^T)V` | 状态递推 `h_t = Āh_{t-1} + B̄x_t` | `modeling_mamba2.py:227-240` |
| **计算复杂度 / 训练并行性** | `O(n²)`；天然全并行 | `O(n)`（chunk scan 额外 `O(n)` 级块间递推）；**靠 dual form 的卷积形态并行** | `:532`（递推）vs `:556`（chunk） |
| **推理时存什么 / 内存** | KV cache（**∝ 序列长度**）；Llama-3-8B @128K ≈ **16 GiB** | `conv_states` + `recurrent_states`（都常量）；Mamba2-2.8B 恒定 ≈ **133 MiB** | `cache_utils.py:902-903`、2.3 节手算 |
| **是否可 CUDA Graph** | 动态 KV → 形状一直在变 | **可以**（静态形状 + `mark_static_address`） | `cache_utils.py:895` / `:1026` |
| **长序列外推** | 需要 RoPE scaling 等技巧 | 状态是常量，天然不随长度退化（但**压缩是固定的**） | — |
| **精确召回（recall）** | **强**（KV 无损，能精确定位） | **弱**（历史被有损压缩进固定容器） | 7.3 |
| **位置信息** | 显式 RoPE | 隐含在递推顺序里，**无位置编码** | 第五节 |
| **生态成熟度** | 极高（所有推理框架一等公民） | 较低（要好用得装 CUDA kernel；混合架构还拿不到全图编译） | `hub_kernels.py:225-237` |
| **归一化 / 子层数** | `LlamaRMSNorm` × 2/层；2 个子层（attention + MLP） | `Mamba2RMSNorm` × 1 + `MambaRMSNormGated` × 1；1 个子层（mixer 内含升维/门控/降维） | 第五 / 六节 |
| **config 里的「层类型」/ 生成侧命名** | `"full_attention"` / `past_key_values` | **`"linear_attention"`** / **`cache_params`** | `configuration_mamba2.py:105`、`generation/utils.py:1944` |

---

## 一句话总结

> **SSM 是「用一个固定大小的状态替代无限增长的 KV cache」的路线**：它用有损压缩换来 `O(n)` 计算和 `O(1)` 推理内存，代价是**召回精度**。让这条路真正可行的是 **dual form**——训练走「卷积形式」（可并行），推理走「递推形式」（省内存），同一组权重两种形态。在 `modeling_mamba2.py` 里这两条路各自对应一个函数：`mamba2_chunk_scan`（`:254`）与 `mamba2_selective_state_update`（`:192`），分支开关就是 `:532` 的 `if use_precomputed_states and seq_len == 1:`。

---

## 验证清单

下面每条都在本机实际跑过（仓库根目录 `/Users/baai/Desktop/baai_proj/transformers`），输出即为实测结果。

```bash
# ① mamba2 没有 modular 文件（本篇直读 modeling 的依据）
ls src/transformers/models/mamba2/ | grep -c modular
#    → 0
wc -l src/transformers/models/mamba2/modeling_mamba2.py src/transformers/models/mamba2/configuration_mamba2.py
#    → 921 / 110

# ② in_proj 的真实切分与 chunk_size
grep -n "projection_size = \|self.conv_dim = \|\.split(" src/transformers/models/mamba2/modeling_mamba2.py
#    → 392: self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
#    → 403: projection_size = self.intermediate_size + self.conv_dim + self.num_heads
#    → 486: gate, hidden_states_B_C, dt = projected_states.split(
#    → 525: hidden_states, B, C = torch.split(
grep -n "chunk_size: int" src/transformers/models/mamba2/configuration_mamba2.py
#    → 87:    chunk_size: int = 256
#    默认值代入：intermediate 8192 / conv_dim 10240 / projection_size 18560
#    切分 = [8192, 10240, 128]；hidden_states_B_C 再切 = [8192, 1024, 1024]
grep -n "num_heads: int\|head_dim: int\|hidden_size: int\|state_size: int\|n_groups: int" \
     src/transformers/models/mamba2/configuration_mamba2.py
#    → 62 / 63 / 65 / 66 / 74 → 128 / 64 / 4096 / 128 / 8

# ③ 没有 dt_proj；time_step_rank 是死字段
grep -n "dt_proj\|time_step_rank" src/transformers/models/mamba2/modeling_mamba2.py
#    → 375:        self.time_step_rank = int(config.time_step_rank)     # 只有这一行，从未被使用
grep -n "dt_proj" src/transformers/models/mamba/modeling_mamba.py
#    → 323/345/356/381/438/439 …… Mamba1 才有 dt_proj

# ④ A 的真实形态与恢复方式
grep -n "A_log\|A = -torch.exp" src/transformers/models/mamba2/modeling_mamba2.py
#    → 416: self.A_log = nn.Parameter(torch.empty(self.num_heads))    # 每头一个标量
#    → 429: A = torch.arange(1, self.num_heads + 1, ...)               # 初始化 log(1..128)
#    → 430: init.copy_(self.A_log, torch.log(A))
#    → 459: A = -torch.exp(self.A_log.float())                        # 保证 A < 0
grep -n "A = A\[:, None, None\].expand" src/transformers/models/mamba2/modeling_mamba2.py
#    → 535: A = A[:, None, None].expand(-1, self.head_dim, self.ssm_state_size)
#    对照 Mamba1: 327: self.A_log = nn.Parameter(torch.empty(self.intermediate_size, self.ssm_state_size))

# ⑤ scan 的真实实现路径 & 有没有 mambapy 开关
grep -n "use_kernel_func_from_hub_with_fallback" src/transformers/models/mamba2/modeling_mamba2.py
#    → 166: @use_kernel_func_from_hub_with_fallback("mamba_split_conv1d_scan_combined", "mamba_ssm")
#    → 191: @use_kernel_func_from_hub_with_fallback("selective_state_update", "mamba_ssm")
#    → 253: @use_kernel_func_from_hub_with_fallback("mamba_chunk_scan_combined", "mamba_ssm")
grep -rn "is_mambapy\|use_mambapy" src/transformers/models/mamba2/
#    → 零命中（Mamba2 没有这个开关）
grep -n "use_mambapy\|is_mambapy_available" src/transformers/models/mamba/configuration_mamba.py \
     src/transformers/models/mamba/modeling_mamba.py src/transformers/utils/import_utils.py
#    → configuration_mamba.py:40 / :90；modeling_mamba.py:196 / :218 / :315 / :470；import_utils.py:984
sed -n '828,831p' src/transformers/integrations/hub_kernels.py
#    → The order of priority is / 1. Hf kernels (if requested) / 2. Original package / 3. Torch only path
sed -n '66,76p' src/transformers/integrations/hub_kernels.py
#    → "mamba_split_conv1d_scan_combined" / "selective_state_update" / "mamba_chunk_scan_combined"
#      / "mamba_inner_fn" / "selective_scan_fn" 共 5 条 mamba 相关映射

# ⑥ 两个 norm 都在，且 gated 版没带 "2"
grep -n "^class Mamba2RMSNorm\|^class MambaRMSNormGated" src/transformers/models/mamba2/modeling_mamba2.py
#    → 105:class MambaRMSNormGated(torch.nn.Module):
#    → 591:class Mamba2RMSNorm(nn.Module):
grep -n "self.norm = MambaRMSNormGated\|MambaRMSNorm(config.hidden_size" src/transformers/models/mamba2/modeling_mamba2.py
#    → 417 (mixer 内的 gated norm) / 614 (block 的普通 norm) / 728 (norm_f)

# ⑦ cache：存的是什么
grep -n "self.conv_states: dict\|self.recurrent_states: dict\|^class LinearAttentionLayer\|^class LinearAttentionAndFullAttentionLayer" \
     src/transformers/cache_utils.py
#    → 902 / 903（定义两个 dict）；1003（LinearAttentionLayer）；1094（LinearAttentionAndFullAttentionLayer）
grep -n "has_previous_state(\|update_conv_state(\|update_recurrent_state(" \
     src/transformers/models/mamba2/modeling_mamba2.py
#    → 453 / 491-492 / 496 / 506 / 578
grep -n "def create_recurrent_attention_mask" src/transformers/masking_utils.py
#    → 1474
grep -n "def layer_types" -A 2 src/transformers/models/mamba2/configuration_mamba2.py
#    → 105-107: 返回 ["linear_attention"] * num_hidden_layers

# ⑧ 混合架构的证据
sed -n '126,140p' src/transformers/models/zamba2/configuration_zamba2.py
#    → "hybrid" / "linear_attention" 交错，默认 54 层里 9 层 hybrid
sed -n '133,134p' src/transformers/models/falcon_h1/configuration_falcon_h1.py
#    → return ["hybrid" for i in range(self.num_hidden_layers)]
grep -n "from ..mamba2.modeling_mamba2 import" -A 6 src/transformers/models/falcon_h1/modular_falcon_h1.py
#    → MambaRMSNormGated / apply_mask_to_padding_states / causal_conv1d_fn / causal_conv1d_update /
#      mamba2_chunk_scan / mamba2_selective_state_update / mamba2_split_conv1d_scan_combined
sed -n '108,110p' src/transformers/models/jamba/configuration_jamba.py
#    → layer_types = ["linear_attention" if x == "mamba" else "full_attention" ...]

# ⑨ Llama 对照面的行号
grep -n "class LlamaRMSNorm\|class LlamaDecoderLayer\|class LlamaAttention" src/transformers/models/llama/modeling_llama.py
#    → 53 / 217 / 284
```

**没有跑的命令**（按任务要求）：`make fix-repo`、`python utils/modular_model_converter.py`、任何模型推理。

---

## 导航

- 路线图：[00_roadmap.md](./00_roadmap.md)（本篇 = 进阶 · 模型结构 · 2M5）
- 上一篇：[2M4_t5_encoder_decoder.md](./2M4_t5_encoder_decoder.md)
- 下一篇：[2M6_llava_next_multimodal.md](./2M6_llava_next_multimodal.md)
- 相关：[1M1_llama_structure.md](./1M1_llama_structure.md)（对照面）、[2A5_modular_code_generation.md](./2A5_modular_code_generation.md)（为什么直读）、[2A3_kv_cache.md](./2A3_kv_cache.md)（`LinearAttentionLayer` 的位置）
