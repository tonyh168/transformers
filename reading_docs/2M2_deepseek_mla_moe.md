# DeepSeek V2 / V3 精读：MLA 低秩 KV 压缩 与 DeepSeekMoE

> 阅读目标：吃透 DeepSeek 的两个原创模块——**MLA（Multi-head Latent Attention）** 与 **DeepSeekMoE**，
> 并看清 V2 → V3 → V3.2 的演进。
>
> - **MLA 是「KV cache 该怎么省」的现代答案**：Llama 的答案是 GQA（少几个 KV 头），MLA 的答案是
>   「根本别缓存 K/V，缓存压缩后的潜向量」。2M7（GLM-5.3）的注意力就是 MLA。
> - **DeepSeekMoE 是「MoE 该怎么切」的现代答案**：Mixtral 用 8 个大专家、每 token 挑 2 个；
>   DeepSeek 用 256 个小专家 + 若干共享专家，每 token 激活 8+1 个。2M3（Qwen3-MoE）的三方对比也要用到它。
>
> 源文件（**三个都要读，本篇覆盖 V2 → V3 的演进**）：
>
> | 文件 | 行数 | 内容 |
> |------|------|------|
> | `src/transformers/models/deepseek_v2/modular_deepseek_v2.py` | 426 | **MLA 与 DeepSeekMoE 首次出现** |
> | `src/transformers/models/deepseek_v3/modular_deepseek_v3.py` | 250 | V3 的改进（路由、RoPE、MLP） |
> | `src/transformers/models/deepseek_v32/modular_deepseek_v32.py` | 436 | V3.2 新增 Indexer / DSA（**本篇只简介**） |
>
> 行数为实测：`wc -l` → 426 / 250 / 436（V2 的生成文件 659 行、V3 的 738 行、V3.2 的 854 行）。

---

## 速读区（TL;DR）

**一句话结论**：MLA 不缓存 K/V，改缓存**压缩潜向量**（512 维潜向量 + 64 维带 RoPE 的共享 K 段 = **576 元素/token/层**），
解码时用 `kv_b_proj`（`expand_kv`）现场展开；DeepSeekMoE 用「256 个细粒度专家 + 1 个共享专家、每 token 激活 8+1」
取代 Mixtral 的「8 个大专家 top-2」，并靠 `sigmoid` + `e_score_correction_bias` + 分组路由替代辅助损失。

**必须记住的断言（12 条）**

| # | 断言 | 出处 |
|---|------|------|
| 1 | cache 里存的是**单头**潜向量 `[B,1,T,512]` 与 `[B,1,T,64]`，共 **576** 元素/token/层——不是展开后的 40,960 | `modular_deepseek_v2.py:360`（实参来自 `:353-354`） |
| 2 | **V3.2 存的是展开后的多头 K/V** `[B,128,T,192] + [B,128,T,128]` = **40,960** 元素/token/层，是 V2/V3 的 **71 倍**，代码里挂着 TODO | `modular_deepseek_v32.py:311,313` |
| 3 | `DeepseekV2Attention` 的基类是 **`nn.Module`**，不是 `LlamaAttention`（整个注意力模块重写） | `modular_deepseek_v2.py:255` |
| 4 | `expand_kv` 定义在 modular:313（生成文件 `modeling_deepseek_v2.py:357`），且发生在 **cache 读取之后**——每步把全部历史重新展开 | `modular_deepseek_v2.py:313,364` |
| 5 | `head_dim` 被改写成 `qk_rope_head_dim`（64）是**刻意的接口适配**，让父类的 RoPE 频率表只覆盖 rope 那一段 | `modular_deepseek_v2.py:137` |
| 6 | `num_key_value_heads` 自动等于 `num_attention_heads`（来自 `LlamaConfig.__post_init__`）→ `repeat_kv` 是空操作，MLA 不需要 GQA 广播 | `modular_deepseek_v2.py:134-138` → `configuration_llama.py:89-90` |
| 7 | 缩放因子是 `qk_head_dim^-0.5`（192）而非 Llama 的 `head_dim^-0.5`，另叠一层 YaRN mscale | `modular_deepseek_v2.py:391,245-252` |
| 8 | 省显存：比 MHA 省 **56.9 倍**（32768/576），**比 GQA-8 还省 3.6 倍**（2048/576）；质量按论文≈MHA | `configuration_deepseek_v3.py:83-104`（第 2.5 节算式） |
| 9 | V3 打分从 `softmax` 换成 **`sigmoid`**，配 `routed_scaling_factor=2.5`；mask 也从 `0.0` 改成 `float("-inf")` | `modular_deepseek_v3.py:98,113,119` |
| 10 | **V2/V3 没有 aux loss**（`grep aux_loss` 0 处匹配，config 里连字段都没有），均衡靠 `e_score_correction_bias`（fp32 buffer） | 第 4.7 节、验证清单 6 |
| 11 | `e_score_correction_bias` **只影响选谁、不影响权重**：选专家用带 bias 的分数，加权用不带 bias 的 | `modular_deepseek_v3.py:99` 与 `:115` |
| 12 | 共享专家**不是一个组，是一个加宽的 MLP**：`moe_intermediate_size * n_shared_experts`，与路由专家输出直接相加、不过门控 | `modular_deepseek_v2.py:199,208` |

**最容易踩的坑 / 反直觉发现**

- **「cache 里存潜向量」这一刀切在 `:360`**，不是 `expand_kv` 之后；MLA 是**用算力换显存**。本仓库**没有**做 weight absorption（把 `kv_b_proj` 吸收进 `q_nope` / `o_proj`），那是 vLLM / SGLang 的优化。
- **V3.2 里「MLA 省显存」没有生效**：稀疏选择必须在**展开后的** K 上做，所以 cache 反而涨到 40,960（71 倍）；V3.2 省的是**注意力算力**（只算 indexer 选出的 2048 个 key），不是显存。这是读代码读出来的事实，不是文档里的说法。
- **`head_dim=64` 看着像 bug，其实是刻意的**——`__post_init__` 的注释写明「RoPE applies only to the rope slice」。
- **V2 删掉的东西 V3 又装回来了**：`del self.norm_topk_prob`（`modular_deepseek_v2.py:164`）→ `self.norm_topk_prob = config.norm_topk_prob`（`modular_deepseek_v3.py:92`，默认 `True`）。
- **V3 的 config 是手写的**（`grep -c 'automatically generated'` → 0），`modular_deepseek_v3.py` 里没有 Config 类，要直接读 `configuration_deepseek_v3.py`。
- **`DeepseekV32DecoderLayer` 那两行 `pass` 不是空类**：它是 V32 → Glm4MoeLite → Glm4Moe → V3 → Llama → `GradientCheckpointingLayer` 共 5 层继承在生成文件里压平的结果。
- **`GlmMoeDsaAttention` 继承的是 `DeepseekV3Attention`，不是 `DeepseekV32Attention`**——不是所有复用都在类继承层面，有些是源码级重复。
- MLA 的「魔法」全在投影里：`expand_kv` 之后就是**普通 MHA**，eager / sdpa / flash 都能吃。

**读完应该能回答的问题**

- cache 里到底存哪两个张量、形状为什么是「1 个头」？这样写能复用什么现成机制？
- 为什么 RoPE 只能作用在 `qk_rope_head_dim` 那一小段上（decoupled RoPE）？
- V3 相对 V2 的路由器改了哪几处？换成 `sigmoid` 后 mask 为什么必须用 `-inf`？
- V3.2 的 cache 为什么反而变大？它省的到底是什么？
- `grep aux_loss` 在 deepseek_v2 / deepseek_v3 里 0 处匹配，为什么不能说「DeepSeek 不做负载均衡」？

---

## 前置知识

- [1M1_llama_structure.md](./1M1_llama_structure.md)：Llama 骨架。没说的部分（RMSNorm、SwiGLU、Pre-Norm 残差、`lm_head`）全部与 Llama 一致。
- [2A5_modular_code_generation.md](./2A5_modular_code_generation.md)：**必读**。`del self.attr`、`nn.Module.__init__(self)`、继承展开成 `GradientCheckpointingLayer` 全靠它解释。
- [1M2_mistral_sliding_window.md](./1M2_mistral_sliding_window.md)：读 modular 只读差异的肌肉。
- [1M3_mixtral_moe.md](./1M3_mixtral_moe.md)：MoE 入门；第 4 章自带 Mixtral 对照表。
- [00_roadmap.md](./00_roadmap.md)：本篇 = 进阶 · 模型结构 · 2M2。

> **MLA 不是 GQA 的参数变体，是换了一套缓存对象**。带着「`num_key_value_heads` 少一点」的思维读会卡在 `expand_kv` 上。

---

## 一、三个源文件与继承全景

### 1.1 继承关系（实测）

```bash
$ grep -n "^class " src/transformers/models/deepseek_v2/modular_deepseek_v2.py
49:DeepseekV2Config(LlamaConfig)   255:DeepseekV2Attention(nn.Module)   # ← 不是 LlamaAttention
157:DeepseekV2Experts(Qwen2MoeExperts)  # pass   161:DeepseekV2TopkRouter(Qwen2MoeTopKRouter)
192:DeepseekV2Moe(nn.Module)   212:DeepseekV2MLP(LlamaMLP)   219:DeepseekV2RMSNorm(LlamaRMSNorm)  # pass
223:DeepseekV2RotaryEmbedding(LlamaRotaryEmbedding)   386:DeepseekV2DecoderLayer(LlamaDecoderLayer)
397:DeepseekV2PreTrainedModel(LlamaPreTrainedModel)  408:DeepseekV2Model(LlamaModel)  # pass
412:DeepseekV2ForCausalLM(LlamaForCausalLM)  416:DeepseekV2ForSequenceClassification(...)  # pass
```

`DeepseekV2Attention` 的基类是 **`nn.Module`**（`:255`）——整个注意力模块重写，差异大到没法继承。
其余大多是 `pass`（`:408/:412/:416/:219/:157`），按 2A5 规则 3 = 「我没意见，别读了」；只覆盖 `__init__` 或 `forward` 的
（`DeepseekV2MLP`、`RotaryEmbedding`、`Moe`、`TopkRouter`、`DecoderLayer`）**才是 V2 真正的新代码**。

### 1.2 V3.2 的 DecoderLayer：一条 5 层深的继承链（2A5 的绝佳案例）

`.../modular_deepseek_v32.py` → `357:class DeepseekV32DecoderLayer(Glm4MoeLiteDecoderLayer):` / `358:    pass`。
顺着往上爬（每一层的基类都实测过）：

```
DeepseekV32DecoderLayer  modular_deepseek_v32.py:357 (pass)
  └─ Glm4MoeLiteDecoderLayer  glm4_moe_lite/modular_glm4_moe_lite.py:162
       └─ Glm4MoeDecoderLayer  glm4_moe/modular_glm4_moe.py:194 (pass)
            └─ DeepseekV3DecoderLayer  modular_deepseek_v3.py:197
                 └─ LlamaDecoderLayer  llama/modeling_llama.py:284
                      └─ GradientCheckpointingLayer  modeling_layers.py:52 → nn.Module
```

**5 层手写继承，生成文件里压平到 1 层**（`.../modeling_deepseek_v32.py` → `627:class DeepseekV32DecoderLayer(GradientCheckpointingLayer):`）。
中间 4 层逐层搬进来再逐层覆盖，落地的 `Glm4MoeLiteDecoderLayer`（`modeling_glm4_moe_lite.py:512`）里
`self.mlp` 按 `config.mlp_layer_types[layer_idx] == "sparse"` 选 `Glm4MoeLiteMoE` 或 `Glm4MoeLiteMLP`，`forward` 与 Llama 逐行相同。
那两行 `pass` 不是「空类」，而是上游 5 个模型 DecoderLayer 逐层展开的最终版本——这也解释了生成文件 854 行 vs modular 436 行。

### 1.3 V3 的 config 是**手写**的（一个真实的例外）

```bash
$ for f in deepseek_v2/configuration_deepseek_v2.py deepseek_v3/configuration_deepseek_v3.py \
           deepseek_v32/configuration_deepseek_v32.py; do
    echo "$f: $(grep -c 'automatically generated' src/transformers/models/$f)"; done
deepseek_v2/...: 1     deepseek_v3/...: 0     # ← 只有 V3 手写、没有生成警告
deepseek_v32/...: 1
```

| | V2 | V3 |
|---|---|---|
| config 类定义在 | `modular_deepseek_v2.py:49`（`class DeepseekV2Config(LlamaConfig)`） | `configuration_deepseek_v3.py:27`（`class DeepseekV3Config(PreTrainedConfig)`，手写） |
| 复用了 V2 的 config 吗 | — | **没有**，字段整份重抄 |

`modular_deepseek_v3.py` 里没有 Config 类，`:30` 直接 `from .configuration_deepseek_v3 import DeepseekV3Config`。
**读法**：V3 的 config 得直接读 `configuration_deepseek_v3.py`（只有 128 行）；V2 的在 `modular_deepseek_v2.py:99-138`，生成文件不要改。

---

## 二、MLA：把 KV cache 换成压缩潜向量

### 2.1 KV cache 是长上下文的显存瓶颈

```
cache_bytes = 2 · B · T · n_layers · n_kv_heads · head_dim · dtype_size   ← 只与 n_kv_heads · head_dim 成正比
```

| 方案 | 缓存的张量 | 每 token 每层的元素数 |
|------|-----------|----------------------|
| MHA | K、V：各 `n_heads · head_dim` | `2 · n_heads · head_dim` |
| GQA | K、V：各 `n_kv_heads · head_dim`（`n_kv_heads < n_heads`） | `2 · n_kv_heads · head_dim` |
| MQA | K、V：各 `1 · head_dim` | `2 · head_dim` |
| **MLA** | **一份压缩潜向量 + 一小段带位置信息的 K** | **`kv_lora_rank + qk_rope_head_dim`** |

注意 MLA 那一行**没有 `2 ·`**：它缓存的**既不是 K 也不是 V**，而是一个共享的、单头的低秩向量；
K 和 V 是**解码时从它现场展开出来的**（`expand_kv`）。V3 真实值：`512 + 64 = 576` 元素/token/层。

### 2.2 MLA 的两处低秩压缩

**(1) Q 侧也低秩**（`q_lora_rank`）：`hidden 7168 ──q_a_proj──▶ 1536 ──q_a_layernorm──▶ 1536 ──q_b_proj──▶ 128·192`

**(2) KV 侧拆两段**（`kv_lora_rank` + `qk_rope_head_dim`）：

```
hidden 7168 ──kv_a_proj_with_mqa──▶ [576] ──split──▶ 潜向量 512  ──kv_a_layernorm──▶ ★缓存这一段★
                                                     k_rot   64  ──RoPE───────▶ ★缓存这一段★
```

### 2.3 维度参数逐个解释（默认值全部实测，见验证清单 4）

| 参数 | V2 默认 | V3 默认 | 含义 |
|------|--------|--------|------|
| `kv_lora_rank` | 512 | 512 | **KV 压缩维度**，cache 的第一个分量 |
| `q_lora_rank` | 1536 | 1536 | **Q 压缩维度**。为 `None` 时不做 Q 低秩（走 `q_proj` 直连）；两条路径都完整实现（`modular:276-291` / `:343-346`） |
| `qk_nope_head_dim` | 128 | 128 | 每个头里**不带位置信息**的那部分 Q/K 宽度 |
| `qk_rope_head_dim` | 64 | 64 | 每个头里**带 RoPE** 的那部分宽度，cache 的第二个分量 |
| `v_head_dim` | 128 | 128 | 每个头的 V 宽度（可以 ≠ `qk_nope_head_dim`） |
| `qk_head_dim` | 192 | 192 | **派生量**：`qk_nope_head_dim + qk_rope_head_dim`（`__post_init__`，modular:136） |
| `head_dim` | 64 | 64 | **被改写成 `qk_rope_head_dim`**（modular:137）——只给 RoPE 频率表用 |
| `num_attention_heads` | 32 | 128 | Q 头数；MLA 里 **K/V 头数 = Q 头数**（没有 GQA 那套 `repeat_kv`） |

两个容易看错的地方：

- **`head_dim` 被故意设置成 64**。`__post_init__` 的注释（同款在 `modular_deepseek_v32.py:150`）：
  > RoPE applies only to the rope slice, so point `head_dim` at it: the inherited (Llama) rotary
  > embedding reads `config.head_dim` and then computes the right frequencies with no override needed.

  即父类的 `LlamaRotaryEmbedding` 用 `config.head_dim` 算频率表，指成 64 就等于「频率表只覆盖 RoPE 那一段」。**不是 bug，是刻意的接口适配**。
- **`num_key_value_heads` 会自动等于 `num_attention_heads`**，这一行**不是 DeepSeek 写的**——modular 的 `__post_init__` 只有 4 行（`modular_deepseek_v2.py:134-138`：`self.qk_head_dim = qk_nope + qk_rope`、`self.head_dim = self.qk_rope_head_dim`），生成出来是 8 行（`configuration_deepseek_v2.py:122`），多出的 4 行来自 `LlamaConfig.__post_init__`（`configuration_llama.py:89-90`，见验证清单 4）——这是 2A5 的「继承展开」在 config 层的样子。结果是 `num_key_value_groups = 32 // 32 = 1`，`repeat_kv` 变成空操作——**MLA 不需要 GQA 那套广播**。

### 2.4 为什么 RoPE 只能作用在一小段上（decoupled RoPE）

> **要对潜向量做 RoPE，就必须先把潜向量展开成 K——那就等于每层都要缓存展开后的 K，压缩白费了。**

`[T, 512]` 的潜向量 `M` 左乘 `W_k`（`512 × 128·192`）得 `[T, 128·192]`，再乘旋转矩阵 `R`；
把 RoPE 结合进投影矩阵要算 `M · (W_k R)`——而 `R` 是**按位置不同**的，没法预乘。所以：

> **每个头切成两半：`nope` 段（128 维，不带位置）走潜向量展开；`rope` 段（64 维，带位置）单独算，不参与压缩、直接缓存。**

这就是 **decoupled RoPE**，在代码里是 `qk_nope_head_dim` / `qk_rope_head_dim` 两个参数，
以及 `kv_a_proj_with_mqa` 的输出被 `split` 成两段（`modular_deepseek_v2.py:351`）。V2 / V3 的 RoPE 实现不同（见 5.2）：

| | V2 | V3 |
|---|---|---|
| 实现 | `apply_rotary_emb`（modular:141），`torch.view_as_complex` 复数乘法 | `apply_rotary_pos_emb`（Llama 的）/ `apply_rotary_pos_emb_interleave`（modular:48） |
| 旋转对象 | **一个复数张量** `freqs_cis`（`DeepseekV2RotaryEmbedding.forward`，modular:236） | `(cos, sin)` 二元组，与 Llama 一致 |
| 维度排布 | 复数成对 `(x0,x1)` | V3 默认 `rope_interleave=True`，交错成对 |

### 2.5 MLA vs GQA：用真实 config 值算一个具体数字

用 **V3 默认值**（实测自 `configuration_deepseek_v3.py:83-104`），batch=1、bf16、128K 上下文（131072 token）：

| 方案 | 每 token 每层元素数 | 计算式 | 61 层 × 128K 的 cache |
|------|---------------------|--------|----------------------|
| MHA（128 头，head_dim 128） | 32,768 | `2·128·128` | **488 GiB** |
| **GQA-8**（8 个 KV 头，head_dim 128） | 2,048 | `2·8·128` | **30.5 GiB** |
| MQA（1 个 KV 头） | 256 | `2·1·128` | **3.8 GiB** |
| **MLA** | **576** | `512 + 64` | **8.6 GiB** |

```
MHA   ████████████████████████████████████████████████████████████████ 32768
GQA-8 ████ 2048      MQA █ 256      MLA █▏ 576   ↑ 比 GQA-8 还小 3.6 倍，质量按论文≈MHA
```

1. **相比 MHA 省 56.9 倍**（`32768 / 576`）。若按「MLA 自己的头宽」算（128×(192+128)=40960），是 **71.1 倍**。
2. **比 GQA-8 还省 3.6 倍**（`2048 / 576`）。这才是 MLA 的真正卖点：GQA 是「用质量换显存」，MLA 是**两个都不换**。
3. MQA 的 cache 更小（256），但质量损失明显。**MLA 是「显存-质量」曲线上更优的点，不是极端的省显存方案。**

> V2 默认值规模对照（32 层、32 头、`kv_lora_rank=512`）：MLA 仍 576/token/层，MHA `2·32·128=8192`，省 **14.2 倍**；32K 上下文时 MLA ≈1.13 GiB、MHA ≈16.0 GiB。
>
> **诚实提醒**：`DeepseekV2Config` 的类默认值（`hidden_size=4096`、32 层、`vocab_size=102400`）
> **不等于任何真实 checkpoint**；要 V2 的真实数字请以 `deepseek-ai/DeepSeek-V2-Lite` 的 `config.json` 为准。
> 上面表格用的 **V3 值对得上真实 V3**——4.2 节用它们算出的 671B / 37B 是已知公开数字，可自校验。

---

## 三、逐行精读 `DeepseekV2Attention`

完整代码在 `modular_deepseek_v2.py:255-383`。

### 3.1 `__init__`：五组投影（modular:255-311）

```python
# modular_deepseek_v2.py:255   —— Q 侧（:276-291）：q_lora_rank 决定走哪条路
        self.q_proj = nn.Linear(hidden, num_heads * qk_head_dim, bias=False) if self.q_lora_rank is None else None
        # 否则 q_a_proj → q_a_layernorm → q_b_proj（低秩路径）；self.qk_head_dim = qk_nope + qk_rope = 192
        # KV 侧（:293-303）：一次投出「潜向量 + 带 RoPE 的 K 段」
        self.kv_a_proj_with_mqa = nn.Linear(self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim, bias=config.attention_bias)   # 512 + 64 = 576
        self.kv_a_layernorm = DeepseekV2RMSNorm(config.kv_lora_rank)
        self.kv_b_proj = nn.Linear(config.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)   # 128 · 256 = 32768
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=config.attention_bias)
        self.scaling = yarn_apply_mscale(config.rope_parameters, self.qk_head_dim ** (-0.5))
```

`kv_b_proj` 是**唯一的「上投影」矩阵**：输入 512 维潜向量，输出 128 头 × 256 维（128 维 K + 128 维 V 拼在一起）；它**不缓存**，只活在 `expand_kv` 里。
最后一行：缩放用的是 `qk_head_dim^-0.5 = 192^-0.5`，而 Llama 用 `head_dim^-0.5`——**MLA 的 K 宽变了（192），缩放就得跟着变**。`yarn_apply_mscale`（modular:245-252）在其上叠一层 YaRN 的 mscale 修正。

### 3.2 `expand_kv`：从潜向量「展开」出 K 和 V（modular:313-330）

**MLA 最核心的函数，也是唯一需要「反压缩」的地方**（生成文件 `modeling_deepseek_v2.py:357`）。

```python
# modular_deepseek_v2.py:313
    def expand_kv(self, kv_nope, k_rot) -> tuple[torch.Tensor, torch.Tensor]:
        """kv_nope: [batch_size, 1, seqlen, kv_lora_rank]（无位置的 K+V）；k_rot: [.., 1, seqlen, qk_rope_head_dim]（带位置的共享 K）
        Returns key/value states of shape [batch, num_heads, seq, (k or v)_head_dim]."""
        kv_nope = self.kv_b_proj(kv_nope).view(batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)  # :322
        k_nope, value_states = torch.split(kv_nope, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)  # :323
        k_rot = k_rot.expand(-1, k_nope.shape[1], -1, -1)  # does not affect the underlying storage
        key_states = kv_nope.new_empty(*kv_nope.shape[:-1], self.qk_nope_head_dim + self.qk_rope_head_dim)
        key_states[..., : self.qk_nope_head_dim].copy_(k_nope)
        key_states[..., self.qk_nope_head_dim :].copy_(k_rot)      # :327-329
        return key_states, value_states
```

逐行（以 V3 的 128 头、512/64/128/128 为例）：

| 行 | 操作 | 形状变化 |
|----|------|---------|
| `:319-320` | 读形状、准备 `key_shape` | 输入 `kv_nope` = `[B, 1, T, 512]`（**单头**） |
| `:322` | `kv_b_proj` + `view` + `transpose(1,2)` | `[B,1,T,512]` → `[B,T,128,256]` → `[B,128,T,256]`（← `-1` 展开成 128 个头） |
| `:323` | `split` | `k_nope` = `[B,128,T,128]`，`value_states` = `[B,128,T,128]` |
| `:324` | `k_rot.expand` | `[B,1,T,64]` → `[B,128,T,64]`（**只是广播视图，不复制显存**） |
| `:327-329` | `new_empty` + 两次 `copy_` | 拼成 `key_states` = `[B,128,T,192]` |

三个设计点：**`kv_b_proj` 一次投出 K 和 V**（`256 = 128(K) + 128(V)`，`split` 一刀切开——这就是 `qk_nope_head_dim` 与 `v_head_dim` 是两个独立参数的原因）；
**`k_rot` 是单头的**，`expand` 只是视图，位置信息对 128 个头共享；**用 `new_empty` + `copy_` 而不是 `torch.cat`**，省掉一次中间分配。

### 3.3 `forward`：完整数据流（modular:332-383）

```python
# modular_deepseek_v2.py:340
        if self.q_lora_rank is None:  q = self.q_proj(hidden_states)
        else:  q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))    # :346
        q_nope, q_pe = torch.split(q.view(B, S, -1, self.qk_head_dim).transpose(1, 2),
                                   [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)   # :348
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)                        # :350
        kv_nope, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)  # :351
        # Both latents are viewed as single-head, 4D tensors so all cache layers handle them correctly
        k_nope = self.kv_a_layernorm(kv_nope).view(batch_size, 1, seq_length, self.kv_lora_rank)   # :353
        k_pe = k_pe.view(batch_size, 1, seq_length, self.qk_rope_head_dim)            # :354
        q_pe, k_pe = apply_rotary_emb(q_pe, k_pe, position_embeddings.to(q_pe.device))  # :356
        # Cache read / write is performed while the latent KV is still compressed      # :358
        if past_key_values is not None:
            k_nope, k_pe = past_key_values.update(k_nope, k_pe, self.layer_idx)       # :360
        query_states = torch.cat((q_nope, q_pe), dim=-1)                              # :362
        key_states, value_states = self.expand_kv(k_nope, k_pe)                       # :364
```

- **`:348` Q 被切成 `q_nope` / `q_pe`**，`:362` 又拼回来——因为 RoPE（`:356`）**只能作用在 `q_pe` 上**。切—转—拼是 decoupled RoPE 的必然形态。
- **`:353-354` 把两个潜向量都 reshape 成 `[B, 1, S, ·]`**（注释：`Both latents are viewed as single-head, 4D tensors so all cache layers handle them correctly`）：cache 层按 `[B, num_heads, S, dim]` 4 维存张量，写成「1 个头」就能复用同一套 cache 代码，不用为 MLA 定制 cache。**这是「不侵入 cache_utils」的工程取舍**。
- **`:356` `position_embeddings` 是个张量，不是元组**：V2 的 `DeepseekV2RotaryEmbedding.forward` 返回复数 `freqs_cis`（modular:236），配合 `:141` 的 `torch.view_as_complex` 做复数旋转（V3 改回 cos/sin，见 5.2）。
- **`:366-379` 调用标准注意力接口**：`expand_kv` 之后就是**普通的 MHA**（`[B, 128, T, ·]` 普通张量，eager / sdpa / flash 都能吃）——**MLA 的「魔法」全在投影里**。`:381-382` 是 `reshape(B, S, -1)` 后过 `o_proj`。

### 3.4 cache 里到底存的是哪个张量？（本篇最关键的一问）

答案是 `modular_deepseek_v2.py:360`：`k_nope, k_pe = past_key_values.update(k_nope, k_pe, self.layer_idx)`，
两个实参就是 `:353-354` 算出来的：

| 实参 | 形状 | 元素数 | 含义 |
|------|------|--------|------|
| `k_nope` | `[B, 1, S, kv_lora_rank=512]` | 512 | **压缩后的 KV 潜向量**（过完 `kv_a_layernorm`） |
| `k_pe` | `[B, 1, S, qk_rope_head_dim=64]` | 64 | **带 RoPE 共享 K 段** |

**注意这两个张量的形状是「1 个头」**，不是 128 个头。所以：

```
★ cache 里存的是 [B, 1, T, 512] 与 [B, 1, T, 64]，共 576 个元素/token/层 ★
   而不是 expand_kv 之后的 [B,128,T,192] 和 [B,128,T,128]（= 40960 个元素）
   decode 第 t 步：cache ──▶ [B,1,t,512] + [B,1,t,64] ──kv_b_proj──▶ [B,128,t,256] ──▶ attention
                            └── 存的是这个（576/token）      └── 每步重算，不存
```

**这就是 MLA 省显存的全部秘密**：`expand_kv`（`:364`）发生在 cache **读取之后**——每步都把**全部历史**重新展开一次。**用算力换显存**。

> **一个延伸（不在本仓库里）**：既然「每步重算 `kv_b_proj`」在长上下文下很贵，就有另一种算法：把 `kv_b_proj` 的 K 半部分**乘进 `q_nope` 的投影**、V 半部分**乘进 `o_proj`**（weight absorption），这样**根本不需要展开**，直接在 512 维潜空间里算 `[S,512] × [T,512]`。**本仓库没有做这个优化**——`:364` 每次解码都老老实实 `expand_kv`。真实推理引擎（vLLM / SGLang）里能看到吸收版与非吸收版两条路径的取舍。读 2A3（KV cache 体系）时会再次遇到。

### 3.5 MLA 的完整数据流图（重点：cache 的那一刀切在哪）

```
   hidden_states [B,S,hidden_size]
     ├─ Q 侧：q_a_proj → q_a_layernorm → q_b_proj → split(:348) → q_nope[128] + q_rot[64]
     └─ KV 侧：kv_a_proj_with_mqa → split(:351) ─┬─ kv_nope[512] → kv_a_layernorm(:353) → view [B,1,S,512]
                                                 └─ k_rot[64]
                    ╔══════════════════════════════════════════════════════╗
   RoPE(:356) ──────║  past_key_values.update(:360)  ★ cache 从这一刀切 ★   ║
   只转 q_rot/k_rot  ║  只存 [B,1,T,512] 和 [B,1,T,64] = 576 元素/层（单头！） ║
                    ╚══════════════════════════════════════════════════════╝
                          │  expand_kv(:364) ← 「反压缩」在这里：kv_b_proj 512 → heads·(128+128)
                          │    split → k_nope[heads,128] / v[heads,128]；k_rot.expand → [heads,64]
                          │    cat(k_nope, k_rot) → K [B,heads,T,192] / V [B,heads,T,128]
                          ▼
   Q [B,heads,S,192] ──▶ attention(eager/sdpa/flash)（之后就是普通 MHA，无任何特殊）──▶ o_proj
```

---

## 四、DeepSeekMoE：细粒度专家 + 共享专家

MoE 讲的是前馈层的「大而不贵」：**参数量大（知识容量），但每 token 激活的算力小**。

### 4.1 两个模块拼起来的 MoE 层（modular:192-209）

```python
# modular_deepseek_v2.py:192
class DeepseekV2Moe(nn.Module):
    def __init__(self, config):
        self.experts = DeepseekV2Experts(config)
        self.gate = DeepseekV2TopkRouter(config)
        self.shared_experts = DeepseekV2MLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts   # :199
        )
    def forward(self, hidden_states):
        residuals = hidden_states
        _, topk_weights, topk_indices = self.gate(hidden_states)
        hidden_states = self.experts(hidden_states.view(-1, hidden_states.shape[-1]), topk_indices, topk_weights)
        return hidden_states.view(*orig_shape) + self.shared_experts(residuals)   # :208
```

**`self.shared_experts` 是一个普通 MLP，不是「一组专家」**：`:199` 用 `intermediate_size = moe_intermediate_size * n_shared_experts`
把 `n_shared_experts` 个共享专家**拼成一个加宽的 MLP**（V3：`2048 × 1`；V2 默认：`1407 × 2 = 2814`）；数学上等价于「各自算完再相加」，但只占一个算子。
**`:208` 是接入点**：它吃的是 **`residuals`（未 reshape 的原始输入）**，与路由专家输出**直接相加**，不经过任何门控。

### 4.2 细粒度专家（fine-grained experts）

| | Mixtral（1M3） | DeepSeek-V2 默认 | **DeepSeek-V3** |
|---|---|---|---|
| 路由专家数 `n_routed_experts` | 8 | 64 | **256** |
| 每 token 激活 `num_experts_per_tok` | 2 | `None`（由 checkpoint 指定） | **8** |
| 单专家中间维 `moe_intermediate_size` | 14336（= 整层 MLP 宽） | 1407 | **2048** |
| 隐藏维 | 4096 | 4096 | 7168 |
| 单专家参数量 `3·h·m` | `3·4096·14336` ≈ 176M | ≈ 17.3M | ≈ 44.0M |

（Mixtral 实测自 `configuration_mixtral.py:69-87`；DeepSeek 实测自 `modular_deepseek_v2.py:122,132` 与 `configuration_deepseek_v3.py:84,87,90-91,100`。）

**「细粒度」= 把 Mixtral 的「8 个大专家」切成「256 个小专家」**。总参数量可以相同，但细粒度带来两个好处：
**组合更多**（从 256 里选 8 个，`C(256,8)` 远大于 `C(8,2)` 的 28 种）；**知识不重叠度更高**（DeepSeekMoE 论文的核心论点）。
代价是**路由器必须更准**（256 选 8 比 8 选 2 难得多），这就是 4.3 节路由器复杂得多的原因。用 V3 默认值验算，能对上公开的 671B / 37B：

```
单个专家 3 × 7168 × 2048 = 44.04M  →  单 MoE 层 (256 路由 + 1 共享) × 44.04M ≈ 11.32B
MoE 层数 = 61 − first_k_dense_replace(3) = 58  →  专家总量 11.32B × 58 ≈ 656B
+ 注意力(61 层 ≈ 11.4B) + 3 个 dense 层(≈1.2B) + 词嵌入(≈0.9B) ≈ 671B ✔
每 token 激活：8 路由 + 1 共享 = 9 × 44.04M × 58 层 + 上面那部分 ≈ 37B ✔
```

> **`first_k_dense_replace`**：前面若干层的 `self.mlp` 是**普通 dense MLP**（实测 `modular_deepseek_v2.py:391` 与 `modular_deepseek_v3.py:204`）：
> `self.mlp = DeepseekV2Moe(config) if layer_idx >= config.first_k_dense_replace else DeepseekV2MLP(config)`。
> 默认 V2 = `0`（modular:118），V3 = `3`（`configuration_deepseek_v3.py:101`）。浅层学通用特征，MoE 化收益低。

### 4.3 共享专家（shared experts）

**动机**：细粒度切分会带来**知识冗余**——多个路由专家可能各自学同一份通用知识（语法、常见搭配）。
**解法**：把「通用知识」抽出来放进 `n_shared_experts` 个**每个 token 都走、不参与路由**的专家。

| | Mixtral | DeepSeek V2（默认） | DeepSeek V3 |
|---|---|---|---|
| 共享专家 | **没有** | `n_shared_experts = 2` | `n_shared_experts = 1` |
| 每 token 总专家数 | 2 | 2 + 若干路由 | **1 + 8 = 9** |

（V2 的 `2` 实测自 `modular_deepseek_v2.py:123`；V3 的 `1` 自 `configuration_deepseek_v3.py:90`、V3.2 的 `1` 自 `modular_deepseek_v32.py:112`——**V3 把共享专家从 2 减到 1**。）

### 4.4 路由图

```
   hidden_states [B, S, 7168]
      ├──► DeepseekV2/V3 TopkRouter (gate)      ├──► shared_experts（加宽 MLP，无门控，每 token 都走）
      │     scores = softmax(·) [V2] / sigmoid(·)+bias [V3]
      │     └─► group-limited routing（n_group=8 组 → 选 4 组 → 组内 top-k）
      │           └─► 256 个细粒度专家 (3D 张量，只算命中的) → 加权求和
      └─────────────────────── 相加（:208）→ 输出
```

### 4.5 `DeepseekV2TopkRouter`：和 Mixtral 的 router 差在哪（modular:161-189）

```python
# modular_deepseek_v2.py:161
class DeepseekV2TopkRouter(Qwen2MoeTopKRouter):
    def __init__(self, config):
        super().__init__(config)
        del self.norm_topk_prob                       # :164  ← 2A5 的「删掉父类赋值」
        self.routed_scaling_factor = config.routed_scaling_factor
        self.topk_method = config.topk_method; self.num_group = config.n_group; self.topk_group = config.topk_group
    def forward(self, hidden_states):
        router_logits = F.linear(hidden_states.view(-1, self.hidden_dim).type(torch.float32), self.weight.type(torch.float32))  # :172
        scores = router_logits.softmax(dim=-1, dtype=torch.float32)                                   # :173
        if self.topk_method == "greedy":                                                              # :174
            topk_weights, topk_indices = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        elif self.topk_method == "group_limited_greedy": ...                                          # :176
        topk_weights = topk_weights * self.routed_scaling_factor                                       # :188
        return router_logits, topk_weights, topk_indices
```

和 Mixtral 的 `MixtralTopKRouter`（`modeling_mixtral.py:96-110`）对比：

| 特性 | Mixtral | DeepSeek V2 | DeepSeek V3 |
|------|---------|-------------|-------------|
| 打分函数 | `softmax` | `softmax` | **`sigmoid`**（`:98`） |
| 权重归一化 | 强制 `topk_val /= sum` | **`del self.norm_topk_prob`（不做）** | `norm_topk_prob=True` → 做（`:116-118`） |
| 路由缩放 | 无 | `routed_scaling_factor=1.0`（`:188`） | **`2.5`**（`:119`） |
| 专家分组 | 无 | 可选 `group_limited_greedy` | **一定开**（`del self.topk_method`） |
| 偏置修正 | 无 | 无 | **`e_score_correction_bias`**（`:93`） |
| 辅助损失 | `router_aux_loss_coef=0.001` + `load_balancing_loss_func` | **没有**（见 4.7） | **没有**（用 bias 替代） |
| 抖动噪声 | `router_jitter_noise` | 无 | 无 |
| 权重存储 | 3D 张量 `MixtralExperts`（`modeling_mixtral.py:57`） | 3D 张量 `Qwen2MoeExperts`（`modeling_qwen2_moe.py:279`），`DeepseekV2Experts` 直接 `pass` 继承 | 同 V2 |

**(1) 用 `sigmoid` 而不是 `softmax`**（V3，`modular_deepseek_v3.py:98`）：softmax 让专家**互相竞争**（权重和恒为 1），sigmoid 让每个专家**独立打分**；V3 配 `routed_scaling_factor=2.5`（`:119`）把权重放大回来，补偿 sigmoid 值域偏小。

**(2) `routed_scaling_factor` 是「路由权重的全局缩放」**：V2 默认 `1.0`（`modular_deepseek_v2.py:126`），V3 是 `2.5`（`configuration_deepseek_v3.py:92`）。它乘在 `topk_weights` 上（`:188` / V3 的 `:119`）——**只影响 MoE 输出的幅度，不影响选谁**。

**(3) `del self.norm_topk_prob` 是 2A5 里那个技巧的真实用例**：父类 `Qwen2MoeTopKRouter.__init__` 里有 `self.norm_topk_prob = config.norm_topk_prob`（`modeling_qwen2_moe.py:323`），`:164` 的 `del` **精确删掉那一行赋值**；生成文件里可以验收（`sed -n '86,97p' .../modeling_deepseek_v2.py`，见验证清单 7）：`DeepseekV2TopkRouter` 的 `__init__` 里完全没有 `self.norm_topk_prob`。而 **V3 又把它加回来了**（`modular_deepseek_v3.py:92`，默认 `True`）——**「V2 删掉的东西 V3 又装回来」**。

### 4.6 group-limited routing（专家分组路由，`modular_deepseek_v2.py:176-187`）

**解决什么问题**：256 个专家全平铺时，`topk` 只在**数值上**选最大的——结果容易「抱团」，少数专家被反复选中（load imbalance），且**跨设备的通信量不可控**。**怎么做**（把 256 个专家分成 `n_group` 组，V3 是 8 组、每组 32 个）：

```
步骤 1  给每个组算「组分数」= 组内专家的最大分数（V2，:184 用 .max(dim=-1).values）/ 前 2 名之和（V3）
步骤 2  从 n_group 个组里选 topk_group 个组（V3：8 组选 4 组）
步骤 3  把没选中的组里所有专家的分数 mask 掉（V2 `masked_fill(~score_mask, 0.0)`，modular:186）
步骤 4  在被选中的 4 组（共 128 个专家）里做 top-k（V3：top-8）
```

效果：**每个 token 最多只能从 4 个组里挑专家**——触达组数有上界 → 专家并行的 all-to-all 通信量可控；组内被选中的专家数自动均衡。
**V2 里它是可选的**（`topk_method` 默认 `"greedy"`，`modular_deepseek_v2.py:128`；`n_group`/`topk_group` 默认 `None`，`:121,127`）；
**V3 里它是强制的**：`DeepseekV3TopkRouter.__init__` 第一件事就是 `del self.topk_method`（`modular_deepseek_v3.py:90`）——典型的「真实训练后发现必须开」的收编。

### 4.7 关于辅助损失：这里**没有** aux_loss（实测）

```bash
$ grep -rn "aux_loss\|seq_aux_loss\|z_loss\|output_router_logits" \
        src/transformers/models/deepseek_v2/ src/transformers/models/deepseek_v3/
（无输出 —— 0 处匹配）
$ grep -rn "aux_loss" src/transformers/models/qwen2_moe/modeling_qwen2_moe.py | head -3
519:def load_balancing_loss_func(    601:        self.router_aux_loss_coef = config.router_aux_loss_coef
```

| | Mixtral / Qwen2-MoE | DeepSeek V2 / V3 |
|---|---|---|
| 负载均衡的机制 | **辅助损失**（`load_balancing_loss_func` + `router_aux_loss_coef`） | **无辅助损失** |
| 均衡手段 | 在 loss 里加惩罚项 | ① group-limited routing ② V3：`e_score_correction_bias` |
| 推理侧影响 | `DeepseekV2Moe.forward` 把 `router_logits` 用 `_` 丢掉（modular:205） | 同 |

**V3 的替代方案是 `e_score_correction_bias`**（`modular_deepseek_v3.py:93`：
`self.e_score_correction_bias = nn.Buffer(torch.zeros(self.num_experts))`）——一个**训练时动态更新的 buffer**（谁被选少了就把它的偏置调高），推理时只从 checkpoint 读进来：

```python
# modular_deepseek_v3.py:99
        scores_for_choice = scores + self.e_score_correction_bias   # 选专家时带偏置
# modular_deepseek_v3.py:115
        topk_weights = scores.gather(1, topk_indices)               # 算权重时不带偏置
```

**「选谁看偏置，算权重不看偏置」**——这是 auxiliary-loss-free load balancing 的精髓。
`_keep_in_fp32_modules_strict = ["e_score_correction_bias"]`（`modular_deepseek_v3.py:214`）是唯一的额外线索——**它必须保持 fp32**，否则累积的小偏置会被 bf16 抹掉。它初始为 0（`_init_weights`，`modular_deepseek_v3.py:222`），之后由 checkpoint 覆盖。

---

## 五、V3 相对 V2 的改进（增量表）

`modular_deepseek_v3.py` 只有 250 行，**逐条列完就是 V3 的全部改动**。

### 5.1 总览

| # | 改动 | V2 | V3 | 代码位置 |
|---|------|----|----|---------|
| 1 | RoPE 实现 | 复数 `freqs_cis` + `view_as_complex` | Llama 的 `(cos,sin)` + 交错旋转 | `modular_deepseek_v3.py:48-84` |
| 2 | 路由打分 | `softmax` | **`sigmoid`** | `:98` |
| 3 | top-k 权重归一化 | `del self.norm_topk_prob`（不做） | `norm_topk_prob=True` → 做 | `:92,116-118` |
| 4 | 路由缩放 | `1.0` | **`2.5`** | `:119` |
| 5 | 分组路由 | 可选（`topk_method`） | **强制**（`del self.topk_method`） | `:90` |
| 6 | 组分数 | 组内**最大**分数 | 组内**前 2 之和** | `:100-104` |
| 7 | 负载均衡 | 无 | **`e_score_correction_bias`**（fp32 buffer） | `:93,99,115` |
| 8 | MLP 基类 | `LlamaMLP` | `Qwen2MoeMLP` | `:44-45` |
| 9 | `num_experts` 来源 | `config.num_experts`（`attribute_map` 转发） | `config.num_local_experts` | `:91` |
| 10 | DecoderLayer 的 `__init__` | 继承 Llama 的实体 | **`nn.Module.__init__(self)`** 重写 | `:199` |
| 11 | 任务头 | 继承 Llama 的 `ForSequenceClassification` | 换成通用 `Generic*` | `:236-241` |
| 12 | 权重加载 | 无特殊处理 | 忽略第 61 层 + `e_score_correction_bias` 保 fp32 | `:214-215` |

### 5.2 逐条看代码

**(1) RoPE：V3 全面改用 Llama 的 cos/sin + 交错旋转**（modular:48-84）

```python
# modular_deepseek_v3.py:48
def apply_rotary_pos_emb_interleave(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """DeepSeek lays the rotary dimensions out in interleaved pairs `(x0, x1), (x2, x3), ...`, each rotated by a
    single frequency; the output is bit-identical to the de-interleaved `rotate_half` formulation."""
    cos = cos[..., : cos.shape[-1] // 2].unsqueeze(unsqueeze_dim)   # `cos`/`sin` are `cat(freqs, freqs)`
    sin = sin[..., : sin.shape[-1] // 2].unsqueeze(unsqueeze_dim)
    q1, q2 = q[..., 0::2], q[..., 1::2]; k1, k2 = k[..., 0::2], k[..., 1::2]
    q_embed = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    k_embed = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
    return q_embed, k_embed
```

**「interleave vs half-split」是 RoPE 的两个等价实现**：Llama 的 `rotate_half` 前后两半配对 `(x0,x32),(x1,x33),...`；DeepSeek 是相邻配对 `(x0,x1),(x2,x3),...`。**数学等价，只是维度排布不同**——但权重按某种排布训出来，必须一致。V3 默认 `rope_interleave=True`（`configuration_deepseek_v3.py:114`），在 `:164` 分流：`apply_rotary_pos_emb_interleave(...)` 或 `apply_rotary_pos_emb(...)`（后者是 Llama 的）。

**(2)-(7) 路由器的全量重写**（modular:87-120）

```python
# modular_deepseek_v3.py:87
class DeepseekV3TopkRouter(DeepseekV2TopkRouter):
    def __init__(self, config):
        super().__init__(config)
        del self.topk_method                                  # :90  V3 只有一条路径
        self.num_experts = config.num_local_experts           # :91
        self.norm_topk_prob = config.norm_topk_prob           # :92  把 V2 删掉的装回来
        self.e_score_correction_bias = nn.Buffer(torch.zeros(self.num_experts))   # :93
    def forward(self, hidden_states):
        router_logits = F.linear(hidden_states.view(-1, self.hidden_dim).type(torch.float32), self.weight.type(torch.float32))
        scores = router_logits.sigmoid()                      # :98  softmax → sigmoid
        scores_for_choice = scores + self.e_score_correction_bias   # :99
        group_scores = scores_for_choice.view(-1, self.num_group, -1).topk(2, dim=-1)[0].sum(dim=-1)  # :102-103
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))   # :113
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]    # :114
        topk_weights = scores.gather(1, topk_indices)                                          # :115
        if self.norm_topk_prob:  topk_weights /= topk_weights.sum(dim=-1, keepdim=True) + 1e-20  # :117-118
        topk_weights = topk_weights * self.routed_scaling_factor                               # :119
        return router_logits, topk_weights, topk_indices
```

注意 `:113` 用 `float("-inf")` 而不是 V2 的 `0.0`（`modular_deepseek_v2.py:186`）——**因为 sigmoid 的打分恒为正**，mask 成 0 掩盖不住（0 可能比真实分数还大）；必须用 `-inf`。这是「换打分函数」的连带改动，读两份 router 时最容易忽略。

**(8) MLP 换个基类**（modular:44-45）：`class DeepseekV3MLP(Qwen2MoeMLP): pass`。V2 用的是 `DeepseekV2MLP(LlamaMLP)`（`modular_deepseek_v2.py:212`）。差异在 `hidden_size`/`intermediate_size` 是实例属性而非 config 属性（`Qwen2MoeMLP` 支持传入覆盖值，`modeling_qwen2_moe.py:130`），因为 MoE 的共享专家要用「加宽的 intermediate_size」构造同一个类。

**(10) `nn.Module.__init__(self)`：一个 2A5 的「咒语」**（modular:197-210）

```python
# modular_deepseek_v3.py:197
class DeepseekV3DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: DeepseekV3Config, layer_idx: int):
        nn.Module.__init__(self)              # :199  ← 不要 Llama 的 __init__，只要 nn.Module
        self.hidden_size = config.hidden_size
        self.self_attn = DeepseekV3Attention(config=config, layer_idx=layer_idx)
        self.mlp = DeepseekV3MoE(config) if layer_idx >= config.first_k_dense_replace else DeepseekV3MLP(config)
        self.input_layernorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

2A5 的表格里写过：`nn.Module.__init__(self)` = 「别展开 modular 父类，只要 `nn.Module`」。生成文件里它变成 `class DeepseekV3DecoderLayer(GradientCheckpointingLayer):` + `super().__init__()`，**父类体没有被抄进来**。**为什么 V2 用 `super().__init__(config, layer_idx)` 而 V3 用这个咒语？** V2 想**继承 Llama 的 `forward` 和 `__init__` 骨架**，只把 `self.self_attn` / `self.mlp` / 两个 norm 换掉；V3 则干脆**把 `__init__` 整个自己写全**。两种写法的生成结果几乎一样（V2 的 `self.hidden_size` 从 Llama 那层继承，生成的 `modeling_deepseek_v2.py:434` 里同样有这一行），只是 V3 更显式。（`Glm4MoeLiteDecoderLayer` 也用了同一个咒语，`glm4_moe_lite/modular_glm4_moe_lite.py:163`。）

**(11) 任务头换成通用实现**（modular:236-241）：`DeepseekV3ForSequenceClassification(GenericForSequenceClassification, DeepseekV3PreTrainedModel)`、`DeepseekV3ForTokenClassification(GenericForTokenClassification, DeepseekV3PreTrainedModel)`，均 `pass`。V2 只有 `DeepseekV2ForSequenceClassification(LlamaForSequenceClassification)`（modular:416）——**继承 Llama 的具体实现**。V3 换成 `modeling_layers.py` 里的通用实现，V3.2 又多了 `ForTokenClassification`；**不影响模型主体结构**。

### 5.3 V3.2 又叠了什么（**只列，不展开**）

`DeepseekV32Attention` 在**完全保留 V3 MLA 的前提下**加了一条旁路（`modular_deepseek_v32.py:295-313`）：

```python
        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))   # :295  额外留一份 Q 的中间结果
        ...
        # Sparse-attention models cache the expanded K/V, not the compressed latents. TODO (remi-or): fix this with topk
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)   # :313
```

| | V2 / V3 | **V3.2** |
|---|---|---|
| cache 里存什么 | **压缩潜向量** `[B,1,T,576]` | **展开后的 K/V** `[B,128,T,192] + [B,128,T,128]` |
| 每 token 每层元素数 | 576 | **40,960**（**71 倍**） |
| 省显存？ | 省 | **不省**（代码里挂着 TODO，见 `:311` 的注释） |
| 省什么？ | 显存 | **注意力算力**（只算 indexer 选出的 2048 个 key） |

**V3.2 = V3 的长上下文版本（`max_position_embeddings=163840`），靠稀疏注意力省算力，但这份参考实现的 cache 反而变大了**。这在设计上是自洽的（稀疏选择必须在**展开后的** K 上做），但读者必须知道「MLA 的省显存」在 V3.2 里**没有生效**——这是读代码读出来的事实。

---

## 六、V3.2 的 Indexer / DSA 是什么（简介，详细见 2M7）

DeepSeek Sparse Attention（DSA）是「**让每个 query 只看少数 key**」的稀疏注意力。它额外挂了一个**轻量小网络 `DeepseekV32Indexer`**（`modular_deepseek_v32.py:174`）：用 `wq_b` 投出 64 个头 × 128 维的 query（复用 MLA 的 `q_lora_rank` 中间结果 `q_resid`，`:295`）、用 `wk` + `k_norm` 投出**单头**的 128 维 key（有自己的独立 cache，`past_key_values.update_indexer`，`:252`），然后按 `Σ_h weight[b,s,h] · q·k` 给每个 (query, key) 打分（`:254-259`），取 **top-`index_topk`**（默认 2048，`:137`）。选出来的索引被**折进 attention mask**（eager/sdpa 路径下是 `masked_fill(..., finfo.min)` 的加性 mask，`:321-333`；flash-mla 路径直接吃 `indices=`，`:348`）。**唯一在 V3 之外新增的结构就是这个 Indexer**；MLA 投影、MoE、RoPE、DecoderLayer 全部继承 V3。

```bash
$ grep -n "class " src/transformers/models/deepseek_v32/modular_deepseek_v32.py
59:DeepseekV32Config(Glm4MoeLiteConfig)   166:DeepseekV32RMSNorm(DeepseekV3RMSNorm)  # pass
170:DeepseekV32RotaryEmbedding(DeepseekV3RotaryEmbedding)  # pass
174:DeepseekV32Indexer(nn.Module)          # ← 唯一的全新结构
273:DeepseekV32Attention(DeepseekV3Attention)   357:DeepseekV32DecoderLayer(Glm4MoeLiteDecoderLayer)  # pass
361:DeepseekV32PreTrainedModel(DeepseekV3PreTrainedModel)
370:DeepseekV32Model(DeepseekV3Model)      # 覆盖了 forward（要传 position_ids）
427:DeepseekV32ForCausalLM(DeepseekV3ForCausalLM)  # pass
```

**它的 cache 是新增的第三种缓存**：`DynamicIndexedLayer`（`cache_utils.py:319`）/ `StaticIndexedLayer`（`:631`），通过 `layer_types = ["deepseek_sparse_attention"] * n_layers`（`modular_deepseek_v32.py:158`）被分发选中（`cache_utils.py:1236,1258`）——**这就是 2A3 讲的 cache 分发机制的真实用例**。

> **详细展开留给 [2M7 GLM-5.3](./2M7_glm_moe_dsa.md)**：那里有 DSA 的完整分析，以及 GLM 独有的 **IndexShare**（相邻层共享 top-k 选择）。`modular_deepseek_v32.py:20-22` 的文档字符串明确说了「跨层共享是 GLM-MoE-DSA 的创新，那个模型继承自这一个」。
>
> 顺手实测继承链会发现一个反直觉的点：`GlmMoeDsaAttention` 继承的是 **`DeepseekV3Attention`**（`modular_glm_moe_dsa.py:197`），不是 `DeepseekV32Attention`；而 `GlmMoeDsaIndexer(DeepseekV32Indexer)`（`:133`）、`GlmMoeDsaDecoderLayer(DeepseekV32DecoderLayer)`（`:298`）、`GlmMoeDsaConfig(DeepseekV32Config)`（`:50`）、`GlmMoeDsaModel(DeepseekV32Model)`（`:336`）都复用了 V3.2。它把 V3.2 的 forward 又抄了一遍再加共享逻辑（`modular_glm_moe_dsa.py:207-209` 多了 `self.skip_topk` / `prev_topk_indices`）。**这是「modular 继承机制会被绕过」的真实例子**——不是所有复用都发生在类继承层面。

---

## 七、Llama → DeepSeek 对照速查

### 7.1 逐模块对比

| 模块 | Llama（1M1） | DeepSeek V2 | DeepSeek V3 | V3.2 |
|------|-------------|-------------|-------------|------|
| RMSNorm | `LlamaRMSNorm` | `pass` 继承 | `pass` 继承 | `pass` 继承（再传两层） |
| 位置编码 | RoPE，半拆分 | RoPE，**复数形式** (`freqs_cis`) | RoPE，**交错** (`interleave`)，复用 Llama 的 cos/sin | 同 V3 |
| Q 投影 | `q_proj` 一步到位 | **可低秩**：`q_a_proj`→`q_a_layernorm`→`q_b_proj` | 同 V2 | 同 V2（额外留 `q_resid` 给 indexer） |
| KV 投影 | `k_proj` / `v_proj` | **`kv_a_proj_with_mqa` → 潜向量 + k_rot** | 同 V2 | 同 V2 |
| K/V 展开 | 无（直接投出） | **`expand_kv`**（`kv_b_proj` 上投影 + `split` + `cat`） | 同 V2 | 同 V2（`expand_kv` 自动拉取） |
| **cache 存什么** | **K、V 多头** | **压缩潜向量 `[B,1,T,512]` + `k_rot [B,1,T,64]`** | 同 V2 | **展开后的 K/V**（TODO 待修） |
| KV cache / token / 层 | `2·n_kv_heads·head_dim` | **576** | **576** | 40,960 |
| 注意力实现 | eager / sdpa / flash | 同（先展开成标准 MHA） | 同 | 同 + **稀疏 mask** |
| 缩放因子 | `head_dim^-0.5` | **`qk_head_dim^-0.5`**（192） + YaRN mscale | 同 V2 | 同 V2 |
| MLP | `LlamaMLP` (SwiGLU) | **MoE**（`DeepseekV2Moe`），前 `first_k_dense_replace` 层 dense | `DeepseekV3MoE`（继承 V2 的 `pass`） | `DeepseekV32MoE`（经 glm4_moe_lite 拉取） |
| 路由专家 | — | 64 个，`moe_intermediate_size=1407` | **256 个，2048** | 256 个，2048 |
| 每 token 激活 | — | `num_experts_per_tok`（默认为 `None`） | **8** | 8 |
| 共享专家 | — | **2 个**（拼成一个加宽 MLP） | **1 个** | 1 个 |
| 路由打分 | — | `softmax` | **`sigmoid` + bias** | 同 V3 |
| 权重归一化 | — | 不做 | **做**（`norm_topk_prob=True`） | 做 |
| 分组路由 | — | 可选（`topk_method`） | **强制**（8 组选 4） | 强制 |
| 负载均衡 | — | 无 aux loss | **无 aux loss**，用 `e_score_correction_bias` | 同 V3 |
| DecoderLayer 基类 | `GradientCheckpointingLayer` | `LlamaDecoderLayer` | `LlamaDecoderLayer` | **`Glm4MoeLiteDecoderLayer`** |
| 类数 / modular 行数 | — | 13 / 426 | 13 / 250 | 9 / 436 |

### 7.2 一句话记住三层关系

```
V3 = V2 + 换 router（sigmoid + bias + 强制分组 + 归一化）+ 换 RoPE（交错）+ 换 MLP 基类
V3.2 = V3 + 一个 Indexer（DSA 稀疏选择），其余全继承
GLM-5.3(2M7) = V3.2 + IndexShare（跨层共享 top-k），其余全继承
```

**这就是为什么 2M7 能被排成压轴**：它把 2M2 的 MLA、2M3 的 MoE、外加 DSA 全用上，而 DecoderLayer 骨架和 1M1 一模一样。

### 7.3 V2 → V3 → V3.2 → GLM 的继承树（实测）

**每条链的根都是「库内通用类」，越往下越专精；每一层都只覆盖一点点。**

```
【注意力链】                        【DecoderLayer 链】
nn.Module                          LlamaDecoderLayer  (llama/modeling_llama.py:284)
  │                                     │
DeepseekV2Attention  (MLA 首次出现)       DeepseekV3DecoderLayer   (v3/modular:197)
  │                                     │
DeepseekV3Attention  (sigmoid 路由那年改的 RoPE)   Glm4MoeDecoderLayer  (glm4_moe/modular:194, pass)
  ├── Glm4MoeLiteAttention (pass)        Glm4MoeLiteDecoderLayer  (glm4_moe_lite/modular:162)
  └── DeepseekV32Attention (+ Indexer/DSA)  DeepseekV32DecoderLayer (v32/modular:357, pass)
          └── GlmMoeDsaIndexer (glm_moe_dsa/modular:133)   GlmMoeDsaDecoderLayer (glm_moe_dsa/modular:298, pass)
      注：GlmMoeDsaAttention 走的是 DeepseekV3Attention，不是 V32Attention（见第六章）

【MoE 路由链】nn.Module → Qwen2MoeTopKRouter
  → DeepseekV2TopkRouter  ← del self.norm_topk_prob   (v2/modular:164)
  → DeepseekV3TopkRouter  ← del self.topk_method + self.e_score_correction_bias   (v3/modular:90-93)
  → Glm4MoeTopkRouter     ← nn.Module.__init__(self)，bias 改 fp32 buffer   (glm4_moe/modular:176)
  → Glm4MoeLiteTopkRouter ← pass
  → DeepseekV32TopkRouter ← 由生成器改名拉取（V32 的 modular 里没写这个类），生成文件里压平成一个 nn.Module 子类
```

---

## 八、一句话总结

> **MLA = 「不缓存 K/V，缓存压缩后的潜向量（512 潜 + 64 位置段 = 576/token），
> 解码时用 `kv_b_proj` 现场展开」**——比 GQA-8 还省 3.6 倍显存，表达能力接近 MHA；
> 代价是每步重算展开，以及 RoPE 必须拆出一小段单独处理（decoupled RoPE）。
>
> **DeepSeekMoE = 「256 个小专家 + 1 个共享专家，每 token 激活 8+1 个」**——
> 细粒度提高专家组合数，共享专家吸收通用知识；路由用 **sigmoid + `e_score_correction_bias`
> + group-limited routing（8 组选 4）** 替代 Mixtral 的辅助损失。
>
> **V2 立框架、V3 调路由（sigmoid/bias/强制分组/归一化）与 RoPE（交错）、V3.2 加 Indexer（DSA）。**
> **DecoderLayer 从 V2 到 V3.2 一行没变（`pass`），变的永远只有 attention 和 mlp 两个子模块。**

---

## 验证清单

以下命令全部实际执行过，输出即文中所引。在仓库根目录运行。

```bash
# ── 1. 继承关系（本篇最重要的实测）────────────────────────────
grep -n "^class " src/transformers/models/deepseek_v2/modular_deepseek_v2.py
#   → 49:DeepseekV2Config(LlamaConfig)      255:DeepseekV2Attention(nn.Module)
#     386:DeepseekV2DecoderLayer(LlamaDecoderLayer)
grep -n "class DeepseekV32DecoderLayer" src/transformers/models/deepseek_v32/modular_deepseek_v32.py
#   → 357:class DeepseekV32DecoderLayer(Glm4MoeLiteDecoderLayer):
#     358:    pass
grep -n "class Glm4MoeLiteDecoderLayer\|class Glm4MoeDecoderLayer" \
     src/transformers/models/glm4_moe_lite/modular_glm4_moe_lite.py \
     src/transformers/models/glm4_moe/modular_glm4_moe.py
#   → glm4_moe_lite/modular_glm4_moe_lite.py:162:class Glm4MoeLiteDecoderLayer(Glm4MoeDecoderLayer, nn.Module):
#   → glm4_moe/modular_glm4_moe.py:194:class Glm4MoeDecoderLayer(DeepseekV3DecoderLayer):
# 全链：V32 → Glm4MoeLite → Glm4Moe → V3 → Llama → GradientCheckpointingLayer → nn.Module
# 生成文件里压平成一层：
grep -n "class DeepseekV32DecoderLayer" src/transformers/models/deepseek_v32/modeling_deepseek_v32.py
#   → 627:class DeepseekV32DecoderLayer(GradientCheckpointingLayer):

# ── 2. expand_kv 的真实定义位置 ────────────────────────────────
grep -rn "def expand_kv" src/transformers/models/
#   → deepseek_v2/modular_deepseek_v2.py:313   ← 真实定义（modular）
#     deepseek_v2/modeling_deepseek_v2.py:357  ← 生成结果
#     deepseek_v3/modeling_deepseek_v3.py:419、deepseek_v32/modeling_deepseek_v32.py:413
#   （另有 9 个模型复用同名实现：glm_moe_dsa、glm4_moe_lite、minicpm3、mistral4、longcat_flash、
#     glm5_next、axk1、axk2、youtu —— 共 12 处 `def expand_kv`，3 处属于 DeepSeek）
grep -rn "self.expand_kv(" src/transformers/models/deepseek_v2/modular_deepseek_v2.py
#   → 364:        key_states, value_states = self.expand_kv(k_nope, k_pe)

# ── 3. cache 里到底存了哪个张量 ────────────────────────────────
grep -n "past_key_values.update\|update_indexer" \
     src/transformers/models/deepseek_v2/modular_deepseek_v2.py \
     src/transformers/models/deepseek_v3/modular_deepseek_v3.py \
     src/transformers/models/deepseek_v32/modular_deepseek_v32.py
#   → v2/modular:360:  k_nope, k_pe = past_key_values.update(k_nope, k_pe, self.layer_idx)
#        ↑ 传的是 :353-354 的两个「单头」潜向量 [B,1,S,512] 与 [B,1,S,64] = 576/token/层
#   → v3/modular:171:  kv_nope, k_rot = past_key_values.update(kv_nope, k_rot, self.layer_idx)
#   → v32/modular:313: key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
#        ↑ V3.2 传的是「展开后的」多头 K/V（上一行 311 的注释承认这是待修的 TODO）
#   → v32/modular:252: k = past_key_values.update_indexer(k, self.layer_idx)   ← indexer 独立的第三种 cache

# ── 4. MLA 各维度的真实默认值 ──────────────────────────────────
sed -n '119,132p' src/transformers/models/deepseek_v2/modular_deepseek_v2.py
#   → kv_lora_rank: 512        q_lora_rank: 1536      qk_nope_head_dim: 128
#     qk_rope_head_dim: 64     v_head_dim: 128        moe_intermediate_size: 1407
#     n_routed_experts: 64     n_shared_experts: 2    first_k_dense_replace: 0
#     n_group: None            topk_group: None       topk_method: "greedy"
#     routed_scaling_factor: 1.0                       norm_topk_prob: False
sed -n '134,138p' src/transformers/models/deepseek_v2/modular_deepseek_v2.py
#   → :136 self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim   (= 192)
#     :137 self.head_dim = self.qk_rope_head_dim                             (= 64, 给 RoPE 频率表用)
#   ↓ modular 只写了 4 行，生成文件里却有 LlamaConfig.__post_init__ 被就地展开的 4 行（2A5 的继承展开）
sed -n '122,130p' src/transformers/models/deepseek_v2/configuration_deepseek_v2.py
#   → :126 if self.head_dim is None: / :127 self.head_dim = self.hidden_size // self.num_attention_heads
#     :128 if self.num_key_value_heads is None: / :129 self.num_key_value_heads = self.num_attention_heads
grep -n "num_key_value_heads" src/transformers/models/llama/configuration_llama.py
#   → 89:        if self.num_key_value_heads is None:
#     90:            self.num_key_value_heads = self.num_attention_heads     ← 来源就是这里
grep -n "kv_lora_rank\|q_lora_rank\|qk_rope_head_dim\|v_head_dim\|qk_nope_head_dim\|n_routed_experts\|n_shared_experts\|num_experts_per_tok\|moe_intermediate_size\|first_k_dense_replace\|n_group\|topk_group\|routed_scaling_factor\|norm_topk_prob\|hidden_size\|num_hidden_layers\|num_attention_heads" \
     src/transformers/models/deepseek_v3/configuration_deepseek_v3.py
#   → 84 hidden_size 7168 / 87 num_hidden_layers 61 / 88 num_attention_heads 128
#     90 n_shared_experts 1 / 91 n_routed_experts 256 / 92 routed_scaling_factor 2.5
#     93 kv_lora_rank 512 / 94 q_lora_rank 1536 / 95 qk_rope_head_dim 64
#     96 v_head_dim 128 / 97 qk_nope_head_dim 128 / 98 n_group 8 / 99 topk_group 4
#     100 num_experts_per_tok 8 / 101 first_k_dense_replace 3 / 102 norm_topk_prob True
grep -n "index_topk\|index_head_dim\|index_n_heads" src/transformers/models/deepseek_v32/modular_deepseek_v32.py
#   → 137 index_topk: 2048 / 138 index_head_dim: 128 / 139 index_n_heads: 64

# ── 5. n_routed_experts / n_shared_experts 的真实默认值 ────────
grep -rn "n_routed_experts: int\|n_shared_experts: int" src/transformers/models/deepseek_v2/modular_deepseek_v2.py \
     src/transformers/models/deepseek_v3/configuration_deepseek_v3.py \
     src/transformers/models/deepseek_v32/modular_deepseek_v32.py
#   → v2 :122 n_routed_experts = 64   / :123 n_shared_experts = 2
#   → v3 :91 n_routed_experts = 256   / :90 n_shared_experts = 1
#   → v32:113 n_routed_experts = 256  / :112 n_shared_experts = 1
#   （对照 Mixtral：num_local_experts = 8，configuration_mixtral.py:87）

# ── 6. 没有 aux loss（0 处匹配）───────────────────────────────
grep -rn "aux_loss\|seq_aux_loss\|z_loss\|output_router_logits" \
     src/transformers/models/deepseek_v2/ src/transformers/models/deepseek_v3/
#   → 无输出
grep -rn "aux_loss" src/transformers/models/qwen2_moe/modeling_qwen2_moe.py | head -3
#   → 519:def load_balancing_loss_func(   … 601: self.router_aux_loss_coef = config.router_aux_loss_coef
#   每个 token 都走的共享专家在 :198-200 / 相加在 :208

# ── 7. del self.attr 咒语的真实用例 ────────────────────────────
grep -n "del self\." src/transformers/models/deepseek_v2/modular_deepseek_v2.py \
     src/transformers/models/deepseek_v3/modular_deepseek_v3.py
#   → v2:164  del self.norm_topk_prob     （V2 不做归一化）
#   → v3:90   del self.topk_method        （V3 强制走 group_limited_greedy）
sed -n '86,97p' src/transformers/models/deepseek_v2/modeling_deepseek_v2.py
#   → 生成文件里确实没有 self.norm_topk_prob（被 del 精确删掉了一行赋值）

# ── 8. V3 的 config 是手写的（不是生成文件）────────────────────
for f in src/transformers/models/deepseek_v2/configuration_deepseek_v2.py \
         src/transformers/models/deepseek_v3/configuration_deepseek_v3.py \
         src/transformers/models/deepseek_v32/configuration_deepseek_v32.py; do
  echo "$f: $(grep -c 'automatically generated' $f)"; done
#   → 1 / 0 / 1   ← 只有 V3 的 config 没有生成警告
grep -n "^from .configuration" src/transformers/models/deepseek_v3/modular_deepseek_v3.py
#   → 30:from .configuration_deepseek_v3 import DeepseekV3Config

# ── 9. DSA 的 cache 是新增的第三种 cache 层 ────────────────────
grep -n "class DynamicIndexedLayer\|class StaticIndexedLayer\|def update_indexer" src/transformers/cache_utils.py
#   → 319 / 631 / 338（层）、664（静态层）、1434（Cache.update_indexer）
grep -n "deepseek_sparse_attention" src/transformers/cache_utils.py src/transformers/masking_utils.py \
     src/transformers/configuration_utils.py
#   → cache_utils.py:1236 DynamicIndexedLayer / :1258 StaticIndexedLayer
#     masking_utils.py:1514 create_causal_mask / configuration_utils.py:75 层类型白名单

# ── 10. 行数核对 ───────────────────────────────────────────────
wc -l src/transformers/models/deepseek_v2/modular_deepseek_v2.py \
      src/transformers/models/deepseek_v2/modeling_deepseek_v2.py \
      src/transformers/models/deepseek_v3/modular_deepseek_v3.py \
      src/transformers/models/deepseek_v3/modeling_deepseek_v3.py \
      src/transformers/models/deepseek_v32/modular_deepseek_v32.py \
      src/transformers/models/deepseek_v32/modeling_deepseek_v32.py
#   → 426 / 659 / 250 / 738 / 436 / 854
```

> ⚠️ **不要运行** `make fix-repo` 或 `python utils/modular_model_converter.py deepseek_v2`
> 来做「验证」——它们会重写 `modeling_*.py`。本篇所有结论都可以用上面的**只读**命令复核。

---

## 导航

- 路线图：[00_roadmap.md](./00_roadmap.md)（本篇 = 进阶 · 模型结构 · 2M2）
- 上一篇：[2M1_gemma2.md](./2M1_gemma2.md)（待写）—— Gemma 2 的软上限注意力与交替局部/全局注意力
- 下一篇：[2M3_qwen3_moe.md](./2M3_qwen3_moe.md)—— Mixtral / DeepSeek / Qwen3-MoE 三方 MoE 取舍对比
- **建议紧接着读**：
  - 前置补课：[1M3_mixtral_moe.md](./1M3_mixtral_moe.md) —— 没有它，第 4 章的 router 对比会缺基准。
  - 库机制侧：[2A5_modular_code_generation.md](./2A5_modular_code_generation.md) —— 回头用本篇的
    `del self.norm_topk_prob`、`nn.Module.__init__(self)`、5 层继承压缩成一个 `GradientCheckpointingLayer`
    这三个真实例子，重看一遍 2A5 的「特殊约定速查表」。
  - 终点站：[2M7_glm_moe_dsa.md](./2M7_glm_moe_dsa.md) —— 它继承本篇的 Indexer 与 DecoderLayer，
    再加上 IndexShare；本篇第 6 章的「简介」在那里展开。
