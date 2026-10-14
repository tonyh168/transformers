# GLM-5.3 精读：MLA + DSA + IndexShare（压轴）

> 阅读目标：把整条路线图的所有线索收口到最后一个模型上。
>
> GLM-5.3 的 HF 模型名是 **`glm_moe_dsa`**。它把前面学过的东西**全用上了**：
> MLA（2M2）、DeepSeekMoE（2M2）、稀疏注意力选择器 DSA（2M2 末尾预告）、
> 以及本篇独有的 **IndexShare（跨层复用 top-k）**。
> 但它的 DecoderLayer **骨架和 1M1 的 Llama 一模一样**——Pre-Norm + 两个残差。
> 读到这里如果能一眼看穿「这就是 Llama + 换了 Attention 和 MLP」，整条线就闭环了。
>
> 源文件：`src/transformers/models/glm_moe_dsa/modular_glm_moe_dsa.py`（404 行，**有 modular**）
> 生成文件：`src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py`（834 行，勿直接编辑）

## 速读区（TL;DR）

**一句话结论**：**GLM-5.3 = Llama 的 DecoderLayer 骨架（穿过 5 层继承传下来）+ MLA + DeepSeekMoE + DSA 稀疏选择器 + IndexShare（GLM 独门）**。
分层看：骨架层（norm / 残差 / 循环）与 1M1 逐行一致，子模块层全部来自 DeepSeek / GLM 家族，
**真正属于 GLM 的创新在「调度层」**（mask 逐层选择 / top-k 层间流动 / cache 双份）。

### 必须记住的断言

| # | 断言 | 出处（文件:行号） |
|---|------|------------------|
| 1 | `GlmMoeDsaDecoderLayer` **不是**直接继承 `LlamaDecoderLayer`，而是 **5 层继承链**：`Llama → DeepseekV3 → Glm4Moe → Glm4MoeLite → DeepseekV32 → GlmMoeDsa` | 第零节 `:27-44`、`:298`；验证 `:599-610` |
| 2 | 与 `LlamaDecoderLayer.forward` 逐行对比，差异**只有三处**：多一个入参 `prev_topk_indices`、把它传给 `self_attn`、多返回一个 `topk_indices` | `:399-425` |
| 3 | ⚠️ MLA 的 cache **存的是 `expand_kv` 之后的完整 K/V，不是压缩潜向量**——代码自带 `TODO` 承认这一点 | `:203-217`（`modular_glm_moe_dsa.py:243-245`）；验证 `:631-634` |
| 4 | ⚠️ **这个取舍是 DeepSeek V3.2 做的，不是 GLM 做的**（同一行代码和同一条 `TODO` 也在 `models/deepseek_v32/modular_deepseek_v32.py:311-313`），GLM 只是继承了这个决定 | `:219-236` |
| 5 | indexer 的激活用 **`F.relu` 而不是 softmax**；它的 `k` 是**单头**（所有 32 个 query 头共享）；多头分数用**学出来的** `weights_proj` 加权聚合 | `:286-293` |
| 6 | `GlmMoeDsaIndexer.forward` 上有 **`@torch.no_grad()`**：indexer 不参与梯度，是一个「冻结的选择器」 | `:295` |
| 7 | RoPE 从「半拆分」改成「**交错**」（`apply_rotary_pos_emb_interleave`），且**主 attention 和 indexer 必须都用交错**，只改一处位置语义就对不上 | `:298-320`（主 attention `:237` / indexer `:164`） |
| 8 | 稀疏落地有**两条路径**：`eager`/`sdpa` 只是把 top-k 之外填 `finfo.min`（**不省任何计算**）；真稀疏靠 `flash_mla_with_kvcache` 的 `indices=` kwarg | `:352-362`（分支判定 `:264`） |
| 9 | `skip_topk` 为真的层**连 `GlmMoeDsaIndexer` 对象都不创建**（`self.indexer = None`），省的不只是计算，还有参数和显存 | `:380-381` |
| 10 | ⚠️ 默认 `index_topk_freq=1` → **每一层都是 `"full"`，IndexShare 实际不生效**（默认关闭，偏保守） | `:143-146`、`:447-449` |
| 11 | `first_k_dense_replace=3` + 78 层 → **前 3 层 dense MLP，后 75 层 MoE**（256 路由专家 + 1 共享，top-8，专家宽 2048） | `:478`、`:484-489`、config 表 `:112` |
| 12 | 每一层要缓存**两份**东西：主 attention 的 K/V（`DynamicLayer`）+ indexer 的单头 K（`DynamicIndexedLayer`，`index_head_dim=128`） | `:512-521`、`cache_utils.py:319` |

### 最容易踩的坑 / 反直觉发现

- **「GLM 直接继承 Llama」是错的**：正文明确写「我最初以为…**实测后发现不是**」——结论仍成立，但理由是「**穿过 5 层继承传下来的**」；这条链同时也是 2A5「为什么需要拓扑排序」的答案。
- **「MLA 省显存」这个卖点在 GLM 上被 DSA 抵消了**：每 token 每层的 cache 从 V3 的 576 元素涨到 40960（**涨 71 倍**）——**两个优化不总是正交**，这是本篇最有价值的发现之一。
- **eager / sdpa 路径下的 DSA 只省「语义」不省计算**：它把分数算完再扔掉，真正的加速必须靠专用 kernel，这解释了 `_attn_implementation` 在这个模型上格外重要。
- **IndexShare 默认不生效**：不显式给 `index_topk_freq` / `index_topk_pattern`，78 层全是 `"full"`。
- **`index_topk_pattern` / `index_topk_freq` / `index_skip_topk_offset` 是「隐式超参」**：它们不是声明的 dataclass 字段，而是通过 `kwargs.get(...)` 读的——`config.json` 里能塞，但 `GlmMoeDsaConfig()` 签名里看不到。
- **数学等价但权重不通用**：交错 RoPE 与半拆分只是维度排列不同，所以 GLM 的 checkpoint 不能直接喂给 DeepSeek V3.2。
- **`layer 78` 的忽略规则是「推断」**：`_keys_to_ignore_on_load_unexpected = [r"model\.layers\.78.*"]`，代码只表明了「忽略 layer 78」，正文明确把它标注为推断（大概率是 MTP 附加层）。

### 读完应该能回答的问题

- `GlmMoeDsaDecoderLayer` 的父类是谁？完整继承链有几层、终点是哪个类？
- GLM 的 KV cache 里存的是压缩潜向量还是完整 K/V？这个决定是谁做的、为什么这么做？
- DSA 的 indexer 为什么用 ReLU 而不是 softmax？它的 key 为什么是单头的？
- IndexShare 是什么？默认配置下它生效吗？要打开得改哪个字段？
- `eager` / `sdpa` 与 `flash_mla` 两条稀疏路径的代价差在哪里？为什么这决定了 `_attn_implementation` 的选择？

## 前置知识

- [1M1_llama_structure.md](./1M1_llama_structure.md)：**必读**。本篇的骨架就是它
- [2M2_deepseek_mla_moe.md](./2M2_deepseek_mla_moe.md)：**必读**。MLA 和 DeepSeekMoE 在那里讲透，本篇不重复
- [2M3_qwen3_moe.md](./2M3_qwen3_moe.md)：MoE 的横向对比（可选）
- [2A3_kv_cache.md](./2A3_kv_cache.md)：`DynamicIndexedLayer` / `update_indexer` 在那里讲
- [2A5_modular_code_generation.md](./2A5_modular_code_generation.md)：**读本篇前必读**，否则看不懂下面的继承链

## 零、先纠正一个常见误解：GLM 不是「直接继承 Llama」

我最初以为 `GlmMoeDsaDecoderLayer` 直接继承 `LlamaDecoderLayer`。**实测后发现不是**——它继承的是 `DeepseekV32DecoderLayer`，而这条链有 **5 层深**：

```
models/llama/modeling_llama.py        LlamaDecoderLayer
  ↓  models/deepseek_v3/modular_…:197      DeepseekV3DecoderLayer(LlamaDecoderLayer)
  ↓  models/glm4_moe/modular_…:194         Glm4MoeDecoderLayer(DeepseekV3DecoderLayer)
  ↓  models/glm4_moe_lite/modular_…:162    Glm4MoeLiteDecoderLayer(Glm4MoeDecoderLayer, nn.Module)
  ↓  models/deepseek_v32/modular_…:357     DeepseekV32DecoderLayer(Glm4MoeLiteDecoderLayer)
  ↓  models/glm_moe_dsa/modular_…:298      GlmMoeDsaDecoderLayer(DeepseekV32DecoderLayer)
```

**为什么这很重要**：① **结论依然成立，但理由不同**——骨架确实还是 Llama 那套 Pre-Norm + 残差，但它是**穿过 5 层继承传下来的**。② **这是 2A5「继承展开」的最佳案例**：modular converter 每次只展开**一层**，但读的是**已生成好的**父模型文件（拓扑序保证 llama → … → deepseek_v32 先转完），所以**逐层累积**，最终 `modeling_glm_moe_dsa.py` 里的类是完全内联的独立代码。③ **这链条本身就是「为什么需要拓扑排序」的答案**（回看 2A5 第五节）：顺序反了，`glm_moe_dsa` 会抄到旧的、未展开的 `deepseek_v32`。

> **读法提醒**：`GlmMoeDsaDecoderLayer.forward` 里那几行 norm/残差是**从 Llama 一路抄下来的**，真正属于 GLM 的新东西只有 `topk_indices` 的传递。这是 2A5「规则 3：`pass` 是『我没意见』的信号」的延伸。

## 一、模块清单：9 个类，但真正的新东西只有 3 个

`modular_glm_moe_dsa.py` 定义 9 个类，`__all__` 只导出 4 个（`:399-404`）：

| 行号 | 类 | 父类 | 真实的改动量 |
|------|-----|------|-------------|
| `:50` | `GlmMoeDsaConfig` | `DeepseekV32Config` | 新增 DSA 相关字段 + `indexer_types` 派生逻辑 |
| `:125` | `GlmMoeDsaRMSNorm` | `DeepseekV3RMSNorm` | **`pass`**（3 行） |
| `:129` | `GlmMoeDsaRotaryEmbedding` | `DeepseekV32RotaryEmbedding` | **`pass`**（3 行） |
| `:133` | **`GlmMoeDsaIndexer`** | `DeepseekV32Indexer` | ⭐ 重写 `forward`：改用 **interleaved RoPE** |
| `:197` | **`GlmMoeDsaAttention`** | `DeepseekV3Attention` | ⭐⭐ 加 **IndexShare**（跨层复用 top-k） |
| `:298` | **`GlmMoeDsaDecoderLayer`** | `DeepseekV32DecoderLayer` | ⭐ 多传一个 `prev_topk_indices` |
| `:332` | `GlmMoeDsaPreTrainedModel` | `DeepseekV32PreTrainedModel` | 一行 `_keys_to_ignore_on_load_unexpected` |
| `:336` | `GlmMoeDsaModel` | `DeepseekV32Model` | 改写 forward：层间传 `topk_indices` + 逐层选 mask |
| `:395` | `GlmMoeDsaForCausalLM` | `DeepseekV32ForCausalLM` | 一行 `_fsdp_plan` |

**404 行的 modular 文件，减去 config 的 75 行文档字符串和样板，真正承载新机制的代码不到 150 行**——印证 1M2 的核心读法：**modular 文件的行数反映的是「与父模型的差异量」，不是模型复杂度。** 两个 `pass` 类（RMSNorm / RoPE）是**显式声明**，本可被自动拉取（2A5 第四节），写出来只为让 `__all__` 和类名前缀整齐。

## 二、Config：所有超参一表打尽

`GlmMoeDsaConfig`（`:50`）的字段（`:82-106`）——**全部实测，本篇的「事实底座」**：

| 字段 | 默认值 | 属于哪一层机制 |
|------|--------|---------------|
| `vocab_size` / `hidden_size` / `intermediate_size` / `num_hidden_layers` | 154880 / 6144 / 12288 / 78 | 主干：词表 / 宽度 / dense MLP 宽度 / 层数 |
| `num_attention_heads` / `num_key_value_heads` | 64 / 64 | MLA 的 Q 头数；MLA 下后者**不产生 GQA 效果**（见 2M2） |
| `q_lora_rank` / `kv_lora_rank` | 2048 / 512 | MLA：Q / KV 的低秩压缩维度 ⭐ |
| `qk_nope_head_dim` / `qk_rope_head_dim` / `v_head_dim` | 192 / 64 / 256 | MLA：不带 RoPE / **专门带 RoPE 的那一小段** / V 的头维度 |
| `n_routed_experts` / `n_shared_experts` / `num_experts_per_tok` | 256 / 1 / 8 | MoE：路由专家 / 共享专家（每 token 都走）/ 每 token 激活几个 |
| `moe_intermediate_size` | 2048 | MoE：**单个专家**的宽度（远小于 dense 的 12288） |
| `routed_scaling_factor` / `n_group` / `topk_group` | 2.5 / 1 / 1 | MoE：路由输出缩放 / 分组路由（此模型未启用，都是 1） |
| `first_k_dense_replace` | 3 | **前 3 层是 dense MLP，其余 75 层是 MoE**（继承自 `DeepseekV32Config:142`） |
| `max_position_embeddings` / `rms_norm_eps` | 202752 / 1e-5 | 位置编码（约 198K）/ Norm |
| **`index_topk`** | **2048** | ⭐ DSA：每个 query 选多少个 token |
| **`index_head_dim`** | **128** | ⭐ DSA：indexer 的头维度 |
| **`index_n_heads`** | **32** | ⭐ DSA：indexer 的头数 |
| **`indexer_types`** | `None` | ⭐ DSA：逐层的 indexer 模式（`"full"` / `"shared"`） |

### `indexer_types` 的派生逻辑（`:108-124`）——理解 IndexShare 的入口

```python
def __post_init__(self, **kwargs):
    # Per-layer indexer mode: a pattern (e.g. `"FSSF..."`) overrides the freq/offset schedule.
    if self.indexer_types is None:
        pattern = kwargs.get("index_topk_pattern")
        if pattern is not None:
            self.indexer_types = ([{"F": "full", "S": "shared"}[c] for c in pattern]
                                  if isinstance(pattern, str) else list(pattern))
        else:
            freq = max(kwargs.get("index_topk_freq", 1), 1)
            offset = kwargs.get("index_skip_topk_offset", 2)
            self.indexer_types = ["full" if (max(i - offset + 1, 0) % freq) == 0 else "shared"
                                  for i in range(self.num_hidden_layers)]
    super().__post_init__(**kwargs)
```

**三种指定方式，优先级从高到低**：① `indexer_types` 直接给列表；② `index_topk_pattern` 给字符串（`"FSSF"`，`F`=full、`S`=shared）；③ `index_topk_freq` + `index_skip_topk_offset` 给周期规则（默认 `freq=1` → **每一层都是 `"full"`**）。

> **注意**：`index_topk_pattern` / `index_topk_freq` / `index_skip_topk_offset` **不是声明的 dataclass 字段**，而是通过 `kwargs.get(...)` 读的——`config.json` 里可以直接塞，但 `GlmMoeDsaConfig()` 的签名里看不到它们。这是一个「隐式超参」的实现模式，读代码时容易漏。

**另一个隐蔽字段 `layer_types`**：`GlmMoeDsaModel.forward` 用 `self.config.layer_types[i]`（`:381`）逐层选 mask，但 `layer_types` **不在 `GlmMoeDsaConfig` 字段列表里**——继承自 `DeepseekV32Config:145`；`models/deepseek_v32/modular_deepseek_v32.py:157` 里 `if self.layer_types is None: self.layer_types = ["deepseek_sparse_attention"] * self.num_hidden_layers`。**本模型 78 层全是 `"deepseek_sparse_attention"`**，所以 `causal_mask_mapping` 只有这一个 key（`:370`）。这是为将来支持「混合注意力层」（像 Gemma2 那样交替）留的扩展点。

> **对比 1M2**：Mistral 用 `create_causal_mask if sliding_window is None else create_sliding_window_causal_mask` 做二选一（全模型一个选择）；GLM 用 `layer_types` 列表做**逐层**选择。

## 三、三个核心机制

### 机制 1：MLA —— 来自 2M2，本篇只确认「它还在」

`GlmMoeDsaAttention`（`:197`）继承 `DeepseekV3Attention`，**完全继承 MLA 的全部结构**：

```python
# modular_glm_moe_dsa.py:226-241（节选）
q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))          # Q 低秩：6144 → 2048
q_states = self.q_b_proj(q_resid).view(query_shape).transpose(1, 2) # 2048 → 64 头
q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

compressed_kv = self.kv_a_proj_with_mqa(hidden_states)              # KV 压缩：6144 → 512+64
kv_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
k_pass = self.kv_a_layernorm(kv_pass).view(batch_size, 1, seq_length, self.kv_lora_rank)

cos, sin = position_embeddings
q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)  # 只旋转 64 维那一小段
query_states = torch.cat((q_pass, q_rot), dim=-1)
key_states, value_states = self.expand_kv(k_pass, k_rot)            # 潜向量 → 完整 K/V
```

逐段对照 2M2 的 MLA：Q 低秩（`q_a_proj` → norm → `q_b_proj`）、KV 压缩（`kv_a_proj_with_mqa` 拆 latent + rope 段）、decoupled RoPE（只转 64 维）、`expand_kv` 展开。**一模一样，没有改动。** 但有一个极其重要的细节：

```python
# modular_glm_moe_dsa.py:243-245
# Sparse-attention models cache the expanded K/V, not the compressed latents. TODO (remi-or): fix this with topk
if past_key_values is not None:
    key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
```

2M2 说 MLA 的精髓是「**cache 里存压缩后的潜向量**」——**但这里存的是 `expand_kv` 之后的完整 K/V**，代码自己挂了 `TODO` 承认。**为什么**：DSA 要按 top-k 索引去 **select** 具体的 K/V 行，若 cache 里只有压缩潜向量就没法在 KV cache 上直接做稀疏选择（得先全展开再选，白算一遍）。这是「**MLA 省显存 vs DSA 稀疏选择**」之间的真实取舍——**最终选了后者，牺牲了 MLA 的显存优势**。

> **归属修正（2M2 交叉验证所得）**：这个取舍**不是 GLM 做的，是 DeepSeek V3.2 做的**（DSA 正是在 V3.2 引入）。同一行代码和同一条 `TODO` 也出现在 `models/deepseek_v32/modular_deepseek_v32.py:311-313`，GLM 只是继承了这个决定。
>
> | 版本 | cache 里存什么 | 每 token 每层 |
> |------|---------------|--------------|
> | DeepSeek **V2** / **V3** | 压缩潜向量（`k_nope` + `k_pe`） | **576** 元素 |
> | DeepSeek **V3.2** | ⚠️ **展开后的完整 K/V** | 40960 元素（**涨 71 倍**） |
> | **GLM-5.3** | 同上（继承 V3.2） | 40960 元素 |
>
> 所以在 GLM 上，「MLA 省显存」这个卖点**实际上被 DSA 抵消了**。**两个优化不总是正交的**——本篇最有价值的发现之一。详见 [2M2_deepseek_mla_moe.md](./2M2_deepseek_mla_moe.md)（MLA 三种版本的完整对照 + 显存算例）。

### 机制 2：DSA —— 用一个小网络选出该看哪些 token

序列长度 20 万时（`max_position_embeddings=202752`），即使有 MLA 压缩，attention 仍要算 `S × T` 的分数矩阵。**DSA（DeepSeek Sparse Attention）**用一个**独立的小网络**（Indexer）给每个 query 算出「最值得看的 top-k 个 token」，然后**只对这 k 个位置做真正的 attention**。`index_topk=2048`——20 万长度里只看 2048 个，**100 倍稀疏**。

#### `GlmMoeDsaIndexer`（`:133-195`）

它自己的投影（`__init__` 在父类 `DeepseekV32Indexer:181-192`）：

```python
self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)  # 2048 → 32×128
self.wk = nn.Linear(self.hidden_size, self.head_dim, bias=False)                   # 6144 → 128（单头！）
self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
self.weights_proj = nn.Linear(self.hidden_size, self.n_heads, bias=False)          # 6144 → 32（每头一个权重）
self.softmax_scale = self.head_dim**-0.5
```

**它和主 MLA attention 完全独立**：自己的 Q/K 投影、自己的 head_dim（128）、自己的 `k_norm`（用 LayerNorm，不是 RMSNorm）。**它不共享 MLA 的权重。** 前向流程（`GlmMoeDsaIndexer.forward`，`:134-194`）：

```
hidden_states [B,S,6144]   q_resid [B,S,2048]
        │  self.wk                 │  self.wq_b
   k_norm → [B,S,1,128]      [B,S,32,128] → split → 只对 rope 段转 RoPE
        └──────────┬───────────────┘
         scores = q · kᵀ × softmax_scale   [B,S,1,32,T]
              F.relu(scores)               ← 注意：不是 softmax！（:167）
   weights = weights_proj(hidden) × n_heads^-0.5   [B,S,1,32]
   index_scores = weights @ scores                  [B,S,T]   ← 32 个头加权求和
        因果 mask（padding + 未来位置置 -inf）→ topk(index_topk) → [B,S,2048] int32
```

**三个值得注意的设计**：① **`k` 是单头的**（`self.wk` 输出 `head_dim` 而非 `n_heads × head_dim`）——省钱，32 个 query 头共享同一份 key。② **激活用 `F.relu`，不是 softmax**（`:167`）——ReLU 让大量分数变 0，天然稀疏且比 softmax 便宜，**这是 DSA 的一个关键设计**。③ **多头分数用「学出来的权重」聚合**（`weights_proj`），而不是平均或 max——网络自己学「哪个 indexer 头更可信」。

`@torch.no_grad()` 装饰在 `forward` 上（`:134`）——**indexer 不参与梯度**，是一个「冻结的选择器」。

#### GLM 对 Upstream 的改动：RoPE 从「半拆分」换成「交错」

父类 `DeepseekV32Indexer` 用 `apply_rotary_pos_emb`（半拆分），GLM 覆盖成 `apply_rotary_pos_emb_interleave`（交错）：

```python
# GlmMoeDsaIndexer.forward（:164-165）
# GLM-MoE-DSA uses interleaved RoPE in the indexer
q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin, unsqueeze_dim=2)
```

| 实现 | 函数 | 来源 | 用在哪 |
|------|------|------|--------|
| 半拆分 | `apply_rotary_pos_emb` | `llama` | DeepSeek V3.2 的 indexer |
| 交错 | `apply_rotary_pos_emb_interleave` | `deepseek_v3`（GLM 从这里 import，`:31`） | **GLM 的 indexer 和主 attention** |

**两种情况必须一致**：GLM 的**主 attention**（`:237`）和**indexer**（`:164`）**都用交错**；只改一处，indexer 选出的 top-k 位置和主 attention 的位置语义就对不上。**数学等价，但权重不通用**——所以 GLM 的 checkpoint 不能直接喂给 DeepSeek V3.2。

#### 稀疏怎么落地：两条路径（`:264-291`）

```python
sparse_indices = None
if self.config._attn_implementation in ("eager", "sdpa"):                          # 分支判定 :264
    index_mask = (topk_indices.new_ones((batch_size, seq_length, key_states.shape[2]), dtype=torch.bool)
                  .scatter(-1, topk_indices.long(), False).unsqueeze(1))            # 把选中的位置设成 False
    if attention_mask is None:
        key_positions = torch.arange(key_states.shape[2], device=hidden_states.device)
        index_mask = index_mask | (key_positions[None, None, None, :] > position_ids[:, None, :, None])
        attention_mask = hidden_states.new_zeros((batch_size, 1, seq_length, key_states.shape[2]))
    attention_mask = attention_mask.masked_fill(index_mask, torch.finfo(hidden_states.dtype).min)
else:
    sparse_indices = topk_indices
attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, eager_attention_forward)
attn_output, attn_weights = attention_interface(   # indices= 由 flash_mla_with_kvcache 消费，eager / SDPA 忽略
    self, query_states, key_states, value_states, attention_mask, scaling=self.scaling,
    dropout=0.0 if not self.training else self.attention_dropout, indices=sparse_indices, **kwargs)
```

| 后端 | 怎么实现稀疏 | 代价 |
|------|-------------|------|
| `eager` / `sdpa` | 把 top-k **之外**的位置在 mask 里填 `finfo.min` | **没有省任何计算**——只是把分数算完再扔掉 |
| `flash_mla` 等 | 把 `topk_indices` 交给 kernel，通过 `indices=` kwarg 传真稀疏 | 省算力，但依赖专用 kernel |

**关键洞察**：eager/sdpa 路径下的 DSA **只省 attention 的语义（等价性），不省计算**；真正加速必须靠 `flash_mla_with_kvcache` 这类专用 kernel——这解释了为什么 `_attn_implementation` 在这个模型上**格外重要**。（回看 2A1：`get_interface` 的 `fallback` 参数保底到 `eager_attention_forward`。）

### 机制 3：IndexShare —— 相邻层共享 top-k（GLM 的独门）

**观察**：相邻层的 attention 模式高度相似——第 5 层觉得该看的 token，第 6 层大概率也觉得该看。**IndexShare 的做法**：让一部分层「跳过自己的 indexer」，直接**复用上一个 full 层的 top-k 结果**。

```python
# GlmMoeDsaAttention.__init__（:207-211；Refer: https://arxiv.org/abs/2603.12201）
self.skip_topk = config.indexer_types[layer_idx] == "shared"
self.indexer = None if self.skip_topk else GlmMoeDsaIndexer(config, layer_idx)
```

**`skip_topk` 为真的层，连 `GlmMoeDsaIndexer` 对象都不创建**（`self.indexer = None`）——省的不只是计算，还有参数和显存。forward 里的分流（`:250-258`）：

```python
if self.indexer is not None:
    indexer_mask = attention_mask[:, 0, :, :] if attention_mask is not None else None
    topk_indices = self.indexer(hidden_states, q_resid, position_embeddings,
                                indexer_mask, position_ids, past_key_values=past_key_values)
else:
    if prev_topk_indices is None:
        raise ValueError("Shared DSA layers require top-k indices from a previous full indexer layer.")
    topk_indices = prev_topk_indices
```

**注意那个防御性检查**：配置错了（第一层就是 `"shared"`，没有前序 full 层）会明确报错，而不是静默给出错误结果。

**topk_indices 怎么在层间流动**——这是 GLM-5.3 唯一改动 DecoderLayer 的地方：

```python
# GlmMoeDsaDecoderLayer.forward（:298-330，节选）
def forward(self, ..., prev_topk_indices: torch.Tensor | None = None,   # MAIN DIFF with DSV3.2
            **kwargs):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, _, topk_indices = self.self_attn(hidden_states=hidden_states, ...,
        prev_topk_indices=prev_topk_indices, **kwargs)                  # MAIN DIFF with DSV3.2
    hidden_states = residual + hidden_states          # ← 1M1 的残差
    residual = hidden_states
    hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))   # ← 1M1 的残差
    return hidden_states, topk_indices                # ← 多返回一个东西
```

**和 1M1 的 `LlamaDecoderLayer` 逐行对比，差异只有三处**（代码里用 `# MAIN DIFF with DSV3.2` 标注）：① 多一个入参 `prev_topk_indices`；② 把它传给 `self_attn`；③ 多返回一个 `topk_indices`。**两个残差、两个 Pre-Norm、顺序完全一致。**

`GlmMoeDsaModel.forward` 负责在层间传递（`:376-388`）：

```python
topk_indices = None  # MAIN DIFF with DSV3.2
for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
    hidden_states, topk_indices = decoder_layer(
        hidden_states, attention_mask=causal_mask_mapping[self.config.layer_types[i]],   # 逐层选 mask
        position_embeddings=position_embeddings, position_ids=position_ids,              # 复用的 cos/sin
        past_key_values=past_key_values, use_cache=use_cache,
        prev_topk_indices=topk_indices, **kwargs)   # MAIN DIFF with DSV3.2
```

**对比 1M1 的 `LlamaModel`**：循环体一模一样（层列表、mask、position_embeddings、cache），只是 hidden_states 旁边多了一个 `topk_indices` 一起往下传。**默认配置下收益为零**：`freq=1` → **每层都是 `"full"`，IndexShare 实际不生效**；要启用在 `config.json` 里给 `index_topk_freq`（比如 2 → 每隔一层复用）或 `index_topk_pattern`——**默认关闭，默认配置偏保守的一个例子。**

> **一个有意思的旁证（推断）**：`:333` 有 `_keys_to_ignore_on_load_unexpected = [r"model\.layers\.78.*"]`。模型只有 78 层（索引 0–77），layer 78 是**多出来的**——大概率是 checkpoint 里的 MTP（Multi-Token Prediction）附加层。`glm4_moe_lite` 有同款写法（`[r"model\.layers\.47.*"]`）。**这个推断标注为「推断」——代码只表明了「忽略 layer 78」，没说明它是什么。**

## 四、MoE：来自 2M2，用 `mlp_layer_types` 逐层开关

GLM 的 MLP 部分由继承链上的 `Glm4MoeLiteDecoderLayer.__init__` 决定（`models/glm4_moe_lite/modular_glm4_moe_lite.py:163-170`）：

```python
self.mlp = Glm4MoeLiteMoE(config) if config.mlp_layer_types[layer_idx] == "sparse" else Glm4MoeLiteMLP(config)
```

而 `mlp_layer_types` 由 `first_k_dense_replace` 派生（`models/deepseek_v32/modular_deepseek_v32.py:152-155`）：`n_dense = min(self.first_k_dense_replace, self.num_hidden_layers)`，`self.mlp_layer_types = ["dense"] * n_dense + ["sparse"] * (num_hidden_layers - n_dense)`。

**`first_k_dense_replace=3` + 78 层 → 前 3 层 dense，后 75 层 MoE。** 浅层处理所有 token 都要用的低级特征，放 MoE 没意义（每 token 都会选到相似专家），还增加路由不稳定性和负载均衡难度——**这是 MoE 的一个普遍模式**（对比 1M3 Mixtral：它**全是** MoE 层，没有 dense 保护）。MoE 本体（256 路由专家 + 1 共享，每 token 激活 8 个）见 2M2——**GLM 在此处没有任何改动**，只是把 `first_k_dense_replace` 设成 3。

> `moe_intermediate_size=2048` vs `intermediate_size=12288`：单个专家只有 dense MLP 的 1/6 宽（细粒度），但有 256 个 + 1 共享——这就是 2M2 讲的「细粒度专家」在具体数字上的体现。

## 五、Cache：DSA 需要第二套缓存（呼应 2A3）

indexer 有自己的 key（32 头共享一份单头 key），也得缓存：

```python
# GlmMoeDsaIndexer.forward（:170-171）
if past_key_values is not None:
    k = past_key_values.update_indexer(k, self.layer_idx)
```

`update_indexer` 定义在 `cache_utils.py` 的 **`DynamicIndexedLayer`**（`:319`）和 **`StaticIndexedLayer`**（`:631`）上：`cache_utils.py:338` `DynamicIndexedLayer.update_indexer(...)` = 懒初始化 + `torch.cat` 追加；`cache_utils.py:664` `StaticIndexedLayer.update_indexer(...)` = 就地写入预分配 buffer；`cache_utils.py:1434` `Cache.update_indexer(..., layer_idx)` = 容器层按 `layer_idx` 分发。

**所以 GLM 的每一层要缓存两份东西**：

| 缓存 | 存什么 | 存在哪 |
|------|--------|--------|
| 主 attention 的 K/V | **展开后的** K/V（见第三节的 TODO） | `DynamicLayer` |
| indexer 的 K | 单头、`index_head_dim=128` | `DynamicIndexedLayer` ⭐ |

第二份正是 2A3 那张 Layer 子类表里的 `DynamicIndexedLayer` / `StaticIndexedLayer`——**「给稀疏注意力用的」那个，就是为 DSA 这样的机制准备的。** 从 2A3 到 2M7，这条线索在这里收口。

## 六、一次前向的完整数据流

```
input_ids [B, S] → embed_tokens → [B, S, 6144]
  → rotary_emb → cos/sin（算一次，78 层复用，同 1M1）
  → causal_mask_mapping = {"deepseek_sparse_attention": create_causal_mask(...)}
  → topk_indices = None                                  ← 层间流动的载体
  → × 78 层（前 3 层 dense MLP，后 75 层 MoE）:
       ┌─ input_layernorm → MLA: q_a_proj → q_a_layernorm → q_b_proj → 64 头 Q（切 192 nope / 64 rope）
       │      kv_a_proj_with_mqa → (512 latent, 64 rope)；RoPE(交错) 只作用在 64 维 rope 段
       │      expand_kv(512 latent, 64 rope) → 完整 K/V ⭐（cache 存这个）
       │      ┌─ DSA indexer（"shared" 层则跳过，复用 prev_topk）: wk/k_norm → 单头 key；wq_b → 32 头 query
       │      │    scores = relu(q·kᵀ × scale) → weights_proj 加权求和 → index_scores[B,S,T] → topk(2048)
       │      └─ 稀疏落地：eager/sdpa → mask 填 min；flash_mla → 传 indices= → attention → o_proj
       ├─ + residual → post_attention_layernorm → MLP: dense（前 3 层）或 MoE（256 路由 + 1 共享，top-8）
       └─ + residual → 同时把 topk_indices 传给下一层
  → norm → lm_head → logits [B, S, 154880]
```

## 七、Llama → GLM-5.3 完整对照速查

（1M1 那张表的最终版；`1M1` 早期版本说「GLM 直接继承 Llama」是简化说法，见第零节）

| 模块 | Llama | GLM-5.3 (`glm_moe_dsa`) | 真实来源 |
|------|-------|--------------------------|----------|
| 归一化 / 位置编码 | `LlamaRMSNorm`；`LlamaRotaryEmbedding`（半拆分） | `GlmMoeDsaRMSNorm`；`GlmMoeDsaRotaryEmbedding` | `DeepseekV3RMSNorm`（**`pass`**，`:125`）/ `DeepseekV32RotaryEmbedding`（**`pass`**，`:129`）；**实际用交错** |
| 注意力 | MHA + GQA | **MLA** + **DSA 稀疏选择** | `DeepseekV3Attention` + Indexer（`:197`） |
| 稀疏选择器 / 跨层优化 | 无 | **`GlmMoeDsaIndexer`**（`index_topk=2048`）；**IndexShare**：`"shared"` 层复用前序 full 层的 top-k | `DeepseekV32Indexer`，RoPE 改交错（`:133`）；IndexShare GLM 独有（`:210`） |
| 前馈 | SwiGLU MLP | 前 **3** 层 dense（12288），后 **75** 层 **MoE**（256 路由 + 1 共享，激活 8，专家宽 2048） | `Glm4MoeLite`（`first_k_dense_replace=3`） |
| DecoderLayer | Pre-Norm + 残差 ×2 | **同**，额外传 `prev_topk_indices` | 5 层继承链，终点 `LlamaDecoderLayer` |
| Mask | `create_causal_mask` | `causal_mask_mapping[config.layer_types[i]]`（逐层选择） | `DeepseekV32Model`（`:381`） |
| KV cache | `DynamicCache` | `DynamicLayer`（**展开后的 K/V**）+ `DynamicIndexedLayer`（indexer 的 K） | `cache_utils.py:319` |
| 词表 / 层数 / 上下文 | 32K 量级 / 32 / 8K–128K | **154880 / 78 / 202752** | config（`:82` / `:86` / `:100`） |

## 八、一句话总结

> **GLM-5.3 = Llama 的 DecoderLayer 骨架（穿过 5 层继承传来）+ MLA + DeepSeekMoE + DSA 稀疏选择器 + IndexShare 跨层复用。**
> 分层看：**骨架层**（norm / 残差 / 循环）和 1M1 逐行一致；**子模块层**（Attention / MLP）全部来自 DeepSeek 和 GLM 家族，GLM 自己只加了 IndexShare；**调度层**（mask 逐层选择 / top-k 层间流动 / cache 双份）才是 GLM 真正的创新。
> **大模型的演进，大部分时候不是重新设计骨架，而是在骨架上换零件、加调度。**

## 验证清单

在仓库根目录跑（全部为本篇实际使用过的命令）：

```bash
# 1. 继承链：GLM 的 DecoderLayer 父类不是 Llama
grep -n "^class GlmMoeDsa" src/transformers/models/glm_moe_dsa/modular_glm_moe_dsa.py
#    → 298:class GlmMoeDsaDecoderLayer(DeepseekV32DecoderLayer):

# 2. 逐层回溯到 Llama（5 层）
grep -n "^class DeepseekV32DecoderLayer" src/transformers/models/deepseek_v32/modular_deepseek_v32.py
#    → 357:class DeepseekV32DecoderLayer(Glm4MoeLiteDecoderLayer):
grep -n "^class Glm4MoeLiteDecoderLayer" src/transformers/models/glm4_moe_lite/modular_glm4_moe_lite.py
#    → 162:class Glm4MoeLiteDecoderLayer(Glm4MoeDecoderLayer, nn.Module):
grep -n "^class Glm4MoeDecoderLayer" src/transformers/models/glm4_moe/modular_glm4_moe.py
#    → 194:class Glm4MoeDecoderLayer(DeepseekV3DecoderLayer):
grep -n "^class DeepseekV3DecoderLayer" src/transformers/models/deepseek_v3/modular_deepseek_v3.py
#    → 197:class DeepseekV3DecoderLayer(LlamaDecoderLayer):

# 3. config 字段（本篇「事实底座」）
grep -n "vocab_size: int\|n_routed_experts\|index_topk: int\|indexer_types: list" \
     src/transformers/models/glm_moe_dsa/modular_glm_moe_dsa.py
#    → 82 / 90 / 102 / 106

# 4. 两个 pass 类
sed -n '125,131p' src/transformers/models/glm_moe_dsa/modular_glm_moe_dsa.py

# 5. IndexShare 的开关
grep -n "skip_topk\|prev_topk_indices" src/transformers/models/glm_moe_dsa/modular_glm_moe_dsa.py

# 6. 两条稀疏路径的分界
grep -n '_attn_implementation in ("eager", "sdpa")' src/transformers/models/glm_moe_dsa/modular_glm_moe_dsa.py
#    → 264

# 7. cache：indexer 的第二套缓存
grep -n "def update_indexer" src/transformers/cache_utils.py
#    → 338（DynamicIndexedLayer）/ 664（StaticIndexedLayer）/ 1434（Cache 容器）

# 8. MLA 缓存的是展开后的 K/V（不是潜向量）
grep -n "Sparse-attention models cache the expanded K/V" \
     src/transformers/models/glm_moe_dsa/modular_glm_moe_dsa.py
#    → 243

# 9. 生成文件的开头警告 + 行数对比
head -6 src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py
wc -l src/transformers/models/glm_moe_dsa/modular_glm_moe_dsa.py \
      src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py
#    → 404 / 834
```

## 导航

- 路线图：[00_roadmap.md](./00_roadmap.md)（本篇 = 进阶 · 模型结构 · 2M7，**整条线的终点**）
- 上一篇：[2M6_llava_next_multimodal.md](./2M6_llava_next_multimodal.md)
- 下一篇：无——**读到这里，整条路线图闭环了**
- 回看：[1M1_llama_structure.md](./1M1_llama_structure.md)（用本篇的视角重读，应该已经「一眼看穿」）
