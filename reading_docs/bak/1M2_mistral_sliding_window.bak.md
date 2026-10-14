# Mistral 模型结构精读

> 阅读目标:理解滑动窗口注意力（SWA）。
> Mistral 对 Llama 的改动极少,几乎只有这一件事。
>
> 核心源文件:`src/transformers/models/mistral/modular_mistral.py`（modular 文件,直接读差异）
> 生成文件:`src/transformers/models/mistral/modeling_mistral.py`（由 `make fix-repo` 生成,勿直接编辑）

---

## 前置知识

读这篇之前需要理解 [1M1_llama_structure.md](./1M1_llama_structure.md) 里的：
- `LlamaAttention`：GQA、QKV 投影、RoPE、KV cache
- `LlamaDecoderLayer`：Pre-Norm + 残差结构

---

## modular 文件说明

`modular_mistral.py` 的开头直接 `from ..llama.modeling_llama import LlamaAttention, LlamaDecoderLayer, ...`，
然后只覆盖有差异的部分。

```
MistralAttention     继承 LlamaAttention     ← 只改了这一行
MistralDecoderLayer  继承 LlamaDecoderLayer  ← 只换了 attn 和 mlp 的实例
MistralModel         继承 LlamaModel         ← 只改 mask 选择逻辑
MistralForCausalLM   继承 LlamaForCausalLM   ← 几乎原样
```

整个文件 ~170 行。Llama 有的东西，这里一字不写。

---

## 唯一核心差异：滑动窗口注意力（SWA）

### 标准注意力的问题

在标准 causal attention 里，序列第 `t` 个 token 可以看到位置 `0` 到 `t-1` 的所有 token。
序列越长，KV cache 越大，注意力矩阵也越大（`O(n²)`）。

### 滑动窗口的思路

每个 token 只看**最近 `W` 个 token**（`W` = `sliding_window`，Mistral 7B 默认 4096）。
更早的上下文通过堆叠多层间接传递——第 `k` 层理论上能看到距离 `k×W` 以内的上下文。

### 代码实现

**第一处：attention forward（`modular_mistral.py:78-88`）**

```python
attn_output, attn_weights = attention_interface(
    self,
    query_states,
    key_states,
    value_states,
    attention_mask,
    dropout=0.0 if not self.training else self.attention_dropout,
    scaling=self.scaling,
    sliding_window=getattr(self.config, "sliding_window", None),  # main diff with Llama
    **kwargs,
)
```

就加了这一个参数。`attention_interface` 是个分发器，会根据当前后端
（eager / sdpa / flash）选择不同实现，每种实现都理解 `sliding_window`。

**第二处：mask 选择（`modeling_mistral.py:364`）**

```python
mask_function = (
    create_causal_mask
    if self.config.sliding_window is None
    else create_sliding_window_causal_mask
)
```

`LlamaModel` 固定用 `create_causal_mask`；`MistralModel` 在这里做了一个分支。

**第三处：mask 逻辑本身（`masking_utils.py:92-99`）**

```python
def sliding_window_overlay(sliding_window: int) -> Callable:
    def mask_function(batch_idx, head_idx, q_idx, kv_idx):
        return kv_idx > q_idx - sliding_window   # 只允许最近 W 个位置
    return mask_function
```

最终的 `create_sliding_window_causal_mask` 是
`causal_mask（只看过去）` AND `sliding_window_overlay（只看最近 W 个）`
的组合，用 `and_masks` 合并。

---

## 对 KV Cache 的影响

标准 Llama 推理时 KV cache 随序列长度无限增长。
开启 SWA 后，**只需要保留最近 `W` 个位置的 K、V**，cache 大小变为常数。

### 裁剪到底是谁做的？（实测修正）

我的初稿写的是「cache 仍然用 `DynamicCache`，裁剪由 attention 实现（sdpa / flash）处理，
或者靠 mask 把超出窗口的 score 置 `-inf`」。**这个说法是错的**，实测后的真实链路是：

**第一步：`DynamicCache` 会「变形」成一堆 `DynamicSlidingWindowLayer`。**
`modeling_mistral.py:357` 确实写的是 `DynamicCache(config=self.config)`——
但 `DynamicCache.__init__` 会根据 config 逐层选类
（`cache_utils.py:1791-1793`）：

```python
layer_types, layer_kwargs = get_layer_types_and_kwargs(decoder_config)
layers = [DYNAMIC_LAYER_TYPE_MAPPING[layer_type](**layer_kwargs) for layer_type in layer_types]
```

而 `get_layer_types_and_kwargs`（`:1701`）在 config 没显式给 `layer_types` 时会**从字段推断**：

```python
# cache_utils.py:1708-1714
if config.sliding_window is not None:
    layer_types = ["sliding_attention" for _ in range(config.num_hidden_layers)]
```

Mistral 的 `sliding_window = 4096`（`configuration_mistral.py:80`），
所以 32 层全部映射成 `sliding_attention`
（`DYNAMIC_LAYER_TYPE_MAPPING`，`cache_utils.py:1223`）→ **每层都是 `DynamicSlidingWindowLayer`**。

> 所以「用的是 `DynamicCache`」这句话**形式上对、实质上误导**：
> `DynamicCache` 在这里只是个容器，真正干活的是它内部的 32 个滑动窗口层。

**第二步：裁剪是 cache 层自己做的**（`cache_utils.py:245-254`）：

```python
full_key_states = torch.cat([self.keys, key_states], dim=-2)
full_value_states = torch.cat([self.values, value_states], dim=-2)
# Only cache the last `self.sliding_window - 1` tokens (or all of them if lower than that)
if not self.record_past:
    self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
    self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
```

**注意三个细节**：

| 细节 | 事实 |
|------|------|
| 存几个 | `sliding_window - 1`，**不是** `sliding_window` |
| 返回什么 | `return full_key_states, full_value_states`（`:261`）——**返回未裁剪的完整状态** |
| mask 怎么对齐 | `get_mask_sizes()`（`:265`）返回 `kv_offset = max(cumulative_length - sliding_window + 1, 0)`，让 mask 知道当前窗口在绝对坐标里的位置 |

**为什么返回未裁剪的**：当前这一步的 query 需要看到刚进来的那些 token，
而 cache 里只留 `W-1` 个历史 + 本步的 `query_length` 个新 token，
两者拼起来才是本步注意力真正需要的范围。**裁剪是为了省显存，不是为了限制本步的计算范围**——
限制范围是 mask 的职责。

> 完整链路（2A3 实测）：
> `configuration_mistral.py:80`（`sliding_window=4096`）
> → `modeling_mistral.py:357`（`DynamicCache`）
> → `cache_utils.py:1709-1710`（推断出 `sliding_attention`）
> → `:1226`（映射到 `DynamicSlidingWindowLayer`）
> → `:1793`（实例化）。
> 详见 [2A3_kv_cache.md](./2A3_kv_cache.md)。

### 一句话

cache 大小 = `(W-1) × num_kv_heads × head_dim × 2 × dtype`，**与序列长度无关**。
Mistral 7B（`W=4096`，8 个 KV 头，`head_dim=128`）在 fp16 下约 **0.5 GiB 封顶**。

---

## 其余部分：和 Llama 完全一致

| 模块 | Mistral | Llama |
|------|---------|-------|
| RMSNorm | `MistralRMSNorm`，代码与 `LlamaRMSNorm` 相同 | 同 |
| RoPE | `MistralRotaryEmbedding`，代码相同 | 同 |
| MLP | `MistralMLP`（SwiGLU），代码相同 | 同 |
| GQA | 支持，`num_key_value_heads` 可配置 | 同 |
| Pre-Norm + 残差 | `MistralDecoderLayer` 继承 `LlamaDecoderLayer` | 同 |

---

## 一句话总结

> Mistral = Llama + `sliding_window=W` 这一个参数。
> 结构上没有新概念，读 modular 文件 5 分钟就能确认差异只有这一处。

---

## 下一步

- **[1M3_mixtral_moe.md](./1M3_mixtral_moe.md)**：Mixtral 把 Mistral 的 MLP 换成混合专家（MoE），
  是理解稀疏激活和路由的入口。
- 或者先读 Mistral 的配置文件 `configuration_mistral.py`，看看 `sliding_window`、
  `num_key_value_heads` 这些超参的默认值和约束。

---

## 验证清单

本篇所有行号都在本仓库实测过。在仓库根目录跑：

```bash
# 1. 唯一的核心差异：attention forward 只多了一个参数
sed -n '78,88p' src/transformers/models/mistral/modular_mistral.py
#    → 第 86 行：sliding_window=getattr(self.config, "sliding_window", None),  # main diff with Llama

# 2. sliding_window 的默认值 = 4096
sed -n '80p' src/transformers/models/mistral/configuration_mistral.py
#    → sliding_window: int | None = 4096

# 3. mask 的二选一分支
grep -n "mask_function = create_causal_mask" src/transformers/models/mistral/modeling_mistral.py
#    → 364:  mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask

# 4. 窗口 mask 的定义（只看最近 W 个）
grep -n "def sliding_window_overlay" -A 8 src/transformers/masking_utils.py
#    → 92:  def sliding_window_overlay(sliding_window: int) -> Callable:
#    （内部返回 `kv_idx > q_idx - sliding_window`）

# 5. 【关键】裁剪是 cache 层做的，不是 attention 后端
grep -n "class DynamicSlidingWindowLayer" src/transformers/cache_utils.py
#    → 203
sed -n '243,254p' src/transformers/cache_utils.py
#    → 249-250: full_key_states = torch.cat([self.keys, key_states], dim=-2)
#      253-254: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
#               ↑ 只留 W-1 个（不是 W 个）
sed -n '261p' src/transformers/cache_utils.py
#    → return full_key_states, full_value_states
#    ↑ 但返回的是【未裁剪】的完整状态 —— 裁剪只为省显存，限制范围是 mask 的职责

# 6. 「用 DynamicCache」为什么会变成滑动窗口层
grep -n "layer_types = \[\"sliding_attention\"" src/transformers/cache_utils.py
#    → 1710（get_layer_types_and_kwargs 里，config.sliding_window 非 None 就推断出来）
grep -n "\"sliding_attention\": DynamicSlidingWindowLayer" src/transformers/cache_utils.py
#    → 1226（DYNAMIC_LAYER_TYPE_MAPPING）
grep -n "DYNAMIC_LAYER_TYPE_MAPPING\[layer_type\]" src/transformers/cache_utils.py
#    → 1793（DynamicCache.__init__ 逐层实例化）

# 7. 模型侧确实写的是 DynamicCache（这才是容易误读的地方）
sed -n '357p' src/transformers/models/mistral/modeling_mistral.py
#    → past_key_values = DynamicCache(config=self.config)
#    ↑ 形式上对，但它内部已经是 32 个 DynamicSlidingWindowLayer

# 8. 显存：与序列长度无关
python3 -c "
W=4095; kv_heads=8; head_dim=128; layers=32; bpe=2
per_layer = W*kv_heads*head_dim*2*bpe      # ×2 是 K 和 V
print(f'{per_layer/1024**2:.1f} MiB/层 → 全模型 {per_layer*layers/1024**3:.2f} GiB')
"
#    → 16.0 MiB/层 → 全模型 0.50 GiB（Mistral 7B，fp16，封顶不随 S 增长）

# 9. 确认 mistral 有 modular（所以要读 modular 文件）
ls src/transformers/models/mistral/ | grep modular
#    → modular_mistral.py
```

> **第 5、6、7 条是本篇最容易搞错的地方**（我初稿就写错了）：
> 「Mistral 用 `DynamicCache`」这句话形式上对，但会让人以为裁剪发生在别处。
> 真实链路是 `DynamicCache` **变形**成 32 个 `DynamicSlidingWindowLayer`，裁剪由它们自己做。

---

## 导航

- 路线图：[00_roadmap.md](./00_roadmap.md)（本篇 = 入门 · 模型结构 · 1M2）
- 上一篇：[1M1_llama_structure.md](./1M1_llama_structure.md) —— Llama 骨架
