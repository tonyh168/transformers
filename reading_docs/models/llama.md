# Llama 模型结构 · 字符画

> 源码：`src/transformers/models/llama/modeling_llama.py`（**512 行**，无 `modular_*.py`，完整可直读）。
> 图里的 `:NNN` = 本仓库当前版本的行号，全部实测（见文末[验证清单](#验证清单)）。文字精读见 [`../1M1_llama_structure.md`](../1M1_llama_structure.md)。
## 图 1 · 总图：一次 forward 走完全程
```
input_ids  [B, S]
      |
      v
+--------------------------------------+
|  embed_tokens                  :353  |
+--------------------------------------+
      |  hidden_states [B, S, H]
      v
+------------------------------------------------------------------+
|  LlamaModel                                 :347                 |
|                                                                  |
|  position_embeddings = rotary_emb(...)   :400   (只算一次)       |
|  causal_mask = create_causal_mask(...)   :391   (也只算一次)     |
|                                                                  |
|    +----------------------------------------------------------+  |
|    |  LlamaDecoderLayer   x   num_hidden_layers         :284  |  |
|    |  （内部细节见下面「图 4」）                              |  |
|    +----------------------------------------------------------+  |
|                                                                  |
|  norm  (LlamaRMSNorm)              :357 :413                     |
+------------------------------------------------------------------+
      |  last_hidden_state [B, S, H]
      v
+--------------------------------------+
|  lm_head                       :431  |
+--------------------------------------+
      |  logits [B, S, vocab_size]
      v
```

链路只有三段：**embed → N 层 → norm → lm_head**。`LlamaModel` 负责前三段，`LlamaForCausalLM` 只多加 `lm_head`（`:431`）。
## 图 2 · 模型栈：谁包含谁
```
LlamaForCausalLM                                                    :421
|
+-- .model : LlamaModel                                             :429
|     +-- .embed_tokens : nn.Embedding(vocab_size, hidden_size)     :353
|     +-- .layers       : nn.ModuleList([LlamaDecoderLayer] * N)    :354
|     +-- .norm         : LlamaRMSNorm(hidden_size)                 :357
|     +-- .rotary_emb   : LlamaRotaryEmbedding                      :358
|
+-- .lm_head : nn.Linear(hidden_size, vocab_size, bias=False)       :431
```

`.model` 是**主干**，任务头 `lm_head` 挂在 `LlamaForCausalLM` 上、不在主干里，所以主干最后输出 `last_hidden_state` 而非 logits；`_tied_weights_keys`（`:422`）声明 `lm_head.weight <-> model.embed_tokens.weight` 的配对，但**是否真共享由 `config.tie_word_embeddings` 决定**，默认 `False`（`configuration_llama.py:79`）。
## 图 3 · 张量形状流
```
input_ids                                                                         [B, S]
    |
    |  embed_tokens                                                               :381
    v
hidden_states                                                                     [B, S, H]
    |
    |  position_embeddings = rotary_emb(...)    <- 全模型只算一次                 :400
    |  causal_mask = create_causal_mask(...)    <- 也只算一次                     :391
    |
    |  for decoder_layer in self.layers[:num_hidden_layers]                       :402
    v
hidden_states                                                                     [B, S, H]
    |
    |  norm  (LlamaRMSNorm)                                                       :413
    v
last_hidden_state                                                                 [B, S, H]
    |
    |  lm_head(hidden_states[:, slice(-logits_to_keep, None), :])                 :479 :480
    v
logits                                                                            [B, S, vocab_size]
```

两处「只算一次」：`position_embeddings`（`:400`）与 `causal_mask`（`:391`）算好后传给每层复用；`logits_to_keep` 默认 `0`，而 `slice(-0, None)` 等价于 `slice(0, None)` —— 即**取全部位置**，只有显式传正整数才切片。
## 图 4 · DecoderLayer 内部：Pre-Norm + 双残差
```
LlamaDecoderLayer.forward   (基类 GradientCheckpointingLayer)               :295
    |
    |  residual = hidden_states                                             :305
    v
input_layernorm   (LlamaRMSNorm)                                            :306
    |
    v
self_attn   (LlamaAttention：RoPE + KV cache + attention_interface)         :308
    |
    v
hidden_states = residual + hidden_states                                    :317
    |
    |  residual = hidden_states                                             :320
    v
post_attention_layernorm   (LlamaRMSNorm)                                   :321
    |
    v
mlp   (LlamaMLP / SwiGLU)                                                   :322
    |
    v
hidden_states = residual + hidden_states                                    :323
    |
    v
return hidden_states
```

骨架就是「存 residual → Pre-Norm → 子层 → 加回」，重复两次，子层是 `self_attn`（`:308`）与 `mlp`（`:322`）。`self_attn` 返回 `(attn_output, attn_weights)`，`:308` 写 `hidden_states, _ = self.self_attn(...)`，第二个返回值被丢掉，要 attention 权重走 `_can_record_outputs`（`:340-343`）；基类 `GradientCheckpointingLayer`（`:284`）+ `supports_gradient_checkpointing = True`（`:331`）→ 这一层可直接套 gradient checkpointing。
## 图 5 · Attention 内部（含 `:264` 的 attention_interface 分派）
```
LlamaAttention.forward                                                                :243
    |
    |  input_shape  = hidden_states.shape[:-1]                                        :251
    |  hidden_shape = (*input_shape, -1, self.head_dim)                               :252
    |
    |  query_states = q_proj(x).view(hidden_shape).transpose(1, 2)                    :254
    |  key_states   = k_proj(x).view(hidden_shape).transpose(1, 2)                    :255
    |  value_states = v_proj(x).view(hidden_shape).transpose(1, 2)                    :256
    v
Q [B, num_attention_heads, S, head_dim]     K, V [B, num_key_value_heads, S, head_dim]
    |
    |  q, k = apply_rotary_pos_emb(q, k, cos, sin)      <- 只转 Q/K，不转 V           :259
    |  k, v = past_key_values.update(k, v, layer_idx)   <- KV cache 在这里写入        :262
    v
+-----------------------------------------------------------------+
|  attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(   |
|      self.config._attn_implementation, eager_attention_forward  |
|  )                                                              |
+-----------------------------------------------------------------+
    |
    |  attn_output, attn_weights = attention_interface(self, q, k, v, ...)            :268
    v
attn_output = attn_output.reshape(*input_shape, -1).contiguous()                      :279
attn_output = self.o_proj(attn_output)                                                :280
    v
attn_output                                                                           [B, S, H]
```

- 三个投影只是头数不同：Q 是 `num_attention_heads` 个头，K/V 是 `num_key_value_heads` 个头（GQA）；KV cache 在 `:262` 写入**旋转后**的 K（`:259` 先转，`:262` 后存）。
- `:264-266` 不是直接调用注意力函数，而是**先查表拿实现**、`:268` 才真正调用；这张表是全局默认表（`modeling_utils.py:5156`，类 `AttentionInterface`），`get_interface(impl, default)` 的第二个参数是**兜底**（查不到用 `eager_attention_forward`），`impl` 写了但不在表里会直接 `raise KeyError`（`modeling_utils.py:5177-5180`）：

```
config._attn_implementation   被调用的函数                    实现在哪
------------------------------------------------------------------------------------
"eager"                       eager_attention_forward         llama :191
"sdpa"                        sdpa_attention_forward          integrations/sdpa_attention.py:79
"flash_attention_2"           flash_attention_forward         integrations/flash_attention.py:26
"flash_attention_3"           flash_attention_forward         同上
"flash_attention_4"           flash_attention_forward         同上
"flex_attention"              flex_attention_forward          integrations/flex_attention.py
"paged|sdpa"                  sdpa_attention_paged_forward
"paged|eager"                 eager_paged_attention_forward
```

> 图里把 `query_states` / `key_states` / `value_states` 简写成 `q` / `k` / `v`，源码里**没有**这三个变量名。
## 图 6 · GQA：`repeat_kv` 把少数 KV 头铺开
```
Q  [B, 32, S, D]                                                              <-- q_proj  :230
K  [B,  8, S, D]                                                              <-- k_proj  :233  }  num_key_value_heads = 8
V  [B,  8, S, D]                                                              <-- v_proj  :236  }  （GQA 检查点举例）
         |
         |  repeat_kv(K, num_key_value_groups) = repeat_kv(K, 4)              :201   :225
         v
[B, 8, S, D] --[:, :, None]--> [B, 8, 4, S, D] --reshape--> [B, 32, S, D]
                     :187                          :188
```

`num_key_value_heads` 默认 `None`，在 `configuration_llama.py:89-90` 退化成 `= num_attention_heads`，也就是**纯 MHA**（`n_rep = 1`，直接返回原张量）；上图 32/8 只是 GQA 检查点举例。`repeat_kv` 用 `expand` 不真复制（`:187`），最后 `reshape`（`:188`）才落地；KV cache 只存 `num_key_value_heads` 个头 —— 这就是 GQA 省显存的地方。
## 图 7 · MLP（SwiGLU）
```
x  [B, S, H]
     |
     +----------------------+----------------------+
     |                                             |
     v                                             v
gate_proj [H -> I]  :169              up_proj [H -> I]  :170
     |                                             |
     v                                             |
act_fn = ACT2FN["silu"]  :172                      |
     |                                             |
     +---------------------( * )--------------------+
     |
     v
[B, S, I]
     |
     |  down_proj [I -> H]  :171
     v
out  [B, S, H]
```

一行写完就是（`:175`）：`down_proj(act_fn(gate_proj(x)) * up_proj(x))`；那个 `( * )` 就是门控 —— `gate` 过激活后与 `up` **逐元素相乘**，再由 `down` 降回原维度。默认 `H = 4096`、`I = 11008`（`configuration_llama.py:65-66`）；三个 Linear 的 `bias` 由 `config.mlp_bias` 控制，默认 `False`。
## 图 8 · RoPE：`rotate_half` 是「前后半拆分」
```
one head, head_dim = 8:

  x  = [ x0 x1 x2 x3 | x4 x5 x6 x7 ]
         ^^^^^^^^^^^   ^^^^^^^^^^^
           half 1        half 2

rotate_half(x) = [ -x4 -x5 -x6 -x7 | x0 x1 x2 x3 ]        :130

split at the middle, NOT interleaved:
    x1 = x[..., : x.shape[-1] // 2]                       :132
    x2 = x[..., x.shape[-1] // 2 :]                       :133
    return torch.cat((-x2, x1), dim=-1)                   :134
```

```
q_rot = (q * cos) + (rotate_half(q) * sin)                :158
k_rot = (k * cos) + (rotate_half(k) * sin)                :159

inv_freq = 1.0 / (base ** (arange(0, dim, 2) / dim))       :108
freqs    = inv_freq_expanded @ position_ids_expanded       :122
emb      = cat((freqs, freqs), dim=-1)                     :123
cos      = emb.cos() * attention_scaling                   :124
sin      = emb.sin() * attention_scaling                   :125

base = rope_parameters["rope_theta"]    dim = head_dim      :103 :104
```

Llama 用**「前后半拆分」**（`head_dim` 从中间切开，前后两半交换取负；有的实现用「奇偶交错」，数学等价、只是维度排列不同）。RoPE 全程只有乘加，**没有可学习参数** —— `inv_freq` 是 `nn.Buffer` 而非 `nn.Parameter`（`:88`，`persistent=False`，连 `state_dict` 都不进）。
## 图 9 · RMSNorm：为什么先升 fp32
```
hidden_states  [B, S, H]  (bf16)
     |
     |  input_dtype = hidden_states.dtype                                       :63
     |  hidden_states = hidden_states.to(torch.float32)   <- 升精度             :64
     v
x32  [B, S, H]  (fp32)
     |
     |  variance = x.pow(2).mean(-1, keepdim=True)                              :65
     |  x = x * torch.rsqrt(variance + variance_epsilon)                        :66
     v
x_normed  [B, S, H]  (fp32)
     |
     |  return self.weight * x.to(input_dtype)  <- 缩放 + 降回 bf16             :67
     v
out  [B, S, H]  (bf16)
```

RMSNorm **不减均值、没有 bias**，只有「除以均方根 + 逐元素缩放」；先转 `float32` 再算（`:64`）是数值稳定手段，算完在 `:67` 转回原精度。`eps` 来自 `config.rms_norm_eps`，默认 `1e-6`（`configuration_llama.py:73`）。
## 行号速查
| 行号 | 对象 | 一句话 |
|:--|:--|:--|
| `:52` / `:53` | `@use_kernel_forward_from_hub("RMSNorm")` / `LlamaRMSNorm` | 归一化，先升 fp32 |
| `:73` | `LlamaRotaryEmbedding` | RoPE；`inv_freq` 在 `:108` |
| `:130` / `:138` | `rotate_half` / `apply_rotary_pos_emb` | 前后半拆分旋转 |
| `:163` | `LlamaMLP` | SwiGLU，一行在 `:175` |
| `:179` | `repeat_kv` | GQA 的 KV 铺开 |
| `:191` | `eager_attention_forward` | 最朴素注意力，也是兜底实现 |
| `:216` / `:217` | `@use_kernelized_func(...)` / `LlamaAttention` | 注意力层；分派在 `:264` |
| `:284` | `LlamaDecoderLayer` | 一层；继承 `GradientCheckpointingLayer` |
| `:328` | `LlamaPreTrainedModel` | 基类；`_no_split_modules` 在 `:332` |
| `:347` | `LlamaModel` | 主干（embed + layers + norm） |
| `:421` | `LlamaForCausalLM` | 主干 + `lm_head` |
| `:495` / `:498` / `:502` | 三个任务头 | 都是 `Generic*` 一行继承 |
## 验证清单
本篇所有行号都在本仓库实测过。在仓库根目录跑：

```bash
# 1. 文件规模：512 行，且没有 modular 文件
wc -l src/transformers/models/llama/modeling_llama.py
ls src/transformers/models/llama/ | grep -c modular          # -> 0

# 2. 全部类 / 函数 / 装饰器的真实行号（本篇图里用的就是它）
grep -nE "^class |^def |^@" src/transformers/models/llama/modeling_llama.py
```

```text
52:@use_kernel_forward_from_hub("RMSNorm")
53:class LlamaRMSNorm(nn.Module):
73:class LlamaRotaryEmbedding(nn.Module):
130:def rotate_half(x):
137:@use_kernel_forward_from_hub("rotary_pos_emb")
138:def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
163:class LlamaMLP(nn.Module):
179:def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
191:def eager_attention_forward(
216:@use_kernelized_func(apply_rotary_pos_emb)
217:class LlamaAttention(nn.Module):
284:class LlamaDecoderLayer(GradientCheckpointingLayer):
327:@auto_docstring
328:class LlamaPreTrainedModel(PreTrainedModel):
346:@auto_docstring
347:class LlamaModel(LlamaPreTrainedModel):
420:@auto_docstring
421:class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
495:class LlamaForSequenceClassification(GenericForSequenceClassification, LlamaPreTrainedModel): ...
498:class LlamaForQuestionAnswering(GenericForQuestionAnswering, LlamaPreTrainedModel):
502:class LlamaForTokenClassification(GenericForTokenClassification, LlamaPreTrainedModel): ...
```

> `grep "^class"` 拿到的是**类定义**的行号（`:53` 而不是 `:52`）——
> 因为 `@use_kernel_forward_from_hub` 这类装饰器占掉了前一行。

```bash
# 3. attention_interface 分派（图 5 的那个 box）
sed -n '264,268p' src/transformers/models/llama/modeling_llama.py
#     attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
#         self.config._attn_implementation, eager_attention_forward
#     )
#     (空行)
#     attn_output, attn_weights = attention_interface(

# 4. 分派表本体（在 modeling_utils.py，不在 llama 目录）
sed -n '5156,5167p' src/transformers/modeling_utils.py
#     "flash_attention_4": flash_attention_forward,
#     "flash_attention_3": flash_attention_forward,
#     "flash_attention_2": flash_attention_forward,
#     "flex_attention": flex_attention_forward,
#     "sdpa": sdpa_attention_forward,
#     "paged|flash_attention_4": paged_attention_forward,
#     ... "paged|sdpa" / "paged|eager" ...

# 5. RMSNorm 的 fp32 升降精度（图 9）
sed -n '62,67p' src/transformers/models/llama/modeling_llama.py

# 6. SwiGLU 那一行（图 7）
sed -n '175p' src/transformers/models/llama/modeling_llama.py
#     down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

# 7. cos/sin 只算一次（图 3）
sed -n '400p' src/transformers/models/llama/modeling_llama.py
#     position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

# 8. rotate_half 是「前后半拆分」（图 8）
sed -n '130,134p' src/transformers/models/llama/modeling_llama.py
#     x1 = x[..., : x.shape[-1] // 2]
#     x2 = x[..., x.shape[-1] // 2 :]
#     return torch.cat((-x2, x1), dim=-1)

# 9. config 默认值（图 1 / 图 6 / 图 7 / 图 9 引用的那些）
grep -nE "hidden_size|intermediate_size|num_key_value_heads|rms_norm_eps|tie_word_embeddings|head_dim" \
    src/transformers/models/llama/configuration_llama.py
```

> **为什么 llama 没有 modular**：它是「祖宗级」模型，只被别人继承，自己不继承别人，
> 所以 `modeling_llama.py` 就是手写源文件，可以直接改。
> 有 `modular_*.py` 的模型要读 modular 文件 —— 见 [`../2A5_modular_code_generation.md`](../2A5_modular_code_generation.md)。

导航：长文精读 [`../1M1_llama_structure.md`](../1M1_llama_structure.md) · 路线图 [`../00_roadmap.md`](../00_roadmap.md) · 下一篇 [`../1M2_mistral_sliding_window.md`](../1M2_mistral_sliding_window.md)（Mistral 滑动窗口注意力）
