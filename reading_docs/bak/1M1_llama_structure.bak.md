# Llama 模型结构精读

> 阅读目标:吃透 Llama,它是现代所有 decoder-only LLM(含 GLM / DeepSeek)的骨架。
> 后面看任何模型都是「Llama + 若干改动」。
>
> 源码:`src/transformers/models/llama/modeling_llama.py`(仅 512 行,无 `modular` 文件,完整独立可直读)。

---

## 为什么从 Llama 开始

- Llama 是「祖宗级」模型:它被别人继承,自己不继承任何人。
- 所以 `modeling_llama.py` 就是完整源码,没有任何跳转。
- GLM-5.3 (`glm_moe_dsa`) 的 DecoderLayer 骨架与 Llama **完全一致**(Pre-Norm + 双残差),
  只是把标准 Attention 换成 MLA+DSA、把 MLP 换成 MoE。理解 Llama = 拿到读 GLM 的钥匙。

> **注意一个细节(实测修正)**:GLM 的 `GlmMoeDsaDecoderLayer` **不是**直接继承 `LlamaDecoderLayer`,
> 中间隔着 5 层:`Llama` → `DeepseekV3` → `Glm4Moe` → `Glm4MoeLite` → `DeepseekV32` → `GlmMoeDsa`。
> 骨架确实是 Llama 那套(所以「一眼看穿」的结论成立),但它是穿过整条链传下来的。
> 这个链条为什么能逐层展开成独立代码,见 [2A5](./2A5_modular_code_generation.md);
> 完整链条和 GLM 全部机制见 [2M7](./2M7_glm_moe_dsa.md)。

---

## 整体结构:七个积木块(自底向上)

### 1. `LlamaRMSNorm`(第 52-70 行)—— 归一化

不像传统 LayerNorm 要减均值,RMSNorm 只除以「均方根」:

```python
variance = hidden_states.pow(2).mean(-1, keepdim=True)
hidden_states = hidden_states * torch.rsqrt(variance + eps)
return self.weight * hidden_states
```

**关键点**:先转 `float32` 计算(数值稳定),算完再转回原精度。
GLM、DeepSeek 的 RMSNorm 全是直接抄这个。

---

### 2. `LlamaRotaryEmbedding` + `apply_rotary_pos_emb`(第 73-160 行)—— 位置编码 RoPE

唯一有点数学的地方。核心思想:**不给 token 加位置向量,而是把 Q、K 向量按位置「旋转」一个角度**。相对位置差决定旋转角度差,注意力天然感知相对距离。

- `inv_freq`:不同维度用不同频率(第 108 行)。
- `forward`:根据 `position_ids` 算出每个位置的 `cos` / `sin`。
- `rotate_half` + `apply_rotary_pos_emb`:执行旋转。

> GLM 里的 `apply_rotary_pos_emb_interleave` 就是这个的变体 —— Llama 是「前后半拆分」
> 旋转,GLM 是「交错」旋转,数学等价,只是维度排列不同。

---

### 3. `LlamaMLP`(第 163-176 行)—— 前馈网络(SwiGLU)

一行就是精髓:

```python
down_proj(act_fn(gate_proj(x)) * up_proj(x))
```

三个线性层:`gate` 和 `up` 把维度从 `hidden_size` 升到 `intermediate_size`,
`gate` 过激活函数(SiLU)后与 `up` **逐元素相乘**(这就是 GLU 门控),再用 `down` 降回来。

> 对比 GLM:GLM 大部分层把这个 MLP 换成 **MoE**(256 个这样的专家,每 token 只激活 8 个)。
> 但结构本质就是这个 MLP 的堆叠。

---

### 4. `LlamaAttention`(第 217-280 行)—— 注意力(核心)

标准多头注意力 + **GQA(分组查询注意力)**:

- `q_proj` / `k_proj` / `v_proj` / `o_proj` 四个投影。
  K、V 的头数 `num_key_value_heads` 可以比 Q 的头数少(GQA,省 KV cache)。
- forward 流程(第 254-279 行):
  1. 投影出 Q/K/V,reshape 成多头 `[batch, heads, seq, head_dim]`
  2. 对 Q、K 施加 RoPE
  3. 更新 KV cache(`past_key_values.update`)
  4. 调用注意力实现(eager / sdpa / flash)
  5. `o_proj` 输出投影
- `eager_attention_forward`(第 191-213 行):最朴素实现,`softmax(QK^T · scaling) · V`。
  `repeat_kv` 把少量 KV 头复制成和 Q 头一样多(GQA 的实现细节)。

> **理解 GLM 的关键锚点**:GLM/DeepSeek 把这个标准 attention 换成
> **MLA(Q/KV 低秩压缩)+ DSA(稀疏选 topk)**,但外层 forward 骨架
> (投影 → RoPE → cache → attention → o_proj)和 Llama 一模一样。

---

### 5. `LlamaDecoderLayer`(第 284-324 行)—— 一层

组装上面的积木,记住 **残差 + Pre-Norm** 模式:

```python
# 注意力子层
residual = hidden_states
hidden_states = input_layernorm(hidden_states)   # 先归一化
hidden_states = self_attn(hidden_states, ...)
hidden_states = residual + hidden_states          # 残差连接

# MLP 子层
residual = hidden_states
hidden_states = post_attention_layernorm(hidden_states)
hidden_states = mlp(hidden_states)
hidden_states = residual + hidden_states
```

> GLM 的 `GlmMoeDsaDecoderLayer` 骨架**完全一致**,差异只有三处:多一个入参
> `prev_topk_indices`、多返回一个 `topk_indices`、以及把它传给 `self_attn`。
> 两个残差和两个 Pre-Norm 一字不差。(它继承自 `DeepseekV32DecoderLayer`,不是直接继承 Llama
> ——见本篇开头那条实测修正。)

---

### 6. `LlamaModel`(第 347-411 行)—— 堆叠

```python
embed_tokens(input_ids)          # 1. token → 向量
→ 算 causal_mask 和 RoPE 的 cos/sin
→ for layer in layers:           # 2. 过 N 层 DecoderLayer
      hidden_states = layer(...)
→ norm(hidden_states)            # 3. 最后归一化
```

> RoPE 的 `cos/sin` 在这里**算一次**(第 400 行),然后传给每一层复用,不用每层重算。

---

### 7. `LlamaForCausalLM`(第 421 行)—— 加语言模型头

在 `LlamaModel` 上加一个 `lm_head`(线性层),把 hidden_states 投影到词表大小,
输出每个 token 的预测 logits。训练时算 loss。

---

## 一次前向的完整数据流

```
input_ids [B, S]
  → embed_tokens → [B, S, hidden_size]
  → (预计算 cos/sin 和 causal mask)
  → × N 层:
       ┌─ input_layernorm → attention(RoPE, KV cache) → +residual
       └─ post_attn_layernorm → MLP(SwiGLU) → +residual
  → norm
  → lm_head → logits [B, S, vocab_size]
```

---

## Llama → GLM-5.3 对照速查

（本表为入门版；完整版见 [2M7](./2M7_glm_moe_dsa.md)）

| 模块 | Llama | GLM-5.3 (`glm_moe_dsa`) |
|------|-------|--------------------------|
| 归一化 | RMSNorm | 同(继承链上传下来的 `DeepseekV3RMSNorm`) |
| 位置编码 | RoPE(半拆分) | RoPE(interleaved 交错) |
| 注意力 | MHA + GQA | **MLA**(Q/KV 低秩压缩)+ **DSA**(稀疏 topk) |
| 前馈 | SwiGLU MLP | 前 3 层 dense MLP,其余 **MoE**(256 专家/激活 8) |
| DecoderLayer | Pre-Norm + 残差 | 同,额外传 `topk_indices`(历经 5 层继承) |
| 跨层优化 | 无 | **IndexShare**:相邻层共享 topk 选择 |

---

## 下一步

1. **确认对号**:回头瞄一眼 GLM 的 `GlmMoeDsaDecoderLayer.forward`,发现它就是
   Llama 的 DecoderLayer + 一个 `topk_indices`。这个「原来一样」的瞬间就是入门成功。
2. **选一块深挖**:推荐 RoPE(第 2 块)或 Attention(第 4 块),这俩从 Llama 到 GLM
   变化最大,吃透收益最高。

---

## 验证清单

本篇所有行号都在本仓库实测过。在仓库根目录跑：

```bash
# 1. 文件规模：512 行
wc -l src/transformers/models/llama/modeling_llama.py

# 2. 类清单（这就是「七个积木块」的真实位置）
grep -nE "^class Llama" src/transformers/models/llama/modeling_llama.py
#    53:  LlamaRMSNorm          73:  LlamaRotaryEmbedding    163: LlamaMLP
#   217:  LlamaAttention        284:  LlamaDecoderLayer      328: LlamaPreTrainedModel
#   347:  LlamaModel            421:  LlamaForCausalLM
#
#  ⚠️ 注意：注意力/模型这类带装饰器的类，装饰器在前一行
#     （`@use_kernelized_func` 在 216、`@auto_docstring` 在 346）。
#     用 `grep "^class"` 才是类定义的真实行号。

# 3. RMSNorm 的 float32 处理（第 1 块的核心）
sed -n '63,66p' src/transformers/models/llama/modeling_llama.py
#    input_dtype = hidden_states.dtype
#    hidden_states = hidden_states.to(torch.float32)      ← 先升精度
#    variance = hidden_states.pow(2).mean(-1, keepdim=True)
#    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
#    （再 to(input_dtype) 转回，在下一行）

# 4. RoPE：inv_freq（第 108 行）与 cos/sin 预计算（第 400 行）
grep -n "inv_freq = 1.0 /" src/transformers/models/llama/modeling_llama.py
#    → 108:  inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, ...) / dim))
grep -n "position_embeddings = self.rotary_emb" src/transformers/models/llama/modeling_llama.py
#    → 400:  position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
#    ↑ 全模型只算这一次，78 层（Llama 是 32 层）复用

# 5. rotate_half：确认是「前后半拆分」而非交错
sed -n '130,134p' src/transformers/models/llama/modeling_llama.py
#    x1 = x[..., : x.shape[-1] // 2]
#    x2 = x[..., x.shape[-1] // 2 :]
#    return torch.cat((-x2, x1), dim=-1)
#    ↑ 对比 GLM 的 interleaved 版：2M7

# 6. MLP 就是一行（第 3 块）
grep -n "down_proj = self.down_proj" src/transformers/models/llama/modeling_llama.py
#    → 175:  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
#    ↑ gate 过激活、与 up 逐元素相乘（GLU 门控）、再 down 降维

# 7. DecoderLayer 的 Pre-Norm + 双残差（第 5 块）
sed -n '305,324p' src/transformers/models/llama/modeling_llama.py
#    residual = hidden_states                              ← 存残差
#    hidden_states = self.input_layernorm(hidden_states)    ← 先 norm
#    hidden_states, _ = self.self_attn(...)
#    hidden_states = residual + hidden_states               ← 再加回
#    （MLP 子层同样两步）
#    ↑ 对照 BERT 的 `LayerNorm(hidden + input)`（Post-Norm）：1M5

# 8. GQA 的手动展开与 eager 实现（第 4 块）
grep -n "def repeat_kv\|def eager_attention_forward\|repeat_kv(" src/transformers/models/llama/modeling_llama.py
#    → 179: def repeat_kv(...)          191: def eager_attention_forward(...)
#    → 201/202: key_states = repeat_kv(key, module.num_key_value_groups)
#               value_states = repeat_kv(value, module.num_key_value_groups)
#    ↑ 少量 KV 头复制成和 Q 头一样多 —— 这就是 GQA 的实现细节
sed -n '204,210p' src/transformers/models/llama/modeling_llama.py
#    最朴素的注意力：softmax(QKᵀ · scaling) · V（这里就是全篇第 4 块说的那个公式）

# 9. 确认 llama 目录没有 modular 文件（所以本篇直读生成的 modeling 文件）
ls src/transformers/models/llama/ | grep -c modular    # → 0
```

> **为什么 llama 没有 modular**：它是「祖宗级」模型，不被别人继承的源头。
> 有 modular 的模型要读 `modular_*.py` —— 见 [2A5](./2A5_modular_code_generation.md)。

---

## 导航

- 路线图:[00_roadmap.md](./00_roadmap.md)(本篇 = 入门 · 模型结构 · 1M1)
- 下一篇:[1M2_mistral_sliding_window.md](./1M2_mistral_sliding_window.md) —— Mistral 的滑动窗口注意力
