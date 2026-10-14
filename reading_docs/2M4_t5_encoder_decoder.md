# T5 精读：Encoder-Decoder 架构

> 目标：搞清楚**编码器-解码器**和 decoder-only 差在哪，以及 T5 三个「看代码才会发现」的细节：
> **相对位置偏置的分桶**、**`T5Attention` 一个类演三种角色**、**`EncoderDecoderCache` 的双 cache**。
> 它是**唯一一个输出序列与输入序列完全独立**的主流结构族；读完你会明白
> 为什么 `generate()` 在 encoder-decoder 上要**先跑一遍 encoder 再进循环**。
> 源文件：`models/t5/modeling_t5.py`（**1518 行，无 `modular` 文件**）、`configuration_t5.py`（98 行）；
> 对照 `cache_utils.py`（`EncoderDecoderCache`，:1947）、`generation/utils.py`、`integrations/sdpa_attention.py`。

---

## 速读区（TL;DR）

**一句话结论**：encoder-decoder = **两条序列 + 一条 cross-attention 通路**（Q 来自 decoder、K/V 来自 encoder 输出）；
T5 用 **4 个开关**让同一个 `T5Attention` 演三种角色，用**分桶的相对位置偏置**（只在第 0 层算一次）替代位置编码，
并把 cache 拆成 self / cross 两份（`EncoderDecoderCache`）。

**必须记住的断言（12 条）**

| # | 断言 | 出处 |
|---|------|------|
| 1 | T5 **没有 `modular` 文件**（`ls ... \| grep -c modular` → 0），直读 `modeling_t5.py`；文件里那 2 处 `Copied from` 是 docstring 散文，不是同步标记 | 第二节、验证清单 1、16 |
| 2 | 决定角色的是 4 个值：`has_relative_attention_bias`（`:180`）、`is_causal`（`:182`）、`config.is_decoder`（`:186`）、`key_value_states is not None`（`:302`） | `modeling_t5.py:180,182,186,302` |
| 3 | **Q 永远来自 `hidden_states`**；K/V 来自 `hidden_states` 还是 `key_value_states`，只看 `is_cross_attention` | `modeling_t5.py:304`、`:318` |
| 4 | cross-attention 的 K/V **只投影一次**（encoder 输出不变），靠 `is_updated` 这个 per-layer 字典复用 | `modeling_t5.py:319-322`、`:332` |
| 5 | 相对位置偏置**只在第 0 层计算，其余层复用**（`has_relative_attention_bias=bool(i == 0)` → `:740` 只更新不重算） | `modeling_t5.py:648`、`:740`、`:335-348` |
| 6 | **cross-attention 的 `position_bias` 恒为 0**（`:408` 传 `has_relative_attention_bias=False` → `:337` 走全 0 分支） | `modeling_t5.py:408`、`:337-342` |
| 7 | **T5 不做 `1/sqrt(d_k)` 缩放**：`self.scaling = 1.0`，折进了权重初始化 | `modeling_t5.py:197`（对照 `modeling_llama.py:226`） |
| 8 | T5 的 norm 摆放是 **Pre-Norm**（`x + Sublayer(LayerNorm(x))`），只是写在**子层模块内部** | `modeling_t5.py:392-400`、`:138-140` |
| 9 | **输入 embedding 没有缩放**；运行时唯一的 `**-0.5` 在 **decoder 输出侧**，由 `scale_decoder_outputs` 控制 | `modeling_t5.py:1044-1045`、`configuration_t5.py:82` |
| 10 | MLP 默认是 **ReLU 两层**（无门控），但 **config 驱动**的 gated-GELU 变体就在同一个文件里 | `configuration_t5.py:56,69-75`、`modeling_t5.py:126-135` |
| 11 | `T5Block.layer` 长度 **2（encoder）/ 3（decoder）**；`sublayer[0]` 恒为 self-attn、`sublayer[-1]` 恒为 FF | `modeling_t5.py:440-446`、`:459,479,498` |
| 12 | T5 **不能用 FlashAttention**（`_supports_flash_attn = False`），因为 position bias 必须加到 logits 上；SDPA 靠把它折进 mask 才支持 | `modeling_t5.py:540`、`integrations/sdpa_attention.py:39-76` |

**最容易踩的坑 / 反直觉发现**

- **「T5 是 Post-Norm」被实测推翻**（`00_roadmap.md` 的 2M4 描述里写的就是 Post-Norm）：T5 的两段代码正是 `x + Sublayer(LayerNorm(x))` 的形状，判据是看 `self.layer_norm(...)` 出现在残差相加的**之前**（输入侧）还是**之后**（输出侧）。真正的 Post-Norm 是 BERT（`modeling_bert.py:292`、`:350`，先加后归一化）。
- **「T5 有 embedding scaling」也被推翻**：当前代码**没有**对输入 embedding 做缩放，缩放在 decoder 输出侧（`:1044-1045`）；`scale_decoder_outputs` 是把老 config 里的 `tie_word_embeddings` pop 出来当输入的（`configuration_t5.py:82`），而 `tie_word_embeddings` 本身恒为 `True`（`:83`）——**字段名和历史语义完全脱钩**。
- **`encoder_decoder_position_bias` 这个名字会误导**：它恒为 0，cross-attention 是纯内容注意力，decoder 的每个 query 对 encoder 各位置一视同仁。
- **相对 bias 的正确性依赖 `T5Stack` 的顺序遍历**：单独调 `decoder.block[1]` 且不传 `position_bias`，它会**静默地失去相对位置信息**；`:731` / `:734` 是**位置参数**（为 gradient checkpointing），改 `T5Block.forward` 的形参顺序会静默破坏传递。
- **「T5 是 ReLU 不是 GELU」只在默认配置成立**：`feed_forward_proj` 是 config 驱动的，`gated-gelu` 变体（`T5DenseGatedActDense`）就在同一个文件里。
- **decoder 的近端分辨率是 encoder 的两倍**：`num_buckets //= 2` 导致 encoder 8 个精确桶、decoder 16 个精确桶。
- `decoder_start_token_id` **不是 `T5Config` 的声明字段**；且**只在 `decoder_input_ids is None` 时才右移 labels**——自己传了就不右移，传错得自己负责。
- encoder stack 的 cache 被**强制置 None**（`:695`），cross cache 在 T5 里只被写一次。

**读完应该能回答的问题**

- 一个 `T5Attention` 怎么变成三种角色？四个开关各由谁、在什么时候决定？
- 为什么相对 bias 只在第 0 层算一次就够了？其余层凭什么能复用？
- cross-attention 的 Q 和 K/V 分别来自哪里？它的 `position_bias` 是什么？
- T5 为什么用不了 FlashAttention，SDPA 又是怎么绕过去的？
- T5 的 `d_model**-0.5` 到底作用在哪里？（提示：不是 embedding）

---

## 前置知识

- [1M1_llama_structure.md](./1M1_llama_structure.md)：decoder-only 骨架，本篇全程的对照面
- [1M5_bert_modernbert.md](./1M5_bert_modernbert.md)：encoder 对照面；§2.4 论证「BERT 才是 Post-Norm」，与第六节互为正反面
- [2A5_modular_code_generation.md](./2A5_modular_code_generation.md)：决定「读哪个文件」的那篇；本篇第二节是它的实例
- [1A5_generate_loop.md](./1A5_generate_loop.md)：`generate` 主循环（第九节讲 encoder-decoder 差异）
- [2A3_kv_cache.md](./2A3_kv_cache.md)：`EncoderDecoderCache` 的父级上下文

---

## 一、Encoder-Decoder 和 decoder-only 的本质区别

### 1.1 一句话版本

> decoder-only：**一条序列**，用因果 mask 把自己切成一串「前缀 → 下一个词」。
> encoder-decoder：**两条序列**，第一条被 encoder 双向读成一个「表示库」，
> 第二条由 decoder 自回归生成，且**每一步都额外去查一次那个表示库**。

这个「额外查一次」就是 **cross-attention**，是整篇文档的主线。

### 1.2 三个子层的分工

| 子层 | Q 来自 | K / V 来自 | mask | 在 T5 里的构造点 |
|------|--------|------------|------|------------------|
| ① self-attention（encoder） | 输入序列 | 输入序列 | **双向** | `T5LayerSelfAttention` `:372-380` |
| ① self-attention（decoder） | 已生成的部分 | 已生成的部分 | **因果** | 同上，`is_causal=config.is_decoder` `:379` |
| ② cross-attention（只有 decoder 有） | decoder 的当前状态 | **encoder 的输出** | 只 mask padding | `T5LayerCrossAttention` `:404-409` |
| ③ FFN | — | — | — | `T5LayerFF` `:126-141` |

### 1.3 一次 forward 的完整数据流（含 cross-attention 这条线）

```
                    input_ids [B, S]
                          │
   ┌──────────────────────▼────────────────────────────────────────────┐
   │ ENCODER = T5Stack(is_decoder=False)              modeling_t5.py:640│
   │  embed_tokens(共享, :765/:912) :678 → dropout :725                 │
   │  ×num_layers 个 T5Block（只有 ①③）:440-442；① 双向 mask :706-711  │
   │       └ 相对位置 bias 只在第 0 层算一次，其余层复用（:648 ➜ :739） │
   │    ③ FFN (ReLU) :498；final_layer_norm → dropout :744-745          │
   └──────────────────────┬────────────────────────────────────────────┘
        │  encoder_hidden_states [B, S, d_model]  ★ encoder 只跑一次
   ┌──────────────────────▼────────────────────────────────────────────┐
   │ DECODER = T5Stack(is_decoder=True)               modeling_t5.py:640│
   │  decoder_input_ids [B, T] → embed → dropout  :678 / :725；×num_decoder_layers 个 T5Block（①②③）:443-444 │
   │    ① self-attn：causal mask :697-705，K/V 写 self cache（随步长）  │
   │    ② cross-attn：Q=decoder :304，K/V=encoder_hidden_states         │
   │       （:318/:325-326），写 cross cache（第 1 步后不变，:319-322） │
   │    ③ FFN (ReLU)；final_layer_norm :744 → × d_model**-0.5 :1044-1045│
   └──────────────────────┬────────────────────────────────────────────┘
                          ▼
   lm_head（bias=False, :924；与 shared 共享权重, :902-906）→ logits [B, T, vocab]
                          ▼
   CrossEntropyLoss(ignore_index=-100) :1051-1054
```

**读图要点**：从 encoder 指向 decoder ②的箭头**不经过 softmax 之前的任何参数**——encoder 的输出只是一组 K/V，被 `T5Attention` 的 `k` / `v` 两个线性层**投影一次**（不是每层重新编码一遍）。真正的「两序列交互」只发生在这里。

### 1.4 为什么这种形状适合翻译 / 摘要

① encoder 可以**双向读**（`create_bidirectional_mask` vs decoder 的 `create_causal_mask`，`:706-711`）；② decoder 每一步都能**直接 attention 到输入的任何位置**（Bahdanau attention 的动机）；③ 输出长度 `T` 与输入长度 `S` **解耦**；④ 代价：extra 的一次完整 encoder 前向，且不能把 prompt 和回答塞进同一条序列——今天大模型集体转向 decoder-only，主因就是后者。

### 1.5 一个容易被忽略的事实：encoder 和 decoder 是同一个类

`T5Stack` 是共用的类（`:640`），靠 `config.is_decoder` 分叉。两份实例的差别只有 4 处：

| # | 差别 | encoder | decoder | 行号 |
|---|------|---------|---------|------|
| 1 | `is_decoder` | `False` | `True` | `:768` / `:773`（deepcopy config 后改字段） |
| 2 | 层数 | `config.num_layers` | `config.num_decoder_layers`（默认同前者，config:65-67） | `:774` |
| 3 | 有没有 cross-attention 子层 | 没有 | 有 | `:443-444` |
| 4 | mask / cache | 双向 mask，`past_key_values` 强制 None | 因果 mask，创建 `EncoderDecoderCache` | `:706-711` / `:697-705`、`:684-695` |

encoder 副本显式关 cache：`encoder_config.use_cache = False`（`:769`、`:916`）。

---

## 二、读法：T5 没有 `modular` 文件

2A5 第一条规则是「先看有没有 `modular_<name>.py`」；T5 **没有**（实测 0 个 modular 文件），直读 `modeling_t5.py`。T5 **自己**不抄别人，却被别人抄了 90 处（**物理复制**、不是继承）：`mt5` 30 处、`umt5` 19、`longt5` 12、`pop2piano` 11、`udop` 9、`pix2struct` 8、`kosmos2_5` 1（`modeling_mt5.py` 1596 行、`modeling_udop.py` 1679 行，绝大多数与 T5 逐行一致）——**读 T5 一遍 = 读懂了半个 T5 家族**。想找某个 T5 类有没有被改过，就 `grep "Copied from transformers.models.t5"`，没被标的那部分才是它自己的改动。而 `modeling_t5.py` 里 `grep -c "Copied from"` 得 **2**，那两处（`:1209`、`:1431`）是 **docstring 的散文**、不是同步标记（对照 2A5 的正则 `_re_copy_warning`，`check_copies.py:541`：匹配不上）→ **T5 没有任何被同步的代码块，整份文件都是原始代码**，可放心逐行读。

---

## 三、`T5Attention`：一个类的三种身份

**encoder self-attention / decoder self-attention / decoder cross-attention 是同一个类 `T5Attention`**（`:176`）。

### 3.1 四个开关

```python
# modeling_t5.py:176-212
class T5Attention(nn.Module):
    def __init__(
        self, config: T5Config,
        has_relative_attention_bias=False,   # ← 开关 A：是不是「第 0 层」（构造参数, :180）
        layer_idx: int | None = None,
        is_causal: bool = False,             # ← 开关 B：causal 还是双向（构造参数, :182）
    ):
        self.is_decoder = config.is_decoder  # ← 开关 C：从 config 读（:186）
        self.is_causal = is_causal           # :187
        self.has_relative_attention_bias = has_relative_attention_bias   # :188
        self.q, self.k, self.v, self.o = ...  # nn.Linear(..., bias=False)  # :206-209
        if self.has_relative_attention_bias:                             # :211
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)  # :212
```

再加一个 **forward 时才确定的开关**：`is_cross_attention = key_value_states is not None`（`:301-302`）。
开关 A / B 由上层（`T5Stack` → `T5Block` → `T5LayerSelfAttention`）**构造时**传入；开关 C 从 **config** 读；开关 D **每次 forward** 由 `T5Block` 决定。

### 3.2 三种角色的真实构造点

```
              T5Attention (modeling_t5.py:176) —— 一个类，三份实例（构造 :375-380 / :404-409）
   开关 A has_relative_attention_bias :180（只有 block.0 是 True，:648）· B is_causal :182 · C is_decoder :186 · D key_value_states :302
  ┌ encoder self-attn（构造 :375-380）│ decoder self-attn（:375-380）│ decoder cross-attn（:407-409）┐
  │ is_decoder=False · is_causal=False │ is_decoder=True · is_causal=True │ is_decoder=True · is_causal=False（:408）│
  │ rel_bias=第 0 层 · K/V←自己 · bucket 双向（:273）│ rel_bias=第 0 层 · K/V←自己 · bucket 单向（:273）│ rel_bias 恒 False（:408）· K/V←encoder 输出 · position_bias 恒 0（:337）│
  │ 类内部名 SelfAttention           │ self.SelfAttention             │ 类内部名 EncDecAttention │
  └───────────────────────────────────┴─────────────────────────────────┴──────────────────────────────┘
```

构造点（都在 `modeling_t5.py`）：self-attn 用 `is_causal=config.is_decoder`（`:379`，开关 B 来源）；cross-attn 用 `has_relative_attention_bias=False, is_causal=False`（`:408`）；`T5Block.__init__` 里 `if self.is_decoder: self.layer.append(T5LayerCrossAttention(...))`（`:443-444`）。
**记忆锚点**：`T5Block.layer` 是 `ModuleList`，长度 **2（encoder）/ 3（decoder）**；`sublayer[0]` 永远 self-attn、`sublayer[-1]` 永远 FF（`:459`、`:479`、`:498`）。

### 3.3 cross-attention 的 K/V 到底从哪来（关键）

`T5Attention.forward` 里最该记住的 12 行：

```python
# modeling_t5.py:301-332
        is_cross_attention = key_value_states is not None                            # :302
        query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)      # :304 Q 永远来自自己
        if isinstance(past_key_values, EncoderDecoderCache):
            is_updated = past_key_values.is_updated.get(self.layer_idx)
            curr_past_key_values = (past_key_values.cross_attention_cache if is_cross_attention
                                    else past_key_values.self_attention_cache)       # :312 / :314
        current_states = key_value_states if is_cross_attention else hidden_states   # :318 ★
        if is_cross_attention and past_key_values is not None and is_updated:        # 复用分支
            key_states = curr_past_key_values.layers[self.layer_idx].keys            # :321 不投影
            value_states = curr_past_key_values.layers[self.layer_idx].values        # :322
        else:                                                                        # :325 / :326 投影一次
            key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
            value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
```

`key_value_states` 溯源链（4 跳）：`T5ForConditionalGeneration.forward` 把 `encoder_outputs[0]` 传给 `self.decoder(..., encoder_hidden_states=...)`（`:1024`、`:1031-1037`）→ `T5Stack.forward` 形参（`:667`）逐层传入（`:728-737`）→ `T5Block.forward` 的 `self.layer[1](..., key_value_states=encoder_hidden_states, ...)`（`:453`、`:477`、`:479-486`）→ `T5LayerCrossAttention.forward`（`:416`、`:423-430`）→ `T5Attention`（`:302`/`:318`）。

三句话总结：① **Q 永远来自 `hidden_states`**（`:304`）；② **K/V 来自 `hidden_states` 还是 `key_value_states`，只看 `is_cross_attention`**（`:318`），而 `key_value_states` 一路溯源就是 **encoder 的最终输出**；③ **cross-attention 的 K/V 只投影一次**（`:319-322`）——encoder 输出在生成过程中不变，跟自注意力的 K/V（每步都在长）是两种 cache（见第八节）。

### 3.4 反直觉点：T5 不做 `1/sqrt(d_k)` 缩放

```python
# modeling_t5.py:196-197
        self.scaling = 1.0   # T5 folds the relative position bias into the attention scores and does not scale q/k dot product.
```

对照 Llama `modeling_llama.py:226` 是 `self.head_dim**-0.5`。`scaling` 传给注意力实现（T5 `:361`），作用于 `attn_weights = matmul(q, k^T) * scaling`（`eager_attention_forward` `:155-159`）。**T5 用 `1.0`**——原作者把 `1/sqrt(d_k)` 折进权重初始化（`_init_weights`，`:611-616`：`q` 用 `(d_model*d_kv)**-0.5`，`k`/`v` 用 `d_model**-0.5`），所以运行时不缩放。刻意的，不是笔误。

---

## 四、相对位置偏置（relative position bias）

### 4.1 数学直觉：不是给 token 加位置向量，而是给「距离」学一个 bias

| 方案 | 作用对象 | 加在哪里 | 参数量 | T5 里在哪 |
|------|----------|----------|--------|-----------|
| 绝对位置编码（BERT） | 每个**位置** | 加在**输入 embedding** 上 | `max_position × d_model` | BERT: `modeling_bert.py:59` |
| RoPE（Llama） | 每个**位置** | **旋转 Q/K**（不改 logits 形状） | 0 | `modeling_llama.py:73-161` |
| **相对位置偏置（T5）** | 每一对 **(i, j) 的距离** | **加到 attention logits 上** | `num_buckets × num_heads`（32×8=256） | `:212` / `:277` |

```
attention_weights[i, j] = softmax( Q_i · K_j * 1.0 + position_bias[bucket(j - i)] + mask )
                                   每个 (i,j) 一个可学标量，每颗头一套（:212 的第二维是 n_heads）
```

好处：**加性的、per-head 的、且只跟距离有关**。代价：logits 形状必须显式构造成 `[1, H, T, S]`——直接导致 T5 不能上 FlashAttention、SDPA 要把它塞进 mask（第九节）。

### 4.2 `compute_bias` 的真实实现

```python
# modeling_t5.py:264-279
    def compute_bias(self, query_length, key_length, device=None, past_seen_tokens=0):
        context_position = torch.arange(query_length, dtype=torch.long)[:, None] + past_seen_tokens  # :268
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]                        # :269
        relative_position = memory_position - context_position   # (query_length, key_length)        # :270
        relative_position_bucket = self._relative_position_bucket(                                   # :273 ★
            relative_position, bidirectional=(not self.is_decoder),   # encoder 双向、decoder 单向
            num_buckets=self.relative_attention_num_buckets, max_distance=self.relative_attention_max_distance)  # :274-275
        values = self.relative_attention_bias(relative_position_bucket)  # (q, k, heads)   :277
        return values.permute([2, 0, 1]).unsqueeze(0)  # (1, heads, q, k)                   :278
```

四个细节：**`:270` 是 j 减 i，往左看的距离是负数**（符号有意义——`bidirectional` 分支要拿符号分正负两半桶）；**`past_seen_tokens`**（`:268`，增量解码时 query 位置要加已生成 token 数，否则第 10 步的 query 会被当成第 0 个，值来自 `past_key_values.get_seq_length(self.layer_idx)` `:297-299`）；**`bidirectional=(not self.is_decoder)`**（`:273` 看的是 **config** 不是 `key_value_states`，所以 cross-attention 根本没机会生效、`:337` 就被短路）；**输出形状 `[1, H, q, k]`**（`:278`，batch 维是 1，广播到所有 batch）。

### 4.3 分桶：`num_buckets` 与 `max_distance` 怎么用

```python
# modeling_t5.py:238-262   T5Attention._relative_position_bucket（静态方法，:216）
        if bidirectional:
            num_buckets //= 2                                        # :240 一半给正、一半给负
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets   # :241 正距离偏移半区
            relative_position = torch.abs(relative_position)         # :242
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))  # :244
        max_exact = num_buckets // 2                                 # :248  近端：精确桶
        is_small = relative_position < max_exact                     # :249
        relative_position_if_large = max_exact + (                  # :252  远端：对数桶
            torch.log(relative_position.float() / max_exact) / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)).to(torch.long)
        relative_position_if_large = torch.min(                     # :257-259 钳到最后一桶
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1))
        return relative_buckets + torch.where(is_small, relative_position, relative_position_if_large)  # :261
```

**为什么分桶**：附近的位置差别是「语法」级别、需要精确；远处是「话题」级别、粗一点无所谓。于是**一半桶给近距离（一桶一距离）；另一半给远距离（按 log 缩放）；超过 `max_distance` 的全部压进最后一个桶**（`:257-259`，docstring `:226`）。`num_buckets=32`（config:51）、`max_distance=128`（config:52）默认值下：

```
 相对距离 d = memory_position - query_position   (:270，负数 = 往左看)

 ENCODER（bidirectional=True, :273）—— 正负各占一半桶，共 32 桶
   -128 ←─ 8 个对数桶(桶 8..15) ─ 8..1 │ 0 │ 1..8 ─ 8 个精确桶(桶 0..7 / 16..23) ─ 8 个对数桶(桶 24..31) ─→ 128
   编码：:241 正距离统一 +num_buckets/2（=16）的偏移

 DECODER（bidirectional=False, :244）—— 只有 d<=0 合法，正距离一律压成 0
   -128 ←──── 16 个对数桶(桶 16..31) ──── 16..1 │ 0 │ (d>0 → 全部当 0)
   16 个精确桶即桶 0..15 —— 这就是「decoder 近端分辨率是 encoder 两倍」（num_buckets 先被砍半 :240）
```

encoder 有「8 个精确桶」、decoder 有「16 个精确桶」——因为 `num_buckets` 在双向时被先砍半（`:240`）。**decoder 的近端分辨率是 encoder 的两倍**。

**真实桶值**（本机没有 torch，下表用 numpy **逐行复刻 `:238-262` 的公式**算出；脚本与取值范围见验证清单第 7 条，唯一差别是 dtype，桶号完全一致）：encoder 的 `|d|=0..7` 各自一个精确桶，`8-11→8、12-15→9、16-22→10、23-31→11、32-45→12、46-63→13、64-90→14、91 及以上→15`（含 ≥128 全部）；decoder 的 `0..15` 各自一个精确桶，`16-18→16 ... 113 及以上→31`。两边的**正距离**：encoder 用同一张表 `+16`（`:241`），decoder 一律映射到桶 0（`:244` 把正数钳成 0）；桶号范围都是 `[0, 31]`。encoder 侧「91 及以上 → 15」是「公式值 + 钳位（`:257-259`）」叠加的结果。小坑：`torch.log(0)` 是 `-inf`，`.to(torch.long)` 在 `-inf` 上是未定义值——但 `d=0` 时 `is_small` 必为 True，`:261` 的 `torch.where` 会把垃圾值丢掉。

### 4.4 ★ 相对 bias 只在第 0 层**计算**，其余层**复用**

**结论：只在第 0 层算。**证据 4 条，互相独立：

1. **只有第 0 层拿到开关 A**：`:648` `[T5Block(config, has_relative_attention_bias=bool(i == 0), layer_idx=i) for i in range(config.num_layers)]`
2. **只有第 0 层创建那张 embedding 表**：`if self.has_relative_attention_bias: self.relative_attention_bias = nn.Embedding(...)`（`:211-212`）→ 参数只存在于 `block.0.layer.0.SelfAttention.relative_attention_bias.weight`。（旁证：加载器要忽略 `decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight` 这个历史 key，见 `:756`、`:899-901`——**老 checkpoint 的 cross-attention 曾经也有过这张表**。）
3. **`compute_bias` 只在 `has_relative_attention_bias` 为真时被调用**：

```python
# modeling_t5.py:335-348
        if position_bias is None:                                   # :335 ← 别人传了就不算
            key_length = key_states.shape[-2]
            if not self.has_relative_attention_bias:                # :337
                position_bias = torch.zeros((1, self.n_heads, input_shape[1], key_length),  # :338-342 全 0
                                            device=query_states.device, dtype=query_states.dtype)
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(
                    input_shape[1], key_length, device=query_states.device, past_seen_tokens=past_seen_tokens)  # :346-348
```

4. **`T5Stack.forward` 把第 0 层结果当参数传给后面所有层**：

```python
# modeling_t5.py:722-742
        position_bias = None; encoder_decoder_position_bias = None   # :722 / :723  进循环前是 None
        for layer_module in self.block:            # :727
            hidden_states, self_attention_position_bias, cross_attention_position_bias = layer_module(
                hidden_states, attention_mask,
                position_bias,                     # :731  ← 上一层的产物直接喂给下一层（位置参数！）
                encoder_hidden_states, encoder_extended_attention_mask,
                encoder_decoder_position_bias,     # :734  "as a positional argument for gradient checkpointing"
                past_key_values=past_key_values, **kwargs)
            position_bias = self_attention_position_bias                        # :740 ★ 只更新，不重算
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = cross_attention_position_bias   # :742
```

于是第 1..N-1 层的 `position_bias` 进来时**不是 None**（`:335` 分支跳过），既不查表也不算分桶，**只是把第 0 层算好的 `[1, H, q, k]` 张量原样加到自己的 logits 上**（`eager_attention_forward` `:161-162`）。这叫「**算一次、用 N 次**」——因为相对 bias 只依赖 `(i, j, head)`，跟层无关、跟 hidden_states 无关。

**两个必须记住的推论**：① 相对 bias 的正确性是「外部约定」不是类内保证——`T5Attention.forward` 的单层语义是「position_bias 为 None 且我没有表 → 返回全 0」，**单独**调 `decoder.block[1]` 且不传 `position_bias` 会静默失去相对位置信息；正确性依赖 `T5Stack` 的**顺序遍历**（循环不能乱序、不能并行）。② `:731` / `:734` 是**位置参数**（注释写了原因：gradient checkpointing），改动 `T5Block.forward` 的形参顺序会静默破坏传递。

### 4.5 cross-attention 的 position_bias 恒为 0

`T5LayerCrossAttention` 用 `has_relative_attention_bias=False` 构造（`:408`）→ 走 `:337` 全 0 分支 → `:338-342` 生成**全 0** 张量。所以 **cross-attention 是纯内容注意力，不带任何位置信息**：decoder 第 t 个 query 对 encoder 每个位置一视同仁。`encoder_decoder_position_bias` 这个名字（`:455`、`:723`、`:742`）容易误以为里面有东西，实际恒为 0。

### 4.6 位置编码三方对比

| | 1M1 · Llama（RoPE） | 1M5 · BERT（绝对位置编码） | **本篇 · T5（相对位置偏置）** |
|---|---|---|---|
| 载体 / 参数 | 旋转 Q/K 向量 / 0 | 加在输入 embedding，`max_position_embeddings × d_model`（bert:59） | 加在 **attention logits**，`num_buckets × n_heads`（t5:212），**默认 32×8** |
| 作用范围 | 每层、每次前向都算 | 只在 embedding 层加一次 | **只在第 0 层算一次**，全层复用（t5:740） |
| 外推性 | 好（可插值/缩放） | 差（超过 `max_position` 直接没有 embedding） | 好（远处都在最后一个桶，docstring t5:226-227 明确说是为了外推） |
| 位置粒度 | 每个位置精确编码 | 每个位置一个向量 | **距离分桶**：近处精确、远处对数 |
| 加在 mask 里？ | 否 | 否 | **是**（这就是 T5 用不了 FlashAttention 的原因，t5:539-540） |
| 代码位置 | `modeling_llama.py:73-161` | `modeling_bert.py:59`、`:103` | `modeling_t5.py:216-279` |

---

## 五、`T5LayerNorm`

```python
# modeling_t5.py:50-72
        self.weight = nn.Parameter(torch.ones(hidden_size))     # :56 只有 weight，没有 bias
        self.variance_epsilon = eps                             # :57
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)   # :65 不减均值、fp32
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)  # :66
        if self.weight.dtype in [torch.float16, torch.bfloat16]:   # :69
            hidden_states = hidden_states.to(self.weight.dtype)    # :70
        return self.weight * hidden_states                         # :72
```

| | `T5LayerNorm` | `LlamaRMSNorm` | `BertLayerNorm`（= `nn.LayerNorm`） |
|---|---|---|---|
| 减均值 | ❌（`:65` 只有 `pow(2).mean(-1)`） | ❌（`llama:65` 同式） | ✅ |
| bias | ❌（`:56` 只有 `weight`） | ❌（`llama:59`） | ✅（`bias=True` 默认） |
| 精度 | 先转 fp32 算，再**按条件**转回（`:65-70`） | 先转 fp32 算，**无条件**转回输入 dtype（`llama:63-67`） | 按 autocast 走 |
| 定义行 / eps | `:50-72` / `layer_norm_epsilon=1e-6`（config:54） | `llama:53-67` / `rms_norm_eps=1e-6` | `bert:62` 等处 / `layer_norm_eps=1e-12` |

**两者数学上等价——Llama 的 docstring 自己写了**（`modeling_llama.py:53-57`：`LlamaRMSNorm is equivalent to T5LayerNorm`）。唯一实质差别是 `:69-70` 的**条件转换**：T5 只在 `weight` 是 fp16/bf16 时才转回低精度，否则保持 fp32；Llama 无条件转回。正常路径下一致，T5 的写法在「weight 是 fp32、输入是 fp16」的混合精度场景下会保留 fp32 输出（老代码兼容痕迹）。`T5LayerNorm` 的 weight 在 `_init_weights` 里初始化为 `factor * 1.0`（`:567-568`）。

---

## 六、`T5Block` 的 norm 摆放（以及对路线图的一处修正）

### 6.1 先看代码

```python
# modeling_t5.py:384-401  T5LayerSelfAttention.forward
        normed_hidden_states = self.layer_norm(hidden_states)          # :392  ① 先归一化
        attention_output, position_bias, attn_weights = self.SelfAttention(
            normed_hidden_states, ...)                                 # :394  ② 再进子层
        attention_output = hidden_states + self.dropout(attention_output)   # :400 ③ 残差加的是未归一化的输入
# modeling_t5.py:137-141  T5LayerFF.forward
        forwarded_states = self.DenseReluDense(self.layer_norm(hidden_states))   # :138-139
        hidden_states = hidden_states + self.dropout(forwarded_states)           # :140
```

而 **`T5Block.forward` 里一个 norm 都没有**（`:448-509`：只有 `self.layer[0]` / `self.layer[1]` / `self.layer[-1]` 和四段 fp16 `clamp`）。norm 被藏在**每个子层模块内部**。

### 6.2 修正：这是 **Pre-Norm 的摆放**，不是 Post-Norm

路线图（`00_roadmap.md`）在 2M4 描述里写的是「Post-Norm」，**实测代码不支持这个说法**：

- **Post-Norm**（BERT 原版）：`x = LayerNorm(x + Sublayer(x))`——先加残差，再归一化。`modeling_bert.py` 里就是 `hidden_states = self.LayerNorm(hidden_states + input_tensor)`（`BertSelfOutput` `:292`、`BertOutput` `:350` 同形）。
- **Pre-Norm**：`x = x + Sublayer(LayerNorm(x))`——归一化在**输入侧**，残差是干净的通路。T5 的两段代码（`:392-400`、`:138-140`）**正是这个形状**；对照 Llama（`modeling_llama.py:305-317`）：`hidden_states = self.input_layernorm(hidden_states)`（`:306`）→ self_attn → `hidden_states = residual + hidden_states`（`:317`）。

**为什么容易看错**：`T5Block.forward` 把 norm 藏起来了，你只看到 `hidden_states + dropout(attention_output)`（`:400`）这种「残差 + 某个东西」的形状，就会误以为那个「某个东西」是被归一化过的输出。实际上它是 `Sublayer(LayerNorm(x))`。**T5 的 norm 摆放 = Llama 的 Pre-Norm，只是书写位置在子层模块内部（更接近原始 Mesh 代码的写法）。** 判据可复核：看 `self.layer_norm(...)` 出现在残差相加的**之前**（输入侧）还是**之后**（输出侧）。

### 6.3 三个子层的 norm 位置一览

三个子层各自持有 `self.layer_norm`（①`:381`、②`:410`、③`:134`），都对**子层输入**归一化，再 `hidden_states + dropout(...)`（①`:400`、②`:431`、③`:140`）；stack 末尾另有 `self.final_layer_norm`（`:650`，用在 `:744`）。相关行号：①`:381/:392/:400`、②`:410/:422/:431`、③`:134/:138/:140`。

另：**T5 在残差分支里多一个 dropout**（`:400`、`:431`、`:140` 及 stack 级别 `:725`、`:745`），Llama 全程没有（T5 是 2019 年的训练配方，`dropout_rate=0.1`，config:53）。

---

## 七、T5 的经典设计

### 7.1 无 bias 的线性层

真正的 `bias=False` 线性层有 10 处（`grep -c "bias=False"` 得 14，多出的 4 个是 `has_relative_attention_bias=False` 的子串匹配，在 `:180`/`:373`/`:408`/`:436`）：q/k/v/o（`:206-209`）、MLP 非 gated（`:78` `wi`、`:79` `wo`）、MLP gated（`:100-102`）、`lm_head`（`:924`，与 embedding 共享权重 `:902-906`）。例外只在任务头上（`T5ClassificationHead.dense`/`out_proj`，`:517-519` 默认带 bias，`_init_weights` 里 `zeros_(bias)`，`:585-589`）。对照 Llama：`q/k/v/o_proj` 用 `bias=config.attention_bias`，MLP 硬编码 `bias=False`。T5 更彻底：全部不要 bias。

### 7.2 MLP：默认是 **ReLU 两层**，但**有** gated-GELU 变体

**有，而且是同一个文件里的两条分支**，由 `config.feed_forward_proj` 决定：默认 `"relu"`（config:56）→ `act_info = feed_forward_proj.split("-")`，`dense_act_fn = act_info[-1]`（config:70），`is_gated_act = act_info[0] == "gated"`（config:71），`"gated-gelu"` 时 `dense_act_fn = "gelu_new"`（config:74-75）。选择在 `T5LayerFF`：`if config.is_gated_act: T5DenseGatedActDense(config)`（`:130`）`else: T5DenseActDense(config)`（`:132`）。

| | `T5DenseActDense`（默认，T5 1.0） | `T5DenseGatedActDense`（T5 v1.1 / `gated-gelu`） | Llama 的 `LlamaMLP` |
|---|---|---|---|
| 定义 | `:75-94` | `:97-123` | `modeling_llama.py:163-176` |
| 线性层 | `wi`（`:78`）、`wo`（`:79`）——**2 个** | `wi_0`、`wi_1`（`:100-101`）、`wo`（`:102`）——**3 个** | `gate_proj`/`up_proj`/`down_proj` |
| 前向 | `wo(dropout(act(wi(x))))`（`:84-93`） | `wo(dropout(act(wi_0(x)) * wi_1(x)))`（`:107-109`） | `down(act(gate(x)) * up(x))` |
| 激活 | `ACT2FN[config.dense_act_fn]`（`:81`），默认 **ReLU** | 默认 **gelu_new**（config:75） | **SiLU**（有门控） |

准确的说法：**T5 1.0 用 ReLU 的两层 MLP；T5 1.1 用 gated-GELU 的三层 MLP。两条路径都在 `modeling_t5.py` 里，`T5LayerFF` 按 config 选一个。**「T5 是 ReLU 不是 GELU」对默认配置成立，但它是 **config 驱动**的，不是写死的。

### 7.3 共享 embedding，和那个「scaled」的真相

**共享**是三处权重绑成一个（`_tied_weights_keys`，`:902-906`）：`lm_head.weight` / `encoder.embed_tokens.weight` / `decoder.embed_tokens.weight` 都指向 `shared.weight`。

**关于「scaled embedding（`d_model**-0.5`）」——实测：当前代码没有对输入 embedding 做缩放。** 运行时出现的 `**-0.5` 只有一处（其余全在 `_init_weights` 里当初始化标准差，`:577-616`）：

```python
# modeling_t5.py:1042-1047
        sequence_output = decoder_outputs[0]
        if self.config.scale_decoder_outputs:                              # :1044
            sequence_output = sequence_output * (self.model_dim**-0.5)     # :1045
# configuration_t5.py:82-83（注释 :77-81）
        self.scale_decoder_outputs = kwargs.pop("tie_word_embeddings", None) is not False   # :82
        self.tie_word_embeddings = True                                                     # :83
```

**T5 对 decoder 的输出（进 `lm_head` 之前）乘 `d_model**-0.5`**，作用是把共享 embedding 的尺度压回来、控制 logits 量级。老代码是 `if self.config.tie_word_embeddings: sequence_output = sequence_output * (self.model_dim**-0.5)`（可实测：`git show v4.40.0:...modeling_t5.py | grep -n "Rescale output" -B2 -A3` → `1765: if self.config.tie_word_embeddings:` / `1768: ... * (self.model_dim**-0.5)`）。而权重**实际上一直都是绑的**，`tie_word_embeddings` 只被当成「checkpoint 是 T5 1.0 还是 1.1」的标记在用。现在的代码把两个语义**拆开**：`tie_word_embeddings` 恒为 `True`，缩放交给新的 `scale_decoder_outputs`，并从老 config 里 pop 出 `tie_word_embeddings` 当输入——保证老 checkpoint 行为不变。**字段名可能和历史语义完全脱钩。**

### 7.4 loss 与 teacher forcing

`_shift_right`（`:618-637`）做三件事：整体右移一位 `shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()`（`:629`）→ 首位塞 `decoder_start_token_id`（`:630`，T5 里通常 = `pad_token_id`，见 `:624`）→ `.masked_fill_(shifted_input_ids == -100, pad_token_id)`（`:635`，否则 embedding 查表越界）。它**只在 `decoder_input_ids is None` 时**被调用（`:1026`，调用点 `decoder_input_ids = self._shift_right(labels)` `:1028`）；loss 是 `CrossEntropyLoss(ignore_index=-100)`（`:1051`，计算在 `:1054`），`-100` 不计入。

`decoder_start_token_id` **不是 `T5Config` 的声明字段**（`configuration_t5.py:44-62` 里没有它），是 `GenerationConfig` 的字段（`generation/configuration_utils.py:455`），且 `PreTrainedConfig` 允许把未声明的 key 直接 `setattr`（`configuration_utils.py:339-343`）——T5 的 checkpoint config.json 里就带着它（测试里也显式传：`tests/models/t5/test_modeling_t5.py:79`）。`_shift_right` 还被 `prepare_decoder_input_ids_from_labels` 复用（`:1068-1069`）。

---

## 八、`EncoderDecoderCache`（呼应 2A3）

### 8.1 为什么需要两个 cache

自注意力的 K/V 每步**长 1**；cross-attention 的 K/V 在第 1 步**一次写满**（长度 = S），之后永不变化。放进同一个 cache 会冲突：① 长度语义冲突（`get_seq_length()` 答不出是 T 还是 S）；② 白白重算（每步都把 encoder 输出重新投影一遍）。所以设计成两个独立 cache 装在一个壳里。它在 T5 里由 **decoder stack 自建、encoder stack 自毁**：

```python
# cache_utils.py:1996-2008   EncoderDecoderCache（class 定义 :1947）
            self.self_attention_cache = caches[0]        # :2000
            self.cross_attention_cache = caches[1]       # :2001
        self.is_updated = {}                             # :2006 ★ 每层 cross K/V 写没写过
        # :2008  is_updated[i] = cross_attention_cache.get_seq_length(i) > 0（i 遍历各 decoder 层）
# modeling_t5.py:684-695
        if self.is_decoder:
            if use_cache and past_key_values is None:
                if self.config.is_encoder_decoder:
                    past_key_values = EncoderDecoderCache(
                        DynamicCache(config=self.config), DynamicCache(config=self.config))   # :687-689
                else:
                    past_key_values = DynamicCache(config=self.config)   # :691 纯 decoder 的 stack
        elif not self.is_decoder:
            past_key_values = None                                           # :695 ★ encoder 强制不用 cache
```

### 8.2 cache 是怎么被路由的

`T5Attention.forward`（`:307-316`、`:319-322`、`:328-332`）：取出 `is_updated[layer_idx]`（`:309`，这一层的 cross K/V 是否已写过）→ 选 cache（`:310-314`）→ 复用（`:319-322`，cross + 已写过 → **直接读 cache 里的 K/V，不做投影**）→ 写入（`:328-332`，否则投影后写入并给 cross-attn 置 `is_updated=True`，`:332`）。**T5 特有细节**：`:321-322` 直接读 `curr_past_key_values.layers[self.layer_idx].keys`，而 self-attn 走正常 `update()` 路径（`:329`）。所以 **cross cache 在 T5 里只被写一次**。`EncoderDecoderCache` 其余被代理的方法（`cache_utils.py`）：`__len__` `:2021-2026`（= 层数）、`get_seq_length` `:2028-2030`（**代理给 self cache**——生成长度只由 decoder 决定）、`reset` `:2036-2040`（两个都清，`is_updated` 全置 False）、`reorder_cache` `:2042-2045`（beam search 两个都要重排）、`crop` `:2058-2063`（**只裁 self cache**）、`batch_repeat_interleave` `:2065-2069` / `batch_select_indices` `:2071-2075`（两层都动）、`get_mask_sizes` `:2077-2078`、`is_sliding` `:2081-2082`、`is_compileable` `:2085-2086`。`__iter__`（`:2010-2013`）把两层 cache 拼成 6 元组（为 torch DDP 兼容）。

---

## 九、`generate()` 里的 encoder-decoder 差异（呼应 1A5）

### 9.1 大图：encoder 在循环外跑一次

```
 generate()
   ├─ ① _prepare_encoder_decoder_kwargs_for_generation（generation/utils.py:2569-2573 → :826-846）
   │      ★ 在这里把 encoder 跑完，结果塞进 model_kwargs["encoder_outputs"] (:846)；顺手过滤 decoder_* / cross_attn / use_cache / past_key_values (:826-832)
   ├─ ② _prepare_decoder_input_ids_for_generation (:850-898)
   │      起点 = decoder_start_token_id（:2066，BC 时回退到 bos，:2068-2072）；用户传了但不以 start 开头 → 自动补一个（:895-898）
   └─ ③ 逐 token 循环：model_inputs = prepare_inputs_for_generation(...) (:519-643)
          ★ input_ids_key = "decoder_input_ids" (:541)；position_ids_key = "decoder_position_ids" (:559)；原始 attention_mask 仍叫 attention_mask（喂 cross-attn）(:575-578, :605-606)，decoder 自己的 mask 改叫 decoder_attention_mask (:576-578)
          ★ encoder_outputs 走「通用 kwargs 透传」(:608-621)
        调用 T5ForConditionalGeneration.forward → encoder_outputs is not None → 跳过 encoder（:1009-1022）
```

### 9.2 T5 **没有**重写 `prepare_inputs_for_generation`

实测：`grep -c "prepare_inputs_for_generation" src/transformers/models/t5/modeling_t5.py` → **0**。T5 完全吃 `GenerationMixin` 的通用实现（`generation/utils.py:519`），靠 `self.config.is_encoder_decoder` 分叉——关键是 `input_ids_key = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"`（`:541-543`）与 `:575-578` 的 mask 改名。**是「配置驱动」而不是「模型自实现」。** 命名陷阱：进来的 `attention_mask`（长度 S）→ 被当成 **cross-attention 的 mask**（`encoder_attention_mask`，`:575`），并在 `:605-606` 重新以 `attention_mask` 的名字塞进 `model_inputs`；而 `input_ids`（此刻已是 `decoder_input_ids`，长度 T）对应的 mask → 改名 `decoder_attention_mask`（`:576-578`）。

### 9.3 `encoder_outputs` 参数的作用

- `generate` 内部：encoder 跑一次，结果存进 `model_kwargs["encoder_outputs"]`（`generation/utils.py:846`）；beam search 要按 beam 数 `repeat_interleave`（`:934-936`）。
- 每步 forward：非 None → **跳过 encoder**，直接 `hidden_states = encoder_outputs[0]`（`modeling_t5.py:1009-1022`、`:1024`）。
- 用户手传的 tuple 会被重新包成 `BaseModelOutput`（`:1017-1022`）；不用 `generate` 时可显式传 `encoder_outputs` 形参（`:945`）省一次前向；静态 cache / 编译用 `encoder_outputs[0].shape[1]`（= S）当 `max_cache_len` 依据（`generation/utils.py:1884`）。

**这就是「先跑 encoder」的落地方式**：不是循环里的特殊分支，而是「循环开始前把结果塞进 kwargs，让 forward 自己发现已经算过了」。

### 9.4 工程后果：T5 用不了 FlashAttention

```python
# modeling_t5.py:536-543
    _keep_in_fp32_modules = ["wo"]        # :537
    _supports_flash_attn = False          # :540 ★  "Using position bias API ... not yet supported for FA"
    _supports_flex_attn = True            # :541
    _supports_sdpa = True                 # :542
    _can_compile_fullgraph = True         # :543
# integrations/sdpa_attention.py:39-76  create_position_bias_mask(...)：把 position_bias 折进 mask
        position_bias_mask = position_bias            # :68  非 causal → position_bias 直接当加性 mask
        position_bias_mask = torch.where(attention_mask, position_bias, min_dtype)   # :71
# integrations/sdpa_attention.py:107-108   is_causal 从模块属性上读
    is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
```

原因就是 position bias：**它必须被加到 logits 上，而 FlashAttention 的 kernel 不接受任意加性 bias。** SDPA 能支持，是因为它接受一个浮点 mask——框架把 `position_bias` 折进 mask（docstring 原话：`usually prevent sdpa from dispatching to the most efficient kernels`）。`is_causal` 是**从模块属性上读**的——这是 `T5Attention.self.is_causal`（`:187`）在**库内唯一被读到的地方**（`grep -rn 'module, "is_causal"' src/transformers/` 只有 sdpa_attention.py:108 一条命中；`modeling_t5.py` 自身从不读它，全文只出现 `:182`/`:187`/`:379`/`:408` 四次）。2A1 会把这条路讲完。

---

## 三种架构横向对比（decoder-only / encoder-only / encoder-decoder）

| | **decoder-only**（1M1 Llama） | **encoder-only**（1M5 BERT） | **encoder-decoder**（本篇 T5） |
|---|---|---|---|
| 注意力方向 | 因果（只看左） | **双向** | **encoder 双向 + decoder 因果 + cross** |
| 位置编码 | RoPE（旋转 Q/K，`llama:73-161`） | 绝对位置加在 embedding（`bert:59`、`:103`） | **相对 bias 加到 logits，分桶，只在第 0 层算**（`t5:216-279`） |
| norm 摆放 | Pre-Norm（`llama:305-317`） | **Post-Norm**（`bert:292`、`:350`） | **Pre-Norm**，但写在子层内部（`t5:392-400`、`:138-140`） |
| MLP 激活 | SwiGLU 门控（`llama:163-176`） | GELU 两层 | 默认 **ReLU 两层**（`t5:75-94`），另有 gated-GELU 变体（`t5:97-123`） |
| 线性层 bias | 可选（默认 False） | 有 bias | **全部 `bias=False`**（`t5:206-209`、`:78-79`） |
| 权重共享 | 无（未绑定时） | 无 | **`lm_head` + encoder/decoder embedding 三绑一**（`t5:902-906`） |
| cache 结构 | 1 个（`DynamicCache`/`StaticCache`） | 无 | **2 个**（self + cross，`EncoderDecoderCache` cache_utils:1947） |
| 典型任务 → 训练目标 | 通用对话（下一个 token） | 分类 / MLM | 翻译 / 摘要 + teacher forcing（`t5:1026-1028`） |
| `generate()` | 直接进循环 | 不适用 | **先跑一次 encoder，结果进 kwargs**（gen/utils:846 → t5:1009-1022） |
| 后端支持 | 全后端 | 全后端 | **不能用 FlashAttention**（`t5:540`），position bias 要折进 mask（sdpa:39-76） |

**一句话记忆**：encoder-decoder = **encoder-only 的表示 + decoder-only 的自回归 + 一条跨栈的 K/V 通路**。三个架构共用同一批积木（embedding / attention / FFN / 残差 / norm），差别只在 **mask、位置信息怎么进网络、以及 K/V 从哪来**（序列数：decoder-only 1 条、encoder-only 1 条、encoder-decoder **2 条**）。

---

## 一句话总结

> **encoder-decoder 的核心是 cross-attention：Q 来自 decoder，K/V 来自 encoder 的输出**（`t5:318`）；
> 「一个 `T5Attention` 演三种角色」靠 4 个开关（`t5:180`/`:182`/`:186`/`:302`）；T5 的位置信息是**给每一对 (i, j) 的距离分桶后学一个 bias 加到 attention logits 上**，且**只在第 0 层算一次、全层复用**（`t5:648` → `t5:740`）；cache 也因此要分两个（`cache_utils:1947`）。

---

## 验证清单

以下命令都在仓库根目录实际跑过（`cd /Users/baai/Desktop/baai_proj/transformers`）。

```bash
# 1. 没有 modular 文件（所以本篇直读 modeling_t5.py）
ls src/transformers/models/t5/ | grep -c modular
#   → 0

# 2. 行数
wc -l src/transformers/models/t5/modeling_t5.py src/transformers/models/t5/configuration_t5.py
#   → 1518  modeling_t5.py
#   →   98  configuration_t5.py

# 3. ★ 相对 bias 只在第 0 层（证据 1）
grep -n "has_relative_attention_bias=bool(i == 0)" src/transformers/models/t5/modeling_t5.py
#   → 648:            [T5Block(config, has_relative_attention_bias=bool(i == 0), layer_idx=i) for i in range(config.num_layers)]

# 4. ★ 不缩放 QK^T（scaling = 1.0）
grep -n "self.scaling" src/transformers/models/t5/modeling_t5.py
#   → 197:        self.scaling = 1.0
#   → 361:            scaling=self.scaling,
#   对照 Llama：grep -n "self.scaling" src/transformers/models/llama/modeling_llama.py
#   → 226:        self.scaling = self.head_dim**-0.5

# 5. ★ cross-attention 的判定与 K/V 来源
grep -n "is_cross_attention = key_value_states is not None\|current_states = key_value_states\|key_states = self.k(current_states)\|value_states = self.v(current_states)" \
     src/transformers/models/t5/modeling_t5.py
#   → 302:        is_cross_attention = key_value_states is not None
#   → 318:        current_states = key_value_states if is_cross_attention else hidden_states
#   → 325:            key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
#   → 326:            value_states = self.v(current_states).view(kv_shape).transpose(1, 2)

# 6. cross-attention 的 T5Attention 永远 rel_bias=False / is_causal=False
sed -n '404,409p' src/transformers/models/t5/modeling_t5.py
#   → class T5LayerCrossAttention(nn.Module):
#         def __init__(self, config, layer_idx: int | None = None):
#             super().__init__()
#             self.EncDecAttention = T5Attention(
#                 config, has_relative_attention_bias=False, layer_idx=layer_idx, is_causal=False
#             )

# 7. 分桶公式复刻（本机无 torch，用 numpy 逐行重写 :238-262 后取值）
#    脚本 /tmp/bucket2.py：
#      d = np.arange(0, 300); b = buckets(-d, bidirectional=True); u = buckets(-d, bidirectional=False)
#      # buckets() 与 modeling_t5.py:238-262 逐行对应（num_buckets=32, max_distance=128）
#    输出（桶 → 距离区间）：
#      ENCODER: bucket 0..7 → |d| 0..7 各自一个桶；8→8-11；9→12-15；10→16-22；11→23-31；
#               12→32-45；13→46-63；14→64-90；15→91-299
#      DECODER: bucket 0..15 → dist 0..15 各自一个桶；16→16-18；17→19-20；18→21-23；19→24-26；
#               20→27-30；21→31-34；22→35-39；23→40-45；24→46-51；25→52-58；26→59-66；
#               27→67-76；28→77-86；29→87-98；30→99-112；31→113-299

# 8. ★ config 默认值
grep -n "relative_attention_num_buckets\|relative_attention_max_distance\|feed_forward_proj\|d_kv:\|num_heads:\|layer_norm_epsilon\|scale_decoder_outputs" \
     src/transformers/models/t5/configuration_t5.py
#   →  46:    d_kv: int = 64
#   →  50:    num_heads: int = 8
#   →  51:    relative_attention_num_buckets: int = 32
#   →  52:    relative_attention_max_distance: int = 128
#   →  54:    layer_norm_epsilon: float = 1e-6
#   →  56:    feed_forward_proj: str = "relu"
#   →  82:        self.scale_decoder_outputs = kwargs.pop("tie_word_embeddings", None) is not False

# 9. ★ 输入 embedding 没有缩放（runtime 的 **-0.5 只有一处，在 decoder 输出侧）
grep -n "scale\|0.5" src/transformers/models/t5/modeling_t5.py | grep -v "init.normal_"
#   → 196:        # T5 folds the relative position bias into the attention scores and does not scale the query/key dot product.
#   → 197:        self.scaling = 1.0
#   → 1044:        if self.config.scale_decoder_outputs:
#   → 1045:            sequence_output = sequence_output * (self.model_dim**-0.5)
#   历史对照：git show v4.40.0:src/transformers/models/t5/modeling_t5.py | grep -n "Rescale output" -B2 -A3
#   → 1765:        if self.config.tie_word_embeddings:
#   → 1766-1767:            # Rescale output before projecting on vocab / # See https://github.com/tensorflow/mesh/...
#   → 1768:            sequence_output = sequence_output * (self.model_dim**-0.5)

# 10. bias=False 的所有位置
grep -n "bias=False" src/transformers/models/t5/modeling_t5.py        # 共 14 行命中
grep -c "bias=False" src/transformers/models/t5/modeling_t5.py        # → 14
#   → 真正的线性层：78, 79, 100, 101, 102, 206, 207, 208, 209, 924（10 处）
#   → 另有 4 个是 `has_relative_attention_bias=False` 的子串命中：180, 373, 408, 436

# 11. T5 没有重写 prepare_inputs_for_generation
grep -c "prepare_inputs_for_generation" src/transformers/models/t5/modeling_t5.py
#   → 0（用的是 generation/utils.py:519 的通用实现）

# 12. EncoderDecoderCache 的创建与用点
grep -n "EncoderDecoderCache\|is_updated" src/transformers/models/t5/modeling_t5.py
#   →  26（import）、307、308、309、319、331、332、687（创建）、703

# 13. T5 用不了 FlashAttention
grep -n "_supports_flash_attn\|_supports_sdpa\|_keep_in_fp32_modules" src/transformers/models/t5/modeling_t5.py
#   → 537:    _keep_in_fp32_modules = ["wo"]
#   → 540:    _supports_flash_attn = False
#   → 542:    _supports_sdpa = True

# 14. position_bias 折进 SDPA 的 mask；is_causal 从模块属性上读
grep -n "def create_position_bias_mask" -A 3 src/transformers/integrations/sdpa_attention.py
#   →  39:def create_position_bias_mask(
#   →  42:    is_causal: bool,
grep -rn 'module, "is_causal"' src/transformers/
#   → src/transformers/integrations/sdpa_attention.py:108（全库唯一一处）

# 15. 没有 modular，但被 90 处 # Copied from 引用（物理复制，不是继承）
for f in $(grep -rl "Copied from transformers.models.t5.modeling_t5" src/transformers/models/); do
    printf "%4d  %s\n" $(grep -c "Copied from transformers.models.t5.modeling_t5" $f) $f; done | sort -rn
#   → 30 mt5 / 19 umt5 / 12 longt5 / 11 pop2piano / 9 udop / 8 pix2struct / 1 kosmos2_5
wc -l src/transformers/models/mt5/modeling_mt5.py src/transformers/models/udop/modeling_udop.py
#   → 1596 modeling_mt5.py / 1679 modeling_udop.py（T5 本体是 1518 行）

# 16. modeling_t5.py 里的 2 处 "Copied from" 是 docstring 散文，不是同步标记
grep -n "Copied from" src/transformers/models/t5/modeling_t5.py
#   → 1209、1431（都在注释/文档串里；对照 2A5 的正则 check_copies.py:541，匹配不上）
```

**没能在本机实测的部分（如实说明）**：本机没有任何装了 torch 的 Python 环境（`python -c "import torch"` → `ModuleNotFoundError`），所以**没有真正实例化模型、没有跑前向**。第 3、6、7 条里「只在第 0 层算」的结论来自**代码路径的静态推导**（`:648` 只给第 0 层开关 → `:211` 只给它建表 → `:337` 只对它调用 `compute_bias` → `:740` 只更新不重算），四条证据互相印证；分桶表是**用 numpy 逐行复刻 `:238-262`** 算出来的，不是调用原函数得到的。

---

## 导航

- 路线图：[00_roadmap.md](./00_roadmap.md)（本篇 = 进阶 · 模型结构 · 2M4）
- 上一篇：[2M3_qwen3_moe.md](./2M3_qwen3_moe.md)
- 下一篇：[2M5_mamba2_ssm.md](./2M5_mamba2_ssm.md)
- **建议紧接着读**：[2A3_kv_cache.md](./2A3_kv_cache.md)——把本篇第八节的 `EncoderDecoderCache` 放回 `CacheLayerMixin` / `DynamicLayer` / `StaticLayer` 的体系里看（2A3 §5.3 就是讲它，并且明确点名了本篇）；再看 [1M5_bert_modernbert.md](./1M5_bert_modernbert.md) §2.4，用「Post-Norm 的真身长什么样」对照本篇第六节的修正。
