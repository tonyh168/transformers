# T5 精读：Encoder-Decoder 架构

> 阅读目标：搞清楚**编码器-解码器**这种形状和 decoder-only 到底差在哪，
> 以及 T5 那三个「看代码才会发现」的设计细节：
> **相对位置偏置的分桶**、**`T5Attention` 一个类演三种角色**、**`EncoderDecoderCache` 的双 cache**。
>
> **这篇在进阶里的位置**：1M1 给了你 decoder-only 的骨架，1M5 给了 encoder 的对照面，
> 而这一篇给出「两个 stack 拼起来怎么跑」——它是**唯一一个输出序列和输入序列完全独立**的主流结构族。
> 读完你会明白：为什么 `generate()` 在 encoder-decoder 上要**先跑一遍 encoder 再进循环**。
>
> 源文件：`src/transformers/models/t5/modeling_t5.py`（**1518 行，无 `modular` 文件，直读生成文件**）、
> `src/transformers/models/t5/configuration_t5.py`（98 行）。
> 顺带对照：`cache_utils.py`（`EncoderDecoderCache`，:1947）、`generation/utils.py`、
> `integrations/sdpa_attention.py`。

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

- [1M1_llama_structure.md](./1M1_llama_structure.md)：decoder-only 骨架（RMSNorm / RoPE / SwiGLU / Pre-Norm / KV cache），本篇全程拿它当对照面
- [1M5_bert_modernbert.md](./1M5_bert_modernbert.md)：encoder 对照面（双向注意力 / 绝对位置编码 / 真 Post-Norm）——；其 §2.4 用同一批 `modeling_bert.py` 行号论证了「BERT 才是 Post-Norm」，正好和本篇第六节的修正互为正反面
- [2A5_modular_code_generation.md](./2A5_modular_code_generation.md)：决定「读哪个文件」的那篇；本篇第二节是它的一个实例
- [1A5_generate_loop.md](./1A5_generate_loop.md)：`generate` 主循环（本篇第九节讲 encoder-decoder 的差异）
- [2A3_kv_cache.md](./2A3_kv_cache.md)：KV Cache 体系；`EncoderDecoderCache` 的父级上下文（它那边也标了「呼应 2M4 T5」）

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
| ① self-attention（encoder） | 输入序列 | 输入序列 | **双向**（每个 token 看得见所有 token） | `T5LayerSelfAttention` `:372-380` |
| ① self-attention（decoder） | 已生成的部分 | 已生成的部分 | **因果**（只能看左边） | 同上，`is_causal=config.is_decoder` `:379` |
| ② cross-attention（只有 decoder 有） | decoder 的当前状态 | **encoder 的输出** | 只 mask padding，不看因果 | `T5LayerCrossAttention` `:404-409` |
| ③ FFN | — | — | — | `T5LayerFF` `:126-141` |

### 1.3 一次 forward 的完整数据流（含 cross-attention 这条线）

```
                       input_ids [B, S]
                             │
        ┌────────────────────▼─────────────────────────────────────────────────────┐
        │ ENCODER = T5Stack(config.is_decoder=False)            modeling_t5.py:640 │
        │                                                                          │
        │   embed_tokens (共享权重, :765/:912)        :678                         │
        │        │                                                                 │
        │   dropout                                   :725                         │
        │        │                                                                 │
        │   ┌────▼──── × num_layers 个 T5Block（只有 ①③，没有 ②）:440-442 ───────┐ │
        │   │   ① self-attention：双向 mask（:706-711 create_bidirectional_mask）│ │
        │   │      └─ 相对位置 bias 只在第 0 层算一次，其余层复用（:648 ➜ :739） │ │
        │   │   ③ FFN (ReLU)                                :498                 │ │
        │   └───────────────────────────────────────────────────────────────────┘ │
        │        │                                                                 │
        │   final_layer_norm → dropout                :744-745                     │
        └────────┼─────────────────────────────────────────────────────────────────┘
                 │
                 │  encoder_hidden_states [B, S, d_model]
                 │  ★ 注意：encoder 只跑一次，之后被 decoder 的每一层反复使用
                 │
        ┌────────▼─────────────────────────────────────────────────────────────────┐
        │ DECODER = T5Stack(config.is_decoder=True)             modeling_t5.py:640 │
        │                                                                          │
        │   decoder_input_ids [B, T] → embed_tokens → dropout      :678 / :725     │
        │        │                                                                 │
        │   ┌────▼──── × num_decoder_layers 个 T5Block（①②③）:443-444 ───────────┐ │
        │   │   ① self-attention：causal mask（:697-705），K/V 写进                │ │
        │   │      self_attention_cache（随步数增长）                              │ │
        │   │   ② cross-attention：Q = decoder 状态（:304），                        │ │
        │   │      K/V = encoder_hidden_states（:318/:325-326），                   │ │
        │   │      写进 cross_attention_cache（第 1 步之后就是常量，:319-322 直接复用）│ │
        │   │   ③ FFN (ReLU)                                                        │ │
        │   └──────────────────────────────────────────────────────────────────────┘ │
        │        │                                                                 │
        │   final_layer_norm :744 → × d_model**-0.5（:1044-1045）                   │
        └────────┼─────────────────────────────────────────────────────────────────┘
                 │
                 ▼
        lm_head（bias=False, :924；与 shared 共享权重, :902-906）
                 │
                 ▼
        logits [B, T, vocab] ──→ CrossEntropyLoss(ignore_index=-100) :1051-1054
```

**读图要点**：图中那条从 encoder 指向 decoder ②的箭头，**不经过 softmax 之前的任何参数**——
encoder 的输出只是一组 K/V，被 `T5Attention` 的 `k` / `v` 两个线性层**投影一次**（不是每层重新编码一遍）。
真正的「两序列交互」只发生在这里。

### 1.4 为什么这种形状适合翻译 / 摘要

翻译和摘要的共同点：**输入和输出是两条不同的序列**，且输出「不完全由输入决定」。

1. **encoder 可以双向读**。翻译「bank」时，需要看完整句才知道是「银行」还是「河岸」。
   decoder-only 的编码路径是因果的，第一个 token 只能看到自己（`modeling_t5.py:706-711` 用的是
   `create_bidirectional_mask`，decoder 用的是 `create_causal_mask`，这就是差别的代码形态）。
2. **decoder 每一步都能看到全部输入**。第 5 个输出 token 依然能直接 attention 到输入的第 1 个 token，
   不需要像 decoder-only 那样「一路靠中间状态传递」。这就是 Bahdanau attention 的原始动机，
   T5 把它结构化成了每一层的第二个子层。
3. **输出长度与输入长度解耦**。`T` 和 `S` 无约束关系（1:1、1:many 都行）。
4. **代价**：encoder 的前向是**额外的一次完整前向**（`generate` 里显式跑，见第九节）；
   而且 encoder-decoder 不能像 decoder-only 那样「把 prompt 和回答塞进同一条序列」。
   今天的大模型集体转向 decoder-only，主要就是后一个原因——**同一条序列可以吃下任意任务、任意模态的 prompt**。

### 1.5 一个容易被忽略的事实：encoder 和 decoder 是同一个类

`T5Stack` 是 encoder 和 decoder **共用的类**（`:640`），靠 `config.is_decoder` 分叉：

```python
# modeling_t5.py:640-651  同一个 T5Stack，两份实例
class T5Stack(T5PreTrainedModel):
    def __init__(self, config):
        ...
        self.is_decoder = config.is_decoder                                   # :645
        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0), layer_idx=i)
             for i in range(config.num_layers)]                               # :648 ← 相对 bias 只在第 0 层
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)   # :650
```

两份实例的差别只有 4 处，全部可查：

| # | 差别 | encoder | decoder | 行号 |
|---|------|---------|---------|------|
| 1 | `is_decoder` | `False` | `True` | `:768` / `:773`（`T5Model` 里 deepcopy config 后改字段） |
| 2 | 层数 | `config.num_layers` | `config.num_decoder_layers`（默认为前者，config:65-67） | `:774` |
| 3 | 有没有 cross-attention 子层 | 没有 | 有 | `:443-444` `if self.is_decoder: self.layer.append(T5LayerCrossAttention(...))` |
| 4 | mask / cache | 双向 mask，`past_key_values` 被强制置 None | 因果 mask，创建 `EncoderDecoderCache` | `:706-711` / `:697-705`、`:684-695` |

另外 encoder 的副本被显式关掉 cache：`encoder_config.use_cache = False`（`:769`、`:916`）。

---

## 二、读法：T5 没有 `modular` 文件

2A5 的第一条实操规则是「先看有没有 `modular_<name>.py`」。T5 的答案是**没有**（实测：该目录下 0 个 modular 文件），
所以：

| 看到 | 读哪个 | T5 的情况 |
|------|--------|-----------|
| 有 `modular_<name>.py` | 读 modular（读的是「差异」） | ✗ |
| 没有 | 直读 `modeling_<name>.py`，见到 `# Copied from` 顺着跳源头 | ✓ **就是它** |

**但 T5 有一个和 Llama 不同的地方**，读之前必须知道：

T5 自己**不抄别人**，但**被别人抄了 90 处**。这些「别人」是**物理复制**（不是继承）：

```bash
$ for f in $(grep -rl "Copied from transformers.models.t5.modeling_t5" src/transformers/models/); do
      printf "%4d  %s\n" $(grep -c "Copied from transformers.models.t5.modeling_t5" $f) $f; done | sort -rn
  30  src/transformers/models/mt5/modeling_mt5.py
  19  src/transformers/models/umt5/modeling_umt5.py
  12  src/transformers/models/longt5/modeling_longt5.py
  11  src/transformers/models/pop2piano/modeling_pop2piano.py
   9  src/transformers/models/udop/modeling_udop.py
   8  src/transformers/models/pix2struct/modeling_pix2struct.py
   1  src/transformers/models/kosmos2_5/modeling_kosmos2_5.py
```

对阅读的意义：

1. **T5 是「源头」，`mt5` / `umt5` / `longt5` 是「复制品」**。`modeling_mt5.py` 有 1596 行、`modeling_udop.py` 有 1679 行，
   里面绝大多数代码和 T5 逐行一致（T5 本体是 1518 行）。**读 T5 一遍 = 读懂了半个 T5 家族**。
2. 反过来，如果你想找某个 T5 类有没有被改过，就 `grep "Copied from transformers.models.t5"`，
   没被标 `# Copied from` 的那部分才是它自己的改动。
3. `modeling_t5.py` 里 `grep -c "Copied from"` 得到 **2**，但那两处（`:1209`、`:1431`）是 **docstring 里的散文**，
   不是真正的同步标记（真正的标记长这样：`# Copied from transformers.models.x.modeling_x.Class`）。
   对照 2A5 的正则 `_re_copy_warning`（`check_copies.py:541`）可以确认：它们匹配不上。
   → **T5 没有任何被同步的代码块，整份文件都是「原始代码」**，可以放心逐行读。

---

## 三、`T5Attention`：一个类的三种身份

这是 T5 源码里最重要的设计之一，也是第一次读最容易迷路的地方：
**encoder self-attention / decoder self-attention / decoder cross-attention 是同一个类 `T5Attention`**（`:176`）。

### 3.1 四个开关

```python
# modeling_t5.py:176-212
class T5Attention(nn.Module):
    def __init__(
        self,
        config: T5Config,
        has_relative_attention_bias=False,     # ← 开关 A：是不是「第 0 层」（构造参数, :180）
        layer_idx: int | None = None,
        is_causal: bool = False,               # ← 开关 B：causal 还是双向（构造参数, :182）
    ):
        super().__init__()
        self.config = config
        self.is_decoder = config.is_decoder    # ← 开关 C：从 config 读（:186）
        self.is_causal = is_causal             # :187
        self.has_relative_attention_bias = has_relative_attention_bias   # :188
        ...
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)     # :206
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)     # :207
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)     # :208
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)     # :209

        if self.has_relative_attention_bias:                             # :211
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)  # :212
```

再加一个 **forward 时才确定的开关**：

```python
# modeling_t5.py:301-302
        # if key_value_states are provided this layer is used as a cross-attention layer for the decoder
        is_cross_attention = key_value_states is not None
```

**「是哪种注意力」不是由类决定的，是由上面 4 个值决定的。** 其中：
- 开关 A / B：**构造时**由上层（`T5Stack` → `T5Block` → `T5LayerSelfAttention`）传进来；
- 开关 C：从 **config** 读，也就是「这份 stack 是 encoder 还是 decoder」；
- 开关 D（`key_value_states`）：**每次 forward** 由 `T5Block` 决定要不要传。

### 3.2 三种角色的真实构造点

```
              T5Attention (modeling_t5.py:176) —— 一个类，三份实例
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ 开关 A  has_relative_attention_bias  :180   ← 只有 block.0 是 True（:648）      │
  │ 开关 B  is_causal                    :182   ← self-attn 才有意义               │
  │ 开关 C  is_decoder = config.is_decoder :186 ← encoder stack / decoder stack    │
  │ 开关 D  key_value_states is not None :302   ← forward 时才知道                  │
  └───────┬─────────────────────┬─────────────────────┬─────────────────────────────┘
          │                     │                     │
 ┌────────▼──────────┐ ┌────────▼──────────┐ ┌────────▼───────────────────┐
 │ encoder self-attn │ │ decoder self-attn │ │ decoder cross-attn         │
 │ 构造 :375-380     │ │ 构造 :375-380     │ │ 构造 :407-409              │
 ├───────────────────┤ ├───────────────────┤ ├────────────────────────────┤
 │ is_decoder = False│ │ is_decoder = True │ │ is_decoder = True          │
 │ is_causal  = False│ │ is_causal  = True │ │ is_causal  = False（:408） │
 │ rel_bias = 第 0 层│ │ rel_bias = 第 0 层│ │ rel_bias 恒为 False（:408）│
 │ K/V ← 自己        │ │ K/V ← 自己        │ │ K/V ← encoder 的输出       │
 │ bucket 双向 :273  │ │ bucket 单向 :273  │ │ position_bias 恒为 0 :337  │
 │ 类内部名 SelfAttention │ self.SelfAttention │ 类内部名 EncDecAttention │
 └───────────────────┘ └───────────────────┘ └────────────────────────────┘
```

构造点的源码（三段，都在 `modeling_t5.py`）：

```python
# ① encoder / decoder 的 self-attention：T5LayerSelfAttention.__init__  :372-380
class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False, layer_idx: int | None = None):
        super().__init__()
        self.SelfAttention = T5Attention(
            config,
            has_relative_attention_bias=has_relative_attention_bias,   # ← 透传「是不是第 0 层」
            layer_idx=layer_idx,
            is_causal=config.is_decoder,                               # ← 开关 B 的来源 :379
        )
```

```python
# ② decoder 的 cross-attention：T5LayerCrossAttention.__init__  :404-409
class T5LayerCrossAttention(nn.Module):
    def __init__(self, config, layer_idx: int | None = None):
        super().__init__()
        self.EncDecAttention = T5Attention(
            config, has_relative_attention_bias=False, layer_idx=layer_idx, is_causal=False   # :408
        )
```

```python
# ③ 谁决定要建 cross-attention 子层：T5Block.__init__  :440-446
        self.layer.append(
            T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias, layer_idx=layer_idx)
        )
        if self.is_decoder:                                                    # :443
            self.layer.append(T5LayerCrossAttention(config, layer_idx=layer_idx))   # :444

        self.layer.append(T5LayerFF(config))                                   # :446
```

**记忆锚点**：`T5Block.layer` 是一个 `ModuleList`，长度 **2（encoder）或 3（decoder）**；
`sublayer[0]` 永远是 self-attn，`sublayer[-1]` 永远是 FF（`T5Block.forward` 里用的就是 `self.layer[0]`、`self.layer[1]`、`self.layer[-1]`，见 `:459`、`:479`、`:498`）。

### 3.3 cross-attention 的 K/V 到底从哪来（关键）

链条一共 4 跳，全部可查：

```
T5ForConditionalGeneration.forward
   hidden_states = encoder_outputs[0]                         :1024
   self.decoder(..., encoder_hidden_states=hidden_states)     :1031-1037
        │
        ▼
T5Stack.forward(encoder_hidden_states=...)                    :667（形参）
   layer_module(..., encoder_hidden_states, ...)               :728-737
        │
        ▼
T5Block.forward(encoder_hidden_states=...)                    :453（形参）
   do_cross_attention = self.is_decoder and encoder_hidden_states is not None   :477
   self.layer[1](..., key_value_states=encoder_hidden_states, ...)              :479-486
        │
        ▼
T5LayerCrossAttention.forward(key_value_states=...)           :416（形参）
   self.EncDecAttention(..., key_value_states=key_value_states, ...)  :423-430
        │
        ▼
T5Attention.forward
   is_cross_attention = key_value_states is not None           :302  ← 判定
   current_states = key_value_states if is_cross_attention else hidden_states   :318  ← ★ K/V 的来源
   key_states   = self.k(current_states)...                    :325
   value_states = self.v(current_states)...                    :326
```

完整片段（这是本篇最该记住的 12 行）：

```python
# modeling_t5.py:301-332
        # if key_value_states are provided this layer is used as a cross-attention layer for the decoder
        is_cross_attention = key_value_states is not None

        query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)   # :304 Q 永远来自自己

        # Check is encoder-decoder model is being used. Otherwise we'll get `DynamicCache`
        is_updated = False
        if isinstance(past_key_values, EncoderDecoderCache):
            is_updated = past_key_values.is_updated.get(self.layer_idx)
            if is_cross_attention:
                curr_past_key_values = past_key_values.cross_attention_cache      # :312
            else:
                curr_past_key_values = past_key_values.self_attention_cache       # :314

        current_states = key_value_states if is_cross_attention else hidden_states    # :318 ★
        if is_cross_attention and past_key_values is not None and is_updated:
            # reuse k,v, cross_attentions
            key_states = curr_past_key_values.layers[self.layer_idx].keys        # :321
            value_states = curr_past_key_values.layers[self.layer_idx].values    # :322
        else:
            kv_shape = (*current_states.shape[:-1], -1, self.key_value_proj_dim)
            key_states = self.k(current_states).view(kv_shape).transpose(1, 2)   # :325 ★
            value_states = self.v(current_states).view(kv_shape).transpose(1, 2) # :326
```

三句话总结：
1. **Q 永远来自 `hidden_states`**（decoder 自己的状态，`:304`），与角色无关。
2. **K/V 来自 `hidden_states` 还是 `key_value_states`，只看 `is_cross_attention`**（`:318`）；
   而 `key_value_states` 一路溯源就是 **encoder 的最终输出**。
3. **cross-attention 的 K/V 只投影一次**（`:319-322` 复用分支）——因为 encoder 输出在生成过程中不变，
   跟自注意力的 K/V（每步都在长）是两种 cache（见第八节）。

### 3.4 顺带一个反直觉点：T5 不做 `1/sqrt(d_k)` 缩放

```python
# modeling_t5.py:196-197
        # T5 folds the relative position bias into the attention scores and does not scale the query/key dot product.
        self.scaling = 1.0
```

对照 Llama：

```python
# modeling_llama.py:226
        self.scaling = self.head_dim**-0.5
```

`scaling` 会被传给注意力实现（T5 `:361`），最终作用于 `attn_weights = matmul(q, k^T) * scaling`（`eager_attention_forward`，`:155-159`）。
**T5 用的是 `1.0`**——原作者把 `1/sqrt(d_k)` 折进了权重初始化（`_init_weights`，`:611-616`：`q` 用 `(d_model*d_kv)**-0.5`，`k`/`v` 用 `d_model**-0.5`），
所以运行时不缩放。这不是笔误，是刻意的。

---

## 四、相对位置偏置（relative position bias）

### 4.1 数学直觉：不是给 token 加位置向量，而是给「距离」学一个 bias

| 方案 | 作用对象 | 加在哪里 | 参数量 | T5 里在哪 |
|------|----------|----------|--------|-----------|
| 绝对位置编码（BERT） | 每个**位置** | 加在**输入 embedding** 上 | `max_position × d_model` | —（BERT: `modeling_bert.py:59`） |
| RoPE（Llama） | 每个**位置** | **旋转 Q/K**（不改 logits 形状） | 0 | —（`modeling_llama.py:73-161`） |
| **相对位置偏置（T5）** | 每一对 **(i, j) 的距离** | **加到 attention logits 上** | `num_buckets × num_heads`（32×8=256） | `:212` / `:277` |

T5 的做法：

```
attention_weights[i, j] = softmax( Q_i · K_j * scaling(1.0) + position_bias[bucket(j - i)] + mask )
                                                                            ^^^^^^^^^^^^^^^^^^^^^
                                                                            每个 (i,j) 一个可学的标量，
                                                                            每一颗头一套（:212 的第二维是 n_heads）
```

「加到 logits 上」而不是「加到 embedding 上」的好处：**它是加性的、per-head 的、且只跟距离有关**。
代价是 logits 的形状从 `[B, H, T, S]` 变成了必须显式构造的 `[1, H, T, S]` 张量——
这也直接导致了 T5 的几个工程后果（不能上 FlashAttention、SDPA 要把它塞进 mask，见 5.4 和第九节）。

### 4.2 `compute_bias` 的真实实现

```python
# modeling_t5.py:264-279
    def compute_bias(self, query_length, key_length, device=None, past_seen_tokens=0):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None] + past_seen_tokens  # :268
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]                      # :269
        relative_position = memory_position - context_position  # shape (query_length, key_length)               # :270
        relative_position_bucket = self._relative_position_bucket(
            relative_position,                        # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),      # :273 ★ encoder 双向、decoder 单向
            num_buckets=self.relative_attention_num_buckets,      # :274
            max_distance=self.relative_attention_max_distance,    # :275
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (q, k, heads)   :277
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, heads, q, k)                 :278
        return values
```

四个细节：
- **`relative_position = memory_position - context_position`**（`:270`）：j 减 i，**往左看的距离是负数**。
  符号是有意义的，因为 `bidirectional` 分支要拿符号分正负两半桶（见 4.3）。
- **`past_seen_tokens`**（`:268`）：增量解码时 query 的位置不是从 0 开始，要加上已经生成的 token 数，
  否则第 10 步的 query 会被当成第 0 个位置。这个值来自 `past_key_values.get_seq_length(self.layer_idx)`（`:297-299`）。
- **`bidirectional=(not self.is_decoder)`**（`:273`）：注意这里看的是 **config**，不是 `key_value_states`。
  所以 **cross-attention 这一项根本没机会生效**——它在 `:337` 就被短路了（见 4.6）。
- **输出形状是 `[1, H, q, k]`**（`:278`，batch 维是 1，广播到所有 batch）。

### 4.3 分桶：`num_buckets` 与 `max_distance` 怎么用

```python
# modeling_t5.py:238-262   T5Attention._relative_position_bucket（静态方法，:216）
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2                                        # :240 一半给正、一半给负
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets   # :241 正距离偏移半区
            relative_position = torch.abs(relative_position)         # :242
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))  # :244
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2                                 # :248  近端：精确桶
        is_small = relative_position < max_exact                     # :249

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (                  # :252  远端：对数桶
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)   # :257-259 钳到最后一桶
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)   # :261
        return relative_buckets
```

**为什么分桶**：附近的位置差别是「语法」级别的，需要精确（i 和 i+1 的关系 ≠ i 和 i+2）；
远处的差别是「话题」级别的，粗一点无所谓。于是：**一半桶给近距离、一桶一距离；另一半桶给远距离、按 log 缩放。
超过 `max_distance` 的全部压进最后一个桶**（`:257-259`，docstring `:226` 说得很清楚：
"All relative positions >=max_distance map to the same bucket"）。

`num_buckets=32`（config:51）、`max_distance=128`（config:52）的默认值下：

```
 相对距离 d = memory_position - query_position   (:270，负数 = 往左看)

 ENCODER（bidirectional=True, :273）—— 正负各占一半桶，共 32 桶
   d:  ...... -128 ←────── -8 .. -1 │ 0 │ 1 .. 7 ──────→ 128 ......
       └── 8 个对数桶 ──────────┘└8个精确桶┘ └8个精确桶┘└── 8 个对数桶 ──┘
                桶 8..15              桶 0..7      桶 16..23          桶 24..31
                                     ↑ :241 正距离统一 +num_buckets/2（=16）的偏移

 DECODER（bidirectional=False, :244）—— 只有 d<=0 合法，正距离一律压成 0
   d:  ...... -128 ←──────────── 16 .. 1 │ 0 │ (d>0 → 全部当 0)
       └── 16 个对数桶 ─────────────────┘└─ 16 个精确桶 ─┘
                桶 16..31                    桶 0..15
```

注意 encoder 有「8 个精确桶」而 decoder 有「16 个精确桶」——因为 `num_buckets` 在双向时被先砍半（`:240`）。
**decoder 的近端分辨率是 encoder 的两倍**，这是 `num_buckets //= 2` 的一个不易察觉的后果。

### 4.4 真实桶表（把式子跑一遍）

本机没有 torch，所以下面这张表是用 numpy **逐行复刻 `:238-262` 的公式**算出来的取值
（脚本见验证清单第 7 条；复刻的唯一差别是 dtype，桶号完全一致）：

| | encoder（`bidirectional=True`） | decoder（`bidirectional=False`） |
|---|---|---|
| 距离 → 桶 | `0→0, 1→1, ... 7→7` | `0→0, 1→1, ... 15→15` |
| | `8-11→8, 12-15→9, 16-22→10, 23-31→11` | `16-18→16, 19-20→17, 21-23→18, 24-26→19` |
| | `32-45→12, 46-63→13, 64-90→14` | `27-30→20, 31-34→21, 35-39→22, 40-45→23` |
| | `91 及以上 → 15`（含 ≥128 全部） | `46-51→24 ... 99-112→30, 113 及以上 → 31` |
| 正距离 | 同一个表 +16（`:241`） | 一律映射到桶 0（`:244` 把正数钳成 0） |
| 桶号范围 | `[0, 31]` | `[0, 31]` |

（encoder 侧「91 及以上 → 15」是「公式值 + 钳位（`:257-259`）」叠加的结果：
|d| = 91..127 时公式已经给出 15，|d| ≥ 128 时公式给 16 及以上，被钳到 15。）

一个小坑：`torch.log(0)` 是 `-inf`，`.to(torch.long)` 在 `-inf` 上是未定义值——
但 `d=0` 时 `is_small` 必为 True，`:261` 的 `torch.where` 会把那个垃圾值丢掉。
（numpy 复刻版会打印 `divide by zero` 警告，但最终输出与上表一致，原因就是这个 `where`。）

### 4.5 ★ 相对 bias 只在第 0 层**计算**，其余层**复用**

这是本篇要求「务必实测」的点。**结论：是的，只在第 0 层算。**证据有 4 条，互相独立：

**证据 1：只有第 0 层拿到开关 A**

```python
# modeling_t5.py:648
            [T5Block(config, has_relative_attention_bias=bool(i == 0), layer_idx=i) for i in range(config.num_layers)]
```

**证据 2：只有第 0 层的 `T5Attention` 会创建那张 embedding 表**

```python
# modeling_t5.py:211-212
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
```

→ 参数只存在于 `block.0.layer.0.SelfAttention.relative_attention_bias.weight`。
（旁证：加载器要忽略 `decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight` 这个历史 key，
见 `:756` 和 `:899-901`——说明**老 checkpoint 的 cross-attention 曾经也有过这张表**，现在的代码不再创建它。）

**证据 3：`compute_bias` 只在 `has_relative_attention_bias` 为真时被调用**

```python
# modeling_t5.py:335-348
        if position_bias is None:                                   # :335 ← 别人传了就不算
            key_length = key_states.shape[-2]
            if not self.has_relative_attention_bias:                # :337
                position_bias = torch.zeros(
                    (1, self.n_heads, input_shape[1], key_length),  # :338-342 全 0
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(
                    input_shape[1], key_length, device=query_states.device, past_seen_tokens=past_seen_tokens
                )                                                   # :346-348 ← 真正算的地方
```

**证据 4：`T5Stack.forward` 把第 0 层的结果**当成参数**传给后面所有层**

```python
# modeling_t5.py:722-742
        position_bias = None                       # :722  进循环前是 None
        encoder_decoder_position_bias = None       # :723
        ...
        for layer_module in self.block:            # :727
            hidden_states, self_attention_position_bias, cross_attention_position_bias = layer_module(
                hidden_states,
                attention_mask,
                position_bias,                     # :731  ← 上一层的产物直接喂给下一层（位置参数！）
                encoder_hidden_states,
                encoder_extended_attention_mask,
                encoder_decoder_position_bias,     # :734  "as a positional argument for gradient checkpointing"
                past_key_values=past_key_values,
                **kwargs,
            )

            # We share the position biases between the layers - the first layer stores them
            position_bias = self_attention_position_bias                        # :740 ★ 只更新，不重算
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = cross_attention_position_bias   # :742
```

于是第 1..N-1 层的 `position_bias` 进来时**不是 None**（`:335` 的分支直接跳过），
它们既不查表也不算分桶，**只是把第 0 层算好的 `[1, H, q, k]` 张量原样加到自己的 logits 上**
（`eager_attention_forward` `:161-162`）。这叫「**算一次、用 N 次**」。

**为什么可以这么省**：相对 bias 只依赖 `(i, j, head)`，跟层无关、跟 hidden_states 无关。
这是个「在模型里唯一」的量，`compute_bias` 里也**没有可学参数参与前向**（只是查 `nn.Embedding`）。

**两个必须记住的推论**：

1. **相对 bias 有 3 个开关的摆放是「外部约定」而不是类内保证**：
   `T5Attention.forward` 的单层语义是「position_bias 为 None 且我没有表 → 返回全 0」。
   也就是说，如果你**单独**调 `decoder.block[1]` 且不传 `position_bias`，它会静默地失去相对位置信息。
   相对 bias 的正确性依赖 `T5Stack` 的**顺序遍历**（因此这个循环不能乱序、不能并行）。
2. `T5Stack.forward` 里 `:731` 和 `:734` 是**位置参数**（不是关键字参数），
   注释也写了原因（`gradient checkpointing`）。改动 `T5Block.forward` 的形参顺序会静默破坏这个传递。

### 4.6 cross-attention 的 position_bias 恒为 0

`T5LayerCrossAttention` 用 `has_relative_attention_bias=False` 构造（`:408`）
→ 走到 `:337` 的 `if not self.has_relative_attention_bias:` 分支 → `:338-342` 生成一个**全 0** 张量。

所以 **cross-attention 是纯内容注意力，不带任何位置信息**：
decoder 的第 t 个 query 对 encoder 的每个位置一视同仁（只看内容 + padding mask）。
`encoder_decoder_position_bias` 这个名字（`:455`、`:723`、`:742`）容易让人误以为它里面有东西，实际恒为 0。

### 4.7 位置编码三方对比

| | 1M1 · Llama（RoPE） | 1M5 · BERT（绝对位置编码） | **本篇 · T5（相对位置偏置）** |
|---|---|---|---|
| 载体 | 旋转 Q/K 向量 | 加在输入 embedding 上 | 加在 **attention logits** 上 |
| 参数 | 0 | `max_position_embeddings × d_model`（bert:59） | `num_buckets × n_heads`（t5:212），**默认 32×8** |
| 作用范围 | 每层、每次前向都算 | 只在 embedding 层加一次 | **只在第 0 层算一次**，全层复用（t5:740） |
| 外推性 | 好（可插值/缩放） | 差（超过 `max_position` 直接没有 embedding） | 好（远处都在最后一个桶里，docstring t5:226-227 明确说是为了外推） |
| 位置粒度 | 每个位置精确编码 | 每个位置一个向量 | **距离分桶**：近处精确、远处对数 |
| 加在 mask 里？ | 否 | 否 | **是**（这就是 T5 用不了 FlashAttention 的原因，t5:539-540） |
| 代码位置 | `modeling_llama.py:73-161` | `modeling_bert.py:59`、`:103` | `modeling_t5.py:216-279` |

---

## 五、`T5LayerNorm`

```python
# modeling_t5.py:50-72
class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))     # :56 只有 weight，没有 bias
        self.variance_epsilon = eps                             # :57

    def forward(self, hidden_states):
        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://huggingface.co/papers/1910.07467 thus variance is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32                       # :60-63

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)   # :65 不减均值、fp32
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)  # :66

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:   # :69
            hidden_states = hidden_states.to(self.weight.dtype)    # :70

        return self.weight * hidden_states                         # :72
```

三方对比（实测三份源码）：

| | `T5LayerNorm` | `LlamaRMSNorm` | `BertLayerNorm`（= `nn.LayerNorm`） |
|---|---|---|---|
| 减均值 | ❌（`:65` 只有 `pow(2).mean(-1)`） | ❌（`llama:65` 同式） | ✅（`nn.LayerNorm` 标准实现） |
| bias | ❌（`:56` 只有 `weight`） | ❌（`llama:59`） | ✅（`bias=True` 默认） |
| 精度 | 先转 fp32 算，再**按条件**转回（`:65-70`） | 先转 fp32 算，**无条件**转回输入 dtype（`llama:63-67`） | 按 autocast 走 |
| extra_repr | 无 | 有（`llama:69-70`） | 有 |
| 定义行 | `:50-72` | `llama:53-67` | `bert:62` 等处 |
| eps 默认 | `layer_norm_epsilon=1e-6`（config:54） | `rms_norm_eps=1e-6` | `layer_norm_eps=1e-12` |

**两者数学上等价——这一点不用猜，Llama 的 docstring 自己写了**：

```python
# modeling_llama.py:53-57
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
```

那 T5 这份「历史实现方式不同」在哪？**唯一实质差别是第 69-70 行的条件转换**：
T5 只在 `weight` 是 fp16/bf16 时才把结果转回低精度，否则保持 fp32；
Llama 是拿到输入 dtype 就无条件转回去。功能上二者在正常路径下一致，
但 T5 的写法在「weight 是 fp32、输入是 fp16」的混合精度场景下会保留 fp32 输出（老代码的兼容痕迹）。

顺带一句：T5 的 `_init_weights` 把 LayerNorm 的 weight 初始化为 `factor * 1.0`（`:567-568`），
和 RMSNorm 的「初始化为全 1」是同一件事。

---

## 六、`T5Block` 的 norm 摆放（以及对路线图的一处修正）

### 6.1 先看代码

```python
# modeling_t5.py:384-401  T5LayerSelfAttention.forward
    def forward(self, hidden_states, attention_mask=None, position_bias=None, past_key_values=None, **kwargs):
        normed_hidden_states = self.layer_norm(hidden_states)          # :392  ① 先归一化
        attention_output, position_bias, attn_weights = self.SelfAttention(
            normed_hidden_states,                                     # :394  ② 再进子层
            mask=attention_mask, position_bias=position_bias, past_key_values=past_key_values, **kwargs,
        )
        attention_output = hidden_states + self.dropout(attention_output)   # :400 ③ 残差加的是「未归一化」的输入
        return attention_output, position_bias, attn_weights
```

```python
# modeling_t5.py:137-141  T5LayerFF.forward
    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)      # :138
        forwarded_states = self.DenseReluDense(forwarded_states)  # :139
        hidden_states = hidden_states + self.dropout(forwarded_states)  # :140
        return hidden_states
```

而 **`T5Block.forward` 里一个 norm 都没有**（`:448-509` 通读一遍：只有 `self.layer[0]` / `self.layer[1]` / `self.layer[-1]`
和四段 fp16 `clamp`）。norm 被藏在**每个子层模块内部**。

### 6.2 修正：这是 **Pre-Norm 的摆放**，不是 Post-Norm

路线图（`00_roadmap.md`）在 2M4 的描述里写的是「Post-Norm」，**实测代码不支持这个说法**：

- **Post-Norm 的定义**（BERT 原版）：`x = LayerNorm(x + Sublayer(x))`——先加残差，再归一化。
  在 `modeling_bert.py` 里就是：
  ```python
  # modeling_bert.py:290-292  BertSelfOutput
          hidden_states = self.dense(hidden_states)
          hidden_states = self.dropout(hidden_states)
          hidden_states = self.LayerNorm(hidden_states + input_tensor)   # :292 ★ 先加后归一化
  # modeling_bert.py:348-350  BertOutput
          hidden_states = self.dense(hidden_states)
          hidden_states = self.dropout(hidden_states)
          hidden_states = self.LayerNorm(hidden_states + input_tensor)   # :350 ★
  ```
- **Pre-Norm 的定义**：`x = x + Sublayer(LayerNorm(x))`——归一化在**输入侧**，残差是干净的通路。
  T5 的两段代码（`:392-400`、`:138-140`）**正是这个形状**；对照 Llama：
  ```python
  # modeling_llama.py:305-317  （1M1 里读过的那段）
          residual = hidden_states
          hidden_states = self.input_layernorm(hidden_states)   # :306
          hidden_states, _ = self.self_attn(hidden_states=hidden_states, ...)
          hidden_states = residual + hidden_states              # :317
  ```

**为什么容易看错**：`T5Block.forward` 把 norm 藏起来了，你只看到 `hidden_states + dropout(attention_output)`（`:400`）这种
「残差 + 某个东西」的形状，就会误以为那个「某个东西」是被归一化过的输出。
实际上它是 `Sublayer(LayerNorm(x))`。**T5 的 norm 摆放 = Llama 的 Pre-Norm，只是书写位置在子层模块内部（更接近原始 Mesh 代码的写法）。**

> 备注：如果你在别处看到「T5 是 post-norm」，它大概率是把「norm 写在子层里」这件事误读了。
> 本篇的判据是可复核的：看 `self.layer_norm(...)` 出现在残差相加的**之前**（输入侧）还是**之后**（输出侧）。

### 6.3 T5Block 里两个（或三个）子层的 norm 位置一览

| 子层 | 自己的 norm | 归一化对象 | 残差相加 | 行号 |
|------|-------------|-----------|----------|------|
| ① self-attn | `self.layer_norm`（`:381`） | 子层输入 | `hidden_states + dropout(attn_out)`（`:400`） | `:381`、`:392`、`:400` |
| ② cross-attn（仅 decoder） | `self.layer_norm`（`:410`） | 子层输入 | `hidden_states + dropout(attn_out)`（`:431`） | `:410`、`:422`、`:431` |
| ③ FFN | `self.layer_norm`（`:134`） | 子层输入 | `hidden_states + dropout(ff_out)`（`:140`） | `:134`、`:138`、`:140` |
| （stack 末尾） | `self.final_layer_norm`（`:650`） | 最后一层的输出 | — | `:744` |

和 Llama 的另一处结构差异：**T5 在残差分支里多一个 dropout**（`:400`、`:431`、`:140` 以及 stack 级别 `:725`、`:745`），
Llama 全程没有 dropout。T5 是 2019 年的训练配方，dropout 是标配（`dropout_rate=0.1`，config:53）。

---

## 七、T5 的经典设计

### 7.1 无 bias 的线性层

全文件真正的 `bias=False` 线性层有 10 处（`grep -c "bias=False"` 得到 14，
多出来的 4 个是 `has_relative_attention_bias=False` 的子串匹配，在 `:180`/`:373`/`:408`/`:436`），覆盖了所有主干投影：

| 模块 | 行号 | 说明 |
|------|------|------|
| attention 的 q/k/v/o | `:206-209` | 四个投影全部无 bias |
| MLP（非 gated） | `:78`（`wi`）、`:79`（`wo`） | |
| MLP（gated） | `:100-102`（`wi_0`/`wi_1`/`wo`） | |
| `lm_head` | `:924` | 且与 embedding 共享权重（`:902-906`） |

少数例外在任务头上（`T5ClassificationHead.dense` / `out_proj`，`:517-519` 没写 `bias=False`，默认带 bias；
`_init_weights` 里还给它们 `zeros_(bias)`，`:585-589`）。

对照 Llama：Llama 的 `q/k/v/o_proj` 用 `bias=config.attention_bias`（默认 False），
而 MLP 的三个投影是**无参数控制地硬编码 `bias=False`**。T5 更彻底：一句话，全部不要 bias。

### 7.2 MLP：默认是 **ReLU 两层**，但**有** gated-GELU 变体

路线图问「T5 有 gated-GELU 变体吗」——**有，而且是同一个代码文件里的两条分支**，由 `config.feed_forward_proj` 决定：

```python
# configuration_t5.py:56, 69-75
    feed_forward_proj: str = "relu"           # :56  默认 relu
...
        act_info = self.feed_forward_proj.split("-")   # :69
        self.dense_act_fn = act_info[-1]               # :70  "relu" / "gelu_new" / ...
        self.is_gated_act = act_info[0] == "gated"     # :71  "gated-gelu" → True

        # for backwards compatibility
        if self.feed_forward_proj == "gated-gelu":     # :74
            self.dense_act_fn = "gelu_new"             # :75
```

```python
# modeling_t5.py:126-135  T5LayerFF 选择实现
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(config)   # :130
        else:
            self.DenseReluDense = T5DenseActDense(config)        # :132
```

| | `T5DenseActDense`（默认，T5 1.0） | `T5DenseGatedActDense`（T5 v1.1 / `gated-gelu`） | Llama 的 `LlamaMLP` |
|---|---|---|---|
| 定义 | `:75-94` | `:97-123` | `modeling_llama.py:163-176` |
| 线性层 | `wi`（`:78`）、`wo`（`:79`）——**2 个** | `wi_0`、`wi_1`（`:100-101`）、`wo`（`:102`）——**3 个** | `gate_proj`/`up_proj`/`down_proj` |
| 前向 | `wo(dropout(act(wi(x))))`（`:84-93`） | `wo(dropout(act(wi_0(x)) * wi_1(x)))`（`:107-109`） | `down(act(gate(x)) * up(x))` |
| 激活 | `ACT2FN[config.dense_act_fn]`（`:81`），默认 **ReLU** | 默认 **gelu_new**（config:75） | **SiLU** |
| 有无门控 | 无 | **有**（`wi_0` 过激活后与 `wi_1` 逐元素相乘，`:109`） | 有 |

所以准确的说法是：**T5 1.0 用 ReLU 的两层 MLP（`d_ff → d_model`，无门控）；
T5 1.1 用 gated-GELU 的三层 MLP。两条路径都在 `modeling_t5.py` 里，`T5LayerFF` 按 config 选一个。**
「T5 是 ReLU 不是 GELU」这句话对默认配置成立，但要记住它是 **config 驱动的**，不是写死的。

### 7.3 共享 embedding，和那个「scaled」的真相

**共享**是三处权重绑成一个（`T5ForConditionalGeneration._tied_weights_keys`，`:902-906`）：

```python
# modeling_t5.py:902-906
    _tied_weights_keys = {
        "lm_head.weight": "shared.weight",
        "encoder.embed_tokens.weight": "shared.weight",
        "decoder.embed_tokens.weight": "shared.weight",
    }
```

**关于「scaled embedding（`d_model**-0.5`）」——实测结论：当前代码里没有对输入 embedding 做缩放。**
全文件运行时出现的 `**-0.5` 只有一处：

```python
# modeling_t5.py:1042-1047
        sequence_output = decoder_outputs[0]

        if self.config.scale_decoder_outputs:                              # :1044
            sequence_output = sequence_output * (self.model_dim**-0.5)     # :1045
```

（其余 `**-0.5` 全在 `_init_weights` 里，是初始化标准差，`:577-616`。）

正确的说法是：**T5 对 decoder 的输出（进 `lm_head` 之前）乘 `d_model**-0.5`**，
作用是把共享 embedding 的尺度压回来、控制 logits 的量级（原始 Mesh 代码的链接就写在老版本的注释里，见下）。

**这里还藏着一段 config 历史包袱**，值得单独看：

```python
# configuration_t5.py:77-83
        # Super weird feature of T5 because we support T5 and T51.1 from the same
        # model code. Original T5 always scaled outputs, but the 1.1v does not.
        # The model code was relying on saved configs where `tie_word_embeddings` is
        # set to `False` in 1.1v and using it as indicator of whether to scale or not
        # But in fact we tie weights always and force it to be `True`
        self.scale_decoder_outputs = kwargs.pop("tie_word_embeddings", None) is not False   # :82
        self.tie_word_embeddings = True                                                     # :83
```

翻译成人话：
- 老版本的代码是 `if self.config.tie_word_embeddings: sequence_output = sequence_output * (self.model_dim**-0.5)`
  （可实测：`git show v4.40.0:src/transformers/models/t5/modeling_t5.py | grep -n "Rescale output" -B2 -A3`
  → `1765: if self.config.tie_word_embeddings:` / `1768: sequence_output = sequence_output * (self.model_dim**-0.5)`）。
- 而权重**实际上一直都是绑的**，`tie_word_embeddings` 只是被当成「这个 checkpoint 是 T5 1.0 还是 1.1」的标记在用。
- 现在的代码把这两个语义**拆开**：`tie_word_embeddings` 恒为 `True`（`:83`），
  缩放交给新的 `scale_decoder_outputs`（`:82`），并且**从老 config 里把 `tie_word_embeddings` pop 出来当输入**——
  保证老 checkpoint 加载后行为不变。

> 这是一个很好的「读 config 时为什么要看注释」的例子：字段名可能和历史语义完全脱钩。

### 7.4 loss 与 teacher forcing

```python
# modeling_t5.py:1026-1028
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)          # :1028
```

```python
# modeling_t5.py:618-637  _shift_right
    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id   # :619
        pad_token_id = self.config.pad_token_id                       # :620
        if decoder_start_token_id is None:
            raise ValueError(...)                                     # :622-626
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)      # :628
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()      # :629  ★ 整体右移一位
        shifted_input_ids[..., 0] = decoder_start_token_id            # :630  ★ 首位塞 start token
        if pad_token_id is None:
            raise ValueError(...)                                     # :632-633
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)   # :635
        return shifted_input_ids
```

```python
# modeling_t5.py:1049-1054
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)     # :1051  ★ -100 被忽略
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))   # :1054
```

| 机制 | 作用 | 行号 |
|------|------|------|
| `labels` 里 `-100` | 不算 loss（padding / prompt / 不关心的位置） | `:1051` |
| `decoder_input_ids` 右移一位 | teacher forcing：第 t 步的输入是第 t-1 个真实 token | `:629-630` |
| 首位 = `decoder_start_token_id` | T5 里通常等于 `pad_token_id`（见 `:624` 的报错文案） | `:630` |
| `-100` 在右移时被替换成 `pad` | 否则 embedding 查表会越界 | `:635` |
| 只在 `decoder_input_ids is None` 时右移 | **你自己传了 `decoder_input_ids` 就不会右移**——传错就得自己负责 | `:1026` |

注意 `decoder_start_token_id` **不是 `T5Config` 的声明字段**（`configuration_t5.py:44-62` 里没有它），
它是 `GenerationConfig` 的字段（`generation/configuration_utils.py:455`），
并且 `PreTrainedConfig` 允许把未声明的 key 直接 `setattr`（`configuration_utils.py:339-343`）——
T5 的 checkpoint config.json 里就带着它（测试里也是显式传的：`tests/models/t5/test_modeling_t5.py:79`）。
`T5ForQuestionAnswering` 之外，`_shift_right` 还被 `prepare_decoder_input_ids_from_labels` 复用（`:1068-1069`）。

---

## 八、`EncoderDecoderCache`（呼应 2A3）

### 8.1 为什么需要两个 cache

自注意力的 K/V 每步**长 1**（长度 = 已生成 token 数）；
cross-attention 的 K/V 在第 1 步**一次写满**（长度 = encoder 序列长度 S），之后永不变化。

放进同一个 cache 会出两个问题：
1. **长度语义冲突**：「当前序列长度」到底是 `T` 还是 `S`？`get_seq_length()` 无法回答。
2. **白白重算**：每步都会把 encoder 的输出重新投影一遍 K/V（S 个位置的两次矩阵乘），完全是浪费。

所以设计成两个独立的 cache，装在一个壳里：

```python
# cache_utils.py:1947-1958
class EncoderDecoderCache(Cache):
    """
    Base, abstract class for all encoder-decoder caches. Can be used to hold combinations of self-attention and
    cross-attention caches.
    ...
        caches (`Iterable`):
            Usually an iterable of length 2, containing 2 `Cache` objects, the first one for self-attention, the
            second one for cross-attention. ...
    """
```

```python
# cache_utils.py:1996-2008
        # Otherwise, we should get two arguments, a self-attention cache and a cross-attention cache
        elif len(caches) == 2:
            if not isinstance(caches[0], Cache) or not isinstance(caches[1], Cache):
                raise TypeError(f"One of the two arguments is not a Cache: {type(caches[0]) = }, {type(caches[1]) = }")
            self.self_attention_cache = caches[0]        # :2000
            self.cross_attention_cache = caches[1]       # :2001
        ...
        self.is_updated = {}                             # :2006 ★ 每个 decoder 层的 cross K/V 写没写过
        for layer_idx in range(len(self.cross_attention_cache)):
            self.is_updated[layer_idx] = bool(self.cross_attention_cache.get_seq_length(layer_idx) > 0)   # :2008
```

**在 T5 里它是被谁创建的**（decoder stack 自建，encoder stack 自毁）：

```python
# modeling_t5.py:684-695
        if self.is_decoder:
            if use_cache and past_key_values is None:
                if self.config.is_encoder_decoder:
                    past_key_values = EncoderDecoderCache(
                        DynamicCache(config=self.config), DynamicCache(config=self.config)
                    )                                                    # :687-689
                else:
                    past_key_values = DynamicCache(config=self.config)   # :691 纯 decoder 的 stack
        elif not self.is_decoder:
            # do not pass cache object down the line for encoder stack
            # it messes indexing later in decoder-stack because cache object is modified in-place
            past_key_values = None                                           # :695 ★ encoder 强制不用 cache
```

### 8.2 cache 是怎么被路由的

`T5Attention.forward` 里的那 12 行（`:307-316`、`:319-322`、`:328-332`）：

| 步骤 | 代码 | 含义 |
|------|------|------|
| 取出 `is_updated[layer_idx]` | `:309` | 这一层的 cross K/V 是否已经写过 |
| 选 cache | `:310-314` | cross-attn → `cross_attention_cache`；self-attn → `self_attention_cache` |
| 复用 | `:319-322` | cross + 已写过 → **直接读 cache 里的 K/V，不做投影** |
| 写入 | `:328-332` | 否则投影后写入，并给 cross-attn 置 `is_updated=True`（`:332`） |

**一个 t5 特有的细节**：`:321-322` 直接读 `curr_past_key_values.layers[self.layer_idx].keys`，
而 self-attn 走的是正常的 `update()` 路径（`:329`）。也就是说 **cross cache 在 T5 里只被写一次**，
这个「写一次」的语义由 `is_updated` 这个 per-layer 字典承载。

`EncoderDecoderCache` 其余被代理的方法（`cache_utils.py`）：`__len__` `:2021-2026`（= 层数）、
`get_seq_length` `:2028-2030`（**代理给 self cache**——生成长度只由 decoder 决定）、
`reset` `:2036-2040`（两个都清，`is_updated` 全置 False）、`reorder_cache` `:2042-2045`（beam search 时两个都要重排）、
`crop` `:2058-2063`（**只裁 self cache**——`:2063` 只调了 `self_attention_cache.crop`）、
`batch_repeat_interleave` :2065-2069 / `batch_select_indices` :2071-2075（这两个两层都要动）、
`get_mask_sizes` `:2077-2078`、`is_sliding` `:2081-2082`、`is_compileable` `:2085-2086`。
`__iter__`（`:2010-2013`）会把两层 cache 拼成 6 元组（为 torch DDP 兼容）。

---

## 九、`generate()` 里的 encoder-decoder 差异（呼应 1A5）

### 9.1 大图：encoder 在循环外跑一次

```
 generate()
   │
   ├─ ① _prepare_model_inputs → _prepare_encoder_decoder_kwargs_for_generation
   │      generation/utils.py:2569-2573 → :826-846
   │      ★ 在这里把 encoder 跑完，结果塞进 model_kwargs["encoder_outputs"]  (:846)
   │      ★ 顺手过滤掉 decoder_* / cross_attn / use_cache / past_key_values  (:826-832)
   │
   ├─ ② _prepare_decoder_input_ids_for_generation   (:850-898)
   │      起点 = decoder_start_token_id（:2066，BC 时回退到 bos，:2068-2072）
   │      用户传了 decoder_input_ids 但不以 start token 开头 → 自动补一个（:895-898）
   │
   └─ ③ 逐 token 循环
          每次取 model_inputs = prepare_inputs_for_generation(...)   (:519-643)
                ★ input_ids_key      = "decoder_input_ids"       (:541)
                ★ position_ids_key   = "decoder_position_ids"    (:559)
                ★ 原始 attention_mask 仍然叫 attention_mask（喂给 cross-attn）(:575-578, :605-606)
                ★ decoder 自己的 mask 改叫 decoder_attention_mask  (:576-578)
                ★ encoder_outputs 走「通用 kwargs 透传」那条路（:608-621）
          调用 model(**model_inputs) → T5ForConditionalGeneration.forward
                encoder_outputs is not None → 跳过 encoder（:1009-1022）
```

### 9.2 T5 **没有**重写 `prepare_inputs_for_generation`

实测：`grep -c "prepare_inputs_for_generation" src/transformers/models/t5/modeling_t5.py` → **0**。

也就是说 T5 完全吃 `GenerationMixin` 的通用实现（`generation/utils.py:519`），
靠 `self.config.is_encoder_decoder` 分叉（`:541`、`:543`、`:559`、`:575-578`）。
这和 1A5 里读到的「模型可以覆盖它」不冲突——**T5 选择不覆盖**，是「配置驱动」而不是「模型自实现」。

关键的两行：

```python
# generation/utils.py:541-543
        input_ids_key = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step for every prompt.
        if not self.config.is_encoder_decoder and inputs_embeds is not None and is_first_iteration:
```

```python
# generation/utils.py:575-578
        encoder_attention_mask = attention_mask if self.config.is_encoder_decoder else None
        attention_mask_key = "decoder_attention_mask" if self.config.is_encoder_decoder else "attention_mask"
        attention_mask = (
            kwargs.pop("decoder_attention_mask", None) if self.config.is_encoder_decoder else attention_mask
        )
```

**注意这里有个命名陷阱**：对 encoder-decoder 来说，
- 进来的 `attention_mask` 参数（原始那一条，长度 S）→ 会被当成 **cross-attention 的 mask**，
  并在 `:605-606` 重新以 `attention_mask` 的名字塞进 `model_inputs`；
- 而 `input_ids`（此刻已经是 `decoder_input_ids`，长度 T）对应的 mask → 改名 `decoder_attention_mask`。

### 9.3 `encoder_outputs` 参数的作用

| 场景 | 代码 | 行为 |
|------|------|------|
| `generate` 内部 | `generation/utils.py:846` | encoder 跑一次，结果存进 `model_kwargs["encoder_outputs"]` |
| beam search 复制 batch | `:934-936` | `encoder_outputs` 必须存在，且要按 beam 数 `repeat_interleave` |
| 每步 forward | `modeling_t5.py:1009-1022` | 非 None → **跳过 encoder**，直接 `hidden_states = encoder_outputs[0]`（`:1024`） |
| 用户手传 tuple | `:1017-1022` | 会被重新包成 `BaseModelOutput`（`last_hidden_state`/`hidden_states`/`attentions`） |
| 显式传参（不用 `generate`） | `T5ForConditionalGeneration.forward` 形参 `:945` | 想省一次 encoder 前向就自己传 |
| 静态 cache / 编译 | `generation/utils.py:1884` | 用 `encoder_outputs[0].shape[1]`（= S）当作 `max_cache_len` 的依据 |

**这就是 `generate()` 在 encoder-decoder 上「先跑 encoder」的落地方式**：
它不是循环里的特殊分支，而是「循环开始前把结果塞进 kwargs，让 forward 自己发现已经算过了」。

### 9.4 一个工程后果：T5 用不了 FlashAttention

```python
# modeling_t5.py:536-543
    _no_split_modules = ["T5Block"]
    _keep_in_fp32_modules = ["wo"]

    # Using position bias API for attention; not yet supported for FA
    _supports_flash_attn = False          # :540 ★
    _supports_flex_attn = True            # :541
    _supports_sdpa = True                 # :542
    _can_compile_fullgraph = True         # :543
```

原因就是第四节的 position bias：**它必须被加到 logits 上，而 FlashAttention 的 kernel 不接受任意加性 bias。**
SDPA 能支持，是因为它接受一个浮点 mask——框架把 `position_bias` **折进 mask** 里：

```python
# integrations/sdpa_attention.py:39-76（节选）
def create_position_bias_mask(position_bias, attention_mask, is_causal, query, key):
    """
    ... The mask contains the values of `position_bias` to positions where we should
    attend to tokens, and -inf where we should not. It will be added to the QK^T result in the attention, before the softmax.
    Note that using such a mask will usually prevent sdpa from dispatching to the most efficient kernel implementations.
    """
    ...
        else:
            position_bias_mask = position_bias        # :68  非 causal → position_bias 直接当加性 mask
    ...
        position_bias_mask = torch.where(attention_mask, position_bias, min_dtype)   # :71
```

而 `is_causal` 从哪来？**从模块属性上读**：

```python
# integrations/sdpa_attention.py:107-108
    # Instead of relying on the value set in the module directly, we use the is_causal passed in kwargs if it is presented
    is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
```

——这就是 `T5Attention` 里那个 `self.is_causal`（`:187`）在**库内唯一被读到的地方**
（`grep -rn 'module, "is_causal"' src/transformers/` 只有 sdpa_attention.py:108 这一条命中；
在 `modeling_t5.py` 内部它自己从不读它，全文只出现 `:182`/`:187`/`:379`/`:408` 四次）。2A1（注意力后端）会把这条路讲完，这里只需要知道闭环在哪。

---

## 三种架构横向对比（decoder-only / encoder-only / encoder-decoder）

| | **decoder-only**（1M1 Llama） | **encoder-only**（1M5 BERT） | **encoder-decoder**（本篇 T5） |
|---|---|---|---|
| 序列数 | 1（prompt+response 同序列） | 1（只进不出） | **2**（源序列 / 目标序列） |
| 注意力方向 | 因果（只看左） | **双向** | **encoder 双向 + decoder 因果 + cross** |
| mask 来源 | `create_causal_mask` | `create_bidirectional_mask` | 两者都要（t5:697-720） |
| 位置编码 | RoPE（旋转 Q/K，`llama:73-161`） | 绝对位置加在 embedding（`bert:59`、`:103`） | **相对 bias 加到 logits，分桶，只在第 0 层算**（`t5:216-279`） |
| norm 摆放 | Pre-Norm（`llama:305-317`） | **Post-Norm**（`bert:292`、`:350`） | **Pre-Norm**，但写在子层内部（`t5:392-400`、`:138-140`） |
| norm 类型 | RMSNorm（`llama:53`） | `nn.LayerNorm`（有 bias、减均值） | RMSNorm 等价物 `T5LayerNorm`（`t5:50-72`） |
| MLP 激活 | SwiGLU 门控（`llama:163-176`） | GELU 两层 | 默认 **ReLU 两层**（`t5:75-94`），另有 gated-GELU 变体（`t5:97-123`） |
| 线性层 bias | 可选（默认 False） | 有 bias | **全部 `bias=False`**（`t5:206-209`、`:78-79`） |
| 权重共享 | 无（未绑定时） | 无 | **`lm_head` + encoder/decoder embedding 三绑一**（`t5:902-906`） |
| cache 结构 | 1 个（`DynamicCache`/`StaticCache`） | 无 | **2 个**（self + cross，`EncoderDecoderCache` cache_utils:1947） |
| 典型任务 | 通用对话 / 生成 | 分类 / MLM / 检索 | 翻译 / 摘要 / 文本到文本 |
| `generate()` | 直接进循环 | 不适用 | **先跑一次 encoder，结果进 kwargs**（gen/utils:846 → t5:1009-1022） |
| 在后端支持上 | 全后端 | 全后端 | **不能用 FlashAttention**（`t5:540`），position bias 要折进 mask（sdpa:39-76） |
| 训练目标 | 下一个 token | MLM / 分类 | 下一个 token（decoder 侧）+ teacher forcing（`t5:1026-1028`） |

**一句话记忆**：encoder-decoder = **encoder-only 的表示 + decoder-only 的自回归 + 一条跨栈的 K/V 通路**。
三个架构共用同一批积木（embedding / attention / FFN / 残差 / norm），差别只在
**mask、位置信息怎么进网络、以及 K/V 从哪来**。

---

## 一句话总结

> **encoder-decoder 的核心是 cross-attention：Q 来自 decoder，K/V 来自 encoder 的输出**（`t5:318`），
> 而「一个 `T5Attention` 演三种角色」靠的是 4 个开关——`has_relative_attention_bias`（`t5:180`）、
> `is_causal`（`t5:182`）、`config.is_decoder`（`t5:186`）、`key_value_states is not None`（`t5:302`）。
> T5 的位置信息不是加在 token 上，而是**给每一对 (i, j) 的距离分桶后学一个 bias，加到 attention logits 上**，
> 并且**只在第 0 层算一次、全层复用**（`t5:648` → `t5:740`）。
> cache 也因此要分两个：self 的每步增长，cross 的一次写满（`cache_utils:1947`）。

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

**没能在本机实测的部分（如实说明）**：本机没有任何装了 torch 的 Python 环境
（`python -c "import torch"` → `ModuleNotFoundError`），所以
**没有真正实例化模型、没有跑前向**。第 3、6、7 条里「只在第 0 层算」的结论来自
**代码路径的静态推导**（`:648` 只给第 0 层开关 → `:211` 只给它建表 → `:337` 只对它调用 `compute_bias` → `:740` 只更新不重算），
四条证据互相印证；分桶表是**用 numpy 逐行复刻 `:238-262`** 算出来的，不是调用原函数得到的。

---

## 导航

- 路线图：[00_roadmap.md](./00_roadmap.md)（本篇 = 进阶 · 模型结构 · 2M4）
- 上一篇：[2M3_qwen3_moe.md](./2M3_qwen3_moe.md)
- 下一篇：[2M5_mamba2_ssm.md](./2M5_mamba2_ssm.md)
- **建议紧接着读**：[2A3_kv_cache.md](./2A3_kv_cache.md)——把本篇第八节的 `EncoderDecoderCache` 放回
  `CacheLayerMixin` / `DynamicLayer` / `StaticLayer` 的体系里看（2A3 §5.3 就是讲它，并且明确点名了本篇）；
  再看 [1M5_bert_modernbert.md](./1M5_bert_modernbert.md) §2.4，
  用「Post-Norm 的真身长什么样」对照本篇第六节的修正。
