# BERT / ModernBERT 模型结构精读

> 阅读目标：吃透 **encoder-only** 这一族，它是 decoder-only（1M1 起的全部内容）的**对照面**。
> 前四篇你只见过一种架构：单向、自回归、RoPE、Pre-Norm。这一篇换一个架构族，
> 目的是**防止思维定势**——同样叫「Transformer」，BERT 和 Llama 有四处方向相反的设计。
>
> 本篇还会顺带解决一个实用问题：**为什么 BERT 的代码你能直接读，ModernBERT 的不能**。
> 这正好是 [2A5](./2A5_modular_code_generation.md) 那条「有 modular 读 modular」原则的活例子。
>
> 源文件（三种读法各不相同）：
>
> | 模型 | 文件 | 行数 | 读法 |
> |------|------|------|------|
> | BERT | `src/transformers/models/bert/modeling_bert.py` | 1390 | **无 modular**，直读生成物；行号即真相 |
> | BERT config | `src/transformers/models/bert/configuration_bert.py` | 66 | 直读 |
> | ModernBERT | `src/transformers/models/modernbert/modular_modernbert.py` | **915** | **有 modular**，读这个（源文件） |
> | ModernBERT 生成物 | `src/transformers/models/modernbert/modeling_modernbert.py` | 865 | 只读，不要引用行号（会被重生成） |
> | ModernBERT config | `src/transformers/models/modernbert/configuration_modernbert.py` | 170 | 由 modular 生成 |

---

## 速读区（TL;DR）

**一句话结论**：encoder 与 decoder 的差别能缩到三行代码——`kv_idx <= q_idx` 还是 `q_idx >= 0`、
有没有 `shift`、以及 Norm 在残差里面还是外面；BERT = 双向 + 绝对位置 + Post-Norm + `[CLS]`，
ModernBERT = BERT 换上 Llama 那一代零件（RoPE / GeGLU / Pre-Norm / 无 bias）+ 双向 local-global 交替。

| # | 断言 | 出处（文件:行号） |
|---|------|------------------|
| 1 | 两个 mask 函数只差一行返回值：`causal_mask_function` → `kv_idx <= q_idx`；`bidirectional_mask_function` → `q_idx >= 0` | `masking_utils.py:76`/`:80`、`:83`/`:89` |
| 2 | BERT 的方向分发点是 `BertModel._create_attention_masks`，按 `config.is_decoder` 二选一，**默认走 bidirectional** | `modeling_bert.py:688` / `:696-704` |
| 3 | **BERT 是 Post-Norm**：`self.LayerNorm(hidden_states + input_tensor)`（注意力子层与 FFN 子层各一处）；Llama 是 `residual + hidden_states` | `modeling_bert.py:292` / `:350` |
| 4 | BERT 的位置信息是**绝对位置 embedding 加进输入**（`max_position_embeddings=512`），不是 RoPE | `modeling_bert.py:59` / `:100-106`、`configuration_bert.py:52` |
| 5 | **「loss 只对 mask 位置算」不是模型实现的**：模型算全序列 CE，靠数据整理器把其余位置填 `-100` | `modeling_bert.py:974` / `data_collator.py:831` |
| 6 | MLM 的 15% / 80% / 10% 三个默认值；那 10% 原样保留，但该位置仍要预测 | `data_collator.py:687-689` |
| 7 | MLM **没有 shift**，CLM 才 shift（`pad` 后错一位） | `loss/loss_utils.py:61-64` / `:86-87` |
| 8 | NSP 及其一堆类（`BertOnlyNSPHead` / `BertPreTrainingHeads` / `BertForNextSentencePrediction`）**只为老 checkpoint 兼容而留着** | `modeling_bert.py:509` / `:519` / `:990` |
| 9 | MLM 头是三层套娃 `BertOnlyMLMHead` → `BertLMPredictionHead` → `BertPredictionHeadTransform`（`Linear→GELU→LayerNorm`），权重绑到 `word_embeddings` | `modeling_bert.py:499` / `:483` / `:466` / `:910-913` |
| 10 | 读法分岔：BERT **无 modular**（直读 1390 行生成物）；ModernBERT **有 modular**（读 915 行源文件，生成物 865 行带 🚨 警告） | 本篇 5.0 节 |
| 11 | ModernBERT 换 RoPE：直接继承 `Gemma3RotaryEmbedding`；双 base `{"global": 160000.0, "local": 10000.0}` | `modular_modernbert.py:234`、`configuration_modernbert.py:77` |
| 12 | ModernBERT 的 FFN 是 **GeGLU 单块**：`Wi` 一次出 `2×intermediate` 再 `.chunk(2, -1)`；`mlp_bias=False` | `modular_modernbert.py:224` / `:230`、`configuration_modernbert.py:101` |
| 13 | ModernBERT 回到 **Pre-Norm**，第 0 层 `attn_norm` 是 `nn.Identity()`，并**补上了 BERT 没有的 `final_norm`** | `modular_modernbert.py:359` / `:375-381` / `:469` |
| 14 | `local_attention=128` 是**总窗口**，property 除 2 → `sliding_window=64`（半窗口），注意力层再 `+1` 适配 flash attention 闭区间 | `configuration_modernbert.py:99` / `:160-162`、`modular_modernbert.py:302` |
| 15 | `num_hidden_layers=22` 里 **8 层 `full_attention` + 14 层 `sliding_attention`**（每 3 层一个 global） | `configuration_modernbert.py:82` / `:113-120` |

**最容易踩的坑 / 反直觉发现**：
- **`-100` 是模型与数据的分界线**：`BertForMaskedLM.forward` 里没有任何「只挑 mask 位置」的逻辑，它算的是全序列 cross entropy，靠 `CrossEntropyLoss()` 默认的 `ignore_index=-100` 生效。想改 mask 策略要动 collator，不是模型。
- **BERT 的 mask 可能根本不存在**：`create_bidirectional_mask` 的 `allow_is_bidirectional_skip=True` 在「纯双向且无 padding」时直接返回 `None`——decoder 无论如何都得造一个下三角。
- **ModernBERT 的 local attention 是双向 band**（`masking_utils.py:149` 的 `abs(q_idx - kv_idx) <= sliding_window`），与 1M2 Mistral 的因果 SWA（`:99` 的 `kv_idx > q_idx - sliding_window`）方向相反。
- **第 0 层不归一化**（`:359` 的 `nn.Identity()`），且 ModernBERT **用的仍是 `nn.LayerNorm` 而不是 RMSNorm**，只是 `norm_bias=False`（减均值保留了）。
- **ModernBERT 必须先补 `final_norm`**：BERT 的 Post-Norm 让每层出口都已归一化，改 Pre-Norm 后主干上不再有任何归一化。
- **NSP 那一堆类别不要当重要机制去钻研**：RoBERTa 已用对照实验推翻它（去掉 NSP 下游更好），代码里留着纯为 `google-bert/bert-base-uncased` 的 `cls.seq_relationship.*` 张量。ModernBERT 里连 `token_type_embeddings` 都删了（`grep -c "token_type"` → 0，BERT 是 35）。
- 顺带：`ModernBertMLP` 的 `int(config.intermediate_size) * 2` 是防御性写法——`int()` 包在乘法**里面**。

**读完应该能回答的问题**：
- encoder 和 decoder 的差别能缩到哪几行代码？怎么用两条 `grep` 一眼判断陌生模型是哪种？
- MLM 的 loss 只对 mask 位置算——这件事是谁决定的？想换 mask 策略要动哪个文件？
- 「BERT 只能吃 512 个 token」的根源是什么？ModernBERT 为什么能到 8192？
- ModernBERT 的滑动窗口到底是多少？和 Mistral 的 SWA 差在哪？
- 为什么 encoder 取 `[CLS]`，而 decoder-only 只能取最后一个 token？

---

## 前置知识

- [1M1_llama_structure.md](./1M1_llama_structure.md)：Decoder-only 骨架。**本篇每一节都在和它对照**，
  没有 1M1 的 RoPE / Pre-Norm / SwiGLU 印象，本篇的对照表读起来会没有着力点。
- [2A5_modular_code_generation.md](./2A5_modular_code_generation.md)：为什么有 modular 就别读生成文件。
  本篇的 `bert`（无 modular）vs `modernbert`（有 modular）是一个教科书级案例。
- [1M2_mistral_sliding_window.md](./1M2_mistral_sliding_window.md)：SWA 的第一课（**因果**滑动窗口）。
  ModernBERT 的 local attention 是它的**双向**版本，读完第五节可以回头对照。

---

## 一、Encoder 与 Decoder：四处方向相反的差异

先把结论摆出来，后面全部用真实代码支撑。

| 维度 | Decoder-only（Llama） | Encoder-only（BERT） |
|------|----------------------|---------------------|
| 注意力方向 | **单向**（下三角） | **双向**（全 1） |
| 训练目标 | CLM：预测**下一个** token | MLM：完形填空，预测**被挖掉的** token |
| loss 覆盖范围 | 每个位置都算（序列长度 N 个监督信号） | **只有被 mask 的位置**（约 15%） |
| 表示怎么取 | 最后一个 token 的隐状态（`last_hidden_state[:, -1]`） | `[CLS]` 位置 pooling（或 mean pooling） |
| 位置编码 | RoPE（旋转 Q/K） | 绝对位置 embedding（**和 token embedding 相加**） |
| 归一化顺序 | Pre-Norm：`x + f(norm(x))` | **Post-Norm**：`norm(x + f(x))` |
| 能否生成 | 天生就会，逐 token 循环 | 默认不会；要生成必须 `config.is_decoder=True` 切成因果模式 |
| 能否看到未来 | 不能 | 能 |
| KV cache | 核心机制（1A5 / 2A3） | 不存在（没有「过去」需要缓存） |
| 典型用途 | 对话 / 生成 / 推理 | 句向量 / rerank / 分类 / 抽取式问答 |

### 1.1 差异一：mask 的形状——下三角 vs 全 1

`transformers` 把两种 mask 收敛成两个**纯函数**，定义在 `src/transformers/masking_utils.py`：

```python
# masking_utils.py:76
def causal_mask_function(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    """
    This creates a basic lower-diagonal causal mask.
    """
    return kv_idx <= q_idx


# masking_utils.py:83
def bidirectional_mask_function(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    """
    This creates a full bidirectional mask.

    NOTE: It is important to keep an index-based version for non-vmap expansion.
    """
    return q_idx >= 0
```

**这是全库最精炼的一处对照**：两个函数的差别只有一行返回值。

- `causal_mask_function`：位置 `q` 只能看 `kv_idx <= q_idx` 的 key——**下三角**。
- `bidirectional_mask_function`：`q_idx >= 0` 恒为真——**全 1，没有任何屏蔽**。
  （写成 `q_idx >= 0` 而不是 `return True`，是为了拿到一个既能在 eager 下做索引展开、
  又能被 vmap 批量化的形式；注释里那句 NOTE 说的就是这件事。）

ASCII 对比：

```
        ┌── Decoder-only (Llama) ──┐        ┌── Encoder-only (BERT) ──┐
        │       kv →               │        │       kv →               │
        │     0  1  2  3  4        │        │     0  1  2  3  4        │
        │ q 0 █  ·  ·  ·  ·        │        │ q 0 █  █  █  █  █        │
        │   1 █  █  ·  ·  ·        │        │   1 █  █  █  █  █        │
        │ ↓ 2 █  █  █  ·  ·        │        │ ↓ 2 █  █  █  █  █        │
        │   3 █  █  █  █  ·        │        │   3 █  █  █  █  █        │
        │   4 █  █  █  █  █        │        │   4 █  █  █  █  █        │
        │                         │        │                         │
        │  kv_idx <= q_idx        │        │  q_idx >= 0             │
        │  「只能看自己和过去」     │        │  「谁都能看谁」          │
        └─────────────────────────┘        └─────────────────────────┘
```

BERT 在这里的**分发点**是 `BertModel._create_attention_masks`（`modeling_bert.py:688`），
它按 `config.is_decoder` 二选一：

```python
# modeling_bert.py:696-704
        if self.config.is_decoder:
            attention_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=embedding_output,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
        else:
            attention_mask = create_bidirectional_mask(     # ← 默认走这条
```

注意 `else` 分支**没有传 `past_key_values`**：双向注意力不需要缓存，因为整条序列一次算完。
对比 `create_causal_mask` 的签名里 `past_key_values` 是必填位置参数——差别在此。

> **实用副产物**：`create_bidirectional_mask` 有一个 `allow_is_bidirectional_skip=True` 参数，
> 文档字符串原文是
> 「Whether to allow returning `None` (no bias) when the mask is plain bidirectional with no padding」。
> 也就是说：**如果 batch 里没有 padding，BERT 的 mask 直接省掉、返回 `None`**。
> 这是 encoder 相对 decoder 的一个天然优势——decoder 无论如何都得造一个下三角。

### 1.2 差异二：为什么 encoder 能双向

这不是「BERT 作者喜欢双向」，而是**任务定义决定的**：

- **Decoder 的任务是生成**。第 `t` 步的输出要拿去当第 `t+1` 步的输入，如果第 `t` 步偷看了 `t+1`，
  训练时它学会了「抄答案」，推理时答案根本不存在——loss 会很好看，模型完全没用。
  所以 causal mask 不是可选项，是**自回归任务的正确性约束**。
- **Encoder 的任务是理解**。输入是完整已知的一条序列（一句话、一个文档），
  输出是「每个 token 的上下文相关表示」。既然整条序列本来就摆在面前，
  强行遮住右边一半只会**白白丢掉一半信息**。双向注意力把每个 token 的表示都建立在
  完整上下文上，对分类 / 抽取 / 检索这类任务直接更强。

一句话：**decoder 遮住未来是为了不撒謊，encoder 不需要遮是因为它不预测未来。**

### 1.3 差异三：训练目标——CLM vs MLM

| | CLM（Causal LM） | MLM（Masked LM，完形填空） |
|---|---|---|
| 输入处理 | 原样 | 随机挖掉 15% 的 token，填 `[MASK]` |
| 预测目标 | 位置 `t` 预测 `t+1` | 位置 `t` 预测**被挖掉的那个 token 自己** |
| 监督信号来源 | 序列本身（移位一位） | 需要额外构造 `labels`（挖掉的位置填真值，其余填 `-100`） |
| 方向 | 单向 | 双向 |

CLM 的实现就是一行 `shift`（`src/transformers/loss/loss_utils.py:61-64`）：

```python
    if shift_labels is None:
        # Shift so that tokens < n predict n
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()
```

MLM 则**没有 shift**（`loss_utils.py:86-87`）：`logits` 和 `labels` 直接对齐在同一位置上。
真正的「挖词」动作不在模型里，而在**数据整理器**里——这是本篇一个容易误解的点，第三节细讲。

### 1.4 差异四：表示怎么取

decoder-only 要拿到句向量，取最后一个位置（`last_hidden_state[:, -1]`）；
BERT 走的是**专门的 pooling 层**，取第一个 token 也就是 `[CLS]`，见第二节 5。

---

## 二、BERT 的积木（自底向上，逐个标行号）

先给全景类图（全部来自 `src/transformers/models/bert/modeling_bert.py`）：

```
BertEmbeddings        :53     token + position + token_type 三路相加 → LayerNorm → dropout
BertSelfAttention     :139    多头双向注意力（Q/K/V 三个独立投影，无 GQA，无 RoPE）
BertCrossAttention    :206    交叉注意力（仅 is_decoder 时启用）
BertSelfOutput        :282    残差 + Post-LayerNorm   ← Post-Norm 在这里
BertAttention         :296    = BertSelfAttention + BertSelfOutput
BertIntermediate      :325    Linear(hidden → intermediate) + GELU
BertOutput            :340    Linear(intermediate → hidden) + 残差 + Post-LayerNorm  ← Post-Norm 在这里
BertLayer             :354    一层 = (Attention) + (Intermediate → Output)
BertEncoder           :419    nn.ModuleList([BertLayer × num_hidden_layers])
BertPooler            :451    取 [CLS]，Linear + tanh
BertModel             :594    堆叠 + pooling
BertForMaskedLM       :909    BertModel + BertOnlyMLMHead
```

### 2.1 `BertEmbeddings`（:53）—— 三路 embedding 相加

```python
# modeling_bert.py:58-60
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
```

`config.type_vocab_size = 2`（`configuration_bert.py:53`）——两个「段」的 id，
对应 NSP 任务的句子 A / 句子 B（第四节）。相加发生在 forward：

```python
# modeling_bert.py:100-106
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings

        position_embeddings = self.position_embeddings(position_ids)
        embeddings = embeddings + position_embeddings

        embeddings = self.LayerNorm(embeddings)
```

**与 Llama 的对照**：

| | Llama（1M1） | BERT |
|---|---|---|
| token embedding | `embed_tokens` | `word_embeddings` |
| 位置信息 | **不进 embedding**，RoPE 在每层旋转 Q/K | `position_embeddings`，**加进 embedding** |
| 段落/类型信息 | 无 | `token_type_embeddings`（ModernBERT 已删除） |
| 相加后再做什么 | 直接进第一层 | 还要过一次 `LayerNorm`（:106）+ dropout |

### 2.2 绝对位置编码 vs RoPE

BERT 的 `position_ids` 是一个预注册的 buffer（`modeling_bert.py:65`）：

```python
        self.position_ids = nn.Buffer(torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False)
```

`max_position_embeddings = 512`（`configuration_bert.py:52`）——这就是那个著名的「BERT 只能吃 512 个 token」。

| | 绝对位置 embedding（BERT） | RoPE（Llama，详见 2A4） |
|---|---|---|
| 存放形式 | 一张 `[512, 768]` 的可学习表 | 无参数（`inv_freq` 是按公式算的 buffer） |
| 作用位置 | **输入层加一次** | **每一层的 Q/K 上转一次** |
| 外推能力 | **差**：第 513 个位置根本没有对应的行，直接报错 | **好**：位置是旋转角度，理论上可外推（实际还需 scaling） |
| 相对性 | 无内建相对性，靠模型自己从绝对位置学出相对模式 | 内建：注意力分数只取决于 `q`、`k` 的角度差，即**相对距离** |
| 参数量 | `max_position_embeddings × hidden_size`（768×512 ≈ 39 万） | 0 |
| 超长上下文 | 必须重训 / 插值 | linear / dynamic / yarn 等各种 scaling（2A4） |

**这就是 ModernBERT 换掉它的原因**：要把上下文从 512 提到 8192，
learnable 位置表的参数量会涨到 8192×768 ≈ 629 万，而且外推依然差。换成 RoPE 直接解决（第五节）。

### 2.3 `BertSelfAttention`（:139）—— 纯正的多头双向注意力

```python
# modeling_bert.py:149-156
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.scaling = self.attention_head_size**-0.5

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
```

**逐条和 Llama 对照**：

| 特征 | Llama（`modeling_llama.py:220-232`） | BERT（`modeling_bert.py:149-156`） |
|------|--------------------------------------|-----------------------------------|
| GQA | **有**：`num_key_value_groups = num_attention_heads // num_key_value_heads`，K/V 投影到更少的头 | **没有**：`k`/`v` 也是 `all_head_size`，和 `q` 一样多 |
| RoPE | **有**：`position_embeddings` 传入，`apply_rotary_pos_emb(q, k, cos, sin)` | **完全没有**：forward 里连 `position_ids` 都不接 |
| `is_causal` | `self.is_causal = True`（硬编码） | `self.is_causal = is_causal`（:161，由 `config.is_decoder` 传入，**默认 False**） |
| KV cache | `past_key_values.update(...)` 是流程的一部分 | 有 `past_key_values` 参数，但**双向场景下 `BertModel` 根本不会创建 cache** |
| 投影切分 | `q/k/v/o_proj` 分两个矩阵（`q` 用一个、`k`/`v` 共用形状） | `query`/`key`/`value` 三个独立 `nn.Linear`，外加 `BertSelfOutput.dense` 当 `o_proj` |
| bias | `bias=config.attention_bias`（可配） | **恒有 bias**（`nn.Linear(...)` 默认 `bias=True`） |

forward 的数据流（`modeling_bert.py:175-203`）：

```python
# modeling_bert.py:175-177
        query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
```

**注意这里没有 RoPE 那两行**——Llama 的对应位置（`modeling_llama.py:259`）有
`query_states, key_states = apply_rotary_pos_emb(...)`，BERT 没有。

然后是标准的后端分发（和 Llama 一模一样，2A1 会细讲）：

```python
# modeling_bert.py:188-202
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self, query_layer, key_layer, value_layer, attention_mask,
            dropout=0.0 if not self.training else self.dropout.p,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
```

最朴素的 `eager_attention_forward`（`modeling_bert.py:111`）内容是：

```python
# modeling_bert.py:123-131
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
```

**「是因果还是双向」这件事，这个函数自己完全不知道**——它只是把外面传进来的 `attention_mask`
加到注意力分数上。方向性 100% 由 `BertModel._create_attention_masks`（:688）决定。
这是理解全库注意力机制的关键抽象（详见 2A2）。

### 2.4 `BertLayer`（:354）—— **Post-Norm**，和 Llama 完全相反

**这是全篇最值得记住的一处差异。**

先看 BERT。注意 `BertSelfOutput` 和 `BertOutput` 两个类的 forward 长得一模一样：

```python
# modeling_bert.py:282-294
class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)   # ← :292  Post-Norm
        return hidden_states
```

```python
# modeling_bert.py:340-352
class BertOutput(nn.Module):
    ...
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)   # ← :350  Post-Norm
        return hidden_states
```

再看 Llama（`modeling_llama.py:305-323`）：

```python
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)     # 先 norm，再进子层
        hidden_states, _ = self.self_attn(...)
        hidden_states = residual + hidden_states                # 残差加在 norm 之外

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
```

写成公式：

```
Pre-Norm  (Llama)：  x ← x + Sublayer(Norm(x))
Post-Norm (BERT) ：  x ← Norm(x + Sublayer(x))
```

```
┌─────────── Pre-Norm (Llama) ───────────┐   ┌─────────── Post-Norm (BERT) ───────────┐
│                                         │   │                                         │
│  x ──┬──────────────────────────►(+)──► │   │  x ──┬───────────────►(+)──► Norm ──►   │
│      │                            ▲     │   │      │                ▲                  │
│      └─► Norm ─► Sublayer ────────┘     │   │      └─► Sublayer ────┘                  │
│                                         │   │   (Sublayer 内部自己带残差输入 input_tensor)│
│  归一化在「旁路」上                        │   │  归一化在「主干」上                      │
└─────────────────────────────────────────┘   └─────────────────────────────────────────┘
```

**为什么现代模型都改用 Pre-Norm**：Pre-Norm 让残差路径是一条**无归一化的恒等高速公路**，
梯度可以一路直通底层，深层网络（几十上百层）训练稳定得多，也不那么依赖 warmup。
Post-Norm 的归一化直接压在主干上，深层时梯度容易被反复缩放，需要小心调 warmup。
BERT 是 12 层，Post-Norm 扛得住；到 Llama 那种几十层（Llama 3 70B 有 80 层）就不行了。

**读代码时的辨别技巧**：看 `LayerNorm` 出现在 `forward` 的**哪一行**。
`norm(x + sub(x))` = Post-Norm；`x + sub(norm(x))` = Pre-Norm。

`BertLayer` 自己基本不干活，只是把上面的积木串起来：

```python
# modeling_bert.py:354-362
class BertLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config, is_causal=config.is_decoder, layer_idx=layer_idx)  # ← :358
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
```

`is_causal=config.is_decoder`（:358）就是 1.1 节那个开关的**起点**：默认 `is_decoder=False`，
于是 `is_causal=False`，于是 `BertSelfAttention` 不做任何因果限制。

`BertEncoder`（:419）是纯堆叠，连 `LayerNorm` 都没有（对比 Llama 的 `LlamaModel` 有一层
`norm(hidden_states)` 收尾）——因为 Post-Norm 已经把每层出口都归一化过了。
BERT 的收尾归一化在 ModernBERT 里才有（`self.final_norm`），见第五节。

### 2.5 `BertPooler`（:451）—— 取 `[CLS]`

```python
# modeling_bert.py:451-463
class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]      # ← :460 就是 [CLS]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
```

`[CLS]` 是词表里的一个特殊 token，永远在序列第一位。因为双向注意力，
经过 12 层后它的隐状态已经聚合了**整条序列**的信息，所以可以当句子表示用。

**对照**：

| | Llama | BERT |
|---|---|---|
| 取什么 | `last_hidden_state[:, -1]`（最后一个 token） | `last_hidden_state[:, 0]`（`[CLS]`） |
| 后面接什么 | 直接接 `lm_head` | `Linear` + `tanh`，再交给任务头 |
| 为什么这样 | 最后一个位置因为是单向的，天然见过全序列 | `[CLS]` 是双向聚合点，位置固定、与输入长度无关 |
| 额外开销 | 无 | `BertPooler` 一层 `768×768` ≈ 59 万参数 |

ModernBERT 里**没有 pooler 类**（`grep -c "pooler" modular_modernbert.py` → 0），
改成在任务头里按 `classifier_pooling` 现取：

```python
# modular_modernbert.py:658-660
        if self.config.classifier_pooling == "cls":
            last_hidden_state = last_hidden_state[:, 0]
        elif self.config.classifier_pooling == "mean":
```

`classifier_pooling: Literal["cls", "mean"] = "cls"`（`configuration_modernbert.py:104`）。
注意 config docstring 里的一句提醒（`configuration_modernbert.py:47-48`）：

> In local attention layers, the CLS token doesn't attend to all tokens on long sequences.

——这是 local/global 交替注意力的直接后果（5.4 节）。

---

## 三、MLM 头与 loss

### 3.1 三个真实类名

```python
# modeling_bert.py:466   BertPredictionHeadTransform
# modeling_bert.py:483   BertLMPredictionHead
# modeling_bert.py:499   BertOnlyMLMHead
```

关系是**逐层包一层**：

```
BertOnlyMLMHead          (:499)   只干一件事：调 self.predictions(...)
  └─ BertLMPredictionHead(:483)   = BertPredictionHeadTransform + decoder + bias
       └─ BertPredictionHeadTransform (:466) = Linear → 激活 → LayerNorm
```

```python
# modeling_bert.py:483-497
class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=True)   # ← :490
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))                     # ← :491

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states
```

两个值得注意的细节：

1. **权重绑定**：`BertForMaskedLM._tied_weights_keys`（`modeling_bert.py:910-913`）把
   `cls.predictions.decoder.weight` 绑到 `bert.embeddings.word_embeddings.weight`——
   输入 embedding 和输出投影共用同一张 `[30522, 768]` 的表（这就是注释里那句
   "The output weights are the same as the input embeddings"）。
   对比 `LlamaForCausalLM` 的同一机制（1M1 的 `lm_head`）。而 `BertLMPredictionHead`
   还额外有 `decoder.bias` 和 `self.bias` 两个**输出专用**的偏置——这对孪生偏置是 BERT 的历史包袱
   （`self.bias` 保留只是为了老 checkpoint 兼容），`_init_weights` 里把它们都置零（`modeling_bert.py:551`）。
2. **`transform` 的结构和 Llama 的 LM head 完全不同**：Llama 的 `lm_head` 是**一个裸 `nn.Linear`**；
   BERT 在它前面插了 `Linear → GELU → LayerNorm` 三层，叫 `BertPredictionHeadTransform`。
   这也是为什么 `bert-base` 比同尺寸 decoder 多出一点参数。

### 3.2 loss 到底对哪些位置算？——关键在 `-100`

`BertForMaskedLM.forward`（`modeling_bert.py:939-983`）里 loss 只有两行：

```python
# modeling_bert.py:972-975
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
```

**注意这里没有任何「只挑 mask 位置」的逻辑**。它算的是**全序列**的 cross entropy——
只不过 `CrossEntropyLoss()` 的默认 `ignore_index=-100`，而**被 mask 的位置以外的 label
全是 `-100`**。

所以答案是：**loss 只对 mask 位置算，但「哪些位置被 mask」这件事是数据整理器决定的，不是模型。**

真实代码在 `src/transformers/data/data_collator.py` 的 `DataCollatorForLanguageModeling`（`:621`）：

```python
# data_collator.py:808-831
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        ...
        probability_matrix.masked_fill_(no_mask_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix, generator=self.generator).bool()
        ...
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
```

默认超参（`data_collator.py:687-689`）：

```python
    mlm_probability: float | None = 0.15
    mask_replace_prob: float = 0.8
    random_replace_prob: float = 0.1
```

也就是经典的 **15% / 80% / 10%**：随机选 15% 的位置，其中 80% 填 `[MASK]`、
10% 填随机词、10% 原样保留（但该位置仍要预测）。最后 10% 存在的理由是：
推理时根本不会出现 `[MASK]`，训练时必须让模型见过「没被破坏的输入也要能预测自己」。

**完整链路**（这是本篇最容易搞错的一处，务必记住边界）：

```
DataCollatorForLanguageModeling (data_collator.py:831)
    构造 labels：mask 位置 = 真 token id，其余 = -100
                    │
                    ▼
BertForMaskedLM.forward (modeling_bert.py:974)
    loss_fct = CrossEntropyLoss()        ← 默认 ignore_index=-100
    loss = loss_fct(logits.view(-1, V), labels.view(-1))
                    │
                    ▼
    梯度只在 mask 位置（约 15%）上产生
```

顺带对照 CLM：`ForCausalLMLoss`（`loss_utils.py:49`）**不**用数据整理器造 label，
它拿原样的 `input_ids` 然后**移位一位**（`loss_utils.py:61-64`），并且每个位置都算 loss——
监督信号密度是 MLM 的约 6.7 倍。这就是「MLM 训练慢」的根本原因，也是后续
ELECTRA / 各种 span-mask 方法的动机。

> **`loss_function` 是怎么选到 `ForMaskedLMLoss` 的**：
> `ModernBertForMaskedLM` 里写的是 `self.loss_function(...)`，看起来是个魔法属性。
> 真实机制在 `modeling_utils.py:1364-1372`——`__init__` 时用类名去 `LOSS_MAPPING` 的正则里匹配：
> `ModernBertForMaskedLM` 命中 `"ForMaskedLM"` → `ForMaskedLMLoss`（`loss_utils.py:74`，**不 shift**）。
> 而 `ModernBertForSequenceClassification` 命中 `"ForSequenceClassification"`。

---

## 四、NSP：一个被证伪、但还留在代码里的目标

### 4.1 它是什么

```
输入构造（BERT 原始论文的预训练数据）：
  A = [CLS] 句子1 [SEP] 句子2 [SEP]
      token_type_ids = 0 0 0 0  1 1 1 1

50% 概率「句子2 真的是句子1 的下一句」 → label 0 (IsNext)
50% 概率「句子2 是从语料里随机抽的」   → label 1 (NotNext)
```

模型要在 `[CLS]` 上做二分类。代码里就是 `BertOnlyNSPHead`（`modeling_bert.py:509-516`）：

```python
class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score
```

`Linear(768, 2)` —— 一个**区区 1538 参数**的分类头。这就是 NSP 的全部新参数。

同时训练两个目标时，`BertForPreTraining`（`modeling_bert.py:727`）把两个 loss 直接相加：

```python
# modeling_bert.py:804-808
        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
```

（`BertPreTrainingHeads`（:519）就是 `BertLMPredictionHead` + `Linear(hidden, 2)` 的组合。）

### 4.2 为什么被弃用

一句话：**NSP 太简单，模型靠「话题是否一致」就能做对，学不到句间关系。**
随机抽来的句子大概率话题完全不同，模型只要判断「这两句在不在说同一件事」就能拿到高分，
这对下游任务毫无帮助。

RoBERTa（2019）做了对照实验，结论是：

| 结论 | 说明 |
|------|------|
| 去掉 NSP，只留 MLM，**下游更好** | 直接推翻了 NSP 的必要性 |
| 换成 **sentence-order prediction (SOP)** 仍有收益，但收益来自「相邻句顺序」这个 harder 的目标 | 说明问题不在「句间任务」，在 NSP 的**难度太低** |
| 真正的关键超参是：更大 batch、更多数据、动态 mask、更长训练 | NSP 属于「看起来有道理但实测没用」的一类 |

所以今天的实践是：**用 MLM（或它的变体）训 encoder，完全不要 NSP。**
代码里保留 `BertForNextSentencePrediction`（:990）、`BertOnlyNSPHead`（:509）、
`BertPreTrainingHeads`（:519）纯粹是**为了老 checkpoint 的向后兼容**
（`google-bert/bert-base-uncased` 的权重里有 `cls.seq_relationship.*` 这两个张量）。
ModernBERT 里则**一个 NSP 相关的类都没有**，连 `token_type_embeddings` 也一并删了。

> **对阅读源码的意义**：看到 BERT 里 NSP 那一堆类，不要当成「重要机制」去钻研，
> 当作历史遗留物即可。真正要读的是 `BertForMaskedLM` 那条路径。

---

## 五、ModernBERT 的现代化改造（进阶部分）

ModernBERT（2024-12）是 Answer.AI + LightOn 的作品，一句话概括：

> **把 BERT 的骨架保留，把所有零件换成 Llama 那一代的现代化组件。**

官方 doc 的原话（`docs/source/en/model_doc/modernbert.md:27`）：

> It brings many improvements to the original architecture such as rotary positional embeddings to support
> sequences of up to 8192 tokens, unpadding to avoid wasting compute on padding tokens, GeGLU layers,
> and alternating attention.

下面逐条用代码验证。**所有行号来自 `modular_modernbert.py`（源文件），不是生成物。**

### 5.0 先读哪个文件？——顺带一个 2A5 的实例

```bash
$ ls src/transformers/models/bert/
__init__.py  configuration_bert.py  modeling_bert.py  tokenization_bert.py  ...
                                        ↑ 没有 modular_bert.py

$ ls src/transformers/models/modernbert/
__init__.py  configuration_modernbert.py  modeling_modernbert.py  modular_modernbert.py
                                                                    ↑ 有
```

这正是 [2A5](./2A5_modular_code_generation.md) 规则 1 的两半：

| | BERT | ModernBERT |
|---|---|---|
| modular 文件 | **无** | 有（915 行） |
| 生成物 | `modeling_bert.py` 就是源文件，1390 行 | `modeling_modernbert.py` 865 行，顶上有 `🚨🚨🚨 Do NOT edit this file manually` |
| 读法 | **直读 `modeling_bert.py`**，行号稳定 | **读 `modular_modernbert.py`**；引用生成物行号会被下次 `make fix-repo` 打乱 |
| modular 的 import | — | `from ..gemma3.modeling_gemma3 import Gemma3RotaryEmbedding, rotate_half`（`modular_modernbert.py:46`）——**跨模型真继承**，正是 modular 才能写的东西 |

生成文件的警告头（`modeling_modernbert.py:1-6`）：

```
#                🚨🚨🚨 ... 🚨🚨🚨
#           This file was automatically generated from src/transformers/models/modernbert/modular_modernbert.py.
#               Do NOT edit this file manually as any edits will be overwritten by the generation of
#             the file from the modular. ...
```

**一个有 modular、一个没有，这两个文件恰好构成本路线图里「读哪种文件」的完整对照样本。**

### 5.1 RoPE 替换绝对位置编码 ✅

```python
# modular_modernbert.py:45-46
from ..align.modeling_align import eager_attention_forward
from ..gemma3.modeling_gemma3 import Gemma3RotaryEmbedding, rotate_half
```

```python
# modular_modernbert.py:234-236
class ModernBertRotaryEmbedding(Gemma3RotaryEmbedding):
    def __init__(self, config: ModernBertConfig, device=None):
        super().__init__(config)
```

**注意这个 `pass` 级别的继承**（2A5 规则 3）：ModernBERT 的 RoPE 就是 Gemma3 的 RoPE，
一个字都没改，只多了一个 `compute_default_rope_parameters` 转发的薄壳（:238-243）。
类本身是从别的模型**真继承**过来的——这在生成文件里会被展开成完整代码，
但在 modular 里就两行。

位置编码的双套参数是 ModernBERT 的特色（global / local 用不同 base）：

```python
# configuration_modernbert.py:77
    default_theta = {"global": 160_000.0, "local": 10_000.0}
```

- 全局注意力层用 `rope_theta = 160000`（大 base，适合长距离）
- 局部注意力层用 `rope_theta = 10000`（标准 base，窗口只有 128，不需要长距离分辨率）

应用位置（`modular_modernbert.py:329`）：

```python
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)
```

—— 和 Llama 的 `apply_rotary_pos_emb` 是同一件事（1M1 的第 2 块）。**embedding 里不再加任何位置向量**。

### 5.2 GeGLU 替换 GELU MLP，且去掉 bias ✅

BERT 的 FFN 是三段式（`BertIntermediate` :325 → GELU → `BertOutput` :340），
且 `nn.Linear` 默认带 bias：

```python
# modeling_bert.py:328   （BertIntermediate.__init__）
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
```

ModernBERT 换成单块的 **GLU 门控**：

```python
# modular_modernbert.py:214-231
class ModernBertMLP(nn.Module):
    """Applies the GLU at the end of each ModernBERT layer.

    Compared to the default BERT architecture, this block replaces :class:`~transformers.model.bert.modeling_bert.BertIntermediate`
    and :class:`~transformers.model.bert.modeling_bert.SelfOutput` with a single module that has similar functionality.
    """

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.Wi = nn.Linear(config.hidden_size, int(config.intermediate_size) * 2, bias=config.mlp_bias)  # ← :224
        self.act = ACT2FN[config.hidden_activation]
        self.drop = nn.Dropout(config.mlp_dropout)
        self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input, gate = self.Wi(hidden_states).chunk(2, dim=-1)          # ← :230
        return self.Wo(self.drop(self.act(input) * gate))
```

拆解这段：

| 点 | 说明 |
|----|------|
| 一个 `Wi` 出 `2 × intermediate_size` | 一次矩阵乘同时算出 `input` 和 `gate` 两半（比两个独立 `Linear` 快） |
| `.chunk(2, dim=-1)` | 把输出对半切开：前一半当值、后一半当门 |
| `act(input) * gate` | 门控：`gelu(input) ⊙ gate` = **GeGLU**（activation 是 `gelu` 时） |
| `bias=config.mlp_bias` | `mlp_bias: bool = False`（`configuration_modernbert.py:101`）→ **无 bias** |
| 无 `BertOutput` 那种残差 + LayerNorm | 残差与 norm 全部挪到 `ModernBertEncoderLayer` 里（5.3） |

**「去掉 bias」是 Llama 那一代的共识**：`LlamaMLP` 的 `gate/up/down_proj` 都是 `bias=False`。
理由是大模型里 bias 几乎不影响表达力，却会引入额外参数和跨设备通信。

`intermediate_size` 也换了算法：`hidden_size=768` 时 BERT 用 `3072`（4×），
ModernBERT 用 `1152`（`configuration_modernbert.py:81`，= 1.5×）。
因为 GLU 的 `Wi` 输出是 `2×intermediate`，`2 × 1152 = 2304 ≈ 3 × 768`，
和 Llama 的 `8/3 × hidden` 是同一套惯例（1M1 的 SwiGLU 也是这个比例）。
顺便一提，`int(config.intermediate_size) * 2`（:224）——注意这个 `int(...)` 包在乘法**里面**，
是个防御性写法。

> `hidden_activation` 的源码默认是 `"gelu"`（`configuration_modernbert.py:84`）。
> 所以默认拼出来就是 **GeGLU**。注意 GLU 的门控结构**与激活函数无关**——
> 换成 `"silu"` 就是 SwiGLU（即 Llama 的 MLP），换 `"gelu"` 就是 GeGLU。
> 这就是 1M1 里那个 `down_proj(act_fn(gate_proj(x)) * up_proj(x))` 的**合并写法**。

### 5.3 Pre-Norm（向 Llama 靠拢）✅

```python
# modular_modernbert.py:353-381
class ModernBertEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: ModernBertConfig, layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx == 0:
            self.attn_norm = nn.Identity()                    # ← :359  第 0 层不归一化
        else:
            self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.attn = ModernBertAttention(config=config, layer_idx=layer_idx)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.mlp = ModernBertMLP(config)
        self.attention_type = config.layer_types[layer_idx]   # ← :365  "full" 还是 "sliding"

    def forward(
        self, hidden_states, attention_mask=None, position_embeddings=None, **kwargs
    ) -> torch.Tensor:
        attn_output, _ = self.attn(
            self.attn_norm(hidden_states),                     # ← :375  Pre-Norm
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = hidden_states + attn_output            # ← :380  残差在 norm 之外
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))   # ← :381
        return hidden_states
```

`:380-381` 两行就是 1M1 里 Llama 的 `residual + hidden_states` 的紧凑写法——
**Pre-Norm 回来了**。这直接推翻了 2.4 节讲的 Post-Norm。

两个额外细节：

1. **第 0 层不归一化**（:359，`nn.Identity()`）。理由：输入已经过
   `ModernBertEmbeddings.norm`（:201）了，再来一次是冗余。这是 **identity 直连**的设计。
2. **`norm_bias: bool = False`**（`configuration_modernbert.py:89`）→ 用的是 `bias=False` 的 LayerNorm，
   和 Llama 的 `RMSNorm`（无 bias、无均值）口味一致。但注意 **ModernBERT 用的仍是 `nn.LayerNorm`
   而不是 RMSNorm**——`nn.LayerNorm(..., bias=config.norm_bias)`（:361、:363），减均值保留了。
3. 还有一个和 BERT 的重要结构差异：**ModernBERT 把最后的归一化补上了**：

```python
# modular_modernbert.py:469
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
```

   `BertEncoder`（:419）结尾**没有** norm（Post-Norm 每层出口已归一化），
   而 ModernBERT 改 Pre-Norm 后，主干上不再有任何归一化，所以必须像 `LlamaModel` 那样收尾补一层。

### 5.4 交替 local / global attention（呼应 1M2）✅

`layer_types` 在 config 里生成（`configuration_modernbert.py:113-120`）：

```python
    def __post_init__(self, **kwargs):
        # BC -> the pattern used to be a simple int, and it's still present in configs on the Hub
        global_attn_every_n_layers = kwargs.get("global_attn_every_n_layers", 3)
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool(i % global_attn_every_n_layers) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
```

`num_hidden_layers = 22`（`:82`），所以：

```
层号:  0     1     2     3     4     5     6   ...  21
类型: full  slid  slid  full  slid  slid  full ...  slid
      ↑每 3 层一个 global
```

22 层里 **8 层 full_attention（i = 0,3,6,9,12,15,18,21）+ 14 层 sliding_attention**。

窗口大小：`local_attention: int = 128`（`:99`）是**总窗口**，
而 config 用一个 property 把它换算成半窗口（`:160-162`）：

```python
    def sliding_window(self):
        """Half-window size: `local_attention` is the total window, so we divide by 2."""
        return self.local_attention // 2
```

→ 默认 `sliding_window = 64`。注意力层再 `+1` 适配 flash attention 的闭区间语义
（`modular_modernbert.py:299-304`）：

```python
        if config.layer_types[layer_idx] == "sliding_attention":
            # config.sliding_window = local_attention // 2 (half-window size, e.g. 64 for local_attention=128)
            # +1 is needed because flash attention sets inclusive boundaries (see modeling_flash_attention_utils.py)
            self.sliding_window = config.sliding_window + 1
        else:
            self.sliding_window = None
```

**关键：这个滑动窗口是双向的**，和 1M2 的 Mistral 完全不同。对照两个 mask 函数：

```python
# masking_utils.py:141-150   ← ModernBERT 用这个
def sliding_window_bidirectional_overlay(sliding_window: int) -> Callable:
    """..."""
    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        """A token can attend to any other token if their absolute distance is within
        the (inclusive) sliding window size (distance <= sliding_window)."""
        return abs(q_idx - kv_idx) <= sliding_window     # ← 绝对值！左右都看

    return inner_mask


# masking_utils.py:92-101    ← Mistral 用这个
def sliding_window_overlay(sliding_window: int) -> Callable:
    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return kv_idx > q_idx - sliding_window           # ← 只有左边，右边被 causal 挡掉
    return inner_mask
```

组合方式也不同（`masking_utils.py:134` vs `:154`）：

```python
sliding_window_causal_mask_function(sliding_window)
    = and_masks(sliding_window_overlay(sliding_window), causal_mask_function)          # 因果 ∧ 窗口

sliding_window_bidirectional_mask_function(sliding_window)
    = and_masks(sliding_window_bidirectional_overlay(sliding_window), bidirectional_mask_function)
```

```
   Mistral 的 SWA（因果 + 窗口）        ModernBERT 的 local（双向 + 窗口，±64）
        kv →                                kv →
     0  1  2  3  4  5  6                 0  1  2  3  4  5  6
q 0  █  ·  ·  ·  ·  ·  ·            q 0  █  █  █  ·  ·  ·  ·
  1  █  █  ·  ·  ·  ·  ·              1  █  █  █  █  ·  ·  ·
  2  █  █  █  ·  ·  ·  ·              2  █  █  █  █  █  ·  ·
  3  ·  █  █  █  ·  ·  ·              3  ·  █  █  █  █  █  ·
  4  ·  ·  █  █  █  ·  ·              4  ·  ·  █  █  █  █  █
  5  ·  ·  ·  █  █  █  ·              5  ·  ·  ·  █  █  █  █
  6  ·  ·  ·  ·  █  █  █              6  ·  ·  ·  ·  █  █  █
   ↑ 一个移动的「下三角带」             ↑ 一条对角线带（band），左右对称
```

**为什么要交替**：全都 global 在 8192 长度下 O(n²) 太贵；全都 local 则
`[CLS]` 看不到远处（config docstring 那句提醒就是这个），全局信息传不出去。
交替的方案是 Transformer 界的标准折中（[2M1_gemma2.md](./2M1_gemma2.md) 的 Gemma 2 用的是同一思路；
[1M2_mistral_sliding_window.md](./1M2_mistral_sliding_window.md) 的 Mistral 更激进——**全部层**都是 SWA）。

代码上的落地在 `ModernBertModel.forward`（`modular_modernbert.py:502-523`）：

```python
        if not isinstance(attention_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": hidden_states,
                "attention_mask": attention_mask,
            }
            attention_mask_mapping = {
                "full_attention": create_bidirectional_mask(**mask_kwargs),
                "sliding_attention": create_bidirectional_sliding_window_mask(**mask_kwargs),
            }

        position_embeddings = {}
        for layer_type in set(self.config.layer_types):
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=attention_mask_mapping[encoder_layer.attention_type],
                position_embeddings=position_embeddings[encoder_layer.attention_type],
                **kwargs,
            )
```

**两种 mask、两套 RoPE 的 cos/sin 各算一次**（不是每层重算——和 1M1 里 Llama
「RoPE 的 cos/sin 算一次传下去复用」是同一种优化），再按 `attention_type` 分发给每层。
另外注意 `attention_mask` 允许**直接传一个 dict 进来**（`isinstance(..., dict)` 那个判断）：
调用方可以自己预先造好两张 mask、按 `{"full_attention": ..., "sliding_attention": ...}`
分桶传进去，跳过内部的构造步骤。

### 5.5 支持更长的上下文 ✅

| | BERT | ModernBERT |
|---|---|---|
| `max_position_embeddings` | **512**（`configuration_bert.py:52`） | **8192**（`configuration_modernbert.py:85`） |
| 位置编码 | learnable 表，`512×768` ≈ 39 万参数 | RoPE，**0 参数** |
| 长度翻 16 倍代价 | 位置表参数涨 16 倍，且外推仍差 | 参数不涨，靠 RoPE + local/global 交替控住计算量 |

对比 Llama 的 `max_position_embeddings`（源码默认 `2048`，`configuration_llama.py:71`）：
ModernBERT 的 8192 是**训练时的真实上下文长度**（官方 doc 原文就是 "to support sequences of up to
8192 tokens"），而 Llama 那个值主要用于 RoPE 的 prefill 缓存分配与 `StaticCache` 尺寸，
真实可用长度由训练数据决定。

### 5.6 去掉 `token_type_ids` ✅

```bash
$ grep -c "token_type" src/transformers/models/modernbert/modular_modernbert.py
0
$ grep -c "token_type" src/transformers/models/bert/modeling_bert.py
35
```

ModernBERT 的 config 里**没有 `type_vocab_size`**，`ModernBertEmbeddings` 只有一路 token embedding：

```python
# modular_modernbert.py:197-211
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.drop = nn.Dropout(config.embedding_dropout)

    def forward(self, input_ids=None, inputs_embeds=None) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = self.drop(self.norm(inputs_embeds))
        else:
            hidden_states = self.drop(self.norm(self.tok_embeddings(input_ids)))
        return hidden_states
```

注意签名里**既没有 `token_type_ids` 也没有 `position_ids`**——对比
`BertEmbeddings.forward`（`modeling_bert.py:69-75`）那一长串参数。
`position_ids` 挪到了 `ModernBertModel.forward`（`modular_modernbert.py:483-490`），只用于算 RoPE。

去掉 `token_type_ids` 的理由很直接：**它的存在意义完全来自 NSP**。
NSP 废了，段落 id 就没有用途了。这是「一个被弃用的预训练目标会连锁删掉一批结构」的典型例子
——`token_type_embeddings`（`768×2` = 1536 参数）和整个 `type_vocab_size` 字段一起消失。

---

## 六、今天的 BERT 用来做什么

BERT 早已不做「预训练基座」的角色了，但它是**encoder 的主流落地形态**：

| 用途 | 怎么用 | 为什么不用 decoder |
|------|--------|-------------------|
| **句向量 / 语义检索** | 取 `[CLS]`（或 mean pooling）当句向量，进向量库做 ANN 检索 | decoder 是单向的，只有最后一个 token 见过全序列，所以只能取它；而那个位置的表示是「为了预测下一个词」训出来的，不是「句子摘要」 |
| **Reranker** | `[CLS]` 上二分类「query 与 doc 是否相关」，对召回结果精排 | 双向注意力让 query 和 doc 的每个 token 互相看见，判断更准 |
| **分类 / 情感 / 意图** | `BertForSequenceClassification`（:1072），`[CLS]` + `Linear` | 同上 |
| **抽取式问答** | `BertForQuestionAnswering`（:1311），预测答案的 start/end 位置 | 必须同时看问题与上下文，双向是刚需 |
| **Token 级任务** | `BertForTokenClassification`（:1251），NER / POS | 每个 token 的表示都需要完整上下文 |
| **fill-mask** | `BertForMaskedLM` + `pipeline("fill-mask")` | MLM 本身就能当「完形填空」工具用 |

### 6.1 那 decoder-only 为什么也能做这些？（LLM-as-embedder）

2023 年之后出现了一条新路线：**拿 decoder-only LLM 来当 embedding 模型**
（E5-Mistral、GTE-Qwen、LLM2Vec、bge-en-icl 等）。它们的做法是：

| 改造 | 目的 |
|------|------|
| 把 causal mask **换成 bidirectional**（LLM2Vec 的核心） | 让每个 token 看到全序列，和 BERT 一样 |
| 取最后一个 token 的隐状态，或做 mean pooling | 得到句向量 |
| 加一个 instruction 前缀（"Represent this sentence for retrieval: ..."） | 用 decoder 的指令泛化能力 |
| 用对比学习（contrastive）继续训练 | 从「语言建模」适配到「表示学习」 |

**所以两条路线的差距不在骨架，只在训练目标和 mask**：
把 `causal_mask_function` 换成 `bidirectional_mask_function`（`masking_utils.py:76/83`）——
就这两行函数的差别。这也反过来说明：**BERT 的核心优势从来不是架构，而是
「双向 + MLM」这套组合天然更适合学表示。**

---

## BERT → ModernBERT 对照速查

先上主表（**这是全篇重点**）：

| 维度 | **BERT**（`modeling_bert.py`） | **ModernBERT**（`modular_modernbert.py`） |
|------|-------------------------------|------------------------------------------|
| 定位 | 原版 encoder-only | 把 BERT 换上 Llama 那一代零件 |
| 注意力方向 | 双向（`create_bidirectional_mask`，:704） | 双向（`create_bidirectional_mask` / `..._sliding_window_mask`，:509-510） |
| `is_causal` | `is_causal`（:161），默认 `False`（`is_decoder=False`） | `self.is_causal = False`（:306）**写死** |
| 位置编码 | 绝对位置 embedding，`nn.Embedding(512, 768)`（:59） | **RoPE**，继承 `Gemma3RotaryEmbedding`（:234） |
| RoPE base | — | 双套：`global 160000 / local 10000`（`configuration_modernbert.py:77`） |
| 段落编码 | `token_type_embeddings`（:60），`type_vocab_size=2` | **无**（`grep -c token_type` → 0） |
| FFN | `BertIntermediate`（GELU）+ `BertOutput` 两段（:325/:340） | `ModernBertMLP` 单块 **GeGLU 门控**（:214，`chunk(2)` + `act(input) * gate`） |
| FFN 比例 | `4 × hidden`（768→3072，`configuration_bert.py:48`） | `1.5 × hidden`（768→1152，`:81`），`Wi` 输出 `×2` |
| Linear bias | 有（`nn.Linear` 默认） | **无**：`attention_bias=False`、`mlp_bias=False`（`:95`/`:101`） |
| 归一化 | `LayerNorm`，有 bias，`eps=1e-12` | `LayerNorm`，**`bias=False`**（`norm_bias`，`:89`），`eps=1e-5` |
| 归一化顺序 | **Post-Norm**：`LayerNorm(x + sub(x))`（:292 / :350） | **Pre-Norm**：`x + sub(norm(x))`（:375-381） |
| 首层 norm | 每层都有 | 第 0 层是 `nn.Identity()`（:359） |
| 收尾 norm | **无**（`BertEncoder` 直接返回） | **有** `final_norm`（:469） |
| 注意力实现 | 拆成 `query`/`key`/`value` 三个独立 `nn.Linear`（:154-156） | **合并成单个 `Wqkv`**（:295），输出 `3 × head_dim × heads` |
| GQA | 无 | 无 |
| 层类型 | 全部全连接 | **交替**：每 3 层一个 `full_attention`，其余 `sliding_attention`（`configuration_modernbert.py:117-119`） |
| 滑动窗口 | 无 | `local_attention=128` → `sliding_window=64`（半窗口，:162），注意力层 `+1`（:302） |
| 窗口方向 | — | **双向**（`abs(q_idx - kv_idx) <= w`，`masking_utils.py:149`） |
| 最大长度 | **512**（`configuration_bert.py:52`） | **8192**（`configuration_modernbert.py:85`） |
| 层数 | 12（base，`:46`） | **22**（`:82`） |
| `[CLS]` 提取 | 专门的 `BertPooler`（`Linear` + `tanh`，:451） | **无 pooler 类**，任务头里按 `classifier_pooling` 现取（:658-660） |
| MLM 头 | `BertOnlyMLMHead` → `BertLMPredictionHead` → `BertPredictionHeadTransform`（:499/:483/:466） | `ModernBertPredictionHead`（`Linear→act→LayerNorm`，:530）+ `decoder`（:555） |
| MLM loss | `CrossEntropyLoss()` 手写在 forward（:974） | `self.loss_function(...)`，类名匹配到 `ForMaskedLMLoss`（:603） |
| 稀疏预测 | 无 | `sparse_prediction` 开关，只对非 `-100` 位置算 logits（:589-597） |
| NSP | **有**：`BertOnlyNSPHead`（:509）、`BertForPreTraining`（:727）、`BertForNextSentencePrediction`（:990） | **完全没有** |
| KV cache | 有（`is_decoder=True` 时用 `DynamicCache` / `EncoderDecoderCache`） | **没有**（整个文件只出现在 config 的 `keys_to_ignore_at_inference`） |
| 交叉注意力 | 有：`BertCrossAttention`（:206），`add_cross_attention` 开关 | 无 |
| 权重绑定 | `cls.predictions.decoder.weight` ↔ `word_embeddings` | `decoder.weight` ↔ `model.embeddings.tok_embeddings.weight`（:548） |
| modular | **无**（直读 1390 行生成物） | **有**（读 915 行源文件；生成物 865 行带 🚨 警告） |
| 跨模型继承 | 无 | 有：`Gemma3RotaryEmbedding`、`Gemma3` 的 `rotate_half`（:46） |

### 与 Llama（1M1）的三方速查

| 维度 | Llama（decoder） | BERT（encoder） | ModernBERT（modern encoder） |
|------|------------------|-----------------|------------------------------|
| mask | causal（下三角） | bidirectional（全 1） | bidirectional + sliding band |
| 位置编码 | RoPE | 绝对位置 embedding | **RoPE**（双 base） |
| 归一化 | RMSNorm | LayerNorm | LayerNorm（`bias=False`） |
| 归一化顺序 | Pre-Norm | **Post-Norm** | **Pre-Norm** |
| FFN | SwiGLU，`8/3×` | GELU，`4×` | **GeGLU**，`1.5×` |
| bias | 无（`attention_bias` 可配） | 有 | 无 |
| 取表示 | 最后一 token | `[CLS]` + pooler | `[CLS]` 或 mean（无 pooler 层） |
| 生成 | 天生 | 需 `is_decoder=True` | 不支持 |

**一句话记法**：

```
BERT        = 双向 + 绝对位置 + Post-Norm
ModernBERT  = BERT 换上 Llama 的现代化组件（RoPE / GeGLU / Pre-Norm / 无 bias）
              + 保留双向，外加 local/global 交替
```

---

## 一句话总结

> **encoder 和 decoder 的差别只有三行代码：`kv_idx <= q_idx` 还是 `q_idx >= 0`、
> 有没有 `shift`、以及 norm 在残差里面还是外面。**
> BERT = 双向 + 绝对位置 + Post-Norm + `[CLS]`；ModernBERT = BERT 换上 Llama 的
> RoPE / GeGLU / Pre-Norm / 无 bias，再叠上双向的 local-global 交替。
> 从此以后，`grep -n "is_causal"` 和 `grep -n "LayerNorm(hidden_states + "` 这两条命令
> 就能一眼判断一个陌生模型是 encoder 还是 decoder。

---

## 验证清单

以下命令全部在仓库根目录（`/Users/baai/Desktop/baai_proj/transformers`）实际跑过，输出为真实结果。

```bash
# ── 0. 哪个文件有 modular（本篇的读法分岔点）──────────────────
$ ls src/transformers/models/bert/ | grep modular        # → 无输出
$ ls src/transformers/models/modernbert/ | grep modular  # → modular_modernbert.py
$ wc -l src/transformers/models/bert/modeling_bert.py \
        src/transformers/models/modernbert/modular_modernbert.py \
        src/transformers/models/modernbert/modeling_modernbert.py
# → 1390 / 915 / 865
$ head -6 src/transformers/models/modernbert/modeling_modernbert.py
# → 顶部 🚨 "Do NOT edit this file manually" 警告（生成物）

# ── 1. 因果 vs 双向：两个 mask 函数的唯一差别 ─────────────────
$ grep -n "def causal_mask_function" -A 5 src/transformers/masking_utils.py
# → 76:def causal_mask_function(...)   / 80:    return kv_idx <= q_idx
$ grep -n "def bidirectional_mask_function" -A 8 src/transformers/masking_utils.py
# → 83:def bidirectional_mask_function(...)  / 89:    return q_idx >= 0

# ── 2. BERT 的双向分发点 ─────────────────────────────────────
$ sed -n '696,704p' src/transformers/models/bert/modeling_bert.py
# → 696: if self.config.is_decoder:  697: create_causal_mask(...)
#   703: else:                       704: create_bidirectional_mask(...)   ← 默认走这里
$ grep -n "is_causal=config.is_decoder" src/transformers/models/bert/modeling_bert.py
# → 358:        self.attention = BertAttention(config, is_causal=config.is_decoder, layer_idx=layer_idx)

# ── 3. 【重点】BERT 的 Post-Norm 位置 ────────────────────────
$ grep -n "self.LayerNorm(hidden_states + input_tensor)" src/transformers/models/bert/modeling_bert.py
# → 292:  （BertSelfOutput.forward，注意力子层）
# → 350:  （BertOutput.forward，FFN 子层）
$ grep -n "residual + hidden_states" src/transformers/models/llama/modeling_llama.py
# → 317、323（Llama 的 Pre-Norm：残差加在 norm 之外）

# ── 4. ModernBERT 的真实 config 默认值 ──────────────────────
$ grep -n "max_position_embeddings: int\|num_hidden_layers: int\|intermediate_size: int\|local_attention: int\|default_theta\|attention_bias: bool\|mlp_bias: bool\|norm_bias: bool" \
       src/transformers/models/modernbert/configuration_modernbert.py
# → 77:  default_theta = {"global": 160_000.0, "local": 10_000.0}
# → 81:  intermediate_size: int = 1152
# → 82:  num_hidden_layers: int = 22
# → 85:  max_position_embeddings: int = 8192
# → 89:  norm_bias: bool = False
# → 95:  attention_bias: bool = False
# → 99:  local_attention: int = 128
# → 101: mlp_bias: bool = False
$ sed -n '160,162p' src/transformers/models/modernbert/configuration_modernbert.py
# → def sliding_window(self):  "Half-window size..."  /  return self.local_attention // 2   （→ 64）
$ sed -n '113,120p' src/transformers/models/modernbert/configuration_modernbert.py
# → global_attn_every_n_layers = kwargs.get("global_attn_every_n_layers", 3)
#   layer_types = ["sliding_attention" if bool(i % 3) else "full_attention" for i in range(...)]

# ── 5. BERT 的 config 默认值 ────────────────────────────────
$ grep -n "max_position_embeddings: int\|type_vocab_size: int\|intermediate_size: int\|hidden_size: int" \
       src/transformers/models/bert/configuration_bert.py
# → 45: hidden_size: int = 768   48: intermediate_size: int = 3072
# → 52: max_position_embeddings: int = 512   53: type_vocab_size: int = 2

# ── 6. ModernBERT 真的没有 token_type_ids ───────────────────
$ grep -c "token_type" src/transformers/models/modernbert/modular_modernbert.py   # → 0
$ grep -c "token_type" src/transformers/models/bert/modeling_bert.py              # → 35

# ── 7. ModernBERT 的 Pre-Norm 与 GeGLU ──────────────────────
$ sed -n '375,381p' src/transformers/models/modernbert/modular_modernbert.py
# → 375: self.attn_norm(hidden_states)   ← Pre-Norm
#   380: hidden_states = hidden_states + attn_output
#   381: hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
$ sed -n '229,231p' src/transformers/models/modernbert/modular_modernbert.py
# → input, gate = self.Wi(hidden_states).chunk(2, dim=-1)
#   return self.Wo(self.drop(self.act(input) * gate))     ← GeGLU

# ── 8. 双向滑动窗口 vs 因果滑动窗口 ─────────────────────────
$ grep -n "def sliding_window_bidirectional_overlay" -A 10 src/transformers/masking_utils.py
# → 141; 149:  return abs(q_idx - kv_idx) <= sliding_window   ← 双向（band）
$ grep -n "def sliding_window_overlay" -A 10 src/transformers/masking_utils.py
# → 92; 99:   return kv_idx > q_idx - sliding_window          ← 因果（带）

# ── 9. MLM 的 loss 只算 mask 位置（-100 由 collator 造）─────
$ grep -n "We only compute loss on masked tokens" src/transformers/data/data_collator.py
# → 831:        labels[~masked_indices] = -100  # We only compute loss on masked tokens
$ grep -n "mlm_probability: float\|mask_replace_prob: float\|random_replace_prob: float" \
       src/transformers/data/data_collator.py
# → 687: mlm_probability: float | None = 0.15
#   688: mask_replace_prob: float = 0.8
#   689: random_replace_prob: float = 0.1
$ sed -n '61,64p' src/transformers/loss/loss_utils.py    # CLM 要 shift
$ sed -n '86,87p' src/transformers/loss/loss_utils.py    # MLM 不 shift

# ── 10. NSP 的存在证据 ──────────────────────────────────────
$ grep -n "class BertOnlyNSPHead\|class BertForNextSentencePrediction\|class BertPreTrainingHeads" \
       src/transformers/models/bert/modeling_bert.py
# → 509 / 519 / 990
$ sed -n '808p' src/transformers/models/bert/modeling_bert.py
# → total_loss = masked_lm_loss + next_sentence_loss
$ grep -c "NSP\|seq_relationship\|next_sentence" src/transformers/models/modernbert/modular_modernbert.py
# → 0（ModernBERT 里一点都没有）
```

**跑完这 10 组命令，本篇所有引用过的行号、类名、默认值都被独立验证过。**

---

## 导航

- 路线图：[00_roadmap.md](./00_roadmap.md)（本篇 = 入门 · 模型结构 · 1M5）
- 上一篇：[1M4_qwen3_dense.md](./1M4_qwen3_dense.md)
- 下一篇：[2M1_gemma2.md](./2M1_gemma2.md)（进阶 · 模型结构开篇）
- 相关：
  - [1M1_llama_structure.md](./1M1_llama_structure.md)：本篇每一节的对照基准
  - [2A5_modular_code_generation.md](./2A5_modular_code_generation.md)：为什么 BERT 读 `modeling_`、ModernBERT 读 `modular_`
  - [1M2_mistral_sliding_window.md](./1M2_mistral_sliding_window.md)：因果 SWA；本篇 5.4 是它的双向版本
- **入门 · 模型结构到此收尾**。你已经见过：
  - decoder-only 骨架（1M1 Llama）
  - 局部注意力（1M2 Mistral）
  - MoE（1M3 Mixtral）
  - 现代 dense 细节（1M4 Qwen3）
  - encoder-only 对照面（本篇）
  → 接下来 2M 系列全是「Llama + 若干改动」，2A 系列是库为这些模型提供的机制层。
