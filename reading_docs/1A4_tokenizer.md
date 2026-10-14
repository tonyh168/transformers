# Tokenizer 体系精读：fast / slow 两套与 `__call__` 全链路

> 阅读目标：搞清楚「一段文本怎么变成 `input_ids`」——共同的基类是谁、fast / slow 两条路径各自走到哪、
> padding / truncation / special tokens 在代码的哪一行被决定，以及 `chat_template` 为什么成了现代
> LLM 的实际入口。
>
> 源文件：`tokenization_utils_base.py`（3687 行，基类 + 共享逻辑）、
> `tokenization_utils_tokenizers.py`（1478 行，fast）、`tokenization_python.py`（1424 行，slow）、
> `tokenization_utils_sentencepiece.py`（315 行，slow 的 sentencepiece 变体）、
> `convert_slow_tokenizer.py`（2081 行，slow → fast 转换器）
>
> **一句话定位**：这一篇讲「数据怎么进模型」的第一道门。它的输出 `input_ids` / `attention_mask`
> 就是 1A5 `generate` 的输入。

---

## 速读区（TL;DR）

**一句话**：一个基类 `PreTrainedTokenizerBase` 定**流程**，三个后端只实现 `_encode_plus` 等几个**钩子**；
而 v5 里「fast / slow」已经不是两个类，而是**同一个类的别名**，`use_fast` 参数被**静默丢掉**。

| # | 断言 | 出处（文件:行号） |
|---|------|------------------|
| 1 | 基类不变（`PreTrainedTokenizerBase`）；slow = `PythonBackend`、fast = `TokenizersBackend`，`PreTrainedTokenizer` / `PreTrainedTokenizerFast` **只是别名** | `base.py:962`、`tokenization_python.py:400`/`:1424`、`tokenization_utils_tokenizers.py:85`/`:1478` |
| 2 | `AutoTokenizer(use_fast=...)` 在 v5 **被吞掉**：`_ = kwargs.pop("use_fast", None)`；映射表从「(slow, fast) 二元组」变成**一个类** | `models/auto/tokenization_auto.py:739-740`、`:61` |
| 3 | `LlamaTokenizerFast` 只是 `LlamaTokenizer` 的**别名**（一行 `LlamaTokenizerFast = LlamaTokenizer`），不是另一个类，别去找实现 | `models/llama/tokenization_llama.py:149` |
| 4 | `padding=True` + `max_length=128` 且无 truncation 时，**`max_length` 被忽略**（只补到 batch 最长），且只 `warnings.warn` | `base.py:2353-2357` |
| 5 | 只给 `max_length`（无 padding / truncation）会**自动开启截断** `truncation="longest_first"`（历史包袱）；`max_length` 缺省从 `model_max_length` 取，`> LARGE_INTEGER` 时反而**关掉策略** | `base.py:2342-2346`、`:2378-2390` |
| 6 | 没有 `pad_token` 却要求 padding → 直接 `ValueError` | `base.py:2391-2396` |
| 7 | fast 的 padding / truncation 是**先写进 Rust backend 的全局状态**再 `encode_batch` 一次算完；`truncation_side` / `padding_side` 在这里生效 | `tokenization_utils_tokenizers.py:850`、`:885-921`、`:1027` |
| 8 | 返回哪些 key 由 `model_input_names` 决定：Llama **不会**返回 `token_type_ids`，Bert 会 | `models/llama/tokenization_llama.py:89`、`models/bert/tokenization_bert.py:76` |
| 9 | `word_ids` / `char_to_token` / `offset_mapping` **只有 fast 有**；slow 下调这些 API 直接 `ValueError("... not available when using Python based tokenizers")` | `base.py:368-369` |
| 10 | `vocab_size` 是**基础**词表（`with_added_tokens=False`），`len(tokenizer)` 含 added tokens；两者不一致时必须 `resize_token_embeddings(len(tokenizer))` | `tokenization_utils_tokenizers.py:667`/`:718`、`modeling_utils.py:2709` |
| 11 | `bos_token_id` **不是独立存储字段**——每次访问时由 `bos_token` 现算 `convert_tokens_to_ids`，没设时返回 `None` 并打一行 error 日志 | `base.py:1281-1299` |
| 12 | BOS/EOS 两条路径：slow 走 `build_inputs_with_special_tokens`（配置驱动 `special_tokens_pattern`），fast 走 Rust **post-processor 模板**（`$A:0` / `<s>:0`） | `tokenization_python.py:865`、`tokenization_utils_tokenizers.py:590-611` |
| 13 | `add_bos_token` 的默认值来自 Hub 上的 `tokenizer_config.json`；**只给一个 `tokenizer.model` 不加载 config 时它是 `False`，BOS 不会被加** | `tokenization_utils_tokenizers.py:417-418` |
| 14 | `apply_chat_template` 内部回到 `self(...)` 时传 **`add_special_tokens=False`**（模板已手写控制符）→ chat 场景 BOS 不会翻倍 | `base.py:3122`、`:3123-3131` |
| 15 | `special_tokens_map.json` 在 v5 **只读不写**（注释 `# kept only for legacy`）；fast 也不再写 `added_tokens_decoder` | `base.py:1623`、`:2081-2089` |
| 16 | `_from_pretrained` 读到 `tokenizer_class` 后**立刻 `pop` 掉** → 直接 `LlamaTokenizer.from_pretrained(...)` 时该字段完全不影响结果，只有走 `AutoTokenizer` 才有意义 | `base.py:1771-1772` |
| 17 | `AutoTokenizer` 会把解析到的 `PythonBackend` **强制改写成 `TokenizersBackend`**（两处），并把 `Fast` 后缀剥掉 | `models/auto/tokenization_auto.py:925-926`、`:970`、`:864-865` |
| 18 | 只给一个 `.model` 文件也能跑：`convert_to_native_format` 第 ④ 步用 `SLOW_TO_FAST_CONVERTERS`（55 项）现场搭出 Rust backend | `tokenization_utils_tokenizers.py:211-284`、`convert_slow_tokenizer.py:1981` |

**最容易踩的坑 / 反直觉**

- **`use_fast` 已废弃**：v5 里传它**不报错也不生效**（`_ = kwargs.pop("use_fast", None)`）。看到 `use_fast=False` 的老教程/老代码，直接当噪音。
- **`XXXTokenizerFast` 不是类**：`grep` 出 14 处 `XXXFast = XXX` 全是别名；`PreTrainedTokenizerFast = TokenizersBackend`（`tokenization_utils_tokenizers.py:1477-1478`）。
- `padding_side` 的双重身份：`tokenizer.padding_side = "left"` 是**持久状态**（改掉 Rust backend），`tokenizer(..., padding_side="left")` 只是**本次调用的临时覆盖**；`set_truncation_and_padding` 的 docstring 自己承认退出后会被重置。
- `vocab_size` 与 `len(tokenizer)` **不等**是常态：加过 `pad_token` / `<image>` / `<|tool|>` 就会不一致，忘了 `resize_token_embeddings(len(tokenizer))` 就是 `embed_tokens` 的 `IndexError`（`models/llama/modeling_llama.py:353` 按 `config.vocab_size` 建 embedding）。
- `special_tokens_map.json` / `added_tokens.json` 在 v5 是「**只读的遗物**」：前者不再写，后者仅在 `index >= vocab_size` 时才写。
- 副产物结论：`TokenizersBackend._post_init` 在当前仓库**主流程没有调用点**（`grep -rn "\._post_init()"` 只剩个别老模型的 `tokenization_*.py`），更像是留给自定义 / 远程代码的钩子。
- AutoTokenizer 的选类有**九级优先级**，其中第 [5] 步专门给「Hub 上的类名写错了」打补丁（`MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS`，`:371`）——这是真实存在的历史数据问题，不是设计洁癖。

**读完应该能回答**

- 同一个 `tokenizer("hi", padding=True)`，fast 和 slow 各自的第一步分别是什么？padding 是在哪一层、用什么类型的状态传递给 Rust 的？
- `return_offsets_mapping=True` 在 slow 后端会发生什么？为什么？
- 为什么 `tokenizer.bos_token_id` 会打出一行 error 日志？它存在哪里？
- 给 Llama 加了一个 `pad_token` 之后，模型侧必须做什么、传什么值进去？
- 现代 LLM 的入口为什么是 `apply_chat_template` 而不是 `__call__`？它内部最终还是走到哪一行？

---

## 前置知识

- [1A1 仓库骨架与 auto 机制](./1A1_repo_and_auto.md)（待写）：`models/<name>/` 五件套里的 `tokenization_*.py` 是什么；`AutoTokenizer` 与 `AutoModel` 共用同一套「映射表 + 运行时选类」机制。
- [1M1_llama_structure.md](./1M1_llama_structure.md)：`nn.Embedding(config.vocab_size, hidden_size)` 是 tokenizer 的下游——`len()` 与 `config.vocab_size` 对不上，模型会在 embedding 处越界。

---

## 一、先纠正一个常见误解：v5 里「fast / slow」已经收编成「后端」

「transformers 有两套 tokenizer：slow 纯 Python，fast 是 Rust `tokenizers` 包装」——**历史正确、现状已过时**。

| 说法 | 老版本（v4 及以前） | 本仓库现状 |
|------|---------------------|-----------|
| 基类 | `PreTrainedTokenizerBase` | **不变**，`tokenization_utils_base.py:962` |
| slow 基类 | `PreTrainedTokenizer`（类） | `PythonBackend`（类），`tokenization_python.py:400`；旧名做别名保留 |
| fast 基类 | `PreTrainedTokenizerFast`（类） | `TokenizersBackend`（类），`tokenization_utils_tokenizers.py:85`；旧名做别名保留 |
| 模型 tokenizer 文件 | `tokenization_x.py` + `tokenization_x_fast.py` 两个文件 | **只剩 `tokenization_x.py`**，文件末尾挂别名 |
| `AutoTokenizer(use_fast=...)` | 有效参数 | **被吞掉**，不再起作用 |

别名就一行，写在文件末尾（模型侧 `LlamaTokenizerFast = LlamaTokenizer` 同理，`models/llama/tokenization_llama.py:149`）：

```python
# tokenization_python.py:1423-1424          # tokenization_utils_tokenizers.py:1477-1478
PreTrainedTokenizer = PythonBackend         PreTrainedTokenizerFast = TokenizersBackend  # Backward-compatible alias
```

**读源码判据**：`class XXX(PreTrainedTokenizer)` 继承 `PythonBackend`（slow）；`class XXX(TokenizersBackend)`
走 Rust；`XXXTokenizerFast = XXXTokenizer` 只是兼容别名，别去找它的实现。

| 继承写法 | 类数量 | 运行时实际基类 |
|----------|--------|----------------|
| `(TokenizersBackend)` | 50 | `TokenizersBackend`（Rust） |
| `(PreTrainedTokenizer)` | 32 | `PythonBackend`（纯 Python，写的是旧别名） |
| `(SentencePieceBackend)` | 6 | `SentencePieceBackend` → `PythonBackend` |
| `(PreTrainedTokenizerFast)` / `(BertTokenizer)` 等具体类 | 2 / ~10 | `TokenizersBackend`（旧别名）/ 视父类而定 |
| `XXXFast = XXX`（别名，不是类定义） | 14 | 同左值 |

**绝大多数现代模型（Llama / Qwen / GPT2 / Bert / T5）已统一到 `TokenizersBackend`**；还留在 slow 路径上的
多是老模型（BioGpt、CTRL、M2M100、Marian、XLM、Tapas、Wav2Vec2CTC…）。

### v5 怎么收编的

早期只有 Python 版（慢、**没有 offset mapping**——slow `_encode_plus:702` 连 `return_offsets_mapping` 参数都没有）；
引入 `tokenizers` 后为不破坏已有权重/代码两条线并存，于是每个模型两个文件。slow 下调对齐 API 直接报错
（`base.py:368-369`：`if not self._encodings:` → `raise ValueError("word_ids() is not available when using Python based tokenizers")`）。
v5 收编思路是**保留 fast 的实现能力、保留 slow 的名字**——映射表从「每个 model type → (slow 类, fast 类) 二元组」
变成**一个类**：

```python
# models/auto/tokenization_auto.py:739-740          # :61
_ = kwargs.pop("use_fast", None)   # V5: Simplified mapping - single tokenizer class per model type
```

---

## 二、类继承层级图

```
     PushToHubMixin
┌──────────────────────────────────────────────────────────────────────────┐
│ PreTrainedTokenizerBase                          tokenization_utils_base.py:962 │
│ 类属性 vocab_files_names :967 · model_input_names :973 · padding_side :974 · truncation_side :975 │
│      SPECIAL_TOKENS_ATTRIBUTES :980                                         │
│ 模板方法（必须实现）：_encode_plus :2543(:2566) · _add_tokens :1250 · vocab_size :1438 · __len__ :1434 │
│      added_tokens_decoder :1419 · save_vocabulary :2193                       │
│ 共享方法：__call__ :2418（唯一公共入口）· pad :2568 · _pad :2758 · add_special_tokens :1102 │
│      _get_padding_truncation_strategies :2336 · add_tokens :1209 · special_tokens_map :1344 │
│      all_special_tokens :1366 · from_pretrained :1489 · _from_pretrained :1751 │
│      save_pretrained :1976 · _save_pretrained :2137 · apply_chat_template :2989 · get_chat_template :3225 │
└──────────────────────────────────────────────────────────────────────────┘
   │ 分支 1（Rust）                          │ 分支 2 / 3（纯 Python）
   ▼                                         ▼
┌─────────────────────────────┐  ┌──────────────────────────────────────────┐
│ TokenizersBackend           │  │ PythonBackend                            │
│ tokenization_utils_tokenizers.py:85 │  tokenization_python.py:400          │
│ 别名 PreTrainedTokenizerFast :1478 │  别名 PreTrainedTokenizer :1424       │
│ is_fast :492 · __init__ :328 │  │ is_fast :454 · __init__ :413 · __len__ :499 │
│ convert_to_native_format :103│  │ _add_tokens :516 · tokenize :625          │
│ update_post_processor :590   │  │ _encode_plus :702 · build_inputs_w_special_… :865 │
│ vocab_size :667 · __len__ :718│ │ get_special_tokens_mask :969 · convert_ids_to_tokens :1055 │
│ _convert_encoding :738       │  │ prepare_for_model :1121 · truncate_sequences :1231 │
│ _add_tokens :794 · _decode :1086 │ create_token_type_ids_… :1298 · save_vocabulary :1362 │
│ set_truncation_and_padding:850│ └──────────────────────────────────────────┘
│ _encode_plus :925 · _save_pretrained :1129 │        │ 继承
│ train_new_from_iterator :1146 · → tokenizer.json │   ▼
└─────────────────────────────┘  ┌──────────────────────────────────────────┐
                                 │ SentencePieceBackend                     │
                                 │ tokenization_utils_sentencepiece.py:45   │
                                 │ class SentencePieceBackend(PreTrainedTokenizer) ← :26 │
                                 │ sp_model / vocab_size :100 · save_vocabulary :237 │
                                 └──────────────────────────────────────────┘
旁支（同层）：MistralCommonBackend  tokenization_mistral_common.py:186
```

关键区别一句话：`PreTrainedTokenizerBase` 定义「**流程**」（策略解析、padding、truncation、special tokens
编排、存取盘），三个后端只实现「**流程里的两三个钩子**」（`_encode_plus`、`_add_tokens`、`vocab_size`、
`__len__`、`added_tokens_decoder`）。标准模板方法模式。

### 三、四个文件各自的角色

| 文件 | 行数 | 角色 | 关键类/函数 |
|------|------|------|------------|
| `tokenization_utils_base.py` | 3687 | **公共基类 + 公共流程**。`BatchEncoding` 也在这里 | `PreTrainedTokenizerBase:962`、`BatchEncoding:195`、`__call__:2418` |
| `tokenization_utils_tokenizers.py` | 1478 | **fast 后端**（Rust `tokenizers`） | `TokenizersBackend:85`、`_encode_plus:925`、`set_truncation_and_padding:850` |
| `tokenization_python.py` | 1424 | **slow 后端**（纯 Python），含 `Trie` 等工具 | `PythonBackend:400`、`prepare_for_model:1121`、`truncate_sequences:1231` |
| `tokenization_utils_sentencepiece.py` | 315 | slow 的 **sentencepiece 变体**（继承 `PythonBackend`） | `SentencePieceBackend:45`、`SentencePieceExtractor:289` |
| `convert_slow_tokenizer.py` | 2081 | **slow → fast 转换器**（不参与日常 encode） | `SLOW_TO_FAST_CONVERTERS:1981`、`convert_slow_tokenizer:2041` |

---

## 四、`__call__` 全流程（用户最常走的路径）

### 4.0 全流程

`__call__`（`base.py:2418`）先做四件事：① 20+ 具名参数塞进 `all_kwargs` `:2469-2486`；②
`update(tokenizer_kwargs)` → `update(kwargs)`，显式参数优先级最高 `:2492-2494`；③ 解析策略
`_get_padding_truncation_strategies` `:2498` → `:2336`（出入见 4.1）；④ `text → self._encode_plus(...)` `:2512`，
`text_target → labels` `:2523`。随后**多态**进入后端：

- **fast** `TokenizersBackend._encode_plus` `:925` — ① 校验 text 形状 `_is_valid_text_input` `:946` ② `is_batched` `:979`
  ③ `set_truncation_and_padding` `:1010` → `:850`（策略写进 Rust backend 全局状态）④ `self._tokenizer.encode_batch` `:1027`
  （真正干活的是 Rust）⑤ `_convert_encoding` `:1034` → `:738`。
- **slow** `PythonBackend._encode_plus` `:702` — ① `is_batched` `:724` ② 逐条递归调 `_encode_plus` ③ 批量 `pad` `:790`；
  单条 `get_input_ids → tokenize → convert_tokens_to_ids` `:818` → `prepare_for_model` `:1121`（`a` 策略 `:1150` ·
  `b` `num_special` `:1179` · `c` `truncate_sequences` `:1184` → `:1231` · `d` `build_inputs_with_special_tokens` `:1194` → `:865`
  · `e` `create_token_type_ids` `:1195` · `f` `self.pad` `:1217` → `base:2568` · `g` `BatchEncoding(...)` `:1229`）。
  **SentencePieceBackend** 继承 slow，不重写。
- **出口** `BatchEncoding`（`base.py:195`，dict 子类）：`convert_to_tensors` `:665` 把 list 变成 `torch.Tensor`；
  `is_fast` `:306`（= `self._encodings is not None`）；`char_to_token` / `word_ids` / `token_to_chars` 等只在 fast 下有值。

### 4.1 第一步：`_get_padding_truncation_strategies`（base.py:2336）

**所有 padding/truncation 语义的唯一裁判**：`__call__` 里 20 多个参数先在这里坍缩成两个枚举 + 一个
`max_length`。五条规则（实测自 `base.py:2336-2418`）：

```python
# :2342-2346  只给 max_length（padding=False, truncation=None）→ 自动开启截断（历史包袱）
if max_length is not None and padding is False and truncation is None: truncation = "longest_first"

# :2348-2363  padding 归一化：True → LONGEST（补到 batch 最长）；字符串 → PaddingStrategy(padding)
if padding is not False: padding_strategy = PaddingStrategy.LONGEST if padding is True else PaddingStrategy(padding)
else:                    padding_strategy = PaddingStrategy.DO_NOT_PAD

# :2366-2376  truncation 归一化：True → TruncationStrategy.LONGEST_FIRST
# :2378-2390  没给 max_length 时从 model_max_length 取；模型不限长（> LARGE_INTEGER）则反向关掉策略
# :2391-2396  没有任何 pad_token 时 raise ValueError("Asking to pad but the tokenizer does not have a padding token. ...")
```

枚举定义在 `utils/generic.py:579`（`PaddingStrategy`：`LONGEST / MAX_LENGTH / DO_NOT_PAD`，`:585-587`）与
`base.py:154`（`TruncationStrategy`：`ONLY_FIRST / ONLY_SECOND / LONGEST_FIRST / DO_NOT_TRUNCATE`，`:160-163`）。

### 4.2 参数到策略的对照表（必背）

| 你写的 | `padding_strategy` | 实际长度 |
|--------|--------------------|----------|
| `padding=False`（默认） | `DO_NOT_PAD` | 各条自己长度，batch 内不等长 |
| `padding=True` / `"longest"` | `LONGEST` | batch 内最长条的长度 |
| `padding="max_length"` | `MAX_LENGTH` | `max_length`，或 `model_max_length` |
| `padding="max_length", max_length=128` | `MAX_LENGTH` | 128 |
| `padding=True` + `max_length=128`（无 truncation） | `LONGEST` | **128 被忽略**，只补到 batch 最长；且会 `warnings.warn`（`base.py:2353-2357`） |

| 你写的 | `truncation_strategy` | 效果 |
|--------|----------------------|------|
| `truncation=False`（默认） | `DO_NOT_TRUNCATE` | 不截 |
| `truncation=True` | `LONGEST_FIRST` | 单序列=截尾；序列对=先截长的 |
| `truncation="only_first"` / `"only_second"` | `ONLY_FIRST` / `ONLY_SECOND` | 只截第一 / 第二条 |
| 只给 `max_length`（无 padding/truncation 参数） | `LONGEST_FIRST` | 见 4.1 第 1 条规则 |

> **`padding_side` 决定补在哪边**：`"right"`（多数模型默认，`base.py:974`）补在尾，`"left"`
> （`LlamaTokenizer` 覆盖为 `left`，`models/llama/tokenization_llama.py:88`）补在头。生成式模型的
> batch 推理必须用 `left`，否则各条最后一位不是真正的「当前 token」。`truncation_side` 同理，默认 `"right"`。

### 4.3 fast 路径：真正的活是 Rust 干的

`TokenizersBackend._encode_plus`（`:925`）把旧的 `_call_one` / `_batch_encode_plus` 合并成了一个
（源码注释原文标明来源：`# Input validation (from _call_one)` `:946`、`# Batch detection (from _call_one)`
`:974`、`# Set tokenizer configuration (from _batch_encode_plus)` `:1010`），最后一行进 Rust
（`:1027`：`self._tokenizer.encode_batch(batch_text_or_text_pairs, add_special_tokens=..., is_pretokenized=...)`）。
**关键设计**：padding / truncation 不是逐条做的，而是**先写进 backend 的全局状态**，再让 `encode_batch`
一次算完。`set_truncation_and_padding`（`:850`）把策略 `target` dict 交给 backend（`:885-907` 截断 /
`:909-921` 补齐），`truncation_side` 作为 `direction` 在这里生效，`MAX_LENGTH` 给具体 `length`、
`LONGEST` 给 `None`。**这解释了 `padding_side` 的双重身份**：`tokenizer.padding_side = "left"` 是**持久状态**
（改掉 backend）；`tokenizer(..., padding_side="left")` 只是本次调用的临时覆盖——docstring 自认退出 managed
section 后策略会被 reset。

回到 Python 世界，`_convert_encoding`（`:738`）把 Rust 的 `Encoding` 摊平成 dict（`:759-782`，`e.ids` 落在
`:770`）。其中 `return_token_type_ids` / `return_attention_mask` 的默认值来自 `model_input_names`（`:467-469`），
不是写死的：Llama `["input_ids","attention_mask"]`（`models/llama/tokenization_llama.py:89`）→ 不返回
`token_type_ids`；Bert `["input_ids","token_type_ids","attention_mask"]`（`models/bert/tokenization_bert.py:76`）
→ 返回。**这是「哪些 key 出现在返回 dict 里」的真正开关。**

### 4.4 slow 路径：`prepare_for_model` 拆解

slow 的 `_encode_plus`（`tokenization_python.py:702`）笨重得多：batch 时自己写循环逐条调用（`:737-800`），
单条时 `tokenize → convert_tokens_to_ids → prepare_for_model`（`:802-839`）。核心在 `prepare_for_model`（`:1121`）：
`:1150-1157` 再解析一次策略（slow 的 `_encode_plus` 也接受原始参数）；`:1177-1190` 算
`total_len = len(ids) + len(pair_ids) + num_special`（`num_special_tokens_to_add`），超了才 `truncate_sequences`；
`:1192-1198` `build_inputs_with_special_tokens` 加特殊 token + `create_token_type_ids_from_sequences` 生成 type ids；
`:1215-1224` `self.pad(...)` 补 padding。

`truncate_sequences`（`:1231`）有个**必须知道的顺序陷阱**：**先截断、再加特殊 token**，`total_len` 已把
`num_special` 算进去，所以 `max_length=2048` 时 BOS/EOS 占了名额。（对照 fast：Rust 侧
`enable_truncation(max_length=...)` 同样把 post-processor 的特殊 token 计入长度。）

### 4.5 `pad` / `_pad`：padding 的最后一步（base.py:2568 / 2758）

`pad` 处理 batch（`:2731` 逐条调 `_pad` 后拼回），`_pad` 处理单条。`_pad` 的核心（`base.py:2799-2830`）：
`LONGEST` 场景下 `max_length = len(required_input)`（batch 场景 `pad()` 已换成 batch 最长值）；按
`padding_side` 给 `attention_mask` 前/后补 `[0]*difference`，再给 `model_input_names[0]`（即 `input_ids`）
补 `[self.pad_token_id]*difference`；`return_attention_mask` 时先全 1 再补 0。
**`attention_mask` 的语义在这里定死**：真实 token 位 = 1，pad 位 = 0。它不是模型算出来的，也不是可选的——
只要 `"attention_mask" in model_input_names` 就会出现。

### 4.6 返回 dict 的 key 从哪来

| key | 何时出现 | 生成位置 |
|-----|----------|----------|
| `input_ids` | 永远 | fast `_convert_encoding:770` / slow `prepare_for_model:1196` |
| `attention_mask` | `"attention_mask" in model_input_names`（几乎总是） | `_pad:2807-2827` |
| `token_type_ids` | `"token_type_ids" in model_input_names`（Bert 系） | slow `create_token_type_ids_from_sequences:1298`；fast 来自 Rust `e.type_ids` |
| `special_tokens_mask` | `return_special_tokens_mask=True` | `get_special_tokens_mask:1311`（slow）/ Rust（fast） |
| `offset_mapping` | `return_offsets_mapping=True` | **只有 fast 有**（`e.offsets`） |
| `overflowing_tokens` / `num_truncated_tokens` | `return_overflowing_tokens=True` | slow `prepare_for_model:1208-1210`；fast 走 `e.overflowing` |
| `length` / `labels` | `return_length=True` / 传了 `text_target` | 各自实现 / `__call__:2540`（`encodings["labels"] = target_encodings["input_ids"]`） |

---

## 五、special tokens 机制

### 5.1 七个「有名字的特殊 token」+ 一堆「额外特殊 token」

v5 把特殊 token 拆成两类（`base.py:978-988`）：`SPECIAL_TOKENS_ATTRIBUTES = ["bos_token", "eos_token",
"unk_token", "sep_token", "pad_token", "cls_token", "mask_token"]`。

| 类别 | 存哪 | 访问方式 |
|------|------|----------|
| 7 个具名 token | `self._special_tokens_map: dict[str, AddedToken]` | `tokenizer.bos_token` / `tokenizer.bos_token_id` |
| 额外 token（含 `<image>` `<|tool|>` 之类） | `self._extra_special_tokens: list[AddedToken]` | `tokenizer.extra_special_tokens` |
| 模型自定义具名 token（如 `<image_token>`） | `_special_tokens_map`（`_set_model_specific_special_tokens:1401` 动态扩展 `SPECIAL_TOKENS_ATTRIBUTES`） | 同名属性 |

访问靠 `__getattr__` / `__setattr__`（`base.py:1281-1299`）：把 `key` 去掉 `_id` / `_ids` 后缀得到
`key_without_id`，若是具名 token 就从 `_special_tokens_map` 取值——没设则 `logger.error("Using X, but it is
not set yet.")` 并返回 `None`；带 `_id` 后缀时返回 `self.convert_tokens_to_ids(str(token_value))`。

**两条结论**：① `bos_token_id` 不是独立存储字段，每次访问时**现算**（`bos_token` → `convert_tokens_to_ids`），
没设时返回 `None` 并打一行 error；② `tokenizer.bos_token = "<s>"` 会被 `__setattr__:1257` 拦截，写进
`_special_tokens_map`。`special_tokens_map`（`:1344`）只返回**具名的 7 个**（docstring: "**V5 Change**:
This now returns only named tokens"）；全量用 `all_special_tokens`（`:1366`，合并去重 `:1373-1390`）。

### 5.2 BOS/EOS 到底加在哪：两条路径两种做法

**slow**：`build_inputs_with_special_tokens`（`tokenization_python.py:865`），v5 做成**配置驱动**
（docstring `:871-878`：`"none"` / `"cls_sep"` / `"eos"` / `"bos"` / `"bos_eos"` / `"cls_double_sep"` /
`"prefix_suffix"`）；模式由 `self.special_tokens_pattern` 决定（`PythonBackend.__init__:436`，默认 `None` →
落到最后的 `else`，即「不加」）：

```python
# tokenization_python.py:925-936
elif self.special_tokens_pattern == "bos_eos":
    if token_ids_1 is None: return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    return [self.bos_token_id] + token_ids_0 + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]
```

**fast**：在 Rust 的 **post-processor** 里做；`update_post_processor`（`:590`）现场拼模板（`$A:0` = 序列 A、
type_id 0，`<s>:0` = 插入 BOS）；`add_bos_token` / `add_eos_token` 从 `tokenizer_config.json` 读入
（`__init__:417-418`：`self._add_bos_token = kwargs.get("add_bos_token", False)`）：

```python
# tokenization_utils_tokenizers.py:604-611
single = f"{(bos + ':0 ') if self.add_bos_token else ''}$A:0{(' ' + eos + ':0') if self.add_eos_token else ''}"
pair = f"{single}{(' ' + bos + ':1') if self.add_bos_token else ''} $B:1{(' ' + eos + ':1') if self.add_eos_token else ''}"
self._tokenizer.post_processor = processors.TemplateProcessing(single=single, pair=pair, special_tokens=special_tokens)
```

**分水岭**：Llama docstring 写默认 `True`（`models/llama/tokenization_llama.py:77-79`），但该默认值实际来自 Hub 的
`tokenizer_config.json`——**只给一个 `tokenizer.model`、不加载 config 时它就是 `False`，BOS 不会被加。**

| slow | fast |
|---|---|
| `build_inputs_with_special_tokens`；`special_tokens_pattern`；下次 `encode` 生效 | Rust post-processor 模板；`add_bos_token` / `add_eos_token`；需 `update_post_processor()`（setter 已自动挂：`:625-632`） |

### 5.3 `special_tokens_map.json` / `added_tokens.json` 在 v5 是「只读的遗物」

`special_tokens_map.json` **只读不写**（读写矩阵见第七节）。读入时的**优先级**（`base.py:1848-1855`）：
仓库解析出的文件路径覆盖 config 里的同名字段。这段注释记录了一个真实的安全修复：`tokenizer_config.json`
里塞 `vocab_file: /etc/passwd` 会被仓库解析出的真实路径覆盖，只有调用方**显式传的 kwargs** 才能覆盖。
另外 fast 后端不再写 `added_tokens_decoder`（`base.py:2081-2084`），因为 `tokenizer.json` 里已经有了。

### 5.4 为什么 `len(tokenizer)` 可能不等于 `tokenizer.vocab_size`

```python
# fast: tokenization_utils_tokenizers.py:667-671 / :718-722
def vocab_size(self):  """`int`: Size of the base vocabulary (without the added tokens)."""
    return self._tokenizer.get_vocab_size(with_added_tokens=False)
def __len__(self):     """Size of the full vocabulary with the added tokens."""
    return self._tokenizer.get_vocab_size(with_added_tokens=True)
```

slow 侧（`tokenization_python.py:499-509`）同义：`__len__` 返回 `total_vocab_size`，由
`_update_total_vocab_size()` 用 `len(self.get_vocab())` 现算（注释：数 `keys` 不数 `values`，否则 vocab 有洞时
会加错位置；该操作慢，只在加 token 时更新）。即 `vocab_size` = 基础词表（不含 added tokens），
`len(tokenizer)` = 全词表（**含 added / special tokens**）。**什么时候不一致**：`add_special_tokens()` /
`add_tokens()` 加了词表原本没有的 token（Llama 加 `pad_token`、多模态加 `<image>`、工具调用加 `<|tool|>`）。
**后果**：embedding 按 `config.vocab_size` 建，产出 `id >= config.vocab_size` 的 token 会
`IndexError: index out of range`，所以加 token 后**必须**扩模型（`add_special_tokens` docstring 也强调，`base.py:1111-1113`）：

```python
# models/llama/modeling_llama.py:351-353
self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
# modeling_utils.py:2709  resize_token_embeddings
model_embeds = self._resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)
```

传什么：只用内置特殊 token 不用调；`add_special_tokens({"pad_token": ...})` / `add_tokens([...])` 之后传
**`len(tokenizer)`**（不是 `vocab_size`）；想对齐 TP / Tensor Core 的 128 倍数再传 `pad_to_multiple_of=128`。

### 5.5 `add_special_tokens` 内部做了什么（base.py:1102）

`base.py:1173-1207`：`allowed_keys`（`SPECIAL_TOKENS_ATTRIBUTES` + `extra_special_tokens`）校验后，把字符串
包成 `AddedToken(value, rstrip=False, lstrip=False, normalized=False, special=True)`，`setattr(self, key, value)`
（走 `__setattr__` 写进 `_special_tokens_map`），最后 `return self.add_tokens(tokens_to_add, special_tokens=True)`。

1. **它不只是「打个标签」**——`add_tokens(..., special_tokens=True)` 会真的把 token 加进词表，`len(tokenizer)` 会变大。
2. `AddedToken(..., lstrip=False, rstrip=False, normalized=False, special=True)` 决定这个 token **不被切成子词、不被归一化、不吞两侧空格**——这就是 `tokenizer.tokenize("<s>")` 得到 `['<s>']` 的原因。
3. 别名兼容：`additional_special_tokens` 会被改写成 `extra_special_tokens`（`:1165-1171`）。

---

## 六、`chat_template`：现代 LLM 的实际入口

### 6.1 它是什么

`chat_template` 是一个 **Jinja 模板字符串**，存在 tokenizer 上（`base.py:1078-1082`）：
`self.chat_template = kwargs.pop("chat_template", None)`；若是 list/tuple，按
`{template["name"]: template["template"]}` 重建成 dict。两种形态：`str`（单模板，直接用）/
`dict[str, str]`（多模板，如 `{"default": ..., "tool_use": ...}`，`get_chat_template:3247-3262` 按名字选，
传了 `tools` 时优先 `"tool_use"`）。

### 6.2 `apply_chat_template`（base.py:2989）做什么

它**不是**简单的 `str.format`。四步：① `get_chat_template(chat_template, tools)` `:3085` → `:3225` 取出 Jinja
源串（或按名字 / dict 选）；② `render_jinja_template(...)` `:3111-3120` → `utils/chat_template_utils.py:498`
（编译缓存 Jinja；`tools`（dict 或 callable 自动转 JSON Schema，`:517-530`）注入；★
`template_kwargs = {**self.special_tokens_map, **kwargs}` → 模板里能直接用 `{{ bos_token }}`）；③ `tokenize=True`
时调 `self(...)` `:3123-3131`：`out = self(rendered_chat, padding=..., truncation=..., add_special_tokens=False,
return_tensors=...)`；④ `return_dict=True` 返回 `BatchEncoding`，否则返回 `out["input_ids"]`；若
`return_assistant_tokens_mask=True`，再用 `char_to_token` 把 `generation_indices`（字符区间）翻译成 token 级
mask `:3133-3154`。

**两个要点**：**`add_special_tokens=False`**——模板已手写 `<|im_start|>` / `<s>` 等控制符，再自动加 BOS 就重复了
（这是「chat 场景 BOS 会不会翻倍」的答案）；**`return_assistant_tokens_mask`**——只能对含 `{% generation %}`
关键字的模板生效（`utils/chat_template_utils.py:508-511` 会 warning），是 SFT「只对 assistant 回复算 loss」的
基础，靠 `BatchEncoding.char_to_token`（`base.py:541`）把字符区间映射回 token。

### 6.3 为什么要单独讲它

**现代 LLM 训练和推理几乎不再直接调 `tokenizer(text)`**：预训练用 `tokenizer(text)` 或 token 流；
**SFT / 对话推理**用 `tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")`；
工具调用 `apply_chat_template(messages, tools=[...])`；解析输出用 `tokenizer.parse_response(...)`（`:3337`）/
`get_response_parser`（`:3422`）。即 **`__call__` 是底座，`apply_chat_template` 是入口**——后者最终仍回到前者
（`:3122` 那行 `self(...)`），只是先把「对话」渲染成「一段带控制符的文本」。

保存有新格式（`save_chat_templates:3279`，`:3299-3306`）：`save_jinja_files` 且模板是 str 时，单模板存成独立文件
`chat_template.jinja` 并从 `tokenizer_config.json` 里 `pop("chat_template")` 删掉；多模板存成
`additional_chat_templates/<name>.jinja`（`utils/hub.py:68`）。读取时反过来，独立文件**优先于** config 内嵌
（`_from_pretrained:1784-1800`）。

---

## 七、保存与加载：文件格式对照表

### 7.1 各个文件是什么

| 文件 | 格式 | 常量 / 位置 | v5 读 | v5 写 |
|------|------|-------------|-------|-------|
| `tokenizer_config.json` | JSON dict | `base.py:147` | ✅ 主配置 | ✅ `save_pretrained:2110-2113` |
| `tokenizer.json` | JSON（tokenizers 库序列化格式） | `base.py:150` | ✅ fast 主格式 | ✅ `TokenizersBackend._save_pretrained:1141` |
| `vocab.json` | JSON `{token: id}` | 各模型 `vocab_files_names` | ✅ | ✅ slow `save_vocabulary:1362` |
| `vocab.txt` | 纯文本，一行一个 token | `models/bert/tokenization_bert.py:27` | ✅ | ✅ |
| `merges.txt` | 纯文本，一行一个 merge pair（首行可能 `#version: 0.2`） | `models/gpt2/tokenization_gpt2.py:27` | ✅ | ✅ |
| `tokenizer.model` / `spiece.model` | SentencePiece protobuf | Llama `:24` / T5 `:27` | ✅ | ✅ |
| `special_tokens_map.json` | JSON | `base.py:145` | ✅ 旧格式回退（`_from_pretrained:1862-1863`） | ❌ **不再写**（注释 `base.py:1623`：`# kept only for legacy`） |
| `added_tokens.json` | JSON `{token: id}` | `base.py:146` | ✅ 旧格式回退（`_from_pretrained:1889`） | ✅ `_save_pretrained:2157-2166`，仅在 `index >= vocab_size` 时才写 |
| `chat_template.jinja` | 纯文本 Jinja | `utils/hub.py:67` | ✅ 优先级高于 config 内嵌 | ✅ `save_chat_templates:3299` |

- **`tokenizer_config.json`**：类名（`tokenizer_class`）、7 个具名特殊 token、`model_max_length`、`padding_side`、`add_bos_token` / `add_eos_token`、`chat_template`（旧位置）、`added_tokens_decoder`（slow）、`auto_map`（自定义代码）。
- **`tokenizer.json`**：vocab + merges + normalizer + pre_tokenizer + post_processor + decoder + added_tokens，全部在**一个文件**里 —— 这就是 fast 后端读它一个就够的原因。

### 7.2 哪个模型用哪套

| 模型 | `vocab_files_names` | 位置 | 主格式 |
|------|---------------------|------|--------|
| GPT2 | `{"vocab_file":"vocab.json", "merges_file":"merges.txt"}` | `models/gpt2/tokenization_gpt2.py:25-28` | BPE，两文件 |
| Qwen2 | 同上 + `tokenizer_file: tokenizer.json` | `models/qwen2/tokenization_qwen2.py:25-29` | 两者都有，fast 优先 |
| Bert | `{"vocab_file":"vocab.txt", "tokenizer_file":"tokenizer.json"}` | `models/bert/tokenization_bert.py:27` | WordPiece，单文件 |
| Llama | `{"vocab_file":"tokenizer.model", "tokenizer_file":"tokenizer.json"}` | `models/llama/tokenization_llama.py:24` | SPM → 转 BPE backend |
| T5 | `{"vocab_file":"spiece.model", "tokenizer_file":"tokenizer.json"}` | `models/t5/tokenization_t5.py:27` | SPM Unigram |
| `TokenizersBackend` 默认 | `{"tokenizer_file":"tokenizer.json", "vocab_file":"tokenizer.model"}` | `tokenization_utils_tokenizers.py:81` | 后者是 tiktoken 兜底（`:60` `TIKTOKEN_VOCAB_FILE = "tokenizer.model"`） |

### 7.3 加载时的文件解析流程

```
from_pretrained(...)  base.py:1489：vocab_files = {**cls.vocab_files_names, **additional_files_names} :1633
   additional_files_names（:1621-1627）= {added_tokens.json, special_tokens_map.json,
       tokenizer_config.json, tokenizer.json, chat_template.jinja}
   ├─ 版本化 tokenizer.json：config 里若有 "fast_tokenizer_files" → get_fast_tokenizer_file(...) :1653 → :3458
   ├─ chat_template_*.jinja 逐个登记 :1659-1673；兜底正则匹配 tekken.json / tokenizer.model / tiktoken.model :1686-1696
   └─ cls._from_pretrained(resolved_vocab_files, ...) :1736 → :1751
        ├─ 读 tokenizer_config.json → init_kwargs :1765-1775；读 chat_template.jinja（优先于 config 内嵌） :1784-1800
        ├─ added_tokens_decoder / special_tokens_map.json / added_tokens.json 回退 :1845-1920
        ├─ init_kwargs = cls.convert_to_native_format(**init_kwargs) :1929
        └─ tokenizer = cls(*init_inputs, **init_kwargs) :1932
```

**`convert_to_native_format` 是 fast 后端的「万能入口」**（`tokenization_utils_tokenizers.py:103`），按优先级尝试：
① `tokenizer.json` 存在 → `TokenizerFast.from_file` `:113-118`；② 存在但类有自己的 `__init__` → 抽
post_processor / padding / truncation / vocab / merges 再交回 `:119-203`；③ `vocab_file` 是 `tekken.json` →
MistralConverter `:204-209`；④ 以 `.model` 结尾 → SentencePieceExtractor `:211-284`（抽 vocab/merges/
spm_precompiled；有 `convert_from_spm` 就用它重排词表；都没有 → `build_tokenizer_from_spm_proto` 现搭 `:255-258`；
失败 → `TikTokenConverter` `:278-283`）；⑤ 普通 vocab.json / merges.txt `:286-292`；⑥ BPE 但没给 merges →
`generate_merges()` 现场生成 `:294-330`。这也解释了「只给一个 `tokenizer.model` 也能跑」：第 ④ 步现场从
protobuf 抽词表、搭出一个 Rust backend。

### 7.4 保存时写了什么

`save_pretrained`（`base.py:1976`）→ 写 `tokenizer_config.json`（`:2114`），然后 `_save_pretrained`（`:2137`）落盘。
一处 v5 变化（`:2081-2089`）：若 MRO 里有 `TokenizersBackend`，则 `tokenizer_config.pop("added_tokens_decoder")` 与
`pop("additional_special_tokens")`（`tokenizer.json` 里已有）→ fast 目录通常只有 `tokenizer_config.json` +
`tokenizer.json` + `chat_template.jinja`。另一处历史兼容（`:2086-2089`）：类名以 `Fast` 结尾且
`can_save_slow_tokenizer` 为真时去掉后缀再写进 `tokenizer_config["tokenizer_class"]`。`can_save_slow_tokenizer`
（`tokenization_utils_tokenizers.py:496`）：若 `vocab_file` 是 `.model`（sentencepiece），只有原文件还在才返回 `True`。

---

## 八、`convert_slow_tokenizer.py`：没有现成 fast 时怎么造一个

**一句话**：把「词表文件 + 一个模型专属的转换规则类」编译成 `tokenizers.Tokenizer` 对象，让本来只有 slow
实现的模型也能拥有 Rust backend。结构：`Converter` 基类（`:220`，唯一接口 `converted()` `:224`）；各家
`XxxConverter`（Bert `:228` / GPT2 `:422` / Roberta `:535` / …，每个模型一份：怎么拼 normalizer /
pre_tokenizer / decoder）；`SpmConverter`（SPM 系基类 `:634`，子类只重写 `vocab(proto):718` /
`post_processor():789`，v5 新增 `build_tokenizer_from_spm_proto` `:640`、`convert_from_spm` `:688`）；
`TikTokenConverter`（`:1903`，tiktoken 格式如 gpt-oss）；注册表 `SLOW_TO_FAST_CONVERTERS`（`:1981`，
**类名 → 转换器类**，共 55 项）；老入口 `convert_slow_tokenizer(tok, from_tiktoken=False)`（`:2041`）。

注册表形态（`:1981-1984` / `:2034-2038`）：`{"AlbertTokenizer": AlbertConverter, "BartTokenizer":
RobertaConverter, …, "LlamaTokenizer": LlamaConverter, "CodeLlamaTokenizer": LlamaConverter,
"GemmaTokenizer": GemmaConverter, "Phi3Tokenizer": LlamaConverter}`。两个调用点：老入口
（`:2057-2060`，`SLOW_TO_FAST_CONVERTERS[tokenizer_class_name](tok).converted()`）；**v5 主路径**
（`tokenization_utils_tokenizers.py:222-233`，加载 `.model` 时 `SLOW_TO_FAST_CONVERTERS.get(cls.__name__)`
再调 `convert_from_spm(**local_kwargs)` 重排词表）。**为什么需要**：sentencepiece 的 `.model` protobuf 里
piece 排列顺序、`add_dummy_prefix` 开关、`▁` 替换规则各模型不完全一样，需要一份按类名索引的自定义规则。
转换失败会退到 `TikTokenConverter`（`:2069-2080`），再失败才抛错——错误信息里会把
`SLOW_TO_FAST_CONVERTERS` 的键全列出来（`:2080`）。

---

## 九、`AutoTokenizer` 怎么从 `tokenizer_config.json` 选类

呼应 [1A1](./1A1_repo_and_auto.md)：`AutoTokenizer` 与 `AutoModel` 同一思路（用 model type 查映射表），只是
**信息来源更多、优先级更绕**。两个数据结构（`tokenization_auto.py:61` / `:423`）：`TOKENIZER_MAPPING_NAMES =
OrderedDict[str, str | None]([("bert", "BertTokenizer" if is_tokenizers_available() else None), ...])`
（注释 `# V5: Simplified mapping - single tokenizer class per model type`）+ `TOKENIZER_MAPPING =
_LazyAutoMapping(CONFIG_MAPPING_NAMES, TOKENIZER_MAPPING_NAMES)`。`("bert", "BertTokenizer")` 里的值就是第二节
那个统一的类——`BertTokenizer` 本身已是 `TokenizersBackend` 的子类（`models/bert/tokenization_bert.py:41`）。
所以「简化映射」**不是映射表变短**，而是「**一个 model_type 只对应一个类，不再有 slow/fast 二元组**」。

### 9.1 选类优先级（`AutoTokenizer.from_pretrained`，`tokenization_auto.py:656`）

```
入口 :656 │ kwargs.pop("use_fast") 直接丢掉 :739-740
[0] mistral_format=True → MistralCommonBackend :746-750
[1] 显式 tokenizer_type= → TOKENIZER_MAPPING_NAMES 查表 :752-767
[2] config（config.json 或 gguf）→ config.model_type :769-779
[3] 读 tokenizer_config.json → tokenizer_config_class :784-785（= tok_cfg["tokenizer_class"]）；
    tok_cfg["auto_map"]["AutoTokenizer"] → tokenizer_auto_map :788-797
[4] MODEL_IDS_TO_TOKENIZERS_BACKEND 名单命中（deepseek-ai/deepseek-r1-distill-llama-* 等）→ 强制 TokenizersBackend :800-805
[5] Hub 类名 vs 映射表不一致时的裁决 :811-859：config_model_type 在 MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS
    → 用映射表里的类，否则信 Hub 的 :828-834；映射表给 MistralCommonBackend 且可用 → 用它 :844-851；都不行 → TokenizersBackend :853-854
[6] remote code（auto_map + trust_remote_code）：class_ref 优先 auto_map[1]（fast），否则 auto_map[0] :896-908
[7] tokenizer_config_class 查到 → 用它 :920-931（查不到试 "<类名>Fast" :923-924；结果是 PythonBackend →
    强制换成 TokenizersBackend :925-926 ★；还查不到 → TokenizersBackend :928-929）
[8] config.tokenizer_class → 用它（去掉 Fast 后缀） :932-937
[9] 兜底：TOKENIZER_MAPPING.get(type(config), TokenizersBackend) :951-960
```

### 9.2 三个值得记住的点

① `tokenizer_config.json` 的 `tokenizer_class` 不是「唯一真相」：`_from_pretrained` 读进来后立刻扔掉
（`base.py:1771-1772`，注释 `# used in the past to check if the tokenizer class matches the class in the repo`：
`init_kwargs.pop("tokenizer_class", None)`）。真正用它选类的是 `AutoTokenizer`（第 [7] 步）——所以**直接
`LlamaTokenizer.from_pretrained(...)` 时该字段完全不影响结果**，只有走 `AutoTokenizer` 才有意义。

② 「Hub 上的类名写错了」是真实存在、必须在代码里打补丁的问题（`MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS`，
`models/auto/tokenization_auto.py:371`）：`:811-859` 在「信 Hub 类名」和「信本地映射表」之间权衡，并把
`PythonBackend` 硬改写成 `TokenizersBackend`（`:925-926`，另一处兜底在 `:970`）——这就是「v5 强制走 fast」：

```python
# models/auto/tokenization_auto.py:925-926
if tokenizer_class is not None and tokenizer_class.__name__ == "PythonBackend":
    tokenizer_class = TokenizersBackend
```

③ 后缀 `Fast` 被当作噪音剥掉（`:864-865`，`tokenizer_config_class[:-4]`）：老 checkpoint 里存的
`"tokenizer_class": "LlamaTokenizerFast"` 剥掉后缀后正好命中 `LlamaTokenizer`。

---

## 一句话总结

> 一个基类 `PreTrainedTokenizerBase` 定流程，三个后端 `TokenizersBackend` / `PythonBackend` /
> `SentencePieceBackend` 只实现 `_encode_plus` 等几个钩子。`__call__` 先把 padding / truncation 归一成两个枚举
> （`_get_padding_truncation_strategies:2336`），fast 把策略写进 Rust backend 后一次 `encode_batch`，slow 走
> `prepare_for_model:1121` 的「算长度 → 截断 → 加特殊 token → 补齐 → BatchEncoding」五步；返回的 key 由
> `model_input_names` 决定，BOS/EOS 在 fast 里由 post-processor 模板加、在 slow 里由
> `build_inputs_with_special_tokens` 加；`vocab_size` 是基础词表、`len(tokenizer)` 含 added tokens，不一致时必须
> `resize_token_embeddings(len(tokenizer))`；现代 LLM 的真正入口已不是 `__call__`，是把它当底座用的 `apply_chat_template`。

---

## 验证清单

以下命令全部在仓库根目录 `/Users/baai/Desktop/baai_proj/transformers` 实跑过。

```bash
# ── 0. 四个（+1）源文件的行数 ──────────────────────────────────
wc -l src/transformers/tokenization_utils_base.py \
      src/transformers/tokenization_utils_tokenizers.py \
      src/transformers/tokenization_utils_sentencepiece.py \
      src/transformers/tokenization_python.py \
      src/transformers/convert_slow_tokenizer.py
# → 3687 / 1478 / 315 / 1424 / 2081

# ── 1. 类的定义位置与基类 ─────────────────────────────────────
grep -n "^class " src/transformers/tokenization_utils_base.py \
                    src/transformers/tokenization_utils_tokenizers.py \
                    src/transformers/tokenization_utils_sentencepiece.py \
                    src/transformers/tokenization_python.py
# → tokenization_utils_base.py:195:class BatchEncoding(UserDict, Generic[_V]):
# → tokenization_utils_base.py:962:class PreTrainedTokenizerBase(PushToHubMixin):
# → tokenization_utils_tokenizers.py:85:class TokenizersBackend(PreTrainedTokenizerBase):
# → tokenization_utils_sentencepiece.py:45:class SentencePieceBackend(PreTrainedTokenizer):
# → tokenization_python.py:400:class PythonBackend(PreTrainedTokenizerBase):

# ── 2. 「别名」而非「两个类」──────────────────────────────────
tail -3 src/transformers/tokenization_python.py
# → # Backward compatibility alias
# → PreTrainedTokenizer = PythonBackend
tail -2 src/transformers/tokenization_utils_tokenizers.py
# → # Backward-compatible alias: allow referring to TokenizersBackend as PreTrainedTokenizerFast
# → PreTrainedTokenizerFast = TokenizersBackend
tail -2 src/transformers/models/llama/tokenization_llama.py
# → # Backward alias
# → LlamaTokenizerFast = LlamaTokenizer

# ── 3. 模型侧继承分布（说明「现代模型都走 fast」）─────────────
grep -rho "^class [A-Za-z0-9_]*(\([A-Za-z0-9_, .]*\))" src/transformers/models/*/tokenization_*.py \
  | sed 's/.*(//' | sort | uniq -c | sort -rn | head
# → 50 TokenizersBackend)  32 PreTrainedTokenizer)  6 SentencePieceBackend)
# →  6 BertTokenizer)       2 PreTrainedTokenizerFast)  ...
grep -rn "^[A-Za-z0-9_]*Fast = " src/transformers/models/*/tokenization_*.py | wc -l
# → 14（都是别名，不是类定义）

# ── 3b. slow 不支持对齐类 API ────────────────────────────────
sed -n '368,369p' src/transformers/tokenization_utils_base.py
# →         if not self._encodings:
# →             raise ValueError("word_ids() is not available when using Python based tokenizers")

# ── 4. use_fast 参数在 v5 被丢弃、映射表已简化 ────────────────
sed -n '61p'    src/transformers/models/auto/tokenization_auto.py
# → # V5: Simplified mapping - single tokenizer class per model type (always prefer tokenizers-based)
sed -n '739,740p' src/transformers/models/auto/tokenization_auto.py
# →        # V5: Always use fast tokenizers, ignore use_fast parameter
# →        _ = kwargs.pop("use_fast", None)

# ── 5. __call__ 与各内部函数的确切行号 ────────────────────────
grep -n "def _get_padding_truncation_strategies\|def __call__\|def _encode_plus\|def pad(\|def _pad(" \
     src/transformers/tokenization_utils_base.py
# → 2336 / 2418 / 2543 / 2568 / 2758
#   （注意不能用 "def pad"，它会连 1254: def pad_token_type_id 一起匹配）
sed -n '2566p' src/transformers/tokenization_utils_base.py   # → raise NotImplementedError（基类模板方法）

# ── 6. fast / slow 的 _encode_plus 与关键调用点 ───────────────
grep -n "def _encode_plus\|self.set_truncation_and_padding(\|self._tokenizer.encode_batch(\|def set_truncation_and_padding" \
     src/transformers/tokenization_utils_tokenizers.py
# → 850 / 925 / 1010 / 1027
grep -n "def _encode_plus\|def prepare_for_model\|def truncate_sequences\|def build_inputs_with_special_tokens\|def create_token_type_ids_from_sequences" \
     src/transformers/tokenization_python.py
# → 702 / 865 / 1121 / 1231 / 1298

# ── 7. vocab_size vs len() 的语义差异 ────────────────────────
grep -n "def vocab_size\|def __len__" src/transformers/tokenization_utils_tokenizers.py
# → 667 / 718
sed -n '669,670p;720,721p' src/transformers/tokenization_utils_tokenizers.py
# → """`int`: Size of the base vocabulary (without the added tokens)."""
# → """Size of the full vocabulary with the added tokens."""
#   with_added_tokens=False :669 / with_added_tokens=True :721

# ── 8. BOS/EOS 的两条实现路径 ────────────────────────────────
grep -n "def update_post_processor" -A 20 src/transformers/tokenization_utils_tokenizers.py | head -25
# → 590；604-611 行拼 post_processor 模板
grep -n "def build_inputs_with_special_tokens" src/transformers/tokenization_python.py
# → 865

# ── 9. chat_template 的位置与保存格式 ────────────────────────
grep -n "def apply_chat_template\|def get_chat_template\|def save_chat_templates\|def encode_message_with_chat_template" \
     src/transformers/tokenization_utils_base.py
# → 2989 / 3165 / 3225 / 3279
grep -n "CHAT_TEMPLATE_FILE\s*=\|CHAT_TEMPLATE_DIR\s*=" src/transformers/utils/hub.py
# → 67:CHAT_TEMPLATE_FILE = "chat_template.jinja"
# → 68:CHAT_TEMPLATE_DIR = "additional_chat_templates"

# ── 10. 文件名常量 ──────────────────────────────────────────
grep -n "^SPECIAL_TOKENS_MAP_FILE\|^ADDED_TOKENS_FILE\|^TOKENIZER_CONFIG_FILE\|^FULL_TOKENIZER_FILE\|^VERY_LARGE_INTEGER" \
     src/transformers/tokenization_utils_base.py
# → 130 / 145 / 146 / 147 / 150
grep -n "^TOKENIZER_FILE\|^TIKTOKEN_VOCAB_FILE\|^VOCAB_FILES_NAMES" src/transformers/tokenization_utils_tokenizers.py
# → 57 / 60 / 81

# ── 11. 枚举定义 ────────────────────────────────────────────
grep -n "class PaddingStrategy" -A 6 src/transformers/utils/generic.py
# → 579；LONGEST/MAX_LENGTH/DO_NOT_PAD 在 585-587
sed -n '154,163p' src/transformers/tokenization_utils_base.py   # TruncationStrategy

# ── 12. 模型 vocab_files_names ──────────────────────────────
grep -n "VOCAB_FILES_NAMES" src/transformers/models/llama/tokenization_llama.py \
      src/transformers/models/t5/tokenization_t5.py src/transformers/models/bert/tokenization_bert.py \
      src/transformers/models/gpt2/tokenization_gpt2.py src/transformers/models/qwen2/tokenization_qwen2.py
# → llama:24 {vocab_file: tokenizer.model, tokenizer_file: tokenizer.json}
# → t5:27    {vocab_file: spiece.model,   tokenizer_file: tokenizer.json}
# → bert:27  {vocab_file: vocab.txt,      tokenizer_file: tokenizer.json}
# → gpt2:25  {vocab_file: vocab.json, merges_file: merges.txt}
# → qwen2:25 {vocab_file: vocab.json, merges_file: merges.txt, tokenizer_file: tokenizer.json}

# ── 13. model_input_names 决定返回哪些 key ──────────────────
grep -n "model_input_names" src/transformers/models/llama/tokenization_llama.py \
      src/transformers/models/bert/tokenization_bert.py
# → llama:89  ["input_ids", "attention_mask"]          （无 token_type_ids）
# → bert:76   ["input_ids", "token_type_ids", "attention_mask"]

# ── 14. convert_slow_tokenizer 的注册表与入口 ────────────────
grep -n "^SLOW_TO_FAST_CONVERTERS\|^def convert_slow_tokenizer" src/transformers/convert_slow_tokenizer.py
# → 1981 / 2041
sed -n '1981,1983p' src/transformers/convert_slow_tokenizer.py
# → SLOW_TO_FAST_CONVERTERS = {
# →     "AlbertTokenizer": AlbertConverter,
# →     "BartTokenizer": RobertaConverter,
sed -n '2034,2038p' src/transformers/convert_slow_tokenizer.py
# →     "LlamaTokenizer": LlamaConverter,
# →     "CodeLlamaTokenizer": LlamaConverter,
# →     "GemmaTokenizer": GemmaConverter,
# →     "Phi3Tokenizer": LlamaConverter,
# → }
grep -n "SLOW_TO_FAST_CONVERTERS" src/transformers/tokenization_utils_tokenizers.py
# → 222（v5 主路径：加载 .model 时按类名找 convert_from_spm）

# ── 15. AutoTokenizer 选类链路的关键行 ──────────────────────
grep -n "MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS: set\|TOKENIZER_MAPPING = \|MODEL_IDS_TO_TOKENIZERS_BACKEND = \|def tokenizer_class_from_name\|class AutoTokenizer" \
     src/transformers/models/auto/tokenization_auto.py
# → 371 / 423 / 427 / 490 / 640
grep -n 'tokenizer_class.__name__ == "PythonBackend"' src/transformers/models/auto/tokenization_auto.py
# → 925（主裁决分支）、970（最终兜底分支）：两处都强制换成 TokenizersBackend

# ── 16. embedding 尺寸与 resize ─────────────────────────────
grep -n "self.embed_tokens = nn.Embedding" src/transformers/models/llama/modeling_llama.py
# → 353:  self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
grep -n "def resize_token_embeddings" src/transformers/modeling_utils.py
# → 2709

# ── 17. 一个副产物结论：TokenizersBackend._post_init 在当前仓库主流程无调用点 ──
grep -rn "_post_init" src/transformers/tokenization_utils_tokenizers.py
# → 634:    def _post_init(self):
# → 640:        Child classes should call super()._post_init() ...（docstring 说由 from_pretrained 调用）
grep -rn "\._post_init()" src/transformers/ | grep -v "def _post_init"
# → tokenization_utils_tokenizers.py:640（就是那句 docstring 本身，不是调用）
# → models/cohere/tokenization_cohere.py:184        self._post_init()
# → models/mluke/tokenization_mluke.py:391          self._post_init()
# → models/mluke/tokenization_mluke.py:397          super()._post_init()
# → models/reformer/tokenization_reformer.py:112    super()._post_init()
# → models/rembert/tokenization_rembert.py:172      super()._post_init()
# 结论：只在个别老模型的 tokenization_*.py 里被显式调用，
#       主流程（from_pretrained / __init__）里没有调用点。
#       TokenizersBackend._post_init 更像是留给自定义/远程代码的钩子，v5 主流程不再自动调它。
```

---

## 导航

- 路线图：[00_roadmap.md](./00_roadmap.md)（本篇 = 入门 · 架构 · 1A4）
- 前置：[1A1 仓库骨架与 auto 机制](./1A1_repo_and_auto.md)（`AutoTokenizer` 与 `AutoModel` 共用同一套映射机制）
- 上一篇：[1A3_from_pretrained.md](./1A3_from_pretrained.md)
- 下一篇：[1A5_generate_loop.md](./1A5_generate_loop.md)
  （本篇产出的 `input_ids` / `attention_mask` 正是它的输入）
- 相关：
  [1M1_llama_structure.md](./1M1_llama_structure.md)（`embed_tokens` 的输入就是本篇的输出）
  · [2A5_modular_code_generation.md](./2A5_modular_code_generation.md)
  （tokenizer 也能由 modular 文件生成：`TYPE_TO_FILE_TYPE` 里有 `Tokenizer` → `tokenization_*.py`）
