# Tokenizer 体系精读：fast / slow 两套与 `__call__` 全链路

> 阅读目标：搞清楚「一段文本怎么变成 `input_ids`」——
> 共同的基类是谁、fast / slow 两条路径各自走到哪、padding / truncation / special tokens
> 在代码的哪一行被决定，以及 `chat_template` 为什么成了现代 LLM 的实际入口。
>
> 源文件：`src/transformers/tokenization_utils_base.py`（3687 行，基类 + 共享逻辑）、
> `src/transformers/tokenization_utils_tokenizers.py`（1478 行，fast）、
> `src/transformers/tokenization_python.py`（1424 行，slow）、
> `src/transformers/tokenization_utils_sentencepiece.py`（315 行，slow 的 sentencepiece 变体）、
> `src/transformers/convert_slow_tokenizer.py`（2081 行，slow → fast 转换器）
>
> **一句话定位**：这一篇讲的是「数据怎么进模型」的第一道门。它的输出
> `input_ids` / `attention_mask` 就是 1A5 `generate` 的输入。

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

- [1A1 仓库骨架与 auto 机制](./1A1_repo_and_auto.md)（待写）：`models/<name>/` 五件套里的 `tokenization_*.py` 到底是什么；
  `AutoTokenizer` 和 `AutoModel` 用的是同一套「映射表 + 运行时选类」的机制（本篇第九节细讲）。
  *（该文件尚未创建，文件名以最终落地为准。）*
- [1M1_llama_structure.md](./1M1_llama_structure.md)：`nn.Embedding(config.vocab_size, hidden_size)`
  是 tokenizer 的下游——tokenizer 的 `len()` 与 `config.vocab_size` 一旦对不上，模型就会在 embedding 处越界。

---

## 一、先纠正一个常见误解：v5 里「fast / slow」已经收编成「后端」

你大概听过这句话：

> transformers 里有两套 tokenizer：slow 是纯 Python 实现，fast 是 Rust 的 `tokenizers` 库包装。

这句话**历史正确、现状已经过时**。本仓库（v5 主线）里：

| 说法 | 老版本（v4 及以前） | 本仓库现状 |
|------|---------------------|-----------|
| 基类 | `PreTrainedTokenizerBase` | **不变**，`tokenization_utils_base.py:962` |
| slow 基类 | `PreTrainedTokenizer`（类） | `PythonBackend`（类），`tokenization_python.py:400`；旧名做别名保留 |
| fast 基类 | `PreTrainedTokenizerFast`（类） | `TokenizersBackend`（类），`tokenization_utils_tokenizers.py:85`；旧名做别名保留 |
| 模型 tokenizer 文件 | `tokenization_x.py` + `tokenization_x_fast.py` 两个文件 | **只剩 `tokenization_x.py`**，文件末尾挂别名 |
| `AutoTokenizer(use_fast=...)` | 有效参数 | **被吞掉**，不再起作用 |

两个别名就写在文件末尾，一行：

```python
# src/transformers/tokenization_python.py:1423-1424
# Backward compatibility alias
PreTrainedTokenizer = PythonBackend
```

```python
# src/transformers/tokenization_utils_tokenizers.py:1477-1478
# Backward-compatible alias: allow referring to TokenizersBackend as PreTrainedTokenizerFast
PreTrainedTokenizerFast = TokenizersBackend
```

模型侧也是一样，`LlamaTokenizerFast` 只是 `LlamaTokenizer` 的别名
（`models/llama/tokenization_llama.py:149`）：

```python
# Backward alias
LlamaTokenizerFast = LlamaTokenizer
```

**所以在本仓库里读源码时**：

- 看到 `class XXXTokenizer(PreTrainedTokenizer)` → 它继承的是 `PythonBackend`（slow 路径）。
- 看到 `class XXXTokenizer(TokenizersBackend)` → 走 Rust（fast 路径）。
- 看到 `XXXTokenizerFast = XXXTokenizer` → 只是兼容别名，**不是另一个类**，别去找它的实现。

### 实测的分布（`grep` 数出来的）

| 继承写法 | 类数量 | 运行时实际基类 |
|----------|--------|----------------|
| `(TokenizersBackend)` | 50 | `TokenizersBackend`（Rust） |
| `(PreTrainedTokenizer)` | 32 | `PythonBackend`（纯 Python，写的是旧别名） |
| `(SentencePieceBackend)` | 6 | `SentencePieceBackend` → `PythonBackend` |
| `(PreTrainedTokenizerFast)` | 2 | `TokenizersBackend`（旧别名） |
| `(BertTokenizer)` 等具体类 | ~10 | 视父类而定 |
| `XXXFast = XXX`（别名，不是类定义） | 14 | 同左值 |

结论：**绝大多数现代模型（Llama / Qwen / GPT2 / Bert / T5）已经统一到 `TokenizersBackend`**；
还留在 slow 路径上的多是老模型（BioGpt、CTRL、M2M100、Marian、XLM、Tapas、Wav2Vec2CTC…）。

### 为什么当初会有两套，以及为什么现在收编了

| | slow（Python 实现） | fast（Rust 实现） |
|---|---|---|
| 实现语言 | 纯 Python，逻辑在 `tokenization_python.py` | 编译好的 Rust，Python 侧只做胶水 |
| 词表算法 | 自己实现 BPE / WordPiece / Unigram / SentencePiece | `tokenizers` 库内置 |
| 批处理 | Python 循环，逐条处理 | `encode_batch` 一次进 Rust，多线程 |
| **offset mapping** | 不支持（slow `_encode_plus:702` 连 `return_offsets_mapping` 参数都没有） | 原生支持 char↔token 对齐 |
| 与原文对齐 | — | `word_ids` / `token_to_chars` / `char_to_token` 等全套 |
| 序列化 | 多个文件（vocab / merges / tokenizer.model + config） | 单个 `tokenizer.json` |
| 可读性 | 高（能读 Python 源码） | 低（逻辑在 Rust 里，只能读配置） |

slow 下调用这些对齐 API 会直接报错（`base.py:368-369`）：

```python
if not self._encodings:
    raise ValueError("word_ids() is not available when using Python based tokenizers")
```

历史原因很直白：早期 transformers 只有 Python 版，速度慢、没有 offset；引入
HuggingFace 的 `tokenizers` 库后，为了**不破坏已有权重/代码**，两条线并存了很久，
于是每个模型出现 `tokenization_x.py` + `tokenization_x_fast.py` 两个文件。

v5 的收编思路是：**保留 fast 的实现能力，保留 slow 的名字**。证据在 AutoTokenizer 里：

```python
# src/transformers/models/auto/tokenization_auto.py:739-740
# V5: Always use fast tokenizers, ignore use_fast parameter
_ = kwargs.pop("use_fast", None)
```

以及映射表头部那句注释：

```python
# src/transformers/models/auto/tokenization_auto.py:61
# V5: Simplified mapping - single tokenizer class per model type (always prefer tokenizers-based)
```

意思是：映射表从「每个模型 type → (slow 类, fast 类) 二元组」变成**一个类**。

---

## 二、类继承层级图

```
                              PushToHubMixin
                                    │
┌───────────────────────────────────────────────────────────────────────┐
│  PreTrainedTokenizerBase                tokenization_utils_base.py:962 │
│  ── 「所有 tokenizer 后端的基类」（docstring 原文：Base class for all    │
│     tokenizer backends）                                                │
│                                                                        │
│  类属性：                                                               │
│    vocab_files_names: dict[str, str] = {}            :967              │
│    model_input_names = ["input_ids","attention_mask"] :973              │
│    padding_side = "right"  :974    truncation_side = "right"  :975      │
│    SPECIAL_TOKENS_ATTRIBUTES = [bos,eos,unk,sep,pad,cls,mask]  :980     │
│                                                                        │
│  模板方法（子类必须实现）：                                              │
│    _encode_plus        :2543  → raise NotImplementedError  :2566       │
│    _add_tokens         :1250  → raise NotImplementedError              │
│    vocab_size          :1438  → raise NotImplementedError              │
│    __len__             :1434  → raise NotImplementedError              │
│    added_tokens_decoder:1419  → raise NotImplementedError              │
│    save_vocabulary     :2193  → raise NotImplementedError              │
│                                                                        │
│  具体方法（所有后端共享）：                                              │
│    __call__                                :2418  ← 唯一公共入口        │
│    _get_padding_truncation_strategies      :2336                       │
│    pad                                     :2568                       │
│    _pad                                    :2758                       │
│    add_special_tokens                      :1102                       │
│    add_tokens                              :1209                       │
│    special_tokens_map / all_special_tokens :1344 / :1366               │
│    from_pretrained / _from_pretrained      :1489 / :1751               │
│    save_pretrained / _save_pretrained      :1976 / :2137               │
│    apply_chat_template                     :2989                       │
│    get_chat_template                       :3225                       │
└───────────────────────────────────────────────────────────────────────┘
        │                                                    │
        │ 后端分支 1                                          │ 后端分支 2 / 3
        ▼                                                    ▼
┌────────────────────────────────────┐   ┌────────────────────────────────────────┐
│ TokenizersBackend                  │   │ PythonBackend                          │
│ tokenization_utils_tokenizers.py:85│   │ tokenization_python.py:400             │
│ 别   名：PreTrainedTokenizerFast:1478│   │ 别   名：PreTrainedTokenizer   :1424   │
│ Rust tokenizers 库                  │   │ 纯 Python                              │
│                                    │   │                                        │
│ is_fast → True             :492    │   │ is_fast → False              :454       │
│ __init__                   :328    │   │ __init__                     :413       │
│ convert_to_native_format   :103    │   │ __len__                      :499       │
│ update_post_processor      :590    │   │ _add_tokens                  :516       │
│ vocab_size                 :667    │   │ tokenize                     :625       │
│ __len__                    :718    │   │ _encode_plus                 :702       │
│ _convert_encoding          :738    │   │ build_inputs_with_special_…  :865       │
│ _add_tokens                :794    │   │ get_special_tokens_mask      :969       │
│ set_truncation_and_padding :850    │   │ convert_ids_to_tokens        :1055      │
│ _encode_plus               :925    │   │ prepare_for_model            :1121      │
│ _decode                    :1086   │   │ truncate_sequences           :1231      │
│ _save_pretrained           :1129   │   │ create_token_type_ids_…      :1298      │
│ train_new_from_iterator    :1146   │   │ save_vocabulary              :1362      │
│ 序列化 → tokenizer.json            │   │ 序列化 → vocab/merges/…                │
└────────────────────────────────────┘   └────────────────────────────────────────┘
                                                      │
                                                      │ 继承（是 PythonBackend 的子类）
                                                      ▼
                                         ┌────────────────────────────────────────┐
                                         │ SentencePieceBackend                   │
                                         │ tokenization_utils_sentencepiece.py:45 │
                                         │ class SentencePieceBackend(            │
                                         │     PreTrainedTokenizer)   ← :26 导入   │
                                         │ 背靠 sentencepiece 库（spm）             │
                                         │ sp_model / vocab_size :100              │
                                         │ save_vocabulary       :237              │
                                         └────────────────────────────────────────┘

旁支（同层，不属于以上三支）：
  MistralCommonBackend   tokenization_mistral_common.py:186
  （Mistral 自家 mistral-common 包的后端，AutoTokenizer 会按 repo 特征选它）
```

三条分支的**关键区别就一句话**：

> `PreTrainedTokenizerBase` 定义了「**流程**」（策略解析、padding、truncation、special tokens 编排、
> 存取盘），三个后端只负责实现「**流程里的两三个钩子**」（`_encode_plus`、`_add_tokens`、
> `vocab_size`、`__len__`、`added_tokens_decoder`）。这是标准的模板方法模式。

---

## 三、四个文件各自的角色

| 文件 | 行数 | 角色 | 关键类/函数 |
|------|------|------|------------|
| `tokenization_utils_base.py` | 3687 | **公共基类 + 公共流程**。`BatchEncoding` 也在这里 | `PreTrainedTokenizerBase:962`、`BatchEncoding:195`、`__call__:2418` |
| `tokenization_utils_tokenizers.py` | 1478 | **fast 后端**（Rust `tokenizers`） | `TokenizersBackend:85`、`_encode_plus:925`、`set_truncation_and_padding:850` |
| `tokenization_python.py` | 1424 | **slow 后端**（纯 Python），含 `Trie` 等工具 | `PythonBackend:400`、`prepare_for_model:1121`、`truncate_sequences:1231` |
| `tokenization_utils_sentencepiece.py` | 315 | slow 的 **sentencepiece 变体**（继承 `PythonBackend`） | `SentencePieceBackend:45`、`SentencePieceExtractor:289` |
| `convert_slow_tokenizer.py` | 2081 | **slow → fast 转换器**（不参与日常 encode） | `SLOW_TO_FAST_CONVERTERS:1981`、`convert_slow_tokenizer:2041` |

---

## 四、`__call__` 全流程（用户最常走的路径）

### 4.0 流程图

```
  tokenizer("Hello world", padding=True, truncation=True, return_tensors="pt")
        │
        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ [1] PreTrainedTokenizerBase.__call__                base.py:2418         │
│     ① 把 20+ 个具名参数塞进 all_kwargs dict             :2469-2486        │
│     ② all_kwargs.update(tokenizer_kwargs) → update(kwargs)              │
│        （显式参数优先级最高）                            :2492-2494        │
│     ③ 解析策略：_get_padding_truncation_strategies      :2498  → :2336   │
│        入: padding=True / truncation=True / max_length=None             │
│        出: PaddingStrategy.LONGEST, TruncationStrategy.LONGEST_FIRST,   │
│            max_length=model_max_length                                  │
│     ④ if text is not None:  self._encode_plus(...)      :2512           │
│        if text_target is not None: 再来一遍 → labels    :2523           │
└──────────────────────────────────────────────────────────────────────────┘
        │
        │  self._encode_plus  ← 多态！取决于后端
        │
        ├───────────────────────────────┬───────────────────────────────────┐
        ▼                               ▼                                   ▼
┌──────────────────────────┐  ┌──────────────────────────────┐  ┌────────────────────┐
│ fast: TokenizersBackend  │  │ slow: PythonBackend          │  │ SentencePieceBackend│
│ _encode_plus  :925       │  │ _encode_plus  :702           │  │ 继承 slow，不重写    │
└──────────────────────────┘  └──────────────────────────────┘  └────────────────────┘
        │                               │
        ▼                               ▼
① 校验 text 形状                    ① 判定 is_batched
    _is_valid_text_input  :946          isinstance(text, (list,tuple))  :724
② 判定 is_batched          :979     ② 逐条递归调 _encode_plus
③ set_truncation_and_padding           （batch 内 padding 推迟到最后）
      :1010 → :850                   ③ 批量 pad  :790
    把 padding/truncation 写进              （PaddingStrategy.LONGEST
    Rust backend 的全局状态                 在这一层才解析成具体长度）
④ self._tokenizer.encode_batch  :1027
     ★ 真正干活的是 Rust                   单条路径：
⑤ _convert_encoding  :1034 → :738        get_input_ids → tokenize → convert_tokens_to_ids :818
    拆成 input_ids / attention_mask     ④ self.prepare_for_model(...)  :1121
    / token_type_ids / offset_mapping       a) _get_padding_truncation_strategies  :1150
                                            b) num_special_tokens_to_add  :1179
                                            c) truncate_sequences  :1184 → :1231
                                            d) build_inputs_with_special_tokens  :1194 → :865
                                            e) create_token_type_ids_from_sequences  :1195
                                            f) self.pad(...)  :1217 → base:2568
                                            g) BatchEncoding(...)  :1229
        │                               │
        └───────────────┬───────────────┘
                        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ [2] BatchEncoding            base.py:195                                 │
│     · dict 子类（UserDict），键 = input_ids / attention_mask / ...        │
│     · convert_to_tensors  :665  ← 在这里把 list 变成 torch.Tensor         │
│     · is_fast  :306  （= self._encodings is not None）                    │
│     · char_to_token / word_ids / token_to_chars 等只在 fast 下有值         │
└──────────────────────────────────────────────────────────────────────────┘
        │
        ▼
   {"input_ids": tensor[...], "attention_mask": tensor[...]}  →  喂给模型
```

### 4.1 第一步：`_get_padding_truncation_strategies`（base.py:2336）

这是**所有 padding/truncation 语义的唯一裁判**。`__call__` 里 20 多个参数先在这里坍缩成
两个枚举 + 一个 `max_length`。

五条规则（全部实测自 `base.py:2336-2418`）：

```python
# :2342-2346  只给 max_length → 自动开启截断（历史包袱）
if max_length is not None and padding is False and truncation is None:
    truncation = "longest_first"

# :2348-2363  padding 归一化
if padding is not False:
    if padding is True:
        padding_strategy = PaddingStrategy.LONGEST       # True → 补到 batch 内最长
    elif not isinstance(padding, PaddingStrategy):
        padding_strategy = PaddingStrategy(padding)      # 字符串 "max_length" / "longest"
else:
    padding_strategy = PaddingStrategy.DO_NOT_PAD

# :2366-2376  truncation 归一化
if truncation is True:
    truncation_strategy = TruncationStrategy.LONGEST_FIRST

# :2378-2390  没给 max_length 时，从 model_max_length 取；模型不限长则反向关掉策略
if max_length is None:
    if padding_strategy == PaddingStrategy.MAX_LENGTH:
        if self.model_max_length > LARGE_INTEGER: padding_strategy = PaddingStrategy.DO_NOT_PAD
        else:                                    max_length = self.model_max_length
    if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE:
        if self.model_max_length > LARGE_INTEGER: truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE
        else:                                     max_length = self.model_max_length

# :2391-2396  没有任何 pad_token 时直接报错
if padding_strategy != PaddingStrategy.DO_NOT_PAD and (self.pad_token is None or self.pad_token_id < 0):
    raise ValueError("Asking to pad but the tokenizer does not have a padding token. ...")
```

枚举定义在 `utils/generic.py:579` / `tokenization_utils_base.py:154`：

| `PaddingStrategy`（`utils/generic.py:585-587`） | `TruncationStrategy`（`base.py:160-163`） |
|----------------------------------------------|------------------------------------------|
| `LONGEST = "longest"` | `ONLY_FIRST = "only_first"` |
| `MAX_LENGTH = "max_length"` | `ONLY_SECOND = "only_second"` |
| `DO_NOT_PAD = "do_not_pad"` | `LONGEST_FIRST = "longest_first"` |
| | `DO_NOT_TRUNCATE = "do_not_truncate"` |

### 4.2 参数到策略的对照表（必背）

| 你写的 | `padding_strategy` | 实际长度 |
|--------|--------------------|----------|
| `padding=False`（默认） | `DO_NOT_PAD` | 各条自己长度，batch 内不等长 |
| `padding=True` | `LONGEST` | batch 内最长条的长度 |
| `padding="longest"` | `LONGEST` | 同上 |
| `padding="max_length"` | `MAX_LENGTH` | `max_length`，或 `model_max_length` |
| `padding="max_length", max_length=128` | `MAX_LENGTH` | 128 |
| `padding=True` + `max_length=128`（无 truncation） | `LONGEST` | **128 被忽略**，只补到 batch 最长；且会 `warnings.warn`（`base.py:2353-2357`） |

| 你写的 | `truncation_strategy` | 效果 |
|--------|----------------------|------|
| `truncation=False`（默认） | `DO_NOT_TRUNCATE` | 不截 |
| `truncation=True` | `LONGEST_FIRST` | 单序列=截尾；序列对=先截长的 |
| `truncation="only_first"` | `ONLY_FIRST` | 只截第一条 |
| `truncation="only_second"` | `ONLY_SECOND` | 只截第二条 |
| 只给 `max_length`（无 padding/truncation 参数） | `LONGEST_FIRST` | 见上面第 1 条规则 |

> **`padding_side` 决定补在哪边**：`"right"`（Llama 之外多数模型的默认，`base.py:974`）
> 补在尾，`"left"`（`LlamaTokenizer` 覆盖为 `left`，`models/llama/tokenization_llama.py:88`）
> 补在头。生成式模型的 batch 推理必须用 `left`，否则 batch 里各条的最后一位不是真正的「当前 token」。
> `truncation_side` 同理，默认 `"right"`（截尾）。

### 4.3 fast 路径：真正的活是 Rust 干的

`TokenizersBackend._encode_plus`（`tokenization_utils_tokenizers.py:925`）把 old 的
`_call_one` / `_batch_encode_plus` 两个方法合并成了一个（注意源码里的注释原文）：

```python
# tokenization_utils_tokenizers.py:946, 974, 1010, 1027
# Input validation (from _call_one)     ← 注释明确写了「来自 _call_one」
# Batch detection (from _call_one)
# Set tokenizer configuration (from _batch_encode_plus)   ← 「来自 _batch_encode_plus」
self.set_truncation_and_padding(...)

# Direct rust backend call             ← 一行进 Rust
encodings = self._tokenizer.encode_batch(
    batch_text_or_text_pairs,
    add_special_tokens=add_special_tokens,
    is_pretokenized=is_split_into_words,
)
```

**关键设计**：padding / truncation 不是逐条做的，而是**先写进 backend 的全局状态**，
再让 `encode_batch` 一次算完。`set_truncation_and_padding`（`tokenization_utils_tokenizers.py:850`）做的就是这个：

```python
# tokenization_utils_tokenizers.py:885-907 / :909-921  —— 策略写进 backend 的全局状态
_truncation = self._tokenizer.truncation
if truncation_strategy == TruncationStrategy.DO_NOT_TRUNCATE:
    if _truncation is not None: self._tokenizer.no_truncation()
else:
    target = {"max_length": max_length, "stride": stride,
              "strategy": truncation_strategy.value,
              "direction": self.truncation_side}      # ← truncation_side 在这里生效
    if current != target: self._tokenizer.enable_truncation(**target)

if padding_strategy == PaddingStrategy.DO_NOT_PAD:
    if _padding is not None: self._tokenizer.no_padding()
else:
    length = max_length if padding_strategy == PaddingStrategy.MAX_LENGTH else None
    target = {"length": length,                  # ← MAX_LENGTH 给具体长度；LONGEST 给 None
              "direction": padding_side if padding_side is not None else self.padding_side,
              "pad_id": self.pad_token_id, "pad_token": self.pad_token,
              "pad_type_id": self.pad_token_type_id, "pad_to_multiple_of": pad_to_multiple_of}
    if _padding != target: self._tokenizer.enable_padding(**target)
```

**这解释了一个容易困惑的现象**：`tokenizer.padding_side = "left"` 是**可变状态**，
它会持久地改掉 backend；而 `tokenizer(..., padding_side="left")` 只是本次调用内的临时覆盖。
`set_truncation_and_padding` 的 docstring 自己承认了这点：

> The provided tokenizer has no padding / truncation strategy before the managed section.
> If your tokenizer set a padding / truncation strategy before, then it will be reset to
> no padding / truncation when exiting the managed section.

回到 Python 世界后，`_convert_encoding`（`:738`）负责把 Rust 的 `Encoding` 摊平成 dict：

```python
# tokenization_utils_tokenizers.py:759-782（节选）
if return_token_type_ids is None:
    return_token_type_ids = "token_type_ids" in self.model_input_names
if return_attention_mask is None:
    return_attention_mask = "attention_mask" in self.model_input_names
if return_overflowing_tokens and encoding.overflowing is not None:
    encodings = [encoding] + encoding.overflowing
else:
    encodings = [encoding]

encoding_dict = defaultdict(list)
for e in encodings:
    encoding_dict["input_ids"].append(e.ids)                # :770
    if return_token_type_ids:
        encoding_dict["token_type_ids"].append(e.type_ids)
    if return_attention_mask:
        encoding_dict["attention_mask"].append(e.attention_mask)
    if return_special_tokens_mask:
        encoding_dict["special_tokens_mask"].append(e.special_tokens_mask)
    if return_offsets_mapping:
        encoding_dict["offset_mapping"].append(e.offsets)
    if return_length:
        encoding_dict["length"].append(len(e.ids))
```

**注意 `return_token_type_ids` / `return_attention_mask` 的默认值来源**：不是写死的，
而是看 `self.model_input_names`。Llama 的 `model_input_names = ["input_ids", "attention_mask"]`
（`models/llama/tokenization_llama.py:89`）→ 不会返回 `token_type_ids`；
Bert 的 `model_input_names = ["input_ids", "token_type_ids", "attention_mask"]`
（`models/bert/tokenization_bert.py:76`）→ 会返回。**这就是「哪些 key 出现在返回 dict 里」的真正开关。**

### 4.4 slow 路径：`prepare_for_model` 拆解

slow 的 `_encode_plus`（`tokenization_python.py:702`）逻辑要笨重得多：batch 情况下
自己写循环逐条调用（`:737-800`），单条情况下 `tokenize → convert_tokens_to_ids →
prepare_for_model`（`:802-839`）。

核心在 `prepare_for_model`（`:1121`）。这是 slow 路径真正的「组装车间」：

```python
# tokenization_python.py:1150-1157 —— 再解析一次策略（因为 slow 的 _encode_plus 也接受原始参数）
padding_strategy, truncation_strategy, max_length, _ = self._get_padding_truncation_strategies(...)

# tokenization_python.py:1177-1190 —— 算上 special token 的总长，超了才截断
pair = pair_ids is not None
num_special = self.num_special_tokens_to_add(pair=pair) if add_special_tokens else 0
total_len = len(ids) + len(pair_ids or []) + num_special

overflowing_tokens = []
if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length and total_len > max_length:
    ids, pair_ids, overflowing_tokens = self.truncate_sequences(
        ids,
        pair_ids=pair_ids,
        num_tokens_to_remove=total_len - max_length,
        truncation_strategy=truncation_strategy,
        stride=stride,
    )

# tokenization_python.py:1192-1198 —— 加特殊 token + 生成 token_type_ids
if add_special_tokens:
    sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
    token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
else:
    sequence = ids + (pair_ids if pair_ids else [])
    token_type_ids = [0] * len(sequence)

# tokenization_python.py:1215-1224 —— 补 padding
if padding_strategy != PaddingStrategy.DO_NOT_PAD or return_attention_mask:
    encoded_inputs = self.pad(encoded_inputs, max_length=max_length, ...)
```

`truncate_sequences`（`:1231`）里有个**必须知道的顺序陷阱**：它是**先截断、再加特殊 token**。
`total_len` 已经把 `num_special` 算进去了，所以 `max_length=2048` 时，BOS/EOS 占了名额。
（对照 fast：Rust 侧 `enable_truncation(max_length=...)` 同样是把 post-processor 的
特殊 token 计入长度的。）

### 4.5 `pad` / `_pad`：padding 的最后一步（base.py:2568 / 2758）

`pad` 处理 batch（`:2731` 逐条调 `_pad` 后拼回），`_pad` 处理单条。`_pad` 的核心二十行：

```python
# tokenization_utils_base.py:2799-2830（节选）
if padding_strategy == PaddingStrategy.LONGEST:
    max_length = len(required_input)          # ← batch 场景下 pad() 已把 max_length 换成了 batch 最长值

if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
    max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

if return_attention_mask and "attention_mask" not in encoded_inputs:
    encoded_inputs["attention_mask"] = [1] * len(required_input)   # 先全 1，再补 0

if needs_to_be_padded:
    difference = max_length - len(required_input)
    padding_side = padding_side if padding_side is not None else self.padding_side
    if padding_side == "right":
        if return_attention_mask:
            encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"] + [0] * difference
        ...
        encoded_inputs[self.model_input_names[0]] = required_input + [self.pad_token_id] * difference
    elif padding_side == "left":
        if return_attention_mask:
            encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
        ...
        encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input
```

**`attention_mask` 的语义在这里定死**：真实 token 位 = 1，pad 位 = 0。
它既不是模型算出来的，也不是可选的——只要 `"attention_mask" in model_input_names` 就会出现。

### 4.6 返回 dict 的 key 从哪来

| key | 何时出现 | 生成位置 |
|-----|----------|----------|
| `input_ids` | 永远 | fast `_convert_encoding:770` / slow `prepare_for_model:1196` |
| `attention_mask` | `"attention_mask" in model_input_names`（几乎总是） | `_pad:2807-2827` |
| `token_type_ids` | `"token_type_ids" in model_input_names`（Bert 系） | slow `create_token_type_ids_from_sequences:1298`；fast 来自 Rust `e.type_ids` |
| `special_tokens_mask` | `return_special_tokens_mask=True` | `get_special_tokens_mask:1311`（slow）/ Rust（fast） |
| `offset_mapping` | `return_offsets_mapping=True` | **只有 fast 有**（`e.offsets`） |
| `overflowing_tokens` / `num_truncated_tokens` | `return_overflowing_tokens=True` | slow `prepare_for_model:1208-1210`；fast 走 `e.overflowing` |
| `length` | `return_length=True` | 各自实现 |
| `labels` | 传了 `text_target` | `__call__:2540`（`encodings["labels"] = target_encodings["input_ids"]`） |

---

## 五、special tokens 机制

### 5.1 七个「有名字的特殊 token」+ 一堆「额外特殊 token」

v5 把特殊 token 明确拆成两类（`base.py:978-988`）：

```python
# Special tokens support (moved from SpecialTokensMixin)
# V5: Clean separation of named special tokens from extra special tokens
SPECIAL_TOKENS_ATTRIBUTES = [
    "bos_token", "eos_token", "unk_token", "sep_token",
    "pad_token", "cls_token", "mask_token",
]
```

| 类别 | 存哪 | 访问方式 |
|------|------|----------|
| 7 个具名 token | `self._special_tokens_map: dict[str, AddedToken]` | `tokenizer.bos_token` / `tokenizer.bos_token_id` |
| 额外 token（含 `<image>` `<|tool|>` 之类） | `self._extra_special_tokens: list[AddedToken]` | `tokenizer.extra_special_tokens` |
| 模型自定义具名 token（如 `<image_token>`） | `_special_tokens_map`（`_set_model_specific_special_tokens:1401` 动态扩展 `SPECIAL_TOKENS_ATTRIBUTES`） | 同名属性 |

访问全靠 `__getattr__` / `__setattr__` 这对魔法方法：

```python
# tokenization_utils_base.py:1281-1299（__getattr__）
def __getattr__(self, key):
    # Handle _id/_ids suffix (eg. bos_token_id -> bos_token)
    key_without_id = key.removesuffix("_ids").removesuffix("_id") if key.endswith(("_id", "_ids")) else key

    if key_without_id in self.SPECIAL_TOKENS_ATTRIBUTES:
        token_value = self.__dict__.get("_special_tokens_map", {}).get(key_without_id)
        if token_value is None:
            if self.verbose:
                logger.error(f"Using {key}, but it is not set yet.")
            return None
        return self.convert_tokens_to_ids(str(token_value)) if key != key_without_id else str(token_value)
```

**两条实用结论**：

1. `tokenizer.bos_token_id` 不是独立存储的字段——它是每次访问时**现算**的
   （`bos_token` → `convert_tokens_to_ids`）。所以 `bos_token` 没设时
   `bos_token_id` 返回 `None`，而且日志里会打一行 error。
2. `tokenizer.bos_token = "<s>"` 会被 `__setattr__:1257` 拦截，写进 `_special_tokens_map`
   而不是变成普通实例属性。

`special_tokens_map`（`:1344`）只返回**具名的 7 个**（v5 的行为变化，docstring 明说
"**V5 Change**: This now returns only named tokens"）；想要全量用 `all_special_tokens`（`:1366`），
它把具名 + extra 合并去重（`:1373-1390`）。

### 5.2 BOS/EOS 到底加在哪：两条路径两种做法

**slow 路径**：调用 `build_inputs_with_special_tokens`（`tokenization_python.py:865`）。
v5 把它做成了**配置驱动**，不再为每个模型手写一份：

```python
# tokenization_python.py:871-878（docstring 摘要）
# - "none":          No special tokens
# - "cls_sep":       [CLS] seq0 [SEP] 或 [CLS] seq0 [SEP] seq1 [SEP]
# - "eos":           seq0 [EOS] 或 seq0 [EOS] seq1 [EOS]
# - "bos":           [BOS] seq0 或 [BOS] seq0 [BOS] seq1
# - "bos_eos":       [BOS] seq0 [EOS] 或 [BOS] seq0 [EOS] seq1 [EOS]
# - "cls_double_sep":[CLS] seq0 [SEP] 或 [CLS] seq0 [SEP] [SEP] seq1 [SEP]
# - "prefix_suffix": <prefix_tokens> seq0 [seq1] <suffix_tokens>
```

分支体就是直接拼 list：

```python
# tokenization_python.py:925-936
elif self.special_tokens_pattern == "bos_eos":
    if token_ids_1 is None:
        return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    return [self.bos_token_id] + token_ids_0 + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]
```

模式由 `self.special_tokens_pattern` 决定（在 `PythonBackend.__init__:436` 从 kwargs 里取，
默认 `None` → 落到最后的 `else` 分支，即「不加」）。

**fast 路径**：完全不走 Python，而是在 Rust 的 **post-processor** 里做。
`update_post_processor`（`tokenization_utils_tokenizers.py:590`）现场拼一个模板：

```python
# tokenization_utils_tokenizers.py:604-611
single = f"{(bos + ':0 ') if self.add_bos_token else ''}$A:0{(' ' + eos + ':0') if self.add_eos_token else ''}"
pair = f"{single}{(' ' + bos + ':1') if self.add_bos_token else ''} $B:1{(' ' + eos + ':1') if self.add_eos_token else ''}"

special_tokens = []
if self.add_bos_token:
    special_tokens.append((bos, bos_token_id))
if self.add_eos_token:
    special_tokens.append((eos, eos_token_id))
self._tokenizer.post_processor = processors.TemplateProcessing(
    single=single, pair=pair, special_tokens=special_tokens
)
```

模板字符串 `$A:0` = 序列 A、type_id 0；`<s>:0` = 插入 BOS。`add_bos_token` / `add_eos_token`
是从 `tokenizer_config.json` 读进来的布尔量（`__init__:417-418`）：

```python
# tokenization_utils_tokenizers.py:417-418
explicit_bos_eos_in_kwargs = "add_bos_token" in kwargs or "add_eos_token" in kwargs
self._add_bos_token = kwargs.get("add_bos_token", False)
```

**这就是「同一个模型 BOS 加没加」的分水岭**：Llama 的 docstring 写 `add_bos_token` 默认
`True`（`models/llama/tokenization_llama.py:77-79`），但这个默认值实际来自 Hub 上的
`tokenizer_config.json`。**你本地如果只给一个 `tokenizer.model`、不加载 config，
`add_bos_token` 就是 `False`，BOS 不会被加。**（行为差异的经典来源。）

| | slow | fast |
|---|---|---|
| 谁加 BOS/EOS | Python 的 `build_inputs_with_special_tokens` | Rust post-processor 模板 |
| 配置来源 | `special_tokens_pattern` | `add_bos_token` / `add_eos_token` |
| 改配置后 | 下次 `encode` 自动生效 | 需要 `update_post_processor()`（setter 里已自动挂：`:625-632`） |

### 5.3 `special_tokens_map.json` / `added_tokens.json` 在 v5 是「只读的遗物」

v4 时代保存 tokenizer 会写出四五个文件，v5 里 `special_tokens_map.json` **只读不写**了
（完整读写矩阵见第七节的表）。唯一要记住的是读入时的**优先级**：

```python
# tokenization_utils_base.py:1848-1855 —— 仓库解析出的文件路径覆盖 config 里的同名字段
for args_name, file_path in resolved_vocab_files.items():
    # `init_kwargs` also carries the values loaded from the (untrusted) `tokenizer_config.json`,
    # which `save_pretrained` never serializes for these vocab-file arguments. A value present
    # here therefore originates from the config and would be opened verbatim, so it could point
    # at an arbitrary location outside the repository (path traversal, CWE-22). Let the
    # repo-resolved path take precedence; only an explicit caller-provided path (in `kwargs`)
    # is allowed to override it.
    if args_name not in kwargs or kwargs[args_name] is None:
        init_kwargs[args_name] = file_path
```

> 这段注释记录了一个真实的安全修复：`tokenizer_config.json` 里如果塞了 `vocab_file: /etc/passwd`
> 这种路径，会被仓库解析出的真实路径覆盖掉；只有调用方**显式传的 kwargs** 才能覆盖。
> 只有 fast 后端不再写 `added_tokens_decoder`（`base.py:2081-2084`），因为 `tokenizer.json` 里已经有了。

### 5.4 为什么 `len(tokenizer)` 可能不等于 `tokenizer.vocab_size`

这个区别是**最容易踩的坑**，两边都实现了，但语义不同：

```python
# fast: tokenization_utils_tokenizers.py:667-671
@property
def vocab_size(self) -> int:
    """`int`: Size of the base vocabulary (without the added tokens)."""
    return self._tokenizer.get_vocab_size(with_added_tokens=False)

# fast: tokenization_utils_tokenizers.py:718-722
def __len__(self) -> int:
    """Size of the full vocabulary with the added tokens."""
    return self._tokenizer.get_vocab_size(with_added_tokens=True)
```

```python
# slow: tokenization_python.py:499-509
def __len__(self):
    """Size of the full vocabulary with the added tokens."""
    if self.total_vocab_size == 0:
        self._update_total_vocab_size()
    return self.total_vocab_size

def _update_total_vocab_size(self):
    # Counts the `keys` and not the `values` because otherwise if there is a hole in the vocab,
    # we will add tokenizers at a wrong index. This operation is slow and is only updated when adding tokens.
    self.total_vocab_size = len(self.get_vocab())
```

| | 含义 | 是否含 added tokens |
|---|------|---------------------|
| `vocab_size` | 基础词表大小 | ❌ |
| `len(tokenizer)` | 全词表大小（**含 added / special tokens**） | ✅ |

**什么时候会不一致**：只要你 `add_special_tokens()` / `add_tokens()` 加了词表里原本没有的 token。
典型场景：给 Llama 加 `pad_token`、给多模态模型加 `<image>`、给工具调用模型加 `<|tool|>`。

**不一致的后果**：模型侧的 embedding 是按 `config.vocab_size` 建的

```python
# src/transformers/models/llama/modeling_llama.py:351-353
self.vocab_size = config.vocab_size
...
self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
```

如果 tokenizer 产出了 `id >= config.vocab_size` 的 token，`embed_tokens` 直接
`IndexError: index out of range`。所以加了 token 之后**必须**扩模型：

```python
# modeling_utils.py:2709  resize_token_embeddings
model_embeds = self._resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)
```

`add_special_tokens` 的 docstring 里也反复强调这件事（`base.py:1111-1113`）：

> When adding new tokens to the vocabulary, you should make sure to also resize the token embedding
> matrix of the model so that its embedding matrix matches the tokenizer.
> In order to do that, please use the [`~PreTrainedModel.resize_token_embeddings`] method.

| 场景 | 传什么给 `resize_token_embeddings` |
|------|-----------------------------------|
| 只用内置特殊 token，没新增 | 不用调 |
| `add_special_tokens({"pad_token": ...})` / `add_tokens([...])` 之后 | **`len(tokenizer)`**（不是 `vocab_size`） |
| 想对齐 TP / Tensor Core 的 128 倍数 | 额外传 `pad_to_multiple_of=128` |

### 5.5 `add_special_tokens` 内部做了什么（base.py:1102）

```python
# tokenization_utils_base.py:1173-1207（节选）
allowed_keys = set(self.SPECIAL_TOKENS_ATTRIBUTES) | {"extra_special_tokens"}
tokens_to_add = []
for key, value in special_tokens_dict.items():
    if key not in allowed_keys:
        raise ValueError(f"Key {key} is not a valid special token. Valid keys are: {allowed_keys}")
    ...
    if isinstance(value, str):
        value = AddedToken(value, rstrip=False, lstrip=False, normalized=False, special=True)
    setattr(self, key, value)       # ← 走 __setattr__，写进 _special_tokens_map
    tokens_to_add.append(value)

return self.add_tokens(tokens_to_add, special_tokens=True)
```

三个要点：

1. **它不只是「打个标签」**——`add_tokens(..., special_tokens=True)` 会真的把 token 加进词表，
   所以 `len(tokenizer)` 会变大。
2. `AddedToken(..., lstrip=False, rstrip=False, normalized=False, special=True)` 这组参数
   决定这个 token **不会被切成子词、不会被归一化、不吞两侧空格**。这就是
   「`tokenizer.tokenize("<s>")` 得到 `['<s>']` 而不是 `['<','s','>']`」的原因。
3. 别名兼容：`additional_special_tokens` 会被改写成 `extra_special_tokens`（`:1165-1171`）。

---

## 六、`chat_template`：现代 LLM 的实际入口

### 6.1 它是什么

`chat_template` 是一个 **Jinja 模板字符串**，存在 tokenizer 上（`base.py:1078`）：

```python
# tokenization_utils_base.py:1078-1082
self.chat_template = kwargs.pop("chat_template", None)
if isinstance(self.chat_template, (list, tuple)):
    # Chat templates are stored as lists of dicts with fixed key names,
    # we reconstruct that into a single dict while loading them.
    self.chat_template = {template["name"]: template["template"] for template in self.chat_template}
```

它有**两种形态**：

| 形态 | 内容 | 选择逻辑 |
|------|------|----------|
| `str` | 单个模板 | 直接使用 |
| `dict[str, str]` | 多模板，如 `{"default": ..., "tool_use": ...}` | `get_chat_template:3247-3262` 按名字选；传了 `tools` 时优先 `"tool_use"` |

### 6.2 `apply_chat_template`（base.py:2989）做什么

它**不是**简单的 `str.format`。四步：

```
conversation: [{"role":"user","content":"hi"}, ...]
        │
        ▼
① get_chat_template(chat_template, tools)         :3085 → :3225
      从 tokenizer.chat_template 取出 Jinja 源串（或按名字/dict 选）
        │
        ▼
② render_jinja_template(conversations, tools, documents, ...)  :3111-3120
      utils/chat_template_utils.py:498
      · 编译缓存 Jinja（_compile_jinja_template，同一模板不重复编译）
      · 把 tools（dict 或 callable 都能自动转 JSON Schema，:517-530）注入
      · 返回 (rendered_chat, generation_indices)
      ★ template_kwargs = {**self.special_tokens_map, **kwargs}
        —— 所以模板里能直接用 {{ bos_token }} / {{ eos_token }}
        │
        ▼
③ tokenize=True 时，调 self(...)                  :3123-3131
      out = self(rendered_chat, padding=..., truncation=...,
                 add_special_tokens=False,          ← 注意：特殊 token 已由模板显式写入
                 return_tensors=...)
        │
        ▼
④ return_dict=True 时返回 BatchEncoding，否则返回 out["input_ids"]
   若 return_assistant_tokens_mask=True，再用 char_to_token 把
   generation_indices（字符区间）翻译成 token 级 mask  :3133-3154
```

**三个设计要点**：

- **`add_special_tokens=False`**：模板里已经手写了 `<|im_start|>` / `<s>` 这类控制符，
  再让 tokenizer 自动加 BOS 就重复了。这是「chat 场景 BOS 会不会翻倍」的答案。
- **`return_assistant_tokens_mask`**：只能对含 `{% generation %}` 关键字的模板生效
  （`utils/chat_template_utils.py:508-511` 会 warning）。它是 SFT 阶段「只对 assistant 回复
  算 loss」的实现基础——靠 `BatchEncoding.char_to_token`（`base.py:541`）把字符区间映射回 token。

### 6.3 为什么要单独讲它

因为**现代 LLM 训练和推理几乎不再直接调 `tokenizer(text)`**：

| 场景 | 实际调用 |
|------|----------|
| 预训练 / 继续预训练 | `tokenizer(text)` 或直接喂 token 流 |
| **SFT / 对话推理** | `tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")` |
| 工具调用 | `apply_chat_template(messages, tools=[...])` |
| 解析模型输出成结构 | `tokenizer.parse_response(...)`（`:3337`）/ `get_response_parser`（`:3422`） |

也就是说：**`__call__` 是底座，`apply_chat_template` 是入口**。
后者最终还是会回到前者（`:3122` 那一行 `self(...)`），只是先把「对话」渲染成「一段带控制符的文本」。

保存时有个新格式的变化（`save_chat_templates:3279`）：

```python
# tokenization_utils_base.py:3299-3306
if save_jinja_files and isinstance(self.chat_template, str):
    # New format for single templates is to save them as chat_template.jinja
    ...
    saved_raw_chat_template_files.append(chat_template_file)
    if "chat_template" in tokenizer_config:
        tokenizer_config.pop("chat_template")  # To ensure it doesn't somehow end up in the config too
```

→ 单个模板存成独立文件 `chat_template.jinja` 并从 `tokenizer_config.json` 里**删掉**；
多个模板存成 `additional_chat_templates/<name>.jinja`（`utils/hub.py:68` 定义了目录名）。
读取时反过来，独立文件**优先于** config 内嵌（`_from_pretrained:1784-1800`）。

---

## 七、保存与加载：文件格式对照表

### 7.1 各个文件是什么

| 文件 | 格式 | 常量 / 位置 | v5 读 | v5 写 |
|------|------|-------------|-------|-------|
| `tokenizer_config.json` | JSON dict | `base.py:147` | ✅ 主配置 | ✅ `save_pretrained:2110-2113` |
| `tokenizer.json` | JSON（tokenizers 库的序列化格式） | `base.py:150` | ✅ fast 主格式 | ✅ `TokenizersBackend._save_pretrained:1141` |
| `vocab.json` | JSON `{token: id}` | 各模型 `vocab_files_names` | ✅ | ✅ slow `save_vocabulary:1362` |
| `vocab.txt` | 纯文本，一行一个 token | `models/bert/tokenization_bert.py:27` | ✅ | ✅ |
| `merges.txt` | 纯文本，一行一个 merge pair（首行可能 `#version: 0.2`） | `models/gpt2/tokenization_gpt2.py:27` | ✅ | ✅ |
| `tokenizer.model` / `spiece.model` | SentencePiece protobuf | Llama `:24` / T5 `:27` | ✅ | ✅ |
| `special_tokens_map.json` | JSON | `base.py:145` | ✅ 旧格式回退（`_from_pretrained:1862-1863`） | ❌ **不再写**（注释 `base.py:1623`：`# kept only for legacy`） |
| `added_tokens.json` | JSON `{token: id}` | `base.py:146` | ✅ 旧格式回退（`_from_pretrained:1889`） | ✅ `_save_pretrained:2157-2166`，仅在 `index >= vocab_size` 时才写 |
| `chat_template.jinja` | 纯文本 Jinja | `utils/hub.py:67` | ✅ 优先级高于 config 内嵌 | ✅ `save_chat_templates:3299` |

各文件装了什么：

- **`tokenizer_config.json`**：类名（`tokenizer_class`）、7 个具名特殊 token、`model_max_length`、
  `padding_side`、`add_bos_token` / `add_eos_token`、`chat_template`（旧位置）、
  `added_tokens_decoder`（slow）、`auto_map`（自定义代码）。
- **`tokenizer.json`**：vocab + merges + normalizer + pre_tokenizer + post_processor + decoder +
  added_tokens，全部在**一个文件**里 —— 这就是为什么 fast 后端读它一个就够。

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
from_pretrained(...)                                  base.py:1489
   ├─ vocab_files = {**cls.vocab_files_names, **additional_files_names}   :1633
   │     additional_files_names（:1621-1627）= {
   │         added_tokens_file: added_tokens.json, special_tokens_map_file: special_tokens_map.json,
   │         tokenizer_config_file: tokenizer_config.json, tokenizer_file: tokenizer.json,
   │         chat_template_file: chat_template.jinja }
   ├─ 版本化 tokenizer.json：config 里若有 "fast_tokenizer_files"，
   │    用 get_fast_tokenizer_file(...) 选合适版本        :1653 → :3458
   ├─ chat_template_*.jinja 逐个登记                      :1659-1673
   ├─ 兜底：tokenizer.json 不在 repo 时正则匹配
   │    tekken.json / tokenizer.model / tiktoken.model    :1686-1696
   └─ resolved_vocab_files = {key: cached_file(...)}
         ↓
   cls._from_pretrained(resolved_vocab_files, ...)      :1736 → :1751
         ├─ 读 tokenizer_config.json → init_kwargs       :1765-1775
         ├─ 读 chat_template.jinja（优先于 config 内嵌）  :1784-1800
         ├─ added_tokens_decoder / special_tokens_map.json / added_tokens.json 回退  :1845-1920
         ├─ init_kwargs = cls.convert_to_native_format(**init_kwargs)             :1929
         └─ tokenizer = cls(*init_inputs, **init_kwargs)                          :1932
```

**`convert_to_native_format` 是 fast 后端的「万能入口」**（`tokenization_utils_tokenizers.py:103`）。
它按优先级依次尝试：

```
① tokenizer.json 存在 → TokenizerFast.from_file          :113-118
② tokenizer.json 存在但类有自己的 __init__ → 抽出
   post_processor / padding / truncation / vocab / merges 再交回去   :119-203
③ vocab_file 是 tekken.json → MistralConverter            :204-209
④ vocab_file 以 .model 结尾 → SentencePieceExtractor      :211-284
     ├─ 抽 vocab / merges / spm_precompiled
     ├─ 若有 SLOW_TO_FAST_CONVERTERS 对应的 convert_from_spm → 用它重排词表
     ├─ 类自己有 convert_from_spm_model → 再用它调
     ├─ 都没有 → SpmConverter.build_tokenizer_from_spm_proto 现搭一个      :255-258
     └─ 失败 → TikTokenConverter 兜底                       :278-283
⑤ 普通 vocab.json / merges.txt                            :286-292
⑥ BPE 但没给 merges → generate_merges() 现场生成            :294-330
```

> 这也解释了「为什么给一个只有 `tokenizer.model` 的目录，`LlamaTokenizer.from_pretrained`
> 也能跑」：第 ④ 步现场从 protobuf 里抽词表、搭出一个 Rust backend。

### 7.4 保存时写了什么

`save_pretrained`（`base.py:1976`）→ 写 `tokenizer_config.json`（`:2114`），
然后 `_save_pretrained`（`:2137`）落盘。有一处 v5 的变化值得注意：

```python
# tokenization_utils_base.py:2081-2089（节选）
# tokenizers backend don't need to save added_tokens_decoder and additional_special_tokens
if any(base.__name__ == "TokenizersBackend" for base in self.__class__.__mro__):
    tokenizer_config.pop("added_tokens_decoder", None)
    tokenizer_config.pop("additional_special_tokens", None)
```

→ **fast 后端不再把 `added_tokens_decoder` 写进 config**，因为 `tokenizer.json` 里已经有了。
（所以 fast 存出来的目录通常只有 `tokenizer_config.json` + `tokenizer.json` + `chat_template.jinja`。）

另一个历史兼容点——保存的类名会去掉 `Fast` 后缀（`:2086-2089`），前提是能存 slow 格式：

```python
# tokenization_utils_base.py:2086-2089
# Remove the Fast at the end if we can save the slow tokenizer
if tokenizer_class.endswith("Fast") and getattr(self, "can_save_slow_tokenizer", False):
    tokenizer_class = tokenizer_class[:-4]
tokenizer_config["tokenizer_class"] = tokenizer_class
```

`can_save_slow_tokenizer`（`tokenization_utils_tokenizers.py:496`）的逻辑：如果
`vocab_file` 是 `.model`（sentencepiece），只有原文件还在才返回 `True`。

---

## 八、`convert_slow_tokenizer.py`：没有现成 fast 时怎么造一个

**一句话**：它把「词表文件 + 一个模型专属的转换规则类」编译成 `tokenizers.Tokenizer` 对象，
让本来只有 slow 实现的模型也能拥有 Rust backend。

结构就三层：

| 层 | 位置 | 作用 |
|----|------|------|
| `Converter` | `:220`，接口 `converted()` 在 `:224` | 基类，唯一接口 |
| 各家 `XxxConverter` | Bert `:228` / GPT2 `:422` / Roberta `:535` / … | 每个模型一份：怎么拼 normalizer / pre_tokenizer / decoder |
| `SpmConverter`（SentencePiece 系的基类） | `:634` | 子类只重写 `vocab(proto):718` / `post_processor():789` 等钩子；v5 新增了 `build_tokenizer_from_spm_proto` `:640` 和 `convert_from_spm` `:688` |
| `TikTokenConverter` | `:1903` | tiktoken 格式（如 gpt-oss）|
| `SLOW_TO_FAST_CONVERTERS` | `:1981` | 注册表：**类名 → 转换器类**，共 55 项 |
| `convert_slow_tokenizer(tok, from_tiktoken=False)` | `:2041` | 老入口 |

注册表形态（节选）：

```python
# convert_slow_tokenizer.py:1981-1984
SLOW_TO_FAST_CONVERTERS = {
    "AlbertTokenizer": AlbertConverter,
    "BartTokenizer": RobertaConverter,
    ...
# convert_slow_tokenizer.py:2034-2038
    "LlamaTokenizer": LlamaConverter,
    "CodeLlamaTokenizer": LlamaConverter,
    "GemmaTokenizer": GemmaConverter,
    "Phi3Tokenizer": LlamaConverter,
}
```

两个调用点：

```python
# 1) 老入口：直接给一个 slow tokenizer 实例，换一个 Tokenizer 出来
# convert_slow_tokenizer.py:2057-2060
tokenizer_class_name = transformer_tokenizer.__class__.__name__
if tokenizer_class_name in SLOW_TO_FAST_CONVERTERS and not from_tiktoken:
    converter_class = SLOW_TO_FAST_CONVERTERS[tokenizer_class_name]
    return converter_class(transformer_tokenizer).converted()
```

```python
# 2) v5 的实际主路径：加载 .model 文件时按类名找 converter 重排词表
# tokenization_utils_tokenizers.py:222-233
from .convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS
converter_class = SLOW_TO_FAST_CONVERTERS.get(cls.__name__)
if converter_class is not None and hasattr(converter_class, "convert_from_spm"):
    local_kwargs = converter_class.convert_from_spm(**local_kwargs)
```

**为什么需要它**：sentencepiece 的 `.model` protobuf 里 piece 的排列顺序、`add_dummy_prefix`
开关、`▁` 替换规则，各模型都不完全一样；把「SPM 的语义」翻译成「tokenizers 库的配置」
需要一份按类名索引的自定义规则。转换失败会退到 `TikTokenConverter`（`:2069-2080`），
再失败才抛错——错误信息里会把 `SLOW_TO_FAST_CONVERTERS` 的键全列出来（`:2080`）。

---

## 九、`AutoTokenizer` 怎么从 `tokenizer_config.json` 选类

呼应 [1A1](./1A1_repo_and_auto.md) 的 auto 机制：`AutoModel` 用 `config.model_type`
查 `CONFIG_MAPPING`，`AutoTokenizer` 的思路一样，只是**信息来源更多、优先级更绕**。

### 9.1 两个数据结构

```python
# models/auto/tokenization_auto.py:61
# V5: Simplified mapping - single tokenizer class per model type (always prefer tokenizers-based)
TOKENIZER_MAPPING_NAMES = OrderedDict[str, str | None]([
    ("EvollaModel", "TokenizersBackend" if is_tokenizers_available() else None),
    ("aimv2",       "CLIPTokenizer"   if is_tokenizers_available() else None),
    ("bert",        "BertTokenizer"   if is_tokenizers_available() else None),
    ...
])

# models/auto/tokenization_auto.py:423
TOKENIZER_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, TOKENIZER_MAPPING_NAMES)
```

注意 `("bert", "BertTokenizer")` 里的值就是**我们第二节看到的那个统一的类**——
`BertTokenizer` 本身已经是 `TokenizersBackend` 的子类（`models/bert/tokenization_bert.py:41`）。
所以「简化映射」的意思不是「映射表变短」，而是「**一个 model_type 只对应一个类，不再有 slow/fast 二元组**」。

### 9.2 选类优先级（`AutoTokenizer.from_pretrained`，`tokenization_auto.py:656`）

```
入口: AutoTokenizer.from_pretrained(path, **kwargs)            :656
  │  kwargs.pop("use_fast") 直接丢掉                            :739-740
  │
  ├─[0] mistral_format=True → MistralCommonBackend              :746-750
  │
  ├─[1] 显式 tokenizer_type= → TOKENIZER_MAPPING_NAMES 查表      :752-767
  │
  ├─[2] config（从 config.json 或 gguf）→ config.model_type      :769-779
  │
  ├─[3] 读 tokenizer_config.json → tokenizer_config_class        :784-785
  │        = tok_cfg["tokenizer_class"]
  │     读 tok_cfg["auto_map"]["AutoTokenizer"] → tokenizer_auto_map  :788-797
  │
  ├─[4] MODEL_IDS_TO_TOKENIZERS_BACKEND 名单命中
  │     （deepseek-ai/deepseek-r1-distill-llama-* 等）
  │     → 强制 TokenizersBackend                                 :800-805
  │
  ├─[5] Hub 类名 vs 映射表不一致时的裁决                          :811-859
  │      · 若 config_model_type 在 MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS
  │        → 用映射表里的类；否则信 Hub 的 tokenizer_class         :828-834
  │      · 若映射表给的是 MistralCommonBackend 且可用 → 用它       :844-851
  │      · 都不行 → 直接 TokenizersBackend                        :853-854
  │
  ├─[6] remote code（auto_map + trust_remote_code）             :896-908
  │      · class_ref 优先取 auto_map[1]（fast），否则 auto_map[0]   :897-901
  │
  ├─[7] tokenizer_config_class 能查到 → 用它                      :920-931
  │      · 查不到时试 "<类名>Fast"                                :923-924
  │      · 解析结果是 PythonBackend → 强制换成 TokenizersBackend   :925-926   ★
  │      · 还查不到 → TokenizersBackend                           :928-929
  │
  ├─[8] config.tokenizer_class → 用它（去掉 Fast 后缀）           :932-937
  │
  └─[9] 最后兜底：TOKENIZER_MAPPING.get(type(config), TokenizersBackend)  :951-960
```

### 9.3 三个值得记住的点

**① `tokenizer_config.json` 的 `tokenizer_class` 不是「唯一真相」**

`_from_pretrained` 把 `tokenizer_class` 读进来后立刻扔掉了：

```python
# tokenization_utils_base.py:1771-1772
# used in the past to check if the tokenizer class matches the class in the repo
init_kwargs.pop("tokenizer_class", None)
```

真正用 `tokenizer_class` 选类的是 `AutoTokenizer`（第 [7] 步）。所以：
**直接 `LlamaTokenizer.from_pretrained(...)` 时，`tokenizer_class` 字段完全不影响结果**；
只有走 `AutoTokenizer` 才有意义。

**② 「Hub 上的类名写错了」是一个真实存在的、必须在代码里打补丁的问题**

```python
# models/auto/tokenization_auto.py:371
MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS: set[str] = {
    ...
}
```

`AutoTokenizer` 有一整套逻辑（`:811-859`）在「信 Hub 类名」和「信本地映射表」之间做权衡，
还把 `PythonBackend` 硬改写成 `TokenizersBackend`（`:925-926`）：

```python
# models/auto/tokenization_auto.py:925-926
if tokenizer_class is not None and tokenizer_class.__name__ == "PythonBackend":
    tokenizer_class = TokenizersBackend
```

这就是「v5 强制走 fast」在选类层的体现。

**③ 后缀 `Fast` 被当作噪音剥掉**

```python
# models/auto/tokenization_auto.py:864-865
if tokenizer_config_class and tokenizer_config_class.endswith("Fast"):
    tokenizer_config_class = tokenizer_config_class[:-4]
```

因为 v5 的类名里已经不带 `Fast` 了（别名才对）。老 checkpoint 里存的是
`"tokenizer_class": "LlamaTokenizerFast"`，剥掉后缀后正好命中 `LlamaTokenizer`。

---

## 一句话总结

> 一个基类 `PreTrainedTokenizerBase` 定流程，三个后端 `TokenizersBackend` / `PythonBackend` /
> `SentencePieceBackend` 只实现 `_encode_plus` 等几个钩子；
> `__call__` 先把 padding / truncation 归一成两个枚举（`_get_padding_truncation_strategies:2336`），
> fast 把策略写进 Rust backend 后一次 `encode_batch`，slow 走 `prepare_for_model:1121`
> 的「算长度 → 截断 → 加特殊 token → 补齐 → BatchEncoding」五步；
> 返回的 key 由 `model_input_names` 决定，BOS/EOS 在 fast 里由 post-processor 模板加、
> 在 slow 里由 `build_inputs_with_special_tokens` 加；
> `vocab_size` 是基础词表、`len(tokenizer)` 含 added tokens，两者不一致时必须
> `resize_token_embeddings(len(tokenizer))`；
> 而现代 LLM 的真正入口已经不是 `__call__`，是把它当底座用的 `apply_chat_template`。

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
