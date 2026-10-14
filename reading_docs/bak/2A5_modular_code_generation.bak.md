# 代码生成机制精读：modular 与 `# Copied from`

> 阅读目标：搞清楚 `modular_*.py` 与 `modeling_*.py` 的关系，
> 以及为什么「不要直接改生成文件」。
>
> **这篇为什么要在进阶里排第一个**：它不依赖任何模型结构知识，但它决定你**怎么读后面所有模型的源码**。
> 读完这篇你会知道：以后翻开哪个模型，该读哪个文件。
>
> 源文件：`utils/modular_model_converter.py`（2172 行）、`utils/check_modular_conversion.py`（296 行）、
> `utils/check_copies.py`（1055 行）、`utils/create_dependency_mapping.py`、`Makefile`
> 人类向文档：`docs/source/en/modular_transformers.md`

---

## 速读区（TL;DR）

**一句话结论**：`modular_<name>.py` 是**手写源文件**，`modeling_<name>.py` 是**构建产物**；
converter 只做四件事——继承展开一层、`super()` 内联、全局改名、依赖自动拉取——
所以「**有 modular 就读 modular，读的是差异不是全貌**」。

| # | 断言 | 出处（文件:行号） |
|---|------|------------------|
| 1 | 两套机制**互斥**：modular 生成的 `modeling_*.py` 里 `# Copied from` 数量是 **0** | 已验证 mistral 生成文件 / GLM 生成文件 |
| 2 | 现状用量：`# Copied from` **1164 处**；modular **288 个模型**（510 个模型目录里） | 正文第 30-33 行对照表 |
| 3 | mistral 对照：modular 188 行 / 9 类 → modeling 486 行 / 11 类，跨模型 import **0**、继承 **0** | `modular_mistral.py` vs `modeling_mistral.py` |
| 4 | 生成文件顶上有一段大写警告（`AUTO_GENERATED_MESSAGE`）：认它就是生成文件，改它白改 | `modular_model_converter.py:44` |
| 5 | 一个 modular 文件能生成 5 种文件（`modeling` / `configuration` / `tokenization` / `processing` / `image_processing*`），按**类名后缀**分发 | `TYPE_TO_FILE_TYPE` `:1180`、`find_file_type()` `:1197` |
| 6 | 展开不是「父类在前、子类在后」的追加，而是**按语句名匹配后替换** | `replace_class_node()` `:971`、`_merge_assignments` `:861` |
| 7 | 只展开**一层**：生成文件里基类变成父类的基类（`nn.Module`），`super()` 指向祖父类 | `modular_mistral.py:44` → `modeling_mistral.py:122` |
| 8 | **类级别赋值不是无脑覆盖**：只有匹配 `ASSIGNMENTS_REGEX_TO_KEEP*` 且值非 `None` 才用 modular 的（为了不逼你重抄 docstring） | `_merge_assignments` docstring |
| 9 | 收尾两步：`super().__init__()` 强制挪到 `__init__` 最前面、`self.post_init()` 强制挪到最后 | `:356` / `:341` |
| 10 | 依赖自动拉取：modular 里 `grep -c "MistralRMSNorm"` = **0**，生成文件 `:182` 却有这个类 | `modeling_mistral.py:182` |
| 11 | 类顺序按依赖**拓扑排序**（被依赖的在前）；`logger` / `_CHECKPOINT_FOR_DOC` / `_CONFIG_FOR_DOC` 强制置顶 | `compute_relative_order()` `:779`、`VARIABLES_AT_THE_BEGINNING` `:1215` |
| 12 | CI 判定「生成文件过期」= 现场生成 + `difflib.unified_diff` 有差异；`--fix_and_overwrite` 会先备份成 `.modular_backup` | `check_modular_conversion.py:35` |

**最容易踩的坑 / 反直觉发现**

- **modular 短 ≠ 模型简单**：188 行的 `modular_mistral.py` 生成 486 行；行数只反映「和父模型的差异量」，Mistral 真正的新代码只有 60 行左右，其余都是 `pass` 和 `super().__init__()`。
- **`pass` 会生成一大坨**：`class MistralForCausalLM(LlamaForCausalLM): pass` 展开成完整实现（`modeling_mistral.py:394`），因为 `pass` = 「原样继承，什么都别覆盖」。
- **生成文件里有 modular 没写的类是正常的**（`MistralRMSNorm`、`MistralRotaryEmbedding`）——那是自动拉取，等价于 `pass`，**不是漏读、也不是新东西**。
- **`del self.attr` 只删父类体里的那一行赋值**，引用它的 `if` 块还在 → 运行时报错；必须自己覆写 `forward()` 或其他引用它的方法。
- **类名前缀必须统一**：`Emu3TextMLP` 配 `Emu3DecoderLayer` 会触发告警并只按「最多数」的前缀改高层依赖名——多模态模型（前缀带 `Text`/`Vision`）最容易中招。
- **converter 自身改动会让所有生成文件「过期」**：`CONVERTER_FILES`（`:111`）+ `converter_changed_in_diff()`（`:144`）——升级版本后一堆 `modeling_*.py` 有 diff 而 modular 没动，别慌。
- 另有两条硬约定：每个 modular 文件末尾必须写全 `logger` + `__all__`（`modular_mistral.py:181-188`）；config 的 docstring **不支持部分继承**，加/删字段必须整个手写。

**读完应该能回答的问题**

- 打开一个新模型目录，应该读哪个文件？判断依据是什么？遇到 `# Copied from` 怎么走？
- `pass`、`super().__init__(config)`、`del self.attr`、`**super_kwargs`、`nn.Module.__init__(self)` 各自生成出什么？
- 生成文件里冒出一个 modular 里没有的类，它从哪来？怎么找回它的「父型」？
- 为什么改 `modeling_*.py` 是白改？正确改法是什么命令？
- modular 文件之间的转换顺序为什么不能颠倒？（`llama → mistral → llama4 → glm4`）

---

## 前置知识

- [1M1_llama_structure.md](./1M1_llama_structure.md)：Llama 的类和继承关系（本篇所有例子都基于它）
- [1M2_mistral_sliding_window.md](./1M2_mistral_sliding_window.md)：Mistral 的 modular 文件长什么样

---

## 一、库里有**两套**防重复机制

`transformers` 有个历史包袱：早期设计哲学是「**single model, single file**」——
每个模型一个完全独立的文件，不跨模型 import，不跨模型继承。这样单文件可读、可复制粘贴给用户，
但直接后果是**巨量重复代码**：500 个模型，RMSNorm 抄 500 遍。

为了在「保持单文件形态」和「别抄 500 遍」之间平衡，库先后搞了两套机制：

| 机制 | 年代 | 写法 | 同步方式 | 当前用量 |
|------|------|------|----------|----------|
| `# Copied from` | 旧 | 在文件里直接复制一份代码，顶上挂个注释指回源头 | `utils/check_copies.py` 比对 + 回写 | **1164 处**（大多是老模型） |
| `modular` | 新 | 写 `modular_<name>.py` 用真继承，由脚本**生成** `modeling_<name>.py` | `utils/modular_model_converter.py` 重新生成 | **288 个模型**（510 个模型目录里） |

**关键结论（本篇最重要的一句）**：

> 这两套机制是**互斥**的。由 modular 生成的 `modeling_*.py` 里，
> `# Copied from` 的数量是 **0**（已验证：mistral 生成的 486 行、GLM 生成的 `modeling_glm_moe_dsa.py`，都是 0 处）。
>
> 所以对**新模型**：只读 `modular_*.py`。
> 对**老模型**：读 `modeling_*.py`，见到 `# Copied from` 就顺着注释跳回源头。

CLAUDE.md 里的原话就是这件事：

> Do not edit a `# Copied from` block, as it will be reverted by `make fix-repo`.
> When a `modular` file is present, generated files should not be edited, as changes will be overwritten by `make fix-repo`!

---

## 二、modular 机制全貌

### 一句话

`modular_<name>.py` 是**你手写的、允许继承的源文件**；
converter 把它**展开成**符合「单模型单文件」哲学的 `modeling_<name>.py`。

```
                    ┌─────────────────────────────────┐
 你手写/要读的  →   │  models/mistral/modular_mistral.py │  188 行，9 个类
                    │  （真继承：MistralAttention      │
                    │    extends LlamaAttention）      │
                    └────────────────┬────────────────┘
                                     │
              utils/modular_model_converter.py
              └─ convert_modular_file()  :2011
                 ├─ cst.parse_module()       ← 解析成具体语法树
                 ├─ ModularFileMapper        :1455  ← 遍历 modular 文件
                 ├─ create_modules()         :1917  ← 展开继承 + 拉依赖
                 └─ save_modeling_files()    :2059
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
 生成/不该改的   →   │ models/mistral/modeling_mistral.py │  486 行，11 个类
                    │  （零继承、零跨模型 import）       │
                    └─────────────────────────────────┘
```

**已验证的对照数字**（mistral，可自己复现）：

| | modular | 生成后 |
|---|---|---|
| 行数 | 188 | 486 |
| 类数 | 9 | 11 |
| 跨模型 `import` | 有（`from ..llama.modeling_llama import ...`） | **0** |
| 类继承其他模型 | 有 | **0**（基类全是 `nn.Module` / `PreTrainedModel` 等库内通用类） |
| `# Copied from` | 0 | 0 |

### 生成文件的开头

每个生成文件顶上有一段大写警告（`AUTO_GENERATED_MESSAGE`，`modular_model_converter.py:44`）：

```python
#                🚨🚨🚨 ... 🚨🚨🚨
#           This file was automatically generated from src/transformers/models/mistral/modular_mistral.py.
#               Do NOT edit this file manually as any edits will be overwritten by the generation of
#             the file from the modular. If any change should be done, please apply the change to the
#                          modular_mistral.py file directly. One of our CI enforces this.
#                🚨🚨🚨 ... 🚨🚨🚨
```

**认这行警告**：看到它就是生成文件，改它白改。

### 一个 modular 文件能生成多个文件

converter 不只生成 `modeling_*.py`。它按类名后缀分发
（`TYPE_TO_FILE_TYPE`，`modular_model_converter.py:1180`；`find_file_type()`，同文件 `:1197`）：

| 类名后缀 | 生成到 |
|----------|--------|
| （无匹配，默认）| `modeling_*.py` |
| `Config` | `configuration_*.py` |
| `Tokenizer` | `tokenization_*.py` |
| `Processor` | `processing_*.py` |
| `ImageProcessor` / `ImageProcessorPil` | `image_processing*.py` |
| `VideoProcessor` | `video_processing_*.py` |
| `FeatureExtractor` | `feature_extraction_*.py` |

所以 `modular_mistral.py` 里如果写了 `MistralConfig`，它会落到 `configuration_mistral.py` 而不是 `modeling_`。

---

## 三、converter 到底做了什么

核心函数 `replace_class_node()`（`modular_model_converter.py:971`），docstring 直接写了算法四步：

```
- start from the methods and class attributes of the original modeling code node, and replace their definition
  if overridden in the modular
- append all new methods and class attributes defined in the child class
- all potential method/class docstrings and decorators use the ones found in modular if any, else in original modeling
- replace all calls to super() with the unravelled code
```

翻成中文就是四件事：**继承展开一层、`super()` 内联、全局改名、依赖自动拉取**。
下面逐个看，全部用 Mistral 验证过。

### 1. 继承展开（inline）——把父类的方法体搬进子类

这是整个机制的心脏。**modular 里的 `class MistralAttention(LlamaAttention)` 在生成文件里变成 `class MistralAttention(nn.Module)`**，
父类名消失，方法体被搬了过来，**被子类重定义的语句则被替换掉**。

真实对照（`modular_mistral.py:44` → `modeling_mistral.py:122`）：

```python
# ── 源头：LlamaAttention.__init__（modeling_llama.py:221）──
super().__init__()
self.config = config
self.layer_idx = layer_idx
self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
self.scaling = self.head_dim**-0.5
self.attention_dropout = config.attention_dropout
self.is_causal = True
self.q_proj = nn.Linear(..., bias=config.attention_bias)
self.k_proj = nn.Linear(..., bias=config.attention_bias)
self.v_proj = nn.Linear(..., bias=config.attention_bias)
self.o_proj = nn.Linear(..., bias=config.attention_bias)

# ── modular 里只写了这几行（modular_mistral.py:45-51）──
super().__init__(config, layer_idx)          # ← 信号：「父类体抄进来」
self.head_dim = getattr(config, "head_dim", None) or ...    # 覆盖
self.q_proj = nn.Linear(..., bias=False)                    # 覆盖
self.k_proj = nn.Linear(..., bias=False)                    # 覆盖
self.v_proj = nn.Linear(..., bias=False)                    # 覆盖
self.o_proj = nn.Linear(..., bias=False)                    # 覆盖

# ── 生成结果（modeling_mistral.py:125-137）──
super().__init__()                    # 对 nn.Module 的调用，不是 LlamaAttention
self.config = config                  # ↓ 以下 6 行原样来自 Llama
self.layer_idx = layer_idx
self.head_dim = getattr(config, "head_dim", None) or ...   # ← 已被 Mistral 的版本替换
self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
self.scaling = self.head_dim**-0.5
self.attention_dropout = config.attention_dropout
self.is_causal = True
self.q_proj = nn.Linear(..., bias=False)    # ← 已被 Mistral 的版本替换
self.k_proj = nn.Linear(..., bias=False)
self.v_proj = nn.Linear(..., bias=False)
self.o_proj = nn.Linear(..., bias=False)
```

**关键**：不是「父类在前、子类在后」的简单追加——那样会留下两套投影、让 `bias` 变成 `config.attention_bias`。
实际是**按语句名匹配后替换**（`_merge_assignments`，`:861`），同名赋值以 modular 的为准。
所以生成结果干净、无冗余。

**注意两点**：
- 基类变成了 `nn.Module`（**父类的基类**），不是 `LlamaAttention`。这就是文档说的
  「The linter flattens inheritance to a single level」——**只展开一层**。
  如果 `LlamaAttention` 自己还有父类，那一层不会被继续内联。
- `super().__init__()` 变成了对**祖父类**的调用，函数体被真正写出来。所以生成文件里没有 `super()` 指向跨模型类。

**一个例外规则**（`_merge_assignments` 的 docstring 原文）：

> if any assignment with the same name was redefined in the modular, we use it and its dependencies
> ONLY if it matches a pattern in `ASSIGNMENTS_REGEX_TO_KEEP_IF_NOT_NONE` and its value is not None,
> or if it matches a pattern in `ASSIGNMENTS_REGEX_TO_KEEP`. Otherwise, we use the original value and dependencies.
> **This rule was chosen to avoid having to rewrite the big docstrings.**

也就是说，**类级别的赋值**（`_tied_weights_keys` 这类、以及大段 docstring）不是无脑覆盖：
只有符合特定模式、且值不为 `None` 才采用 modular 的版本，否则沿用父类的。
——纯粹是为了不逼你在 modular 里重抄一大坨文档字符串。

**顺序修正**：合并完成后还有两步收尾（`:356` / `:341`）——
`super().__init__()` 被强制挪到 `__init__` 最前面，`self.post_init()` 被强制挪到最后。
因为展开后语句顺序会乱，而这两个位置的语义有硬要求。


### 2. `super()` 调用内联（unravel）

`is_call_to_super()`（同文件 `:388`）和 `ReplaceSuperCallTransformer`（`:274`）负责这件事。
它在 modular 文件里搜索 `super().<func_name>(...)` 这种形态，把父类同名方法体接上来。

**两个例外写法**（后面「特殊约定速查表」详列）：
- 直接点名调用 `nn.Module.__init__(self)` → 生成文件里写成 `super().__init__()`，意思是「别展开 modular 父类，我只想要 nn.Module」
- `**super_kwargs`（`:1134`）→ 展开父类**签名**但不抄函数体，用来「只换 docstring / 只加一个装饰器」

### 3. 全局改名（rename）

`ReplaceNameTransformer`（`:142`）做全局替换，带大小写保持：

```
LlamaAttention  → MistralAttention
LlamaModel      → MistralModel
LlamaPreTrained → MistralPreTrained
```

**这解释了一个看似魔法的现象**：modular 里写 `class MistralForCausalLM(LlamaForCausalLM): pass`
（`modular_mistral.py:166-167`，真的只有一个 `pass`），生成出来却是完整的一大坨：

```python
# modeling_mistral.py:394
class MistralForCausalLM(MistralPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    _fsdp_plan = {"lm_head": "keep_full_weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = MistralModel(config)          # ← 原来是 LlamaModel(config)，被改名了
        ...
```

`pass` 的含义是「**原样继承，什么都别覆盖**」，于是父类整块被抄下来，名字批量替换。
注意基类 `LlamaPreTrainedModel` 被改成了 `MistralPreTrainedModel`——因为 Mistral 定义了这个类，
所以它变成**同文件引用**，而不是跨模型继承。

### 4. 依赖拉取（dependency pull-in）

converter 会自动把「被展开的代码依赖到、但 modular 里没显式定义」的类也生成出来。
这就是为什么 **modular 有 9 个类、生成文件有 11 个类**：

| 多出来的类 | 为什么被拉进来 |
|-----------|---------------|
| `MistralRMSNorm` | `MistralDecoderLayer` 的父类 `LlamaDecoderLayer` 里 `self.input_layernorm = LlamaRMSNorm(...)` 用到它 |
| `MistralRotaryEmbedding` | 同上，`LlamaModel` 里 `self.rotary_emb = LlamaRotaryEmbedding(config)` |

**实证**：`grep -c "MistralRMSNorm" modular_mistral.py` → **0**，modular 里根本提都没提，
但生成文件 `modeling_mistral.py:182` 有 `class MistralRMSNorm(nn.Module):`。

效果等价于在 modular 里手写：

```python
class MistralRMSNorm(LlamaRMSNorm):
    pass
```

**这条规则对阅读有实操价值**：如果你在读生成的 `modeling_mistral.py`，看到一个 modular 里没有的类，
不要以为漏读了——那是自动拉进来的，等价于 `pass`。想确认它的「父型」是谁，
去 modular 里看谁用了它，再回源头模型找同名类。

**如果你不想要自动拉取**（比如想让 MLP 换个父模型），就得**显式覆盖**：

```python
# 文档里的例子：不要 Olmo 的 MLP，改继承 Mistral 的
from ..mistral.modeling_mistral import MistralMLP

class Olmo2MLP(MistralMLP):
    pass
```

### 5. 排序

生成文件里的类顺序不是随意的，按依赖拓扑排序（`ModelFileMapper.compute_relative_order()`，`:779`），
保证被依赖的类在前。验证：`modeling_mistral.py` 里 `MistralPreTrainedModel` 在 `:244`，
而依赖它的 `MistralModel` 在 `:320`。

顶部还有几个强制置顶的变量（`VARIABLES_AT_THE_BEGINNING`，`:1215`）：
`logger`、`_CHECKPOINT_FOR_DOC`、`_CONFIG_FOR_DOC`。

---

## 四、特殊约定速查表

写/读 modular 文件时，这几个「咒语」的含义必须记住。全部来自 `docs/source/en/modular_transformers.md`
和 converter 源码。

| 写法 | 位置 | 含义 | 生成文件里 |
|------|------|------|-----------|
| `pass` | 类体 | 原样继承，零覆盖 | 父类整块被抄下来 + 全局改名 |
| `super().__init__(config)` | `__init__` | 「把父类 `__init__` 体抄进来」 | 父类体被内联 |
| `super().__init__(...)` + 后续赋值 | `__init__` | 父类体抄完，再追加/覆盖自己的 | 保留父类的重复赋值（不删） |
| `del self.attr` | `__init__` | **只**删掉父类体里的 `self.attr = ...` 这一行 | 那一行消失 |
| `attr = AttributeError()` | config 类体 | 删除继承来的 config 字段 | 字段声明整个消失 |
| `raise AttributeError("...")` | 方法体 | 删除继承来的方法 | 方法整个消失 |
| `**super_kwargs` | 方法签名 | 展开父类完整签名，但不抄函数体 | 父类签名 + 你的 body |
| `nn.Module.__init__(self)` | `__init__` 第一行 | 别展开 modular 父类，只要 `nn.Module` | `super().__init__()` |
| 自己加装饰器 | 方法上 | **替换**父类的装饰器 | 用你的 |
| 不加装饰器 | 方法上 | **继承**父类的装饰器 | 用父类的 |

### 几个容易踩的坑（文档明确警告的）

**坑 1：`del self.attr` 只删赋值那一行**

```python
class DummyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attribute = config.attribute
        if self.attribute:              # ← 这个 if 块不会被删
            ...                          #    引用了已删除的属性 → 运行时报错
```

> 删了赋值行，但引用它的代码还在。必须自己覆写 `forward()` 或其他引用它的方法。

**坑 2：类名前缀必须统一**

```python
class MyModelIncredibleMLP(LlamaMLP): ...   # ✗ 前缀是 MyModelIncredible
class MyModelDecoderLayer(LlamaDecoderLayer): ...  # 前缀是 MyModel
```

converter 不知道该用 `MyModelIncredible` 还是 `MyModel` 去改高层依赖的名字，会报警：

```
We detected multiple prefix names when inheriting from transformers.models.llama.modeling_llama:
('Emu3Text', 'Emu3'). We will only use the most used 'Emu3' prefix when grabbing args and dependencies.
```

多模态模型最容易中招（前缀里带 `Text` / `Vision` 之类的模态限定词）。修法是显式 `pass` 一个类来定名：

```python
class Emu3TextMLP(LlamaMLP):
    pass
```

**坑 3：`__all__` 必须列全**

每个 modular 文件末尾必须声明 `logger` 和完整的 `__all__`（`modular_mistral.py:181-188`）。
漏掉的类不会被正确导出。

**坑 4：config docstring 不支持部分继承**

加/删 config 字段时，docstring 必须整个手写在 modular 文件里，不能只写差异。

---

## 五、转换顺序与 `make fix-repo`

### 拓扑排序：父模型必须先转

modular 文件之间也有依赖链：`llama` → `mistral` → `llama4` → `glm4`。
`create_dependency_mapping.py:7` 的 `topological_sort()` 把它排成层级：

```python
[
    ["../modular_mistral.py", "../modular_gemma.py"],  # level 0：不依赖任何 modular
    ["../modular_llama4.py", "../modular_gemma2.py"],  # level 1：依赖 level 0
    ["../modular_glm4.py"],                            # level 2：依赖 level 1
]
```

**为什么要排序**：`mistral` 的生成结果依赖 `llama` 的生成结果。
如果先转 `mistral` 再转 `llama`，`mistral` 抄到的是旧的 `llama`。按层级推进就没这问题。

### Makefile 里的位置

`modular_conversion` 是 `utils/checkers.py` 的一个 checker，挂在 `REPO_CONSISTENCY_CHECKERS` 里（Makefile:17）：

```makefile
REPO_CONSISTENCY_CHECKERS := \
	auto_mappings, imports, import_complexity, copies, \
	modular_conversion, \      # ← 就是它
	inits, doc_toc, reviewers, ...
```

| 命令 | 对 modular 做什么 |
|------|------------------|
| `make check-repo` | 跑所有 checker，**只报错不改**（`--keep-going`） |
| `make fix-repo` | 跑所有 checker 并**自动修**（`--fix --keep-going`）→ 重新生成所有 `modeling_*.py` |
| `make check-code-quality` | 不含 modular（那是 CircleCI 的另一半） |

本 checker 的配置（`check_modular_conversion.py:19-26`）：

```python
CHECKER_CONFIG = {
    "name": "modular_conversion",
    "cache_globs": ["src/transformers/models/**/modular_*.py",
                    "src/transformers/models/**/modeling_*.py"],
    "fix_args": ["--fix_and_overwrite"],
}
```

### CI 怎么判定「生成文件过期了」

`process_file()`（`check_modular_conversion.py:35`）的逻辑很直白：

```
1. 用 converter 现场生成一份内容
2. 和磁盘上真实的 modeling_*.py 做 difflib.unified_diff
3. 有差异 → 报错（CI 失败）
4. --fix_and_overwrite 模式下 → 先备份成 .modular_backup，再写入生成内容
```

`.modular_backup` 这个后缀就是「被覆盖前的原文件」，用来在需要时还原。

### 一个性能优化

`guaranteed_no_diff()`（`:149`）会跳过大量文件的转换：

```
该模型在 diff 里        → 不能保证无差异，必须转
该模型依赖的模型在 diff 里 → 不能保证无差异，必须转
否则                    → 保证无差异，跳过
```

另外 `CONVERTER_FILES`（`:111`）声明：只要 `utils/modular_model_converter.py` 或
`utils/create_dependency_mapping.py` 被改动，**所有** modular 模型都必须重查——
因为改 converter 可能影响每一个生成文件（`converter_changed_in_diff()`，`:144`）。

> **这对你的意义**：如果你升级了 transformers 版本，发现一堆 `modeling_*.py` 都有 diff 但 modular 没动，
> 别慌——很可能只是 converter 本身改了。

---

## 六、`# Copied from` 机制（老模型还得会认）

新模型不用，但你在 `llama4` / `pixtral` / `nllb_moe` 这些文件里还会大量遇到（全库 1164 处）。

### 形态

```python
# 单文件/跨文件
# Copied from transformers.models.mask2former.modeling_mask2former.sigmoid_cross_entropy_loss
def sigmoid_cross_entropy_loss(...):
    ...

# 带字符串替换：OldName->NewName
# Copied from transformers.models.m2m_100.modeling_m2m_100.M2M100SinusoidalPositionalEmbedding with M2M100->NllbMoe
class NllbMoeSinusoidalPositionalEmbedding(nn.Module):
    ...
```

正则（`check_copies.py:541`）：

```python
_re_copy_warning = re.compile(r"^(\s*)#\s*Copied from\s+transformers\.(\S+\.\S+)\s*($|\S.*$)")
```

捕获三组：**缩进**、**对象名**（`models.xxx.modeling_xxx.ClassName`）、**替换模式**（`A->B`，可为空）。

### 校验逻辑

`is_copy_consistent()`（`check_copies.py:645`）：

```
1. 从注释解析出源头对象名和替换模式
2. find_code_and_splits() 去源头文件把那段代码抠出来
3. 用 replace_code() 对源头代码和本地代码同时施加替换（A->B）
4. 两侧 stylify() 后用 difflib 比对
5. 不一致 → 报错；--fix_and_overwrite → 用源头代码覆盖本地
```

`stylify()`（`:576`）的存在很关键：比对前会跑格式化，
所以**纯缩进/换行差异不会被误报**，只有实质代码差异才报。

### 和 modular 的分工

| | `# Copied from` | `modular` |
|---|---|---|
| 代码形态 | 物理重复，注释指回源头 | 真继承，脚本展开 |
| 展开层级 | 单块（一个类/函数） | 一个模型的所有类 |
| 源头变更后 | 下游文件下次 `fix-repo` 被覆盖 | 下游生成文件下次 `fix-repo` 重新生成 |
| 跨文件 import | 无（所以保持单文件形态） | 生成后也无 |
| 适用 | 一个函数级别的小复用、老模型维护 | 新模型 |

**核心区别**：`# Copied from` 是「**复制后标记**」，modular 是「**继承后展开**」。
两者殊途同归——最终产物都是「无跨模型依赖的独立单文件」。

---

## 七、对阅读源码的实操影响

这是本篇最实用的部分。

### 规则 1：先看一眼有没有 modular

```bash
ls src/transformers/models/<name>/
```

| 看到 | 读哪个 |
|------|--------|
| 有 `modular_<name>.py` | **读 modular**。差异一眼可见，通常短很多 |
| 没有 | 读 `modeling_<name>.py`，遇到 `# Copied from` 顺着跳源头 |

已核对的分布（覆盖本路线图全部模型）：

| 有 modular（读 modular） | 无 modular（直读 modeling） |
|--------------------------|------------------------------|
| mistral、mixtral、gemma2、deepseek_v2、qwen3、qwen3_moe、modernbert、glm_moe_dsa | llama、bert、t5、mamba2、llava_next |

### 规则 2：modular 短 ≠ 模型简单

`modular_mistral.py` 188 行，但生成 `modeling_mistral.py` 486 行。
modular 的行数只反映「**和父模型的差异量**」，不反映模型的绝对复杂度。

反过来也成立：**Mistral 这个模型真的很简单**——188 行里真正属于它的新代码
（去掉 import、`__all__`、样板）也就 60 行左右。剩下的都是 `pass` 和 `super().__init__()`。

### 规则 3：`pass` 是「我没意见」的信号

读 modular 时看到：

```python
class MistralForCausalLM(LlamaForCausalLM):
    pass
```

意思是「完全一样，别看了」。**不要在生成的 `modeling_*.py` 里费劲读这个类**——
它和 `LlamaForCausalLM` 逐行一致（只是名字换了）。

这和第 1M 篇的读法完全互补：1M1 教你读 Llama，1M2 往后你只需要读「差异」。

### 规则 4：注意「隐式生成」的类

生成的 `modeling_*.py` 里会有 modular 里没写的类（如 `MistralRMSNorm`、`MistralRotaryEmbedding`）。
它们等价于 `pass`，**不是新东西**。别浪费时间。

### 规则 5：改动只能在 modular 里做

如果你要改模型代码：

```bash
# ✗ 错：直接改 modeling_mistral.py → 下次 make fix-repo 被覆盖
# ✓ 对：改 modular_mistral.py，然后
python utils/modular_model_converter.py mistral     # 单模型
make fix-repo                                        # 全仓库
```

CLAUDE.md 也把这条列为硬规则（「Do not edit a `# Copied from` block」+「changes will be overwritten by `make fix-repo`」）。

---

## 八、一句话总结

> **modular 是源文件，modeling 是构建产物。**
> converter 做的四件事：继承展开一层、`super()` 内联、全局改名、依赖自动拉取。
> 生成的 `modeling_*.py` 零跨模型 import、零继承、零 `# Copied from`。
> 以后读任何模型：**有 modular 就读 modular，读的是「差异」不是「全貌」**。

---

## 验证清单

想自己动手确认本篇结论，跑这几条（在仓库根目录）：

```bash
# 1. 生成文件里没有 Copied from（两套机制互斥）
grep -c "Copied from" src/transformers/models/mistral/modeling_mistral.py     # → 0

# 2. 生成文件里没有跨模型 import
#    （跨模型 import = 恰好两个点的相对 import，如 `from ..llama.modeling_llama import ...`）
grep -nE "^from \.\.[a-z_0-9]+\.(modeling|configuration|tokenization|processing)" \
     src/transformers/models/mistral/modeling_mistral.py                      # → 无匹配
# 对照：modular 文件里有（modular_mistral.py:18）
grep -nE "^from \.\.[a-z_0-9]+\.(modeling|configuration)" \
     src/transformers/models/mistral/modular_mistral.py                       # → from ..llama.modeling_llama import (

# 3. modular 里没有 RMSNorm，但生成文件里有（依赖自动拉取）
grep -c "MistralRMSNorm" src/transformers/models/mistral/modular_mistral.py   # → 0
grep -n "class MistralRMSNorm" src/transformers/models/mistral/modeling_mistral.py
#    → 182:class MistralRMSNorm(nn.Module):

# 4. 行数/类数对比
wc -l src/transformers/models/mistral/modular_mistral.py \
      src/transformers/models/mistral/modeling_mistral.py                     # → 188 / 486

# 5. 亲手跑一次转换（会覆盖生成文件，谨慎）
python utils/modular_model_converter.py mistral
```

---

## 导航

- 路线图：[00_roadmap.md](./00_roadmap.md)（本篇 = 进阶 · 架构 · 2A5）
- 上一篇：2A4 RoPE 抽象层（待写）
- 下一篇：2A1 注意力后端机制（待写）
- **建议紧接着读**：回到 [1M2_mistral_sliding_window.md](./1M2_mistral_sliding_window.md)，
  用本篇的视角重看一遍那个 188 行的 modular 文件——「哪些是 Mistral 的、哪些是自动拉进来的」应该一目了然了。
