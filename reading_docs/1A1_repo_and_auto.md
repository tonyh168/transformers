# 仓库骨架与 auto 机制精读

> 阅读目标：搞清楚两件事，它们是后面所有篇章的地基。
>
> 1. **物理层面**：`src/transformers/models/<name>/` 这个目录里到底该有哪些文件、每个文件负责什么、
>    哪些是可选的。
> 2. **机制层面**：`AutoModel.from_pretrained("meta-llama/Llama-3-8B")` 只拿到一个字符串和一个
>    `config.json`，它凭什么知道要 `import` 哪个文件的哪个类？
>
> 这是架构线的**第一篇**——它不解释任何一个模型长什么样（那是 1M 线的活），
> 只解释「库是怎么把 500 个模型塞进一个 `import transformers` 里还能瞬间定位到一个类的」。
>
> 源文件：`src/transformers/models/auto/auto_factory.py`（699 行）、
> `src/transformers/models/auto/auto_mappings.py`（1328 行）、
> `src/transformers/models/auto/modeling_auto.py`（2662 行）、
> `src/transformers/models/auto/configuration_auto.py`（456 行）、
> `utils/check_auto.py`（331 行）、`utils/sort_auto_mappings.py`（130 行）、`Makefile`

---

## 速读区（TL;DR）

**一句话**：本篇讲「定位机制」——① `models/<name>/` 里该有哪些文件（「五件套」是**候选清单**不是必填清单）；
② `AutoModel.from_pretrained("xxx")` 靠**两次查表 + 两次惰性 import** 找到类，全程没有任何「遍历 500 个目录去猜」的动作。

| # | 断言 | 出处（文件:行号） |
|---|------|------------------|
| 1 | `auto_mappings.py` 是**生成物**（6 张名字表，从各 `configuration_*.py` 反推），而 `modeling_auto.py` 那 53 张 `MODEL_*_MAPPING_NAMES` 是**手写物** | `utils/check_auto.py:42`、`:69`；`modeling_auto.py:692` 起 |
| 2 | `_LazyAutoMapping` **没有 `__getattr__`**；惰性全部来自 `_load_attr_from_module` 里显式的 `importlib.import_module` | `auto_factory.py:612` |
| 3 | `_get_model_class` 传的是 config 的**类**（`model_mapping[type(config)]`），`_LazyAutoMapping` 又用 config 类的 `__name__` **字符串**去反查 | `auto_factory.py:179`、`:599` |
| 4 | 模型映射表里 **`None` 出现 0 次**（`grep -c "None" modeling_auto.py` → 0），「不支持这个任务头」= **该 key 根本不出现** | `modeling_auto.py` |
| 5 | 映射表里 value 是 tuple 的只有 **4 处**，如 `("funnel", ("FunnelModel", "FunnelBaseModel"))`；`architectures` 决定选哪个，否则取第一个 | `modeling_auto.py:183`、`auto_factory.py:184-191` |
| 6 | `SPECIAL_MODEL_TYPE_TO_MODULE_NAME` 有 **248** 条：`llama4_text` / `llama4_vision_model` 没有自己的目录，模块是 `models/llama4/` | `auto_mappings.py:733`、`configuration_auto.py:64-76` |
| 7 | `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES` 里 **`"gemma3"` 指向 `Gemma3ForConditionalGeneration`**（多模态包装类），`Gemma3ForCausalLM` 留给了 `gemma3_text` | `modeling_auto.py:745-746` |
| 8 | `models/<name>/__init__.py` **不手写导出**：类清单是 `define_import_structure()` 扫目录扫出来的 | `llama/__init__.py:28`、`utils/import_utils.py:3301` |
| 9 | 加新模型时 `CONFIG_MAPPING_NAMES` 会自动长出来，但 `MODEL_FOR_*_MAPPING_NAMES` **必须人肉加一行** | `modeling_auto.py:692` 起 |
| 10 | 排序 checker 名是 **`sort_auto_mappings`**（不是 `check_auto`），它的 `fix_args` 为空 → **只有 `make style` 能修排序** | `sort_auto_mappings.py:37-42`、`Makefile:13` |
| 11 | `MODEL_FOR_AUDIO_TOKENIZATION_NAMES` 少了 `_MAPPING` 后缀，因此**不会被 `check_repo.py` 的后缀扫描检查到**——给新模型起名别学这个特例 | `modeling_auto.py:2021`、`check_repo.py:1046` |
| 12 | tokenizer / processor 映射里的 `None` 意思是「**可选依赖没装**」，且被 `getattribute_from_module` **静默返回 `None`**（调用方得自己判空） | `tokenization_auto.py:67`、`auto_factory.py:522-523` |

**最容易踩的坑 / 反直觉**

- 网上常见说法「mapping 里的 `None` 表示该模型不支持这个任务头」**在本仓库的模型映射上不成立**：`modeling_auto.py` 全文零个 `None`，不支持的表达方式是「不在表里」，后果是 `ValueError: Unrecognized configuration class ...`。
- 找不到 `_LazyAutoMapping.__getattr__` 是正常的：`grep` 全文只有 `:683` 的一句注释。模块级 `__getattr__` 的真实位置是 `utils/import_utils.py:2500`（`_LazyModule`）和 `transformers/__init__.py:835`。
- 文件名用的是**模块名**不是 `model_type`：看到 `"model_type": "llama4_text"` **不要去**找 `models/llama4_text/`，那个目录不存在；`-` 还会被换成 `_`（`bert-generation` → `bert_generation/`）。
- 「忘了在 `modeling_auto.py` 里加一行」这件事：`auto_mappings` checker **抓不到**（它只管 config 映射），但 `repo` checker 的 `check_models_are_auto_configured()` 会抓到你（白名单常量 `IGNORE_NON_AUTO_CONFIGURED`，`check_repo.py:348`）。
- `_LazyAutoMappingValue = tuple[type | None, type | None]`（`auto_factory.py:49`）是**查表返回值的类型标注**，不是「表里有 `None`」；实际返回值可能是单个类、tuple，也可能是 dict（`IMAGE_PROCESSOR_MAPPING_NAMES` 的 `{"pil": ..., "torchvision": ...}`，`auto_mappings.py:1200`）。
- 本篇**没有做运行时验证**（本环境缺 `regex`，import `transformers` 直接失败），所有结论由 `grep` / `sed` / `awk` / `wc` 加 `ast` 静态解析得出。

**读完应该能回答**

- 从 `config.json` 的 `model_type` 到 `LlamaForCausalLM`，一共查了几次表、`importlib.import_module` 跑了几次？
- 想让 `MyModel` 支持 `AutoModelForCausalLM`，哪些文件必须**手改**、哪一步是脚本不会帮你做的？
- 三个 checker（`auto_mappings` / `sort_auto_mappings` / `repo`）各自管什么？报错时分别用什么命令修？
- 为什么 `import transformers` 不会 import 那 495 个 `modeling_*.py`？

---

## 前置知识

- **无**。本篇是架构线的起点，不需要任何前置。
- 可选先读 [2A5_modular_code_generation.md](./2A5_modular_code_generation.md)：
  它讲 `modular_*.py` 与 `modeling_*.py` 的生成关系。路线图建议「越早读越好」，
  因为读完你会发现「`modular` 是个构建产物源头」这件事本身就属于仓库骨架的一部分。
- 模型结构内容（RMSNorm / RoPE / Attention）不在本篇范围，见
  [1M1_llama_structure.md](./1M1_llama_structure.md)。

---

## 一、物理骨架：一个模型目录里有什么

### 1.1 先看真实的四个目录

```text
src/transformers/models/llama/          src/transformers/models/mistral/
├── __init__.py                         ├── __init__.py
├── configuration_llama.py              ├── configuration_mistral.py
├── modeling_llama.py                   ├── modeling_mistral.py
├── tokenization_llama.py               └── modular_mistral.py
└── convert_llama_weights_to_hf.py

src/transformers/models/llava_next/     src/transformers/models/qwen2_vl/
├── __init__.py                         ├── __init__.py
├── configuration_llava_next.py         ├── configuration_qwen2_vl.py
├── modeling_llava_next.py              ├── modeling_qwen2_vl.py
├── processing_llava_next.py            ├── processing_qwen2_vl.py
├── image_processing_llava_next.py      ├── image_processing_qwen2_vl.py
└── image_processing_pil_llava_next.py  ├── image_processing_pil_qwen2_vl.py
                                        └── video_processing_qwen2_vl.py
```

三个目录，三种形态。**没有哪个模型同时拥有全部文件**——这就是「五件套」的真相：
它是一份**候选清单**，不是一份必填清单。

### 1.2 「五件套」职责表

| 文件 | 里面定义什么 | 职责 | 全库数量 | 可选？ |
|------|-------------|------|---------|--------|
| `configuration_<name>.py` | `<Name>Config(PreTrainedConfig)` | **一切的入口**。类体里的 `model_type = "llama"` 是 auto 机制唯一的锚点 | 497 / 510 目录 | 几乎必须 |
| `modeling_<name>.py` | `<Name>Model`、`<Name>ForCausalLM` … | PyTorch 模型本体（1M 线的主角） | 495 | 见下方注 1 |
| `tokenization_<name>.py` | `<Name>Tokenizer` | 文本 → token id。慢/快两套后端 | 101 | **可选** |
| `processing_<name>.py` | `<Name>Processor(ProcessorMixin)` | 把 tokenizer + image/video processor 打包成一个 `__call__` | 151 | **可选** |
| `modular_<name>.py` | 手写的、允许继承的源文件 | 由脚本生成上面几个文件，见 [2A5](./2A5_modular_code_generation.md) | 288 | **可选** |

> **注 1**：`modeling_*.py` 有 495 个但目录有 510 个——差的 15 个是**纯 tokenizer 目录**，
> 比如 `barthez/`、`bert_japanese/`、`byt5/`、`phobert/`、`nllb/`、`code_llama/`、
> `wav2vec2_with_lm/`。它们复用了别的模型的 `modeling`，自己只提供分词器。
> 反过来，`deprecated/` 是个子目录，不是模型。

**「五件套」之外还有一批同族文件**（同样按 `<name>` 命名，同样可以出现在 modular 的生成产物里）：

| 文件 | 全库数量 | 什么时候出现 |
|------|---------|-------------|
| `image_processing_<name>.py` | 217 | 多模态模型（视觉输入） |
| `image_processing_pil_<name>.py` | 91 | 同一模型的 PIL 后端版本（如 `qwen2_vl`） |
| `feature_extraction_<name>.py` | 33 | 音频模型（波形 → 特征） |
| `video_processing_<name>.py` | 31 | 视频输入 |
| `generation_<name>.py` | 11 | 模型有自己的一套 `generate` 逻辑（如 `whisper`、`musicgen`） |
| `convert_<name>_weights_to_hf.py` | 大量 | **不是库的一部分**，是开发者的权重转换脚本，不会被 import |

### 1.3 命名规范：文件名用的是「模块名」，不是「model_type」

这是最容易踩的坑。三条规则，全部来自 `configuration_auto.py:64-76` 的
`model_type_to_module_name()`：

```python
def model_type_to_module_name(key) -> str:                       # configuration_auto.py:64
    """Converts a config key to the corresponding module."""
    # Special treatment
    if key in SPECIAL_MODEL_TYPE_TO_MODULE_NAME:                 # :67
        key = SPECIAL_MODEL_TYPE_TO_MODULE_NAME[key]             # :68
        ...
    key = key.replace("-", "_")                                  # :74
```

| 规则 | model_type | 模块名 / 文件名 |
|------|-----------|----------------|
| `-` → `_` | `bert-generation` | `bert_generation/configuration_bert_generation.py` |
| `-` → `_` | `blenderbot-small` | `blenderbot_small/` |
| 特殊映射表 | `bert-generation` | 同上（`SPECIAL_MODEL_TYPE_TO_MODULE_NAME` 里也记了一笔，`auto_mappings.py:744`） |
| 一个目录扛多个 model_type | `llama4_text` / `llama4_vision_model` | 都在 `llama4/` 里（`auto_mappings.py:857-858`） |
| 一个目录扛多个 model_type | `blip-2` / `blip_2_qformer` / `blip_2_vision_model` | 都在 `blip_2/` 里（`auto_mappings.py:746-748`） |

`SPECIAL_MODEL_TYPE_TO_MODULE_NAME` 有 **248 条**（`auto_mappings.py:733` 起）。
**绝大多数多模态模型都会在这里留名**——因为它们的 `model_type` 是 `xxx_text` / `xxx_vision_model`，
但目录只叫 `xxx`。

> 实操含义：看到 `config.json` 里 `"model_type": "llama4_text"`，**不要去**找
> `models/llama4_text/`——那个目录不存在。真正的模块是 `models/llama4/`。

### 1.4 `__init__.py` 里没有 export 语句

`src/transformers/models/llama/__init__.py` 全文只有 28 行，核心就这几行：

```python
if TYPE_CHECKING:                                            # llama/__init__.py:20
    from .configuration_llama import *                       # :21
    from .modeling_llama import *                            # :22
    from .tokenization_llama import *                        # :23
else:                                                        # :24
    import sys
    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
```

`define_import_structure()`（`utils/import_utils.py:3301`）会**扫描该目录下的文件**，
读出每个文件里的 `__all__`，自动拼出一张「类名 → 模块」的表。
所以 `transformers.models.llama.LlamaForCausalLM` 能取到，是**扫出来的，不是手写的**。

**这条对阅读源码很关键**：`__init__.py` 里看不到类清单，
想知道一个模型对外暴露了什么，去读 `modeling_*.py` 的 `__all__`。

---

## 二、核心问题：`AutoModel.from_pretrained("xxx")` 怎么知道用哪个类

### 2.1 三层跳转，一张图看完

```
config.json 里的 "model_type": "llama"
        │
        │  ①  AutoConfig.from_pretrained()  读 json，拿 model_type
        ▼
CONFIG_MAPPING_NAMES  ── "llama" ──→ "LlamaConfig"          （字符串表，auto_mappings.py:356）
        │
        │  ②  CONFIG_MAPPING[...] 惰性 import 出真正的类
        ▼
LlamaConfig  （类对象，它的 __name__ 是 "LlamaConfig"）
        │
        │  ③  _LazyAutoMapping 反查：类名 → model_type → 模型类名
        ▼
MODEL_FOR_CAUSAL_LM_MAPPING_NAMES ── "llama" ──→ "LlamaForCausalLM"  （modeling_auto.py:789）
        │
        │  ④  惰性 import transformers.models.llama
        ▼
LlamaForCausalLM   ← 最终实例化这个
```

全程**没有**任何「遍历 500 个目录去猜」的动作，全是查表。表是两份字符串字典 + 两次 `importlib`。

### 2.2 完整调用链（带真实行号）

入口是 `_BaseAutoModelClass.from_pretrained`（`auto_factory.py:261`）：

```python
@classmethod
def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike[str], *model_args, **kwargs):
    config = kwargs.pop("config", None)                       # :262
    trust_remote_code = kwargs.get("trust_remote_code")        # :263
    kwargs["_from_auto"] = True                               # :264  ← 「我是从 auto 进来的」标记
```

`kwargs["_from_auto"] = True` 这个标记一路往下传给 `model_class.from_pretrained()`，
让模型自己的 `from_pretrained` 知道「config 已经被 auto 校验过了」——
这是 1A3 会展开的伏笔。

| 步 | 位置 | 做什么 |
|----|------|--------|
| 1 | `auto_factory.py:336` | `config, kwargs = AutoConfig.from_pretrained(...)` — 读 `config.json` |
| 2 | `configuration_auto.py:388` | `PreTrainedConfig.get_config_dict(...)` 把 json 读成 dict |
| 3 | `configuration_auto.py:419` | `config_class = CONFIG_MAPPING[config_dict["model_type"]]` ← **查表 ①** |
| 4 | `configuration_auto.py:431` | `return config_class.from_dict(config_dict, **unused_kwargs)` — 得到 `LlamaConfig` 实例 |
| 5 | `auto_factory.py:355-357` | 判断 `has_remote_code` / `has_local_code` / `explicit_local_code` |
| 6 | `auto_factory.py:389` | `model_class = _get_model_class(config, cls._model_mapping)` ← **查表 ②** |
| 7 | `auto_factory.py:402` | `return model_class.from_pretrained(..., config=config, ...)` — 交棒给真正的模型类 |

第 4 步值得停一下，看看 `config.json` 里到底有什么：

```json
{
  "architectures": ["LlamaForCausalLM"],
  "model_type": "llama",
  "hidden_size": 4096,
  ...
}
```

`model_type` 决定**用哪个 Config 类**，`architectures` 决定**用哪个模型类**（见 2.5）。

### 2.3 查表 ①：`model_type` → Config 类

`configuration_auto.py:153`：

```python
CONFIG_MAPPING = _LazyConfigMapping(CONFIG_MAPPING_NAMES)
```

`_LazyConfigMapping`（`:93`）的 `__getitem__`（`:103`）逻辑：

```python
def __getitem__(self, key: str) -> type[PreTrainedConfig]:        # :103
    if key in self._extra_content:                                # :104   远程代码注册的
        return self._extra_content[key]
    if key not in self._mapping:                                  # :106
        raise KeyError(key)
    value = self._mapping[key]                                    # :108   ← "LlamaConfig" 字符串
    module_name = model_type_to_module_name(key)                   # :109   "llama"
    if module_name not in self._modules:                           # :110
        self._modules[module_name] = importlib.import_module(f".{module_name}", "transformers.models")
    if hasattr(self._modules[module_name], value):                 # :112
        return getattr(self._modules[module_name], value)          # :113   ← 真类到手
    # 兜底：有些映射指向别的模型的 config，那就去顶层找
    transformers_module = importlib.import_module("transformers")   # :118
    return getattr(transformers_module, value)
```

**注意 `CONFIG_MAPPING` 是 `model_type` → 类的映射，而 `CONFIG_MAPPING_NAMES` 是
`model_type` → 类名字符串的映射。** 前者是后者 + 惰性 import。

### 2.4 查表 ②：Config 类 → 模型类

`_get_model_class`（`auto_factory.py:178`）只有 14 行，但信息量最大：

```python
def _get_model_class(config, model_mapping):                        # :178
    supported_models = model_mapping[type(config)]                  # :179  ← 传的是「类」不是实例
    if not isinstance(supported_models, (list, tuple)):             # :180
        return supported_models                                     # :181  唯一候选，直接返回
    name_to_model = {model.__name__: model for model in supported_models}
    architectures = getattr(config, "architectures", [])            # :184  ← config.json 里的那个字段
    for arch in architectures:                                      # :185
        if arch in name_to_model:
            return name_to_model[arch]                              # :187  命中！
    # If not architecture is set in the config or match the supported models, the first element of the tuple is the
    # defaults.
    return supported_models[0]                                      # :191  兜底：取第一个
```

`model_mapping[type(config)]` 走的是 `_LazyAutoMapping.__getitem__`（`auto_factory.py:596`）：

```python
def __getitem__(self, key: type[PreTrainedConfig]) -> _LazyAutoMappingValue:   # :596
    if key in self._extra_content:                                             # :597  远程注册优先
        return self._extra_content[key]
    model_type = self._reverse_config_mapping[key.__name__]                    # :599  "LlamaConfig" → "llama"
    if model_type in self._model_mapping:                                      # :600
        model_name = self._model_mapping[model_type]                           # :601  → "LlamaForCausalLM"
        return self._load_attr_from_module(model_type, model_name)             # :602

    # Maybe there was several model types associated with this config.
    model_types = [k for k, v in self._config_mapping.items() if v == key.__name__]   # :605
    for mtype in model_types:                                                  # :606
        if mtype in self._model_mapping:
            model_name = self._model_mapping[mtype]
            return self._load_attr_from_module(mtype, model_name)
    raise KeyError(key)                                                        # :610
```

两个关键点：

- `:599` 用的是 **config 类的 `__name__` 字符串**（`"LlamaConfig"`），不是类对象。
  所以 `_LazyAutoMapping` 只认「库自己定义的那个 Config 类」——
  你如果用 `class MyLlamaConfig(LlamaConfig)` 传进去，`key.__name__` 是 `"MyLlamaConfig"`，
  在 `_reverse_config_mapping` 里查不到，直接掉到 `:605` 的兜底分支再 `KeyError`。
- `:605-608` 的兜底分支处理「多个 `model_type` 共用同一个 Config 类」的情况
  （`CONFIG_MAPPING_NAMES` 里有重复 value，`utils/check_auto.py:164-173` 会专门检查这件事）。

### 2.5 为什么 `_get_model_class` 要判「list / tuple」

因为映射表的**值可以是 tuple**——一个 `model_type` 对应多个候选类。
全库只有 **4 处**这样写（用 `ast` 扫 `modeling_auto.py` 得到）：

| 表 | 行 | 条目 |
|----|-----|------|
| `MODEL_MAPPING_NAMES` | `:183` | `("funnel", ("FunnelModel", "FunnelBaseModel")),` |
| `MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES` | `:966-969` | `("deit", ("DeiTForImageClassification", "DeiTForImageClassificationWithTeacher")),` |
| 同上 | `:980-983` | `("levit", ("LevitForImageClassification", "LevitForImageClassificationWithTeacher")),` |
| 同上 | `:989-995` | perceiver 的三元 tuple（`Learned` / `Fourier` / `ConvProcessing`） |

于是 `_get_model_class` 的 `:180` 分支就有了意义。以 `funnel` 为例，完整的判定过程：

```
model_mapping[FunnelConfig]
   └─ _LazyAutoMapping.__getitem__ :599  → model_type = "funnel"
        └─ _load_attr_from_module :612   → import transformers.models.funnel
             └─ getattribute_from_module :524 命中 tuple 分支
                  → (FunnelModel, FunnelBaseModel)          ← 返回的是 tuple！
_get_model_class :180  isinstance(supported_models, (list, tuple)) → True，不直接返回
_get_model_class :184  architectures = config.architectures      → ["FunnelModel"]
_get_model_class :187  name_to_model["FunnelModel"] 命中        → 返回 FunnelModel
```

**若 `config.json` 里没有 `architectures`，或写的名字不在 tuple 里，就落到 `:191` 取 tuple 第一个。**
所以 `("funnel", ("FunnelModel", "FunnelBaseModel"))` 的语义是：
「默认 `FunnelModel`（第一个），但 `architectures` 说了算」。

同一个 `getattribute_from_module` 的 dict 分支（`:526-527`）服务于另一类值——
`IMAGE_PROCESSOR_MAPPING_NAMES` 的「双后端」写法（`auto_mappings.py:1200`）：

```python
("aria", {"pil": "AriaImageProcessorPil", "torchvision": "AriaImageProcessor"}),
```

这就是 `_LazyAutoMappingValue = tuple[type[Any] | None, type[Any] | None]`（`auto_factory.py:49`）
那个类型标注的来历——**返回值可能是单个类、tuple、也可能是 dict**。

### 2.6 惰性导入：为什么 `import transformers` 不会 import 500 个模型

`_LazyAutoMapping._load_attr_from_module`（`auto_factory.py:612`）：

```python
def _load_attr_from_module(self, model_type, attr):                      # :612
    module_name = model_type_to_module_name(model_type)                  # :613
    if module_name not in self._modules:                                 # :614  ← 缓存，只 import 一次
        self._modules[module_name] = importlib.import_module(f".{module_name}", "transformers.models")  # :615
    return getattribute_from_module(self._modules[module_name], attr)    # :616
```

配合 `getattribute_from_module`（`:521`）：

```python
def getattribute_from_module(module, attr):                              # :521
    if attr is None:                                                     # :522
        return None                                                      # :523
    if isinstance(attr, tuple):                                          # :524
        return tuple(getattribute_from_module(module, a) for a in attr)  # :525
    if isinstance(attr, dict):                                           # :526
        return {k: getattribute_from_module(module, v) for k, v in attr.items()}
    if hasattr(module, attr):                                            # :529
        return getattr(module, attr)                                      # :530
    # 有些映射的值来自别的模型 —— 那就去顶层 `transformers` 命名空间找
    transformers_module = importlib.import_module("transformers")        # :533
    ...
```

**这三处的分工很清晰**：

| 层 | 机制 | 惰性在哪 |
|----|------|---------|
| `transformers.AutoModel` | `_LazyModule.__getattr__`（`utils/import_utils.py:2500`） | 访问属性时才 import 对应模块 |
| `transformers.models.llama.LlamaForCausalLM` | `_LazyModule` + `define_import_structure`（`llama/__init__.py:28`） | `__init__.py` 里没有真 import |
| `_LazyAutoMapping[...]` | `_load_attr_from_module`（`auto_factory.py:612`） | 每个 `model_type` 首次被查表时才 `importlib.import_module` |

> **纠正一个常见的想当然**：翻 `auto_factory.py` 你会**找不到** `_LazyAutoMapping.__getattr__`——
> 这个类里根本没有 `__getattr__`（`grep -n "__getattr__" auto_factory.py` 只有 `:683` 的一句注释）。
> 惰性全部发生在 `_load_attr_from_module` 的显式 `importlib.import_module` 上。
> 真正用 `__getattr__` 的是**模块级**的两个地方：`utils/import_utils.py:2500` 的
> `_LazyModule.__getattr__`，和 `transformers/__init__.py:835` 给历史路径装的
> `module.__getattr__ = lambda name: getattr(_get_target(), name)`。
>
> 收益是实打实的：`import transformers` 不会 import 495 个 `modeling_*.py`；
> `AutoModel.from_pretrained("...")` 只 import `models/llama/` 这一个。

### 2.7 最后一步：交棒

`auto_factory.py:389` 拿到 `LlamaForCausalLM` 之后，`:402` 一行把活交出去：

```python
model_class = _get_model_class(config, cls._model_mapping)       # :389
...
return model_class.from_pretrained(                              # :402
    pretrained_model_name_or_path, *model_args, config=config, **hub_kwargs, **kwargs
)
```

**auto 层的职责到此结束**——它只负责「挑类」，挑完就退出。
下载权重、建模块、灌 state_dict 全是 `PreTrainedModel.from_pretrained` 的活（见 1A3）。

如果两个分支都没进（`has_remote_code` 假、`has_local_code` 假），
就是那句你可能见过的报错（`:405-407`）：

```
Unrecognized configuration class <class ...> for this kind of AutoModel: AutoModelForCausalLM.
Model type should be one of <...>.
```

**这句话的字面含义就是「`type(config)` 不在 `cls._model_mapping` 里」**——
不是模型不存在，是这个模型没有这个任务头。

---

## 三、`modeling_auto.py` 里的清单

### 3.1 文件结构：一半是表，一半是类

`modeling_auto.py` 2662 行，可以切成四块：

| 区域 | 行范围 | 内容 | 数量 |
|------|--------|------|------|
| 表区 | `:41` – `:2056` | `MODEL_*_MAPPING_NAMES = OrderedDict([...])` | 53 个 |
| 构建区 | `:2058` – `:2179` | `MODEL_*_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_*_MAPPING_NAMES)` | 54 个 |
| 类区 | `:2181` – `:2555` | `class AutoModelForXxx(_BaseAutoModelClass)` | 48 个 |
| 导出 | `:2558` – 末尾 | `__all__ = [...]` | 48 个 `AutoModel*` |

「53 个 NAMES」对「54 个 MAPPING」差一个，是因为有个特例：
`MODEL_FOR_AUDIO_TOKENIZATION_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES,
MODEL_FOR_AUDIO_TOKENIZATION_NAMES)`——它对应的表叫 `MODEL_FOR_AUDIO_TOKENIZATION_NAMES`
（`modeling_auto.py:2021`），**名字里没有 `_MAPPING` 后缀**。

**这不是纯记账问题。** `check_repo.py` 是靠后缀字符串找表的
（`:1046` 和 `:1088`，两处一样）：

```python
mapping_names = [x for x in dir(module) if x.endswith("_MAPPING_NAMES")]   # check_repo.py:1046
```

所以 `MODEL_FOR_AUDIO_TOKENIZATION_NAMES` **不会被** `check_all_auto_object_names_being_defined`
和 `check_all_auto_mapping_names_in_config_mapping_names` 检查到。
`sort_auto_mappings` 的正则（`:53`）也用了 `(_MAPPING|_MODEL_TYPE_TO_MODULE)` 前缀匹配，
同样只认带 `_MAPPING` 的名字。给新模型起名时**别学这个特例**。

### 3.2 挑四个常见的看

| Auto 类 | 行 | 对应 `_model_mapping` | 表大小 | 任务 |
|---------|-----|----------------------|--------|------|
| `AutoModel` | `:2201` | `MODEL_MAPPING` | 543 | **基座**：输出最后一层 hidden states，不加任何头 |
| `AutoModelForCausalLM` | `:2215` | `MODEL_FOR_CAUSAL_LM_MAPPING` | 176 | **自回归语言模型**：`LlamaForCausalLM`、`GlmMoeDsaForCausalLM` … |
| `AutoModelForSequenceClassification` | `:2250` | `MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING` | 124 | **整句分类**：`BertForSequenceClassification`、`LlamaForSequenceClassification` |
| `AutoModelForImageTextToText` | `:2425` | `MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING` | 85 | **图文多模态**：`LlavaNextForConditionalGeneration`、`Qwen2VLForConditionalGeneration` |
| `AutoModelForMaskedLM` | `:2232` | `MODEL_FOR_MASKED_LM_MAPPING` | 47 | **完形填空**（BERT 系），1M5 会用到 |

（表大小来自 `MODEL_*_MAPPING_NAMES` 的条目数：`MODEL_MAPPING_NAMES` 543 条
（`:41`）、`MODEL_FOR_CAUSAL_LM_MAPPING_NAMES` 176 条（`:692`）、
`MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES` 124 条（`:1378`）、
`MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES` 85 条（`:1080`）、
`MODEL_FOR_MASKED_LM_MAPPING_NAMES` 47 条（`:1191`）。）

每个类的定义体只有一行，剩下的都是文档字符串装饰：

```python
class AutoModelForCausalLM(_BaseAutoModelClass):                  # modeling_auto.py:2215
    _model_mapping = MODEL_FOR_CAUSAL_LM_MAPPING                  # :2216

    # override to give better return typehint
    @classmethod
    def from_pretrained(...) -> "_BaseModelWithGenerate": ...      # :2219-2226

AutoModelForCausalLM = auto_class_update(AutoModelForCausalLM, head_doc="causal language modeling")
```

`auto_class_update`（`auto_factory.py:480`）干的事很机械：把一个通用模板
（`CLASS_DOCSTRING` / `FROM_CONFIG_DOCSTRING` / `FROM_PRETRAINED_TORCH_DOCSTRING`）
复制一份，替换掉类名和 `head_doc` 描述，再把 `from_config` / `from_pretrained`
重新包成这个类的 classmethod。**这也是为什么 `_LazyAutoMapping.__init__` 里要写
`self._model_mapping._model_mapping = self`（`auto_factory.py:588`）**——
`replace_list_option_in_docstrings` 要靠它把「List options」替换成真实的模型清单。

### 3.3 一个反直觉的例子

`MODEL_FOR_CAUSAL_LM_MAPPING_NAMES` 里这两行（`modeling_auto.py:745-746`）：

```python
("gemma3", "Gemma3ForConditionalGeneration"),
("gemma3_text", "Gemma3ForCausalLM"),
```

**`gemma3` 这个 `model_type` 指向的是 `Gemma3ForConditionalGeneration`（多模态包装类），
不是 `Gemma3ForCausalLM`。** 后者留给了 `gemma3_text`。

这是「一个上游 checkpoint、两个子配置」的常见形态：
`Gemma3Config` 里嵌 `text_config` / `vision_config`，用 `AutoModelForCausalLM` 加载
完整 checkpoint 时走 `ForConditionalGeneration`；只想拿纯文本部分才用 `gemma3_text`。

处理这个「取子配置」动作的代码在 `from_config` / `from_pretrained` 里（`auto_factory.py:236-245`
和 `:390-401`，两处逻辑一致）：

```python
text_config_class = config.sub_configs.get("text_config", None)      # :236
if text_config_class is not None and getattr(model_class, "config_class", None) == text_config_class:
    parent_config = config                                          # :241
    config = config.get_text_config()                               # :242  ← 把 config 换成子配置
    ...
```

**判据是「选中的模型类的 `config_class` 恰好等于 `config.text_config` 的类型」**——
也就是「这个任务头只吃文本，那我把文本子配置交给它」。1A2 会展开 `sub_configs` / `get_text_config()`。

---

## 四、`auto_mappings.py` 的角色：它是**半自动生成物**

### 4.1 它里面有什么

`auto_mappings.py` 只有 **6 个顶层常量**，全部是 `OrderedDict`：

| 常量 | 行 | 条目 | 内容 |
|------|-----|------|------|
| `CONFIG_MAPPING_NAMES` | `:24` | **704** | `model_type` → Config 类名 |
| `SPECIAL_MODEL_TYPE_TO_MODULE_NAME` | `:733` | **248** | `model_type`（非自然命名）→ 模块名 |
| `VIDEO_PROCESSOR_MAPPING_NAMES` | `:986` | 30 | `model_type` → 视频处理器类名 |
| `FEATURE_EXTRACTOR_MAPPING_NAMES` | `:1021` | 27 | `model_type` → 特征提取器类名 |
| `PROCESSOR_MAPPING_NAMES` | `:1053` | 140 | `model_type` → Processor 类名 |
| `IMAGE_PROCESSOR_MAPPING_NAMES` | `:1198` | 118 | `model_type` → `{"pil": ..., "torchvision": ...}` |

**它里面没有任何 `MODEL_*_MAPPING_NAMES`。** 那 53 个模型映射表全在 `modeling_auto.py`。

### 4.2 它是生成出来的——文件头上就写着

`auto_mappings.py` 第 1-5 行是一段大写警告（由 `utils/check_auto.py:42` 的 `AUTO_GENERATED_HEADER` 拼进文件头）：

```text
#             This file was automatically generated from existing config files and their `model_type`s. Do NOT edit this file
#               manually as any edits will be overwritten by auto-generation of the file. If any change should be done,
#          please add the correct `cls.model_type` in your config class and run `python utils/check_auto.py --fix_and_overwrite`.
```

生成逻辑在 `utils/check_auto.py:69` 的 `build_config_mapping_names()`：
它用 `ast` 解析**所有** `src/transformers/models/**/configuration_*.py`，
找出每个继承 `PreTrainedConfig` 的类、读出它的 `model_type` 赋值，然后：
- `model_type == 目录名`（自然匹配）→ 记进 `CONFIG_MAPPING_NAMES`
- 否则 → 同时记进 `SPECIAL_MODEL_TYPE_TO_MODULE_NAME`（判定在 `:108-119`：`is_natural = model_type == module_name`，
  自然匹配优先且不被非自然匹配覆盖）

所以 **`model_type` 是唯一的真相来源**，映射表是它的投影。

### 4.3 分工表

| | `auto_mappings.py` | `auto_factory.py` | `modeling_auto.py` |
|---|---|---|---|
| 是什么 | **数据**（6 张名字表） | **引擎**（查表 + 惰性 import + 挑类） | **数据 + 门面**（53 张模型表 + 48 个 `AutoModelForXxx`） |
| 怎么来的 | `utils/check_auto.py` 从 config 文件生成 | 手写 | **手写**（`_MAPPING_NAMES` 要人加） |
| 谁 import 它 | `configuration_auto.py:26`、`video_processing_auto.py:37`、`processing_auto.py:31`、`feature_extraction_auto.py:26`、`image_processing_auto.py:39` | `configuration_auto.py:26`、`modeling_auto.py:21` | `modeling_auto.py:27`（从 `configuration_auto` 转发） |
| 加新模型时 | 不用管（自动生成） | 不用管 | **要手动加一行** |

**这一格差异是本章最实用的结论**：
`CONFIG_MAPPING_NAMES` 会自动长出来，但 `MODEL_FOR_*_MAPPING_NAMES` **不会**。
新模型必须在 `modeling_auto.py` 里手写指定它支持哪些任务头。

### 4.4 `check_auto.py` 的几个硬约束

| 约束 | 位置 | 触发条件 |
|------|------|---------|
| 键不能重复 | `check_auto.py:157-162` | `CONFIG_MAPPING_NAMES` 的 key 有重复 → 直接 raise |
| 值不能重复（白名单除外） | `check_auto.py:164-173` | value 有重复且不在 `IGNORE_DUPLICATE_CONFIG`（`:64`，只有 3 个：`GPT2Config`、`EvollaConfig`、`MLCDVisionConfig`）→ raise |
| 不能手工重复登记 | `check_auto.py:269`（`check_duplicates`） | 你在 `processing_auto.py` 的 `MISSING_*` 里写的 key 和自动生成的重叠 → raise |

第二条的原因写得很直白（`check_auto.py:155-156` 的注释）：

> The config mapping has to be one-to-one for correct `AutoConfig.from_pretrained()` because
> `LazyMapping` reverts keys/values and creates a dict from it.

也就是 2.4 里 `_reverse_config_mapping = {v: k for k, v in config_mapping.items()}` 那一行——
**一旦有两个 `model_type` 指同一个 Config 类名，反查表就会被覆盖掉一个。**

---

## 五、`None` 值到底表示什么

网上很多讲 auto 机制的文章说「mapping 里的 `None` 表示该模型不支持这个任务头」。
**在你正在读的这个版本里，这句话对模型映射是不成立的**，得分开说。

### 5.1 模型映射（`modeling_auto.py`）：没有 `None`

```bash
$ grep -c "None" src/transformers/models/auto/modeling_auto.py
0
```

53 张 `MODEL_*_MAPPING_NAMES` 表里**一个 `None` 都没有**。

「这个模型不支持这个任务头」的表达方式是——**它干脆不出现在那张表里**。
后果就是 2.7 里那句 `Unrecognized configuration class ...`。
想确认某个模型支不支持某个头，正确做法是查表，而不是找 `None`：

```python
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
"glm_moe_dsa" in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES     # → True（modeling_auto.py:760）
```

### 5.2 tokenizer / processor 映射：`None` = **可选依赖没装**

真正会**在运行时把 `None` 写进表里**的是另外几处，含义完全不同：

| 位置 | 代码 | `None` 的含义 |
|------|------|--------------|
| `tokenization_auto.py:67` 起 | `("EvollaModel", "TokenizersBackend" if is_tokenizers_available() else None),`（同形 400+ 条） | `tokenizers` 没装 → 这类本次不可用 |
| `tokenization_auto.py:421` | `TOKENIZER_MAPPING_NAMES[model_type] = "TokenizersBackend" if is_tokenizers_available() else None` | 同上（补漏，`:420` 的 for 循环里） |
| `video_processing_auto.py:76-80` | 注释原文：`# If the torchvision is not available, we set it to None`，紧跟 `fast_video_processor_class = None` | `torchvision` 没装 |
| `processing_auto.py:48` / `image_processing_auto.py:67` | **仅类型标注**：`OrderedDict[str, str \| None]` | 表示「允许是 `None`」，不是「表里有 `None`」——这两张表的实际内容都是字符串 / dict |

**而且这个 `None` 是被下游显式处理的**——`getattribute_from_module` 开头两行
（`auto_factory.py:522-523`）：

```python
if attr is None:        # :522
    return None         # :523
```

所以 `_LazyAutoMapping[...]` 查到一个 `None` 名字时**静默返回 `None`**，不报错。
调用方得自己判空，例如 `tokenization_auto.py:816`：

```python
and TOKENIZER_MAPPING_NAMES.get(config_model_type) is not None
```

### 5.3 一句话对照

| | 模型映射 | tokenizer/processor 映射 |
|---|---|---|
| `None` 出现 | **从不** | 可选依赖缺失时 |
| 「不支持」怎么表达 | **不出现该 key** | — |
| 不支持时的报错 | `ValueError: Unrecognized configuration class ...` | 静默返回 `None`，由调用方判空 |
| 类型标注 | `_LazyAutoMappingValue = tuple[type \| None, type \| None]`（`auto_factory.py:49`） | `str \| None` |

> `auto_factory.py:49` 那个 `tuple[type[Any] | None, type[Any] | None]` 是**查表返回值的类型标注**，
> 不是「表里有 `None`」——`| None` 只是说「名字查不到对应类时可能拿到 `None`」。
> `getattribute_from_module:524-525` 对 tuple 的处理服务于 2.5 讲的
> `("funnel", ("FunnelModel", "FunnelBaseModel"))` 这类多候选写法。

---

## 六、实操：把自己的模型接进 auto

### 6.1 最小清单

假设你的模型叫 `MyModel`，`model_type = "mymodel"`，目录 `src/transformers/models/mymodel/`。

| # | 要做的事 | 文件 | 为什么 |
|---|---------|------|--------|
| 1 | 目录名 = `model_type`（`-` 换 `_`） | `models/mymodel/` | `model_type_to_module_name` 靠这个找目录（`configuration_auto.py:64`） |
| 2 | 写 `class MyModelConfig(PreTrainedConfig)`，类体里 `model_type = "mymodel"` | `configuration_mymodel.py` | `check_auto.py:84-104` 从这里读 `model_type` |
| 3 | 目录名不自然就登记 `SPECIAL_MODEL_TYPE_TO_MODULE_NAME` | `configuration_auto.py:50` 的 `.update()` | 生成脚本管不到的情况 |
| 4 | 写 `MyModelForCausalLM` 等 | `modeling_mymodel.py` | 模型本体 |
| 5 | **在 `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES` 里加一行 `("mymodel", "MyModelForCausalLM"),`** | `modeling_auto.py:692` 起那张表 | **必须手动**，脚本不会帮你 |
| 6 | 有 tokenizer 就加 `TOKENIZER_MAPPING_NAMES` 一条 | `tokenization_auto.py:65` 起 | 手动 |
| 7 | 有 Processor 就加 `PROCESSOR_MAPPING_NAMES`（或 `MISSING_PROCESSOR_MAPPING_NAMES`） | `processing_auto.py:50` | 手写 |
| 8 | `__init__.py` | `models/mymodel/__init__.py` | 照抄 `llama/__init__.py`，**不用列类名**（`define_import_structure` 会扫，`:1.4`） |

第 5 步加完之后**不用管 `auto_mappings.py`**——第 2 步一落地，
`CONFIG_MAPPING_NAMES` 会在下次生成时自动多出 `("mymodel", "MyModelConfig")`。

### 6.2 加了表就要**排序**

`MODEL_*_MAPPING_NAMES` 和 `CONFIG_MAPPING_NAMES` 都要求**按键名 ASCII 升序**。
checker 名是 **`sort_auto_mappings`**（不是 `check_auto`）：

```makefile
# Makefile:13
STYLE_CHECKERS := ruff_check, ruff_format, init_isort, sort_auto_mappings
```

实现是 `utils/sort_auto_mappings.py`：

```python
CHECKER_CONFIG = {                                              # sort_auto_mappings.py:37
    "name": "sort_auto_mappings",                               # :38  ← 这就是 checker 名
    "label": "Sort auto mappings",                              # :39
    "cache_globs": ["src/transformers/models/auto/*.py"],       # :40  ← 整个 auto 目录都归它管
    "check_args": ["--check_only"],                             # :41
    "fix_args": [],                                             # :42  ← 不可自动修，只能 make style
}
```

它匹配的正则是 `_re_intro_mapping`（`:53`）：

```python
_re_intro_mapping = re.compile(r"[A-Z_]+(_MAPPING|_MODEL_TYPE_TO_MODULE)(\s+|_[A-Z_]+\s+)=\s+OrderedDict(?!\(\*\))")
```

注意 `(?!\(\*\))` 这个负向断言——**用 `OrderedDict(*list(...))` 展开别表的那张表会被跳过**，
正是 `MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES`（`modeling_auto.py:1172`：

```python
MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES = OrderedDict(
    [
        *list(MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.items()),   # :1174
        ("glmasr", "GlmAsrForConditionalGeneration"),
        ...
```

）不会被排序、也不该被排序的原因。

`cache_globs` 是 `src/transformers/models/auto/*.py`（`:40`），
所以它**同时管 `auto_mappings.py` 和 `modeling_auto.py`**。

### 6.3 `make check-repo` 里的两个 auto checker

从 `Makefile:17` 起，`REPO_CONSISTENCY_CHECKERS` 的第一项就是 `auto_mappings`：

```makefile
REPO_CONSISTENCY_CHECKERS := \
	auto_mappings, \
	imports, \
	...
```

| checker 名 | 实现文件 | 检查什么 | 报错时怎么修 |
|-----------|---------|---------|-------------|
| `auto_mappings` | `utils/check_auto.py`（`CHECKER_CONFIG` 在 `:34`，`name` 是 `"auto_mappings"`） | `auto_mappings.py` 能否从 config 文件重新生成出一份**逐字节相同**的内容 | `python utils/check_auto.py --fix_and_overwrite` 或 `make fix-repo` |
| `sort_auto_mappings` | `utils/sort_auto_mappings.py`（`:37`） | `src/transformers/models/auto/*.py` 里所有 `*_MAPPING_NAMES` 是否按键升序 | `make style`（`fix_args` 为空，checker 自己不会修） |

`check_auto.py` 的判定方式是**现场重新生成一遍再 diff**（`main()`，`:278`；diff 段落在 `:306-324`）：

```python
if old_content != new_content:                          # :306
    if not overwrite:
        diff = "".join(                                 # :308
            difflib.unified_diff(...)
        )
        raise Exception(                                # :317
            "Generated auto-mapping is not consistent with the contents of `models/auto/auto_mappings.py`.\n"
            "Run `make fix-repo` or `python utils/check_auto.py --fix_and_overwrite` to fix them.\n\n"
            f"Diff (on disk → regenerated):\n{diff}"
        )
    else:
        with open(AUTO_FILENAME, "w") as f:             # :323
            f.write(new_content)
```

注意生成前后都会跑 `run_ruff_and_sort()`（`:234`）——先 `sort_auto_mapping(overwrite=True)`
再 `ruff check --fix` + `ruff format`。**这就是它和 `sort_auto_mappings` checker 的接口**：
`auto_mappings.py` 的排序由 `sort_auto_mappings.py` 提供函数，由 `check_auto.py` 调用。

### 6.4 还有第三个 checker 在管 auto：`repo`

`check_repo.py` 里有**三个**专门针对 auto 映射的检查函数，全在 `repo` checker 下：

| 函数 | 行 | 检查什么 |
|------|-----|---------|
| `check_all_auto_object_names_being_defined()` | `check_repo.py:1031` | 映射表里的**类名**必须真的存在于库里（`hasattr(transformers, class_name)`），`None` 直接 `continue`（`:1056-1057`） |
| `check_all_auto_mapping_names_in_config_mapping_names()` | `:1073` | 映射表里的 **key** 必须在 `CONFIG_MAPPING_NAMES` 里（否则报「`xxx` appears in the mapping ... but it is not defined in the keys of `CONFIG_MAPPING_NAMES`」） |
| `check_all_auto_mappings_importable()` | `:1104` | 每张 `_MAPPING_NAMES` 都要能 import |

外加 `check_models_are_auto_configured()`（`:991`）——**每个模型必须至少出现在一张 auto 表里**：

```python
for model_name, _ in defined_models:
    if model_name not in all_auto_models and not ignore_unautoclassed(model_name):
        failures.append(
            f"{model_name} is defined in {module.__name__} but is not present in any of the auto mapping. "
            "If that is intended behavior, add its name to `IGNORE_NON_AUTO_CONFIGURED` in the file "
            "`utils/check_repo.py`."
        )
```

它的模型清单来自 `get_all_auto_configured_models()`（`:970`），
遍历 `MODEL_*MAPPING_NAMES` 用 `get_values()`（`auto_factory.py:510`）摊平。

> 所以「忘了在 `modeling_auto.py` 里加一行」这件事，CI **有两种方式抓到**：
> 第 6.3 节的 `auto_mappings` 抓不到（它只管 config 映射），
> 但 `repo` checker 的 `check_models_are_auto_configured` 会抓到你。
> 白名单常量是 `IGNORE_NON_AUTO_CONFIGURED`（`check_repo.py:348`）。

### 6.5 三个命令的边界

| 命令 | 定义 | 行为 |
|------|------|------|
| `make check-repo` | `Makefile:58` → `utils/checkers.py $(ALL_CHECKERS) --keep-going` | 全跑，**只报不改** |
| `make fix-repo` | `Makefile:62` → `... --fix --keep-going` | 全跑并自动修（会重写 `auto_mappings.py`） |
| `make style` | `Makefile:42` → `utils/checkers.py $(STYLE_CHECKERS) --fix` | 含 `sort_auto_mappings`，**这是唯一能修排序的入口** |

`utils/checkers.py` 用 `ast` 扫描 `utils/*.py` 里的 `CHECKER_CONFIG` 字典来发现 checker
（`_discover_checkers()`，`checkers.py:108`），所以「checker 名」就是各文件 `CHECKER_CONFIG["name"]` 的值——
本篇出现的三个：`auto_mappings`（`check_auto.py:35`）、`sort_auto_mappings`
（`sort_auto_mappings.py:38`）、`repo`（`check_repo.py:55`）。

---

## 一句话总结

> **`models/<name>/` 里的文件名就是「职责 + 命名规范」的乘积**：
> `<职责前缀>_<模块名>.py`，模块名由 `model_type` 把 `-` 换成 `_` 得来，
> 不自然的对应关系记在 `SPECIAL_MODEL_TYPE_TO_MODULE_NAME`（248 条）里。
> `tokenization` / `processing` / `modular` 都是可选的。
>
> **auto 机制是两次查表 + 两次惰性 import**：
> `config.json` 的 `model_type` → `CONFIG_MAPPING_NAMES` → Config 类 →
> `MODEL_FOR_*_MAPPING_NAMES` → 模型类，由 `_LazyAutoMapping` 在
> `_load_attr_from_module`（`auto_factory.py:612`）里按需 `importlib.import_module`。
> **没有 `__getattr__` 魔法**——惰性来自显式 import，模块级 `__getattr__` 属于 `_LazyModule`。
>
> **`auto_mappings.py` 是生成物，`modeling_auto.py` 是手写物**：
> 前者由 `utils/check_auto.py` 从 `configuration_*.py` 反推，
> 后者那 53 张 `MODEL_*_MAPPING_NAMES` 必须人肉维护。
> 模型映射里**没有 `None`**——「不支持这个头」的表达方式是「不在表里」，
> 结果是 `ValueError: Unrecognized configuration class ...`。

---

## 验证清单

下面每一条都在仓库根目录实际执行过，输出为真实粘贴。用到的命令只有
`grep` / `sed` / `awk` / `wc` / `ls`，加上 `python -B` + `ast`（**静态解析源码，不 import `transformers`**）。

### 1. 文件规模与目录计数

```bash
$ wc -l src/transformers/models/auto/*.py
     33 src/transformers/models/auto/__init__.py
    699 src/transformers/models/auto/auto_factory.py
   1328 src/transformers/models/auto/auto_mappings.py
    456 src/transformers/models/auto/configuration_auto.py
    373 src/transformers/models/auto/feature_extraction_auto.py
    756 src/transformers/models/auto/image_processing_auto.py
   2662 src/transformers/models/auto/modeling_auto.py
    366 src/transformers/models/auto/processing_auto.py
   1014 src/transformers/models/auto/tokenization_auto.py
    417 src/transformers/models/auto/video_processing_auto.py
   8104 total

$ ls -d src/transformers/models/*/ | wc -l                      # 510 个目录
$ ls src/transformers/models/*/configuration_*.py | wc -l       # 497
$ ls src/transformers/models/*/modeling_*.py | wc -l            # 495
$ ls src/transformers/models/*/modular_*.py | wc -l             # 288
$ ls src/transformers/models/*/tokenization_*.py | wc -l        # 101
$ ls src/transformers/models/*/processing_*.py | wc -l          # 151
$ ls src/transformers/models/*/image_processing_*.py | wc -l    # 217
$ ls src/transformers/models/*/image_processing_pil_*.py | wc -l # 91
$ ls src/transformers/models/*/video_processing_*.py | wc -l    # 31
$ ls src/transformers/models/*/feature_extraction_*.py | wc -l  # 33
$ ls src/transformers/models/*/generation_*.py | wc -l          # 11
```

> `ls -d src/transformers/models/*/` 计的是目录数（含 `deprecated/` 本身，不含其子目录）。
> `modeling_*.py` 比目录少 15 个，因为 `barthez` / `bert_japanese` / `byt5` / `code_llama` /
> `phobert` / `nllb` 等是纯 tokenizer 目录。

### 2. 四个真实目录的构成

```bash
$ ls src/transformers/models/llama/ | grep -v convert
__init__.py
configuration_llama.py
modeling_llama.py
tokenization_llama.py

$ ls src/transformers/models/mistral/ | grep -v convert
__init__.py
configuration_mistral.py
modeling_mistral.py
modular_mistral.py

$ ls src/transformers/models/qwen2_vl/ | grep -v convert
__init__.py
configuration_qwen2_vl.py
image_processing_pil_qwen2_vl.py
image_processing_qwen2_vl.py
modeling_qwen2_vl.py
processing_qwen2_vl.py
video_processing_qwen2_vl.py
```

### 3. 命名规范（`model_type` → 模块名）

```bash
$ grep -n "def model_type_to_module_name" -A 12 src/transformers/models/auto/configuration_auto.py
64:def model_type_to_module_name(key) -> str:
65-    """Converts a config key to the corresponding module."""
66-    # Special treatment
67-    if key in SPECIAL_MODEL_TYPE_TO_MODULE_NAME:
68-        key = SPECIAL_MODEL_TYPE_TO_MODULE_NAME[key]
69-
70-        if key in DEPRECATED_MODELS:
71-            key = f"deprecated.{key}"
72-        return key
73-
74-    key = key.replace("-", "_")
75-    if key in DEPRECATED_MODELS:
76-        key = f"deprecated.{key}"
```

```bash
$ sed -n 744p src/transformers/models/auto/auto_mappings.py
        ("bert-generation", "bert_generation"),

$ sed -n 746,748p src/transformers/models/auto/auto_mappings.py
        ("blip-2", "blip_2"),
        ("blip_2_qformer", "blip_2"),
        ("blip_2_vision_model", "blip_2"),

$ sed -n 857,858p src/transformers/models/auto/auto_mappings.py
        ("llama4_text", "llama4"),
        ("llama4_vision_model", "llama4"),

$ ls -d src/transformers/models/blenderbot_small/ && ls src/transformers/models/blenderbot_small/ | grep configuration
src/transformers/models/blenderbot_small/
configuration_blenderbot_small.py
```

### 4. `auto_mappings.py` 的 6 张表（`ast` 静态计数）

```text
CONFIG_MAPPING_NAMES                     704 entries   (line 24)
SPECIAL_MODEL_TYPE_TO_MODULE_NAME        248 entries   (line 733)
VIDEO_PROCESSOR_MAPPING_NAMES             30 entries   (line 986)
FEATURE_EXTRACTOR_MAPPING_NAMES           27 entries   (line 1021)
PROCESSOR_MAPPING_NAMES                  140 entries   (line 1053)
IMAGE_PROCESSOR_MAPPING_NAMES            118 entries   (line 1198)
```

### 5. `modeling_auto.py` 的结构计数

```bash
$ grep -c "^class AutoModel" src/transformers/models/auto/modeling_auto.py
48
$ grep -c "= _LazyAutoMapping" src/transformers/models/auto/modeling_auto.py
54
$ grep -n '^class AutoModel' src/transformers/models/auto/modeling_auto.py | head
2201:class AutoModel(_BaseAutoModelClass):
2208:class AutoModelForPreTraining(_BaseAutoModelClass):
2215:class AutoModelForCausalLM(_BaseAutoModelClass):
...
2425:class AutoModelForImageTextToText(_BaseAutoModelClass):
```

`ast` 统计顶层赋值：`MAPPING_NAMES` 53 个、`MAPPING` 54 个。唯一对不上的一对是
`MODEL_FOR_AUDIO_TOKENIZATION_MAPPING`（对应的表叫 `MODEL_FOR_AUDIO_TOKENIZATION_NAMES`，
`modeling_auto.py:2021`，少了 `_MAPPING` 后缀）。

各表条目数：

```text
MODEL_MAPPING_NAMES = 543 entries (line 41)
MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = 176 entries (line 692)
MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES = 85 entries (line 1080)
MODEL_FOR_MASKED_LM_MAPPING_NAMES = 47 entries (line 1191)
MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES = 124 entries (line 1378)
```

### 6. `_LazyAutoMapping` 的真实机制

```bash
$ grep -n "def _get_model_class\|class _LazyAutoMapping\|def _load_attr_from_module\|def getattribute_from_module\|def __getitem__" src/transformers/models/auto/auto_factory.py
178:def _get_model_class(config, model_mapping):
521:def getattribute_from_module(module, attr):
575:class _LazyAutoMapping(OrderedDict[type[PreTrainedConfig], _LazyAutoMappingValue]):
596:    def __getitem__(self, key: type[PreTrainedConfig]) -> _LazyAutoMappingValue:
612:    def _load_attr_from_module(self, model_type, attr):

$ sed -n 612,616p src/transformers/models/auto/auto_factory.py
    def _load_attr_from_module(self, model_type, attr):
        module_name = model_type_to_module_name(model_type)
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(f".{module_name}", "transformers.models")
        return getattribute_from_module(self._modules[module_name], attr)
```

**`_LazyAutoMapping` 里没有 `__getattr__`**：

```bash
$ grep -n "__getattr__" src/transformers/models/auto/auto_factory.py
683:        # Register the new mapping (this will always take precedence in __getattr__ and __contains__ compared to base mapping)
```

全文唯一出现是第 683 行的一句注释。模块级 `__getattr__` 的真实位置：

```bash
$ grep -n "__getattr__" src/transformers/utils/import_utils.py src/transformers/__init__.py
src/transformers/utils/import_utils.py:2500:    def __getattr__(self, name: str) -> Any:       # _LazyModule.__getattr__
src/transformers/__init__.py:835:        module.__getattr__ = lambda name: getattr(_get_target(), name)
```

### 7. 映射表里的 tuple 值只有 4 处

`ast` 扫 `modeling_auto.py` 每张 `*_MAPPING_NAMES`，找 value 是 tuple 的条目：

```text
TUPLE VALUE: MODEL_MAPPING_NAMES line 183 ('funnel', ('FunnelModel', 'FunnelBaseModel'))
TUPLE VALUE: MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES line 966 ('deit', ('DeiTForImageClassification', 'DeiTForImageClassificationWithTeacher'))
TUPLE VALUE: MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES line 980 ('levit', ('LevitForImageClassification', 'LevitForImageClassificationWithTeacher'))
TUPLE VALUE: MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES line 989 ('perceiver', ('PerceiverForImageClassificationLearned', 'PerceiverForImageClassificationFourier', 'PerceiverForImageClassificationConvProcessing'))
any tuple values in modeling_auto mappings: True
```

```bash
$ sed -n 183p src/transformers/models/auto/modeling_auto.py
        ("funnel", ("FunnelModel", "FunnelBaseModel")),

$ sed -n 980,983p src/transformers/models/auto/modeling_auto.py
        (
            "levit",
            ("LevitForImageClassification", "LevitForImageClassificationWithTeacher"),
        ),

$ sed -n 1200p src/transformers/models/auto/auto_mappings.py   # dict 值
        ("aria", {"pil": "AriaImageProcessorPil", "torchvision": "AriaImageProcessor"}),
```

### 8. 映射表里真的没有 `None`

```bash
$ grep -c "None" src/transformers/models/auto/modeling_auto.py
0

$ grep -n "is_tokenizers_available() else None" src/transformers/models/auto/tokenization_auto.py | head -3
67:        ("EvollaModel", "TokenizersBackend" if is_tokenizers_available() else None),
68:        ("aimv2", "CLIPTokenizer" if is_tokenizers_available() else None),
69:        ("albert", "AlbertTokenizer" if is_tokenizers_available() else None),

$ sed -n 76,80p src/transformers/models/auto/video_processing_auto.py
    # If the torchvision is not available, we set it to None
    if not is_torchvision_available():
        fast_video_processor_class = None

    VIDEO_PROCESSOR_MAPPING_NAMES[model_type] = fast_video_processor_class

$ sed -n 521,525p src/transformers/models/auto/auto_factory.py   # None 被静默接受
def getattribute_from_module(module, attr):
    if attr is None:
        return None
    if isinstance(attr, tuple):
```

### 9. 具体条目行号

```bash
$ grep -n '"llama"' src/transformers/models/auto/auto_mappings.py | head -1
356:        ("llama", "LlamaConfig"),

$ grep -n '"llama", "Llama' src/transformers/models/auto/modeling_auto.py
299:        ("llama", "LlamaModel")
789:        ("llama", "LlamaForCausalLM"),
1438:        ("llama", "LlamaForSequenceClassification"),
1543:        ("llama", "LlamaForQuestionAnswering"),
1663:        ("llama", "LlamaForTokenClassification"),

$ grep -n '"gemma3", "Gemma3ForConditionalGeneration"\|"gemma3_text", "Gemma3ForCausalLM"' \
      src/transformers/models/auto/modeling_auto.py
620:        ("gemma3", "Gemma3ForConditionalGeneration"),     # MODEL_FOR_PRETRAINING_MAPPING_NAMES
745:        ("gemma3", "Gemma3ForConditionalGeneration"),     # MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
746:        ("gemma3_text", "Gemma3ForCausalLM"),
1102:        ("gemma3", "Gemma3ForConditionalGeneration"),     # MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

$ grep -n '"llava_next", "LlavaNextForConditionalGeneration"\|"qwen2_vl", "Qwen2VLForConditionalGeneration"' \
      src/transformers/models/auto/modeling_auto.py
636:        ("llava_next", "LlavaNextForConditionalGeneration"),      # MODEL_FOR_PRETRAINING_MAPPING_NAMES
1131:        ("llava_next", "LlavaNextForConditionalGeneration"),     # MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
1151:        ("qwen2_vl", "Qwen2VLForConditionalGeneration"),         # 同上

$ grep -n '"glm_moe_dsa"' src/transformers/models/auto/auto_mappings.py src/transformers/models/auto/modeling_auto.py
src/transformers/models/auto/auto_mappings.py:253:        ("glm_moe_dsa", "GlmMoeDsaConfig"),
src/transformers/models/auto/modeling_auto.py:218:        ("glm_moe_dsa", "GlmMoeDsaModel"),
src/transformers/models/auto/modeling_auto.py:760:        ("glm_moe_dsa", "GlmMoeDsaForCausalLM"),
```

### 10. checker 名与注册位置

```bash
$ grep -n "auto_mappings\|sort_auto_mappings" Makefile
10:# so they can never drift out of sync (e.g. silently dropping `auto_mappings`
13:STYLE_CHECKERS := ruff_check, ruff_format, init_isort, sort_auto_mappings
18:	auto_mappings, \

$ grep -n "^style:\|^check-repo:\|^fix-repo:" Makefile
42:style:
58:check-repo:
62:fix-repo:
```

```bash
$ grep -n "CHECKER_CONFIG" -A 6 utils/check_auto.py | head -8
34:CHECKER_CONFIG = {
    "name": "auto_mappings",
    "label": "Generate auto mappings",
    "cache_globs": [],
    "check_args": [],
    "fix_args": ["--fix_and_overwrite"],
}

$ grep -n "CHECKER_CONFIG" -A 6 utils/sort_auto_mappings.py | head -8
37:CHECKER_CONFIG = {
38-    "name": "sort_auto_mappings",
39-    "label": "Sort auto mappings",
40-    "cache_globs": ["src/transformers/models/auto/*.py"],
41-    "check_args": ["--check_only"],
42-    "fix_args": [],
43-}

$ sed -n 55p utils/check_repo.py
    "name": "repo",
```

`check_repo.py` 的三个 auto 检查函数：

```bash
$ grep -n "def check_all_auto_object_names_being_defined\|def check_all_auto_mapping_names_in_config_mapping_names\|def check_all_auto_mappings_importable\|def check_models_are_auto_configured\|def get_all_auto_configured_models" utils/check_repo.py
970:def get_all_auto_configured_models() -> list[str]:
991:def check_models_are_auto_configured(module: types.ModuleType, all_auto_models: list[str]) -> list[str]:
1031:def check_all_auto_object_names_being_defined():
1073:def check_all_auto_mapping_names_in_config_mapping_names():
1104:def check_all_auto_mappings_importable():

$ sed -n 20,21p utils/check_repo.py       # 文件头自述它管什么
- checking all models are in at least one auto class
- checking all the auto mapping are properly defined (no typos, importable)

$ sed -n 1046p utils/check_repo.py        # 靠后缀找表 → 漏掉 MODEL_FOR_AUDIO_TOKENIZATION_NAMES
    mapping_names = [x for x in dir(module) if x.endswith("_MAPPING_NAMES")]
```

### 11. 生成脚本的关键行

```bash
$ grep -n "AUTO_FILENAME\|def build_config_mapping_names\|def main\|def run_ruff_and_sort" utils/check_auto.py
66:AUTO_FILENAME = "src/transformers/models/auto/auto_mappings.py"
69:def build_config_mapping_names() -> tuple[dict, dict]:
234:def run_ruff_and_sort(file: str):
278:def main(overwrite: bool):

$ sed -n 76,77p utils/check_auto.py
    # `glob.glob` is filesystem-order dependent — sort to make the output deterministic.
    all_files = sorted(glob.glob("src/transformers/models/**/configuration_*.py", recursive=True))

$ sed -n 25p utils/check_auto.py           # 两个 checker 之间的接口
from sort_auto_mappings import sort_auto_mapping

$ awk 'NR>=306 && NR<=324 {printf "%d:%s\n", NR, $0}' utils/check_auto.py
306:    if old_content != new_content:
307:        if not overwrite:
308:            diff = "".join(
309:                difflib.unified_diff(
310:                    old_content.splitlines(keepends=True),
311:                    new_content.splitlines(keepends=True),
312:                    fromfile=f"{AUTO_FILENAME} (on disk)",
313:                    tofile=f"{AUTO_FILENAME} (regenerated)",
314:                    n=3,
315:                )
316:            )
317:            raise Exception(
318:                "Generated auto-mapping is not consistent with the contents of `models/auto/auto_mappings.py`.\n"
319:                "Run `make fix-repo` or `python utils/check_auto.py --fix_and_overwrite` to fix them.\n\n"
320:                f"Diff (on disk → regenerated):\n{diff}"
321:            )
322:        else:
323:            with open(AUTO_FILENAME, "w") as f:
324:                f.write(new_content)
```

```bash
$ sed -n 65,67p src/transformers/models/auto/tokenization_auto.py
TOKENIZER_MAPPING_NAMES = OrderedDict[str, str | None](
    [
        ("EvollaModel", "TokenizersBackend" if is_tokenizers_available() else None),

$ sed -n 419,421p src/transformers/models/auto/tokenization_auto.py
for model_type in MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS:
    if model_type not in TOKENIZER_MAPPING_NAMES:
        TOKENIZER_MAPPING_NAMES[model_type] = "TokenizersBackend" if is_tokenizers_available() else None
```

### 12. `__init__.py` 不手写导出

```bash
$ wc -l src/transformers/models/llama/__init__.py
      28 src/transformers/models/llama/__init__.py

$ awk 'NR>=20 && NR<=28 {printf "%d:%s\n", NR, $0}' src/transformers/models/llama/__init__.py
20:if TYPE_CHECKING:
21:    from .configuration_llama import *
22:    from .modeling_llama import *
23:    from .tokenization_llama import *
24:else:
25:    import sys
26:
27:    _file = globals()["__file__"]
28:    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)

$ grep -n "def define_import_structure" src/transformers/utils/import_utils.py
3301:def define_import_structure(module_path: str, prefix: str | None = None) -> IMPORT_STRUCTURE_T:
```

### 未能验证的部分（诚实声明）

- **没有做运行时验证**。本环境缺少依赖，我尝试的
  `PYTHONPATH=src python -B -c "from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING"`
  直接失败：

  ```text
  File "/Users/baai/Desktop/baai_proj/transformers/src/transformers/utils/auto_docstring.py", line 25, in <module>
      import regex as re
  ModuleNotFoundError: No module named 'regex'
  ```

  所以「`MODEL_FOR_CAUSAL_LM_MAPPING[LlamaConfig]` 返回 `LlamaForCausalLM`」、
  「`funnel` 走到 `:191` 兜底」这类结论是从源码逐行读出来的，**没有实际执行**。
  本文所有行号、常量名、条目数均由 `grep` / `sed` / `awk` / `wc` / `ls` 实测
  （条目数用 `python -B` + `ast` 静态解析源码，不 import `transformers`）。
- **没有运行任何会写文件的命令**。`make fix-repo`、`make style`、
  `python utils/check_auto.py --fix_and_overwrite`、
  `python utils/sort_auto_mappings.py`（无 `--check_only`）、
  `python utils/modular_model_converter.py` 全部**未执行**。

---

## 导航

- 路线图：[00_roadmap.md](./00_roadmap.md)（本篇 = 入门 · 架构 · 1A1）
- 上一篇：无（架构线起点）
- 下一篇：[1A2_config_system.md](./1A2_config_system.md)
- **建议补读**：[2A5_modular_code_generation.md](./2A5_modular_code_generation.md)——
  本篇讲清了「一个模型目录里有什么」，2A5 讲清「其中哪些文件是手写的、哪些是生成的」。
  两篇合起来，你翻任何一个模型目录都不会迷路。
