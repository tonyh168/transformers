# 仓库骨架与 auto 机制精读

> 阅读目标：① `src/transformers/models/<name>/` 目录里该有哪些文件、各负责什么、哪些可选；
> ② `AutoModel.from_pretrained("meta-llama/Llama-3-8B")` 只拿到一个字符串和一份 `config.json`，凭什么知道要 `import` 哪个文件的哪个类。
> 这是架构线**第一篇**——它不讲任何模型长什么样（那是 1M 线的活）。
> 源文件：`src/transformers/models/auto/` 下的 `auto_factory.py`（699 行）、`auto_mappings.py`（1328 行）、`modeling_auto.py`（2662 行）、`configuration_auto.py`（456 行），加上 `utils/check_auto.py`（331 行）、`utils/sort_auto_mappings.py`（130 行）、`Makefile`

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

无。可选先读 [2A5_modular_code_generation.md](./2A5_modular_code_generation.md)；模型结构见 [1M1_llama_structure.md](./1M1_llama_structure.md)。

---

## 一、物理骨架：一个模型目录里有什么

真实四例（完整 `ls` 输出见验证清单 §2）：`llama/` = `__init__.py` + `configuration_` + `modeling_` + `tokenization_`（另有 `convert_*` 脚本，不被 import）；`mistral/` 多一个 `modular_mistral.py`；`llava_next/`、`qwen2_vl/` 多 `processing_` 与 `image_processing_[_pil_]`（`qwen2_vl` 还有 `video_processing_`）。**没有哪个模型同时拥有全部文件**——「五件套」是一份**候选清单**，不是必填清单。

### 1.1 五件套与同族文件（括号 = 全库数量）

- `configuration_<name>.py`（497 / 510 目录，几乎必须）：`<Name>Config(PreTrainedConfig)`，**一切的入口**，类体里 `model_type = "llama"` 是 auto 机制唯一的锚点；`modeling_<name>.py`（495）：`<Name>Model`、`<Name>ForCausalLM` … 模型本体（1M 线的主角）。
- `tokenization_<name>.py`（101）、`processing_<name>.py`（151，`<Name>Processor(ProcessorMixin)` 把 tokenizer + image/video processor 打包成一个 `__call__`）、`modular_<name>.py`（288，手写、允许继承的源文件，由脚本生成上面几个，见 [2A5](./2A5_modular_code_generation.md)）——**均可选**。
- 同族：`image_processing_<name>.py`（217）、`image_processing_pil_<name>.py`（91）、`feature_extraction_<name>.py`（33）、`video_processing_<name>.py`（31）、`generation_<name>.py`（11，如 `whisper` / `musicgen`）；`convert_<name>_weights_to_hf.py` **不是库的一部分**，不会被 import。
  `modeling_*.py`（495）比目录（510）少的 15 个是**纯 tokenizer 目录**（`barthez` / `bert_japanese` / `byt5` / `phobert` / `nllb` / `code_llama` / `wav2vec2_with_lm` 等），它们复用别的模型的 `modeling`；`deprecated/` 是子目录不是模型。

### 1.2 命名规范：文件名用的是「模块名」，不是 `model_type`

`configuration_auto.py:64` 的 `model_type_to_module_name()`：先查 `SPECIAL_MODEL_TYPE_TO_MODULE_NAME`（`:67-68`），再把 `-` 换成 `_`（`:74`）。所以 `bert-generation` → `bert_generation/`（特殊表里也记一笔，`auto_mappings.py:744`）、`blenderbot-small` → `blenderbot_small/`；`llama4_text` / `llama4_vision_model` 都在 `llama4/`（`auto_mappings.py:857-858`）、`blip-2` / `blip_2_qformer` / `blip_2_vision_model` 都在 `blip_2/`（`auto_mappings.py:746-748`）。

`SPECIAL_MODEL_TYPE_TO_MODULE_NAME` 有 **248 条**（`auto_mappings.py:733` 起）——**绝大多数多模态模型都在这里留名**，它们的 `model_type` 是 `xxx_text` / `xxx_vision_model`，目录却只叫 `xxx`。实操：看到 `"model_type": "llama4_text"`，**不要去**找 `models/llama4_text/`——真模块是 `models/llama4/`。

### 1.3 `__init__.py` 里没有 export 语句

`llama/__init__.py` 全文 28 行：`TYPE_CHECKING` 分支是给静态检查看的假 `import *`（`:20-23`），真正执行的是 `else` 分支的 `sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)`（`:28`）。`define_import_structure()`（`utils/import_utils.py:3301`）**扫描该目录下的文件**、读出每个 `__all__`，自动拼出「类名 → 模块」表——类清单是**扫出来的，不是手写的**。想知道一个模型对外暴露什么，去读 `modeling_*.py` 的 `__all__`。

---

## 二、核心问题：`AutoModel.from_pretrained("xxx")` 怎么知道用哪个类

### 2.1 三层跳转

```
config.json 里的 "model_type": "llama"
 ① AutoConfig.from_pretrained() 读 json，拿 model_type
 ▼  CONFIG_MAPPING_NAMES ──"llama"──→ "LlamaConfig"                    （字符串表，auto_mappings.py:356）
 ② CONFIG_MAPPING[...] 惰性 import 出真正的类
 ▼  LlamaConfig（类对象，它的 __name__ 是 "LlamaConfig"）
 ③ _LazyAutoMapping 反查：类名 → model_type → 模型类名
 ▼  MODEL_FOR_CAUSAL_LM_MAPPING_NAMES ──"llama"──→ "LlamaForCausalLM"  （modeling_auto.py:789）
 ④ 惰性 import transformers.models.llama → LlamaForCausalLM  ← 最终实例化这个
```

全程**没有**任何「遍历 500 个目录去猜」的动作，全是查表：两份字符串字典 + 两次 `importlib`。

### 2.2 完整调用链

入口 `_BaseAutoModelClass.from_pretrained`（`auto_factory.py:261`）先设 `kwargs["_from_auto"] = True`（`:264`，「我是从 auto 进来的」标记，一路传给 `model_class.from_pretrained()` 让它知道 config 已被 auto 校验，1A3 展开），`config = kwargs.pop("config", None)` 在 `:262`：

| 步 | 位置 | 做什么 |
|----|------|--------|
| 1 | `auto_factory.py:336` | `AutoConfig.from_pretrained(...)` — 读 `config.json` |
| 2 | `configuration_auto.py:388` | `PreTrainedConfig.get_config_dict(...)` 把 json 读成 dict |
| 3 | `configuration_auto.py:419` | `config_class = CONFIG_MAPPING[config_dict["model_type"]]` ← **查表 ①** |
| 4 | `configuration_auto.py:431` | `config_class.from_dict(config_dict, **unused_kwargs)` — 得到 `LlamaConfig` 实例 |
| 5 | `auto_factory.py:355-357` | 判断 `has_remote_code` / `has_local_code` / `explicit_local_code` |
| 6 | `auto_factory.py:389` | `model_class = _get_model_class(config, cls._model_mapping)` ← **查表 ②** |
| 7 | `auto_factory.py:402` | `model_class.from_pretrained(..., config=config, ...)` — 交棒 |

`model_type` 决定**用哪个 Config 类**，`architectures` 决定**用哪个模型类**（见 2.5）。

### 2.3 查表 ①：`model_type` → Config 类

`CONFIG_MAPPING = _LazyConfigMapping(CONFIG_MAPPING_NAMES)`（`configuration_auto.py:153`）。`_LazyConfigMapping.__getitem__`（`:103`）的顺序：`_extra_content`（远程代码注册的）优先（`:104`）→ 不在 `_mapping` 就 `KeyError`（`:106`）→ 取出类名字符串（`:108`）→ `model_type_to_module_name` 得模块名（`:109`）→ 该模块不在缓存就 `importlib.import_module(f".{module_name}", "transformers.models")`（`:110`）→ `getattr` 拿真类（`:112-113`）→ 拿不到就去顶层 `transformers` 模块找（`:118`）。**`CONFIG_MAPPING` 是 `model_type` → 类，`CONFIG_MAPPING_NAMES` 是 `model_type` → 类名字符串；前者 = 后者 + 惰性 import。**

### 2.4 查表 ②：Config 类 → 模型类

`_get_model_class`（`auto_factory.py:178`）只有 14 行但信息量最大：

```python
def _get_model_class(config, model_mapping):                        # auto_factory.py:178
    supported_models = model_mapping[type(config)]                  # :179  ← 传的是「类」不是实例
    if not isinstance(supported_models, (list, tuple)):             # :180
        return supported_models                                     # :181  唯一候选，直接返回
    name_to_model = {model.__name__: model for model in supported_models}
    architectures = getattr(config, "architectures", [])            # :184  ← config.json 里的那个字段
    for arch in architectures:                                      # :185
        if arch in name_to_model:
            return name_to_model[arch]                              # :187  命中！
    return supported_models[0]                                      # :191  兜底：取第一个
```

`model_mapping[type(config)]` 走 `_LazyAutoMapping.__getitem__`（`auto_factory.py:596`）：`_extra_content` 优先（`:597`）→ `model_type = self._reverse_config_mapping[key.__name__]`，用 **config 类的 `__name__` 字符串**（`"LlamaConfig"`）反查（`:599`）→ 查 `_model_mapping` 得模型类名（`:601`）→ `_load_attr_from_module`（`:602`）；反查不到就扫 `_config_mapping`，处理「多个 `model_type` 共用同一个 Config 类」（`:605-608`），仍无则 `KeyError`（`:610`）。因为 `:599` 用的是**字符串**而不是类对象，它只认「库自己定义的那个 Config 类」——`class MyLlamaConfig(LlamaConfig)` 的 `__name__` 是 `"MyLlamaConfig"`，反查不到、一路掉到 `KeyError`；`CONFIG_MAPPING_NAMES` 的 value 不能重复（`utils/check_auto.py:164-173` 专门检查），否则反查表被覆盖。

### 2.5 为什么 `_get_model_class` 要判「list / tuple」

因为映射表的**值可以是 tuple**——一个 `model_type` 对应多个候选类，全库只有 **4 处**（`ast` 扫 `modeling_auto.py`）：`("funnel", ("FunnelModel", "FunnelBaseModel"))`（`:183`）、`("deit", ...)`（`:966-969`）、`("levit", ...)`（`:980-983`）、perceiver 三元 tuple（`:989-995`）。`getattribute_from_module:524` 命中 tuple 分支返回的是 tuple，`:180` 判 True 不直接返回；随后 `architectures`（`:184`）命中就返回对应类（`:187`），**没有 `architectures` 或名字不匹配就落到 `:191` 取第一个**——`("funnel", (...))` 的语义即「默认第一个，`architectures` 说了算」。dict 分支（`getattribute_from_module:526-527`）服务双后端写法 `("aria", {"pil": "AriaImageProcessorPil", "torchvision": "AriaImageProcessor"})`（`auto_mappings.py:1200`）——这也是 `_LazyAutoMappingValue = tuple[type[Any] | None, type[Any] | None]`（`auto_factory.py:49`）的来历：**返回值可能是单个类、tuple、也可能 dict**。

### 2.6 惰性导入：为什么 `import transformers` 不会 import 500 个模型

`_load_attr_from_module`（`auto_factory.py:612`）把模块名交给 `model_type_to_module_name`（`:613`），**每个模块只 `importlib.import_module` 一次**（`:614-615`），再 `getattribute_from_module(self._modules[module_name], attr)`（`:616`；定义在 `:521`：`attr is None` 直接返回 `None`，`:522-523`；tuple / dict 分支递归 `:524-527`；`hasattr` 命中就 `getattr :529-530`；否则去顶层 `transformers` 找 `:533`）。三层惰性分工：`transformers.AutoModel` 靠 `_LazyModule.__getattr__`（`utils/import_utils.py:2500`）；`transformers.models.llama.LlamaForCausalLM` 靠 `_LazyModule` + `define_import_structure`（`llama/__init__.py:28`）；`_LazyAutoMapping[...]` 靠 `_load_attr_from_module`（`auto_factory.py:612`）。

> **纠正一个常见的想当然**：翻 `auto_factory.py` 你会**找不到** `_LazyAutoMapping.__getattr__`——这个类里根本没有 `__getattr__`（`grep -n "__getattr__" auto_factory.py` 只有 `:683` 的一句注释）。惰性全部发生在 `_load_attr_from_module` 的显式 `importlib.import_module` 上；真正用 `__getattr__` 的是**模块级**两处：`utils/import_utils.py:2500` 的 `_LazyModule.__getattr__`，和 `transformers/__init__.py:835` 给历史路径装的 `module.__getattr__ = lambda name: getattr(_get_target(), name)`。收益是实打实的：`import transformers` 不会 import 495 个 `modeling_*.py`。

### 2.7 最后一步：交棒

`:389` 拿到模型类后，`:402` 一行 `return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **hub_kwargs, **kwargs)` 把活交出去。**auto 层的职责到此结束**——它只负责「挑类」；下载权重、建模块、灌 state_dict 全是 `PreTrainedModel.from_pretrained` 的活（见 1A3）。两个分支都没进（`has_remote_code` 假、`has_local_code` 假）就是你见过的报错（`:405-407`），字面内容为 `Unrecognized configuration class <class ...> for this kind of AutoModel: AutoModelForCausalLM. Model type should be one of <...>.`——**字面含义就是「`type(config)` 不在 `cls._model_mapping` 里」**，不是模型不存在，是这个模型没有这个任务头。

---

## 三、`modeling_auto.py` 里的清单

### 3.1 文件结构：一半是表，一半是类

| 区域 | 行范围 | 内容 | 数量 |
|------|--------|------|------|
| 表区 | `:41` – `:2056` | `MODEL_*_MAPPING_NAMES = OrderedDict([...])` | 53 个 |
| 构建区 | `:2058` – `:2179` | `MODEL_*_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_*_MAPPING_NAMES)` | 54 个 |
| 类区 | `:2181` – `:2555` | `class AutoModelForXxx(_BaseAutoModelClass)` | 48 个 |
| 导出 | `:2558` – 末尾 | `__all__ = [...]` | 48 个 `AutoModel*` |

「53 NAMES」对「54 MAPPING」差一个：`MODEL_FOR_AUDIO_TOKENIZATION_MAPPING` 对应的表叫 `MODEL_FOR_AUDIO_TOKENIZATION_NAMES`（`modeling_auto.py:2021`），**名字里没有 `_MAPPING` 后缀**。`check_repo.py` 靠后缀字符串找表（`:1046`、`:1088`：`[x for x in dir(module) if x.endswith("_MAPPING_NAMES")]`），`sort_auto_mappings` 的正则（`:53`）同理——所以这张表**不会被** `check_all_auto_object_names_being_defined` / `check_all_auto_mapping_names_in_config_mapping_names` 检查到。给新模型起名时**别学这个特例**。

### 3.2 挑四个常见的看

| Auto 类 | 行 | 对应 `_model_mapping` | 表大小 | 任务 |
|---------|-----|----------------------|--------|------|
| `AutoModel` | `:2201` | `MODEL_MAPPING` | 543（`MODEL_MAPPING_NAMES` `:41`） | **基座**：输出最后一层 hidden states |
| `AutoModelForCausalLM` | `:2215` | `MODEL_FOR_CAUSAL_LM_MAPPING` | 176（`:692`） | **自回归 LM** |
| `AutoModelForSequenceClassification` | `:2250` | `MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING` | 124（`:1378`） | **整句分类** |
| `AutoModelForImageTextToText` | `:2425` | `MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING` | 85（`:1080`） | **图文多模态** |
| `AutoModelForMaskedLM` | `:2232` | `MODEL_FOR_MASKED_LM_MAPPING` | 47（`:1191`） | **完形填空**（BERT 系，1M5 用） |

每个类的定义体只有一行（如 `AutoModelForCausalLM` 的 `_model_mapping = MODEL_FOR_CAUSAL_LM_MAPPING`，`modeling_auto.py:2215-2216`；`:2219-2226` 只重写了 `from_pretrained` 的返回类型标注），随后 `auto_class_update`（`auto_factory.py:480`）把一个通用模板（`CLASS_DOCSTRING` / `FROM_CONFIG_DOCSTRING` / `FROM_PRETRAINED_TORCH_DOCSTRING`）复制一份、替换类名与 `head_doc`。**这也是 `_LazyAutoMapping.__init__` 里要写 `self._model_mapping._model_mapping = self`（`auto_factory.py:588`）的原因**——`replace_list_option_in_docstrings` 靠它把「List options」换成真实模型清单。

### 3.3 一个反直觉的例子：`gemma3` ≠ `Gemma3ForCausalLM`

`("gemma3", "Gemma3ForConditionalGeneration")` / `("gemma3_text", "Gemma3ForCausalLM")`（`modeling_auto.py:745-746`）——**`gemma3` 指向的是多模态包装类**，`Gemma3ForCausalLM` 留给了 `gemma3_text`。这是「一个上游 checkpoint、两个子配置」的常见形态。取子配置的代码在 `from_config` / `from_pretrained`（`auto_factory.py:236-245` 和 `:390-401`，逻辑一致）：若 `config.sub_configs.get("text_config", None)`（`:236`）非空、且恰等于选中模型类的 `config_class`，就把 `config = config.get_text_config()`（`:242`）换成子配置。**判据是「这个任务头只吃文本，就把文本子配置交给它」**；1A2 展开 `sub_configs` / `get_text_config()`。

---

## 四、`auto_mappings.py` 的角色：半自动生成物

### 4.1 只有 6 个顶层常量（全是 `OrderedDict`，**没有任何 `MODEL_*_MAPPING_NAMES`**，那 53 张全在 `modeling_auto.py`）

`CONFIG_MAPPING_NAMES`（`:24`，**704** 条，`model_type` → Config 类名）、`SPECIAL_MODEL_TYPE_TO_MODULE_NAME`（`:733`，**248** 条，非自然命名的 `model_type` → 模块名）、`VIDEO_PROCESSOR_MAPPING_NAMES`（`:986`，30 条）、`FEATURE_EXTRACTOR_MAPPING_NAMES`（`:1021`，27 条）、`PROCESSOR_MAPPING_NAMES`（`:1053`，140 条）、`IMAGE_PROCESSOR_MAPPING_NAMES`（`:1198`，118 条，值是 `{"pil": ..., "torchvision": ...}`）。

### 4.2 它是生成出来的

第 1-5 行是一段大写警告（由 `utils/check_auto.py:42` 的 `AUTO_GENERATED_HEADER` 拼进文件头）：「This file was automatically generated from existing config files and their `model_type`s. Do NOT edit this file manually … please add the correct `cls.model_type` in your config class and run `python utils/check_auto.py --fix_and_overwrite`.」

生成逻辑在 `utils/check_auto.py:69` 的 `build_config_mapping_names()`：用 `ast` 解析**所有** `configuration_*.py`，找出每个继承 `PreTrainedConfig` 的类、读出它的 `model_type`，然后 `model_type == 目录名`（自然匹配）→ 记进 `CONFIG_MAPPING_NAMES`；否则 → 同时记进 `SPECIAL_MODEL_TYPE_TO_MODULE_NAME`（判定在 `:108-119`，自然匹配优先且不被非自然匹配覆盖）。所以 **`model_type` 是唯一的真相来源，映射表是它的投影。**

### 4.3 分工表

| | `auto_mappings.py` | `auto_factory.py` | `modeling_auto.py` |
|---|---|---|---|
| 是什么 | **数据**（6 张名字表） | **引擎**（查表 + 惰性 import + 挑类） | **数据 + 门面**（53 张模型表 + 48 个 `AutoModelForXxx`） |
| 怎么来的 | `utils/check_auto.py` 从 config 文件生成 | 手写 | **手写** |
| 加新模型时 | 不用管（自动生成） | 不用管 | **要手动加一行** |

谁 import 它：`auto_mappings.py` ← `configuration_auto.py:26`、`video_processing_auto.py:37`、`processing_auto.py:31`、`feature_extraction_auto.py:26`、`image_processing_auto.py:39`；`auto_factory.py` ← `configuration_auto.py:26`、`modeling_auto.py:21`；`modeling_auto.py` ← 自己 `:27`（从 `configuration_auto` 转发）。

**这一格差异是本章最实用的结论**：`CONFIG_MAPPING_NAMES` 会自动长出来，但 `MODEL_FOR_*_MAPPING_NAMES` **不会**——新模型必须手写它支持哪些任务头。

### 4.4 `check_auto.py` 的几个硬约束

`check_auto.py:157-162`：`CONFIG_MAPPING_NAMES` 的 key 重复 → 直接 raise；`:164-173`：value 有重复且不在 `IGNORE_DUPLICATE_CONFIG`（`:64`，只有 `GPT2Config`、`EvollaConfig`、`MLCDVisionConfig`）→ raise；`:269`（`check_duplicates`）：`MISSING_*` 里写的 key 和自动生成的重叠 → raise。第二条的原因（`:155-156` 注释）：config 映射必须一对一，因为 `LazyMapping` 反转 keys/values——一旦两个 `model_type` 指同一个 Config 类名，反查表就会被覆盖掉一个。

---

## 五、`None` 值到底表示什么

网上很多文章说「mapping 里的 `None` 表示该模型不支持这个任务头」。**在本仓库版本里，这句话对模型映射不成立**。

### 5.1 模型映射（`modeling_auto.py`）：没有 `None`

```bash
$ grep -c "None" src/transformers/models/auto/modeling_auto.py
0
```

53 张 `MODEL_*_MAPPING_NAMES` 表里**一个 `None` 都没有**。「不支持这个任务头」的表达方式是——**它干脆不出现在那张表里**，后果就是 2.7 里那句 `Unrecognized configuration class ...`。正确做法是查表而不是找 `None`：`"glm_moe_dsa" in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES` → `True`（`modeling_auto.py:760`）。

### 5.2 tokenizer / processor 映射：`None` = **可选依赖没装**

`tokenization_auto.py:67` 起 `("EvollaModel", "TokenizersBackend" if is_tokenizers_available() else None),`（同形 400+ 条）；`:421` 补漏（`:420` 的 for 循环里）；`video_processing_auto.py:76-80` 注释原文 `# If the torchvision is not available, we set it to None` + `fast_video_processor_class = None`；`processing_auto.py:48` / `image_processing_auto.py:67` 的 `OrderedDict[str, str | None]` **只是类型标注**（「允许是 `None`」），实际内容都是字符串 / dict。

**而且这个 `None` 被下游显式处理**——`getattribute_from_module` 开头两行（`auto_factory.py:522-523`）`if attr is None: return None`，所以查到 `None` 名字时**静默返回 `None`**，不报错；调用方得自己判空，例如 `tokenization_auto.py:816` 的 `and TOKENIZER_MAPPING_NAMES.get(config_model_type) is not None`。

### 5.3 一句话对照

| | 模型映射 | tokenizer/processor 映射 |
|---|---|---|
| `None` 出现 | **从不** | 可选依赖缺失时 |
| 「不支持」怎么表达 | **不出现该 key** | — |
| 不支持时的报错 | `ValueError: Unrecognized configuration class ...` | 静默返回 `None`，由调用方判空 |
| 类型标注 | `_LazyAutoMappingValue = tuple[type \| None, type \| None]`（`auto_factory.py:49`） | `str \| None` |

> `auto_factory.py:49` 那个 `| None` 是**查表返回值的类型标注**，不是「表里有 `None`」；`getattribute_from_module:524-525` 对 tuple 的处理服务于 2.5 那类多候选写法。

---

## 六、实操：把自己的模型接进 auto

### 6.1 最小清单（模型 `MyModel`，`model_type = "mymodel"`，目录 `src/transformers/models/mymodel/`）

| # | 要做的事 | 文件 | 为什么 |
|---|---------|------|--------|
| 1 | 目录名 = `model_type`（`-` 换 `_`） | `models/mymodel/` | `model_type_to_module_name` 靠这个找目录（`configuration_auto.py:64`） |
| 2 | 写 `class MyModelConfig(PreTrainedConfig)`，类体里 `model_type = "mymodel"` | `configuration_mymodel.py` | `check_auto.py:84-104` 从这里读 `model_type` |
| 3 | 目录名不自然就登记 `SPECIAL_MODEL_TYPE_TO_MODULE_NAME` | `configuration_auto.py:50` 的 `.update()` | 生成脚本管不到的情况 |
| 4 | 写 `MyModelForCausalLM` 等 | `modeling_mymodel.py` | 模型本体 |
| 5 | **在 `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES` 里加一行 `("mymodel", "MyModelForCausalLM"),`** | `modeling_auto.py:692` 起那张表 | **必须手动**，脚本不会帮你 |
| 6 | 有 tokenizer 就加 `TOKENIZER_MAPPING_NAMES` 一条 | `tokenization_auto.py:65` 起 | 手动 |
| 7 | 有 Processor 就加 `PROCESSOR_MAPPING_NAMES`（或 `MISSING_PROCESSOR_MAPPING_NAMES`） | `processing_auto.py:50` | 手写 |
| 8 | `__init__.py` | `models/mymodel/__init__.py` | 照抄 `llama/__init__.py`，**不用列类名**（`define_import_structure` 会扫） |

第 5 步加完之后**不用管 `auto_mappings.py`**——第 2 步一落地，`CONFIG_MAPPING_NAMES` 会在下次生成时自动多出 `("mymodel", "MyModelConfig")`。

### 6.2 加了表就要排序

`MODEL_*_MAPPING_NAMES` 和 `CONFIG_MAPPING_NAMES` 都要求**按键名 ASCII 升序**。checker 名是 **`sort_auto_mappings`**（不是 `check_auto`），它被写进 `STYLE_CHECKERS := ruff_check, ruff_format, init_isort, sort_auto_mappings`（`Makefile:13`）。`CHECKER_CONFIG`（`sort_auto_mappings.py:37`）：`name = "sort_auto_mappings"`（`:38`）、`cache_globs = ["src/transformers/models/auto/*.py"]`（`:40`，**整个 auto 目录都归它管**）、`fix_args = []`（`:42`，**不可自动修，只能 `make style`**）。它匹配的正则是 `_re_intro_mapping`（`:53`）：`r"[A-Z_]+(_MAPPING|_MODEL_TYPE_TO_MODULE)(\s+|_[A-Z_]+\s+)=\s+OrderedDict(?!\(\*\))"`。负向断言 `(?!\(\*\))` 让用 `OrderedDict(*list(...))` 展开别表的 `MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES`（`modeling_auto.py:1172`，`:1174` 是 `*list(MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.items())`）被跳过——这就是它不会被排序、也不该被排序的原因。

### 6.3 `make check-repo` 里的两个 auto checker

从 `Makefile:17` 起，`REPO_CONSISTENCY_CHECKERS` 的第一项就是 `auto_mappings`。

| checker 名 | 实现文件 | 检查什么 | 报错时怎么修 |
|-----------|---------|---------|-------------|
| `auto_mappings` | `utils/check_auto.py`（`CHECKER_CONFIG` 在 `:34`） | `auto_mappings.py` 能否从 config 文件重新生成出一份**逐字节相同**的内容 | `python utils/check_auto.py --fix_and_overwrite` 或 `make fix-repo` |
| `sort_auto_mappings` | `utils/sort_auto_mappings.py`（`:37`） | `models/auto/*.py` 里所有 `*_MAPPING_NAMES` 是否按键升序 | `make style`（`fix_args` 为空，checker 自己不会修） |

`check_auto.py` 的判定方式是**现场重新生成一遍再 diff**（`main()`，`:278`；diff 段落在 `:306-324`，不一致且未 `--overwrite` 就 `raise Exception(... Run make fix-repo ...)`，否则写回 `AUTO_FILENAME`）；生成前后都会跑 `run_ruff_and_sort()`（`:234`）——先 `sort_auto_mapping(overwrite=True)` 再 ruff。**这就是它和 `sort_auto_mappings` checker 的接口**：排序函数由 `sort_auto_mappings.py` 提供、由 `check_auto.py:25` 导入调用。

### 6.4 还有第三个 checker 在管 auto：`repo`

| 函数 | 行 | 检查什么 |
|------|-----|---------|
| `check_all_auto_object_names_being_defined()` | `check_repo.py:1031` | 映射表里的**类名**必须真的存在于库里（`hasattr(transformers, class_name)`），`None` 直接 `continue`（`:1056-1057`） |
| `check_all_auto_mapping_names_in_config_mapping_names()` | `:1073` | 映射表里的 **key** 必须在 `CONFIG_MAPPING_NAMES` 里 |
| `check_all_auto_mappings_importable()` | `:1104` | 每张 `_MAPPING_NAMES` 都要能 import |
| `check_models_are_auto_configured()` | `:991` | **每个模型必须至少出现在一张 auto 表里**，否则要求写进白名单 `IGNORE_NON_AUTO_CONFIGURED`（`check_repo.py:348`） |

模型清单来自 `get_all_auto_configured_models()`（`:970`），遍历 `MODEL_*MAPPING_NAMES` 用 `get_values()`（`auto_factory.py:510`）摊平。

> 「忘了在 `modeling_auto.py` 里加一行」：6.3 的 `auto_mappings` 抓不到（它只管 config 映射），但 `repo` checker 的 `check_models_are_auto_configured` 会抓到你。

### 6.5 三个命令的边界

| 命令 | 定义 | 行为 |
|------|------|------|
| `make check-repo` | `Makefile:58` → `utils/checkers.py $(ALL_CHECKERS) --keep-going` | 全跑，**只报不改** |
| `make fix-repo` | `Makefile:62` → `... --fix --keep-going` | 全跑并自动修（会重写 `auto_mappings.py`） |
| `make style` | `Makefile:42` → `utils/checkers.py $(STYLE_CHECKERS) --fix` | 含 `sort_auto_mappings`，**唯一能修排序的入口** |

`utils/checkers.py` 用 `ast` 扫 `utils/*.py` 的 `CHECKER_CONFIG` 发现 checker（`_discover_checkers()`，`checkers.py:108`）；三个 checker 名：`auto_mappings`（`check_auto.py:35`）、`sort_auto_mappings`（`sort_auto_mappings.py:38`）、`repo`（`check_repo.py:55`）。

---

## 一句话总结

> **`models/<name>/` 的文件名 = 职责前缀 + 模块名**，模块名由 `model_type` 把 `-` 换成 `_` 得来，不自然的对应记在 `SPECIAL_MODEL_TYPE_TO_MODULE_NAME`（248 条）；`tokenization` / `processing` / `modular` 都可选。
> **auto 机制 = 两次查表 + 两次惰性 import**：`model_type` → `CONFIG_MAPPING_NAMES` → Config 类 → `MODEL_FOR_*_MAPPING_NAMES` → 模型类，惰性来自 `_load_attr_from_module`（`auto_factory.py:612`）的显式 `importlib.import_module`——**没有 `__getattr__` 魔法**。
> **`auto_mappings.py` 是生成物，`modeling_auto.py` 是手写物**；模型映射里**没有 `None`**——「不支持这个头」的表达方式是「不在表里」，结果是 `ValueError: Unrecognized configuration class ...`。

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
