# 配置系统精读：`config.json` ↔ Python 对象

> 阅读目标：搞清楚一个 `config.json` 是怎么变成 `Cohere2Config` 实例的，
> 以及实例上的每个字段**分别控制模型里的哪一维**。
> 读完这篇，你看到任何 `config.json` 都能在脑子里画出模型的形状，
> 也能回答「这个字段是从哪来的、为什么会被写回 JSON」。
>
> 源文件：`src/transformers/configuration_utils.py`（**1500 行**，本仓库版本 `5.16.0.dev0`）
> 辅助源文件：`src/transformers/models/llama/configuration_llama.py`、
> `src/transformers/models/cohere2/configuration_cohere2.py`、
> `src/transformers/models/qwen2_5_vl/configuration_qwen2_5_vl.py`、
> `src/transformers/modeling_rope_utils.py`、`utils/rules.toml`
> 人类向文档：`docs/source/en/modular_transformers.md`、`docs/source/en/modeling_rules.md`
>
> 下文所有 `文件:行号` 均为本仓库实测（命令见文末「验证清单」）。

---

## 速读区（TL;DR）

**一句话**：`config.json` 里的每个键 = `PreTrainedConfig` dataclass 上的一个**带默认值的字段声明**，
加载是「文件 → dict → `cls(**dict)` → `__post_init__`」，保存是「`self.__dict__` → diff 掉基类字段 → `validate()`」。

| # | 断言 | 出处（文件:行号） |
|---|------|------------------|
| 1 | `PreTrainedConfig` 是 **dataclass**（`@dataclass_transform(kw_only_default=True)` + `@strict(accept_kwargs=True)`），字段写法从 `__init__` 参数变成**类体注解 + 默认值** | `configuration_utils.py:145-148`、`:146` |
| 2 | 加载链：`from_pretrained(:619)` → `get_config_dict(:730)` → `_get_config_dict(:763)` → `from_dict(:863)` → `cls(**dict)(:904)` → `__post_init__(:278)` | 同文件 |
| 3 | `cls(**config_dict)` 调的 `__init__` **不是** dataclass 生成的那个，而是抄自 `huggingface_hub` 的 `wrap_init_to_accept_kwargs`：「传了用传的 / 没传用类默认值 / 多的一律交给 `__post_init__`」 | `configuration_utils.py:104-141`、`:372` |
| 4 | `model_type` 是 `ClassVar`（不在实例 `__dict__` 里），`to_dict` 里**手动补上**——这是「`ClassVar` 不落盘」的唯一例外 | `:262`、`:1083-1088` |
| 5 | `to_diff_dict` 的默认基线是**基类实例** `PreTrainedConfig()` 的 dict，不是同类 → **模型自己的字段一律照写**（哪怕等于类默认值） | `:1020-1026`、`:1032-1055` |
| 6 | `@strict` 买到的是**字段类型校验（含事后赋值）+ `validate_*` 框架**，**不是**「拒绝未知字段」 | `:146`、`:494`、`:511` |
| 7 | `validate_*` 方法**构造时不跑**，只在 `save_pretrained`（`:604-606`）和显式 `config.validate()` 时跑 | `:604-606` |
| 8 | 派生字段在 `__post_init__` 里 `self.head_dim = ...` 会进 `__dict__` → **落盘**（`config.json` 里的 `head_dim` 就是这么来的） | `configuration_cohere2.py:88-104`、`:707-719` |
| 9 | `super().__post_init__(**kwargs)` 必须放**最后** | `configuration_cohere2.py:104`、`configuration_llama.py:92` |
| 10 | `model_type` 不匹配**只打 warning、不报错**（跨架构子集加载是官方支持的用法） | `:710-726` |
| 11 | 生成参数在 `__post_init__` 里被 **pop 掉**（`:323-324`），`save_pretrained` 再拦一次（`:576-582`）——这就是 `use_cache` 之类「怎么设都不生效」的原因 | 同文件 |
| 12 | 复合 config 自己**不含** `hidden_size`（实测 `Qwen2_5_VLConfig`），拿文本维度必须 `get_text_config()` 下钻；`get_text_config(encoder=True)` 在没有 `text_encoder` 时返回**它自己** | `:1304-1379`、`:1306-1316` |
| 13 | `tie_word_embeddings=True` 时会给 `tp_plan` **反向补一条** `embed_tokens: embedding_rowwise`（Llama 默认 False 就没有） | `:353-357` |

**最容易踩的坑 / 反直觉**

- **未知 kwargs 不会被拒绝**：实测 `Cohere2Config(hidden_size=512, totally_unknown_field=123)` 不但被接受，还会经 `__post_init__:339-346` 变成属性、**被 `to_dict` 写进 `config.json`**。仓库文档 `docs/source/en/modular_transformers.md:73` 那句「`@strict` rejects unknown kwargs」对 config 类**当前不成立**；全库 1047 处 `@strict` **没有一处**用 `accept_kwargs=True`（只有基类 `:146` 用）。**别乱传键。**
- 规则与实际不符：`utils/rules.toml:203-213` 的 TRF010 要求「必须 `@strict(accept_kwargs=True)`」，但被它点名的 `nemotron_h` / `vibevoice_asr` 实测写的是**普通 `@strict`**（`configuration_nemotron_h.py:26`、`configuration_vibevoice_asr.py:30`）。**以代码为准。**
- 类型校验发生在 `__setattr__`，所以 `config.hidden_size = "2048"` 这种**事后改坏**也会当场抛 `StrictDataclassFieldValidationError`（不只是构造时）。
- 「模型字段等于默认值就会从 `config.json` 里消失」是**错的**——被 diff 掉的只有**基类**字段（`output_hidden_states` / `return_dict` / `architectures`），实测全默认和改一个字段的 key 集合完全相同。
- 校验也得为存量数据让路：`validate_token_ids` 碰到非法 token id 只 `logger.warning_once` 不抛异常，因为 Hub 上真有 `pad_token_id=-1` 的 config（`:519-520`）。
- 老字段会自动搬运：`torch_dtype` → `dtype`（`:281-289`）；`attribute_map` 让 `bloom` 的 `n_layer` 与 `num_hidden_layers` **双向等价**（`:476-484`）。
- 并行计划**不会**污染 `config.json`：`_remove_keys_not_serialized`（`:1227-1247`）显式剔除 `base_model_tp_plan` / `distributed_config` 等。

**读完应该能回答**

- `config.json` 里**没写**的字段，实例上的值从哪来？多出来的键又去了哪？
- 为什么很多 `config.json` 里有 `head_dim`、`num_key_value_heads` 这种「显然能算出来」的字段？
- 只给 `Qwen2_5_VLTextConfig.from_pretrained(一个 qwen2_5_vl 目录)`，会读到什么？`base_config_key` 起什么作用？
- `config.hidden_size = "2048"` 什么时候报错、报什么错？`validate_layer_type` 又在什么时候才跑？

---

## 前置知识

- [1A1_repo_and_auto.md](./1A1_repo_and_auto.md)：`AutoModel` 如何用 `config.model_type` 找到模型类（本篇第五章接着讲这个键本身）
- [1M1_llama_structure.md](./1M1_llama_structure.md)：Llama 的 `LlamaAttention` / `LlamaMLP` / `LlamaModel` —— 第二章的「字段 → 维度」对照全靠它
- [2A5_modular_code_generation.md](./2A5_modular_code_generation.md)：`configuration_*.py` 现在也由 `modular_*.py` 生成，所以看到 `configuration_cohere2.py` 顶部的警告不要惊讶（`configuration_cohere2.py:1-6` 就是那段警告）

---

## 一、`PreTrainedConfig` 是什么：一个双向映射

### 一句话

`PreTrainedConfig` 是 `config.json` ↔ Python 对象的**唯一定义**：
JSON 里的每个键对应类体上的一个**带默认值的字段声明**，
`from_pretrained` 把 JSON 灌进对象，`save_pretrained` 把对象摊回 JSON。

```
config.json  ──from_pretrained──▶  Cohere2Config 实例（字段即属性）
     ▲                                    │
     └──────── save_pretrained ───────────┘
```

### 关键认知：它是「结构表的唯一真相源」

模型代码本身不写死任何维度。看 `LlamaMLP` 的三行（`modeling_llama.py:169-171`）：

```python
self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
```

`hidden_size` / `intermediate_size` 全部来自 `config`。
所以**换一个 `config.json` 就是换一个模型形状**——这正是 `AutoModel.from_config(config)`
能在不下载权重的前提下凭空造出一个模型的原因（1A3 会展开）。

### 一个必须先知道的版本事实：config 现在是 `dataclass`

在 v5 里 `PreTrainedConfig` 不再是「手写 `__init__` 的普通类」，而是一个 **dataclass**
（`configuration_utils.py:145-148`）：

```python
@dataclass_transform(kw_only_default=True)
@strict(accept_kwargs=True)
@dataclass(repr=False)
class PreTrainedConfig(PushToHubMixin, RotaryEmbeddingConfigMixin, HeterogeneousConfigMixin):
```

于是「字段」的写法从 `__init__` 参数变成了**类体上的注解 + 默认值**。
对照 Llama（`configuration_llama.py:64-84`，节选）：

```python
vocab_size: int = 32000
hidden_size: int = 4096
intermediate_size: int = 11008
num_hidden_layers: int = 32
num_attention_heads: int = 32
num_key_value_heads: int | None = None
head_dim: int | None = None
```

**读源码时注意三处**：

| 位置 | 含义 |
|------|------|
| 类体注解（`hidden_size: int = 4096`） | 字段声明 + 默认值（dataclass field） |
| `ClassVar[...]` 注解（`configuration_utils.py:241-258`） | **不是字段**，不进 `__dict__`、不落盘，纯类级元数据 |
| `__post_init__` 里的赋值 | 派生字段（`head_dim` 等），见第八章 |

---

## 二、字段如何决定结构

这张表是本篇的核心：**每个字段落到 model 里的哪一行、控制哪一维**。
所有行号出自 `modeling_llama.py`（1M1 的模型）。

| config 字段 | 消费点（`modeling_llama.py:行号`） | 控制什么 |
|-------------|-----------------------------------|----------|
| `vocab_size` | `:353` `nn.Embedding(config.vocab_size, hidden_size)`；`:431` `lm_head = nn.Linear(hidden_size, vocab_size)` | 词表大小 = embedding 行数 = logits 最后一维 |
| `hidden_size` | `:292-293` RMSNorm 宽度；`:353` embedding 列数；`:167` MLP 输入维；`:224` head_dim 的兜底来源 | 残差流宽度（模型「宽度」） |
| `intermediate_size` | `:168-171` MLP 的 gate/up 升维目标、down 的降维来源 | FFN 中间层宽度（通常 ≈ 8/3 × hidden_size） |
| `num_hidden_layers` | `:355` `[LlamaDecoderLayer(config, i) for i in range(config.num_hidden_layers)]`；`:402` 遍历上界 | 层数（模型「深度」） |
| `num_attention_heads` | `:231`/`:240` q_proj/o_proj 的列数/行数；`:225` `num_key_value_groups` | 注意力头数（Q 侧） |
| `num_key_value_heads` | `:234`/`:237` k_proj/v_proj 输出维；`:225` 与 Q 头数相除得 GQA 复制倍数 | **KV 头数**。小于 `num_attention_heads` 就是 GQA，直接决定 KV cache 体积 |
| `head_dim` | `:104` RoPE 的旋转维度；`:224` `self.head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)` | 单个头的宽度。v5 里它可与 `hidden_size / num_attention_heads` **解耦**（Qwen3 就是这么干的） |
| `max_position_embeddings` | `:77-78` `RotaryEmbedding` 的缓存上界 | 训练时的最大序列长度（RoPE 缓存长度） |
| `rms_norm_eps` | `:292-293`、`:357` `LlamaRMSNorm(hidden_size, eps=config.rms_norm_eps)` | 归一化的数值稳定项 |
| `sliding_window` | `modular_mistral.py:86`、`:137` `mask_function = create_causal_mask if config.sliding_window is None else create_sliding_window_causal_mask` | 注意力窗口大小；`None` = 全注意力（Mistral 与 Llama 的唯一区别） |
| `tie_word_embeddings` | `modeling_utils.py:2559-2560` `tie_word_embeddings = getattr(self.config, "tie_word_embeddings", False)` | 是否让 `lm_head.weight` 与 `embed_tokens.weight` 共享（省一份词表参数） |
| `rope_parameters`（旧名 `rope_theta` / `rope_scaling`） | `modeling_rope_utils.py:173` `config.rope_parameters[layer_type]` | RoPE 的频率基数与 scaling 策略（2A4 展开） |
| `dtype`（旧名 `torch_dtype`） | `configuration_utils.py:284-289` 把字符串转成 `torch.dtype` | 加载权重时的默认精度 |

**三个容易记混的点**：

1. `hidden_size` 是**残差流宽度**，不是「每个头多宽」。头宽是 `head_dim`，两者通过
   `num_attention_heads * head_dim` 联系——而 v5 允许它们不相等（`head_dim` 独立声明）。
2. `num_key_value_heads` 只影响 K/V 投影的输出维，**不影响 Q**。这就是 GQA 省显存的全部秘密。
3. `sliding_window` 是个「开关型」字段：它不是某层的参数，而是决定**用哪个 mask 函数**
   （`modular_mistral.py:137`）。1M2 讲的就是这个。

---

## 三、从 `config.json` 到实例：完整链条

`from_pretrained` 是一条四段式流水线（`configuration_utils.py`）：

```
PreTrainedConfig.from_pretrained          :619
   └─ cls.get_config_dict(...)            :730    文件 → dict
        └─ cls._get_config_dict(...)      :763    下载 / 读 JSON / 记 commit hash
             └─ cls._dict_from_json_file  :946    json.loads
                  └─ cls.from_dict(...)   :863    dict → 实例
                       └─ cls(**config_dict)      :904   ← 真正的构造
                            └─ __init__（其实是 :105 的包装器）
                                 └─ __post_init__  :278
```

### 1) `_get_config_dict`：把文件变成 dict

`configuration_utils.py:789-816` 是下载/读取的位置，`cached_file` 负责「本地目录 / 本地缓存 / Hub 下载」三选一。
真正解析 JSON 只有一行（`:946-951`）：

```python
@classmethod
def _dict_from_json_file(cls, json_file: str | os.PathLike):
    with open(json_file, encoding="utf-8") as reader:
        text = reader.read()
    config_dict = json.loads(text)

    return cls._decode_special_floats(config_dict)
```

`_decode_special_floats`（`:979`）专门处理 `NaN` / `Infinity`——它们在标准 JSON 里不合法，
所以库把它们编码成 `{"__float__": "Infinity"}` 再解回来（`_encode_special_floats`，`:954`）。

最后两行是「错字也能救」的兜底（`:838-853`）：

```python
# timm models are not saved with the model_type in the config file
if "model_type" not in config_dict and is_timm_config_dict(config_dict):
    config_dict["model_type"] = "timm_wrapper"
...
if "model_type" in kwargs and config_dict["model_type"] != kwargs["model_type"]:
    logger.warning(...)          # 用户显式覆盖 model_type 时告警
    config_dict["model_type"] = kwargs["model_type"]
```

### 2) `get_config_dict`：一层「间接跳转」

`:750-760` 处理 Hub 上的一种历史遗留写法——`config.json` 里放的不是配置本身，
而是一个 `configuration_files` 列表（老模型的「配置清单」）：

```python
if "configuration_files" in config_dict:
    configuration_file = get_configuration_file(config_dict["configuration_files"])
    config_dict, kwargs = cls._get_config_dict(
        pretrained_model_name_or_path, _configuration_file=configuration_file, **original_kwargs
    )
```

`get_configuration_file`（`:1425`）按当前 transformers 版本从清单里挑一个**版本号 ≤ 自己的最新文件**。

### 3) `base_config_key`：从复合 config 里「下钻」到子 config

`:707-708` 只有两行，但它是组合式 config 能工作的前提：

```python
if cls.base_config_key and cls.base_config_key in config_dict:
    config_dict = config_dict[cls.base_config_key]
```

即：如果你用 `Qwen2_5_VLTextConfig.from_pretrained(...)` 去读一个 `qwen2_5_vl` 的 config.json，
它会自动钻进 `config_dict["text_config"]`。实测（第七节给出完整实验）：

```
TextConfig.from_pretrained -> hidden_size: 1234 | base_config_key: text_config
```

### 4) `from_dict`：dict → 实例

`:863-925` 干了三件事：

```python
config = cls(**config_dict)              # :904  ← 唯一的构造点
```

```python
for key, value in kwargs.items():                     # :906
    if hasattr(config, key):
        current_attr = getattr(config, key)
        # To authorize passing a custom subconfig as kwarg in models that have nested configs.
        if isinstance(current_attr, PreTrainedConfig) and isinstance(value, dict):
            current_attr_updated = current_attr.to_dict()
            current_attr_updated.update(value)
            value = current_attr.__class__(**current_attr_updated)   # :914 子 config 合并而非整体替换
        setattr(config, key, value)
```

也就是说：**`config_dict` 里的键走 dataclass 字段，`kwargs` 里的键走 `setattr` 覆盖**。
显式传参（`from_pretrained(..., output_attentions=True)`）因此能压过文件里的值。

### 5) `__init__` 并不是 dataclass 生成的

这是本篇最容易看漏的一环。`from_dict` 里的 `cls(**config_dict)` 调用的 `__init__`，
在大多数 config 上**不是** dataclass 自动生成的那个，而是 `configuration_utils.py:104-141`
这个「拷贝自 huggingface_hub」的包装器：

```python
# copied from huggingface_hub.dataclasses.strict when `accept_kwargs=True`
def wrap_init_to_accept_kwargs(cls: dataclass):
    # Get the original dataclass-generated __init__
    original_init = cls.__init__

    @wraps(original_init)
    def __init__(self, *args, **kwargs: Any) -> None:
        dataclass_fields = {f.name for f in fields(cls)}                      # :112
        standard_kwargs = {k: v for k, v in kwargs.items() if k in dataclass_fields}
        ...
        for f in fields(cls):                                                  # :122
            if f.name in standard_kwargs:
                setattr(self, f.name, standard_kwargs[f.name])                 # ① 传了就用传的
            elif f.default is not MISSING:
                setattr(self, f.name, f.default)                               # ② 没传就用类默认值
            elif f.default_factory is not MISSING:
                setattr(self, f.name, f.default_factory())
            else:
                raise TypeError(f"Missing required field - '{f.name}'")        # :130
        ...
        self.__post_init__(**additional_kwargs)                                # :139
```

**「config.json 里没出现的字段怎么办」的答案就在这里**：
走 ② 分支，取类体上声明的默认值（`f.default`）或工厂（`f.default_factory`）。
`additional_kwargs`（不在字段列表里的键）则原样交给 `__post_init__`。

对比一下 `__init__` 被换成 dataclass 原生版本的后果：dataclass 生成的 `__init__`
会在结尾调 `__post_init__()`，**不带 kwargs**——那些多余键就丢了。
包装器的注释说得很直白（`:125-127`）：

> We need to call bare `__init__` without `__post_init__` but the `original_init` of
> any dataclass contains a call to post-init at the end (without kwargs)

### 6) `__post_init__`：最后一站

`configuration_utils.py:278-357`，所有 config 的 `__post_init__` 最终都汇到这里。
它负责三类收尾（详见第八、九章）：

| 行号 | 做什么 |
|------|--------|
| `:281-289` | BC：`torch_dtype` → `dtype`；字符串 dtype → `torch.dtype` |
| `:323-324` | 把 `GenerationConfig` 的生成参数**丢掉**（在 model config 里设生成参数已废弃） |
| `:327-328` | 收 `name_or_path` / `_commit_hash`（存到 `_name_or_path`，避免污染 `attribute_map`） |
| `:331-333` | 收 `_attn_implementation` / `_experts_implementation`（setter 会递归写进子 config） |
| `:339-346` | **未知键 → `setattr(self, key, value)`** |
| `:353-357` | `tie_word_embeddings=True` 时给 `base_model_tp_plan` 补一条 `embed_tokens: rowwise` |

---

## 四、反方向：保存时怎么变回 JSON

| 方法 | 行号 | 作用 |
|------|------|------|
| `save_pretrained` | `:556` | 建目录 → 写 `config.json`（`CONFIG_NAME`，`:599`/`:606`） |
| `to_json_file` | `:1148` | `json.dumps` + 落盘 |
| `to_json_string` | `:1126` | 默认 `use_diff=True` → 调 `to_diff_dict` |
| `to_diff_dict` | `:1011` | 只写「和默认值不同」的部分 |
| `to_dict` | `:1076` | 完整 `__dict__` + `model_type` + `transformers_version` |

### `to_dict` 的两处「无中生有」

`:1083-1088`：

```python
output = copy.deepcopy(self.__dict__)
if hasattr(self.__class__, "model_type"):
    output["model_type"] = self.__class__.model_type
# Transformers version when serializing the model
output["transformers_version"] = __version__
```

`model_type` 是 `ClassVar`（`:262`），**不在实例 `__dict__` 里**，所以必须手动补上——
这就是「`ClassVar` 不落盘」的唯一例外，也是它成为 JSON 必备键的原因。

### `to_diff_dict` 的「默认值」是**基类**的默认值，不是子类的

`:1020-1026`：

```python
config_dict = self.to_dict()
# Get the default config dict (from a fresh PreTrainedConfig instance)
default_config_dict = PreTrainedConfig().to_dict()      # ← 基类实例
# get class specific config dict
class_config_dict = self.__class__().to_dict() if not self.has_no_defaults_at_init else {}
```

判定条件在 `:1032-1055`，其中起决定作用的是 `key not in default_config_dict`。
**实测结论**（见验证清单第 6 条）：

```
全默认 keys 数: 24 | 改一个字段后: 24 | key 集合相同: True
全默认也写了 attention_dropout: 0.0 | vocab_size: 256000
有没有 output_hidden_states/return_dict: False False
```

即：**模型自己的字段一律照写**（哪怕等于类默认值），被 diff 掉的只有基类字段
（`output_hidden_states`、`return_dict`、`architectures`……）。
所以别指望「模型字段会因等于默认值而消失」。

嵌套子 config 有单独的递归清理（`:1042` `recursive_diff_dict`，定义在 `:1455`），
并强制保留 `model_type`（`:1043-1045`）。

### 保存前会「体检」

`:604-606`：

```python
# Strict validation at save-time: prevent bad patterns from propagating
if hasattr(self, "validate"):
    self.validate()
self.to_json_file(output_config_file, use_diff=True)
```

`validate()` 是 `@strict` 装上的（第六章），它会把 `validate_*` 方法全跑一遍。
另外 `save_pretrained` 一开始就拦住生成参数（`:576-582`），
这是 v5 里 `use_cache` 之类的键「怎么设都不生效」的原因（`:323-324` 已经把它们 pop 掉了）。

### 不落盘的键

`_remove_keys_not_serialized`（`:1227-1247`）显式剔除一批内部键，包括
`_is_quantized`、`_auto_class`、`_commit_hash`、`_attn_implementation_internal`、
`base_model_tp_plan`、`base_model_pp_plan`、`base_model_fsdp_plan`、`distributed_config`。
所以**并行计划不会污染 `config.json`**，它们是「代码里声明、运行时使用」的元数据（第九章）。

---

## 五、`model_type`：auto 映射的键

`model_type` 是 `ClassVar[str]`（`:262`），两个方向各用一次：

| 方向 | 位置 | 代码 |
|------|------|------|
| 写 | `to_dict` `:1084-1085` | `output["model_type"] = self.__class__.model_type` |
| 读 | `AutoConfig.from_pretrained` `configuration_auto.py:419` | `config_class = CONFIG_MAPPING[config_dict["model_type"]]` |

`CONFIG_MAPPING` 的键就是各 config 类的 `model_type`（`configuration_auto.py:293-298`
的 `for_model` 用的是同一张表）。实测：

```
Cohere2Config.model_type = cohere2
LlamaConfig.model_type = llama
Qwen2_5_VLConfig.model_type = qwen2_5_vl
```

### 拿错 config 类时会发生什么：只警告，不报错

`from_pretrained` 里 `:710-726`：

```python
if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
    # sometimes the config has no `base_config_key` if the config is used in several composite models
    for v in config_dict.values():
        if isinstance(v, dict) and v.get("model_type") == cls.model_type:
            config_dict = v          # ← 先去子 config 里找有没有对得上号的
    if config_dict["model_type"] != cls.model_type:
        logger.warning(f"You are using a model of type `{config_dict['model_type']}` to instantiate a model of type `{cls.model_type}`...")
```

实测：用 `LlamaConfig.from_pretrained()` 去读一个 `cohere2` 的目录，**只打警告、能读进来**：

```
[transformers] You are using a model of type `cohere2` to instantiate a model of type `llama`. ...
实际类型: LlamaConfig | hidden_size: 512
```

这就是「`model_type` 是**弱校验**」的证据——因为跨架构子集加载（`sam2_video` → `Sam2Model`）
是官方支持的用法，所以不能做成硬失败。

---

## 六、`@strict`：类型校验，而不是拒绝未知字段

这一章要小心——**文档说的和代码做的不完全一样**，我们逐层拆。

### 1) 它来自 `huggingface_hub`，不在本仓库

`configuration_utils.py:25`：

```python
from huggingface_hub.dataclasses import strict
```

依赖声明在 `setup.py:90`：`"huggingface-hub>=1.5.0,<2.0"`。
本仓库没有它的副本，所以下面引的是 PyPI 上 `huggingface_hub==1.33.0` 的
`huggingface_hub/dataclasses.py`（获取方式见验证清单第 1 条）。

### 2) `@strict` 做三件事

| 机制 | hub 源码位置 | 效果 |
|------|-------------|------|
| 字段类型校验器 | `dataclasses.py:120-134` | 每个 field 按注解生成一个校验器，存进 `cls.__validators__` |
| 劫持 `__setattr__` | `dataclasses.py:137-151` | **每次赋值**都跑校验，失败抛 `StrictDataclassFieldValidationError` |
| 收集 `validate_*` | `dataclasses.py:227-267` | 类体里所有 `validate_xxx(self)` 收成类校验器，装上 `validate()` 方法，失败抛 `StrictDataclassClassValidationError` |

三种错误类定义在 `huggingface_hub/errors.py:540-557`。

实测（用真 hub 包，命令见验证清单第 2 条）：

```
1) 正常: Plain(hidden_size=512, num_hidden_layers=12)
2) 未知 kwarg -> TypeError : Plain.__init__() got an unexpected keyword argument 'typo_field'
3) 类型错 -> StrictDataclassFieldValidationError : Validation error for field 'hidden_size': |     TypeError: Field 'hidden_size' expected int, got str (value: '512')
4) 事后赋值类型错 -> StrictDataclassFieldValidationError : ... expected int, got float (value: 3.5)
```

**第 4 条最关键**：校验发生在 `__setattr__`，所以 `config.hidden_size = "2048"` 这种
「事后改坏」也会当场被拦。这是 `@strict` 相对于「只在 `__init__` 校验」的额外收益。

### 3) transformers 怎么用它：基础类故意「宽容」

两处声明，**参数不一样**：

| 位置 | 声明 |
|------|------|
| `configuration_utils.py:146` | `@strict(accept_kwargs=True)` ← 基础类 |
| `configuration_cohere2.py:29`（`Cohere2Config`）、`configuration_qwen2_5_vl.py:35`/`:68`/`:158`（三个类各一处） | `@strict` ← 具体模型（**全库 1047 处，没有一处用 `accept_kwargs=True`**） |

`accept_kwargs=True` 会把 `__init__` 换成一个「只挑字段、其余丢给 `__post_init__`」的版本
（hub `dataclasses.py:153-192`），**未知 kwargs 不再被拒绝**。而本仓库把这段逻辑**抄了一份**放在
`configuration_utils.py:104-141`（注释写明了来源），并在 `__init_subclass__` 里给每个子类安上：

```python
def __init_subclass__(cls, *args, **kwargs):          # :359
    super().__init_subclass__(*args, **kwargs)
    cls_has_custom_init = "__init__" in cls.__dict__   # :361
    # kw_only=True ensures fields without defaults in subclasses can follow
    # parent fields that have defaults (Python dataclass ordering rule).
    cls = dataclass(cls, repr=False, kw_only=True)     # :365
    if not cls_has_custom_init:
        # Wrap all subclasses to accept arbitrary kwargs for BC
        cls = wrap_init_to_accept_kwargs(cls)          # :372
```

所以**具体模型 config 也是宽容的**。实测：

```
--- 2) 未知 kwargs 会怎样？
accepted; new attr = 123
attr: 123 | to_dict 里有吗: 123
```

（`Cohere2Config(hidden_size=512, totally_unknown_field=123)` 不但被接受，
还会经 `__post_init__:339-346` 变成属性、进而被 `to_dict` 写进 `config.json`。**别乱传键**。）

### 4) 那「拒绝未知 kwargs」这句话错了吗？

没错，但只对**没有 `__init__` 包装器**的类成立。仓库文档 `docs/source/en/modular_transformers.md:73` 写的是：

> `@strict` rejects unknown kwargs at instantiation time, catching typos and stale arguments early.

实测证据表明 transformer 的 config 类当前**不拒绝**（上面两条实验）。
需要留意的还有 `utils/rules.toml:203-213` 的 TRF010 规则：

```toml
[rules.TRF010]
description = "Direct config definitions must use @strict(accept_kwargs=True)."
# Empty: `nemotron_h` and `vibevoice_asr` both carry `@strict(accept_kwargs=True)` now.
```

它要求写 `accept_kwargs=True`，而实测这两个被点名的模型写的都是**普通 `@strict`**：

```
src/transformers/models/nemotron_h/configuration_nemotron_h.py:26:@strict
src/transformers/models/vibevoice_asr/configuration_vibevoice_asr.py:30:@strict
```

**结论（读源码的实操规则）**：以代码为准。当前 `@strict` 在 transformers 里买到的是

1. **字段级类型校验**（含事后赋值）；
2. **`validate_*` 类校验器框架**（下面有真实例子）；
3. **dataclass 字段声明**——`from_dict` / `to_dict` / `to_diff_dict` / `@auto_docstring`
   整套机器都建立在这上面；
4. 未知键的宽容处理（BC 需要，Hub 上大量老 config 带着已废弃字段）。

### 5) transformers 自己的校验器

字段级用 `utils/type_validators.py` 里的工厂（`configuration_llama.py:72`）：

```python
initializer_range: float = interval(min=0.0, max=1.0)(default=0.02)
```

类级用 `validate_` 前缀方法，**不写装饰器、不注册，靠命名约定被 `@strict` 收集**。
真实例子：`configuration_utils.py:526-546` 的 `validate_layer_type`
（校验 `layer_types` 取值合法、且长度等于 `num_hidden_layers`），
`configuration_llama.py:94-100` 的 `validate_architecture`：

```python
def validate_architecture(self):
    """Part of `@strict`-powered validation. Validates the architecture of the config."""
    if self.hidden_size % self.num_attention_heads != 0:
        raise ValueError(...)
```

实测触发：

```
REJECTED: StrictDataclassClassValidationError Class validation error for validator 'validate_layer_type': |     ValueError: The `layer_types` entries must be in ('full_attention', 'sliding_attention', ...
```

注意它们**在构造时不跑**，只在 `save_pretrained`（`:604-606`）和显式 `config.validate()` 时跑。
代码里这批方法的 docstring 统一写着 `"""Part of `@strict`-powered validation. ..."""`（`:494`、`:511`）——
**见到这句 docstring 就知道它被 `validate()` 收集了**。

（两个补充：`validate_layer_type` 的 docstring 就没有那句话（`:526-527`），照样会被收集——
收集靠的是**方法名前缀**而不是 docstring；另外 `validate_token_ids` 碰到非法 token id
只 `logger.warning_once` 而不抛异常，注释写明了原因——Hub 上真有 `pad_token_id=-1` 的 config
（`:519-520`）。**校验也得为存量数据让路**。）

---

## 七、组合式 config：`sub_configs` / `base_config_key` / `get_text_config`

多模态模型（VL）的 config 是**嵌套**的：`Qwen2_5_VLConfig` 里放 `text_config` + `vision_config`。
三个部件分工明确。

### 1) `sub_configs`：声明「哪些字段是子 config」

`configuration_utils.py:246` 的 `ClassVar`，具体模型在类体里填（`configuration_qwen2_5_vl.py:177`）：

```python
model_type = "qwen2_5_vl"
sub_configs = {"vision_config": Qwen2_5_VLVisionConfig, "text_config": Qwen2_5_VLTextConfig}
keys_to_ignore_at_inference = ["past_key_values"]

text_config: dict | PreTrainedConfig | None = None
vision_config: dict | PreTrainedConfig | None = None
```

它有两个消费者：

| 消费者 | 位置 | 用途 |
|--------|------|------|
| 子 config 实例化 | 各模型自己的 `__post_init__` | 把 dict 变成对象 |
| `_attn_implementation` setter | `configuration_utils.py:425-437` | **递归**把注意力实现写进每个子 config |

### 2) 子 config 的实例化：写在模型自己的 `__post_init__` 里

`configuration_qwen2_5_vl.py:188-204`（这是全库的标准套路）：

```python
def __post_init__(self, **kwargs):
    if isinstance(self.vision_config, dict):
        self.vision_config = self.sub_configs["vision_config"](**self.vision_config)
    elif self.vision_config is None:
        self.vision_config = self.sub_configs["vision_config"]()

    # Hub configs are saved as flat dicts so we pop some of kwargs to init `TextConfig`
    text_params = inspect.signature(self.sub_configs["text_config"].__init__).parameters.keys()
    text_params = list(text_params) + ["rope_parameters", "rope_scaling", "rope_theta"]
    text_kwargs = {key: kwargs.pop(key) for key in text_params if key in kwargs}

    if isinstance(self.text_config, dict):
        self.text_config = self.sub_configs["text_config"](**self.text_config)
    elif self.text_config is None:
        text_kwargs["dtype"] = kwargs.get("torch_dtype", kwargs.get("dtype"))
        self.text_config = self.sub_configs["text_config"](**text_kwargs)
    ...
    super().__post_init__(**kwargs)
```

**三个要点**：

1. **`isinstance(..., dict)` 判断**是机制的核心：`__init__` 只会原样赋值，转成对象得自己来。
2. `sub_configs` 在这里是**查表**用的（`self.sub_configs["text_config"]`），不是自动触发的——
   `sub_configs` 本身**不会**替你实例化子 config。
3. 「Hub configs are saved as flat dicts」这句注释很重要：很多 VL 的 `config.json`
   把文本侧的键**平铺在外层**，所以模型要主动从 `kwargs` 里 `pop` 出属于 text 的部分。

### 3) `base_config_key`：反方向的下钻键

每个子 config 类声明自己**住在父 config 的哪个键下**：

```
Qwen2_5_VLTextConfig.base_config_key: text_config
Qwen2_5_VLVisionConfig.base_config_key: vision_config
Cohere2Config.base_config_key: ''          ← 普通模型是空串
```

配合 `from_pretrained:707-708`（第三章 3)）实现「用子 config 类直接读父 checkpoint」。
实测端到端（验证清单第 7 条）：

```
TextConfig.from_pretrained -> hidden_size: 1234 | base_config_key: text_config
AutoConfig -> Qwen2_5_VLConfig | text: Qwen2_5_VLTextConfig 1234 | vision: Qwen2_5_VLVisionConfig 5
get_text_config() is a.text_config: True
外层 a.hidden_size: <无>
```

注意最后一行：**复合 config 自己不含 `hidden_size`**，要拿文本侧维度必须显式下钻。

### 4) `get_text_config()`：统一的「拿文本侧 config」入口

`configuration_utils.py:1304-1379`。它解决「不同模型的文本 config 叫不同名字」的问题。

```python
return_both = decoder == encoder  # both unset or both set -> search all possible names   # :1322

decoder_possible_text_config_names = ("decoder", "generator", "text_config")               # :1324
encoder_possible_text_config_names = ("text_encoder",)                                     # :1325
...
for text_config_name in possible_text_config_names:                                        # :1334
    if hasattr(self, text_config_name):
```

三种结局（docstring `:1306-1316` 自己列了）：

| 情形 | 返回 |
|------|------|
| 普通模型（如 Llama） | `self` 自身 |
| 2024+ 复合模型 | 名字匹配到的子 config（实测 `Qwen2_5_VLConfig` → `Qwen2_5_VLTextConfig`） |
| 2023- 老式 flat encoder-decoder（如 T5） | 一份 deepcopy，把 `decoder_*` 前缀键改名成规范名 |

第三种最绕（`:1351-1377`）：它**不删键**，只做改名，因为
「dataclass 的类属性删不掉、也不能设 `None`（会被 strict 校验拦住）」——注释原文 `:1356-1358`。
改名规则：`decoder_layers`→`num_hidden_layers`、`decoder_attention_heads`→`num_attention_heads`，
其余去掉前缀；改名前还要过一次 `attribute_map`（`:1372-1373`）。

两个坑：

- **歧义会报错**：同时存在多个合法名时抛 `ValueError`（`:1340-1345`）。
- `decoder=True` / `encoder=True` 是**双向**的：`Qwen2_5_VLConfig` 没有 `text_encoder`，
  所以 `get_text_config(encoder=True)` 返回的是**它自己**（实测），不是「vision config」。

`get_text_config` 在库内被大量使用，例如 `validate_token_ids`（`:510-512`）第一行就是
`text_config = self.get_text_config(decoder=True)`，量化器也用它下钻（`quantizers/quantizer_compressed_tensors.py:140-141`）。

> 相关但不同的 `get_mtp_config()`（`:1381`）用于 MTP（多 token 预测）层：
> 它 deepcopy 文本侧 config，然后把 `layer_types` / `mlp_layer_types` 换成 `mtp_*` 版本、
> 把 `num_hidden_layers` 改成 MTP 层数——**同一个 config 类被复用两次**的典型做法。

---

## 八、`__post_init__` 与派生属性

「派生属性」= 用户没传、由 config 自己算出来的字段。两个真实例子。

**例 1：Cohere2（`configuration_cohere2.py:88-104`）**

```python
def __post_init__(self, **kwargs):
    if self.num_key_value_heads is None:
        self.num_key_value_heads = self.num_attention_heads      # :90  GQA 缺省 = MHA

    # Need to specify head_dim in the config so it can be used in the attention forward functions
    self.head_dim = self.hidden_size // self.num_attention_heads # :93  ← 派生 head_dim

    # BC -> the pattern used to be a simple int, and it's still present in configs on the Hub
    if self.layer_types is None:
        _sliding_window_pattern = kwargs.pop("sliding_window_pattern", 4)   # :98  老字段
        self.layer_types = [                                                # :99-102
            "sliding_attention" if bool((i + 1) % _sliding_window_pattern) else "full_attention"
            for i in range(self.num_hidden_layers)
        ]

    super().__post_init__(**kwargs)          # :104  ← 必须最后调
```

**例 2：Llama（`configuration_llama.py:84-92`）**

```python
head_dim: int | None = None                  # :84  ← 声明为「可选」

def __post_init__(self, **kwargs):           # :86
    if self.head_dim is None:
        self.head_dim = self.hidden_size // self.num_attention_heads   # :88
    if self.num_key_value_heads is None:
        self.num_key_value_heads = self.num_attention_heads            # :90

    super().__post_init__(**kwargs)          # :92
```

注意 Llama 的写法更规范：`head_dim` **先声明成字段**（`None`），再在 `__post_init__` 里兜底。
Cohere2 是**直接凭空赋值**——两种都能跑，但只有声明过的那种会参与 `from_pretrained`
的「字段值覆盖」（`from_dict:906-916` 靠 `hasattr(config, key)` 判断，派生属性也在 `__dict__` 里，
所以同样能被覆盖）。

### 派生字段的副作用：会被写进 `config.json`

`to_dict` 用的是 `self.__dict__`（`:1083`），`__post_init__` 里 `self.head_dim = ...`
把值放进了 `__dict__`，于是**保存时会落盘**。实测一次真实保存（验证清单第 5 条）：

```
model_type: cohere2
hidden_size: 512
head_dim: 64                      ← 用户从没传过，却出现在 config.json 里
num_key_value_heads: 8
layer_types: ['sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention']
```

**这解释了一个常见困惑**：为什么很多 `config.json` 里有 `head_dim`、`num_key_value_heads`
这类「显然能算出来」的字段——它们是 `__post_init__` 物化后被序列化的。
`LlamaConfig().to_json_string()` 同样带 `"head_dim": 128`。

### 顺序约定

`super().__post_init__(**kwargs)` 必须放**最后**（Cohere2 `:104`、Llama `:92`、Qwen2.5-VL `:212`）。
原因：父类的 `__post_init__` 会处理剩下的 `kwargs`（`:339-346` 的 `setattr`），
你得先把自己的派生字段算完，再让父类收尾。

---

## 九、结构相关的类级属性

这些是**不落盘**的 `ClassVar`（`:241-258`），声明「模型长什么样」给下游工具用。

| 属性 | 行号 | 声明者 | 谁消费 |
|------|------|--------|--------|
| `model_type` | `:262` | 每个 config | `AutoConfig` / `AutoModel`（第 1A1 篇） |
| `base_config_key` | `:245` | 子 config | `from_pretrained:707-708` |
| `sub_configs` | `:246` | 复合 config | 各模型 `__post_init__`、`_attn_implementation` setter `:425-437` |
| `has_no_defaults_at_init` | `:247` | 必须传子 config 的类 | `to_diff_dict:1026`、`_get_generation_parameters:1295`；`EncoderDecoderConfig.has_no_defaults_at_init: True`（实测，无参构造会报错） |
| `keys_to_ignore_at_inference` | `:248` | 各 config | 输出 `ModelOutput` 时忽略哪些 key |
| `attribute_map` | `:249` | 老模型 | `__setattr__`/`__getattribute__`（`:476-484`）、`get_text_config:1372-1373` |
| `base_model_tp_plan` | `:250` | 各 config | `distributed/mixin.py:72`、`modeling_utils.py:4682` |
| `base_model_fsdp_plan` | `:251-255` | **基类给了默认值** | `distributed/mixin.py:74` |
| `base_model_pp_plan` | `:256` | 各 config | `distributed/mixin.py:71`、`modeling_utils.py:4703` |
| `base_model_ep_plan` | `:257` | MoE 模型 | `distributed/mixin.py:73` |

### `attribute_map`：老模型的字段改名器

`configuration_utils.py:476-484` 把读写**双向**重定向：

```python
def __setattr__(self, key, value):
    if key in super().__getattribute__("attribute_map"):
        key = super().__getattribute__("attribute_map")[key]
    super().__setattr__(key, value)

def __getattribute__(self, key):
    if key != "attribute_map" and key in super().__getattribute__("attribute_map"):
        key = super().__getattribute__("attribute_map")[key]
    return super().__getattribute__(key)
```

Bloom 的真实映射（`configuration_bloom.py:53-56`）：

```python
attribute_map = {
    "num_hidden_layers": "n_layer",
    "num_attention_heads": "n_head",
}
```

实测双向生效：

```
b.n_layer: 2 | b.num_hidden_layers: 2
设 b.num_hidden_layers=7 后 b.n_layer: 7
设 b.n_layer=9 后 b.num_hidden_layers: 9
```

意义：**checkpoint 里叫 `n_layer`，库内一律叫 `num_hidden_layers`**。
所有按规范名取维度的下游（device_map 规划、TP/PP 计划、量化、PEFT）因此不需要知道老名字。
`utils/rules.toml` 的 TRF023 专门禁止新模型起 `d_model` / `n_head` 这种名字，理由同上。

### `_tied_weights_keys` 在 model 侧，config 侧对应 `tie_word_embeddings`

`_tied_weights_keys` 写在**模型类**上（`modeling_llama.py:422`、`modeling_cohere2.py:421`）：

```python
_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
```

而 config 侧对应的开关是 `tie_word_embeddings`，消费点在 `modeling_utils.py:2559-2560`。
`utils/rules.toml:294-299` 的 TRF015 规定：模型类声明了非空 `_tied_weights_keys` 时，
config **必须**提供 `tie_word_embeddings` 字段，否则用户无法关掉绑定。

### `tie_word_embeddings` 还会反向改 `tp_plan`

`configuration_utils.py:353-357`：

```python
if getattr(self, "tie_word_embeddings", False) and self.base_model_tp_plan is not None:
    self.base_model_tp_plan = {
        **self.base_model_tp_plan,
        "embed_tokens": "embedding_rowwise",
    }
```

实测（Cohere2 默认 `tie_word_embeddings=True`）：

```
Cohere2 tie: True | tp_plan: {..., 'embed_tokens': 'embedding_rowwise'}
```

（Llama 默认 `tie_word_embeddings=False`，所以它的 plan 里**没有**这一条。）

---

## 十、典型 `config.json` 字段速查表

| 字段 | 一句话 |
|------|--------|
| `model_type` | 模型类型标识，`AutoConfig`/`AutoModel` 靠它查表找类；由类属性写入，不是实例字段 |
| `hidden_size` | 残差流宽度；embedding 列数、RMSNorm 宽度、MLP 输入维 |
| `intermediate_size` | FFN 中间层宽度（gate/up 的升维目标） |
| `num_hidden_layers` | DecoderLayer 的堆叠层数 |
| `num_attention_heads` | Q 侧头数；q_proj 输出维 = 它 × `head_dim` |
| `num_key_value_heads` | K/V 头数；< `num_attention_heads` 即 GQA，决定 KV cache 大小 |
| `head_dim` | 单头宽度；RoPE 的旋转维度。v5 允许它 ≠ `hidden_size / num_attention_heads` |
| `max_position_embeddings` | 位置编码缓存的最大长度（Llama 里同时也影响 RoPE 缓存） |
| `rope_theta` | RoPE 频率基数，v5 已迁进 `rope_parameters`（`{"rope_theta":..., "rope_type":...}`），顶层 `rope_theta` 仍被 BC 接受 |
| `rms_norm_eps` | RMSNorm 的 `eps`（T5/Cohere2 叫 `layer_norm_eps`，语义相同） |
| `sliding_window` | 滑动窗口大小；`None` = 全注意力，非 `None` 时换用 sliding mask 函数 |
| `tie_word_embeddings` | 是否共享 `lm_head.weight` 与 `embed_tokens.weight`；`True` 时还会给 `tp_plan` 补 `embed_tokens` 一项 |
| `torch_dtype`（旧） / `dtype`（新） | 权重的默认精度；`torch_dtype` 已废弃，`__post_init__:281-289` 会自动搬到 `dtype` |
| `vocab_size` | 词表大小 = embedding 行数 = logits 最后一维 |
| `layer_types` | 逐层的注意力类型列表，长度必须等于 `num_hidden_layers`（由 `validate_layer_type:526-546` 校验） |
| `transformers_version` | 保存时的库版本，每次 `to_dict` 自动写入（`:1088`） |

真实输出（`LlamaConfig().to_json_string()`，节选，本仓库实测）：

```json
{
  "attention_bias": false,
  "bos_token_id": 1,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "intermediate_size": 11008,
  "max_position_embeddings": 2048,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "rms_norm_eps": 1e-06,
  "rope_parameters": { "rope_theta": 10000.0, "rope_type": "default" },
  "tie_word_embeddings": false,
  "transformers_version": "5.16.0.dev0",
  "vocab_size": 32000
}
```

（`head_dim` 是派生字段被序列化的结果；`rope_parameters` 是 `rope_theta` + `rope_scaling` 合并后的新形态。）

---

## 一句话总结

> `PreTrainedConfig` 是**一张带默认值的结构表**：类体注解定义字段（`ClassVar` 除外），
> `wrap_init_to_accept_kwargs`（`:105`）负责「传了用传的、没传用默认值、多的一律交给 `__post_init__`」，
> `__post_init__`（`:278`）负责派生字段与 BC 搬运。
> 加载 = `from_pretrained(:619)` → `get_config_dict(:730)` → `_get_config_dict(:763)` → `from_dict(:863)` → `cls(**dict)`；
> 保存 = `to_dict(:1076)`（补 `model_type`）→ `to_diff_dict(:1011)`（diff 掉基类字段）→ `validate()`。
> `@strict` 给的是**类型校验 + `validate_*` 框架**，不是「拒绝未知字段」；
> 复合 config 靠 `sub_configs` + `base_config_key` + `get_text_config(:1304)` 三层配合。

---

## 验证清单

环境说明：本机没有装 torch，配置系统只在 `is_torch_available() == False` 下验证
（这不影响结论：`configuration_utils.py` 里运行期真正碰 torch 的地方只有 `dtype` 转
`torch.dtype` 这一处，`:284-289`，被 `is_torch_available()` 守着）。为了能 import `transformers`，
额外从 PyPI 取了 `huggingface_hub`、`regex`、`filelock`、`safetensors` 四个包，
并把模块级 import 的 `modeling_gguf_pytorch_utils`（`:34`，它要求 torch）
用一个空壳模块顶掉。**没有修改仓库任何文件，没有跑任何会写仓库的命令。**

```bash
# 0. 版本与规模
wc -l src/transformers/configuration_utils.py                 # → 1500
grep -n "__version__" src/transformers/__init__.py            # → 21:__version__ = "5.16.0.dev0"
grep -n "huggingface-hub" setup.py                            # → 90: "huggingface-hub>=1.5.0,<2.0"

# 1. @strict 的来源（不在本仓库，从 PyPI 取同 major 版本对照）
pip download "huggingface-hub==1.33.0" --no-deps -d /tmp/hubdl
python -m zipfile -e /tmp/hubdl/huggingface_hub-1.33.0-py3-none-any.whl /tmp/hubdl/hub133
grep -n "def strict" /tmp/hubdl/hub133/huggingface_hub/dataclasses.py
# → 49/53/56: 三个 overload + 真实定义（:56）
grep -n "class StrictDataclass" /tmp/hubdl/hub133/huggingface_hub/errors.py
# → 540/544/548/557

# 2. @strict 的三种行为（真 hub 包，真跑）
python -c "...Plain(hidden_size='512')..."
# → 2) 未知 kwarg -> TypeError : Plain.__init__() got an unexpected keyword argument 'typo_field'
# → 3) 类型错 -> StrictDataclassFieldValidationError : ... expected int, got str (value: '512')
# → 4) 事后赋值类型错 -> StrictDataclassFieldValidationError : ... got float (value: 3.5)

# 3. transformers 侧的声明
grep -n "^@strict" src/transformers/configuration_utils.py  # → 146:@strict(accept_kwargs=True)
grep -rh "^@strict$" src/transformers/models/*/configuration_*.py \
        src/transformers/models/*/modular_*.py | wc -l         # → 1047（全是普通 @strict）
grep -rl "@strict(accept_kwargs=True)" src/transformers/       # → 只有 configuration_utils.py
grep -n "@strict(accept_kwargs=True)" src/transformers/models/nemotron_h/configuration_nemotron_h.py \
        src/transformers/models/vibevoice_asr/configuration_vibevoice_asr.py   # → 无匹配（规则注释与实际不符）

# 4. 类级属性 / 关键方法的行号
grep -n "wrap_init_to_accept_kwargs\|def __post_init__\|def __init_subclass__\|def get_text_config\|def from_dict\|def from_pretrained\|def to_diff_dict\|def to_dict" src/transformers/configuration_utils.py
# → 105 / 278 / 359 / 372 / 619 / 863 / 1011 / 1076 / 1304

# 5. 真实保存一份 config.json（派生字段落盘）
python -c "...Cohere2Config(hidden_size=512,...).save_pretrained(tmp)..."
# → keys 数量: 24；head_dim: 64；layer_types: ['sliding_attention', 'sliding_attention',
#    'sliding_attention', 'full_attention']；是否有 base_model_tp_plan: False
# → 完整 keys 里没有 output_hidden_states / return_dict / architectures

# 6. to_diff_dict 的基线是「基类」，不是「同类」
python -c "...Cohere2Config().to_diff_dict() vs Cohere2Config(hidden_size=512).to_diff_dict()..."
# → 全默认 keys 数: 24 | 改一个字段后: 24 | key 集合相同: True
# → 全默认也写了 attention_dropout: 0.0 | vocab_size: 256000

# 7. 组合式 config（本地造一个 Hub 风格的扁平 config.json，离线跑）
python -c "...json.dump({'model_type':'qwen2_5_vl','text_config':{...1234...},'vision_config':{...}})..."
# → TextConfig.from_pretrained -> hidden_size: 1234 | base_config_key: text_config
# → AutoConfig -> Qwen2_5_VLConfig | text: Qwen2_5_VLTextConfig 1234 | vision: Qwen2_5_VLVisionConfig 5
# → get_text_config() is a.text_config: True
# → 外层 a.hidden_size: <无>
python /tmp/t_config.py     # get_text_config 的三种结局
# → QC.get_text_config() -> Qwen2_5_VLTextConfig
# → QC.get_text_config(encoder=True) -> Qwen2_5_VLConfig（没有 text_encoder，回落到自身）

# 8. model_type：AutoConfig 用得上，跨类型加载只告警
# → AutoConfig 靠 model_type 找到类: type: Cohere2Config
# → LlamaConfig.from_pretrained(cohere2 目录) → 打印 "You are using a model of type `cohere2` to
#    instantiate a model of type `llama`" 但成功返回，hidden_size: 512

# 9. attribute_map 双向
# → BloomConfig.attribute_map: {'num_hidden_layers': 'n_layer', 'num_attention_heads': 'n_head'}
# → 设 b.num_hidden_layers=7 后 b.n_layer: 7 ／ 设 b.n_layer=9 后 b.num_hidden_layers: 9

# 10. tie_word_embeddings 影响 tp_plan
# → Cohere2 tie: True | tp_plan 里 'embed_tokens': 'embedding_rowwise'
# → Llama tie: False | tp_plan['embed_tokens'] = None

# 11. 字段校验器真的会拦
# → StrictDataclassClassValidationError: Class validation error for validator 'validate_layer_type':
#    ValueError: The `layer_types` entries must be in ('full_attention', 'sliding_attention', ...)
# → Cohere2Config(hidden_size='1024') -> StrictDataclassFieldValidationError（构造时）
# → c.hidden_size = '2048'          -> StrictDataclassFieldValidationError（事后赋值）

# 12. 有引用未在正文给行号的关键位置
grep -n "def _get_generation_parameters\|def get_mtp_config\|def recursive_diff_dict\|def get_configuration_file" src/transformers/configuration_utils.py
# → 1288 / 1381 / 1455 / 1425
grep -n "base_model_fsdp_plan\|base_model_tp_plan\|base_model_pp_plan" src/transformers/distributed/mixin.py
# → 71 / 72 / 74（config 的计划在这里被拷到模型上）
grep -n "config.base_model_tp_plan\|config.base_model_pp_plan" src/transformers/modeling_utils.py
# → 4682 / 4703
```

---

## 导航

- 路线图：[00_roadmap.md](./00_roadmap.md)（本篇 = 入门 · 架构 · 1A2）
- 上一篇：[1A1_repo_and_auto.md](./1A1_repo_and_auto.md) —— 仓库骨架与 auto 机制
- 下一篇：[1A3_from_pretrained.md](./1A3_from_pretrained.md) —— `from_pretrained` 全链路
  （config 拿到之后，权重怎么灌进 `_from_config` 造出来的模型）
- 穿插：[2A5_modular_code_generation.md](./2A5_modular_code_generation.md) ——
  本篇引用的 `configuration_cohere2.py` 就是它生成的，读完那篇你会知道
  「config 该改 `modular_*.py` 而不是 `configuration_*.py`」
