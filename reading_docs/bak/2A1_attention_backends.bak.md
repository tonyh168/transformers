# 注意力后端机制精读：`ALL_ATTENTION_FUNCTIONS` 分发注册表

> 阅读目标：搞清楚 `transformers` 怎么做到「**模型代码只写一遍，注意力后端随便换**」。
> 核心是一个全局注册表 `ALL_ATTENTION_FUNCTIONS`，模型 forward 里一行 `get_interface(...)` 就把
> `eager` / `sdpa` / `flash_attention_2` / `flex_attention` 之一取出来调用。
>
> **这篇是「注意力三件套」的第一篇**。三件套是同一件事的三个侧面：
>
> | 篇 | 问题 | 主题 |
> |----|------|------|
> | **2A1（本篇）** | **算什么** | 后端分发：`softmax(QK^T/√d)V` 到底由谁来算 |
> | [2A2](./2A2_masking_system.md) | 在哪算 | mask 系统：哪些位置允许 attend |
> | [2A3](./2A3_kv_cache.md) | 缓存什么 | KV cache：过去的 K/V 存哪、怎么复用 |
>
> 三篇的分界线很清楚：**本篇只讲「函数被选中并调用」这件事**。mask 的具体构造留给 2A2，
> cache 的 update 语义留给 2A3，这里只指出它们在调用契约里的位置。
>
> 源文件（行号为实测，见文末验证清单）：
>
> | 文件 | 行数 | 角色 |
> |------|------|------|
> | `src/transformers/modeling_utils.py` | 5206 | `AttentionInterface` 类定义（:5147）、注册表实例（:5185）、`_attn_implementation` 校验（:1798 / :1926） |
> | `src/transformers/utils/generic.py` | 1210 | 基类 `GeneralInterface`（:1097） |
> | `src/transformers/models/llama/modeling_llama.py` | 512 | 最朴素的 `eager_attention_forward`（:191）、模型侧调用点（:264） |
> | `src/transformers/integrations/sdpa_attention.py` | 170 | `sdpa_attention_forward`（:79） |
> | `src/transformers/integrations/flash_attention.py` | 102 | `flash_attention_forward`（:26） |
> | `src/transformers/modeling_flash_attention_utils.py` | 830 | `_flash_attention_forward`（:694）、FA 版本兼容矩阵（:78） |
> | `src/transformers/integrations/flex_attention.py` | 368 | `flex_attention_forward`（:262） |
> | `src/transformers/integrations/eager_paged.py` | 80 | `eager_paged_attention_forward`（:19） |
> | `src/transformers/integrations/flash_paged.py` | 147 | `paged_attention_forward`（:7） |
> | `src/transformers/integrations/sdpa_paged.py` | 71 | `sdpa_attention_paged_forward`（:18） |
> | `src/transformers/masking_utils.py` | 1601 | mask 侧的同构注册表 `ALL_MASK_ATTENTION_FUNCTIONS`（:725）——**细节见 2A2** |
>
> 人类向文档：`docs/source/en/attention_interface.md`
> 相关：1A2 配置系统（[00_roadmap.md](./00_roadmap.md) 中列出，**尚未写出**）——
> `_attn_implementation` 正是住在 config 上的一个属性。

---

## 速读区（TL;DR）

**一句话**：`transformers` 把「怎么算注意力」抽象成一次**函数查找**——`config._attn_implementation` 决定
**名字**，注册表 `ALL_ATTENTION_FUNCTIONS` 把**名字**翻译成**函数**，**函数**内部再决定真正的 kernel。
模型代码只写一次 `get_interface(...)`，后端随便换。

**必须记住的断言**

| # | 断言 | 出处（文件:行号） |
|---|------|------------------|
| 1 | `AttentionInterface` 类本体就在 `modeling_utils.py`，不在别处（`dummy_pt_objects.py:405` 是无 torch 环境的占位类） | `modeling_utils.py:5147` |
| 2 | 全局单例：`ALL_ATTENTION_FUNCTIONS = AttentionInterface()` | `modeling_utils.py:5185` |
| 3 | 注册表**只有 10 个 key，没有 `"eager"`** | `modeling_utils.py:5156-5167` |
| 4 | `eager` 走 `get_interface(name, default)` 的 `default` 回退，由**每个模型文件自带**一份 | `modeling_utils.py:5169-5181`、`modeling_llama.py:191` |
| 5 | 两处校验表达式一模一样：`not in ["eager"] + ALL_ATTENTION_FUNCTIONS.valid_keys()` | `modeling_utils.py:1928`、`:2126` |
| 6 | 默认后端是 `"sdpa"`，不可用时**静默降级** `eager` | `modeling_utils.py:1927`、`:1953-1960` |
| 7 | FA2/3/4 三个 key 指向**同一个** `flash_attention_forward`，版本分派在函数内部 | `flash_attention.py:26` → `modeling_flash_attention_utils.py:694` |
| 8 | 只有 `eager` 返回真实 `attn_weights`；`sdpa`/`flash` 返回 `None`，`flex` 返回 `lse` | `modeling_llama.py:213`、`sdpa_attention.py:170`、`flex_attention.py:368` |
| 9 | `sliding_window` 语义是「**总**窗口大小 W」，传给 FA 要 `-1` | `modeling_flash_attention_utils.py:655` |
| 10 | `paged\|flash` 的窗口右侧是 `0` 而不是 `W-1` | `flash_paged.py:55` |
| 11 | `flash` 只接受 **2D padding mask**，causal 靠内核参数表达 | `masking_utils.py:607`（详见 2A2） |
| 12 | `sdpa` 的 `is_causal` 被收紧为 `q_length > 1 and attention_mask is None and is_causal` | `sdpa_attention.py:130` |

**最容易踩的坑 / 反直觉**

- **「没指定后端时按硬件自动挑最优」是错的**：`_autoset_attn_implementation` 与 config 上的 `_attn_implementation_autoset` **已从库代码移除**，只剩 3 个权重转换脚本在清理旧 config 的残留字段。真实的默认逻辑只有「`sdpa`，不行静默降级 `eager`」；唯一的自动升级在连续批处理 `switch_to_cb_friendly_attn`（`continuous_api.py:652`）。
- **`eager` 是硬编码特例**：注册表里根本没有 `"eager"` 这个 key，`get_interface` 里那行 `elif attn_implementation != "eager" and attn_implementation not in self` 就是为了绕开查表。
- **位置参数顺序不可信**：`eager` 是 `(…, scaling, dropout)`，`sdpa`/`flash` 是 `(…, dropout, scaling)`；调用方全用关键字传参所以没出问题。**真正的契约是「名字对得上」，不是「位置对得上」。**
- **`sdpa` 有 mask 就用不了原生 GQA**：只要传了 `attention_mask`，PyTorch 会 fallback 到 math kernel，而 math kernel 不支持 `enable_gqa`，所以库自己 `repeat_kv`（`sdpa_attention.py:119-127`）。
- **`register()` 和 `obj[k] = v` 改的不是同一张表**：前者是 `classmethod`，改类属性 `_global_mapping`（全局生效）；后者改实例的 `_local_mapping`（只影响这个实例）。Doge 用的是后者 + 重新赋值模块级名字。
- **能不能动态切换后端，靠 `inspect.getsource` 搜字符串**：搜不到 `ALL_ATTENTION_FUNCTIONS.get_interface(` 就只打 warning，切换不生效（`modeling_utils.py:2009`）。
- **`flex` 遇到 `dropout > 0` 直接抛错**，不是忽略（`flex_attention.py:276`）。

**读完应该能回答**

- `get_interface(name, default)` 为什么必须带第二个参数？那两处 `["eager"] + valid_keys()` 是为什么？
- `register()` / `obj[k] = v` 分别落到哪张表，可见范围差在哪？「加后端」的三条路径分别是什么？
- FA 的 `window_size` 为什么是 `sliding_window - 1`？`paged|flash` 为什么又是 `(W-1, 0)`？
- 显式传 `"sdpa"` 而模型 `_supports_sdpa = False` 时会怎样？什么都不传呢？

---

## 前置知识

- 1A2 配置系统（**尚未写出**，见 [00_roadmap.md](./00_roadmap.md)）：`config._attn_implementation`
  是 config 上的一个属性，`from_pretrained(attn_implementation=...)` 最终写到它上面。
  本篇第七章会回到这里（也可以直接读 `configuration_utils.py:418-427`）。
- [1M1 Llama 结构](./1M1_llama_structure.md)：`LlamaAttention`（`modeling_llama.py:217`）的
  forward 流程 —— 投影 → RoPE → cache → attention → `o_proj`。
  **本篇的对照代码全部基于它**，重点看它的第 4 步。
- [2A5 代码生成机制](./2A5_modular_code_generation.md)：本篇大量用到 Mistral 与 Llama 的对照，
  需要知道「modular 是源文件、modeling 是产物」这条规则。

---

## 一、问题的起源：两行数学，一百种实现

注意力的数学核心就一行：

```
attn = softmax(Q @ K^T / sqrt(d) + mask) @ V
```

`eager_attention_forward`（`modeling_llama.py:191`）就是这么写的，朴素、可读、**慢**：

```python
# modeling_llama.py:191-213（节选）
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights
```

问题在于：**要实现 FlashAttention，必须把上面这几步融成一个 kernel**——
不能先物化完整的 `[B, H, S, S]` 注意力矩阵（那是 O(S²) 显存），要在 SRAM 里分块算 online softmax。
这意味着「算注意力」这件事有**多套互不兼容的实现**：

| 实现 | 谁提供 | 关键约束 |
|------|--------|----------|
| 朴素 matmul + softmax | PyTorch 基础算子 | 无约束，但显存 O(S²) |
| SDPA | `torch.nn.functional.scaled_dot_product_attention` | PyTorch 自动选 kernel，接受 4D mask |
| FlashAttention | 外部包 `flash_attn` / `flash_attn_interface` / kernels hub | 需要 half dtype、只接受 2D padding mask、不接受 dropout=>0（新版） |
| FlexAttention | `torch.nn.attention.flex_attention`（torch 2.5+） | mask 必须表达成函数 / `BlockMask` |

**如果把这些实现直接写进模型 forward，会发生什么？**

```python
# ✗ 反面教材（库里不这么写）
if self.config._attn_implementation == "flash_attention_2":
    from flash_attn import flash_attn_func
    ...
elif self.config._attn_implementation == "sdpa":
    attn_output = F.scaled_dot_product_attention(...)
else:
    ...
```

这会带来三个后果：

1. **每个模型的 forward 里都要塞一遍这套 if-else**——510 个模型目录，抄 510 遍。
2. **调用契约不统一**：flash 要 `(q, k, v)` 转置过的、sdpa 要 4D mask、eager 要可加性 mask，
   每换一个后端都要改写 `LlamaAttention.forward` 的中间代码。
3. **用户加自定义 kernel 无从下手**——只能改库源码，不能从外部注册。

`transformers` 的解法是引入**一层抽象**：把「怎么算注意力」抽象成一个**函数**，用一张**注册表**存起来，
模型只用一句话问注册表「当前 config 要求哪个函数」，然后**按统一签名调用**。

这就是本篇的主题。一句话概括这套设计：

> **模型代码不知道后端的存在，只知道「有一个函数，签名是 (module, q, k, v, mask, **kwargs)，返回值是 (out, weights)」。**

---

## 二、`AttentionInterface`：注册表本体

### 2.1 实测定位：类在哪里

路线图和很多二手资料会说「`AttentionInterface` 定义在别处，`modeling_utils.py` 只是实例化」。
**实测结论是相反的**——类本体就在 `modeling_utils.py`：

```bash
$ grep -rn "class AttentionInterface" src/transformers/
src/transformers/modeling_utils.py:5147:class AttentionInterface(GeneralInterface):
src/transformers/utils/dummy_pt_objects.py:405:class AttentionInterface(metaclass=DummyObject):
```

（`dummy_pt_objects.py` 那个是给没装 PyTorch 的环境用的占位类，`metaclass=DummyObject`，
一访问就抛「你没装 torch」。**不是定义源**。）

真实结构是：

```
utils/generic.py:1097          class GeneralInterface(MutableMapping)      ← 通用基类：两层字典
        ↑
modeling_utils.py:5147         class AttentionInterface(GeneralInterface)  ← 只做三件事：
                                   1. 填 _global_mapping（10 个默认实现）
                                   2. 重写 get_interface()（加校验 + 警告）
                                   3. 什么都不做（剩下全靠基类）
        ↓
modeling_utils.py:5185         ALL_ATTENTION_FUNCTIONS = AttentionInterface()   ← 全局单例，模型 import 的就是它
```

### 2.2 `GeneralInterface`：一个「全局表 + 局部表」的两层字典

`GeneralInterface`（`utils/generic.py:1097`）是 `MutableMapping` 的子类，全文只有 40 行，核心是两张表：

```python
# utils/generic.py:1097-1135（完整）
class GeneralInterface(MutableMapping):
    """
    Dict-like object keeping track of a class-wide mapping, as well as a local one. Allows to have library-wide
    modifications through the class mapping, as well as local modifications in a single file with the local mapping.
    """

    # Class instance object, so that a call to `register` can be reflected into all other files correctly, even if
    # a new instance is created (in order to locally override a given function)
    _global_mapping = {}

    def __init__(self):
        self._local_mapping = {}

    def __getitem__(self, key):
        # First check if instance has a local override
        if key in self._local_mapping:
            return self._local_mapping[key]
        return self._global_mapping[key]

    def __setitem__(self, key, value):
        # Allow local update of the default functions without impacting other instances
        self._local_mapping.update({key: value})

    def __delitem__(self, key):
        del self._local_mapping[key]

    def __iter__(self):
        # Ensure we use all keys, with the overwritten ones on top
        return iter({**self._global_mapping, **self._local_mapping})

    def __len__(self):
        return len(self._global_mapping.keys() | self._local_mapping.keys())

    @classmethod
    def register(cls, key: str, value: Callable):
        cls._global_mapping.update({key: value})

    def valid_keys(self) -> list[str]:
        return list(self.keys())
```

| 操作 | 落到哪张表 | 可见范围 |
|------|-----------|---------|
| `register(k, v)`（**classmethod**） | `_global_mapping`（**类属性**） | 所有实例、所有文件**全局可见** |
| `obj[k] = v`（`__setitem__`） | `self._local_mapping`（**实例属性**） | 只有这个实例可见 |
| `obj[k]`（`__getitem__`） | 先查局部，再查全局 | — |
| `k in obj` / `obj.valid_keys()` | 两张表的并集（`__iter__` / `__len__`） | — |

**一个容易被忽略的细节**：`register` 是 `classmethod`，它只动 `_global_mapping`。

```bash
$ PYTHONPATH=src python -c "..."
# 伪代码推演（本环境无 regex 依赖，未实跑；由源码语义直接得出）：
ALL_ATTENTION_FUNCTIONS.register("my_attn", fn)   # cls = AttentionInterface → 改类属性 → 全局生效
ALL_ATTENTION_FUNCTIONS["my_attn"] = fn           # 走 __setitem__ → 只改这个实例的 _local_mapping
```

**库内两种用法都有真实例子**：

| 用法 | 真实位置 | 目的 |
|------|----------|------|
| `register`（全局） | `integrations/hub_kernels.py:726` `ALL_ATTENTION_FUNCTIONS.register(attn_implementation, kernel_function)` | 从 Hub 加载的 kernel 注册成全库可用 |
| `[k] = v`（局部） | `models/doge/modular_doge.py:193-194`：先 `ALL_ATTENTION_FUNCTIONS = AttentionInterface()`，再塞一个 `"doge_flex_attention"` | 只给 Doge 这一个文件加一个后端名，**不污染别的模型** |

Doge 那两行是「局部覆盖」的教科书写法，原文（`models/doge/modeling_doge.py:248-249`，生成产物）：

```python
ALL_ATTENTION_FUNCTIONS = AttentionInterface()
ALL_ATTENTION_FUNCTIONS["doge_flex_attention"] = flex_attention_forward
```

注意它**重新赋值了模块级名字** `ALL_ATTENTION_FUNCTIONS`——所以在这个文件里，
`ALL_ATTENTION_FUNCTIONS` 指向一个**新的实例**，它的 `_local_mapping` 里多了 `doge_flex_attention`，
而 `_global_mapping` 仍然和全局那个共享（类属性）。这正是 `GeneralInterface` 注释里说的
「even if a new instance is created (in order to locally override a given function)」。

### 2.3 `AttentionInterface` 覆盖了什么

子类只做两件事：填表 + 重写 `get_interface`。

```python
# modeling_utils.py:5147-5181（完整）
class AttentionInterface(GeneralInterface):
    """
    Dict-like object keeping track of allowed attention functions. You can easily add a new attention function
    with a call to `register()`. If a model needs to locally overwrite an existing attention function, say `sdpa`,
    it needs to declare a new instance of this class inside the `modeling_<model>.py`, and declare it on that instance.
    """

    # Class instance object, so that a call to `register` can be reflected into all other files correctly, even if
    # a new instance is created (in order to locally override a given function)
    _global_mapping = {
        "flash_attention_4": flash_attention_forward,
        "flash_attention_3": flash_attention_forward,
        "flash_attention_2": flash_attention_forward,
        "flex_attention": flex_attention_forward,
        "sdpa": sdpa_attention_forward,
        "paged|flash_attention_4": paged_attention_forward,
        "paged|flash_attention_3": paged_attention_forward,
        "paged|flash_attention_2": paged_attention_forward,
        "paged|sdpa": sdpa_attention_paged_forward,
        "paged|eager": eager_paged_attention_forward,
    }

    def get_interface(self, attn_implementation: str, default: Callable) -> Callable:
        """Return the requested `attn_implementation`. Also strictly check its validity, and raise if invalid."""
        if attn_implementation is None:
            logger.warning_once(
                "You tried to access the `AttentionInterface` with a `config._attn_implementation` set to `None`. This "
                "is expected if you use an Attention Module as a standalone Module. If this is not the case, something went "
                "wrong with the dispatch of `config._attn_implementation`"
            )
        elif attn_implementation != "eager" and attn_implementation not in self:
            raise KeyError(
                f"`{attn_implementation}` is not a valid attention implementation registered in the `AttentionInterface`"
            )
        return super().get(attn_implementation, default)
```

`get_interface(attn_implementation, default)` 有**三条分支**，`default` 的语义是「没有这个 key 时用谁」：

| `attn_implementation` | 行为 | 返回 |
|-----------------------|------|------|
| `None` | `warning_once` 提醒「config 上没设」，不报错 | 注册表里没有 `None` → 返回 `default` |
| `"eager"` | **不检查**（因为没有 `"eager"` 这个 key，见下节） | 返回 `default` |
| 其它字符串 | `attn_implementation not in self` → **抛 `KeyError`** | 命中则返回注册表里的函数 |
| 其它 + 已在 hub 注册 | 命中 → 返回那个 kernel 函数 | — |

**关键点：`"eager"` 是硬编码的特例**（`elif attn_implementation != "eager" and attn_implementation not in self`），
因为注册表里**根本没有 `"eager"` 这个 key**。`eager` 走的是 `default` 那条路，
而 `default` 是**模型自己传进来的**——`LlamaAttention` 传的是 `modeling_llama.eager_attention_forward`，
Mistral 传的也是从 Llama import 来的同一个函数。

> **这就是「注册表 + 回退」设计的精髓**：
> 库不强制你为 `eager` 提供一个全局实现，而是让每个模型文件自带一份 `eager_attention_forward`。
> 这样模型可以在自己的 eager 实现里塞私有逻辑（soft-cap、attention sink、动态 mask……），
> 而 `sdpa` / `flash` / `flex` 这些**通用后端**才放进全局注册表共享。
>
> 副作用是：`eager` 的实现是**每模型一份**的（`grep -c "def eager_attention_forward"` 能查到很多份拷贝，
> 由 `# Copied from` 或 modular 机制同步）。这正是 2A5 讲的「单文件哲学」在运行时层面的投影。

---

## 三、注册表里到底有哪些 key

用 AST 直接解析 `modeling_utils.py` 抽出 `_global_mapping`（不 import 库，避免依赖问题）：

```bash
$ python3 -c "import ast; ..."   # 完整命令见验证清单
class AttentionInterface at line 5147
_global_mapping at line 5156
    'flash_attention_4' -> flash_attention_forward
    'flash_attention_3' -> flash_attention_forward
    'flash_attention_2' -> flash_attention_forward
    'flex_attention' -> flex_attention_forward
    'sdpa' -> sdpa_attention_forward
    'paged|flash_attention_4' -> paged_attention_forward
    'paged|flash_attention_3' -> paged_attention_forward
    'paged|flash_attention_2' -> paged_attention_forward
    'paged|sdpa' -> sdpa_attention_paged_forward
    'paged|eager' -> eager_paged_attention_forward
```

**共 10 个 key，没有 `"eager"`**。整理成表：

| key | 实现函数 | 定义位置 | 依赖 |
|-----|----------|----------|------|
| `"sdpa"` | `sdpa_attention_forward` | `integrations/sdpa_attention.py:79` | torch（内置） |
| `"flex_attention"` | `flex_attention_forward` | `integrations/flex_attention.py:262` | torch ≥ 2.5 |
| `"flash_attention_2"` | `flash_attention_forward` | `integrations/flash_attention.py:26` | `flash_attn`（或 hub kernel） |
| `"flash_attention_3"` | 同上（**同一个函数**） | 同上 | `flash_attn_interface` |
| `"flash_attention_4"` | 同上（**同一个函数**） | 同上 | `flash_attn`（`flash-attn-4` 发行名） |
| `"paged\|eager"` | `eager_paged_attention_forward` | `integrations/eager_paged.py:19` | `generation/continuous_batching/` |
| `"paged\|sdpa"` | `sdpa_attention_paged_forward` | `integrations/sdpa_paged.py:18` | 同上 |
| `"paged\|flash_attention_2/3/4"` | `paged_attention_forward` | `integrations/flash_paged.py:7` | 同上 + `flash_attn` |
| `"eager"` | **不在表里** | 各模型文件自带（`modeling_llama.py:191`） | 无 |

两个值得注意的设计：

**1. FA2 / FA3 / FA4 共用同一个 `flash_attention_forward`。**
三个 key 指向同一个 Python 函数——真正的版本分派发生在**函数内部**：

```python
# integrations/flash_attention.py:76-97（节选）
attn_output = _flash_attention_forward(
    query, key, value, attention_mask,
    query_length=seq_len,
    is_causal=is_causal,
    dropout=dropout,
    softmax_scale=scaling,
    sliding_window=sliding_window,
    softcap=softcap,
    use_top_left_mask=_use_top_left_mask,
    target_dtype=target_dtype,
    attn_implementation=module.config._attn_implementation,   # ← 版本信息在这里传下去
    layer_idx=module.layer_idx if hasattr(module, "layer_idx") else None,
    ...
)
```

`_flash_attention_forward`（`modeling_flash_attention_utils.py:694`）拿到 `attn_implementation` 后，
调 `lazy_import_flash_attention(attn_implementation)`（`:248`）→ `_lazy_imports`（`:143`）真正去 import
对应版本的包（`_lazy_imports` 里对 `flash_attn` / `flash_attn_interface` / `flash_attn.cute` / NPU / hub kernel
分别有分支）。**注册表只负责「选包装函数」，版本分派是包装函数内部的事。**

**2. `paged|` 是一个前缀，不是一个独立实现。**
从 key 的形态能看出两级结构：`"paged|eager"` = `paged` 前缀 + `eager` 后缀。
拆解逻辑在 `utils/generic.py:333`：

```python
def split_attention_implementation(implementation: str | None) -> tuple[bool, str | None]:
    """
    Split the optional `paged|` prefix from an attention implementation string.
    """
    if implementation is None:
        return False, None

    is_paged = implementation.startswith("paged|")
    return is_paged, implementation.removeprefix("paged|")
```

**为什么要两级？** 因为 `paged` 不是「另一种注意力数学」，而是「**另一种 KV 存取方式**」——
它换的是 cache 的读写接口（block table、read/write index），不是 softmax 的算法。
所以 `paged|eager` 和 `eager` 的数学完全一样，只是 K/V 从 `PagedAttentionCache` 里取。

---

## 四、模型侧怎么用：`LlamaAttention.forward` 的三行

### 4.1 调用点

这是「模型代码与后端解耦」的唯一接口，`modeling_llama.py:264-277`：

```python
# modeling_llama.py:243-281（LlamaAttention.forward，节选）
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
```

**这 5 行是整个机制的枢纽**。分解一下：

| 行 | 做什么 | 为什么这样写 |
|----|--------|-------------|
| `self.config._attn_implementation` | 读当前后端名 | 后端选择**存在 config 上**，不在模型上。同一个模型类可以跑任何后端 |
| `eager_attention_forward` | 传 fallback | `eager` 不在注册表里，必须由模型提供 |
| `attention_interface(self, q, k, v, mask, ...)` | 把 `self` 作为第一个参数 | 后端需要读 `module.num_key_value_groups`（GQA）、`module.num_key_value_heads`、`module.training`、`module.is_causal`、`module.layer_idx` |
| `dropout=` / `scaling=` | 从模块状态算出来 | 投影出来的 Q/K/V 信息在后端里拿不到，只能 caller 传 |
| `**kwargs` | 透传 | `sliding_window`、`softcap`、`cu_seq_lens_q`、`output_attentions`…… |

注意前 5 步（投影 → RoPE → cache）**完全与后端无关**，是后端无关的公共前缀。
后端只管「从 Q/K/V 到输出」这一段。这条分界线画得很干净。

### 4.2 后端的函数签名契约

所有后端**必须**接受这套前缀参数（位置 + 关键字）：

```
f(
    module,            # 第一个参数永远是注意力模块本身（位置参数）
    query, key, value, # [batch, heads, seq, head_dim]
    attention_mask,    # 第五个参数（位置参数）——语义随后端剧烈变化！见 2A2
    *,
    dropout=...,       # eager / sdpa / flash 支持；flex 明确拒绝
    scaling=...,       # 全部支持（QK^T 的缩放系数，通常是 head_dim**-0.5）
    ...                # 后端特有的关键字参数
) -> (attn_output, attn_weights_or_None)
```

返回值统一是**二元组**：`(attn_output, attn_weights)`。
第二个元素只有 `eager` 真的返回权重，其余一律 `None`（或 flex 的 lse）：

| 后端 | 返回值第二个元素 | 依据 |
|------|------------------|------|
| `eager` | 真实的 `attn_weights` | `modeling_llama.py:213` `return attn_output, attn_weights` |
| `sdpa` | `None` | `sdpa_attention.py:170` `return attn_output, None` |
| `flash` | `None` | `flash_attention.py:102` `return attn_output, None` |
| `flex` | `lse`（log-sum-exp）或 `None` | `flex_attention.py:368` `return attention_output, lse` |
| `paged\|eager` | 真实权重 | `eager_paged.py:80` |
| `paged\|flash` | `None` | `flash_paged.py:97` |
| `paged\|sdpa` | `None` | `sdpa_paged.py:71` |

这解释了 `config.output_attentions = True` 时代码里的硬性检查（`configuration_utils.py:411-415`）：
设了 `output_attentions` 就**必须**用 `eager`，否则抛 `ValueError`——
因为只有 eager 真的产出注意力矩阵。

**参数顺序不一致的坑**：`eager` / `sdpa` / `flash` 的 `dropout` 与 `scaling` 位置不同：

```python
# eager（modeling_llama.py:191-199）
def eager_attention_forward(module, query, key, value, attention_mask, scaling: float, dropout: float = 0.0, **kwargs)
#                                                                 ↑ 先 scaling
# sdpa（sdpa_attention.py:79-89）
def sdpa_attention_forward(module, query, key, value, attention_mask, dropout: float = 0.0, scaling: float | None = None, ...)
#                                                                  ↑ 先 dropout
# flash（flash_attention.py:26-38）
def flash_attention_forward(module, query, key, value, attention_mask, dropout: float = 0.0, scaling: float | None = None,
                            sliding_window: int | None = None, softcap: float | None = None, is_causal: bool | None = None,
                            s_aux: torch.Tensor | None = None, **kwargs)
```

**为什么不出问题**：调用方（`LlamaAttention.forward:274-276`）全部用**关键字**传
（`dropout=...`、`scaling=...`），所以顺序差异被绕过了。
读源码时别被位置骗了——真正的契约是「名字对得上」。

### 4.3 模型间的差异：都只在「多传什么 kwargs」

对比 Mistral（`modular_mistral.py:53` 的 `MistralAttention.forward` → 生成到 `modeling_mistral.py`）：

```python
# modeling_mistral.py:160-174（节选）
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=getattr(self.config, "sliding_window", None),  # main diff with Llama
            **kwargs,
        )
```

**Mistral 与 Llama 在这个调用点上只差一行**——多传了 `sliding_window`。
源码注释自己都写了 `# main diff with Llama`。

这就是这套设计最好的证明：**加一个新特性（滑动窗口），模型侧只多传一个 kwarg**。
至于这个 kwarg 每个后端怎么消化，是第八节的内容。

---

## 五、分发流程图（本篇核心产出）

```
═══════════ 阶段 0 · 建模时（一次）════════════════════════════════════════
from_pretrained(attn_implementation="flash_attention_2")
   │ modeling_utils.py:1506   config._attn_implementation = kwargs.pop(...)
   │ modeling_utils.py:1345   config._attn_implementation_internal =
   ▼                             self._check_and_adjust_attn_implementation(..., is_init_check=True)  :1798
 ├─ split_attention_implementation()  → (is_paged, base)        generic.py:333
 ├─ _compatible_flash_implementations 自动纠偏                  :1829-1844
 ├─ FA 未装但有 kernels？→ 换成 hub kernel 名                   :1863-1892
 └─ get_correct_attn_implementation()                           :1926
      ├─ 校验 not in ["eager"] + ALL_ATTENTION_FUNCTIONS.valid_keys()  ← 不合法抛 ValueError   :1928
      ├─ _flash_attn_can_dispatch(2)：_supports_flash_attn / 包可 import / dtype / 权重在 GPU   :1665
      └─ 返回最终名字，写回 config

═══════════ 阶段 1 · 每次 forward（每层每步）════════════════════════════════
LlamaAttention.forward                                         modeling_llama.py:243
   │ ①投影 :254  ②RoPE :259  ③KV cache :262（见 2A3）  ④mask :391（见 2A2，由 LlamaModel 传入）
   ▼
attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                          self.config._attn_implementation,     ← 名字
                          eager_attention_forward)             :264-266  ← fallback
   │
   │ get_interface 内部（:5169）：名为 None → warning+default；名为 "eager" → 不查表+default；
   │                              名不在表 → KeyError；否则查 _local_mapping 再查 _global_mapping
   ▼
   ┌────────────┬────────────┬───────────────┬──────────────┬───────────────┐
   ▼            ▼            ▼               ▼              ▼               ▼
 "eager"      "sdpa"   "flex_attention"  "flash_attention_2/3/4"  "paged|*"  "kernels-community/xxx"
   │            │            │               │               │               │
   ▼            ▼            ▼               ▼               ▼               ▼
default =   sdpa_attn_  flex_attn_   flash_attention_forward  eager_paged /  hub_kernels.py:726
模型自带    forward     forward      :26  ─┐                 sdpa_paged  /  register()
eager_attn_ :79         :262           │ 传                paged_attn_    （动态注册）
forward :191                           │ attn_impl=         forward
                                       │ config._attn_impl
                                       ▼
                       _flash_attention_forward :694 → lazy_import_flash_attention :248 → _lazy_imports :143
                                       ▼
              ┌──────────┬────────────────┬──────────────┬─────────┬──────────────┐
              ▼          ▼                ▼              ▼         ▼
         flash_attn  flash_attn_    flash_attn.cute    NPU 自定义  hub kernel
         (FA2)       interface(FA3)      (FA4)                     (kernels-community/…)
   │
   ▼  统一调用（modeling_llama.py:268-277）：
       attention_interface(self, query, key, value, attention_mask,
                           dropout=..., scaling=..., **kwargs)
   ▼
   (attn_output, attn_weights) → reshape → o_proj → 返回        :279-281
```

**一句话读图**：`config` 决定**名字**，注册表把**名字**翻译成**函数**，
**函数**内部再决定**真正的 kernel**。三层解耦，每层都可替换。

---

## 六、逐个后端

### 6.1 `eager`：最朴素，也是唯一能给出注意力权重的

位置：`models/llama/modeling_llama.py:191`（**每个模型文件自带一份**，不是全局注册的）。

核心代码在第一节已经贴过，这里看它的两个关键点：

**(1) GQA 靠 `repeat_kv` 手动展开。**

```python
# modeling_llama.py:179-188
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
```

`num_key_value_groups = num_attention_heads // num_key_value_heads`（`modeling_llama.py:225`）。
Llama-3-8B 是 32 Q 头 / 8 KV 头，`n_rep = 4`：把 8 个 KV 头复制成 32 个（用 `expand` + `reshape`，
不真实分配内存）。**这是 GQA 在数学上的「展开」，不是最优解——但它是其余后端要超越的基准。**

**(2) mask 是「可加性浮点矩阵」。**

```python
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask          # :205-206
```

加上去——所以 `eager` 期望 mask 是 **`0` / `-inf` 的浮点张量**（或能被 broadcast 的布尔转浮点结果）。
mask 的具体构造是 2A2 的内容。

**(3) 只有它能返回权重。**

```python
    return attn_output, attn_weights                          # :213
```

`output_attentions=True` 时代码强制要求 `eager`，原因就在这。

**性能定位**：`repeat_kv` 造成 KV 头物理复制（显存 + 多余计算），softmax 前物化 `[B,H,S,S]`
（显存 O(S²)）。S=8K、H=32、bf16 时单层注意力矩阵 ≈ 4 GB —— 这就是必须要有 flash 的原因。

---

### 6.2 `sdpa`：默认后端，GQA 交给 PyTorch

位置：`integrations/sdpa_attention.py:79`。核心调用：

```python
# integrations/sdpa_attention.py:160-169（节选）
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
        **sdpa_kwargs,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None
```

**PyTorch 会在内部自动选 kernel**（flash / memory-efficient / cuDNN / math 四选一），
库这一层不用管。

**GQA 怎么处理？答案：能交给 PyTorch 就交，不能就手动 `repeat_kv`。**
这是「实测确认」的关键点——不是无条件手动展开：

```python
# integrations/sdpa_attention.py:27-37
def use_gqa_in_sdpa(attention_mask: torch.Tensor | None, key: torch.Tensor, value: torch.Tensor) -> bool:
    # GQA can only be used under the following conditions
    # 1.cuda or Ascend NPU
    #   - attention_mask is None (otherwise it will fall back to the math kernel)
    #   - key head_dim == value head_dim <= 256 (otherwise it will fall back to the math kernel)
    # 2.xpu
    #   - torch version >= 2.8
    if _is_torch_xpu_available:
        return _is_torch_greater_or_equal_than_2_8
    return attention_mask is None and key.shape[-1] == value.shape[-1] <= 256
```

```python
# integrations/sdpa_attention.py:119-127
    sdpa_kwargs = {}
    if hasattr(module, "num_key_value_groups") and module.num_key_value_groups > 1:
        if not use_gqa_in_sdpa(attention_mask, key, value):
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
        else:
            sdpa_kwargs = {"enable_gqa": True}
```

翻译成人话：

| 条件 | 做法 |
|------|------|
| 有 `attention_mask`（哪怕只是 padding mask） | **手动 `repeat_kv`**，然后正常算 |
| 无 mask 且 `head_dim ≤ 256` 且 q/v head_dim 相同 | 传 `enable_gqa=True`，**让 PyTorch 原生 GQA** |
| 在 XPU 上 | 只看 torch ≥ 2.8，其它条件不检查 |

**为什么有 mask 就不能用原生 GQA**：源码注释写得很直白——有 mask 时 PyTorch 会 fall back 到
math kernel，而 math kernel 不支持 `enable_gqa`，所以必须自己展开。

**`is_causal` 的判定同样绕**（`sdpa_attention.py:138-146`）：
先从 kwargs 取、取不到再读 `module.is_causal`，最后**还要再与两个条件取与**：

```python
    is_causal = q_length > 1 and attention_mask is None and is_causal
```

即**只有「没有 mask + 不是解码阶段的单 token」才敢把 `is_causal=True` 交给 PyTorch**，
这样 SDPA 才能走最快的 flash/cuDNN kernel；只要传了 mask（哪怕是纯因果的），就只能退化成通用路径。
（上面这几行的原始注释解释了为什么条件要这样排：先比形状再比别的，否则 `torch.compile`
会拿到 `SymBool` 而报 `argument 'is_causal' must be bool, not SymBool`。）

还有一段「prefill 空 StaticCache」的裁剪（`:155-163`）：当 `is_causal=True` 且
`kv_length > q_length` 时，直接 `key = key[:, :, :q_length, :]` 切掉后面的空位——
因为 SDPA 只在 `q_length == kv_length` 时才派发 flash kernel。

**版本说明**：`sdpa` 是**默认后端**——`get_correct_attn_implementation` 里
`applicable_attention = "sdpa" if requested_attention is None else requested_attention`（`modeling_utils.py:1927`），
且 sdpa 不可用时自动降级 `eager`（`:1953-1960`）。

---

### 6.3 `flash_attention_2 / 3 / 4`：换来的性能，以及硬性前提

位置：`integrations/flash_attention.py:26`（包装层）+ `modeling_flash_attention_utils.py:694`（真正的实现）。

**包装层做三件事**（`flash_attention.py`）：

```python
# integrations/flash_attention.py:46-64（节选）
    # This is before the transpose
    seq_len = query.shape[2]

    if any(dim == 0 for dim in query.shape):
        raise ValueError(...)          # FA 不支持 dim=0
    # FA2 uses non-transposed inputs
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # FlashAttention requires the query and value to share a head dim; pad `value` up to the
    # query head dim (e.g. MLA, where `v_head_dim < qk_head_dim`) and crop the output below.
    head_dim, v_head_dim = query.shape[-1], value.shape[-1]
    if v_head_dim != head_dim:
        value = torch.nn.functional.pad(value, [0, head_dim - v_head_dim])
```

1. **转置约定**：库里内部统一是 `[batch, heads, seq, head_dim]`，FA 要 `[batch, seq, heads, head_dim]`，
   所以进出各转一次。
2. **空张量直接报错**（FA 不支持 dim=0）。
3. **MLA 适配**：当 `v_head_dim < qk_head_dim`（DeepSeek MLA 的特征）时把 V pad 到 Q 的维度，
   算完再 `attn_output[..., :v_head_dim]` 裁回来（`:99-100`）。**这是 2M2 的 MLA 能跑在 FA 上的原因。**

还有一个 dtype 兜底：`get_target_dtype(query, module)`（`flash_attention.py:12`）在 query 是 fp32 时
返回一个 FA 能接受的 dtype（autocast dtype / 量化权重 dtype / 模型第一个 Linear 的权重 dtype），
在 `:71` 取出来一路传到 `fa_peft_integration_check`
（`modeling_flash_attention_utils.py:552`）里把 fp32 的 Q/K/V 转回 half。
**起因**：PEFT 训练时 LayerNorm 常被 cast 成 fp32，会连带把 hidden states 也变成 fp32。

**`attn_implementation="flash_attention_2"` 需要的前提**——全部在
`_flash_attn_can_dispatch`（`modeling_utils.py:1665`）里检查：

| 前提 | 检查点 | 不满足时 |
|------|--------|----------|
| 模型声明支持 | `self._supports_flash_attn`（`:1678`） | 抛 `ValueError`（`:1679-1682`） |
| 版本合法 | `flash_attn_version in [2, 3, 4]`（`:1685`） | 抛 `ValueError` |
| 包可 import | `self._flash_attn_import_error(**FLASH_ATTENTION_COMPATIBILITY_MATRIX[v])`（`:1689`） | 抛 `ImportError`，提示装包或换 kernel |
| GPU 计算能力够 | `major < cuda_min_major_version`（`:1660`）→ FA3 要 8/Ampere，FA4 要 9/Hopper | 抛 `ImportError` |
| dtype ∈ {fp16, bf16} | `dtype not in [torch.float16, torch.bfloat16]`（`:1702`） | **只 warning**（`:1708-1711`） |
| 权重不在纯 CPU | `len(param_devices) == 1 and param_devices[0].type == "cpu"`（`:1715`） | 抛 `ValueError`，或提示搬到 GPU（`:1732`） |
| dropout（FA3/4） | `flash_attn_version > 2` 且 `attention_dropout > 0`（`:1691`） | **只 warning**（`:1694-1698`） |

**版本兼容矩阵**在 `modeling_flash_attention_utils.py:78`（`FLASH_ATTENTION_COMPATIBILITY_MATRIX`），
每个版本一个 dict，字段为：

| 字段 | 作用 |
|------|------|
| `general_availability_check` | 该版 FA 是否可用（`is_flash_attn_{2,3,4}_available`） |
| `pkg_availability_check` | 用 `importlib.util.find_spec` + 发行名更严格地判一次（FA2 和 FA4 都发行成 `flash_attn`，靠发行名区分） |
| `supported_devices` | `((is_torch_cuda_available, "cuda"), (…mlu…), (…npu…), (…xpu…))`——FA2 支持 cuda/mlu/npu/xpu，FA3/4 只支持 cuda |
| `cuda_min_major_version` | FA3 = 8（Ampere），FA4 = 9（Hopper） |

**没有 `flash_attn` 时的兜底**：`FLASH_ATTN_KERNEL_FALLBACK`（`modeling_flash_attention_utils.py:65`）把
标准名映射到 Hugging Face Hub 上的预编译 kernel：

```python
FLASH_ATTN_KERNEL_FALLBACK = {
    "flash_attention_2": "kernels-community/flash-attn2",
    "flash_attention_3": (
        "kernels-community/aiter-flash-attn" if is_rocm_platform() else "kernels-community/vllm-flash-attn3"
    ),
    "flash_attention_4": "kernels-community/flash-attn4",
}
```

`_check_and_adjust_attn_implementation`（`modeling_utils.py:1863-1892`）会：如果没有 `flash_attn`
但装了 `kernels`，就悄悄换成对应的 kernel 名并 `logger.warning_once(...)`；
kernel 加载成功后通过 `hub_kernels.py:726` 的 `ALL_ATTENTION_FUNCTIONS.register(...)` 把自己注册进注册表。

**`flash` 的 mask 语义与其他后端完全不同**：它只接受 **2D padding mask**
（`[batch, seq]`，`1` = 真实 token），因为 FA 是「无 padding」的——它靠 `_upad_input`
把序列压扁成 varlen 格式（`modeling_flash_attention_utils.py:773`），
causal 性交给 `causal=True` 这个布尔参数。mask 侧的对应实现是
`masking_utils.py:607` 的 `flash_attention_mask`，它**在无 padding 时直接返回 `None`**：

```python
    if attention_mask is not None:
        # Here we need to slice from the right if using sliding or chunked (for full attention, this is equivalent to doing nothing)
        attention_mask = attention_mask[:, -kv_length:]
        # We only return an actual mask if there is at least 1 padding token AND the length is the same as the kv_length ...
        if attention_mask.shape[1] == kv_length and attention_mask.all():
            attention_mask = None

    return attention_mask
```

—— 这就是「flash 后端与 mask 系统的耦合点」，展开见 2A2。

---

### 6.4 `flex_attention`：mask 变成函数，和 2A2 关系最紧

位置：`integrations/flex_attention.py:262`。torch 2.5+ 的可编程注意力。

核心是 `score_mod`——一个**对注意力打分矩阵逐元素调用的闭包**：

```python
# integrations/flex_attention.py:262-300（节选）
def flex_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Union[torch.Tensor, "BlockMask"],
    scaling: float | None = None,
    softcap: float | None = None,
    s_aux: torch.Tensor | None = None,
    position_bias: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if kwargs.get("dropout", 0.0) > 0:
        raise ValueError(
            "`flex_attention` does not support `dropout`. Please use it with inference"
            " only (`model.eval()`) or turn off the attention dropout in the respective config."
        )

    block_mask = None
    score_mask = None
    if isinstance(attention_mask, BlockMask):
        block_mask = attention_mask
    else:
        score_mask = attention_mask

    if score_mask is not None:
        score_mask = score_mask[:, :, :, : key.shape[-2]]

    def score_mod(score, batch_idx, head_idx, q_idx, kv_idx):
        if softcap is not None:
            score = softcap * torch.tanh(score / softcap)
        if score_mask is not None:
            score = score + score_mask[batch_idx][0][q_idx][kv_idx]
        if position_bias is not None:
            score = score + position_bias[batch_idx, head_idx, q_idx, kv_idx]
        # Note: attention sinks cannot be correctly implemented in score_mod
        # because it requires operating on the full attention matrix before softmax.
        # ==> this is done after flex attention
        return score
```

三个要点：

**(1) mask 有两种形态**，取决于上游 mask 系统给的是 `BlockMask` 还是普通张量：

| `attention_mask` 类型 | flex 收到什么 | 干什么 |
|------------------------|---------------|--------|
| `BlockMask`（`torch.nn.attention.flex_attention`） | `block_mask=` | **真正的稀疏化**——只算允许的块，快 |
| `torch.Tensor` | `score_mask=` | `score_mod` 里逐元素加，只是等价于 eager 的 mask，**不快** |
| `None` | 两个都 None | 纯 causal（由 `is_causal` 之外的默认行为决定） |

`masking_utils.py:650` 的 `flex_attention_mask` 返回的就是 `BlockMask`
（`masking_utils.py:36` 在 torch 不可用时才把它退化 `BlockMask = torch.Tensor`）。
**「什么样的 mask 会变成 `BlockMask`」是 2A2 的主题**，这里只需要知道：
**`flex` 是唯一一个「mask 形状直接决定性能」的后端**，这也是它和 2A2 绑得最紧的原因。

**(2) GQA 也是原生的**，而且带一个 TP（张量并行）的补丁：

```python
# integrations/flex_attention.py:301-310（节选）
    enable_gqa = True
    num_local_query_heads = query.shape[1]

    # When running TP this helps:
    if (num_local_query_heads & (num_local_query_heads - 1)) != 0:
        key = repeat_kv(key, query.shape[1] // key.shape[1])
        value = repeat_kv(value, query.shape[1] // value.shape[1])
        enable_gqa = False
```

判据很妙：**本地 Q 头数不是 2 的幂**（`n & (n-1) != 0`）说明被 TP 切过（比如 32 头切成 24 头），
此时 `enable_gqa` 不支持，退回手动 `repeat_kv`。

**(3) 编译是必须的，而且被做成了单例。**
`WrappedFlexAttention`（`flex_attention.py:59`）用 `__new__` 单例 + `torch.compile(flex_attention, ...)`
把编译开销摊到只付一次；`compile_friendly_flex_attention`（`:114`）在
`torch.compiler.is_compiling()` 时**不**用编译版（否则嵌套编译报错）：

```python
    # Do not use compiled version if already compiling forward (it raises issues)
    flex_attention_compiled = WrappedFlexAttention(training)() if not is_torchdynamo_compiling() else flex_attention
```

**flex 也是唯一支持 attention sink（`s_aux`）和 `softcap` 的后端之一**，
而且 sink 必须**在 flex 之外**做重归一化（`flex_attention.py:349-357`），
因为 `score_mod` 拿不到 softmax 之前的完整矩阵。

**Doge 的真实用例**（`models/doge/modular_doge.py:193-194`）：
Doge（Dynamic Mask Attention）需要一个「动态 mask」，无法用静态 mask 表达，
于是它用 `AttentionInterface()` 的**局部实例**注册了一个绕过 `BlockMask` 的 flex 变体
`"doge_flex_attention"`。这是「加后端」的最小示例：
**不改库源码，一个新实例 + 一行 `[]=`。**

---

### 6.5 `paged` 变体：给连续批处理用的

三个文件：`eager_paged.py:19`、`sdpa_paged.py:18`、`flash_paged.py:7`。

**一句话讲清它与 `generation/continuous_batching/` 的关系**：

> 连续批处理要求 KV cache 是**分页（paged）**的——按固定大小的 block 分配显存，
> 不同请求的序列可以共用零散 block（这就是 vLLM 的核心思路）。
> `paged|` 后端的 `query/key/value` 是**打包（packed）**的，
> 形状是 `[1, nheads, total_tokens, headdim]`，需要额外的 `cu_seq_lens_q/k`、`max_seqlen_q/k`、
> `block_table` 等参数，并通过 `PagedAttentionCache.update(..., read_index=..., write_index=...)` 读写 cache。

关键证据——签名完全不同（多了 6 个参数），且**没有它就不能用**：

```python
# integrations/flash_paged.py:7-24（节选）
def paged_attention_forward(
    module: torch.nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None,  # Unused in flash
    cache: PagedAttentionCache,
    cu_seq_lens_q: torch.Tensor,
    cu_seq_lens_k: torch.Tensor | dict[str, torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int | dict[str, int],
    block_table: torch.Tensor | None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
```

`paged|eager` / `paged|sdpa` 则更保守（`eager_paged.py:29-35`）：

```python
    # Add KV cache to the key and value tensors
    cache: PagedAttentionCache | None = kwargs.pop("cache", None)
    if cache is None:
        raise ValueError(
            "`paged|eager` was called without a paged attention cache. This implementation expects the packed "
            "inputs and the 4D mask that continuous batching prepares; on a standard forward it would attend "
            "bidirectionally. Use `eager` for a standard forward."
        )
```

`paged|eager` / `paged|sdpa` 也是**唯二允许在 CPU 上做连续批处理**的后端
（`generation/continuous_batching/utils.py:56`）。

**谁把模型切到 `paged|`**：`ContinuousBatchingManager.switch_to_cb_friendly_attn`
（`generation/continuous_batching/continuous_api.py:652`），它先尝试升级到 flash（FA3 → FA2），
然后无条件加 `paged|` 前缀：

```python
        # Switch to a paged implementation (always entered if conversion to flash happened)
        if "paged|" not in target_implem:
            model.set_attn_implementation(f"paged|{target_implem}")
            self._original_attn_impl = original_attn_impl
```

**这是全局唯一的「自动挑后端」逻辑**——按硬件可用性在 FA3/FA2 之间选，且只发生在连续批处理路径上。
普通 `generate()` 不会自动升级。细节见 2A3。

---

## 七、`config._attn_implementation` 是怎么定下来的

### 7.1 它住在 config 上，是个 property

`configuration_utils.py:418-427`：

```python
    @property
    def _attn_implementation(self):
        return self._attn_implementation_internal

    @_attn_implementation.setter
    def _attn_implementation(self, value: str | dict | None):
        """We set it recursively on the sub-configs as well"""
        # Set if for current config
        current_attn = getattr(self, "_attn_implementation", None)
        attn_implementation = value if not isinstance(value, dict) else value.get("", current_attn)
        self._attn_implementation_internal = attn_implementation

        # Set it recursively on the subconfigs
        for subconfig_key in self.sub_configs:
            ...
```

三个要点：

1. **真名是 `_attn_implementation_internal`**，`_attn_implementation` 是带 setter 的 property。
   库内部改值时**故意写 `_attn_implementation_internal`**（`modeling_utils.py:2073` 的注释
   写着 "Apply the change (on the internal attr, to avoid setting it recursively)"）——绕开递归。
2. **支持 `dict`**：多模态模型的每个 backbone 可以有不同后端。
   `{"vision_config": "sdpa", "text_config": "flash_attention_2"}` 是合法值
   （`docs/source/en/attention_interface.md` 有例子）。
3. **`output_attentions` 与它联动**（`configuration_utils.py:411-415`）：
   设 `output_attentions=True` 且还没定后端时，**自动 dispatch 到 `eager`**；
   若已经定了别的，直接抛 `ValueError`。

`from_pretrained` 里传入的 `attn_implementation` 在最外层被 pop 掉写进 config
（`modeling_utils.py:1506-1507`）。

### 7.2 校验：那道 `not in ["eager"] + valid_keys()` 题

**第一次校验**（建模早期，`modeling_utils.py:1926-1944`）：

```python
    def get_correct_attn_implementation(self, requested_attention: str | None, is_init_check: bool = False) -> str:
        applicable_attention = "sdpa" if requested_attention is None else requested_attention
        if applicable_attention not in ["eager"] + ALL_ATTENTION_FUNCTIONS.valid_keys():
            message = (
                f'Specified `attn_implementation="{applicable_attention}"` is not supported. The only possible arguments are '
                '`attn_implementation="eager"`, `"paged|eager"`'
            )
            # check `supports_flash_attn_2` for BC with custom code. TODO: remove after a few releases
            if self._supports_flash_attn or getattr(self, "_supports_flash_attn_2", False):
                message += ", "
                for fa_version in FLASH_ATTENTION_COMPATIBILITY_MATRIX.keys():
                    message += f'`"attn_implementation=flash_attention_{fa_version}"`, `"attn_implementation=paged|flash_attention_{fa_version}"`, '
                message = message[:-2]  # remove trailing comma
            if self._supports_sdpa:
                message += ', `"attn_implementation=sdpa"`, `"attn_implementation=paged|sdpa"`'
            if self._supports_flex_attn:
                message += ', `"attn_implementation=flex_attention"`'
            raise ValueError(message + ".")
```

**第二次校验**（多模态 sub-config 的兜底，`modeling_utils.py:2124-2134`）：

```python
                    if sub_implementation not in ["eager"] + ALL_ATTENTION_FUNCTIONS.valid_keys():
                        raise ValueError(
                            f'Specified `attn_implementation="{sub_implementation}"` is not supported for {subconfig_key}. '
                            'The only possible arguments are "eager" (manual attention implementation)'
                            f"or one of the following: {list(ALL_ATTENTION_FUNCTIONS.valid_keys())}"
                        )
```

两处的**表达式完全一样**：`["eager"] + ALL_ATTENTION_FUNCTIONS.valid_keys()`。
——再次印证第三节的结论：**`eager` 必须手动加进去**，因为注册表里没有它。
`valid_keys()` 的值就是第三节那 10 个 key。

**为什么有两处**：第一处是正常的 config 校验；第二处是 `set_attn_implementation` 分发到
「没有对应 PreTrainedModel 子模块的 sub-config」时的兜底（`modeling_utils.py:2108-2134`），
这时候没法调用 `get_correct_attn_implementation`（没模型可查 `_supports_*`），只能查注册表。

### 7.3 自动选择：其实只是「默认 sdpa，不行就 eager」

**这里要纠正一个常见误解。** 很多资料（包括路线图最初的表述）说
「没指定时按硬件/dtype 自动挑最优」。**实测表明这个版本里已经没有这套逻辑了**：

```bash
$ grep -rni "autoset" src/transformers/ | grep -v "models/"
（无输出）
$ grep -rn "_attn_implementation_autoset" src/transformers/
src/transformers/models/ovis2/convert_ovis2_weights_to_hf.py:80
src/transformers/models/internvl/convert_internvl_weights_to_hf.py:54
src/transformers/models/csm/convert_csm.py:88
```

`_autoset_attn_implementation` 这个方法和 config 上的 `_attn_implementation_autoset` 属性
**都已从库代码中移除**，只剩三个**权重转换脚本**里在清理旧 config 的残留字段。

**真实的默认逻辑只有两条**：

```python
# modeling_utils.py:1927
applicable_attention = "sdpa" if requested_attention is None else requested_attention
...
# modeling_utils.py:1956-1962
        elif "sdpa" in applicable_attention:
            # Sdpa is the default, so we try it and fallback to eager otherwise when not possible
            try:
                self._sdpa_can_dispatch(is_init_check)
            except (ValueError, ImportError) as e:
                if requested_attention is not None and "sdpa" in requested_attention:
                    raise e
                applicable_attention = "eager"
```

| 你做了什么 | 最终后端 |
|-----------|---------|
| 什么都不传 | `"sdpa"` |
| 什么都不传 + 该模型 `_supports_sdpa = False` | `"eager"`（静默降级） |
| 显式传 `"sdpa"` 但模型不支持 | **抛错**（不降级） |
| 显式传 `"flash_attention_2"` | 走 flash 检查链；失败即抛错 |
| 走连续批处理 | `switch_to_cb_friendly_attn` 可能升到 `paged\|flash_attention_{3,2}` |

（`modeling_utils.py:1237-1239` 是三面开关的默认值：
`_supports_sdpa = False` / `_supports_flash_attn = False` / `_supports_flex_attn = False`，
每个模型自己覆写成 `True`。Llama / Mistral 都是三个全 `True`
——`modeling_llama.py:334-336`、`modeling_mistral.py:250-252`。）

### 7.4 校验的入口：`PreTrainedModel.__init__`

`modeling_utils.py:1345-1349`：

```python
        # Check the attention implementation is supported, or set it if not yet set (on the internal attr, to avoid
        # setting it recursively)
        self.config._attn_implementation_internal = self._check_and_adjust_attn_implementation(
            self.config._attn_implementation,
            is_init_check=True,
            allow_all_kernels=hub_kernels.ALLOW_ALL_KERNELS,
        )
```

`is_init_check=True` 的含义（见 `_flash_attn_can_dispatch` 的 docstring）：
**这个阶段权重还没搬上设备**，所以「权重是否在 GPU」这类检查往后放（`_flash_attn_can_dispatch`
里那个 `if not is_init_check:` 分支，`:1714`）；早期能确定的错误（比如包没装）就早失败。
注意写的是 `_attn_implementation_internal` 而不是 `_attn_implementation`——
绕开 7.1 里那个会递归到 sub-config 的 setter。

### 7.5 运行时切换：`set_attn_implementation`

`modeling_utils.py:2041`：

```python
    def set_attn_implementation(self, attn_implementation: str | dict, allow_all_kernels: bool = False):
        ...
        if requested_implementation != self.config._attn_implementation:
            # In this case, raise
            if not self._can_set_attn_implementation():
                logger.warning(
                    f"{self.__class__.__name__} does not support setting its attention implementation dynamically, because it "
                    "does not follow the functional approach based on AttentionInterface "
                    "(see https://huggingface.co/docs/transformers/en/attention_interface)"
                )
            else:
                requested_implementation = self._check_and_adjust_attn_implementation(...)
                # Apply the change (on the internal attr, to avoid setting it recursively)
                self.config._attn_implementation_internal = requested_implementation
```

**能不能动态切换，取决于模型是否「functional」**——这是个巧妙的启发式检查（`:1988-2015`）：

```python
        # Heuristic: if we find an `*Attention*(nn.Module)` class, check whether the interface is used
        if re.search(r"^class \w*Attention\w*\(nn\.Module\):", code, re.MULTILINE):
            can_set = "ALL_ATTENTION_FUNCTIONS.get_interface(" in code
        # If no attention layer, assume `True`. Most probably a multimodal model or inherits from existing models
        else:
            can_set = True
```

**它用 `inspect.getsource` 读模型模块的源码，在里面搜字符串 `"ALL_ATTENTION_FUNCTIONS.get_interface("`。**
—— 这个检查的存在，本身就等于库在说：「想让我支持后端切换？照 `get_interface` 的写法写 attention 类。」
（同时它也是「自定义模型不该随便切后端」的保护：没有这行代码，切了也不生效，所以警告而不是静默。）

切换会**递归应用到所有 sub-model**（`modeling_utils.py:2079-2106`），
多模态模型靠 `config.sub_configs` 的**对象身份比较**（`getattr(self.config, subconfig_key) is submodule.config`）
找到对应关系。切换前会 snapshot `self._original_attn_impl`（连续批处理就是这么还原的，
`continuous_api.py:750`）。

---

## 八、kwargs 怎么穿透到后端（以 `sliding_window` 为例）

上一章讲「后端怎么被选中」，这一章讲「同一份模型代码怎么把参数喂给不同后端」。
`sliding_window` 是最完美的例子，因为它**在每个后端的消化方式都不一样**。

### 全链路

```
MistralConfig.sliding_window = 4096
        │
        ├──→ ① 决定 mask 类型（模型侧，MistralModel.forward）
        │       modeling_mistral.py:364
        │       mask_function = create_causal_mask if self.config.sliding_window is None \
        │                        else create_sliding_window_causal_mask
        │       → 生成一个「带窗口」的因果 mask            ← 细节见 2A2
        │
        └──→ ② 作为 kwarg 传给后端（模型侧，MistralAttention.forward）
                modeling_mistral.py:172
                sliding_window=getattr(self.config, "sliding_window", None),   # main diff with Llama
                        │
        ┌───────────────┼────────────────┬──────────────────┬─────────────────┐
        ▼               ▼                ▼                  ▼                 ▼
      eager           sdpa            flex               flash            paged|flash
        │               │                │                  │                 │
   被 **kwargs      被 **kwargs      被 **kwargs      window_size=      window_size=
   吞掉，**完全     吞掉，**完全     吞掉，**完全     (sw-1, sw-1)      (sw-1, 0)
   不用**           不用**           不用**
        │               │                │                  │                 │
        ▼               ▼                ▼                  ▼                 ▼
   靠 ① 的 mask    靠 ① 的 mask   靠 ① 的 mask     靠 FA 的          靠 FA 的
   实现窗口        实现窗口        实现窗口         window_size       window_size
                                   （BlockMask）     参数             参数
```

### 逐后端确认

**eager**（`modeling_llama.py:191-199`）：签名里没有 `sliding_window`，`**kwargs` 全收下、不读。
窗口完全由 ① 产生的 mask 保证（mask 里窗口外是 `-inf`）。

**sdpa**（`sdpa_attention.py:79-90`）：同样走 `**kwargs` 落地无声，窗口靠 mask。

**flex**（`flex_attention.py:262-273`）：同样走 `**kwargs`。它连 `dropout` 都是从 `kwargs` 里读的
（`kwargs.get("dropout", 0.0)`）——所以「Mistral 传 `dropout=`」这件事对 flex 有影响，而 `sliding_window` 没有。

**flash**：签名显式声明（`flash_attention.py:34`），并一路往下传：

```python
# modeling_flash_attention_utils.py:650-655
    if supports_mapping["window_size"] and sliding_window is not None and key_length > sliding_window:
        # The flash attention API sets inclusive boundaries, i.e. (4, 0) would take 4 tokens to the left
        # and the current token for a total size of 5. However, we usually define our window sizes by
        # their total window size (when causal). Encoder models as of now seldom use SWA and when they
        # do, they must align with this symmetric logic, i.e. for a total of `2*sliding_window + 1`.
        flash_kwargs["window_size"] = (sliding_window - 1, sliding_window - 1)
```

**注意这里的数学转换**：库里的 `sliding_window` 语义是「**总窗口大小**」（含当前 token），
FA 的 `window_size=(left, right)` 是「**向左/向右各看几个**」。所以是 `sliding_window - 1`。
**这是最容易读错的一处**：Mistral 的 `sliding_window=4096` 传到 FA 变成 `window_size=(4095, 4095)`。

还有三个细节：

1. **名字映射表**（`modeling_flash_attention_utils.py:135-140`）——HF 的叫法和 FA 的叫法不一致的都在这里：

```python
_hf_api_to_flash_mapping = {
    "dropout": "dropout_p",
    "sliding_window": "window_size",
}
# alternative names within the different flash attention APIs, e.g. for attention sinks
_flash_api_alternative_names = {"s_aux": "learnable_sink"}
```

2. **`key_length > sliding_window` 才设**——序列比窗口还短时不必设，省一次 kernel 分支。

3. **`supports_mapping` 是运行时探测出来的**（`_lazy_define_process_function`，`:224-245`）：
   它用 `inspect.signature` 看具体那版 flash 函数的参数表，只传对方支持的 kwarg。
   —— 这就是「同一个 `flash_attention_forward` 服务 FA2/3/4 和各家 hub kernel」能成立的原因：
   **能力差异用运行时反射抹平**。

**注意这个探测是「一次性的」而不是「每次调用的」**——`_lazy_define_process_function` 的 docstring
明确说了原因（`modeling_flash_attention_utils.py:224-232`）：

> Depending on the version and kernel some features are not supported. Due to limitations in
> `torch.compile`, we opt to statically type which (optional) kwarg parameters are supported
> within `_process_flash_attention_kwargs`.

它在 `lazy_import_flash_attention` 里被调用**一次**（`:271-274`），结果缓存到模块级全局变量
`_process_flash_kwargs_fn`。所以要换成 `functools.partial(..., supports_mapping=...)` 这种静态形式，
而不是每次 forward 都 `inspect.signature`——**否则 torch.compile 会因为动态参数表而放弃 fullgraph**。
（另一个原因见 `lazy_import_flash_attention` 的 docstring：`fullgraph` 需要预先 import，
这也正是 `_check_and_adjust_attn_implementation` 要提前 `lazy_import_flash_attention` 的原因，
`modeling_utils.py:1907-1909` 的注释写着 "preload flash attention here to allow compile with fullgraph"。）

**paged|flash** 的窗口语义又不一样（`flash_paged.py:55`）：

```python
    sliding_window = (-1, -1) if not getattr(module, "sliding_window", False) else (module.sliding_window - 1, 0)
    layer_type = "full_attention" if sliding_window == (-1, -1) else "sliding_attention"
```

——**右侧是 `0` 而不是 `sliding_window - 1`**，因为它只做解码阶段（每个 query 只往后看 1 个 token）。
同时它还用这个值反过来选 mask 类型（`attention_mask` 是 `dict` 时按 `layer_type` 取）。

### 其他穿透的 kwargs

| kwarg | 谁传 | eager | sdpa | flash | flex |
|-------|------|-------|------|-------|------|
| `dropout` | 模型（按 `module.training`） | ✅ 用 | ✅ 用 | ✅ 用（FA3/4 只 warning） | ❌ **>0 直接抛错** |
| `scaling` | 模型（`head_dim**-0.5`） | ✅ | ✅ `scale=` | ✅ `softmax_scale=` | ✅ `scale=` |
| `is_causal` | kwargs / `module.is_causal` | ❌ 不读 | ✅ | ✅ | ❌ 不读 |
| `softcap` | Gemma2 | ❌ | ❌ | ✅ `softcap=` | ✅ 在 `score_mod` 里 |
| `s_aux`（attention sink） | GPT-OSS 等 | ❌ | ❌ | ✅（名字可能是 `learnable_sink`） | ✅ 算完 LSE 后重归一化 |
| `position_bias` | T5 等相对位置 | ❌ | ✅ 合成浮点 mask | ❌ | ✅ 在 `score_mod` 里 |
| `output_attentions` | 用户 | ✅ | ⚠️ warning | ⚠️ warning | — |
| `cu_seq_lens_q/k`、`max_seqlen_q/k` | packing | ❌ | ❌ | ✅（varlen 路径） | ❌ |
| `cache`、`read_index`、`write_index` | 连续批处理 | ❌ | ✅ `paged\|sdpa` | ✅ `paged\|flash` | ❌ |

**这张表就是「为什么需要注册表」的最终答案**：
同一个模型代码，同一组 kwarg，10 个后端各有各的消化方式——**差异全被关在注册表后面了**。

---

## 九、实操对照表

### 9.1 选后端

| 后端 | 依赖 | 支持 GQA | 支持滑动窗口 | 支持自定义 mask | 支持返回权重 | 典型适用场景 |
|------|------|----------|--------------|------------------|--------------|--------------|
| `eager` | 无 | ✅ 手动 `repeat_kv` | ✅ 靠 mask | ✅ 任意可加性浮点 mask | ✅ | 调试、需要 `output_attentions`、新模型还没接后端 |
| `sdpa` | torch（内置） | ✅ 原生 `enable_gqa`（无 mask 时）否则手动 | ✅ 靠 mask | ✅ 4D mask | ❌ | **默认**；CPU / MPS / 通用 GPU 兜底 |
| `flash_attention_2` | `flash_attn>=2`（或无则 hub kernel） | ✅ 原生 | ✅ `window_size=(sw-1, sw-1)` | ❌ 只接受 2D padding mask | ❌ | 训练 / 长序列推理，**GPU + fp16/bf16** |
| `flash_attention_3` | `flash_attn_interface` | ✅ 原生 | ✅ 同上 | ❌ 同上 | ❌ | Hopper 及以上；比 FA2 更快的 prefill |
| `flash_attention_4` | `flash-attn-4` 发行包 | ✅ 原生 | ✅ 同上 | ❌ 同上 | ❌ | Blackwell/Hopper 最新卡；**不支持 dropout 之外的 sink 写法差异大** |
| `flex_attention` | torch ≥ 2.5 | ✅ 原生（TP 切过则手动） | ✅ 靠 `BlockMask`（**真稀疏**） | ✅ **mask 即函数** | ✅ 返回 lse | 自定义稀疏模式（Doge 动态 mask、块稀疏）、需要 softcap/sink |
| `paged\|eager` | 无 | ✅ 手动 | ✅ 靠 mask | ✅ | ✅ | **CPU 上的连续批处理** |
| `paged\|sdpa` | torch | ✅ 手动 | ✅ 靠 mask | ✅ | ❌ | CPU/GPU 连续批处理兜底 |
| `paged\|flash_attention_{2,3,4}` | `flash_attn` | ✅ 原生 | ✅ `(sw-1, 0)` | ❌ | ❌ | **生产环境连续批处理（vLLM 式）** |
| `"kernels-community/xxx"` | `kernels` 库 | 看 kernel | 看 kernel | 看 kernel | 看 kernel | 无 `flash_attn` 环境（版本/编译冲突），从 Hub 拉预编译 kernel |

### 9.2 排错对照

| 现象 | 原因 | 在哪 |
|------|------|------|
| `ValueError: Specified attn_implementation="xxx" is not supported` | 名字不在 `["eager"] + valid_keys()` | `modeling_utils.py:1928` / `:2126` |
| `KeyError: ... is not a valid attention implementation registered` | 名字过了 config 校验但没进注册表（通常是自定义 kernel 注册失败） | `modeling_utils.py:5178-5180` |
| `ValueError: ... does not support Flash Attention 2 yet` | 模型没声明 `_supports_flash_attn = True` | `modeling_utils.py:1678` |
| `ImportError: ... FlashAttention2 requires compute capability >= 8` | GPU 太老 | `modeling_utils.py:1660` |
| `warning: ... only supports torch.float16 and torch.bfloat16` | dtype 是 fp32 | `modeling_utils.py:1708` |
| `ValueError: ... called without a paged attention cache` | 直接用了 `paged|eager` 而没走连续批处理 | `eager_paged.py:32` |
| `ValueError: flex_attention does not support dropout` | `model.training=True` 且 `attention_dropout > 0` | `flex_attention.py:276` |
| `ValueError: The output_attentions attribute is not supported` | 后端不是 `eager` | `configuration_utils.py:412` |
| `warning: ... does not support setting its attention implementation dynamically` | 模型源码里没有 `ALL_ATTENTION_FUNCTIONS.get_interface(` | `modeling_utils.py:2009` |
| `warning: You do not have flash_attn installed, using kernels-community/... instead` | 走 kernel 兜底了 | `modeling_utils.py:1891` |

### 9.3 三种「加后端」的路径

| 想干什么 | 怎么做 | 真实例子 |
|----------|--------|----------|
| 全局加一个新后端（如 hub kernel） | `ALL_ATTENTION_FUNCTIONS.register(name, fn)` | `integrations/hub_kernels.py:726` |
| 只给一个模型加后端 | 文件内 `ALL_ATTENTION_FUNCTIONS = AttentionInterface()` + `ALL_ATTENTION_FUNCTIONS[name] = fn` | `models/doge/modular_doge.py:193-194` |
| 覆盖已有后端的实现 | 同上（局部实例的 `_local_mapping` 优先于 `_global_mapping`） | `AttentionInterface` docstring（`modeling_utils.py:5148-5152`） |

---

## 一句话总结

> **注册表是「名字 → 函数」，config 决定「用哪个名字」，模型只写一次调用。**
>
> - `AttentionInterface`（`modeling_utils.py:5147`）是两层字典：`_global_mapping`（全局，`register()` 改）
>   + `_local_mapping`（局部，`obj[k]=v` 改），局部优先。
> - 注册表**只有 10 个 key**（`sdpa` / `flex_attention` / `flash_attention_2|3|4` / 5 个 `paged|*`），
>   **没有 `eager`**——`eager` 由每个模型文件自带，通过 `get_interface(name, default)` 的
>   `default` 参数回退提供。这就是那两处 `not in ["eager"] + valid_keys()` 校验的由来。
> - 模型侧只有一行 `ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, eager_attention_forward)`，
>   Mistral 与 Llama 的差别只是多传了一个 `sliding_window=`。
> - 默认后端是 `"sdpa"`（不支持则静默降级 `eager`）；**本版本已无基于硬件的自动挑 flash 逻辑**，
>   唯一的自动升级发生在连续批处理路径（`switch_to_cb_friendly_attn`）。
> - 后端的差异全在内部：GQA（手动 expand vs 原生 `enable_gqa`）、mask 形态
>   （可加性浮点 vs `BlockMask` vs 2D padding）、`sliding_window` 语义
>   （mask vs `window_size=(sw-1, sw-1)` vs `(sw-1, 0)`）。

---

## 验证清单

以下命令全部在仓库根目录实跑过，输出为原文摘录。

```bash
# 环境说明：本机未装 transformers（缺 `regex` 依赖），所以第 2 条改用 AST 静态解析源码；
# 其余全部是 grep / sed 实跑，`#` 后是原文输出。

# ── 1. `AttentionInterface` 类的真实定义位置（关键结论：就在 modeling_utils.py，不是别处）──
grep -rn "class AttentionInterface" src/transformers/
#   src/transformers/modeling_utils.py:5147:class AttentionInterface(GeneralInterface):
#   src/transformers/utils/dummy_pt_objects.py:405:class AttentionInterface(metaclass=DummyObject):   ← 无 torch 环境的占位类
grep -n "ALL_ATTENTION_FUNCTIONS: AttentionInterface" src/transformers/modeling_utils.py
#   5185:ALL_ATTENTION_FUNCTIONS: AttentionInterface = AttentionInterface()

# ── 2. 注册表的真实 key 列表（AST 解析，不 import 库）──
python3 -c "
import ast
src = open('src/transformers/modeling_utils.py').read()
for node in ast.walk(ast.parse(src)):
    if isinstance(node, ast.ClassDef) and node.name == 'AttentionInterface':
        print('class AttentionInterface at line', node.lineno)
        for item in node.body:
            if isinstance(item, ast.Assign) and any(getattr(t,'id',None)=='_global_mapping' for t in item.targets):
                print('_global_mapping at line', item.lineno)
                for k,v in zip(item.value.keys, item.value.values):
                    print('   ', repr(k.value), '->', ast.unparse(v))
"
# class AttentionInterface at line 5147
# _global_mapping at line 5156
#     'flash_attention_4' -> flash_attention_forward       'flex_attention' -> flex_attention_forward
#     'flash_attention_3' -> flash_attention_forward       'sdpa' -> sdpa_attention_forward
#     'flash_attention_2' -> flash_attention_forward       'paged|flash_attention_4' -> paged_attention_forward
#     'paged|flash_attention_3' -> paged_attention_forward 'paged|sdpa' -> sdpa_attention_paged_forward
#     'paged|flash_attention_2' -> paged_attention_forward 'paged|eager' -> eager_paged_attention_forward
# → 10 个 key，没有 "eager"（这正是第二节「eager 走 default 回退」的根据）

# ── 3. GeneralInterface 基类锚点 ──
grep -n "class GeneralInterface\|_global_mapping = {}\|def __getitem__\|def __setitem__\|def __iter__\|def __len__\|def register\|def valid_keys" \
     src/transformers/utils/generic.py
#   1097 / 1105 / 1110 / 1116 / 1123 / 1127 / 1131 / 1134（依次对应上面各模式）
#   （521 与 535 是另一个类里的同名方法，无关）

# ── 4. 模型侧调用点（Llama / Mistral）与 Mistral 的唯一差异 ──
grep -n "get_interface" src/transformers/models/llama/modeling_llama.py src/transformers/models/mistral/modeling_mistral.py
#   llama:264   mistral:160
grep -n "sliding_window" src/transformers/models/mistral/modeling_mistral.py
#   172:  sliding_window=getattr(self.config, "sliding_window", None),  # main diff with Llama
#   364:  mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask

# ── 5. eager 实现在模型文件里，不在注册表中 ──
grep -n "def eager_attention_forward\|def repeat_kv\|class LlamaAttention" src/transformers/models/llama/modeling_llama.py
#   179 repeat_kv ／ 191 eager_attention_forward ／ 217 class LlamaAttention

# ── 6. 各后端函数定义行号 ──
grep -n "^def sdpa_attention_forward\|^def use_gqa_in_sdpa" src/transformers/integrations/sdpa_attention.py   # 79 / 27
grep -n "^def flash_attention_forward" src/transformers/integrations/flash_attention.py                       # 26
grep -n "^def flex_attention_forward\|^def compile_friendly_flex_attention" src/transformers/integrations/flex_attention.py  # 262 / 114
grep -n "^def eager_paged_attention_forward" src/transformers/integrations/eager_paged.py                      # 19
grep -n "^def paged_attention_forward" src/transformers/integrations/flash_paged.py                            # 7
grep -n "^def sdpa_attention_paged_forward" src/transformers/integrations/sdpa_paged.py                        # 18
grep -n "^def _flash_attention_forward" src/transformers/modeling_flash_attention_utils.py                     # 694
# 返回值（确认只有 eager 给出真实权重）
grep -n "return attn_output, None" src/transformers/integrations/sdpa_attention.py src/transformers/integrations/flash_attention.py  # 170 / 102
grep -n "return attention_output, lse" src/transformers/integrations/flex_attention.py                          # 368
grep -n "return attn_output, attn_weights" src/transformers/models/llama/modeling_llama.py                      # 213 / 281

# ── 7. 两处校验（任务点名的 1928 / 2126）──
sed -n '1927,1928p;2126p' src/transformers/modeling_utils.py
#   applicable_attention = "sdpa" if requested_attention is None else requested_attention
#   if applicable_attention not in ["eager"] + ALL_ATTENTION_FUNCTIONS.valid_keys():
#   if sub_implementation not in ["eager"] + ALL_ATTENTION_FUNCTIONS.valid_keys():

# ── 8. 默认值与降级（7.3 节的依据）──
sed -n '1953,1960p' src/transformers/modeling_utils.py
#   elif "sdpa" in applicable_attention:
#       # Sdpa is the default, so we try it and fallback to eager otherwise when not possible
#       try:
#           self._sdpa_can_dispatch(is_init_check)
#       except (ValueError, ImportError) as e:
#           if requested_attention is not None and "sdpa" in requested_attention:
#               raise e
#           applicable_attention = "eager"

# ── 9. 反证：本版本已无 autoset 逻辑 ──
grep -rni "autoset" src/transformers/ | grep -v "models/"    # → 无输出
grep -rn "_attn_implementation_autoset" src/transformers/
#   models/ovis2/convert_ovis2_weights_to_hf.py:80    models/csm/convert_csm.py:88
#   models/internvl/convert_internvl_weights_to_hf.py:54
#   → 只剩权重转换脚本在清理旧 config 字段

# ── 10. sliding_window → flash window_size 的转换 ──
grep -n "_hf_api_to_flash_mapping\|_flash_api_alternative_names" src/transformers/modeling_flash_attention_utils.py   # 135 / 140
grep -n 'supports_mapping\["window_size"\]\|flash_kwargs\["window_size"\]' src/transformers/modeling_flash_attention_utils.py  # 650 / 655
grep -n "sliding_window = " src/transformers/integrations/flash_paged.py                                       # 55

# ── 11. FA 兼容矩阵、kernel 兜底、各 can_dispatch ──
grep -n "FLASH_ATTENTION_COMPATIBILITY_MATRIX = \|FLASH_ATTN_KERNEL_FALLBACK = " src/transformers/modeling_flash_attention_utils.py  # 78 / 65
grep -n "def _flash_attn_can_dispatch\|def _sdpa_can_dispatch\|def _flex_attn_can_dispatch" src/transformers/modeling_utils.py  # 1665 / 1740 / 1771

# ── 12. 动态切换的能力探测（用源码字符串搜 get_interface）──
grep -n 'can_set = "ALL_ATTENTION_FUNCTIONS.get_interface(" in code' src/transformers/modeling_utils.py   # 2009

# ── 13 / 14. 局部覆盖（Doge）与全局注册（hub kernel）的真实例子 ──
grep -n 'ALL_ATTENTION_FUNCTIONS = AttentionInterface()\|ALL_ATTENTION_FUNCTIONS\["doge_flex_attention"\]' \
     src/transformers/models/doge/modular_doge.py src/transformers/models/doge/modeling_doge.py
#   modular_doge.py:193 / :194      modeling_doge.py:248 / :249
grep -n "ALL_ATTENTION_FUNCTIONS.register" src/transformers/integrations/hub_kernels.py    # 726

# ── 15 / 16. mask 侧同构注册表（2A2 入口）与 paged 前缀 ──
grep -n "class AttentionMaskInterface\|^ALL_MASK_ATTENTION_FUNCTIONS" src/transformers/masking_utils.py    # 711 / 725
grep -n "def split_attention_implementation" src/transformers/utils/generic.py                              # 333
grep -n "paged|" src/transformers/generation/continuous_batching/continuous_api.py                          # 660 / 672 / 675 / 681 / 682
```

**没跑的命令**（有意为之）：`make fix-repo` / `python utils/modular_model_converter.py` /
任何模型推理。`PYTHONPATH=src python -c "import transformers"` 因为本机缺 `regex` 依赖而未能完成，
所以第三节的 key 列表改用 **AST 静态解析**获得（等价且更可靠：不依赖环境，读的就是源码字面量）。

---

## 导航

- 路线图：[00_roadmap.md](./00_roadmap.md)（本篇 = 进阶 · 架构 · 2A1，前置 1A2、1M1）
- 上一篇：[1A6_trainer.md](./1A6_trainer.md)
- **同题补篇**：[2A7_attention_contract.md](./2A7_attention_contract.md) ——
  本篇讲「名字 → 函数」（分发），2A7 讲「函数 → 函数」（**接口契约与能力协商**）：
  契约的三层（硬契约 / `**kwargs` 协商层 / `_supports_*` 准入层）、
  为什么 `_supports_sdpa` 默认是 `False`、以及「特化了 eager 的模型还能不能用 FA」。
  本篇第四节留下的「位置参数顺序不可信」在 2A7 里有完整接续。
- **强烈建议先读**：[2A5_modular_code_generation.md](./2A5_modular_code_generation.md)
  ——读完你才知道为什么「每个模型文件都自带一份 `eager_attention_forward`」，
  以及为什么 `models/*/modeling_*.py` 里的重复代码不能直接改。
- 下一篇：[2A2_masking_system.md](./2A2_masking_system.md) —— **「在哪算」**。
  `ALL_MASK_ATTENTION_FUNCTIONS`（`masking_utils.py:725`）是 `ALL_ATTENTION_FUNCTIONS` 的同构副本，
  两边的 key 必须**严格对应**（`hub_kernels.py:727` 就是一行同时注册进两张表）。
  本篇留下的三个悬念：为什么 `flash` 只要 2D mask 而 `sdpa` 要 4D、`BlockMask` 什么时候生成、
  `sliding_window` 到底怎么变成 mask——都在 2A2。
- 再下一篇：[2A3_kv_cache.md](./2A3_kv_cache.md) —— **「缓存什么」**。
  本篇里 `paged|*` 那几个后端就是 cache 形状决定后端选择的直接证据。

**建议紧接着读**：回到 [1M1](./1M1_llama_structure.md) 的 `LlamaAttention.forward`，
用本篇的视角重看一遍——现在应该能一眼分辨：**哪些行与后端无关（投影 / RoPE），
哪 5 行是分发点，哪些参数是给后端的「暗号」**。
