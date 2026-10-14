# 注意力接口契约与能力协商：为什么「接口一样」还不够

> 阅读目标：回答一个看起来很简单、但决定了很多 bug 的问题——
> **`eager` / `sdpa` / `flash_attention_2` 之间凭什么能互换？互换之后语义还一样吗？**
>
> **与 2A1 的分工**（两篇不重复）：[2A1](./2A1_attention_backends.md) 讲**分发**（名字 → 函数：注册表 `ALL_ATTENTION_FUNCTIONS`、`get_interface`、10 个 key）；**本篇（2A7）** 讲**可替换性**（函数 → 函数：接口契约、扩展点 `**kwargs`、能力协商、准入声明 `_supports_*`）。
> 读完 2A1 你知道「怎么换」；读完本篇你知道「**换了之后会不会悄悄算错**」。
>
> 源文件（行号为实测，见文末验证清单）：`models/llama/modeling_llama.py`（参照实现：`eager_attention_forward` :191、调用点 :264-277）、
> `integrations/sdpa_attention.py`（:79-170）、`integrations/flash_attention.py`（:26-102）、`integrations/flex_attention.py`（:262-368）、
> `modeling_flash_attention_utils.py`（FA 运行时能力探测 :224-245、kwargs 过滤 :591-692、版本分派 :143-181）、
> `modeling_utils.py`（校验与准入 :1237-1239、:1740-1758、:1926-1960；注册表 :5147-5185）、
> `masking_utils.py`（mask 侧同构注册表 :711-725）、`configuration_utils.py`（`_attn_implementation` :418-437、`output_attentions` setter :407-414）。
>
> 相关：**2A1**（后端分发）、**2A2**（mask 格式）、**2M1 Gemma 2**（soft-cap 被静默丢弃的完整案例）。

---

## 速读区（TL;DR）

**一句话**：接口只统一了「**怎么调用**」，没有统一「**能力**」。所以「能换」和「换了还对」是两件事——
前者靠注册表，后者靠**每个模型自己声明的准入标志**。

**必须记住的断言**

| # | 断言 | 出处 |
|---|------|------|
| 1 | 硬契约只有两样：**5 个位置参数** + **返回 2-tuple** | `modeling_llama.py:268-277`、`sdpa_attention.py:79-95` |
| 2 | 输入布局 `[B, H, S, D]`，输出必须是 `[B, S, H, D]`——调用方紧接着 `reshape(*input_shape, -1)` 送 `o_proj` | `modeling_llama.py:279-280` |
| 3 | 只有 `eager` 返回真实权重；`sdpa`/`flash` 返回 `None`，`flex` 返回 `lse` | `modeling_llama.py:213`、`sdpa_attention.py:170`、`flash_attention.py:102`、`flex_attention.py:368` |
| 4 | 模型特有的数学**不在**签名里，而是作为**额外关键字参数**往下传 | `modeling_gemma2.py:282`（`softcap=`）、`modeling_gpt_oss.py:329`（`s_aux=`） |
| 5 | 后端对不认识的 kwarg 是**静默丢弃**（sdpa）或**静默不传**（flash），**不报错** | `sdpa_attention.py:89` 的 `**kwargs`；`modeling_flash_attention_utils.py:662` |
| 6 | FA 的能力是**运行时反射**出来的：`inspect.signature` → `supports_mapping` | `modeling_flash_attention_utils.py:224-245` |
| 7 | 同一语义在不同内核里**可能不同名**：`s_aux` ↔ `learnable_sink` | `modeling_flash_attention_utils.py:140`、`:665-668` |
| 8 | `_supports_sdpa` / `_supports_flash_attn` / `_supports_flex_attn` 的**默认值全是 `False`** | `modeling_utils.py:1237-1239` |
| 9 | 显式要 `sdpa` 而模型不支持 → **抛错**；什么都不传 → **静默降级 eager** | `modeling_utils.py:1953-1960` |
| 10 | 实测 **15 个模型**显式写着 `_supports_sdpa = False` | 见第七节 |
| 11 | 其中 5 个是同一原因：**attention sink** 无法用 sdpa 的接口表达 | `modeling_gpt_oss.py:251-252`、`:387` |
| 12 | `eager` 不是「一个」实现，是 **316 份**——每个模型文件自带一份 | `grep -rl "^def eager_attention_forward" src/transformers/models/*/modeling_*.py \| wc -l` |

**最容易踩的坑 / 反直觉**

- **「特化了 eager 就等于不能用 sdpa/FA」是错的。** Gemma 2 特化了 eager（soft-cap），
  但它的 `attn_implementation="flash_attention_2"` **完全正确**——因为 FA 的签名里恰好有 `softcap`。
  能不能换取决于「**后端有没有对应的表达方式**」，不是「模型有没有特化」。详见第六节。
- **最危险的不是报错，是静默。** Gemma 2 + 默认 `sdpa`：soft-cap 被 `**kwargs` 吞掉，
  不报错、不警告、结果不同（**2M1 已单独立案**）。所以作者宁可写 `_supports_sdpa = False` 把路堵死。
- **`_supports_sdpa` 默认是 `False`，不是 `True`。** 模型必须**显式 opt-in**。
  实测 495 个 `modeling_*.py` 里只有 340 个写了 `= True`，**141 个没写**——
  其中 **13 个**走了 functional 接口（有 `get_interface`）却没声明，
  于是「什么都不传」时它们**静默跑 eager**。详见第七节。
- **`output_attentions` 会自动改后端，但有顺序依赖**：只有 `_attn_implementation` 还是 `None` 时才顺手设成 eager；
  已经是 `sdpa` 就直接抛错（`configuration_utils.py:407-414`）。
- **`paged|eager` 走的不是 `eager_attention_forward`**，是 `eager_paged_attention_forward`，
  它从 kwargs 里捞 paged cache，普通 forward 用它会抛 `ValueError`（`eager_paged.py:29-35`）。
- **FA 没装包时不降级到 eager**，而是从 Hub 拉内核，或直接报错（`modeling_utils.py:1862-1880`）。

**读完应该能回答**

- 一个模型自定义的 `softcap`，为什么 sdpa 会算错、flash 却算对？
- `_supports_sdpa = False` 的模型，显式传 `"sdpa"` 和什么都不传，分别发生什么？
- 为什么 `s_aux` 到 FA4 要改名 `learnable_sink`？谁做的映射？
- eager 一共会在哪 5 种情况下被用到？

---

## 前置知识

- [2A1 注意力后端机制](./2A1_attention_backends.md)：**本篇的前置**（注册表结构、10 个 key、`get_interface(name, default)` 的回退语义、「位置参数顺序不可信」）。
- [1M1 Llama 结构](./1M1_llama_structure.md)（本篇对照代码全部基于它）、[2A2 Mask 系统](./2A2_masking_system.md)（`AttentionMaskInterface` 与 mask 格式）、[2M1 Gemma 2](./2M1_gemma2.md)（soft-cap 静默丢弃的完整证据链——本篇把它当**案例**引用）。

---

## 一、核心心智模型：契约有三层

这是本篇最重要的一张图。把「接口」想成三层，就能解释后面所有的怪现象：

```
┌─ 第三层：准入层 ─────────────────────────────────────────────┐
│  _supports_sdpa / _supports_flash_attn / _supports_flex_attn / _compatible_flash_implementations
│  作用：作者**预判**后端能不能表达这个模型的数学，不能就把路堵死；失败方式：抛错（安全）
└──────────────────────────────────────────────────────────────┘
┌─ 第二层：协商层（唯一的扩展点）──────────────────────────────┐
│  额外关键字参数：softcap / s_aux / sliding_window / position_bias
│  作用：模型把私有数学「喊」出来，后端**按名字**决定认不认；失败方式：静默丢弃（危险 ← 所有坑都在这层）
└──────────────────────────────────────────────────────────────┘
┌─ 第一层：硬契约（类型签名里的）──────────────────────────────┐
│  5 个位置参数 + 返回 2-tuple + 张量布局
│  作用：保证「能被调用」，是可替换性的**下限**；失败方式：TypeError / 形状报错（安全）
└──────────────────────────────────────────────────────────────┘
```

**关键推论**：注册表（2A1 的主题）只保证**第一层**。后两层是约定，不在类型系统里，只能靠文档 + 类属性 + 各实现的自觉。

---

## 二、第一层：硬契约逐条拆解

### 2.1 五个位置参数

```python
# modeling_llama.py:268-277（调用方，唯一的真相来源）
attn_output, attn_weights = attention_interface(
    self,              # ① 模块实例本身（不是 config！）
    query_states,      # ② [B, num_heads,    Sq,  head_dim]
    key_states,        # ③ [B, num_kv_heads, Skv, head_dim]
    value_states,      # ④ 同 key
    attention_mask,    # ⑤ 格式随后端而变，见 2A2
    dropout=0.0 if not self.training else self.attention_dropout,
    scaling=self.scaling,
    **kwargs,
)
```

**① `module` 而不是 `config`**：后端只**读**模块上的运行期信息——`module.num_key_value_groups`（GQA 是否 `repeat_kv`，`modeling_llama.py:201-202`、`sdpa_attention.py:97`）、`module.training`（dropout 是否生效，`modeling_llama.py:209`）、`module.is_causal`（因果内核，`sdpa_attention.py:108`、`flash_attention.py:74`）、`module.layer_idx`（paged 写 cache，`flash_attention.py:90`）、`module.config._is_quantized`（量化 dtype 修正，`flash_attention.py:19`）。

**② 输入布局 `[B, H, S, D]`**（sequence-major，"头在前"，与 sdpa 约定一致）；FA 需要 `[B, S, H, D]`，所以 **FA 自己转**（`flash_attention.py:55-58`，`query/key/value = ... .transpose(1, 2)`）——**转换责任在后端，不在模型**：这是「契约选的是 sdpa 的布局，FA 迁就它」的证据。
**③ 输出必须是 `[B, S, H, D]`**：调用方紧接着 `attn_output.reshape(*input_shape, -1).contiguous()` → `self.o_proj(...)`（`modeling_llama.py:279-281`，`input_shape` = `hidden_states.shape[:-1]`）。所以每个实现结尾都 `transpose(1, 2).contiguous()`：`eager`（`modeling_llama.py:211`）、`sdpa`（`sdpa_attention.py:168`）、`flex`（内部处理，`flex_attention.py:367`）。

### 2.2 返回 2-tuple，第二个元素基本是 `None`

| 后端 | 第二项 | 出处 |
|---|---|---|
| `eager` | **真实 `attn_weights`** | `modeling_llama.py:213` |
| `sdpa` / `flash_attention_*` | `None` | `sdpa_attention.py:170` / `flash_attention.py:102` |
| `flex_attention` | `lse`（log-sum-exp）或 `None` | `flex_attention.py:368` |
| `paged\|eager` | 真实权重 | `eager_paged.py:80` |

调用方按位置解包（`attn_output, attn_weights = ...`），少一项直接 `ValueError`——所以 `None` 成了「我没算权重」的统一表达。**下游后果**：Llama 拿到后直接丢掉（`modeling_llama.py:308` 的 `hidden_states, _ = ...`），真正取权重靠 hook 录制（`_can_record_outputs = {"attentions": LlamaAttention}`，`modeling_llama.py:340-343`）——非 eager 后端录到的全是 `None`。

### 2.3 硬契约里**没有**的东西

| 不在契约里 | 实际现状 |
|---|---|
| 参数**顺序** | `eager` 是 `(…, scaling, dropout)`，`sdpa`/`flash` 是 `(…, dropout, scaling)`；靠调用方全用关键字传绕过（2A1 已详述） |
| `head_dim` 相等 | FA 要求 `v_head_dim == qk_head_dim`，不满足就 pad V 再 crop（MLA 场景，`flash_attention.py:62-64`、`:100`） |
| GQA 怎么实现 | eager 手动 `repeat_kv`；sdpa 优先 `enable_gqa=True`（无 mask 时），否则手动 |
| **模型私有数学** | **靠 kwargs 传 → 这就是第三节** |

---

## 三、第二层：协商层——`**kwargs` 是唯一的扩展点

### 3.1 模型怎么"喊"出自己的私有数学

```python
# modeling_llama.py:268-277 —— Llama 什么都不多传（只有 dropout=... / scaling=self.scaling / **kwargs）
# modeling_gemma2.py:273-283 —— Gemma 2 多传两个：
#     sliding_window=self.sliding_window,        # ← 多
#     softcap=self.attn_logit_softcapping,       # ← 多
# modeling_gpt_oss.py:329 —— GPT-OSS 多传 sink：
#     s_aux=self.sinks,  # diff with Llama
```

**这套接口真正的扩展点就是 `**kwargs`**：模型用关键字名声明需求，后端自己决定认不认。

### 3.2 后端对特化 kwarg 的三种反应

| 反应 | 机制 | 例子 | 后果 |
|---|---|---|---|
| **A. 认得** → 用原生参数实现 | 签名里显式声明 | FA 的 `softcap`（`flash_attention.py:35`） | ✅ 正确 |
| **B. 不认识** → 落进 `**kwargs` 静默丢弃 | 签名只有 `**kwargs` | sdpa 的 `softcap`（`sdpa_attention.py:79-95` 里没有它） | ❌ **算错，不报错** |
| **C. 作者提前拦死** | 准入层声明 | `_supports_sdpa = False` | ⛔ 报错，用不了（但安全） |

**同一个 kwarg，三种结局都可能出现**——这就是为什么不能一刀切地说"特化模型不能用标准后端"。

### 3.3 FA 的特殊之处：能力是**运行时反射**出来的

FA 面对**一堆不同版本、不同来源的内核**（FA2 / FA3 / FA4 / 各家 hub kernel），没法写死能力表，于是改成**反射探测**（`modeling_flash_attention_utils.py:224-245`）：

```python
flash_parameters = inspect.signature(flash_function).parameters
process_parameters = inspect.signature(_process_flash_attention_kwargs).parameters
supports_mapping = {p: p in flash_parameters for p in process_parameters}   # p 先经 _hf_api_to_flash_mapping 改名
return partial(_process_flash_attention_kwargs, supports_mapping=supports_mapping)
```

然后按这张表决定塞什么（`modeling_flash_attention_utils.py:650-668`）：

```python
if supports_mapping["window_size"] and sliding_window is not None and key_length > sliding_window:
    flash_kwargs["window_size"] = (sliding_window - 1, sliding_window - 1)
if supports_mapping["softcap"] and softcap is not None: flash_kwargs["softcap"] = softcap
if ((legacy_sink_param := supports_mapping["s_aux"]) or supports_mapping["learnable_sink"]) and s_aux is not None:
    flash_kwargs["s_aux" if legacy_sink_param else "learnable_sink"] = s_aux   # FA3(vllm) / FA4
```

**三个必须注意的点**：① **名字映射表**（`modeling_flash_attention_utils.py:135-140`）：`_hf_api_to_flash_mapping = {"dropout": "dropout_p", "sliding_window": "window_size"}`、`_flash_api_alternative_names = {"s_aux": "learnable_sink"}`；② `s_aux`/`learnable_sink` 那行是「同一语义、两个名字」的转接头（FA2/3/4 共用一个 Python 函数的关键零件）；③ **它依然是「静默不传」**——`supports_mapping[...]` 为假时**不报错**，只是不传（与 sdpa 的丢弃同源）→ 准入层白名单不是装饰。
> **为什么探测只做一次**：`_lazy_define_process_function` docstring 明说——「Due to limitations in `torch.compile`, we opt to statically type which (optional) kwarg parameters are supported」；结果缓存成 `_process_flash_kwargs_fn`（每次 forward 都反射的话 torch.compile 会放弃 fullgraph）。

### 3.4 一条容易被忽略的边界：kwarg 该谁用

同一个 kwarg 在不同后端可能是「原生参数」也可能是「靠 mask 实现」，这时**它有两条路径**：`sliding_window` —— FA 用原生 `window_size=(sw-1, sw-1)`，eager / sdpa / flex 靠 ① 的 mask（`create_sliding_window_causal_mask`）；`position_bias` —— 只能走 mask 路径（sdpa 用 `create_position_bias_mask` 合成浮点 mask，`sdpa_attention.py:39`、调用点 `:155`；flex 用 `score_mod`）；`is_causal` —— sdpa / flash 是内核参数，没有 mask 路径。

**判别方法**：看模型有没有同时**改 mask 类型**。Mistral 两边都做了（`modeling_mistral.py:364` 换 mask 函数 + `:172` 传 kwarg），所以它在任何后端下窗口都成立；而 Gemma 2 的 `softcap` **只有 kwarg 一条路径**，后端不认就彻底丢失。

---

## 四、第三层：准入层——作者怎么预先声明能力

### 4.1 三个开关，默认全是 `False`

```python
# modeling_utils.py:1237-1239（PreTrainedModel 的类属性）
_supports_sdpa: bool = False ; _supports_flash_attn: bool = False ; _supports_flex_attn: bool = False
```
模型侧显式打开（Llama 三个全开，`modeling_llama.py:334-336`）。

### 4.2 三处校验点

`_supports_sdpa` → `_sdpa_can_dispatch`（只查标志位，`modeling_utils.py:1740-1758`）；`_supports_flash_attn` → `_flash_attn_can_dispatch`（标志位 + GPU 算力 + dtype，`modeling_utils.py:1665`）；`_supports_flex_attn` → `_flex_attn_can_dispatch`（标志位 + torch 版本，`modeling_utils.py:1771`）。
`_sdpa_can_dispatch` 的报错文案本身就是逃生指南（`modeling_utils.py:1751-1756`）：「... does not support an attention implementation through torch.nn.functional.scaled_dot_product_attention yet. ... load your model with the argument `attn_implementation="eager"` meanwhile.」

### 4.3 更细的一层：`_compatible_flash_implementations`

光有 `_supports_flash_attn = True` 还不够——有些特性只在**部分内核**里实现，于是再加一道白名单（GPT-OSS 的 sink 就是这种情况）：`modeling_gpt_oss.py:399-403` 写着 `_compatible_flash_implementations = ["kernels-community/vllm-flash-attn3", "flash_attention_4", "kernels-community/metal-flash-sdpa"]`。
请求不在名单里的 FA 版本会被**自动改写**（`modeling_utils.py:1829-1844`）：`logger.warning_once` 后把 `attn_implementation` 换成 `compatible_flash_implementations[0]`（paged 时加 `paged|` 前缀）。实测 `grep -rn "_compatible_flash_implementations" src/transformers/models/*/modeling_*.py` 命中 **6 个文件**：`gpt_oss`（vllm-flash-attn3 / flash_attention_4 / metal-flash-sdpa）、`granite_swa`、`granitemoe_swa`（各 vllm-flash-attn3 / flash_attention_4）、`mimo_v2_flash`（flash_attention_4）、`minimax_m3_vl`（`MiniMaxAI/msa`）、`openai_privacy_filter`（见文件 `:410`）。

### 4.4 准入层的"缺席"——它管不到的地方

准入层是**作者写的**：作者没想到的组合 → 没有声明 → 静默出错（Gemma 2 + sdpa 就是这类）；静态分析也帮不上忙（`_supports_sdpa` 与「有没有传 sdpa 处理不了的 kwarg」之间**没有自动关联**）。**这是整套设计里唯一没有自动防线的接缝。**

---

## 五、`eager` 到底什么时候出场

`eager` 有**两条进入路径**：显式点名，或系统替你选。共 5 种情况。

### 5.1 显式（3 种）

| # | 触发方式 | 机制 | 出处 |
|---|---|---|---|
| 1 | `attn_implementation="eager"` | 用户直接点名（`from_pretrained` / `set_attn_implementation` / 构造 config） | `modeling_utils.py:2041` |
| 2 | `output_attentions=True` | config setter **自动**把实现设成 eager | `configuration_utils.py:407-414` |
| 3 | 库内工具强制 | Whisper 的 `return_token_timestamps`、注意力可视化工具 | `generation_whisper.py:705-707`、`attention_visualizer.py:211` |

第 2 条有**顺序依赖**（`configuration_utils.py:407-414`）：

```python
def output_attentions(self, value):   # @output_attentions.setter
    if value and self._attn_implementation is None: self._attn_implementation = "eager"   # ← 只在"还没定"时顺手设置
    if value and self._attn_implementation != "eager": raise ValueError(...)               # ← 已经是 sdpa 就直接抛错
```
所以「先 `attn_implementation="sdpa"` 再 `output_attentions=True`」会炸，反过来不会。

### 5.2 自动（2 种）

**④ 默认不指定 → 先试 sdpa → 失败则降级 eager**（`modeling_utils.py:1926-1960`）：

```python
applicable_attention = "sdpa" if requested_attention is None else requested_attention
elif "sdpa" in applicable_attention:   # Sdpa is the default, so we try it and fallback to eager otherwise when not possible
    try: self._sdpa_can_dispatch(is_init_check)
    except (ValueError, ImportError) as e:
        if requested_attention is not None and "sdpa" in requested_attention: raise e   # 你点名要 sdpa → 报错
        applicable_attention = "eager"                                                   # 你没表态 → 静默降级（无日志！）
```
**这个不对称被测试固化**（`tests/utils/test_modeling_utils.py:2971-2982`，用例当前被 `@unittest.skip("Just a bit annoying")` 跳过，但语义仍是权威说明）：`test_error_no_sdpa_available` —— 显式 `attn_implementation="sdpa"` → `assertRaises(ValueError)`；不传 → 不抛。
**⑤ `config._attn_implementation is None`**（`modeling_utils.py:5171-5176`）：`get_interface` 会 `warning_once` 后回退 default。正常路径下 `PreTrainedModel.__init__` 已把名字落定（`:1345-1350`），所以剩下的是**脱离模型单独用 attention 模块**的场景（`attn = LlamaAttention(LlamaConfig(), layer_idx=0)`）。

### 5.3 三个**不会**走 eager 的反例

要 FA 但没装 `flash_attn` → 报错，或从 Hub 拉内核（`modeling_utils.py:1862-1880`）——**不降级**；要注册表里不存在的名字 → `KeyError`（`modeling_utils.py:5177-5180`）——只有 `"eager"` 和 `None` 被特赦；后端把模型私有数学丢了 → **静默算错，不降级**。

---

## 六、特化的三种结局：谁能换、谁不能换

### 6.1 案例 A：能换 —— Gemma 2 的 soft-cap + FA

| 后端 | soft-cap 是否生效 | 为什么 |
|---|---|---|
| `eager` | ✅ | Gemma 2 自带实现真做了 `x/c → tanh → *c`（`modeling_gemma2.py:203-206`） |
| `flash_attention_2/3/4` | ✅ | 签名有 `softcap`（`flash_attention.py:35`）；FA 侧还有 `supports_mapping` 二次确认 |
| `flex_attention` | ✅ | 在 `score_mod` 里做（`flex_attention.py:292`） |
| `sdpa`（**默认**） | ❌ **静默丢弃** | 签名里没有 `softcap`（`grep -c softcap src/transformers/integrations/sdpa_attention.py` → **0**） |
| 三个 `*_paged` | ❌ | 同 sdpa |

**结论：特化 ≠ 不能换后端**——决定因素是「后端有没有对应的表达方式」。

### 6.2 案例 B：不能用 —— attention sink + sdpa

GPT-OSS 的 attention 有个可学习 sink：把 sinks 拼成**额外一列**再 softmax（`modeling_gpt_oss.py:251-252`，`combined_logits = torch.cat([attn_weights, sinks], dim=-1)`）。
**这个数学没法用 sdpa 的现有参数表达**（sdpa 没有"多加一列"的接口），所以作者直接把 sdpa 堵死（`modeling_gpt_oss.py:386-388`）：`_supports_flash_attn = True`、`_supports_sdpa = False`、`_supports_flex_attn = True`。FA 侧则因 `s_aux` 只有部分内核支持，再加白名单（见 4.3）。**同一个 sink 特性投影出三件事**：禁 sdpa + FA 限内核 + flex 可用（flex 在算完 LSE 后重归一化）。

### 6.3 特性 × 后端矩阵

| 特性 | eager | sdpa | flash_attention_* | flex | 依据 |
|---|---|---|---|---|---|
| soft-cap | ✅ 参考实现 | ❌ 静默丢 | ✅ 原生参数 | ✅ `score_mod` | `modeling_gemma2.py:203`、`flash_attention.py:35` |
| attention sink | ✅ | ❌（已被作者禁） | ⚠️ 视内核，名字可能是 `learnable_sink` | ✅ | `modeling_gpt_oss.py:251`、`modeling_flash_attention_utils.py:665` |
| sliding window | ✅ 靠 mask | ✅ 靠 mask | ✅ 原生 `window_size` | ✅ `BlockMask` | `modeling_flash_attention_utils.py:650` |
| `position_bias`（T5 式） | ❌ | ✅ 合成浮点 mask | ❌ | ✅ `score_mod` | `sdpa_attention.py:155` |
| `v_head_dim ≠ qk_head_dim` | ✅ | ✅ | ⚠️ pad V 再 crop | ✅ | `flash_attention.py:62-64` |
| 返回真实权重 | ✅ | ❌ | ❌ | ⚠️ 返回 `lse` | `flex_attention.py:368` |
| dropout | ✅ | ✅ | ✅ | ❌ **>0 直接抛错** | `flex_attention.py:274-278` |

**读法**：❌ = 静默丢弃（危险）；⚠️ = 条件可用；⛔ = 已被准入层堵死（安全）。

---

## 七、实测盘点：谁禁了 sdpa，以及一个反直觉的默认值

### 7.1 15 个显式 `_supports_sdpa = False` 的模型

`grep -rln "_supports_sdpa = False" src/transformers/models/*/modeling_*.py | sed 's|.*/models/||;s|/modeling_.*||' | sort` 得到这 15 个（两者都统一禁 sdpa，区别在原因）：

- `deepseek_v4`（**三个开关全 False**，见 7.3）；
- **sink 家族 5 个**：`gpt_oss`、`granite_swa`、`granitemoe_swa`、`mimo_v2_flash`、`openai_privacy_filter`；
- **其余 9 个**（历史/结构原因）：`nllb_moe`、`esm`、`got_ocr2`（**无 `get_interface`**，压根没接 functional 接口）、`pegasus_x`、`pp_doclayout_v2`、`speech_to_text`、`time_series_transformer`、`videoprism`、`wavlm`（**无 `get_interface`**）。

**sink 家族 5 个成员是同一个原因**——实测 `self.sinks` + `s_aux=` 同时出现的正是这 5 个（`grep -c 'self.sinks'` → 2 = 定义 + 初始化、`grep -c 's_aux='` → 1）。

> **`pp_doclayout_v2` 是个特例**：它的基类是 `True`（`:730`），只有子类 `PPDocLayoutV2ReadingOrder` 覆盖成 `False`（`:810`）——属于「同一文件内按子模型分别声明」。

### 7.2 反直觉：`_supports_sdpa` 的默认值是 `False`

```bash
grep -rl "_supports_sdpa = True"  src/transformers/models/*/modeling_*.py | wc -l   # 340
grep -rl "_supports_sdpa = False" src/transformers/models/*/modeling_*.py | wc -l   # 15
# 总 modeling_*.py 文件数：495  → 还有 141 个没提这件事（继承默认 False）
```
（340 + 15 + 141 = 496 > 495，因为 `pp_doclayout_v2` **同时**出现在 True 和 False 两个集合里——「开关」的粒度可以细到**同一文件里的不同子模型**。）
这 141 个里大部分是非注意力模型（ConvNeXt、GLPN 之类），**但有 13 个走了 functional 接口却没声明**：`align / bigbird_pegasus / bridgetower / clap / decision_transformer / git / informer / layoutlm / longt5 / markuplm / patchtsmixer / splinter / switch_transformers`。

**推论（静态代码路径推导，未实跑——本机无 torch）**：这些模型在**什么都不传**时，`get_correct_attn_implementation` 走 `_sdpa_can_dispatch` → 抛 `ValueError` → 被吞掉 → `applicable_attention = "eager"`（`modeling_utils.py:1953-1960`）。**静默，无日志。**
> ⚠️ 只做了代码路径推导 + 静态扫描，没有实跑。复现：`git blame` 看 `_supports_sdpa = True` 是不是漏加，或在有 torch 的机器上打印 `model.config._attn_implementation`。**不确定的地方不要当成结论用。**
### 7.3 极端案例：`deepseek_v4` 三个开关全 False

```python
# modeling_deepseek_v4.py:1170-1172
_supports_flash_attn = False ; _supports_sdpa = False ; _supports_flex_attn = False
```
它确实接了 functional 接口（`get_interface` 在 `:837-838`，自带 `eager_attention_forward` 在 `:708`），但三个开关全关 → **实际只剩 eager**。这是「能不能换后端」由**作者声明**而非**接口存在**决定的极端例证。

---

## 八、一句话总结

> **注册表保证"能被调用"，准入层保证"不会算错"，中间那层（`**kwargs`）什么保证都没有。**
> 第一层（硬契约）：5 个位置参数 + 2-tuple + `[B,H,S,D]`→`[B,S,H,D]`——**注册表管这层**；第二层（协商层）：模型用关键字名喊出私有数学，后端按名字认领——**认不出就静默丢，所有坑在这**；第三层（准入层）：`_supports_*` + `_compatible_flash_implementations`，作者预判能力边界——**报错是安全的**。
> **判断一个模型能不能换后端**，不要问"它有没有特化 eager"，要问"**我要的那个后端，认不认它传的每一个 kwarg**"。

---

## 验证清单

```bash
# 环境说明：本机**未安装 torch/transformers**（`python3 -c "import transformers"` 失败），
# 所以本篇全部结论来自 grep / sed / AST 静态扫描，**没有任何一条是实跑模型得到的**。
# 唯一一条纯代码路径推导（7.2 的 13 个模型）已在正文里标注。

cd /path/to/transformers

# ── 1. 硬契约：调用点与收尾 ──
grep -n "get_interface" -A 14 src/transformers/models/llama/modeling_llama.py
#   264:        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
#   268-277:  调用，5 个位置参数 + dropout/scaling 关键字
#   279-281:  attn_output.reshape(*input_shape, -1) → o_proj
grep -n "input_shape = \|hidden_shape = " src/transformers/models/llama/modeling_llama.py
#   251: input_shape = hidden_states.shape[:-1]      → [B, S]
#   252: hidden_shape = (*input_shape, -1, self.head_dim)

# ── 2. 返回值 2-tuple，第二项只有 eager 是真的 ──
grep -n "return attn_output, None\|return attn_output, attn_weights\|return attention_output, lse" \
  src/transformers/integrations/{sdpa_attention,flash_attention,flex_attention,eager_paged}.py \
  src/transformers/models/llama/modeling_llama.py
#   sdpa_attention.py:170:    return attn_output, None
#   flash_attention.py:102:   return attn_output, None
#   flex_attention.py:368:    return attention_output, lse
#   eager_paged.py:80:        return attn_output, attn_weights
#   modeling_llama.py:213:    return attn_output, attn_weights
#   modeling_llama.py:281:    return attn_output, attn_weights   ← forward 的返回（非后端）

# ── 3. FAQ 的布局转换责任在后端 ──
sed -n '55,58p' src/transformers/integrations/flash_attention.py
#   # FA2 uses non-transposed inputs
#   query = query.transpose(1, 2) ...
sed -n '62,64p;100p' src/transformers/integrations/flash_attention.py
#   value = torch.nn.functional.pad(value, [0, head_dim - v_head_dim])   ← pad V
#   attn_output = attn_output[..., :v_head_dim]                          ← crop 回来

# ── 4. 模型私有数学靠 kwargs 传（一模型一处）──
grep -n "softcap=" src/transformers/models/gemma2/modeling_gemma2.py
#   282:            softcap=self.attn_logit_softcapping,
grep -n "s_aux=self.sinks" src/transformers/models/gpt_oss/modeling_gpt_oss.py
#   329:            s_aux=self.sinks,  # diff with Llama

# ── 5. sdpa 静默丢弃 softcap（零命中是关键证据）──
grep -c softcap src/transformers/integrations/sdpa_attention.py        # → 0
grep -c softcap src/transformers/integrations/flash_attention.py       # → 2（签名 + 传参）

# ── 6. FA 的运行时能力探测与名字映射 ──
sed -n '224,245p' src/transformers/modeling_flash_attention_utils.py   # inspect.signature → supports_mapping
sed -n '135,141p' src/transformers/modeling_flash_attention_utils.py   # 映射表
#   _hf_api_to_flash_mapping = {"dropout": "dropout_p", "sliding_window": "window_size"}
#   _flash_api_alternative_names = {"s_aux": "learnable_sink"}
sed -n '662,668p' src/transformers/modeling_flash_attention_utils.py   # legacy_sink_param 分支

# ── 7. 三个开关默认全 False ──
sed -n '1237,1239p' src/transformers/modeling_utils.py
#   _supports_sdpa: bool = False
#   _supports_flash_attn: bool = False
#   _supports_flex_attn: bool = False

# ── 8. sdpa → eager 的静默降级（注意 except 里没有 log）──
sed -n '1953,1960p' src/transformers/modeling_utils.py
#   elif "sdpa" in applicable_attention:
#       # Sdpa is the default, so we try it and fallback to eager otherwise when not possible
#       try: self._sdpa_can_dispatch(is_init_check)
#       except (ValueError, ImportError) as e:
#           if requested_attention is not None and "sdpa" in requested_attention: raise e
#           applicable_attention = "eager"

# ── 9. 这个不对称有测试固化（虽然被 skip）──
sed -n '2971,2982p' tests/utils/test_modeling_utils.py
#   test_error_no_sdpa_available：显式 sdpa → assertRaises；不传 → 不抛

# ── 10. output_attentions 的顺序依赖 ──
sed -n '406,415p' src/transformers/configuration_utils.py
#   if value and self._attn_implementation is None: self._attn_implementation = "eager"
#   if value and self._attn_implementation != "eager": raise ValueError(...)

# ── 11. 库内强制 eager 的两处 ──
grep -n '_attn_implementation = "eager"' src/transformers/models/whisper/generation_whisper.py \
  src/transformers/utils/attention_visualizer.py
#   generation_whisper.py:707 / attention_visualizer.py:211

# ── 12. 准入层白名单 ──
grep -rn "_compatible_flash_implementations" src/transformers/models/*/modeling_*.py | wc -l   # 6 个文件
sed -n '399,403p' src/transformers/models/gpt_oss/modeling_gpt_oss.py
sed -n '1829,1844p' src/transformers/modeling_utils.py    # 自动改写的实现

# ── 13. 15 个禁 sdpa 的模型 + sink 家族 ──
grep -rln "_supports_sdpa = False" src/transformers/models/*/modeling_*.py | wc -l            # 15
grep -c "self.sinks" src/transformers/models/gpt_oss/modeling_gpt_oss.py                        # 2
sed -n '251,252p' src/transformers/models/gpt_oss/modeling_gpt_oss.py                          # cat([attn, sinks])
sed -n '386,388p' src/transformers/models/gpt_oss/modeling_gpt_oss.py                          # 三开关

# ── 14. 默认值分布 + 13 个"没声明却接了 functional 接口"的模型 ──
grep -rl "_supports_sdpa = True"  src/transformers/models/*/modeling_*.py | wc -l              # 340
grep -rl "_supports_sdpa = False" src/transformers/models/*/modeling_*.py | wc -l              # 15
grep -n "_supports_sdpa" src/transformers/models/pp_doclayout_v2/modeling_pp_doclayout_v2.py
#   730: _supports_sdpa = True    ← 基类
#   810: _supports_sdpa = False   ← 子类覆盖（所以 340+15+141 会比 495 多 1）
python3 - <<'EOF'
import glob
print(sorted(f.split('/')[3] for f in glob.glob('src/transformers/models/*/modeling_*.py')
             if 'get_interface' in open(f).read() and '_supports_sdpa' not in open(f).read()))
EOF
# ['align','bigbird_pegasus','bridgetower','clap','decision_transformer','git','informer',
#  'layoutlm','longt5','markuplm','patchtsmixer','splinter','switch_transformers']

# ── 15. eager 是"316 份"而不是一份 ──
grep -rl "^def eager_attention_forward" src/transformers/models/*/modeling_*.py | wc -l        # 316

# ── 16. 极端案例：三个开关全 False ──
sed -n '1170,1172p' src/transformers/models/deepseek_v4/modeling_deepseek_v4.py
grep -n "get_interface\|^def eager_attention_forward" src/transformers/models/deepseek_v4/modeling_deepseek_v4.py
#   708: def eager_attention_forward
#   837: get_interface(...)      ← 接口在，但三个开关全关 → 实际只剩 eager
```

**未验证的部分（明确声明）**

- 7.2 的「13 个模型静默走 eager」是**代码路径推导**，没有实跑模型确认（环境缺 torch）。
- `_supports_sdpa` 默认 `False` 的实际影响面（141 个未声明文件里有多少真的会走到 attention）
  只做了文件级统计，没有逐个确认。

---

## 导航

- 上一篇：[2A6 权重加载、量化与并行](./2A6_weights_quant_parallel.md)
- 直接前置：[2A1 注意力后端机制](./2A1_attention_backends.md)（**建议先读**）
- 相关：[2A2 Mask 系统](./2A2_masking_system.md)、[2M1 Gemma 2](./2M1_gemma2.md)（soft-cap 案例）
- 返回：[00_roadmap.md](./00_roadmap.md) ｜ [index.html](./index.html)
