# Qwen3（dense）模型结构精读

> 阅读目标：把 Llama（2023）到 Qwen3（2025）这两年间，dense decoder 发生的**「小但重要」的演进**一次收齐。
> 这些改动单看都不大，但每一处都改在「训练稳定性」和「结构自由度」上：
> **QK-Norm**、**`head_dim` 与 `hidden_size` 解耦**、**RoPE 参数体系改名**、**`tie_word_embeddings` 的真实默认值**。
>
> 读完这篇你应该能回答：为什么 Qwen3 的 DecoderLayer 骨架和 Llama 一模一样，
> 但它的 Attention 里多了两个 128 维的 RMSNorm，而且这两个 norm 卡在一个很讲究的位置上。
>
> 源文件：`src/transformers/models/qwen3/modular_qwen3.py`（**161 行**，有 modular，只读它）
> 生成文件：`src/transformers/models/qwen3/modeling_qwen3.py`（**529 行**，`make fix-repo` 生成，勿直接编辑）
> 手写文件：`src/transformers/models/qwen3/configuration_qwen3.py`（**101 行**，**不在 modular 里**，它不是生成物）
>
> ⚠️ 本篇是 **`qwen3`（dense）**。MoE 版是另一个模型目录（`models/qwen3_moe/`），在路线图里是 2M3，别混。

---

## 速读区（TL;DR）

**一句话结论**：Qwen3 dense 没有新概念——它是 **Llama 骨架 + 三个稳定性补丁**（QK-Norm、`head_dim` 解耦、`rope_parameters` 改名体系）。
DecoderLayer 与 Llama **逐行相同**，所以说「读 Qwen3 的成本 ≈ 读 Llama + 读第一节那张演进表」。

### 必须记住的断言

| # | 断言 | 出处（文件:行号） |
|---|------|------------------|
| 1 | `Qwen3DecoderLayer` 与 `LlamaDecoderLayer` **逐行相同**（`diff` 空输出），连 `forward` 签名都一样 | `modeling_qwen3.py:283` vs `modeling_llama.py:284`（验证清单 `:607-609`） |
| 2 | QK-Norm **在 RoPE 之前**、且**逐头**（作用在 `view(hidden_shape)` 之后，`hidden_shape = (-1, head_dim)`），只 norm Q/K、不碰 V | `modeling_qwen3.py:252-253`（norm）→ `:257`（RoPE）；`modular_qwen3.py:79-80` |
| 3 | `Qwen3Attention.forward` 与 Llama 的差异**只有两处**：`q_norm`/`k_norm` 包住 q/k 投影 + 多传 `sliding_window=self.sliding_window` | 第七节表（`:495`）、验证清单 `:603-605` |
| 4 | `head_dim: int = 128` 是**显式常量**；Llama 是 `int \| None = None` 再在 `__post_init__` 里推导，Qwen2 **完全没有这个字段** | `configuration_qwen3.py:68` / `configuration_llama.py:84`、`:87-88` |
| 5 | ⚠️ **本仓库里 `Qwen3Config` 上没有 `rope_scaling`、也没有 `rope_theta` 字段**，只有 `rope_parameters`；`rope_scaling` 已降级成 **property 别名**（旧名仍能用） | `configuration_qwen3.py:75`、`configuration_utils.py:548-554` |
| 6 | `rope_theta` 的**代码默认值是 `10000.0`**（不是常被写的 1000000）；Qwen3 dense **不覆盖** `default_theta` | `modeling_rope_utils.py:739`、`:752`；验证清单 `:637` |
| 7 | Qwen3 的 `rope_type` 就是 **`"default"`**（不带 scaling），走 `compute_default_rope_parameters`；`"default"` **不在** `ROPE_INIT_FUNCTIONS` 那张表里 | `modeling_qwen3.py:95-99`、`:116-121`；`modeling_rope_utils.py:668-675` |
| 8 | `tie_word_embeddings = False`（与 Llama 同）；`_tied_weights_keys` **只是一份声明**，为 `False` 时完全不生效，`lm_head` 独立占 622.3M 参数 | `configuration_qwen3.py:74`、`modeling_qwen3.py:432`、`modeling_utils.py:2555-2560` |
| 9 | `Qwen3MLP` 继承 **`GemmaMLP`**（与 `Qwen2MLP` 逐字节相同），且 **bias 硬编码 `False`**（Llama 是 `bias=config.mlp_bias`，Qwen3Config 也没有 `mlp_bias` 字段） | `modular_qwen3.py:52`、`modeling_qwen3.py:76-78`、第七节表（`:491`） |
| 10 | 滑动窗口**默认不激活**：`use_sliding_window=False` → `sliding_window=None` → `layer_types` 全是 `"full_attention"` → `has_sliding_layers=False` | `configuration_qwen3.py:86-97`、`modeling_qwen3.py:359`、`:392-407` |
| 11 | `Qwen3Model` 的 mask 是**字典 `causal_mask_mapping`**，逐层按 `layer_types[i]` 取（Llama 只造一个 `causal_mask`） | `modeling_qwen3.py:392-407`、`:415` |

### 最容易踩的坑 / 反直觉发现

- **「norm 在 RoPE 之前还是之后」是最容易被讲错的一点**（正文自己这么说）：代码顺序就是证据——`q_norm` 在 `:252`，`apply_rotary_pos_emb` 在 `:257`。
- **「先 norm 再旋转」不是数学必需**：旋转不改变向量范数，放后面效果一样，只是多一次 reshape / 计算，所以这个顺序是「更省事」，不是「更强」。
- **`getattr(config, "head_dim", ...)` 的兜底是 Llama 时代就有的通用写法，Qwen3 没改这段代码**——解耦发生在 **config 层**，下游因此永远走不到兜底分支。
- **只看默认参数会误以为没解耦**：`4096 / 32 = 128 = head_dim` 恰好相等；要理解这条改动得看「它允许什么」（对照 Qwen3-Next：`2048/16=128` 但 `head_dim=256`）。
- **`rope_theta=1000000` 来自 checkpoint 的 `config.json`，不是本仓库代码的默认值**——想用 1e6 必须在模型里显式覆盖（仓库里 72 个文件这么做了）。
- **`self.layer_type` 写在 `super().__init__()` 之前是 modular 的写法**：converter 会强制把 `super().__init__()` 挪到最前，生成文件里顺序被翻过来（`:215` / `:216`）。
- **`noqa: F822` 是「自动拉取」的信号**：`Qwen3Model` / `Qwen3DecoderLayer` 在 modular 里查无此名，但生成文件里有。

### 读完应该能回答的问题

- Qwen3 的 QK-Norm 归一化在哪个维度上、卡在 RoPE 的哪一侧？为什么 OLMo2 必须在 `view` **之前**做而 Qwen3 不需要 reshape？
- 为什么说 `head_dim` 解耦的收益要在 `qwen3_next`（`2048/16=128 ≠ 256`）上才看得出来？
- 本仓库里 `Qwen3Config()` 的 `rope_theta` 到底是多少？「1e6」这个数字出自哪里？
- `tie_word_embeddings=False` 时 `_tied_weights_keys` 还有作用吗？不 tie 的 `lm_head` 值多少参数？
- 为什么 `q_norm` / `k_norm` 在 `base_model_tp_plan` 里是 `replicated_with_grad_allreduce` 而不是切分？

---

## 前置知识

- [1M1_llama_structure.md](./1M1_llama_structure.md)：**必备**。`Qwen3Attention` 就是继承 `LlamaAttention` 来的，
  RMSNorm / RoPE / SwiGLU / GQA / Pre-Norm 残差全部复用，本篇只讲「改了什么」。
- [1M2_mistral_sliding_window.md](./1M2_mistral_sliding_window.md)：读 modular 文件的方法（只读差异，不读全貌）。
- [2A5_modular_code_generation.md](./2A5_modular_code_generation.md)：**本篇会反复引用它的三条规则**——
  「`pass` = 原样继承」「不加装饰器 = 继承父类装饰器」「`**super_kwargs` = 只换 docstring」。
  不读 2A5 的话，你会以为 `modular_qwen3.py` 里没写的东西是漏了。
- 2A4 RoPE 抽象层（待写）：`rope_type` 各变体的数学细节。本篇只讲 Qwen3 **用了哪个**、默认值是多少。

---

## 一、本篇定位：两年的 dense 演进账本

Llama 2 发布于 2023，Qwen3 发布于 2025。中间注意力机制、MoE、SSM 都吵得很热闹，
但**纯 dense 的 decoder 结构本体几乎没有变化**：还是 Pre-Norm + RMSNorm + RoPE + SwiGLU + GQA。
真正落地的改动是下面这几条——本篇就是这份账本。

| # | 演进点 | Llama（2023） | Qwen3（2025） | 影响 |
|---|--------|---------------|---------------|------|
| 1 | **QK-Norm** | 无 | Q/K 逐头各一个 RMSNorm，**在 RoPE 之前** | 训练稳定性（防 attention logits 爆炸）|
| 2 | **`head_dim` 解耦** | `head_dim: int \| None = None` → 推导 | `head_dim: int = 128` 显式常量 | 结构自由度（不再要求整除）|
| 3 | **RoPE 参数体系** | `rope_scaling` + `rope_theta` 两个字段 | 统一进 `rope_parameters` 一个 dict | 接口整洁（旧名降级成 property 别名）|
| 4 | **layer_types 交替** | 无（纯 full attention）| 从 Qwen2 继承 `layer_types` + 滑动窗口 mask 映射 | 为 SWA 混合留了开关（Qwen3 dense 默认关闭）|
| 5 | **hub kernel 装饰器** | 有 | 有（继承来的）| RMSNorm / RoPE 可换成融合 kernel |
| 6 | `tie_word_embeddings` | `False` | `False` | **同**（但值得算一遍省多少参数）|

**一句话**：Qwen3 dense 没有新概念，它把 Llama 之后两年「攒下来的稳定化技巧」收进了同一个骨架里。
换句话说，**读 Qwen3 的成本 ≈ 读 Llama + 读这张表**。

---

## 二、modular 全景：161 行里谁写的、谁被自动拉进来

先看 `modular_qwen3.py` 的导入区（`:26-40`，节选）：

```python
from ..gemma.modeling_gemma import GemmaMLP
from ..llama.modeling_llama import (
    LlamaAttention,
)
from ..qwen2.modeling_qwen2 import (
    Qwen2ForCausalLM, ..., Qwen2RMSNorm, Qwen2RotaryEmbedding,
    apply_rotary_pos_emb, eager_attention_forward,
)
```

**注意这里的「三国拼装」**——同一个模型，四个类分别从三个不同模型目录拉：

| modular 里写的类 | 显式基类 | 为什么是这个基类 |
|------------------|----------|------------------|
| `Qwen3RMSNorm`（`:48`）| `Qwen2RMSNorm` | 与 `LlamaRMSNorm` 逐行相同，取近的 |
| `Qwen3MLP`（`:52`）| **`GemmaMLP`** | 实测与 `Qwen2MLP` / `GemmaMLP` 的函数体**逐字节相同**（`diff` 输出为空），选哪个都不影响数值 |
| `Qwen3RotaryEmbedding`（`:56`）| `Qwen2RotaryEmbedding` | 同上，代码相同 |
| `Qwen3Attention`（`:60`）| **`LlamaAttention`** | **有讲究，见下** |
| `Qwen3ForCausalLM`（`:110`）| `Qwen2ForCausalLM` | 只加一个 `_fsdp_plan` |
| `Qwen3For{SequenceClassification,TokenClassification,QuestionAnswering}`（`:142/:146/:150`）| `Qwen2For*` | 三个 `pass` |

### 为什么 Attention 偏偏选 `LlamaAttention` 而不是 `Qwen2Attention`？

这是本篇第一个「读 modular 才有用」的细节。实测 `Qwen2Attention` 与 `LlamaAttention` 的差异（`diff` 输出）：

```python
# ── Qwen2Attention（modeling_qwen2.py:189-192）──
self.q_proj = nn.Linear(..., bias=True)     # 硬编码 bias
self.k_proj = nn.Linear(..., bias=True)
self.v_proj = nn.Linear(..., bias=True)

# ── LlamaAttention（modeling_llama.py:230-241）──
self.q_proj = nn.Linear(..., bias=config.attention_bias)   # 由 config 决定
```

Qwen3 希望 attention 投影的 bias **可配**（`Qwen3Config.attention_bias: bool = False`，`configuration_qwen3.py:76`），
所以从 Llama 拿 `__init__`；但 forward 又必须带滑动窗口参数（Qwen2 的 forward 里有），
于是 **forward 整段重写在 modular 里**（`:68-107`）。
生成文件里 `Qwen3Attention` 的投影层确实是 `bias=config.attention_bias`（`modeling_qwen3.py:225-236`）——
**继承链和最终代码是一致的，不是抄错**。

### 没写在 modular 里的类 = 自动拉取

`modular_qwen3.py` 只有 8 个类（其中 6 个是纯 `pass`，只有 `Qwen3Attention` 和 `Qwen3ForCausalLM` 有真代码），
生成的 `modeling_qwen3.py` 有 **11 个类 + 4 个函数**。多出来的：

| 生成文件里的类 | 行号 | 来源 |
|----------------|------|------|
| `Qwen3PreTrainedModel` | `:327` | 自动拉取（来自 `Qwen2PreTrainedModel`，仅改名）|
| `Qwen3Model` | `:346` | **自动拉取（来自 `Qwen2Model`）——本篇有实质差异，见第七节** |
| `Qwen3DecoderLayer` | `:283` | 自动拉取（来自 `Qwen2DecoderLayer`，实测与 `LlamaDecoderLayer` 逐行相同）|
| `rotate_half` / `repeat_kv` / `eager_attention_forward` | `:140/:173/:185` | 自动拉取（与 Llama 的对应函数实测相同）|

modular 里的 `__all__` 用一个技巧声明了它们（`:154-161`）：

```python
__all__ = [
    "Qwen3ForCausalLM",
    "Qwen3ForQuestionAnswering",
    "Qwen3PreTrainedModel",  # noqa: F822   ← modular 里没定义，生成文件里才有
    "Qwen3Model",  # noqa: F822
    ...
]
```

> **2A5 规则 4 的实例**：`Qwen3Model` 和 `Qwen3DecoderLayer` 在 modular 里查无此名，
> 但它们不是「漏读」，是 converter 从 Qwen2 那一支拉进来的。**看到 `noqa: F822` 就是自动拉取的信号。**

### `Qwen3Config` 是手写的

`modular_qwen3.py:40` 只有 `from .configuration_qwen3 import Qwen3Config`——**config 类不在 modular 里**。
判断方法就是看文件开头有没有那段大写警告：

```bash
head -1 src/transformers/models/qwen3/modeling_qwen3.py       # → 🚨🚨🚨 自动生成的警告
head -1 src/transformers/models/qwen3/configuration_qwen3.py  # → # Copyright 2024 ...（没有警告 ⇒ 手写）
```

**所以 `configuration_qwen3.py` 是纯手写文件**，`make fix-repo` 不会覆盖它。
本篇第三、四、五节的事实全部来自这个文件。

---

## 三、QK-Norm（核心改动 1）

### 代码

定义（`modular_qwen3.py:60-66`，生成物在 `modeling_qwen3.py:237-239`）：

```python
class Qwen3Attention(LlamaAttention):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        super().__init__(config, layer_idx)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None
```

前向（`modular_qwen3.py:79-80` = `modeling_qwen3.py:252-253`）：

```python
query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
```

### 关键细节：norm 在 RoPE **之前**

这是本篇最容易被讲错的一点，所以把它钉死。把 `Qwen3Attention.forward` 与 `LlamaAttention.forward`
做逐行 `diff`（去掉类名），**全篇只有两处语义差异**：

```diff
-        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
-        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
+        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
+        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
             ...
             scaling=self.scaling,
+            sliding_window=self.sliding_window,  # diff with Llama
             **kwargs,
```

而 RoPE 永远在它们后面（`modeling_qwen3.py:256-257`）：

```python
cos, sin = position_embeddings
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
```

所以执行顺序是：

```
q_proj ─→ view(head_dim) ─→ q_norm ─→ transpose ─→ apply_rotary_pos_emb ─→ attention
          └── 逐头、在 head_dim 上归一化        └── 旋转 ────────────────┘
```

三句话记住：

1. **norm 在 RoPE 之前**，不在之后。代码顺序就是证据：`q_norm` 在 `view` 的返回值上直接调用（`:252`），而 `apply_rotary_pos_emb` 在四行之后（`:257`）。
2. **norm 是逐头的**，因为它作用在 `.view(hidden_shape)` 之后，`hidden_shape = (*input_shape, -1, self.head_dim)`（`:250`）。
   这就是代码注释里那句 `unlike olmo, only on the head dim!` 的意思——对比 OLMo2（`modeling_olmo2.py:221-222`）：
   `Olmo2RMSNorm(config.num_attention_heads * self.head_dim, ...)`，它在整个 `num_heads × head_dim` 上归一化
   （因此必须在 `view` **之前**做，之后还得 reshape）；Qwen3 直接在 `head_dim` 上归一化，
   **因此不需要 reshape**（注释第二句 `thus post q_norm does not need reshape`）。
3. **只 norm Q、K，不碰 V**。V 不参与 logits 计算，norm 它对稳定性没帮助。

### 为什么能稳定训练

attention logits 是 `q·k / √head_dim`。没有 QK-Norm 时，`q`、`k` 的尺度由前面几十层的激活尺度决定，
训练中一旦激活变大，logits 就跟着爆，softmax 饱和成 one-hot（梯度趋零）甚至数值溢出。
QK-Clip、logit soft-cap 之类都是在事后打补丁。

RMSNorm 把每个头归一化到「RMS ≈ 1」，于是 `‖q‖ ≈ ‖k‖ ≈ √head_dim`，logits 的量级被硬夹在

```
|q·k| / √d  ≤  ‖q‖‖k‖ / √d  ≈  √d      （d = head_dim = 128 时 ≈ 11.3）
```

**一个与层数、激活尺度无关的常数**。而 norm 在 RoPE 之前也不吃亏：旋转不改变向量范数，
上界在旋转后照样成立。若反过来把 norm 放在 RoPE 之后，效果一样，但要多一次 reshape / 计算，
所以「先 norm 再旋转」是更省事的顺序。

**代价极小**：每层 `q_norm` + `k_norm` 共 `2 × head_dim = 256` 个参数，
而同一层 `q_proj` 有 `4096 × 4096 ≈ 16.7M` 个参数（`Qwen3Config` 默认值），占比约 0.0015%。
这也是它值得在 `base_model_tp_plan` 里单独声明的原因（`configuration_qwen3.py:49-50`）：

```python
"layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
"layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
```

> 张量并行时 `q_proj` 按列切（`colwise`），每个 rank 拿到一部分头；而 `q_norm` 的参数**每个头都要用**，
> 所以是 `replicated`（复制）+ 梯度 all-reduce，不是切分。这是 QK-Norm 存在的旁证。

---

## 四、`head_dim` 与 `hidden_size` 解耦（核心改动 2）

### 三方对比（实测）

| 模型 | config 里的 `head_dim` | 文件:行号 | 含义 |
|------|------------------------|-----------|------|
| Llama | `head_dim: int \| None = None` + `__post_init__` 里推导 | `configuration_llama.py:84`、`:87-88` | **可为 None**，缺省时 `hidden_size // num_attention_heads` |
| Qwen2 | **没有这个字段** | — | 完全没有解耦入口 |
| **Qwen3** | `head_dim: int = 128` | `configuration_qwen3.py:68` | **显式常量，永远不推导** |

Llama 的推导代码（`configuration_llama.py:87-88`）：

```python
def __post_init__(self, **kwargs):
    if self.head_dim is None:
        self.head_dim = self.hidden_size // self.num_attention_heads
```

Qwen3 的字段（`configuration_qwen3.py:62-68`）：

```python
vocab_size: int = 151936
hidden_size: int = 4096
intermediate_size: int = 22016
num_hidden_layers: int = 32
num_attention_heads: int = 32
num_key_value_heads: int | None = 32
head_dim: int = 128
```

### 消费点：用 `getattr` 兜底

Attention 里（`modeling_qwen3.py:219`，与 `modeling_llama.py:224` **逐字符相同**）：

```python
self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
```

RoPE 里（`modeling_qwen3.py:117`，与 `modeling_llama.py:104` 相同）：

```python
dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
```

**这里要说清楚**：`getattr` 的兜底逻辑是 Llama 时代就有的通用写法，Qwen3 没有改这段代码。
Qwen3 的解耦体现在 **config 层**——`head_dim` 从「可选的推导值」变成了「有默认值的显式字段」，
于是所有下游代码（`q_proj` 的输出维度、`q_norm` 的归一化维度、RoPE 的频率维度）
都无条件地跟着 config 走，永远走不到兜底分支。

### 动机：去掉整除约束，让 head_dim 独立可调

一旦 `head_dim` 是独立参数，投影矩阵的形状就变成：

```python
# modeling_qwen3.py:225-236
self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, ...)
self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, ...)
self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, ...)
```

注意 **`num_attention_heads * head_dim` 完全可以不等于 `hidden_size`**：
q_proj 的输入是 `hidden_size`，输出是 `heads × head_dim`，两者不再绑定。

仓库里就有现成的验证——同一家族的 **Qwen3-Next**（`configuration_qwen3_next.py`）：

| 字段 | 值 | 行号 |
|------|-----|------|
| `hidden_size` | 2048 | `:94` |
| `num_attention_heads` | 16 | `:97` |
| `head_dim` | **256** | `:108` |

`hidden_size // num_attention_heads = 2048 / 16 = 128`，但 `head_dim = 256` —— **差一倍**。
如果沿用 Llama 的推导，这个配置根本无法表达；而在 Qwen3 的写法里它就是一个字段。
（`qwen3_5` 同样如此，`configuration_qwen3_5.py:97` 也是 `head_dim: int = 256`。）

顺带说明 Qwen3 dense 自己的默认值为什么看不出差别：`4096 / 32 = 128 = head_dim`，
**默认配置里恰好相等**，所以只看默认参数会误以为没解耦。要理解这条改动，得看「它允许什么」，而不是「默认是多少」。

**收益**：`head_dim` 直接决定 QK-Norm 的归一化粒度、RoPE 的频率分布、以及单个头的表达能力。
解耦之后它可以独立于 `hidden_size` / `num_attention_heads` 调优——比如「头数不变、把每个头做宽」。

---

## 五、RoPE 参数体系：`rope_parameters` / `rope_theta` / `rope_scaling`（核心改动 3）

### 字段的真名已经变了

在**本仓库这个版本**里，`Qwen3Config` 上**没有 `rope_scaling`，也没有 `rope_theta`**，只有一个（`configuration_qwen3.py:75`）：

```python
rope_parameters: RopeParameters | dict | None = None
```

`rope_scaling` 仍然能用，但它只是**属性别名**（`configuration_utils.py:548-554`）：

```python
@property
def rope_scaling(self):
    return self.rope_parameters

@rope_scaling.setter
def rope_scaling(self, value):
    self.rope_parameters = value
```

而 `rope_theta` 作为**旧式顶层 kwarg** 也仍然被接住——配置初始化时会在 `PreTrainedConfig.__post_init__` 里统一归一化
（`configuration_utils.py:313-314` 调用 `convert_rope_params_to_dict`）。

### `rope_theta` 的真实默认值：**10000.0**

**不是网上常写的 1000000。** 那个数字来自 checkpoint 的 `config.json`，不是本仓库代码的默认值。
代码里的实际来源是 `RotaryEmbeddingConfigMixin`（`modeling_rope_utils.py`）：

```python
# modeling_rope_utils.py:739
class RotaryEmbeddingConfigMixin:
    default_theta = 10_000.0
```

取值优先级与默认填充（`modeling_rope_utils.py:742`、`:752`）：

```python
def convert_rope_params_to_dict(self, **kwargs):
    rope_scaling = kwargs.pop("rope_scaling", None)      # 旧名兼容
    self.rope_parameters = rope_scaling or self.rope_parameters
    self.rope_parameters = self.rope_parameters if self.rope_parameters is not None else {}

    # 优先级：rope_parameters 里的值 > kwargs > config 属性 > 默认值
    rope_theta = kwargs.pop("rope_theta", getattr(self, "rope_theta", self.default_theta))
    self.rope_parameters.setdefault("rope_theta", rope_theta)
```

**Qwen3Config 没有覆盖 `default_theta`**，所以 `Qwen3Config()` 的 `rope_theta` 就是 `10000.0`。
对照组：本仓库里有 **72 个文件**显式覆盖了它（`grep -rl "default_theta = " src/transformers/models/ | wc -l`），
例如同族的 `qwen3_omni_moe/configuration_qwen3_omni_moe.py:133`：

```python
default_theta = 1000000.0
```

**「谁显式覆盖了 1e6」这件事本身就是可查的**——Qwen3 dense 不在其中。
（想确认某个具体 checkpoint 用了多少，去看它 `config.json` 里的 `rope_theta`；那是权重侧的数据，不是本仓库的代码默认值。）

### `rope_type` 的可选值

由 `ROPE_INIT_FUNCTIONS` 决定（`modeling_rope_utils.py:668-675`）：

```python
ROPE_INIT_FUNCTIONS: dict[str, Callable[..., tuple["torch.Tensor", float]]] = {
    "linear": _compute_linear_scaling_rope_parameters,
    "dynamic": _compute_dynamic_ntk_parameters,
    "yarn": _compute_yarn_parameters,
    "longrope": _compute_longrope_parameters,
    "llama3": _compute_llama3_parameters,
    "proportional": _compute_proportional_rope_parameters,
}
```

`"default"` 不在这张表里——它的含义是「不做任何 scaling」，走每个模型自己的
`compute_default_rope_parameters`。对应的校验函数也在同文件：
`_validate_default_rope_parameters`（`:850`）、`_validate_linear_...`（`:859`）、`_validate_dynamic_...`（`:872`）、
`_validate_yarn_...`（`:885`）、`_validate_longrope_...`（`:937`）、`_validate_llama3_...`（`:987`）、
`_validate_proportional_...`（`:1030`）。

所以 `RopeParameters`（`modeling_rope_utils.py:678`）的合法取值集合是：
**`default` / `linear` / `dynamic` / `yarn` / `longrope` / `llama3` / `proportional`**，加上 `rope_theta`、
`factor`、`original_max_position_embeddings`、`partial_rotary_factor` 等参数。

### Qwen3 用了哪个？——`default`，也就是「不带 scaling 的原始 RoPE」

路径只有三步（`modeling_qwen3.py`，这段是自动拉取来的，代码与 Qwen2 一致）：

```python
# :95
self.rope_type = self.config.rope_parameters["rope_type"]
# :96-98
rope_init_fn: Callable = self.compute_default_rope_parameters
if self.rope_type != "default":
    rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
# :99
inv_freq, self.attention_scaling = rope_init_fn(self.config, device)
```

而 `compute_default_rope_parameters` 里读的正是我们上面确认的两个值：

```python
# modeling_qwen3.py:116-121
base = config.rope_parameters["rope_theta"]
dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

attention_factor = 1.0  # Unused in this type of RoPE
inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
```

**`dim` 那一行是第四节的收尾**：RoPE 的频率维度也跟着 `head_dim` 走，而不是 `hidden_size // heads`。
所以「head_dim 解耦」不是只影响投影形状，它同时改变了位置编码的频率表。

> 各 `rope_type` 的数学细节（NTK、YaRN 的分频段处理、longrope 的 short/long factor 等）留给 **2A4**。
> 本篇只需要记住：**Qwen3 的默认是 `{"rope_type": "default"}`，靠 `max_position_embeddings`（默认 32768）
> 决定预训练上下文长度，缩放靠 checkpoint 里改 `rope_theta` 或加 `rope_parameters` 实现。**

---

## 六、`tie_word_embeddings`：真实默认值与取舍

### 默认值：`False`（与 Llama 相同）

```python
# configuration_qwen3.py:74
tie_word_embeddings: bool = False
```

所以这一条是**同**，不是差异（`configuration_llama.py:79` 也是 `False`）。但它的机制值得单独看，
因为它决定了一个 622M 的大矩阵要不要存——而且**「声明 tie」和「真的 tie」是两件事**。

生成文件里 `Qwen3ForCausalLM` 声明了映射（`modeling_qwen3.py:432`）：

```python
_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
```

这**只是一份声明**。真正的开关在 `modeling_utils.py`：

```python
# modeling_utils.py:2555-2560
tied_mapping = self._tied_weights_keys
tie_word_embeddings = getattr(self.config, "tie_word_embeddings", False)
if not tie_word_embeddings:
    return {}
```

也就是说：**`tie_word_embeddings=False` 时，`_tied_weights_keys` 完全不生效，`lm_head.weight` 是一份独立参数。**
（这份声明是 converter 从 `Qwen2ForCausalLM` 原样带过来的，所有模块化的 CausalLM 都会带着它。）

### tie 与不 tie 差多少

默认 `vocab_size = 151936`、`hidden_size = 4096`，`lm_head = nn.Linear(hidden_size, vocab_size, bias=False)`：

| | 参数量 | 说明 |
|---|---|---|
| 不 tie（Qwen3 默认）| `151936 × 4096 ≈ 622.3M` | 独立的输出头 |
| tie | 0 | 复用 `embed_tokens.weight`（形状恰好也是 `vocab_size × hidden_size`）|

对 8B 级别的模型，tie 掉省约 **7.8%** 的参数（622M / ~8B）。省的不只是显存：
tie 会把「输入查表」和「输出投影」的梯度绑在一起，对**小模型 / 小词表**通常有正则化作用
（输入输出共享语义空间），对大模型则常带来训练不稳定或表达能力受限。

**Qwen3 的选择是不 tie**，与 Llama 一致：把 622M 参数留给输出侧，换更好的收敛与质量。
旁证是 `configuration_qwen3.py:44-55` 的 `base_model_tp_plan` / `base_model_pp_plan` 都单独给 `lm_head` 留了位置。

---

## 七、Llama → Qwen3 对照速查（逐模块）

下表全部由「`sed` 抠出两段代码 + `diff`」实测得出（类名差异用 `sed` 抹掉），不是凭印象。

| 模块 | Llama（`modeling_llama.py`）| Qwen3（`modeling_qwen3.py`）| 同/异 | 差异点 |
|------|------------------------------|------------------------------|-------|--------|
| `RMSNorm` | `LlamaRMSNorm` `:53` | `Qwen3RMSNorm` `:50` | **同** | 逐行相同（含 `@use_kernel_forward_from_hub("RMSNorm")`），只是继承链是 `Qwen2RMSNorm` |
| `MLP` | `LlamaMLP` `:163` | `Qwen3MLP` `:70` | **异（小）** | 形状/SwiGLU 相同；**bias 来源不同**：Llama `bias=config.mlp_bias`（`:169-171`），Qwen3 硬编码 `bias=False`（`:76-78`），Qwen3Config 也没有 `mlp_bias` 字段 |
| `RoPE`（类）| `LlamaRotaryEmbedding` `:73` | `Qwen3RotaryEmbedding` `:86` | **同** | 逻辑相同，读的都是 `config.rope_parameters` |
| `RoPE`（维度）| `dim = getattr(config,"head_dim",None) or hidden//heads` `:104` | 同一行 `:117` | **同** | 代码相同，但 Qwen3 的 config 保证 `head_dim` 永远存在 → 永不走兜底 |
| `Attention` 初始化 | `LlamaAttention` `:217` | `Qwen3Attention` `:211` | **异** | +`self.layer_type`（`:216`）、+`q_norm`/`k_norm`（`:237-238`）、+`self.sliding_window`（`:239`）|
| `Attention` 前向 | `:244-283` | `:241-279` | **异（仅 2 处）** | ① `q_norm`/`k_norm` 包住 q/k 投影（`:252-253`）；② 多传 `sliding_window=self.sliding_window`（`:274`）。**RoPE 位置不变，仍在 norm 之后** |
| `eager_attention_forward` / `repeat_kv` / `rotate_half` / `apply_rotary_pos_emb` | `:191/:179/:130/:138` | `:185/:173/:140/:148` | **同** | `diff` 输出为空 |
| `DecoderLayer` | `LlamaDecoderLayer` `:284` | `Qwen3DecoderLayer` `:283` | **同** | 逐行相同（Pre-Norm + 两条残差），连 `forward` 签名都一样 |
| `PreTrainedModel` | `LlamaPreTrainedModel` `:328` | `Qwen3PreTrainedModel` `:327` | **同** | 属性表一致（`_supports_flash_attn` / `_supports_sdpa` / `_supports_flex_attn` / `_can_record_outputs` …）|
| `Model` | `LlamaModel` `:347` | `Qwen3Model` `:346` | **异（中）** | ① `has_sliding_layers`（`:359`）；② mask 从「一个 `causal_mask`」变成 **`causal_mask_mapping` 字典**（`:392-407`）；③ 逐层按 `layer_types[i]` 取 mask（`:415`）；④ 返回值加了 `if use_cache else None`（`:426`）|
| `ForCausalLM` | `LlamaForCausalLM` `:421` | `Qwen3ForCausalLM` `:431` | **同** | 结构、`lm_head`、`_tied_weights_keys` 全同，**只有 docstring 里的 checkpoint 名不同** |
| config 关键字段 | `tie_word_embeddings=False` `:79`、`head_dim: int\|None=None` `:84` | `tie_word_embeddings=False` `:74`、`head_dim: int=128` `:68`、`rope_parameters` `:75` | **异** | 见第四、五、六节 |

### 唯一需要展开的一行：`Qwen3Model` 的 mask 映射

Llama 只造一个 mask（`modeling_llama.py:391-397`）：

```python
causal_mask = create_causal_mask(config=self.config, inputs_embeds=inputs_embeds, ...)
...
    attention_mask=causal_mask,
```

Qwen3 造一个**字典**，按层类型分发（`modeling_qwen3.py:392-407`）：

```python
if not isinstance(causal_mask_mapping := attention_mask, dict):
    mask_kwargs = {...}
    causal_mask_mapping = {
        "full_attention": create_causal_mask(**mask_kwargs),
    }
    # The sliding window alternating layers are not always activated depending on the config
    if self.has_sliding_layers:
        causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
```

取用时（`:412-415`）：

```python
for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
    hidden_states = decoder_layer(
        hidden_states,
        attention_mask=causal_mask_mapping[self.config.layer_types[i]],
        ...
```

这套机制是 1M2 的滑动窗口的**进阶版**：Mistral 是「整个模型统一一个窗口」，
Qwen3 是「**每一层可以不一样**」，由 `config.layer_types` 逐层指定。
`layer_types` 由 config 在 `__post_init__` 里生成（`configuration_qwen3.py:86-97`）：
`use_sliding_window` 为 `False`（默认）时 `sliding_window` 被置为 `None`，于是 `layer_types` 全是 `"full_attention"`
→ `has_sliding_layers` 为 `False` → **滑动窗口这条路默认不激活**。
Qwen3 dense 因此是**纯 full attention**；这套交替机制是为将来（以及同族其他模型）留的开关，
`Qwen3Attention` 里的 `layer_type` / `sliding_window`（`:216`、`:239`）也只有配置真的启用时才起作用。

---

## 八、对照 2A5：这个文件里哪些写法值得单独记

`modular_qwen3.py` 是 2A5 那张「特殊约定速查表」的集中演示，四处都踩到了：

| modular 里的写法 | 行号 | 2A5 里的规则 | 生成结果 |
|------------------|------|--------------|----------|
| `class Qwen3RMSNorm(Qwen2RMSNorm): pass` | `:48-49` | `pass` = 原样继承 + 全局改名 | 完整类体 + 名字替换（`:50`）|
| 不写装饰器 | `:48` | 不加装饰器 = **继承父类的** | `@use_kernel_forward_from_hub("RMSNorm")`（`:49`）——modular 里根本没有这行 |
| `self.layer_type = ...` 写在 `super().__init__()` **之前** | `:62-63` | converter 会把 `super().__init__()` 强制挪到最前 | 生成文件里顺序被翻过来：`super().__init__()` 在 `:215`，`self.layer_type` 在 `:216` |
| `def forward(self, **super_kwargs: Unpack[...])` + 只写了 docstring | `:113-139` | `**super_kwargs` = 展开父类完整签名与函数体，但保留你的 docstring | `Qwen3ForCausalLM.forward` 拿到 Qwen2 的完整签名/函数体 + Qwen3 的示例 docstring（`:448` 起，docstring 正文在 `:460`）|
| `_fsdp_plan = {"lm_head": "keep_full_weight"}` | `:111` | 类级赋值**按模式匹配才覆盖** | 生成物里与 Qwen2 的值相同（`modeling_qwen2.py:410`），属于冗余但无害的显式声明 |

> **一条实用的读码结论**：`modular_qwen3.py` 只有 161 行，但其中真正属于「Qwen3 的改动」的代码
> 不超过 30 行（`q_norm`/`k_norm` 两行 + forward 里两行 + `layer_type` 一行 + `sliding_window` 一行 + config 里的字段）。
> 剩下的全是 import、`pass`、`__all__` 和样板。**这正是「读 modular」的效率所在。**

---

## 一句话总结

> **Qwen3 dense = Llama + QK-Norm（逐头、在 RoPE 之前）+ `head_dim` 与 `hidden_size` 解耦（config 显式 `head_dim=128`）**
> **+ `rope_parameters` 参数体系（`rope_type="default"`、`rope_theta` 代码默认 10000.0）+ 若干默认值调整。**
>
> DecoderLayer 一行没变，`ForCausalLM` 只换了 docstring。
> 读它是为了确认一件事：**2023 年定下的 dense 骨架，到 2025 年依然没被推翻，只是被打上了几个稳定性补丁。**

---

## 验证清单

本篇所有行号、默认值都由下面这些只读命令实测得出（在仓库根目录执行）。
**注意本环境没有安装 `torch`/`transformers`，无法 `import` 后实例化 config，因此默认值一律通过读源码 + `grep` 交叉确认，
而不是靠运行，也没有联网核对任何 checkpoint 的 `config.json`。**

```bash
# ── 文件规模与身份 ──
wc -l src/transformers/models/qwen3/{modular_qwen3.py,modeling_qwen3.py,configuration_qwen3.py}
# → 161 / 529 / 101

head -2 src/transformers/models/qwen3/modeling_qwen3.py
# → # 🚨🚨🚨 ... / # This file was automatically generated from .../modular_qwen3.py     ← 生成物
head -1 src/transformers/models/qwen3/configuration_qwen3.py
# → # Copyright 2024 The Qwen team ...                                                   ← 无警告 → 手写文件

# ── QK-Norm：位置与维度 ──
grep -n "q_norm\|k_norm" src/transformers/models/qwen3/modular_qwen3.py
# → 64: self.q_norm = Qwen3RMSNorm(self.head_dim, ...)  # unlike olmo, only on the head dim!
#    65: self.k_norm = Qwen3RMSNorm(self.head_dim, ...)
#    79: query_states = self.q_norm(self.q_proj(...).view(hidden_shape)).transpose(1, 2)
#    80: key_states = self.k_norm(...)
grep -n "query_states = \|apply_rotary_pos_emb(" src/transformers/models/qwen3/modeling_qwen3.py
# → 148: def apply_rotary_pos_emb(...)
#    252: query_states = self.q_norm(...).transpose(1, 2)     ← norm 在前
#    257: query_states, key_states = apply_rotary_pos_emb(...) ← RoPE 在后
#    ∴ 顺序 = proj → view → norm → transpose → RoPE

# 与 Llama 的 forward 逐行 diff（只应有两处语义差异）
diff <(sed -n '244,283p' src/transformers/models/llama/modeling_llama.py | sed 's/Llama/X/g') \
     <(sed -n '241,279p' src/transformers/models/qwen3/modeling_qwen3.py   | sed 's/Qwen3/X/g')
# → 只有 q_norm/k_norm 两行 + sliding_window 一行（外加签名写法差异）

diff <(sed -n '284,325p' src/transformers/models/llama/modeling_llama.py | sed 's/Llama/X/g') \
     <(sed -n '283,324p' src/transformers/models/qwen3/modeling_qwen3.py   | sed 's/Qwen3/X/g')
# → 空输出 ⇒ DecoderLayer 逐行相同

diff <(sed -n '191,214p' src/transformers/models/llama/modeling_llama.py | sed 's/Llama/X/g') \
     <(sed -n '185,208p' src/transformers/models/qwen3/modeling_qwen3.py   | sed 's/Qwen3/X/g'); echo "EXIT=$?"
# → EXIT=0 ⇒ eager_attention_forward 相同

# ── QK-Norm 的对照：OLMo2 是在整个 heads*head_dim 上归一化 ──
grep -n "q_norm\|k_norm" src/transformers/models/olmo2/modeling_olmo2.py
# → 221: Olmo2RMSNorm(config.num_attention_heads * self.head_dim, ...)   ← 全宽
#    235: query_states = self.q_norm(self.q_proj(hidden_states))         ← 在 view 之前

# ── head_dim 解耦 ──
grep -n "head_dim" src/transformers/models/qwen3/configuration_qwen3.py
# → 68: head_dim: int = 128            ← 显式常量
grep -n "head_dim" src/transformers/models/llama/configuration_llama.py
# → 84: head_dim: int | None = None / 87-88: if self.head_dim is None: 推导
grep -n "head_dim" src/transformers/models/qwen2/configuration_qwen2.py   # → 无匹配（Qwen2 没有该字段）
grep -nE "^    (hidden_size|num_attention_heads|head_dim)" src/transformers/models/qwen3_next/configuration_qwen3_next.py
# → 94: hidden_size: int = 2048 / 97: num_attention_heads: int = 16 / 108: head_dim: int = 256
#    ⇒ 2048/16 = 128 ≠ 256 —— 解耦的实证

# ── RoPE ──
sed -n '62,80p' src/transformers/models/qwen3/configuration_qwen3.py
# → 75: rope_parameters: RopeParameters | dict | None = None   ← 没有 rope_scaling / rope_theta 字段
grep -n "class RotaryEmbeddingConfigMixin" -A 5 src/transformers/modeling_rope_utils.py
# → 739: default_theta = 10_000.0                              ← rope_theta 的真实代码默认
grep -n "rope_theta = kwargs.pop" src/transformers/modeling_rope_utils.py
# → 752: rope_theta = kwargs.pop("rope_theta", getattr(self, "rope_theta", self.default_theta))
grep -rn "default_theta" src/transformers/models/qwen3/*.py                          # → 无匹配（Qwen3 dense 不覆盖）
grep -n "default_theta" src/transformers/models/qwen3_omni_moe/configuration_qwen3_omni_moe.py
# → 133: default_theta = 1000000.0                             ← 对照组：想用 1e6 必须显式覆盖
grep -rl "default_theta = " src/transformers/models/ | wc -l                         # → 72 个文件
grep -n "class RopeParameters" -A 4 src/transformers/modeling_rope_utils.py          # → 678
grep -n "ROPE_INIT_FUNCTIONS: dict" -A 8 src/transformers/modeling_rope_utils.py     # → 668-675，6 种 scaling
grep -n "rope_type = self.config.rope_parameters\|base = config.rope_parameters\|dim = getattr" src/transformers/models/qwen3/modeling_qwen3.py
# → 95 / 116 / 117
grep -n "def rope_scaling" -A 6 src/transformers/configuration_utils.py
# → 548-554：rope_scaling 只是 rope_parameters 的 property 别名

# ── tie_word_embeddings ──
grep -n "tie_word_embeddings" src/transformers/models/qwen3/configuration_qwen3.py   # → 74: = False
grep -n "tie_word_embeddings" src/transformers/models/llama/configuration_llama.py   # → 79: = False（同）
grep -n "_tied_weights_keys" src/transformers/models/qwen3/modeling_qwen3.py         # → 432（仅是声明）
grep -n "tie_word_embeddings" src/transformers/modeling_utils.py
# → 2559-2561: 若为 False，则 get_expanded_tied_weights_keys 直接 `return {}`
#    ⇒ 622.3M 的 lm_head 保持独立（151936 × 4096）

# ── 滑动窗口 / layer_types ──
grep -n "use_sliding_window\|sliding_window\|layer_types\|max_window_layers" src/transformers/models/qwen3/configuration_qwen3.py
# → 77: use_sliding_window = False / 78: sliding_window = 4096 / 79: max_window_layers = 28
#    80: layer_types = None / 87: sliding_window = ... if use_sliding_window else None
#    91-97: layer_types 为空时按 max_window_layers 逐层生成
grep -n "has_sliding_layers\|causal_mask_mapping" src/transformers/models/qwen3/modeling_qwen3.py
# → 359 / 392 / 402-407 / 415
```

---

## 导航

- 路线图：[00_roadmap.md](./00_roadmap.md)（本篇 = 入门 · 模型结构 · 1M4）
- 上一篇：[1M3_mixtral_moe.md](./1M3_mixtral_moe.md)（待写）—— MoE 第一课，同样的 modular 读法
- 下一篇：[1M5_bert_modernbert.md](./1M5_bert_modernbert.md)（待写）—— 换一个架构族：encoder 与双向注意力
- 建议对照：[1M1_llama_structure.md](./1M1_llama_structure.md) 第八节的 Llama→GLM 表，
  和本篇第七节的 Llama→Qwen3 表摆在一起看：**同一个 Llama 骨架，两条不同的改造路线。**
- 建议顺读：[2A5_modular_code_generation.md](./2A5_modular_code_generation.md) 第四节「特殊约定速查表」，
  再回看本篇第八节——`modular_qwen3.py` 是那张表最完整的实例。
