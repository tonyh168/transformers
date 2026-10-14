# Gemma 2 模型结构精读

> 阅读目标：搞懂 Gemma 2 在 Llama 骨架上加的三件事——**软上限注意力（logit soft-cap）**、
> **交替局部 / 全局注意力**、**Pre + Post 双 Norm**。
>
> 这是进阶模型线的第 1 篇。它承上启下：
> - 承 1M2（Mistral 的滑动窗口）：把「SWA」从「全部层」升级成「每隔一层」；
> - 启 2M 后面的模型：soft-cap 是第一次出现「**必须在 softmax 之前改 logits**」的结构需求，
>   它直接撞上「融合 kernel 不接受任意中间操作」这堵墙（后端兼容性问题，见 §2.5）。
>
> | 读哪个文件 | 路径 | 行数 | 说明 |
> |---|---|---|---|
> | **主读** | `src/transformers/models/gemma2/modular_gemma2.py` | 475 | 手写源文件，含 11 个类 |
> | 生成产物（勿改） | `src/transformers/models/gemma2/modeling_gemma2.py` | 559 | `make fix-repo` 生成 |
> | 配置文件（生成产物） | `src/transformers/models/gemma2/configuration_gemma2.py` | 111 | 由 modular 里的 `Gemma2Config` 落盘 |
> | 父模型 | `src/transformers/models/gemma/modular_gemma.py` | 268 | Gemma 1，它又继承 Llama |
> | 祖父 | `src/transformers/models/llama/modeling_llama.py` | — | 无 modular，直读 |
>
> **一句话定位**（先记住，读完回头验证）：
> **Gemma2 = Llama + 交替 SWA + logit soft-cap + 双 norm + GeGLU。**

---

## 速读区（TL;DR）

> ⚠️ **本篇最贵的发现：Gemma 2 的 `attn_logit_softcapping` 在默认 sdpa 后端下被静默丢弃。**
> `grep -c softcap src/transformers/integrations/sdpa_attention.py` → **0**；
> `sdpa_attention_forward` 的签名（`sdpa_attention.py:79-89`）只有 `**kwargs`，`softcap` 被吞掉，**不报错、不警告**。
> 而 HF 的默认后端就是 sdpa（`modeling_utils.py:1927`：`applicable_attention = "sdpa" if requested_attention is None else ...`）——
> 也就是说 `Gemma2ForCausalLM.from_pretrained(...)` 不显式指定 `attn_implementation` 时，
> **拿到的模型 attention 层没有 soft-cap**（`final_logit_softcapping` 仍生效，因为它写在模型 `forward` 里，与后端无关）。
> 要完整复现官方数值必须显式用 `"eager"` / `"flash_attention_2"` / `"flex_attention"`；
> 三个 paged 后端（`sdpa_paged` / `eager_paged` / `flash_paged`）同样 `grep -c softcap` = 0。

**一句话结论**：**Gemma2 = Llama + 交替 SWA（13 local + 13 global）+ logit soft-cap（attention 处 c=50 / 输出处 c=30）+ Pre+Post 双 norm + GeGLU**，
全部塞在同一份 Llama 骨架里；最容易被忽略的一处是 `query_pre_attn_scalar` 让 attention scaling **不走 `head_dim`**。

| # | 断言 | 出处（文件:行号） |
|---|------|------------------|
| 1 | Gemma2 不是直接继承 Llama，中间隔着 **Gemma 1**（Gemma 只改 3 处，Gemma2 在其上再改 4 处） | `modular_gemma2.py:38-50`、`modular_gemma.py:139-156` |
| 2 | attention soft-cap 的顺序是 **scale → soft-cap → 加 mask → softmax**，`c = 50.0` | `modular_gemma2.py:202-205`（整块 `:200-210`） |
| 3 | 第二处 soft-cap 在 `Gemma2ForCausalLM.forward` 末尾、**`loss_function` 之前**，`c = 30.0`；与后端无关，所有后端都执行 | `modular_gemma2.py:438-449` |
| 4 | 后端兼容性：eager ✅ / `flash_attention_2` ✅ / flex ✅ / **sdpa ❌ 静默失效** / paged 全 ❌ | `sdpa_attention.py`（grep 0）、`flash_attention.py:35,86`、`flex_attention.py:269,291-292` |
| 5 | flash 对 `softcap` 的支持是 `inspect.signature(flash_function).parameters` **现场探测**出来的，老版本 kernel 装了也白装 | `modeling_flash_attention_utils.py:226-245`、`:662` |
| 6 | `layer_types` 在 `Gemma2Config.__post_init__` 里**预计算**：`"sliding_attention" if (i+1) % 2 else "full_attention"`，**第 0 层是局部层**；26 层 = 13 + 13 | `modular_gemma2.py:122-128` |
| 7 | attention / mask / **KV cache** 三处必须同步：cache 逐层派出 `DynamicLayer` 或 `DynamicSlidingWindowLayer`（后者只留 `sliding_window - 1` 个 K/V） | `modular_gemma2.py:226`、`:377`、`cache_utils.py:1224-1227`、`:1785-1793` |
| 8 | DecoderLayer 有 **4 个 RMSNorm**（input / post_attention / pre_feedforward / post_feedforward），残差里加的是**归一化之后**的输出 | `modular_gemma2.py:278-282`、`:284-315` |
| 9 | MLP 是 **GeGLU**：字段名是 `hidden_activation`（不是 `hidden_act`），默认 `"gelu_pytorch_tanh"` | `modular_gemma2.py:103`、`:143-146`、`activations.py:330` |
| 10 | `self.scaling = config.query_pre_attn_scalar**-0.5`（默认 **256**），**不是** `head_dim**-0.5` | `modular_gemma2.py:225`、`modeling_llama.py:226` |
| 11 | 真实 checkpoint：9B `qpas=224 / head_dim=256`（`0.066815`）；27B `qpas=144 / head_dim=128`（`0.083333`）——**head_dim 与 `hidden_size // heads` 解耦** | `convert_gemma2_weights_to_hf.py:43-69` |
| 12 | embedding 缩放 `× sqrt(hidden_size)` 由 `Gemma2TextScaledWordEmbedding` 实现，modular 里 `grep -c "TextScaled"` = **0**（依赖自动拉取） | `modeling_gemma2.py:338`、`:387-389` |
| 13 | `Gemma2RMSNorm` = `(1 + weight) * x`，weight **初始化为 0**（Llama 是 `weight * x`、初始 1） | `modular_gemma.py:119-136`、`:159-167` |

**最容易踩的坑 / 反直觉发现**

- **默认 sdpa 静默丢掉 attention soft-cap**（见开头 ⚠️）——没有异常、没有 warning，只是数值悄悄变了。
- **`query_pre_attn_scalar` 的默认值恰好等于默认 `head_dim`（256）**，所以「用默认 config 试一下」会发现两者一致，**真实 checkpoint 却不一致**；任何「把 Gemma2 的 scaling 按 `head_dim` 重算」的推理优化都会算错（9B 放大 ~7%、27B 缩小 ~6%）。
- **老 checkpoint 的 `config.json` 写的是 `hidden_act`，不会被读进 `hidden_activation`**，于是回落到默认值——恰好也是 `gelu_pytorch_tanh`，所以不出错（沉默的巧合，别当成机制保证）。
- **`layer_types` 不是 `layer_idx % 2` 现算的**，而是 config 里预计算好的列表；用户可以覆盖，但 attention / mask / cache 三处会同时跟着变。
- **modular 里 `self.layer_type = ...` 写在 `super().__init__()` 之前，生成文件里被挪到了之后**——这正是 2A5 讲的 converter 收尾规则（`modeling_gemma2.py:223-224`）。
- **别把它说成「全层 SWA」**（那是 Mistral）——Gemma 2 是「13 层窗口 4096 + 13 层全序列」，而且当 `n > W` 时 global 层**反而更贵**。
- **`base_model_tp_plan` 与 Llama 的逐行相同，但含义不同**：Gemma2 的 `q_proj` 输出是 `num_heads × head_dim`（9B = 4096 ≠ `hidden_size` 3584），做切分要按前者算（`modular_gemma2.py:81-89`）。
- 顺带记一笔：Gemma 3 的 config 把两个 softcap **都设成了 `None`**（`gemma3/configuration_gemma3.py:99-100`），且它的交替模式是 `_sliding_window_pattern = 6`——soft-cap 是 Gemma 2 一代的选择，不是 Gemma 家族的必备件。

**读完应该能回答的问题**

- 默认 `from_pretrained` 加载 Gemma 2 时，哪一处 soft-cap 生效、哪一处不生效？为什么？
- soft-cap 为什么必须发生在 softmax **之前**？这和融合 kernel「不接受任意中间操作」的冲突点在哪？
- 哪些后端真正支持 attention soft-cap？flash 的「支持」是怎么判定的？
- `layer_types` 在哪里、按什么规则生成？改它会影响哪三处？
- Gemma 2 的 attention scaling 与 Llama 的本质区别是什么？举一个真实 checkpoint 的数字。
- 为什么 Gemma 2 被诟病「同参数量下更慢」？正文给了哪两条代价？

---

## 前置知识

- [1M1_llama_structure.md](./1M1_llama_structure.md)：`LlamaAttention` / `LlamaMLP` / `LlamaDecoderLayer` / `LlamaRMSNorm` 的骨架——本篇所有「对照」都以它为基准。
- [1M2_mistral_sliding_window.md](./1M2_mistral_sliding_window.md)：滑动窗口注意力（SWA）怎么实现、`sliding_window_overlay` 长什么样。Gemma2 是它的「交替版」。
- [2A5_modular_code_generation.md](./2A5_modular_code_generation.md)：**强烈建议先读**。本篇会多次用到它的结论：
  - `modular_gemma2.py` 里 `pass` 的类 = 原样继承；
  - 生成文件里「modular 里没写过」的类是**依赖自动拉取**的，不是新东西；
  - 生成文件里的 `super().__init__()` 是被强制挪到 `__init__` 最前面的。

---

## 一、Gemma 2 的继承树（实测基类）

Gemma2 不是直接继承 Llama，而是隔着 Gemma 1。modular 文件的开头就是证据
（`modular_gemma2.py:38-50`）：

```python
from ..gemma.modeling_gemma import (
    GemmaAttention,
    GemmaForCausalLM,
    GemmaForSequenceClassification,
    GemmaForTokenClassification,
    GemmaMLP,
    GemmaModel,
    GemmaPreTrainedModel,
    GemmaRMSNorm,
    GemmaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)
```

于是链条是：

```
LlamaAttention ─┐
                ├─→ GemmaAttention ─→ Gemma2Attention
LlamaMLP ───────┴─→ GemmaMLP ───────→ Gemma2MLP
LlamaModel ───────→ GemmaModel ─────→ Gemma2Model
LlamaForCausalLM ─→ GemmaForCausalLM → Gemma2ForCausalLM
```

`modular_gemma.py` 里 Gemma 对 Llama 的改动只有三处（`modular_gemma.py:139-156`）：

| Gemma 1 的改动 | 代码位置 | 说明 |
|---|---|---|
| `GemmaRMSNorm` 换成 `(1.0 + weight)` 形式、weight 初始化为 0 | `modular_gemma.py:119-136`、`159-167` | 与 Llama 的 `weight * x`（weight 初始化为 1）数学等价，实现不同 |
| `GemmaMLP` 三个投影强制 `bias=False` | `modular_gemma.py:139-144` | 忽略 `config.mlp_bias` |
| `GemmaTextScaledWordEmbedding`（embedding 乘 `sqrt(hidden_size)`） | `modular_gemma.py:105-116`、`170-176` | 下文 §6.1 细讲 |

**而 Gemma2 在 Gemma 之上又改了 4 处**——这就是本篇的全部内容：

| # | Gemma2 的改动 | 主要位置 |
|---|---|---|
| 1 | attention logits 与最终 logits 的 **soft-cap** | `modular_gemma2.py:202-205`、`442-445` |
| 2 | `layer_types` **交替** sliding / full | `modular_gemma2.py:123-126` |
| 3 | DecoderLayer 从 2 个 norm 加到 **4 个 norm** | `modular_gemma2.py:278-282`、`293-315` |
| 4 | attention scaling 从 `head_dim**-0.5` 改为 `query_pre_attn_scalar**-0.5`；激活函数换成 `gelu_pytorch_tanh` | `modular_gemma2.py:225`、`143-146` |

---

## 二、软上限注意力（logit soft-cap）

### 2.1 数学：一个被 `tanh` 夹住的软饱和函数

```
softcap(x) = c · tanh(x / c)
```

性质：
- `x → 0` 时 `tanh(x/c) ≈ x/c`，所以 `softcap(x) ≈ x`：**小 logits 几乎不受影响**；
- `x → ±∞` 时 `tanh → ±1`，所以 `softcap(x) → ±c`：**logits 被关进 `(-c, +c)` 开区间**；
- 处处可导、单调，梯度 `sech²(x/c)` 在饱和区趋近 0 —— 是一个「软」的 clamp，不是硬截断。

实测代码（`modular_gemma2.py:202-205`，展开写法）：

```python
if softcap is not None:
    attn_weights = attn_weights / softcap
    attn_weights = torch.tanh(attn_weights)
    attn_weights = attn_weights * softcap
```

三行就是 `c * tanh(x / c)`，一行不多。

```
 softcap(x) = c·tanh(x/c)，以 c = 50 为例（示意图，非严格坐标）
   │
+50┤                                 ······················ ← 渐近线 +c，永远达不到
   │                          ,─────'
+40┤                    ,────'
   │              ,────'
+30┤         ,───'
   │       ,'
+20┤     ,'
   │   ,'
+10┤ ,'
   │,'
 0 ┼───────────────────────────────────────────────────► x（原始 attention logit）
   0    50   100   150   200   250   300   350   400
   |←─ 近似恒等 ─→|←──────── 强压缩区 ────────→
   小 logits 照旧     大 logits 被「削平」，梯度也随之变小
```

对应的真实数值（`python3` 实算，`c=50`）：

| 原始 logit `x` | 0 | 10 | 25 | 50 | 75 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|---|---|
| `50·tanh(x/50)` | 0.000 | 9.869 | 23.106 | 38.080 | 45.257 | 48.201 | 49.966 | 50.000 |

**读法**：`x ≤ 25` 时几乎没变化；`x = 50` 已经被压掉 24%；`x = 200` 压缩率 75%。
`|x| ≫ c` 之后彻底饱和。

### 2.2 作用位置 A：attention logits（softmax 之前）— **实测行号**

这是本篇最关键的一处。soft-max 之前对 `QK^T * scaling` 的结果做 soft-cap。

**代码 1：模型自带的 eager 实现**（`modular_gemma2.py:183-214`，函数 `eager_attention_forward`）

```python
# modular_gemma2.py:200-210
attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

if softcap is not None:                       # ← 202
    attn_weights = attn_weights / softcap     # ← 203
    attn_weights = torch.tanh(attn_weights)   # ← 204
    attn_weights = attn_weights * softcap     # ← 205
if attention_mask is not None:
    attn_weights = attn_weights + attention_mask

# upcast attention to fp32
attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
```

顺序值得记：**scale → soft-cap → 加 mask → softmax**。
soft-cap 在 mask 之前（mask 是 `+ (-inf)`，若顺序颠倒会把 `-inf` 也喂给 `tanh`）。
生成文件里同一段在 `modeling_gemma2.py:203-206`。

**代码 2：把 soft-cap 值传进后端分发器**（`modular_gemma2.py:249-264`）

```python
# modular_gemma2.py:249-264
attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
    self.config._attn_implementation, eager_attention_forward
)

attn_output, attn_weights = attention_interface(
    self,
    query_states,
    key_states,
    value_states,
    attention_mask,
    dropout=self.attention_dropout if self.training else 0.0,
    scaling=self.scaling,
    sliding_window=self.sliding_window,        # ← 226 行算好的：full 层是 None
    softcap=self.attn_logit_softcapping,       # ← 262：这里传出去
    **kwargs,
)
```

`self.attn_logit_softcapping` 在 `__init__` 里取自 config（`modular_gemma2.py:222`）：

```python
self.attn_logit_softcapping = self.config.attn_logit_softcapping   # 默认 50.0
```

> **注意 `get_interface(impl, eager_attention_forward)` 的第二个参数**：
> `"eager"` 并不在 `AttentionInterface._global_mapping` 里（`modeling_utils.py:5150-5165` 只注册了
> `flash_attention_2/3/4`、`flex_attention`、`sdpa`、`paged|*`），所以当 `_attn_implementation == "eager"` 时
> `get_interface` 返回的是**默认值**，也就是 **Gemma2 自己写的这个 `eager_attention_forward`**
> （不是 `llama` 的那份——Llama 的 eager 实现根本没有 `softcap` 参数）。

### 2.3 作用位置 B：最终 lm_head 输出 logits — **实测行号**

第二处不在 attention 里，而在 `Gemma2ForCausalLM.forward` 的末尾
（`modular_gemma2.py:438-450`，生成文件对应 `modeling_gemma2.py:523-535`）：

```python
# modular_gemma2.py:438-449
hidden_states = outputs.last_hidden_state
# Only compute necessary logits, and do not upcast them to float if we are not computing the loss
slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
logits = self.lm_head(hidden_states[:, slice_indices, :])
if self.config.final_logit_softcapping is not None:                      # ← 442
    logits = logits / self.config.final_logit_softcapping                # ← 443
    logits = torch.tanh(logits)                                          # ← 444
    logits = logits * self.config.final_logit_softcapping                # ← 445

loss = None
if labels is not None:
    loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)  # ← 449
```

三个含义：

1. 同一个 `c·tanh(x/c)` 公式，但 `c` 换成了 `final_logit_softcapping = 30.0`（`modular_gemma2.py:118`）。
2. **它在 `loss_function` 之前**：交叉熵吃的是 soft-cap 之后的 logits。
   logits 被关进 `(-30, +30)`，因此任意两个 token 的 logit 差有硬上界 60，
   `log_softmax` 的输入不会再被某个巨大 logit 拖爆——这正是「稳定训练」的直接机制。
3. **它与注意力后端无关**：这段代码在模型 `forward` 里，无论 eager / sdpa / flash / flex 都会执行。
   返回给用户的 `outputs.logits` 也是 soft-cap 后的值（`(-30, +30)`），
   这也是为什么 Gemma2 的采样分布比同规模模型「平」——温度再高也拉不出超过 60 的 logit 差。

### 2.4 动机：为什么需要 soft-cap

| 问题 | 不 soft-cap 时 | soft-cap 之后 |
|---|---|---|
| attention logits 爆炸（长上下文时 `QK^T` 随 head_dim / 序列增长） | softmax 饱和成 one-hot，梯度消失、fp16 溢出 | logits ∈ `(-50, 50)`，softmax 保持「软」 |
| 输出 logits 爆炸（训练后期 + 大词表 256k） | 交叉熵对个别 token 过自信，loss 尖峰 / 发散 | logits ∈ `(-30, 30)`，loss 有界 |
| 数值精度 | bf16/fp16 下大数相加丢精度 | 有界，量化/低精度训练更稳 |

（这两个超参的文档字符串与默认值都在 config 里：`modular_gemma2.py:62-65` 的说明、`:118-119` 的默认值。）

### 2.5 副作用 / 限制：`tanh` 与融合 kernel 的兼容性（**实测**）

`c·tanh(x/c)` 必须作用在 **softmax 之前**的注意力分数矩阵上。
而 flash-attention 这类融合 kernel 把「scale → mask → softmax → @V」整条链焊死在一个 CUDA kernel 里，
**默认不允许在中间插入任意算子**——除非 kernel 自己把这个口子暴露出来
（FA2/FA3 就是为了 Gemma 2 专门加了 `softcap` 参数）。
所以各家后端只能各自决定「认不认这个参数」：

| 后端 | attention soft-cap 是否生效 | 实测依据 |
|---|---|---|
| `eager` | ✅ 生效 | `modular_gemma2.py:202-205`，模型自带实现 |
| `flash_attention_2` | ✅ 生效（FA kernel 原生支持 `softcap`） | `modeling_flash_attention_utils.py:628-629` 注释直接写了 “Softcap for the attention logits, used e.g. in gemma2.”；`:662-663` 传参；`integrations/flash_attention.py:35,86` |
| `flex_attention` | ✅ 生效（在 `score_mod` 里做） | `integrations/flex_attention.py:269, 291-292`：`score = softcap * torch.tanh(score / softcap)` |
| `sdpa` | ❌ **静默失效** | `integrations/sdpa_attention.py` 整个文件 `grep -c softcap` = **0**；`sdpa_attention_forward` 签名（`:79-89`）只有 `**kwargs`，`softcap` 被吞掉，既不报错也不警告 |
| `paged\|*`（continuous batching） | ❌ 都不生效 | `integrations/sdpa_paged.py`、`eager_paged.py`、`flash_paged.py` 三个文件里 `grep -c softcap` 均为 0 |

**两个必须记住的结论**：

1. **HF 的默认后端是 `sdpa`**（`modeling_utils.py:1927`：`applicable_attention = "sdpa" if requested_attention is None else requested_attention`）。
   也就是说——**直接 `Gemma2ForCausalLM.from_pretrained(...)` 而不指定 `attn_implementation` 时，
   attention 的 soft-cap 是被悄悄跳过的**；只有 `final_logit_softcapping` 还生效（它在模型 forward 里）。
   想完整复现官方数值，需要显式 `attn_implementation="eager"` 或 `"flash_attention_2"` / `"flex_attention"`。
2. **flash 支持也是「有条件」的**：`supports_mapping` 是用 `inspect.signature(flash_function).parameters`
   现场探测出来的（`modeling_flash_attention_utils.py:226-245`），
   只有装的那版 kernel 的签名里真有 `softcap` 参数，才会把它传下去（`:662`）。
   老版本 flash-attn 装了也白装。

> 这也解释了 1M2 里看不到的问题：SWA 只改 **mask**（喂给 kernel 的输入），所以所有后端都支持；
> soft-cap 要改 **kernel 内部的中间结果**，于是只有「自己暴露了这个口子」的后端支持。

### 2.6 两个超参的默认值

| 参数 | 默认值 | 位置 | 作用 |
|---|---|---|---|
| `attn_logit_softcapping` | `50.0` | `modular_gemma2.py:119` = `configuration_gemma2.py:91` | attention logits 的上界 `c` |
| `final_logit_softcapping` | `30.0` | `modular_gemma2.py:118` = `configuration_gemma2.py:90` | lm_head logits 的上界 `c` |

两个都是 `float | None`，**置 `None` 就关闭**（`if softcap is not None` / `if ... is not None`）。
Gemma 3 的 config 默认就把这两个都设成了 `None`（`gemma3/configuration_gemma3.py:99-100`），
可见这是 Gemma 2 一代的选择，不是 Gemma 家族的必备件。

---

## 三、交替局部 / 全局注意力

### 3.1 层类型判定逻辑（**实测：不是 `layer_idx % 2`，而是预计算好的 `config.layer_types`**）

判定发生**一次**，在 `Gemma2Config.__post_init__` 里（`modular_gemma2.py:122-128`，
生成文件 `configuration_gemma2.py:94-99`）：

```python
# modular_gemma2.py:122-128
def __post_init__(self, **kwargs):
    if self.layer_types is None:
        self.layer_types = [
            "sliding_attention" if bool((i + 1) % 2) else "full_attention" for i in range(self.num_hidden_layers)
        ]

    super().__post_init__(**kwargs)
```

`(i + 1) % 2` 的细节：`i` 是 0-based 层号，所以

| 层号 `i` | 0 | 1 | 2 | 3 | … | 24 | 25 |
|---|---|---|---|---|---|---|---|
| `(i+1) % 2` | 1 | 0 | 1 | 0 | … | 1 | 0 |
| `layer_types[i]` | `sliding_attention` | `full_attention` | `sliding_attention` | `full_attention` | … | `sliding_attention` | `full_attention` |

用 `python3` 复算 `num_hidden_layers=26`：**13 个 `sliding_attention` + 13 个 `full_attention`**，
**第 0 层是局部层**（不是先全局）。

```
 层号     0        1        2        3        4        5      ...    24       25
        ┌────────┬────────┬────────┬────────┬────────┬────────┬────┬────────┬────────┐
 类型    │ local  │ global │ local  │ global │ local  │ global │ …  │ local  │ global │
        │ sliding│  full  │ sliding│  full  │ sliding│  full  │    │ sliding│  full  │
        └────────┴────────┴────────┴────────┴────────┴────────┴────┴────────┴────────┘
 窗口     4096      ∞        4096      ∞        4096      ∞            4096      ∞
        │        │        │        │        │        │            │        │
        └ 相邻两层的 local 层之间夹着一个 global 层，长程信息直接从这里「跳过」若干窗口 ┘
```

`layer_types` 可以直接被用户覆盖（它是 config 字段，`modular_gemma2.py:117`），
但一旦覆盖，**attention / mask / cache 三处会同时跟着变**（下面 3.3）。

### 3.2 三个消费点（必须三处一致，否则算错）

| # | 消费点 | 代码 | 作用 |
|---|---|---|---|
| 1 | `Gemma2Attention.__init__` | `modular_gemma2.py:219`、`226` | 决定本层给后端传不传 `sliding_window` |
| 2 | `Gemma2Model.forward` | `modular_gemma2.py:355-368`、`374-382` | 决定本层用哪个 mask |
| 3 | KV cache 构造 | `cache_utils.py:1700-1734`、`1785-1793` | 决定本层 cache 是定长还是无限增长 |

**① attention 层**（`modular_gemma2.py:218-226`）：

```python
def __init__(self, config: Gemma2Config, layer_idx: int):
    self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None

    super().__init__(config, layer_idx)
    self.attn_logit_softcapping = self.config.attn_logit_softcapping
    self.attention_dropout = self.config.attention_dropout
    self.is_causal = not getattr(config, "use_bidirectional_attention", False)
    self.scaling = config.query_pre_attn_scalar**-0.5
    self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None   # ← 226
```

**full 层的 `self.sliding_window` 是 `None`**，所以它传给 flash 的 `window_size` 也是 None——
这就是「同一个模型里两种注意力模式」在**同一份 attention 代码**里的落地方式：
不换类，只换一个属性。

> **modular 视角的小彩蛋（对照 2A5）**：modular 里 `self.layer_type = ...` 写在 `super().__init__()` **之前**，
> 而生成文件里它被挪到了 `super().__init__()` **之后**（`modeling_gemma2.py:223-224`）。
> 这正是 2A5 讲的「converter 强制把 `super().__init__()` 放到 `__init__` 最前面」那条收尾规则。

**② mask 选择**（`modular_gemma2.py:355-368`、`374-382`）：

```python
# modular_gemma2.py:355-368
# It may already have been prepared by e.g. `generate`
if not isinstance(causal_mask_mapping := attention_mask, dict):
    mask_kwargs = {
        "config": self.config,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
    }
    # Create the masks
    causal_mask_mapping = {
        "full_attention": create_causal_mask(**mask_kwargs),
        "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
    }

# ...
for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):   # ← 374
    hidden_states = decoder_layer(
        hidden_states,
        attention_mask=causal_mask_mapping[self.config.layer_types[i]],            # ← 377：按层类型取 mask
        ...
    )
```

和 1M2 的 Mistral 对比：

| | Mistral（1M2） | Gemma 2 |
|---|---|---|
| mask 数量 | 一个（要么 causal 要么 sliding-causal） | **两个**（`full_attention` + `sliding_attention` 各一个） |
| 选择方式 | 建模型时二选一（`modular_mistral.py:137`） | 每层用 `layer_types[i]` 查字典（`modular_gemma2.py:377`） |
| 两种 mask 的底层实现 | `masking_utils.py:133-138` 的 `sliding_window_causal_mask_function` = `and_masks(sliding_window_overlay(W), causal_mask_function)` | 同一个函数，只是只有一半的层用它 |

**③ KV cache**（这是「交替 SWA 省算力」真正落地的地方）：

`Gemma2Model.forward` 里 `past_key_values = DynamicCache(config=self.config)`（`modular_gemma2.py:347`），
`DynamicCache.__init__` 拿着 config 去查 `layer_types`，逐层决定 cache 类
（`cache_utils.py:1785-1793`）：

```python
# cache_utils.py:1789-1793
layer_types, layer_kwargs = get_layer_types_and_kwargs(decoder_config)
# Dispatch the layer types
layers = [DYNAMIC_LAYER_TYPE_MAPPING[layer_type](**layer_kwargs) for layer_type in layer_types]
```

映射表（`cache_utils.py:1224-1227`）：

| `layer_types` | cache 层类 | 行为 |
|---|---|---|
| `full_attention` | `DynamicLayer` | 长度随序列无限增长 |
| `sliding_attention` | `DynamicSlidingWindowLayer`（`cache_utils.py:203`） | 只保留最近 `sliding_window - 1` 个 token 的 K/V（`:251-254`） |

测试里对这一点有硬断言（`tests/models/gemma2/test_modeling_gemma2.py:434-441`，
测试名 `test_generation_beyond_sliding_window_dynamic`，第 387 行）：

```python
for layer, layer_type in zip(dynamic_cache.layers, model.config.layer_types):
    if layer_type == "sliding_attention":
        self.assertTrue(isinstance(layer, DynamicSlidingWindowLayer))
        self.assertEqual(layer.keys.shape[-2], model.config.sliding_window - 1)
    else:
        self.assertTrue(isinstance(layer, DynamicLayer))
```

### 3.3 交替 vs 全 SWA：为什么这么做

| | Mistral（全部 SWA） | Gemma 2（交替） |
|---|---|---|
| 每层能看到多远 | 最近 `W` 个 token | local 层 `W`，global 层**全序列** |
| KV cache 大小 | 全部层 O(W)，常数 | 一半层 O(W)、一半层 O(n) |
| 长程信息 | 只能靠**层层叠加**间接传递（第 k 层理论感受野 ≈ k·W，且逐层衰减） | global 层**直接**一跳到底，不衰减 |
| 算力 | 每层 O(n·W) | 一半 O(n·W)、一半 O(n²)；`W=4096`、`n=8192` 时 global 层反而更贵 |
| 定位 | 追求「短窗口也够用」 | 追求「长程能力不打折，同时压一半的 cache」 |

**核心权衡一句话**：交替 = 用「一半的层退回全注意力」这个代价，换回「长程信息不需要靠 10 层传递」这个能力。
纯 SWA 的模型在长上下文里会「记得近处、忘了远处」；全全局的模型 KV cache 线性膨胀。
Gemma 2 的答案是各让一半。

### 3.4 `sliding_window` 的默认值（实测）

| 模型 | `sliding_window` 默认值 | 位置 |
|---|---|---|
| Gemma 2 | **4096** | `modular_gemma2.py:116` = `configuration_gemma2.py:88` |
| Mistral | 4096 | `mistral/configuration_mistral.py:80` |

一样都是 4096。但**语义差别很大**：
Mistral 是「全部 32 层都 4096」，Gemma 2 是「13 层 4096 + 13 层全序列」。
转换脚本里 9B / 27B 也显式写了 `sliding_window=4096`
（`convert_gemma2_weights_to_hf.py:53`、`:66`）。

mask 端的窗口定义（`masking_utils.py:92-99`，与 1M2 同一个函数）：

```python
def sliding_window_overlay(sliding_window: int) -> Callable:
    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return kv_idx > q_idx - sliding_window
    return inner_mask
```

`masking_utils.py:133-138` 把它和因果 mask 用 `and_masks` 交起来：
`sliding_window_causal_mask_function = and_masks(sliding_window_overlay(W), causal_mask_function)`。
窗口大小 `W` 从 config 读（`masking_utils.py:1180`：`sliding_window = getattr(config, "sliding_window", None)`）。

> 顺带对照 **Gemma 3**：它也是「局部 + 全局」交替，但**不是隔一层**，
> 而是 `_sliding_window_pattern = 6`（`gemma3/modular_gemma3.py:120-124`）——每 6 层才夹一个全局层。
> 从 2 → 6 的方向上，可以看出「全局层更贵」这个认知在演化。

---

## 四、Pre + Post 双 Norm

### 4.1 4 个 norm + 真实 forward

`Gemma2DecoderLayer.__init__`（`modular_gemma2.py:271-282`）里一共有 **4 个 RMSNorm**：

```python
# modular_gemma2.py:271-282
class Gemma2DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Gemma2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        self.self_attn = Gemma2Attention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma2MLP(config)
        self.input_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.pre_feedforward_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

forward（`modular_gemma2.py:284-315`，**原样照贴**）：

```python
def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
            position_ids=None, past_key_values=None, **kwargs):
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, _ = self.self_attn(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        **kwargs,
    )
    hidden_states = self.post_attention_layernorm(hidden_states)   # ← 出 attention 后再 norm 一次
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.pre_feedforward_layernorm(hidden_states)   # ← 进 MLP 前 norm
    hidden_states = self.mlp(hidden_states)
    hidden_states = self.post_feedforward_layernorm(hidden_states)  # ← 出 MLP 后再 norm 一次
    hidden_states = residual + hidden_states

    return hidden_states
```

### 4.2 与 Llama 的 Pre-Norm 对照

Llama（`modeling_llama.py:305-324`）：

```python
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)
hidden_states, _ = self.self_attn(...)
hidden_states = residual + hidden_states          # 残差里加的是「attention 的原始输出」

residual = hidden_states
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = self.mlp(hidden_states)
hidden_states = residual + hidden_states
```

```
        Llama（Pre-Norm，2 个 norm）                Gemma2（Pre + Post，4 个 norm）
        ─────────────────────────                  ──────────────────────────────
 x ──┬──────────────────────────►            x ──┬──────────────────────────────►
     │  ┌───────────────┐                         │  ┌───────────────┐
     └─►│ input_norm    │                         └─►│ input_norm    │
        └──────┬────────┘                            └──────┬────────┘
               ▼                                            ▼
         ┌───────────┐                                ┌───────────┐
         │ attention │                                │ attention │
         └─────┬─────┘                                └─────┬─────┘
               │                                            ▼
               │                                    ┌──────────────────┐
               │                                    │ post_attention_norm│ ← 新增
               │                                    └────────┬─────────┘
               │  ┌───────────┐                              │
               └─►│  (+)      │◄─────────────────────────────┘
                  └─────┬─────┘   （残差加的是「norm 后的输出」）
                        │
   （MLP 子层同理，Llama 是 post_attention_norm → mlp → (+)，
     Gemma2 是 pre_feedforward_norm → mlp → post_feedforward_norm → (+)）

```

**本质区别**：Llama 是「**norm 在子层入口**」（Pre-Norm），残差分支里流过的是子层**未经归一的输出**；
Gemma2 在**子层出口**又加了一次 norm，残差分支里流过的是**归一化之后的输出**。

### 4.3 动机：控制残差流的数值范围

残差流（residual stream）是所有层共享的那条主干。Pre-Norm 的问题在于：
子层的输出幅度不受约束，逐层累加到主干上后，主干的方差会随深度增长
（这就是为什么 Pre-Norm 模型常常需要在最后加一个 `final norm`，也是为什么训练时要用 warmup、要压小 LR）。

Gemma2 的做法是：**每次往主干上「加东西」之前，先把要加的东西归一化**。
于是主干上每一项的尺度都被 RMSNorm 钉在同一个量级，`hidden_states` 的方差不会随深度漂移。
这和 §2 的 soft-cap 是**同一件事的两个方向**：soft-cap 管「传出去的值」有界，
双 norm 管「层与层之间累加的值」有界。

代价很直接：**每层多 2 个 RMSNorm 的参数和计算**（26 层 × 2 = 52 个额外 norm），
以及多了 2 次 fp32 的 upcast（`GemmaRMSNorm.forward` 内部把输入转 fp32，`modular_gemma.py:129-133`）。
这是 Gemma 2 被诟病「同参数量下更慢」的原因之一。

---

## 五、GeGLU：激活函数实测

**结论：Gemma 2 的 MLP 是 GeGLU（门控 GELU），不是 Llama 的 SwiGLU。**

推理链（三步实测）：

**① config 的字段叫 `hidden_activation`，不是 Llama 的 `hidden_act`**：

| | Llama | Gemma 2 |
|---|---|---|
| 字段名 | `hidden_act` | `hidden_activation` |
| 默认值 | `"silu"`（`llama/configuration_llama.py:70`） | `"gelu_pytorch_tanh"`（`modular_gemma2.py:103`） |
| 位置 | — | `modular_gemma2.py:103` = `configuration_gemma2.py:75` |

**② `Gemma2MLP` 只在 `__init__` 里重绑了激活函数**（`modular_gemma2.py:143-146`）：

```python
class Gemma2MLP(GemmaMLP):
    def __init__(self, config):
        super().__init__(config)
        self.act_fn = ACT2FN[config.hidden_activation]   # ← 读的是 hidden_activation
```

（`GemmaMLP` 继承 `LlamaMLP`，父类 `LlamaMLP.__init__` 用的是 `ACT2FN[config.hidden_act]`——
`modeling_llama.py:172`。所以这里是「继承过来再改一行」，正是 2A5 说的那种覆盖。）

**③ `"gelu_pytorch_tanh"` 查表落到 `GELUTanh`**（`activations.py:330`）：

```python
"gelu_pytorch_tanh": GELUTanh,
```

而门控结构本身一字没动，还是 `LlamaMLP.forward`（`modeling_llama.py:174-176`）：

```python
down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

`act_fn` 从 SiLU 换成 GELU-tanh-approx → **SwiGLU 变成 GeGLU**，形状和参数量完全不变。

| | Llama | Gemma 1 | Gemma 2 | Mistral |
|---|---|---|---|---|
| 门控结构 | GLU（3 个投影） | 同 | 同 | 同 |
| 激活 | `silu`（SwiGLU） | `gelu_pytorch_tanh`（GeGLU） | `gelu_pytorch_tanh`（GeGLU） | `silu`（SwiGLU） |
| 字段名 | `hidden_act` | `hidden_act` | **`hidden_activation`** | `hidden_act` |
| 三个投影的 bias | `config.mlp_bias`（默认 False） | **强制 False** | 强制 False | `config.mlp_bias` |

> 注意 Gemma 2 的字段名从 `hidden_act` 改成了 `hidden_activation`：
> 老 checkpoint 的 `config.json` 里写的是 `hidden_act`，加载 Gemma2 时不会被读进 `hidden_activation`，
> 于是回落到默认的 `gelu_pytorch_tanh`——恰好也是 Gemma 1 用的值，所以不会出错。

---

## 六、其余 Gemma 家族特色（实测）

### 6.1 embedding 的 `normalizer`：Gemma 1 的遗产，Gemma 2 保留了

先澄清历史（`git log -S` 实测）：

| 时期 | 实现 | 证据 |
|---|---|---|
| 早期 Gemma 1 | `GemmaModel.forward` 里临时算一个 `normalizer` 张量，乘到 hidden_states 上 | 提交 `8e2fc52ea3`（PR #29729）里加过 `normalizer = torch.tensor(self.config.hidden_size**0.5, ...)` |
| 现在（Gemma 1 与 Gemma 2） | 改成一个 `nn.Embedding` 子类，缩放内建在 embedding 里 | 提交 `0e7cb4e3ea`（PR #44432「Make paligemma embed tokens standard」）把 `normalizer` 整个删掉 |

**当前实现**（`modular_gemma.py:105-116`，Gemma 1 定义，Gemma 2 继承）：

```python
class GemmaTextScaledWordEmbedding(nn.Embedding):
    """
    This module overrides nn.Embeddings' forward by multiplying with embeddings scale.
    """
    def __init__(self, num_embeddings, embedding_dim, padding_idx, embed_scale: float = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.scalar_embed_scale = embed_scale
        self.embed_scale = nn.Buffer(torch.tensor(embed_scale), persistent=False)

    def forward(self, input_ids: torch.Tensor):
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)
```

缩放系数在 `GemmaModel.__init__` 里给定（`modular_gemma.py:170-176`）：

```python
self.embed_tokens = GemmaTextScaledWordEmbedding(
    config.vocab_size, config.hidden_size, self.padding_idx, embed_scale=self.config.hidden_size**0.5
)
```

**Gemma 2 是否保留？保留。** 证据：
- `modular_gemma2.py` 里 **`grep -c "TextScaled"` = 0**（modular 里根本提都没提）；
- 但生成文件里有 `class Gemma2TextScaledWordEmbedding(nn.Embedding)`（`modeling_gemma2.py:338`），
  并且在 `Gemma2Model.__init__` 里被使用（`modeling_gemma2.py:387-389`）。

也就是说它是**依赖自动拉取**的类（2A5 规则 4），名字做了全局替换（`Gemma` → `Gemma2`）。
效果等价于在 modular 里写 `class Gemma2TextScaledWordEmbedding(GemmaTextScaledWordEmbedding): pass`。

**为什么要有它**：Gemma 系用 `sqrt(hidden_size)` 放大了 embedding，让 embedding 与后续 hidden state 同量级
（配合 §4 的双 norm 一起，维持残差流的尺度一致性）。

### 6.2 bias 的默认值

| 参数 | Gemma 2 默认 | 位置 | 说明 |
|---|---|---|---|
| `attention_bias` | **`False`** | `modular_gemma2.py:113` | 与 Llama 相同（`configuration_llama.py:81`） |
| MLP 的 bias | 无（GemmaMLP 强制 `bias=False`） | `modular_gemma.py:142-144` | Llama 用的是 `config.mlp_bias`，Gemma 系直接写死 False |

所以 Q/K/V/O 和 gate/up/down **全部无 bias**，和 Llama 的常规配置一致。

### 6.3 `query_pre_attn_scalar`：**attention scaling 不是 `head_dim**-0.5`**

这是 Gemma 2 最容易被忽略、但直接影响数值的一处。**实测代码**（`modular_gemma2.py:225`，
生成文件 `modeling_gemma2.py:229`）：

```python
self.scaling = config.query_pre_attn_scalar**-0.5      # Gemma 2
```

对照 Llama（`modeling_llama.py:226`）：

```python
self.scaling = self.head_dim**-0.5                     # Llama / Mistral
```

| | Llama | Gemma 2 |
|---|---|---|
| scaling | `1 / sqrt(head_dim)` | `1 / sqrt(query_pre_attn_scalar)` |
| `query_pre_attn_scalar` 默认值 | 无此字段 | **256**（`modular_gemma2.py:115`） |
| 默认 config 下 | `head_dim=256` → `1/16 = 0.0625` | `256**-0.5 = 1/16 = 0.0625`（**恰好相同**） |

**关键点在「默认 config 恰好相同，真实 checkpoint 并不相同」**。
转换脚本里的真实值（`convert_gemma2_weights_to_hf.py:43-69` 实测）：

| 模型 | `hidden_size` | `num_attention_heads` | `head_dim` | `query_pre_attn_scalar` | `1/sqrt(qpas)` | `1/sqrt(head_dim)` |
|---|---|---|---|---|---|---|
| 默认 config（≈2B） | 2304 | 8 | 256 | 256 | 0.062500 | 0.062500 |
| **9B** | 3584 | 16 | 256 | **224** | **0.066815** | 0.062500 |
| **27B** | 4608 | 32 | 128 | **144** | **0.083333** | 0.088388 |

**两个发现**：

1. Gemma 2 **把 `head_dim` 从 `hidden_size // num_attention_heads` 里解耦了**：
   9B 的 `hidden_size/heads = 3584/16 = 224`，但 `head_dim = 256` —— Q 的输出是 `16 × 256 = 4096`，
   比 `hidden_size = 3584` 还大。Llama 永远做不到这一点（它的 `head_dim` 就是 `hidden_size // heads`，
   见 `configuration_llama.py:87-88`）。
2. 而 `query_pre_attn_scalar` **恰好等于 `hidden_size // num_attention_heads`**（9B: 224 ✓、27B: 144 ✓）。
   换句话说是：**Q/K/V 用 256（或 128）维做投影，但 attention 的缩放系数仍然按「理想 head_dim」算**。
   - 9B：`1/sqrt(224) > 1/sqrt(256)` → attention logits 比 Llama 惯例**更大**（约 7% 放大）；
   - 27B：`1/sqrt(144) < 1/sqrt(128)` → 比 Llama 惯例**更小**（约 6% 缩小）。

   这个差异不会改变模型正确性（权重是训练出来的），但**任何「把 Gemma2 的 scaling 按 head_dim 重算」的推理优化都会算错。**

> 顺带一个容易读漏的后果：Gemma 2 的 `base_model_tp_plan`（`modular_gemma2.py:81-89`）与
> Llama 的（`llama/configuration_llama.py:49-57`）**逐行相同**，都是显式列 `q/k/v/o/gate/up/down`；
> 但含义不同——Llama 的 `q_proj` 输出是 `hidden_size`（因 `head_dim = hidden_size // heads`），
> 而 Gemma 2 的 `q_proj` 输出是 `num_heads × head_dim`（9B 时 = 4096 ≠ 3584）。
> 所以「q_proj colwise 切分后各 rank 的本地头数」在两家是不同的量，做并行/切分时需要按 `num_heads × head_dim` 算。

### 6.4 `Gemma2RMSNorm`：`(1 + weight)` 且 weight 初始化为 0

`Gemma2RMSNorm` 在 modular 里就一句（`modular_gemma2.py:139-140`）：

```python
class Gemma2RMSNorm(GemmaRMSNorm):
    pass
```

真正的实现在 Gemma 1（`modular_gemma.py:119-136`）：

```python
def forward(self, x):
    output = self._norm(x.float())
    # Llama does x.to(float16) * w whilst Gemma is (x * w).to(float16)
    output = output * (1.0 + self.weight.float())    # ← 1 + weight，不是 weight
    return output.type_as(x)
```

配合初始化（`modular_gemma.py:159-167` 的 `GemmaPreTrainedModel._init_weights`）：

```python
if "RMSNorm" in module.__class__.__name__:
    init.zeros_(module.weight)      # ← 初始化为 0，于是 (1 + 0) = 1，与 Llama 等价
```

| | Llama | Gemma 2 |
|---|---|---|
| 公式 | `weight * x` | `(1 + weight) * x` |
| weight 初始化 | 1（`_init_weights` 里 `init.ones_`） | **0** |
| 数学等价性 | — | 初始化时等价；训练后参数化不同（Gemma 的 weight 是「相对 1 的偏移量」） |

Gemma2 的 `_init_weights` 里这条规则同样生效（生成文件 `modeling_gemma2.py:371-377`），
因为它继承了 `GemmaPreTrainedModel`（`modular_gemma2.py:318-319` 是个 `pass`）。

### 6.5 `use_bidirectional_attention`（一句）

`Gemma2Config.use_bidirectional_attention`（`modular_gemma2.py:120`，默认 `None` → 视为 False）
让 `is_causal = False`（`modular_gemma2.py:224`），此时 `create_causal_mask` /
`create_sliding_window_causal_mask` 会走双向分支（`masking_utils.py:1154-1165`，
`return create_bidirectional_sliding_window_mask(...)`）。
用途是「把 decoder-only 模型当 encoder 用」（例如嵌入/掩码建模），
默认关闭，读 Gemma2 时不必深究。

---

## 七、modular 视角：哪些是本模型写的，哪些是自动拉取的（对照 2A5）

`modular_gemma2.py` 里 `^class ` 计数 = **11**，生成文件 `modeling_gemma2.py` 里 `^class ` 计数也 = **11**，
但两份**不是一一对应**：

| 生成文件里的类 | 行号 | modular 里有吗 | 来源 |
|---|---|---|---|
| `Gemma2RMSNorm` | `:49` | ✅ `modular_gemma2.py:139`（`pass` 继承 `GemmaRMSNorm`） | 显式 |
| `Gemma2MLP` | `:69` | ✅ `:143` | 显式（改 1 行：`act_fn`） |
| `Gemma2RotaryEmbedding` | `:85` | ✅ `:149` | 显式重写整块（**代码与 Llama 版逐行相同**，只是类型注解换成 `Gemma2Config`，并保留 `@deprecate_kwarg`） |
| `eager_attention_forward`（函数） | `:184` | ✅ `:183` | 显式（**Gemma 2 独有**：多了 `softcap`） |
| `Gemma2Attention` | `:219` | ✅ `:217` | 显式（soft-cap + scaling + layer_type） |
| `Gemma2DecoderLayer` | `:291` | ✅ `:271` | 显式（4 个 norm） |
| `Gemma2PreTrainedModel` | `:353` | ✅ `:318`（`pass`） | 显式 |
| `Gemma2Model` | `:381` | ✅ `:322` | 显式（双 mask 映射） |
| `Gemma2ForCausalLM` | `:466` | ✅ `:392` | 显式（final logit soft-cap） |
| `Gemma2ForSequenceClassification` | `:545` | ✅ `:460`（`pass`） | 显式 |
| `Gemma2ForTokenClassification` | `:549` | ✅ `:464`（`pass`） | 显式 |
| **`Gemma2TextScaledWordEmbedding`** | **`:338`** | ❌ **`grep -c "TextScaled"` = 0** | **自动拉取**（等价 `pass`） |
| `rotate_half` / `apply_rotary_pos_emb` / `repeat_kv`（函数） | `:139/147/172` | ❌ 未提及 | 自动拉取 |
| `Gemma2Config` | —（落到 `configuration_gemma2.py:28`） | ✅ `modular_gemma2.py:58` | 显式，但**生成到另一个文件**（2A5 的 `TYPE_TO_FILE_TYPE` 规则） |

**所以 modular 有 11 个类、modeling 有 11 个类、多出来的那个是自动拉取的、少掉的那个跑去 configuration 里了。**

另外两条 2A5 规则在本文件里也能实证（命令与输出见文末「验证清单」A 段）：
生成文件里 `grep -c "Copied from"` = **0**；生成文件里**没有任何**跨模型 `from ..<model>.modeling_*` import，
而 `modular_gemma2.py:38` 有 `from ..gemma.modeling_gemma import (`。

**一个能肉眼看到「全局改名」痕迹的彩蛋**：`modular_gemma.py:173` 的注释写的是
「Gemma3 downcasts the below to bfloat16, causing sqrt(3072)=55.4256 to become 55.5」，
经过 `Gemma` → `Gemma2` 的全局替换后，生成文件里变成了

```
modeling_gemma2.py:387:        # Gemma23 downcasts the below to bfloat16, causing sqrt(3072)=55.4256 to become 55.5. ...
```

——`Gemma3` 被改成了 `Gemma23`。纯注释瑕疵，但正好是 2A5 讲的「`ReplaceNameTransformer` 做全局字符串替换」
的活证据（替换不区分「是名字还是注释里的词」）。

---

## 八、Llama / Mistral / Gemma 2 三方对照速查

| 模块 | Llama（1M1） | Mistral（1M2） | **Gemma 2** |
|---|---|---|---|
| DecoderLayer 骨架 | Pre-Norm + 双残差 | 同 Llama | **Pre+Post 双 Norm**（4 个 norm） |
| norm 数量 / 层 | 2 | 2 | **4**（`modular_gemma2.py:278-282`） |
| Norm 类型 | `LlamaRMSNorm`（`weight*x`，w 初始 1） | 同 | `Gemma2RMSNorm`（`(1+weight)*x`，w 初始 0，`modular_gemma2.py:139`） |
| 位置编码 | RoPE | 同 | 同（`Gemma2RotaryEmbedding` 代码与 Llama 版相同） |
| 注意力 | MHA + GQA，`scaling = head_dim**-0.5` | 同 | GQA，**`scaling = query_pre_attn_scalar**-0.5`**（`:225`） |
| `head_dim` | `hidden_size // heads`（不可解耦） | 可显式给 | **可显式给且与 heads 无关**（9B: 224 vs 256） |
| 注意力窗口 | 无（全 causal） | **全层 SWA `W=4096`** | **交替**：13 层 `W=4096` + 13 层全因果（`:123-126`） |
| 层类型机制 | 无 `layer_types` | 无（即全 sliding） | **`config.layer_types`**（`__post_init__` 预计算） |
| mask 数量 | 1 个 causal | 1 个 sliding-causal | **2 个**，按 `layer_types[i]` 逐层查表（`:377`） |
| KV cache | `DynamicLayer` 全部层 | `DynamicSlidingWindowLayer` 全部层 | **混合**：一半 `DynamicLayer`、一半 `DynamicSlidingWindowLayer`（`cache_utils.py:1224-1227`） |
| attention logits soft-cap | 无 | 无 | **有**，`c=50`（`:202-205`，softmax 前） |
| 输出 logits soft-cap | 无 | 无 | **有**，`c=30`（`:442-445`，loss 之前） |
| soft-cap 后端兼容 | — | — | eager ✅ / flash_2 ✅ / flex ✅ / **sdpa ❌ 静默失效** |
| MLP | SwiGLU（`silu`，`config.mlp_bias`） | 同 | **GeGLU（`gelu_pytorch_tanh`）**，三投影强制无 bias |
| 激活字段名 | `hidden_act` | `hidden_act` | **`hidden_activation`**（`:103`） |
| embedding 缩放 | 无 | 无 | **`Gemma2TextScaledWordEmbedding`，× `sqrt(hidden_size)`**（`modeling_gemma2.py:387`） |
| embedding 参数绑定 | 默认 `False`（`llama/configuration_llama.py:79`） | 默认 `False`（`mistral/configuration_mistral.py:78`） | **默认 `True`**（`:111`，与 Gemma 1 一致） |
| `attention_bias` | False | False | **False**（`:113`） |
| 双向注意力开关 | 无 | 无 | `use_bidirectional_attention`（`:120`） |
| 继承来源 | 无（祖宗） | Llama | **Gemma 1 → Llama**（`modular_gemma2.py:38`） |

---

## 一句话总结

> **Gemma2 = Llama + 交替 SWA（13 local + 13 global）+ logit soft-cap（attention 处 c=50 / 输出处 c=30）+ Pre+Post 双 norm + GeGLU。**
> 它把三件事同时做进了同一份 Llama 骨架里：
> 1. 用 `config.layer_types` 让「一半的层」变成另一种注意力（attention / mask / cache 三处同步切换）；
> 2. 用 `c·tanh(x/c)` 把两处 logits 关进有界区间——**代价是它必须发生在 softmax 之前，于是只有原生支持的后端才认**；
> 3. 用「出口再 norm 一次」压住残差流的方差增长。
> 另外 `query_pre_attn_scalar` 提醒：**Gemma 2 的 attention scaling 不是 `1/sqrt(head_dim)`**，别按 Llama 的直觉去改。

---

## 验证清单

以下命令全部在仓库根目录 `/Users/baai/Desktop/baai_proj/transformers` 实跑过，输出即文中引用。

### A. 规模与生成关系

```bash
wc -l src/transformers/models/gemma2/modular_gemma2.py \
      src/transformers/models/gemma2/modeling_gemma2.py \
      src/transformers/models/gemma2/configuration_gemma2.py \
      src/transformers/models/gemma/modular_gemma.py
# → 475 / 559 / 111 / 268

grep -c "^class " src/transformers/models/gemma2/modular_gemma2.py   # → 11（含 Gemma2Config）
grep -c "^class " src/transformers/models/gemma2/modeling_gemma2.py  # → 11

grep -c "Copied from" src/transformers/models/gemma2/modeling_gemma2.py   # → 0
grep -nE "^from \.\.[a-z_0-9]+\.(modeling|configuration)" \
     src/transformers/models/gemma2/modeling_gemma2.py    # → 无匹配
grep -nE "^from \.\.[a-z_0-9]+\.(modeling|configuration)" \
     src/transformers/models/gemma2/modular_gemma2.py
# → 38:from ..gemma.modeling_gemma import (

# 依赖自动拉取：modular 里 0 次提及，生成文件里存在
grep -c "TextScaled" src/transformers/models/gemma2/modular_gemma2.py   # → 0
grep -n "class Gemma2TextScaledWordEmbedding" src/transformers/models/gemma2/modeling_gemma2.py
# → 338:class Gemma2TextScaledWordEmbedding(nn.Embedding):

# 全局改名的痕迹：modular 注释写 "Gemma3"，生成文件变成 "Gemma23"
sed -n '173p' src/transformers/models/gemma/modular_gemma.py          # → # Gemma3 downcasts ...
sed -n '387p' src/transformers/models/gemma2/modeling_gemma2.py        # → # Gemma23 downcasts ...
```

### B. soft-cap 的两处作用位置

```bash
grep -n "softcap" src/transformers/models/gemma2/modular_gemma2.py
# → 191:    softcap: float | None = None,        （eager_attention_forward 形参）
# → 202-205: if softcap is not None: / / softcap / tanh / * softcap   ← 位置 1（softmax 之前）
# → 222:    self.attn_logit_softcapping = self.config.attn_logit_softcapping
# → 262:            softcap=self.attn_logit_softcapping,              （传给后端）
# → 442-445: if self.config.final_logit_softcapping is not None: ...  ← 位置 2（最终 logits）

sed -n '200,210p' src/transformers/models/gemma2/modular_gemma2.py    # 看 softcap → mask → softmax 的顺序
sed -n '438,450p' src/transformers/models/gemma2/modular_gemma2.py    # 看 softcap 在 loss_function 之前

grep -n "final_logit_softcapping\|attn_logit_softcapping" \
     src/transformers/models/gemma2/modular_gemma2.py
# → 118: final_logit_softcapping: float | None = 30.0
# → 119: attn_logit_softcapping: float | None = 50.0
```

### C. soft-cap 的后端兼容性（正文 §2.5 的依据）

```bash
grep -c "softcap" src/transformers/integrations/sdpa_attention.py     # → 0   ❗sdpa 不认
grep -n "softcap" src/transformers/integrations/flash_attention.py    # → 35, 86（透传）
grep -n "softcap" src/transformers/integrations/flex_attention.py     # → 269, 291-292（score_mod 里做）
grep -rln "softcap" src/transformers/integrations/sdpa_paged.py \
        src/transformers/integrations/eager_paged.py \
        src/transformers/integrations/flash_paged.py                  # → 无匹配（paged 全部不支持）

sed -n '79,89p' src/transformers/integrations/sdpa_attention.py
# → 签名里只有 **kwargs，softcap 被吞掉（不报错、不警告）

grep -n "supports_mapping\[.softcap.\]" src/transformers/modeling_flash_attention_utils.py
# → 662:    if supports_mapping["softcap"] and softcap is not None:
sed -n '226,245p' src/transformers/modeling_flash_attention_utils.py
# → flash_parameters = inspect.signature(flash_function).parameters  ← 支持与否是探测出来的

grep -n "applicable_attention = \"sdpa\"" src/transformers/modeling_utils.py
# → 1927:        applicable_attention = "sdpa" if requested_attention is None else requested_attention
#         ← 默认后端是 sdpa ⇒ 默认路径下 attention soft-cap 不生效
```

### D. 层类型判定与三个消费点

```bash
sed -n '122,128p' src/transformers/models/gemma2/modular_gemma2.py
# → __post_init__ 里生成 layer_types：
#   "sliding_attention" if bool((i + 1) % 2) else "full_attention"

# 用纯 python 复算（不 import transformers）
python3 -c "
lt = ['sliding_attention' if bool((i+1)%2) else 'full_attention' for i in range(26)]
print(lt); print(lt.count('sliding_attention'), lt.count('full_attention'))"
# → ['sliding_attention','full_attention',...] / 13 13

grep -n "layer_type\|sliding_window" src/transformers/models/gemma2/modular_gemma2.py
# → 219: self.layer_type = config.layer_types[layer_idx] ...
# → 226: self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None
# → 377: attention_mask=causal_mask_mapping[self.config.layer_types[i]],

grep -n "DYNAMIC_LAYER_TYPE_MAPPING\|sliding_attention\":" src/transformers/cache_utils.py | head
# → 1224-1227: {"full_attention": DynamicLayer, "sliding_attention": DynamicSlidingWindowLayer, ...}
grep -n "def get_layer_types_and_kwargs" src/transformers/cache_utils.py   # → 1700
grep -n "sliding_window = getattr(config" src/transformers/masking_utils.py # → 1180

sed -n '434,441p' tests/models/gemma2/test_modeling_gemma2.py
# → 测试断言：sliding 层是 DynamicSlidingWindowLayer 且 keys.shape[-2] == sliding_window - 1
```

### E. `query_pre_attn_scalar` 与 scaling

```bash
grep -n "query_pre_attn_scalar" src/transformers/models/gemma2/modular_gemma2.py \
                                src/transformers/models/gemma2/configuration_gemma2.py
# → 115 / 87 默认 256；modular:225 是唯一使用点：self.scaling = config.query_pre_attn_scalar**-0.5

grep -n "query_pre_attn_scalar" \
     src/transformers/models/gemma2/convert_gemma2_weights_to_hf.py
# → 54: gemma_9b_config 里 =224（head_dim=256）
# → 67: gemma_27b_config 里 =144（head_dim=128）

# 对照 Llama：scaling 走 head_dim
grep -n "self.scaling" src/transformers/models/llama/modeling_llama.py   # → 226: self.head_dim**-0.5
grep -n "self.scaling" src/transformers/models/gemma2/modular_gemma2.py  # → 225: query_pre_attn_scalar**-0.5

python3 -c "
for n,hs,nh,hd,q in [('9B',3584,16,256,224),('27B',4608,32,128,144)]:
    print(n,'hidden/heads=',hs//nh,'qpas=',q,'1/sqrt(qpas)=',round(q**-0.5,6),'1/sqrt(head_dim)=',round(hd**-0.5,6))"
# → 9B hidden/heads=224 1/sqrt=0.066815 vs 0.0625; 27B hidden/heads=144 1/sqrt=0.083333 vs 0.088388
```

### F. 双 norm / GeGLU / embedding

```bash
grep -n "layernorm\|feedforward" src/transformers/models/gemma2/modular_gemma2.py
# → 278-282: input / post_attention / pre_feedforward / post_feedforward 四个 norm
sed -n '284,315p' src/transformers/models/gemma2/modular_gemma2.py      # 4 次 norm 的 forward 全文
sed -n '305,324p' src/transformers/models/llama/modeling_llama.py       # Llama 只有 2 次，对照用

grep -n "hidden_activation\|act_fn" src/transformers/models/gemma2/modular_gemma2.py
# → 103: hidden_activation: str = "gelu_pytorch_tanh"
# → 146: self.act_fn = ACT2FN[config.hidden_activation]
grep -n "gelu_pytorch_tanh" src/transformers/activations.py            # → 330, 365
grep -n "hidden_act: str" src/transformers/models/llama/configuration_llama.py  # → 70: "silu"

grep -c "normalizer" src/transformers/models/gemma/modeling_gemma.py   # → 0（已改名/重构）
git log -S "normalizer = torch.tensor" --oneline -- src/transformers/models/gemma/modeling_gemma.py
# → 0e7cb4e3ea（删除）与 8e2fc52ea3（当初加入）
git log -S "GemmaTextScaledWordEmbedding" --oneline -- src/transformers/models/gemma/modeling_gemma.py
# → 0e7cb4e3ea Make paligemma embed tokens standard (#44432)
sed -n '105,116p' src/transformers/models/gemma/modular_gemma.py       # 缩放 embedding 本体
sed -n '387,389p' src/transformers/models/gemma2/modeling_gemma2.py     # Gemma2 里使用它
```

> 备注：本机 python 环境缺 `torch`，无法真正 `import transformers` 跑 config 实例化，
> 所以 C/D/E 段里所有数值都用 **纯 python 复算**（不依赖库）或 **源码逐行核对** 得到；
> 凡是引用的行号都来自上面这些 `grep -n` / `sed -n` 的原始输出。**没有跑任何模型推理。**

---

## 导航

- 路线图：[00_roadmap.md](./00_roadmap.md)（本篇 = 进阶 · 模型结构 · 2M1）
- 上一篇：[1M5_bert_modernbert.md](./1M5_bert_modernbert.md)
- 下一篇：[2M2_deepseek_mla_moe.md](./2M2_deepseek_mla_moe.md)
- **配套回读**：
  - 想补 soft-cap 的「后端分发」细节 → [2A1_attention_backends.md](./2A1_attention_backends.md)（`ALL_ATTENTION_FUNCTIONS` 如何选实现）
  - 想补「两个 mask 是怎么建出来的」 → [2A2_masking_system.md](./2A2_masking_system.md)（`LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING` 与 `create_mask_mapping`）
  - 想补「混合 cache 怎么逐层分发」 → [2A3_kv_cache.md](./2A3_kv_cache.md)（`DynamicSlidingWindowLayer` / `get_layer_types_and_kwargs`）
  - 想补「modular 为什么能生出这个文件」 → [2A5_modular_code_generation.md](./2A5_modular_code_generation.md)
