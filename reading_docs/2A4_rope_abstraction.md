# RoPE 抽象层精读：`ROPE_INIT_FUNCTIONS` 与十来种 scaling

> 阅读目标：搞清楚库如何用**一张字典**统一「同一个 RoPE 的十几种变体」，以及 config 里那坨 `rope_type` / `factor` / `beta_fast`
> 到底怎么变成 `cos` / `sin`。
>
> **这是 1M1 的「抽象层升级版」**：1M1 第 2 节讲**数学**（`inv_freq` 怎么算、Q/K 怎么被旋转），本篇讲**工程**——
> Llama / Phi-3 / DeepSeek / Llama-3.1 / Gemma-4 各自改了哪一步，库把这些改动收敛成一个注册表 + 一个 config 校验器 + 一个装饰器。
> **所有 scaling 方法改的都只是「`inv_freq` 怎么算」和「`cos`/`sin` 乘不乘系数」两件事。**
>
> 源文件：`src/transformers/modeling_rope_utils.py`（1090 行）
> 人类向文档：`docs/source/en/internal/rope_utils.md`（83 行，**已过时**，见 §6.8）
> 相关：`src/transformers/configuration_utils.py`（config 侧接线）、`tests/utils/test_modeling_rope_utils.py`（1575 行，全部数值行为的裁判）

---

## 速读区（TL;DR）

**一句话结论**：库用一张只有 6 个 key 的字典 `ROPE_INIT_FUNCTIONS` 统一「`inv_freq` 怎么算」，
config 侧再用 `convert` / `standardize` / `validate` 三件套做改名、补字段、校验；
**所有 scaling 方法改的都只是「`inv_freq` 怎么算」和「`cos`/`sin` 乘不乘系数」这两件事。**

| # | 断言 | 出处（文件:行号） |
|---|------|------------------|
| 1 | `ROPE_INIT_FUNCTIONS` 只有 6 个 key（`linear`/`dynamic`/`yarn`/`longrope`/`llama3`/`proportional`），**没有 `"default"`** | `modeling_rope_utils.py:665-675` |
| 2 | `default` 走另一条路：每个模型自己的 `compute_default_rope_parameters` | `modeling_llama.py:83-85`、`:92-110` |
| 3 | 模型侧唯一分发模式是 `ROPE_INIT_FUNCTIONS[self.rope_type]`（143 个文件；per-layer 版另有 35 个） | `modeling_llama.py:85` |
| 4 | `rope_scaling` 已改名 `rope_parameters`，旧名只剩 property 别名；顶层 `config.rope_theta` 不再可访问 | `configuration_utils.py:548-554`、`MIGRATION_GUIDE_V5.md:513` |
| 5 | `rope_config_validation` **已废弃**成薄壳：只剩一条 `FutureWarning` + 转调，全库无任何调用者 | `modeling_rope_utils.py:1075-1089` |
| 6 | `validate_rope` 靠 `@strict`（命名约定，`validate_` 前缀）在每次 `__init__` 后**隐式**调用，不是 base `__init__` 调的 | `configuration_utils.py:146`、`:823-826`、`huggingface_hub/dataclasses.py:227-232` |
| 7 | `dynamic_rope_update` 只在 `"dynamic" in rope_type`（**子串匹配**）或 `rope_type == "longrope"` 时动手，其余 type 纯静态 | `:124-126` |
| 8 | 重算触发条件是 `seq_len > max_seq_len_cached`，且 `seq_len = torch.max(position_ids) + 1`；但**数值真正变化的门槛是 `config.max_position_embeddings`** | `:88`、`:100`、`:329-337` |
| 9 | 第二个返回值（`attention_scaling`）**只有 yarn / longrope 会非 1.0**；llama3 恒为 `1.0`（所以 llama3 ≠ yarn + mscale） | `:417-444`（yarn）、`:563-567`（longrope）、`:639`（llama3） |
| 10 | `proportional` 用 `inv_freq = 0` 实现「不旋转」，不改 forward、不改 `head_dim` | `:239-266` |
| 11 | `rotate_half` 有两个流派，**数学等价（维度置换 `P=(0,2,1,3)`）但权重不通用** | `modeling_llama.py:130-152` vs `modeling_glm_moe_dsa.py:125-161` |
| 12 | 缺必填键 → `raise KeyError`（硬失败）；多了不认识的键 / 值范围不对 → 只 `logger.warning` | `:1062-1072` |

**最容易踩的坑 / 反直觉发现**

- **「default 不在字典里」**：config 的合法 `rope_type` 有 7 个，注册表只有 6 个——`default` 由模型自带的静态方法兜底，因此「换个模型 `default` 的含义可能微调」。
- **`original_max_position_embeddings` 缺失会被静默兜底**成 `max_position_embeddings`（`:790-798`），它决定「哪些维度算低频」，填错就是整套分段按错的边界切——**而 yarn 的校验器偏偏不报**（`:926-927` 显式排除 `implicit_factor == 1`），llama3 只是 warning（`:1024-1028`）。必须手写。
- **「超出训练长度就重算」是错的表述**：`dynamic` 内部有 `max(seq_len, max_position_embeddings)` 兜底，所以超过 `max_seq_len_cached` 但没超过 `max_position_embeddings` 时只是**白算一遍**。
- **`longrope_frequency_update` 里 `hasattr` 检查的名字和 `setattr` 写的名字不一致**（`"None_long_inv_freq"` vs `"long_inv_freq"`），per-layer 路径第二次进入长序列分支会 `UnboundLocalError`——目前没有模型触发（`:64-74`，静态阅读发现）。
- **yarn 的 `truncate` 读的是顶层 `config.rope_parameters`**（`:474`），不是本函数顶部统一取的 per-layer 子字典——嵌套 config 下会找错地方。
- **`mscale` / `mscale_all_dim` 不参与 `inv_freq`**，只决定乘在 `cos`/`sin` 上的标量；所以 DeepSeek-V4 要显式把 `attention_factor` 钉成 1.0（`configuration_deepseek_v4.py:300-303`）。
- **两个流派混用没有任何报错信号**：无异常、无 shape mismatch、无 nan，只是输出彻底崩坏（困惑度爆表 / 复读 / 胡言，`:1247-1253`）。
- **`dynamic` / `longrope` 与 `torch.compile` 有已知冲突**：代码自带 `# TODO joao: may break with compilation`（`:108`）。
- 官方文档 `docs/source/en/internal/rope_utils.md` **已过时**：`ROPE_VALIDATION_FUNCTIONS` 这个符号根本不存在；type 表漏了 `proportional`（`:940-949`）。

**读完应该能回答的问题**

- 为什么 `"default"` 不在 `ROPE_INIT_FUNCTIONS` 里？给模型加新 RoPE 有哪两条路？
- `dynamic` 的 `inv_freq` 在什么条件下**真正**变化？和 `max_seq_len_cached`、`max_position_embeddings` 分别是什么关系？
- `llama3` 比 `linear` 好在哪？「按波长分三段」的两个边界值（λ=2048 / λ=8192）分别怎么来的？
- 两派 `rotate_half` 为什么数学等价却不能共用权重？混用的症状是什么？
- `validate_rope` 是谁调用的？为什么「改个方法名就失效」？`rope_type` 拼错会怎样？

---

## 前置知识

- [1M1_llama_structure.md](./1M1_llama_structure.md)：**必读**。第 2 节的 `LlamaRotaryEmbedding` / `inv_freq` / `rotate_half` / `apply_rotary_pos_emb` 是本篇的全部地基。
- [2A5_modular_code_generation.md](./2A5_modular_code_generation.md)：§7.5 用到它的「依赖自动拉取」。

---

## 一、先对齐一个前提：这个仓库是 **v5 语法**

| 旧（≤ v4） | 新（本仓库） | 证据 |
|-----------|-------------|------|
| `config.rope_scaling` | `config.rope_parameters` | 全库只在兼容层里出现 `rope_scaling`（4 处，见 §6.7） |
| `config.rope_theta`（顶层属性） | `config.rope_parameters["rope_theta"]` | `modeling_llama.py:103` 读的是 dict 里的键 |
| `rope_config_validation(config)` | `config.validate_rope()` | `modeling_rope_utils.py:1075-1089`（**已废弃，只剩转调**） |
| `ROPE_VALIDATION_FUNCTIONS` | `ROPE_INIT_FUNCTIONS` | 旧名只活在过时文档里 |

官方迁移说明在 `MIGRATION_GUIDE_V5.md:513`：所有 RoPE 参数现在都放在 `mode.rope_parameters` 下（含 `rope_theta` 和 `rope_type`），
**试图访问 `config.rope_theta` 会直接抛 attribute error**。**但兼容层还在**——**`rope_scaling` 不是不存在，
它变成了 `rope_parameters` 的一个 property 别名**（`configuration_utils.py:548-554`）：

```python
# configuration_utils.py:548-554
    @property
    def rope_scaling(self):
        return self.rope_parameters

    @rope_scaling.setter
    def rope_scaling(self, value):
        self.rope_parameters = value
```

---

## 二、问题的起源：一个数学骨架，十几种改法

1M1 里的 RoPE 核心只有几行：`inv_freq[i] = 1 / (theta ** (2i / dim))`（i = 0..dim/2-1）→ `freqs = inv_freq ⊗ position_ids`
→ `cos, sin = freqs.cos(), freqs.sin()` → `q_rot = q * cos + rotate_half(q) * sin`。
它在**训练长度内**完美，一推理到更长就崩：`position_ids` 超出训练范围后 `freqs` 落到模型没见过的角度上。
2023-2024 的「上下文扩展」论文每篇改的**都是上面几行里的一行**：

| 改哪一行 | 方法 | 直白说法 |
|----------|------|----------|
| `inv_freq /= factor` | **linear** | 把所有位置「压缩」进训练范围 |
| 改 `theta`，让 `inv_freq` 整体变小 | **dynamic (NTK)** | 不动位置，改底数，且按当前序列长度实时调整 |
| 前两者**分段混合** | **yarn** | 低频插值、高频外推，中间平滑过渡 |
| 多组 `factor` 数组，按长度切换 | **longrope** | 短上下文用 `short_factor`，长上下文用 `long_factor` |
| 按**波长**分三段处理 | **llama3** | Meta 的做法，最巧妙，见 §4.6 |
| 只旋转前 1/4 个 head_dim | **proportional** | Gemma-4：剩下的维度不旋转（NoPE） |

**解法**：库抽出「**签名统一的计算函数**」这一层。`grep -rl "ROPE_INIT_FUNCTIONS" src/transformers/models/ | sed 's|.*models/||;s|/.*||' | sort -u | wc -l` → **154**
（否则就是 154 个模型目录 × 各自一套 `inv_freq` 代码）。

---

## 三、统一抽象：`ROPE_INIT_FUNCTIONS`

### 3.1 真实定义与**完整 key 列表**

`src/transformers/modeling_rope_utils.py:665-675`：

```python
# modeling_rope_utils.py:665-675（"You can append new {'rope_type': callable} pairs … as long as the callable has the same signature"）
ROPE_INIT_FUNCTIONS: dict[str, Callable[..., tuple["torch.Tensor", float]]] = {
    "linear": _compute_linear_scaling_rope_parameters,  "dynamic": _compute_dynamic_ntk_parameters,
    "yarn": _compute_yarn_parameters,                   "longrope": _compute_longrope_parameters,
    "llama3": _compute_llama3_parameters,               "proportional": _compute_proportional_rope_parameters,
}
```

**key 就这 6 个**（AST 解析确认，见验证清单第 1 条）：`linear`→`:133`、`dynamic`→`:269`、`yarn`→`:345`、`longrope`→`:486`、`llama3`→`:580`、`proportional`→`:193`。

### 3.2 最重要的一个「不存在」：没有 `"default"` 这个 key

`rope_type` 合法值有 7 个（`RopeParameters` docstring，`:685-686` 列举 `['default', 'linear', 'dynamic', 'yarn', 'longrope', 'llama3']`，
加上后加的 `proportional`），但字典里只有 6 个。`"default"` 走**另一条路**——每个模型自己实现的
`compute_default_rope_parameters`。所以所有模型里都有这段**一模一样的三行**：

```python
# modeling_llama.py:83-85
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)
```

**为什么这样设计**：`default` 就是要让模型**保留自定义权**。Llama 的版本（`modeling_llama.py:93-110`）用
`dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads`，**完全不理 `partial_rotary_factor`**；
GLM 的版本（`modeling_glm_moe_dsa.py:88-103`）长得一样，但它的 `config.head_dim` 被 config 显式指到了 rope 切片上（见 §8.4）。
换个模型 `default` 的含义就可能微调——所以库不把它放进公共字典。

### 3.3 分发链

```
 config.rope_parameters（"rope_type" / "factor" / …）
   │ PreTrainedConfig.__init__  configuration_utils.py:313-320（BC 钩子：老键 rope_scaling 在这里被收编）
   ▼
 convert_rope_params_to_dict :742 → rope_scaling→rope_parameters :743-745；setdefault("rope_theta") :752-753（default_theta = 10000.0, :739）
   └ standardize_rope_params :765 → 老 "type" → 新 "rope_type" :785；为 llama3/yarn/longrope 补 original_max_position_embeddings :791-798
   │ @strict（huggingface_hub）在每次 __init__ 后自动调用 validate_*（见 §6.6）
   ▼
 validate_rope :818 → 按 rope_type 找 _validate_<type>_rope_parameters :839-848
   ▼
 LlamaRotaryEmbedding.__init__  modeling_llama.py:73-89
   rope_type = config.rope_parameters["rope_type"] :82；rope_init_fn = compute_default_rope_parameters :83  ← default 走这里
   …else ROPE_INIT_FUNCTIONS[rope_type] :85  ← ★ 唯一分发点；inv_freq, attention_scaling = rope_init_fn(config, device) :86
   self.inv_freq = nn.Buffer(inv_freq, persistent=False) :88
   ▼
 forward(x, position_ids) → cos / sin（× attention_scaling）:112-128 → apply_rotary_pos_emb :138-152
```

**记住那个 ★**：全库模型侧取 RoPE 初始化函数**只有这一处模式**——`ROPE_INIT_FUNCTIONS[self.rope_type]`（143 个文件里），
per-layer 模型则是 `ROPE_INIT_FUNCTIONS[self.rope_type[layer_type]]`（另有 35 个文件）。
给模型加新 RoPE，要么往字典里加，要么覆盖 `compute_default_rope_parameters`。

### 3.4 统一签名

六个函数签名完全一致（`default` 的静态方法也是），这是整套抽象成立的前提：

```python
def _compute_xxx_parameters(
    config: "PreTrainedConfig",
    device: Optional["torch.device"] = None,
    seq_len: int | None = None,          # ← 只有 dynamic / longrope 真正会用
    layer_type: str | None = None,       # ← per-layer 模型（Gemma-3 系）才非 None
) -> tuple["torch.Tensor", float]:
    return inv_freq, attention_factor    # ← 第一个是 inv_freq，第二个是 cos/sin 的乘数
```

返回的第二个值在模型里被命名成 `attention_scaling`（`modeling_llama.py:86`），在 `forward` 里乘到 `cos`/`sin` 上（`:124-125`）。
**只有 yarn / longrope 会返回非 1.0 的值**。

---

## 四、逐个拆解

### 4.1 `default`：原始 RoPE（不在字典里）

```python
# modeling_llama.py:103-110
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        attention_factor = 1.0  # Unused in this type of RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        return inv_freq.to(device), attention_factor
```

就是把 1M1 的那一行原样写出来；`attention_factor = 1.0`（注释直说 unused）。

### 4.2 `linear`：最朴素的频率缩放

`_compute_linear_scaling_rope_parameters`（`:133-190`，参数只有 `factor` :174），核心两行：

```python
# modeling_rope_utils.py:184-190
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    # NOTE: originally, scaling was applied to the position_ids. However, we get `embs = inv_freq @ position_ids`, so
    # applying scaling to the inverse frequencies is equivalent.
    inv_freq /= factor
    return inv_freq, attention_factor
```

`freqs = inv_freq @ position_ids`，所以把 `inv_freq` 除以 `factor` 与把 `position_ids` 除以 `factor` **完全等价**，且不用改输入。
**代价**：所有频率无差别压缩，高频（短距离）分辨率也一起被压掉 → 短文本变差，直接催生了 yarn / llama3。

### 4.3 `dynamic`：NTK-aware，按长度实时调 `theta`

`_compute_dynamic_ntk_parameters`（`:269-342`，参数 `factor` :325）：

```python
# modeling_rope_utils.py:328-342
    # seq_len: default to max_position_embeddings, e.g. at init time
    if seq_len is None:
        seq_len = config.max_position_embeddings
    elif isinstance(seq_len, torch.Tensor):
        seq_len = torch.maximum(seq_len, torch.tensor(config.max_position_embeddings, ...))
    else:
        seq_len = max(seq_len, config.max_position_embeddings)

    base = base * ((factor * seq_len / config.max_position_embeddings) - (factor - 1)) ** (dim / (dim - 2))
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
```

不动 `position_ids`，而是放大底数 `theta`（`theta` 变大 → `inv_freq` 变小 → 所有角度变小，效果类似「拉伸」）。
当 `seq_len == max_position_embeddings` 时括号内为 1，**base 不变**（与 default 完全一致）。

| | linear | dynamic |
|---|---|---|
| 改什么 | `inv_freq /= factor`（固定） | 改 `theta`（**随 `seq_len` 变**） |
| `seq_len` 参数 | 不用（docstring 明写 "Unused"） | 用 |
| 短序列 | 一样被压缩（信息损失） | `max(seq_len, max_position_embeddings)` 兜底 → 不压缩 |

它需要配合 `dynamic_rope_update` 装饰器才真正「动」起来——见 §5。

### 4.4 `yarn`：参数最多，NTK + 分段插值

`_compute_yarn_parameters`（`:345-483`）。**步骤 1：确定 `factor` 和 `attention_factor`**（`:412-444`）：

```python
# modeling_rope_utils.py:417-444（节选）
    factor = rope_parameters_dict["factor"]
    attention_factor = rope_parameters_dict.get("attention_factor")
    mscale = rope_parameters_dict.get("mscale"); mscale_all_dim = rope_parameters_dict.get("mscale_all_dim")
    original_max_position_embeddings = rope_parameters_dict["original_max_position_embeddings"]
    if factor is None:   # NOTE: DeepSeek-V3 等用 max_position_embeddings / original_max_position_embeddings 的比值代替 factor
        factor = config.max_position_embeddings / original_max_position_embeddings
    def get_mscale(scale, mscale=1):
        return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0
    if attention_factor is None:
        attention_factor = (float(get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim))
                            if (mscale and mscale_all_dim) else get_mscale(factor))
```

`attention_factor` 可以显式给，也可以由 `factor` / `mscale` / `mscale_all_dim` **推**出来。反直觉的一点：
**`mscale` / `mscale_all_dim` 不参与 `inv_freq`**，只用来算那个乘在 `cos`/`sin` 上的标量。这就是为什么 DeepSeek-V4 的 config
要专门把 `attention_factor` 硬钉成 1.0（`configuration_deepseek_v4.py:300-303` 注释原文：V4 reference **不**把 cos/sin 乘上 YaRN
算出的 mscale，而留空会让 `_compute_yarn_parameters` 推出 `0.1·log(factor)+1 ≈ 1.277`）。

**步骤 2：找分段边界**（`:447-466`）——把「转多少圈」翻译成「第几个维度」：

```python
# modeling_rope_utils.py:447-466（三个内部函数）
    def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))
    def find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings, truncate):
        low = find_correction_dim(low_rot, ...); high = find_correction_dim(high_rot, ...)
        if truncate: low = math.floor(low); high = math.ceil(high)
        return max(low, 0), min(high, dim - 1)
    def linear_ramp_factor(min, max, dim):   # min == max 时 max += 0.001，防奇点
        return torch.clamp((torch.arange(dim, dtype=torch.float32) - min) / (max - min), 0, 1)
```

`beta_fast` / `beta_slow` 就是这里的 `low_rot` / `high_rot`——**单位是「旋转圈数」，不是维度**（默认 32 / 1，见 `:443-444`）。

**步骤 3：插值与外推按维度混合**（`:468-483`）：

```python
    pos_freqs = base ** (torch.arange(0, dim, 2).to(device=device, dtype=torch.float) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)
    truncate = config.rope_parameters.get("truncate", True)
    low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_max_position_embeddings, truncate)
    inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, dim // 2).to(device=device, dtype=torch.float)
    inv_freq = (inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
                + inv_freq_extrapolation * inv_freq_extrapolation_factor)
    return inv_freq, attention_factor
```

一句话：**高频维度不缩放（外推），低频维度除以 `factor`（插值），中间的维度按线性斜坡混合**——已经和 llama3 的「三段」同源，
只是 llama3 用的是**波长**而不是**维度序号**来分段。

> ⚠️ **阅读陷阱**：`:474` 读的是 `config.rope_parameters.get("truncate", True)`，而不是本函数顶部统一取的 `rope_parameters_dict`（`:410`）。
> 在 **per-layer**（嵌套）config 下，`truncate` 会被到顶层去找而不是当前 `layer_type` 的子字典里。目前没有模型这么用。

参数：`factor`、`attention_factor`、`mscale`、`mscale_all_dim`、`beta_fast`、`beta_slow`、`original_max_position_embeddings`、`truncate`。
**真实使用者**：GPT-OSS（`configuration_gpt_oss.py:81-88`，`factor=32.0, beta_fast=32.0, beta_slow=1.0, truncate=False, original_max_position_embeddings=4096`）、
Ministral-3（`configuration_ministral3.py:97-110`，带 `mscale`/`mscale_all_dim`，**用的是老键 `"type": "yarn"`**）、OpenAI-Privacy-Filter、DeepSeek-V2 / V3 / V4。

> **`attention_factor` 有两种消费方式**：多数模型乘在 `cos`/`sin` 上（`modeling_llama.py:124`）；DeepSeek-V2 把它折进注意力的 softmax scale：
> `modular_deepseek_v2.py:245-252` 的 `yarn_apply_mscale()`，调用点 `:311` `self.scaling = yarn_apply_mscale(config.rope_parameters, self.qk_head_dim ** (-0.5))`。

### 4.5 `longrope`：两套 factor 数组，按长度切换

`_compute_longrope_parameters`（`:486-577`），关键是「选数组」：

```python
# modeling_rope_utils.py:569-577
    if seq_len and seq_len > original_max_position_embeddings:
        ext_factors = torch.tensor(long_factor, dtype=torch.float32, device=device)
    else:
        ext_factors = torch.tensor(short_factor, dtype=torch.float32, device=device)
    inv_freq_shape = torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim
    inv_freq = 1.0 / (ext_factors * base**inv_freq_shape)
```

**与其它所有方法的根本差异**：`short_factor` / `long_factor` 是**逐维度的数组**（长度 `dim // 2`），不是标量，由论文搜索出来。
`attention_factor` 默认值也是 LongRoPE 专有公式（`:563-567`）：`factor <= 1.0` 时取 1.0，否则 `math.sqrt(1 + math.log(factor) / math.log(original_max_position_embeddings))`。
**真实使用者**：Phi-3 / Phi-MoE / Phi-4-MM——它们的 config 里有**改名兼容**（`configuration_phi3.py:102-104`）：`"su"` / `"yarn"` 会被悄悄改写成 `longrope`。

### 4.6 `llama3`：按**波长**分三段（重点）

`_compute_llama3_parameters`（`:580-662`）。前面都是「按维度序号分段」，llama3 换用**波长 `wavelen = 2π / inv_freq`**（该维度转一圈需要多少 token）：

```python
# modeling_rope_utils.py:644-660
    factor = rope_parameters_dict["factor"]  # `8` in the original implementation
    low_freq_factor = rope_parameters_dict["low_freq_factor"]  # `1`
    high_freq_factor = rope_parameters_dict["high_freq_factor"]  # `4`
    old_context_len = rope_parameters_dict["original_max_position_embeddings"]  # `8192`
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    wavelen = 2 * math.pi / inv_freq
    # wavelen < high_freq_wavelen: do nothing ; wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
    return inv_freq_llama, attention_factor
```

用 Llama-3.1 的真实常数（`factor=8, low_freq_factor=1, high_freq_factor=4, old_context_len=8192`）算出来：

| 分段 | 条件 | 缩放 | 边界值（λ = 2π/inv_freq） |
|------|------|------|--------------------------|
| ① 高频 | `wavelen < high_freq_wavelen` | **不动**（×1） | λ < 2048（= 8192/4） |
| ② 中频 | 介于两者之间 | **线性平滑** | 2048 ≤ λ ≤ 8192 |
| ③ 低频 | `wavelen > low_freq_wavelen` | **除以 factor**（×1/8） | λ > 8192（= 8192/1） |

```
 inv_freq 的乘数（1.0 = 原样；1/factor = 压缩）
   1.0 ┤━━━━━━━━━━━╲
       │  ① 高频     ╲   ② 中频：线性平滑过渡
 1/factor┤                ┗━━━━━━━━━━━━━━━━━━━━━━━
 (=0.125)│                    ③ 低频：inv_freq / factor
       └───────────┬──────────────┬──────────────► 波长 λ = 2π / inv_freq
              λ=2048          λ=8192
        high_freq_wavelen   low_freq_wavelen（= old_ctx / high_freq_factor、/ low_freq_factor）
```

**为什么这样最巧妙**：高频维度 = 短波长 = 负责区分相邻 token，训练长度内被反复「转过很多圈」，模型见过所有角度，
**不需要插值**，插了反而破坏短距离分辨率 → 不动；低频维度 = 长波长 = 负责长距离语义，训练内根本没转满一圈，
直接外推一定出界 → 老实做线性插值（÷ factor）；中间一段若用硬阈值切会让相邻维度一个被压一个不被压、破坏频率单调性，
所以用一条**连续斜坡**过渡（两端连续：λ = low_freq_wavelen 时 `smooth_factor = (8192/8192 - 1)/(4 - 1) = 0` → `inv_freq/factor` ✓ 接③；
λ = high_freq_wavelen 时 `smooth_factor = (8192/2048 - 1)/3 = 1` → `inv_freq` ✓ 接①）。

**注**：`attention_factor` 在这里恒为 `1.0`（`:639`）——llama3 **没有** attention scaling，只有频率缩放。**llama3 ≠ yarn + mscale**。
**真实使用者**：Llama-3.1 / 3.2 / Llama-Guard-3（`convert_llama_weights_to_hf.py:376-383`，`"factor": 32.0 if llama_version == "3.2" else 8.0`）、mllama、Apertus（`configuration_apertus.py:85-91`，`rope_theta=12000000.0`）、CWM、Higgs-Audio-v2、CSM。

### 4.7 `proportional`：最新的一个，只旋转一部分维度

`_compute_proportional_rope_parameters`（`:193-266`），给 Gemma-4 加的：

```python
# modeling_rope_utils.py:239-266（节选）
    rope_proportion = rope_parameters_dict.get("partial_rotary_factor", 1.0)
    attention_factor = 1.0  # Unused in this type of RoPE
    rope_angles = int(rope_proportion * head_dim // 2)
    inv_freq_rotated = 1.0 / (base ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / head_dim))
    nope_angles = head_dim // 2 - rope_angles
    if nope_angles > 0:
        inv_freq = torch.cat((inv_freq_rotated, torch.zeros(nope_angles, dtype=torch.float32, device=device)), dim=0)
    else:
        inv_freq = inv_freq_rotated
    inv_freq /= factor
    return inv_freq, attention_factor
```

**技巧**：不旋转的维度直接给 `inv_freq = 0`（`:258` 的 `torch.zeros`）——角度恒为 0，`cos=1, sin=0`，**旋转恒等**。
这样上层 `apply_rotary_pos_emb` 一行都不用改，`head_dim` 也保持原尺寸（docstring `:216-219` 特意强调）——**这是「用数据表示控制流」的例子**。
真实使用者：Gemma-4（`configuration_gemma4.py:204-208`）——全局注意力层用 `{"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1_000_000.0}`，
滑动窗口层用 `{"rope_type": "default", "rope_theta": 10_000.0}`，**同一个模型里两种 RoPE 并存**（见 §7.2）。

### 4.8 对比总表

| 方法 | 核心思想 | 改的东西 | 关键参数 | `attention_factor` | 典型使用者 |
|------|---------|---------|---------|-------------------|-----------|
| `default` | 原始 RoPE | —— | —— | 恒 1.0 | Llama 2、Mistral、Qwen2/3、GLM |
| `linear` | 频率整体除以 factor（位置插值） | `inv_freq` | `factor` | 恒 1.0 | 早期 Llama 长上下文微调 |
| `dynamic` | NTK-aware，按 `seq_len` 实时放大 `theta` | `theta` | `factor` | 恒 1.0 | Gemma-3 全局层、Nomic-BERT |
| `yarn` | 插值 + 外推按维度混合，外加 mscale | `inv_freq` + `cos/sin` 系数 | `factor`、`beta_fast`、`beta_slow`、`mscale`、`mscale_all_dim`、`truncate`、`original_max_position_embeddings`、`attention_factor` | 可推算 | GPT-OSS、Ministral-3、DeepSeek-V2/V3/V4 |
| `longrope` | 两套逐维度 factor 数组，按长度切换 | `inv_freq` | `short_factor[]`、`long_factor[]`、`factor`、`original_max_position_embeddings`、`attention_factor` | 专有公式 | Phi-3 / Phi-MoE / Phi-4-MM |
| `llama3` | 按**波长**三段：高频不动 / 中频平滑 / 低频插值 | `inv_freq` | `factor`、`low_freq_factor`、`high_freq_factor`、`original_max_position_embeddings` | 恒 1.0 | Llama-3.1/3.2、mllama、Apertus、CWM |
| `proportional` | 只旋转前 k 个维度，其余 `inv_freq=0` | `inv_freq`（补零） | `partial_rotary_factor`、`rope_theta`、`factor` | 恒 1.0 | Gemma-4 |

---

## 五、`dynamic_rope_update`：推理时按需重算 `inv_freq`

`modeling_rope_utils.py:34-130`。**本篇唯一一个「装饰器」式的抽象**。

```python
# modeling_rope_utils.py:120-130
    @wraps(rope_forward)
    def wrapper(self, x, position_ids, layer_type=None):
        rope_type = self.rope_type if layer_type is None else self.rope_type[layer_type]
        kwargs = {"layer_type": layer_type} if layer_type is not None else {}
        if "dynamic" in rope_type:
            dynamic_frequency_update(self, position_ids, device=x.device, **kwargs)
        elif rope_type == "longrope":
            longrope_frequency_update(self, position_ids, device=x.device, **kwargs)
        return rope_forward(self, x, position_ids, **kwargs)

    return wrapper
```

它做两件事：① **判断要不要更新**——只在 `"dynamic" in rope_type`（**子串匹配**，目前七个 type 里只有 `"dynamic"` 含该子串，
等价于 `==`，但将来若加 `"dynamic_xxx"` 会自动纳入）或 `rope_type == "longrope"` 时动手；其余 type 完全纯静态。
② **调用真正的 forward**（`rope_forward`），透传 `layer_type`。
用法（`modeling_llama.py:111-113`）：`@torch.no_grad()` + `@dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)`
（该注释全库统一）；`@wraps` 保证 `forward` 的 docstring / 名字不被破坏。

### 5.1 触发条件（逐行核对）

```python
# modeling_rope_utils.py:82-118（节选）
    # docstring: 1 - growing beyond the cached sequence length (allow scaling)
    #            2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
    seq_len = torch.max(position_ids) + 1
    if seq_len > max_seq_len_cached:  # growth
        inv_freq, self.attention_scaling = ROPE_INIT_FUNCTIONS[rope_type](self.config, device, seq_len=seq_len, layer_type=layer_type)
        # TODO joao: may break with compilation
        self.register_buffer(f"{prefix}inv_freq", inv_freq, persistent=False)
        setattr(self, f"{prefix}max_seq_len_cached", seq_len)
    if seq_len < self.original_max_seq_len and max_seq_len_cached > self.original_max_seq_len:  # reset
        original_inv_freq = original_inv_freq.to(device)
        self.register_buffer(f"{prefix}inv_freq", original_inv_freq, persistent=False)
        setattr(self, f"{prefix}original_inv_freq", original_inv_freq); setattr(self, f"{prefix}max_seq_len_cached", self.original_max_seq_len)
```

| 问题 | 答案 | 依据 |
|------|------|------|
| 在哪触发？ | 每次 `forward` 开头，装饰器里 | `:124-127` |
| 用什么算「当前长度」？ | `seq_len = torch.max(position_ids) + 1`，**是最大位置 +1，不是张量长度** | `:88` |
| 增长分支条件？ | `seq_len > max_seq_len_cached` | `:100` |
| `max_seq_len_cached` 初值？ | `config.max_position_embeddings` | `modeling_llama.py:77` |
| 重算什么？ | 用 `seq_len` 调 `rope_init_fn`，**同时更新 `self.attention_scaling`** | `:101-107` |
| 怎么落地？ | `register_buffer(f"{prefix}inv_freq", inv_freq, persistent=False)` | `:109` |
| 会写进 state_dict 吗？ | **不会**，`persistent=False` | `:109` |
| 回退分支？ | `seq_len < original_max_seq_len` **且** `max_seq_len_cached > original_max_seq_len` → 恢复 `original_inv_freq` | `:112-118` |
| 为什么要有回退？ | 长序列跑过后缓存被放大，短序列再进来时若不复位，精度会白白损失 | docstring `2 - avoid losing precision with small sequences` |

**一个必须澄清的细节**：`seq_len > max_seq_len_cached` 是**触发重算**的条件，但**实际数值变化**的门槛更高——
`_compute_dynamic_ntk_parameters` 内部有兜底（`:329-337`）`seq_len = max(seq_len, config.max_position_embeddings)`，
所以 `seq_len ≤ max_position_embeddings` 时算出的 `inv_freq` 与初始化时完全一致（`base` 不变）。

> 结论：**「超出训练长度就重算」**要精确表述为**「超出 `config.max_position_embeddings` 时 `inv_freq` 才真正变化」**；
> 超过 `max_seq_len_cached` 但没超过 `max_position_embeddings` 时只是白算一遍。

**已知代价**（代码里自己写的 TODO，`:108`）：`forward` 里根据**数据相关**的 `seq_len` 做 Python 分支、还现场 `register_buffer` 换 buffer，
这两个都是 `torch.compile` 最不喜欢的形态。所以 `dynamic` / `longrope` 与 `torch.compile` 一起用时，得留意序列长度边界处是否被重新编译。

### 5.2 `longrope` 分支：一段可疑代码

`longrope_frequency_update`（`:47-80`）用「长序列 → 用 `long_factor` 重算，短序列 → 用回 `original_inv_freq`」：

```python
# modeling_rope_utils.py:64-74
        if seq_len > original_max_position_embeddings:
            if not hasattr(self, f"{layer_type}_long_inv_freq"):
                rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
                long_inv_freq, _ = rope_init_fn(self.config, device, seq_len=original_max_position_embeddings + 1, layer_type=layer_type)
            self.register_buffer(f"{prefix}inv_freq", long_inv_freq, persistent=False)
            setattr(self, f"{prefix}long_inv_freq", long_inv_freq)
```

> ⚠️ **静态阅读发现的两个不一致点（未被任何当前模型触发，仅作备忘）**：
> 1. `layer_type is None` 时 `prefix == ""`，但 `hasattr` 检查的是 `f"{layer_type}_long_inv_freq"` = `"None_long_inv_freq"`，
>    而 `setattr` 写的是 `"long_inv_freq"`——**检查与写入的名字不一致**，这个「只初始化一次」的缓存**永远命不中**。
> 2. `layer_type` 非 `None` 时两者名字一致，于是**第二次**进入长序列分支时 `if` 被跳过，`long_inv_freq` 从未被赋值，
>    第 73 行引用它 → `UnboundLocalError`。这条路径要求「`rope_type="longrope"` + per-layer `rope_parameters` + 长序列走两次」，
>    目前仓库里没有这样的模型（`longrope` 只出现在 `tests/utils/test_modeling_rope_utils.py`，而那里直接调 `rope_fn(...)`，不经过 `forward`）。

---

## 六、config 侧：`RopeParameters` 与 `RotaryEmbeddingConfigMixin`

### 6.1 `RopeParameters`（TypedDict，`:678-731`）

字段（`:720-731`）：`rope_theta`、`rope_type`、`partial_rotary_factor`、`factor`、`original_max_position_embeddings`、`attention_factor`、
`beta_fast`、`beta_slow`、`short_factor`、`long_factor`、`low_freq_factor`、`high_freq_factor`。

> 注意：这个 TypedDict **不是白名单**。校验器实际接受的键还要多几个（`mscale` / `mscale_all_dim` / `truncate` 在 yarn 的 `optional_keys` 里，
> `:887-895`），但**不在** TypedDict 里。「TypedDict 里没有 = 不能用」是错的；真权威是 `_validate_*` 的 `optional_keys`。

### 6.2 三件套：`convert` → `standardize` → `validate`

| 方法 | 位置 | 干什么 |
|------|------|--------|
| `convert_rope_params_to_dict` | `:742-763` | 把散落的 RoPE 参数收进 `rope_parameters`，再转调 `standardize` |
| `standardize_rope_params` | `:765-816` | **补全/改名**：`type`→`rope_type`、补 `rope_theta`、补 `original_max_position_embeddings` |
| `validate_rope` | `:818-848` | **校验**：按 `rope_type` 派发到 `_validate_<type>_rope_parameters` |

`standardize_rope_params` 分三种 Case（`:779-814`）：Case 0（什么 RoPE 参数都没有 → 打 warning 直接返回，非 RoPE 模型）；
Case 1（`layer_types is None`，或 key 不是 `layer_types` 子集 → **一个全局 dict**）；Case 2（key 全部属于 `layer_types` → **per-layer 嵌套 dict**，Gemma-3 系）。
Case 1 关键两行（`:785-786`）：`setdefault("rope_type", rope_parameters.get("type", "default"))` + `setdefault("rope_theta", rope_theta)`。

**全新 `LlamaConfig()` 的 `rope_parameters` 就已经被归一化好了**（测试断言 `tests/utils/test_modeling_rope_utils.py:409`）：

```python
        config = LlamaConfig()
        self.assertEqual(config.rope_parameters, {"rope_type": "default", "rope_theta": 10000.0})
```

即 `LlamaConfig().rope_parameters["rope_type"] == "default"`，`rope_theta` 从 `default_theta = 10_000.0`（`:739`）兜底填进来。
紧随其后的 `:412` 还断言 `assertFalse(hasattr(config, "partial_rotary_factor"))`，和「顶层属性不再可访问」是同一件事。给 llama3/yarn/longrope 补字段的一段（`:790-798`）：

```python
            # Move pretraining-time maximum length to rope parameter dict for RoPE types with scaling
            if rope_parameters["rope_type"] in ["llama3", "yarn", "longrope"]:
                if hasattr(self, "original_max_position_embeddings"):
                    # NOTE: Phi3 (and potentially other models) save `original_max_position_embeddings` field
                    # outside rope parameters. This is an exception case where we give priority to `self.original_max_position_embeddings
                    self.rope_parameters["original_max_position_embeddings"] = self.original_max_position_embeddings
                else:
                    self.rope_parameters.setdefault("original_max_position_embeddings", self.max_position_embeddings)
```

**这解释了一件事**：`rope_type` 是 llama3 / yarn / longrope 但 config 里没写 `original_max_position_embeddings` 时会被**静默**兜底成
`max_position_embeddings`。这个值决定「哪些维度算低频」，填错就等于整套 scaling 按错误的边界分段——**而 yarn 的校验器偏偏不报这个**
（`_validate_yarn_rope_parameters` 显式排除了 `implicit_factor == 1`，`:926-927`）；llama3 相对严格，`original_max_position_embeddings >= max_position_embeddings` 会 warning（`:1024-1028`）。
**结论**：这三个 type 的 `original_max_position_embeddings` 一定要手写清楚。

### 6.3 校验器清单（required / optional）

| `rope_type` | 校验函数 | required | optional |
|-------------|---------|----------|----------|
| `default` | `:850` | `{rope_type}` | `{rope_theta}` |
| `linear` | `:859` | `{rope_type, factor}` | `{rope_theta}` |
| `dynamic` | `:872` | `{rope_type, factor}` | `{rope_theta}` |
| `yarn` | `:885` | `{rope_type, factor, original_max_position_embeddings}` | `{rope_theta, attention_factor, beta_fast, beta_slow, mscale, mscale_all_dim, truncate}` |
| `longrope` | `:937` | `{rope_type, short_factor, long_factor, original_max_position_embeddings}` | `{rope_theta, attention_factor, factor}` |
| `llama3` | `:987` | `{rope_type, factor, original_max_position_embeddings, low_freq_factor, high_freq_factor, rope_theta}` | 无 |
| `proportional` | `:1030` | `{rope_type, rope_theta}` | 无 |

外加一条全局规则：`partial_rotary_factor` 会被**自动加进所有类型的 optional**（`:1059-1060`）。

### 6.4 校验抛什么错

真正干活的是 `_check_received_keys`（`:1044-1072`），末尾：

```python
# modeling_rope_utils.py:1062-1072
        if ignore_keys is not None:
            received_keys -= set(ignore_keys)
        missing_keys = required_keys - received_keys
        if missing_keys:
            raise KeyError(f"Missing required keys in `rope_parameters` for 'rope_type'='{rope_type}': {missing_keys}")
        unused_keys = received_keys - required_keys - optional_keys
        if unused_keys:
            logger.warning(f"Unrecognized keys in `rope_parameters` for 'rope_type'='{rope_type}': {unused_keys}")
```

**两级严格度**（别记反）：**缺**必填键 → **`raise KeyError`**（硬失败，`:1068`）；多了不认识的键 → `logger.warning`（`:1072`）；
值的类型/范围不对（如 `factor < 1`）→ `logger.warning`；`rope_type` 没有对应校验函数 → `logger.warning("Missing validation function ...")`（`:846-848`）。
**模型可以申请豁免**：`ignore_keys_at_rope_validation`（类属性，`:740`，默认空集），Qwen2-VL 用它放行 `mrope_sections`
（`tests/utils/test_modeling_rope_utils.py:122`：`config.ignore_keys_at_rope_validation = {"mrope_sections"}`）。

### 6.5 `rope_config_validation`：已废弃的薄壳

`modeling_rope_utils.py:1075-1089`——只剩一个转调 + 一条 `FutureWarning`（`"""This is a deprecated function. It has been kept for
backward compatibility with custom code models."""`），`config.standardize_rope_params()` + `config.validate_rope()`。
`grep -rn "rope_config_validation" src/ tests/` → **只有定义与自身消息两行，没有任何调用者**。它是留给**外部 custom code** 的兼容口。

### 6.6 `validate_rope` 是怎么被自动调用的

`RotaryEmbeddingConfigMixin`（`:734-742`）没有 `__init__`，`PreTrainedConfig` 也只是继承它（`configuration_utils.py:148`：
`class PreTrainedConfig(PushToHubMixin, RotaryEmbeddingConfigMixin, HeterogeneousConfigMixin)`）。答案在 `configuration_utils.py:146` 的
`@strict(accept_kwargs=True)`——它来自 `huggingface_hub.dataclasses`，会把**所有以 `validate_` 开头的单参数方法**收集成「类校验器」，
并在每次 `__init__` 之后自动跑一遍（`huggingface_hub/dataclasses.py:227-232, 273-278`）。
**所以：`validate_rope` 是命名约定触发的隐式钩子——改名字就失效。** 它 docstring（`:823-824`）说「validation runs every time a
new config is created, even if config is non-RoPE」，第一件事就是 `if not rope_parameters_dict: return`（`:825-826`）。
子类可整体覆盖，例如 `DeepseekV4Config.validate_rope`（`configuration_deepseek_v4.py:200-222`），因为 V4 的 `rope_parameters`
是**按 rope 类型标签**（`main` / `compress`）嵌套而不是按 `layer_types` 嵌套（`:198` 的 `_rope_type_labels = ("main", "compress")`，`modeling_rope_utils.py:776` 专门读了它）。

### 6.7 `rope_type` vs `type`：历史兼容的七处证据

| 位置 | 代码 | 作用 |
|------|------|------|
| `standardize_rope_params:785`（Case 1） | `setdefault("rope_type", rope_parameters.get("type", "default"))` | 老 config 的 `"type"` 读成 `rope_type` |
| `standardize_rope_params:806`（Case 2） | 同上，per-layer 版本 | 嵌套 dict 也兼容 |
| `_check_received_keys:1053-1056` | `if "type" in received_keys: received_keys -= {"type"}; required_keys.add("rope_type")` | **校验时**把 `type` 当 `rope_type`，避免误报 missing |
| `validate_rope:839` | `rope_type = rope_parameters.get("rope_type", rope_parameters.get("type", "default"))` | 派发校验函数时也再兜一次 |
| `PreTrainedConfig.rope_scaling`（`configuration_utils.py:548-554`） | property 别名 | 老**属性名**也兼容 |
| `convert_rope_params_to_dict:743` | `rope_scaling = kwargs.pop("rope_scaling", None)` | 老 **kwargs 键**也兼容 |
| `configuration_utils.py:313-320` | 手动判断有没有 `rope_parameters` 属性，没有就警告 | 自定义 config 的兜底 |
| `configuration_phi3.py:102-104` | `"su"` / `"yarn"` → `"longrope"` | Phi-3 专有的**第三次改名** |

真实世界里确实还有老 config 在用 `type`——本仓库自己的默认值里就有（`configuration_ministral3.py:97-98` 的 `"type": "yarn"`，靠 standardize 改写）。
**读 config.json 的判断顺序**：先看 `rope_parameters`（新），没有就看 `rope_scaling`（v4），里面再先看 `rope_type`（新），没有就看 `type`（最老）。
**最终以 `standardize_rope_params()` 跑完后的内容为准**——`print(config.rope_parameters)` 即可看归一化结果。

### 6.8 `docs/source/en/internal/rope_utils.md` 已过时

| 文档里写的 | 真实情况 |
|-----------|---------|
| `[`~modeling_rope_utils.ROPE_VALIDATION_FUNCTIONS`]` | 这个符号**不存在**，真名是 `ROPE_INIT_FUNCTIONS`（`:668`） |
| 表格只列了 6 个 rope type（有 `default`，**没有 `proportional`**） | 实际是 7 个（含 `proportional`，`:674`） |

它唯一正确的部分：「Per-Layer-Type RoPE Configuration」一节给的 Gemma-3 嵌套 config 样例。

---

## 七、模型侧怎么用

### 7.1 三步模板（每个模型一份，154 个目录）

**第 1 步**：定义 class，初始化时取一次 `inv_freq`（`modeling_llama.py:73-89`）——见 §3.2 的 `:83-85` 三行，外加
`self.max_seq_len_cached = config.max_position_embeddings`（`:77`）、`self.inv_freq = nn.Buffer(inv_freq, persistent=False)`（`:88`）、
`self.original_inv_freq = nn.Buffer(inv_freq.clone(), persistent=False)`（`:89`，**存的是 clone**——就是 §5.1「回退分支」要恢复的那一份）。

**第 2 步**：forward 算 cos/sin（`modeling_llama.py:111-128`）：

```python
    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1).to(dtype=torch.float, device=x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):   # Disable any outside autocast context, to really force fp32
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
```

**`torch.cat((freqs, freqs), dim=-1)` 这一行是「前后半拆分」流派的标志**（`:122`）——和 `rotate_half` 里 `x1 = x[..., :d//2]` 配对使用（见 §8）。
**第 3 步**：在 model forward 里算一次，传给所有层（`modeling_llama.py:399-400`）：`position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)`。

### 7.2 per-layer 变体：`layer_type` 参数

Gemma-3 系（`layer_types` 有值的模型）把 `rope_type` 变成**字典**，每层按自己的类型取。`Olmo3RotaryEmbedding`（`modeling_olmo3.py:270-318`）最清晰：

```python
# modeling_olmo3.py:277-291（节选）
        self.layer_types = sorted(set(config.layer_types));  self.rope_type = {}
        for layer_type in self.layer_types:
            rope_params = self.config.rope_parameters[layer_type]
            if rope_params is None: continue
            self.rope_type[layer_type] = rope_params["rope_type"]
            rope_init_fn = self.compute_default_rope_parameters
            if self.rope_type[layer_type] != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type[layer_type]]
            curr_inv_freq, curr_attention_scaling = rope_init_fn(self.config, device, layer_type=layer_type)
            setattr(self, f"{layer_type}_inv_freq", nn.Buffer(curr_inv_freq, persistent=False))
            # 同法再设 f"{layer_type}_original_inv_freq"（clone）与 f"{layer_type}_attention_scaling"
```

`layer_type` 从头传到尾：初始化取 `inv_freq`（`:287`）、forward 签名多一个参数（`:322` `def forward(self, x, position_ids, layer_type)`）、
model 里按类型各算一次（`:436` `position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)`）、
权重初始化时也要按类型（`:366-372`）、装饰器把它透传（`modeling_rope_utils.py:122-123`）。
对照 Gemma-3：`modeling_gemma3.py:561` 是**完全相同**的一行。**读法**：看到 `position_embeddings[layer_type] = ...` 就知道这个模型是 per-layer RoPE。

### 7.3 从 config 到分发：几个模型的真实对照

| 模型 | config 默认值 | 位置 | 用的是哪条分支 |
|------|--------------|------|---------------|
| Llama-2 | `rope_parameters = None` → 归一化成 `{"rope_type": "default", "rope_theta": 10000.0}` | `configuration_llama.py:80` | `compute_default_rope_parameters` |
| GPT-OSS | `{"rope_type": "yarn", "factor": 32.0, "beta_fast": 32.0, "beta_slow": 1.0, "truncate": False, "original_max_position_embeddings": 4096}` | `configuration_gpt_oss.py:81-88` | `_compute_yarn_parameters` |
| Apertus | `{"rope_type": "llama3", "rope_theta": 12000000.0, "factor": 8.0, "original_max_position_embeddings": 8192, "low_freq_factor": 1.0, "high_freq_factor": 4.0}` | `configuration_apertus.py:85-91` | `_compute_llama3_parameters` |
| Gemma-4 | 嵌套：`sliding_attention` 用 `default`；`full_attention` 用 `proportional` + `partial_rotary_factor: 0.25` | `configuration_gemma4.py:203-208` | 两条分支并存 |
| DeepSeek-V4 | 嵌套，但 key 是 `main` / `compress`（rope 类型标签），不是 `layer_types` | `configuration_deepseek_v4.py:198, 305-322` | `default` + `yarn` |

> **`max_seq_len_cached` 的初始化时机值得单独记一笔**：它是 `config.max_position_embeddings`（`modeling_llama.py:77`），**不是**
> `original_max_position_embeddings`。所以 `dynamic` 的重算阈值是「超过 config 声称的最大长度」而不是「超过预训练长度」——
> 这两个数在很多 config 里被故意设成不一样（如 yarn 的 `factor` 就是它们的比值）。

### 7.4 权重初始化里的第二次分发

`modeling_utils.py:2427-2433`——`_init_weights` 对 RoPE buffer 也要重算一遍（因为 `persistent=False`，这些 buffer **不在 checkpoint 里**）：

```python
        # This matches all the usual RotaryEmbeddings modules
        elif "RotaryEmbedding" in module.__class__.__name__ and hasattr(module, "original_inv_freq"):
            rope_fn = (ROPE_INIT_FUNCTIONS[module.rope_type] if module.rope_type != "default"
                       else module.compute_default_rope_parameters)
            buffer_value, _ = rope_fn(module.config)
            init.copy_(module.inv_freq, buffer_value)
            init.copy_(module.original_inv_freq, buffer_value)
```

**用类名匹配**（`"RotaryEmbedding" in module.__class__.__name__`）——你的模型类**必须**叫 `XxxRotaryEmbedding`，否则这段初始化会漏掉它。
同文件上文对 LayerNorm / RMSNorm 也是同样手段（`:2411-2415`）。

### 7.5 变体怎么继承：两个方向

**(a) 直接继承父模型的 RoPE 类** —— Qwen3（`modular_qwen3.py:36, 56-57`）：`from ..qwen2.modeling_qwen2 import (Qwen2ForCausalLM, ..., Qwen2RotaryEmbedding, ...)`
+ `class Qwen3RotaryEmbedding(Qwen2RotaryEmbedding): pass`。Qwen3 的 QK-Norm 是新东西，**但 RoPE 不是**——所以 `pass`。
`Qwen2RotaryEmbedding`（`modeling_qwen2.py:51-102`）的 `__init__`（`:51-67`）和 `compute_default_rope_parameters`（`:71-87`）与 Llama 的版本**逐行一致**（只换了类型注解），
`forward`（`:90-102`）写法略有出入但等价。这正是 2A5 §3.3 讲的「抄一份 + 全局改名」的产物。

**(b) 依赖自动拉取** —— Mistral。`grep -c "RotaryEmbedding" src/transformers/models/mistral/modular_mistral.py` → **0**；
`modular_mistral.py`（188 行）里一个字都没提 RoPE，但生成文件里有 `modeling_mistral.py:262` `class MistralRotaryEmbedding(nn.Module)`、
`:331` `self.rotary_emb = MistralRotaryEmbedding(config=config)`、`:374` `position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)`。
**这正是 2A5 §3.4 的「依赖自动拉取」**：`MistralModel` 的父类 `LlamaModel` 里有 `self.rotary_emb = LlamaRotaryEmbedding(config)`，
converter 顺着这条依赖把 `LlamaRotaryEmbedding` 整块抄进生成文件并改名。**别把它当成 Mistral 的创新**——它是 154 份重复代码中的一份。

### 7.6 小结：读任何一个模型的 RoPE，只看四行

```
1. config.rope_parameters  → 全局 dict 还是嵌套 dict？rope_type 是什么？
2. rope_init_fn = ?        → "default" → 模型自己的 compute_default_rope_parameters
                              其它     → ROPE_INIT_FUNCTIONS[rope_type]
3. cos/sin 的构造方式       → cat((freqs, freqs))？还是 polar 复数？（决定 §8 的流派）
4. apply_rotary_pos_emb     → 半拆分 / 交错？（决定权重能不能跨模型复用）
```

---

## 八、`rotate_half` 的两个流派

**这是 RoPE 里唯一一个「数学等价但权重不通用」的坑。**

### 8.1 流派 A：前后半拆分（Llama / Qwen / Mistral 全系）

```python
# modeling_llama.py:130-134
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# modeling_llama.py:136-152（节选）
@use_kernel_forward_from_hub("rotary_pos_emb")
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim); sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

**维度配对**：`cos` 是 `cat(freqs, freqs)`，第 `i` 个频率同时应用到维度 `i` 和 `i + d/2`，即旋转平面是 **(0, d/2), (1, d/2+1), …**。

### 8.2 流派 B：交错 interleaved（GLM / DeepSeek 系）

```python
# modeling_glm_moe_dsa.py:125-161（节选）
def apply_rotary_pos_emb_interleave(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    r"""... DeepSeek lays the rotary dimensions out in interleaved pairs `(x0, x1), (x2, x3), ...`, each rotated by a
    single frequency. We compute that rotation directly on the even/odd slices instead of de-interleaving with a
    `view`/`transpose`/`reshape`; the output is bit-identical to the de-interleaved `rotate_half` formulation while
    avoiding the extra contiguous copy. ..."""
    # `cos`/`sin` are `cat(freqs, freqs)`; the first half holds the per-pair angle.
    cos = cos[..., : cos.shape[-1] // 2].unsqueeze(unsqueeze_dim)
    sin = sin[..., : sin.shape[-1] // 2].unsqueeze(unsqueeze_dim)
    q1, q2 = q[..., 0::2], q[..., 1::2]
    k1, k2 = k[..., 0::2], k[..., 1::2]
    q_embed = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    k_embed = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
    return q_embed, k_embed
```

**维度配对**：`cos` 只取前一半，作用于 **(0,1), (2,3), (4,5), …** 这样的相邻对。两处最容易读漏的差异：
① `cos` / `sin` **要切一半**（`:154-155`）——`cat` 后长度是 `d`，交错流派每个频率只用一次；**半拆分流派不切**（每个频率用两次）；
② 它直接写展开的旋转矩阵，而半拆分流派写成 `(q * cos) + (rotate_half(q) * sin)`。

### 8.3 为什么数学等价

两个流派**算的是同一个旋转**，只是**维度排列不同**——存在固定置换 `P`，使「把输入按 `P` 重排后做半拆分旋转」与「对原输入做交错旋转」结果一致。
以 `d = 4` 手算（`freqs = (f0, f1)`，故 `cos = (cos f0, cos f1, cos f0, cos f1)`）：

| 派别 | 配对 | 第 0 对 | 第 1 对 |
|------|------|---------|---------|
| 半拆分 | 取 `x[..., :2]` 与 `x[..., 2:]` | `(x0, x2)` 用 `f0` | `(x1, x3)` 用 `f1` |
| 交错 | 取 `x[..., 0::2]` 与 `x[..., 1::2]` | `(x0, x1)` 用 `f0` | `(x2, x3)` 用 `f1` |

把交错布局的 `(x0, x1, x2, x3)` 先按 `P = (0, 2, 1, 3)` 排成 `y = (x0, x2, x1, x3)` 再做半拆分：`y` 的前半 `(x0, x2)` 配 `f0` → 正是交错的 `(x0, x1)` 那一对；
后半 `(x1, x3)` 配 `f1` → 正是 `(x2, x3)`。**配对完全对上**。这段可**用纯 Python（无需 torch）逐元素验证**，见验证清单第 15 条：
`half(perm(x))` 与 `inter(x)` **逐位相同**，而 `half(x)` 与 `inter(x)` 明显不同。

**代价**：`P` 是一次**权重列重排**——拿一个按交错布局训练的 checkpoint 用半拆分代码去算 RoPE，被旋转的维度对就全错了，
但 **shape 完全合法、代码不报任何错**。**症状**：模型能跑、能出 token，但输出**彻底崩坏**（困惑度爆表、复读、胡言）——
**没有异常、没有 shape mismatch、没有 nan**。唯一的解药是记住 checkpoint 的训练布局，或者像 DeepSeek 那样留一个 `rope_interleave` 开关。

### 8.4 两种流派的使用者与开关

| 流派 | 典型模型 | 实现位置 |
|------|---------|---------|
| 半拆分 | Llama、Qwen2/Qwen3、Mistral、Gemma、OLMo、GPT-OSS | `modeling_llama.py:130-152` |
| 交错 | DeepSeek-V2/V3/V3.2、GLM-4-MoE-Lite、GLM-MoE-DSA、Mistral-4、AXK1/2、Youtu、LongCat-Flash | `modeling_glm_moe_dsa.py:125`、`modeling_deepseek_v3.py:322`、`modeling_longcat_flash.py:295` 等 |

**有些模型两派都有**，用一个 config 开关切：

```python
# modeling_deepseek_v3.py:464-467
        if self.config.rope_interleave:  # support using interleaved weights for efficiency
            q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        else:
            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
```

开关在 config：`configuration_deepseek_v3.py:114` `rope_interleave: bool | None = True`（同样的开关见 `configuration_axk1.py:112`、
`configuration_youtu.py:92`、`configuration_glm4_moe_lite.py:113`）。**含义**：`True` = 权重已按交错布局排好，`False` = 半拆分布局。
**切换这个开关必须配合权重重排，不能只改 flag。**

> GLM-MoE-DSA 里还有个细节：MLA 的 Q/K 只有 `qk_rope_head_dim` 那一小段参与旋转（`modeling_glm_moe_dsa.py:227-232` 先 `torch.split`
> 出 `q_rot` / `q_pass`，再只对 `q_rot` 施加 RoPE）。这也是为什么 GLM 的 config 把 `head_dim` 指向 `qk_rope_head_dim`
> （`configuration_glm_moe_dsa.py:150-152` 注释："RoPE applies only to the rope slice, so point `head_dim` at it"）——
> **这样 `compute_default_rope_parameters` 不用任何覆写就能算对频率**。

### 8.5 第三个流派（顺带）：复数表示

DeepSeek-V2 把 `freqs` 变成复数再逐元素相乘（`modular_deepseek_v2.py:230-236`）：

```python
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.to(x.device) @ position_ids_expanded).transpose(1, 2)
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # Convert to complex representation
            freqs_cis = freqs_cis * self.attention_scaling
        return freqs_cis
```

**它把 `cos`/`sin` 合并成一个复数张量**，上层直接 `torch.view_as_real(q * freqs_cis)`。对比 §7.1 里 Llama 返回 `(cos, sin)` 两个张量
——**返回类型都不一样**，所以 `apply_rotary_pos_emb` 这类函数在这两个模型间**不通用**。这也说明 `ROPE_INIT_FUNCTIONS` 抽象的边界：
它统一的是 **`inv_freq` 的计算**，**不统一 `cos`/`sin` 的构造和施加方式**。

---

## 九、实操：给自己的模型换一个 RoPE scaling

### 9.1 只改 config（不需要重训的场景）

`linear` / `dynamic` 是纯推理时的频率重算，不改任何权重，可以直接改：

```python
config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
# 不是改 config.rope_scaling（v5 里只是 property 别名），直接改 rope_parameters
config.rope_parameters = {"rope_type": "dynamic", "rope_theta": 10000.0, "factor": 8.0}   # 目标 8× 原长度
# 缺必填键会直接 KeyError（多的键只是 warning，见 §6.4），dynamic 至少要 rope_type + factor
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", config=config)
```

| 陷阱 | 后果 | 怎么发现 |
|------|------|---------|
| 用 `yarn` / `llama3` / `longrope` 但漏了 `original_max_position_embeddings` | **静默兜底**成 `max_position_embeddings` → 分段边界全错，效果打折但不报错 | yarn **不报警**（`:926-927` 显式排除）；llama3 会 warning（`:1024-1028`）。最稳的办法是手写 |
| `rope_type` 拼错 | `validate_rope` 找不到 `_validate_<x>_rope_parameters` → 只 warning；继续跑会在 `ROPE_INIT_FUNCTIONS[x]` 处 **KeyError** | 必须同时加校验函数和注册项才安全 |
| 给**自定义 config**（没有 `rope_parameters` 属性）传 `rope_scaling=` + `rope_theta=` | 走 BC 分支（`configuration_utils.py:315-320`），会有一条 warning 提示改用 `rope_parameters` | 看日志 |
| 用了 `yarn` 却期望 `attention_factor` 是 1.0 | 会被推算成 `0.1·log(factor)+1`（如 factor=32 → ≈1.347） | 显式写 `"attention_factor": 1.0` |

### 9.2 需要重训 / 换 checkpoint 的场景

| 方法 | 能否只改 config | 原因 |
|------|----------------|------|
| `linear` / `dynamic` | **可以**（推理期行为） | 只重算 `inv_freq`，权重不动 |
| `proportional` | **不可以** | 不旋转的维度在训练时学的就是「无位置信息」的表示 |
| `yarn` | **看情况** | `inv_freq` 是推理期算的，但 `attention_factor` 会改注意力尺度，训练时没见过就可能退化 |
| `longrope` | **不可以** | `short_factor` / `long_factor` 是论文**搜索**出来的逐维度数组 |
| `llama3` | **不可以** | Meta 是拿 8192 长度的模型继续预训练出来的；参数只是描述，模型本身是重训的 |

**口诀**：**改的是「怎么算 `inv_freq`」→ 可以试；改的是「模型见过什么位置」→ 必须重训。**

### 9.3 加一种全新 RoPE

```python
# 1) 写一个签名一致的计算函数
def _compute_my_parameters(config, device=None, seq_len=None, layer_type=None):
    config.standardize_rope_params()          # ← 所有内置函数第一件事都是这个
    rope_parameters_dict = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters
    base = rope_parameters_dict["rope_theta"]
    ...
    return inv_freq, attention_factor

# 2) 注册（文件末尾注释明说了可以这样扩展，:665-667）
ROPE_INIT_FUNCTIONS["my_rope"] = _compute_my_parameters
```

**别忘了第三步**：在 `RotaryEmbeddingConfigMixin` 上补一个 `_validate_my_rope_parameters(self, rope_parameters, ignore_keys=None)`，
否则校验阶段只会打一句 `Missing validation function`（`:846-848`）就放过，config 里少写 `factor` 也不会报错——**静默失败**。

---

## 一句话总结

> `ROPE_INIT_FUNCTIONS` 只统一了一件事：**`inv_freq` 怎么算**——六种 scaling 各自改的是「频率整体缩放 / 按维度混插 / 按波长分三段 /
> 按长度切换数组 / 只旋转一部分」，返回的都是 `(inv_freq, attention_scaling)` 二元组；`default` 不在这张表里，
> 由每个模型的 `compute_default_rope_parameters` 兜底。
> config 侧由 `RotaryEmbeddingConfigMixin` 做「`type`→`rope_type` 改名 + 补字段 + 按类型校验」，
> `rope_config_validation` 已废弃成薄壳，改为由 `@strict` 隐式调用的 `validate_rope`。
> `dynamic_rope_update` 是个只在 `dynamic` / `longrope` 下生效的装饰器：在 forward 里比较 `torch.max(position_ids) + 1`
> 与缓存的 `max_seq_len_cached`，超了就重算 `inv_freq` 并 `register_buffer` 换掉（`persistent=False`，不落 state_dict），
> 回到短序列还能复位——但**真正的数值变化门槛是 `config.max_position_embeddings`**，不是那个缓存值。
> 而 `cos`/`sin` 一旦构造出来，就分成「前后半拆分」和「交错」两派，**数学等价但权重不通用**，混用就是一次无异常、无 shape 错误的静默崩坏。

---

## 验证清单

全部在仓库根目录（`/Users/baai/Desktop/baai_proj/transformers`）下跑过，输出即下文所述。

```bash
# 1. ROPE_INIT_FUNCTIONS 的完整 key（AST 解析真实字面量，不靠肉眼/记忆）
python3 -c "
import ast
src = open('src/transformers/modeling_rope_utils.py').read()
for node in ast.parse(src).body:
    if isinstance(node, ast.AnnAssign) and getattr(node.target,'id',None)=='ROPE_INIT_FUNCTIONS':
        print('line', node.lineno)
        for k,v in zip(node.value.keys, node.value.values): print(' ', k.value, '->', v.id)
"
# → line 668
#    linear -> _compute_linear_scaling_rope_parameters
#    dynamic -> _compute_dynamic_ntk_parameters
#    yarn -> _compute_yarn_parameters
#    longrope -> _compute_longrope_parameters
#    llama3 -> _compute_llama3_parameters
#    proportional -> _compute_proportional_rope_parameters
#   ★ 只有 6 个，没有 "default"

# 2. 所有顶层定义的真实行号
grep -n "^def \|^class \|^ROPE_INIT\|^    def " src/transformers/modeling_rope_utils.py
# → :34 dynamic_rope_update / :47 longrope_frequency_update / :82 dynamic_frequency_update
#    :121 wrapper / :133 linear / :193 proportional / :269 dynamic / :345 yarn
#    :486 longrope / :580 llama3 / :668 ROPE_INIT_FUNCTIONS / :678 RopeParameters
#    :734 RotaryEmbeddingConfigMixin / :742 convert / :765 standardize / :818 validate_rope
#    :850/:859/:872/:885/:937/:987/:1030 六个 _validate_* / :1045 _check_received_keys
#    :1075 rope_config_validation

# 3. rope_config_validation 已无任何调用者
grep -rn "rope_config_validation" --include="*.py" src/ tests/
# → 只有 modeling_rope_utils.py:1075（定义）与 :1081（warning 文本）

# 4. 文件行数
wc -l src/transformers/modeling_rope_utils.py            # → 1089
wc -l tests/utils/test_modeling_rope_utils.py            # → 1575

# 5. 模型侧分发点：143 个文件是 self.rope_type 版，35 个是 per-layer 版
grep -rln "ROPE_INIT_FUNCTIONS\[self.rope_type\]" --include="*.py" src/transformers/models/ | wc -l   # → 143
grep -rl "ROPE_INIT_FUNCTIONS" --include="*.py" src/transformers/models/ \
  | sed 's|src/transformers/models/||;s|/.*||' | sort -u | wc -l                                     # → 154

# 6. "default" 走模型自带的静态方法
grep -n "rope_init_fn\|compute_default_rope_parameters" src/transformers/models/llama/modeling_llama.py
# → :83/:85 是分发，:93 是 compute_default_rope_parameters
#   :82 rope_type = self.config.rope_parameters["rope_type"]，:88/:89 两个 buffer
grep -n "ROPE_INIT_FUNCTIONS\[module.rope_type\]" -B 3 -A 6 src/transformers/modeling_utils.py
# → :2427-2433，_init_weights 里按类名匹配 "RotaryEmbedding" 二次分发（:2411-2415 对 RMSNorm 同理）

# 7. dynamic_rope_update 的真实触发条件
sed -n '34,130p' src/transformers/modeling_rope_utils.py
# → :124      if "dynamic" in rope_type        （子串匹配，目前只命中 "dynamic"）
#   :126      elif rope_type == "longrope"
#   :88       seq_len = torch.max(position_ids) + 1
#   :100      if seq_len > max_seq_len_cached:            ← 重算触发
#   :102-107  rope_init_fn(..., seq_len=seq_len) 并更新 attention_scaling
#   :109      register_buffer(f"{prefix}inv_freq", inv_freq, persistent=False)
#   :112      if seq_len < original_max_seq_len and max_seq_len_cached > original_max_seq_len  ← 复位
#   :108      # TODO joao: may break with compilation
sed -n '328,341p' src/transformers/modeling_rope_utils.py
# → 数值真正变化的门槛是 seq_len > config.max_position_embeddings（内部 max() 兜底）

# 8. "type" → "rope_type" 的历史兼容（§6.7 列了七处独立逻辑，这里核对其中几处）
sed -n '783,786p;800,810p' src/transformers/modeling_rope_utils.py   # standardize Case1 / Case2
sed -n '1053,1057p' src/transformers/modeling_rope_utils.py          # _check_received_keys 里减掉 "type"
sed -n '839,841p'  src/transformers/modeling_rope_utils.py           # 派发校验函数时兜底
sed -n '548,554p'  src/transformers/configuration_utils.py           # @property rope_scaling 别名
sed -n '311,321p'  src/transformers/configuration_utils.py           # __init__ 里的 BC 钩子
sed -n '102,105p'  src/transformers/models/phi3/configuration_phi3.py # "su"/"yarn" → "longrope"
grep -n '"type": "yarn"' src/transformers/models/ministral3/configuration_ministral3.py
# → :97（本仓库自己的默认值仍在用老键）
grep -n "rope_parameters" MIGRATION_GUIDE_V5.md
# → :513 / :605（v5 迁移说明）

# 9. validate_rope 由 @strict 隐式调用（不是 base __init__）
grep -n "validate_rope" --include="*.py" src/transformers/configuration_utils.py   # → 无匹配
grep -n "validate_rope" --include="*.py" -r src/transformers/models/ | head
# → 只有 phi3 / phimoe / phi4_multimodal / cohere2_moe / deepseek_v4 覆写
grep -n "strict" src/transformers/configuration_utils.py
# → :26 from huggingface_hub.dataclasses import strict
#   :146 @strict(accept_kwargs=True)；:148 class PreTrainedConfig(..., RotaryEmbeddingConfigMixin, ...)
sed -n '227,278p' /private/tmp/hubdl/hub200/huggingface_hub/dataclasses.py
# → 收集所有 validate_* 方法，在 __init__ 之后自动跑（本机 hub200 / hub133 两份副本内容一致，纯属本地缓存）

# 10. 两个 rotate_half 流派
sed -n '130,152p' src/transformers/models/llama/modeling_llama.py     # 半拆分 + apply_rotary_pos_emb
sed -n '125,165p' src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py  # 交错
grep -rn "apply_rotary_pos_emb_interleave" --include="*.py" src/ | wc -l       # → 32 处
grep -n "rope_interleave" src/transformers/models/deepseek_v3/configuration_deepseek_v3.py
# → :114 rope_interleave: bool | None = True；调用点 modeling_deepseek_v3.py:464-467

# 11. llama3 的真实常数
sed -n '376,383p' src/transformers/models/llama/convert_llama_weights_to_hf.py
# → factor 8.0(3.1) / 32.0(3.2)，low_freq_factor 1.0，high_freq_factor 4.0，orig 8192

# 12. 各 rope_type 的 required/optional 键
sed -n '850,930p;937,945p;987,999p;1030,1035p' src/transformers/modeling_rope_utils.py
# 13. Mistral 的 RotaryEmbedding 是自动拉取来的（modular 里 0 处提及）
grep -c "RotaryEmbedding" src/transformers/models/mistral/modular_mistral.py  # → 0
grep -n "class MistralRotaryEmbedding" src/transformers/models/mistral/modeling_mistral.py  # → 262

# 14. 全新 LlamaConfig() 的 rope_parameters 长什么样（测试里的权威断言）
sed -n '408,412p' tests/utils/test_modeling_rope_utils.py
# → config = LlamaConfig()
#   self.assertEqual(config.rope_parameters, {"rope_type": "default", "rope_theta": 10000.0})
#   self.assertEqual(config.hidden_size, 4096)
#   self.assertEqual(config.num_attention_heads, 32)
#   self.assertFalse(hasattr(config, "partial_rotary_factor"))   ← 顶层属性已不可访问
```

```bash
# 15. 【唯一一条真的跑了数值的验证】两个 rotate_half 流派的配对关系（纯 Python，不需要 torch）
#     half = 半拆分（Llama），inter = 交错（GLM/DeepSeek），P = (0,2,1,3) 是两者之间的维度置换
python3 -c "
import math
d=4; freqs=[0.5,1.3]
def cs(pos):
    c=[math.cos(f*pos) for f in freqs]; s=[math.sin(f*pos) for f in freqs]
    return c+c, s+s                      # cos = cat(freqs, freqs)
def half(x,pos):
    c,s=cs(pos); x1=x[:d//2]; x2=x[d//2:]; rh=[-v for v in x2]+x1     # rotate_half
    return [x[i]*c[i]+rh[i]*s[i] for i in range(d)]
def inter(x,pos):
    c,s=cs(pos); c=c[:d//2]; s=s[:d//2]  # 交错流派只取 cos 的前一半
    q1=x[0::2]; q2=x[1::2]
    a=[q1[i]*c[i]-q2[i]*s[i] for i in range(d//2)]
    b=[q2[i]*c[i]+q1[i]*s[i] for i in range(d//2)]
    return a+b
x=[1.0,2.0,3.0,4.0]; pos=3.0; P=[0,2,1,3]
for lbl,v in [('half(x)      ',half(x,pos)),('inter(x)     ',inter(x,pos)),
              ('half(perm(x))',half([x[i] for i in P],pos)),('inter(x,0)   ',inter(x,0.0))]:
    print(lbl,'=',[round(t,6) for t in v])
"
# → half(x)       = [-2.921748, 1.2992, 1.209707, -4.279262]
#   inter(x)      = [-1.924253, 0.573268, 1.138969, -4.967028]
#   half(perm(x)) = [-1.924253, 0.573268, 1.138969, -4.967028]   ← 与 inter(x) 逐位相同
#   inter(x,0)    = [1.0, 3.0, 2.0, 4.0]                         ← 位置 0 时输出被置换
# 结论：half(perm(x)) == inter(x)  ⇒ 两派只差一个维度置换（§8.3 的等价性成立）
#       half(x) != inter(x)       ⇒ 配对不同，所以权重不可跨流派复用（混用必崩）
```

> 本机环境**没有 `torch` / `regex`**，`import transformers` 会
> `ModuleNotFoundError: No module named 'regex'`，所以本篇的验证手段只有两种：
> 对仓库的 `grep -n` / `sed -n` / `ast` 静态核对（第 1-14 条），
> 以及第 15 条那个「把 §8 的公式抄成纯 Python」的数值检查。
> **没有任何 transformer 前向被跑过**——第 5 节里那段可疑代码因此标注为「静态阅读发现」。
> 反过来这也意味着：文中每个行号、每个 key、每个参数名都可以用上面这些命令复现。

---

## 导航

- 路线图：[00_roadmap.md](./00_roadmap.md)（本篇 = 进阶 · 架构 · 2A4）
- 上一篇：[2A3_kv_cache.md](./2A3_kv_cache.md)
- 下一篇：[2A6_weights_quant_parallel.md](./2A6_weights_quant_parallel.md)
- **配套回看**：[1M1_llama_structure.md](./1M1_llama_structure.md) 第 2 节——用本篇的视角重看 `LlamaRotaryEmbedding`，留意三个细节：
  ① `rope_type` 是从 `config.rope_parameters` 而不是 `config.rope_scaling` 里取的（路线图 `00_roadmap.md:64` 专门标注了这个改名）；
  ② `forward` 头上那个 `@dynamic_rope_update` 在 `rope_type == "default"` 时**完全空转**；
  ③ `emb` 用的是 `torch.cat((freqs, freqs))`，这一行就决定了它属于「半拆分」流派。
- **顺带建议**：[2A5_modular_code_generation.md](./2A5_modular_code_generation.md) §3.4（依赖自动拉取）——本篇 §7.5 的 `MistralRotaryEmbedding` 就是那条规则最典型的产物。
