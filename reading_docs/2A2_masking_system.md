# Mask 系统精读：`mask_function` 的组合代数

> 阅读目标：搞清楚**注意力 mask 到底是什么**，以及它为什么从「一张张量」变成了「一个函数」。
>
> **这是「注意力三件套」的第二篇**。三篇是同一件事的三个侧面：
> **2A1 算什么（后端）、2A2 在哪算（mask）、2A3 缓存什么（cache）**。
> 本篇只回答一件事：**哪些位置可以互相看**，以及这个约束是怎么被**拼**出来的。
>
> 源文件：`src/transformers/masking_utils.py`（1601 行，全篇主角）、
> `src/transformers/modeling_attn_mask_utils.py`（503 行，**已被废弃的旧实现**）
> 人类向文档：`docs/source/en/attention_interface.md`（`## Build an attention mask` 一节）
>
> 一句话预告：**mask 不再是张量，而是一个 `(b, h, q, kv) -> bool` 的纯函数；
> causal / sliding-window / padding / packing 都是函数，用 `and_masks` / `or_masks` 拼起来。**
> 全篇最实用的产出是文末的 [§9 实操](#九实操给定-pattern-怎么写-mask)。

---

## 速读区（TL;DR）

**一句话**：mask 从「物化一张 `[B,1,S,S]` 浮点张量」变成「一个 `(b,h,q,kv) -> bool` 的纯函数」，
causal / sliding-window / padding / packing 都是函数，用 `and_masks` / `or_masks` 两个组合子拼起来。
但**对外的 API 形态仍然是张量或 `None`**——「mask 是函数」只活在实现内部。

**必须记住的断言**

| # | 断言 | 出处（文件:行号） |
|---|------|------------------|
| 1 | 组合子**只有两个**：`and_masks`（交集 / 收权限）、`or_masks`（并集 / 加权限），没有 `not_mask` / `xor` | `masking_utils.py:48`、`:62` |
| 2 | 起点常量：`and` 用 `q_idx.new_ones`（空 = 全允许），`or` 用 `q_idx.new_zeros`（空 = 全屏蔽）；用 `&`/`\|` 而非 `and`/`or` | `:54`、`:68` |
| 3 | 签名 `mask_function(batch_idx, head_idx, q_idx, kv_idx) -> bool`，**`True` = 允许看** | `docs/source/en/attention_interface.md:219`；全库无例外 |
| 4 | `create_*_mask` 里组合顺序**固定死**：先 OR 槽位（`:962`）再 AND 槽位（`:968`），源码反复警告 | `:956-958`、`:1066-1068` |
| 5 | `sliding_window_overlay` **单独用是错的**——它不排除未来位置 | docstring `:93-96` |
| 6 | `causal_mask_function` 用 `kv_idx <= q_idx`（是 `<=` 不是 `<`，对角线允许看自己） | `:80` |
| 7 | 窗口是**包含式**计数：库的 `W` 换算成 FA 的 `window_size=(W-1, W-1)` | `modeling_flash_attention_utils.py:655` |
| 8 | 后端没注册 mask 接口 → **直接返回 `None`，所有约束被静默丢弃** | `:819-820` |
| 9 | `create_causal_mask` **从不返回函数**，返回类型写死 `torch.Tensor \| BlockMask \| None`；唯一保留函数形态的是 flex | `:875`、`:699-707` |
| 10 | `position_ids` 在 mask 层**只用于 packed 检测**；绝对位置来自 cache 的 `q_offset`/`kv_offset` + `arange` | `:851-859`、`:507-510`、`:828-833` |
| 11 | 与 attention 侧有一处**刻意非对称**：`ALL_MASK_ATTENTION_FUNCTIONS` **有** `"eager"`，**没有** paged 变体 | `:714-721`（对照 `modeling_utils.py:5156-5167`） |
| 12 | `modeling_attn_mask_utils.py` **已废弃**（`v5.10` 删除承诺已过期，当前 `5.16.0.dev0`），src 内零 import | `modeling_attn_mask_utils.py:14-18`、`:29-32` |
| 13 | eager 下 causal mask **永远物化**（跳过能力被主动丢弃），转成 `finfo(dtype).min` 加性 bias | `:584`、`:600-603` |
| 14 | `flash` 在无 padding（或补齐后全有效）时**直接返回 `None`** | `:638-647` |

**最容易踩的坑 / 反直觉**

- **`and_mask_function` 是「与」上去的，表达不了「放松」**：「前 N 个 token 内部双向」必须走 **OR 槽位**；只用 AND 槽位会得到一个错误网格（§9.3 的两版对照实测）。
- **`sliding_window_overlay` 不是 mask，是 overlay**：单独用会允许未来位置（`q_idx=0, kv_idx=100` 也返回 `True`）。必须配基底 + `and_masks`。
- **手动传 `or_mask_function` 会绕过内部 padding**（Gemma3 源码注释原话："it's passed as an `or_mask_function` which bypasses internal padding"），必须自己调 `maybe_pad_block_sequence_ids`（`:217-227`）。**这是最容易漏的一步。**
- **自定义函数必须 index-based**：`bidirectional_mask_function` 特意写 `return q_idx >= 0` 而不是 `return True`，否则非 vmap 的广播路径会断（`:83-89` 的 NOTE）。
- **eager 慢在 mask 这一层的具体体现**：`_ = kwargs.pop("allow_is_causal_skip", None)` 把「不建 mask」的可能性直接扔掉，`q_length == 1` 也得建。
- **旧路径的 `_unmask_unattended`（修复「整行被屏蔽」）在新路径没有对应实现**（`grep unmask` 零命中）。mask 只保证「哪些位置参与」，**不保证每行至少有一个位置参与**（左 padding 的 q0/q1 行就是这种情况）。
- **Gemma2 是逐层交替**（奇数层 sliding、偶数层 full），不是「前 3 个 token 全局」——后者是 Gemma3 的 blockwise 图像 token。
- **4D float mask 的 `1`/`0` 约定是经典坑**：mask 是**加**到 score 上的，所以 `0.0` 才是保留，`1.0` 只是加了个小偏置。
- 用了 `or/and_mask_function` 就**强制 `use_vmap=True` 且要求 `torch>=2.6`**，否则直接 `ValueError`；同时 `allow_is_causal_skip = False`，哪怕 pattern 恰好等价 causal 也会物化。

**读完应该能回答**

- 组合子有几个？起点常量为什么是 ones / zeros？为什么用 `&`/`|` 而不是 `and`/`or`？
- 为什么 OR 槽位必须先于 AND 槽位？想「整体收权限」该把逻辑写在哪？
- 自定义后端忘记注册 `AttentionMaskInterface` 会发生什么？为什么这是全套机制里最危险的坑？
- `position_ids` 在 mask 层到底干什么？绝对位置从哪来？`past_seen_tokens` 和 `q_offset` 是什么关系？
- 什么时候 mask 会变成 `None`？`compileable cache` + `q_length == 1` 时为什么反而不许跳过？

---

## 前置知识

- [1M2_mistral_sliding_window.md](./1M2_mistral_sliding_window.md)：滑动窗口注意力的动机。
  本篇要把它拆到底——1M2 里那句「`sliding_window_overlay` 就是 `kv_idx > q_idx - sliding_window`」，
  在这一篇会变成一整张代数表。
- [1M1_llama_structure.md](./1M1_llama_structure.md)：`LlamaAttention.forward` 把 `attention_mask` 一路传给
  `attention_interface` 的那条链路。**mask 是「算之前」的事，不是 attention 内部的事**——
  这一句是理解本篇的钥匙。
- [2A5_modular_code_generation.md](./2A5_modular_code_generation.md)：读源码的方法论（本篇引用
  `modular_gemma3.py` 时用得上）。
- [2A1_attention_backends.md](./2A1_attention_backends.md)（待写）：`ALL_ATTENTION_FUNCTIONS` 注册表。
  本篇讲的是它的孪生兄弟 `ALL_MASK_ATTENTION_FUNCTIONS`。

---

## 一、范式转变：从「物化一张 `[B,1,S,S]`」到「用一个函数描述它」

这是本篇的主线，也是理解 `masking_utils.py` 存在的唯一理由。

### 1.1 旧世界：`AttentionMaskConverter`

旧做法在 `modeling_attn_mask_utils.py`。核心是一个 dataclass（`modeling_attn_mask_utils.py:36`），
它的 `to_causal_4d()` / `to_4d()` 直接**在显存里写出一整张 4D 张量**：

```python
# modeling_attn_mask_utils.py:163-175（_make_causal_mask 的主体）
bsz, tgt_len = input_ids_shape
mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
mask_cond = torch.arange(mask.size(-1), device=device)
mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
mask = mask.to(dtype)
...
return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)
```

注意最后那行 `expand`——它给出的是一个 `[B, 1, S, S]` 的加性 bias：`0` 表示可以看，
`finfo(dtype).min` 表示不能看，然后被 attention 直接**加**到 score 上
（`models/llama/modeling_llama.py:204-206`）：

```python
attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
if attention_mask is not None:
    attn_weights = attn_weights + attention_mask
```

滑动窗口也是同一张张量上的一个 `tril` + `masked_fill_`
（`modeling_attn_mask_utils.py:185-192`）：

```python
if sliding_window is not None:
    diagonal = past_key_values_length - sliding_window - 1
    context_mask = torch.tril(torch.ones_like(mask, dtype=torch.bool), diagonal=diagonal)
    if is_torchdynamo_compiling():
        mask = mask.clone()
    mask.masked_fill_(context_mask, torch.finfo(dtype).min)
```

拼装逻辑在 `to_4d()`（`modeling_attn_mask_utils.py:131-165`）：先造 causal 张量，
再把 padding 张量 `masked_fill` 进去。**一切都是张量算子**。

### 1.2 为什么必须改

| 痛点 | 物化 mask | 函数 mask |
|------|-----------|-----------|
| **显存** | `[B,1,S,S]`：S=8192、float32 时是 `8192² × 4B = 256 MiB`，**每一层每个 batch**。eager 下这是硬开销 | 一个闭包，**0 字节**。真正物化时也只需要 bool（1/4 大小），或者干脆不物化 |
| **能否跳过** | 已经算出来了，只能带着算 | 可以在运行时判断「这个 mask 等于 `is_causal`，不用建」，直接 `return None` |
| **后端兼容** | flash / flex 根本不吃 4D 加性 mask。给了就得降级到 sdpa | `flex_attention` 要的就是 `mask_mod` **函数**，直接透传 |
| **表达能力** | 想加一种新 pattern 就得多写一段张量算子（`tril`、`cumsum`、`roll`…） | 写一行 `return kv_idx <= q_idx`。组合靠 `&` / `|` |
| **可编译性** | 形状随 `S` 变，`torch.compile` 反复重编 | 函数在 `create_block_mask` 里由 torch 内部编译，图更稳定 |

`flex_attention` 这条最关键：PyTorch 的 flex attention 官方 API 就是
`create_block_mask(mask_mod=fn, B, H, Q_LEN, KV_LEN)`，其中 `mask_mod(b, h, q, kv) -> bool`。
新的 `mask_function` 与它**逐参数同构**，所以 `flex_attention_mask` 是唯一一个
**不做任何元素级转换、直接把函数交出去**的后端（`masking_utils.py:699-707`）。

### 1.3 新旧对比图

```
【旧】物化（modeling_attn_mask_utils.AttentionMaskConverter）
┌────────────────────────────────────────────────────────────┐
│ 2D padding mask [B, S] ──┐                                 │
│                          ├─ masked_fill_ ─┐                │
│ tril(S,S) 三角 ──┐       │                │                │
│ sliding_window ──┴───────┴────────────────┘                │
│                          ▼                                 │
│   float32 张量 [B, 1, S, S]   ← 每个元素真实占显存          │
│                          ▼                                 │
│   attn_weights + mask   （eager 专用）                      │
└────────────────────────────────────────────────────────────┘
   S=8192、fp32 → 每层 256 MiB

【新】函数（masking_utils.mask_function）
┌────────────────────────────────────────────────────────────┐
│ def inner_mask(b, h, q_idx, kv_idx):                       │
│     return kv_idx > q_idx - sliding_window                 │
│                          ▼                                 │
│   and_masks(causal_mask_function, sliding_window_overlay)  │
│                          ▼                                 │
│   一个 Python 闭包   ← 0 字节                              │
│                          ▼                                 │
│ sdpa_mask: 广播 arange（可选 vmap）→ bool [B,1,S,S]        │
│ flex_attention_mask: → BlockMask（函数被包进去，仍不展开） │
│ flash_attention_mask: → None 或 2D [B,S]（只要 seq_lens）   │
└────────────────────────────────────────────────────────────┘
   只有「后端真正需要的形态」才被物化
```

**关键认识**：`create_*_mask` 返回的**永远是物化结果或 `None`，从不返回函数**。
函数只活在中间阶段——唯一的例外是 flex 后端，它把函数封进 `BlockMask` 里带走。
§6 会把这条讲透。

---

## 二、核心抽象：`mask_function`

### 2.1 签名与约定

源码里**没有 TypeAlias**（全文只有 `Callable` 标注），语义定义在两处：

1. `sdpa_mask` 的 docstring（`masking_utils.py:404-405`）：
   > `mask_function (Callable)` — The mask factory function describing the mask pattern.
2. 人类向文档 `docs/source/en/attention_interface.md:219`，写得最清楚：
   > The `mask_function` argument is a `Callable` that mimics PyTorch's
   > [mask_mod](https://pytorch.org/blog/flexattention/) functions.
   > It takes 4 indices `(batch_idx, head_idx, q_idx, kv_idx)` and returns a boolean
   > **indicating whether that position contributes to the attention computation**.

结论：

```
mask_function(batch_idx, head_idx, q_idx, kv_idx) -> bool
             ↑          ↑         ↑        ↑
           batch 维    头维      query 行  key/value 列
```

- **`True` = 允许看**（`False` = 屏蔽）。这一点在全库一致，没有例外。
- 四个参数**都可以被忽略**：你可以只写 `def m(b, h, q, kv): return kv <= q`，
  也可以只依赖 `batch_idx`（padding）。
- 返回值在实现里通常是**一个 bool 标量张量**，不是 Python `bool`——
  因为要能参加 `&` / `|` 的向量化运算（见 §3.2）。

### 2.2 为什么是这四维，而不是二维

因为**「能看」这件事天然依赖 batch 和 head**：

| 维度 | 谁在用 | 例子 |
|------|--------|------|
| `batch_idx` | padding / packing / blockwise | `padding_mask_function`（`masking_utils.py:177`）、`blockwise_overlay`（`:127`） |
| `head_idx` | ALiBi、per-head 位置偏置、某些稀疏注意力 | 目前 `masking_utils.py` 内部**没有**用它，但签名保留 |
| `q_idx` | causal / sliding / chunked | `causal_mask_function`（`:80`） |
| `kv_idx` | causal / sliding / padding | 同上 |

四个参数全部保留，是为了让「函数」这套接口能表达**任意** pattern——
正是 `flex_attention` 的 `mask_mod` 的签名（`docs/source/en/attention_interface.md:219`）。

### 2.3 `True` / `False` 的边界情况：`bidirectional_mask_function`

```python
# masking_utils.py:83-89
def bidirectional_mask_function(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    """
    This creates a full bidirectional mask.

    NOTE: It is important to keep an index-based version for non-vmap expansion.
    """
    return q_idx >= 0
```

「全部允许」为什么要写成 `q_idx >= 0` 而不是 `return True`？

因为**非 vmap 展开路径**（`_non_vmap_expansion_sdpa`，`masking_utils.py:352-368`）
把四个索引张量广播成 `[B,1,1,1]` / `[1,H,1,1]` / `[1,1,Q,1]` / `[1,1,1,KV]`，
再用 `&` / `|` 组合。**如果某个函数返回一个不依赖任何输入的 Python `True`，
广播就断了**，后面的 `mask.expand(...)`（`:518`）也可能对不上。
写成 `q_idx >= 0` 就得到一个形状正确的 bool 张量。

> 这个 NOTE 是源码里唯一一处「为什么必须保持 index-based」的显式提醒，
> 也是 §9 里写自定义 mask 时必须遵守的纪律。

---

## 三、组合代数：只有两个组合子

### 3.1 全部组合子：实测确认只有 `and_masks` / `or_masks`

```bash
$ grep -n "^def and_masks\|^def or_masks" src/transformers/masking_utils.py
48:def and_masks(*mask_functions: Callable) -> Callable:
62:def or_masks(*mask_functions: Callable) -> Callable:
```

就这两个。没有 `not_mask`、没有 `xor`、没有 `compose`。
（`not` 不需要——「屏蔽」用 `and` 一个反向断言的函数表达即可，见 §9。）

真实代码（`masking_utils.py:48-73`，两段完整贴出）：

```python
def and_masks(*mask_functions: Callable) -> Callable:
    """Returns a mask function that is the intersection of provided mask functions"""
    if not all(callable(arg) for arg in mask_functions):
        raise RuntimeError(f"All inputs should be callable mask_functions: {mask_functions}")

    def and_mask(batch_idx, head_idx, q_idx, kv_idx):
        result = q_idx.new_ones((), dtype=torch.bool)
        for mask in mask_functions:
            result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        return result

    return and_mask


def or_masks(*mask_functions: Callable) -> Callable:
    """Returns a mask function that is the union of provided mask functions"""
    if not all(callable(arg) for arg in mask_functions):
        raise RuntimeError(f"All inputs should be callable mask_functions: {mask_functions}")

    def or_mask(batch_idx, head_idx, q_idx, kv_idx):
        result = q_idx.new_zeros((), dtype=torch.bool)
        for mask in mask_functions:
            result = result | mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        return result

    return or_mask
```

三个必须记住的实现细节：

| 细节 | 代码 | 含义 |
|------|------|------|
| **起点常量** | `and` 用 `q_idx.new_ones`；`or` 用 `q_idx.new_zeros` | 空 `and_masks()` = 全允许；空 `or_masks()` = 全屏蔽。这也是「单位元」的正确选择 |
| **用 `&` / `\|` 而非 `and` / `or`** | `result = result & mask(...)` | `and` 会触发 `bool()` 求值 → 图断裂 / vmap 失败。位运算是逐元素的，可编译 |
| **`.to(result.device)`** | `mask(...).to(result.device)` | 每个 mask 可能内部持有不同 device 的常量张量（如 padding mask），这里统一 |

### 3.2 组合的**顺序在 `create_*_mask` 里是固定死的**

这是最容易踩的坑，务必记住。以 `create_causal_mask` 为例（`masking_utils.py:942-979`）：

```python
mask_factory_function = causal_mask_function          # :942  基底
...
if or_mask_function is not None:                      # :959
    mask_factory_function = or_masks(mask_factory_function, or_mask_function)   # :962
if and_mask_function is not None:                     # :965
    mask_factory_function = and_masks(mask_factory_function, and_mask_function) # :968
if packed_sequence_mask is not None:                  # :973
    mask_factory_function = and_masks(mask_factory_function, packed_sequence_mask_function(packed_sequence_mask))
if block_sequence_ids is not None:                    # :976
    mask_factory_function = or_masks(mask_factory_function, blockwise_overlay(block_sequence_ids))
```

所以**最终形式恒为**：

```
final = (((base  OR  or_mask_function)  AND  and_mask_function)
                  AND packed?)  OR  blockwise?
```

| 槽位 | 语义 | 直觉 |
|------|------|------|
| `or_mask_function` | **加**权限（并集） | 「除了 base 允许的，这些位置**也**允许」 |
| `and_mask_function` | **收**权限（交集） | 「在 base 的基础上，**只**保留这些位置」 |
| `packed_sequence_mask`（自动检测） | 收紧 | 不同 packed 序列之间不许互看 |
| `block_sequence_ids` | 放松 | 同一 block 内双向（图像 token、prefix-LM） |

源码在 `:956-958` 和 `:1066-1068` 反复警告这件事：

> ```
> # Note that it is very important to apply this before any other deviations of the mask (such as packed sequence mask,
> # padding mask, etc) as the resulting mask may otherwise not be correct!
> ```

**推论**：「先 AND 再 OR」是表达不出来的。要做「基础 causal ∪ 特殊放松，但整体再被某个约束卡住」，
就把整段逻辑塞进 `and_mask_function` 里自己组合（§9.3 的例子就是这么写的）。

### 3.3 已包装好的成品（3 个）

这三个是「已经用组合子拼好、可以直接当 `mask_function` 用」的：

```python
# masking_utils.py:134-138
def sliding_window_causal_mask_function(sliding_window: int) -> Callable:
    """
    This return the mask_function function to create a sliding window mask.
    """
    return and_masks(sliding_window_overlay(sliding_window), causal_mask_function)


# masking_utils.py:154-158
def sliding_window_bidirectional_mask_function(sliding_window: int) -> Callable:
    return and_masks(sliding_window_bidirectional_overlay(sliding_window), bidirectional_mask_function)


# masking_utils.py:161-165
def chunked_causal_mask_function(chunk_size: int, left_padding: torch.Tensor) -> Callable:
    return and_masks(chunked_overlay(chunk_size, left_padding), causal_mask_function)
```

**注意 `and_masks` 的参数顺序**：overlay 在前、base 在后。
因为 `and` 可交换，顺序不影响语义，但读的时候容易看反——「overlay 是**叠加物**，
base 是**底**」这个心智模型更准确。

### 3.4 Overlay vs 成品 vs 数据驱动工厂

`masking_utils.py` 里的 mask 相关工厂函数可以分三类，**别搞混**：

| 类别 | 函数 | 返回 | 用法 |
|------|------|------|------|
| **组合子** | `and_masks` `:48`、`or_masks` `:62` | 新函数 | 组合任意两个 mask |
| **基底** | `causal_mask_function` `:76`、`bidirectional_mask_function` `:83` | 完整 mask | 直接当 `mask_function=` |
| **Overlay（叠加物）** | `sliding_window_overlay` `:92`、`chunked_overlay` `:104`、`blockwise_overlay` `:116`、`sliding_window_bidirectional_overlay` `:141` | **半个** mask | 必须配基底 + `and_masks`/`or_masks`。单独用会得到荒谬结果——见下方警告 |
| **成品** | `sliding_window_causal_mask_function` `:134`、`sliding_window_bidirectional_mask_function` `:154`、`chunked_causal_mask_function` `:161` | 完整 mask | 直接当 `mask_function=` |
| **数据驱动工厂** | `padding_mask_function` `:168`、`packed_sequence_mask_function` `:182` | 完整 mask（依赖传入张量） | 由框架自动拼，也可自己传 |
| **修饰器** | `add_offsets_to_mask_function` `:193` | 新函数 | 只给 flex 用（见 §4.9） |

> ⚠️ **`sliding_window_overlay` 单独用是错的**：它只返回 `kv_idx > q_idx - sliding_window`，
> **不排除未来位置**——`q_idx=0, kv_idx=100` 也会返回 `True`。
> 源码 docstring 说得很直白（`masking_utils.py:93-96`）：
> > This is an overlay depicting a sliding window pattern.
> > **Add it on top of a causal mask for a proper sliding window mask.**
>
> 这就是为什么它叫 `overlay`（叠加物）而不是 `mask_function`。

---

## 四、原子 mask 函数逐个讲

下面全部按源码顺序，贴真实片段 + 行号。

### 4.1 `causal_mask_function` — `masking_utils.py:76-80`

```python
def causal_mask_function(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    """
    This creates a basic lower-diagonal causal mask.
    """
    return kv_idx <= q_idx
```

整个库最核心的一行。`kv_idx <= q_idx` 就是「只看过去（含自己）」。
注意**用 `<=` 而不是 `<`**——对角线（看自己）是允许的。

### 4.2 `bidirectional_mask_function` — `masking_utils.py:83-89`

见 §2.3。全允许。

### 4.3 `sliding_window_overlay` — `masking_utils.py:92-101`

```python
def sliding_window_overlay(sliding_window: int) -> Callable:
    """
    This is an overlay depicting a sliding window pattern. Add it on top of a causal mask for a proper sliding
    window mask.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return kv_idx > q_idx - sliding_window

    return inner_mask
```

**窗口大小的语义**：`kv_idx > q_idx - W` ⟺ `kv_idx - q_idx >= -(W-1)` ⟺ 向左最多看 `W-1` 个位置，
加上自己共 **`W` 个 token**。这是**包含式**计数，与 FA 的 `window_size` 参数（不含自己）不同——
换算发生在 `modeling_flash_attention_utils.py:655`：

```python
flash_kwargs["window_size"] = (sliding_window - 1, sliding_window - 1)
```

注意这里是**手写 `inner_mask` 而不是用 `and_masks`**：单个条件不需要组合，
直接写函数更快（少一层 Python 调用）。

### 4.4 `sliding_window_causal_mask_function` — `masking_utils.py:134-138`

```python
return and_masks(sliding_window_overlay(sliding_window), causal_mask_function)
```

**这就是 1M2 里 Mistral 的 SWA 的全部真相**：
`create_sliding_window_causal_mask` → 取 `config.sliding_window` → 用这个函数建 mask。
注意 Mistral 的 `sliding_window=4096` 意味着「最多看 4096 个 token」，是个**动态**值——
序列短于 4096 时，sliding window 和 full attention 等效，
所以 sdpa 路径可以直接跳过 mask 建立（§6.3）。

### 4.5 `chunked_overlay` — `masking_utils.py:104-113`

```python
def chunked_overlay(chunk_size: int, left_padding: torch.Tensor) -> Callable:
    """
    This is an overlay depicting a chunked attention pattern. Add it on top of a causal mask for a proper chunked
    attention mask.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return (kv_idx - left_padding[batch_idx]) // chunk_size == (q_idx - left_padding[batch_idx]) // chunk_size

    return inner_mask
```

「同 chunk 才能互看」——这是 Llama4 的 chunked attention。
`left_padding` 是为了处理左 padding：不减去 padding 数，chunk 边界就会错位
（测试 `tests/utils/test_masking_utils.py:213-237` 专门验证了这个 case，注释在 `:214-215`）。
`left_padding` 由 `create_chunked_causal_mask` 现场算（`masking_utils.py:1417-1420`）：

```python
if attention_mask is not None:
    # Only count the left padding tokens, not all of them
    left_padding_tokens = (attention_mask.cumsum(dim=-1) == torch.zeros_like(attention_mask)).sum(dim=-1)
else:
    left_padding_tokens = torch.zeros(batch_size, device=device, dtype=int)
```

### 4.6 `blockwise_overlay` — `masking_utils.py:116-131`

```python
def blockwise_overlay(block_sequence_ids: torch.Tensor) -> Callable:
    """
    This is an overlay depicting a blockwise masking pattern. Instead of a single
    token, each block consists of arbitrary length tokens. In causal setup, each block
    can attend to prev block causally and can't attend to future blocks. Within one block
    the attention is always bidirectional.
    Mostly used in MLLMs when non-text data attends bidirectionally to itself.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        # Unmask if the q and kv come from same group which is not -1 (i.e. non-text)
        q_group = block_sequence_ids[batch_idx, q_idx]
        kv_group = block_sequence_ids[batch_idx, kv_idx]
        return (q_group == kv_group) & (q_group >= 0)

    return inner_mask
```

`block_sequence_ids` 的形状和 input_ids 一样，同一块（如同一张图片的所有 patch）给同一个非负 id，
文本 token 给 `-1`。语义：**同一块内双向，跨块保持 causal**。

这个 overlay 通常走 **OR 槽位**（`masking_utils.py:978` / `:1220`），
所以等效于 `causal OR same-block`。生产用例见 §9.4（Gemma3）。

### 4.7 `padding_mask_function` — `masking_utils.py:168-179`

```python
def padding_mask_function(padding_mask: torch.Tensor) -> Callable:
    """
    This return the mask_function function corresponding to a 2D padding mask.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        # Note that here the mask should ALWAYS be at least of the max `kv_index` size in the dimension 1. This is because
        # we cannot pad it here in the mask_function as we don't know the final size, and we cannot try/except, as it is not
        # vectorizable on accelerator devices
        return padding_mask[batch_idx, kv_idx]
```

注意两点：

- **只看 `kv_idx`**：padding token 不能被「看到」。至于 padding 行自己会不会全被屏蔽——
  那由 causal 决定（左 padding 时第 0 行确实会全被屏蔽），见 §9.2 的网格图。
- **`padding_mask` 必须预先 pad 到够长**：因为 mask_function 里不能做 try/except，
  也不能现算长度（不可向量化）。所以有 `prepare_padding_mask`（`:205-214`）在外面补齐：

```python
def prepare_padding_mask(attention_mask: torch.Tensor | None, kv_length: int, kv_offset: int) -> torch.Tensor | None:
    local_padding_mask = attention_mask
    if attention_mask is not None:
        # Pad it if necessary
        if (padding_length := kv_length + kv_offset - attention_mask.shape[-1]) > 0:
            local_padding_mask = torch.nn.functional.pad(attention_mask, (0, padding_length))
    return local_padding_mask
```

padding 是**框架自动拼**的，不需要你手动传：

- `sdpa_mask`：`masking_utils.py:504-505`
  ```python
  if padding_mask is not None:
      mask_function = and_masks(mask_function, padding_mask_function(padding_mask))
  ```
- `flex_attention_mask`：`masking_utils.py:684-693`（逻辑一样，多了 torch 2.5 的 block 对齐补零）

### 4.8 `packed_sequence_mask_function` — `masking_utils.py:182-190`

```python
def packed_sequence_mask_function(packed_sequence_mask: torch.Tensor) -> Callable:
    """
    This return the mask_function function corresponding to a 2D packed sequence mask.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return packed_sequence_mask[batch_idx, q_idx] == packed_sequence_mask[batch_idx, kv_idx]
```

「同一条序列的 token 才能互看」。与 `blockwise_overlay` 结构完全一样，
区别是语义与**触发方式**：

| | packed | blockwise |
|---|---|---|
| 谁触发 | **框架自动检测** `position_ids` 是否连续（`:854`） | 调用方显式传 `block_sequence_ids=` |
| 槽位 | `and_masks`（收紧，`:974`） | `or_masks`（放松，`:978`） |
| 等价语义 | 「跨序列不许互看」的额外收紧 | 「同块内双向」的额外放松 |

检测函数 `find_packed_sequence_indices`（`:728-757`）用差分 + `cumsum` 找出序列边界：

```python
first_dummy_value = position_ids[:, :1] - 1  # We just need the diff on this first value to be 1
position_diff = torch.diff(position_ids, prepend=first_dummy_value, dim=-1)
packed_sequence_mask = (position_diff != 1).cumsum(-1)

# Sadly this is a dynamic control flow, so we cannot enable this check on anything compile related
if not is_tracing(packed_sequence_mask) and (packed_sequence_mask[:, -1] == 0).all():
    return None
```

测试 `tests/utils/test_masking_utils.py:147-150` 给出了确定答案：

```python
position_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 0, 1, 2, 3]])
EXPECTED_SEQUENCE_INDICES = torch.tensor([[0, 0, 0, 0, 1, 1, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]])
```

### 4.9 `add_offsets_to_mask_function` — `masking_utils.py:193-202`

```python
def add_offsets_to_mask_function(mask_function: Callable, q_offset: int, kv_offset: int) -> Callable:
    """
    This function adds the correct offsets to the `q_idx` and `kv_idx` as the torch API can only accept lengths,
    not start and end indices.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return mask_function(batch_idx, head_idx, q_idx + q_offset, kv_idx + kv_offset)

    return inner_mask
```

**只有 flex 后端需要它**（`masking_utils.py:696`）：

```python
# Add the offsets on top (because flex interface only allows length, not start and end indices)
mask_function = add_offsets_to_mask_function(mask_function, q_offset, kv_offset)
```

原因：`create_block_mask(mask_mod, Q_LEN=q_length, KV_LEN=kv_length)` 只接受长度，
没法告诉 torch「我是从第 1024 个 token 开始解码的」。所以偏移量被搬进函数内部——
这就是「mask 是函数」带来的灵活性：**一个参数化的闭包可以携带任何上下文**。

### 4.10 关于「`_ignore_mask` / 全 False mask」

**实测：不存在。**

```bash
$ grep -rn "_ignore_mask\|all_false\|full_false" src/transformers/masking_utils.py src/transformers/modeling_attn_mask_utils.py
(no output)
```

名字里带 `_ignore_` 的只有三个 **跳过判断** 函数，它们返回的是 `bool`（要不要跳过建 mask），
**不是** mask：

| 函数 | 行号 | 作用 |
|------|------|------|
| `_ignore_causal_mask_sdpa` | `:235-278` | sdpa 专用：这个 causal mask 能不能不建，改用 `is_causal=True` |
| `_ignore_bidirectional_mask_sdpa` | `:308-336` | sdpa 专用：双向 mask 能否跳过（无 padding + 无局部窗口） |
| `_can_skip_bidirectional_mask_xpu` | `:281-305` | XPU 上的分支版本 |

「全 False」这种效果用 `or_masks()`（空参）就能得到（`:68` 起点是 `new_zeros`），
库里没有专门函数，因为没人需要——**全屏蔽的 mask 没有意义**。

---

## 五、四个「组装好的」入口函数

`masking_utils.py` 提供 5 个 `create_*_mask`，模型直接调用：

| 入口 | 行号 | 基底 | 谁在用 |
|------|------|------|--------|
| `create_causal_mask` | `:864-996` | `causal_mask_function` | Llama、Qwen、DeepSeek…（默认） |
| `create_bidirectional_mask` | `:999-1099` | `bidirectional_mask_function` | BERT / 编码器 / cross-attention |
| `create_sliding_window_causal_mask` | `:1102-1239` | `sliding_window_causal_mask_function` | Mistral、Gemma2/3 的局部层 |
| `create_bidirectional_sliding_window_mask` | `:1242-1339` | `sliding_window_bidirectional_mask_function` | 双向模型 + SWA、ESM 类 |
| `create_chunked_causal_mask` | `:1342-1471` | `chunked_causal_mask_function` | Llama4 |
| `create_recurrent_attention_mask` | `:1474-1504` | —（不做 mask_function） | Mamba / linear attention，**只返回 2D padding mask** |

### 5.1 参数逐个说

以 `create_causal_mask`（`:864-875`）为模板：

```python
def create_causal_mask(
    config: PreTrainedConfig,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None,
    position_ids: torch.Tensor | None = None,
    or_mask_function: Callable | None = None,
    and_mask_function: Callable | None = None,
    block_sequence_ids: torch.Tensor | None = None,
    layer_idx: int | None = None,
    allow_is_causal_skip: bool = True,
) -> torch.Tensor | BlockMask | None:
```

| 参数 | 必需? | 作用 | 关键代码 |
|------|-------|------|----------|
| `config` | ✅ | 提供 `_attn_implementation`（选后端）、`is_causal`、`sliding_window`、`attention_chunk_size` | `:917`（`is_causal` 回退）、`:943`（选后端）、`:1180`（取窗口） |
| `inputs_embeds` | ✅ | **只用来读元信息**：`batch_size` / `q_length` / `dtype` / `device`。**不看数值** | `:941` `batch_size, dtype, device = inputs_embeds.shape[0], ...`；`:826` `q_length = inputs_embeds.shape[1]` |
| `attention_mask` | ✅（可 None） | 2D `[B, S]` 的 padding mask。**如果已经是 4D，直接原样返回**（用户自定义 mask 的入口） | `:811-812` |
| `past_key_values` | ✅（可 None） | 提供 `q_offset` / `kv_length` / `kv_offset`——**mask 的绝对位置来源** | `:828-833` |
| `position_ids` | 可省 | **只用于 packed sequence 检测**（见 §5.2） | `:854-859` |
| `or_mask_function` / `and_mask_function` | 可省 | 自定义 pattern 的 OR / AND 槽位。**用了就强制 `use_vmap=True` 且要求 torch≥2.6** | `:959-970` |
| `block_sequence_ids` | 可省 | blockwise overlay（OR 槽位） | `:976-979` |
| `layer_idx` | 可省 | 混合 cache 里选哪一层来定尺寸（不同层的 seen tokens 可能不同） | `:929-933` |
| `allow_is_causal_skip` | 默认 `True` | 是否允许「返回 `None`，让 sdpa 用 `is_causal`」 | `:874`、`:952`、`:990` |

### 5.2 `position_ids` 到底干什么 —— 一个必须纠正的直觉

很多人（包括 1M2 的读者）会以为 `position_ids` 是给 mask 提供**绝对位置**的，
因为「sliding window / cache 判断需要绝对位置，不能靠 arange」。**实测结论是：在 mask 层面，这个说法不成立。**

`position_ids` 在 `masking_utils.py` 里的**全部**用处只有一处（`:851-859`）：

```python
# We check the position_ids for potential packed sequence format (only if the 2D attention mask is explicitly None,
# and we don't have past_key_values, i.e. generally a training setup)
packed_sequence_mask = None
if position_ids is not None and attention_mask is None and past_key_values is None:
    batch_size = inputs_embeds.shape[0]
    # The position ids are sometimes just unsqueezed, without being expanded
    if batch_size != position_ids.shape[0]:
        position_ids = position_ids.expand(batch_size, -1)
    packed_sequence_mask = find_packed_sequence_indices(position_ids)
```

（`create_bidirectional_mask` / `create_bidirectional_sliding_window_mask` 甚至直接传 `None`：
`:1047`、`:1290`，因为编码器不 packing。）

**那「绝对位置」从哪来？** 来自 **cache 给出的偏移量**，然后套在 `arange` 上（`masking_utils.py:507-510`）：

```python
batch_arange = torch.arange(batch_size, device=device)
head_arange = torch.arange(1, device=device)
q_arange = torch.arange(q_length, device=device) + q_offset
kv_arange = torch.arange(kv_length, device=device) + kv_offset
```

`q_offset` / `kv_offset` 来自（`masking_utils.py:828-833`）：

```python
if past_key_values is not None:
    q_offset = past_key_values.get_query_offset(layer_idx)
    ...
    kv_length, kv_offset = past_key_values.get_mask_sizes(q_length, layer_idx)
```

也就是说：**「我算到第几个 token 了」这件事由 cache 回答，不由 `position_ids` 回答。**
（cache 的这两个方法属于 2A3 的主题，见 [2A3_kv_cache.md](./2A3_kv_cache.md)。）

那为什么模型代码里 `position_ids` 看起来是必需的？因为它至少有两个**别的**用途：

1. **RoPE**：`models/llama/modeling_llama.py:400` `position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)`。
2. **FlashAttention 的变长推导**：`modeling_flash_attention_utils.py:531`
   `prepare_fa_kwargs_from_position_ids(position_ids)` → `cu_seqlens`，
   源码注释在 `:490` 明确说：
   > We should use `cu_seq_lens` instead of `position_ids` to get the max length since position_ids is not always increasing

所以准确的表述是：

| 层 | `position_ids` 的作用 |
|----|----------------------|
| **mask 层（本篇）** | **只做 packed 检测**。绝对位置由 cache 的 `q_offset`/`kv_offset` + `arange` 提供 |
| **RoPE** | 必需：决定旋转角度 |
| **FA2** | 必需：从位置差分推出 `cu_seqlens`，实现 unpadded 变长 attention |

而模型里的默认值恰好是从 cache 推出来的（`models/llama/modeling_llama.py:386-389`）：

```python
if position_ids is None:
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
    position_ids = position_ids.unsqueeze(0)
```

> 顺带一提：**`past_seen_tokens` 和 `q_offset` 是同一件事的两个名字**，
> 一个给 RoPE 用，一个给 mask 用。两者必须一致，否则 RoPE 和 mask 会对不上——
> PEFT 的 prefix tuning 是这套「双账本」最容易被搞坏的地方，
> 源码在 `masking_utils.py:842-849` 专门为它留了注释和分支：
> > This is a very unique and special case where an encoder utilizes a cache and expects its length
> > to be accounted for ... **Conclusion: "prefix tuning is evil"**

### 5.3 `_preprocess_mask_arguments`：入口的共同前处理

`:760-861`，所有 `create_*_mask` 第一件事就是调它。返回 7 元组（`:861`）：

```python
return False, attention_mask, packed_sequence_mask, q_length, kv_length, q_offset, kv_offset
```

三个**提前退出**（early exit）条件：

| 条件 | 行号 | 返回 |
|------|------|------|
| 传入的 `attention_mask` 已经是 4D | `:811-812` | **原样返回**（用户自定义 mask） |
| `config._attn_implementation` 不在 `ALL_MASK_ATTENTION_FUNCTIONS` 里 | `:819-820` | **`None`** |
| | | |

第二条非常重要，它解释了「TGI / vLLM 为什么不建 mask」（`:814-818` 原文注释）：

> ```
> # For TGI/vLLM backends, or other custom attention without equivalent mask creation: we don't need a mask!
> # Note: it's not ideal to check the `_global_mapping` attribute instead of the object itself, however otherwise
> # full graph dynamo tracing (i.e. torch.export or compile with `fullgraph=True`) will fail on Python<3.11
> ```

对应的人类向文档（`docs/source/en/attention_interface.md:135`）说得更明白：

> [!WARNING]
> Register a matching attention mask function when you register a custom attention function.
> If the custom `attn_implementation` name is not registered in [`AttentionMaskInterface`],
> Transformers skips mask creation and passes `attention_mask=None` to the attention layers.
> Your attention function must handle causal, padding, packing, or sliding-window constraints itself,
> **or those constraints can be silently dropped**.

**「静默丢弃约束」**——这是全套 mask 机制里最危险的一个坑。自定义后端必须同时注册
`AttentionMaskInterface`（见 §7.6）。

### 5.4 `create_masks_for_generate`：generate 的预建 mask

`:1523-1601`。为什么需要它？因为在 StaticCache + `torch.compile` 下，
**mask 必须在编译前就建好**，否则每步都在主图里建 mask 会导致重编。
调用点在 `generation/utils.py:580-598`：

```python
if (
    isinstance(past_key_values, Cache)
    and past_key_values.is_compileable
    and attention_mask is not None
    and attention_mask.ndim == 2
):
    # Some models may overwrite the general one
    causal_mask_creation_function = getattr(self, "create_masks_for_generate", create_masks_for_generate)
    attention_mask = causal_mask_creation_function(
        config=self.config,
        # we only need batch size, seq_length, dtype and device here - so we pass a 0-sized tensor with only the metadata
        inputs_embeds=torch.empty((batch_size, sequence_length, 0), dtype=self.dtype, device=input_ids.device),
        ...
```

注意那个 **0 尺寸的 `inputs_embeds`**——再次印证 §5.1 的表：这个参数只用来读元信息。

它按 `config.layer_types` 逐层类型分发（`masking_utils.py:1577-1593`），
映射表在 `:1507-1520`：

```python
LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING = {
    "full_attention": create_causal_mask,
    "sliding_attention": create_sliding_window_causal_mask,
    "chunked_attention": create_chunked_causal_mask,
    "compressed_sparse_attention": create_sliding_window_causal_mask,
    "heavily_compressed_attention": create_sliding_window_causal_mask,
    "minimax_m3_sparse": create_causal_mask,
    "deepseek_sparse_attention": create_causal_mask,
    "qwen_sparse_attention": create_causal_mask,
    "linear_attention": create_recurrent_attention_mask,
    "conv": create_recurrent_attention_mask,
    "hybrid": {"full_attention": create_causal_mask, "linear_attention": create_recurrent_attention_mask},
    "hybrid_sliding": {"sliding_attention": create_causal_mask, "linear_attention": create_recurrent_attention_mask},
}
```

返回的是一个 **dict**（`{"full_attention": mask_a, "sliding_attention": mask_b}`），
模型里就是这样按 `layer_types[i]` 取的（`models/gemma2/modular_gemma2.py:365-368`）：

```python
causal_mask_mapping = {
    "full_attention": create_causal_mask(**mask_kwargs),
    "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
}
```

Gemma2 的交替模式由此而来——**交替发生在「层」这一维，不是 mask 内部**。
`layer_types` 的生成逻辑在 `models/gemma2/configuration_gemma2.py:95-98`：

```python
if self.layer_types is None:
    self.layer_types = [
        "sliding_attention" if bool((i + 1) % 2) else "full_attention" for i in range(self.num_hidden_layers)
    ]
```

> 📌 **这纠正了任务描述里一个常见误解**：Gemma2 并没有「前 3 个 token 全局、后面滑动窗口」这种 pattern。
> Gemma2 是**逐层交替**（奇数层滑动、偶数层全局）。真正的「某些 token 全局」是
> **Gemma3 的 blockwise 图像 token** 和 **prefix-LM 式做法**，前者是真实生产代码（§9.4），
> 后者需要你自己组合（§9.3 给出骨架）。

---

## 六、`create_causal_mask` 内部的选择逻辑（关键实测）

这一节回答一个具体问题：**它会不会根据 `config._attn_implementation` 或序列长度，
决定「返回函数」还是「返回物化张量」？**

### 6.1 完整流水线

```
create_causal_mask(config, inputs_embeds, attention_mask,
                   past_key_values, position_ids, or_mask_function, ...)
│
├─ [1] config.is_causal == False ?                                  :917
│        └─ 转发 create_bidirectional_mask(...)                      :918-926
│
├─ [2] layer_idx 未指定 → 从 past_key_values.is_sliding 挑一层       :929-933
│        （混合 cache：找第一个「非滑动」层；sliding 版本找第一个滑动层 :1168-1172）
│
├─ [3] _preprocess_mask_arguments(...)                              :935-937
│        ├─ attention_mask 已是 4D    → 原样返回（early exit）       :811
│        └─ 后端未注册 mask 接口      → 返回 None（early exit）      :819
│
├─ [4] 基底 + 后端查表                                               :942-943
│        mask_factory_function = causal_mask_function
│        mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]
│
├─ [5] 组合（顺序固定：先 OR 后 AND，见 §3.2）                        :959-979
│        or_mask_function    → or_masks(...)  ，并 use_vmap=True      :962-964
│        and_mask_function   → and_masks(...) ，并 use_vmap=True      :968-970
│        packed_sequence_mask→ and_masks(...)                        :974
│        block_sequence_ids  → or_masks(...)                         :978
│
├─ [6] allow_is_causal_skip 收紧                                     :952-954
│        （compileable cache + q_length==1 → 禁止跳过）
│
└─ [7] mask_interface(...) ← ★ 唯一真正决定输出形态的地方             :982-995
         传参：batch_size, q_length, kv_length, q_offset, kv_offset,
               mask_function, attention_mask, allow_is_causal_skip,
               dtype, config, use_vmap, device
         └─ return causal_mask                                       :996
```

### 6.2 谁决定「函数」还是「物化」——三个层次的答案

**答案分三层，必须分开说，否则会读错。**

**第 1 层：`config._attn_implementation` 决定「用哪个 interface」**
—— `masking_utils.py:943`：

```python
mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]
```

这是**唯一**根据后端做分支的地方。查表结果决定输出形态（§7 的对照表）。

**第 2 层：只有 `flex_attention` 把 mask 保持为「函数」形态**
—— `flex_attention_mask` 把 `mask_function` 直接交给 PyTorch（`:699-707`）：

```python
block_mask = create_block_mask(
    mask_mod=mask_function,
    B=batch_size,
    H=None,
    Q_LEN=q_length,
    KV_LEN=kv_length,
    device=device,
    _compile=_is_torch_greater_or_equal_than_2_6,
)
return block_mask
```

`BlockMask` 内部持有的就是那个 `mask_mod` 回调，**没有被展开成 `[B,1,S,S]`**。
其余三个后端（`sdpa_mask` / `eager_mask` / `flash_attention_mask`）都在函数内部完成物化或转交。

> **更重要的结论**：`create_causal_mask` **从不返回 `mask_function`**。
> 返回类型签名就写死了（`:875`）：`torch.Tensor | BlockMask | None`。
> 「mask 是函数」是**实现内部的一种表达方式**，不是对外的 API 形态。

**第 3 层：序列长度不改变「形态」，但决定「要不要建」**
—— 通过 `allow_is_causal_skip` 传到 `sdpa_mask`，再进 `_ignore_causal_mask_sdpa`。
命中就直接 `return None`（`masking_utils.py:496-499`）：

```python
if allow_is_causal_skip and _ignore_causal_mask_sdpa(
    padding_mask, q_length, kv_length, q_offset, kv_offset, local_size
):
    return None
if allow_is_bidirectional_skip and _ignore_bidirectional_mask_sdpa(padding_mask, kv_length, local_size):
    return None
```

### 6.3 「什么时候 mask 会变成 `None`」完整表

来自 `_ignore_causal_mask_sdpa`（`:235-278`）和 `_ignore_bidirectional_mask_sdpa`（`:308-336`）。
这是**性能上最关键的一张表**——`None` 意味着 sdpa 能派发到 flash kernel。

| 条件 | 行号 | 结果 |
|------|------|------|
| 正在 tracing（`torch.export` / ONNX） | `:259-260` | **不跳过**（`is_causal` 会被硬编码进图，见 `:255-258` 注释） |
| 有局部窗口且 `kv_length >= local_attention_size` | `:262-263` | **不跳过**（窗口本身就得靠 mask 表达） |
| `q_length == 1`（解码）**或** `kv_length == q_length`（prefill），且无 padding（或 padding 全 1） | `:268-269` | **跳过 → `None`**，改用 sdpa 的 `is_causal` |
| `q_offset == 0`（cache 空）且 padding 只出现在「kv 超出 q 的那一段」（StaticCache 的未来位） | `:273-276` | **跳过 → `None`**（靠 sdpa `is_causal` 的**左上对齐**） |
| 双向 + 无 padding + `kv_length < local_attention_size` | `:328-334` | **跳过 → `None`**（无 bias 可加） |
| XPU 分支 | `:281-305` | 同上逻辑的独立实现 |

**两条 `q_length == 1` 的注释值得抄下来**（`:265-267`）：

> ```
> # If `q_length == 1`, we then use `is_causal=False` in sdpa integration to mimic lower-right alignment.
> # If `kv_length == q_length`, we use `is_causal=True` as upper-left alignment (torch's default) is the same as
> # lower-right in this case. If we have padding, we need to add padding to the mask, so cannot be skipped
> ```

这解释了 sdpa 里一个反直觉的判断（`integrations/sdpa_attention.py:130`）：

```python
is_causal = q_length > 1 and attention_mask is None and is_causal
```

**解码时 `is_causal` 必须是 `False`**——因为单 query token 要看到**全部**历史 KV，
`is_causal=True` 会让它只看最后一个（左上对齐的三角对 1×N 来说恰好只留最后一列）。

**另一条限制**（`masking_utils.py:950-954`，`:1192-1196`，`:1428-1432` 各出现一次）：

```python
# Do not allow skip if we are compiling and decoding (but for prefill, we still allow skip to optimize the perfs since
# prefill is not compiled)
allow_is_causal_skip = allow_is_causal_skip and not (
    getattr(past_key_values, "is_compileable", False) and q_length == 1
)
```

**compileable cache（StaticCache）+ 解码时不能跳过**：
否则 `is_causal` 的取值会随 `q_length` 变成符号量，图不稳定。
测试 `tests/utils/test_masking_utils.py:173-183` 验证了这一点（`torch.compile` 下 mask 不能被跳过）。

### 6.4 `use_vmap` 是什么

`:948` 默认 `False`，但一旦传入 `or_mask_function` / `and_mask_function` 就强制 `True`（`:964`、`:970`）。
两条展开路径都在 `sdpa_mask` 里：

| 路径 | 代码 | 适用 |
|------|------|------|
| `use_vmap=False`（默认，快） | `:514-518` | 纯 index-based 函数，靠 `_non_vmap_expansion_sdpa`（`:352-368`）广播 |
| `use_vmap=True`（慢但通用） | `:521-526` | 任意函数（可以含 `cumsum`、`where`、查表…），靠 `torch.vmap`（`_vmap_expansion_sdpa`，`:339-349`） |
| `use_vmap=True` 但 torch<2.6 | `:529-533` | **抛错** |

```python
# masking_utils.py:514-518（快路径）
if not use_vmap:
    # Apply mask function element-wise through broadcasting
    attention_mask = mask_function(*_non_vmap_expansion_sdpa(batch_arange, head_arange, q_arange, kv_arange))
    # Expand the mask to match batch size and query length if they weren't used in the mask function
    attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)

# masking_utils.py:521-526（vmap 路径）
elif _is_torch_greater_or_equal_than_2_6:
    with TransformGetItemToIndex():
        attention_mask = _vmap_expansion_sdpa(mask_function)(batch_arange, head_arange, q_arange, kv_arange)
```

`TransformGetItemToIndex`（`:41-42` 导入）是 vmap 必需的上下文管理器，
因为 vmap 不能处理「从张量里切标量再 `.item()`」（`:522-524` 注释）。

**实践含义**：自定义 mask function 哪怕写得很朴素，也能跑；
但**必须 torch≥2.6**，否则直接 `ValueError`（`:960-961`）。

---

## 七、三种 attention 接口下的 mask 形态

### 7.1 对照总表

| 后端（`config._attn_implementation`） | interface 函数 | 行号 | 输出类型 | 形状 | `True`/值 的含义 |
|---|---|---|---|---|---|
| `"eager"` | `eager_mask` | `:538-604` | **float 张量** | `[B, 1, Q, KV]` | `0.0` = 允许，`finfo(dtype).min` = 屏蔽 |
| `"sdpa"` | `sdpa_mask` | `:372-535` | **bool 张量** 或 `None` | `[B, 1, Q, KV]` | `True` = 允许 |
| `"flash_attention_2/3/4"` | `flash_attention_mask` | `:607-647` | **2D bool 张量** 或 `None` | `[B, KV]` | `True` = 有效 token（非 pad） |
| `"flex_attention"` | `flex_attention_mask` | `:650-708` | **`BlockMask`** | 块稀疏结构 | 内含 `mask_mod` 函数 |
| 未注册（TGI/vLLM/custom） | — | `:819-820` | **`None`** | — | 约束由后端自己负责 |

### 7.2 eager：必须物化成加性 bias

```python
# masking_utils.py:583-604
# The masks for eager attention are simply boolean mask from sdpa, casted to 0 and -inf
_ = kwargs.pop("allow_is_causal_skip", None)
mask = sdpa_mask(
    ...
    allow_is_causal_skip=False,
    ...
)
# only bidirectional masks can be skipped, otherwise we convert bool -> float
if mask is not None:
    min_dtype = torch.finfo(dtype).min
    # we need 0s where the tokens should be taken into account, and -inf otherwise (mask is already of boolean type)
    mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
return mask
```

**三件事**：

1. eager 的实现就是「调 sdpa_mask 再转 float」——**没有独立的 mask 逻辑**，一份 pattern 描述两种后端共用。
2. `_ = kwargs.pop("allow_is_causal_skip", None)`（`:584`）**把跳过的可能性直接扔掉**。
   所以 **eager 下的 causal mask 永远会被物化**，哪怕 `q_length == 1`。
   这就是「eager 慢且吃显存」在 mask 这一层的具体体现。
3. 用 `torch.finfo(dtype).min`（有限大负数，如 `-3.4e38`）而**不是** `-inf`。
   副作用见 §9.2 的注：整行被屏蔽时 `softmax` 得到均匀分布而不是 `NaN`。

消费方（`models/llama/modeling_llama.py:204-206`）：直接相加。

### 7.3 sdpa：bool 或加性都能吃

`sdpa_mask` 返回 **bool** `[B,1,Q,KV]`（`:535`），直接交给 PyTorch：

```python
# integrations/sdpa_attention.py:158-167
attn_output = torch.nn.functional.scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=attention_mask,   # ← bool（True=参与）或 float 加性，都接受
    dropout_p=dropout,
    scale=scaling,
    is_causal=is_causal,
    **sdpa_kwargs,
)
```

三种情况（`docs/source/en/attention_interface.md` 的表格 + 代码）：

| `attention_mask` | 结果 | 代码 |
|---|---|---|
| `None` | 走 `is_causal`，可能派发到 flash kernel | `sdpa_attention.py:130` |
| bool 张量 | 直接传 `attn_mask`（PyTorch 接受 bool） | `:163` |
| float 张量（用户自定义 4D） | 直接传 `attn_mask`（加性语义） | `:163` |
| 有 `position_bias`（ALiBi/T5 相对位置） | 合成一张 float mask | `create_position_bias_mask`，`integrations/sdpa_attention.py:49-77` |

### 7.4 flash：根本不用 4D mask

这是最反直觉的一档。FA2 是 **unpadded** 的：它把 `[B, S, H, D]` 压成 `[total_tokens, H, D]`，
用 `cu_seqlens` 描述变长边界。所以它只需要**一个 2D 的 padding mask** 来算 `cu_seqlens`：

```python
# masking_utils.py:607-647
def flash_attention_mask(batch_size, q_length, kv_length, q_offset=0, kv_offset=0,
                        mask_function=causal_mask_function, attention_mask=None, **kwargs):
    """
    Create the attention mask necessary to use FA2. Since FA2 is un-padded by definition, here we simply return
    `None` if the mask is fully causal, or we return the 2D mask which will then be used to extract the seq_lens.
    We just slice it in case of sliding window.
    """
    if attention_mask is not None:
        # Here we need to slice from the right if using sliding or chunked (for full attention, this is equivalent to doing nothing)
        attention_mask = attention_mask[:, -kv_length:]
        if attention_mask.shape[1] == kv_length and attention_mask.all():
            attention_mask = None

    return attention_mask
```

**causal / sliding window 完全不靠 mask 表达**，而是变成内核参数
（`modeling_flash_attention_utils.py:642-655`）：

```python
flash_kwargs = {
    "causal": is_causal and not (use_top_left_mask and query_length == 1),
    "softmax_scale": softmax_scale,
}
...
if supports_mapping["window_size"] and sliding_window is not None and key_length > sliding_window:
    # The flash attention API sets inclusive boundaries, i.e. (4, 0) would take 4 tokens to the left
    # and the current token for a total size of 5. However, we usually define our window sizes by
    # their total window size (when causal). Encoder models as of now seldom use SWA and when they
    # do, they must align with this symmetric logic, i.e. for a total of `2*sliding_window + 1`.
    flash_kwargs["window_size"] = (sliding_window - 1, sliding_window - 1)
```

那个 `-1` 就是 §4.3 提到的**包含式 vs 排他式**换算。
`sliding_window` 参数是 attention 层传下去的（`models/mistral/modular_mistral.py:86`）：
`sliding_window=getattr(self.config, "sliding_window", None)`。

**所以 `create_causal_mask` 传给 `flash_attention_mask` 的 `mask_function` 被完全忽略**
（除了 padding，而 padding 是模型传入的 2D mask 本身）。这是一个真实存在的不对称：
**同一个 pattern，在 sdpa 上写成 mask，在 flash 上写成内核参数。**

### 7.5 flex：唯一吃「函数」的后端

```python
# integrations/flex_attention.py:280-285
block_mask = None
score_mask = None
if isinstance(attention_mask, BlockMask):
    block_mask = attention_mask
else:
    score_mask = attention_mask
```

`BlockMask` 直接喂给 `compile_friendly_flex_attention(..., block_mask=block_mask, ...)`（`:321-331`）。
如果拿到的是普通张量（比如用户自定义 4D），就退化成 `score_mod` 里的加法（`:290-300`）——
**能跑，但丢掉了块稀疏的加速**。

### 7.6 想加自定义后端：必须同时注册两个接口

```python
# docs/source/en/attention_interface.md:145-153
AttentionInterface.register("my_new_sdpa", my_new_sdpa)
AttentionMaskInterface.register("my_new_sdpa", sdpa_mask)  # must have the same name as the registered attention function
```

`AttentionMaskInterface` 在 `masking_utils.py:711-721`，
它的孪生兄弟 `AttentionInterface` 在 `modeling_utils.py:5147`。
两个注册表有一个**细微但重要**的差异：

| | `ALL_ATTENTION_FUNCTIONS`（`modeling_utils.py:5156-5167`） | `ALL_MASK_ATTENTION_FUNCTIONS`（`masking_utils.py:714-721`） |
|---|---|---|
| `"eager"` 在映射里吗 | **不在**（靠调用方传的 default 兜底） | **在**：`"eager": eager_mask` |
| paged 变体 | 有 `"paged|sdpa"` 等 4 个 | **没有** |

所以 `ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]`（注意是 `[]` 不是 `.get_interface()`）
遇到 `"paged|sdpa"` 会怎样？回到 `:819`——**不在映射里 → 直接返回 `None`**，不建 mask。
这正是 paged attention 想要的（它自带 block table 语义）。

**「两边 key 必须严格对应」这条规则适用于「新注册的后端」**，而不是内置映射表——
内置表存在上面两处刻意的非对称。诚实的做法是**成对注册**
（[2A1](./2A1_attention_backends.md) 提到的 `integrations/hub_kernels.py:726-727` 就是一行注册两张表）：

```python
# src/transformers/integrations/hub_kernels.py:726-727
ALL_ATTENTION_FUNCTIONS.register(attn_implementation, kernel_function)
ALL_MASK_ATTENTION_FUNCTIONS.register(attn_implementation, ALL_MASK_ATTENTION_FUNCTIONS[mask_implementation])
```

注意右边那行：**新内核直接复用某个已有后端的 mask 实现**（`mask_implementation`，常见是 `"sdpa"`）。
这就是「一张表决定 attention 怎么算，另一张表决定 mask 长什么样」的落地方式。

---

## 八、`modeling_attn_mask_utils.py` 现在还剩什么

### 8.1 它是一份**已废弃**的实现

文件头（`modeling_attn_mask_utils.py:14-18`）就是一份判决书：

```python
"""
IMPORTANT NOTICE: Every class and function in this file is deprecated in favor of using the much more general
`masking_utils.py` primitives. New code should not rely on it, it is only kept for backward compatibility for now,
and will be removed in the future.
"""
```

`AttentionMaskConverter.__init__` 第一行就 `warnings.warn(DEPRECATION_MESSAGE, FutureWarning)`
（`:70-71`），`DEPRECATION_MESSAGE`（`:29-32`）：

> ```
> "The attention mask API under `transformers.modeling_attn_mask_utils` (`AttentionMaskConverter`) "
> "is deprecated and will be removed in Transformers v5.10. Please use the new API in `transformers.masking_utils`."
> ```

### 8.2 实测：src 里已经没有任何人 import 它

```bash
$ grep -rn "modeling_attn_mask_utils" src/ tests/ docs/ utils/   # ← 只看 .py，零命中（除自身）
$ grep -rn "modeling_attn_mask_utils" . | grep -v "^./.git/" | wc -l
4
```

那 4 处是：

| 位置 | 性质 |
|------|------|
| `src/transformers/modeling_attn_mask_utils.py:30, 48` | 文件**自己**（deprecation 字符串 + docstring 示例） |
| `.github/workflows/push-important-models.yml:108` | CI 的「核心文件」监视列表 |
| `reading_docs/00_roadmap.md:77` | 本篇的路线图条目 |

**结论：它是纯粹的兼容层，不是死代码但也不是活路径。** 具体说：

- 它**没有进** `src/transformers/__init__.py`（`grep "modeling_attn_mask_utils" __init__.py` 无命中），
  所以不是公开 API。对照：`masking_utils` 的 `AttentionMaskInterface` **是**公开导出的
  （`src/transformers/__init__.py:459`、`:645`）。
- 它仍然在一个 CI 监视列表里，意味着**改它会触发全模型回归**——这是它「还活着」的唯一证据。
- `v5.10` 的删除承诺**已经过期**（当前 `__version__ = "5.16.0.dev0"`，`src/transformers/__init__.py:21`），
  但文件还在。所以别指望它明天就消失。

### 8.3 但它的**思想**仍然活着

值得对照读一遍的是：`masking_utils.py` 里几乎所有「跳过判断」都在旧文件里有原型。

| 概念 | 旧（`modeling_attn_mask_utils.py`） | 新（`masking_utils.py`） |
|------|-----------------------------------|------------------------|
| causal mask 构造 | `_make_causal_mask`（静态方法，`:162`） | `causal_mask_function`（纯函数，`:76`） |
| padding 展开 | `_expand_mask`（把 `[B,S]` 展开成 `[B,1,S,S]`，`:198`） | `padding_mask_function`（只查 `[b, kv]`，`:168`） |
| sdpa 跳过判断 | `AttentionMaskConverter._ignore_causal_mask_sdpa`（静态方法，`:265`） | `_ignore_causal_mask_sdpa`（模块级函数，`:235`） |
| 空行修复 | `_unmask_unattended`（`:214`） | **无对应实现** |
| 4D mask 生成 | `to_causal_4d` / `to_4d`（`:81` / `:114`） | 删除（**不再需要 4D mask**） |

**新旧哲学的分水岭在于 `_expand_mask` 的消失**：旧世界里 mask 的**原生形态就是 4D 张量**，
所以 `[B,S]` 的 padding 必须「展开」；新世界里 padding 只是一条 `return padding_mask[b, kv]` 的规则，
展开这件事推迟到后端真正需要时（`sdpa_mask` 的广播）才做，而且**只对需要的维度做**。

> ⚠️ **一处行为差异，值得记一笔**：旧路径在 sdpa 上会调 `_unmask_unattended`（`:214-261`）
> 处理「整行被屏蔽」（左 padding 时 pad token 那一行）。
> 新路径 `grep -rn "unmask" src/transformers/masking_utils.py` **零命中**。
>
> 后果：eager 路径用 `finfo.min` 这种**有限**大负数，整行 softmax 退化为**均匀分布**（不产生 NaN）；
> 而 sdpa 收到的是 **bool** mask，整行全 `False` 的语义由 PyTorch 内核决定。
> **实践建议**：不要把 pad token 位置的输出当成有意义的值（训练时 loss mask 掉、推理时忽略），
> 这在两套实现下都是安全做法。

---

## 九、实操：给定 pattern 怎么写 mask

### 9.1 先看清三层叠加后的真实可看区域

以 `S = 8`、`sliding_window = 3`、batch 0 有 **2 个左 padding token** 为例。
`●` = 可看，`·` = 被 mask，`⊗` = **本来是 `●`，但被 padding 砍掉**。

**第 1 层：只叠 causal**（`kv_idx <= q_idx`）

```
        kv0 kv1 kv2 kv3 kv4 kv5 kv6 kv7
  q0     ●   ·   ·   ·   ·   ·   ·   ·
  q1     ●   ●   ·   ·   ·   ·   ·   ·
  q2     ●   ●   ●   ·   ·   ·   ·   ·
  q3     ●   ●   ●   ●   ·   ·   ·   ·
  q4     ●   ●   ●   ●   ●   ·   ·   ·
  q5     ●   ●   ●   ●   ●   ●   ·   ·
  q6     ●   ●   ●   ●   ●   ●   ●   ·
  q7     ●   ●   ●   ●   ●   ●   ●   ●
```

**第 2 层：再 AND 上 `sliding_window_overlay(3)`**（`kv_idx > q_idx - 3`，即向左 2 格）
→ 就是 `sliding_window_causal_mask_function(3)`：

```
        kv0 kv1 kv2 kv3 kv4 kv5 kv6 kv7
  q0     ●   ·   ·   ·   ·   ·   ·   ·
  q1     ●   ●   ·   ·   ·   ·   ·   ·
  q2     ●   ●   ●   ·   ·   ·   ·   ·
  q3     ·   ●   ●   ●   ·   ·   ·   ·     ← kv0 被窗口裁掉
  q4     ·   ·   ●   ●   ●   ·   ·   ·
  q5     ·   ·   ·   ●   ●   ●   ·   ·
  q6     ·   ·   ·   ·   ●   ●   ●   ·
  q7     ·   ·   ·   ·   ·   ●   ●   ●
```

（这张图与 `masking_utils.py:451-466` 的 docstring 示例完全一致，
也是测试 `tests/utils/test_masking_utils.py` 系列所验证的形状。）

**第 3 层：再 AND 上 `padding_mask_function`**（kv0、kv1 是左 padding）
→ 就是 `create_sliding_window_causal_mask(..., attention_mask=[[0,0,1,1,1,1,1,1]])` 的结果：

```
        kv0 kv1 kv2 kv3 kv4 kv5 kv6 kv7
  q0     ⊗   ⊗   ·   ·   ·   ·   ·   ·     ← ⚠ 整行无 ●（见下方注）
  q1     ⊗   ⊗   ·   ·   ·   ·   ·   ·
  q2     ⊗   ⊗   ●   ·   ·   ·   ·   ·
  q3     ⊗   ⊗   ●   ●   ·   ·   ·   ·
  q4     ⊗   ⊗   ●   ●   ●   ·   ·   ·
  q5     ⊗   ⊗   ·   ●   ●   ●   ·   ·     ← kv2 既被 padding 前的窗口裁掉、又被 padding 砍掉
  q6     ⊗   ⊗   ·   ·   ●   ●   ●   ·
  q7     ⊗   ⊗   ·   ·   ·   ●   ●   ●
```

**三层叠完，mask 的代数表达就是一行**：

```
sliding_window_causal_mask_function(3)
    = and_masks(sliding_window_overlay(3), causal_mask_function)     # :138
    ⊕ and_masks(·, padding_mask_function(pad))                        # :505 自动加上
```

> **注：左 padding 会产生「整行被屏蔽」**（上图的 q0、q1 行）。
> 这是真实的边界情况，也是旧路径 `_unmask_unattended` 存在的原因（§8.3）。
> 记住：**mask 只保证「哪些位置参与」，它不保证「每行至少有一个位置参与」**。

### 9.2 三条铁律

**铁律 1：`True` 表示「允许看」，`or_masks` 是加权限、`and_masks` 是收权限。**

**铁律 2：`create_*_mask` 里顺序固定 = 先 OR 后 AND**（§3.2）。
想「整体收权限」就必须把整个逻辑写进 `and_mask_function` 里自己组合。

**铁律 3：自定义函数必须是 index-based（依赖四个索引张量），且 torch≥2.6。**
依赖 `q_idx` / `kv_idx` 等参数（§2.3）；用了 `or/and_mask_function` 会强制 `use_vmap=True`（`:964`、`:970`）。

### 9.3 例 1：前 3 个 token 全局 + 其余滑动窗口

**目标**（prefix-LM 风格）：

- 前 3 个 token 之间**双向互看**（它们是一个「全局前缀」）；
- **所有** token 都能看到前 3 个 token（不管它们离多远）；
- 其余位置遵守滑动窗口 `W`；
- 整体保持 causal（后面的 token 不能提前透露给前面）。

**解法**：直接写一条「允许判据」，塞进 `and_mask_function`——
因为 `and` 槽位是唯一能「在 causal 基础上收权限」的位置，而 causal 基底已经保证了因果性。

```python
from collections.abc import Callable

import torch

from transformers.masking_utils import create_causal_mask


def global_prefix_or_sliding_mask_function(n_global: int, sliding_window: int) -> Callable:
    """前 n_global 个 token 参与全局注意力（含彼此双向）；其余位置只保留最近 sliding_window 个。

    注意：这个函数只描述「窗口」这一件事，因果性由 create_causal_mask 的 causal 基底负责，
    所以这里特意 *不* 写 `kv_idx <= q_idx` —— 那会跟基底重复，也会让「所有人看到前缀」失效。
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        # 1) 在前缀里 → 允许（前缀内部双向 + 所有人都能看到前缀）
        is_prefix = kv_idx < n_global
        # 2) 在滑动窗口内 → 允许（包含式：窗口总大小 = sliding_window）
        in_window = kv_idx > q_idx - sliding_window
        return is_prefix | in_window

    return inner_mask


# 组装：base = causal，and_mask_function = 我们的窗口判据
causal_mask = create_causal_mask(
    config=self.config,                     # config._attn_implementation 决定后端形态
    inputs_embeds=inputs_embeds,            # 只用来读 (B, q_length, dtype, device)
    attention_mask=attention_mask,          # 2D [B, S]，padding 会被自动 and 进来
    past_key_values=past_key_values,        # 提供 q_offset / kv_length（绝对位置来源）
    position_ids=position_ids,              # 只用于 packed 检测
    and_mask_function=global_prefix_or_sliding_mask_function(n_global=3, sliding_window=4096),
)
```

**逐格验证**（`n_global=3`、`W=3`、无 padding）：

```
        kv0 kv1 kv2 kv3 kv4 kv5 kv6 kv7
  q0     ●   ·   ·   ·   ·   ·   ·   ·     ← 前缀列全 ●（人人可见），但 q0 自己仍看不到未来
  q1     ●   ●   ·   ·   ·   ·   ·   ·
  q2     ●   ●   ●   ·   ·   ·   ·   ·
  q3     ●   ●   ●   ●   ·   ·   ·   ·
  q4     ●   ●   ●   ●   ●   ·   ·   ·
  q5     ●   ●   ●   ●   ●   ●   ·   ·     ← 前缀 3 列 + 窗口 3 列 = 6 个 ●
  q6     ●   ●   ●   ·   ●   ●   ●   ·     ← kv3 既非前缀、又已出窗口 → 屏蔽
  q7     ●   ●   ●   ·   ·   ●   ●   ●
```

> ⚠️ **这不完全是我们要的**：`q0` 行只有 `kv0`，`q1` 行只有 `kv0/kv1`——
> 前缀内部**没有**双向。原因在 §3.2：`and_mask_function` 是**与**上去的，
> `and(causal, our_fn)` 在 `kv > q` 处永远是 `False`。
> 「前缀内部双向」是一个**放松**，放松必须走 **OR 槽位**。
> 所以完整表达式要**同时用两个槽位**（这正是 §9.2 铁律 2 的实战演练）：

```python
# 完整版：前缀双向（OR 槽位） + 窗口/前缀可见（AND 槽位）
from transformers.masking_utils import and_masks, causal_mask_function, or_masks


def prefix_bidirectional_overlay(n_global: int) -> Callable:
    """前缀内部双向：只放松 (q, kv) 双落在前缀内的象限。"""

    def inner_mask(batch_idx, head_idx, q_idx, kv_idx):
        return (q_idx < n_global) & (kv_idx < n_global)

    return inner_mask


def prefix_or_window_restriction(n_global: int, sliding_window: int) -> Callable:
    """在前缀里、或在窗口内 → 允许。其余一律屏蔽。"""

    def inner_mask(batch_idx, head_idx, q_idx, kv_idx):
        return (kv_idx < n_global) | (kv_idx > q_idx - sliding_window)

    return inner_mask


causal_mask = create_causal_mask(
    config=self.config,
    inputs_embeds=inputs_embeds,
    attention_mask=attention_mask,
    past_key_values=past_key_values,
    position_ids=position_ids,
    # OR 槽位：把「前缀内双向」加进 causal   （:962）
    or_mask_function=prefix_bidirectional_overlay(n_global=3),
    # AND 槽位：整体收窄到「前缀 + 窗口」    （:968）
    and_mask_function=prefix_or_window_restriction(n_global=3, sliding_window=4096),
)
```

**最终 `and(or(causal, prefix_quadrant), prefix_or_window)` 的真实网格**：

```
        kv0 kv1 kv2 kv3 kv4 kv5 kv6 kv7
  q0     ●   ●   ●   ·   ·   ·   ·   ·     ← 前缀内部双向 ✔（OR 槽位救回来的）
  q1     ●   ●   ●   ·   ·   ·   ·   ·
  q2     ●   ●   ●   ·   ·   ·   ·   ·
  q3     ●   ●   ●   ●   ·   ·   ·   ·     ← 前缀列全 ●（所有人可见）✔
  q4     ●   ●   ●   ●   ●   ·   ·   ·
  q5     ●   ●   ●   ●   ●   ●   ·   ·     ← 前缀 3 列 + 窗口 3 列
  q6     ●   ●   ●   ·   ●   ●   ●   ·     ← kv3 出窗口且非前缀 → 被 AND 砍掉
  q7     ●   ●   ●   ·   ·   ●   ●   ●
```

**如果只想用 `create_sliding_window_causal_mask`**（base 已经含窗口），
那就只需要 OR 槽位（`:1204`），等价写法：

```python
from transformers.masking_utils import create_sliding_window_causal_mask

sliding_mask = create_sliding_window_causal_mask(
    config=self.config,
    inputs_embeds=inputs_embeds,
    attention_mask=attention_mask,
    past_key_values=past_key_values,
    position_ids=position_ids,
    # base = sliding_window_causal_mask_function(W)
    # 结果 = OR(base, or_fn)  → 窗口内 + 前缀全可见
    or_mask_function=lambda b, h, q, kv: (kv < 3) | ((q < 3) & (kv < 3)),
)
```

**两种写法的取舍**：

| | `create_causal_mask` + `and_mask_function` | `create_sliding_window_causal_mask` + `or_mask_function` |
|---|---|---|
| 表达力 | 更强（可以同时用 OR 和 AND 两个槽位） | 只开放 OR 槽位 |
| 语义清晰度 | 「base=causal，我来收紧」更直白 | 「base=sliding，我加例外」 |
| 与 `config.sliding_window` 的关系 | 需要自己把 `W` 传进函数 | **自动读 `config.sliding_window`**（`:1180-1182`） |

### 9.4 例 2：真实生产代码 —— Gemma3 的 blockwise

`models/gemma3/modular_gemma3.py:607-667` 是「组合代数」在库里的最佳范例，
值得逐行读。先是 block id 的构造（`:607-616`）：

```python
def get_block_sequence_ids_for_mask(token_type_ids: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
    # First find where a new image block starts: 1 if image and previous not image
    # The images cannot attend to future images, but can attend to all prev images and to itself bidirectionally
    is_image = (token_type_ids == 1).to(device=device)
    is_previous_image = nn.functional.pad(is_image, (1, 0), value=0)[:, :-1]
    new_image_start = is_image & ~is_previous_image
    group_ids = torch.cumsum(new_image_start.int(), dim=1) - 1
    block_sequence_ids = torch.where(is_image, group_ids, -1)   # 文本 token → -1
    return block_sequence_ids
```

然后是**一张 mask 的正负反例对照**（`:618-667`，注意 docstring 把两个槽位的语义直接写成了公式）：

```python
def create_masks_for_vision_model(
    config, inputs_embeds, attention_mask, past_key_values, position_ids, block_sequence_ids,
) -> dict:
    """Create full_attention and sliding_attention masks with correct composition.

    For global (full attention) layers:  OR(causal, blockwise)
    For local (sliding window) layers:  AND(sliding_window, OR(causal, blockwise))
    """
    mask_kwargs = { ... }

    # Full attention: OR(causal, blockwise) — use block_sequence_ids directly
    full_mask = create_causal_mask(**mask_kwargs, block_sequence_ids=block_sequence_ids)

    # We need to manually pad the sequence IDs for the sliding mask
    # as it's passed as an `or_mask_function` which bypasses internal padding.
    early_exit, _, _, _, kv_length, _, kv_offset = _preprocess_mask_arguments(**mask_kwargs, layer_idx=0)
    if early_exit:
        padded_block_sequence_ids = block_sequence_ids
    else:
        padded_block_sequence_ids = maybe_pad_block_sequence_ids(
            block_sequence_ids, attention_mask, kv_length, kv_offset
        )

    # Sliding attention: AND(sliding_window, OR(causal, blockwise))
    # Pass blockwise as or_mask_function (applied as step 2 in create_causal_mask)
    # Pass sliding_window as and_mask_function (applied as step 3, after OR)
    sliding_mask = create_causal_mask(
        **mask_kwargs,
        or_mask_function=blockwise_overlay(padded_block_sequence_ids),
        and_mask_function=sliding_window_overlay(config.sliding_window),
    )

    return {"full_attention": full_mask, "sliding_attention": sliding_mask}
```

**这段代码里有三个直接可复用的知识点**：

1. **`block_sequence_ids=` 和 `or_mask_function=blockwise_overlay(...)` 是等价的**——
   前者是框架内置的便捷入口（`:976-979` 帮你做 `maybe_pad_block_sequence_ids` + `or_masks`），
   后者是手动写法。
2. **手动传 `or_mask_function` 会绕过自动 padding**（代码注释原话：
   > "it's passed as an `or_mask_function` which bypasses internal padding"）。
   所以必须自己调 `maybe_pad_block_sequence_ids`（`masking_utils.py:217-227`）。**这是最容易漏的一步。**
3. **两个槽位的语义被写成了公式**：
   `AND(sliding_window, OR(causal, blockwise))` ——
   正是 §3.2 里「先 OR 后 AND」的固定顺序，也正是注释里说的 "step 2 / step 3"。

**其他真实用例**（都在同一套 API 上）：

| 模型 | 用法 | 行号 |
|------|------|------|
| Gemma4 | `or_mask_function=blockwise_overlay(...)` + `and_mask_function=sliding_window_overlay(...)` | `models/gemma4/modeling_gemma4.py:2138-2139` |
| ESMFold2 | `and_mask_function=sliding_window_bidirectional_overlay(w // 2)` | `models/esmfold2/modeling_esmfold2.py:516` |
| ESM-C | `and_mask_function=packed_sequence_mask_function(sequence_id)` | `models/esmc/modeling_esmc.py:387` |
| Moonshine-streaming | `and_mask_function=sliding_window_mask_function(...)`（**模型自己定义的**非对称窗口） | `models/moonshine_streaming/modular_moonshine_streaming.py:329` |
| t5gemma2 | `and_mask_function=sliding_window_mask_function(W, is_causal=False)` | `models/t5gemma2/modular_t5gemma2.py:595` |
| Nemotron-ASR-streaming | `and_mask_function=chunked_limited_mask_function(...)` | `models/nemotron_asr_streaming/modular_nemotron_asr_streaming.py:885` |

模型自己写 mask function 的模板（`models/gemma4/modeling_gemma4.py:1896-1909`，非对称窗口）：

```python
def sliding_window_mask_function(sliding_window: tuple[int, int]) -> Callable:
    """
    This creates uni/bidirectional attention mask with sliding window.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        left_window_size, right_window_size = sliding_window

        dist = q_idx - kv_idx
        left_mask = (dist >= 0) & (dist < left_window_size)
        right_mask = (dist < 0) & (-dist < right_window_size)
        return left_mask | right_mask

    return inner_mask
```

**注意它把 `sliding_window` 作为元组闭包进去**——这就是 mask 是函数的好处：
窗口参数不需要出现在任何张量里。

### 9.5 自定义 mask 的注意事项清单

| 事项 | 依据 | 后果 |
|------|------|------|
| 必须 `torch>=2.6` | `:960-961`、`:966-967`、`:1070-1071`（每处都 raise） | 否则 `ValueError: Using or_mask_function or and_mask_function arguments require torch>=2.6` |
| 会强制 vmap 展开（更慢） | `:964`、`:970` `use_vmap = True` | 模型小时 mask 构建时间占比明显（文档明说 ~200M 参数级别最明显） |
| 必须 index-based | `:352-368` 的 NOTE + `bidirectional_mask_function` 的 NOTE | 用 vmap 路径能容忍更多，但 `use_vmap=False` 时非 index-based 会出错 |
| 与 ExecuTorch **不兼容** | `docs/source/en/attention_interface.md` 的 WARNING | 导出时换回内置 pattern |
| 用了就一定不能跳过 mask | `:963`、`:969` `allow_is_causal_skip = False` | 哪怕 pattern 恰好等价于 causal，也会物化一张 `[B,1,S,S]` |
| 4D mask 会被原样返回 | `:811-812` | 你可以在 `attention_mask` 里直接塞 4D（见下） |

**最后一条的兜底方案**（`docs/source/en/attention_interface.md` 的 "Pass a custom 4D attention mask"）：
任何 `create_*_mask` 都能表达不了的 pattern，都可以自己造 4D 张量直接当 `attention_mask` 传进去。
但记住**后端差异**（文档原话）：

> [!IMPORTANT]
> The accepted convention depends on the attention backend. `sdpa` takes a boolean or a float mask.
> `eager` adds the mask to the scores, so it takes a float mask only. `flash_attention_2` and `flex_attention`
> consume their own formats (a 2D padding mask and a [BlockMask]) and **don't accept a raw 4D mask**.

以及最经典的坑（同一份文档）：

> A common mistake is to reuse the `1`/`0` convention of a 2D padding mask in a float 4D mask.
> Because the mask is added to the scores, `0.0` keeps a position and `1.0` only adds a small bias.

```text
        wrong_mask                          correct_mask
   (1 attends, 0 masks)              (0 attends, -inf masks)
      k0 k1 k2 k3 k4                     k0   k1   k2   k3   k4
   q0  1  0  0  0  0                  q0  0  -inf -inf -inf -inf
   q1  1  1  0  0  0                  q1  0   0   -inf -inf -inf
   q2  1  1  1  0  0                  q2  0   0    0   -inf -inf
   q3  1  1  1  1  0                  q3  0   0    0    0   -inf
   q4  1  1  1  1  1                  q4  0   0    0    0    0
```

（浮点 4D mask 的正确构造：`torch.where(mask.bool(), 0.0, float("-inf"))`。）

---

## 一句话总结

> **mask 从「一张 `[B,1,S,S]` 张量」变成了「一个 `(b,h,q,kv)->bool` 的纯函数」。**
> 组合子只有两个：**`and_masks`（交集、收权限）、`or_masks`（并集、加权限）**，
> 起点常量分别是全 `True`（`:54`）和全 `False`（`:68`），用 `&` / `|` 而非 `and` / `or`。
> 所有 pattern 都由「基底（causal / bidirectional）+ 叠加物（sliding_window / chunked / blockwise）+ 数据规则（padding / packed）」拼成，
> 而 `create_*_mask` 里 **OR 槽位先于 AND 槽位**（`:962` → `:968`），这个顺序是硬约束。
>
> `create_causal_mask` **从不返回函数**：它按 `config._attn_implementation` 查
> `ALL_MASK_ATTENTION_FUNCTIONS`（`:943`）选形态——`eager` 物化成 float、`sdpa` 物化成 bool、
> `flash` 只要 2D seq_lens、`flex` 存 `BlockMask`（唯一保留函数形态的），
> 未注册的后端**直接返回 `None` 并静默丢弃所有约束**（`:819`）。
> 序列长度不改变形态，但能让 sdpa 路径**直接跳过建 mask**（`:268-276`，条件见 §6.3）。

---

## 验证清单

> 本篇所有结论都用 `grep -n` / `sed -n` 在
> `/Users/baai/Desktop/baai_proj/transformers` 上实测；行号即当前 `main`（`42ca97014c`）的行号。
> **环境说明**：本机没有安装 torch（`python -c "import torch"` → `ModuleNotFoundError`），
> 因此**无法做运行时验证**。下面的命令都是静态验证，逐条附上实际输出。
> 涉及「运行时会返回什么」的判断（如 `sdpa_mask` 返回 bool 的 `[B,1,Q,KV]`）是从源码路径推导的，
> 测试文件 `tests/utils/test_masking_utils.py` 提供了等价的断言作为佐证。

```bash
cd /Users/baai/Desktop/baai_proj/transformers

# ── 1. 组合子只有两个 ──────────────────────────────────────────
grep -n "^def and_masks\|^def or_masks" src/transformers/masking_utils.py
# → 48:def and_masks(*mask_functions: Callable) -> Callable:
# → 62:def or_masks(*mask_functions: Callable) -> Callable:

# ── 2. 组合子的全部调用点（src 内共 22 处，全在 masking_utils.py）──
grep -n "and_masks\|or_masks" src/transformers/masking_utils.py | wc -l
# → 22
#    其中定义 2 处（:48 :62），成品包装 3 处（:138 :158 :165），
#    内部使用 4 处（:505 :693 :974 :1452 等），create_*_mask 里 13 处

# ── 3. 起点常量（and=True / or=False）───────────────────────────
sed -n '54p;68p' src/transformers/masking_utils.py
# → 54:        result = q_idx.new_ones((), dtype=torch.bool)
# → 68:        result = q_idx.new_zeros((), dtype=torch.bool)

# ── 4. 原子 mask 函数的精确行号 ─────────────────────────────────
grep -n "return kv_idx <= q_idx\|return q_idx >= 0\|return kv_idx > q_idx - sliding_window\|return padding_mask\[batch_idx, kv_idx\]" src/transformers/masking_utils.py
# → 80:    return kv_idx <= q_idx            (causal_mask_function :76)
# → 89:    return q_idx >= 0                 (bidirectional_mask_function :83)
# → 99:        return kv_idx > q_idx - sliding_window   (sliding_window_overlay :92)
# → 177:        return padding_mask[batch_idx, kv_idx]  (padding_mask_function :168)

# ── 5. 成品包装确实是 and_masks 拼的 ────────────────────────────
grep -n "return and_masks(sliding_window_overlay\|return and_masks(chunked_overlay\|return and_masks(sliding_window_bidirectional_overlay" src/transformers/masking_utils.py
# → 138:    return and_masks(sliding_window_overlay(sliding_window), causal_mask_function)
# → 158:    return and_masks(sliding_window_bidirectional_overlay(sliding_window), bidirectional_mask_function)
# → 165:    return and_masks(chunked_overlay(chunk_size, left_padding), causal_mask_function)

# ── 6. 后端 → mask interface 的映射表（含 "eager"）──────────────
sed -n '714,721p' src/transformers/masking_utils.py
# → _global_mapping = { "sdpa": sdpa_mask, "eager": eager_mask,
#      "flash_attention_2": flash_attention_mask, "flash_attention_3": ...,
#      "flash_attention_4": ..., "flex_attention": flex_attention_mask, }

# ── 7. create_causal_mask 的分发点 ──────────────────────────────
grep -n "mask_interface = ALL_MASK_ATTENTION_FUNCTIONS" src/transformers/masking_utils.py
# → 943 (causal) / 1059 (bidirectional) / 1186 (sliding) / 1306 (bidir sliding) / 1422 (chunked)

# ── 8. OR 槽位先于 AND 槽位（顺序硬约束）───────────────────────
sed -n '959,970p' src/transformers/masking_utils.py
# → :962  mask_factory_function = or_masks(mask_factory_function, or_mask_function)
# → :968  mask_factory_function = and_masks(mask_factory_function, and_mask_function)

# ── 9. 未注册后端 → 静默返回 None ───────────────────────────────
sed -n '819,820p' src/transformers/masking_utils.py
# → if config._attn_implementation not in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping:
# →     return True, None, None, None, None, None, None      ← early_exit=True

# ── 10. 4D mask 直通 ───────────────────────────────────────────
sed -n '811,812p' src/transformers/masking_utils.py
# → if isinstance(attention_mask, (torch.Tensor, BlockMask)) and len(attention_mask.shape) == 4:
# →     return True, attention_mask, None, None, None, None, None

# ── 11. 序列长度决定「跳过」的两个分支 ─────────────────────────
sed -n '262,263p;268,269p' src/transformers/masking_utils.py
# → 262:    if local_attention_size is not None and kv_length >= local_attention_size:
# → 263:        return False
# → 268:    if (q_length == 1 or kv_length == q_length) and (padding_mask is None or fast_all(padding_mask)):
# → 269:        return True

# ── 12. eager 永不跳过 causal mask ────────────────────────────
sed -n '584p' src/transformers/masking_utils.py
# → _ = kwargs.pop("allow_is_causal_skip", None)      ← 直接丢弃

# ── 13. eager 的 bool → float 转换 ────────────────────────────
sed -n '600,603p' src/transformers/masking_utils.py
# → if mask is not None:
# →     min_dtype = torch.finfo(dtype).min
# →     mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)

# ── 14. flex 是唯一把函数交给 torch 的路径 ─────────────────────
sed -n '699,707p' src/transformers/masking_utils.py
# → block_mask = create_block_mask(mask_mod=mask_function, B=batch_size, H=None,
#                    Q_LEN=q_length, KV_LEN=kv_length, device=device, _compile=...)

# ── 15. flash 只要 2D mask，或用内核参数代替 mask ──────────────
sed -n '638,647p' src/transformers/masking_utils.py
# → if attention_mask is not None:
# →     attention_mask = attention_mask[:, -kv_length:]
# →     if attention_mask.shape[1] == kv_length and attention_mask.all():
# →         attention_mask = None
grep -n 'flash_kwargs\["window_size"\] = (sliding_window - 1, sliding_window - 1)' src/transformers/modeling_flash_attention_utils.py
# → 655:        flash_kwargs["window_size"] = (sliding_window - 1, sliding_window - 1)

# ── 16. eager 消费 mask：加性 ─────────────────────────────────
sed -n '204,206p' src/transformers/models/llama/modeling_llama.py
# → attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
# → if attention_mask is not None:
# →     attn_weights = attn_weights + attention_mask

# ── 17. sdpa 消费 mask：attn_mask= 直传 bool 或 float ─────────
sed -n '158,167p' src/transformers/integrations/sdpa_attention.py
# → attn_output = torch.nn.functional.scaled_dot_product_attention(
# →     query, key, value, attn_mask=attention_mask, dropout_p=dropout,
# →     scale=scaling, is_causal=is_causal, **sdpa_kwargs)

# ── 18. flex 消费 mask：BlockMask 走 block_mask= ──────────────
sed -n '280,285p' src/transformers/integrations/flex_attention.py
# → block_mask = None ; score_mask = None
# → if isinstance(attention_mask, BlockMask): block_mask = attention_mask
# → else: score_mask = attention_mask

# ── 19. position_ids 在 masking_utils 里只有 packed 检测一处用 ──
grep -n "position_ids" src/transformers/masking_utils.py | grep -v "torch.Tensor | None\|A 2D tensor\|(`torch.Tensor`\|position_ids (`"
# → 749:    first_dummy_value = position_ids[:, :1] - 1     ← find_packed_sequence_indices 内部
# → 750:    position_diff = torch.diff(position_ids, prepend=first_dummy_value, dim=-1)
# → 851:    # We check the position_ids for potential packed sequence format (only if ...)
# → 854:    if position_ids is not None and attention_mask is None and past_key_values is None:
# → 857:        if batch_size != position_ids.shape[0]:
# → 858:            position_ids = position_ids.expand(batch_size, -1)
# → 859:        packed_sequence_mask = find_packed_sequence_indices(position_ids)
# → 936 / 1175 / 1397: _preprocess_mask_arguments(..., position_ids, layer_idx)  ← 透传
# → 1570:        "position_ids": position_ids,                     ← create_masks_for_generate 透传
# 即真正「使用」的只有 :854-859（加 :749-750 的辅助计算）；
# 其余全是签名 / docstring / 透传。create_bidirectional_mask(:1047) 与
# create_bidirectional_sliding_window_mask(:1290) 更是直接传 None

# ── 20. 绝对位置来自 cache 的 offset + arange ───────────────────
sed -n '507,510p;828,833p' src/transformers/masking_utils.py
# → 509:    q_arange = torch.arange(q_length, device=device) + q_offset
# → 510:    kv_arange = torch.arange(kv_length, device=device) + kv_offset
# → 829:        q_offset = past_key_values.get_query_offset(layer_idx)
# → 833:        kv_length, kv_offset = past_key_values.get_mask_sizes(q_length, layer_idx)

# ── 21. 遗留文件的引用（全仓库仅 4 处，且无 src 内 import）─────
grep -rn "modeling_attn_mask_utils" . 2>/dev/null | grep -v "^./.git/"
# → .github/workflows/push-important-models.yml:108:  'modeling_attn_mask_utils.py',   (CI 监视列表)
# → reading_docs/00_roadmap.md:77                                                  (路线图)
# → src/transformers/modeling_attn_mask_utils.py:30                                (自身 DEPRECATION_MESSAGE)
# → src/transformers/modeling_attn_mask_utils.py:48                                (自身 docstring 示例)
sed -n '14,18p;29,32p' src/transformers/modeling_attn_mask_utils.py
# → """IMPORTANT NOTICE: Every class and function in this file is deprecated in favor of using
#       the much more general `masking_utils.py` primitives. ... will be removed in the future."""
# → DEPRECATION_MESSAGE = "... is deprecated and will be removed in Transformers v5.10."

# ── 21b. 遗留文件各方法/函数的确切行号（§8.3 表格的来源）──────
grep -n "def to_causal_4d\|def to_4d\|def _make_causal_mask\|def _expand_mask\|def _unmask_unattended\|def _ignore_causal_mask_sdpa" src/transformers/modeling_attn_mask_utils.py
# → 81:    def to_causal_4d(          114:    def to_4d(
# → 162:    def _make_causal_mask(     198:    def _expand_mask(
# → 214:    def _unmask_unattended(    265:    def _ignore_causal_mask_sdpa(

# ── 22. 遗留文件的 `_unmask_unattended` 没有迁移 ───────────────
grep -rn "unmask" src/transformers/masking_utils.py
# → （无输出）

# ── 23. 版本（决定「v5.10 删除承诺已过期」这句）────────────────
grep -n "__version__" src/transformers/__init__.py | head -1
# → 21:__version__ = "5.16.0.dev0"
git log -1 --format="%h %ad %s" $(git log --diff-filter=A --format=%H -1 -- src/transformers/masking_utils.py)
# → 163138a911 Thu May 22 11:38:26 2025 +0200 🚨🚨[core] Completely rewrite the masking logic for all attentions (#37866)

# ── 24. Gemma3 的 blockwise 生产用例（§9.4 的代码来源）────────
sed -n '639,662p' src/transformers/models/gemma3/modular_gemma3.py
# → 640:    full_mask = create_causal_mask(**mask_kwargs, block_sequence_ids=block_sequence_ids)
# → 658:    sliding_mask = create_causal_mask(**mask_kwargs,
# → 660:        or_mask_function=blockwise_overlay(padded_block_sequence_ids),
# → 661:        and_mask_function=sliding_window_overlay(config.sliding_window))
# （get_block_sequence_ids_for_mask 在 :607；create_masks_for_vision_model 在 :618-667）

# ── 25. Gemma2 是「逐层交替」而不是「逐 token 全局」────────────
sed -n '95,98p' src/transformers/models/gemma2/configuration_gemma2.py
# → if self.layer_types is None:
# →     self.layer_types = ["sliding_attention" if bool((i + 1) % 2) else "full_attention"
#                           for i in range(self.num_hidden_layers)]

# ── 26. 库测试提供的等价断言（无法运行时验证时的佐证）──────────
sed -n '77,96p' tests/utils/test_masking_utils.py
# → test_packed_sequence_mask_sdpa：create_causal_mask(position_ids=packed) 得到 EXPECTED_PACKED_MASK
sed -n '152,183p' tests/utils/test_masking_utils.py
# → test_nonpacked_sequence_mask_skip：sdpa + 非 packed → causal_mask is None
#    （torch.compile 后不可跳过 → triu 形状的 bool mask）
```

**未能验证的部分（诚实声明）**：本机无 torch，所以
「`sdpa_mask` 实际返回 `[B,1,Q,KV]` bool 张量」「`eager_mask` 实际返回 float 张量」
「`flex_attention_mask` 返回 `BlockMask`」这类**运行时行为**只做了源码路径推导，
没有实跑。用于佐证的是 `tests/utils/test_masking_utils.py:77-183` 的断言
（`causal_mask == EXPECTED_PACKED_MASK`、`causal_mask is None` 等），
它们与推导一致。若要在自己环境复核，`RUN_SLOW=0 pytest tests/utils/test_masking_utils.py -q` 即可。

---

## 导航

- 路线图：[00_roadmap.md](./00_roadmap.md)（本篇 = 进阶 · 架构 · 2A2）
- 上一篇：[2A1_attention_backends.md](./2A1_attention_backends.md) —— 「算什么」：`ALL_ATTENTION_FUNCTIONS` 如何分发 eager / sdpa / flash / flex
- 下一篇：[2A3_kv_cache.md](./2A3_kv_cache.md) —— 「缓存什么」：`CacheLayerMixin` 与 `DynamicCache` / `StaticCache`
- 回顾：[1M2_mistral_sliding_window.md](./1M2_mistral_sliding_window.md) —— 用本篇的 `and_masks` 视角重看 Mistral 那 5 行差异
- 延伸：[2A5_modular_code_generation.md](./2A5_modular_code_generation.md) —— §9.4 引用的 `modular_gemma3.py` 该怎么读
- 建议紧接着读：`src/transformers/cache_utils.py` 的
  `get_mask_sizes`（`:1562` / `:2077` / `:2108` / `:2127`）与 `get_query_offset`（`:1592` / `:2103` / `:2122`）——
  §5.2 说「绝对位置由 cache 回答」，这两组方法就是答案，正好是 2A3 的入口。
