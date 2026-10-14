# KV Cache 体系精读：层（Layer）与容器（Cache）的双层结构

> 阅读目标：搞清楚 `src/transformers/cache_utils.py` 里那 23 个类的**体系结构**——
> 为什么要有「层」和「容器」两层抽象、每种子类解决什么问题、`update()` 的调用契约是什么。
>
> **这是「注意力三件套」的第三篇**：
> [2A1](./2A1_attention_backends.md) 讲**算什么**（后端），[2A2](./2A2_masking_system.md) 讲**在哪算**（mask），
> 本篇讲**缓存什么**。三篇合起来才是完整的注意力机制。
>
> 一句话定位：**2A2 讲的是「哪些位置不许看」，本篇讲的是「哪些 K/V 不用重算」。**
> 两者在代码里通过 `past_key_values` 这一个参数耦合——这耦合点就是本篇第 7 节。
>
> 源文件：`src/transformers/cache_utils.py`（**2131 行，23 个类**）、
> `src/transformers/generation/continuous_batching/`（13 个文件，6083 行）
> 人类向文档：`docs/source/en/kv_cache.md`、`docs/source/en/cache_explanation.md`、`docs/source/en/paged_attention.md`

---

## 速读区（TL;DR）

**一句话**：`cache_utils.py` 是**两层结构**——`CacheLayerMixin` 管「单层怎么存、怎么裁」，
`Cache` 管「所有层怎么组织、按 `layer_idx` 分发」。选哪个 Layer 由 config 的 `layer_types`
字符串经 `DYNAMIC/STATIC_LAYER_TYPE_MAPPING` 决定；位置信息由 cache **反向报给**外面。

**必须记住的断言**

| # | 断言 | 出处（文件:行号） |
|---|------|------------------|
| 1 | 23 个类分成两支：`CacheLayerMixin`（`:27`）与 `Cache`（`:1269`） | `cache_utils.py` |
| 2 | `update()` 返回**完整**的 K/V（历史 + 本次），不是本次新算的；`StaticLayer` 返回的**就是 `self.keys` 本身** | `:487` |
| 3 | **SWA 的裁剪是 cache 自己做的**（既不是 attention、也不是 mask）：只留 `W-1`，但**返回未裁剪的 `full_key_states`** | `:249-261` |
| 4 | 滑动窗口层的 `get_seq_length()` 返回**真实总数** `cumulative_length`，不是 `keys.shape[-2]` | `:275-277` |
| 5 | `get_mask_sizes()` → `(kv_length, kv_offset)` 是 cache 与 mask 的**唯一接口** | `:60-61`、`masking_utils.py:833` |
| 6 | `cache_position` / `cache_kwargs` **在本版本已删除**，只剩 remote code 的兼容 shim | `generation/utils.py:623-633` |
| 7 | `layers` 和 `layer_class_to_replicate` **二选一**，都传或都不传会报错 | `:1296-1305`、`:1374-1376` |
| 8 | `sliding_attention` 与 `chunked_attention` 映射到**同一个** Layer 类（cache 只管存多少，不管能不能看） | `:1225`、`:1223-1244` |
| 9 | `StaticLayer` 为 compile 付三处代价：`cumulative_length` 必须是 tensor、`mark_static_address`、`get_mask_sizes` 恒返回 `max_cache_len` | `:414-415`、`:448-451`、`:489-493` |
| 10 | `EncoderDecoderCache` 是**组合**不是继承；`crop` **只裁 self**（cross 不能裁） | `:1947`、`:2000-2001`、`:2057-2063` |
| 11 | `MtpCache` / `DFlashCache` 只覆写 `get_query_offset` / `get_mask_sizes`——是投机解码的**偏移量适配器**，不是新的存储策略 | `:2102-2106`、`:2114-2131` |
| 12 | `PagedAttentionCache` **不是** `Cache` 的子类，`cache.py` 里**没有任何 `CacheLayerMixin`** | `continuous_batching/cache.py:89` |
| 13 | `QuantizedCache` **只支持全部层都是 full attention**，否则显式 `raise ValueError` | `:1934-1939` |
| 14 | `LinearAttentionCacheLayerMixin` **不是** `CacheLayerMixin` 的子类，**没有 `update` / `get_seq_length` / `get_mask_sizes`** | `:891`、`:915-927` |

**最容易踩的坑 / 反直觉**

- **1M2 的悬案要修正**：sliding window 的裁剪既不由 attention 实现处理、也不由 mask 兜底，而是 `DynamicSlidingWindowLayer` **自己裁到 `W-1`**，mask 靠 `kv_offset` 对齐绝对位置（`:249-261`、`:263-273`）。
- **为什么是 `W-1` 而不是 `W`**：缓存的 `W-1` 个是「历史」，本步新算的 1 个是「当前」，`W-1 + 1 = W` 刚好组成一个完整窗口。
- **「返回完整 K/V」不是形式主义**：SWA 若返回裁剪后的 `self.keys` 会丢掉最新位置；`StaticLayer` 形状必须静态；`QuantizedLayer` 要让量化对 attention 透明（`LinearAttention` 除外——它根本没有 `update`）。
- **不要把 `update()` 的返回值再存回 cache**：契约是原地写回；`StaticLayer` 的返回值甚至就是 `self.keys` 本身，再存一遍会把地址搞乱（`torch.compile` / cudagraph 依赖地址固定）。
- **顺序必须是「先 RoPE 再 cache」**：cache 里存的必须是**旋转后**的 K，否则每步取出来还得按历史位置重转一遍。
- **`layer_types` 决定了 mask 的分层**：mask 按「层**类型**」而不是「层号」建（同类型层的 `get_mask_sizes` 必然相同），映射表是 `LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING`（`masking_utils.py:1507-1520`）。
- **`q_offset` 默认就是 `get_seq_length()`**，所以普通 `DynamicLayer` 上 `q_offset == kv_offset`，「规整 causal 可跳过 mask」的优化才命中；`MtpCache` / `DFlashCache` 覆写它就是为了**打破这个相等**。
- **不要按老教程去找 `cache_position`**：位置现在由 cache 自己算完报出来（`get_seq_length` / `get_query_offset` / `get_mask_sizes`），`cache_utils.py` 里剩下的 `cache_position` 只是 `StaticLayer.update` 内部的局部下标变量。
- 滑动窗口 + 量化**目前不支持**。

**读完应该能回答**

- SWA 的裁剪到底是谁做的？为什么 `update()` 返回的反而不是裁剪后的 K/V？
- `get_seq_length()` / `keys.shape[-2]` / `get_mask_sizes()` 的 `kv_length` 这三个「长度」分别是什么？
- `cache_position` / `cache_kwargs` 被什么替代了？「位置由 cache 反向报给外面」的好处是什么？
- 「同一层类型共用一张 mask」为什么成立？`create_causal_mask` 在 `layer_idx` 为 `None` 时怎么挑代表层？
- `MtpCache` / `DFlashCache` 改的是什么？为什么连续批处理要另起一套 `PagedAttentionCache`？

---

## 前置知识

- [1M1_llama_structure.md](./1M1_llama_structure.md)：`LlamaAttention.forward` 的第 3 步就是 `past_key_values.update`
  （`modeling_llama.py:262`）。本篇把它展开。
- [1M2_mistral_sliding_window.md](./1M2_mistral_sliding_window.md)：SWA。**那篇留了一个问题**——
  「sliding window 的裁剪实际上由 attention 实现处理，或者通过 mask 把超出窗口的位置置 `-inf`」。
  **这个说法在本篇第 4.2 节会被修正**：裁剪是 **cache 自己**做的。
- [2A5_modular_code_generation.md](./2A5_modular_code_generation.md)：本篇的写法对齐它。
- 1A5（`generate` 主循环，待写）：首步 / 后续步的区别。
- 2A1（注意力后端，待写）、2A2（mask 系统，待写）：本篇第 7、10 节会指出交叉点，但不展开。

> **版本提示（重要）**：本仓库是 transformers v5 开发版。`cache_position` 这个参数**已经被删除**
> （只剩 `generation/utils.py:623-633` 给 remote code 的兼容 shim），`cache_kwargs` 也已不存在。
> 本篇第 6.3 节会讲清楚「没有它们之后，位置信息从哪来」。**不要按老教程的记忆去读这个文件。**

---

## 一、为什么需要 KV cache

### 1.1 一次 forward 的两种形态

呼应 1A5：`generate` 的循环里，**首步（prefill）** 一次送进整个 prompt，
**后续步（decode）** 每次只送 1 个新 token。

这两种形态在这个库里的差别，最终都体现在 `past_key_values` 上：

| | 首步 prefill | 后续步 decode |
|---|---|---|
| `input_ids` 长度 | `T`（prompt 长度） | `1` |
| 进入 attention 前 | `past_key_values` 为空（或已预填） | `past_key_values` 里已有 `T, T+1, ...` 个 token |
| mask 形状 | `[1, 1, T, T]` | `[1, 1, 1, T+1]` |
| Q 的长度 / K、V 的长度 | `T` / `T` | `1` / `T+1` |
| 计算量主项 | `2P·T` | `2P·1`（P = 参数量） |

### 1.2 没有 cache 时，第 t 步在算什么

自回归生成的本质是「每次只多一个 token」，但如果**每一步都把「历史 + 新 token」整个重跑一遍**：

- 第 1 步：对 1 个 token 算 QKV 投影 → 得到 `k₁ v₁`
- 第 2 步：对 2 个 token 算 QKV 投影 → 得到 `k₁ v₁`（**和上一步一模一样**）、`k₂ v₂`
- 第 3 步：对 3 个 token 算 → `k₁ v₁`（**第三次**）、`k₂ v₂`（**第二次**）、`k₃ v₃`
- …
- 第 t 步：对 t 个 token 算，其中前 `t-1` 个 token 的 K/V **已经算过 `t-1` 次了**

**为什么重算结果必然相同**：K/V 只依赖该位置的 hidden state 和**绝对位置**（RoPE 旋转角只由
`position_ids` 决定，见 [2A4](./2A4_rope_abstraction.md)）。历史 token 的 hidden state 和位置都不会变，
所以重算出来的 K/V 是**逐比特相同**的——纯浪费。

**用真实数字量化**（Llama-3-8B，8.03B 参数）：

| | 公式 | 生成 1000 个 token |
|---|---|---|
| 有 cache | `2P · N` | `2 × 8.03e9 × 1000` = **16.06 TFLOP** |
| 无 cache | `2P · N(N+1)/2` | `2 × 8.03e9 × 500500` = **8038 TFLOP** |
| 比值 | `≈ N/2` | **500.5 ×** |

> 生成 1000 token 时，无 cache 要多算 **500 倍**的线性层 FLOPs。这就是 KV cache 存在的全部理由。

**但要说清一件事：cache 消不掉的部分**。

缓存只是让「线性层（QKV 投影 / o_proj / MLP）每步只处理 1 个 token」。
注意力本身 `QK^T` 仍然要拿这 1 个 query 去和缓存里全部 `t` 个 key 做点积——
这部分是 `O(t·d)`，**cache 一点也帮不上忙**。

这正是「长上下文」难题的根源：序列长度上去以后，
**KV cache 的显存是 `O(t)`，注意力计算是 `O(t²)`**。
后面所有花活（SWA / DSA / MLA / 量化 / 分页）都是在砍这两项中的一个：

| 手段 | 砍什么 | 在哪篇 |
|---|---|---|
| 滑动窗口（SWA） | cache 的 `t` → 常数 `W` | [1M2](./1M2_mistral_sliding_window.md)、本篇 4.2 |
| 稀疏 top-k（DSA） | 每步实际参与点积的 key 数 | 2M7、本篇 4.3 |
| MLA（低秩压缩） | 每 token 的 cache 宽度 | 2M2 |
| KV 量化 | 每元素的字节数 | 本篇 4.6 |
| Paged / 连续批处理 | 不是「砍」，是「别浪费」 | 本篇第 10 节 |

---

## 二、双层架构：Layer 管单层，Cache 管所有层

这是本篇最重要的结构。打开 `cache_utils.py` 你会看到 23 个类，
它们**分成两支，各管一层抽象**：

```
┌──────────────────────────────────────────────────────────────────────────┐
│  第一层：Layer —— 「单层」的 K/V 存储                                     │
│                                                                          │
│  cache_utils.py:27   CacheLayerMixin (ABC)                              │
│                      负责：keys / values 两个张量，怎么长、怎么裁         │
│                      典型接口：update() / get_seq_length() /              │
│                                get_max_length() / crop()                  │
├──────────────────────────────────────────────────────────────────────────┤
│  第二层：Cache —— 「所有层」的集合 + 层间逻辑                             │
│                                                                          │
│  cache_utils.py:1269 Cache                                              │
│                      负责：self.layers 列表、按 layer_idx 分发、          │
│                            跨层统一的 get_mask_sizes() / get_seq_length() │
│                            层的懒创建、offloading 调度、beam search 重排   │
└──────────────────────────────────────────────────────────────────────────┘
```

**为什么要分两层？** 三个理由，都在代码里有痕迹：

1. **层与层的策略可以不同。** 混合模型（Gemma 2/3、Llama 4、GLM 的
   「sliding + full 交替」）里，不同层要用不同的裁法。
   `Cache.__init__` 拿到的是一张 `layer_types` 列表，**逐层选类**（`cache_utils.py:1793`）：
   `layers = [DYNAMIC_LAYER_TYPE_MAPPING[layer_type](**layer_kwargs) for layer_type in layer_types]`
2. **容器统一对外。** 模型侧只调 `past_key_values.update(k, v, self.layer_idx)`，
   完全不知道第 5 层是动态的还是静态的。这是[路线图](./00_roadmap.md)里说的「分发与组合」的标准套路，
   和 [2A1](./2A1_attention_backends.md) 的 `ALL_ATTENTION_FUNCTIONS` 是同一个设计思想。
3. **容器还管「不属于任何一层」的事**：offloading（把层数据搬去 CPU）、
   `is_compileable`（所有层都静态才可编译）、`batch_size` 一致性校验。

### 2.1 真实继承树（Layer 一支）

```
CacheLayerMixin (ABC)                                              cache_utils.py:27
  │  抽象方法：lazy_initialization / update / get_mask_sizes /
  │            get_seq_length / get_max_length
  │  类属性：is_compileable=False(:30) supports_early_init=True(:31) _layer_type(:34)
  │
  ├─ DynamicLayer                                                  :113   is_sliding=False(:119)
  │    「cat 生长」，最常用。keys/values 用 torch.cat 追加（:144-145）
  │    │
  │    ├─ DynamicSlidingWindowLayer                                :203   is_sliding=True(:209)
  │    │    只保留最近 sliding_window-1 个历史（:252-254）
  │    │
  │    ├─ DynamicIndexedLayer                                      :319
  │    │    多一份 indexer_keys，给 DSA 稀疏注意力用（:330、:338）
  │    │
  │    └─ QuantizedLayer (ABC)                                     :703
  │         抽象 _quantize(:768) / _dequantize(:771)；residual_length 机制
  │         │
  │         ├─ QuantoQuantizedLayer                                :779   optimum-quanto 后端
  │         └─ HQQQuantizedLayer                                   :834   HQQ 后端
  │
  └─ StaticLayer                                                   :398   is_compileable=True(:408)
       「预分配 + 原地写」，为 torch.compile / CUDA graph 而生
       │
       ├─ StaticSlidingWindowLayer                                 :504   is_sliding=True(:517)
       │    max_cache_len = min(sliding_window, max_cache_len)（:520）
       │
       └─ StaticIndexedLayer                                       :631
            静态版 indexer_keys（:643、:664）

LinearAttentionCacheLayerMixin (ABC)                               :891
  │  is_compileable=True(:895)  supports_early_init=False(:897)
  │  抽象方法：lazy_initialization / update_conv_state / update_recurrent_state
  │  完全不做空 update()，因为它根本没有 seq_len 维度
  │
  └─ LinearAttentionLayer                                          :1003
       conv_states / recurrent_states 两个 dict（:902-903）
       │
       ├─ LinearAttentionAndFullAttentionLayer                     :1094
       ├─ LinearAttentionAndSlidingWindowAttentionLayer            :1134
       ├─ LinearAttentionAndStaticFullAttentionLayer               :1166
       └─ LinearAttentionAndStaticSlidingWindowAttentionLayer      :1198
```

> **注意这里有个「菱形继承」的例外**：最后 4 个类用 **多重继承**
> （`LinearAttentionLayer` + 对应的注意力层），因为它们要**同时**维护两种状态：
> 线性注意力层（Mamba/SSM）的递归状态 + 普通注意力层的 K/V。
> 它们手写了 `lazy_initialization`（:1102、:1142、:1171、:1203），
> 靠「位置参数个数」来判定这次调用是给哪一半用的：
> `if len(args) == 2 and len(kwargs) == 0: DynamicLayer.lazy_initialization(self, *args)`。

### 2.2 真实继承树（Cache 一支）

```
Cache                                                              cache_utils.py:1269
  │  self.layers: list[CacheLayerMixin | LinearAttentionCacheLayerMixin]
  │  self.layer_class_to_replicate（懒创建用，:1307）
  │  self.offloading / self.prefetch_stream（:1309-1311）
  │
  ├─ DynamicCache                                                  :1737
  │    默认实现。传 config → 逐层按 layer_types 建 Layer（:1793）
  │    不传 config → 懒创建，第一次 update 时按 layer_idx 补（Cache.update :1374-1376）
  │    │
  │    ├─ MtpCache                                                :2102
  │    └─ DFlashCache                                             :2114
  │
  ├─ StaticCache                                                   :1829
  │    要求 config + max_cache_len；全部层预分配
  │    （别名 SlidingWindowCache = StaticCache，:2099，v5 移除的旧名）
  │
  ├─ QuantizedCache                                                :1884
  │    只支持「全部层都是 full_attention」的模型（:1934-1939 会显式报错）
  │
  └─ EncoderDecoderCache                                            :1947
       组合而非继承：内部持有 self_attention_cache + cross_attention_cache 两个 Cache
```

**为什么 `EncoderDecoderCache` 是「组合」而不是「第四种 Layer」**：
encoder-decoder 模型（T5、Whisper，见 2M4）的解码层有**两套** attention——
self-attention（有 cache、按位置增长）和 cross-attention（对 encoder 输出，**不增长**）。
两者增长规律完全不同，硬塞进一个 Layer 列表会互相污染，所以用两个独立 Cache 装。

---

## 三、`CacheLayerMixin`：单层的抽象契约

`cache_utils.py:27`。它定了 5 个**必须实现**的抽象方法 + 一组有默认实现的辅助方法。

### 3.1 必须实现的 5 个

| 抽象方法 | 行号 | 签名 | 契约 |
|---|---|---|---|
| `lazy_initialization` | :52-53 | `(key_states, value_states) -> None` | 「我不知道 dtype / device / num_heads 是多少，你第一次 update 时告诉我」。**这就是为什么 Layer 创建时不分配显存** |
| `update` | :55-58 | `(key_states, value_states, *args, **kwargs) -> (Tensor, Tensor)` | 写入 + **返回给 attention 用的完整 K/V**。契约细节见第 6 节 |
| `get_mask_sizes` | :60-61 | `(query_length: int) -> (kv_length, kv_offset)` | 给 mask 用的「这批 K/V 有多长、起点在哪」。**这是 cache 和 mask 的唯一接口** |
| `get_seq_length` | :63-64 | `() -> int` | 已经见过多少 token。**注意对滑动窗口层它返回的是「真实总数」而不是「当前存了几个」**（:277） |
| `get_max_length` | :66-73 | `() -> int` | 上限。**`-1` 表示「无上限或没有序列维度」**（动态层、线性注意力层都返回 `-1`） |

> `get_max_cache_shape` 改名了：`cache_utils.py:106-110` 是兼容 shim，
> 会打 deprecation 警告（"will be removed in version 5.16"），内容就是 `return self.get_max_length()`。
> **读源码时看到 `get_max_cache_shape` 一律翻译成 `get_max_length`。**

### 3.2 有默认实现的辅助方法

| 方法 | 行号 | 作用 |
|---|---|---|
| `offload` / `prefetch` | :75 / :81 | 层数据搬到 CPU / 搬回 device（**KV cache offloading**，`kv_cache.md` 的 "Cache offloading" 一节） |
| `reset` | :87 | 原地清零，保留对象（`crop(全部)` 的廉价替代）。注意它会顺手重置 `cumulative_length`（:92-98） |
| `reorder_cache` | :100 | beam search 换 beam 时按 `beam_idx` 重排 batch 维 |
| `get_max_cache_shape` | :106 | 已废弃，见上 |

**两个类属性值得单独记**（它们决定了容器能不能 `torch.compile`）：

```python
is_compileable = False        # :30   → DynamicLayer 不改，StaticLayer 改成 True（:408）
supports_early_init = True    # :31   → LinearAttentionCacheLayerMixin 改成 False（:897）
```

`Cache.is_compileable`（:1651-1657）就是 `all(layer.is_compileable for layer in self.layers)`。
`supports_early_init=False` 的意义见 6.4 节。

### 3.3 `_layer_type`：让模型文件能注册自己的 Layer

`cache_utils.py:34`，配套 `__init_subclass__`（:36-42）：

```python
def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    if cls._layer_type is not None:
        if issubclass(cls, StaticLayer):
            STATIC_LAYER_TYPE_MAPPING[cls._layer_type] = cls
        else:
            DYNAMIC_LAYER_TYPE_MAPPING[cls._layer_type] = cls
```

**这是一个巧妙的解耦**：`cache_utils.py` 不需要 import 任何模型文件，
模型文件用 `_layer_type = "..."` 就能把自己注册进映射表，**在它被 import 的时候自动生效**。

真实用例（实测）：

| 模型 | 类所在文件:行号 | `_layer_type` |
|---|---|---|
| DeepSeek V4（激进压缩注意力） | `models/deepseek_v4/modeling_deepseek_v4.py:185` | `"heavily_compressed_attention"` |
| DeepSeek V4（压缩稀疏注意力） | `models/deepseek_v4/modeling_deepseek_v4.py:265` | `"compressed_sparse_attention"` |
| MiniMax M3（多模态稀疏） | `models/minimax_m3_vl/modeling_minimax_m3_vl.py:54`、`:94` | `"minimax_m3_sparse"` |

---

## 四、Layer 子类逐个拆

### 4.1 `DynamicLayer`（:113）—— 最常用，也是默认

```python
# cache_utils.py:144-146
self.keys = torch.cat([self.keys, key_states], dim=-2)
self.values = torch.cat([self.values, value_states], dim=-2)
return self.keys, self.values
```

- 张量形状 `[batch_size, num_heads, seq_len, head_dim]`，`seq_len` 随生成**一直长**。
- `get_max_length()` 返回 `-1`（无上限，:160-162）。
- `get_mask_sizes()` 返回 `(self.get_seq_length() + query_length, 0)`（:148-152）——
  **kv_offset 恒为 0**，因为动态层从不丢弃历史。
- 代价：每步 `torch.cat` 都会**新建一块更大的张量**并拷贝。张量地址在变，
  所以 `is_compileable = False`（继承基类的默认值）。

**适用**：短序列、变长序列、需要简单可靠的场景。`generate` 不指定 `cache_implementation` 时就是它。

### 4.2 `DynamicSlidingWindowLayer`（:203）—— 回答 1M2 的悬案

[1M2](./1M2_mistral_sliding_window.md) 说 SWA 的裁剪由 attention 实现或 mask 兜底。
**实测结论：两者都不对。裁剪是 cache 层自己做的。**

看 `update` 的核心三行（`cache_utils.py:249-261`）：

```python
# 先把「已有的 + 新的」拼起来
full_key_states = torch.cat([self.keys, key_states], dim=-2)
full_value_states = torch.cat([self.values, value_states], dim=-2)
# 只缓存最后 sliding_window - 1 个 token
if not self.record_past:
    self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
    self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
...
# 返回的是「完整的」，不是裁剪后的！
return full_key_states, full_value_states
```

**为什么是 `sliding_window - 1` 而不是 `sliding_window`？**
因为 mask 的定义是 `kv_idx > q_idx - W`（Mistral 的窗口含当前位置，共 W 个位置）。
缓存的 W-1 个是「历史」，本步新算的 1 个是「当前」——**W-1 + 1 = W**，刚好组成一个完整窗口。

**为什么返回的是完整的、而不是裁剪后的？**
因为 `full` 里已经包含了本次需要参与注意力的「W-1 个历史 + 本次的 query 对应的 K/V」。
如果返回裁剪后的 `self.keys`，反而会丢掉最近的位置。这里和 `StaticSlidingWindowLayer` 的注释
（:601-602「we should return the whole states instead of `self.keys/values` here, as otherwise we lose some context」）
说的是同一件事。

**那 mask 怎么知道这 W 个 K/V 对应哪些绝对位置？** 靠 `get_mask_sizes`（:263-273）：

```python
is_full = self.cumulative_length >= self.sliding_window
kv_offset = max(self.cumulative_length - self.sliding_window + 1, 0)
if is_full:
    kv_length = self.sliding_window - 1 + query_length
else:
    kv_length = self.cumulative_length + query_length
return kv_length, kv_offset
```

**逐步骤推演（W = 4）**：

| 步骤 | 更新前 `self.keys` | 返回给 attention 的 K/V | 更新后 `self.keys` | `cumulative_length` | `get_mask_sizes(1)` |
|---|---|---|---|---|---|
| 1（prefill 4 个 token） | `[]` | `[k1 k2 k3 k4]`（长 4） | `[k2 k3 k4]`（长 3） | 4 | `(4, 1)` |
| 2（decode） | `[k2 k3 k4]` | `[k2 k3 k4 k5]`（长 4） | `[k3 k4 k5]` | 5 | `(4, 2)` |
| 3（decode） | `[k3 k4 k5]` | `[k3 k4 k5 k6]`（长 4） | `[k4 k5 k6]` | 6 | `(4, 3)` |

**三条结论**：
1. **内部存储恒定** = `W - 1` 个 token。不管生成到第 100 步还是第 100000 步，显存都不涨。
2. **返回的 K/V 恒定** = `W` 个（填满后）。所以 attention 端的开销也是常数。
3. **`get_seq_length()` 返回 `cumulative_length`（真实总数，:275-277），不是 `keys.shape[-2]`**。
   这一点非常关键——`LlamaModel.forward` 用它算 `position_ids`（`modeling_llama.py:387`），
   位置必须是绝对的，不能因为 cache 裁过就打回 0。

**Mistral 就是这样被接上的**（不需要模型侧写任何代码）：

```
configuration_mistral.py:80          sliding_window = 4096
  ↓
modeling_mistral.py:357              past_key_values = DynamicCache(config=self.config)
  ↓
（mistral 没有显式 layer_types）
cache_utils.py:1709-1710             layer_types = ["sliding_attention"] * num_hidden_layers
  ↓
cache_utils.py:1226                  DYNAMIC_LAYER_TYPE_MAPPING["sliding_attention"] = DynamicSlidingWindowLayer
  ↓
cache_utils.py:1793                  layers = [DynamicSlidingWindowLayer(sliding_window=4096) for _ in ...]
```

**所以 Mistral 7B 的 KV cache 上限是常数 `32 × 2 × 8 × 4095 × 128 × 2B ≈ 512 MiB`，
与生成长度完全无关**——1M2 那个「cache 大小变为常数」的直觉，落地路线是这一条。
（`record_past` 开关是给 assistant / speculative decoding 用的：
需要「回滚」时先全存着，等 `crop` 再裁，见 :218-223 和 :283-316。）

### 4.3 `DynamicIndexedLayer`（:319）—— **这就是 2M7 GLM 的 DSA 需要的东西**

动态稀疏注意力（DSA / DeepSeek Sparse Attention）在标准 K/V 之外，
还要维护一份**单头、低维的 indexer key**，用来给每个 query 打分挑 top-k。

```python
# cache_utils.py:319-326 docstring 原文
"""A cache layer that extends `DynamicLayer` with an extra indexer key cache for Dynamic Sparse Attention (DSA)
models (e.g. GLM MoE DSA, DeepSeek V32).

The main K/V cache stores tensors of shape `[batch_size, num_heads, seq_len, head_dim]` (inherited).
The indexer key cache stores a tensor of shape `[batch_size, seq_len, index_head_dim]` (3D, single-head).
"""
```

| 成员 | 行号 | 说明 |
|---|---|---|
| `indexer_keys` | :330 | 单独一份张量，**不是 4D 而是 3D**（单头） |
| `lazy_initialization_indexer` | :333 | 独立的懒初始化 |
| `update_indexer` | :338-351 | 和 `update` 一样是 cat，但**沿 `dim=1` 而不是 `dim=-2`**（因为去掉了 heads 维） |
| `offload` / `prefetch` / `reset` / `reorder_cache` / `crop` | :353 / :358 / :363 / :368 / :373 | 全部 `super()` 之后再处理 `indexer_keys` |

**模型侧怎么调**（实测）：

```python
# models/glm_moe_dsa/modeling_glm_moe_dsa.py:237
k = past_key_values.update_indexer(k, self.layer_idx)
```

调用点：`glm_moe_dsa`（`modeling_glm_moe_dsa.py:237` / `modular_glm_moe_dsa.py:176`）、
`deepseek_v32`（`modeling_deepseek_v32.py:238`）、`glm5_next`（`modeling_glm5_next.py:811`）、
`qwen4_exp`（`modular_qwen4_exp.py:411`）。

**GLM-5.3 是怎么接上的**（`configuration_glm_moe_dsa.py:158-159`，实测）：

```python
if self.layer_types is None:
    self.layer_types = ["deepseek_sparse_attention"] * self.num_hidden_layers
```

→ `cache_utils.py:1236`：`DYNAMIC_LAYER_TYPE_MAPPING["deepseek_sparse_attention"] = DynamicIndexedLayer`。

> **读 2M7 时请带着这一条**：GLM 的 DSA 之所以能用上 `DynamicIndexedLayer`，
> 只是因为 config 里那行字符串 + 映射表里那一行。**模型代码里没有任何 cache 的存储逻辑**，
> 它只是调了一次 `past_key_values.update_indexer(...)`。

### 4.4 `StaticLayer`（:398）—— 预分配，为 compile 而生

```python
# cache_utils.py:432-441（lazy_initialization 里）
self.keys = torch.zeros(
    (self.batch_size, self.num_heads, self.max_cache_len, self.k_head_dim),
    dtype=self.dtype, device=self.device,
)
```

`update` 的核心是**原地写**（:474-481）：

```python
kv_length = key_states.shape[-2]
cache_position = torch.arange(kv_length, device=self.device) + self.cumulative_length
self.cumulative_length.add_(kv_length)          # 原地 add_，不能重新赋值
self.keys.index_copy_(2, cache_position, key_states)
self.values.index_copy_(2, cache_position, value_states)
```

**为 compile 付出的三处代价**（都是真实注释）：

1. `cumulative_length` 必须是 **tensor** 而不是 int（:414-415）：
   `# Very important that it's a tensor here, to avoid recompiling when we update it and use it to create positions`
2. `mark_static_address`（:448-451）：
   `# Without this, we cannot use cudagraphs!!`——它把张量标记成「地址固定」，
   否则 cudagraph 会因为指针变化而跳过捕获。
3. `get_mask_sizes` 恒返回 `self.max_cache_len`（:489-493），**不管实际填了多少**：
   形状必须静态。多出来的未填位置全是 `0`，靠 mask 屏蔽掉。
   这就是 `kv_cache.md` 里说的「a lot of tokens will actually be masked」——
   **静态 cache 是用算力换「可编译」**。

### 4.5 `StaticSlidingWindowLayer`（:504）/ `StaticIndexedLayer`（:631）

**`StaticSlidingWindowLayer`**：分配时直接取 `min(sliding_window, max_cache_len)`（:520），
填满之后用 `roll` 实现「整体左移一位」（:551-563）：

```python
new_keys = self.keys.roll(-1, dims=-2)      # 全部左移，最老的滚出
index = torch.tensor([-1], dtype=int, device=self.device)
new_keys[:, :, index] = key_states          # 最新一个写到末尾
self.keys.copy_(new_keys)                    # copy_ 而不是赋值，保住静态地址
```

注释里明确说这是绕 dynamo 的 bug（`# dynamo is currently bugged when doing it - see pytorch/pytorch#159855`）。
**这就是 `cache_explanation.md` 里那句「existing tokens are shifted out of the cache」的实现。**

**`StaticIndexedLayer`**：`DynamicIndexedLayer` 的静态版，`indexer_cumulative_length` 是**独立的**一个 tensor
（:645-647 注释：indexer 的 update 在主 K/V 的 update **之后**跑，所以不能复用同一个计数器）。
同样 `mark_static_address`（:659-661）。

### 4.6 量化三兄弟：`QuantizedLayer`（:703）/ `QuantoQuantizedLayer`（:779）/ `HQQQuantizedLayer`（:834）

按 [KIVI 论文](https://huggingface.co/papers/2402.02750) 的思路（docstring :705 明确引用）。

**双层存储**（这是理解它的关键）：

| 存储 | 精度 | 容量 | 行号 |
|---|---|---|---|
| `self.keys` / `self.values` | 原始精度 | `residual_length`（默认 128） | :728 |
| `self._quantized_keys` / `self._quantized_values` | 量化 | 无限（但每通道组量化） | :758-759 |

**工作流**（`update` :731-766）：

```
累积到 residual_length：
  新 K/V 先放原始精度区，和上一步解量化出来的历史拼在一起返回
达到 residual_length：
  把「全部历史」重新 _quantize 一次 → 塞进量化区，原始精度区清空（:760-761）
```

**返回值永远是「解量化后的历史 + 原始精度区 + 本次」**（:755-756），
所以注意力端拿到的始终是近似的完整 K/V——**量化对 attention 是透明的**。

**两个后端的差异**：

| | `QuantoQuantizedLayer` | `HQQQuantizedLayer` |
|---|---|---|
| 依赖 | `optimum-quanto >= 0.2.5`（:802） | `hqq`（:851） |
| `nbits` 支持 | `[2, 4]`（:809-810） | `[1, 2, 3, 4, 8]`（:857-860） |
| `axis` 支持 | `[0, -1]`（:812-818） | `[0, 1]`（:862-866） |
| 量化实现 | `MaxOptimizer` + `quantize_weight`（:823-828） | `HQQQuantizer.quantize`（:870-883） |

**限制**：`QuantizedCache` 只支持**全部层都是 full attention** 的模型——
`cache_utils.py:1934-1939` 会检查 `layer_types - {"full_attention"}` 是否为空，不为空直接 `raise ValueError`。
（滑动窗口 + 量化目前不支持。）

### 4.7 线性注意力系列：`LinearAttentionCacheLayerMixin`（:891）+ `LinearAttentionLayer`（:1003）

**给 SSM / 线性注意力用的**（Mamba 2、Jamba、Zamba、Falcon-H1、LFM2、Qwen3-Next 等）。

**它和上面所有 Layer 的根本区别：没有序列长度维度。**

```
普通注意力层缓存的：seq_len 个 token 的 K/V        → 形状 [B, H, S, D]，S 随生成增长
线性注意力层缓存的：一个固定的递归状态             → 形状固定，与 S 无关
```

所以它的抽象契约完全不同（`LinearAttentionCacheLayerMixin`，:891）：

| 抽象方法 | 行号 |
|---|---|
| `lazy_initialization(conv_states=None, recurrent_states=None, state_idx=0)` | :915-921 |
| `update_conv_state(conv_states, state_idx=0)` | :923-924 |
| `update_recurrent_state(recurrent_states, state_idx=0)` | :926-927 |

**注意：它根本没有 `update` / `get_seq_length` / `get_mask_sizes`。** 这带来两个后果：

1. 它**不是** `CacheLayerMixin` 的子类（是独立的 `ABC`）；
2. `Cache.get_seq_length` / `get_mask_sizes` 里要专门处理这种情况
   （:1498-1514 / :1574-1588）：如果目标是线性注意力层，就**自动跳到第一个真正的注意力层**
   去取序列长度——注释写得很直白：
   `# For alternating attention/linear attention caches, ... needs to use attention layer idx when called with default layer_idx`

**两个 dict 装状态**（:902-903），因为一层里可能有**多个** conv / recurrent 状态
（`number_of_states`，由 `number_of_conv_states` 配置决定，
`get_layer_types_and_kwargs` :1731-1732）：

```python
self.conv_states: dict[int, torch.Tensor | None] = dict.fromkeys(range(number_of_states))
self.recurrent_states: dict[int, torch.Tensor | None] = dict.fromkeys(range(number_of_states))
```

实测用 `number_of_conv_states` 的模型：`models/inkling/configuration_inkling.py:176`（= 4）、
`models/qwen4_exp/configuration_qwen4_exp.py:172`（= 3 或 1）。

**`moe` / `mlp` 层也被映射成 `LinearAttentionLayer`**（:1242-1243 / :1264-1265），注释解释了原因：
它们是**空的占位层**（MoE/MLP 不需要 cache），但用 `LinearAttentionLayer` 可以让
「按层取序列长度」的逻辑跳过它们，避免在 mask 构造时打架。

### 4.8 `layer_type` → Layer 的映射表

`cache_utils.py:1223-1244`（动态）和 `:1246-1266`（静态）。**这张表是「config 里的字符串」到「类」的唯一入口**：

| `layer_type` 字符串 | 动态映射 | 静态映射 | 典型模型 |
|---|---|---|---|
| `full_attention` | `DynamicLayer` | `StaticLayer` | 几乎所有模型 |
| `sliding_attention` | `DynamicSlidingWindowLayer` | `StaticSlidingWindowLayer` | Mistral、Gemma 2/3 |
| `chunked_attention` | `DynamicSlidingWindowLayer` | `StaticSlidingWindowLayer` | Llama 4 |
| `conv` / `linear_attention` | `LinearAttentionLayer` | `LinearAttentionLayer` | Mamba 2（`configuration_mamba2.py:106-107` 返回 `["linear_attention"] * num_hidden_layers`）、LFM2 |
| `hybrid` / `hybrid_sliding` | `LinearAttentionAndFullAttentionLayer` / `...SlidingWindowAttentionLayer` | `LinearAttentionAndStatic*Layer` | Zamba、Falcon-H1、Jamba |
| `deepseek_sparse_attention` / `qwen_sparse_attention` | `DynamicIndexedLayer` | `StaticIndexedLayer` | **GLM-5.3（2M7）**、DeepSeek V3.2 |
| `moe` / `mlp` | `LinearAttentionLayer`（空占位） | 同左 | MoE 模型 |
| `heavily_compressed_attention` / `compressed_sparse_attention` | 由模型文件注册 | 同左 | DeepSeek V4（`modeling_deepseek_v4.py:185`、`:265`） |

**`sliding_attention` 和 `chunked_attention` 映射到同一个类**，注释解释了（:1225）：
`# From a cache point of view, sliding and chunked are the same in how they should behave, only the mask differs`
——**cache 只管存多少，不管能不能看**，后者是 2A2 的事。

### 4.9 Layer 子类对比总表

| Layer 类 | 行号 | 存储 | `get_max_length()` | 可编译 | 显存 | 适用场景 |
|---|---|---|---|---|---|---|
| `DynamicLayer` | :113 | `cat` 生长 | `-1` | ✗ | `O(S)` | 默认；短序列、变长 |
| `DynamicSlidingWindowLayer` | :203 | 只留 `W-1` | `W` | ✗ | **`O(W)` 常数** | SWA（Mistral / Gemma） |
| `DynamicIndexedLayer` | :319 | K/V + `indexer_keys` | `-1`（父类） | ✗ | `O(S)` + 小头 | **DSA（GLM / DSV3.2）** |
| `StaticLayer` | :398 | 预分配 `O(max_len)` | `max_cache_len` | **✓** | `O(max_len)` | `torch.compile` / CUDA graph |
| `StaticSlidingWindowLayer` | :504 | 预分配 `min(W, max_len)` | `min(W,max_len)` | **✓** | **`O(W)` 常数** | 编译 + SWA |
| `StaticIndexedLayer` | :631 | 预分配 + indexer | `max_cache_len` | **✓** | 同 StaticLayer | 编译 + DSA |
| `QuantizedLayer`（ABC） | :703 | 量化区 + 残留区 | `-1`（父类） | ✗ | **`O(S × nbits/16)`** | 长序列省显存 |
| `QuantoQuantizedLayer` | :779 | 同上 | 同上 | ✗ | 同上 | `backend="quanto"` |
| `HQQQuantizedLayer` | :834 | 同上 | 同上 | ✗ | 同上 | `backend="hqq"` |
| `LinearAttentionLayer` | :1003 | `conv_states` + `recurrent_states` | `-1` | **✓** | **`O(1)`** | **Mamba 2（2M5）等 SSM** |
| `LinearAttentionAnd*Layer`（4 个） | :1094 / :1134 / :1166 / :1198 | 两套状态 | 混合 | 动态版 ✗ / 静态版 ✓ | `O(S) + O(1)` | 混合模型（Jamba / Zamba） |

---

## 五、Cache 容器子类

`Cache`（:1269）是所有容器的基类。它自己不存 K/V，只做三件事：

1. **按 `layer_idx` 分发**（`update` :1356-1388）；
2. **提供跨层聚合的接口**（`get_seq_length` / `get_mask_sizes` / `get_max_length`）；
3. **调度与所有层相关的事**（offloading :1324-1354、`early_initialization` :1455-1490）。

关键成员（实测行号）：

| 成员 | 行号 | 说明 |
|---|---|---|
| `self.layers` | :1306 | Layer 列表。**这是整个 cache 的全部存储** |
| `self.layer_class_to_replicate` | :1307 | 懒创建用：`layers` 和它**二选一**，都传或都不传会报错（:1296-1305） |
| `update(key, value, layer_idx)` | :1356-1388 | 分发给 `self.layers[layer_idx].update(...)` |
| `update_conv_state` / `update_recurrent_state` | :1390 / :1412 | 线性注意力专用，内部有 `isinstance` 校验 |
| `update_indexer` | :1434-1453 | DSA 专用；层不是 Indexed 层会给出很清楚的报错 |
| `early_initialization` | :1455-1490 | 见 6.4 |
| `get_seq_length(layer_idx=0)` | :1492-1514 | **自动跳过线性注意力层** |
| `get_max_length(layer_idx=None)` | :1516-1530 | 不传参数时返回**所有层的最大值** |
| `get_mask_sizes(query_length, layer_idx)` | :1562-1590 | 转给对应 Layer |
| `get_query_offset(layer_idx=0)` | :1592-1597 | 默认就是 `get_seq_length()`；**MTP / DFlash 会覆写它** |
| `crop` / `reset` / `reorder_cache` | :1609 / :1599 / :1604 | 逐层遍历 |
| `is_compileable` / `is_sliding` / `is_linear` / `batch_size` | :1651 / :1665 / :1670 / :1638 | 逐层聚合出来的属性 |

### 5.1 `DynamicCache`（:1737）—— 默认实现

构造逻辑（:1780-1822）分三条路：

```python
if config is not None:                      # ① 有 config → 逐层按 layer_types 建（:1789-1793）
    layer_types, layer_kwargs = get_layer_types_and_kwargs(decoder_config)
    layers = [DYNAMIC_LAYER_TYPE_MAPPING[t](**layer_kwargs) for t in layer_types]
if ddp_cache_data is not None:              # ② DDP 反序列化路径（:1796-1812）
    ...
if len(layers) == 0:                        # ③ 都没有 → 懒创建 DynamicLayer（:1815-1820）
    super().__init__(layer_class_to_replicate=DynamicLayer, ...)
```

`get_layer_types_and_kwargs`（:1701-1734）是「config → layer_types + 初始化参数」的翻译函数：

```python
if layer_types is None:
    if getattr(config, "sliding_window", None) is not None:
        layer_types = ["sliding_attention"] * config.num_hidden_layers      # :1709-1710
    elif getattr(config, "attention_chunk_size", None) is not None:
        layer_types = ["chunked_attention"] * config.num_hidden_layers      # :1711-1712
    else:
        layer_types = ["full_attention"] * config.num_hidden_layers         # :1713-1714
```

**这就是「为什么 Mistral 不用改一行模型代码就能拿到 SWA cache」的答案**（见 4.2 的链路）。
另外注意 :1717-1719 会砍掉 `num_kv_shared_layers`（Gemma 3n 那种共享 KV 的层不需要 cache）。

### 5.2 `StaticCache`（:1829）/ `QuantizedCache`（:1884）

`StaticCache` 强制要求 `config` + `max_cache_len`，构造时把 `max_cache_len` 塞进 `layer_kwargs`
一起分发（:1877-1881）。**`max_cache_len` 必须由用户给出**——这就是
`GenerationConfig.max_cache_len` 存在的原因（`generation/configuration_utils.py:691-694` 会校验
「只有 static cache 才能用这个参数」）。

`QuantizedCache` 不逐层分发，而是**直接建 `num_hidden_layers` 个同构层**（:1940-1943），
并在前面做「只准 full attention」的校验（:1934-1939）。

### 5.3 `EncoderDecoderCache`（:1947）—— 呼应 2M4 T5

**结构：组合，不是继承。**

```python
self.self_attention_cache = caches[0]     # :2000
self.cross_attention_cache = caches[1]    # :2001
self.is_updated = {}                      # :2006-2008  每层 cross cache 是否已经填过
```

| 字段 | 含义 |
|---|---|
| `self_attention_cache` | 解码器**自注意力**的 cache。**随生成增长**（这是"记忆"） |
| `cross_attention_cache` | 解码器**交叉注意力**的 cache。**对 encoder 输出，只算一次**，之后整段复用 |
| `is_updated` | 逐层记录 cross cache 是否已填（`get_seq_length > 0`）。因为 cross cache 只需 prefill 一次，后续步要跳过 |

**它的所有方法几乎都是「转发给 self_attention_cache」**：

| 方法 | 行号 | 转发给 |
|---|---|---|
| `get_seq_length` | :2028 | `self_attention_cache` |
| `get_max_length` | :2032 | `self_attention_cache` |
| `get_mask_sizes` | :2077 | `self_attention_cache` |
| `crop` | :2057 | **只裁 self**（cross 不能裁，见 :2062-2063） |
| `batch_repeat_interleave` / `batch_select_indices` | :2065 / :2071 | **两个都做** |

`crop` 只裁 self cache 这一条很值得记：**cross-attention 的 K/V 是 encoder 输出的函数，
和生成进度无关，裁了就错了。**

`__iter__`（:2010-2013）产出 6 元组：
`(self_k, self_v, self_sliding, cross_k, cross_v, cross_sliding)`——这是给 DDP 序列化用的。

### 5.4 `MtpCache`（:2102）/ `DFlashCache`（:2114）—— 实测确认它们是什么

两个都是 `DynamicCache` 的子类，**覆写点都只有一个：`get_query_offset` 和 `get_mask_sizes`**。
也就是说，它们改变的不是「存什么」，而是「**query 的位置从哪里开始算**」。

**`MtpCache`**（Multi-Token Prediction，多 token 预测 / MTP 投机解码）：

```python
# cache_utils.py:2103-2106
def get_query_offset(self, layer_idx: int = 0):
    # Queries of MTP depth k run k+1 tokens ahead of the main_model, i.e. they have an offset of k+1
    mtp_offset = layer_idx + 1
    return super().get_query_offset(layer_idx) + mtp_offset
```

- 使用点：`generation/candidate_generator.py:1423` `class MTPCandidateGenerator`，
  在 `:1451` 建 `MtpCache(config=main_model.config.get_mtp_config())`，
  并立刻 `self.mtp_cache.activate_past_recording()`（因为投机解码要能回滚）。
- 语义：第 k 层 MTP 预测的是比主模型**超前 k+1 个位置**的 token，
  所以它的 query 在 mask 里的偏移要 `+ k+1`。

**`DFlashCache`**（`cache_utils.py:2114-2131`）：

```python
def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._previous_number_of_accepted_tokens = 0

def set_previous_accepted_tokens(self, number_of_accepted_tokens: int): ...   # :2119

def get_query_offset(self, layer_idx: int = 0):
    # During one forward, the hidden states that will create queries do NOT have the same length as those that
    # will create k/v, so all the kv length is shifted by the previous number of accepted tokens
    return super().get_query_offset(layer_idx) + self._previous_number_of_accepted_tokens
```

- 使用点：`generation/candidate_generator.py:1564` `class DFlashTokenCandidateGenerator`，
  在 `:1611` 建 `DFlashCache(config=self.assistant_model.config)`。
- 语义：DFlash 是**扩散式**草稿（有 `block_size` / `mask_token_id` / `target_layer_ids`，
  见 `candidate_generator.py:1605-1610`），一个 forward 里 Q 和 K/V 的长度不同，
  所以要用「上一轮被接受多少个 token」来平移。

> **一句话**：`MtpCache` / `DFlashCache` 是**投机解码的偏移量适配器**，
> 不是新的存储策略。读 `cache_utils.py` 读到最后这两个类时不要期待有架构创新。

### 5.5 Cache 容器对比表

| 容器 | 行号 | 层类型 | 要 config? | 要 max_len? | 可 compile | 适用 |
|---|---|---|---|---|---|---|
| `Cache` | :1269 | 任意（由入参决定） | 否（可只给 `layer_class_to_replicate`） | 否 | 看层 | **基类，不直接用** |
| `DynamicCache` | :1737 | 动态系列 | 可选（给了才支持 SWA/hybrid） | 否 | ✗ | **默认** |
| `StaticCache` | :1829 | 静态系列 | **是** | **是** | **✓** | `cache_implementation="static"` |
| `QuantizedCache` | :1884 | `Quanto/HQQQuantizedLayer` | **是** | 否 | ✗ | `cache_implementation="quantized"` |
| `EncoderDecoderCache` | :1947 | 组合两个 Cache | 由子 Cache 决定 | 由子 Cache 决定 | 看子 Cache | T5 / Whisper（2M4） |
| `MtpCache` | :2102 | `DynamicCache` | 可选 | 否 | ✗ | MTP 投机解码 |
| `DFlashCache` | :2114 | `DynamicCache` | 可选 | 否 | ✗ | DFlash 扩散草稿 |

**`generate` 侧怎么选**（`generation/utils.py:1924` `_prepare_cache_for_generation`）：

| `GenerationConfig.cache_implementation` | 得到的容器 | 行号 |
|---|---|---|
| `None` / `"dynamic"` / `"offloaded"` | `DynamicCache` | :2013-2014 |
| `"static"` / `"offloaded_static"`（+ 4 个已废弃别名） | `StaticCache` | :1980-2000 |
| `"quantized"` | `QuantizedCache`（`backend` 从 `cache_config` 取） | :2001-2011 |
| 只要 `config.is_encoder_decoder` | **外面再套一层** `EncoderDecoderCache` | :2019-2024 |
| `"paged"` | 不走这里，转连续批处理 | :2392-2394 |

常量定义在 `generation/configuration_utils.py:47-58`（`STATIC_CACHE_IMPLEMENTATIONS`、
`DEPRECATED_STATIC_CACHE_IMPLEMENTATIONS = ("sliding_window", "hybrid", "hybrid_chunked", ...)`
——**v5 把 `"sliding_window"` / `"hybrid"` 这些名字废弃了，改成从 config 自动推断层结构**）。

---

## 六、`update()` 的调用契约

### 6.1 模型侧长什么样

[1M1](./1M1_llama_structure.md) 里那一行（`models/llama/modeling_llama.py:262`，实测）：

```python
if past_key_values is not None:
    key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
```

**它在 attention 模块内部，在 RoPE 之后、后端调用之前**（`modeling_llama.py:255-271` 的顺序）：

```
hidden_states
  → q_proj / k_proj / v_proj          # 投影
  → apply_rotary_pos_emb(q, k)        # 旋转
  → past_key_values.update(k, v, layer_idx)   # ★ 缓存
  → ALL_ATTENTION_FUNCTIONS.get_interface(...)(q, k, v, ...)   # 后端
  → o_proj
```

**顺序为什么必须是「先 RoPE 再 cache」**：cache 里存的必须是**旋转后的 K**，
否则每步取出来还得按历史位置重转一遍。这也是为什么 cache 里的 K 是「位置相关」的，
不能跨位置复用（除非用相对位置编码，那是 2A4 的话题）。

**三条契约**：

1. **`key_states` / `value_states` 是本次新算的**（不是历史），形状 `[B, H_kv, S_q, D]`。
2. **返回值是「完整」的 K/V**（历史 + 本次），形状 `[B, H_kv, S_kv, D]`，`S_kv ≥ S_q`。
   attention 直接拿它去算，不需要再拼。
3. **必须原地写回语义**——模型侧**不能**把返回值当成「新的 cache」再存回去。
   对 `StaticLayer` 来说，返回值甚至**就是 `self.keys` 本身**（:487），再存一遍会把地址搞乱。

**「返回完整 K/V」不是形式主义，而是三种 Layer 的必然选择**：

| Layer | 返回什么 | 为什么 |
|---|---|---|
| `DynamicLayer` | `self.keys, self.values` | 就是全部 |
| `DynamicSlidingWindowLayer` | **`full_key_states`（未裁剪的）** | 裁剪后的 `self.keys` 少了最新的位置；mask 用 `kv_offset` 把窗口对齐到绝对位置 |
| `StaticLayer` | **`self.keys`（整块预分配缓冲）** | 形状必须静态；未填位置靠 mask 屏蔽 |
| `QuantizedLayer` | **解量化历史 + 原始区 + 本次** | 量化对 attention 透明 |

### 6.2 `layer_idx` 怎么定位到对应层

`Cache.update`（:1356-1388）就三步：

```python
if self.layer_class_to_replicate is not None:          # ① 懒创建缺口
    while len(self.layers) <= layer_idx:
        self.layers.append(self.layer_class_to_replicate())     # :1374-1376
if self.offloading:                                     # ② offloading 调度
    torch.cuda.default_stream(key_states.device).wait_stream(self.prefetch_stream)
    self.prefetch(layer_idx + 1, self.only_non_sliding)
keys, values = self.layers[layer_idx].update(key_states, value_states, *args, **kwargs)   # ③ 真干活
if self.offloading:
    self.offload(layer_idx, self.only_non_sliding)
```

**`layer_idx` 从哪来？** 来自 `nn.Module` 自己的 `self.layer_idx`
（`LlamaAttention.__init__` 里 `self.layer_idx = layer_idx`，实测在 `modeling_llama.py:223`；
DecoderLayer 构造时逐层传入）。**模型侧不需要知道 cache 里有多少层**。

**懒创建**（①）是 `DynamicCache()` 不带 config 时的工作模式：
第一次 forward 会按层数逐步把 `layers` 撑起来。`Cache.__len__`（:1316-1322）注释明确说：
`# Note: for DynamicCache, layers are initialized lazily, so this will not be accurate before the first forward through all the layers`。

**offloading**（②）是个容易被忽略的细节：它用的是 `prefetch(layer_idx + 1)`——
**在算第 `layer_idx` 层的时候，已经在用后台 stream 把第 `layer_idx + 1` 层拉回 device 了**（:1324-1345）。
`only_non_sliding=True` 时滑动窗口层不参与 offload（:1332-1335 注释：滑动层本来就小，搬来搬去反而慢）。

### 6.3 `cache_position` / `cache_kwargs`：**在这个版本里已经没有了**

老教程里模型会接收 `cache_position`（一个 `arange`）和 `cache_kwargs`（一个 dict）。
**实测：这两个东西在本仓库已经不存在。** 证据：

```bash
$ grep -rn "cache_position" src/transformers/generation/utils.py
623:        # BC for remote code models only: create `cache_position` on the fly here, as we don't want to maintain them in kwargs
625:        if self.is_remote_code() and "cache_position" in set(inspect.signature(self.forward).parameters):
627:            "The remote code model you are currently using seems to expect `cache_position`. This arg has been "
629:            "removed from the Transformers library, and will stop being created in `generate` even for remote code models "
632:            cache_position = torch.arange(sequence_length, device=input_ids.device) + past_seen_tokens
633:            model_inputs["cache_position"] = cache_position
```

`cache_utils.py` 里只剩 `StaticLayer.update` 内部**局部变量**叫 `cache_position`（:474、:583、:680）——
那是「这次要写到缓冲区哪几个槽位」的下标，**和模型接口无关**。

**它们被什么替代了？** 两条各自解决一半：

| 老 `cache_position` 的用途 | 现在的替代 | 代码位置 |
|---|---|---|
| 算 `position_ids`（RoPE 用） | `past_key_values.get_seq_length()` | `modeling_llama.py:387` |
| 算 mask 的 q 偏移 | `past_key_values.get_query_offset(layer_idx)` | `masking_utils.py:829` |
| 算 mask 的 kv 长度/偏移 | `past_key_values.get_mask_sizes(q_length, layer_idx)` | `masking_utils.py:833` |
| `cache_kwargs`（给 cache 传额外参数） | cache 在构造时就知道一切（`config` / `sliding_window` / `max_cache_len`） | `cache_utils.py:1793`、`:1880` |

**设计意图**：位置信息不再由外面推给 cache，而是 **cache 自己算完报给外面**。
好处是「缓存的裁剪策略」和「位置/mask 的正确性」被绑在一起，
`DynamicSlidingWindowLayer` 那种「内部存 3 个、但绝对位置已经到第 100000 个」的情况
才不可能算错。

### 6.4 懒初始化与 `early_initialization`

**为什么懒初始化**（`StaticLayer.lazy_initialization` docstring，:418-425 原文摘要）：

> This allows to get all properties (dtype, device, num_heads in case of TP etc...) at runtime directly,
> which is extremely practical as it avoids moving devices, dtypes etc later on for each `update`

也就是说：**建 cache 的时候不分配显存**。因为那一刻我们不知道：
- 激活的 dtype 是什么（`torch_dtype` 可能是 `bfloat16`，也可能被 autocast 改过）
- TP 之后每张卡上 `num_heads` 是多少
- 甚至 `v_head_dim` 和 `k_head_dim` 可能不同（MLA 模型）

这些都是「第一次 forward 时」才知道的。

**例外：`early_initialization`**（`Cache.early_initialization` :1455-1490）。
`torch.export` / chunked prefill 要求**在编译前**形状就固定，所以需要提前分配。
它用一个巧妙的假张量（:1488-1490，注释 `fake tensor`）：

```python
fake_kv_tensor = torch.zeros((batch_size, layer_num_heads, 0, layer_head_dim), dtype=dtype, device=device)
layer.lazy_initialization(fake_kv_tensor, fake_kv_tensor)
```

**在 seq_len 维上长度是 0**——不占任何显存，只传递形状 / dtype / device 信息。

`supports_early_init`（`:31`）这个类属性的意义就在这：
`LinearAttentionCacheLayerMixin` 把它设成 `False`（:897），
因为线性注意力层「状态形状是自己算出来的」，没有 `[.., S, D]` 这个维度可供 fake。
`early_initialization` 里会 `continue` 跳过它们（:1483-1484）。

---

## 七、cache 与 mask 的耦合（→ 2A2）

**这是「三件套」串起来的地方。** 全篇最需要理解的一节。

### 7.1 模型侧的两个使用点

`LlamaModel.forward`（`modeling_llama.py:383-399`，实测）：

```python
if use_cache and past_key_values is None:
    past_key_values = DynamicCache(config=self.config)          # :384

if position_ids is None:
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0   # :387
    position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
    position_ids = position_ids.unsqueeze(0)                    # :388-389

causal_mask = create_causal_mask(                               # :391
    config=self.config,
    inputs_embeds=inputs_embeds,
    attention_mask=attention_mask,
    past_key_values=past_key_values,                            # ★ 传给 mask
    position_ids=position_ids,
)
```

**两处耦合**：`position_ids` 用 `get_seq_length()`；mask 用 `past_key_values` 本身。

### 7.2 `create_causal_mask` 拿 `past_key_values` 干什么

`create_causal_mask`（`masking_utils.py:864`）把它转交给 `_preprocess_mask_arguments`（:760）。
**核心就两行**（`masking_utils.py:828-833`，实测）：

```python
if past_key_values is not None:
    q_offset = past_key_values.get_query_offset(layer_idx)
    q_offset = q_offset.to(inputs_embeds.device) if isinstance(q_offset, torch.Tensor) else q_offset
    kv_length, kv_offset = past_key_values.get_mask_sizes(q_length, layer_idx)
```

**它用这几个信息做三件事**：

| 用途 | 说明 |
|---|---|
| **决定 mask 的形状** | `kv_length` → mask 的最后两维是 `(q_length, kv_length)`。这就是「mask 要覆盖缓存里的历史」这一步 |
| **对齐绝对位置** | `q_offset` / `kv_offset` 是**绝对位置**的起点。mask 函数内部比较的是 `kv_idx > q_idx - W` 这种**绝对位置**，所以两端都得加上 offset |
| **省掉不必要的计算** | `allow_is_causal_skip`（:953-955）：如果 mask 恰好就是一个规整 causal mask（`q_offset == kv_offset` 且长度对得上），可以**根本不生成 mask 矩阵**，直接交给后端自己 causally 算（`is_causal=True`） |

**`q_offset` 的默认值是 `get_seq_length()`**（`Cache.get_query_offset` :1592-1597）——
所以 `q_offset == kv_offset == past_len` 在普通 `DynamicLayer` 上恒成立，
「规整 causal」的优化就能命中。而 `MtpCache` / `DFlashCache` 覆写它就是为了**打破这个相等**
（见 5.4），让 mask 知道「query 跑在更前面」。

### 7.3 mask 是按「层类型」而不是「层号」建的

`create_causal_mask` 里有一段（`masking_utils.py:929-933`）：

```python
if layer_idx is None:
    if hasattr(past_key_values, "is_sliding") and False in past_key_values.is_sliding:
        layer_idx = past_key_values.is_sliding.index(False)     # 找一个 full attention 层当代表
    else:
        layer_idx = 0
```

**为什么能这么干**：**同类型的层，`get_mask_sizes` 的返回值必然相同**。
所以混合模型里只需要按 `layer_types` 里的**每种类型各建一张 mask**，然后各层共用。

这个「按类型建 mask」的动作在 `generate` 里由 `create_masks_for_generate`
（`masking_utils.py:1523`）统一做，映射表是 `LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING`（:1507-1520）：

| `layer_type` | mask 函数 |
|---|---|
| `full_attention` | `create_causal_mask` |
| `sliding_attention` / `chunked_attention` | `create_sliding_window_causal_mask` / `create_chunked_causal_mask` |
| `deepseek_sparse_attention` / `qwen_sparse_attention` / `minimax_m3_sparse` | `create_causal_mask` |
| `compressed_sparse_attention` / `heavily_compressed_attention` | `create_sliding_window_causal_mask` |
| `linear_attention` / `conv` | `create_recurrent_attention_mask` |
| `hybrid` / `hybrid_sliding` | **dict**：`{full: create_causal_mask, linear: create_recurrent_attention_mask}` |

模型侧拿到的是**一个 dict**（`masking_utils.py:1582-1592` 的 `causal_masks`），
然后逐层取，例如 GLM（`modeling_glm_moe_dsa.py:743`，实测）：

```python
attention_mask=causal_mask_mapping[self.config.layer_types[i]],
```

> **一句话**：**cache 的分层（`layer_types`）决定了 mask 的分层。**
> 这就是 2A2 和 2A3 的接口——`LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING` 那张表。

---

## 八、序列生长示意：动态 vs 静态 vs 滑动窗口

以「已经生成到第 6 个 token、窗口 W = 4」为例。

```
═══ DynamicLayer（无上限） ═══
内部 keys:  [k1 k2 k3 k4 k5 k6]                    ← 每步 cat，越来越长
返回给 attn:[k1 k2 k3 k4 k5 k6]
mask:        [1, 1, 1, 6]
显存:        O(S)   随 S 线性涨   ✗ 不可编译（地址每步都变）

═══ StaticLayer（max_cache_len = 8） ═══
内部 keys:  [k1 k2 k3 k4 k5 k6  ·  · ]            ← 预分配，index_copy_ 原地写
返回给 attn:[k1 k2 k3 k4 k5 k6  ·  · ]            ← 静态形状，多了 2 个 0
mask:        [1, 1, 1, 8]                          ← ★ 永远是 8，靠 mask 屏蔽后两列
显存:        O(max_len)  与 S 脱钩（但按 max_len 付费）   ✓ 可编译

═══ DynamicSlidingWindowLayer（W = 4） ═══
内部 keys:  [k4 k5 k6]                             ← 只留 W-1 = 3 个
返回给 attn:[k3 k4 k5 k6]                          ← 恒为 W = 4 个
get_mask_sizes(1) → (4, 3)                         ← kv_offset = 绝对位置 3
mask:        [1, 1, 1, 4]（对齐到绝对位置 3..6）
显存:        O(W)   常数，和 S 无关                      ✗ 不可编译（动态层）

═══ StaticSlidingWindowLayer（W = 4） ═══
内部 keys:  [k3 k4 k5 k6]                           ← roll + copy_，地址不变
返回给 attn:[k3 k4 k5 k6]（或更长的未裁剪 full）
显存:        O(W)   常数                                ✓ 可编译

═══ LinearAttentionLayer（SSM） ═══
内部:        recurrent_states[0] = 固定形状的状态张量   ← 完全没有 S 维度
             conv_states[0]      = 长度 conv_kernel 的滑窗缓冲
显存:        O(1)   和 S 完全无关                        ✓ 可编译
```

**读这张图要抓住的三个「长度」**（很容易混）：

| 概念 | 出处 | 含义 |
|---|---|---|
| `get_seq_length()` | `Cache` :1492 | **真实已经见过多少 token**。用来算 `position_ids`。滑动窗口层返回的是真实总数（:277） |
| `keys.shape[-2]` | Layer 内部 | **实际存了多少**。滑动窗口层 = `W-1`，静态层 = `max_cache_len` |
| `get_mask_sizes()` 的 `kv_length` | Layer :60 | **这次要传给 attention 多长**。可以和上两个都不等 |

---

## 九、显存计算实操

### 9.1 公式

```
KV cache 字节数 = 2 × L × B × H_kv × S × D_head × bytes_per_element
                  │   │   │    │     │    │
                  │   │   │    │     │    └─ 每元素字节：fp16=2, fp8=1, int4=0.5
                  │   │   │    │     └─ head_dim
                  │   │   │    └─ 每 token 每层的 K/V 宽度 = 2 × H_kv × D_head
                  │   │   └─ KV 头数（GQA 时 < num_attention_heads）
                  │   └─ batch size
                  └─ 层数
                  （最前面的 2 = K 和 V 各一份）
```

**公式在源码里的落点**：`StaticLayer.lazy_initialization`（`cache_utils.py:432-441`）分配的正是

```python
self.keys  = torch.zeros((batch_size, num_heads, max_cache_len, k_head_dim), ...)
self.values= torch.zeros((batch_size, num_heads, max_cache_len, v_head_dim), ...)
```

即 `2 × B × H × S × D` 一份**每层**；`H_kv` 的取值规则见
`generation/continuous_batching/cache.py:29-39`（`find_num_kv_heads`：
优先 `num_key_value_heads`，没有就用 `num_attention_heads`）
和 `:42-53`（`find_head_dim`：优先 `head_dim`，没有就用 `hidden_size // num_attention_heads`）。

### 9.2 具体算一遍（Llama-3-8B，batch=1，seq=8192，fp16）

取 `L = 32`、`H_kv = 8`（GQA）、`D_head = 128`、`bytes = 2`：

```
2 × 32 × 1 × 8 × 8192 × 128 × 2 = 1,073,741,824 字节 = 1.000 GiB
```

**对照模型权重**：8.03B 参数 × 2 字节 ≈ **14.96 GiB**。
也就是说 seq=8192 时 KV cache 已经占掉权重的 **6.7%**；seq 上去以后会迅速反超。

### 9.3 显存表（全部按上面的公式手算，数字可复现）

| 场景 | 参数 | 结果 |
|---|---|---|
| Llama-3-8B，b=1，S=8192，fp16 | `2×32×1×8×8192×128×2` | **1.000 GiB** |
| Llama-3-8B，b=1，S=32768，fp16 | 同上但 S×4 | **4.000 GiB** |
| Llama-3-8B，b=32，S=4096，fp16 | `2×32×32×8×4096×128×2` | **16.000 GiB** |
| （假设）MHA 版：H_kv=32 而非 8 | `2×32×1×32×8192×128×2` | **4.000 GiB**（GQA 省了 4×） |
| Llama-3-8B，b=1，S=8192，fp8 | 每元素 1 字节 | **0.500 GiB** |
| Llama-3-8B，b=1，S=8192，int4 | 每元素 0.5 字节 | **0.250 GiB** |
| **Mistral 7B（SWA W=4096），任意 S** | `2×32×1×8×4095×128×2` | **≈ 0.500 GiB（上限，与 S 无关）** |

**从这张表能读出四个工程结论**：

1. **GQA 是免费的 4 倍**（32 头 → 8 头）。这就是为什么 Llama-2 之后的模型全上 GQA
   （呼应 [1M1](./1M1_llama_structure.md) 里「K、V 的头数可以比 Q 少，省 KV cache」那句）。
2. **batch 维是乘性的**。`b=32, S=4096` 就直接吃 16 GiB——**吞吐的代价是显存**。
   这就是为什么生产服务要上连续批处理（第 10 节）。
3. **量化是线性的**：int4 直接 1/4。
4. **SWA 是唯一「和 S 无关」的省法**。改 `sliding_window` 是长上下文最粗暴有效的手段
   （代价是模型能力，见 1M2）。

> **本节的 config 数值说明**：`L / H_kv / D_head` 用的是 Llama-3-8B 公开的 config 常量
> （32 层 / 8 个 KV 头 / head_dim 128）。**本机无网络，没能现场拉 `config.json` 校验**，
> 所以这里标注为「公开常量」。公式本身和 GQA→显存的换算关系**是源码验证过的**
> （`cache_utils.py:432-441` 的分配形状 + `cache.py:29-53` 的推导规则）。
> 想完全自己复现，用等价的本地配置即可：`LlamaConfig` 默认值就是
> `hidden_size=4096, num_hidden_layers=32, num_attention_heads=32` → `head_dim=128`
> （`models/llama/configuration_llama.py:65-69`、`:87-90`），只是 `num_key_value_heads` 默认等于 32（MHA）。

---

## 十、`generation/continuous_batching/`：连续批处理为什么需要另一套 cache

前面所有的 Cache（`DynamicCache` / `StaticCache` / ...）都有一个共同假设：
**一次 generate 调用 = 一个 batch = 一条固定长度的序列集合**。
batch 里的序列长度不一样时，短的要陪长的等到最后，GPU 利用率大幅浪费。

**连续批处理（continuous batching）** 换了假设：**每个请求独立进出**，
每一步动态组一个 batch（新请求随时插入、完成的随时退出）。
代价是：**batch 维不再是固定的，一个 batch 里的序列长度也各不相同**——
扁平张量 `[B, H, S, D]` 装不下了。

### 10.1 目录结构（实测 `ls src/transformers/generation/continuous_batching/`）

| 文件 | 行数 | 职责 |
|---|---|---|
| `__init__.py` | 29 | 导出 `ContinuousBatchingManager`、`ContinuousMixin`、`PagedAttentionCache`、`FIFOScheduler`、`PrefillFirstScheduler`、`RequestState`、`RequestStatus`、`Scheduler` |
| `cache.py` | 814 | **`PagedAttentionCache`（:89）** + 内存规划 `PagedAttentionMemoryHandler`（:565） |
| `cache_manager.py` | 550 | 分页管理：`Block`（:37）、`BlockManager`（:58）、`CacheAllocator`（:297）、`FullAttentionCacheAllocator`（:365）、`SlidingAttentionCacheAllocator`（:456） |
| `continuous_api.py` | 1352 | 对外 API（`ContinuousBatchingManager` / `ContinuousMixin`），负责切到 `paged|` 后端（:660-683） |
| `scheduler.py` | 432 | 请求调度（FIFO / Prefill-first） |
| `requests.py` | 365 | `RequestState` / `RequestStatus`，逐请求的生命周期 |
| `model_runner.py` | 313 | 一次 forward 的驱动 |
| `input_outputs.py` | 822 | **`PagedAttentionArgs`（:31）**、静态缓冲区管理（`ContinuousBatchingIOs`，:70） |
| `offloading_manager.py` | 380 | CPU offloading（和 `Cache.offload` 是两个层面的东西） |
| `distributed.py` | 189 | TP 支持 |
| `initialization.py` | 290 | 初始化参数 |
| `cb_logits_processors.py` | 328 | 连续批处理下的 logits 处理（每请求独立采样参数） |
| `utils.py` | 219 | 小工具 |

### 10.2 `PagedAttentionCache`（`cache.py:89`）：三级层次

docstring（`:90-147`）自己讲得很清楚，翻成中文：

```
Page（页）   —— 最小单位。[num_heads, head_size]，即「一个 token、一层的 K 或 V」。
               全 full-attention 的模型，一个 token 要 2 × num_layers 个 page（K、V 各一半）
Block（块）  —— block_size 个 page 的集合，**分配单元**（按块分配/释放，不按页）。
               一个 block 只属于一个「层组」（layer group），组内所有层共享这一块
Cache tensor —— 物理载体。层组里有几层就有几个 tensor，
               形状 [num_blocks * block_size, num_heads, head_size]
```

**为什么按「层组」分配**（docstring 里的例子）：模型有 8 个 block，
一个组是 3 层 full attention，另一个组是 3 层 sliding attention：

```
分配 1 块给每个组：
  cache_tensor_0: ✖ ◉ □ □ □ □ □ □
  cache_tensor_1: ✖ ◉ □ □ □ □ □ □
  cache_tensor_2: ✖ ◉ □ □ □ □ □ □
继续生成，sliding 组已经填满窗口，只有 full 组需要新块：
  cache_tensor_0: ✖ ◉ ✖ □ □ □ □ □
  ...
```

**如果所有层在同一个组里会怎样**：sliding 组明明不需要新块，
也会被迫跟着分配一块——**纯浪费**。这就是分组的全部意义。

**它是怎么和 `attention` 后端接上的**（这是和前面所有 Cache 最不同的地方）：

| 步骤 | 代码 | 说明 |
|---|---|---|
| ① `ContinuousMixin` 把 `_attn_implementation` 改成 `paged|xxx` | `continuous_api.py:681-683` `model.set_attn_implementation(f"paged|{target_implem}")` | 没装 flash 就自动降级 `paged\|eager` / `paged\|sdpa` |
| ② 注册表里 `paged\|*` 指向专门的实现 | `modeling_utils.py:5161-5165` | `paged\|flash_attention_2/3/4` → `paged_attention_forward`；`paged\|sdpa` → `sdpa_attention_paged_forward`；`paged\|eager` → `eager_paged_attention_forward` |
| ③ **cache 不走 `past_key_values`，走 `cache=` kwarg** | `input_outputs.py:488` `cache=self.cache`；`:496` `use_cache=False` | 注释直白：`use_cache: Whether to use caching (always False in continuous batching as the cache is managed externally)` |
| ④ **后端自己调 `cache.update(...)`** | `eager_paged.py:37-42`、`sdpa_paged.py:37`、`flash_paged.py`（`flash_attn_with_kvcache`） | 这是和前 9 节最大的差别！ |

**两种读写的语义**（`cache.py:393-452` 的 `update` docstring，实测）：

- **full attention**：先写新 K/V 进 cache，再 `index_select` 把「整段历史 + 新 token」读出来。
- **sliding window**：**必须先读后写**——因为新 K/V 可能覆盖还没读的旧 K/V
  （chunked prefill 时才会出现）。实现上用 `sentinel_index` 占位 + `masked_scatter_` 补真值
  （`cache.py:441-446`，`sentinel_index` 定义在 `:248`）。
- **`layer_read_index.numel() == 0`**：整批都是非 chunked prefill，没有历史要读
  → 只写不读，直接返回输入 K/V（`cache.py:424-426`）。

**两种 kernel 路径**（`docs/source/en/paged_attention.md`）：

| 路径 | kernel | cache 交互 | 何时用 |
|---|---|---|---|
| varlen | `flash_attn_varlen_func` | **无**——cache 要手动读/写，文档明说 `This can become a bottleneck when sequence length grows large` | 大量请求处于 prefill |
| decode | `flash_attn_with_kvcache` | **由 kernel 内部用 `block_table` 直接寻址** | 整批都只有 1 个 query token |

`block_table` 的形状是 `(num_groups, batch_size, max_blocks_per_seq)`，
每个请求一行、元素是「第 i 个逻辑块在物理 cache 里的位置」，不足的用 `-1` 填充
（`flash_paged.py:36-40`）。

### 10.3 前缀共享（prefix caching）

因为分页之后「一块物理 cache = 一段确定的 token 序列」，
**多个请求如果 prompt 前缀相同，可以共享同一批 block**。实现分两层：

- `PagedAttentionCache.search_prefix_match`（`cache.py:470`）/ `mark_shareable_blocks_as_complete`（:496）/
  `fork_request`（:543）
- `BlockManager.compute_hash`（`cache_manager.py:279`）+ `hit` 计数 + `increase_ref_count` / `decrease_ref_count`
  （:187 / :194）——**引用计数**，最后一个引用消失才真正释放。
- `FullAttentionCacheAllocator`（:365）/ `SlidingAttentionCacheAllocator`（:456）分化出两套索引规则
  （滑动窗口层不需要「读整段历史」，`get_read_indices` 只给窗口内的位置）。

> **和前面 9 节的关系**：`PagedAttentionCache` **不是** `Cache` 的子类，
> `PagedAttentionCache` 里也**没有任何 `CacheLayerMixin`**（实测：`cache.py` 未 import `cache_utils`）。
> 它是**平行的一套体系**，因为它的核心矛盾（变长 batch + 显存碎片 + 前缀共享）
> 和「单序列怎么存 K/V」不是同一个问题。

---

## 十一、一句话总结

> **`cache_utils.py` 是两层结构：`CacheLayerMixin` 管「单层怎么存」，`Cache` 管「所有层怎么组织」。**
>
> - **Layer 一支**：`Dynamic*`（cat 生长）/ `Static*`（预分配 + 原地写，为 compile）/
>   `Quantized*`（省显存）/ `LinearAttention*`（无 seq 维度，给 SSM）/
>   `*IndexedLayer`（多一份 indexer key，给 DSA）。
>   选哪个由 config 的 `layer_types` 字符串经 `DYNAMIC/STATIC_LAYER_TYPE_MAPPING` 决定。
> - **Cache 一支**：`DynamicCache`（默认）/ `StaticCache` / `QuantizedCache` /
>   `EncoderDecoderCache`（组合两个）/ `MtpCache` 与 `DFlashCache`（只改 query 偏移）。
> - **契约**：`update()` 返回**完整**的 K/V，`layer_idx` 定位层，
>   位置信息由 cache 反向报给外面（`get_seq_length` / `get_query_offset` / `get_mask_sizes`）。
>   `cache_position` / `cache_kwargs` 在本版本**已删除**。
> - **和 2A2 的接口**是 `get_mask_sizes() → (kv_length, kv_offset)`；
>   **和 2A1 的接口**是「普通后端看不到 cache，paged 后端自己调 `cache.update`」。
> - **连续批处理**用另一套 `PagedAttentionCache`（block/页表/前缀共享），
>   不继承 `Cache`，靠 `paged|*` 后端 + `cache=` kwarg 接上。

---

## 验证清单

以下命令均在仓库根目录（`/Users/baai/Desktop/baai_proj/transformers`）实际跑过。

```bash
# ── 0. 规模 ────────────────────────────────────────────────────────────────
wc -l src/transformers/cache_utils.py
#   → 2131 src/transformers/cache_utils.py

# ── 1. 全部 23 个类的真实行号（本篇所有继承树都以此为准）──────────────────
grep -n "^class " src/transformers/cache_utils.py
#   → 27:CacheLayerMixin(ABC)          113:DynamicLayer          203:DynamicSlidingWindowLayer
#     319:DynamicIndexedLayer          398:StaticLayer           504:StaticSlidingWindowLayer
#     631:StaticIndexedLayer           703:QuantizedLayer        779:QuantoQuantizedLayer
#     834:HQQQuantizedLayer             891:LinearAttentionCacheLayerMixin(ABC)
#     1003:LinearAttentionLayer       1094:LinearAttentionAndFullAttentionLayer
#     1134:LinearAttentionAndSlidingWindowAttentionLayer
#     1166:LinearAttentionAndStaticFullAttentionLayer
#     1198:LinearAttentionAndStaticSlidingWindowAttentionLayer
#     1269:Cache                        1737:DynamicCache          1829:StaticCache
#     1884:QuantizedCache               1947:EncoderDecoderCache    2102:MtpCache
#     2114:DFlashCache

# ── 2. CacheLayerMixin 的 5 个抽象方法（真实方法名）──────────────────────
sed -n '52,73p' src/transformers/cache_utils.py
#   → :52 lazy_initialization(self, key_states, value_states)
#     :55 update(self, key_states, value_states, *args, **kwargs)
#     :60 get_mask_sizes(self, query_length)
#     :63 get_seq_length(self)
#     :66 get_max_length(self)        # docstring 说明 -1 = 无上限

# ── 3. LinearAttentionCacheLayerMixin 的 3 个抽象方法（注意没有 update）──
sed -n '915,928p' src/transformers/cache_utils.py
#   → :915 lazy_initialization(conv_states, recurrent_states, state_idx)
#     :923 update_conv_state(conv_states, state_idx)
#     :926 update_recurrent_state(recurrent_states, state_idx)

# ── 4. 类属性（决定 compile / early-init 能力）────────────────────────────
grep -n "is_sliding = \|is_compileable = \|supports_early_init = " src/transformers/cache_utils.py
#   → :30 is_compileable=False  :31 supports_early_init=True   (CacheLayerMixin)
#     :119 is_sliding=False                                    (DynamicLayer)
#     :209 is_sliding=True                                     (DynamicSlidingWindowLayer)
#     :408 is_compileable=True  :409 is_sliding=False           (StaticLayer)
#     :517 is_sliding=True                                     (StaticSlidingWindowLayer)
#     :895 is_compileable=True  :897 supports_early_init=False  (LinearAttentionCacheLayerMixin)
#     :1096 / :1136 is_compileable=False                       (两个动态 hybrid 层)

# ── 5. SWA 的裁剪（回答 1M2 的悬案）──────────────────────────────────────
sed -n '246,261p' src/transformers/cache_utils.py
#   → :253 self.keys   = full_key_states[:, :, -self.sliding_window + 1 :, :]
#     :261 return full_key_states, full_value_states   # 返回未裁剪的全量
sed -n '263,273p' src/transformers/cache_utils.py
#   → :267 kv_offset = max(self.cumulative_length - self.sliding_window + 1, 0)

# ── 6. Mistral 的链路：config → layer_types → DynamicSlidingWindowLayer ──
grep -n "sliding_window" src/transformers/models/mistral/configuration_mistral.py | head -1
#   → 80:    sliding_window: int | None = 4096
grep -n "DynamicCache(config" src/transformers/models/mistral/modeling_mistral.py
#   → 357:            past_key_values = DynamicCache(config=self.config)
sed -n '1706,1714p' src/transformers/cache_utils.py
#   → :1709  if getattr(config, "sliding_window", None) is not None:
#     :1710      layer_types = ["sliding_attention" for _ in range(config.num_hidden_layers)]

# ── 7. update() 契约（模型侧那一行）──────────────────────────────────────
grep -n "past_key_values.update" src/transformers/models/llama/modeling_llama.py
#   → 262:            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

# ── 8. cache 与 mask 的耦合 ──────────────────────────────────────────────
grep -n "past_key_values.get_seq_length()\|create_causal_mask\|DynamicCache(config" \
     src/transformers/models/llama/modeling_llama.py
#   → 28:from ...masking_utils import create_causal_mask
#     384:            past_key_values = DynamicCache(config=self.config)
#     387:            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
#     391:        causal_mask = create_causal_mask(
sed -n '828,833p' src/transformers/masking_utils.py
#   → :829 q_offset = past_key_values.get_query_offset(layer_idx)
#     :833 kv_length, kv_offset = past_key_values.get_mask_sizes(q_length, layer_idx)
sed -n '1507,1520p' src/transformers/masking_utils.py
#   → LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING（layer_type → mask 函数）

# ── 9. cache_position / cache_kwargs 在本版本已不存在 ────────────────────
grep -rn "cache_position" src/transformers/generation/utils.py
#   → 只有 :623-633 的 remote-code BC shim
grep -rn "cache_kwargs" src/transformers/ | grep -v "cross_attn_cache_kwargs\|dynamic_cache_kwargs"
#   → 无匹配（只剩各种同名的局部变量）

# ── 10. DSA：GLM 需要的 DynamicIndexedLayer ──────────────────────────────
grep -n "update_indexer\|DynamicIndexedLayer" src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py
#   → 174 / 237
grep -n "layer_types = \[" src/transformers/models/glm_moe_dsa/configuration_glm_moe_dsa.py
#   → 159:            self.layer_types = ["deepseek_sparse_attention"] * self.num_hidden_layers

# ── 11. 模型文件注册自定义 Layer（_layer_type 机制）──────────────────────
grep -rn "_layer_type = " src/transformers/models/
#   → deepseek_v4/modeling_deepseek_v4.py:185  "heavily_compressed_attention"
#     deepseek_v4/modeling_deepseek_v4.py:265  "compressed_sparse_attention"
#     minimax_m3_vl/modeling_minimax_m3_vl.py:54, :94  "minimax_m3_sparse"

# ── 12. 线性注意力：Mamba2 的 layer_types ───────────────────────────────
sed -n '106,107p' src/transformers/models/mamba2/configuration_mamba2.py
#   →     def layer_types(self):
#              return ["linear_attention"] * self.num_hidden_layers

# ── 13. 连续批处理：文件结构与 paged 后端接法 ────────────────────────────
ls src/transformers/generation/continuous_batching/
#   → __init__.py cache.py cache_manager.py cb_logits_processors.py continuous_api.py
#     distributed.py initialization.py input_outputs.py model_runner.py
#     offloading_manager.py requests.py scheduler.py utils.py
grep -n "cache.update" src/transformers/integrations/eager_paged.py src/transformers/integrations/sdpa_paged.py
#   → eager_paged.py:37 / sdpa_paged.py:37
grep -n "masked_scatter_\|layer_read_index.numel() == 0\|sentinel_index" \
     src/transformers/generation/continuous_batching/cache.py
#   → :248 sentinel_index 定义（= num_blocks*block_size+1）
#     :424 只写不读的分支（非 chunked prefill）
#     :442/:444/:446 masked_scatter_（滑动窗口「先读后写」）
grep -n "paged|" src/transformers/modeling_utils.py
#   → :5161-5165 _global_mapping 里的 paged|flash_attention_* / paged|sdpa / paged|eager
sed -n '681,683p' src/transformers/generation/continuous_batching/continuous_api.py
#   →         if "paged|" not in target_implem:
#                 model.set_attn_implementation(f"paged|{target_implem}")

# ── 13b. 其它被引用的行号 ────────────────────────────────────────────────
grep -n "self.layer_idx = layer_idx" src/transformers/models/llama/modeling_llama.py
#   → 223:        self.layer_idx = layer_idx
grep -n "block_table: (num_groups" src/transformers/integrations/flash_paged.py
#   → 36:        block_table: (num_groups, batch_size, max_blocks_per_seq), dtype int32. ...

# ── 14. 显存公式：源码里的分配形状 + 复算 ───────────────────────────────
sed -n '432,441p' src/transformers/cache_utils.py
#   → keys/values 分配形状 (batch_size, num_heads, max_cache_len, k/v_head_dim)
python3 -c "print(2*32*1*8*8192*128*2/1024**3)"    # → 1.0（GiB，Llama-3-8B GQA b=1 S=8192 fp16）
python3 -c "print(2*32*1*8*32768*128*2/1024**3)"   # → 4.0
python3 -c "print(2*32*32*8*4096*128*2/1024**3)"   # → 16.0
python3 -c "print(2*32*1*32*8192*128*2/1024**3)"   # → 4.0（假想 MHA，GQA 省 4 倍）
```

**没能现场验证的一处（如实标注）**：Llama-3-8B 的 `config.json` 数值
（32 层 / 8 KV 头 / head_dim 128）来自公开常识，**本机无网络**（`curl https://huggingface.co/...`
返回 `HTTP=000`，`WebFetch` 被域名策略拦截），未能拉取原文件核对。
公式、换算关系、以及「GQA 省 4 倍」的结论均来自上表第 14 条的源码与复算，不受影响。

---

## 导航

- 路线图：[00_roadmap.md](./00_roadmap.md)（本篇 = 进阶 · 架构 · 2A3）
- 上一篇：[2A2_masking_system.md](./2A2_masking_system.md)
- 下一篇：[2A4_rope_abstraction.md](./2A4_rope_abstraction.md)
- **建议紧接着做两件事**：
  1. 回 [1M2_mistral_sliding_window.md](./1M2_mistral_sliding_window.md) 看「对 KV Cache 的影响」那一节——
     它说「裁剪由 attention 或 mask 处理」，按本篇 4.2 节的实测应该改成
     「由 `DynamicSlidingWindowLayer` 自己裁到 `W-1`，mask 靠 `kv_offset` 对齐绝对位置」。
  2. 读完本篇可以直接进 **2M2（MLA）→ 2M7（GLM-5.3）**：
     MLA 砍的是「每 token 的 cache 宽度」，DSA 走的是 `DynamicIndexedLayer`，
     两者都只是往本篇这套机制里插一个 config 字符串。
