# KV Cache 体系精读：层（Layer）与容器（Cache）的双层结构

> 阅读目标：搞清楚 `src/transformers/cache_utils.py` 那 23 个类的**体系结构**——两层抽象为什么存在、每种子类解决什么问题、`update()` 的调用契约。
>
> **「注意力三件套」第三篇**：[2A1](./2A1_attention_backends.md) 讲**算什么**（后端），
> [2A2](./2A2_masking_system.md) 讲**在哪算**（mask），本篇讲**缓存什么**。
> 一句话定位：**2A2 是「哪些位置不许看」，本篇是「哪些 K/V 不用重算」**；两者靠 `past_key_values` 这一个参数耦合（第 7 节）。
>
> 源文件：`src/transformers/cache_utils.py`（**2131 行，23 个类**）、`src/transformers/generation/continuous_batching/`（13 文件，6083 行）
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

- [1M1_llama_structure.md](./1M1_llama_structure.md)：`past_key_values.update` 在 `modeling_llama.py:262`，本篇展开它。
- [1M2_mistral_sliding_window.md](./1M2_mistral_sliding_window.md)：SWA。「裁剪由 attention 或 mask 处理」的说法在本篇 4.2 被修正。
- [2A1](./2A1_attention_backends.md)、[2A2](./2A2_masking_system.md)、[2A4](./2A4_rope_abstraction.md)、[2A5_modular_code_generation.md](./2A5_modular_code_generation.md)（本篇写法对齐它）。

> **版本提示（重要）**：本仓库是 transformers v5 开发版。`cache_position` **已被删除**（只剩
> `generation/utils.py:623-633` 给 remote code 的兼容 shim），`cache_kwargs` 也已不存在（见 6.3 节）。
> **不要按老教程的记忆去读这个文件。**

---

## 一、为什么需要 KV cache

**首步（prefill）** 一次送进整个 prompt，**后续步（decode）** 每次只送 1 个新 token：

| | 首步 prefill | 后续步 decode |
|---|---|---|
| `input_ids` 长度 | `T`（prompt 长度） | `1` |
| 进入 attention 前 | `past_key_values` 为空（或已预填） | 已含 `T, T+1, ...` 个 token |
| mask 形状 | `[1, 1, T, T]` | `[1, 1, 1, T+1]` |
| Q 长度 / K、V 长度 | `T` / `T` | `1` / `T+1` |
| 计算量主项 | `2P·T` | `2P·1`（P = 参数量） |

无 cache 时第 t 步要把前 `t-1` 个 token 的 K/V 重算 `t-1` 次，而结果**必然逐比特相同**（K/V 只依赖该位置的
hidden state 与绝对位置，RoPE 角度只由 `position_ids` 决定，见 [2A4](./2A4_rope_abstraction.md)）。
Llama-3-8B（8.03B 参数）生成 1000 token：有 cache `2P·N` = **16.06 TFLOP**，无 cache `2P·N(N+1)/2` = **8038 TFLOP**，差 **500.5 ×**。
cache 消不掉的是注意力本身：`QK^T` 仍要拿这 1 个 query 与缓存里全部 `t` 个 key 做点积，是 `O(t·d)`
——所以长上下文的成本是 `O(t)` 显存 + `O(t²)` 计算，所有花活都在砍这两项之一：

| 手段 | 砍什么 | 在哪篇 |
|---|---|---|
| 滑动窗口（SWA） | cache 的 `t` → 常数 `W` | [1M2](./1M2_mistral_sliding_window.md)、本篇 4.2 |
| 稀疏 top-k（DSA） | 每步实际参与点积的 key 数 | 2M7、本篇 4.3 |
| MLA（低秩压缩） | 每 token 的 cache 宽度 | 2M2 |
| KV 量化 | 每元素的字节数 | 本篇 4.6 |
| Paged / 连续批处理 | 不是「砍」，是「别浪费」 | 本篇第 10 节 |

---

## 二、双层架构：Layer 管单层，Cache 管所有层

```
┌──────────────────────────────────────────────────────────────────────────┐
│  第一层：Layer —— 「单层」的 K/V 存储                                     │
│  cache_utils.py:27   CacheLayerMixin (ABC)                              │
│      keys / values 两个张量，怎么长、怎么裁                               │
│      接口：update() / get_seq_length() / get_max_length() / crop()       │
├──────────────────────────────────────────────────────────────────────────┤
│  第二层：Cache —— 「所有层」的集合 + 层间逻辑                             │
│  cache_utils.py:1269 Cache                                              │
│      self.layers 列表、按 layer_idx 分发、跨层统一的 get_mask_sizes() /    │
│      get_seq_length()、层的懒创建、offloading 调度、beam search 重排       │
└──────────────────────────────────────────────────────────────────────────┘
```

分两层的理由：① 各层策略可不同——混合模型（Gemma 2/3、Llama 4、GLM 的 sliding+full 交替）逐层选类
`layers = [DYNAMIC_LAYER_TYPE_MAPPING[layer_type](**layer_kwargs) for layer_type in layer_types]`（`cache_utils.py:1793`）；
② 容器统一对外——模型侧只调 `past_key_values.update(k, v, self.layer_idx)`，与 [2A1](./2A1_attention_backends.md) 的
`ALL_ATTENTION_FUNCTIONS` 同设计；③ 容器还管 offloading（:1324-1354）、`is_compileable`、`batch_size` 校验。

### 2.1 真实继承树

```
CacheLayerMixin (ABC)                                              cache_utils.py:27
  │  抽象：lazy_initialization / update / get_mask_sizes / get_seq_length / get_max_length
  │  类属性：is_compileable=False(:30) supports_early_init=True(:31) _layer_type(:34)
  ├─ DynamicLayer :113（cat 生长 :144-145；is_sliding=False :119）
  │    ├─ DynamicSlidingWindowLayer :203（is_sliding=True :209；只留 sliding_window-1 个历史 :252-254）
  │    ├─ DynamicIndexedLayer :319（多一份 indexer_keys :330、:338）
  │    └─ QuantizedLayer (ABC) :703（_quantize :768 / _dequantize :771；residual_length 机制）
  │         ├─ QuantoQuantizedLayer :779（optimum-quanto）  └─ HQQQuantizedLayer :834（hqq）
  └─ StaticLayer :398（is_compileable=True :408；「预分配 + 原地写」，为 torch.compile / CUDA graph 而生）
       ├─ StaticSlidingWindowLayer :504（is_sliding=True :517；max_cache_len = min(sliding_window, max_cache_len) :520）
       └─ StaticIndexedLayer :631（静态 indexer_keys :643、:664）

LinearAttentionCacheLayerMixin (ABC) :891（is_compileable=True :895；supports_early_init=False :897）
  │  抽象：lazy_initialization / update_conv_state / update_recurrent_state（无 update()，无 seq_len 维度）
  └─ LinearAttentionLayer :1003（conv/recurrent 两个 dict :902-903）
       ├─ LinearAttentionAndFullAttentionLayer :1094        ├─ ...AndSlidingWindowAttentionLayer :1134
       ├─ ...AndStaticFullAttentionLayer :1166              └─ ...AndStaticSlidingWindowAttentionLayer :1198

Cache :1269
  │  self.layers: list[CacheLayerMixin | LinearAttentionCacheLayerMixin]（:1306）
  │  self.layer_class_to_replicate（懒创建，:1307）；self.offloading / prefetch_stream（:1309-1311）
  ├─ DynamicCache :1737（传 config → 逐层按 layer_types 建 :1793；不传 → 懒创建 :1374-1376）
  │    ├─ MtpCache :2102      └─ DFlashCache :2114
  ├─ StaticCache :1829（要求 config + max_cache_len；别名 SlidingWindowCache = StaticCache :2099）
  ├─ QuantizedCache :1884（只支持全 full_attention，:1934-1939 报错）
  └─ EncoderDecoderCache :1947（组合而非继承：self_attention_cache + cross_attention_cache）
```

> **「菱形继承」例外**：最后 4 个线性注意力类用**多重继承**（`LinearAttentionLayer` + 对应注意力层），同时维护
> 线性注意力（Mamba/SSM）递归状态与普通注意力层 K/V。它们手写 `lazy_initialization`（:1102、:1142、:1171、:1203），
> 靠「位置参数个数」判定调用给哪一半：`if len(args) == 2 and len(kwargs) == 0: DynamicLayer.lazy_initialization(self, *args)`。

`EncoderDecoderCache` 用「组合」而非「第四种 Layer」，因为它要装**两套增长规律完全不同**的 attention：
self-attention（按位置增长）与 cross-attention（对 encoder 输出，**不增长**），硬塞进一个 Layer 列表会互相污染。

---

## 三、`CacheLayerMixin`：单层的抽象契约

### 3.1 必须实现的 5 个

| 抽象方法 | 行号 | 签名 | 契约 |
|---|---|---|---|
| `lazy_initialization` | :52-53 | `(key_states, value_states) -> None` | 「我不知道 dtype / device / num_heads，你第一次 update 时告诉我」——**这就是 Layer 创建时不分配显存的原因** |
| `update` | :55-58 | `(key_states, value_states, *args, **kwargs) -> (Tensor, Tensor)` | 写入 + **返回给 attention 用的完整 K/V**（契约见第 6 节） |
| `get_mask_sizes` | :60-61 | `(query_length: int) -> (kv_length, kv_offset)` | **这是 cache 和 mask 的唯一接口** |
| `get_seq_length` | :63-64 | `() -> int` | 已经见过多少 token。**滑动窗口层返回「真实总数」而不是「当前存了几个」**（:277） |
| `get_max_length` | :66-73 | `() -> int` | 上限。**`-1` 表示「无上限或没有序列维度」**（动态层、线性注意力层都返回 `-1`） |

`get_max_cache_shape` 已改名：`:106-110` 是兼容 shim，打 deprecation 警告（"will be removed in version 5.16"），内容就是 `return self.get_max_length()`。

### 3.2 有默认实现的辅助方法与类属性

`offload` / `prefetch`（:75 / :81，层数据搬到 CPU / 搬回 device，即 **KV cache offloading**）；
`reset`（:87，原地清零，顺手重置 `cumulative_length` :92-98）；`reorder_cache`（:100，beam search 按 `beam_idx` 重排 batch 维）。

```python
is_compileable = False        # :30   → DynamicLayer 不改，StaticLayer 改成 True（:408）
supports_early_init = True    # :31   → LinearAttentionCacheLayerMixin 改成 False（:897）
```

`Cache.is_compileable`（:1651-1657）就是 `all(layer.is_compileable for layer in self.layers)`。

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

模型用 `_layer_type = "..."` 就能把自己注册进映射表，**在被 import 时自动生效**，`cache_utils.py` 不需要 import 任何模型文件。
真实用例（实测）：`models/deepseek_v4/modeling_deepseek_v4.py:185`、`:265`（`"heavily_compressed_attention"`、
`"compressed_sparse_attention"`）、`models/minimax_m3_vl/modeling_minimax_m3_vl.py:54`、`:94`（`"minimax_m3_sparse"`）。

---

## 四、Layer 子类逐个拆

### 4.1 `DynamicLayer`（:113）—— 最常用，也是默认

```python
# cache_utils.py:144-146
self.keys = torch.cat([self.keys, key_states], dim=-2)
self.values = torch.cat([self.values, value_states], dim=-2)
return self.keys, self.values
```

形状 `[B, num_heads, seq_len, head_dim]`，`seq_len` 一直长；`get_max_length()` 返回 `-1`（:160-162）；
`get_mask_sizes()` 返回 `(self.get_seq_length() + query_length, 0)`（:148-152）——**kv_offset 恒为 0**。
每步 `torch.cat` 新建更大张量并拷贝，地址在变，所以 `is_compileable = False`。`generate` 不指定 `cache_implementation` 时就是它。

### 4.2 `DynamicSlidingWindowLayer`（:203）—— 回答 1M2 的悬案

[1M2](./1M2_mistral_sliding_window.md) 说 SWA 的裁剪由 attention 实现或 mask 兜底。
**实测结论：两者都不对。裁剪是 cache 层自己做的。**

```python
# cache_utils.py:249-261
full_key_states = torch.cat([self.keys, key_states], dim=-2)     # 先把「已有的 + 新的」拼起来
full_value_states = torch.cat([self.values, value_states], dim=-2)
if not self.record_past:                                          # 只缓存最后 sliding_window - 1 个
    self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
    self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
return full_key_states, full_value_states                        # 返回的是「完整的」，不是裁剪后的！
```

- **为什么是 `W-1` 而不是 `W`**：mask 的定义是 `kv_idx > q_idx - W`（窗口含当前位置，共 W 个）。缓存的 W-1 个是
  「历史」，本步新算的 1 个是「当前」——**W-1 + 1 = W**。
- **为什么返回完整 K/V**：`full` 已含本次需要的「W-1 个历史 + 本次的 K/V」；返回裁剪后的 `self.keys` 反而会丢掉最近的位置（同一件事在 `StaticSlidingWindowLayer` 注释：:601-602 也说了）。

mask 靠 `get_mask_sizes`（:263-273）对齐绝对位置：

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
| 1（prefill 4 token） | `[]` | `[k1 k2 k3 k4]`（长 4） | `[k2 k3 k4]`（长 3） | 4 | `(4, 1)` |
| 2（decode） | `[k2 k3 k4]` | `[k2 k3 k4 k5]`（长 4） | `[k3 k4 k5]` | 5 | `(4, 2)` |
| 3（decode） | `[k3 k4 k5]` | `[k3 k4 k5 k6]`（长 4） | `[k4 k5 k6]` | 6 | `(4, 3)` |

三条结论：① **内部存储恒定** = `W-1` 个；② **返回的 K/V 恒定** = `W` 个（填满后），attention 端开销也是常数；
③ **`get_seq_length()` 返回 `cumulative_length`（真实总数，:275-277），不是 `keys.shape[-2]`**——`LlamaModel.forward`
用它算 `position_ids`（`modeling_llama.py:387`），位置必须绝对。

**Mistral 就是这样被接上的**（模型侧不写任何代码）：

```
configuration_mistral.py:80   sliding_window = 4096
  ↓ modeling_mistral.py:357   past_key_values = DynamicCache(config=self.config)
  ↓ cache_utils.py:1709-1710  layer_types = ["sliding_attention"] * num_hidden_layers（mistral 无显式 layer_types）
  ↓ cache_utils.py:1226       DYNAMIC_LAYER_TYPE_MAPPING["sliding_attention"] = DynamicSlidingWindowLayer
  ↓ cache_utils.py:1793       layers = [DynamicSlidingWindowLayer(sliding_window=4096) for _ in ...]
```

**所以 Mistral 7B 的 KV cache 上限是常数 `32 × 2 × 8 × 4095 × 128 × 2B ≈ 512 MiB`，与生成长度无关。**
（`record_past` 开关给 assistant / speculative decoding 用：需要「回滚」时先全存着，等 `crop` 再裁，见 :218-223 和 :283-316。）

### 4.3 `DynamicIndexedLayer`（:319）—— 2M7 GLM 的 DSA 需要的东西

DSA（DeepSeek Sparse Attention）在标准 K/V 之外维护一份**单头、低维的 indexer key** 给 query 打分挑 top-k。
docstring（`cache_utils.py:319-326`）：main K/V 是 `[B, num_heads, seq_len, head_dim]`（继承而来），
indexer key cache 是 `[B, seq_len, index_head_dim]`（**3D、单头**）。

| 成员 | 行号 | 说明 |
|---|---|---|
| `indexer_keys` | :330 | 单独一份张量，**3D 而非 4D**（单头） |
| `lazy_initialization_indexer` | :333 | 独立的懒初始化 |
| `update_indexer` | :338-351 | 和 `update` 一样是 cat，但**沿 `dim=1` 而非 `dim=-2`**（去掉了 heads 维） |
| `offload` / `prefetch` / `reset` / `reorder_cache` / `crop` | :353 / :358 / :363 / :368 / :373 | 全部 `super()` 之后再处理 `indexer_keys` |

调用点（实测）：`models/glm_moe_dsa/modeling_glm_moe_dsa.py:237` `k = past_key_values.update_indexer(k, self.layer_idx)`；
还有 `modular_glm_moe_dsa.py:176`、`models/deepseek_v32/modeling_deepseek_v32.py:238`、`modeling_glm5_next.py:811`、`modular_qwen4_exp.py:411`。
**GLM-5.3 怎么接上的**（`configuration_glm_moe_dsa.py:158-159`）：`self.layer_types = ["deepseek_sparse_attention"] * self.num_hidden_layers`
→ `cache_utils.py:1236` `DYNAMIC_LAYER_TYPE_MAPPING["deepseek_sparse_attention"] = DynamicIndexedLayer`。
**模型代码里没有任何 cache 的存储逻辑**，只是 config 里那行字符串 + 映射表里那一行。

### 4.4 `StaticLayer`（:398）—— 预分配，为 compile 而生

```python
# cache_utils.py:432-441（lazy_initialization）
self.keys = torch.zeros((self.batch_size, self.num_heads, self.max_cache_len, self.k_head_dim),
                        dtype=self.dtype, device=self.device)

# cache_utils.py:474-481（update 的核心：原地写）
kv_length = key_states.shape[-2]
cache_position = torch.arange(kv_length, device=self.device) + self.cumulative_length
self.cumulative_length.add_(kv_length)          # 原地 add_，不能重新赋值
self.keys.index_copy_(2, cache_position, key_states)
self.values.index_copy_(2, cache_position, value_states)
```

**为 compile 付出的三处代价**（都是真实注释）：① `cumulative_length` 必须是 **tensor** 而非 int（:414-415，
`# Very important that it's a tensor here, to avoid recompiling when we update it and use it to create positions`）；
② `mark_static_address`（:448-451，`# Without this, we cannot use cudagraphs!!`）；③ `get_mask_sizes` 恒返回
`self.max_cache_len`（:489-493），**不管实际填了多少**——形状必须静态，多出的未填位置全是 `0`，靠 mask 屏蔽。
**静态 cache 是用算力换「可编译」**（`kv_cache.md`：「a lot of tokens will actually be masked」）。

### 4.5 `StaticSlidingWindowLayer`（:504）/ `StaticIndexedLayer`（:631）

`StaticSlidingWindowLayer` 分配时取 `min(sliding_window, max_cache_len)`（:520），填满后用 `roll` 实现「整体左移一位」
（:551-563）：`new_keys = self.keys.roll(-1, dims=-2)` → 最新一个写到末尾 → `self.keys.copy_(new_keys)`
（`copy_` 而非赋值，保住静态地址）。注释说这是绕 dynamo 的 bug（`# dynamo is currently bugged when doing it - see pytorch/pytorch#159855`）
——**这就是 `cache_explanation.md` 那句「existing tokens are shifted out of the cache」的实现。**

`StaticIndexedLayer` 是 `DynamicIndexedLayer` 的静态版，`indexer_cumulative_length` 是**独立的**一个 tensor
（:645-647 注释：indexer 的 update 在主 K/V 的 update **之后**跑，不能复用一个计数器）。同样 `mark_static_address`（:659-661）。

### 4.6 量化三兄弟：`QuantizedLayer`（:703）/ `QuantoQuantizedLayer`（:779）/ `HQQQuantizedLayer`（:834）

按 [KIVI 论文](https://huggingface.co/papers/2402.02750) 的思路（docstring :705 明确引用）。**双层存储**：
`self.keys` / `self.values` 存原始精度、容量 `residual_length`（默认 128，:728）；
`self._quantized_keys` / `self._quantized_values` 存量化值、容量无限（每通道组量化，:758-759）。

**工作流**（`update` :731-766）：新 K/V 先放原始精度区，与上一步解量化出来的历史拼在一起返回；达到 `residual_length`
就把「全部历史」重新 `_quantize` 塞进量化区，原始精度区清空（:760-761）。**返回值永远是「解量化后的历史 + 原始精度区 + 本次」**
（:755-756），所以**量化对 attention 透明**。

| | `QuantoQuantizedLayer` | `HQQQuantizedLayer` |
|---|---|---|
| 依赖 | `optimum-quanto >= 0.2.5`（:802） | `hqq`（:851） |
| `nbits` | `[2, 4]`（:809-810） | `[1, 2, 3, 4, 8]`（:857-860） |
| `axis` | `[0, -1]`（:812-818） | `[0, 1]`（:862-866） |
| 量化实现 | `MaxOptimizer` + `quantize_weight`（:823-828） | `HQQQuantizer.quantize`（:870-883） |

`QuantizedCache` 只支持**全部层都是 full attention** 的模型——`:1934-1939` 检查 `layer_types - {"full_attention"}` 是否为空，
不为空直接 `raise ValueError`。（滑动窗口 + 量化目前不支持。）

### 4.7 线性注意力系列：`LinearAttentionCacheLayerMixin`（:891）+ `LinearAttentionLayer`（:1003）

给 SSM / 线性注意力用（Mamba 2、Jamba、Zamba、Falcon-H1、LFM2、Qwen3-Next 等）。**根本区别：没有序列长度维度**
——普通层缓存 `[B, H, S, D]`（S 随生成增长），线性注意力层缓存一个固定形状的递归状态。

| 抽象方法 | 行号 |
|---|---|
| `lazy_initialization(conv_states=None, recurrent_states=None, state_idx=0)` | :915-921 |
| `update_conv_state(conv_states, state_idx=0)` | :923-924 |
| `update_recurrent_state(recurrent_states, state_idx=0)` | :926-927 |

**它根本没有 `update` / `get_seq_length` / `get_mask_sizes`**：① 它**不是** `CacheLayerMixin` 的子类（是独立的 `ABC`）；
② `Cache.get_seq_length` / `get_mask_sizes`（:1498-1514 / :1574-1588）遇到它会**自动跳到第一个真正的注意力层**去取序列长度。
状态装在两个 dict 里（:902-903），因为一层里可能有**多个**状态（`number_of_states` 由 `number_of_conv_states` 配置决定，
`get_layer_types_and_kwargs` :1731-1732）：

```python
self.conv_states: dict[int, torch.Tensor | None] = dict.fromkeys(range(number_of_states))
self.recurrent_states: dict[int, torch.Tensor | None] = dict.fromkeys(range(number_of_states))
```

实测用 `number_of_conv_states` 的模型：`models/inkling/configuration_inkling.py:176`（= 4）、
`models/qwen4_exp/configuration_qwen4_exp.py:172`（= 3 或 1）。**`moe` / `mlp` 层也被映射成 `LinearAttentionLayer`**
（:1242-1243 / :1264-1265）：它们是**空占位层**，但这样能让「按层取序列长度」的逻辑跳过它们。

### 4.8 `layer_type` → Layer 的映射表

`cache_utils.py:1223-1244`（动态）和 `:1246-1266`（静态）。**这是「config 里的字符串」到「类」的唯一入口**：

| `layer_type` 字符串 | 动态映射 | 静态映射 | 典型模型 |
|---|---|---|---|
| `full_attention` | `DynamicLayer` | `StaticLayer` | 几乎所有模型 |
| `sliding_attention` | `DynamicSlidingWindowLayer` | `StaticSlidingWindowLayer` | Mistral、Gemma 2/3 |
| `chunked_attention` | `DynamicSlidingWindowLayer` | `StaticSlidingWindowLayer` | Llama 4 |
| `conv` / `linear_attention` | `LinearAttentionLayer` | `LinearAttentionLayer` | Mamba 2（`configuration_mamba2.py:106-107`）、LFM2 |
| `hybrid` / `hybrid_sliding` | `LinearAttentionAndFullAttentionLayer` / `...SlidingWindowAttentionLayer` | `LinearAttentionAndStatic*Layer` | Zamba、Falcon-H1、Jamba |
| `deepseek_sparse_attention` / `qwen_sparse_attention` | `DynamicIndexedLayer` | `StaticIndexedLayer` | **GLM-5.3（2M7）**、DeepSeek V3.2 |
| `moe` / `mlp` | `LinearAttentionLayer`（空占位） | 同左 | MoE 模型 |
| `heavily_compressed_attention` / `compressed_sparse_attention` | 由模型文件注册 | 同左 | DeepSeek V4（`modeling_deepseek_v4.py:185`、`:265`） |

`sliding_attention` 和 `chunked_attention` 映射到同一个类（注释 :1225：`# From a cache point of view, sliding and chunked are the same in how they should behave, only the mask differs`）
——**cache 只管存多少，不管能不能看**，后者是 2A2 的事。

---

## 五、Cache 容器子类

`Cache`（:1269）是基类，自己不存 K/V，只做三件事：① **按 `layer_idx` 分发**（`update` :1356-1388）；
② **跨层聚合接口**（`get_seq_length` / `get_mask_sizes` / `get_max_length`）；③ **调度层间事务**
（offloading :1324-1354、`early_initialization` :1455-1490）。关键成员（实测行号）：

| 成员 | 行号 | 说明 |
|---|---|---|
| `self.layers` | :1306 | Layer 列表。**这是整个 cache 的全部存储** |
| `self.layer_class_to_replicate` | :1307 | 懒创建用：与 `layers` **二选一**，都传或都不传会报错（:1296-1305） |
| `update(key, value, layer_idx)` | :1356-1388 | 分发给 `self.layers[layer_idx].update(...)` |
| `update_conv_state` / `update_recurrent_state` | :1390 / :1412 | 线性注意力专用，内部 `isinstance` 校验 |
| `update_indexer` | :1434-1453 | DSA 专用；层不是 Indexed 层会给出清楚报错 |
| `early_initialization` | :1455-1490 | 见 6.4 |
| `get_seq_length(layer_idx=0)` | :1492-1514 | **自动跳过线性注意力层** |
| `get_max_length(layer_idx=None)` | :1516-1530 | 不传参数时返回**所有层的最大值** |
| `get_mask_sizes(query_length, layer_idx)` | :1562-1590 | 转给对应 Layer |
| `get_query_offset(layer_idx=0)` | :1592-1597 | 默认就是 `get_seq_length()`；**MTP / DFlash 会覆写它** |
| `crop` / `reset` / `reorder_cache` | :1609 / :1599 / :1604 | 逐层遍历 |
| `is_compileable` / `is_sliding` / `is_linear` / `batch_size` | :1651 / :1665 / :1670 / :1638 | 逐层聚合出来 |

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

`get_layer_types_and_kwargs`（:1701-1734）是「config → layer_types + 初始化参数」的翻译函数，也是「Mistral 不改一行模型代码
就拿到 SWA cache」的答案（见 4.2）：

```python
if layer_types is None:
    if getattr(config, "sliding_window", None) is not None:
        layer_types = ["sliding_attention"] * config.num_hidden_layers      # :1709-1710
    elif getattr(config, "attention_chunk_size", None) is not None:
        layer_types = ["chunked_attention"] * config.num_hidden_layers      # :1711-1712
    else:
        layer_types = ["full_attention"] * config.num_hidden_layers         # :1713-1714
```

另外 :1717-1719 会砍掉 `num_kv_shared_layers`（Gemma 3n 那种共享 KV 的层不需要 cache）。

### 5.2 `StaticCache`（:1829）/ `QuantizedCache`（:1884）

`StaticCache` 强制要求 `config` + `max_cache_len`，构造时把 `max_cache_len` 塞进 `layer_kwargs` 一起分发（:1877-1881）。
**`max_cache_len` 必须由用户给出**——这就是 `GenerationConfig.max_cache_len` 存在的原因
（`generation/configuration_utils.py:691-694` 校验「只有 static cache 才能用这个参数」）。
`QuantizedCache` 不逐层分发，而是**直接建 `num_hidden_layers` 个同构层**（:1940-1943），前面做「只准 full attention」校验（:1934-1939）。

### 5.3 `EncoderDecoderCache`（:1947）—— 呼应 2M4 T5

```python
self.self_attention_cache = caches[0]     # :2000   解码器自注意力 cache，随生成增长（这是"记忆"）
self.cross_attention_cache = caches[1]    # :2001   对 encoder 输出，只算一次，之后整段复用
self.is_updated = {}                      # :2006-2008  逐层记录 cross cache 是否已填（get_seq_length > 0）
```

| 方法 | 行号 | 转发给 |
|---|---|---|
| `get_seq_length` / `get_max_length` / `get_mask_sizes` | :2028 / :2032 / :2077 | `self_attention_cache` |
| `crop` | :2057 | **只裁 self**（cross 不能裁，见 :2062-2063）——cross 的 K/V 是 encoder 输出的函数，和生成进度无关 |
| `batch_repeat_interleave` / `batch_select_indices` | :2065 / :2071 | **两个都做** |

`__iter__`（:2010-2013）产出 6 元组 `(self_k, self_v, self_sliding, cross_k, cross_v, cross_sliding)`——给 DDP 序列化用。

### 5.4 `MtpCache`（:2102）/ `DFlashCache`（:2114）—— 实测确认它们是什么

两个都是 `DynamicCache` 的子类，**覆写点都只有一个：`get_query_offset` 和 `get_mask_sizes`**。
它们改变的不是「存什么」，而是「**query 的位置从哪里开始算**」——是投机解码的**偏移量适配器**。

```python
# cache_utils.py:2103-2106  MtpCache：第 k 层 MTP 的 query 比主模型超前 k+1 个位置
def get_query_offset(self, layer_idx: int = 0):
    # Queries of MTP depth k run k+1 tokens ahead of the main_model, i.e. they have an offset of k+1
    mtp_offset = layer_idx + 1
    return super().get_query_offset(layer_idx) + mtp_offset

# cache_utils.py:2114-2131  DFlashCache：一个 forward 里 Q 与 K/V 长度不同，按上一轮被接受的 token 数平移
def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs); self._previous_number_of_accepted_tokens = 0
def set_previous_accepted_tokens(self, number_of_accepted_tokens: int): ...   # :2119
def get_query_offset(self, layer_idx: int = 0):
    return super().get_query_offset(layer_idx) + self._previous_number_of_accepted_tokens
```

使用点：`generation/candidate_generator.py:1423` `class MTPCandidateGenerator`，`:1451` 建
`MtpCache(config=main_model.config.get_mtp_config())` 并立刻 `activate_past_recording()`（投机解码要能回滚）；
`generation/candidate_generator.py:1564` `class DFlashTokenCandidateGenerator`，`:1611` 建
`DFlashCache(config=self.assistant_model.config)`。DFlash 是**扩散式**草稿（`block_size` / `mask_token_id` /
`target_layer_ids`，见 `candidate_generator.py:1605-1610`）。

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

`models/llama/modeling_llama.py:262`（实测），**在 attention 模块内部、RoPE 之后、后端调用之前**（顺序见 `:255-271`）：

```python
if past_key_values is not None:
    key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
```

```
hidden_states → q/k/v_proj → apply_rotary_pos_emb(q, k) → past_key_values.update(k, v, layer_idx)  ★
              → ALL_ATTENTION_FUNCTIONS.get_interface(...)(q, k, v, ...) → o_proj
```

**顺序为什么必须是「先 RoPE 再 cache」**：cache 里存的必须是**旋转后的 K**，否则每步取出来还得按历史位置重转一遍
（除非用相对位置编码，那是 2A4 的话题）。**三条契约**：① `key_states` / `value_states` 是**本次新算的**，
形状 `[B, H_kv, S_q, D]`；② **返回值是「完整」的 K/V**（历史 + 本次），形状 `[B, H_kv, S_kv, D]`，`S_kv ≥ S_q`；
③ **必须原地写回语义**——模型侧**不能**把返回值当成「新的 cache」再存回去（对 `StaticLayer`，返回值甚至**就是 `self.keys` 本身**，`:487`）。

| Layer | 返回什么 | 为什么 |
|---|---|---|
| `DynamicLayer` | `self.keys, self.values` | 就是全部 |
| `DynamicSlidingWindowLayer` | **`full_key_states`（未裁剪的）** | 裁剪后的 `self.keys` 少了最新位置；mask 用 `kv_offset` 对齐绝对位置 |
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

`layer_idx` 来自 `nn.Module` 自己的 `self.layer_idx`（`modeling_llama.py:223`，DecoderLayer 构造时逐层传入），
**模型侧不需要知道 cache 里有多少层**。① 懒创建是 `DynamicCache()` 不带 config 时的工作模式（`Cache.__len__` :1316-1322 注释
`# Note: for DynamicCache, layers are initialized lazily, so this will not be accurate before the first forward through all the layers`）；
② offloading 用 `prefetch(layer_idx + 1)`——**在算第 `layer_idx` 层时，已经在用后台 stream 把第 `layer_idx + 1` 层拉回 device 了**
（:1324-1345）；`only_non_sliding=True` 时滑动窗口层不参与 offload（:1332-1335：滑动层本来就小，搬来搬去反而慢）。

### 6.3 `cache_position` / `cache_kwargs`：**在这个版本里已经没有了**

老教程里模型接收 `cache_position`（`arange`）和 `cache_kwargs`（dict）。**实测：本仓库已不存在。**
`grep -rn "cache_position" src/transformers/generation/utils.py` 只剩 `:623-633` 的 remote-code BC shim
（给 `trust_remote_code` 模型现场造一个并附 FutureWarning）；`cache_utils.py` 里只剩 `StaticLayer.update`
内部**局部变量** `cache_position`（:474、:583、:680）——那是「写到缓冲区哪几个槽位」的下标，**和模型接口无关**。

| 老 `cache_position` 的用途 | 现在的替代 | 代码位置 |
|---|---|---|
| 算 `position_ids`（RoPE 用） | `past_key_values.get_seq_length()` | `modeling_llama.py:387` |
| 算 mask 的 q 偏移 | `past_key_values.get_query_offset(layer_idx)` | `masking_utils.py:829` |
| 算 mask 的 kv 长度/偏移 | `past_key_values.get_mask_sizes(q_length, layer_idx)` | `masking_utils.py:833` |
| `cache_kwargs`（给 cache 传额外参数） | cache 构造时就知道一切（`config` / `sliding_window` / `max_cache_len`） | `cache_utils.py:1793`、`:1880` |

**设计意图**：位置信息不再由外面推给 cache，而是 **cache 自己算完报给外面**——这样「缓存的裁剪策略」与
「位置/mask 的正确性」被绑在一起，`DynamicSlidingWindowLayer` 那种「内部存 3 个、绝对位置已到第 100000 个」
的情况才不可能算错。

### 6.4 懒初始化与 `early_initialization`

懒初始化（`StaticLayer.lazy_initialization` docstring，:418-425）：dtype / device / num_heads 是「第一次 forward 时」
才知道的（激活 dtype、TP 后每张卡的 `num_heads`、甚至 MLA 模型 `v_head_dim` 与 `k_head_dim` 不同），
**所以建 cache 时不分配显存**。**例外是 `early_initialization`**（`Cache.early_initialization` :1455-1490）：
`torch.export` / chunked prefill 要求**编译前**形状固定，用**假张量**（:1488-1490，注释 `fake tensor`）：

```python
fake_kv_tensor = torch.zeros((batch_size, layer_num_heads, 0, layer_head_dim), dtype=dtype, device=device)
layer.lazy_initialization(fake_kv_tensor, fake_kv_tensor)
```

**在 seq_len 维上长度为 0**——不占显存，只传递形状 / dtype / device。`supports_early_init`（`:31`）的意义就在这：
`LinearAttentionCacheLayerMixin` 把它设成 `False`（:897），因为线性注意力层的状态形状是自己算出来的，
没有 `[.., S, D]` 这一维可供 fake；`early_initialization` 会 `continue` 跳过它们（:1483-1484）。

---

## 七、cache 与 mask 的耦合（→ 2A2）

`LlamaModel.forward`（`modeling_llama.py:383-399`，实测）：

```python
if use_cache and past_key_values is None:
    past_key_values = DynamicCache(config=self.config)          # :384
if position_ids is None:
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0   # :387
    position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
    position_ids = position_ids.unsqueeze(0)                    # :388-389
causal_mask = create_causal_mask(                               # :391
    config=self.config, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
    past_key_values=past_key_values, position_ids=position_ids)  # ★ 传给 mask
```

**两处耦合**：`position_ids` 用 `get_seq_length()`；mask 用 `past_key_values` 本身。
`create_causal_mask`（`masking_utils.py:864`）转交给 `_preprocess_mask_arguments`（:760），**核心两行**（`:828-833`）：

```python
if past_key_values is not None:
    q_offset = past_key_values.get_query_offset(layer_idx)
    kv_length, kv_offset = past_key_values.get_mask_sizes(q_length, layer_idx)
```

| 用途 | 说明 |
|---|---|
| **决定 mask 形状** | `kv_length` → mask 最后两维是 `(q_length, kv_length)`，即「覆盖缓存里的历史」 |
| **对齐绝对位置** | `q_offset` / `kv_offset` 是**绝对位置**起点；mask 内部比较 `kv_idx > q_idx - W` 这种绝对位置，两端都得加 offset |
| **省掉不必要的计算** | `allow_is_causal_skip`（:953-955）：mask 恰好是规整 causal（`q_offset == kv_offset` 且长度对得上）时**根本不生成 mask 矩阵**，交给后端 causally 算（`is_causal=True`） |

`q_offset` 的默认值是 `get_seq_length()`（`Cache.get_query_offset` :1592-1597）——所以普通 `DynamicLayer` 上
`q_offset == kv_offset == past_len` 恒成立，「规整 causal」优化才能命中；`MtpCache` / `DFlashCache` 覆写它
就是为了**打破这个相等**（见 5.4）。

### 7.3 mask 是按「层类型」而不是「层号」建的

`create_causal_mask` 里（`masking_utils.py:929-933`）：

```python
if layer_idx is None:
    if hasattr(past_key_values, "is_sliding") and False in past_key_values.is_sliding:
        layer_idx = past_key_values.is_sliding.index(False)     # 找一个 full attention 层当代表
    else:
        layer_idx = 0
```

**同类型的层，`get_mask_sizes` 返回值必然相同**，所以混合模型只需按 `layer_types` 的**每种类型各建一张 mask**。
这个动作在 `generate` 里由 `create_masks_for_generate`（`masking_utils.py:1523`）统一做，映射表是
`LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING`（:1507-1520）：

| `layer_type` | mask 函数 |
|---|---|
| `full_attention` | `create_causal_mask` |
| `sliding_attention` / `chunked_attention` | `create_sliding_window_causal_mask` / `create_chunked_causal_mask` |
| `deepseek_sparse_attention` / `qwen_sparse_attention` / `minimax_m3_sparse` | `create_causal_mask` |
| `compressed_sparse_attention` / `heavily_compressed_attention` | `create_sliding_window_causal_mask` |
| `linear_attention` / `conv` | `create_recurrent_attention_mask` |
| `hybrid` / `hybrid_sliding` | **dict**：`{full: create_causal_mask, linear: create_recurrent_attention_mask}` |

模型侧拿到的是**一个 dict**（`masking_utils.py:1582-1592` 的 `causal_masks`），逐层取，例如 GLM
（`modeling_glm_moe_dsa.py:743`）：`attention_mask=causal_mask_mapping[self.config.layer_types[i]]`。

> **一句话**：**cache 的分层（`layer_types`）决定了 mask 的分层。** 这就是 2A2 和 2A3 的接口。

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

**读这张图要抓住的三个「长度」**：

| 概念 | 出处 | 含义 |
|---|---|---|
| `get_seq_length()` | `Cache` :1492 | **真实已经见过多少 token**。用来算 `position_ids`。滑动窗口层返回真实总数（:277） |
| `keys.shape[-2]` | Layer 内部 | **实际存了多少**。滑动窗口层 = `W-1`，静态层 = `max_cache_len` |
| `get_mask_sizes()` 的 `kv_length` | Layer :60 | **这次要传给 attention 多长**。可以和上两个都不等 |

---

## 九、显存计算实操

```
KV cache 字节数 = 2 × L × B × H_kv × S × D_head × bytes_per_element
                  │   │   │    │     │    │
                  │   │   │    │     │    └─ 每元素字节：fp16=2, fp8=1, int4=0.5
                  │   │   │    │     └─ head_dim
                  │   │   │    └─ 每 token 每层的 K/V 宽度 = 2 × H_kv × D_head
                  │   │   └─ KV 头数（GQA 时 < num_attention_heads）
                  │   └─ batch size
                  └─ 层数        （最前面的 2 = K 和 V 各一份）
```

**公式在源码里的落点**：`StaticLayer.lazy_initialization`（`cache_utils.py:432-441`）分配的正是
`(batch_size, num_heads, max_cache_len, k/v_head_dim)`，即 `2 × B × H × S × D` 一份**每层**；
`H_kv` 取值规则见 `generation/continuous_batching/cache.py:29-39`（`find_num_kv_heads`：优先
`num_key_value_heads`，没有就用 `num_attention_heads`）与 `:42-53`（`find_head_dim`：优先 `head_dim`，
没有就用 `hidden_size // num_attention_heads`）。

**具体算一遍**（Llama-3-8B，`L=32`、`H_kv=8`、`D_head=128`、`bytes=2`，batch=1、seq=8192、fp16）：
`2 × 32 × 1 × 8 × 8192 × 128 × 2 = 1,073,741,824 字节 = 1.000 GiB`；对照模型权重 8.03B × 2 ≈ **14.96 GiB**，
即 seq=8192 时 KV cache 已占权重的 **6.7%**。

| 场景 | 参数 | 结果 |
|---|---|---|
| Llama-3-8B，b=1，S=8192，fp16 | `2×32×1×8×8192×128×2` | **1.000 GiB** |
| Llama-3-8B，b=1，S=32768，fp16 | 同上但 S×4 | **4.000 GiB** |
| Llama-3-8B，b=32，S=4096，fp16 | `2×32×32×8×4096×128×2` | **16.000 GiB** |
| （假设）MHA 版：H_kv=32 而非 8 | `2×32×1×32×8192×128×2` | **4.000 GiB**（GQA 省了 4×） |
| Llama-3-8B，b=1，S=8192，fp8 | 每元素 1 字节 | **0.500 GiB** |
| Llama-3-8B，b=1，S=8192，int4 | 每元素 0.5 字节 | **0.250 GiB** |
| **Mistral 7B（SWA W=4096），任意 S** | `2×32×1×8×4095×128×2` | **≈ 0.500 GiB（上限，与 S 无关）** |

四个结论：① **GQA 是免费的 4 倍**（32 头 → 8 头）；② **batch 维是乘性的**（`b=32, S=4096` 吃 16 GiB），
所以生产服务要上连续批处理；③ **量化是线性的**；④ **SWA 是唯一「和 S 无关」的省法**，代价是模型能力（见 1M2）。

> **本节的 config 数值说明**：`L / H_kv / D_head` 用的是 Llama-3-8B 公开的 config 常量（32 层 / 8 KV 头 /
> head_dim 128）。**本机无网络，没能现场拉 `config.json` 校验**，标注为「公开常量」。公式本身与 GQA→显存的
> 换算是**源码验证过的**（`cache_utils.py:432-441` 的分配形状 + `cache.py:29-53` 的推导规则）。
> 想完全自己复现，`LlamaConfig` 默认值就是 `hidden_size=4096, num_hidden_layers=32, num_attention_heads=32`
> → `head_dim=128`（`models/llama/configuration_llama.py:65-69`、`:87-90`），只是 `num_key_value_heads` 默认等于 32（MHA）。

---

## 十、`generation/continuous_batching/`：连续批处理为什么需要另一套 cache

前面所有 Cache 都假设：**一次 generate 调用 = 一个 batch = 一条固定长度的序列集合**。
**连续批处理（continuous batching）** 换成 **每个请求独立进出**、每步动态组 batch，
代价是 **batch 维不再固定、同 batch 序列长度各不相同**——扁平张量 `[B, H, S, D]` 装不下了。

### 10.1 目录结构（实测 `ls src/transformers/generation/continuous_batching/`）

| 文件 | 行数 | 职责 |
|---|---|---|
| `cache.py` | 814 | **`PagedAttentionCache`（:89）** + `PagedAttentionMemoryHandler`（:565） |
| `cache_manager.py` | 550 | `Block`（:37）、`BlockManager`（:58）、`CacheAllocator`（:297）、`FullAttentionCacheAllocator`（:365）、`SlidingAttentionCacheAllocator`（:456） |
| `continuous_api.py` | 1352 | 对外 API（`ContinuousBatchingManager` / `ContinuousMixin`），负责切到 `paged\|` 后端（:660-683） |
| `input_outputs.py` | 822 | **`PagedAttentionArgs`（:31）**、静态缓冲区（`ContinuousBatchingIOs`，:70） |
| 其余 | — | `__init__.py`(29，导出 `ContinuousBatchingManager` / `ContinuousMixin` / `PagedAttentionCache` / `FIFOScheduler` / `PrefillFirstScheduler` / `RequestState` / `RequestStatus` / `Scheduler`)、`scheduler.py`(432，FIFO / Prefill-first)、`requests.py`(365，`RequestState`/`RequestStatus`)、`model_runner.py`(313)、`offloading_manager.py`(380，CPU offloading)、`distributed.py`(189，TP)、`initialization.py`(290)、`cb_logits_processors.py`(328)、`utils.py`(219) |

### 10.2 `PagedAttentionCache`（`cache.py:89`）：三级层次

docstring（`:90-147`）翻成中文：

```
Page（页）   —— 最小单位。[num_heads, head_size]，即「一个 token、一层的 K 或 V」。
               全 full-attention 的模型，一个 token 要 2 × num_layers 个 page（K、V 各一半）
Block（块）  —— block_size 个 page 的集合，**分配单元**（按块分配/释放，不按页）。
               一个 block 只属于一个「层组」（layer group），组内所有层共享这一块
Cache tensor —— 物理载体。层组里有几层就有几个 tensor，形状 [num_blocks * block_size, num_heads, head_size]
```

按「层组」分配的原因（docstring 的例子）：8 个 block，一组是 3 层 full attention，另一组是 3 层 sliding attention——
继续生成时 sliding 组已填满窗口，只有 full 组需要新块；**若所有层在同一个组里**，sliding 组也会被迫跟着分配一块，**纯浪费**。

**它怎么和 attention 后端接上**（和前面所有 Cache 最不同的地方）：

| 步骤 | 代码 | 说明 |
|---|---|---|
| ① `ContinuousMixin` 把 `_attn_implementation` 改成 `paged\|xxx` | `continuous_api.py:681-683` `model.set_attn_implementation(f"paged\|{target_implem}")` | 没装 flash 就自动降级 |
| ② 注册表里 `paged\|*` 指向专门实现 | `modeling_utils.py:5161-5165` | `paged\|flash_attention_2/3/4` → `paged_attention_forward`；`paged\|sdpa` → `sdpa_attention_paged_forward`；`paged\|eager` → `eager_paged_attention_forward` |
| ③ **cache 不走 `past_key_values`，走 `cache=` kwarg** | `input_outputs.py:488` `cache=self.cache`；`:496` `use_cache=False` | 注释：`use_cache: Whether to use caching (always False in continuous batching as the cache is managed externally)` |
| ④ **后端自己调 `cache.update(...)`** | `eager_paged.py:37-42`、`sdpa_paged.py:37`、`flash_paged.py` | 这是和前 9 节最大的差别！ |

**两种读写语义**（`cache.py:393-452` 的 `update` docstring，实测）：**full attention** 先写新 K/V 再
`index_select` 读「整段历史 + 新 token」；**sliding window 必须先读后写**——新 K/V 可能覆盖还没读的旧 K/V
（chunked prefill 时出现），用 `sentinel_index` 占位 + `masked_scatter_` 补真值（`cache.py:441-446`，
`sentinel_index` 定义在 `:248`）；**`layer_read_index.numel() == 0`** 表示整批都是非 chunked prefill，
没有历史要读，只写不读、直接返回输入 K/V（`cache.py:424-426`）。

**两种 kernel 路径**（`docs/source/en/paged_attention.md`）：

| 路径 | kernel | cache 交互 | 何时用 |
|---|---|---|---|
| varlen | `flash_attn_varlen_func` | **无**——cache 要手动读/写，文档明说 `This can become a bottleneck when sequence length grows large` | 大量请求处于 prefill |
| decode | `flash_attn_with_kvcache` | **由 kernel 内部用 `block_table` 直接寻址** | 整批都只有 1 个 query token |

`block_table` 形状 `(num_groups, batch_size, max_blocks_per_seq)`，每个请求一行、元素是「第 i 个逻辑块在
物理 cache 里的位置」，不足的用 `-1` 填充（`flash_paged.py:36-40`）。

### 10.3 前缀共享（prefix caching）

分页之后「一块物理 cache = 一段确定的 token 序列」，所以**多个请求如果 prompt 前缀相同，可共享同一批 block**：
`PagedAttentionCache.search_prefix_match`（`cache.py:470`）/ `mark_shareable_blocks_as_complete`（:496）/ `fork_request`（:543）；
`BlockManager.compute_hash`（`cache_manager.py:279`）+ `hit` 计数 + `increase_ref_count` / `decrease_ref_count`
（:187 / :194，**引用计数**，最后一个引用消失才真正释放）；`FullAttentionCacheAllocator`（:365）/
`SlidingAttentionCacheAllocator`（:456）分化两套索引规则（滑动窗口层 `get_read_indices` 只给窗口内的位置）。

> **和前面 9 节的关系**：`PagedAttentionCache` **不是** `Cache` 的子类，也**没有任何 `CacheLayerMixin`**
> （实测：`cache.py` 未 import `cache_utils`）。它是**平行的一套体系**，因为它的核心矛盾（变长 batch +
> 显存碎片 + 前缀共享）和「单序列怎么存 K/V」不是同一个问题。

---

## 十一、一句话总结

> **`cache_utils.py` 是两层结构：`CacheLayerMixin` 管「单层怎么存」，`Cache` 管「所有层怎么组织」。**
>
> - **Layer 一支**：`Dynamic*`（cat 生长）/ `Static*`（预分配 + 原地写，为 compile）/
>   `Quantized*`（省显存）/ `LinearAttention*`（无 seq 维度，给 SSM）/ `*IndexedLayer`（多一份 indexer key，给 DSA）。
>   选哪个由 config 的 `layer_types` 字符串经 `DYNAMIC/STATIC_LAYER_TYPE_MAPPING` 决定。
> - **Cache 一支**：`DynamicCache`（默认）/ `StaticCache` / `QuantizedCache` /
>   `EncoderDecoderCache`（组合两个）/ `MtpCache` 与 `DFlashCache`（只改 query 偏移）。
> - **契约**：`update()` 返回**完整**的 K/V，`layer_idx` 定位层，位置信息由 cache 反向报给外面
>   （`get_seq_length` / `get_query_offset` / `get_mask_sizes`）。`cache_position` / `cache_kwargs` 在本版本**已删除**。
> - **和 2A2 的接口**是 `get_mask_sizes() → (kv_length, kv_offset)`；
>   **和 2A1 的接口**是「普通后端看不到 cache，paged 后端自己调 `cache.update`」。
> - **连续批处理**用另一套 `PagedAttentionCache`（block/页表/前缀共享），不继承 `Cache`，
>   靠 `paged|*` 后端 + `cache=` kwarg 接上。

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
