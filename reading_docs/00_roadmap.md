# Transformers 精读路线图

> 目标：系统学习 `transformers` 这个库。
> 两条线并行：**架构**（库自身的机制与源码调用链——「原理」）与**模型结构**（主流模型长什么样——「模型」）。
> 每条线分两级：**入门**先把骨架摸清，**进阶**再啃稀疏/压缩/多模态这些真难点。

---

## 2×2 总览

|          | 架构（库机制）            | 模型结构                      |
|----------|---------------------------|-------------------------------|
| **入门** | 1A1–1A6：一次 forward / generate / train 走过哪些文件 | 1M1–1M5：decoder-only 骨架 + encoder 对照 |
| **进阶** | 2A1–2A6：注意力后端 / mask / cache / RoPE / 代码生成 / 权重 | 2M1–2M7：MoE、MLA、SSM、多模态、GLM 压轴 |

**两个象限的关系**：1A 讲「库怎么把模型跑起来」，1M 讲「模型本身是什么」。
进阶同理——2A 讲库为支持这些模型做了哪些机制，2M 讲这些模型的创新点在哪。
两边对照读，收益最大（例：读完 2M2 的 MLA，再读 2A3 看 KV cache 怎么被压缩，闭环）。

**文件命名**：`<等级><线><序号>_<描述>.md`。等级 `1`=入门 `2`=进阶，线 `A`=架构 `M`=模型结构。
1A1 → 1M5 → 2A1 → 2M7 就是建议的排序。

---

# Part 1 · 入门

先把「库的骨架」和「最主流的模型骨架」各摸一遍。读完这一部分，看任何一个新模型都能定位到「它改了哪一块」。

## 1A · 架构入门（库机制）

目标：搞清楚一次 `from_pretrained` / `generate` / `train` 在库内部经过哪些文件，每个文件负责什么。

| 篇 | 标题 | 核心概念 | 源文件 | 状态 | 依赖 |
|----|------|----------|--------|------|------|
| 1A1 | 仓库骨架与 auto 机制 | `models/<name>/` 五件套（config / modeling / tokenization / processing / auto）；`AutoModel` 如何从 config 找到类。**两个易错点**：`_LazyAutoMapping` **没有** `__getattr__`（惰性来自 `_load_attr_from_module` 的 `importlib`）；模型映射表里**没有 `None`**，「不支持某任务头」= 不在表里 | `models/auto/auto_factory.py`、`models/auto/auto_mappings.py`、`models/auto/modeling_auto.py` | ✅ | — |
| 1A2 | 配置系统 | v5 已 dataclass 化（`@dataclass_transform` / `@strict` / `@dataclass`）；字段如何决定建模结构。**`@strict` 来自 `huggingface_hub.dataclasses`（不是本库）**；`sub_configs` / `get_text_config()` 处理组合式 config | `configuration_utils.py` | ✅ | 1A1 |
| 1A3 | `from_pretrained` 全链路 | 下载 → `_from_config` 建模 → 灌权重 → `_init_weights`；missing / unexpected keys 的含义 | `modeling_utils.py`、`core_model_loading.py`、`utils/hub.py` | ✅ | 1A1、1A2 |
| 1A4 | Tokenizer 体系 | **v5 已重构**：不再有 fast / slow 两个类。真实类是 `TokenizersBackend` / `PythonBackend` / `SentencePieceBackend`（`PreTrainedTokenizer` / `PreTrainedTokenizerFast` 降级成**别名**）；模型侧只剩**一个** `tokenization_<name>.py`；`AutoTokenizer(use_fast=)` 被静默忽略 | `tokenization_utils_base.py`、`tokenization_utils_tokenizers.py`、`tokenization_python.py`、`tokenization_utils_sentencepiece.py` | ✅ | 1A1 |
| 1A5 | `generate` 主循环 | `GenerationConfig`、`prepare_inputs_for_generation`、逐 token 循环、logits 处理与停止条件 | `generation/utils.py`、`generation/configuration_utils.py`、`generation/logits_process.py`、`generation/stopping_criteria.py` | ✅ | 1A3 |
| 1A6 | `Trainer` 主循环 | `TrainingArguments`（实测 **112** 个 `field()`，不是常说的 200）、三层循环（epoch / update_step / micro-batch）、回调钩子 | `trainer.py`、`training_args.py`、`trainer_callback.py`、`trainer_pt_utils.py` | ✅ | 1A3 |

> **1A6 的一个反直觉发现**：`Trainer.autocast_smart_context_manager`（`trainer.py:2081`）
> 现在是个**空壳**——它直接 `return contextlib.nullcontext()`，注释写着
> 「We rely on accelerate for autocast, hence we do nothing here.」。
> 混合精度实际由 `accelerate` 在 `Accelerator` 里接管，不是 Trainer 自己做的。
> 读老版本代码/博客时要知道这个函数曾经是有实现的。

**读法**：1A1 → 1A3 是一条线（模型怎么被造出来），1A4 → 1A5 → 1A6 是另一条线（数据怎么进去、结果怎么出来）。
两层之间唯一的耦合点是 1A3——训练和推理都用它造模型。

## 1M · 模型结构入门

目标：吃透 decoder-only 骨架，再拿到 encoder 的对照面。

| 篇 | 标题 | 核心概念 | 源文件 | 状态 | 依赖 |
|----|------|----------|--------|------|------|
| 1M1 | **Llama** | Decoder-only 骨架：RMSNorm / RoPE / SwiGLU / GQA / Pre-Norm 残差 | `models/llama/modeling_llama.py`（512 行，无 modular，完整可直读） | ✅ | — |
| 1M2 | **Mistral** | 滑动窗口注意力（SWA） | `models/mistral/modular_mistral.py` | ✅ | 1M1 |
| 1M3 | **Mixtral** | MoE 第一课：稀疏激活、router、top-k gating。**专家类现名 `MixtralExperts`**（3D 张量） | `models/mixtral/modular_mixtral.py` | ✅ | 1M2 |
| 1M4 | **Qwen3**（dense） | QK-Norm（在 RoPE **之前**、逐头）、`head_dim` 与 `hidden_size` 解耦、**RoPE 参数统一进 `rope_parameters`** | `models/qwen3/modular_qwen3.py` | ✅ | 1M1 |
| 1M5 | **BERT / ModernBERT** | Encoder 对照面：双向注意力（**band 窗口，非因果**）、MLM、`[CLS]`；ModernBERT 的现代化改造 | `models/bert/modeling_bert.py`（无 modular）、`models/modernbert/modular_modernbert.py` | ✅ | 1M1 |

> **两个读书时的「坑」预告**（都已在对应文档里讲清）：
> - **1M4 / 2A4**：`config.rope_scaling` 已改名为 **`rope_parameters`**。
>   `rope_scaling` 降级成 `property` 别名（`configuration_utils.py:548-554`），兼容层之外全库基本不再用它。
>   读老博客/老代码时看到 `rope_scaling`，要立刻想到「它现在叫 rope_parameters」。
> - **1M5**：ModernBERT 的滑动窗口是 **`abs(q_idx - kv_idx) <= w`（中心带）**，
>   不是 Mistral 的 `kv_idx > q_idx - w`（只看左边）。encoder 不能看未来这句话不成立，所以窗口变成双向的。

**为什么这个顺序**：
- 1M1 是祖宗级模型——它被别人继承，自己不继承任何人，所以 `modeling_llama.py` 是完整源码、没有跳转。
- 1M2 只用 5 分钟确认「差异只有 `sliding_window` 一处」，练的是**读 modular 文件的肌肉**。
- 1M3 是第一次真正的结构创新（不是改超参，是换模块）。
- 1M4 补齐现代 dense 的细节，1M5 换一个完全不同的架构族（encoder）来防止思维定势。
- **1M1 的锚点价值**：GLM / DeepSeek / Qwen 的 DecoderLayer 结构与 Llama 完全一致，只是换了 Attention 和 MLP。
  1M1 学扎实，2M 全是「Llama + 若干改动」。

---

# Part 2 · 进阶

## 2A · 架构进阶（库机制）

目标：理解库为了让上面那些模型跑得动、跑得快，造了哪些**抽象层**。
这一部分的共同主题是「**分发与组合**」——都是先定义一组可替换实现，再在运行时按 config 选一个。

| 篇 | 标题 | 核心概念 | 源文件 | 状态 | 依赖 |
|----|------|----------|--------|------|------|
| 2A1 | 注意力后端机制 | `ALL_ATTENTION_FUNCTIONS` 注册表（**实测 10 个 key，没有 `"eager"`**——eager 是回退默认值）；`AttentionInterface` 本体在 `modeling_utils.py:5147`。**`_attn_implementation_autoset` 已从库中移除** | `modeling_utils.py`、`modeling_flash_attention_utils.py`、`integrations/sdpa_attention.py`、`integrations/flash_attention.py` | ✅ | 1A2、1M1 |
| 2A2 | Mask 系统 | `mask_function` 的组合代数（`and_masks` / `or_masks`）；causal / sliding-window / padding 如何拼成一个 mask | `masking_utils.py`、`modeling_attn_mask_utils.py` | ✅ | 1M2、2A1 |
| 2A3 | KV Cache 体系 | `CacheLayerMixin` → `DynamicLayer` / `StaticLayer` / `QuantizedLayer` / `LinearAttentionLayer`；`Cache` / `DynamicCache` / `StaticCache` / `EncoderDecoderCache` | `cache_utils.py`、`generation/continuous_batching/` | ✅ | 1A5、2A2 |
| 2A4 | RoPE 抽象层 | `ROPE_INIT_FUNCTIONS`（**实测只有 6 个 key，没有 `"default"`**——default 走各模型自带的静态方法）；linear / dynamic / yarn / longrope / llama3 / proportional。**字段已改名 `rope_parameters`**；`rope_config_validation` 已废弃、零调用者 | `modeling_rope_utils.py` | ✅ | 1M1 |
| 2A5 | **代码生成机制** | `modular_*.py` 如何生成 `modeling_*.py`；`# Copied from` 的同步规则；为什么不能直接改生成文件 | `utils/modular_model_converter.py`、`utils/check_modular_conversion.py`、`utils/check_copies.py`、Makefile 的 `fix-repo` | ✅ | 1M2 |
| 2A6 | 权重加载、量化与并行 | 跨实现权重映射；量化后端如何挂进来；张量并行 / 专家并行。**两个路径修正**：张量并行本体在 `distributed/tensor_parallel.py`（`integrations/tensor_parallel.py` 只是兼容 shim）；`integrations/moe.py` 不是分片逻辑而是 experts forward 的内核分派 | `core_model_loading.py`、`conversion_mapping.py`、`quantizers/`、`distributed/tensor_parallel.py`、`integrations/moe.py` | ✅ | 1A3 |

**建议穿插**：**2A5 已写完（[2A5_modular_code_generation.md](./2A5_modular_code_generation.md)），建议现在就插进来读**——
它不依赖任何模型结构知识，而且**越早读越好**：它决定了你读后续所有源码的方式（读 modular 而不是生成文件，
看到自动拉取的类不要当成新东西）。
2A1 / 2A2 / 2A3 是「注意力三件套」，本质是同一件事的三个侧面：**算什么（后端）、在哪算（mask）、缓存什么（cache）**。建议连着读。

> **⚠️ 一个提醒**：这 6 篇里有多处「流传很广但已经过时」的说法被实测推翻了。
> 每篇的验证清单里都写了实测命令，**遇到和记忆冲突的地方，以验证清单为准**。已知的被推翻项：
> - `_attn_implementation_autoset` 已从库中移除（2A1）
> - 注册表里没有 `"eager"`，它是回退默认值（2A1）
> - `cache_position` / `cache_kwargs` 已被删除（2A3、1A5 独立确认）
> - `MixtralBlockSparseTop2MLP` 已不存在，现名 `MixtralExperts`（1M3）
> - `TrainingArguments` 是 112 个字段，不是 200（1A6）
> - `Trainer.autocast_smart_context_manager` 是空壳（1A6）
> - `LogitsWarper` 基类不存在（1A5）
> - `_LazyAutoMapping` 没有 `__getattr__`；模型映射表里没有 `None`（1A1）
> - tokenizer 的 fast / slow 双类已取消，旧类名降级为别名（1A4）
> - 张量并行本体在 `distributed/`，`integrations/tensor_parallel.py` 只是 shim（2A6）
> - **T5 是 Pre-Norm，不是 Post-Norm**（2M4）——norm 藏在各子层内部（`T5LayerSelfAttention` /
>   `T5LayerFF`），`T5Block.forward` 里一个 norm 都没有，所以「看起来像 post-norm」。
>   真正的 Post-Norm 是 BERT（`modeling_bert.py:292`/`:350`），与 1M5 一致
> - **T5 没有 embedding scaling**（2M4）——`model_dim**-0.5` 只用在 **decoder 输出**
>   （`modeling_t5.py:1045`，由 `scale_decoder_outputs` 控制），不是 embed 层
> - **`@strict` 来自 `huggingface_hub.dataclasses`**，不是本库；它是**类型校验**而非拒绝未知字段（1A2）
> - **Gemma 2 的 soft-cap 在默认后端下被静默丢弃**（2M1）⚠️ 见下方单独的警告块
> - `ROPE_INIT_FUNCTIONS` 只有 6 个 key，**没有 `"default"`**（2A4）；
>   `rope_config_validation` 已废弃、**零调用者**（2A4）

> ## ⚠️ 一个值得注意的实测发现：Gemma 2 的 soft-cap 默认不生效
>
> `attn_logit_softcapping=50.0` 是 Gemma 2 的核心设计之一，但**它只在 `eager` 和
> `flash_attention_2` 后端下真正执行**：
>
> | 后端 | 是否应用 soft-cap | 证据 |
> |------|------------------|------|
> | `eager` | ✅ | Gemma2 自带 `eager_attention_forward` 里三行 `x/c → tanh → *c`（`modular_gemma2.py:202-205`） |
> | `sdpa`（**默认**） | ❌ **静默丢弃，无警告** | `grep -c softcap integrations/sdpa_attention.py` → **0**；签名只有 `**kwargs` 吞掉 |
> | `flash_attention_2` | ✅ 运行时探测 | `modeling_flash_attention_utils.py:226-245` 用 `inspect.signature` 判断 |
> | `flex_attention` | ✅ | 在 `score_mod` 里做（`integrations/flex_attention.py:291-292`） |
> | 三个 `*_paged` | ❌ | — |
>
> 而 HF 的默认后端就是 sdpa（`modeling_utils.py:1927`）。
> **所以「加载一个 Gemma 2 直接推理」拿到的是没有 soft-cap 的模型**——
> 数值上与官方实现不一致，却不报错、不警告。
> 要复现官方行为需显式 `attn_implementation="eager"`（或 flash）。
>
> 这条是 2M1 精读时发现的，**尚未确认是否为有意设计**（可能上游认为 soft-cap 可省）。
> 读到 2M1 时注意它的验证清单里有完整证据。

## 2M · 模型结构进阶

目标：每个模型对应一类**独立的创新**。读完这一部分，主流结构基本无死角。

| 篇 | 标题 | 核心概念 | 源文件 | 状态 | 依赖 |
|----|------|----------|--------|------|------|
| 2M1 | **Gemma 2** | 软上限注意力（`attn_logit_softcapping=50.0` / `final_logit_softcapping=30.0`，两处位置）、交替局部 / 全局注意力、`query_pre_attn_scalar=256` 决定 scaling、Pre+Post 双 norm | `models/gemma2/modular_gemma2.py` | ✅ | 1M2 |
| 2M2 | **DeepSeek V2 → V3** | MLA（低秩 KV 压缩）、DeepSeekMoE（细粒度路由 + 共享专家） | `models/deepseek_v2/modular_deepseek_v2.py`、`models/deepseek_v3/modular_deepseek_v3.py`、`models/deepseek_v32/modular_deepseek_v32.py` | ✅ | 1M3 |
| 2M3 | **Qwen3-MoE** | 三种 MoE 的取舍对比：Mixtral / DeepSeek / Qwen3-MoE；通用 MoE 组件抽取 | `models/qwen3_moe/modular_qwen3_moe.py` | ✅ | 2M2 |
| 2M4 | **T5** | Encoder-Decoder：相对位置偏置（`num_buckets=32` / `max_distance=128`，**只在第 0 层算**）、cross-attention（一个类三种角色，4 个开关）、`EncoderDecoderCache`。**实测是 Pre-Norm，不是 Post-Norm** | `models/t5/modeling_t5.py`（无 modular） | ✅ | 1M5 |
| 2M5 | **Mamba 2** | SSM：状态空间模型，序列建模完全不用注意力 | `models/mamba2/modeling_mamba2.py`（`Mamba2Mixer`） | ✅ | 1M1 |
| 2M6 | **LLaVA-NeXT** | 多模态：视觉编码器 + 语言模型的拼接与投影 | `models/llava_next/modeling_llava_next.py` | ✅ | 1M1 |
| 2M7 | **GLM-5.3**（`glm_moe_dsa`） | 压轴：MLA + DSA 稀疏 top-k + IndexShare 跨层复用 | `models/glm_moe_dsa/modular_glm_moe_dsa.py` | ✅ | 2M2、2M3 |

> **2M2 为什么是三个文件**：DeepSeek 的演进分了三个模型目录，而且**它们是层层继承的**：
> `deepseek_v2`（MLA + DeepSeekMoE 首次出现）→ `deepseek_v3`（改进）→ `deepseek_v32`（加 Indexer/DSA）。
> 读完这三层，2M7 的 GLM 就只剩 IndexShare 一个全新概念了。

> **一条贯穿全篇的继承链**（实测确认，也是 2A5「继承展开」的最佳案例）：
> ```
> LlamaDecoderLayer
>   → DeepseekV3DecoderLayer        (models/deepseek_v3)
>     → Glm4MoeDecoderLayer         (models/glm4_moe)
>       → Glm4MoeLiteDecoderLayer   (models/glm4_moe_lite)
>         → DeepseekV32DecoderLayer (models/deepseek_v32)
>           → GlmMoeDsaDecoderLayer (models/glm_moe_dsa)
> ```
> 5 层深。**GLM 的 DecoderLayer 骨架仍然是 Llama 的 Pre-Norm + 双残差**，只是穿过整条链传下来。
> 2M7 会详细拆解这条链——它也是「为什么 converter 必须按拓扑序转换」的答案。


**内部顺序**：
- 2M1 是 1M2 的自然延伸（还是局部/全局这事，多了 soft-cap）。
- 2M2 → 2M3 是一组：先看 MLA + DeepSeekMoE，再横向比三种 MoE 的取舍。
- 2M4 补上 encoder-decoder 形状（1M5 给了 encoder，这里给出解码侧）。
- 2M5 是「无注意力」替代路线，独立成环。
- 2M6 依赖对 1M1 的熟悉，但和别的篇不冲突，随时可插。
- **2M7 是终点**：它把 2M2 的 MLA、2M3 的 MoE、外加 DSA 稀疏选 top-k 全用上，
  DecoderLayer 骨架却和 1M1 一模一样。读到这里如果能一眼看穿「这就是 Llama + 换了两个模块」，整条线就闭环了。

---

## 推荐阅读路径

**最短主线**（想快速建立直觉，≈ 8 篇）：
```
1A1 → 1A2 → 1A3 → 1M1 → 1M2 → 1M3 → 2M2 → 2M7
```
> 强烈建议在 1M2 之后插入 2A5（代码生成机制）——5 分钟读完，但省下后面每次读源码时的困惑。

**完整系统**（按 2×2 逐象限推进）：
```
1A1 → 1A2 → 1A3 → 1M1 → 1M2 → 1M3 → 1M4 → 1M5
      → 2A5（越早越好，已写完）
      → 1A4 → 1A5 → 1A6
      → 2A1 → 2A2 → 2A3 → 2A4 → 2A6
      → 2M1 → 2M2 → 2M3 → 2M4 → 2M5 → 2M6 → 2M7
```

**直奔 GLM**（如果你的目标是读懂某个具体大模型）：
```
1M1（打地基）→ 2M2（MLA + MoE 概念）→ 2A3（KV cache 怎么被压缩）
→ 2A1 / 2A2（稀疏注意力需要的后端与 mask 机制）→ 2M7
```

---

## 现有文件对照

| 文件名 | 所属 | 状态 |
|--------|------|------|
| `1M1_llama_structure.md` | 入门 · 模型结构 | ✅（原 `01_llama_structure.md`） |
| `1M2_mistral_sliding_window.md` | 入门 · 模型结构 | ✅（原 `02_mistral_sliding_window.md`） |
| `2A5_modular_code_generation.md` | 进阶 · 架构 | ✅ |

> 文档之间用相对链接互引（如 `[1M1](./1M1_llama_structure.md)`）。
> 新增文档时记得回填「上一篇 / 下一篇」导航。

---

## modular 文件说明

大部分模型有 `modular_<name>.py`，直接继承父模型并只覆盖差异部分。
`make fix-repo` 会从 modular 生成完整的 `modeling_<name>.py`。

**读 modular 文件**比读生成文件效率高——差异一目了然。已确认的分布：

| 有 modular（读 modular） | 无 modular（直读 modeling） |
|--------------------------|------------------------------|
| mistral、mixtral、gemma2、deepseek_v2、qwen3、qwen3_moe、modernbert、glm_moe_dsa | llama、bert、t5、mamba2、llava_next |

> 生成文件**不要直接编辑**——改动会被 `make fix-repo` 覆盖。
> 完整机制见 [2A5_modular_code_generation.md](./2A5_modular_code_generation.md)。
