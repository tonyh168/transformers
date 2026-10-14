# Transformers 精读路线图

> 目标：系统学习 `transformers` 这个库。
> 两条线并行：**架构**（库自身的机制与源码调用链——「原理」）与**模型结构**（主流模型长什么样——「模型」）。
> 每条线分两级：**入门**先把骨架摸清，**进阶**再啃稀疏/压缩/多模态这些真难点。
>
> **25 篇全部完成**（29,171 行）。随手可用的两个入口：
> - **[index.html](./index.html)** —— 交互式学习地图：阅读路径 + 进度追踪 + 速查表
> - **[附录 · 31 条被实测推翻的说法](#附录--29-条被实测推翻的说法)** —— 读之前先扫一眼，能省很多困惑

---

## 读之前：这套文档的一个前提

写这 25 篇的过程中，**有 31 条流传很广的说法被实测推翻了**（`cache_position` 已删除、
T5 不是 Post-Norm、张量并行不在 `integrations/`……）。
其中一条甚至会**改变推理结果**——Gemma 2 的 soft-cap 在默认 `sdpa` 后端下被静默丢弃。

所以：**这套文档的每个断言都带真实行号，文末都有「验证清单」列出实际跑过的命令。**
遇到和记忆冲突的地方，以验证清单为准，不要以直觉为准。

完整清单见文末[附录](#附录--29-条被实测推翻的说法)。

---

## 2×2 总览

|          | 架构（库机制）            | 模型结构                      |
|----------|---------------------------|-------------------------------|
| **入门** | 1A1–1A6：一次 forward / generate / train 走过哪些文件 | 1M1–1M5：decoder-only 骨架 + encoder 对照 |
| **进阶** | 2A1–2A7：注意力后端 / mask / cache / RoPE / 代码生成 / 权重 / 接口契约 | 2M1–2M7：MoE、MLA、SSM、多模态、GLM 压轴 |

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
| 2A7 | 注意力接口契约与能力协商 | 契约有三层：硬契约（5 位置参数 + 2-tuple）／协商层（`**kwargs` 是唯一扩展点，**认不出就静默丢**）／准入层（`_supports_*` + `_compatible_flash_implementations`）。**`_supports_sdpa` 默认是 `False`（要显式 opt-in）**；实测 15 个模型禁 sdpa。特化 eager ≠ 不能换后端 | `modeling_utils.py`、`configuration_utils.py`、`modeling_flash_attention_utils.py`、`integrations/*_attention.py` | ✅ | 2A1 |

**建议穿插**：**2A5 已写完（[2A5_modular_code_generation.md](./2A5_modular_code_generation.md)），建议现在就插进来读**——
它不依赖任何模型结构知识，而且**越早读越好**：它决定了你读后续所有源码的方式（读 modular 而不是生成文件，
看到自动拉取的类不要当成新东西）。
2A1 / 2A2 / 2A3 是「注意力三件套」，本质是同一件事的三个侧面：**算什么（后端）、在哪算（mask）、缓存什么（cache）**。建议连着读。
**2A7 是 2A1 的补篇**：2A1 讲「名字怎么变成函数」，2A7 讲「函数之间凭什么能互换」——
读完 2A1 你知道怎么换后端，读完 2A7 你知道**换了之后会不会悄悄算错**。建议紧跟在 2A1 后面读。

> **⚠️ 读之前先看一眼**：这套文档涉及的大量「流传很广的说法」**已经在 v5 里失效了**。
> 完整的 **31 条清单**见文末 [附录 · 31 条被实测推翻的说法](#附录--29-条被实测推翻的说法)。
> 其中一条会**改变推理结果**（Gemma 2 的 soft-cap），在附录里单独高亮。
> 遇到和记忆冲突的地方，以各篇的「验证清单」为准——那些命令都是实际跑过的。

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
      → 2A1 → 2A7 → 2A2 → 2A3 → 2A4 → 2A6
      → 2M1 → 2M2 → 2M3 → 2M4 → 2M5 → 2M6 → 2M7
```

**直奔 GLM**（如果你的目标是读懂某个具体大模型）：
```
1M1（打地基）→ 2M2（MLA + MoE 概念）→ 2A3（KV cache 怎么被压缩）
→ 2A1 / 2A2（稀疏注意力需要的后端与 mask 机制）→ 2M7
```

---

## 全部文件索引

25 篇全部完成。**23 篇长文都带「速读区（TL;DR）」**——复习时只看前 50 行即可；
`1M1` / `1M2` 不足 300 行，不需要速读区。25 篇都有「验证清单」。

| 篇 | 文件 | 行数 | 速读区 |
|----|------|------|--------|
| 1A1 | `1A1_repo_and_auto.md` | 1242 | ✅ |
| 1A2 | `1A2_config_system.md` | 1018 | ✅ |
| 1A3 | `1A3_from_pretrained.md` | 2019 | ✅ |
| 1A4 | `1A4_tokenizer.md` | 1396 | ✅ |
| 1A5 | `1A5_generate_loop.md` | 1390 | ✅ |
| 1A6 | `1A6_trainer.md` | 1185 | ✅ |
| 1M1 | `1M1_llama_structure.md` | 252 | — |
| 1M2 | `1M2_mistral_sliding_window.md` | 265 | — |
| 1M3 | `1M3_mixtral_moe.md` | 1091 | ✅ |
| 1M4 | `1M4_qwen3_dense.md` | 716 | ✅ |
| 1M5 | `1M5_bert_modernbert.md` | 1275 | ✅ |
| 2A1 | `2A1_attention_backends.md` | 1552 | ✅ |
| 2A2 | `2A2_masking_system.md` | 1851 | ✅ |
| 2A3 | `2A3_kv_cache.md` | 1431 | ✅ |
| 2A4 | `2A4_rope_abstraction.md` | 1596 | ✅ |
| 2A5 | `2A5_modular_code_generation.md` | 642 | ✅ |
| 2A6 | `2A6_weights_quant_parallel.md` | 1389 | ✅ |
| 2A7 | `2A7_attention_contract.md` | 765 | ✅ |
| 2M1 | `2M1_gemma2.md` | 1073 | ✅ |
| 2M2 | `2M2_deepseek_mla_moe.md` | 1392 | ✅ |
| 2M3 | `2M3_qwen3_moe.md` | 1270 | ✅ |
| 2M4 | `2M4_t5_encoder_decoder.md` | 1325 | ✅ |
| 2M5 | `2M5_mamba2_ssm.md` | 1100 | ✅ |
| 2M6 | `2M6_llava_next_multimodal.md` | 1238 | ✅ |
| 2M7 | `2M7_glm_moe_dsa.md` | 693 | ✅ |
| | **合计** | **29,171** | 23 / 25 |

> **`index.html`** 是同目录下的学习地图：3 条阅读路径、可勾选的进度追踪、以及下面的 31 条清单。
> 浏览器直接打开即可（进度存在浏览器本地）。

> **历史沿革**：最早的 `01_llama_structure.md` / `02_mistral_sliding_window.md`
> 已重命名为 `1M1` / `1M2`，纳入本矩阵。
> 文档之间用相对链接互引（如 `[1M1](./1M1_llama_structure.md)`）。

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

---

# 附录 · 31 条被实测推翻的说法

这套文档在写作过程中逐个实测源码，**有 31 条流传很广的说法被推翻**。
它们散落在各篇的正文和「验证清单」里，这里汇总成一张速查表。

**遇到和记忆冲突的地方，以这里和验证清单为准。** 每条都标了在哪一篇、以及判据。

> ## ⚠️ 其中一条会改变推理结果
>
> **Gemma 2 的 `attn_logit_softcapping` 在默认后端下被静默丢弃。**
> 它只在 `eager` / `flash_attention_2` / `flex_attention` 下真正执行：
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
> `final_logit_softcapping` 不受影响，它作用在 lm_head 输出上。
>
> 这条是 2M1 精读时发现的，**尚未确认是否为有意设计**（可能上游认为 soft-cap 可省）。
> 完整证据见 [2M1](./2M1_gemma2.md) 的验证清单。

## 一、已经不存在了（删除与改名）— 8 条

| 流传的说法 | 实测真相 | 篇 |
|-----------|---------|-----|
| `_attn_implementation_autoset` 会按硬件自动挑后端 | 已从库中移除。只剩「默认 `sdpa`，不可用则静默降级 `eager`」（`modeling_utils.py:1927` / `:1953-1960`）。唯一的自动升级在连续批处理里 | 2A1 |
| `cache_position` / `cache_kwargs` | **已删除**。源码原话：*"This arg has been removed from the Transformers library"*（`generation/utils.py:628`），只剩给 remote code 的兼容 shim | 2A3、1A5 |
| `LogitsWarper` 基类 | 不存在。warper 直接继承 `LogitsProcessor`（`logits_process.py:238` / `:473` / `:542`） | 1A5 |
| `MixtralBlockSparseTop2MLP` | 已删除，现名 **`MixtralExperts`**（`nn.Module` + 3D 张量，不是 `ModuleList`） | 1M3 |
| tokenizer 有 `PreTrainedTokenizer` / `PreTrainedTokenizerFast` 两个类 | 都降级成**别名**（`tokenization_python.py:1424` / `tokenization_utils_tokenizers.py:1478`）。真实类是 `TokenizersBackend` / `PythonBackend` / `SentencePieceBackend`；`AutoTokenizer(use_fast=)` 被静默吞掉 | 1A4 |
| `config.rope_scaling` | 已改名 **`rope_parameters`**，`rope_scaling` 只剩 `property` 别名（`configuration_utils.py:548-554`） | 1A2、2A4 |
| `rope_config_validation` 负责 RoPE 校验 | 已废弃，**全库零调用者**。校验搬到 `RotaryEmbeddingConfigMixin.validate_rope` | 2A4 |
| Mamba2 有 `is_mambapy` 慢速回退开关 | 没有。Mamba2 把它整个移交给了 kernel 分发；该开关只在 **Mamba1** 里（`configuration_mamba.py:40`） | 2M5 |

## 二、位置指错了（东西在别的文件里）— 6 条

| 流传的说法 | 实测真相 | 篇 |
|-----------|---------|-----|
| 张量并行在 `integrations/tensor_parallel.py` | 那只是 93 行的兼容 shim（文件头自称 BC shim）。本体在 **`distributed/tensor_parallel.py`**（843 行） | 2A6 |
| `integrations/moe.py` 是专家分片逻辑 | 它是 experts forward 的**内核分派**（deepgemm / grouped_mm / batched_mm / sonicmoe）。分片逻辑在 `distributed/` | 2A6 |
| `@strict` 是本库的装饰器 | 来自 **`huggingface_hub.dataclasses`**（`configuration_utils.py:25`），不是本库 | 1A2 |
| `ROPE_INIT_FUNCTIONS` 里有 `"default"` | 只有 **6 个 key**，**没有 `default`**——它由各模型自带的 `compute_default_rope_parameters` 静态方法兜底 | 2A4 |
| 注意力注册表含 `"eager"` | 只有 **10 个 key**，**没有 `eager`**——它是 `get_interface(name, default)` 的回退默认值 | 2A1 |
| llama / mistral 有自己的 `prepare_inputs_for_generation` | 已上提到 **`GenerationMixin`**（`generation/utils.py:519`）；全库 500+ 模型目录只剩 75 处定义 | 1A5 |

## 三、语义反了（方向或含义搞错）— 7 条

| 流传的说法 | 实测真相 | 篇 |
|-----------|---------|-----|
| T5 是 Post-Norm | 实测是 **Pre-Norm**——norm 藏在各子层内部（`modeling_t5.py:392-400` / `:138-140`），`T5Block.forward` 里一个 norm 都没有，所以「看起来像 post-norm」。真正的 Post-Norm 是 BERT（`modeling_bert.py:292` / `:350`） | 2M4 |
| T5 对 input embedding 做了缩放 | 没有。`model_dim**-0.5` 只用在 **decoder 输出**侧（`modeling_t5.py:1045`，由 `scale_decoder_outputs` 控制） | 2M4 |
| `fusion_mapping.py` 把 fused 层拆开 | **方向相反**：它把运行时模型改成 fused（Conv3d patch_embed → Linear），再注册映射让原始 checkpoint 仍能加载 | 2A6 |
| 专家并行用 all-to-all 通信 | `all_to_all` 全库**零命中**。是 **all-reduce** 合成输出（`tensor_parallel.py:320` / `:353`） | 2A6 |
| ModernBERT 的滑动窗口与 Mistral 同向 | 它是**双向 band**（`abs(q_idx - kv_idx) <= w`，`masking_utils.py:149`）；Mistral 是因果的（只看左边）。encoder 里「不能看未来」不成立 | 1M5 |
| `_LazyAutoMapping` 靠 `__getattr__` 做惰性导入 | 它**没有** `__getattr__`。惰性来自 `_load_attr_from_module` 里的 `importlib.import_module`（`auto_factory.py:612-616`） | 1A1 |
| 特化了 attention 的模型不能用 sdpa / FA | **看后端有没有对应的表达方式，不是看模型有没有特化**。Gemma 2 特化了 soft-cap，但 `flash_attention_2` 的签名里恰好有 `softcap`（`flash_attention.py:35`）→ **完全可用**；真正不能用的是 attention sink 那类（`gpt_oss` 等 5 个，`modeling_gpt_oss.py:251-252`），因为 sdpa 没有"多加一列"的接口，作者才写 `_supports_sdpa = False` | 2A7 |

## 四、静默行为（不报错，但结果不同）— 6 条

| 流传的说法 | 实测真相 | 篇 |
|-----------|---------|-----|
| Gemma 2 的 soft-cap 总生效 | 默认 `sdpa` 后端**静默丢弃**（见上方警告块） | 2M1 |
| 三家 MoE 都有负载均衡辅助损失 | DeepSeek V2/V3 **没有**（`grep -c aux_loss` 两个文件都是 0），靠 `e_score_correction_bias` 做无辅助损失均衡：**选专家看偏置**（`scores + bias` → `topk`，`modular_deepseek_v3.py:99` / `:114`），**算权重不看偏置**（`topk_weights = scores.gather(...)`，`:115`） | 2M3 |
| BERT 的 MLM loss 只对被 mask 的位置算 | 模型算**全序列** cross-entropy（`modeling_bert.py:972-975` 用裸 `CrossEntropyLoss()`）；位置筛选在数据侧（`data_collator.py:831` 把其余位置填 `-100`） | 1M5 |
| 映射表用 `None` 表示「不支持该任务头」 | 模型映射表里 `None` 出现 **0 次**。「不支持」= 不在表里 → `ValueError`。`None` 只在 tokenizer / processor 映射里表示「可选依赖没装」 | 1A1 |
| LLaVA-NeXT 屏蔽 image token 的 loss | 模型内**没有**这个屏蔽，图像位置默认要算 loss，屏蔽由数据侧负责 | 2M6 |
| 不指定 `attn_implementation` 的模型都用 sdpa | `_supports_sdpa` 默认是 **`False`**（`modeling_utils.py:1237`，必须显式 opt-in）。实测 495 个 `modeling_*.py` 只有 340 个写 `= True`；没声明的 141 个里有 **13 个**走了 functional 接口 → 什么都不传时经 `_sdpa_can_dispatch` 抛错被吞、**静默降级 eager**（`modeling_utils.py:1953-1960`，`except` 里无任何日志） | 2A7 |

## 五、数字与结构（记错了个数和层级）— 4 条

| 流传的说法 | 实测真相 | 篇 |
|-----------|---------|-----|
| `TrainingArguments` 约有 200 个字段 | **112** 个 `field()` | 1A6 |
| `Trainer` 自己做混合精度 autocast | `autocast_smart_context_manager` 是**空壳**（`trainer.py:2081`），直接 `return contextlib.nullcontext()`；实际由 accelerate 接管 | 1A6 |
| GLM 的 DecoderLayer 直接继承 Llama | **5 层继承链**：`Llama → DeepseekV3 → Glm4Moe → Glm4MoeLite → DeepseekV32 → GlmMoeDsa`。骨架结论仍成立，但理由是「穿过整条链传下来的」 | 2M7 |
| Gemma 2 的层类型按 `layer_idx % 2` 判断 | 由 `__post_init__` 预计算的 `config.layer_types` 查表（`(i+1) % 2`，**第 0 层是局部层**） | 2M1 |

---

### 还有两条「不是错，但值得单独记」的实测发现

| 发现 | 出处 |
|------|------|
| **MLA 的省显存在 V3.2 / GLM 上被 DSA 抵消了**：cache 从 V3 的 **576 元素/token/层** 涨到 **40960**（涨 71 倍），因为要按 top-k 索引去 select 具体的 K/V 行，压缩潜向量没法直接选。代码自带 `TODO` 承认。这个取舍是 **DeepSeek V3.2 做的，不是 GLM 做的** | 2M2、2M7 |
| **仓库自己的规则文件和实际不符**：`utils/rules.toml` 的 TRF010 要求 config 用 `@strict(accept_kwargs=True)`，注释还点名 `nemotron_h` 和 `vibevoice_asr` 已合规——但这两个文件实际都是普通 `@strict`（`:26` / `:30`），全库 `accept_kwargs=True` 只有 1 处（基类），普通 `@strict` 有 752 处 | 1A2 |

