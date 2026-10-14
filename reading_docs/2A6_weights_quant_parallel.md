# 权重加载、量化与并行精读：当 checkpoint 和模型对不上时

> 阅读目标：搞清楚 `from_pretrained` 在**「正常情况下」之外**的三条追加路径——
> **名字/布局对不上怎么办**（权重映射）、**权重不是 fp16 怎么办**（量化）、**一张卡装不下怎么办**（张量/专家并行）。
>
> **这篇在 2A 线里的位置**：2A 的收官篇，也是 1A3 的「进阶版」。1A3 讲**顺利路径**（下载 → 建骨架 → 按 key 灌权重 → 补 missing keys）；
> 2A6 讲**不顺利路径**：key 名字不一样、一个 key 要拆成三个、一堆 key 要拼成一个、权重是 4bit 的、模型要横切到 8 张卡上。
> **三件事共用同一套管线**——本篇最重要的一句：**权重映射就是加载器本身**，量化和并行都是往这条管线里「插一层 op」。
>
> 源文件（全部实测过行数）：`core_model_loading.py`（1853 行）、`conversion_mapping.py`（2005 行）、`fusion_mapping.py`（270 行）、
> `quantizers/`（`base.py` 342 行 + `auto.py` 370 行 + 25 个 `quantizer_*.py`）、
> `integrations/tensor_parallel.py`（93 行，**shim**）、`distributed/tensor_parallel.py`（843 行，真实现）、
> `integrations/moe.py`（582 行）、`integrations/`（47 个 `.py` + 2 个子目录）
> 人类向文档：`docs/source/en/weightconverter.md`、`docs/source/en/fusion_mapping.md`、`docs/source/en/tensor_parallelism.md`、`docs/source/en/monkey_patching.md`

---

## 速读区（TL;DR）

**一句话结论**：1A3 的加载是「按名字装」，2A6 的加载是「**算出来**」——
`conversion_mapping.py` 用注册表描述「checkpoint 想变成什么」，`core_model_loading.py:1511`
用两趟循环（先归类、后计算）执行它；**权重映射就是加载器本身**，量化和并行都是往这条管线里「插一层 op」。

| # | 断言 | 出处（文件:行号） |
|---|------|------------------|
| 1 | 注册表规模：`_MODEL_TO_CONVERSION_PATTERN` 94 个继承关系 → `_build_checkpoint_conversion_mapping()` **93 个 key** | `conversion_mapping.py:42`、`:144` |
| 2 | **`WeightRenaming` 是主力**（621 处），`WeightConverter` 只有 55 处——绝大多数「对不上」只是纯改名 | `conversion_mapping.py`（`grep -c`） |
| 3 | `WeightConverter` 只允许 1:1 / 1:N / N:1；**N:N 必须用白名单** `_INTERNAL_MANY_TO_MANY_CONVERSIONS`（实测只有 Ernie 那 2 个类） | `core_model_loading.py:1152`、`:1147` |
| 4 | 改名发生在 **chain 的最后一环**；`MergeModulelist` 必须在 `Concatenate` **之前**（注释：needs to be first） | `MergeModulelist.get_target_pattern()` `:257`、`conversion_mapping.py:1005` |
| 5 | 映射查找是**三级**：类名 → `model_type` → `legacy`（兜底 `LayerNorm.gamma/beta`），**类名优先** | `get_model_conversion_mapping()` `:1902`、`:1995-1996` |
| 6 | 保存是反向的（`revert_weight_conversion()`）；**挂了 `quantization_operation` 或 TP 时直接 `raise ValueError`** | `core_model_loading.py:1798`、`:936-938` |
| 7 | `fusion_mapping.py` 方向**正好反过来**：改**运行时模型**成 fused（Conv3d→Linear），让**原始 checkpoint 照旧加载**；只有 `patch_embeddings` 一个家族 | `fusion_mapping.py:44`、`:94`、`:235` |
| 8 | `integrations/tensor_parallel.py`（93 行）与 `integrations/fsdp.py` 都只是 **shim**，真实现在 `distributed/`（843 行） | `integrations/tensor_parallel.py:14-17` |
| 9 | **量化是加载期行为**：三条路 A 预量化 / B on-the-fly / C 加载后 `dequantize()`；`requires_calibration=True` 的 **9 个后端必须用已量化 checkpoint** | `core_model_loading.py:1672-1679`、`quantizers/base.py:91-95` |
| 10 | on-the-fly 量化会**主动关掉异步加载**（否则后台线程把全精度张量先塞满显存）；`GLOBAL_WORKERS = min(4, os.cpu_count() or 4)` | `core_model_loading.py:1629-1636`、`:1233` |
| 11 | dequant **必须插在 merge/concat 之前**（per-block scale 否则找不回主人）→ `MergeModulelist`「isn't always `operations[0]`」 | `quantizers/base.py:297-309`、`core_model_loading.py:1705-1708` |
| 12 | **EP 用 all-reduce 合输出，不是 all-to-all**：`grep -rn "all_to_all" src/ docs/` → 零命中 | `distributed/tensor_parallel.py:320`、`:353`、`:558`、`:582` |
| 13 | QK-Norm 破坏 fuse：q/k/v 降级为 `colwise_gather_output`、o_proj 降级为 `rowwise_split_input` | `olmo2/modular_olmo2.py:69-78`（同注释还在 olmo3 / olmo_hybrid / flex_olmo） |
| 14 | TP 策略共 **19 个**（`ALL_PARALLEL_STYLES`）；且 `_global_mapping` **只有 torch.distributed 可用时才非空** | `distributed/tensor_parallel.py:785`、`:758` |

**最容易踩的坑 / 反直觉发现**

- **`fusion_mapping.py` 不是「把 fused 层拆成多个层」**，而是把运行时模型改成 fused 形态（`docs/source/en/fusion_mapping.md`：replace submodules at load time **while preserving the original checkpoint format**）。
- **`integrations/moe.py` 不是 EP 的分片逻辑**，而是 experts forward 的 4 种实现（`deepgemm`/`batched_mm`/`grouped_mm`/`sonicmoe`）；EP 分片在 `distributed/tensor_parallel.py`。
- **EP 不是教科书里的 all-to-all**：每卡持有全部 token、只算自己的专家、输出 all-reduce 相加——代价是「大部分是零」的 GEMM 浪费（`integrations/moe.py:128-131` 直言 "we pay the wasted GEMM compute"）。唯一例外是 DeepGEMM 的 MegaMoE 在 kernel 内部做 dispatch，但 **transformers 侧没有发起 all-to-all**。
- **`WeightConverter` 定义在 `core_model_loading.py:1152`**，不在 `conversion_mapping.py`（后者只是 import 进来用）。
- **25 个 quantizer 类 ≠ 26 个 key**（bnb 拆 4bit/8bit；`fp8`/`mxfp8` 共用同一个类）；而且 `param_needs_quantization()` 基类**默认返回 `False`**（默认不量化任何参数），`get_quantize_ops()` 基类直接 `raise NotImplementedError`。
- **没有 expert 顺序重排算子**：expert 维上只有 stack / split / transpose；checkpoint 的专家顺序与模型不一致时，当前代码里没有现成手段（作者如实标注）。
- **MLA 里 `k_rot` 不经过 `kv_b_proj`**，梯度是 partial 的，所以 `MlaKvAProjParallel` 要单独补 `_AllReduceBackward`，还**要求 config 上有 `qk_rope_head_dim`**，否则直接 raise（`distributed/tensor_parallel.py:372-410`）。
- 「量化 → 并行 → device_map → offload」这条顺序是从 `from_pretrained` 的**注入点先后**读出来的经验顺序，**仓库文档未明确背书**。

**读完应该能回答的问题**

- checkpoint 的 key 和模型对不上时，先过哪个函数？「真 unexpected」是哪一步之后才判定的？
- 为什么 FP8 的 dequant op 必须插在 `MergeModulelist` 之前？
- 显存不够时按什么顺序试？每一步的注入点在 `from_pretrained` 的哪一行？
- 为什么带 QK-Norm 的模型必须把 q/k/v 的 TP 策略降级成 gather？MLA 又难在哪？
- `fusion_mapping` 与 `conversion_mapping` 在「运行时布局 / checkpoint 布局」上各管哪一边？

---

## 前置知识

- [1A3_from_pretrained.md](./1A3_from_pretrained.md)：**本篇的前置**（§4.1 / §4.2 已逐行讲过 `_load_pretrained_model` 与主循环，本篇不重复）、[1M1_llama_structure.md](./1M1_llama_structure.md)：Llama 的参数名（例子对照用）。
- [2A4_rope_abstraction.md](./2A4_rope_abstraction.md)：§7.3「为什么 RoPE 让 fused QKV 拆不干净」的背景、[2A5_modular_code_generation.md](./2A5_modular_code_generation.md)：`_tp_plan = {"lm_head": "colwise_gather_output"}` 的来历、[1M2_mistral_sliding_window.md](./1M2_mistral_sliding_window.md)：Mixtral 的 MoE 结构（第一个 `WeightConverter` 案例靠它）。

---

## 一、先划清边界：1A3 到哪为止，2A6 从哪开始

`from_pretrained` 的代码在 `modeling_utils.py:3881`，中间有近 500 行。按「顺不顺利」切开：

| 阶段 | 1A3 讲（顺利路径） | 2A6 讲（追加路径） | 代码位置（实测） |
|------|-------------------|-------------------|-----------------|
| 解析 `quantization_config` | — | ✅ 选 quantizer | `quantizers/auto.py:334` `get_hf_quantizer()` |
| 决定 `device_map` | — | ✅ 多卡 / offload | `integrations/accelerate.py:338` `_get_device_map()` |
| 建骨架 | ✅ `cls(config)` on meta | ✅ quantizer 替换模块 | `modeling_utils.py:4349` |
| 取权重映射表 | — | ✅ **本篇第三节** | `conversion_mapping.py:1902` `get_model_conversion_mapping()` |
| 灌权重 | ✅ 「key 一一对应」的直觉模型 | ✅ **两趟循环 + 张量变换** | `core_model_loading.py:1511` `convert_and_load_state_dict_in_model()` |
| 补 missing / tie weights | ✅ `_init_weights` | ✅ 与量化/Tie 交互 | `modeling_utils.py:4394` → `:4550` `_finalize_model_loading()` |

**一句话**：1A3 里「checkpoint 的 key == 模型的 key」是**默认假设**；2A6 讲的全是这个假设不成立时的补救，以及「补救手段可以被量化和并行复用」。

> **⚠️ 先破一个直觉**：老版本是「遍历 state_dict，`param.copy_(tensor)`」；现代版本已换成 **`conversion_mapping` 驱动的匹配引擎**——每个 checkpoint key 先被「改名」，再被「收集」到一个 transform 对象里，最后由它**算出**目标张量（`core_model_loading.py:1511` 的 docstring 就是全篇最好的文档，作者手写了 5 个例子）。

---

## 二、三张表看懂全貌

| 子系统 | 输入 | 往管线里插什么 | 连接点（实测行号） |
|--------|------|---------------|------------------|
| **权重映射** | checkpoint 的 key 名字 / 布局 | 改 key 名 + `ConversionOps`（split/stack/transpose） | `WeightConverter.operations`（`core_model_loading.py:1152`） |
| **量化** | 4bit/8bit/fp8 权重 | ① 替换 `nn.Linear` 为量化模块；② 给 transform 挂 `quantization_operation`（`core_model_loading.py:1679`） | `HfQuantizer.preprocess_model()`（`quantizers/base.py:155`）、`get_quantize_ops()`（`:289`） |
| **张量/专家并行** | 一张卡装不下 | 把参数换成 `DTensor` 分片占位符 + 给 `forward` 装 input/output 变换 | `apply_tensor_parallelism()`（`distributed/tensor_parallel.py:797`） |

**入口 / 出口**：权重映射 —— 入口 `_MODEL_TO_CONVERSION_PATTERN`（`conversion_mapping.py:42`，94 个 model_type→base 的继承关系）→ `_build_checkpoint_conversion_mapping()`（`:144`，**93 个 key / 640 条 transform**），出口加载 `WeightTransform.convert()` / 保存 `revert_weight_conversion()`（`core_model_loading.py:1798`）；量化 —— `AUTO_QUANTIZER_MAPPING`（`quantizers/auto.py:75`，**26 个 key → 25 个类**）→ `HfQuantizer.postprocess_model()`（`quantizers/base.py:176`）；并行 —— `base_model_tp_plan` / `base_model_ep_plan`（config 类属性，**154 个文件**里有）→ `ALL_PARALLEL_STYLES`（`distributed/tensor_parallel.py:785`，**19 个策略**）。

**名字对照（本篇最容易搞混的地方）**：

| 你可能以为是 | 实际是 |
|-------------|--------|
| `integrations/tensor_parallel.py` 是 TP 实现 | **它是 shim**（93 行）。真实现在 `distributed/tensor_parallel.py`（843 行）。见 `integrations/tensor_parallel.py:14-17` 的 docstring |
| `integrations/fsdp.py` 是 FSDP 实现 | 同上，`"""Backward-compatible re-exports. Prefer ``transformers.distributed.fsdp``."""` |
| `integrations/moe.py` 是「专家并行」 | 它主要是**专家 forward 的多种实现**（deepgemm / grouped_mm / batched_mm / sonicmoe）。EP 的**分片逻辑**在 `distributed/tensor_parallel.py` |
| `WeightConverter` 在 `conversion_mapping.py` 里定义 | 定义在 `core_model_loading.py:1152`，`conversion_mapping.py` 只是 `import` 进来用 |

---

## 三、权重映射：`WeightRenaming` 与 `WeightConverter`

### 3.1 多态的根：`WeightTransform`（`core_model_loading.py:769`）

两个子类的公共父类，只做一件事：**把 source/target 模式编译成正则**。

```python
# core_model_loading.py:786  __init__ 的关键
self.source_patterns = source_patterns ; self.target_patterns = target_patterns
# target 的捕获组预处理（process_target_pattern :719）+ source 的 \1 解析回捕获组（process_source_pattern :753）
self.compiled_sources = re.compile("|".join(branches))     # :851
```

`*.` 会被展开成 `.*\.`（`:849`），所以 `layers.*.self_attn.q_proj` 就是「任意层号」。两个设计细节：**① `source_patterns` / `target_patterns` 一旦赋值就不能改**（`__setattr__`，`:856`），因为两者通过捕获组互相耦合；**② `scope_prefix`**（`:871` `_scoped_match`）让子模型的映射**只在它自己的子树里生效**（匹配前剥掉前缀、匹配后接回），由 `get_model_conversion_mapping()`（`conversion_mapping.py:1902`）为每个子模块设置。

### 3.2 `WeightRenaming`（`core_model_loading.py:992`）：只改名，不碰张量

`convert()` 只把收集到的张量挂到新 key 上（`materialize_tensors()` → `collected_tensors`）。**它是主力**：`conversion_mapping.py` 里 **621 行**含 `WeightRenaming(`，而 `WeightConverter(` 只有 **55 处**（`grep -c` 实测）——绝大多数「对不上」只是纯改名。最简单的真实例子（`conversion_mapping.py:711-712`）：`"CLIPVisionModel": [PrefixChange(prefix_to_remove="vision_model")]`、`"CLIPTextModel": [PrefixChange(prefix_to_remove="text_model")]`。

### 3.3 `WeightConverter`（`core_model_loading.py:1152`）：改名 + 张量变换

```python
class WeightConverter(WeightTransform):        # core_model_loading.py:1155
    __slots__ = ("operations", "force_cpu")
    # 只允许 1:1 / 1:N / N:1，除非用了内部专门的 op
    if bool(len(self.source_patterns) - 1) + bool(len(self.target_patterns) - 1) >= 2:
        if not any(isinstance(op, _INTERNAL_MANY_TO_MANY_CONVERSIONS) for op in self.operations):
            raise ValueError("... you can only have one to many, one to one or many to one.")
```
**基数限制是硬约束**：`N:1`（多 key 合成一个，如 mixtral expert 堆叠）、`1:N`（一个 key 拆多个，如 fused QKV 拆 q/k/v）、`N:N` **只允许 `_INTERNAL_MANY_TO_MANY_CONVERSIONS`**（`core_model_loading.py:1147`）——实测只有 `ErnieFuseAndSplitTextVisionExperts`（`:596`）与 `ErnieSplitAndDecoupleTextVisionExperts`（`:657`）。
`convert()`（`:1175`）的执行顺序：`materialize_tensors()`（Future → 真张量，异步加载的收口点）→ `for op in self.operations: op.convert(...)`（链式，前一个的输出是后一个的输入）→ 把 target 短名展开成完整 key → 若挂了 `quantization_operation` 再跑一遍。

### 3.4 `ConversionOps`：真实存在的张量算子全集

`ConversionOps`（`core_model_loading.py:81`）是 ABC，只有 `convert()` + `reverse_op` 两个抽象成员。实测这一类里**一共 15 个类**（1 个 ABC + 14 个具体算子，其中两个 `Vision*ForRope` 合并成一行）：

| 算子 | 行号 | 干什么 | 反向算子 |
|------|------|--------|----------|
| `_IdentityOp` | `:101` | 直通（给反量化用，权重已是目标 dtype） | — |
| `Chunk` / `Concatenate` | `:112` / `:150` | 沿 `dim` 均分（**1:N**）/ 拼接（**N:1**） | 互为反向 |
| `Interleave` | `:195` | 沿 `dim` 拆两半后转置 → 去交错（gate/up 交错布局 → 拼接布局） | `Interleave(inverse=True)` |
| `MergeModulelist` / `SplitModulelist` | `:222` / `:273` | `torch.stack` 成一块（**N:1，expert 堆叠专用**）/ `torch.chunk` 拆回 | 互为反向 |
| `Transpose` | `:312` | 交换两个维度（可选 `check_dims` 校验） | 同（交换回去） |
| `Conv3dToLinear` / `LinearToConv3d` | `:362` / `:402` | Conv3d patch embedding ↔ 等价的 Linear | 互为反向 |
| `PermuteForRope` | `:428` | **complex-RoPE 权重 → split sin/cos 布局的置换** | 自带 `inverse` |
| `VisionFuseAndPermuteForRope` | `:488` | `PermuteForRope` + `Concatenate` 的合并版（只对子集做置换） | `VisionUnfuseAndPermuteForRope` `:542` |
| `ErnieFuseAndSplitTextVisionExperts` / `ErnieSplitAndDecoupleTextVisionExperts` | `:596` / `:657` | text/vision 两套 expert 的 fuse+split（**N:N**）/ 其反向 | 互为反向 |

`MergeModulelist` 的 docstring：「Merge a list of tensors into a single tensor along the first dimension. **We explicitly define this because for EP or TP you want to make sure you know what you are doing!**」

### 3.5 五个真实案例

**案例 A：N 个独立 expert 张量 → 一块 3D `nn.Parameter`（`mixtral`）**

Mixtral 的 HF checkpoint 把每个专家存成独立 key（`model.layers.N.block_sparse_moe.experts.{i}.w1.weight`，8 专家 × 3 投影 = 24 key/层），但模型里 `experts` 是**打包成 3D 张量**的（`gate_up_proj` / `down_proj` 各一块 `nn.Parameter`）。解法（`conversion_mapping.py:992-1018`）：

```python
"mixtral": [
    WeightRenaming(".block_sparse_moe.", ".mlp."),        # :993  纯改名，先做
    WeightConverter(
        source_patterns=[".experts.*.w1.weight", ".experts.*.w3.weight"],  # 2 source
        target_patterns=".experts.gate_up_proj",                           # 1 target → N:1
        operations=[MergeModulelist(dim=0),   # 各 source 的 8 个张量各自 stack → 2 个 (8, I, H)
                    Concatenate(dim=1)],      # 拼成 (8, 2I, H)；同时完成「改名」
    ),
    WeightConverter(source_patterns=[".experts.*.w2.weight"], target_patterns=".experts.down_proj",
                    operations=[MergeModulelist(dim=0)]),   # 只需 stack
],
```

两个必须记住的机制点：**① 改名发生在 chain 的最后一环**——`MergeModulelist` 之后 key 还是 source 模式（它要处理一整组 key，不能提前收窄），只有 `Concatenate`/单 op 结束时才落到 target 名（规则写在 `MergeModulelist.get_target_pattern()`，`:257` 的注释里）；**② 顺序不能颠倒**——`MergeModulelist` 必须在 `Concatenate` 之前（`conversion_mapping.py:1005` 原文注释「we can't shard after concats and merge, needs to be first」）。

同一套写法被**复用了 9 次**（精确字符串 `MergeModulelist(dim=0), Concatenate(dim=1)` 命中 9 行；`MergeModulelist(dim=0)` 总计 **20 处**），覆盖 9 个模型 key：`axk2`（`:236`）、`deepseek_v4`（`:483`）、`glm5_next`（`:535`）、`minimax_m3_vl`（`:590`）、`deepseek_ocr2`（`:759`）、`qwen2_moe`（`:1019`）、`phimoe`（`:1067`）、`lfm2_moe`（`:1084`）、`jamba`（`:1287`）。
`qwen2_moe` 的映射（`:1019-1032`）是**继承枢纽**：`_MODEL_TO_CONVERSION_PATTERN`（`:42`）里有 **22 个 model_type 指向 `"qwen2_moe"`**（`deepseek_v2` / `deepseek_v3` / `glm4_moe` / `glm_moe_dsa` / `qwen3_moe` / `olmoe` / `hunyuan_v1_moe` …），最后在 `:1836-1839` 用 `mapping[model_type] = mapping[base_pattern].copy()` 展开。

**案例 B：一块 fused QKV → q/k/v 三份（`Chunk`）** —— 三个实测例子：`radio`（`conversion_mapping.py:289-297`，`attn.qkv` → `attention.attention.{query,key,value}`）、`Tipsv2VisionModel`（`:1603`，converter 在 `:1619-1627`，`.attn.qkv.weight` → 三份 weight）、`hrm_text`（`:299`，converter 在 `:305-314`，`attn.gqkv_proj.weight` → **4 份**：`gate_proj` / `q_proj` / `k_proj` / `v_proj`），operations 都是 `[Chunk(dim=0)]`。
`hrm_text` 是 1:4，最能说明语义：`torch.chunk(tensor, num_shards, dim=self.dim)`（`core_model_loading.py:130`），**切几份完全由 target 个数决定**。

**案例 C：先置换再拆 —— RoPE 布局转换（`kimi_k25`）**

「fused QKV **而且** RoPE 用 complex 布局」：checkpoint 里 q/k 权重按 `(real, imag)` 交错排，模型里要「前半 sin、后半 cos」的 split 布局。解法（`conversion_mapping.py:469-481`）：

```python
# conversion_mapping.py:469-481  Unfuse qkv and apply rope permutation
WeightConverter(source_patterns=[r"wqkv"], target_patterns=[r"attn.q_proj", r"attn.k_proj", r"attn.v_proj"],
    operations=[Chunk(dim=0),   # 先拆成三份
        PermuteForRope(subconfig_key="vision_config",               # ← vision_config 的 num_attention_heads
                       permute_layer_names=["q_proj", "k_proj"])])  # ← 再对 q/k 置换，v 不置换
```

`PermuteForRope`（`core_model_loading.py:428`）的核心（`:445-451`）：`head_shape = (2, half_head) if self.inverse else (half_head, 2)` → `tensor.view(n_heads, *head_shape, ...)`（`:450`）→ `transpose(1, 2).reshape(dim0, ...)`（`:451`）。
**这就是「RoPE 不能直接 fuse」的第一个硬证据**：RoPE 让 Q/K 的**权重内存布局**和 V 不同类，拆 fused QKV 时**不能只 Chunk**。同一套 `Chunk + PermuteForRope` 实测只有 **2 处**：`kimi_k25`（`:469-481`，key 在 `:430`，1→3，只对 q/k 置换）与 `step3p5_vision`（`:903-927`，key 在 `:886`，weight 和 bias 各置换一次）。
**反例**：`Tipsv2VisionModel`（`:1603`）的 `attn.qkv.weight`（`:1619-1627`）和 `attn.qkv.bias`（`:1628-1636`）就是**纯 `Chunk(dim=0)`，没有 `PermuteForRope`**（原因不在这份映射文件里，**不做推断**）。结论：**「拆 fused QKV 要不要附带置换」是逐模型的。**

**案例 D：只转置（`qwen3_vl_moe`，`conversion_mapping.py:1034-1044`）** —— `source_patterns` 与 `target_patterns` **同名**（都是 `mlp.experts.gate_up_proj`），`operations=[Transpose(1, 2, check_dims=True)]`。同名不是笔误：模型侧形状写死在 `models/qwen3_vl_moe/modeling_qwen3_vl_moe.py:97`（`nn.Parameter(torch.empty(num_experts, 2 * intermediate_dim, hidden_dim))`），checkpoint 那边只能是 `(E, hidden, 2I)`，差一个 `transpose(1, 2)`；`check_dims=True` 让维度不符时**报错而不是静默出错**。`down_proj` 同理（`:1040-1044`，模型形状见同文件 `:98`）。

**案例 E：N:N 的唯一用法（`ernie4_5_vl_moe`，`:1099`）** —— text MoE 与 vision MoE 在 checkpoint 里**共用一组 expert key**，模型里是两套独立参数，所以用了真正的 2→2（`conversion_mapping.py:1139-1149`）：

```python
# conversion_mapping.py:1139-1149
WeightConverter(
    source_patterns=["experts.*.gate_proj.weight", "experts.*.up_proj.weight"],   # 2
    target_patterns=["text_moe.experts.gate_up_proj", "vision_moe.experts.gate_up_proj"],  # 2
    operations=[ErnieFuseAndSplitTextVisionExperts(stack_dim=0, concat_dim=1)])
```

**这是仓库里唯一真正用到 2→2 的地方**（`grep -o 'ErnieFuseAndSplitTextVisionExperts('` 实测 2 处，另一处 `:1131-1138` 是 1→2 的 `down_proj`），靠 `_INTERNAL_MANY_TO_MANY_CONVERSIONS`（`core_model_loading.py:1147`）的白名单机制。

> **一句实话**：我特意找了「MoE expert **顺序重排**」这种需求，在 `conversion_mapping.py` 里**没有找到**——expert 维上仓库只做 stack / split / transpose，**没有 expert 顺序的置换算子**（以仓库实际代码为准）。

### 3.6 两个 `WeightRenaming` 子类

- `PrefixChange`（`core_model_loading.py:1098`）：「整块前缀加/删」，**11 处**：`:711` `CLIPVisionModel` 删 `vision_model`、`:751` 加 `language_model`、`:929` `colqwen2` 删 `model`、`:930` `shieldgemma2` 加 `model`、`:931` `timm_wrapper` 加 `timm_model`、`:978` `qwen3_5_text` 删 `language_model`、`:1819-1820` 删 `model` / `mtp`。
- `GroupWeightRename`（`:1032`）：**链式改名会重复施加**（`norm0→norm1` 且 `norm1→norm2`，加载**已转换过**的 checkpoint 时第二条会被误触发）。**2 处**，都在 `kimi_k25`（`:436-455`）。机制是「守卫模式」：**只有第一个（guard）source 出现过，后续 rename 才激活**（`rename_source_key`，`:1056`）。docstring 的关键约束：「order `source_patterns` so that the one that is absent in an already-converted checkpoint comes first. The state dict is iterated in sorted key order, so the guard pattern must be lexicographically smaller than the dependent patterns.」——state_dict 是**按 key 排序**遍历的。

### 3.7 映射是按什么注册的？（三级查找，实测）

```python
# conversion_mapping.py:144
def _build_checkpoint_conversion_mapping():
    mapping = { "cosmos3_edge": [...], "axk2": [...], ... }   # 93 个 key
    for model_type, base_pattern in _MODEL_TO_CONVERSION_PATTERN.items():   # :1836
        if model_type in mapping: continue
        mapping[model_type] = mapping[base_pattern].copy()   # :1839
    return mapping
```

**注册表名**：`_checkpoint_conversion_mapping_cache`（`conversion_mapping.py:1844`，模块级懒加载缓存，`deepcopy` 出去防止污染，见 `:1851`）。**查找顺序**（`get_model_conversion_mapping()`，`:1902`）——本篇关于「名字对不上那一层」最关键的答案：

```
for module_name, submodule in model.named_modules():
    1. 是 PreTrainedModel 吗？不是就跳过
    2. 是 remote code（非用户注册）吗？是就跳过（防止同名 model_type 撞车）
    3. 祖先模块已经认领过这个类名吗？是就跳过
    4. 按【类名】查（优先！允许 ForXxx 级覆盖）  get_checkpoint_conversion_mapping(class_name)
    5. 没查到 → 按【model_type】查
    6. 子模块 → 给每条 transform 设 scope_prefix，只在子树内生效
    7. 最后追加 get_checkpoint_conversion_mapping("legacy")  ← :1995-1996  通用兜底（LayerNorm.gamma/beta → weight/bias，映射表在 :1399）
    8. 交给 quantizer 改写：weight_conversions = hf_quantizer.update_weight_conversions(...)  ← :2003
```

**三级 = 类名 → model_type → legacy**。类名优先是为了让 `LlavaModel` 嵌在 `LlavaForConditionalGeneration` 里时还能拿到自己的 scoped 映射（注释 `:1957-1961`）。**用户扩展 API**（`:1857`）：`register_checkpoint_conversion_mapping(model_type_or_class_name, mapping, overwrite=False)`，会记进 `USER_REGISTERED_MAPPINGS`（`:1854`）。

> **保存是反向的**：`revert_weight_conversion()`（`core_model_loading.py:1798`）把每条 transform `reverse_transform()`（`:934`）——source/target 对调，`operations` 用各自的 `reverse_op` 反向排列。触发点是 `save_pretrained(save_original_format=True)`（`modeling_utils.py:3310` / `:3520`）。**不支持反向的情况**：挂了 `quantization_operation` 或 TP 时会直接 `raise ValueError`（`:936-938`）。

---

## 四、`core_model_loading.py`：名字对不上时那一层到底在哪

### 4.1 与 1A3 的衔接

`from_pretrained()`（`modeling_utils.py:3881`）里，2A6 相关的步骤依次是：`get_hf_quantizer`（`:4270`）→ config 解析 / fusion 注册 / kernel 注册（`:4331-4341`）→ `cls(config)` meta 骨架（`:4350`，1A3）→ `hf_quantizer.preprocess_model`（`:4354`）→ `_get_dtype_plan`（`:4364`）→ `get_model_conversion_mapping`（`:4367`）→ `maybe_distribute_model`（`:4369`，TP/FSDP/PP）→ `_get_device_map`（`:4373`）→ `LoadStateDictConfig(..., weight_mapping=...)`（`:4377`）→ `cls._load_pretrained_model`（`:4393`）→ `convert_and_load_state_dict_in_model`（`core_model_loading.py:1511` ★）。

**`_load_pretrained_model`（`modeling_utils.py:4436`，调用点在 `:4393`）只负责「准备」**：读 checkpoint 文件（safetensors 用 `safe_open` 拿 `get_slice`，**故意不 materialize**，`:4527`）、`verify_tp_plan()` 检查策略名/覆盖率（`:4455`）、`accelerate_disk_offload()`（`:4461`）、`caching_allocator_warmup()`（`:4473`，函数定义在 `:5071`）、DeepSpeed ZeRO-3 独立分支（`:4477`，那条路**不做** conversion）。**真正的转换发生在 `convert_and_load_state_dict_in_model()`**（`core_model_loading.py:1511`）。

### 4.2 这个函数在干什么（两趟循环）

主循环骨架（`sorted(..., dot_natural_key)` → `rename_source_key` → `param_name_to_load.setdefault` → `spawn_materialize` → `mapping.convert`）**1A3 的 §4.2 已逐行讲过**，这里只补三件事。

**（1）`rename_source_key()`（`:1412`）——「名字对不上」的唯一入口**：加载（不 reverse）时先跑【全部】`WeightRenaming`（可链式多条），再跑【至多一条】`WeightConverter`（首个匹配即停）；保存（reverse）时顺序完全反过来。docstring 的理由：「there is no need for a Converter-then-Rename order because Converters act only on specific leaf patterns, so no subsequent Renamings should ever target their output.」第 3 步还有前缀修补（`:1472-1481`）：改名后若 meta state dict 里有带 / 不带 `base_model_prefix` 的两个版本，就自动补上或去掉一级前缀——这是「用 BaseModel 权重加载 ForXXX 模型」能工作的原因。

**（2）这一趟里还有 4 件与量化 / 并行相关的动作**：dtype 决策（`:1681-1703`，量化反序列化时保持 checkpoint dtype，否则按 `dtype_plan` / meta 参数 dtype）；量化算子挂载（`:1672-1679`，`param_needs_quantization()` 为真 → `mapping.quantization_operation = get_quantize_ops()`）；EP expert 索引（`:1705-1714`，op chain 里有 `MergeModulelist` → 传 `tensor_idx`，让 DTensor 按**整个 expert** 切；注释说它「isn't always `operations[0]`」，因为 FP8 quantizer 会在链头插 scale-decode op）；分片 / offload 决策（`:1716-1728`，meta 参数是 `DTensor` → `DtensorShardOperation`；`force_cpu=True` → 先落 cpu 再交给 accelerate，例子是 ~95 GiB 的 `qwen4_exp` 巨型 embedding）。

**（3）异步加载在这里收口，且有 3 个「退化回同步」的条件**：`spawn_materialize()`（`:1244`）把「读盘 + cast + shard」塞进 `ThreadPoolExecutor`（`GLOBAL_WORKERS = min(4, os.cpu_count() or 4)`，`:1233`，注释说 16 线程反而会慢一倍）。退化条件（`:1629-1636`）：① 环境变量 `HF_DEACTIVATE_ASYNC_LOAD` 为真；② `device_map` 含 `"disk"`；③ **on-the-fly 量化**（否则后台线程会把全精度张量先塞满显存，主线程来不及量化）。
第 3 条是量化和加载管线的第一个真实耦合点；第二个是第二趟：`mapping.convert()` 里**先跑完所有张量变换，最后才跑量化 op**（`WeightConverter.convert()` 里 `if hf_quantizer is not None and self.quantization_operation is not None:` 分支，`:1214-1224`）。

### 4.3 流程图

```
checkpoint 文件 (safetensors) → safe_open(...).get_slice(k)（不 materialize，只拿 slice 对象）→ merged_state_dict
第一趟（归类）:  sorted(by dot_natural_key) → rename_source_key(original_key)  :902 正则 match
  ① 全部 WeightRenaming（可链式） ② 首个匹配的 WeightConverter → (renamed_key, src_pat) ③ base_model_prefix 加/删
  renamed_key ∈ meta_state_dict ?  是 → param_name_to_load[key] = deepcopy(pattern_to_converter[src_pat])
      + add_tensor + 挂 quantization_operation/tensor_idx/sharding_op + spawn_materialize()（线程池读盘+cast+shard）
    否 → _add_unmatched_checkpoint_key()  :1485（PP 其它 stage 的 key → skipped_pp_keys，不算 unexpected）
第二趟（计算）:  materialize_tensors()（Future → Tensor）→ for op: op.convert(...)（Chunk/stack/transpose…）
  → if quantization_operation: 再跑量化/反量化 → set_param_for_module(model, target_name, param)  :1343（或 offload_and_maybe_resave_param → disk）
  ↓ loading_info = {missing, unexpected, mismatched, conversion_errors, skipped_pp_keys} → _finalize_model_loading()  modeling_utils.py:4394
```

### 4.4 与 1A3 的分工（避免重复）

| 问题 | 1A3 的答案 | 2A6 的答案 |
|------|-----------|-----------|
| key 对不上会怎样？ | 「报 unexpected_keys」 | 先过 `rename_source_key()`，**真对不上的才报**（`_add_unmatched_checkpoint_key`，`:1485`） |
| missing_keys 是什么？ | 「模型里有、checkpoint 没有的参数」 | 初始化起点是**全部** meta 参数（`:1616`），每匹配一个删一个——所以 missing 是「这条管线没覆盖到的」 |
| `_init_weights` 何时跑？ | 「加载完补 missing」 | 只在 `_finalize_model_loading()`（`:4550`）里对**没打上 `_is_hf_initialized` 标记**的参数跑，且会先 `mark_tied_weights_as_initialized()` 跳过 tie 权重 |

---

## 五、`fusion_mapping.py`：方向恰好**反过来**

**实测结论（读了全文）**：它**不是**「把 fused 层拆成多个层」，而是**反方向**——**把运行时模型改成 fused 形态**，然后注册权重映射让**原始的、未 fuse 的 checkpoint 仍能加载**。

```python
# fusion_mapping.py:44  ModuleFusionSpec 的三个抽象点
class ModuleFusionSpec:
    target_modules_patterns: tuple[str, ...] = ()
    def is_fusable(self, module) -> bool: ...             # 这个模块能不能 fuse
    def make_fused_class(self, original_cls) -> type: ...  # 造运行时替换类
    def make_transforms(self, config) -> list[WeightTransform]: ...  # 原始布局 ↔ fused 布局的映射
# fusion_mapping.py:94  PatchEmbeddingsFusionSpec（唯一的 fusion 家族，_FUSION_REGISTRY :235）
#   target_modules_patterns = (r"(^|\.)patch_embed$",)
#   is_fusable: isinstance(proj, nn.Conv3d) and proj.stride == proj.kernel_size and proj.padding == (0,0,0)
#               and proj.dilation == (1,1,1) and proj.groups == 1
```

即**patch 之间不重叠的 Conv3d patch embedding → 等价的一个 `nn.Linear`**（`_FusedPatchEmbeddingMixin`，`:71`；运行时 `self.linear_proj` 取代 `self.proj`，`:77-85`）；配套权重映射是 `Conv3dToLinear`（`:129`）+ 一个 `WeightRenaming` 给 bias（`:135`）。**注册流程**（`_register_module_fusion()`，`:190`）：① `_discover_fusable_modules`（`:142`）在 `torch.device("meta")` 上实例化一遍、扫 `named_modules()`，按 pattern 预筛 → `is_fusable()` 结构判定（结果缓存进 `_FUSION_DISCOVERY_CACHE`，`:41`）；② `register_patch_mapping(...)`（交给 `monkey_patching.py:85`）；③ `spec.make_transforms(config)`；④ source_patterns 撞车 → raise（`:223`）；⑤ `register_checkpoint_conversion_mapping(model_type, existing + converters, overwrite=True)`。

**调用点唯一**：`from_pretrained`（`modeling_utils.py:4333-4335`），由 `fusion_config` 参数或 `config.fusion_config` 触发，默认不开（`:266`）。它必须在 `:4349` 的 `with ContextManagers(...)` **之前**，因为 `monkey_patching.py`（371 行）负责「在模型实例化**之前**把类换掉」（`register_patch_mapping` `:85`、`apply_patches` `:234`）。

| 方向 | 谁 | 做什么 | 运行时布局 | checkpoint 布局 |
|------|----|--------|-----------|----------------|
| 常规映射 | `conversion_mapping.py` | checkpoint → 模型 | 模型说了算（如 fused QKV 拆开） | 任意 |
| **fusion** | `fusion_mapping.py` | 模型 → fused 运行时 | **被改成 fused**（Conv3d→Linear） | **保持原样** |

文档 `docs/source/en/fusion_mapping.md`（99 行）明确写了「replace model submodules at load time **while preserving the original checkpoint format**」；生成文件时用了 `# TODO: allow compatible fusions`（`:229`），所以当前只支持一个家族。

---

## 六、量化

### 6.1 目录全景：实测 25 个 quantizer 类 + 26 个注册 key

`ls src/transformers/quantizers/` → `__init__.py`、`auto.py`、`base.py`、`quantizers_utils.py` + **25 个 `quantizer_*.py`**。

**`requires_calibration = True` 的 9 个**（**必须**用已量化好的 checkpoint，`pre_quantized=False` 会在 `HfQuantizer.__init__`（`base.py:91-95`）直接 raise）：
`quantizer_aqlm.py` `AqlmHfQuantizer`(aqlm)、`quantizer_auto_round.py` `AutoRoundQuantizer`(auto-round)、`quantizer_awq.py` `AwqQuantizer`(awq)、`quantizer_bitnet.py` `BitNetHfQuantizer`(bitnet)、`quantizer_compressed_tensors.py` `CompressedTensorsHfQuantizer`(compressed-tensors)、`quantizer_gemma.py` `GemmaQuantizer`(gemma)、`quantizer_quark.py` `QuarkHfQuantizer`(quark)、`quantizer_spqr.py` `SpQRHfQuantizer`(spqr)、`quantizer_vptq.py` `VptqHfQuantizer`(vptq)。

**其余 16 个 `requires_calibration = False`**（可走 on-the-fly）：
`quantizer_bnb_4bit.py` `Bnb4BitHfQuantizer`(bitsandbytes_4bit)、`quantizer_bnb_8bit.py` `Bnb8BitHfQuantizer`(bitsandbytes_8bit)、`quantizer_eetq.py` `EetqHfQuantizer`(eetq)、`quantizer_fbgemm_fp8.py` `FbgemmFp8HfQuantizer`(fbgemm_fp8)、`quantizer_finegrained_fp8.py` `FineGrainedFP8HfQuantizer`(**`fp8` 和 `mxfp8` 共用**)、`quantizer_fouroversix.py` `FourOverSixHfQuantizer`(fouroversix)、`quantizer_fp_quant.py` `FPQuantHfQuantizer`(fp_quant)、`quantizer_gptq.py` `GptqHfQuantizer`(gptq)、`quantizer_higgs.py` `HiggsHfQuantizer`(higgs)、`quantizer_hqq.py` `HqqHfQuantizer`(hqq)、`quantizer_metal.py` `MetalHfQuantizer`(metal)、`quantizer_mxfp4.py` `Mxfp4HfQuantizer`(mxfp4)、`quantizer_nvfp4.py` `NVFP4HfQuantizer`(nvfp4)、`quantizer_quanto.py` `QuantoHfQuantizer`(quanto)、`quantizer_sinq.py` `SinqHfQuantizer`(sinq)、`quantizer_torchao.py` `TorchAoHfQuantizer`(torchao)。

**25 个类 ≠ 26 个 key**：`bitsandbytes` 一个 config 拆成 4bit/8bit 两个 key（`AutoHfQuantizer.from_config()`，`auto.py:207-215`），`fp8`/`mxfp8` 共用 `FineGrainedFP8HfQuantizer`（`auto.py:96-99` 注释说明 MXFP8 复用同一条 dequant 路径，只是 E8M0 字节解包在 `Fp8Dequantize._dequantize_one` 里）。`utils/quantization_config.py:43-68` 的 `QuantizationMethod` 枚举有 **25 个成员**（另外 `fouroversix` 对应 `FOUR_OVER_SIX`、`auto-round` 对应 `AUTOROUND`）。

### 6.2 `HfQuantizer` 基类要求子类实现什么

`quantizers/base.py:73`。**实测真正的 `@abstractmethod` 只有两个**（`grep -n abstractmethod` 只命中 `:14` import、`:269`、`:273`）：`is_serializable`（`:269`）与 property `is_trainable`（`:272`）。其余全是**有默认实现的钩子**（覆盖才生效），按加载时序排：

| 钩子 | 行号 | 默认行为 / 何时被调用 |
|------|------|---------|
| `requires_calibration`（类属性）/ `validate_environment` | `:83` / `:136` | `False`（`__init__` 里 `not pre_quantized and requires_calibration` → raise，`:91-95`）/ `return`（`auto.py:358-361` **和** `_get_device_map()` 里再来一次，`accelerate.py:373`） |
| `update_dtype` / `update_device_map` / `param_element_size` / `adjust_max_memory` | `:99` / `:111` / `:123` / `:126` | 默认原样返回；bnb 的 `None` → 当前 GPU（`quantizer_bnb_4bit.py:102-115`）；bnb-4bit 的 `param_element_size` 返回 `0.5`（`:83-89`）、max_memory 乘 0.90（`:97-100`）——**device_map 自动规划才能算对显存** |
| `update_tp_plan` / `update_ep_plan` / `get_weight_conversions` / `update_weight_conversions` | `:144` / `:148` / `:294` / `:297` | 原样返回 / `[]` / `weight_conversions + self.get_weight_conversions()`；给 scale/block 参数补并行规则（实测 `quantizer_mxfp4.py:213-224` 给 GptOss 加 4 条 `grouped_gemm`，EP 版 `:226-237`）；调用点 `conversion_mapping.py:2003`——**quantizer 能改写整条映射管线** |
| `preprocess_model` / `_process_model_before_weight_loading` / `_convert_model_for_quantization` | `:155` / `:152` / `:276` | 设 `is_quantized` / `quantization_method`；若 `pre_quantized` 先 `_convert_model_for_quantization()`（按 `MODULES_TO_PATCH_FOR_QUANTIZATION` 替换模块）；再调 `_process_model_before_weight_loading()`（bnb 在这里把 `nn.Linear` 换成 `bnb.nn.Linear4bit`，`quantizer_bnb_4bit.py:121-143`）；调用点 `modeling_utils.py:4354` |
| `param_needs_quantization` / `get_quantize_ops` | `:130` / `:289` | **`False`**（默认不量化任何参数）/ `raise NotImplementedError`；调用点 `core_model_loading.py:1673`，决定是否挂 `quantization_operation` |
| `postprocess_model` / `_process_model_after_weight_loading` | `:176` / `:173` | 写 `config.quantization_config`；`pre_quantized and dequantize` → `remove_quantization_config()`，否则 `_assign_is_quantized()`；再调 `_process_model_after_weight_loading()`（bnb 设 `is_loaded_in_4bit`，`quantizer_bnb_4bit.py:145-148`）；调用点 `modeling_utils.py:4417` |
| `is_qat_trainable` / `is_compileable`（property）/ `dequantize` / `_dequantize` / `get_state_dict_and_metadata` | `:256` / `:261` / `:208` / `:222` / `:265` | `False`（训练 / `torch.compile` 前检查）/ 走 `_dequantize` / `None, {}`（保存时） |

**`update_weight_conversions` 的 docstring 是本节最值得读的一段**（`:297-309`）：「Loading runs ``renamings → converters → (dequant → merge → concat)``. Dequant has to happen *before* any merge/concat op because those operations aren't aware of per-block scales, so the per-expert (weight, scale) pairs need to be collapsed into full-precision tensors first.」
**翻译**：FP8 checkpoint 里每个 expert 的权重旁边还有一块 scale。如果先 `MergeModulelist` 把 8 个 expert stack 起来，scale 就找不回自己的主人了。所以 dequant op **必须被插到 chain 的最前面**（`core_model_loading.py:1705-1708` 的注释也说了同一件事：`MergeModulelist`「isn't always `operations[0]`」）。

### 6.3 一次量化加载的全流程（ASCII）

```
from_pretrained(quantization_config=BitsAndBytesConfig(load_in_4bit=True), ...)
  │ ① 选 quantizer + 改环境：get_hf_quantizer(...)  quantizers/auto.py:334
  │    config 已有 quantization_config？（= pre_quantized）→ supports_quant_method() :282；merge_quantization_configs() :232
  │    AutoHfQuantizer.from_config() :198 → bnb 按 load_in_8bit 补后缀（:207-215）→ AUTO_QUANTIZER_MAPPING[quant_method](cfg, pre_quantized=...)  :223-224
  │    validate_environment  auto.py:358 ; update_device_map :362 ; update_tp_plan :363 ; update_ep_plan :364
  │ ② 建骨架 + 改模型（权重还是 meta）：copy.deepcopy(config)  modeling_utils.py:4348 ; with ContextManagers(model_init_context): :4349
  │    model = cls(config) ← meta 骨架 ; hf_quantizer.preprocess_model(...) :4354 → base.py:155
  │      is_quantized = True ; if pre_quantized: _convert_model_for_quantization  base.py:276 ; _process_model_before_weight_loading  base.py:152
  │      → bnb replace_with_bnb_linear  quantizer_bnb_4bit.py:138
  │ ③ 取权重映射：get_model_conversion_mapping(...)  conversion_mapping.py:1902 → :2003
  │      （FP8 dequantize 模式：给每条 WeightConverter 的 operations 头部插 Fp8Dequantize）
  │ ④ 加载：cls._load_pretrained_model(...)  modeling_utils.py:4393 → :4436 → convert_and_load_state_dict_in_model  core_model_loading.py:1511
  │      param_needs_quantization? True → mapping.quantization_operation = ... :1678 ; has_on_the_fly_quantization → 禁用异步线程 :1628
  │      第二趟：mapping.convert() 里先跑张量变换，再跑量化 op
  │ ⑤ 后处理：accelerate_dispatch  modeling_utils.py:4413 ; postprocess_model  :4416-4419 → base.py:176 ; return model
```

### 6.4 「量化只在加载时发生」还是「运行时也能量化」？

**实测答案：主路径是「加载时」。但二分法不准，准确的划分是三条路**：

| 路径 | 权重来源 | 触发 | 证据 |
|------|---------|------|------|
| **A. 加载预量化 checkpoint** | checkpoint 里就是 4bit/fp8 | config 里有 `quantization_config` → `pre_quantized=True` | `auto.py:334-345` |
| **B. 加载时 on-the-fly 量化** | checkpoint 是全精度 | `param_needs_quantization()` 返回 True → `get_quantize_ops()`，在灌权重那一刻量化 | `core_model_loading.py:1672-1679`；`base.py:289` |
| **C. 加载后反量化** | 加载好的量化模型 | `HfQuantizer.dequantize(model, dtype)` | `base.py:208` / `_dequantize` `:222`；bnb 实现见 `quantizer_bnb_4bit.py:157-161` |

**关键点**：路径 B（`requires_calibration=False` 的那 16 个后端才能走）**依然发生在 `from_pretrained` 内部**，不在 forward 里，而且会主动**关掉异步加载**（`core_model_loading.py:1629-1636`，注释原文：「we also use sync loading to avoid worker threads loading full-precision tensors to GPU faster than the main thread can quantize them, which would cause a large memory spike」）。**边界说明**：`src/transformers/` 里没有任何「加载完之后再对模型调一次 `quantize(model)`」的公开 API（`grep -rn "def quantize"` 的命中全是模型内部的 VQ 模块或 integration 里的单张量量化函数）——**本仓库里，量化是加载期的行为**；如需运行时改精度，得走 `dequantize()`（`base.py:208`）或库外工具。

### 6.5 `QuantizedCache` 与本篇的关系（一句话）

它是「**KV cache** 的量化」，跟权重量化**是两条独立的路**：实现在 `cache_utils.py:1884`（`class QuantizedCache(Cache)`），由 `GenerationConfig(cache_implementation="quantized")` 触发。**详见 [2A3_kv_cache.md](./2A3_kv_cache.md)**。

---

## 七、张量并行

### 7.1 `base_model_tp_plan` 的真实形态

它是 **config 类上的一个 dict 类属性**：`{参数名模式: 策略名}`。实测两个对照（`models/cohere2/modular_cohere2.py:72` 是普通 GQA，q/k/v 用 `colwise`、o_proj 用 `rowwise`；`models/olmo2/modular_olmo2.py:69` 有 QK-Norm）：

```python
# models/olmo2/modular_olmo2.py:69  —— 有 QK-Norm，注意注释
base_model_tp_plan = {
    "layers.*.self_attn.q_proj": "colwise_gather_output",  # we need to replicate here due to the added norm on q and k
    "layers.*.self_attn.k_proj": "colwise_gather_output",  # we need to replicate here due to the added norm on q and k
    "layers.*.self_attn.v_proj": "colwise_gather_output",  # we need to replicate here due to the added norm on q and k
    "layers.*.self_attn.o_proj": "rowwise_split_input",    # input is replicated due to the added norm on q and k
    "layers.*.mlp.gate_proj": "colwise", "layers.*.mlp.up_proj": "colwise", "layers.*.mlp.down_proj": "rowwise",
}
```

**同一段注释出现在 4 个模型的 modular 文件里，一字不差**（`grep -rn 'we need to replicate here due to the added norm on q and k'` 实测 24 行 = 4 个 `modular_*.py` × 3 + 4 个生成的 `configuration_*.py` × 3）：`olmo2/modular_olmo2.py:70`、`olmo3/modular_olmo3.py:69`、`olmo_hybrid/modular_olmo_hybrid.py:110`、`flex_olmo/modular_flex_olmo.py:63`。**这是「QK-Norm 破坏 fuse」的第一个直接证据**（见 7.3）。

**plan 是怎么拼起来的**（`distributed/mixin.py:60` `init_parallel_plans()`）：

```
self._tp_plan = dict(模型类上的 _tp_plan)            # 如 MistralForCausalLM._tp_plan = {"lm_head": "colwise_gather_output"}
if self.base_model is self: self._tp_plan.update(self.config.base_model_tp_plan or {})   # ← config 上的 plan 是主力
for name, module in self.named_children():
    if plan := getattr(module, "_tp_plan", None):
        self._tp_plan.update({f"{name}.{k}": v for k, v in plan.items()})   # 子模型贡献（自动加前缀）
```

`tp_plan` property（`:87`）有个分叉：`config.distributed_config.enable_expert_parallel` 为真时**返回 `_ep_plan` 而不是 `_tp_plan`**，且 EP plan 缺失会直接 raise（`:90-95`）。（`_tp_plan` 在生成文件里也是真的：`models/llama/modeling_llama.py:423`、`models/mistral/modeling_mistral.py:396`、`models/qwen2_moe/modeling_qwen2_moe.py:591` 都写着 `_tp_plan = {"lm_head": "colwise_gather_output"}`。）

### 7.2 真实策略名全集（19 个）

注册表是 `ParallelInterface`（`distributed/tensor_parallel.py:755`），实例名 `ALL_PARALLEL_STYLES`（`:785`）。`_global_mapping`（`:758`）**只有 torch.distributed 可用时才非空**。

| 策略名 | 类 | 行号 | 语义 |
|--------|----|------|------|
| `colwise` | `ColwiseParallel` | `:161` | 权重 `Shard(0)`（Embedding `Shard(1)`）；输入 Replicate，输出 `Shard(-1)` |
| `colwise_rep` / `colwise_gather_output` | 同上 | — | `ColwiseParallel(input_layouts=Replicate(), output_layouts=Replicate())`；后者语义 = colwise + 输出 gather 回全量，**QK-Norm 模型专用** |
| `rowwise` | `RowwiseParallel` | `:244` | 权重 `Shard(1)`、bias `Replicate`（Embedding 权重 `Shard(0)`）；输入 `Shard(-1)`，输出 Partial → 归约 |
| `rowwise_rep` / `rowwise_split_input` / **`embedding_rowwise`** | `RowwiseParallel` | — | `RowwiseParallel(input_layouts=Replicate(), output_layouts=Replicate())`；`rowwise_split_input` = 输入已是全量再切（**QK-Norm 专用**）；`embedding_rowwise` = 输入/输出都 Replicate |
| `sequence_parallel` / `all_reduce` / `replicated_with_grad_allreduce` | `SequenceParallel` / `AllReduceParallel` / `ReplicatedWithGradAllReduce` | `:413` / `:359` / `:337` | 序列维切分 + norm 参数复制 / 对模块 partial 输出做 all-reduce / 参数复制、**梯度 partial**（Qwen3 的 per-head `q_norm`/`k_norm`） |
| `mla_kv_a_proj` | `MlaKvAProjParallel` | `:372` | MLA 的 `kv_a_proj_with_mqa`：KV 部分走 colwise、RoPE 部分单独 all-reduce 梯度 |
| `packed_colwise` / `packed_rowwise` | `PackedColwiseParallel` / `PackedRowwiseParallel` | `:441` / `:503` | 打包权重（如 `gate_up_proj`）的 colwise / rowwise |
| `grouped_gemm` / `ep_router` / `moe_tp_experts` | `MoEParamShard(Shard(0), shards_expert_dim=True)` / `EpRouterParallel` / `MoeExpertsParallel` | `:520` / `:656` / `:586` | **专家并行**：专家维 `Shard(0)` 且 `module.num_experts` 改成本地数 / router 输出的后处理切片 / experts forward 的输入/输出变换 + all_reduce |
| `moe_identity_expert` / `megamoe_router` / `megamoe_experts` | `MoeIdentityParallel` / `RouterParallelMegaMoe` / `MoeTensorParalellMegaMoeExperts` | `:645` / `:723` / `:735` | 单个 `nn.Identity` 专家的去重缩放 / MegaMoE 的 router（dispatch 在 kernel 里）/ experts |

**应用流程**（`apply_tensor_parallelism()`，`:797`）：

```
_validate_tp_plan_styles(model.tp_plan)     # :788  有未知策略名 → raise ValueError 并列出支持列表
for name, module in model.named_modules():
    # 第一遍（is_weight=True，按 ::参数名 查 plan）：validate_param（:807，如 colwise 要求 output_size % tp_size == 0）
    #   → shard_param（把 meta 参数换成 DTensor(Shard/N, src_data_rank=None)）
    for p_name, _ in module.named_parameters(recurse=False):
        style = _get_parameter_tp_plan(f"{name}.{p_name}", model.tp_plan, is_weight=True)  # ; style.validate_param(...) ; style.shard_param(...)
    # 第二遍（is_weight=False，按 ::模块名 查 plan）：install_forward(module, tp_mesh)（:819）包住 forward 做 input/output 变换 ; module._is_hooked = True
```

**参数名匹配规则**（`distributed/tensor_parallel.py:74` `_get_parameter_tp_plan`）：先把层号换成通配符（`replace_layer_number_by_wildcard()`，`:35`，只换「点之间的数字」，所以 `w1`/`w2` 里的数字不会被误伤）→ 精确匹配整名；不中则退一级 `layers.0.self_attn.q_proj.weight` → `layers.*.self_attn.q_proj`；`is_weight=True` 时再退一级到模块名。时序上，`apply_tensor_parallelism` 由 `maybe_distribute_model()` 调用（`modeling_utils.py:4369` → `distributed/mixin.py:183`），**在权重加载之前**——所以加载器看到的是 DTensor 占位符，能据此只读自己那一份（`core_model_loading.py:1719-1720` `DtensorShardOperation`）。

### 7.3 为什么 RoPE / QK-Norm 导致**不能直接 fuse**

**证据 1**：带 QK-Norm 的模型把 q/k/v 降级为 gather、o_proj 降级为 split（注释就写在 plan 上，见 7.1）。标准 TP + GQA 能省通信，是因为 `q_proj` 的 **colwise 输出分片**正好能被「这一卡上的那几个 head」直接消费。但加了 QK-Norm 后，`q_norm` 是**沿 head_dim 归一化的参数**，需要看到**完整的 head**（`Shard(-1)` 会把 head_dim 切一半，norm 就错了）→ q/k 之后必须 **gather 成全量**（`colwise_gather_output`），`o_proj` 的输入也就不再分片，得用 `rowwise_split_input`。

**证据 2**：RoPE 让 Q/K 的权重布局和 V 不同类。`PermuteForRope`（`core_model_loading.py:428`）专门做「complex-RoPE 布局 ↔ split sin/cos 布局」的置换，而且**只对 q/k 做**（`permute_layer_names=["q_proj", "k_proj"]`，`conversion_mapping.py:479`）。所以「把 q/k/v 合成一个 `qkv` 权重」在 RoPE 这里**不保序**：必须先置换再拼，拆时必须先拆再置换（`Chunk` → `PermuteForRope`）。`kimi_k25`（`:469-481`）和 `step3p5_vision`（`:903`）两处都写了 `# Unfuse qkv and apply rope permutation`；而 `Tipsv2VisionModel`（`:1619-1636`）同样拆 fused QKV、却**没有**这个置换——「要不要置换」是逐模型的。

**证据 3（最强）**：MLA 里 RoPE 部分和 KV 部分必须分开处理。`MlaKvAProjParallel`（`distributed/tensor_parallel.py:372`）的 docstring 直接画了图：

```
kv_a_proj_with_mqa 输出 = [kv_lora_rank + qk_rope_head_dim]（split 成两段）
      split → k_pass 路径：kv_a_layernorm → kv_b_proj (colwise) → k_pass
              k_rot  <-- "bypasses kv_b_proj"（goes straight to attention, never touches kv_b_proj）
      cat(k_pass, k_rot) → key_states
```

为什么不能整体 colwise：`k_rot`（RoPE 那一段）**不经过 `kv_b_proj`**，所以它的梯度是 **partial** 的——只有 `kv_b_proj` 有 `built-in all_reduce_backward`。docstring 最后一句：「However, `k_rot` goes straight to attention, never touches `kv_b_proj`. So we need to average gradient across all ranks otherwise we only get gradient for one rank.」代码（`:400-410`）：按 `qk_rope_head_dim` 把输出切两段（`:408`），对 RoPE 那段补一次 `_AllReduceBackward`（`:409`），再 `torch.cat` 拼回。**这个策略还需要 config 上有 `qk_rope_head_dim`，否则直接 raise**（`:401-406`）。

**小结（可背）**：RoPE / QK-Norm 是**逐 head、跨 head_dim 的操作**，而 TP 的 colwise 恰好切在 head_dim 上。两者正交时可以靠 `colwise`/`rowwise` 省通信；一旦 norm 或 RoPE 需要看到完整 head，就必须 gather/split 或单独补梯度同步。

### 7.4 与 `torch.distributed` 的关系：**DTensor + DeviceMesh**（实测）

- 用 `torch.distributed.init_device_mesh()` 建 mesh（`distributed/utils.py:160` / `:197` / `:227`）；参数用 `torch.distributed.tensor.distribute_tensor()` 变成 **`DTensor`**（`distributed/tensor_parallel.py:197`、`:272`、`:477`、`:515`、`:545` 共 5 处）；通信交给 DTensor 的 `redistribute()`（`:223`、`:240`、`:288`、`:328`）——`Replicate ← Partial` 自动变 all-reduce、`Shard(1) ← Shard(-1)` 走 all-gather / reduce-scatter。
- **少数几处手写 collective**：`dist.all_reduce`（`:320` 纯推理 fast path、`:353` `_all_reduce_grads`）和两个自定义 autograd Function：`_AllReduceForward`（`:552`）、`_AllReduceBackward`（`:565`，注释解释为什么不能用 DTensor 的 `Replicate` 反向：「DTensor's Replicate backward does incorrectly」——它会除以 world_size）。
- **推理 fast path**：当「输入已是本地张量 + 模块是 `nn.Linear` + 不在 grad 模式」时，`ColwiseParallel.transform_inputs_pre_forward()`（`:201` 起，fast path `:205-208`）**直接原样返回，根本不包 DTensor**；`RowwiseParallel.transform_output_post_forward()`（`:311`，fast path `:312-322`）也有一条纯 `dist.all_reduce` 的快路径。所以「推理 + 单卡权重已是本地分片」时，TP 开销退化到几乎只是一次 all-reduce。
- **存盘 / kernel 冲突**：`gather_state_dict_for_save()`（`:825`）用 `DTensor.full_tensor()` 把分片参数聚回完整 CPU 张量——**每个 rank 都必须调用**，否则 collective 不齐会挂。DeepGEMM 这类 kernel 只吃本地张量，所以有 `_use_local_dtensor_params()`（`:93`）contextmanager：forward 期间把 DTensor 换成 `to_local()`（「kernel 会新建 Parameter」的特例见 `:108-120`）。
- **旧 API 已废弃**：`integrations/tensor_parallel.py:54` 的 `shard_and_distribute_module()` 会 `warnings.warn(FutureWarning)` 后 `raise RuntimeError`，提示改用 `transformers.distributed.tensor_parallel.apply_tensor_parallelism`。

---

## 八、专家并行（`integrations/moe.py` + `distributed/tensor_parallel.py`）

### 8.1 `integrations/moe.py` 真实内容：**不是**分片逻辑，而是 forward 内核分派

| 内容 | 行号 | 说明 |
|------|------|------|
| `batched_mm_experts_forward` | `:112` | `torch.bmm` + `gate_up_proj[expert_ids]` 的朴素实现；**处理 EP sentinel**（`expert_ids.clamp(0, num_experts-1)`，`:132`） |
| `_grouped_mm_fallback` 系列 / `_grouped_linear` | `:179`-`:306` / `:339` | 无原生 `grouped_mm` 时的 autograd 兜底（含 backward）/ 按 offsets 做分组 GEMM 的封装 |
| `grouped_mm_experts_forward` | `:377` | 主力实现：按专家排序 → `histc` 算 offsets → 两段 grouped linear → 按 `inv_perm` 还原。**EP sentinel 处理极详细**（`:405-419`） |
| `ExpertsInterface` + `ALL_EXPERTS_FUNCTIONS` | `:481` / `:506` | 注册表：`deepgemm` / `batched_mm` / `grouped_mm` / `sonicmoe` |
| `use_experts_implementation` | `:523` | 装饰器：把 `Experts` 类的 `forward` 换成按 `config._experts_implementation` 分派的版本；顺带注入 `has_gate`/`has_bias`/`is_transposed`/`is_concatenated` |

它的 EP 相关性在这一句注释：「EP sentinel handling: leave `expert_ids` unclamped so the sort pushes sentinels to the tail, `histc(max=num_experts-1)` drops them…」——即 EP 下「不属于本 rank 的专家」被 router 打成 sentinel（索引 = `num_local_experts`），experts forward 要能优雅跳过它们。**分派本身不在这里**。

### 8.2 EP 的真实实现：`config.enable_expert_parallel=True` → 走 `base_model_ep_plan`

**入口**：`DistributedConfig(enable_expert_parallel=True)`（`distributed/configuration_utils.py:50`）→ `model.tp_plan` property 改返回 `_ep_plan`（`distributed/mixin.py:87-97`）。plan 真实形态（实测：`deepseek_v2` / `qwen3_vl_moe` / `glm4_moe` / `minimax` 等 **14 个模型**定义了真实 dict，另有 2 处显式 `= None`）：

```python
# models/deepseek_v2/modular_deepseek_v2.py:86
base_model_ep_plan = {
    "layers.*.mlp.gate": "ep_router", "layers.*.mlp.experts": "moe_tp_experts",
    "layers.*.mlp.experts.gate_up_proj": "grouped_gemm", "layers.*.mlp.experts.down_proj": "grouped_gemm",
}
# ① gate : ep_router → EpRouterParallel.transform_output_post_forward()  :700
#    router 每卡「复制地」跑一遍；后处理：非本卡专家 score 置 0、index 置 -1；本卡专家全局 id → 本地 id（idx % num_local_experts）；
#    丢弃的槽位置成 sentinel = num_local_experts（grouped_gemm 跳过）（docstring 有 4 token × top_k=4 × 128 experts × EP=8 数值例子，:671-698）
# ② experts : moe_tp_experts → MoeExpertsParallel  :586（输入去 DTensor → _AllReduceBackward；输出若分片 → _AllReduceForward）
# ③ gate_up_proj / down_proj : grouped_gemm → MoEParamShard  :520（专家维 Shard(0)；num_experts 改 per-rank 本地数，:535-543，与 ① 的 sentinel 一致）
```

**它解决的问题**：单卡放不下 `num_experts × 2I × H` 的专家权重。切法是把**专家维度**分到不同卡（而不是像普通 TP 那样切每个矩阵的行/列），所以每张卡只持有 `num_experts / ep_size` 个专家。

### 8.3 通信是 **all-reduce，不是 all-to-all**（一个需要纠正的常见说法）

**实测**：`grep -rn "all_to_all" src/ docs/` → **零命中**。`distributed/tensor_parallel.py` 里所有 collective 只有 `all_reduce`（`:320`、`:353`、`:558`、`:582`）和 DTensor 的 `redistribute()`。**本仓库的 EP 是「token 复制 + 输出求和」模型**：

```
每张卡都持有【全部 token】（hidden_states 不去切）
每张卡只算【自己那几个专家】（router 把别人的专家 score 置 0）
各卡的输出在 TP group 上 all-reduce 加起来
```

代价是**每张卡都对全部 token 做了一次「大部分是零」的 GEMM**（计算浪费），换来**不需要 all-to-all 的通信/缓冲复杂度**——`batched_mm_experts_forward` 的注释直说了（`:128-131`）：「we pay the wasted GEMM compute because batched_mm has no offset to skip.」
**唯一的例外**是 DeepGEMM 的 **MegaMoE**（`RouterParallelMegaMoe` `:723` / `MoeTensorParalellMegaMoeExperts` `:735`）：它在 **kernel 内部**做 EP dispatch + combine + per-rank token 分片（docstring：「Mega MoE handles EP dispatch inside the kernel」），所以**关掉了** `MoeExpertsParallel` 的梯度同步 hook（注释 `:735-742`）。但 transformers 侧**没有发起 all-to-all**。

> 结论：若某篇文章说「transformers 的 EP 用 all-to-all」，那是把 vLLM/SGLang 的做法套过来了。**本仓库当前代码不是这样**，以实测为准。

---

## 九、`integrations/` 目录全景表

实测：`ls src/transformers/integrations/` → **47 个 `.py` + 2 个子目录**（`heterogeneity/`、`mistral/`）。

- **A. 注意力后端 / 推理内核（11 个）**：`sdpa_attention.py`、`flash_attention.py`、`flex_attention.py`、`npu_flash_attention.py`、`msa_attention.py`（`MSA_SUPPORTED_TOPK` / `MSA_SUPPORTED_BLOCK_SIZE` / `MSA_SUPPORTED_HEAD_DIM`）、`eager_paged.py`、`sdpa_paged.py`、`flash_paged.py`、`hub_kernels.py`（`KERNELS_MIN_VERSION` / `KERNELS_MAX_VERSION`，`from_pretrained(use_kernels=True)` 的落点）、`deepgemm.py`（SM90/SM100 + nvcc JIT，含 MegaMoE）、`sonicmoe.py`（CuteDSL）。
- **B. 量化（22 个）**：与 `quantizers/` 一一对应——`aqlm.py`、`awq.py`、`bitnet.py`、`bitsandbytes.py`、`compressed_tensors.py`、`eetq.py`、`fbgemm_fp8.py`、`finegrained_fp8.py`、`fouroversix.py`、`fp_quant.py`、`gemma_quant.py`、`higgs.py`、`hqq.py`、`metal_quantization.py`、`mxfp4.py`、`nvfp4.py`、`quanto.py`、`quark.py`、`sinq.py`、`spqr.py`、`torchao.py`、`vptq.py`。**为什么是 25 → 23 → 22**：25 个 `quantizer_*.py` 里 23 个能找到实现文件，bnb 4bit/8bit 两文件**共用** `bitsandbytes.py`；**对不上号的两个**是 `quantizer_auto_round.py` 与 `quantizer_gptq.py`——`import` 指向库外的 `auto_round.inference.convert_model` / `optimum.gptq`（`quantizer_auto_round.py:50`、`quantizer_gptq.py:54`），`grep "from \.\.integrations"` 无命中。
- **C/D/E. 并行·训练·其他**：`accelerate.py`（1023 行：`infer_auto_device_map` `:642`、`accelerate_dispatch` `:399`、`accelerate_disk_offload` `:459`、`compute_module_sizes` `:144`）、`tensor_parallel.py`（93 行 **shim**）、`fsdp.py`（re-export）、`moe.py`（582 行）、`tpu.py`、`heterogeneity/`；`deepspeed.py`（`_load_pretrained_model` 有独立 ZeRO-3 分支 `modeling_utils.py:4477`）、`liger.py`、`neftune.py`、`peft.py`（**直接 import `get_model_conversion_mapping`**，`integrations/peft.py:23`、`:222`，adapter 也要走权重映射）；`executorch.py`、`ggml.py`（GGUF/GGML 读写）、`tiktoken.py`（`convert_tiktoken_to_fast()`）、`integration_utils.py`、`mistral/`（`constants.py` 的 `TEKKEN_VOCAB_FILE` / `is_tekken_vocab_filename`）。

**与本篇三个主题的交叉**：权重映射 → `accelerate.py`（`accelerate_disk_offload()` 也吃 `weight_mapping`，调用点 `modeling_utils.py:4461`）、`peft.py:222`；量化 → 22 个量化文件（`HfQuantizer.get_quantize_ops()` / `get_weight_conversions()`）；并行 → `accelerate.py` / `tensor_parallel.py` / `fsdp.py` / `moe.py` / `tpu.py` / `heterogeneity/`（`DistributedConfig` / `ALL_PARALLEL_STYLES`）。

---

## 十、实操决策树：显存不够了，按什么顺序试

**先说清楚**：仓库文档里没有「官方背书」的严格顺序，但代码里的**注入点顺序**可以读出来：量化 → 并行 → device_map → offload。

```
显存不够？
  ├─ ① 【量化】成本最低、收益最大 → get_hf_quantizer()  modeling_utils.py:4270（最早执行的一步）
  │     from_pretrained(..., quantization_config=BitsAndBytesConfig(load_in_4bit=True))  或 load_in_4bit=True
  │     · param_element_size() 被 device_map 规划用来算「量化后」显存（bnb-4bit → 0.5 字节/参数，quantizer_bnb_4bit.py:83-89）
  │     · adjust_max_memory() 给规划留余量（bnb 乘 0.90，:97-100）；⚠ requires_calibration=True 的 9 个后端必须用已量化 checkpoint
  ├─ ② 【并行】单卡装得下「一份模型」但不装下「整份」→ cls.maybe_distribute_model(...)  modeling_utils.py:4369 → distributed/mixin.py:183
  │     from_pretrained(..., distributed_config=DistributedConfig(tp_size=4))  或 enable_expert_parallel=True（MoE）
  │     · 旧写法「直接传 tp_plan=/tp_size=」已 deprecated（:4200）；要求模型有 base_model_tp_plan（supports_tp_plan :4671）
  │     · TP 在**加载前**把参数换成 DTensor 占位符，每张卡只读自己那份（core_model_loading.py:1719-1720）
  ├─ ③ 【device_map 分层】多卡 + CPU → _get_device_map()  modeling_utils.py:4373 → integrations/accelerate.py:338
  │     from_pretrained(..., device_map="auto", max_memory={0: "20GiB", "cpu": "60GiB"})
  │     · "auto" = 在 max_memory 约束下自动切块（层为粒度，_no_split_modules）→ :240 get_balanced_memory() → :642 infer_auto_device_map()
  │     · 字符串策略 "balanced" / "balanced_low_0" / "sequential"（docstring :344-345）；param_element_size 直接影响算不算得对
  └─ ④ 【disk offload】兜底（慢，但能跑起来）→ device_map="auto", offload_folder="./offload", offload_buffers=True
        · _load_pretrained_model 判断 "disk" in device_map.values() → accelerate_disk_offload()  modeling_utils.py:4461 → accelerate.py:459
        · 单参数级：core_model_loading.py:1765-1768 → offload_and_maybe_resave_param() :1386；用完后 accelerate_dispatch()  :4413 → accelerate.py:399
        · device_map 含 disk 时**异步加载被禁用**（core_model_loading.py:1629-1634）；GGUF 不支持 disk offload（:4279-4285）
```

**四条顺序上的真实约束**：① **量化必须在 device_map 规划之前**（`param_element_size` / `adjust_max_memory` 要参与估算；`get_hf_quantizer`（`:4270`）远早于 `_get_device_map`（`:4373`））；② **`update_device_map()` 会改你的 device_map**——bnb 默认「你没给 device_map 且没 CUDA？就落到 cpu」（`quantizer_bnb_4bit.py:102-115`）；③ **TP 和 device_map 是两套东西**：`maybe_distribute_model`（`:4369`）在 `_get_device_map`（`:4373`）**之前**（TP 走 DTensor 分片，device_map 走 accelerate 的层间放置）；④ **量化 + 并行有交叉钩子**：`update_tp_plan()` / `update_ep_plan()`（`base.py:144`/`:148`）专给量化权重（scale、block 等伴生参数）补并行规则——实测在 `quantizer_mxfp4.py:213-237`（给 GptOss 的 `*_blocks` / `*_scales` 加 4 条 `grouped_gemm`）。

---

## 一句话总结

> **1A3 的加载是「按名字装」，2A6 的加载是「算出来」。**
> `conversion_mapping.py` 用 93 个 model_type 的注册表 + 640 条 transform 描述「checkpoint 想变成什么」；
> `core_model_loading.py:1511` 用两趟循环（先归类、后计算）把它执行掉——改名走 `WeightRenaming`，改布局走 `WeightConverter` + `ConversionOps`。
> **量化和并行都是往这条管线里插一层**：量化插的是「先反量化再 merge/concat」的 op 和「换掉 `nn.Linear`」的 preprocess，并行插的是「参数换成 DTensor 占位符 + forward 前后加通信」。
> `fusion_mapping.py` 方向相反——它把**运行时**模型改成 fused，让原始 checkpoint 能加载。
> 记住三条硬事实：**量化是加载期行为**（没有公开的运行时 `quantize()` 入口）；
> **TP 的 colwise 切在 head_dim 上，所以 QK-Norm / MLA 的 RoPE 段必须 gather 或单独补梯度**；
> **MoE 的 EP 用 all-reduce 合输出，不是 all-to-all**（`grep all_to_all` 零命中）。

---

## 验证清单

以下命令全部在本仓库根目录（`/Users/baai/Desktop/baai_proj/transformers`）实际跑过，输出即文中引用。

```bash
# ── 文件规模 ────────────────────────────────────────────────
wc -l src/transformers/core_model_loading.py src/transformers/conversion_mapping.py \
      src/transformers/fusion_mapping.py src/transformers/integrations/tensor_parallel.py \
      src/transformers/integrations/moe.py src/transformers/distributed/tensor_parallel.py
# → 1853 / 2005 / 270 / 93 / 582 / 843

# ── 量化后端真实清单（25 个类文件）────────────────────────────
ls src/transformers/quantizers/                       # → 29 个条目 = 4 个基础文件 + 25 个 quantizer_*.py
for f in src/transformers/quantizers/quantizer_*.py; do echo "$(basename $f): $(grep -m1 '^class ' $f)"; done
# → quantizer_bnb_4bit.py: class Bnb4BitHfQuantizer(HfQuantizer)
#   quantizer_finegrained_fp8.py: class FineGrainedFP8HfQuantizer(HfQuantizer)
#   quantizer_torchao.py: class TorchAoHfQuantizer(HfQuantizer)   ...（共 25 个）
grep -n "AUTO_QUANTIZER_MAPPING = {" -A 35 src/transformers/quantizers/auto.py
# → 26 个 key（bitsandbytes 拆 4bit/8bit，fp8+mxfp8 共用 FineGrainedFP8HfQuantizer）
python3 -c "import ast;t=ast.parse(open('src/transformers/quantizers/auto.py').read());\
print([len(n.value.keys) for n in t.body if isinstance(n,ast.Assign) \
and getattr(n.targets[0],'id',None)=='AUTO_QUANTIZER_MAPPING'])"   # → [26]
grep -rn "requires_calibration = True" src/transformers/quantizers/*.py | wc -l     # → 9
grep -n "abstractmethod" src/transformers/quantizers/base.py                        # → 14(import) / 269 / 273 只有两个

# ── TP 策略真实名字（19 个）──────────────────────────────────
grep -n "ALL_PARALLEL_STYLES: ParallelInterface" src/transformers/distributed/tensor_parallel.py  # → 785
sed -n '758,780p' src/transformers/distributed/tensor_parallel.py
# → colwise / colwise_rep / colwise_gather_output / rowwise / rowwise_rep / rowwise_split_input
#   embedding_rowwise / sequence_parallel / all_reduce / replicated_with_grad_allreduce
#   mla_kv_a_proj / packed_colwise / packed_rowwise
#   grouped_gemm / ep_router / moe_tp_experts / moe_identity_expert / megamoe_router / megamoe_experts

# ── 权重映射注册表规模 ────────────────────────────────────────
python3 -c "
import ast; t=ast.parse(open('src/transformers/conversion_mapping.py').read())
for n in t.body:
    if isinstance(n, ast.FunctionDef) and n.name=='_build_checkpoint_conversion_mapping':
        for d in ast.walk(n):
            if isinstance(d, ast.Dict):
                print('keys:', len(d.keys)); break
        break"
# → keys: 93（含 "legacy"）
grep -c "WeightRenaming(" src/transformers/conversion_mapping.py     # → 621
grep -o "WeightConverter(" src/transformers/conversion_mapping.py | wc -l   # → 55
grep -o "PrefixChange(" src/transformers/conversion_mapping.py | wc -l      # → 11
grep -o "GroupWeightRename(" src/transformers/conversion_mapping.py | wc -l # → 2（都在 kimi_k25:436,446）
grep -c "MergeModulelist(dim=0)" src/transformers/conversion_mapping.py     # → 20
grep -c "MergeModulelist(dim=0), Concatenate(dim=1)" src/transformers/conversion_mapping.py  # → 9
grep -o "ErnieFuseAndSplitTextVisionExperts(" src/transformers/conversion_mapping.py | wc -l   # → 2
grep -n "Unfuse qkv and apply rope permutation" src/transformers/conversion_mapping.py        # → :469, :903 两处

# ── 关键行号抽查 ─────────────────────────────────────────────
grep -n "class WeightTransform\|class WeightRenaming\|class WeightConverter\|class PrefixChange\|class GroupWeightRename" \
     src/transformers/core_model_loading.py
# → 769 / 992 / 1152 / 1098 / 1032
grep -n "def rename_source_key\|def convert_and_load_state_dict_in_model\|def revert_weight_conversion" \
     src/transformers/core_model_loading.py
# → 1412 / 1511 / 1798
grep -n "def _load_pretrained_model\|hf_quantizer.preprocess_model\|hf_quantizer.postprocess_model\|accelerate_dispatch(model" \
     src/transformers/modeling_utils.py
# → 4436 / 4354 / 4417 / 4413

# ── QK-Norm / RoPE 的三条证据 ────────────────────────────────
sed -n '69,78p' src/transformers/models/olmo2/modular_olmo2.py
# → colwise_gather_output + "# we need to replicate here due to the added norm on q and k"
grep -rn "we need to replicate here due to the added norm on q and k" src/transformers/models/ | wc -l
# → 24 行 = 4 个模型的 modular_*.py × 3 + 它们的 configuration_*.py × 3
grep -rln "we need to replicate here due to the added norm on q and k" src/transformers/models/
# → olmo2 / olmo3 / olmo_hybrid / flex_olmo（各 2 个文件，注释一字不差）
sed -n '469,481p' src/transformers/conversion_mapping.py
# → "# Unfuse qkv and apply rope permutation" + Chunk + PermuteForRope(q_proj,k_proj)
sed -n '372,411p' src/transformers/distributed/tensor_parallel.py
# → MlaKvAProjParallel：qk_rope_head_dim 分段 + _AllReduceBackward

# ── EP 通信方式（all-to-all 零命中）──────────────────────────
grep -rn "all_to_all" src/ docs/    # → 无输出
grep -rn "dist.all_reduce\|reduce_scatter\|all_gather" src/transformers/distributed/tensor_parallel.py
# → :320 :353 :558 :582（全是 all_reduce；没有任何 all_to_all / all_gather / reduce_scatter 的原生调用）

# ── integrations/ 规模 ───────────────────────────────────────
ls src/transformers/integrations/*.py | wc -l    # → 47
ls src/transformers/integrations/ | wc -l        # → 49（47 文件 + heterogeneity/ + mistral/）

# ── fusion_mapping 的真实方向 ────────────────────────────────
wc -l src/transformers/fusion_mapping.py      # → 270；只有 PatchEmbeddingsFusionSpec 一个家族（:235）
grep -n "_FUSION_REGISTRY\|def register_fusion_patches" src/transformers/fusion_mapping.py
# → 235 / 255
grep -n "register_fusion_patches" src/transformers/modeling_utils.py   # → 4333, 4335（唯一调用点）
grep -n "Conv3d\b" src/transformers/fusion_mapping.py   # → :100 is_fusable 判定 nn.Conv3d
```

**未能完全确认的两点（如实标注）**：

1. 我没有找到任何「MoE expert **顺序**重排」的 `ConversionOps`（只有 stack / split / transpose）。
   如果确有这种需求，当前 `conversion_mapping.py` 里没有现成手段。
2. 「量化 → 并行 → offload」这条顺序是从 `from_pretrained` 里**注入点的先后**读出来的经验顺序，
   仓库文档未明确背书。以仓库实际代码为准。

---

## 导航

- 路线图：[00_roadmap.md](./00_roadmap.md)（本篇 = 进阶 · 架构 · 2A6，2A 线收官）
- 上一篇：[2A4_rope_abstraction.md](./2A4_rope_abstraction.md)（RoPE 抽象层——第 7.3 节的背景）
- 下一篇：[2M1_gemma2.md](./2M1_gemma2.md)（回到模型结构线）
- 前置：[1A3_from_pretrained.md](./1A3_from_pretrained.md)——本篇是它的进阶版，基础链路一律不回抄
- 相关：[2A5_modular_code_generation.md](./2A5_modular_code_generation.md)（`_tp_plan` 出现在生成文件里的原因）、
  [2A3_kv_cache.md](./2A3_kv_cache.md)（`QuantizedCache` 的完整机制）、
  [1M2_mistral_sliding_window.md](./1M2_mistral_sliding_window.md)（第一个 `WeightConverter` 案例的另一半）
