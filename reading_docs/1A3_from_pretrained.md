# `from_pretrained` 全链路精读：从 repo id 到显存

> 阅读目标：把 `from_pretrained` 从「一行 API」拆成一条**可追踪的调用链**——
> 哪些参数在哪一层被吃掉的、`config.json` 和 `*.safetensors` 分别怎么被取回、
> 空壳模型什么时候被造出来、checkpoint 里的权重名怎么变成模块参数名、
> 没被权重覆盖的参数又由谁初始化。
>
> 训练（1A6 `Trainer`）和推理（1A5 `generate`）两条线**唯一的耦合点**就是本篇：都要先经过它把模型造出来。
>
> 源文件（行数）：`modeling_utils.py`（5206，入口与编排）/ `core_model_loading.py`（1853，**权重加载的心脏**，
> 本篇重点）/ `utils/hub.py`（972，下载与缓存）/ `conversion_mapping.py`（2005，权重名映射表）/
> `utils/loading_report.py`（318，missing/unexpected 报告）/ `dynamic_module_utils.py`（850，`trust_remote_code`）。
>
> **代码基线**：commit `42ca97014c`（`git log -1 --format="%H"`）。所有行号都是这个版本实测的。

---

## 速读区（TL;DR）

**一句话**：`from_pretrained` 是个**调度器**——它把「取文件」和「写参数」用一行 `model = cls(config, ...)` 分开：
之上在 **meta device** 上造空壳并把 dtype / 目标设备定死，之下用 safe_open 惰性句柄 + 改名 + 张量变换把权重灌进去。
**代码基线**：commit `42ca97014c`，所有行号都是这个版本实测的。

| # | 断言 | 出处（文件:行号） |
|---|------|------------------|
| 1 | 整条链只有一个真分水岭：`model = cls(config, ...)`——之上读文件，之下写参数 | `modeling_utils.py:4350` |
| 2 | 空壳是 **meta device** 上建的：`get_init_context` 返回 `[local_torch_dtype, no_tie_weights, apply_patches, torch.device("meta"), meta_device_safe_creation_ops]` | `modeling_utils.py:3795-3820` |
| 3 | `_from_config`（`:1480`）**没有** `no_tie_weights()` 和 `torch.device("meta")` → 造出的是「随机权重、真实设备」；`from_pretrained` 是「随机权重、meta」 | `modeling_utils.py:1480-1522` |
| 4 | safetensors 主路径不实体化张量：`safe_open(..., backend="mmap")` + `file_pointer.get_slice(k)`，注释原文 `# don't materialize yet`；MPS / Windows 改用 `pread` | `modeling_utils.py:4524-4527` |
| 5 | `missing_keys` **初始值是模型的全部参数**，加载成功一个 `discard` 一个 → 「missing = 走完全程也没被 discard 掉的」 | `core_model_loading.py:1615`、`:1367` |
| 6 | `mismatched` 的参数会被 `discard` 出 missing（`:1367`）**但不 `setattr`** → 排查时**不能只看 missing / unexpected** | `core_model_loading.py:1373` |
| 7 | `_is_hf_initialized = True` 是「已读回执」，消费点是 `_initialize_weights` 的第一行 `return`（注释：`super important otherwise _init_weight will re-init the param`） | `core_model_loading.py:1382`、`modeling_utils.py:2441` |
| 8 | **llama / mistral / qwen3 的 `modeling_*.py` 里 `_init_weights` 出现 0 处**（`grep -c` 实测），完全依赖基类实现 | `modeling_utils.py:2374` |
| 9 | 基类对 norm 和 RoPE 用**类名字符串匹配**而不是 `isinstance`（`"RMSNorm" in module.__class__.__name__`），因为「自定义模块的类名带模型前缀，各不相同」 | `modeling_utils.py:2409-2414`、`:2427` |
| 10 | `original_inv_freq` 是 **`persistent=False`** 的 buffer → 不进 state_dict → 必须在 `_init_weights` 里**重算**（不是 fill） | `models/llama/modeling_llama.py:89`、`modeling_utils.py:2427-2435` |
| 11 | `*.` 编译成 `.*\.` 是**层级通配符不是 glob 星号**；每个 source pattern 是命名捕获组 `g0`/`g1`，一次 `re.compile("|".join(...))` 匹配全部分支 | `core_model_loading.py:849`、`:871` |
| 12 | 改名顺序：正向（加载）`WeightRenaming → WeightConverter`；反向（保存）`Converter → Renaming` | `core_model_loading.py:1448-1482`、`:1427-1432` |
| 13 | `_was_used` 记录哪些变换生效过，挂成 `model._weight_conversions`，`save_pretrained` 时据此 `revert_weight_conversion` | `core_model_loading.py:981`、`:1792`、`:1798` |
| 14 | 报告的**三段触发**：`error_msgs` → 最高优先级 `RuntimeError`（连报告都不打）；`conversion_errors` → 报告后 `RuntimeError`（不可绕过）；`mismatched_keys` → 报告后 `RuntimeError`（只有它能用 `ignore_mismatched_sizes=True` 绕过）；**只有 missing / unexpected 只打 warning** | `utils/loading_report.py:289-317` |
| 15 | 本地路径**优先且互斥**：只要 `os.path.isdir(path_or_repo_id)` 成立就直接拼路径返回，**完全不走 Hub** | `hub.py:391`、`:404` |
| 16 | `trust_remote_code` 三态：有本地代码 → 静默 `False`；有远端无本地且不传参 → **交互式 `input()` 等待**；传 `False` → `ValueError` | `dynamic_module_utils.py:755-793`、`:764` |

**最容易踩的坑 / 反直觉**

- **这几个开关被静默丢弃**：`low_cpu_mem_usage` / `_fast_init` / `offload_state_dict` / `mirror` / `from_tf` / `from_flax`——`modeling_utils.py:4124-4152` 里一行 `_ = kwargs.pop(name, None)`。它们是 v4 时代的开关，**在新版里什么都不做**，老教程里的说明可以直接跳过。
- 看到模型文件里**没有 `_init_weights` 不代表它没有**——先去 `modeling_utils.py:2374`；覆写的四个模型（bert `:547` / t5 `:563` / mixtral `:399` / glm_moe_dsa `:670`）第一行都是 `super()._init_weights(module)`。
- `ignore_mismatched_sizes=True` 是**重新随机初始化（reinit）**，不是「截断 / 补齐」形状。
- 「weight 加载后被重新初始化」= `_is_hf_initialized` 标记没打上（`core_model_loading.py:1382`）；「norm 权重变成全 1」= 基类字符串匹配分支命中（`modeling_utils.py:2411-2414`）。
- `trust_remote_code` 不传时**不是报错而是卡住**——它真的在等你在终端敲 `y`/`n`。
- `hf-mount`（FUSE）上的文件会**放弃 mmap 改全读**，因为 `hf-mount's mmap + readahead interaction deadlocks under parallel page-faults`（`modeling_utils.py:293`）；`device_map={"": "disk"}` 时权重**根本不进内存**（`:4460-4468`）。
- 本篇的 auto 层入口在 `auto_factory.py:461`，它先「偷偷」下一次 `config.json`（`:282-296`）**只为抢 commit hash**，好让后面所有文件都按同一个 commit 取——否则会出现「config 是 A 版本、权重是 B 版本」的错配。

**读完应该能回答**

- 一个 70B 模型 `cls(config)` 为什么几乎不占内存？内存是在哪一步、哪个函数里被真正填上的？
- `AutoModel.from_config(config)` 与 `from_pretrained` 造出来的模型，差别**只在哪几个 context manager**？
- 报告里只有 `MISSING` 行时该不该慌？`MISMATCH` 行为什么不进 missing 集合？
- `Unrecognized` / `CONVERSION` / `size mismatch` 三类报错分别该往哪查（`Details` 里的 `Ckpt contains: N` 是什么意思）？
- `rename_source_key` 第 3 步的 `base_model_prefix` 逻辑靠什么判断该补还是该去前缀？

---

## 前置知识

[1A1_repo_and_auto.md](./1A1_repo_and_auto.md) · [1A2_config_system.md](./1A2_config_system.md) ·
[2A5_modular_code_generation.md](./2A5_modular_code_generation.md)

> **一个前置结论**：`PreTrainedModel` 已把「造模型」拆成三个可覆写的钩子——`_from_config`（建模上下文）、
> `__init__`（结构）、`_init_weights`（初始化）。`from_pretrained` **不实现**这三件事中的任何一件，只负责**编排**：
> **它是调度器，不是实现者。**

---

## 零、总览：一张调用链图

```
用户: AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", dtype="auto")
                    │
                    ▼
┌─ ① auto_factory.py:461  _BaseAutoModelClass.from_pretrained ───────────────────────┐
│  ├─ cached_file(..., "config.json", _raise_exceptions_for_* = False)  :287        │
│  │     └─ 只为抢一个 commit_hash（不解析 config 内容！）                            │
│  ├─ AutoConfig.from_pretrained(...) → config                          :336        │
│  ├─ has_remote_code / has_local_code / explicit_local_code 三态判定    :355        │
│  └─ 远端代码 → get_class_from_dynamic_module(...)  :378                            │
│     本地代码 → _get_model_class(config, mapping)   :394                            │
│                → model_class.from_pretrained(path, config=config, **hub_kwargs) :385│
└───────────────────────────────────┬───────────────────────────────────────────────┘
                                    ▼
┌─ ② modeling_utils.py:3881  PreTrainedModel.from_pretrained ◀── 编排中心 ───────────┐
│  ── 参数分流 ──                                                                   │
│  kwargs.pop("cache_dir"/"state_dict"/"dtype"/"device_map"/…)      :4124-4152      │
│  download_kwargs = {cache_dir, force_download, proxies, local_files_only,          │
│                     token, revision, subfolder}                   :4167          │
│  ── 元数据 ──                                                                     │
│  config          ← config_class.from_pretrained(...)              :4233          │
│  hf_quantizer    ← get_hf_quantizer(...)                          :4270          │
│  checkpoint_files← _get_resolved_checkpoint_files(...)            :4293          │
│  dtype           ← _get_dtype(...)                                :4308          │
│  ── 建模（空壳，权重在 meta device 上）──                                          │
│  model_init_context = cls.get_init_context(...)  :4346 ──┐                        │
│      ContextManagers([local_torch_dtype, no_tie_weights, │  :3795                │
│                        apply_patches, torch.device("meta")])                      │
│  model = cls(config, *model_args, **model_kwargs)        :4350 ◀──┘               │
│  ── 准备加载 ──                                                                   │
│  dtype_plan     ← model._get_dtype_plan(dtype)                    :4364          │
│  weight_mapping ← get_model_conversion_mapping(...)               :4367          │
│  load_config    ← LoadStateDictConfig(...)                        :4376          │
│  ── 灌权重 ──                                                                     │
│  _load_pretrained_model(model, state_dict, ckpt_files, load_config)   :4393       │
│  _finalize_model_loading(model, load_config, loading_info)            :4394       │
└───────────────────────────────────┬───────────────────────────────────────────────┘
            ┌───────────────────────┴────────────────────────┐
            ▼                                                ▼
 [下载] utils/hub.py                            [灌权重] core_model_loading.py
 cached_files :299                              convert_and_load_state_dict_in_model :1511
  ├─ 本地目录? :391,404                            ├─ 遍历 state_dict（自然序）  :1647
  │   └─ 拼路径直接返回，不下载                      ├─ rename_source_key(...)     :1650
  ├─ 否则 :408                                     ├─ param_name_to_load[t] = conv :1666
  │   ├─ try_to_load_from_cache :417               ├─ spawn_materialize(→device, dtype)
  │   ├─ hf_hub_download        :438               │      线程池异步读盘            :1730
  │   └─ snapshot_download      :453               ├─ mapping.convert(...)        :1754
  └─ 分片: get_checkpoint_shard_files :859          └─ set_param_for_module(...)   :1770
                                                          （setattr 进模块）

 ── 落盘位置 ──                                  ── _finalize_model_loading :4550 ──
 ~/.cache/huggingface/hub/                        ├─ mark_tied_weights_as_initialized :4898
   models--meta-llama--Llama-3.2-1B/              ├─ _move_missing_keys_from_meta_to_device
     ├── blobs/<sha256>          ← 真实字节        │       :4778 （meta → 目标设备）
     ├── refs/main               ← "main"→commit  ├─ _initialize_missing_keys :4828
     └── snapshots/<commit>/     ← 符号链接视图    │    └─ initialize_weights :2464
          ├── config.json                         │         └─ model._init_weights :2374
          └── model.safetensors                   ├─ tie_weights(missing_keys=...) :2606
                                                  ├─ _adjust_missing_and_unexpected_keys
                                                  │       :4864 （过滤白名单）
                                                  └─ log_state_dict_report(...)
                                                          loading_report.py:273
```

**读图要点**：整条链只有一个真正的「分水岭」——`model = cls(config, ...)`（`modeling_utils.py:4350`）：
**它之上是「读文件」，它之下是「写参数」**。出问题先问「是模型结构不对，还是权重没对上」。

---

## 一、入口层：`from_pretrained` 的参数分流

调用 `AutoModelForCausalLM.from_pretrained(...)` **先经过 auto 类**，再落到具体模型类：

| 层 | 位置 | 职责 |
|----|------|------|
| `_BaseAutoModelClass.from_pretrained` | `auto_factory.py:461` | 解析 config、判定本地/远端代码、**选出模型类** |
| `PreTrainedModel.from_pretrained` | `modeling_utils.py:3881` | 下载权重、建模、灌权重、初始化 |

auto 层做的事（`auto_factory.py:461-386`，函数体 `:461-473`）：把 7 个 hub 参数
（`cache_dir` / `force_download` / `local_files_only` / `proxies` / `revision` / `subfolder` / `token`）
摘进 `hub_kwargs` 并以 `**hub_kwargs` 透传给下两层（**真正使用在下两层的 `cached_file` 里**），
并广播一个 `kwargs["_from_auto"] = True` 标记。紧接着它先「偷偷」下一次 `config.json`（`auto_factory.py:282-296`）：

```python
# auto_factory.py:282-296（节选）— 为了拿 commit_hash，先下一次 config.json
resolved_config_file = cached_file(pretrained_model_name_or_path, CONFIG_NAME,
    _raise_exceptions_for_gated_repo=False, _raise_exceptions_for_missing_entries=False,
    _raise_exceptions_for_connection_errors=False, **hub_kwargs)
commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
```

**只为让后面所有文件按同一个 commit 取**，否则出现「config 是 A 版本、权重是 B 版本」的错配；
三个 `...=False` 是为了**容忍失败**（config 后面还会正式读一次）。
然后 `AutoConfig.from_pretrained`（`auto_factory.py:336`）＋ 模型类选择（`:355-410`，判定代码 `:355-362`）：
`has_remote_code`（config 有 `auto_map` 且含本类名）/ `has_local_code`（`type(config)` 在 `_model_mapping` 里）/
`explicit_local_code`（在库里但不在 `transformers.` 命名空间），组合后交给 `resolve_trust_remote_code`；
远端代码走 `get_class_from_dynamic_module`（`:378`），本地代码走 `_get_model_class`（`:394`），
最后都是 `model_class.from_pretrained(path, config=config, **hub_kwargs)`（`:385`）。

### 1.1 参数在哪一层被吃掉

| 参数 | 首次被识别 | 最终消费点 |
|------|-----------|-----------|
| `cache_dir` | `auto_factory.py:265`（进 `hub_kwargs`） | `hub.py:408` 的 `cache_dir = constants.HF_HUB_CACHE` 默认值被覆盖处 |
| `revision` / `token` | 同上 | `hub.py:438` `hf_hub_download(revision=...)` |
| `local_files_only` | 同上；`modeling_utils.py:4164` 离线模式强制 `True` | `hub.py:379` |
| `subfolder` | 同上 | `hub.py:386` |
| `force_download` | 同上；`modeling_utils.py:4176` 也进 `download_kwargs_with_commit` | `hub.py:438` |
| `state_dict` | **只在** `modeling_utils.py:4124` | `_load_pretrained_model`（跳过读盘） |
| `dtype` / `torch_dtype` | `modeling_utils.py:4130-4131` | `_get_dtype`（`modeling_utils.py:815`） |
| `device_map` | `modeling_utils.py:4132` | `check_and_set_device_map`（`integrations/accelerate.py:96`）→ `_get_device_map`（同文件 `:338`） |
| `quantization_config` | `modeling_utils.py:4136` | `get_hf_quantizer`（`quantizers/auto.py:334`） |
| `trust_remote_code` | `modeling_utils.py:4148`；**auto 层也用**（`auto_factory.py:366`） | `dynamic_module_utils.resolve_trust_remote_code`（`:712`） |

`modeling_utils.py:4124-4152` 是「分流器」，把 kwargs 拆成三类：显式参数（`pop` 出来直接用）、
透传参数、以及**第三类被静默丢弃**的 v4 时代开关——`mirror` / `_fast_init` / `low_cpu_mem_usage` /
`from_tf` / `from_flax` / `offload_state_dict`，一行 `_ = kwargs.pop(name, None)`（现在默认就是 meta 建模 + 流式加载，
参数留着只为不破坏旧代码）。剩下的 kwargs 由 `:4233-4256` 处理（`:4233` 起）：**是 config 的字段 → 被 config 吃掉**
（如 `num_labels=10`）；**不是 → 留在 `model_kwargs`** 里喂给 `cls(config, *model_args, **model_kwargs)`（`:4350`）。
靠的是 `config_class.from_pretrained(..., return_unused_kwargs=True)` 的 `model_kwargs` 回传
（`commit_hash = model_kwargs.pop("_commit_hash", commit_hash)`）；**显式传 `config` 时走 `else` 分支，
全部 kwargs 都进模型 `__init__`**。

---

## 二、下载层：`utils/hub.py` 与本地缓存

### 2.1 唯一的下载入口：`cached_files`

`cached_file`（`hub.py:239`）是薄壳（函数体 `:294-297`），真正干活的是 `cached_files`（`hub.py:299`）。
其判定逻辑（`hub.py:373-466`）三步：

```python
# hub.py:379 起（节选）：离线强制 → 本地目录 → 缓存 → 下载
if is_offline_mode() and not local_files_only: local_files_only = True   # :379
full_filenames = [os.path.join(subfolder, file) for file in filenames]   # :386
#
# 第 1 步：本地目录 —— 逐个 filename 判断（:391 的 for + if os.path.isdir(path_or_repo_id)）
#     resolved_file 不存在时：config.json 例外（:395）→ continue，否则按 _raise_..._missing_entries 决定 raise 或 continue
#
if os.path.isdir(path_or_repo_id):
    return existing_files if existing_files else None                    # :404 ← 直接返回，不碰网络
if cache_dir is None: cache_dir = constants.HF_HUB_CACHE                 # :408 第 2 步
resolved_file = try_to_load_from_cache(..., revision=_commit_hash, ...)  # :417
hf_hub_download(..., revision=revision, ...)                             # :438 第 3 步
snapshot_download(..., allow_patterns=full_filenames, ...)               # :453（多文件时）
```

三点：① **本地路径优先且互斥**（`:391`/`:404`）——只要 `os.path.isdir(path_or_repo_id)` 成立就拼路径返回、
**完全不走 Hub**，这是「同名本地目录导致加载到错的东西」的根源（`modeling_utils.py:758` 的报错文案也提了）；
② `constants.HF_HUB_CACHE`（`:408`）来自 **`huggingface_hub` 的 `constants`**（`hub.py:36` 的 import 已说明），
`HF_HOME` / `HF_HUB_CACHE` 语义由该库定义；③ `config.json` 是**唯一被特批可以不存在**的文件（`:395`），
因为 `auto_factory.py:282` 的探测只是想探 commit hash。

### 2.2 各文件由哪条路径取回

| 文件 | 取回函数 | 调用点 | 说明 |
|------|---------|--------|------|
| `config.json` | `cached_file` | `configuration_utils.py:802`（`get_config_dict` 内） | 常量 `CONFIG_NAME`，`utils/__init__.py:284` |
| `model.safetensors` | `cached_file` | `modeling_utils.py:676`（`_get_resolved_checkpoint_files` 内） | `SAFE_WEIGHTS_NAME`，`utils/__init__.py:282` |
| `model.safetensors.index.json` | `cached_file` | `modeling_utils.py:681`（首次尝试失败后的 fallback） | `SAFE_WEIGHTS_INDEX_NAME`，`utils/__init__.py:283` |
| 各分片 `model-0000X-of-0000Y.safetensors` | `cached_files`（批量） | `hub.py:902`（`get_checkpoint_shard_files` 内） | 从 index.json 的 `weight_map` 反推 |
| `pytorch_model.bin` | `cached_file` | `modeling_utils.py:703` | 老格式，仅在 safetensors 全找不到时才用 |

四个权重文件名常量见 `utils/__init__.py:280-284`（`WEIGHTS_NAME` / `WEIGHTS_INDEX_NAME` /
`SAFE_WEIGHTS_NAME` / `SAFE_WEIGHTS_INDEX_NAME` / `CONFIG_NAME`）。
`variant` 参数由 `_add_variant`（`modeling_utils.py:527`）实现：把 `model.safetensors` → `model.fp16.safetensors`
（`rsplit(".", 1)` 后插 `.{variant}`），所以 `from_pretrained(..., variant="fp16")` 找的是后者。

### 2.3 分片、缓存目录结构与 disk offload

解析到 `*.index.json` 时 `is_sharded=True`，走 `hub.py:859`：读 `index.json`，`weight_map` 的**值**去重得到
`shard_filenames`，`metadata` 与 `weight_map` 一起塞进 `sharded_metadata`（`hub.py:887-892`）。
`sharded_metadata` 后面挂在 `LoadStateDictConfig` 上（`modeling_utils.py:4379`）：给 `_get_dtype` 提供
`metadata["dtype"]`（省掉「先读一个张量」），并给 disk offload / 并行加载提供 key→shard 映射。
`index.json` 的 `weight_map` 形如 `{"metadata": {"total_size": ..., "dtype": "bfloat16"},
"weight_map": {"model.embed_tokens.weight": "model-00001-of-00002.safetensors", ...}}`，
其 **key 就是 checkpoint 的「原始权重名」**——正是第四节要映射的对象。

```
~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/   ← HF_HOME / HF_HUB_CACHE（:408）由 huggingface_hub 定义
  ├── blobs/<sha256>            ← 文件真实内容（内容寻址，同一份字节只存一次）
  ├── refs/main                 ← 内容是一行 commit sha
  └── snapshots/<commit>/       ← 以 commit 命名的「视图」，全是指回 blobs 的符号链接
~/.cache/huggingface/modules/transformers_modules/<repo>/... ← HF_MODULES_CACHE（hub.py:108）
```

所以 **`ls snapshots/<commit>/` 看到的是「有哪些文件」，`du blobs/` 才是真实占用**。
`HF_MODULES_CACHE` 与 `TRANSFORMERS_DYNAMIC_MODULE_NAME` 的定义就两行（`hub.py:108`、`hub.py:109`），
后者恒为 `"transformers_modules"`。

`_get_resolved_checkpoint_files` 只负责「把文件弄到本地」，但 `_load_pretrained_model` 里有一个把两者串起来的动作
（`modeling_utils.py:4461`，代码 `:4460-4468`）：若 `device_map` 里出现 `"disk"`，就调 `accelerate_disk_offload`
把权重直接重写到 `offload_folder` 并留在 `disk_offload_index` 里。**`device_map={"": "disk"}` 时权重根本不进内存**，
等 forward 时由 accelerate 的 hook 按需读回。

---

## 三、建模层：config → `cls(config)`

### 3.1 `_get_dtype`：`dtype="auto"` 到底怎么定的

默认 `dtype="auto"`（`modeling_utils.py:4161-4162`）。`_get_dtype`（`modeling_utils.py:815`）的优先级
**从便宜到昂贵**递进（`:834-849`）：① config 里已有 `config.dtype`；② `is_sharded` 且有
`sharded_metadata["dtype"]` → **只看 JSON 字段、完全不读文件**（这是「dtype 判定很快」的原因）；
③ 内存里已有 `state_dict` → `get_state_dict_dtype`；④ `.gguf` → `torch.float32`；⑤ 否则
`load_state_dict(checkpoint_files[0], map_location="meta")` 读第一个分片（meta，不占内存）再猜。
结果写进 config 和所有子 config（`:890-893`，`:890` 起 `config.dtype = main_dtype` 后遍历 `config.sub_configs`）——
**`dtype` 在「建模前」就通过修改 config 介入**，所以 `cls(config)` 里的 `nn.Linear(...)` 会按正确的 dtype 创建。

### 3.2 `get_init_context`：为什么模型是「空壳」

这是整个建模阶段最容易被忽略、但最重要的一件事（`modeling_utils.py:3791`，构造在 `:3795-3820`）：

```python
# modeling_utils.py:3795-3820（节选）
init_contexts = [local_torch_dtype(dtype, cls.__name__), init.no_tie_weights(), apply_patches()]
if is_deepspeed_zero3_enabled(): ...
else:  # meta_device_safe_creation_ops patches torch.linspace to default to CPU
    init_contexts.extend([torch.device("meta"), init.meta_device_safe_creation_ops()])
```

| Context manager | 位置 | 作用 |
|-----------------|------|------|
| `local_torch_dtype(dtype, cls.__name__)` | `modeling_utils.py:221`（实现 `:221-242`） | 临时 `torch.set_default_dtype(dtype)`，`finally` 一定还原（`:227` 对非浮点 dtype 直接 `ValueError`） |
| `init.no_tie_weights()` | `initialization.py` | 建模期间**禁止 tie**（等权重全灌完再 tie） |
| `torch.device("meta")` | — | **所有参数创建在 meta device 上：只占 shape，不占内存** |
| `init.meta_device_safe_creation_ops()` | `initialization.py` | 把 `torch.linspace` 等重定向到 CPU，让自定义模型在 meta 下也能 `.item()` |

**「meta device 建模」是理解 `from_pretrained` 内存行为的核心**：一个 70B 模型 `cls(config)` 时**几乎不占内存**，
显存/内存在 `spawn_materialize` 阶段才被填。这也是 `_move_missing_keys_from_meta_to_device`（`modeling_utils.py:4778`）
名字里出现 "from_meta" 的原因——**没被权重覆盖的参数会一直留在 meta 上**，必须显式搬回真实设备才能初始化。

### 3.3 `_from_config`：另一条入口

`cls._from_config(config, **kwargs)`（`modeling_utils.py:1480`，实现 `:1480-1522`）是「无权重版本」的建模段：
同样先 `config.dtype = dtype` 并同步到所有子 config，再在 `[apply_patches(), local_torch_dtype(...)]`
（**注意：没有 ZeRO-3 以外的 meta / no_tie_weights**）里执行 `model = cls(config, **kwargs)` 并 `patch_output_recorders(model)`。

| | `_from_config`（`:1480`） | `from_pretrained` 的建模段（`:4346-4356`） |
|---|---|---|
| dtype context / ZeRO-3 | 有 | 有 |
| `no_tie_weights()` | **无** | 有 |
| `torch.device("meta")` | **无** | 有 |
| 之后做什么 | 直接返回（权重随机初始化） | 进入 `_load_pretrained_model` 灌权重 |

**`AutoModel.from_config(config)` 造出的是「随机权重、真实设备」，`from_pretrained` 造出的是「随机权重、meta」**——
两者都调同一个 `cls(config)`，差别**只在 context manager**。这正是它虽带 `_` 前缀却依然是正经钩子的原因。

### 3.4 `device_map` 的介入时机

分三次、每次更精确：① `check_and_set_device_map`（`integrations/accelerate.py:96`，很早，在
`modeling_utils.py:4226`，因为 `get_hf_quantizer`（`:4270`）需要它）把 `None`/`int`/`str`/`torch.device`
统一成 `dict` 并捕获 device context manager；② `_get_device_map`（同文件 `:338`）在 `cls(config)` **之后**
（`modeling_utils.py:4372-4373`）用 `infer_auto_device_map` 真正算分配方案——**必须等模型实例存在**，
才知道 `_no_split_modules` 和参数量；③ 加载时每个参数由 `get_device(device_map, renamed_key, valid_torch_device=True)`
（`core_model_loading.py:1717`）查一次自己的目标设备。

---

## 四、权重加载层：`core_model_loading.py`（本篇重点）

### 4.1 `_load_pretrained_model`：读盘策略的三岔口

`modeling_utils.py:4436` 是「读文件」的总调度，主体（`:4477-4536`）是三分支：A) ZeRO-3 且未量化时
先把所有分片读进内存合并成一个 `state_dict`；**B) safetensors —— 用 `safe_open` 拿「切片句柄」，不实体化（默认路径）**；
C) `.bin` —— `torch.load` 全读。分支 B（`modeling_utils.py:4500-4529`）的核心：

```python
# modeling_utils.py:4504-4527（节选）
for file in checkpoint_files:
    if load_config.disable_mmap or _is_on_hf_mount(file):      # ← 放弃 mmap，全读
        with open(file, "rb") as _fh: merged_state_dict.update(_safe_load_bytes(_fh.read()))
        continue
    if is_mps:                     backend, device = "pread", "mps"   # MPS 不支持 mmap
    elif sys.platform == "win32":  backend, device = "pread", "cpu"   # Windows mmap 会为整个文件预留 COW 提交量
    else:                          backend, device = "mmap", "cpu"
    file_pointer = safe_open(file, framework="pt", device=device, backend=backend)   # :4524
    all_pointer.add(file_pointer)
    for k in file_pointer.keys():
        merged_state_dict[k] = file_pointer.get_slice(k)   # :4527  don't materialize yet
```

`safe_open(..., backend="mmap")` **mmap 打开文件不读内容**（按需分页）；`get_slice(k)` 拿到的是
**`PySafeSlice` 惰性句柄**而不是 `torch.Tensor`——所以循环结束时 `merged_state_dict` 里是**一堆句柄**，内存占用 ≈ 0。
句柄的 `__exit__` 统一在 `:4540-4542` 关闭（`for k in all_pointer: k.__exit__(None, None, None)`），
这是异常路径下文件也能被 GC 回收的原因。三个 backend 的取舍：

| backend | 何时用 | 原因 |
|---------|--------|------|
| `mmap`（默认） | Linux / macOS，CPU | 零拷贝、按需分页，最快 |
| `pread` | MPS 设备、Windows | MPS 不支持 mmap；Windows 的 mmap 会为整个文件预留写时复制提交量 |
| 全读进内存 | `disable_mmap=True` 或文件在 **hf-mount FUSE** 上 | `_is_on_hf_mount`（`modeling_utils.py:293`）docstring：`hf-mount's mmap + readahead interaction deadlocks under parallel page-faults` |

`_is_on_hf_mount`（`modeling_utils.py:303-317`）读 `/proc/mounts`，把挂载点按长度降序排列做**最长前缀匹配**，
判断命中项的设备名是不是 `"hf-mount"`。

### 4.2 `convert_and_load_state_dict_in_model`：主循环

签名在 `core_model_loading.py:1511`。它的 docstring（`:1517-1602`）本身就是设计文档——**读懂
`param_name_to_load` 的形态（`:1521-1535`）就读懂了整个加载策略**：值是 `WeightConverter` + `collected_tensors`
（**只收句柄，不实体化**）+ `layer_targets`（一个 source 对应多个 target，如 `qkv` → `q`/`k`/`v`）。
主循环（`:1647-1748`）四步，每步一职：

```python
# core_model_loading.py:1647（节选四步）
state_dict = sorted(state_dict.items(), key=lambda kv: dot_natural_key(kv[0]))   # 自然序：layers.10 在 layers.9 之后
for original_key, tensor in state_dict:
    renamed_key, source_pattern = rename_source_key(original_key, renamings, converters,
                                                    base_model_prefix, meta_model_state_dict)   # ① 改名
    # 改名失败但原名本身就对 → 只补/去 base_model_prefix，再试一次
    if renamed_key in meta_model_state_dict:
        mapping = param_name_to_load.setdefault(renamed_key, ...)     # ② 收进容器（:1666 setdefault）
        ...                                                          # ③ dtype 决策
        future_or_tensor = spawn_materialize(thread_pool, tensor, param_device, _dtype,
                                             sharding_op=sharding_op, tensor_idx=tensor_idx)   # ④ 异步实体化（:1730）
        mapping.add_tensor(renamed_key, original_key, source_pattern, future_or_tensor)
    elif source_pattern is not None:      # 一源多目标只匹配上一个 → 其余目标记 unexpected
        for k in pattern_to_converter[source_pattern].target_patterns: _add_unmatched_checkpoint_key(...)
    else: _add_unmatched_checkpoint_key(renamed_key, model, loading_info)   # ← 记 unexpected
```

排序键 `dot_natural_key`（`core_model_loading.py:1272`）：split on `"."`，数字按数值、字符串按字典序，
并 emit tuple 让 int 排在 str 前以免比较失败。**为什么必须自定义排序**：`model.layers.10.x` 要排在
`model.layers.9.x` **之后**（纯字符串排序会反过来）；`GroupWeightRename` 的 docstring（`:1043`）明确依赖这个顺序：
guard pattern 必须在字典序上小于被依赖的 pattern。

### 4.3 `mapping.convert` → `set_param_for_module`

收完所有 key 之后是第二个循环（`core_model_loading.py:1752-1781`）：对 `param_name_to_load` 里每个 mapping 调
`mapping.convert(...)` 拿到 `realized_value`，逐个 target 查目标设备；`"disk"` 走 `offload_and_maybe_resave_param`，
否则 `set_param_for_module(...)`（`:1770`），最后 `del realized_value` 及时释放避免峰值内存翻倍。
**两级 `try` 的分工**：内层 `except SkipParameters` —— 单个参数转换失败被
`log_conversion_errors`（`core_model_loading.py:1294`）记进 `loading_info.conversion_errors`，再抛哨兵异常
`SkipParameters`（`:1405`）跳过该参数，**不毁掉整个加载**；外层 `finally`（`:1787-1789`）无论成功失败还是 Ctrl-C
都关线程池，且 `cancel_futures=True`。`set_param_for_module`（`core_model_loading.py:1343`，实现 `:1343-1382`）
是「写进模型」的最后一步，三行代码定义了三类 key 的归属：

```python
# core_model_loading.py:1343-1382（节选）
ref = getattr(module_obj, param_name)
if ref is None: loading_info.unexpected_keys.add(target_name)                 # :1360 → unexpected
else:
    loading_info.missing_keys.discard(target_name)                            # :1367 ← 加载成功一个就 discard
    expected_shape = ref._local_tensor.shape if is_dtensor(ref) else ref.shape
    if ref is not None and param_value.shape != expected_shape and hf_quantizer is None:
        loading_info.mismatched_keys.add((target_name, param_value.shape, expected_shape))  # :1373
    else:
        # super important otherwise _init_weight will re-init the param
        param_value._is_hf_initialized = True                                 # :1382
        setattr(module_obj, param_name, param_value)
```

| 条件 | 结果 |
|------|------|
| `ref is None`（模型里没这个属性） | → `unexpected_keys` |
| shape 不等（且 `hf_quantizer is None`） | → `mismatched_keys`（**注意：仍然被 `discard` 出 missing**，且不 `setattr`） |
| 形状匹配 | → `setattr`，并打上 **`_is_hf_initialized = True`**（「已读回执」，第五节消费） |

### 4.4 三层改名机制：`WeightRenaming` / `WeightConverter` / `WeightTransform`

`conversion_mapping.py` 定义**声明式**映射表，`core_model_loading.py` 定义**执行引擎**：

```
WeightTransform                    core_model_loading.py:769   ← 基类：模式编译 + 反向变换
├── WeightRenaming                 core_model_loading.py:992   ← 只改名，不做张量运算
│   ├── GroupWeightRename          core_model_loading.py:1032  ← N:N 成组改名，带 guard 门控
│   └── PrefixChange               core_model_loading.py:1098  ← 专门用于增删 `base_model_prefix`
└── WeightConverter                core_model_loading.py:1152  ← 改名 + 张量运算（Chunk/Concat/...）
```

**`WeightTransform.__init__`（`core_model_loading.py:786`）只做一件事：把字符串模式编译成正则**（`:845-851`）：
`source_pattern.replace("*.", r".*\.")`，每个 source pattern 包成命名捕获组 `(?P<g{i}>...)`，
最后 `re.compile("|".join(branches))`。三个设计细节：① `*.` → `.*\.`——**`*` 是「层级通配符」而不是 glob 星号**
（`mlp.experts.*.gate_proj.weight` 能匹配 `mlp.experts.7.gate_proj.weight`）；② 命名捕获组让匹配后能反查
「是第几个 source pattern 命中的」（`rename_source_key`，`:917-919`）；③ 单次编译匹配所有分支，是热路径优化。
`WeightRenaming`/`WeightConverter` 自己的 `rename_source_key`（`:913-932`）在命中时打 `self._was_used = True`，
并**先统一替换成 `self.target_patterns[0]`**（一源多目标时以第一个 target 作为 key 归并）。
模块级的自由函数 `rename_source_key`（`:1412`）负责编排顺序（`:1448-1482`）：
**正向（加载）`Renaming → Converter`，反向（保存）`Converter → Renaming`**——docstring（`:1427-1432`）说明
Converters 只作用于特定 leaf pattern，其输出不会再被 Renaming 命中，故无需反方向。
第 3 步（`:1472-1480`）用 `meta_state_dict` 做存在性检查**双向**探测补/去 `base_model_prefix`，
效果是**让 `LlamaForCausalLM` 能直接吃 `LlamaModel` 的权重（反之亦然）**。
`_was_used`（`:981`）docstring：有些改名不是双射（`PrefixChange` 会丢掉名字的一部分），
不记住「是否用过」就无法还原——加载结束时把生效过的变换挂成 `model._weight_conversions`（`:1792-1793`），
**这是 `save_pretrained` 时 `revert_weight_conversion`（`core_model_loading.py:1798`）的依据**。

### 4.5 `WeightConverter` 的运算与 `weight_mapping` 的选择

`ConversionOps`（`core_model_loading.py:81`）是一个 ABC，实现全在这一个文件里：

| Op | 行号 | 作用 | 典型场景 |
|----|------|------|---------|
| `Chunk` | `:112` | 沿 dim 切 N 份 | `qkv` → `q`/`k`/`v` |
| `Concatenate` | `:150` | 沿 dim 拼多份 | 多个 expert → 一个堆叠张量 |
| `Interleave` | `:195` | 交错拼接（不是简单 concat） | `w13` → `gate_proj`/`up_proj` 的交错布局 |
| `MergeModulelist` | `:222` | 把 `experts.0.x`、`experts.1.x`… 堆成一个 | MoE 逐 expert 权重 → 融合权重 |
| `SplitModulelist` | `:273` | `MergeModulelist` 的逆 | 保存时 |
| `Transpose` | `:312` | 转置 | 有些实现存的是转置权重 |
| `Conv3dToLinear` / `LinearToConv3d` | `:362` / `:402` | 卷积和线性互转 | 视觉模型 patch embed |
| `PermuteForRope` | `:428` | RoPE 权重重排 | 旋转位置编码的 Q/K 重排 |
| `_IdentityOp` | `:101` | 什么都不做 | 占位 |

真实例子（`conversion_mapping.py:198-216`，DeepSeek 系 MoE）：`:198-202` 用
`WeightConverter(source_patterns="shared_w13_weight", target_patterns=["gate_proj", "up_proj"],
operations=[Interleave(dim=1), Chunk(dim=1)])`；`:204-208` 用
`source_patterns="mlp.experts.w13_weight"` / `target_patterns=["mlp.experts.gate_up_proj"]` / `[Interleave(dim=1)]`。
对照真实模型结构（`models/mixtral/modeling_mixtral.py:57-67`，`:57` 处的 `MixtralExperts`）：
`gate_up_proj` / `down_proj` 都是 3D `nn.Parameter`。**「模型里是一个 3D 参数、checkpoint 里是 N 个独立 expert」
这个落差，就是 `MergeModulelist` + `Concatenate` 存在的全部理由**——也是理解 `conversion_mapping.py` 的最好入口。

`get_model_conversion_mapping`（`conversion_mapping.py:1902`）一句话：**遍历模型所有子模块，按「类名优先、
`model_type` 兜底」查表，命中的挂上去**（docstring `:1902-1930`）：root transforms 全局生效；子模块的 transform
设 `scope_prefix`，只匹配该前缀下的 key；处理过子模块后其 class name 与 `model_type` 都记为 seen，
避免 `XForY`/`XModel` 经不同路径把同一映射应用两次（去重逻辑 `:1930-1996`，实现 `:1936-1961`）。
**`scope_prefix` 是多模态模型的关键**：`LlavaForConditionalGeneration` 里嵌套 `LlavaModel`，两层 `model_type` 都是
`llava`，不加 scope 就会把映射重复应用两次；`_scoped_match`（`core_model_loading.py:871`）负责剥离前缀后再匹配。
最后 `conversion_mapping.py:2003`（`:2003-2004`）给量化器一个**插入自己的 op** 的机会——
FP8 反量化会**前插** `Fp8Dequantize`，保证 per-block scale 在任何 expert-merge 之前生效。
**规模**（`grep -c` 实测）：`WeightRenaming(` **621** 次、`WeightConverter(` **55** 次、
`PrefixChange(` **11** 次、`GroupWeightRename(` **2** 次，覆盖 **84** 个 model_type / 类名条目。

---

## 五、`_init_weights`：没被 checkpoint 覆盖的参数

### 5.1 基类实现与三条兜底链

`PreTrainedModel._init_weights`（`modeling_utils.py:2374`，实现 `:2374-2381`）刻意写得**又长又通用**：
**std 有四级兜底**（历史包袱的化石）——`initializer_range` → `init_std`（老 BERT 系）→ `initializer_factor`（T5）
→ 文本子 config → `0.02`（注释：`0.02 is the standard default value across the library`）。
分支结构（同函数 `:2390-2435`）：

| 模块类型 | 判断方式 | 初始化 |
|---------|---------|--------|
| `nn.Linear` / `nn.Conv*` | `isinstance` | `normal_(mean=0, std=std)`；bias `zeros_` |
| `nn.LSTM` | `isinstance` | weight `xavier_uniform_`；bias `constant_(0)` |
| `nn.Embedding` | `isinstance` | `normal_`；`padding_idx` 那行**额外** `zeros_` |
| `nn.MultiheadAttention` | `isinstance` | 交给 torch 自己的 `_reset_parameters()` |
| **任意 Norm** | **类名字符串匹配** `"LayerNorm" in ... or "RMSNorm" in ...` | weight `ones_`；bias `zeros_` |
| **任意 RoPE** | **类名字符串匹配** `"RotaryEmbedding" in ... and hasattr(module, "original_inv_freq")` | 重算 `inv_freq` |

**那两个「类名字符串匹配」是这里最反直觉的设计**，代码里也给了理由（`:2409-2410`，原文 `:2409`）：
`We cannot use isinstance on the RMSNorms or LayerNorms, as they usually are custom modules which change names
between modelings (because they are prefixed with the model name)`。500 个模型有 500 个 `XxxRMSNorm`，
但它们都是「名字里带 RMSNorm 的类」——用字符串匹配一次性全覆盖。
RoPE 那一支（`:2427-2435`）**不是简单的 fill，而是重算**：按 `module.rope_type` 选
`ROPE_INIT_FUNCTIONS[...]`（`"default"` 时用 `module.compute_default_rope_parameters`），算出 `buffer_value` 后
`init.copy_` 到 `module.inv_freq` 与 `module.original_inv_freq`。对照 `models/llama/modeling_llama.py:82-89`，
`:89` 是 `self.original_inv_freq = nn.Buffer(inv_freq.clone(), persistent=False)`。
**`persistent=False` + `_init_weights` 里重算，是一个精妙的配合**：`persistent=False` 意味着这个 buffer
**不进 state_dict**（checkpoint 里根本没有它），必须在建模时被算出来——`_init_weights` 的这一支就是正式场所。

### 5.2 三个模型的 `_init_weights` 对比

**实测结论（`grep -c "_init_weights"`）**：`models/llama/modeling_llama.py` / `mistral` / `qwen3` 各 **0** 处——
**这三个现代 decoder-only 模型完全依赖基类实现**，因为 Llama 的组件（`LlamaRMSNorm` / `nn.Linear` /
`nn.Embedding` / `LlamaRotaryEmbedding`）**全部命中基类的字符串匹配规则**。

| 模型 | 是否覆写 | 覆写内容 | 为什么 |
|------|---------|---------|--------|
| `LlamaForCausalLM` / `MistralForCausalLM` / `Qwen3ForCausalLM` | **否** | — | 组件全被基类覆盖 |
| `BertPreTrainedModel` | 是（`bert/modeling_bert.py:547`，见 `:547-554`） | 加 `BertLMPredictionHead` 的 bias 置零；`position_ids` / `token_type_ids` 填值 | 有非持久 buffer 需要填 |
| `T5PreTrainedModel` | 是（`t5/modeling_t5.py:563`，见 `:563-570`） | **逐模块手写 std**：`factor * (d_model ** -0.5)`、`factor * (d_ff ** -0.5)` | T5 的初始化是架构定义的一部分，不能用统一 std |
| `MixtralForCausalLM` | 是（`mixtral/modeling_mixtral.py:399`，见 `:399-406`） | 加 `MixtralExperts` 和 `MixtralTopKRouter` | 融合专家参数是 `nn.Parameter`，基类不认识 |
| `GlmMoeDsaForCausalLM` | 是（`glm_moe_dsa/modeling_glm_moe_dsa.py:670`，见 `:670-677`） | 加 `TopkRouter` / `Experts` | 同上 |

**共同模式**：① `super()._init_weights(module)` 永远是第一行（T5 的 `T5LayerNorm` 分支是覆盖而非追加）——
不做这一步，norm 和 embedding 就没人管了；② 分支条件一定是「基类不认识的模块」；③ `Mixtral` 与 `GlmMoeDsa`
几乎逐字相同——正是 [2A5](./2A5_modular_code_generation.md) 讲的「跨模型重复」，只不过 `_init_weights`
没被 modular 机制抽出来共享。T5 的 `T5LayerNorm` 初始化是 `factor * 1.0` **而不是** `1.0`；
Mixtral/GlmMoeDsa 用 `std = self.config.initializer_range`，GLM 版多一行 `init.zeros_(module.e_score_correction_bias)`（偏置必须置零）。
**实操价值**：想知道「某个新参数怎么初始化」时，**不要在这个模型文件里找不到就完事**——
先去确认它在不在，不在就去 `modeling_utils.py:2374`。

### 5.3 `initialize_weights` 与 `smart_apply`

`_initialize_weights`（`modeling_utils.py:2437`，实现 `:2437-2458`）是「跳过已初始化」的守门人：
第一行 `if getattr(module, "_is_hf_initialized", False): return`（`:2441`），否则 `self._init_weights(module)`
并打上 `module._is_hf_initialized = True`。**这就是第四节 `set_param_for_module` 里 `_is_hf_initialized = True`
（`core_model_loading.py:1382`）的消费点**。因果链：
`set_param_for_module 打标 → _initialize_weights 见标记直接 return → 已加载的权重不被覆盖`。
`initialize_weights`（`modeling_utils.py:2464`）是**支持复合模型**的递归分发器，靠 `smart_apply`（`:2471-2486`）：
它在**每一个 `PreTrainedModel` 边界上切换函数**，所以 `LlavaForConditionalGeneration` 里嵌的 `LlavaModel` /
`CLIPVisionModel` 各用自己的 `_init_weights`，而不是最外层那个（否则每个复合模型都要在最外层手动递归，
「extremely error prone and inefficient」）。`@init.guard_torch_init_functions()` 装饰器（`modeling_utils.py:2463`）
与 `transformers.initialization`（`initialization.py`，340 行）配合，在 `init.*` 调用时**自动打 `_is_hf_initialized` 标记**——
这是为什么模型代码里用 `init.normal_` 而不是 `torch.nn.init.normal_`。

---

## 六、missing / unexpected / mismatched

### 6.1 定义与记录点

定义在 `utils/loading_report.py:156`（docstring 原文 `:156-186`）：`missing_keys` = 模型要、checkpoint 没有；
`unexpected_keys` = checkpoint 有、模型不要；`skipped_pp_keys` = 属于别的流水线 rank、故意不加载；
`mismatched_keys`（三元组 set `(参数名, ckpt 形状, 模型形状)`）= 两边都有但形状不同；
`error_msgs` = 加载硬错误；`conversion_errors` = 权重变换过程抛的异常。

| 类别 | 一句话直觉 | 记录点 / 代码 |
|------|-----------|--------------|
| **missing** | **模型多了东西**（或 ckpt 少了） | 初始值 `core_model_loading.py:1615` `missing_keys=set(meta_model_state_dict.keys())`；加载成功时 `:1367` `discard(target_name)` |
| **unexpected** | **checkpoint 多了东西**（或模型少了） | `:1744-1748` `_add_unmatched_checkpoint_key`；属性不存在时 `:1360` `unexpected_keys.add(target_name)` |
| **mismatched** | **对上了名字，对不上形状** | `:1373` `mismatched_keys.add((target_name, param_value.shape, expected_shape))` |
| `error_msgs` | **会直接 `RuntimeError`** | `utils/loading_report.py:289-317` |
| `conversion_errors` | **会打报告，但单个参数被跳过** | `core_model_loading.py:1294` |
| `skipped_pp_keys` | **正常，不算 unexpected** | `:1485`（docstring `:1489-1495`） |

```python
# core_model_loading.py:1614-1621
# We start from all missing keys, and we will remove/add them from the proper containers as loading advances
loading_info = LoadStateDictInfo(missing_keys=set(meta_model_state_dict.keys()), unexpected_keys=set(),
    mismatched_keys=set(), conversion_errors={}, error_msgs=[], skipped_pp_keys=set())
```

**所以「missing」在语义上是「走完全程也没被 discard 掉的那些」**——这解释了为什么一个参数如果转换失败
（`conversion_errors`），它同时也会是 missing：两者是「症状」和「原因」的关系。
`skipped_pp_keys`（新引入，`:1485`）的动机：流水线并行下每个 stage 都收到整个模型的 key，
但本地 state dict 只有本 stage 的参数，属于别的 stage 的 key 记成 intentionally skipped 而不是 unexpected。

### 6.2 报告：`log_state_dict_report` 与真实的报错文本

`_finalize_model_loading`（`modeling_utils.py:4550`）的 `finally` 块里**一定会调用**报告函数（`:4577-4585`，
传入 `model` / `pretrained_model_name_or_path` / `ignore_mismatched_sizes` / `loading_info` / `logger`）。
`log_state_dict_report`（`utils/loading_report.py:273`）分三段（`:289-317`）：**先抛硬错误，再打报告，最后按条件抛异常**——
`error_msgs` 非空时立刻 `RuntimeError`（`size mismatch` 还会附赠「考虑 `ignore_mismatched_sizes=True`」的提示），
否则用 `logger.warning` 打出 `LOAD REPORT`，之后 `conversion_errors` 非空再 `RuntimeError`，
`mismatched_keys` 非空且未开 `ignore_mismatched_sizes` 再 `RuntimeError`。

| 触发条件 | 行为 | 能否绕过 |
|---------|------|---------|
| `error_msgs` 非空 | **最高优先级 `RuntimeError`**，报告都不打 | 否 |
| 报告内容非空 | 用 `logger.warning` 打表 | — |
| `conversion_errors` 非空 | 报告之后 `RuntimeError` | 否（要修 conversion mapping） |
| `mismatched_keys` 非空 | 报告之后 `RuntimeError` | ✅ `ignore_mismatched_sizes=True` |
| 只有 `missing` / `unexpected` | **只打 warning，不抛异常** | 本来就是 warning |

表由 `_make_table`（`utils/loading_report.py:101`）生成，表头 `Key | Status`（终端宽度 > 200 时额外加 `Details` 列），
行状态四个字面量：`UNEXPECTED` / `MISSING` / `MISMATCH` / `CONVERSION`（`utils/loading_report.py:204-235`）；
MISMATCH 行由 `:222-232` 构造（`Reinit due to size mismatch - ckpt: ... vs model: ...`）。实际输出：

```
LlamaForCausalLM LOAD REPORT from: /path/to/ckpt
Key                                                | Status     | Details
---------------------------------------------------+------------+------------------
lm_head.weight                                     | UNEXPECTED |
model.layers.{0..31}.self_attn.rotary_emb.inv_freq | UNEXPECTED |
model.layers.5.mlp.gate.weight                     | MISSING    |
Notes:
- UNEXPECTED:  can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING:     those params were newly initialized because missing from the checkpoint. ...
- MISMATCH:    ckpt weights were loaded, but they did not match the original empty weight shapes.
```

**表里的 key 是被合并过的**——`update_key_name`（`utils/loading_report.py:38`）把 `layers.0.x`、`layers.1.x`…
合并成 `layers.{0, 1}.x`，所以 32 层的问题只占一行；`_DIGIT_RX`（`utils/loading_report.py:25-32`，定义在 `:25`）
是 `re.compile(r"(?<=\.)(\d+)(?=\.|$)")`，`_pattern_of` 用它把每个点间整数替换成 `*`。

### 6.3 白名单过滤与排查顺序

`_keys_to_ignore_on_load_missing` / `_keys_to_ignore_on_load_unexpected` 两个类属性声明白名单，
过滤逻辑在 `modeling_utils.py:4864`（实现 `:4868-4896`）。其中有两个**条件性的「自动白名单」**：
只有当模型**当前**确实有 `rotary_emb.inv_freq` / `position_ids` 这类 buffer 时，才把
`r"rotary_emb\.inv_freq"` / `r"(^|\.)position_ids$"` 加进 `unexpected_patterns`
（处理「旧 checkpoint 有、新模型没有」），最后用正则把匹配的 unexpected 从集合里剔除。
`_keys_to_ignore_on_load_unexpected` 的真实用法（全库 **123** 个文件里有）里两个动机不同：

- `models/t5/modeling_t5.py:1074-1075`（`:1075`）`_keys_to_ignore_on_load_unexpected = [r"decoder"]`：
  `T5EncoderModel` 只有 encoder，但完整 checkpoint 里必然有 `decoder.*` → 一次性忽略整个子树。
  **典型的「用 base model 的类加载完整 checkpoint」。**
- `models/glm_moe_dsa/modeling_glm_moe_dsa.py:666` `= [r"model\.layers\.78.*"]`：忽略**第 78 层**
  （MTP 的额外层，checkpoint 里有但默认模型不建）。**典型的「checkpoint 比模型多一层」。**

| 症状 | 最可能的原因 | 查什么 |
|------|-------------|--------|
| 大量 `UNEXPECTED` + 大量 `MISSING`，且名字成对相似 | **命名体系不同**（`to_q` vs `q_proj`、`Wq` vs `q_proj`） | `conversion_mapping.py` 里有没有这个 `model_type` 的条目 |
| 全部 `UNEXPECTED`（模型没有 missing） | 加载了**任务头不同**的 checkpoint（如用 base 权重加载 `ForCausalLM`） | 用对应的 auto 类；或看 `_keys_to_ignore_on_load_unexpected` |
| `...rotary_emb.inv_freq` 为 `UNEXPECTED` | 旧 checkpoint 的 RoPE buffer | **无害**，`modeling_utils.py:4872` 已自动过滤 |
| `lm_head.weight` 为 `MISSING` 且 `tie_word_embeddings=True` | 权重被 tie 到 `embed_tokens` | **无害**，`tie_weights`（`:2606`）会处理 |
| 单个 key 的 `MISMATCH` + `Reinit due to size mismatch` | **shape 对不上**（改过 `num_labels` / `vocab_size` / `hidden_size`） | `ignore_mismatched_sizes=True` 会**重新随机初始化**该参数 |
| `CONVERSION` 行 | `WeightConverter` 的 op 抛异常 | 看 `Details` 里的 traceback；问题在 `ConversionOptions` 的参数（dim / chunks） |
| 全局 `RuntimeError: Error(s) in loading state_dict` | `error_msgs` 非空 | 报告都不会打，直接看错误文本 |

---

## 七、磁盘上不存在的模型：`trust_remote_code`

### 7.1 三态判定与交互式确认

`auto_factory.py:355-362` 判定三个布尔量：`has_remote_code`（config 有 `auto_map` 且含本类名）、
`has_local_code`（`type(config)` 在 `cls._model_mapping` 里）、`explicit_local_code`（有本地代码但不在
`transformers.` 命名空间）；若 `auto_map` 的 `class_ref` 里带 `--`（如 `"sgugger/my-bert-model--modeling.MyBertModel"`），
说明代码在**另一个 repo** 里，`upstream_repo` 会被提出来供 `resolve_trust_remote_code`（`dynamic_module_utils.py:712`）
构造报错信息。`auto_map` 是 config.json 里的一段（由 `register_for_auto_class` 在 `modeling_utils.py:4612` 写入）：

```json
{"model_type": "my_awesome_model",
 "auto_map": {"AutoConfig": "configuration_my_awesome_model.MyAwesomeConfig",
              "AutoModelForCausalLM": "modeling_my_awesome_model.MyAwesomeForCausalLM"}}
```

`resolve_trust_remote_code`（`dynamic_module_utils.py:755-793`）三态：

| 条件 | 行为 |
|------|------|
| `has_local_code=True`（库内已有这个 model_type） | **静默 `False`**——不需要执行远端代码 |
| 有远端代码、无本地代码、未显式传参 | **交互式 `input()` 询问**（`:764`，SIGALRM 超时保护；CI 里 `TIME_OUT_REMOTE_CODE=0` 走 `_raise_timeout_error`） |
| 有远端代码、无本地代码、传了 `False` | **`ValueError`**（提示 `Please pass the argument trust_remote_code=True`） |

**这就是为什么 `trust_remote_code` 不传时会卡住**——它在等你敲 `y`/`n`。

### 7.2 远端代码被下载到哪、怎么加载

`get_class_from_dynamic_module`（`dynamic_module_utils.py:516`）是入口（`:606-615`）：
`class_reference` 带 `--` 时拆出真正的 `repo_id`，否则用 `pretrained_model_name_or_path`；
按 `.` 拆出 `module_file` 与 `class_name`，交给 `get_cached_module_file`（`:346`）下载并落到模块缓存，
再 `get_class_in_module` 取出类。`get_cached_module_file` 三件事：① **下载**走和权重**完全一样的 `cached_file`**（`:417`）——
这也是 `trust_remote_code` 会继承所有 hub 参数的原因；② **校验依赖**：`check_imports(resolved_module_file)`（`:442`）；
③ **落到模块缓存**（`:413`，`:451-453`）：`full_submodule = TRANSFORMERS_DYNAMIC_MODULE_NAME + os.path.sep + submodule`，
`submodule_path = Path(HF_MODULES_CACHE) / full_submodule`，最终落点是
`~/.cache/huggingface/modules/transformers_modules/<owner>/<repo>/modeling_foo.py`。
**本地路径**的模型额外套一层源码 hash（`:443-448`），同一个目录改一次 `.py` 缓存目录就变一次，
避免「改了源码但加载到旧模块」。
`get_class_in_module`（`:266`）用内容 hash（模块文件 + 所有相对 import 文件的 sha256）决定要不要重新 exec，
exec 在进程级锁 `_HF_REMOTE_CODE_LOCK`（`:289`）保护下进行，代码在 `:288-307`；
docstring（`:534-535`）警告 `Calling this function will execute the code in the module file`。
**`exec_module` 就是 `trust_remote_code` 里「trust」所指的那一行**。
之后模型类完全走正常流程：`auto_factory.py:382` 先 `cls.register(config.__class__, model_class, exist_ok=True)`
注册进 `_model_mapping`，再 `:385-387` 调 `model_class.from_pretrained(...)`。
**`trust_remote_code` 只替换了「模型类从哪来」这一步，下载、建模、灌权重一模一样**；
那句 `cls.register(...)` 也解释了「第一次加载远程模型后，`AutoConfig` 就能认它了」。

---

## 八、实操排查手册

### 8.1 遇到「weights not used in ...」/ unexpected keys

报错文案来自 `from_pretrained` 的 docstring（`modeling_utils.py:3908-3909`）：

> The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY,
> therefore those weights are discarded.

新版会渲染成 `LOAD REPORT` 里的 `UNEXPECTED` 行。**按这个顺序查**：

| 步 | 假设 | 怎么验证 / 怎么修 |
|----|------|------------------|
| 1 | **用 base 类加载了带任务头的 checkpoint** | 换用对应的 Auto 类。官方承认的这类情况会写进白名单，如 `models/t5/modeling_t5.py:1075` `_keys_to_ignore_on_load_unexpected = [r"decoder"]` |
| 2 | **名字体系不同**（`to_q` vs `q_proj`） | `grep -n '"<model_type>"' src/transformers/conversion_mapping.py`；有映射但没生效 → 看是 class name 还是 `model_type` 命中（`get_model_conversion_mapping`，`conversion_mapping.py:1902`，**class name 优先**） |
| 3 | **嵌套子模型导致的 scope 问题** | 多模态模型里 `XxxForConditionalGeneration` 套 `XxxModel`，看 `scope_prefix` 有没有被正确设置（`conversion_mapping.py:1980-1984`） |
| 4 | **`base_model_prefix` 层面的差异**（BaseModel ↔ ForCausalLM） | `rename_source_key` 第 3 步（`core_model_loading.py:1472-1480`）会自动补/去前缀，**但前提是 key 的其余部分完全一致** |

### 8.2 遇到「size mismatch」

真实报错文本由 `log_state_dict_report`（`utils/loading_report.py:288-296`，`:290-296`）拼出：`error_msgs` 非空时
把消息 join 起来，若含 `"size mismatch"` 就追加「`You may consider adding ignore_mismatched_sizes=True ...`」，
再 `raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}: ...")`。

| 步 | 查什么 | 线索 |
|----|--------|------|
| 1 | **差在哪个维度** | 报告里的 MISMATCH 行给完整形状对比（`loading_report.py:231`）：<br>`"Reinit due to size mismatch - ckpt: torch.Size([32000, 4096]) vs model:torch.Size([128256, 4096])"`<br>最后一维常见于 `vocab_size`；第一维常见于 `hidden_size` / `intermediate_size` / `num_labels` |
| 2 | 是 `vocab_size` 吗 | 加过 special token 就要 `resize_token_embeddings`（`modeling_utils.py:2709`） |
| 3 | 是 `num_labels` 吗 | 分类头；只有 `hf_quantizer is None` 时才会判 mismatched（`core_model_loading.py:1372`） |
| 4 | 想「容忍」它 | `ignore_mismatched_sizes=True`：形状不对的参数**不加载**，退回随机初始化（`_initialize_missing_keys`，`modeling_utils.py:4828`）。这叫 **reinit**，不是「截断/补齐」 |
| 5 | name 对不上但形状对？ | 那不是 mismatch，是 unexpected + missing 这一对 → 回 8.1 |

**一个易踩的坑**：`mismatched` 的参数会被 `set_param_for_module` **从 missing 里 discard 掉**（`core_model_loading.py:1367`）
但不 `setattr`，所以它在 missing 集合里**看不到**——排查时**不能只看 missing/unexpected，必须看 mismatched 那一栏**。

### 8.3 遇到「CONVERSION」行

这是 `WeightConverter` 的 op 抛了异常（`core_model_loading.py:1294`，拼装 `:1313-1321`）：
把 traceback 拼成字符串，连同 `Error: {op_name} on tensors destined for {target_keys}. Ckpt contains: {length}`
写进 `loading_info.conversion_errors[first_target_key]`。

| 报告里的文本 | 含义 |
|------|------|
| `Error: Chunk on tensors destined for model.layers.0.q_proj.weight. Ckpt contains: 1` | `Chunk` 在目标参数上失败；**checkpoint 里只提供了 1 个张量**（而 op 期望 N 个） |
| `via <OpName>` | 失败的 op 名 |

**`Ckpt contains: N` 是最有用的线索**——N 与 op 期望值不符通常说明 **source_pattern 没匹配上全部需要的 key**。
排查方向：① `conversion_mapping.py` 里这个 `model_type` 的 `source_patterns` 写对了吗
（`*.` 是层级通配，`core_model_loading.py:849`，不是 glob）；② op 的 `dim` / `chunks` 参数对吗；
③ 是不是换了 checkpoint 格式（fp8 版）——量化器会前插 op（`conversion_mapping.py:2003`）。

### 8.4 一张总览：症状 → 机制 → 位置

| 症状 | 底层机制 | 位置 |
|------|---------|------|
| 加载了本地同名目录而不是 Hub 模型 | 本地目录优先且互斥 | `hub.py:391`, `:404` |
| 反复下载同一个文件 | 缓存命中失败（`try_to_load_from_cache` 返回 `_CACHED_NO_EXIST`） | `hub.py:417` |
| `HF_HUB_OFFLINE=1` 但仍尝试联网 | `is_offline_mode()` 未生效（该分支在 `hub.py:379` 强制 `local_files_only`） | `hub.py:379` |
| 加载时内存爆了 | `disable_mmap` / hf-mount 走了全读路径 | `modeling_utils.py:4507`, `_is_on_hf_mount` `:293` |
| 参数初始化的 std 不对 | `_init_weights` 的 std 兜底链取错了一级 | `modeling_utils.py:2380-2388` |
| 加载的权重被重新初始化了 | `_is_hf_initialized` 标记没打上 | `core_model_loading.py:1382` |
| norm 层的权重变成了全 1 | 基类 `_init_weights` 的字符串匹配分支命中 | `modeling_utils.py:2411-2414` |
| 保存后权重名和加载前不一致 | `_was_used` 为 `False`，逆变换没生效 | `core_model_loading.py:981`, `:1798` |
| 交互式卡住等输入 | `resolve_trust_remote_code` 的 `input()` | `dynamic_module_utils.py:764` |

---

## 一句话总结

> **`from_pretrained` 是一个调度器，它把「取文件」和「写参数」两件事用一行 `cls(config)` 分开。**
> 上行：`download_kwargs`（`modeling_utils.py:4167`）→ `hub.cached_files`（`hub.py:299`）→ 三选一 → `checkpoint_files`；
> 中间：`get_init_context`（`:3791`）在 **meta device** 上造空壳，`_get_dtype`（`:815`）和
> `_get_device_map`（`integrations/accelerate.py:338`）把 dtype 和目标设备定死；
> 下行：`convert_and_load_state_dict_in_model`（`core_model_loading.py:1511`）用 `safe_open` 惰性句柄 + 线程池 +
> `rename_source_key` + `WeightConverter` 把每个张量搬到该去的地方，落进 `set_param_for_module`（`:1343`）并打上
> `_is_hf_initialized`；收尾：`_finalize_model_loading`（`:4550`）把**没打上标记的**参数交给 `_init_weights`（`:2374`）初始化。
>
> **三句话记忆**：① `WeightRenaming` 只改名字，`WeightConverter` 改名字 + 动张量。
> ② `missing` = 模型的参数没被填上；`unexpected` = checkpoint 的 key 没处放；`mismatched` = 名字对上了但形状不对
> （**会被 reinit，不会进 missing**）。③ `_init_weights` 只碰**没被 checkpoint 覆盖**的参数——判据是 `_is_hf_initialized`。

---

## 验证清单

以下命令都在仓库根目录跑过，输出即文中所引（`# →` 后面是实测输出）。

```bash
# ── 0. 代码基线 ────────────────────────────────────────────────
git log -1 --format="%H"
#   → 42ca97014c85d71a88ad60d55f08cb9fb4d26e2c

wc -l src/transformers/modeling_utils.py src/transformers/core_model_loading.py \
      src/transformers/utils/hub.py src/transformers/conversion_mapping.py \
      src/transformers/utils/loading_report.py src/transformers/dynamic_module_utils.py
#   → 5206 / 1853 / 972 / 2005 / 318 / 850

# ── 1. from_pretrained 的入口位置 ───────────────────────────────
grep -n "def from_pretrained" src/transformers/modeling_utils.py \
      src/transformers/models/auto/auto_factory.py src/transformers/configuration_utils.py
#   → auto_factory.py:261（from_config 分支）、auto_factory.py:461
#     configuration_utils.py:619、modeling_utils.py:3881

# ── 2. 参数分流与 download_kwargs ───────────────────────────────
sed -n '4124,4152p;4164,4176p' src/transformers/modeling_utils.py
#   → 4130: dtype = kwargs.pop("dtype", None)
#     4132: device_map = kwargs.pop("device_map", None)
#     4164: if is_offline_mode() and not local_files_only:
#     4167: download_kwargs = {

# ── 3. 权重文件名常量 ───────────────────────────────────────────
sed -n '280,284p' src/transformers/utils/__init__.py
#   → WEIGHTS_NAME / WEIGHTS_INDEX_NAME / SAFE_WEIGHTS_NAME / SAFE_WEIGHTS_INDEX_NAME / CONFIG_NAME

# ── 4. hub 层：本地目录优先 / 缓存 / 下载 ────────────────────────
grep -n "def cached_file\|def cached_files\|def get_checkpoint_shard_files\|if os.path.isdir(path_or_repo_id):\|cache_dir = constants.HF_HUB_CACHE\|try_to_load_from_cache(\|hf_hub_download(\|snapshot_download(" \
     src/transformers/utils/hub.py
#   → 239 / 299 / 859；391,404（isdir）；408（HF_HUB_CACHE 默认值）
#     417 / 438 / 453

# ── 5. 动态模块缓存目录 ─────────────────────────────────────────
grep -n "HF_MODULES_CACHE\|TRANSFORMERS_DYNAMIC_MODULE_NAME" src/transformers/utils/hub.py
#   → 108: HF_MODULES_CACHE = os.getenv("HF_MODULES_CACHE", os.path.join(constants.HF_HOME, "modules"))
#     109: TRANSFORMERS_DYNAMIC_MODULE_NAME = "transformers_modules"

# ── 6. 建模上下文（meta device）────────────────────────────────
sed -n '3791,3820p' src/transformers/modeling_utils.py
#   → 3818: init_contexts.extend([torch.device("meta"), init.meta_device_safe_creation_ops()])

# ── 7. safetensors 惰性加载 ─────────────────────────────────────
grep -n "file_pointer = safe_open(\|get_slice(k)" src/transformers/modeling_utils.py
#   → 4524: file_pointer = safe_open(file, framework="pt", device=device, backend=backend)
#     4527:     merged_state_dict[k] = file_pointer.get_slice(k)  # don't materialize yet

# ── 8. 权重加载主循环的关键行 ───────────────────────────────────
grep -n "def convert_and_load_state_dict_in_model\|state_dict = sorted(state_dict.items()\|rename_source_key(\|param_name_to_load.setdefault\|spawn_materialize(\|mapping.convert(\|set_param_for_module(" \
     src/transformers/core_model_loading.py | head -20
#   → 1511 / 1647 / 1650 / 1666 / 1730 / 1754 / 1770

# ── 9. `_is_hf_initialized` 的写入与消费 ─────────────────────────
grep -n "_is_hf_initialized" src/transformers/core_model_loading.py src/transformers/modeling_utils.py | head
#   → core_model_loading.py:1382: param_value._is_hf_initialized = True
#     modeling_utils.py:2441: if getattr(module, "_is_hf_initialized", False): return

# ── 10. WeightTransform 家族与 `*.` 的编译 ──────────────────────
grep -n "^class WeightTransform\|^class WeightRenaming\|^class GroupWeightRename\|^class PrefixChange\|^class WeightConverter\|source_pattern.replace" \
     src/transformers/core_model_loading.py
#   → 769 / 992 / 1032 / 1098 / 1152；849: pattern = source_pattern.replace("*.", r".*\.")

# ── 11. conversion_mapping 的规模 ───────────────────────────────
grep -c "WeightRenaming(" src/transformers/conversion_mapping.py        # → 621
grep -c "WeightConverter(" src/transformers/conversion_mapping.py       # → 55
grep -c "PrefixChange("    src/transformers/conversion_mapping.py       # → 11
grep -cE '^        "[a-zA-Z0-9_]+": \[$' src/transformers/conversion_mapping.py   # → 84

# ── 12. llama / mistral / qwen3 不覆写 _init_weights（实测）─────
grep -c "_init_weights" src/transformers/models/llama/modeling_llama.py        # → 0
grep -c "_init_weights" src/transformers/models/mistral/modeling_mistral.py    # → 0
grep -c "_init_weights" src/transformers/models/qwen3/modeling_qwen3.py        # → 0

# ── 13. 覆写 _init_weights 的模型 / 基类的类名匹配分支 ───────────
grep -n "def _init_weights" src/transformers/models/{bert,t5,mixtral,glm_moe_dsa}/modeling_*.py
#   → bert:547 / t5:563 / mixtral:399 / glm_moe_dsa:670
grep -n '"RMSNorm" in module.__class__.__name__\|"RotaryEmbedding" in module.__class__.__name__' \
     src/transformers/modeling_utils.py
#   → 2414 / 2427
grep -n "original_inv_freq" src/transformers/models/llama/modeling_llama.py
#   → 89: self.original_inv_freq = nn.Buffer(inv_freq.clone(), persistent=False)

# ── 14. LoadStateDictInfo 的三类 key 定义 ───────────────────────
sed -n '156,186p' src/transformers/utils/loading_report.py

# ── 15. 真实报错文本 ────────────────────────────────────────────
grep -n "Error(s) in loading state_dict for\|We encountered some issues during automatic conversion\|ignore_mismatched_sizes\` to \`False" \
     src/transformers/utils/loading_report.py
#   → 295 / 311 / 315

# ── 16. mismatched 会被从 missing 里移除 ────────────────────────
sed -n '1360,1382p' src/transformers/core_model_loading.py
#   → 1360: loading_info.unexpected_keys.add(target_name)
#     1367: loading_info.missing_keys.discard(target_name)
#     1373: loading_info.mismatched_keys.add((target_name, param_value.shape, expected_shape))
#     1382: param_value._is_hf_initialized = True

# ── 17. 白名单机制 ──────────────────────────────────────────────
grep -n "_keys_to_ignore_on_load_unexpected = " src/transformers/models/t5/modeling_t5.py \
      src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py
#   → t5/modeling_t5.py:1075  = [r"decoder"]
#     glm_moe_dsa/modeling_glm_moe_dsa.py:666 = [r"model\.layers\.78.*"]
grep -rl "_keys_to_ignore_on_load_unexpected = " src/transformers/models/ | wc -l   # → 123

# ── 18. 报告表格如何合并层号 ────────────────────────────────────
sed -n '25,32p' src/transformers/utils/loading_report.py
#   → _DIGIT_RX = re.compile(r"(?<=\.)(\d+)(?=\.|$)")

# ── 19. trust_remote_code 的交互与动态模块缓存 ───────────────────
grep -n "def resolve_trust_remote_code\|def get_class_from_dynamic_module\|def get_cached_module_file\|def get_class_in_module" \
     src/transformers/dynamic_module_utils.py
#   → 712 / 516 / 346 / 266
grep -n "full_submodule = TRANSFORMERS_DYNAMIC_MODULE_NAME\|submodule_path = Path(HF_MODULES_CACHE) / full_submodule\|_HF_REMOTE_CODE_LOCK" \
     src/transformers/dynamic_module_utils.py
#   → 451 / 453 / 289

# ── 20. auto 层如何选类 ─────────────────────────────────────────
grep -n "def _get_model_class" src/transformers/models/auto/auto_factory.py     # → 178
sed -n '355,410p' src/transformers/models/auto/auto_factory.py
```

> **没有跑的验证**（按任务要求，不联网）：
> 没有实际执行任何 `from_pretrained("meta-llama/...")` 之类的下载，
> 也没有跑 `make fix-repo` / `python utils/modular_model_converter.py` 这类会改写仓库的命令。
> 本机 `~/.cache/huggingface/` 不存在，因此第 2.4 节的缓存目录结构是**按 `huggingface_hub` 的
> 约定**（`HF_HOME` / `HF_HUB_CACHE` 由该库定义，见 `hub.py:36` 的
> `from huggingface_hub import constants`）描述的，不是本机实测的 `ls` 结果。

---

## 导航

- 路线图：[00_roadmap.md](./00_roadmap.md)（本篇 = 入门 · 架构 · 1A3）
- 上一篇：[1A2_config_system.md](./1A2_config_system.md)
- 下一篇：[1A4_tokenizer.md](./1A4_tokenizer.md)
- **相关**：
  - [2A5_modular_code_generation.md](./2A5_modular_code_generation.md)——读本篇时看到 `modeling_*.py`
    里缺 `_init_weights`，用 2A5 的视角就能明白「缺的是继承，不是实现」
  - [2A6_weights_quant_parallel.md](./2A6_weights_quant_parallel.md)——本篇第四节的延伸：
    `WeightConverter` 的全量算子、量化器如何插 op、张量并行如何分片
