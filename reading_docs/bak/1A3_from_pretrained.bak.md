# `from_pretrained` 全链路精读：从 repo id 到显存

> 阅读目标：把 `from_pretrained` 从「一行 API」拆成一条**可追踪的调用链**——
> 哪些参数在哪一层被吃掉的、`config.json` 和 `*.safetensors` 分别怎么被取回、
> 空壳模型什么时候被造出来、checkpoint 里的权重名怎么变成模块参数名、
> 没被权重覆盖的参数又由谁初始化。
>
> **这篇为什么是架构线的枢纽**：路线图里 1A1 → 1A3 是「模型怎么被造出来」这条线，
> 1A4 → 1A5 → 1A6 是「数据怎么进去、结果怎么出来」这条线。
> 两条线**唯一的耦合点**就是本文的主角——训练（1A6 `Trainer`）和推理（1A5 `generate`）
> 都要先经过它把模型造出来。
>
> 源文件：
> - `src/transformers/modeling_utils.py`（5206 行）——入口与编排
> - `src/transformers/core_model_loading.py`（1853 行）——**权重加载的心脏**（本篇重点）
> - `src/transformers/utils/hub.py`（972 行）——下载与缓存
> - `src/transformers/conversion_mapping.py`（2005 行）——权重名映射表
> - `src/transformers/utils/loading_report.py`（318 行）——missing/unexpected 报告的生成
> - `src/transformers/dynamic_module_utils.py`（850 行）——`trust_remote_code` 的动态加载
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

- [1A1_repo_and_auto.md](./1A1_repo_and_auto.md)：`AutoModel` 如何从 config 找到类、`models/<name>/` 五件套
- [1A2_config_system.md](./1A2_config_system.md)：`PreTrainedConfig` 的字段如何决定建模结构
- [2A5_modular_code_generation.md](./2A5_modular_code_generation.md)：`modular_*.py` 与生成文件的关系
  （本篇会用到它的结论：看到 `modeling_*.py` 里没有 `_init_weights` 时，**不代表它没有**，
  可能是父模型继承来的）

> **一个前置结论**：本库里 `PreTrainedModel` 已经把「造模型」拆成了三个可覆写的钩子——
> `_from_config`（建模上下文）、`__init__`（结构）、`_init_weights`（初始化）。
> `from_pretrained` 本身**不实现**这三件事中的任何一件，它只负责**编排**。
> 读本篇时始终记住这条分界线：**`from_pretrained` 是调度器，不是实现者。**

---

## 零、总览：一张调用链图

先看全局，后面每一节都是在放大这张图的某个框。

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

**读图要点**：整条链路只有一个真正的「分水岭」——`model = cls(config, ...)`（`modeling_utils.py:4350`）。
**它之上是「读文件」，它之下是「写参数」**。所有排查问题的直觉都应该建立在这条分界线上：
出问题先问「是模型结构不对，还是权重没对上」。

---

## 一、入口层：`from_pretrained` 的参数分流

### 1.1 两个 `from_pretrained`，不是一个

调用 `AutoModelForCausalLM.from_pretrained(...)` 时，**先经过 auto 类**，再落到具体模型类：

| 层 | 位置 | 职责 |
|----|------|------|
| `_BaseAutoModelClass.from_pretrained` | `auto_factory.py:461` | 解析 config、判定本地/远端代码、**选出模型类** |
| `PreTrainedModel.from_pretrained` | `modeling_utils.py:3881` | 下载权重、建模、灌权重、初始化 |

auto 层做的事（`auto_factory.py:461-386`）：

```python
# auto_factory.py:461-473（节选；列表原本每项一行，这里折成一行）
def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
    config = kwargs.pop("config", None)
    trust_remote_code = kwargs.get("trust_remote_code")
    kwargs["_from_auto"] = True               # ← 广播一个「我是从 Auto 来的」标记
    hub_kwargs_names = ["cache_dir", "force_download", "local_files_only",
                        "proxies", "revision", "subfolder", "token"]
    hub_kwargs = {name: kwargs.pop(name) for name in hub_kwargs_names if name in kwargs}
```

注意这 7 个 **hub 参数在 auto 层就被摘出来了**，之后以 `**hub_kwargs` 的形式透传给
`config_class.from_pretrained` 和 `model_class.from_pretrained`。
也就是说 `cache_dir` / `revision` / `local_files_only` **在 auto 层被识别，
但真正使用是在下两层的 `cached_file` 调用里**。

紧接着的一个细节很值得注意（`auto_factory.py:282-296`）：

```python
# auto_factory.py:282-296（节选）— 为了拿 commit_hash，先「偷偷」下一次 config.json
if commit_hash is None:
    if not isinstance(config, PreTrainedConfig):
        # We make a call to the config file first (which may be absent) to get the commit hash as soon as possible
        resolved_config_file = cached_file(
            pretrained_model_name_or_path, CONFIG_NAME,
            _raise_exceptions_for_gated_repo=False,
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
            **hub_kwargs,
        )
        commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
```

**为什么要这样**：后面所有文件（config、权重、tokenizer、远端代码）都要按**同一个 commit**
去取，否则会出现「config 是 A 版本、权重是 B 版本」的错配。所以先用一次 `cached_file`
把 `config.json` 落到缓存里并读出它的 commit hash，后面全部带上 `_commit_hash=commit_hash`。
三个 `_raise_exceptions_for_*=False` 是为了**容忍失败**——这一步失败不该直接报错，
因为 config 后面还会正式读一次。

然后再走 `AutoConfig.from_pretrained`（`auto_factory.py:336`）和模型类选择（`auto_factory.py:355-410`）：

```python
# auto_factory.py:355-366
has_remote_code = hasattr(config, "auto_map") and cls.__name__ in config.auto_map
has_local_code = type(config) in cls._model_mapping
explicit_local_code = has_local_code and not _get_model_class(config, cls._model_mapping).__module__.startswith("transformers.")
...
trust_remote_code = resolve_trust_remote_code(
    trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code, upstream_repo=upstream_repo
)
kwargs["trust_remote_code"] = trust_remote_code
```

### 1.2 参数在哪一层被吃掉：一张对照表

| 参数 | 在哪一层首次被识别 | 最终消费点 |
|------|-------------------|-----------|
| `cache_dir` | `auto_factory.py:265`（进 `hub_kwargs`） | `hub.py:408` `cache_dir = constants.HF_HUB_CACHE` 的默认值被覆盖处 |
| `revision` | 同上 | `hub.py:438` `hf_hub_download(revision=...)` |
| `local_files_only` | 同上；另外 `modeling_utils.py:4164` 在离线模式下被强制置 `True` | `hub.py:379` |
| `subfolder` | 同上 | `hub.py:386` |
| `force_download` | 同上；`modeling_utils.py:4176` 也进 `download_kwargs_with_commit` | `hub.py:438` |
| `token` | 同上 | `hub.py:438` |
| `state_dict` | **只在** `modeling_utils.py:4124` | `_load_pretrained_model`（跳过读盘） |
| `dtype` / `torch_dtype` | `modeling_utils.py:4130-4131` | `_get_dtype`（`modeling_utils.py:815`） |
| `device_map` | `modeling_utils.py:4132` | `check_and_set_device_map`（`integrations/accelerate.py:96`）→ `_get_device_map`（同文件 `:338`） |
| `quantization_config` | `modeling_utils.py:4136` | `get_hf_quantizer`（`quantizers/auto.py:334`） |
| `trust_remote_code` | `modeling_utils.py:4148`；**auto 层也用**（`auto_factory.py:366`） | `dynamic_module_utils.resolve_trust_remote_code`（`:712`） |

`modeling_utils.py:4124-4152` 这一整块就是「分流器」——**它把 kwargs 拆成三类**：

```python
# modeling_utils.py:4124-4152（节选）
state_dict = kwargs.pop("state_dict", None)
...
dtype = kwargs.pop("dtype", None)
torch_dtype = kwargs.pop("torch_dtype", None)  # kept for BC
device_map = kwargs.pop("device_map", None)
...
trust_remote_code = kwargs.pop("trust_remote_code", None)
...
key_mapping = kwargs.pop("key_mapping", None)

# Not used anymore -- remove them from the kwargs
for name in ["mirror", "_fast_init", "low_cpu_mem_usage", "from_tf", "from_flax", "offload_state_dict"]:
    _ = kwargs.pop(name, None)
```

**第三类**（`low_cpu_mem_usage` / `_fast_init` / `offload_state_dict` …）特别值得注意：
它们被**静默丢弃**。这几个是 v4 时代的开关，现在的实现已经默认就是
「meta device 建模 + 流式加载」，所以参数留着只为不破坏旧代码。
如果你读老教程看到 `low_cpu_mem_usage=True` 的说明，可以直接跳过——
在新版里它**什么都不做**。

剩下没被 pop 的 kwargs 会走两条路（`modeling_utils.py:4233-4256`）：

```python
# modeling_utils.py:4233
if not isinstance(config, PreTrainedConfig):
    config_path = config if config is not None else pretrained_model_name_or_path
    config_class = cls.config_class
    ...
    config, model_kwargs = config_class.from_pretrained(
        config_path, return_unused_kwargs=True, gguf_file=gguf_file,
        _from_auto=from_auto_class, _from_pipeline=from_pipeline,
        **download_kwargs, **kwargs,
    )
    ...
    commit_hash = model_kwargs.pop("_commit_hash", commit_hash)
else:
    config = copy.deepcopy(config)
    model_kwargs = kwargs                       # ← 显式传 config 时，全部 kwargs 都进模型 __init__
    commit_hash = getattr(config, "_commit_hash", commit_hash)
```

**这条 `return_unused_kwargs=True` 的机制是理解「参数去哪了」的钥匙**：

- 用户传的 kwarg 如果**是 config 的字段** → 被 config 吃掉（比如 `num_labels=10`）
- 如果**不是** → 原样留在 `model_kwargs` 里，最后喂给 `cls(config, *model_args, **model_kwargs)`（`:4350`）

所以 `BertModel.from_pretrained(path, output_attentions=True)` 之所以能生效，
是因为 `output_attentions` 恰好是 `PreTrainedConfig` 的字段。

---

## 二、下载层：`utils/hub.py` 与本地缓存

### 2.1 唯一的下载入口：`cached_files`

`cached_file`（`hub.py:239`）是个薄壳，真正干活的是 `cached_files`（`hub.py:299`）：

```python
# hub.py:294-297（cached_file 的全部函数体）
def cached_file(path_or_repo_id, filename, **kwargs) -> str | None:
    file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    file = file[0] if file is not None else file
    return file
```

`cached_files` 的判定逻辑（`hub.py:373-466`）可以概括成三步：

```python
# hub.py:379
if is_offline_mode() and not local_files_only:
    logger.info("Offline mode: forcing local_files_only=True")
    local_files_only = True
...
# hub.py:386
full_filenames = [os.path.join(subfolder, file) for file in filenames]

# ── 第 1 步：本地目录 ────────────────────────────────
# hub.py:391
for filename in full_filenames:
    if os.path.isdir(path_or_repo_id):
        resolved_file = os.path.join(path_or_repo_id, filename)
        if not os.path.isfile(resolved_file):
            if _raise_exceptions_for_missing_entries and filename != os.path.join(subfolder, "config.json"):
                ...raise OSError(f"{path_or_repo_id} does not appear to have a file named {filename}...")
            else:
                continue
        existing_files.append(resolved_file)

# hub.py:404
if os.path.isdir(path_or_repo_id):
    return existing_files if existing_files else None          # ← 本地目录直接返回，不碰网络

# ── 第 2 步：缓存 ───────────────────────────────────
# hub.py:408
if cache_dir is None:
    cache_dir = constants.HF_HUB_CACHE
...
# hub.py:417
resolved_file = try_to_load_from_cache(
    path_or_repo_id, filename, cache_dir=cache_dir, revision=_commit_hash, repo_type=repo_type
)

# ── 第 3 步：下载 ───────────────────────────────────
# hub.py:438
hf_hub_download(path_or_repo_id, filenames[0], subfolder=..., revision=revision, ...)
# 或（多文件时）
# hub.py:453
snapshot_download(path_or_repo_id, allow_patterns=full_filenames, ...)
```

**三个判定要点**：

1. **本地路径优先且互斥**（`:391`/`:404`）。只要 `os.path.isdir(path_or_repo_id)` 成立，
   就直接拼路径返回，**完全不走 Hub**。这就是「我有一个同名本地目录，结果加载到了错的东西」这类
   bug 的根源——`from_pretrained("meta-llama/Llama-3.2-1B")` 如果当前目录下恰好有
   `meta-llama/Llama-3.2-1B/` 这个目录，库会优先用本地的。
   `modeling_utils.py:758` 的报错文案也特意提醒了这一点。
2. `constants.HF_HUB_CACHE`（`:408`）来自 **`huggingface_hub` 的 `constants`**，
   不是 transformers 自己的（`hub.py:36` 的 import 已经说明）。所以
   `HF_HOME` / `HF_HUB_CACHE` 这两个环境变量的语义由 `huggingface_hub` 定义。
3. `config.json` 是**唯一被特批可以不存在**的文件（`:395` 的 `filename != os.path.join(subfolder, "config.json")`）——
   因为 `AutoModel` 的第一步探测（`auto_factory.py:282`）可能只是想探一探 commit hash。

### 2.2 `config.json` 和 `model.safetensors` 分别是哪条路径取回的

| 文件 | 取回函数 | 调用点 | 说明 |
|------|---------|--------|------|
| `config.json` | `cached_file` | `configuration_utils.py:802`（`get_config_dict` 内） | 常量 `CONFIG_NAME`，`utils/__init__.py:284` |
| `model.safetensors` | `cached_file` | `modeling_utils.py:676`（`_get_resolved_checkpoint_files` 内） | 常量 `SAFE_WEIGHTS_NAME`，`utils/__init__.py:282` |
| `model.safetensors.index.json` | `cached_file` | `modeling_utils.py:681`（首次尝试失败后的 fallback） | `SAFE_WEIGHTS_INDEX_NAME`，`utils/__init__.py:283` |
| 各分片 `model-0000X-of-0000Y.safetensors` | `cached_files`（批量） | `hub.py:902`（`get_checkpoint_shard_files` 内） | 从 index.json 的 `weight_map` 反推 |
| `pytorch_model.bin` | `cached_file` | `modeling_utils.py:703` | 老格式，仅在 safetensors 全找不到时才用 |

四个权重文件名常量（`utils/__init__.py:280-284`）：

```python
WEIGHTS_NAME = "pytorch_model.bin"
WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"
```

**`variant` 参数是个小彩蛋**：`_add_variant`（`modeling_utils.py:527`）会把
`model.safetensors` 变成 `model.fp16.safetensors`：

```python
# modeling_utils.py:527
def _add_variant(weights_name: str, variant: str | None = None) -> str:
    if variant is not None:
        path, name = weights_name.rsplit(".", 1)
        weights_name = f"{path}.{variant}.{name}"
    return weights_name
```

所以 `from_pretrained(..., variant="fp16")` 找的是 `model.fp16.safetensors`。

### 2.3 分片：`get_checkpoint_shard_files`

当解析到的是 `*.index.json` 时，`is_sharded=True`，走 `hub.py:859`：

```python
# hub.py:887-892
with open(index_filename) as f:
    index = json.loads(f.read())

shard_filenames = sorted(set(index["weight_map"].values()))   # ← weight_map 的值去重
sharded_metadata = index["metadata"]
sharded_metadata["all_checkpoint_keys"] = list(index["weight_map"].keys())
sharded_metadata["weight_map"] = index["weight_map"].copy()
```

**`sharded_metadata` 后面会挂在 `LoadStateDictConfig` 上**（`modeling_utils.py:4379`），
它有两个用途：给 `_get_dtype` 提供 `metadata["dtype"]`（快速判断 dtype，
省掉「先读一个张量」这一步），以及给 disk offload / 并行加载提供 key→shard 的映射。

`index.json` 的结构决定了一切：

```json
{
  "metadata": {"total_size": 2468422144, "dtype": "bfloat16"},
  "weight_map": {
    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
    "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors"
  }
}
```

`weight_map` 的 **key 就是 checkpoint 的「原始权重名」**——这正是第四节要映射的对象。

### 2.4 本地缓存目录结构

```
~/.cache/huggingface/                       ← HF_HOME（huggingface_hub.constants 定义）
├── hub/                                    ← HF_HUB_CACHE（hub.py:408 的默认值）
│   └── models--meta-llama--Llama-3.2-1B/
│       ├── blobs/
│       │   ├── a1b2c3...                   ← 文件真实内容，按 sha256 命名
│       │   └── d4e5f6...
│       ├── refs/
│       │   └── main                        ← 内容是一行 commit sha
│       └── snapshots/
│           └── 8c6b1a.../                  ← 以 commit hash 命名的「视图」
│               ├── config.json      ──┐
│               └── model.safetensors ─┴─→ 符号链接，指回 ../../blobs/<sha256>
└── modules/                                ← HF_MODULES_CACHE（hub.py:108）
    └── transformers_modules/
        └── <repo_id>/...                   ← trust_remote_code 下载的 *.py，见第七节
```

**为什么要 `blobs` + `snapshots` 两层**：同一份字节在多个 commit / 多个 revision 下可能是相同的，
用内容寻址（`blobs/<sha256>`）只存一份，`snapshots/<commit>/` 里全是符号链接。
所以：**`ls snapshots/<commit>/` 看到的是「有哪些文件」，`du blobs/` 才是真实占用**。

`HF_MODULES_CACHE` 的定义就一行（`hub.py:108`）：

```python
# hub.py:108
HF_MODULES_CACHE = os.getenv("HF_MODULES_CACHE", os.path.join(constants.HF_HOME, "modules"))
TRANSFORMERS_DYNAMIC_MODULE_NAME = "transformers_modules"      # hub.py:109
```

### 2.5 disk offload 与 `device_map` 的交叉

`_get_resolved_checkpoint_files` 只负责「把文件弄到本地」，不管设备。
但 `_load_pretrained_model` 里有一个把两者串起来的动作（`modeling_utils.py:4461`）：

```python
# modeling_utils.py:4460-4468
if load_config.device_map is not None and "disk" in load_config.device_map.values():
    disk_offload_index = accelerate_disk_offload(
        model, load_config.disk_offload_folder, checkpoint_files,
        load_config.device_map, load_config.sharded_metadata, load_config.weight_mapping,
    )
```

**`device_map={"": "disk"}` 时，权重根本不进内存**——直接重写到 `offload_folder`，
留在 `disk_offload_index` 里，等 forward 时由 accelerate 的 hook 按需读回。

---

## 三、建模层：config → `cls(config)`

### 3.1 `_get_dtype`：`dtype="auto"` 到底怎么定的

`from_pretrained` 的默认 `dtype` 是 `"auto"`（`modeling_utils.py:4161-4162`）。
`_get_dtype`（`modeling_utils.py:815`）的优先级是**从便宜到昂贵**递进的：

```python
# modeling_utils.py:834-849（节选）
if dtype == "auto":
    if hasattr(config, "dtype") and config.dtype is not None:
        dtype = config.dtype
        logger.info(f"Will use dtype={dtype} as defined in model's config object")
    else:
        if is_sharded and "dtype" in sharded_metadata:
            dtype = sharded_metadata["dtype"]                  # ① 读 index.json 的 metadata
        elif state_dict is not None:
            dtype = get_state_dict_dtype(state_dict)           # ② 从内存里的 state_dict 猜
        elif checkpoint_files is not None and checkpoint_files[0].endswith(".gguf"):
            dtype = torch.float32
        else:
            state_dict = load_state_dict(
                checkpoint_files[0], map_location="meta", weights_only=weights_only
            )
            dtype = get_state_dict_dtype(state_dict)           # ③ 读第一个分片（meta，不占内存）
```

**这里的顺序很重要**：`metadata["dtype"]` 存在时**完全不读文件**，只看一个 JSON 字段。
这是「加载大模型时 `dtype` 判定很快」的原因。

最后一步是把这个 dtype 写进 config 和所有子 config（`modeling_utils.py:890-893`）：

```python
# modeling_utils.py:890
config.dtype = main_dtype
for sub_config_key in config.sub_configs:
    if (sub_config := getattr(config, sub_config_key)) is not None:
        sub_config.dtype = main_dtype
```

**`dtype` 在「建模前」就已经介入**，而且是通过修改 config 实现的——
所以 `cls(config)` 里的 `nn.Linear(...)` 会按正确的 dtype 创建。

### 3.2 `get_init_context`：为什么模型是「空壳」

这是整个建模阶段最容易被忽略、但最重要的一件事（`modeling_utils.py:3791`）：

```python
# modeling_utils.py:3795-3820（节选）
init_contexts = [local_torch_dtype(dtype, cls.__name__), init.no_tie_weights(), apply_patches()]
if allow_all_kernels:
    init_contexts.append(allow_all_hub_kernels())
if is_deepspeed_zero3_enabled():
    ...
else:
    # meta_device_safe_creation_ops patches torch.linspace to default to CPU
    init_contexts.extend([torch.device("meta"), init.meta_device_safe_creation_ops()])
return init_contexts
```

三个 context manager 各管一件事：

| Context manager | 位置 | 作用 |
|-----------------|------|------|
| `local_torch_dtype(dtype, cls.__name__)` | `modeling_utils.py:221` | 临时 `torch.set_default_dtype(dtype)`，退出时还原 |
| `init.no_tie_weights()` | `initialization.py` | 建模期间**禁止 tie**（等权重全灌完再 tie） |
| `torch.device("meta")` | — | **所有参数创建在 meta device 上：只占 shape，不占内存** |
| `init.meta_device_safe_creation_ops()` | `initialization.py` | 把 `torch.linspace` 等重定向到 CPU，让自定义模型在 meta 下也能 `.item()` |

**「meta device 建模」是理解 `from_pretrained` 内存行为的核心**：
一个 70B 模型 `cls(config)` 时**几乎不占内存**（只占参数形状的元数据），
显存/内存在第四节的 `spawn_materialize` 阶段才真正被填。
这也是为什么 `_move_missing_keys_from_meta_to_device`（`modeling_utils.py:4778`）
这个函数名里会出现 "from_meta"——**没被权重覆盖的参数会一直留在 meta 上**，
必须显式搬回真实设备才能初始化。

`local_torch_dtype` 的实现很短，但它对报错很友好（`modeling_utils.py:221-242`）：

```python
# modeling_utils.py:227
if not dtype.is_floating_point:
    if model_class_name is not None:
        error_message = (
            f"{model_class_name} cannot be instantiated under `dtype={dtype}` as it's not a floating-point dtype"
        )
    ...
    raise ValueError(error_message)

original_dtype = torch.get_default_dtype()
try:
    torch.set_default_dtype(dtype)
    yield
finally:
    torch.set_default_dtype(original_dtype)          # ← 一定会还原
```

### 3.3 `_from_config`：另一条入口

`cls(config)` 是一条路；`cls._from_config(config, **kwargs)` 是另一条（`modeling_utils.py:1480`）。
它的存在意义是：**`from_config` 不需要权重，但需要相同的建模上下文**。

```python
# modeling_utils.py:1480-1522（节选）
@classmethod
def _from_config(cls, config, **kwargs):
    dtype = kwargs.pop("dtype", config.dtype)
    ...
    config.dtype = dtype                       # ← 和 from_pretrained 一样先定 dtype
    for sub_config_key in config.sub_configs:
        if (sub_config := getattr(config, sub_config_key)) is not None:
            sub_config.dtype = dtype
    ...
    init_contexts = [apply_patches()]
    if dtype is not None:
        init_contexts.append(local_torch_dtype(dtype, cls.__name__))
    ...
    with ContextManagers(init_contexts):
        model = cls(config, **kwargs)           # ← 同一个 cls(config)，零权重加载
        patch_output_recorders(model)
```

**对照两张表**：

| | `_from_config`（`:1480`） | `from_pretrained` 的建模段（`:4346-4356`） |
|---|---|---|
| dtype context | 有 | 有 |
| `no_tie_weights()` | **无** | 有 |
| `torch.device("meta")` | **无** | 有 |
| ZeRO-3 | 有 | 有 |
| 之后做什么 | 直接返回（权重是随机初始化的） | 进入 `_load_pretrained_model` 灌权重 |

**所以 `AutoModel.from_config(config)` 造出来的是「随机权重、在真实设备上」的模型，
而 `from_pretrained` 造出来的是「随机权重、在 meta 上」的空壳。**
两者都调用了同一个 `cls(config)`，差别**只在 context manager**。
这正是「`from_config` 有 `_` 前缀（半私有）但依然是个正经钩子」的原因——
它是 `from_pretrained` 建模段的「无权重版本」。

### 3.4 `device_map` 的介入时机

`device_map` 的处理**分三次**，每次都更精确：

| 阶段 | 位置 | 做什么 |
|------|------|--------|
| ① 规范化 | `check_and_set_device_map`，`integrations/accelerate.py:96` | 把 `None`/`int`/`str`/`torch.device` 统一成 `dict`；顺便捕获 `torch.device` context manager |
| ② 扩展 | `_get_device_map`，`integrations/accelerate.py:338` | `"auto"`/`"balanced"` 时用 `infer_auto_device_map` 真正算分配方案 |
| ③ 逐参数查 | `core_model_loading.py:1717` `get_device(device_map, renamed_key, valid_torch_device=True)` | 加载时每个参数查一次自己的目标设备 |

**顺序上有个细节**：② 必须在 `cls(config)` **之后**（`modeling_utils.py:4372-4373`），
因为 `infer_auto_device_map` 需要模型实例才能知道 `_no_split_modules` 和参数量：

```python
# modeling_utils.py:4372-4373
if device_map is not None:
    device_map = _get_device_map(model, device_map, max_memory, hf_quantizer)
```

而 ① 在很早（`modeling_utils.py:4226`），因为 `get_hf_quantizer`（`:4270`）需要它。

---

## 四、权重加载层：`core_model_loading.py`（本篇重点）

### 4.1 `_load_pretrained_model`：读盘策略的三岔口

`modeling_utils.py:4436` 的 `_load_pretrained_model` 是「读文件」的总调度。它的主体是一个三分支：

```python
# modeling_utils.py:4477-4536（结构概览）
if is_deepspeed_zero3_enabled() and not is_quantized:
    # 分支 A：ZeRO-3 —— 先把所有分片读进内存合并成一个 state_dict
    ...
elif ... checkpoint_files[0].endswith(".safetensors") ...:
    # 分支 B：safetensors —— 用 safe_open 拿「切片句柄」，不实体化
else:
    # 分支 C：.bin —— torch.load 全读
```

**分支 B 是默认路径**，也是本节的主角（`modeling_utils.py:4500-4529`）：

```python
# modeling_utils.py:4504-4527（节选）
merged_state_dict = {}
for file in checkpoint_files:
    if load_config.disable_mmap or _is_on_hf_mount(file):
        with open(file, "rb") as _fh:
            merged_state_dict.update(_safe_load_bytes(_fh.read()))
        continue
    is_mps = load_config.device_map is not None and any(
        (d.type if isinstance(d, torch.device) else d) == "mps"
        for d in load_config.device_map.values()
    )
    # Use pread on MPS (mmap incompatible) and Windows (mmap reserves
    # copy-on-write commit charge for the entire file, exhausting memory
    # for large multi-shard checkpoints).
    if is_mps:
        backend, device = "pread", "mps"
    elif sys.platform == "win32":
        backend, device = "pread", "cpu"
    else:
        backend, device = "mmap", "cpu"
    file_pointer = safe_open(file, framework="pt", device=device, backend=backend)
    all_pointer.add(file_pointer)
    for k in file_pointer.keys():
        merged_state_dict[k] = file_pointer.get_slice(k)   # don't materialize yet
```

**这四行是理解「大模型加载不爆内存」的关键**：

1. `safe_open(..., backend="mmap")` —— **mmap 打开文件，不读内容**。
   操作系统按页惰性加载，只有真正访问的页才进内存。
2. `file_pointer.get_slice(k)` —— 拿到的是一个 **`PySafeSlice` 惰性句柄**，
   不是 `torch.Tensor`。注释直接写了 `# don't materialize yet`。
3. 所以循环结束时 `merged_state_dict` 里装的是**一堆句柄**，内存占用 ≈ 0。
4. 句柄的 `__exit__` 统一在 `:4540-4542` 关闭（`for k in all_pointer: k.__exit__(None, None, None)`），
   即使中途异常也会走 `finally` 语义（因为 `convert_and_load_state_dict_in_model` 抛错时
   这行不会执行——**这是为什么异常路径下文件也会被 GC 回收**）。

**三个 backend 的取舍**（代码注释 + `_is_on_hf_mount` 一起看）：

| backend | 何时用 | 原因 |
|---------|--------|------|
| `mmap`（默认） | Linux / macOS，CPU | 零拷贝、按需分页，最快 |
| `pread` | MPS 设备、Windows | MPS 不支持 mmap；Windows 的 mmap 会为整个文件预留写时复制提交量 |
| 全读进内存 | `disable_mmap=True` 或文件在 **hf-mount FUSE** 上 | `_is_on_hf_mount`（`modeling_utils.py:293`）的 docstring 写了原因：<br>`hf-mount's mmap + readahead interaction deadlocks under parallel page-faults` |

`_is_on_hf_mount` （`modeling_utils.py:293`）的做法是读 `/proc/mounts`，找出路径所在的挂载点，
判断设备名是不是 `"hf-mount"`：

```python
# modeling_utils.py:303-317（节选）
real = os.path.realpath(os.fspath(path))
with open("/proc/mounts", encoding="utf-8") as fh:
    entries = sorted(
        ((p[0], p[1]) for p in (l.split() for l in fh) if len(p) >= 2),
        key=lambda e: len(e[1]), reverse=True,      # ← 按挂载点长度降序，保证最长前缀匹配
    )
for dev, mp in entries:
    if real == mp or real.startswith(mp.rstrip("/") + "/"):
        return dev == "hf-mount"
```

### 4.2 `convert_and_load_state_dict_in_model`：主循环

这是整个权重加载的心脏，签名在 `core_model_loading.py:1511`。
它的 docstring（`:1517-1602`）本身就是一份极好的设计文档——**读懂 `param_name_to_load` 的形态，
就读懂了整个加载策略**（`core_model_loading.py:1521-1535`，原文节选）：

```python
# core_model_loading.py:1521-1535（docstring 原文）
The `param_name_to_load` will look like this:
{
    "model.layers.0.attention.q.weight": # Notice here there is only the first key of the target keys
        WeightConverter(
            source_patterns=["qkv"],
            target_patterns=["q", "k","v"],
            operations=[Chunk(dim=0, chunks=3)]),
            collected_tensors={"qkv": [Future]},          # ← 只收句柄，不实体化
            layer_targets={                               # ← 一个 source 对应三个 target
                "model.layers.0.attention.q.weight": {"model.layers.0.attention.qkv.weight"},
                "model.layers.0.attention.k.weight": {"model.layers.0.attention.qkv.weight"},
                "model.layers.0.attention.v.weight": {"model.layers.0.attention.qkv.weight"},
            }
        ),
    ...
}
```

主循环本身（`core_model_loading.py:1647-1748`）分四步，每一步都有明确职责：

```python
# core_model_loading.py:1647
state_dict = sorted(state_dict.items(), key=lambda kv: dot_natural_key(kv[0]))
for original_key, tensor in state_dict:
    # ① 改名：把 checkpoint key 映射成模型参数名
    renamed_key, source_pattern = rename_source_key(
        original_key, renamings, converters, base_model_prefix, meta_model_state_dict
    )
    if renamed_key not in meta_model_state_dict and original_key in meta_model_state_dict:
        # 改名失败但原名本身就对 → 只补/去 base_model_prefix
        renamed_key, source_pattern = rename_source_key(
            original_key, [], [], base_model_prefix=base_model_prefix, meta_state_dict=meta_model_state_dict
        )

    # ② 收进容器
    if renamed_key in meta_model_state_dict:
        empty_param = meta_model_state_dict.get(renamed_key)
        if source_pattern is not None:
            new_converter = deepcopy(pattern_to_converter[source_pattern])
            mapping = param_name_to_load.setdefault(renamed_key, new_converter)
        else:
            mapping = param_name_to_load.setdefault(renamed_key, WeightRenaming(original_key, renamed_key))
            source_pattern = original_key
        ...
        # ③ dtype 决策（dtype_plan / 量化 / 空参数 dtype）
        ...
        # ④ 设备与分片 → 异步实体化
        future_or_tensor = spawn_materialize(
            thread_pool, tensor, param_device, _dtype, sharding_op=sharding_op, tensor_idx=tensor_idx,
        )
        mapping.add_tensor(renamed_key, original_key, source_pattern, future_or_tensor)
    elif source_pattern is not None:
        # 一源多目标但只匹配上一个 → 其余目标记 unexpected
        mapping = pattern_to_converter[source_pattern]
        for k in mapping.target_patterns:
            _add_unmatched_checkpoint_key(renamed_key.replace(mapping.target_patterns[0], k), model, loading_info)
    else:
        _add_unmatched_checkpoint_key(renamed_key, model, loading_info)     # ← 记 unexpected
```

注意 `dot_natural_key`（`core_model_loading.py:1272`）这个排序键。它的 docstring：

> Sort key for state-dict names: split on `"."` and sort digits numerically and strings alphabetically.
> It emits a tuple at each point to sort ints first and strings second to avoid int-string comparison failures.

**为什么要自定义排序**：`model.layers.10.x` 必须排在 `model.layers.9.x` **之后**。
纯字符串排序会把 `10` 排在 `9` 前面。这个顺序不是审美问题——
`GroupWeightRename` 的 docstring（`:1043`）明确依赖它：

> NOTE: order `source_patterns` so that the one that is absent in an already-converted checkpoint
> comes first. The state dict is iterated in sorted key order, so the guard pattern must be
> lexicographically smaller than the dependent patterns.

### 4.3 主循环的 `param_name_to_load` → `mapping.convert` → `set_param_for_module`

收完所有 key 之后是第二个循环（`core_model_loading.py:1752-1781`）：

```python
# core_model_loading.py:1752-1779
try:
    for first_param_name, mapping in tqdm(param_name_to_load.items(), desc="Loading weights"):
        try:
            realized_value = mapping.convert(
                first_param_name, model=model, config=model.config,
                hf_quantizer=hf_quantizer, loading_info=loading_info,
            )
            for target_name, param in realized_value.items():
                param = param[0] if isinstance(param, list) else param
                param_device = get_device(device_map, target_name)
                # Offloading support
                if param_device == "disk" and (target_name not in model_buffers or offload_buffers):
                    disk_offload_index = offload_and_maybe_resave_param(
                        target_name, param, loading_info, disk_offload_folder, disk_offload_index, mapping
                    )
                else:
                    set_param_for_module(model, target_name, param, loading_info, hf_quantizer)
            del realized_value       # ← 及时释放，避免峰值内存翻倍
        except SkipParameters:
            continue
finally:
    if thread_pool is not None:
        thread_pool.shutdown(wait=False, cancel_futures=True)
```

**两级 `try` 的分工**：

- **内层** `except SkipParameters` —— 单个参数的转换失败（比如 `Chunk` 切不动）会被
  `log_conversion_errors`（`core_model_loading.py:1294`）捕获，错误记进
  `loading_info.conversion_errors`，然后抛 `SkipParameters` 哨兵异常（`:1405`）跳过一个参数。
  **一个参数转换失败不会毁掉整个加载**，这一点和 `error_msgs` 的 `RuntimeError` 完全不同。
- **外层** `finally`（`:1787-1789`）—— 无论成功、失败还是 Ctrl-C，线程池一定会关，且 `cancel_futures=True`
  避免退出时还在白读文件。

`set_param_for_module`（`core_model_loading.py:1343`）是「写进模型」的最后一步：

```python
# core_model_loading.py:1343-1382（节选）
def set_param_for_module(model, target_name, param_value, loading_info, hf_quantizer):
    module_path, _, param_name = target_name.rpartition(".")
    module_obj = model.get_submodule(module_path) if module_path else model
    ...
    ref = getattr(module_obj, param_name)
    if ref is None:
        loading_info.unexpected_keys.add(target_name)
    else:
        if not isinstance(param_value, torch.nn.Parameter) and not is_dtensor(ref):
            if param_name not in module_obj._buffers:
                param_value = torch.nn.Parameter(param_value, requires_grad=param_value.is_floating_point())

        # Remove from missing keys (it's either mismatched, or all good)
        loading_info.missing_keys.discard(target_name)

        # For DTensor parameters, compare against the local shard loaded on this rank.
        expected_shape = ref._local_tensor.shape if is_dtensor(ref) else ref.shape

        if ref is not None and param_value.shape != expected_shape and hf_quantizer is None:
            loading_info.mismatched_keys.add((target_name, param_value.shape, expected_shape))
        else:
            if is_dtensor(ref):
                ...
            # super important otherwise _init_weight will re-init the param
            param_value._is_hf_initialized = True
            setattr(module_obj, param_name, param_value)
```

**三行代码定义了三类 key 的归属**，这是第六节的基础：

| 条件 | 结果 |
|------|------|
| `ref is None`（模型里没这个属性） | → `unexpected_keys` |
| shape 不等 | → `mismatched_keys`（**注意：仍然被 `discard` 出 missing**） |
| 形状匹配 | → `setattr`，并打上 **`_is_hf_initialized = True`** |

**`_is_hf_initialized` 这个标记是整个机制里的「已读回执」**。
注释写得很直白：`# super important otherwise _init_weight will re-init the param`。
第五节会看到它怎么被消费。

### 4.4 三层改名机制：`WeightRenaming` / `WeightConverter` / `WeightTransform`

`conversion_mapping.py` 定义的是**声明式**映射表，`core_model_loading.py` 定义的是**执行引擎**。
三个类的关系：

```
WeightTransform                    core_model_loading.py:769   ← 基类：模式编译 + 反向变换
├── WeightRenaming                 core_model_loading.py:992   ← 只改名，不做张量运算
│   ├── GroupWeightRename          core_model_loading.py:1032  ← N:N 成组改名，带 guard 门控
│   └── PrefixChange               core_model_loading.py:1098  ← 专门用于增删 `base_model_prefix`
└── WeightConverter                core_model_loading.py:1152  ← 改名 + 张量运算（Chunk/Concat/...）
```

**`WeightTransform.__init__`（`core_model_loading.py:786`）做的事只有一件：把字符串模式编译成正则**：

```python
# core_model_loading.py:845-851
# Construct the regex we will use to rename keys from the sources to the targets
branches = []
for i, source_pattern in enumerate(self.source_patterns):
    group_name = f"g{i}"
    pattern = source_pattern.replace("*.", r".*\.")
    branches.append(f"(?P<{group_name}>{pattern})")
self.compiled_sources = re.compile("|".join(branches))
```

三个设计细节：

1. `*.` 被替换成 `.*\.` —— **`*` 是「层级通配符」而不是 glob 星号**。
   `mlp.experts.*.gate_proj.weight` 能匹配 `mlp.experts.7.gate_proj.weight`。
2. 每个 source pattern 是一个**命名捕获组** `g0` / `g1` / …，
   这样匹配后能反查「是第几个 source pattern 命中的」（`rename_source_key`，`:917-919`）。
3. `re.compile("|".join(branches))` 是**单次搜索匹配所有分支**，
   而不是逐个 pattern 试——这是热路径上的性能优化。

`WeightTransform.rename_source_key`（`:902-932`）的核心是「一源多目标时，先统一替换成**第一个** target」：

```python
# core_model_loading.py:913-932（节选）
def rename_source_key(self, source_key: str) -> tuple[str, str | None]:
    matched = self._scoped_match(source_key)
    if matched is None:
        return source_key, None
    prefix_dot, key_to_match, match_object = matched
    self._was_used = True                       # ← 打「我用过」标记，save_pretrained 时要用
    matching_group_name = next(name for name, val in match_object.groupdict().items() if val is not None)
    source_pattern_that_matched = self.source_patterns[int(matching_group_name[1:])]
    replacement = self.target_patterns[0]       # ← 总是第一个 target
    ...
    renamed_key = key_to_match.replace(match_object.group(0), replacement, 1)
    return renamed_key, source_pattern_that_matched
```

模块级的 `rename_source_key` 自由函数（`:1412`）负责**编排顺序**：

```python
# core_model_loading.py:1448-1482（节选）
renamed_key = source_key
converter_source_pattern = None
# 1. 正向：先跑所有 WeightRenaming（可链式），再跑 WeightConverter（最多命中一个）
first_iterable = weight_renamings if not reverse else weight_converters
for transform in first_iterable:
    renamed_key, source_pattern = transform.rename_source_key(renamed_key)
    if reverse and source_pattern is not None:
        converter_source_pattern = source_pattern
        break
...
# 3. 补/去 base_model_prefix（只在加载时）
if base_model_prefix is not None and meta_state_dict is not None:
    if (renamed_key.startswith(base_model_prefix)
            and meta_state_dict.get(re.sub(f"^{base_model_prefix}.", "", renamed_key, count=1)) is not None):
        renamed_key = re.sub(f"^{base_model_prefix}.", "", renamed_key, count=1)
    elif meta_state_dict.get(f"{base_model_prefix}.{renamed_key}") is not None:
        renamed_key = f"{base_model_prefix}.{renamed_key}"
```

**顺序规则一句话**：正向（加载）是 `Renaming → Converter`，反向（保存）是 `Converter → Renaming`。
docstring（`:1427-1432`）解释了为什么不需要 `Converter → Renaming` 这个方向：

> Note that we proceed in this way because there is no need for a Converter-then-Rename order
> because Converters act only on specific leaf patterns, so no subsequent Renamings should ever target their output.

**第 3 步的 `base_model_prefix` 逻辑值得单独记**（`:1472-1480`）：它让
`LlamaForCausalLM` 能直接吃 `LlamaModel` 的权重（反之亦然）——
模型里的参数叫 `model.layers.0...`，checkpoint 里可能叫 `layers.0...`，
这一步会**双向**探测补/去前缀，靠的是拿 `meta_state_dict` 做存在性检查（而不是猜）。

关于 `_was_used`（`:981`）：它的 docstring 说明了为什么需要它——

> This is needed as some weight renaming transforms are not bijective, i.e. if we drop/add full parts
> of a name with PrefixChange, we lose some information that we cannot get back if we don't know if
> the Transform was used before already

最后 `convert_and_load_state_dict_in_model` 把它挂到模型上（`core_model_loading.py:1792-1793`）：

```python
# core_model_loading.py:1792
model_specific_conversions = [conversion for conversion in weight_mapping if conversion.was_used()]
model._weight_conversions = model_specific_conversions
```

**这是 `save_pretrained` 时 `revert_weight_conversion`（`core_model_loading.py:1798`）的依据**：
加载时用了哪些变换，保存时就逆着变换回去。
`_was_used` 就是「这个变换到底生效没有」的记录。

### 4.5 `WeightConverter` 的运算：`ConversionOps` 家族

`ConversionOps`（`core_model_loading.py:81`）是一个 ABC，实现全在这一个文件里。
常用的几个：

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

真实例子（`conversion_mapping.py:198-216`，DeepSeek 系的 MoE）：

```python
# conversion_mapping.py:198-202
WeightConverter(
    source_patterns="shared_w13_weight",
    target_patterns=["gate_proj", "up_proj"],
    operations=[Interleave(dim=1), Chunk(dim=1)],
),
# conversion_mapping.py:204-208
WeightConverter(
    source_patterns="mlp.experts.w13_weight",
    target_patterns=["mlp.experts.gate_up_proj"],
    operations=[Interleave(dim=1)],
),
```

对照本仓库里真实的模型结构（`models/mixtral/modeling_mixtral.py:57-67`）：

```python
# models/mixtral/modeling_mixtral.py:57
class MixtralExperts(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
```

**「模型里是一个 3D 参数、checkpoint 里是 N 个独立 expert」这个落差，就是 `MergeModulelist` +
`Concatenate` 存在的全部理由**。这是理解 `conversion_mapping.py` 的最好入口。

### 4.6 `weight_mapping` 是怎么被挑出来的

`get_model_conversion_mapping`（`conversion_mapping.py:1902`）做的事，一句话：
**遍历模型的所有子模块，按「类名优先、`model_type` 兜底」查表，命中的挂上去**。

```python
# conversion_mapping.py:1902-1930（docstring 原文节选）
Each `PreTrainedModel` sub-module is looked up by class name then `model_type`.
Root transforms are applied globally; sub-module transforms have their `scope_prefix` set
so they only match keys under that prefix. After any sub-module is processed, both its class name
and `model_type` are marked seen to prevent `XForY` / `XModel` pairs from applying the same mapping
twice via different lookup paths.
```

关键的三层去重（`conversion_mapping.py:1930-1996`）：

```python
# conversion_mapping.py:1936-1961（节选）
for module_name, submodule in model.named_modules():
    if not isinstance(submodule, PreTrainedModel):
        continue
    class_name = type(submodule).__name__
    model_type = submodule.config.model_type
    ...
    # Skip if an ancestor already claimed this class (its unscoped transforms already cover this subtree).
    if any(seen == "" or module_name.startswith(seen + ".") for seen in seen_identifiers[class_name]):
        continue

    # Class name takes priority — a class-specific mapping bypasses the model_type deduplication check
    conversions = get_checkpoint_conversion_mapping(class_name)
    found_via_class = conversions is not None
    ...
    is_root_model = module_name == ""
    if not is_root_model:
        scope_prefix = module_name.removeprefix(model.base_model_prefix)
        scope_prefix = scope_prefix.removeprefix(".")
        for transform in conversions:
            transform.scope_prefix = scope_prefix
            transform.base_model_prefix = model.base_model_prefix
```

**`scope_prefix` 是多模态模型的关键**：`LlavaForConditionalGeneration` 里嵌套了一个
`LlavaModel`，如果两层的 `model_type` 都是 `llava`，不加 scope 就会把映射重复应用两次。
`scope_prefix` 让子模块的映射**只匹配该前缀下的 key**，
`_scoped_match`（`core_model_loading.py:871`）负责剥离前缀后再匹配。

**最后一行很关键**（`conversion_mapping.py:2003`）：

```python
# conversion_mapping.py:2003-2004
if hf_quantizer is not None:
    weight_conversions = hf_quantizer.update_weight_conversions(weight_conversions)
```

量化器有机会**插入自己的 op**。上面的注释给了具体例子：
FP8 反量化会**前插** `Fp8Dequantize`，保证 per-block scale 在任何 expert-merge 之前生效。

**规模**（`grep -c` 实测）：`conversion_mapping.py` 里
`WeightRenaming(` 出现 **621** 次、`WeightConverter(` **55** 次、`PrefixChange(` **11** 次、
`GroupWeightRename(` **2** 次，覆盖 **84** 个 model_type / 类名条目。

---

## 五、`_init_weights`：没被 checkpoint 覆盖的参数

### 5.1 基类实现与三条兜底链

`PreTrainedModel._init_weights`（`modeling_utils.py:2374`）刻意写得**又长又通用**：

```python
# modeling_utils.py:2374-2381
def _init_weights(self, module):
    """
    Initialize the weights. This is quite general on purpose, in the spirit of what we usually do. For more complex
    initialization scheme, it should be overridden by the derived `PreTrainedModel` class. In case a model adds an explicit
    `nn.Parameter`, this method should also be overridden in order to initialize it correctly.
    """
    if hasattr(self.config, "initializer_range"):
        std = self.config.initializer_range or 0.02
    elif hasattr(self.config, "init_std"):
        std = self.config.init_std
    elif hasattr(self.config, "initializer_factor"):
        std = self.config.initializer_factor
    else:
        # 0.02 is the standard default value across the library
        std = getattr(self.config.get_text_config(), "initializer_range", 0.02)
```

**std 的取值有四级兜底**，这是历史包袱的化石：
`initializer_range`（现代模型）→ `init_std`（老 BERT 系）→ `initializer_factor`（T5）→ 文本子 config → `0.02`。

分支结构（同函数 `:2390-2435`）：

| 模块类型 | 判断方式 | 初始化 |
|---------|---------|--------|
| `nn.Linear` / `nn.Conv*` | `isinstance` | `normal_(mean=0, std=std)`；bias `zeros_` |
| `nn.LSTM` | `isinstance` | weight `xavier_uniform_`；bias `constant_(0)` |
| `nn.Embedding` | `isinstance` | `normal_`；`padding_idx` 那行**额外** `zeros_` |
| `nn.MultiheadAttention` | `isinstance` | 交给 torch 自己的 `_reset_parameters()` |
| **任意 Norm** | **类名字符串匹配** `"LayerNorm" in ... or "RMSNorm" in ...` | weight `ones_`；bias `zeros_` |
| **任意 RoPE** | **类名字符串匹配** `"RotaryEmbedding" in ... and hasattr(module, "original_inv_freq")` | 重算 `inv_freq` |

**那两个「类名字符串匹配」是这里最反直觉的设计**，代码里也给了理由（`:2409-2410`）：

```python
# modeling_utils.py:2409
# We cannot use `isinstance` on the RMSNorms or LayerNorms, as they usually are custom modules which change names
# between modelings (because they are prefixed with the model name)
```

> **这条注释直接解释了为什么 `_init_weights` 能覆盖所有模型**：
> 500 个模型有 500 个 `XxxRMSNorm`，但它们都是「名字里带 RMSNorm 的类」。
> 用字符串匹配就一次性全覆盖了。

RoPE 那一支（`:2427-2435`）值得单独看，因为它**不是简单的 fill，而是重算**：

```python
# modeling_utils.py:2427-2435
elif "RotaryEmbedding" in module.__class__.__name__ and hasattr(module, "original_inv_freq"):
    rope_fn = (
        ROPE_INIT_FUNCTIONS[module.rope_type]
        if module.rope_type != "default"
        else module.compute_default_rope_parameters
    )
    buffer_value, _ = rope_fn(module.config)
    init.copy_(module.inv_freq, buffer_value)
    init.copy_(module.original_inv_freq, buffer_value)
```

对照 `models/llama/modeling_llama.py:82-89`，这些属性确实存在，
而且 `original_inv_freq` 是 **`persistent=False` 的 buffer**：

```python
# models/llama/modeling_llama.py:82-89（节选）
self.rope_type = self.config.rope_parameters["rope_type"]
...
if self.rope_type != "default":
    rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
...
self.original_inv_freq = nn.Buffer(inv_freq.clone(), persistent=False)
```

**`persistent=False` + `_init_weights` 里重算，是一个精妙的配合**：
`persistent=False` 意味着这个 buffer **不进 state_dict**，所以 checkpoint 里根本没有它；
它必须在建模时被算出来。`_init_weights` 里的这一支就是「算出来」的正式场所。

### 5.2 三个模型的 `_init_weights` 对比

**实测结论（可以用 `grep -c "_init_weights"` 复现）**：

```
src/transformers/models/llama/modeling_llama.py:        0 处
src/transformers/models/mistral/modeling_mistral.py:    0 处
src/transformers/models/qwen3/modeling_qwen3.py:        0 处
```

**这三个现代 decoder-only 模型根本没有覆写 `_init_weights`**——它们**完全依赖基类实现**。
原因：Llama 的组件（`LlamaRMSNorm` / `nn.Linear` / `nn.Embedding` / `LlamaRotaryEmbedding`）
**全部命中了基类的字符串匹配规则**，一个都不需要特殊处理。

| 模型 | 是否覆写 | 覆写内容 | 为什么 |
|------|---------|---------|--------|
| `LlamaForCausalLM` | **否** | — | 组件全被基类覆盖 |
| `MistralForCausalLM` | **否** | — | 同上 |
| `Qwen3ForCausalLM` | **否** | — | 同上 |
| `BertPreTrainedModel` | 是（`bert/modeling_bert.py:547`） | 加 `BertLMPredictionHead` 的 bias 置零；`position_ids` / `token_type_ids` 填值 | 有非持久 buffer 需要填 |
| `T5PreTrainedModel` | 是（`t5/modeling_t5.py:563`） | **逐模块手写 std**：`factor * (d_model ** -0.5)`、`factor * (d_ff ** -0.5)` | T5 的初始化是架构定义的一部分，不能用统一 std |
| `MixtralForCausalLM` | 是（`mixtral/modeling_mixtral.py:399`） | 加 `MixtralExperts` 和 `MixtralTopKRouter` | 融合专家参数是 `nn.Parameter`，基类不认识 |
| `GlmMoeDsaForCausalLM` | 是（`glm_moe_dsa/modeling_glm_moe_dsa.py:670`） | 加 `TopkRouter` / `Experts` | 同上 |

四个覆写的写法逐一看：

```python
# models/bert/modeling_bert.py:547-554
def _init_weights(self, module):
    """Initialize the weights"""
    super()._init_weights(module)                                  # ← 先调基类！
    if isinstance(module, BertLMPredictionHead):
        init.zeros_(module.bias)
    elif isinstance(module, BertEmbeddings):
        init.copy_(module.position_ids, torch.arange(module.position_ids.shape[-1]).expand((1, -1)))
        init.zeros_(module.token_type_ids)
```

```python
# models/t5/modeling_t5.py:563-570
def _init_weights(self, module):
    """Initialize the weights"""
    super()._init_weights(module)                                  # ← 先调基类！
    factor = self.config.initializer_factor  # Used for testing weights initialization
    if isinstance(module, T5LayerNorm):
        init.constant_(module.weight, factor * 1.0)                # ← 不是 1.0，是 factor * 1.0
    elif isinstance(module, (T5Model, T5ForConditionalGeneration, T5EncoderModel, T5ForQuestionAnswering)):
        init.normal_(module.shared.weight, mean=0.0, std=factor * 1.0)
        ...
```

```python
# models/mixtral/modeling_mixtral.py:399-406          # glm_moe_dsa/modeling_glm_moe_dsa.py:670-677 几乎逐字相同
def _init_weights(self, module):
    super()._init_weights(module)
    std = self.config.initializer_range
    if isinstance(module, MixtralExperts):                       # GlmMoeDsaExperts
        init.normal_(module.gate_up_proj, mean=0.0, std=std)
        init.normal_(module.down_proj, mean=0.0, std=std)
    elif isinstance(module, MixtralTopKRouter):                  # GlmMoeDsaTopkRouter
        init.normal_(module.weight, mean=0.0, std=std)
        # GLM 版多一行：init.zeros_(module.e_score_correction_bias)  ← 偏置必须置零
```

**四条都能提炼出的共同模式**：
1. **`super()._init_weights(module)` 永远是第一行**（除了 T5 的 `T5LayerNorm` 分支是覆盖而非追加）——
   不做这一步，norm 和 embedding 就没人管了。
2. **分支条件一定是「基类不认识的模块」**：自定义 `nn.Parameter` 容器（`Experts`）、
   需要特殊方向的 router、带 `bias` 的预测头。
3. `Mixtral` 与 `GlmMoeDsa` 几乎逐字相同——这正是 2A5 讲的「跨模型重复」，
   只不过 `_init_weights` 没有被 modular 机制抽出来共享。

**对阅读源码的实操价值**：当你在读一个模型想知道「某个新参数怎么初始化」时，
**不要在这个模型文件里找 `_init_weights` 就完事**——
先确认它在不在，不在就去 `modeling_utils.py:2374`。

### 5.3 `initialize_weights` 与 `smart_apply`

`_initialize_weights`（`modeling_utils.py:2437`）是「跳过已初始化」的守门人：

```python
# modeling_utils.py:2437-2458（节选）
def _initialize_weights(self, module, is_custom_code: bool = False):
    """
    Initialize the weights if they are not already initialized.
    """
    if getattr(module, "_is_hf_initialized", False):
        return
    ...
    self._init_weights(module)
    module._is_hf_initialized = True
```

**这就是第四节 `set_param_for_module` 里那个 `_is_hf_initialized = True`（`core_model_loading.py:1382`）的消费点**。
全程的因果链是：

```
set_param_for_module 打标  →  _initialize_weights 见到标记直接 return  →  已加载的权重不会被覆盖
```

`initialize_weights`（`modeling_utils.py:2464`）则是**支持复合模型**的递归分发器：

```python
# modeling_utils.py:2471-2486（节选）
if not hasattr(torch.nn.Module, "smart_apply"):
    def smart_apply(module: nn.Module, fn: Callable[[nn.Module, bool], None], is_custom_code: bool):
        for child in module.children():
            # We found a sub-model: recursively dispatch its own init function now!
            if isinstance(child, PreTrainedModel):
                smart_apply(child, child._initialize_weights, is_custom_code)
            else:
                smart_apply(child, fn, is_custom_code)
        fn(module, is_custom_code)
        return module
    setattr(torch.nn.Module, "smart_apply", smart_apply)

# Let the magic happen with this simple call
smart_apply_fn = getattr(self, "smart_apply")
smart_apply_fn(self._initialize_weights, self.is_custom_code())
```

**它解决的问题（docstring 原文）**：

> This function dynamically dispatches the correct `init_weights` function to the modules as we advance in the
> module graph along the recursion. It can handle an arbitrary number of sub-models. Without it, every composite
> model would have to recurse a second time on all sub-models explicitly in the outer-most `_init_weights`, which
> is extremely error prone and inefficient.

翻译：一个 `LlavaForConditionalGeneration` 里嵌着 `LlavaModel` 和 `CLIPVisionModel`，
三者各有自己的 `_init_weights`。`smart_apply` 在**每一个 `PreTrainedModel` 边界上切换函数**，
所以每个子模型用的是自己的初始化逻辑，而不是最外层那个。

`@init.guard_torch_init_functions()` 装饰器（`modeling_utils.py:2463`）与
`transformers.initialization`（`initialization.py`，340 行）配合，
负责在 `init.*` 调用时**自动打 `_is_hf_initialized` 标记**——这是为什么模型代码里
用 `init.normal_` 而不是 `torch.nn.init.normal_` 的原因。

---

## 六、missing / unexpected / mismatched

### 6.1 三者的定义（来自 `LoadStateDictInfo`）

定义在 `utils/loading_report.py:156`，docstring 本身就是权威表述：

```python
# utils/loading_report.py:156-186（docstring 原文）
@dataclass
class LoadStateDictInfo:
    """
    Mutable container for state-dict loading results and diagnostics. Each entry in this structure is mutable,
    and will usually be mutated in-place during the loading pipeline.

    Attributes:
        missing_keys (`set[str]`):
            Keys that are missing from the loaded checkpoints but expected in the model's architecture.
        unexpected_keys (`set[str]`):
            Keys that are found in the checkpoints, but not expected in the model's architecture.
        skipped_pp_keys (`set[str]`):
            Checkpoint keys owned by another pipeline-parallel rank and intentionally not loaded here.
        mismatched_keys (`set[tuple[str, tuple[int], tuple[int]]]`):
            Keys that are found in the checkpoints and are expected in the model's architecture, but with a different shape.
        error_msgs ( `list[str]`):
            Some potential error messages.
        conversion_errors (`dict[str, str]`):
            Errors happening during the on-the-fly weight conversion process.
    """

    missing_keys: set[str]
    unexpected_keys: set[str]
    mismatched_keys: set[tuple[str, tuple[int], tuple[int]]]
    error_msgs: list[str]
    conversion_errors: dict[str, str]
    skipped_pp_keys: set[str]
```

翻译成一张表：

| 类别 | 含义 | 一句话直觉 |
|------|------|-----------|
| **missing** | 模型要、checkpoint 没有 | **模型多了东西**（或 checkpoint 少了） |
| **unexpected** | checkpoint 有、模型不要 | **checkpoint 多了东西**（或模型少了） |
| **mismatched** | 两边都有，但 shape 不同 | **对上了名字，对不上形状** |
| `error_msgs` | 加载时的硬错误字符串 | **会直接 `RuntimeError`** |
| `conversion_errors` | 权重变换 op 内部抛的异常 | **会打报告，但单个参数被跳过** |
| `skipped_pp_keys` | 属于别的流水线 rank | **正常，不算 unexpected** |

`mismatched_keys` 是三元组的 set：`(参数名, checkpoint 的形状, 模型的形状)`。

### 6.2 三者分别是怎么被记录的

| 类别 | 记录点 | 代码 |
|------|--------|------|
| missing（初始值） | `core_model_loading.py:1615` | `missing_keys=set(meta_model_state_dict.keys())` ← **从「模型的全部参数」开始，加载成功一个就 discard 一个** |
| 成功加载 | `core_model_loading.py:1367` | `loading_info.missing_keys.discard(target_name)` |
| mismatched | `core_model_loading.py:1373` | `loading_info.mismatched_keys.add((target_name, param_value.shape, expected_shape))` |
| unexpected（key 完全对不上） | `core_model_loading.py:1744-1748` | `_add_unmatched_checkpoint_key(renamed_key, model, loading_info)` |
| unexpected（属性不存在） | `core_model_loading.py:1360` | `if ref is None: loading_info.unexpected_keys.add(target_name)` |

**「missing 从全量开始倒着减」这个设计很重要**：它意味着 `missing_keys` 的**初始值是模型的全部参数**
（`core_model_loading.py:1614-1621`）：

```python
# core_model_loading.py:1614-1621
# We start from all missing keys, and we will remove/add them from the proper containers as loading advances
loading_info = LoadStateDictInfo(
    missing_keys=set(meta_model_state_dict.keys()),
    unexpected_keys=set(),
    mismatched_keys=set(),
    conversion_errors={},
    error_msgs=[],
    skipped_pp_keys=set(),
)
```

**所以「missing」在语义上是「走完全程也没被 discard 掉的那些」**——
这也解释了为什么一个参数如果转换失败（`conversion_errors`），它同时也会是 missing。
两者是「症状」和「原因」的关系。

`skipped_pp_keys` 的引入很新（`:1485`），docstring 说明了动机：

```python
# core_model_loading.py:1489-1495
"""Classify a checkpoint key that does not match the current model state dict.

During pipeline-parallel loading, each stage receives keys for the entire model even though its local state dict
contains only that stage's parameters. A key owned by another stage is therefore recorded as intentionally skipped
instead of unexpected. Without the key, it would be considered unexpected.
"""
```

### 6.3 报告：`log_state_dict_report` 与真实的报错文本

`_finalize_model_loading`（`modeling_utils.py:4550`）的 `finally` 块里**一定会调用**报告函数
（`:4577-4585`）：

```python
# modeling_utils.py:4577-4585
finally:
    log_state_dict_report(
        model=model,
        pretrained_model_name_or_path=load_config.pretrained_model_name_or_path,
        ignore_mismatched_sizes=load_config.ignore_mismatched_sizes,
        loading_info=loading_info,
        logger=logger,
    )
```

`log_state_dict_report`（`utils/loading_report.py:273`）分三段：**先抛硬错误，再打报告，最后按条件抛异常**：

```python
# utils/loading_report.py:289-317（节选）
# Re-raise errors early if needed
if loading_info.error_msgs:
    error_msg = "\n\t".join(loading_info.error_msgs)
    if "size mismatch" in error_msg:
        error_msg += (
            "\n\tYou may consider adding `ignore_mismatched_sizes=True` to `from_pretrained(...)` if appropriate."
        )
    raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")
...
prelude = f"{model.__class__.__name__} LOAD REPORT from: {pretrained_model_name_or_path}\n"
logger.warning(prelude + report)                          # ← 报告以 warning 打出
...
if loading_info.conversion_errors:
    raise RuntimeError(
        "We encountered some issues during automatic conversion of the weights. For details look at the `CONVERSION` entries of "
        "the above report!"
    )
if not ignore_mismatched_sizes and loading_info.mismatched_keys:
    raise RuntimeError(
        "You set `ignore_mismatched_sizes` to `False`, thus raising an error. For details look at the above report!"
    )
```

**这三段的关系要记牢**：

| 触发条件 | 行为 | 能否绕过 |
|---------|------|---------|
| `error_msgs` 非空 | **最高优先级 `RuntimeError`**，报告都不打 | 否 |
| 报告内容非空 | 用 `logger.warning` 打表 | — |
| `conversion_errors` 非空 | 报告之后 `RuntimeError` | 否（要修 conversion mapping） |
| `mismatched_keys` 非空 | 报告之后 `RuntimeError` | ✅ `ignore_mismatched_sizes=True` |
| 只有 `missing` / `unexpected` | **只打 warning，不抛异常** | 本来就是 warning |

**真实报告长什么样**。表由 `_make_table`（`utils/loading_report.py:101`）生成，
表头是 `Key | Status`（终端宽度 > 200 时额外加 `Details` 列），
行状态有四个字面量：`UNEXPECTED` / `MISSING` / `MISMATCH` / `CONVERSION`
（`utils/loading_report.py:204-235`）：

```python
# utils/loading_report.py:222-232
if self.mismatched_keys:
    tips += f"\n- {_style('MISMATCH:', 'yellow')}\t" + _style(
        "ckpt weights were loaded, but they did not match the original empty weight shapes.", "italic"
    )
    iterator = {a: (b, c) for a, b, c in self.mismatched_keys}
    for key, (shape_ckpt, shape_model) in update_key_name(iterator).items():
        status = _style("MISMATCH", "yellow")
        data = [
            key, status,
            f"Reinit due to size mismatch - ckpt: {str(shape_ckpt)} vs model:{str(shape_model)}",
        ]
        rows.append(data)
```

所以实际输出形态是：

```
LlamaForCausalLM LOAD REPORT from: /path/to/ckpt
Key                                    | Status     | Details
---------------------------------------+------------+------------------------------
lm_head.weight                         | UNEXPECTED |
model.layers.{0..31}.self_attn.rotary_emb.inv_freq | UNEXPECTED |
model.layers.5.mlp.gate.weight         | MISSING    |

Notes:
- UNEXPECTED:  can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING:     those params were newly initialized because missing from the checkpoint. ...
- MISMATCH:    ckpt weights were loaded, but they did not match the original empty weight shapes.
```

**表里的 key 是被合并过的**——`update_key_name`（`utils/loading_report.py:38`）把
`layers.0.x`、`layers.1.x`… 合并成 `layers.{0, 1}.x`，
所以 32 层的问题只占一行。`_DIGIT_RX`（同文件 `:25`）负责识别「点之间的整数」：

```python
# utils/loading_report.py:25-32
_DIGIT_RX = re.compile(r"(?<=\.)(\d+)(?=\.|$)")  # numbers between dots or at the end

def _pattern_of(key: str) -> str:
    """Replace every dot-delimited integer with '*' to get the structure."""
    return _DIGIT_RX.sub("*", key)
```

### 6.4 白名单过滤：`_adjust_missing_and_unexpected_keys`

很多 missing/unexpected 是**已知且无害**的，模型用两个类属性声明白名单：

| 类属性 | 含义 | 真实例子 |
|--------|------|---------|
| `_keys_to_ignore_on_load_missing` | 匹配的 missing 会被删掉 | — |
| `_keys_to_ignore_on_load_unexpected` | 匹配的 unexpected 会被删掉 | 见下 |

过滤逻辑在 `modeling_utils.py:4864`：

```python
# modeling_utils.py:4868-4896（节选）
# Old checkpoints may have keys for rotary_emb.inv_freq for each layer, however we moved this buffer to the main model
# (so the buffer name has changed). Remove them in such a case.
has_inv_freq_buffers = any(buffer.endswith("rotary_emb.inv_freq") for buffer, _ in self.named_buffers())
additional_unexpected_patterns = {r"rotary_emb\.inv_freq"} if has_inv_freq_buffers else set()
# Same idea for `position_ids`: used to be a persistent buffer, now `persistent=False` in most models.
has_position_ids_buffers = any(buffer.endswith("position_ids") for buffer, _ in self.named_buffers())
if has_position_ids_buffers:
    additional_unexpected_patterns.add(r"(^|\.)position_ids$")

missing_patterns = self._keys_to_ignore_on_load_missing or set()
unexpected_patterns = (self._keys_to_ignore_on_load_unexpected or set()) | additional_unexpected_patterns
...
# Clean-up unexpected keys
if ignore_unexpected_regex is not None:
    loading_info.unexpected_keys = {
        key for key in loading_info.unexpected_keys if ignore_unexpected_regex.search(key) is None
    }
```

**注意两个「自动白名单」是条件性的**：只有当模型**当前**确实有 `rotary_emb.inv_freq`
这种 buffer 时才加这条正则。这是「旧 checkpoint 有、新模型没有」这类问题的自动化处理。

`_keys_to_ignore_on_load_unexpected` 的真实用法（全库 **123** 个文件里有）：

```python
# models/t5/modeling_t5.py:1074-1075
class T5EncoderModel(T5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"decoder"]
```

```python
# models/glm_moe_dsa/modeling_glm_moe_dsa.py:666
_keys_to_ignore_on_load_unexpected = [r"model\.layers\.78.*"]
```

**两个例子的动机不同，值得对照**：

- **T5**：`T5EncoderModel` 只有 encoder，但 `T5ForConditionalGeneration` 的 checkpoint
  里必然有 `decoder.*`。用 `[r"decoder"]` 一次性忽略整个 decoder 子树。
  **这是典型的「用 base model 的类加载完整 checkpoint」。**
- **GLM**：`r"model\.layers\.78.*"` 忽略**第 78 层**。
  这是 MTP（multi-token prediction）的额外层——checkpoint 里有，但默认模型不建这一层。
  **这是典型的「checkpoint 比模型多一层」**。

### 6.5 排查顺序表

看到报告时按这个顺序查：

| 症状 | 最可能的原因 | 查什么 |
|------|-------------|--------|
| 大量 `UNEXPECTED` + 大量 `MISSING`，且名字成对相似 | **命名体系不同**（`to_q` vs `q_proj`、`Wq` vs `q_proj`） | `conversion_mapping.py` 里有没有这个 `model_type` 的条目 |
| 全部 `UNEXPECTED`（模型没有 missing） | 加载了**任务头不同**的 checkpoint（如用 base 权重加载 `ForCausalLM`） | 用对应的 auto 类；或看 `_keys_to_ignore_on_load_unexpected` |
| `model.layers.{0..N}.self_attn.rotary_emb.inv_freq` 为 `UNEXPECTED` | 旧 checkpoint 的 RoPE buffer | **无害**，`modeling_utils.py:4872` 已自动过滤 |
| `lm_head.weight` 为 `MISSING` 且 `tie_word_embeddings=True` | 权重被 tie 到 `embed_tokens` | **无害**，`tie_weights`（`:2606`）会处理 |
| 单个 key 的 `MISMATCH` + `Reinit due to size mismatch` | **shape 对不上**（改过 `num_labels` / `vocab_size` / `hidden_size`） | `ignore_mismatched_sizes=True` 会**重新随机初始化**该参数 |
| `CONVERSION` 行 | `WeightConverter` 的 op 抛异常 | 看 `Details` 里的 traceback；问题在 `ConversionOptions` 的参数（dim / chunks） |
| 全局 `RuntimeError: Error(s) in loading state_dict` | `error_msgs` 非空 | 报告都不会打，直接看错误文本 |

---

## 七、磁盘上不存在的模型：`trust_remote_code`

### 7.1 三态判定

`auto_factory.py:355-366` 判定的是三个布尔量的组合：

```python
# auto_factory.py:355-362
has_remote_code = hasattr(config, "auto_map") and cls.__name__ in config.auto_map
has_local_code = type(config) in cls._model_mapping
explicit_local_code = has_local_code and not _get_model_class(config, cls._model_mapping).__module__.startswith("transformers.")
upstream_repo = None
if has_remote_code:
    class_ref = config.auto_map[cls.__name__]
    if "--" in class_ref:
        upstream_repo = class_ref.split("--")[0]
```

`auto_map` 是 config.json 里的一段（由 `register_for_auto_class` 在 `modeling_utils.py:4612` 写入）：

```json
{
  "model_type": "my_awesome_model",
  "auto_map": {
    "AutoConfig": "configuration_my_awesome_model.MyAwesomeConfig",
    "AutoModelForCausalLM": "modeling_my_awesome_model.MyAwesomeForCausalLM"
  }
}
```

**`class_ref` 里可能带 `--`**：`"sgugger/my-bert-model--modeling.MyBertModel"`，
表示代码在**另一个 repo**里。这时 `upstream_repo` 会被提出来，
`resolve_trust_remote_code`（`dynamic_module_utils.py:712`）会用它构造报错信息。

### 7.2 `resolve_trust_remote_code`：交互式确认

```python
# dynamic_module_utils.py:755-793（节选）
if trust_remote_code is None:
    if has_local_code:
        trust_remote_code = False
    elif has_remote_code and TIME_OUT_REMOTE_CODE > 0:
        ...
        while trust_remote_code is None:
            answer = input(
                f"{error_message} You can inspect the repository content at https://hf.co/{model_name}.\n"
                f"You can avoid this prompt in future by passing the argument `trust_remote_code=True`.\n\n"
                f"Do you wish to run the custom code? [y/N] "
            )
            ...
    elif has_remote_code:
        # For the CI which puts the timeout at 0
        _raise_timeout_error(None, None)
...
if has_remote_code and not has_local_code and not trust_remote_code:
    raise ValueError(
        f"{error_message} You can inspect the repository content at https://hf.co/{model_name}.\n"
        f"Please pass the argument `trust_remote_code=True` to allow custom code to be run."
    )
```

**三条规则**：

| 条件 | 行为 |
|------|------|
| `has_local_code=True`（库内已有这个 model_type） | **静默 `False`**——不需要执行远端代码 |
| 有远端代码、无本地代码、未显式传参 | **交互式 `input()` 询问**（SIGALRM 超时保护） |
| 有远端代码、无本地代码、传了 `False` | **`ValueError`** |

**这就是为什么 `trust_remote_code` 不传时会卡住**——它在等你敲 `y`/`n`。

### 7.3 远端代码被下载到哪、怎么加载

`get_class_from_dynamic_module`（`dynamic_module_utils.py:516`）是入口：

```python
# dynamic_module_utils.py:606-615
if "--" in class_reference:
    repo_id, class_reference = class_reference.split("--")   # 代码在另一个 repo 里
else:
    repo_id = pretrained_model_name_or_path
module_file, class_name = class_reference.split(".")         # "modeling_foo.MyModel"

final_module = get_cached_module_file(
    repo_id, module_file + ".py", cache_dir=cache_dir, ..., revision=code_revision, ...
)
return get_class_in_module(class_name, final_module, force_reload=force_download)
```

`get_cached_module_file`（`:346`）做三件事：

1. **下载**：走和权重**完全一样的 `cached_file`**（`:417`）——这也是为什么 `trust_remote_code`
   会继承 `cache_dir` / `revision` / `local_files_only` 等所有 hub 参数。
2. **校验依赖**：`check_imports(resolved_module_file)`（`:442`）扫 import，缺包就报错。
3. **落到模块缓存**（`:451-453`）：

```python
# dynamic_module_utils.py:413, 451-453
submodule = os.path.sep.join(map(_sanitize_module_name, pretrained_model_name_or_path.split("/")))
...
# Now we move the module inside our cached dynamic modules.
full_submodule = TRANSFORMERS_DYNAMIC_MODULE_NAME + os.path.sep + submodule
create_dynamic_module(full_submodule)
submodule_path = Path(HF_MODULES_CACHE) / full_submodule
```

最终落点：`~/.cache/huggingface/modules/transformers_modules/<owner>/<repo>/modeling_foo.py`。
**本地路径**的模型额外套一层源码 hash（`:443-448`）——同一个目录改一次 `.py`，缓存目录就变一次，
避免「改了源码但加载到旧模块」：

```python
# dynamic_module_utils.py:443-448
if is_local:
    local_model_name = _sanitize_module_name(os.path.basename(os.path.normpath(pretrained_model_name_or_path)))
    local_source_files_hash = _compute_local_source_files_hash(pretrained_model_name_or_path, resolved_module_file)
    if local_model_name:
        submodule = os.path.sep.join([local_model_name, local_source_files_hash])
```

`get_class_in_module`（`:266`）负责把 `.py` 变成类，**用内容 hash 决定要不要重新 exec**：

```python
# dynamic_module_utils.py:288-307（节选）
module_file: Path = Path(HF_MODULES_CACHE) / module_path
with _HF_REMOTE_CODE_LOCK:                        # ← 进程级锁，避免并发 exec 同一模块
    ...
    # Hash the module file and all its relative imports to check if we need to reload it
    module_files: list[Path] = [module_file] + sorted(map(Path, get_relative_import_files(module_file)))
    module_hash: str = hashlib.sha256(b"".join(bytes(f) + f.read_bytes() for f in module_files)).hexdigest()
    ...
    if getattr(module, "__transformers_module_hash__", "") != module_hash:
        module_spec.loader.exec_module(module)    # ← 真正执行任意代码的时刻
        module.__transformers_module_hash__ = module_hash
    return getattr(module, class_name)
```

**`exec_module` 就是 `trust_remote_code` 里「trust」所指的那一行**。docstring 也警告了
（`dynamic_module_utils.py:534-535`）：`Calling this function will execute the code in the module file`。

**加载完之后，模型类就完全走正常的 `model.from_pretrained` 流程了**（`auto_factory.py:385-387`）：

```python
# auto_factory.py:382, 385-387
cls.register(config.__class__, model_class, exist_ok=True)          # ← 注册进 _model_mapping
model_class.register_for_auto_class(auto_class=cls)
...
return model_class.from_pretrained(
    pretrained_model_name_or_path, *model_args, config=config, **hub_kwargs, **kwargs
)
```

**所以第七节和前面六节是串联的，不是并列的**：`trust_remote_code` 只替换了
「模型类从哪来」这一步，之后的下载、建模、灌权重**一模一样**。
那句 `cls.register(...)` 也解释了「第一次加载远程模型后，`AutoConfig` 就能认它了」。

---

## 八、实操排查手册

本节把前面所有机制串成一个**可执行的检查顺序**。

### 8.1 遇到「weights not used in ...」/ unexpected keys

报错文案来自 `from_pretrained` 的 docstring（`modeling_utils.py:3908-3909`）：

> The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY,
> therefore those weights are discarded.

新版会渲染成 `LOAD REPORT` 里的 `UNEXPECTED` 行。**按这个顺序查**：

| 步 | 假设 | 怎么验证 / 怎么修 |
|----|------|------------------|
| 1 | **用 base 类加载了带任务头的 checkpoint**（`AutoModel.from_pretrained(一个 ForCausalLM 的 repo)`） | 换用对应的 Auto 类。官方承认的这类情况会写进白名单，如 `models/t5/modeling_t5.py:1075` `_keys_to_ignore_on_load_unexpected = [r"decoder"]` |
| 2 | **名字体系不同**（`to_q` vs `q_proj`） | `grep -n '"<model_type>"' src/transformers/conversion_mapping.py`；有映射但没生效 → 看是 class name 还是 `model_type` 命中（`get_model_conversion_mapping`，`conversion_mapping.py:1902`，**class name 优先**） |
| 3 | **嵌套子模型导致的 scope 问题** | 多模态模型里 `XxxForConditionalGeneration` 套 `XxxModel`，看 `scope_prefix` 有没有被正确设置（`conversion_mapping.py:1980-1984`） |
| 4 | **`base_model_prefix` 层面的差异**（BaseModel ↔ ForCausalLM） | `rename_source_key` 第 3 步（`core_model_loading.py:1472-1480`）会自动补/去前缀，**但前提是 key 的其余部分完全一致** |

### 8.2 遇到「size mismatch」

真实报错文本由 `log_state_dict_report`（`utils/loading_report.py:288-296`）拼出：

```python
# utils/loading_report.py:290-296
if loading_info.error_msgs:
    error_msg = "\n\t".join(loading_info.error_msgs)
    if "size mismatch" in error_msg:
        error_msg += (
            "\n\tYou may consider adding `ignore_mismatched_sizes=True` to `from_pretrained(...)` if appropriate."
        )
    raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")
```

**按这个顺序查**：

| 步 | 查什么 | 线索 |
|----|--------|------|
| 1 | **差在哪个维度** | 报告里的 MISMATCH 行给完整形状对比（`loading_report.py:231`）：<br>`"Reinit due to size mismatch - ckpt: torch.Size([32000, 4096]) vs model:torch.Size([128256, 4096])"`<br>最后一维常见于 `vocab_size`；第一维常见于 `hidden_size` / `intermediate_size` / `num_labels` |
| 2 | 是 `vocab_size` 吗 | 加过 special token 就要 `resize_token_embeddings`（`modeling_utils.py:2709`） |
| 3 | 是 `num_labels` 吗 | 分类头；只有 `hf_quantizer is None` 时才会判 mismatched（`core_model_loading.py:1372`） |
| 4 | 想「容忍」它 | `ignore_mismatched_sizes=True`：形状不对的参数**不加载**，退回随机初始化（`_initialize_missing_keys`，`modeling_utils.py:4828`）。注意这叫 **reinit**，不是「截断/补齐」 |
| 5 | name 对不上但形状对？ | 那不是 mismatch，是 unexpected + missing 这一对 → 回 8.1 |

**一个易踩的坑**：`mismatched` 的参数会被 `set_param_for_module` **从 missing 里 discard 掉**
（`core_model_loading.py:1367`），但不会 `setattr`。
所以它在 missing 集合里**看不到**——排查时**不能只看 missing/unexpected，必须看 mismatched 那一栏**。

### 8.3 遇到「CONVERSION」行

这是 `WeightConverter` 的 op 抛了异常（`core_model_loading.py:1294`）：

```python
# core_model_loading.py:1313-1321（节选）
tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
if isinstance(extras, tuple) and len(extras) == 2:
    length, target_keys = extras
    descriptor = f"{op_name} " if op_name else ""
    loading_info.conversion_errors[first_target_key] = (
        f"{tb_str}{e}\nError: {descriptor}on tensors destined for {target_keys}. Ckpt contains: {length}"
    )
```

| 报告里的文本 | 含义 |
|------|------|
| `Error: Chunk on tensors destined for model.layers.0.q_proj.weight. Ckpt contains: 1` | `Chunk` 在目标参数上失败；**checkpoint 里只提供了 1 个张量**（而 op 期望 N 个） |
| `via <OpName>` | 失败的 op 名 |

**`Ckpt contains: N` 是最有用的线索**——它告诉你 `collected_tensors` 里到底收进来几个张量；
N 与 op 期望值不符，通常说明 **source_pattern 没匹配上全部需要的 key**。排查方向：
① `conversion_mapping.py` 里这个 `model_type` 的 `source_patterns` 写对了吗
（`*.` 是层级通配，`core_model_loading.py:849`，不是 glob）；
② op 的 `dim` / `chunks` 参数对吗（`Chunk(dim=0, chunks=3)` vs `Chunk(dim=1, ...)`）；
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
>
> 上行：`download_kwargs`（`modeling_utils.py:4167`）→ `hub.cached_files`（`hub.py:299`）
> → 本地目录 / 缓存 / 下载三选一 → `checkpoint_files`；
> 中间：`get_init_context`（`:3791`）在 **meta device** 上造空壳，`_get_dtype`（`:815`）和
> `_get_device_map`（`integrations/accelerate.py:338`）把 dtype 和目标设备定死；
> 下行：`convert_and_load_state_dict_in_model`（`core_model_loading.py:1511`）用
> `safe_open` 惰性句柄 + 线程池 + `rename_source_key` + `WeightConverter` 把每个张量搬到该去的地方，
> 落进 `set_param_for_module`（`:1343`）并打上 `_is_hf_initialized`；
> 收尾：`_finalize_model_loading`（`:4550`）把 **没打上标记的**参数
> 交给 `_init_weights`（`:2374`）初始化，然后 tie weights、出报告。
>
> **三句话记忆**：
> 1. `WeightRenaming` 只改名字，`WeightConverter` 改名字 + 动张量。
> 2. `missing` = 模型的参数没被填上；`unexpected` = checkpoint 的 key 没处放；
>    `mismatched` = 名字对上了但形状不对（**会被 reinit，不会进 missing**）。
> 3. `_init_weights` 只碰**没被 checkpoint 覆盖**的参数——判据是 `_is_hf_initialized` 这个标记。

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
