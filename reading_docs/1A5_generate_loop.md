# `generate` 主循环精读：从 prompt 到逐 token

> 阅读目标：`model.generate(...)` 一次调用在库内部经过哪些函数、每步做什么。读完应能回答：
> ① 为什么第一步慢、后面快（prefill / decode）；② `GenerationConfig` 的 `do_sample` / `temperature` / `top_p` /
> `repetition_penalty` 分别在哪一行生效；③ 模型侧唯一钩子 `prepare_inputs_for_generation` 负责什么。
>
> 源文件（本仓库实测行数）：`src/transformers/generation/utils.py`（4089 行）、
> `configuration_utils.py`（1874 行）、`logits_process.py`（3222 行）、`stopping_criteria.py`（643 行）、
> `streamers.py`（406 行）以及 `continuous_batching/`（13 个文件，6083 行，本篇只指路）。
>
> **前置：[1A3_from_pretrained.md](./1A3_from_pretrained.md)**。`generate` 是 `from_pretrained` 之后的下一个动作。

---

## 速读区（TL;DR）

**一句话结论**：`generate()` = **九步准备** + **一个 while 循环**；模型侧唯一接口是
`prepare_inputs_for_generation`（本仓库已上提到 `GenerationMixin`），prefill 被抽到循环外的
`_prefill`，循环里每步只喂 1 个 token。

| # | 断言 | 出处（文件:行号） |
|---|------|------------------|
| 1 | **贪心和采样共用 `_sample`**，两者只在 token 选择那两行不同；DoLa / contrastive / group-beam / constrained-beam 的右值已是 **Hub 仓库名**（策略已搬离核心库） | `generation/utils.py:137-150` |
| 2 | `generate` 自己不跑循环：九步准备后查表 `getattr(type(self), GENERATION_MODES_MAPPING[mode])` 再调用 | `generation/utils.py:2481` |
| 3 | `max_new_tokens` 只在 `generate` 第 6 步被翻译成 `max_length`；之后循环里只有 `max_length` 一个概念，`stopping_criteria` 里只会有 `MaxLengthCriteria` | `generation/utils.py:1734` |
| 4 | 默认值 `max_length: 20` / `do_sample: False` / `top_k: 50` / `temperature: 1.0`——所以 `do_sample=True` 时**默认就带 top-k 50 过滤** | `generation/configuration_utils.py:601` |
| 5 | **Llama / Mistral 都没有自己的 `prepare_inputs_for_generation`**；通用实现在 `GenerationMixin`，全库 500+ 模型目录只剩 75 处定义 | `generation/utils.py:519` |
| 6 | 首步 / 后续步的分界就是一行：`next_sequence_length = 1 if model_kwargs["use_cache"] else None`，切片在 `input_ids[:, -next_sequence_length:]` | `generation/utils.py:2878` / `:552` |
| 7 | **`LogitsWarper` 基类已不存在**，只剩 `LogitsProcessor`；「processor vs warper」只剩语义差别（由装配处决定，不在类定义里） | `logits_process.py:49` |
| 8 | warper 整段被 `if generation_config.do_sample:` 包住——`temperature` 不配 `do_sample=True` 完全无效 | `generation/utils.py:1296` / `logits_process.py:242-246` |
| 9 | warper 顺序 `temperature → top_h → top_k → top_p → min_p`；`LogitNormalization` **永远最后** | `generation/utils.py:1296-1343` / `:1345-1356` |
| 10 | `RepetitionPenalty` 是「负数乘、正数除」，且**默认也作用在 prompt 上**；3D logits 分支只给连续批处理用，故 `supports_continuous_batching = False` | `logits_process.py:373` / `:308-309` / `:356` |
| 11 | stopping criteria 是 **OR（按位或）组合**，返回 per-batch bool；`max_length` 之后一定非 `None`，所以**至少有一条 `MaxLengthCriteria` 兜底** | `stopping_criteria.py:620` / `generation/utils.py:1358` |
| 12 | **`cache_position` 已从库中移除**，位置改由模型 `forward` 用 `past_key_values.get_seq_length()` 推算 | `generation/utils.py:623-633` / `models/llama/modeling_llama.py:385-389` |

**最容易踩的坑 / 反直觉发现**：
- **「`_sample` 是个 while 循环、第一次 forward 处理完整 prompt」已不成立**：prefill 被抽成独立方法 `_prefill`（`:3904`），在 `while` 之外先跑一次，循环体靠 `prefill_consumed` 跳过首轮 forward（`:2876-2883`）。
- **`cache_position` 是老教程的遗迹**：源码原话「This arg has been removed from the Transformers library」（`:628`），只有 remote-code 模型才临时造一个。
- **processor 处理的永远是未归一化的 logits**：`softmax` 只在 `do_sample=True` 那两行里做（`:2920-2925`）；warper 移除 token 是**填 `-inf` 而不是删掉**（改值不改形状）。
- **DoLa / contrastive search / group beam / constrained beam 已不在核心库**：右值是 Hub 仓库名，`generate` 会去下载一份 `generate.py` 跑（`_get_deprecated_gen_repo`，`:2203`）；`grep -c _contrastive_search` → 0。
- **`streamer` 不能与 beam search 同用**（`:1567-1569` 直接报错），`_beam_search` 的签名里压根没有该参数（`:3208`）。
- **`use_cache=False` 时多模态输入不会被丢**（`:613-620`），因为每步都是「第一次」——两处必须是同一个条件，否则多模态模型会缺输入。

**读完应该能回答的问题**：
- 不传任何生成参数时，`generate` 默认生成多少 token？为什么？
- `do_sample=False` 时把 `temperature` 设成 0.7 会发生什么？
- 首步慢、后续快，这个差别落在哪一行代码上？
- `max_new_tokens` 为什么在 `stopping_criteria` 里看不见？
- 为什么 `_beam_search` 要取 `max(2, 1+n_eos) * num_beams` 而不是 `num_beams`？

---

## 前置知识

- [1A3_from_pretrained.md](./1A3_from_pretrained.md)：`PreTrainedModel` / `model.config` / `model_kwargs` 的含义
- [1M1_llama_structure.md](./1M1_llama_structure.md)：`LlamaAttention.forward` 里 `past_key_values.update(...)` 那一层（本篇讲循环外层，它讲模型内部）
- [2A5_modular_code_generation.md](./2A5_modular_code_generation.md)：读源码的方式（为什么先 `grep` 再下结论）

---

## 一、`generate()` 在哪、由谁提供

入口是一个 **mixin 方法**，不在任何模型文件里：`generation/utils.py:359` 是
`class GenerationMixin(ContinuousMixin):`（生成能力的家，同时继承连续批处理），
`:2261` 是 `def generate(self, inputs=None, generation_config=None, logits_processor=None, stopping_criteria=None,
prefix_allowed_tokens_fn=None, synced_gpus=None, assistant_model=None, streamer=None, negative_prompt_ids=None,
negative_prompt_attention_mask=None, custom_generate=None, **kwargs)`。模型类通过多重继承拿到它
（`models/llama/modeling_llama.py:421`：`class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin)`）。

**`generate` 自己不跑循环**：准备工作后从一张表里查出「该调哪个循环函数」再调用（`:137-150`；取值那行在 `:2481`
`decoding_method = getattr(type(self), GENERATION_MODES_MAPPING[generation_mode])`）：

```python
GENERATION_MODES_MAPPING = {
    GenerationMode.SAMPLE: "_sample", GenerationMode.GREEDY_SEARCH: "_sample",
    GenerationMode.BEAM_SEARCH: "_beam_search", GenerationMode.BEAM_SAMPLE: "_beam_search",
    GenerationMode.ASSISTED_GENERATION: "_assisted_decoding",
    # Deprecated methods —— 右值已是 Hub 仓库名，策略已搬离核心库
    GenerationMode.DOLA_GENERATION: "transformers-community/dola",
    GenerationMode.CONTRASTIVE_SEARCH: "transformers-community/contrastive-search",
    GenerationMode.GROUP_BEAM_SEARCH: "transformers-community/group-beam-search",
    GenerationMode.CONSTRAINED_BEAM_SEARCH: "transformers-community/constrained-beam-search",
}
```

> greedy 和 sample **共用 `_sample`**（只在 token 选择那两行不同）；最后四项的右值是 **Hub 仓库名**，
> `generate` 会改成去 Hub 下载一份 `generate.py` 跑（`_get_deprecated_gen_repo`，`:2203`）。

---

## 二、`generate()` 的九个步骤

`generate` 正文里留着编号注释（官方主流程地图），行号为本仓库实测：

| 步骤 | 行号 | 做什么 | 干活的函数 |
|------|------|--------|-----------|
| 0.a | 2369 | 用户要 `custom_generate`（Hub 上一份自定义 `generate.py`）→ 直接调它返回 | `load_custom_generate`（`:454`） |
| 0.b | 2391 | `cache_implementation == "paged"` → 转 `generate_batch`（连续批处理） | `continuous_api.py:1237` |
| 1 | 2456 | 合并 kwargs 到 config，**判定 generation mode** | `_extract_generation_mode_kwargs`（`:2227`）、`_prepare_generation_config`（`:1771`）、`get_generation_mode`（`configuration_utils.py:534`） |
| 2 | 2507 | 建空的 processor / criteria 列表 | —— |
| 3 | 2514 | 决定真正的模型输入（`input_ids` 还是 `inputs_embeds` / `pixel_values`） | `_prepare_model_inputs`（`:646`） |
| 4 | 2549 | 补 `attention_mask` / `position_ids`；encoder-decoder 这里跑 encoder | `_prepare_attention_mask_for_generation`（`:775`）、`_prepare_position_ids_for_generation`（`:751`）、`_prepare_encoder_decoder_kwargs_for_generation`（`:809`） |
| 5 | 2575 | 准备自回归用的 `input_ids`，按 num_beams / num_return_sequences 展开 | `_prepare_decoder_input_ids_for_generation`（`:850`）、`_expand_inputs_for_generation`（`:910`） |
| 6 | 2601 | **`max_new_tokens` → `max_length`** 的换算 | `_prepare_generated_length`（`:1715`）、`_validate_generated_length`（`:1669`） |
| 7 | 2620 | **建 KV cache** | `_prepare_cache_for_generation`（`:1929`） |
| 8 | 2646 | **建 logits processor 表和 stopping criteria 表** | `_get_logits_processor`（`:1123`）、`_get_stopping_criteria`（`:1358`） |
| 9 | 2667 | 调循环函数 | `decoding_method(...)`（`:2668`） |

几个关键点：**第 6 步是 `max_new_tokens` 唯一的落地点**（`:1734`：
`generation_config.max_length = generation_config.max_new_tokens + input_ids_length`），之后整个循环里只有
`max_length`，**`stopping_criteria` 里只会有 `MaxLengthCriteria`**；**第 5 步会先往 streamer 里塞 prompt**
（`:2598-2599`：`streamer.put(input_ids.cpu())`）——这是 `skip_prompt=True` 能工作的前提；**第 7 步用 `max_length`
给静态 cache 定容量**（`:2624`：`max_cache_length = generation_config.max_length - 1`，`-1` 是因为最后一个 token 的
forward 不再写 cache），`:2631` 调 `_prepare_cache_for_generation(...)`。

```
model.generate(input_ids, max_new_tokens=32, do_sample=True, temperature=0.7)
   ├─[0.a] custom_generate? → Hub 下载 generate.py 并调用  ✗   ├─[0.b] cache_implementation=="paged"? → generate_batch() ✗
   ▼ 输入准备（不碰权重，纯 tensor 搬运与配置合并）
   [1] _prepare_generation_config  kwargs > self.generation_config > 全局默认 → get_generation_mode()
   [3] _prepare_model_inputs → inputs_tensor, model_input_name     [4] attention_mask / position_ids
   [5] _expand_inputs_for_generation → [batch*beams, prompt_len]；若 streamer: streamer.put(prompt)
   [6] max_new_tokens + prompt_len ──> generation_config.max_length；logits_to_keep = 1
   [7] _prepare_cache_for_generation ──> model_kwargs["past_key_values"] = DynamicCache
   [8] _get_logits_processor / _get_stopping_criteria ──> 两张有序列表
   [9] decoding_method(self, input_ids, logits_processor=…, stopping_criteria=…, **model_kwargs) → _sample / _beam_search
```

---

## 三、`GenerationConfig`：循环的「参数面板」

`generation/configuration_utils.py:100` 是类定义，`:385` 是 `__init__`，字段全是 `kwargs.pop(name, None)`（默认
`None`，真默认值在别处给）。

| 字段 | 定义行 | 控制什么 | 在哪里真正生效 |
|------|--------|----------|---------------|
| `max_new_tokens` | `:393` | 新生成多少 token | `generate` 第 6 步 → 换算进 `max_length`（`utils.py:1734`） |
| `max_length` | `:392` | 总长度（prompt + 新） | `MaxLengthCriteria`（`stopping_criteria.py:62`） |
| `do_sample` | `:401` | 走采样还是贪心 | `_sample` 里的 `if do_sample:`（`utils.py:2920`） |
| `temperature` | `:412` | logits 缩放，控制随机性 | `TemperatureLogitsWarper.__call__`（`logits_process.py:301`） |
| `top_k` | `:413` | 只留概率最高的 k 个 | `TopKLogitsWarper`（`logits_process.py:542`） |
| `top_p` | `:414` | 只留累积概率 ≥ p 的最小集合 | `TopPLogitsWarper.__call__`（`logits_process.py:527`） |
| `repetition_penalty` | `:420` | 已出现 token 的分数缩放 | `RepetitionPenaltyLogitsProcessor.__call__`（`logits_process.py:373`） |
| `num_beams` | `:402` | beam 数；1 = 不用 beam | `get_generation_mode`（`configuration_utils.py:550`）、`_beam_search` |
| `eos_token_id` | `:451` | 停止符 | `EosTokenCriteria`（`stopping_criteria.py:543`） |

**默认值集中在静态方法 `_get_default_generation_params()`（`:601`）**（类上的静态方法，不是实例属性）：
`max_length: 20`、`min_length: 0`、`do_sample: False`、`use_cache: True`、`early_stopping: False`、`num_beams: 1`、
`temperature: 1.0`、`top_k: 50`、`top_p: 1.0`、`repetition_penalty: 1.0`、`num_return_sequences: 1`。
> `max_length: 20` 就是「不传参数只生成 20 个 token」的来源；`top_k: 50` 意味着 `do_sample=True` 时**默认带 top-k 50 过滤**（除非显式设 `top_k=0`）。

**三层来源合并**（`_prepare_generation_config`，`utils.py:1771`；优先级注释 `:1778-1780`：
`user-defined kwargs or generation_config > self.generation_config > global default values`）实现为三次
`update`（`:1802-1808`）：先 `self.generation_config.to_dict()`、再 `global_defaults`、最后 `kwargs`；前两次带
`defaults_only=True`（含义「只改当前还是 `None` 的字段」，`configuration_utils.py:1315`），**最后一次不带**，故用户
kwargs 无条件覆盖。`update` 返回「没被消费掉的 kwargs」——那些是模型自己的 forward 参数（如 `pixel_values`），
被收进 `model_kwargs` 继续往下传。

**从 `generation_config.json` 到 `model.generation_config`**：加载发生在 `from_pretrained` 里
（`modeling_utils.py:4402` 调 `model.adjust_generation_fn(generation_config, from_auto_class, from_pipeline, ...)`，
`utils.py:391`）：用户显式传了就用它；否则 `GenerationConfig.from_pretrained(path)`（`configuration_utils.py:932`，
默认文件名 `generation_config.json` 见 `:886`），拿不到就退化为 `from_pretrained(..., config_file_name="config.json",
_from_model_config=True)`。相关入口：`from_dict`（`:1104`）、`from_model_config`（`:1266`）。模型**造好时**也会先给
兜底（`modeling_utils.py:1356-1361`）：`if self.can_generate(): self.generation_config =
self.generation_config_class.from_model_config(config)`（`NotImplementedError` 时用 `from_model_config` 失败则建空实例）；
`can_generate()` 判断的就是「有没有 `GenerationMixin`」。

**特殊 token 在这里被张量化**：`_prepare_special_tokens`（`utils.py:2037`）把 `eos_token_id` / `pad_token_id` 转成
tensor **写进另一组下划线开头的属性**（`:2114-2117`）：`generation_config._eos_token_tensor` /
`_pad_token_tensor` / `_decoder_start_token_tensor`。理由（注释 `:2110-2112`）是为了 end-to-end 编译
（`torch.compile` 不喜欢同一属性在两次 trace 之间换类型），所以循环里见的都是 `_eos_token_tensor`。顺带：
`pad_token_id` 没设时**直接用 `eos_token_id[0]` 顶上**（`:2086-2087`）。

---

## 四、`prepare_inputs_for_generation`：模型侧唯一的钩子

**反直觉事实：Llama 和 Mistral 都没有自己的 `prepare_inputs_for_generation` 了。**

```bash
$ grep -rn "def prepare_inputs_for_generation" src/transformers/models/llama/    # （无输出）
$ grep -rn "def prepare_inputs_for_generation" src/transformers/models/mistral/  # （无输出）
$ grep -rln "def prepare_inputs_for_generation" src/transformers/models/ | wc -l
75          # 全库 500 多个模型目录里只有 75 处
```

原因：通用实现已**上提（hoist）到 `GenerationMixin` 自己**（`generation/utils.py:519`）。正确心智模型：
> **通用实现在 `GenerationMixin`（`:519`），模型文件里的覆盖是「补丁」，不是「主体」。**

通用实现的六个编号步骤（`:519-644`）：

| 步骤 | 行号 | 干什么 |
|------|------|--------|
| 1 | `:540` | 选 `input_ids`（decoder-only）或 `decoder_input_ids`（enc-dec），**按 `next_sequence_length` 切片** |
| 2 | `:556-563` | 挂 `past_key_values` / `position_ids` / `token_type_ids` |
| 3 | `:565` | 把 `position_ids` / `token_type_ids` 切到和 `input_ids` 等长 |
| 4 | `:573` | cache 是可编译类型时，2D attention mask 提升成 4D causal mask |
| 5 | `:608` | 其余 kwargs 原样透传 `forward`，**但多模态输入在 decode 阶段被丢掉** |
| 6 | `:635` | 调用前把 tensor 挪到模型设备上 |

**第 1 步是「首步 / 后续步」差异的物理实现**（`:552`）：

```python
# `clone` calls in this function ensure a consistent stride. See #32227
input_ids = input_ids[:, -next_sequence_length:] if next_sequence_length is not None else input_ids
```

- `next_sequence_length=None` → 不切片，**整个 prompt 进去**（prefill）；`=1` → 只取最后一个 token（decode）。

**第 5 步的「丢多模态输入」是一张显式白名单**（`utils.py:150-165`）：
`MULTIMODAL_INPUTS_TO_DROP_OUTSIDE_PREFILL = ("pixel_values", "pixel_mask", "input_features", "input_features_mask",
"pixel_values_videos", "num_local_patches", "high_res_pixel_values", "image_patches_indices", "image_patches",
"image_sizes", "image_sizes_videos", "pixel_attention_mask", "pixel_values_images", "num_local_patches")`，
判断在 `:616-620`：不在 prefill 且 `use_cache=True` 时 `continue` 掉（图像的 KV 已在 prefill 写进 cache）。

**三个真实覆盖对比**：

| 模型 | 行号 | 覆盖干什么 |
|------|------|-----------|
| Llama / Mistral / Qwen3 等 | —— | **无**，直接用 `GenerationMixin` 通用实现 |
| `MllamaForConditionalGeneration` | `models/mllama/modeling_mllama.py:1520` | prefill 后**把图像输入置 `None`**，并把 `logits_to_keep` 透传（实现 `:1536-1559`：`if not is_first_iteration and use_cache: model_inputs["pixel_values"] = None`） |
| `Gemma3ForConditionalGeneration` | `models/gemma3/modeling_gemma3.py:1052` | decode 阶段**把 `token_type_ids` 置 `None`**（避免双向 mask 套到文本上） |
| `Mamba2ForCausalLM` | `models/mamba2/modeling_mamba2.py:838` | decode 阶段**把 `attention_mask` 置 `None`**（SSM 不需要），注意缓存参数叫 `cache_params` 而非 `past_key_values` |

**共同模式**：全都 `super().prepare_inputs_for_generation(...)` 打底，然后**只针对 decode 阶段改一两个键**。
判断条件三种写法 `if not is_first_iteration and use_cache:` / `if not (is_first_iteration or not use_cache):` /
`if use_cache and not is_first_iteration:` **是同一个布尔式**：`is_first_iteration == False and use_cache == True`
——含义是「我现在处于 decode 阶段，cache 里有历史，所以只传增量」。

---

## 五、采样循环：`_sample`

`utils.py:2783` 起，本篇主角。

### 5.1 重要的结构变化：prefill 挪到循环外

老版本文档把 `_sample` 描述成「一个 while 循环，第一次 forward 处理完整 prompt」。**本仓库不成立了**——
第一次 forward 被抽成独立方法 `_prefill`（`:3904`），在循环外先跑一次（`:2867-2873`）：
`outputs = self._prefill(input_ids, generation_config, model_kwargs, is_first_iteration=not generation_config.is_assistant)`。

然后循环体第一件事是判断「这轮要不要真的 forward」（`:2876-2883`）：

```python
with self._optimize_model_for_decode():
    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        if prefill_consumed:
            next_sequence_length = 1 if model_kwargs["use_cache"] else None
            model_inputs = self.prepare_inputs_for_generation(input_ids, next_sequence_length=next_sequence_length, **model_kwargs)
            outputs = model_forward(**model_inputs, return_dict=True)
        prefill_consumed = True
```

- 第一次进循环：`prefill_consumed` 还是 `False` → **跳过 forward**，直接拿 `_prefill` 的 `outputs`
- 第二次起：`next_sequence_length = 1` → `prepare_inputs_for_generation` 在 `:552` 切出 `input_ids[:, -1:]` → 只喂 1 个 token

`_prefill` 自己用 `next_sequence_length=None` 走完整 prompt（`:3941-3948`）；它只在**从中断的 cache 续跑**时非
`None`（`:3923-3937`，判断依据 `past_length = cache.get_seq_length()`）。

```
prompt = [t0, t1, t2, t3]        max_new_tokens=3, use_cache=True
=== 首步 prefill（在 while 之外）===
  _sample:2868 outputs = self._prefill(input_ids, ...)
  _prefill:3941 prepare_inputs_for_generation(next_sequence_length=None) → prepare:552 不切 → [t0,t1,t2,t3]
    embed → 4 个位置 hidden；每层 QKV 全算 4 个位置；past_key_values.update() 写 4 组 K/V
    lm_head 只取最后一个位置 logits（logits_to_keep=1）→ 预测 t4；cache: [K/V × 4] ← 只此一次
=== 后续步 decode（在 while 里）===
  _sample:2878 next_sequence_length = 1 → prepare:552 input_ids[:, -1:] → 只 [t4]
    embed → 1 个位置 hidden；每层 Q = 1 个位置，K/V 新算 1 组 + 从 cache 读 4 组
      → KV 长度 5，Q 长度 1（这就是 KV cache 的全部意义）；update() 追加 1 组；lm_head → t5
  （重复 max_new_tokens 次，每步只多算 1 个位置）
```

**为什么快**：prefill 的 forward 里 Q 长度 = prompt 长度（矩阵乘法打得满）；decode 里 Q 长度 = 1，唯一的「大」
操作是去 cache 读历史 K/V——**已算过的不重算，这就是 `use_cache=True` 的全部收益**。`use_cache=False` 时
`next_sequence_length=None`（`:2878` 三元表达式），每步重喂整个 `input_ids`、重算所有 K/V，结果正确但整个生成
过程 forward 的 token 总数从 O(N) 涨到 O(N²)。

### 5.2 循环体逐行

| 行号 | 代码 | 说明 |
|------|------|------|
| `:2876` | `while self._has_unfinished_sequences(...)` | 循环条件（`:2680`）；`synced_gpus` 时走 all_reduce 保证多卡不早退 |
| `:2877` | `if prefill_consumed:` | 首步跳过 forward |
| `:2884` | `model_kwargs = self._update_model_kwargs_for_generation(...)` | **把新 cache / 新位置 / 新 mask 写回 kwargs** |
| `:2894` | `next_token_logits = outputs.logits[:, -1].to(copy=True, dtype=torch.float32, ...)` | 只取最后位置；转 fp32 算概率；`copy=True` 不拖住巨大 logits |
| `:2897` | `next_token_scores = logits_processor(input_ids, next_token_logits)` | **所有 logits 变换在这里一次做完**（第六节） |
| `:2899-2918` | `if return_dict_in_generate: scores += (...)` | 可选收集 scores / attentions / hidden_states |
| `:2920-2925` | 见下 | token 选择 |
| `:2928` | `if has_eos_stopping_criteria:` | 已结束的序列强制喂 pad |
| `:2932` | `input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)` | **序列增长** |
| `:2933-2934` | `if streamer is not None: streamer.put(next_tokens.cpu())` | 流式挂载点 2 |
| `:2936` | `unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)` | **只管不增**：`&` 意味着老早结束的序列不会复活 |
| `:2937` | `this_peer_finished = unfinished_sequences.max() == 0` | 全 batch 结束才退出 |
| `:2941` | `del outputs` | 不删的话下一步的 logits 被上一轮引用拖住，显存峰值翻倍 |

token 选择那两行（`:2920-2925`）是 greedy 和 sample 的**全部分歧**：

```python
# token selection
if do_sample:
    probs = nn.functional.softmax(next_token_scores, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
else:
    next_tokens = torch.argmax(next_token_scores, dim=-1)
```

`torch.argmax` vs `torch.multinomial`；`softmax` 只在采样时做——**`logits_processor` 处理的始终是未归一化 logits**。
结束时 `if streamer is not None: streamer.end()`（`:2943-2944`）。

### 5.3 `_has_unfinished_sequences` 与 `_update_model_kwargs_for_generation`

**循环条件**（`:2680`）单卡下就是 `not this_peer_finished`；多卡 FSDP/ZeRO-3 时会 `dist.all_reduce` 一个 flag，
**所有 rank 都完成才退出**（否则先跑完的 rank 停了、其他 rank 还在 collective，会死锁）。

**状态更新**（`:940`）是循环里唯一负责「把这一步的结果变成下一步的输入」的函数：

| 行号 | 更新什么 |
|------|----------|
| `:954-962` | 遍历 `ALL_CACHE_NAMES`，把 `outputs.past_key_values` 写回 `model_kwargs` |
| `:964-966` | `token_type_ids` 续一个 |
| `:967-971` | `mm_token_type_ids` 续零 |
| `:974-985` | `position_ids` **末位 +1 再续一个**（不重算：`torch.arange(num_new_tokens, ...) + position_ids[..., -1:] + 1`，再 `cat`） |
| `:987-992` | `attention_mask` 续一个 1 |

`ALL_CACHE_NAMES`（`:129-135`）= `["past_key_values", "cache_params"(mamba), "state"(rwkv), "mems"(xlnet),
"past_buckets_states"(reformer)]`——不同模型族把 cache 放在不同名字下，loop 这边只认这张表。

---

## 六、`logits_process.py`：logits 上的流水线

### 6.1 一个类，两个名字

本仓库里**没有 `LogitsWarper` 这个基类了**（`grep -rn "class LogitsWarper" src/` 无输出）。只有
`LogitsProcessor`（`logits_process.py:49`）一个抽象基类，`TemperatureLogitsWarper` / `TopPLogitsWarper` /
`TopKLogitsWarper` / `MinPLogitsWarper` 这些**名字里带 Warper 的类直接继承它**：

```python
class TemperatureLogitsWarper(LogitsProcessor):     # :238
class TopPLogitsWarper(LogitsProcessor):            # :473
class TopKLogitsWarper(LogitsProcessor):            # :542
class MinPLogitsWarper(LogitsProcessor):            # :704
```

所以「processor vs warper」**只剩语义差别，没有类型差别**：processor = **不管采样不采样都生效**的约束
（`MinLengthLogitsProcessor:101`、`RepetitionPenaltyLogitsProcessor:306`、`NoRepeatNGramLogitsProcessor:1073`）；
warper（名字带 Warper）= **只在 `do_sample=True` 时才挂上去**的分布整形。这个语义差别**不在类定义里，而在装配处**。

### 6.2 `LogitsProcessorList`：调用顺序就是列表顺序

`logits_process.py:63` 是 `list` 的子类，`__call__` 在 `:70`：纯顺序执行，**上一个的输出是下一个的输入**，
无并行无重排（`inspect.signature` 决定要不要透传 kwargs）。所以「temperature 在 top_p 之前还是之后」是实打实
影响结果的，而这个顺序由 `_get_logits_processor` 的 `append` 顺序决定。

### 6.3 装配顺序（`utils.py:1123`）

**第一段：无条件 processor**（行号即顺序）：`guidance_scale != 1` → `UnbatchedClassifierFreeGuidanceLogitsProcessor`
`:1145`；`sequence_bias` → `SequenceBiasLogitsProcessor` `:1155`；`encoder_repetition_penalty != 1.0` →
`EncoderRepetitionPenaltyLogitsProcessor` `:1162`；**`repetition_penalty != 1.0` → `RepetitionPenaltyLogitsProcessor`
`:1175`**；`no_repeat_ngram_size > 0` → `NoRepeatNGramLogitsProcessor` `:1183`；`encoder_no_repeat_ngram_size > 0` →
`EncoderNoRepeatNGramLogitsProcessor` `:1195`；`bad_words_ids` → `NoBadWordsLogitsProcessor` `:1208`；
`min_length > 0` → `MinLengthLogitsProcessor` `:1219`；`min_new_tokens > 0` → `MinNewTokensLengthLogitsProcessor` `:1231`；
`prefix_allowed_tokens_fn` → `PrefixConstrainedLogitsProcessor` `:1240`；`forced_bos_token_id` →
`ForcedBOSTokenLogitsProcessor` `:1247`；`forced_eos_token_id` → `ForcedEOSTokenLogitsProcessor` `:1253`；
`remove_invalid_values` → `InfNanRemoveLogitsProcessor` `:1261`；`exponential_decay_length_penalty` →
`ExponentialDecayLengthPenalty` `:1263`；`suppress_tokens` → `SuppressTokensLogitsProcessor` `:1271`；
`begin_suppress_tokens` → `SuppressTokensAtBeginLogitsProcessor` `:1284`。之后合并用户自定义的（`:1293`）：
`processors = self._merge_criteria_processor_list(processors, logits_processor)`（`:1396`，策略是「**同类型以用户的
为准**」：遇到同类型对象换成用户的，再把用户独有的追加到尾部）。

**第二段：采样专属 warper（`:1296-1343`）**，整段被 `if generation_config.do_sample:`（`:1296`）包住——这是
`temperature` 不配 `do_sample=True` 就完全无效的原因（docstring 也直说，`logits_process.py:242-246`）。
顺序由 `append` 决定：**temperature → top_h → top_k → top_p → min_p → typical_p → epsilon → eta**，且 `min_p`
的注释写了它**必须在 temperature 之后**。判据：`temperature != 1.0`、`top_h is not None`、`top_k != 0`、
`top_p < 1.0`、`min_p is not None`；beam 场景 `min_tokens_to_keep = len(eos)+1`，否则 1。

**收尾两段**（`:1345-1356`）：`watermarking_config` → 追加 watermarking processor（注释：必须在所有 logits
处理之后，见 #34630）；`renormalize_logits is True` → 追加 `LogitNormalization()`（`logits_process.py:1779`）
**永远最后**——因为它是 log-softmax，之后任何改 logits 绝对值的操作都会破坏归一化。

### 6.4 三个 warper 的真实实现

- **`TemperatureLogitsWarper.__call__`（`logits_process.py:301`）一行**：`return scores / self.temperature`。
  `__init__`（`:288`）校验必须是严格正 float（`:289`），`temperature == 0.0` 时在报错里提示「想要贪心就设
  `do_sample=False`」（`:294-297`）。
- **`TopPLogitsWarper.__call__`（`:527`）8 行**：`descending=False` **升序**排序 → `softmax` 后 `cumsum`，
  移除条件是 `sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)`（等价「保留概率最大的那一撮，
  累积到 p」）→ `sorted_indices_to_remove[..., -min_tokens_to_keep:] = 0`（至少留 k 个，避免全被滤光）→
  `scatter` 回原索引 → `scores.masked_fill(indices_to_remove, self.filter_value)`。被移除的 token **不是删掉而是
  填 `-inf`**（`filter_value` 默认 `-float("Inf")`），后续 `softmax` 给它 0 概率——**这是 warper 的统一手法：改值不改形状**。
- **`RepetitionPenaltyLogitsProcessor.__call__`（`:373`）核心 4 行**：`score = torch.gather(scores, 1, input_ids)`；
  `score = torch.where(score < 0, score * self.penalty, score / self.penalty)`；`scores.scatter(1, input_ids, score)`。
  **「负数乘、正数除」**：logits 可正可负，要让一个 token 更不可能，正数除以 penalty（变小）、负数乘以 penalty
  （更负），两者 softmax 后都是概率下降（docstring 表格 `:310-315`：penalty > 1 惩罚，0 < penalty < 1 鼓励）。
  `input_ids` 是**到当前步为止的完整序列（含 prompt）**，所以 penalty **默认也作用在 prompt 上**（docstring 原话
  `:308-309`），想排除得用 `prompt_ignore_length`（`:358`）。它上面的 `if scores.dim() == 3:` 分支（`:375-401`）
  是给连续批处理用的，普通 `generate` 走不到——这也是 `supports_continuous_batching = False`（`:356`）的原因。

---

## 七、`stopping_criteria.py`：什么时候停

| 类 | 行号 | 判据 |
|---|------|------|
| `StoppingCriteria`（ABC） | `:48` | 基类，`__call__(input_ids, scores, **kwargs) -> BoolTensor` |
| `MaxLengthCriteria` | `:62` | `cur_len >= max_length`（`__call__` 在 `:79`，只有三行实质代码；`max_position_embeddings` 只发一次警告，**不参与判定**） |
| `MaxTimeCriteria` | `:93` | 超时 |
| `StopStringCriteria` | `:118` | 解码后文本命中 `stop_strings`（要 tokenizer） |
| `EosTokenCriteria` | `:543` | 最新 token 命中 eos（`__call__` 在 `:560`：`torch.isin(input_ids[:, -new_token_length:], self.eos_token_id).any(dim=-1)`，`new_token_length` 默认 1；支持多个 eos `:553-558`） |
| `ConfidenceCriteria` | `:595` | 投机解码用 |
| `StoppingCriteriaList` | `:618` | 容器 |

**OR 组合**（`StoppingCriteriaList.__call__`，`:620`）：

```python
is_done = torch.full((input_ids.shape[0],), False, ...)
for criteria in self:
    is_done = is_done | criteria(input_ids, scores, **kwargs)
return is_done
```

`|`（OR）：任意一条命中就停该序列；返回**每个 batch 元素一个 bool**，所以同一 batch 里不同序列可以在不同步停下
——正好对应 `_sample:2936` 的 `unfinished_sequences & ~stopping_criteria(...)`。

装配处 `_get_stopping_criteria`（`utils.py:1358`）顺序：① `max_length is not None` → `MaxLengthCriteria`（顺带把
`config.max_position_embeddings` 传进去做提醒）；② `max_time is not None` → `MaxTimeCriteria`；③ `stop_strings is
not None` → `StopStringCriteria`（**没传 tokenizer 直接报错**）；④ `_eos_token_tensor is not None` →
`EosTokenCriteria`；⑤ `is_assistant and assistant_confidence_threshold > 0` → `ConfidenceCriteria`。再
`_merge_criteria_processor_list` 合并用户的（`:1393`）。**第 1 条是无条件的**——`max_length` 在
`_prepare_generated_length` 之后**一定非 `None`**，所以生成循环**至少有一条 `MaxLengthCriteria` 兜底**，不可能无限跑。

循环里只有一处用它（`:2936`）。而**进入循环之前**有一段「判断要不要把最后那个 token 换成 pad」的逻辑
（`:2832` 计算、`:2928-2930` 使用）：`has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in
stopping_criteria)`，命中时 `next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)`。
**小技巧**：用 `hasattr(criteria, "eos_token_id")` 反查（duck typing，不是 `isinstance`）来决定要不要做 pad 替换。

---

## 八、KV cache 在循环里的角色

**本篇只讲「循环怎么把它传下去」，cache 内部结构留给 2A3。**

三个接触点：**建**（`utils.py:2631` → `_prepare_cache_for_generation:1929`，按 `cache_implementation` 造 cache 写进
`model_kwargs["past_key_values"]`）；**传**（`prepare_inputs_for_generation` `:557-558`：
`if past_key_values is not None: model_inputs["past_key_values"] = past_key_values`）；**回写**
（`_update_model_kwargs_for_generation:953-961`）——第三条是循环能持续的关键：模型 forward 返回的是「这一步之后的
cache」，必须替换掉旧的，写回用 `ALL_CACHE_NAMES`（`:951-961`，注释 `# update past_key_values keeping its naming used
in model code`；`past_buckets_states` / `mems` 这两个老模型的名字会被映射回 `past_key_values`）。

**建哪种 cache**（`:1929`）先走三条快速通道——用户自己传了 cache（`:1947`，校验冲突后直接用）、`use_cache=False`
（`:1962`）、模型不支持 default `DynamicCache`（`:1966`，见 `_supports_default_dynamic_cache:1908`），都直接返回不建。
否则按 `cache_implementation` 分派：`"offloaded"` → `DynamicCache(offloading=True)` `:1977`；静态族 →
`_prepare_static_cache`（`:1980` / `:1858`）；`"quantized"` → `QuantizedCache` `:2001`；其余（`None` / `"dynamic"`）→
`DynamicCache(...)` `:2014`（**绝大多数模型走这条**）；encoder-decoder 再包一层 `EncoderDecoderCache` `:2021`。

**循环里「当前步的位置」从 `get_seq_length()` 来**：本仓库里 `generate` 和 `prepare_inputs_for_generation` **都不再
传 `cache_position`**。证据（`utils.py:623-633`）：只有 remote-code 老模型才临时造一个，且带警告——注释原话
「**This arg has been removed from the Transformers library**」（`:628`），
`if self.is_remote_code() and "cache_position" in set(inspect.signature(self.forward).parameters)` 才走
`past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0`。现在位置由**模型 forward
自己**从 cache 长度推出，Llama（`models/llama/modeling_llama.py:385-389`）与 Mistral
（`models/mistral/modeling_mistral.py:359-362`）写法一样：

```python
if position_ids is None:
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
    position_ids = position_ids.unsqueeze(0)
```

prefill 时 `past_seen_tokens = 0`、`input_length = prompt_len` → `position_ids = [0..prompt_len-1]`；decode 时
`past_seen_tokens = prompt_len + n`、`input_length = 1` → `position_ids = [prompt_len + n]`——**一个 token 一个位置号，
永远连续**，这就是 RoPE 能跨 cache 正确工作的原因。`get_seq_length()` 的链（细节留给 2A3）：`Cache.get_seq_length`
（`cache_utils.py:1492`，`DynamicCache` 继承它）转发到 `DynamicLayer.get_seq_length`（`cache_utils.py:154`），返回
`self.keys.shape[-2]`（未初始化则 0）——所以 `DynamicCache` **不需要任何显式位置参数，它的长度就是答案**。

**`use_cache=False` 的分支**：`generate` 第 9 步前一行（`utils.py:2665`）`model_kwargs["use_cache"] =
generation_config.use_cache`，于是 `_sample:2878` 的三元表达式给出下表；`prepare_inputs_for_generation` 第 5 步
（`:613-620`）也读 `kwargs.get("use_cache", True)`，**`use_cache=False` 时多模态输入不会被丢**（每步都是「第一次」）
——两处必须是同一个条件，否则多模态模型在 `use_cache=False` 下会缺输入。

| `use_cache` | `next_sequence_length` | 每步 forward 的输入 |
|---|---|---|
| `True`（默认） | `1` | `input_ids[:, -1:]`，靠 cache 看历史 |
| `False` | `None` | 整个 `input_ids`，全量重算 |

---

## 九、流式输出：`streamers.py` 的三个挂载点

流式完全靠 `generate` / `_sample` 里对 `streamer.put(...)` 的三次调用实现，**核心循环本身不知道「流式」这回事**：
① `utils.py:2598-2599` `streamer.put(input_ids.cpu())`（先喂 prompt，`skip_prompt=True` 靠它）；② `:2933-2934`
`streamer.put(next_tokens.cpu())`（每个新 token）；③ `:2943-2944` `streamer.end()`。都在 `.cpu()` 上——streamer 是纯
Python 侧的东西。

接口只有两个方法（`streamers.py:28`）：`BaseStreamer.put(value)` / `end()`，都 `raise NotImplementedError`。
`TextStreamer.put`（`:80`）**只在完整的词之后才输出**：换行 → 全刷；CJK 字符 → 直接刷；否则 → **只刷到最后一个空格**
（`text[self.print_len : text.rfind(" ") + 1]`）；`end()`（`:114`）刷尾巴。所以流式按「词」而非「token」呈现。

`TextIteratorStreamer`（`:157`）把「打印」换成「塞队列」：`__init__`（`:197`）建 `queue.Queue()`，`on_finalized_text`
（`:209`）`put`，`__next__`（`:218`）`get(timeout=...)` 并在 `stop_signal`（`None`）时 `raise StopIteration`。
**`generate()` 是阻塞的**，所以要边生成边消费必须把 `generate` 扔到另一个线程（docstring 例子 `:181-195` 用
`threading.Thread(target=model.generate, kwargs=...)`），主线程 `for new_text in streamer:`——**Queue 就是两个线程之间
的桥**。同文件另有 `AsyncTextIteratorStreamer`（`:226`，换 `asyncio.Queue` 配 `async for`）和 `TextDiffusionStreamer`
（`:314`，多一个 `put_draft` 于 `:376`）。

一个已知限制：`_validate_generation_mode`（`utils.py:1557`）里 `:1567-1569` 对
`BEAM_SEARCH` + `"streamer" in generation_mode_kwargs` 直接 `raise ValueError(...)`——beam 的「当前序列」会被反复改写
（beam 换轨），流出去的内容没法撤回，所以禁掉；**`_beam_search` 的签名里压根没有 `streamer` 参数**（`:3208`）。

---

## 十、解码策略对比

| 策略 | `GENERATION_MODES_MAPPING` 的值 | 入口函数 | 行号 | 触发条件 |
|------|-------------------------------|----------|------|----------|
| 贪心 greedy | `"_sample"` | `_sample` | `utils.py:2783` | `num_beams=1, do_sample=False` |
| 多项式采样 | `"_sample"` | `_sample` | `utils.py:2783` | `num_beams=1, do_sample=True` |
| beam search | `"_beam_search"` | `_beam_search` | `utils.py:3208` | `num_beams>1, do_sample=False` |
| beam sampling | `"_beam_search"` | `_beam_search` | `utils.py:3208` | `num_beams>1, do_sample=True` |
| 投机解码 assisted | `"_assisted_decoding"` | `_assisted_decoding` | `utils.py:3562` | 传了 `assistant_model` / `use_mtp` / `prompt_lookup_num_tokens` |
| DoLa | `"transformers-community/dola"` | Hub 仓库 | —— | `dola_layers is not None`（已弃用） |
| contrastive search | `"transformers-community/contrastive-search"` | Hub 仓库 | —— | `top_k>1 and penalty_alpha>0` |
| group beam | `"transformers-community/group-beam-search"` | Hub 仓库 | —— | `num_beam_groups>1` |
| constrained beam | `"transformers-community/constrained-beam-search"` | Hub 仓库 | —— | `constraints` / `force_words_ids` |

**判定逻辑全在 `get_generation_mode`（`configuration_utils.py:534`）**，主干（`:546-568`）：`constraints` /
`force_words_ids` 非空 → CONSTRAINED_BEAM_SEARCH；否则 `num_beams is None or == 1` 时，`do_sample is not True` → 再看
`top_k > 1 and penalty_alpha > 0` 则 CONTRASTIVE_SEARCH，否则 GREEDY_SEARCH，`do_sample` 为真 → SAMPLE；多 beam 时
`num_beam_groups > 1` → GROUP_BEAM_SEARCH，`do_sample is True` → BEAM_SAMPLE，否则 BEAM_SEARCH。**注意 `do_sample is not
True` 用的是 `is not True`**，不是 `not`——所以 `do_sample=None`（未设）会被当成贪心；同样 `num_beams is None` 也算 1。

**`_beam_search` 和 `_sample` 的结构差异**：循环体骨架一模一样（`_prefill` 在循环外、`prefill_consumed` 判断、
`_update_model_kwargs_for_generation`、`del model_outputs`），差异在「怎么选下一步」：`_get_top_k_continuations`
（`:3077`，从 `num_beams * vocab_size` 个候选里取 top-K，K = `max(2, 1+n_eos) * num_beams`）；
`_check_early_stop_heuristic`（`:3008`）；`_beam_search_has_unfinished_sequences`（`:3055`，比 `_has_unfinished_sequences`
更严，还要考虑「还有没有可能找到更好的」）；`_get_running_beams_for_next_iteration`（`:3131`）；
`_update_finished_beams`（`:3153`）；`_flatten_beam_dim` / `_unflatten_beam_dim`（`:2976` / `:2982`，`[B, beams, ...] ↔
[B*beams, ...]`，因为 `forward` 只认 2D batch）。设计 K > num_beams 的理由写在 `:3288-3292`：**如果 top-`num_beams`
全都选了 EOS，就没有活的 beam 可以继续了**。数值上全程在 log 域做加法（`:3388` `log_probs =
nn.functional.log_softmax(logits, dim=-1)`），因为要**累加**多个 token 的分数，概率连乘会下溢。

---

## 十一、另一套实现：`generation/continuous_batching/`

`generate()` 里那条「0.b」分支（`utils.py:2391`）把人送到 `generation/continuous_batching/`——它不是主循环的选项，
而是**另一套完全独立的执行引擎**（13 个文件共 6083 行）：门面 `continuous_api.py`（1352 行；`ContinuousMixin:1118`
被 `GenerationMixin` 继承，入口 `generate_batch:1237`）、`scheduler.py:432` / `model_runner.py:313`（每个 decode step
组一个 batch、执行一次 forward）、`cache.py:814` / `cache_manager.py:550`（分页 cache 本体、分配与回收）、
`cb_logits_processors.py:328`（**只在连续批处理下可用的 processor 子集**）、`requests.py:365` / `input_outputs.py:822`，
其余 4 个（卸载 / 分布式 / 初始化 / 工具）共 1078 行。

与主循环的两个直接对照：① **logits processor 是受限子集**——`LogitsProcessor` 上多了类属性
`supports_continuous_batching: bool | None = None`（`logits_process.py:52-54`），例如
`TemperatureLogitsWarper.supports_continuous_batching = True`（`:286`）而 `RepetitionPenaltyLogitsProcessor` 是
`False`（`:356`）——它需要「每个请求自己的历史」，成批处理时 `input_ids` 是 3D 结构（这就是
`logits_process.py:375-401` 那个 3D 分支存在的原因）。② **不支持 `stopping_criteria` / `streamer` /
`assistant_model`**——`generate` 的 0.b 分支里逐个 `raise NotImplementedError`（`utils.py:2411-2433`）。

---

## 一句话总结

> `generate()` = **九步准备** + **一个 while 循环**。准备阶段把 `max_new_tokens` 换算成 `max_length`、把 kwargs
> 合并成 `GenerationConfig`、建好 cache 和两张有序列表（logits processor / stopping criteria）；循环阶段只有六个
> 动作：**取最后位置 logits → 过 processor 列表 → argmax 或 multinomial → 接到 `input_ids` 尾部 → 判停止 → 写回
> cache**。首步（prefill）在循环外用 `next_sequence_length=None` 吃完整 prompt，后续步用 `next_sequence_length=1`
> 只吃 1 个 token —— **这一行三元表达式就是 KV cache 加速的全部秘密**。模型侧唯一的接口是
> `prepare_inputs_for_generation`，而在本仓库里它已上提到 `GenerationMixin`（`utils.py:519`），Llama / Mistral 都没有自己的实现。

---

## 验证清单

以下命令都在仓库根目录 `/Users/baai/Desktop/baai_proj/transformers` 实跑过，
输出已核对（`$ ` 后是命令，紧接着是实际输出）。

```bash
# 1. 入口与文件规模
$ wc -l src/transformers/generation/utils.py src/transformers/generation/configuration_utils.py \
        src/transformers/generation/logits_process.py src/transformers/generation/stopping_criteria.py \
        src/transformers/generation/streamers.py
    4089 src/transformers/generation/utils.py
    1874 src/transformers/generation/configuration_utils.py
    3222 src/transformers/generation/logits_process.py
     643 src/transformers/generation/stopping_criteria.py
     406 src/transformers/generation/streamers.py

# 2. generate / _sample / _beam_search / _assisted_decoding 的定义行
$ grep -n "^    def generate(\|^    def _sample(\|^    def _beam_search(\|^    def _assisted_decoding(\|^    def _prefill(\|^class GenerationMixin" \
        src/transformers/generation/utils.py
359:class GenerationMixin(ContinuousMixin):
2261:    def generate(
2783:    def _sample(
3208:    def _beam_search(
3562:    def _assisted_decoding(
3904:    def _prefill(

# 3. 「首步吃完整 prompt、后续步只吃 1 个 token」的两处真实代码
$ grep -n "next_sequence_length = 1 if model_kwargs\[\"use_cache\"\] else None" \
        src/transformers/generation/utils.py
2878:                    next_sequence_length = 1 if model_kwargs["use_cache"] else None
3367:                next_sequence_length = 1 if model_kwargs["use_cache"] else None
$ sed -n '552p' src/transformers/generation/utils.py
            input_ids = input_ids[:, -next_sequence_length:] if next_sequence_length is not None else input_ids

# 4. Llama / Mistral 都没有自己的 prepare_inputs_for_generation（关键更正）
$ grep -rn "def prepare_inputs_for_generation" src/transformers/models/llama/
（无输出）
$ grep -rn "def prepare_inputs_for_generation" src/transformers/models/mistral/
（无输出）
$ grep -rln "def prepare_inputs_for_generation" src/transformers/models/ | wc -l
75
# 对照：真正的覆盖长什么样
$ grep -n "def prepare_inputs_for_generation" \
    src/transformers/models/gemma3/modeling_gemma3.py \
    src/transformers/models/mllama/modeling_mllama.py \
    src/transformers/models/mamba2/modeling_mamba2.py
src/transformers/models/gemma3/modeling_gemma3.py:1052:    def prepare_inputs_for_generation(self, input_ids, use_cache=True, is_first_iteration=False, **kwargs):
src/transformers/models/mllama/modeling_mllama.py:1520:    def prepare_inputs_for_generation(
src/transformers/models/mamba2/modeling_mamba2.py:838:    def prepare_inputs_for_generation(

# 5. LogitsWarper 基类已被移除（只有一个基类 LogitsProcessor）
$ grep -rn "class LogitsWarper" src/
（无输出）
$ grep -n "^class LogitsProcessor" src/transformers/generation/logits_process.py
49:class LogitsProcessor:

# 6. warper 只在 do_sample=True 时挂载（装配处）
$ grep -n "        if generation_config.do_sample:\|# Processors previously known as" \
        src/transformers/generation/utils.py
1295:        # Processors previously known as `LogitsWarpers`, only applied with sampling strategies
1296:        if generation_config.do_sample:
$ grep -n "processors = self._merge_criteria_processor_list(processors, logits_processor)\|processors.append(TemperatureLogitsWarper" \
        src/transformers/generation/utils.py
1293:        processors = self._merge_criteria_processor_list(processors, logits_processor)
1312:                processors.append(TemperatureLogitsWarper(generation_config.temperature))

# 7. stopping criteria 的 OR 组合语义
$ sed -n '624,626p' src/transformers/generation/stopping_criteria.py
        for criteria in self:
            is_done = is_done | criteria(input_ids, scores, **kwargs)
        return is_done

# 8. cache_position 已从 generate 主流程移除（只剩 remote-code 兼容）
$ grep -c "cache_position" src/transformers/generation/utils.py
6          # 全部集中在 :623-633 这一段
$ grep -n "cache_position" src/transformers/generation/utils.py
623:        # BC for remote code models only: create `cache_position` on the fly here, ...
625:        if self.is_remote_code() and "cache_position" in set(inspect.signature(self.forward).parameters):
627:                "The remote code model you are currently using seems to expect `cache_position`. This arg has been "
629:                "in a future release. Please open a PR on the remote code hub repo to remove any usage of `cache_position`."
632:            cache_position = torch.arange(sequence_length, device=input_ids.device) + past_seen_tokens
633:            model_inputs["cache_position"] = cache_position
$ sed -n '623,633p' src/transformers/generation/utils.py
        # BC for remote code models only: create `cache_position` on the fly here, as we don't want to maintain them in kwargs
        # between `forward`s
        if self.is_remote_code() and "cache_position" in set(inspect.signature(self.forward).parameters):
            logger.warning_once(
                "The remote code model you are currently using seems to expect `cache_position`. This arg has been "
                "removed from the Transformers library, and will stop being created in `generate` even for remote code models "
                "in a future release. Please open a PR on the remote code hub repo to remove any usage of `cache_position`."
            )
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(sequence_length, device=input_ids.device) + past_seen_tokens
            model_inputs["cache_position"] = cache_position
$ grep -n "cache_position" src/transformers/models/llama/modeling_llama.py
（无输出）

# 9. 位置由 get_seq_length() 推出（Llama / Mistral 同款三行）
$ grep -n "past_seen_tokens = past_key_values.get_seq_length()" \
        src/transformers/models/llama/modeling_llama.py \
        src/transformers/models/mistral/modeling_mistral.py
src/transformers/models/llama/modeling_llama.py:387:            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
src/transformers/models/mistral/modeling_mistral.py:360:            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
$ grep -n "def get_seq_length" src/transformers/cache_utils.py
64, 154, 275, 495, 622, 774, 1492, 2028   # 154 = DynamicLayer；1492 = Cache（DynamicCache 继承它）

# 10. contrastive search / DoLa 已迁到 Hub（不是本地函数）
$ grep -n "contrastive-search\|transformers-community/dola" src/transformers/generation/utils.py
145:    GenerationMode.CONTRASTIVE_SEARCH: "transformers-community/contrastive-search",
144:    GenerationMode.DOLA_GENERATION: "transformers-community/dola",
$ grep -c "_contrastive_search" src/transformers/generation/utils.py
0

# 11. 连续批处理是独立一套
$ ls src/transformers/generation/continuous_batching/
__init__.py  cache.py  cache_manager.py  cb_logits_processors.py  continuous_api.py
distributed.py  initialization.py  input_outputs.py  model_runner.py
offloading_manager.py  requests.py  scheduler.py  utils.py
$ wc -l src/transformers/generation/continuous_batching/*.py | tail -1
    6083 total
```

**两条值得单独记下的「与老文档不一致」的发现**：

1. **Llama / Mistral 已经没有 `prepare_inputs_for_generation`**（全库 500+ 模型目录里只剩 75 处定义）。
   通用实现上提到 `GenerationMixin`（`generation/utils.py:519`），模型文件里的是补丁。
2. **`cache_position` 已从库中移除**。源码原话是「This arg has been removed from the Transformers
   library」（`generation/utils.py:628`）。当前由模型 `forward` 通过
   `past_key_values.get_seq_length()` 自行推算位置。

---

## 导航

- 路线图：[00_roadmap.md](./00_roadmap.md)（本篇 = 入门 · 架构 · 1A5）
- 上一篇：[1A4_tokenizer.md](./1A4_tokenizer.md)
- 下一篇：[1A6_trainer.md](./1A6_trainer.md)
- **本篇伏笔的回收**：[2A3_kv_cache.md](./2A3_kv_cache.md)——
  本篇只讲了「循环怎么用 cache」，2A3 讲 `CacheLayerMixin` → `DynamicLayer` / `StaticLayer` /
  `QuantizedLayer` 的内部结构和分页。
- **配套重读**：读完后回看 [1M1_llama_structure.md](./1M1_llama_structure.md) 的
  `LlamaAttention.forward`（`modeling_llama.py:243`），那里 `past_key_values.update(key_states,
  value_states, self.layer_idx)` 这一行就是本篇 `DynamicCache` 在模型内部被写的地方——
  循环在外、写法在内，两边合起来才是完整的 KV cache 故事。
