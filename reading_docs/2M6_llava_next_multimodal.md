# LLaVA-NeXT 多模态精读：视觉编码器 + 语言模型的拼接与投影

> 阅读目标：搞清楚「一个只懂 token 的语言模型，怎么就能看图了」。
>
> 答案只有一句话：**把图片变成一串「视觉 token」，塞进同一个序列，然后走 1M1 那套标准 decoder 流程。**
> 本篇最重要的结论是——**语言模型侧几乎不用改**，全部的改造都集中在「输入端」。
>
> 源码：
> - `src/transformers/models/llava_next/modeling_llava_next.py`（637 行，**无 `modular` 文件，直读本文件**）
> - `src/transformers/models/llava_next/configuration_llava_next.py`（99 行）
> - `src/transformers/models/llava_next/processing_llava_next.py`（179 行）
> - `src/transformers/models/llava_next/image_processing_llava_next.py`（243 行）
>
> 版本：`transformers 5.16.0.dev0`，仓库 HEAD = `42ca97014c`。

## 速读区（TL;DR）

**一句话结论**：多模态 LLM = **一个把图片翻译成向量序列的外挂 + 一个原封不动的语言模型**。
三步：ViT 编码 → MLP 投影到语言隐空间 → `masked_scatter` 把视觉向量**整行替换**进 `inputs_embeds`；
`self.language_model` 就是 `AutoModel.from_config(config.text_config)` 造出的原版 `LlamaModel`，**一行没改**。
「NeXT」相对 1.5 只加了**分辨率**这一件事（AnyRes + `unpad_image` + `image_newline`）。

### 必须记住的断言

| # | 断言 | 出处（文件:行号） |
|---|------|------------------|
| 1 | `modeling_llava_next.py` **是手写源文件（无 `modular`）**，全文只有 **1 处 `# Copied from`** | 第二节；验证 `:1118-1120`（该处在 `:199`） |
| 2 | 那处 `# Copied from` 就是投影器 → **投影器与 LLaVA 1.5 逐行相同**（改名后 `diff` 为空） | `:199-219`；验证 `:1123-1126` |
| 3 | 视觉塔和语言模型都用 `AutoModel.from_config`：`config.vision_config` 默认 `clip_vision_model`、`config.text_config` 是 `llama` | `:255`、`:262`、`configuration_llava_next.py:70`、`modeling_auto.py:89` |
| 4 | 投影器是**两层 MLP + gelu**：`1024 → 4096 → 4096`，默认两层都带 bias | `:200-219`、`configuration_llava_next.py:60`、`:63` |
| 5 | 替换用的是 **`Tensor.masked_scatter`**（不是 `torch.where`、不是布尔索引），按**展平顺序**依次填入 | `:471`；验证 `:1148-1149` |
| 6 | 数量校验：`n_image_tokens * hidden == image_features.numel()`，不匹配抛 `ValueError`；用 `torch_compilable_check` 是为了 TorchDynamo | `:424-427`、`utils/import_utils.py:1810` |
| 7 | `pixel_values` 是 **5 维** `[B, max_patches, 3, 336, 336]`（多出「每图几个 patch」一维），进模型第一件事就是压回 4 维 | `:288`、`:292-294`、`:368-374` |
| 8 | `get_anyres_image_grid_shape` 的第三个参数传的是 **`image_size`(336)** 而不是 `patch_size`(14)——336 是「切图粒度」，14 是「ViT patch 粒度」 | `:294`、`:364`、`:564-567` |
| 9 | `image_newline` 形状是 **`[hidden]`（4096，一个向量，不是矩阵）**，用法是「每行末尾追加一列」 | `:258-259`、`_init_weights` `:237-242` |
| 10 | **loss 与纯文本 CausalLM 完全一样，模型内没有任何把图像位置设成 -100 的代码** | `:616-625`；验证 `:1166-1167`（唯一命中在 `:574` 的 docstring） |
| 11 | **不覆写 `prepare_inputs_for_generation`**（`llava` 与 `llava_next` 都无匹配）；视觉只算一次靠 `MULTIMODAL_INPUTS_TO_DROP_OUTSIDE_PREFILL` 白名单 | `:863-872`、`generation/utils.py:150-165`、`:616` |
| 12 | `image_token_index = 32000`，正好等于 `LlamaConfig.vocab_size` 的默认值——`<image>` 是**追加在词表末尾**的特殊 token | `configuration_llava_next.py:59`、`configuration_llama.py:64` |

### 最容易踩的坑 / 反直觉发现

- **图像 token 位置默认是要算 loss 的**（正文明确说「与直觉相反」）：屏蔽工作由数据侧 collator / 训练脚本写 `labels[labels == IMAGE_TOKEN_INDEX] = -100` 完成；Gemma3 / Idefics3 在模型里就自动忽略了，**LLaVA-NeXT 没有**。
- **processor 算 1525、model 侧要 1526，差 1 是正常现象**：processor 侧那个 `-1` 在补偿 **CLS**，必须和 `num_additional_image_tokens` 配套；该测试只校验 `input_ids`、不跑 model，所以不会失败。
- **`patch_size` / `num_additional_image_tokens` / `vision_feature_select_strategy` 是 processor 与 model 之间的隐式契约**（processor 不持有 config），这也是换 checkpoint 时最容易踩坑的地方。
- **SigLIP 没有 CLS，`vision_feature_select_strategy` 必须用 `"full"`**，否则 `[:, 1:]` 会误删一个真实 patch token；官方 7B/13B checkpoint 仍是 CLIP。
- **`config.image_token_id` 其实叫 `image_token_index`**：靠 `attribute_map`（`{"image_token_id": "image_token_index"}`）别名转发，两种写法取到同一个 32000。
- **`:487` 的 `image_hidden_states` 有个边界 bug**：`:459` 检查了 `size(0) > 0`，而 `:487` 只检查 `is not None`，空 batch 会 `NameError`；该字段只是便利输出，视觉特征早在 `:471` 就用掉了。

### 读完应该能回答的问题

- 一张 `[3, 336, 336]` 的图，经过哪几步变成语言模型能吃的 `[S, 4096]`？每一步的维度变化是多少？
- 一个 `<image>` 占位符为什么要展开成 N 个 `32000`（N 怎么算）？谁负责展开、谁负责校验？
- 逐 token 生成时视觉塔为什么不会被每步重算？这条机制定义在哪个文件的哪个常量里？
- `image_newline` 解决了什么问题？为什么 patch token 数总是 `h' × (w' + 1)`？
- LLaVA-NeXT 相对 1.5 到底新在哪？哪几样东西是逐行没变的？

## 前置知识

- [1M1_llama_structure.md](./1M1_llama_structure.md)：Llama 的七块积木。**本篇的「语言模型侧」就是它，一行没改。**
- [1A1_repo_and_auto.md](./1A1_repo_and_auto.md)：`AutoModel.from_config()` 如何从 config 找到类（第 5 节实战）。
- [1A5_generate_loop.md](./1A5_generate_loop.md)：`prepare_inputs_for_generation` / `model_kwargs`（第 10 节）。
- [2A5_modular_code_generation.md](./2A5_modular_code_generation.md)：**llava_next 没有 modular 文件**（第 2 节）。

## 一、多模态的根本问题

语言模型（Llama）的全部世界观只有 `input_ids`（整数）和 `inputs_embeds`（向量）——它看不到像素，只会把一个 `[B, S, hidden]` 序列在 N 层里滚一遍。所以问题是：**怎么把 `[3, 336, 336]` 的像素张量变成语言模型愿意接受的 `[S, hidden]` 向量序列？**

LLaVA 的答案分三步，每一层都在「降维投影到语言模型的隐空间」：

| 步 | 做什么 | 维度变化（llava-v1.6-mistral-7b 风格） |
|---|--------|----------------------------------------|
| 1 | **视觉编码器**（CLIP ViT）：图片 → patch 特征 | `[3, 336, 336]` → `[576, 1024]`（+1 CLS = 577） |
| 2 | **多模态投影器**（两层 MLP）：视觉隐空间 → 语言隐空间 | `[576, 1024]` → `[576, 4096]` |
| 3 | **占位符替换**：把这 576 个向量写进 `inputs_embeds` 的指定位置 | `[S, 4096]` 里 576 行被换掉 |

**关键洞察：语言模型侧几乎不用改。** `modeling_llava_next.py:262` 的 `self.language_model = AutoModel.from_config(config.text_config)` 拿到的就是原封不动的 `LlamaModel`——attention、RoPE、MLP、RMSNorm 全是原版。所谓「多模态 LLM」，本质是**一个把图像翻译成 token 的外挂 + 一个完全不变的语言模型**；改造的战场在**输入端**，不在模型内部。

## 二、这份源码怎么读：无 modular，且只有一处 `# Copied from`

`ls src/transformers/models/llava_next/`（实测）：`__init__.py` / `configuration_llava_next.py` / `convert_llava_next_weights_to_hf.py` / `image_processing_llava_next.py` / `image_processing_pil_llava_next.py` / `modeling_llava_next.py` / `processing_llava_next.py`。**没有 `modular_llava_next.py`**，所以按 2A5：它是**手写源文件**，可直接读、直接改，也**可以**用 `# Copied from`（互斥只对 modular 生成的文件成立）。实测 `grep -c "Copied from" modeling_llava_next.py` → **1**：

```python
# modeling_llava_next.py:199
# Copied from transformers.models.llava.modeling_llava.LlavaMultiModalProjector with Llava->LlavaNext
class LlavaNextMultiModalProjector(nn.Module):
```

即**投影器 = LLaVA 1.5 的投影器，逐行相同**（把 `modeling_llava.py:87-106` 抠出来做 `s/Llava/LlavaNext/g` 与 `:200-219` `diff` → 完全一致）——投影器这个「翻译层」在 1.5 → NeXT 之间没被改动，改动全在别处（第十二节）。同目录族的 `llava_next_video` / `llava_onevision` **有** modular 文件——**同一模型族里两种机制可以并存**（路线图 `00_roadmap.md:184`）。

文件内的类/函数清单（`grep -n` 实测）：`41 get_anyres_image_grid_shape`、`72 image_size_to_num_patches`、`109 unpad_image`、`154 LlavaNextModelOutputWithPast`、`175 LlavaNextCausalLMOutputWithPast`、`200 LlavaNextMultiModalProjector`、`223 LlavaNextPreTrainedModel`、`250 LlavaNextModel`（方法 `265 pack_image_features`、`337 get_image_features`、`407 get_placeholder_mask`、`433 forward`）、`496 LlavaNextForConditionalGeneration`（`499 __init__`、`505 get_output_embeddings`、`508 pack_image_features`【纯委托】、`519 get_image_features`【纯委托】、`551 forward`）。只有**四个**类值得读，其余是 output dataclass 和三个 helper。

## 三、全链路数据流（主图）

以 llava-v1.6-mistral-7b 风格的 config 默认值为例（`configuration_llava_next.py:59-66` + `:73-82`：vision `hidden=1024 / patch=14 / image=336 / 24 层`，text `LlamaConfig` → `hidden=4096`）。

```
   图片（PIL / ndarray / tensor）        ← 原始像素，尺寸任意
        │ ① LlavaNextImageProcessor._preprocess   image_processing_llava_next.py:161
        │    _get_image_patches:336  :114  select_best_resolution 选网格（AnyRes）→ resize → pad → divide_to_patches(336)
        │    原图再单独缩到 336×336 作「base patch」放在最前
        ▼  pixel_values list[B] of (num_patches_i, 3, 336, 336)，num_patches_i = 1 + N_h·N_w
           → _pad_for_batching → [B, max_patches, 3, 336, 336]（5 维！）；image_sizes [B, 2] = 每图原始 (H₀, W₀)
        │ ② get_image_features   modeling_llava_next.py:337   5 维 → 4 维（按 image_num_patches 切片后 concat）
        ▼  self.vision_tower = AutoModel.from_config(config.vision_config)   :255
        │    默认 model_type = "clip_vision_model" → CLIPVisionModel，24 层；576 patch + 1 CLS = 577
        │    hidden_states = tuple(25) of [Σp, 577, 1024]（强制 output_hidden_states=True）；
        │    vision_feature_layer = -2 取倒数第二层；strategy == "default" → 丢 CLS：[:, 1:]（:390-391）→ [Σp, 576, 1024]
        │ ③ LlavaNextMultiModalProjector   :200   linear_1(1024→4096) → gelu → linear_2(4096→4096)
        ▼  [Σp, 576, 4096] ← 已落在语言模型的隐空间里；torch.split(image_num_patches) → list[B]
        │ ④ pack_image_features   :265   unpad（:310）+ 每行插 image_newline（:311-320）+ 拼回 base（:322）
        ▼  image_features list[B] of [L_i, 4096]（本例每图 L = 1526，算法见第七节）；torch.cat(dim=0) → [ΣL, 4096]（:467）
        │ ⑤ get_placeholder_mask   :407   mask = (input_ids == 32000)（:420），校验 ΣL == 占位符个数（:424）
        │ ⑥ inputs_embeds.masked_scatter(mask, image_features)   :471
        ▼  inputs_embeds [B, S, 4096] ← 视觉向量已「长」在文本序列里，语言模型看不出来
        │ ⑦ self.language_model = AutoModel.from_config(config.text_config)   :262
        │    = 原版 LlamaModel：embed → N × DecoderLayer(attn+MLP) → RMSNorm
        ▼  last_hidden_state [B, S, 4096] → ⑧ self.lm_head :502 → logits [B, S, text_config.vocab_size]
```

**对照 1M1 的数据流**——从第 ⑦ 步往后，一字不差：

```
1M1:  input_ids → embed_tokens → [cos/sin + mask] → N×DecoderLayer → norm → lm_head → logits
本篇: input_ids → embed_tokens → ★被 masked_scatter 换掉几行★ → N×DecoderLayer → norm → lm_head → logits
                                 └──────── 唯一的区别就在这个星号 ────────┘
```

## 四、输入端：图片怎么变成 `pixel_values`（AnyRes）

**动机**：LLaVA 1.5 把每张图**硬缩放到 336×336**，细节全丢。NeXT 的「NeXT」第一件事就是 **AnyRes**：从预设**分辨率菜单**里挑一个和原图长宽比最接近的网格，resize 到该尺寸、切成若干张 336×336 patch **分别编码**，再额外编码一张整图缩到 336×336 的「**base patch**」提供全局信息。菜单是 `image_grid_pinpoints`（`configuration_llava_next.py:90-94` 真实默认值 `[[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]`）。配 `patch_size=14`、`image_size=336`（即 24×24 token/patch）：`[336,672]`/`[672,336]` → 1×2 / 2×1 网格（2×576 token）；`[672,672]` → 2×2（4×576）；`[1008,336]`/`[336,1008]` → 3×1 / 1×3（3×576）。最坏情况 ≈ 5 × 576 ≈ 2880 个视觉 token，是 1.5 的 5 倍——**这就是「NeXT」用 token 换分辨率的地方**。

### 4.2 真实实现：`_get_image_patches`

```python
# image_processing_llava_next.py:114-146（节选，注释为笔者所加）
best_resolution = select_best_resolution(image_size, possible_resolutions)   # ← AnyRes 的「选」
resized_image  = self._resize_for_patching(image, best_resolution, ...)
padded_image   = self._pad_for_patching(resized_image, best_resolution)      # 补到正好整除
patches        = divide_to_patches(padded_image, patch_size=patch_size)      # ← AnyRes 的「切」
return [self.resize(image=image, size=SizeDict(height=size_height, width=size_width), ...)] + patches  # base 在最前
```

三个要点：
1. **`select_best_resolution`**（`image_processing_utils.py:633-668`）的判据是「最大化有效像素、最小化浪费像素」：对每个候选算 `scale = min(w/W, h/H)`，比较 `effective`（缩小后面积）与 `wasted`。**它同时被 processor 和 modeling 调用**（见第 7 节的算术），所以两侧必然选到同一个网格。
2. `image_patches = [base] + patches`：**「整图缩略 patch」在前，切图 patch 在后**，该顺序在 `pack_image_features` 里被硬编码依赖（第七节）。
3. `divide_to_patches`（`image_transforms.py:839-862`）按**行优先**切，patch 排列天然是 `(N_h, N_w)` 的展平——这是第七节 `view(num_patch_height, num_patch_width, ...)` 能成立的前提。

### 4.3 后处理与输出的张量形状

`rescale_and_normalize`（`image_processing_llava_next.py:226-240`）做 CLIP mean/std 归一化；`_pad_for_batching`（`:148-159`）用 `F.pad(...)` 在 **patch 维度**补零，让 batch 里 patch 数不同的图片拼成规整张量：

```
pixel_values : [B, max_patches, 3, 336, 336]     ← 注意是 5 维
image_sizes  : [B, 2]，每项是原始 (H₀, W₀)
```

> **5 维 `pixel_values` 是一个坑**：普通 ViT 是 4 维 `[B, C, H, W]`，这里多了一维「每图几个 patch」，所以 `get_image_features` 一进来第一件事就是把它压回 4 维（`modeling_llava_next.py:368-374`）。

## 五、视觉编码器：`AutoModel.from_config` 的实战（1A1 呼应）

**实测 config 默认值**（`configuration_llava_next.py:70` 写死了后备值 `"clip_vision_model"`；`:73-82` 的默认实例化参数为 `intermediate_size=4096 / hidden_size=1024 / patch_size=14 / image_size=336 / num_hidden_layers=24 / num_attention_heads=16 / projection_dim=768`）→ **默认是 CLIP-ViT-L/14-336**，即 `CLIPVisionModel`。`vision_feature_layer` 语义：CLIP 24 层 → `hidden_states` 长 25，`-2` = 倒数第二层输出。它是「可插拔」的，靠 `AutoModel.from_config`（`modeling_llava_next.py:253-263`）：

```python
def __init__(self, config: LlavaNextConfig):
    super().__init__(config)
    self.vision_tower = AutoModel.from_config(config.vision_config)      # ← 关键
    self.multi_modal_projector = LlavaNextMultiModalProjector(config)
    self.image_newline = nn.Parameter(...)                               # :258-259 见第七节
    self.vocab_size = config.text_config.vocab_size
    self.language_model = AutoModel.from_config(config.text_config)      # ← 关键
    self.post_init()
```

| 位置 | 传入 config | 解析结果 | 机制位置 |
|---|---|---|---|
| `:255` | `config.vision_config`（`"clip_vision_model"`） | `CLIPVisionModel` | `modeling_auto.py:89` |
| `:262` | `config.text_config`（`"llama"`） | `LlamaModel` | `AutoModel` 映射 |

`AutoModel.from_config`（`auto_factory.py:206`）最终走到 `auto_factory.py:250` 的 `return model_class._from_config(config, **kwargs)`——即 1A1 的「config → 类 → `_from_config`」三步曲。这里多了一层：`config.sub_configs`（`configuration_llava_next.py:55`）让两个子 config 在 `__post_init__`（`:68-88`）就被 resolve 成真正的 config 对象。**好处**：换视觉编码器**不需要改一行代码**，只要 `config.json` 里写不同的 `vision_config.model_type`。真实证据（`tests/models/llava_next/test_modeling_llava_next.py:334-338` 的 `test_granite_vision`）：SigLIP 主干（无 CLS）的 `ibm-granite/granite-vision-3.1-2b-preview` 被 `LlavaNextForConditionalGeneration` 直接加载。准确说法：**NeXT 把视觉编码器从「写死的 CLIP」改成了「由 `vision_config` 决定的任意 ViT」**——官方 7B/13B 仍是 CLIP。（roadmap 原话 `docs/source/en/model_doc/llava_next.md:27` 重点也在分辨率。）

### 5.3 视觉编码器的三种配置旋钮

```python
# modeling_llava_next.py:376-393
image_outputs = self.vision_tower(pixel_values, output_hidden_states=True, return_dict=True, **kwargs)  # 强制打开
if isinstance(vision_feature_layer, int):
    selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
else:
    hs_pool = [image_outputs.hidden_states[k] for k in vision_feature_layer]
    selected_image_feature = torch.cat(hs_pool, dim=-1)
if vision_feature_select_strategy == "default":
    selected_image_feature = selected_image_feature[:, 1:]
```

| 旋钮 | config 字段 | 默认 | 作用 |
|---|---|---|---|
| 取哪一层的特征 | `vision_feature_layer` | `-2` | 可 `int` 或 `list[int]`（多层在**最后一维** concat） |
| 要不要丢掉 CLS | `vision_feature_select_strategy` | `"default"` | `"default"` = `[:, 1:]` 去 CLS；`"full"` = 全留着 |
| 输入尺寸 | `vision_config.image_size` / `patch_size` | `336` / `14` | 每 patch 出 `(336/14)² = 576` 个 token |

list 时 `torch.cat(dim=-1)` **在特征维拼接**，所以投影器输入维度必须乘上层数（下一节 `num_feature_layers` 的来历）。SigLIP **没有 CLS**，用 `"default"` 会误删一个真实 patch token，必须用 `"full"`——两侧必须和 checkpoint 对齐。

## 六、投影器 `LlavaNextMultiModalProjector`：真实结构

**实测结论：两层 MLP，中间 `gelu`，两层的 bias 由 config 控制**（19 行，与 LLaVA 1.5 逐行相同）：

```python
# modeling_llava_next.py:199-219
# Copied from transformers.models.llava.modeling_llava.LlavaMultiModalProjector with Llava->LlavaNext
class LlavaNextMultiModalProjector(nn.Module):
    def __init__(self, config: LlavaNextConfig):
        super().__init__()
        num_feature_layers = 1 if isinstance(config.vision_feature_layer, int) else len(config.vision_feature_layer)
        self.linear_1 = nn.Linear(config.vision_config.hidden_size * num_feature_layers,
                                  config.text_config.hidden_size, bias=config.multimodal_projector_bias)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size,
                                  bias=config.multimodal_projector_bias)

    def forward(self, image_features):
        return self.linear_2(self.act(self.linear_1(image_features)))
```

| 项 | 真实值 | 行号 |
|---|---|---|
| 形状 | **两层 MLP** | `:205` / `:211` |
| `linear_1` | `1024 → 4096` | `:205-209` |
| 激活 | `ACT2FN["gelu"]` → **GELU** | `:210` + config `:60` |
| `linear_2` | `4096 → 4096` | `:211-213` |
| bias | `multimodal_projector_bias = True` → 两层都带 | `:63` |

> 中间层不是「窄→宽→窄」而是「同宽→同宽」：第一层已把视觉维度撑到语言维度，第二层只在语言空间做一次非线性变换——本质是一个**对齐器**。若 `vision_feature_layer=[-1,-2,-3]`，特征最后一维是 `1024×3=3072`，投影器输入必须同步变 3072：唯一的同步方式就是**读同一个 config 字段**。

投影器**从不单独使用**，只在 `get_image_features` 里被调一次（`modeling_llava_next.py:393`）：`vision_tower → 选层(hidden_states[-2]) → 去 CLS([:, 1:]) → projector(linear_1→gelu→linear_2)`。

## 七、`pack_image_features`：AnyRes 的缝合、`unpad` 与 `image_newline`

投影完后每图有 `num_patches_i` 个 patch（patch 0 = base，patch 1..N = 拼图网格），**直接拼起来不能用**，原因有二：① `best_resolution` ≠ 原图长宽比，`_resize_for_patching` + `_pad_for_patching` 补了黑边，必须**裁掉**；② 网格是二维的、序列是**一维**的，直接拉平会让「同一行」的边界消失——所以每行末尾要插一个**换行标记** `image_newline`。

**`image_newline` 是可学习参数**（`modeling_llava_next.py:258-259`，`_init_weights` `:237-242` 里 `init.normal_(... , std=embed_std)`）：

```python
embed_std = 1 / math.sqrt(config.text_config.hidden_size)
self.image_newline = nn.Parameter(torch.randn(config.text_config.hidden_size, dtype=self.dtype) * embed_std)
```

**形状是 `[hidden]`（4096，一个向量，不是矩阵）**，用法是「每行末尾追加一列」——纯空间标记，不含内容。

### 7.2 缝合的真实实现

```python
# modeling_llava_next.py:283-330（节选，注释为笔者所加）
for image_idx, image_feature in enumerate(image_features):
    if image_feature.shape[0] > 1:                       # 有拼图网格
        base_image_feature = image_feature[0]            # ← 第 0 个是 base patch，单独拿出来
        image_feature = image_feature[1:]
        height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size   # 336//14 = 24
        num_patch_height, num_patch_width = get_anyres_image_grid_shape(
            image_sizes[image_idx], self.config.image_grid_pinpoints,
            self.config.vision_config.image_size,        # ← 注意：patch_size 参数传的是 336，不是 14
        )
        image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)   # [N_h, N_w, 24, 24, D]
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()                          # [D, N_h, 24, N_w, 24]
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)                                  # [D, N_h*24, N_w*24]
        image_feature = unpad_image(image_feature, image_sizes[image_idx])                          # 裁掉黑边
        if image_newline is not None:                                                              # 每行末尾追加一列 → [D, h', w'+1]
            image_feature = torch.cat((image_feature, image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(...)), dim=-1)
        image_feature = image_feature.flatten(1, 2).transpose(0, 1)                                 # [h'*(w'+1), D]
        image_feature = torch.cat((base_image_feature, image_feature), dim=0)                       # base 在前
    else:                                                                                           # 只有 base：直接补一个换行
        image_feature = image_feature[0] if image_newline is None else torch.cat(
            (image_feature[0], image_newline[None].to(image_feature[0])), dim=0)
    new_image_features.append(image_feature); feature_lens.append(image_feature.size(0))
feature_lens = torch.tensor(feature_lens, dtype=torch.long, device=image_features[0].device)
return new_image_features, feature_lens
```

**这段「四维 → 一维」的魔术**（设 `N_h=2, N_w=1, h=w=24, D=4096`）：`[2,576,4096]` —`view`/`permute`(4,0,2,1,3)/`flatten`→ `[4096,48,24]`（一张 48×24 的 token 图）—`unpad_image`→ `[4096,38,24]` —`cat(换行)`→ `[4096,38,25]` —`flatten`+`transpose`→ `[950,4096]` —拼 base→ `[576+950, 4096] = [1526, 4096]`。

`unpad_image`（`modeling_llava_next.py:109-145`）的判据是**长宽比**：原图更「宽」则说明 padding 加在上下、裁行，否则裁列——正是 processor 端 `_get_unpadded_features` 的对偶计算。

> **`get_anyres_image_grid_shape` 的第三个参数名叫 `patch_size`，传进去的却是 `vision_config.image_size`（336）**（`modeling_llava_next.py:294`）。不是笔误：每个 patch 已被 resize 成 336×336 再编码，所以「网格」粒度是 336，不是 14。同理 `image_size_to_num_patches(..., patch_size=self.config.vision_config.image_size)`（`:364`）。**记住：336 是「切图粒度」，14 是「ViT patch 粒度」。**

### 7.3 一个真实数字的例子（把 1525 算出来）

`tests/models/llava_next/test_processing_llava_next.py:80-107` 的真实参数：图 `(3, 503, 316)`；`patch_size=14`；`image_grid_pinpoints=[[672, 336]]`；`strategy="default"`；`expected_image_tokens = 1525`。processor 侧（`processing_llava_next.py:92-132`）逐步算：

```
select_best_resolution((503,316), [[672,336]])        → (672, 336)
scale_height, scale_width = 672//336, 336//336        → (2, 1)
patches_height = 336//14, patches_width = 336//14     → (24, 24)
_get_unpadded_features(503, 316, 24, 24, 2, 1):
    current_height = 48 ; current_width = 24
    original_aspect = 316/503 = 0.628 > current_aspect = 24/48 = 0.5   → 原图更宽，裁行
    new_height = round(503 * 24/316) = 38 ; padding = (48-38)//2 = 5   → current_height = 38
    unpadded_features = 38*24 = 912 ; newline_features = 38
base_features = 24*24 + num_additional_image_tokens = 576 + 0 = 576
num_image_tokens = 912 + 38 + 576 = 1526 → strategy=="default" 再 -1 → 1525
```

model 侧按 7.2 的表格：`576 (base) + 38×(24+1) = 576 + 950 = 1526`。

> **差 1 是正常现象**：processor 侧那个 `-1`（`processing_llava_next.py:88-89`）在补偿 **CLS**，必须和 `num_additional_image_tokens` 配套。`docs/source/en/model_doc/llava_next.md:158`：「`num_additional_image_tokens` should be `1` if the vision backbone adds a `CLS` token or `0` if nothing extra is added.」CLIP 有 CLS，真实 checkpoint 的 processor 里这个值是 `1`（对照 `tests/models/llava/test_processing_llava.py:39`，显式写了 `1`）。而 llava_next 这个测试用 tiny processor 默认值 `0`（`processing_llava_next.py:55`），于是 processor 算 1525、model 要 1526——**差 1**；测试只校验 `input_ids`、不跑 model，所以不会失败。
>
> **实操结论**：`patch_size` / `num_additional_image_tokens` / `vision_feature_select_strategy` 是 processor 与 model 之间的**隐式契约**，错一个就是第 8 节那个报错。换 checkpoint 时最容易踩坑的是 processor 而不是 modeling。

## 八、占位符替换：`get_placeholder_mask` + `masked_scatter`

文本里的 `<image>` 只是**占位标记**，必须先在文本侧展开成 N 个 `image_token_id`，再在向量侧被视觉特征**整行替换**。

### 8.1 文本侧：`<image>` → 1525 个 `32000`

`image_token_index` 默认 **32000**（`configuration_llava_next.py:59`），`LlamaConfig.vocab_size` 默认也是 **32000**（`models/llama/configuration_llama.py:64`）——`<image>` 正好是**追加在词表末尾**的特殊 token。processor 侧展开（`processing_llava_next.py:81-90`）：

```python
def replace_image_token(self, image_inputs, image_idx, **kwargs) -> str:
    num_image_tokens = self._get_number_of_features(orig_height, orig_width, height, width)
    if self.vision_feature_select_strategy == "default":
        num_image_tokens -= 1
    return self.image_token * num_image_tokens        # "<image>" * 1525
```

一句话里 1 次 `<image>` 被字符串复制成 1525 次再交给 tokenizer（`ProcessorMixin` 通用机制：`processing_utils.py:769-779` 调 `self.replace_image_token`，`:810-811` 只是 `raise NotImplementedError` 的接口）。

### 8.2 模型侧：先做检查，再替换

```python
# modeling_llava_next.py:407-428  （完整函数，注释为笔者所加）
def get_placeholder_mask(self, input_ids, inputs_embeds, image_features):
    """... checks that the placeholder token count is equal to the length of multimodal features. ..."""
    if input_ids is None:      # 兜底：用 embed_tokens(32000) 的结果逐元素比对反推占位符位置
        special_image_mask = (inputs_embeds == self.get_input_embeddings()(
            torch.full((), self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
        )).all(-1)
    else:
        special_image_mask = input_ids == self.config.image_token_id        # :420
    n_image_tokens = special_image_mask.sum()                              # :422
    special_image_mask = special_image_mask.unsqueeze(-1).to(inputs_embeds.device)   # :423 → [B, S, 1]
    torch_compilable_check(                                                # :424-427
        n_image_tokens * inputs_embeds.shape[-1] == image_features.numel(),
        f"Image features and image tokens do not match, tokens: {n_image_tokens}, features: {image_features.shape[0]}",
    )
    return special_image_mask
```

四件事：
1. **`self.config.image_token_id`** 其实叫 `image_token_index`，靠 config 的 `attribute_map`（`configuration_llava_next.py:54`，`{"image_token_id": "image_token_index"}`）经 `PreTrainedConfig.__getattribute__`（`configuration_utils.py:481-483`）别名转发——两种写法取到同一个 32000。
2. **`input_ids is None` 的兜底**（`:414-418`）：用 `embed_tokens(32000)` 拿 `<image>` 的 embedding，再逐元素相等比较 `.all(-1)` 反推哪些位置是图像占位符。
3. **形状 `[B, S] → [B, S, 1]`**（`:423`），为了广播到 `[B, S, hidden]`；`.to(device)` 只搬设备、不转 dtype。
4. **那个检查**（`:424-427`）：`n_image_tokens * hidden == image_features.numel()`。因为 `masked_scatter` 在**广播后**的 mask 上按顺序取 `source` 元素，广播后 mask 有 `n_image_tokens × hidden` 个 `True`，`source` 必须**恰好**有这么多元素。**数量不匹配会抛 `ValueError`**，形如 `Image features and image tokens do not match, tokens: 1525, features: 1526`。用 `torch_compilable_check`（`utils/import_utils.py:1810`）而不是 `if ... : raise`，是为了给 TorchDynamo 真值提示，避免 `torch.compile` 在数据相关的分支上编译失败。

### 8.3 真正的替换：`masked_scatter`

```python
# modeling_llava_next.py:456-471
if inputs_embeds is None:
    inputs_embeds = self.get_input_embeddings()(input_ids)          # :457  普通查表
if pixel_values is not None and pixel_values.size(0) > 0:
    image_features = self.get_image_features(...).pooler_output     # :460-466
    image_features = torch.cat(image_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)   # :467
    special_image_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds,
                                                   image_features=image_features)                     # :468-470
    inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)   # :471  ★
```

**用的是 `Tensor.masked_scatter`（不是 `torch.where`、不是布尔索引赋值）**：mask 为 True 的位置按**行优先展平顺序**依次填入 source。因为 `image_features` 和 `input_ids` 里图像 token 的顺序严格对应（图 0 → 图 1…，每图内部「base patch → 网格 patch 逐行」），顺序填入恰好正确。

> `get_input_embeddings()` 在 `LlavaNextModel` 上能直接工作，靠 `PreTrainedModel.get_input_embeddings` 的兜底规则——`modeling_utils.py:1102-1109` 的注释点名了这种布局："Multimodal LMs that wrap a text sub-model at `self.language_model`"。

### 8.4 序列拼接示意（核心产出之一）

```
① 文本侧（processor 输出）：<image> 被展开成 N 个 image_token_id（N = 1525）
   input_ids(B=1) = [ BOS │ [INST] │ 32000 ×1525（一个 "<image>" 占位符被字符串复制 1525 次）│ 13 │ 1056 │ … ]
② 向量侧（embed_tokens 之后）：inputs_embeds [1, S, 4096]，那 1525 行 e(32000) 是「无用」的占位 embedding
③ masked_scatter 之后：这 1525 行被整行替换为视觉特征（image_features [1526, 4096]）
   = [ BOS │ [INST] │ v₀ v₁ v₂ … v₁₅₂₅（1526 行视觉向量）│ e(\n) │ e(What) │ … ]
     └ 前 576 行 = base patch（整图缩略）；中间每 25 行 = 网格一行（24 patch token + 1 个 image_newline）
④ 交给语言模型：self.language_model(inputs_embeds=..., attention_mask=..., position_ids=..., past_key_values=...)
   ↑ 参数和 1M1 的 LlamaModel.forward 一一对应，一个都不多
```

视觉信息在**序列长度方向**和文本拼接，在**通道方向**已对齐（都投影到 4096）。语言模型收到的依然是 `[B, S, hidden]`，它对图像一无所知。

### 8.5 一处值得留意的边界情况

`image_features` 只在 `:460` 那个 `if pixel_values is not None and pixel_values.size(0) > 0:` 块里定义；`:487` 的 `image_hidden_states=image_features if pixel_values is not None else None`（`:482-488`）只检查 `is not None`，而 `:459` 检查了 `size(0) > 0`。若空 batch 传了 `pixel_values`，`:487` 会引用未定义变量 → `NameError`。正常路径不触发；`image_hidden_states` 只是便利字段，视觉特征早在 `:471` 用掉了。

## 九、语言模型侧：几乎零改动 + loss

```python
# modeling_llava_next.py:473-480
outputs = self.language_model(
    attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values,
    inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs,
)
```

对照 1M1 的 `LlamaModel.forward`：**参数名一个不差**。没有额外位置偏移（图像 token 也是普通位置）、没有特殊 attention mask（图像 token 和文本 token 互相可见）、没有跨模态 attention 层。顶层 `LlavaNextForConditionalGeneration`（`:496-503`）也只做两件事，且与 `LlamaForCausalLM` 只差两处：`self.model` 是 `LlavaNextModel` 而非 `LlamaModel`；`lm_head` 输入维度取 `config.text_config.hidden_size`（`LlavaNextConfig` 自己没有 `hidden_size`，文本维度藏在子 config 里——这就是 `get_text_config()` 那套机制存在的原因）：

```python
class LlavaNextForConditionalGeneration(LlavaNextPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    def __init__(self, config: LlavaNextConfig):
        super().__init__(config)
        self.model = LlavaNextModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()
```

`tie_word_embeddings` 默认值是 **`False`**（`configuration_llava_next.py:64`），只有为 True 时 `tie_weights` 才生效（`modeling_utils.py:2559-2560`）。

### 9.2 loss：和纯文本 CausalLM **完全一样**

```python
# modeling_llava_next.py:616-625
hidden_states = outputs[0]
slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
logits = self.lm_head(hidden_states[:, slice_indices, :])
loss = None
if labels is not None:
    loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs)
```

**模型内部没有任何「把图像位置设为 -100」的代码。** 实测 `grep -rn "\-100\|IGNORE_INDEX\|ignore_index" src/transformers/models/llava_next/` 只有一处 docstring（`modeling_llava_next.py:574`）。`loss_function` 是 `PreTrainedModel` 上的属性（`modeling_utils.py:4708-4720`），默认 `ForCausalLMLoss`，内部就是标准 `cross_entropy(..., ignore_index=-100)`。

> **结论（与直觉相反）**：LLaVA-NeXT 的**图像 token 位置默认是要算 loss 的**。屏蔽由数据侧 collator / 训练脚本负责（经典写法 `labels[labels == IMAGE_TOKEN_INDEX] = -100`，在训练脚本里，不在 modeling 里）。Gemma3 / Idefics3 在模型里就做了自动忽略，**LLaVA-NeXT 没有**。读多模态模型时要专门确认这一点，它决定了「图像 token 会不会污染 loss」。

## 十、生成：图片只算一次

实测 `grep -n "prepare_inputs_for_generation" modeling_llava_next.py modeling_llava.py` → **无任何匹配**。保证「不每步重算视觉」的是 `GenerationMixin` 的**通用实现**：

```python
# generation/utils.py:150-165
MULTIMODAL_INPUTS_TO_DROP_OUTSIDE_PREFILL = (
    "pixel_values", "pixel_mask", "input_features", "input_features_mask", "pixel_values_videos",
    "num_local_patches", "high_res_pixel_values", "image_patches_indices", "image_patches",
    "image_sizes", "image_sizes_videos", "pixel_attention_mask", "pixel_values_images", "num_local_patches")
# generation/utils.py:608-621（节选）
for key, value in kwargs.items():
    if key in ("labels", "next_sequence_length"):                        # kwargs_to_avoid_forwarding
        continue
    # 只在 prefill（或新 batch / 无 cache）时才转发
    elif key in MULTIMODAL_INPUTS_TO_DROP_OUTSIDE_PREFILL and (not is_first_iteration and kwargs.get("use_cache", True)):
        continue
    elif key not in model_inputs:
        model_inputs[key] = value
```

`is_first_iteration` 的生命周期：`generation/utils.py:519-528`（通用 `prepare_inputs_for_generation(..., is_first_iteration: bool | None = False)`）；`:3941-3948`（`_sample` 主循环调用点）；`:3861`（每轮末尾置 False）。

```
step 0（prefill）：is_first_iteration=True
    → pixel_values / image_sizes 照常传给 forward → LlavaNextModel.forward:471 做 masked_scatter，KV cache 建好
step 1..N（decode）：is_first_iteration=False 且 use_cache=True
    → pixel_values / image_sizes 被 continue 掉，根本不进 model_inputs
    → LlavaNextModel.forward:459 的 `if pixel_values is not None` 为 False → 视觉塔完全不被调用
    → input_ids 只有上一轮新生成的那个 token，走 :457 的普通查表
```

**这不是可选的优化，而是多模态生成能跑起来的前提**：视觉塔是 24 层 ViT，一张图的 anyres 切成 3~6 个 patch，每个 patch 送进去是 `[3, 336, 336]`、输出 577 个向量——一次前向的开销和「一段 1000~3000 token 的语言模型 prefill」同量级；若每步 decode 都重算，生成 100 个 token 就要跑 100 次 ViT，慢到不可用。

> 对照 1A5：`pixel_values` 是「**在 `model_kwargs` 里一直存在，但每步前向时被丢掉**」。排查多模态生成问题时，先确认你传的 key 名在这个白名单里。

### 10.3 多图 / 不同分辨率的一个副作用

`docs/source/en/model_doc/llava_next.md:152`：LLaVA-NeXT 对不同图用不同 patch 数并在 modeling code 里 pad，默认 `eval()` 模式下是 **left-padding**，否则 right-padding。因为 `masked_scatter` 按**展平顺序**填充，batch 内的 token 顺序必须和 `image_features` 的拼接顺序严格一致。`tests/models/llava_next/test_modeling_llava_next.py:285` 的 `test_small_model_integration_test_batch_matches_single` 就是在守这条不变量。

## 十一、Processor：把 image_processor + tokenizer 串起来

`LlavaNextProcessor`（`processing_llava_next.py:44-79`）很薄，职责只有一个：**在文本进 tokenizer 之前，把 `<image>` 换成正确数量的占位符**。

```python
# processing_llava_next.py:47-79（节选）
def __init__(self, image_processor=None, tokenizer=None, patch_size=None,
             vision_feature_select_strategy=None, chat_template=None,
             image_token="<image>", num_additional_image_tokens=0, **kwargs):
    self.patch_size = patch_size
    self.num_additional_image_tokens = num_additional_image_tokens
    self.vision_feature_select_strategy = vision_feature_select_strategy
    self.image_token = tokenizer.image_token if hasattr(tokenizer, "image_token") else image_token
    self.image_token_id = (tokenizer.image_token_id if getattr(tokenizer, "image_token_id", None)
                           else tokenizer.convert_tokens_to_ids(self.image_token))
    super().__init__(image_processor, tokenizer, chat_template=chat_template)
```

`__call__(text=..., images=...)` 的真实流程（`ProcessorMixin.__call__`，`processing_utils.py:656-727`）：

```
① validate_inputs  :667      ② _merge_kwargs  :669（合并 text_kwargs / images_kwargs）
③ _process_images(images)  :678 → :769
      ├─ self.image_processor(images, do_pad=True) → pixel_values(5 维) + image_sizes
      └─ for idx: self.replace_image_token(processed_images, image_idx=idx)  :777
④ get_text_with_replacements(text, ...)  :692 → :819  扫描文本里的 <image>，按出现顺序替换
⑤ self.tokenizer(text, **text_kwargs)  :698 → input_ids（此时已含 N 个 32000）
⑥ 合并 data = {**text_inputs, **processed_images, ...}  :708      ⑦ return BatchFeature(...)  :714
```

几处细节：**`images_kwargs` 默认 `do_pad: True`**（`processing_llava_next.py:37-39`）——5 维 `pixel_values` 的来源；**替换顺序敏感**，第 i 个 `<image>` 对应第 i 张图（`processing_utils.py:834-835`）；**`_get_num_multimodal_tokens`**（`processing_llava_next.py:134-176`）是给推理引擎（vLLM 等）的「先问数量、再占位」接口，返回 `MultiModalData(num_image_tokens=[...], num_image_patches=[...])`，`num_image_patches` 恒为 `[1]*len(image_sizes)`（`:165`）；**`patch_size` 必须显式配置**——processor 不持有 config，拿不到 vision_config，所以 7.3 那份隐式契约才会存在。

## 十二、增量与改动清单：LLaVA 1.5 → NeXT、纯文本 → 多模态

| 组件 / 维度 | 基线（LLaVA 1.5 / 纯文本 Llama） | LLaVA-NeXT | 状态 | 关键代码 |
|---|---|---|---|---|
| **分辨率** | 固定 336×336，一图一 patch | **AnyRes**：从 `image_grid_pinpoints` 选网格，切成多张 336×336 分别编码 | **改动** | `image_processing_llava_next.py:114`、`modeling:41` / `:72` |
| **extra 输入** | 无 | **`image_sizes`**（original (H₀,W₀)）成为 forward 的必填参数 | **改动** | `modeling:437` |
| **pixel_values 形状** | 4 维 `[B, C, H, W]` | **5 维** `[B, num_patches, C, 336, 336]` | **改动** | `modeling:368-374` |
| **patch 缝合** | 无 | **`pack_image_features`**：`unpad_image` 裁黑边 + 逐行插 `image_newline` + 拼 base patch | **全新增** | `modeling:265-330` |
| **行分隔符** | 无 | **`image_newline`** 可学习参数（`[hidden]`） | **全新增** | `modeling:259`、`:311-320` |
| **CLS 处理** | `[:, 1:]`（硬编码去 CLS） | 同样 `[:, 1:]`，但由 `vision_feature_select_strategy` 控制 | **改动** | `modeling:390-391` |
| **视觉编码器** | `AutoModel.from_config`（默认 CLIP） | **同样**（默认 CLIP，可换 SigLIP） | **复用** | `modeling:255` |
| **多模态投影器** | 1.5：两层 MLP + gelu | **逐行相同**（`# Copied from`） | **复用** | `modeling:199` |
| **语言模型** | `AutoModel.from_config(text_config)` | **完全相同** | **复用** | `modeling:262` |
| **token 化 / 输入 embedding** | `input_ids → embed_tokens(input_ids)` | 多一种特殊 token `<image>`（id 32000），随后**部分行被覆盖** | **复用 + 覆盖** | `modeling:457` + `:471`、config `:59` |
| **占位符替换** | 无 | `masked_scatter` | **全新增** | `modeling:471` |
| **图像预处理 / processor 组装** | 只需 tokenizer | image_processor（AnyRes 切图 + pad + 归一化）+ image_processor/tokenizer + `<image>` 展开 | **全新增** | `image_processing_llava_next.py:161`、`processing_llava_next.py:44` |
| **位置编码 / Attention / Mask / MLP / 归一化 / DecoderLayer** | RoPE；GQA self-attention；causal mask；SwiGLU/RMSNorm；Pre-Norm+残差 | 同（**没有**跨模态 attention，图像 token 与文本 token 互相可见） | **复用** | — |
| **`lm_head`** | `[hidden → vocab]` | 同（输入维度取自 `text_config`） | **复用** | `modeling:502` |
| **loss** | `ForCausalLMLoss` | 同（**不自动**屏蔽图像 token，需数据侧写 -100） | **复用** | `modeling:623` |
| **`generate` / `prepare_inputs_for_generation`** | `GenerationMixin` | 同（靠 `MULTIMODAL_INPUTS_TO_DROP_OUTSIDE_PREFILL` 只算一次视觉）；**不覆写** | **复用 + 库级优化** | `generation/utils.py:150` |
| **modular 文件** | 无 | 无（两者都是老式手写文件） | **相同** | `ls` 实测 |

**统计**：新增的实质性组件 **5 个**（vision tower / projector / image_newline / image_processor 逻辑 / processor 占位符展开），复用的组件 **10+ 个**（整个 decoder 栈、lm_head、loss、generate）。

> **NeXT 的改动 100% 集中在「图像 → token 序列」这一段**：输入端多切几张图（AnyRes）、中间多一步缝合（`pack_image_features` + `unpad` + `image_newline`）、多传一个 `image_sizes`。**投影器、语言模型、loss、生成逻辑，全部原封不动。** 而 `image_newline` 这个「每行一个标记 + base patch 在前」的二维布局，正是「把二维 token 图压成一维序列」唯一需要补的信息——这也解释了 patch token 数总是 `h' × (w' + 1)`。

## 一句话总结

> **多模态 LLM = 一个把图片翻译成向量序列的编码器 + 一个原封不动的语言模型。**
> 三步：**ViT 编码**（`[3,336,336] → [576,1024]`）→ **MLP 投影**（`→ [576,4096]`）→ **`masked_scatter` 替换** `inputs_embeds` 里 `<image>` 占位符对应的行；剩下的全是 1M1。**改造全在输入端，模型内部零改动**——这是读所有多模态模型的钥匙。LLaVA-NeXT 的「NeXT」只加了**分辨率**这一件事，投影器和语言模型和 1.5 逐行相同（有一处 `# Copied from` 为证）。

## 验证清单

本篇所有行号、类名、默认值均用以下命令实测（在仓库根目录，`transformers 5.16.0.dev0`，HEAD `42ca97014c`）：

```bash
# 1. 没有 modular 文件；同族的 llava_next_video / llava_onevision 有
ls src/transformers/models/llava_next/
# → __init__.py  configuration_llava_next.py  convert_llava_next_weights_to_hf.py
#   image_processing_llava_next.py  image_processing_pil_llava_next.py
#   modeling_llava_next.py  processing_llava_next.py      （无 modular_llava_next.py）
wc -l src/transformers/models/llava_next/*.py
# → modeling 637 / configuration 99 / processing 179 / image_processing 243

# 2. 类与函数清单（四个类 + 三个 helper）
grep -n "^class \|^def \|^    def " src/transformers/models/llava_next/modeling_llava_next.py
# → 41:get_anyres_image_grid_shape  72:image_size_to_num_patches  109:unpad_image
#   154:LlavaNextModelOutputWithPast  175:LlavaNextCausalLMOutputWithPast
#   200:LlavaNextMultiModalProjector  223:LlavaNextPreTrainedModel  250:LlavaNextModel
#   265:pack_image_features  337:get_image_features  407:get_placeholder_mask  433:forward
#   496:LlavaNextForConditionalGeneration  499:__init__  505:get_output_embeddings
#   508:pack_image_features  519:get_image_features  551:forward

# 3. 全文只有 1 处 `Copied from`，就是投影器
grep -c "Copied from" src/transformers/models/llava_next/modeling_llava_next.py   # → 1
grep -n "Copied from" src/transformers/models/llava_next/modeling_llava_next.py
# → 199:# Copied from transformers.models.llava.modeling_llava.LlavaMultiModalProjector with Llava->LlavaNext

# 4. 投影器与 1.5 逐行相同（改名后 diff 为空）
sed -n '200,219p' src/transformers/models/llava_next/modeling_llava_next.py > /tmp/a.py
sed -n '87,106p'  src/transformers/models/llava/modeling_llava.py           > /tmp/b.py
sed -i '' 's/Llava/LlavaNext/g' /tmp/b.py && diff /tmp/a.py /tmp/b.py
# → 无输出（完全一致）

# 5. 视觉编码器：AutoModel.from_config，且 config 默认是 CLIP
grep -n "AutoModel.from_config" src/transformers/models/llava_next/modeling_llava_next.py
# → 255:  self.vision_tower = AutoModel.from_config(config.vision_config)
# → 262:  self.language_model = AutoModel.from_config(config.text_config)
grep -n "clip_vision_model" src/transformers/models/llava_next/configuration_llava_next.py
# → 70, 73（默认 vision_config 是 CLIP）
grep -n "clip_vision_model" src/transformers/models/auto/modeling_auto.py   # → 89:("clip_vision_model", "CLIPVisionModel")
grep -n "def from_config" src/transformers/models/auto/auto_factory.py      # → 206
sed -n '250p' src/transformers/models/auto/auto_factory.py                  # → return model_class._from_config(config, **kwargs)

# 6. config 真实默认值
grep -n "image_token_index\|projector_hidden_act\|vision_feature_select_strategy\|vision_feature_layer\|multimodal_projector_bias\|tie_word_embeddings\|image_grid_pinpoints\|image_seq_length\|attribute_map" \
  src/transformers/models/llava_next/configuration_llava_next.py
# → 54: attribute_map = {"image_token_id": "image_token_index"}
# → 59: image_token_index = 32000        60: projector_hidden_act = "gelu"
# → 61: vision_feature_select_strategy = "default"   62: vision_feature_layer = -2
# → 63: multimodal_projector_bias = True  64: tie_word_embeddings = False
# → 65: image_grid_pinpoints = None（默认值在 :90-94 填入）  66: image_seq_length = 576

# 7. 占位符替换：masked_scatter（不是 torch.where / 布尔索引）
grep -n "masked_scatter\|get_placeholder_mask" src/transformers/models/llava_next/modeling_llava_next.py
# → 407:def get_placeholder_mask(...)   471:inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
grep -n "torch_compilable_check" src/transformers/models/llava_next/modeling_llava_next.py
# → 32（import）  424（调用，报错文案 "Image features and image tokens do not match"）

# 8. 数量不匹配 → ValueError；检查本身在 utils/import_utils.py
sed -n '1810,1830p' src/transformers/utils/import_utils.py   # def torch_compilable_check

# 9. 不覆写 prepare_inputs_for_generation
grep -n "prepare_inputs_for_generation" \
  src/transformers/models/llava_next/modeling_llava_next.py \
  src/transformers/models/llava/modeling_llava.py     # → 两者都无匹配
# 视觉只算一次的真正机制：
grep -n "MULTIMODAL_INPUTS_TO_DROP_OUTSIDE_PREFILL" src/transformers/generation/utils.py
# → 150（定义白名单）  616（丢弃逻辑）
sed -n '3861p' src/transformers/generation/utils.py   # is_first_iteration = False

# 10. loss：模型内没有 -100 屏蔽
grep -rn "\-100\|IGNORE_INDEX\|ignore_index" src/transformers/models/llava_next/
# → 只有 modeling_llava_next.py:574 的 docstring

# 11. AnyRes 的真实算术（1525 这个数字来自仓库里的测试）
grep -n "expected_image_tokens" tests/models/llava_next/test_processing_llava_next.py   # → 89: 1525
grep -n "num_additional_image_tokens" docs/source/en/model_doc/llava_next.md            # → 154, 158
grep -n "num_additional_image_tokens" tests/models/llava/test_processing_llava.py       # → 39（: 1）

# 12. SigLIP 主干确实被 LlavaNextForConditionalGeneration 支持
sed -n '334,345p' tests/models/llava_next/test_modeling_llava_next.py
# → test_granite_vision: "ibm-granite/granite-vision-3.1-2b-preview"（siglip, no CLS）

# 13. Processor 的 __call__ 流程（图片替换发生在 tokenize 之前）
grep -n "_process_images\|get_text_with_replacements\|self.tokenizer(text" src/transformers/processing_utils.py
# → 692: text, text_replacement_offsets = self.get_text_with_replacements(...)
# → 698: text_inputs = self.tokenizer(text, **merged_kwargs["text_kwargs"])
```

> 本章**未运行**任何会改写仓库的命令（`make fix-repo` / `modular_model_converter.py`），也没有下载模型或跑推理。

## 导航

- 路线图：[00_roadmap.md](./00_roadmap.md)（本篇 = 进阶 · 模型结构 · 2M6）
- 上一篇：[2M5_mamba2_ssm.md](./2M5_mamba2_ssm.md)
- 下一篇：[2M7_glm_moe_dsa.md](./2M7_glm_moe_dsa.md)
- **对照读**：[1M1_llama_structure.md](./1M1_llama_structure.md) —— 本篇的第 ⑦ 步之后就是它；
  [2A5_modular_code_generation.md](./2A5_modular_code_generation.md) —— 解释为什么 llava_next 要直读 `modeling_` 文件。
