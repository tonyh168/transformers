# Qwen3-MoE 精读：三种 MoE 的取舍对比

> 阅读目标：读完 Mixtral 的 MoE 之后，很多人会以为「MoE 就是 top-k 路由 + 一堆专家」。
> 这一篇要打破这个印象——**Mixtral / DeepSeek / Qwen3-MoE 三家的 router 实现互不兼容**，
> 而且它们的设计取舍对应着三种完全不同的工程哲学。
>
> **这篇的独特价值是「横向对比」，不是再讲一遍 MoE 基础。**
> MoE 的基本概念（稀疏激活、router、top-k gating）属于
> [1M3_mixtral_moe.md](./1M3_mixtral_moe.md)；本篇只做一件事：
> **把三家的 `configuration_*.py` 和 router 代码摊在一起，逐字段、逐行比对。**
>
> **附加收益（本篇最高价值的洞察）**：读到这里你会发现在 transformers 里，
> **MoE 的专家实现被抽成了一条继承链**——`Qwen3MoeExperts` 里一个 `pass` 都没有，
> 它的全部行为来自 `Qwen2MoeExperts` → `MixtralExperts`。
> 换句话说：**读一个 MoE 模型 = 读所有 MoE 模型**（第 六 节给出完整继承图，35 个 Experts 类追溯到同一个根）。
>
> 源文件：`src/transformers/models/qwen3_moe/modular_qwen3_moe.py`（205 行）、
> `src/transformers/models/qwen3_moe/configuration_qwen3_moe.py`（120 行）、
> 生成产物 `src/transformers/models/qwen3_moe/modeling_qwen3_moe.py`（706 行）
> 对照源文件：`models/mixtral/modular_mixtral.py`、`models/qwen2_moe/modular_qwen2_moe.py`、
> `models/deepseek_v2/modular_deepseek_v2.py`、`models/deepseek_v3/modular_deepseek_v3.py`、
> `integrations/moe.py`
> 人类向文档：`docs/source/en/modular_transformers.md`

---

## 速读区（TL;DR）

**一句话结论**：Mixtral / DeepSeek / Qwen3-MoE **三家的 router 互不兼容**（打分函数、归一化开关、分组、缩放各不相同），
但**三家的专家计算是同一份代码**（`MixtralExperts.forward`）——所以读 MoE 模型要「**router 逐个读、专家只读一次**」。

**必须记住的断言（12 条）**

| # | 断言 | 出处 |
|---|------|------|
| 1 | `Qwen3MoeExperts` 与 `Qwen3MoeTopKRouter` 都是 **`pass`**（零新代码，全继承 Qwen2Moe 系） | `modular_qwen3_moe.py:55,59` |
| 2 | Qwen3-MoE 唯一的新结构代码是 `Qwen3MoeSparseMoeBlock`（12 行），比 Qwen2-MoE **少了 `shared_expert` / `shared_expert_gate`**——**它取消了共享专家** | `modular_qwen3_moe.py:63-74` |
| 3 | **DeepSeek V2/V3 没有实现 aux loss**（`grep load_balancing_loss_func\|router_aux_loss_coef` 无输出，config 里连字段都没有）；Mixtral / Qwen3-MoE 有 | 第 8.1 节、验证清单 4 |
| 4 | Qwen3-MoE 的 `norm_topk_prob` 默认 **`False`** → top-k 权重之和 **< 1**；Mixtral 是**无条件归一化**，和恒为 1 | `configuration_qwen3_moe.py:106`、第 3.2 节 |
| 5 | `attribute_map = {"num_experts": "num_local_experts"}` **不是注释**，在 `__setattr__` / `__getattribute__` 里生效——两个名字指向同一个存储槽 | `configuration_qwen3_moe.py:51-53`、`configuration_utils.py:477-484` |
| 6 | `decoder_sparse_step` 的判据是 **1-based**：`(layer_idx + 1) % step == 0`；默认 `1` → **100% MoE、没有 dense 层** | `modular_qwen2_moe.py:133-147`、`configuration_qwen3_moe.py:102,116` |
| 7 | 「每 token 实际参与的专家总宽度 = 一个 dense MLP 的宽度」：Qwen3-MoE `8 × 768 = 6144 = intermediate_size`（Mixtral 是 2 倍，唯一例外） | 第 1 节、验证清单 10 |
| 8 | 35 个 Experts 类追溯到 **`MixtralExperts`**；20 个 Router 类追溯到 **`Qwen2MoeTopKRouter`** | 第 6.2 节 |
| 9 | 你读到的那段 `for expert_idx in expert_hit:` 循环是 **`eager` 后备实现**；`config._experts_implementation` 决定实际走哪条（默认 `None` → 打一句 warning 后退回 eager） | `integrations/moe.py:523,551-577` |
| 10 | `integrations/moe.py` **不是通用 MoE 组件库**：全文件只有一个类 `ExpertsInterface`（kernel 注册表），`:39` 的 `# class Experts(torch.nn.Module):` 是注释 | 第 6.1 节、验证清单 5 |
| 11 | DeepSeek V3 的打分是 **`sigmoid`**，不是 softmax；mask 用 `float("-inf")` | `modular_deepseek_v3.py:104` |
| 12 | Qwen3-MoE 的账本：**≈15.6B 总参 / ≈2.0B 激活**（top-8 只用到专家的 5.8%） | 第 8.2 节、验证清单 10 |

**最容易踩的坑 / 反直觉发现**

- **「三家都有 aux loss」是错的**——正文点名这是**最容易被凭记忆写错的一条**：Mixtral 和 Qwen3-MoE 有，**DeepSeek 没有**（它靠 `e_score_correction_bias` 做无辅助损失均衡）。
- **`qwen3` 与 `qwen3_moe` 是两套独立代码**，只共享 `Qwen3Attention` 一个类；把两者搞混是读这段源码最常见的错误。
- **Roadmap 草案把 Qwen3-MoE 的粒度标为「中」，被实测数据推翻**：768/6144 = 1/8 ≈ 0.125 vs DeepSeek 2048/18432 = 1/9 ≈ 0.111，**两者同档（细粒度）**，建议改成两档：Mixtral = 粗 / DeepSeek & Qwen3-MoE = 细。
- **默认不归一化会改变输出尺度**（置信度低 → 输出整体变小，相当于给残差外加了个软缩放）；所以**不能把 Mixtral 的 router 直接套到 Qwen3-MoE 上**。
- **`Qwen3MoeAttention` 用 `del self.layer_type` 削掉了 dense Qwen3 的逐层 attention 类型**，退化成全层统一的 `sliding_window`（`modular_qwen3_moe.py:47-48`）。
- **`ForCausalLM` 换到了 Mixtral 系**（`MixtralForCausalLM`），输出类型变成带 `aux_loss` / `router_logits` 的 `Moe*OutputWithPast`——所以「Qwen3-MoE = Qwen3 + MoE」在 config 和输出层面不成立。
- Qwen3-MoE 的 `_init_weights` 与 Mixtral 的同名方法**逐行相同，只是类名不同**——这是第 6 节继承链的第一个证据。

**读完应该能回答的问题**

- 三家的 router 各差在哪四步（打分 → top-k → 归一化 → 缩放）？
- 既然 `Qwen3MoeExperts` / `Qwen3MoeTopKRouter` 都是 `pass`，为什么不必去读生成文件里的同名类？
- `attribute_map` 为什么能让「读一次专家实现」在字段命名各异的三家之间成立？
- `decoder_sparse_step=2` 时哪些层是 MoE？写成 0-based 会差在哪？
- 为什么「三家都有 aux loss」这句话是错的？

---

## 前置知识

- [1M1_llama_structure.md](./1M1_llama_structure.md)：`LlamaDecoderLayer` 骨架。**Qwen3-MoE 的 DecoderLayer 与它逐行同构**，只是 MLP 换成了 MoE
- [1M2_mistral_sliding_window.md](./1M2_mistral_sliding_window.md)：modular 文件怎么读
- [1M3_mixtral_moe.md](./1M3_mixtral_moe.md)：MoE 基础（稀疏激活 / router / top-k / 负载均衡）。本篇假设你已读过
- [1M4_qwen3_dense.md](./1M4_qwen3_dense.md)：dense Qwen3（QK-Norm、`head_dim` 解耦、`layer_types`）。第 七 节要做 dense / MoE 对照
- [2A5_modular_code_generation.md](./2A5_modular_code_generation.md)：`pass` / `super().__init__()` / `del self.attr` 的含义。**不读这篇，本篇第 六 节的继承图会看不懂**
- [2M2_deepseek_mla_moe.md](./2M2_deepseek_mla_moe.md)：DeepSeekMoE 的「细粒度 + 共享专家」原始设计

> 注意：本篇是 **`qwen3_moe`**，不是 `qwen3`（dense）。
> 两者是**独立的两套代码**，`qwen3_moe` 只从 `qwen3` 里继承了一个类（`Qwen3Attention`）。
> 把这两者搞混是读这段源码最常见的错误——第 七 节专门处理这个对照。

---

## 一、三家的结构对比：一张并排图

先看形状，再看代码。以下是三家在**一个 DecoderLayer 的 MLP 位置**上做的事：

```
        ┌────────────────── Mixtral ──────────────────┐
        │  hidden (h)                                  │
        │      │                                       │
        │      ├──► gate: Linear(h → 8)                │
        │      │      softmax ─► topk(2) ─► 无条件归一化 │
        │      │                            │          │
        │      └────────────────────────────┼──────────┤
        │                                   ▼          │
        │   ┌───────────────────────────────────────┐  │
        │   │ Experts = 8 个「整宽」MLP 摞成 3D 张量  │  │
        │   │   每个专家中间维 = intermediate_size    │  │
        │   │   = 14336  ← 和 dense MLP 一样宽        │  │
        │   └───────────────────────────────────────┘  │
        │      挑 2 个 → 加权求和                       │
        │                  │                           │
        │              out (h)  ← 没有共享专家          │
        └──────────────────────────────────────────────┘

      ┌──────────────── DeepSeek V3 ────────────────┐
      │  hidden (h)                                  │
      │      │                                       │
      │      ├──► gate: Linear(h → 256)              │
      │      │      sigmoid ─► +bias ─► 分组选组      │
      │      │              ─► topk(8) ─► 可选归一化   │
      │      │              ─► × 2.5 (缩放)           │
      │      │                            │          │
      │      ├────────────────────────────┼──────────┤
      │      │                            ▼          │
      │      │   ┌──────────────────────────────┐    │
      │      │   │ 256 个「细粒度」专家          │    │
      │      │   │   每个中间维 = 2048           │    │
      │      │   │   = dense 的 1/9             │    │
      │      │   └──────────────────────────────┘    │
      │      │      挑 8 个 → 加权求和                │
      │      │                                       │
      │      └──► shared_experts: 1 个「整宽」MLP     │
      │            中间维 = 1×2048，永远激活          │
      │                  │                            │
      │         相加 ────┴────► out (h)               │
      └───────────────────────────────────────────────┘

      ┌────────────── Qwen3-MoE ────────────────┐
      │  hidden (h)                             │
      │      │                                  │
      │      ├──► gate: Linear(h → 128)         │
      │      │      softmax ─► topk(8)          │
      │      │      ─► norm_topk_prob? 默认 False│
      │      │                        │         │
      │      ├────────────────────────┼─────────┤
      │      │                        ▼         │
      │      │   ┌─────────────────────────┐    │
      │      │   │ 128 个「细粒度」专家     │    │
      │      │   │   每个中间维 = 768      │    │
      │      │   │   = dense 的 1/8        │    │
      │      │   └─────────────────────────┘    │
      │      │      挑 8 个 → 加权求和           │
      │      │                                  │
      │      └──► (无共享专家)                   │
      │                  │                       │
      │             out (h)                      │
      └──────────────────────────────────────────┘
```

**一眼看出的三个差异**：

| | Mixtral | DeepSeek V3 | Qwen3-MoE |
|---|---|---|---|
| 打分函数 | softmax | **sigmoid** | softmax |
| top-k 后归一化 | **无条件** | 可开关（默认开） | 可开关（**默认关**） |
| 额外缩放 | 无 | **× 2.5** | 无 |
| 选专家前是否分组 | 否 | **是**（8 组选 4 组） | 否 |
| 共享专家 | 无 | **1 个** | 无 |

这里的「打分函数 → top-k → 归一化 → 缩放」四步，就是三家 router 的全部差异所在。
第三节贴真实代码逐行确认。

**一个漂亮的数字巧合（自己算一遍就知道不是巧合）**：

| 模型 | top-k × 专家中间维 | 对照的 dense 中间维 |
|---|---|---|
| Mixtral | 2 × 14336 = 28672 | `intermediate_size` = 14336（**2 倍**） |
| Qwen2-MoE | 4 × 1408 = **5632** | `intermediate_size` = **5632**（也 = `shared_expert_intermediate_size`） |
| Qwen3-MoE | 8 × 768 = **6144** | `intermediate_size` = **6144** |
| DeepSeek V3 | (8 + 1) × 2048 = **18432** | `intermediate_size` = **18432** |

三家都让「**每个 token 实际参与的专家总宽度 = 一个 dense MLP 的宽度**」（Mixtral 是 2 倍，唯一例外）。
也就是说 MoE 换来的不是「每 token 算得更多」，而是「**总参数更多，但激活量对齐 dense**」——
这正是 MoE 的核心卖点。第 八 节展开。

---

## 二、Qwen3-MoE 全景：205 行 modular 文件里的 MoE 新代码只有 12 行

打开 `src/transformers/models/qwen3_moe/modular_qwen3_moe.py`（205 行），
去掉 license、import 和 `__all__`，**真正的类只有 10 个，其中 6 个是 `pass`**：

| 行号 | 类 | 基类 | 内容 |
|------|----|------|------|
| `:44` | `Qwen3MoeAttention` | `Qwen3Attention` | **真改动**（2 行：删 `layer_type`、设 `sliding_window`） |
| `:51` | `Qwen3MoeMLP` | `Qwen2MoeMLP` | `pass` |
| `:55` | `Qwen3MoeExperts` | `Qwen2MoeExperts` | **`pass`** ← 专家实现零改动 |
| `:59` | `Qwen3MoeTopKRouter` | `Qwen2MoeTopKRouter` | **`pass`** ← router 零改动 |
| `:63` | `Qwen3MoeSparseMoeBlock` | `nn.Module` | **真改动**（12 行，见下） |
| `:77` | `Qwen3MoeRMSNorm` | `LlamaRMSNorm` | `pass` |
| `:81` | `Qwen3MoeDecoderLayer` | `Qwen2MoeDecoderLayer` | `pass` |
| `:85` | `Qwen3MoePreTrainedModel` | `MixtralPreTrainedModel` | 只加了 `_can_record_outputs` |
| `:93` | `Qwen3MoeModel` | `MixtralModel` | `pass` |
| `:97` | `Qwen3MoeForCausalLM` | `MixtralForCausalLM` | 改了 `__init__` + 重写 `forward` 签名 |

**唯一一处真正属于 Qwen3-MoE 的新结构代码**（`modular_qwen3_moe.py:63-74`）：

```python
class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.experts = Qwen3MoeExperts(config)
        self.gate = Qwen3MoeTopKRouter(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        final_hidden_states = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
```

对比 `Qwen2MoeSparseMoeBlock`（`modular_qwen2_moe.py:111-131`），
Qwen3-MoE 的版本**少了 `shared_expert` 和 `shared_expert_gate` 两行**——
这是它和 Qwen2-MoE 唯一的实质结构差异。**Qwen3-MoE 取消了共享专家。**

> **这就是 2A5 讲的那件事的活例子**：`modular_qwen3_moe.py` 只有 205 行，
> 但生成的 `modeling_qwen3_moe.py` 有 706 行。
> `Qwen3MoeDecoderLayer` 在 modular 里是 `pass`，生成出来却是 45 行的完整实现
> （`modeling_qwen3_moe.py:305-348`），而且里面调用的是 `Qwen3MoeSparseMoeBlock`——
> 类名被 converter 按前缀 `Qwen3Moe` 批量改写过了。

---

## 三、Router 实测对比（贴真实代码）

这是本篇的主菜。三家 router 的 `forward` 并排读，差异一目了然。

### 3.1 Mixtral：softmax → top-k → **无条件**归一化

`src/transformers/models/mixtral/modular_mixtral.py:162-177`：

```python
class MixtralTopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok          # 2
        self.num_experts = config.num_local_experts      # 8
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
        router_probs = torch.nn.functional.softmax(router_logits.float(), dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)  # (seq_len, top_k)
        router_top_value /= router_top_value.sum(dim=-1, keepdim=True)   # ← 无条件重新归一化
        router_scores = router_top_value
        return router_logits, router_scores, router_indices
```

三个要点：
1. **`softmax` 作用在全部 8 个专家上**，然后才 `topk`。
2. **归一化没有开关**——`/= sum(...)` 这一行硬编码。归一化后 top-2 权重和恒为 1。
3. 权重以 **float32** 传给 experts（`softmax(router_logits.float(), ...)` 的结果没有被 cast 回去）。

### 3.2 Qwen3-MoE：softmax → top-k → **可开关**归一化（默认关）

实测确认：Qwen3-MoE 的 router 类叫 **`Qwen3MoeTopKRouter`**
（`modular_qwen3_moe.py:59`，定义时是 `pass`），生成后的实体在
`modeling_qwen3_moe.py:249-268`：

```python
class Qwen3MoeTopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok          # 8
        self.num_experts = config.num_experts            # 128
        self.norm_topk_prob = config.norm_topk_prob      # ← 开关，默认 False
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
        router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)  # (seq_len, top_k)
        if self.norm_topk_prob:                          # ← 默认不进这个分支
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        router_scores = router_top_value
        return router_logits, router_scores, router_indices
```

**和 Mixtral 的四处逐行差异**（这是本篇最该记住的对照）：

| 差异点 | Mixtral | Qwen3-MoE |
|--------|---------|-----------|
| 归一化 | 硬编码执行 | `if self.norm_topk_prob:`（**默认 False → 不执行**） |
| 默认下 top-k 权重之和 | **恒为 1** | 是 128 路 softmax 里的 top-8 原值，**和 < 1**（分布越平坦越小，极端平坦时 ≈ 8/128 = 6.25%；越尖锐越接近 1） |
| 权重 dtype | 保持 float32 | `router_top_value.to(router_logits.dtype)` **cast 回 bf16/fp16** |
| 权重初始化 | `torch.empty` | `torch.zeros`（再由 `_init_weights` 填正态） |

> **为什么「默认不归一化」这件事重要**：归一化与否改变了「专家输出的整体幅度」。
> 归一化时输出是 top-k 概率的**加权平均**（凸组合，尺度稳定）；
> 不归一化时输出尺度随路由置信度浮动（置信度低 → 输出整体变小 → 相当于给残差外加了个软缩放）。
> 两种做法在不同模型上都被验证能训得动，属于**调参习惯**而非对错，
> 但它让你**不能把 Mixtral 的 router 直接套到 Qwen3-MoE 上**。

**权重初始化**（`modeling_qwen3_moe.py:371-378`）：

```python
def _init_weights(self, module):
    super()._init_weights(module)
    std = self.config.initializer_range
    if isinstance(module, Qwen3MoeExperts):
        init.normal_(module.gate_up_proj, mean=0.0, std=std)
        init.normal_(module.down_proj, mean=0.0, std=std)
    elif isinstance(module, Qwen3MoeTopKRouter):
        init.normal_(module.weight, mean=0.0, std=std)
```

对照 Mixtral 的同名方法（`modular_mixtral.py:257-265`）——**逐行相同，只是类名不同**。
这是第 六 节「继承链」的第一个证据。

### 3.3 DeepSeek V2 / V3：sigmoid + 分组 + 缩放

DeepSeek 的 router 走的是另一条设计路线，分两代。

**V2**（`modular_deepseek_v2.py:161-190`）——相对 Qwen 系的差异：

```python
class DeepseekV2TopkRouter(Qwen2MoeTopKRouter):     # ← 继承 Qwen2-MoE 的 router！
    def __init__(self, config: DeepseekV2Config):
        super().__init__(config)
        del self.norm_topk_prob                      # 删掉继承来的开关
        self.routed_scaling_factor = config.routed_scaling_factor
        self.topk_method = config.topk_method
        self.num_group = config.n_group
        self.topk_group = config.topk_group

    def forward(self, hidden_states):
        ...
        scores = router_logits.softmax(dim=-1, dtype=torch.float32)   # V2 还是 softmax
        if self.topk_method == "greedy":
            topk_weights, topk_indices = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        elif self.topk_method == "group_limited_greedy":
            # 把 64 个专家分成 n_group 组，每组取最高分，组内再做 top-k
            group_scores = scores.view(-1, self.num_group, self.num_experts // self.num_group).max(dim=-1).values
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            ...
        topk_weights = topk_weights * self.routed_scaling_factor       # ← 缩放
        return router_logits, topk_weights, topk_indices
```

**V3**（`modular_deepseek_v3.py:87-117`）——关键的两处变化：

```python
class DeepseekV3TopkRouter(DeepseekV2TopkRouter):
    def __init__(self, config):
        super().__init__(config)
        del self.topk_method
        self.num_experts = config.num_local_experts      # 256
        self.norm_topk_prob = config.norm_topk_prob      # ← 开关又回来了（这里默认 True）
        self.e_score_correction_bias = nn.Buffer(torch.zeros(self.num_experts))   # ← 新增

    def forward(self, hidden_states):
        ...
        scores = router_logits.sigmoid()                 # ★ softmax → sigmoid
        scores_for_choice = scores + self.e_score_correction_bias   # ★ 选专家用带 bias 的分，加权用不带 bias 的
        group_scores = (scores_for_choice.view(-1, self.num_group, self.num_experts // self.num_group)
                        .topk(2, dim=-1)[0].sum(dim=-1))     # 每组取 top-2 之和作为组得分
        ...
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_indices)     # ★ 权重来自「不带 bias」的 scores
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return router_logits, topk_weights, topk_indices
```

**V3 的三个关键设计**（每一个都不是微调）：

1. **`sigmoid` 取代 `softmax`**：softmax 让所有专家**互相竞争**（一个变大必然导致另一个变小）；
   sigmoid 让每个专家**独立打分**。配合 top-k，这意味着「理论上可以所有专家都很高分」，
   更适合 256 个专家的细粒度场景——softmax 在 256 路里会把概率稀释得很厉害。
2. **`e_score_correction_bias`**：一个 `torch.zeros(num_experts)` 的 buffer，
   **只影响「选哪几个专家」，不影响「加权时的权重」**。
   这是 DeepSeek 做负载均衡的手段之一（无辅助损失的 bias 调整）。
3. **分组选专家（`group_limited_greedy`）**：先按组打分选 4 组（共 8 组），
   在这 4 组内做 top-8。目的是防止「top-8 全落在同一组」导致路由塌缩。

### 3.4 三家 router 差异总表

| 维度 | Mixtral | DeepSeek V2 | DeepSeek V3 | Qwen3-MoE |
|------|---------|-------------|-------------|-----------|
| 类名 | `MixtralTopKRouter` | `DeepseekV2TopkRouter` | `DeepseekV3TopkRouter` | `Qwen3MoeTopKRouter` |
| 定义位置（modular） | `mixtral/modular_mixtral.py:162` | `deepseek_v2/modular_deepseek_v2.py:161` | `deepseek_v3/modular_deepseek_v3.py:87` | `qwen3_moe/modular_qwen3_moe.py:59` |
| **基类** | `nn.Module`（**根**） | `Qwen2MoeTopKRouter` | `DeepseekV2TopkRouter` | `Qwen2MoeTopKRouter`（**`pass`**） |
| 专家数 | 8 | 64 | 256 | 128 |
| top-k | 2 | 可配 | 8 | 8 |
| 打分 | `softmax` | `softmax` | **`sigmoid`** | `softmax` |
| top-k 后归一化 | **无条件** | 无（`del` 掉了开关） | `norm_topk_prob`（类默认 `True`） | `norm_topk_prob`（类默认 **`False`**） |
| 额外缩放 | 无 | `× routed_scaling_factor` | `× routed_scaling_factor` | 无 |
| 选专家前分组 | 否 | `topk_method` 可开 | **是**（8 组选 4） | 否 |
| 修正 bias | 无 | 无 | **`e_score_correction_bias`** | 无 |
| 权重 dtype | float32 | float32 | 继承 V2 的 float32 路径 | **cast 回 `router_logits.dtype`** |

> **读这段代码的正确姿势**：不要试图记「谁是对的」。
> 这四列里的每一个格子都是**训练超参**，改动一行代码就能切换。
> 真正需要记牢的是「**softmax vs sigmoid**」和「**归一化开关**」这两条分水岭——
> 它们决定了 router 的数值行为，也决定了任何一个 MoE 实现能不能直接移植。

---

## 四、Experts 实测对比：同一个 kernel，三种配置

三家路由差异巨大，但**专家计算本身是同一份代码**。
实测：`Qwen3MoeExperts` 的 `forward` 与 `MixtralExperts` 的 `forward`
**逐字符相同**（下面两份代码请逐行对照）。

**Mixtral**（`modular_mixtral.py:122-159`，`@use_experts_implementation` 在 `:122`，
`forward` 在 `:135`）：

```python
@use_experts_implementation
class MixtralExperts(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.num_experts = config.num_local_experts      # 8
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.intermediate_size # 14336
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states
```

**Qwen3-MoE**（`modeling_qwen3_moe.py:209-247`，`@use_experts_implementation` 在 `:209`）——
**只有 `__init__` 的 3 行不同**：

```python
@use_experts_implementation
class Qwen3MoeExperts(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts               # ← 128（Mixtral 是 num_local_experts）
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size # ← 768（Mixtral 是 intermediate_size）
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states, top_k_index, top_k_weights):
        # ↑↑↑ 以下与 MixtralExperts.forward 逐行一致 ↑↑↑
```

**三家的专家实现差异，压缩成一张表**：

| 配置项从哪个字段读 | Mixtral | DeepSeek V2/V3 | Qwen3-MoE |
|---|---|---|---|
| 专家数量 | `config.num_local_experts` (8) | `config.num_local_experts` (V3: 256) | `config.num_experts` (128) |
| 专家中间维 | `config.intermediate_size` (14336) | 继承 Qwen2Moe：`config.moe_intermediate_size` | `config.moe_intermediate_size` (768) |
| 权重布局 | `gate_up_proj` (E, 2I, H) + `down_proj` (E, H, I) | 同 | 同 |
| `forward` 主体 | — | 同（`pass`） | 同（`pass`） |

**这就是「同一个 kernel，三种配置」**：`forward` 是完全共享的，
唯一的「配置」是 `num_experts` / `intermediate_dim` 从哪个 config 字段读。

### `@use_experts_implementation`：为什么它值得单独看一眼

`MixtralExperts` 和 `Qwen3MoeExperts` 头上都有这个装饰器
（定义在 `integrations/moe.py:523`）。它做两件事（`moe.py:551-577`）：

```python
def wrapper(experts_class):
    @wraps(original_forward)
    def forward(self, *args, **kwargs):
        experts_forward = experts_interface.get_interface(self.config._experts_implementation, original_forward)
        return experts_forward(self, *args, **kwargs)
    ...
```

也就是说，**你读到的那段 `for expert_idx in expert_hit:` 循环是 `eager` 后备实现，
不一定真的在执行**。`config._experts_implementation` 决定走哪条路：

| 值 | 实现 | 位于 |
|----|------|------|
| `None` / `"eager"` | 就是上面那段 Python 循环 | `MixtralExperts.forward` 自身 |
| `"batched_mm"` | `batched_mm_experts_forward` | `integrations/moe.py:112` |
| `"grouped_mm"` | `grouped_mm_experts_forward` | `integrations/moe.py:377` |
| `"deepgemm"` | `deepgemm_bf16_experts_forward` | `integrations/deepgemm.py` |
| `"sonicmoe"` | `sonicmoe_experts_forward` | `integrations/sonicmoe.py` |

注册表实体（`integrations/moe.py:481-506`）：

```python
class ExpertsInterface(GeneralInterface):
    """Interface for registering custom experts forward functions."""

    _global_mapping = {
        "deepgemm": deepgemm_bf16_experts_forward,
        "batched_mm": batched_mm_experts_forward,
        "grouped_mm": grouped_mm_experts_forward,
        "sonicmoe": sonicmoe_experts_forward,
    }
```

> **踩坑提醒**：`config._experts_implementation` 默认是 `None`（`configuration_utils.py:333`
> 从 `kwargs.pop("experts_implementation", None)` 来），此时会打印一句 warning 然后退回 eager。
> 所以「读源码时看到的 forward 循环」和「实际跑的时候」可能不是同一份代码。
> 这是 2A1「注意力后端」机制在 MoE 上的翻版（对照 `ALL_ATTENTION_FUNCTIONS`）。

---

## 五、Qwen3-MoE 的 MoE 超参全表

**所有默认值均逐行取自 `src/transformers/models/qwen3_moe/configuration_qwen3_moe.py`。**

| 超参 | 行号 | 默认值 | 含义 | 对照：Qwen2-MoE | 对照：Mixtral |
|------|------|--------|------|-----------------|---------------|
| `num_experts` | `:105` | **128** | 路由专家总数 | 60 | 8（叫 `num_local_experts`） |
| `num_experts_per_tok` | `:104` | **8** | 每 token 激活数（top-k） | 4 | 2 |
| `moe_intermediate_size` | `:103` | **768** | **每个专家**的中间维 | 1408 | 无（用 `intermediate_size`=14336） |
| `norm_topk_prob` | `:106` | **`False`** | top-k 后是否重新归一化 | `False` | 无此开关（恒归一化） |
| `decoder_sparse_step` | `:102` | **1** | **每隔几层用 MoE** | 1 | 无（全层 MoE） |
| `mlp_only_layers` | `:109` | `None` → `[]`（`:116`） | 白名单：这些层强制用 dense MLP | 同 | 无 |
| `router_aux_loss_coef` | `:108` | **0.001** | 负载均衡损失系数 | 0.001 | 0.001 |
| `output_router_logits` | `:107` | `False` | 是否输出 router logits（算 aux loss 用） | `False` | `False` |
| `intermediate_size` | `:87` | 6144 | **dense** MLP 中间维（只在非 MoE 层用） | 同（5632，`:79`） | 14336（＝专家宽度） |
| `hidden_size` | `:86` | 2048 | 隐藏维 | — | 4096 |
| `num_hidden_layers` | `:88` | 24 | 层数 | — | 32 |
| `shared_expert_intermediate_size` | — | **不存在** | — | **5632（存在）** | 不存在 |

### 三个必须实测才能确认的结论

**结论 1：`shared_expert_intermediate_size` 在 Qwen3-MoE 的 config 里根本不存在。**

```bash
$ grep -rn "shared_expert" src/transformers/models/qwen3_moe/
（无输出）
```

它只出现在 Qwen2-MoE 的 config（`configuration_qwen2_moe.py:96`，默认 5632）里。
这印证了第 二 节的观察：Qwen3-MoE 的 `Qwen3MoeSparseMoeBlock` 里没有 `shared_expert`。
**这是一次明确的设计回退**——Qwen2-MoE 有共享专家，Qwen3-MoE 删掉了。

**结论 2：`num_experts` 与 `num_local_experts` 是同一个字段的两个名字。**

config 里有这么一行（`configuration_qwen3_moe.py:51-53`）：
`attribute_map = {"num_experts": "num_local_experts"}`。

**`attribute_map` 不是注释**——它在 `PreTrainedConfig` 的 `__setattr__` / `__getattribute__`
里生效（`configuration_utils.py:477-484`）：

```python
def __getattribute__(self, key):
    if key != "attribute_map" and key in super().__getattribute__("attribute_map"):
        key = super().__getattribute__("attribute_map")[key]     # ← 读的时候重定向
    return super().__getattribute__(key)
```

因为读写都过同一张映射表（`__setattr__` 同理，`:477-479`），
所以 `config.num_experts` 和 `config.num_local_experts` **指向同一个存储槽**。
这个机制是为了**向后兼容旧 checkpoint**（旧 `config.json` 里可能写的是 `num_local_experts`）。

`DeepseekV3Config` 用了同样的技巧，方向相反（`configuration_deepseek_v3.py:78-81`）：
`attribute_map = {"num_local_experts": "n_routed_experts", ...}`，
所以 `DeepseekV3Experts` 能从 `config.num_local_experts` 读到 256——
尽管 config 里声明的字段叫 `n_routed_experts`（`:91`）。

> 这三家的命名风格差异（`num_experts` / `num_local_experts` / `n_routed_experts`）
> 全部靠 `attribute_map` 抹平。**这就是为什么第 六 节的继承链能成立**——
> 子类只要在 `__init__` 里写一句 `self.num_experts = config.<本地名字>` 就对齐了。

**结论 3：`decoder_sparse_step` 是 Qwen 系（Qwen2-MoE / Qwen3-MoE）的特色，默认「全层都是 MoE」。**

这个参数控制「**每隔几层放一个 MoE 层**」，即 dense 层与 MoE 层交替。
真实代码在 `modular_qwen2_moe.py:133-147`（`Qwen3MoeDecoderLayer` 是它的 `pass`，
生成后落在 `modeling_qwen3_moe.py:305-317`）：

```python
class Qwen3MoeDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3MoeAttention(config, layer_idx)
        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3MoeSparseMoeBlock(config)          # ← MoE 层
        else:
            self.mlp = Qwen3MoeMLP(config, intermediate_size=config.intermediate_size)  # ← dense 层
        self.input_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_size = config.hidden_size
```

选层的规则是**两条**（`and` 连接，`layer_idx` 从 0 开始）：

| 条件 | 含义 |
|------|------|
| `layer_idx not in config.mlp_only_layers` | 不在白名单里的层才考虑 MoE |
| `(layer_idx + 1) % config.decoder_sparse_step == 0` | **1-based** 计数，每 `decoder_sparse_step` 层命中一次 |

**注意 `layer_idx + 1`**：这是 1-based 判断，所以 `decoder_sparse_step=2` 时
命中 `layer_idx ∈ {1, 3, 5, ...}`（第 2、4、6 层）。
写 0-based 的判断（`layer_idx % step == 0`）会命中 `{0, 2, 4, ...}`，**结果不同**——容易看错。

**默认值 `decoder_sparse_step=1` 的含义**：`(layer_idx + 1) % 1 == 0` 对所有层恒真，
所以**默认情况下每一层都是 MoE**。加上 `mlp_only_layers` 默认为 `[]`（`:116`），
Qwen3-MoE 的默认 config 是 **100% MoE、没有 dense 层**。

> **这个参数为什么存在**：它是给**混合深度模型**留的旋钮。
> 直觉上「深层更需要 MoE 容量，浅层用 dense 就够」，或者反过来「浅层做特征提取、深层做知识存储」，
> 两种假设都有实践。`decoder_sparse_step > 1` 就能一键切到「隔层 MoE」形态，无需改结构代码。
> 这是一个**便宜的架构搜索旋钮**，而不是当前默认可比的配置。
>
> **横向对照三种「dense / MoE 混合」的做法**：
> | 模型 | 控制字段 | 默认 | 语义 |
> |------|----------|------|------|
> | Qwen3-MoE / Qwen2-MoE | `decoder_sparse_step` + `mlp_only_layers` | 1 / `[]` | 隔 N 层；或点名白名单 |
> | DeepSeek V2 | `first_k_dense_replace` | 0 | 前 k 层是 dense（`modular_deepseek_v2.py:118,391`） |
> | DeepSeek V3 | `first_k_dense_replace` | **3** | 前 3 层 dense（`modular_deepseek_v3.py:204`） |
> | Mixtral | 无 | — | **全层 MoE，没有旋钮**（`modular_mixtral.py:218` 直接 `self.mlp = MixtralSparseMoeBlock(config)`） |

---

## 六、通用 MoE 组件抽取：为什么读一个 MoE 模型 = 读所有 MoE 模型

这是本篇最高价值的洞察。

### 6.1 先纠正一个常见误解：`integrations/moe.py` 不是「通用 MoE 组件库」

很多人（包括不少文档）以为 `src/transformers/integrations/moe.py` 里有
`Qwen2MoeExperts` / `MixtralExperts` 这些基类。**实测：没有。**

```bash
$ grep -c "^class Experts\b" src/transformers/integrations/moe.py
0
$ grep -n "^class" src/transformers/integrations/moe.py
481:class ExpertsInterface(GeneralInterface):        # ← 唯一一个类，是 kernel 注册表
```

**`integrations/moe.py`（582 行）的真实身份是「MoE kernel 分发层」**：

| 行号 | 内容 |
|------|------|
| `:39-74` | 注释形式保存的一份「Experts 类应该长什么样」的参考实现（**只是注释**，第 39 行是 `# class Experts(torch.nn.Module):`） |
| `:78` | `_batched_linear`（辅助函数） |
| `:112` | `batched_mm_experts_forward` —— `torch.bmm` 批量专家 |
| `:339` | `_grouped_linear`（辅助函数） |
| `:377` | `grouped_mm_experts_forward` —— 分组 GEMM，带 EP sentinel 处理 |
| `:481` | `ExpertsInterface` —— 后端注册表 |
| `:523` | `use_experts_implementation` —— 装饰器，按 `config._experts_implementation` 换 forward |

**真正的「通用 MoE 组件」住在模型目录里**，靠 modular 继承串起来。下一节画图。

### 6.2 三个根：`MixtralExperts` / `MixtralTopKRouter` / `Qwen2MoeTopKRouter`

实测（用脚本解析所有 `models/*/modular_*.py` 的 `class X(Y)` 关系，见验证清单）：

| 根类 | 定义位置 | 直接子类 | 后代总数 | 语义 |
|------|----------|----------|----------|------|
| **`MixtralExperts`** | `mixtral/modular_mixtral.py:123` | 12 | **35** | 「3D 张量存权重 + eager 循环」的通用专家实现 |
| **`MixtralTopKRouter`** | `mixtral/modular_mixtral.py:162` | 6 | 8 | 「softmax + 无条件归一化」路由 |
| **`Qwen2MoeTopKRouter`** | `qwen2_moe/modular_qwen2_moe.py:90` | 6 | **20** | 「softmax + `norm_topk_prob` 开关」路由 |
| `DeepseekV2TopkRouter` | `deepseek_v2/modular_deepseek_v2.py:161` | 2 | 11 | 「+ 分组 + 缩放」（继承 Qwen2Moe） |
| `DeepseekV3TopkRouter` | `deepseek_v3/modular_deepseek_v3.py:87` | 7 | 7 | 「sigmoid + bias + 分组」 |

也就是说：
- **35 个 Experts 类（分布在 30 多个模型目录）的专家实现，最终都追溯到 `MixtralExperts`**；
- **20 个 Router 类的路由实现，追溯到 `Qwen2MoeTopKRouter`**；
- DeepSeek 系是 Qwen2Moe 路由的一个分支（`DeepseekV2TopkRouter` 的后代有 11 个）。

按你的要求逐条复核（**全部实测通过**）：

```bash
$ grep -rn "^class DeepseekV2Experts\|^class DeepseekV4Experts" src/transformers/models/*/modular_*.py
deepseek_v2/modular_deepseek_v2.py:157:class DeepseekV2Experts(Qwen2MoeExperts):
deepseek_v4/modular_deepseek_v4.py:879:class DeepseekV4Experts(MixtralExperts):
$ grep -rn "^class Qwen3MoeExperts" src/transformers/models/qwen3_moe/modular_qwen3_moe.py
55:class Qwen3MoeExperts(Qwen2MoeExperts):
```

### 6.3 ASCII 继承图

```
                       ┌───────────────────────────────────────────┐
   根（唯一手写实现） → │  MixtralExperts(nn.Module)                │
                       │  mixtral/modular_mixtral.py:123           │
                       │  · gate_up_proj (E, 2I, H)                │
                       │  · down_proj    (E, H, I)                 │
                       │  · forward(): one_hot → 循环 → index_add_ │
                       │  @use_experts_implementation              │
                       └────────────────┬──────────────────────────┘
                                        │
        ┌───────────────┬───────────────┼───────────────┬───────────────┐
        ▼               ▼               ▼               ▼               ▼
  ┌───────────┐  ┌───────────┐  ┌─────────────┐  ┌───────────┐  ┌───────────┐
  │ Jamba     │  │ Olmoe     │  │ Qwen2Moe    │  │ GraniteMoe│  │ MiniMaxM2 │ ...
  │ Experts   │  │ Experts   │  │ Experts     │  │ Experts   │  │ Experts   │
  │ (pass)    │  │ (pass)    │  │ 改 2 行 ←★★ │  │ (pass)    │  │ (pass)    │
  └───────────┘  └───────────┘  └──────┬──────┘  └───────────┘  └───────────┘
                                       │   qwen2_moe/modular_qwen2_moe.py:83
                                       │   num_experts        ← config.num_experts
                                       │   intermediate_dim   ← config.moe_intermediate_size
                                       │
        ┌──────────────┬───────────────┼───────────────┬──────────────┐
        ▼              ▼               ▼               ▼              ▼
  ┌──────────┐  ┌───────────┐  ┌───────────┐  ┌────────────┐  ┌───────────┐
  │ Afmoe    │  │ Deepseek  │  │ Deepseek  │  │ Qwen3Moe   │  │ Qwen3Next │
  │ Experts  │  │ V2Experts │  │ V3Experts │  │ Experts    │  │ Experts   │
  │ (pass)   │  │ (pass)    │  │ 改 1 行   │  │  (pass) ★★ │  │  (pass)   │
  └──────────┘  └───────────┘  └─────┬─────┘  └──────┬─────┘  └─────┬─────┘
                                     │               │              │
                    ┌────────────────┤        ┌──────┴──────┐  ┌────┴─────┐
                    ▼        ▼       ▼        ▼      ▼      ▼  ▼          ▼
                 AXK1    Exaone  Mistral4  HYV3  Laguna  Qwen3  Qwen3_5  Qwen4Exp
                Experts  Experts Experts  (pass)(pass) VL/Omni  Moe     Text
                                     ...            Zaya  (pass)  (pass)  (pass)
                                                    (pass)

   合计：35 个 Experts 类，其中 21 个是纯 `pass`（零新代码），只有 14 个真的改了东西。
```

Router 侧同理，但**有两个平行的根**（因为 Mixtral 和 Qwen2-MoE 的路由语义不同）：

```
   ┌──────────────────────────────┐        ┌────────────────────────────────────┐
   │ MixtralTopKRouter(nn.Module) │        │ Qwen2MoeTopKRouter(nn.Module)      │
   │ mixtral/modular_mixtral.py:162│       │ qwen2_moe/modular_qwen2_moe.py:90  │
   │ softmax + 无条件归一化        │        │ softmax + norm_topk_prob 开关       │
   └───────────┬──────────────────┘        └───────────────┬────────────────────┘
               │                                            │
   ┌───────────┼──────────┬────────────┐    ┌──────────────┼──────────────┬──────────┐
   ▼           ▼          ▼            ▼    ▼              ▼              ▼          ▼
 MiniMax   MiniMaxM2   HYV3       Longcat  DeepseekV2   Lfm2Moe       Olmoe    Qwen3Moe  ...
 TopK      TopK        TopK       Flash    TopkRouter   TopKRouter    TopK     TopKRouter
 Router    Router      Router     Topk     (改 6 行)    (pass)        (pass)   (pass) ★★
                                    Router      │
   ┌───────────────┬───────────────────────────┤
   ▼               ▼                           ▼
 DeepseekV3      Mistral4                 DeepseekV4
 TopkRouter ★    TopkRouter               TopKRouter / HashRouter
 (sigmoid)            │
   │                  │
   ├─ AXK1 / Dots1 / ExaoneMoe / Glm4Moe★ / Glm5Next / MiMoV2Flash / NemotronH
   │                                    (后缀 7 个，其中 Glm4Moe 又派生出
   │                                     Glm4MoeLiteTopkRouter / Glm4vMoeTextTopkRouter)
   └─ ...
```

### 6.4 `pass` 的含义：一个 `pass` 顶 40 行

`Qwen3MoeExperts` 和 `Qwen3MoeTopKRouter` 都是 `pass`（`modular_qwen3_moe.py:55, 59`）。
按 2A5 的规则，`pass` ＝「原样继承，零覆盖」，生成时父类整块被抄下来 + 全局改名。
所以生成文件里那 38 行 `Qwen3MoeExperts`（`modeling_qwen3_moe.py:210-247`）
和 20 行 `Qwen3MoeTopKRouter`（`:249-268`）**没有一个字符是 Qwen3-MoE 自己写的**。

**实用推论（这一节的结论）**：

| 你想看的 | 该去哪看 | 不要去哪看 |
|----------|----------|-----------|
| Qwen3-MoE 的专家计算逻辑 | `modular_mixtral.py:135-159`（`MixtralExperts.forward`） | ~~`modeling_qwen3_moe.py:223`~~（同样的代码，白读） |
| Qwen3-MoE 路由的全部行为差异 | **`modular_qwen2_moe.py:90-108`**（`Qwen2MoeTopKRouter`，20 行） | 上面那些 `pass` 类 |
| DeepSeek V3 路由的 sigmoid / bias / 分组 | `modular_deepseek_v3.py:87-117` | — |

**一句话**：读 MoE 模型的正确姿势是**顺着继承链往上跳到「有内容的那一层」，然后只读那一层**。
`Qwen3MoeExperts` 是 `pass` 就说明「专家部分和 Mixtral 一模一样」，
`Qwen3MoeTopKRouter` 是 `pass` 就说明「路由和 Qwen2-MoE 一模一样」——
这两句结论把「读一个 MoE 模型」的工作量砍掉了 90%。

---

## 七、对照表：dense Qwen3 vs MoE Qwen3

`qwen3`（dense，[1M4](./1M4_qwen3_dense.md)）和 `qwen3_moe` 是**两个独立的模型目录**。
`modular_qwen3_moe.py` 只从 dense 那边 import 了 `Qwen3Attention`（`:37`），
其余全部从 Mixtral / Qwen2-MoE 继承（`:30-36`）。

| 维度 | `qwen3`（dense） | `qwen3_moe` |
|------|------------------|-------------|
| 目录 | `models/qwen3/` | `models/qwen3_moe/` |
| `model_type` | `qwen3` | `qwen3_moe` |
| MLP 位置放什么 | `Qwen3MLP`（`modular_qwen3.py:52`，`pass` 自 `GemmaMLP`） | `Qwen3MoeSparseMoeBlock`（`modular_qwen3_moe.py:63`） |
| config 里的 MLP 宽度 | `intermediate_size` = 22016（`:64`） | `intermediate_size`=6144 + `moe_intermediate_size`=768 |
| **层类型控制** | `layer_types` + `max_window_layers`=28（`configuration_qwen3.py:79-80`），逐层 `"full_attention"` / `"sliding_attention"` | **只有全局 `use_sliding_window` / `sliding_window`**，没有 `layer_types` |
| Attention 里的 `layer_type` | `self.layer_type = config.layer_types[layer_idx]`（`modular_qwen3.py:62`），逐层判 `sliding_window` | **`del self.layer_type`**（`modular_qwen3_moe.py:47`），`sliding_window` 由 config 全局决定（`:48`） |
| QK-Norm | 有（`q_norm` / `k_norm`，`modular_qwen3.py:65-66`） | **有**（继承同一份 `Qwen3Attention.__init__`，没有删） |
| `ForCausalLM` 的基类 | `Qwen2ForCausalLM`（`modular_qwen3.py:110`） | **`MixtralForCausalLM`**（`modular_qwen3_moe.py:97`） |
| 输出类型 | `CausalLMOutputWithPast` / `BaseModelOutputWithPast` | **`MoeCausalLMOutputWithPast` / `MoeModelOutputWithPast`**（多了 `aux_loss`、`router_logits`） |
| 额外 loss | 无 | `load_balancing_loss_func`（`modeling_qwen3_moe.py:514`）+ `router_aux_loss_coef`=0.001 |
| 额外 config 字段 | — | `num_experts`、`num_experts_per_tok`、`moe_intermediate_size`、`norm_topk_prob`、`decoder_sparse_step`、`mlp_only_layers`、`output_router_logits`、`router_aux_loss_coef` |
| `_can_record_outputs` | 无（默认） | 加了 `router_logits` 采集器（`modular_qwen3_moe.py:86-90`） |

### 差异不止「换了个 MLP」——有四处

1. **Attention 被削了一层能力**：dense Qwen3 支持**逐层**选择 sliding / full attention
   （`layer_types`），`Qwen3MoeAttention` 用 `del self.layer_type` 把它删掉了，
   退化成「全层统一」的 `sliding_window`（`modular_qwen3_moe.py:47-48`）。
   > **为什么**：MoE 层本身已经让 config 复杂了一档，逐层 attention 类型 × 隔层 MoE
   > 组合起来会让「哪层是什么」变得很难推理。Qwen3-MoE 选择只保留其中一个旋钮。
2. **`ForCausalLM` 换了一条继承线**：不是 Qwen 系，而是 **Mixtral 系**
   （`MixtralForCausalLM`）。这带来 `router_aux_loss_coef` / `num_experts` /
   `num_experts_per_tok` 三个属性和整段 aux loss 逻辑。
3. **输出对象换了类型**：`MoeCausalLMOutputWithPast` 比 `CausalLMOutputWithPast` 多两个字段
   （`aux_loss`、`router_logits`）。`Qwen3MoeModel.forward` 返回 `MoeModelOutputWithPast`
   （`modeling_qwen3_moe.py:508`），注释里写明「only diff with Mistral is the output type」。
4. **多了一个训练期 loss**：`output_router_logits=True` 且传了 `labels` 时，
   `loss += router_aux_loss_coef * aux_loss`（`modeling_qwen3_moe.py:667-674`）。
   dense Qwen3 没有这一项。

**保留的部分（也很重要）**：QK-Norm 保留了，RoPE 实现一致，RMSNorm 一致，
DecoderLayer 的残差结构逐行一致（`modeling_qwen3_moe.py:319-348`）。
**所以「Qwen3-MoE = Qwen3 + MoE」这个说法在骨架层面成立，在 config 和输出层面不成立。**

---

## 八、MoE 的工程权衡

### 8.1 训练：为什么负载均衡损失是必需的

**问题：专家坍缩（expert collapse）。** Router 是学出来的。如果某几个专家早期碰巧
拿到稍好的梯度，它们会被更多 token 选中 → 拿到更多梯度 → 更强 → 更被选中。
正反馈下，256 个专家可能只有 3~5 个真正在工作，其余是「死专家」——
你付了 256 份显存，只用到 5 份算力。

**解法：加一个惩罚不均衡的辅助损失。** 实现就是 `load_balancing_loss_func`，
Mixtral 和 Qwen3-MoE **共用同一份代码**（Mixtral 的是源头 `modular_mixtral.py:53-119`，
Qwen3-MoE 的实体在 `modeling_qwen3_moe.py:514`，逐行一致）。

Switch Transformer 的公式（(4)-(6) 式，docstring 在 `modular_mixtral.py:60-63`）：

```
loss = num_experts · Σ_i ( f_i · P_i )
        f_i = 分给专家 i 的 token 比例   （硬计数）
        P_i = 专家 i 的平均路由概率       （软概率）
```

`f_i · P_i` 这个乘积是关键：它在 **i 的分配比例** 和 **i 的平均置信度** 之间做惩罚。
两个都均匀分布时取最小值 `1/num_experts`，乘 `num_experts` 归一化到 1。
所以**完美均衡时 loss = 1，完全坍缩时 loss → num_experts**。

实现上的一个细节（`modular_mixtral.py:82-84`，这是近期优化过的写法）：

```python
# Accumulate assignment counts and probability sums layer by layer, normalizing at the end,
# so peak memory stays O(seq_len * num_experts) regardless of the number of layers.
```

即**逐层累加、最后一次性归一化**，而不是「每层算一份 loss 再平均」——
把峰值显存从 `O(L · S · E)` 降到 `O(S · E)`。
对 Qwen3-MoE（24 层 × 128 专家）这种规模不是小事。

**最终的 loss**（`modeling_qwen3_moe.py:667-674`）：

```python
aux_loss = load_balancing_loss_func(
    outputs.router_logits, self.num_experts, self.num_experts_per_tok, attention_mask,
)
if labels is not None:
    loss += self.router_aux_loss_coef * aux_loss.to(loss.device)
```

注意 `router_aux_loss_coef = 0.001`（三家都一样）——**惩罚很轻**，是「轻轻拽一下」而不是主损失。

> **DeepSeek 用的是另一条路**：V3 用 `e_score_correction_bias`
> （`modular_deepseek_v3.py:93`）—— 直接给每个专家的**选择分**加一个可调的偏置，
> 谁被选多了就把它的偏置往下压。这是**无辅助损失**（auxiliary-loss-free）的均衡策略，
> 好处是不用担心 aux loss 和主 loss 打架。
>
> **实测确认**：transformers 里的 DeepSeek V2/V3 **完全没有实现 aux loss**——
> ```bash
> $ grep -rn "load_balancing_loss_func\|router_aux_loss_coef" \
>        src/transformers/models/deepseek_v3/ src/transformers/models/deepseek_v2/
> （无输出）
> ```
> 它们的 config 里连 `router_aux_loss_coef` / `output_router_logits` 这两个字段都没有。
> 所以「三家都有 aux loss」是错的：**Mixtral 和 Qwen3-MoE 有，DeepSeek 没有**。
> 这是本篇最容易被凭记忆写错的一条，务必记住。

### 8.2 推理：显存 vs 计算

MoE 的核心账本是「**总参数 vs 激活参数**」。用 Qwen3-MoE 的默认 config
（`hidden=2048, L=24, E=128, I=768`）一阶估算：

| 项 | 参数量 | 占比 |
|----|--------|------|
| 128 个专家的全部权重 | ≈ 14.5 B | 93% |
| 其中**每 token 真正用到的**（top-8） | ≈ **0.91 B** | 5.8% |
| Attention（q/k/v/o） | ≈ 0.45 B | 2.9% |
| Embedding + lm_head（untied） | ≈ 0.62 B | 4.0% |
| **总计** | **≈ 15.6 B** | 100% |
| **每 token 激活** | **≈ 2.0 B** | — |

这与 checkpoint 名 `Qwen/Qwen3-MoE-15B-A2B`（`modeling_qwen3_moe.py:628`）
自洽：**15B 总参、2B 激活**。同一套算术在 Mixtral 上是 8 专家 top-2 → 25% 激活，
在 DeepSeek V3 上是 (8 路由 + 1 共享) / (256 路由 + 1 共享) ≈ 3.5%，激活比例更低。

**推理的三个后果**：

1. **显存按总参数算，算力按激活参数算**。这是 MoE 的整个商业逻辑：
   同样的 FLOPs 预算下能塞进更多参数（更多知识），代价是显存。
   所以 MoE 在**显存充足、算力受限**的场景（批处理推理、数据中心）最划算。
2. **专家并行（EP）**：单卡装不下 128 个专家时，把专家切到不同卡上。
   Qwen3-MoE 的 config 直接准备好了两套计划
   （`configuration_qwen3_moe.py:56-78`）：
   ```python
   base_model_tp_plan = { ..., "layers.*.mlp.experts": "moe_tp_experts", ... }
   # Expert-only EP plan: only shards MoE experts, not attention.
   base_model_ep_plan = {
       "layers.*.mlp.gate": "ep_router",
       "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
       "layers.*.mlp.experts.down_proj": "grouped_gemm",
       "layers.*.mlp.experts": "moe_tp_experts",
   }
   ```
   这两个名字（`moe_tp_experts` / `ep_router` / `grouped_gemm`）注册在
   `distributed/tensor_parallel.py:773` 附近：
   ```python
   "grouped_gemm": MoEParamShard(Shard(0), shards_expert_dim=True),
   "ep_router": EpRouterParallel(),
   "moe_tp_experts": MoeExpertsParallel(),        # ← 类定义在 :586
   ```
   > **注意注释里的设计考量**（`configuration_qwen3_moe.py:70-72`）：
   > 「EP 只切专家、不切注意力，因为注意力不切才能让 EP 规模突破 `num_kv_heads` 的限制
   > （Qwen3-30B 只有 4 个 KV head）」。这是一条很典型的 MoE 并行实践。
   > **细节留到 [2A6](./2A6_weights_quant_parallel.md)**（权重加载、量化与并行）。
3. **Expert offloading**：专家权重放 CPU / NVMe，用时按需拉。因为每个 token 只碰
   8/128 个专家，**专家访问有很强的局部性**，offload 的收益比 dense 模型大得多。
   代价是延迟——这也是 MoE 服务端常见的「吞吐换延迟」调优点。

### 8.3 为什么现在的旗舰模型几乎都是 MoE

把 8.2 的账本反过来读：**dense 模型要提升能力，必须同步提升 FLOPs；MoE 可以把
「参数（知识容量）」和「FLOPs（推理成本）」解耦。**

| | dense | MoE |
|---|---|---|
| 参数 ↑ | FLOPs ↑ 同比例 | FLOPs 基本不变（top-k 固定） |
| 显存 | 随参数 | 随参数（**不变差**） |
| 训练稳定性 | 简单 | 需要负载均衡 / 路由技巧 |
| 推理延迟 | 好预测 | 有路由开销 + 负载不均风险（长尾专家拖慢整批） |
| 适合场景 | 端侧、低显存 | 数据中心、高吞吐 |

三家的最终选择也体现了这个权衡的**不同落点**：

- **Mixtral**（2023）：证明 MoE 可行，但设计保守——8 个整宽专家、top-2、
  全层 MoE、无共享专家。**激活比例 25%，是最"重"的 MoE。**
- **DeepSeek V3**（2024）：把它推到极致——256 个细粒度专家 + 1 个共享专家、
  top-8、sigmoid 打分、分组路由、无辅助损失均衡。**激活比例 ≈ 3.5%。**
- **Qwen3-MoE**（2025）：**取中间路线**——128 个细粒度专家、top-8、
  softmax 打分（不用 sigmoid）、**取消共享专家**（相对 Qwen2-MoE 的回退）。
  激活比例 6.25%。**保留了 `decoder_sparse_step` 这个旋钮但默认不用。**

---

## 三家 MoE 横向对比

**所有数字来自各自的 `configuration_*.py`，逐行实测，非记忆。**

| 维度 | Mixtral | DeepSeek V3 | Qwen3-MoE |
|------|---------|-------------|-----------|
| config 文件 | `configuration_mixtral.py` | `configuration_deepseek_v3.py` | `configuration_qwen3_moe.py` |
| **专家数量** | **8**（`num_local_experts`，`:87`） | **256**（`n_routed_experts`，`:91`） | **128**（`num_experts`，`:105`） |
| **每 token 激活** | **2**（`:86`） | **8**（`:100`） | **8**（`:104`） |
| 激活比例 | 25%（2/8） | ≈ 3.5%（9/257） | 6.25%（8/128） |
| **共享专家** | **无** | **1 个**（`n_shared_experts`，`:90`） | **无**（config 里无 `shared_expert_*` 字段） |
| 专家中间维 | `intermediate_size` = 14336（`:71`） | `moe_intermediate_size` = 2048（`:86`） | `moe_intermediate_size` = 768（`:103`） |
| **路由粒度**（专家宽 / dense 宽） | **1.0×（粗）** | **1/9 ≈ 0.11×** | **1/8 = 0.125×** |
| 隐藏维 / 层数 | 4096 / 32 | 7168 / 61 | 2048 / 24 |
| **打分函数** | **softmax** | **sigmoid**（`DeepseekV3TopkRouter.forward`，`modular_deepseek_v3.py:104`） | **softmax** |
| **归一化** | **无条件** | `norm_topk_prob` = **`True`**（`:102`） | `norm_topk_prob` = **`False`**（`:106`） |
| 额外缩放 | 无 | `routed_scaling_factor` = **2.5**（`:92`） | 无 |
| 选专家前分组 | 否 | **是**：8 组选 4 组（`:98` `n_group`、`:99` `topk_group`） | 否 |
| 路由修正 bias | 无 | **有**（`e_score_correction_bias`） | 无 |
| **dense / MoE 混合** | **全层 MoE**（无旋钮） | `first_k_dense_replace` = **3**（`:101`） | `decoder_sparse_step` = **1**（`:102`），即默认全层 MoE |
| MLP 白名单 | 无 | 无 | `mlp_only_layers` = `[]`（`:109`/`:116`） |
| 负载均衡 | aux loss（`router_aux_loss_coef`=0.001） | **无 aux loss**（config 里连字段都没有），只靠 `e_score_correction_bias` 做无辅助损失均衡 | aux loss（`router_aux_loss_coef`=0.001） |
| **Experts 类** | `MixtralExperts`（**根**） | `DeepseekV3Experts`（→ `Qwen2MoeExperts` → `MixtralExperts`） | `Qwen3MoeExperts`（**`pass`** → `Qwen2MoeExperts`） |
| **Router 类** | `MixtralTopKRouter`（**根**） | `DeepseekV3TopkRouter`（→ `DeepseekV2TopkRouter` → `Qwen2MoeTopKRouter`） | `Qwen3MoeTopKRouter`（**`pass`** → `Qwen2MoeTopKRouter`） |
| MoE 块类 | `MixtralSparseMoeBlock`（`modular:180`） | `DeepseekV3MoE`（`modular:129`） | `Qwen3MoeSparseMoeBlock`（`modular:63`） |

### 关于「路由粒度」这一行的一处诚实修正

Roadmap 的草案表里把 Qwen3-MoE 的粒度标为「中」、DeepSeek 标为「细」。**实测数据不完全支持这个分级**：

- DeepSeek V3：2048 / 18432 = **1/9 ≈ 0.111**
- Qwen3-MoE：768 / 6144 = **1/8 = 0.125** ← 值和 DeepSeek 几乎一样

按「单个专家的宽度相对 dense MLP」这个标准，**两者是同一档（细粒度）**，
Mixtral 单独一档（1.0×，粗粒度）。真正区分 DeepSeek 与 Qwen3-MoE 的是
**专家总数（256 vs 128）** 和 **共享专家（有 vs 无）**，而不是粒度。
把 Qwen3-MoE 叫「中」会掩盖「它也是细粒度 MoE」这个事实——**建议按本表的数据理解**，
把三分法改成两档：`Mixtral = 粗` / `DeepSeek & Qwen3-MoE = 细`。

### 三个「取舍」的一句话总结

| 模型 | 核心取舍 |
|------|----------|
| **Mixtral** | 用最简单的 router（softmax + 无条件归一化）和最大的专家，**换取实现的确定性**。适合作为教学第一课 |
| **DeepSeek V3** | 把每个旋钮都推到极致（256 专家 / sigmoid / 分组 / bias 均衡），**换取最低的激活比例**——用最少的算力激活最多的参数 |
| **Qwen3-MoE** | 走中间：softmax（而非 sigmoid）+ 128 专家 + **取消共享专家**，但把 `decoder_sparse_step` / `mlp_only_layers` 留作旋钮。**换取的是可配置性和实现的简单性** |

---

## 一句话总结

> **三家的 router 完全不同（softmax/sigmoid、归一化开关、分组、缩放），
> 三家的专家计算完全相同（同一个 `MixtralExperts.forward`）。**
>
> 前者是**训练超参**，改几行就能切；后者是**共享实现**，
> 靠 `MixtralExperts → Qwen2MoeExperts → Qwen3MoeExperts` 这条继承链串起 35 个 Experts 类。
>
> 所以：**读 MoE 模型时，router 要逐个读、专家只需要读一次。**
> 看到 `pass` 就顺着继承链往上跳——`Qwen3MoeExperts` 是 `pass` 意味着「专家和 Mixtral 一样」，
> `Qwen3MoeTopKRouter` 是 `pass` 意味着「路由和 Qwen2-MoE 一样」。
> 这两句话就是本篇的全部内容。

---

## 验证清单

以下命令均在仓库根目录 `/Users/baai/Desktop/baai_proj/transformers` 实际执行过。

### 1. 类名与真实行号

`grep -n "^class \|^def " <file>` 的摘录：

| 文件 | 类 : 行号 |
|------|-----------|
| `qwen3_moe/modular_qwen3_moe.py`（手写，205 行） | `Qwen3MoeAttention`:44 · `Qwen3MoeMLP`:51 · **`Qwen3MoeExperts`:55** · **`Qwen3MoeTopKRouter`:59** · `Qwen3MoeSparseMoeBlock`:63 · `Qwen3MoeRMSNorm`:77 · `Qwen3MoeDecoderLayer`:81 · `Qwen3MoePreTrainedModel`:85 · `Qwen3MoeModel`:93 · `Qwen3MoeForCausalLM`:97 |
| `qwen3_moe/modeling_qwen3_moe.py`（生成，706 行） | `Qwen3MoeAttention`:122 · `Qwen3MoeMLP`:193 · **`Qwen3MoeExperts`:210**（`@use_experts_implementation` 在 :209）· **`Qwen3MoeTopKRouter`:249** · `Qwen3MoeSparseMoeBlock`:270 · `Qwen3MoeRMSNorm`:285 · `Qwen3MoeDecoderLayer`:305 · `Qwen3MoePreTrainedModel`:352 · `Qwen3MoeModel`:439 · `load_balancing_loss_func`:514 · `Qwen3MoeForCausalLM`:584 |

其余被本篇引用的行号（同一轮核对，原样摘录输出）：

```bash
$ grep -n "only diff with Mistral" src/transformers/models/qwen3_moe/modeling_qwen3_moe.py
508:        return MoeModelOutputWithPast(  # only diff with Mistral is the output type, we need MoE

$ grep -n "load_balancing_loss_func(\|router_aux_loss_coef \* aux_loss" \
       src/transformers/models/qwen3_moe/modeling_qwen3_moe.py
667:            aux_loss = load_balancing_loss_func(
674:                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)

$ grep -n "Qwen3-MoE-15B-A2B" src/transformers/models/qwen3_moe/modeling_qwen3_moe.py
628:        >>> model = Qwen3MoeForCausalLM.from_pretrained("Qwen/Qwen3-MoE-15B-A2B")

$ grep -n "def _init_weights" src/transformers/models/qwen3_moe/modeling_qwen3_moe.py \
                                  src/transformers/models/mixtral/modular_mixtral.py
qwen3_moe/modeling_qwen3_moe.py:371:    def _init_weights(self, module):
mixtral/modular_mixtral.py:257:    def _init_weights(self, module):

$ grep -n "def forward" src/transformers/models/mixtral/modular_mixtral.py | head -5
135:    def forward(          # ← MixtralExperts.forward（Qwen3MoeExperts 用的就是它）
170:    def forward(self, hidden_states):          # MixtralTopKRouter
188:    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:   # MixtralSparseMoeBlock

$ grep -n "e_score_correction_bias = nn.Buffer" src/transformers/models/deepseek_v3/modular_deepseek_v3.py
93:        self.e_score_correction_bias = nn.Buffer(torch.zeros(self.num_experts))

$ grep -n "first_k_dense_replace" src/transformers/models/deepseek_v2/modular_deepseek_v2.py
118:    first_k_dense_replace: int = 0
391:        self.mlp = DeepseekV2Moe(config) if layer_idx >= config.first_k_dense_replace else DeepseekV2MLP(config)

$ grep -n "self.mlp = MixtralSparseMoeBlock" src/transformers/models/mixtral/modular_mixtral.py
218:        self.mlp = MixtralSparseMoeBlock(config)    # Mixtral 全层 MoE，无开关

$ grep -n '"grouped_gemm"\|"ep_router"\|"moe_tp_experts"' src/transformers/distributed/tensor_parallel.py
770:            "grouped_gemm": MoEParamShard(Shard(0), shards_expert_dim=True),
771:            "ep_router": EpRouterParallel(),
773:            "moe_tp_experts": MoeExpertsParallel(),     # 类定义在 :586
```

### 2. 三家 config 的 MoE 超参默认值

```bash
$ grep -n "decoder_sparse_step\|mlp_only_layers\|norm_topk_prob\|num_experts\|moe_intermediate_size\|router_aux_loss_coef\|output_router_logits" \
       src/transformers/models/qwen3_moe/configuration_qwen3_moe.py
102:    decoder_sparse_step: int = 1
103:    moe_intermediate_size: int = 768
104:    num_experts_per_tok: int = 8
105:    num_experts: int = 128
106:    norm_topk_prob: bool = False
107:    output_router_logits: bool = False
108:    router_aux_loss_coef: float = 0.001
109:    mlp_only_layers: list[int] | None = None
116:        self.mlp_only_layers = [] if self.mlp_only_layers is None else self.mlp_only_layers

$ sed -n '86,92p' src/transformers/models/deepseek_v3/configuration_deepseek_v3.py
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 61
    num_attention_heads: int = 128
    num_key_value_heads: int | None = 128
    n_shared_experts: int = 1
    n_routed_experts: int = 256
    routed_scaling_factor: float = 2.5
# 另经 grep 确认：:98 n_group=8 / :99 topk_group=4 / :100 num_experts_per_tok=8
#                :101 first_k_dense_replace=3 / :102 norm_topk_prob=True

$ sed -n '86,90p' src/transformers/models/mixtral/configuration_mixtral.py
    num_experts_per_tok: int = 2
    num_local_experts: int = 8
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    router_jitter_noise: float = 0.0
```

> **一点诚实说明**：本机环境缺 `regex` 依赖，**无法** `import transformers` 来实例化 config
> 打印属性（`python3 -c "import transformers"` → `ModuleNotFoundError: No module named 'regex'`，
> 加 `sys.path.insert(0,'src')` 后同样在 `regex` 处失败）。
> 因此以上默认值全部**直接取自源码字段声明**，并交叉核对了 `__post_init__`
> （`configuration_qwen3_moe.py:114-117`：`mlp_only_layers` 的 `None` 会在 `__post_init__`
> 里被转成 `[]`；`sliding_window` 在 `use_sliding_window=False` 时被置 `None`）。
> 结论：表中数字是**类默认值**，不是某个具体 checkpoint 的 `config.json` 值。

### 3. `shared_expert_intermediate_size` 在 Qwen3-MoE 中确实不存在

```bash
$ grep -rn "shared_expert" src/transformers/models/qwen3_moe/
（无输出 → 不存在）

$ grep -n "shared_expert_intermediate_size" src/transformers/models/qwen2_moe/configuration_qwen2_moe.py
29:    shared_expert_intermediate_size (`int`, *optional*, defaults to 5632):
96:    shared_expert_intermediate_size: int = 5632
```

### 4. DeepSeek V2/V3 **没有** aux loss（本篇最容易凭记忆写错的一条）

```bash
$ grep -rn "load_balancing_loss_func\|router_aux_loss_coef" \
       src/transformers/models/deepseek_v3/ src/transformers/models/deepseek_v2/
（无输出；退出码 1）
# 对照：Mixtral 与 Qwen3-MoE 有
$ grep -c "load_balancing_loss_func" src/transformers/models/mixtral/modular_mixtral.py
1
$ grep -c "load_balancing_loss_func" src/transformers/models/qwen3_moe/modeling_qwen3_moe.py
1
```

### 5. `integrations/moe.py` 里**没有** `Experts` 基类

```bash
$ grep -rn "^class Experts" src/transformers/
src/transformers/integrations/moe.py:481:class ExpertsInterface(GeneralInterface):
# ↑ 唯一以 `class Experts` 开头的仍是一个「接口 / 注册表」类，不是专家实现基类

$ grep -n "^class" src/transformers/integrations/moe.py
481:class ExpertsInterface(GeneralInterface):
# ↑ 整个 582 行文件里只有一个类定义

$ grep -n "class Experts(torch.nn.Module)" src/transformers/integrations/moe.py
39:# class Experts(torch.nn.Module):
# ↑ 带 `#` 前缀，是注释（一份参考实现），不是可导入的代码
```

### 6. 完整 MoE 继承图（脚本解析，非人工）

```python
# 解析 src/transformers/models/*/modular_*.py 里所有 `class X(Y)` 关系，
# 只保留名字含 Experts / Router / TopK 的类，取第一个在库内的基类。输出（节选）：
MixtralExperts:       direct=12  total_descendants=35
Qwen2MoeExperts:      direct=6   total_descendants=18
Qwen3MoeExperts:      direct=5   total_descendants=5
MixtralTopKRouter:    direct=6   total_descendants=8
Qwen2MoeTopKRouter:   direct=6   total_descendants=20
Qwen3MoeTopKRouter:   direct=1   total_descendants=1
DeepseekV2TopkRouter: direct=2   total_descendants=11
DeepseekV3Experts:    direct=5   total_descendants=5
# 同一脚本的全局统计：modular 文件里名字含 Experts / Router / Topk / TopK 的类共 100 个，
# 其中 40 个的类体只有一句 `pass`（含 Qwen3MoeExperts 与 Qwen3MoeTopKRouter）——
# 也就是「40% 的 MoE 组件类是零代码的转发声明」。
```

### 7. 任务里给出的两个例子（**均实测通过**）

```bash
$ grep -rn "^class DeepseekV2Experts\|^class DeepseekV4Experts" src/transformers/models/*/modular_*.py
deepseek_v2/modular_deepseek_v2.py:157:class DeepseekV2Experts(Qwen2MoeExperts):
deepseek_v4/modular_deepseek_v4.py:879:class DeepseekV4Experts(MixtralExperts):

$ grep -rn "^class Qwen3MoeExperts\|^class Qwen3MoeTopKRouter" src/transformers/models/qwen3_moe/modular_qwen3_moe.py
55:class Qwen3MoeExperts(Qwen2MoeExperts):
59:class Qwen3MoeTopKRouter(Qwen2MoeTopKRouter):
```

### 8. 两个 experts 的 `forward` 逐行相同

```bash
$ sed -n '223,247p' src/transformers/models/qwen3_moe/modeling_qwen3_moe.py   # Qwen3 版
$ sed -n '135,159p' src/transformers/models/mixtral/modular_mixtral.py        # Mixtral 版
# 对照结果：除缩进与类名外无差异。唯一不同的是 __init__ 的 3 行：
#   Qwen3:   self.num_experts = config.num_experts / self.intermediate_dim = config.moe_intermediate_size
#   Mixtral: self.num_experts = config.num_local_experts / self.intermediate_dim = config.intermediate_size
```

### 9. `attribute_map` 是活代码，不是注释

```bash
$ sed -n '477,484p' src/transformers/configuration_utils.py
def __setattr__(self, key, value):
    if key in super().__getattribute__("attribute_map"):
        key = super().__getattribute__("attribute_map")[key]
    super().__setattr__(key, value)

def __getattribute__(self, key):
    if key != "attribute_map" and key in super().__getattribute__("attribute_map"):
        key = super().__getattribute__("attribute_map")[key]
    return super().__getattribute__(key)
```

### 10. 参数量算术（一阶估算，用于核对 checkpoint 名）

```python
h, L, E, I, k = 2048, 24, 128, 768, 8       # Qwen3-MoE 默认 config
expert_total  = E * 3*h*I * L  → 14.50 B
expert_active = k * 3*h*I * L  →  0.91 B
attn          = (h*32*128 + 2*h*4*128 + 32*128*h) * L  → 0.45 B
embed*2       = 2048*151936*2  →  0.62 B
total ≈ 15.57 B   active ≈ 1.98 B
# 对照 checkpoint 名 "Qwen/Qwen3-MoE-15B-A2B"（modeling_qwen3_moe.py:628）→ 自洽
```

同时得到「激活宽度 ≈ dense 宽度」这个设计不变量：

```
Qwen3-MoE    8 × 768  = 6144  = intermediate_size 6144
Qwen2-MoE    4 × 1408 = 5632  = shared_expert_intermediate_size 5632
DeepSeek V3  (8+1) × 2048 = 18432 = intermediate_size 18432
Mixtral      2 × 14336 = 28672 = 2 × intermediate_size 14336（唯一例外，2 倍）
```

### 11. **未执行**的操作（按任务要求）

- 未运行 `make fix-repo` / `python utils/modular_model_converter.py`（会改写仓库）
- 未运行任何模型推理（config 实例化也未能进行，见第 2 条）
- 未修改除本文件外的任何文件（`git status --short` 只有 `?? reading_docs/`）

---

## 导航

- 路线图：[00_roadmap.md](./00_roadmap.md)（本篇 = 进阶 · 模型结构 · 2M3）
- 上一篇：[2M2_deepseek_mla_moe.md](./2M2_deepseek_mla_moe.md)
- 下一篇：[2M4_t5_encoder_decoder.md](./2M4_t5_encoder_decoder.md)
- **建议紧接着读**：
  - [2A6_weights_quant_parallel.md](./2A6_weights_quant_parallel.md)：
    本篇第 8.2 节的 `base_model_ep_plan`（`moe_tp_experts` / `ep_router` / `grouped_gemm`）
    在那里完整展开
  - [1M4_qwen3_dense.md](./1M4_qwen3_dense.md)：回来对照第 七 节的表，
    确认 dense Qwen3 的 `layer_types` 逐层机制是怎么被 Qwen3-MoE 削掉的
  - [1M3_mixtral_moe.md](./1M3_mixtral_moe.md)：本篇假设你已读过它的 MoE 基础，
    如果 `expert_mask.permute(2, 1, 0)` 那段看不懂，回去补
