# Mixtral 混合专家（MoE）精读

> 阅读目标：理解**稀疏激活（sparse activation）**。
> 这是本路线图里第一次遇到**真正的结构创新**——不是改超参（如 1M2 的 `sliding_window`），而是把一整块子模块换掉。
>
> Mixtral 对 Mistral 的改动**只有一处**：把每个 `DecoderLayer` 里的 `MLP` 换成 `MixtralSparseMoeBlock`。
> Attention / Norm / RoPE / 残差结构一字未动。
>
> 核心源文件：`src/transformers/models/mixtral/modular_mixtral.py`（436 行，**有 modular，读这个**）
> 生成文件：`src/transformers/models/mixtral/modeling_mixtral.py`（677 行，由 `make fix-repo` 生成，勿直接编辑）
> 配置：`src/transformers/models/mixtral/configuration_mixtral.py`（100 行，**手写**，不是生成的）

---

## 速读区（TL;DR）

**一句话结论**：Mixtral 相对 Mistral 只改了一处——把 `DecoderLayer` 里的 `MLP` 换成
`MixtralSparseMoeBlock`；MoE 换来的是「参数 46.70B、每 token 只用 12.88B（27.6%）」，
**只省计算、不省显存**。

| # | 断言 | 出处（文件:行号） |
|---|------|------------------|
| 1 | 唯一实质改动是 `self.mlp = MixtralSparseMoeBlock(config)`；Attention / Norm / RoPE / 残差一字未动 | `modular_mixtral.py:218` |
| 2 | 核心默认值：`num_experts_per_tok=2`、`num_local_experts=8`、`intermediate_size=14336`（**每个专家**的宽度）、`hidden_size=4096`、`num_hidden_layers=32` | `configuration_mixtral.py:68-91` |
| 3 | 与 Mistral 相比：`sliding_window` 默认 `None`（Mistral 是 4096）、`rms_norm_eps=1e-5`、RoPE theta `1000000.0` | `configuration_mixtral.py:84` / `:78` / `:44` |
| 4 | 参数 46.70B / 每 token 激活 12.88B → **27.6%**；专家小计 45.10B，`8x7B` 的「7B」= 单个专家串 32 层 ≈ 5.63B + 共享的 attention 与词嵌入 | 本篇 1.3 节（验证清单脚本实测） |
| 5 | router = `F.linear`（**无 bias**）+ `nn.Parameter[8, 4096]`（仅 32,768 参数）；`softmax(router_logits.float())` **显式升 float32** | `modular_mixtral.py:168` / `:173` |
| 6 | top-k 之后**无条件重新归一化**（`router_top_value /= router_top_value.sum(-1)`）——Mixtral 没有 `norm_topk_prob` 开关 | `modular_mixtral.py:175` |
| 7 | 专家就是 `LlamaMLP`，没有任何新数学；存储改成 3D 张量 `gate_up_proj [E, 2I, H]`、`down_proj [E, H, I]` | `modular_mixtral.py:153-155` |
| 8 | 老类名 **`MixtralBlockSparseTop2MLP` 已彻底消失**，现在叫 `MixtralExperts`（`nn.Module`，不是 `ModuleList`） | `modular_mixtral.py:123` |
| 9 | 循环只遍历 **`expert_hit`（命中的专家）**；`n_e` 是变长的；回填用 `index_add_`（一个 token 被 2 个专家各写一次，必须累加不能覆盖） | `modular_mixtral.py:145` / `:147` / `:157` |
| 10 | `:156` 的 `top_k_weights[token_idx, top_k_pos, None]` 是**逐元素配对索引**，不是外积 | `modular_mixtral.py:156` |
| 11 | `load_balancing_loss_func` **手写在 modular 里**（67 行），生成物 `modeling_mixtral.py:485` 只是副本 | `modular_mixtral.py:53` |
| 12 | aux loss **只在 `output_router_logits=True` 时才算**；系数 `router_aux_loss_coef=0.001` | `modular_mixtral.py:396` / `configuration_mixtral.py:89` |
| 13 | 这版 aux loss 下界是 `k`（均匀路由 L=2），不是论文里的 1——整体大 k 倍；全坍缩时 L=16 | 本篇 8.4 节 |
| 14 | 判断「手写 vs 自动拉取」：modular 里写了完整函数体 = 手写；只写 `pass` = 自动拉取（`MixtralRMSNorm` / `RotaryEmbedding` / `Attention` 都只有 `pass`） | `modular_mixtral.py:199` / `:203` / `:207` |

**最容易踩的坑 / 反直觉发现**：
- **MoE 不省显存，只省计算**：8x7B 的权重显存（bf16 约 93 GB）与 dense 46.7B 平手；单卡装不下 46.7B 时 MoE 一点忙都帮不上。
- **`view(-1, H)` 不可省**：路由是 **per-token** 的；合并 `B`、`S` 的副作用是 **MoE 对 padding 无感**，这也是 `load_balancing_loss_func` 必须处理 `attention_mask`（`:92-113`）的原因。
- **`:149-150` 的 `if expert_idx == self.num_experts: continue` 是不可达的防御分支**（`topk` 的上界就是 `E`），沿袭自「把无专家编码成哨兵值 `E`」的实现，别当 bug 改。
- **「`k == E` 就不用循环」这个捷径当前代码里不存在**：全仓库搜相关模式零命中，唯一分支是 `:149` 那个 `continue`；真正写进代码的优化是「只遍历命中的专家」。
- **`output_router_logits` 默认 `False`** ⇒ 默认配置下 `MixtralForCausalLM` **不会**自动平衡专家，训练 MoE 必须显式打开。
- **公共 docstring 那句「including load balancing loss and router z-loss」对 Mixtral 不成立**：Mixtral 没有 z-loss（config 里没有 `router_z_loss_coef`，输出类型里也没有该字段），它靠 softmax 前转 fp32（`:173`）稳数值。
- **`router_jitter_noise` 默认 `0.0`（默认不生效）**，官方训练主要靠辅助损失。

**读完应该能回答的问题**：
- Mixtral 相对 Mistral 到底改了哪一处？
- 46.70B / 12.88B / 27.6% 这三个数字分别是什么？
- 为什么 router 的 softmax 要显式升到 float32？
- `:149-150` 那个 `continue` 为什么不可达？
- 为什么 `load_balancing_loss_func` 要专门处理 `attention_mask`？
- 默认配置下训练 MoE 会自动做负载均衡吗？

---

## 前置知识

- [1M1_llama_structure.md](./1M1_llama_structure.md)：`LlamaMLP`（SwiGLU）、`LlamaDecoderLayer`（Pre-Norm + 残差）。
  **本篇最重要的锚点是那一句 `down_proj(act_fn(gate_proj(x)) * up_proj(x))`**——专家就是它。
- [1M2_mistral_sliding_window.md](./1M2_mistral_sliding_window.md)：Mixtral 直接继承 Mistral，
  连 `sliding_window` 都带过来了。
- [2A5_modular_code_generation.md](./2A5_modular_code_generation.md)：**读 modular 而不是生成文件**的规则。
  本篇会大量用到「这个类在 modular 里没写，是自动拉进来的」这条判断。

---

## 一、MoE 要解决的问题：加参数，但不加计算

### 1.1 dense 模型的死结

先看 [1M1](./1M1_llama_structure.md) 里 Llama 的 MLP（`modeling_llama.py:163-176`）：

```python
def forward(self, x):
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj
```

**每一个 token 都要过这同一个 MLP**。于是 dense 模型有一条铁律：

> 参数量 ∝ 每 token 的计算量（FLOPs）。

想让模型更强通常要加参数（加宽 `intermediate_size`、加深 `num_hidden_layers`），
但每一份新增参数都会被**所有** token 使用一遍 —— 训练/推理都按比例变慢。

### 1.2 MoE 的赌注

MoE 的赌注是：**同一份知识，不需要每个 token 都用同一套参数去处理**。

于是把一个大 MLP 拆成 `E` 个小 MLP（叫「专家」/ expert），每个 token **只挑 `k` 个专家**去算：

| | 参数量 | 每 token 计算量 |
|---|---|---|
| dense 扩到 46.7B | 46.7B | 全量（∝46.7B） |
| MoE 8 专家 / 激活 2 个 | **46.7B**（参数照存，显存照占） | **≈12.9B 的量级**（只算 2 个专家） |

一句话：**参数量（显存）按总量算，计算量（FLOPs）按激活量算**。
这就是「稀疏激活」——参数是稠密的，计算是稀疏的。

代价也随之而来：
- 显存不省（要装下全部专家）→ 见第十节。
- 需要一套「谁来选专家」的机制 → router，见第五节。
- 需要防止「所有 token 都挤到同一个专家」→ 负载均衡损失，见第八节。

### 1.3 Mixtral 的真实数字（实测自 `configuration_mixtral.py`）

先把 config 的默认值钉死。下面这些**不是网上的说法，是 AST 解析文件拿到的源码字面值**
（`configuration_mixtral.py:68-91`）：

```python
vocab_size: int = 32000                    # :68
hidden_size: int = 4096                    # :69
intermediate_size: int = 14336             # :70   ← 单个专家的 FFN 宽度
num_hidden_layers: int = 32                # :71
num_attention_heads: int = 32              # :72
num_key_value_heads: int = 8               # :73
num_experts_per_tok: int = 2               # :86   ← top-k，每 token 选 2 个专家
num_local_experts: int = 8                 # :87   ← 每层 8 个专家
router_aux_loss_coef: float = 0.001        # :89
router_jitter_noise: float = 0.0           # :90
output_router_logits: bool = False         # :88
sliding_window: int | None = None          # :84
tie_word_embeddings: bool = False          # :83
```

**注意两个容易记错的点**：

1. `num_experts_per_tok = 2` / `num_local_experts = 8` —— 不是 1/8，也不是「top-1 路由」。
2. `intermediate_size = 14336` 是**每个专家**的 FFN 宽度，和 Mistral 7B 的 dense MLP 宽度一样。
   也就是说 Mixtral **没有**把每个专家做小，是老老实实放了 8 个满宽的 MLP。

用这些真实数字算一遍参数量（脚本见文末验证清单）：

| 组成 | 每层 | ×32 层 |
|------|------|--------|
| 专家 `gate_up_proj`（8×2·I·H） | 939,524,096 | — |
| 专家 `down_proj`（8·H·I） | 469,762,048 | — |
| **专家小计** | **1,409,286,144** | **45,097,156,608 ≈ 45.10B** |
| Attention（q/k/v/o，GQA：k、v 头数是 8） | 41,943,040 | ≈ 1.34B |
| Router（8×4096） | 32,768 | ≈ 0.001B |
| RMSNorm ×3 | 12,288 | ≈ 0.0004B |
| `embed_tokens` + `lm_head`（tie=False，各算一份） | — | 2 × 131,072,000 ≈ 0.26B |
| **总计** | | **≈ 46.70B** |
| **每 token 激活**（专家只算 2/8） | | **≈ 12.88B** |

两个数字都能对上名字里的含义：

- **`8x7B` 里的 7B**：把**一个**专家在 32 层串起来 = 45.10B / 8 ≈ 5.63B，
  再加共享的 attention 1.34B 和词嵌入 0.26B ≈ **7.2B** ——「每个专家单独拿出来大约是个 7B 模型」。
- **激活 12.88B ≈ 13B**：`2 × 5.63B + 1.34B + 0.26B ≈ 12.9B`。

所以每一层的前向：

```
dense 46.7B 模型：  每 token 跑 46.70B 参数  → 100% 计算量
Mixtral 8x7B：      每 token 跑 12.88B 参数  → 27.6% 计算量（12.88 / 46.70）
```

**27.6%** 就是 MoE 换来的东西：质量上限按 46.7B 的模型看，账单按 12.9B 的模型付。

---

## 二、mixtral 的 modular 文件长什么样

因为 Mixtral **有** modular 文件，按 [2A5 的规则 1](./2A5_modular_code_generation.md)，我们读 `modular_mixtral.py`。
它的开头就是「我继承 Mistral」（`modular_mixtral.py:36-46`）：

```python
from ..mistral.modeling_mistral import (
    MistralAttention,
    MistralForCausalLM,
    MistralForQuestionAnswering,
    MistralForSequenceClassification,
    MistralForTokenClassification,
    MistralModel,
    MistralPreTrainedModel,
    MistralRMSNorm,
    MistralRotaryEmbedding,
)
from .configuration_mixtral import MixtralConfig
```

然后整个文件里一共 **13 个类 + 1 个模块级函数**，其中 **6 个类只写了 `pass`**：

| modular 行号 | 类 | 内容 | 含义 |
|---|---|---|---|
| `:53` | `load_balancing_loss_func`（函数） | 67 行完整实现 | **Mixtral 独有，手写在 modular 里** |
| `:123` | `MixtralExperts` | 完整实现 | 专家集合（3D 张量） |
| `:162` | `MixtralTopKRouter` | 完整实现 | router |
| `:180` | `MixtralSparseMoeBlock` | 17 行（forward 占 9 行） | **本篇主角** |
| `:199` | `MixtralRMSNorm(MistralRMSNorm)` | `pass` | 白送 |
| `:203` | `MixtralRotaryEmbedding(MistralRotaryEmbedding)` | `pass` | 白送 |
| `:207` | `MixtralAttention(MistralAttention)` | `pass` | 白送（滑动窗口注意力一起白送） |
| `:211` | `MixtralDecoderLayer` | 36 行 | 只换了 `self.mlp` |
| `:249` | `MixtralPreTrainedModel(MistralPreTrainedModel)` | 16 行 | 加 router_logits 采集 |
| `:267` | `MixtralModel(MistralModel)` | 54 行 | 只覆盖 forward（换输出类型） |
| `:323` | `MixtralForCausalLM(MistralForCausalLM)` | 92 行 | 加 aux loss |
| `:417/421/425` | 三个分类头 | `pass` | 白送 |

**读法结论**：436 行里，真正属于「MoE 这件事」的只有 `MixtralExperts` + `MixtralTopKRouter` +
`MixtralSparseMoeBlock` + `load_balancing_loss_func` ≈ **140 行**。剩下的都是 `pass`、
继承赋值、和为了塞下 MoE 而必须覆盖的样板。

顺带一个观察：**Mixtral 是 MoE 家族的「Llama」**。全库有 **9 个别的模型**在 modular 里
`class XxxPreTrainedModel(MixtralPreTrainedModel)`（实测 `grep -rn "class .*PreTrainedModel(MixtralPreTrainedModel)"`）：
qwen2_moe、qwen3_moe、phimoe、jetmoe、minimax、minimax_m2、ernie4_5_moe、flex_olmo、deepseek_v4。
`qwen3_moe` 就是直接 `class Qwen3MoeModel(MixtralModel)`（`modular_qwen3_moe.py:93`）。
这和 1M1 里「Llama 被别人继承，自己不继承任何人」的位置完全对称。

---

## 三、MoE 前向数据流图（对照 dense MLP）

```
【dense MLP（Llama 的 LlamaMLP）】  —— 所有 token 共享同一套参数

  x [B, S, H]
      │  （没有选择，全员入场）
      ▼
  ┌───────────────────────────────┐
  │ gate_proj → SiLU ────┐        │
  │                      × → down_proj
  │ up_proj  ────────────┘        │
  └───────────────┬───────────────┘
                  ▼
              y [B, S, H]
  参数量 3·H·I      计算量 3·H·I·T      ← 两者 1:1


【Mixtral MoE】  —— 每个 token 各自挑 k=2 个专家

  x [B, S, H]
      │  view(-1, H)                    modular_mixtral.py:192
      ▼
  tokens [T, H]        T = B × S
      │
      ▼
  ┌──────────────────────────┐
  │ router / gate            │  一个 Linear：H → E(=8)，无 bias     :168
  │   F.linear(x, weight)    │                                       :172
  └──────────┬───────────────┘
             │  router_logits [T, 8]
             ▼
     softmax(logits.float(), -1)  →  probs [T, 8]                     :173
             │
             ▼
     torch.topk(probs, k=2, -1)                                       :174
             │
     ┌───────┴────────────────────────┐
     ▼                                ▼
  top_k_index [T, 2]           top_k_weights [T, 2]
  「选了哪 2 个」                「每个的权重」
                                 再归一化 /= sum(-1, keepdim)         :175
     │                                │
     └──────────────┬─────────────────┘
                    ▼
   ┌────────────────────────────────────────────────────┐
   │ for e in 命中的专家（最多 min(E, T×k) 个）：         │   :147
   │     tok = 选中 e 的那些 token 的下标                 │   :151
   │     只把 x[tok] 这一小撮行送进专家 e（SwiGLU MLP）   │   :152-155
   │     out[tok] += weight · e(x[tok])                  │   :156-157
   └──────────────────────┬─────────────────────────────┘
                          ▼
                    y [T, H]  →  reshape → [B, S, H]                 :195

  参数量 8 × 3·H·I     计算量 2 × 3·H·I·T      ← 两者差 4 倍（= E/k）
```

把两张图叠起来看，MoE 相对 dense 只多了**中间那一段**：

```
   dense:  x ──────────────────────────────────────────► MLP ──► y
   MoE:    x ──► [router → top-k → 分组重排] ──► 2 个 MLP ──► 加权 ──► y
                  └── 有参数、要训练        └── 有参数、要训练
                  └── 计算量极小（H×E 的矩阵）  └── 只有 2/8 被触发
```

**一句话读懂这张图**：MoE 在 MLP 前面插了一个「分诊台」，让大部分 token 走不同的诊室。
MLP 本身没变——这一点在第六节会用代码确认。

---

## 四、核心精读 1：`MixtralSparseMoeBlock`

这是本篇的主角，也是 [1M1](./1M1_llama_structure.md) 里 `LlamaMLP` 在 Mixtral 里的**替代品**。
`MixtralDecoderLayer` 里就是这一行换掉了（`modular_mixtral.py:218`）：

```python
self.mlp = MixtralSparseMoeBlock(config)      # Mistral/Llama 这里是 MistralMLP(config)
```

完整代码（`modular_mixtral.py:180-196`，**逐字**）：

```python
class MixtralSparseMoeBlock(nn.Module):                                    # :180
    def __init__(self, config):                                            # :181
        super().__init__()
        self.top_k = config.num_experts_per_tok                            # :183  → 2
        self.jitter_noise = config.router_jitter_noise                     # :184  → 0.0
        self.gate = MixtralTopKRouter(config)                              # :185  router
        self.experts = MixtralExperts(config)                              # :186  8 个专家

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:        # :188
        batch_size, sequence_length, hidden_dim = hidden_states.shape      # :189
        if self.training and self.jitter_noise > 0:                        # :190
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])    # :192
        _, top_k_weights, top_k_index = self.gate(hidden_states)           # :193
        hidden_states = self.experts(hidden_states, top_k_index, top_k_weights)  # :194
        hidden_states = hidden_states.reshape(batch_size, sequence_length, hidden_dim)  # :195
        return hidden_states                                               # :196
```

`forward` 只有 9 行，但每一行都有信息量：

| 行 | 做了什么 | 为什么 |
|----|---------|--------|
| `:189` | 记下 `[B, S, H]` | 待会儿要 reshape 回来 |
| `:190-191` | 训练时给输入乘 `uniform(0.9, 1.1)` 的噪声 | jitter：扰动输入让 router 不至于过早锁定某几个专家（默认关闭，`router_jitter_noise=0.0`） |
| `:192` | **压平成 `[B*S, H]`** | **MoE 的选择是 per-token 的**，必须把 batch 和 seq 两个维度合并成一维才能按 token 路由。详见第七节 |
| `:193` | 过 router，拿到权重和专家下标 | 注意返回顺序被解包成 `(weights, index)` —— 而 router 返回的是 `(logits, scores, indices)`，所以 `_` 丢掉的正是 `router_logits`（它有别的用途，见第八节） |
| `:194` | 把 token 分发给专家 | 真正的稀疏计算在这里 |
| `:195` | reshape 回 `[B, S, H]` | 保持和其它子模块（attention）一致的形状契约 |

**`:193` 那个下划线 `_` 值得停一下**：`router_logits` 在 forward 里被丢弃了，
但它并没有消失——它被 hook 抓走了（`MixtralPreTrainedModel._can_record_outputs`，
`modular_mixtral.py:251`），专门留给辅助损失用。这是第八节的主题。

`MixtralSparseMoeBlock` 自己的 forward 里**没有任何 `for` 循环**：
循环在 `MixtralExperts.forward` 里（第六节），top-k 选择在 router 里（第五节）。
这三块职责切得很干净：

```
MixtralSparseMoeBlock  → 形状管理 + 训练期噪声（"我负责把 token 摆好"）
  ├─ MixtralTopKRouter → 选谁 + 给权重        （"我负责分诊"）
  └─ MixtralExperts    → 分组计算 + 加权求和   （"我负责干活"）
```

---

## 五、核心精读 2：`MixtralTopKRouter`（router）

### 5.1 router 就是一个线性层

`modular_mixtral.py:162-177`（逐字）：

```python
class MixtralTopKRouter(nn.Module):                                        # :162
    def __init__(self, config):                                            # :163
        super().__init__()
        self.top_k = config.num_experts_per_tok                            # :165  k = 2
        self.num_experts = config.num_local_experts                        # :166  E = 8
        self.hidden_dim = config.hidden_size                               # :167  H = 4096
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))  # :168  [8, 4096]

    def forward(self, hidden_states):                                      # :170
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)         # :171
        router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)   :172
        router_probs = torch.nn.functional.softmax(router_logits.float(), dim=-1)       # :173
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1) # :174
        router_top_value /= router_top_value.sum(dim=-1, keepdim=True)     # :175
        router_scores = router_top_value                                   # :176
        return router_logits, router_scores, router_indices                # :177
```

**这就是 MoE 的全部「智能」**：

| 行 | 数学 | 说明 |
|----|------|------|
| `:168` | `W ∈ R^{E×H}` | 只有 `8 × 4096 = 32,768` 个参数。对比专家的 1.4B——**router 小到可以忽略** |
| `:172` | `logits = x Wᵀ`，形状 `[T, 8]` | 一个**无 bias** 的线性层。注意它是 `F.linear` 而不是 `nn.Linear`：没有 `bias`、没有 `config`，纯粹是个矩阵乘 |
| `:173` | `softmax(logits.float())` | **先转 float32 再 softmax**，见 5.3 |
| `:174` | `torch.topk(probs, 2)` | 每个 token 挑概率最大的 2 个专家 |
| `:175` | `v /= v.sum(-1, keepdim=True)` | **top-k 之后再归一化**，见 5.2 |
| `:177` | 返回 | `(logits, scores, indices)`，三样都要 |

对比 [1M1](./1M1_llama_structure.md) 里的 `LlamaMLP`：Llama 是 3 个 `nn.Linear` 干一件确定的事；
router 是 1 个 `F.linear` 干一件**不确定**的事（用户输入什么它就路由到哪）。这就是「学习出来的路由」的含义。

### 5.2 为什么 softmax 之后要再归一化一次

`softmax` 天然满足 `Σ_i p_i = 1`。但 `topk` 只保留了 2 个，**这 2 个的和不再是 1**（一般 < 1）。
`:175` 把它们除回来，使 `w₁ + w₂ = 1`。

两种做法的取舍（这就是后面的「三种 MoE 对比」要展开的点之一）：

| 做法 | 效果 | 谁在用 |
|------|------|--------|
| **top-k 后重新归一化**（重新分配） | 权重和恒为 1，输出尺度稳定 | **Mixtral**（`:175` 无条件执行） |
| 不归一化（保留原始概率） | 权重和 < 1，未被选中的概率被「浪费」掉 | qwen2_moe 系列（由 `norm_topk_prob` 开关控制） |

`qwen2_moe` 的对照写法很直观（`modular_qwen2_moe.py:104-105`）：

```python
if self.norm_topk_prob:
    router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
```

Mixtral 没有这个开关——它**永远**归一化。这个细节在 [2M3_qwen3_moe.md](./2M3_qwen3_moe.md) 会再次遇到。

### 5.3 为什么 softmax 用 float32

`:173` 写的是 `softmax(router_logits.float(), dim=-1)` —— 显式把 logits 升到 float32。

原因：router logits 是**学出来的**，训练久了数值范围会拉大；
而且第八节的辅助损失要拿这批概率去算一个很小的数（乘 0.001 的系数），
bf16 下 softmax 的精度损失会让辅助损失失真。这是 MoE 实现里少数几个「精度必须小心」的地方。

但**计算完的权重会回到原 dtype**——看 `:194` 传给 experts 的 `top_k_weights`，
以及 `MixtralExperts` 最后一行 `:157` 的 `.to(final_hidden_states.dtype)`。省显存、保精度，两头都要。

### 5.4 一个常见的误解

> router 不是「每个专家算一遍再挑最好的」。

它是**先算权重、再挑**：`x @ Wᵀ` 一次性给出所有 8 个专家的分数（成本只有 `T×H×E`），
排序取前 2，然后**才**去看专家本身。顺序上「分诊」完全发生在「看病」之前。

---

## 六、核心精读 3：`MixtralExperts`——专家就是 MLP

### 6.1 类名的实测结果（老类名 `MixtralBlockSparseTop2MLP` 已经不存在了）

如果你在网上或旧笔记里看到这个类叫 `MixtralBlockSparseTop2MLP`——**实测：这个名字在当前代码里已经彻底消失**
（`grep -rn "MixtralBlockSparseTop2MLP" src/ docs/ tests/` → 无匹配）。
它在历史上被改名/重构了两次，可以用 git 查到：

```bash
git log --oneline -S "MixtralBlockSparseTop2MLP" -- src/transformers/models/mixtral/
# 7938e91faa MoE + vllm (#40132)        ← 删除，改名 MixtralExperts
git log --oneline -S "gate_up_proj = nn.Parameter(torch.empty(self.num_experts" \
    -- src/transformers/models/mixtral/modular_mixtral.py
# 6f6095e0cf Refactor weight loading (#41580)  ← 8 个 nn.Linear 合并成 3D 张量
```

演化路径（三代）：

| 代 | 类名 | 专家的存储 | 循环 |
|----|------|-----------|------|
| 2023–2024 | `MixtralBlockSparseTop2MLP` | 每个专家一个 module（`nn.ModuleList`） | 逐个专家取 module |
| 2025-10-02（`7938e91faa`） | `MixtralMLP` + `MixtralExperts(nn.ModuleList)` | 同上 | 同上，但专家纯是 MLP |
| **现在**（`6f6095e0cf` 之后） | **`MixtralExperts(nn.Module)`**（`modular_mixtral.py:123`） | **3D 参数字典** `gate_up_proj [E, 2I, H]`、`down_proj [E, H, I]` | 循环里直接切张量 |

改名不是没代价的——它**丢掉了那段很值钱的 docstring**。老版本 `MixtralSparseMoeBlock` 上写着：

> This implementation is strictly equivalent to standard MoE with full capacity (no dropped tokens).
> It's faster since it formulates MoE operations in terms of **block-sparse operations** to accommodate
> imbalanced assignments of tokens to experts, whereas standard MoE either (1) drop tokens at the cost of
> reduced performance or (2) set capacity factor to number of experts and thus waste computation and memory
> on padding.

（引自 `git show 7938e91faa -- src/transformers/models/mixtral/modular_mixtral.py`。）

**这段话点出了 HF 实现的核心取舍**：不做 capacity factor、不丢 token（drop token 是 Switch Transformer 的做法），
而是「按需分组、变长计算」—— 这正是下面这段代码在做的事。

### 6.2 单个专家 = 一个普通 SwiGLU MLP

把 `MixtralExperts` 里「单个专家」的公式抽出来（`modular_mixtral.py:153-155`）：

```python
gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)  # :153
current_hidden_states = self.act_fn(gate) * up                                                   # :154
current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])   # :155
```

和 [1M1](./1M1_llama_structure.md) 的 `LlamaMLP.forward`（`modeling_llama.py:174-176`）逐项对照：

| LlamaMLP | Mixtral 的单个专家 | 等价性 |
|----------|-------------------|--------|
| `self.gate_proj(x)` | `self.gate_up_proj[e]` 的前半 `.chunk(2,-1)[0]` 与 x 相乘 | 同一个 `[I, H]` 矩阵 |
| `self.up_proj(x)` | `self.gate_up_proj[e]` 的后半 `.chunk(2,-1)[1]` 与 x 相乘 | 同一个 `[I, H]` 矩阵 |
| `self.act_fn(...)` | `self.act_fn(...)`，`config.hidden_act = "silu"`（`:75`） | 同一个 SiLU |
| `gate * up` | `act_fn(gate) * up` | 同一个 GLU 门控 |
| `self.down_proj(...)` | `self.down_proj[e]`，形状 `[H, I]` | 同一个 `[H, I]` 矩阵 |

**结论：专家就是 [1M1](./1M1_llama_structure.md) 里那个 MLP，一模一样，没有任何新的数学。**
唯一的实现差异是「两个投影被拼成一个大矩阵」：

```
   LlamaMLP 的一个专家:   gate_proj [I, H]   +   up_proj [I, H]     两个独立 nn.Linear
   Mixtral 的一个专家:    gate_up_proj[e]  =  [2I, H]               一个 2I×H 的参数切片
                                                  ↑
                                     前 I 行 = gate，后 I 行 = up（.chunk(2, dim=-1) 就是切这个）

   8 个专家摞起来:        gate_up_proj [E, 2I, H] = [8, 28672, 4096]  ← 3D 张量，不是 ModuleList
```

为什么要改成 3D 张量？看 `config` 里这句（`configuration_mixtral.py:50`）：

```python
"layers.*.mlp.experts.gate_up_proj": "packed_colwise",
```

**为了并行 / 高效 kernel**：3D 张量可以直接喂给 grouped-GEMM / vLLM 的 MoE kernel，
而 `nn.ModuleList` 里 8 个独立 `nn.Linear` 只能一个一个算。
装饰器 `@use_experts_implementation`（`modular_mixtral.py:122`）就是这套分发机制的入口
（定义在 `integrations/moe.py:523`，可替换实现注册在 `ALL_EXPERTS_FUNCTIONS`），
这让同一个模块在 CPU/GPU/TensorRT-LLM/vLLM 下能有不同 backend。这部分见 [2A6](./00_roadmap.md)。

### 6.3 `MixtralExperts.forward` 逐行

`modular_mixtral.py:135-159`（逐字）：

```python
    def forward(                                                       # :135
        self,
        hidden_states: torch.Tensor,          # [T, H]  已压平
        top_k_index: torch.Tensor,            # [T, k]  每个 token 选的专家下标
        top_k_weights: torch.Tensor,          # [T, k]  对应权重，已归一化
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)           # :141
        with torch.no_grad():                                          # :142
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)  # :143
            expert_mask = expert_mask.permute(2, 1, 0)                  # :144
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()  # :145

        for expert_idx in expert_hit:                                  # :147
            expert_idx = expert_idx[0]                                 # :148
            if expert_idx == self.num_experts:                         # :149
                continue                                               # :150
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])  # :151
            current_state = hidden_states[token_idx]                    # :152
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)  # :153
            current_hidden_states = self.act_fn(gate) * up              # :154
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])  # :155
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]  # :156
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))  # :157

        return final_hidden_states                                      # :159
```

**`one_hot` + `permute` 得到的到底是什么**（`:143-144`）：

```
top_k_index                    [T, k]      每个 token 的 2 个选择
  ↓ one_hot(num_classes=E)
                               [T, k, E]   展开成 one-hot
  ↓ permute(2, 1, 0)   （把 E 维挪到最前，T 维挪到最后）
expert_mask                    [E, k, T]   ← 转置成「每个专家一张命中表」

expert_mask[e]                 [k, T]     专家 e 的命中表
  │  torch.where(...)                       :151
  └─► top_k_pos [n_e]  （命中发生在该 token 的第几号 top-k 位，取值 0 或 1）
      token_idx [n_e]  （哪些 token 命中了专家 e）
      n_e = 专家 e 收到的 (token, slot) 对的数量，随 token 而变
```

**把形状翻译成一句话**：`expert_mask` 从「token 视角」（我选了谁）翻成「专家视角」（谁选了我）。
之后 `:151-152` 就是在专家视角下把属于 `e` 的那些 token 行**挑出来**：

```
hidden_states [T, H]
      │ hidden_states[token_idx]            :152   ← 花式索引，只取命中的行
      ▼
current_state [n_e, H]      ← 变长！每个专家看到的 token 数不同
      │ :153-155  过一遍和 LlamaMLP 完全相同的 SwiGLU
      ▼
current_hidden_states [n_e, H]
      │ × top_k_weights[token_idx, top_k_pos]   :156   ← 加权（每个 token 权重不同）
      ▼
final_hidden_states.index_add_(0, token_idx, ...)  :157   ← 累加回原位置
```

**这份「变长」就是稀疏性的实现本体**：一个专家只处理 `n_e` 行（`n_e` 平均是 `T×k/E`），
而不是全部 `T` 行。`T×k/E` 这个比例就是「计算量的稀疏率」：`k/E = 2/8 = 25%`
（和 1.3 节算出的 27.6% 是同一件事，差在 attention 与词嵌入的固定开销）。

### 6.4 `:156` 的加权与 `:157` 的累加，为什么非要这么做

**`:156` 乘权重**：一个 token 在两个专家里各出一次结果，两次结果的权重不同，所以必须在「回填之前」乘好。

**`:157` 用 `index_add_` 而不是赋值**：因为一个 token 会被两个专家**写两次**，
必须累加而不是覆盖。注意 `token_idx` 里同一个 token 会出现 2 次（每个命中的专家一次）——
`index_add_` 天然支持这种重复下标累加，这也是它比 `final[token_idx] = ...` 更合适的原因。

还有一个小陷阱在 `:156` 的索引：`top_k_weights[token_idx, top_k_pos, None]`。
`token_idx` 和 `top_k_pos` 是同长度的一维张量，这种写法是**逐元素的配对索引**
（第 `i` 个命中对应的权重），而不是外积。最后那个 `None` 是把 `[n_e]` 扩成 `[n_e, 1]`
以便按行广播到 `[n_e, H]`。

---

## 七、稀疏性的实现细节

### 7.1 为什么必须 `view(-1, H)`（`:192`）

这是整篇最容易被略过、但最该理解的一行。

**因为 MoE 的选择是 per-token 的，不是 per-sequence 的。**
router 对序列里每一个位置**独立**做判断：一句话里描述代码的 token 去「代码专家」，
描述情感的 token 去「情感专家」。所以路由必须发生在「每个 token」这个粒度上。

但 transformer 的 hidden_states 是 `[B, S, H]` 三维的。要把 `B` 和 `S` 两个维度都摊平：

```
hidden_states [B, S, H]
   │ .view(-1, H)                        :192
   ▼
[ B × S, H ]  =  [T, H]  ← 一维的「token 列表」，router 和 experts 都只认这个

   ... router 路由 / experts 计算（都在 [T, H] 上进行）...

   │ .reshape(batch_size, sequence_length, hidden_dim)   :195
   ▼
[B, S, H]   ← 还原，因为 attention 那边按 [B, S, H] 工作
```

**为什么不能省掉这个 reshape**：如果直接在 `[B, S, H]` 上做 top-k，你会得到「每个 batch 选一批专家」，
整个序列被迫共用同一套专家——那不是 MoE，那只是「给不同样本用不同模型」。
`B` 和 `S` 在这里的地位完全一样：都是「独立的 token 来源」。

**副作用（值得记住）**：既然 `B` 和 `S` 被合并，**MoE 对 padding 是无感的**——
padding 位置的 token 也会被路由、也会占用专家算力。这也是 `load_balancing_loss_func`
为什么要专门处理 `attention_mask`（`:92-113`）的原因：**训练时得把 padding 的贡献从统计里剔掉**，
否则辅助损失会被 padding 带偏。这是「压平 token」这个设计带来的连锁反应。

### 7.2 关于「`num_experts_per_tok == num_local_experts` 时可以不循环」

一个常被问到的问题是：**如果 `num_experts_per_tok == num_local_experts`（即 `k == E`），
是不是就不需要循环、可以直接批量算？**

**实测结论：当前代码里没有这个分支**——
全仓库搜 `top_k == self.num_experts` / `num_experts_per_tok ==` / `top_k >= ` 等模式，
在 `src/transformers/models/` 和 `src/transformers/integrations/` 里**零命中**。
`MixtralSparseMoeBlock` 和 `MixtralExperts` 里唯一的分支是 `:149` 那个 `continue`。

不过这个「捷径」在理论上确实成立，值得说清它成立在哪：

> 当 `k == E` 时，`torch.topk(probs, k=E)` 返回**全部**专家（只是按分数排序），
> 于是每个 token 都会命中所有专家。此时 `expert_mask` 的每一列都是满的，
> 循环会遍历全部 E 个专家、且每个专家都拿到全部 T 个 token（`n_e = T`），
> 变长退化成定长。加权求和也就变成了「E 个满宽 MLP 输出按 router 权重做加权组合」——
> 这正是 dense MLP 的一种推广形式，只是参数从 1 份变成了 E 份。
> 所以此时可以不分派、直接批量算（等价于一个 `[E, 2I, H]` 的 grouped GEMM 不做分组）。

而 Mixtral 的默认配置是 `k=2`、`E=8`（`configuration_mixtral.py:86-87`），
`k/E = 25%`，**离这个退化情形很远**，所以代码里根本没有必要写这个分支。

真正与「不循环」相关的设计，是另一件事：**循环只遍历 `expert_hit`（命中的专家）而不是 `range(E)`**（`:145`、`:147`）。
当 `T` 小（例如自回归解码时 `T=1`）或路由集中的时候，8 个专家里可能只有 2 个被命中——
这时候循环体只跑 2 次。这个优化和 `k` 与 `E` 的关系无关，是通用的。

### 7.3 一个容易误读的防御性分支：`:149-150`

```python
if expert_idx == self.num_experts:
    continue
```

`top_k_index` 来自 `torch.topk`，取值范围必然是 `[0, E-1]`，**永远不会等于 `E`**。
所以这个 `continue` 在当前调用路径下是**不可达的**——它是防御性的，
沿袭自那些会把「无专家」编码成哨兵值 `E` 的路由实现（全库确实存在这种写法，见
`models/zaya/modular_zaya.py:438` 的 `self.num_router_classes = config.num_experts + 1`）。

**读源码时的启示**：见到这种「看着多余」的判断，先怀疑它是为别的调用路径准备的，
别当成 bug 去改。它同时也提示你：`one_hot(num_classes=E)` 的上界就是 `E`，
不存在越界风险。

---

## 八、负载均衡与辅助损失

### 8.1 专家坍缩（expert collapse）

MoE 有一个自增强的坏循环：

```
某几个专家初始分数高 → 拿到更多 token → 梯度更多 → 学得更好 → 分数更高 → ...
                                    ↓
              剩下 6 个专家几乎收不到 token → 收不到梯度 → 永远学不动
```

极端情况下 8 个专家里只有 1–2 个在干活，**MoE 退化成 dense 模型，你白付了 8 倍的显存**。
这叫专家坍缩。

工程上有三类对策，Mixtral 用了前两类：

| 对策 | 机制 | Mixtral 里的位置 |
|------|------|-----------------|
| **辅助损失（aux loss）** | 在训练 loss 里加一项，惩罚路由不均衡 | `load_balancing_loss_func`（`:53`）+ `router_aux_loss_coef=0.001`（`:89`） |
| **训练期噪声（jitter）** | 给 router 输入加噪声，防止早期过早锁定 | `:190-191`，`router_jitter_noise` 默认 `0.0`（关闭） |
| **z-loss** | 惩罚 router logits 的绝对大小，稳定训练 | **Mixtral 没有**，见 8.5 |

注意 `router_jitter_noise` 默认是 `0.0`，也就是**默认不生效**——Mixtral 官方训练主要靠辅助损失。

### 8.2 `load_balancing_loss_func` 的位置：**它就在 modular 文件里**

一个自然的怀疑是：这个函数会不会只存在于生成的 `modeling_mixtral.py` 里（即 modular 里没有）？
**实测：不是。**

- `modular_mixtral.py:53` —— 完整定义，67 行（`:53-119`），**手写在 modular 里**。
- `modeling_mixtral.py:485` —— 生成文件里的副本，内容逐行相同（converter 直接搬过去的）。

所以答案是：**它是 Mixtral 手写的模块级函数，不是自动拉取的依赖，也不是生成文件独有的东西。**
之所以容易误解，是因为 Mixtral 的 modular 文件里**只有这一个模块级函数**（其它 modular 文件通常一个都没有），
而生成的 `modeling_mixtral.py` 里它被排在了 `MixtralModel` 之后（`:485`），看起来像「补充进去的工具函数」。
按 [2A5](./2A5_modular_code_generation.md) 的规则，converter 生成的代码顺序是拓扑排序 + 依赖顺序：
`load_balancing_loss_func` 被 `MixtralForCausalLM`（`:555`）依赖，所以排在它前面、模型定义之后。

**顺带一个反面例子**：同文件里 `MixtralRMSNorm`（`:199`）、`MixtralRotaryEmbedding`（`:203`）、
`MixtralAttention`（`:207`）在 modular 里只写了 `pass`，但生成文件里是完整实现
（`modeling_mixtral.py:134` / `:154` / `:282`）——**那才是「自动拉取」**。
所以判断标准是：**modular 里写了完整函数体 = 手写；只写 `pass` 或根本没出现 = 自动拉取。**

### 8.3 逐段读 `load_balancing_loss_func`

函数签名（`modular_mixtral.py:53-58`）：

```python
def load_balancing_loss_func(
    gate_logits: torch.Tensor | tuple[torch.Tensor] | None,
    num_experts: int | None = None,
    top_k=2,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor | int:
```

| 段 | 行 | 做什么 |
|----|----|--------|
| 早退 | `:82-83` | `if gate_logits is None or not isinstance(gate_logits, tuple): return 0`。用 `isinstance(..., tuple)` 判断是因为 hook 收集的是「每层一个张量」的元组 |
| 累加器 | `:87-90` | `tokens_per_expert_sum [E]`、`router_prob_sum [E]`、`total_rows`。注释（`:85-86`）说明这是为了显存优化：「逐层累加、最后归一化，峰值显存 O(seq_len × num_experts)，与层数无关」 |
| 掩码预备 | `:92-94` | 把 `attention_mask` 摊平成 `[T]` 的 float |
| 逐层循环 | `:96` | `for layer_gate in gate_logits:` |
| softmax | `:97` | `softmax(layer_gate)` —— 把该层的 **router logits** 转成概率 |
| top-k | `:98` | `torch.topk(routing_weights, top_k)` 只取模长，`_` 丢掉值 |
| 无 mask 支路 | `:99-106` | `torch.bincount` 数每个专家被选中几次（`:101-103`）；`routing_weights.sum(dim=0)` 累加每个专家的概率和（`:105`）；`total_rows += T`（`:106`） |
| 有 mask 支路 | `:107-113` | 同样的量，但用 `scatter_add_` + `flat_mask.repeat_interleave(top_k)`（`:109-111`）把 padding 的贡献置零；分母换成 `flat_mask.sum()`（`:113`） |
| 归一化 | `:115-116` | `tokens_per_expert = 命中数 / token 数`；`router_prob_per_expert = 概率和 / token 数` |
| 汇总 | `:118-119` | `overall_loss = Σ_i (f_i · P_i)`，然后 `return overall_loss * num_experts` |

### 8.4 这个损失在算什么（以及一个实测出来的细节）

形式上和 Switch Transformer 论文的式 (4)-(6) 一致（docstring `:60-64` 提到 papers/2101.03961）：

$$
\mathcal{L}_{aux} = E \cdot \sum_{i=1}^{E} f_i \cdot P_i
$$

- **`f_i`**：专家 `i` 分到的份额（token 视角的「谁在干活」）
- **`P_i`**：专家 `i` 在所有 token 上的平均路由概率（模型视角的「谁显得重要」）

乘积的两个因子必须**同时**都对某个专家高，损失才会大；这就在惩罚「分数高但拿不到 token」（P 高 f 低）
和「拿了一堆 token 但分数低」（f 高 P 低）两种畸形。

**一个实测出来的实现细节（容易看漏）**：本实现里 `f_i` 是「top-k **命中次数** / token 数」，
每个 token 贡献 `k` 次命中，所以 **`Σ_i f_i = k`（=2），而不是 1**。代入均匀路由验证：

```
均匀路由： f_i = k/E = 2/8 = 0.25，P_i = 1/E = 0.125
          Σ_i f_i·P_i = 8 × 0.25 × 0.125 = 0.25
          × E = 8  →  L = 2   （= k）

全部坍缩到 1 个专家： f_j = k = 2，P_j ≈ 1
          Σ = 2      × E = 8  →  L = 16  （= E·k）
```

所以这版实现的**下界是 `k` 而不是 1**（比论文公式整体大了 `k` 倍），
但「越不均衡越大、最差情况是均匀时的 `E` 倍」这个性质没变——
再加上 `router_aux_loss_coef = 0.001`（`configuration_mixtral.py:89`）这个很小的系数，
常数偏移不影响实际训练。读到 `:119` 那个 `* num_experts` 时知道它在放大不平衡信号即可。

### 8.5 `output_router_logits`：logits 是怎么从 forward 里「消失」又被找回来的

回顾第四节：`MixtralSparseMoeBlock.forward` 里 `router_logits` 被 `_` 丢掉了（`:193`）。
它靠 **hook** 被捡回来。机制在 `MixtralPreTrainedModel`（`modular_mixtral.py:249-254`）：

```python
class MixtralPreTrainedModel(MistralPreTrainedModel):                 # :249
    _can_record_outputs = {                                          # :250
        "router_logits": OutputRecorder(MixtralTopKRouter, index=0),  # :251
        "hidden_states": MixtralDecoderLayer,                        # :252
        "attentions": MixtralAttention,                              # :253
    }
```

`OutputRecorder(MixtralTopKRouter, index=0)` 的含义是：
**在每个 `MixtralTopKRouter` 上挂钩子，抓它返回元组的第 0 项**——
而 router 的返回值正是 `(router_logits, router_scores, router_indices)`（`:177`），
所以第 0 项恰好就是我们要的 logits。

这与 [1M2](./1M2_mistral_sliding_window.md) 的 `MistralPreTrainedModel` 形成对照——
`modular_mistral.py:102-106` 里只记 `hidden_states` 和 `attentions`，**没有 `router_logits`**。
MoE 模型多出来的就是这一行。

**什么时候真的去抓**（`utils/output_capturing.py:238-256`）：

```python
recordable_keys = {
    f"output_{k}": kwargs.get(f"output_{k}", getattr(self.config, f"output_{k}", False))
    for k in capturable_flags
}
```

即 `output_router_logits=True`（kwarg 或 config，默认 `config` 里是 `False`，`:88`）时才安装 hook 并收集。
**所以 `router_logits` 是「按需付费」的**——不开这个开关，hook 不装、显存不涨。

最后 `MixtralModel` 用 `MoeModelOutputWithPast` 把结果带出来（`modular_mixtral.py:317-320`，
代码里那句注释写得很直白）：

```python
return MoeModelOutputWithPast(  # only diff with Mistral is the output type, we need MoE   # :317
    last_hidden_state=hidden_states,
    past_key_values=past_key_values,
)
```

### 8.6 aux loss 怎么进总 loss

`MixtralForCausalLM.forward` 的尾部（`modular_mixtral.py:395-404`）：

```python
aux_loss = None                                                          # :395
if output_router_logits:                                                 # :396
    aux_loss = load_balancing_loss_func(                                 # :397
        outputs.router_logits,                                           # :398  ← hook 抓来的
        self.num_experts,                                                # :399  E = 8
        self.num_experts_per_tok,                                        # :400  k = 2
        attention_mask,                                                  # :401  ← padding 屏蔽
    )
    if labels is not None:                                               # :403
        loss += self.router_aux_loss_coef * aux_loss.to(loss.device)     # :404  系数 0.001
```

两个必须记住的点：

1. **`aux_loss` 只在 `output_router_logits=True` 时才算**（`:396`）。
   也就是说默认配置下 `MixtralForCausalLM` **不会**自动平衡专家——训练 MoE 时要显式打开
   （或设 `config.output_router_logits=True`）。
2. **`aux_loss` 不会被 `backward()` 单独用到**，它是加进 `loss` 里的（`:404`）——
   梯度一起回传，router 的 `weight` 因此既能从主任务学到「该怎么路由」，
   也能从辅助损失学到「别太偏心」。
3. `:404` 的 `.to(loss.device)` 是因为 hook 收集的张量可能留在别的 device 上（例如多卡/offload 场景）。

### 8.7 z-loss：Mixtral 没有

z-loss（router z-loss，出自 papers/2202.08906）惩罚 router logits 的 log-sum-exp 平方，
目的是压住 logits 的量级、让 softmax 数值稳定：

```python
# 参考实现（不在 mixtral 里）
#   src/transformers/models/switch_transformers/modeling_switch_transformers.py:874
def router_z_loss_func(router_logits: torch.Tensor) -> float:
```

**实测结论**：Mixtral **没有** z-loss。证据链：

| 检查项 | 命令/位置 | 结果 |
|--------|-----------|------|
| config 里有 `router_z_loss_coef` 吗 | `configuration_mixtral.py` | **没有**（只有 `router_aux_loss_coef` / `router_jitter_noise`） |
| forward 里有 z_loss 吗 | `modular_mixtral.py:395-404` | 只有 `aux_loss` |
| 输出类型里有 z_loss 字段吗 | `modeling_outputs.py:353` `MoeCausalLMOutputWithPast` | 只有 `loss` / `aux_loss` / `router_logits`，**没有** `z_loss` |
| 那谁有 | `configuration_switch_transformers.py:77`、`configuration_nllb_moe.py:108` | `router_z_loss_coef: float = 0.001`，以及 `modeling_outputs.py:866-867` 的 `encoder_z_loss` / `decoder_z_loss` |

**注意一个坑**：`output_router_logits` 的公共 docstring 写着
「Enabling this will also allow the model to output the auxiliary loss, **including load balancing loss and router z-loss**」
（`utils/auto_docstring.py:1129-1133`）。这句是**通用 MoE 文档**，对 Mixtral 不成立——
Mixtral 只输出 load balancing loss。读文档时要对照代码。

所以 Mixtral 稳定 router 数值的手段是另一条路：**softmax 前转 float32**（`:173`，见 5.3）。

---

## 九、Mixtral vs Mistral：差异清单

先看 `MixtralDecoderLayer` 自己（`modular_mixtral.py:211-246`），
它的 Pre-Norm + 残差骨架和 [1M1](./1M1_llama_structure.md) 的 `LlamaDecoderLayer` 一字不差：

```python
        residual = hidden_states                                    # :231
        hidden_states = self.input_layernorm(hidden_states)         # :232
        hidden_states, _ = self.self_attn(...)                      # :233
        hidden_states = residual + hidden_states                    # :241
        residual = hidden_states                                    # :242
        hidden_states = self.post_attention_layernorm(hidden_states)  # :243
        hidden_states = self.mlp(hidden_states)                     # :244   ← 就是这一行换了
        hidden_states = residual + hidden_states                    # :245
```

**整份对照表**：

| 模块 | Mistral（1M2） | Mixtral | 差异 |
|------|---------------|---------|------|
| `RMSNorm` | `MistralRMSNorm` | `MixtralRMSNorm(MistralRMSNorm)`: `pass`（`:199`） | **无** |
| `RoPE` | `MistralRotaryEmbedding` | `MixtralRotaryEmbedding(MistralRotaryEmbedding)`: `pass`（`:203`） | **无**（但 config 默认 theta 从 1e4 变 1e6，见下） |
| `Attention` | `MistralAttention`（滑动窗口） | `MixtralAttention(MistralAttention)`: `pass`（`:207`） | **无**，滑动窗口一起继承 |
| **MLP** | `MistralMLP`（SwiGLU） | **`MixtralSparseMoeBlock`**（`:218`） | ⭐ **唯一的实质改动** |
| `DecoderLayer` 骨架 | Pre-Norm + 残差 | 同（`:231-245`） | 无 |
| `Model` | `MistralModel` | `MixtralModel(MistralModel)`，只覆盖 `forward`（`:267-320`） | 输出类型换 `MoeModelOutputWithPast`（`:317`）；mask 选择逻辑照抄（`:292`） |
| `PreTrainedModel` | 无 `router_logits` 记录 | `_can_record_outputs` 多一项（`:251`） | 见 8.5 |
| `ForCausalLM` | 只有 lm loss | 多 aux loss 分支（`:395-404`） | 见 8.6 |
| 三个分类头 | — | `pass`（`:417/421/425`） | 无 |
| `sliding_window` 默认值 | `4096`（`configuration_mistral.py:80`） | **`None`**（`configuration_mixtral.py:84`） | 能力继承了，但**默认关闭** |
| `rms_norm_eps` 默认值 | `1e-6`（`configuration_mistral.py:73`） | `1e-5`（`configuration_mixtral.py:78`） | 纯超参差异 |
| RoPE `default_theta` | 继承默认 `10000.0`（`modeling_rope_utils.py:739`） | `1000000.0`（`configuration_mixtral.py:44`） | 纯超参差异（长上下文） |
| 专家相关超参 | — | `num_local_experts=8`、`num_experts_per_tok=2` 等（`:86-90`） | **新增** |

**两条推论**：

1. **滑块窗口是「白送」的**：因为 `MixtralAttention` 是 `pass`，1M2 学的那套 SWA
   在 Mixtral 里原样存在，只是默认 `sliding_window=None` 把它关掉了。
2. **一篇 1M2 只学到了「一个功能」，1M3 学到了「一个机制」**。
   1M2 的改动是「加一个参数」，1M3 的改动是「换一整块计算」——
   这就是路线图里说的「第一次真正的结构创新」。

---

## 十、MoE 的工程挑战

这一节只给动机和方向，机制细节在架构线里。

### 10.1 专家并行（EP）：MoE 是「为并行而生」的

**一句话**：46.7B 参数放不进一张卡，但 8 个专家可以分散到 8 张卡上，token 通过 all-to-all
被送到「选中专家所在的那张卡」算完再送回来——代价是每层两次 all-to-all，瓶颈从算力变成带宽。
机制细节见 2A6（`integrations/moe.py`、`integrations/tensor_parallel.py`）。

配置里已经把这件事写清了（`configuration_mixtral.py:59-64`）：

```python
base_model_ep_plan = {
    "layers.*.mlp.gate": "ep_router",
    "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
    "layers.*.mlp.experts.down_proj": "grouped_gemm",
    "layers.*.mlp.experts": "moe_tp_experts",
}
```

对照张量并行（TP）计划里的专家部分（`:50-52`）：
`"layers.*.mlp.experts.gate_up_proj": "packed_colwise"`。
**注意 `packed_` 前缀**——这就是 6.2 节把 8 个 `nn.Linear` 合并成 3D 张量的直接原因：
切分一个 3D 张量比切分 8 个独立 module 好写得多。

### 10.2 显存 vs 计算量的权衡

反直觉的一点：**MoE 不省显存**，只省计算。

| 资源 | dense 46.7B | Mixtral 8x7B | 谁赢 |
|------|------------|--------------|------|
| 权重显存（bf16） | ≈ 93 GB | ≈ 93 GB | 平手 |
| 每 token FLOPs | 100% | ≈ 27.6% | **MoE** |
| KV cache | 与 dense 同结构 | 同（注意力没变） | 平手 |
| 通信 | DP 下无 | 每层 2 次 all-to-all | dense |
| 训练稳定性 | 稳 | 需 aux loss / 调参 | dense |

所以 MoE 的适用场景是「**算力紧张、显存充裕**」——尤其是推理（decode 阶段 attention 已经吃掉大量带宽，
MoE 的稀疏 FFN 反而摊薄了权重读取量）。
反过来说，在**单卡显存装不下 46.7B** 的场景下，MoE 一点忙都帮不上。

### 10.3 推理时的 expert offloading

因为「参数量 = 显存」这条不省，推理部署 MoE 有一个绕不开的现实问题：
8x7B 用 bf16 大约要 93 GB 权重，单张 80 GB 卡都放不下。

于是有了 **expert offloading**：把冷门专家放在 CPU 内存 / NVMe 上，
按 router 的选择动态把需要的那几个专家搬进显存。

它的可行性恰恰来自稀疏性：**每层每步只需要 2 个专家**。
如果每步都要用到全部 8 个，offloading 的搬运开销会吃掉全部收益；
正因为 25% 的稀疏率，才可能「按需搬运」。
（这是 MoE 在**单卡**推理场景下唯一能压缩显存的手段，代价是把 compute-bound 问题
变成了 memory-bandwidth-bound 问题。）

---

## 一句话总结

> **MoE = 把 MLP 换成「router + 一堆 MLP + 加权求和」。**
> router 是一个 `H → E` 的线性层（`:168`），top-k 挑 2 个（`:174`），
> softmax 后重新归一化（`:175`），命中的专家只对选中它的那些 token 做 SwiGLU（`:152-155`），
> 结果加权累加回原位置（`:156-157`）。
> 专家本身**没有任何新数学**——就是 [1M1](./1M1_llama_structure.md) 的 `LlamaMLP`。
> 全部新意在于「哪些 token 走哪个专家」是**学出来的、每 token 独立的、稀疏的**。
> Mixtral 8x7B：参数 46.7B、每 token 只用 12.9B（27.6%）——这就是「加参数不变慢」的全部内容。

---

## 验证清单

所有结论都可以在仓库根目录复现（全是只读命令）：

```bash
cd /Users/baai/Desktop/baai_proj/transformers

# ── 1. 有 modular，所以读 modular（2A5 规则 1）
ls src/transformers/models/mixtral/
#   → __init__.py  configuration_mixtral.py  convert_mixtral_weights_to_hf.py
#     modeling_mixtral.py  modular_mixtral.py        ← 有 modular
wc -l src/transformers/models/mixtral/modular_mixtral.py \
      src/transformers/models/mixtral/modeling_mixtral.py
#   → 436 modular_mixtral.py / 677 modeling_mixtral.py

# ── 2. config 的真实默认值（本篇 1.3 节的数字来源）
grep -n "num_experts_per_tok\|num_local_experts\|intermediate_size\|hidden_size\|num_hidden_layers" \
     src/transformers/models/mixtral/configuration_mixtral.py
#   → 69: hidden_size: int = 4096
#   → 70: intermediate_size: int = 14336
#   → 71: num_hidden_layers: int = 32
#   → 86: num_experts_per_tok: int = 2
#   → 87: num_local_experts: int = 8
sed -n '44p;78p;83,91p' src/transformers/models/mixtral/configuration_mixtral.py
#   → default_theta = 1000000.0 / rms_norm_eps = 1e-5 / tie_word_embeddings=False
#     sliding_window=None / num_experts_per_tok=2 / num_local_experts=8
#     output_router_logits=False / router_aux_loss_coef=0.001 / router_jitter_noise=0.0

# ── 3. 参数量（1.3 节表格）
python3 - <<'EOF'
H,I,L,E,K=4096,14336,32,8,2
expert=E*(2*I*H)+E*(H*I)              # 每层专家
attn=H*H+2*H*(8*128)+H*H               # q/o + k/v (GQA 8 头 × 128)
tot=expert*L+attn*L+E*H*L+3*H*L+32000*H*2
act=(K/E)*expert*L+attn*L+E*H*L+3*H*L+32000*H*2
print(f"总参数 {tot/1e9:.2f}B  激活 {act/1e9:.2f}B  比例 {act/tot:.1%}")
EOF
#   → 总参数 46.70B  激活 12.88B  比例 27.6%

# ── 4. 主角与三个子模块的真实位置
grep -n "class MixtralExperts\|class MixtralTopKRouter\|class MixtralSparseMoeBlock\|def load_balancing_loss_func" \
     src/transformers/models/mixtral/modular_mixtral.py
#   → 53:def load_balancing_loss_func( / 123:class MixtralExperts
#     162:class MixtralTopKRouter / 180:class MixtralSparseMoeBlock
grep -n "self.mlp = MixtralSparseMoeBlock" src/transformers/models/mixtral/modular_mixtral.py
#   → 218:        self.mlp = MixtralSparseMoeBlock(config)

# ── 5. load_balancing_loss_func 在 modular 里（不是自动拉取、也不只在生成文件里）
grep -n "def load_balancing_loss_func" src/transformers/models/mixtral/modular_mixtral.py \
                                      src/transformers/models/mixtral/modeling_mixtral.py
#   → modular_mixtral.py:53  /  modeling_mixtral.py:485   ← 两边都有，手写在 modular

# ── 6. 对照：真正「自动拉取」的类长什么样（modular 里只有 pass）
grep -n "class MixtralRMSNorm\|class MixtralRotaryEmbedding\|class MixtralAttention" \
     src/transformers/models/mixtral/modular_mixtral.py
#   → 199 / 203 / 207  （后面都只有 pass）
grep -n "class MixtralRMSNorm\|class MixtralRotaryEmbedding\|class MixtralAttention" \
     src/transformers/models/mixtral/modeling_mixtral.py
#   → 134 / 154 / 282  （生成文件里是完整实现）

# ── 7. 老类名 MixtralBlockSparseTop2MLP 已不存在
grep -rn "MixtralBlockSparseTop2MLP" src/ docs/ tests/          # → 无匹配
git log --oneline -S "MixtralBlockSparseTop2MLP" -- src/transformers/models/mixtral/
#   → 7938e91faa MoE + vllm (#40132)      ← 删除
git log --oneline -S "gate_up_proj = nn.Parameter(torch.empty(self.num_experts" \
    -- src/transformers/models/mixtral/modular_mixtral.py
#   → 6f6095e0cf Refactor weight loading (#41580)  ← 引入 3D 张量

# ── 8. 「top_k == num_experts 就不循环」这个捷径在当前代码里不存在
grep -rn "top_k == \|top_k >= \|num_experts_per_tok == " src/transformers/models/ src/transformers/integrations/
#   → 零命中（所以是理论捷径，不写在代码里）

# ── 9. z-loss：Mixtral 没有
grep -n "z_loss" src/transformers/models/mixtral/configuration_mixtral.py      # → 无
grep -n "z_loss" src/transformers/models/mixtral/modular_mixtral.py            # → 无
grep -n "router_z_loss_coef" src/transformers/models/switch_transformers/configuration_switch_transformers.py
#   → 77:    router_z_loss_coef: float = 0.001     ← 别人有

# ── 10. 两套机制互斥 + 生成文件无跨模型 import（2A5 的结论在 mixtral 上复现）
grep -c "Copied from" src/transformers/models/mixtral/modeling_mixtral.py       # → 0
grep -nE "^from \.\.[a-z_0-9]+\.(modeling|configuration)" \
     src/transformers/models/mixtral/modeling_mixtral.py                        # → 无匹配
grep -nE "^from \.\.[a-z_0-9]+\.(modeling|configuration)" \
     src/transformers/models/mixtral/modular_mixtral.py
#   → 36:from ..mistral.modeling_mistral import (

# ── 11. Mixtral 是 MoE 家族的「Llama」（9 个别的模型继承它）
grep -rn "MixtralPreTrainedModel" src/transformers/models/*/modular_*.py | grep "class "
#   → deepseek_v4:1028、flex_olmo:185、ernie4_5_moe:186、jetmoe:418、minimax_m2:172、
#     minimax:409、phimoe:341、qwen3_moe:85、qwen2_moe:149   ← 9 个
#     （mixtral 自己是 249 行的 class MixtralPreTrainedModel(MistralPreTrainedModel)）
grep -n "class Qwen3MoeModel" src/transformers/models/qwen3_moe/modular_qwen3_moe.py
#   → 93:class Qwen3MoeModel(MixtralModel):

# ── 12. modular 的类清单（13 个类 + 1 个函数，6 个只写 pass）
grep -n "^class \|^def " src/transformers/models/mixtral/modular_mixtral.py
#   → 53 def load_balancing_loss_func / 123 MixtralExperts / 162 MixtralTopKRouter
#     180 MixtralSparseMoeBlock / 199 MixtralRMSNorm / 203 MixtralRotaryEmbedding
#     207 MixtralAttention / 211 MixtralDecoderLayer / 249 MixtralPreTrainedModel
#     267 MixtralModel / 323 MixtralForCausalLM / 417 / 421 / 425
```

---

## 导航

- 路线图：[00_roadmap.md](./00_roadmap.md)（本篇 = 入门 · 模型结构 · 1M3）
- 上一篇：[1M2_mistral_sliding_window.md](./1M2_mistral_sliding_window.md)
- 下一篇：[1M4_qwen3_dense.md](./1M4_qwen3_dense.md)
- 机制背景：[2A5_modular_code_generation.md](./2A5_modular_code_generation.md)（为什么读 modular 而不是生成文件）
- 顺着 MoE 往下走（进阶）：[2M2_deepseek_mla_moe.md](./2M2_deepseek_mla_moe.md)（细粒度路由 + 共享专家，待写）→
  [2M3_qwen3_moe.md](./2M3_qwen3_moe.md)（三种 MoE 的取舍对比，待写）→ [2M7_glm_moe_dsa.md](./2M7_glm_moe_dsa.md)（压轴）
