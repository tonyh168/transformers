# 主流模型结构演化 · 速查总表

> 以 Llama 为基线，横向对比当前主流 decoder-only 大模型的结构差异。
> 所有断言都指向本仓库源码（`modular_*.py` 优先，见 [2A5](./2A5_modular_code_generation.md)），行号见文中表格。
>
> **定位**：本篇是总览/速查，不是精读。带行号考证的逐模块拆解，去读对应篇目：
> [1M1 Llama](./1M1_llama_structure.md) · [1M4 Qwen3](./1M4_qwen3_dense.md) · [2M1 Gemma2](./2M1_gemma2.md) · [2M2 DeepSeek MLA](./2M2_deepseek_mla_moe.md) · [2M3 Qwen3-MoE](./2M3_qwen3_moe.md) · [2M5 Mamba2](./2M5_mamba2_ssm.md) · [2M7 GLM-5.3](./2M7_glm_moe_dsa.md)。
> 本篇覆盖的 **DeepSeek V4 / GLM5.3（`glm5_next`）/ MiniMax M3 / HY4（`hy_v3`）/ Gemma4 / Nemotron-H** 都是 25 篇矩阵之外的新模型——矩阵收官时的 GLM 是 `glm_moe_dsa`（2M7），这里的 GLM5.3 是它的下一代 `glm5_next`。

---

## 零、先立基线：Llama 的「最小完整件」

```
embed → [ RMSNorm → Attention → (+) → RMSNorm → SwiGLU-MLP → (+) ] × N → RMSNorm → lm_head
```

DecoderLayer 的四个固定件（`models/llama/modeling_llama.py`，512 行、无 modular、可直读）：

| 件 | 类 | 行号 |
|----|----|------|
| RMSNorm（Pre-Norm） | `LlamaRMSNorm` | `:71` |
| 全量 causal 注意力（GQA = repeat KV） | `LlamaAttention` | `:154` |
| SwiGLU dense MLP | `LlamaMLP` | `:51` |
| 标准 RoPE | `LlamaRotaryEmbedding` | `:90` |

**后面每个模型都可以写成「Llama + 若干改动」**。差异集中在四条轴：

1. **注意力怎么省**（压缩 / 稀疏 / 滑窗 / 线性化）
2. **MLP 要不要 MoE 化**（稠密 / 粗粒度 MoE / 细粒度 + 共享专家）
3. **层间怎么混**（同质 / 滑窗-全局交替 / 线性-全注意力混合）
4. **残差要不要改**（标准残差 / Hyper-Connection 多路残差流）

---

## 一、总览表（九行，一行一个模型）

| 模型（目录） | 注意力结构 | RoPE | MLP / MoE | 层间排布 | 特有机构 |
|---|---|---|---|---|---|
| **Llama**（`llama`） | GQA，`eager_attention_forward` | 标准 | SwiGLU dense | 同质 × N | ——（基线） |
| **Qwen3**（`qwen3`） | GQA + **QK-Norm** | 标准 | SwiGLU dense | 同质 × N | QK-Norm：逐头 RMSNorm，在 RoPE **之前**（1M4） |
| **Gemma3/4**（`gemma3`、`gemma4`） | GQA + QK-Norm + **V-Norm**；Gemma4 加 `attention_k_eq_v` | **双 θ**：局部 10k / 全局 1M，全局层 `proportional` + `partial_rotary_factor=0.25` | dense GeGLU（`gelu_pytorch_tanh`）；embedding 用 `√H` 缩放 | **5:1 滑窗/全局交替**（`layer_types`，滑窗 4096/512） | **4× RMSNorm**（input/post-attn/pre-ffn/post-ffn）；尾层起 **KV 共享**（共享层不建 k/v_proj）；Gemma4 的 `ClippableLinear` 权重裁剪 |
| **DeepSeek V3**（`deepseek_v3`） | **MLA**：KV 联合低秩压缩 + Q 低秩 | **解耦 rope key**：RoPE 只加在专门的 rope 维度，nope 维度不转 | DeepSeekMoE：细粒度路由专家 + **共享专家**；前 3 层 dense | 前 3 dense + 其余 MoE，注意力同质 | sigmoid 路由 + `e_score_correction_bias` 无辅助损失均衡（**选专家看偏置、算权重不看**） |
| **DeepSeek V4**（`deepseek_v4`） | **CSA / HCA 压缩注意力**：分块压缩 + **Indexer** 稀疏选块；共享-KV MQA | **交错 RoPE**（`repeat_interleave(2)`），只转尾部 rope 切片、留 nope；按 rope-type 分多套 θ（Laguna 模式） | Mixtral 式 MoE + **HashRouter**（确定性哈希路由，`modular:930`） | `layer_types`：2×HCA 引导 + CSA/HCA 逐层交替；`mlp_layer_types` 另管 dense/sparse | **HyperConnection**（mHC，Xie et al. 2026，`modular:769`）+ **HyperHead**（`:848`）：单残差流 → `hc_mult` 路并行流，最终学权重坍缩 |
| **GLM5.3**（`glm5_next`） | **DSA**（Indexer 打分稀疏选 token）＋ **KDA**（Kimi Delta 线性注意力，chunk 化 delta rule） | 标准 RoPE（DSA 层用）；KDA 层不用 RoPE | DeepSeekMoE：`n_routed_experts=288`，sigmoid 门控，**共享专家**；前 3 层 dense | **混合架构**：`layer_types` 交替 `linear_attention`（KDA）/ `deepseek_sparse_attention`；Indexer 还有 full/shared 模式 | **ForgetGate**（`modular:375`）、SafeGate（sigmoid 下限保护，`:399`）、门控化 `RMSNormGated`（Qwen3_5 血统，`:409`） |
| **MiniMax M3**（`minimax_m3_vl`） | **DSA 稀疏注意力**（自带 Indexer，`modular:464`）＋**稀疏 KV cache** | 标准 RoPE；视觉侧 3D RoPE（M-RoPE） | DeepSeekV4 式 Experts + sigmoid TopKRouter；前 3 层 dense | 稀疏注意力 + `mlp_layer_types` dense/sparse 混合 | 稀疏 cache **存两份**：K/V（`SparseCacheLayer`，`modular:234`）+ Indexer 单头 K（`idx_keys` 同层追加）；VL 侧 CLIP 编码器 + Qwen2VL 处理器 |
| **HY4**（`hy_v3`） | **ApertusAttention**（= Llama + QK-Norm，`apertus/modular:134-155`） | 标准 | Qwen3MoE 式 Experts + Mixtral 式 TopKRouter | 复用 MiniMax M2 排布 | 最纯粹的「**modular 拼装**」产物——注意力取 Apertus、骨架取 DeepSeekV3 DecoderLayer、Experts 取 Qwen3MoE、Router 取 Mixtral（`hy_v3/modular:35-46`） |
| **Nemotron-H**（`nemotron_h`） | Jamba 式全注意力 **＋ Mamba-2**（Zamba2 血统，门控 SSM） | 仅注意力层用；**SSM 层无位置编码** | dense MLP（NemotronMLP）或 DSv3 式 MoE 变体 | **Block 级混合**：`NemotronHBlock` 按 config 在 Mamba2 / Attention / MoE 三种块间排布（`modular:226`） | SSM 块用 `RMSNormGated`（Zamba2）；SSM 层**没有 KV cache**，只有 conv state + 循环状态 → 推理显存恒定 |

---

## 二、演化脉络（五条线，每条都能对上表）

### 线 1 · KV cache 的「省」法：存全量 → 低秩压缩 → 稀疏选块

```
Llama(全量 K/V) → Mistral(滑窗只留 w) → MLA(低秩压缩潜向量) → DSA/CSA(压缩 + Indexer 只选 top-k 块)
```

- MLA 省显存的前提在 V3.2/GLM 上被 DSA **抵消**——要按 top-k 索引选 K/V 行，压缩潜向量没法直接选，
  只能缓存展开后的完整 K/V（576 → 40960 元素/token/层，**涨 71 倍**，代码自带 TODO 承认）。
  详见 [2M2](./2M2_deepseek_mla_moe.md) 与 [2M7](./2M7_glm_moe_dsa.md)。
- DSv4 的 HCA 走得更彻底：重压缩率（m'=128）+ 滑窗 cache（`DeepseekV4HCACache` 继承 `DynamicSlidingWindowLayer`）。

### 线 2 · 层间混合：同质 → 滑窗/全局 → 线性/全注意力

| 路线 | 代表 | 机制 |
|------|------|------|
| 滑窗/全局交替 | Mistral → Gemma2/3/4 | 5:1 局部/全局（Gemma），局部层小 θ、全局层大 θ |
| 线性层替代多数注意力层 | **GLM5.3**（KDA）、**Nemotron-H**（Mamba-2） | 少数层保留全注意力兜底**精确检索**，其余层 O(S) 或 O(1) |
| 压缩注意力逐层交替 | **DSv4** | HCA / CSA 按 `layer_types` 排，配 `mlp_layer_types` 同步调 MLP |

### 线 3 · MoE 细化：粗粒度 → 细粒度+共享 → 确定性路由

```
Mixtral(top-2 粗粒度, softmax) → Qwen3MoE(细粒度) → DeepSeek(细粒度+共享专家, sigmoid+偏置均衡) → DSv4(HashRouter 确定性)
```

- DeepSeek 系 **没有** 负载均衡辅助损失（`aux_loss` 零命中），靠偏置项做无辅助损失均衡（见 [2M3](./2M3_qwen3_moe.md)）。
- DSv4 的 `HashRouter`（`deepseek_v4/modular:930`）用 token 哈希**确定性**地分专家，完全去掉学习路由。

### 线 4 · 归一化加码：1× → 2×（QK-Norm）→ 4× → 门控化

| 模型 | norm 数量 / 位置 |
|------|------------------|
| Llama / Qwen2 | 每层 2 个（attn 前、MLP 前） |
| Qwen3 / Gemma / HY4 | 2 个 + 注意力内 **QK-Norm**（逐头） |
| Gemma2/3/4 | **4 个**（input / post-attn / pre-ffn / post-ffn） |
| GLM5.3 / Nemotron-H | **RMSNormGated**（norm 里带门，Qwen3_5 / Zamba2 血统） |

### 线 5 · 残差流：单流 → 多流（HyperConnection）

DSv4 与 GLM5.3 引入 **HyperConnection**（mHC，Xie et al. 2026，`deepseek_v4/modular:769`）：
残差从一条流扩成 `hc_mult` 条并行流，每个子层学一组 collapse/expand 权重；模型出口再用
`HyperHead`（`:848`）把多流加权坍缩回单流进最终 RMSNorm。这是对「Pre-Norm 残差」这个最古老组件的第一次主流改动。

---

## 三、modular 继承关系速查（读哪个文件）

| 模型 | 读哪个 | 主要父类来源（import 实测） |
|------|--------|------------------------------|
| `llama` | `modeling_llama.py`（无 modular） | 祖宗，不继承任何人 |
| `qwen3` | `modular_qwen3.py` | Qwen2（RMSNorm/Rotary/任务头）、Llama（Attention）、**Gemma（MLP）** |
| `gemma3` | `modular_gemma3.py` | Gemma2、PaliGemma、Siglip |
| `gemma4` | `modular_gemma4.py` | Gemma3、Gemma3n、Llama（Rotary）、Mixtral（Experts）、moonshine_streaming（滑窗 mask） |
| `deepseek_v3` | `modular_deepseek_v3.py` | DeepSeekV2（Router/MoE/Attention）、Qwen2MoE（MLP/Experts）、Llama（骨架） |
| `deepseek_v4` | `modular_deepseek_v4.py` | DeepSeekV3（RMSNorm）、**Laguna（Rotary）**、Llama（MLP/Model）、Mixtral（Experts/Router/PreTrainedModel） |
| `glm5_next` | `modular_glm5_next.py` | **12 个模型目录**（GlmMoeDsa、DeepseekV3/V4、Qwen2Moe/Qwen3_5/Qwen3Next、MiniMaxM3VL、Mixtral、Llama、GlmOcr/Glm46V/Glmga、Exaone4_5、inkling） |
| `minimax_m3_vl` | `modular_minimax_m3_vl.py` | MiniMaxM2、DeepseekV4（Experts）、Laguna（SparseMoeBlock）、Gemma3（RMSNorm）、CLIP、Qwen2.5VL/Qwen2VL、Llava |
| `hy_v3` | `modular_hy_v3.py` | Apertus（Attention）、DeepseekV3（DecoderLayer）、Llama（全套基础件）、MiniMaxM2（Model/MoE）、Qwen3MoE（Experts）、Mixtral（Router） |
| `nemotron_h` | `modular_nemotron_h.py` | Zamba2（MambaMixer/RMSNormGated）、Jamba（Attention）、DeepseekV3（MoE/Router）、Nemotron（MLP）、Zamba（ForCausalLM）、Llama（RMSNorm） |

> **看点**：`glm5_next` 一个文件 import 了 **12 个模型目录**，是 modular 复用的极端案例；
> `hy_v3` 则是「纯拼装」的代表——自己几乎不写新模块，全部从六个家族各取一件。

---

## 四、验证清单（本篇所有断言的实测命令）

环境限制：本机 conda Python 3.13 无 torch，**全部静态验证，未实跑**。

```bash
# 1. 各模型 modular 的类清单与 import（一、三两节全部表格行）
grep -nE "^class |^def |^from \.\." src/transformers/models/<name>/modular_<name>.py
#    已跑：llama（无 modular）、qwen3、gemma3、gemma4、deepseek_v3、deepseek_v4、
#          glm5_next、minimax_m3_vl、hy_v3、nemotron_h

# 2. HY4 的 Attention 确实是 QK-Norm 变体（不是裸 Llama）
sed -n 134,155p src/transformers/models/apertus/modular_apertus.py
#    → class ApertusAttention(LlamaAttention) 里有 q_norm/k_norm = ApertusRMSNorm(head_dim)

# 3. Gemma 双 RoPE 与 5:1 排布
sed -n 97,160p src/transformers/models/gemma3/configuration_gemma3.py
sed -n 190,205p src/transformers/models/gemma4/configuration_gemma4.py
#    → default_theta = {"global": 1e6, "local": 1e4}；pattern=6；Gemma4 全局层 proportional + 0.25

# 4. Gemma 的 4× norm 与 KV 共享
grep -nE "input_layernorm|post_attention|pre_feedforward|post_feedforward" src/transformers/models/gemma3/modular_gemma3.py   # :402-405
sed -n 1019,1058p src/transformers/models/gemma4/modular_gemma4.py   # is_kv_shared_layer 时不建 k/v_proj

# 5. Gemma MLP 激活是 gelu_pytorch_tanh（GeGLU 而非 SwiGLU）
grep -n "hidden_activation" src/transformers/models/gemma2/modular_gemma2.py   # :103

# 6. DSv4 压缩注意力 / HashRouter / HyperConnection / 交错 RoPE
grep -nE "class DeepseekV4(HCA|CSA|Indexer|HashRouter|HyperConnection|HyperHead)" src/transformers/models/deepseek_v4/modular_deepseek_v4.py
#    → :298 :382 :525 :648 :769 :848 :930
sed -n 46,64p src/transformers/models/deepseek_v4/modular_deepseek_v4.py       # repeat_interleave(2) 交错
sed -n 46,54p src/transformers/models/deepseek_v4/configuration_deepseek_v4.py # 2×HCA + CSA/HCA 交替

# 7. GLM5.3 的 KDA / DSA 混合排布与 ForgetGate
sed -n 160,198p src/transformers/models/glm5_next/configuration_glm5_next.py   # layer_types 交替
grep -nE "class Glm5NextText(ForgetGate|LinearAttention|Indexer|RMSNormGated)" src/transformers/models/glm5_next/modular_glm5_next.py
#    → :375 :597 :749 :409；kda 函数 :441/:495

# 8. MiniMax M3 稀疏 cache 双份（K/V + idx_keys）
sed -n 234,250p src/transformers/models/minimax_m3_vl/modular_minimax_m3_vl.py  # SparseCacheLayer.update_index
grep -n "class MiniMaxM3VLIndexer" src/transformers/models/minimax_m3_vl/modular_minimax_m3_vl.py  # :464

# 9. Nemotron-H 三种 Block 与 Mamba-2 来源
grep -nE "class NemotronH(Mamba2Mixer|Block|MoE|Attention)" src/transformers/models/nemotron_h/modular_nemotron_h.py
#    → :47（Zamba2MambaMixer）:99 :160 :207 :226

# 10. DeepSeek V3 MLA 血统（继承 DeepseekV2）
grep -nE "^class DeepseekV3" src/transformers/models/deepseek_v3/modular_deepseek_v3.py
#    → :87 TopkRouter(DeepseekV2) :129 MoE(DeepseekV2Moe) :135 Attention(DeepseekV2Attention)
```

---

## 五、和 25 篇矩阵的关系

- 本篇是**横向对照**，25 篇是**纵向精读**。表中 DeepSeek V3 的 MLA 细节读 2M2，Qwen3 的 QK-Norm 读 1M4，
  Gemma 的 soft-cap 与交替注意力读 2M1，GLM `glm_moe_dsa` 的 DSA/IndexShare 读 2M7，Mamba-2 的 SSM 机制读 2M5。
- 本篇新出现的 **DSv4 / GLM5.3 / MiniMax M3 / HY4 / Gemma4 / Nemotron-H** 六个模型目前**没有对应精读篇目**。
  若要扩矩阵，建议顺序：DSv4（压缩注意力，承接 2M2）→ GLM5.3（混合层收官）→ Nemotron-H（SSM 混合，承接 2M5）。
