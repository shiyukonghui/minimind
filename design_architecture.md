# MiniMind 融合实验架构总体设计

> 版本：v1.0（2026-10-11 定稿）
> 主题：冻结底座 + LoRA 专家池 + Jev 决策路由 + 循环深度 + Engram 记忆轴 + 字节化通用记忆 + RSI 自举
> 配套资料：[ssr-lora.md](file:///f:/MachineLearn/minimind/ssr-lora.md)、[rsi_loop_transformer_research.md](file:///f:/MachineLearn/minimind/rsi_loop_transformer_research.md)

---

## 1. 目标与核心主张

在**个人算力约束**下（无高端 GPU、128GB RAM），验证一套"参数随知识密度自我扩展"的小型语言模型架构。核心约束与主张：

1. **底座全程冻结**：MiniMind-512（hidden=512、8 层、8 头×head_dim 64、2 KV 头、vocab=6400、SwiGLU inter≈1365，已有 `checkpoints/pretrain_512.pth`）不参与梯度更新，**只训练 LoRA 与轻量附加模块**。
2. **参数扩展 = 稀疏解锁，而非稠密增长**：预留 LoRA 专家池（K_max=32），由决策路由按需激活/生长，实现"大参数空间预留 + 逐步启用"。
3. **三条正交增长轴**，全部由 Jev 决策模型驱动：
   - **深度轴**：Looped block（循环体 ×T，测试时展开）；
   - **宽度轴**：MoL 模拟 MoE（共享 FFN + LoRA 专家池，top-2 激活）；
   - **能力轴**：RSI 自举（先蒸馏教师工程能力 → 自举写记忆 / 写新技能）。
4. **记忆轴独立**：Engram/PLE 式哈希查表负责 O(1) 陈述性记忆（海马体），MoL 主体负责慢速程序性学习（新皮层），构成互补学习系统。
5. **字节化打通通用记忆**：输入从 subword token 改为字节后，Engram 的查表键与词表解耦，记忆表可跨模型复用/移植。

---

## 2. 总体架构

```
输入字节流 (UTF-8)
   │  [路线α：ALP 两阶段转换，复用现有 subword 模型]
   ▼
字节嵌入 (byte embedding) ──────────────┐
   │                                    │
   ▼                                    │
┌────────────────────────────────────────────────┐
│            Looped Block（循环体 ×T）             │
│  timestep 编码专家（第 t 步用第 t 个 timestep 专家）│
│  DeepLoop 残差缩放：x_{t+1} = x_t + α_t · f(x_t)  │
│                                                │
│  ┌──────────────┐   ┌───────────────────────┐  │
│  │ 共享 FFN      │   │ MoL LoRA 专家池 K_max=32│ │
│  │ (冻结底座)    │   │ top-2 激活 + 负载辅助损失 │  │
│  └──────────────┘   └───────────────────────┘  │
│                                                │
│  Jev 决策头（System-1，输出带类型决策+校准概率）：  │
│   ① halt：继续循环 / 停止（深度分配）              │
│   ② route：选专家（宽度分配）                     │
│   ③ abstain：弃权 → 触发新专家出生（能力生长）      │
│  校准损失：Brier / 温度缩放                       │
│                                                │
│  MemoryBackend（可切换标准接口）：                 │
│   retrieve(ngram_key) → value(d_c) + gate       │
│   ├ HashTableMemory   （自训字节 N-gram 表）      │
│   ├ DistilledMemory   （离线蒸馏 Qwen BF16 表）   │
│   ├ QwenPLEBackend    （键空间桥接复用 Qwen PLE）  │
│   └ ExternalKVMemory  （RSI 热写记忆）            │
└────────────────────────────────────────────────┘
   │
   ▼
lm_head（字节词表 256）→ 输出
```

底座 MiniMind-512 的各层在循环模式下退化为 ALBERT 式单层复制（E1-D 移植初始化与循环架构天然契合）。

---

## 3. 组件设计

### 3.1 初始化消融（E1，四组对照）

| 组 | 初始化方式 | 验证假设 |
|---|---|---|
| E1-A | 随机初始化 | 基线 |
| E1-B | 官方 MiniMind 权重 | 上限参照 |
| E1-C | **分布匹配随机**：按大模型权重统计画像（逐层类型 mean/std + 重尾形状）采样 | "统计形态先验"独立假设 |
| E1-D | **真实权重移植**：从 Qwen2.5-0.5B bf16 切块（head_dim=64 与 MiniMind 同构，整头对齐） | "已学习特征迁移"独立假设 |

- 供体注意：本机 Qwen3.8-Flash GGUF 是 qwen4exp 架构（512 专家 MoE、head_dim=256），**不能**作 E1-D 供体；IQ3_S 量化反量化有噪声，不能作 E1-C 画像源。
- 字节化后 embedding/lm_head 的词表不匹配问题消失（键空间统一为 256 字节）。

### 3.2 MoL 模拟 MoE

- 共享 FFN（冻结底座）+ LoRA 专家池 K_max=32，每 token top-2 激活。
- 负载辅助损失（沿用 MiniMind MoE 配置的 `aux_loss_alpha` 思路）。
- 路由对比三方案：softmax 路由器 / **Jev 决策路由** / 随机路由（消融）。

### 3.3 Jev 式决策路由（System-1）

- 输出**带类型决策 + 校准概率**：halt（循环停止）、route（专家选择）、abstain（弃权升级）。
- 训练目标 = 决策损失 + 校准损失（Brier / 温度缩放）。
- abstain 触发"新专家出生"：用 E1-C 分布画像作为出生初始化原语。

### 3.4 Looped Block（深度轴）

- 循环体 ×T，timestep 编码区分第 t 步（对应专家/嵌入）。
- DeepLoop 残差缩放控制循环稳定性。
- halt 头实现逐 token 动态深度（MoR / Ouro 思路，见 rsi_loop 资料第 3 节）。

### 3.5 记忆轴：Engram/PLE 与通用标准模块

**技术族**：DeepSeek Engram（arXiv 2601.07372）——N-gram 哈希查表 + 门控，O(1) 静态记忆；固定算力预算下神经计算与记忆存在 **U 型分配定律**，混合配置最优。

**通用标准接口 MemoryBackend**：

```python
class MemoryBackend:
    def retrieve(self, ngram_key) -> Tensor[value_dim=d_c]  # 规范值空间
    def gate(self, hidden) -> Tensor[0..1]                   # 宿主侧门控
    # 宿主侧低秩投影 W_proj: d_c → hidden_size（唯一需训练部分）
```

统一键 = **字节 N-gram 哈希**；统一值 = 规范嵌入空间 d_c；宿主差异由 W_proj + gate 吸收 → 表可跨模型切换/复用。

**Qwen3.8-Flash PLE 复用（三条路径）**：

已核实事实（GGUF 解析 + HF 索引验证）：
- 表：`per_layer_token_embd.weight` [160, 320,001,536]，**IQ4_NL 量化**（4.5bpw，28.8GB，分片2 数据段基址=192、每行 90 字节），16 头 × ~20M 槽（head_offsets/vocab_sizes 累加=320,001,446，pad 90 行到 256 块边界），ngram_size=3，仅插 blk.1；门控 ple_key/ple_value 为 **BF16**（非量化）；
- 哈希公式（已逐行核实 `qwen4exp.cpp` L1036-1047）：**对 n=2 和 n=3 两种窗口分别哈希**，`mixed_n=(t[p]·m0)^(t[p-1]·m1)^…^(t[p-n+1]·m_{n-1})`，`row=mixed_n%head_vocab[h]+head_offset[h]`；每种 n-gram 各 8 头（heads_per_ngram=8），共 16 头；EOS 重置窗口（EOS 之前的位置读作 EOS）；
- 门控公式（L1130-1212）：`key=W_key·emb, value=W_val·emb`；分组 RMSNorm(key/query)；`s=Σ(key⊙query)/√d`；`gate=σ(sign(s)·√|s|)`（带符号平方根）；`out = hidden + gate⊙value + SiLU(causal depthwise conv(k=4, dilation=3))`；
- 超参（GGUF 元数据核实）：multipliers=3×i64、head_offsets/vocab_sizes 各 16×i64（每头 ~20M 槽）、`ple.layers=[1]` 仅插 blk.1、hyper-connection `hc_mult=4`（ple_key 形状 [10240,2560]=4×2560）、image_token=248056、eos=248044；
- config：`heads_per_ngram=8`、`ngram_vocab_size_base=20M`、`split_ngram_parts=128`、`ple_embed_dim=2560`（=16×160）；
- HF 存储：128 个 `ngram_embedding.shard_*.weight`，BF16，每片 [2500012, 160]，行主序（GGUF 为转置），分布于 `model-00005~00037` 共 33 片 ≈104GB。

| 路径 | 做法 | 训练量 | 依赖 |
|---|---|---|---|
| 1. QwenPLEBackend 桥接 | bytes → 现场 Qwen BPE → 哈希 → mmap 查表（IQ4_NL 反量化）→ 低秩投影+门控（约 17.6 万参数） | ~10-20 万参数 | 现有 GGUF，**已实现并测试通过** |
| 2. 离线蒸馏建自有字节表 | 用 BF16 源表蒸馏出字节键小表 | 蒸馏管道 | BF16 分片下载 |
| 3. 探针式知识先验清单 | 只读表做知识探针/评测先验 | 0 | 现有 GGUF |

### 3.6 字节化路线（Engram 通用的前提）

- 依据：ByT5 / BLT / Bolmo / u-ByT5 / FastBLT 调研（见会话资料）。
- **路线α（推荐）**：Bolmo ALP 两阶段转换，把现有 `pretrain_512.pth` 低成本转为字节模型，无需从头重训；MiniMind-512 尺寸恰可作 BLT 熵模型。
- 代价：序列长度膨胀，中文 512 上下文 ≈170 字，需配合分块/压缩缓解。
- 收益：E1-D 词表阻断消失 + Engram 键通用 + Qwen PLE 表可蒸馏复用。

### 3.7 RSI 能力轴（三阶段）

1. **蒸馏**：DeepSeek-V4.1-Flash（闭源 API，序列级蒸馏，复用 `trainer/train_distillation.py` 的 distillation_loss）获得工程能力种子。
2. **自举写入**：低风险 = 写记忆（ExternalKVMemory 热写）；高风险 = 写技能（abstain → 新 LoRA 专家出生）。
3. **档案库进化**：DGM 式档案 + 教师锚定防 model collapse（参照 rsi_loop 资料第 2 节）。

### 3.8 LoRA 生命周期与 SSR-Merge

```
出生(E1-C画像初始化) → Jev路由调度 → 共激活矩阵聚类 → SSR-Merge固化 → 归档/回滚
```

- SSR-Merge（arXiv 2606.10617）：免训练，R=QG⁻¹ 子空间信号路由，单样本校准，重参数化为标准 LoRA，推理零开销。
- 聚类依据 = Jev route 头的共激活矩阵；避免语义重叠/域冲突簇合并退化。

---

## 4. 实验矩阵

| 编号 | 配置 | 目的 |
|---|---|---|
| E0 | 冻结底座 + 单 LoRA（无附加模块） | 全架构基线 |
| E1-A/B/C/D | §3.1 四组初始化 | 初始化假设消融 |
| A2 | E0 + MoL 专家池 | 宽度轴收益 |
| A3 | A2 + Jev 路由（vs softmax vs 随机） | 路由方案对比 |
| A4 | A3 + Looped block（T 扫描） | 深度轴收益 |
| A5 | A4 + HashTableMemory（字节 N-gram 自训表） | 记忆轴收益 / U 型分配验证 |
| A6 | A5 记忆源换 QwenPLEBackend | PLE 复用有效性（路径1） |
| A7 | A5 + 路线α字节化（vs subword 对照） | 字节化代价/收益 |
| A8 | A4 + DistilledMemory（BF16 蒸馏表） | 路径2 蒸馏保真 |
| A9 | A8 + SSR 固化 + 专家池轮换 | 合并质量/推理开销 |
| A10 | A9 + RSI 自举（蒸馏→写记忆→写技能） | 能力轴闭环 |
| A10b | A10 去教师锚定 | 坍缩风险消融 |

观测指标：困惑度/基准分、halt 校准度（Brier）、专家负载熵、弃权触发率、记忆命中率、合并后串扰激活图。

---

## 5. 落地路线图（按依赖顺序）

| 步 | 任务 | 依赖 | 状态 |
|---|---|---|---|
| ① | 提取 Qwen tokenizer（HF `merges.txt`/`vocab.json` 落盘）+ 核对哈希公式 | 小文件，优先 | ✅ 完成（tokenizer.json/vocab.json/merges.txt 已下载验证；公式逐行核对并修正文档） |
| ② | QwenPLEBackend：IQ4_NL mmap 读取 + 反量化 + LRU + 哈希查表，只训低秩投影+门控 | ① | ✅ 完成并测试（[model/ple_backend.py](file:///f:/MachineLearn/minimind/model/ple_backend.py)，17.6 万参数） |
| ③ | Looped block 改造：timestep 编码、DeepLoop 残差缩放、Jev 三决策头 + MoL 专家池 | 独立 | ✅ 完成并测试（[model/model_loop_mol.py](file:///f:/MachineLearn/minimind/model/model_loop_mol.py)，54.6 万可训练参数） |
| ④ | E1-C 统计画像脚本 + E1-D 移植脚本（供体 Qwen2.5-0.5B bf16） | 独立 | ✅ 完成并测试（[scripts/e1_init_ablation.py](file:///f:/MachineLearn/minimind/scripts/e1_init_ablation.py)；供体已下载，画像/采样/移植产物均可 strict 加载并前向） |
| ⑤ | MemoryBackend 标准接口（HashTable/ExternalKV/QwenPLEAdapter） | ② | ✅ 完成并测试（[model/memory_backend.py](file:///f:/MachineLearn/minimind/model/memory_backend.py)） |
| ⑥ | 字节化路线α：ALP 转换 `pretrain_512.pth` | 独立 | ⏳ |
| ⑦ | SSR 合并工具 + Jev 共激活聚类 | ③ | ✅ 完成并测试（[model/ssr_merge.py](file:///f:/MachineLearn/minimind/model/ssr_merge.py)；Q 矩阵按 ssr-lora.md 文字近似实现，精确公式待 tech_report 精校） |
| ⑧ | 蒸馏管道（V4.1-Flash API 序列级） | 独立 | ⏳ |
| 后台 | BF16 PLE 分片下载（~104GB，hf-mirror，见 §6） | 不阻塞② | ⏳ 进行中（~9/33 片） |
| 后台 | E1-D 供体 Qwen2.5-0.5B bf16 下载（~1GB，见 §6） | 不阻塞 | ⏳ 进行中 |

**关键实现事实修正**：GGUF 分片2 的 PLE 大表经字节核算实为 **IQ4_NL**（4.5bpw，非早前误记的 IQ3_S）；门控 ple_key/ple_value 为 BF16。反量化码本取自本地 `ggml-common.h kvalues_iq4nl`，端到端测试与 C++ 哈希公式手算一致（含 EOS 重置）。

---

## 6. 数据资产与下载

- 本机 GGUF：`H:\Strata-data\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000{1,2}-of-00002.gguf`（PLE 表在分片 2）。
- llama.cpp 实现参考：`F:\RustProjects\llama.cpp\src\models\qwen4exp.cpp`（哈希/门控/平滑逻辑可直接移植）。
- 自研探针脚本：[scripts/gguf_meta_probe.py](file:///f:/MachineLearn/minimind/scripts/gguf_meta_probe.py)、[scripts/gguf_tensor_scan.py](file:///f:/MachineLearn/minimind/scripts/gguf_tensor_scan.py)。
- BF16 PLE 下载（仅 33 个含表分片 + tokenizer，≈104GB）：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
hf download Qwen/Qwen3.8-Flash-Next `
  --include "model-0000[5-9]-of-00131.safetensors" "model-000[12][0-9]-of-00131.safetensors" "model-0003[0-7]-of-00131.safetensors" `
            "config.json" "tokenizer*.json" "vocab.json" "merges.txt" `
  --local-dir H:\Strata-data\models\Qwen3.8-Flash-Next-BF16
```

---

## 7. 风险登记

1. **512 维自举天花板**：循环增加的是计算而非记忆，存在能力上限墙（记忆轴部分缓解）。
2. **字节化序列膨胀**：中文有效上下文 ~512→170 字，需分块/压缩策略。
3. **IQ4_NL 反量化噪声**：路径1 的表值精度受 4.5bpw 量化限制，需与 BF16 蒸馏表做保真对照。
4. **Jev 信号不足**：小模型上决策/校准头可能欠拟合，弃权阈值需保守初始化。
5. **SSR 域冲突**：语义重叠专家簇合并会退化，聚类质量依赖共激活矩阵。
6. **RSI 坍缩**：自举数据回灌风险，教师锚定 + 档案库多样性约束兜底。

---

## 8. 关键参考

- Engram/PLE：DeepSeek arXiv 2601.07372；Qwen3.8-Flash 技术报告（待下载 PDF 核对公式）
- 字节模型：ByT5 (2105.13627)、BLT (2402.01821)、Bolmo/ALP (2408.15413)、u-ByT5、FastBLT
- 循环深度：Universal Transformers、Huginn (2502.05171)、Ouro (2510.25741)、MoR (2507.10524)、DeepLoop
- 决策路由/RSI：Jev、DGM (2505.22954)、SEAL (2506.10943)、AZR、R-Zero
- LoRA 合并：SSR-Merge (2606.10617)，本地 [ssr-lora.md](file:///f:/MachineLearn/minimind/ssr-lora.md)
