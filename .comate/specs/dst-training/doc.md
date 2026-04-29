# DST 动态稀疏训练 - 规范文档

## 1. 需求概述

根据 `thinking/DST.md` 文档的设计思路，在 `DST-train/` 目录下实现 MiniMind 模型的动态稀疏训练（Dynamic Sparse Training, DST）完整流程。

**核心目标**：从头训练两个相同架构的模型进行对比实验：
- **基线模型（Baseline）**：标准预训练流程，不使用 DST
- **DST 模型**：在预训练过程中引入动态稀疏训练（RigL），实现"边训练边重组"

两个模型使用**相同的模型大小、相同的训练数据、相同的 epochs**，训练完成后统一用 `eval_benchmark.py` 测评，对比 DST 训练的有效性。

DST 训练流程包含三个核心阶段：
1. **阶段一：学习与成长** — 标准预训练 + RigL 动态稀疏训练（定期剪枝/生长连接）
2. **阶段二：压缩与巩固** — 训练结束后，基于绝对值大小的剪枝，进一步压缩模型
3. **阶段三：恢复与再成长** — 低学习率微调恢复性能 + 可选的知识蒸馏

## 2. 技术架构

### 2.1 整体流程

```
                  ┌────────────────────────────────────────────┐
                  │        从头训练（不加载预训练权重）           │
                  └──────────────────┬─────────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
                    ▼                                 ▼
    ┌──────────────────────────┐      ┌──────────────────────────────┐
    │ 基线模型 (Baseline)      │      │ DST 模型                      │
    │ 标准预训练流程            │      │ 预训练 + RigL动态稀疏训练      │
    │ train_baseline.py        │      │ train_dst.py                  │
    │ (复用train_pretrain逻辑) │      │ 阶段一: 学习+RigL剪枝/生长    │
    └──────────┬───────────────┘      │ 阶段二: 压缩剪枝              │
               │                      │ 阶段三: 恢复微调+蒸馏          │
               │                      └──────────┬───────────────────┘
               │                                 │
               ▼                                 ▼
    DST-train/model/baseline_512.pth    DST-train/model/dst_phase1_512.pth
                                        DST-train/model/dst_pruned_512_sp{X}.pth
                                        DST-train/model/dst_recovered_512_sp{X}.pth
               │                                 │
               ▼                                 ▼
    DST-train/model/baseline_hf/       DST-train/model/dst_hf/
               │                                 │
               └────────────┬────────────────────┘
                            ▼
                   eval_benchmark.py
                   C-Eval / CMMLU 评测结果对比
```

### 2.2 文件结构

```
DST-train/
├── train_baseline.py     # 基线模型训练脚本（标准预训练，从头训练）
├── train_dst.py          # DST训练主脚本（三阶段循环，从头训练）
├── dst_pruning.py        # 剪枝器：Magnitude Pruning + RigL动态稀疏训练
├── dst_hooks.py          # 诊断Hook：MBE熵监控 + 假死神经元检测
├── convert_for_eval.py   # 将训练产出转为HuggingFace格式供测评
├── run_eval.sh           # 一键测评脚本（同时测评基线+DST模型）
└── model/                # 训练产出目录
    ├── baseline_512.pth            # 基线模型权重
    ├── baseline_hf/                # 基线模型HuggingFace格式
    ├── dst_phase1_512.pth          # DST阶段一产出权重
    ├── pruning_masks_sp{X}.pt      # 剪枝掩码
    ├── dst_pruned_512_sp{X}.pth    # 剪枝后权重
    ├── dst_recovered_512_sp{X}.pth # 恢复后最终权重
    └── dst_hf/                     # DST模型HuggingFace格式
```

## 3. 模块详细设计

### 3.1 dst_hooks.py — 诊断Hook模块

**职责**：实时监控模型训练状态，提供阶段切换的判断依据。

#### 核心类：`MBEMonitor`

- **矩阵基熵（MBE）计算**：对每层权重矩阵做SVD分解，计算归一化熵
  ```python
  def compute_mbe(weight_matrix):
      # weight_matrix: [out_features, in_features]
      U, S, Vh = torch.linalg.svd(weight_matrix.float(), full_matrices=False)
      S_norm = S / S.sum()
      entropy = -(S_norm * torch.log(S_norm + 1e-10)).sum()
      max_entropy = torch.log(torch.tensor(S.shape[0], dtype=torch.float))
      return entropy / max_entropy  # 归一化到 [0, 1]
  ```
- **饱和度判断**：当 MBE 持续 N 步低于阈值（如 0.3），认为模型"学累了"
- **Hook注册**：在 Transformer Block 的 Self-Attention 和 FFN 层后注册 `register_forward_hook`
- **稀疏度统计**：计算模型中零权重占比

#### 核心类：`DeadNeuronDetector`

- **假死检测**：统计每个神经元在最近 N 个 batch 中激活值全为零的次数
- **唤醒策略**：对检测到的假死神经元，用极小随机值重新初始化对应权重行
  ```python
  def wake_dead_neurons(model, scale=1e-4):
      for name, param in model.named_parameters():
          if 'weight' in name and param.dim() == 2:
              dead_mask = (param.abs().sum(dim=1) == 0)
              if dead_mask.any():
                  param.data[dead_mask] = torch.randn_like(param.data[dead_mask]) * scale
  ```

### 3.2 dst_pruning.py — 剪枝器模块

**职责**：执行结构化和非结构化剪枝，管理稀疏性掩码，实现 RigL 动态稀疏训练。

#### 核心类：`MagnitudePruner`

- **剪枝方法**：基于权重绝对值大小
  ```python
  def compute_mask(model, sparsity=0.2):
      """计算剪枝掩码，保留 (1-sparsity) 比例的权重"""
      masks = {}
      for name, param in model.named_parameters():
          if 'weight' in name and param.dim() == 2:
              threshold = torch.quantile(param.data.abs().flatten(), sparsity)
              masks[name] = (param.data.abs() > threshold).float()
      return masks
  ```

- **应用掩码**：将掩码乘以权重，实际执行剪枝
  ```python
  def apply_mask(model, masks):
      for name, param in model.named_parameters():
          if name in masks:
              param.data *= masks[name]
  ```

- **稀疏度统计**：计算模型整体和各层的实际稀疏度

#### 核心类：`RigLScheduler`（RigL动态稀疏训练）

- **核心思想**：在训练过程中周期性地"剪枝最不重要的连接，同时生长新的连接"
- **关键参数**：
  - `initial_sparsity`: 初始稀疏度（如 0.5，即训练开始就只保留50%的连接）
  - `update_frequency`: 每 N 步执行一次剪枝-生长
  - `T_end`: 在总步数的什么比例后停止动态调整（如 0.8 = 80%后固定掩码）
- **生长策略**：基于梯度的生长（RigL），选择梯度绝对值最大的未激活连接进行生长
  ```python
  def update(self, model, masks, step, total_steps):
      """RigL更新：剪枝最不重要的连接，生长梯度最大的连接"""
      if step / total_steps > self.T_end:
          return masks  # 超过T_end后固定掩码
      if step % self.update_frequency != 0:
          return masks
      for name, param in model.named_parameters():
          if name in masks and param.grad is not None:
              # 计算需要调整的连接数量
              n_params = param.numel()
              n_prune = int(n_params * self.prune_fraction)
              # 在活跃连接中剪枝权重绝对值最小的
              active_weights = param.data.abs() * masks[name]
              _, prune_idx = active_weights.flatten().topk(n_prune, largest=False)
              # 在非活跃连接中生长梯度绝对值最大的
              grad_abs = param.grad.abs() * (1 - masks[name])
              _, grow_idx = grad_abs.flatten().topk(n_prune)
              # 执行剪枝和生长
              masks[name].flatten()[prune_idx] = 0.0
              masks[name].flatten()[grow_idx] = 1.0
      return masks
  ```

### 3.3 train_baseline.py — 基线模型训练脚本

**职责**：标准预训练流程，从头训练（不加载预训练权重），作为对照组。

- 复用 `PretrainDataset` 和标准训练循环（与 `train_pretrain.py` 逻辑一致）
- `from_weight='none'`，即随机初始化后直接训练
- 训练参数与 DST 模型的阶段一完全一致（相同 epochs、batch_size、lr 等）
- 训练完成后保存权重到 `model/baseline_512.pth`

#### 命令行参数

```python
--hidden_size          # 隐藏层维度 (默认512)
--num_hidden_layers    # 层数 (默认8)
--use_moe              # 是否使用MoE (默认0)
--device               # 设备
--dtype                # 精度
--epochs               # 训练轮数 (默认1)
--batch_size           # 批大小 (默认32)
--learning_rate        # 学习率 (默认5e-4)
--accumulation_steps   # 梯度累积 (默认8)
--data_path            # 预训练数据路径
--max_seq_len          # 最大序列长度 (默认512)
--save_interval        # 保存间隔
--log_interval         # 日志间隔
```

### 3.4 train_dst.py — DST训练主脚本

**职责**：实现三阶段训练循环的完整编排，从头训练。

#### 阶段一：学习与成长 + RigL

- `from_weight='none'`，从头训练
- 复用 `PretrainDataset` 和标准训练循环
- 训练参数：`lr=5e-4`, `batch_size=32`, `accumulation_steps=8`（与基线完全一致）
- **RigL 动态稀疏**：训练开始时按 `initial_sparsity` 随机初始化稀疏掩码，每隔 `update_frequency` 步执行剪枝-生长
- **MBE 监控**：每 `monitor_interval` 步调用 `MBEMonitor` 检查模型状态
- **掩码应用**：每个训练步后，将掩码应用到权重上，确保被剪枝的连接权重为零

#### 阶段二：压缩与巩固

- 调用 `MagnitudePruner.compute_mask()` 计算剪枝掩码（在RigL已有的稀疏基础上进一步剪枝）
- 保存阶段一的原始模型作为教师模型（用于阶段三蒸馏）
- 调用 `MagnitudePruner.apply_mask()` 应用剪枝
- 保存剪枝掩码到 `model/pruning_masks_sp{X}.pt`
- 打印各层稀疏度统计

#### 阶段三：恢复与再成长

- **微调恢复**：使用 SFT 数据集，极低学习率（`lr=5e-7`），少量 epoch（2~3）
- **知识蒸馏**（可选）：使用阶段一模型作为教师，`alpha=0.5, temperature=1.5`
- **假死唤醒**：每个 epoch 开始时调用 `DeadNeuronDetector.wake_dead_neurons()`
- 产出最终模型权重

#### 命令行参数

```python
# 通用参数
--hidden_size          # 隐藏层维度 (默认512)
--num_hidden_layers    # 层数 (默认8)
--use_moe              # 是否使用MoE (默认0)
--device               # 设备
--dtype                # 精度

# 阶段一参数（与基线一致）
--phase1_data_path     # 预训练数据路径
--phase1_epochs        # 预训练轮数 (默认1, 与基线相同)
--phase1_batch_size    # 批大小 (默认32)
--phase1_lr            # 学习率 (默认5e-4)
--phase1_accumulation_steps  # 梯度累积 (默认8)

# 阶段二参数
--sparsity            # 额外剪枝比例 (默认0.2)

# 阶段三参数
--phase3_data_path    # 恢复训练数据路径
--phase3_epochs       # 恢复训练轮数 (默认3)
--phase3_batch_size   # 批大小 (默认16)
--phase3_lr           # 学习率 (默认5e-7)
--use_distillation    # 是否使用知识蒸馏 (默认1)
--alpha               # 蒸馏CE权重 (默认0.5)
--temperature         # 蒸馏温度 (默认1.5)

# DST参数
--initial_sparsity    # RigL初始稀疏度 (默认0.5)
--update_frequency    # RigL更新频率 (默认100步)
--T_end               # RigL停止调整比例 (默认0.8)
--monitor_interval    # MBE监控间隔 (默认100步)
--mbe_threshold       # MBE饱和阈值 (默认0.3)
```

### 3.5 convert_for_eval.py — 模型格式转换

**职责**：将 `DST-train/model/` 下的 PyTorch 权重转换为 HuggingFace 格式，供 `eval_benchmark.py` 使用。

- 支持转换多个模型：基线模型、DST 各阶段模型
- 读取 PyTorch 权重文件 → 构建 `MiniMindForCausalLM` 模型 → `save_pretrained()` → 复制分词器
- 命令行参数 `--model_type` 可选 `baseline`/`dst_phase1`/`dst_recovered`

### 3.6 run_eval.sh — 一键测评脚本

```bash
#!/bin/bash
# 对基线模型和DST模型进行统一评测并对比
echo "========== 评测基线模型 =========="
python eval_benchmark.py --model_path DST-train/model/baseline_hf --dataset all

echo "========== 评测DST模型 =========="
python eval_benchmark.py --model_path DST-train/model/dst_hf --dataset all
```

## 4. 数据流路径

```
=== 基线模型 ===
dataset/pretrain_t2t_mini.jsonl
    → PretrainDataset → DataLoader → 标准预训练 → model/baseline_512.pth
    → convert_for_eval.py → model/baseline_hf/

=== DST模型 ===
dataset/pretrain_t2t_mini.jsonl
    → PretrainDataset → DataLoader → 阶段一(RigL预训练) → model/dst_phase1_512.pth

dst_phase1_512.pth
    → MagnitudePruner.compute_mask() → model/pruning_masks_sp20.pt
    → MagnitudePruner.apply_mask() → model/dst_pruned_512_sp20.pth

dataset/sft_t2t_mini.jsonl + dst_phase1_512.pth(教师)
    → SFTDataset → DataLoader → 阶段三(微调+蒸馏) → model/dst_recovered_512_sp20.pth

dst_recovered_512_sp20.pth
    → convert_for_eval.py → model/dst_hf/

=== 评测 ===
model/baseline_hf/ + model/dst_hf/
    → eval_benchmark.py → C-Eval/CMMLU 评测结果对比
```

## 5. 受影响的文件

| 文件路径 | 修改类型 | 说明 |
|---------|---------|------|
| `DST-train/train_baseline.py` | 新建 | 基线模型训练脚本（标准预训练，从头训练） |
| `DST-train/train_dst.py` | 新建 | DST训练主脚本（三阶段循环，从头训练） |
| `DST-train/dst_pruning.py` | 新建 | 剪枝器模块（MagnitudePruner + RigLScheduler） |
| `DST-train/dst_hooks.py` | 新建 | 诊断Hook模块（MBEMonitor + DeadNeuronDetector） |
| `DST-train/convert_for_eval.py` | 新建 | 模型格式转换脚本 |
| `DST-train/run_eval.sh` | 新建 | 一键测评脚本 |
| `DST-train/model/` | 新建目录 | 训练产出模型保存目录 |

**不修改的现有文件**：`model/model_minimind.py`、`trainer/trainer_utils.py`、`eval_benchmark.py` 等均保持不变，DST训练代码通过 import 复用现有模块。

## 6. 边界条件与异常处理

1. **剪枝比例过高**（>50%）：可能导致性能崩溃，限制最大剪枝比例为 50%，并打印警告
2. **MBE 计算异常**：SVD 分解可能数值不稳定，使用 `torch.linalg.svd` 的默认实现，加入 epsilon 防止 log(0)
3. **假死唤醒后的梯度爆炸**：唤醒时使用极小值（1e-4），并在恢复训练时启用梯度裁剪（1.0）
4. **知识蒸馏时教师模型不存在**：若未指定教师模型权重，自动跳过蒸馏，仅使用 CE 损失微调
5. **GPU 内存不足**：RigL 的梯度统计需要额外内存，在小显卡上可降低 `initial_sparsity` 或增大 `update_frequency`
6. **HuggingFace 转换失败**：权重 key 不匹配时使用 `strict=False` 加载，并打印缺失/多余的 key
7. **两个模型公平对比**：确保使用完全相同的随机种子、数据顺序、训练步数

## 7. 预期产出

1. **训练完成**后，`DST-train/model/` 下包含：
   - `baseline_512.pth` — 基线模型权重（标准预训练）
   - `dst_phase1_512.pth` — DST阶段一训练后的权重
   - `pruning_masks_sp{X}.pt` — 剪枝掩码
   - `dst_pruned_512_sp{X}.pth` — 剪枝后（未恢复）的模型权重
   - `dst_recovered_512_sp{X}.pth` — 恢复后的最终DST模型权重
   - `baseline_hf/` — 基线模型HuggingFace格式
   - `dst_hf/` — DST模型HuggingFace格式

2. **评测结果**：对比基线模型与 DST 模型在 C-Eval/CMMLU 上的准确率，验证 DST 训练流程的有效性。关键对比维度：
   - 相同训练步数下的模型性能
   - DST 模型的实际参数量（去除零权重后）vs 基线模型参数量
   - 压缩比对性能的影响
