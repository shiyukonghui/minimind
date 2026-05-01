# DST训练改进 Spec

## Why
DST（动态稀疏训练）实验中发现三个问题：1) MBE饱和指标始终很高（0.9840→0.9805，几乎不变），无法触发压缩阶段的条件判断；2) DST总训练轮次（阶段1+阶段3=4轮）远超基线（1轮），对比不公平；3) 当前batch size较小导致训练速度偏慢，可适度增大以提速。

## 根因分析：MBE为何对小模型失效

用户观察到的MBE值从 0.9840 降到 0.9805，降幅仅 0.0035（~0.35%），这意味着MBE对于MiniMind(hidden_size=512)这类小模型几乎是一条水平线。

**数学原理**：MBE对权重矩阵做SVD分解后，将归一化奇异值视为概率分布计算香农熵，再除以均匀分布的最大熵。对于 `[512, 512]` 的权重矩阵，根据Marchenko-Pastur定律，随机矩阵的奇异值分布本身就接近均匀（quarter-circle law），归一化熵天然≈1.0。训练过程中权重虽有调整，但512个奇异值的分布形态变化极小——熵值始终悬浮在0.98以上，下降空间几乎为零。

**结论**：MBE是为大模型（ViT、ResNet等权重矩阵尺寸数千上万）设计的指标，在MiniMind的512维权重矩阵上**完全失效**。即使将阈值降到0.05也无法触发，因为MBE根本不会降到那个水平。

**替代方案**：引入 **梯度范数衰减 (Gradient Norm Decay, GND)** 监控。训练初期梯度大（权重随机初始化），随着学习收敛梯度逐渐变小。梯度范数直接度量"模型还在学多少"。当梯度范数降至历史峰值的某个比例（如10%）以下并持续若干个检查点，即判定为饱和。

## What Changes
- **用GND替换MBE作为饱和检测指标**：新增 `GNDMonitor` 类（在 `dst_hooks.py` 中），跟踪梯度L2范数的衰减比例，当 `grad_norm / peak_grad_norm < threshold` 持续多次时判定饱和；移除MBE的饱和触发角色，降级为可选的诊断日志
- **饱和触发阶段二的提前终止逻辑**：在阶段一训练循环中，当GND检测到饱和时通过 `break` 提前跳出（新增 `phase1_max_epochs` 作为硬上限）
- **公平对比轮次**：将阶段三默认轮数 `phase3_epochs` 从 3 降为 0，使DST总训练轮次与基线对齐（1轮），同时保留用户手动指定多轮蒸馏恢复的能力
- **增大batch size**：将阶段一的 `phase1_batch_size` 从 32 提升到 300，去除梯度累积（`phase1_accumulation_steps` 从 8 改为 1），通过增大单batch样本量来提升训练速度
- **BREAKING**: 阶段三默认跳过（`phase3_epochs=0`），之前默认运行3轮的恢复训练现在需要用户显式指定 `--phase3_epochs 3`

## Impact
- Affected specs: 无（新功能）
- Affected code:
  - `DST-train/dst_hooks.py` — 新增 `GNDMonitor` 类，`MBEMonitor` 保留但不再用于饱和判定
  - `DST-train/train_dst.py` — 主训练脚本：参数默认值调整、GND监控集成、阶段一提前终止逻辑、阶段三跳过逻辑
  - `DST-train/train_baseline.py` — 基线batch size同步调整

## ADDED Requirements

### Requirement: 梯度范数衰减 (GND) 监控替代MBE作为饱和检测
系统应当使用梯度范数的衰减比例来判断模型是否达到训练饱和状态，因为梯度范数直接反映"模型还在学多少"，且对任意模型尺寸都有效。

#### Scenario: 梯度范数正常衰减触发饱和
- **GIVEN** 阶段一训练开始，`GNDMonitor` 记录初始梯度范数峰值 `G_peak`
- **WHEN** 在每个监控间隔点，当前梯度范数 `G_cur` 满足 `G_cur / G_peak < gnd_threshold`（默认0.1）且连续 `patience` 次
- **THEN** 系统判定为饱和，提前结束阶段一训练，触发进入阶段二

#### Scenario: 梯度范数未衰减到阈值
- **GIVEN** 阶段一训练开始，`GNDMonitor` 监控中
- **WHEN** 完成 `phase1_max_epochs`（默认=phase1_epochs）轮训练后梯度范数仍未降至阈值以下
- **THEN** 系统输出日志"梯度范数未衰减至饱和阈值，继续进入阶段二"，正常触发阶段二

#### Scenario: GND监控的计算
- **WHEN** `GNDMonitor.check_saturation()` 被调用
- **THEN** 系统遍历所有 `param.grad`（非None的2D权重），计算 `sum(||grad||_2^2)` 再开方得到全局梯度L2范数，返回 `is_saturated` 布尔值、当前GND比值和最新梯度范数值

### Requirement: 阶段一提前终止
系统应当在阶段一训练循环中，在每次GND监控检查后，若检测到饱和则立即通过 `break` 跳出epoch循环，进入阶段二。

#### Scenario: GND饱和触发提前终止
- **GIVEN** 阶段一训练循环运行中，GND监控启用
- **WHEN** `gnd_monitor.check_saturation()` 返回 `is_saturated=True`
- **THEN** 打印日志"[DST] 梯度范数衰减至饱和阈值，提前结束阶段一"，`break` 跳出循环

#### Scenario: 达到硬上限
- **GIVEN** 阶段一训练循环运行中
- **WHEN** epoch索引 >= `phase1_max_epochs`
- **THEN** 自然结束，打印日志"[DST] 达到阶段一最大轮数限制，进入阶段二"

### Requirement: 公平的训练轮次对比
系统应当确保DST和基线训练的总epochs数量在同一量级，使对比实验公平。

#### Scenario: 默认配置下DST和基线训练轮次相同
- **WHEN** 用户使用默认参数运行DST训练（不显式指定 `--phase3_epochs`）
- **THEN** 阶段三被跳过（`phase3_epochs=0`），DST总训练epochs为1，与基线默认的1轮一致

#### Scenario: 用户手动启用蒸馏恢复
- **WHEN** 用户显式指定 `--phase3_epochs 3`
- **THEN** 阶段三正常运行3轮蒸馏恢复训练

### Requirement: 更大的batch size加快训练
系统应当使用更大的默认batch size以减少训练迭代次数，提升训练速度。

#### Scenario: 阶段一使用大batch训练
- **WHEN** 用户使用默认参数运行DST训练
- **THEN** 阶段一使用 `phase1_batch_size=300`、`phase1_accumulation_steps=1` 进行训练

## MODIFIED Requirements

### Requirement: MBE监控角色降级（从"饱和判定"变为"可选诊断日志"）
`MBEMonitor` 类保留（含 `compute_mbe`, `check_model_mbe`, `report` 等方法），但在阶段一中不再用于触发饱和判定。其 `check_model_mbe()` 返回的 `is_saturated` 结果仅用于日志打印（如 `[DST-MBE-Diag]` 前缀），不作为阶段控制逻辑的依据。用户可通过 `--mbe_threshold` 查看其诊断输出，但默认跳过的MBE阈值是 0.3。

#### Scenario: MBE仅作诊断日志
- **WHEN** 阶段一训练中 `step % monitor_interval == 0`
- **THEN** 打印MBE诊断信息（如 `[DST-MBE-Diag] Step 100: MBE=0.9832`），但不检查 `is_saturated`

### Requirement: 基线训练参数（同步增大batch size）
基线训练脚本 `train_baseline.py` 的 `--batch_size` 默认值从 32 提升到 300，`--accumulation_steps` 从 8 改为 1，保持与DST阶段一的完全一致性。

## REMOVED Requirements
无移除的需求。
