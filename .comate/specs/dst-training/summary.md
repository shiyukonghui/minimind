# DST 动态稀疏训练 - 任务完成总结

## 完成概览

根据 `thinking/DST.md` 的设计思路，在 `DST-train/` 目录下实现了完整的动态稀疏训练（DST）框架，支持从头训练两个模型进行对比实验。

## 产出文件

| 文件 | 说明 |
|------|------|
| `DST-train/dst_hooks.py` | 诊断Hook模块：MBEMonitor（矩阵基熵监控）+ DeadNeuronDetector（假死神经元检测与唤醒） |
| `DST-train/dst_pruning.py` | 剪枝器模块：MagnitudePruner（绝对值剪枝）+ RigLScheduler（RigL动态稀疏训练） |
| `DST-train/train_baseline.py` | 基线模型训练脚本：标准预训练，从头训练，作为对照组 |
| `DST-train/train_dst.py` | DST训练主脚本：三阶段循环（学习+RigL → 剪枝 → 微调+蒸馏），从头训练 |
| `DST-train/convert_for_eval.py` | 模型格式转换：PyTorch → HuggingFace，支持 baseline/dst_phase1/dst_recovered |
| `DST-train/run_eval.sh` | 一键测评脚本：依次评测基线模型和DST模型 |
| `DST-train/model/` | 训练产出目录 |

## 核心设计

### 对比实验架构
- **基线模型**：`train_baseline.py`，标准预训练流程，从头训练（`from_weight='none'`）
- **DST模型**：`train_dst.py`，三阶段动态稀疏训练，从头训练，与基线使用相同的 epochs、batch_size、lr

### DST 三阶段流程
1. **阶段一：学习与成长 + RigL** — 标准预训练 + RigL动态稀疏训练（定期剪枝/生长连接），MBE监控模型饱和状态
2. **阶段二：压缩与巩固** — Magnitude Pruning 剪枝，保存剪枝前模型作为教师
3. **阶段三：恢复与再成长** — 低学习率微调 + 可选知识蒸馏 + 假死神经元唤醒

### 关键参数
- RigL 初始稀疏度：50%（保留50%的连接）
- 额外剪枝比例：20%（阶段二）
- 蒸馏温度：1.5，CE权重：0.5
- MBE饱和阈值：0.3

## 使用方式

```bash
# 1. 训练基线模型
python DST-train/train_baseline.py --epochs 1 --hidden_size 512

# 2. 训练DST模型
python DST-train/train_dst.py --phase1_epochs 1 --hidden_size 512

# 3. 转换为HuggingFace格式
python DST-train/convert_for_eval.py --model_type all

# 4. 统一测评
bash DST-train/run_eval.sh
```

## 验证结果
- 5个Python文件全部通过语法检查（`py_compile`）
- 当前环境未安装torch，无法做运行时验证，需在GPU环境中执行
