# MiniMind Datasets

将所有下载的数据集文件放置到当前目录.

Place the downloaded dataset file in the current directory.

## 数据集统计信息

| 数据集文件 | 数据量（行数） |
|-----------|--------------|
| pretrain_t2t.jsonl | 8,468,827 |
| sft_t2t.jsonl | 5,109,432 |
| pretrain_t2t_mini.jsonl | 1,270,238 |
| sft_t2t_mini.jsonl | 905,718 |
| lora_exam.jsonl | 52,762 |
| agent_rl.jsonl | 39,988 |
| lora_medical.jsonl | 25,276 |
| agent_rl_math.jsonl | 20,000 |
| rlaif.jsonl | 19,502 |
| lora_identity.jsonl | 91 |

## 数据集说明

### 预训练数据集
- **pretrain_t2t.jsonl**: 完整预训练数据集，包含 8,468,827 条样本
- **pretrain_t2t_mini.jsonl**: 迷你预训练数据集，包含 1,270,238 条样本（用于快速实验）

### 监督微调数据集
- **sft_t2t.jsonl**: 完整 SFT 数据集，包含 5,109,432 条样本
- **sft_t2t_mini.jsonl**: 迷你 SFT 数据集，包含 905,718 条样本（用于快速实验）

### LoRA 微调数据集
- **lora_exam.jsonl**: 考试相关数据集，包含 52,762 条样本
- **lora_medical.jsonl**: 医疗相关数据集，包含 25,276 条样本
- **lora_identity.jsonl**: 身份识别数据集，包含 91 条样本

### 强化学习数据集
- **agent_rl.jsonl**: Agent 强化学习数据集，包含 39,988 条样本
- **agent_rl_math.jsonl**: Agent 数学强化学习数据集，包含 20,000 条样本
- **rlaif.jsonl**: RLAIF 数据集，包含 19,502 条样本
