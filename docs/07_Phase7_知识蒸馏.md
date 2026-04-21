# Phase 7: 知识蒸馏

> 本阶段目标：理解知识蒸馏原理，掌握白盒蒸馏的实现方法

## 一、知识蒸馏概述

### 1.1 什么是知识蒸馏？

知识蒸馏（Knowledge Distillation, KD）是一种模型压缩技术，让小模型（学生）学习大模型（教师）的知识。

**核心思想**：
- SFT：学生直接学习"硬标签"（真实答案）
- KD：学生学习教师的"软标签"（概率分布）

```
SFT:  学生 → 学习答案 "正确答案是 token 1234"
KD:   学生 → 学习教师的思考过程 "token 1234 概率 0.8, token 5678 概率 0.15..."
```

### 1.2 白盒蒸馏 vs 黑盒蒸馏

| 类型 | 教师模型 | 学习内容 | 数据来源 |
|------|---------|---------|---------|
| 黑盒蒸馏 | 闭源模型（如 GPT-4） | 教师的输出文本 | 教师生成的对话数据 |
| 白盒蒸馏 | 开源模型（可获取内部状态） | 教师的 logits 分布 | 任意数据 |

MiniMind 实现了两种蒸馏：
- **黑盒蒸馏**：使用 `sft_1024.jsonl`、`sft_2048.jsonl`（来自 Qwen2.5 的对话数据）
- **白盒蒸馏**：使用 `train_distillation.py`，让小模型学习大模型的 logits

### 1.3 为什么蒸馏有效？

教师的 softmax 输出包含丰富的"暗知识"：

```
教师对 "1+1=?" 的输出分布:
token "=": 0.80
token "2": 0.15
token "等于": 0.04
token "是": 0.01
```

这个分布告诉学生：
- "=" 是最可能的答案
- "2" 和 "等于" 也是相关的候选
- 这种"相关性"信息是硬标签无法提供的

## 二、蒸馏损失函数

### 2.1 KL 散度损失

```python
def distillation_loss(student_logits, teacher_logits, temperature=1.0):
    """
    计算学生和教师 logits 之间的 KL 散度
    temperature: 软化 logits 的温度参数
    """
    with torch.no_grad():
        # 教师的软标签（软化后的概率分布）
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()

    # 学生的 log softmax
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # KL 散度
    kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')

    # 温度缩放的补偿
    return (temperature ** 2) * kl
```

### 2.2 温度参数的作用

温度 `T` 软化概率分布：

```
原始 logits: [10, 5, 1]
softmax(logits): [0.999, 0.001, 0.000]  → 几乎是硬标签

softmax(logits/T=2): [0.86, 0.13, 0.01]  → 更平滑的分布
softmax(logits/T=5): [0.55, 0.30, 0.15]  → 非常平滑
```

**温度选择**：
- `T=1`：原始分布，接近硬标签
- `T=2~5`：推荐范围，平衡平滑性和信息量
- `T` 过大：分布过于平滑，信息丢失

### 2.3 组合损失

```python
# 总损失 = α × CE损失 + (1-α) × 蒸馏损失
loss = alpha * ce_loss + (1 - alpha) * distill_loss
```

- `alpha=0.5`：CE 和蒸馏各占一半
- `alpha=1.0`：纯 CE（相当于普通 SFT）
- `alpha=0.0`：纯蒸馏（完全学习教师分布）

## 三、白盒蒸馏实现

### 3.1 启动训练

```bash
cd trainer
python train_distillation.py
```

### 3.2 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--student_hidden_size` | 512 | 学生模型维度 |
| `--teacher_hidden_size` | 768 | 教师模型维度 |
| `--alpha` | 0.5 | CE 损失权重 |
| `--temperature` | 1.5 | 蒸馏温度 |
| `--from_student_weight` | full_sft | 学生模型初始权重 |
| `--from_teacher_weight` | full_sft | 教师模型权重 |

### 3.3 训练核心逻辑

```python
def train_epoch(epoch, loader, iters, teacher_model, lm_config_student, ...):
    # 教师模型冻结
    teacher_model.eval()
    teacher_model.requires_grad_(False)

    for step, (X, Y, loss_mask) in enumerate(loader):
        # 学生模型前向传播
        with autocast_ctx:
            res = model(X)
            student_logits = res.logits

        # 教师模型前向传播（不计算梯度）
        with torch.no_grad():
            teacher_logits = teacher_model(X).logits
            # 如果词表大小不同，截断教师 logits
            vocab_size_student = student_logits.size(-1)
            teacher_logits = teacher_logits[..., :vocab_size_student]

        # 1) CE 损失（硬标签）
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            Y.view(-1),
            reduction='none'
        )
        ce_loss = (ce_loss * loss_mask.view(-1)).sum() / loss_mask.sum()

        # 2) 蒸馏损失（软标签）
        distill_loss = distillation_loss(
            student_logits.view(-1, V)[loss_mask == 1],
            teacher_logits.view(-1, V)[loss_mask == 1],
            temperature=temperature
        )

        # 3) 组合损失
        loss = alpha * ce_loss + (1 - alpha) * distill_loss
        loss.backward()
        optimizer.step()
```

### 3.4 词表大小处理

如果学生和教师的词表大小不同：

```python
# 教师词表更大时，截断教师 logits
vocab_size_student = student_logits.size(-1)
teacher_logits = teacher_logits[..., :vocab_size_student]
```

## 四、黑盒蒸馏

### 4.1 数据来源

`sft_1024.jsonl` 和 `sft_2048.jsonl` 来自 Qwen2.5-7B/72B-Instruct 的对话数据，本质上是黑盒蒸馏的结果。

### 4.2 使用方法

黑盒蒸馏与普通 SFT 完全一致，只是数据来源不同：

```bash
# 使用 Qwen2.5 蒸馏数据进行 SFT
python train_full_sft.py --data_path ../dataset/sft_1024.jsonl --max_seq_len 1024
```

## 五、动手练习

### 练习 1：白盒蒸馏训练

使用 MiniMind2-Small（512dim）作为学生，MiniMind2（768dim）作为教师：

```bash
cd trainer
python train_distillation.py --student_hidden_size 512 --teacher_hidden_size 768 --epochs 6
```

### 练习 2：对比蒸馏效果

```bash
# 测试学生模型（蒸馏后）
python eval_llm.py --weight full_dist

# 测试原始学生模型（未蒸馏）
python eval_llm.py --weight full_sft --hidden_size 512
```

对比两种模型的回答质量。

### 练习 3：调整温度参数

尝试不同的温度值（1.0, 2.0, 5.0），观察蒸馏效果的变化。

## 六、下一阶段预告

下一阶段 [Phase 8: 强化学习后训练](./08_Phase8_强化学习后训练.md)，我们将学习：
- DPO 直接偏好优化
- PPO 近端策略优化
- GRPO 分组相对策略优化
- SPO 单流策略优化