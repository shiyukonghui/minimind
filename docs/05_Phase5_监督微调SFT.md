# Phase 5: 监督微调（SFT）

> 本阶段目标：理解 SFT 的原理与实现，让预训练模型学会对话

## 一、SFT 概述

### 1.1 为什么需要 SFT？

预训练后的模型只会"词语接龙"，还不会与人聊天。SFT（Supervised Fine-Tuning）阶段需要：

1. 给模型施加聊天模板，让它知道什么是"问题"、什么是"回答"
2. 让模型学会在合适的位置停止生成（而不是无限接龙）
3. 让模型学会遵循指令格式

**类比**：让学富五车的"牛顿"先生适应 21 世纪智能手机的聊天习惯——学习屏幕左侧是对方消息，右侧是本人消息这个规律。

### 1.2 SFT vs 预训练的关键区别

| 方面 | 预训练 | SFT |
|------|--------|-----|
| 数据格式 | 纯文本 | 对话格式 |
| 损失计算 | 所有 token | 仅 assistant 回复 token |
| 学习率 | 5e-4 | 5e-7（小 1000 倍） |
| 训练目标 | 学习语言规律 | 学习对话方式 |

## 二、训练脚本详解

### 2.1 启动训练

```bash
cd trainer
python train_full_sft.py

# 多卡训练
torchrun --nproc_per_node N train_full_sft.py

# 断点续训
python train_full_sft.py --from_resume 1
```

### 2.2 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 2 | 训练轮数 |
| `--batch_size` | 16 | 批次大小 |
| `--learning_rate` | 5e-7 | 学习率（远小于预训练） |
| `--from_weight` | pretrain | 基于预训练权重 |
| `--data_path` | ../dataset/sft_mini_512.jsonl | SFT 数据路径 |
| `--max_seq_len` | 512 | 最大序列长度 |

**注意**：SFT 的学习率（5e-7）远小于预训练（5e-4），因为微调阶段只需小幅调整模型参数，避免"遗忘"预训练知识。

## 三、SFT 核心机制

### 3.1 ChatML 对话模板

MiniMind 使用 ChatML 格式组织对话：

```
<|im_start|>system
你是一个有用的AI助手<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好！很高兴见到你！<|im_end|>
```

`apply_chat_template` 自动将对话数据转为上述格式：

```python
messages = [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！"}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
```

### 3.2 Loss Mask — 只学习回答

SFT 最关键的设计是 **Loss Mask**：只在 assistant 回复位置计算损失。

```python
def _generate_loss_mask(self, input_ids):
    loss_mask = [0] * len(input_ids)  # 初始化全 0
    i = 0
    while i < len(input_ids):
        # 找到 <|im_start|>assistant 标记
        if input_ids[i:i + len(self.bos_id)] == self.bos_id:
            start = i + len(self.bos_id)
            # 找到 <|im_end|> 标记
            end = start
            while end < len(input_ids):
                if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                    break
                end += 1
            # assistant 回复区域设为 1
            for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                loss_mask[j] = 1
            i = end + len(self.eos_id)
        else:
            i += 1
    return loss_mask
```

**可视化示例**：

```
Token:  <|im_start|> user 你好 <|im_end|> <|im_start|> assistant 你好！ <|im_end|>
Mask:       0        0    0    0      0         0          1     1    1     0
                                                                    ↑
                                                            只在回答部分计算损失
```

### 3.3 损失计算

```python
with autocast_ctx:
    res = model(X)
    # 逐 token 计算交叉熵
    loss = loss_fct(
        res.logits.view(-1, res.logits.size(-1)),
        Y.view(-1)
    ).view(Y.size())
    # 应用 loss_mask
    loss = (loss * loss_mask).sum() / loss_mask.sum()
    # MoE 辅助损失
    loss += res.aux_loss
    loss = loss / args.accumulation_steps
```

**关键**：`(loss * loss_mask).sum() / loss_mask.sum()` 确保只在 assistant 回复位置计算损失，且损失值不受 padding 影响。

## 四、训练流程对比

### 4.1 与预训练的代码差异

```python
# 预训练：使用 PretrainDataset，所有 token 参与 loss
train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)

# SFT：使用 SFTDataset，只有 assistant 回复参与 loss
train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
```

```python
# 预训练：from_weight='none'（从头训练）
model, tokenizer = init_model(lm_config, 'none', device=args.device)

# SFT：from_weight='pretrain'（基于预训练权重）
model, tokenizer = init_model(lm_config, 'pretrain', device=args.device)
```

### 4.2 训练开销

| 模型 | 数据集 | 单卡 3090 时间 | 成本 |
|------|--------|---------------|------|
| MiniMind2-Small | sft_mini_512.jsonl | ~1h | ~1.3元 |
| MiniMind2 | sft_mini_512.jsonl | ~3.3h | ~4.29元 |

## 五、SFT 数据选择策略

### 5.1 快速训练方案

使用 `sft_mini_512.jsonl`（1.2GB），最快速度获得可对话模型。

### 5.2 完整训练方案

使用 `sft_512.jsonl` + `sft_2048.jsonl`，效果更好但耗时更长。

### 5.3 长度外推

SFT 默认 `max_seq_len=512`，如需更长的对话能力：

1. 使用 `sft_1024.jsonl` 或 `sft_2048.jsonl` 进行微调
2. 设置对应的 `max_seq_len`
3. 配合 RoPE 长度外推（YaRN 算法）

## 六、动手练习

### 练习 1：启动 SFT 训练

```bash
cd trainer
python train_full_sft.py --epochs 2 --batch_size 8
```

### 练习 2：测试 SFT 模型

```bash
python eval_llm.py --weight full_sft
```

对比预训练模型和 SFT 模型的输出差异。

### 练习 3：自定义 SFT 数据

创建一个自定义的 SFT 数据文件，格式如下：

```json
{"conversations": [{"role": "user", "content": "自定义问题"}, {"role": "assistant", "content": "自定义回答"}]}
```

使用自定义数据进行 SFT 训练，观察模型行为变化。

## 七、下一阶段预告

下一阶段 [Phase 6: LoRA 微调](./06_Phase6_LoRA微调.md)，我们将学习：
- LoRA 低秩适配的原理
- 如何用极少参数微调模型
- 垂域模型迁移（医疗、自我认知等）
