# Phase 4: 预训练（Pretrain）

> 本阶段目标：理解预训练的完整流程，掌握学习率调度、混合精度训练、分布式训练等技术

## 一、预训练概述

### 1.1 什么是预训练？

预训练是让模型"学知识"的阶段。模型通过大量无标注文本学习语言的统计规律和世界知识。

**核心目标**：学会"词语接龙"——给定前文，预测下一个词。

```
输入: "秦始皇"
模型预测: "是中国的第一位皇帝"
```

### 1.2 预训练 vs SFT

| 阶段 | 数据 | 目标 | 监督方式 |
|------|------|------|---------|
| 预训练 | 大规模无标注文本 | 学习语言规律和知识 | 自监督（预测下一个词） |
| SFT | 对话数据 | 学习对话格式和指令遵循 | 监督学习（模仿回答） |

### 1.3 训练开销参考

| 模型 | 数据集 | 单卡 3090 时间 | 成本 |
|------|--------|---------------|------|
| MiniMind2-Small (26M) | pretrain_hq.jsonl | ~1.1h | ~1.43元 |
| MiniMind2 (104M) | pretrain_hq.jsonl | ~3.9h | ~5.07元 |

## 二、训练脚本详解

### 2.1 启动训练

```bash
# 单卡训练
python trainer/train_pretrain.py

# 多卡 DDP 训练
torchrun --nproc_per_node N trainer/train_pretrain.py

# 断点续训
python trainer/train_pretrain.py --from_resume 1
```

### 2.2 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 1 | 训练轮数（建议 1-2 轮） |
| `--batch_size` | 32 | 批次大小 |
| `--learning_rate` | 5e-4 | 初始学习率 |
| `--accumulation_steps` | 8 | 梯度累积步数 |
| `--max_seq_len` | 512 | 最大序列长度 |
| `--hidden_size` | 512 | 隐藏层维度 |
| `--num_hidden_layers` | 8 | Transformer 层数 |
| `--use_moe` | 0 | 是否使用 MoE |
| `--dtype` | bfloat16 | 混合精度类型 |
| `--data_path` | ../dataset/pretrain_hq.jsonl | 数据路径 |

### 2.3 训练脚本核心流程

```python
# 1. 初始化环境和随机种子
local_rank = init_distributed_mode()
setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

# 2. 配置模型参数
lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers)

# 3. 设置混合精度
dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
autocast_ctx = torch.cuda.amp.autocast(dtype=dtype)

# 4. 初始化模型和分词器
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)

# 5. 加载数据集
train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)

# 6. 配置优化器
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

# 7. DDP 包装（多卡训练）
if dist.is_initialized():
    model = DistributedDataParallel(model, device_ids=[local_rank])

# 8. 开始训练
for epoch in range(args.epochs):
    train_epoch(epoch, loader, len(loader), ...)
```

## 三、核心训练逻辑

### 3.1 训练循环

```python
def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()

    for step, (X, Y, loss_mask) in enumerate(loader, start=start_step + 1):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # 学习率调度
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 混合精度前向传播
        with autocast_ctx:
            res = model(X)
            # 计算交叉熵损失
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            # 应用 loss_mask（忽略 padding）
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            # MoE 辅助损失
            loss += res.aux_loss
            # 梯度累积归一化
            loss = loss / args.accumulation_steps

        # 反向传播
        scaler.scale(loss).backward()

        # 梯度累积
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

        # 日志记录
        if step % args.log_interval == 0:
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) loss:{loss.item():.6f} lr:{lr:.12f}')

        # 保存检查点
        if step % args.save_interval == 0:
            torch.save(model.state_dict(), f'{args.save_dir}/pretrain_{lm_config.hidden_size}.pth')
```

### 3.2 学习率调度

MiniMind 使用余弦退火学习率调度：

```python
def get_lr(it, warmup_it, max_it, learning_rate):
    """余弦退火学习率调度"""
    # 1. Warmup 阶段：线性增加
    if it < warmup_it:
        return learning_rate * it / warmup_it
    # 2. 余弦退火阶段：逐渐降低到 min_lr
    progress = (it - warmup_it) / (max_it - warmup_it)
    min_lr = learning_rate * 0.1  # 最低学习率
    return min_lr + 0.5 * (learning_rate - min_lr) * (1 + math.cos(math.pi * progress))
```

**学习率曲线**：

```
学习率
    │
    │     ╱─────────────────────────
    │    ╱                          ╲
    │   ╱                            ╲
    │  ╱                              ╲
    │ ╱                                ╲
    │╱                                  ╲
    └─────────────────────────────────────→ 训练步数
      Warmup      Cosine Annealing
```

### 3.3 混合精度训练

```python
# GradScaler 用于 FP16 混合精度训练
scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))

# 前向传播使用 autocast
with torch.cuda.amp.autocast(dtype=torch.float16):
    loss = model(X)

# 反向传播使用 scaler
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**混合精度说明**：
- `bfloat16`：不需要 GradScaler，数值稳定
- `float16`：需要 GradScaler 防止梯度下溢

### 3.4 梯度累积

```python
# 梯度累积：模拟更大的 batch_size
loss = loss / args.accumulation_steps

# 每 accumulation_steps 步更新一次参数
if (step + 1) % args.accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

**效果**：`effective_batch_size = batch_size × accumulation_steps`

### 3.5 梯度裁剪

```python
# 防止梯度爆炸
torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
```

## 四、分布式训练（DDP）

### 4.1 初始化分布式环境

```python
def init_distributed_mode():
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        return local_rank
    return 0
```

### 4.2 DDP 包装模型

```python
if dist.is_initialized():
    model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
    model = DistributedDataParallel(model, device_ids=[local_rank])
```

**注意**：RoPE 的频率缓存不需要同步，加入 `_ddp_params_and_buffers_to_ignore`。

### 4.3 分布式采样器

```python
train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)

# 每个 epoch 设置不同的随机种子
train_sampler.set_epoch(epoch)
```

## 五、断点续训

### 5.1 保存检查点

```python
def lm_checkpoint(lm_config, weight, model, optimizer, scaler, epoch, step, wandb, save_dir):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'epoch': epoch,
        'step': step,
        'wandb_id': wandb.run.id if wandb else None,
    }
    torch.save(checkpoint, f'{save_dir}/{weight}_{lm_config.hidden_size}_resume.pth')
```

### 5.2 加载检查点

```python
if args.from_resume == 1:
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints')
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
```

### 5.3 跳过已训练的 batch

```python
if epoch == start_epoch and start_step > 0:
    batch_sampler = SkipBatchSampler(train_sampler, args.batch_size, start_step + 1)
    loader = DataLoader(train_ds, batch_sampler=batch_sampler, ...)
```

## 六、训练监控

### 6.1 使用 SwanLab

```bash
python train_pretrain.py --use_wandb
```

```python
import swanlab as wandb
wandb.init(project="MiniMind-Pretrain", name=run_name)
wandb.log({"loss": loss, "lr": lr})
```

### 6.2 训练曲线分析

正常训练的 Loss 曲线应该：
1. 初期快速下降（Warmup 阶段）
2. 中期平稳下降
3. 后期趋于收敛

如果出现：
- Loss 震荡剧烈 → 降低学习率
- Loss 不下降 → 检查数据或模型配置
- Loss 爆炸 → 检查梯度裁剪设置

## 七、动手练习

### 练习 1：启动预训练

```bash
# 下载预训练数据集
# 放到 ./dataset/pretrain_hq.jsonl

# 启动训练
cd trainer
python train_pretrain.py --epochs 1 --batch_size 8
```

### 练习 2：观察训练日志

记录训练过程中的 Loss 和学习率变化，分析训练是否正常。

### 练习 3：测试预训练模型

```bash
python eval_llm.py --weight pretrain
```

预训练模型应该能够进行文本续写，但不会进行对话。

## 八、下一阶段预告

下一阶段 [Phase 5: 监督微调 SFT](./05_Phase5_监督微调SFT.md)，我们将学习：
- SFT 的完整流程
- ChatML 对话模板
- Loss Mask 的作用
- 如何让模型学会对话
