"""
基线模型训练脚本
标准预训练流程，从头训练（不加载预训练权重），作为 DST 对比实验的对照组
"""
import os
import sys

# 确保可以导入项目根目录的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """训练一个epoch"""
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()

    for step, (X, Y, loss_mask) in enumerate(loader, start=start_step + 1):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # 余弦退火学习率
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 混合精度前向传播
        with autocast_ctx:
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        # 梯度累积 + 梯度裁剪
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

        # 日志记录
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f'[Baseline] Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} eta:{eta_min}min')
            if wandb:
                wandb.log({"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min})

        # 定期保存
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/baseline_{lm_config.hidden_size}{moe_suffix}.pth'
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            state_dict = {k: v.half() for k, v in state_dict.items()}
            torch.save(state_dict, ckp)
            model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Baseline Pretraining (for DST comparison)")
    parser.add_argument("--save_dir", type=str, default="./model", help="模型保存目录")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=0, help="数据加载线程数(Windows建议0)")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=512, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE")
    parser.add_argument("--data_path", type=str, default="dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Baseline", help="wandb项目名")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录和模型参数 ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 配置wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_run_name = f"Baseline-H{args.hidden_size}-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name)

    # ========== 5. 定义模型、数据、优化器 ==========
    # from_weight='none' — 从头训练，不加载预训练权重
    model, tokenizer = init_model(lm_config, 'none', device=args.device)
    Logger(f'[Baseline] 模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M')

    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6. DDP包模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 7. 开始训练 ==========
    Logger(f'[Baseline] 开始从头训练 — epochs={args.epochs}, lr={args.learning_rate}, bs={args.batch_size}')
    for epoch in range(args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
            sampler=train_sampler, num_workers=args.num_workers, pin_memory=True
        )
        train_epoch(epoch, loader, len(loader), 0, wandb)

    # ========== 8. 保存最终模型 ==========
    if is_main_process():
        model.eval()
        moe_suffix = '_moe' if lm_config.use_moe else ''
        ckp = f'{args.save_dir}/baseline_{lm_config.hidden_size}{moe_suffix}.pth'
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        state_dict = {k: v.half() for k, v in state_dict.items()}
        torch.save(state_dict, ckp)
        Logger(f'[Baseline] 训练完成，模型已保存至: {ckp}')
