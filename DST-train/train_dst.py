"""
DST 动态稀疏训练主脚本
实现三阶段训练循环：学习与成长(RigL) → 压缩与巩固(剪枝) → 恢复与再成长(微调+蒸馏)
从头训练，与 train_baseline.py 形成对比实验
"""
import os
import sys
import copy

# 确保可以导入项目根目录的模块和DST-train目录下的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import argparse
import time
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import PretrainDataset, SFTDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, init_distributed_mode, setup_seed, init_model
from dst_pruning import MagnitudePruner, RigLScheduler
from dst_hooks import MBEMonitor, DeadNeuronDetector

warnings.filterwarnings('ignore')


# ============================================================
# 蒸馏损失函数（复用项目已有实现）
# ============================================================
def distillation_loss(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    """KL散度蒸馏损失"""
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction=reduction)
    return (temperature ** 2) * kl


# ============================================================
# 阶段一：学习与成长 + RigL 动态稀疏训练
# ============================================================
def train_phase1(epoch, loader, iters, rigl_scheduler, mbe_monitor, start_step=0, wandb=None):
    """阶段一训练循环：标准预训练 + RigL动态稀疏训练"""
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    total_steps = args.phase1_epochs * iters

    for step, (X, Y, loss_mask) in enumerate(loader, start=start_step + 1):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # 余弦退火学习率
        lr = get_lr(epoch * iters + step, total_steps, args.phase1_lr)
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
            loss = loss / args.phase1_accumulation_steps

        scaler.scale(loss).backward()

        # 梯度累积 + 梯度裁剪
        if (step + 1) % args.phase1_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # RigL更新：剪枝-生长
            rigl_scheduler.update(epoch * iters + step, total_steps)
            # 应用稀疏掩码，确保被剪枝的连接保持为零
            rigl_scheduler.apply_masks()

            torch.cuda.empty_cache()

        # 日志记录
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.phase1_accumulation_steps
            current_lr = optimizer.param_groups[-1]['lr']
            sparsity = rigl_scheduler.get_sparsity()
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f'[DST-Phase1] Epoch:[{epoch+1}/{args.phase1_epochs}]({step}/{iters}) '
                   f'loss:{current_loss:.6f} lr:{current_lr:.12f} sparsity:{sparsity:.2%} eta:{eta_min}min')
            if wandb:
                wandb.log({"phase1_loss": current_loss, "lr": current_lr, "sparsity": sparsity})

        # MBE监控
        if step % args.monitor_interval == 0 and step > 0:
            _, avg_mbe, is_saturated = mbe_monitor.check_model_mbe()
            Logger(f'[DST-Monitor] Step {step}: MBE={avg_mbe:.4f}, 饱和={is_saturated}')

        # 定期保存
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            _save_model(f'dst_phase1_{lm_config.hidden_size}')


# ============================================================
# 阶段二：压缩与巩固（剪枝）
# ============================================================
def run_phase2():
    """阶段二：对阶段一产出的模型进行Magnitude Pruning"""
    Logger('\n' + '=' * 60)
    Logger('阶段二：压缩与巩固 — Magnitude Pruning')
    Logger('=' * 60)

    # 保存剪枝前的模型作为教师（用于阶段三蒸馏）
    teacher_state = copy.deepcopy(model.state_dict())
    moe_suffix = '_moe' if lm_config.use_moe else ''
    teacher_path = f'{args.save_dir}/dst_phase1_{lm_config.hidden_size}{moe_suffix}_teacher.pth'
    torch.save({k: v.half() for k, v in teacher_state.items()}, teacher_path)
    Logger(f'教师模型已保存至: {teacher_path}')

    # 执行剪枝
    pruner = MagnitudePruner(model, max_sparsity=0.5)
    masks = pruner.prune_and_report(sparsity=args.sparsity)

    # 保存剪枝掩码
    mask_path = f'{args.save_dir}/pruning_masks_sp{int(args.sparsity * 100)}.pt'
    torch.save(masks, mask_path)
    Logger(f'剪枝掩码已保存至: {mask_path}')

    # 保存剪枝后的模型
    _save_model(f'dst_pruned_{lm_config.hidden_size}_sp{int(args.sparsity * 100)}')

    return teacher_state, masks


# ============================================================
# 阶段三：恢复与再成长（微调 + 蒸馏 + 假死唤醒）
# ============================================================
def train_phase3(epoch, loader, iters, teacher_model, dead_detector, start_step=0, wandb=None):
    """阶段三训练循环：低学习率微调恢复 + 可选知识蒸馏"""
    start_time = time.time()

    # 每个epoch开始时唤醒假死神经元
    woken = dead_detector.wake_dead_neurons(scale=1e-4)
    if woken > 0:
        Logger(f'[DST-Phase3] Epoch {epoch+1}: 唤醒了 {woken} 个假死神经元')

    if teacher_model is not None:
        teacher_model.eval()
        teacher_model.requires_grad_(False)

    loss_fct = nn.CrossEntropyLoss(reduction='none')

    for step, (X, Y, loss_mask) in enumerate(loader, start=start_step + 1):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # 余弦退火学习率（阶段三使用更低的学习率）
        lr = get_lr(epoch * iters + step, args.phase3_epochs * iters, args.phase3_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(X)
            student_logits = res.logits

        # 教师模型前向传播
        if teacher_model is not None:
            with torch.no_grad():
                teacher_logits = teacher_model(X).logits
                vocab_size_student = student_logits.size(-1)
                teacher_logits = teacher_logits[..., :vocab_size_student]

        # 1) CE损失
        loss_mask_flat = loss_mask.view(-1)
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            Y.view(-1),
            ignore_index=0,
            reduction='none'
        )
        ce_loss = torch.sum(ce_loss * loss_mask_flat) / loss_mask_flat.sum()
        if lm_config.use_moe:
            ce_loss += res.aux_loss

        # 2) 蒸馏损失
        if teacher_model is not None:
            distill_loss = distillation_loss(
                student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],
                teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat == 1],
                temperature=args.temperature
            )
        else:
            distill_loss = torch.tensor(0.0, device=args.device)

        # 3) 总损失 = alpha * CE + (1-alpha) * Distill
        loss = (args.alpha * ce_loss + (1 - args.alpha) * distill_loss) / args.phase3_accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.phase3_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

        # 日志记录
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.phase3_accumulation_steps
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f'[DST-Phase3] Epoch:[{epoch+1}/{args.phase3_epochs}]({step}/{iters}) '
                   f'loss:{current_loss:.6f} ce:{ce_loss.item():.4f} distill:{distill_loss.item():.4f} lr:{current_lr:.12f} eta:{eta_min}min')
            if wandb:
                wandb.log({
                    "phase3_loss": current_loss,
                    "ce_loss": ce_loss.item(),
                    "distill_loss": distill_loss.item() if teacher_model is not None else 0.0,
                    "lr": current_lr
                })

        # 定期保存
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            _save_model(f'dst_recovered_{lm_config.hidden_size}_sp{int(args.sparsity * 100)}')


# ============================================================
# 辅助函数
# ============================================================
def _save_model(name_prefix):
    """保存模型权重"""
    model.eval()
    moe_suffix = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/{name_prefix}{moe_suffix}.pth'
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    state_dict = {k: v.half() for k, v in state_dict.items()}
    torch.save(state_dict, ckp)
    model.train()


def _get_raw_model():
    """获取原始模型（去除DDP包装）"""
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind DST Training (Dynamic Sparse Training)")

    # 通用参数
    parser.add_argument("--save_dir", type=str, default="./model", help="模型保存目录")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")

    # 阶段一参数（与基线一致）
    parser.add_argument("--phase1_data_path", type=str, default="dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
    parser.add_argument("--phase1_epochs", type=int, default=1, help="预训练轮数(与基线相同)")
    parser.add_argument("--phase1_batch_size", type=int, default=32, help="批大小")
    parser.add_argument("--phase1_lr", type=float, default=5e-4, help="学习率")
    parser.add_argument("--phase1_accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument('--max_seq_len', default=512, type=int, help="训练的最大截断长度")

    # 阶段二参数
    parser.add_argument("--sparsity", type=float, default=0.2, help="额外剪枝比例(0.2=20%%)")

    # 阶段三参数
    parser.add_argument("--phase3_data_path", type=str, default="dataset/sft_t2t_mini.jsonl", help="恢复训练数据路径")
    parser.add_argument("--phase3_epochs", type=int, default=3, help="恢复训练轮数")
    parser.add_argument("--phase3_batch_size", type=int, default=16, help="批大小")
    parser.add_argument("--phase3_lr", type=float, default=5e-7, help="学习率(极低)")
    parser.add_argument("--phase3_accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--use_distillation", type=int, default=1, choices=[0, 1], help="是否使用知识蒸馏")
    parser.add_argument("--alpha", type=float, default=0.5, help="蒸馏CE损失权重")
    parser.add_argument("--temperature", type=float, default=1.5, help="蒸馏温度")

    # DST专用参数
    parser.add_argument("--initial_sparsity", type=float, default=0.5, help="RigL初始稀疏度(0.5=保留50%%连接)")
    parser.add_argument("--update_frequency", type=int, default=100, help="RigL更新频率(步)")
    parser.add_argument("--T_end", type=float, default=0.8, help="RigL停止调整比例")
    parser.add_argument("--prune_fraction", type=float, default=0.2, help="RigL每次调整的连接比例")
    parser.add_argument("--monitor_interval", type=int, default=100, help="MBE监控间隔(步)")
    parser.add_argument("--mbe_threshold", type=float, default=0.3, help="MBE饱和阈值")

    # wandb
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-DST", help="wandb项目名")

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
        wandb_run_name = f"DST-H{args.hidden_size}-SP{args.initial_sparsity}-Epoch-{args.phase1_epochs}"
        wandb.init(project=args.wandb_project, name=wandb_run_name)

    # ========== 5. 定义模型、数据、优化器 ==========
    # from_weight='none' — 从头训练
    model, tokenizer = init_model(lm_config, 'none', device=args.device)
    Logger(f'[DST] 模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M')

    # ========== 6. DDP包模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ================================================================
    # 阶段一：学习与成长 + RigL
    # ================================================================
    Logger('\n' + '=' * 60)
    Logger('阶段一：学习与成长 + RigL 动态稀疏训练')
    Logger(f'初始稀疏度: {args.initial_sparsity:.0%}, 更新频率: {args.update_frequency}步')
    Logger('=' * 60)

    # 初始化RigL调度器和MBE监控
    rigl_scheduler = RigLScheduler(
        _get_raw_model(),
        initial_sparsity=args.initial_sparsity,
        update_frequency=args.update_frequency,
        T_end=args.T_end,
        prune_fraction=args.prune_fraction
    )
    rigl_scheduler.initialize_masks()

    mbe_monitor = MBEMonitor(_get_raw_model(), threshold=args.mbe_threshold)

    # 阶段一训练
    train_ds = PretrainDataset(args.phase1_data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.phase1_lr)

    for epoch in range(args.phase1_epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        loader = DataLoader(
            train_ds, batch_size=args.phase1_batch_size, shuffle=(train_sampler is None),
            sampler=train_sampler, num_workers=args.num_workers, pin_memory=True
        )
        train_phase1(epoch, loader, len(loader), rigl_scheduler, mbe_monitor, 0, wandb)

    # 保存RigL掩码
    rigl_mask_path = f'{args.save_dir}/rigl_masks_sp{int(args.initial_sparsity * 100)}.pt'
    rigl_scheduler.save_masks(rigl_mask_path)

    # 打印MBE诊断报告
    Logger(mbe_monitor.report())
    mbe_monitor.remove_hooks()

    _save_model(f'dst_phase1_{lm_config.hidden_size}')
    Logger('[DST] 阶段一训练完成')

    # ================================================================
    # 阶段二：压缩与巩固（剪枝）
    # ================================================================
    teacher_state, pruning_masks = run_phase2()

    # ================================================================
    # 阶段三：恢复与再成长
    # ================================================================
    Logger('\n' + '=' * 60)
    Logger('阶段三：恢复与再成长 — 微调 + 蒸馏 + 假死唤醒')
    Logger(f'学习率: {args.phase3_lr}, 轮数: {args.phase3_epochs}, 蒸馏: {bool(args.use_distillation)}')
    Logger('=' * 60)

    # 加载教师模型（剪枝前的模型）
    teacher_model = None
    if args.use_distillation:
        teacher_model = MiniMindForCausalLM(lm_config).to(args.device)
        teacher_model.load_state_dict(teacher_state, strict=False)
        teacher_model.eval()
        teacher_model.requires_grad_(False)
        Logger(f'[DST] 教师模型已加载(剪枝前模型)')

    # 假死神经元检测器
    dead_detector = DeadNeuronDetector(_get_raw_model())

    # 阶段三训练数据（SFT数据集）
    sft_ds = SFTDataset(args.phase3_data_path, tokenizer, max_length=args.max_seq_len)
    sft_sampler = DistributedSampler(sft_ds) if dist.is_initialized() else None

    # 重新初始化优化器（更低的学习率）
    optimizer = optim.AdamW(model.parameters(), lr=args.phase3_lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))

    for epoch in range(args.phase3_epochs):
        sft_sampler and sft_sampler.set_epoch(epoch)
        loader = DataLoader(
            sft_ds, batch_size=args.phase3_batch_size, shuffle=(sft_sampler is None),
            sampler=sft_sampler, num_workers=args.num_workers, pin_memory=True
        )
        train_phase3(epoch, loader, len(loader), teacher_model, dead_detector, 0, wandb)

    # 保存最终模型
    _save_model(f'dst_recovered_{lm_config.hidden_size}_sp{int(args.sparsity * 100)}')
    Logger('[DST] 阶段三训练完成')

    # 打印假死神经元报告
    Logger(dead_detector.report())

    # 打印最终模型稀疏度
    _, overall_sp = MagnitudePruner.compute_sparsity(_get_raw_model())
    Logger(f'\n[DST] 最终模型稀疏度: {overall_sp:.2%}')
    Logger('[DST] 全部训练流程完成！')
