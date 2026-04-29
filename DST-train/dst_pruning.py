"""
DST 剪枝器模块
提供基于绝对值大小的剪枝(Magnitude Pruning)和 RigL 动态稀疏训练功能
"""
import torch
import torch.nn as nn


class MagnitudePruner:
    """基于权重绝对值大小的剪枝器
    遍历模型的所有线性层，按指定的稀疏度剪掉绝对值最小的权重。
    """

    def __init__(self, model, max_sparsity=0.5):
        """
        Args:
            model: 要剪枝的模型
            max_sparsity: 最大允许剪枝比例，超过此值会打印警告
        """
        self.model = model
        self.max_sparsity = max_sparsity

    def compute_mask(self, sparsity=0.2):
        """计算剪枝掩码
        
        对每个线性层权重，将绝对值最小的 sparsity 比例的权重置零。
        
        Args:
            sparsity: 剪枝比例，0.2表示剪掉20%的权重
            
        Returns:
            dict: {参数名: 掩码张量}，掩码中1表示保留，0表示剪枝
        """
        if sparsity > self.max_sparsity:
            print(f'警告: 剪枝比例 {sparsity:.0%} 超过最大限制 {self.max_sparsity:.0%}，可能导致性能严重下降')

        masks = {}
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.dim() == 2:
                # 计算权重绝对值的分位数作为阈值
                flat_abs = param.data.abs().flatten()
                if flat_abs.numel() == 0:
                    masks[name] = torch.ones_like(param.data)
                    continue
                threshold = torch.quantile(flat_abs, sparsity)
                masks[name] = (param.data.abs() > threshold).float()

        return masks

    @staticmethod
    def apply_mask(model, masks):
        """将掩码应用到模型权重，执行实际剪枝
        
        Args:
            model: 模型
            masks: compute_mask返回的掩码字典
        """
        for name, param in model.named_parameters():
            if name in masks:
                param.data *= masks[name]

    @staticmethod
    def compute_sparsity(model):
        """计算模型各层和整体的稀疏度
        
        Returns:
            dict: 各层稀疏度
            float: 整体稀疏度
        """
        layer_sparsity = {}
        total_zeros = 0
        total_params = 0

        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() == 2:
                n_zeros = (param.data == 0).sum().item()
                n_total = param.numel()
                layer_sparsity[name] = n_zeros / n_total if n_total > 0 else 0.0
                total_zeros += n_zeros
                total_params += n_total

        overall = total_zeros / total_params if total_params > 0 else 0.0
        return layer_sparsity, overall

    def prune_and_report(self, sparsity=0.2):
        """执行剪枝并打印报告
        
        Args:
            sparsity: 剪枝比例
            
        Returns:
            dict: 剪枝掩码
        """
        print(f'\n{"=" * 60}')
        print(f'执行剪枝 — 目标稀疏度: {sparsity:.0%}')
        print(f'{"=" * 60}')

        # 剪枝前的稀疏度
        before_sp, before_overall = self.compute_sparsity(self.model)
        print(f'剪枝前整体稀疏度: {before_overall:.2%}')

        # 计算并应用掩码
        masks = self.compute_mask(sparsity)
        self.apply_mask(self.model, masks)

        # 剪枝后的稀疏度
        after_sp, after_overall = self.compute_sparsity(self.model)
        print(f'剪枝后整体稀疏度: {after_overall:.2%}')

        # 各层详细报告
        print(f'\n{"层名":<55} {"剪枝前":>8} {"剪枝后":>8}')
        print('-' * 75)
        for name in after_sp:
            short_name = name[-53:] if len(name) > 53 else name
            print(f'{short_name:<55} {before_sp.get(name, 0):>8.2%} {after_sp[name]:>8.2%}')

        return masks


class RigLScheduler:
    """RigL 动态稀疏训练调度器
    
    在训练过程中周期性地"剪枝最不重要的连接，同时生长新的连接"，
    实现边训练边重组网络结构。
    
    参考: "Rigging the Lottery: Making All Tickets Winners" (Evci et al., 2020)
    """

    def __init__(self, model, initial_sparsity=0.5, update_frequency=100, T_end=0.8, prune_fraction=0.2):
        """
        Args:
            model: 训练的模型
            initial_sparsity: 初始稀疏度（训练开始时随机剪枝保留的比例）
                              0.5表示初始保留50%的连接
            update_frequency: 每N步执行一次剪枝-生长更新
            T_end: 在总训练步数的T_end比例后停止动态调整（固定掩码）
            prune_fraction: 每次更新时调整的连接占活跃连接的比例
        """
        self.model = model
        self.initial_sparsity = initial_sparsity
        self.update_frequency = update_frequency
        self.T_end = T_end
        self.prune_fraction = prune_fraction
        self.masks = {}  # 当前稀疏掩码
        self._initialized = False

    def initialize_masks(self):
        """初始化稀疏掩码
        按initial_sparsity随机选择保留的连接
        """
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.dim() == 2:
                n_total = param.numel()
                n_keep = int(n_total * (1 - self.initial_sparsity))
                # 随机选择保留的连接
                flat_mask = torch.zeros(n_total, device=param.device)
                indices = torch.randperm(n_total, device=param.device)[:n_keep]
                flat_mask[indices] = 1.0
                self.masks[name] = flat_mask.view(param.shape)
                # 应用初始掩码
                param.data *= self.masks[name]

        self._initialized = True
        sparsity = 1.0 - sum(m.sum().item() for m in self.masks.values()) / sum(m.numel() for m in self.masks.values())
        print(f'RigL: 初始化稀疏掩码完成，实际稀疏度: {sparsity:.2%}')

    def update(self, step, total_steps):
        """执行RigL剪枝-生长更新
        
        在活跃连接中剪掉权重绝对值最小的，在非活跃连接中生长梯度绝对值最大的。
        
        Args:
            step: 当前训练步数
            total_steps: 总训练步数
        """
        if not self._initialized:
            self.initialize_masks()

        # 超过T_end后固定掩码，不再调整
        if step / total_steps > self.T_end:
            return

        # 按频率更新
        if step % self.update_frequency != 0 or step == 0:
            return

        total_pruned = 0
        total_grown = 0

        for name, param in self.model.named_parameters():
            if name not in self.masks or param.grad is None:
                continue

            mask = self.masks[name]
            n_active = mask.sum().int().item()
            if n_active == 0:
                continue

            # 计算需要调整的连接数量
            n_adjust = max(1, int(n_active * self.prune_fraction))

            # === 剪枝：在活跃连接中移除权重绝对值最小的 ===
            active_weights = param.data.abs() * mask
            flat_active = active_weights.flatten()
            # 只考虑活跃连接
            active_indices = mask.flatten().nonzero(as_tuple=True)[0]
            if len(active_indices) <= n_adjust:
                continue
            # 找活跃连接中权重绝对值最小的
            active_weight_vals = flat_active[active_indices]
            _, prune_order = active_weight_vals.sort()
            prune_flat_indices = active_indices[prune_order[:n_adjust]]

            # === 生长：在非活跃连接中选择梯度绝对值最大的 ===
            grad_abs = param.grad.abs() * (1 - mask)
            flat_grad = grad_abs.flatten()
            inactive_indices = (1 - mask).flatten().nonzero(as_tuple=True)[0]
            if len(inactive_indices) < n_adjust:
                continue
            inactive_grad_vals = flat_grad[inactive_indices]
            _, grow_order = inactive_grad_vals.sort(descending=True)
            grow_flat_indices = inactive_indices[grow_order[:n_adjust]]

            # === 执行剪枝和生长 ===
            flat_mask = mask.flatten()
            flat_mask[prune_flat_indices] = 0.0
            flat_mask[grow_flat_indices] = 1.0
            self.masks[name] = flat_mask.view(mask.shape)

            # 新生长的连接权重初始化为0（将在后续训练中更新）
            param.data.flatten()[prune_flat_indices] = 0.0
            # 生长的连接保持0值，通过后续训练学习

            total_pruned += n_adjust
            total_grown += n_adjust

        if total_pruned > 0:
            print(f'RigL Step {step}: 剪枝 {total_pruned} 连接, 生长 {total_grown} 连接')

    def apply_masks(self):
        """将当前掩码应用到模型权重
        每个训练步后调用，确保被剪枝的连接权重保持为零
        """
        if not self._initialized:
            return
        for name, param in self.model.named_parameters():
            if name in self.masks:
                param.data *= self.masks[name]

    def get_sparsity(self):
        """获取当前掩码的实际稀疏度
        
        Returns:
            float: 整体稀疏度
        """
        if not self.masks:
            return 0.0
        total = sum(m.numel() for m in self.masks.values())
        zeros = total - sum(m.sum().item() for m in self.masks.values())
        return zeros / total if total > 0 else 0.0

    def report(self):
        """生成RigL状态报告"""
        if not self._initialized:
            return "RigL: 尚未初始化掩码"

        lines = ["=" * 50, "RigL 状态报告", "=" * 50]
        lines.append(f"整体稀疏度: {self.get_sparsity():.2%}")
        lines.append(f"更新频率: 每 {self.update_frequency} 步")
        lines.append(f"T_end: {self.T_end}")
        lines.append(f"每次调整比例: {self.prune_fraction:.0%}")
        lines.append("-" * 50)

        for name, mask in self.masks.items():
            sparsity = 1.0 - mask.sum().item() / mask.numel()
            short_name = name[-48:] if len(name) > 48 else name
            lines.append(f"{short_name:<50} 稀疏度: {sparsity:.2%}")

        return "\n".join(lines)

    def save_masks(self, path):
        """保存当前掩码到文件"""
        torch.save(self.masks, path)
        print(f'RigL掩码已保存至: {path}')

    def load_masks(self, path):
        """从文件加载掩码"""
        self.masks = torch.load(path, map_location='cpu', weights_only=False)
        self._initialized = True
        print(f'RigL掩码已从 {path} 加载，稀疏度: {self.get_sparsity():.2%}')
