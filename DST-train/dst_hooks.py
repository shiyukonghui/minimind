"""
DST 诊断 Hook 模块
提供矩阵基熵(MBE)监控和假死神经元检测功能
"""
import torch
import torch.nn as nn
from collections import defaultdict


class MBEMonitor:
    """矩阵基熵(MBE)监控器 — 已禁用
    MBE基于SVD奇异值分布熵，对MiniMind小模型(hidden_size=512)失效：
    512个奇异值天然接近均匀分布(Marchenko-Pastur定律)，归一化熵≈1.0永不下降。
    同时forward hooks中act.float()产生大量临时显存(~369MB)，触发RTX 4090 swap。
    饱和检测功能已由GNDMonitor替代。
    """

    def __init__(self, model, monitor_layers=None, threshold=0.3, patience=5):
        self.model = model
        self.threshold = threshold
        self.patience = patience
        self.hooks = []

    # def _register_hooks(self, monitor_layers):
    #     pass

    # def _make_hook(self, name):
    #     pass

    # def compute_mbe(self, weight_matrix):
    #     pass

    # def check_model_mbe(self):
    #     pass

    # def compute_sparsity(self):
    #     pass

    # def report(self):
    #     pass

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class DeadNeuronDetector:
    """假死神经元检测器
    检测并唤醒训练过程中出现的大量零激活神经元。
    """

    def __init__(self, model, dead_threshold=0.99):
        """
        Args:
            model: 要检测的模型
            dead_threshold: 神经元激活为零的比例阈值，超过则视为假死
        """
        self.model = model
        self.dead_threshold = dead_threshold
        self._activation_records = defaultdict(list)  # 记录每层的零激活比例

    def detect_dead_neurons(self):
        """检测模型中的假死神经元
        
        Returns:
            dict: 每层中假死神经元的统计信息
            {
                layer_name: {
                    'dead_count': int,       # 假死神经元数量
                    'total_count': int,      # 总神经元数量
                    'dead_ratio': float,     # 假死比例
                    'dead_indices': tensor   # 假死神经元的索引
                }
            }
        """
        dead_info = {}
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.dim() == 2:
                # 检测权重矩阵中全零的行（对应的输出神经元完全没有连接）
                row_norms = param.data.abs().sum(dim=1)
                dead_mask = (row_norms == 0)
                dead_count = dead_mask.sum().item()
                total_count = param.shape[0]
                dead_ratio = dead_count / total_count

                dead_info[name] = {
                    'dead_count': dead_count,
                    'total_count': total_count,
                    'dead_ratio': dead_ratio,
                    'dead_indices': dead_mask.nonzero(as_tuple=True)[0],
                }

                self._activation_records[name].append(dead_ratio)

        return dead_info

    def wake_dead_neurons(self, scale=1e-4):
        """唤醒假死神经元
        对检测到的假死神经元，用极小的随机值重新初始化对应权重行，
        保持网络未来的发展潜力。
        
        Args:
            scale: 重新初始化的随机值标准差，使用极小值避免梯度爆炸
            
        Returns:
            int: 唤醒的神经元总数
        """
        total_woken = 0
        dead_info = self.detect_dead_neurons()

        for name, param in self.model.named_parameters():
            if name in dead_info:
                info = dead_info[name]
                if info['dead_count'] > 0:
                    dead_idx = info['dead_indices']
                    # 用极小随机值重新初始化假死神经元的权重行
                    param.data[dead_idx] = torch.randn(
                        len(dead_idx), param.shape[1],
                        device=param.device, dtype=param.dtype
                    ) * scale
                    total_woken += info['dead_count']

        return total_woken

    def report(self):
        """生成假死神经元检测报告"""
        dead_info = self.detect_dead_neurons()
        total_dead = 0
        total_neurons = 0

        report_lines = ["=" * 50, "假死神经元检测报告", "=" * 50]
        report_lines.append(f"{'层名':<50} {'假死数':>8} {'总数':>8} {'比例':>8}")
        report_lines.append("-" * 50)

        for name, info in dead_info.items():
            total_dead += info['dead_count']
            total_neurons += info['total_count']
            short_name = name[-48:] if len(name) > 48 else name
            report_lines.append(
                f"{short_name:<50} {info['dead_count']:>8} {info['total_count']:>8} {info['dead_ratio']:>8.2%}"
            )

        overall_ratio = total_dead / total_neurons if total_neurons > 0 else 0.0
        report_lines.append("-" * 50)
        report_lines.append(f"总计: {total_dead}/{total_neurons} ({overall_ratio:.2%})")

        return "\n".join(report_lines)


class GNDMonitor:
    """梯度范数衰减(GND)监控器
    通过追踪梯度L2范数的衰减比例，判断模型是否达到训练饱和状态。
    梯度范数直接度量"模型还在学多少"，对所有模型尺寸均有效。
    替代MBE(矩阵基熵)在小模型(hidden_size=512)上失效的问题。
    """

    def __init__(self, model, threshold=0.1, patience=3):
        """
        Args:
            model: 要监控的模型（需传入原始模型，非DDP包装）
            threshold: 梯度范数衰减比例阈值，当前/峰值 < 此值认为饱和
            patience: 连续低于阈值的次数，超过则判定为饱和
        """
        self.model = model
        self.threshold = threshold
        self.patience = patience
        self.peak_grad_norm = None  # 历史梯度范数峰值
        self.low_count = 0  # 连续低于阈值的计数
        self.history = []  # (ratio, grad_norm) 历史记录

    def compute_grad_norm(self):
        """计算所有2D权重的全局梯度L2范数
        遍历模型中所有2维weight参数（Linear层权重），累加其梯度的L2范数平方和再开方。

        Returns:
            float: 全局梯度L2范数
        """
        total_norm_sq = 0.0
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.dim() == 2 and param.grad is not None:
                total_norm_sq += param.grad.detach().float().norm().item() ** 2
        return total_norm_sq ** 0.5

    def check_saturation(self):
        """检查梯度是否衰减至饱和水平
        首次调用时记录梯度范数峰值，后续调用计算当前范数与峰值的比值。
        当比值连续低于阈值超过patience次时，判定为饱和。

        Returns:
            bool: 是否饱和
            float: 当前梯度范数 / 峰值的比值
            float: 当前梯度范数值
        """
        grad_norm = self.compute_grad_norm()

        if self.peak_grad_norm is None or grad_norm > self.peak_grad_norm:
            self.peak_grad_norm = grad_norm

        ratio = grad_norm / (self.peak_grad_norm + 1e-10)
        self.history.append((ratio, grad_norm))

        if ratio < self.threshold:
            self.low_count += 1
        else:
            self.low_count = 0

        is_saturated = self.low_count >= self.patience
        return is_saturated, ratio, grad_norm

    def report(self):
        """生成当前GND状态报告"""
        lines = ["=" * 50, "GND (梯度范数衰减) 诊断报告", "=" * 50]
        if self.peak_grad_norm is not None:
            lines.append(f"峰值梯度范数: {self.peak_grad_norm:.4f}")
        else:
            lines.append("峰值梯度范数: 未记录")
        if self.history:
            last_ratio, last_norm = self.history[-1]
            lines.append(f"最近梯度范数: {last_norm:.4f}, 衰减比例: {last_ratio:.6f}")
        lines.append(f"饱和阈值: {self.threshold}, 连续低范数计数: {self.low_count}/{self.patience}")
        lines.append("-" * 50)
        if self.history:
            lines.append("GND历史 (最近10次):")
            for i, (r, n) in enumerate(self.history[-10:]):
                marker = " <-- 低于阈值" if r < self.threshold else ""
                lines.append(f"  [{len(self.history) - 10 + i + 1}] ratio={r:.6f}, norm={n:.4f}{marker}")
        return "\n".join(lines)
