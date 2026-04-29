"""
DST 诊断 Hook 模块
提供矩阵基熵(MBE)监控和假死神经元检测功能
"""
import torch
import torch.nn as nn
from collections import defaultdict


class MBEMonitor:
    """矩阵基熵(MBE)监控器
    通过对模型权重矩阵做SVD分解计算归一化熵，
    判断模型是否达到"学习饱和"状态。
    """

    def __init__(self, model, monitor_layers=None, threshold=0.3, patience=5):
        """
        Args:
            model: 要监控的模型
            monitor_layers: 指定监控的层名称列表，None则监控所有Linear层
            threshold: MBE饱和阈值，低于此值认为模型"学累了"
            patience: 连续低于阈值的步数，超过则判定为饱和
        """
        self.model = model
        self.threshold = threshold
        self.patience = patience
        self.hooks = []
        self.mbe_history = defaultdict(list)  # 每层的MBE历史
        self.low_mbe_count = 0  # 连续低于阈值的计数
        self.activation_stats = {}  # 激活值统计

        # 注册forward hook来收集激活值统计
        self._register_hooks(monitor_layers)

    def _register_hooks(self, monitor_layers):
        """在指定层注册forward hook"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if monitor_layers is None or any(layer in name for layer in monitor_layers):
                    hook = module.register_forward_hook(self._make_hook(name))
                    self.hooks.append(hook)

    def _make_hook(self, name):
        """为指定层创建hook回调"""
        def hook_fn(module, input, output):
            # 记录激活值的统计信息（不计算梯度）
            with torch.no_grad():
                act = output.detach()
                self.activation_stats[name] = {
                    'mean': act.float().mean().item(),
                    'std': act.float().std().item(),
                    'zero_ratio': (act == 0).float().mean().item(),
                    'shape': tuple(act.shape),
                }
        return hook_fn

    def compute_mbe(self, weight_matrix):
        """计算单个权重矩阵的矩阵基熵(MBE)
        
        Args:
            weight_matrix: 2D权重张量 [out_features, in_features]
        Returns:
            归一化熵值，范围[0, 1]，越高表示信息分布越均匀
        """
        # 转为float32确保数值稳定
        W = weight_matrix.float().detach()
        if W.numel() == 0:
            return 1.0
        try:
            # SVD分解，取奇异值
            S = torch.linalg.svdvals(W)
            # 归一化奇异值作为概率分布
            S_norm = S / (S.sum() + 1e-10)
            # 计算熵
            entropy = -(S_norm * torch.log(S_norm + 1e-10)).sum()
            # 最大熵（均匀分布时的熵）
            max_entropy = torch.log(torch.tensor(S.shape[0], dtype=torch.float32, device=S.device))
            if max_entropy == 0:
                return 1.0
            return (entropy / max_entropy).item()
        except Exception:
            # SVD分解失败时返回1.0（假设未饱和）
            return 1.0

    def check_model_mbe(self):
        """检查模型所有线性层的MBE
        
        Returns:
            dict: 每层的MBE值
            float: 平均MBE
            bool: 是否饱和
        """
        layer_mbe = {}
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.dim() == 2:
                mbe = self.compute_mbe(param.data)
                layer_mbe[name] = mbe
                self.mbe_history[name].append(mbe)

        if not layer_mbe:
            return {}, 1.0, False

        avg_mbe = sum(layer_mbe.values()) / len(layer_mbe)

        # 判断是否饱和
        if avg_mbe < self.threshold:
            self.low_mbe_count += 1
        else:
            self.low_mbe_count = 0

        is_saturated = self.low_mbe_count >= self.patience
        return layer_mbe, avg_mbe, is_saturated

    def compute_sparsity(self):
        """计算模型当前的权重稀疏度（零权重占比）
        
        Returns:
            dict: 每层的稀疏度
            float: 整体稀疏度
        """
        layer_sparsity = {}
        total_zeros = 0
        total_params = 0
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.dim() == 2:
                n_zeros = (param.data == 0).sum().item()
                n_total = param.numel()
                layer_sparsity[name] = n_zeros / n_total
                total_zeros += n_zeros
                total_params += n_total

        overall = total_zeros / total_params if total_params > 0 else 0.0
        return layer_sparsity, overall

    def report(self):
        """生成当前模型状态报告"""
        layer_mbe, avg_mbe, is_saturated = self.check_model_mbe()
        layer_sparsity, overall_sparsity = self.compute_sparsity()

        report_lines = ["=" * 50, "DST 模型诊断报告", "=" * 50]
        report_lines.append(f"平均MBE: {avg_mbe:.4f} (阈值: {self.threshold}, 饱和: {is_saturated})")
        report_lines.append(f"整体稀疏度: {overall_sparsity:.2%}")
        report_lines.append(f"连续低MBE计数: {self.low_mbe_count}/{self.patience}")
        report_lines.append("-" * 50)
        report_lines.append(f"{'层名':<50} {'MBE':>8} {'稀疏度':>8}")
        report_lines.append("-" * 50)
        for name in layer_mbe:
            mbe = layer_mbe[name]
            sp = layer_sparsity.get(name, 0.0)
            short_name = name[-48:] if len(name) > 48 else name
            report_lines.append(f"{short_name:<50} {mbe:>8.4f} {sp:>8.2%}")

        return "\n".join(report_lines)

    def remove_hooks(self):
        """移除所有注册的hook"""
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
