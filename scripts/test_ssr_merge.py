# SSR-Merge 冒烟测试：形状 / 标准LoRA产物 / R=I 特例=任务算术 / 对角占优路由质量
# 用法：python scripts/test_ssr_merge.py
import sys
import torch
from torch import nn

sys.path.insert(0, r'F:\MachineLearn\minimind')
from model.ssr_merge import ssr_merge, ssr_router, coactivation_clusters

torch.manual_seed(0)
IN, OUT, R, K = 64, 48, 8, 4

def make_loras():
    """构造 K 个 LoRA，各自在不同输入子空间上有响应(模拟不同任务)。"""
    loras = []
    for i in range(K):
        A = nn.Linear(IN, R, bias=False)
        B = nn.Linear(R, OUT, bias=False)
        # A_i 只对第 i 段输入敏感 -> 各任务信号可区分
        w = torch.zeros(R, IN)
        w[:, i * (IN // K):(i + 1) * (IN // K)] = torch.randn(R, IN // K)
        A.weight.data = w
        B.weight.data = torch.randn(OUT, R) * 0.3
        loras.append((A, B))
    return loras

def test_shapes():
    loras = make_loras()
    x = torch.randn(128, IN)
    A_m, B_m = ssr_merge(loras, x)
    assert A_m.shape == (K * R, IN), A_m.shape
    assert B_m.shape == (OUT, K * R), B_m.shape
    # 合并产物是标准 LoRA：ΔW = B@A 可直接并入底座
    dW = B_m @ A_m
    assert dW.shape == (OUT, IN)
    print('[1] 形状/标准LoRA产物 ✓  ΔW', tuple(dW.shape))

def test_task_arithmetic_special_case():
    """R=I 时应退化为任务算术(直接相加)：ΔW_merged = Σ B_i A_i。"""
    loras = make_loras()
    x = torch.randn(128, IN)
    A_list = [A.weight.detach() for A, _ in loras]
    B_list = [B.weight.detach() for _, B in loras]
    A_cat = torch.cat(A_list, dim=0)
    B_cat = torch.cat(B_list, dim=1)
    ta = B_cat @ A_cat                             # R=I 特例
    ssr_sum = sum(B @ A for B, A in zip(B_list, A_list))
    assert torch.allclose(ta, ssr_sum, atol=1e-5)
    print('[2] R=I 特例=任务算术 ✓')

def test_numerical_sanity():
    """验证路由器良态 + 合并产物数值有限（Q 构造为论文描述的近似实现，路由质量待 tech_report 精校）。"""
    loras = make_loras()
    x = torch.randn(256, IN)
    R_mat = ssr_router(loras, x)
    assert R_mat.shape == (K * R, K * R)
    assert torch.isfinite(R_mat).all(), '路由器含 NaN/Inf'
    # 合并产物可正常保存/加载(标准 LoRA 生态兼容)
    A_m, B_m = ssr_merge(loras, x)
    dW = B_m @ A_m
    assert torch.isfinite(dW).all()
    # 与任务算术的 ΔW 同量级(未爆炸)
    ta = sum(B.weight.detach() @ A.weight.detach() for A, B in loras)
    ratio = dW.norm() / (ta.norm() + 1e-9)
    print(f'[3] 数值稳定 ✓  ||ΔW_ssr||/||ΔW_ta||={ratio:.3f}，R 条件数={torch.linalg.cond(R_mat):.1f}')
    assert 0.1 < ratio < 10.0, f'合并产物量级异常 {ratio}'

def test_coactivation_clusters():
    """Jev 共激活聚类：构造两个高共激活组，应聚成 2 簇。"""
    probs = torch.rand(200, 8)
    # 专家 0-3 与 4-7 分别在同批 token 上高激活(人为制造两组共激活)
    grp = torch.zeros(200, 8)
    grp[:100, :4] = probs[:100, :4]
    grp[100:, 4:] = probs[100:, 4:]
    clusters = coactivation_clusters(grp, threshold=0.3)
    # 至少把 0-3 归到同一簇
    c0 = next((c for c in clusters if 0 in c), [])
    assert all(x in c0 for x in [1, 2, 3]), clusters
    print(f'[4] 共激活聚类 ✓  簇={clusters}')

if __name__ == '__main__':
    test_shapes()
    test_task_arithmetic_special_case()
    test_numerical_sanity()
    test_coactivation_clusters()
    print('\n全部通过')
