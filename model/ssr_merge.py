# SSR-Merge：Subspace Signal Routing 免训练 LoRA 合并（设计文档 §3.8，路线图任务⑦）
#
# 参考：arXiv 2606.10617，本地 ssr-lora.md。核心三步(全部闭式、免训练)：
#   1) 合并通道：K 个 LoRA 的下投影沿秩拼接 A_cat=[A_1;...;A_K](Kr×in)，
#      上投影横向拼接 B_cat=[B_1|...|B_K](out×Kr)，得到加宽通道(串扰的物理基础)；
#   2) 解混：单样本校准前向，统计通道内信号相关矩阵 G(Kr×Kr)，乘 G⁻¹ 拆回独立成分；
#   3) 导航：方向引导矩阵 Q 把第 i 路信号对准第 i 个 LoRA 出口(由校准二阶统计量算出)。
#   路由器 R=Q·G⁻¹ 是 OLS 投影的解析解(最优，非启发式)。
#   部署：把 R 吸收进 B_cat -> 合并产物是标准 LoRA(ΔW=B_merged·A_cat)，推理零额外开销。
#
# 与本项目闭环：聚类依据用 Jev route 头的共激活矩阵(§3.8)，仅合并同簇专家。
# 注意：Q 的具体构造按 ssr-lora.md 的文字描述实现(近似)；论文精确公式需 tech_report 核对后精校。
import torch
from torch import nn
from typing import List, Tuple


def collect_channel_signals(loras: List[Tuple[nn.Linear, nn.Linear]], x: torch.Tensor) -> torch.Tensor:
    """对校准输入 x:[N,in]，返回合并通道内信号 S:[N, K*r]。

    第 i 个 LoRA 的通道信号 = A_i(x) 展平拼接。A_i 是 LoRA 下投影(in->r)。
    """
    feats = []
    for A, _ in loras:
        with torch.no_grad():
            feats.append(A(x))                     # [N, r]
    return torch.cat(feats, dim=-1)                # [N, K*r]


def ssr_router(loras: List[Tuple[nn.Linear, nn.Linear]], x: torch.Tensor,
               ridge: float = 1e-4) -> torch.Tensor:
    """由单样本校准统计量闭式求解路由器 R=Q·G⁻¹:[K*r, K*r]。

    G = SᵀS/N 通道内信号相关矩阵(解混用)；
    Q[i,j] = 期望的第 i 路对第 j 出口的导航权重，取 S 分段与"目标 LoRA 自身信号"的互相关，
    即把每个 LoRA 的原始通道信号作为该出口的对齐目标(OLS 目标)。
    """
    S = collect_channel_signals(loras, x)          # [N, Kr]
    N, Kr = S.shape
    G = (S.T @ S) / N + ridge * torch.eye(Kr, device=S.device, dtype=S.dtype)  # 相关矩阵+岭
    # 目标：第 i 个出口希望重建第 i 个 LoRA 单独前向的贡献。用各 LoRA 的 A_i 输出段作目标。
    # 构造 Q:[Kr, Kr]，行块 i 对应出口 i，值 = <S, s_i>/N（s_i 为第 i 段通道信号）。
    r = Kr // len(loras)
    Q = torch.zeros(Kr, Kr, device=S.device, dtype=S.dtype)
    for i in range(len(loras)):
        s_i = S[:, i * r:(i + 1) * r]              # [N, r]
        # 出口 i 的 r 行，每行 = Sᵀ s_i 列 / N
        Q[i * r:(i + 1) * r, :] = (S.T @ s_i).T / N
    R = Q @ torch.linalg.inv(G)                    # [Kr, Kr]
    return R


def ssr_merge(loras: List[Tuple[nn.Linear, nn.Linear]], x: torch.Tensor,
              ridge: float = 1e-4) -> Tuple[nn.Parameter, nn.Parameter]:
    """返回合并后的标准 LoRA 参数 (A_merged, B_merged)：
        A_merged = concat A_i (Kr×in)，B_merged = B_cat·R (out×Kr)。
    合并产物 ΔW = B_merged @ A_merged 可直接并入底座，推理零额外开销。
    """
    A_list = [A.weight.detach() for A, _ in loras]   # each [r, in]
    B_list = [B.weight.detach() for _, B in loras]   # each [out, r]
    A_cat = torch.cat(A_list, dim=0)                 # [Kr, in]
    B_cat = torch.cat(B_list, dim=1)                 # [out, Kr]
    R = ssr_router(loras, x, ridge)                  # [Kr, Kr]
    B_merged = B_cat @ R                             # [out, Kr]
    return nn.Parameter(A_cat), nn.Parameter(B_merged)


def coactivation_clusters(route_probs: torch.Tensor, threshold: float = 0.5) -> List[List[int]]:
    """由 Jev route 共激活矩阵对专家聚类，供 SSR 分簇合并。

    route_probs:[N, k_max] 每 token 的路由概率。共激活 C[i,j]=平均同时高激活的次数。
    简化：按每专家总负载排序，负载相近且共激活高的归为一簇(贪心阈值)。
    """
    N, K = route_probs.shape
    active = (route_probs > route_probs.mean(1, keepdim=True)).float()   # 每 token 的活跃掩码
    C = (active.T @ active) / N                        # [K,K] 共激活矩阵
    load = route_probs.mean(0)
    order = torch.argsort(load, descending=True).tolist()
    clusters, used = [], set()
    for i in order:
        if i in used:
            continue
        grp = [i]
        used.add(i)
        for j in order:
            if j not in used and C[i, j] > threshold * C[i, i]:
                grp.append(j)
                used.add(j)
        clusters.append(sorted(grp))
    return clusters
