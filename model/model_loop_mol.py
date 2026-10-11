# Loop-MoL-MiniMind 融合架构（设计文档 §3.2/§3.3/§3.4，路线图任务③）
#
# 组成：
#   - LoopedBlock：单个 Transformer 块(注意力+MoL FFN)沿深度循环 T 次，
#     timestep 编码区分第 t 步，DeepLoop 残差缩放 α_t=1/√(1+t) 保证循环稳定；
#   - MoL 专家池：冻结共享 FFN + K_max 个 LoRA 低秩专家，top-2 稀疏激活 + 负载辅助损失；
#   - Jev 决策头(System-1)：halt(循环停止/继续)、route(专家路由)、abstain(弃权→生长信号)，
#     输出带类型决策 + 校准概率，配 Brier 自洽校准损失与温度缩放。
#
# 训练约束：底座 MiniMind-512 全程冻结，仅训练专家池/决策头/timestep 编码/温度。
# 推理：halt 权重 ACT 式加权求和；可早退(p_t>阈值即停)。
import math
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Tuple, List

from model.model_minimind import MiniMindConfig, Attention, FeedForward, RMSNorm


class TimestepEncoder(nn.Module):
    """循环步编码：可学习 per-step 嵌入，零初始化使循环起点等价于底座单层。"""

    def __init__(self, hidden_size: int, max_steps: int = 8):
        super().__init__()
        self.emb = nn.Embedding(max_steps, hidden_size)
        nn.init.zeros_(self.emb.weight)

    def forward(self, t: int) -> torch.Tensor:
        return self.emb.weight[t]


class LoRAExpert(nn.Module):
    """FFN 槽位的低秩专家 delta：B·SiLU(A(x))，hidden→hidden，B 零初始化(出生即中性)。"""

    def __init__(self, hidden_size: int, rank: int = 16):
        super().__init__()
        self.A = nn.Linear(hidden_size, rank, bias=False)
        self.B = nn.Linear(rank, hidden_size, bias=False)
        nn.init.normal_(self.A.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.B(F.silu(self.A(x)))


class MoLFFN(nn.Module):
    """MoL 模拟 MoE：冻结共享 FFN + LoRA 专家池，按路由权重 top-k 稀疏激活。"""

    def __init__(self, config: MiniMindConfig, k_max: int = 32, top_k: int = 2, rank: int = 16):
        super().__init__()
        self.shared = FeedForward(config)          # 底座权重，冻结
        self.experts = nn.ModuleList([LoRAExpert(config.hidden_size, rank) for _ in range(k_max)])
        self.k_max, self.top_k = k_max, top_k

    def forward(self, x: torch.Tensor, topk_idx: torch.Tensor, topk_w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x:[N,h]；topk_idx:[N,k]；topk_w:[N,k]。返回 (输出, 负载辅助损失)。"""
        y = self.shared(x)
        N = x.shape[0]
        flat_idx = topk_idx.reshape(-1)
        flat_w = topk_w.reshape(-1)
        delta = torch.zeros_like(y)
        for e in range(self.k_max):
            mask = flat_idx == e
            if not mask.any():
                continue
            tok = mask.nonzero(as_tuple=True)[0] // self.top_k   # 还原 token 下标
            expert_in = x[tok]
            out = self.experts[e](expert_in) * flat_w[mask].unsqueeze(-1)
            delta.index_add_(0, tok, out)
        # Switch 式负载辅助损失：P_i·f_i（各专家平均路由概率 × 被选频率）
        # probs:[N,k] 每专家概率质量(把 top-k 权重 scatter 回全专家空间)，p_i 为其 token 均值
        probs = torch.zeros(N, self.k_max, device=x.device, dtype=topk_w.dtype)
        probs.scatter_(1, topk_idx, topk_w)
        p_i = probs.mean(0)
        f_i = torch.bincount(flat_idx, minlength=self.k_max).float() / (N * self.top_k)
        aux = self.k_max * (p_i * f_i).sum()
        return y + delta, aux


class JevHeads(nn.Module):
    """Jev 式三决策头：halt / route / abstain，route 带可学习温度缩放。"""

    def __init__(self, hidden_size: int, k_max: int):
        super().__init__()
        self.halt = nn.Linear(hidden_size, 1)
        self.route = nn.Linear(hidden_size, k_max)
        self.abstain = nn.Linear(hidden_size, 1)
        self.log_temp = nn.Parameter(torch.zeros(1))              # 温度=exp(log_temp)，初值 1

    def forward(self, x: torch.Tensor, top_k: int) -> dict:
        temp = self.log_temp.exp().clamp(0.1, 10.0)
        halt_logit = self.halt(x).squeeze(-1)                     # [N]
        route_logits = self.route(x) / temp                       # [N,k]
        abstain_logit = self.abstain(x).squeeze(-1)               # [N]
        route_probs = F.softmax(route_logits, dim=-1)
        topk_w, topk_idx = torch.topk(route_probs, k=top_k, dim=-1)
        topk_w = topk_w / (topk_w.sum(-1, keepdim=True) + 1e-20)  # top-k 重归一
        return {
            'halt_prob': torch.sigmoid(halt_logit),               # P(在第 t 步停止)
            'route_probs': route_probs,
            'topk_idx': topk_idx, 'topk_w': topk_w,
            'abstain_prob': torch.sigmoid(abstain_logit),
            'route_conf': topk_w.sum(-1),                         # top-k 置信度(弃权目标源)
        }


class LoopedBlock(nn.Module):
    """循环块：注意力(冻结) + MoL FFN，循环 T 步，halt 头做 ACT 式深度分配。"""

    def __init__(self, config: MiniMindConfig, k_max: int = 32, top_k: int = 2,
                 rank: int = 16, max_steps: int = 8):
        super().__init__()
        h = config.hidden_size
        self.attn = Attention(config)                              # 底座权重，冻结
        self.norm1 = RMSNorm(h, eps=config.rms_norm_eps)
        self.norm2 = RMSNorm(h, eps=config.rms_norm_eps)
        self.mol = MoLFFN(config, k_max, top_k, rank)
        self.temb = TimestepEncoder(h, max_steps)
        self.jev = JevHeads(h, k_max)
        self.top_k, self.max_steps, self.k_max = top_k, max_steps, k_max

    def forward(self, h: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                T: int = 4, attention_mask: Optional[torch.Tensor] = None) -> dict:
        """h:[B,S,hid]。返回含加权隐状态与各步决策概率的 dict。"""
        B, S, H = h.shape
        x = h
        step_outs, halt_ps, abstain_ps, route_ps = [], [], [], []
        confs, aux_sum = [], 0.0
        for t in range(T):
            xin = x + self.temb(t)                                 # timestep 编码注入
            a, _ = self.attn(self.norm1(xin), position_embeddings,
                             None, False, attention_mask)
            x = x + a                                              # 注意力残差(不缩放，与底座一致)
            hn = self.norm2(x).reshape(-1, H)                      # [N,H]，N=B*S
            dec = self.jev(hn, self.top_k)
            f, aux = self.mol(hn, dec['topk_idx'], dec['topk_w'])
            alpha = 1.0 / math.sqrt(1.0 + t)                       # DeepLoop 残差缩放
            x = x + alpha * f.reshape(B, S, H)
            step_outs.append(x)
            halt_ps.append(dec['halt_prob'].reshape(B, S))          # [B,S]
            abstain_ps.append(dec['abstain_prob'].reshape(B, S))
            route_ps.append(dec['route_probs'].reshape(B, S, self.k_max))
            confs.append(dec['route_conf'].reshape(B, S))
            aux_sum = aux_sum + aux
        # ACT 加权：w_t = p_t·Π_{s<t}(1-p_s)，末步强制停止
        hp = torch.stack(halt_ps, dim=2)                           # [B,S,T]
        cum = torch.cumprod(1.0 - hp, dim=2)
        w = hp * F.pad(cum[:, :, :-1], (1, 0), value=1.0)          # [B,S,T]
        w = torch.cat([w[:, :, :-1], 1.0 - w[:, :, :-1].sum(2, keepdim=True)], dim=2)
        w = w / (w.sum(2, keepdim=True) + 1e-20)                   # 归一化
        stack = torch.stack(step_outs, dim=2)                      # [B,S,T,H]
        out = (stack * w.unsqueeze(-1)).sum(2)                     # [B,S,H]
        return {
            'hidden': out,
            'halt_w': w,
            'halt_p': hp,
            'abstain_p': torch.stack(abstain_ps, dim=2),
            'route_p': torch.stack(route_ps, dim=2),               # [B,S,T,k]
            'route_conf': torch.stack(confs, dim=2),               # [B,S,T]
            'aux_loss': aux_sum / T,
            'mean_depth': (w * torch.arange(T, device=h.device, dtype=h.dtype)).sum(-1).mean(),
        }


def jev_calibration_losses(block_out: dict, conf_tau: float = 0.5,
                           depth_ent_coef: float = 0.01, abstain_coef: float = 0.1) -> dict:
    """Jev 校准损失：
    - 深度熵正则：防止 halt 分布坍缩到全早停/全晚停(Ouro 思路)；
    - 弃权 Brier 自洽：abstain 概率应预测 route 置信度是否低于阈值(校准一致性)。
    """
    w = block_out['halt_w']                                        # [B,S,T]
    depth_ent = -(w.clamp_min(1e-8) * w.clamp_min(1e-8).log()).sum(-1).mean()
    target = (block_out['route_conf'] < conf_tau).float()          # 低置信 → 应弃权
    brier = ((block_out['abstain_p'] - target) ** 2).mean()
    return {
        'depth_ent': -depth_ent * depth_ent_coef,                  # 熵最大化 → 取负
        'abstain_brier': brier * abstain_coef,
    }


class LoopMiniMindForCausalLM(nn.Module):
    """完整模型：冻结底座(embed/attn/共享FFN/norm/lm_head) + 可训练循环块组件。

    底座权重从 pretrain_512.pth 的指定层载入(单层复制=ALBERT 式，E1-D 天然契合)。
    """

    def __init__(self, config: MiniMindConfig = None, k_max: int = 32, top_k: int = 2,
                 rank: int = 16, max_steps: int = 8):
        super().__init__()
        self.config = config or MiniMindConfig()
        h = self.config.hidden_size
        self.embed_tokens = nn.Embedding(self.config.vocab_size, h)
        self.block = LoopedBlock(self.config, k_max, top_k, rank, max_steps)
        self.norm = RMSNorm(h, eps=self.config.rms_norm_eps)
        self.lm_head = nn.Linear(h, self.config.vocab_size, bias=False)
        # RoPE 频率与底座一致
        from model.model_minimind import precompute_freqs_cis
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=h // self.config.num_attention_heads,
            end=self.config.max_position_embeddings,
            rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
        self.register_buffer('freqs_cos', freqs_cos, persistent=False)
        self.register_buffer('freqs_sin', freqs_sin, persistent=False)
        self.freeze_base()                                         # 架构约束：底座全程冻结

    def load_base(self, ckpt_path: str, layer_id: int = 3):
        """从 MiniMind 预训练权重载入指定层到循环块(单层复制=ALBERT 式)，并冻结全部底座参数。"""
        sd = torch.load(ckpt_path, map_location='cpu')
        sd = sd.get('model_state', sd) if isinstance(sd, dict) and 'model_state' in sd else sd
        pre = f'layers.{layer_id}.'                                # kk 已剥离 'model.' 前缀
        # 底座层内前缀 -> 循环块内前缀 的改名表
        renames = {
            'self_attn.': 'block.attn.',
            'input_layernorm.weight': 'block.norm1.weight',
            'post_attention_layernorm.weight': 'block.norm2.weight',
            'mlp.': 'block.mol.shared.',
        }
        mapping = {}
        for k, v in sd.items():
            kk = k[len('model.'):] if k.startswith('model.') else k
            if kk.startswith(pre):
                suffix = kk[len(pre):]
                for src, dst in renames.items():
                    if suffix.startswith(src):
                        mapping[dst + suffix[len(src):]] = v.to(torch.float32)   # ckpt 可能是 fp16
                        break
            elif kk == 'lm_head.weight':
                mapping['lm_head.weight'] = v.to(torch.float32)
            elif kk == 'embed_tokens.weight':
                mapping['embed_tokens.weight'] = v.to(torch.float32)
            elif kk == 'norm.weight':
                mapping['norm.weight'] = v.to(torch.float32)
        load_result = self.load_state_dict(mapping, strict=False)
        self.freeze_base()
        return mapping.keys(), load_result

    def freeze_base(self):
        """冻结底座：embed/attn/共享FFN/两个norm/lm_head。专家池与决策头保持可训练。"""
        for p in self.embed_tokens.parameters(): p.requires_grad = False
        for p in self.lm_head.parameters(): p.requires_grad = False
        for p in self.norm.parameters(): p.requires_grad = False
        for p in self.block.attn.parameters(): p.requires_grad = False
        for p in self.block.norm1.parameters(): p.requires_grad = False
        for p in self.block.norm2.parameters(): p.requires_grad = False
        for p in self.block.mol.shared.parameters(): p.requires_grad = False

    @property
    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_ids: torch.Tensor, T: int = 4,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> dict:
        B, S = input_ids.shape
        h = self.embed_tokens(input_ids)
        pos_emb = (self.freqs_cos[:S], self.freqs_sin[:S])
        out = self.block(h, pos_emb, T=T, attention_mask=attention_mask)
        hidden = self.norm(out['hidden'])
        logits = self.lm_head(hidden)
        res = {'logits': logits, **out}
        if labels is not None:
            res['lm_loss'] = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                                             labels[:, 1:].reshape(-1), ignore_index=-1)
            cal = jev_calibration_losses(out)
            res.update(cal)
            res['loss'] = res['lm_loss'] + res['aux_loss'] + cal['depth_ent'] + cal['abstain_brier']
        return res
