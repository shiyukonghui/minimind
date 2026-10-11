# E1 初始化消融脚本（设计文档 §3.1，路线图任务④）
#
# 供体：Qwen2.5-0.5B bf16（hidden=896, 14头×64, 2KV头, inter=4864, 24层）。
# 与 MiniMind-512（hidden=512, 8头×64, 2KV头, inter≈1365, 8层）head_dim 同为 64，注意力可整头切块。
#
# 子命令：
#   profile  —— E1-C：按层类型统计供体权重的 mean/std 及重尾形状(kurtosis)，存为画像 JSON；
#   sample   —— E1-C：按画像为 MiniMind 各张量采样"分布匹配随机"初始化，存 .pth；
#   transplant —— E1-D：从供体切注意力头(14选8)整头移植到 MiniMind，MLP 维度不符则按画像采样，存 .pth。
import os, sys, json, argparse
import torch
from safetensors.torch import load_file

DONOR_DIR = r'H:\Strata-data\models\Qwen2.5-0.5B'
MINIMIND_CFG = r'F:\MachineLearn\minimind\checkpoints\pretrain_512.pth'
PROFILE_OUT = r'F:\MachineLearn\minimind\model\e1c_profile.json'
sys.path.insert(0, r'F:\MachineLearn\minimind')
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def load_donor() -> dict:
    files = [f for f in os.listdir(DONOR_DIR) if f.endswith('.safetensors')]
    if not files:
        raise FileNotFoundError(f'供体权重未就绪：{DONOR_DIR} 无 safetensors（请等待下载）')
    sd = {}
    for f in files:
        sd.update(load_file(os.path.join(DONOR_DIR, f)))
    return sd


def tensor_kind(name: str) -> str:
    """按张量名归类层类型(用于分布画像/移植对齐)。"""
    for key in ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'):
        if key in name:
            return key
    if 'norm' in name:
        return 'norm'
    if 'embed_tokens' in name or 'lm_head' in name:
        return 'embed'
    return 'other'


def cmd_profile():
    """E1-C 画像：逐层类型统计供体权重的 mean/std/kurtosis(重尾形状)。"""
    sd = load_donor()
    from collections import defaultdict
    groups = defaultdict(list)
    for k, v in sd.items():
        if v.dtype in (torch.bfloat16, torch.float16, torch.float32) and v.dim() >= 2:
            groups[tensor_kind(k)].append(v.float())
    profile = {}
    for kind, tensors in groups.items():
        flat = torch.cat([t.reshape(-1) for t in tensors])
        mean, std = flat.mean().item(), flat.std().item()
        # 超额峰度刻画重尾程度
        z = (flat - mean) / (std + 1e-9)
        kurt = (z.pow(4).mean().item() - 3.0)
        profile[kind] = {'mean': mean, 'std': std, 'excess_kurtosis': kurt, 'n': flat.numel()}
        print(f'{kind:10s} mean={mean:+.5f} std={std:.5f} kurt={kurt:+.3f} n={flat.numel():,}')
    json.dump(profile, open(PROFILE_OUT, 'w', encoding='utf-8'), indent=1)
    print(f'画像写入 {PROFILE_OUT}')


def sample_from_profile(shape, prof: dict, gen: torch.Generator) -> torch.Tensor:
    """按画像采样：重尾用 Student-t(自由度由峰度近似)混合，匹配 mean/std。"""
    m, s, k = prof['mean'], prof['std'], prof['excess_kurtosis']
    n = shape[0] * shape[1] if len(shape) >= 2 else int(torch.tensor(shape).prod())
    if k > 1.0:
        # 峰度高 -> 重尾：t 分布，nu≈4+6/(k+1) 使尾部与峰度同向(近似)
        nu = max(3.0, 4.0 + 6.0 / (k + 1.0))
        t = torch.distributions.StudentT(nu).sample((n,))
        t = t / t.std() * s + m
        x = t.reshape(shape)
    else:
        x = torch.randn(shape, generator=gen) * s + m
    return x


def cmd_sample(seed=0):
    """E1-C：为 MiniMind 每层按画像采样初始化(embed 保持随机)。"""
    prof = json.load(open(PROFILE_OUT, encoding='utf-8'))
    cfg = MiniMindConfig(hidden_size=512, num_hidden_layers=8, num_attention_heads=8,
                         num_key_value_heads=2, vocab_size=6400)
    model = MiniMindForCausalLM(cfg)
    gen = torch.Generator().manual_seed(seed)
    sd = model.state_dict()
    new_sd = {}
    for k, v in sd.items():
        kind = tensor_kind(k)
        if kind in prof and v.dim() >= 2 and kind != 'embed':
            new_sd[k] = sample_from_profile(tuple(v.shape), prof[kind], gen)
        else:
            new_sd[k] = v.clone()
    torch.save({'model_state': new_sd}, r'F:\MachineLearn\minimind\checkpoints\e1c_dist_sample.pth')
    print('E1-C 分布匹配初始化已保存 -> checkpoints/e1c_dist_sample.pth')


def cmd_transplant(seed=0):
    """E1-D：注意力整头移植(供体14头选8头)，MLP/norm/embed 按 E1-C 画像或随机。"""
    sd = load_donor()
    prof = json.load(open(PROFILE_OUT, encoding='utf-8')) if os.path.exists(PROFILE_OUT) else {}
    cfg = MiniMindConfig(hidden_size=512, num_hidden_layers=8, num_attention_heads=8,
                         num_key_value_heads=2, vocab_size=6400)
    model = MiniMindForCausalLM(cfg)
    gen = torch.Generator().manual_seed(seed)
    HD = 64                                              # head_dim，供体与 MiniMind 一致
    # 供体层 -> MiniMind 层：均匀映射(供体24层取8层)
    donor_layers = [round(i * (24 - 1) / (8 - 1)) for i in range(8)]
    new_sd = model.state_dict()
    for ml in range(8):
        dl = donor_layers[ml]
        dp = f'model.layers.{dl}.'
        mp = f'model.layers.{ml}.'
        # q_proj: 供体[14*64,896] -> 选前8头行 -> [8*64,512]；列需从896投到512? 供体输入=896≠512
        # 输入维度不同(896 vs 512)：对 q/k/v 取前 HD*heads 行 + 前 hidden 列(子矩阵)，保持头结构对齐
        # 供体键形如 model.layers.{dl}.self_attn.q_proj.weight（hidden=896）
        for name, heads in (('self_attn.q_proj.weight', 8), ('self_attn.k_proj.weight', 2),
                            ('self_attn.v_proj.weight', 2)):
            src = sd[dp + name].float()                            # 供体 [heads*64, 896]
            rows = heads * HD
            sub = src[:rows, :512]                                  # 取 heads 头行 × 512 列子块
            new_sd[mp + name] = sub.reshape(new_sd[mp + name].shape)
        # o_proj: 供体[896, 14*64] -> 取 [512, 8*64] 子块
        o = sd[dp + 'self_attn.o_proj.weight'].float()
        new_sd[mp + 'self_attn.o_proj.weight'] = o[:512, :8 * HD].reshape(new_sd[mp + 'self_attn.o_proj.weight'].shape)
        # MLP 维度不符(4864 vs 1365)：按画像采样(若无画像则保留随机)
        for name in ('mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight'):
            kind = name.split('.')[1]
            if kind in prof:
                new_sd[mp + name] = sample_from_profile(tuple(new_sd[mp + name].shape), prof[kind], gen)
        # norm 直接复制(同维度 512? 供体 norm 是 896，需截断)
        for name in ('input_layernorm.weight', 'post_attention_layernorm.weight'):
            src = sd[dp + name].float()
            new_sd[mp + name] = src[:512] if src.shape[0] >= 512 else src
    torch.save({'model_state': new_sd}, r'F:\MachineLearn\minimind\checkpoints\e1d_transplant.pth')
    print('E1-D 真实权重移植已保存 -> checkpoints/e1d_transplant.pth')
    print(f'  供体层映射: {donor_layers}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['profile', 'sample', 'transplant'])
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    {'profile': cmd_profile, 'sample': cmd_sample, 'transplant': cmd_transplant}[a.cmd](**({'seed': a.seed} if a.cmd != 'profile' else {}))
