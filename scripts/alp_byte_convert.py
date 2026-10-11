# ALP 阶段1：subword 模型 -> 字节模型的闭式嵌入转换（设计文档 §3.6 路线α，任务⑥前置）
#
# 依据 Bolmo(ALP, arXiv 2408.15430) 思想：不从头重训，而是把现有 subword 模型的
# 词嵌入"折叠"进字节嵌入，得到高质量的字节嵌入初始化：
#   - 字节 b 的嵌入 = 所有"文本以字节 b 开头"的 subword token 嵌入的均值
#     （ALP 的 prefix-mean 初始化；lm_head 用 suffix-mean，因输出侧关心 token 收尾字节）
#   - 特殊 token（pad/bos/eos）保留原 subword 模型对应特殊嵌入（取同 id 行或零初始化）
#   - Transformer 主体（注意力/FFN/norm）原样保留 —— 字节化只换输入输出侧
#
# 产物：checkpoints/pretrain_512_byteinit.pth（vocab 6400 -> 260 的字节化模型权重），
#       供阶段2（字节端到端 LoRA 微调，需真实语料训练）直接作为起点。
# 用法：python scripts/alp_byte_convert.py [--ckpt 路径] [--out 路径]
import argparse
import torch

sys_path = __import__('sys').path
sys_path.insert(0, r'F:\MachineLearn\minimind')
from model.byte_tokenizer import BYTE_VOCAB_SIZE, BYTE_PAD, BYTE_BOS, BYTE_EOS

SRC_CKPT = r'F:\MachineLearn\minimind\checkpoints\pretrain_512.pth'
DST_CKPT = r'F:\MachineLearn\minimind\checkpoints\pretrain_512_byteinit.pth'


def build_id2bytes(tokenizer, vocab_size: int):
    """每个 subword id -> 其文本的 UTF-8 字节序列。"""
    id2bytes = {}
    for tid in range(vocab_size):
        try:
            txt = tokenizer.decode([tid])
        except Exception:
            txt = ''
        if not txt:
            continue
        id2bytes[tid] = list(txt.encode('utf-8'))
    return id2bytes


def alp_fold(embed: torch.Tensor, id2bytes: dict, mode: str = 'prefix'):
    """把 subword 嵌入折叠为字节嵌入：mode=prefix 取首字节分组均值，suffix 取末字节。"""
    H = embed.shape[1]
    out = torch.zeros(BYTE_VOCAB_SIZE, H, dtype=torch.float32)
    cnt = torch.zeros(BYTE_VOCAB_SIZE, dtype=torch.long)
    for tid, bs in id2bytes.items():
        if not bs:
            continue
        b = bs[0] if mode == 'prefix' else bs[-1]
        out[b] += embed[tid].float()
        cnt[b] += 1
    nz = cnt > 0
    out[nz] /= cnt[nz].unsqueeze(-1)
    # 无 token 覆盖的字节（如控制字节）：用全局均值的小扰动初始化，避免零向量死槽
    miss = ~nz
    if miss.any():
        g = embed.float().mean(0)
        out[miss] = g.unsqueeze(0) + 0.01 * torch.randn(int(miss.sum()), H)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=SRC_CKPT)
    ap.add_argument('--out', default=DST_CKPT)
    args = ap.parse_args()

    # 用 tokenizers 库(纯 Rust，无 jinja 依赖)读 id->文本，避开 transformers 环境冲突
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(r'F:\MachineLearn\minimind\model\tokenizer.json')

    sd = torch.load(args.ckpt, map_location='cpu')
    sd = sd.get('model_state', sd) if isinstance(sd, dict) and 'model_state' in sd else sd
    embed = sd['model.embed_tokens.weight'].float()        # [6400, 512]
    tied = 'lm_head.weight' not in sd or torch.equal(sd['lm_head.weight'].float(), embed)
    head = sd['lm_head.weight'].float() if 'lm_head.weight' in sd else embed
    print(f'源词表 {embed.shape[0]}，lm_head {"与 embed 绑定" if tied else "独立"}')

    id2bytes = build_id2bytes(tok, embed.shape[0])
    byte_embed = alp_fold(embed, id2bytes, mode='prefix')  # 输入侧：首字节均值
    byte_head = alp_fold(head, id2bytes, mode='suffix')    # 输出侧：末字节均值

    # 特殊 token 行：用原模型特殊 id 的嵌入(截断到 260 行内不存在则保留折叠值)
    # 字节特殊 id 256-259 在 0-255 折叠之外，直接零初始化+小扰动，交给阶段2 训练
    for sid in (BYTE_PAD, BYTE_BOS, BYTE_EOS):
        byte_embed[sid] = torch.randn(embed.shape[1]) * 0.02
        byte_head[sid] = torch.randn(embed.shape[1]) * 0.02

    new_sd = dict(sd)
    new_sd['model.embed_tokens.weight'] = byte_embed.half()
    if 'lm_head.weight' in new_sd:
        new_sd['lm_head.weight'] = (byte_embed if tied else byte_head).half()
    # 其余主体（注意力/FFN/norm）原样保留
    torch.save({'model_state': new_sd}, args.out)
    print(f'字节化初始化模型 -> {args.out}（vocab {embed.shape[0]} -> {BYTE_VOCAB_SIZE}）')

    # 摘要统计：每个字节槽的 token 覆盖数
    import collections
    first_cnt = collections.Counter(bs[0] for bs in id2bytes.values() if bs)
    print(f'首字节覆盖 {len(first_cnt)}/256 字节；'
          f'平均折叠 token 数={sum(first_cnt.values())/max(len(first_cnt),1):.1f}')


if __name__ == '__main__':
    main()
