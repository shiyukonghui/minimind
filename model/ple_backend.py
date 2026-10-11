# Qwen3.8-Flash PLE(N-gram 记忆表)复用后端 —— 设计文档 §3.5 路径1「键空间桥接」
#
# 功能：把 MiniMind 的输入文本经 Qwen BPE 编码为 token 序列，按 qwen4exp 的 3-gram XOR 哈希
#       从冻结的 per_layer_token_embd 大表(IQ4_NL, 28.8GB, mmap 随机读)查得每 token 的
#       规范记忆向量(16 头 × 160 维 = 2560)，宿主侧仅训练 W_proj(2560→hidden) + gate。
#
# 关键事实(均已从 GGUF/源码逐行核实)：
#   - 表张量 per_layer_token_embd.weight 形状 [160, 320001536]，IQ4_NL，存于分片2，数据段基址=192，每行 5 block×18B=90B
#   - 哈希(qwen4exp.cpp L1036-1047)：n∈{2,3} 分别算 mixed=(ctx0*m0)^...^(ctx_{n-1}*m_{n-1})，
#     每种 n 各 8 头共享同一 mixed，row=mixed%head_vocab[h]+head_offset[h]，共 16 头；EOS 重置窗口
#   - IQ4_NL 反量化(ggml-quants.c)：block=f16 d + 16 字节(32 nibble)，y=d*kvalues[nibble]，
#     码本 kvalues=[-127,-104,...,113]，block 内顺序为 16 个低 nibble 再 16 个高 nibble
#   - 门控(build_ple L1130-1212)：s=Σ(key⊙query)/√d，gate=σ(sign(s)·√|s|)，out=hidden+gate⊙value+conv
import json
import numpy as np
import torch
from torch import nn
from collections import OrderedDict

# IQ4_NL 码本(源自 ggml-common.h kvalues_iq4nl)
IQ4NL_KVALUES = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=np.float32)
QK4_NL = 32
BLOCK_BYTES = 18          # 2(f16 d) + 16(qs)
IQ4NL_BPW = BLOCK_BYTES / QK4_NL  # 0.5625 = 4.5 bit/元素

META_PATH = r'F:\MachineLearn\minimind\model\ple_meta_qwen38.json'


def dequant_iq4_nl(buf: np.ndarray) -> np.ndarray:
    """反量化一段连续 IQ4_NL 字节(长度须为 18 的倍数) -> float32 数组(长度=字节数/18*32)。"""
    assert buf.size % BLOCK_BYTES == 0
    b = np.asarray(buf).reshape(-1, BLOCK_BYTES)
    # view 后 shape 为 (nb,1)，必须 reshape(-1) 展平，否则 d[:,None] 变 (nb,1,1) 触发错误广播
    d = b[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1)   # 每块 scale
    qs = b[:, 2:]                                                         # [nb,16]
    lo = IQ4NL_KVALUES[qs & 0xF]                                          # 低 nibble -> 元素 0..15
    hi = IQ4NL_KVALUES[qs >> 4]                                           # 高 nibble -> 元素 16..31
    vals = np.concatenate([lo, hi], axis=1)                              # [nb,32] 顺序: 先16低后16高
    return (vals * d[:, None]).ravel()


class IQ4NLMmapTable:
    """对 GGUF 分片2 的 IQ4_NL 大表做 mmap 行查表 + LRU 缓存(避免重复反量化热点行)。"""

    def __init__(self, shard2_path: str, data_start: int, n_rows: int, head_dim: int = 160, cache_size: int = 4096):
        # 用 shape 显式指定一维字节数组，规避 numpy memmap 在带 offset 时的二维推断
        total_bytes = n_rows * (head_dim // QK4_NL * BLOCK_BYTES)
        self.mm = np.memmap(shard2_path, dtype=np.uint8, mode='r',
                            offset=data_start, shape=(total_bytes,))
        self.data_start = 0  # memmap 已从 data_start 起，内部索引相对化
        self.n_rows = n_rows
        self.head_dim = head_dim
        self.row_bytes = head_dim // QK4_NL * BLOCK_BYTES            # 160/32*18 = 90
        self.cache_size = cache_size
        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self.hits = self.misses = 0

    def row(self, idx: int) -> np.ndarray:
        """返回第 idx 行的 head_dim 维 float32 记忆向量。"""
        if idx in self._cache:
            self._cache.move_to_end(idx)
            self.hits += 1
            return self._cache[idx]
        self.misses += 1
        s = idx * self.row_bytes
        buf = np.asarray(self.mm[s:s + self.row_bytes]).reshape(-1)
        vec = dequant_iq4_nl(buf)
        if len(self._cache) >= self.cache_size:
            self._cache.popitem(last=False)
        self._cache[idx] = vec
        return vec

    def rows(self, idxs: np.ndarray) -> np.ndarray:
        """批量查行 -> [len(idxs), head_dim]。"""
        return np.stack([self.row(int(i)) for i in idxs], axis=0)


class PLEHashIndexer:
    """按 qwen4exp 公式把 Qwen token 序列映射为每 token 的 16 个表行索引。"""

    def __init__(self, meta: dict):
        kv = meta['kv_meta']
        self.multipliers = np.array(kv['qwen4exp.ple.layer_multipliers'], dtype=np.uint64)
        self.head_offsets = np.array(kv['qwen4exp.ple.head_offsets'], dtype=np.int64)
        self.head_vocab = np.array(kv['qwen4exp.ple.head_vocab_sizes'], dtype=np.int64)
        self.ngram = int(kv['qwen4exp.ple.ngram_size'])
        self.per_gram = int(kv['qwen4exp.ple.heads_per_ngram'])
        self.n_heads = len(self.head_offsets)                        # 16
        self.eos = int(kv['qwen4exp.ple.eos_token_id'])
        self.M64 = np.uint64(0xFFFFFFFFFFFFFFFF)

    def _mixed(self, ctx: list) -> int:
        """ctx[0]=当前token, ctx[j]=前第j个; 由调用方保证长度>=n。"""
        acc = np.uint64(ctx[0]) * self.multipliers[0]
        for j in range(1, len(ctx)):
            acc ^= np.uint64(ctx[j]) * self.multipliers[j]
        return int(acc & self.M64)

    def index(self, token_ids: list) -> np.ndarray:
        """token_ids: Qwen BPE id 列表 -> [T, n_heads] 行索引(int64)。"""
        T = len(token_ids)
        out = np.zeros((T, self.n_heads), dtype=np.int64)
        n_prev = self.ngram - 1
        for i in range(T):
            # 构造窗口 ctx，EOS 重置：EOS 之前(含缺失前驱)读作 EOS
            ctx = [token_ids[i]]
            cut = False
            for s in range(1, self.ngram):
                k = i - s
                t = self.eos if (cut or k < 0) else token_ids[k]
                if k >= 0 and (token_ids[k] == self.eos):
                    cut = True
                ctx.append(t)
            # n=2 用 ctx[0..1]; n=3 用 ctx[0..2]; 每种 n 的 8 头共享同一 mixed
            for n in range(2, self.ngram + 1):
                mixed = self._mixed(ctx[:n])
                base = (n - 2) * self.per_gram
                for g in range(self.per_gram):
                    h = base + g
                    out[i, h] = mixed % self.head_vocab[h] + self.head_offsets[h]
        return out


class QwenPLEBackend(nn.Module):
    """冻结 PLE 表 + 可训练宿主侧投影/门控。

    forward(qwen_token_ids, host_hidden) -> 记忆增强后的 hidden(与输入同形状)。
    仅 emb_down / key_proj / query_proj / value_proj / out_norm 参与训练(约 17.6 万参数)，
    冻结的 PLE 大表本身只读。
    """

    def __init__(self, shard2_path: str, host_hidden: int, embed_dim: int = 2560,
                 meta_path: str = META_PATH, data_start: int = 192, proj_rank: int = 48):
        super().__init__()
        meta = json.load(open(meta_path, encoding='utf-8'))
        tbl = next(t for t in meta['tensors'] if 'per_layer_token_embd' in t['name'])
        self.head_dim = tbl['shape'][0]                              # 160
        self.n_rows = tbl['shape'][1]                                # 320001536
        self.table = IQ4NLMmapTable(shard2_path, data_start, self.n_rows, self.head_dim)
        self.indexer = PLEHashIndexer(meta)
        self.n_heads = self.indexer.n_heads
        self.embed_dim = self.n_heads * self.head_dim                # 2560
        assert self.embed_dim == embed_dim, f'embed_dim {embed_dim} != n_heads*head_dim {self.embed_dim}'

        # 宿主侧可训练模块(表冻结)。低秩共享 emb_down 把参数量压到 ~17 万(设计目标 10-20 万)：
        #   emb(2560) --emb_down--> z(r) --key_proj--> key(r) ; hidden --query_proj--> query(r)
        #   gate = σ(sign(s)·√|s|), s=key·query/√r ; z --value_proj--> value(hidden)
        self.proj_rank = proj_rank
        self.emb_down = nn.Linear(self.embed_dim, proj_rank, bias=False)   # 2560*48
        self.key_proj = nn.Linear(proj_rank, proj_rank)                    # 48*48
        self.query_proj = nn.Linear(host_hidden, proj_rank)               # 512*48
        self.value_proj = nn.Linear(proj_rank, host_hidden)               # 48*512
        self.out_norm = nn.LayerNorm(host_hidden)
        self.host_hidden = host_hidden

    @torch.no_grad()
    def retrieve(self, qwen_token_ids: list, device) -> torch.Tensor:
        """查表得每 token 的 2560 维规范记忆向量(冻结)。"""
        idx = self.indexer.index(qwen_token_ids)                    # [T,16]
        emb = self.table.rows(idx.ravel())                          # [T*16,160]
        emb = emb.reshape(len(qwen_token_ids), self.embed_dim)      # [T,2560]
        return torch.from_numpy(emb).to(device)

    def forward(self, qwen_token_ids: list, host_hidden: torch.Tensor) -> torch.Tensor:
        """host_hidden: [1,T,hidden]。返回记忆增强后的 [1,T,hidden]。"""
        assert host_hidden.shape[1] == len(qwen_token_ids)
        emb = self.retrieve(qwen_token_ids, host_hidden.device)     # [T,2560]
        z = self.emb_down(emb)                                      # [T,r]
        key = self.key_proj(z)                                      # [T,r]
        query = self.query_proj(host_hidden.squeeze(0))            # [T,r]
        s = (key * query).sum(-1) / (self.proj_rank ** 0.5)        # [T]
        gate = torch.sigmoid(torch.sign(s) * torch.sqrt(s.abs() + 1e-6))  # 带符号平方根门控
        value = self.value_proj(z)                                  # [T,hidden]
        gated = gate.unsqueeze(-1) * value                          # [T,hidden]
        out = self.out_norm(host_hidden.squeeze(0) + gated)         # 残差 + 归一化
        return out.unsqueeze(0)

    @property
    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
