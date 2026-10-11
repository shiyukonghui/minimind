# QwenPLEBackend 冒烟测试：验证 IQ4_NL 反量化、哈希索引、端到端前向
# 用法：python scripts/test_ple_backend.py
import sys, json
import numpy as np
import torch

sys.path.insert(0, r'F:\MachineLearn\minimind')
from model.ple_backend import (dequant_iq4_nl, IQ4NLMmapTable, PLEHashIndexer,
                               QwenPLEBackend, BLOCK_BYTES, META_PATH)

SHARD2 = r'H:\Strata-data\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf'

# ---------- 1. 反量化正确性：手造 block 与已知 scale 对齐 ----------
def test_dequant():
    # 构造一个 block：d=2.0(f16)，16 字节 = 32 nibble，第0个低nibble选码本索引8(值=1)
    d = np.array([2.0], dtype=np.float16).view(np.uint8)   # 2 字节
    qs = np.zeros(16, dtype=np.uint8)
    qs[0] = 0x08          # 低 nibble=8 -> kvalues[8]=1 ; 高 nibble=0 -> kvalues[0]=-127
    blk = np.concatenate([d, qs])
    out = dequant_iq4_nl(blk)
    assert out.size == 32
    assert abs(out[0] - 2.0 * 1) < 1e-3, out[0]        # 元素0 = d*1
    assert abs(out[16] - 2.0 * (-127)) < 1e-1, out[16] # 元素16 = d*(-127)
    print('[1] 反量化正确 ✓  元素0=%.3f 元素16=%.3f' % (out[0], out[16]))

# ---------- 2. 哈希索引：与 C++ 逻辑一致性(手算对照) ----------
def test_indexer():
    meta = json.load(open(META_PATH, encoding='utf-8'))
    idx = PLEHashIndexer(meta)
    toks = [100, 200, 300, 400]
    rows = idx.index(toks)
    assert rows.shape == (4, 16)
    # 手算 token[2]=300 的 3-gram mixed 与 head0 行号
    m = idx.multipliers
    mixed3 = (np.uint64(300) * m[0]) ^ (np.uint64(200) * m[1]) ^ (np.uint64(100) * m[2])
    mixed3 = int(mixed3 & np.uint64(0xFFFFFFFFFFFFFFFF))
    exp_h8 = mixed3 % idx.head_vocab[8] + idx.head_offsets[8]   # n=3 的头基址=8
    assert rows[2, 8] == exp_h8, (rows[2, 8], exp_h8)
    # head0(n=2) 用 2-gram
    mixed2 = (np.uint64(300) * m[0]) ^ (np.uint64(200) * m[1])
    mixed2 = int(mixed2 & np.uint64(0xFFFFFFFFFFFFFFFF))
    exp_h0 = mixed2 % idx.head_vocab[0] + idx.head_offsets[0]
    assert rows[2, 0] == exp_h0, (rows[2, 0], exp_h0)
    # EOS 重置：序列 [300, eos, 400]，位置2 的 2-gram 应把前驱读作 eos
    eos = idx.eos
    rows2 = idx.index([300, eos, 400])
    mixed_eos = (np.uint64(400) * m[0]) ^ (np.uint64(eos) * m[1])
    mixed_eos = int(mixed_eos & np.uint64(0xFFFFFFFFFFFFFFFF))
    assert rows2[2, 0] == mixed_eos % idx.head_vocab[0] + idx.head_offsets[0]
    print('[2] 哈希索引正确 ✓  与 C++ 公式手算一致(含 EOS 重置)')

# ---------- 3. 端到端：真实表 mmap 查表 + 前向形状 ----------
def test_forward():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(r'H:\Strata-data\models\Qwen3.8-Flash-Next-BF16\tokenizer.json')
    ids = tok.encode('记忆模块复用测试', add_special_tokens=False).ids
    be = QwenPLEBackend(SHARD2, host_hidden=512)
    print(f'[3a] 可训练参数量 = {be.trainable_param_count:,} (目标 ~10-20万)')
    h = torch.randn(1, len(ids), 512)
    out = be.forward(ids, h)
    assert out.shape == h.shape, out.shape
    assert torch.isfinite(out).all()
    print(f'[3] 端到端前向 ✓  输入 {len(ids)} token -> 输出 {tuple(out.shape)}，表缓存命中/缺失={be.table.hits}/{be.table.misses}')

if __name__ == '__main__':
    test_dequant()
    test_indexer()
    test_forward()
    print('\n全部通过')
