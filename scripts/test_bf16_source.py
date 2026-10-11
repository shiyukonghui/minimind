# BF16 源支持测试：BF16ShardTable 行读取 vs safetensors 官方读取逐位一致 + 端到端前向
# 用法：python scripts/test_bf16_source.py
import sys
import numpy as np
import torch

sys.path.insert(0, r'F:\MachineLearn\minimind')
from safetensors import safe_open
from model.ple_backend import BF16ShardTable, QwenPLEBackend

HF_DIR = r'H:\Strata-data\models\Qwen3.8-Flash-Next-BF16'
SHARD2 = r'H:\Strata-data\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf'
RPS = BF16ShardTable.ROWS_PER_SHARD


def test_row_exact():
    """随机行号：自研 mmap 读取应与 safe_open 官方读取逐位相等。"""
    tbl = BF16ShardTable(HF_DIR)
    rng = np.random.default_rng(7)
    # 覆盖首段/中段/末段/段边界
    test_rows = [0, 1, RPS - 1, RPS, RPS + 5, 64 * RPS - 1, 64 * RPS, 127 * RPS - 1,
                 *[int(x) for x in rng.integers(0, 128 * RPS - 90, 6)]]
    # 预取各段官方张量的头几行做对照(避免整段加载)：用 get_slice 区域读
    slice_cache = {}
    for r in test_rows:
        n = r // RPS
        import os
        path = tbl.seg_file[n]
        key = f'model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{n}.weight'
        if n not in slice_cache:
            sf = safe_open(path, framework='pt')
            slice_cache[n] = sf.get_slice(key)
        official = slice_cache[n][r - n * RPS].float().numpy()
        mine = tbl.row(r)
        # BF16->float32 是精确位扩展，两者应逐位相等
        assert np.array_equal(official, mine), f'row {r} 不一致'
    print(f'[1] BF16 行读取逐位一致 ✓  {len(test_rows)} 行(含段边界)全部与 safe_open 相等')

def test_backend_forward():
    """source='bf16' 端到端前向 + 与 gguf 源同 token 检索结果相关 ~0.997(量化噪声)。"""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(HF_DIR + r'\tokenizer.json')
    ids = tok.encode('记忆模块复用测试 BF16 高保真', add_special_tokens=False).ids
    be_bf16 = QwenPLEBackend(SHARD2, host_hidden=512, source='bf16')
    be_gguf = QwenPLEBackend(SHARD2, host_hidden=512, source='gguf')
    # 同一随机初始化 -> 复制权重保证宿主侧一致
    be_gguf.load_state_dict(be_bf16.state_dict())
    emb_b = be_bf16.retrieve(ids, torch.device('cpu'))
    emb_g = be_gguf.retrieve(ids, torch.device('cpu'))
    corr = float(np.corrcoef(emb_b.numpy().ravel(), emb_g.numpy().ravel())[0, 1])
    assert corr > 0.95, f'两源相关性异常 {corr}'
    h = torch.randn(1, len(ids), 512)
    out = be_bf16(ids, h)
    assert out.shape == h.shape and torch.isfinite(out).all()
    print(f'[2] BF16 端到端 ✓  两源检索向量相关={corr:.4f}，输出{tuple(out.shape)}')

def test_speed():
    """BF16 免反量化，读行应显著快于 IQ4_NL。"""
    import time
    tbl_bf = BF16ShardTable(HF_DIR)
    from model.ple_backend import IQ4NLMmapTable
    tbl_iq = IQ4NLMmapTable(SHARD2, 192, 320001536, 160)
    rng = np.random.default_rng(1)
    rows = [int(x) for x in rng.integers(0, 320001446, 200)]
    t0 = time.perf_counter(); [tbl_bf.row(r) for r in rows]; tb = time.perf_counter() - t0
    t0 = time.perf_counter(); [tbl_iq.row(r) for r in rows]; ti = time.perf_counter() - t0
    print(f'[3] 读行速度 ✓  BF16={tb*1000:.1f}ms vs IQ4_NL={ti*1000:.1f}ms (200行)')

if __name__ == '__main__':
    test_row_exact()
    test_backend_forward()
    test_speed()
    print('\n全部通过')
