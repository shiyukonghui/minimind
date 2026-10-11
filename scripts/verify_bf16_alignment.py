# 交叉验证：HF BF16 分片拼接行序 与 GGUF IQ4_NL 列索引是否对齐
# 方法：取若干全局行号 r，分别从 GGUF(IQ4_NL 反量化) 与 HF(BF16 直读) 取该行向量，
#       计算皮尔逊相关系数。对齐 => 相关 ~0.99 以上；错位 => ~0。
# 用法：python scripts/verify_bf16_alignment.py
import json, struct, sys
import numpy as np
from safetensors import safe_open

sys.path.insert(0, r'F:\MachineLearn\minimind')
from model.ple_backend import IQ4NLMmapTable, dequant_iq4_nl

HF_DIR = r'H:\Strata-data\models\Qwen3.8-Flash-Next-BF16'
SHARD2 = r'H:\Strata-data\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf'
ROWS_PER_SHARD = 2500012
HEAD_DIM = 160

# ---- 1. 从 HF 索引建立 全局行号 -> (分片文件, 片内行号) 映射 ----
idx = json.load(open(HF_DIR + r'\model.safetensors.index.json', encoding='utf-8'))
wm = idx['weight_map']
shard_files = sorted({v for k, v in wm.items() if 'ngram_embedding.shard_' in k})
# 分片内 shard_i 的键名 -> 需要知道每个文件里有哪些 shard 段以及顺序
# 按经验：shard 编号全局递增按文件排序；用文件内 keys 排序确认
shard_ranges = []   # (file, key, start_row, end_row)
for f in shard_files:
    with safe_open(HF_DIR + '\\' + f, framework='pt') as sf:
        keys = sorted([k for k in sf.keys() if 'ngram_embedding.shard_' in k],
                      key=lambda k: int(k.split('shard_')[1].split('.')[0]))
        for k in keys:
            n = int(k.split('shard_')[1].split('.')[0])
            shard_ranges.append((f, k, n * ROWS_PER_SHARD, (n + 1) * ROWS_PER_SHARD))
shard_ranges.sort(key=lambda t: t[2])
print(f'共 {len(shard_ranges)} 段，覆盖行 0..{shard_ranges[-1][3]:,}')

def hf_row(r: int) -> np.ndarray:
    """按全局行号从 HF BF16 分片读取 160 维向量。"""
    for f, k, s, e in shard_ranges:
        if s <= r < e:
            with safe_open(HF_DIR + '\\' + f, framework='pt') as sf:
                t = sf.get_tensor(k)                    # [2500012,160] BF16
                return t[r - s].float().numpy()
    raise KeyError(r)

# ---- 2. GGUF IQ4_NL 行 ----
gg = IQ4NLMmapTable(SHARD2, data_start=192, n_rows=320001536, head_dim=HEAD_DIM)

# ---- 3. 对比：均匀采样 + 热点区(前几行) ----
rng = np.random.default_rng(0)
test_rows = [0, 1, 2, 100, 7777] + [int(x) for x in rng.integers(0, 320001446, 10)]
corrs = []
for r in test_rows:
    a = gg.row(r)
    b = hf_row(r)
    c = float(np.corrcoef(a, b)[0, 1])
    corrs.append(c)
    print(f'row {r:>12,}  corr={c:+.4f}  |a|={np.linalg.norm(a):.3f} |b|={np.linalg.norm(b):.3f}')

med = np.median(corrs)
print(f'\n中位相关 = {med:.4f}')
if med > 0.95:
    print('✓ 对齐：HF 拼接行序 == GGUF 列索引，BF16 源可直接复用同一哈希')
elif med < 0.1:
    print('✗ 错位：需要找到真实置换（可能按 head 交错存储）')
else:
    print('? 部分相关：检查转置或维度错位')
