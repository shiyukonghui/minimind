# 从 Qwen3.8-Flash GGUF 提取 PLE 模块的完整元数据：
#   1) KV 段中的 ple.* 超参（multipliers / head_offsets / head_vocab_sizes 等实际数值）
#   2) 张量段中 per_layer_token_embd / ple_* 的名称、形状、量化类型、数据偏移
# 多分片 GGUF：门控小张量在分片1，N-gram 大表在分片2，逐个文件扫描后合并。
# 输出 JSON 供 QwenPLEBackend 直接加载，避免运行时重复解析 GGUF。
# 用法：python scripts/extract_ple_meta.py [输出json] [gguf1] [gguf2] ...
import struct, sys, json, re

OUT = sys.argv[1] if len(sys.argv) > 1 else r'F:\MachineLearn\minimind\model\ple_meta_qwen38.json'
GGUF_FILES = sys.argv[2:] or [r'H:\Strata-data\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf',
                              r'H:\Strata-data\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf']

FMT = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'d',16:'e'}
SZ  = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,16:2}
# ggml 权威枚举（与 ggml.h 对齐）：20=IQ4_NL, 21=IQ3_S, 23=IQ4_XS, 30=BF16
DT = {0:'F32',1:'F16',2:'Q4_0',3:'Q4_1',6:'Q5_0',7:'Q5_1',8:'Q8_0',9:'Q8_1',
      10:'Q2_K',11:'Q3_K',12:'Q4_K',13:'Q5_K',14:'Q6_K',15:'Q8_K',
      16:'IQ2_XXS',17:'IQ2_XS',18:'IQ3_XXS',19:'IQ1_S',20:'IQ4_NL',21:'IQ3_S',
      22:'IQ2_S',23:'IQ4_XS',24:'I8',25:'I16',26:'I32',27:'I64',28:'F64',29:'IQ1_M',30:'BF16'}
INT_ARRAY_KEYS = ('layer_multipliers', 'head_offsets', 'head_vocab_sizes')  # 按原始整数读

def scan(path):
    """解析单个 GGUF 文件，返回 (kv超参dict, ple相关张量列表, alignment)"""
    f = open(path, 'rb')
    assert f.read(4) == b'GGUF', path
    f.read(4)  # version
    nt = struct.unpack('<Q', f.read(8))[0]
    nkv = struct.unpack('<Q', f.read(8))[0]

    def rstr():
        l = struct.unpack('<Q', f.read(8))[0]
        assert l < 10**7, f'bad str len {l} @ {f.tell()}'
        return f.read(l).decode('utf8', 'replace')

    def rval(t, raw=False):
        if t == 8: return rstr()
        if t == 9:
            et = struct.unpack('<I', f.read(4))[0]
            n = struct.unpack('<Q', f.read(8))[0]
            if et == 8: return [rstr() for _ in range(n)]
            if et == 9: return [('nested_arr', n)]
            blob = f.read(n * SZ[et])
            if raw and et == 10:  # f64 位模式 -> 原始 u64
                return list(struct.unpack('<' + 'Q'*n, blob))
            return list(struct.unpack('<' + FMT[et]*n, blob))
        if t == 7:
            f.read(1); return None
        return struct.unpack('<' + FMT[t], f.read(SZ[t]))[0]

    meta, alignment = {}, 32
    for _ in range(nkv):
        k = rstr(); t = struct.unpack('<I', f.read(4))[0]
        v = rval(t, raw=any(s in k for s in INT_ARRAY_KEYS))
        if k == 'general.alignment':
            alignment = v
        if k.startswith('qwen4exp.ple') or k in ('general.architecture', 'qwen4exp.embedding_length', 'qwen4exp.block_count'):
            meta[k] = v

    # 按 GGUF 规范，张量信息段紧跟 KV 段、无 alignment 填充（只有数据段才对齐）。
    # 先直接按 KV 结束位置解析；若首名字非法，再尝试对齐后的位置与向后搜索。
    kv_end = f.tell()

    def _name_ok(pos):
        f.seek(pos)
        l = struct.unpack('<Q', f.read(8))[0]
        return 0 < l < 200 and f.read(l).decode('utf8', 'replace').isprintable()

    start = None
    for cand in (kv_end, kv_end + (-kv_end % alignment)):
        if _name_ok(cand):
            start = cand
            break
    if start is None:
        # 兼容不规范导出：搜索已知张量名，回退 8 字节补长度前缀
        f.seek(0); blob = f.read(11_000_000)
        m = re.search(rb'per_layer_token_embd\.weight', blob)
        assert m and m.start() >= 8, f'张量段起点未找到: {path}'
        start = m.start() - 8
    f.seek(start)

    tensors = []
    for _ in range(nt):
        name = rstr()
        nd = struct.unpack('<I', f.read(4))[0]
        dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(nd)]
        dt = struct.unpack('<I', f.read(4))[0]
        off = struct.unpack('<Q', f.read(8))[0]
        if 'per_layer_token_embd' in name or 'ple' in name.lower():
            tensors.append({'name': name, 'shape': dims, 'dtype': DT.get(dt, dt),
                            'offset': off, 'file': path})
    f.close()
    return meta, tensors, alignment

all_tensors, kv_meta, align = [], {}, 32
for p in GGUF_FILES:
    m, ts, a = scan(p)
    if m: kv_meta.update(m)
    align = a
    all_tensors += ts
    print(f'{p}: 张量 {len(ts)} 个')

out = {'files': GGUF_FILES, 'alignment': align, 'kv_meta': kv_meta, 'tensors': all_tensors}
json.dump(out, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(f'写入 {OUT}')
for k, v in kv_meta.items():
    s = str(v)
    print(' ', k, '=', s[:100] + ('...' if len(s) > 100 else ''))
for t in all_tensors:
    print(' ', t['name'], t['shape'], t['dtype'], 'off=', t['offset'], 'file=', t['file'][-14:])
