# GGUF 元数据探针：逐 KV 解析并打印偏移，定位解析错位点
import struct, sys

path = sys.argv[1] if len(sys.argv) > 1 else r'H:\Strata-data\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf'
f = open(path, 'rb')
magic = f.read(4); ver = struct.unpack('<I', f.read(4))[0]
nt = struct.unpack('<Q', f.read(8))[0]; nkv = struct.unpack('<Q', f.read(8))[0]
print('magic', magic, 'ver', ver, 'n_tensors', nt, 'n_kv', nkv)

def tell(): return f.tell()

def rstr():
    off = tell(); l = struct.unpack('<Q', f.read(8))[0]
    if l > 10**7:
        raise RuntimeError(f'bad str len {l} at offset {off}')
    return f.read(l).decode('utf8', 'replace')

FMT = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'd', 16: 'e'}
SZ  = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 16: 2}

def rval(t, depth=0):
    if t == 8: return rstr()
    if t == 9:  # array
        et = struct.unpack('<I', f.read(4))[0]; n = struct.unpack('<Q', f.read(8))[0]
        if et == 8:
            return [rstr() for _ in range(n)]
        if et == 9 and depth < 2:  # nested array
            out = []
            for _ in range(min(n, 4)):
                et2 = struct.unpack('<I', f.read(4))[0]; n2 = struct.unpack('<Q', f.read(8))[0]
                out.append(('arr', et2, n2))
                if et2 in SZ: f.read(n2 * SZ[et2])
                else: raise RuntimeError(f'bad nested elem {et2}')
            if n > 4: f.read((n - 4) * 0)  # 无法安全跳变长，交给异常
            return out
        if et in SZ:
            blob = f.read(n * SZ[et]); return ('arr', et, n, blob[:16])
        raise RuntimeError(f'bad elem type {et}')
    if t in SZ:
        return struct.unpack('<' + FMT[t], f.read(SZ[t]))[0]
    raise RuntimeError(f'bad kv type {t} at offset {tell()}')

for i in range(nkv):
    off = tell()
    try:
        k = rstr(); t = struct.unpack('<I', f.read(4))[0]; v = rval(t)
        if isinstance(v, list) and v and isinstance(v[0], str): v = v[:2]
        print(f'[{i:02d}] off={off} {k} = {v}')
    except Exception as e:
        print(f'[{i:02d}] off={off} ERROR: {e}')
        break
