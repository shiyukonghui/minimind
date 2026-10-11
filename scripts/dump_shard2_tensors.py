# 直接打印分片2 张量信息段的原始解析结果（含 dtype 枚举原值），核实表量化类型
import struct, sys

path = sys.argv[1] if len(sys.argv) > 1 else r'H:\Strata-data\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf'
f = open(path, 'rb')
assert f.read(4) == b'GGUF'
ver = struct.unpack('<I', f.read(4))[0]
nt = struct.unpack('<Q', f.read(8))[0]
nkv = struct.unpack('<Q', f.read(8))[0]
print(f'ver={ver} nt={nt} nkv={nkv}')

def rstr():
    l = struct.unpack('<Q', f.read(8))[0]
    assert l < 10**7
    return f.read(l).decode('utf8', 'replace')

FMT = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'d',16:'e'}
SZ  = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,16:2}

def rval(t):
    if t == 8: return rstr()
    if t == 9:
        et = struct.unpack('<I', f.read(4))[0]
        n = struct.unpack('<Q', f.read(8))[0]
        if et == 8:
            for _ in range(n): rstr()
        elif et == 9:
            for _ in range(n): rval(et)
        else:
            f.read(n * SZ[et])
        return None
    if t == 7:
        f.read(1); return None
    f.read(SZ[t]); return None

for _ in range(nkv):
    k = rstr(); t = struct.unpack('<I', f.read(4))[0]
    if k == 'general.alignment':
        alignment = struct.unpack('<I', f.read(4))[0]
    else:
        rval(t)
print('kv end @', f.tell())

# 张量段紧跟 KV 段（GGUF v3 规范，无填充）
for i in range(min(nt, 10)):
    name = rstr()
    nd = struct.unpack('<I', f.read(4))[0]
    dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(nd)]
    dt = struct.unpack('<I', f.read(4))[0]
    off = struct.unpack('<Q', f.read(8))[0]
    print(f'[{i}] {name} dims={dims} dtype_enum={dt} off={off}')
