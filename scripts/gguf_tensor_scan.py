# GGUF 张量清单扫描：列出 ple/ngram/engram/embed 相关张量的名称、形状与量化类型
# 用途：核实 Qwen3.8-Flash GGUF 中 Engram 类模块（元数据键前缀 ple.）的实际表结构与容量
import struct, sys, re
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else r'H:\Strata-data\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf'
f = open(path, 'rb')
assert f.read(4) == b'GGUF'
ver = struct.unpack('<I', f.read(4))[0]
nt = struct.unpack('<Q', f.read(8))[0]
nkv = struct.unpack('<Q', f.read(8))[0]

def rstr():
    l = struct.unpack('<Q', f.read(8))[0]
    assert l < 10**7, f'bad str len {l} at {f.tell()}'
    return f.read(l).decode('utf8', 'replace')

FMT = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'd', 16: 'e'}
SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 16: 2}

alignment = 32

def rval(t):
    if t == 8: return rstr()
    if t == 9:
        et = struct.unpack('<I', f.read(4))[0]; n = struct.unpack('<Q', f.read(8))[0]
        if et == 8:
            for _ in range(n): rstr()
        elif et == 9:  # 嵌套数组，逐元素递归
            for _ in range(n): rval(et)
        else:
            f.read(n * SZ[et])
        return None
    if t == 7:  # bool 占 1 字节
        f.read(1); return None
    f.read(SZ[t]); return None

# 单遍解析 KV 段，顺带取出 general.alignment
for _ in range(nkv):
    k = rstr(); t = struct.unpack('<I', f.read(4))[0]
    if k == 'general.alignment':
        alignment = struct.unpack('<I', f.read(4))[0]
    else:
        rval(t)

# 张量信息段按 alignment 对齐；若对齐后首个名字长度异常，则向后搜索合法起点（兼容部分导出工具的不规范填充）
f.read((-f.tell()) % alignment)
def _valid_name_start(pos):
    f.seek(pos)
    l = struct.unpack('<Q', f.read(8))[0]
    return 0 < l < 200 and f.read(l).decode('utf8', 'replace').isprintable()
if not _valid_name_start(f.tell()):
    f.seek(0); blob = f.read(11_000_000)
    m = re.search(rb'\x0d\x00\x00\x00\x00\x00\x00\x00output\.weight', blob)
    assert m, '张量段起点未找到'
    f.seek(m.start())

DT = {0:'F32',1:'F16',2:'Q4_0',3:'Q4_1',7:'Q8_0',8:'MIXED',10:'Q5_0',11:'Q5_1',12:'Q2_K',13:'Q3_K_S',14:'Q3_K_M',15:'Q3_K_L',16:'Q4_K_S',17:'Q4_K_M',18:'Q5_K_S',19:'Q5_K_M',20:'Q6_K',21:'IQ2_XXS',22:'IQ2_XS',23:'Q2_S',25:'Q3_0',26:'IQ1_S',27:'IQ4_NL',28:'IQ3_S',29:'IQ2_S',30:'IQ4_XS',31:'IQ1_M',32:'BF16'}

tensors = []
for i in range(nt):
    name = rstr()
    nd = struct.unpack('<I', f.read(4))[0]
    dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(nd)]
    dt = struct.unpack('<I', f.read(4))[0]
    off = struct.unpack('<Q', f.read(8))[0]
    tensors.append((name, dims, dt))

pat = re.compile(r'ple|ngram|enram|engram|memory|hash', re.I)
hits = [t for t in tensors if pat.search(t[0])]
print(f'总张量 {len(tensors)}，匹配 ple/ngram/engram 的张量 {len(hits)}')
for name, dims, dt in hits[:60]:
    print(f'  {name}  shape={dims}  type={DT.get(dt, dt)}')

# 名称模式统计，帮助理解整体结构
prefixes = Counter(re.sub(r'\d+', 'N', t[0]).rsplit('.', 1)[0] for t in tensors)
print('\n张量名称前缀分布（top 25）:')
for p, c in prefixes.most_common(25):
    print(f'  {c:5d}  {p}')
