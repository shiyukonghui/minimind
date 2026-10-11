# 交叉验证 PLE 元数据一致性：
#   1) head_offsets[i+1] == head_offsets[i] + head_vocab_sizes[i]（分片布局自洽）
#   2) 各 head 容量之和 == 表总行数（[160, N] 的 N）
#   3) 表 dtype 的每块字节/元素 × 总行数 ≈ 分片2 中表实际占用字节
import json, os, struct

m = json.load(open(r"F:\MachineLearn\minimind\model\ple_meta_qwen38.json", encoding="utf-8"))
kv = m["kv_meta"]
offs = kv["qwen4exp.ple.head_offsets"]
vsz = kv["qwen4exp.ple.head_vocab_sizes"]
n_heads = len(offs)
print(f"head 数 = {n_heads}")

# 1) 偏移自洽
bad = [i for i in range(n_heads - 1) if offs[i] + vsz[i] != offs[i + 1]]
print("偏移不自洽处:", bad if bad else "无")

total_rows = offs[-1] + vsz[-1]
print(f"各 head 容量之和 = {total_rows}")

# 2) 与表张量形状对照（表行数会 pad 到量化块边界）
tbl = next(t for t in m["tensors"] if "per_layer_token_embd" in t["name"])
print(f"表张量 shape = {tbl['shape']} dtype = {tbl['dtype']}")
assert tbl["shape"][1] >= total_rows, "表行数小于 head 容量之和"
print(f"✓ 表行数 {tbl['shape'][1]} >= 容量之和 {total_rows}，padding={tbl['shape'][1]-total_rows} 行")

# 3) 估算表字节占用（Q6_K: 每 256 元素一块，块 210 字节 -> 0.8203 B/元素）
shard2 = tbl["file"]
fsize = os.path.getsize(shard2)
q6k_bytes_per_elem = 210 / 256
est = total_rows * tbl["shape"][0] * q6k_bytes_per_elem
print(f"分片2 文件 = {fsize/1e9:.2f}GB, Q6_K 估算表 = {est/1e9:.2f}GB")
