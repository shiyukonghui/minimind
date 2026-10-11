# MemoryBackend 统一接口冒烟测试：三后端契约一致性 + 形状 + 可训练性
# 用法：python scripts/test_memory_backend.py
import sys
import torch

sys.path.insert(0, r'F:\MachineLearn\minimind')
from model.memory_backend import MemoryBackend, HashTableMemory, ExternalKVMemory, QwenPLEAdapter
from model.ple_backend import QwenPLEBackend

SHARD2 = r'H:\Strata-data\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf'
H = 512

def check_contract(mb, keys, hidden, name):
    assert isinstance(mb, MemoryBackend), name
    v = mb.retrieve(keys)
    g = mb.gate(hidden)
    assert v.shape[-1] == mb.d_c, (name, v.shape, mb.d_c)
    assert g.shape[-1] == 1 and g.min() >= 0 and g.max() <= 1, (name, g.shape)
    out = mb(keys, hidden)
    assert out.shape == hidden.shape, (name, out.shape)
    assert torch.isfinite(out).all(), name
    print(f'  [{name}] retrieve{tuple(v.shape)} gate{tuple(g.shape)} out{tuple(out.shape)} ✓')

def test_hash():
    mb = HashTableMemory(d_c=64, host_hidden=H, num_slots=1 << 16)
    bytes_ids = torch.tensor(list('深度学习'.encode('utf-8')), dtype=torch.long)
    hidden = torch.randn(len(bytes_ids), H)
    check_contract(mb, bytes_ids, hidden, 'HashTable')
    # 哈希确定性：同输入同槽
    r1 = mb.hash_rows(bytes_ids)
    r2 = mb.hash_rows(bytes_ids)
    assert torch.equal(r1, r2)
    assert r1.shape == (len(bytes_ids), 2)   # ngram=3 -> 2 阶(2,3)
    print('  HashTable 哈希确定性 ✓  rows', tuple(r1.shape))

def test_extkv():
    mb = ExternalKVMemory(d_c=64, host_hidden=H, capacity=128)
    # 空表检索返回零
    q = torch.randn(5, 64)
    assert mb.retrieve(q).abs().sum() == 0
    # 写入后最近邻命中
    for i in range(10):
        mb.write(torch.randn(64), torch.full((64,), float(i)))
    probe = mb.keys[3:4].clone()
    hit = mb.retrieve(probe)
    assert torch.allclose(hit[0], mb.vals[3], atol=1e-5), '最近邻应命中刚写入项'
    hidden = torch.randn(5, H)
    check_contract(mb, q, hidden, 'ExternalKV')
    print('  ExternalKV 热写+检索 ✓  size=', int(mb.size.item()))

def test_qwen_adapter():
    ple = QwenPLEBackend(SHARD2, host_hidden=H)
    ad = QwenPLEAdapter(ple, host_hidden=H)
    assert ad.d_c == 2560
    ids = [100, 200, 300, 400]
    hidden = torch.randn(1, len(ids), H)
    out = ad(ids, hidden)                    # 委托 ple.forward
    assert out.shape == hidden.shape and torch.isfinite(out).all()
    print(f'  QwenPLEAdapter 委托前向 ✓  out{tuple(out.shape)}')

if __name__ == '__main__':
    torch.manual_seed(0)
    print('[HashTableMemory]'); test_hash()
    print('[ExternalKVMemory]'); test_extkv()
    print('[QwenPLEAdapter]'); test_qwen_adapter()
    print('\n全部通过')
