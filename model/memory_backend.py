# MemoryBackend 标准接口（设计文档 §3.5，路线图任务⑤）
#
# 统一"检索 + 门控"两方法契约，使记忆源可插拔切换：
#   - HashTableMemory：自训字节 N-gram 哈希表（字节化路线，键=字节 3-gram 哈希）；
#   - QwenPLEAdapter：包装已实现的 QwenPLEBackend，桥接到统一接口；
#   - ExternalKVMemory：RSI 热写的外部 KV 记忆（§3.7 低风险写入路径）。
#
# 契约：
#   retrieve(keys) -> value:[*, d_c]   # 规范值空间向量(冻结或可训由子类决定)
#   gate(hidden)   -> g:[*, 1]         # 宿主侧门控标量(0..1)
#   宿主侧统一用 W_proj(d_c -> hidden) 把记忆向量投影回模型维度(可训练)。
import math
import torch
import torch.nn.functional as F
from torch import nn
from abc import ABC, abstractmethod

from model.ple_backend import QwenPLEBackend


class MemoryBackend(ABC, nn.Module):
    """所有记忆后端的抽象基类：定义 retrieve/gate 契约 + 统一宿主投影。"""

    def __init__(self, d_c: int, host_hidden: int, proj_rank: int = 48):
        super().__init__()
        self.d_c = d_c
        self.host_hidden = host_hidden
        # 规范值空间 -> 宿主维度 的低秩投影（所有后端共享的宿主侧适配）
        self.w_proj = nn.Sequential(
            nn.Linear(d_c, proj_rank, bias=False),
            nn.GELU(),
            nn.Linear(proj_rank, host_hidden, bias=False),
        )

    @abstractmethod
    def retrieve(self, keys: torch.Tensor) -> torch.Tensor:
        """keys -> 规范记忆向量 [*, d_c]。"""
        raise NotImplementedError

    @abstractmethod
    def gate(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden -> 门控标量 [*, 1]。"""
        raise NotImplementedError

    def forward(self, keys: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """默认融合：hidden + gate ⊙ W_proj(retrieve(keys))，残差后 LayerNorm。"""
        value = self.w_proj(self.retrieve(keys))          # [*, hidden]
        g = self.gate(hidden)                             # [*, 1]
        return hidden + g * value


class HashTableMemory(MemoryBackend):
    """自训字节 N-gram 哈希记忆表（§3.6 字节化路线的落地后端）。

    键 = 字节 3-gram 经 XOR 哈希到 num_slots 槽；值 = 可训练 embedding(num_slots, d_c)。
    与 Qwen PLE 同构但键空间为字节、表小而可全量训练。
    """

    def __init__(self, d_c: int, host_hidden: int, num_slots: int = 1 << 20,
                 ngram: int = 3, proj_rank: int = 48, gate_bias: float = -2.0):
        super().__init__(d_c, host_hidden, proj_rank)
        self.num_slots, self.ngram = num_slots, ngram
        # 每个 n-gram 阶一个乘数（XOR 混合），取 int64 安全范围内的大质数
        self.register_buffer('mult', torch.tensor(
            [2654435761, 2246822519, 3266489917][:ngram], dtype=torch.int64))
        self.table = nn.Embedding(num_slots, d_c)
        nn.init.normal_(self.table.weight, mean=0.0, std=0.02)
        self.gate_head = nn.Linear(host_hidden, 1)
        nn.init.constant_(self.gate_head.bias, gate_bias)   # 初始门控偏低，避免随机表污染底座

    def hash_rows(self, byte_ids: torch.Tensor) -> torch.Tensor:
        """byte_ids:[T] 字节序列 -> 每位置各 n-gram 阶(2..ngram)的槽行号 [T, ngram-1]。

        与 Qwen PLE 同构：mixed_n = byte[i]·m0 ^ byte[i-1]·m1 ^ ...，左侧补零(等价 EOS 起点)。
        """
        T = byte_ids.shape[-1]
        off = self.ngram - 1                                   # 左侧补零个数
        pad = F.pad(byte_ids.long(), (off, 0), value=0)        # [T+off]
        rows = []
        for n in range(2, self.ngram + 1):
            mixed = torch.zeros(T, dtype=torch.int64, device=byte_ids.device)
            for j in range(n):
                # 输出位置 i 对应 pad[i+off-j]，切片起点 off-j、长度 T
                seg = pad[off - j : off - j + T]
                mixed = mixed ^ (seg * self.mult[j].to(byte_ids.device))
            rows.append((mixed & 0x7FFFFFFFFFFFFFFF) % self.num_slots)   # 取非负并映射到槽
        return torch.stack(rows, dim=-1)                       # [T, ngram-1]

    def retrieve(self, keys: torch.Tensor) -> torch.Tensor:
        """keys:[T] 字节 id -> [T, d_c]（各 n-gram 阶查表后求和）。"""
        rows = self.hash_rows(keys)                               # [T, ngram]
        vals = self.table(rows)                                   # [T, ngram, d_c]
        return vals.sum(dim=1)                                    # [T, d_c]

    def gate(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate_head(hidden))              # [*, 1]


class QwenPLEAdapter(MemoryBackend):
    """把 QwenPLEBackend(token 键, IQ4_NL 大表) 适配到统一 MemoryBackend 接口。

    d_c=2560(16头×160)。retrieve 需外部提供 Qwen token id（键空间桥接在调用方完成）；
    门控与投影已在 QwenPLEBackend 内部实现，forward 直接委托，避免重复参数。
    """

    def __init__(self, ple: QwenPLEBackend, host_hidden: int, proj_rank: int = 48):
        super().__init__(ple.embed_dim, host_hidden, proj_rank)
        self.ple = ple                                            # 冻结大表 + 已训宿主侧

    def retrieve(self, keys: torch.Tensor) -> torch.Tensor:
        """keys 为 Qwen token id（张量或列表）。"""
        ids = keys.tolist() if torch.is_tensor(keys) else list(keys)
        return self.ple.retrieve(ids, torch.device('cpu'))        # [T, 2560]

    def gate(self, hidden: torch.Tensor) -> torch.Tensor:
        # QwenPLEBackend 的门控内嵌于其 forward，这里返回中性值(1)表示委托模式
        return torch.ones(*hidden.shape[:-1], 1, device=hidden.device, dtype=hidden.dtype)

    def forward(self, qwen_ids, hidden):
        """直接复用 QwenPLEBackend 的完整前向（含其内部门控与低秩投影）。"""
        return self.ple.forward(qwen_ids, hidden)


class ExternalKVMemory(MemoryBackend):
    """RSI 热写外部 KV 记忆（§3.7 低风险写入路径）。

    键 = 任意可哈希上下文签名；值 = 热写入的 d_c 向量，存于动态字典，检索用最近邻/精确匹配。
    训练时仅宿主投影与门控可学，写入内容来自模型自身(自举)。
    """

    def __init__(self, d_c: int, host_hidden: int, capacity: int = 4096, proj_rank: int = 48):
        super().__init__(d_c, host_hidden, proj_rank)
        self.capacity = capacity
        # 用固定大小 buffer 池 + 环形指针模拟可增长 KV 存储(便于随模型上 GPU/存 checkpoint)
        self.register_buffer('keys', torch.zeros(capacity, d_c))
        self.register_buffer('vals', torch.zeros(capacity, d_c))
        self.register_buffer('size', torch.zeros(1, dtype=torch.long))
        self.gate_head = nn.Linear(host_hidden, 1)

    @torch.no_grad()
    def write(self, key: torch.Tensor, value: torch.Tensor) -> int:
        """热写一条记忆(环形覆盖最旧)，返回写入下标。"""
        idx = int(self.size.item()) % self.capacity
        self.keys[idx] = key
        self.vals[idx] = value
        self.size += 1
        return idx

    def retrieve(self, keys: torch.Tensor) -> torch.Tensor:
        """keys:[T,d_c] 查询签名 -> 最近邻命中值 [T,d_c]（余弦相似度 top-1）。"""
        n = min(int(self.size.item()), self.capacity)
        if n == 0:
            return torch.zeros(keys.shape[0], self.d_c, device=keys.device)
        K, V = self.keys[:n], self.vals[:n]
        sim = F.normalize(keys, dim=-1) @ F.normalize(K, dim=-1).T   # [T,n]
        best = sim.argmax(dim=-1)
        return V[best]

    def gate(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate_head(hidden))
