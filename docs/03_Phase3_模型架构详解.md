# Phase 3: 模型架构详解

> 本阶段目标：深入理解 MiniMind 的模型架构，包括 Transformer Decoder、RoPE、GQA、MoE

## 一、整体架构概览

MiniMind 采用 **Transformer Decoder-Only** 架构，与 Llama3.1 一致，相比 GPT-3 有以下改进：

| 组件 | GPT-3 | MiniMind / Llama3 |
|------|-------|-------------------|
| 归一化位置 | 子层输出（Post-LN） | 子层输入（Pre-LN / RMSNorm） |
| 激活函数 | ReLU | SwiGLU |
| 位置编码 | 可学习绝对位置编码 | RoPE 旋转位置编码 |
| 注意力机制 | MHA（多头注意力） | GQA（分组查询注意力） |
| 前馈网络 | 标准 FFN | SwiGLU FFN / MoE |

### 模型参数配置

| 模型 | d_model | n_layers | n_heads | kv_heads | 参数量 |
|------|---------|----------|---------|----------|-------|
| MiniMind2-Small | 512 | 8 | 8 | 2 | 26M |
| MiniMind2-MoE | 640 | 8 | 8 | 2 | 145M |
| MiniMind2 | 768 | 16 | 8 | 2 | 104M |

## 二、核心组件详解

### 2.1 MiniMindConfig — 模型配置

```python
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(self,
        hidden_size=512,           # 隐藏层维度 d_model
        num_hidden_layers=8,       # Transformer 层数
        num_attention_heads=8,     # 查询头数
        num_key_value_heads=2,     # KV 头数（GQA）
        vocab_size=6400,           # 词表大小
        max_position_embeddings=32768,  # 最大序列长度
        rope_theta=1000000.0,      # RoPE 基础频率
        rms_norm_eps=1e-5,         # RMSNorm epsilon
        hidden_act='silu',         # 激活函数
        # MoE 配置
        use_moe=False,             # 是否使用 MoE
        num_experts_per_tok=2,     # 每个 token 选择的专家数
        n_routed_experts=4,        # 可路由专家总数
        n_shared_experts=1,        # 共享专家数
        ...
    ):
```

### 2.2 RMSNorm — 均方根归一化

RMSNorm 是 LayerNorm 的简化版本，计算更高效：

```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习缩放参数

    def _norm(self, x):
        # 均方根归一化：x / sqrt(mean(x^2) + eps)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 先归一化，再缩放；用 float32 计算保证精度
        return self.weight * self._norm(x.float()).type_as(x)
```

**与 LayerNorm 的区别**：
- RMSNorm 不需要计算均值（减去均值步骤），只做缩放归一化
- 计算量更小，推理速度更快
- 实验表明效果与 LayerNorm 相当

### 2.3 RoPE — 旋转位置编码

RoPE（Rotary Position Embedding）通过旋转矩阵为 Q 和 K 注入位置信息：

```python
def precompute_freqs_cis(dim, end=32768, rope_base=1e6, rope_scaling=None):
    # 计算频率：freqs_i = 1 / (theta^(2i/d))
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # YaRN 长度外推（可选）
    if rope_scaling is not None:
        # ... YaRN 缩放逻辑 ...

    # 计算外积得到每个位置的频率
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    # 拼接 cos 和 sin
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin
```

**应用 RoPE**：

```python
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        # 将向量旋转一半：[-x后半, x前半]
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    # q_embed = q * cos + rotate_half(q) * sin
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed
```

**RoPE 的优势**：
- 相对位置编码：内积只依赖相对位置差
- 长度外推：通过 YaRN 算法可扩展到更长序列
- 计算高效：只需逐元素乘加

### 2.4 GQA — 分组查询注意力

GQA（Grouped-Query Attention）是 MHA 和 MQA 的折中方案：

```
MHA (Multi-Head Attention):  Q: 8头, K: 8头, V: 8头  → 每个Q头独立对应KV头
GQA (Grouped-Query Attention): Q: 8头, K: 2头, V: 2头  → 4个Q头共享1组KV头
MQA (Multi-Query Attention):  Q: 8头, K: 1头, V: 1头  → 所有Q头共享1组KV头
```

MiniMind 使用 `n_heads=8, kv_heads=2`，即每 4 个查询头共享 1 组 KV 头。

```python
class Attention(nn.Module):
    def __init__(self, args):
        self.n_local_heads = args.num_attention_heads      # 8
        self.n_local_kv_heads = args.num_key_value_heads   # 2
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 4
        self.head_dim = args.hidden_size // args.num_attention_heads  # 64

        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, args.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, args.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
```

**KV 复制函数**：

```python
def repeat_kv(x, n_rep):
    """将 KV 头复制 n_rep 次以匹配 Q 头数"""
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(bs, slen, num_kv_heads, n_rep, head_dim).reshape(...)
```

**KV Cache**：

```python
# 推理时缓存 KV 以加速自回归生成
if past_key_value is not None:
    xk = torch.cat([past_key_value[0], xk], dim=1)
    xv = torch.cat([past_key_value[1], xv], dim=1)
past_kv = (xk, xv) if use_cache else None
```

**Flash Attention**：

```python
if self.flash and seq_len > 1:
    # 使用 PyTorch 2.0+ 的 Flash Attention 加速
    output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask, is_causal=is_causal)
else:
    # 手动计算注意力（兼容模式）
    scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
    scores = scores + causal_mask  # 因果掩码
    scores = F.softmax(scores.float(), dim=-1)
    output = scores @ xv
```

### 2.5 SwiGLU FFN — 前馈网络

SwiGLU 是 GLU（Gated Linear Unit）的变体，用 SiLU 激活函数替代 ReLU：

```python
class FeedForward(nn.Module):
    def __init__(self, config):
        # intermediate_size 自动对齐到 64 的倍数
        intermediate_size = int(config.hidden_size * 8 / 3)
        config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)

        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]  # SiLU

    def forward(self, x):
        # SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))
```

**计算流程**：

```
x → gate_proj → SiLU ──┐
                        ├→ 逐元素乘 → down_proj → 输出
x → up_proj ────────────┘
```

### 2.6 MoE — 混合专家模块

MoE（Mixture of Experts）通过门控机制为每个 token 选择部分专家处理：

```python
class MOEFeedForward(nn.Module):
    def __init__(self, config):
        # 多个路由专家
        self.experts = nn.ModuleList([FeedForward(config) for _ in range(config.n_routed_experts)])
        # 门控网络
        self.gate = MoEGate(config)
        # 共享专家（始终参与计算）
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([FeedForward(config) for _ in range(config.n_shared_experts)])
```

**门控网络**：

```python
class MoEGate(nn.Module):
    def forward(self, hidden_states):
        # 计算每个专家的分数
        logits = F.linear(hidden_states, self.weight, None)
        scores = logits.softmax(dim=-1)

        # 选择 Top-K 个专家
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1)

        # 归一化权重
        if self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)

        # 计算辅助损失（负载均衡）
        aux_loss = ...

        return topk_idx, topk_weight, aux_loss
```

**MoE 前向传播**：

```python
def forward(self, x):
    identity = x
    # 门控选择专家
    topk_idx, topk_weight, aux_loss = self.gate(x)

    if self.training:
        # 训练：逐专家处理
        for i, expert in enumerate(self.experts):
            y[flat_topk_idx == i] = expert(x[flat_topk_idx == i])
        # 加权求和
        y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
    else:
        # 推理：批量处理优化
        y = self.moe_infer(x, flat_topk_idx, flat_expert_weights)

    # 加上共享专家的输出
    if self.n_shared_experts > 0:
        for expert in self.shared_experts:
            y = y + expert(identity)

    self.aux_loss = aux_loss
    return y
```

**辅助损失（负载均衡）**：

MoE 需要辅助损失来防止"路由崩塌"（所有 token 都被分配到同一专家）：

```python
# 辅助损失 = α * Σ(专家选择频率 × 专家平均分数)
aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
```

### 2.7 Transformer Block

```python
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id, config):
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        # Pre-LN: 先归一化再计算注意力
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        # 残差连接
        hidden_states += residual
        # FFN / MoE
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value
```

### 2.8 完整模型

```python
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.model = MiniMindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 权重共享：embedding 和 lm_head 使用同一套权重
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0):
        h, past_kvs, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache)
        logits = self.lm_head(h[:, -logits_to_keep:, :])
        return CausalLMOutputWithPast(logits=logits, past_key_values=past_kvs, aux_loss=aux_loss)
```

**权重共享（Weight Tying）**：`embed_tokens.weight = lm_head.weight`，减少参数量。

## 三、模型参数量计算

以 MiniMind2-Small（d_model=512, n_layers=8, n_heads=8, kv_heads=2, vocab=6400）为例：

| 组件 | 计算公式 | 参数量 |
|------|---------|-------|
| Embedding | 6400 × 512 | 3,276,800 |
| Q Projection | 512 × (8×64) | 262,144 |
| K Projection | 512 × (2×64) | 65,536 |
| V Projection | 512 × (2×64) | 65,536 |
| O Projection | (8×64) × 512 | 262,144 |
| Gate Proj | 512 × 1365 | 698,880 |
| Up Proj | 512 × 1365 | 698,880 |
| Down Proj | 1365 × 512 | 698,880 |
| RMSNorm ×2 | 512 × 2 | 1,024 |
| **单层合计** | | **~2.75M** |
| **8层合计** | | **~22M** |
| **+ Embedding + lm_head** | | **~25.8M** |

## 四、动手练习

### 练习 1：构建模型并查看结构

```python
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

config = MiniMindConfig(hidden_size=512, num_hidden_layers=8)
model = MiniMindForCausalLM(config)

# 查看参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params / 1e6:.2f}M")

# 查看模型结构
print(model)
```

### 练习 2：对比 Dense 和 MoE 模型

```python
# Dense 模型
config_dense = MiniMindConfig(hidden_size=512, use_moe=False)
model_dense = MiniMindForCausalLM(config_dense)

# MoE 模型
config_moe = MiniMindConfig(hidden_size=512, use_moe=True, n_routed_experts=4, n_shared_experts=1)
model_moe = MiniMindForCausalLM(config_moe)

print(f"Dense 参数量: {sum(p.numel() for p in model_dense.parameters()) / 1e6:.2f}M")
print(f"MoE 参数量: {sum(p.numel() for p in model_moe.parameters()) / 1e6:.2f}M")
```

### 练习 3：理解 RoPE 的位置编码效果

编写代码可视化不同位置的 RoPE 编码，观察其旋转特性。

## 五、下一阶段预告

下一阶段 [Phase 4: 预训练](./04_Phase4_预训练.md)，我们将学习：
- 预训练的完整流程
- 学习率调度策略
- 混合精度训练
- 分布式训练（DDP）
