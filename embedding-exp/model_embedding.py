"""
多种向量化表示方法的模型定义
支持四种嵌入方案：A(高维基线)、B(二维固定坐标)、C1(三维固定模式)、C2(三维可学习瓶颈)
基于 MiniMind 模型架构，仅替换嵌入层部分
"""
import os
import sys

# 确保可以导入项目根目录的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from model.model_minimind import (
    MiniMindConfig, MiniMindModel, MiniMindForCausalLM,
    MiniMindBlock, RMSNorm, MOEFeedForward,
    precompute_freqs_cis
)


# ============================================================
#  配置类
# ============================================================

class MiniMindEmbeddingConfig(MiniMindConfig):
    """扩展配置，支持多种嵌入方案"""
    model_type = "minimind_embedding"

    def __init__(self, embed_type='A', coord_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.embed_type = embed_type    # 'A', 'B', 'C1', 'C2'
        self.coord_dim = coord_dim      # 坐标维度：B=2, C1=3, C2=3


# ============================================================
#  固定编码生成函数
# ============================================================

def generate_fixed_2d_coords(vocab_size):
    """生成方案B的二维固定坐标，归一化到[0,1]

    每个token映射到 (x, y)：
    - x = token_id % 128 / 127.0
    - y = token_id // 128 / 127.0
    这将6400个token映射到一个 50×128 的二维网格上
    """
    coords = torch.zeros(vocab_size, 2)
    for i in range(vocab_size):
        coords[i, 0] = (i % 128) / 127.0   # x: 0~1
        coords[i, 1] = (i // 128) / 127.0   # y: 0~1
    return coords


def generate_fixed_3d_sinusoid(vocab_size):
    """生成方案C-1的三维正弦固定向量，归一化到[0,1]

    使用三个不同相位的正弦函数生成三维编码：
    - x = sin(idx * 2π / period)
    - y = sin(idx * 2π / period + 2π/3)
    - z = sin(idx * 2π / period + 4π/3)
    period = vocab_size^(1/3) ≈ 18.6，使每个字落在三维空间的平滑曲线上
    """
    period = vocab_size ** (1 / 3)
    indices = torch.arange(vocab_size, dtype=torch.float)
    x = torch.sin(indices * 2 * math.pi / period)
    y = torch.sin(indices * 2 * math.pi / period + 2 * math.pi / 3)
    z = torch.sin(indices * 2 * math.pi / period + 4 * math.pi / 3)
    # 从[-1,1]映射到[0,1]
    coords = torch.stack([(x + 1) / 2, (y + 1) / 2, (z + 1) / 2], dim=-1)
    return coords


# ============================================================
#  模型类
# ============================================================

class MiniMindEmbeddingModel(MiniMindModel):
    """支持多种嵌入方案的MiniMind模型

    根据embed_type选择不同的嵌入方式：
    - A: 标准nn.Embedding(vocab, hidden_size)
    - B: 固定二维坐标 + Linear(2, hidden_size)
    - C1: 固定三维正弦编码 + Linear(3, hidden_size)
    - C2: 可学习Embedding(vocab, 3) + Linear(3, hidden_size)
    """

    def __init__(self, config: MiniMindEmbeddingConfig):
        # 不调用父类 __init__，手动初始化以替换嵌入层
        nn.Module.__init__(self)
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_type = config.embed_type

        # 根据方案创建不同的嵌入层
        if config.embed_type == 'A':
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        elif config.embed_type == 'B':
            fixed_vecs = generate_fixed_2d_coords(config.vocab_size)
            self.fixed_embed = nn.Embedding.from_pretrained(fixed_vecs, freeze=True)
            self.coord_proj = nn.Linear(2, config.hidden_size, bias=False)
        elif config.embed_type == 'C1':
            fixed_vecs = generate_fixed_3d_sinusoid(config.vocab_size)
            self.fixed_embed = nn.Embedding.from_pretrained(fixed_vecs, freeze=True)
            self.coord_proj = nn.Linear(3, config.hidden_size, bias=False)
        elif config.embed_type == 'C2':
            self.tiny_embed = nn.Embedding(config.vocab_size, 3)
            self.tiny_proj = nn.Linear(3, config.hidden_size, bias=False)
        else:
            raise ValueError(f"不支持的嵌入方案: {config.embed_type}，可选: A/B/C1/C2")

        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [MiniMindBlock(l, config) for l in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids=None, attention_mask=None,
                past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # 根据方案选择不同的嵌入计算方式
        if self.embed_type == 'A':
            hidden_states = self.dropout(self.embed_tokens(input_ids))
        elif self.embed_type in ('B', 'C1'):
            coords = self.fixed_embed(input_ids)
            hidden_states = self.dropout(self.coord_proj(coords))
        elif self.embed_type == 'C2':
            tiny = self.tiny_embed(input_ids)
            hidden_states = self.dropout(self.tiny_proj(tiny))

        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states, position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )

        return hidden_states, presents, aux_loss

    def get_embedding_weights(self):
        """获取嵌入层的映射后权重（用于语义分析）

        返回 (vocab_size, hidden_size) 的张量
        """
        if self.embed_type == 'A':
            return self.embed_tokens.weight.data
        elif self.embed_type in ('B', 'C1'):
            # 固定编码通过投影层映射
            with torch.no_grad():
                return self.coord_proj(self.fixed_embed.weight.data)
        elif self.embed_type == 'C2':
            # 可学习编码通过投影层映射
            with torch.no_grad():
                return self.tiny_proj(self.tiny_embed.weight.data)

    def get_raw_embedding_weights(self):
        """获取嵌入层的原始低维权重（用于可视化低维表示）

        返回 (vocab_size, coord_dim) 的张量
        """
        if self.embed_type == 'A':
            return self.embed_tokens.weight.data
        elif self.embed_type in ('B', 'C1'):
            return self.fixed_embed.weight.data
        elif self.embed_type == 'C2':
            return self.tiny_embed.weight.data

    def get_embed_param_count(self):
        """获取嵌入相关可训练参数量"""
        total = 0
        if self.embed_type == 'A':
            total += sum(p.numel() for p in self.embed_tokens.parameters() if p.requires_grad)
        elif self.embed_type in ('B', 'C1'):
            total += sum(p.numel() for p in self.coord_proj.parameters() if p.requires_grad)
        elif self.embed_type == 'C2':
            total += sum(p.numel() for p in self.tiny_embed.parameters() if p.requires_grad)
            total += sum(p.numel() for p in self.tiny_proj.parameters() if p.requires_grad)
        return total


class MiniMindEmbeddingForCausalLM(MiniMindForCausalLM):
    """支持多种嵌入方案的CausalLM

    关键差异：
    - 方案A：保持权重绑定 (embed_tokens.weight = lm_head.weight)
    - 方案B/C1/C2：断开权重绑定，lm_head独立参数
    """
    config_class = MiniMindEmbeddingConfig

    def __init__(self, config: MiniMindEmbeddingConfig = None):
        self.config = config or MiniMindEmbeddingConfig()
        PreTrainedModel.__init__(self, self.config)
        self.model = MiniMindEmbeddingModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        # 仅方案A保持权重绑定
        if self.config.embed_type == 'A':
            self.model.embed_tokens.weight = self.lm_head.weight

        self.OUT = CausalLMOutputWithPast()

    def get_embed_param_count(self):
        """获取嵌入相关可训练参数量（包含lm_head差异）"""
        embed_params = self.model.get_embed_param_count()
        # 方案B/C1/C2的lm_head是独立的，也应计入
        if self.config.embed_type != 'A':
            embed_params += sum(p.numel() for p in self.lm_head.parameters() if p.requires_grad)
        return embed_params


# ============================================================
#  验证与参数统计工具
# ============================================================

def print_model_stats(embed_type='A', hidden_size=512, vocab_size=6400):
    """打印模型参数统计信息"""
    config = MiniMindEmbeddingConfig(
        hidden_size=hidden_size,
        num_hidden_layers=8,
        embed_type=embed_type,
        vocab_size=vocab_size
    )
    model = MiniMindEmbeddingForCausalLM(config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    embed_params = model.get_embed_param_count()

    print(f"\n{'='*60}")
    print(f"方案 {embed_type} 模型参数统计:")
    print(f"{'='*60}")
    print(f"  总参数量:         {total_params:>12,} ({total_params/1e6:.3f}M)")
    print(f"  可训练参数量:     {trainable_params:>12,} ({trainable_params/1e6:.3f}M)")
    print(f"  嵌入相关参数量:   {embed_params:>12,} ({embed_params/1e3:.3f}K)")
    print(f"  非嵌入参数量:     {trainable_params - embed_params:>12,} ({(trainable_params - embed_params)/1e6:.3f}M)")
    print(f"  权重绑定:         {'是' if embed_type == 'A' else '否'}")
    print(f"{'='*60}")

    return model


if __name__ == '__main__':
    # 验证各方案模型能正常前向传播
    print("验证多种嵌入方案模型...")

    for embed_type in ['A', 'B', 'C1', 'C2']:
        model = print_model_stats(embed_type)

        # 测试前向传播
        input_ids = torch.randint(0, 6400, (2, 32))
        with torch.no_grad():
            output = model(input_ids)
        print(f"  forward test: logits shape = {output.logits.shape} OK")

        # 测试嵌入权重提取
        emb_weights = model.model.get_embedding_weights()
        print(f"  embed weights: shape = {emb_weights.shape} OK")

        raw_weights = model.model.get_raw_embedding_weights()
        print(f"  raw embed weights: shape = {raw_weights.shape} OK")

        print()

    print("所有方案验证通过!")
