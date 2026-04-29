# 多种向量化表示方法实验框架

## 1. 需求场景与处理逻辑

### 1.1 实验背景
基于 `thinking/多种向量化表示方法测试.md` 的设计方案，实现四组并行嵌入实验，对比不同语义编码方式对语言模型性能的影响：

| 方案 | 嵌入表示 | 映射到 d_model 的方式 | 可学习参数量 |
|------|----------|----------------------|--------------|
| **A（高维基线）** | `Embedding(6400, 512)` | 直接作为输入 | `6400 × 512 = 3,276,800` |
| **B（二维固定坐标）** | 固定 `(x, y)`，值域 0~127 | `Linear(2, 512, bias=False)` | `2 × 512 = 1,024` |
| **C-1（三维固定模式）** | 正弦生成的三维固定向量 | `Linear(3, 512, bias=False)` | `3 × 512 = 1,536` |
| **C-2（三维可学习瓶颈）** | `Embedding(6400, 3)` | `Linear(3, 512, bias=False)` | `6400×3 + 3×512 = 20,736` |

### 1.2 处理逻辑
- 四组实验共享相同的 Transformer 主体架构（d_model=512, 8层, GQA等）
- 仅替换嵌入层部分，其余完全一致
- **方案 A（高维基线）直接使用现有的预训练模型** `MiniMind2-Pretrain-512/`，无需从头训练
- 方案 B、C-1、C-2 各自独立训练、独立评测
- 最终汇总对比：收敛速度、PPL、参数效率比、语义拓扑质量

> **公平性说明**：方案 A 的基线模型已经过充分预训练，而 B/C-1/C-2 将在相同数据上从头训练。这反映了实际场景中的对比——现有多维嵌入模型 vs 极低维替代方案从零训练。在分析结论中需注意此差异。

---

## 2. 架构与技术方案

### 2.1 目录结构
```
embedding-exp/                     # 新建实验目录
├── model_embedding.py             # 支持多嵌入方案的模型定义
├── train_embedding_exp.py         # 统一训练脚本（--embed_type 切换方案）
├── eval_embedding_exp.py          # 评测脚本（基准评测 + 语义分析）
├── convert_for_eval.py            # PyTorch → HuggingFace 格式转换
├── run_all.bat                    # Windows 全流程脚本
└── run_all.sh                     # Linux 全流程脚本
```

### 2.2 模型改造方案

核心思路：创建 `MiniMindEmbeddingConfig` 继承 `MiniMindConfig`，新增 `embed_type` 和 `coord_dim` 参数；创建 `MiniMindEmbeddingModel` 继承 `MiniMindModel`，重写嵌入层部分。

**关键设计决策**：

1. **权重绑定（Weight Tying）处理**：
   - 原始模型：`embed_tokens.weight = lm_head.weight`（6400×512）
   - 方案 A：保持权重绑定
   - 方案 B/C-1/C-2：嵌入维度与 lm_head 维度不匹配，**必须断开权重绑定**，lm_head 独立为 `Linear(512, 6400, bias=False)`

2. **固定编码生成**：
   - 方案 B：每个 token 的二维坐标 = `(token_id % 128, token_id // 128)`，归一化到 [0, 1]
   - 方案 C-1：三维正弦编码：
     ```python
     period = vocab_size ** (1/3)  # 约18.6
     x = sin(idx * 2π / period)
     y = sin(idx * 2π / period + 2π/3)
     z = sin(idx * 2π / period + 4π/3)
     # 归一化到 [0, 1]
     ```

3. **嵌入层实现**：
   ```python
   # MiniMindEmbeddingModel.__init__ 中的嵌入分支
   if embed_type == 'A':
       self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
   elif embed_type == 'B':
       self.fixed_embed = nn.Embedding.from_pretrained(fixed_2d_coords, freeze=True)
       self.coord_proj = nn.Linear(2, hidden_size, bias=False)
   elif embed_type == 'C1':
       self.fixed_embed = nn.Embedding.from_pretrained(fixed_3d_sinusoid, freeze=True)
       self.coord_proj = nn.Linear(3, hidden_size, bias=False)
   elif embed_type == 'C2':
       self.tiny_embed = nn.Embedding(vocab_size, 3)
       self.tiny_proj = nn.Linear(3, hidden_size, bias=False)
   ```

4. **前向传播适配**：
   ```python
   def forward(self, input_ids, ...):
       if self.embed_type == 'A':
           hidden_states = self.dropout(self.embed_tokens(input_ids))
       elif self.embed_type in ('B', 'C1'):
           coords = self.fixed_embed(input_ids)
           hidden_states = self.dropout(self.coord_proj(coords))
       elif self.embed_type == 'C2':
           tiny = self.tiny_embed(input_ids)
           hidden_states = self.dropout(self.tiny_proj(tiny))
   ```

### 2.3 训练脚本设计

训练脚本基于 `trainer/train_pretrain.py` 和 `DST-train/train_baseline.py` 的模式，关键差异：
- 新增 `--embed_type` 参数（B/C1/C2），方案 A 不需训练脚本
- 根据方案创建不同的模型实例
- 记录并打印各方案的可训练参数量、嵌入相关参数量
- 训练日志中标注方案名称
- 模型保存路径包含方案标识：`{save_dir}/{embed_type}_512.pth`

**方案 A（基线）处理**：
- 直接使用 `MiniMind2-Pretrain-512/` 目录下的 HuggingFace 格式模型
- 该模型已是 `model_type=llama` 格式，`eval_benchmark.py` 可直接加载
- 语义分析时直接读取该模型的 `embed_tokens` 权重

### 2.4 评测脚本设计

评测脚本包含两部分：

#### 2.4.1 基准评测（C-Eval + CMMLU）
- 与现有 `eval_benchmark.py` 逻辑一致
- 使用 ABCD token 概率对比法
- 对每个训练好的模型分别评测

#### 2.4.2 语义分析（新增）
- **最邻近字分析**：提取嵌入权重，对常见字（高频 top-200）计算余弦相似度，列出每个字的 top-5 邻居
- **语义拓扑可视化**：对嵌入向量做 t-SNE 降维到 2D，绘制散点图并标注汉字
- **参数效率比计算**：`语义耦合度 = 最终PPL / 嵌入相关参数总量`
- **训练曲线对比**：读取各组实验的 loss 记录，绘制对比曲线

### 2.5 格式转换脚本
- 将 B/C-1/C-2 的 PyTorch 权重转换为 HuggingFace 格式
- 方案 A 无需转换，已是 HuggingFace 格式
- 复制分词器到输出目录
- 与 `DST-train/convert_for_eval.py` 逻辑一致，但支持多方案模型

### 2.6 全流程脚本
- 参考 `DST-train/run_all.bat` 的结构
- 流程：Step1 训练 A → Step2 训练 B → Step3 训练 C1 → Step4 训练 C2 → Step5 转换格式 → Step6 评测 → Step7 语义分析对比
- 支持跳过已完成的步骤

---

## 3. 受影响的文件

### 3.1 新建文件
| 文件 | 类型 | 说明 |
|------|------|------|
| `embedding-exp/model_embedding.py` | 新建 | 多嵌入方案模型定义 |
| `embedding-exp/train_embedding_exp.py` | 新建 | 统一训练脚本 |
| `embedding-exp/eval_embedding_exp.py` | 新建 | 评测+语义分析脚本 |
| `embedding-exp/convert_for_eval.py` | 新建 | 格式转换脚本 |
| `embedding-exp/run_all.bat` | 新建 | Windows全流程脚本 |
| `embedding-exp/run_all.sh` | 新建 | Linux全流程脚本 |

### 3.2 依赖的现有文件（只读，不修改）
| 文件 | 用途 |
|------|------|
| `model/model_minimind.py` | 继承 MiniMindConfig、MiniMindModel、MiniMindForCausalLM |
| `dataset/lm_dataset.py` | 使用 PretrainDataset |
| `trainer/trainer_utils.py` | 使用 get_lr、Logger、init_model 等工具函数 |
| `model/tokenizer.json` | 分词器 |
| `eval_benchmark.py` | 参考基准评测逻辑 |

---

## 4. 实现细节

### 4.1 model_embedding.py 关键代码

```python
class MiniMindEmbeddingConfig(MiniMindConfig):
    """扩展配置，支持多种嵌入方案"""
    model_type = "minimind_embedding"

    def __init__(self, embed_type='A', coord_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.embed_type = embed_type  # 'A', 'B', 'C1', 'C2'
        self.coord_dim = coord_dim    # 坐标维度：B=2, C1=3, C2=3


def generate_fixed_2d_coords(vocab_size):
    """生成方案B的二维固定坐标，归一化到[0,1]"""
    coords = torch.zeros(vocab_size, 2)
    for i in range(vocab_size):
        coords[i, 0] = (i % 128) / 127.0   # x: 0~1
        coords[i, 1] = (i // 128) / 127.0   # y: 0~1
    return coords


def generate_fixed_3d_sinusoid(vocab_size):
    """生成方案C-1的三维正弦固定向量，归一化到[0,1]"""
    period = vocab_size ** (1 / 3)
    indices = torch.arange(vocab_size, dtype=torch.float)
    x = torch.sin(indices * 2 * math.pi / period)
    y = torch.sin(indices * 2 * math.pi / period + 2 * math.pi / 3)
    z = torch.sin(indices * 2 * math.pi / period + 4 * math.pi / 3)
    # 从[-1,1]映射到[0,1]
    coords = torch.stack([(x + 1) / 2, (y + 1) / 2, (z + 1) / 2], dim=-1)
    return coords


class MiniMindEmbeddingModel(MiniMindModel):
    """支持多种嵌入方案的MiniMind模型"""
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

        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(config.num_hidden_layers)])
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

        # 根据方案选择不同的嵌入计算
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


class MiniMindEmbeddingForCausalLM(MiniMindForCausalLM):
    """支持多种嵌入方案的CausalLM"""
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
```

### 4.2 train_embedding_exp.py 关键差异

```python
# 新增参数（仅 B/C1/C2 需要训练）
parser.add_argument('--embed_type', default='B', type=str,
                    choices=['B', 'C1', 'C2'], help="嵌入方案类型")

# 模型创建
lm_config = MiniMindEmbeddingConfig(
    hidden_size=args.hidden_size,
    num_hidden_layers=args.num_hidden_layers,
    use_moe=bool(args.use_moe),
    embed_type=args.embed_type
)
model = MiniMindEmbeddingForCausalLM(lm_config)

# 统计并打印嵌入相关参数量
embed_params = 0
if args.embed_type == 'A':
    embed_params = sum(p.numel() for p in model.model.embed_tokens.parameters())
elif args.embed_type in ('B', 'C1'):
    embed_params = sum(p.numel() for p in model.model.coord_proj.parameters())
elif args.embed_type == 'C2':
    embed_params = sum(p.numel() for p in model.model.tiny_embed.parameters())
    embed_params += sum(p.numel() for p in model.model.tiny_proj.parameters())

# 保存路径包含方案标识
ckp = f'{args.save_dir}/{args.embed_type.lower()}_{lm_config.hidden_size}.pth'
```

### 4.3 eval_embedding_exp.py 语义分析部分

```python
def analyze_embeddings(model, tokenizer, embed_type, output_dir):
    """语义拓扑分析"""
    # 1. 提取嵌入权重
    if embed_type == 'A':
        weights = model.model.embed_tokens.weight.data.cpu()
    elif embed_type in ('B', 'C1'):
        weights = model.model.fixed_embed.weight.data.cpu()
        # 通过投影层映射后的表示
        proj_weight = model.model.coord_proj.weight.data.cpu()
        weights = weights @ proj_weight.T  # (vocab, hidden_size)
    elif embed_type == 'C2':
        weights = model.model.tiny_embed.weight.data.cpu()
        proj_weight = model.model.tiny_proj.weight.data.cpu()
        weights = weights @ proj_weight.T  # (vocab, hidden_size)

    # 2. 高频字最邻近分析
    freq_tokens = get_high_freq_tokens(tokenizer, top_k=200)
    for token_id in freq_tokens[:20]:
        vec = weights[token_id]
        sims = F.cosine_similarity(vec.unsqueeze(0), weights)
        top5_ids = sims.topk(6).indices[1:]  # 排除自身
        neighbors = [tokenizer.decode([i]) for i in top5_ids]
        # 记录结果

    # 3. t-SNE 可视化
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42)
    coords_2d = tsne.fit_transform(weights[freq_tokens].numpy())
    # 绘制散点图并标注

    # 4. 参数效率比
    # 语义耦合度 = 最终PPL / 嵌入相关参数总量
```

### 4.4 convert_for_eval.py

与 `DST-train/convert_for_eval.py` 结构类似，但：
- 使用 `MiniMindEmbeddingConfig` 和 `MiniMindEmbeddingForCausalLM`
- 支持 `--embed_type` 参数
- 输出目录命名为 `{embed_type}_hf`

### 4.5 run_all.bat / run_all.sh

```
流程：
1. 评测基线方案A (使用现有 MiniMind2-Pretrain-512 模型，无需训练)
2. 训练方案B (embed_type=B)
3. 训练方案C1 (embed_type=C1)
4. 训练方案C2 (embed_type=C2)
5. 转换B/C1/C2模型为HuggingFace格式
6. 基准评测 (C-Eval + CMMLU，四组全部评测)
7. 语义分析对比 (四组全部分析)
```

---

## 5. 边界条件与异常处理

1. **权重绑定断开**：方案 B/C-1/C-2 的 lm_head 需要独立初始化，不能与嵌入层共享权重
2. **固定编码冻结**：方案 B/C-1 的 `fixed_embed` 必须设置 `freeze=True`，训练过程中不更新
3. **lm_head 参数量**：方案 B/C-1/C-2 的 lm_head 独立后，可训练参数量会比方案 A 多 `512 × 6400 = 3,276,800`（lm_head 参数），这在对比实验中需注明
4. **显存考量**：四组实验的 Transformer 主体完全一致，显存差异仅来自嵌入层，差异极小
5. **PPL 计算**：确保各方案使用相同的评估数据集和计算方式
6. **格式转换兼容性**：HuggingFace 格式需要正确保存和加载 `MiniMindEmbeddingConfig` 的额外参数

---

## 6. 数据流路径

```
训练阶段（仅 B/C1/C2）：
  数据集 (pretrain_t2t_mini.jsonl)
    → PretrainDataset (tokenize + padding)
    → DataLoader (batch)
    → MiniMindEmbeddingForCausalLM (根据 embed_type 选择嵌入方式)
    → CrossEntropyLoss (带 loss_mask)
    → 优化器更新
    → 保存 {embed_type}_512.pth

基线方案A（直接使用现有模型）：
  MiniMind2-Pretrain-512/ (HuggingFace格式，直接用于评测)
    → eval_benchmark.py → 基线评测结果
    → eval_embedding_exp.py → 基线语义分析

评测阶段（B/C1/C2）：
  {embed_type}_512.pth
    → convert_for_eval.py → {embed_type}_hf/ (HuggingFace格式)
    → eval_benchmark.py → 基准评测结果
    → eval_embedding_exp.py → 语义分析结果 (最邻近 + t-SNE + 参数效率比)

汇总阶段：
  四组方案结果 → 对比表格 + 可视化图表
```

---

## 7. 预期成果

1. 方案 A 基线评测结果（使用现有预训练模型）
2. 方案 B/C-1/C-2 各自训练完成，得到对应的模型权重
3. 四组实验的 C-Eval/CMMLU 基准评测结果
4. 语义分析报告：
   - 各方案高频字最邻近对比表
   - t-SNE 语义拓扑可视化图
   - 参数效率比对比（PPL / 嵌入参数量）
5. 训练 loss 曲线对比图（B/C1/C2）
6. 全流程自动化脚本，一键运行实验
