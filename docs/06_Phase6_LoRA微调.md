# Phase 6: LoRA 微调

> 本阶段目标：理解 LoRA 低秩适配原理，掌握垂域模型迁移方法

## 一、LoRA 概述

### 1.1 什么是 LoRA？

LoRA（Low-Rank Adaptation）是一种参数高效的微调方法，核心思想：

- **不修改**原始预训练权重
- 在权重矩阵旁**引入低秩分解**的旁路
- 只训练旁路的少量参数

```
原始权重 W (d×d):  冻结，不更新
             ↓
LoRA 旁路:  ΔW = A × B
            A: d×r (r << d)
            B: r×d
             ↓
输出: y = Wx + ΔWx = Wx + ABx
```

### 1.2 LoRA 的优势

| 方面 | 全参数微调 | LoRA 微调 |
|------|----------|----------|
| 可训练参数量 | 26M（100%） | ~0.5M（~2%） |
| 显存占用 | 高 | 低 |
| 训练速度 | 慢 | 快 |
| 灾难性遗忘 | 风险高 | 风险低 |
| 多任务切换 | 需要多个完整模型 | 只需切换 LoRA 权重 |

### 1.3 LoRA 的应用场景

- **垂域迁移**：让通用模型学会特定领域知识（医疗、法律等）
- **自我认知**：让模型知道"我是谁"
- **风格调整**：改变模型的回答风格
- **低成本定制**：在有限算力下定制模型

## 二、LoRA 代码实现

### 2.1 LoRA 层定义

MiniMind 在 `model/model_lora.py` 中从零实现了 LoRA：

```python
class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        d_in = original_layer.in_features
        d_out = original_layer.out_features

        # 低秩矩阵 A 和 B
        self.lora_A = nn.Parameter(torch.zeros(d_in, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, d_out))

        # 初始化：A 用高斯，B 用零
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # 原始输出 + LoRA 旁路输出
        original_output = self.original_layer(x)
        lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling
        return original_output + lora_output
```

**关键设计**：
- `lora_A` 用 Kaiming 初始化，`lora_B` 初始化为零
- 初始时 `lora_A @ lora_B = 0`，即 LoRA 旁路初始输出为零
- 这确保训练开始时模型行为与原始模型一致
- `scaling = alpha / rank` 控制旁路的整体缩放

### 2.2 应用 LoRA

```python
def apply_lora(model, rank=8, alpha=16, target_modules=None):
    """将模型中的指定线性层替换为 LoRA 层"""
    if target_modules is None:
        # 默认对 Q、K、V、O 投影层应用 LoRA
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                          'gate_proj', 'up_proj', 'down_proj']

    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # 替换为 LoRA 层
                lora_layer = LoRALayer(module, rank=rank, alpha=alpha)
                # 设置父模块的属性
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = dict(model.named_modules())[parent_name]
                setattr(parent, child_name, lora_layer)
```

### 2.3 保存和加载 LoRA 权重

```python
def save_lora(model, path):
    """只保存 LoRA 参数"""
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if 'lora' in name:
            lora_state_dict[name] = param.data
    torch.save(lora_state_dict, path)

def load_lora(model, path):
    """加载 LoRA 参数到模型"""
    lora_state_dict = torch.load(path)
    model_state_dict = model.state_dict()
    model_state_dict.update(lora_state_dict)
    model.load_state_dict(model_state_dict, strict=False)
```

## 三、LoRA 训练流程

### 3.1 启动训练

```bash
cd trainer
python train_lora.py

# 指定 LoRA 名称和数据
python train_lora.py --lora_name lora_medical --data_path ../dataset/lora_medical.jsonl
```

### 3.2 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lora_name` | lora_identity | LoRA 权重名称 |
| `--epochs` | 50 | 训练轮数（LoRA 通常需要更多轮次） |
| `--learning_rate` | 1e-4 | 学习率（比全参 SFT 大） |
| `--from_weight` | full_sft | 基础模型权重 |
| `--data_path` | ../dataset/lora_identity.jsonl | LoRA 数据路径 |

### 3.3 训练核心逻辑

```python
# 1. 加载基础模型
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)

# 2. 应用 LoRA
apply_lora(model)

# 3. 冻结非 LoRA 参数
lora_params = []
for name, param in model.named_parameters():
    if 'lora' in name:
        param.requires_grad = True
        lora_params.append(param)
    else:
        param.requires_grad = False

# 4. 只优化 LoRA 参数
optimizer = optim.AdamW(lora_params, lr=args.learning_rate)

# 5. 训练循环（与 SFT 类似，但只更新 LoRA 参数）
for step, (X, Y, loss_mask) in enumerate(loader):
    with autocast_ctx:
        res = model(X)
        loss = (loss_fct(res.logits.view(-1, V), Y.view(-1)).view(Y.size()) * loss_mask).sum() / loss_mask.sum()
    scaler.scale(loss).backward()
    # 梯度裁剪只针对 LoRA 参数
    torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
    scaler.step(optimizer)
```

### 3.4 参数统计

```python
total_params = sum(p.numel() for p in model.parameters())
lora_params_count = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)
print(f"LLM 总参数量: {total_params / 1e6:.3f} M")
print(f"LoRA 参数量: {lora_params_count / 1e6:.3f} M")
print(f"LoRA 参数占比: {lora_params_count / total_params * 100:.2f}%")
```

## 四、垂域迁移实战

### 4.1 医疗场景

准备医疗问答数据 `lora_medical.jsonl`：

```json
{"conversations": [{"role": "user", "content": "请问颈椎病的人枕头多高才最好？"}, {"role": "assistant", "content": "颈椎病患者选择枕头的高度应该根据..."}]}
```

训练：

```bash
python train_lora.py --lora_name lora_medical --data_path ../dataset/lora_medical.jsonl --epochs 20
```

测试：

```bash
python eval_llm.py --weight full_sft --lora_weight lora_medical
```

### 4.2 自我认知场景

准备自我认知数据 `lora_identity.jsonl`：

```json
{"conversations": [{"role": "user", "content": "你叫什么名字？"}, {"role": "assistant", "content": "我叫MiniMind"}]}
{"conversations": [{"role": "user", "content": "你是谁"}, {"role": "assistant", "content": "我是MiniMind，一个由...开发的AI助手"}]}
```

训练后，模型将学会自我认知，而不会遗忘通用对话能力。

### 4.3 LoRA + 基础模型的工作方式

```
基础模型 (full_sft_512.pth)
    + LoRA 外挂 (lora_medical_512.pth)
    = 医疗增强模型
```

LoRA 权重是"外挂"，不修改基础模型本身。可以随时切换不同的 LoRA 权重来获得不同的能力。

## 五、动手练习

### 练习 1：训练自我认知 LoRA

```bash
cd trainer
python train_lora.py --lora_name lora_identity --epochs 50
```

### 练习 2：测试 LoRA 效果

```bash
# 不使用 LoRA
python eval_llm.py --weight full_sft

# 使用 LoRA
python eval_llm.py --weight full_sft --lora_weight lora_identity
```

对比两种情况下模型对"你是谁"的回答。

### 练习 3：创建自定义 LoRA 数据集

选择一个你感兴趣的领域（如编程、历史、美食等），创建对话数据集并训练 LoRA。

## 六、下一阶段预告

下一阶段 [Phase 7: 知识蒸馏](./07_Phase7_知识蒸馏.md)，我们将学习：
- 白盒蒸馏 vs 黑盒蒸馏
- KL 散度与软标签
- 如何让小模型学习大模型的知识
