# Phase 2: Tokenizer 与数据集

> 本阶段目标：理解分词器原理，掌握各阶段数据集的格式与处理方式

## 一、分词器（Tokenizer）原理

### 1.1 什么是分词器？

分词器是 LLM 的"词典"，负责将自然语言文本转换为数字序列（Token IDs），以及将数字序列还原为文本。

```
自然语言: "你好世界"
    ↓ Tokenizer 编码
Token IDs: [523, 891, 234, 567]
    ↓ 模型处理
输出 IDs: [891, 345, ...]
    ↓ Tokenizer 解码
自然语言: "你好！我是..."
```

LLM 的输出本质上是对词表中 N 个词的 Softmax 多分类问题，分词器就是"词典"的页码映射。

### 1.2 词表大小的权衡

| 分词器                    | 词表大小      | 来源         | 特点       |
| ---------------------- | --------- | ---------- | -------- |
| yi tokenizer           | 64,000    | 01万物       | 中文友好     |
| qwen2 tokenizer        | 151,643   | 阿里云        | 中文压缩率高   |
| glm tokenizer          | 151,329   | 智谱AI       | 中文压缩率高   |
| mistral tokenizer      | 32,000    | Mistral AI | 英文友好     |
| llama3 tokenizer       | 128,000   | Meta       | 英文友好     |
| **minimind tokenizer** | **6,400** | **自定义**    | **极小体积** |

**MiniMind 选择小词表的原因**：

- 词嵌入层参数量 = `vocab_size × hidden_size`
- 词表太大 → Embedding 层参数占比过高 → 模型"头重脚轻"
- 词表 6400 × 512 维 = 328 万参数，仅占 26M 模型的 12.6%
- 若词表 128000 × 512 维 = 6554 万参数，远超模型其他部分

**小词表的代价**：压缩率低，"hello" 可能被拆分为 "h" "e" "l" "l" "o" 五个独立 token。

### 1.3 BPE 分词算法

MiniMind 使用 BPE（Byte Pair Encoding）分词算法：

1. **初始化**：将文本拆分为字符级 token
2. **统计频率**：统计相邻 token 对的出现频率
3. **合并最高频对**：将最高频的相邻 token 对合并为新 token
4. **重复**：直到达到目标词表大小

```
初始: h e l l o   h e l l o   w o r l d
统计: "ll" 出现 2 次（最高频）
合并: h e ll o   h e ll o   w o r l d
统计: "he" 出现 2 次
合并: he ll o   he ll o   w o r l d
...
最终: hello   hello   world
```

### 1.4 训练自定义分词器

MiniMind 提供了训练自定义分词器的脚本 `scripts/train_tokenizer.py`，核心流程：

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

tokenizer = Tokenizer(BPE(unk_token="<unk>"))
trainer = BpeTrainer(
    vocab_size=6400,
    special_tokens=["<unk>", "<|im_start|>", "<|im_end|>"],
)
tokenizer.train(files=["corpus.txt"], trainer=trainer)
```

> 注意：MiniMind 已自带训练好的分词器，通常无需重新训练。

## 二、数据集详解

### 2.1 数据集下载

从 [ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) 或 [HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main) 下载，放到 `./dataset/` 目录下。

### 2.2 预训练数据集

**文件**：`pretrain_hq.jsonl`（1.6GB）

**数据格式**：

```json
{"text": "如何才能摆脱拖延症？治愈拖延症并不容易，但以下建议可能有所帮助..."}
```

每行一条纯文本记录，无对话结构。预训练阶段的目标是让模型学会"词语接龙"。

### 2.3 SFT 数据集

**文件**：`sft_mini_512.jsonl`（1.2GB，推荐）、`sft_512.jsonl`（7.5GB）、`sft_1024.jsonl`（5.6GB）、`sft_2048.jsonl`（9GB）

**数据格式**：

```json
{
    "conversations": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
        {"role": "user", "content": "再见"},
        {"role": "assistant", "content": "再见！"}
    ]
}
```

### 2.4 DPO 数据集

**文件**：`dpo.jsonl`（55MB）

**数据格式**：

```json
{
    "chosen": [
        {"content": "问题内容", "role": "user"},
        {"content": "好的回答", "role": "assistant"}
    ],
    "rejected": [
        {"content": "问题内容", "role": "user"},
        {"content": "差的回答", "role": "assistant"}
    ]
}
```

### 2.5 RLAIF 数据集

**文件**：`rlaif-mini.jsonl`（1MB）

assistant 内容为"无"，因为 RLAIF 训练中由策略模型实时采样生成回答。

### 2.6 LoRA 数据集

- `lora_identity.jsonl`（22.8KB）：自我认知数据
- `lora_medical.jsonl`（34MB）：医疗问答数据

格式与 SFT 一致，但专注于特定领域。

### 2.7 推理蒸馏数据集

**文件**：`r1_mix_1024.jsonl`（340MB）

assistant 回复包含思考链标签，格式为：`思考过程...最终回答`

## 三、数据集类代码解读

MiniMind 在 `dataset/lm_dataset.py` 中定义了 4 个数据集类。

### 3.1 PretrainDataset

```python
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def __getitem__(self, index):
        sample = self.samples[index]
        # 对纯文本进行 tokenize
        encoding = self.tokenizer(
            str(sample['text']),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding.input_ids.squeeze()
        # loss_mask: 非 padding 位置为 1，padding 位置为 0
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        # 自回归: X = input_ids[:-1], Y = input_ids[1:]
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        return X, Y, loss_mask
```

**关键点**：

- 预训练数据是纯文本，所有 token 都参与损失计算
- `loss_mask` 标记非 padding 位置，避免对 padding token 计算损失
- 自回归方式：输入 X 是前 n-1 个 token，标签 Y 是后 n-1 个 token

### 3.2 SFTDataset

```python
class SFTDataset(Dataset):
    def __getitem__(self, index):
        sample = self.samples[index]
        # 使用 chat_template 构建对话提示
        prompt = self._create_chat_prompt(sample['conversations'])
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        # 生成动态损失掩码：只在 assistant 回复位置计算损失
        loss_mask = self._generate_loss_mask(input_ids)

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        return X, Y, loss_mask
```

**关键点**：

- 使用 `apply_chat_template` 将对话转为 ChatML 格式
- `loss_mask` 只在 assistant 回复位置为 1，user 部分为 0
- 这确保模型只学习"如何回答"，而不是"如何提问"（思考：如果模型既学习如何回答也学习如何提问呢？）
- deepseek解答：选择哪种策略取决于具体训练目标：
  - **如果你的目标是...**
    - **训练一个标准助手**，执行指令和生成回答是核心。
    - **数据中Prompt高度重复**，想避免模型死记硬背。
    - **追求模型稳定**，降低产生不当言论的风险。
    - **此时，经典的“仅计算Assistant损失”（PLW=0）策略是稳妥且有效的选择。**
  - **如果你的目标是...**
    - **追求模型在特定任务上的极致性能**（如多轮对话或长文本生成）。
    - **希望提升模型对多样化指令的理解和泛化能力**。
    - **那么，可以考虑尝试“加权指令微调”策略（PLW在0.1-0.5之间），以获得更优的表现。**

**Loss Mask 生成逻辑**：

```python
def _generate_loss_mask(self, input_ids):
    loss_mask = [0] * len(input_ids)
    i = 0
    while i < len(input_ids):
        # 找到 <|im_start|>assistant 标记
        if input_ids[i:i + len(self.bos_id)] == self.bos_id:
            start = i + len(self.bos_id)
            # 找到 <|im_end|> 标记
            while end < len(input_ids):
                if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                    break
                end += 1
            # assistant 回复区域设为 1
            for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                loss_mask[j] = 1
    return loss_mask
```

### 3.3 DPODataset

DPO 数据集同时返回 chosen 和 rejected 两组数据：

```python
def __getitem__(self, index):
    item = self.data[index]
    chosen = item['chosen']
    rejected = item['rejected']
    # 分别 tokenize chosen 和 rejected
    chosen_prompt = self.tokenizer.apply_chat_template(chosen, ...)
    rejected_prompt = self.tokenizer.apply_chat_template(rejected, ...)
    return {
        'x_chosen': x_chosen, 'y_chosen': y_chosen, 'mask_chosen': mask_chosen,
        'x_rejected': x_rejected, 'y_rejected': y_rejected, 'mask_rejected': mask_rejected
    }
```

### 3.4 RLAIFDataset

RLAIF 数据集返回 prompt 文本和 answer 文本，不做 tokenize：

```python
def __getitem__(self, index):
    sample = self.samples[index]
    prompt, answer = self._create_chat_prompt(sample['conversations'])
    return {'prompt': prompt, 'answer': answer}
```

因为 RLAIF 训练中需要模型实时生成回答，tokenize 在训练循环中进行。

## 四、动手练习

### 练习 1：探索分词器

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./model")

# 测试中文编码
text = "你好，我是MiniMind"
tokens = tokenizer.encode(text)
print(f"原文: {text}")
print(f"Token IDs: {tokens}")
print(f"Token 数量: {len(tokens)}")
print(f"词表大小: {len(tokenizer)}")

# 逐 token 解码
for tid in tokens:
    print(f"  ID {tid} -> '{tokenizer.decode([tid])}'")
```

### 练习 2：查看数据集样本

```python
import json

# 查看预训练数据
with open("./dataset/pretrain_hq.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 3: break
        data = json.loads(line)
        print(f"样本 {i}: {data['text'][:100]}...")

# 查看 SFT 数据
with open("./dataset/sft_mini_512.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 3: break
        data = json.loads(line)
        print(f"样本 {i}: {data['conversations']}")
```

### 练习 3：理解 Loss Mask

编写代码验证 SFTDataset 的 loss\_mask 是否正确标记了 assistant 回复位置。

## 五、下一阶段预告

下一阶段 [Phase 3: 模型架构详解](./03_Phase3_模型架构详解.md)，我们将深入学习：

- Transformer Decoder-Only 架构
- RoPE 旋转位置编码
- GQA 分组查询注意力
- MoE 混合专家架构

