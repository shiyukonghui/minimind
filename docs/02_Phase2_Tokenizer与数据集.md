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
| Yi tokenizer           | 64,000    | 01万物（中国）      | 中文友好     |
| Qwen2 tokenizer        | 151,643   | 阿里云（中国）        | 中文压缩率高   |
| ChatGLM tokenizer          | 151,329   | 智谱AI（中国）       | 中文压缩率高   |
| Mistral tokenizer      | 32,000    | Mistral AI（法国） | 英文友好     |
| Llama 3 tokenizer       | 128,000   | Meta（美国）       | 英文友好     |
| **MiniMind tokenizer** | **6,400** | **自定义**    | **极小体积** |

**MiniMind 选择小词表的原因**：

- 词嵌入层参数量 = `vocab_size × hidden_size`
- 词表太大 → Embedding 层参数占比过高 → 模型"头重脚轻"
- 词表 6400 × 512 维 = 328 万参数，仅占  26M 模型的 12.6%
- 若词表 128000 × 512 维 = 6554 万参数，远超模型其他部分

**小词表的代价**：压缩率低，"hello" 可能被拆分为 "h" "e" "l" "l" "o" 五个独立 token。

> 当前主线为避免历史版本歧义并控制整体体积，统一使用 `minimind_tokenizer`，不再维护 `mistral_tokenizer` 版本。
> 
> 尽管 `minimind_tokenizer` 的词表只有 `6400`，编解码效率弱于 `qwen2`、`glm` 等更偏中文友好的 tokenizer，但它能显著压缩 embedding 层和输出层的参数占比，更适合 MiniMind 这类小模型的体积约束。
> 
> 从实际使用效果看，这套 tokenizer 并没有明显带来生僻词解码失败的问题，整体仍然足够稳定可用。

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

> 注意：MiniMind 已自带训练好的分词器，通常无需重新训练。不建议重新训练 tokenizer，因为词表和切分规则一旦变化，模型权重、数据格式、推理接口与社区生态的兼容性都会下降，也会削弱模型的传播性。同时，tokenizer 还会影响 PPL 这类按 token 统计的指标，因此跨 tokenizer 比较时，BPB（Bits Per Byte）往往更有参考价值。

## 二、数据集详解

### 2.1 数据集下载

从 [ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) 或 [HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main) 下载，放到 `./dataset/` 目录下。

> 无需全部 clone，可单独下载所需的文件

将下载的数据集文件放到`./dataset/`目录下（✨为推荐的必须项）

```bash
./dataset/
├── agent_rl.jsonl (86MB)
├── agent_rl_math.jsonl (18MB)
├── dpo.jsonl (53MB)
├── pretrain_t2t_mini.jsonl (1.2GB, ✨)
├── pretrain_t2t.jsonl (10GB)
├── rlaif.jsonl (24MB, ✨)
├── sft_t2t_mini.jsonl (1.6GB, ✨)
└── sft_t2t.jsonl (14GB)
```

### 2.2 预训练数据集

**文件**：`pretrain_t2t.jsonl`（10GB）/ `pretrain_t2t_mini.jsonl`（1.2GB，✨推荐）

`MiniMind-3` 当前主线预训练数据为 `pretrain_t2t.jsonl` / `pretrain_t2t_mini.jsonl`。这两份数据已经整理成统一的 `text -> next token prediction` 训练格式，目标是在较小算力下兼顾：

- 文本质量
- 长度分布
- 中英混合能力
- 与后续 SFT / Tool Calling / RLAIF 阶段的模板衔接

数据来源包括但不限于通用文本语料、对话整理语料、蒸馏补充语料，以及各类**宽松开源协议**可用的数据集；主线数据会在清洗、去重、长度控制与格式统一后再进入训练。数据来源于：[匠数大模型数据集](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data)、[Magpie-Align](https://www.modelscope.cn/organization/Magpie-Align) 等公开数据源。

其中：

- `pretrain_t2t_mini.jsonl` 更适合快速复现
- `pretrain_t2t.jsonl` 更适合完整训练 `MiniMind-3` 主线模型

**数据格式**：

```jsonl
{"text": "如何才能摆脱拖延症？治愈拖延症并不容易，但以下建议可能有所帮助。"}
{"text": "清晨的阳光透过窗帘洒进房间，桌上的书页被风轻轻翻动。"}
{"text": "Transformer 通过自注意力机制建模上下文关系，是现代大语言模型的重要基础结构。"}
```

每行一条纯文本记录，无对话结构。预训练阶段的目标是让模型学会"词语接龙"。

### 2.3 SFT 数据集

**文件**：`sft_t2t.jsonl`（14GB）/ `sft_t2t_mini.jsonl`（1.6GB，✨推荐）

`MiniMind-3` 当前主线 SFT 数据为 `sft_t2t.jsonl` / `sft_t2t_mini.jsonl`。相比更早期的 `sft_512 / sft_1024 / sft_2048` 方案，当前版本更强调：

- 统一模板
- 更适合对话 + 思考标签 + Tool Calling 的混合训练
- 尽量减少数据预处理分叉，降低复现成本

其数据来源包括但不限于高质量指令跟随数据、公开对话数据、模型蒸馏合成数据，以及协议友好的开源数据集；在进入 `t2t` 主线前，会统一为当前仓库使用的多轮对话格式。当前主线中也包含大量合成数据，例如基于 `qwen3-4b` 合成的约 `10w` 条 `tool call` 数据，以及 `qwen3` 系列的 `reasoning` 数据等。其中社区主要来源有：[匠数大模型数据集](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data)、[Magpie-Align](https://www.modelscope.cn/organization/Magpie-Align)、[R1-Distill-SFT](https://www.modelscope.cn/datasets/AI-ModelScope/R1-Distill-SFT)、[COIG](https://huggingface.co/datasets/BAAI/COIG)、[Step-3.5-Flash-SFT](https://huggingface.co/datasets/stepfun-ai/Step-3.5-Flash-SFT) 等。

其中：

- `sft_t2t_mini.jsonl`：适合快速训练对话模型
- `sft_t2t.jsonl`：适合完整复现主线版本
- `toolcall` 能力已经并入主线 SFT 数据

**数据格式**（包含对话数据、Tool Use 数据）：

```jsonl
{
    "conversations": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
        {"role": "user", "content": "再见"},
        {"role": "assistant", "content": "再见！"}
    ]
}
{
    "conversations": [
        {"role": "system", "content": "# Tools ...", "tools": "[...]"},
        {"role": "user", "content": "把'你好世界'翻译成english"},
        {"role": "assistant", "content": "", "tool_calls": "[{\"name\":\"translate_text\",\"arguments\":{\"text\":\"你好世界\",\"target_language\":\"english\"}}]"},
        {"role": "tool", "content": "{\"translated_text\":\"Hello World\"}"},
        {"role": "assistant", "content": "Hello World"}
    ]
}
```

### 2.4 DPO 数据集

**文件**：`dpo.jsonl`（53MB）

`MiniMind` 当前主线 RL 数据为 `dpo.jsonl`。数据抽样自 [DPO-En-Zh-20k](https://huggingface.co/datasets/llamafactory/DPO-En-Zh-20k)。

主线中会将这部分样本统一重组为当前仓库使用的偏好学习格式，用于奖励模型或偏好优化阶段训练；其中 `chosen` 表示更符合偏好的回复，`rejected` 表示相对较差的回复。

**数据格式**：

```json
{
  "chosen": [
    {"content": "Q", "role": "user"}, 
    {"content": "good answer", "role": "assistant"}
  ], 
  "rejected": [
    {"content": "Q", "role": "user"}, 
    {"content": "bad answer", "role": "assistant"}
  ]
}
```

### 2.5 RLAIF 数据集

**文件**：`rlaif.jsonl`（24MB，✨推荐）

RLAIF 训练数据集，用于 PPO/GRPO/CISPO 等强化学习算法训练。

数据格式与 SFT 数据一致，通常是从 SFT 数据中按总长度和对话轮次筛选得到，并将最后一个 `assistant` 位置留空，供 rollout 阶段续写使用。

### 2.6 Agent RL 数据集

**文件**：`agent_rl.jsonl`（86MB）/ `agent_rl_math.jsonl`（18MB）

- `agent_rl.jsonl`：Agentic RL 主线训练数据，用于 `train_agent.py` 的多轮 Tool-Use / CISPO / GRPO 训练
- `agent_rl_math.jsonl`：Agentic RL 纯数学补充数据，适合带最终校验目标的多轮推理/工具使用场景（用于 RLVR）

### 2.7 数据集推荐配置

| 数据集 | 推荐设置 `max_seq_len` | 说明 |
|-------|----------------------|------|
| `pretrain_t2t_mini` | ≈768 | 轻量预训练数据，适合快速复现 |
| `pretrain_t2t` | ≈380 | 主线预训练数据 |
| `sft_t2t_mini` | ≈768 | 轻量 SFT 数据，用于快速训练 Zero 模型 |
| `sft_t2t` | - | 主线 SFT 数据，适合完整复现 |

> 训练参数 `max_seq_len` 目前指的是 tokens 长度，而非绝对字符数。
> 
> 本项目 tokenizer 在中文文本上大约 `1.5~1.7 字符/token`，纯英文的压缩比在 `4~5 字符/token`，不同数据分布会有波动。
> 
> 数据集命名标注的"最大长度"均为字符数，100 长度的字符串可粗略换算成 `100/1.5≈67` 的 tokens 长度。
> 
> 例如：
> - 中文：`白日依山尽` 5 个字符可能被拆分为 [`白日`,`依`,`山`,`尽`] 4 个 tokens
> - 英文：`The sun sets in the west` 24 个字符可能被拆分为 [`The `,`sun `,`sets `,`in `,`the`,`west`] 6 个 tokens
> 
> "推荐设置"给出了各个数据集上最大 tokens 长度的粗略估计。须知 `max_seq_len` 可以激进 / 保守 / 均衡地调整，因为更大或更小均无法避免副作用：一些样本短于 `max_seq_len` 后被 padding 浪费算力，一些样本长于 `max_seq_len` 后被截断语义。
> 
> 在算力效率与语义完整性之间找到平衡点即可。

### 2.8 推荐训练方案

- `minimind-3` 主线推荐采用 `pretrain_t2t` + `sft_t2t` + `rlaif/agent_rl` 的阶段式训练组合
- 想要最快速度从 0 实现 Zero 模型，推荐使用 `pretrain_t2t_mini.jsonl` + `sft_t2t_mini.jsonl` 的数据组合
- 推荐具备一定算力资源或更在意效果的朋友完整复现 `minimind-3`；仅有单卡 GPU 或更在意快速复现的朋友强烈推荐 mini 组合
- 当前 `sft_t2t / sft_t2t_mini` 已经混入 Tool Call 数据，因此通常不需要再额外做一轮独立的 Tool Calling 监督微调

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
- deepseek 解答：选择哪种策略取决于具体训练目标：
  - **如果你的目标是...**
    - 训练一个标准助手，执行指令和生成回答是核心。
    - **数据中 Prompt 高度重复**，想避免模型死记硬背。
    - **追求模型稳定**，降低产生不当言论的风险。
    - **此时，经典的"仅计算 Assistant 损失"（PLW=0）策略是稳妥且有效的选择。**
  - **如果你的目标是...**
    - 追求模型在特定任务上的极致性能（如多轮对话或长文本生成）。
    - 希望提升模型对多样化指令的理解和泛化能力。
    - **那么，可以考虑尝试"加权指令微调"策略（PLW 在 0.1-0.5 之间），以获得更优的表现。**

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
with open("./dataset/pretrain_t2t_mini.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 3: break
        data = json.loads(line)
        print(f"样本 {i}: {data['text'][:100]}...")

# 查看 SFT 数据
with open("./dataset/sft_t2t_mini.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 3: break
        data = json.loads(line)
        print(f"样本 {i}: {data['conversations']}")
```

### 练习 3：理解 Loss Mask

编写代码验证 SFTDataset 的 loss_mask 是否正确标记了 assistant 回复位置。

## 五、下一阶段预告

下一阶段 [Phase 3: 模型架构详解](./03_Phase3_模型架构详解.md)，我们将深入学习：

- Transformer Decoder-Only 架构
- RoPE 旋转位置编码
- GQA 分组查询注意力
- MoE 混合专家架构