# Phase 9: 推理模型与部署

> 本阶段目标：掌握推理模型的蒸馏训练，了解模型部署与服务化的方法

## 一、推理模型概述

### 1.1 什么是推理模型？

推理模型（Reasoning Model）在回答问题前会先进行"思考"，生成思考链（Chain of Thought），然后给出最终答案。

**回复模板**：

```
思考过程
...
最终回答
...
```

### 1.2 推理模型的训练方式

| 方式 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| RL 训练 | 通过强化学习引导模型学会思考 | 效果上限高 | 小模型几乎不可能成功 |
| 蒸馏训练 | 从大推理模型蒸馏思考链数据 | 简单直接，效果可控 | 上限受教师模型限制 |

**MiniMind 的选择**：由于参数量极小（0.1B），直接通过 RL 获得推理能力几乎不可能，因此选择蒸馏路线。

## 二、推理蒸馏训练

### 2.1 数据准备

使用 `r1_mix_1024.jsonl`（340MB），整合自 DeepSeek-R1 蒸馏数据集。

数据格式示例：

```json
{
    "conversations": [
        {"role": "user", "content": "你好"},
        {
            "role": "assistant",
            "content": "嗯，用户让我介绍一下自己。我需要先理解用户的需求...\n\n你好！我是由中国的个人开发者开发的智能助手MiniMind-R1。"
        }
    ]
}
```

### 2.2 特殊的损失加权

推理蒸馏的关键技巧：对思考标签位置增加损失权重。

```python
def train_epoch(epoch, loader, iters, tokenizer, lm_config, ...):
    # 思考标签的 token IDs
    start_of_think_ids = tokenizer('思索').input_ids
    end_of_think_ids = tokenizer('思索结束').input_ids
    start_of_answer_ids = tokenizer('<answer>').input_ids
    end_of_answer_ids = tokenizer('</answer>').input_ids

    for step, (X, Y, loss_mask) in enumerate(loader):
        with autocast_ctx:
            res = model(X)
            loss = loss_fct(res.logits.view(-1, V), Y.view(-1)).view(Y.size())

            # 特殊标签位置增加权重
            sp_ids = torch.isin(Y.view(-1),
                                torch.tensor(start_of_think_ids + end_of_think_ids
                                             + start_of_answer_ids + end_of_answer_ids
                                             ).to(args.device))
            loss_mask = loss_mask.view(-1)
            loss_mask_sum = loss_mask.sum()
            loss_mask[sp_ids] = 10  # 对思考标签增加 10 倍权重
            loss_mask = loss_mask.view(Y.size())

            loss = (loss * loss_mask).sum() / loss_mask_sum
```

**为什么需要加权？**

模型容易"忘记"使用思考标签，导致输出不符合推理模板。增加标签位置的损失权重，强制模型学会正确使用思考标签。

### 2.3 启动训练

```bash
cd trainer
python train_distill_reason.py
```

关键参数：
- `--from_weight dpo`：基于 DPO 后的模型训练
- `--data_path ../dataset/r1_mix_1024.jsonl`：推理蒸馏数据
- `--max_seq_len 1024`：推理数据最大长度

### 2.4 训练后的模型

权重保存为 `reason_*.pth`，测试：

```bash
python eval_llm.py --weight reason
```

## 三、RLAIF 训练推理模型

PPO 和 GRPO 也支持训练推理模型，通过 `--reasoning 1` 参数启用：

```bash
# PPO 训练推理模型
python train_ppo.py --reasoning 1

# GRPO 训练推理模型
python train_grpo.py --reasoning 1
```

启用后，奖励函数会额外计算：
1. **格式奖励**：检查思考标签格式是否正确（0.5 分）
2. **标记奖励**：检查各标签是否恰好出现一次（每个 0.25 分）
3. **内容奖励**：对 `<answer>` 内容单独计算 Reward Model 评分

## 四、模型部署

### 4.1 OpenAI API 兼容服务

MiniMind 实现了兼容 OpenAI API 协议的服务端：

```bash
cd scripts
python serve_openai_api.py
```

启动后可通过标准 OpenAI SDK 调用：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="minimind")
response = client.chat.completions.create(
    model="minimind",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=512,
    temperature=0.85
)
print(response.choices[0].message.content)
```

**兼容的第三方 ChatUI**：FastGPT、Open-WebUI 等。

### 4.2 Streamlit WebUI

```bash
streamlit run scripts/web_demo.py
```

浏览器访问 `http://localhost:8501` 即可使用聊天界面。

### 4.3 Ollama 部署

```bash
# 直接运行
ollama run jingyaogong/minimind2
```

### 4.4 vLLM 部署

```bash
# 启动 vLLM 服务
vllm serve ./MiniMind2/ --served-model-name "minimind"
```

### 4.5 llama.cpp 部署

MiniMind 兼容 llama.cpp 推理引擎，支持 CPU 推理和量化部署。

### 4.6 模型格式转换

使用 `scripts/convert_model.py` 在 PyTorch 原生格式和 Transformers 格式之间转换：

```bash
# PyTorch → Transformers
python scripts/convert_model.py --convert_to transformers

# Transformers → PyTorch
python scripts/convert_model.py --convert_to pytorch
```

## 五、模型评估

### 5.1 第三方评测

MiniMind 在 C-Eval、C-MMLU、OpenBookQA 等榜单上进行测试。

### 5.2 主观对比

通过固定随机种子的对话测试，对比不同训练阶段模型的效果：

| 模型阶段 | 简洁性 | 信息准确性 | 对话自然度 |
|---------|--------|----------|----------|
| full_sft | 较好 | 较好 | 一般 |
| dpo | 一般 | 略有损失 | 较好 |
| reason | 有思考链 | 有提升 | 较好 |

### 5.3 YaRN 长度外推

MiniMind 支持 YaRN 算法进行 RoPE 长度外推：

```python
# 在 MiniMindConfig 中启用
config = MiniMindConfig(inference_rope_scaling=True)
# 外推长度 = factor × original_max_position_embeddings
# 默认 factor=4，即 2048 × 4 = 8192
```

## 六、完整训练流程回顾

从零到部署的完整流程：

```
1. 环境搭建 (Phase 1)
2. 准备数据 (Phase 2)
3. 预训练 → pretrain_*.pth (Phase 4)
4. SFT → full_sft_*.pth (Phase 5)
5. DPO → dpo_*.pth (Phase 8)
6. 推理蒸馏 → reason_*.pth (Phase 9)
7. 部署服务 (Phase 9)
```

**可选步骤**：
- LoRA 微调（Phase 6）：垂域迁移
- 知识蒸馏（Phase 7）：模型压缩
- PPO/GRPO/SPO（Phase 8）：RLAIF 训练

## 七、动手练习

### 练习 1：推理蒸馏训练

```bash
cd trainer
python train_distill_reason.py --epochs 1
```

### 练习 2：测试推理模型

```bash
python eval_llm.py --weight reason
```

观察模型是否会在回答前进行"思考"。

### 练习 3：启动 API 服务

```bash
cd scripts
python serve_openai_api.py
```

使用 curl 或 Python OpenAI SDK 测试 API。

### 练习 4：完整训练流程

尝试从零开始完成完整的训练流程：
1. 预训练 → 2. SFT → 3. DPO → 4. 推理蒸馏 → 5. 部署

## 八、总结与进阶

### 8.1 你已经学会了什么

通过本教程的学习，你已经掌握了：

- LLM 的完整训练流程（预训练 → SFT → RL → 部署）
- Transformer Decoder-Only 架构的核心组件
- RoPE 旋转位置编码、GQA 分组查询注意力
- MoE 混合专家架构
- LoRA 参数高效微调
- 知识蒸馏（白盒/黑盒）
- 强化学习后训练（DPO/PPO/GRPO/SPO）
- 推理模型的训练方法
- 模型部署与服务化

### 8.2 进阶方向

- **多模态**：探索 [MiniMind-V](https://github.com/jingyaogong/minimind-v) 视觉语言模型
- **更大模型**：尝试训练 768 维度的 MiniMind2
- **自定义数据**：在自己的领域数据上训练垂域模型
- **RL 探索**：尝试 GRPO 训练推理模型
- **量化部署**：使用 llama.cpp 进行 INT4/INT8 量化
- **Agent 应用**：基于 MiniMind 构建 AI Agent

### 8.3 参考资源

- [MiniMind GitHub](https://github.com/jingyaogong/minimind)
- [MiniMind ModelScope](https://www.modelscope.cn/models/gongjy/MiniMind2)
- [MiniMind HuggingFace](https://huggingface.co/jingyaogong/minimind)
- [SwanLab 训练可视化](https://swanlab.cn/)
