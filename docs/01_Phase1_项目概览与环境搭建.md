# Phase 1: 项目概览与环境搭建

> 本阶段目标：了解 MiniMind 项目结构，搭建训练环境，体验已有模型的推理效果

## 一、项目简介

MiniMind 是一个从零开始训练超小语言模型的开源项目，核心特点：

- **极小体积**：最小版本仅 25.8M 参数（GPT-3 的 1/7000）
- **低成本**：单卡 3090 + 2 小时 + 3 块钱即可完成全流程训练
- **全流程开源**：包含预训练、SFT、LoRA、DPO、PPO、GRPO、知识蒸馏等全部代码
- **原生实现**：核心算法均使用 PyTorch 原生重构，不依赖第三方抽象接口

## 二、项目目录结构

```
minimind/
├── model/                      # 模型定义
│   ├── model_minimind.py       # MiniMind 核心模型（Dense + MoE）
│   ├── model_lora.py           # LoRA 低秩适配模块
│   ├── tokenizer.json          # 分词器配置
│   └── tokenizer_config.json   # 分词器元信息
│
├── dataset/                    # 数据集处理
│   ├── lm_dataset.py           # 数据集类定义（Pretrain/SFT/DPO/RLAIF）
│   └── dataset.md              # 数据集说明文档
│
├── trainer/                    # 训练脚本
│   ├── train_pretrain.py       # 预训练
│   ├── train_full_sft.py       # 全参数 SFT
│   ├── train_lora.py           # LoRA 微调
│   ├── train_dpo.py            # DPO 偏好优化
│   ├── train_ppo.py            # PPO 强化学习
│   ├── train_grpo.py           # GRPO 强化学习
│   ├── train_spo.py            # SPO 强化学习
│   ├── train_distillation.py   # 白盒知识蒸馏
│   ├── train_distill_reason.py # 推理模型蒸馏
│   └── trainer_utils.py        # 训练工具函数
│
├── scripts/                    # 辅助脚本
│   ├── train_tokenizer.py      # 分词器训练脚本
│   ├── web_demo.py             # Streamlit WebUI
│   ├── serve_openai_api.py     # OpenAI API 服务端
│   ├── chat_openai_api.py      # OpenAI API 客户端
│   └── convert_model.py        # 模型格式转换
│
├── eval_llm.py                 # 模型推理评估脚本
├── requirements.txt            # Python 依赖
├── out/                        # 模型权重输出目录（训练后生成）
├── checkpoints/                # 检查点目录（训练后生成）
└── docs/                       # 本教程文档目录
```

## 三、环境搭建

### 3.1 克隆项目

```bash
git clone https://github.com/jingyaogong/minimind.git
cd minimind
```

### 3.2 安装依赖

```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
```

**核心依赖说明**：

| 包名 | 用途 |
|-----|------|
| `torch` | PyTorch 深度学习框架 |
| `transformers` | HuggingFace 模型接口（用于兼容第三方框架） |
| `tokenizers` | 分词器库 |
| `accelerate` | 分布式训练加速 |
| `swanlab` | 训练可视化（国内替代 WandB） |

### 3.3 验证 CUDA 环境

```python
import torch
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"CUDA 版本: {torch.version.cuda}")
print(f"GPU 数量: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
```

如果 CUDA 不可用，请前往 [PyTorch 官网](https://pytorch.org/get-started/locally/) 下载对应 CUDA 版本的 whl 文件安装。

### 3.4 推荐硬件配置

| 配置项 | 推荐值 | 说明 |
|-------|-------|------|
| GPU | NVIDIA 3090 (24GB) | 单卡即可完成全流程 |
| CPU | 多核处理器 | 数据预处理加速 |
| RAM | 16GB+ | 数据加载缓存 |
| CUDA | 12.x | 与 PyTorch 版本匹配 |

## 四、下载已有模型

### 4.1 从 ModelScope 下载

```bash
# 下载 MiniMind2 模型（Transformers 格式）
git clone https://www.modelscope.cn/models/gongjy/MiniMind2

# 下载 PyTorch 原生权重（用于训练）
git clone https://www.modelscope.cn/models/gongjy/MiniMind2-PyTorch
```

### 4.2 从 HuggingFace 下载

```bash
git clone https://huggingface.co/jingyaogong/MiniMind2
git clone https://huggingface.co/jingyaogong/MiniMind2-Pytorch
```

### 4.3 模型文件说明

下载后的目录结构：

```
MiniMind2/                     # Transformers 格式
├── config.json                # 模型配置
├── model.safetensors          # 模型权重
├── tokenizer.json             # 分词器
└── generation_config.json     # 生成配置

MiniMind2-PyTorch/             # PyTorch 原生格式
├── pretrain_512.pth           # 预训练权重
├── full_sft_512.pth           # SFT 权重
├── dpo_512.pth                # DPO 权重
├── reason_512.pth             # 推理模型权重
└── ...
```

## 五、体验模型推理

### 5.1 命令行推理

使用 Transformers 格式模型：

```bash
python eval_llm.py --load_from ./MiniMind2
```

使用 PyTorch 原生权重：

```bash
# 将权重文件放到 ./out/ 目录下
python eval_llm.py --weight full_sft
```

**常用参数说明**：

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| `--load_from` | 模型加载路径 | `model`（原生权重） |
| `--weight` | 权重名称前缀 | `full_sft` |
| `--save_dir` | 权重目录 | `out` |
| `--max_new_tokens` | 最大生成长度 | 8192 |
| `--temperature` | 生成温度 | 0.85 |
| `--top_p` | nucleus 采样阈值 | 0.85 |

### 5.2 WebUI 推理

启动 Streamlit 界面：

```bash
streamlit run scripts/web_demo.py
```

浏览器访问 `http://localhost:8501` 即可体验对话。

### 5.3 第三方推理框架

**Ollama**：

```bash
ollama run jingyaogong/minimind2
```

**vLLM**：

```bash
vllm serve ./MiniMind2/ --served-model-name "minimind"
```

## 六、推理代码解读

### 6.1 eval_llm.py 核心流程

```python
# 1. 加载模型配置
lm_config = MiniMindConfig(hidden_size=args.hidden_size, ...)

# 2. 初始化模型和分词器
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)

# 3. 构建对话输入
messages = [{"role": "user", "content": user_input}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 4. Tokenize
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# 5. 生成回复
output_ids = model.generate(
    input_ids=input_ids,
    max_new_tokens=args.max_new_tokens,
    temperature=args.temperature,
    top_p=args.top_p,
    ...
)

# 6. 解码输出
response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
```

### 6.2 ChatML 对话模板

MiniMind 使用 ChatML 格式的对话模板：

```
<|im_start|>system
系统提示内容<|im_end|>
<|im_start|>user
用户问题<|im_end|>
<|im_start|>assistant
模型回复<|im_end|>
```

特殊 token：
- `<|im_start|>`：对话角色开始标记
- `<|im_end|>`：对话角色结束标记

## 七、动手练习

### 练习 1：验证环境

运行以下代码，确认环境正确：

```python
import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from transformers import AutoTokenizer

# 加载配置
config = MiniMindConfig()
print(f"模型配置: vocab_size={config.vocab_size}, hidden_size={config.hidden_size}")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("./model")
print(f"词表大小: {len(tokenizer)}")

# 测试分词
text = "你好，我是MiniMind"
tokens = tokenizer.encode(text)
print(f"原文: {text}")
print(f"Token IDs: {tokens}")
print(f"解码: {tokenizer.decode(tokens)}")
```

### 练习 2：体验推理

1. 下载 MiniMind2 模型
2. 使用 `eval_llm.py` 进行对话测试
3. 尝试不同的 `temperature` 和 `top_p` 参数，观察生成效果变化

### 练习 3：探索项目结构

浏览项目目录，重点关注：
- `model/model_minimind.py`：理解模型类定义
- `trainer/train_pretrain.py`：理解训练脚本结构
- `dataset/lm_dataset.py`：理解数据集类定义

## 八、常见问题

### Q1: CUDA 不可用怎么办？

检查 PyTorch 与 CUDA 版本是否匹配，必要时重新安装：

```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Q2: 模型下载速度慢？

推荐使用 ModelScope（国内镜像）：

```bash
pip install modelscope
modelscope download --model gongjy/MiniMind2
```

### Q3: 推理时显存不足？

尝试减小模型规模：
- 使用 `hidden_size=512` 的 Small 版本
- 减小 `max_new_tokens` 参数

## 九、下一阶段预告

完成本阶段后，你已经：
- ✅ 了解 MiniMind 项目结构
- ✅ 搭建好训练环境
- ✅ 体验了模型推理效果

下一阶段 [Phase 2: Tokenizer 与数据集](./02_Phase2_Tokenizer与数据集.md)，我们将深入学习：
- 分词器的工作原理
- 如何训练自定义分词器
- 各阶段数据集的格式与处理