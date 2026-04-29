# MiniMind 模型性能评分测试

## 需求场景

模型已训练完成（权重位于 `MiniMind2-PyTorch/` 目录），现需对模型进行性能评分测试，评估其在标准中文语言榜单上的表现。运行环境使用项目下的 `.venv`，依赖管理使用 `uv`。

## 技术方案

### 1. 评测框架选择

根据项目 README 推荐，采用 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 框架进行客观 benchmark 评测。

评测榜单：
- **C-Eval**：中文综合能力评测
- **CMMLU**：中文大规模多任务语言理解
- 可选扩展：A-CLUE、TMMLU+

### 2. 模型格式转换

`lm_eval` 需要 HuggingFace 格式的模型。当前模型权重为 PyTorch `.pth` 格式，需使用项目自带的 `scripts/convert_model.py` 进行转换。

转换方式有两种：
- `convert_torch2transformers_minimind`：转为 MiniMind 结构的 HF 格式
- `convert_torch2transformers_llama`：转为 LlamaForCausalLM 兼容格式（推荐，第三方生态兼容性更好）

**选择 Llama 格式转换**，因为 `lm_eval` 对 Llama 架构有更好的原生支持，无需额外配置 `trust_remote_code`。

### 3. 环境准备

- `.venv` 虚拟环境已就绪，无需创建
- 使用 `uv` 安装缺失依赖：`lm-evaluation-harness` 等
- 运行命令均通过 `.venv` 环境执行

### 4. 评测模型选择

优先评测 `full_sft_768`（Base-104M，效果最好的 SFT 模型），如需可扩展到其他模型变体。

## 受影响文件

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `f:\MachineLearn\minimind\.venv\` | 新增 | uv 创建的虚拟环境 |
| `f:\MachineLearn\minimind\MiniMind2\` | 新增 | HuggingFace 格式模型输出目录 |
| `f:\MachineLearn\minimind\scripts\convert_model.py` | 修改 | 适配当前路径和模型参数 |
| `f:\MachineLearn\minimind\eval_benchmark.py` | 新增 | 一键式 benchmark 评测脚本 |

## 实现细节

### 步骤 1：创建虚拟环境并安装依赖

```bash
cd f:\MachineLearn\minimind
uv venv .venv
.venv\Scripts\activate
uv pip install -r requirements.txt
uv pip install lm-evaluation-harness
```

### 步骤 2：模型格式转换

修改 `scripts/convert_model.py` 中的路径参数，将 `full_sft_768.pth` 转换为 HuggingFace Llama 格式：

```python
lm_config = MiniMindConfig(hidden_size=768, num_hidden_layers=16, use_moe=False)
torch_path = f"./MiniMind2-PyTorch/full_sft_768.pth"
transformers_path = './MiniMind2'
convert_torch2transformers_llama(torch_path, transformers_path)
```

### 步骤 3：运行 lm_eval 评测

```bash
lm_eval --model hf --model_args pretrained=./MiniMind2,device=cuda,dtype=auto --tasks ceval_valid,cmmlu --batch_size 8
```

注意：使用 `ceval_valid` 而非 `ceval*`，因为完整 C-Eval 需要申请密钥，`ceval_valid` 是公开验证集。

### 步骤 4：编写一键评测脚本

创建 `eval_benchmark.py`，封装以下功能：
- 自动检测并转换模型格式
- 调用 lm_eval API 运行评测
- 输出格式化的评测结果（各任务准确率、总分）

## 边界条件与异常处理

1. **CUDA 不可用**：回退到 CPU 运行，但速度极慢；提示用户检查 GPU 驱动
2. **C-Eval 数据集下载**：首次运行需从 HuggingFace 下载，网络问题需配置镜像
3. **模型格式转换失败**：检查权重文件路径和模型配置是否匹配
4. **内存不足**：减小 `batch_size`，或使用 `dtype=float16`

## 数据流路径

```
.pth 权重文件 → convert_model.py → HuggingFace 格式目录
                                        ↓
lm_eval API ← 加载 HF 模型 + tokenizer
     ↓
下载 C-Eval/CMMLU 数据集 → 模型推理 → 计算准确率 → 输出评分
```

## 预期结果

根据 README 中的历史评测数据，MiniMind2 (768) 在各榜单上的参考分数：
- C-Eval: ~26.52
- CMMLU: ~24.42

实际评分可能因环境差异略有浮动。评测完成后将输出各任务的详细准确率。
