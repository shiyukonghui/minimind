# 多种向量化表示方法实验 - 完成总结

## 实现概览

在 `embedding-exp/` 目录下实现了完整的多种向量化表示方法对比实验框架，包括模型定义、训练、评测和全流程脚本。

## 文件清单

| 文件 | 说明 |
|------|------|
| `embedding-exp/model_embedding.py` | 多嵌入方案模型定义（A/B/C1/C2四种方案） |
| `embedding-exp/train_embedding_exp.py` | 统一训练脚本（--embed_type 切换B/C1/C2） |
| `embedding-exp/eval_embedding_exp.py` | 评测+语义分析脚本（benchmark/semantic/compare三子命令） |
| `embedding-exp/convert_for_eval.py` | PyTorch权重转HuggingFace格式 |
| `embedding-exp/run_all.bat` | Windows全流程脚本 |
| `embedding-exp/run_all.sh` | Linux全流程脚本 |

## 关键实现细节

### 模型架构
- `MiniMindEmbeddingConfig` 继承 `MiniMindConfig`，新增 `embed_type` 和 `coord_dim` 参数
- `MiniMindEmbeddingModel` 继承 `MiniMindModel`，重写嵌入层（四分支实现）
- `MiniMindEmbeddingForCausalLM` 处理权重绑定：方案A保持绑定，B/C1/C2断开
- 新增 `get_embedding_weights()` / `get_raw_embedding_weights()` / `get_embed_param_count()` 方法用于语义分析

### 四种嵌入方案验证结果

| 方案 | 总参数 | 嵌入相关参数 | 权重绑定 |
|------|--------|-------------|---------|
| A (高维基线) | 25.830M | 3,276.8K | 是 |
| B (二维固定坐标) | 25.844M | 3,277.8K | 否 |
| C1 (三维固定模式) | 25.851M | 3,278.3K | 否 |
| C2 (三维可学习瓶颈) | 25.851M | 3,297.5K | 否 |

> 四种方案的非嵌入参数量一致（22.553M），差异仅来自嵌入层。

### 训练脚本
- 仅覆盖 B/C1/C2（方案A使用 `MiniMind2-Pretrain-512` 现有预训练模型）
- 保存路径包含方案标识：`{save_dir}/{embed_type}_512.pth`
- 同时保存 loss 历史到 `{embed_type}_loss_history.json`

### 评测脚本
- **benchmark 子命令**：C-Eval + CMMLU 基准评测
- **semantic 子命令**：最邻近分析 + t-SNE可视化 + 参数效率比
- **compare 子命令**：四组方案完整对比，自动生成汇总报告和loss对比图

### 全流程脚本
7步流水线：评测基线A → 训练B → 训练C1 → 训练C2 → 格式转换 → 基准评测 → 语义分析+对比报告
支持跳过已完成的步骤（检测模型文件是否存在）

## 使用方式

```bash
# 一键运行全流程（Windows）
embedding-exp\run_all.bat

# 一键运行全流程（Linux）
bash embedding-exp/run_all.sh

# 单独训练某个方案
python embedding-exp/train_embedding_exp.py --embed_type B --epochs 1

# 单独评测
python embedding-exp/eval_embedding_exp.py benchmark --model_path ./MiniMind2-Pretrain-512 --embed_type A

# 单独语义分析
python embedding-exp/eval_embedding_exp.py semantic --model_path ./model/b_hf --embed_type B

# 四组对比
python embedding-exp/eval_embedding_exp.py compare --baseline_path ./MiniMind2-Pretrain-512 --model_dir ./model
```
