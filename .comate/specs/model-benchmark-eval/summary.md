# MiniMind 模型性能评分测试 - 总结

## 评测概览

对 MiniMind2-GRPO-768 模型（104.03M 参数，Llama 格式）进行了 C-Eval 和 CMMLU 两个中文语言榜单的性能评分测试。

## 评测结果

| 榜单 | 总准确率 | 正确/总题数 | STEM | 人文社科 |
|------|---------|-----------|------|---------|
| **C-Eval** | 25.63% | 345/1346 | 26.20% | 26.55% |
| **CMMLU** | 24.78% | 2870/11582 | 24.77% | 26.30% |

### C-Eval 各科目成绩（52 科目，1346 题验证集）

表现较好的科目：
- 离散数学: 50.00%
- 计算机网络: 42.11%
- 基础医学: 42.11%
- 法律: 41.67%
- 中学历史: 40.91%

表现较弱的科目：
- 高中历史: 5.00%
- 高中政治: 5.26%
- 高等数学: 10.53%
- 兽医学: 13.04%
- 消防工程师: 16.13%

### CMMLU 各科目成绩（67 科目，11582 题）

表现较好的科目：
- 大学医学统计: 30.19%
- 公务员考试: 28.75%
- 全球事实: 28.19%
- 机器学习: 27.87%
- 高中数学: 27.44%

表现较弱的科目：
- 大学教育: 14.95%
- 大学数学: 18.10%
- 遗传学: 19.89%
- 小学语文: 21.43%
- 计算机安全: 22.22%

## 结果分析

1. **总体水平**：模型在两个榜单上的准确率均在 25% 附近，接近四选一随机猜测的 25% 基线。这与 README 中记录的历史评测数据（C-Eval: 26.52%, CMMLU: 24.42%）基本一致。

2. **与历史数据对比**：当前 GRPO 模型的 C-Eval 得分（25.63%）略低于 README 中 full_sft 模型的参考值（26.52%），这可能是由于评测的模型变体不同（GRPO vs SFT）以及评测数据集来源的差异（zacharyxxxxcr/ceval-exam vs 官方 C-Eval 数据集）。

3. **学科差异**：STEM 和人文社科表现相近，离散数学和计算机网络等部分科目显著高于随机基线，说明模型在这些领域学到了一定的知识。

## 评测方法

- **方法**：ABCD token 概率对比法（与 lm-evaluation-harness 一致）
- **原理**：将题目和选项构建为 prompt，获取答案位置处 A/B/C/D 四个 token 的 logits 概率，选择概率最高的作为预测答案
- **模型**：MiniMind2-GRPO-768（LlamaForCausalLM 格式，FP16 精度）
- **数据集来源**：
  - C-Eval: `zacharyxxxxcr/ceval-exam`（HuggingFace 镜像，validation split）
  - CMMLU: `svjack/cmmlu`（HuggingFace 镜像，train split）

## 实施过程中遇到的问题与解决

1. **Git LFS 权重未下载**：full_sft_768.pth 等文件为 LFS 指针，改用本地已有的 grpo_768.pth
2. **HuggingFace 网络不可达**：配置 `HF_ENDPOINT=https://hf-mirror.com` 使用国内镜像
3. **C-Eval 官方数据集脚本不兼容**：新版 datasets 库不再支持 dataset script，改用 zacharyxxxxcr/ceval-exam 的 parquet 格式
4. **C-Eval split 命名差异**：官方为 `validation`，zacharyxxxxcr 版本为 `val`
5. **PyTorch 2.6 weights_only 默认值变更**：在 convert_model.py 中添加 `weights_only=False`

## 产出文件

| 文件 | 说明 |
|------|------|
| `eval_benchmark.py` | 一键式 benchmark 评测脚本 |
| `eval_results/benchmark_results.json` | 完整评测结果（JSON 格式） |
| `MiniMind2/` | HuggingFace 格式模型目录 |
| `scripts/convert_model.py` | 已修复的模型转换脚本 |
