#!/bin/bash
# DST 训练模型一键测评脚本
# 依次对基线模型和DST模型进行 C-Eval/CMMLU 评测并对比结果
#
# 使用方式：
#   bash DST-train/run_eval.sh
#
# 前置条件：
#   1. 已完成基线模型训练（train_baseline.py）
#   2. 已完成DST模型训练（train_dst.py）
#   3. 已完成模型格式转换（convert_for_eval.py）

set -e

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# 模型目录
MODEL_DIR="DST-train/model"

echo "============================================================"
echo "  DST 训练模型统一评测"
echo "============================================================"

# 检查HuggingFace格式模型是否存在
BASELINE_HF="${MODEL_DIR}/baseline_hf"
DST_HF="${MODEL_DIR}/dst_hf"

if [ ! -d "$BASELINE_HF" ]; then
    echo "错误: 基线模型HuggingFace格式不存在: $BASELINE_HF"
    echo "请先运行: python DST-train/convert_for_eval.py --model_type baseline"
    exit 1
fi

if [ ! -d "$DST_HF" ]; then
    echo "错误: DST模型HuggingFace格式不存在: $DST_HF"
    echo "请先运行: python DST-train/convert_for_eval.py --model_type dst_recovered"
    exit 1
fi

# 评测基线模型
echo ""
echo "============================================================"
echo "  评测基线模型 (Baseline)"
echo "============================================================"
python eval_benchmark.py --model_path "$BASELINE_HF" --dataset all

# 评测DST模型
echo ""
echo "============================================================"
echo "  评测DST模型 (Dynamic Sparse Training)"
echo "============================================================"
python eval_benchmark.py --model_path "$DST_HF" --dataset all

echo ""
echo "============================================================"
echo "  评测完成！请对比上方两组结果"
echo "============================================================"
