#!/bin/bash
# ============================================================
#  Embedding Experiment Auto Train + Eval Script (Linux/Mac)
#  Step1: Eval Baseline A -> Step2: Train B -> Step3: Train C1
#  -> Step4: Train C2 -> Step5: Convert -> Step6: Eval -> Step7: Compare
# ============================================================

set -e

echo "============================================================"
echo "  Embedding Experiment: Multi-Embedding Comparison"
echo "============================================================"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

PYTHON="python"
EXP_DIR="embedding-exp"
MODEL_DIR="embedding-exp/model"
BASELINE_PATH="MiniMind2-Pretrain-512"

$PYTHON -c "import torch; print(f'[INFO] torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

mkdir -p "$MODEL_DIR"

# ============================================================
#  Step 1: Evaluate Baseline A (using existing pretrained model)
# ============================================================
echo ""
echo "============================================================"
echo "  Step 1: Evaluate Baseline A (existing pretrained model)"
echo "============================================================"

if [ -f "$BASELINE_PATH/model.safetensors" ]; then
    echo "[START] Evaluating baseline A..."
    $PYTHON $EXP_DIR/eval_embedding_exp.py benchmark --model_path $BASELINE_PATH --embed_type A --device cuda:0 || echo "[WARN] Baseline A evaluation failed"
    echo "[DONE] Baseline A evaluation complete"
else
    echo "[SKIP] Baseline model not found: $BASELINE_PATH"
fi

# ============================================================
#  Step 2: Train Embedding B (2D Fixed Coordinates)
# ============================================================
echo ""
echo "============================================================"
echo "  Step 2: Train Embedding B (2D Fixed Coordinates)"
echo "============================================================"

if [ -f "$MODEL_DIR/b_512.pth" ]; then
    echo "[SKIP] Model B already exists: $MODEL_DIR/b_512.pth"
else
    echo "[START] Training model B..."
    $PYTHON $EXP_DIR/train_embedding_exp.py --embed_type B --epochs 1 --hidden_size 512 --data_path dataset/pretrain_t2t_mini.jsonl --save_dir $MODEL_DIR --device cuda:0
    echo "[DONE] Model B training complete"
fi

# ============================================================
#  Step 3: Train Embedding C1 (3D Fixed Sinusoid)
# ============================================================
echo ""
echo "============================================================"
echo "  Step 3: Train Embedding C1 (3D Fixed Sinusoid)"
echo "============================================================"

if [ -f "$MODEL_DIR/c1_512.pth" ]; then
    echo "[SKIP] Model C1 already exists: $MODEL_DIR/c1_512.pth"
else
    echo "[START] Training model C1..."
    $PYTHON $EXP_DIR/train_embedding_exp.py --embed_type C1 --epochs 1 --hidden_size 512 --data_path dataset/pretrain_t2t_mini.jsonl --save_dir $MODEL_DIR --device cuda:0
    echo "[DONE] Model C1 training complete"
fi

# ============================================================
#  Step 4: Train Embedding C2 (3D Learnable Bottleneck)
# ============================================================
echo ""
echo "============================================================"
echo "  Step 4: Train Embedding C2 (3D Learnable Bottleneck)"
echo "============================================================"

if [ -f "$MODEL_DIR/c2_512.pth" ]; then
    echo "[SKIP] Model C2 already exists: $MODEL_DIR/c2_512.pth"
else
    echo "[START] Training model C2..."
    $PYTHON $EXP_DIR/train_embedding_exp.py --embed_type C2 --epochs 1 --hidden_size 512 --data_path dataset/pretrain_t2t_mini.jsonl --save_dir $MODEL_DIR --device cuda:0
    echo "[DONE] Model C2 training complete"
fi

# ============================================================
#  Step 5: Convert B/C1/C2 to HuggingFace Format
# ============================================================
echo ""
echo "============================================================"
echo "  Step 5: Convert Models to HuggingFace Format"
echo "============================================================"

echo "[START] Converting B/C1/C2 models..."
$PYTHON $EXP_DIR/convert_for_eval.py --model_dir $MODEL_DIR --embed_type all --hidden_size 512 --tokenizer_path ./model/
echo "[DONE] All models converted"

# ============================================================
#  Step 6: Evaluate All Models (C-Eval / CMMLU)
# ============================================================
echo ""
echo "============================================================"
echo "  Step 6: Evaluate All Models (C-Eval / CMMLU)"
echo "============================================================"

echo "------------------------------------------------------------"
echo "  Evaluating Model B"
echo "------------------------------------------------------------"
$PYTHON $EXP_DIR/eval_embedding_exp.py benchmark --model_path $MODEL_DIR/b_hf --embed_type B --device cuda:0 || echo "[WARN] Model B evaluation failed"

echo "------------------------------------------------------------"
echo "  Evaluating Model C1"
echo "------------------------------------------------------------"
$PYTHON $EXP_DIR/eval_embedding_exp.py benchmark --model_path $MODEL_DIR/c1_hf --embed_type C1 --device cuda:0 || echo "[WARN] Model C1 evaluation failed"

echo "------------------------------------------------------------"
echo "  Evaluating Model C2"
echo "------------------------------------------------------------"
$PYTHON $EXP_DIR/eval_embedding_exp.py benchmark --model_path $MODEL_DIR/c2_hf --embed_type C2 --device cuda:0 || echo "[WARN] Model C2 evaluation failed"

# ============================================================
#  Step 7: Semantic Analysis + Comparison Report
# ============================================================
echo ""
echo "============================================================"
echo "  Step 7: Semantic Analysis + Comparison Report"
echo "============================================================"

echo "[START] Running semantic analysis and comparison..."
$PYTHON $EXP_DIR/eval_embedding_exp.py compare --baseline_path $BASELINE_PATH --model_dir $MODEL_DIR --loss_dir $MODEL_DIR --output_dir comparison_results --device cuda:0 || echo "[WARN] Comparison generation failed"

echo ""
echo "============================================================"
echo "  All done! Check comparison_results/ for the report."
echo "============================================================"
