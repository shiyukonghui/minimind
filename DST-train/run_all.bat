@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM  DST Auto Train + Eval Script (Windows)
REM  Step1: Train Baseline -> Step2: Train DST -> Step3: Convert -> Step4: Eval
REM ============================================================

echo ============================================================
echo   DST Auto Train and Eval Pipeline
echo ============================================================

set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%"

set "PYTHON=.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo ERROR: Python not found: %PYTHON%
    exit /b 1
)

echo [INFO] Python: %PYTHON%
"%PYTHON%" -c "import torch; print(f'[INFO] torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

if not exist "DST-train\model" mkdir "DST-train\model"

REM ============================================================
REM  Step 1: Train Baseline Model
REM ============================================================
echo.
echo ============================================================
echo   Step 1: Train Baseline Model
echo ============================================================

if exist "DST-train\model\baseline_512.pth" (
    echo [SKIP] Baseline model already exists: DST-train\model\baseline_512.pth
    echo        Delete it first if you want to retrain
) else (
    echo [START] Training baseline model...
    "%PYTHON%" DST-train\train_baseline.py --epochs 1 --hidden_size 512 --data_path dataset/pretrain_t2t_mini.jsonl --save_dir DST-train/model --device cuda:0
    if errorlevel 1 (
        echo [ERROR] Baseline training failed!
        exit /b 1
    )
    echo [DONE] Baseline model training complete
)

REM ============================================================
REM  Step 2: Train DST Model (3 phases)
REM ============================================================
echo.
echo ============================================================
echo   Step 2: Train DST Model (3 phases)
echo ============================================================

if exist "DST-train\model\dst_recovered_512_sp20.pth" (
    echo [SKIP] DST model already exists: DST-train\model\dst_recovered_512_sp20.pth
    echo        Delete it first if you want to retrain
) else (
    echo [START] Training DST model...
    echo   Phase 1: Learn + RigL
    echo   Phase 2: Prune
    echo   Phase 3: Recover + Distill
    "%PYTHON%" DST-train\train_dst.py --phase1_epochs 1 --hidden_size 512 --phase1_data_path dataset/pretrain_t2t_mini.jsonl --phase3_data_path dataset/sft_t2t_mini.jsonl --save_dir DST-train/model --device cuda:0
    if errorlevel 1 (
        echo [ERROR] DST training failed!
        exit /b 1
    )
    echo [DONE] DST model training complete
)

REM ============================================================
REM  Step 3: Convert to HuggingFace Format
REM ============================================================
echo.
echo ============================================================
echo   Step 3: Convert Models to HuggingFace Format
echo ============================================================

echo [START] Converting baseline model...
"%PYTHON%" DST-train\convert_for_eval.py --model_type baseline --hidden_size 512 --model_dir DST-train/model --tokenizer_path ./model/
if errorlevel 1 (
    echo [ERROR] Baseline conversion failed!
    exit /b 1
)
echo [DONE] Baseline model converted

echo [START] Converting DST model...
"%PYTHON%" DST-train\convert_for_eval.py --model_type dst_recovered --hidden_size 512 --sparsity 20 --model_dir DST-train/model --tokenizer_path ./model/
if errorlevel 1 (
    echo [ERROR] DST conversion failed!
    exit /b 1
)
echo [DONE] DST model converted

REM ============================================================
REM  Step 4: Evaluate Both Models
REM ============================================================
echo.
echo ============================================================
echo   Step 4: Evaluate (C-Eval / CMMLU)
echo ============================================================

echo.
echo ------------------------------------------------------------
echo   Evaluating Baseline Model
echo ------------------------------------------------------------
"%PYTHON%" eval_benchmark.py --model_path "%CD%\DST-train\model\baseline_hf" --dataset all
if errorlevel 1 (
    echo [WARN] Baseline evaluation failed, continuing...
)

echo.
echo ------------------------------------------------------------
echo   Evaluating DST Model
echo ------------------------------------------------------------
"%PYTHON%" eval_benchmark.py --model_path "%CD%\DST-train\model\dst_hf" --dataset all
if errorlevel 1 (
    echo [WARN] DST evaluation failed
)

echo.
echo ============================================================
echo   All done! Compare the results above.
echo ============================================================

endlocal
pause
