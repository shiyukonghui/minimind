@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM  Embedding Experiment Auto Train + Eval Script (Windows)
REM  Step1: Eval Baseline A -> Step2: Train B -> Step3: Train C1
REM  -> Step4: Train C2 -> Step5: Convert -> Step6: Eval -> Step7: Compare
REM ============================================================

echo ============================================================
echo   Embedding Experiment: Multi-Embedding Comparison
echo ============================================================

set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%"

set "PYTHON=.venv\Scripts\python.exe"
set "EXP_DIR=embedding-exp"
set "MODEL_DIR=embedding-exp\model"
set "BASELINE_PATH=MiniMind2-Pretrain-512"

if not exist "%PYTHON%" (
    echo ERROR: Python not found: %PYTHON%
    exit /b 1
)

echo [INFO] Python: %PYTHON%
"%PYTHON%" -c "import torch; print(f'[INFO] torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"

REM ============================================================
REM  Step 1: Evaluate Baseline A (using existing pretrained model)
REM ============================================================
echo.
echo ============================================================
echo   Step 1: Evaluate Baseline A (existing pretrained model)
echo ============================================================

if exist "%BASELINE_PATH%\model.safetensors" (
    echo [START] Evaluating baseline A...
    "%PYTHON%" %EXP_DIR%\eval_embedding_exp.py benchmark --model_path %BASELINE_PATH% --embed_type A --device cuda:0
    if errorlevel 1 (
        echo [WARN] Baseline A evaluation failed, continuing...
    )
    echo [DONE] Baseline A evaluation complete
) else (
    echo [SKIP] Baseline model not found: %BASELINE_PATH%
)

REM ============================================================
REM  Step 2: Train Embedding B (2D Fixed Coordinates)
REM ============================================================
echo.
echo ============================================================
echo   Step 2: Train Embedding B (2D Fixed Coordinates)
echo ============================================================

if exist "%MODEL_DIR%\b_512.pth" (
    echo [SKIP] Model B already exists: %MODEL_DIR%\b_512.pth
    echo        Delete it first if you want to retrain
) else (
    echo [START] Training model B...
    "%PYTHON%" %EXP_DIR%\train_embedding_exp.py --embed_type B --epochs 1 --hidden_size 512 --data_path dataset/pretrain_t2t_mini.jsonl --save_dir %MODEL_DIR% --device cuda:0
    if errorlevel 1 (
        echo [ERROR] Model B training failed!
        exit /b 1
    )
    echo [DONE] Model B training complete
)

REM ============================================================
REM  Step 3: Train Embedding C1 (3D Fixed Sinusoid)
REM ============================================================
echo.
echo ============================================================
echo   Step 3: Train Embedding C1 (3D Fixed Sinusoid)
echo ============================================================

if exist "%MODEL_DIR%\c1_512.pth" (
    echo [SKIP] Model C1 already exists: %MODEL_DIR%\c1_512.pth
    echo        Delete it first if you want to retrain
) else (
    echo [START] Training model C1...
    "%PYTHON%" %EXP_DIR%\train_embedding_exp.py --embed_type C1 --epochs 1 --hidden_size 512 --data_path dataset/pretrain_t2t_mini.jsonl --save_dir %MODEL_DIR% --device cuda:0
    if errorlevel 1 (
        echo [ERROR] Model C1 training failed!
        exit /b 1
    )
    echo [DONE] Model C1 training complete
)

REM ============================================================
REM  Step 4: Train Embedding C2 (3D Learnable Bottleneck)
REM ============================================================
echo.
echo ============================================================
echo   Step 4: Train Embedding C2 (3D Learnable Bottleneck)
echo ============================================================

if exist "%MODEL_DIR%\c2_512.pth" (
    echo [SKIP] Model C2 already exists: %MODEL_DIR%\c2_512.pth
    echo        Delete it first if you want to retrain
) else (
    echo [START] Training model C2...
    "%PYTHON%" %EXP_DIR%\train_embedding_exp.py --embed_type C2 --epochs 1 --hidden_size 512 --data_path dataset/pretrain_t2t_mini.jsonl --save_dir %MODEL_DIR% --device cuda:0
    if errorlevel 1 (
        echo [ERROR] Model C2 training failed!
        exit /b 1
    )
    echo [DONE] Model C2 training complete
)

REM ============================================================
REM  Step 5: Convert B/C1/C2 to HuggingFace Format
REM ============================================================
echo.
echo ============================================================
echo   Step 5: Convert Models to HuggingFace Format
REM ============================================================

echo [START] Converting B/C1/C2 models...
"%PYTHON%" %EXP_DIR%\convert_for_eval.py --model_dir %MODEL_DIR% --embed_type all --hidden_size 512 --tokenizer_path ./model/
if errorlevel 1 (
    echo [ERROR] Model conversion failed!
    exit /b 1
)
echo [DONE] All models converted

REM ============================================================
REM  Step 6: Evaluate All Models (C-Eval / CMMLU)
REM ============================================================
echo.
echo ============================================================
echo   Step 6: Evaluate All Models (C-Eval / CMMLU)
REM ============================================================

echo.
echo ------------------------------------------------------------
echo   Evaluating Model B
echo ------------------------------------------------------------
"%PYTHON%" %EXP_DIR%\eval_embedding_exp.py benchmark --model_path %MODEL_DIR%\b_hf --embed_type B --device cuda:0
if errorlevel 1 (
    echo [WARN] Model B evaluation failed, continuing...
)

echo.
echo ------------------------------------------------------------
echo   Evaluating Model C1
echo ------------------------------------------------------------
"%PYTHON%" %EXP_DIR%\eval_embedding_exp.py benchmark --model_path %MODEL_DIR%\c1_hf --embed_type C1 --device cuda:0
if errorlevel 1 (
    echo [WARN] Model C1 evaluation failed, continuing...
)

echo.
echo ------------------------------------------------------------
echo   Evaluating Model C2
echo ------------------------------------------------------------
"%PYTHON%" %EXP_DIR%\eval_embedding_exp.py benchmark --model_path %MODEL_DIR%\c2_hf --embed_type C2 --device cuda:0
if errorlevel 1 (
    echo [WARN] Model C2 evaluation failed, continuing...
)

REM ============================================================
REM  Step 7: Semantic Analysis + Comparison Report
REM ============================================================
echo.
echo ============================================================
echo   Step 7: Semantic Analysis + Comparison Report
REM ============================================================

echo [START] Running semantic analysis and comparison...
"%PYTHON%" %EXP_DIR%\eval_embedding_exp.py compare --baseline_path %BASELINE_PATH% --model_dir %MODEL_DIR% --loss_dir %MODEL_DIR% --output_dir comparison_results --device cuda:0
if errorlevel 1 (
    echo [WARN] Comparison generation failed
)

echo.
echo ============================================================
echo   All done! Check comparison_results/ for the report.
echo ============================================================

endlocal
pause
