@echo off
REM ===========================================================================
REM  TRAIN_MODELS.bat - train both final models, then test them side by side.
REM
REM    0. Download the IAM handwriting dataset (~270 MB) and add it to training
REM    1. Quick self-test (a few minutes) - stops here if anything is wrong.
REM    2. Transformer model: fine-tune TrOCR on our data      (several hours)
REM    3. CNN model: 30-epoch training on the same data       (about 3 hours)
REM    4. Test every model on the same held-out test images
REM
REM  Keep the laptop PLUGGED IN with the LID OPEN. It is kept awake automatically.
REM  If it stops (power cut, crash), just double-click this file again:
REM  finished steps are skipped and the Transformer resumes from its last epoch.
REM  Do not run START.bat at the same time - the GPU only has 4 GB.
REM ===========================================================================
setlocal
title Training both OCR models
cd /d "%~dp0."
set "VPY=.venv\Scripts\python.exe"
set "LOGS=artifacts\logs"
set "FLAG=%LOGS%\training.running"

if not exist "%VPY%" (
    echo [ERROR] Python environment missing. Run SETUP.bat first.
    pause & exit /b 1
)
if not exist "%LOGS%" mkdir "%LOGS%"

> "%LOGS%\TRAIN_STATUS.txt" echo started %DATE% %TIME%
echo running> "%FLAG%"
start "" /b "%VPY%" scripts\keep_awake.py "%FLAG%"

echo ===========================================================================
echo  Step 0/4 - Downloading the IAM handwriting dataset (about 270 MB)
echo ===========================================================================
"%VPY%" -c "import pyarrow" >nul 2>&1 || "%VPY%" -m pip install pyarrow
"%VPY%" scripts\prepare_iam_hf.py
if errorlevel 1 goto fail_data
>> "%LOGS%\TRAIN_STATUS.txt" echo IAM dataset ready %DATE% %TIME%

echo.
echo ===========================================================================
echo  Step 1/4 - Quick self-test (a few minutes)
echo ===========================================================================
"%VPY%" -c "import torch; print('PyTorch:', torch.__version__, '| CUDA build:', torch.version.cuda); assert torch.cuda.is_available(), 'CUDA is unavailable to this PyTorch install'; print('GPU OK:', torch.cuda.get_device_name(0))"
if errorlevel 1 goto fail_gpu

if exist "artifacts\checkpoints\_smoke_trocr" rmdir /s /q "artifacts\checkpoints\_smoke_trocr"
"%VPY%" scripts\train_trocr.py --epochs 1 --max-samples 96 --val-samples 16 --log-every 4 --output-dir artifacts/checkpoints/_smoke_trocr
if errorlevel 1 goto fail_smoke
rmdir /s /q "artifacts\checkpoints\_smoke_trocr"

if exist "artifacts\checkpoints\_smoke_crnn" rmdir /s /q "artifacts\checkpoints\_smoke_crnn"
"%VPY%" scripts\train_crnn.py --config configs/experiments/crnn_full_v2_smoke.yaml --device cuda --epochs 1 --max-samples 200 --max-val-samples 64 --run-name _smoke_crnn --checkpoint-dir artifacts/checkpoints/_smoke_crnn --skip-validation-check
if errorlevel 1 (
    "%VPY%" scripts\check_training_complete.py artifacts\checkpoints\_smoke_crnn\_smoke_crnn
    if errorlevel 1 goto fail_smoke
    echo [WARNING] The smoke trainer exited abnormally after saving a verified completed run.
)
rmdir /s /q "artifacts\checkpoints\_smoke_crnn"
>> "%LOGS%\TRAIN_STATUS.txt" echo self-test passed %DATE% %TIME%
echo Self-test passed.

echo.
echo ===========================================================================
echo  Step 2/4 - Transformer model: fine-tuning TrOCR (several hours)
echo  Progress and time remaining are printed every 100 steps.
echo ===========================================================================
"%VPY%" scripts\train_trocr.py --resume
if errorlevel 1 goto fail_trocr
>> "%LOGS%\TRAIN_STATUS.txt" echo transformer trained %DATE% %TIME%

echo.
echo ===========================================================================
echo  Step 3/4 - CNN model: 30-epoch training (about 3 hours)
echo ===========================================================================
if exist "artifacts\checkpoints\crnn_full_v2\DONE" (
    echo Already trained - skipping.
) else (
    "%VPY%" scripts\train_crnn.py --config configs/experiments/crnn_full_v2.yaml
    if errorlevel 1 (
        "%VPY%" scripts\check_training_complete.py artifacts\checkpoints\crnn_full_v2\crnn_full_v2
        if errorlevel 1 goto fail_crnn
        echo [WARNING] The trainer exited abnormally after saving a verified completed run.
    )
    echo done> "artifacts\checkpoints\crnn_full_v2\DONE"
)
>> "%LOGS%\TRAIN_STATUS.txt" echo cnn trained %DATE% %TIME%

echo.
echo ===========================================================================
echo  Step 4/4 - Testing all models on the same held-out test images
echo ===========================================================================
set "MODELS=crnn_full crnn_full_v2 trocr_finetuned trocr_handwritten"
> "%LOGS%\final_eval.txt" echo ==== combined_iam test (all three datasets) ====
"%VPY%" scripts\evaluate.py --models %MODELS% --dataset combined_iam --split test >> "%LOGS%\final_eval.txt" 2>&1
if errorlevel 1 goto fail_eval
>> "%LOGS%\final_eval.txt" echo ==== IAM test only (standard benchmark) ====
"%VPY%" scripts\evaluate.py --models %MODELS% --dataset iam --split test >> "%LOGS%\final_eval.txt" 2>&1
if errorlevel 1 goto fail_eval
>> "%LOGS%\final_eval.txt" echo ==== combined test (old test set, for comparison) ====
"%VPY%" scripts\evaluate.py --models %MODELS% --dataset combined --split test >> "%LOGS%\final_eval.txt" 2>&1
if errorlevel 1 goto fail_eval
type "%LOGS%\final_eval.txt"
>> "%LOGS%\TRAIN_STATUS.txt" echo ALL DONE %DATE% %TIME%

del "%FLAG%" >nul 2>&1
echo.
echo ===========================================================================
echo  ALL DONE. Results are in artifacts\logs\final_eval.txt
echo  Tell Claude "training finished" and it will check the results and
echo  switch the app to the best two models.
echo ===========================================================================
pause
exit /b 0

:fail_data
>> "%LOGS%\TRAIN_STATUS.txt" echo FAILED: IAM download %DATE% %TIME%
echo [ERROR] IAM dataset preparation failed. Read artifacts\logs\prepare_iam.log for the actual cause.
goto end_fail
:fail_gpu
>> "%LOGS%\TRAIN_STATUS.txt" echo FAILED: no GPU %DATE% %TIME%
echo [ERROR] PyTorch cannot use CUDA. Check the version and CUDA build shown above.
echo If the CUDA build is None, run SETUP.bat or reinstall PyTorch with the CUDA command in README.md.
goto end_fail
:fail_smoke
>> "%LOGS%\TRAIN_STATUS.txt" echo FAILED: self-test %DATE% %TIME%
echo [ERROR] The self-test failed - nothing long was started. Tell Claude.
goto end_fail
:fail_trocr
>> "%LOGS%\TRAIN_STATUS.txt" echo FAILED: transformer training %DATE% %TIME%
echo [ERROR] Transformer training stopped. Run this file again to resume, or tell Claude.
goto end_fail
:fail_crnn
>> "%LOGS%\TRAIN_STATUS.txt" echo FAILED: cnn training %DATE% %TIME%
echo [ERROR] CNN training stopped. Run this file again, or tell Claude.
goto end_fail
:fail_eval
>> "%LOGS%\TRAIN_STATUS.txt" echo FAILED: evaluation %DATE% %TIME%
echo [ERROR] Testing failed - the trained models are saved. Tell Claude.
goto end_fail

:end_fail
del "%FLAG%" >nul 2>&1
pause
exit /b 1
