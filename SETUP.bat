@echo off
REM ===========================================================================
REM  SETUP.bat - reinstall everything the project needs on a fresh Windows.
REM  Double-click it. Takes 15-30 minutes (PyTorch is a ~2.7 GB download).
REM  Safe to run again if something fails part-way.
REM ===========================================================================
setlocal
title Efficient Text Recognition - setup
cd /d "%~dp0."
set "ROOT=%CD%"

REM Keep big caches on D: instead of the small C: drive
set "PIP_CACHE_DIR=%ROOT%\.cache\pip"
set "npm_config_cache=%ROOT%\.cache\npm"
set "HF_HOME=%ROOT%\.cache\huggingface"
set "TMP=%ROOT%\.cache\tmp"
set "TEMP=%ROOT%\.cache\tmp"
if not exist "%TMP%" mkdir "%TMP%"

echo ===========================================================================
echo   Step 1/5 - Installing Python 3.13, Node.js LTS and Git (via winget)
echo   Click "Yes" if Windows asks for permission.
echo ===========================================================================
where winget >nul 2>&1
if errorlevel 1 (
    echo [ERROR] winget not found. Open Microsoft Store, update "App Installer",
    echo         then run this file again.
    pause & exit /b 1
)
winget install -e --id Python.Python.3.13 --scope user --accept-package-agreements --accept-source-agreements --silent
winget install -e --id OpenJS.NodeJS.LTS --accept-package-agreements --accept-source-agreements --silent
winget install -e --id Git.Git --accept-package-agreements --accept-source-agreements --silent

REM New installs are not on this window's PATH yet - find them directly.
set "PY="
if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if not defined PY if exist "%ProgramFiles%\Python313\python.exe" set "PY=%ProgramFiles%\Python313\python.exe"
if not defined PY for /f "delims=" %%i in ('where python 2^>nul') do if not defined PY set "PY=%%i"
if not defined PY (
    echo [ERROR] Python was not found after install. Restart the PC and run SETUP.bat again.
    pause & exit /b 1
)
set "PATH=%ProgramFiles%\nodejs;%ProgramFiles%\Git\cmd;%PATH%"
echo Using Python: %PY%

echo.
echo ===========================================================================
echo   Step 2/5 - Rebuilding the Python environment (.venv)
echo ===========================================================================
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import sys" >nul 2>&1
    if errorlevel 1 (
        echo Old .venv is broken - moving it aside to .venv_broken
        if exist ".venv_broken" rmdir /s /q ".venv_broken"
        move ".venv" ".venv_broken" >nul
    )
) else (
    if exist ".venv" (
        if exist ".venv_broken" rmdir /s /q ".venv_broken"
        move ".venv" ".venv_broken" >nul
    )
)
if not exist ".venv\Scripts\python.exe" "%PY%" -m venv .venv
set "VPY=%ROOT%\.venv\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip

echo.
echo ===========================================================================
echo   Step 3/5 - Installing PyTorch (GPU build, ~2.7 GB - be patient)
echo ===========================================================================
REM A newer CPU-only wheel can appear "satisfied" to pip and block CUDA install.
"%VPY%" -c "import torch; assert torch.cuda.is_available()" >nul 2>&1
if errorlevel 1 (
    echo Installing or replacing PyTorch with the CUDA 12.8 build...
    "%VPY%" -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
    if errorlevel 1 (
        echo [ERROR] PyTorch install failed - check your internet and run SETUP.bat again.
        pause & exit /b 1
    )
)
"%VPY%" -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available to PyTorch'"
if errorlevel 1 (
    echo [ERROR] CUDA is still unavailable. Check the NVIDIA driver and CUDA setup.
    pause & exit /b 1
)

echo.
echo ===========================================================================
echo   Step 4/5 - Installing the other Python packages
echo ===========================================================================
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] requirements.txt install failed - run SETUP.bat again.
    pause & exit /b 1
)

echo.
echo ===========================================================================
echo   Step 5/5 - Installing frontend packages (npm)
echo ===========================================================================
pushd frontend
if exist node_modules rmdir /s /q node_modules
call npm install
popd

echo.
echo ===========================================================================
echo   Checking the result (also saved to SETUP_RESULT.txt)
echo ===========================================================================
> SETUP_RESULT.txt (
    echo Setup finished %DATE% %TIME%
    echo --- python
    "%VPY%" --version
    echo --- torch / CUDA
    "%VPY%" -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
    echo --- key packages
    "%VPY%" -c "import fastapi, cv2, transformers, sqlalchemy; print('fastapi', fastapi.__version__, '| opencv', cv2.__version__, '| transformers', transformers.__version__, '| sqlalchemy', sqlalchemy.__version__)"
    echo --- node / npm
    node --version
    call npm --version
    echo --- git
    git --version
    echo --- nvidia driver
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    echo --- frontend
    if exist frontend\node_modules (echo node_modules OK) else (echo node_modules MISSING)
) 2>&1
type SETUP_RESULT.txt

echo.
echo ===========================================================================
echo   Done. If CUDA says False above, install the NVIDIA driver for your
echo   RTX 2050 from nvidia.com/drivers, then reboot (the app still works on CPU).
echo   You can now double-click START.bat.
echo   Once everything works you can delete the .venv_broken folder.
echo ===========================================================================
pause
