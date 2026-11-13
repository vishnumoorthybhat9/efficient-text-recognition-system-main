@echo off
REM ===========================================================================
REM  Efficient Text Recognition from Images Using Deep Learning and
REM  Self-Supervised Learning
REM
REM  Double-click this file to start the whole application.
REM  It opens two windows (backend and frontend) and then your browser.
REM  Close those two windows, or run STOP.bat, to shut everything down.
REM ===========================================================================

title Efficient Text Recognition - launcher

REM "%~dp0" is this file's folder and always ends with a backslash, so the
REM trailing "." stops the closing quote being escaped by it.
cd /d "%~dp0."

echo ===========================================================================
echo   Efficient Text Recognition - starting
echo ===========================================================================
echo.

REM --- check the Python environment exists -----------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Python environment not found at .venv\Scripts\python.exe
    echo.
    echo Create it first:
    echo     python -m venv .venv
    echo     .venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    echo     .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM --- check the frontend dependencies exist ---------------------------------
if not exist "frontend\node_modules" (
    echo [WARN] Frontend dependencies are missing. Installing them now.
    echo        This takes a couple of minutes the first time.
    echo.
    pushd frontend
    call npm install
    popd
    echo.
)

REM --- warn if something is already holding our ports ------------------------
REM A previous run still running is the most common reason a start appears to
REM fail: the new server exits at once with "address already in use".
netstat -ano | findstr /R /C:"LISTENING" | findstr ":8000" >nul 2>&1
if %errorlevel%==0 (
    echo [WARN] Port 8000 is already in use - the backend may already be running.
    echo        Run STOP.bat first if the application misbehaves.
    echo.
)

REM --- start the two servers -------------------------------------------------
REM `start /D` sets each window's working directory, which avoids nesting
REM quotes inside the command string - a classic source of cmd breakage.
echo Starting the backend  ... http://127.0.0.1:8000
start "OCR Backend" /D "%~dp0." cmd /k .venv\Scripts\python.exe scripts\run_api.py

echo Waiting for the backend to load its model registry (about 30 seconds)...
timeout /t 30 /nobreak >nul

echo Starting the frontend ... http://localhost:5173
start "OCR Frontend" /D "%~dp0frontend" cmd /k npm run dev

timeout /t 8 /nobreak >nul

echo.
echo Opening the dashboard in your browser...
start "" http://localhost:5173

echo.
echo ===========================================================================
echo   Running.
echo.
echo     Dashboard   http://localhost:5173
echo     API docs    http://127.0.0.1:8000/docs
echo.
echo   Two other windows have opened: "OCR Backend" and "OCR Frontend".
echo   Close both, or run STOP.bat, to shut the application down.
echo ===========================================================================
echo.
pause
