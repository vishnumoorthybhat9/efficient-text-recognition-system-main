@echo off
REM ===========================================================================
REM  Stop the application.
REM
REM  Use this when a server is still holding port 8000 or 5173 - usually
REM  because a previous run's window was closed without the process exiting,
REM  which makes the next start fail with "address already in use".
REM ===========================================================================

title Efficient Text Recognition - stop

echo Looking for processes holding ports 8000 (backend) and 5173 (frontend)...
echo.

set FOUND=0

for %%P in (8000 5173) do (
    for /f "tokens=5" %%A in ('netstat -ano ^| findstr /R /C:"LISTENING" ^| findstr ":%%P"') do (
        echo   Stopping PID %%A on port %%P
        taskkill /PID %%A /F >nul 2>&1
        set FOUND=1
    )
)

echo.
if "%FOUND%"=="0" (
    echo Nothing was running on those ports.
) else (
    echo Done.
)
echo.
pause
