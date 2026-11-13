@echo off
REM ---------------------------------------------------------------------------
REM  Commits everything (code + datasets) and pushes to the existing GitHub repo
REM  (origin = vishnumoorthybhat9/efficient-text-recognition-system-main).
REM  Data is pushed in several parts so no single push goes over GitHub's 2 GB limit.
REM  Double-click this file from the project folder.
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

where git >nul 2>nul || (echo git is not installed or not on PATH. & pause & exit /b 1)
git config --global core.longpaths true
git config --global http.postBuffer 524288000

echo Remote:
git remote get-url origin || goto :fail
echo.

REM --- get any commits already on GitHub first -------------------------------
git pull --rebase --autostash origin main

REM --- stop tracking files that are now in .gitignore ------------------------
git rm -r --cached --quiet --ignore-unmatch major.lnk "New folder" EFFICIENT/data.zip >nul 2>nul

REM --- 1. code, config and .gitignore files ----------------------------------
echo.
echo === [1/4] Code, config and .gitignore files ===
git add -A -- . ":(exclude)data" ":(exclude)EFFICIENT/data" || goto :fail
git commit -m "Add .gitignore files for root and subfolders" || echo (nothing new to commit)
git push origin main || goto :fail

REM --- 2. data folder --------------------------------------------------------
echo.
echo === [2/4] data/ ===
git add -A -- data || goto :fail
git commit -m "Add data folder" || echo (nothing new to commit)
git push origin main || goto :fail

REM --- 3. EFFICIENT validation + test + CSVs ---------------------------------
echo.
echo === [3/4] EFFICIENT validation/test sets and CSVs ===
git add -A -- EFFICIENT/data/validation_v2 EFFICIENT/data/test_v2 EFFICIENT/data/*.csv || goto :fail
git commit -m "Add EFFICIENT validation and test sets" || echo (nothing new to commit)
git push origin main || goto :fail

REM --- 4. EFFICIENT train set ------------------------------------------------
echo.
echo === [4/4] EFFICIENT train set (big, be patient) ===
git add -A -- EFFICIENT/data || goto :fail
git commit -m "Add EFFICIENT train set" || echo (nothing new to commit)
git push origin main || goto :fail

echo.
echo ======================================================
echo  Done. Everything is committed and pushed.
echo ======================================================
pause
exit /b 0

:fail
echo.
echo *** Something failed above. Copy the error and send it to Claude. ***
pause
exit /b 1
