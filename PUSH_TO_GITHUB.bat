@echo off
REM ---------------------------------------------------------------------------
REM  Commits the project (no dataset images) to the existing GitHub repo, then
REM  uploads EFFICIENT\data.zip to a GitHub Release named "dataset-v1".
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

where git >nul 2>nul || (echo git is not installed or not on PATH. & pause & exit /b 1)
git config --global core.longpaths true
git config --global http.postBuffer 524288000

REM --- clear a stale lock left by an earlier git run -------------------------
if exist .git\index.lock (
  echo Removing stale .git\index.lock
  del /f /q .git\index.lock
)

echo Remote:
git remote get-url origin || goto :fail
echo.
git pull --rebase --autostash origin main

REM --- stop tracking anything that is now ignored (files stay on disk) -------
git rm -r --cached --quiet --ignore-unmatch major.lnk "New folder" EFFICIENT/data.zip >nul 2>nul

echo.
echo === [1/2] Committing and pushing code, configs, labels and .gitignore files ===
git add -A || goto :fail
git commit -m "Add .gitignore files for root and subfolders; ignore dataset images" || echo (nothing new to commit)
git push origin main || goto :fail

echo.
echo === [2/2] Uploading EFFICIENT\data.zip to a GitHub Release (1.37 GB, be patient) ===
where gh >nul 2>nul || (echo gh not found, skipping the release upload. & goto :done)
if not exist EFFICIENT\data.zip (echo EFFICIENT\data.zip not found, skipping. & goto :done)
gh release view dataset-v1 >nul 2>nul
if errorlevel 1 (
  gh release create dataset-v1 EFFICIENT/data.zip --title "Dataset v1" --notes "EFFICIENT dataset images (train_v2, validation_v2, test_v2). Unzip into EFFICIENT/ so the folders end up in EFFICIENT/data/." || goto :fail
) else (
  gh release upload dataset-v1 EFFICIENT/data.zip --clobber || goto :fail
)

:done
echo.
echo ======================================================
echo  Done. Code is pushed; data.zip is on the Releases page.
echo ======================================================
pause
exit /b 0

:fail
echo.
echo *** Something failed above. Copy the error and send it to Claude. ***
pause
exit /b 1
