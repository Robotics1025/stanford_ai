@echo off
REM Quick Launcher - Opens Jupyter Notebook at Cell 22 (GUI Menu)

echo ============================================================
echo AI ASSIGNMENTS - QUICK LAUNCHER
echo ============================================================
echo.

REM Check if notebook exists
if not exist "AI_Assignments.ipynb" (
    echo [ERROR] AI_Assignments.ipynb not found!
    echo Make sure you're in the AI ASSIGNMENT folder.
    pause
    exit /b 1
)

echo Starting Jupyter Notebook...
echo.
echo Instructions:
echo   1. Notebook will open in your browser
echo   2. Navigate to Cell 22
echo   3. Click 'Run' or press Shift+Enter
echo   4. The GUI menu will appear!
echo.
echo ============================================================
echo.

REM Activate conda environment if it exists
conda activate cs221 2>nul

REM Open Jupyter notebook
jupyter notebook AI_Assignments.ipynb

pause
