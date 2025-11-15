@echo off
REM One-Command Setup for AI Assignments
REM Creates conda environment 'cs221' with all dependencies

echo ============================================================
echo AI ASSIGNMENTS - ONE-COMMAND SETUP
echo ============================================================
echo.

REM Check if conda is available
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Conda not found!
    echo.
    echo Please install Miniconda or Anaconda first:
    echo   https://docs.conda.io/en/latest/miniconda.html
    echo.
    echo Or use pip instead:
    echo   pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo [OK] Conda found
echo.

REM Check if environment already exists
conda env list | findstr /C:"cs221" >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [WARNING] Environment 'cs221' already exists!
    set /p REMOVE="Do you want to remove and recreate? (y/N): "
    if /i "%REMOVE%"=="y" (
        echo.
        echo Removing old environment...
        conda env remove -n cs221 -y
        echo [OK] Old environment removed
    ) else (
        echo.
        echo Skipping environment creation.
        echo To update existing environment, run:
        echo   conda env update -f environment.yml --prune
        echo.
        pause
        exit /b 0
    )
)

echo.
echo ============================================================
echo Creating conda environment 'cs221'...
echo ============================================================
echo.

conda env create -f environment.yml

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Failed to create environment!
    echo.
    echo Try manual installation:
    echo   conda create -n cs221 python=3.12 -y
    echo   conda activate cs221
    echo   pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo SETUP COMPLETE!
echo ============================================================
echo.
echo [OK] Environment 'cs221' created successfully!
echo.
echo Next steps:
echo   1. Activate the environment:
echo      conda activate cs221
echo.
echo   2. Start Jupyter:
echo      jupyter notebook AI_Assignments.ipynb
echo.
echo   3. Or run the GUI launcher:
echo      python main.py
echo.
echo ============================================================
echo.
pause
