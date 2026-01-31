@echo off
REM ============================================
REM run_pump_pipeline_v2.bat
REM Pipeline completo para treino do modelo v2
REM ============================================

echo ============================================
echo PUMP CNN 2D v2 - FULL PIPELINE
echo ============================================

cd /d "%~dp0"

REM Ativar venv se existir
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo [INFO] Virtual environment activated
) else (
    echo [WARN] No venv found, using system Python
)

echo.
echo [STEP 1/3] Generating dataset v3...
echo ============================================
python generators\generate_dataset_v3_pump.py
if errorlevel 1 (
    echo [ERROR] Dataset generation failed!
    pause
    exit /b 1
)

echo.
echo [STEP 2/3] Building windows...
echo ============================================
python scripts\build_windows.py
if errorlevel 1 (
    echo [ERROR] Window building failed!
    pause
    exit /b 1
)

echo.
echo [STEP 3/3] Training model v2...
echo ============================================
python training\train_pump_cnn2d_v2.py
if errorlevel 1 (
    echo [ERROR] Training failed!
    pause
    exit /b 1
)

echo.
echo ============================================
echo [SUCCESS] Pipeline completed!
echo ============================================
echo.
echo Model exported to: models\
echo.
pause
