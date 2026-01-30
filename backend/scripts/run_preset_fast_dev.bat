@echo off
setlocal

REM === Fast Dev: iterações rápidas para validar pipeline ===
set "DT_WIN_SET=96,128,192"
set "DT_ENSEMBLE=1"
set "DT_TS2VEC_EPOCHS=3"
set "DT_PRETRAIN_EPOCHS=3"
set "DT_USE_CBAM=0"
set "DT_EPOCHS=30"
set "DT_BATCH_SIZE=256"
set "DT_EXPORT_TFLITE=1"

if not exist models mkdir models

echo [Preset Fast Dev] a treinar...
python train_tf.py

echo.
echo [OK] Terminado (Fast Dev).
pause
endlocal
