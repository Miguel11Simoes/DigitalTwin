@echo off
setlocal

REM === Max Accuracy: máximo detalhe/realismo e melhor acerto possível ===
set "DT_WIN_SET=96,128,192,256"
set "DT_ENSEMBLE=5"
set "DT_TS2VEC_EPOCHS=20"
set "DT_PRETRAIN_EPOCHS=20"
set "DT_USE_CBAM=1"
set "DT_EPOCHS=200"
set "DT_BATCH_SIZE=256"
set "DT_EXPORT_TFLITE=1"

if not exist models mkdir models

echo [Preset Max Accuracy] a treinar...
python train_tf.py

echo.
echo [OK] Terminado (Max Accuracy).
pause
endlocal

