@echo off
REM =========================================================
REM  Shot Peening AI — Train the Model
REM  Run this AFTER running PREPARE_DATA.bat
REM =========================================================

SET SCRIPT_DIR=%~dp0

ECHO.
ECHO  =====================================================
ECHO   Shot Peening — Training U-Net Model
ECHO  =====================================================
ECHO.
ECHO  Architecture : U-Net + ResNet-34
ECHO  Epochs       : 100 (with early stopping)
ECHO  Data dir     : %SCRIPT_DIR%data
ECHO  Output dir   : %SCRIPT_DIR%outputs
ECHO.
ECHO  TIP: Monitor training in TensorBoard with:
ECHO    tensorboard --logdir outputs\logs\tensorboard
ECHO.

python "%SCRIPT_DIR%src\train.py" ^
    --data_dir "%SCRIPT_DIR%data" ^
    --output_dir "%SCRIPT_DIR%outputs" ^
    --encoder resnet34 ^
    --img_size 512 ^
    --batch_size 8 ^
    --epochs 100 ^
    --patience 15

IF %ERRORLEVEL% NEQ 0 (
    ECHO.
    ECHO  Training failed. Common fixes:
    ECHO  - Reduce --batch_size to 4 if you get "out of memory"
    ECHO  - Reduce --img_size to 256 for less GPU memory
    PAUSE
    EXIT /B 1
)

ECHO.
ECHO  Training complete! Best model saved to outputs\models\best_model.pth
ECHO  Next step: run PREDICT.bat to test on new images.
ECHO.
PAUSE
