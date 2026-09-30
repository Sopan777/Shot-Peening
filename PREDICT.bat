@echo off
REM =========================================================
REM  Shot Peening AI — Run Predictions on New Images
REM =========================================================

SET SCRIPT_DIR=%~dp0
SET MODEL_PATH=%SCRIPT_DIR%outputs\models\best_model.pth
SET INPUT_DIR=%SCRIPT_DIR%data\raw\images
SET OUTPUT_DIR=%SCRIPT_DIR%results

ECHO.
ECHO  =====================================================
ECHO   Shot Peening — Surface Deformation Detection
ECHO  =====================================================
ECHO.
ECHO  Model    : %MODEL_PATH%
ECHO  Input    : %INPUT_DIR%
ECHO  Output   : %OUTPUT_DIR%
ECHO.
ECHO  For each image, outputs will be saved:
ECHO    *_mask.png     - Binary mask (white = non-deformed)
ECHO    *_overlay.png  - Color overlay (red = non-deformed)
ECHO    *_report.txt   - Coverage statistics
ECHO    batch_summary.json - Summary for all images
ECHO.

IF NOT EXIST "%MODEL_PATH%" (
    ECHO  ERROR: Model not found at %MODEL_PATH%
    ECHO  Please run TRAIN_MODEL.bat first.
    PAUSE
    EXIT /B 1
)

python "%SCRIPT_DIR%src\predict.py" ^
    --model_path "%MODEL_PATH%" ^
    --input "%INPUT_DIR%" ^
    --output_dir "%OUTPUT_DIR%" ^
    --threshold 0.5

ECHO.
ECHO  Results saved to: %OUTPUT_DIR%
ECHO.
PAUSE
