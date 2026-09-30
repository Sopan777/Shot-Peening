@echo off
REM =========================================================
REM  Shot Peening AI — Convert Annotations to Training Masks
REM  Run this AFTER annotating with LabelMe
REM =========================================================

SET SCRIPT_DIR=%~dp0

ECHO.
ECHO  =====================================================
ECHO   Converting LabelMe Annotations to Binary Masks
ECHO  =====================================================
ECHO.

python "%SCRIPT_DIR%tools\labelme_to_masks.py" ^
    --json_dir "%SCRIPT_DIR%data\raw\annotations" ^
    --output_dir "%SCRIPT_DIR%data\raw\masks"

IF %ERRORLEVEL% NEQ 0 (
    ECHO.
    ECHO  ERROR: Conversion failed. Check the output above.
    PAUSE
    EXIT /B 1
)

ECHO.
ECHO  =====================================================
ECHO   Splitting data into Train / Val / Test sets
ECHO  =====================================================
ECHO.

python "%SCRIPT_DIR%src\prepare_data.py" ^
    --images_dir "%SCRIPT_DIR%data\raw\images" ^
    --masks_dir "%SCRIPT_DIR%data\raw\masks" ^
    --output_dir "%SCRIPT_DIR%data"

IF %ERRORLEVEL% NEQ 0 (
    ECHO.
    ECHO  ERROR: Data split failed. Check the output above.
    PAUSE
    EXIT /B 1
)

ECHO.
ECHO  =====================================================
ECHO   Done! Your data is ready to train.
ECHO   Next step: run TRAIN_MODEL.bat
ECHO  =====================================================
ECHO.
PAUSE
