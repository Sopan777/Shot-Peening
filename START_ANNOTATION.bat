@echo off
REM =========================================================
REM  Shot Peening AI — LabelMe Annotation Launcher
REM  Double-click this file to open LabelMe with your images
REM =========================================================

SET SCRIPT_DIR=%~dp0
SET IMAGE_DIR=%SCRIPT_DIR%data\raw\images
SET ANNO_DIR=%SCRIPT_DIR%data\raw\annotations

REM Create directories if they don't exist
IF NOT EXIST "%IMAGE_DIR%" mkdir "%IMAGE_DIR%"
IF NOT EXIST "%ANNO_DIR%" mkdir "%ANNO_DIR%"

ECHO.
ECHO  =====================================================
ECHO   Shot Peening — LabelMe Annotation Tool
ECHO  =====================================================
ECHO.
ECHO  Images folder  : %IMAGE_DIR%
ECHO  Save folder    : %ANNO_DIR%
ECHO.
ECHO  HOW TO ANNOTATE:
ECHO  1. Open an image from the left panel
ECHO  2. Click "Edit" ^> "Create Polygons"
ECHO  3. Draw polygon around each NON-DEFORMED area
ECHO  4. When prompted for a label, type:  non_deformed
ECHO  5. Press Ctrl+S to save
ECHO  6. Repeat for all images
ECHO.
ECHO  TIP: Use "Edit" ^> "Create Rectangle" for simple regions
ECHO.
ECHO  Starting LabelMe...
ECHO.

labelme "%IMAGE_DIR%" --output "%ANNO_DIR%" --labels non_deformed,deformed --autosave --nodata

PAUSE
