@echo off
REM install.bat - Windows installation script for ML Pipeline

echo ==================================================
echo 🚀 Instalando Pipeline ML E2E
echo ==================================================

REM Create virtual environment
echo.
echo [1/4] Creando entorno virtual...
python -m venv venv

REM Activate environment
echo [2/4] Activando entorno...
call venv\Scripts\activate.bat

REM Install dependencies
echo [3/4] Instalando dependencias...
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

REM Create directories
echo [4/4] Creando directorios...
if not exist data\raw mkdir data\raw
if not exist data\processed mkdir data\processed
if not exist data\models mkdir data\models
if not exist data\monitoring mkdir data\monitoring
if not exist data\postprocessed mkdir data\postprocessed
if not exist data\replica\s3 mkdir data\replica\s3
if not exist data\replica\athena mkdir data\replica\athena
if not exist data\replica\onpremise mkdir data\replica\onpremise

echo.
echo ==================================================
echo ✅ Instalación completada
echo ==================================================
echo.
echo Siguientes pasos:
echo 1. Verificar instalación: python test_installation.py
echo 2. Ejecutar pipeline: python main.py --mode training --input data/raw/
echo.
echo Documentación: README.md
echo ==================================================
pause
