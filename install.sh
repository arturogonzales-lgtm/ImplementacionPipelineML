#!/bin/bash
# install.sh - Script de instalación para el pipeline

echo "=================================================="
echo "🚀 Instalando Pipeline ML E2E"
echo "=================================================="

# Crear entorno virtual
echo ""
echo "[1/4] Creando entorno virtual..."
python -m venv venv

# Activar entorno
echo "[2/4] Activando entorno..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Instalar dependencias
echo "[3/4] Instalando dependencias..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Crear directorios
echo "[4/4] Creando directorios..."
mkdir -p data/{raw,processed,models,monitoring,postprocessed,replica/{s3,athena,onpremise}}

echo ""
echo "=================================================="
echo "✅ Instalación completada"
echo "=================================================="
echo ""
echo "Siguientes pasos:"
echo "1. Verificar instalación: python test_installation.py"
echo "2. Generar datos de prueba: python -c 'import pandas as pd; ...'"
echo "3. Ejecutar pipeline: python main.py --mode training --input data/raw/"
echo ""
echo "Documentación: README.md"
echo "=================================================="
