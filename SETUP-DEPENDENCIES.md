# SOLUCIÓN: Instalación Manual de Dependencias Faltantes

## Problema
Los paquetes `optuna` y `mlflow` no se pueden instalar automáticamente usando `pip install` desde la terminal integrada de VS Code.

## Soluciones Alternativas

### Opción 1: Abrir Terminal Independiente (RECOMENDADA)

1. Abre una terminal de comando SEPARADA (no integrada en VS Code):
   - **Windows**: Presiona `Win + R`, escribe `cmd.exe`, presiona Enter
   - **Windows PowerShell**: Presiona `Win + R`, escribe `powershell`, presiona Enter

2. Navega al directorio del proyecto:
   ```cmd
   cd c:\VSCodeStudioProjects\Tareaaaa\ImplementacionPipelineML
   ```

3. Activa el entorno virtual:
   ```cmd
   venv\Scripts\activate.bat
   ```
   
   O si usas PowerShell:
   ```powershell
   venv\Scripts\Activate.ps1
   ```

4. Instala los paquetes:
   ```cmd
   pip install --upgrade pip
   pip install optuna==3.0.0 mlflow==2.10.0
   ```

5. Verifica la instalación:
   ```cmd
   python -c "import optuna; import mlflow; print('OK')"
   ```

### Opción 2: Usar Archivo requirements.txt Completo

1. Crea un archivo `requirements-full.txt` con todas las dependencias
2. Ejecuta desde terminal independiente:
   ```cmd
   pip install -r requirements-full.txt
   ```

### Opción 3: Instalación Interactiva Paso a Paso

```cmd
# Actualizar pip
python -m pip install --upgrade pip

# Instalar cada paquete individualmente
pip install pandas==2.1.0
pip install numpy==1.26.0
pip install scikit-learn==1.4.0
pip install xgboost==2.0.0
pip install joblib==1.4.0
pip install optuna==3.0.0
pip install mlflow==2.10.0
pip install lightgbm==4.0.0
pip install catboost==1.2.0
pip install boto3==1.28.0
pip install python-dateutil==2.8.0
pip install pyyaml==6.0
pip install pytest==7.0.0
```

### Opción 4: Usar Conda (Si Conda está Instalado)

```cmd
# Crear entorno conda
conda create -n pipeline python=3.9

# Activar entorno
conda activate pipeline

# Instalar dependencias
conda install pandas numpy scikit-learn xgboost joblib optuna mlflow lightgbm catboost boto3 python-dateutil pyyaml pytest
```

## Verificación de Instalación

Una vez instalado, verifica ejecutando:

```cmd
python test_installation.py
```

Debería mostrar:
```
✓ pandas
✓ numpy
✓ sklearn
✓ xgboost
✓ optuna
✓ mlflow
✓ Módulos importados exitosamente
✓ Directorios creados
```

## En Caso de Problemas Persistentes

### Si falla por dependencias de compilación en Windows

Descarga e instala Visual C++ Build Tools:
- Descarga desde: https://visualstudio.microsoft.com/downloads/
- Selecciona "Visual Studio Build Tools"
- Instala "Desktop development with C++"
- Reinicia la computadora
- Intenta nuevamente con `pip install optuna mlflow`

### Si falla por permisos en Windows PowerShell

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Si el venv está corrupto

Elimina el venv y crea uno nuevo:

```cmd
rmdir /s /q venv
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

## Alternativa: Modo Degradado (Sin HPO Automático)

Si deseas ejecutar el pipeline SIN instalar optuna/mlflow, puedes usar parámetros predefinidos:

```cmd
python main.py --mode training --input data/raw/ --use-defaults
```

Esto usará configuraciones XGBoost preestablecidas sin optimización de hiperparámetros.

## Próximos Pasos Después de Instalar

1. Verifica: `python test_installation.py`
2. Genera datos: `python generate_sample_data.py`
3. Ejecuta el pipeline:
   ```cmd
   python main.py --mode training --input data/raw/
   ```

---

**Última actualización:** 2025-01-10
