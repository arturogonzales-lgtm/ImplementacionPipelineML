# INSTALACIÓN MANUAL - Pipeline ML E2E

## Requisitos Previos
- Python 3.9 o superior
- pip (gestor de paquetes de Python)
- Git (para control de versiones)

## Pasos de Instalación

### 1. Crear Entorno Virtual

**En Windows (CMD):**
```batch
cd c:\VSCodeStudioProjects\Tareaaaa\ImplementacionPipelineML
python -m venv venv
venv\Scripts\activate.bat
```

**En Windows (PowerShell):**
```powershell
cd c:\VSCodeStudioProjects\Tareaaaa\ImplementacionPipelineML
python -m venv venv
venv\Scripts\Activate.ps1
```

**En Linux/Mac:**
```bash
cd /ruta/a/ImplementacionPipelineML
python -m venv venv
source venv/bin/activate
```

### 2. Actualizar pip

```bash
python -m pip install --upgrade pip setuptools wheel
```

### 3. Instalar Dependencias

```bash
pip install pandas numpy scikit-learn xgboost joblib optuna mlflow lightgbm catboost boto3 python-dateutil pyyaml pytest
```

O utilizar el archivo requirements.txt:
```bash
pip install -r requirements.txt
```

### 4. Verificar Instalación

```bash
python test_installation.py
```

**Salida esperada:**
```
✓ pandas
✓ numpy
✓ sklearn
✓ xgboost
✓ optuna
✓ mlflow
✓ Todos los módulos importados exitosamente
✓ Directorios creados
```

### 5. Generar Datos de Prueba

```bash
python generate_sample_data.py
```

Esto creará:
- 10 archivos CSV en `data/raw/` (p1_extrac.csv - p10_extrac.csv)
- Estructura completa de directorios

### 6. Ejecutar Pipeline

**Modo Entrenamiento:**
```bash
python main.py --mode training --input data/raw/
```

**Modo Inferencia:**
```bash
python main.py --mode inference --input data/raw/p1_extrac.csv
```

**Modo Inferencia con Auto-Retrain:**
```bash
python main.py --mode inference --input data/raw/p1_extrac.csv --auto-retrain --psi-threshold 0.25
```

## Solución de Problemas

### Error: "ModuleNotFoundError: No module named 'optuna'"

**Solución:**
```bash
pip install optuna
```

### Error: "ModuleNotFoundError: No module named 'mlflow'"

**Solución:**
```bash
pip install mlflow
```

### Error: Venv no activado

Asegúrate de estar en el entorno virtual:
- Windows: `venv\Scripts\activate.bat` (CMD) o `venv\Scripts\Activate.ps1` (PowerShell)
- Linux/Mac: `source venv/bin/activate`

### Error: Archivos CSV no encontrados

1. Ejecuta: `python generate_sample_data.py`
2. O coloca tus archivos CSV en `data/raw/`

### Error: Permisos insuficientes

En Windows PowerShell:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Estructura de Directorios Esperada

```
ImplementacionPipelineML/
├── venv/                          # Entorno virtual
├── data/
│   ├── raw/                       # Datos brutos de entrada
│   ├── processed/                 # Datos preprocesados
│   ├── models/                    # Modelos entrenados
│   ├── monitoring/                # Reportes de monitoreo
│   ├── postprocessed/             # Datos con scoring TLV
│   └── replica/                   # Réplicas para almacenamiento
│       ├── s3/
│       ├── athena/
│       └── onpremise/
├── main.py                        # Orquestador principal
├── preprocessing.py               # Preprocesamiento
├── training.py                    # Entrenamiento e HPO
├── monitoring.py                  # Monitoreo de drift
├── postprocessing.py              # Scoring TLV
├── requirements.txt               # Dependencias
├── test_installation.py           # Script de verificación
├── generate_sample_data.py        # Generador de datos
└── README.md                      # Documentación
```

## Verificación de Instalación Exitosa

1. **Entorno Virtual Activado:**
   - Deberías ver `(venv)` en tu terminal

2. **Dependencias Instaladas:**
   - Ejecuta: `pip list` para verificar todos los paquetes

3. **Módulos Python:**
   - Ejecuta: `python test_installation.py`

4. **Datos de Prueba:**
   - Verifica que existan archivos en `data/raw/p*_extrac.csv`

5. **Ejecución:**
   - Ejecuta: `python main.py --help` para ver opciones disponibles

## Próximos Pasos

1. Revisa [README.md](README.md) para documentación completa
2. Ejecuta `python generate_sample_data.py` para crear datos de prueba
3. Ejecuta `python main.py --mode training --input data/raw/`
4. Monitorea la ejecución en la consola
5. Revisa los outputs en los directorios de `data/`

## Soporte

Para problemas adicionales:
1. Verifica que Python 3.9+ esté instalado: `python --version`
2. Asegúrate de estar en el directorio correcto
3. Revisa que el entorno virtual esté activado
4. Consulta la documentación en README.md
5. Revisa los logs en `data/monitoring/` para errores específicos

---

**Fecha de creación:** 2025-01-10
**Versión:** 1.0
**Última actualización:** 2025-01-10
