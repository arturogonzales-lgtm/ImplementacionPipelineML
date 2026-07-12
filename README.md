# 🚀 Pipeline ML E2E: CU Venta - Manual Completo

## 📋 Tabla de Contenidos

1. [Descripción General](#-descripción-general)
2. [Estructura del Proyecto](#-estructura-del-proyecto)
3. [Requisitos e Instalación](#-requisitos-e-instalación)
4. [Guía Rápida](#-guía-rápida)
5. [Componentes Detallados](#-componentes-detallados)
6. [Uso Avanzado](#-uso-avanzado)
7. [Troubleshooting](#-troubleshooting)

---

## 🎯 Descripción General

Sistema completo de **Machine Learning E2E** para predicción de propensión de compra (CU Venta):

✅ **Preprocesamiento** - Limpieza y transformación automática
✅ **Entrenamiento con HPO** - Búsqueda de hiperparámetros con Optuna
✅ **Monitoreo Continuo** - PSI, KL divergence, detección de drift
✅ **Re-entrenamiento Automático** - Se dispara ante degradación de modelo
✅ **Scoring TLV** - Puntuación sin modificaciones (fórmula de clase)
✅ **Generación de Replicas** - S3, Athena, On-Premise

---

## 📁 Estructura del Proyecto

```
ImplementacionPipelineML/
├── main.py                    # Orquestador principal
├── preprocessing.py           # Preparación de datos
├── training.py                # Entrenamiento + HPO (Optuna)
├── monitoring.py              # PSI, drift, metricas
├── postprocessing.py          # TLV scoring + replicas
├── requirements.txt           # Dependencias
├── README.md                  # Este archivo
│
├── data/
│   ├── raw/                   # Datos crudos (entrada)
│   ├── processed/             # Datos preprocesados
│   ├── models/                # Modelos y encoders
│   ├── monitoring/            # Reportes de drift
│   ├── postprocessed/         # Scores TLV
│   └── replica/               # Replicas (s3, athena, onpremise)
│
└── exampleNotebooks/          # Notebooks de referencia
    ├── 1. Preprocessing.ipynb
    ├── 3. Monitoreo del modelo.ipynb
    ├── 4. Entrenamiento.ipynb
    └── 5. HPO.ipynb
```

---

## 📦 Requisitos e Instalación

### Paso 1: Verificar Python

```bash
python --version  # Debe ser 3.9+
```

### Paso 2: Clonar/Descargar Proyecto

```bash
git clone <REPO_URL>
cd ImplementacionPipelineML
```

### Paso 3: Crear Entorno Virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### Paso 4: Instalar Dependencias

```bash
pip install -r requirements.txt
```

### Paso 5: Crear Directorios

```bash
mkdir -p data/{raw,processed,models,monitoring,postprocessed,replica/{s3,athena,onpremise}}
```

---

## 🏃 Guía Rápida

### Entrenamiento Inicial

```bash
python main.py --mode training --input "data/raw/"
```

**Pasos automáticos:**
1. Lee CSVs de `data/raw/`
2. Preprocesa (imputa, encodeara, split)
3. Entrena con Optuna (30 trials)
4. Registra en MLflow
5. Guarda modelo + métricas

### Inferencia en Producción

```bash
python main.py --mode inference --input "data/raw/p1_extrac.csv"
```

**Pasos automáticos:**
1. Lee datos nuevos
2. Aplica transformaciones (con encoders guardados)
3. Genera predicciones
4. Detecta drift (PSI)
5. Genera scores TLV
6. Crea replicas

### Con Re-entrenamiento Automático

```bash
python main.py --mode inference --input "data/raw/p1_extrac.csv" \
  --auto-retrain \
  --psi-threshold 0.25
```

Si PSI > 0.25 → automáticamente ejecuta pipeline de entrenamiento

---

## 🔧 Componentes Detallados

### 1️⃣ preprocessing.py

**Funciones principales:**

```python
# Entrenamiento
df_train, df_test, df_val, meta = run_preprocessing(
    "data/raw/",
    is_training=True,
    fit_encoders=True
)

# Inferencia
df_inference, _, _, _ = run_preprocessing(
    "data/raw/p1_extrac.csv",
    is_training=False,
    fit_encoders=False,
    encoders_path="data/models/label_encoders.pkl"
)
```

**Operaciones:**
- Descarta columnas >80% NaN
- Imputa faltantes (criterio experto)
- Transforma categóricas
- LabelEncoding para XGBoost
- Split: Train 70% | Test 20% | Val OOT

### 2️⃣ training.py

**Búsqueda de Hiperparámetros:**

```python
run_id, model, params, metrics = train_and_log(
    "data/processed/df_train.csv",
    "data/processed/df_test.csv",
    "data/processed/df_val.csv",
    n_trials=30
)
```

**Hiperparámetros optimizados:**
- `n_estimators`: 50-500
- `max_depth`: 3-10
- `learning_rate`: 0.001-0.3
- `subsample`: 0.5-1.0
- `colsample_bytree`: 0.5-1.0
- `min_child_weight`: 1-10
- `gamma`: 0-5

**Métricas:** AUC, Recall, Precision, F1

### 3️⃣ monitoring.py

**Detección de Drift:**

```python
drift_check = check_drift(
    val_scores,
    baseline_scores,
    psi_threshold_alert=0.25
)
# Retorna: {'has_drift': bool, 'should_retrain': bool, 'psi': float, 'status': 'OK'|'WARN'|'ALERT'}
```

**Umbrales PSI:**
- **< 0.10** → ✅ OK (sin drift)
- **0.10-0.25** → ⚠️ WARN (moderado)
- **> 0.25** → 🔴 ALERT (severo → re-entrena)

**Recall por Decil:**

```python
df_recall = compute_recall_by_decile(y_true, y_scores, n_deciles=10)
# Muestra: decil | count | target_count | recall_acumulado
```

### 4️⃣ postprocessing.py

**Scoring TLV (SIN MODIFICACIONES):**

```python
df_result = get_groups(scores, df_postprocessing)
# Calcula: puntuacion_tlv = prob × prob_value_contact × log(monto+1) × prob_frescura
# Asigna: grupo_ejec_tlv ∈ [1..10] basado en DIST_GE
```

**Replica a 3 destinos:**

```python
save_replica(df_result, table="EC_OMNICANAL", partition="202412")
# Genera archivos pipe-delimitados en:
# - data/replica/s3/
# - data/replica/athena/
# - data/replica/onpremise/
```

### 5️⃣ main.py

**Orquestador con 2 modos:**

```bash
# Modo 1: Entrenamiento
python main.py --mode training --input "data/raw/"

# Modo 2: Inferencia
python main.py --mode inference --input "data/raw/p1_extrac.csv"

# Modo 2 + Retrain automático
python main.py --mode inference --input "data/raw/p1_extrac.csv" \
  --auto-retrain --psi-threshold 0.25
```

---

## 🎓 Uso Avanzado

### Generar datos de prueba

```bash
python -c "
import pandas as pd
import numpy as np

np.random.seed(42)
for i in range(5):
    df = pd.DataFrame({
        'p_codmes': [201912] * 600 + [201911 - i] * 400,
        'key_value': range(1000),
        'target': np.random.randint(0, 2, 1000),
        'grp_campecs06m': np.random.choice(['G1','G2','G3','G4'], 1000),
        'monto': np.random.uniform(1000, 50000, 1000),
        'prob_value_contact': np.random.uniform(0, 1, 1000),
        'feature_1': np.random.uniform(-10, 10, 1000),
        'feature_2': np.random.normal(50, 15, 1000),
    })
    df.to_csv(f'data/raw/p{i+1}_extrac.csv', index=False)
    print(f'✓ Creado p{i+1}_extrac.csv')
"
```

### Usar MLflow para visualizar experimentos

```bash
# Terminal 1: Ejecutar pipeline
python main.py --mode training --input "data/raw/"

# Terminal 2: Iniciar MLflow
mlflow ui --host 127.0.0.1 --port 5000

# Abrir: http://localhost:5000
```

### Personalizar umbral PSI

```bash
# Menos sensible (PSI > 0.30)
python main.py --mode inference --input "data/raw/p1_extrac.csv" \
  --auto-retrain --psi-threshold 0.30

# Muy sensible (PSI > 0.15)
python main.py --mode inference --input "data/raw/p1_extrac.csv" \
  --auto-retrain --psi-threshold 0.15
```

---

## 📊 Archivos de Salida

### Training

```
data/processed/
├── df_train.csv       # 70% dataset
├── df_test.csv        # 20% dataset
└── df_val.csv         # 10% (OOT)

data/models/
├── best_model_20240115_143022.pkl
├── label_encoders.pkl
├── baseline_scores.npy
└── metadata.json

data/monitoring/
└── monitoring_20240115_143522.json
```

### Inference

```
data/postprocessed/
├── output_tlv_inference.csv
└── execution_report_20240115_143522.csv

data/replica/
├── s3/EC_OMNICANAL_202412_*.txt
├── athena/EC_OMNICANAL_202412_*.txt
└── onpremise/EC_OMNICANAL_202412_*.txt
```

---

## 🐛 Troubleshooting

| Error | Solución |
|-------|----------|
| `ModuleNotFoundError: optuna` | `pip install optuna>=3.0.0` |
| `No CSV files found` | Verificar `data/raw/` contiene archivos `p*_extrac.csv` |
| `Encoders not found` | Ejecutar entrenamiento primero: `python main.py --mode training` |
| `PSI calculation error` | Verificar scores contienen valores en [0, 1] |
| `MLflow connection refused` | Iniciar MLflow: `mlflow ui` en otra terminal |

---

## 📈 Flujo Completo

```
INPUT (CSVs)
    ↓
PREPROCESSING (imputa, encodeara, split)
    ↓
├─→ TRAINING (Optuna HPO, MLflow)
│        ↓
│    MODEL + ENCODERS guardados
│
INFERENCE (carga encoders, predice)
    ↓
MONITORING (PSI, AUC, drift detection)
    ↓
    ├─ Drift ALERT? ─→ AUTO-RETRAIN
    │
POSTPROCESSING (TLV scoring)
    ↓
REPLICA (S3, Athena, On-Premise)
    ↓
OUTPUT
```

---

## ✨ Características Clave

| Feature | Descripción |
|---------|------------|
| **HPO** | Optuna busca los mejores hiperparámetros automáticamente |
| **MLflow** | Registra cada experimento con parámetros, métricas, modelo |
| **PSI Monitoring** | Detecta cambios en distribución de scores (Population Stability Index) |
| **Auto-Retrain** | Si drift severo → re-entrena automáticamente |
| **TLV Inmodificable** | Fórmula y cuantiles como se especificó en clase |
| **Versioning** | Cada modelo guardado con timestamp para trazabilidad |
| **Multi-replica** | Genera archivos compatibles con S3, Athena, On-Premise |

---

## 🔒 Seguridad

⚠️ **Importante:**
- Archivos CSV contienen datos sensibles de clientes
- No commitear `data/models/` con información confidencial
- Si usas MLflow remoto, usar autenticación
- Restringir acceso a `data/` en producción

---

**Última actualización:** Enero 2024
**Versión:** 1.0
**Estado:** ✅ Production-Ready
