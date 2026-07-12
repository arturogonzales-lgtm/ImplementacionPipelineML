# ImplementacionPipelineML

Pipeline E2E de Machine Learning para el caso CU Venta:
Preprocesamiento, Entrenamiento con HPO (Optuna), Monitoreo (PSI/AUC/Recall por decil),
Postprocesamiento (TLV) y generación de archivos de réplica.

## Estructura

```text
.
|-- main.py
|-- preprocessing.py
|-- training.py
|-- monitoring.py
|-- postprocessing.py
|-- requirements.txt
|-- data/
|   |-- raw/
|   |-- processed/
|   |-- monitoring/
|   |-- models/
|   |-- postprocessed/
|   `-- replica/
`-- exampleNotebooks/
```

## Requisitos

- Python 3.10+
- Dependencias en `requirements.txt`
- `optuna` opcional (si deseas HPO bayesiano; si no, se usa random search)
- `mlflow` opcional (si deseas tracking completo de experimentos)

Instalación:

```bash
pip install -r requirements.txt
```

Opcional para tracking con MLflow:

```bash
pip install mlflow
```

Opcional para HPO con Optuna:

```bash
pip install optuna
```

## Ejecución rápida

Usando como entrada los CSV de `data/raw`:

```bash
python main.py --input_dir data/raw
```

## Parámetros principales

```bash
python main.py \
	--input_dir data/raw \
	--processed_dir data/processed \
	--model_dir data/models \
	--monitoring_dir data/monitoring \
	--post_path data/postprocessed/output_tlv.csv \
	--n_trials 30 \
	--nan_threshold 80 \
	--validation_partition p10 \
	--experiment_name cu_venta_e2e \
	--replica_table EC_OMNICANAL \
	--replica_partition 202412
```

## Qué hace cada módulo

- `preprocessing.py`:
	- Lee todos los CSV de entrada.
	- Elimina columnas con exceso de nulos (> `nan_threshold`).
	- Imputa nulos según reglas de negocio vistas en clase.
	- Aplica codificación de categóricas.
	- Genera `df_train.csv`, `df_test.csv`, `df_val.csv` y `metadata.json`.

- `training.py`:
	- Ejecuta búsqueda de hiperparámetros con Optuna sobre XGBoost.
	- Incluye criterio de estabilidad por validación cruzada.
	- Evalúa AUC train/test/val y decaimiento.
	- Registra experimento en MLflow (si está disponible).
	- Guarda modelo final en `data/models/model.joblib`.

- `monitoring.py`:
	- Calcula PSI de score train vs validación.
	- Calcula AUC en validación.
	- Genera recall acumulado por decil.
	- Guarda artefactos de monitoreo en `data/monitoring`.

- `postprocessing.py`:
	- Aplica fórmula TLV:
		- `puntuacion_tlv = prob x prob_value_contact x log(monto + 1) x prob_frescura`
	- Segmenta en grupos `grupo_ejec_tlv` con cuantiles `DIST_GE` del enunciado.
	- Genera archivo final y réplica pipe-delimited para 3 destinos.

- `main.py`:
	- Orquesta todo el flujo E2E.
	- Si monitoreo detecta alerta severa (PSI `ALERT`) o falla de grupos,
		ejecuta reentrenamiento automático.

## Salidas esperadas

- `data/processed/df_train.csv`
- `data/processed/df_test.csv`
- `data/processed/df_val.csv`
- `data/models/model.joblib`
- `data/models/metrics.json`
- `data/monitoring/monitoring_summary.json`
- `data/monitoring/recall_by_decile.csv`
- `data/postprocessed/output_tlv.csv`
- `data/replica/s3/*.txt`
- `data/replica/athena/*.txt`
- `data/replica/onpremise/*.txt`

## Nota sobre Notebook de referencia

La implementación sigue la arquitectura y lineamientos de `exampleNotebooks/instruciones/estructuraPipeline`
y la guía del PDF: modularidad, HPO, monitoreo y postproceso con réplica.
