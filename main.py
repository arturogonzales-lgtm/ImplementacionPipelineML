"""
main.py -- Orquestador del pipeline ML E2E.

Ejecutar:
    # Pipeline de entrenamiento
    python main.py --mode training --input "data/raw/"
    
    # Pipeline de inferencia
    python main.py --mode inference --input "data/raw/p1_extrac.csv"
    
    # Con monitoreo y re-entrenamiento automático
    python main.py --mode inference --input "data/raw/p1_extrac.csv" --auto-retrain
"""

import argparse
import sys
import pickle
import os
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, recall_score

from preprocessing import run_preprocessing
from training import train_and_log, load_best_model, predict_inference
from monitoring import (
    run_monitoring, compute_recall_by_decile, check_drift, check_execution_matrix
)
from postprocessing import run_postprocessing, save_replica, generate_execution_report

# Configuracion
INPUT_PATH = "data/raw/"
OUTPUT_DIR = "data/processed"
MODEL_DIR = "data/models"
MONITORING_DIR = "data/monitoring"
POSTPROCESSING_DIR = "data/postprocessed"

# Crear directorios
for d in [OUTPUT_DIR, MODEL_DIR, MONITORING_DIR, POSTPROCESSING_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)


def pipeline_training(input_path):
    """
    Ejecuta el pipeline completo de entrenamiento.
    """
    print("\n" + "="*80)
    print("PIPELINE DE ENTRENAMIENTO")
    print("="*80 + "\n")
    
    # 1. Preprocesamiento
    print("[1/4] PREPROCESAMIENTO...")
    df_train, df_test, df_val, meta_prep = run_preprocessing(
        input_path, is_training=True, fit_encoders=True
    )
    print(f"✓ Preprocesamiento completado")
    print(f"  Train: {df_train.shape}, Test: {df_test.shape}, Val: {df_val.shape}\n")
    
    # 2. Entrenamiento con HPO
    print("[2/4] ENTRENAMIENTO CON OPTUNA (HPO)...")
    run_id, model, best_params, metrics = train_and_log(
        os.path.join(OUTPUT_DIR, "df_train.csv"),
        os.path.join(OUTPUT_DIR, "df_test.csv"),
        os.path.join(OUTPUT_DIR, "df_val.csv"),
        n_trials=2,
        experiment_name="cu_venta_e2e"
    )
    print(f"✓ Entrenamiento completado")
    print(f"  Run ID: {run_id}")
    print(f"  Metricas: {metrics}\n")
    
    # 3. Monitoreo
    print("[3/4] MONITOREO...")
    ID_COLS = ["p_codmes", "key_value"]
    TARGET_COL = "target"
    drop_cols = [c for c in ID_COLS + [TARGET_COL] if c in df_val.columns]
    X_val = df_val.drop(columns=drop_cols, errors="ignore")
    object_cols = X_val.select_dtypes(include=["object", "category"]).columns.tolist()
    if object_cols:
        X_val = X_val.drop(columns=object_cols)
    
    val_scores = model.predict_proba(X_val)[:, 1]
    monitoring_results = run_monitoring(df_train, df_val, val_scores)
    
    recall_df = compute_recall_by_decile(df_val[TARGET_COL], val_scores)
    print(f"✓ Monitoreo completado")
    print(f"  PSI: {monitoring_results.get('psi', 'N/A')}")
    print(f"  Recall by decile:\n{recall_df}\n")
    
    # 4. Postprocesamiento
    print("[4/4] POSTPROCESAMIENTO...")
    cols_post = ["p_codmes", "key_value", "grp_campecs06m", "monto"]
    cols_post = [c for c in cols_post if c in df_val.columns]
    
    if not cols_post or len(df_val[cols_post]) == 0:
        print("Advertencia: Columnas de postprocesamiento no encontradas. Usando defaults.")
        df_val_post = df_val[[c for c in df_val.columns if c not in drop_cols]].head()
        df_val_post["grp_campecs06m"] = "G1"
        df_val_post["monto"] = 1000
        df_val_post["prob_value_contact"] = 0.5
    else:
        df_val_post = df_val[cols_post + ["prob_value_contact"]].copy()
        df_val_post["prob_value_contact"] = df_val_post.get("prob_value_contact", 0.5).fillna(0.5)
    
    df_resultado = run_postprocessing(val_scores, df_val_post, 
                                      os.path.join(POSTPROCESSING_DIR, "output_tlv_train.csv"))
    
    exec_report = generate_execution_report(df_resultado)
    print(f"✓ Postprocesamiento completado\n")
    print(exec_report)
    
    # Guardar metadata
    metadata = {
        "pipeline": "training",
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id,
        "metrics": metrics,
        "best_params": best_params,
        "preprocessing": meta_prep
    }
    
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4, default=str)
    
    print("="*80)
    print(f"✓ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("="*80 + "\n")


def pipeline_inference(input_path, auto_retrain=False, retrain_threshold_psi=0.25):
    """
    Ejecuta el pipeline completo de inferencia.
    """
    print("\n" + "="*80)
    print("PIPELINE DE INFERENCIA")
    print("="*80 + "\n")
    
    # 1. Preprocesamiento
    print("[1/4] PREPROCESAMIENTO...")
    encoder_path = os.path.join(MODEL_DIR, "label_encoders.pkl")
    df_inference, _, _, _ = run_preprocessing(
        input_path, is_training=False, fit_encoders=False, 
        encoders_path=encoder_path
    )
    print(f"✓ Preprocesamiento completado: {df_inference.shape}\n")
    
    # 2. Cargar modelo
    print("[2/4] CARGANDO MODELO...")
    model = load_best_model()
    print(f"✓ Modelo cargado\n")
    
    # 3. Predicciones
    print("[3/4] GENERANDO PREDICCIONES...")
    inference_scores = predict_inference(df_inference, model)
    print(f"✓ Predicciones completadas: {len(inference_scores)} registros\n")
    
    # 4. Monitoreo y detección de drift
    print("[4/4] MONITOREO Y DETECCION DE DRIFT...")
    
    # Cargar baseline de entrenamiento si existe
    baseline_file = os.path.join(MODEL_DIR, "baseline_scores.npy")
    if os.path.exists(baseline_file):
        baseline_scores = np.load(baseline_file)
        drift_check = check_drift(
            inference_scores, df_inference, 
            train_baseline_scores=baseline_scores,
            psi_threshold_alert=retrain_threshold_psi
        )
    else:
        drift_check = {"has_drift": False, "should_retrain": False}
    
    print(f"✓ Drift check completado: {drift_check}\n")
    
    # 5. Postprocesamiento
    print("[5/5] POSTPROCESAMIENTO...")
    cols_post = ["grp_campecs06m", "monto", "key_value"]
    cols_post = [c for c in cols_post if c in df_inference.columns]
    
    if not cols_post:
        df_inf_post = df_inference.copy()
        df_inf_post["grp_campecs06m"] = "G1"
        df_inf_post["monto"] = 1000
        df_inf_post["prob_value_contact"] = 0.5
    else:
        df_inf_post = df_inference[cols_post + ["prob_value_contact"]].copy() if "prob_value_contact" in df_inference.columns else df_inference[cols_post].copy()
        df_inf_post["prob_value_contact"] = df_inf_post.get("prob_value_contact", 0.5).fillna(0.5)
    
    # Obtener particion o mes del dataset de origen para nombrar los archivos de salida
    partition_val = "unknown"
    if "partition" in df_inference.columns and len(df_inference) > 0:
        partition_val = str(df_inference["partition"].iloc[0]).strip()
    elif "p_codmes" in df_inference.columns and len(df_inference) > 0:
        partition_val = str(int(df_inference["p_codmes"].iloc[0]))
    elif "p_fecinformacion" in df_inference.columns and len(df_inference) > 0:
        partition_val = str(int(df_inference["p_fecinformacion"].iloc[0]))
        
    output_filename = f"output_tlv_inference_{partition_val}.csv"
    print(f"Los resultados se guardaran en: {output_filename}")
    
    df_resultado = run_postprocessing(inference_scores, df_inf_post,
                                      os.path.join(POSTPROCESSING_DIR, output_filename))

    # Verificar matriz de ejecucion
    matrix_check = check_execution_matrix(df_resultado, min_count_per_group=50)
    
    print(f"✓ Postprocesamiento completado")
    print(f"✓ Matriz de ejecucion: {matrix_check}\n")
    
    # Guardar en un archivo csv matrix_check con referencia a la particion de origen
    matrix_data = {
        "partition": [partition_val],
        "matrix_healthy": [matrix_check.get("matrix_healthy", True)],
        "should_retrain": [matrix_check.get("should_retrain", False)],
        "reason": [matrix_check.get("reason", "")]
    }
    for g in range(1, 11):
        matrix_data[f"group_{g}"] = [matrix_check.get("groups", {}).get(str(g), 0)]
        
    df_matrix_report = pd.DataFrame(matrix_data)
    matrix_report_path = os.path.join(MONITORING_DIR, f"matrix_check_{partition_val}.csv")
    df_matrix_report.to_csv(matrix_report_path, index=False)
    print(f"✓ Reporte de matriz de ejecucion guardado en: {matrix_report_path}")
    
    # Guardar replica
    print("Guardando replicas...")
    partition = datetime.now().strftime("%Y%m%d")
    save_replica(df_resultado, table="EC_OMNICANAL", partition=partition)
    print("✓ Replicas guardadas\n")
    
    # Guardar baseline si no existe
    if not os.path.exists(baseline_file):
        np.save(baseline_file, inference_scores)
        print(f"Baseline guardado en {baseline_file}")
    
    # RE-ENTRENAMIENTO AUTOMATICO
    if auto_retrain and (drift_check.get("should_retrain") or matrix_check.get("should_retrain")):
        print("\n" + "="*80)
        print("⚠  RE-ENTRENAMIENTO AUTOMATICO DISPARADO")
        print("="*80 + "\n")
        print(f"Razon (Drift): {drift_check.get('reason', 'N/A')}")
        print(f"Razon (Matrix): {matrix_check.get('reason', 'N/A')}\n")
        
        # Ejecutar entrenamiento
        pipeline_training(INPUT_PATH)
        
        print("\n" + "="*80)
        print("✓ RE-ENTRENAMIENTO COMPLETADO")
        print("="*80 + "\n")
    
    # Resumen final
    print("="*80)
    print(f"✓ INFERENCIA COMPLETADA")
    print(f"  Registros procesados: {len(df_resultado)}")
    print(f"  Drift status: {drift_check.get('status', 'OK')}")
    print(f"  Matriz de ejecucion saludable: {matrix_check.get('matrix_healthy', True)}")
    print("="*80 + "\n")


def main():
    """Orquestador principal."""
    parser = argparse.ArgumentParser(
        description="Pipeline ML E2E para CU Venta (Entrenamiento + Inferencia)"
    )
    parser.add_argument(
        "--mode",
        choices=["training", "inference"],
        required=True,
        help="Modo de ejecucion"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=INPUT_PATH,
        help="Ruta de entrada (directorio para training, CSV para inference)"
    )
    parser.add_argument(
        "--auto-retrain",
        action="store_true",
        help="Habilitar re-entrenamiento automatico en caso de drift"
    )
    parser.add_argument(
        "--psi-threshold",
        type=float,
        default=0.25,
        help="Umbral de PSI para disparar re-entrenamiento"
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == "training":
            pipeline_training(args.input)
        elif args.mode == "inference":
            pipeline_inference(args.input, auto_retrain=args.auto_retrain, 
                             retrain_threshold_psi=args.psi_threshold)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
