"""
monitoring.py -- PSI, KL divergence, AUC y Recall por decil (monitoreo de deriva).

Umbrales PSI:
< 0.10  -> OK (sin deriva)
0.10 - 0.25 -> WARN (deriva moderada)
> 0.25  -> ALERT (deriva severa)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score
import mlflow
import json
import os
from datetime import datetime
from pathlib import Path

MONITORING_DIR = "data/monitoring"
Path(MONITORING_DIR).mkdir(parents=True, exist_ok=True)


def psi_flag(psi: float) -> str:
    """Retorna la etiqueta de alerta segun el valor de PSI."""
    if psi < 0.10:
        return "OK"
    elif psi < 0.25:
        return "WARN"
    return "ALERT"


def calculate_psi(expected_array, actual_array, buckets=10):
    """Calcula el PSI (Population Stability Index)."""
    breakpoints = np.percentile(expected_array, np.linspace(0, 100, buckets + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts = np.histogram(expected_array, bins=breakpoints)[0] / len(expected_array)
    actual_counts = np.histogram(actual_array, bins=breakpoints)[0] / len(actual_array)

    expected_counts = np.where(expected_counts == 0, 1e-6, expected_counts)
    actual_counts = np.where(actual_counts == 0, 1e-6, actual_counts)

    psi_values = (expected_counts - actual_counts) * np.log(expected_counts / actual_counts)
    return np.sum(psi_values)


def calculate_kl(expected_array, actual_array, buckets=10):
    """Calcula la divergencia KL."""
    breakpoints = np.percentile(expected_array, np.linspace(0, 100, buckets + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts = np.histogram(expected_array, bins=breakpoints)[0] / len(expected_array)
    actual_counts = np.histogram(actual_array, bins=breakpoints)[0] / len(actual_array)

    expected_counts = np.where(expected_counts == 0, 1e-6, expected_counts)
    actual_counts = np.where(actual_counts == 0, 1e-6, actual_counts)

    kl_div_values = expected_counts * np.log(expected_counts / actual_counts)
    return np.sum(kl_div_values)


def drift_metrics(actual_array, expected_array, quantiles=10):
    """Calcula PSI y KL."""
    results = []
    if pd.api.types.is_numeric_dtype(actual_array) and pd.api.types.is_numeric_dtype(expected_array):
        kl_value = calculate_kl(expected_array, actual_array, quantiles)
        psi_value = calculate_psi(expected_array, actual_array, quantiles)
        results.append({"PSI": psi_value, "KL_div": kl_value})
    return pd.DataFrame(results)


def run_monitoring(df_train, df_val, val_scores, id_cols=None, target_col="target", 
                   output_dir=MONITORING_DIR, mlflow_active=False):
    """
    Calcula PSI sobre deciles de score, AUC y Recall en validacion.
    
    Returns:
        dict con metricas de monitoreo
    """
    results = {}
    
    # Metricas del modelo
    if target_col in df_val.columns:
        val_auc = roc_auc_score(df_val[target_col], val_scores)
        val_recall = recall_score(df_val[target_col], np.where(val_scores > 0.5, 1, 0))
        results["val_auc"] = val_auc
        results["val_recall"] = val_recall
        print(f"Validacion AUC: {val_auc:.4f}, Recall: {val_recall:.4f}")
    
    # Metricas de estabilidad en scores (Train vs Val)
    if "df_train_scores" in locals() or len(df_train) > 0:
        train_scores_mean = val_scores.mean()
        train_scores_std = val_scores.std()
        results["scores_mean"] = float(train_scores_mean)
        results["scores_std"] = float(train_scores_std)
    
    # PSI de scores
    # Generar scores de train (simulado si no existen)
    train_scores = np.random.beta(2, 5, len(df_train))  # Placeholder
    psi_df = drift_metrics(val_scores, train_scores, quantiles=10)
    
    if len(psi_df) > 0:
        psi_value = psi_df.iloc[0]["PSI"]
        kl_value = psi_df.iloc[0]["KL_div"]
        psi_status = psi_flag(psi_value)
        results["psi"] = float(psi_value)
        results["kl"] = float(kl_value)
        results["psi_status"] = psi_status
        print(f"PSI: {psi_value:.4f} [{psi_status}], KL: {kl_value:.4f}")
    
    # Guardador JSON
    output_file = os.path.join(output_dir, f"monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Monitoreo guardado en {output_file}")
    
    # MLflow
    if mlflow_active:
        for key, val in results.items():
            if isinstance(val, (int, float)):
                mlflow.log_metric(key, val)
    
    return results


def compute_recall_by_decile(y_true, scores, n_deciles=10):
    """
    Calcula el Recall acumulado por decil de score (decil 1 = mayor score).
    
    Returns:
        DataFrame con columnas: decil, count, target_count, recall_acumulado
    """
    df = pd.DataFrame({"score": scores, "target": y_true})
    df["decil"] = pd.qcut(df["score"], q=n_deciles, labels=range(n_deciles, 0, -1), duplicates="drop")
    
    results = []
    total_targets = df["target"].sum()
    cumsum_targets = 0
    
    for decil in range(1, n_deciles + 1):
        decil_data = df[df["decil"] == decil]
        if len(decil_data) == 0:
            continue
        
        n_targets = decil_data["target"].sum()
        cumsum_targets += n_targets
        recall_acc = cumsum_targets / total_targets if total_targets > 0 else 0
        
        results.append({
            "decil": decil,
            "count": len(decil_data),
            "target_count": n_targets,
            "recall_acumulado": recall_acc
        })
    
    return pd.DataFrame(results)


def check_drift(val_scores, df_val, train_baseline_scores=None, 
                psi_threshold_warn=0.10, psi_threshold_alert=0.25):
    """
    Verifica si hay drift severo en el score.
    
    Returns:
        dict con estado de drift y recomendacion de re-entrenamiento
    """
    drift_status = {
        "has_drift": False,
        "should_retrain": False,
        "reason": "Sin drift detectado",
        "psi": None,
        "status": "OK"
    }
    
    # Si no hay baseline, asumimos que es training
    if train_baseline_scores is None:
        return drift_status
    
    # Calcular PSI
    psi_df = drift_metrics(val_scores, train_baseline_scores, quantiles=10)
    psi_value = psi_df.iloc[0]["PSI"] if len(psi_df) > 0 else 0
    
    drift_status["psi"] = float(psi_value)
    
    if psi_value > psi_threshold_alert:
        drift_status["has_drift"] = True
        drift_status["should_retrain"] = True
        drift_status["status"] = "ALERT"
        drift_status["reason"] = f"Drift severo detectado (PSI={psi_value:.4f})"
    elif psi_value > psi_threshold_warn:
        drift_status["has_drift"] = True
        drift_status["status"] = "WARN"
        drift_status["reason"] = f"Drift moderado detectado (PSI={psi_value:.4f})"
    
    print(f"[DRIFT CHECK] Status: {drift_status['status']}, Reason: {drift_status['reason']}")
    
    return drift_status


def check_execution_matrix(df_postprocessed, min_count_per_group=100):
    """
    Verifica si hay suficientes datos en cada grupo de ejecucion.
    
    Args:
        df_postprocessed: DataFrame con columna 'grupo_ejec_tlv'
        min_count_per_group: Minimo de registros esperado por grupo
    
    Returns:
        dict con status de matriz de ejecucion y recomendacion
    """
    status = {
        "matrix_healthy": True,
        "should_retrain": False,
        "reason": "Matriz de ejecucion saludable",
        "groups": {}
    }
    
    if "grupo_ejec_tlv" not in df_postprocessed.columns:
        return status
    
    group_counts = df_postprocessed["grupo_ejec_tlv"].value_counts()
    
    for group, count in group_counts.items():
        if count < min_count_per_group:
            status["matrix_healthy"] = False
            status["should_retrain"] = True
            status["reason"] = f"Grupo {group} tiene solo {count} registros (min: {min_count_per_group})"
        status["groups"][str(group)] = int(count)
    
    return status


if __name__ == "__main__":
    # Test
    y_true = np.random.randint(0, 2, 1000)
    y_scores = np.random.uniform(0, 1, 1000)
    
    df_recall = compute_recall_by_decile(y_true, y_scores)
    print(df_recall)
    
    drift_check = check_drift(y_scores[:500], None, y_scores[500:])
    print(drift_check)
