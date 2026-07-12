"""
postprocessing.py -- Puntuacion TLV y grupos de ejecucion.

Formula:
puntuacion_tlv = prob x prob_value_contact x log(monto + 1) x prob_frescura
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime
from pathlib import Path

# Configuracion
DIST_GE = [0, 0.035, 0.087, 0.237, 0.393, 0.529, 0.664, 0.787, 0.862, 0.95, 1.0]
POSTPROCESSING_DIR = "data/postprocessed"
REPLICA_DIR = "data/replica"

Path(POSTPROCESSING_DIR).mkdir(parents=True, exist_ok=True)
Path(REPLICA_DIR).mkdir(parents=True, exist_ok=True)


def get_groups(scores, df_post):
    """
    Calcula puntuacion_tlv y asigna grupo_ejec_tlv (1-10).
    
    ** IMPLEMENTACION SIN MODIFICACIONES TAL CUAL COMO SE REQUIERE **
    
    Args:
        scores: Array con probabilidades del modelo [0, 1].
        df_post: DataFrame con columnas: grp_campecs06m, prob_value_contact, monto.
    
    Returns:
        df_post con columnas adicionales: prob, prob_frescura, puntuacion_tlv, grupo_ejec_tlv.
    """
    df_post["prob"] = scores
    df_post["prob_frescura"] = np.where(
        df_post["grp_campecs06m"] == "G1", 0.066, np.where(
            df_post["grp_campecs06m"] == "G2", 0.028, np.where(
                df_post["grp_campecs06m"] == "G3", 0.022, np.where(
                    df_post["grp_campecs06m"] == "G4", 0.008, 0.004
                )
            )
        )
    )
    df_post["prob_value_contact"] = df_post["prob_value_contact"].fillna(0.000001)
    df_post["puntuacion_tlv"] = (
        df_post["prob"]
        * df_post["prob_value_contact"]
        * np.log(df_post["monto"] + 1)
        * df_post["prob_frescura"]
    )
    df_post["grupo_ejec_tlv"] = pd.qcut(
        df_post["puntuacion_tlv"],
        q=DIST_GE,
        labels=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        duplicates="drop"
    )
    return df_post


def run_postprocessing(scores, df_post, output_path=None):
    """Wrapper de get_groups con guardado opcional a CSV."""
    result = get_groups(scores, df_post.copy())
    if output_path:
        result.to_csv(output_path, index=False)
    return result


def save_replica(df_post, table, partition, 
                 dir_s3="data/replica/s3", 
                 dir_athena="data/replica/athena", 
                 dir_onpremise="data/replica/onpremise"):
    """
    Genera el archivo de replica pipe-delimitado (|) para tres destinos.
    
    Columnas esperadas en df_post:
        - codunicocli, tipdoc, coddoc, p_codmes
        - prob (score del modelo)
        - puntuacion_tlv, grupo_ejec_tlv
        - grp_campecs06m, key_value
    """
    # Crear directorios si no existen
    for dir_path in [dir_s3, dir_athena, dir_onpremise]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Preparar datos de replica
    replica_cols = []
    replica_data = df_post.copy()
    
    # Asegurar que tenemos las columnas necesarias
    required_cols = {
        "codunicocli": "codunicocli",
        "tipdoc": "tipdoc", 
        "coddoc": "coddoc",
        "puntuacion": "puntuacion_tlv",
        "modelo": table,
        "fec_replica": datetime.now().strftime("%Y%m%d"),
        "grupo_ejec": "grupo_ejec_tlv",
        "score": "prob"
    }
    
    # Crear DataFrame de replica
    replica_result = pd.DataFrame()
    for out_col, in_col in required_cols.items():
        if out_col == "modelo":
            replica_result[out_col] = in_col
        elif out_col == "fec_replica":
            replica_result[out_col] = in_col
        elif in_col in replica_data.columns:
            replica_result[out_col] = replica_data[in_col]
        else:
            replica_result[out_col] = ""
    
    # Agregar columnas adicionales si existen
    additional_cols = [col for col in replica_data.columns 
                      if col not in required_cols.values() and col not in [
                          "prob", "prob_frescura", "prob_value_contact", 
                          "puntuacion_tlv", "grupo_ejec_tlv"
                      ]]
    for col in additional_cols[:3]:  # Limitar a 3 columnas adicionales
        if col in replica_data.columns:
            replica_result[col] = replica_data[col]
    
    # Guardar en tres destinos
    file_name = f"{table}_{partition}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    for dest_dir in [dir_s3, dir_athena, dir_onpremise]:
        dest_path = os.path.join(dest_dir, file_name)
        replica_result.to_csv(dest_path, sep="|", index=False)
        print(f"Replica guardada en {dest_path}")
    
    # Tambien guardar en postprocessed
    postproc_path = os.path.join(POSTPROCESSING_DIR, f"output_tlv_{partition}.csv")
    replica_result.to_csv(postproc_path, index=False)
    print(f"Postprocessing guardado en {postproc_path}")
    
    return replica_result


def generate_execution_report(df_postprocessed, output_dir=POSTPROCESSING_DIR):
    """
    Genera un reporte de distribucion de grupos de ejecucion.
    
    Returns:
        DataFrame con distribucion por grupo
    """
    if "grupo_ejec_tlv" not in df_postprocessed.columns:
        print("Advertencia: no hay columna 'grupo_ejec_tlv'")
        return None
    
    report = df_postprocessed["grupo_ejec_tlv"].value_counts().sort_index(ascending=False)
    report_df = pd.DataFrame({
        "grupo_ejec_tlv": report.index,
        "count": report.values,
        "pct": (report.values / report.values.sum() * 100).round(2)
    })
    
    # Guardar reporte
    report_path = os.path.join(output_dir, f"execution_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    report_df.to_csv(report_path, index=False)
    print(f"Reporte de ejecucion guardado en {report_path}")
    
    return report_df


if __name__ == "__main__":
    # Test
    df_test = pd.DataFrame({
        "grp_campecs06m": ["G1", "G2", "G3", "G4", "G1"] * 20,
        "prob_value_contact": [0.5, 0.6, 0.7, 0.8, 0.9] * 20,
        "monto": [1000, 2000, 3000, 4000, 5000] * 20,
        "codunicocli": range(100),
        "tipdoc": ["DNI"] * 100,
        "coddoc": range(100)
    })
    
    scores = np.random.uniform(0, 1, 100)
    result = run_postprocessing(scores, df_test)
    print(result.head())
    
    save_replica(result, "EC_OMNICANAL", "202412")
