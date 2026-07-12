"""
preprocessing.py -- Limpieza y transformacion del dataset CU Venta.

Produce: df_train.csv, df_test.csv, df_val.csv
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# Configuracion
NAN_THRESHOLD = 80
VALIDATION_CODMES = 202212.0
TEST_SIZE = 0.30
RANDOM_STATE = 123
OUTPUT_DIR = "data/processed"
ENCODER_DIR = "data/models"

# Crear directorios si no existen
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(ENCODER_DIR).mkdir(parents=True, exist_ok=True)


def calculo_nan(df, threshold=NAN_THRESHOLD):
    """Identifica columnas con exceso de NaN."""
    cols_drop = []
    for c in df.columns:
        porc_nan = df[c].isna().sum() / len(df[c]) * 100
        if porc_nan > threshold:
            print(f"Descartando {c}: {porc_nan:.2f}% NaN")
            cols_drop.append(c)
    return cols_drop


def imputar_variables(df):
    """Imputa valores faltantes segun criterio experto."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if "cnt" in col or "num" in col or "ctd" in col or "flg" in col:
            df[col] = df[col].fillna(0)
    if "edad" in df.columns:
        df["edad"] = df["edad"].fillna(df["edad"].median())
    if "ubigeo_buro" in df.columns:
        df["ubigeo_buro"] = df["ubigeo_buro"].fillna("Otros")
    if "grp_riesgociiu" in df.columns:
        df["grp_riesgociiu"] = df["grp_riesgociiu"].fillna("grupo_0")
    if "region" in df.columns:
        df["region"] = df["region"].fillna("Otro")
    if "grp_camptot06m" in df.columns:
        df["grp_camptot06m"] = df["grp_camptot06m"].fillna("Otro")
    if "grp_campecs06m" in df.columns:
        df["grp_campecs06m"] = df["grp_campecs06m"].fillna("Otro")
    df_numeric = df.select_dtypes(include=[np.number])
    for col in df_numeric.columns:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(0)
    return df


def transformar_categoricas(df):
    """Transforma variables categoricas."""
    if "grp_riesgociiu" in df.columns:
        df["grp_riesgociiu"] = pd.Series(np.where(
            df["grp_riesgociiu"].isin(["grupo_2", "grupo_3", "grupo_9", "grupo_8", "grupo_1"]),
            "grupo_11", df["grp_riesgociiu"]
        ))
    if "seg_un" in df.columns:
        df["seg_un"] = pd.Series(np.where(
            df["seg_un"].isin([0, 3]), 0, df["seg_un"]
        ))
    return df


def encodear_variables(df, encoders=None, fit=True):
    """Encodea variables categoricas."""
    features_encoder = [
        "grp_camptottlv06m", "grp_campecstlv06m", "grp_camptot06m", 
        "grp_campecs06m", "region", "grp_riesgociiu", "ubigeo_buro"
    ]
    features_encoder = [c for c in features_encoder if c in df.columns]
    if encoders is None:
        encoders = {}
    for col in features_encoder:
        if fit:
            encoder = LabelEncoder()
            encoder.fit(df[col].astype(str))
            encoders[col] = encoder
        else:
            encoder = encoders.get(col)
            if encoder is None:
                encoder = LabelEncoder()
                encoder.fit(df[col].astype(str))
                encoders[col] = encoder
        df[col] = encoder.transform(df[col].astype(str))
    return df, encoders


def asignar_tipos_datos(df):
    """Asigna tipos de datos."""
    if "target" in df.columns:
        df["target"] = df["target"].astype("int32")
    if "monto" in df.columns:
        df["monto"] = df["monto"].astype("float64")
    if "p_codmes" in df.columns:
        df["p_codmes"] = df["p_codmes"].astype("float64")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ["target", "monto", "p_codmes"]:
            df[col] = df[col].astype("float64")
    return df


def run_preprocessing(data_path, nan_threshold=NAN_THRESHOLD, fit_encoders=True, 
                      encoders_path=None, is_training=True):
    """Ejecuta el pipeline completo de preprocesamiento."""
    if os.path.isdir(data_path):
        csv_files = list(Path(data_path).glob("*.csv"))
        dfs = []
        for f in csv_files:
            try:
                dfs.append(pd.read_csv(f))
            except Exception as e:
                print(f"Error: {f}: {e}")
        df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    else:
        df = pd.read_csv(data_path)
    
    print(f"Dataset original: {df.shape}")
    
    if "p_codmes" not in df.columns:
        if "p_fecinformacion" in df.columns:
            print("Deriving 'p_codmes' from 'p_fecinformacion'...")
            df["p_codmes"] = (df["p_fecinformacion"].astype(float) // 100).astype(float)
        else:
            print("Warning: Neither 'p_codmes' nor 'p_fecinformacion' found. Using default 202212.0.")
            df["p_codmes"] = 202212.0
    
    if is_training and len(df) > 50000:
        print(f"Downsampling dataset from {len(df)} to 50000 rows for development speed...")
        df_val_rows = df[df["p_codmes"] == VALIDATION_CODMES]
        df_other_rows = df[df["p_codmes"] != VALIDATION_CODMES]
        sampled_val = df_val_rows.sample(n=min(len(df_val_rows), 10000), random_state=RANDOM_STATE) if len(df_val_rows) > 0 else df_val_rows
        sampled_other = df_other_rows.sample(n=min(len(df_other_rows), 40000), random_state=RANDOM_STATE) if len(df_other_rows) > 0 else df_other_rows
        df = pd.concat([sampled_val, sampled_other], ignore_index=True)
        print(f"New shape after downsampling: {df.shape}")
        
    cols_drop = calculo_nan(df, nan_threshold)
    df = df.drop(columns=cols_drop, errors="ignore")
    print(f"Dataset procesado: {df.shape}")
    
    df = imputar_variables(df)
    df = transformar_categoricas(df)
    
    if fit_encoders or encoders_path is None:
        df, encoders = encodear_variables(df, fit=True)
        encoder_file = os.path.join(ENCODER_DIR, "label_encoders.pkl")
        with open(encoder_file, "wb") as f:
            pickle.dump(encoders, f)
    else:
        with open(encoders_path, "rb") as f:
            encoders = pickle.load(f)
        df, _ = encodear_variables(df, encoders=encoders, fit=False)
    
    df = asignar_tipos_datos(df)
    
    metadata = {"dropped": cols_drop, "n_samples": len(df), "n_features": df.shape[1]}
    
    if is_training and "p_codmes" in df.columns:
        df_val = df[df["p_codmes"] == VALIDATION_CODMES].copy()
        df_main = df[df["p_codmes"] != VALIDATION_CODMES].copy()
        df_train, df_test = train_test_split(
            df_main, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        df_train.to_csv(os.path.join(OUTPUT_DIR, "df_train.csv"), index=False)
        df_test.to_csv(os.path.join(OUTPUT_DIR, "df_test.csv"), index=False)
        df_val.to_csv(os.path.join(OUTPUT_DIR, "df_val.csv"), index=False)
        return df_train, df_test, df_val, metadata
    else:
        df.to_csv(os.path.join(OUTPUT_DIR, "df_inference.csv"), index=False)
        return df, None, None, metadata
