"""
training.py -- Entrenamiento XGBoost + busqueda de hiperparametros con Optuna.

Registra parametros, metricas y modelo en MLflow.
"""

import optuna
import xgboost as xgb
import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from datetime import datetime
from pathlib import Path

# Configuracion
ID_COLS = ["p_codmes", "key_value"]
TARGET_COL = "target"
MODEL_DIR = "data/models"
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)


def _xy(df):
    """Separa features y target."""
    drop = [c for c in ID_COLS + [TARGET_COL] if c in df.columns]
    X = df.drop(columns=drop, errors="ignore")
    # Descarta cualquier columna tipo object o categórica restante
    object_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if object_cols:
        print(f"Descantado columnas no numéricas desprovistas de encoders: {object_cols}")
        X = X.drop(columns=object_cols)
    return X, df[TARGET_COL]


def train_and_log(train_path, test_path, val_path=None, n_trials=30, 
                  experiment_name="cu_venta_e2e", save_best_model=True):
    """
    Busca hiperparametros con Optuna y registra el mejor modelo en MLflow.
    
    Args:
        train_path: Ruta a df_train.csv
        test_path: Ruta a df_test.csv
        val_path: Ruta a df_val.csv (opcional, para validacion final)
        n_trials: Numero de trials de Optuna
        experiment_name: Nombre del experimento en MLflow
        save_best_model: Si True, guarda el modelo en disco
    
    Returns:
        (run_id, model, best_params, metrics)
    """
    # Cargar datasets
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    X_train, y_train = _xy(df_train)
    X_test, y_test = _xy(df_test)
    
    if val_path and os.path.exists(val_path):
        df_val = pd.read_csv(val_path)
        X_val, y_val = _xy(df_val)
    else:
        X_val, y_val = None, None
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Callback para pruning
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": 123,
            "early_stopping_rounds": 20,
        }
        
        model = xgb.XGBClassifier(**params, verbosity=0)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Metricas en test
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        
        return auc
    
    # Optimizacion con Optuna
    print(f"Iniciando busqueda de hiperparametros ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_params = study.best_params
    print(f"Mejores parametros: {best_params}")
    
    # Entrenar modelo final con mejores parametros
    best_params["use_label_encoder"] = False
    best_params["eval_metric"] = "logloss"
    best_params["random_state"] = 123
    
    model = xgb.XGBClassifier(**best_params, verbosity=0)
    model.fit(X_train, y_train)
    
    # Calcula metricas
    metrics = {}
    
    # Train metrics
    y_train_pred = model.predict_proba(X_train)[:, 1]
    metrics["train_auc"] = roc_auc_score(y_train, y_train_pred)
    metrics["train_recall"] = recall_score(y_train, model.predict(X_train))
    
    # Test metrics
    y_test_pred = model.predict_proba(X_test)[:, 1]
    metrics["test_auc"] = roc_auc_score(y_test, y_test_pred)
    metrics["test_recall"] = recall_score(y_test, model.predict(X_test))
    metrics["test_precision"] = precision_score(y_test, model.predict(X_test))
    metrics["test_f1"] = f1_score(y_test, model.predict(X_test))
    
    # Val metrics
    if X_val is not None:
        y_val_pred = model.predict_proba(X_val)[:, 1]
        metrics["val_auc"] = roc_auc_score(y_val, y_val_pred)
        metrics["val_recall"] = recall_score(y_val, model.predict(X_val))
    
    # MLflow logging
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        # Log params
        mlflow.log_params(best_params)
        
        # Log metrics
        for metric_name, metric_val in metrics.items():
            mlflow.log_metric(metric_name, metric_val)
        
        # Log model
        mlflow.xgboost.log_model(
            model, 
            "model",
            registered_model_name=f"{experiment_name}_xgb"
        )
        
        run_id = run.info.run_id
        print(f"MLflow run_id: {run_id}")
    
    # Guardar modelo en disco
    if save_best_model:
        model_file = os.path.join(MODEL_DIR, f"best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
        pickle.dump(model, open(model_file, "wb"))
        print(f"Modelo guardado en {model_file}")
        
        # Guardar parametros
        params_file = os.path.join(MODEL_DIR, "best_params.pkl")
        pickle.dump(best_params, open(params_file, "wb"))
    
    return run_id, model, best_params, metrics


def load_best_model(model_path=None):
    """Carga el mejor modelo guardado."""
    if model_path is None:
        # Encontrar el modelo mas reciente
        models = list(Path(MODEL_DIR).glob("best_model_*.pkl"))
        if not models:
            raise FileNotFoundError("No se encontraron modelos entrenados")
        model_path = max(models, key=os.path.getctime)
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"Modelo cargado desde {model_path}")
    return model


def predict_inference(df_inference, model=None):
    """Realiza predicciones en modo inferencia."""
    if model is None:
        model = load_best_model()
    
    X, _ = _xy(df_inference)
    predictions = model.predict_proba(X)[:, 1]
    
    return predictions


if __name__ == "__main__":
    run_id, model, params, metrics = train_and_log(
        "data/processed/df_train.csv",
        "data/processed/df_test.csv",
        "data/processed/df_val.csv"
    )
    print(f"\nEntrenamiento completado.")
    print(f"Metricas: {metrics}")
