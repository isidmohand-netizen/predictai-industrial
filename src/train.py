"""
Training script for PredictAI Industrial — Random Forest RUL prediction.
Loads CMAPSS FD001, builds features, trains model, saves to models/ folder.

Usage:
    python src/train.py
"""

import pandas as pd
import numpy as np
import pickle
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from features import FEATURE_COLS, USEFUL_SENSORS, build_features

# ─── PATHS ───────────────────────────────────────────────────────────────────

# When running from src/, ROOT is one level up
ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ─── CONFIG ──────────────────────────────────────────────────────────────────

# Column names for CMAPSS dataset — no header in raw files
COLUMNS = (
    ['unit_id', 'cycle'] +
    ['op_1', 'op_2', 'op_3'] +
    [f'sensor_{i}' for i in range(1, 22)]
)

# Model hyperparameters
N_ESTIMATORS = 20       # 20 trees for production (lightweight)
RANDOM_STATE = 42       # Reproducibility
TEST_SIZE = 0.2         # 80% train, 20% test


def load_data(dataset: str = "FD001") -> pd.DataFrame:
    """
    Loads raw CMAPSS dataset and computes RUL for each row.
    
    Args:
        dataset: which CMAPSS subset to load (FD001, FD002, FD003, FD004)
    Returns:
        pd.DataFrame with all columns + RUL target
    """
    path = DATA_RAW / f"train_{dataset}.txt"
    
    # Load raw data — space separated, no header
    df = pd.read_csv(path, sep=r'\s+', header=None, names=COLUMNS)
    
    # Compute RUL : max_cycle - current_cycle per engine
    max_cycles = df.groupby('unit_id')['cycle'].max().reset_index()
    max_cycles.columns = ['unit_id', 'max_cycle']
    df = df.merge(max_cycles, on='unit_id')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df.drop('max_cycle', axis=1, inplace=True)
    
    print(f"Loaded {dataset} : {df['unit_id'].nunique()} engines, {len(df)} rows")
    return df


def train(dataset: str = "FD001") -> dict:
    """
    Full training pipeline : load → features → split → train → evaluate → save.
    
    Args:
        dataset: CMAPSS subset to train on
    Returns:
        dict with rmse, mae, and model_path
    """
    # 1. Load data
    df = load_data(dataset)
    
    # 2. Build features
    df_features = build_features(df)
    df_features['RUL'] = df['RUL'].values
    
    # 3. Split — always before training, never after
    X = df_features[FEATURE_COLS]
    y = df_features['RUL']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # 4. Train model
    print(f"Training Random Forest ({N_ESTIMATORS} trees)...")
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print(f"RMSE : {rmse:.2f} cycles")
    print(f"MAE  : {mae:.2f} cycles")
    
    # 6. Save model as pickle for production deployment
    model_path = MODELS_DIR / "random_forest_rul.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")
    
    # 7. Log to MLflow for experiment tracking
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("predictai-cmapss-rul")
    
    with mlflow.start_run(run_name=f"rf-{dataset}-script"):
        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("dataset", dataset)
        mlflow.log_param("n_features", len(FEATURE_COLS))
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
    
    return {"rmse": rmse, "mae": mae, "model_path": str(model_path)}


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("PredictAI Industrial — Training Pipeline")
    print("=" * 50)
    results = train(dataset="FD001")
    print(f"\nTraining complete.")
    print(f"RMSE : {results['rmse']:.2f} cycles")
    print(f"Model : {results['model_path']}")