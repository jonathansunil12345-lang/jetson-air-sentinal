#!/usr/bin/env python3
"""Train baseline regression models for PM2.5 forecasting."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import mlflow
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "air_quality_combined.parquet"
MODEL_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"


def load_dataset() -> pd.DataFrame:
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found at {PROCESSED_PATH}")
    df = pd.read_parquet(PROCESSED_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    df = df.sort_values("timestamp")
    return df


def engineer_features(df: pd.DataFrame):
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["dayofyear"] = df["timestamp"].dt.dayofyear

    # Lag features (previous 1, 3, 6 hours)
    target_col = "pm2.5" if "pm2.5" in df.columns else "pm2_5"
    if target_col not in df.columns:
        raise ValueError("Target column PM2.5 not found in dataset.")

    pollutant_cols = [target_col] + [c for c in ["pm10", "no2", "no", "nox", "nh3", "co", "so2", "o3"] if c in df.columns]
    for col in pollutant_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[target_col])
    df = df.set_index("timestamp")

    for lag in [1, 3, 6, 12, 24]:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

    df = df.reset_index()

    # Drop non-numeric identifiers from features later
    if "stationid" in df.columns:
        df = df.rename(columns={"stationid": "station_id"})
    return df, target_col, pollutant_cols


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end = pd.Timestamp("2022-12-31")
    valid_end = pd.Timestamp("2023-12-31")

    if len(df) == 0:
        raise ValueError("No samples available after feature engineering.")

    train_df = df[df["timestamp"] <= train_end]
    valid_df = df[(df["timestamp"] > train_end) & (df["timestamp"] <= valid_end)]
    test_df = df[df["timestamp"] > valid_end]

    if valid_df.empty:
        # fallback: temporal split by proportion
        n = len(df)
        if n < 3:
            raise ValueError("Not enough samples to create train/valid/test splits.")
        train_end_idx = max(int(n * 0.7), 1)
        valid_end_idx = max(int(n * 0.85), train_end_idx + 1)
        train_df = df.iloc[:train_end_idx]
        valid_df = df.iloc[train_end_idx:valid_end_idx]
        test_df = df.iloc[valid_end_idx:]
        if valid_df.empty:
            valid_df = df.iloc[[train_end_idx - 1]]
        if test_df.empty:
            test_df = df.iloc[valid_end_idx - 1 :]
    return train_df, valid_df, test_df


def train_model(train_df: pd.DataFrame, valid_df: pd.DataFrame, target_col: str) -> tuple[GradientBoostingRegressor, dict]:
    excluded = {target_col, "timestamp", "source", "station_id", "date", "aqi_bucket"}
    feature_cols = [c for c in train_df.columns if c not in excluded]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train_df[c])]
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_valid, y_valid = valid_df[feature_cols], valid_df[target_col]

    X_train = X_train.ffill().bfill().fillna(0)
    X_valid = X_valid.ffill().bfill().fillna(0)

    model = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_valid)
    metrics = compute_metrics(y_valid, preds)
    metrics["samples"] = len(y_valid)
    return model, metrics, feature_cols


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-3))) * 100
    return {"mae": mae, "rmse": rmse, "mape": mape}


def evaluate(model: GradientBoostingRegressor, df: pd.DataFrame, feature_cols: list[str], target_col: str) -> dict:
    X = df[feature_cols].ffill().bfill().fillna(0)
    preds = model.predict(X)
    metrics = compute_metrics(df[target_col], preds)
    metrics["samples"] = len(df)
    return metrics


def persist(model: GradientBoostingRegressor, feature_cols: list[str], metrics: dict) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump({"model": model, "features": feature_cols}, MODEL_DIR / "baseline_pm25.joblib")
    metrics_df = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "split"})
    metrics_df.to_csv(DOCS_DIR / "baseline_metrics.csv", index=False)


def main() -> None:
    df = load_dataset()
    df_features, target_col, _ = engineer_features(df)
    train_df, valid_df, test_df = split_dataset(df_features)

    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment("air-sentinel-baseline")

    model, valid_metrics, feature_cols = train_model(train_df, valid_df, target_col)
    test_metrics = evaluate(model, test_df, feature_cols, target_col)
    train_metrics = evaluate(model, train_df, feature_cols, target_col)

    metrics = {
        "train": train_metrics,
        "valid": valid_metrics,
        "test": test_metrics,
    }

    with mlflow.start_run(run_name="gbr_pm25_baseline"):
        mlflow.log_param("model", "GradientBoostingRegressor")
        mlflow.log_params(
            {
                "n_estimators": 500,
                "learning_rate": 0.05,
                "max_depth": 5,
                "feature_count": len(feature_cols),
            }
        )

        for split, values in metrics.items():
            for metric_name, metric_value in values.items():
                mlflow.log_metric(f"{split}_{metric_name}", float(metric_value))

        persist(model, feature_cols, metrics)
        mlflow.log_artifact(DOCS_DIR / "baseline_metrics.csv", artifact_path="reports")
        mlflow.log_artifact(MODEL_DIR / "baseline_pm25.joblib", artifact_path="models")

    print("Baseline metrics:")
    for split, values in metrics.items():
        print(split, values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline PM2.5 forecasting model.")
    _ = parser.parse_args()
    main()
