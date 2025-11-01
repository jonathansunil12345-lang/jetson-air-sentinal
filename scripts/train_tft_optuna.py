#!/usr/bin/env python3
"""Optuna-powered sweep for feed-forward TFT surrogate (baseline) with MLflow logging."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "air_quality_combined.parquet"
MODEL_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["dayofyear"] = df["timestamp"].dt.dayofyear

    target_col = "pm2.5" if "pm2.5" in df.columns else "pm2_5"
    df["stationid"] = df.get("stationid", "unknown").fillna("unknown")

    numeric_cols = [target_col, "pm10", "no", "no2", "nox", "nh3", "co", "so2", "o3"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[target_col])
    df = df.sort_values(["stationid", "timestamp"])

    lags = [1, 3, 6, 12]
    group_cols = ["stationid"]
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df.groupby(group_cols)[target_col].shift(lag)

    df[f"{target_col}_roll_mean_12"] = (
        df.groupby(group_cols)[target_col].transform(lambda x: x.rolling(window=12, min_periods=1).mean())
    )
    df[f"{target_col}_roll_std_12"] = (
        df.groupby(group_cols)[target_col].transform(lambda x: x.rolling(window=12, min_periods=1).std().fillna(0))
    )

    df = df.dropna(subset=[f"{target_col}_lag_{lag}" for lag in lags])
    df = df.reset_index(drop=True)
    df = df.rename(columns={"stationid": "station_id"})
    return df, target_col


def split_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end = pd.Timestamp("2022-12-31")
    valid_end = pd.Timestamp("2023-12-31")

    train_df = df[df["timestamp"] <= train_end]
    valid_df = df[(df["timestamp"] > train_end) & (df["timestamp"] <= valid_end)]
    test_df = df[df["timestamp"] > valid_end]

    if valid_df.empty or test_df.empty:
        n = len(df)
        if n < 3:
            raise ValueError("Dataset too small for splitting.")
        train_df = df.iloc[: int(n * 0.7)]
        valid_df = df.iloc[int(n * 0.7) : int(n * 0.85)]
        test_df = df.iloc[int(n * 0.85) :]
    return train_df, valid_df, test_df


def to_tensor_dataset(df: pd.DataFrame, target_col: str, feature_cols: list[str]) -> TensorDataset:
    X = df[feature_cols].ffill().bfill().fillna(0.0).to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy(dtype=np.float32)
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))


class AirRegressor(pl.LightningModule):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, lr: float):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.criterion = nn.L1Loss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        self.log("train_mae", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        self.log("val_mae", loss, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def objective(trial: optuna.trial.Trial, train_ds: TensorDataset, valid_ds: TensorDataset, input_dim: int) -> float:
    hidden_dim = trial.suggest_int("hidden_dim", 32, 256, step=32)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    max_epochs = trial.study.user_attrs.get("max_epochs", 20)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    model = AirRegressor(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout, lr=lr)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    with mlflow.start_run(run_name=f"optuna_trial_{trial.number}", nested=True):
        mlflow.log_params({"hidden_dim": hidden_dim, "dropout": dropout, "lr": lr, "batch_size": batch_size})
        trainer.fit(model, train_loader, valid_loader)
        val_results = trainer.validate(model, valid_loader, verbose=False)
        val_mae = val_results[0].get("val_mae") if val_results else None
        if val_mae is None:
            raise optuna.TrialPruned()
        mlflow.log_metric("val_mae", float(val_mae))
    return float(val_mae)


def main(args: argparse.Namespace) -> None:
    df = pd.read_parquet(DATA_PATH)
    processed_df, target_col = engineer_features(df)
    train_df, valid_df, test_df = split_dataset(processed_df)

    excluded = {target_col, "timestamp", "source", "station_id", "date"}
    feature_cols = [c for c in train_df.columns if c not in excluded and pd.api.types.is_numeric_dtype(train_df[c])]
    input_dim = len(feature_cols)
    if input_dim == 0:
        raise ValueError("No numeric features available for modeling.")

    train_ds = to_tensor_dataset(train_df, target_col, feature_cols)
    valid_ds = to_tensor_dataset(valid_df, target_col, feature_cols)
    test_ds = to_tensor_dataset(test_df, target_col, feature_cols)

    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment("air-sentinel-tft")

    study = optuna.create_study(direction="minimize", study_name="tft_surrogate")
    study.set_user_attr("max_epochs", args.max_epochs)

    with mlflow.start_run(run_name="tft_optuna_sweep"):
        mlflow.log_params({"trials": args.n_trials, "max_epochs": args.max_epochs})
        study.optimize(lambda trial: objective(trial, train_ds, valid_ds, input_dim), n_trials=args.n_trials, timeout=args.timeout)

        best_trial = study.best_trial
        mlflow.log_metric("best_val_mae", best_trial.value)
        mlflow.log_params({f"best_{k}": v for k, v in best_trial.params.items()})

        # Train final model on train+valid
        combined_train = to_tensor_dataset(pd.concat([train_df, valid_df]), target_col, feature_cols)
        final_model = AirRegressor(
            input_dim=input_dim,
            hidden_dim=best_trial.params.get("hidden_dim", 128),
            dropout=best_trial.params.get("dropout", 0.1),
            lr=best_trial.params.get("lr", 1e-3),
        )
        train_loader = DataLoader(combined_train, batch_size=best_trial.params.get("batch_size", 128), shuffle=True)
        valid_loader = DataLoader(valid_ds, batch_size=best_trial.params.get("batch_size", 128), shuffle=False)
        trainer = pl.Trainer(max_epochs=args.max_epochs, accelerator="cpu", logger=False, enable_checkpointing=False)
        trainer.fit(final_model, train_loader, valid_loader)

        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
        preds = []
        targets = []
        final_model.eval()
        with torch.no_grad():
            for xb, yb in test_loader:
                preds.append(final_model(xb).numpy())
                targets.append(yb.numpy())
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)

        mae = np.mean(np.abs(targets - preds))
        rmse = np.sqrt(np.mean((targets - preds) ** 2))
        mape = np.mean(np.abs((targets - preds) / np.maximum(np.abs(targets), 1e-3))) * 100

        metrics = {"test_mae": float(mae), "test_rmse": float(rmse), "test_mape": float(mape)}
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_DIR / "tft_optuna.pt"
        torch.save({"model_state_dict": final_model.state_dict(), "feature_cols": feature_cols}, model_path)
        mlflow.log_artifact(model_path, artifact_path="models")

        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = DOCS_DIR / "tft_optuna_metrics.csv"
        pd.DataFrame([metrics]).to_csv(report_path, index=False)
        mlflow.log_artifact(report_path, artifact_path="reports")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna sweep for surrogate TFT model.")
    parser.add_argument("--n-trials", type=int, default=10, help="Number of Optuna trials to run.")
    parser.add_argument("--max-epochs", type=int, default=20, help="Training epochs per trial.")
    parser.add_argument("--timeout", type=int, default=600, help="Optional timeout for the sweep in seconds.")
    args = parser.parse_args()
    main(args)
