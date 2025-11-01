#!/usr/bin/env python3
"""Train a Temporal Fusion Transformer using PyTorch Forecasting with MLflow logging."""
from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import torch
from lightning.pytorch import Trainer, seed_everything
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "air_quality_bias_corrected.parquet"
MODEL_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"


def load_and_prepare_data(target_col: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    df.columns = [c.replace('.', '_') for c in df.columns]
    df = df.sort_values(["stationid", "timestamp"])
    df = df.rename(columns={"stationid": "station_id"})
    original_target_col = target_col.replace('.', '_')
    corrected_col = f"{original_target_col}_corrected"
    if corrected_col not in df.columns:
        raise ValueError(f"Corrected column {corrected_col} not found")
    df[original_target_col] = pd.to_numeric(df[corrected_col], errors="coerce")
    target_col = original_target_col
    df = df.dropna(subset=[target_col])
    df["group_id"] = df["station_id"].astype(str)
    df["time_idx"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds() // 3600
    df["time_idx"] = df["time_idx"].astype(int)
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["hour"] = df["timestamp"].dt.hour
    return df, target_col


def build_datasets(df: pd.DataFrame, target_col: str, max_encoder: int, max_prediction: int):
    min_size = max_encoder + max_prediction
    valid_groups = df.groupby("group_id").size()
    keep_groups = valid_groups[valid_groups > min_size].index
    df = df[df["group_id"].isin(keep_groups)].copy()

    training_cutoff = df["time_idx"].max() - max_prediction * 30
    training = TimeSeriesDataSet(
        df[df["time_idx"] <= training_cutoff],
        time_idx="time_idx",
        target=target_col,
        group_ids=["group_id"],
        max_encoder_length=max_encoder,
        max_prediction_length=max_prediction,
        min_encoder_length=24,
        min_prediction_length=1,
        static_categoricals=["group_id"],
        time_varying_known_reals=["time_idx", "day_of_week", "hour"],
        time_varying_unknown_reals=[target_col],
        target_normalizer=GroupNormalizer(groups=["group_id"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )
    validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
    train_loader = training.to_dataloader(train=True, batch_size=128, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=128, num_workers=0)
    return training, validation, train_loader, val_loader


def main(args: argparse.Namespace) -> None:
    original_target = "pm2.5"
    df, target_col = load_and_prepare_data(original_target)
    training_ds, validation_ds, train_loader, val_loader = build_datasets(df, target_col, args.encoder_length, args.prediction_length)

    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment("air-sentinel-tft-pf")

    model = TemporalFusionTransformer.from_dataset(
        training_ds,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        attention_head_size=args.attention_heads,
        dropout=args.dropout,
        hidden_continuous_size=args.hidden_cont_size,
        loss=QuantileLoss(),
        output_size=7,
        log_interval=100,
        log_val_interval=1,
    )

    seed_everything(42, workers=True)

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="cpu",
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
    )

    with mlflow.start_run(run_name="tft_pf_baseline"):
        mlflow.log_params(
            {
                "encoder_length": args.encoder_length,
                "prediction_length": args.prediction_length,
                "learning_rate": args.learning_rate,
                "hidden_size": args.hidden_size,
                "hidden_continuous_size": args.hidden_cont_size,
                "attention_heads": args.attention_heads,
                "dropout": args.dropout,
                "epochs": args.epochs,
            }
        )

        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        best_model = model
        val_loader = validation_ds.to_dataloader(train=False, batch_size=128, num_workers=0)
        preds = best_model.predict(val_loader, mode="prediction").detach().cpu().numpy().reshape(-1)
        val_loader = validation_ds.to_dataloader(train=False, batch_size=128, num_workers=0)
        targets_list = []
        for x, y in val_loader:
            target = y[0].detach().cpu().numpy().reshape(-1)
            targets_list.append(target)
        targets = np.concatenate(targets_list)

        mae = float(np.mean(np.abs(targets - preds)))
        rmse = float(np.sqrt(np.mean((targets - preds) ** 2)))
        mape = float(np.mean(np.abs((targets - preds) / np.maximum(np.abs(targets), 1e-3))) * 100)
        mlflow.log_metrics({"val_mae": mae, "val_rmse": rmse, "val_mape": mape})

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_DIR / "tft_pf_model.ckpt"
        trainer.save_checkpoint(model_path)
        mlflow.log_artifact(model_path, artifact_path="models")

        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = DOCS_DIR / "tft_pf_metrics.csv"
        pd.DataFrame([
            {
                "val_mae": mae,
                "val_rmse": rmse,
                "val_mape": mape,
            }
        ]).to_csv(report_path, index=False)
        mlflow.log_artifact(report_path, artifact_path="reports")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TFT with PyTorch Forecasting.")
    parser.add_argument("--encoder-length", type=int, default=168)
    parser.add_argument("--prediction-length", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--hidden-cont-size", type=int, default=16)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()
    main(args)
