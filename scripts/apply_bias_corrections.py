#!/usr/bin/env python3
"""Compute station-level bias corrections and apply them to the processed dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "air_quality_combined.parquet"
ADJUSTED_PATH = PROJECT_ROOT / "data" / "processed" / "air_quality_bias_corrected.parquet"
DOCS_DIR = PROJECT_ROOT / "docs"


def load_dataset() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    df["date"] = df["timestamp"].dt.date
    df["stationid"] = df.get("stationid", "unknown").fillna("unknown")
    df["city"] = df.get("city", "unknown").fillna("unknown")
    df["pm2.5"] = pd.to_numeric(df.get("pm2.5", df.get("pm2_5")), errors="coerce")
    return df


def compute_bias_table(df: pd.DataFrame) -> pd.DataFrame:
    city_daily = (
        df.groupby(["city", "date"])["pm2.5"]
        .median()
        .rename("city_median_pm25")
        .reset_index()
    )
    merged = df.merge(city_daily, on=["city", "date"], how="left")
    merged["bias"] = merged["pm2.5"] - merged["city_median_pm25"]
    bias_table = (
        merged.groupby("stationid")["bias"]
        .mean()
        .rename("mean_bias_pm25")
        .reset_index()
    )
    return bias_table


def apply_bias(df: pd.DataFrame, bias_table: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(bias_table, on="stationid", how="left")
    df["mean_bias_pm25"] = df["mean_bias_pm25"].fillna(0.0)
    df["pm2.5_corrected"] = df["pm2.5"] - df["mean_bias_pm25"]
    return df


def save_outputs(df: pd.DataFrame, bias_table: pd.DataFrame) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    bias_table.to_csv(DOCS_DIR / "station_bias_table.csv", index=False)
    df.to_parquet(ADJUSTED_PATH, index=False)
    print(f"Saved bias-corrected dataset to {ADJUSTED_PATH}")


def main() -> None:
    df = load_dataset()
    bias_table = compute_bias_table(df)
    adjusted_df = apply_bias(df, bias_table)
    save_outputs(adjusted_df, bias_table)


if __name__ == "__main__":
    main()
