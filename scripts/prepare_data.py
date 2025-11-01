#!/usr/bin/env python3
"""Clean and align air-quality datasets for model training."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"


def load_raw_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Standardise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def preprocess_dataset(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df.copy()
    # Basic timestamp normalisation
    if "date" in df.columns and "time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"], errors="coerce")
    elif "datetime" in df.columns:
        df["timestamp"] = pd.to_datetime(df["datetime"], errors="coerce")
    elif "date" in df.columns:
        df["timestamp"] = pd.to_datetime(df["date"], errors="coerce")
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        raise ValueError(f"{source}: could not find timestamp columns.")

    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp")
    df["source"] = source
    return df


def merge_datasets(datasets: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(datasets, ignore_index=True)
    combined = combined.drop_duplicates(subset=["timestamp", "source"])
    return combined


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    dataframes: list[pd.DataFrame] = []
    metadata_frames: dict[str, pd.DataFrame] = {}
    for path in RAW_DIR.glob("*.csv"):
        df = load_raw_csv(path)
        try:
            processed = preprocess_dataset(df, source=path.stem)
            dataframes.append(processed)
        except ValueError:
            metadata_frames[path.stem] = df

    if not dataframes:
        raise RuntimeError("No CSV files found in data/raw/. Run download_datasets.py first.")

    combined = merge_datasets(dataframes)
    combined.to_parquet(PROCESSED_DIR / "air_quality_combined.parquet", index=False)
    combined.describe().to_csv(INTERIM_DIR / "summary_stats.csv")
    combined["timestamp"].sort_values().to_frame().to_csv(INTERIM_DIR / "timeline.csv", index=False)
    for name, meta_df in metadata_frames.items():
        meta_df.to_csv(INTERIM_DIR / f"{name}_metadata.csv", index=False)
        print(f"Saved metadata for {name}")
    print("Processed dataset saved to data/processed/air_quality_combined.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare air-quality datasets for modeling.")
    _ = parser.parse_args()
    main()
