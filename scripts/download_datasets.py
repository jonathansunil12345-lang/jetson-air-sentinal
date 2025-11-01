#!/usr/bin/env python3
"""Download air-quality datasets from Kaggle and external sources."""
from __future__ import annotations

import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def run_command(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")


def download_kaggle_dataset(dataset: str) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset,
        "-p",
        str(RAW_DIR),
        "--unzip",
    ]
    run_command(cmd)


def main() -> None:
    datasets = [
        "shivanshuman592/air-quality-dataset",
        "paultimothymooney/air-quality",
    ]
    for ds in datasets:
        print(f"Downloading {ds}...")
        download_kaggle_dataset(ds)
    print("All downloads complete.")


if __name__ == "__main__":
    main()
