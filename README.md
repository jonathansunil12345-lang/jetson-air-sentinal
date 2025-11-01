# Jetson Air-Quality Sentinel

Real-time spatio-temporal air-quality forecasting and alerting on NVIDIA Jetson hardware.

## Table of Contents

1. [Overview](#overview)
2. [Targets & Success Criteria](#targets--success-criteria)
3. [Data Strategy](#data-strategy)
4. [Model Plan](#model-plan)
5. [Jetson Deployment Architecture](#jetson-deployment-architecture)
6. [Implementation Roadmap](#implementation-roadmap)
7. [Repository Layout](#repository-layout)
8. [Immediate Next Steps](#immediate-next-steps)

---

## Overview

- Build an edge service that ingests low-cost air-quality sensors (PM2.5, PM10, NO₂, O₃) plus weather feeds.
- Train advanced spatio-temporal models (Temporal Graph Attention Network, diffusion head) for short-term forecasting.
- Deliver actionable dashboards and low-latency alerts from an NVIDIA Jetson Nano/Orin node.

---

## Targets & Success Criteria

- **Accuracy**: <5% MAPE on PM2.5 forecasts versus reference stations; calibrated uncertainty (CRPS improvement over baseline).
- **Latency**: <150 ms inference time per prediction on Jetson (TensorRT FP16), <50 ms data ingestion delay.
- **Reliability**: Fallback to baseline model when GPU is unavailable; watchdog for service restart.
- **Usability**: Grafana/Node-RED dashboard with forecast bands, alerts when thresholds exceeded.

---

## Data Strategy

1. **Sources**
   - Kaggle datasets: `shivanshuman592/air-quality-dataset`, `paultimothymooney/air-quality`.
   - Government/agency reference data (EPA AQI, local monitoring stations).
   - Weather API (OpenWeather, NOAA) to provide temperature, humidity, wind, pressure.

2. **Processing Pipeline**
   - Standardize timestamps, time zones, and units.
   - Calibrate sensor biases using co-location with reference stations.
   - Compute location graph (k-nearest neighbors based on geospatial distance).
   - Split data: train (2017–2022), validation (2023), test (2024).

3. **Storage**
   - Versioned datasets in `data/processed/` (Parquet/Delta).
   - Raw archives under `data/raw/` with metadata in `data/interim/`.
   - Optional DVC or LakeFS integration for reproducibility (future step).
4. **Exploration & Calibration**
   - `notebooks/01_eda.ipynb` visualises seasonal patterns, pollutant correlations, calibration error, and feature engineering options.
   - `scripts/apply_bias_corrections.py` produces station-level bias offsets (`docs/station_bias_table.csv`) and bias-corrected dataset `data/processed/air_quality_bias_corrected.parquet`.
   - `docs/baseline_metrics.csv` captures baseline Gradient Boosting results for PM2.5 forecasting.

---

## Model Plan

| Stage | Goal | Deliverables |
| --- | --- | --- |
| Baseline | Seasonal naive, Gradient Boosting regression | `scripts/train_baseline.py` – Test MAE ≈ 20.1, RMSE ≈ 35.3 (PM2.5); metrics in `docs/baseline_metrics.csv`; MLflow experiment `air-sentinel-baseline`. |
| Optuna TFT surrogate | Feed-forward Lightning model | `scripts/train_tft_optuna.py` – Test MAE ≈ 15.6 after 3 trials (MLflow exp `air-sentinel-tft`). |
| Temporal Fusion Transformer | PyTorch Forecasting implementation | `scripts/train_tft_pf.py` – Val MAE ≈ 4.36 on bias-corrected data (MLflow exp `air-sentinel-tft-pf`). |
| Advanced | Temporal Graph Attention Network (TGAT) / full TFT with spatial graph | Next phase: integrate graph structure using bias-corrected dataset. |
| Uncertainty | Diffusion/probabilistic head or quantile loss | Calibrated forecast intervals, evaluation via CRPS and coverage. |
| Deployment Model | TensorRT-optimized version | Exported engine (`models/tensorrt/`) with latency benchmarks. |

---

## Jetson Deployment Architecture

1. **Data Ingestion**: MQTT/REST gateway pulling low-cost sensor data + weather feed.
2. **Inference Service**: FastAPI container running TensorRT model, with fallback CPU model.
3. **Queue/Cache**: Redis or InfluxDB for smoothing and historical access.
4. **Dashboard**: Node-RED/Grafana container visualizing live data and forecasts.
5. **Alerts**: Threshold-based triggers (MQTT/IFTTT/email/SMS).

Deployment scripts and Dockerfiles will live under `deployment/` and `services/`.

---

## Implementation Roadmap

| Week | Milestone | Focus |
| --- | --- | --- |
| 1 | Dataset ingestion & cleaning | Kaggle downloads, station metadata, weather enrichment. |
| 2 | Bias calibration & EDA | Sensor-reference alignment, outlier handling, feature engineering. |
| 3 | Baseline modeling & dashboards | Seasonal + Gradient Boosting models, initial Grafana dashboard. |
| 4 | TGAT development | Graph construction, spatio-temporal model training. |
| 5 | Probabilistic forecasts | Diffusion/quantile head, uncertainty evaluation. |
| 6 | TensorRT deployment | Conversion, latency profiling, failover logic. |
| 7 | Edge integration | MQTT ingestion, container orchestration, dashboard polishing. |
| 8 | Validation & release | Field validation, documentation, release packaging. |

---

## Repository Layout

```
jetson-air-sentinel/
├── data/
│   ├── raw/          # Downloaded datasets
│   ├── processed/    # Cleaned feature stores
│   └── interim/      # Metadata, calibration tables
├── scripts/          # Data processing and training scripts
├── notebooks/        # EDA, modeling, deployment notebooks
├── models/           # Checkpoints, TensorRT engines
├── services/         # FastAPI inference service, MQTT bridges
├── deployment/       # Dockerfiles, compose manifests, Jetson setup
├── docs/             # Architecture diagrams, evaluation reports
└── mlruns/           # MLflow tracking artifacts
```

---

## Immediate Next Steps

1. Configure Kaggle/NOAA/OpenWeather credentials in `.env` (not committed).  
2. Implement `scripts/download_datasets.py` to fetch and stage raw data.  
3. Use `scripts/apply_bias_corrections.py` outputs to experiment with additional bias-adjustment strategies.  
4. Run larger Optuna sweeps / TFT PF trainings and document comparison vs baseline (update `docs/tft_optuna_metrics.csv`, `docs/tft_pf_metrics.csv`).  
5. Begin TGAT graph construction and Lightning module integration (extend `scripts/train_tft_pf.py` or introduce TGAT module).  
6. Set up MLflow experiment scaffolding (`air-sentinel-baseline`, `air-sentinel-tft`, `air-sentinel-tft-pf`, `air-sentinel-tgat`).  
7. Define Docker base images for Jetson (L4T, CUDA/cuDNN versions) in `deployment/`.
