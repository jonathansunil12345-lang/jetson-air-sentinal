# Jetson Air-Quality Sentinel Roadmap

| Week | Milestone | Deliverables |
| --- | --- | --- |
| 1 | Data ingestion & credential setup | Kaggle downloads, weather API integration, raw data catalog |
| 2 | Cleaning & calibration | Outlier handling, sensor bias correction, feature schema |
| 3 | Baseline models & dashboard | Seasonal + Gradient Boosting models, initial Grafana dashboard |
| 4 | Spatio-temporal graph model | TGAT/TFT training scripts, Optuna sweep results |
| 5 | Probabilistic forecasts | Diffusion/quantile head, uncertainty evaluation (CRPS, coverage) |
| 6 | TensorRT deployment | Engine conversion, latency benchmarks, watchdog/failover |
| 7 | Edge integration | FastAPI inference service, MQTT bridge, Node-RED dashboard |
| 8 | Validation & release | Field tests, power/latency report, deployment guide, v1 release |

### Technical Threads
- Graph construction (geospatial kNN, correlation-based edges)
- Data versioning (DVC/LakeFS) and MLflow experiment tracking
- TensorRT optimization and resource monitoring (tegrastats)
- Alerting pipeline (MQTT → IFTTT/SMS/email)

### Dependencies
- NVIDIA Jetson Nano/Orin with CUDA/cuDNN stack
- Low-cost PM/NO₂/O₃ sensors + microcontroller interface
- Weather API key (OpenWeather or NOAA)
- Reference AQ monitoring data (EPA, local agencies)
