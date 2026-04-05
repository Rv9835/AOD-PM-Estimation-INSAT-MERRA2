# Air Pollution Monitoring from Space (India)

Structured AI/ML pipeline to estimate surface PM2.5 and PM10 by combining satellite AOD, ground observations, and meteorological reanalysis.

## Project Structure

- `src/airpollution/`: source package
- `configs/`: YAML configuration
- `scripts/`: runnable entry scripts
- `tests/`: unit and integration tests
- `data/`: local data workspace (ignored)
- `artifacts/`: trained models and outputs (ignored)

## Quick Start

1. Create and activate a Python environment.
2. Install dependencies:
   ```bash
   pip install -e .[dev]
   ```
3. Run training pipeline:
   ```bash
   python scripts/train_and_evaluate.py --config configs/base.yaml
   ```
4. Run tests:
   ```bash
   pytest
   ```

## Current Scope (Phase 1)

- Robust tabular ML baseline (`RandomForestRegressor`)
- Deterministic train/test split
- Strong input validation and logging
- Metrics: RMSE, MAE, R²

Later phases will add INSAT/MERRA/CPCB specific ingestion connectors and geospatial map generation.
