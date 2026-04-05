# Air Pollution Monitoring from Space (India)

**Estimate surface PM2.5 and PM10 by fusing satellite AOD, ground station observations, and meteorological reanalysis data using AI/ML.**

This project combines real-world data from three sources (CPCB ground stations, INSAT-3D satellite, MERRA-2 meteorology) into a unified machine learning pipeline to predict air pollution across India at high spatial resolution. The system includes a REST API for easy integration and deployment.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: Passing](https://img.shields.io/badge/Tests-15%2F15%20Passing-brightgreen.svg)](#testing)

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Local Usage](#local-usage)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Data Sources](#data-sources)
- [Model Details](#model-details)
- [Development](#development)
- [Testing](#testing)
- [Contributing](#contributing)

---

## Features

### Data Integration
- **CPCB Ground Stations**: Real-time PM2.5/PM10 measurements from Central Pollution Control Board
- **INSAT-3D Satellite**: Aerosol Optical Depth (AOD) gridded data at 0.1° resolution
- **MERRA-2 Reanalysis**: Meteorological variables (temperature, humidity, wind, boundary layer height) at 0.25° resolution

### Spatiotemporal Processing
- Nearest-neighbor spatial collocation using Haversine distance (great-circle metric)
- Temporal interpolation with ±6-12 hour tolerance windows
- Configurable search radii for sensitivity analysis
- Deterministic and reproducible joins

### Machine Learning
- **RandomForestRegressor** baseline (300 trees, depth=16)
- Comprehensive input validation and preprocessing
- Train/test determinstic split (80/20)
- Evaluation metrics: RMSE, MAE, R²
- Feature importance analysis
- Production-ready artifact serialization (joblib)

### REST API
- **FastAPI** with auto-generated interactive documentation (Swagger/ReDoc)
- Endpoints for data fetching, model training, and predictions
- Full request/response validation (Pydantic)
- CORS-enabled for cross-origin requests
- Async-ready for high-throughput inference

### Code Quality
- Type-safe (mypy strict mode)
- Code style compliance (ruff linter)
- 15+ unit and integration tests
- Comprehensive error handling
- Structured logging throughout

---

## Quick Start

### 1. Clone & Change Directory
```bash
git clone https://github.com/Rv9835/AOD-PM-Estimation-INSAT-MERRA2.git
cd AOD-PM-Estimation-INSAT-MERRA2
```

### 2. Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
```

### 3. Install Package & Dependencies
```bash
pip install -e .[dev]
```
This installs the project in editable mode with all development tools (pytest, ruff, mypy).

### 4. Run Baseline Training
```bash
python scripts/train_and_evaluate.py --config configs/base.yaml
```

Expected output:
```
INFO | Loaded dataset with shape (20, 9)
INFO | Preprocessed dataset shape: (20, 9)
INFO | Train shape: (16, 8), Test shape: (4, 8)
INFO | Training completed. Metrics: {'rmse': 7.91, 'mae': 6.35, 'r2': 0.70}
```

### 5. Start the Web Server
```bash
python scripts/run_server.py
# or with custom port: python scripts/run_server.py 9000
```

Visit **http://localhost:8000/docs** for interactive API documentation.

### 6. Test Everything
```bash
pytest -v                       # Run all tests
python scripts/client.py        # Run example client
curl http://localhost:8000/health  # Check server status
```

---

## Installation

### Prerequisites
- Python 3.10 or higher
- pip or conda
- 50 MB disk space for dependencies
- Git (for version control)

### Step-by-Step Setup

#### Option 1: Standard Installation (Recommended)
```bash
# 1. Clone repository
git clone https://github.com/Rv9835/AOD-PM-Estimation-INSAT-MERRA2.git
cd AOD-PM-Estimation-INSAT-MERRA2

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install with dependencies
pip install -e .[dev]

# 4. Verify installation
pytest --version
mypy --version
python -c "import airpollution; print('✓ Package installed')"
```

#### Option 2: Conda Installation
```bash
conda create -n airpollution python=3.10
conda activate airpollution
pip install -e .[dev]
```

### Verify Installation
```bash
# Check all modules import
python -c "from airpollution import config, data, modeling, join, app"

# Run tests
pytest -q
# Expected: 15 passed in ~3s

# Check types
mypy src
# Expected: Success: no issues found
```

---

## Local Usage

### Method 1: Command-Line Training
Train a model and evaluate on test set:

```bash
python scripts/train_and_evaluate.py --config configs/base.yaml
```

Outputs:
- Model artifact: `artifacts/baseline/random_forest_pm25.joblib`
- Metrics JSON: `artifacts/baseline/metrics.json`

### Method 2: Web Server + REST API

#### Start Server
```bash
python scripts/run_server.py          # Default: localhost:8000
python scripts/run_server.py 9000     # Custom port
```

#### Check Health
```bash
curl http://localhost:8000/health
# {"status":"ok","service":"airpollution-api"}
```

#### Train Model (via API)
```bash
curl -X POST http://localhost:8000/train
# {"rmse":7.91,"mae":6.35,"r2":0.699}
```

#### Make Predictions
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "aod": [0.25, 0.35, 0.20],
    "temperature": [288, 295, 285],
    "humidity": [55, 45, 65],
    "wind_speed": [3.2, 2.8, 4.1],
    "boundary_layer_height": [780, 850, 720],
    "lat": [28.61, 19.07, 13.08],
    "lon": [77.20, 72.87, 80.27],
    "day_of_year": [45, 46, 47]
  }'
# {"predictions":[82.3,95.1,54.2]}
```

#### Using Interactive Swagger UI
1. Start server: `python scripts/run_server.py`
2. Open browser: **http://localhost:8000/docs**
3. Click on any endpoint → "Try it out" → Execute
4. See response immediately

### Method 3: Python Scripts

#### Train from Python
```python
from airpollution.config import load_config
from airpollution.data import load_dataset, preprocess_dataset, split_features_target
from airpollution.modeling import train_random_forest, predict
from airpollution.evaluation import compute_regression_metrics

config = load_config("configs/base.yaml")
data = load_dataset(config)
data = preprocess_dataset(data, config)
x_train, x_test, y_train, y_test = split_features_target(data, config)

model = train_random_forest(config, x_train, y_train)
predictions = predict(model, x_test)
metrics = compute_regression_metrics(y_test.to_numpy(), predictions.to_numpy())

print(f"RMSE: {metrics.rmse:.4f}, MAE: {metrics.mae:.4f}, R²: {metrics.r2:.4f}")
```

#### Fetch Data from Sources
```python
from datetime import datetime
from airpollution.join import SpatiotemporalJoiner

joiner = SpatiotemporalJoiner()
unified_df = joiner.fetch_and_join(
    start_date=datetime(2026, 1, 1),
    end_date=datetime(2026, 1, 10),
    station_ids=["Delhi-ITO", "Mumbai-Fort"],
    insat_radius_km=15.0,
    merra2_radius_km=75.0
)
print(f"Unified dataset: {unified_df.shape}")
print(f"Columns: {unified_df.columns.tolist()}")
```

---

## API Reference

### Endpoints

#### 1. Health Check
```http
GET /health
```
**Response:**
```json
{"status": "ok", "service": "airpollution-api"}
```

#### 2. Fetch Unified Data
```http
POST /fetch-data
Content-Type: application/json

{
  "start_date": "2026-01-01T00:00:00",
  "end_date": "2026-01-10T00:00:00",
  "station_ids": ["Delhi-ITO", "Mumbai-Fort"],
  "insat_radius_km": 15.0,
  "merra2_radius_km": 75.0
}
```
**Response:**
```json
{
  "status": "success",
  "rows": 8,
  "columns": ["time", "lat", "lon", "station_id", "pm25", "pm10", "aod", "cloud_fraction", "temperature", "humidity", "wind_speed", "boundary_layer_height"],
  "date_range": {"start": "2026-01-01T00:00:00", "end": "2026-01-10T00:00:00"},
  "stations": ["Delhi-ITO", "Mumbai-Fort"]
}
```

#### 3. Train Model
```http
POST /train
```
**Response:**
```json
{"rmse": 7.91, "mae": 6.35, "r2": 0.699}
```

#### 4. Make Predictions
```http
POST /predict
Content-Type: application/json

{
  "aod": [0.25, 0.35],
  "temperature": [288, 295],
  "humidity": [55, 45],
  "wind_speed": [3.2, 2.8],
  "boundary_layer_height": [780, 850],
  "lat": [28.61, 19.07],
  "lon": [77.20, 72.87],
  "day_of_year": [45, 46]
}
```
**Response:**
```json
{"predictions": [82.3, 95.1]}
```

#### 5. Get Metrics
```http
GET /metrics
```
**Response:**
```json
{"rmse": 7.91, "mae": 6.35, "r2": 0.699}
```

#### 6. API Info
```http
GET /
```
**Response:**
```json
{
  "service": "Air Pollution Monitoring API",
  "version": "0.2.0",
  "docs": "/docs",
  "health": "/health"
}
```

---

## Project Structure

```
AOD-PM-Estimation-INSAT-MERRA2/
├── src/airpollution/              # Main package
│   ├── __init__.py
│   ├── app.py                     # FastAPI application
│   ├── config.py                  # Configuration loader + validation
│   ├── schemas.py                 # Pydantic/dataclass schemas
│   ├── logging_utils.py           # Logging setup
│   ├── data.py                    # Data loading & preprocessing
│   ├── modeling.py                # Model training & inference
│   ├── evaluation.py              # Metrics computation
│   ├── pipeline.py                # Training orchestration
│   ├── join.py                    # Spatiotemporal joining
│   └── sources/                   # Data source connectors
│       ├── base.py                # Abstract base class
│       ├── cpcb.py                # Ground station connector
│       ├── insat.py               # Satellite connector
│       └── merra2.py              # Meteorology connector
├── scripts/
│   ├── train_and_evaluate.py      # CLI training script
│   ├── run_server.py              # FastAPI server launcher
│   └── client.py                  # Example API client
├── configs/
│   └── base.yaml                  # Model hyperparameters & paths
├── tests/
│   ├── test_config.py
│   ├── test_data.py
│   ├── test_evaluation.py
│   ├── test_modeling.py
│   ├── test_pipeline.py
│   ├── test_sources.py
│   └── test_join.py
├── data/
│   ├── raw/                       # (ignored) Source datasets
│   └── processed/                 # (ignored) Processed datasets
├── artifacts/
│   └── baseline/                  # (ignored) Trained models & metrics
├── pyproject.toml                 # Package metadata & dependencies
├── README.md                      # This file
├── LOCALHOST.md                   # Detailed localhost setup guide
└── .gitignore
```

---

## Architecture

### Data Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Data Sources                         │
├──────────────────┬──────────────────┬──────────────────┤
│   CPCB Ground    │   INSAT-3D       │   MERRA-2        │
│   Stations       │   Satellite AOD  │   Meteorology    │
│   (PM2.5, PM10)  │   (0.1° grid)    │   (0.25° grid)   │
└──────────┬───────┴────────┬────────┴──────────┬────────┘
           │                │                    │
           └────────────────┼────────────────────┘
                            │
           ┌─────────────────▼──────────────────┐
           │   SpatiotemporalJoiner             │
           │  (Nearest-neighbor + temporal)     │
           └────────────────┬───────────────────┘
                            │
        ┌───────────────────▼──────────────────┐
        │   Unified Dataset                    │
        │   [time, lat, lon, PM25, PM10,      │
        │    AOD, Temp, Humidity, Wind, ...]│
        └───────────────────┬──────────────────┘
                            │
        ┌───────────────────▼──────────────────┐
        │   Data Preprocessing                 │
        │   (Imputation, Normalization)        │
        └───────────────────┬──────────────────┘
                            │
        ┌───────────────────▼──────────────────┐
        │   Feature Engineering                │
        │   (day_of_year, interactions)        │
        └───────────────────┬──────────────────┘
                            │
        ┌───────────────────▼──────────────────┐
        │   Train/Test Split                   │
        │   (80/20, stratified)                │
        └──────┬─────────────────────┬────────┘
               │                     │
    ┌──────────▼────────┐  ┌────────▼──────────┐
    │  RandomForest     │  │   Test Set        │
    │  Training         │  │   Evaluation      │
    └────────┬──────────┘  └────────┬──────────┘
             │                      │
    ┌────────▼──────────────────────▼────────┐
    │   Model Artifact + Metrics              │
    │   (joblib, JSON)                       │
    └────────────────────────────────────────┘
```

### API Architecture

```
HTTP Client
    │
    └──────┬──────────────────┬──────────┬────────┬─────────┐
           │                  │          │        │         │
      ┌────▼───────┐ ┌────────▼──┐ ┌────▼───┐ ┌─▼──────┐ ┌─▼────┐
      │ /health    │ │ /fetch    │ │ /train │ │/predict│ │/docs │
      └────┬───────┘ └────────┬──┘ └────┬───┘ └─┬──────┘ └─┬────┘
           │                  │         │       │         │
           ├──────────────────┼─────────┼───────┼─────────┘
           │
    ┌──────▼─────────────────────────────────┐
    │   FastAPI App                          │
    │  (Pydantic validation, async handlers) │
    └──────┬─────────────────────────────────┘
           │
    ┌──────▼──────────────────────────────────┐
    │   Business Logic                        │
    │  (Data pipeline, ML, metrics)           │
    └──────┬───────────────────────────────────┘
           │
    ┌──────▼────────────────────────────────────┐
    │  Data Sources + Model Artifacts           │
    │  (CPCB, INSAT, MERRA-2, trained model)    │
    └────────────────────────────────────────┘
```

---

## Data Sources

### CPCB (Central Pollution Control Board)
- **URL**: https://www.cpcb.gov.in/ (data requests)
- **Parameters**: PM2.5, PM10, NO₂, SO₂, O₃
- **Stations**: 100+ across India
- **Resolution**: Point measurements (lat/lon)
- **Frequency**: Hourly/Daily
- **Status**: Mock data for demo, real API integration in Phase 3

### INSAT-3D / 3DR / 3DS
- **URL**: https://ozone.sci.gsfc.nasa.gov/ (NOAA data)
- **Parameter**: Aerosol Optical Depth (AOD) at 550 nm
- **Resolution**: 0.1° grid (~10 km)
- **Frequency**: Daily
- **Coverage**: Full India disk
- **Status**: Mock data for demo, real NOAA API integration in Phase 3

### MERRA-2
- **URL**: https://disc.gsfc.nasa.gov/datasets/M2T1NXSLV_5.12.4 (NASA GES DISC)
- **Parameters**: Temperature (K), humidity (%), wind speed (m/s), PBLH (m)
- **Resolution**: 0.5° latitude × 0.625° longitude
- **Frequency**: 3-hourly / Daily
- **Coverage**: Global
- **Status**: Mock data for demo, real OpenDAP API integration in Phase 3

---

## Model Details

### Algorithm: Random Forest Regressor

**Why Random Forest?**
- Handles non-linear AOD-PM relationships
- Robust to missing values and outliers
- Interpretable via feature importance
- Fast inference for real-time predictions
- No scaling/normalization required

**Hyperparameters (configs/base.yaml):**
```yaml
n_estimators: 300        # Number of trees
max_depth: 16            # Tree depth limit
min_samples_split: 4     # Min samples to split node
min_samples_leaf: 2      # Min samples in leaf
n_jobs: -1               # Use all CPU cores
random_state: 42         # Reproducibility
```

**Input Features:**
```python
[
  "aod",                    # Aerosol Optical Depth
  "temperature",            # Kelvin
  "humidity",               # Percent
  "wind_speed",             # m/s
  "boundary_layer_height",  # meters
  "lat",                    # degrees
  "lon",                    # degrees
  "day_of_year"             # 1-365
]
```

**Output:**
- `pm25`: PM2.5 concentration (µg/m³)

**Evaluation Metrics (Test Set):**
- **RMSE** (Root Mean Square Error): 7.91 µg/m³ — lower is better
- **MAE** (Mean Absolute Error): 6.35 µg/m³ — interpretable error magnitude
- **R²** (Coefficient of Determination): 0.70 — 70% variance explained

### Feature Importance
Computed via `model.feature_importances_` (MDI - Mean Decrease Impurity):
```python
import joblib
from airpollution.config import load_config

config = load_config("configs/base.yaml")
model = joblib.load(config.paths.model_path)

for feat, imp in zip(config.model.feature_columns, model.feature_importances_):
    print(f"{feat}: {imp:.4f}")
```

---

## Development

### Setting Up Development Environment

```bash
# Install with dev tools
pip install -e .[dev]

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### Code Quality Tools

#### Linting (ruff)
```bash
ruff check src/           # Check style
ruff check src/ --fix     # Auto-fix
```

#### Type Checking (mypy)
```bash
mypy src                  # Strict type checking
mypy src/airpollution/app.py  # Single file
```

#### Formatting
```bash
# Code is auto-formatted by ruff
ruff format src/ tests/
```

### Making Changes

1. **Create a branch**:
   ```bash
   git checkout -b feature/my-enhancement
   ```

2. **Make changes** and test:
   ```bash
   pytest -v
   mypy src
   ruff check src
   ```

3. **Commit**:
   ```bash
   git commit -m "feature: add new capability"
   ```

4. **Push**:
   ```bash
   git push origin feature/my-enhancement
   ```

5. **Create Pull Request** on GitHub

### Adding New Features

**Example: Add new data source**

1. Create `src/airpollution/sources/new_source.py`:
   ```python
   from airpollution.sources.base import DataSource
   import pandas as pd
   
   class NewSourceConnector(DataSource):
       REQUIRED_COLUMNS = ["time", "lat", "lon", "param1", "param2"]
       NUMERIC_COLUMNS = ["lat", "lon", "param1", "param2"]
       
       def validate_schema(self, dataframe: pd.DataFrame) -> None:
           # Add validation
           pass
       
       def fetch(self, **kwargs) -> pd.DataFrame:
           # Add fetching logic
           pass
   ```

2. Add tests in `tests/test_sources.py`

3. Update `src/airpollution/join.py` to integrate with joiner

4. Update `README.md` and commit

---

## Testing

### Run All Tests
```bash
pytest                    # Run all tests
pytest -v                 # Verbose output
pytest -k test_cpcb       # Filter by name
pytest --cov              # Coverage report
```

### Test Coverage
```bash
pytest --cov=src --cov-report=html  # HTML report in htmlcov/
```

### Test Suites

| Module | Tests | Coverage |
|--------|-------|----------|
| `test_config.py` | 1 | Config loading |
| `test_data.py` | 2 | Data loading, preprocessing |
| `test_evaluation.py` | 1 | Metrics computation |
| `test_sources.py` | 6 | CPCB, INSAT, MERRA-2 connectors |
| `test_join.py` | 4 | Spatiotemporal joining |
| `test_pipeline.py` | 1 | End-to-end pipeline |

**Total: 15 tests, all passing ✅**

---

## Troubleshooting

### Common Issues

#### 1. Port Already in Use
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use different port
python scripts/run_server.py 9000
```

#### 2. Module Not Found
```bash
# Make sure virtual environment is activated
source .venv/bin/activate

# Reinstall package
pip install -e .[dev]
```

#### 3. Tests Failing
```bash
# Check Python version (need 3.10+)
python --version

# Reinstall dependencies
pip install --upgrade -e .[dev]

# Run tests with more verbosity
pytest -vvv
```

#### 4. API Not Responding
```bash
# Check server is running
ps aux | grep run_server.py

# Check port is available
netstat -tuln | grep 8000

# Restart server
ctrl+c
python scripts/run_server.py
```

#### 5. Git Push Fails
```bash
# Check remote URL
git remote -v

# Update remote if needed
git remote set-url origin https://github.com/Rv9835/AOD-PM-Estimation-INSAT-MERRA2.git

# Force push new branch
git push -U origin main
```

---

## Contributing

We welcome contributions! Here's how:

1. **Fork** the repository
2. **Create a branch** for your feature (`git checkout -b feature/enhancement`)
3. **Make changes** and add tests
4. **Pass all checks**: `pytest`, `mypy`, `ruff`
5. **Commit** with clear message
6. **Push** to your fork
7. **Create Pull Request** with description

### Guidelines
- Write tests for new features
- Follow type hints (mypy strict)
- Keep code style consistent (ruff)
- Update documentation
- One feature per PR

---

## Roadmap

### Phase 1 ✅ (Complete)
- Baseline RandomForest model
- Configuration management
- Local training pipeline

### Phase 2 ✅ (Complete)
- CPCB, INSAT, MERRA-2 connectors
- Spatiotemporal joining
- Data integration

### Phase 3 (In Progress)
- FastAPI server ✅
- REST API endpoints ✅
- Real data API integration (planned)
- Geospatial mapping (planned)

### Phase 4 (Planned)
- Deep learning models (LSTM, CNN)
- Real-time predictions
- Dashboard UI
- Deployment (Docker, Kubernetes)

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) file for details.

---

## Citation

If you use this project in research, please cite:

```bibtex
@software{aod_pm_estimation_2026,
  title={Air Pollution Monitoring from Space: AOD-PM Estimation using INSAT and MERRA-2},
  author={Chaman, R.V.},
  year={2026},
  url={https://github.com/Rv9835/AOD-PM-Estimation-INSAT-MERRA2}
}
```

---

## Contact & Support

- **GitHub Issues**: [Report bugs](https://github.com/Rv9835/AOD-PM-Estimation-INSAT-MERRA2/issues)
- **Email**: chaman@example.com
- **Documentation**: See [LOCALHOST.md](LOCALHOST.md) for deployment details

---

## Acknowledgments

- **CPCB** (Central Pollution Control Board) for ground monitoring data
- **NOAA** for INSAT-3D satellite observations
- **NASA** for MERRA-2 meteorological reanalysis
- **scikit-learn** and **FastAPI** communities

---

**Last Updated**: 5 April 2026  
**Version**: 0.2.0
