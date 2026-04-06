# Real Data Integration Guide

This guide explains how to integrate **real, actual datasets** from CPCB, INSAT, and MERRA-2 instead of mock data, and how to train the model for accurate predictions.

---

## Table of Contents

1. [Data Sources & APIs](#data-sources--apis)
2. [Downloading Real Data](#downloading-real-data)
3. [Integrating Real Data](#integrating-real-data)
4. [Training with Real Data](#training-with-real-data)
5. [Making Predictions](#making-predictions)
6. [Production Setup](#production-setup)

---

## Data Sources & APIs

### 1. CPCB (Central Pollution Control Board)

**🎯 Ground Station Data: PM2.5, PM10, NO₂, SO₂, O₃**

#### Primary Sources

| Source | URL | Format | Auth | Cost |
|--------|-----|--------|------|------|
| **CPCB Official** | https://www.cpcb.gov.in/ccr/ | CSV, JSON | Email request | Free |
| **AQI India** | https://www.aqi.in/ | Web scrape | None | Free |
| **OpenAQ API** | https://openaq.org/ | JSON API | API Key (free) | Free |
| **IQAir** | https://www.iqair.com/india | Web scrape | None | Free |
| **WAQI (World AQI)** | https://waqi.info/api/ | JSON API | API Key | Free |

#### Download Data (2023-2024)

**Option 1: CPCB Official Portal (Recommended)**
```bash
# Visit: https://www.cpcb.gov.in/ccr/
# Steps:
# 1. Go to "Data Downloads"
# 2. Select State, City, Station
# 3. Choose Date Range
# 4. Download CSV
# 5. Save to: data/raw/cpcb_stations_2023_2024.csv
```

**Option 2: OpenAQ API (Programmatic)**
```bash
# Install OpenAQ client
pip install openaq

# Python script to fetch data
python scripts/fetch_cpcb_openaq.py
```

**Option 3: Web Scraping (AQI India)**
```bash
pip install requests beautifulsoup4

# Fetch latest data
python scripts/scrape_aqi_india.py --start-date 2023-01-01 --end-date 2024-12-31
```

#### Data Format Expected

```csv
time,lat,lon,pm25,pm10,station_id,no2,so2,o3,city,state
2024-01-01 00:00:00,28.6139,77.2090,45.2,78.5,Delhi-ITO,25.3,12.1,5.2,Delhi,DL
2024-01-01 01:00:00,28.6139,77.2090,48.5,82.1,Delhi-ITO,26.1,13.2,4.9,Delhi,DL
2024-01-01 02:00:00,28.6139,77.2090,52.1,85.3,Delhi-ITO,28.5,14.8,4.5,Delhi,DL
```

**Key Columns:**
- `time` (datetime): Measurement timestamp (hourly or daily)
- `lat`, `lon` (float): Station latitude/longitude in degrees
- `pm25`, `pm10` (float): PM concentrations in µg/m³
- `station_id` (str): Unique station identifier
- Optional: NO₂, SO₂, O₃, city, state

---

### 2. INSAT-3D / 3DR / 3DS (Satellite AOD Data)

**🛰️ Aerosol Optical Depth (AOD) at 550 nm**

#### Primary Sources

| Source | URL | Format | Auth | Cost |
|--------|-----|--------|------|------|
| **NOAA NOAA** | https://www.ncei.noaa.gov/ | NetCDF, HDF5 | Free account | Free |
| **NASA GSFC** | https://ozone.sci.gsfc.nasa.gov/ | NetCDF, HDF5 | Free account | Free |
| **Copernicus** | https://cds.climate.copernicus.eu/ | NetCDF, Grib | Free account | Free |
| **EOSDIS** | https://earthdata.nasa.gov/ | NetCDF, HDF5 | Free account | Free |

#### Download Data (2023-2024)

**Option 1: NASA GSFC AERONET Data**
```bash
# Visit: https://aeronet.gsfc.nasa.gov/
# Steps:
# 1. Select "Data Download"
# 2. Choose Indian stations
# 3. Select Date Range
# 4. Download Level 2.0 (quality-assured)
# 5. Save to: data/raw/aeronet_insat_aod.txt
```

**Option 2: NOAA Daily Aerosol Data**
```bash
# Visit: https://www.ncei.noaa.gov/products/satellite-aerosol-optical-depth
# Steps:
# 1. Select region: India (lat 8°-35°N, lon 68°-97°E)
# 2. Choose NetCDF format
# 3. Save to: data/raw/insat_aod_daily_2023_2024.nc
```

**Option 3: Programmatic Download (rasterio + requests)**
```bash
# Install tools
pip install rasterio xarray netcdf4

# Fetch via script
python scripts/fetch_insat_aod.py --start 2023-01-01 --end 2024-12-31
```

#### Data Format Expected

```csv
time,lat,lon,aod,cloud_fraction,quality_flag
2024-01-01 00:00:00,28.5,77.0,0.28,0.15,good
2024-01-01 00:00:00,28.6,77.1,0.31,0.12,good
2024-01-01 00:00:00,28.7,77.2,0.26,0.18,fair
2024-01-01 01:00:00,28.5,77.0,0.30,0.14,good
```

**Key Columns (after regridding to 0.1°):**
- `time` (datetime): Observation timestamp
- `lat`, `lon` (float): Grid cell center in degrees
- `aod` (float): Aerosol Optical Depth (unitless, range 0-5)
- `cloud_fraction` (float): Cloud coverage (0-1)
- Optional: quality_flag, retrieval_method

**If NetCDF file:**
```python
import xarray as xr
import pandas as pd

# Load NetCDF
ds = xr.open_dataset("data/raw/insat_aod_daily_2023_2024.nc")
print(ds)  # Check structure

# Convert to DataFrame
df = ds.to_dataframe('aod').reset_index()
df.to_csv("data/raw/insat_aod_gridded.csv", index=False)
```

---

### 3. MERRA-2 (Meteorological Reanalysis)

**🌡️ Temperature, Humidity, Wind, Boundary Layer Height**

#### Primary Sources

| Source | URL | Format | Auth | Cost |
|--------|-----|--------|------|------|
| **NASA GES DISC** | https://disc.gsfc.nasa.gov/ | NetCDF, HDF5 | NASA Earthdata | Free |
| **RENCI MERRA** | https://rda.ucar.edu/datasets/ds633.3/ | NetCDF, Grib | UCAR account | Free |
| **OpenDAP** | https://goldsmr5.gesdisc.eosdis.nasa.gov/ | NetCDF via HTTP | NASA Earthdata | Free |
| **AWS MERRA-2** | https://registry.opendata.aws/nasa-merra2/ | NetCDF (AWS S3) | AWS account | Free tier |

#### Download Data (2023-2024)

**Option 1: NASA GES DISC (Recommended)**
```bash
# Visit: https://disc.gsfc.nasa.gov/datasets/M2T1NXSLV_5.12.4/summary
# Steps:
# 1. Login with NASA Earthdata (free, create at https://urs.earthdata.nasa.gov/)
# 2. Select "Online Archive" → "Get Data"
# 3. Choose Data Range: 2023-2024
# 4. Apply regional filter: lat 8-35°N, lon 68-97°E
# 5. Download NetCDF monthly files
# 6. Save to: data/raw/merra2_monthly_2023_2024.nc
```

**Option 2: OpenDAP (Programmatic)**
```bash
pip install pydap

python scripts/fetch_merra2_opendap.py --start 2023-01-01 --end 2024-12-31
```

**Option 3: AWS S3 (Fastest)**
```bash
pip install boto3 xarray

python scripts/fetch_merra2_aws.py --start 2023-01-01 --end 2024-12-31
```

#### Data Format Expected

```csv
time,lat,lon,temperature,humidity,wind_speed,boundary_layer_height
2024-01-01 00:00:00,28.5,77.0,285.2,55.3,3.2,780
2024-01-01 03:00:00,28.5,77.0,283.1,60.2,2.8,650
2024-01-01 06:00:00,28.5,77.0,281.5,68.5,2.2,520
2024-01-01 09:00:00,28.5,77.0,288.3,45.1,4.1,950
```

**Key Columns:**
- `time` (datetime): Model timestamp (3-hourly or daily)
- `lat`, `lon` (float): 0.5° or 0.625° grid in degrees
- `temperature` (float): K (Kelvin), range 200-330 K
- `humidity` (float): %, range 0-100
- `wind_speed` (float): m/s
- `boundary_layer_height` (float): m (meters)

**If NetCDF file:**
```python
import xarray as xr
import pandas as pd

# Load MERRA-2 NetCDF
ds = xr.open_dataset("data/raw/merra2_monthly_2023_2024.nc")
print(ds)  # Variables: T, QV, U, V, PBLH, etc.

# Extract and convert to DataFrame
merra2_df = ds[['T', 'QV', 'U', 'V', 'PBLH']].to_dataframe().reset_index()
merra2_df = merra2_df.rename(columns={
    'T': 'temperature',
    'QV': 'specific_humidity',
    'U': 'u_wind',
    'V': 'v_wind',
    'PBLH': 'boundary_layer_height'
})
merra2_df.to_csv("data/raw/merra2_gridded.csv", index=False)
```

---

## Downloading Real Data

### Step-by-Step Workflow

#### 1. Create Data Directory
```bash
mkdir -p data/raw data/processed
```

#### 2. Download CPCB Data
```bash
# Download from CPCB official or OpenAQ
# Save as: data/raw/cpcb_2023_2024.csv
```

#### 3. Download INSAT AOD Data
```bash
# Download from NOAA GSFC or AERONET
# Save as: data/raw/insat_aod_2023_2024.csv or .nc
```

#### 4. Download MERRA-2 Data
```bash
# Download from NASA GES DISC
# Save as: data/raw/merra2_2023_2024.nc
```

#### 5. Verify Downloaded Files
```bash
ls -lh data/raw/
# Expected:
# cpcb_2023_2024.csv         (5-50 MB)
# insat_aod_2023_2024.csv    (10-100 MB)
# merra2_2023_2024.nc        (100-500 MB)
```

---

## Integrating Real Data

### Modify Data Connectors to Use Real Files

#### Step 1: Update CPCB Connector

**File: `src/airpollution/sources/cpcb.py`**

Replace the `fetch()` method:

```python
def fetch(  # type: ignore[override]
    self,
    start_date: datetime,
    end_date: datetime,
    station_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Fetch real CPCB data from CSV file."""
    
    # Path to real CPCB CSV
    cpcb_file = "data/raw/cpcb_2023_2024.csv"
    
    # Load data
    df = pd.read_csv(cpcb_file)
    df['time'] = pd.to_datetime(df['time'])
    
    # Filter by date range
    df = df[(df['time'] >= start_date) & (df['time'] <= end_date)]
    
    # Filter by station IDs if provided
    if station_ids:
        df = df[df['station_id'].isin(station_ids)]
    
    # Validate schema
    self.validate_schema(df)
    
    return df
```

#### Step 2: Update INSAT Connector

**File: `src/airpollution/sources/insat.py`**

Replace the `fetch()` method:

```python
def fetch(  # type: ignore[override]
    self,
    start_date: datetime,
    end_date: datetime,
    region_bounds: tuple[float, float, float, float] | None = None,
) -> pd.DataFrame:
    """Fetch real INSAT AOD data."""
    
    # Handle NetCDF or CSV
    insat_file = "data/raw/insat_aod_2023_2024.nc"
    
    if insat_file.endswith('.nc'):
        import xarray as xr
        ds = xr.open_dataset(insat_file)
        df = ds[['aod', 'cloud_fraction']].to_dataframe().reset_index()
    else:
        # CSV format
        df = pd.read_csv(insat_file)
    
    df['time'] = pd.to_datetime(df['time'])
    
    # Filter by date range
    df = df[(df['time'] >= start_date) & (df['time'] <= end_date)]
    
    # Filter by region bounds if provided
    if region_bounds:
        lat_min, lon_min, lat_max, lon_max = region_bounds
        df = df[(df['lat'] >= lat_min) & (df['lat'] <= lat_max) &
                (df['lon'] >= lon_min) & (df['lon'] <= lon_max)]
    
    self.validate_schema(df)
    return df
```

#### Step 3: Update MERRA-2 Connector

**File: `src/airpollution/sources/merra2.py`**

Replace the `fetch()` method:

```python
def fetch(  # type: ignore[override]
    self,
    start_date: datetime,
    end_date: datetime,
    region_bounds: tuple[float, float, float, float] | None = None,
) -> pd.DataFrame:
    """Fetch real MERRA-2 meteorological data."""
    
    # Handle NetCDF or CSV
    merra2_file = "data/raw/merra2_2023_2024.nc"
    
    if merra2_file.endswith('.nc'):
        import xarray as xr
        ds = xr.open_dataset(merra2_file)
        df = ds[['temperature', 'humidity', 'wind_speed', 'boundary_layer_height']]\
            .to_dataframe().reset_index()
    else:
        # CSV format
        df = pd.read_csv(merra2_file)
    
    df['time'] = pd.to_datetime(df['time'])
    
    # Filter by date range
    df = df[(df['time'] >= start_date) & (df['time'] <= end_date)]
    
    # Filter by region bounds if provided
    if region_bounds:
        lat_min, lon_min, lat_max, lon_max = region_bounds
        df = df[(df['lat'] >= lat_min) & (df['lat'] <= lat_max) &
                (df['lon'] >= lon_min) & (df['lon'] <= lon_max)]
    
    self.validate_schema(df)
    return df
```

---

## Training with Real Data

### Method 1: Train via Command Line

```bash
python scripts/train_and_evaluate.py --config configs/base.yaml
```

This will:
1. Fetch real data from all three sources
2. Join them spatiotemporally
3. Preprocess features
4. Train RandomForest model
5. Evaluate on test set
6. Save artifacts to `artifacts/baseline/`

### Method 2: Train via Python Script

Create a new script: `scripts/train_with_real_data.py`

```python
from datetime import datetime, timedelta
from airpollution.join import SpatiotemporalJoiner
from airpollution.data import preprocess_dataset, split_features_target
from airpollution.modeling import train_random_forest, predict
from airpollution.evaluation import compute_regression_metrics
from airpollution.config import load_config

# Load config
config = load_config("configs/base.yaml")

# Fetch real data
print("Fetching real data from sources...")
joiner = SpatiotemporalJoiner()
unified_df = joiner.fetch_and_join(
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 31),
    station_ids=None,  # All stations
    insat_radius_km=15.0,
    merra2_radius_km=75.0
)

print(f"Unified dataset shape: {unified_df.shape}")
print(f"Columns: {unified_df.columns.tolist()}")
print(f"Date range: {unified_df['time'].min()} to {unified_df['time'].max()}")
print(f"Stations: {unified_df['station_id'].nunique()}")

# Preprocess
print("\nPreprocessing data...")
data = preprocess_dataset(unified_df, config)

# Split
print("Splitting train/test...")
x_train, x_test, y_train, y_test = split_features_target(data, config)

print(f"Train: {x_train.shape}, Test: {x_test.shape}")

# Train
print("\nTraining RandomForest...")
model = train_random_forest(config, x_train, y_train)

# Evaluate
print("Evaluating on test set...")
predictions = predict(model, x_test)
metrics = compute_regression_metrics(y_test.to_numpy(), predictions.to_numpy())

print(f"\nMetrics:")
print(f"  RMSE: {metrics.rmse:.4f} µg/m³")
print(f"  MAE:  {metrics.mae:.4f} µg/m³")
print(f"  R²:   {metrics.r2:.4f}")

# Feature importance
print(f"\nFeature Importance (Top 5):")
importance = sorted(
    zip(config.model.feature_columns, model.feature_importances_),
    key=lambda x: x[1],
    reverse=True
)
for feat, imp in importance[:5]:
    print(f"  {feat}: {imp:.4f}")
```

Run it:
```bash
python scripts/train_with_real_data.py
```

### Method 3: Train via API

Start server:
```bash
python scripts/run_server.py
```

Train via API:
```bash
curl -X POST http://localhost:8000/train
```

---

## Making Predictions

### Method 1: Batch Predictions from CSV

Create: `scripts/predict_batch.py`

```python
import pandas as pd
import numpy as np
import joblib
from airpollution.config import load_config

# Load trained model
config = load_config("configs/base.yaml")
model = joblib.load(config.paths.model_path)

# Load test data or new data
input_file = "data/test_features.csv"  # Your prediction input
df = pd.read_csv(input_file)

# Extract features in correct order
X = df[config.model.feature_columns]

# Make predictions
predictions = model.predict(X)

# Add to dataframe
df['predicted_pm25'] = predictions

# Save
df.to_csv("data/predictions_output.csv", index=False)

print(f"Predictions saved to data/predictions_output.csv")
print(f"Min: {predictions.min():.2f}, Max: {predictions.max():.2f}, Mean: {predictions.mean():.2f}")
```

Run it:
```bash
python scripts/predict_batch.py
```

### Method 2: Single Prediction via API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "aod": [0.28],
    "temperature": [288.2],
    "humidity": [55.3],
    "wind_speed": [3.2],
    "boundary_layer_height": [780],
    "lat": [28.61],
    "lon": [77.20],
    "day_of_year": [45]
  }'
# Response: {"predictions": [45.3]}
```

### Method 3: Python Client

```python
import requests
import numpy as np

API_URL = "http://localhost:8000"

# Single prediction
response = requests.post(f"{API_URL}/predict", json={
    "aod": [0.28, 0.35, 0.20],
    "temperature": [288, 295, 285],
    "humidity": [55, 45, 65],
    "wind_speed": [3.2, 2.8, 4.1],
    "boundary_layer_height": [780, 850, 720],
    "lat": [28.61, 19.07, 13.08],
    "lon": [77.20, 72.87, 80.27],
    "day_of_year": [45, 46, 47]
})

predictions = response.json()['predictions']
print(f"Predictions: {predictions}")
```

### Method 4: Production Inference Script

Create: `scripts/inference_server.py` (standalone server for predictions)

```python
import joblib
import pandas as pd
from datetime import datetime
from airpollution.config import load_config

class InferenceEngine:
    def __init__(self, model_path: str):
        self.config = load_config("configs/base.yaml")
        self.model = joblib.load(model_path)
    
    def predict(self, features_dict: dict) -> float:
        """Single prediction."""
        X = pd.DataFrame([{
            col: features_dict.get(col)
            for col in self.config.model.feature_columns
        }])
        return float(self.model.predict(X)[0])
    
    def predict_batch(self, features_list: list[dict]) -> list[float]:
        """Batch predictions."""
        X = pd.DataFrame(features_list)[self.config.model.feature_columns]
        return self.model.predict(X).tolist()

# Usage
engine = InferenceEngine("artifacts/baseline/random_forest_pm25.joblib")

# Single
pm25 = engine.predict({
    "aod": 0.28,
    "temperature": 288.2,
    "humidity": 55.3,
    "wind_speed": 3.2,
    "boundary_layer_height": 780,
    "lat": 28.61,
    "lon": 77.20,
    "day_of_year": 45
})
print(f"PM2.5 Prediction: {pm25:.2f} µg/m³")

# Batch
predictions = engine.predict_batch([
    {"aod": 0.28, "temperature": 288, "humidity": 55, ...},
    {"aod": 0.35, "temperature": 295, "humidity": 45, ...},
])
```

---

## Production Setup

### Recommended Architecture

```
┌─────────────────────────────────────┐
│   Real Data Sources                 │
│  (CPCB, INSAT, MERRA-2)             │
└────────────────┬────────────────────┘
                 │
    ┌────────────▼────────────┐
    │  Data Ingestion Layer   │
    │  (Scheduled ETL)        │
    │  (Daily/Weekly)         │
    └────────────┬────────────┘
                 │
    ┌────────────▼────────────┐
    │  PostgreSQL DB          │
    │  (Cache joined data)    │
    └────────────┬────────────┘
                 │
    ┌────────────▼────────────┐
    │  Training Pipeline      │
    │  (Weekly/Monthly)       │
    │  (Update model)         │
    └────────────┬────────────┘
                 │
    ┌────────────▼────────────┐
    │  Model Registry         │
    │  (Version control)      │
    │  (S3/GCS bucket)        │
    └────────────┬────────────┘
                 │
    ┌────────────▼────────────┐
    │  FastAPI Inference      │
    │  (Load balanced)        │
    │  (Real-time)            │
    └────────────┬────────────┘
                 │
    ┌────────────▼────────────┐
    │  Monitoring & Alerts    │
    │  (Datadog, Prometheus)  │
    └─────────────────────────┘
```

### Docker Deployment

Create: `Dockerfile`

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY artifacts/ artifacts/

RUN pip install -e .

CMD ["python", "scripts/run_server.py"]
```

Build & Run:
```bash
docker build -t aod-pm-api .
docker run -p 8000:8000 aod-pm-api
```

### Scheduled Training (Using APScheduler)

Create: `scripts/scheduled_training.py`

```python
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
import subprocess

def retrain_weekly():
    """Retrain model every Monday at 00:00 UTC."""
    print(f"[{datetime.now()}] Starting model retraining...")
    subprocess.run(["python", "scripts/train_with_real_data.py"])
    print(f"[{datetime.now()}] Retraining complete.")

scheduler = BackgroundScheduler()
scheduler.add_job(retrain_weekly, 'cron', day_of_week=0, hour=0)  # Monday at midnight
scheduler.start()

if __name__ == "__main__":
    print("Scheduler started. Press Ctrl+C to exit.")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        scheduler.shutdown()
```

---

## Troubleshooting

### Issue 1: "File not found" for real data
```python
# Solution: Verify file exists
import os
assert os.path.exists("data/raw/cpcb_2023_2024.csv"), "CPCB file missing"
```

### Issue 2: "Column mismatch" error
```python
# Solution: Check column names
df = pd.read_csv("data/raw/cpcb_2023_2024.csv")
print(df.columns.tolist())  # Compare with REQUIRED_COLUMNS
```

### Issue 3: Date parsing issues
```python
# Solution: Explicit datetime parsing
df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
```

### Issue 4: Memory error with large files
```python
# Solution: Process in chunks
chunksize = 100000
for chunk in pd.read_csv("data/raw/large_file.csv", chunksize=chunksize):
    # Process chunk
    pass
```

### Issue 5: Model accuracy too low
```
Solutions:
1. Increase training data (2+ years)
2. Add more features (satellite angle, cloud type, etc.)
3. Try advanced models (XGBoost, LSTM)
4. Adjust hyperparameters
5. Check data quality (outliers, missing values)
```

---

## Next Steps

1. ✅ Download real data (CPCB, INSAT, MERRA-2)
2. ✅ Update connector files to use real paths
3. ✅ Test data loading: `pytest tests/test_sources.py`
4. ✅ Train with real data: `python scripts/train_with_real_data.py`
5. ✅ Evaluate metrics
6. ✅ Deploy API: `python scripts/run_server.py`
7. ✅ Make predictions
8. ✅ Monitor model performance
9. ✅ Schedule retraining

---

## References

- [CPCB Data Portal](https://www.cpcb.gov.in/)
- [OpenAQ API](https://openaq.org/api)
- [AERONET Data](https://aeronet.gsfc.nasa.gov/)
- [NASA GES DISC](https://disc.gsfc.nasa.gov/)
- [MERRA-2 Dataset](https://disc.gsfc.nasa.gov/datasets/M2T1NXSLV_5.12.4/summary)
- [Copernicus Climate Data](https://cds.climate.copernicus.eu/)

---

**Last Updated**: 6 April 2026
