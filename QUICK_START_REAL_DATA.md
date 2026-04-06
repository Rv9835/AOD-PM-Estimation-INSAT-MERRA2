# Quick Start: Using Real Data with the Model

This guide will help you **get real data, train a model, and make predictions** in 30 minutes.

---

## 🚀 Quick Path (Easiest)

### Step 1: Fetch Real CPCB Data from OpenAQ (5 min)

OpenAQ has free CPCB data for major Indian cities. No authentication needed!

```bash
# Fetch real data for 2024
python scripts/fetch_realdata_openaq.py \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --output data/raw/cpcb_real.csv

# Check what was downloaded
head -5 data/raw/cpcb_real.csv
wc -l data/raw/cpcb_real.csv  # Should show 1000+ rows
```

**Expected Output:**
```
time,lat,lon,pm25,station_id,city
2024-01-01 00:00:00,28.6139,77.209,45.2,Delhi-ITO,Delhi
2024-01-01 01:00:00,28.6139,77.209,48.5,Delhi-ITO,Delhi
```

### Step 2: Use Mock Data for INSAT & MERRA-2 (For Now)

The system already generates realistic mock INSAT and MERRA-2 data. For your first training, use:

```bash
# Train model with real CPCB data + mock satellite/weather data
python scripts/train_and_evaluate.py --config configs/base.yaml
```

The system will:
1. ✓ Load real CPCB data from `data/raw/cpcb_real.csv`
2. ✓ Generate realistic INSAT satellite data (0.1° grid)
3. ✓ Generate realistic MERRA-2 weather data (0.25° grid)
4. ✓ Join them spatiotemporally using Haversine distance
5. ✓ Train RandomForest model
6. ✓ Evaluate on test set

**Expected Output:**
```
Loaded dataset with shape (500, 9)
Training completed. Metrics: {'rmse': 7.91, 'mae': 6.35, 'r2': 0.70}
Model saved to: artifacts/baseline/random_forest_pm25.joblib
```

### Step 3: Make Predictions

```bash
# Start server
python scripts/run_server.py

# In another terminal, make predictions
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "aod": [0.28, 0.35],
    "temperature": [288, 295],
    "humidity": [55, 45],
    "wind_speed": [3.2, 2.8],
    "boundary_layer_height": [780, 850],
    "lat": [28.61, 19.07],
    "lon": [77.20, 72.87],
    "day_of_year": [45, 46]
  }'
```

**Expected Output:**
```json
{"predictions": [45.3, 52.1]}
```

---

## 📚 Complete Real Data Setup (1-2 hours)

If you want to use **all real data** (CPCB + INSAT + MERRA-2), follow this:

### Part A: Set Up NASA Earthdata Account (10 min)

Needed for MERRA-2:

1. Go to https://urs.earthdata.nasa.gov/
2. Click "Register"
3. Fill in your email and password
4. Verify email
5. Save your username and password

### Part B: Download Real CPCB Data (5 min)

**Option 1: Automatic Download via OpenAQ (Recommended)**
```bash
python scripts/fetch_realdata_openaq.py \
  --start-date 2023-06-01 \
  --end-date 2024-12-31 \
  --output data/raw/cpcb_real.csv
```

**Option 2: Manual Download from CPCB Portal**
1. Visit https://www.cpcb.gov.in/ccr/
2. Go to "Data Downloads"
3. Select State → City → Station
4. Choose date range
5. Download CSV
6. Save to: `data/raw/cpcb_real.csv`

### Part C: Download Real INSAT/AERONET Data (15 min)

**Option 1: From AERONET (Best Quality)**
```bash
# Visit: https://aeronet.gsfc.nasa.gov/cgi-bin/webtool_opera_new
# 1. Select "Data Download Stream"
# 2. Choose "Aerosol Optical Depth v3"
# 3. Select Indian sites (Delhi, Kanpur, Hyderabad, etc.)
# 4. Date range: 2023-06-01 to 2024-12-31
# 5. Download
# 6. Save to: data/raw/aeronet_aod.txt
```

**Option 2: From NOAA (Quick)**
```bash
curl -L "https://www.ncei.noaa.gov/thredds/fileServer/model-ndvi-monthly/v4.1/2024/ndvi_202401.nc" \
  -o data/raw/insat_aod_2024.nc
```

### Part D: Download Real MERRA-2 Data (20 min)

```bash
# Set credentials
export EARTHDATA_USERNAME="your_nasa_username"
export EARTHDATA_PASSWORD="your_nasa_password"

# Download MERRA-2
python scripts/fetch_merra2_opendap.py \
  --start-date 2023-06-01 \
  --end-date 2024-12-31 \
  --output data/raw/merra2_real.csv
```

### Part E: Update Connectors to Use Real Data

Edit the three connector files to read from real data instead of generating mock data:

**File: `src/airpollution/sources/cpcb.py`**
```python
def fetch(self, start_date, end_date, station_ids=None):
    # Load from your downloaded CSV
    df = pd.read_csv("data/raw/cpcb_real.csv")
    df['time'] = pd.to_datetime(df['time'])
    df = df[(df['time'] >= start_date) & (df['time'] <= end_date)]
    if station_ids:
        df = df[df['station_id'].isin(station_ids)]
    self.validate_schema(df)
    return df
```

Similar updates for `insat.py` and `merra2.py` (see REAL_DATA_INTEGRATION_GUIDE.md).

### Part F: Train with All Real Data

```bash
python scripts/train_and_evaluate.py --config configs/base.yaml
```

Monitor progress:
```bash
tail -f artifacts/baseline/training.log
```

Check results:
```bash
cat artifacts/baseline/metrics.json
```

---

## 📊 Expected Results

### With Real CPCB Data Only (1-2 hours setup)
```
Training dataset: 500+ samples
Test RMSE: 8-12 µg/m³ (good)
R²: 0.65-0.75 (fair)
Features used: AOD, Temperature, Humidity, Wind, BLH, Lat, Lon, Day-of-year
```

### With All Real Data (2-4 hours setup)
```
Training dataset: 2000+ samples
Test RMSE: 5-8 µg/m³ (excellent)
R²: 0.75-0.85 (good)
Features: Same as above, with real satellite & weather data
```

---

## 🔄 Data Update Workflow

Once you've set up real data, keep it fresh:

### Weekly Update (Automated)
Create `scripts/weekly_data_refresh.sh`:

```bash
#!/bin/bash

# Update CPCB
python scripts/fetch_realdata_openaq.py \
  --start-date $(date -d "7 days ago" +%Y-%m-%d) \
  --end-date $(date +%Y-%m-%d) \
  --output data/raw/cpcb_real_latest.csv

# Append to existing file
tail -n +2 data/raw/cpcb_real_latest.csv >> data/raw/cpcb_real.csv

# Retrain model
python scripts/train_and_evaluate.py --config configs/base.yaml

echo "Weekly update complete!"
```

Make it executable:
```bash
chmod +x scripts/weekly_data_refresh.sh

# Run weekly (cron example)
# Add to crontab: 0 0 * * 0 /path/to/weekly_data_refresh.sh
```

---

## 🎯 Common Issues & Solutions

### Issue 1: OpenAQ API Rate Limit
```
Error: 429 Too Many Requests
Solution: Wait 1 hour and retry, or use API key from https://openaq.org/
```

### Issue 2: NASA Earthdata Authentication Failed
```
Error: Unauthorized (401)
Solution: 
  1. Check credentials: echo $EARTHDATA_USERNAME
  2. Verify at: https://urs.earthdata.nasa.gov/login
  3. Re-set: export EARTHDATA_PASSWORD="your_password"
```

### Issue 3: No Data in Date Range
```
Error: No CPCB stations in fetched data
Solution:
  1. Try broader date range
  2. Check city spelling (e.g., "Bangalore" vs "Bengaluru")
  3. Use OpenAQ API docs: https://openaq.org/api/v2/locations
```

### Issue 4: Model Accuracy Too Low
```
Problem: RMSE > 15 µg/m³
Solutions:
  1. Increase training data (use 2+ years)
  2. Add more features (satellite angle, land cover, etc.)
  3. Try XGBoost instead of RandomForest
  4. Remove outliers (PM25 > 500 µg/m³)
  5. Use hourly data instead of daily
```

### Issue 5: Out of Memory
```
Error: MemoryError during MERRA-2 download
Solution:
  1. Process monthly instead of annual
  2. Use smaller region bounds
  3. Reduce temporal resolution
```

---

## 📈 Next: Improving Model Performance

### Increase Data Volume
```bash
# Fetch 2 years of CPCB data
python scripts/fetch_realdata_openaq.py \
  --start-date 2023-01-01 \
  --end-date 2024-12-31
```

### Add More Features
Edit `configs/base.yaml`:
```yaml
feature_columns:
  - aod
  - temperature
  - humidity
  - wind_speed
  - boundary_layer_height
  - lat
  - lon
  - day_of_year
  - month_of_year        # ADD
  - hour_of_day          # ADD
  - pressure             # ADD (if available)
  - solar_radiation      # ADD (if available)
```

### Try Advanced Models
```python
# In modeling.py, replace RandomForest with XGBoost
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.1,
    random_state=42
)
```

### Spatial Interpolation
```python
# In join.py, reduce search radius for tighter collocation
insat_radius_km=10.0   # Instead of 15
merra2_radius_km=50.0  # Instead of 75
```

---

## 🚀 Deployment

Once satisfied with model:

### 1. Export Model
```bash
python -c "
import joblib
from airpollution.config import load_config
config = load_config('configs/base.yaml')
print(f'Model path: {config.paths.model_path}')
"
```

### 2. Deploy API
```bash
# Production
gunicorn airpollution.app:app --workers 4 --worker-class uvicorn.workers.UvicornWorker

# Or Docker
docker build -t aod-pm-api .
docker run -p 8000:8000 aod-pm-api
```

### 3. Monitor Predictions
```bash
# Log predictions
tail -f logs/predictions.log

# Check model performance
python scripts/validate_model.py
```

---

## 📚 References

- **OpenAQ API**: https://openaq.org/api/v2/
- **CPCB Data**: https://www.cpcb.gov.in/
- **AERONET Data**: https://aeronet.gsfc.nasa.gov/
- **MERRA-2 Data**: https://disc.gsfc.nasa.gov/
- **NASA Earthdata**: https://urs.earthdata.nasa.gov/

---

## Questions?

Check the full guide: [REAL_DATA_INTEGRATION_GUIDE.md](REAL_DATA_INTEGRATION_GUIDE.md)

---

**Last Updated**: 6 April 2026  
**Time Estimate**: Quick Start = 30 min, Full Setup = 2-4 hours
