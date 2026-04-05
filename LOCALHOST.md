# Running on Localhost

Quick-start guide for running the Air Pollution Monitoring API on your local machine.

## Installation

1. **Create and activate virtual environment** (if not already done):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -e .[dev]
   ```
   This installs the project, including FastAPI, uvicorn, and all data science libraries.

## Starting the Server

### Option 1: Simple Start (Recommended)
```bash
python scripts/run_server.py
```
Server will start on **http://localhost:8000**

### Option 2: Custom Port
```bash
python scripts/run_server.py 9000  # Use port 9000 instead
```

### Option 3: Direct uvicorn (Advanced)
```bash
uvicorn airpollution.app:app --host 127.0.0.1 --port 8000 --reload
```

## Using the API

### Interactive API Documentation
Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

Both provide interactive endpoints where you can test requests directly in your browser.

### Available Endpoints

#### 1. **Health Check**
```bash
curl http://localhost:8000/health
```
Response:
```json
{"status": "ok", "service": "airpollution-api"}
```

#### 2. **Fetch Unified Data** (CPCB + INSAT + MERRA-2)
```bash
curl -X POST http://localhost:8000/fetch-data \
  -H "Content-Type: application/json" \
  -d '{
    "start_date": "2026-01-01T00:00:00",
    "end_date": "2026-01-05T00:00:00",
    "station_ids": ["Delhi-ITO", "Mumbai-Fort"],
    "insat_radius_km": 15.0,
    "merra2_radius_km": 75.0
  }'
```
Response:
```json
{
  "status": "success",
  "rows": 8,
  "columns": ["time", "lat", "lon", "station_id", "pm25", "pm10", "aod", "cloud_fraction", "temperature", "humidity", "wind_speed", "boundary_layer_height"],
  "date_range": {"start": "2026-01-01T00:00:00", "end": "2026-01-05T00:00:00"},
  "stations": ["Delhi-ITO", "Mumbai-Fort"]
}
```

#### 3. **Train a Model**
```bash
curl -X POST http://localhost:8000/train
```
Response:
```json
{"rmse": 7.91, "mae": 6.35, "r2": 0.699}
```

#### 4. **Make Predictions**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "aod": [0.25, 0.35],
    "temperature": [288, 295],
    "humidity": [55, 45],
    "wind_speed": [3.2, 2.8],
    "boundary_layer_height": [780, 850],
    "lat": [28.61, 19.07],
    "lon": [77.20, 72.87],
    "day_of_year": [45, 46]
  }'
```
Response:
```json
{"predictions": [82.3, 95.1]}
```

#### 5. **Get Model Metrics**
```bash
curl http://localhost:8000/metrics
```
Response:
```json
{"rmse": 7.91, "mae": 6.35, "r2": 0.699}
```

## Python Client Example

Run the example client script to test all endpoints:
```bash
python scripts/client.py
```

This runs through all main endpoints and prints results.

## Web Browser Testing

1. Go to http://localhost:8000/docs
2. Click on any endpoint to expand it
3. Click "Try it out" button
4. Fill in parameters (or use defaults)
5. Click "Execute"
6. View the response

## Common Issues

### Port Already in Use
```bash
# Find and kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use a different port
python scripts/run_server.py 8001
```

### Module Not Found
Make sure you activated the virtual environment:
```bash
source .venv/bin/activate
```

### Connection Refused
Make sure the server is running in another terminal. Check:
```bash
ps aux | grep run_server.py
```

## Architecture

```
localhost:8000
├── GET  /           → API info
├── GET  /health     → Server status
├── POST /fetch-data → Fetch unified data from sources
├── POST /train      → Train RandomForest model
├── POST /predict    → Predict PM values
└── GET  /metrics    → Get cached model metrics

Data Flow:
  /fetch-data → SpatiotemporalJoiner (CPCB + INSAT + MERRA-2)
                    ↓
                Unified DataFrame
                    ↓
  /train → Preprocessing → RandomForestRegressor → /metrics
                    ↓
  /predict ← (uses cached model)
```

## API Features

- **Auto-reload on code changes** (development mode)
- **Interactive API documentation** (Swagger/ReDoc)
- **Request/response validation** (Pydantic models)
- **Error messages** with helpful context
- **Logging** to console

## Next Steps

- Deploy to production (e.g., Heroku, AWS, GCP)
- Add database layer (PostgreSQL) for persistent storage
- Add authentication (JWT tokens)
- Create geospatial visualization endpoints
- Build real-time monitoring dashboard

---

For more details, see [README.md](README.md) and [API documentation](http://localhost:8000/docs).
