# 🌍 Air Quality Prediction Dashboard

A **professional, interactive Streamlit web dashboard** for real-time PM2.5 air pollution predictions using satellite AOD, meteorological data, and machine learning.

## 📸 Dashboard Features

### **Tab 1: Predict** 🔮
- **Interactive sliders & inputs** for meteorological parameters:
  - Aerosol Optical Depth (AOD)
  - Temperature (Kelvin)
  - Humidity (%)
  - Wind Speed (m/s)
  - Boundary Layer Height (m)
  - Geographic location (Latitude/Longitude)
  - Date selection (auto-calculates day of year)

- **Real-time predictions** from FastAPI backend
- **Color-coded air quality status**:
  - 🟢 GOOD (≤12 µg/m³)
  - 🟡 MODERATE (12-35 µg/m³)
  - 🟠 UNHEALTHY FOR SENSITIVE (35-55 µg/m³)
  - 🔴 UNHEALTHY (55-150 µg/m³)
  - 🟣 HAZARDOUS (>150 µg/m³)

- **Air Quality Index (AQI)** calculation
- **Health recommendations** based on PM2.5 levels
- **Parameter details** showing why each input matters

### **Tab 2: Model Metrics** 📊
- **RMSE, MAE, R² scores** with interpretation
- **Model architecture details** (Random Forest: 300 trees, depth=16)
- **Feature importance** explanations
- **Performance interpretation** in practical terms

### **Tab 3: History** 📋
- **Prediction history** (during session)
- **Summary statistics** (count, average, maximum PM2.5)
- **Trend visualization** chart
- **Clear history** button

---

## 🚀 Quick Start

### **Step 1: Start the FastAPI Server** (Terminal 1)

```bash
cd "/Users/chaman/Desktop/HP challenge"
source .venv/bin/activate
python scripts/run_server.py
```

Wait for: `Application startup complete`

### **Step 2: Launch the Dashboard** (Terminal 2)

```bash
cd "/Users/chaman/Desktop/HP challenge"
source .venv/bin/activate
streamlit run app_dashboard.py
```

**Or use the helper script:**
```bash
chmod +x run_dashboard.sh
./run_dashboard.sh
```

### **Step 3: Open Browser**

Streamlit will automatically open at:
```
🌍 http://localhost:8501
```

---

## 📖 How to Use

### **Making a Prediction**

1. **Go to "Predict" tab** (default tab)

2. **Enter meteorological data** using sliders:
   - Set AOD value (0.0-1.0)
   - Set temperature in Kelvin (270-310K)
   - Set humidity percentage (0-100%)
   - Set wind speed (0-10 m/s)
   - Set boundary layer height (300-2000 m)

3. **Set location** (pre-filled for Greater Noida):
   - Latitude: 28.47°N
   - Longitude: 77.48°E

4. **Select date** (day picker automatically calculates day of year)

5. **Click "🔮 Predict PM2.5"** button

6. **View results**:
   - PM2.5 value (µg/m³)
   - Air Quality Index (0-500)
   - Category with color coding
   - Health recommendations
   - Input parameter summary

### **Understanding Model Metrics**

Go to **"📈 Model Metrics"** tab:

| Metric | Meaning | What It Tells Us |
|--------|---------|-----------------|
| **RMSE: 8.89** | Root Mean Squared Error | On average, predictions are off by ±8.89 µg/m³ |
| **MAE: 6.90** | Mean Absolute Error | Typical error is ±6.90 µg/m³ (simpler measure) |
| **R²: 0.687** | Variance Explained | Model explains 68.7% of PM2.5 variation |

### **Viewing Prediction History**

Go to **"📋 History"** tab:
- See all predictions from current session
- View statistics (average, max PM2.5)
- See trend chart if multiple predictions
- Clear history with button

---

## 🎯 Example Scenarios

### **Scenario 1: Clear Day (Low Pollution)**
```
AOD: 0.15
Temperature: 290K
Humidity: 45%
Wind Speed: 4.5 m/s
BLH: 900m
→ Prediction: ~20 µg/m³ (🟢 GOOD)
```

### **Scenario 2: Hazy Day (Moderate Pollution)**
```
AOD: 0.35
Temperature: 288K
Humidity: 65%
Wind Speed: 2.5 m/s
BLH: 700m
→ Prediction: ~35 µg/m³ (🟡 MODERATE)
```

### **Scenario 3: Polluted Day (High Pollution)**
```
AOD: 0.50
Temperature: 283K
Humidity: 75%
Wind Speed: 1.5 m/s
BLH: 400m
→ Prediction: ~55 µg/m³ (🟠 UNHEALTHY FOR SENSITIVE)
```

---

## 🔧 Technical Details

### **Built With**
- **Frontend**: Streamlit 1.56+
- **Backend API**: FastAPI (localhost:8000)
- **Machine Learning**: scikit-learn RandomForest
- **Data Processing**: pandas, numpy
- **HTTP**: requests library

### **Architecture**
```
User Browser (http://localhost:8501)
         ↓
   Streamlit Dashboard (app_dashboard.py)
         ↓
   FastAPI Server (http://localhost:8000)
         ↓
   RandomForest Model
         ↓
   PM2.5 Prediction
```

### **API Endpoints Used**
- `GET /health` - Check server status
- `GET /metrics` - Fetch model performance metrics
- `POST /predict` - Make PM2.5 predictions

---

## 🐛 Troubleshooting

### **"Cannot connect to API server!"**
**Solution**: Make sure the FastAPI server is running:
```bash
# In another terminal
python scripts/run_server.py
```

### **Port 8501 already in use**
```bash
# Find and kill process on port 8501
lsof -ti:8501 | xargs kill -9
streamlit run app_dashboard.py
```

### **"ModuleNotFoundError: No module named 'streamlit'"**
```bash
source .venv/bin/activate
pip install streamlit requests pandas numpy
```

### **Slow predictions**
- Ensure server is responsive: `curl http://localhost:8000/health`
- Check internet connection (API calls might timeout)
- Verify system resources (close other apps)

---

## 📊 Air Quality Standards Reference

### **WHO PM2.5 Guidelines**
| Category | PM2.5 (µg/m³) | Health Risk |
|----------|---------------|------------|
| Good | 0-12 | No health risk |
| Moderate | 12-35 | Sensitive groups affected |
| Unhealthy (Sensitive) | 35-55 | Outdoor activities discouraged |
| Unhealthy | 55-150 | Everyone at risk outdoors |
| Hazardous | >150 | Extreme health risk |

### **Indian CPCB Standards**
- **24-hour limit**: 60 µg/m³
- **Annual average**: 40 µg/m³

---

## 🎨 Customization

### **Change Default Location**
Edit [app_dashboard.py](app_dashboard.py):
```python
latitude = st.number_input(
    "Latitude (°N)",
    value=28.47,  # ← Change here
)
longitude = st.number_input(
    "Longitude (°E)",
    value=77.48,  # ← Change here
)
```

### **Change API URL**
```python
API_BASE_URL = "http://localhost:8000"  # ← Change here for remote API
```

### **Adjust Slider Ranges**
```python
aod = st.slider(
    "Aerosol Optical Depth (AOD)",
    min_value=0.0,      # ← Adjust ranges
    max_value=2.0,      # ← as needed
    value=0.28,
    step=0.01,
)
```

---

## 📱 Advanced Features

### **Use on Mobile Device**
1. Get your computer's IP:
   ```bash
   ifconfig | grep "inet " 
   # Look for something like 192.168.1.100
   ```

2. Run dashboard with custom endpoints:
   ```bash
   streamlit run app_dashboard.py --server.address=0.0.0.0
   ```

3. From mobile, visit: `http://<YOUR_IP>:8501`

### **Deploy to Cloud**
1. **Streamlit Cloud** (easiest, free):
   - Push code to GitHub
   - Connect to Streamlit Cloud
   - Auto-deploys on each push

2. **Heroku**:
   ```bash
   git push heroku main
   ```

3. **Docker**:
   ```dockerfile
   FROM python:3.10
   RUN pip install streamlit fastapi uvicorn
   CMD ["streamlit", "run", "app_dashboard.py"]
   ```

---

## 📈 Monitoring

### **Check API Performance**
```bash
# Monitor real-time metrics
watch -n 1 'curl -s http://localhost:8000/metrics | python -m json.tool'
```

### **View Recent Predictions**
- Predictions are stored in **History tab** during session
- No persistent database (session-based)

---

## 📞 Support

**If dashboard doesn't load:**
1. Check FastAPI server: `curl http://localhost:8000/health`
2. Check Streamlit logs (bottom of terminal)
3. Clear browser cache: Cmd+Shift+Delete
4. Restart both servers

**For API issues:**
See main [README.md](../README.md) for troubleshooting

---

**Version**: 1.0  
**Last Updated**: April 2026  
**Status**: ✅ Production Ready
