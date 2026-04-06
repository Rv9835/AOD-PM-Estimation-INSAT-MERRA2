# 🚀 Quick Start Guide: Running the Complete System

## What You're Running

1. **FastAPI Server** (localhost:8000) - The AI prediction engine
2. **Streamlit Dashboard** (localhost:8501) - The user-friendly web interface

---

## ⚡ Quickest Way (2 Terminals)

### **Terminal 1: Start the API Server**
```bash
cd "/Users/chaman/Desktop/HP challenge"
source .venv/bin/activate
python scripts/run_server.py
```

Wait for output:
```
✓ Application startup complete
[Press Ctrl+C to stop]
```

### **Terminal 2: Start the Dashboard**
```bash
cd "/Users/chaman/Desktop/HP challenge"
source .venv/bin/activate
streamlit run app_dashboard.py
```

You'll see:
```
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

---

## 🌐 Access the Dashboard

Open your browser and go to:
```
http://localhost:8501
```

You'll see this interface:

```
┌─────────────────────────────────────────────────────┐
│ 🌍 Air Quality Prediction System                    │
│                                                     │
│ [📊 Predict]  [📈 Model Metrics]  [📋 History]     │
│                                                     │
│ Make a Prediction                                   │
│                                                     │
│ Satellite & Meteorological Data  │  Location & BL  │
│ ─────────────────────────────────────────────────  │
│ AOD: [━━●━━] 0.28                 Latitude: 28.47  │
│ Temp: [━━●━━] 288K                Longitude: 77.48 │
│ Humidity: [━●━] 55%               Date: [Apr 6]   │
│ Wind Speed: [━●━] 3.2 m/s         BLH: [━━●━━] 780 │
│                                                     │
│  [🔮 Predict PM2.5]  [🔄 Reset]                    │
│                                                     │
│ ✅ Prediction successful!                           │
│                                                     │
│ PM2.5 Level: 29.63 µg/m³                            │
│ AQI: 65 (EPA Scale)                                 │
│ Status: 🟢 GOOD                                     │
│                                                     │
│ 💡 Recommendation: Air quality is satisfactory.    │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 📱 Dashboard Walkthrough

### **Tab 1: 📊 Predict** (Main Tab)
1. **Adjust sliders** for current atmospheric conditions
2. **Click "🔮 Predict PM2.5"** to get predictions
3. **View results** with color-coded air quality status
4. **See recommendations** for outdoor activities

### **Tab 2: 📈 Model Metrics**
- View RMSE, MAE, R² scores
- Understand what metrics mean
- See model architecture (300-tree Random Forest)
- Learn about feature importance

### **Tab 3: 📋 History**
- Track all predictions from this session
- View statistics (average, max PM2.5)
- See trends in a chart
- Clear history when done

---

## 🎯 Example: Making a Prediction

**Step 1: Set Parameters**
```
AOD: 0.28 (use slider)
Temperature: 288K
Humidity: 55%
Wind Speed: 3.2 m/s
Boundary Layer Height: 780m
Location: Greater Noida (28.47°N, 77.48°E)
Date: April 6, 2026
```

**Step 2: Click Predict**
```
🔮 Predict PM2.5
```

**Step 3: View Results**
```
✅ Prediction successful!

┌─── RESULTS ───┐
│ PM2.5: 29.63 µg/m³
│ AQI: 65
│ Status: 🟢 GOOD
│
│ Recommendation: Air quality is satisfactory.
│ Enjoy your outdoor activities!
└───────────────┘
```

---

## 🔌 How It Works (Behind the Scenes)

```
You enter data in browser
         ↓
Streamlit (localhost:8501)
         ↓
   [sends to API]
         ↓
FastAPI Server (localhost:8000)
         ↓
Random Forest Model
         ↓
     [calculates]
         ↓
PM2.5 Prediction (29.63 µg/m³)
         ↓
   [returns to browser]
         ↓
Dashboard displays result 🟢 GOOD
```

---

## 🛠️ Common Commands

### **Stop the API Server**
```bash
# In Terminal 1 where server is running
Ctrl + C
```

### **Stop the Dashboard**
```bash
# In Terminal 2 where Streamlit is running
Ctrl + C
```

### **Check API Health**
```bash
curl http://localhost:8000/health
# Should return: {"status":"ok","service":"airpollution-api"}
```

### **Get Current Model Metrics**
```bash
curl http://localhost:8000/metrics
# Returns: {"rmse":8.89,"mae":6.90,"r2":0.687}
```

---

## 📊 Understanding the Air Quality Status

| Status | PM2.5 Range | What It Means | Recommendation |
|--------|------------|--------------|-----------------|
| 🟢 **GOOD** | 0-12 µg/m³ | Air quality is satisfactory | Enjoy outdoor activities |
| 🟡 **MODERATE** | 12-35 µg/m³ | Acceptable air quality | Sensitive groups should limit exertion |
| 🟠 **UNHEALTHY FOR SENSITIVE** | 35-55 µg/m³ | Unhealthy for some | Avoid prolonged outdoor exertion |
| 🔴 **UNHEALTHY** | 55-150 µg/m³ | Unhealthy for everyone | Avoid outdoor activities |
| 🟣 **HAZARDOUS** | >150 µg/m³ | Extremely unhealthy | Stay indoors, close windows |

---

## 🚨 Troubleshooting

### **Problem: "Cannot connect to API server!"**
**Solution:**
1. Check Terminal 1 - is the server still running?
2. If not, restart it:
   ```bash
   python scripts/run_server.py
   ```
3. Verify it's working:
   ```bash
   curl http://localhost:8000/health
   ```

### **Problem: Dashboard won't load**
**Solution:**
1. Check if Streamlit is running on port 8501
2. Kill any process on that port:
   ```bash
   lsof -ti:8501 | xargs kill -9
   ```
3. Restart Streamlit:
   ```bash
   streamlit run app_dashboard.py
   ```

### **Problem: "ModuleNotFoundError"**
**Solution:**
```bash
source .venv/bin/activate
pip install streamlit requests pandas numpy
```

### **Problem: Sliders don't work / Dashboard freezes**
**Solution:**
1. Clear browser cache: `Cmd+Shift+Delete`
2. Close and reopen browser tab
3. Hard refresh: `Cmd+Shift+R`

---

## 📈 Next Steps

1. **Try different scenarios** - Change AOD, humidity, wind speed to see how PM2.5 changes
2. **Monitor trends** - Use History tab to track predictions over time
3. **Share with others** - Network URL shows how to access from other devices
4. **Deploy to cloud** - Push to GitHub and deploy to Streamlit Cloud (free!)

---

## 📚 Full Documentation

For more details, see:
- [DASHBOARD_README.md](DASHBOARD_README.md) - Complete dashboard guide
- [README.md](README.md) - Full project documentation
- [REAL_DATA_INTEGRATION_GUIDE.md](REAL_DATA_INTEGRATION_GUIDE.md) - Data sources and setup

---

**Status**: ✅ Ready to use  
**Version**: 1.0  
**Last Updated**: April 2026
