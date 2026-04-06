import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Page configuration
st.set_page_config(
    page_title="Air Quality Prediction",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .good { color: #00cc44; font-weight: bold; }
    .moderate { color: #ffdd00; font-weight: bold; }
    .unhealthy { color: #ff6600; font-weight: bold; }
    .very_unhealthy { color: #cc0033; font-weight: bold; }
    .hazardous { color: #660099; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# API configuration
API_BASE_URL = "http://localhost:8000"

# Title and description
st.title("🌍 Air Quality Prediction System")
st.markdown("### Real-time PM2.5 Prediction for Greater Noida")

# Initialize session state for history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    location_name = st.text_input("📍 Location Name", value="Greater Noida", help="For reference only")
    st.divider()
    st.markdown("**Model Information**")
    st.info("""
    - **Algorithm**: Random Forest (300 trees)
    - **Training Data**: 800 samples from CPCB
    - **Test Performance**: R² = 0.687
    - **RMSE**: 8.89 µg/m³
    """)

# Main content area with tabs
tab1, tab2, tab3 = st.tabs(["📊 Predict", "📈 Model Metrics", "📋 History"])

with tab1:
    st.header("Make a Prediction")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📡 Satellite & Meteorological Data")
        
        aod = st.slider(
            "Aerosol Optical Depth (AOD)",
            min_value=0.0,
            max_value=1.0,
            value=0.28,
            step=0.01,
            help="Lower values = clearer air, higher = more aerosols"
        )
        
        temperature = st.slider(
            "Temperature (Kelvin)",
            min_value=270.0,
            max_value=310.0,
            value=288.0,
            step=1.0,
            help="Typical range 270-310K. 288K ≈ 15°C"
        )
        
        humidity = st.slider(
            "Humidity (%)",
            min_value=0,
            max_value=100,
            value=55,
            step=5,
            help="Relative humidity percentage"
        )
        
        wind_speed = st.slider(
            "Wind Speed (m/s)",
            min_value=0.0,
            max_value=10.0,
            value=3.2,
            step=0.1,
            help="Higher wind = better dispersion of pollutants"
        )
    
    with col2:
        st.subheader("🌦️ Location & Boundary Layer")
        
        boundary_layer_height = st.slider(
            "Boundary Layer Height (m)",
            min_value=300.0,
            max_value=2000.0,
            value=780.0,
            step=50.0,
            help="Higher BLH = better pollutant dispersion"
        )
        
        latitude = st.number_input(
            "Latitude (°N)",
            min_value=-90.0,
            max_value=90.0,
            value=28.47,
            step=0.01,
            help="Greater Noida: 28.47°N"
        )
        
        longitude = st.number_input(
            "Longitude (°E)",
            min_value=-180.0,
            max_value=180.0,
            value=77.48,
            step=0.01,
            help="Greater Noida: 77.48°E"
        )
        
        # Date input to calculate day of year
        date_input = st.date_input(
            "Date",
            value=datetime.now(),
            help="Select date to calculate day of year"
        )
        day_of_year = date_input.timetuple().tm_yday
    
    st.divider()
    
    # Prediction button
    col_button1, col_button2, col_button3 = st.columns([2, 2, 2])
    
    with col_button1:
        predict_btn = st.button("🔮 Predict PM2.5", use_container_width=True, type="primary")
    
    with col_button2:
        reset_btn = st.button("🔄 Reset", use_container_width=True)
    
    if reset_btn:
        st.rerun()
    
    # Make prediction
    if predict_btn:
        with st.spinner("📡 Fetching prediction from API..."):
            try:
                # Prepare request payload
                payload = {
                    "aod": [aod],
                    "temperature": [temperature],
                    "humidity": [humidity],
                    "wind_speed": [wind_speed],
                    "boundary_layer_height": [boundary_layer_height],
                    "lat": [latitude],
                    "lon": [longitude],
                    "day_of_year": [day_of_year]
                }
                
                # Call API
                response = requests.post(
                    f"{API_BASE_URL}/predict",
                    json=payload,
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    pm25_value = result['predictions'][0]
                    
                    # Determine air quality category
                    if pm25_value <= 12:
                        category = "🟢 GOOD"
                        color = "good"
                        aqi = int(pm25_value / 12 * 50)
                        advice = "Air quality is satisfactory. Enjoy your outdoor activities!"
                    elif pm25_value <= 35:
                        category = "🟡 MODERATE"
                        color = "moderate"
                        aqi = int(50 + (pm25_value - 12) / 23 * 50)
                        advice = "Sensitive individuals should reduce prolonged outdoor exertion."
                    elif pm25_value <= 55:
                        category = "🟠 UNHEALTHY FOR SENSITIVE GROUPS"
                        color = "unhealthy"
                        aqi = int(100 + (pm25_value - 35) / 20 * 50)
                        advice = "Members of sensitive groups should avoid outdoor activities."
                    elif pm25_value <= 150:
                        category = "🔴 UNHEALTHY"
                        color = "very_unhealthy"
                        aqi = int(150 + (pm25_value - 55) / 95 * 50)
                        advice = "Everyone should avoid outdoor activities."
                    else:
                        category = "🟣 HAZARDOUS"
                        color = "hazardous"
                        aqi = 500
                        advice = "Everyone should avoid ALL outdoor exertion."
                    
                    # Display results
                    st.success("✅ Prediction successful!")
                    
                    # Results card
                    result_col1, result_col2, result_col3 = st.columns(3)
                    
                    with result_col1:
                        st.metric(
                            "PM2.5 Level",
                            f"{pm25_value:.2f} µg/m³",
                            delta=f"{pm25_value - 25:.2f} (vs avg 25)"
                        )
                    
                    with result_col2:
                        st.metric(
                            "Air Quality Index",
                            aqi,
                            delta="EPA Scale 0-500"
                        )
                    
                    with result_col3:
                        st.markdown(f"<div class='{color}'>{category}</div>", unsafe_allow_html=True)
                    
                    st.divider()
                    
                    # Advice box
                    st.info(f"💡 **Recommendation**: {advice}")
                    
                    # Input parameters summary
                    with st.expander("📋 Input Parameters"):
                        params_df = pd.DataFrame({
                            "Parameter": ["AOD", "Temperature", "Humidity", "Wind Speed", "Boundary Layer Height", "Latitude", "Longitude", "Day of Year"],
                            "Value": [f"{aod:.2f}", f"{temperature:.1f}K", f"{humidity}%", f"{wind_speed:.1f}m/s", f"{boundary_layer_height:.0f}m", f"{latitude:.2f}°", f"{longitude:.2f}°", f"{day_of_year}"],
                            "Relevance to PM2.5": [
                                "Higher AOD = more aerosols in air",
                                "Affects pollutant behavior & mixing",
                                "High humidity can increase PM2.5",
                                "Higher wind disperses pollutants",
                                "Higher BLH better disperses pollutants",
                                "Geographic location affects patterns",
                                "Geographic location affects patterns",
                                "Seasonal variation in emissions"
                            ]
                        })
                        st.dataframe(params_df, use_container_width=True, hide_index=True)
                    
                    # Store in history
                    st.session_state.prediction_history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "location": location_name,
                        "pm25": pm25_value,
                        "aqi": aqi,
                        "category": category,
                        "aod": aod,
                        "temperature": temperature,
                        "humidity": humidity,
                        "wind_speed": wind_speed
                    })
                else:
                    st.error(f"❌ API Error: {response.status_code}")
                    st.error(f"Response: {response.text}")
            
            except requests.exceptions.ConnectionError:
                st.error("❌ **Cannot connect to API server!**")
                st.warning("Make sure the FastAPI server is running:")
                st.code("""
cd "/Users/chaman/Desktop/HP challenge"
source .venv/bin/activate
python scripts/run_server.py
                """)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

with tab2:
    st.header("Model Performance Metrics")
    
    try:
        with st.spinner("📊 Loading model metrics..."):
            metrics_response = requests.get(f"{API_BASE_URL}/metrics", timeout=10)
            
            if metrics_response.status_code == 200:
                metrics = metrics_response.json()
                
                # Display metrics in cards
                m_col1, m_col2, m_col3 = st.columns(3)
                
                with m_col1:
                    st.metric(
                        "RMSE (Root Mean Squared Error)",
                        f"{metrics['rmse']:.2f} µg/m³",
                        help="Average prediction error magnitude"
                    )
                
                with m_col2:
                    st.metric(
                        "MAE (Mean Absolute Error)",
                        f"{metrics['mae']:.2f} µg/m³",
                        help="Average absolute prediction error"
                    )
                
                with m_col3:
                    st.metric(
                        "R² (Coefficient of Determination)",
                        f"{metrics['r2']:.4f}",
                        help="Proportion of variance explained (0-1)"
                    )
                
                st.divider()
                
                # Interpretation
                st.subheader("📖 What These Metrics Mean")
                
                col_interp1, col_interp2 = st.columns(2)
                
                with col_interp1:
                    st.success(f"""
                    **RMSE: {metrics['rmse']:.2f}**
                    - On average, predictions are off by ~{metrics['rmse']:.1f} µg/m³
                    - This is about {(metrics['rmse']/25)*100:.1f}% of typical PM2.5 (25 µg/m³)
                    - Acceptable range for air quality prediction
                    """)
                
                with col_interp2:
                    st.info(f"""
                    **R²: {metrics['r2']:.4f}**
                    - Model explains {metrics['r2']*100:.1f}% of PM2.5 variation
                    - Remaining {(1-metrics['r2'])*100:.1f}% from other factors
                    - Good performance for real-world air quality
                    """)
                
                st.divider()
                
                # Model details
                st.subheader("🤖 Model Architecture")
                model_info = pd.DataFrame({
                    "Aspect": ["Algorithm", "Training Samples", "Test Samples", "Features", "Max Depth", "Trees", "Train-Test Split"],
                    "Details": ["Random Forest Regressor", "800", "200", "8", "16", "300", "80-20"]
                })
                st.dataframe(model_info, use_container_width=True, hide_index=True)
                
                st.subheader("📌 Feature Importance (Model Learning)")
                st.info("""
                Based on training data, the model learned these relationships:
                - **Aerosol Optical Depth (AOD)**: Strong positive correlation with PM2.5
                - **Boundary Layer Height**: Inverse relationship - higher BLH disperses pollutants
                - **Wind Speed**: Helps disperse accumulated pollutants
                - **Latitude/Longitude**: Spatial patterns in Greater Noida region
                - **Day of Year**: Seasonal emissions and meteorological patterns
                """)
            
            else:
                st.error(f"Cannot fetch metrics: {metrics_response.status_code}")
    
    except Exception as e:
        st.error(f"Error loading metrics: {str(e)}")
        st.warning("Make sure the API server is running")

with tab3:
    st.header("Prediction History")
    
    if st.session_state.prediction_history:
        # Convert history to dataframe
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        # Display summary
        col_h1, col_h2, col_h3 = st.columns(3)
        
        with col_h1:
            st.metric("Total Predictions", len(history_df))
        
        with col_h2:
            avg_pm25 = history_df['pm25'].mean()
            st.metric("Average PM2.5", f"{avg_pm25:.2f} µg/m³")
        
        with col_h3:
            max_pm25 = history_df['pm25'].max()
            st.metric("Maximum PM2.5", f"{max_pm25:.2f} µg/m³")
        
        st.divider()
        
        # Show history table
        st.subheader("Recent Predictions")
        st.dataframe(
            history_df[['timestamp', 'location', 'pm25', 'aqi', 'category']],
            use_container_width=True,
            hide_index=True
        )
        
        # Chart
        if len(history_df) > 1:
            st.subheader("PM2.5 Trend")
            history_df['index'] = range(len(history_df))
            chart_data = history_df[['index', 'pm25']].rename(columns={'index': 'Prediction #', 'pm25': 'PM2.5 (µg/m³)'})
            st.line_chart(chart_data.set_index('Prediction #'), use_container_width=True)
        
        # Clear history button
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.prediction_history = []
            st.rerun()
    else:
        st.info("📊 No predictions yet. Go to the **Predict** tab to make your first prediction!")

# Footer
st.divider()
st.markdown("""
---
<div style='text-align: center; color: #666;'>
<small>Air Quality Prediction System v1.0 | Powered by FastAPI + Streamlit | Data: Greater Noida CPCB</small>
</div>
""", unsafe_allow_html=True)
