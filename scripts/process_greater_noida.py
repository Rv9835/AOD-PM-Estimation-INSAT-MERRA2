#!/usr/bin/env python3
"""
Process Greater Noida CPCB data and train the model.

Converts OpenAQ format to training dataset with synthetic satellite/weather features.
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def process_greater_noida_data(input_file: str, output_file: str) -> None:
    """Process Greater Noida CPCB data for training."""
    
    print(f"📂 Loading Greater Noida CPCB data from {input_file}...")
    df_raw = pd.read_csv(input_file)
    
    print(f"✓ Total records: {len(df_raw)}")
    print(f"✓ Unique parameters: {df_raw['parameter'].unique()}")
    
    # Extract PM2.5 measurements
    df_pm25 = df_raw[df_raw["parameter"] == "pm25"].copy()
    print(f"✓ PM2.5 records: {len(df_pm25)}")
    
    if df_pm25.empty:
        print("❌ No PM2.5 data found! Checking available parameters...")
        print(df_raw['parameter'].value_counts())
        return
    
    # Extract PM10 measurements
    df_pm10 = df_raw[df_raw["parameter"] == "pm10"].copy()
    print(f"✓ PM10 records: {len(df_pm10)}")
    
    # Process PM2.5
    df_pm25 = df_pm25.rename(columns={"value": "pm25", "datetimeUtc": "time"})
    df_pm25["time"] = pd.to_datetime(df_pm25["time"])
    df_pm25 = df_pm25[["time", "latitude", "longitude", "pm25", "location_name"]].copy()
    df_pm25.columns = ["time", "lat", "lon", "pm25", "station_id"]
    
    # Process PM10
    if len(df_pm10) > 0:
        df_pm10 = df_pm10.rename(columns={"value": "pm10", "datetimeUtc": "time"})
        df_pm10["time"] = pd.to_datetime(df_pm10["time"])
        df_pm10 = df_pm10[["time", "latitude", "longitude", "pm10"]].copy()
        df_pm10.columns = ["time", "lat", "lon", "pm10"]
        
        # Merge PM2.5 and PM10 by time
        df_cpcb = df_pm25.merge(df_pm10[["time", "pm10"]], on="time", how="left")
    else:
        df_cpcb = df_pm25.copy()
        df_cpcb["pm10"] = df_cpcb["pm25"] * 1.5  # Estimate
    
    print(f"\n✓ Merged CPCB data: {len(df_cpcb)} samples")
    print(f"  PM2.5 range: {df_cpcb['pm25'].min():.1f} - {df_cpcb['pm25'].max():.1f} µg/m³")
    print(f"  PM2.5 mean: {df_cpcb['pm25'].mean():.1f} µg/m³")
    
    # Generate synthetic satellite and weather features
    print(f"\n🔄 Generating synthetic satellite/weather features...")
    
    np.random.seed(42)
    
    # Add day_of_year
    df_cpcb["day_of_year"] = df_cpcb["time"].dt.dayofyear
    
    # AOD: correlates with PM2.5
    df_cpcb["aod"] = np.clip(
        df_cpcb["pm25"] / 100.0 + np.random.normal(0, 0.12, len(df_cpcb)),
        0.05, 2.5
    )
    
    # Cloud fraction: random
    df_cpcb["cloud_fraction"] = np.clip(
        np.random.beta(2.5, 5, len(df_cpcb)), 0, 1
    )
    
    # Temperature: seasonal variation
    month = df_cpcb["time"].dt.month
    temp_base = np.where(
        month.isin([5, 6]), 305,  # Summer
        np.where(month.isin([12, 1]), 285, 295)  # Winter vs other
    )
    df_cpcb["temperature"] = np.clip(
        temp_base + np.random.normal(0, 5, len(df_cpcb)),
        270, 315
    )
    
    # Humidity: seasonal
    humidity_base = np.where(
        month.isin([7, 8, 9]), 75,  # Monsoon
        np.where(month.isin([11, 12, 1, 2]), 55, 60)  # Winter vs other
    )
    df_cpcb["humidity"] = np.clip(
        humidity_base + np.random.normal(0, 12, len(df_cpcb)),
        15, 95
    )
    
    # Wind speed: inverse correlation with PM25
    wind_base = 2 + np.random.normal(0, 1, len(df_cpcb))
    df_cpcb["wind_speed"] = np.clip(
        wind_base - (df_cpcb["pm25"] - 50) / 100.0,
        0.2, 15
    )
    
    # Boundary layer height: inverse correlation with PM25
    blh_base = 800
    df_cpcb["boundary_layer_height"] = np.clip(
        blh_base * (1 - (df_cpcb["pm25"] - 50) / 200.0) + np.random.normal(0, 150, len(df_cpcb)),
        100, 3000
    )
    
    # Reorder columns
    df_final = df_cpcb[[
        "time", "lat", "lon", "pm25", "pm10", "aod", "cloud_fraction",
        "temperature", "humidity", "wind_speed", "boundary_layer_height",
        "day_of_year", "station_id"
    ]]
    
    # Save
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_file, index=False)
    
    print(f"✅ Training data saved to {output_file}")
    print(f"\n📊 Final Dataset:")
    print(f"  Samples: {len(df_final)}")
    print(f"  Date range: {df_final['time'].min()} to {df_final['time'].max()}")
    print(f"  PM2.5: {df_final['pm25'].min():.1f} - {df_final['pm25'].max():.1f} µg/m³")
    print(f"  All features present: ✓")


if __name__ == "__main__":
    process_greater_noida_data(
        "openaq_location_6978_measurments.csv",
        "data/processed/greater_noida_training.csv"
    )
