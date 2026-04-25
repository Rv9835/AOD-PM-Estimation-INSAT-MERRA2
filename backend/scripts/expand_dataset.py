"""
Expand unified_dataset.csv with synthetic multi-city data for training
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(PROJECT_ROOT / "backend"))
sys.path.append(str(PROJECT_ROOT / "backend" / "src"))

import numpy as np
import pandas as pd

# City bounds and characteristics
CITIES_DATA = {
    "delhi": {
        "lat_range": (28.40, 29.00),
        "lon_range": (76.80, 77.50),
        "pm25_mean": 120,
        "pm25_std": 40,
        "samples": 120,
    },
    "mumbai": {
        "lat_range": (18.80, 19.40),
        "lon_range": (72.50, 73.20),
        "pm25_mean": 85,
        "pm25_std": 25,
        "samples": 120,
    },
    "bangalore": {
        "lat_range": (12.80, 13.40),
        "lon_range": (77.40, 78.10),
        "pm25_mean": 65,
        "pm25_std": 20,
        "samples": 100,
    },
    "kolkata": {
        "lat_range": (22.40, 23.00),
        "lon_range": (88.20, 88.80),
        "pm25_mean": 95,
        "pm25_std": 30,
        "samples": 100,
    },
    "hyderabad": {
        "lat_range": (17.20, 17.80),
        "lon_range": (78.40, 79.10),
        "pm25_mean": 75,
        "pm25_std": 22,
        "samples": 100,
    },
}


def generate_city_data(city_name, city_info, random_seed=42):
    """Generate synthetic data for a city."""
    np.random.seed(random_seed)
    
    n_samples = city_info["samples"]
    
    # Spatial data
    lats = np.random.uniform(city_info["lat_range"][0], city_info["lat_range"][1], n_samples)
    lons = np.random.uniform(city_info["lon_range"][0], city_info["lon_range"][1], n_samples)
    
    # Meteorological features
    temps = np.random.uniform(15, 40, n_samples)
    humidities = np.random.uniform(30, 95, n_samples)
    wind_speeds = np.random.uniform(0.5, 8.0, n_samples)
    boundary_layers = np.random.uniform(500, 2000, n_samples)
    
    # Aerosol Optical Depth
    aods = np.random.uniform(0.1, 1.0, n_samples)
    
    # Target: PM2.5 (influenced by features)
    pm25 = (
        city_info["pm25_mean"]
        + np.random.normal(0, city_info["pm25_std"], n_samples)
        + aods * 50
        + (40 - temps) * 1.5
        + (humidities / 100) * 30
        - wind_speeds * 5
    )
    pm25 = np.clip(pm25, 10, 300)  # Realistic PM2.5 range
    
    # Day of year (for temporal splits)
    days = np.random.randint(1, 366, n_samples)
    
    return pd.DataFrame({
        "lat": lats,
        "lon": lons,
        "aod": aods,
        "temperature": temps,
        "humidity": humidities,
        "wind_speed": wind_speeds,
        "boundary_layer_height": boundary_layers,
        "day_of_year": days,
        "pm25": pm25,
    })


def main():
    """Generate expanded multi-city dataset."""
    output_path = Path("data/processed/unified_dataset.csv")
    
    print("📊 Generating expanded multi-city synthetic dataset...")
    
    all_data = []
    
    for city_name, city_info in CITIES_DATA.items():
        print(f"  • Generating {city_info['samples']} samples for {city_name}...")
        city_data = generate_city_data(city_name, city_info)
        all_data.append(city_data)
    
    # Combine all cities
    df = pd.concat(all_data, ignore_index=True)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Created {output_path}")
    print(f"  • Total samples: {len(df)}")
    print(f"  • Columns: {', '.join(df.columns)}")
    print(f"  • PM2.5 range: {df['pm25'].min():.1f} - {df['pm25'].max():.1f} µg/m³")
    
    # City distribution
    print("\n  City distribution:")
    for city_name in CITIES_DATA.keys():
        city_bounds = CITIES_DATA[city_name]
        city_mask = (
            (df["lat"] >= city_bounds["lat_range"][0])
            & (df["lat"] <= city_bounds["lat_range"][1])
            & (df["lon"] >= city_bounds["lon_range"][0])
            & (df["lon"] <= city_bounds["lon_range"][1])
        )
        count = city_mask.sum()
        print(f"    {city_name:12s}: {count:3d} samples")


if __name__ == "__main__":
    main()
