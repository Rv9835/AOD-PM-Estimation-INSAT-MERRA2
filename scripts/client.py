#!/usr/bin/env python
"""Example client for interacting with the Air Pollution API."""

from datetime import datetime

import requests

BASE_URL = "http://localhost:8000"


def health_check():
    """Check server status."""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:")
    print(response.json())
    print()


def fetch_data():
    """Fetch unified data from sources."""
    payload = {
        "start_date": "2026-01-01T00:00:00",
        "end_date": "2026-01-05T00:00:00",
        "station_ids": ["Delhi-ITO", "Mumbai-Fort"],
        "insat_radius_km": 15.0,
        "merra2_radius_km": 75.0,
    }
    print("Fetching unified data...")
    response = requests.post(f"{BASE_URL}/fetch-data", json=payload)
    result = response.json()
    print(f"Status: {result['status']}")
    print(f"Rows: {result['rows']}")
    print(f"Columns: {result['columns']}")
    print(f"Date range: {result['date_range']}")
    print(f"Stations: {result['stations']}")
    print()


def train():
    """Train a new model."""
    print("Training model...")
    response = requests.post(f"{BASE_URL}/train")
    metrics = response.json()
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"R²: {metrics['r2']:.4f}")
    print()


def predict():
    """Make predictions for sample features."""
    features = {
        "aod": [0.25, 0.35, 0.20],
        "temperature": [288, 295, 285],
        "humidity": [55, 45, 65],
        "wind_speed": [3.2, 2.8, 4.1],
        "boundary_layer_height": [780, 850, 720],
        "lat": [28.61, 19.07, 13.08],
        "lon": [77.20, 72.87, 80.27],
        "day_of_year": [45, 46, 47],
    }
    print("Making predictions...")
    response = requests.post(f"{BASE_URL}/predict", json=features)
    result = response.json()
    print(f"Predictions: {result['predictions']}")
    print()


def get_metrics():
    """Retrieve cached model metrics."""
    print("Getting model metrics...")
    response = requests.get(f"{BASE_URL}/metrics")
    metrics = response.json()
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"R²: {metrics['r2']:.4f}")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Air Pollution Monitoring API - Example Client")
    print("=" * 60)
    print()

    try:
        health_check()
        fetch_data()
        train()
        predict()
        get_metrics()
        print("All requests completed successfully!")
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to server.")
        print("Make sure the server is running: python scripts/run_server.py")
    except Exception as e:
        print(f"❌ Error: {e}")
