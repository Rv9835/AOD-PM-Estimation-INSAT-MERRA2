#!/usr/bin/env python3
"""
Fetch real CPCB ground station data from OpenAQ API.

This script downloads actual PM2.5 and PM10 measurements from Indian stations
using the free OpenAQ API.

Usage:
    python scripts/fetch_realdata_openaq.py --start-date 2024-01-01 --end-date 2024-12-31
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests


class OpenAQFetcher:
    """Fetch air quality data from OpenAQ API."""

    BASE_URL = "https://api.openaq.org/v2"
    TIMEOUT = 30

    # Major Indian cities with reference lat/lon
    INDIAN_CITIES = {
        "Delhi": {"lat": 28.6139, "lon": 77.2090},
        "Mumbai": {"lat": 19.0760, "lon": 72.8777},
        "Bangalore": {"lat": 12.9716, "lon": 77.5946},
        "Hyderabad": {"lat": 17.3850, "lon": 78.4867},
        "Chennai": {"lat": 13.0827, "lon": 80.2707},
        "Kolkata": {"lat": 22.5726, "lon": 88.3639},
        "Pune": {"lat": 18.5204, "lon": 73.8567},
        "Ahmedabad": {"lat": 23.0225, "lon": 72.5714},
        "Jaipur": {"lat": 26.9124, "lon": 75.7873},
        "Lucknow": {"lat": 26.8467, "lon": 80.9462},
    }

    def __init__(self, api_key: str | None = None):
        """
        Initialize fetcher.

        Args:
            api_key: Optional OpenAQ API key (not required for free tier)
        """
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"X-API-Key": api_key})

    def fetch_measurements(
        self,
        start_date: datetime,
        end_date: datetime,
        country: str = "IN",
        cities: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Fetch air quality measurements.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            country: ISO country code (default: India)
            cities: List of cities to fetch (default: all major cities)

        Returns:
            DataFrame with columns: time, lat, lon, pm25, pm10, station_id
        """
        if cities is None:
            cities = list(self.INDIAN_CITIES.keys())

        all_data = []

        for city in cities:
            print(f"\nFetching data for {city}...")
            try:
                # Fetch locations in city
                locations = self._get_locations(city, country)

                if not locations:
                    print(f"  ⚠️  No stations found in {city}")
                    continue

                print(f"  Found {len(locations)} station(s)")

                # Fetch measurements for each location
                for location in locations:
                    location_id = location["id"]
                    location_name = location.get("name", location_id)

                    measurements = self._get_measurements(
                        location_id, start_date, end_date
                    )

                    if measurements:
                        # Extract PM2.5 and PM10
                        pm_data = self._extract_pm_data(
                            measurements, location, city
                        )
                        all_data.extend(pm_data)
                        print(f"    ✓ {location_name}: {len(pm_data)} records")
                    else:
                        print(f"    - {location_name}: no data")

            except Exception as e:
                print(f"  ✗ Error fetching {city}: {e}")

        if not all_data:
            print("\n⚠️  No data fetched!")
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        print(f"\n✓ Total records: {len(df)}")
        return df

    def _get_locations(self, city: str, country: str) -> list[dict]:
        """Get all monitoring locations in a city."""
        endpoint = f"{self.BASE_URL}/locations"
        params = {
            "country": country,
            "city": city,
            "limit": 100,
        }

        try:
            response = self.session.get(
                endpoint, params=params, timeout=self.TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])
        except requests.RequestException as e:
            print(f"    Error fetching locations: {e}")
            return []

    def _get_measurements(
        self, location_id: int, start_date: datetime, end_date: datetime
    ) -> list[dict]:
        """Get measurements for a location in date range."""
        endpoint = f"{self.BASE_URL}/measurements"
        params = {
            "location_id": location_id,
            "date_from": start_date.isoformat(),
            "date_to": end_date.isoformat(),
            "limit": 10000,
        }

        try:
            response = self.session.get(
                endpoint, params=params, timeout=self.TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])
        except requests.RequestException as e:
            print(f"    Error fetching measurements: {e}")
            return []

    @staticmethod
    def _extract_pm_data(
        measurements: list[dict], location: dict, city: str
    ) -> list[dict]:
        """Extract PM2.5 and PM10 data from measurements."""
        pm_records = []
        location_id = location["id"]
        location_name = location.get("name", str(location_id))
        lat = location["coordinates"]["latitude"]
        lon = location["coordinates"]["longitude"]

        for measurement in measurements:
            param = measurement.get("parameter", "").lower()

            # Skip non-PM measurements
            if "pm" not in param:
                continue

            try:
                value = float(measurement["value"])
                timestamp = measurement["date"]["utc"]
                time_obj = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

                if param == "pm25":
                    pm_records.append(
                        {
                            "time": time_obj,
                            "lat": lat,
                            "lon": lon,
                            "pm25": value,
                            "station_id": f"{city}-{location_name}",
                            "city": city,
                        }
                    )
                elif param == "pm10":
                    # Find corresponding PM2.5 record or create new one
                    existing = next(
                        (
                            r
                            for r in pm_records
                            if r["time"] == time_obj and r["lat"] == lat
                        ),
                        None,
                    )
                    if existing:
                        existing["pm10"] = value
                    else:
                        pm_records.append(
                            {
                                "time": time_obj,
                                "lat": lat,
                                "lon": lon,
                                "pm10": value,
                                "station_id": f"{city}-{location_name}",
                                "city": city,
                            }
                        )

            except (ValueError, KeyError, TypeError):
                continue

        return pm_records

    def save_to_csv(self, df: pd.DataFrame, output_path: str) -> None:
        """Save data to CSV file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Ensure required columns
        required = ["time", "lat", "lon", "pm25", "station_id"]
        df = df[required] if set(required).issubset(df.columns) else df

        df.to_csv(output_path, index=False)
        print(f"\n✓ Data saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch real CPCB data from OpenAQ API"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2024-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-12-31",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--cities",
        type=str,
        default=None,
        help="Comma-separated cities (default: all major Indian cities)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/cpcb_real_data.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAQ API key (optional)",
    )

    args = parser.parse_args()

    # Parse dates
    try:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError:
        print("❌ Invalid date format. Use YYYY-MM-DD")
        return

    # Parse cities
    if args.cities:
        cities = [c.strip() for c in args.cities.split(",")]
    else:
        cities = None

    # Fetch data
    print("🔄 Initializing OpenAQ fetcher...")
    fetcher = OpenAQFetcher(api_key=args.api_key)

    print(f"📊 Fetching data from {start_date.date()} to {end_date.date()}...")
    df = fetcher.fetch_measurements(start_date, end_date, cities=cities)

    if not df.empty:
        fetcher.save_to_csv(df, args.output)

        # Summary statistics
        print("\n📈 Data Summary:")
        print(f"  Unique stations: {df['station_id'].nunique()}")
        print(f"  Date range: {df['time'].min()} to {df['time'].max()}")
        print(f"  PM2.5: {df['pm25'].min():.1f} - {df['pm25'].max():.1f} µg/m³")
        if "pm10" in df.columns and not df["pm10"].isna().all():
            print(f"  PM10:  {df['pm10'].min():.1f} - {df['pm10'].max():.1f} µg/m³")
    else:
        print("\n⚠️  No data to save")


if __name__ == "__main__":
    main()
