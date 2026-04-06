#!/usr/bin/env python3
"""
Fetch real MERRA-2 meteorological data from NASA GES DISC using OpenDAP.

This script downloads temperature, humidity, wind, and boundary layer height data.

Prerequisites:
    1. Create NASA Earthdata account: https://urs.earthdata.nasa.gov/
    2. Set environment variables:
       export EARTHDATA_USERNAME="your_username"
       export EARTHDATA_PASSWORD="your_password"

Usage:
    python scripts/fetch_merra2_opendap.py --start-date 2024-01-01 --end-date 2024-12-31
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

try:
    import xarray as xr
except ImportError:
    print("Installing xarray...")
    os.system("pip install xarray netcdf4")
    import xarray as xr


class MERRA2Fetcher:
    """Fetch MERRA-2 data from NASA GES DISC."""

    # OpenDAP endpoint
    OPENDAP_BASE = (
        "https://goldsmr5.gesdisc.eosdis.nasa.gov/opendap"
        "/MERRA2/M2T1NXSLV.5.12.4/{year:04d}{month:02d:d}/"
    )

    # OpenDAP dataset variables
    VARIABLES = {
        "T": "temperature",  # K
        "QV": "specific_humidity",  # kg/kg
        "U": "u_wind",  # m/s
        "V": "v_wind",  # m/s
        "PBLH": "boundary_layer_height",  # m
    }

    def __init__(self, username: str | None = None, password: str | None = None):
        """
        Initialize fetcher.

        Args:
            username: NASA Earthdata username (default: env var)
            password: NASA Earthdata password (default: env var)
        """
        self.username = username or os.getenv("EARTHDATA_USERNAME")
        self.password = password or os.getenv("EARTHDATA_PASSWORD")

        if not self.username or not self.password:
            raise ValueError(
                "Missing NASA Earthdata credentials. "
                "Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables."
            )

    def fetch_monthly(
        self,
        start_date: datetime,
        end_date: datetime,
        lat_min: float = 8.0,
        lat_max: float = 35.0,
        lon_min: float = 68.0,
        lon_max: float = 97.0,
    ) -> pd.DataFrame:
        """
        Fetch MERRA-2 data for date range.

        Args:
            start_date: Start date
            end_date: End date
            lat_min, lat_max, lon_min, lon_max: Region bounds

        Returns:
            DataFrame with meteorological variables
        """
        all_data = []

        # Generate dates
        current_date = datetime(start_date.year, start_date.month, 1)
        while current_date <= end_date:
            year = current_date.year
            month = current_date.month

            print(f"Fetching MERRA-2 for {year}-{month:02d}...")

            try:
                url = self.OPENDAP_BASE.format(year=year, month=month)
                url += (
                    f"MERRA2_400.tavg1_2d_slv_Nx.{year}{month:02d}01.SUB.nc"
                )

                # Download and subset
                ds = xr.open_dataset(
                    url,
                    engine="netcdf4",
                    decode_times=True,
                )

                # Subset region
                ds_subset = ds.sel(
                    lat=slice(lat_max, lat_min),  # Reversed for decreasing order
                    lon=slice(lon_min, lon_max),
                )

                # Extract variables
                df = ds_subset[list(self.VARIABLES.keys())].to_dataframe()
                df = df.reset_index()

                # Rename columns
                df = df.rename(columns=self.VARIABLES)

                # Calculate wind speed
                if "u_wind" in df.columns and "v_wind" in df.columns:
                    df["wind_speed"] = (
                        df["u_wind"] ** 2 + df["v_wind"] ** 2
                    ) ** 0.5
                    df = df.drop(columns=["u_wind", "v_wind"])

                # Calculate relative humidity from specific humidity
                if "specific_humidity" in df.columns and "temperature" in df.columns:
                    df["humidity"] = self._specific_to_relative_humidity(
                        df["specific_humidity"], df["temperature"]
                    )
                    df = df.drop(columns=["specific_humidity"])

                all_data.append(df)
                print(f"  ✓ Success: {len(df)} records")

            except Exception as e:
                print(f"  ✗ Error: {e}")

            # Move to next month
            if month == 12:
                current_date = datetime(year + 1, 1, 1)
            else:
                current_date = datetime(year, month + 1, 1)

        if not all_data:
            print("⚠️  No data fetched!")
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)
        print(f"\n✓ Total records: {len(result)}")
        return result

    @staticmethod
    def _specific_to_relative_humidity(q: pd.Series, t: pd.Series) -> pd.Series:
        """
        Convert specific humidity (kg/kg) to relative humidity (%).

        Args:
            q: Specific humidity (kg/kg)
            t: Temperature (K)

        Returns:
            Relative humidity (0-100%)
        """
        # Magnus formula constants
        a = 17.27
        b = 237.7

        # Convert T to Celsius
        tc = t - 273.15

        # Saturation vapor pressure (hPa)
        es = 6.112 * ((a * tc) / (b + tc)).apply(lambda x: 2.718281828 ** x)

        # Actual vapor pressure (hPa)
        # Using approximate conversion from specific humidity
        ea = q * 1013.25 / 0.622

        # Relative humidity
        rh = (ea / es) * 100

        # Clip to valid range
        return rh.clip(0, 100)

    def save_to_csv(self, df: pd.DataFrame, output_path: str) -> None:
        """Save data to CSV."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Select required columns
        columns = ["time", "lat", "lon", "temperature", "humidity", "wind_speed", "boundary_layer_height"]
        available = [col for col in columns if col in df.columns]

        df[available].to_csv(output_path, index=False)
        print(f"\n✓ Data saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch real MERRA-2 meteorological data from NASA GES DISC"
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
        "--output",
        type=str,
        default="data/raw/merra2_gridded.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--username",
        type=str,
        default=None,
        help="NASA Earthdata username (default: env var)",
    )
    parser.add_argument(
        "--password",
        type=str,
        default=None,
        help="NASA Earthdata password (default: env var)",
    )

    args = parser.parse_args()

    # Parse dates
    try:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError:
        print("❌ Invalid date format. Use YYYY-MM-DD")
        return

    # Fetch data
    print("🔄 Initializing MERRA-2 fetcher...")
    try:
        fetcher = MERRA2Fetcher(username=args.username, password=args.password)
    except ValueError as e:
        print(f"❌ {e}")
        print(
            "\nSet credentials via environment variables:\n"
            "  export EARTHDATA_USERNAME='your_username'\n"
            "  export EARTHDATA_PASSWORD='your_password'\n"
        )
        return

    print(f"📊 Fetching MERRA-2 from {start_date.date()} to {end_date.date()}...")
    try:
        df = fetcher.fetch_monthly(start_date, end_date)

        if not df.empty:
            fetcher.save_to_csv(df, args.output)

            # Summary statistics
            print("\n📈 Data Summary:")
            print(f"  Grid points: {df[['lat', 'lon']].drop_duplicates().shape[0]}")
            print(f"  Date range: {df['time'].min()} to {df['time'].max()}")
            if "temperature" in df.columns:
                print(
                    f"  Temperature: {df['temperature'].min():.1f} - "
                    f"{df['temperature'].max():.1f} K"
                )
            if "humidity" in df.columns:
                print(
                    f"  Humidity: {df['humidity'].min():.1f} - "
                    f"{df['humidity'].max():.1f} %"
                )
            if "wind_speed" in df.columns:
                print(
                    f"  Wind Speed: {df['wind_speed'].min():.1f} - "
                    f"{df['wind_speed'].max():.1f} m/s"
                )
        else:
            print("\n⚠️  No data to save")

    except Exception as e:
        print(f"❌ Error: {e}")
        return


if __name__ == "__main__":
    main()
