from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from airpollution.sources.base import DataSource, DataSourceError


class CPCBConnector(DataSource):
    """
    Central Pollution Control Board (CPCB) ground station connector.

    Returns ground-based PM2.5 and PM10 measurements with location and time.
    """

    REQUIRED_COLUMNS = ["time", "lat", "lon", "pm25", "pm10", "station_id"]
    NUMERIC_COLUMNS = ["lat", "lon", "pm25", "pm10"]

    def validate_schema(self, dataframe: pd.DataFrame) -> None:
        self._validate_required_columns(dataframe, self.REQUIRED_COLUMNS)
        self._validate_numeric_columns(dataframe, self.NUMERIC_COLUMNS)

        if not pd.api.types.is_datetime64_any_dtype(dataframe["time"]):
            raise DataSourceError("Column 'time' must be datetime type")

        if ((dataframe["lat"] < -90) | (dataframe["lat"] > 90)).any():
            raise DataSourceError("Latitude values out of valid range [-90, 90]")

        if ((dataframe["lon"] < -180) | (dataframe["lon"] > 180)).any():
            raise DataSourceError("Longitude values out of valid range [-180, 180]")

    def fetch(  # type: ignore[override]
        self,
        start_date: datetime,
        end_date: datetime,
        station_ids: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Fetch CPCB ground station data.

        Args:
            start_date: Start of period.
            end_date: End of period.
            station_ids: Optional list of station IDs to fetch.

        Returns:
            DataFrame with columns: time, lat, lon, pm25, pm10, station_id
        """
        # Production: would call actual CPCB API here
        # For now, return mock data for demonstration
        mock_data = self._generate_mock_data(start_date, end_date, station_ids)
        self.validate_schema(mock_data)
        return mock_data

    @staticmethod
    def _generate_mock_data(
        start_date: datetime,
        end_date: datetime,
        station_ids: list[str] | None = None,
    ) -> pd.DataFrame:
        """Generate mock CPCB data for testing."""
        if station_ids is None:
            station_ids = ["Delhi-ITO", "Delhi-RK", "Mumbai-Fort", "Bangalore-BTM"]

        records = []
        current = start_date
        while current <= end_date:
            for station_id in station_ids:
                # Deterministic synthetic values
                import hashlib

                seed = int(
                    hashlib.md5(f"{station_id}-{current.isoformat()}".encode()).hexdigest(), 16
                ) % 1000
                lat = {
                    "Delhi-ITO": 28.61,
                    "Delhi-RK": 28.55,
                    "Mumbai-Fort": 18.93,
                    "Bangalore-BTM": 12.94,
                }.get(station_id, 20.0)
                lon = {
                    "Delhi-ITO": 77.25,
                    "Delhi-RK": 77.20,
                    "Mumbai-Fort": 72.84,
                    "Bangalore-BTM": 77.62,
                }.get(station_id, 75.0)

                records.append(
                    {
                        "time": current,
                        "lat": lat,
                        "lon": lon,
                        "pm25": 30 + seed % 100,
                        "pm10": 50 + (seed + 50) % 120,
                        "station_id": station_id,
                    }
                )

            current += timedelta(days=1)

        return pd.DataFrame(records)
