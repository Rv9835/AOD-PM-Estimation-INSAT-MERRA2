from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from airpollution.sources.base import DataSource, DataSourceError


class INSATConnector(DataSource):
    """
    INSAT-3D/3DR/3DS satellite connector.

    Returns Aerosol Optical Depth (AOD) measurements on a regular grid.
    """

    REQUIRED_COLUMNS = ["time", "lat", "lon", "aod", "cloud_fraction"]
    NUMERIC_COLUMNS = ["lat", "lon", "aod", "cloud_fraction"]

    def validate_schema(self, dataframe: pd.DataFrame) -> None:
        self._validate_required_columns(dataframe, self.REQUIRED_COLUMNS)
        self._validate_numeric_columns(dataframe, self.NUMERIC_COLUMNS)

        if not pd.api.types.is_datetime64_any_dtype(dataframe["time"]):
            raise DataSourceError("Column 'time' must be datetime type")

        if ((dataframe["lat"] < -90) | (dataframe["lat"] > 90)).any():
            raise DataSourceError("Latitude out of range [-90, 90]")

        if ((dataframe["lon"] < -180) | (dataframe["lon"] > 180)).any():
            raise DataSourceError("Longitude out of range [-180, 180]")

        if ((dataframe["aod"] < 0) | (dataframe["aod"] > 5)).any():
            raise DataSourceError("AOD must be in [0, 5] range")

        if ((dataframe["cloud_fraction"] < 0) | (dataframe["cloud_fraction"] > 1)).any():
            raise DataSourceError("Cloud fraction must be in [0, 1] range")

    def fetch(  # type: ignore[override]
        self,
        start_date: datetime,
        end_date: datetime,
        region_bounds: tuple[float, float, float, float] | None = None,
    ) -> pd.DataFrame:
        """
        Fetch INSAT satellite AOD data.

        Args:
            start_date: Start of period.
            end_date: End of period.
            region_bounds: (lat_min, lat_max, lon_min, lon_max) or None for all India.

        Returns:
            DataFrame with columns: time, lat, lon, aod, cloud_fraction
        """
        mock_data = self._generate_mock_data(start_date, end_date, region_bounds)
        self.validate_schema(mock_data)
        return mock_data

    @staticmethod
    def _generate_mock_data(
        start_date: datetime,
        end_date: datetime,
        region_bounds: tuple[float, float, float, float] | None = None,
    ) -> pd.DataFrame:
        """Generate mock INSAT satellite grid data."""
        if region_bounds is None:
            lat_min_f: float = 8
            lat_max_f: float = 35
            lon_min_f: float = 68
            lon_max_f: float = 97
        else:
            lat_min_f, lat_max_f, lon_min_f, lon_max_f = region_bounds

        # Regular grid with 0.1 degree spacing for realistic collocation
        step: float = 0.1
        lat_count = int((lat_max_f - lat_min_f) / step) + 1
        lon_count = int((lon_max_f - lon_min_f) / step) + 1
        lats: list[float] = [
            round(lat_min_f + i * step, 1) for i in range(lat_count)
        ]
        lons: list[float] = [
            round(lon_min_f + i * step, 1) for i in range(lon_count)
        ]

        records = []
        current = start_date
        while current <= end_date:
            for lat in lats:
                for lon in lons:
                    import hashlib

                    hash_input = f"{lat}-{lon}-{current.isoformat()}"
                    seed = int(
                        hashlib.md5(hash_input.encode()).hexdigest(), 16
                    ) % 1000
                    records.append(
                        {
                            "time": current,
                            "lat": lat,
                            "lon": lon,
                            "aod": 0.2 + (seed % 400) / 1000.0,
                            "cloud_fraction": (seed % 100) / 100.0,
                        }
                    )

            current += timedelta(days=1)

        return pd.DataFrame(records)
