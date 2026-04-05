from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from airpollution.sources.base import DataSource, DataSourceError


class MERRA2Connector(DataSource):
    """
    NASA MERRA-2 reanalysis connector.

    Returns meteorological variables on a global grid (0.5° x 0.625°).
    """

    REQUIRED_COLUMNS = [
        "time",
        "lat",
        "lon",
        "temperature",
        "humidity",
        "wind_speed",
        "boundary_layer_height",
    ]
    NUMERIC_COLUMNS = [
        "lat",
        "lon",
        "temperature",
        "humidity",
        "wind_speed",
        "boundary_layer_height",
    ]

    def validate_schema(self, dataframe: pd.DataFrame) -> None:
        self._validate_required_columns(dataframe, self.REQUIRED_COLUMNS)
        self._validate_numeric_columns(dataframe, self.NUMERIC_COLUMNS)

        if not pd.api.types.is_datetime64_any_dtype(dataframe["time"]):
            raise DataSourceError("Column 'time' must be datetime type")

        if ((dataframe["temperature"] < 200) | (dataframe["temperature"] > 330)).any():
            raise DataSourceError("Temperature must be in Kelvin [200, 330]")

        if ((dataframe["humidity"] < 0) | (dataframe["humidity"] > 100)).any():
            raise DataSourceError("Humidity must be in [0, 100] (%)")

        if (dataframe["wind_speed"] < 0).any():
            raise DataSourceError("Wind speed must be non-negative")

        if (
            (dataframe["boundary_layer_height"] < 0)
            | (dataframe["boundary_layer_height"] > 3000)
        ).any():
            raise DataSourceError(
                "Boundary layer height must be in [0, 3000] meters"
            )

    def fetch(  # type: ignore[override]
        self,
        start_date: datetime,
        end_date: datetime,
        region_bounds: tuple[float, float, float, float] | None = None,
    ) -> pd.DataFrame:
        """
        Fetch MERRA-2 meteorological data.

        Args:
            start_date: Start of period.
            end_date: End of period.
            region_bounds: (lat_min, lat_max, lon_min, lon_max) or None.

        Returns:
            DataFrame with meteorological columns.
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
        """Generate mock MERRA-2 reanalysis data."""
        if region_bounds is None:
            lat_min_f: float = 8
            lat_max_f: float = 35
            lon_min_f: float = 68
            lon_max_f: float = 97
        else:
            lat_min_f, lat_max_f, lon_min_f, lon_max_f = region_bounds

        # Regular grid with 0.25 degree spacing (realistic MERRA-2 resolution)
        step: float = 0.25
        lat_count = int((lat_max_f - lat_min_f) / step) + 1
        lon_count = int((lon_max_f - lon_min_f) / step) + 1
        lats: list[float] = [
            round(lat_min_f + i * step, 2) for i in range(lat_count)
        ]
        lons: list[float] = [
            round(lon_min_f + i * step, 2) for i in range(lon_count)
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
                    # Simulate seasonal variation
                    day_of_year = current.timetuple().tm_yday
                    temp_offset = 10 * ((day_of_year - 1) % 365 - 182) / 182  # ±10K seasonal swing

                    records.append(
                        {
                            "time": current,
                            "lat": lat,
                            "lon": lon,
                            "temperature": 288 + temp_offset + (seed % 20 - 10),  # Kelvin
                            "humidity": 50 + (seed % 40) - 20,  # Percent
                            "wind_speed": 3 + (seed % 10) / 2,  # m/s
                            "boundary_layer_height": 500 + (seed % 700),  # meters
                        }
                    )

            current += timedelta(days=1)

        return pd.DataFrame(records)
