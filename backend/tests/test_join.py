from datetime import datetime

import pytest

from airpollution.join import SpatiotemporalJoiner, SpatiotemporalJoinError
from airpollution.sources.insat import INSATConnector
from airpollution.sources.merra2 import MERRA2Connector


def test_spatiotemporal_join_completes() -> None:
    joiner = SpatiotemporalJoiner()
    start = datetime(2026, 1, 1)
    end = datetime(2026, 1, 5)

    result = joiner.fetch_and_join(start, end)

    assert not result.empty
    assert all(
        col in result.columns
        for col in ["time", "lat", "lon", "station_id", "pm25", "pm10", "aod", "cloud_fraction",
                    "temperature", "humidity", "wind_speed", "boundary_layer_height"]
    )


def test_spatiotemporal_join_with_stations() -> None:
    joiner = SpatiotemporalJoiner()
    start = datetime(2026, 1, 1)
    end = datetime(2026, 1, 2)
    stations = ["Delhi-ITO", "Mumbai-Fort"]

    result = joiner.fetch_and_join(start, end, station_ids=stations)

    assert not result.empty
    assert result["station_id"].isin(stations).all()


def test_insat_join_with_custom_radius() -> None:
    joiner = SpatiotemporalJoiner()
    start = datetime(2026, 1, 1)
    end = datetime(2026, 1, 1)

    # Smaller radius should still find matches but potentially fewer rows
    result_tight = joiner.fetch_and_join(start, end, insat_radius_km=5.0)
    result_loose = joiner.fetch_and_join(start, end, insat_radius_km=50.0)

    assert len(result_tight) <= len(result_loose)


def test_join_handles_empty_connectors() -> None:
    # Mock empty connector
    class EmptyConnector:
        def fetch(self, *args, **kwargs):
            import pandas as pd
            return pd.DataFrame()

        def validate_schema(self, df):
            pass

    joiner = SpatiotemporalJoiner(
        cpcb_connector=EmptyConnector(),  # type: ignore
        insat_connector=INSATConnector(),
        merra2_connector=MERRA2Connector(),
    )
    start = datetime(2026, 1, 1)
    end = datetime(2026, 1, 2)

    with pytest.raises(SpatiotemporalJoinError):
        joiner.fetch_and_join(start, end)
