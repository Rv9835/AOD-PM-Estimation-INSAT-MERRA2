from datetime import datetime

import pytest

from airpollution.sources.base import DataSourceError
from airpollution.sources.cpcb import CPCBConnector
from airpollution.sources.insat import INSATConnector
from airpollution.sources.merra2 import MERRA2Connector


def test_cpcb_fetch_and_validate() -> None:
    connector = CPCBConnector()
    start = datetime(2026, 1, 1)
    end = datetime(2026, 1, 3)

    data = connector.fetch(start, end)

    assert not data.empty
    assert all(col in data.columns for col in connector.REQUIRED_COLUMNS)
    assert len(data) > 0
    assert (data["lat"] >= -90).all() and (data["lat"] <= 90).all()
    assert (data["lon"] >= -180).all() and (data["lon"] <= 180).all()


def test_cpcb_fetch_with_station_ids() -> None:
    connector = CPCBConnector()
    start = datetime(2026, 1, 1)
    end = datetime(2026, 1, 2)
    stations = ["Delhi-ITO"]

    data = connector.fetch(start, end, station_ids=stations)

    assert data["station_id"].unique().tolist() == stations


def test_insat_fetch_and_validate() -> None:
    connector = INSATConnector()
    start = datetime(2026, 1, 1)
    end = datetime(2026, 1, 2)

    data = connector.fetch(start, end)

    assert not data.empty
    assert all(col in data.columns for col in connector.REQUIRED_COLUMNS)
    assert (data["aod"] >= 0).all() and (data["aod"] <= 5).all()
    assert (data["cloud_fraction"] >= 0).all() and (data["cloud_fraction"] <= 1).all()


def test_insat_regional_bounds() -> None:
    connector = INSATConnector()
    start = datetime(2026, 1, 1)
    end = datetime(2026, 1, 1)
    bounds = (10, 20, 70, 80)

    data = connector.fetch(start, end, region_bounds=bounds)

    assert (data["lat"] >= 10).all() and (data["lat"] <= 20).all()
    assert (data["lon"] >= 70).all() and (data["lon"] <= 80).all()


def test_merra2_fetch_and_validate() -> None:
    connector = MERRA2Connector()
    start = datetime(2026, 1, 1)
    end = datetime(2026, 1, 2)

    data = connector.fetch(start, end)

    assert not data.empty
    assert all(col in data.columns for col in connector.REQUIRED_COLUMNS)
    assert (data["temperature"] >= 200).all() and (data["temperature"] <= 330).all()
    assert (data["humidity"] >= 0).all() and (data["humidity"] <= 100).all()
    assert (data["wind_speed"] >= 0).all()
    assert (data["boundary_layer_height"] >= 0).all()


def test_schema_validation_fails_on_invalid_data() -> None:
    import pandas as pd

    connector = CPCBConnector()
    bad_data = pd.DataFrame({"aod": [0.1, 0.2]})

    with pytest.raises(DataSourceError):
        connector.validate_schema(bad_data)
