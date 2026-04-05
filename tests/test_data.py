import pandas as pd
import pytest

from airpollution.config import load_config
from airpollution.data import DataValidationError, preprocess_dataset


def test_preprocess_fails_for_missing_required_columns() -> None:
    config = load_config("configs/base.yaml")
    dataframe = pd.DataFrame({"aod": [0.1, 0.2], "pm25": [10, 20]})

    with pytest.raises(DataValidationError):
        _ = preprocess_dataset(dataframe, config)


def test_preprocess_imputes_missing_values() -> None:
    config = load_config("configs/base.yaml")
    dataframe = pd.read_csv("data/processed/unified_dataset.csv")
    dataframe.loc[0, "temperature"] = None

    processed = preprocess_dataset(dataframe, config)
    assert processed["temperature"].isna().sum() == 0
