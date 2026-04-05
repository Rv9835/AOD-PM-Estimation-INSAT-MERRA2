from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split

from airpollution.join import SpatiotemporalJoiner
from airpollution.schemas import AppConfig

logger = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Raised when input data fails validation constraints."""


def _validate_columns(dataframe: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise DataValidationError(f"Missing required columns: {missing_str}")


def load_dataset(config: AppConfig) -> pd.DataFrame:
    """
    Load dataset from CSV (cached offline data).
    Use load_dataset_from_sources() for live data ingestion.
    """
    dataframe = pd.read_csv(config.paths.input_csv)
    required = config.model.feature_columns + [config.model.target_column]
    _validate_columns(dataframe, required)
    logger.info("Loaded dataset with shape %s", dataframe.shape)
    return dataframe


def load_dataset_from_sources(
    start_date: datetime,
    end_date: datetime,
    station_ids: list[str] | None = None,
    insat_radius_km: float = 10.0,
    merra2_radius_km: float = 50.0,
) -> pd.DataFrame:
    """
    Load dataset by fetching and joining CPCB, INSAT, and MERRA-2 data.

    This directly queries live data sources and performs spatiotemporal fusion.
    """
    joiner = SpatiotemporalJoiner()
    dataframe = joiner.fetch_and_join(
        start_date,
        end_date,
        station_ids=station_ids,
        insat_radius_km=insat_radius_km,
        merra2_radius_km=merra2_radius_km,
    )
    logger.info("Loaded dataset from sources with shape %s", dataframe.shape)
    return dataframe


def preprocess_dataset(dataframe: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    processed = dataframe.copy()
    required = config.model.feature_columns + [config.model.target_column]
    _validate_columns(processed, required)

    if config.quality.drop_missing_target:
        before = len(processed)
        processed = processed.dropna(subset=[config.model.target_column])
        logger.info("Dropped %d rows with missing target", before - len(processed))

    # Engineer day_of_year from time column if present
    if "time" in processed.columns and "day_of_year" in config.model.feature_columns:
        if not pd.api.types.is_datetime64_any_dtype(processed["time"]):
            processed["time"] = pd.to_datetime(processed["time"])
        processed["day_of_year"] = processed["time"].dt.dayofyear

    feature_columns = config.model.feature_columns
    missing_fraction = processed[feature_columns].isna().mean()
    high_missing_columns = missing_fraction[
        missing_fraction > config.quality.max_missing_feature_fraction
    ].index.tolist()

    if high_missing_columns:
        columns_str = ", ".join(high_missing_columns)
        raise DataValidationError(
            "Feature columns exceed missing threshold "
            f"({config.quality.max_missing_feature_fraction}): {columns_str}"
        )

    processed[feature_columns] = processed[feature_columns].interpolate(
        limit_direction="both"
    )
    processed[feature_columns] = processed[feature_columns].fillna(
        processed[feature_columns].median()
    )

    if processed.empty:
        raise DataValidationError("Dataset became empty after preprocessing.")

    logger.info("Preprocessed dataset shape: %s", processed.shape)
    return processed


def split_features_target(
    dataframe: pd.DataFrame,
    config: AppConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    features = dataframe[config.model.feature_columns]
    target = dataframe[config.model.target_column]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=config.training.test_size,
        shuffle=config.training.shuffle,
        random_state=config.project.random_seed,
    )
    logger.info("Train shape: %s, Test shape: %s", x_train.shape, x_test.shape)
    return x_train, x_test, y_train, y_test
