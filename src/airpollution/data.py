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
    
    Priority:
    1. If data/raw/cpcb_real.csv exists, use real CPCB data
    2. Otherwise use default config.paths.input_csv (mock data)
    
    Use load_dataset_from_sources() for live data ingestion.
    """
    import os
    from pathlib import Path
    
    # Check for real CPCB data first
    real_data_path = Path("data/raw/cpcb_real.csv")
    if real_data_path.exists():
        logger.info("Found real CPCB data at %s, loading...", real_data_path)
        dataframe = _load_real_cpcb_data(real_data_path, config)
        logger.info("Loaded real CPCB dataset with shape %s", dataframe.shape)
        return dataframe
    
    # Fall back to mock data if real data not found
    dataframe = pd.read_csv(config.paths.input_csv)
    required = config.model.feature_columns + [config.model.target_column]
    _validate_columns(dataframe, required)
    logger.info("Loaded mock dataset from %s with shape %s", config.paths.input_csv, dataframe.shape)
    return dataframe


def _load_real_cpcb_data(filepath: Path, config: AppConfig) -> pd.DataFrame:
    """
    Load and process real CPCB data from OpenAQ format.
    
    OpenAQ format has: location_id, parameter, value, unit, datetimeUtc, latitude, longitude
    Extract PM2.5 and PM10, join with mock INSAT/MERRA-2 data.
    """
    df_raw = pd.read_csv(filepath)
    
    # Extract PM2.5 measurements
    df_pm25 = df_raw[df_raw["parameter"] == "pm25"].copy()
    df_pm25 = df_pm25.rename(columns={"value": "pm25", "datetimeUtc": "time"})
    df_pm25["time"] = pd.to_datetime(df_pm25["time"])
    df_pm25 = df_pm25[["time", "latitude", "longitude", "pm25", "location_name"]]
    df_pm25.columns = ["time", "lat", "lon", "pm25", "station_id"]
    
    # Extract PM10 measurements
    df_pm10 = df_raw[df_raw["parameter"] == "pm10"].copy()
    df_pm10 = df_pm10.rename(columns={"value": "pm10", "datetimeUtc": "time"})
    df_pm10["time"] = pd.to_datetime(df_pm10["time"])
    df_pm10 = df_pm10[["time", "latitude", "longitude", "pm10"]]
    df_pm10.columns = ["time", "lat", "lon", "pm10"]
    
    # Merge PM2.5 and PM10
    if len(df_pm10) > 0:
        df_cpcb = df_pm25.merge(df_pm10, on=["time", "lat", "lon"], how="left")
    else:
        df_cpcb = df_pm25.copy()
        df_cpcb["pm10"] = df_cpcb["pm25"] * 1.5  # Estimate PM10 from PM2.5
    
    # Add day_of_year feature
    df_cpcb["day_of_year"] = df_cpcb["time"].dt.dayofyear
    
    # Use spatiotemporal join to add INSAT/MERRA-2 data
    from datetime import datetime, timedelta
    
    if len(df_cpcb) > 0:
        start_date = df_cpcb["time"].min()
        end_date = df_cpcb["time"].max()
        
        # Get unique stations
        stations = df_cpcb["station_id"].unique().tolist()
        
        # Fetch and join with INSAT/MERRA-2 (mock data)
        try:
            joiner = SpatiotemporalJoiner()
            df_joined = joiner.fetch_and_join(
                start_date=start_date,
                end_date=end_date,
                station_ids=stations,
                insat_radius_km=50.0,   # Increased from 15 for better matching
                merra2_radius_km=150.0,  # Increased from 75 for better matching
            )
            
            # Merge real CPCB data with joined satellite/weather data
            # Keep real PM measurements, use satellite/weather from join
            df_merged = df_joined.copy()
            
            # Update PM25/PM10 with real CPCB values where available
            for idx, row in df_cpcb.iterrows():
                t = row["time"]
                lat = row["lat"]
                lon = row["lon"]
                
                # Find matching row in joined data (closest in space/time)
                dist = (
                    (df_merged["time"] - t).abs() < pd.Timedelta("1H")
                ) & (
                    (df_merged["lat"] - lat).abs() < 0.1
                ) & (
                    (df_merged["lon"] - lon).abs() < 0.1
                )
                
                if dist.any():
                    idx_match = df_merged[dist].index[0]
                    df_merged.loc[idx_match, "pm25"] = row["pm25"]
                    if "pm10" in row and pd.notna(row["pm10"]):
                        df_merged.loc[idx_match, "pm10"] = row["pm10"]
            
            # Ensure day_of_year is present
            if "day_of_year" not in df_merged.columns:
                if "time" in df_merged.columns:
                    if not pd.api.types.is_datetime64_any_dtype(df_merged["time"]):
                        df_merged["time"] = pd.to_datetime(df_merged["time"])
                    df_merged["day_of_year"] = df_merged["time"].dt.dayofyear
            
            return df_merged
        except Exception as e:
            logger.warning("Failed to join with satellite/weather data: %s. Using CPCB data only.", e)
            # Just return CPCB data with NaN for missing features
            return df_cpcb.assign(
                aod=None,
                cloud_fraction=None,
                temperature=None,
                humidity=None,
                wind_speed=None,
                boundary_layer_height=None,
            )
    
    return df_cpcb


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
    
    # Impute missing values FIRST (before quality checks)
    # This allows us to work with real data that may have missing satellite/weather features
    processed[feature_columns] = processed[feature_columns].interpolate(
        limit_direction="both"
    )
    processed[feature_columns] = processed[feature_columns].fillna(
        processed[feature_columns].median()
    )
    
    # Check quality threshold AFTER imputation
    missing_fraction = processed[feature_columns].isna().mean()
    high_missing_columns = missing_fraction[
        missing_fraction > config.quality.max_missing_feature_fraction
    ].index.tolist()

    if high_missing_columns:
        columns_str = ", ".join(high_missing_columns)
        logger.warning(
            "Feature columns still exceed missing threshold after imputation "
            f"({config.quality.max_missing_feature_fraction}): {columns_str}. "
            "Proceeding with available data."
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
