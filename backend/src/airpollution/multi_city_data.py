"""
Multi-City Data Loading and Splitting Module
Handles city-boundary-aware data filtering, scenario-based train/test splits, and NaN handling.
"""

import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import pandas as pd
import numpy as np
import yaml

from airpollution.cities import CityManager, CityConfig

logger = logging.getLogger(__name__)


# =====================================================
# TYPE ALIASES
# =====================================================
DataFrame = pd.DataFrame
Series = pd.Series
TrainTestSplit = Tuple[DataFrame, Series, DataFrame, Series]  # X_train, y_train, X_test, y_test


# =====================================================
# CONFIG LOADER
# =====================================================
class ConfigLoader:
    """Load and cache YAML configuration."""

    _config: Optional[Dict] = None
    _config_path: Optional[str] = None

    @classmethod
    def load_config(cls, config_path: str) -> Dict:
        """
        Load YAML config from file. Caches result for subsequent calls.

        Args:
            config_path: Path to base.yaml

        Returns:
            Parsed configuration dictionary

        Raises:
            FileNotFoundError: If config file not found
            yaml.YAMLError: If config is malformed
        """
        if cls._config is not None and cls._config_path == config_path:
            logger.debug(f"Using cached config from {config_path}")
            return cls._config

        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_file, "r") as f:
                cls._config = yaml.safe_load(f)
                cls._config_path = config_path
            logger.info(f"✓ Loaded configuration from {config_path}")
            return cls._config
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse YAML config: {e}")

    @classmethod
    def get_config(cls) -> Dict:
        """Get cached config. Raises error if not loaded."""
        if cls._config is None:
            raise RuntimeError("Config not loaded. Call load_config() first.")
        return cls._config


# =====================================================
# MULTI-CITY DATA LOADER
# =====================================================
class MultiCityDataLoader:
    """
    Loads unified dataset and provides city-aware filtering, splitting, and preprocessing.
    """

    def __init__(
        self,
        config_path: str,
        unified_csv_path: Optional[str] = None,
    ) -> None:
        """
        Initialize MultiCityDataLoader.

        Args:
            config_path: Path to base.yaml config
            unified_csv_path: Override path to unified CSV (if None, uses config path)

        Raises:
            FileNotFoundError: If files not found
        """
        self.config_path = Path(config_path).resolve()
        self.project_root = self.config_path.parents[2]
        self.config = ConfigLoader.load_config(str(self.config_path))
        self.city_manager = CityManager()

        # Determine CSV path
        if unified_csv_path:
            self.csv_path = Path(unified_csv_path)
        else:
            self.csv_path = Path(self.config["data"]["unified_dataset_path"])

        if not self.csv_path.is_absolute():
            self.csv_path = (self.project_root / self.csv_path).resolve()

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Unified dataset not found: {self.csv_path}")

        # Initialize data cache
        self._raw_data: Optional[DataFrame] = None
        self._processed_data: Optional[DataFrame] = None

        # Get required columns from config
        self.required_columns = self.config["data"]["validation"]["required_columns"]

        logger.info(f"✓ Initialized MultiCityDataLoader with CSV: {self.csv_path}")

    def load_raw_data(self, force_reload: bool = False) -> DataFrame:
        """
        Load raw unified CSV into memory with lightweight validation.

        Args:
            force_reload: If True, bypass cache and reload from disk

        Returns:
            Raw DataFrame

        Raises:
            ValueError: If required columns missing or data is empty
        """
        if self._raw_data is not None and not force_reload:
            logger.debug("Using cached raw data")
            return self._raw_data

        logger.info(f"Loading raw data from {self.csv_path}...")
        try:
            df = pd.read_csv(self.csv_path)
        except Exception as e:
            raise ValueError(f"Failed to read CSV: {e}")

        if df.empty:
            raise ValueError("Loaded CSV is empty")

        # Validate required columns
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            available = ", ".join(df.columns[:5])
            raise ValueError(
                f"Missing required columns: {missing_cols}. Available: {available}..."
            )

        logger.info(f"✓ Loaded {len(df)} rows with {len(df.columns)} columns")
        self._raw_data = df
        return df

    def load_city_data(self, city: str) -> DataFrame:
        """
        Load and filter data for a specific city using bounding box.

        Args:
            city: City key (e.g., "delhi", "mumbai")

        Returns:
            Filtered DataFrame containing only data points within city bounds

        Raises:
            ValueError: If city not found in registry
        """
        raw_df = self.load_raw_data()
        city_config = self.city_manager.get_city_strict(city)

        # Filter by bounding box
        filtered_df = raw_df[
            (raw_df["lat"] >= city_config.bounding_box[0])
            & (raw_df["lat"] <= city_config.bounding_box[1])
            & (raw_df["lon"] >= city_config.bounding_box[2])
            & (raw_df["lon"] <= city_config.bounding_box[3])
        ].copy()

        logger.info(
            f"✓ Filtered {len(filtered_df)} rows for {city} "
            f"({len(filtered_df) / len(raw_df) * 100:.1f}% of total)"
        )

        if filtered_df.empty:
            logger.warning(f"⚠ No data found for city '{city}' within bounding box")
            return filtered_df

        return filtered_df

    def preprocess_city_data(self, df: DataFrame) -> DataFrame:
        """
        Apply preprocessing pipeline: NaN handling, outlier removal, feature selection.

        Args:
            df: Input DataFrame

        Returns:
            Preprocessed DataFrame

        Raises:
            ValueError: If dataframe becomes empty after preprocessing
        """
        if df.empty:
            return df

        original_len = len(df)
        nan_config = self.config["data"]["validation"]

        # Step 1: NaN handling (forward fill then backward fill)
        nan_strategy = nan_config.get("nan_strategy", "forward_fill_backward_fill")
        if nan_strategy == "forward_fill_backward_fill":
            df = df.ffill().bfill()
        elif nan_strategy == "drop":
            df = df.dropna()

        nan_remaining = df.isnull().sum().sum()
        logger.debug(f"  NaN handling: {original_len} rows → {len(df)} rows (NaN cells: {nan_remaining})")

        # Step 2: PM2.5 outlier removal
        pm25_col = "pm25"
        if pm25_col in df.columns:
            min_pm25 = nan_config.get("pm25_min_threshold", 0)
            max_pm25 = nan_config.get("pm25_max_threshold", 500)

            before_outlier = len(df)
            df = df[(df[pm25_col] >= min_pm25) & (df[pm25_col] <= max_pm25)].copy()
            removed = before_outlier - len(df)
            if removed > 0:
                logger.debug(f"  Removed {removed} PM2.5 outliers")

        # Step 3: Drop any remaining rows with NaN in critical columns
        critical_cols = [
            col
            for col in self.required_columns
            if col in df.columns
        ]
        df = df.dropna(subset=critical_cols, how="any")

        if df.empty:
            raise ValueError("DataFrame became empty after preprocessing")

        logger.info(
            f"✓ Preprocessed: {original_len} → {len(df)} rows "
            f"({len(df) / original_len * 100:.1f}% retained)"
        )
        return df

    def load_and_preprocess_city(self, city: str) -> DataFrame:
        """
        Convenience method: load city data and immediately preprocess.

        Args:
            city: City key

        Returns:
            Preprocessed DataFrame
        """
        df = self.load_city_data(city)
        return self.preprocess_city_data(df)

    def split_by_scenario(
        self,
        scenario_name: str,
    ) -> Tuple[DataFrame, Series, DataFrame, Series]:
        """
        Split data according to a predefined scenario from config.

        Args:
            scenario_name: Name of scenario (e.g., "delhi_to_mumbai", "multi_city_to_mumbai")

        Returns:
            (X_train, y_train, X_test, y_test)

        Raises:
            ValueError: If scenario not found or data loading fails
        """
        # Find scenario in config
        scenarios = self.config.get("training_scenarios", [])
        scenario = None
        for s in scenarios:
            if s["name"] == scenario_name:
                scenario = s
                break

        if scenario is None:
            available = ", ".join([s["name"] for s in scenarios])
            raise ValueError(
                f"Scenario '{scenario_name}' not found. Available: {available}"
            )

        logger.info(f"🎯 Splitting data for scenario: {scenario_name}")
        logger.info(f"   Description: {scenario.get('description', 'N/A')}")

        training_cities = scenario.get("training_cities", [])
        test_cities = scenario.get("test_cities", [])

        # Load and preprocess training cities
        logger.info(f"   Loading training cities: {training_cities}")
        train_dfs = []
        for city in training_cities:
            try:
                city_df = self.load_and_preprocess_city(city)
                if not city_df.empty:
                    train_dfs.append(city_df)
                else:
                    logger.warning(f"   ⚠ No data for training city '{city}'")
            except Exception as e:
                logger.error(f"   ✗ Error loading training city '{city}': {e}")
                raise

        if not train_dfs:
            raise ValueError(f"No training data available for scenario '{scenario_name}'")

        # Concatenate all training data
        train_df = pd.concat(train_dfs, ignore_index=True)
        logger.info(f"   ✓ Combined training data: {len(train_df)} rows")

        # Load and preprocess test cities
        logger.info(f"   Loading test cities: {test_cities}")
        test_dfs = []
        for city in test_cities:
            try:
                city_df = self.load_and_preprocess_city(city)
                if not city_df.empty:
                    test_dfs.append(city_df)
                else:
                    logger.warning(f"   ⚠ No data for test city '{city}'")
            except Exception as e:
                logger.error(f"   ✗ Error loading test city '{city}': {e}")
                raise

        if not test_dfs:
            raise ValueError(f"No test data available for scenario '{scenario_name}'")

        test_df = pd.concat(test_dfs, ignore_index=True)
        logger.info(f"   ✓ Combined test data: {len(test_df)} rows")

        # Determine if we need to further split train/test by temporal or random
        boundary_type = scenario.get("train_test_boundary", "temporal")
        test_size = scenario.get("test_size", 0.2)

        # If test_size == 1.0, use all test cities as test (city-boundary split)
        if test_size >= 1.0:
            logger.info(f"   Using city-boundary split (all test cities are test set)")
            X_train = train_df.drop(columns=["pm25"])
            y_train = train_df["pm25"]
            X_test = test_df.drop(columns=["pm25"])
            y_test = test_df["pm25"]
        else:
            # For same-city scenarios, need to temporal/random split all combined data
            if boundary_type == "temporal":
                logger.info(f"   Using temporal split (80/20)")
                # Assuming timestamp column exists; if not, use random split
                if "timestamp" in train_df.columns:
                    train_df_sorted = train_df.sort_values("timestamp").reset_index(drop=True)
                    split_idx = int(len(train_df_sorted) * (1 - test_size))
                    train_part = train_df_sorted.iloc[:split_idx]
                    test_part = train_df_sorted.iloc[split_idx:]
                elif "day_of_year" in train_df.columns:
                    logger.warning("   ⚠ No timestamp column; using day_of_year for temporal sort")
                    train_df_sorted = train_df.sort_values("day_of_year").reset_index(drop=True)
                    split_idx = int(len(train_df_sorted) * (1 - test_size))
                    train_part = train_df_sorted.iloc[:split_idx]
                    test_part = train_df_sorted.iloc[split_idx:]
                else:
                    logger.warning("   ⚠ No timestamp or day_of_year column; using random split instead")
                    train_part = train_df.sample(
                        frac=1 - test_size,
                        random_state=scenario.get("random_state", 42),
                    )
                    test_part = train_df.drop(train_part.index)
            else:
                # Random split
                train_part = train_df.sample(
                    frac=1 - test_size,
                    random_state=scenario.get("random_state", 42),
                )
                test_part = train_df.drop(train_part.index)

            X_train = train_part.drop(columns=["pm25"])
            y_train = train_part["pm25"]
            X_test = test_part.drop(columns=["pm25"])
            y_test = test_part["pm25"]

        logger.info(
            f"✓ Scenario '{scenario_name}' split complete:\n"
            f"    Train: {len(X_train)} samples\n"
            f"    Test:  {len(X_test)} samples"
        )

        return X_train, y_train, X_test, y_test

    def get_scenario_names(self) -> List[str]:
        """Get list of all available scenario names from config."""
        scenarios = self.config.get("training_scenarios", [])
        return [s["name"] for s in scenarios]

    def get_scenario_description(self, scenario_name: str) -> str:
        """Get description of a specific scenario."""
        scenarios = self.config.get("training_scenarios", [])
        for s in scenarios:
            if s["name"] == scenario_name:
                return s.get("description", "No description available")
        return "Scenario not found"
