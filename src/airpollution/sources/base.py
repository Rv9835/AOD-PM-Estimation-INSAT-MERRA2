from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class DataSourceError(Exception):
    """Raised when data source operations fail."""


class DataSource(ABC):
    """Abstract base class for air quality data sources."""

    @abstractmethod
    def validate_schema(self, dataframe: pd.DataFrame) -> None:
        """Validate that dataframe has required columns and types."""

    @abstractmethod
    def fetch(self, **kwargs: Any) -> pd.DataFrame:
        """
        Fetch data from the source, returning validated dataframe.
        
        Subclasses may define specific keyword arguments.
        """

    @staticmethod
    def _validate_required_columns(
        dataframe: pd.DataFrame,
        required: list[str],
    ) -> None:
        """Check all required columns are present."""
        missing = [col for col in required if col not in dataframe.columns]
        if missing:
            missing_str = ", ".join(missing)
            raise DataSourceError(f"Missing required columns: {missing_str}")

    @staticmethod
    def _validate_numeric_columns(
        dataframe: pd.DataFrame,
        numeric_columns: list[str],
    ) -> None:
        """Check numeric columns are actually numeric."""
        for col in numeric_columns:
            if col in dataframe.columns and not pd.api.types.is_numeric_dtype(dataframe[col]):
                raise DataSourceError(f"Column '{col}' must be numeric, got {dataframe[col].dtype}")
