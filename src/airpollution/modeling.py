from __future__ import annotations

from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from airpollution.schemas import AppConfig


@dataclass(frozen=True)
class TrainedModel:
    estimator: RandomForestRegressor


def train_random_forest(
    config: AppConfig,
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> TrainedModel:
    rf_config = config.model.random_forest
    estimator = RandomForestRegressor(
        n_estimators=rf_config.n_estimators,
        max_depth=rf_config.max_depth,
        min_samples_split=rf_config.min_samples_split,
        min_samples_leaf=rf_config.min_samples_leaf,
        n_jobs=rf_config.n_jobs,
        random_state=config.project.random_seed,
    )
    estimator.fit(x_train, y_train)
    return TrainedModel(estimator=estimator)


def predict(model: TrainedModel, x_test: pd.DataFrame) -> pd.Series:
    predictions = model.estimator.predict(x_test)
    return pd.Series(predictions, index=x_test.index, name="prediction")


def save_model(model: TrainedModel, config: AppConfig) -> None:
    config.paths.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model.estimator, config.paths.model_path)
