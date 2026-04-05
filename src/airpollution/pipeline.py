from __future__ import annotations

import json
import logging
from pathlib import Path

from airpollution.config import load_config
from airpollution.data import load_dataset, preprocess_dataset, split_features_target
from airpollution.evaluation import compute_regression_metrics
from airpollution.modeling import predict, save_model, train_random_forest

logger = logging.getLogger(__name__)


def run_training_pipeline(config_path: str | Path) -> dict[str, float]:
    config = load_config(config_path)

    data = load_dataset(config)
    data = preprocess_dataset(data, config)
    x_train, x_test, y_train, y_test = split_features_target(data, config)

    model = train_random_forest(config, x_train, y_train)
    predictions = predict(model, x_test)
    metrics = compute_regression_metrics(y_test.to_numpy(), predictions.to_numpy())

    save_model(model, config)
    config.paths.output_dir.mkdir(parents=True, exist_ok=True)
    with config.paths.metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics.to_dict(), file, indent=2)

    logger.info("Training completed. Metrics: %s", metrics.to_dict())
    return metrics.to_dict()
