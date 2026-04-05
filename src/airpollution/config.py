from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from airpollution.schemas import (
    AppConfig,
    ModelConfig,
    PathsConfig,
    ProjectConfig,
    QualityConfig,
    RandomForestConfig,
    TrainingConfig,
)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)
    if not isinstance(loaded, dict):
        raise ValueError("Configuration file must contain a top-level mapping.")
    return loaded


def load_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path)
    raw = _read_yaml(path)

    project = raw["project"]
    paths = raw["paths"]
    model = raw["model"]
    training = raw["training"]
    quality = raw["quality"]

    rf_cfg = RandomForestConfig(**model["random_forest"])
    app_config = AppConfig(
        project=ProjectConfig(**project),
        paths=PathsConfig(
            input_csv=Path(paths["input_csv"]),
            output_dir=Path(paths["output_dir"]),
            model_path=Path(paths["model_path"]),
            metrics_path=Path(paths["metrics_path"]),
        ),
        model=ModelConfig(
            target_column=model["target_column"],
            feature_columns=list(model["feature_columns"]),
            algorithm=model["algorithm"],
            random_forest=rf_cfg,
        ),
        training=TrainingConfig(**training),
        quality=QualityConfig(**quality),
    )

    if app_config.model.algorithm != "random_forest":
        raise ValueError("Only 'random_forest' is currently supported.")

    if not (0.0 < app_config.training.test_size < 1.0):
        raise ValueError("training.test_size must be in (0, 1).")

    if not (0.0 <= app_config.quality.max_missing_feature_fraction < 1.0):
        raise ValueError("quality.max_missing_feature_fraction must be in [0, 1).")

    return app_config
