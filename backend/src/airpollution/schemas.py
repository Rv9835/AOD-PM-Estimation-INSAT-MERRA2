from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathsConfig:
    input_csv: Path
    output_dir: Path
    model_path: Path
    metrics_path: Path


@dataclass(frozen=True)
class RandomForestConfig:
    n_estimators: int
    max_depth: int
    min_samples_split: int
    min_samples_leaf: int
    n_jobs: int


@dataclass(frozen=True)
class ModelConfig:
    target_column: str
    feature_columns: list[str]
    algorithm: str
    random_forest: RandomForestConfig


@dataclass(frozen=True)
class TrainingConfig:
    test_size: float
    shuffle: bool


@dataclass(frozen=True)
class QualityConfig:
    drop_missing_target: bool
    max_missing_feature_fraction: float


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    random_seed: int


@dataclass(frozen=True)
class AppConfig:
    project: ProjectConfig
    paths: PathsConfig
    model: ModelConfig
    training: TrainingConfig
    quality: QualityConfig
