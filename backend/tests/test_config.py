from pathlib import Path

from airpollution.config import load_config


def test_load_config_success() -> None:
    config = load_config(Path("configs/base.yaml"))
    assert config.project.random_seed == 42
    assert config.model.target_column == "pm25"
    assert len(config.model.feature_columns) > 0
