import json
from pathlib import Path

from airpollution.pipeline import run_training_pipeline


def test_pipeline_runs_and_writes_artifacts(tmp_path: Path) -> None:
    config_text = Path("configs/base.yaml").read_text(encoding="utf-8")
    config_text = config_text.replace("artifacts/baseline", str(tmp_path / "baseline"))

    cfg_path = tmp_path / "test_config.yaml"
    cfg_path.write_text(config_text, encoding="utf-8")

    metrics = run_training_pipeline(cfg_path)

    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics

    metrics_path = tmp_path / "baseline" / "metrics.json"
    model_path = tmp_path / "baseline" / "random_forest_pm25.joblib"

    assert metrics_path.exists()
    assert model_path.exists()

    loaded_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert set(loaded_metrics.keys()) == {"rmse", "mae", "r2"}
