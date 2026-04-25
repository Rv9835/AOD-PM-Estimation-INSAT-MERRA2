import numpy as np

from airpollution.evaluation import compute_regression_metrics


def test_compute_regression_metrics_values() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 2.1, 2.9, 3.9])

    metrics = compute_regression_metrics(y_true, y_pred)

    assert metrics.rmse > 0
    assert metrics.mae > 0
    assert -1.0 <= metrics.r2 <= 1.0
