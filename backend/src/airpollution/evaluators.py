"""
Model Evaluation and Robustness Metrics Module
Provides CrossCityEvaluator with RMSE, MAE, R², and custom robustness_score.
"""

import logging
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


# =====================================================
# TYPE ALIASES
# =====================================================
NDArray = np.ndarray


# =====================================================
# ROBUSTNESS SCORE CALCULATION
# =====================================================
class RobustnessScoreCalculator:
    """
    Computes custom robustness score (0-100 scale) based on:
    - Overfitting ratio between train and test performance
    - Transfer learning capability (test RMSE degradation)
    """

    def __init__(
        self,
        overfitting_weight: float = 0.4,
        transfer_weight: float = 0.6,
        acceptable_degradation: float = 1.5,
    ) -> None:
        """
        Initialize robustness calculator.

        Args:
            overfitting_weight: Weight for overfitting penalty (0.0-1.0)
            transfer_weight: Weight for transfer performance (0.0-1.0)
            acceptable_degradation: Max ratio of test_rmse/train_rmse before penalty
        """
        if overfitting_weight + transfer_weight != 1.0:
            raise ValueError("Weights must sum to 1.0")

        self.overfitting_weight = overfitting_weight
        self.transfer_weight = transfer_weight
        self.acceptable_degradation = acceptable_degradation

    def compute(
        self,
        train_rmse: float,
        test_rmse: float,
        train_r2: float,
        test_r2: float,
    ) -> float:
        """
        Compute robustness score.

        Args:
            train_rmse: RMSE on training data
            test_rmse: RMSE on test data
            train_r2: R² on training data
            test_r2: R² on test data

        Returns:
            Robustness score (0-100)
        """
        if train_rmse <= 0 or test_rmse < 0:
            raise ValueError("RMSE values must be positive")

        # Component 1: Overfitting Penalty (0-1)
        overfitting_ratio = test_rmse / train_rmse if train_rmse > 0 else 1.0

        if overfitting_ratio <= 1.0:
            # No overfitting, full score
            overfitting_score = 1.0
        elif overfitting_ratio <= self.acceptable_degradation:
            # Acceptable degradation, linear penalty
            overfitting_score = 1.0 - (
                (overfitting_ratio - 1.0) / (self.acceptable_degradation - 1.0)
            ) * 0.5
        else:
            # Severe overfitting
            overfitting_score = max(0.0, 1.0 - (overfitting_ratio - 1.0) / 2.0)

        # Component 2: Transfer Performance (0-1)
        # Based on test R² (how well it generalizes)
        transfer_score = max(0.0, min(1.0, test_r2))

        # Weighted combination
        robustness_score = (
            self.overfitting_weight * overfitting_score
            + self.transfer_weight * transfer_score
        )

        # Scale to 0-100
        robustness_score_scaled = robustness_score * 100.0

        return robustness_score_scaled


# =====================================================
# CROSS-CITY EVALUATOR
# =====================================================
class CrossCityEvaluator:
    """
    Evaluates model performance on train/test splits with cross-city robustness metrics.
    Computes RMSE, MAE, R², and custom robustness_score.
    """

    def __init__(
        self,
        overfitting_weight: float = 0.4,
        transfer_weight: float = 0.6,
        acceptable_degradation: float = 1.5,
    ) -> None:
        """
        Initialize evaluator.

        Args:
            overfitting_weight: Weight for overfitting in robustness score
            transfer_weight: Weight for transfer in robustness score
            acceptable_degradation: Acceptable RMSE degradation ratio
        """
        self.robustness_calc = RobustnessScoreCalculator(
            overfitting_weight=overfitting_weight,
            transfer_weight=transfer_weight,
            acceptable_degradation=acceptable_degradation,
        )
        self._evaluation_history: List[Dict[str, Any]] = []

    def evaluate(
        self,
        y_train: NDArray,
        y_train_pred: NDArray,
        y_test: NDArray,
        y_test_pred: NDArray,
        model_name: Optional[str] = None,
        scenario_name: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model predictions on train and test sets.

        Args:
            y_train: Ground truth training labels
            y_train_pred: Predictions on training data
            y_test: Ground truth test labels
            y_test_pred: Predictions on test data
            model_name: Optional model name for logging
            scenario_name: Optional scenario name for reporting

        Returns:
            Dictionary with metrics:
            - train_rmse, test_rmse
            - train_mae, test_mae
            - train_r2, test_r2
            - rmse_ratio (overfitting indicator)
            - robustness_score (0-100)

        Raises:
            ValueError: If shapes don't match or arrays empty
        """
        # Validation
        if len(y_train) == 0 or len(y_test) == 0:
            raise ValueError("Cannot evaluate with empty arrays")

        if len(y_train) != len(y_train_pred):
            raise ValueError(
                f"Train shape mismatch: y_train ({len(y_train)}) vs pred ({len(y_train_pred)})"
            )

        if len(y_test) != len(y_test_pred):
            raise ValueError(
                f"Test shape mismatch: y_test ({len(y_test)}) vs pred ({len(y_test_pred)})"
            )

        # Convert to numpy if needed
        y_train = np.asarray(y_train).flatten()
        y_train_pred = np.asarray(y_train_pred).flatten()
        y_test = np.asarray(y_test).flatten()
        y_test_pred = np.asarray(y_test_pred).flatten()

        # Compute train metrics
        train_rmse = float(np.sqrt(mean_squared_error(y_train, y_train_pred)))
        train_mae = float(mean_absolute_error(y_train, y_train_pred))
        train_r2 = float(r2_score(y_train, y_train_pred))

        # Compute test metrics
        test_rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
        test_mae = float(mean_absolute_error(y_test, y_test_pred))
        test_r2 = float(r2_score(y_test, y_test_pred))

        # Compute robustness score
        robustness_score = self.robustness_calc.compute(
            train_rmse=train_rmse,
            test_rmse=test_rmse,
            train_r2=train_r2,
            test_r2=test_r2,
        )

        # Compute overfitting ratio
        rmse_ratio = test_rmse / train_rmse if train_rmse > 0 else float("inf")

        # Assemble results
        metrics = {
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "train_mae": train_mae,
            "test_mae": test_mae,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "rmse_ratio": rmse_ratio,
            "robustness_score": robustness_score,
        }

        # Log to history
        record = {
            "model_name": model_name or "unknown",
            "scenario_name": scenario_name or "unknown",
            **metrics,
        }
        self._evaluation_history.append(record)

        # Log info
        if model_name:
            logger.info(
                f"✓ Evaluated {model_name} on {scenario_name or 'scenario'}:\n"
                f"    Train: RMSE={train_rmse:.4f}, MAE={train_mae:.4f}, R²={train_r2:.4f}\n"
                f"    Test:  RMSE={test_rmse:.4f}, MAE={test_mae:.4f}, R²={test_r2:.4f}\n"
                f"    Robustness: {robustness_score:.2f}/100 (RMSE ratio: {rmse_ratio:.2f}x)"
            )

        return metrics

    def evaluate_multiple_models(
        self,
        predictions_dict: Dict[str, Tuple[NDArray, NDArray]],
        y_train: NDArray,
        y_test: NDArray,
        scenario_name: Optional[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate multiple models at once.

        Args:
            predictions_dict: Dict mapping model_name -> (y_train_pred, y_test_pred)
            y_train: Ground truth training labels
            y_test: Ground truth test labels
            scenario_name: Optional scenario name

        Returns:
            Dict mapping model_name -> metrics_dict
        """
        results = {}

        for model_name, (y_train_pred, y_test_pred) in predictions_dict.items():
            try:
                metrics = self.evaluate(
                    y_train=y_train,
                    y_train_pred=y_train_pred,
                    y_test=y_test,
                    y_test_pred=y_test_pred,
                    model_name=model_name,
                    scenario_name=scenario_name,
                )
                results[model_name] = metrics
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
                results[model_name] = {"error": str(e)}

        return results

    def get_evaluation_summary(self) -> pd.DataFrame:
        """
        Get summary of all evaluations as DataFrame.

        Returns:
            DataFrame with all evaluation records
        """
        if not self._evaluation_history:
            return pd.DataFrame()

        return pd.DataFrame(self._evaluation_history)

    def get_best_models_per_scenario(
        self, metric: str = "robustness_score"
    ) -> Dict[str, str]:
        """
        Get best model name per scenario for a given metric.

        Args:
            metric: Metric to rank by (default: robustness_score)

        Returns:
            Dict mapping scenario_name -> best_model_name
        """
        if not self._evaluation_history:
            return {}

        df = self.get_evaluation_summary()

        best_models = {}
        for scenario in df["scenario_name"].unique():
            scenario_df = df[df["scenario_name"] == scenario]
            best_idx = scenario_df[metric].idxmax()
            best_model = scenario_df.loc[best_idx, "model_name"]
            best_models[scenario] = best_model

        return best_models

    def get_ranking_by_scenario(
        self, scenario_name: str, metric: str = "robustness_score", ascending: bool = False
    ) -> pd.DataFrame:
        """
        Get model rankings for a specific scenario.

        Args:
            scenario_name: Scenario to rank
            metric: Metric to rank by
            ascending: If True, lower is better; if False, higher is better

        Returns:
            DataFrame with models ranked by metric
        """
        if not self._evaluation_history:
            return pd.DataFrame()

        df = self.get_evaluation_summary()
        scenario_df = df[df["scenario_name"] == scenario_name].copy()

        if scenario_df.empty:
            logger.warning(f"No evaluations found for scenario: {scenario_name}")
            return pd.DataFrame()

        scenario_df = scenario_df.sort_values(metric, ascending=ascending).reset_index(
            drop=True
        )
        scenario_df["rank"] = range(1, len(scenario_df) + 1)

        return scenario_df[["rank", "model_name", metric]]


# =====================================================
# COMPARISON REPORT GENERATOR
# =====================================================
class ComparisonReportGenerator:
    """Generate markdown and JSON comparison reports from evaluation results."""

    def __init__(self, evaluator: CrossCityEvaluator) -> None:
        """
        Initialize report generator.

        Args:
            evaluator: CrossCityEvaluator instance with evaluation history
        """
        self.evaluator = evaluator

    def generate_markdown_report(self) -> str:
        """
        Generate comprehensive markdown comparison report.

        Returns:
            Markdown string
        """
        df = self.evaluator.get_evaluation_summary()

        if df.empty:
            return "# Comparison Report\n\nNo evaluations available.\n"

        report = "# Air Pollution Model Comparison Report\n\n"

        report += "## Executive Summary\n\n"
        report += f"- **Total Scenarios Evaluated**: {df['scenario_name'].nunique()}\n"
        report += f"- **Total Models Evaluated**: {df['model_name'].nunique()}\n"
        report += f"- **Total Evaluations**: {len(df)}\n\n"

        # Best models per scenario
        report += "## Best Models by Scenario (Robustness Score)\n\n"
        best_models = self.evaluator.get_best_models_per_scenario(
            metric="robustness_score"
        )

        for scenario, model in sorted(best_models.items()):
            row = df[(df["scenario_name"] == scenario) & (df["model_name"] == model)].iloc[0]
            score = row["robustness_score"]
            report += f"- **{scenario}**: {model} (Score: {score:.2f}/100)\n"

        report += "\n## Detailed Rankings by Scenario\n\n"

        for scenario in sorted(df["scenario_name"].unique()):
            report += f"### {scenario}\n\n"
            ranking_df = self.evaluator.get_ranking_by_scenario(scenario)

            report += "| Rank | Model | RMSE (Test) | MAE (Test) | R² (Test) | Robustness |\n"
            report += "|------|-------|-----------|---------|---------|-------------|\n"

            for _, row in ranking_df.iterrows():
                test_rmse = df[
                    (df["scenario_name"] == scenario) & (df["model_name"] == row["model_name"])
                ]["test_rmse"].values[0]
                test_mae = df[
                    (df["scenario_name"] == scenario) & (df["model_name"] == row["model_name"])
                ]["test_mae"].values[0]
                test_r2 = df[
                    (df["scenario_name"] == scenario) & (df["model_name"] == row["model_name"])
                ]["test_r2"].values[0]

                report += (
                    f"| {int(row['rank'])} | {row['model_name']} | "
                    f"{test_rmse:.4f} | {test_mae:.4f} | {test_r2:.4f} | "
                    f"{row['robustness_score']:.2f}/100 |\n"
                )

            report += "\n"

        # Summary metrics
        report += "## Overall Model Performance\n\n"
        report += "| Model | Avg Robustness | Avg Test RMSE | Avg Test R² |\n"
        report += "|-------|---------|---------|----------|\n"

        for model in sorted(df["model_name"].unique()):
            model_df = df[df["model_name"] == model]
            avg_robustness = model_df["robustness_score"].mean()
            avg_test_rmse = model_df["test_rmse"].mean()
            avg_test_r2 = model_df["test_r2"].mean()

            report += (
                f"| {model} | {avg_robustness:.2f}/100 | {avg_test_rmse:.4f} | "
                f"{avg_test_r2:.4f} |\n"
            )

        return report

    def generate_json_dict(self) -> Dict[str, Any]:
        """
        Generate evaluation results as JSON-serializable dict.

        Returns:
            Nested dictionary with all metrics
        """
        df = self.evaluator.get_evaluation_summary()

        if df.empty:
            return {"scenarios": []}

        results = {"scenarios": []}

        for scenario in sorted(df["scenario_name"].unique()):
            scenario_data = {
                "name": scenario,
                "models": [],
            }

            scenario_df = df[df["scenario_name"] == scenario]

            for _, row in scenario_df.iterrows():
                model_data = {
                    "name": row["model_name"],
                    "train": {
                        "rmse": round(row["train_rmse"], 6),
                        "mae": round(row["train_mae"], 6),
                        "r2": round(row["train_r2"], 6),
                    },
                    "test": {
                        "rmse": round(row["test_rmse"], 6),
                        "mae": round(row["test_mae"], 6),
                        "r2": round(row["test_r2"], 6),
                    },
                    "robustness_score": round(row["robustness_score"], 2),
                    "rmse_ratio": round(row["rmse_ratio"], 2),
                }
                scenario_data["models"].append(model_data)

            results["scenarios"].append(scenario_data)

        return results
