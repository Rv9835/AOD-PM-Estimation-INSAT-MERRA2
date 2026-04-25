#!/usr/bin/env python3
"""
Multi-City Model Evaluation Pipeline
Evaluates all trained models on all scenarios with robustness metrics.
Generates comparison reports (Markdown + JSON).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(PROJECT_ROOT / "backend"))
sys.path.append(str(PROJECT_ROOT / "backend" / "src"))

import logging
import json
from typing import Dict, List, Optional
import argparse

import pandas as pd
import yaml

from airpollution.multi_city_data import MultiCityDataLoader
from airpollution.models import ModelFactory
from airpollution.evaluators import CrossCityEvaluator, ComparisonReportGenerator

# =====================================================
# LOGGING SETUP
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "backend" / "logs" / "evaluate_pipeline.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# =====================================================
# EVALUATION ORCHESTRATOR
# =====================================================
class EvaluationOrchestrator:
    """
    Orchestrates model evaluation across all scenarios.
    """

    def __init__(self, config_path: str) -> None:
        """
        Initialize orchestrator.

        Args:
            config_path: Path to base.yaml configuration
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        # Load config
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.data_loader = MultiCityDataLoader(str(self.config_path))

        # Setup evaluator with config parameters
        robustness_config = self.config["evaluation"].get("robustness_score_config", {})
        self.evaluator = CrossCityEvaluator(
            overfitting_weight=robustness_config.get("overfitting_weight", 0.4),
            transfer_weight=robustness_config.get("transfer_weight", 0.6),
            acceptable_degradation=robustness_config.get("acceptable_degradation", 1.5),
        )

        # Setup artifacts directory
        self.model_dir = Path(self.config["artifacts"]["models_dir"])
        self.metrics_dir = Path(self.config["artifacts"]["metrics_dir"])
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"✓ Initialized EvaluationOrchestrator")
        logger.info(f"  Config: {self.config_path}")
        logger.info(f"  Model dir: {self.model_dir}")
        logger.info(f"  Metrics dir: {self.metrics_dir}")

    def get_model_algorithms(self) -> List[Dict]:
        """Get enabled model algorithms from config."""
        return [
            algo
            for algo in self.config["models"]["algorithms"]
            if algo.get("enabled", True)
        ]

    def load_trained_model(self, scenario_name: str, model_name: str) -> Optional[any]:
        """
        Load a saved model from disk.

        Args:
            scenario_name: Scenario identifier
            model_name: Model algorithm name

        Returns:
            Loaded model or None if file not found
        """
        model_path = self.model_dir / scenario_name / model_name / "model.joblib"

        if not model_path.exists():
            logger.warning(f"Model file not found: {model_path}")
            return None

        try:
            # Determine model class
            model_class = {
                "linear_regression": "LinearRegressionRegressor",
                "random_forest": "RandomForestRegressor",
                "xgboost": "XGBoostRegressor",
                "lightgbm": "LightGBMRegressor",
                "neural_network": "MLPNeuralNetworkRegressor",
            }.get(model_name)

            if not model_class:
                logger.error(f"Unknown model type: {model_name}")
                return None

            # Import the model class
            from airpollution.models.factory import (
                LinearRegressionRegressor,
                RandomForestRegressor,
                XGBoostRegressor,
                LightGBMRegressor,
                MLPNeuralNetworkRegressor,
            )

            classes = {
                "LinearRegressionRegressor": LinearRegressionRegressor,
                "RandomForestRegressor": RandomForestRegressor,
                "XGBoostRegressor": XGBoostRegressor,
                "LightGBMRegressor": LightGBMRegressor,
                "MLPNeuralNetworkRegressor": MLPNeuralNetworkRegressor,
            }

            model_cls = classes[model_class]
            model = model_cls.load(str(model_path))
            logger.debug(f"✓ Loaded model: {scenario_name}/{model_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            return None

    def evaluate_scenario(self, scenario_name: str) -> Dict[str, Dict]:
        """
        Evaluate all models for a specific scenario.

        Args:
            scenario_name: Scenario identifier

        Returns:
            Dict mapping model_name -> metrics

        Raises:
            ValueError: If data or models not found
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 EVALUATING SCENARIO: {scenario_name}")
        logger.info(f"{'='*60}")

        # Load data
        try:
            X_train, y_train, X_test, y_test = self.data_loader.split_by_scenario(
                scenario_name
            )
        except Exception as e:
            logger.error(f"✗ Failed to load data: {e}")
            raise

        logger.info(
            f"Data: {len(X_train)} train samples, {len(X_test)} test samples"
        )

        # Get enabled models
        algorithms = self.get_model_algorithms()
        scenario_results = {}

        for algo_config in algorithms:
            model_name = algo_config["name"]
            logger.info(f"\n  🔍 Evaluating: {model_name}")

            # Load model
            model = self.load_trained_model(scenario_name, model_name)
            if model is None:
                logger.warning(f"    ⚠ Skipping {model_name} (model not found)")
                continue

            try:
                # Generate predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # Evaluate
                metrics = self.evaluator.evaluate(
                    y_train=y_train,
                    y_train_pred=y_train_pred,
                    y_test=y_test,
                    y_test_pred=y_test_pred,
                    model_name=model_name,
                    scenario_name=scenario_name,
                )

                scenario_results[model_name] = metrics

                # Save metrics to CSV
                metrics_path = (
                    self.metrics_dir / scenario_name / f"{model_name}_metrics.json"
                )
                metrics_path.parent.mkdir(parents=True, exist_ok=True)

                with open(metrics_path, "w") as f:
                    json.dump(metrics, f, indent=2)
                logger.debug(f"    ✓ Saved metrics to {metrics_path}")

            except Exception as e:
                logger.error(f"    ✗ Evaluation failed: {e}")
                continue

        if not scenario_results:
            raise RuntimeError(f"No models evaluated for scenario: {scenario_name}")

        logger.info(f"\n✓ Scenario complete: {len(scenario_results)} models evaluated")
        return scenario_results

    def evaluate_all_scenarios(self) -> pd.DataFrame:
        """
        Evaluate all scenarios and compile results.

        Returns:
            DataFrame with all evaluation results
        """
        scenarios = self.config.get("training_scenarios", [])
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 STARTING EVALUATION PIPELINE")
        logger.info(f"{'='*60}")
        logger.info(f"Scenarios to evaluate: {len(scenarios)}")

        failed_scenarios = []

        for scenario_config in scenarios:
            scenario_name = scenario_config["name"]
            try:
                self.evaluate_scenario(scenario_name)
            except Exception as e:
                logger.error(f"✗ Scenario '{scenario_name}' failed: {e}")
                failed_scenarios.append(scenario_name)

        # Compile results
        results_df = self.evaluator.get_evaluation_summary()

        logger.info(f"\n{'='*60}")
        logger.info(f"✓ EVALUATION PIPELINE COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"✓ Evaluated scenarios: {results_df['scenario_name'].nunique()}")
        logger.info(f"✓ Total evaluations: {len(results_df)}")
        if failed_scenarios:
            logger.warning(f"⚠ Failed scenarios: {failed_scenarios}")

        return results_df

    def generate_reports(self) -> None:
        """Generate markdown and JSON comparison reports."""
        logger.info(f"\n{'='*60}")
        logger.info(f"📋 GENERATING REPORTS")
        logger.info(f"{'='*60}")

        results_df = self.evaluator.get_evaluation_summary()
        if results_df.empty:
            logger.warning("No evaluation results to report")
            return

        # Generate markdown report
        report_gen = ComparisonReportGenerator(self.evaluator)
        markdown_report = report_gen.generate_markdown_report()

        markdown_path = self.metrics_dir / "COMPARISON_REPORT.md"
        with open(markdown_path, "w") as f:
            f.write(markdown_report)
        logger.info(f"✓ Markdown report saved: {markdown_path}")

        # Generate JSON report
        json_dict = report_gen.generate_json_dict()

        json_path = self.metrics_dir / "comparison_results.json"
        with open(json_path, "w") as f:
            json.dump(json_dict, f, indent=2)
        logger.info(f"✓ JSON report saved: {json_path}")

        # Save full results CSV
        csv_path = self.metrics_dir / "all_results.csv"
        results_df.to_csv(csv_path, index=False)
        logger.info(f"✓ Results CSV saved: {csv_path}")

        logger.info(f"\n{'='*60}")
        logger.info(f"✓ ALL REPORTS GENERATED")
        logger.info(f"{'='*60}")


# =====================================================
# CLI INTERFACE
# =====================================================
def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained multi-city air pollution models"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="backend/configs/base.yaml",
        help="Path to configuration file (default: backend/configs/base.yaml)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Evaluate only a specific scenario (default: all)",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip report generation",
    )

    args = parser.parse_args()

    # Find config path
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).parent.parent / args.config
    if not config_path.exists():
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)

    # Initialize orchestrator
    try:
        orchestrator = EvaluationOrchestrator(str(config_path))
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {e}")
        sys.exit(1)

    # Evaluate
    try:
        if args.scenario:
            # Evaluate single scenario
            logger.info(f"Evaluating specific scenario: {args.scenario}")
            orchestrator.evaluate_scenario(args.scenario)
        else:
            # Evaluate all scenarios
            orchestrator.evaluate_all_scenarios()

        # Generate reports
        if not args.no_report:
            orchestrator.generate_reports()

    except Exception as e:
        logger.error(f"Evaluation pipeline failed: {e}")
        sys.exit(1)

    logger.info("✓ All evaluation complete!")


if __name__ == "__main__":
    main()
