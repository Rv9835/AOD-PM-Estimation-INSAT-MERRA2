#!/usr/bin/env python3
"""
Multi-City Model Training Pipeline
Trains all 5 ML models across all training scenarios defined in base.yaml.
Saves trained models to artifacts/models/{scenario}/{algorithm}/.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(PROJECT_ROOT / "backend"))
sys.path.append(str(PROJECT_ROOT / "backend" / "src"))

import logging
from typing import Dict, List, Optional
import argparse

import pandas as pd
import yaml

from airpollution.multi_city_data import MultiCityDataLoader
from airpollution.models import ModelFactory

# =====================================================
# LOGGING SETUP
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "backend" / "logs" / "train_pipeline.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# =====================================================
# TRAINING ORCHESTRATOR
# =====================================================
class TrainingOrchestrator:
    """
    Orchestrates multi-city model training across scenarios.
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
        self.trained_models: Dict[str, Dict] = {}  # {scenario: {model_name: model}}

        # Setup artifacts directory
        self.model_dir = Path(self.config["artifacts"]["models_dir"])
        self.model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"✓ Initialized TrainingOrchestrator")
        logger.info(f"  Config: {self.config_path}")
        logger.info(f"  Model output: {self.model_dir}")

    def get_model_algorithms(self) -> List[Dict]:
        """Get enabled model algorithms from config."""
        return [
            algo
            for algo in self.config["models"]["algorithms"]
            if algo.get("enabled", True)
        ]

    def train_scenario(self, scenario_name: str) -> Dict[str, any]:
        """
        Train all enabled models for a specific scenario.

        Args:
            scenario_name: Name of scenario from config

        Returns:
            Dict mapping model_name -> trained model instance

        Raises:
            ValueError: If scenario not found or training fails
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 TRAINING SCENARIO: {scenario_name}")
        logger.info(f"{'='*60}")

        # Split data by scenario
        try:
            X_train, y_train, X_test, y_test = self.data_loader.split_by_scenario(
                scenario_name
            )
        except Exception as e:
            logger.error(f"✗ Failed to load data for scenario '{scenario_name}': {e}")
            raise

        # Get enabled models
        algorithms = self.get_model_algorithms()
        trained_models = {}

        for algo_config in algorithms:
            model_name = algo_config["name"]
            logger.info(f"\n  🤖 Training: {model_name}")

            try:
                # Create model
                model = ModelFactory.create_from_config(algo_config)

                # Train
                model.fit(X_train, y_train)

                # Save model
                model_path = self.model_dir / scenario_name / model_name / "model.joblib"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                model.save(str(model_path))
                logger.info(f"    ✓ Saved to {model_path}")

                # Save feature importance
                importance = model.get_feature_importance()
                importance_df = pd.DataFrame(
                    [{"feature": k, "importance": v} for k, v in importance.items()]
                )
                importance_path = model_path.parent / "feature_importance.csv"
                importance_df.to_csv(importance_path, index=False)
                logger.info(f"    ✓ Feature importance saved")

                trained_models[model_name] = model

            except Exception as e:
                logger.error(f"    ✗ Training failed for {model_name}: {e}")
                continue

        if not trained_models:
            raise RuntimeError(f"No models succeeded for scenario: {scenario_name}")

        self.trained_models[scenario_name] = trained_models
        logger.info(f"\n✓ Scenario complete: {len(trained_models)} models trained")

        return trained_models

    def train_all_scenarios(self) -> Dict[str, Dict]:
        """
        Train models for all scenarios in config.

        Returns:
            Dict mapping scenario_name -> {model_name -> trained_model}
        """
        scenarios = self.config.get("training_scenarios", [])
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 STARTING TRAINING PIPELINE")
        logger.info(f"{'='*60}")
        logger.info(f"Scenarios to train: {len(scenarios)}")
        logger.info(f"Models to train per scenario: {len(self.get_model_algorithms())}")

        failed_scenarios = []

        for scenario_config in scenarios:
            scenario_name = scenario_config["name"]
            try:
                self.train_scenario(scenario_name)
            except Exception as e:
                logger.error(f"✗ Scenario '{scenario_name}' failed: {e}")
                failed_scenarios.append(scenario_name)

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info(f"✓ TRAINING PIPELINE COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"✓ Successfully trained: {len(self.trained_models)} scenarios")
        if failed_scenarios:
            logger.warning(f"⚠ Failed scenarios: {failed_scenarios}")

        return self.trained_models

    def get_trained_model(self, scenario_name: str, model_name: str) -> Optional[any]:
        """Get a trained model instance."""
        return self.trained_models.get(scenario_name, {}).get(model_name)


# =====================================================
# CLI INTERFACE
# =====================================================
def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train multi-city air pollution prediction models"
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
        help="Train only a specific scenario (default: all)",
    )

    args = parser.parse_args()

    # Find config path (search from script directory or CWD)
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).parent.parent / args.config
    if not config_path.exists():
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)

    # Initialize orchestrator
    try:
        orchestrator = TrainingOrchestrator(str(config_path))
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {e}")
        sys.exit(1)

    # Train
    try:
        if args.scenario:
            # Train single scenario
            logger.info(f"Training specific scenario: {args.scenario}")
            orchestrator.train_scenario(args.scenario)
        else:
            # Train all scenarios
            orchestrator.train_all_scenarios()
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        sys.exit(1)

    logger.info("✓ All training complete!")


if __name__ == "__main__":
    main()
