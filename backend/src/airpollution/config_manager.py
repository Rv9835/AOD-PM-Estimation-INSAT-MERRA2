"""
Configuration Management Module

Provides runtime configuration validation, scenario management, and city registry updates.
Ensures consistency across the multi-city air pollution prediction pipeline.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
import yaml

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

from airpollution.cities import CityConfig, CityManager, CITIES_REGISTRY

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "backend" / "configs" / "base.yaml"


# =====================================================
# CONFIG VALIDATOR
# =====================================================
@dataclass
class ConfigValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ConfigValidator:
    """Validates YAML configuration files for consistency and completeness."""
    
    REQUIRED_TOP_LEVEL_KEYS = [
        "data",
        "models",
        "training_scenarios",
        "evaluation",
        "artifacts",
    ]
    
    REQUIRED_MODEL_FIELDS = ["name", "enabled"]  # hyperparams optional
    REQUIRED_SCENARIO_FIELDS = ["name", "training_cities", "test_cities"]  # description optional
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> ConfigValidationResult:
        """
        Validate entire configuration dictionary.
        
        Args:
            config: Configuration dictionary loaded from YAML
            
        Returns:
            ConfigValidationResult with validation status, errors, and warnings
        """
        result = ConfigValidationResult(is_valid=True)
        
        # Check top-level keys
        for key in ConfigValidator.REQUIRED_TOP_LEVEL_KEYS:
            if key not in config:
                result.errors.append(f"Missing required top-level key: '{key}'")
                result.is_valid = False
        
        # Validate data section
        if "data" in config:
            ConfigValidator._validate_data_section(config["data"], result)
        
        # Validate models section
        if "models" in config:
            ConfigValidator._validate_models_section(config["models"], result)
        
        # Validate training_scenarios section
        if "training_scenarios" in config:
            ConfigValidator._validate_scenarios_section(
                config["training_scenarios"],
                config.get("models", {}),
                result
            )
        
        # Validate evaluation section
        if "evaluation" in config:
            ConfigValidator._validate_evaluation_section(config["evaluation"], result)
        
        # Validate artifacts section
        if "artifacts" in config:
            ConfigValidator._validate_artifacts_section(config["artifacts"], result)
        
        return result
    
    @staticmethod
    def _validate_data_section(data_section: Dict, result: ConfigValidationResult) -> None:
        """Validate data configuration."""
        if not isinstance(data_section, dict):
            result.errors.append("'data' section must be a dictionary")
            return
        
        # Check for either dataset_path or unified_dataset_path
        has_dataset = "dataset_path" in data_section or "unified_dataset_path" in data_section
        if not has_dataset:
            result.warnings.append("No dataset path found in 'data' section")
        
        # Check dataset path exists if specified
        dataset_key = "unified_dataset_path" if "unified_dataset_path" in data_section else "dataset_path"
        if dataset_key in data_section:
            path = Path(data_section[dataset_key])
            if not path.exists():
                result.warnings.append(f"Dataset path does not exist: {data_section[dataset_key]}")
    
    @staticmethod
    def _validate_models_section(models_section: Dict, result: ConfigValidationResult) -> None:
        """Validate models configuration."""
        if not isinstance(models_section, dict):
            result.errors.append("'models' section must be a dictionary")
            return
        
        if "algorithms" not in models_section:
            result.errors.append("Missing 'models.algorithms'")
            result.is_valid = False
            return
        
        algorithms = models_section["algorithms"]
        if not isinstance(algorithms, list) or len(algorithms) == 0:
            result.errors.append("'models.algorithms' must be a non-empty list")
            result.is_valid = False
            return
        
        # Validate each algorithm
        enabled_count = 0
        valid_algorithms = {"linear_regression", "random_forest", "xgboost", "lightgbm", "neural_network"}
        
        for i, algo in enumerate(algorithms):
            if not isinstance(algo, dict):
                result.errors.append(f"Algorithm {i} is not a dictionary")
                result.is_valid = False
                continue
            
            for field in ConfigValidator.REQUIRED_MODEL_FIELDS:
                if field not in algo:
                    result.errors.append(f"Algorithm {i} missing required field: '{field}'")
                    result.is_valid = False
            
            if "name" in algo and algo["name"] not in valid_algorithms:
                result.errors.append(f"Unknown algorithm: '{algo['name']}' (Algorithm {i})")
                result.is_valid = False
            
            if algo.get("enabled", True):
                enabled_count += 1
        
        if enabled_count == 0:
            result.warnings.append("No algorithms are enabled. At least one algorithm should be enabled.")
    
    @staticmethod
    def _validate_scenarios_section(
        scenarios: List,
        models_section: Dict,
        result: ConfigValidationResult
    ) -> None:
        """Validate training scenarios."""
        if not isinstance(scenarios, list) or len(scenarios) == 0:
            result.errors.append("'training_scenarios' must be a non-empty list")
            result.is_valid = False
            return
        
        city_manager = CityManager()
        valid_cities = set(city_manager.list_cities())
        
        for i, scenario in enumerate(scenarios):
            if not isinstance(scenario, dict):
                result.errors.append(f"Scenario {i} is not a dictionary")
                result.is_valid = False
                continue
            
            for field in ConfigValidator.REQUIRED_SCENARIO_FIELDS:
                if field not in scenario:
                    result.errors.append(f"Scenario {i} missing required field: '{field}'")
                    result.is_valid = False
            
            # Validate cities
            for city in scenario.get("training_cities", []):
                if city not in valid_cities:
                    result.errors.append(f"Scenario {i}: Unknown training city '{city}'")
                    result.is_valid = False
            
            for city in scenario.get("test_cities", []):
                if city not in valid_cities:
                    result.errors.append(f"Scenario {i}: Unknown test city '{city}'")
                    result.is_valid = False
            
            # Check overlap (only warn for cross-city scenarios)
            train_set = set(scenario.get("training_cities", []))
            test_set = set(scenario.get("test_cities", []))
            
            if not train_set.isdisjoint(test_set):
                scenario_type = scenario.get("type", "unknown")
                if scenario_type != "standalone":
                    overlap = train_set & test_set
                    result.warnings.append(
                        f"Scenario {i} ({scenario.get('name', 'unknown')}) has overlapping cities: {overlap}"
                    )
    
    @staticmethod
    def _validate_evaluation_section(eval_section: Dict, result: ConfigValidationResult) -> None:
        """Validate evaluation configuration."""
        if not isinstance(eval_section, dict):
            result.errors.append("'evaluation' section must be a dictionary")
            return
        
        if "robustness_score" in eval_section:
            robustness = eval_section["robustness_score"]
            weights = robustness.get("weights", {})
            overfitting_weight = weights.get("overfitting_weight", 0)
            transfer_weight = weights.get("transfer_weight", 0)
            
            total = overfitting_weight + transfer_weight
            if abs(total - 1.0) > 0.01:
                result.warnings.append(
                    f"Robustness weights sum to {total}, not 1.0. "
                    "This may lead to unexpected scoring."
                )
    
    @staticmethod
    def _validate_artifacts_section(artifacts_section: Dict, result: ConfigValidationResult) -> None:
        """Validate artifacts configuration."""
        if not isinstance(artifacts_section, dict):
            result.errors.append("'artifacts' section must be a dictionary")
            return
        
        required_dirs = ["models_dir", "metrics_dir", "logs_dir"]
        for dir_key in required_dirs:
            if dir_key not in artifacts_section:
                result.warnings.append(f"Missing 'artifacts.{dir_key}'")


# =====================================================
# SCENARIO MANAGER
# =====================================================
class ScenarioManager:
    """Manages multi-city training scenarios with validation and persistence."""
    
    def __init__(self, config_path: str = str(DEFAULT_CONFIG_PATH)):
        """
        Initialize ScenarioManager.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.city_manager = CityManager()
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Validate
        validation = ConfigValidator.validate_config(config)
        if not validation.is_valid:
            errors_str = "\n".join(validation.errors)
            raise ValueError(f"Configuration validation failed:\n{errors_str}")
        
        # Log warnings
        for warning in validation.warnings:
            logger.warning(f"Config validation warning: {warning}")
        
        return config
    
    def _save_config(self) -> None:
        """Save current configuration back to YAML file."""
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
    
    def list_scenarios(self) -> List[str]:
        """
        List all scenario names.
        
        Returns:
            List of scenario names
        """
        return [s["name"] for s in self.config.get("training_scenarios", [])]
    
    def get_scenario(self, scenario_name: str) -> Optional[Dict[str, Any]]:
        """
        Get scenario configuration by name.
        
        Args:
            scenario_name: Name of the scenario
            
        Returns:
            Scenario dictionary or None if not found
        """
        for scenario in self.config.get("training_scenarios", []):
            if scenario["name"] == scenario_name:
                return scenario
        return None
    
    def add_scenario(
        self,
        name: str,
        description: str,
        training_cities: List[str],
        test_cities: List[str],
        scenario_type: str = "cross_city",
    ) -> None:
        """
        Add a new training scenario.
        
        Args:
            name: Scenario name
            description: Scenario description
            training_cities: List of training city keys
            test_cities: List of test city keys
            scenario_type: Type of scenario (cross_city, temporal, etc.)
            
        Raises:
            ValueError: If scenario already exists or cities are invalid
        """
        # Validate inputs
        if self.get_scenario(name) is not None:
            raise ValueError(f"Scenario '{name}' already exists")
        
        valid_cities = set(self.city_manager.list_cities())
        
        for city in training_cities:
            if city not in valid_cities:
                raise ValueError(f"Unknown training city: '{city}'")
        
        for city in test_cities:
            if city not in valid_cities:
                raise ValueError(f"Unknown test city: '{city}'")
        
        train_set = set(training_cities)
        test_set = set(test_cities)
        
        if not train_set.isdisjoint(test_set):
            raise ValueError(
                f"Training and test cities must not overlap. "
                f"Overlap: {train_set & test_set}"
            )
        
        # Add scenario
        scenario = {
            "name": name,
            "description": description,
            "training_cities": training_cities,
            "test_cities": test_cities,
            "type": scenario_type,
        }
        
        self.config["training_scenarios"].append(scenario)
        self._save_config()
        logger.info(f"Added scenario: {name}")
    
    def remove_scenario(self, scenario_name: str) -> None:
        """
        Remove a training scenario.
        
        Args:
            scenario_name: Name of scenario to remove
            
        Raises:
            ValueError: If scenario not found
        """
        scenarios = self.config.get("training_scenarios", [])
        original_count = len(scenarios)
        
        self.config["training_scenarios"] = [
            s for s in scenarios if s["name"] != scenario_name
        ]
        
        if len(self.config["training_scenarios"]) == original_count:
            raise ValueError(f"Scenario '{scenario_name}' not found")
        
        self._save_config()
        logger.info(f"Removed scenario: {scenario_name}")
    
    def update_scenario(
        self,
        scenario_name: str,
        **kwargs
    ) -> None:
        """
        Update scenario configuration.
        
        Args:
            scenario_name: Name of scenario to update
            **kwargs: Fields to update (description, training_cities, test_cities, etc.)
            
        Raises:
            ValueError: If scenario not found or validation fails
        """
        scenario = self.get_scenario(scenario_name)
        if scenario is None:
            raise ValueError(f"Scenario '{scenario_name}' not found")
        
        # Validate new cities if provided
        valid_cities = set(self.city_manager.list_cities())
        
        if "training_cities" in kwargs:
            for city in kwargs["training_cities"]:
                if city not in valid_cities:
                    raise ValueError(f"Unknown training city: '{city}'")
        
        if "test_cities" in kwargs:
            for city in kwargs["test_cities"]:
                if city not in valid_cities:
                    raise ValueError(f"Unknown test city: '{city}'")
        
        # Check overlap if both are being set
        if "training_cities" in kwargs and "test_cities" in kwargs:
            train_set = set(kwargs["training_cities"])
            test_set = set(kwargs["test_cities"])
            if not train_set.isdisjoint(test_set):
                raise ValueError(
                    f"Training and test cities must not overlap. "
                    f"Overlap: {train_set & test_set}"
                )
        
        # Update scenario
        scenario.update(kwargs)
        self._save_config()
        logger.info(f"Updated scenario: {scenario_name}")
    
    def get_scenario_stats(self, scenario_name: str) -> Dict[str, Any]:
        """
        Get statistics about a scenario.
        
        Args:
            scenario_name: Name of scenario
            
        Returns:
            Dictionary with scenario statistics
        """
        scenario = self.get_scenario(scenario_name)
        if scenario is None:
            return {}
        
        return {
            "name": scenario_name,
            "description": scenario.get("description", ""),
            "training_cities": scenario.get("training_cities", []),
            "test_cities": scenario.get("test_cities", []),
            "num_training_cities": len(scenario.get("training_cities", [])),
            "num_test_cities": len(scenario.get("test_cities", [])),
            "type": scenario.get("type", "unknown"),
        }


# =====================================================
# CITY REGISTRY MANAGER
# =====================================================
class CityRegistryManager:
    """Manages city registry with dynamic add/remove capabilities."""
    
    def __init__(self):
        """Initialize CityRegistryManager."""
        self.city_manager = CityManager()
    
    def get_all_cities(self) -> Dict[str, CityConfig]:
        """
        Get all registered cities.
        
        Returns:
            Dictionary mapping city keys to CityConfig objects
        """
        return self.city_manager.get_all_cities()
    
    def get_city(self, city_key: str) -> Optional[CityConfig]:
        """
        Get city configuration.
        
        Args:
            city_key: City identifier key
            
        Returns:
            CityConfig or None if not found
        """
        return self.city_manager.get_city_strict(city_key)
    
    def list_cities(self) -> List[str]:
        """List all city keys."""
        return self.city_manager.list_cities()
    
    def validate_bounding_box(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ) -> Tuple[bool, List[str]]:
        """
        Validate bounding box coordinates.
        
        Args:
            lat_min: Minimum latitude
            lat_max: Maximum latitude
            lon_min: Minimum longitude
            lon_max: Maximum longitude
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check latitude range
        if not (-90 <= lat_min <= 90 and -90 <= lat_max <= 90):
            errors.append("Latitude must be between -90 and 90")
        
        if lat_min >= lat_max:
            errors.append("lat_min must be less than lat_max")
        
        # Check longitude range
        if not (-180 <= lon_min <= 180 and -180 <= lon_max <= 180):
            errors.append("Longitude must be between -180 and 180")
        
        if lon_min >= lon_max:
            errors.append("lon_min must be less than lon_max")
        
        # Check box area (rough validation)
        lat_span = abs(lat_max - lat_min)
        lon_span = abs(lon_max - lon_min)
        
        if lat_span < 0.1 or lon_span < 0.1:
            errors.append("Bounding box is too small (minimum ~0.1 degree span recommended)")
        
        if lat_span > 10 or lon_span > 10:
            errors.append("Bounding box is very large (may inadvertently capture multiple cities)")
        
        return len(errors) == 0, errors
    
    def print_city_registry(self) -> None:
        """Print formatted city registry."""
        cities = self.get_all_cities()
        
        print("\n" + "="*80)
        print("CITY REGISTRY")
        print("="*80)
        
        for city_key, city_config in cities.items():
            lat_min, lat_max, lon_min, lon_max = city_config.bounding_box
            print(f"\n📍 {city_config.display_name} (Key: {city_key})")
            print(f"   Coordinates: ({city_config.center_lat:.4f}°N, {city_config.center_lon:.4f}°E)")
            print(f"   Bounding Box: [{lat_min:.4f}, {lat_max:.4f}] x "
                  f"[{lon_min:.4f}, {lon_max:.4f}]")
            print(f"   Altitude: {city_config.altitude_m}m")
            print(f"   Population Density: {city_config.population_density:.0f}/km²")


# =====================================================
# MAIN & CLI
# =====================================================
def main() -> None:
    """CLI for config management."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Air Pollution Pipeline - Configuration Manager"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to configuration file",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate configuration file",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List all training scenarios",
    )
    parser.add_argument(
        "--scenario-stats",
        type=str,
        help="Show statistics for a specific scenario",
    )
    parser.add_argument(
        "--list-cities",
        action="store_true",
        help="List all registered cities",
    )
    parser.add_argument(
        "--print-registry",
        action="store_true",
        help="Print formatted city registry",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    try:
        # Validate configuration
        if args.validate:
            print(f"\n📋 Validating configuration: {args.config}")
            with open(args.config, "r") as f:
                config = yaml.safe_load(f)
            
            validation = ConfigValidator.validate_config(config)
            
            if validation.is_valid:
                print("✓ Configuration is valid!\n")
            else:
                print("✗ Configuration validation failed:\n")
                for error in validation.errors:
                    print(f"  ✗ {error}")
            
            if validation.warnings:
                print("\n⚠️  Warnings:")
                for warning in validation.warnings:
                    print(f"  ⚠️  {warning}")
        
        # List scenarios
        elif args.list_scenarios:
            scenario_manager = ScenarioManager(args.config)
            scenarios = scenario_manager.list_scenarios()
            
            print(f"\n📊 Training Scenarios ({len(scenarios)} total):")
            for scenario_name in scenarios:
                scenario = scenario_manager.get_scenario(scenario_name)
                print(f"  • {scenario_name}")
                print(f"    Description: {scenario.get('description', 'N/A')}")
                print(f"    Training: {', '.join(scenario.get('training_cities', []))}")
                print(f"    Test: {', '.join(scenario.get('test_cities', []))}\n")
        
        # Show scenario stats
        elif args.scenario_stats:
            scenario_manager = ScenarioManager(args.config)
            stats = scenario_manager.get_scenario_stats(args.scenario_stats)
            
            if stats:
                print(f"\n📊 Scenario Statistics: {args.scenario_stats}")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
            else:
                print(f"✗ Scenario not found: {args.scenario_stats}")
        
        # List cities
        elif args.list_cities:
            city_manager = CityRegistryManager()
            cities = city_manager.list_cities()
            
            print(f"\n🏙️  Registered Cities ({len(cities)} total):")
            for city_key in cities:
                city_config = city_manager.get_city(city_key)
                if city_config:
                    print(f"  • {city_config.display_name} ({city_key})")
        
        # Print registry
        elif args.print_registry:
            city_manager = CityRegistryManager()
            city_manager.print_city_registry()
        
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
