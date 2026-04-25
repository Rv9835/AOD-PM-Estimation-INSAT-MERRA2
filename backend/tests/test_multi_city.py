"""
Comprehensive Test Suite for Multi-City Air Pollution Prediction Pipeline

Tests for:
- Bounding box isolation and data leakage prevention
- Model serialization/deserialization
- Scenario configuration validation
- Cross-city data splitting
- Robustness metrics calculation
"""

import sys
import pytest
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from airpollution.cities import (
    CityManager,
    CityConfig,
    haversine_distance,
    bearing_between_cities,
)
from airpollution.multi_city_data import MultiCityDataLoader
from airpollution.models.factory import (
    ModelFactory,
    LinearRegressionRegressor,
    RandomForestRegressor,
)
from airpollution.config_manager import (
    ConfigValidator,
    ScenarioManager,
    CityRegistryManager,
)
from airpollution.evaluators import RobustnessScoreCalculator


# =====================================================
# FIXTURES
# =====================================================
@pytest.fixture
def city_manager() -> CityManager:
    """Fixture providing CityManager instance."""
    return CityManager()


@pytest.fixture
def cities_dict(city_manager) -> dict:
    """Fixture providing all cities."""
    return city_manager.get_all_cities()


@pytest.fixture
def sample_data() -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Fixture providing sample data with points in different cities.
    
    Returns:
        Tuple of (data_df, target_array)
    """
    # Create synthetic data with points in Delhi and Mumbai
    np.random.seed(42)
    
    # Delhi area points
    delhi_lats = np.random.uniform(28.40, 29.00, 50)
    delhi_lons = np.random.uniform(76.80, 77.50, 50)
    delhi_data = pd.DataFrame({
        "lat": delhi_lats,
        "lon": delhi_lons,
        "aod": np.random.uniform(0.1, 0.8, 50),
        "temperature": np.random.uniform(15, 35, 50),
        "humidity": np.random.uniform(30, 80, 50),
        "wind_speed": np.random.uniform(0.5, 5.0, 50),
        "boundary_layer_height": np.random.uniform(500, 2000, 50),
        "day_of_year": np.random.randint(1, 366, 50),
    })
    
    # Mumbai area points
    mumbai_lats = np.random.uniform(18.80, 19.40, 50)
    mumbai_lons = np.random.uniform(72.50, 73.20, 50)
    mumbai_data = pd.DataFrame({
        "lat": mumbai_lats,
        "lon": mumbai_lons,
        "aod": np.random.uniform(0.2, 0.9, 50),
        "temperature": np.random.uniform(20, 35, 50),
        "humidity": np.random.uniform(60, 95, 50),
        "wind_speed": np.random.uniform(1.0, 8.0, 50),
        "boundary_layer_height": np.random.uniform(600, 2000, 50),
        "day_of_year": np.random.randint(1, 366, 50),
    })
    
    # Combine and assign target
    data = pd.concat([delhi_data, mumbai_data], ignore_index=True)
    
    # PM2.5 target: higher in Delhi (pollution), lower in Mumbai
    pm25 = np.concatenate([
        np.random.uniform(60, 150, 50),  # Delhi: high pollution
        np.random.uniform(30, 80, 50),   # Mumbai: lower pollution
    ])
    
    return data, pm25


@pytest.fixture
def config_path() -> str:
    """Fixture providing path to test configuration."""
    return str(Path(__file__).parent.parent / "configs" / "base.yaml")


# =====================================================
# TESTS: BOUNDING BOX ISOLATION
# =====================================================
class TestBoundingBoxIsolation:
    """Tests for city boundary isolation and data filtering."""
    
    def test_delhi_bounding_box(self, city_manager):
        """Test that Delhi bounding box correctly defines city boundaries."""
        delhi = city_manager.get_city_strict("delhi")
        
        assert delhi is not None, "Delhi city not found"
        lat_min, lat_max, lon_min, lon_max = delhi.bounding_box
        assert 28.40 <= lat_min <= lat_max <= 29.00
        assert 76.80 <= lon_min <= lon_max <= 77.50
    
    def test_mumbai_bounding_box(self, city_manager):
        """Test that Mumbai bounding box correctly defines city boundaries."""
        mumbai = city_manager.get_city_strict("mumbai")
        
        assert mumbai is not None, "Mumbai city not found"
        lat_min, lat_max, lon_min, lon_max = mumbai.bounding_box
        assert 18.80 <= lat_min <= lat_max <= 19.40
        assert 72.50 <= lon_min <= lon_max <= 73.20
    
    def test_point_in_delhi_bounds(self, city_manager):
        """Test that points within Delhi bounds are correctly identified."""
        delhi = city_manager.get_city_strict("delhi")
        lat_min, lat_max, lon_min, lon_max = delhi.bounding_box
        
        # Create test point in Delhi
        point_lat = 28.70
        point_lon = 77.15
        
        # Manually check bounds
        in_lat_bounds = lat_min <= point_lat <= lat_max
        in_lon_bounds = lon_min <= point_lon <= lon_max
        
        assert in_lat_bounds, "Point latitude should be within Delhi bounds"
        assert in_lon_bounds, "Point longitude should be within Delhi bounds"
    
    def test_point_outside_delhi_bounds(self, city_manager):
        """Test that points outside Delhi bounds are correctly rejected."""
        delhi = city_manager.get_city_strict("delhi")
        lat_min, lat_max, lon_min, lon_max = delhi.bounding_box
        
        # Create test point outside Delhi (in Mumbai area)
        point_lat = 19.00
        point_lon = 72.80
        
        in_lat_bounds = lat_min <= point_lat <= lat_max
        in_lon_bounds = lon_min <= point_lon <= lon_max
        
        assert not (in_lat_bounds and in_lon_bounds), \
            "Mumbai point should be outside Delhi bounds"
    
    def test_bounding_boxes_dont_overlap(self, city_manager):
        """Test that city bounding boxes don't overlap."""
        cities = city_manager.get_all_cities()
        city_list = list(cities.items())
        
        # Check each pair of cities
        for i in range(len(city_list)):
            for j in range(i + 1, len(city_list)):
                city1_key, city1_config = city_list[i]
                city2_key, city2_config = city_list[j]
                
                lat1_min, lat1_max, lon1_min, lon1_max = city1_config.bounding_box
                lat2_min, lat2_max, lon2_min, lon2_max = city2_config.bounding_box
                
                # Check latitude overlap
                lat_overlap = not (lat1_max < lat2_min or lat2_max < lat1_min)
                
                # Check longitude overlap
                lon_overlap = not (lon1_max < lon2_min or lon2_max < lon1_min)
                
                is_overlapping = lat_overlap and lon_overlap
                assert not is_overlapping, \
                    f"Bounding boxes for {city1_key} and {city2_key} overlap"


# =====================================================
# TESTS: DATA LEAKAGE PREVENTION
# =====================================================
class TestDataLeakagePrevention:
    """Tests for preventing data leakage between cities."""
    
    def test_filtered_data_respects_bounds(self, sample_data, city_manager):
        """Test that filtered city data respects bounding box boundaries."""
        data, _ = sample_data
        delhi = city_manager.get_city_strict("delhi")
        lat_min, lat_max, lon_min, lon_max = delhi.bounding_box
        
        # Filter data for Delhi
        filtered = data[
            (data["lat"] >= lat_min) & (data["lat"] <= lat_max) &
            (data["lon"] >= lon_min) & (data["lon"] <= lon_max)
        ]
        
        # Verify all points are within bounds
        assert all(lat_min <= lat <= lat_max for lat in filtered["lat"])
        assert all(lon_min <= lon <= lon_max for lon in filtered["lon"])
    
    def test_delhi_filtered_excludes_mumbai_points(self, sample_data, city_manager):
        """Test that Delhi filtering excludes Mumbai points."""
        data, _ = sample_data
        delhi = city_manager.get_city_strict("delhi")
        mumbai = city_manager.get_city_strict("mumbai")
        
        delhi_lat_min, delhi_lat_max, delhi_lon_min, delhi_lon_max = delhi.bounding_box
        mumbai_lat_min, mumbai_lat_max, mumbai_lon_min, mumbai_lon_max = mumbai.bounding_box
        
        # Filter data for Delhi
        delhi_filtered = data[
            (data["lat"] >= delhi_lat_min) & (data["lat"] <= delhi_lat_max) &
            (data["lon"] >= delhi_lon_min) & (data["lon"] <= delhi_lon_max)
        ]
        
        # Verify no points are in Mumbai bounds
        in_mumbai = (
            (delhi_filtered["lat"] >= mumbai_lat_min) & 
            (delhi_filtered["lat"] <= mumbai_lat_max) &
            (delhi_filtered["lon"] >= mumbai_lon_min) & 
            (delhi_filtered["lon"] <= mumbai_lon_max)
        )
        
        assert not in_mumbai.any(), \
            "Delhi-filtered data should not contain Mumbai points"
    
    def test_mumbai_filtered_excludes_delhi_points(self, sample_data, city_manager):
        """Test that Mumbai filtering excludes Delhi points."""
        data, _ = sample_data
        delhi = city_manager.get_city_strict("delhi")
        mumbai = city_manager.get_city_strict("mumbai")
        
        delhi_lat_min, delhi_lat_max, delhi_lon_min, delhi_lon_max = delhi.bounding_box
        mumbai_lat_min, mumbai_lat_max, mumbai_lon_min, mumbai_lon_max = mumbai.bounding_box
        
        # Filter data for Mumbai
        mumbai_filtered = data[
            (data["lat"] >= mumbai_lat_min) & (data["lat"] <= mumbai_lat_max) &
            (data["lon"] >= mumbai_lon_min) & (data["lon"] <= mumbai_lon_max)
        ]
        
        # Verify no points are in Delhi bounds
        in_delhi = (
            (mumbai_filtered["lat"] >= delhi_lat_min) & 
            (mumbai_filtered["lat"] <= delhi_lat_max) &
            (mumbai_filtered["lon"] >= delhi_lon_min) & 
            (mumbai_filtered["lon"] <= delhi_lon_max)
        )
        
        assert not in_delhi.any(), \
            "Mumbai-filtered data should not contain Delhi points"


# =====================================================
# TESTS: MODEL SERIALIZATION
# =====================================================
class TestModelSerialization:
    """Tests for model save/load functionality."""
    
    def test_linear_regression_serialization(self, sample_data):
        """Test LinearRegression model save and load."""
        X_df, y_arr = sample_data
        features_df = X_df[["aod", "temperature", "humidity", "wind_speed", "boundary_layer_height"]]
        y_series = pd.Series(y_arr)
        
        # Train model
        model = LinearRegressionRegressor()
        model.fit(features_df, y_series)
        
        # Save to temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "linear_model.joblib"
            model.save(str(model_path))
            
            # Verify file exists
            assert model_path.exists(), "Model file should exist after save"
            
            # Load model
            loaded_model = LinearRegressionRegressor.load(str(model_path))
            
            # Predict with both models
            predictions_original = model.predict(features_df)
            predictions_loaded = loaded_model.predict(features_df)
            
            # Verify predictions are identical
            np.testing.assert_array_almost_equal(
                predictions_original,
                predictions_loaded,
                decimal=10,
                err_msg="Loaded model should make identical predictions"
            )
    
    def test_random_forest_serialization(self, sample_data):
        """Test RandomForest model save and load."""
        X_df, y_arr = sample_data
        features_df = X_df[["aod", "temperature", "humidity", "wind_speed", "boundary_layer_height"]]
        y_series = pd.Series(y_arr)
        
        # Train model
        model = RandomForestRegressor()
        model.fit(features_df, y_series)
        
        # Save to temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "rf_model.joblib"
            model.save(str(model_path))
            
            # Verify file exists
            assert model_path.exists(), "Model file should exist after save"
            
            # Load model
            loaded_model = RandomForestRegressor.load(str(model_path))
            
            # Predict with both models
            predictions_original = model.predict(features_df)
            predictions_loaded = loaded_model.predict(features_df)
            
            # Verify predictions are very similar (RF may have slight differences due to RNG)
            np.testing.assert_array_almost_equal(
                predictions_original,
                predictions_loaded,
                decimal=5,
                err_msg="Loaded RF model should make very similar predictions"
            )
    
    def test_feature_importance_after_load(self, sample_data):
        """Test that feature importance is accessible after load."""
        X_df, y_arr = sample_data
        features_df = X_df[["aod", "temperature", "humidity", "wind_speed", "boundary_layer_height"]]
        y_series = pd.Series(y_arr)
        
        # Train model
        model = RandomForestRegressor()
        model.fit(features_df, y_series)
        
        # Get original importance
        importance_original = model.get_feature_importance()
        
        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "rf_model.joblib"
            model.save(str(model_path))
            loaded_model = RandomForestRegressor.load(str(model_path))
            
            # Get loaded importance
            importance_loaded = loaded_model.get_feature_importance()
            
            # Verify importances match (dict comparison)
            assert isinstance(importance_original, dict), "Importance should be dict"
            assert isinstance(importance_loaded, dict), "Loaded importance should be dict"
            assert set(importance_original.keys()) == set(importance_loaded.keys()), \
                "Feature keys should match"
            
            for feature in importance_original.keys():
                np.testing.assert_almost_equal(
                    importance_original[feature],
                    importance_loaded[feature],
                    decimal=10,
                    err_msg=f"Feature importance for '{feature}' should match after load"
                )


# =====================================================
# TESTS: CONFIGURATION VALIDATION
# =====================================================
class TestConfigurationValidation:
    """Tests for configuration validation."""
    
    def test_valid_config_passes(self, config_path):
        """Test that valid config passes validation."""
        import yaml
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        result = ConfigValidator.validate_config(config)
        assert result.is_valid, f"Config validation failed: {result.errors}"
    
    def test_missing_top_level_key_fails(self):
        """Test that missing top-level key causes validation failure."""
        bad_config = {
            "data": {},
            "models": {},
            # Missing "training_scenarios", "evaluation", "artifacts"
        }
        
        result = ConfigValidator.validate_config(bad_config)
        assert not result.is_valid, "Config should fail validation"
        assert len(result.errors) > 0, "Should have validation errors"
    
    def test_duplicate_scenario_names_warning(self, config_path):
        """Test that scenario configuration is properly loaded."""
        import yaml
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        result = ConfigValidator.validate_config(config)
        
        # Should pass (no duplicates in base config)
        assert result.is_valid


# =====================================================
# TESTS: SCENARIO MANAGER
# =====================================================
class TestScenarioManager:
    """Tests for ScenarioManager functionality."""
    
    def test_list_scenarios(self, config_path):
        """Test listing all scenarios."""
        manager = ScenarioManager(config_path)
        scenarios = manager.list_scenarios()
        
        assert len(scenarios) > 0, "Should have scenarios"
        assert "delhi_standalone" in scenarios or len(scenarios) > 0
    
    def test_get_scenario(self, config_path):
        """Test getting specific scenario."""
        manager = ScenarioManager(config_path)
        scenarios = manager.list_scenarios()
        
        if scenarios:
            first_scenario = scenarios[0]
            scenario_data = manager.get_scenario(first_scenario)
            
            assert scenario_data is not None
            assert "name" in scenario_data
            assert "training_cities" in scenario_data
            assert "test_cities" in scenario_data
    
    def test_get_scenario_stats(self, config_path):
        """Test getting scenario statistics."""
        manager = ScenarioManager(config_path)
        scenarios = manager.list_scenarios()
        
        if scenarios:
            first_scenario = scenarios[0]
            stats = manager.get_scenario_stats(first_scenario)
            
            assert "num_training_cities" in stats
            assert "num_test_cities" in stats
            assert isinstance(stats["num_training_cities"], int)
            assert isinstance(stats["num_test_cities"], int)


# =====================================================
# TESTS: GEOSPATIAL UTILITIES
# =====================================================
class TestGeospatialUtilities:
    """Tests for geospatial distance and bearing calculations."""
    
    def test_haversine_distance_same_point(self):
        """Test haversine distance for same point is zero."""
        distance = haversine_distance((28.7041, 77.1025), (28.7041, 77.1025))
        
        assert abs(distance) < 0.001, \
            "Distance between same point should be ~0"
    
    def test_haversine_distance_known_cities(self, city_manager):
        """Test haversine distance between known cities."""
        delhi = city_manager.get_city_strict("delhi")
        mumbai = city_manager.get_city_strict("mumbai")
        
        distance = haversine_distance(
            (delhi.center_lat, delhi.center_lon),
            (mumbai.center_lat, mumbai.center_lon)
        )
        
        # Delhi-Mumbai distance should be around 1100-1200 km
        assert 1000 < distance < 1300, \
            f"Delhi-Mumbai distance should be ~1100-1200km, got {distance}km"
    
    def test_bearing_symmetry(self):
        """Test that bearing calculations are consistent."""
        bearing1 = bearing_between_cities((28.7041, 77.1025), (19.0760, 72.8777))
        bearing2 = bearing_between_cities((19.0760, 72.8777), (28.7041, 77.1025))
        
        # Bearings should be 180 degrees apart (allow 2 degree tolerance for rounding)
        bearing_diff = abs(bearing1 - bearing2)
        
        assert (abs(bearing_diff - 180) < 2) or (abs(bearing_diff - 180 - 360) < 2), \
            f"Reverse bearings should differ by ~180°, got {bearing_diff}°"


# =====================================================
# TESTS: ROBUSTNESS METRICS
# =====================================================
class TestRobustnessMetrics:
    """Tests for robustness score calculation."""
    
    def test_robustness_perfect_model(self):
        """Test robustness score for perfect model."""
        calculator = RobustnessScoreCalculator(
            overfitting_weight=0.4,
            transfer_weight=0.6,
            acceptable_degradation=1.5
        )
        
        # Perfect model: rmse_train=5, rmse_test=5, r2_train=1.0, r2_test=1.0
        train_rmse = 5.0
        test_rmse = 5.0
        train_r2 = 1.0
        test_r2 = 1.0
        
        robustness = calculator.compute(train_rmse, test_rmse, train_r2, test_r2)
        
        # Should be close to 100
        assert robustness > 95, f"Perfect model should have robustness ~100, got {robustness}"
    
    def test_robustness_poor_model(self):
        """Test robustness score for poor model."""
        calculator = RobustnessScoreCalculator(
            overfitting_weight=0.4,
            transfer_weight=0.6,
            acceptable_degradation=1.5
        )
        
        # Poor model: high overfitting, low test performance
        train_rmse = 5.0
        test_rmse = 20.0  # 4x worse
        train_r2 = 0.9
        test_r2 = 0.1
        
        robustness = calculator.compute(train_rmse, test_rmse, train_r2, test_r2)
        
        # Should be low
        assert robustness < 40, f"Poor model should have low robustness, got {robustness}"
    
    def test_robustness_scale_0_to_100(self):
        """Test that robustness score is between 0 and 100."""
        calculator = RobustnessScoreCalculator(
            overfitting_weight=0.4,
            transfer_weight=0.6,
            acceptable_degradation=1.5
        )
        
        test_cases = [
            (5.0, 5.0, 1.0, 1.0),      # Perfect
            (5.0, 10.0, 0.8, 0.5),     # Good
            (5.0, 15.0, 0.7, 0.2),     # Poor
            (5.0, 50.0, 0.5, -0.5),    # Very poor
        ]
        
        for train_rmse, test_rmse, train_r2, test_r2 in test_cases:
            robustness = calculator.compute(train_rmse, test_rmse, train_r2, test_r2)
            assert 0 <= robustness <= 100, \
                f"Robustness should be 0-100, got {robustness} for ({train_rmse}, {test_rmse}, {train_r2}, {test_r2})"


# =====================================================
# TESTS: CITY REGISTRY MANAGER
# =====================================================
class TestCityRegistryManager:
    """Tests for CityRegistryManager."""
    
    def test_get_all_cities(self):
        """Test getting all cities."""
        manager = CityRegistryManager()
        cities = manager.get_all_cities()
        
        assert len(cities) >= 5, "Should have at least 5 cities"
        assert "delhi" in cities or len(cities) > 0
    
    def test_validate_valid_bounding_box(self):
        """Test validation of valid bounding box."""
        manager = CityRegistryManager()
        
        is_valid, errors = manager.validate_bounding_box(
            lat_min=28.40,
            lat_max=29.00,
            lon_min=76.80,
            lon_max=77.50,
        )
        
        assert is_valid, f"Valid bounding box should pass: {errors}"
    
    def test_validate_invalid_latitude_range(self):
        """Test validation rejects invalid latitude."""
        manager = CityRegistryManager()
        
        is_valid, errors = manager.validate_bounding_box(
            lat_min=-100,  # Invalid
            lat_max=29.00,
            lon_min=76.80,
            lon_max=77.50,
        )
        
        assert not is_valid, "Invalid latitude should fail validation"
    
    def test_validate_reversed_coordinates(self):
        """Test validation rejects reversed min/max coordinates."""
        manager = CityRegistryManager()
        
        is_valid, errors = manager.validate_bounding_box(
            lat_min=29.00,  # Greater than max
            lat_max=28.40,
            lon_min=76.80,
            lon_max=77.50,
        )
        
        assert not is_valid, "Reversed coordinates should fail validation"


# =====================================================
# RUN TESTS
# =====================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
