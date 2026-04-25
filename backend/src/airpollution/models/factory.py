"""
Model Factory and Regressor Implementations
Provides abstract BaseRegressor and 5 concrete ML algorithms with save/load/importance.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Type

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor as SKRandomForestRegressor
from xgboost import XGBRegressor as XGBRegressorModel
from lightgbm import LGBMRegressor as LGBMRegressorModel

try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None  # type: ignore
    keras = None  # type: ignore

logger = logging.getLogger(__name__)


# =====================================================
# TYPE ALIASES
# =====================================================
DataFrame = pd.DataFrame
Series = pd.Series
NDArray = np.ndarray


# =====================================================
# ABSTRACT BASE CLASS
# =====================================================
class BaseRegressor(ABC):
    """
    Abstract base class for all regression models.
    Enforces consistent interface for training, prediction, serialization, and analysis.
    """

    def __init__(self, name: str, random_state: int = 42) -> None:
        """
        Initialize base regressor.

        Args:
            name: Model name identifier (e.g., "random_forest", "xgboost")
            random_state: Random seed for reproducibility
        """
        self.name = name
        self.random_state = random_state
        self.model: Any = None
        self.feature_names_: Optional[List[str]] = None
        self.is_fitted = False

        logger.debug(f"Initialized {self.__class__.__name__} (name={name})")

    @abstractmethod
    def fit(self, X: DataFrame, y: Series, **kwargs: Any) -> None:
        """
        Fit the model to training data.

        Args:
            X: Training features DataFrame
            y: Training target Series
            **kwargs: Additional model-specific parameters
        """
        pass

    @abstractmethod
    def predict(self, X: DataFrame) -> NDArray:
        """
        Generate predictions on input features.

        Args:
            X: Features DataFrame

        Returns:
            Predictions array

        Raises:
            RuntimeError: If model not fitted
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores from the trained model.

        Returns:
            Dictionary mapping feature names to importance values

        Raises:
            RuntimeError: If model not fitted or doesn't support feature importance
        """
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """
        Save trained model to disk.

        Args:
            filepath: Target path for model file

        Raises:
            IOError: If save fails
            RuntimeError: If model not fitted
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, filepath: str) -> "BaseRegressor":
        """
        Load saved model from disk.

        Args:
            filepath: Path to saved model file

        Returns:
            Loaded BaseRegressor instance

        Raises:
            FileNotFoundError: If file doesn't exist
            IOError: If load fails
        """
        pass

    def _check_fitted(self) -> None:
        """Check if model has been fitted. Raises RuntimeError if not."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before prediction."
            )


# =====================================================
# 1. LINEAR REGRESSION
# =====================================================
class LinearRegressionRegressor(BaseRegressor):
    """
    Linear Regression model with closed-form solution.
    Baseline model with OLS (Ordinary Least Squares) fitting.
    """

    def __init__(self, random_state: int = 42) -> None:
        """Initialize Linear Regression."""
        super().__init__(name="linear_regression", random_state=random_state)
        self.model = LinearRegression()

    def fit(self, X: DataFrame, y: Series, **kwargs: Any) -> None:
        """Fit linear regression model."""
        if X.empty or y.empty:
            raise ValueError("Cannot fit with empty data")

        self.feature_names_ = list(X.columns)
        logger.info(f"Fitting LinearRegression with {len(X)} samples, {len(X.columns)} features")

        try:
            self.model.fit(X, y)
            self.is_fitted = True
            logger.info(f"✓ LinearRegression fitted. R² (train): {self.model.score(X, y):.4f}")
        except Exception as e:
            logger.error(f"✗ LinearRegression fit failed: {e}")
            raise

    def predict(self, X: DataFrame) -> NDArray:
        """Generate predictions."""
        self._check_fitted()
        try:
            predictions = self.model.predict(X)
            return predictions
        except Exception as e:
            logger.error(f"✗ LinearRegression predict failed: {e}")
            raise

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Return feature coefficients as importance.
        Note: For linear models, higher |coefficient| = more important.
        """
        self._check_fitted()

        if self.feature_names_ is None:
            return {}

        # Use absolute values of coefficients
        importance_dict = {
            name: float(abs(coef))
            for name, coef in zip(self.feature_names_, self.model.coef_)
        }

        # Normalize to [0, 1]
        max_importance = max(importance_dict.values()) if importance_dict else 1.0
        if max_importance > 0:
            importance_dict = {k: v / max_importance for k, v in importance_dict.items()}

        return importance_dict

    def save(self, filepath: str) -> None:
        """Save model using joblib."""
        self._check_fitted()
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            joblib.dump(
                {"model": self.model, "feature_names": self.feature_names_},
                filepath,
            )
            logger.info(f"✓ LinearRegression saved to {filepath}")
        except Exception as e:
            logger.error(f"✗ Failed to save LinearRegression: {e}")
            raise IOError(f"Save failed: {e}")

    @classmethod
    def load(cls, filepath: str) -> "LinearRegressionRegressor":
        """Load model from disk."""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        try:
            data = joblib.load(filepath)
            instance = cls()
            instance.model = data["model"]
            instance.feature_names_ = data["feature_names"]
            instance.is_fitted = True
            logger.info(f"✓ LinearRegression loaded from {filepath}")
            return instance
        except Exception as e:
            logger.error(f"✗ Failed to load LinearRegression: {e}")
            raise IOError(f"Load failed: {e}")


# =====================================================
# 2. RANDOM FOREST
# =====================================================
class RandomForestRegressor(BaseRegressor):
    """
    Random Forest ensemble model with default hyperparameters.
    Highly flexible and prone to overfitting without proper tuning.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 15,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        """
        Initialize Random Forest.

        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            random_state: Random seed
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        super().__init__(name="random_forest", random_state=random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = SKRandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, X: DataFrame, y: Series, **kwargs: Any) -> None:
        """Fit random forest model."""
        if X.empty or y.empty:
            raise ValueError("Cannot fit with empty data")

        self.feature_names_ = list(X.columns)
        logger.info(
            f"Fitting RandomForest (n_estimators={self.n_estimators}, max_depth={self.max_depth}) "
            f"with {len(X)} samples"
        )

        try:
            self.model.fit(X, y)
            self.is_fitted = True
            logger.info(f"✓ RandomForest fitted. R² (train): {self.model.score(X, y):.4f}")
        except Exception as e:
            logger.error(f"✗ RandomForest fit failed: {e}")
            raise

    def predict(self, X: DataFrame) -> NDArray:
        """Generate predictions."""
        self._check_fitted()
        try:
            predictions = self.model.predict(X)
            return predictions
        except Exception as e:
            logger.error(f"✗ RandomForest predict failed: {e}")
            raise

    def get_feature_importance(self) -> Dict[str, float]:
        """Return tree-based feature importance."""
        self._check_fitted()

        if self.feature_names_ is None:
            return {}

        importance_dict = {
            name: float(importance)
            for name, importance in zip(self.feature_names_, self.model.feature_importances_)
        }

        # Already normalized by sklearn
        return importance_dict

    def save(self, filepath: str) -> None:
        """Save model using joblib."""
        self._check_fitted()
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            joblib.dump(
                {
                    "model": self.model,
                    "feature_names": self.feature_names_,
                    "n_estimators": self.n_estimators,
                    "max_depth": self.max_depth,
                },
                filepath,
            )
            logger.info(f"✓ RandomForest saved to {filepath}")
        except Exception as e:
            logger.error(f"✗ Failed to save RandomForest: {e}")
            raise IOError(f"Save failed: {e}")

    @classmethod
    def load(cls, filepath: str) -> "RandomForestRegressor":
        """Load model from disk."""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        try:
            data = joblib.load(filepath)
            instance = cls(
                n_estimators=data.get("n_estimators", 100),
                max_depth=data.get("max_depth", 15),
            )
            instance.model = data["model"]
            instance.feature_names_ = data["feature_names"]
            instance.is_fitted = True
            logger.info(f"✓ RandomForest loaded from {filepath}")
            return instance
        except Exception as e:
            logger.error(f"✗ Failed to load RandomForest: {e}")
            raise IOError(f"Load failed: {e}")


# =====================================================
# 3. XGBOOST
# =====================================================
class XGBoostRegressor(BaseRegressor):
    """
    XGBoost gradient boosting model.
    Powerful and fast with early stopping capabilities.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 7,
        learning_rate: float = 0.1,
        random_state: int = 42,
        n_jobs: Optional[int] = None,
    ) -> None:
        """
        Initialize XGBoost.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Shrinkage parameter (eta)
            random_state: Random seed
            n_jobs: Ignored (XGBoost doesn't support n_jobs parameter)
        """
        super().__init__(name="xgboost", random_state=random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = XGBRegressorModel(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            verbosity=0,
        )

    def fit(self, X: DataFrame, y: Series, **kwargs: Any) -> None:
        """Fit XGBoost model with optional early stopping."""
        if X.empty or y.empty:
            raise ValueError("Cannot fit with empty data")

        self.feature_names_ = list(X.columns)
        logger.info(
            f"Fitting XGBoost (n_estimators={self.n_estimators}, max_depth={self.max_depth}) "
            f"with {len(X)} samples"
        )

        try:
            self.model.fit(X, y, verbose=False)
            self.is_fitted = True
            logger.info(f"✓ XGBoost fitted. R² (train): {self.model.score(X, y):.4f}")
        except Exception as e:
            logger.error(f"✗ XGBoost fit failed: {e}")
            raise

    def predict(self, X: DataFrame) -> NDArray:
        """Generate predictions."""
        self._check_fitted()
        try:
            predictions = self.model.predict(X)
            return predictions
        except Exception as e:
            logger.error(f"✗ XGBoost predict failed: {e}")
            raise

    def get_feature_importance(self) -> Dict[str, float]:
        """Return XGBoost feature importance (gain-based)."""
        self._check_fitted()

        if self.feature_names_ is None:
            return {}

        importance_dict = {
            name: float(importance)
            for name, importance in zip(self.feature_names_, self.model.feature_importances_)
        }

        return importance_dict

    def save(self, filepath: str) -> None:
        """Save model using joblib."""
        self._check_fitted()
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            joblib.dump(
                {
                    "model": self.model,
                    "feature_names": self.feature_names_,
                    "n_estimators": self.n_estimators,
                    "max_depth": self.max_depth,
                    "learning_rate": self.learning_rate,
                },
                filepath,
            )
            logger.info(f"✓ XGBoost saved to {filepath}")
        except Exception as e:
            logger.error(f"✗ Failed to save XGBoost: {e}")
            raise IOError(f"Save failed: {e}")

    @classmethod
    def load(cls, filepath: str) -> "XGBoostRegressor":
        """Load model from disk."""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        try:
            data = joblib.load(filepath)
            instance = cls(
                n_estimators=data.get("n_estimators", 200),
                max_depth=data.get("max_depth", 7),
                learning_rate=data.get("learning_rate", 0.1),
            )
            instance.model = data["model"]
            instance.feature_names_ = data["feature_names"]
            instance.is_fitted = True
            logger.info(f"✓ XGBoost loaded from {filepath}")
            return instance
        except Exception as e:
            logger.error(f"✗ Failed to load XGBoost: {e}")
            raise IOError(f"Load failed: {e}")


# =====================================================
# 4. LIGHTGBM
# =====================================================
class LightGBMRegressor(BaseRegressor):
    """
    LightGBM gradient boosting model.
    Fast training with leaf-wise tree growth.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 7,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        random_state: int = 42,
    ) -> None:
        """
        Initialize LightGBM.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Shrinkage parameter
            num_leaves: Maximum number of leaves per tree
            random_state: Random seed
        """
        super().__init__(name="lightgbm", random_state=random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.model = LGBMRegressorModel(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )

    def fit(self, X: DataFrame, y: Series, **kwargs: Any) -> None:
        """Fit LightGBM model."""
        if X.empty or y.empty:
            raise ValueError("Cannot fit with empty data")

        self.feature_names_ = list(X.columns)
        logger.info(
            f"Fitting LightGBM (n_estimators={self.n_estimators}, num_leaves={self.num_leaves}) "
            f"with {len(X)} samples"
        )

        try:
            self.model.fit(X, y)
            self.is_fitted = True
            logger.info(f"✓ LightGBM fitted. R² (train): {self.model.score(X, y):.4f}")
        except Exception as e:
            logger.error(f"✗ LightGBM fit failed: {e}")
            raise

    def predict(self, X: DataFrame) -> NDArray:
        """Generate predictions."""
        self._check_fitted()
        try:
            predictions = self.model.predict(X)
            return predictions
        except Exception as e:
            logger.error(f"✗ LightGBM predict failed: {e}")
            raise

    def get_feature_importance(self) -> Dict[str, float]:
        """Return LightGBM feature importance (gain-based)."""
        self._check_fitted()

        if self.feature_names_ is None:
            return {}

        importance_dict = {
            name: float(importance)
            for name, importance in zip(self.feature_names_, self.model.feature_importances_)
        }

        return importance_dict

    def save(self, filepath: str) -> None:
        """Save model using joblib."""
        self._check_fitted()
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            joblib.dump(
                {
                    "model": self.model,
                    "feature_names": self.feature_names_,
                    "n_estimators": self.n_estimators,
                    "max_depth": self.max_depth,
                    "learning_rate": self.learning_rate,
                    "num_leaves": self.num_leaves,
                },
                filepath,
            )
            logger.info(f"✓ LightGBM saved to {filepath}")
        except Exception as e:
            logger.error(f"✗ Failed to save LightGBM: {e}")
            raise IOError(f"Save failed: {e}")

    @classmethod
    def load(cls, filepath: str) -> "LightGBMRegressor":
        """Load model from disk."""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        try:
            data = joblib.load(filepath)
            instance = cls(
                n_estimators=data.get("n_estimators", 200),
                max_depth=data.get("max_depth", 7),
                learning_rate=data.get("learning_rate", 0.1),
                num_leaves=data.get("num_leaves", 31),
            )
            instance.model = data["model"]
            instance.feature_names_ = data["feature_names"]
            instance.is_fitted = True
            logger.info(f"✓ LightGBM loaded from {filepath}")
            return instance
        except Exception as e:
            logger.error(f"✗ Failed to load LightGBM: {e}")
            raise IOError(f"Load failed: {e}")


# =====================================================
# 5. NEURAL NETWORK (MLP)
# =====================================================
class MLPNeuralNetworkRegressor(BaseRegressor):
    """
    Multi-Layer Perceptron neural network using Keras/TensorFlow.
    Architecture: Input → Dense(64) → Dropout → Dense(32) → Dropout → Output
    """

    def __init__(
        self,
        hidden_layers: List[int] = None,
        dropout_rate: float = 0.3,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
        random_state: int = 42,
    ) -> None:
        """
        Initialize Neural Network.

        Args:
            hidden_layers: List of hidden layer sizes (e.g., [64, 32])
            dropout_rate: Dropout fraction for regularization
            epochs: Training epochs
            batch_size: Training batch size
            validation_split: Fraction for validation during training
            random_state: Random seed

        Raises:
            ImportError: If TensorFlow not available
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow/Keras not available. Install tensorflow>=2.14.0")

        super().__init__(name="neural_network", random_state=random_state)
        self.hidden_layers = hidden_layers or [64, 32]
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.input_dim_: Optional[int] = None

        logger.debug(f"Initialized NeuralNetwork (layers={self.hidden_layers})")

    def _build_model(self, input_dim: int) -> keras.Model:
        """Build Keras Sequential model."""
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(input_dim,)))

        # Hidden layers with dropout
        for units in self.hidden_layers:
            model.add(keras.layers.Dense(units, activation="relu"))
            model.add(keras.layers.Dropout(self.dropout_rate))

        # Output layer
        model.add(keras.layers.Dense(1))

        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model

    def fit(self, X: DataFrame, y: Series, **kwargs: Any) -> None:
        """Fit neural network model."""
        if X.empty or y.empty:
            raise ValueError("Cannot fit with empty data")

        self.feature_names_ = list(X.columns)
        self.input_dim_ = len(X.columns)

        logger.info(
            f"Fitting Neural Network ({len(self.hidden_layers)} hidden layers) "
            f"with {len(X)} samples, {self.input_dim_} features"
        )

        try:
            # Set random seeds for reproducibility
            tf.random.set_seed(self.random_state)
            np.random.seed(self.random_state)

            self.model = self._build_model(self.input_dim_)

            history = self.model.fit(
                X.values,
                y.values,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                verbose=0,
            )

            self.is_fitted = True
            final_loss = history.history["loss"][-1]
            logger.info(f"✓ Neural Network fitted. Final training loss: {final_loss:.4f}")
        except Exception as e:
            logger.error(f"✗ Neural Network fit failed: {e}")
            raise

    def predict(self, X: DataFrame) -> NDArray:
        """Generate predictions."""
        self._check_fitted()
        try:
            predictions = self.model.predict(X.values, verbose=0)
            return predictions.flatten()
        except Exception as e:
            logger.error(f"✗ Neural Network predict failed: {e}")
            raise

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Approximate feature importance using gradient-based method.
        Compute gradients of model output w.r.t. inputs.
        """
        self._check_fitted()

        if self.feature_names_ is None or self.input_dim_ is None:
            return {}

        logger.warning(
            "Neural Network feature importance computed via gradient-based approximation"
        )

        # Create a simple gradient-based importance
        # For each feature, compute mean absolute gradient
        try:
            # Create dummy data
            dummy_data = tf.zeros((1, self.input_dim_), dtype=tf.float32)

            importance_scores = []
            for i in range(self.input_dim_):
                x_var = tf.Variable(
                    tf.ones((1, self.input_dim_), dtype=tf.float32), trainable=True
                )

                with tf.GradientTape() as tape:
                    output = self.model(x_var)

                gradients = tape.gradient(output, x_var)
                if gradients is not None:
                    grad_magnitude = float(tf.abs(gradients[0, i]))
                    importance_scores.append(grad_magnitude)
                else:
                    importance_scores.append(0.0)

            # Normalize
            max_score = max(importance_scores) if importance_scores else 1.0
            if max_score > 0:
                importance_scores = [s / max_score for s in importance_scores]

            importance_dict = {
                name: score for name, score in zip(self.feature_names_, importance_scores)
            }
            return importance_dict
        except Exception as e:
            logger.warning(f"Could not compute gradient-based importance: {e}. Returning zeros.")
            return {name: 0.0 for name in self.feature_names_}

    def save(self, filepath: str) -> None:
        """Save model using Keras .h5 format."""
        self._check_fitted()
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Replace .joblib with .h5 for neural network
            h5_path = str(path).replace(".joblib", ".h5")
            self.model.save(h5_path)

            # Save metadata separately
            metadata = {
                "feature_names": self.feature_names_,
                "hidden_layers": self.hidden_layers,
                "dropout_rate": self.dropout_rate,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "input_dim": self.input_dim_,
            }
            joblib.dump(metadata, filepath)
            logger.info(f"✓ Neural Network saved to {h5_path} and metadata to {filepath}")
        except Exception as e:
            logger.error(f"✗ Failed to save Neural Network: {e}")
            raise IOError(f"Save failed: {e}")

    @classmethod
    def load(cls, filepath: str) -> "MLPNeuralNetworkRegressor":
        """Load model from disk."""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Metadata file not found: {filepath}")

        try:
            # Load metadata
            metadata = joblib.load(filepath)

            # Load model without training compilation state for compatibility
            h5_path = str(filepath).replace(".joblib", ".h5")
            model = keras.models.load_model(h5_path, compile=False)

            instance = cls(
                hidden_layers=metadata.get("hidden_layers", [64, 32]),
                dropout_rate=metadata.get("dropout_rate", 0.3),
                epochs=metadata.get("epochs", 50),
                batch_size=metadata.get("batch_size", 32),
            )
            instance.model = model
            instance.feature_names_ = metadata["feature_names"]
            instance.input_dim_ = metadata.get("input_dim")
            instance.is_fitted = True
            logger.info(f"✓ Neural Network loaded from {h5_path}")
            return instance
        except Exception as e:
            logger.error(f"✗ Failed to load Neural Network: {e}")
            raise IOError(f"Load failed: {e}")


# =====================================================
# MODEL FACTORY
# =====================================================
class ModelFactory:
    """
    Factory for dynamically instantiating and managing regression models.
    Supports registration and dynamic model creation from config.
    """

    _model_registry: Dict[str, Type[BaseRegressor]] = {
        "linear_regression": LinearRegressionRegressor,
        "random_forest": RandomForestRegressor,
        "xgboost": XGBoostRegressor,
        "lightgbm": LightGBMRegressor,
        "neural_network": MLPNeuralNetworkRegressor,
    }

    @classmethod
    def create_model(
        cls, model_name: str, hyperparams: Optional[Dict[str, Any]] = None
    ) -> BaseRegressor:
        """
        Create a model instance by name with optional hyperparameters.

        Args:
            model_name: Model name from registry (e.g., "xgboost", "random_forest")
            hyperparams: Dictionary of hyperparameters to pass to constructor

        Returns:
            Instantiated BaseRegressor

        Raises:
            ValueError: If model_name not in registry
            TypeError: If hyperparameters invalid
        """
        if model_name not in cls._model_registry:
            available = ", ".join(cls._model_registry.keys())
            raise ValueError(
                f"Unknown model: '{model_name}'. Available models: {available}"
            )

        model_class = cls._model_registry[model_name]
        hyperparams = hyperparams or {}

        try:
            logger.info(f"Creating {model_name} with hyperparams: {hyperparams}")
            model_instance = model_class(**hyperparams)
            return model_instance
        except TypeError as e:
            logger.error(f"Invalid hyperparameters for {model_name}: {e}")
            raise

    @classmethod
    def create_from_config(cls, model_config: Dict[str, Any]) -> BaseRegressor:
        """
        Create model from a config dictionary.

        Args:
            model_config: Config dict with "name" key and optional "hyperparams"

        Returns:
            Instantiated BaseRegressor

        Raises:
            ValueError: If config invalid
        """
        model_name = model_config.get("name")
        if not model_name:
            raise ValueError("Model config must have 'name' key")

        hyperparams = model_config.get("hyperparams", {})
        return cls.create_model(model_name, hyperparams)

    @classmethod
    def list_models(cls) -> List[str]:
        """Get list of available model names."""
        return list(cls._model_registry.keys())

    @classmethod
    def register_model(cls, name: str, model_class: Type[BaseRegressor]) -> None:
        """
        Register a custom model class.

        Args:
            name: Model identifier
            model_class: Class inheriting from BaseRegressor
        """
        if not issubclass(model_class, BaseRegressor):
            raise TypeError(f"{model_class} must inherit from BaseRegressor")

        cls._model_registry[name] = model_class
        logger.info(f"✓ Registered custom model: {name}")
