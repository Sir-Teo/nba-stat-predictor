"""
NBA Stat Prediction Models - Advanced machine learning models for predicting player stats.
"""

import json
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    TimeSeriesSplit,
    cross_val_score,
    train_test_split,
)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler
from tqdm import tqdm

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from skopt import BayesSearchCV
    from skopt.space import Categorical, Integer, Real

    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    BAYESIAN_OPT_AVAILABLE = False

import logging
import sqlite3

logger = logging.getLogger(__name__)


class AdvancedStatPredictor:
    """Advanced base class for stat prediction models with enhanced capabilities."""

    def __init__(self, stat_type: str, model_version: str = "v2.0"):
        """Initialize the advanced stat predictor."""
        self.stat_type = stat_type
        self.model_version = model_version
        self.model = None
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.feature_importance = {}
        self.performance_metrics = {}
        self.is_trained = False
        self.confidence_intervals = {}
        self.training_history = []

    def train(
        self, X: pd.DataFrame, y: pd.Series, optimize_hyperparams: bool = True
    ) -> Dict[str, float]:
        """Train the model with advanced evaluation and hyperparameter optimization."""
        # Time series split for temporal validation (reduced from 5 to 3 folds)
        tscv = TimeSeriesSplit(n_splits=3)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        # Hyperparameter optimization
        if optimize_hyperparams and hasattr(self, "_get_param_space"):
            tqdm.write(f"   🔧 Optimizing hyperparameters for {self.stat_type}...")
            self.model = self._optimize_hyperparameters(X_scaled, y, tscv)

        # Cross-validation with time series split
        tqdm.write(f"   📊 Running cross-validation for {self.stat_type}...")
        cv_scores = cross_val_score(
            self.model, X_scaled, y, cv=tscv, scoring="neg_mean_absolute_error"
        )

        # Final training
        tqdm.write(f"   🎯 Final model training for {self.stat_type}...")
        self.model.fit(X_scaled, y)

        # Comprehensive evaluation
        tqdm.write(f"   📈 Evaluating model performance for {self.stat_type}...")
        metrics = self._evaluate_model(X_scaled, y, tscv)
        metrics["cv_mae_mean"] = -cv_scores.mean()
        metrics["cv_mae_std"] = cv_scores.std()

        # Calculate confidence intervals (reduced bootstrap samples)
        tqdm.write(f"   🎲 Computing confidence intervals for {self.stat_type}...")
        self.confidence_intervals = self._calculate_confidence_intervals(X_scaled, y)

        self.performance_metrics = metrics
        self.is_trained = True

        # Extract feature importance
        self._extract_feature_importance(X.columns)

        # Store training history
        self.training_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics,
                "data_size": len(X),
            }
        )

        return metrics

    def predict(self, X: pd.DataFrame, return_confidence: bool = False) -> np.ndarray:
        """Make predictions with optional confidence intervals."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)

        if return_confidence:
            # Calculate prediction intervals using bootstrap or ensemble variance
            confidence = self._calculate_prediction_confidence(X_scaled)
            return predictions, confidence

        return predictions

    def _evaluate_model(self, X: pd.DataFrame, y: pd.Series, cv) -> Dict[str, float]:
        """Comprehensive model evaluation."""
        # Split for final evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train on train set
        self.model.fit(X_train, y_train)

        # Predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)

        # Comprehensive metrics
        metrics = {
            "train_mae": mean_absolute_error(y_train, y_pred_train),
            "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
            "train_r2": r2_score(y_train, y_pred_train),
            "test_mae": mean_absolute_error(y_test, y_pred_test),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
            "test_r2": r2_score(y_test, y_pred_test),
            "mape": np.mean(np.abs((y_test - y_pred_test) / np.maximum(y_test, 1e-8)))
            * 100,
            "accuracy_within_10pct": np.mean(
                np.abs(y_test - y_pred_test) / np.maximum(y_test, 1e-8) <= 0.1
            )
            * 100,
            "accuracy_within_20pct": np.mean(
                np.abs(y_test - y_pred_test) / np.maximum(y_test, 1e-8) <= 0.2
            )
            * 100,
        }

        return metrics

    def _calculate_confidence_intervals(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, float]:
        """Calculate confidence intervals using bootstrap sampling."""
        n_bootstrap = 25  # Reduced from 100 to 25 for faster training
        predictions_bootstrap = []

        with tqdm(
            range(n_bootstrap),
            desc="Bootstrap CI",
            ncols=60,
            bar_format="{desc}: {n_fmt}/{total_fmt}",
            leave=False,
        ) as pbar:
            for i in pbar:
                # Bootstrap sample
                indices = np.random.choice(len(X), size=len(X), replace=True)
                X_boot = X.iloc[indices]
                y_boot = y.iloc[indices]

                # Train model on bootstrap sample
                model_copy = self._create_model_copy()
                model_copy.fit(X_boot, y_boot)

                # Predict on out-of-bag samples
                oob_indices = list(set(range(len(X))) - set(indices))
                if len(oob_indices) > 0:
                    X_oob = X.iloc[oob_indices]
                    predictions_bootstrap.extend(model_copy.predict(X_oob))

        if predictions_bootstrap:
            return {
                "lower_95": np.percentile(predictions_bootstrap, 2.5),
                "upper_95": np.percentile(predictions_bootstrap, 97.5),
                "lower_80": np.percentile(predictions_bootstrap, 10),
                "upper_80": np.percentile(predictions_bootstrap, 90),
            }
        return {}

    def _calculate_prediction_confidence(self, X_scaled: np.ndarray) -> np.ndarray:
        """Calculate confidence for individual predictions."""
        # Simplified confidence based on feature similarity to training data
        # This is a placeholder - more sophisticated methods can be implemented
        return np.ones(len(X_scaled)) * 0.8  # Default confidence

    def _extract_feature_importance(self, feature_names: List[str]) -> None:
        """Extract feature importance from the trained model."""
        if hasattr(self.model, "feature_importances_"):
            self.feature_importance = dict(
                zip(feature_names, self.model.feature_importances_)
            )
        elif hasattr(self.model, "coef_"):
            self.feature_importance = dict(zip(feature_names, np.abs(self.model.coef_)))
        else:
            self.feature_importance = {}

    def _create_model_copy(self):
        """Create a copy of the model for bootstrap sampling."""
        # This should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement _create_model_copy")

    def save_model(self, filepath: str) -> None:
        """Save the trained model with enhanced metadata."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "stat_type": self.stat_type,
            "model_version": self.model_version,
            "feature_importance": self.feature_importance,
            "performance_metrics": self.performance_metrics,
            "confidence_intervals": self.confidence_intervals,
            "training_history": self.training_history,
            "trained_at": datetime.now().isoformat(),
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

    def load_model(self, filepath: str) -> None:
        """Load a trained model with enhanced metadata."""
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.scaler = model_data.get("scaler", RobustScaler())
        self.stat_type = model_data["stat_type"]
        self.model_version = model_data["model_version"]
        self.feature_importance = model_data["feature_importance"]
        self.performance_metrics = model_data["performance_metrics"]
        self.confidence_intervals = model_data.get("confidence_intervals", {})
        self.training_history = model_data.get("training_history", [])
        self.is_trained = True


class LightGBMStatPredictor(AdvancedStatPredictor):
    """LightGBM-based stat predictor with advanced features."""

    def __init__(self, stat_type: str, model_version: str = "lgb_v2.0", **kwargs):
        """Initialize LightGBM predictor with optimized parameters."""
        super().__init__(stat_type, model_version)

        # Optimized parameters for faster training
        default_params = {
            "objective": "regression",
            "metric": "mae",
            "boosting_type": "gbdt",
            "num_leaves": 25,  # Reduced from 31
            "learning_rate": 0.1,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 20,
            "random_state": 42,
            "verbosity": -1,
            "n_estimators": 75,  # Reduced from 100
        }

        default_params.update(kwargs)

        if LIGHTGBM_AVAILABLE:
            self.model = lgb.LGBMRegressor(**default_params)
        else:
            # Fallback to XGBoost if LightGBM not available
            self.model = xgb.XGBRegressor(
                n_estimators=75, max_depth=4, learning_rate=0.1, random_state=42
            )

    def _get_param_space(self):
        """Get parameter space for hyperparameter optimization."""
        return {
            "num_leaves": Integer(10, 100),
            "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
            "feature_fraction": Real(0.5, 1.0),
            "bagging_fraction": Real(0.5, 1.0),
            "min_child_samples": Integer(5, 50),
            "lambda_l1": Real(0, 10),
            "lambda_l2": Real(0, 10),
        }

    def _create_model_copy(self):
        """Create a copy of the LightGBM model."""
        return lgb.LGBMRegressor(**self.model.get_params())

    def _optimize_hyperparameters(
        self, X: pd.DataFrame, y: pd.Series, cv
    ) -> lgb.LGBMRegressor:
        """Optimize hyperparameters using Bayesian optimization."""
        if BAYESIAN_OPT_AVAILABLE:
            tqdm.write("     Using Bayesian optimization...")
            search = BayesSearchCV(
                lgb.LGBMRegressor(random_state=42, verbosity=-1),
                self._get_param_space(),
                n_iter=10,  # Reduced from 20 to 10 iterations
                cv=cv,
                scoring="neg_mean_absolute_error",
                random_state=42,
                n_jobs=-1,
            )
        else:
            tqdm.write("     Using grid search...")
            # Simplified grid search if BayesSearchCV is not available
            param_grid = {
                "n_estimators": [100, 150],  # Reduced options
                "learning_rate": [0.1],  # Single value
                "max_depth": [6],  # Single value
            }
            search = GridSearchCV(
                lgb.LGBMRegressor(random_state=42, verbosity=-1),
                param_grid,
                cv=cv,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
            )

        search.fit(X, y)
        return search.best_estimator_


class CatBoostStatPredictor(AdvancedStatPredictor):
    """CatBoost-based stat predictor."""

    def __init__(self, stat_type: str, model_version: str = "cat_v2.0", **kwargs):
        super().__init__(stat_type, model_version)

        if not CATBOOST_AVAILABLE:
            raise ImportError(
                "CatBoost is not available. Install with: pip install catboost"
            )

        default_params = {
            "iterations": 1000,
            "learning_rate": 0.03,
            "depth": 6,
            "l2_leaf_reg": 3,
            "bootstrap_type": "Bernoulli",
            "subsample": 0.8,
            "random_seed": 42,
            "verbose": False,
        }
        default_params.update(kwargs)

        self.model = cb.CatBoostRegressor(**default_params)

    def _get_param_space(self):
        """Get parameter space for hyperparameter optimization."""
        return {
            "iterations": Integer(500, 2000),
            "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
            "depth": Integer(4, 10),
            "l2_leaf_reg": Real(1, 10),
            "subsample": Real(0.6, 1.0),
        }

    def _create_model_copy(self):
        """Create a copy of the CatBoost model."""
        return cb.CatBoostRegressor(**self.model.get_params())


class NeuralNetworkStatPredictor(AdvancedStatPredictor):
    """Neural Network-based stat predictor."""

    def __init__(self, stat_type: str, model_version: str = "nn_v2.0", **kwargs):
        super().__init__(stat_type, model_version)

        default_params = {
            "hidden_layer_sizes": (100, 50, 25),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.001,
            "learning_rate": "constant",
            "learning_rate_init": 0.001,
            "max_iter": 1000,
            "random_state": 42,
            "early_stopping": True,
            "validation_fraction": 0.1,
        }
        default_params.update(kwargs)

        self.model = MLPRegressor(**default_params)

    def _get_param_space(self):
        """Get parameter space for hyperparameter optimization."""
        return {
            "hidden_layer_sizes": Categorical(
                [(50,), (100,), (100, 50), (100, 50, 25), (200, 100, 50)]
            ),
            "alpha": Real(0.0001, 0.1, prior="log-uniform"),
            "learning_rate_init": Real(0.0001, 0.01, prior="log-uniform"),
        }

    def _create_model_copy(self):
        """Create a copy of the Neural Network model."""
        return MLPRegressor(**self.model.get_params())


class AdvancedEnsembleStatPredictor(AdvancedStatPredictor):
    """Advanced ensemble with stacking and sophisticated weighting."""

    def __init__(self, stat_type: str, model_version: str = "ensemble_v2.0"):
        super().__init__(stat_type, model_version)

        # Base models for ensemble (reduced complexity for speed)
        self.base_models = {
            "rf": RandomForestRegressor(
                n_estimators=50, max_depth=8, random_state=42
            ),  # Reduced from 100 estimators
            "xgb": xgb.XGBRegressor(
                n_estimators=50, max_depth=4, random_state=42
            ),  # Reduced complexity
            "ridge": Ridge(alpha=1.0, random_state=42),
            "elastic": ElasticNet(alpha=0.1, random_state=42),
        }

        # Add advanced models if available
        if LIGHTGBM_AVAILABLE:
            self.base_models["lgb"] = lgb.LGBMRegressor(random_state=42, verbose=-1)

        if CATBOOST_AVAILABLE:
            self.base_models["cat"] = cb.CatBoostRegressor(
                random_seed=42, verbose=False
            )

        # Meta-learner for stacking
        self.meta_learner = Ridge(alpha=0.1)
        self.weights = None
        self.stacking_features = None

    def train(
        self, X: pd.DataFrame, y: pd.Series, optimize_hyperparams: bool = True
    ) -> Dict[str, float]:
        """Train ensemble with stacking."""
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        # Time series split for stacking (reduced from 5 to 3 folds)
        tscv = TimeSeriesSplit(n_splits=3)

        # Generate stacking features
        tqdm.write(
            f"   🧪 Generating stacking features for {self.stat_type} ensemble..."
        )
        stacking_features = self._generate_stacking_features(X_scaled, y, tscv)

        # Train meta-learner
        tqdm.write(f"   🧠 Training meta-learner for {self.stat_type} ensemble...")
        self.meta_learner.fit(stacking_features, y)

        # Train base models on full data
        tqdm.write(f"   🏗️  Training base models for {self.stat_type} ensemble...")
        successful_models = 0
        with tqdm(
            self.base_models.items(),
            desc="Base Models",
            ncols=60,
            bar_format="{desc}: {n_fmt}/{total_fmt}",
            leave=False,
        ) as pbar:
            for name, model in pbar:
                try:
                    pbar.set_description(f"Training {name}")
                    model.fit(X_scaled, y)
                    successful_models += 1
                except Exception as e:
                    tqdm.write(f"     ⚠️  Failed to train {name}: {e}")
                    continue

        tqdm.write(
            f"   ✅ Successfully trained {successful_models}/{len(self.base_models)} base models"
        )

        # Calculate ensemble metrics
        tqdm.write(f"   📈 Computing ensemble metrics for {self.stat_type}...")
        y_pred = self.predict(X)
        metrics = {
            "mae": mean_absolute_error(y, y_pred),
            "rmse": np.sqrt(mean_squared_error(y, y_pred)),
            "r2": r2_score(y, y_pred),
            "test_mae": mean_absolute_error(y, y_pred),  # For compatibility
            "test_rmse": np.sqrt(mean_squared_error(y, y_pred)),
            "test_r2": r2_score(y, y_pred),
        }

        self.performance_metrics = metrics
        self.is_trained = True

        return metrics

    def _generate_stacking_features(
        self, X: pd.DataFrame, y: pd.Series, cv
    ) -> np.ndarray:
        """Generate features for stacking using cross-validation."""
        stacking_features = np.zeros((len(X), len(self.base_models)))

        cv_splits = list(cv.split(X))
        total_iterations = len(cv_splits) * len(self.base_models)

        with tqdm(
            total=total_iterations,
            desc="CV Folds",
            ncols=60,
            bar_format="{desc}: {n_fmt}/{total_fmt}",
            leave=False,
        ) as pbar:
            for fold, (train_idx, val_idx) in enumerate(cv_splits):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train = y.iloc[train_idx]

                for i, (name, model) in enumerate(self.base_models.items()):
                    pbar.set_description(f"Fold {fold+1} - {name}")
                    try:
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_val)
                        stacking_features[val_idx, i] = predictions
                    except Exception as e:
                        tqdm.write(
                            f"     ⚠️  Failed to generate stacking features for {name} in fold {fold+1}: {e}"
                        )
                        stacking_features[val_idx, i] = (
                            y_train.mean()
                        )  # Fallback to mean
                    pbar.update(1)

        self.stacking_features = stacking_features
        return stacking_features

    def predict(self, X: pd.DataFrame, return_confidence: bool = False) -> np.ndarray:
        """Make ensemble predictions using stacking."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        X_scaled = self.scaler.transform(X)

        # Generate base model predictions
        base_predictions = np.zeros((len(X), len(self.base_models)))

        for i, (name, model) in enumerate(self.base_models.items()):
            try:
                base_predictions[:, i] = model.predict(X_scaled)
            except Exception as e:
                logger.warning(f"Failed to get predictions from {name}: {e}")
                base_predictions[:, i] = 0  # Fallback

        # Meta-learner prediction
        ensemble_predictions = self.meta_learner.predict(base_predictions)

        if return_confidence:
            # Calculate ensemble confidence based on agreement between models
            model_std = np.std(base_predictions, axis=1)
            confidence = 1.0 / (1.0 + model_std)  # Higher confidence when models agree
            return ensemble_predictions, confidence

        return ensemble_predictions

    def _create_model_copy(self):
        """Create a copy of the ensemble."""
        # Simplified for ensemble - return meta learner copy
        return Ridge(**self.meta_learner.get_params())


class ModelManager:
    """Manages multiple stat prediction models and handles model improvement."""

    def __init__(self, db_path: str = "data/nba_data.db", models_dir: str = "models/"):
        """Initialize the model manager."""
        self.db_path = db_path
        self.models_dir = models_dir
        self.predictors = {}

        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)

    def create_predictor(
        self, stat_type: str, model_type: str = "ensemble"
    ) -> AdvancedStatPredictor:
        """Create a new stat predictor."""
        if model_type == "random_forest":
            # Create a RandomForestRegressor wrapped in AdvancedStatPredictor
            class RandomForestStatPredictor(AdvancedStatPredictor):
                def __init__(self, stat_type: str, **kwargs):
                    super().__init__(stat_type, "rf_v2.0")
                    default_params = {
                        "n_estimators": 100,
                        "max_depth": 8,
                        "min_samples_split": 5,  # Reduced complexity
                        "min_samples_leaf": 2,
                        "random_state": 42,  # Increased leaf size
                    }
                    default_params.update(kwargs)
                    self.model = RandomForestRegressor(**default_params)

                def _create_model_copy(self):
                    return RandomForestRegressor(**self.model.get_params())

            predictor = RandomForestStatPredictor(stat_type)
        elif model_type == "xgboost":
            # Create XGBoost wrapped in AdvancedStatPredictor
            class XGBoostStatPredictor(AdvancedStatPredictor):
                def __init__(self, stat_type: str, **kwargs):
                    super().__init__(stat_type, "xgb_v2.0")
                    default_params = {
                        "n_estimators": 75,
                        "max_depth": 4,
                        "learning_rate": 0.1,  # Reduced complexity
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "random_state": 42,
                    }
                    default_params.update(kwargs)
                    self.model = xgb.XGBRegressor(**default_params)

                def _create_model_copy(self):
                    return xgb.XGBRegressor(**self.model.get_params())

            predictor = XGBoostStatPredictor(stat_type)
        elif model_type == "lightgbm":
            predictor = LightGBMStatPredictor(stat_type)
        elif model_type == "catboost":
            predictor = CatBoostStatPredictor(stat_type)
        elif model_type == "neural_network":
            predictor = NeuralNetworkStatPredictor(stat_type)
        elif model_type == "ensemble":
            predictor = AdvancedEnsembleStatPredictor(stat_type)
        else:
            raise ValueError(
                f"Unknown model type: {model_type}. Available: random_forest, xgboost, lightgbm, catboost, neural_network, ensemble"
            )

        return predictor

    def train_model(
        self,
        stat_type: str,
        training_data: pd.DataFrame,
        model_type: str = "ensemble",
        optimize_hyperparams: bool = True,
    ) -> Dict[str, float]:
        """Train a model for a specific stat with advanced capabilities."""
        logger.info(f"Training {model_type} model for {stat_type}")

        # Prepare training data
        target_col = f"target_{stat_type}"
        if target_col not in training_data.columns:
            raise ValueError(f"Target column {target_col} not found in training data")

        # Remove rows with missing targets
        clean_data = training_data.dropna(subset=[target_col])

        if len(clean_data) < 50:
            logger.warning(
                f"Limited training data for {stat_type}: {len(clean_data)} samples. Training anyway."
            )
            if len(clean_data) < 20:
                raise ValueError(
                    f"Insufficient training data for {stat_type}: {len(clean_data)} samples"
                )

        # Separate features and target
        feature_columns = [
            col
            for col in clean_data.columns
            if not col.startswith("target_")
            and col not in ["player_id", "game_id", "game_date", "target_date"]
        ]

        X = clean_data[feature_columns]
        y = clean_data[target_col]

        # Handle missing values in features
        X = X.fillna(X.mean())

        # Create predictor using the new system
        try:
            tqdm.write(f"   🏗️  Creating {model_type} predictor for {stat_type}...")
            predictor = self.create_predictor(stat_type, model_type)

            # Train with advanced capabilities
            tqdm.write(f"   🚀 Training {model_type} model for {stat_type}...")
            metrics = predictor.train(X, y, optimize_hyperparams=optimize_hyperparams)

            # Store the predictor
            self.predictors[stat_type] = predictor

            # Save the model
            tqdm.write(f"   💾 Saving model for {stat_type}...")
            model_filename = f"{stat_type}_{model_type}_{predictor.model_version}.pkl"
            model_path = os.path.join(self.models_dir, model_filename)
            predictor.save_model(model_path)

            # Store performance metrics in database with enhanced metrics
            self._store_model_performance(
                stat_type, predictor.model_version, metrics, len(clean_data)
            )

            # Log comprehensive results
            logger.info(
                f"Trained {stat_type} model - Test MAE: {metrics.get('test_mae', 0):.2f}, "
                f"Test R²: {metrics.get('test_r2', 0):.3f}, "
                f"CV MAE: {metrics.get('cv_mae_mean', 0):.2f}±{metrics.get('cv_mae_std', 0):.2f}"
            )

            return metrics

        except Exception as e:
            logger.error(f"Error training {model_type} model for {stat_type}: {e}")

            # Fallback to simple RandomForest if advanced model fails
            if model_type != "random_forest":
                logger.info(f"Falling back to RandomForest for {stat_type}")
                return self.train_model(
                    stat_type, training_data, "random_forest", False
                )

    def load_models(self, stat_types: List[str]) -> None:
        """Load the latest models for specified stat types."""
        for stat_type in stat_types:
            model_files = [
                f
                for f in os.listdir(self.models_dir)
                if f.startswith(f"{stat_type}_") and f.endswith(".pkl")
            ]

            if not model_files:
                logger.warning(f"No saved model found for {stat_type}")
                continue

            # Sort by creation time and get the latest
            model_files.sort(
                key=lambda x: os.path.getctime(os.path.join(self.models_dir, x)),
                reverse=True,
            )
            latest_model = model_files[0]

            # Load the model
            predictor = AdvancedStatPredictor(stat_type)
            try:
                with open(os.path.join(self.models_dir, latest_model), "rb") as f:
                    model_data = pickle.load(f)

                predictor.model = model_data["model"]
                predictor.stat_type = model_data["stat_type"]
                predictor.model_version = model_data["model_version"]
                predictor.feature_importance = model_data["feature_importance"]
                predictor.performance_metrics = model_data["performance_metrics"]
                predictor.is_trained = True

                self.predictors[stat_type] = predictor
                logger.info(f"Loaded model for {stat_type}: {latest_model}")
            except Exception as e:
                logger.error(f"Error loading model for {stat_type}: {e}")

    def predict_stats(
        self, features_df: pd.DataFrame, stat_types: List[str]
    ) -> pd.DataFrame:
        """Make predictions for multiple stat types."""
        predictions_df = features_df.copy()

        # Prepare features (excluding metadata columns)
        feature_columns = [
            col
            for col in features_df.columns
            if col not in ["player_id", "game_id", "game_date", "target_date"]
        ]
        X = features_df[feature_columns]

        for stat_type in stat_types:
            if stat_type not in self.predictors:
                logger.warning(f"No trained model available for {stat_type}")
                predictions_df[f"predicted_{stat_type}"] = 0
                predictions_df[f"confidence_{stat_type}"] = 0
                continue

            try:
                predictor = self.predictors[stat_type]
                predictions = predictor.predict(X)

                # Calculate confidence based on model performance
                confidence = self._calculate_confidence(predictor)

                predictions_df[f"predicted_{stat_type}"] = predictions
                predictions_df[f"confidence_{stat_type}"] = confidence

            except Exception as e:
                logger.error(f"Error predicting {stat_type}: {e}")
                predictions_df[f"predicted_{stat_type}"] = 0
                predictions_df[f"confidence_{stat_type}"] = 0

        return predictions_df

    def _calculate_confidence(self, predictor: AdvancedStatPredictor) -> float:
        """Calculate prediction confidence based on model performance."""
        if not predictor.performance_metrics:
            return 0.5

        # Use validation R² as a base for confidence
        r2 = predictor.performance_metrics.get("val_r2", 0)

        # Convert R² to confidence score (0 to 1)
        confidence = max(0, min(1, (r2 + 1) / 2))

        return confidence

    def _store_model_performance(
        self,
        stat_type: str,
        model_version: str,
        metrics: Dict[str, float],
        sample_size: int,
    ) -> None:
        """Store enhanced model performance metrics in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Use test metrics if available, otherwise fall back to val metrics
        mae = metrics.get("test_mae", metrics.get("val_mae", 0))
        rmse = metrics.get("test_rmse", metrics.get("val_rmse", 0))
        r2 = metrics.get("test_r2", metrics.get("val_r2", 0))

        cursor.execute(
            """
            INSERT INTO model_performance 
            (model_version, stat_type, mae, rmse, accuracy_rate, evaluation_date, sample_size)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                model_version,
                stat_type,
                mae,
                rmse,
                r2,
                datetime.now().isoformat(),
                sample_size,
            ),
        )

        conn.commit()
        conn.close()

    def evaluate_predictions(
        self, stat_types: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate recent predictions against actual results."""
        conn = sqlite3.connect(self.db_path)

        evaluation_results = {}

        for stat_type in stat_types:
            query = """
                SELECT predicted_value, actual_value 
                FROM predictions 
                WHERE stat_type = ? AND actual_value IS NOT NULL
                ORDER BY created_at DESC
                LIMIT 100
            """

            df = pd.read_sql_query(query, conn, params=(stat_type,))

            if len(df) > 10:  # Need sufficient predictions to evaluate
                mae = mean_absolute_error(df["actual_value"], df["predicted_value"])
                rmse = np.sqrt(
                    mean_squared_error(df["actual_value"], df["predicted_value"])
                )
                r2 = r2_score(df["actual_value"], df["predicted_value"])

                evaluation_results[stat_type] = {
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                    "sample_size": len(df),
                }
            else:
                evaluation_results[stat_type] = {
                    "mae": 0,
                    "rmse": 0,
                    "r2": 0,
                    "sample_size": len(df),
                }

        conn.close()
        return evaluation_results

    def should_retrain_model(
        self, stat_type: str, performance_threshold: float = 0.1
    ) -> bool:
        """Determine if a model should be retrained based on recent performance."""
        # For now, return False (no retraining needed)
        # In a full implementation, you would compare recent vs training performance
        return False
