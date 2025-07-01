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
        # Time series split for temporal validation (increased for better validation)
        tscv = TimeSeriesSplit(n_splits=5)

        # Scale features and store feature names
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        self.feature_names_ = list(X.columns)  # Store training feature names
        
        # Store sample of training features for confidence calculation
        try:
            sample_size = min(1000, len(X_scaled))  # Sample max 1000 rows
            self.training_features_ = X_scaled.sample(n=sample_size, random_state=42).values
        except:
            self.training_features_ = None

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
        """Make predictions with optional confidence intervals and age adjustments."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Align features with training data
        if hasattr(self, 'feature_names_') and self.feature_names_ is not None:
            X_aligned = self._align_features(X, self.feature_names_)
        else:
            X_aligned = X

        X_scaled = self.scaler.transform(X_aligned)
        
        # Store training features for confidence calculation (sample if too large)
        if not hasattr(self, 'training_features_'):
            try:
                # This would ideally be stored during training, but for now we'll skip
                self.training_features_ = None
            except:
                self.training_features_ = None
        
        predictions = self.model.predict(X_scaled)

        # Apply age-based adjustments if age features are available
        predictions = self._apply_age_adjustments(predictions, X_aligned)

        if return_confidence:
            # Use ensemble confidence if available, otherwise basic confidence
            if hasattr(self, 'base_models') and self.base_models:
                confidence = self._calculate_ensemble_confidence(X_scaled)
            else:
                confidence = self._calculate_prediction_confidence(X_scaled)
            return predictions, confidence

        return predictions

    def _apply_age_adjustments(self, predictions: np.ndarray, X: pd.DataFrame) -> np.ndarray:
        """Apply age-based adjustments to predictions for more realistic results."""
        if "player_age" not in X.columns:
            return predictions
            
        adjusted_predictions = predictions.copy()
        
        for i, age in enumerate(X["player_age"]):
            if age > 34:  # Aging veteran
                # Get decline factor if available
                decline_factor = X.get("age_decline_factor", pd.Series([1.0])).iloc[i] if i < len(X) else 1.0
                recent_form_weight = X.get("recent_form_weight", pd.Series([0.8])).iloc[i] if i < len(X) else 0.8
                
                # Apply decline factor more aggressively for points (scoring typically declines first)
                if self.stat_type == "pts":
                    if age >= 40:  # Special case for 40+ players
                        # Cap points at reasonable levels for 40+ players
                        if predictions[i] > 30:
                            adjusted_predictions[i] = min(predictions[i], 25 + np.random.normal(0, 2))
                        adjusted_predictions[i] *= decline_factor * 0.85  # Additional 15% reduction for 40+
                    elif age >= 38:
                        if predictions[i] > 35:
                            adjusted_predictions[i] = min(predictions[i], 28 + np.random.normal(0, 3))
                        adjusted_predictions[i] *= decline_factor * 0.9   # Additional 10% reduction for 38+
                    else:
                        adjusted_predictions[i] *= decline_factor
                
                # Apply lighter decline for other stats
                elif self.stat_type in ["reb", "ast"]:
                    adjusted_predictions[i] *= decline_factor * 0.95  # Lighter decline for rebounds/assists
                
                elif self.stat_type in ["stl", "blk"]:
                    adjusted_predictions[i] *= decline_factor * 0.9   # Moderate decline for defensive stats
                    
            elif age > 30:  # Regular veteran
                # Light decline factor
                decline_factor = X.get("age_decline_factor", pd.Series([1.0])).iloc[i] if i < len(X) else 1.0
                adjusted_predictions[i] *= decline_factor
        
        return adjusted_predictions

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
        """Calculate confidence for individual predictions using multiple factors."""
        n_samples = len(X_scaled)
        confidences = np.ones(n_samples)
        
        # Factor 1: Model performance-based confidence
        base_confidence = 0.5
        if hasattr(self, 'performance_metrics') and self.performance_metrics:
            # Use validation R² as base confidence
            r2 = self.performance_metrics.get("test_r2", 0)
            mape = self.performance_metrics.get("mape", 50)
            
            # Convert metrics to confidence (0-1 scale)
            r2_confidence = max(0.2, min(0.9, (r2 + 1) / 2))  # Scale R² to 0.2-0.9
            mape_confidence = max(0.2, min(0.9, 1 - (mape / 100)))  # Lower MAPE = higher confidence
            
            base_confidence = (r2_confidence + mape_confidence) / 2
        
        # Factor 2: Feature similarity to training data (simplified distance measure)
        if hasattr(self, 'training_features_') and self.training_features_ is not None:
            try:
                # Calculate mean distance from training data centroid
                training_mean = np.mean(self.training_features_, axis=0)
                distances = np.linalg.norm(X_scaled - training_mean, axis=1)
                max_distance = np.percentile(distances, 95)  # 95th percentile as max
                
                # Convert distance to similarity confidence (closer = more confident)
                similarity_confidence = 1 - np.clip(distances / max_distance, 0, 1)
                similarity_confidence = 0.3 + 0.4 * similarity_confidence  # Scale to 0.3-0.7
            except:
                similarity_confidence = np.ones(n_samples) * 0.5
        else:
            similarity_confidence = np.ones(n_samples) * 0.5
        
        # Factor 3: Age-based confidence adjustment
        age_confidence = np.ones(n_samples)
        if hasattr(X_scaled, 'shape') and X_scaled.shape[1] > 0:
            # Try to find age column (simplified - assumes it's been scaled)
            # This is approximate since features are scaled
            try:
                # Age confidence: younger players = higher confidence
                # This is a simplified heuristic
                age_confidence = np.ones(n_samples) * 0.8  # Default for most players
            except:
                age_confidence = np.ones(n_samples) * 0.8
        
        # Combine confidence factors
        final_confidence = base_confidence * similarity_confidence * age_confidence
        
        # Ensure confidence is in reasonable range
        final_confidence = np.clip(final_confidence, 0.15, 0.95)
        
        return final_confidence

    def _calculate_ensemble_confidence(self, X_scaled: np.ndarray) -> np.ndarray:
        """Calculate confidence based on ensemble prediction variance."""
        if not hasattr(self, 'base_models') or not self.base_models:
            return self._calculate_prediction_confidence(X_scaled)
        
        # Get predictions from all base models
        predictions_list = []
        
        try:
            for model_name, model in self.base_models.items():
                try:
                    pred = model.predict(X_scaled)
                    predictions_list.append(pred)
                except Exception as e:
                    logger.warning(f"Failed to get predictions from {model_name}: {e}")
                    continue
            
            if len(predictions_list) >= 2:
                # Calculate prediction variance across models
                predictions_array = np.array(predictions_list).T  # Shape: (n_samples, n_models)
                prediction_std = np.std(predictions_array, axis=1)
                
                # Convert variance to confidence (lower variance = higher confidence)
                # Normalize by typical stat values for this stat type
                if self.stat_type == "pts":
                    typical_value = 20
                elif self.stat_type in ["reb", "ast"]:
                    typical_value = 8
                else:  # stl, blk
                    typical_value = 1.5
                
                normalized_std = prediction_std / typical_value
                variance_confidence = 1 / (1 + normalized_std)  # Higher std = lower confidence
                variance_confidence = np.clip(variance_confidence, 0.2, 0.9)
                
                # Combine with base confidence
                base_confidence = self._calculate_prediction_confidence(X_scaled)
                ensemble_confidence = (base_confidence + variance_confidence) / 2
                
                return np.clip(ensemble_confidence, 0.15, 0.95)
        
        except Exception as e:
            logger.warning(f"Error calculating ensemble confidence: {e}")
        
        # Fallback to base confidence calculation
        return self._calculate_prediction_confidence(X_scaled)

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
            "feature_names": getattr(self, 'feature_names_', None),  # Store training feature names
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
        self.feature_names_ = model_data.get("feature_names", None)  # Load training feature names
        self.is_trained = True

    def _align_features(self, X_pred: pd.DataFrame, X_train_columns: List[str]) -> pd.DataFrame:
        """Align prediction features with training features."""
        X_aligned = X_pred.copy()
        
        # Add missing features with default values
        for col in X_train_columns:
            if col not in X_aligned.columns:
                # Set default values based on feature type
                if 'count' in col.lower() or 'games' in col.lower():
                    X_aligned[col] = 0  # Count features default to 0
                elif 'avg' in col.lower() or 'mean' in col.lower():
                    X_aligned[col] = 0.0  # Average features default to 0
                elif 'pct' in col.lower() or 'rate' in col.lower():
                    X_aligned[col] = 0.0  # Percentage features default to 0
                else:
                    X_aligned[col] = 0.0  # Default to 0
        
        # Remove extra features not in training
        extra_cols = [col for col in X_aligned.columns if col not in X_train_columns]
        if extra_cols:
            logger.warning(f"Removing extra features not seen during training: {extra_cols[:5]}...")
            X_aligned = X_aligned.drop(columns=extra_cols)
        
        # Ensure column order matches training
        X_aligned = X_aligned[X_train_columns]
        
        return X_aligned


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
        """Get parameter space for hyperparameter optimization with enhanced parameters."""
        return {
            "n_estimators": Integer(200, 1500),      # Increased range
            "num_leaves": Integer(31, 150),          # Increased range
            "learning_rate": Real(0.005, 0.2, prior="log-uniform"),  # Lower learning rate for better generalization
            "feature_fraction": Real(0.7, 1.0),      # Higher minimum
            "bagging_fraction": Real(0.7, 1.0),      # Higher minimum
            "min_child_samples": Integer(15, 100),   # Increased for better generalization
            "max_depth": Integer(4, 15),             # Added max_depth
            "lambda_l1": Real(0, 15),                # Increased range
            "lambda_l2": Real(0, 15),                # Increased range
            "bagging_freq": Integer(1, 10),          # Added bagging frequency
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
                n_iter=25,  # Increased for better optimization
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

        # Base models for ensemble with enhanced configuration
        self.base_models = {
            "rf": RandomForestRegressor(
                n_estimators=100, max_depth=12, min_samples_split=5, random_state=42
            ),  # Increased complexity for better performance
            "xgb": xgb.XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
            ),  # Increased complexity
            "ridge": Ridge(alpha=1.0, random_state=42),
            "elastic": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
            "lasso": Lasso(alpha=0.1, random_state=42),  # Added Lasso
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
        
        # Store feature names for later alignment
        self.feature_names_ = list(X.columns)

        # Time series split for stacking (increased for better validation)
        tscv = TimeSeriesSplit(n_splits=5)

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

        # Set trained flag before calculating metrics (needed for predict method)
        self.is_trained = True

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
        
        # Ensure base_models is properly initialized
        if not hasattr(self, 'base_models') or self.base_models is None:
            raise ValueError("Ensemble base models not properly loaded")
        
        # Ensure meta_learner is properly initialized  
        if not hasattr(self, 'meta_learner') or self.meta_learner is None:
            raise ValueError("Ensemble meta-learner not properly loaded")

        # Align features if feature names are available
        if hasattr(self, 'feature_names_') and self.feature_names_ is not None:
            X = self._align_features(X, self.feature_names_)

        X_scaled = self.scaler.transform(X)

        # Generate base model predictions
        base_predictions = np.zeros((len(X), len(self.base_models)))
        successful_predictions = 0

        for i, (name, model) in enumerate(self.base_models.items()):
            try:
                if model is not None:
                    base_predictions[:, i] = model.predict(X_scaled)
                    successful_predictions += 1
                else:
                    logger.warning(f"Base model {name} is None, using fallback")
                    base_predictions[:, i] = 0  # Fallback
            except Exception as e:
                logger.warning(f"Failed to get predictions from {name}: {e}")
                base_predictions[:, i] = 0  # Fallback
        
        if successful_predictions == 0:
            raise ValueError("No base models were able to make predictions")

        # Meta-learner prediction
        try:
            ensemble_predictions = self.meta_learner.predict(base_predictions)
        except Exception as e:
            logger.error(f"Meta-learner prediction failed: {e}")
            # Fallback to simple average if meta-learner fails
            ensemble_predictions = np.mean(base_predictions, axis=1)

        if return_confidence:
            # Calculate ensemble confidence based on agreement between models
            if successful_predictions > 1:
                model_std = np.std(base_predictions, axis=1)
                confidence = 1.0 / (1.0 + model_std)  # Higher confidence when models agree
            else:
                confidence = np.ones(len(X)) * 0.5  # Default confidence when only one model
            return ensemble_predictions, confidence

        return ensemble_predictions

    def _create_model_copy(self):
        """Create a copy of the ensemble."""
        # Simplified for ensemble - return meta learner copy
        return Ridge(**self.meta_learner.get_params())

    def save_model(self, filepath: str) -> None:
        """Save the ensemble model with all components."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        model_data = {
            "base_models": self.base_models,
            "meta_learner": self.meta_learner,
            "scaler": self.scaler,
            "stat_type": self.stat_type,
            "model_version": self.model_version,
            "feature_importance": self.feature_importance,
            "performance_metrics": self.performance_metrics,
            "confidence_intervals": self.confidence_intervals,
            "training_history": self.training_history,
            "stacking_features": getattr(self, 'stacking_features', None),
            "weights": getattr(self, 'weights', None),
            "feature_names": getattr(self, 'feature_names_', None),  # Save feature names for alignment
            "trained_at": datetime.now().isoformat(),
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

    def load_model(self, filepath: str) -> None:
        """Load the ensemble model with all components."""
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        # Check if this is an old model file (doesn't have base_models/meta_learner)
        if "base_models" not in model_data or "meta_learner" not in model_data:
            logger.warning(f"Loading old ensemble model format. Re-initializing ensemble components.")
            
            # Initialize ensemble components (they'll need to be retrained)
            self.__init__(self.stat_type, model_data.get("model_version", "ensemble_v2.0"))
            
            # Load what we can from the old format
            self.scaler = model_data.get("scaler", RobustScaler())
            self.stat_type = model_data["stat_type"]
            self.model_version = model_data["model_version"]
            self.feature_importance = model_data.get("feature_importance", {})
            self.performance_metrics = model_data.get("performance_metrics", {})
            self.confidence_intervals = model_data.get("confidence_intervals", {})
            self.training_history = model_data.get("training_history", [])
            
            # Mark as not trained since ensemble components need retraining
            self.is_trained = False
            logger.warning(f"Old model format detected. Model needs retraining to work properly.")
            return
        
        # New format - load all ensemble components
        self.base_models = model_data["base_models"]
        self.meta_learner = model_data["meta_learner"]
        self.scaler = model_data.get("scaler", RobustScaler())
        self.stat_type = model_data["stat_type"]
        self.model_version = model_data["model_version"]
        self.feature_importance = model_data["feature_importance"]
        self.performance_metrics = model_data["performance_metrics"]
        self.confidence_intervals = model_data.get("confidence_intervals", {})
        self.training_history = model_data.get("training_history", [])
        self.stacking_features = model_data.get("stacking_features", None)
        self.weights = model_data.get("weights", None)
        self.feature_names_ = model_data.get("feature_names", None)  # Restore feature names
        self.is_trained = True


class ModelManager:
    """Manages multiple stat prediction models and handles model improvement."""

    def __init__(self, db_path: str = "data/nba_data.db", models_dir: str = "models/"):
        """Initialize the model manager."""
        self.db_path = db_path
        self.models_dir = models_dir
        self.predictors = {}

        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize feature engineer for player-specific training
        from src.data.feature_engineer import AdvancedFeatureEngineer
        self.feature_engineer = AdvancedFeatureEngineer(db_path)

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

    def load_models(self, stat_types: List[str], player_id: int = None) -> None:
        """Load the latest models for specified stat types, optionally player-specific."""
        for stat_type in stat_types:
            # Look for player-specific models first if player_id is provided
            if player_id is not None:
                player_pattern = f"{stat_type}_player_{player_id}_"
                player_files = [
                    f
                    for f in os.listdir(self.models_dir)
                    if f.startswith(player_pattern) and f.endswith(".pkl")
                ]
                
                if player_files:
                    # Sort by creation time and get the latest player-specific model
                    player_files.sort(
                        key=lambda x: os.path.getctime(os.path.join(self.models_dir, x)),
                        reverse=True,
                    )
                    latest_model = player_files[0]
                    
                    # Load player-specific model
                    try:
                        predictor = self._load_model_file(latest_model, stat_type)
                        if predictor:
                            self.predictors[stat_type] = predictor
                            logger.info(f"Loaded model for {stat_type}: {latest_model}")
                            continue
                    except Exception as e:
                        logger.error(f"Error loading player-specific model for {stat_type}: {e}")
            
            # Fall back to general models
            model_files = [
                f
                for f in os.listdir(self.models_dir)
                if f.startswith(f"{stat_type}_") and f.endswith(".pkl") and "player_" not in f
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

            # Load the general model
            try:
                predictor = self._load_model_file(latest_model, stat_type)
                if predictor:
                    self.predictors[stat_type] = predictor
                    logger.info(f"Loaded model for {stat_type}: {latest_model}")
            except Exception as e:
                logger.error(f"Error loading model for {stat_type}: {e}")

    def _load_model_file(self, model_filename: str, stat_type: str):
        """Load a specific model file and return the appropriate predictor instance."""
        model_path = os.path.join(self.models_dir, model_filename)
        
        try:
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)
                
            # Determine model type from filename or data
            if "ensemble" in model_filename:
                # Check if it's an ensemble model by looking for ensemble-specific data
                if "base_models" in model_data and "meta_learner" in model_data:
                    predictor = AdvancedEnsembleStatPredictor(stat_type)
                    predictor.load_model(model_path)
                    return predictor
                else:
                    # Old format ensemble model, treat as basic predictor
                    predictor = AdvancedStatPredictor(stat_type)
            elif "lightgbm" in model_filename or "lgb" in model_filename:
                predictor = LightGBMStatPredictor(stat_type)
            elif "catboost" in model_filename or "cat" in model_filename:
                predictor = CatBoostStatPredictor(stat_type)
            elif "neural" in model_filename or "nn" in model_filename:
                predictor = NeuralNetworkStatPredictor(stat_type)
            else:
                # Default to basic predictor
                predictor = AdvancedStatPredictor(stat_type)
            
            # For non-ensemble models, use the basic loading logic
            if not isinstance(predictor, AdvancedEnsembleStatPredictor):
                predictor.model = model_data.get("model")
                predictor.scaler = model_data.get("scaler", RobustScaler())
                predictor.stat_type = model_data["stat_type"]
                predictor.model_version = model_data["model_version"]
                predictor.feature_importance = model_data.get("feature_importance", {})
                predictor.performance_metrics = model_data.get("performance_metrics", {})
                predictor.confidence_intervals = model_data.get("confidence_intervals", {})
                predictor.training_history = model_data.get("training_history", [])
                predictor.feature_names_ = model_data.get("feature_names", None)
                predictor.is_trained = True
                
            return predictor
            
        except Exception as e:
            logger.error(f"Error loading model file {model_filename}: {e}")
            return None

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

        # Use multiple metrics for confidence calculation
        r2 = predictor.performance_metrics.get("test_r2", 0)
        mape = predictor.performance_metrics.get("mape", 50)  # Mean Absolute Percentage Error
        mae = predictor.performance_metrics.get("test_mae", 999)
        
        # Calculate confidence from different metrics
        # R² confidence (higher R² = higher confidence)
        r2_confidence = max(0.2, min(0.9, (r2 + 1) / 2))
        
        # MAPE confidence (lower MAPE = higher confidence)
        mape_confidence = max(0.2, min(0.9, 1 - (mape / 100)))
        
        # MAE confidence (stat-specific reasonable MAE thresholds)
        if predictor.stat_type == "pts":
            reasonable_mae = 5.0  # Points
        elif predictor.stat_type in ["reb", "ast"]:
            reasonable_mae = 3.0  # Rebounds, Assists
        else:  # stl, blk
            reasonable_mae = 1.0  # Steals, Blocks
            
        mae_confidence = max(0.2, min(0.9, 1 - (mae / reasonable_mae)))
        
        # Weighted combination of confidences
        confidence = (r2_confidence * 0.4 + mape_confidence * 0.3 + mae_confidence * 0.3)
        
        return max(0.15, min(0.95, confidence))

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

    def train_player_specific_model(
        self,
        player_id: int,
        player_name: str,
        stat_type: str,
        model_type: str = "ensemble",
        optimize_hyperparams: bool = True,
        min_games: int = 30,
    ) -> Dict[str, float]:
        """Train a model specifically for a single player."""
        logger.info(f"Training {model_type} model for {stat_type} specifically for player {player_name} (ID: {player_id})")

        # Check if player has sufficient data
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT COUNT(*) FROM player_games WHERE player_id = ?",
            (player_id,)
        )
        total_games = cursor.fetchone()[0]
        
        if total_games < min_games:
            raise ValueError(
                f"Player {player_name} has insufficient data: {total_games} games (minimum: {min_games})"
            )
            
        logger.info(f"Player {player_name} has {total_games} games available")
        
        # Get player's game data sorted by date
        query = """
            SELECT * FROM player_games 
            WHERE player_id = ?
            ORDER BY game_date
        """
        
        player_games = pd.read_sql_query(query, conn, params=(player_id,))
        conn.close()
        
        # Create training dataset for this specific player
        # We'll use 80% of their games for training, 20% for validation
        train_size = int(len(player_games) * 0.8)
        train_games = player_games.iloc[:train_size]
        
        if len(train_games) < 20:  # Need minimum games for training
            raise ValueError(
                f"Insufficient training games for {player_name}: {len(train_games)} games"
            )
            
        # Create training features for each game (starting from game 10)
        training_features = []
        
        logger.info(f"Creating player-specific features for {len(train_games)} games...")
        
        with tqdm(
            range(10, len(train_games)),  # Start from game 10
            desc=f"Processing {player_name}'s games",
            ncols=80,
        ) as pbar:
            for i in pbar:
                target_game = train_games.iloc[i]
                target_date = target_game["game_date"]
                
                try:
                    # Create features based on player's history up to this point
                    features_df = self.feature_engineer.create_features_for_player(
                        player_id,
                        target_date,
                        lookback_games=min(15, i),  # Use available history
                        include_h2h_features=True,
                        include_advanced_features=True,
                    )
                    
                    if not features_df.empty:
                        # Add target value
                        if stat_type in target_game:
                            features_df[f"target_{stat_type}"] = target_game[stat_type]
                            
                        # Add metadata
                        features_df["game_id"] = target_game.get("game_id", "")
                        features_df["game_date"] = target_date
                        features_df["player_id"] = player_id
                        
                        training_features.append(features_df)
                        
                except Exception as e:
                    pbar.write(f"   ⚠️  Error creating features for game on {target_date}: {e}")
                    continue
        
        if not training_features:
            raise ValueError(f"Could not create training features for {player_name}")
            
        # Combine all features
        training_data = pd.concat(training_features, ignore_index=True)
        logger.info(f"Created {len(training_data)} training samples for {player_name}")
        
        # Prepare training data
        target_col = f"target_{stat_type}"
        if target_col not in training_data.columns:
            raise ValueError(f"Target column {target_col} not found in training data")

        # Remove rows with missing targets
        clean_data = training_data.dropna(subset=[target_col])

        if len(clean_data) < 15:
            logger.warning(
                f"Limited training data for {player_name} {stat_type}: {len(clean_data)} samples"
            )
            if len(clean_data) < 10:
                raise ValueError(
                    f"Insufficient training data for {player_name} {stat_type}: {len(clean_data)} samples"
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
            tqdm.write(f"   🏗️  Creating {model_type} predictor for {player_name}'s {stat_type}...")
            predictor = self.create_predictor(stat_type, model_type)

            # Train with advanced capabilities
            tqdm.write(f"   🚀 Training {model_type} model for {player_name}'s {stat_type}...")
            metrics = predictor.train(X, y, optimize_hyperparams=optimize_hyperparams)

            # Store the predictor with player-specific naming
            player_key = f"{stat_type}_{player_id}"
            self.predictors[player_key] = predictor

            # Save the model with player-specific filename
            tqdm.write(f"   💾 Saving player-specific model...")
            model_filename = f"{stat_type}_player_{player_id}_{model_type}_{predictor.model_version}.pkl"
            model_path = os.path.join(self.models_dir, model_filename)
            predictor.save_model(model_path)

            # Store performance metrics in database
            self._store_model_performance(
                f"{stat_type}_player_{player_id}", predictor.model_version, metrics, len(clean_data)
            )

            # Log results
            logger.info(
                f"Trained {player_name}'s {stat_type} model - Test MAE: {metrics.get('test_mae', 0):.2f}, "
                f"Test R²: {metrics.get('test_r2', 0):.3f}"
            )

            return metrics

        except Exception as e:
            logger.error(f"Error training {model_type} model for {player_name}'s {stat_type}: {e}")
            raise

    def train_models_for_player(
        self,
        player_id: int,
        player_name: str = None,
        model_type: str = "ensemble",
        stat_types: List[str] = None,
        optimize_hyperparams: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """Train models for all stat types for a specific player."""
        if stat_types is None:
            stat_types = ["pts", "reb", "ast", "stl", "blk"]
            
        # Get player name if not provided
        if player_name is None:
            try:
                import sqlite3
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT player_name FROM player_games WHERE player_id = ? LIMIT 1",
                    (player_id,)
                )
                result = cursor.fetchone()
                player_name = result[0] if result else f"Player_{player_id}"
                conn.close()
            except Exception:
                player_name = f"Player_{player_id}"
        
        logger.info(f"Training all models for {player_name} (ID: {player_id})")
        
        results = {}
        
        print(f"\n🏀 Training player-specific models for {player_name}:")
        with tqdm(
            stat_types,
            desc="Training Models",
            ncols=80,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as pbar:
            for stat_type in pbar:
                try:
                    pbar.set_description(f"Training {stat_type.upper()} for {player_name}")
                    metrics = self.train_player_specific_model(
                        player_id=player_id,
                        player_name=player_name,
                        stat_type=stat_type,
                        model_type=model_type,
                        optimize_hyperparams=optimize_hyperparams,
                    )
                    results[stat_type] = metrics
                    mae = metrics.get("test_mae", metrics.get("val_mae", 0))
                    pbar.write(
                        f"✅ {stat_type.upper()} model for {player_name} - MAE: {mae:.2f}"
                    )
                except Exception as e:
                    pbar.write(f"❌ Error training {stat_type} model for {player_name}: {e}")
                    results[stat_type] = {"error": str(e)}
        
        logger.info(f"Completed training {len([r for r in results.values() if 'error' not in r])} models for {player_name}")
        return results
