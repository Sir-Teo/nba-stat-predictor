"""
NBA Stat Prediction Models - Machine learning models for predicting player stats.
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import sqlite3
import logging

logger = logging.getLogger(__name__)


class StatPredictor:
    """Base class for stat prediction models."""
    
    def __init__(self, stat_type: str, model_version: str = "v1.0"):
        """Initialize the stat predictor."""
        self.stat_type = stat_type
        self.model_version = model_version
        self.model = None
        self.feature_importance = {}
        self.performance_metrics = {}
        self.is_trained = False
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Train the model and return performance metrics."""
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_val = self.model.predict(X_val)
        
        # Calculate metrics
        metrics = {
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'train_r2': r2_score(y_train, y_pred_train),
            'val_mae': mean_absolute_error(y_val, y_pred_val),
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_pred_val)),
            'val_r2': r2_score(y_val, y_pred_val)
        }
        
        self.performance_metrics = metrics
        self.is_trained = True
        
        # Extract feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(X.columns, self.model.feature_importances_))
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'stat_type': self.stat_type,
            'model_version': self.model_version,
            'feature_importance': self.feature_importance,
            'performance_metrics': self.performance_metrics,
            'trained_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.stat_type = model_data['stat_type']
        self.model_version = model_data['model_version']
        self.feature_importance = model_data['feature_importance']
        self.performance_metrics = model_data['performance_metrics']
        self.is_trained = True


class RandomForestStatPredictor(StatPredictor):
    """Random Forest-based stat predictor."""
    
    def __init__(self, stat_type: str, model_version: str = "rf_v1.0", **kwargs):
        super().__init__(stat_type, model_version)
        
        # Default parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        default_params.update(kwargs)
        
        self.model = RandomForestRegressor(**default_params)


class XGBoostStatPredictor(StatPredictor):
    """XGBoost-based stat predictor."""
    
    def __init__(self, stat_type: str, model_version: str = "xgb_v1.0", **kwargs):
        super().__init__(stat_type, model_version)
        
        # Default parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        default_params.update(kwargs)
        
        self.model = xgb.XGBRegressor(**default_params)


class EnsembleStatPredictor(StatPredictor):
    """Ensemble of multiple models for stat prediction."""
    
    def __init__(self, stat_type: str, model_version: str = "ensemble_v1.0"):
        super().__init__(stat_type, model_version)
        
        self.models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'xgb': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'ridge': Ridge(alpha=1.0, random_state=42)
        }
        self.weights = None
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Train all models in the ensemble."""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train individual models
        predictions_val = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            predictions_val[name] = model.predict(X_val)
        
        # Calculate optimal weights using validation set
        self.weights = self._calculate_optimal_weights(predictions_val, y_val)
        
        # Calculate ensemble metrics
        ensemble_pred_val = self._ensemble_predict(predictions_val)
        
        metrics = {
            'val_mae': mean_absolute_error(y_val, ensemble_pred_val),
            'val_rmse': np.sqrt(mean_squared_error(y_val, ensemble_pred_val)),
            'val_r2': r2_score(y_val, ensemble_pred_val)
        }
        
        self.performance_metrics = metrics
        self.is_trained = True
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        
        return self._ensemble_predict(predictions)
    
    def _calculate_optimal_weights(self, predictions: Dict[str, np.ndarray], 
                                  y_true: np.ndarray) -> Dict[str, float]:
        """Calculate optimal weights for ensemble using grid search."""
        from itertools import product
        
        # Define weight grid
        weight_options = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        model_names = list(predictions.keys())
        
        best_mae = float('inf')
        best_weights = None
        
        # Grid search over weight combinations
        for weights in product(weight_options, repeat=len(model_names)):
            if sum(weights) == 0:
                continue
            
            # Normalize weights
            normalized_weights = [w / sum(weights) for w in weights]
            weight_dict = dict(zip(model_names, normalized_weights))
            
            # Calculate ensemble prediction
            ensemble_pred = sum(weight_dict[name] * predictions[name] 
                              for name in model_names)
            
            # Calculate MAE
            mae = mean_absolute_error(y_true, ensemble_pred)
            
            if mae < best_mae:
                best_mae = mae
                best_weights = weight_dict
        
        return best_weights
    
    def _ensemble_predict(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine predictions using learned weights."""
        return sum(self.weights[name] * predictions[name] 
                  for name in predictions.keys())


class ModelManager:
    """Manages multiple stat prediction models and handles model improvement."""
    
    def __init__(self, db_path: str = "data/nba_data.db", 
                 models_dir: str = "models/"):
        """Initialize the model manager."""
        self.db_path = db_path
        self.models_dir = models_dir
        self.predictors = {}
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
    def create_predictor(self, stat_type: str, model_type: str = "ensemble") -> StatPredictor:
        """Create a new stat predictor."""
        if model_type == "random_forest":
            predictor = RandomForestStatPredictor(stat_type)
        elif model_type == "xgboost":
            predictor = XGBoostStatPredictor(stat_type)
        elif model_type == "ensemble":
            predictor = EnsembleStatPredictor(stat_type)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return predictor
    
    def train_model(self, stat_type: str, training_data: pd.DataFrame, 
                   model_type: str = "random_forest") -> Dict[str, float]:
        """Train a model for a specific stat."""
        logger.info(f"Training {model_type} model for {stat_type}")
        
        # Prepare training data
        target_col = f'target_{stat_type}'
        if target_col not in training_data.columns:
            raise ValueError(f"Target column {target_col} not found in training data")
        
        # Remove rows with missing targets
        clean_data = training_data.dropna(subset=[target_col])
        
        if len(clean_data) < 50:
            logger.warning(f"Limited training data for {stat_type}: {len(clean_data)} samples. Training anyway.")
            if len(clean_data) < 20:
                raise ValueError(f"Insufficient training data for {stat_type}: {len(clean_data)} samples")
        
        # Separate features and target
        feature_columns = [col for col in clean_data.columns 
                          if not col.startswith('target_') and 
                          col not in ['player_id', 'game_id', 'game_date', 'target_date']]
        
        X = clean_data[feature_columns]
        y = clean_data[target_col]
        
        # Create and train predictor with improved parameters
        predictor = RandomForestStatPredictor(
            stat_type, 
            n_estimators=150,  # Increased for better performance
            max_depth=12,      # Slightly deeper trees
            min_samples_split=3,  # Reduced for more flexibility
            min_samples_leaf=1,   # Reduced for more flexibility
            random_state=42
        )
        metrics = predictor.train(X, y)
        
        # Store the predictor
        self.predictors[stat_type] = predictor
        
        # Save the model
        model_filename = f"{stat_type}_{model_type}_{predictor.model_version}.pkl"
        model_path = os.path.join(self.models_dir, model_filename)
        predictor.save_model(model_path)
        
        # Store performance metrics in database
        self._store_model_performance(stat_type, predictor.model_version, metrics, len(clean_data))
        
        logger.info(f"Trained {stat_type} model - Val MAE: {metrics['val_mae']:.2f}")
        
        return metrics
    
    def load_models(self, stat_types: List[str]) -> None:
        """Load the latest models for specified stat types."""
        for stat_type in stat_types:
            model_files = [f for f in os.listdir(self.models_dir) 
                          if f.startswith(f"{stat_type}_") and f.endswith('.pkl')]
            
            if not model_files:
                logger.warning(f"No saved model found for {stat_type}")
                continue
            
            # Sort by creation time and get the latest
            model_files.sort(key=lambda x: os.path.getctime(os.path.join(self.models_dir, x)), 
                           reverse=True)
            latest_model = model_files[0]
            
            # Load the model
            predictor = StatPredictor(stat_type)
            try:
                with open(os.path.join(self.models_dir, latest_model), 'rb') as f:
                    model_data = pickle.load(f)
                
                predictor.model = model_data['model']
                predictor.stat_type = model_data['stat_type']
                predictor.model_version = model_data['model_version']
                predictor.feature_importance = model_data['feature_importance']
                predictor.performance_metrics = model_data['performance_metrics']
                predictor.is_trained = True
                
                self.predictors[stat_type] = predictor
                logger.info(f"Loaded model for {stat_type}: {latest_model}")
            except Exception as e:
                logger.error(f"Error loading model for {stat_type}: {e}")
    
    def predict_stats(self, features_df: pd.DataFrame, 
                     stat_types: List[str]) -> pd.DataFrame:
        """Make predictions for multiple stat types."""
        predictions_df = features_df.copy()
        
        # Prepare features (excluding metadata columns)
        feature_columns = [col for col in features_df.columns 
                          if col not in ['player_id', 'game_id', 'game_date', 'target_date']]
        X = features_df[feature_columns]
        
        for stat_type in stat_types:
            if stat_type not in self.predictors:
                logger.warning(f"No trained model available for {stat_type}")
                predictions_df[f'predicted_{stat_type}'] = 0
                predictions_df[f'confidence_{stat_type}'] = 0
                continue
            
            try:
                predictor = self.predictors[stat_type]
                predictions = predictor.predict(X)
                
                # Calculate confidence based on model performance
                confidence = self._calculate_confidence(predictor)
                
                predictions_df[f'predicted_{stat_type}'] = predictions
                predictions_df[f'confidence_{stat_type}'] = confidence
                
            except Exception as e:
                logger.error(f"Error predicting {stat_type}: {e}")
                predictions_df[f'predicted_{stat_type}'] = 0
                predictions_df[f'confidence_{stat_type}'] = 0
        
        return predictions_df
    
    def _calculate_confidence(self, predictor: StatPredictor) -> float:
        """Calculate prediction confidence based on model performance."""
        if not predictor.performance_metrics:
            return 0.5
        
        # Use validation R² as a base for confidence
        r2 = predictor.performance_metrics.get('val_r2', 0)
        
        # Convert R² to confidence score (0 to 1)
        confidence = max(0, min(1, (r2 + 1) / 2))
        
        return confidence
    
    def _store_model_performance(self, stat_type: str, model_version: str, 
                                metrics: Dict[str, float], sample_size: int) -> None:
        """Store model performance metrics in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO model_performance 
            (model_version, stat_type, mae, rmse, accuracy_rate, evaluation_date, sample_size)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (model_version, stat_type, metrics['val_mae'], metrics['val_rmse'], 
              metrics['val_r2'], datetime.now().isoformat(), sample_size))
        
        conn.commit()
        conn.close()
    
    def evaluate_predictions(self, stat_types: List[str]) -> Dict[str, Dict[str, float]]:
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
                mae = mean_absolute_error(df['actual_value'], df['predicted_value'])
                rmse = np.sqrt(mean_squared_error(df['actual_value'], df['predicted_value']))
                r2 = r2_score(df['actual_value'], df['predicted_value'])
                
                evaluation_results[stat_type] = {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'sample_size': len(df)
                }
            else:
                evaluation_results[stat_type] = {
                    'mae': 0,
                    'rmse': 0,
                    'r2': 0,
                    'sample_size': len(df)
                }
        
        conn.close()
        return evaluation_results
    
    def should_retrain_model(self, stat_type: str, 
                           performance_threshold: float = 0.1) -> bool:
        """Determine if a model should be retrained based on recent performance."""
        # For now, return False (no retraining needed)
        # In a full implementation, you would compare recent vs training performance
        return False 