"""
NBA Stat Predictor Backtesting Module - Evaluates model performance on historical data.
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

from ..data.nba_data_collector import NBADataCollector
from ..data.feature_engineer import FeatureEngineer
from ..models.stat_predictors import ModelManager

logger = logging.getLogger(__name__)


class NBABacktester:
    """Backtests NBA stat prediction models on historical data."""
    
    def __init__(self, db_path: str = "data/nba_data.db"):
        """Initialize the backtester."""
        self.db_path = db_path
        self.data_collector = NBADataCollector(db_path)
        self.feature_engineer = FeatureEngineer(db_path)
        self.model_manager = ModelManager(db_path)
        self.stat_types = ['pts', 'reb', 'ast', 'stl', 'blk']
        
    def run_season_backtest(self, season: str = "2023-24", 
                          train_months: int = 3, 
                          min_games_per_player: int = 15) -> Dict:
        """
        Run a backtest for an entire season.
        
        Args:
            season: NBA season (e.g., "2023-24")
            train_months: Number of months to use for training
            min_games_per_player: Minimum games needed per player
            
        Returns:
            Dictionary containing backtest results
        """
        logger.info(f"Starting backtest for {season} season")
        
        # Define season dates
        if season == "2023-24":
            season_start = "2023-10-01"
            season_end = "2024-04-15"
        elif season == "2022-23":
            season_start = "2022-10-01"
            season_end = "2023-04-15"
        else:
            raise ValueError(f"Season {season} not supported")
        
        # Calculate training cutoff date
        train_cutoff = datetime.strptime(season_start, "%Y-%m-%d") + timedelta(days=train_months * 30)
        train_cutoff_str = train_cutoff.strftime("%Y-%m-%d")
        
        logger.info(f"Training period: {season_start} to {train_cutoff_str}")
        logger.info(f"Testing period: {train_cutoff_str} to {season_end}")
        
        # Get players with sufficient data
        players_with_data = self._get_players_with_sufficient_data(
            season_start, season_end, min_games_per_player
        )
        
        if len(players_with_data) < 20:
            logger.warning(f"Limited players with data: {len(players_with_data)}. Running with available data.")
            if len(players_with_data) < 1:
                raise ValueError(f"Insufficient players with data: {len(players_with_data)}")
        
        logger.info(f"Found {len(players_with_data)} players with sufficient data")
        
        # Create training dataset
        logger.info("Creating training dataset...")
        training_data = self.feature_engineer.create_training_dataset(
            players_list=players_with_data,
            start_date=season_start,
            end_date=train_cutoff_str,
            target_stats=self.stat_types
        )
        
        if training_data.empty:
            raise ValueError("Could not create training dataset")
        
        logger.info(f"Training dataset: {len(training_data)} samples")
        
        # Train models
        logger.info("Training models...")
        training_metrics = {}
        for stat_type in self.stat_types:
            try:
                metrics = self.model_manager.train_model(stat_type, training_data)
                training_metrics[stat_type] = metrics
                logger.info(f"Trained {stat_type} model - MAE: {metrics['val_mae']:.2f}")
            except Exception as e:
                logger.error(f"Error training {stat_type} model: {e}")
                continue
        
        # Generate predictions for testing period
        logger.info("Generating predictions for testing period...")
        test_predictions = self._generate_test_predictions(
            players_with_data, train_cutoff_str, season_end
        )
        
        if test_predictions.empty:
            raise ValueError("No test predictions generated")
        
        logger.info(f"Generated {len(test_predictions)} test predictions")
        
        # Calculate accuracy metrics
        logger.info("Calculating accuracy metrics...")
        accuracy_metrics = self._calculate_accuracy_metrics(test_predictions)
        
        # Create summary
        backtest_results = {
            'season': season,
            'train_period': f"{season_start} to {train_cutoff_str}",
            'test_period': f"{train_cutoff_str} to {season_end}",
            'players_count': len(players_with_data),
            'training_samples': len(training_data),
            'test_predictions': len(test_predictions),
            'training_metrics': training_metrics,
            'test_metrics': accuracy_metrics,
            'predictions_df': test_predictions
        }
        
        return backtest_results
    
    def _get_players_with_sufficient_data(self, start_date: str, end_date: str, 
                                        min_games: int) -> List[int]:
        """Get players with sufficient games in the date range."""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT player_id, COUNT(*) as game_count
            FROM player_games
            WHERE game_date BETWEEN ? AND ?
            GROUP BY player_id
            HAVING game_count >= ?
            ORDER BY game_count DESC
        """
        
        cursor = conn.cursor()
        cursor.execute(query, (start_date, end_date, min_games))
        players = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return players
    
    def _generate_test_predictions(self, players_list: List[int], 
                                 start_date: str, end_date: str) -> pd.DataFrame:
        """Generate predictions for the testing period."""
        all_predictions = []
        
        conn = sqlite3.connect(self.db_path)
        
        for player_id in players_list:
            # Get all games for this player in test period
            query = """
                SELECT * FROM player_games
                WHERE player_id = ? AND game_date BETWEEN ? AND ?
                ORDER BY game_date
            """
            
            player_games = pd.read_sql_query(query, conn, params=(player_id, start_date, end_date))
            
            if len(player_games) < 5:  # Need at least 5 games to test
                continue
            
            # For each game, create features and make predictions
            for i, (_, game) in enumerate(player_games.iterrows()):
                game_date = game['game_date']
                
                try:
                    # Create features based on data before this game
                    features_df = self.feature_engineer.create_features_for_player(
                        player_id, game_date
                    )
                    
                    if features_df.empty:
                        continue
                    
                    # Make predictions
                    predictions_df = self.model_manager.predict_stats(features_df, self.stat_types)
                    
                    if not predictions_df.empty:
                        # Add actual values and metadata
                        for stat_type in self.stat_types:
                            if stat_type in game:
                                predictions_df[f'actual_{stat_type}'] = game[stat_type]
                        
                        predictions_df['player_id'] = player_id
                        predictions_df['game_date'] = game_date
                        predictions_df['game_id'] = game.get('game_id', '')
                        
                        all_predictions.append(predictions_df)
                        
                except Exception as e:
                    logger.warning(f"Error predicting for player {player_id} on {game_date}: {e}")
                    continue
        
        conn.close()
        
        if all_predictions:
            return pd.concat(all_predictions, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _calculate_accuracy_metrics(self, predictions_df: pd.DataFrame) -> Dict:
        """Calculate accuracy metrics for predictions."""
        metrics = {}
        
        for stat_type in self.stat_types:
            predicted_col = f'predicted_{stat_type}'
            actual_col = f'actual_{stat_type}'
            
            if predicted_col in predictions_df.columns and actual_col in predictions_df.columns:
                # Remove rows with missing values
                valid_rows = predictions_df.dropna(subset=[predicted_col, actual_col])
                
                if len(valid_rows) > 0:
                    y_true = valid_rows[actual_col]
                    y_pred = valid_rows[predicted_col]
                    
                    mae = mean_absolute_error(y_true, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    r2 = r2_score(y_true, y_pred)
                    
                    # Calculate additional metrics
                    mean_actual = np.mean(y_true)
                    baseline_mae = np.mean(np.abs(y_true - mean_actual))
                    improvement = (baseline_mae - mae) / baseline_mae * 100
                    
                    metrics[stat_type] = {
                        'mae': mae,
                        'rmse': rmse,
                        'r2': r2,
                        'mean_actual': mean_actual,
                        'baseline_mae': baseline_mae,
                        'improvement_pct': improvement,
                        'sample_size': len(valid_rows)
                    }
                else:
                    metrics[stat_type] = {
                        'mae': 0, 'rmse': 0, 'r2': 0, 'mean_actual': 0,
                        'baseline_mae': 0, 'improvement_pct': 0, 'sample_size': 0
                    }
        
        return metrics
    
    def print_backtest_results(self, results: Dict) -> None:
        """Print formatted backtest results."""
        print("\n" + "="*80)
        print(f"NBA STAT PREDICTOR BACKTEST RESULTS - {results['season'].upper()}")
        print("="*80)
        
        print(f"Training Period: {results['train_period']}")
        print(f"Testing Period:  {results['test_period']}")
        print(f"Players:         {results['players_count']}")
        print(f"Training Games:  {results['training_samples']}")
        print(f"Test Predictions: {results['test_predictions']}")
        
        print("\n" + "-"*60)
        print("TRAINING PERFORMANCE")
        print("-"*60)
        print(f"{'Stat':<4} {'Train MAE':<10} {'Val MAE':<10} {'Val R²':<8}")
        print("-"*40)
        
        for stat_type in self.stat_types:
            if stat_type in results['training_metrics']:
                metrics = results['training_metrics'][stat_type]
                print(f"{stat_type.upper():<4} {metrics['train_mae']:<10.2f} "
                      f"{metrics['val_mae']:<10.2f} {metrics['val_r2']:<8.3f}")
        
        print("\n" + "-"*60)
        print("BACKTEST PERFORMANCE")
        print("-"*60)
        print(f"{'Stat':<4} {'Test MAE':<9} {'Test R²':<8} {'Baseline':<9} {'Improve':<8} {'Samples':<8}")
        print("-"*55)
        
        for stat_type in self.stat_types:
            if stat_type in results['test_metrics']:
                metrics = results['test_metrics'][stat_type]
                if metrics['sample_size'] > 0:
                    print(f"{stat_type.upper():<4} {metrics['mae']:<9.2f} "
                          f"{metrics['r2']:<8.3f} {metrics['baseline_mae']:<9.2f} "
                          f"{metrics['improvement_pct']:<7.1f}% {metrics['sample_size']:<8d}")
                else:
                    print(f"{stat_type.upper():<4} {'N/A':<9} {'N/A':<8} {'N/A':<9} {'N/A':<8} {'0':<8}")
        
        print("\n" + "="*80)
        print("SUMMARY:")
        
        # Calculate overall improvement
        total_improvements = []
        for stat_type in self.stat_types:
            if (stat_type in results['test_metrics'] and 
                results['test_metrics'][stat_type]['sample_size'] > 0):
                total_improvements.append(results['test_metrics'][stat_type]['improvement_pct'])
        
        if total_improvements:
            avg_improvement = np.mean(total_improvements)
            print(f"Average improvement over baseline: {avg_improvement:.1f}%")
            
            if avg_improvement > 10:
                print("✅ Strong predictive performance!")
            elif avg_improvement > 5:
                print("✅ Good predictive performance")
            elif avg_improvement > 0:
                print("⚠️  Modest improvement over baseline")
            else:
                print("❌ Model needs improvement")
        else:
            print("❌ Insufficient data for evaluation")
        
        print("="*80)
    
    def create_backtest_visualizations(self, results: Dict, save_plots: bool = True) -> None:
        """Create visualizations for backtest results."""
        predictions_df = results['predictions_df']
        
        if predictions_df.empty:
            logger.warning("No predictions data for visualization")
            return
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"NBA Stat Predictor Backtest Results - {results['season']}", fontsize=16)
        
        stat_positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
        
        for idx, stat_type in enumerate(self.stat_types):
            if idx >= len(stat_positions):
                break
                
            row, col = stat_positions[idx]
            ax = axes[row, col]
            
            predicted_col = f'predicted_{stat_type}'
            actual_col = f'actual_{stat_type}'
            
            if predicted_col in predictions_df.columns and actual_col in predictions_df.columns:
                # Remove rows with missing values
                valid_data = predictions_df.dropna(subset=[predicted_col, actual_col])
                
                if len(valid_data) > 0:
                    # Scatter plot
                    ax.scatter(valid_data[actual_col], valid_data[predicted_col], 
                             alpha=0.6, s=20)
                    
                    # Perfect prediction line
                    min_val = min(valid_data[actual_col].min(), valid_data[predicted_col].min())
                    max_val = max(valid_data[actual_col].max(), valid_data[predicted_col].max())
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
                    
                    # Labels and metrics
                    r2 = results['test_metrics'][stat_type]['r2']
                    mae = results['test_metrics'][stat_type]['mae']
                    
                    ax.set_xlabel(f'Actual {stat_type.upper()}')
                    ax.set_ylabel(f'Predicted {stat_type.upper()}')
                    ax.set_title(f'{stat_type.upper()}: R² = {r2:.3f}, MAE = {mae:.2f}')
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{stat_type.upper()}: No Data')
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{stat_type.upper()}: No Data')
        
        # Hide the last subplot if we have fewer than 6 stats
        if len(self.stat_types) < 6:
            axes[1, 2].set_visible(False)
        
        plt.tight_layout()
        
        if save_plots:
            filename = f"backtest_results_{results['season'].replace('-', '_')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved backtest visualization: {filename}")
        
        plt.show()
    
    def run_rolling_backtest(self, season: str = "2023-24", 
                           window_days: int = 30, 
                           step_days: int = 7) -> Dict:
        """
        Run a rolling backtest to see how performance changes over time.
        
        Args:
            season: NBA season
            window_days: Size of training window in days
            step_days: Step size between evaluations
            
        Returns:
            Dictionary with rolling backtest results
        """
        logger.info(f"Running rolling backtest for {season}")
        
        # This would implement a more sophisticated rolling evaluation
        # For now, return a simplified version
        return {"message": "Rolling backtest not yet implemented"}


def main():
    """Main function for running backtests."""
    backtester = NBABacktester()
    
    try:
        # Run backtest for current season
        results = backtester.run_season_backtest(
            season="2023-24",
            train_months=3,
            min_games_per_player=15
        )
        
        # Print results
        backtester.print_backtest_results(results)
        
        # Create visualizations
        backtester.create_backtest_visualizations(results)
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise


if __name__ == "__main__":
    main() 