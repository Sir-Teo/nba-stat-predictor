#!/usr/bin/env python3
"""
NBA Stat Predictor - Main application for predicting NBA player stats with self-improvement.
"""

import argparse
import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd

# Add src to path
sys.path.append('src')

from src.data.nba_data_collector import NBADataCollector
from src.data.feature_engineer import FeatureEngineer
from src.models.stat_predictors import ModelManager
from src.predictions.tonight_predictor import TonightPredictor
from src.evaluation.backtester import NBABacktester

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NBAStatPredictorApp:
    """Main application class for NBA stat prediction."""
    
    def __init__(self, db_path: str = "data/nba_data.db"):
        """Initialize the application."""
        self.db_path = db_path
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
        # Initialize components
        self.data_collector = NBADataCollector(db_path)
        self.feature_engineer = FeatureEngineer(db_path)
        self.model_manager = ModelManager(db_path)
        self.tonight_predictor = TonightPredictor(db_path)
        
    def collect_data(self, players_limit: int = 50):
        """Collect historical NBA data for training."""
        logger.info("Starting comprehensive data collection...")
        
        # Get star players for better training data
        try:
            star_players = self.data_collector.get_team_leaders()
            if players_limit > len(star_players):
                # Add more popular players if we need more
                additional_players = self.data_collector.get_popular_players(players_limit - len(star_players))
                all_players = star_players + additional_players
            else:
                all_players = star_players[:players_limit]
            
            logger.info(f"Selected {len(all_players)} players for data collection")
            
        except Exception as e:
            logger.warning(f"Could not get star players, using fallback method: {e}")
            # Fallback to original method
            all_players_df = self.data_collector.get_all_players()
            if all_players_df.empty:
                logger.error("Could not fetch NBA players data")
                return
            all_players = all_players_df.head(players_limit)['id'].tolist()
        
        logger.info(f"Collecting data for {len(all_players)} players across 4 seasons...")
        
        # Collect historical data across multiple seasons
        self.data_collector.collect_historical_data(
            players_list=all_players,
            seasons=["2023-24", "2022-23", "2021-22", "2020-21"]
        )
        
        # Show collection summary
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM player_games")
        total_games = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT player_id) FROM player_games")
        unique_players = cursor.fetchone()[0]
        
        cursor.execute("SELECT MIN(game_date), MAX(game_date) FROM player_games")
        date_range = cursor.fetchone()
        
        conn.close()
        
        logger.info("="*60)
        logger.info("DATA COLLECTION SUMMARY")
        logger.info("="*60)
        logger.info(f"Total games collected: {total_games}")
        logger.info(f"Unique players: {unique_players}")
        logger.info(f"Date range: {date_range[0]} to {date_range[1]}")
        logger.info("="*60)
        
        logger.info("Data collection completed!")
    
    def train_models(self):
        """Train prediction models."""
        logger.info("Starting model training...")
        
        # Create training dataset
        logger.info("Creating training dataset...")
        
        # Get player IDs that have sufficient data
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get players with at least 20 games
        cursor.execute("""
            SELECT player_id, COUNT(*) as game_count
            FROM player_games
            GROUP BY player_id
            HAVING game_count >= 20
            ORDER BY game_count DESC
            LIMIT 100
        """)
        
        players_with_data = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        if not players_with_data:
            logger.error("No players with sufficient data found. Please collect data first.")
            return
        
        logger.info(f"Training models with data from {len(players_with_data)} players")
        
        # Create training dataset
        start_date = "2022-10-01"
        end_date = "2024-01-01"
        
        training_data = self.feature_engineer.create_training_dataset(
            players_list=players_with_data,
            start_date=start_date,
            end_date=end_date,
            target_stats=['pts', 'reb', 'ast', 'stl', 'blk']
        )
        
        if training_data.empty:
            logger.error("Could not create training dataset")
            return
        
        logger.info(f"Created training dataset with {len(training_data)} samples")
        
        # Train models for each stat
        stat_types = ['pts', 'reb', 'ast', 'stl', 'blk']
        
        for stat_type in stat_types:
            try:
                logger.info(f"Training model for {stat_type}...")
                metrics = self.model_manager.train_model(stat_type, training_data)
                logger.info(f"{stat_type} model trained - Validation MAE: {metrics['val_mae']:.2f}")
            except Exception as e:
                logger.error(f"Error training {stat_type} model: {e}")
        
        logger.info("Model training completed!")
    
    def predict_tonight(self):
        """Make predictions for tonight's games."""
        logger.info("Making predictions for tonight's games...")
        
        predictions_df = self.tonight_predictor.get_tonights_predictions()
        
        if predictions_df.empty:
            logger.info("No predictions generated (no games or insufficient data)")
            return
        
        # Display top predictions
        print("\n" + "="*80)
        print("TONIGHT'S TOP PREDICTIONS")
        print("="*80)
        
        stat_types = ['pts', 'reb', 'ast']
        
        for stat_type in stat_types:
            print(f"\nTop 10 {stat_type.upper()} predictions:")
            top_predictions = self.tonight_predictor.get_top_predictions(stat_type, 10)
            
            if not top_predictions.empty:
                for i, (_, row) in enumerate(top_predictions.iterrows(), 1):
                    print(f"{i:2d}. {row['player_name']:25} - {row['predicted_value']:5.1f} "
                          f"(confidence: {row['confidence']:4.2f})")
            else:
                print("   No predictions available")
        
        print("\n" + "="*80)
    
    def update_results(self, days_back: int = 1):
        """Update predictions with actual game results."""
        logger.info(f"Updating predictions with results from {days_back} days ago...")
        
        update_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        self.tonight_predictor.update_predictions_with_results(update_date)
        
        logger.info("Results update completed!")
    
    def show_accuracy(self):
        """Show recent prediction accuracy."""
        logger.info("Analyzing recent prediction accuracy...")
        
        accuracy = self.tonight_predictor.analyze_recent_accuracy(days=14)
        
        print("\n" + "="*60)
        print("PREDICTION ACCURACY (Last 14 Days)")
        print("="*60)
        
        for stat_type, metrics in accuracy.items():
            if metrics['sample_size'] > 0:
                print(f"{stat_type.upper():4} - MAE: {metrics['mae']:5.2f}, "
                      f"RMSE: {metrics['rmse']:5.2f}, "
                      f"Samples: {metrics['sample_size']:3d}")
            else:
                print(f"{stat_type.upper():4} - No data available")
        
        print("="*60)
    
    def retrain_if_needed(self):
        """Check if models need retraining and retrain if necessary."""
        logger.info("Checking if models need retraining...")
        
        stat_types = ['pts', 'reb', 'ast', 'stl', 'blk']
        retrained = False
        
        for stat_type in stat_types:
            should_retrain = self.model_manager.should_retrain_model(stat_type)
            
            if should_retrain:
                logger.info(f"Retraining {stat_type} model due to performance degradation...")
                try:
                    # Get fresh training data
                    import sqlite3
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    
                    cursor.execute("""
                        SELECT player_id, COUNT(*) as game_count
                        FROM player_games
                        GROUP BY player_id
                        HAVING game_count >= 20
                        ORDER BY game_count DESC
                        LIMIT 100
                    """)
                    
                    players_with_data = [row[0] for row in cursor.fetchall()]
                    conn.close()
                    
                    if players_with_data:
                        # Create fresh training dataset
                        start_date = "2022-10-01"
                        end_date = datetime.now().strftime('%Y-%m-%d')
                        
                        training_data = self.feature_engineer.create_training_dataset(
                            players_list=players_with_data,
                            start_date=start_date,
                            end_date=end_date,
                            target_stats=[stat_type]
                        )
                        
                        if not training_data.empty:
                            metrics = self.model_manager.train_model(stat_type, training_data)
                            logger.info(f"Retrained {stat_type} model - New MAE: {metrics['val_mae']:.2f}")
                            retrained = True
                        
                except Exception as e:
                    logger.error(f"Error retraining {stat_type} model: {e}")
        
        if not retrained:
            logger.info("No models needed retraining")

    def run_backtest(self, season: str = "2023-24"):
        """Run a comprehensive backtest for the specified season."""
        logger.info(f"Running backtest for {season} season...")
        
        # Initialize backtester
        backtester = NBABacktester(self.db_path)
        
        try:
            # Run the backtest
            results = backtester.run_season_backtest(
                season=season,
                train_months=2,  # Reduced for more test data
                min_games_per_player=5  # Reduced for broader player coverage
            )
            
            # Print results
            backtester.print_backtest_results(results)
            
            # Create visualizations (will save plots)
            backtester.create_backtest_visualizations(results, save_plots=True)
            
            logger.info("Backtest completed successfully!")
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            print(f"\nBacktest failed: {e}")
            print("\nTo run a backtest, you need:")
            print("1. Historical data: python main.py collect-data")
            print("2. Sufficient games per player (15+ games)")
            print("3. Data spanning multiple months")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='NBA Stat Predictor with Self-Improvement')
    
    parser.add_argument('command', choices=[
        'collect-data', 'train', 'predict', 'update-results', 
        'accuracy', 'retrain', 'full-pipeline', 'backtest'
    ], help='Command to execute')
    
    parser.add_argument('--players-limit', type=int, default=50,
                       help='Number of players to collect data for (default: 50)')
    
    parser.add_argument('--days-back', type=int, default=1,
                       help='Days back to update results for (default: 1)')
    
    parser.add_argument('--season', default='2023-24',
                       help='NBA season for backtest (default: 2023-24)')
    
    parser.add_argument('--db-path', default='data/nba_data.db',
                       help='Path to SQLite database (default: data/nba_data.db)')
    
    args = parser.parse_args()
    
    # Initialize application
    app = NBAStatPredictorApp(args.db_path)
    
    try:
        if args.command == 'collect-data':
            app.collect_data(args.players_limit)
            
        elif args.command == 'train':
            app.train_models()
            
        elif args.command == 'predict':
            app.predict_tonight()
            
        elif args.command == 'update-results':
            app.update_results(args.days_back)
            
        elif args.command == 'accuracy':
            app.show_accuracy()
            
        elif args.command == 'retrain':
            app.retrain_if_needed()
            
        elif args.command == 'full-pipeline':
            logger.info("Running full pipeline...")
            
            # Check if we have data, if not collect some
            import sqlite3
            conn = sqlite3.connect(args.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM player_games")
            game_count = cursor.fetchone()[0]
            conn.close()
            
            if game_count < 100:
                logger.info("Insufficient data, collecting...")
                app.collect_data(50)  # Increased for better coverage
            
            # Train models
            app.train_models()
            
            # Make predictions
            app.predict_tonight()
            
            # Show accuracy if we have historical predictions
            app.show_accuracy()
            
        elif args.command == 'backtest':
            app.run_backtest(args.season)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 