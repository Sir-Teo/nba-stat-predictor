#!/usr/bin/env python3
"""
NBA Stat Predictor - Main application for predicting NBA player stats with self-improvement.
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.append("src")

from src.data.feature_engineer import FeatureEngineer
from src.data.nba_data_collector import NBADataCollector
from src.evaluation.backtester import NBABacktester
from src.models.stat_predictors import ModelManager
from src.predictions.tonight_predictor import TonightPredictor

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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

    def collect_data(self, players_limit: int = 100):
        """Collect comprehensive historical NBA data for training with enhanced settings."""
        logger.info("Starting enhanced comprehensive data collection...")

        # Get comprehensive player list with better selection
        try:
            # Get star players for better training data
            star_players = self.data_collector.get_team_leaders()
            
            # Get additional popular players for comprehensive coverage
            popular_players = self.data_collector.get_popular_players(players_limit)
            
            # Combine and deduplicate
            all_players = list(set(star_players + popular_players))
            
            # Limit to requested number
            if len(all_players) > players_limit:
                all_players = all_players[:players_limit]

            logger.info(f"Selected {len(all_players)} players for enhanced data collection")

        except Exception as e:
            logger.warning(f"Could not get comprehensive player list, using fallback method: {e}")
            # Fallback to original method
            all_players_df = self.data_collector.get_all_players()
            if all_players_df.empty:
                logger.error("Could not fetch NBA players data")
                return
            all_players = all_players_df.head(players_limit)["id"].tolist()

        logger.info(
            f"Collecting enhanced data for {len(all_players)} players across 9 seasons..."
        )

        # Collect comprehensive historical data across extended seasons with intelligent checking
        self.data_collector.collect_historical_data(
            players_list=all_players,
            seasons=["2024-25", "2023-24", "2022-23", "2021-22", "2020-21", "2019-20", "2018-19", "2017-18", "2016-17"],
            include_playoffs=True,
            include_all_star=False,
            force_refresh=False,  # Don't re-fetch existing data
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

        logger.info("=" * 60)
        logger.info("ENHANCED DATA COLLECTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total games collected: {total_games:,}")
        logger.info(f"Unique players: {unique_players}")
        logger.info(f"Date range: {date_range[0]} to {date_range[1]}")
        
        # Add data quality validation
        quality_report = self.data_collector.validate_data_quality()
        if "data_quality_score" in quality_report:
            logger.info(f"Data quality score: {quality_report['data_quality_score']:.1f}/100")
        
        logger.info("=" * 60)

        logger.info("Enhanced data collection completed!")

    def train_models(self):
        """Train enhanced prediction models with comprehensive features."""
        logger.info("Starting enhanced model training...")

        # Get qualified players with enhanced criteria
        players_with_data = self._get_qualified_players_for_training()

        if not players_with_data:
            logger.error(
                "No players with sufficient data found. Please collect data first."
            )
            return

        logger.info(f"Training enhanced models with data from {len(players_with_data)} players")

        # Create comprehensive training dataset with 8 years of historical data
        start_date = "2016-10-01"  # Go back 8 years for comprehensive training
        end_date = "2024-06-30"    # Include recent data

        print("📊 Creating enhanced training dataset with 530+ features...")
        training_data = self.feature_engineer.create_training_dataset(
            players_list=players_with_data,
            start_date=start_date,
            end_date=end_date,
            target_stats=["pts", "reb", "ast", "stl", "blk"],
            include_h2h_features=True,
            include_advanced_features=True,
        )

        if training_data.empty:
            logger.error("Could not create training dataset")
            return

        logger.info(f"Created enhanced training dataset with {len(training_data)} samples")
        logger.info(f"Features: {len(training_data.columns)}")

        # Log dataset statistics
        self._log_enhanced_dataset_statistics(training_data)

        # Train models for each stat with enhanced settings
        stat_types = ["pts", "reb", "ast", "stl", "blk"]
        training_results = {}
        successful_models = 0

        print("\n🏀 Training enhanced models with advanced optimization:")
        with tqdm(
            stat_types,
            desc="Training Models",
            ncols=80,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as pbar:
            for stat_type in pbar:
                try:
                    pbar.set_description(f"Training {stat_type.upper()}")
                    metrics = self.model_manager.train_model(
                        stat_type, training_data, optimize_hyperparams=True
                    )
                    training_results[stat_type] = metrics
                    successful_models += 1
                    
                    mae = metrics.get("test_mae", metrics.get("val_mae", 0))
                    r2 = metrics.get("test_r2", metrics.get("val_r2", 0))
                    pbar.write(
                        f"✅ {stat_type.upper()} model trained - MAE: {mae:.2f}, R²: {r2:.3f}"
                    )
                except Exception as e:
                    pbar.write(f"❌ Error training {stat_type} model: {e}")
                    training_results[stat_type] = {"error": str(e)}

        # Generate enhanced training summary
        self._log_enhanced_training_summary(training_results, successful_models)

        print("\n✅ Enhanced model training completed!")
        print("💡 Models now include 530+ features with advanced hyperparameter optimization.")

    def _get_qualified_players_for_training(self, min_games: int = 25, min_seasons: int = 2) -> List[int]:
        """Get players with sufficient data for enhanced training."""
        logger.info(f"Finding players with at least {min_games} games and {min_seasons} seasons")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get players with sufficient games and multiple seasons
            cursor.execute(
                """
                SELECT 
                    player_id,
                    COUNT(*) as total_games,
                    COUNT(DISTINCT 
                        CASE 
                            WHEN game_date >= '2024-10-01' THEN '2024-25'
                            WHEN game_date >= '2023-10-01' THEN '2023-24'
                            WHEN game_date >= '2022-10-01' THEN '2022-23'
                            WHEN game_date >= '2021-10-01' THEN '2021-22'
                            WHEN game_date >= '2020-10-01' THEN '2020-21'
                            WHEN game_date >= '2019-10-01' THEN '2019-20'
                            WHEN game_date >= '2018-10-01' THEN '2018-19'
                            WHEN game_date >= '2017-10-01' THEN '2017-18'
                            WHEN game_date >= '2016-10-01' THEN '2016-17'
                            ELSE 'Older'
                        END
                    ) as seasons_played
                FROM player_games
                GROUP BY player_id
                HAVING total_games >= ? AND seasons_played >= ?
                ORDER BY total_games DESC
                LIMIT 200
            """, (min_games, min_seasons))

            players_with_data = [row[0] for row in cursor.fetchall()]
            conn.close()

            logger.info(f"Found {len(players_with_data)} qualified players for enhanced training")
            return players_with_data

        except Exception as e:
            logger.error(f"Error getting qualified players: {e}")
            return []

    def _log_enhanced_dataset_statistics(self, training_data: pd.DataFrame):
        """Log comprehensive dataset statistics for enhanced training."""
        logger.info("Enhanced Dataset Statistics:")
        logger.info(f"   Total samples: {len(training_data):,}")
        logger.info(f"   Features: {len(training_data.columns)}")
        
        # Target variable statistics
        stat_types = ["pts", "reb", "ast", "stl", "blk"]
        for stat in stat_types:
            target_col = f"target_{stat}"
            if target_col in training_data.columns:
                mean_val = training_data[target_col].mean()
                std_val = training_data[target_col].std()
                min_val = training_data[target_col].min()
                max_val = training_data[target_col].max()
                logger.info(f"   {stat.upper()}: mean={mean_val:.2f}, std={std_val:.2f}, range=[{min_val:.1f}, {max_val:.1f}]")
        
        # Feature categories
        feature_categories = {
            "rolling": len([col for col in training_data.columns if "rolling" in col]),
            "trend": len([col for col in training_data.columns if "trend" in col]),
            "age": len([col for col in training_data.columns if "age" in col]),
            "opponent": len([col for col in training_data.columns if "opp" in col]),
            "h2h": len([col for col in training_data.columns if "h2h" in col]),
            "situational": len([col for col in training_data.columns if "situational" in col]),
            "momentum": len([col for col in training_data.columns if "momentum" in col])
        }
        
        logger.info("Feature Categories:")
        for category, count in feature_categories.items():
            if count > 0:
                logger.info(f"   {category}: {count} features")

    def _log_enhanced_training_summary(self, training_results: Dict, successful_models: int):
        """Log comprehensive training summary for enhanced models."""
        logger.info("=" * 80)
        logger.info("ENHANCED MODEL TRAINING SUMMARY")
        logger.info("=" * 80)
        
        # Calculate average metrics
        mae_values = []
        r2_values = []
        
        for stat_type, metrics in training_results.items():
            if "error" not in metrics:
                mae = metrics.get("test_mae", metrics.get("val_mae", 0))
                r2 = metrics.get("test_r2", metrics.get("val_r2", 0))
                if mae > 0:
                    mae_values.append(mae)
                if r2 != 0:
                    r2_values.append(r2)
        
        summary = {
            "total_models": len(training_results),
            "successful_models": successful_models,
            "failed_models": len(training_results) - successful_models,
            "avg_mae": sum(mae_values) / len(mae_values) if mae_values else 0,
            "avg_r2": sum(r2_values) / len(r2_values) if r2_values else 0,
            "best_mae": min(mae_values) if mae_values else 0,
            "worst_mae": max(mae_values) if mae_values else 0,
            "best_r2": max(r2_values) if r2_values else 0,
            "worst_r2": min(r2_values) if r2_values else 0
        }
        
        for key, value in summary.items():
            if isinstance(value, float):
                logger.info(f"{key}: {value:.3f}")
            else:
                logger.info(f"{key}: {value}")
        
        logger.info("=" * 80)

    def predict_tonight(self):
        """Make predictions for tonight's games."""
        logger.info("Making predictions for tonight's games...")

        predictions_df = self.tonight_predictor.get_tonights_predictions()

        if predictions_df.empty:
            logger.info("No predictions generated (no games or insufficient data)")
            return

        # Display top predictions
        print("\n" + "=" * 80)
        print("TONIGHT'S TOP PREDICTIONS")
        print("=" * 80)

        stat_types = ["pts", "reb", "ast"]

        for stat_type in stat_types:
            print(f"\nTop 10 {stat_type.upper()} predictions:")
            top_predictions = self.tonight_predictor.get_top_predictions(stat_type, 10)

            if not top_predictions.empty:
                for i, (_, row) in enumerate(top_predictions.iterrows(), 1):
                    print(
                        f"{i:2d}. {row['player_name']:25} - {row['predicted_value']:5.1f} "
                        f"(confidence: {row['confidence']:4.2f})"
                    )
            else:
                print("   No predictions available")

        print("\n" + "=" * 80)

    def update_results(self, days_back: int = 1):
        """Update predictions with actual game results."""
        logger.info(f"Updating predictions with results from {days_back} days ago...")

        update_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        self.tonight_predictor.update_predictions_with_results(update_date)

        logger.info("Results update completed!")

    def show_accuracy(self):
        """Show recent prediction accuracy."""
        logger.info("Analyzing recent prediction accuracy...")

        accuracy = self.tonight_predictor.analyze_recent_accuracy(days=14)

        print("\n" + "=" * 60)
        print("PREDICTION ACCURACY (Last 14 Days)")
        print("=" * 60)

        for stat_type, metrics in accuracy.items():
            if metrics["sample_size"] > 0:
                print(
                    f"{stat_type.upper():4} - MAE: {metrics['mae']:5.2f}, "
                    f"RMSE: {metrics['rmse']:5.2f}, "
                    f"Samples: {metrics['sample_size']:3d}"
                )
            else:
                print(f"{stat_type.upper():4} - No data available")

        print("=" * 60)

    def retrain_if_needed(self):
        """Check if models need retraining and retrain if necessary."""
        logger.info("Checking if models need retraining...")

        stat_types = ["pts", "reb", "ast", "stl", "blk"]
        retrained = False

        for stat_type in stat_types:
            should_retrain = self.model_manager.should_retrain_model(stat_type)

            if should_retrain:
                logger.info(
                    f"Retraining {stat_type} model due to performance degradation..."
                )
                try:
                    # Get fresh training data
                    import sqlite3

                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()

                    cursor.execute(
                        """
                        SELECT player_id, COUNT(*) as game_count
                        FROM player_games
                        GROUP BY player_id
                        HAVING game_count >= 20
                        ORDER BY game_count DESC
                        LIMIT 100
                    """
                    )

                    players_with_data = [row[0] for row in cursor.fetchall()]
                    conn.close()

                    if players_with_data:
                        # Create fresh training dataset
                        start_date = "2022-10-01"
                        end_date = datetime.now().strftime("%Y-%m-%d")

                        training_data = self.feature_engineer.create_training_dataset(
                            players_list=players_with_data,
                            start_date=start_date,
                            end_date=end_date,
                            target_stats=[stat_type],
                        )

                        if not training_data.empty:
                            metrics = self.model_manager.train_model(
                                stat_type, training_data
                            )
                            logger.info(
                                f"Retrained {stat_type} model - New MAE: {metrics['val_mae']:.2f}"
                            )
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
                min_games_per_player=5,  # Reduced for broader player coverage
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
    parser = argparse.ArgumentParser(
        description="NBA Stat Predictor with Self-Improvement"
    )

    parser.add_argument(
        "command",
        choices=[
            "collect-data",
            "train",
            "predict",
            "update-results",
            "accuracy",
            "retrain",
            "full-pipeline",
            "enhanced-collect",
            "enhanced-train",
            "force-refresh",
            "backtest",
        ],
        help="Command to execute",
    )

    parser.add_argument(
        "--players-limit",
        type=int,
        default=50,
        help="Number of players to collect data for (default: 50)",
    )

    parser.add_argument(
        "--days-back",
        type=int,
        default=1,
        help="Days back to update results for (default: 1)",
    )

    parser.add_argument(
        "--season", default="2023-24", help="NBA season for backtest (default: 2023-24)"
    )

    parser.add_argument(
        "--db-path",
        default="data/nba_data.db",
        help="Path to SQLite database (default: data/nba_data.db)",
    )

    args = parser.parse_args()

    # Initialize application
    app = NBAStatPredictorApp(args.db_path)

    try:
        if args.command == "collect-data":
            app.collect_data(args.players_limit)

        elif args.command == "train":
            app.train_models()

        elif args.command == "predict":
            app.predict_tonight()

        elif args.command == "update-results":
            app.update_results(args.days_back)

        elif args.command == "accuracy":
            app.show_accuracy()

        elif args.command == "retrain":
            app.retrain_if_needed()

        elif args.command == "full-pipeline":
            logger.info("Running enhanced full pipeline...")

            # Check if we have sufficient data
            import sqlite3

            conn = sqlite3.connect(args.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM player_games")
            game_count = cursor.fetchone()[0]
            
            # Check qualified players
            cursor.execute("""
                SELECT COUNT(DISTINCT player_id) 
                FROM player_games 
                GROUP BY player_id 
                HAVING COUNT(*) >= 25
            """)
            qualified_players = len(cursor.fetchall())
            conn.close()

            if game_count < 5000:
                logger.info("Insufficient data, collecting enhanced dataset...")
                app.collect_data(200)  # Enhanced collection with 200 players

            if qualified_players < 50:
                logger.info("Insufficient qualified players, collecting more data...")
                app.collect_data(300)  # Collect more data for better coverage

            # Train enhanced models
            app.train_models()

            # Make predictions
            app.predict_tonight()

            # Show accuracy if we have historical predictions
            app.show_accuracy()

        elif args.command == "enhanced-collect":
            logger.info("Running enhanced data collection...")
            app.collect_data(200)  # Enhanced collection

        elif args.command == "force-refresh":
            logger.info("Running force refresh data collection...")
            # Force refresh by temporarily modifying the collect_historical_data call
            original_collect = app.data_collector.collect_historical_data
            def force_refresh_collect(*args, **kwargs):
                kwargs['force_refresh'] = True
                return original_collect(*args, **kwargs)
            app.data_collector.collect_historical_data = force_refresh_collect
            app.collect_data(200)
            app.data_collector.collect_historical_data = original_collect

        elif args.command == "enhanced-train":
            logger.info("Running enhanced model training...")
            app.train_models()  # Enhanced training

        elif args.command == "backtest":
            app.run_backtest(args.season)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
