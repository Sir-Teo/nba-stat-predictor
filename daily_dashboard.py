#!/usr/bin/env python3
"""
NBA Daily Dashboard with Self-Improvement System
A comprehensive interface for daily NBA stat predictions with automated model improvement.
"""

import sys
import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

# Add src to path
sys.path.append("src")

from main import NBAStatPredictorApp
from src.data.nba_data_collector import NBADataCollector
from src.models.stat_predictors import ModelManager
from src.predictions.tonight_predictor import TonightPredictor
from src.evaluation.backtester import NBABacktester


def setup_logging(log_dir="logs"):
    """Setup comprehensive logging."""
    os.makedirs(log_dir, exist_ok=True)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # File handler
    file_handler = logging.FileHandler(f"{log_dir}/daily_dashboard.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Setup logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class SelfImprovingDashboard:
    """Daily dashboard with automated self-improvement capabilities."""

    def __init__(self, db_path="data/nba_data.db", config_path="config.yaml"):
        self.db_path = db_path
        self.config_path = config_path
        self.logger = setup_logging()

        # Load configuration
        self.config = self._load_config()

        # Initialize components
        self.app = NBAStatPredictorApp(db_path)
        self.predictor = TonightPredictor(db_path)
        self.backtester = NBABacktester(db_path)

        # Performance tracking
        self.performance_log_path = "data/performance_log.json"
        self.last_retrain_path = "data/last_retrain.json"

        # Create necessary directories
        os.makedirs("data", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("reports", exist_ok=True)

    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(
                f"Config file {self.config_path} not found, using defaults"
            )
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            "models": {
                "retraining": {
                    "performance_threshold": 0.15,
                    "min_days_between_retraining": 7,
                }
            },
            "monitoring": {"accuracy_window_days": 14},
        }

    def daily_workflow(self):
        """Execute the complete daily workflow."""
        self.logger.info("🏀 Starting Daily NBA Prediction Workflow")
        self.logger.info("=" * 60)

        workflow_start = time.time()

        try:
            # Step 1: System Health Check
            self.logger.info("Step 1: System Health Check")
            health_status = self._perform_health_check()

            if not health_status["healthy"]:
                self.logger.error(
                    "System health check failed. Attempting auto-repair..."
                )
                self._attempt_auto_repair(health_status)

            # Step 2: Update Previous Results
            self.logger.info("Step 2: Updating Previous Game Results")
            self._update_previous_results()

            # Step 3: Performance Evaluation
            self.logger.info("Step 3: Evaluating Model Performance")
            performance_metrics = self._evaluate_recent_performance()

            # Step 4: Self-Improvement Check
            self.logger.info("Step 4: Checking for Self-Improvement Needs")
            improvement_needed = self._check_improvement_needed(performance_metrics)

            if improvement_needed:
                self.logger.info("🔄 Triggering Self-Improvement Process")
                self._execute_self_improvement()

            # Step 5: Fresh Data Collection
            self.logger.info("Step 5: Collecting Fresh Data")
            self._collect_fresh_data()

            # Step 6: Generate Today's Predictions
            self.logger.info("Step 6: Generating Today's Predictions")
            predictions = self._generate_todays_predictions()

            # Step 7: Create Daily Report
            self.logger.info("Step 7: Creating Daily Report")
            report = self._create_daily_report(performance_metrics, predictions)

            # Step 8: Display Results
            self.logger.info("Step 8: Displaying Results")
            self._display_daily_summary(report)

            workflow_time = time.time() - workflow_start
            self.logger.info(
                f"✅ Daily workflow completed in {workflow_time:.1f} seconds"
            )

        except Exception as e:
            self.logger.error(f"❌ Daily workflow failed: {e}")
            raise

    def _perform_health_check(self) -> Dict:
        """Perform comprehensive system health check."""
        health_status = {"healthy": True, "issues": [], "warnings": []}

        # Check database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM player_games")
            game_count = cursor.fetchone()[0]
            conn.close()

            if game_count < 100:
                health_status["issues"].append("Insufficient training data")
                health_status["healthy"] = False
            elif game_count < 1000:
                health_status["warnings"].append("Limited training data")

        except Exception as e:
            health_status["issues"].append(f"Database error: {e}")
            health_status["healthy"] = False

        # Check models
        model_count = 0
        if os.path.exists("models"):
            for stat in ["pts", "reb", "ast", "stl", "blk"]:
                for filename in os.listdir("models"):
                    if filename.startswith(f"{stat}_") and filename.endswith(".pkl"):
                        model_count += 1
                        break

        if model_count < 3:
            health_status["issues"].append("Missing trained models")
            health_status["healthy"] = False
        elif model_count < 5:
            health_status["warnings"].append("Some models missing")

        return health_status

    def _attempt_auto_repair(self, health_status: Dict):
        """Attempt to automatically repair system issues."""
        for issue in health_status["issues"]:
            if "training data" in issue.lower():
                self.logger.info("Auto-repair: Collecting training data...")
                self.app.collect_data(players_limit=100)

            elif "missing trained models" in issue.lower():
                self.logger.info("Auto-repair: Training models...")
                self.app.train_models()

    def _update_previous_results(self):
        """Update predictions from previous days with actual results."""
        # Update results from the last 3 days
        for days_back in [1, 2, 3]:
            try:
                self.app.update_results(days_back=days_back)
            except Exception as e:
                self.logger.warning(
                    f"Could not update results from {days_back} days ago: {e}"
                )

    def _evaluate_recent_performance(self) -> Dict:
        """Evaluate recent model performance."""
        try:
            conn = sqlite3.connect(self.db_path)

            # Get recent predictions with actual results
            window_days = self.config["monitoring"]["accuracy_window_days"]
            cutoff_date = (datetime.now() - timedelta(days=window_days)).strftime(
                "%Y-%m-%d"
            )

            query = """
                SELECT stat_type, predicted_value, actual_value, confidence
                FROM predictions 
                WHERE game_date >= ? AND actual_value IS NOT NULL
            """

            predictions_df = pd.read_sql_query(query, conn, params=(cutoff_date,))
            conn.close()

            if predictions_df.empty:
                return {
                    "has_data": False,
                    "message": "No recent predictions with results",
                }

            # Calculate metrics by stat type
            metrics = {}
            for stat_type in ["pts", "reb", "ast", "stl", "blk"]:
                stat_data = predictions_df[predictions_df["stat_type"] == stat_type]

                if len(stat_data) >= 5:  # Minimum samples needed
                    mae = np.mean(
                        np.abs(stat_data["predicted_value"] - stat_data["actual_value"])
                    )
                    rmse = np.sqrt(
                        np.mean(
                            (stat_data["predicted_value"] - stat_data["actual_value"])
                            ** 2
                        )
                    )

                    metrics[stat_type] = {
                        "mae": mae,
                        "rmse": rmse,
                        "samples": len(stat_data),
                        "avg_confidence": stat_data["confidence"].mean(),
                    }

            # Store performance log
            self._update_performance_log(metrics)

            return {
                "has_data": True,
                "metrics": metrics,
                "evaluation_date": datetime.now().isoformat(),
                "window_days": window_days,
            }

        except Exception as e:
            self.logger.error(f"Error evaluating performance: {e}")
            return {"has_data": False, "error": str(e)}

    def _check_improvement_needed(self, performance_metrics: Dict) -> bool:
        """Check if model improvement/retraining is needed."""
        if not performance_metrics.get("has_data"):
            return False

        # Check last retrain date
        try:
            with open(self.last_retrain_path, "r") as f:
                last_retrain_data = json.load(f)
                last_retrain_date = datetime.fromisoformat(last_retrain_data["date"])

                min_days = self.config["models"]["retraining"][
                    "min_days_between_retraining"
                ]
                if (datetime.now() - last_retrain_date).days < min_days:
                    self.logger.info(
                        f"Skipping retrain - last retrain was {(datetime.now() - last_retrain_date).days} days ago"
                    )
                    return False

        except FileNotFoundError:
            # No previous retrain recorded
            pass

        # Check performance degradation
        threshold = self.config["models"]["retraining"]["performance_threshold"]

        # Load historical performance
        try:
            with open(self.performance_log_path, "r") as f:
                perf_history = json.load(f)

            if len(perf_history) < 2:
                return False  # Need at least 2 measurements

            # Compare with performance from 2 weeks ago
            current_metrics = performance_metrics["metrics"]
            baseline_metrics = None

            # Find suitable baseline (2+ weeks old)
            for entry in reversed(perf_history[:-1]):
                entry_date = datetime.fromisoformat(entry["date"])
                if (datetime.now() - entry_date).days >= 14:
                    baseline_metrics = entry["metrics"]
                    break

            if not baseline_metrics:
                return False

            # Check for significant performance degradation
            for stat_type in current_metrics:
                if stat_type not in baseline_metrics:
                    continue

                current_mae = current_metrics[stat_type]["mae"]
                baseline_mae = baseline_metrics[stat_type]["mae"]

                degradation = (current_mae - baseline_mae) / baseline_mae

                if degradation > threshold:
                    self.logger.info(
                        f"Performance degradation detected for {stat_type}: {degradation:.2%}"
                    )
                    return True

        except Exception as e:
            self.logger.warning(f"Could not check performance history: {e}")
            return False

        return False

    def _execute_self_improvement(self):
        """Execute the self-improvement process."""
        self.logger.info("🔄 Executing Self-Improvement Process")

        try:
            # Step 1: Collect more recent data
            self.logger.info("Collecting additional training data...")
            self.app.collect_data(players_limit=150)

            # Step 2: Retrain models
            self.logger.info("Retraining models with updated data...")
            self.app.train_models()

            # Step 3: Record retrain event
            retrain_record = {
                "date": datetime.now().isoformat(),
                "reason": "performance_degradation",
                "trigger": "automated",
            }

            with open(self.last_retrain_path, "w") as f:
                json.dump(retrain_record, f)

            self.logger.info("✅ Self-improvement process completed")

        except Exception as e:
            self.logger.error(f"❌ Self-improvement process failed: {e}")
            raise

    def _collect_fresh_data(self):
        """Collect fresh data for today's predictions."""
        try:
            # Light data collection focusing on active players
            self.app.collect_data(players_limit=75)
        except Exception as e:
            self.logger.warning(f"Fresh data collection had issues: {e}")

    def _generate_todays_predictions(self) -> Dict:
        """Generate predictions for today's games."""
        try:
            predictions_df = self.predictor.get_tonights_predictions()

            if predictions_df.empty:
                return {"has_games": False, "message": "No games scheduled for today"}

            # Organize predictions by stat type
            predictions_summary = {}

            for stat_type in ["pts", "reb", "ast"]:
                top_predictions = self.predictor.get_top_predictions(stat_type, 10)
                predictions_summary[stat_type] = (
                    top_predictions.to_dict("records")
                    if not top_predictions.empty
                    else []
                )

            return {
                "has_games": True,
                "total_predictions": len(predictions_df),
                "predictions_by_stat": predictions_summary,
                "generation_time": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error generating predictions: {e}")
            return {"has_games": False, "error": str(e)}

    def _update_performance_log(self, metrics: Dict):
        """Update the performance log with new metrics."""
        try:
            # Load existing log
            try:
                with open(self.performance_log_path, "r") as f:
                    performance_log = json.load(f)
            except FileNotFoundError:
                performance_log = []

            # Add new entry
            log_entry = {"date": datetime.now().isoformat(), "metrics": metrics}

            performance_log.append(log_entry)

            # Keep only last 60 entries
            performance_log = performance_log[-60:]

            # Save updated log
            with open(self.performance_log_path, "w") as f:
                json.dump(performance_log, f, indent=2)

        except Exception as e:
            self.logger.error(f"Error updating performance log: {e}")

    def _create_daily_report(
        self, performance_metrics: Dict, predictions: Dict
    ) -> Dict:
        """Create comprehensive daily report."""
        report = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "timestamp": datetime.now().isoformat(),
            "system_status": self._get_system_status(),
            "performance": performance_metrics,
            "predictions": predictions,
            "recommendations": self._generate_recommendations(
                performance_metrics, predictions
            ),
        }

        # Save report
        report_path = f"reports/daily_report_{datetime.now().strftime('%Y%m%d')}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        return report

    def _get_system_status(self) -> Dict:
        """Get current system status."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM player_games")
            total_games = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT player_id) FROM player_games")
            unique_players = cursor.fetchone()[0]

            cursor.execute(
                "SELECT COUNT(*) FROM predictions WHERE date >= date('now', '-7 days')"
            )
            recent_predictions = cursor.fetchone()[0]

            conn.close()

            return {
                "total_games": total_games,
                "unique_players": unique_players,
                "recent_predictions": recent_predictions,
                "models_trained": self._count_trained_models(),
            }

        except Exception as e:
            return {"error": str(e)}

    def _count_trained_models(self) -> int:
        """Count number of trained models."""
        if not os.path.exists("models"):
            return 0

        model_count = 0
        for stat in ["pts", "reb", "ast", "stl", "blk"]:
            for filename in os.listdir("models"):
                if filename.startswith(f"{stat}_") and filename.endswith(".pkl"):
                    model_count += 1
                    break

        return model_count

    def _generate_recommendations(
        self, performance_metrics: Dict, predictions: Dict
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        if not predictions.get("has_games"):
            recommendations.append(
                "No games today - consider running backtests or data collection"
            )

        if performance_metrics.get("has_data"):
            metrics = performance_metrics["metrics"]
            for stat_type, data in metrics.items():
                if data["mae"] > 5 and stat_type == "pts":
                    recommendations.append(
                        f"Points prediction accuracy could be improved (MAE: {data['mae']:.1f})"
                    )
                elif data["samples"] < 10:
                    recommendations.append(
                        f"Limited recent data for {stat_type} predictions"
                    )

        if self._count_trained_models() < 5:
            recommendations.append("Consider training all stat prediction models")

        return recommendations

    def _display_daily_summary(self, report: Dict):
        """Display the daily summary in a beautiful format."""
        print("\n" + "🏀" * 30)
        print("   NBA DAILY PREDICTION DASHBOARD")
        print("🏀" * 30)
        print(f"📅 Date: {report['date']}")
        print(f"⏰ Generated: {datetime.now().strftime('%H:%M:%S')}")

        # System Status
        print("\n📊 SYSTEM STATUS")
        print("-" * 40)
        status = report["system_status"]
        print(f"Total Games in DB:     {status.get('total_games', 'N/A'):,}")
        print(f"Unique Players:        {status.get('unique_players', 'N/A')}")
        print(f"Models Trained:        {status.get('models_trained', 'N/A')}/5")
        print(f"Recent Predictions:    {status.get('recent_predictions', 'N/A')}")

        # Performance Metrics
        if report["performance"].get("has_data"):
            print("\n📈 RECENT PERFORMANCE")
            print("-" * 40)
            metrics = report["performance"]["metrics"]
            for stat_type, data in metrics.items():
                print(
                    f"{stat_type.upper():4} - MAE: {data['mae']:5.2f}, "
                    f"Samples: {data['samples']:3d}, "
                    f"Confidence: {data['avg_confidence']:4.2f}"
                )

        # Today's Predictions
        if report["predictions"].get("has_games"):
            print("\n🎯 TODAY'S TOP PREDICTIONS")
            print("-" * 40)

            predictions_data = report["predictions"]["predictions_by_stat"]

            for stat_type in ["pts", "reb", "ast"]:
                if stat_type in predictions_data and predictions_data[stat_type]:
                    print(f"\nTop {stat_type.upper()} Predictions:")

                    for i, pred in enumerate(predictions_data[stat_type][:5], 1):
                        print(
                            f"  {i}. {pred['player_name']:20} "
                            f"{pred['predicted_value']:5.1f} "
                            f"(conf: {pred['confidence']:4.2f})"
                        )
        else:
            print("\n🎯 TODAY'S PREDICTIONS")
            print("-" * 40)
            print("No games scheduled for today")

        # Recommendations
        if report["recommendations"]:
            print("\n💡 RECOMMENDATIONS")
            print("-" * 40)
            for i, rec in enumerate(report["recommendations"], 1):
                print(f"  {i}. {rec}")

        print("\n" + "🏀" * 30)
        print("   Dashboard Complete!")
        print("🏀" * 30 + "\n")

    def quick_status(self):
        """Show quick system status."""
        print("🏀 NBA Predictor - Quick Status")
        print("=" * 40)

        health_status = self._perform_health_check()

        if health_status["healthy"]:
            print("✅ System Status: HEALTHY")
        else:
            print("❌ System Status: ISSUES DETECTED")
            for issue in health_status["issues"]:
                print(f"   - {issue}")

        for warning in health_status["warnings"]:
            print(f"⚠️  - {warning}")

    def performance_summary(self, days: int = 14):
        """Show performance summary for recent period."""
        print(f"\n📈 Performance Summary - Last {days} Days")
        print("-" * 50)

        performance_metrics = self._evaluate_recent_performance()

        if not performance_metrics.get("has_data"):
            print("No recent performance data available")
            return

        metrics = performance_metrics["metrics"]

        print(f"{'Stat':<6} {'MAE':<8} {'RMSE':<8} {'Samples':<8} {'Confidence':<10}")
        print("-" * 50)

        for stat_type, data in metrics.items():
            print(
                f"{stat_type.upper():<6} "
                f"{data['mae']:<8.2f} "
                f"{data['rmse']:<8.2f} "
                f"{data['samples']:<8d} "
                f"{data['avg_confidence']:<10.2f}"
            )


def main():
    """Main function for the daily dashboard."""
    import argparse

    parser = argparse.ArgumentParser(
        description="NBA Daily Dashboard with Self-Improvement"
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="daily",
        choices=["daily", "status", "performance", "setup"],
        help="Command to run",
    )
    parser.add_argument(
        "--days", type=int, default=14, help="Number of days for performance analysis"
    )

    args = parser.parse_args()

    dashboard = SelfImprovingDashboard()

    if args.command == "daily":
        dashboard.daily_workflow()
    elif args.command == "status":
        dashboard.quick_status()
    elif args.command == "performance":
        dashboard.performance_summary(args.days)
    elif args.command == "setup":
        print("Setting up system...")
        dashboard.app.collect_data(players_limit=100)
        dashboard.app.train_models()
        print("Setup complete! You can now run 'python daily_dashboard.py daily'")


if __name__ == "__main__":
    main()
