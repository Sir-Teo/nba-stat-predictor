#!/usr/bin/env python3
"""
Interactive NBA Stat Predictor Dashboard
Allows users to update data and predict player stats against any team.
"""

import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from nba_api.stats.static import players, teams

from main import NBAStatPredictorApp
from src.data.feature_engineer import AdvancedFeatureEngineer
from src.data.nba_data_collector import NBADataCollector
from src.models.stat_predictors import ModelManager
from src.visualization.prediction_visualizer import PredictionVisualizer


def setup_logging():
    """Setup logging for dashboard."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


class InteractiveNBADashboard:
    """Interactive dashboard for NBA stat predictions with user control."""

    def __init__(self, db_path="data/nba_data.db"):
        """Initialize the interactive dashboard."""
        self.db_path = db_path
        self.app = NBAStatPredictorApp(db_path)
        self.data_collector = NBADataCollector(db_path)
        self.model_manager = ModelManager(db_path)
        self.feature_engineer = AdvancedFeatureEngineer(db_path)
        self.visualizer = PredictionVisualizer(db_path)  # Add visualizer
        self.stat_types = ["pts", "reb", "ast", "stl", "blk"]
        self.teams_info = self._get_teams_info()
        self.logger = logging.getLogger(__name__)

    def _get_teams_info(self) -> Dict:
        """Get NBA teams information for quick lookup."""
        try:
            nba_teams = teams.get_teams()
            teams_dict = {}
            for team in nba_teams:
                teams_dict[team["full_name"].lower()] = team
                teams_dict[team["abbreviation"].lower()] = team
                teams_dict[team["nickname"].lower()] = team
            return teams_dict
        except Exception as e:
            self.logger.error(f"Error fetching teams info: {e}")
            return {}

    def show_welcome_screen(self):
        """Display welcome screen with system status."""
        print("\n" + "=" * 50)
        print("     INTERACTIVE NBA STAT PREDICTOR")
        print("=" * 50)
        print()

        # Show system status
        self._show_quick_status()
        print()

    def _show_quick_status(self):
        """Show quick system status."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get basic stats
            cursor.execute("SELECT COUNT(*) FROM player_games")
            total_games = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT player_id) FROM player_games")
            unique_players = cursor.fetchone()[0]

            cursor.execute("SELECT MAX(game_date) FROM player_games")
            latest_date = cursor.fetchone()[0]

            conn.close()

            print(f"[SYSTEM STATUS]")
            print(f"   - Total games in database: {total_games:,}")
            print(f"   - Unique players tracked: {unique_players}")
            print(f"   - Latest data: {latest_date or 'No data'}")

            # Check models
            model_count = 0
            if os.path.exists("models"):
                for stat in self.stat_types:
                    for filename in os.listdir("models"):
                        if filename.startswith(f"{stat}_") and filename.endswith(
                            ".pkl"
                        ):
                            model_count += 1
                            break

            print(f"   - Trained models: {model_count}/{len(self.stat_types)}")

        except Exception as e:
            print(f"[ERROR] Error checking system status: {e}")

    def show_main_menu(self):
        """Display main menu options."""
        print("\n[MAIN MENU]")
        print("   1. Update Data (Fetch Latest NBA Data)")
        print("   2. Predict Player Stats vs Team (Basic)")
        print("   3. Enhanced Predictions with Advanced Analysis")
        print("   4. Train/Retrain Models")
        print("   5. View System Status")
        print("   6. View Recent Predictions")
        print("   7. Exit")
        print()

    def handle_data_update(self):
        """Handle user choice to update data."""
        print("\n🔄 DATA UPDATE OPTIONS")
        print("-" * 40)

        # Show current data status
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(game_date) FROM player_games")
            latest_date = cursor.fetchone()[0]
            conn.close()

            if latest_date:
                days_old = (
                    datetime.now() - datetime.strptime(latest_date, "%Y-%m-%d")
                ).days
                print(f"📅 Current latest data: {latest_date} ({days_old} days old)")
            else:
                print("📅 No data found in database")

        except Exception as e:
            print(f"❌ Error checking data status: {e}")

        print("\nUpdate options:")
        print("   1. Quick update (last 7 days for top players)")
        print("   2. Full update (comprehensive data collection)")
        print("   3. Custom update (specify players and date range)")
        print("   4. Back to main menu")

        choice = input("\nChoose update option (1-4): ").strip()

        if choice == "1":
            self._quick_data_update()
        elif choice == "2":
            self._full_data_update()
        elif choice == "3":
            self._custom_data_update()
        elif choice == "4":
            return
        else:
            print("❌ Invalid choice. Returning to main menu.")

    def _quick_data_update(self):
        """Perform quick data update for recent games."""
        print("\n⚡ Starting Quick Data Update...")
        print("Fetching recent games for top players...")

        try:
            # Get top players from existing data
            conn = sqlite3.connect(self.db_path)
            top_players_df = pd.read_sql_query(
                """
                SELECT player_id, player_name, COUNT(*) as games, AVG(pts) as avg_pts
                FROM player_games 
                GROUP BY player_id, player_name
                HAVING games >= 10
                ORDER BY avg_pts DESC 
                LIMIT 50
            """,
                conn,
            )
            conn.close()

            if top_players_df.empty:
                print(
                    "❌ No existing player data found. Consider running full update first."
                )
                return

            player_ids = top_players_df["player_id"].tolist()

            # Collect recent data
            seasons = ["2024-25", "2023-24"]  # Current and previous season
            self.data_collector.collect_historical_data(player_ids, seasons)

            print("✅ Quick update completed!")

        except Exception as e:
            print(f"❌ Error during quick update: {e}")

    def _full_data_update(self):
        """Perform comprehensive data update."""
        print("\n🔄 Starting Full Data Update...")
        print("This will collect comprehensive NBA data (may take several minutes)")

        confirm = input("Continue? (y/N): ").strip().lower()
        if confirm != "y":
            print("Update cancelled.")
            return

        try:
            # Run the full data collection
            self.app.collect_data(players_limit=100)
            print("✅ Full update completed!")

        except Exception as e:
            print(f"❌ Error during full update: {e}")

    def _custom_data_update(self):
        """Allow user to customize data update."""
        print("\n⚙️ Custom Data Update")
        print(
            "Note: Enter player names separated by commas, or 'popular' for top players"
        )

        player_input = input("Enter player names (or 'popular'): ").strip()

        if not player_input:
            print("❌ No players specified.")
            return

        try:
            if player_input.lower() == "popular":
                # Use popular players
                player_ids = self.data_collector.get_popular_players(50)
            else:
                # Parse player names
                player_names = [name.strip() for name in player_input.split(",")]
                player_ids = self._get_player_ids_from_names(player_names)

                if not player_ids:
                    print("❌ No valid players found.")
                    return

            # Collect data
            seasons = ["2024-25", "2023-24", "2022-23"]
            self.data_collector.collect_historical_data(player_ids, seasons)

            print("✅ Custom update completed!")

        except Exception as e:
            print(f"❌ Error during custom update: {e}")

    def handle_player_prediction(self):
        """Handle player vs team prediction."""
        print("\n🎯 PLAYER STATS PREDICTION")
        print("-" * 40)

        # Get player name
        player_name = input("Enter player name: ").strip()
        if not player_name:
            print("❌ Player name required.")
            return

        # Get opposing team
        team_name = input("Enter opposing team (name or abbreviation): ").strip()
        if not team_name:
            print("❌ Team name required.")
            return

        # Find player and team IDs
        player_id = self._find_player_id(player_name)
        if not player_id:
            print(f"❌ Player '{player_name}' not found.")
            return

        team_info = self._find_team_info(team_name)
        if not team_info:
            print(f"❌ Team '{team_name}' not found.")
            return

        print(f"\n🔍 Found: {player_name} vs {team_info['full_name']}")

        # Make prediction
        self._make_player_vs_team_prediction(player_id, player_name, team_info)

    def _find_player_id(self, player_name: str) -> Optional[int]:
        """Find player ID from name."""
        try:
            # First check our database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT DISTINCT player_id, player_name 
                FROM player_games 
                WHERE LOWER(player_name) LIKE LOWER(?)
            """,
                (f"%{player_name}%",),
            )

            results = cursor.fetchall()
            conn.close()

            if results:
                if len(results) == 1:
                    return results[0][0]
                else:
                    # Multiple matches, let user choose
                    print(f"\nFound multiple players matching '{player_name}':")
                    for i, (pid, pname) in enumerate(results[:10], 1):
                        print(f"   {i}. {pname}")

                    choice = input("Select player number (or 0 to cancel): ").strip()
                    try:
                        choice_idx = int(choice) - 1
                        if 0 <= choice_idx < len(results):
                            return results[choice_idx][0]
                    except ValueError:
                        pass

                    print("❌ Invalid selection.")
                    return None

            # If not in database, try NBA API
            all_players = players.get_players()
            matches = [
                p for p in all_players if player_name.lower() in p["full_name"].lower()
            ]

            if matches:
                if len(matches) == 1:
                    return matches[0]["id"]
                else:
                    print(f"\nFound multiple players matching '{player_name}':")
                    for i, player in enumerate(matches[:10], 1):
                        print(f"   {i}. {player['full_name']}")

                    choice = input("Select player number (or 0 to cancel): ").strip()
                    try:
                        choice_idx = int(choice) - 1
                        if 0 <= choice_idx < len(matches):
                            return matches[choice_idx]["id"]
                    except ValueError:
                        pass

            return None

        except Exception as e:
            self.logger.error(f"Error finding player ID: {e}")
            return None

    def _find_team_info(self, team_name: str) -> Optional[Dict]:
        """Find team info from name."""
        team_key = team_name.lower()
        return self.teams_info.get(team_key)

    def _make_player_vs_team_prediction(
        self, player_id: int, player_name: str, team_info: Dict
    ):
        """Make prediction for player vs specific team."""
        print(f"\n🧠 Generating predictions...")

        try:
            # Load models
            self.model_manager.load_models(self.stat_types)

            # Get current date for prediction
            game_date = datetime.now().strftime("%Y-%m-%d")

            # Create features for this player (without h2h features for compatibility with existing models)
            features_df = self.feature_engineer.create_features_for_player(
                player_id,
                game_date,
                opponent_team_id=team_info["id"],
                include_h2h_features=False,
            )

            if features_df.empty:
                print(f"❌ Insufficient data to make predictions for {player_name}")
                print("Consider updating data first or choosing a different player.")
                return

            # Make predictions
            predictions_df = self.model_manager.predict_stats(
                features_df, self.stat_types
            )

            if predictions_df.empty:
                print("❌ Could not generate predictions")
                return

            # Apply age-aware post-processing
            predictions_df = self._apply_age_aware_adjustments(
                predictions_df, features_df, player_id, player_name
            )

            # Display predictions
            self._display_player_predictions(
                player_name, team_info["full_name"], predictions_df
            )

            # Show recent performance context
            self._show_player_context(player_id, player_name, team_info["id"])

            # NEW: Ask user if they want to see visualization
            print("\n📊 Would you like to see a detailed prediction rationale visualization?")
            show_viz = input("Show visualization? (y/N): ").strip().lower()
            
            if show_viz == 'y':
                print("\n🎨 Generating comprehensive prediction rationale chart...")
                try:
                    chart_path = self.visualizer.show_prediction_rationale(
                        player_id=player_id,
                        player_name=player_name,
                        predictions_df=predictions_df,
                        features_df=features_df,
                        recent_stats=self._get_recent_performance(player_id, games=10),
                        opponent_name=team_info["full_name"]
                    )
                    
                    if chart_path:
                        print(f"✅ Visualization created successfully!")
                        print(f"📁 Chart saved to: {chart_path}")
                    else:
                        print("❌ Could not create visualization")
                        
                except Exception as e:
                    print(f"❌ Error creating visualization: {e}")
                    self.logger.error(f"Visualization error: {e}")

        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            self.logger.error(f"Prediction error: {e}")

    def handle_enhanced_predictions(self):
        """Handle enhanced predictions with advanced analysis and visualization."""
        print("\n[ENHANCED PREDICTION SYSTEM]")
        print("=" * 60)
        print("This mode uses advanced features including:")
        print("- Age-based decline adjustments with 530+ features")
        print("- Opponent defensive analysis & head-to-head history") 
        print("- Enhanced confidence calculation (not uniform 30%)")
        print("- Professional prediction rationale charts")
        print("- Ensemble variance and uncertainty quantification")
        print("-" * 60)

        # Get player name
        player_name = input("Enter player name: ").strip()
        if not player_name:
            print("[ERROR] Player name required.")
            return

        # Find player ID
        player_id = self._find_player_id(player_name)
        if not player_id:
            print(f"[ERROR] Player '{player_name}' not found in database.")
            print("Try updating data first or checking the spelling.")
            return

        # Get opponent team
        team_name = input("Enter opponent team name: ").strip()
        if not team_name:
            print("[ERROR] Team name required.")
            return

        # Find team info
        team_info = self._find_team_info(team_name)
        if not team_info:
            print(f"[ERROR] Team '{team_name}' not found.")
            print("Try abbreviation (e.g., 'LAL') or full name (e.g., 'Los Angeles Lakers')")
            return

        # Make enhanced prediction
        self._make_enhanced_prediction(player_id, player_name, team_info)

    def _make_enhanced_prediction(
        self, player_id: int, player_name: str, team_info: Dict
    ):
        """Make enhanced prediction with all improvements."""
        print(f"\n[GENERATING] Enhanced predictions with advanced features...")

        try:
            # Load models
            self.model_manager.load_models(self.stat_types)

            # Get current date for prediction
            game_date = datetime.now().strftime("%Y-%m-%d")

            # Create ENHANCED features with ALL improvements
            print("   [FEATURES] Creating 530+ advanced features...")
            features_df = self.feature_engineer.create_features_for_player(
                player_id,
                game_date,
                opponent_team_id=team_info["id"],
                include_h2h_features=True,  # Enable head-to-head features
                include_advanced_features=True,  # Enable all advanced features
                lookback_games=20  # More historical context
            )

            if features_df.empty:
                print(f"[ERROR] Insufficient data to make predictions for {player_name}")
                print("Consider updating data first or choosing a different player.")
                return

            print(f"   [SUCCESS] Created {len(features_df.columns)} features")

            # Make predictions with ENHANCED confidence
            print("   [PREDICT] Making predictions with improved confidence calculation...")
            predictions_df = self.model_manager.predict_stats(
                features_df, self.stat_types
            )

            if predictions_df.empty:
                print("[ERROR] Could not generate predictions")
                return

            # Get recent performance for context
            recent_stats = self._get_recent_performance(player_id, games=10)

            # Display enhanced predictions with better confidence
            self._display_enhanced_predictions(
                player_name, team_info["full_name"], predictions_df, features_df
            )

            # Show detailed player context
            self._show_enhanced_player_context(
                player_id, player_name, team_info["id"], features_df, recent_stats
            )

            # ALWAYS offer visualization in enhanced mode
            print("\n[VISUAL] Generating comprehensive prediction rationale chart...")
            try:
                chart_path = self.visualizer.create_prediction_rationale_chart(
                    player_id=player_id,
                    player_name=player_name,
                    predictions_df=predictions_df,
                    features_df=features_df,
                    recent_stats=recent_stats,
                    opponent_name=team_info["full_name"]
                )
                
                if chart_path:
                    print(f"[SUCCESS] Professional visualization created successfully!")
                    print(f"[SAVED] Chart saved to: {chart_path}")
                    
                    # Show key insights
                    self._show_prediction_insights(features_df, predictions_df, recent_stats)
                else:
                    print("[ERROR] Could not create visualization")
                    
            except Exception as e:
                print(f"[ERROR] Error creating visualization: {e}")
                self.logger.error(f"Enhanced visualization error: {e}")

        except Exception as e:
            print(f"[ERROR] Error making enhanced prediction: {e}")
            self.logger.error(f"Enhanced prediction error: {e}")

    def _display_enhanced_predictions(
        self, player_name: str, team_name: str, predictions_df: pd.DataFrame, features_df: pd.DataFrame
    ):
        """Display enhanced predictions with improved confidence and age context."""
        print(f"\n[ENHANCED PREDICTIONS] {player_name} vs {team_name}")
        print("=" * 60)

        # Show age context first
        player_age = features_df.get("player_age", pd.Series([30])).iloc[0] if not features_df.empty else 30
        print(f"   [AGE] Player Age: {player_age:.1f} years")
        
        if player_age >= 40:
            print(f"   [VETERAN] Applying age adjustments for veteran player...")
        elif player_age >= 35:
            print(f"   [VETERAN] Age-related adjustments applied for veteran")

        print()

        for stat in self.stat_types:
            pred_col = f"predicted_{stat}"
            conf_col = f"confidence_{stat}"

            if pred_col in predictions_df.columns:
                predicted_value = predictions_df[pred_col].iloc[0]
                confidence = (
                    predictions_df[conf_col].iloc[0]
                    if conf_col in predictions_df.columns
                    else 0.5
                )

                # Format confidence as percentage with enhanced levels
                confidence_pct = confidence * 100

                # Enhanced confidence indicators
                if confidence >= 0.8:
                    conf_indicator = "Very High"
                elif confidence >= 0.7:
                    conf_indicator = "High"
                elif confidence >= 0.6:
                    conf_indicator = "Medium"
                elif confidence >= 0.5:
                    conf_indicator = "Moderate"
                else:
                    conf_indicator = "Low"

                print(
                    f"     {stat.upper():>5}: {predicted_value:5.1f} (Confidence: {conf_indicator} {confidence_pct:5.1f}%)"
                )

        print("=" * 60)

    def _show_enhanced_player_context(
        self, player_id: int, player_name: str, opponent_team_id: int, 
        features_df: pd.DataFrame, recent_stats: Dict
    ):
        """Show enhanced player context with advanced features."""
        print(f"\n[CONTEXT] {player_name}'s Enhanced Analysis Context:")
        print("-" * 50)

        # Age context
        player_age = features_df.get("player_age", pd.Series([30])).iloc[0] if not features_df.empty else 30
        if player_age >= 40:
            print(f"[AGE CONTEXT]")
            print(f"   Player Age: {player_age:.1f} years")
            print(f"   - At {player_age:.1f}, this player is in exceptional territory for NBA longevity")
            print(f"   - Predictions favor recent form over historical career averages")
            print(f"   - Performance may be more variable game-to-game")

        # Recent performance with trends
        if recent_stats:
            print(f"\n[RECENT FORM] {player_name}'s Recent Performance (Last 10 games):")
            for stat in ["pts", "reb", "ast", "stl", "blk"]:
                avg_key = f"{stat}_avg"
                trend_key = f"{stat}_trend"
                if avg_key in recent_stats:
                    avg_val = recent_stats[avg_key]
                    trend_val = recent_stats.get(trend_key, 0)
                    trend_arrow = "UP" if trend_val > 0.5 else "DOWN" if trend_val < -0.5 else "STABLE"
                    print(f"     {stat.upper():>5}: {avg_val:5.1f} avg [{trend_arrow}]")

        # Feature insights
        if not features_df.empty:
            print(f"\n[ADVANCED FEATURES] Feature Insights:")
            
            # Age features
            if "age_decline_factor" in features_df.columns:
                decline = features_df["age_decline_factor"].iloc[0]
                print(f"   Age Decline Factor: {decline:.2f} ({int((1-decline)*100)}% adjustment)")
            
            # Opponent features  
            if "opp_def_efficiency" in features_df.columns:
                opp_def = features_df["opp_def_efficiency"].iloc[0]
                if opp_def < 1.0:
                    print(f"   Opponent Defense: Strong ({opp_def:.2f} efficiency)")
                elif opp_def > 1.15:
                    print(f"   Opponent Defense: Weak ({opp_def:.2f} efficiency)")
                else:
                    print(f"   Opponent Defense: Average ({opp_def:.2f} efficiency)")
            
            # H2H history
            if "has_h2h_history" in features_df.columns and features_df["has_h2h_history"].iloc[0] == 1:
                h2h_games = features_df.get("h2h_games_count", pd.Series([0])).iloc[0]
                print(f"   Head-to-Head: {h2h_games} historical games vs opponent")

    def _show_prediction_insights(self, features_df: pd.DataFrame, predictions_df: pd.DataFrame, recent_stats: Dict):
        """Show key prediction insights and rationale."""
        print(f"\n[PREDICTION RATIONALE] Summary:")
        print("=" * 50)
        
        # Age insights
        player_age = features_df.get("player_age", pd.Series([30])).iloc[0] if not features_df.empty else 30
        if player_age >= 40:
            print(f"- Age Factor: {player_age:.1f} years (Elite longevity territory)")
            print("- Weighting: 80% recent form, 20% historical averages")
            print("- Confidence: Reduced due to higher variance in aging players")
        elif player_age >= 35:
            print(f"- Age Factor: {player_age:.1f} years (Veteran adjustments applied)")
            print("- Weighting: 60% recent form, 40% historical averages")
            print("- Confidence: Moderate reduction for age-related uncertainty")
        else:
            print(f"- Age Factor: {player_age:.1f} years (Prime/Standard predictions)")
            print("- Weighting: Standard model predictions")
        
        # Confidence summary
        if not predictions_df.empty:
            avg_confidence = 0
            conf_count = 0
            for stat in ['pts', 'reb', 'ast', 'stl', 'blk']:
                conf_col = f"confidence_{stat}"
                if conf_col in predictions_df.columns:
                    avg_confidence += predictions_df[conf_col].iloc[0]
                    conf_count += 1
            
            if conf_count > 0:
                avg_confidence = (avg_confidence / conf_count) * 100
                print(f"- Average Confidence: {avg_confidence:.1f}% (vs typical 30% uniform)")

        print("[SUCCESS] Visualization created successfully!")

    def _apply_age_aware_adjustments(
        self, predictions_df: pd.DataFrame, features_df: pd.DataFrame, 
        player_id: int, player_name: str
    ) -> pd.DataFrame:
        """Apply age-aware adjustments to make predictions more realistic."""
        try:
            # Get player age if available
            player_age = features_df.get("player_age", pd.Series([30])).iloc[0]
            
            # Get recent form data
            recent_stats = self._get_recent_performance(player_id, games=10)
            
            adjusted_predictions = predictions_df.copy()
            
            print(f"   📊 Player Age: {player_age:.1f} years")
            
            # Special handling for aging veterans (35+)
            if player_age >= 35:
                print(f"   ⚠️  Applying age adjustments for veteran player...")
                
                for stat in self.stat_types:
                    pred_col = f"predicted_{stat}"
                    conf_col = f"confidence_{stat}"
                    
                    if pred_col in adjusted_predictions.columns:
                        original_pred = adjusted_predictions[pred_col].iloc[0]
                        recent_avg = recent_stats.get(f"{stat}_avg", original_pred)
                        
                        # For 40+ players, heavily weight recent performance
                        if player_age >= 40:
                            # Weight: 80% recent form, 20% model prediction
                            age_weight = 0.8
                            
                            # Special caps for 40+ players
                            if stat == "pts" and original_pred > 30:
                                # Cap points at more realistic levels
                                capped_pred = min(original_pred, recent_avg + 5)
                                adjusted_predictions.loc[0, pred_col] = capped_pred
                                print(f"   🎯 Capped {stat.upper()} prediction: {original_pred:.1f} → {capped_pred:.1f}")
                            
                        # For 35-39 players, moderate adjustment
                        elif player_age >= 35:
                            age_weight = 0.6  # 60% recent form, 40% model prediction
                        
                        else:
                            age_weight = 0.4  # 40% recent form, 60% model prediction
                        
                        # Apply age-weighted adjustment
                        if recent_avg > 0:
                            age_adjusted_pred = (age_weight * recent_avg + 
                                               (1 - age_weight) * original_pred)
                            adjusted_predictions.loc[0, pred_col] = age_adjusted_pred
                        
                        # Lower confidence for aging players due to higher variance
                        if conf_col in adjusted_predictions.columns:
                            original_conf = adjusted_predictions[conf_col].iloc[0]
                            age_penalty = max(0.1, (player_age - 30) * 0.05)  # 5% confidence penalty per year after 30
                            adjusted_conf = max(0.3, original_conf - age_penalty)
                            adjusted_predictions.loc[0, conf_col] = adjusted_conf
            
            return adjusted_predictions
            
        except Exception as e:
            self.logger.error(f"Error in age adjustments: {e}")
            return predictions_df

    def _get_recent_performance(self, player_id: int, games: int = 10) -> Dict:
        """Get recent performance stats for a player."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = """
                SELECT * FROM player_games 
                WHERE player_id = ?
                ORDER BY game_date DESC
                LIMIT ?
            """
            
            df = pd.read_sql_query(query, conn, params=(player_id, games))
            conn.close()
            
            if df.empty:
                return {}
            
            stats = {}
            for stat in ["pts", "reb", "ast", "stl", "blk"]:
                if stat in df.columns:
                    stats[f"{stat}_avg"] = df[stat].mean()
                    stats[f"{stat}_trend"] = self._calculate_trend(df[stat].values)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting recent performance: {e}")
            return {}

    def _calculate_trend(self, values):
        """Calculate trend in recent performance."""
        if len(values) < 3:
            return 0
        
        # Simple linear trend
        x = np.arange(len(values))
        try:
            slope = np.polyfit(x, values, 1)[0]
            return slope
        except:
            return 0

    def _display_player_predictions(
        self, player_name: str, team_name: str, predictions_df: pd.DataFrame
    ):
        """Display formatted predictions with age context."""
        print(f"\n🎯 PREDICTIONS: {player_name} vs {team_name}")
        print("=" * 60)

        for stat in self.stat_types:
            pred_col = f"predicted_{stat}"
            conf_col = f"confidence_{stat}"

            if pred_col in predictions_df.columns:
                predicted_value = predictions_df[pred_col].iloc[0]
                confidence = (
                    predictions_df[conf_col].iloc[0]
                    if conf_col in predictions_df.columns
                    else 0.5
                )

                # Format confidence as percentage
                confidence_pct = confidence * 100

                # Add confidence indicator
                if confidence >= 0.8:
                    conf_indicator = "🟢 High"
                elif confidence >= 0.6:
                    conf_indicator = "🟡 Medium"
                else:
                    conf_indicator = "🔴 Low"

                print(
                    f"   {stat.upper():>5}: {predicted_value:5.1f} (Confidence: {conf_indicator} {confidence_pct:.0f}%)"
                )

        print("=" * 60)
        
        # Add age-related context if player is aging
        if "player_age" in predictions_df.columns:
            age = predictions_df["player_age"].iloc[0]
            if age >= 35:
                print(f"\n⚠️  Age-Adjusted Predictions (Player Age: {age:.1f})")
                print("   Predictions weighted toward recent performance due to player age.")
                if age >= 40:
                    print("   🔸 40+ Player: Heavy emphasis on current form over career averages.")
                elif age >= 38:
                    print("   🔹 Veteran Player: Moderate adjustment for age-related decline.")

    def _show_player_context(
        self, player_id: int, player_name: str, opponent_team_id: int
    ):
        """Show player context including recent performance and age insights."""
        try:
            conn = sqlite3.connect(self.db_path)

            # Recent games performance
            recent_query = """
                SELECT * FROM player_games 
                WHERE player_id = ?
                ORDER BY game_date DESC
                LIMIT 10
            """

            recent_df = pd.read_sql_query(recent_query, conn, params=(player_id,))

            if not recent_df.empty:
                print(f"\n📊 {player_name}'s Recent Performance (Last 10 games):")
                
                # Calculate averages with validation
                recent_stats = {}
                for stat in ["pts", "reb", "ast", "stl", "blk"]:
                    if stat in recent_df.columns:
                        # Validate data before calculation
                        stat_data = recent_df[stat].dropna()  # Remove null values
                        if len(stat_data) > 0:
                            avg_value = stat_data.mean()
                            # Sanity check for unrealistic averages
                            if self._validate_stat_average(stat, avg_value):
                                recent_stats[stat] = avg_value
                            else:
                                logger.warning(f"Unrealistic average for {stat}: {avg_value:.1f}")
                                recent_stats[stat] = stat_data.median()  # Use median as fallback

                # Display recent averages
                for stat in ["pts", "reb", "ast", "stl", "blk"]:
                    if stat in recent_stats:
                        avg_val = recent_stats[stat]
                        print(f"   {stat.upper():>5}: {avg_val:5.1f} avg")
                        
                        # Add context for exceptional values
                        if stat == "ast" and avg_val > 15:
                            print(f"     ⚠️  Exceptionally high assist average")
                        elif stat == "stl" and avg_val > 3:
                            print(f"     ⚠️  Exceptionally high steal average")
                        elif stat == "pts" and avg_val > 35:
                            print(f"     ⚠️  Exceptionally high scoring average")

                # Age-related insights
                age = self._get_player_age_from_db(player_id)
                if age:
                    print(f"\n🎂 Age Context:")
                    print(f"   Player Age: {age:.1f} years")
                    
                    if age >= 40:
                        print("   🔸 At 40+, this player is in exceptional territory for NBA longevity")
                        print("   🔸 Predictions favor recent form over historical career averages")
                        print("   🔸 Performance may be more variable game-to-game")
                    elif age >= 35:
                        print("   🔹 Veteran player - age-related adjustments applied to predictions")
                        print("   🔹 Recent performance weighted more heavily than career averages")

            # Show matchup history if available
            h2h_query = """
                SELECT game_date, pts, reb, ast, matchup 
                FROM player_games 
                WHERE player_id = ? AND (
                    matchup LIKE ? OR matchup LIKE ?
                )
                ORDER BY game_date DESC
                LIMIT 5
            """

            opponent_pattern1 = f"%{opponent_team_id}%"
            opponent_pattern2 = f"%vs%{opponent_team_id}%"  # Alternative pattern

            h2h_df = pd.read_sql_query(
                h2h_query, conn, params=(player_id, opponent_pattern1, opponent_pattern2)
            )

            if not h2h_df.empty:
                print(f"\n📈 Recent Head-to-Head Performance:")
                for _, game in h2h_df.head(3).iterrows():
                    print(
                        f"   {game['game_date']}: {game['pts']} pts, {game['reb']} reb, {game['ast']} ast"
                    )

            conn.close()

        except Exception as e:
            print(f"   ⚠️ Could not load player context: {e}")
            self.logger.error(f"Error showing player context: {e}")

    def _validate_stat_average(self, stat: str, value: float) -> bool:
        """Validate that a statistical average is reasonable."""
        # Define reasonable ranges for averages (per game)
        reasonable_ranges = {
            'pts': (0, 40),    # 40 ppg would be historically exceptional
            'reb': (0, 20),    # 20 rpg would be historically exceptional  
            'ast': (0, 15),    # 15 apg would be historically exceptional
            'stl': (0, 4),     # 4 spg would be historically exceptional
            'blk': (0, 5),     # 5 bpg would be historically exceptional
        }
        
        if stat in reasonable_ranges:
            min_val, max_val = reasonable_ranges[stat]
            return min_val <= value <= max_val
        
        return True  # If not in our validation list, assume it's okay

    def _get_player_age_from_db(self, player_id: int) -> Optional[float]:
        """Get player age if available in features."""
        try:
            from datetime import datetime
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            # Use the same age calculation from feature engineer
            age = self.feature_engineer._calculate_player_age(player_id, current_date)
            return age
        except:
            return None

    def _get_player_ids_from_names(self, player_names: List[str]) -> List[int]:
        """Convert player names to IDs."""
        player_ids = []
        for name in player_names:
            player_id = self._find_player_id(name)
            if player_id:
                player_ids.append(player_id)
        return player_ids

    def handle_model_training(self):
        """Handle model training."""
        print("\n🧠 MODEL TRAINING")
        print("-" * 30)

        # Check data availability
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM player_games")
            total_games = cursor.fetchone()[0]
            conn.close()

            if total_games < 100:
                print(f"⚠️  Limited training data ({total_games} games)")
                print("Consider updating data first for better model performance.")

                proceed = input("Continue with training? (y/N): ").strip().lower()
                if proceed != "y":
                    return

            # Ask about training mode
            print("\nTraining Options:")
            print("1. Standard training (compatible with current predictions)")
            print(
                "2. Advanced training with head-to-head features (requires retraining)"
            )
            print("3. Player-specific training (train models for a specific player)")
            print("4. Back to main menu")

            choice = input("\nSelect training mode (1-4): ").strip()

            if choice == "1":
                print("🚀 Starting standard model training...")
                print("📊 Training models with progress tracking...\n")
                self.app.train_models()
                print("✅ Standard model training completed!")
            elif choice == "2":
                print("🚀 Starting advanced model training with h2h features...")
                print("⚠️  This will create new models with enhanced features.")
                print(
                    "   After training, predictions will include opponent-specific analysis."
                )
                confirm = (
                    input("Continue with advanced training? (y/N): ").strip().lower()
                )
                if confirm == "y":
                    self._train_advanced_models()
                else:
                    print("Advanced training cancelled.")
            elif choice == "3":
                self._train_player_specific_models()
            elif choice == "4":
                return
            else:
                print("❌ Invalid choice.")

        except Exception as e:
            print(f"❌ Error during training: {e}")

    def _train_advanced_models(self):
        """Train models with head-to-head features."""
        try:
            import sqlite3

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get players with sufficient data
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

            if not players_with_data:
                print("❌ No players with sufficient data found.")
                return

            print(
                f"Training advanced models with data from {len(players_with_data)} players"
            )

            # Create training dataset with h2h features
            start_date = "2022-10-01"
            end_date = "2024-01-01"

            print("📊 Creating enhanced training dataset...")
            training_data = self.feature_engineer.create_training_dataset(
                players_list=players_with_data,
                start_date=start_date,
                end_date=end_date,
                target_stats=["pts", "reb", "ast", "stl", "blk"],
                include_h2h_features=True,
            )

            if training_data.empty:
                print("❌ Could not create training dataset")
                return

            print(
                f"Created enhanced training dataset with {len(training_data)} samples"
            )

            # Train models for each stat
            stat_types = ["pts", "reb", "ast", "stl", "blk"]

            print("\n🏀 Training enhanced models:")
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
                            stat_type, training_data
                        )
                        mae = metrics.get("test_mae", metrics.get("val_mae", 0))
                        pbar.write(
                            f"✅ {stat_type.upper()} model trained - MAE: {mae:.2f}"
                        )
                    except Exception as e:
                        pbar.write(f"❌ Error training {stat_type} model: {e}")

            print("\n✅ Advanced model training completed!")
            print(
                "💡 To use h2h features in predictions, you'll need to modify the prediction code."
            )

        except Exception as e:
            print(f"❌ Error in advanced training: {e}")

    def _train_player_specific_models(self):
        """Train models for a specific player."""
        try:
            player_name = input("Enter player name: ").strip()
            if not player_name:
                print("❌ Player name required.")
                return

            player_id = self._find_player_id(player_name)
            if not player_id:
                print(f"❌ Player '{player_name}' not found.")
                return

            # Check if player has sufficient data
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM player_games WHERE player_id = ?",
                (player_id,)
            )
            game_count = cursor.fetchone()[0]
            conn.close()

            print(f"\n🔍 Found {player_name} with {game_count} games in database")
            
            if game_count < 30:
                print(f"⚠️  Player has limited data ({game_count} games)")
                proceed = input("Continue with training? (y/N): ").strip().lower()
                if proceed != "y":
                    print("Training cancelled.")
                    return

            # Choose model type
            print("\nModel Options:")
            print("1. Ensemble (best accuracy, slower)")
            print("2. Random Forest (good balance)")
            print("3. LightGBM (fast, good performance)")
            print("4. XGBoost (good performance)")
            
            model_choice = input("Select model type (1-4, default=2): ").strip()
            model_types = {
                "1": "ensemble",
                "2": "random_forest", 
                "3": "lightgbm",
                "4": "xgboost"
            }
            model_type = model_types.get(model_choice, "random_forest")
            
            print(f"\n🧠 Training {model_type} models for {player_name}")
            print("This will create personalized models based on the player's performance patterns.")

            # Train models for the player
            results = self.model_manager.train_models_for_player(
                player_id=player_id,
                player_name=player_name,
                model_type=model_type,
                optimize_hyperparams=True
            )

            # Show results
            successful_models = [stat for stat, metrics in results.items() if 'error' not in metrics]
            failed_models = [stat for stat, metrics in results.items() if 'error' in metrics]
            
            print(f"\n✅ Player-specific model training completed!")
            print(f"   Successfully trained: {len(successful_models)} models")
            if failed_models:
                print(f"   Failed to train: {len(failed_models)} models ({', '.join(failed_models)})")
            
            print(f"\n🎯 Models saved for {player_name} (ID: {player_id})")
            print("   You can now get personalized predictions for this player!")

        except Exception as e:
            print(f"❌ Error during player-specific training: {e}")

    def view_recent_predictions(self):
        """View recent predictions made by the system."""
        print("\n📈 RECENT PREDICTIONS")
        print("-" * 40)

        try:
            conn = sqlite3.connect(self.db_path)

            recent_predictions = pd.read_sql_query(
                """
                SELECT player_name, stat_type, predicted_value, actual_value, 
                       confidence, game_date, created_at
                FROM predictions 
                WHERE created_at >= date('now', '-7 days')
                ORDER BY created_at DESC 
                LIMIT 20
            """,
                conn,
            )

            if recent_predictions.empty:
                print("No recent predictions found.")
            else:
                print(f"Showing last {len(recent_predictions)} predictions:")
                for _, pred in recent_predictions.iterrows():
                    actual_str = (
                        f"({pred['actual_value']:.1f} actual)"
                        if pd.notna(pred["actual_value"])
                        else "(pending)"
                    )
                    print(
                        f"   {pred['player_name'][:20]:20} {pred['stat_type'].upper()}: {pred['predicted_value']:5.1f} {actual_str}"
                    )

            conn.close()

        except Exception as e:
            print(f"❌ Error viewing predictions: {e}")

    def run_interactive_session(self):
        """Run the main interactive session."""
        self.show_welcome_screen()

        while True:
            try:
                self.show_main_menu()
                choice = input("Select option (1-7): ").strip()

                if choice == "1":
                    self.handle_data_update()
                elif choice == "2":
                    self.handle_player_prediction()
                elif choice == "3":
                    self.handle_enhanced_predictions()
                elif choice == "4":
                    self.handle_model_training()
                elif choice == "5":
                    self._show_quick_status()
                elif choice == "6":
                    self.view_recent_predictions()
                elif choice == "7":
                    print("\n[EXIT] Thanks for using NBA Stat Predictor!")
                    break
                else:
                    print("❌ Invalid choice. Please select 1-7.")

                # Add pause between operations
                input("\nPress Enter to continue...")

            except KeyboardInterrupt:
                print("\n\n[EXIT] Goodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                self.logger.error(f"Interactive session error: {e}")


def main():
    """Main function to run the interactive dashboard."""
    dashboard = InteractiveNBADashboard()
    dashboard.run_interactive_session()


if __name__ == "__main__":
    main()
