#!/usr/bin/env python3
"""
Interactive NBA Stat Predictor Dashboard
Allows users to update data and predict player stats against any team.
"""

import sys
import os
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
from tqdm import tqdm

# Add src to path
sys.path.append('src')

from main import NBAStatPredictorApp
from src.data.nba_data_collector import NBADataCollector
from src.models.stat_predictors import ModelManager
from src.data.feature_engineer import FeatureEngineer
from nba_api.stats.static import players, teams

def setup_logging():
    """Setup logging for dashboard."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

class InteractiveNBADashboard:
    """Interactive dashboard for NBA stat predictions with user control."""
    
    def __init__(self, db_path="data/nba_data.db"):
        self.db_path = db_path
        self.app = NBAStatPredictorApp(db_path)
        self.data_collector = NBADataCollector(db_path)
        self.model_manager = ModelManager(db_path)
        self.feature_engineer = FeatureEngineer(db_path)
        self.logger = setup_logging()
        
        # Initialize available teams and players for quick lookup
        self.teams_info = self._get_teams_info()
        self.stat_types = ['pts', 'reb', 'ast', 'stl', 'blk']
        
    def _get_teams_info(self) -> Dict:
        """Get NBA teams information for quick lookup."""
        try:
            nba_teams = teams.get_teams()
            teams_dict = {}
            for team in nba_teams:
                teams_dict[team['full_name'].lower()] = team
                teams_dict[team['abbreviation'].lower()] = team
                teams_dict[team['nickname'].lower()] = team
            return teams_dict
        except Exception as e:
            self.logger.error(f"Error fetching teams info: {e}")
            return {}
    
    def show_welcome_screen(self):
        """Display welcome screen with system status."""
        print("\n" + "🏀" * 30)
        print("   INTERACTIVE NBA STAT PREDICTOR")
        print("🏀" * 30)
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
            
            print(f"📊 System Status:")
            print(f"   • Total games in database: {total_games:,}")
            print(f"   • Unique players tracked: {unique_players}")
            print(f"   • Latest data: {latest_date or 'No data'}")
            
            # Check models
            model_count = 0
            if os.path.exists("models"):
                for stat in self.stat_types:
                    for filename in os.listdir("models"):
                        if filename.startswith(f"{stat}_") and filename.endswith(".pkl"):
                            model_count += 1
                            break
            
            print(f"   • Trained models: {model_count}/{len(self.stat_types)}")
            
        except Exception as e:
            print(f"❌ Error checking system status: {e}")
    
    def show_main_menu(self):
        """Display main menu options."""
        print("\n📋 Main Menu:")
        print("   1. 🔄 Update Data (Fetch Latest NBA Data)")
        print("   2. 🎯 Predict Player Stats vs Team")
        print("   3. 🧠 Train/Retrain Models")
        print("   4. 📊 View System Status")
        print("   5. 📈 View Recent Predictions")
        print("   6. ❌ Exit")
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
                days_old = (datetime.now() - datetime.strptime(latest_date, '%Y-%m-%d')).days
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
            top_players_df = pd.read_sql_query("""
                SELECT player_id, player_name, COUNT(*) as games, AVG(pts) as avg_pts
                FROM player_games 
                GROUP BY player_id, player_name
                HAVING games >= 10
                ORDER BY avg_pts DESC 
                LIMIT 50
            """, conn)
            conn.close()
            
            if top_players_df.empty:
                print("❌ No existing player data found. Consider running full update first.")
                return
            
            player_ids = top_players_df['player_id'].tolist()
            
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
        if confirm != 'y':
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
        print("Note: Enter player names separated by commas, or 'popular' for top players")
        
        player_input = input("Enter player names (or 'popular'): ").strip()
        
        if not player_input:
            print("❌ No players specified.")
            return
        
        try:
            if player_input.lower() == 'popular':
                # Use popular players
                player_ids = self.data_collector.get_popular_players(50)
            else:
                # Parse player names
                player_names = [name.strip() for name in player_input.split(',')]
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
            
            cursor.execute("""
                SELECT DISTINCT player_id, player_name 
                FROM player_games 
                WHERE LOWER(player_name) LIKE LOWER(?)
            """, (f"%{player_name}%",))
            
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
            matches = [p for p in all_players if player_name.lower() in p['full_name'].lower()]
            
            if matches:
                if len(matches) == 1:
                    return matches[0]['id']
                else:
                    print(f"\nFound multiple players matching '{player_name}':")
                    for i, player in enumerate(matches[:10], 1):
                        print(f"   {i}. {player['full_name']}")
                    
                    choice = input("Select player number (or 0 to cancel): ").strip()
                    try:
                        choice_idx = int(choice) - 1
                        if 0 <= choice_idx < len(matches):
                            return matches[choice_idx]['id']
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
    
    def _make_player_vs_team_prediction(self, player_id: int, player_name: str, team_info: Dict):
        """Make prediction for player vs specific team."""
        print(f"\n🧠 Generating predictions...")
        
        try:
            # Load models
            self.model_manager.load_models(self.stat_types)
            
            # Get current date for prediction
            game_date = datetime.now().strftime('%Y-%m-%d')
            
            # Create features for this player (without h2h features for compatibility with existing models)
            features_df = self.feature_engineer.create_features_for_player(
                player_id, game_date, opponent_team_id=team_info['id'], include_h2h_features=False
            )
            
            if features_df.empty:
                print(f"❌ Insufficient data to make predictions for {player_name}")
                print("Consider updating data first or choosing a different player.")
                return
            
            # Make predictions
            predictions_df = self.model_manager.predict_stats(features_df, self.stat_types)
            
            if predictions_df.empty:
                print("❌ Could not generate predictions")
                return
            
            # Display predictions
            self._display_player_predictions(player_name, team_info['full_name'], predictions_df)
            
            # Show recent performance context
            self._show_player_context(player_id, player_name, team_info['id'])
            
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            self.logger.error(f"Prediction error: {e}")
    
    def _display_player_predictions(self, player_name: str, team_name: str, predictions_df: pd.DataFrame):
        """Display formatted predictions."""
        print(f"\n🎯 PREDICTIONS: {player_name} vs {team_name}")
        print("=" * 60)
        
        for stat in self.stat_types:
            pred_col = f'predicted_{stat}'
            conf_col = f'confidence_{stat}'
            
            if pred_col in predictions_df.columns:
                predicted_value = predictions_df[pred_col].iloc[0]
                confidence = predictions_df[conf_col].iloc[0] if conf_col in predictions_df.columns else 0.5
                
                # Format confidence as percentage
                confidence_pct = confidence * 100
                
                # Add confidence indicator
                if confidence >= 0.8:
                    conf_indicator = "🟢 High"
                elif confidence >= 0.6:
                    conf_indicator = "🟡 Medium"
                else:
                    conf_indicator = "🔴 Low"
                
                print(f"   {stat.upper():>5}: {predicted_value:5.1f} (Confidence: {conf_indicator} {confidence_pct:.0f}%)")
        
        print("=" * 60)
    
    def _show_player_context(self, player_id: int, player_name: str, opponent_team_id: int):
        """Show additional context about player's recent performance."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Recent games average
            recent_games = pd.read_sql_query("""
                SELECT pts, reb, ast, stl, blk, game_date
                FROM player_games 
                WHERE player_id = ?
                ORDER BY game_date DESC 
                LIMIT 10
            """, conn, params=[player_id])
            
            if not recent_games.empty:
                print(f"\n📊 {player_name}'s Recent Performance (Last 10 games):")
                for stat in self.stat_types:
                    if stat in recent_games.columns:
                        avg_value = recent_games[stat].mean()
                        print(f"   {stat.upper():>5}: {avg_value:5.1f} avg")
            
            # Head-to-head if available
            h2h_games = pd.read_sql_query("""
                SELECT pts, reb, ast, stl, blk, game_date, matchup
                FROM player_games 
                WHERE player_id = ? AND matchup LIKE ?
                ORDER BY game_date DESC 
                LIMIT 5
            """, conn, params=[player_id, f"%{opponent_team_id}%"])
            
            if not h2h_games.empty:
                print(f"\n📈 Head-to-Head vs {opponent_team_id} (Recent games):")
                for _, game in h2h_games.iterrows():
                    print(f"   {game['game_date']}: {game['pts']} pts, {game['reb']} reb, {game['ast']} ast")
            
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error showing context: {e}")
    
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
                if proceed != 'y':
                    return
            
            # Ask about training mode
            print("\nTraining Options:")
            print("1. Standard training (compatible with current predictions)")
            print("2. Advanced training with head-to-head features (requires retraining)")
            print("3. Back to main menu")
            
            choice = input("\nSelect training mode (1-3): ").strip()
            
            if choice == "1":
                print("🚀 Starting standard model training...")
                print("📊 Training models with progress tracking...\n")
                self.app.train_models()
                print("✅ Standard model training completed!")
            elif choice == "2":
                print("🚀 Starting advanced model training with h2h features...")
                print("⚠️  This will create new models with enhanced features.")
                print("   After training, predictions will include opponent-specific analysis.")
                confirm = input("Continue with advanced training? (y/N): ").strip().lower()
                if confirm == 'y':
                    self._train_advanced_models()
                else:
                    print("Advanced training cancelled.")
            elif choice == "3":
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
                print("❌ No players with sufficient data found.")
                return
            
            print(f"Training advanced models with data from {len(players_with_data)} players")
            
            # Create training dataset with h2h features
            start_date = "2022-10-01"
            end_date = "2024-01-01"
            
            print("📊 Creating enhanced training dataset...")
            training_data = self.feature_engineer.create_training_dataset(
                players_list=players_with_data,
                start_date=start_date,
                end_date=end_date,
                target_stats=['pts', 'reb', 'ast', 'stl', 'blk'],
                include_h2h_features=True
            )
            
            if training_data.empty:
                print("❌ Could not create training dataset")
                return
            
            print(f"Created enhanced training dataset with {len(training_data)} samples")
            
            # Train models for each stat
            stat_types = ['pts', 'reb', 'ast', 'stl', 'blk']
            
            print("\n🏀 Training enhanced models:")
            with tqdm(stat_types, desc="Training Models", ncols=80, 
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
                for stat_type in pbar:
                    try:
                        pbar.set_description(f"Training {stat_type.upper()}")
                        metrics = self.model_manager.train_model(stat_type, training_data)
                        mae = metrics.get('test_mae', metrics.get('val_mae', 0))
                        pbar.write(f"✅ {stat_type.upper()} model trained - MAE: {mae:.2f}")
                    except Exception as e:
                        pbar.write(f"❌ Error training {stat_type} model: {e}")
            
            print("\n✅ Advanced model training completed!")
            print("💡 To use h2h features in predictions, you'll need to modify the prediction code.")
            
        except Exception as e:
            print(f"❌ Error in advanced training: {e}")
    
    def view_recent_predictions(self):
        """View recent predictions made by the system."""
        print("\n📈 RECENT PREDICTIONS")
        print("-" * 40)
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            recent_predictions = pd.read_sql_query("""
                SELECT player_name, stat_type, predicted_value, actual_value, 
                       confidence, game_date, created_at
                FROM predictions 
                WHERE created_at >= date('now', '-7 days')
                ORDER BY created_at DESC 
                LIMIT 20
            """, conn)
            
            if recent_predictions.empty:
                print("No recent predictions found.")
            else:
                print(f"Showing last {len(recent_predictions)} predictions:")
                for _, pred in recent_predictions.iterrows():
                    actual_str = f"({pred['actual_value']:.1f} actual)" if pd.notna(pred['actual_value']) else "(pending)"
                    print(f"   {pred['player_name'][:20]:20} {pred['stat_type'].upper()}: {pred['predicted_value']:5.1f} {actual_str}")
            
            conn.close()
            
        except Exception as e:
            print(f"❌ Error viewing predictions: {e}")
    
    def run_interactive_session(self):
        """Run the main interactive session."""
        self.show_welcome_screen()
        
        while True:
            try:
                self.show_main_menu()
                choice = input("Select option (1-6): ").strip()
                
                if choice == "1":
                    self.handle_data_update()
                elif choice == "2":
                    self.handle_player_prediction()
                elif choice == "3":
                    self.handle_model_training()
                elif choice == "4":
                    self._show_quick_status()
                elif choice == "5":
                    self.view_recent_predictions()
                elif choice == "6":
                    print("\n👋 Thanks for using NBA Stat Predictor!")
                    break
                else:
                    print("❌ Invalid choice. Please select 1-6.")
                
                # Add pause between operations
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
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