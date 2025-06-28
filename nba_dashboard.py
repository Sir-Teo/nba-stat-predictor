#!/usr/bin/env python3
"""
NBA Stat Predictor Dashboard
A comprehensive interface for managing and using the NBA stat prediction system.
"""

import sys
import os
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append('src')

from main import NBAStatPredictorApp
from src.data.nba_data_collector import NBADataCollector
from src.models.stat_predictors import ModelManager

def setup_logging():
    """Setup logging for dashboard."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

class NBADashboard:
    """Interactive dashboard for NBA Stat Predictor."""
    
    def __init__(self, db_path="data/nba_data.db"):
        self.db_path = db_path
        self.app = NBAStatPredictorApp(db_path)
        self.logger = setup_logging()
        
    def show_system_status(self):
        """Show comprehensive system status."""
        print("🏀 NBA Stat Predictor Dashboard")
        print("=" * 60)
        
        # Database status
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM player_games")
            total_games = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT player_id) FROM player_games")
            unique_players = cursor.fetchone()[0]
            
            cursor.execute("SELECT MIN(game_date), MAX(game_date) FROM player_games")
            date_range = cursor.fetchone()
            
            try:
                cursor.execute("SELECT COUNT(*) FROM predictions WHERE date >= date('now', '-7 days')")
                recent_predictions = cursor.fetchone()[0]
            except:
                recent_predictions = 0
            
            conn.close()
            
            print(f"📊 Data Status:")
            print(f"   Total games: {total_games:,}")
            print(f"   Unique players: {unique_players}")
            print(f"   Date range: {date_range[0]} to {date_range[1]}")
            print(f"   Recent predictions (7 days): {recent_predictions}")
            
        except Exception as e:
            print(f"❌ Database error: {e}")
            return False
        
        # Model status
        print(f"\n🧠 Model Status:")
        model_status = {}
        
        for stat in ['pts', 'reb', 'ast', 'stl', 'blk']:
            # Check for any model file for this stat
            model_found = False
            if os.path.exists("models"):
                for filename in os.listdir("models"):
                    if filename.startswith(f"{stat}_") and filename.endswith(".pkl"):
                        model_found = True
                        break
            
            if model_found:
                model_status[stat] = {'trained': True}
                print(f"   {stat.upper()}: ✅ Trained")
            else:
                model_status[stat] = {'trained': False}
                print(f"   {stat.upper()}: ❌ Not trained")
        
        # System health
        print(f"\n🏥 System Health:")
        health_score = 0
        checks = [
            ("Database accessible", total_games > 0),
            ("Sufficient data", total_games > 100),
            ("Models trained", all(model_status[stat]['trained'] for stat in model_status)),
            ("Recent activity", recent_predictions > 0 or total_games > 500),
        ]
        
        for check_name, check_result in checks:
            status = "✅" if check_result else "❌"
            print(f"   {status} {check_name}")
            if check_result:
                health_score += 1
        
        health_percentage = (health_score / len(checks)) * 100
        print(f"\n💚 Overall System Health: {health_percentage:.0f}%")
        
        return health_percentage >= 75
    
    def show_top_players(self, limit=10):
        """Show top players by various stats."""
        print(f"\n🌟 Top {limit} Players by Category")
        print("-" * 50)
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            categories = [
                ("Points", "pts", "AVG(pts)"),
                ("Rebounds", "reb", "AVG(reb)"),
                ("Assists", "ast", "AVG(ast)"),
                ("Steals", "stl", "AVG(stl)"),
                ("Blocks", "blk", "AVG(blk)")
            ]
            
            for category_name, stat_col, avg_expr in categories:
                query = f"""
                    SELECT player_name, {avg_expr} as avg_stat, COUNT(*) as games
                    FROM player_games 
                    WHERE {stat_col} > 0
                    GROUP BY player_name 
                    HAVING games >= 10
                    ORDER BY avg_stat DESC 
                    LIMIT {limit}
                """
                
                df = pd.read_sql_query(query, conn)
                
                print(f"\n🏀 {category_name}:")
                for i, (_, row) in enumerate(df.iterrows(), 1):
                    print(f"   {i:2d}. {row['player_name']:25} {row['avg_stat']:5.1f} ({row['games']} games)")
            
            conn.close()
            
        except Exception as e:
            print(f"❌ Error showing top players: {e}")
    
    def quick_prediction_demo(self):
        """Run a quick prediction demonstration."""
        print(f"\n🎯 Quick Prediction Demo")
        print("-" * 40)
        
        try:
            # Get a popular player for demo
            conn = sqlite3.connect(self.db_path)
            
            demo_player = pd.read_sql_query("""
                SELECT player_name, AVG(pts) as avg_pts, COUNT(*) as games
                FROM player_games 
                GROUP BY player_name 
                HAVING games >= 15
                ORDER BY avg_pts DESC 
                LIMIT 1
            """, conn)
            
            if demo_player.empty:
                print("❌ No suitable players found for demo")
                return
            
            player_name = demo_player.iloc[0]['player_name']
            avg_pts = demo_player.iloc[0]['avg_pts']
            
            print(f"Demo player: {player_name} (Season avg: {avg_pts:.1f} pts)")
            
            # Show recent games
            recent_games = pd.read_sql_query("""
                SELECT game_date, pts, reb, ast
                FROM player_games 
                WHERE player_name = ? 
                ORDER BY game_date DESC 
                LIMIT 5
            """, conn, params=[player_name])
            
            print(f"\nRecent games:")
            for _, game in recent_games.iterrows():
                print(f"   {game['game_date']}: {game['pts']} pts, {game['reb']} reb, {game['ast']} ast")
            
            conn.close()
            
            # Try to make a prediction
            print(f"\n🔮 Prediction attempt...")
            print("Note: This would work best during NBA season with today's games")
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
    
    def suggest_next_actions(self):
        """Suggest next actions based on system state."""
        print(f"\n🎯 Suggested Next Actions")
        print("-" * 40)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM player_games")
            total_games = cursor.fetchone()[0]
            
            try:
                cursor.execute("SELECT COUNT(*) FROM predictions WHERE date >= date('now', '-1 days')")
                recent_predictions = cursor.fetchone()[0]
            except:
                recent_predictions = 0
            
            conn.close()
            
            suggestions = []
            
            if total_games < 100:
                suggestions.append("🔄 Collect more data: make collect")
                suggestions.append("   Need more training data for reliable predictions")
            
            elif total_games < 1000:
                suggestions.append("📈 Collect additional data: make collect")
                suggestions.append("   More data will improve prediction accuracy")
            
            # Check if models exist
            models_trained = 0
            if os.path.exists("models"):
                for stat in ['pts', 'reb', 'ast', 'stl', 'blk']:
                    for filename in os.listdir("models"):
                        if filename.startswith(f"{stat}_") and filename.endswith(".pkl"):
                            models_trained += 1
                            break
            
            if models_trained < 5:
                suggestions.append("🧠 Train models: make train")
                suggestions.append("   Train all prediction models")
            
            if recent_predictions == 0:
                suggestions.append("🎯 Try predictions: make predict")
                suggestions.append("   Test the prediction system")
            
            suggestions.append("📊 Check accuracy: make status")
            suggestions.append("   Monitor system performance")
            
            suggestions.append("🧪 Run backtest: make backtest")
            suggestions.append("   Evaluate historical performance")
            
            if not suggestions:
                suggestions = [
                    "✅ System looks good!",
                    "🎯 Run daily: make predict",
                    "📊 Check status: make status"
                ]
            
            for suggestion in suggestions:
                print(f"   {suggestion}")
                
        except Exception as e:
            print(f"❌ Error generating suggestions: {e}")
    
    def interactive_menu(self):
        """Show interactive menu for common operations."""
        while True:
            print(f"\n🏀 NBA Stat Predictor - Interactive Menu")
            print("=" * 50)
            print("1. 📊 Show system status")
            print("2. 🌟 Show top players")
            print("3. 🎯 Quick prediction demo")
            print("4. 🔄 Collect data")
            print("5. 🧠 Train models")
            print("6. 🎯 Make predictions")
            print("7. 📊 Check accuracy")
            print("8. 🧪 Run backtest")
            print("9. 💡 Get suggestions")
            print("0. 🚪 Exit")
            
            try:
                choice = input("\nSelect option (0-9): ").strip()
                
                if choice == '0':
                    print("👋 Goodbye!")
                    break
                elif choice == '1':
                    self.show_system_status()
                elif choice == '2':
                    self.show_top_players()
                elif choice == '3':
                    self.quick_prediction_demo()
                elif choice == '4':
                    print("🔄 Collecting data...")
                    self.app.collect_data(30)
                elif choice == '5':
                    print("🧠 Training models...")
                    self.app.train_models()
                elif choice == '6':
                    print("🎯 Making predictions...")
                    self.app.predict_tonight()
                elif choice == '7':
                    print("📊 Checking accuracy...")
                    self.app.show_accuracy()
                elif choice == '8':
                    print("🧪 Running backtest...")
                    self.app.run_backtest()
                elif choice == '9':
                    self.suggest_next_actions()
                else:
                    print("❌ Invalid option. Please choose 0-9.")
                    
                if choice != '0':
                    input("\nPress Enter to continue...")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                input("\nPress Enter to continue...")

def main():
    """Main dashboard function."""
    parser = argparse.ArgumentParser(description='NBA Stat Predictor Dashboard')
    parser.add_argument('--db-path', default='data/nba_data.db',
                       help='Path to SQLite database')
    parser.add_argument('--interactive', action='store_true',
                       help='Run interactive menu')
    parser.add_argument('--status-only', action='store_true',
                       help='Show status and exit')
    
    args = parser.parse_args()
    
    dashboard = NBADashboard(args.db_path)
    
    if args.status_only:
        dashboard.show_system_status()
        dashboard.suggest_next_actions()
    elif args.interactive:
        dashboard.interactive_menu()
    else:
        # Default: show comprehensive overview
        healthy = dashboard.show_system_status()
        dashboard.show_top_players(5)
        dashboard.suggest_next_actions()
        
        if healthy:
            dashboard.quick_prediction_demo()
        
        print(f"\n💡 Pro Tips:")
        print(f"   • Run 'python nba_dashboard.py --interactive' for menu")
        print(f"   • Use 'make predict' for daily predictions")
        print(f"   • Check 'make help' for all available commands")
        print(f"   • Open NBA_Predictor_Tutorial.ipynb for interactive guide")

if __name__ == "__main__":
    main() 