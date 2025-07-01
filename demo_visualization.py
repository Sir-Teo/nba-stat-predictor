#!/usr/bin/env python3
"""
Demo script to showcase the NBA prediction visualization with rationales.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.data.feature_engineer import FeatureEngineer
from src.models.stat_predictors import ModelManager
from src.visualization import PredictionVisualizer
import pandas as pd

def demo_prediction_visualization():
    """Demo the prediction visualization for LeBron James vs Warriors."""
    
    print("🏀 NBA Prediction Visualization Demo")
    print("=" * 50)
    print("🎯 Player: LeBron James vs Golden State Warriors")
    print("📊 Generating comprehensive rationale visualization...")
    
    try:
        # Initialize components
        db_path = "data/nba_data.db"
        feature_engineer = FeatureEngineer(db_path)
        model_manager = ModelManager(db_path)
        visualizer = PredictionVisualizer(db_path)
        
        # LeBron James player ID
        lebron_id = 2544
        player_name = "LeBron James"
        opponent_name = "Golden State Warriors"
        
        # Get current date for prediction
        from datetime import datetime
        game_date = datetime.now().strftime("%Y-%m-%d")
        
        # Create features
        print("🔧 Creating features...")
        features_df = feature_engineer.create_features_for_player(
            lebron_id, game_date, include_h2h_features=False
        )
        
        if features_df.empty:
            print("❌ No data available for demo")
            return
        
        # Load models and make predictions
        print("🤖 Loading models and making predictions...")
        model_manager.load_models(["pts", "reb", "ast", "stl", "blk"])
        predictions_df = model_manager.predict_stats(features_df, ["pts", "reb", "ast", "stl", "blk"])
        
        # Get recent stats for comparison
        import sqlite3
        conn = sqlite3.connect(db_path)
        recent_query = """
            SELECT AVG(pts) as pts_avg, AVG(reb) as reb_avg, AVG(ast) as ast_avg, 
                   AVG(stl) as stl_avg, AVG(blk) as blk_avg
            FROM player_games 
            WHERE player_id = ? 
            ORDER BY game_date DESC 
            LIMIT 10
        """
        recent_df = pd.read_sql_query(recent_query, conn, params=(lebron_id,))
        conn.close()
        
        recent_stats = recent_df.iloc[0].to_dict() if not recent_df.empty else {}
        
        # Create and show visualization
        print("🎨 Creating visualization...")
        chart_path = visualizer.show_prediction_rationale(
            player_id=lebron_id,
            player_name=player_name,
            predictions_df=predictions_df,
            features_df=features_df,
            recent_stats=recent_stats,
            opponent_name=opponent_name
        )
        
        if chart_path:
            print("\n🎊 Demo completed successfully!")
            print(f"📁 Visualization saved to: {chart_path}")
            print("\n🔍 What the visualization shows:")
            print("  📈 Recent performance trends with trend lines")
            print("  🎯 Prediction breakdown vs recent form")
            print("  🎂 Career performance by age with decline zones")
            print("  🎯 Confidence levels for each stat")
            print("  📊 Key insights and prediction methodology")
            print("\n💡 The visualization explains WHY predictions are what they are!")
        else:
            print("❌ Could not create visualization")
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_prediction_visualization() 