#!/usr/bin/env python3
"""
Test script to check LeBron's predictions and identify areas for improvement.
"""

import sys
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import os

# Add src to path
sys.path.append("src")

from src.data.feature_engineer import AdvancedFeatureEngineer
from src.models.stat_predictors import ModelManager

def test_lebron_predictions():
    """Test LeBron's current predictions and analyze the model behavior."""
    
    # LeBron James player ID
    LEBRON_ID = 2544
    
    print("=" * 60)
    print("LEBRON JAMES PREDICTION ANALYSIS")
    print("=" * 60)
    
    # Initialize components
    feature_engineer = AdvancedFeatureEngineer("data/nba_data.db")
    model_manager = ModelManager("data/nba_data.db")
    
    # Get LeBron's recent games
    conn = sqlite3.connect("data/nba_data.db")
    
    # Get LeBron's recent performance
    recent_games_query = """
        SELECT game_date, pts, reb, ast, min, team_id, matchup
        FROM player_games 
        WHERE player_id = ? 
        ORDER BY game_date DESC 
        LIMIT 20
    """
    
    recent_games = pd.read_sql_query(recent_games_query, conn, params=(LEBRON_ID,))
    
    if recent_games.empty:
        print("❌ No recent games found for LeBron James")
        return
    
    print(f"📊 Found {len(recent_games)} recent games for LeBron")
    print(f"📅 Date range: {recent_games['game_date'].min()} to {recent_games['game_date'].max()}")
    
    # Show recent performance
    print("\n📈 RECENT PERFORMANCE (Last 10 games):")
    print("-" * 40)
    recent_10 = recent_games.head(10)
    print(f"Points: {recent_10['pts'].mean():.1f} avg (range: {recent_10['pts'].min()}-{recent_10['pts'].max()})")
    print(f"Rebounds: {recent_10['reb'].mean():.1f} avg (range: {recent_10['reb'].min()}-{recent_10['reb'].max()})")
    print(f"Assists: {recent_10['ast'].mean():.1f} avg (range: {recent_10['ast'].min()}-{recent_10['ast'].max()})")
    print(f"Minutes: {recent_10['min'].mean():.1f} avg")
    
    # Check if models are trained
    print("\n🤖 MODEL STATUS:")
    print("-" * 40)
    
    stat_types = ["pts", "reb", "ast"]
    for stat in stat_types:
        try:
            model_manager.load_models([stat])
            print(f"✅ {stat.upper()} model loaded")
        except Exception as e:
            print(f"❌ {stat.upper()} model not available: {e}")
    
    # Create features for today's prediction
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"\n🔮 PREDICTION FOR {today}:")
    print("-" * 40)
    
    try:
        # Create features
        features_df = feature_engineer.create_features_for_player(
            LEBRON_ID, 
            today,
            lookback_games=30,
            include_advanced_features=True
        )
        
        if not features_df.empty:
            print(f"✅ Features created successfully ({len(features_df.columns)} features)")
            
            # Show some key features
            key_features = [
                'player_age', 'recent_form_weight', 'age_decline_factor',
                'pts_avg_5g', 'pts_avg_10g', 'pts_trend_5g',
                'reb_avg_5g', 'reb_avg_10g', 'reb_trend_5g',
                'ast_avg_5g', 'ast_avg_10g', 'ast_trend_5g'
            ]
            
            print("\n🔍 KEY FEATURES:")
            for feature in key_features:
                if feature in features_df.columns:
                    value = features_df[feature].iloc[0]
                    print(f"  {feature}: {value:.3f}")
            
            # Make predictions
            predictions_df = model_manager.predict_stats(features_df, stat_types)
            
            if not predictions_df.empty:
                print("\n🎯 PREDICTIONS:")
                for stat in stat_types:
                    pred_col = f"predicted_{stat}"
                    conf_col = f"confidence_{stat}"
                    
                    if pred_col in predictions_df.columns:
                        pred_value = predictions_df[pred_col].iloc[0]
                        confidence = predictions_df[conf_col].iloc[0] if conf_col in predictions_df.columns else 0.5
                        
                        print(f"  {stat.upper()}: {pred_value:.1f} (confidence: {confidence:.2f})")
                        
                        # Compare with recent average
                        recent_avg = recent_10[stat].mean()
                        diff = pred_value - recent_avg
                        print(f"    vs recent avg ({recent_avg:.1f}): {diff:+.1f}")
            else:
                print("❌ No predictions generated")
        else:
            print("❌ No features generated")
            
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        import traceback
        traceback.print_exc()
    
    # Analyze feature importance if available
    print("\n🔍 FEATURE IMPORTANCE ANALYSIS:")
    print("-" * 40)
    
    try:
        # Check if we can get feature importance from models
        for stat in stat_types:
            model_file = f"models/{stat}_model.pkl"
            if os.path.exists(model_file):
                print(f"📊 {stat.upper()} model feature importance available")
            else:
                print(f"❌ {stat.upper()} model file not found")
    except Exception as e:
        print(f"❌ Error checking feature importance: {e}")
    
    conn.close()
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_lebron_predictions() 