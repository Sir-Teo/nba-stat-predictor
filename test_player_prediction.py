#!/usr/bin/env python3
"""
Test script for player-specific model predictions.
Demonstrates how to use the trained player-specific models.
"""

import sys
sys.path.append('src')

import pandas as pd
from src.models.stat_predictors import ModelManager
from src.data.feature_engineer import AdvancedFeatureEngineer
from datetime import datetime, timedelta

def test_lebron_prediction():
    """Test predictions with LeBron's player-specific models."""
    print("🏀 Testing Player-Specific Predictions for LeBron James")
    print("=" * 60)
    
    # LeBron James ID
    lebron_id = 2544
    lebron_name = "LeBron James"
    
    # Initialize components
    model_manager = ModelManager()
    feature_engineer = AdvancedFeatureEngineer()
    
    try:
        # Load the player-specific models
        print(f"\n📥 Loading player-specific models for {lebron_name}...")
        
        # Load models manually with player-specific naming
        stat_types = ["pts", "reb", "ast"]
        player_models = {}
        
        for stat in stat_types:
            model_filename = f"models/{stat}_player_{lebron_id}_random_forest_rf_v2.0.pkl"
            try:
                from src.models.stat_predictors import AdvancedStatPredictor
                predictor = AdvancedStatPredictor(stat)
                predictor.load_model(model_filename)
                player_models[stat] = predictor
                print(f"✅ Loaded {stat.upper()} model for {lebron_name}")
            except Exception as e:
                print(f"❌ Failed to load {stat.upper()} model: {e}")
        
        if not player_models:
            print("❌ No models loaded successfully")
            return
        
        # Create features for a recent prediction
        print(f"\n🔍 Creating features for {lebron_name} prediction...")
        
        # Use a date from the validation period (recent but not too recent)
        target_date = "2024-03-01"  # Use a date from recent past
        
        # Create features for this player and date
        features_df = feature_engineer.create_features_for_player(
            player_id=lebron_id,
            target_date=target_date,
            lookback_games=15,
            opponent_team_id=None,  # We'll ignore opponent for this test
            include_h2h_features=True,
            include_advanced_features=True,
        )
        
        if features_df.empty:
            print("❌ Could not create features for prediction")
            return
        
        print(f"✅ Created features: {len(features_df.columns)} features")
        
        # Make predictions with each player-specific model
        print(f"\n🎯 Making predictions with {lebron_name}'s personalized models:")
        print("-" * 50)
        
        for stat_type, predictor in player_models.items():
            try:
                # Prepare features (exclude metadata columns)
                feature_columns = [
                    col for col in features_df.columns 
                    if col not in ["player_id", "game_id", "game_date", "target_date", "opponent_team_id"]
                ]
                X = features_df[feature_columns]
                
                # Make prediction
                prediction = predictor.predict(X)[0]  # Get first prediction
                
                # Get model performance metrics for context
                metrics = predictor.performance_metrics
                mae = metrics.get('test_mae', 0)
                r2 = metrics.get('test_r2', 0)
                
                print(f"   {stat_type.upper()}: {prediction:.1f} (Model MAE: {mae:.2f}, R²: {r2:.3f})")
                
            except Exception as e:
                print(f"   ❌ {stat_type.upper()}: Error - {e}")
        
        print(f"\n📊 These predictions are based on {lebron_name}'s personal performance patterns!")
        print("   The models learned specifically from his game history and tendencies.")
        
    except Exception as e:
        print(f"❌ Error during prediction test: {e}")

def main():
    """Main function to run the test."""
    print("Testing Player-Specific Prediction Feature")
    print("This will test predictions using LeBron's personalized models")
    print()
    
    test_lebron_prediction()

if __name__ == "__main__":
    main() 