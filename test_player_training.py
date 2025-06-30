#!/usr/bin/env python3
"""
Test script for player-specific model training.
Demonstrates how to train models for a specific player.
"""

import sys
sys.path.append('src')

from src.models.stat_predictors import ModelManager
from interactive_dashboard import InteractiveNBADashboard

def test_lebron_training():
    """Test training models specifically for LeBron James."""
    print("🏀 Testing Player-Specific Training with LeBron James")
    print("=" * 60)
    
    # LeBron James ID
    lebron_id = 2544
    lebron_name = "LeBron James"
    
    # Initialize model manager
    model_manager = ModelManager()
    
    try:
        # Train models for LeBron
        print(f"\n🚀 Training models for {lebron_name}...")
        results = model_manager.train_models_for_player(
            player_id=lebron_id,
            player_name=lebron_name,
            model_type="random_forest",  # Faster for demo
            stat_types=["pts", "reb", "ast"],  # Just 3 stats for demo
            optimize_hyperparams=False  # Faster for demo
        )
        
        print(f"\n📊 Training Results for {lebron_name}:")
        print("-" * 40)
        
        for stat, metrics in results.items():
            if 'error' in metrics:
                print(f"❌ {stat.upper()}: {metrics['error']}")
            else:
                mae = metrics.get('test_mae', metrics.get('val_mae', 0))
                r2 = metrics.get('test_r2', metrics.get('val_r2', 0))
                print(f"✅ {stat.upper()}: MAE={mae:.2f}, R²={r2:.3f}")
        
        print(f"\n🎯 Models saved as: pts_player_{lebron_id}_*.pkl")
        print("   These models are personalized for LeBron's performance patterns!")
        
    except ValueError as e:
        print(f"❌ Training failed: {e}")
        print("   This usually means insufficient data for the player.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def main():
    """Main function to run the test."""
    print("Testing Player-Specific Training Feature")
    print("This will train models specifically for LeBron James")
    print()
    
    # Check if user wants to proceed
    proceed = input("Continue with training? (y/N): ").strip().lower()
    if proceed != 'y':
        print("Test cancelled.")
        return
    
    test_lebron_training()

if __name__ == "__main__":
    main() 