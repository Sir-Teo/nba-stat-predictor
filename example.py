#!/usr/bin/env python3
"""
Simple example demonstrating NBA Stat Predictor usage.
"""

import sys
import os

# Add src to path
sys.path.append('src')

from main import NBAStatPredictorApp
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

def main():
    """Simple example of using the NBA Stat Predictor."""
    
    print("NBA Stat Predictor - Example Usage")
    print("=" * 50)
    
    # Initialize the app
    app = NBAStatPredictorApp()
    
    try:
        # Check if we have any data
        import sqlite3
        conn = sqlite3.connect(app.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM player_games")
        game_count = cursor.fetchone()[0]
        conn.close()
        
        print(f"Current database has {game_count} player games")
        
        if game_count < 50:
            print("\nInsufficient data for reliable predictions.")
            print("To get started, run:")
            print("  python main.py collect-data")
            print("  python main.py train")
            print("  python main.py predict")
            return
        
        # Show some example functionality
        print("\n1. Checking prediction accuracy...")
        app.show_accuracy()
        
        print("\n2. Making predictions for tonight...")
        app.predict_tonight()
        
        print("\n3. For more functionality, use main.py with these commands:")
        print("   - collect-data: Gather NBA data")
        print("   - train: Train prediction models")
        print("   - predict: Get tonight's predictions")
        print("   - update-results: Update with actual game results")
        print("   - accuracy: Show prediction accuracy")
        print("   - full-pipeline: Run everything")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTo get started with fresh data:")
        print("1. python main.py collect-data --players-limit 20")
        print("2. python main.py train")
        print("3. python main.py predict")

if __name__ == "__main__":
    main() 