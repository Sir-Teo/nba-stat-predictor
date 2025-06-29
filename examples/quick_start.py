#!/usr/bin/env python3
"""
NBA Stat Predictor - Quick Start Example

A simple example demonstrating the basic usage of the NBA stat prediction system.
This script shows how to:
1. Check system status
2. Run predictions
3. View system health
"""

import os
import sys
from pathlib import Path

# Add src to path (adjust for examples directory)
sys.path.append(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent))

import logging

from main import NBAStatPredictorApp

# Setup simple logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_system_status(app):
    """Check if the system has sufficient data to make predictions."""
    import sqlite3

    try:
        conn = sqlite3.connect(app.db_path)
        cursor = conn.cursor()

        # Check data availability
        cursor.execute("SELECT COUNT(*) FROM player_games")
        total_games = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT player_id) FROM player_games")
        unique_players = cursor.fetchone()[0]

        conn.close()

        print(f"📊 System Status:")
        print(f"   Database: {app.db_path}")
        print(f"   Total games: {total_games:,}")
        print(f"   Unique players: {unique_players}")

        if total_games >= 100:
            print("   ✅ System ready for predictions")
            return True
        else:
            print("   ⚠️  Insufficient data for reliable predictions")
            return False

    except Exception as e:
        print(f"   ❌ Error checking system: {e}")
        return False


def run_example():
    """Run a complete example workflow."""
    print("🏀 NBA Stat Predictor - Quick Start Example")
    print("=" * 60)

    # Initialize the application
    print("\n1️⃣  Initializing NBA Stat Predictor...")
    app = NBAStatPredictorApp()

    # Check system status
    print("\n2️⃣  Checking system status...")
    system_ready = check_system_status(app)

    if not system_ready:
        print("\n🚀 Getting Started:")
        print("   To set up the system with data:")
        print("   1. python main.py collect-data --players-limit 20")
        print("   2. python main.py train")
        print("   3. python main.py predict")
        print("\n   Or run everything at once:")
        print("   python main.py full-pipeline")
        return

    # Show recent accuracy if available
    print("\n3️⃣  Checking prediction accuracy...")
    try:
        app.show_accuracy()
    except Exception as e:
        print(f"   No accuracy data available yet: {e}")

    # Make predictions for tonight
    print("\n4️⃣  Making predictions for tonight's games...")
    try:
        app.predict_tonight()
    except Exception as e:
        print(f"   Predictions unavailable: {e}")
        print("   This is normal when no NBA games are scheduled")

    print("\n✨ Example completed!")
    print("\n🎯 Next Steps:")
    print("   • Run 'python main.py predict' daily for game predictions")
    print("   • Use 'python main.py update-results' after games complete")
    print("   • Try 'python nba_dashboard.py --interactive' for the full interface")
    print("   • Explore 'examples/NBA_Predictor_Tutorial.ipynb' for detailed learning")


if __name__ == "__main__":
    try:
        run_example()
    except KeyboardInterrupt:
        print("\n\n👋 Example interrupted by user")
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're in the project root directory")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Run setup: python setup.py")
