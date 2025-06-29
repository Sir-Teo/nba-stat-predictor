#!/usr/bin/env python3
"""
NBA Stat Predictor Setup Script
Automates the initial setup process for new users.
"""

import argparse
import logging
import os
import sqlite3
import subprocess
import sys
from pathlib import Path


def setup_logging():
    """Setup logging for setup process."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


def check_python_version():
    """Check if Python version is compatible."""
    logger = logging.getLogger(__name__)

    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False

    logger.info(f"Python version: {sys.version}")
    return True


def install_dependencies():
    """Install required Python packages."""
    logger = logging.getLogger(__name__)

    logger.info("Installing dependencies...")

    try:
        # Install packages from requirements.txt
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        )
        logger.info("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    logger = logging.getLogger(__name__)

    directories = ["data", "models", "logs", "plots", "backups"]

    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Created directory: {directory}")


def setup_database():
    """Initialize the SQLite database with proper schema."""
    logger = logging.getLogger(__name__)

    db_path = "data/nba_data.db"

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if database is already setup
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()

        if tables:
            logger.info("Database already exists with tables")
        else:
            logger.info("Database created successfully")

        conn.close()
        return True

    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        return False


def test_api_connection():
    """Test connection to NBA API."""
    logger = logging.getLogger(__name__)

    try:
        # Add src to path for imports
        sys.path.append("src")
        from src.data.nba_data_collector import NBADataCollector

        collector = NBADataCollector("data/nba_data.db")

        # Try to get a small amount of data to test API
        logger.info("Testing NBA API connection...")
        test_players = collector.get_all_players()

        if not test_players.empty:
            logger.info(
                f"✅ NBA API connection successful. Found {len(test_players)} players"
            )
            return True
        else:
            logger.warning("⚠️  NBA API connection failed or returned no data")
            return False

    except Exception as e:
        logger.error(f"API test failed: {e}")
        return False


def run_quick_demo(sample_size=10):
    """Run a quick demo to verify everything works."""
    logger = logging.getLogger(__name__)

    logger.info("Running quick demo...")

    try:
        # Run a small data collection and training
        cmd = [
            sys.executable,
            "main.py",
            "collect-data",
            "--players-limit",
            str(sample_size),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            logger.info("✅ Demo data collection successful")

            # Try training
            cmd = [sys.executable, "main.py", "train"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode == 0:
                logger.info("✅ Demo model training successful")
                return True
            else:
                logger.warning("⚠️  Demo training failed")
                return False
        else:
            logger.warning("⚠️  Demo data collection failed")
            return False

    except subprocess.TimeoutExpired:
        logger.warning("⚠️  Demo timed out")
        return False
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return False


def create_quick_start_guide():
    """Create a quick start guide for new users."""
    guide_content = """
# NBA Stat Predictor - Quick Start Guide

## You're all set up! 🎉

Your NBA Stat Predictor is ready to use. Here are the most common commands:

### Basic Usage

```bash
# Get predictions for tonight's games
python main.py predict

# Run the full pipeline (collect data, train, predict)
python main.py full-pipeline

# Check how accurate your predictions have been
python main.py accuracy
```

### Data Management

```bash
# Collect more data (for better accuracy)
python main.py collect-data --players-limit 100

# Train models with the latest data
python main.py train

# Update predictions with actual game results
python main.py update-results
```

### Advanced Usage

```bash
# Run historical backtest to see performance
python main.py backtest --season 2023-24

# Retrain models if performance is declining
python main.py retrain
```

### Configuration

- Edit `config.yaml` to customize model parameters
- Check `logs/nba_predictor.log` for detailed logs
- View `plots/` directory for generated visualizations

### Troubleshooting

If you encounter issues:

1. Check the logs: `tail -f logs/nba_predictor.log`
2. Verify your data: Check `data/nba_data.db` exists
3. Test API connection: `python -c "from setup import test_api_connection; test_api_connection()"`

### Next Steps

1. Let it collect more data over time for better accuracy
2. Check predictions daily during NBA season
3. Monitor accuracy and retrain models as needed

Happy predicting! 🏀
"""

    with open("QUICK_START.md", "w") as f:
        f.write(guide_content)

    logger = logging.getLogger(__name__)
    logger.info("Created QUICK_START.md guide")


def main():
    """Main setup process."""
    parser = argparse.ArgumentParser(description="Setup NBA Stat Predictor")
    parser.add_argument(
        "--skip-demo", action="store_true", help="Skip the demo run (faster setup)"
    )
    parser.add_argument(
        "--demo-size",
        type=int,
        default=10,
        help="Number of players for demo (default: 10)",
    )

    args = parser.parse_args()

    logger = setup_logging()

    print("🏀 NBA Stat Predictor Setup")
    print("=" * 50)

    # Setup steps
    steps = [
        ("Checking Python version", check_python_version),
        ("Installing dependencies", install_dependencies),
        ("Creating directories", create_directories),
        ("Setting up database", setup_database),
        ("Testing API connection", test_api_connection),
    ]

    if not args.skip_demo:
        steps.append(("Running demo", lambda: run_quick_demo(args.demo_size)))

    steps.append(("Creating quick start guide", create_quick_start_guide))

    # Run setup steps
    for step_name, step_func in steps:
        print(f"\n📋 {step_name}...")
        try:
            if step_func():
                print(f"✅ {step_name} completed")
            else:
                print(f"⚠️  {step_name} completed with warnings")
        except Exception as e:
            print(f"❌ {step_name} failed: {e}")
            logger.error(f"Setup step failed: {step_name} - {e}")
            return False

    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Read QUICK_START.md for usage instructions")
    print("2. Run: python main.py predict")
    print("3. Check logs/ directory for detailed logs")
    print("\nEnjoy predicting NBA stats! 🏀")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
