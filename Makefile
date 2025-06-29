# NBA Stat Predictor Makefile
# Provides simple commands for common operations

.PHONY: help setup install clean test predict train status backtest logs daily-dashboard status performance setup-dashboard

# Default target
help:
	@echo "🏀 NBA Stat Predictor - Available Commands"
	@echo "================================================"
	@echo "Setup Commands:"
	@echo "  make setup     - Complete setup (install deps, create dirs, run demo)"
	@echo "  make install   - Install dependencies only"
	@echo "  make clean     - Clean up generated files"
	@echo ""
	@echo "Daily Usage:"
	@echo "  make predict   - Get predictions for tonight's games"
	@echo "  make train     - Train/retrain models with latest data"
	@echo "  make status    - Check system status and accuracy"
	@echo "  make update    - Update with yesterday's results"
	@echo ""
	@echo "Data & Analysis:"
	@echo "  make collect   - Collect more training data"
	@echo "  make backtest  - Run historical performance test"
	@echo "  make pipeline  - Run full pipeline (collect, train, predict)"
	@echo ""
	@echo "Monitoring:"
	@echo "  make logs      - Show recent logs"
	@echo "  make test      - Test system components"
	@echo ""
	@echo "Configuration:"
	@echo "  Edit config.yaml to customize settings"
	@echo "  Check QUICK_START.md for detailed instructions"
	@echo ""
	@echo "🏀 DAILY DASHBOARD COMMANDS:"
	@echo "  make daily-dashboard    - Run the complete daily workflow"
	@echo "  make status            - Quick system health check"
	@echo "  make performance       - Show recent performance metrics"
	@echo "  make setup-dashboard   - Initial dashboard setup"
	@echo ""
	@echo "🔧 SYSTEM COMMANDS:"
	@echo "  make setup             - Complete system setup"
	@echo "  make collect           - Collect NBA data"
	@echo "  make train             - Train prediction models"
	@echo "  make predict           - Make predictions for tonight"
	@echo "  make accuracy          - Check prediction accuracy"
	@echo "  make update-results    - Update predictions with actual results"
	@echo "  make retrain           - Check and retrain models if needed"
	@echo "  make full-pipeline     - Run complete data→train→predict pipeline"
	@echo "  make clean             - Clean temporary files"

# Setup and installation
setup:
	@echo "🚀 Running complete setup..."
	python setup.py

setup-fast:
	@echo "🚀 Running fast setup (no demo)..."
	python setup.py --skip-demo

install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt

# Daily usage commands
predict:
	@echo "🎯 Getting predictions for tonight's games..."
	python main.py predict

train:
	@echo "🧠 Training models..."
	python main.py train

status:
	@echo "📊 Checking system status..."
	@python -c "import sqlite3; conn = sqlite3.connect('data/nba_data.db'); cursor = conn.cursor(); cursor.execute('SELECT COUNT(*) FROM player_games'); print(f'Games in database: {cursor.fetchone()[0]}'); conn.close()"
	@echo ""
	python main.py accuracy

update:
	@echo "🔄 Updating with yesterday's results..."
	python main.py update-results

# Data collection and analysis
collect:
	@echo "📈 Collecting more training data..."
	python main.py collect-data --players-limit 100

collect-quick:
	@echo "📈 Quick data collection..."
	python main.py collect-data --players-limit 25

backtest:
	@echo "🧪 Running backtest..."
	python main.py backtest --season 2023-24

pipeline:
	@echo "🔄 Running full pipeline..."
	python main.py full-pipeline

# Monitoring and testing
logs:
	@echo "📋 Recent logs:"
	@if [ -f logs/nba_predictor.log ]; then tail -20 logs/nba_predictor.log; else echo "No log file found. Run 'make setup' first."; fi

logs-full:
	@echo "📋 Full logs:"
	@if [ -f logs/nba_predictor.log ]; then cat logs/nba_predictor.log; else echo "No log file found. Run 'make setup' first."; fi

test:
	@echo "🧪 Testing system components..."
	@python -c "from setup import test_api_connection; test_api_connection()"
	@echo "Testing prediction pipeline..."
	@python -c "import sys; sys.path.append('src'); from main import NBAStatPredictorApp; app = NBAStatPredictorApp(); print('✅ System components loaded successfully')"

# Maintenance
clean:
	@echo "🧹 Cleaning up generated files..."
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf src/*/__pycache__/
	rm -rf src/*/*/__pycache__/
	rm -rf *.pyc
	rm -rf .pytest_cache/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
	@echo "✅ Cleanup completed"

clean-all: clean
	@echo "🧹 Deep cleaning (including data and models)..."
	rm -rf data/
	rm -rf models/
	rm -rf logs/
	rm -rf plots/
	@echo "⚠️  All data and models removed. Run 'make setup' to reinitialize."

# Retrain models
retrain:
	@echo "🔄 Checking if models need retraining..."
	python main.py retrain

# Database management
db-info:
	@echo "🗄️  Database information:"
	@python -c "import sqlite3; conn = sqlite3.connect('data/nba_data.db'); cursor = conn.cursor(); cursor.execute('SELECT name FROM sqlite_master WHERE type=\"table\"'); tables = cursor.fetchall(); print('Tables:', [t[0] for t in tables]); cursor.execute('SELECT COUNT(*) FROM player_games'); print(f'Total games: {cursor.fetchone()[0]}'); cursor.execute('SELECT COUNT(DISTINCT player_id) FROM player_games'); print(f'Unique players: {cursor.fetchone()[0]}'); conn.close()"

# Quick checks
check-games:
	@echo "🏀 Checking for today's NBA games..."
	@python -c "import sys; sys.path.append('src'); from src.predictions.tonight_predictor import TonightPredictor; tp = TonightPredictor('data/nba_data.db'); games = tp._get_todays_games(); print(f'Games today: {len(games)}' if games else 'No games today')"

# Development helpers
dev-setup: install
	pip install pytest black flake8 jupyter
	@echo "✅ Development environment ready"

format:
	@echo "🎨 Formatting code..."
	black *.py src/

lint:
	@echo "🔍 Linting code..."
	flake8 *.py src/ --max-line-length=100 --ignore=E203,W503

# Show configuration
config:
	@echo "⚙️  Current configuration:"
	@if [ -f config.yaml ]; then cat config.yaml; else echo "No config.yaml found. Run 'make setup' first."; fi

# Dashboard commands
daily-dashboard:
	@echo "🏀 Running Daily NBA Dashboard..."
	python daily_dashboard.py daily

status:
	@echo "🏀 Checking System Status..."
	python daily_dashboard.py status

performance:
	@echo "📈 Showing Performance Summary..."
	python daily_dashboard.py performance

setup-dashboard:
	@echo "🔧 Setting up NBA Dashboard..."
	python daily_dashboard.py setup 