# NBA Stat Predictor - Quick Reference

This guide provides quick access to the most common commands and operations.

## 🚀 Daily Operations

### Getting Predictions
```bash
# Get tonight's predictions
python main.py predict

# Full pipeline (collect data, train, predict)
python main.py full-pipeline
```

### System Status
```bash
# Check system health
python nba_dashboard.py --status

# Show prediction accuracy
python main.py accuracy

# Interactive dashboard
python nba_dashboard.py --interactive
```

### Data Management
```bash
# Collect more training data
python main.py collect-data --players-limit 100

# Train models
python main.py train

# Update with yesterday's results
python main.py update-results
```

## 🔧 Setup & Installation

### First Time Setup
```bash
# Option 1: Automated setup
python setup.py

# Option 2: Manual setup
pip install -r requirements.txt
python main.py collect-data --players-limit 20
python main.py train
```

### Quick Start
```bash
# Run the example script
python examples/quick_start.py

# Open the tutorial notebook
jupyter notebook examples/NBA_Predictor_Tutorial.ipynb
```

## 📊 Analysis & Evaluation

### Performance Testing
```bash
# Run historical backtest
python main.py backtest --season 2023-24

# Check if models need retraining
python main.py retrain
```

### Data Exploration
```bash
# Launch interactive dashboard
python nba_dashboard.py --interactive

# Check top players by category
python nba_dashboard.py --top-players
```

## 🛠 Advanced Operations

### Model Management
```bash
# Force retrain all models
python main.py train

# Update model performance tracking
python main.py update-results --days-back 7
```

### Configuration
```bash
# Edit configuration
nano config.yaml

# View all available commands
python main.py --help
```

## 🔍 Troubleshooting

### Common Issues
```bash
# Fix database issues
rm data/nba_data.db
python main.py collect-data --players-limit 20

# Reset everything
rm -rf data/ models/
python setup.py

# Check dependencies
pip install -r requirements.txt --upgrade
```

### Logging
```bash
# View recent logs
tail -f logs/nba_predictor.log

# Increase logging verbosity
export LOG_LEVEL=DEBUG
python main.py predict
```

## 📱 Dashboard Features

The interactive dashboard (`python nba_dashboard.py --interactive`) provides:

- **System Status**: Database health, model status, recent activity
- **Top Players**: Leaders by statistical category
- **Prediction Demo**: Sample predictions with explanations
- **Guided Actions**: Suggested next steps based on system state

## 📈 Performance Metrics

### Understanding Accuracy
- **MAE**: Mean Absolute Error (lower is better)
- **RMSE**: Root Mean Square Error (lower is better)
- **R²**: Coefficient of determination (higher is better, max 1.0)

### Typical Performance
- **Points**: MAE ~3.5, RMSE ~4.8
- **Rebounds**: MAE ~2.1, RMSE ~2.9
- **Assists**: MAE ~1.9, RMSE ~2.6

## 🎯 Best Practices

### Daily Workflow
1. **Morning**: Check system status
2. **Evening**: Get tonight's predictions
3. **Next Day**: Update with results
4. **Weekly**: Check accuracy and retrain if needed

### Data Collection
- Start with 20-50 players for quick setup
- Expand to 100+ players for better accuracy
- Include multiple seasons for robust training

### Model Maintenance
- Monitor accuracy weekly
- Retrain when MAE increases >10%
- Update data regularly during NBA season

## 💡 Tips & Tricks

- Use `--help` with any command for detailed options
- The dashboard provides guided assistance for new users
- Check `docs/` directory for detailed documentation
- Monitor system health percentage in dashboard
- Use the Jupyter notebook for interactive learning 