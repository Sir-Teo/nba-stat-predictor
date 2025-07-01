# Enhanced NBA Stat Predictor with Advanced ML Pipeline

A comprehensive machine learning system that predicts NBA player statistics using advanced data collection, feature engineering, and model optimization with 530+ features.

## 🚀 Enhanced Features (Latest Update)
- **📊 Comprehensive Data Collection** - 9 seasons (2016-2024) with 200+ players and quality validation
- **🧠 Advanced Feature Engineering** - 530+ features including age-aware, situational, and momentum features
- **⚡ Enhanced Model Training** - Advanced hyperparameter optimization with 5-fold time series cross-validation
- **🎯 Age-Aware Predictions** - Specialized handling for veteran players (35+) with realistic adjustments
- **🔍 Quality Validation** - Comprehensive data quality scoring and validation
- **🎨 Interactive Dashboard** - Enhanced interface with quality metrics and advanced prediction modes
- **📈 Real-time Predictions** - Advanced predictions with confidence scoring and visualization
- **📊 Accuracy Tracking** - Continuous monitoring with enhanced metrics and performance analysis

## 📁 Project Structure

```
nba-stat-predictor/
├── src/                    # Core application code
│   ├── data/              # Data collection and processing
│   ├── models/            # ML model implementations
│   ├── predictions/       # Prediction logic
│   ├── evaluation/        # Performance evaluation tools
│   └── utils/             # Utility functions
├── docs/                  # Documentation hub
│   ├── README.md          # Documentation index
│   ├── improvements_summary.md
│   ├── system_validation_report.md
│   └── ...                # Other guides and reports
├── examples/              # Tutorials and examples
│   ├── NBA_Predictor_Tutorial.ipynb
│   └── quick_start.py     # Simple usage example
├── data/                  # Database and data files
├── models/                # Trained model files
├── main.py                # Main CLI application
├── nba_dashboard.py       # Interactive dashboard
├── setup.py               # Automated setup script
└── config.yaml            # Configuration file
```

## 🎯 Enhanced Features

- **📊 Comprehensive Data Collection**: 9 seasons of data (2016-2024) with 200+ players, playoff data, and quality validation
- **🧠 Advanced Feature Engineering**: 530+ features including extended rolling windows (up to 30 games), age-aware features, situational analysis, and momentum tracking
- **⚡ Enhanced Model Training**: Advanced ensemble models (LightGBM, XGBoost, Random Forest, Lasso) with hyperparameter optimization and 5-fold time series cross-validation
- **🎯 Age-Aware Predictions**: Specialized handling for veteran players (35+) with realistic age adjustments and confidence scoring
- **🔍 Quality Validation**: Comprehensive data quality scoring and validation with detailed metrics
- **🎨 Interactive Dashboard**: Enhanced interface with quality metrics, advanced prediction modes, and professional visualizations
- **📈 Real-time Predictions**: Advanced predictions with confidence scoring, comprehensive analysis modes, and prediction rationale charts
- **📊 Performance Analytics**: Continuous monitoring with enhanced metrics, model performance tracking, and detailed accuracy analysis

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd nba-stat-predictor

# Install dependencies
pip install -r requirements.txt
```

### 2. Enhanced Interactive Dashboard (Recommended)

```bash
# Launch the enhanced interactive dashboard for full control
python interactive_dashboard.py
```

Enhanced Features:
- **📊 Enhanced Data Updates**: 9 seasons of data, 200+ players, quality validation
- **🎯 Advanced Predictions**: 530+ features, age-aware adjustments, confidence scoring
- **🧠 Enhanced Model Training**: Advanced optimization, comprehensive statistics
- **📈 Quality Metrics**: Data quality scoring, qualified players tracking
- **🎨 Professional Visualizations**: Prediction rationale charts and analysis
- **⚡ Quick vs Comprehensive**: Choose analysis level based on needs

### 3. Enhanced Command Line Interface

```bash
# Run the enhanced full pipeline (collect data, train models, predict)
python main.py full-pipeline

# Enhanced data collection (9 seasons, 200+ players)
python main.py enhanced-collect

# Enhanced model training (530+ features, advanced optimization)
python main.py enhanced-train

# Make predictions for tonight's games
python main.py predict
```

## Detailed Usage

### Data Collection

Collect historical NBA data for training the models:

```bash
# Collect data for 50 players (default)
python main.py collect-data

# Collect data for more players
python main.py collect-data --players-limit 100
```

### Model Training

Train machine learning models on the collected data:

```bash
python main.py train
```

### Making Predictions

Generate predictions for tonight's NBA games:

```bash
python main.py predict
```

This will output top predictions for points, rebounds, and assists like:

```
================================================================================
TONIGHT'S TOP PREDICTIONS
================================================================================

Top 10 PTS predictions:
 1. Luka Doncic              - 28.5 (confidence: 0.82)
 2. Jayson Tatum             - 27.3 (confidence: 0.79)
 3. Giannis Antetokounmpo    - 26.8 (confidence: 0.81)
 ...
```

### Update Results

After games are completed, update predictions with actual results:

```bash
# Update yesterday's results
python main.py update-results

# Update results from 2 days ago
python main.py update-results --days-back 2
```

### Check Accuracy

View recent prediction accuracy:

```bash
python main.py accuracy
```

Output:
```
============================================================
PREDICTION ACCURACY (Last 14 Days)
============================================================
PTS  - MAE:  3.45, RMSE:  4.67, Samples:  89
REB  - MAE:  2.12, RMSE:  2.98, Samples:  89
AST  - MAE:  1.87, RMSE:  2.54, Samples:  89
============================================================
```

### Model Retraining

Check if models need retraining and retrain automatically:

```bash
python main.py retrain
```

## System Architecture

### Components

1. **Data Collector** (`src/data/nba_data_collector.py`)
   - Fetches player game logs from NBA API
   - Stores data in SQLite database
   - Handles rate limiting and error recovery

2. **Feature Engineer** (`src/data/feature_engineer.py`)
   - Creates rolling statistics (3, 5, 10 game averages)
   - Calculates performance trends
   - Generates home/away splits
   - Analyzes rest patterns and fatigue
   - Measures player consistency

3. **Model Manager** (`src/models/stat_predictors.py`)
   - Manages multiple Random Forest models
   - Handles model training and evaluation
   - Stores and loads trained models
   - Tracks model performance over time

4. **Tonight Predictor** (`src/predictions/tonight_predictor.py`)
   - Identifies today's games
   - Generates predictions for all players
   - Stores predictions in database
   - Updates predictions with actual results

### Database Schema

The system uses SQLite with the following main tables:

- **player_games**: Historical game statistics for all players
- **predictions**: Model predictions with confidence scores
- **model_performance**: Model accuracy metrics over time

### Features Used for Prediction

- **Rolling Averages**: 3, 5, and 10-game averages for all stats
- **Trends**: Statistical slopes indicating improving/declining performance
- **Recent Form**: Last game performance vs season average
- **Home/Away Splits**: Performance differences by venue
- **Rest Patterns**: Days between games and back-to-back performance
- **Consistency Metrics**: Standard deviation and coefficient of variation
- **Streaks**: Current above/below average performance runs

## Configuration Options

### Command Line Arguments

- `--players-limit`: Number of players to collect data for (default: 50)
- `--days-back`: Days back to update results for (default: 1)
- `--db-path`: Path to SQLite database (default: data/nba_data.db)

### Model Parameters

Models can be customized by modifying the parameters in `src/models/stat_predictors.py`:

```python
# Random Forest parameters
default_params = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42
}
```

## Self-Improvement Mechanism

The system continuously improves through:

1. **Accuracy Tracking**: All predictions are stored with actual outcomes
2. **Performance Monitoring**: Regular evaluation of model accuracy
3. **Automatic Retraining**: Models retrain when performance degrades
4. **Feature Updates**: New features can be added to improve predictions

## Predicted Statistics

The system predicts the following player statistics:

- **Points (PTS)**: Total points scored
- **Rebounds (REB)**: Total rebounds (offensive + defensive)
- **Assists (AST)**: Total assists
- **Steals (STL)**: Total steals
- **Blocks (BLK)**: Total blocks

## API Rate Limiting

The system respects NBA API rate limits:
- 0.6 second delay between requests
- Automatic retry logic for failed requests
- Graceful handling of API errors

## Data Storage

- **Historical Data**: Stored locally in SQLite database
- **Models**: Saved as pickle files in `models/` directory
- **Predictions**: Tracked in database with timestamps and confidence scores

## Performance Expectations

- **Training Time**: ~5-10 minutes for 50 players
- **Prediction Time**: ~30 seconds for a full slate of games
- **Accuracy**: Typically achieves 15-25% improvement over baseline predictions
- **Data Requirements**: Minimum 20 games per player for reliable training

## Troubleshooting

### Common Issues

1. **No games found**: Check if it's an NBA game day
2. **Insufficient data**: Run `collect-data` first
3. **API errors**: Wait and retry (rate limiting)
4. **Model errors**: Ensure sufficient training data exists

### Logs

The system provides detailed logging. Check output for:
- Data collection progress
- Model training metrics
- Prediction generation status
- Error messages and warnings

## Future Enhancements

Potential improvements:

- **Injury Data Integration**: Factor in player injury reports
- **Opponent Analysis**: Include defensive ratings and matchup history
- **Advanced Models**: Implement neural networks or ensemble methods
- **Real-time Updates**: Live model updates during games
- **Web Interface**: Build a web dashboard for predictions
- **More Statistics**: Predict minutes, field goal percentage, etc.

## Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is for educational and personal use. Please respect NBA API terms of service.

## Disclaimer

This tool is for entertainment and educational purposes only. Do not use for gambling or commercial purposes. NBA statistics and API usage should comply with NBA terms of service. 