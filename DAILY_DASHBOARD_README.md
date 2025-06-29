# 🏀 NBA Daily Dashboard with Self-Improvement

A comprehensive daily dashboard that automatically improves your NBA stat prediction system through continuous performance monitoring and model retraining.

## 🌟 What Makes This Special

This isn't just a prediction dashboard - it's a **self-improving system** that:

- **Monitors its own performance** daily
- **Automatically retrains models** when accuracy drops
- **Learns from new data** continuously  
- **Provides actionable insights** for better predictions
- **Tracks improvement over time**

## 🚀 Quick Start

### 1. Initial Setup
```bash
# Set up the dashboard system
make setup-dashboard

# Or manually:
python daily_dashboard.py setup
```

### 2. Daily Use
```bash
# Run your daily workflow (recommended)
make daily-dashboard

# Or manually:
python daily_dashboard.py daily
```

### 3. Check Status Anytime
```bash
# Quick health check
make status

# Performance metrics
make performance
```

## 🔄 The Self-Improvement Cycle

The dashboard runs an 8-step workflow each day:

### 1. **System Health Check** 🏥
- Verifies database integrity
- Checks model availability
- Auto-repairs common issues

### 2. **Update Previous Results** 📊
- Fetches actual game results from past days
- Updates prediction accuracy records
- Builds performance history

### 3. **Performance Evaluation** 📈
- Calculates prediction accuracy metrics (MAE, RMSE)
- Tracks confidence scores
- Compares recent vs historical performance

### 4. **Self-Improvement Check** 🧠
- Detects performance degradation (>15% drop in accuracy)
- Ensures minimum time between retraining (7 days default)
- Triggers automatic improvement when needed

### 5. **Self-Improvement Process** 🔄 (when triggered)
- Collects additional training data
- Retrains all models with updated data
- Records retraining events for tracking

### 6. **Fresh Data Collection** 🆕
- Gathers latest player data
- Updates rosters and recent performance
- Prepares for today's predictions

### 7. **Generate Today's Predictions** 🎯
- Creates predictions for all games today
- Ranks players by stat categories
- Stores predictions with confidence scores

### 8. **Daily Report & Display** 📋
- Creates comprehensive daily report
- Shows beautiful formatted results
- Provides actionable recommendations

## 📱 Dashboard Commands

### Core Commands
```bash
# Complete daily workflow
python daily_dashboard.py daily

# Quick system status
python daily_dashboard.py status

# Performance summary (last 14 days)
python daily_dashboard.py performance

# Initial setup
python daily_dashboard.py setup
```

### Using Make (Recommended)
```bash
make daily-dashboard    # Daily workflow
make status            # System status
make performance       # Performance metrics
make setup-dashboard   # Initial setup
```

## 🤖 Automation Setup

### Option 1: Simple Automation Script
Use the provided `run_daily.sh` script:

```bash
# Run manually
./run_daily.sh

# Set up daily cron job (runs at 9 AM daily)
crontab -e
# Add this line:
0 9 * * * /path/to/nba-stat-predictor/run_daily.sh
```

### Option 2: Custom Scheduling
```bash
# Every day at 8 AM
0 8 * * * cd /path/to/nba-stat-predictor && python daily_dashboard.py daily

# Twice daily (morning and evening)
0 8,20 * * * cd /path/to/nba-stat-predictor && python daily_dashboard.py daily
```

## 📊 What You'll See Daily

### System Status
```
📊 SYSTEM STATUS
------------------------
Total Games in DB:     15,234
Unique Players:        450
Models Trained:        5/5
Recent Predictions:    85
```

### Performance Metrics
```
📈 RECENT PERFORMANCE
------------------------
PTS  - MAE:  3.45, Samples:  89, Confidence: 0.78
REB  - MAE:  2.12, Samples:  89, Confidence: 0.82
AST  - MAE:  1.87, Samples:  89, Confidence: 0.75
```

### Today's Predictions
```
🎯 TODAY'S TOP PREDICTIONS
------------------------

Top PTS Predictions:
  1. Luka Doncic          28.5 (conf: 0.82)
  2. Jayson Tatum         27.3 (conf: 0.79)
  3. Giannis Antetokounmpo 26.8 (conf: 0.81)
```

### Smart Recommendations
```
💡 RECOMMENDATIONS
------------------------
  1. Points prediction accuracy could be improved (MAE: 5.2)
  2. Consider running backtests during off-season
```

## 🎛️ Configuration

The dashboard uses your existing `config.yaml` with these key settings:

```yaml
models:
  retraining:
    performance_threshold: 0.15  # Retrain if accuracy drops 15%
    min_days_between_retraining: 7  # Wait at least 7 days

monitoring:
  accuracy_window_days: 14  # Track performance over 2 weeks
```

## 📁 Generated Files

The dashboard creates several tracking files:

```
data/
├── performance_log.json      # Historical performance metrics
├── last_retrain.json        # Last retraining information
└── nba_data.db              # Main database

reports/
└── daily_report_YYYYMMDD.json  # Daily reports archive

logs/
├── daily_dashboard.log       # Dashboard activity log
└── daily_run_*.log          # Automation script logs
```

## 🔧 Troubleshooting

### Common Issues

**"No games scheduled for today"**
- Normal during off-season or rest days
- Dashboard will still update previous results and check performance

**"Insufficient training data"** 
- Run: `make setup-dashboard` or `python daily_dashboard.py setup`
- The system will auto-collect data and train models

**"Missing trained models"**
- Auto-repair will trigger automatically
- Or manually run: `make train`

**Performance degradation detected**
- This is expected! The system will automatically retrain
- Monitor the self-improvement process in logs

### Manual Fixes

```bash
# Force data collection
python main.py collect-data --players-limit 150

# Force model retraining  
python main.py train

# Check accuracy manually
python main.py accuracy

# Reset performance tracking (if needed)
rm data/performance_log.json data/last_retrain.json
```

## 📈 Monitoring Your System's Improvement

### Performance Tracking
The dashboard automatically tracks:
- **Prediction accuracy trends** over time
- **Model confidence scores** 
- **Retraining frequency** and triggers
- **Data freshness** and coverage

### Success Indicators
Look for:
- ✅ Stable or improving MAE scores
- ✅ High confidence scores (>0.7)
- ✅ Successful auto-retraining events
- ✅ Regular fresh data collection

### Warning Signs
Watch for:
- ⚠️ Consistently degrading accuracy
- ⚠️ Low confidence scores (<0.5)
- ⚠️ Failed retraining attempts
- ⚠️ Stale data (no recent updates)

## 🎯 Best Practices

### Daily Usage
1. **Run the dashboard every morning** before games start
2. **Check recommendations** and act on them
3. **Monitor performance trends** weekly
4. **Review auto-improvement events** monthly

### Optimization Tips
1. **Let the system self-improve** - don't override too frequently
2. **Monitor during season transitions** for accuracy changes
3. **Adjust thresholds** in config.yaml based on your needs
4. **Keep historical data** for long-term trend analysis

### During NBA Season
- Dashboard is most active and accurate
- Daily predictions for 10+ games
- Frequent self-improvement triggers
- Rich performance data

### During Off-Season
- Focus on backtesting and data collection
- System maintenance and optimization
- Model experimentation
- Historical analysis

## 🚀 Advanced Features

### Custom Analysis
```bash
# Performance over specific period
python daily_dashboard.py performance --days 30

# Status with detailed health check
python daily_dashboard.py status
```

### Integration Ready
The dashboard generates JSON reports perfect for:
- **Web dashboards** (load daily_report_*.json)
- **Mobile apps** (consume API-like data)
- **Analytics tools** (import performance_log.json)
- **Alerting systems** (monitor recommendations)

### Extensible Design
Easy to extend with:
- **Custom prediction models**
- **Additional sports**
- **Different retraining strategies**
- **Enhanced notifications**

---

## 🏀 Ready to Start?

Your NBA prediction system is now self-improving! 

```bash
# Get started today
make setup-dashboard
make daily-dashboard
```

The system will handle the rest, continuously learning and improving to give you better predictions every day.

**🎯 Your edge in NBA predictions starts now!** 