#!/bin/bash

# NBA Daily Dashboard Runner
# Script to run the daily dashboard workflow automatically
# Can be set up as a cron job for daily automation

# Set script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Create log directory
mkdir -p logs

# Log file with timestamp
LOG_FILE="logs/daily_run_$(date +%Y%m%d_%H%M%S).log"

echo "🏀 Starting NBA Daily Dashboard Runner at $(date)" | tee -a "$LOG_FILE"
echo "=================================================" | tee -a "$LOG_FILE"

# Function to log and run commands
run_with_log() {
    echo "Running: $1" | tee -a "$LOG_FILE"
    eval "$1" 2>&1 | tee -a "$LOG_FILE"
    local exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ Success: $1" | tee -a "$LOG_FILE"
    else
        echo "❌ Failed: $1 (exit code: $exit_code)" | tee -a "$LOG_FILE"
    fi
    
    return $exit_code
}

# Main execution
main() {
    # Check if Python is available
    if ! command -v python &> /dev/null; then
        echo "❌ Python not found. Please install Python first." | tee -a "$LOG_FILE"
        exit 1
    fi
    
    # Check if required files exist
    if [ ! -f "daily_dashboard.py" ]; then
        echo "❌ daily_dashboard.py not found. Please ensure you're in the correct directory." | tee -a "$LOG_FILE"
        exit 1
    fi
    
    # Run the daily dashboard
    echo "Starting daily workflow..." | tee -a "$LOG_FILE"
    
    if run_with_log "python daily_dashboard.py daily"; then
        echo "🏀 Daily dashboard completed successfully!" | tee -a "$LOG_FILE"
        
        # Optional: Send notification (uncomment if you have notification setup)
        # notify-send "NBA Dashboard" "Daily workflow completed successfully"
        
    else
        echo "❌ Daily dashboard failed!" | tee -a "$LOG_FILE"
        
        # Optional: Send error notification
        # notify-send "NBA Dashboard" "Daily workflow failed - check logs"
        
        exit 1
    fi
    
    echo "=================================================" | tee -a "$LOG_FILE"
    echo "🏀 NBA Daily Dashboard Runner completed at $(date)" | tee -a "$LOG_FILE"
    
    # Clean up old log files (keep last 30 days)
    find logs -name "daily_run_*.log" -mtime +30 -delete 2>/dev/null || true
}

# Help function
show_help() {
    echo "NBA Daily Dashboard Runner"
    echo ""
    echo "Usage:"
    echo "  ./run_daily.sh        - Run the daily workflow"
    echo "  ./run_daily.sh -h     - Show this help"
    echo "  ./run_daily.sh --help - Show this help"
    echo ""
    echo "To set up as a daily cron job:"
    echo "  crontab -e"
    echo "  Add line: 0 9 * * * /path/to/nba-stat-predictor/run_daily.sh"
    echo "  (This runs daily at 9 AM)"
    echo ""
    echo "Log files are stored in: logs/daily_run_YYYYMMDD_HHMMSS.log"
}

# Check arguments
case "${1:-}" in
    -h|--help)
        show_help
        exit 0
        ;;
    "")
        main
        ;;
    *)
        echo "Unknown argument: $1"
        echo "Use -h or --help for usage information"
        exit 1
        ;;
esac 