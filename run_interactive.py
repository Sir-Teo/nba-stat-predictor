#!/usr/bin/env python3
"""
Simple launcher for the Interactive NBA Stat Predictor Dashboard
"""

import os
import sys

# Add src to path if needed
if "src" not in sys.path:
    sys.path.append("src")

# Import and run the interactive dashboard
try:
    from interactive_dashboard import main

    if __name__ == "__main__":
        print("[LAUNCH] Starting Interactive NBA Stat Predictor Dashboard...")
        main()

except ImportError as e:
    print(f"[ERROR] Error importing dashboard: {e}")
    print("Make sure all dependencies are installed:")
    print("pip install -r requirements.txt")

except Exception as e:
    print(f"[ERROR] Error running dashboard: {e}")
    print("Check that the NBA API is accessible and try again.")
