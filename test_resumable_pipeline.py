#!/usr/bin/env python3
"""
Test script for the resumable NBA data collection pipeline.
This script demonstrates how to use the new resumable functionality.
"""

import sys
import os
import time
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from main import NBAStatPredictorApp
from src.data.nba_data_collector import NBADataCollector


def test_resumable_functionality():
    """Test the resumable data collection functionality."""
    print("🧪 TESTING RESUMABLE PIPELINE FUNCTIONALITY")
    print("=" * 60)
    
    # Initialize the app
    app = NBAStatPredictorApp()
    data_collector = NBADataCollector()
    
    # Test 1: List sessions (should be empty initially)
    print("\n1️⃣ Testing session listing (should be empty initially):")
    sessions = data_collector.list_sessions()
    print(f"   Found {len(sessions)} existing sessions")
    
    # Test 2: Create a new session
    print("\n2️⃣ Testing session creation:")
    session_id = data_collector.create_session("test_session")
    print(f"   Created session: {session_id}")
    
    # Test 3: List sessions again (should show the new session)
    print("\n3️⃣ Testing session listing after creation:")
    sessions = data_collector.list_sessions()
    print(f"   Found {len(sessions)} sessions")
    for session in sessions:
        print(f"   - {session['session_id']} ({session['status']})")
    
    # Test 4: Test checkpoint saving and loading
    print("\n4️⃣ Testing checkpoint functionality:")
    
    # Save a checkpoint
    session_data = {
        "session_id": session_id,
        "start_time": datetime.now().isoformat(),
        "status": "running",
        "progress": {
            "completed_players": [123, 456, 789],
            "failed_players": [999],
            "skipped_players": [],
            "current_season": "2023-24",
            "current_player_index": 5,
            "total_operations": 100,
            "completed_operations": 25,
            "games_collected": 150,
            "last_checkpoint": datetime.now().isoformat()
        }
    }
    data_collector._save_checkpoint(session_data)
    print(f"   Saved checkpoint for session {session_id}")
    
    # Load the checkpoint
    loaded_data = data_collector._load_checkpoint(session_id)
    if loaded_data:
        progress = loaded_data.get("progress", {})
        print(f"   Loaded checkpoint: {progress.get('completed_operations', 0)}/{progress.get('total_operations', 0)} operations completed")
        print(f"   Current season: {progress.get('current_season', 'N/A')}")
        print(f"   Games collected: {progress.get('games_collected', 0)}")
    
    # Test 5: Complete the session
    print("\n5️⃣ Testing session completion:")
    data_collector.complete_session(True)
    print(f"   Marked session {session_id} as completed")
    
    # Test 6: List sessions again (should show completed session)
    print("\n6️⃣ Testing session listing after completion:")
    sessions = data_collector.list_sessions()
    for session in sessions:
        print(f"   - {session['session_id']} ({session['status']})")
    
    print("\n✅ All resumable functionality tests completed!")
    print("\n💡 To test actual resumable collection:")
    print("   python main.py resumable-collect --players-limit 10")
    print("   (Interrupt with Ctrl+C, then resume with:)")
    print("   python main.py resume-session --session-id <SESSION_ID>")


def test_small_resumable_collection():
    """Test a small resumable collection that can be interrupted."""
    print("\n🧪 TESTING SMALL RESUMABLE COLLECTION")
    print("=" * 60)
    
    print("This will start a small collection (5 players) that you can interrupt.")
    print("Press Ctrl+C to interrupt, then resume it later.")
    print()
    
    try:
        # Start a small resumable collection
        app = NBAStatPredictorApp()
        result = app.collect_data_resumable(
            players_limit=5,  # Very small for testing
            session_name="test_small_collection"
        )
        
        print(f"\n✅ Collection completed!")
        print(f"Session ID: {result.get('session_id')}")
        print(f"Status: {result.get('status')}")
        print(f"Games collected: {result.get('total_games_collected', 0)}")
        
    except KeyboardInterrupt:
        print("\n🛑 Collection interrupted by user")
        print("💡 You can resume this session later using:")
        print("   python main.py list-sessions")
        print("   python main.py resume-session --session-id <SESSION_ID>")


if __name__ == "__main__":
    print("🏀 NBA Stat Predictor - Resumable Pipeline Test")
    print("=" * 60)
    
    # Run basic functionality tests
    test_resumable_functionality()
    
    # Ask if user wants to test actual collection
    print("\n" + "=" * 60)
    choice = input("Do you want to test a small resumable collection? (y/N): ").strip().lower()
    
    if choice == 'y':
        test_small_resumable_collection()
    else:
        print("\n💡 To test resumable collection manually:")
        print("   python main.py resumable-collect --players-limit 10")
        print("   python main.py list-sessions")
        print("   python main.py resume-session --session-id <SESSION_ID>")
    
    print("\n✅ Test completed!") 