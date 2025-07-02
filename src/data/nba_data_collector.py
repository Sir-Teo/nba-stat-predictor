"""
NBA Data Collector - Handles data collection from NBA APIs and other sources.
"""

import logging
import sqlite3
import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from nba_api.stats.endpoints import (
    commonplayerinfo,
    leaguegamefinder,
    leaguestandings,
    playergamelog,
    scoreboardv2,
    teamgamelog,
)
from nba_api.stats.static import players, teams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NBADataCollector:
    """Collects and processes NBA data from various sources."""

    def __init__(self, db_path: str = "data/nba_data.db"):
        """Initialize the data collector."""
        self.db_path = db_path
        self.setup_database()
        
        # Rate limiting state
        self.request_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.base_delay = 2.0  # Start with 2 seconds (30 req/min)
        self.last_request_time = 0
        
        # Circuit breaker state
        self.consecutive_failures = 0
        self.circuit_breaker_threshold = 5  # Trip circuit breaker after 5 consecutive failures
        self.circuit_breaker_delay = 30  # Wait 30 seconds when circuit breaker is tripped
        self.circuit_breaker_active = False
        self.last_failure_time = 0
        
        # Resumable pipeline state
        self.checkpoint_dir = "data/checkpoints"
        self.ensure_checkpoint_dir()
        self.current_session_id = None
        self.session_start_time = None

    def ensure_checkpoint_dir(self):
        """Ensure checkpoint directory exists."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def create_session(self, session_name: str = None) -> str:
        """Create a new collection session with unique ID."""
        if session_name is None:
            session_name = f"collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session_id = session_name
        self.session_start_time = datetime.now().isoformat()
        
        # Create session checkpoint file
        session_data = {
            "session_id": self.current_session_id,
            "start_time": self.session_start_time,
            "status": "running",
            "progress": {
                "completed_players": [],
                "failed_players": [],
                "skipped_players": [],
                "current_season": None,
                "current_player_index": 0,
                "total_operations": 0,
                "completed_operations": 0,
                "games_collected": 0,
                "last_checkpoint": datetime.now().isoformat()
            },
            "settings": {},
            "rate_limit_stats": {
                "request_count": 0,
                "success_count": 0,
                "failure_count": 0,
                "base_delay": self.base_delay,
                "consecutive_failures": 0,
                "circuit_breaker_active": False
            }
        }
        
        self._save_checkpoint(session_data)
        logger.info(f"🔄 Created new collection session: {self.current_session_id}")
        return self.current_session_id

    def _save_checkpoint(self, session_data: Dict):
        """Save current session state to checkpoint file."""
        if self.current_session_id is None:
            return
            
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.current_session_id}.json")
        
        # Update progress with current state
        session_data["progress"]["last_checkpoint"] = datetime.now().isoformat()
        session_data["rate_limit_stats"] = {
            "request_count": self.request_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "base_delay": self.base_delay,
            "consecutive_failures": self.consecutive_failures,
            "circuit_breaker_active": self.circuit_breaker_active
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
        
        logger.debug(f"💾 Checkpoint saved: {checkpoint_path}")

    def _load_checkpoint(self, session_id: str) -> Optional[Dict]:
        """Load session state from checkpoint file."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{session_id}.json")
        
        if not os.path.exists(checkpoint_path):
            return None
            
        try:
            with open(checkpoint_path, 'r') as f:
                session_data = json.load(f)
            
            # Restore rate limiting state
            rate_stats = session_data.get("rate_limit_stats", {})
            self.request_count = rate_stats.get("request_count", 0)
            self.success_count = rate_stats.get("success_count", 0)
            self.failure_count = rate_stats.get("failure_count", 0)
            self.base_delay = rate_stats.get("base_delay", 2.0)
            self.consecutive_failures = rate_stats.get("consecutive_failures", 0)
            self.circuit_breaker_active = rate_stats.get("circuit_breaker_active", False)
            
            logger.info(f"📂 Loaded checkpoint for session: {session_id}")
            return session_data
            
        except Exception as e:
            logger.error(f"❌ Error loading checkpoint {session_id}: {e}")
            return None

    def list_sessions(self) -> List[Dict]:
        """List all available collection sessions."""
        sessions = []
        
        for filename in os.listdir(self.checkpoint_dir):
            if filename.endswith('.json'):
                session_id = filename[:-5]  # Remove .json extension
                checkpoint_path = os.path.join(self.checkpoint_dir, filename)
                
                try:
                    with open(checkpoint_path, 'r') as f:
                        session_data = json.load(f)
                    
                    sessions.append({
                        "session_id": session_id,
                        "start_time": session_data.get("start_time"),
                        "status": session_data.get("status", "unknown"),
                        "progress": session_data.get("progress", {}),
                        "last_checkpoint": session_data.get("progress", {}).get("last_checkpoint")
                    })
                except Exception as e:
                    logger.warning(f"Could not read session {session_id}: {e}")
        
        # Sort by start time (newest first), handle None values
        def sort_key(x):
            st = x.get("start_time")
            return st if st is not None else ""
        sessions.sort(key=sort_key, reverse=True)
        return sessions

    def resume_session(self, session_id: str) -> bool:
        """Resume a previous collection session."""
        session_data = self._load_checkpoint(session_id)
        if session_data is None:
            return False
        
        self.current_session_id = session_id
        progress = session_data.get("progress", {})
        
        logger.info(f"🔄 Resuming session {session_id}")
        logger.info(f"   Previous progress: {progress.get('completed_operations', 0)}/{progress.get('total_operations', 0)} operations")
        logger.info(f"   Completed players: {len(progress.get('completed_players', []))}")
        logger.info(f"   Failed players: {len(progress.get('failed_players', []))}")
        
        return True

    def complete_session(self, success: bool = True):
        """Mark current session as completed."""
        if self.current_session_id is None:
            return
            
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.current_session_id}.json")
        
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, 'r') as f:
                    session_data = json.load(f)
                
                session_data["status"] = "completed" if success else "failed"
                session_data["end_time"] = datetime.now().isoformat()
                session_data["progress"]["last_checkpoint"] = datetime.now().isoformat()
                
                with open(checkpoint_path, 'w') as f:
                    json.dump(session_data, f, indent=2, default=str)
                
                logger.info(f"✅ Session {self.current_session_id} marked as {'completed' if success else 'failed'}")
                
            except Exception as e:
                logger.error(f"❌ Error completing session {self.current_session_id}: {e}")

    def collect_historical_data_resumable(
        self,
        players_list: List[int],
        seasons: List[str] = ["2024-25", "2023-24", "2022-23", "2021-22", "2020-21", "2019-20", "2018-19", "2017-18"],
        include_playoffs: bool = True,
        include_all_star: bool = False,
        force_refresh: bool = False,
        session_name: str = None,
        resume_session_id: str = None,
        auto_save_interval: int = 10,  # Save checkpoint every N operations
    ) -> Dict:
        """Collect historical data with full resumable functionality."""
        
        # Initialize or resume session
        if resume_session_id:
            if not self.resume_session(resume_session_id):
                raise ValueError(f"Could not resume session {resume_session_id}")
            session_data = self._load_checkpoint(resume_session_id)
            progress = session_data.get("progress", {})
            
            # Resume from where we left off
            completed_players = set(progress.get("completed_players", []))
            failed_players = set(progress.get("failed_players", []))
            skipped_players = set(progress.get("skipped_players", []))
            current_season = progress.get("current_season")
            current_player_index = progress.get("current_player_index", 0)
            total_games_collected = progress.get("games_collected", 0)
            completed_operations = progress.get("completed_operations", 0)
            
            # Filter out already processed players
            remaining_players = [p for p in players_list if p not in completed_players and p not in failed_players]
            
            logger.info(f"🔄 Resuming from season {current_season}, player index {current_player_index}")
            logger.info(f"   Remaining players: {len(remaining_players)}")
            
        else:
            # Start new session
            self.create_session(session_name)
            session_data = {
                "progress": {
                    "completed_players": [],
                    "failed_players": [],
                    "skipped_players": [],
                    "current_season": None,
                    "current_player_index": 0,
                    "total_operations": len(players_list) * len(seasons),
                    "completed_operations": 0,
                    "games_collected": 0,
                    "last_checkpoint": datetime.now().isoformat()
                },
                "settings": {
                    "players_list": players_list,
                    "seasons": seasons,
                    "include_playoffs": include_playoffs,
                    "include_all_star": include_all_star,
                    "force_refresh": force_refresh
                }
            }
            
            completed_players = set()
            failed_players = set()
            skipped_players = set()
            current_season = None
            current_player_index = 0
            total_games_collected = 0
            completed_operations = 0
            remaining_players = players_list

        # Pre-analyze existing data to avoid unnecessary API calls
        existing_data_summary = self._analyze_existing_data(remaining_players, seasons)
        logger.info(f"Existing data analysis: {existing_data_summary['total_players_with_data']} players already have data")

        successful_players = 0
        total_operations = len(remaining_players) * len(seasons)
        
        # Add batch processing to reduce API load
        batch_size = 10  # Process players in batches
        logger.info(f"Using batch processing with size {batch_size} to reduce API load")
        
        # Track problematic players to skip
        problematic_players = set()
        max_failures_per_player = 3

        try:
            for season_idx, season in enumerate(seasons):
                # Skip seasons that were already completed
                if resume_session_id and current_season and seasons.index(current_season) > season_idx:
                    logger.info(f"Skipping already completed season {season}")
                    continue
                
                current_season = season
                logger.info(f"Collecting data for season {season}")
                season_games = 0
                season_players = 0

                # Determine starting player index for this season
                if resume_session_id and season == current_season:
                    start_player_index = current_player_index
                else:
                    start_player_index = 0

                for player_idx in range(start_player_index, len(remaining_players)):
                    player_id = remaining_players[player_idx]
                    current_player_index = player_idx
                    
                    # Skip problematic players
                    if player_id in problematic_players:
                        logger.debug(f"Skipping problematic player {player_id} for {season}")
                        skipped_players.add(player_id)
                        completed_operations += 1
                        continue
                    
                    try:
                        # Check if we need to collect data for this player in this season
                        if not force_refresh and self._player_has_sufficient_data_for_season(player_id, season):
                            logger.debug(f"Player {player_id} already has sufficient data for {season}, skipping")
                            skipped_players.add(player_id)
                            completed_operations += 1
                            continue

                        # Collect regular season data
                        df = self.get_player_game_logs(player_id, season, "Regular Season")
                        
                        # Collect playoff data if requested
                        if include_playoffs:
                            try:
                                playoff_df = self.get_player_game_logs(player_id, season, "Playoffs")
                                if not playoff_df.empty:
                                    df = pd.concat([df, playoff_df], ignore_index=True)
                                    logger.debug(f"Added {len(playoff_df)} playoff games for player {player_id}")
                            except Exception as e:
                                logger.debug(f"No playoff data for player {player_id} in {season}: {e}")

                        # Filter out All-Star games if requested
                        if not include_all_star and not df.empty:
                            df = df[~df["matchup"].str.contains("All-Star", na=False)]

                        # Process and store the data
                        if not df.empty:
                            processed_df = self._process_game_log_data(df)
                            
                            # Check for existing games to avoid duplicates
                            new_games_count = self._check_existing_games(player_id, season, processed_df)
                            if new_games_count > 0:
                                # Filter to only new games
                                filtered_df = self._filter_new_games(player_id, season, processed_df)
                                
                                if not filtered_df.empty:
                                    # Store the data
                                    for _, row in filtered_df.iterrows():
                                        game_data = row.to_dict()
                                        game_data["player_id"] = player_id
                                        if self.store_player_game_data(game_data):
                                            total_games_collected += 1
                                            season_games += 1
                                    
                                    logger.debug(f"Stored {len(filtered_df)} new games for player {player_id} in {season}")
                                else:
                                    logger.debug(f"No new games to store for player {player_id} in {season}")
                            else:
                                logger.debug(f"All games for player {player_id} in {season} already exist")

                            successful_players += 1
                            season_players += 1
                            completed_players.add(player_id)
                        else:
                            logger.debug(f"No data found for player {player_id} in {season}")
                            skipped_players.add(player_id)

                    except Exception as e:
                        logger.error(f"Error collecting data for player {player_id} in {season}: {e}")
                        failed_players.add(player_id)
                        
                        # Track failures for problematic players
                        if player_id not in problematic_players:
                            problematic_players.add(player_id)
                    
                    completed_operations += 1
                    
                    # Save checkpoint periodically
                    if completed_operations % auto_save_interval == 0:
                        session_data["progress"] = {
                            "completed_players": list(completed_players),
                            "failed_players": list(failed_players),
                            "skipped_players": list(skipped_players),
                            "current_season": current_season,
                            "current_player_index": current_player_index,
                            "total_operations": total_operations,
                            "completed_operations": completed_operations,
                            "games_collected": total_games_collected,
                            "last_checkpoint": datetime.now().isoformat()
                        }
                        self._save_checkpoint(session_data)
                        
                        # Log progress
                        progress_pct = (completed_operations / total_operations) * 100
                        logger.info(f"Progress: {completed_operations}/{total_operations} operations ({progress_pct:.1f}%)")
                        logger.info(f"Rate limit stats: {(self.success_count/(self.request_count+1)*100):.1f}% success rate, {self.base_delay:.2f}s delay, {self.consecutive_failures} consecutive failures")

                logger.info(f"Season {season} complete: {season_games} games from {season_players} players")
                logger.info(f"Progress: {completed_operations}/{total_operations} operations completed")
                
                # Add longer delay between seasons to avoid overwhelming the API
                if season != seasons[-1]:  # Don't delay after the last season
                    logger.info("Adding 5-second delay between seasons to respect API limits...")
                    time.sleep(5)

            # Final checkpoint save
            session_data["progress"] = {
                "completed_players": list(completed_players),
                "failed_players": list(failed_players),
                "skipped_players": list(skipped_players),
                "current_season": current_season,
                "current_player_index": current_player_index,
                "total_operations": total_operations,
                "completed_operations": completed_operations,
                "games_collected": total_games_collected,
                "last_checkpoint": datetime.now().isoformat()
            }
            self._save_checkpoint(session_data)

            # Mark session as completed
            self.complete_session(True)

            logger.info("=" * 60)
            logger.info("RESUMABLE DATA COLLECTION SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Session ID: {self.current_session_id}")
            logger.info(f"Total games collected: {total_games_collected:,}")
            logger.info(f"Successful players: {successful_players}")
            logger.info(f"Failed players: {len(failed_players)}")
            logger.info(f"Skipped players (existing data): {len(skipped_players)}")
            logger.info(f"Seasons covered: {len(seasons)}")
            if successful_players > 0:
                logger.info(f"Average games per player: {total_games_collected / successful_players:.1f}")
            logger.info("=" * 60)

            return {
                "session_id": self.current_session_id,
                "total_games_collected": total_games_collected,
                "successful_players": successful_players,
                "failed_players": len(failed_players),
                "skipped_players": len(skipped_players),
                "seasons_covered": len(seasons),
                "status": "completed"
            }

        except KeyboardInterrupt:
            logger.info("🛑 Collection interrupted by user")
            # Save current state
            session_data["progress"] = {
                "completed_players": list(completed_players),
                "failed_players": list(failed_players),
                "skipped_players": list(skipped_players),
                "current_season": current_season,
                "current_player_index": current_player_index,
                "total_operations": total_operations,
                "completed_operations": completed_operations,
                "games_collected": total_games_collected,
                "last_checkpoint": datetime.now().isoformat()
            }
            self._save_checkpoint(session_data)
            self.complete_session(False)
            
            return {
                "session_id": self.current_session_id,
                "total_games_collected": total_games_collected,
                "successful_players": successful_players,
                "failed_players": len(failed_players),
                "skipped_players": len(skipped_players),
                "status": "interrupted",
                "resume_session_id": self.current_session_id
            }

        except Exception as e:
            logger.error(f"❌ Collection failed: {e}")
            # Save current state
            session_data["progress"] = {
                "completed_players": list(completed_players),
                "failed_players": list(failed_players),
                "skipped_players": list(skipped_players),
                "current_season": current_season,
                "current_player_index": current_player_index,
                "total_operations": total_operations,
                "completed_operations": completed_operations,
                "games_collected": total_games_collected,
                "last_checkpoint": datetime.now().isoformat()
            }
            self._save_checkpoint(session_data)
            self.complete_session(False)
            
            return {
                "session_id": self.current_session_id,
                "total_games_collected": total_games_collected,
                "successful_players": successful_players,
                "failed_players": len(failed_players),
                "skipped_players": len(skipped_players),
                "status": "failed",
                "error": str(e),
                "resume_session_id": self.current_session_id
            }

    def setup_database(self):
        """Set up SQLite database for storing NBA data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create tables for different data types
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS player_games (
                game_id TEXT,
                player_id INTEGER,
                player_name TEXT,
                team_id INTEGER,
                team_abbreviation TEXT,
                game_date TEXT,
                matchup TEXT,
                wl TEXT,
                min REAL,
                pts INTEGER,
                reb INTEGER,
                ast INTEGER,
                stl INTEGER,
                blk INTEGER,
                tov INTEGER,
                pf INTEGER,
                fgm INTEGER,
                fga INTEGER,
                fg_pct REAL,
                fg3m INTEGER,
                fg3a INTEGER,
                fg3_pct REAL,
                ftm INTEGER,
                fta INTEGER,
                ft_pct REAL,
                plus_minus INTEGER,
                created_at TEXT,
                PRIMARY KEY (game_id, player_id)
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT,
                player_id INTEGER,
                player_name TEXT,
                game_date TEXT,
                stat_type TEXT,
                predicted_value REAL,
                actual_value REAL,
                confidence REAL,
                model_version TEXT,
                created_at TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version TEXT,
                stat_type TEXT,
                mae REAL,
                rmse REAL,
                accuracy_rate REAL,
                evaluation_date TEXT,
                sample_size INTEGER
            )
        """
        )

        # Create basic teams table for advanced features
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS teams (
                team_id INTEGER PRIMARY KEY,
                team_name TEXT,
                team_abbreviation TEXT,
                conference TEXT,
                division TEXT
            )
        """
        )

        # Insert basic team data if table is empty
        cursor.execute("SELECT COUNT(*) FROM teams")
        if cursor.fetchone()[0] == 0:
            # Insert some basic NBA teams (simplified for demo)
            teams_data = [
                (1610612737, "Atlanta Hawks", "ATL", "Eastern", "Southeast"),
                (1610612738, "Boston Celtics", "BOS", "Eastern", "Atlantic"),
                (1610612739, "Cleveland Cavaliers", "CLE", "Eastern", "Central"),
                (1610612740, "New Orleans Pelicans", "NOP", "Western", "Southwest"),
                (1610612741, "Chicago Bulls", "CHI", "Eastern", "Central"),
                (1610612742, "Dallas Mavericks", "DAL", "Western", "Southwest"),
                (1610612743, "Denver Nuggets", "DEN", "Western", "Northwest"),
                (1610612744, "Golden State Warriors", "GSW", "Western", "Pacific"),
                (1610612745, "Houston Rockets", "HOU", "Western", "Southwest"),
                (1610612746, "LA Clippers", "LAC", "Western", "Pacific"),
                (1610612747, "Los Angeles Lakers", "LAL", "Western", "Pacific"),
                (1610612748, "Miami Heat", "MIA", "Eastern", "Southeast"),
                (1610612749, "Milwaukee Bucks", "MIL", "Eastern", "Central"),
                (1610612750, "Minnesota Timberwolves", "MIN", "Western", "Northwest"),
                (1610612751, "Brooklyn Nets", "BKN", "Eastern", "Atlantic"),
                (1610612752, "New York Knicks", "NYK", "Eastern", "Atlantic"),
                (1610612753, "Orlando Magic", "ORL", "Eastern", "Southeast"),
                (1610612754, "Indiana Pacers", "IND", "Eastern", "Central"),
                (1610612755, "Philadelphia 76ers", "PHI", "Eastern", "Atlantic"),
                (1610612756, "Phoenix Suns", "PHX", "Western", "Pacific"),
                (1610612757, "Portland Trail Blazers", "POR", "Western", "Northwest"),
                (1610612758, "Sacramento Kings", "SAC", "Western", "Pacific"),
                (1610612759, "San Antonio Spurs", "SAS", "Western", "Southwest"),
                (1610612760, "Oklahoma City Thunder", "OKC", "Western", "Northwest"),
                (1610612761, "Toronto Raptors", "TOR", "Eastern", "Atlantic"),
                (1610612762, "Utah Jazz", "UTA", "Western", "Northwest"),
                (1610612763, "Memphis Grizzlies", "MEM", "Western", "Southwest"),
                (1610612764, "Washington Wizards", "WAS", "Eastern", "Southeast"),
                (1610612765, "Detroit Pistons", "DET", "Eastern", "Central"),
                (1610612766, "Charlotte Hornets", "CHA", "Eastern", "Southeast"),
            ]

            cursor.executemany(
                "INSERT OR IGNORE INTO teams (team_id, team_name, team_abbreviation, conference, division) VALUES (?, ?, ?, ?, ?)",
                teams_data,
            )

        conn.commit()
        conn.close()

    def get_all_players(self) -> pd.DataFrame:
        """Get all NBA players."""
        try:
            all_players = players.get_players()
            return pd.DataFrame(all_players)
        except Exception as e:
            logger.error(f"Error fetching players: {e}")
            return pd.DataFrame()

    def get_player_game_logs(
        self,
        player_id: int,
        season: str = "2023-24",
        season_type: str = "Regular Season",
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> pd.DataFrame:
        """Get game logs for a specific player with robust error handling and retries."""
        for attempt in range(max_retries):
            try:
                # Adaptive rate limiting
                adaptive_delay = self._get_adaptive_delay()
                time.sleep(adaptive_delay)
                
                # Import here to avoid issues
                from nba_api.stats.endpoints import playergamelog
                
                # Set longer timeout for problematic seasons/players
                timeout_multiplier = 1.0
                if season == "2021-22":
                    timeout_multiplier = 2.0  # Double timeout for 2021-22 season
                
                player_gamelog_obj = playergamelog.PlayerGameLog(
                    player_id=player_id, season=season, season_type_all_star=season_type
                )
                
                # Use longer delays for problematic seasons
                if season == "2021-22":
                    time.sleep(3.0)  # Extra delay for 2021-22 season
                
                df = player_gamelog_obj.get_data_frames()[0]
                
                # Handle case where API returns empty DataFrame
                if df.empty:
                    logger.debug(f"No data returned for player {player_id} in {season} {season_type}")
                    self._adjust_rate_limit(True)  # Count as success (no error)
                    return pd.DataFrame()
                
                df["PLAYER_ID"] = player_id
                
                # Validate the data - be very flexible with column requirements
                if not df.empty:
                    # Log available columns for debugging (only for first few players to avoid spam)
                    if player_id % 50 == 0:  # Log every 50th player
                        logger.info(f"Sample columns for player {player_id} in {season}: {list(df.columns)}")
                    
                    # Check for reasonable data size
                    if len(df) > 100:  # NBA season is typically 82 games max
                        logger.debug(f"Large dataset for player {player_id} in {season}: {len(df)} games")
                    
                                    # Only require basic columns that should always be present
                # The NBA API sometimes returns different column names or structures
                if len(df) == 0:
                    logger.debug(f"Empty dataset for player {player_id} in {season}")
                    self._adjust_rate_limit(True)  # Count as success (no error)
                    return pd.DataFrame()
                
                # Track successful request
                self._adjust_rate_limit(True)
                return df

            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for player {player_id} in {season}: {error_msg}")
                
                # Classify errors for better handling
                error_msg_lower = error_msg.lower()
                
                # Don't retry on certain errors
                if "not found" in error_msg_lower or "invalid" in error_msg_lower:
                    logger.debug(f"Player {player_id} not found or invalid for {season}, skipping")
                    break
                
                # Rate limit and timeout errors - retry with longer delays
                if any(keyword in error_msg_lower for keyword in ["timeout", "connection", "aborted"]):
                    logger.warning(f"Rate limit/timeout detected for player {player_id} in {season} (attempt {attempt + 1})")
                    # Use longer delays for rate limit issues
                    retry_delay = 4.0  # Double the base delay for rate limit issues
                    
                    # Progressive backoff for consecutive failures
                    if self.consecutive_failures > 3:
                        retry_delay = min(15.0, retry_delay * (self.consecutive_failures - 2))
                        logger.warning(f"Progressive backoff: {retry_delay:.1f}s delay due to {self.consecutive_failures} consecutive failures")
                
                # Retry with exponential backoff and jitter
                if attempt < max_retries - 1:
                    import random
                    base_sleep = retry_delay * (2 ** attempt)
                    jitter = random.uniform(0.5, 1.5)  # Add 50% random variation
                    sleep_time = base_sleep * jitter
                    logger.debug(f"Retrying in {sleep_time:.1f} seconds (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"All {max_retries} attempts failed for player {player_id} in {season}: {error_msg}")
                    self._adjust_rate_limit(False)  # Track final failure
        
        return pd.DataFrame()

    def get_todays_games(self) -> pd.DataFrame:
        """Get today's NBA games."""
        try:
            today = datetime.now().strftime("%m/%d/%Y")
            scoreboard = scoreboardv2.ScoreboardV2(game_date=today)
            games_df = scoreboard.get_data_frames()[0]  # GameHeader
            return games_df
        except Exception as e:
            logger.error(f"Error fetching today's games: {e}")
            return pd.DataFrame()

    def collect_historical_data(
        self,
        players_list: List[int],
        seasons: List[str] = ["2024-25", "2023-24", "2022-23", "2021-22", "2020-21", "2019-20", "2018-19", "2017-18"],
        include_playoffs: bool = True,
        include_all_star: bool = False,
        force_refresh: bool = False,
    ) -> None:
        """Collect historical data for specified players and seasons with intelligent data checking."""
        conn = sqlite3.connect(self.db_path)

        # Pre-analyze existing data to avoid unnecessary API calls
        existing_data_summary = self._analyze_existing_data(players_list, seasons)
        logger.info(f"Existing data analysis: {existing_data_summary['total_players_with_data']} players already have data")

        total_games_collected = 0
        successful_players = 0
        failed_players = 0
        skipped_players = 0

        logger.info(f"Starting intelligent data collection for {len(players_list)} players across {len(seasons)} seasons")

        # Add a simple progress indicator
        total_operations = len(players_list) * len(seasons)
        completed_operations = 0
        
        # Add batch processing to reduce API load
        batch_size = 10  # Process players in batches
        logger.info(f"Using batch processing with size {batch_size} to reduce API load")
        
        # Track problematic players to skip
        problematic_players = set()
        max_failures_per_player = 3  # Skip player after 3 consecutive failures

        for season in seasons:
            logger.info(f"Collecting data for season {season}")
            season_games = 0
            season_players = 0

            for player_id in players_list:
                # Skip problematic players
                if player_id in problematic_players:
                    logger.debug(f"Skipping problematic player {player_id} for {season}")
                    skipped_players += 1
                    continue
                
                try:
                    # Check if we need to collect data for this player in this season
                    if not force_refresh and self._player_has_sufficient_data_for_season(player_id, season):
                        logger.debug(f"Player {player_id} already has sufficient data for {season}, skipping")
                        skipped_players += 1
                        continue

                    # Collect regular season data
                    df = self.get_player_game_logs(player_id, season, "Regular Season")
                    
                    # Collect playoff data if requested
                    if include_playoffs:
                        try:
                            playoff_df = self.get_player_game_logs(player_id, season, "Playoffs")
                            if not playoff_df.empty:
                                df = pd.concat([df, playoff_df], ignore_index=True)
                                logger.debug(f"Added {len(playoff_df)} playoff games for player {player_id}")
                        except Exception as e:
                            logger.debug(f"No playoff data for player {player_id} in {season}: {e}")
                    
                    # Filter out all-star games if requested (only if MATCHUP column exists)
                    if not include_all_star and not df.empty and 'MATCHUP' in df.columns:
                        df = df[~df['MATCHUP'].str.contains('All-Star', case=False, na=False)]
                    elif not include_all_star and not df.empty:
                        # If no MATCHUP column, try alternative column names
                        matchup_columns = ['Matchup', 'matchup', 'GAME_MATCHUP']
                        for col in matchup_columns:
                            if col in df.columns:
                                df = df[~df[col].str.contains('All-Star', case=False, na=False)]
                                break
                    
                    if not df.empty:
                        # Process and clean the data
                        processed_df = self._process_game_log_data(df)
                        
                        # Check if processing was successful
                        if processed_df.empty:
                            logger.debug(f"Failed to process data for player {player_id} in {season}")
                            continue
                        
                        # Check for sufficient data quality - be more flexible
                        if len(processed_df) >= 1:  # Minimum 1 game (some players only play a few games)
                            # Check for duplicates before inserting
                            existing_count = self._check_existing_games(player_id, season, processed_df)
                            if existing_count > 0:
                                logger.debug(f"Found {existing_count} existing games for player {player_id} in {season}")
                                # Only insert new games
                                new_games = self._filter_new_games(player_id, season, processed_df)
                                if not new_games.empty:
                                    new_games.to_sql("player_games", conn, if_exists="append", index=False)
                                    games_count = len(new_games)
                                    logger.info(f"Added {games_count} new games for player {player_id} in {season}")
                                else:
                                    logger.debug(f"No new games to add for player {player_id} in {season}")
                                    games_count = 0
                            else:
                                # Insert all games
                                processed_df.to_sql("player_games", conn, if_exists="append", index=False)
                                games_count = len(processed_df)
                                logger.info(f"Collected {games_count} games for player {player_id} in {season}")
                            
                            if games_count > 0:
                                season_games += games_count
                                total_games_collected += games_count
                                season_players += 1
                                successful_players += 1
                        else:
                            logger.debug(f"Insufficient data for player {player_id} in {season}: {len(processed_df)} games")
                    else:
                        logger.debug(f"No data found for player {player_id} in {season}")
                        
                except Exception as e:
                    error_msg = str(e).lower()
                    
                    # Don't log errors for players who simply didn't play in a season
                    if "not found" in error_msg or "no data" in error_msg or "invalid" in error_msg:
                        logger.debug(f"Player {player_id} not found or no data for {season}")
                    else:
                        logger.error(
                            f"Error collecting data for player {player_id} in {season}: {e}"
                        )
                        failed_players += 1
                        
                        # Track player failures for this season
                        player_failures = getattr(self, f'_player_failures_{season}', {})
                        player_failures[player_id] = player_failures.get(player_id, 0) + 1
                        setattr(self, f'_player_failures_{season}', player_failures)
                        
                        # Add to problematic players if too many failures
                        if player_failures[player_id] >= max_failures_per_player:
                            problematic_players.add(player_id)
                            logger.warning(f"Added player {player_id} to problematic list after {player_failures[player_id]} failures in {season}")
                    continue

            completed_operations += len(players_list)
            logger.info(f"Season {season} complete: {season_games} games from {season_players} players")
            logger.info(f"Progress: {completed_operations}/{total_operations} operations completed")
            
            # Add longer delay between seasons to avoid overwhelming the API
            if season != seasons[-1]:  # Don't delay after the last season
                logger.info("Adding 5-second delay between seasons to respect API limits...")
                time.sleep(5)

        logger.info("=" * 60)
        logger.info("INTELLIGENT DATA COLLECTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total games collected: {total_games_collected:,}")
        logger.info(f"Successful players: {successful_players}")
        logger.info(f"Failed players: {failed_players}")
        logger.info(f"Skipped players (existing data): {skipped_players}")
        logger.info(f"Seasons covered: {len(seasons)}")
        if successful_players > 0:
            logger.info(f"Average games per player: {total_games_collected / successful_players:.1f}")
        logger.info("=" * 60)
        
        conn.close()

    def _process_game_log_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean game log data with flexible column handling."""
        # Select and rename columns - be very flexible with column names
        columns_mapping = {
            "GAME_ID": "game_id",
            "PLAYER_ID": "player_id",
            "GAME_DATE": "game_date",
            "MATCHUP": "matchup",
            "WL": "wl",
            "MIN": "min",
            "PTS": "pts",
            "REB": "reb",
            "AST": "ast",
            "STL": "stl",
            "BLK": "blk",
            "TOV": "tov",
            "PF": "pf",
            "FGM": "fgm",
            "FGA": "fga",
            "FG_PCT": "fg_pct",
            "FG3M": "fg3m",
            "FG3A": "fg3a",
            "FG3_PCT": "fg3_pct",
            "FTM": "ftm",
            "FTA": "fta",
            "FT_PCT": "ft_pct",
            "PLUS_MINUS": "plus_minus",
        }

        # Handle different possible column names based on actual NBA API response
        column_variations = {
            "GAME_ID": ["GAME_ID", "Game_ID", "game_id"],
            "GAME_DATE": ["GAME_DATE", "Game_Date", "game_date"],
            "MATCHUP": ["MATCHUP", "Matchup", "matchup"],
            "PLAYER_ID": ["PLAYER_ID", "Player_ID", "player_id"],
            "MIN": ["MIN", "Min", "min"],
            "PTS": ["PTS", "Pts", "pts"],
            "REB": ["REB", "Reb", "reb"],
            "AST": ["AST", "Ast", "ast"],
            "STL": ["STL", "Stl", "stl"],
            "BLK": ["BLK", "Blk", "blk"],
            "TOV": ["TOV", "Tov", "tov"],
            "PF": ["PF", "Pf", "pf"],
            "FGM": ["FGM", "Fgm", "fgm"],
            "FGA": ["FGA", "Fga", "fga"],
            "FG_PCT": ["FG_PCT", "FG_PCT", "fg_pct"],
            "FG3M": ["FG3M", "Fg3m", "fg3m"],
            "FG3A": ["FG3A", "Fg3a", "fg3a"],
            "FG3_PCT": ["FG3_PCT", "FG3_PCT", "fg3_pct"],
            "FTM": ["FTM", "Ftm", "ftm"],
            "FTA": ["FTA", "Fta", "fta"],
            "FT_PCT": ["FT_PCT", "FT_PCT", "ft_pct"],
            "PLUS_MINUS": ["PLUS_MINUS", "Plus_Minus", "plus_minus"],
        }

        # Try to find the correct column names
        actual_columns = {}
        for standard_name, variations in column_variations.items():
            for variation in variations:
                if variation in df.columns:
                    actual_columns[standard_name] = variation
                    break

        # Update columns_mapping with actual column names found
        for standard_name, actual_name in actual_columns.items():
            if standard_name in columns_mapping:
                columns_mapping[actual_name] = columns_mapping[standard_name]

        # Select only columns that exist
        available_columns = {
            k: v for k, v in columns_mapping.items() if k in df.columns
        }
        
        if not available_columns:
            logger.debug(f"No recognizable columns found in data: {list(df.columns)}")
            return pd.DataFrame()
        
        processed_df = df[list(available_columns.keys())].rename(
            columns=available_columns
        )

        # Convert game_date to ISO format (YYYY-MM-DD)
        if "game_date" in processed_df.columns:
            try:
                # Try common NBA API format first (abbreviated month)
                processed_df["game_date"] = pd.to_datetime(
                    processed_df["game_date"], format="%b %d, %Y"
                ).dt.strftime("%Y-%m-%d")
            except Exception:
                try:
                    # Fallback: try full month name format
                    processed_df["game_date"] = pd.to_datetime(
                        processed_df["game_date"], format="%B %d, %Y"
                    ).dt.strftime("%Y-%m-%d")
                except Exception:
                    try:
                        # Try with errors='coerce' to handle mixed formats
                        processed_df["game_date"] = pd.to_datetime(
                            processed_df["game_date"], errors="coerce"
                        ).dt.strftime("%Y-%m-%d")
                    except Exception:
                        try:
                            # Final fallback: infer format
                            processed_df["game_date"] = pd.to_datetime(
                                processed_df["game_date"], infer_datetime_format=True
                            ).dt.strftime("%Y-%m-%d")
                        except Exception as e:
                            logger.warning(
                                f"Could not parse dates for some entries, keeping original format. Error: {e}"
                            )

                            # Convert individually for problematic dates
                            def safe_convert_date(date_str):
                                if pd.isna(date_str):
                                    return date_str
                                try:
                                    return pd.to_datetime(
                                        date_str, format="%b %d, %Y"
                                    ).strftime("%Y-%m-%d")
                                except:
                                    try:
                                        return pd.to_datetime(
                                            date_str, format="%B %d, %Y"
                                        ).strftime("%Y-%m-%d")
                                    except:
                                        try:
                                            return pd.to_datetime(date_str).strftime(
                                                "%Y-%m-%d"
                                            )
                                        except:
                                            logger.error(
                                                f"Failed to parse date: {date_str}"
                                            )
                                            return str(
                                                date_str
                                            )  # Keep original if all fails

                            processed_df["game_date"] = processed_df["game_date"].apply(
                                safe_convert_date
                            )

        # Add metadata
        processed_df["created_at"] = datetime.now().isoformat()

        # Handle missing values
        numeric_columns = [
            "min",
            "pts",
            "reb",
            "ast",
            "stl",
            "blk",
            "tov",
            "pf",
            "fgm",
            "fga",
            "fg_pct",
            "fg3m",
            "fg3a",
            "fg3_pct",
            "ftm",
            "fta",
            "ft_pct",
            "plus_minus",
        ]

        for col in numeric_columns:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(
                    processed_df[col], errors="coerce"
                ).fillna(0)

        return processed_df

    def _adjust_rate_limit(self, success: bool):
        """Adjust rate limiting based on success/failure with circuit breaker."""
        self.request_count += 1
        
        if success:
            self.success_count += 1
            self.consecutive_failures = 0  # Reset consecutive failures on success
            
            # If circuit breaker is active, deactivate it
            if self.circuit_breaker_active:
                logger.info("Circuit breaker deactivated - success detected")
                self.circuit_breaker_active = False
            
            # If we're doing well, gradually speed up (but stay conservative)
            if self.success_count > self.failure_count * 3 and self.base_delay > 1.5:
                self.base_delay = max(1.5, self.base_delay * 0.95)
                logger.debug(f"Rate limit adjusted: {self.base_delay:.2f}s (successful)")
        else:
            self.failure_count += 1
            self.consecutive_failures += 1
            
            # Check if we should trip the circuit breaker
            if self.consecutive_failures >= self.circuit_breaker_threshold and not self.circuit_breaker_active:
                logger.warning(f"Circuit breaker tripped after {self.consecutive_failures} consecutive failures")
                self.circuit_breaker_active = True
                self.last_failure_time = time.time()
                self.base_delay = 10.0  # Set very high delay when circuit breaker is active
            
            # If we're failing, slow down
            if self.failure_count > self.success_count * 0.5:
                self.base_delay = min(10.0, self.base_delay * 1.5)  # More aggressive slowdown
                logger.debug(f"Rate limit adjusted: {self.base_delay:.2f}s (failed)")
        
        # Reset counters periodically to adapt to changing conditions
        if self.request_count % 50 == 0:
            success_rate = self.success_count / max(1, self.request_count)
            logger.info(f"Rate limit stats: {success_rate:.1%} success rate, {self.base_delay:.2f}s delay, {self.consecutive_failures} consecutive failures")

    def _get_adaptive_delay(self):
        """Get adaptive delay based on current success rate with circuit breaker."""
        import random
        
        # Check if circuit breaker is active
        if self.circuit_breaker_active:
            # Check if enough time has passed to try again
            time_since_failure = time.time() - self.last_failure_time
            if time_since_failure < self.circuit_breaker_delay:
                remaining_wait = self.circuit_breaker_delay - time_since_failure
                logger.info(f"Circuit breaker active - waiting {remaining_wait:.1f} more seconds")
                time.sleep(remaining_wait)
            else:
                logger.info("Circuit breaker timeout expired - attempting recovery")
                self.circuit_breaker_active = False
                self.base_delay = 5.0  # Start with moderate delay after recovery
        
        # Add some jitter to avoid thundering herd
        jitter = random.uniform(0.8, 1.2)
        return self.base_delay * jitter

    def get_recent_games(self, player_id: int, days: int = 10) -> pd.DataFrame:
        """Get recent games for a player from the database."""
        conn = sqlite3.connect(self.db_path)

        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        query = """
            SELECT * FROM player_games 
            WHERE player_id = ? AND game_date >= ?
            ORDER BY game_date DESC
        """

        df = pd.read_sql_query(query, conn, params=(player_id, cutoff_date))
        conn.close()

        return df

    def get_player_stats_summary(self, player_id: int, games: int = 10) -> Dict:
        """Get statistical summary for a player over recent games."""
        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT * FROM player_games 
            WHERE player_id = ?
            ORDER BY game_date DESC
            LIMIT ?
        """

        df = pd.read_sql_query(query, conn, params=(player_id, games))
        conn.close()

        if df.empty:
            return {}

        stats = {}
        numeric_columns = [
            "pts",
            "reb",
            "ast",
            "stl",
            "blk",
            "tov",
            "fg_pct",
            "fg3_pct",
            "ft_pct",
        ]

        for col in numeric_columns:
            if col in df.columns:
                stats[f"{col}_avg"] = df[col].mean()
                stats[f"{col}_std"] = df[col].std()
                stats[f"{col}_trend"] = self._calculate_trend(df[col])

        return stats

    def _calculate_trend(self, series: pd.Series) -> float:
        """Calculate trend (slope) of a time series."""
        if len(series) < 2:
            return 0.0

        x = np.arange(len(series))
        y = series.values

        # Remove NaN values
        mask = ~np.isnan(y)
        if np.sum(mask) < 2:
            return 0.0

        x_clean, y_clean = x[mask], y[mask]
        slope = np.polyfit(x_clean, y_clean, 1)[0]

        return slope

    def store_prediction(
        self,
        game_id: str,
        player_id: int,
        player_name: str,
        game_date: str,
        stat_type: str,
        predicted_value: float,
        confidence: float,
        model_version: str,
    ) -> None:
        """Store a prediction in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO predictions 
            (game_id, player_id, player_name, game_date, stat_type, 
             predicted_value, confidence, model_version, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                game_id,
                player_id,
                player_name,
                game_date,
                stat_type,
                predicted_value,
                confidence,
                model_version,
                datetime.now().isoformat(),
            ),
        )

        conn.commit()
        conn.close()

    def update_prediction_with_actual(
        self, game_id: str, player_id: int, stat_type: str, actual_value: float
    ) -> None:
        """Update prediction with actual value after the game."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE predictions 
            SET actual_value = ?
            WHERE game_id = ? AND player_id = ? AND stat_type = ?
        """,
            (actual_value, game_id, player_id, stat_type),
        )

        conn.commit()
        conn.close()

    def get_popular_players(self, limit: int = 100) -> List[int]:
        """Get a comprehensive list of popular NBA players with enhanced selection."""
        try:
            # Get all players and filter for recent ones
            all_players = players.get_players()

            # Enhanced filtering for players with recent activity (2018 onwards)
            recent_players = []
            for player in all_players:
                # Include active players
                if player.get("is_active", False):
                    recent_players.append(player)
                # Include recently retired players (played until 2020 or later)
                elif player.get("to_year") and int(player.get("to_year", 0)) >= 2020:
                    recent_players.append(player)

            # Sort by ID (newer players generally have higher IDs)
            recent_players.sort(key=lambda x: x["id"], reverse=True)

            # Return top players by ID
            return [player["id"] for player in recent_players[:limit]]

        except Exception as e:
            logger.error(f"Error getting popular players: {e}")
            # Enhanced fallback list of known star player IDs
            return [
                2544,    # LeBron James
                201939,  # Stephen Curry
                203507,  # Giannis Antetokounmpo
                203999,  # Nikola Jokic
                1628369, # Jayson Tatum
                1627759, # Donovan Mitchell
                1629029, # Luka Doncic
                203076,  # Anthony Davis
                1628983, # Shai Gilgeous-Alexander
                1629630, # Zion Williamson
                203954,  # Joel Embiid
                1628378, # De'Aaron Fox
                1630173, # Paolo Banchero
                1628389, # Bam Adebayo
                201566,  # Russell Westbrook
                203081,  # Damian Lillard
                1627749, # Jaylen Brown
                1629027, # Trae Young
                1630162, # Chet Holmgren
                1629628, # RJ Barrett
                201142,  # Kevin Durant
                203076,  # Anthony Davis
                203954,  # Joel Embiid
                1628369, # Jayson Tatum
                1629029, # Luka Doncic
            ]

    def get_comprehensive_player_list(self, target_count: int = 200) -> List[int]:
        """Get a comprehensive list of players with good data coverage for enhanced training."""
        logger.info(f"Building comprehensive player list (target: {target_count})")
        
        try:
            # Get all active and recently active players
            all_players = players.get_players()
            
            # Filter for players with recent activity (2018 onwards)
            recent_players = []
            for player in all_players:
                # Include active players
                if player.get("is_active", False):
                    recent_players.append(player)
                # Include recently retired players (played until 2020 or later)
                elif player.get("to_year") and int(player.get("to_year", 0)) >= 2020:
                    recent_players.append(player)
            
            # Sort by ID (newer players generally have higher IDs)
            recent_players.sort(key=lambda x: x["id"], reverse=True)
            
            # Take top players
            selected_players = recent_players[:target_count]
            player_ids = [player["id"] for player in selected_players]
            
            logger.info(f"Selected {len(player_ids)} players for enhanced data collection")
            return player_ids
            
        except Exception as e:
            logger.error(f"Error getting comprehensive player list: {e}")
            return self.get_popular_players(target_count)

    def validate_data_quality(self) -> Dict:
        """Validate the quality of collected data with comprehensive checks."""
        logger.info("Validating data quality...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Check for missing values
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_rows,
                    SUM(CASE WHEN pts IS NULL THEN 1 ELSE 0 END) as null_pts,
                    SUM(CASE WHEN reb IS NULL THEN 1 ELSE 0 END) as null_reb,
                    SUM(CASE WHEN ast IS NULL THEN 1 ELSE 0 END) as null_ast,
                    SUM(CASE WHEN game_date IS NULL THEN 1 ELSE 0 END) as null_dates
                FROM player_games
            """)
            null_counts = cursor.fetchone()
            
            # Check for unrealistic values
            cursor.execute("""
                SELECT 
                    COUNT(*) as unrealistic_pts,
                    COUNT(*) as unrealistic_reb,
                    COUNT(*) as unrealistic_ast
                FROM player_games
                WHERE pts > 100 OR reb > 50 OR ast > 30
            """)
            unrealistic_counts = cursor.fetchone()
            
            # Check data consistency
            cursor.execute("""
                SELECT COUNT(*) 
                FROM player_games 
                WHERE game_date > date('now')
            """)
            future_games = cursor.fetchone()[0]
            
            conn.close()
            
            quality_report = {
                "total_rows": null_counts[0],
                "null_points": null_counts[1],
                "null_rebounds": null_counts[2],
                "null_assists": null_counts[3],
                "null_dates": null_counts[4],
                "unrealistic_values": sum(unrealistic_counts),
                "future_games": future_games,
                "data_quality_score": self._calculate_quality_score(null_counts, unrealistic_counts, future_games)
            }
            
            logger.info("Data quality validation complete")
            return quality_report
            
        except Exception as e:
            logger.error(f"Error validating data quality: {e}")
            return {"error": str(e)}
    
    def _calculate_quality_score(self, null_counts, unrealistic_counts, future_games) -> float:
        """Calculate a data quality score (0-100)."""
        total_rows = null_counts[0]
        if total_rows == 0:
            return 0.0
        
        # Calculate null percentage
        null_percentage = sum(null_counts[1:]) / total_rows
        
        # Calculate unrealistic percentage
        unrealistic_percentage = sum(unrealistic_counts) / total_rows
        
        # Calculate future games percentage
        future_percentage = future_games / total_rows
        
        # Quality score (higher is better)
        quality_score = 100 - (null_percentage * 50) - (unrealistic_percentage * 30) - (future_percentage * 20)
        
        return max(0.0, min(100.0, quality_score))

    def _analyze_existing_data(self, players_list: List[int], seasons: List[str]) -> Dict:
        """Analyze existing data to determine what needs to be collected."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get season date ranges
            season_ranges = self._get_season_date_ranges(seasons)
            
            total_players_with_data = 0
            players_data_summary = {}
            
            for player_id in players_list:
                player_games = 0
                player_seasons = 0
                
                for season, (start_date, end_date) in season_ranges.items():
                    cursor.execute("""
                        SELECT COUNT(*) FROM player_games 
                        WHERE player_id = ? AND game_date BETWEEN ? AND ?
                    """, (player_id, start_date, end_date))
                    
                    season_games = cursor.fetchone()[0]
                    if season_games > 0:
                        player_games += season_games
                        player_seasons += 1
                
                if player_games > 0:
                    total_players_with_data += 1
                    players_data_summary[player_id] = {
                        'total_games': player_games,
                        'seasons': player_seasons
                    }
            
            conn.close()
            
            return {
                'total_players_with_data': total_players_with_data,
                'players_data_summary': players_data_summary
            }
            
        except Exception as e:
            logger.error(f"Error analyzing existing data: {e}")
            return {'total_players_with_data': 0, 'players_data_summary': {}}

    def _player_has_sufficient_data_for_season(self, player_id: int, season: str, min_games: int = 1) -> bool:
        """Check if a player already has sufficient data for a specific season."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get season date range
            season_ranges = self._get_season_date_ranges([season])
            if season not in season_ranges:
                conn.close()
                return False
            
            start_date, end_date = season_ranges[season]
            
            cursor.execute("""
                SELECT COUNT(*) FROM player_games 
                WHERE player_id = ? AND game_date BETWEEN ? AND ?
            """, (player_id, start_date, end_date))
            
            existing_games = cursor.fetchone()[0]
            conn.close()
            
            return existing_games >= min_games
            
        except Exception as e:
            logger.error(f"Error checking existing data for player {player_id}: {e}")
            return False

    def _check_existing_games(self, player_id: int, season: str, new_games_df: pd.DataFrame) -> int:
        """Check how many games already exist for a player in a season."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get season date range
            season_ranges = self._get_season_date_ranges([season])
            if season not in season_ranges:
                conn.close()
                return 0
            
            start_date, end_date = season_ranges[season]
            
            cursor.execute("""
                SELECT COUNT(*) FROM player_games 
                WHERE player_id = ? AND game_date BETWEEN ? AND ?
            """, (player_id, start_date, end_date))
            
            existing_count = cursor.fetchone()[0]
            conn.close()
            
            return existing_count
            
        except Exception as e:
            logger.error(f"Error checking existing games for player {player_id}: {e}")
            return 0

    def _filter_new_games(self, player_id: int, season: str, new_games_df: pd.DataFrame) -> pd.DataFrame:
        """Filter out games that already exist in the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get existing game IDs for this player in this season
            season_ranges = self._get_season_date_ranges([season])
            if season not in season_ranges:
                conn.close()
                return new_games_df
            
            start_date, end_date = season_ranges[season]
            
            existing_games_query = """
                SELECT game_id FROM player_games 
                WHERE player_id = ? AND game_date BETWEEN ? AND ?
            """
            
            existing_games_df = pd.read_sql_query(
                existing_games_query, 
                conn, 
                params=(player_id, start_date, end_date)
            )
            
            conn.close()
            
            if existing_games_df.empty:
                return new_games_df
            
            # Filter out games that already exist
            existing_game_ids = set(existing_games_df['game_id'].astype(str))
            new_games_filtered = new_games_df[~new_games_df['game_id'].astype(str).isin(existing_game_ids)]
            
            return new_games_filtered
            
        except Exception as e:
            logger.error(f"Error filtering new games for player {player_id}: {e}")
            return new_games_df

    def _get_season_date_ranges(self, seasons: List[str]) -> Dict[str, Tuple[str, str]]:
        """Get date ranges for NBA seasons."""
        season_ranges = {
            "2024-25": ("2024-10-01", "2025-06-30"),
            "2023-24": ("2023-10-01", "2024-06-30"),
            "2022-23": ("2022-10-01", "2023-06-30"),
            "2021-22": ("2021-10-01", "2022-06-30"),
            "2020-21": ("2020-12-01", "2021-07-30"),  # COVID-19 delayed season
            "2019-20": ("2019-10-01", "2020-10-30"),  # COVID-19 extended season
            "2018-19": ("2018-10-01", "2019-06-30"),
            "2017-18": ("2017-10-01", "2018-06-30"),
            "2016-17": ("2016-10-01", "2017-06-30"),
        }
        
        return {season: season_ranges[season] for season in seasons if season in season_ranges}

    def check_data_freshness(self) -> Dict:
        """Check how fresh the data is and provide recommendations."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get the latest game date
            cursor.execute("SELECT MAX(game_date) FROM player_games")
            latest_date = cursor.fetchone()[0]
            
            if not latest_date:
                conn.close()
                return {
                    'status': 'no_data',
                    'message': 'No data found in database',
                    'recommendation': 'Run full data collection'
                }
            
            # Calculate days since latest data
            latest_dt = datetime.strptime(latest_date, "%Y-%m-%d")
            days_old = (datetime.now() - latest_dt).days
            
            # Get current NBA season info
            current_date = datetime.now()
            current_season = None
            if current_date.month >= 10 or current_date.month <= 6:
                current_season = f"{current_date.year}-{str(current_date.year + 1)[-2:]}"
            else:
                current_season = f"{current_date.year-1}-{str(current_date.year)[-2:]}"
            
            # Check if we have current season data
            cursor.execute("""
                SELECT COUNT(*) FROM player_games 
                WHERE game_date >= ?
            """, (f"{current_date.year}-10-01",))
            
            current_season_games = cursor.fetchone()[0]
            
            conn.close()
            
            # Determine status and recommendation
            if days_old <= 1:
                status = 'fresh'
                message = f'Data is very fresh (last game: {latest_date})'
                recommendation = 'No update needed'
            elif days_old <= 7:
                status = 'recent'
                message = f'Data is recent (last game: {latest_date}, {days_old} days ago)'
                recommendation = 'Consider quick update for latest games'
            elif days_old <= 30:
                status = 'stale'
                message = f'Data is getting stale (last game: {latest_date}, {days_old} days ago)'
                recommendation = 'Run full update to get recent games'
            else:
                status = 'outdated'
                message = f'Data is outdated (last game: {latest_date}, {days_old} days ago)'
                recommendation = 'Run full update to refresh all data'
            
            return {
                'status': status,
                'latest_date': latest_date,
                'days_old': days_old,
                'current_season': current_season,
                'current_season_games': current_season_games,
                'message': message,
                'recommendation': recommendation
            }
            
        except Exception as e:
            logger.error(f"Error checking data freshness: {e}")
            return {
                'status': 'error',
                'message': f'Error checking data freshness: {e}',
                'recommendation': 'Check database connection'
            }

    def get_team_leaders(self) -> List[int]:
        """Get players who are likely team leaders/stars based on common IDs."""
        # These are known star players with consistent data
        star_players = [
            2544,  # LeBron James - PRIORITY PLAYER
            203507,  # Giannis Antetokounmpo
            201939,  # Stephen Curry
            203999,  # Nikola Jokic
            1628369,  # Jayson Tatum
            1627759,  # Jamal Murray
            1629029,  # Luka Doncic
            203076,  # Anthony Davis
            1628983,  # Shai Gilgeous-Alexander
            1629630,  # Josh Giddey
            203954,  # Joel Embiid
            1628378,  # Tyler Herro
            1630173,  # Alperen Sengun
            1628389,  # Bam Adebayo
            201566,  # Russell Westbrook
            203081,  # Damian Lillard
            1627749,  # OG Anunoby
            1629027,  # Tyler Johnson
            1630162,  # Scottie Barnes
            1629628,  # Anfernee Simons
            203915,  # Bradley Beal
            1628973,  # De'Aaron Fox
            1629636,  # Darius Garland
            203897,  # Zion Williamson
            1629058,  # RJ Barrett
            203932,  # Pascal Siakam
            1628960,  # Donovan Mitchell
            1627783,  # Jaylen Brown
            203935,  # Marcus Smart
            1628370,  # Deandre Ayton
        ]
        return star_players

    def store_player_game_data(self, game_data: Dict) -> bool:
        """Store individual game data for a player."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Data validation - check for unrealistic values
            validated_data = self._validate_game_data(game_data)
            if not validated_data:
                logger.warning(f"Invalid game data rejected for player {game_data.get('player_id', 'unknown')}")
                return False

            # Check if this game already exists
            cursor.execute(
                "SELECT COUNT(*) FROM player_games WHERE game_id = ? AND player_id = ?",
                (game_data["game_id"], game_data["player_id"]),
            )

            if cursor.fetchone()[0] > 0:
                logger.debug(f"Game data already exists for player {game_data['player_id']}")
                conn.close()
                return True

            # Insert new game data
            columns = list(game_data.keys())
            placeholders = ", ".join(["?" for _ in columns])
            values = list(game_data.values())

            query = f"""
                INSERT INTO player_games ({', '.join(columns)})
                VALUES ({placeholders})
            """

            cursor.execute(query, values)
            conn.commit()
            conn.close()

            logger.debug(f"Stored game data for player {game_data['player_id']}")
            return True

        except Exception as e:
            logger.error(f"Error storing game data: {e}")
            return False

    def _validate_game_data(self, game_data: Dict) -> bool:
        """Validate game data for realistic values."""
        try:
            # Check for required fields
            required_fields = ['player_id', 'game_id', 'pts', 'reb', 'ast']
            for field in required_fields:
                if field not in game_data or game_data[field] is None:
                    logger.warning(f"Missing required field: {field}")
                    return False

            # Validate statistical ranges (NBA records and reasonable limits)
            validations = {
                'pts': (0, 100),      # Kobe's 81 is highest in modern era
                'reb': (0, 50),       # Wilt had 55, but 50 is reasonable modern limit
                'ast': (0, 30),       # Scott Skiles had 30, but that's the record
                'stl': (0, 15),       # 15 is extremely high but theoretically possible
                'blk': (0, 20),       # 20 is very high but possible for centers
                'tov': (0, 15),       # 15 turnovers is very high
                'pf': (0, 6),         # 6 fouls is maximum before fouling out
                'fgm': (0, 50),       # 50 field goals made is extremely high
                'fga': (0, 60),       # 60 attempts is very high
                'fg3m': (0, 20),      # Klay had 14, but 20 is reasonable upper limit
                'fg3a': (0, 30),      # 30 three-point attempts is very high
                'ftm': (0, 30),       # 30 free throws made is high
                'fta': (0, 35),       # 35 free throw attempts is high
                'min': (0, 60),       # 60 minutes is maximum (OT games)
            }

            for stat, (min_val, max_val) in validations.items():
                if stat in game_data:
                    value = game_data[stat]
                    if not isinstance(value, (int, float)):
                        try:
                            value = float(value)
                        except:
                            logger.warning(f"Non-numeric value for {stat}: {value}")
                            return False

                    if not (min_val <= value <= max_val):
                        logger.warning(f"Unrealistic {stat} value: {value} (should be {min_val}-{max_val})")
                        return False

            # Validate percentage fields
            percentage_fields = ['fg_pct', 'fg3_pct', 'ft_pct']
            for field in percentage_fields:
                if field in game_data and game_data[field] is not None:
                    value = game_data[field]
                    if not isinstance(value, (int, float)):
                        try:
                            value = float(value)
                        except:
                            continue  # Skip validation if can't convert
                    
                    if not (0.0 <= value <= 1.0):
                        logger.warning(f"Invalid percentage for {field}: {value}")
                        return False

            # Cross-validation (made shots can't exceed attempted shots)
            cross_validations = [
                ('fgm', 'fga'),
                ('fg3m', 'fg3a'),
                ('ftm', 'fta')
            ]

            for made_field, attempted_field in cross_validations:
                if made_field in game_data and attempted_field in game_data:
                    made = game_data[made_field] or 0
                    attempted = game_data[attempted_field] or 0
                    if made > attempted:
                        logger.warning(f"Made shots ({made}) exceed attempted ({attempted}) for {made_field}/{attempted_field}")
                        return False

            return True

        except Exception as e:
            logger.error(f"Error validating game data: {e}")
            return False
