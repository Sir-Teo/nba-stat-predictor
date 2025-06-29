"""
NBA Data Collector - Handles data collection from NBA APIs and other sources.
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Tuple
import sqlite3
from nba_api.stats.endpoints import (
    playergamelog, leaguegamefinder, teamgamelog, 
    commonplayerinfo, leaguestandings, scoreboardv2
)
from nba_api.stats.static import players, teams
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NBADataCollector:
    """Collects and processes NBA data from various sources."""
    
    def __init__(self, db_path: str = "data/nba_data.db"):
        """Initialize the data collector."""
        self.db_path = db_path
        self.setup_database()
        
    def setup_database(self):
        """Set up SQLite database for storing NBA data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables for different data types
        cursor.execute("""
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
        """)
        
        cursor.execute("""
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
        """)
        
        cursor.execute("""
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
        """)
        
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
    
    def get_player_game_logs(self, player_id: int, season: str = "2023-24", 
                            season_type: str = "Regular Season") -> pd.DataFrame:
        """Get game logs for a specific player."""
        try:
            time.sleep(0.6)  # Rate limiting
            player_gamelog_obj = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star=season_type
            )
            df = player_gamelog_obj.get_data_frames()[0]
            df['PLAYER_ID'] = player_id
            return df
        except Exception as e:
            logger.error(f"Error fetching game logs for player {player_id}: {e}")
            return pd.DataFrame()
    
    def get_todays_games(self) -> pd.DataFrame:
        """Get today's NBA games."""
        try:
            today = datetime.now().strftime('%m/%d/%Y')
            scoreboard = scoreboardv2.ScoreboardV2(game_date=today)
            games_df = scoreboard.get_data_frames()[0]  # GameHeader
            return games_df
        except Exception as e:
            logger.error(f"Error fetching today's games: {e}")
            return pd.DataFrame()
    
    def collect_historical_data(self, players_list: List[int], 
                               seasons: List[str] = ["2023-24", "2022-23", "2021-22", "2020-21"]) -> None:
        """Collect historical data for specified players and seasons."""
        conn = sqlite3.connect(self.db_path)
        
        total_games_collected = 0
        
        for season in seasons:
            logger.info(f"Collecting data for season {season}")
            season_games = 0
            
            for player_id in players_list:
                try:
                    df = self.get_player_game_logs(player_id, season)
                    if not df.empty:
                        # Process and clean the data
                        processed_df = self._process_game_log_data(df)
                        # Insert into database
                        processed_df.to_sql('player_games', conn, if_exists='append', index=False)
                        games_count = len(processed_df)
                        season_games += games_count
                        total_games_collected += games_count
                        logger.info(f"Collected {games_count} games for player {player_id}")
                except Exception as e:
                    logger.error(f"Error collecting data for player {player_id} in {season}: {e}")
                    continue
            
            logger.info(f"Season {season} complete: {season_games} games collected")
            
        logger.info(f"Total data collection complete: {total_games_collected} games")
        conn.close()
        
    def _process_game_log_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean game log data."""
        # Select and rename columns
        columns_mapping = {
            'GAME_ID': 'game_id',
            'PLAYER_ID': 'player_id',
            'GAME_DATE': 'game_date',
            'MATCHUP': 'matchup',
            'WL': 'wl',
            'MIN': 'min',
            'PTS': 'pts',
            'REB': 'reb',
            'AST': 'ast',
            'STL': 'stl',
            'BLK': 'blk',
            'TOV': 'tov',
            'PF': 'pf',
            'FGM': 'fgm',
            'FGA': 'fga',
            'FG_PCT': 'fg_pct',
            'FG3M': 'fg3m',
            'FG3A': 'fg3a',
            'FG3_PCT': 'fg3_pct',
            'FTM': 'ftm',
            'FTA': 'fta',
            'FT_PCT': 'ft_pct',
            'PLUS_MINUS': 'plus_minus'
        }
        
        # Select only columns that exist
        available_columns = {k: v for k, v in columns_mapping.items() if k in df.columns}
        processed_df = df[list(available_columns.keys())].rename(columns=available_columns)
        
        # Convert game_date to ISO format (YYYY-MM-DD)
        if 'game_date' in processed_df.columns:
            processed_df['game_date'] = pd.to_datetime(processed_df['game_date']).dt.strftime('%Y-%m-%d')
        
        # Add metadata
        processed_df['created_at'] = datetime.now().isoformat()
        
        # Handle missing values
        numeric_columns = ['min', 'pts', 'reb', 'ast', 'stl', 'blk', 'tov', 'pf',
                          'fgm', 'fga', 'fg_pct', 'fg3m', 'fg3a', 'fg3_pct',
                          'ftm', 'fta', 'ft_pct', 'plus_minus']
        
        for col in numeric_columns:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').fillna(0)
        
        return processed_df
    
    def get_recent_games(self, player_id: int, days: int = 10) -> pd.DataFrame:
        """Get recent games for a player from the database."""
        conn = sqlite3.connect(self.db_path)
        
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
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
        numeric_columns = ['pts', 'reb', 'ast', 'stl', 'blk', 'tov', 'fg_pct', 'fg3_pct', 'ft_pct']
        
        for col in numeric_columns:
            if col in df.columns:
                stats[f'{col}_avg'] = df[col].mean()
                stats[f'{col}_std'] = df[col].std()
                stats[f'{col}_trend'] = self._calculate_trend(df[col])
        
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
    
    def store_prediction(self, game_id: str, player_id: int, player_name: str,
                        game_date: str, stat_type: str, predicted_value: float,
                        confidence: float, model_version: str) -> None:
        """Store a prediction in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO predictions 
            (game_id, player_id, player_name, game_date, stat_type, 
             predicted_value, confidence, model_version, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (game_id, player_id, player_name, game_date, stat_type, 
              predicted_value, confidence, model_version, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def update_prediction_with_actual(self, game_id: str, player_id: int, 
                                    stat_type: str, actual_value: float) -> None:
        """Update prediction with actual value after the game."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE predictions 
            SET actual_value = ?
            WHERE game_id = ? AND player_id = ? AND stat_type = ?
        """, (actual_value, game_id, player_id, stat_type))
        
        conn.commit()
        conn.close()
    
    def get_popular_players(self, limit: int = 100) -> List[int]:
        """Get a list of popular NBA players (active and recently active)."""
        try:
            # Get all players and filter for recent ones
            all_players = players.get_players()
            
            # Focus on players from recent years (2018 onwards for active/recently active)
            recent_players = [
                player for player in all_players 
                if player.get('is_active', True) or 
                (player.get('to_year') and int(player.get('to_year', 0)) >= 2020)
            ]
            
            # Sort by ID (newer players have higher IDs generally)
            recent_players.sort(key=lambda x: x['id'], reverse=True)
            
            # Return top players by ID
            return [player['id'] for player in recent_players[:limit]]
            
        except Exception as e:
            logger.error(f"Error getting popular players: {e}")
            # Fallback list of known star player IDs
            return [
                2544, 201939, 203507, 203999, 1628369, 1627759, 1629029,
                203076, 1628983, 1629630, 203954, 1628378, 1630173, 1628389,
                201566, 203081, 1627749, 1629027, 1630162, 1629628
            ]
    
    def get_team_leaders(self) -> List[int]:
        """Get players who are likely team leaders/stars based on common IDs."""
        # These are known star players with consistent data
        star_players = [
            2544,    # LeBron James - PRIORITY PLAYER
            203507,  # Giannis Antetokounmpo
            201939,  # Stephen Curry
            203999,  # Nikola Jokic
            1628369, # Jayson Tatum
            1627759, # Jamal Murray
            1629029, # Luka Doncic
            203076,  # Anthony Davis
            1628983, # Shai Gilgeous-Alexander
            1629630, # Josh Giddey
            203954,  # Joel Embiid
            1628378, # Tyler Herro
            1630173, # Alperen Sengun
            1628389, # Bam Adebayo
            201566,  # Russell Westbrook
            203081,  # Damian Lillard
            1627749, # OG Anunoby
            1629027, # Tyler Johnson
            1630162, # Scottie Barnes
            1629628, # Anfernee Simons
            203915,  # Bradley Beal
            1628973, # De'Aaron Fox
            1629636, # Darius Garland
            203897,  # Zion Williamson
            1629058, # RJ Barrett
            203932,  # Pascal Siakam
            1628960, # Donovan Mitchell
            1627783, # Jaylen Brown
            203935,  # Marcus Smart
            1628370, # Deandre Ayton
        ]
        return star_players 