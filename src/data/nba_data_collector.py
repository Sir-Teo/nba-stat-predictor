"""
NBA Data Collector - Handles data collection from NBA APIs and other sources.
"""

import logging
import sqlite3
import time
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
    ) -> pd.DataFrame:
        """Get game logs for a specific player."""
        try:
            time.sleep(0.6)  # Rate limiting
            player_gamelog_obj = playergamelog.PlayerGameLog(
                player_id=player_id, season=season, season_type_all_star=season_type
            )
            df = player_gamelog_obj.get_data_frames()[0]
            df["PLAYER_ID"] = player_id
            return df
        except Exception as e:
            logger.error(f"Error fetching game logs for player {player_id}: {e}")
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
        seasons: List[str] = ["2023-24", "2022-23", "2021-22", "2020-21"],
    ) -> None:
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
                        processed_df.to_sql(
                            "player_games", conn, if_exists="append", index=False
                        )
                        games_count = len(processed_df)
                        season_games += games_count
                        total_games_collected += games_count
                        logger.info(
                            f"Collected {games_count} games for player {player_id}"
                        )
                except Exception as e:
                    logger.error(
                        f"Error collecting data for player {player_id} in {season}: {e}"
                    )
                    continue

            logger.info(f"Season {season} complete: {season_games} games collected")

        logger.info(f"Total data collection complete: {total_games_collected} games")
        conn.close()

    def _process_game_log_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean game log data."""
        # Select and rename columns
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

        # Select only columns that exist
        available_columns = {
            k: v for k, v in columns_mapping.items() if k in df.columns
        }
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
        """Get a list of popular NBA players (active and recently active)."""
        try:
            # Get all players and filter for recent ones
            all_players = players.get_players()

            # Focus on players from recent years (2018 onwards for active/recently active)
            recent_players = [
                player
                for player in all_players
                if player.get("is_active", True)
                or (player.get("to_year") and int(player.get("to_year", 0)) >= 2020)
            ]

            # Sort by ID (newer players have higher IDs generally)
            recent_players.sort(key=lambda x: x["id"], reverse=True)

            # Return top players by ID
            return [player["id"] for player in recent_players[:limit]]

        except Exception as e:
            logger.error(f"Error getting popular players: {e}")
            # Fallback list of known star player IDs
            return [
                2544,
                201939,
                203507,
                203999,
                1628369,
                1627759,
                1629029,
                203076,
                1628983,
                1629630,
                203954,
                1628378,
                1630173,
                1628389,
                201566,
                203081,
                1627749,
                1629027,
                1630162,
                1629628,
            ]

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
