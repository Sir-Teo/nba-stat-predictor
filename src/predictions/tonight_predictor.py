"""
Tonight's Games Predictor - Makes predictions for NBA games happening today.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import commonteamroster, scoreboardv2
from nba_api.stats.static import players

from ..data.feature_engineer import FeatureEngineer
from ..data.nba_data_collector import NBADataCollector
from ..models.stat_predictors import ModelManager

logger = logging.getLogger(__name__)


class TonightPredictor:
    """Predicts stats for NBA players in tonight's games."""

    def __init__(self, db_path: str = "data/nba_data.db"):
        """Initialize the tonight predictor."""
        self.db_path = db_path
        self.data_collector = NBADataCollector(db_path)
        self.feature_engineer = FeatureEngineer(db_path)
        self.model_manager = ModelManager(db_path)
        self.stat_types = ["pts", "reb", "ast", "stl", "blk"]

    def get_tonights_predictions(self) -> pd.DataFrame:
        """Get predictions for all players in tonight's games."""
        logger.info("Getting predictions for tonight's games...")

        # Get tonight's games
        todays_games = self.data_collector.get_todays_games()

        if todays_games.empty:
            logger.info("No games scheduled for today")
            return pd.DataFrame()

        logger.info(f"Found {len(todays_games)} games today")

        all_predictions = []

        for _, game in todays_games.iterrows():
            game_id = game.get("GAME_ID", "")
            home_team_id = game.get("HOME_TEAM_ID", 0)
            away_team_id = game.get("VISITOR_TEAM_ID", 0)
            game_date = datetime.now().strftime("%Y-%m-%d")

            logger.info(f"Processing game: {game_id}")

            # Get players for both teams
            home_players = self._get_team_players(home_team_id)
            away_players = self._get_team_players(away_team_id)

            all_players = home_players + away_players

            for player_id, player_name in all_players:
                try:
                    # Load player-specific models if available, fall back to general models
                    self.model_manager.load_models(self.stat_types, player_id=player_id)
                    
                    # Create features for this player
                    features_df = self.feature_engineer.create_features_for_player(
                        player_id, game_date
                    )

                    if features_df.empty:
                        logger.warning(
                            f"No features available for player {player_name}"
                        )
                        continue

                    # Make predictions
                    predictions_df = self.model_manager.predict_stats(
                        features_df, self.stat_types
                    )

                    if not predictions_df.empty:
                        # Add game and player info
                        predictions_df["game_id"] = game_id
                        predictions_df["player_name"] = player_name
                        predictions_df["game_date"] = game_date

                        all_predictions.append(predictions_df)

                        # Store predictions in database
                        self._store_predictions(
                            game_id, player_id, player_name, game_date, predictions_df
                        )

                except Exception as e:
                    logger.error(f"Error predicting for player {player_name}: {e}")
                    continue

        if all_predictions:
            final_predictions = pd.concat(all_predictions, ignore_index=True)
            logger.info(
                f"Generated predictions for {len(final_predictions)} player-games"
            )
            return final_predictions
        else:
            logger.warning("No predictions generated")
            return pd.DataFrame()

    def _get_team_players(self, team_id: int) -> List[Tuple[int, str]]:
        """Get active players for a team."""
        try:
            # Try to get roster from NBA API
            roster = commonteamroster.CommonTeamRoster(team_id=team_id)
            roster_df = roster.get_data_frames()[0]

            players_list = []
            for _, player in roster_df.iterrows():
                player_id = player.get("PLAYER_ID", 0)
                player_name = player.get("PLAYER", "Unknown")
                players_list.append((player_id, player_name))

            return players_list

        except Exception as e:
            logger.error(f"Error getting roster for team {team_id}: {e}")

            # Fallback: get players from our database who played recently for this team
            return self._get_team_players_from_db(team_id)

    def _get_team_players_from_db(self, team_id: int) -> List[Tuple[int, str]]:
        """Get team players from our database as fallback."""
        import sqlite3

        conn = sqlite3.connect(self.db_path)

        # Get players who played for this team in the last 30 days
        cutoff_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        query = """
            SELECT DISTINCT player_id, player_name
            FROM player_games 
            WHERE team_id = ? AND game_date >= ?
            ORDER BY game_date DESC
            LIMIT 15
        """

        cursor = conn.cursor()
        cursor.execute(query, (team_id, cutoff_date))
        players = cursor.fetchall()
        conn.close()

        return players

    def _store_predictions(
        self,
        game_id: str,
        player_id: int,
        player_name: str,
        game_date: str,
        predictions_df: pd.DataFrame,
    ) -> None:
        """Store predictions in the database."""
        for stat_type in self.stat_types:
            predicted_col = f"predicted_{stat_type}"
            confidence_col = f"confidence_{stat_type}"

            if predicted_col in predictions_df.columns:
                predicted_value = predictions_df[predicted_col].iloc[0]
                confidence = (
                    predictions_df[confidence_col].iloc[0]
                    if confidence_col in predictions_df.columns
                    else 0.5
                )

                # Get model version (simplified for now)
                model_version = "v1.0"

                self.data_collector.store_prediction(
                    game_id,
                    player_id,
                    player_name,
                    game_date,
                    stat_type,
                    predicted_value,
                    confidence,
                    model_version,
                )

    def update_predictions_with_results(self, game_date: str = None) -> None:
        """Update stored predictions with actual game results."""
        if game_date is None:
            # Update yesterday's games by default
            game_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        logger.info(f"Updating predictions with actual results for {game_date}")

        import sqlite3

        conn = sqlite3.connect(self.db_path)

        # Get predictions that need updating
        query = """
            SELECT DISTINCT game_id, player_id, stat_type 
            FROM predictions 
            WHERE game_date = ? AND actual_value IS NULL
        """

        cursor = conn.cursor()
        cursor.execute(query, (game_date,))
        predictions_to_update = cursor.fetchall()

        for game_id, player_id, stat_type in predictions_to_update:
            # Look for actual game result
            actual_query = """
                SELECT * FROM player_games
                WHERE game_id = ? AND player_id = ?
            """

            actual_df = pd.read_sql_query(
                actual_query, conn, params=(game_id, player_id)
            )

            if not actual_df.empty and stat_type in actual_df.columns:
                actual_value = actual_df[stat_type].iloc[0]

                # Update prediction with actual value
                self.data_collector.update_prediction_with_actual(
                    game_id, player_id, stat_type, actual_value
                )

        conn.close()
        logger.info(
            f"Updated {len(predictions_to_update)} predictions with actual results"
        )

    def get_prediction_summary(self, player_name: str = None) -> pd.DataFrame:
        """Get a summary of tonight's predictions."""
        import sqlite3

        conn = sqlite3.connect(self.db_path)

        today = datetime.now().strftime("%Y-%m-%d")

        if player_name:
            query = """
                SELECT player_name, stat_type, predicted_value, confidence, created_at
                FROM predictions 
                WHERE game_date = ? AND player_name LIKE ?
                ORDER BY player_name, stat_type
            """
            df = pd.read_sql_query(query, conn, params=(today, f"%{player_name}%"))
        else:
            query = """
                SELECT player_name, stat_type, predicted_value, confidence, created_at
                FROM predictions 
                WHERE game_date = ?
                ORDER BY player_name, stat_type
            """
            df = pd.read_sql_query(query, conn, params=(today,))

        conn.close()
        return df

    def get_top_predictions(self, stat_type: str, top_n: int = 10) -> pd.DataFrame:
        """Get top N predictions for a specific stat type for tonight."""
        import sqlite3

        conn = sqlite3.connect(self.db_path)

        today = datetime.now().strftime("%Y-%m-%d")

        query = """
            SELECT player_name, predicted_value, confidence
            FROM predictions 
            WHERE game_date = ? AND stat_type = ?
            ORDER BY predicted_value DESC
            LIMIT ?
        """

        df = pd.read_sql_query(query, conn, params=(today, stat_type, top_n))
        conn.close()

        return df

    def analyze_recent_accuracy(self, days: int = 7) -> Dict[str, float]:
        """Analyze prediction accuracy over recent days."""
        import sqlite3

        conn = sqlite3.connect(self.db_path)

        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        accuracy_results = {}

        for stat_type in self.stat_types:
            query = """
                SELECT predicted_value, actual_value
                FROM predictions 
                WHERE game_date >= ? AND stat_type = ? AND actual_value IS NOT NULL
            """

            df = pd.read_sql_query(query, conn, params=(cutoff_date, stat_type))

            if len(df) > 5:
                mae = np.mean(np.abs(df["predicted_value"] - df["actual_value"]))
                rmse = np.sqrt(
                    np.mean((df["predicted_value"] - df["actual_value"]) ** 2)
                )
                accuracy_results[stat_type] = {
                    "mae": mae,
                    "rmse": rmse,
                    "sample_size": len(df),
                }
            else:
                accuracy_results[stat_type] = {
                    "mae": 0,
                    "rmse": 0,
                    "sample_size": len(df),
                }

        conn.close()
        return accuracy_results
