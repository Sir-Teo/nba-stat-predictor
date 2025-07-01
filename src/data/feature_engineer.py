"""
Feature Engineering Module - Creates advanced features for NBA stat prediction models.
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Known birthdates for major players (can be expanded)
PLAYER_BIRTHDATES = {
    2544: "1984-12-30",  # LeBron James
    203507: "1994-12-06",  # Giannis Antetokounmpo
    201939: "1988-03-14",  # Stephen Curry
    203999: "1995-02-19",  # Nikola Jokic
    1628369: "1998-03-03",  # Jayson Tatum
    1629029: "1999-02-28",  # Luka Doncic
    203076: "1993-03-11",  # Anthony Davis
    203954: "1994-03-16",  # Joel Embiid
    201566: "1988-11-12",  # Russell Westbrook
    203081: "1990-07-15",  # Damian Lillard
    201142: "1988-09-29",  # Kevin Durant
}


class AdvancedFeatureEngineer:
    """Creates and processes advanced features for NBA stat prediction."""

    def __init__(self, db_path: str = "data/nba_data.db"):
        """Initialize the advanced feature engineer."""
        self.db_path = db_path
        self.scaler = RobustScaler()
        self.label_encoders = {}
        self.team_clusters = {}
        self.pace_adjustments = {}

    def _calculate_player_age(self, player_id: int, target_date: str) -> float:
        """Calculate player age in years as of target date."""
        # Check if we have the birthdate
        if player_id not in PLAYER_BIRTHDATES:
            # Estimate age based on first NBA game (rough approximation)
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT MIN(game_date) FROM player_games WHERE player_id = ?",
                    (player_id,)
                )
                result = cursor.fetchone()
                conn.close()
                
                if result and result[0]:
                    # Assume player was 20 when they first played (average rookie age)
                    first_game_date = datetime.strptime(result[0], "%Y-%m-%d")
                    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
                    years_since_rookie = (target_dt - first_game_date).days / 365.25
                    estimated_age = 20 + years_since_rookie
                    return min(estimated_age, 45)  # Cap at reasonable maximum
                else:
                    return 27  # Average NBA player age
            except:
                return 27  # Default fallback
        
        # Calculate exact age from birthdate
        birthdate = datetime.strptime(PLAYER_BIRTHDATES[player_id], "%Y-%m-%d")
        target_dt = datetime.strptime(target_date, "%Y-%m-%d")
        age = (target_dt - birthdate).days / 365.25
        return age

    def _create_age_features(self, player_id: int, target_date: str, df: pd.DataFrame) -> Dict:
        """Create age-related features for prediction."""
        features = {}
        
        age = self._calculate_player_age(player_id, target_date)
        features["player_age"] = age
        
        # Age categories
        features["is_rookie"] = 1 if age <= 22 else 0
        features["is_young"] = 1 if 22 < age <= 26 else 0
        features["is_prime"] = 1 if 26 < age <= 30 else 0
        features["is_veteran"] = 1 if 30 < age <= 34 else 0
        features["is_aging_vet"] = 1 if age > 34 else 0
        
        # Age-related decline factors
        if age > 30:
            # Apply decline curve for aging players
            decline_factor = max(0.7, 1 - ((age - 30) * 0.04))  # 4% decline per year after 30
            features["age_decline_factor"] = decline_factor
            
            # For very old players (38+), apply more aggressive decline
            if age >= 38:
                steep_decline_factor = max(0.5, 1 - ((age - 38) * 0.08))  # 8% decline per year after 38
                features["age_decline_factor"] = min(features["age_decline_factor"], steep_decline_factor)
        else:
            features["age_decline_factor"] = 1.0
        
        # Age-adjusted recent form weight
        # Older players should be judged more on recent form than career averages
        if age > 34:
            features["recent_form_weight"] = 0.8  # 80% weight on recent games
        elif age > 30:
            features["recent_form_weight"] = 0.6  # 60% weight on recent games  
        else:
            features["recent_form_weight"] = 0.4  # 40% weight on recent games
            
        # Career longevity features
        features["years_experience"] = max(0, age - 19)  # Assume entered NBA at 19
        features["is_superhuman_longevity"] = 1 if age > 37 and len(df) > 10 else 0  # Still playing at high level past 37
        
        return features

    def create_features_for_player(
        self,
        player_id: int,
        target_date: str,
        lookback_games: int = 20,
        opponent_team_id: int = None,
        include_h2h_features: bool = True,
        include_advanced_features: bool = True,
    ) -> pd.DataFrame:
        """Create comprehensive advanced features for a player for prediction."""
        conn = sqlite3.connect(self.db_path)

        # Get player's recent games with team info
        query = """
            SELECT pg.*, COALESCE(t.conference, 'Unknown') as conference, 
                   COALESCE(t.division, 'Unknown') as division
            FROM player_games pg
            LEFT JOIN teams t ON pg.team_id = t.team_id
            WHERE pg.player_id = ? AND pg.game_date < ?
            ORDER BY pg.game_date DESC
            LIMIT ?
        """

        df = pd.read_sql_query(
            query, conn, params=(player_id, target_date, lookback_games)
        )

        if df.empty:
            conn.close()
            return pd.DataFrame()

        features = {}

        # *** NEW: Add age-based features first ***
        features.update(self._create_age_features(player_id, target_date, df))

        # Core rolling statistics with multiple windows (reduced for speed)
        features.update(
            self._create_rolling_stats(df, windows=[3, 5, 10])
        )  # Reduced from [3, 5, 10, 15, 20]

        # Advanced form and momentum features
        features.update(self._create_advanced_form_features(df))

        # Situational and contextual features
        features.update(self._create_situational_features(df))

        # Home/Away performance with advanced metrics
        features.update(self._create_home_away_features(df))

        # Rest and fatigue analysis
        features.update(self._create_rest_features(df))
        features.update(self._create_fatigue_features(df))

        # Consistency and reliability metrics
        features.update(self._create_consistency_features(df))

        # Player role and usage features
        features.update(self._create_usage_features(df))

        # Performance vs different opponent types
        features.update(self._create_opponent_type_features(df, conn))

        # Advanced matchup features
        if include_advanced_features:
            features.update(self._create_pace_adjusted_features(df, conn))
            features.update(self._create_clutch_performance_features(df))
            features.update(self._create_momentum_shift_features(df))
            features.update(self._create_game_context_features(df))

        # Team strength and opponent-specific features
        if opponent_team_id is not None and include_h2h_features:
            features.update(
                self._create_team_strength_features(opponent_team_id, target_date, conn)
            )
            features.update(
                self._create_opponent_specific_features(
                    player_id, opponent_team_id, target_date, conn
                )
            )
            features.update(
                self._create_head_to_head_features(
                    player_id, opponent_team_id, target_date, conn
                )
            )

        conn.close()

        # Convert to DataFrame
        features_df = pd.DataFrame([features])
        features_df["player_id"] = player_id
        features_df["target_date"] = target_date
        if opponent_team_id is not None:
            features_df["opponent_team_id"] = opponent_team_id

        return features_df

    def _create_rolling_stats(
        self, df: pd.DataFrame, windows: List[int] = [3, 5, 10]
    ) -> Dict:
        """Create comprehensive rolling statistics for different time windows."""
        features = {}

        stat_columns = [
            "pts",
            "reb",
            "ast",
            "stl",
            "blk",
            "tov",
            "fg_pct",
            "fg3_pct",
            "ft_pct",
            "min",
        ]

        # Ensure data is sorted by date
        df_sorted = df.sort_values("game_date").reset_index(drop=True)

        for window in windows:
            for stat in stat_columns:
                if stat in df_sorted.columns:
                    # Basic rolling statistics
                    rolling_values = df_sorted[stat].rolling(
                        window=window, min_periods=1
                    )

                    features[f"{stat}_avg_{window}g"] = rolling_values.mean().iloc[-1]
                    features[f"{stat}_std_{window}g"] = (
                        rolling_values.std().iloc[-1] or 0
                    )
                    features[f"{stat}_min_{window}g"] = rolling_values.min().iloc[-1]
                    features[f"{stat}_max_{window}g"] = rolling_values.max().iloc[-1]

                    # Advanced rolling statistics
                    features[f"{stat}_median_{window}g"] = rolling_values.median().iloc[
                        -1
                    ]
                    features[f"{stat}_q25_{window}g"] = rolling_values.quantile(
                        0.25
                    ).iloc[-1]
                    features[f"{stat}_q75_{window}g"] = rolling_values.quantile(
                        0.75
                    ).iloc[-1]
                    features[f"{stat}_iqr_{window}g"] = (
                        features[f"{stat}_q75_{window}g"]
                        - features[f"{stat}_q25_{window}g"]
                    )

                    # Coefficient of variation (relative volatility)
                    mean_val = features[f"{stat}_avg_{window}g"]
                    std_val = features[f"{stat}_std_{window}g"]
                    features[f"{stat}_cv_{window}g"] = (
                        (std_val / mean_val) if mean_val > 0 else 0
                    )

                    # Rolling trend and momentum
                    if len(df_sorted) >= window:
                        recent_values = df_sorted[stat].tail(window).values
                        features[f"{stat}_trend_{window}g"] = self._calculate_trend(
                            recent_values
                        )
                        features[f"{stat}_momentum_{window}g"] = (
                            self._calculate_momentum(recent_values)
                        )

                    # Performance vs season average
                    season_avg = df_sorted[stat].mean()
                    features[f"{stat}_vs_season_{window}g"] = (
                        features[f"{stat}_avg_{window}g"] - season_avg
                    )

                    # Recent vs historical performance
                    if window <= 10 and len(df_sorted) > window:
                        recent_avg = df_sorted[stat].tail(window).mean()
                        historical_avg = df_sorted[stat].iloc[:-window].mean()
                        features[f"{stat}_recent_vs_hist_{window}g"] = (
                            recent_avg - historical_avg
                        )

        return features

    def _create_advanced_form_features(self, df: pd.DataFrame) -> Dict:
        """Create advanced performance trends and form features."""
        features = {}

        if len(df) < 3:
            return features

        df_sorted = df.sort_values("game_date").reset_index(drop=True)

        # Multi-game performance patterns
        for stat in ["pts", "reb", "ast"]:
            if stat in df_sorted.columns:
                values = df_sorted[stat].values
                season_avg = values.mean()

                # Performance streaks
                above_avg_streak = self._calculate_current_streak(values, season_avg)
                features[f"{stat}_above_avg_streak"] = above_avg_streak

                # Volatility measures
                features[f"{stat}_volatility_score"] = np.std(values) / (
                    np.mean(values) + 1e-8
                )

                # Form curve (recent 5 vs previous 5)
                if len(values) >= 10:
                    recent_5 = values[-5:].mean()
                    prev_5 = values[-10:-5].mean()
                    features[f"{stat}_form_curve"] = recent_5 - prev_5

                # Peak performance indicators
                features[f"{stat}_games_above_peak"] = np.sum(
                    values >= np.percentile(values, 90)
                )
                features[f"{stat}_consistency_score"] = 1 - (
                    np.std(values) / (np.mean(values) + 1e-8)
                )

                # Hot/cold streaks
                hot_games = np.sum(values >= season_avg * 1.2)  # 20% above average
                cold_games = np.sum(values <= season_avg * 0.8)  # 20% below average
                features[f"{stat}_hot_games_pct"] = hot_games / len(values)
                features[f"{stat}_cold_games_pct"] = cold_games / len(values)

        return features

    def _create_situational_features(self, df: pd.DataFrame) -> Dict:
        """Create features based on game situations and contexts."""
        features = {}

        if len(df) < 2:
            return features

        df_sorted = df.sort_values("game_date").reset_index(drop=True)

        # Day of week effects
        df_sorted["game_date_dt"] = pd.to_datetime(df_sorted["game_date"])
        df_sorted["day_of_week"] = df_sorted["game_date_dt"].dt.dayofweek

        # Performance by day of week
        for day in range(7):
            day_games = df_sorted[df_sorted["day_of_week"] == day]
            if len(day_games) > 0:
                features[f"pts_avg_day_{day}"] = day_games["pts"].mean()
                features[f"games_on_day_{day}"] = len(day_games)

        # Back-to-back performance
        df_sorted["days_rest"] = df_sorted["game_date_dt"].diff().dt.days
        b2b_games = df_sorted[df_sorted["days_rest"] <= 1]
        if len(b2b_games) > 0:
            features["pts_avg_b2b"] = b2b_games["pts"].mean()
            features["b2b_games_count"] = len(b2b_games)
            features["b2b_performance_drop"] = (
                df_sorted["pts"].mean() - b2b_games["pts"].mean()
            )

        # Win/Loss context
        if "wl" in df_sorted.columns:
            wins = df_sorted[df_sorted["wl"] == "W"]
            losses = df_sorted[df_sorted["wl"] == "L"]

            if len(wins) > 0:
                features["pts_in_wins"] = wins["pts"].mean()
                features["win_rate"] = len(wins) / len(df_sorted)

            if len(losses) > 0:
                features["pts_in_losses"] = losses["pts"].mean()

        # Monthly performance trends
        df_sorted["month"] = df_sorted["game_date_dt"].dt.month
        current_month = df_sorted["month"].iloc[-1]
        current_month_games = df_sorted[df_sorted["month"] == current_month]
        if len(current_month_games) > 1:
            features["pts_current_month"] = current_month_games["pts"].mean()
            features["games_current_month"] = len(current_month_games)

        return features

    def _create_usage_features(self, df: pd.DataFrame) -> Dict:
        """Create features related to player usage and role."""
        features = {}

        if len(df) < 3:
            return features

        # Usage indicators
        avg_min = df["min"].mean()
        features["avg_minutes"] = avg_min
        features["high_usage_games"] = np.sum(df["min"] >= avg_min * 1.2)
        features["low_usage_games"] = np.sum(df["min"] <= avg_min * 0.8)

        # Efficiency metrics
        if "fga" in df.columns and "pts" in df.columns:
            total_fga = df["fga"].sum()
            total_pts = df["pts"].sum()
            if total_fga > 0:
                features["points_per_fga"] = total_pts / total_fga

        # Role consistency
        features["minutes_consistency"] = 1 - (
            df["min"].std() / (df["min"].mean() + 1e-8)
        )

        # Production per minute
        if avg_min > 0:
            for stat in ["pts", "reb", "ast"]:
                if stat in df.columns:
                    features[f"{stat}_per_minute"] = df[stat].mean() / avg_min

        return features

    def _create_opponent_type_features(
        self, df: pd.DataFrame, conn: sqlite3.Connection
    ) -> Dict:
        """Create features based on opponent strength and type."""
        features = {}

        # This would require opponent team data - simplified version
        if "matchup" in df.columns:
            # Extract opponent from matchup string
            home_games = df["matchup"].str.contains("vs.").sum()
            away_games = len(df) - home_games

            features["home_game_ratio"] = home_games / len(df)
            features["away_game_ratio"] = away_games / len(df)

            # Performance splits
            if home_games > 0:
                home_df = df[df["matchup"].str.contains("vs.")]
                features["pts_vs_home"] = home_df["pts"].mean()

            if away_games > 0:
                away_df = df[~df["matchup"].str.contains("vs.")]
                features["pts_vs_away"] = away_df["pts"].mean()

        return features

    def _create_pace_adjusted_features(
        self, df: pd.DataFrame, conn: sqlite3.Connection
    ) -> Dict:
        """Create pace-adjusted performance features."""
        features = {}

        # Simplified pace adjustment - would need team pace data for full implementation
        avg_min = df["min"].mean()
        if avg_min > 0:
            # Estimate possessions based on minutes played
            est_possessions = avg_min * 1.2  # Rough estimate

            for stat in ["pts", "reb", "ast"]:
                if stat in df.columns:
                    features[f"{stat}_per_100_poss"] = (
                        df[stat].mean() / est_possessions
                    ) * 100

        return features

    def _create_clutch_performance_features(self, df: pd.DataFrame) -> Dict:
        """Create features for clutch/high-pressure performance."""
        features = {}

        # This would require play-by-play data for true clutch stats
        # Using close games as proxy for clutch situations
        if "plus_minus" in df.columns:
            close_games = df[abs(df["plus_minus"]) <= 5]  # Games decided by 5 or less
            if len(close_games) > 0:
                features["clutch_games_count"] = len(close_games)
                features["clutch_pts_avg"] = close_games["pts"].mean()
                features["clutch_performance"] = (
                    close_games["pts"].mean() - df["pts"].mean()
                )

        return features

    def _create_momentum_shift_features(self, df: pd.DataFrame) -> Dict:
        """Create features capturing momentum and performance shifts."""
        features = {}

        if len(df) < 5:
            return features

        # Performance acceleration/deceleration
        for stat in ["pts", "reb", "ast"]:
            if stat in df.columns:
                values = df.sort_values("game_date")[stat].values

                # Second derivative (acceleration)
                if len(values) >= 3:
                    first_diff = np.diff(values)
                    second_diff = np.diff(first_diff)
                    features[f"{stat}_acceleration"] = np.mean(
                        second_diff[-3:]
                    )  # Recent acceleration

                # Momentum score (weighted recent performance)
                weights = np.exp(np.linspace(0, 1, len(values)))
                features[f"{stat}_momentum_score"] = np.average(values, weights=weights)

        return features

    def _create_game_context_features(self, df: pd.DataFrame) -> Dict:
        """Create features based on broader game context."""
        features = {}

        # Performance in different game outcomes
        if "wl" in df.columns and "plus_minus" in df.columns:
            # Blowout vs close game performance
            blowouts = df[abs(df["plus_minus"]) >= 20]
            close_games = df[abs(df["plus_minus"]) <= 5]

            if len(blowouts) > 0:
                features["blowout_pts_avg"] = blowouts["pts"].mean()
                features["blowout_games_count"] = len(blowouts)

            if len(close_games) > 0:
                features["close_game_pts_avg"] = close_games["pts"].mean()
                features["close_games_count"] = len(close_games)

        return features

    def _create_team_strength_features(
        self, opponent_team_id: int, target_date: str, conn: sqlite3.Connection
    ) -> Dict:
        """Create enhanced team strength and defensive ranking features."""
        features = {}

        try:
            # Get opponent team's overall defensive metrics (last 15 games)
            # Disable this query since opponent_team_id doesn't exist in schema
            # Instead use default values based on league averages
            defense_query = """
                SELECT 
                    AVG(pts) as avg_pts_allowed,
                    AVG(reb) as avg_reb_allowed,
                    AVG(ast) as avg_ast_allowed,
                    AVG(stl) as avg_stl_generated,
                    AVG(blk) as avg_blk_generated,
                    AVG(CAST(fg_pct AS FLOAT)) as avg_fg_pct_allowed,
                    AVG(CAST(fg3_pct AS FLOAT)) as avg_3pt_pct_allowed,
                    COUNT(*) as games_sample
            FROM player_games
                WHERE team_id = ? AND game_date < ? AND game_date >= DATE(?, '-15 days')
                AND fg_pct IS NOT NULL AND fg_pct != ''
            """

            def_df = pd.read_sql_query(
                defense_query, conn, params=(opponent_team_id, target_date, target_date)
            )

            if not def_df.empty and def_df["games_sample"].iloc[0] >= 5:
                # Safe extraction of defensive metrics with None checks
                avg_pts_allowed = def_df["avg_pts_allowed"].iloc[0] or 112
                avg_reb_allowed = def_df["avg_reb_allowed"].iloc[0] or 45
                avg_stl_generated = def_df["avg_stl_generated"].iloc[0] or 8
                avg_blk_generated = def_df["avg_blk_generated"].iloc[0] or 5
                
                # Defensive efficiency metrics with safe division
                features["opp_def_efficiency"] = avg_pts_allowed / 100  # Normalized
                features["opp_reb_def_rate"] = avg_reb_allowed / 45  # Normalized
                features["opp_def_fg_pct"] = def_df["avg_fg_pct_allowed"].iloc[0] or 0.45
                features["opp_def_3pt_pct"] = def_df["avg_3pt_pct_allowed"].iloc[0] or 0.35
                
                # Defensive pressure metrics with safe division
                features["opp_steal_rate"] = avg_stl_generated / 10  # Normalized
                features["opp_block_rate"] = avg_blk_generated / 6   # Normalized
                
                # Calculate relative defensive strength with safe division
                league_avg_pts = 112
                league_avg_fg = 0.46
                
                fg_pct_allowed = def_df["avg_fg_pct_allowed"].iloc[0] or 0.46
                
                features["opp_pts_def_relative"] = avg_pts_allowed / league_avg_pts
                features["opp_fg_def_relative"] = fg_pct_allowed / league_avg_fg
                
            else:
                # Use league averages as defaults
                features["opp_def_efficiency"] = 1.12  # Average
                features["opp_reb_def_rate"] = 0.96
                features["opp_def_fg_pct"] = 0.46
                features["opp_def_3pt_pct"] = 0.35
                features["opp_steal_rate"] = 0.8
                features["opp_block_rate"] = 0.83
                features["opp_pts_def_relative"] = 1.0
                features["opp_fg_def_relative"] = 1.0

            # Team pace and style features
            pace_query = """
                SELECT 
                    COUNT(*) / COUNT(DISTINCT game_date) as avg_possessions_approx,
                    AVG(pts) as avg_team_pts,
                    AVG(ast) as avg_team_ast,
                    COUNT(DISTINCT game_date) as games_played
                FROM player_games 
                WHERE team_id = ? AND game_date < ? AND game_date >= DATE(?, '-10 days')
        """

            pace_df = pd.read_sql_query(
                pace_query, conn, params=(opponent_team_id, target_date, target_date)
            )

            if not pace_df.empty and pace_df["games_played"].iloc[0] >= 3:
                # Safe extraction with None checks
                possessions = pace_df["avg_possessions_approx"].iloc[0] or 15
                avg_team_pts = pace_df["avg_team_pts"].iloc[0] or 110
                avg_team_ast = pace_df["avg_team_ast"].iloc[0] or 25
                
                features["opp_team_pace"] = min(max(possessions / 15, 0.8), 1.2)  # Normalized pace
                features["opp_team_offensive_rating"] = avg_team_pts / 110  # Normalized
                features["opp_team_ball_movement"] = avg_team_ast / 25     # Normalized
            else:
                features["opp_team_pace"] = 1.0  # Average pace
                features["opp_team_offensive_rating"] = 1.0
                features["opp_team_ball_movement"] = 1.0

            # Recent form and momentum
            momentum_query = """
                SELECT 
                    game_date,
                    SUM(pts) as team_pts
                FROM player_games 
                WHERE team_id = ? AND game_date < ? AND game_date >= DATE(?, '-12 days')
                GROUP BY game_date
                ORDER BY game_date DESC
                LIMIT 4
            """

            momentum_df = pd.read_sql_query(
                momentum_query, conn, params=(opponent_team_id, target_date, target_date)
            )

            if len(momentum_df) >= 3:
                recent_scores = momentum_df["team_pts"].values
                features["opp_team_momentum"] = self._calculate_momentum(recent_scores)
                
                # Safe consistency calculation to avoid division by zero
                mean_score = np.mean(recent_scores)
                std_score = np.std(recent_scores)
                if mean_score > 0:
                    features["opp_team_consistency"] = 1.0 / (1.0 + std_score / mean_score)
                else:
                    features["opp_team_consistency"] = 0.8  # Default
                    
                features["opp_recent_avg_pts"] = mean_score
            else:
                features["opp_team_momentum"] = 0.0
                features["opp_team_consistency"] = 0.8
                features["opp_recent_avg_pts"] = 110

        except Exception as e:
            logger.warning(f"Error creating team strength features: {e}")
            # Set reasonable defaults
            features.update({
                "opp_def_efficiency": 1.1,
                "opp_reb_def_rate": 1.0,
                "opp_def_fg_pct": 0.46,
                "opp_def_3pt_pct": 0.35,
                "opp_steal_rate": 0.8,
                "opp_block_rate": 0.8,
                "opp_pts_def_relative": 1.0,
                "opp_fg_def_relative": 1.0,
                "opp_team_pace": 1.0,
                "opp_team_offensive_rating": 1.0,
                "opp_team_ball_movement": 1.0,
                "opp_team_momentum": 0.0,
                "opp_team_consistency": 0.8,
                "opp_recent_avg_pts": 110,
            })

        return features

    def _create_opponent_specific_features(
        self,
        player_id: int,
        opponent_team_id: int,
        target_date: str,
        conn: sqlite3.Connection,
    ) -> Dict:
        """Create features specific to the opponent team and defensive matchups."""
        features = {}

        try:
            # Get opponent team's recent defensive stats
            # Disable this query since opponent_team_id doesn't exist
            # Use simplified query for team averages instead  
            query = """
                SELECT 
                    AVG(pg.pts) as avg_pts_allowed,
                    AVG(pg.reb) as avg_reb_allowed,
                    AVG(pg.ast) as avg_ast_allowed,
                    AVG(pg.stl) as avg_stl_allowed,
                    AVG(pg.blk) as avg_blk_allowed,
                    COUNT(*) as games_played
                FROM player_games pg
                WHERE pg.team_id = ? AND pg.game_date < ? AND pg.game_date >= DATE(?, '-30 days')
            """

            opponent_def_df = pd.read_sql_query(
                query, conn, params=(opponent_team_id, target_date, target_date)
            )

            if not opponent_def_df.empty:
                # Safely extract values with None checks
                pts_allowed = opponent_def_df["avg_pts_allowed"].iloc[0] or 112
                reb_allowed = opponent_def_df["avg_reb_allowed"].iloc[0] or 43
                ast_allowed = opponent_def_df["avg_ast_allowed"].iloc[0] or 25
                stl_allowed = opponent_def_df["avg_stl_allowed"].iloc[0] or 8
                blk_allowed = opponent_def_df["avg_blk_allowed"].iloc[0] or 5
                
                features["opp_def_pts_allowed"] = pts_allowed
                features["opp_def_reb_allowed"] = reb_allowed
                features["opp_def_ast_allowed"] = ast_allowed
                features["opp_def_stl_allowed"] = stl_allowed
                features["opp_def_blk_allowed"] = blk_allowed
                
                # Calculate defensive ratings with safe division
                league_avg_pts = 112  # Approximate NBA average
                features["opp_def_rating_pts"] = pts_allowed / league_avg_pts
                features["opp_def_rating_reb"] = reb_allowed / 43  # Avg rebounds
                features["opp_def_rating_ast"] = ast_allowed / 25  # Avg assists
            else:
                # League average defaults
                features["opp_def_pts_allowed"] = 112
                features["opp_def_reb_allowed"] = 43
                features["opp_def_ast_allowed"] = 25
                features["opp_def_stl_allowed"] = 8
                features["opp_def_blk_allowed"] = 5
                features["opp_def_rating_pts"] = 1.0
                features["opp_def_rating_reb"] = 1.0
                features["opp_def_rating_ast"] = 1.0

            # Get opponent's pace and style of play
            pace_query = """
                SELECT 
                    AVG(CAST(fg_pct AS FLOAT)) as avg_fg_pct,
                    AVG(CAST(fg3_pct AS FLOAT)) as avg_3pt_pct,
                    AVG(CAST(ft_pct AS FLOAT)) as avg_ft_pct,
                    COUNT(*) as games_count
                FROM player_games pg
                WHERE pg.team_id = ? AND pg.game_date < ? AND pg.game_date >= DATE(?, '-20 days')
                AND fg_pct IS NOT NULL AND fg_pct != ''
            """

            pace_df = pd.read_sql_query(
                pace_query, conn, params=(opponent_team_id, target_date, target_date)
            )

            if not pace_df.empty and pace_df["games_count"].iloc[0] > 5:
                features["opp_team_fg_pct"] = pace_df["avg_fg_pct"].iloc[0] or 0.45
                features["opp_team_3pt_pct"] = pace_df["avg_3pt_pct"].iloc[0] or 0.35
                features["opp_team_ft_pct"] = pace_df["avg_ft_pct"].iloc[0] or 0.75
            else:
                features["opp_team_fg_pct"] = 0.45  # League averages
                features["opp_team_3pt_pct"] = 0.35
                features["opp_team_ft_pct"] = 0.75

            # Head-to-head historical performance
            # Disable h2h query since opponent_team_id doesn't exist
            # Return default values instead
            h2h_query = """
                SELECT 
                    0 as avg_h2h_pts,
                    0 as avg_h2h_reb,
                    0 as avg_h2h_ast,
                    0 as avg_h2h_stl,
                    0 as avg_h2h_blk,
                    0 as h2h_games
                WHERE 1=0  -- Always return empty result
            """

            h2h_df = pd.read_sql_query(h2h_query, conn)

            if not h2h_df.empty and h2h_df["h2h_games"].iloc[0] > 0:
                features["h2h_pts_avg"] = h2h_df["avg_h2h_pts"].iloc[0]
                features["h2h_reb_avg"] = h2h_df["avg_h2h_reb"].iloc[0]
                features["h2h_ast_avg"] = h2h_df["avg_h2h_ast"].iloc[0]
                features["h2h_stl_avg"] = h2h_df["avg_h2h_stl"].iloc[0]
                features["h2h_blk_avg"] = h2h_df["avg_h2h_blk"].iloc[0]
                features["h2h_games_count"] = h2h_df["h2h_games"].iloc[0]
                features["has_h2h_history"] = 1
            else:
                features["h2h_pts_avg"] = 0
                features["h2h_reb_avg"] = 0
                features["h2h_ast_avg"] = 0
                features["h2h_stl_avg"] = 0
                features["h2h_blk_avg"] = 0
                features["h2h_games_count"] = 0
                features["has_h2h_history"] = 0

            # Advanced opponent strength features
            # Get opponent's recent form (W/L record approximation based on team performance)
            team_form_query = """
                SELECT 
                    AVG(CASE WHEN pts > 110 THEN 1 ELSE 0 END) as high_scoring_rate,
                    AVG(CASE WHEN pts < 100 THEN 1 ELSE 0 END) as low_scoring_rate,
                    COUNT(*) as team_games
                FROM (
                    SELECT game_date, SUM(pts) as pts
                    FROM player_games 
                    WHERE team_id = ? AND game_date < ? AND game_date >= DATE(?, '-15 days')
                    GROUP BY game_date
                ) team_scores
            """

            form_df = pd.read_sql_query(
                team_form_query, conn, params=(opponent_team_id, target_date, target_date)
            )

            if not form_df.empty:
                features["opp_high_scoring_rate"] = form_df["high_scoring_rate"].iloc[0] or 0.5
                features["opp_low_scoring_rate"] = form_df["low_scoring_rate"].iloc[0] or 0.3
            else:
                features["opp_high_scoring_rate"] = 0.5
                features["opp_low_scoring_rate"] = 0.3

            # Opponent positional strength (simplified)
            # Disable positional query since opponent_team_id and position don't exist
            pos_strength_query = """
                SELECT 
                    20 as guard_pts_allowed,
                    18 as forward_pts_allowed,
                    15 as center_pts_allowed
                WHERE 1=0  -- Always return empty result, use defaults
            """

            pos_df = pd.read_sql_query(pos_strength_query, conn)

            if not pos_df.empty:
                features["opp_guard_def"] = pos_df["guard_pts_allowed"].iloc[0] or 20
                features["opp_forward_def"] = pos_df["forward_pts_allowed"].iloc[0] or 18  
                features["opp_center_def"] = pos_df["center_pts_allowed"].iloc[0] or 15
            else:
                features["opp_guard_def"] = 20
                features["opp_forward_def"] = 18
                features["opp_center_def"] = 15

        except Exception as e:
            logger.warning(f"Error creating opponent features: {e}")
            # Set safe defaults
            features.update({
                "opp_def_pts_allowed": 112,
                "opp_def_reb_allowed": 43,
                "opp_def_ast_allowed": 25,
                "opp_def_stl_allowed": 8,
                "opp_def_blk_allowed": 5,
                "opp_def_rating_pts": 1.0,
                "opp_def_rating_reb": 1.0,
                "opp_def_rating_ast": 1.0,
                "opp_team_fg_pct": 0.45,
                "opp_team_3pt_pct": 0.35,
                "opp_team_ft_pct": 0.75,
                "h2h_pts_avg": 0,
                "h2h_reb_avg": 0,
                "h2h_ast_avg": 0,
                "h2h_games_count": 0,
                "has_h2h_history": 0,
                "opp_high_scoring_rate": 0.5,
                "opp_low_scoring_rate": 0.3,
            })

        return features

    def _create_head_to_head_features(
        self,
        player_id: int,
        opponent_team_id: int,
        target_date: str,
        conn: sqlite3.Connection,
    ) -> Dict:
        """Create head-to-head matchup features."""
        features = {}

        # Get historical performance vs this opponent
        query = """
            SELECT AVG(pts) as h2h_pts_avg, AVG(reb) as h2h_reb_avg, 
                   AVG(ast) as h2h_ast_avg, COUNT(*) as h2h_games,
                   MAX(pts) as h2h_pts_max, MIN(pts) as h2h_pts_min
            FROM player_games pg
            WHERE pg.player_id = ? AND pg.game_date < ?
            AND EXISTS (
                SELECT 1 FROM player_games pg2 
                WHERE pg2.game_date = pg.game_date 
                AND pg2.team_id = ? 
                AND pg2.player_id != pg.player_id
            )
        """

        try:
            result = conn.execute(
                query, (player_id, target_date, opponent_team_id)
            ).fetchone()
            if result and result[3] > 0:  # Has H2H history
                features["h2h_pts_avg"] = result[0] or 0
                features["h2h_reb_avg"] = result[1] or 0
                features["h2h_ast_avg"] = result[2] or 0
                features["h2h_games_count"] = result[3] or 0
                features["h2h_pts_max"] = result[4] or 0
                features["h2h_pts_min"] = result[5] or 0
                features["h2h_pts_range"] = (
                    features["h2h_pts_max"] - features["h2h_pts_min"]
                )
        except Exception as e:
            logger.warning(f"Could not get head-to-head features: {e}")

        return features

    def _create_rest_features(self, df: pd.DataFrame) -> Dict:
        """Create features related to rest and fatigue."""
        features = {}

        if len(df) < 2:
            return features

        # Sort by date
        df_sorted = df.sort_values("game_date").reset_index(drop=True)

        # Convert game_date to datetime
        df_sorted["game_date"] = pd.to_datetime(df_sorted["game_date"])

        # Days between games
        df_sorted["days_rest"] = df_sorted["game_date"].diff().dt.days

        # Recent rest patterns
        features["avg_days_rest"] = df_sorted["days_rest"].mean()
        features["last_rest_days"] = (
            df_sorted["days_rest"].iloc[-1]
            if not pd.isna(df_sorted["days_rest"].iloc[-1])
            else 1
        )

        # Back-to-back games
        features["recent_b2b_games"] = (df_sorted["days_rest"] == 1).sum()

        # Performance after rest
        well_rested_games = df_sorted[df_sorted["days_rest"] >= 2]
        if len(well_rested_games) > 0:
            features["pts_after_rest"] = well_rested_games["pts"].mean()
        else:
            features["pts_after_rest"] = 0

        return features

    def _create_consistency_features(self, df: pd.DataFrame) -> Dict:
        """Create features measuring player consistency."""
        features = {}

        stat_columns = ["pts", "reb", "ast"]

        for stat in stat_columns:
            if stat in df.columns and len(df) > 1:
                values = df[stat].values

                # Coefficient of variation
                mean_val = np.mean(values)
                std_val = np.std(values)
                features[f"{stat}_cv"] = std_val / mean_val if mean_val > 0 else 0

                # Percentage of games above average
                features[f"{stat}_above_avg_pct"] = (values > mean_val).mean()

                # Range (max - min)
                features[f"{stat}_range"] = np.max(values) - np.min(values)

        return features

    def _create_home_away_features(self, df: pd.DataFrame) -> Dict:
        """Create home/away performance features."""
        features = {}

        if "matchup" not in df.columns:
            return features

        # Determine home/away games
        df["is_home"] = df["matchup"].str.contains("vs.")

        home_games = df[df["is_home"] == True]
        away_games = df[df["is_home"] == False]

        # Home performance
        if len(home_games) > 0:
            features["home_pts_avg"] = home_games["pts"].mean()
            features["home_games_count"] = len(home_games)
        else:
            features["home_pts_avg"] = 0
            features["home_games_count"] = 0

        # Away performance
        if len(away_games) > 0:
            features["away_pts_avg"] = away_games["pts"].mean()
            features["away_games_count"] = len(away_games)
        else:
            features["away_pts_avg"] = 0
            features["away_games_count"] = 0

        # Home/away differential
        features["home_away_pts_diff"] = (
            features["home_pts_avg"] - features["away_pts_avg"]
        )

        return features

    def _create_fatigue_features(self, df: pd.DataFrame) -> Dict:
        """Create advanced fatigue and load management features."""
        features = {}

        if len(df) < 3:
            return features

        # Sort by date
        df_sorted = df.sort_values("game_date").reset_index(drop=True)

        # Convert game_date to datetime for calculations
        df_sorted["game_date"] = pd.to_datetime(df_sorted["game_date"])

        # Advanced rest patterns
        df_sorted["days_rest"] = df_sorted["game_date"].diff().dt.days

        if "min" in df_sorted.columns:
            minutes = df_sorted["min"].values

            # Average minutes over different periods
            features["avg_min_season"] = np.mean(minutes)

            if len(minutes) >= 10:
                features["avg_min_last_10"] = np.mean(minutes[-10:])

            if len(minutes) >= 5:
                features["avg_min_last_5"] = np.mean(minutes[-5:])

            # Minutes load trend
            if len(minutes) >= 10:
                features["min_trend"] = self._calculate_trend(minutes[-10:])

            # Heavy minute games (>35 minutes)
            heavy_games = (minutes > 35).sum() if len(minutes) > 0 else 0
            features["heavy_min_games_pct"] = (
                heavy_games / len(minutes) if len(minutes) > 0 else 0
            )

        # Back-to-back patterns
        if "days_rest" in df_sorted.columns:
            b2b_games = (df_sorted["days_rest"] == 1).sum()
            total_games = len(df_sorted)
            features["b2b_games_pct"] = (
                b2b_games / total_games if total_games > 0 else 0
            )

            # Rest distribution
            rest_days = df_sorted["days_rest"].dropna()
            if len(rest_days) > 0:
                features["avg_rest_days"] = rest_days.mean()
                features["rest_consistency"] = 1 / (
                    rest_days.std() + 1
                )  # Higher = more consistent

        return features

    def _calculate_trend(self, values: np.ndarray) -> float:
        """Calculate the trend (slope) of a series of values."""
        if len(values) < 2:
            return 0.0

        # Remove NaN values
        clean_values = values[~np.isnan(values)]
        if len(clean_values) < 2:
            return 0.0

        x = np.arange(len(clean_values))
        slope = np.polyfit(x, clean_values, 1)[0]

        return slope

    def _calculate_momentum(self, values: np.ndarray) -> float:
        """Calculate momentum as weighted recent performance trend."""
        if len(values) < 2:
            return 0.0

        # Weight recent games more heavily
        weights = np.linspace(0.5, 1.0, len(values))
        try:
            weighted_trend = self._calculate_trend(values * weights)
            return weighted_trend
        except:
            return 0.0

    def _calculate_current_streak(self, values: np.ndarray, threshold: float) -> int:
        """Calculate current streak above/below threshold."""
        if len(values) == 0:
            return 0

        streak = 0
        above_threshold = values[-1] > threshold

        for i in range(len(values) - 1, -1, -1):
            if (values[i] > threshold) == above_threshold:
                streak += 1
            else:
                break

        # Return positive for above threshold, negative for below
        return streak if above_threshold else -streak

    def create_training_dataset(
        self,
        players_list: List[int],
        start_date: str,
        end_date: str,
        target_stats: List[str] = ["pts", "reb", "ast"],
        include_h2h_features: bool = True,
        include_advanced_features: bool = True,
    ) -> pd.DataFrame:
        """Create a complete training dataset with advanced features and targets."""
        all_features = []

        conn = sqlite3.connect(self.db_path)
        logger.info(
            f"Creating training dataset for {len(players_list)} players from {start_date} to {end_date}"
        )

        # Progress bar for players
        with tqdm(
            players_list,
            desc="Processing Players",
            ncols=80,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as pbar:
            for player_id in pbar:
                # Get all games for this player in the date range
                query = """
                    SELECT * FROM player_games 
                    WHERE player_id = ? AND game_date BETWEEN ? AND ?
                    ORDER BY game_date
                """

                player_games = pd.read_sql_query(
                    query, conn, params=(player_id, start_date, end_date)
                )

                if len(player_games) < 12:  # Reduced from 15 to 12 games minimum
                    pbar.write(
                        f"   ⚠️  Skipping player {player_id} - insufficient games ({len(player_games)})"
                    )
                    continue

                pbar.set_description(f"Player {player_id} ({len(player_games)} games)")

                # Calculate total games to process for this player
                games_to_process = (
                    len(player_games) - 12
                )  # Start from game 12 instead of 15
                if games_to_process <= 0:
                    continue

                # Progress bar for games within each player
                game_indices = range(12, len(player_games))  # Start from game 12
                player_samples_count = 0

                with tqdm(
                    game_indices,
                    desc=f"Games (P{player_id})",
                    ncols=60,
                    bar_format="{desc}: {n_fmt}/{total_fmt}",
                    leave=False,
                ) as game_pbar:

                    # For each game, create features based on previous games
                    for i in game_pbar:
                        target_game = player_games.iloc[i]
                        target_date = target_game["game_date"]

                        # Extract opponent team ID if including h2h features
                        opponent_team_id = None
                        if include_h2h_features and "matchup" in target_game:
                            matchup = str(target_game["matchup"])
                            # Simple opponent extraction - in production you'd want proper team mapping
                            if "vs." in matchup:
                                opponent_name = matchup.split("vs. ")[-1].strip()
                            elif "@" in matchup:
                                opponent_name = matchup.split("@ ")[-1].strip()
                            else:
                                opponent_name = None

                            if opponent_name:
                                # Use a hash-based approach for demo - replace with actual team IDs
                                opponent_team_id = abs(hash(opponent_name)) % 1000000

                        try:
                            # Create advanced features (reduced lookback for faster processing)
                            features_df = self.create_features_for_player(
                                player_id,
                                target_date,
                                lookback_games=15,  # Reduced from 20 to 15 games
                                opponent_team_id=opponent_team_id,
                                include_h2h_features=include_h2h_features,
                                include_advanced_features=include_advanced_features,
                            )

                            if not features_df.empty:
                                # Add target values
                                for stat in target_stats:
                                    if stat in target_game:
                                        features_df[f"target_{stat}"] = target_game[
                                            stat
                                        ]

                                # Add game metadata
                                features_df["game_id"] = target_game.get("game_id", "")
                                features_df["game_date"] = target_date

                                all_features.append(features_df)
                                player_samples_count += 1

                        except Exception as e:
                            game_pbar.write(
                                f"     ⚠️  Error creating features for player {player_id} on {target_date}: {e}"
                            )
                            continue

                # Update main progress bar with summary
                pbar.write(
                    f"   ✅ Player {player_id}: {player_samples_count} training samples created"
                )

        conn.close()

        if all_features:
            final_dataset = pd.concat(all_features, ignore_index=True)
            tqdm.write(
                f"🎯 Final dataset: {len(final_dataset)} samples with {len(final_dataset.columns)} features"
            )
            logger.info(
                f"Created training dataset with {len(final_dataset)} samples and {len(final_dataset.columns)} features"
            )
            return final_dataset
        else:
            tqdm.write("❌ No training data created")
            logger.warning("No training data created")
            return pd.DataFrame()

    def prepare_features_for_prediction(
        self, features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Prepare features for model prediction with advanced preprocessing."""
        if features_df.empty:
            return features_df

        # Make a copy to avoid modifying the original
        prepared_df = features_df.copy()

        # Handle missing values with more sophisticated imputation
        numeric_columns = prepared_df.select_dtypes(include=[np.number]).columns

        # Exclude metadata columns from processing
        exclude_columns = [
            "player_id",
            "target_date",
            "game_id",
            "game_date",
            "opponent_team_id",
        ]
        exclude_columns.extend(
            [col for col in prepared_df.columns if col.startswith("target_")]
        )

        process_columns = [col for col in numeric_columns if col not in exclude_columns]

        if process_columns:
            # Fill missing values with median (more robust than mean)
            prepared_df[process_columns] = prepared_df[process_columns].fillna(
                prepared_df[process_columns].median()
            )

            # Handle any remaining NaN values
            prepared_df[process_columns] = prepared_df[process_columns].fillna(0)

            # Detect and handle outliers using IQR method
            for col in process_columns:
                Q1 = prepared_df[col].quantile(0.25)
                Q3 = prepared_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Cap outliers instead of removing them
                prepared_df[col] = prepared_df[col].clip(
                    lower=lower_bound, upper=upper_bound
                )

        return prepared_df


# For backward compatibility, create an alias
FeatureEngineer = AdvancedFeatureEngineer
