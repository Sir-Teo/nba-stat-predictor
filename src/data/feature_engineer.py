"""
Feature Engineering Module - Creates advanced features for NBA stat prediction models.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sqlite3
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.cluster import KMeans
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class AdvancedFeatureEngineer:
    """Creates and processes advanced features for NBA stat prediction."""
    
    def __init__(self, db_path: str = "data/nba_data.db"):
        """Initialize the advanced feature engineer."""
        self.db_path = db_path
        self.scaler = RobustScaler()
        self.label_encoders = {}
        self.team_clusters = {}
        self.pace_adjustments = {}
        
    def create_features_for_player(self, player_id: int, target_date: str, 
                                  lookback_games: int = 20, opponent_team_id: int = None, 
                                  include_h2h_features: bool = True,
                                  include_advanced_features: bool = True) -> pd.DataFrame:
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
        
        df = pd.read_sql_query(query, conn, params=(player_id, target_date, lookback_games))
        
        if df.empty:
            conn.close()
            return pd.DataFrame()
        
        features = {}
        
        # Core rolling statistics with multiple windows
        features.update(self._create_rolling_stats(df, windows=[3, 5, 10, 15, 20]))
        
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
            features.update(self._create_team_strength_features(opponent_team_id, target_date, conn))
            features.update(self._create_opponent_specific_features(player_id, opponent_team_id, target_date, conn))
            features.update(self._create_head_to_head_features(player_id, opponent_team_id, target_date, conn))
        
        conn.close()
        
        # Convert to DataFrame
        features_df = pd.DataFrame([features])
        features_df['player_id'] = player_id
        features_df['target_date'] = target_date
        if opponent_team_id is not None:
            features_df['opponent_team_id'] = opponent_team_id
        
        return features_df
    
    def _create_rolling_stats(self, df: pd.DataFrame, windows: List[int] = [3, 5, 10, 15, 20]) -> Dict:
        """Create comprehensive rolling statistics for different time windows."""
        features = {}
        
        stat_columns = ['pts', 'reb', 'ast', 'stl', 'blk', 'tov', 'fg_pct', 'fg3_pct', 'ft_pct', 'min']
        
        # Ensure data is sorted by date
        df_sorted = df.sort_values('game_date').reset_index(drop=True)
        
        for window in windows:
            for stat in stat_columns:
                if stat in df_sorted.columns:
                    # Basic rolling statistics
                    rolling_values = df_sorted[stat].rolling(window=window, min_periods=1)
                    
                    features[f'{stat}_avg_{window}g'] = rolling_values.mean().iloc[-1]
                    features[f'{stat}_std_{window}g'] = rolling_values.std().iloc[-1] or 0
                    features[f'{stat}_min_{window}g'] = rolling_values.min().iloc[-1]
                    features[f'{stat}_max_{window}g'] = rolling_values.max().iloc[-1]
                    
                    # Advanced rolling statistics
                    features[f'{stat}_median_{window}g'] = rolling_values.median().iloc[-1]
                    features[f'{stat}_q25_{window}g'] = rolling_values.quantile(0.25).iloc[-1]
                    features[f'{stat}_q75_{window}g'] = rolling_values.quantile(0.75).iloc[-1]
                    features[f'{stat}_iqr_{window}g'] = features[f'{stat}_q75_{window}g'] - features[f'{stat}_q25_{window}g']
                    
                    # Coefficient of variation (relative volatility)
                    mean_val = features[f'{stat}_avg_{window}g']
                    std_val = features[f'{stat}_std_{window}g']
                    features[f'{stat}_cv_{window}g'] = (std_val / mean_val) if mean_val > 0 else 0
                    
                    # Rolling trend and momentum
                    if len(df_sorted) >= window:
                        recent_values = df_sorted[stat].tail(window).values
                        features[f'{stat}_trend_{window}g'] = self._calculate_trend(recent_values)
                        features[f'{stat}_momentum_{window}g'] = self._calculate_momentum(recent_values)
                    
                    # Performance vs season average
                    season_avg = df_sorted[stat].mean()
                    features[f'{stat}_vs_season_{window}g'] = features[f'{stat}_avg_{window}g'] - season_avg
                    
                    # Recent vs historical performance
                    if window <= 10 and len(df_sorted) > window:
                        recent_avg = df_sorted[stat].tail(window).mean()
                        historical_avg = df_sorted[stat].iloc[:-window].mean()
                        features[f'{stat}_recent_vs_hist_{window}g'] = recent_avg - historical_avg
        
        return features
    
    def _create_advanced_form_features(self, df: pd.DataFrame) -> Dict:
        """Create advanced performance trends and form features."""
        features = {}
        
        if len(df) < 3:
            return features
            
        df_sorted = df.sort_values('game_date').reset_index(drop=True)
        
        # Multi-game performance patterns
        for stat in ['pts', 'reb', 'ast']:
            if stat in df_sorted.columns:
                values = df_sorted[stat].values
                season_avg = values.mean()
                
                # Performance streaks
                above_avg_streak = self._calculate_current_streak(values, season_avg)
                features[f'{stat}_above_avg_streak'] = above_avg_streak
                
                # Volatility measures
                features[f'{stat}_volatility_score'] = np.std(values) / (np.mean(values) + 1e-8)
                
                # Form curve (recent 5 vs previous 5)
                if len(values) >= 10:
                    recent_5 = values[-5:].mean()
                    prev_5 = values[-10:-5].mean()
                    features[f'{stat}_form_curve'] = recent_5 - prev_5
                
                # Peak performance indicators
                features[f'{stat}_games_above_peak'] = np.sum(values >= np.percentile(values, 90))
                features[f'{stat}_consistency_score'] = 1 - (np.std(values) / (np.mean(values) + 1e-8))
                
                # Hot/cold streaks
                hot_games = np.sum(values >= season_avg * 1.2)  # 20% above average
                cold_games = np.sum(values <= season_avg * 0.8)  # 20% below average
                features[f'{stat}_hot_games_pct'] = hot_games / len(values)
                features[f'{stat}_cold_games_pct'] = cold_games / len(values)
        
        return features
    
    def _create_situational_features(self, df: pd.DataFrame) -> Dict:
        """Create features based on game situations and contexts."""
        features = {}
        
        if len(df) < 2:
            return features
        
        df_sorted = df.sort_values('game_date').reset_index(drop=True)
        
        # Day of week effects
        df_sorted['game_date_dt'] = pd.to_datetime(df_sorted['game_date'])
        df_sorted['day_of_week'] = df_sorted['game_date_dt'].dt.dayofweek
        
        # Performance by day of week
        for day in range(7):
            day_games = df_sorted[df_sorted['day_of_week'] == day]
            if len(day_games) > 0:
                features[f'pts_avg_day_{day}'] = day_games['pts'].mean()
                features[f'games_on_day_{day}'] = len(day_games)
        
        # Back-to-back performance
        df_sorted['days_rest'] = df_sorted['game_date_dt'].diff().dt.days
        b2b_games = df_sorted[df_sorted['days_rest'] <= 1]
        if len(b2b_games) > 0:
            features['pts_avg_b2b'] = b2b_games['pts'].mean()
            features['b2b_games_count'] = len(b2b_games)
            features['b2b_performance_drop'] = df_sorted['pts'].mean() - b2b_games['pts'].mean()
        
        # Win/Loss context
        if 'wl' in df_sorted.columns:
            wins = df_sorted[df_sorted['wl'] == 'W']
            losses = df_sorted[df_sorted['wl'] == 'L']
            
            if len(wins) > 0:
                features['pts_in_wins'] = wins['pts'].mean()
                features['win_rate'] = len(wins) / len(df_sorted)
            
            if len(losses) > 0:
                features['pts_in_losses'] = losses['pts'].mean()
        
        # Monthly performance trends
        df_sorted['month'] = df_sorted['game_date_dt'].dt.month
        current_month = df_sorted['month'].iloc[-1]
        current_month_games = df_sorted[df_sorted['month'] == current_month]
        if len(current_month_games) > 1:
            features['pts_current_month'] = current_month_games['pts'].mean()
            features['games_current_month'] = len(current_month_games)
        
        return features
    
    def _create_usage_features(self, df: pd.DataFrame) -> Dict:
        """Create features related to player usage and role."""
        features = {}
        
        if len(df) < 3:
            return features
        
        # Usage indicators
        avg_min = df['min'].mean()
        features['avg_minutes'] = avg_min
        features['high_usage_games'] = np.sum(df['min'] >= avg_min * 1.2)
        features['low_usage_games'] = np.sum(df['min'] <= avg_min * 0.8)
        
        # Efficiency metrics
        if 'fga' in df.columns and 'pts' in df.columns:
            total_fga = df['fga'].sum()
            total_pts = df['pts'].sum()
            if total_fga > 0:
                features['points_per_fga'] = total_pts / total_fga
        
        # Role consistency
        features['minutes_consistency'] = 1 - (df['min'].std() / (df['min'].mean() + 1e-8))
        
        # Production per minute
        if avg_min > 0:
            for stat in ['pts', 'reb', 'ast']:
                if stat in df.columns:
                    features[f'{stat}_per_minute'] = df[stat].mean() / avg_min
        
        return features
    
    def _create_opponent_type_features(self, df: pd.DataFrame, conn: sqlite3.Connection) -> Dict:
        """Create features based on opponent strength and type."""
        features = {}
        
        # This would require opponent team data - simplified version
        if 'matchup' in df.columns:
            # Extract opponent from matchup string
            home_games = df['matchup'].str.contains('vs.').sum()
            away_games = len(df) - home_games
            
            features['home_game_ratio'] = home_games / len(df)
            features['away_game_ratio'] = away_games / len(df)
            
            # Performance splits
            if home_games > 0:
                home_df = df[df['matchup'].str.contains('vs.')]
                features['pts_vs_home'] = home_df['pts'].mean()
            
            if away_games > 0:
                away_df = df[~df['matchup'].str.contains('vs.')]
                features['pts_vs_away'] = away_df['pts'].mean()
        
        return features
    
    def _create_pace_adjusted_features(self, df: pd.DataFrame, conn: sqlite3.Connection) -> Dict:
        """Create pace-adjusted performance features."""
        features = {}
        
        # Simplified pace adjustment - would need team pace data for full implementation
        avg_min = df['min'].mean()
        if avg_min > 0:
            # Estimate possessions based on minutes played
            est_possessions = avg_min * 1.2  # Rough estimate
            
            for stat in ['pts', 'reb', 'ast']:
                if stat in df.columns:
                    features[f'{stat}_per_100_poss'] = (df[stat].mean() / est_possessions) * 100
        
        return features
    
    def _create_clutch_performance_features(self, df: pd.DataFrame) -> Dict:
        """Create features for clutch/high-pressure performance."""
        features = {}
        
        # This would require play-by-play data for true clutch stats
        # Using close games as proxy for clutch situations
        if 'plus_minus' in df.columns:
            close_games = df[abs(df['plus_minus']) <= 5]  # Games decided by 5 or less
            if len(close_games) > 0:
                features['clutch_games_count'] = len(close_games)
                features['clutch_pts_avg'] = close_games['pts'].mean()
                features['clutch_performance'] = close_games['pts'].mean() - df['pts'].mean()
        
        return features
    
    def _create_momentum_shift_features(self, df: pd.DataFrame) -> Dict:
        """Create features capturing momentum and performance shifts."""
        features = {}
        
        if len(df) < 5:
            return features
        
        # Performance acceleration/deceleration
        for stat in ['pts', 'reb', 'ast']:
            if stat in df.columns:
                values = df.sort_values('game_date')[stat].values
                
                # Second derivative (acceleration)
                if len(values) >= 3:
                    first_diff = np.diff(values)
                    second_diff = np.diff(first_diff)
                    features[f'{stat}_acceleration'] = np.mean(second_diff[-3:])  # Recent acceleration
                
                # Momentum score (weighted recent performance)
                weights = np.exp(np.linspace(0, 1, len(values)))
                features[f'{stat}_momentum_score'] = np.average(values, weights=weights)
        
        return features
    
    def _create_game_context_features(self, df: pd.DataFrame) -> Dict:
        """Create features based on broader game context."""
        features = {}
        
        # Performance in different game outcomes
        if 'wl' in df.columns and 'plus_minus' in df.columns:
            # Blowout vs close game performance
            blowouts = df[abs(df['plus_minus']) >= 20]
            close_games = df[abs(df['plus_minus']) <= 5]
            
            if len(blowouts) > 0:
                features['blowout_pts_avg'] = blowouts['pts'].mean()
                features['blowout_games_count'] = len(blowouts)
            
            if len(close_games) > 0:
                features['close_game_pts_avg'] = close_games['pts'].mean()
                features['close_games_count'] = len(close_games)
        
        return features
    
    def _create_team_strength_features(self, opponent_team_id: int, target_date: str, 
                                     conn: sqlite3.Connection) -> Dict:
        """Create features based on opponent team strength."""
        features = {}
        
        # Get opponent team's recent performance
        query = """
            SELECT AVG(pts) as opp_pts_allowed, AVG(reb) as opp_reb_allowed,
                   AVG(ast) as opp_ast_allowed, COUNT(*) as opp_games
            FROM player_games
            WHERE team_id != ? AND game_date < ? AND game_date >= date(?, '-30 days')
        """
        
        try:
            result = conn.execute(query, (opponent_team_id, target_date, target_date)).fetchone()
            if result and result[3] > 5:  # At least 5 games of data
                features['opp_pts_allowed_avg'] = result[0] or 0
                features['opp_reb_allowed_avg'] = result[1] or 0
                features['opp_ast_allowed_avg'] = result[2] or 0
                features['opp_defensive_games'] = result[3] or 0
        except Exception as e:
            logger.warning(f"Could not get opponent strength features: {e}")
        
        return features
    
    def _create_head_to_head_features(self, player_id: int, opponent_team_id: int, 
                                    target_date: str, conn: sqlite3.Connection) -> Dict:
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
            result = conn.execute(query, (player_id, target_date, opponent_team_id)).fetchone()
            if result and result[3] > 0:  # Has H2H history
                features['h2h_pts_avg'] = result[0] or 0
                features['h2h_reb_avg'] = result[1] or 0
                features['h2h_ast_avg'] = result[2] or 0
                features['h2h_games_count'] = result[3] or 0
                features['h2h_pts_max'] = result[4] or 0
                features['h2h_pts_min'] = result[5] or 0
                features['h2h_pts_range'] = features['h2h_pts_max'] - features['h2h_pts_min']
        except Exception as e:
            logger.warning(f"Could not get head-to-head features: {e}")
        
        return features

    def _create_rest_features(self, df: pd.DataFrame) -> Dict:
        """Create features related to rest and fatigue."""
        features = {}
        
        if len(df) < 2:
            return features
        
        # Sort by date
        df_sorted = df.sort_values('game_date').reset_index(drop=True)
        
        # Convert game_date to datetime
        df_sorted['game_date'] = pd.to_datetime(df_sorted['game_date'])
        
        # Days between games
        df_sorted['days_rest'] = df_sorted['game_date'].diff().dt.days
        
        # Recent rest patterns
        features['avg_days_rest'] = df_sorted['days_rest'].mean()
        features['last_rest_days'] = df_sorted['days_rest'].iloc[-1] if not pd.isna(df_sorted['days_rest'].iloc[-1]) else 1
        
        # Back-to-back games
        features['recent_b2b_games'] = (df_sorted['days_rest'] == 1).sum()
        
        # Performance after rest
        well_rested_games = df_sorted[df_sorted['days_rest'] >= 2]
        if len(well_rested_games) > 0:
            features['pts_after_rest'] = well_rested_games['pts'].mean()
        else:
            features['pts_after_rest'] = 0
        
        return features
    
    def _create_consistency_features(self, df: pd.DataFrame) -> Dict:
        """Create features measuring player consistency."""
        features = {}
        
        stat_columns = ['pts', 'reb', 'ast']
        
        for stat in stat_columns:
            if stat in df.columns and len(df) > 1:
                values = df[stat].values
                
                # Coefficient of variation
                mean_val = np.mean(values)
                std_val = np.std(values)
                features[f'{stat}_cv'] = std_val / mean_val if mean_val > 0 else 0
                
                # Percentage of games above average
                features[f'{stat}_above_avg_pct'] = (values > mean_val).mean()
                
                # Range (max - min)
                features[f'{stat}_range'] = np.max(values) - np.min(values)
        
        return features
    
    def _create_home_away_features(self, df: pd.DataFrame) -> Dict:
        """Create home/away performance features."""
        features = {}
        
        if 'matchup' not in df.columns:
            return features
        
        # Determine home/away games
        df['is_home'] = df['matchup'].str.contains('vs.')
        
        home_games = df[df['is_home'] == True]
        away_games = df[df['is_home'] == False]
        
        # Home performance
        if len(home_games) > 0:
            features['home_pts_avg'] = home_games['pts'].mean()
            features['home_games_count'] = len(home_games)
        else:
            features['home_pts_avg'] = 0
            features['home_games_count'] = 0
        
        # Away performance
        if len(away_games) > 0:
            features['away_pts_avg'] = away_games['pts'].mean()
            features['away_games_count'] = len(away_games)
        else:
            features['away_pts_avg'] = 0
            features['away_games_count'] = 0
        
        # Home/away differential
        features['home_away_pts_diff'] = features['home_pts_avg'] - features['away_pts_avg']
        
        return features
    
    def _create_fatigue_features(self, df: pd.DataFrame) -> Dict:
        """Create advanced fatigue and load management features."""
        features = {}
        
        if len(df) < 3:
            return features
        
        # Sort by date
        df_sorted = df.sort_values('game_date').reset_index(drop=True)
        
        # Convert game_date to datetime for calculations
        df_sorted['game_date'] = pd.to_datetime(df_sorted['game_date'])
        
        # Advanced rest patterns
        df_sorted['days_rest'] = df_sorted['game_date'].diff().dt.days
        
        if 'min' in df_sorted.columns:
            minutes = df_sorted['min'].values
            
            # Average minutes over different periods
            features['avg_min_season'] = np.mean(minutes)
            
            if len(minutes) >= 10:
                features['avg_min_last_10'] = np.mean(minutes[-10:])
                
            if len(minutes) >= 5:
                features['avg_min_last_5'] = np.mean(minutes[-5:])
            
            # Minutes load trend
            if len(minutes) >= 10:
                features['min_trend'] = self._calculate_trend(minutes[-10:])
            
            # Heavy minute games (>35 minutes)
            heavy_games = (minutes > 35).sum() if len(minutes) > 0 else 0
            features['heavy_min_games_pct'] = heavy_games / len(minutes) if len(minutes) > 0 else 0
        
        # Back-to-back patterns
        if 'days_rest' in df_sorted.columns:
            b2b_games = (df_sorted['days_rest'] == 1).sum()
            total_games = len(df_sorted)
            features['b2b_games_pct'] = b2b_games / total_games if total_games > 0 else 0
            
            # Rest distribution
            rest_days = df_sorted['days_rest'].dropna()
            if len(rest_days) > 0:
                features['avg_rest_days'] = rest_days.mean()
                features['rest_consistency'] = 1 / (rest_days.std() + 1)  # Higher = more consistent
        
        return features
    
    def _create_opponent_specific_features(self, player_id: int, opponent_team_id: int, 
                                         target_date: str, conn: sqlite3.Connection) -> Dict:
        """Create features specific to matchup against opponent team."""
        features = {}
        
        try:
            # Get historical performance against this opponent
            h2h_query = """
                SELECT pts, reb, ast, stl, blk, min, game_date, wl
                FROM player_games 
                WHERE player_id = ? AND game_date < ?
                AND (matchup LIKE ? OR matchup LIKE ?)
                ORDER BY game_date DESC
                LIMIT 10
            """
            
            # Create matchup patterns for both home and away games
            home_pattern = f"%vs. {opponent_team_id}%"
            away_pattern = f"%@ {opponent_team_id}%"
            
            h2h_df = pd.read_sql_query(h2h_query, conn, 
                                     params=(player_id, target_date, home_pattern, away_pattern))
            
            if not h2h_df.empty:
                # Head-to-head averages
                features['h2h_pts_avg'] = h2h_df['pts'].mean()
                features['h2h_reb_avg'] = h2h_df['reb'].mean()
                features['h2h_ast_avg'] = h2h_df['ast'].mean()
                features['h2h_stl_avg'] = h2h_df['stl'].mean()
                features['h2h_blk_avg'] = h2h_df['blk'].mean()
                features['h2h_min_avg'] = h2h_df['min'].mean()
                features['h2h_games_count'] = len(h2h_df)
                
                # Win rate against opponent
                features['h2h_win_rate'] = (h2h_df['wl'] == 'W').mean()
                
                # Recent performance vs opponent (last 3 games)
                recent_h2h = h2h_df.head(3)
                if len(recent_h2h) > 0:
                    features['h2h_recent_pts_avg'] = recent_h2h['pts'].mean()
                    features['h2h_recent_games'] = len(recent_h2h)
                else:
                    features['h2h_recent_pts_avg'] = 0
                    features['h2h_recent_games'] = 0
                
                # Performance trend vs opponent
                if len(h2h_df) >= 3:
                    features['h2h_pts_trend'] = self._calculate_trend(h2h_df['pts'].values)
                else:
                    features['h2h_pts_trend'] = 0
                    
            else:
                # No historical data against this opponent
                features['h2h_pts_avg'] = 0
                features['h2h_reb_avg'] = 0
                features['h2h_ast_avg'] = 0
                features['h2h_stl_avg'] = 0
                features['h2h_blk_avg'] = 0
                features['h2h_min_avg'] = 0
                features['h2h_games_count'] = 0
                features['h2h_win_rate'] = 0
                features['h2h_recent_pts_avg'] = 0
                features['h2h_recent_games'] = 0
                features['h2h_pts_trend'] = 0
            
            # Get opponent team's defensive stats (if available)
            opponent_defense_query = """
                SELECT AVG(pts) as avg_pts_allowed, COUNT(*) as games_played
                FROM player_games 
                WHERE game_date < ? AND game_date >= date(?, '-30 days')
                AND (matchup LIKE ? OR matchup LIKE ?)
            """
            
            # Patterns to find games against this opponent team
            vs_opponent_home = f"%vs. {opponent_team_id}%"
            vs_opponent_away = f"%@ {opponent_team_id}%"
            
            try:
                opponent_defense = pd.read_sql_query(opponent_defense_query, conn,
                                                   params=(target_date, target_date, 
                                                          vs_opponent_home, vs_opponent_away))
                
                if not opponent_defense.empty and opponent_defense.iloc[0]['games_played'] > 0:
                    features['opponent_avg_pts_allowed'] = opponent_defense.iloc[0]['avg_pts_allowed']
                else:
                    features['opponent_avg_pts_allowed'] = 110  # NBA average
            except:
                features['opponent_avg_pts_allowed'] = 110  # NBA average
            
        except Exception as e:
            logger.error(f"Error creating opponent-specific features: {e}")
            # Set default values if error occurs
            features.update({
                'h2h_pts_avg': 0, 'h2h_reb_avg': 0, 'h2h_ast_avg': 0,
                'h2h_stl_avg': 0, 'h2h_blk_avg': 0, 'h2h_min_avg': 0,
                'h2h_games_count': 0, 'h2h_win_rate': 0,
                'h2h_recent_pts_avg': 0, 'h2h_recent_games': 0,
                'h2h_pts_trend': 0, 'opponent_avg_pts_allowed': 110
            })
        
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
    
    def create_training_dataset(self, players_list: List[int], 
                               start_date: str, end_date: str,
                               target_stats: List[str] = ['pts', 'reb', 'ast'],
                               include_h2h_features: bool = True,
                               include_advanced_features: bool = True) -> pd.DataFrame:
        """Create a complete training dataset with advanced features and targets."""
        all_features = []
        
        conn = sqlite3.connect(self.db_path)
        logger.info(f"Creating training dataset for {len(players_list)} players from {start_date} to {end_date}")
        
        # Progress bar for players
        with tqdm(players_list, desc="Processing Players", ncols=80,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
            for player_id in pbar:
                # Get all games for this player in the date range
                query = """
                    SELECT * FROM player_games 
                    WHERE player_id = ? AND game_date BETWEEN ? AND ?
                    ORDER BY game_date
                """
                
                player_games = pd.read_sql_query(query, conn, params=(player_id, start_date, end_date))
                
                if len(player_games) < 15:  # Need sufficient history for advanced features
                    pbar.write(f"   ⚠️  Skipping player {player_id} - insufficient games ({len(player_games)})")
                    continue
                
                pbar.set_description(f"Player {player_id} ({len(player_games)} games)")
                
                # Calculate total games to process for this player
                games_to_process = len(player_games) - 15
                if games_to_process <= 0:
                    continue
                
                # Progress bar for games within each player
                game_indices = range(15, len(player_games))
                player_samples_count = 0
                
                with tqdm(game_indices, desc=f"Games (P{player_id})", ncols=60, 
                          bar_format="{desc}: {n_fmt}/{total_fmt}", leave=False) as game_pbar:
                    
                    # For each game, create features based on previous games
                    for i in game_pbar:
                        target_game = player_games.iloc[i]
                        target_date = target_game['game_date']
                        
                        # Extract opponent team ID if including h2h features
                        opponent_team_id = None
                        if include_h2h_features and 'matchup' in target_game:
                            matchup = str(target_game['matchup'])
                            # Simple opponent extraction - in production you'd want proper team mapping
                            if 'vs.' in matchup:
                                opponent_name = matchup.split('vs. ')[-1].strip()
                            elif '@' in matchup:
                                opponent_name = matchup.split('@ ')[-1].strip()
                            else:
                                opponent_name = None
                            
                            if opponent_name:
                                # Use a hash-based approach for demo - replace with actual team IDs
                                opponent_team_id = abs(hash(opponent_name)) % 1000000
                        
                        try:
                            # Create advanced features
                            features_df = self.create_features_for_player(
                                player_id, target_date, 
                                lookback_games=20,
                                opponent_team_id=opponent_team_id,
                                include_h2h_features=include_h2h_features,
                                include_advanced_features=include_advanced_features
                            )
                            
                            if not features_df.empty:
                                # Add target values
                                for stat in target_stats:
                                    if stat in target_game:
                                        features_df[f'target_{stat}'] = target_game[stat]
                                
                                # Add game metadata
                                features_df['game_id'] = target_game.get('game_id', '')
                                features_df['game_date'] = target_date
                                
                                all_features.append(features_df)
                                player_samples_count += 1
                                
                        except Exception as e:
                            game_pbar.write(f"     ⚠️  Error creating features for player {player_id} on {target_date}: {e}")
                            continue
                
                # Update main progress bar with summary
                pbar.write(f"   ✅ Player {player_id}: {player_samples_count} training samples created")
        
        conn.close()
        
        if all_features:
            final_dataset = pd.concat(all_features, ignore_index=True)
            tqdm.write(f"🎯 Final dataset: {len(final_dataset)} samples with {len(final_dataset.columns)} features")
            logger.info(f"Created training dataset with {len(final_dataset)} samples and {len(final_dataset.columns)} features")
            return final_dataset
        else:
            tqdm.write("❌ No training data created")
            logger.warning("No training data created")
            return pd.DataFrame()
    
    def prepare_features_for_prediction(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model prediction with advanced preprocessing."""
        if features_df.empty:
            return features_df
        
        # Make a copy to avoid modifying the original
        prepared_df = features_df.copy()
        
        # Handle missing values with more sophisticated imputation
        numeric_columns = prepared_df.select_dtypes(include=[np.number]).columns
        
        # Exclude metadata columns from processing
        exclude_columns = ['player_id', 'target_date', 'game_id', 'game_date', 'opponent_team_id']
        exclude_columns.extend([col for col in prepared_df.columns if col.startswith('target_')])
        
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
                prepared_df[col] = prepared_df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return prepared_df


# For backward compatibility, create an alias
FeatureEngineer = AdvancedFeatureEngineer 