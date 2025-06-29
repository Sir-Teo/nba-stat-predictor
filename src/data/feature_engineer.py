"""
Feature Engineering Module - Creates features for NBA stat prediction models.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sqlite3
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates and processes features for NBA stat prediction."""
    
    def __init__(self, db_path: str = "data/nba_data.db"):
        """Initialize the feature engineer."""
        self.db_path = db_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def create_features_for_player(self, player_id: int, target_date: str, 
                                  lookback_games: int = 15) -> pd.DataFrame:
        """Create comprehensive features for a player for prediction."""
        conn = sqlite3.connect(self.db_path)
        
        # Get player's recent games
        query = """
            SELECT * FROM player_games 
            WHERE player_id = ? AND game_date < ?
            ORDER BY game_date DESC
            LIMIT ?
        """
        
        df = pd.read_sql_query(query, conn, params=(player_id, target_date, lookback_games))
        conn.close()
        
        if df.empty:
            return pd.DataFrame()
        
        features = {}
        
        # Basic rolling averages and trends
        features.update(self._create_rolling_stats(df))
        
        # Recent form features
        features.update(self._create_recent_form_features(df))
        
        # Advanced momentum and form features
        features.update(self._create_advanced_form_features(df))
        
        # Home/Away splits
        features.update(self._create_home_away_features(df))
        
        # Rest and fatigue features (enhanced)
        features.update(self._create_rest_features(df))
        features.update(self._create_fatigue_features(df))
        
        # Consistency features
        features.update(self._create_consistency_features(df))
        
        # Matchup and opponent features
        features.update(self._create_matchup_features(df))
        
        # Opponent-based features (if available)
        features.update(self._create_opponent_features(df))
        
        # Convert to DataFrame
        features_df = pd.DataFrame([features])
        features_df['player_id'] = player_id
        features_df['target_date'] = target_date
        
        return features_df
    
    def _create_rolling_stats(self, df: pd.DataFrame, windows: List[int] = [3, 5, 10, 15]) -> Dict:
        """Create rolling statistics for different time windows."""
        features = {}
        
        stat_columns = ['pts', 'reb', 'ast', 'stl', 'blk', 'tov', 'fg_pct', 'fg3_pct', 'ft_pct', 'min']
        
        # Ensure data is sorted by date
        df_sorted = df.sort_values('game_date').reset_index(drop=True)
        
        for window in windows:
            for stat in stat_columns:
                if stat in df_sorted.columns:
                    # Rolling mean
                    rolling_mean = df_sorted[stat].rolling(window=window, min_periods=1).mean().iloc[-1]
                    features[f'{stat}_avg_{window}g'] = rolling_mean
                    
                    # Rolling standard deviation
                    rolling_std = df_sorted[stat].rolling(window=window, min_periods=1).std().iloc[-1]
                    features[f'{stat}_std_{window}g'] = rolling_std if not pd.isna(rolling_std) else 0
                    
                    # Rolling trend (slope)
                    if len(df_sorted) >= window:
                        recent_values = df_sorted[stat].tail(window).values
                        trend = self._calculate_trend(recent_values)
                        features[f'{stat}_trend_{window}g'] = trend
                    
                    # Rolling min/max
                    if window >= 5:  # Only for larger windows
                        rolling_min = df_sorted[stat].rolling(window=window, min_periods=1).min().iloc[-1]
                        rolling_max = df_sorted[stat].rolling(window=window, min_periods=1).max().iloc[-1]
                        features[f'{stat}_min_{window}g'] = rolling_min
                        features[f'{stat}_max_{window}g'] = rolling_max
                        
                        # Rolling range
                        features[f'{stat}_range_{window}g'] = rolling_max - rolling_min
        
        return features
    
    def _create_recent_form_features(self, df: pd.DataFrame) -> Dict:
        """Create features based on recent performance trends."""
        features = {}
        
        if len(df) < 2:
            return features
            
        # Sort by date
        df_sorted = df.sort_values('game_date').reset_index(drop=True)
        
        # Last game performance
        last_game = df_sorted.iloc[-1]
        features['last_game_pts'] = last_game.get('pts', 0)
        features['last_game_reb'] = last_game.get('reb', 0)
        features['last_game_ast'] = last_game.get('ast', 0)
        features['last_game_min'] = last_game.get('min', 0)
        
        # Performance vs season average
        season_avg_pts = df_sorted['pts'].mean()
        features['last_game_pts_vs_avg'] = last_game.get('pts', 0) - season_avg_pts
        
        # Streak features
        features.update(self._calculate_streaks(df_sorted))
        
        # Recent high/low performance
        if len(df_sorted) >= 5:
            recent_5 = df_sorted.tail(5)
            features['recent_5_max_pts'] = recent_5['pts'].max()
            features['recent_5_min_pts'] = recent_5['pts'].min()
            features['recent_5_games_above_avg'] = (recent_5['pts'] > season_avg_pts).sum()
        
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
    
    def _create_opponent_features(self, df: pd.DataFrame) -> Dict:
        """Create features based on opponent analysis."""
        features = {}
        
        # This is a placeholder for opponent-based features
        # In a full implementation, you would analyze opponent defensive ratings,
        # pace, etc.
        
        features['placeholder_opp_feature'] = 0
        
        return features
    
    def _calculate_streaks(self, df: pd.DataFrame) -> Dict:
        """Calculate performance streaks."""
        features = {}
        
        if len(df) < 2:
            return features
        
        # Points scoring streaks
        pts_values = df['pts'].values
        season_avg = np.mean(pts_values)
        
        # Current streak above/below average
        current_streak = 0
        for i in range(len(pts_values) - 1, -1, -1):
            if pts_values[i] > season_avg:
                current_streak += 1
            else:
                break
        
        features['current_above_avg_streak'] = current_streak
        
        # Similar for other stats
        if 'reb' in df.columns:
            reb_avg = df['reb'].mean()
            features['current_reb_trend'] = 1 if df['reb'].iloc[-1] > reb_avg else 0
        
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
    
    def create_training_dataset(self, players_list: List[int], 
                               start_date: str, end_date: str,
                               target_stats: List[str] = ['pts', 'reb', 'ast']) -> pd.DataFrame:
        """Create a complete training dataset with features and targets."""
        all_features = []
        
        conn = sqlite3.connect(self.db_path)
        
        for player_id in players_list:
            # Get all games for this player in the date range
            query = """
                SELECT * FROM player_games 
                WHERE player_id = ? AND game_date BETWEEN ? AND ?
                ORDER BY game_date
            """
            
            player_games = pd.read_sql_query(query, conn, params=(player_id, start_date, end_date))
            
            if len(player_games) < 10:  # Need sufficient history
                continue
            
            # For each game, create features based on previous games
            for i in range(10, len(player_games)):  # Start after 10 games for feature history
                target_game = player_games.iloc[i]
                target_date = target_game['game_date']
                
                # Create features
                features_df = self.create_features_for_player(player_id, target_date)
                
                if not features_df.empty:
                    # Add target values
                    for stat in target_stats:
                        if stat in target_game:
                            features_df[f'target_{stat}'] = target_game[stat]
                    
                    # Add game metadata
                    features_df['game_id'] = target_game.get('game_id', '')
                    features_df['game_date'] = target_date
                    
                    all_features.append(features_df)
        
        conn.close()
        
        if all_features:
            final_dataset = pd.concat(all_features, ignore_index=True)
            return final_dataset
        else:
            return pd.DataFrame()
    
    def prepare_features_for_prediction(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model prediction (scaling, encoding, etc.)."""
        if features_df.empty:
            return features_df
        
        # Make a copy to avoid modifying the original
        prepared_df = features_df.copy()
        
        # Handle missing values
        numeric_columns = prepared_df.select_dtypes(include=[np.number]).columns
        prepared_df[numeric_columns] = prepared_df[numeric_columns].fillna(0)
        
        # Scale numerical features (excluding IDs and dates)
        exclude_columns = ['player_id', 'target_date', 'game_id', 'game_date']
        scale_columns = [col for col in numeric_columns if col not in exclude_columns and not col.startswith('target_')]
        
        if scale_columns:
            prepared_df[scale_columns] = self.scaler.fit_transform(prepared_df[scale_columns])
        
        return prepared_df

    def _create_advanced_form_features(self, df: pd.DataFrame) -> Dict:
        """Create advanced recent form and momentum features."""
        features = {}
        
        if len(df) < 3:
            return features
            
        # Sort by date
        df_sorted = df.sort_values('game_date').reset_index(drop=True)
        
        # Recent momentum features
        stat_columns = ['pts', 'reb', 'ast']
        
        for stat in stat_columns:
            if stat in df_sorted.columns:
                values = df_sorted[stat].values
                
                # Last 3 games momentum
                if len(values) >= 3:
                    last_3 = values[-3:]
                    features[f'{stat}_momentum_3g'] = self._calculate_momentum(last_3)
                
                # Last 5 games momentum
                if len(values) >= 5:
                    last_5 = values[-5:]
                    features[f'{stat}_momentum_5g'] = self._calculate_momentum(last_5)
                
                # Performance vs season percentiles
                season_values = values
                if len(season_values) >= 10:
                    last_game = values[-1]
                    percentile = (season_values < last_game).mean() * 100
                    features[f'{stat}_percentile'] = percentile
                
                # Hot/cold streaks
                season_avg = np.mean(values)
                current_streak = self._calculate_current_streak(values, season_avg)
                features[f'{stat}_streak'] = current_streak
                
                # Volatility (coefficient of variation)
                if len(values) >= 5 and season_avg > 0:
                    cv = np.std(values) / season_avg
                    features[f'{stat}_volatility'] = cv
        
        return features
    
    def _calculate_momentum(self, values: np.ndarray) -> float:
        """Calculate momentum as weighted recent performance."""
        if len(values) < 2:
            return 0.0
        
        # Weight recent games more heavily
        weights = np.linspace(0.5, 1.0, len(values))
        weighted_trend = self._calculate_trend(values * weights)
        
        return weighted_trend
    
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
    
    def _create_matchup_features(self, df: pd.DataFrame) -> Dict:
        """Create features based on opponent and matchup context."""
        features = {}
        
        if 'matchup' not in df.columns or len(df) == 0:
            return features
        
        # Extract opponent information
        opponent_teams = []
        for matchup in df['matchup']:
            if isinstance(matchup, str):
                if 'vs.' in matchup:
                    opponent = matchup.split('vs. ')[-1]
                elif '@' in matchup:
                    opponent = matchup.split('@ ')[-1]
                else:
                    opponent = 'UNKNOWN'
                opponent_teams.append(opponent)
        
        if opponent_teams:
            # Most common opponents
            from collections import Counter
            opponent_counts = Counter(opponent_teams)
            features['most_common_opponent_games'] = max(opponent_counts.values()) if opponent_counts else 0
            features['unique_opponents'] = len(opponent_counts)
            
            # Performance against different opponents (if enough data)
            if len(df) >= 10:
                df_with_opponents = df.copy()
                df_with_opponents['opponent'] = opponent_teams[:len(df)]
                
                if 'pts' in df.columns:
                    avg_pts_by_opp = df_with_opponents.groupby('opponent')['pts'].mean()
                    if len(avg_pts_by_opp) > 1:
                        features['pts_opponent_variance'] = avg_pts_by_opp.std()
        
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