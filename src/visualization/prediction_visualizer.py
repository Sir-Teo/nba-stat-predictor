"""
Prediction Visualization Module - Creates visual rationales for NBA stat predictions.
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class PredictionVisualizer:
    """Creates visualizations explaining prediction rationales."""

    def __init__(self, db_path: str = "data/nba_data.db"):
        """Initialize the prediction visualizer."""
        self.db_path = db_path
        self.setup_plotting_style()

    def setup_plotting_style(self):
        """Set up consistent plotting style."""
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams.update({
            'figure.figsize': (15, 10),
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11,
            'figure.titlesize': 16
        })

    def create_prediction_rationale_chart(
        self,
        player_id: int,
        player_name: str,
        predictions_df: pd.DataFrame,
        features_df: pd.DataFrame,
        recent_stats: Dict,
        opponent_name: str = "",
        save_path: Optional[str] = None
    ) -> str:
        """Create comprehensive rationale visualization for predictions."""
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # Main title
        player_age = features_df.get("player_age", pd.Series([30])).iloc[0] if not features_df.empty else 30
        fig.suptitle(
            f"🏀 Prediction Rationale: {player_name} vs {opponent_name}\n"
            f"Age: {player_age:.1f} years | Age-Adjusted Predictions",
            fontsize=18, fontweight='bold', y=0.95
        )

        # 1. Recent Performance Trend (top left)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_recent_performance_trend(ax1, player_id, player_name)

        # 2. Prediction Breakdown (top right)
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_prediction_breakdown(ax2, predictions_df, recent_stats, features_df)

        # 3. Age Impact Analysis (middle left)
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_age_impact_analysis(ax3, player_id, player_name, features_df)

        # 4. Confidence Factors (middle right)
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_confidence_factors(ax4, predictions_df, features_df)

        # 5. Statistical Context (bottom)
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_statistical_context(ax5, player_id, player_name, predictions_df, recent_stats)

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            return save_path
        else:
            # Save to default location
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_path = f"plots/prediction_rationale_{player_name.replace(' ', '_')}_{timestamp}.png"
            plt.savefig(default_path, dpi=300, bbox_inches='tight')
            return default_path

    def _plot_recent_performance_trend(self, ax, player_id: int, player_name: str):
        """Plot recent performance trends."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get last 20 games
            query = """
                SELECT game_date, pts, reb, ast, stl, blk 
                FROM player_games 
                WHERE player_id = ?
                ORDER BY game_date DESC
                LIMIT 20
            """
            
            df = pd.read_sql_query(query, conn, params=(player_id,))
            conn.close()
            
            if df.empty:
                ax.text(0.5, 0.5, 'No recent data available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title("Recent Performance Trend")
                return
            
            # Reverse to chronological order
            df = df.iloc[::-1].reset_index(drop=True)
            
            # Plot trends for key stats
            x = range(len(df))
            
            ax.plot(x, df['pts'], 'o-', label='Points', linewidth=2, markersize=6)
            ax.plot(x, df['reb'], 's-', label='Rebounds', linewidth=2, markersize=6)
            ax.plot(x, df['ast'], '^-', label='Assists', linewidth=2, markersize=6)
            
            # Add trend lines
            if len(df) > 3:
                z_pts = np.polyfit(x, df['pts'], 1)
                p_pts = np.poly1d(z_pts)
                ax.plot(x, p_pts(x), '--', alpha=0.7, color='red', label=f'PTS Trend ({z_pts[0]:+.1f}/game)')
            
            ax.set_title("📈 Recent Performance Trend (Last 20 Games)", fontweight='bold')
            ax.set_xlabel("Games Ago (Most Recent →)")
            ax.set_ylabel("Stats Per Game")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Highlight recent form (last 5 games)
            if len(df) >= 5:
                ax.axvspan(len(df)-5, len(df)-1, alpha=0.2, color='yellow', 
                          label='Recent Form (5 games)')
                
        except Exception as e:
            logger.error(f"Error plotting recent performance: {e}")
            ax.text(0.5, 0.5, f'Error loading data: {e}', 
                   ha='center', va='center', transform=ax.transAxes)

    def _plot_prediction_breakdown(self, ax, predictions_df: pd.DataFrame, 
                                 recent_stats: Dict, features_df: pd.DataFrame):
        """Plot breakdown of how predictions were calculated."""
        try:
            stats = ['pts', 'reb', 'ast', 'stl', 'blk']
            stat_labels = ['Points', 'Rebounds', 'Assists', 'Steals', 'Blocks']
            
            predictions = []
            recent_avgs = []
            
            for stat in stats:
                pred_col = f"predicted_{stat}"
                if pred_col in predictions_df.columns:
                    predictions.append(predictions_df[pred_col].iloc[0])
                    recent_avgs.append(recent_stats.get(f"{stat}_avg", 0))
                else:
                    predictions.append(0)
                    recent_avgs.append(0)
            
            x = np.arange(len(stats))
            width = 0.35
            
            # Plot bars
            bars1 = ax.bar(x - width/2, recent_avgs, width, label='Recent Form (10 games)', 
                          alpha=0.8, color='skyblue')
            bars2 = ax.bar(x + width/2, predictions, width, label='Age-Adjusted Prediction', 
                          alpha=0.8, color='orange')
            
            ax.set_title("🎯 Prediction vs Recent Form", fontweight='bold')
            ax.set_xlabel("Statistics")
            ax.set_ylabel("Values")
            ax.set_xticks(x)
            ax.set_xticklabels(stat_labels)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar1, bar2 in zip(bars1, bars2):
                if bar1.get_height() > 0:
                    ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.1,
                           f'{bar1.get_height():.1f}', ha='center', va='bottom', fontsize=9)
                if bar2.get_height() > 0:
                    ax.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.1,
                           f'{bar2.get_height():.1f}', ha='center', va='bottom', fontsize=9)
            
            # Show age adjustment info
            player_age = features_df.get("player_age", pd.Series([30])).iloc[0] if not features_df.empty else 30
            if player_age >= 35:
                age_weight = 0.8 if player_age >= 40 else 0.6
                ax.text(0.02, 0.98, f"Age {player_age:.1f}: {int(age_weight*100)}% recent form weight", 
                       transform=ax.transAxes, va='top', ha='left',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
                       
        except Exception as e:
            logger.error(f"Error plotting prediction breakdown: {e}")

    def _plot_age_impact_analysis(self, ax, player_id: int, player_name: str, features_df: pd.DataFrame):
        """Plot age impact on performance over time."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get career performance by season/year
            query = """
                SELECT 
                    strftime('%Y', game_date) as year,
                    AVG(pts) as avg_pts,
                    AVG(reb) as avg_reb,
                    AVG(ast) as avg_ast,
                    COUNT(*) as games
                FROM player_games 
                WHERE player_id = ?
                GROUP BY strftime('%Y', game_date)
                HAVING games >= 10
                ORDER BY year
            """
            
            df = pd.read_sql_query(query, conn, params=(player_id,))
            conn.close()
            
            if len(df) < 2:
                ax.text(0.5, 0.5, 'Insufficient career data', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title("Career Performance Trend")
                return
            
            years = df['year'].astype(int)
            
            # Calculate estimated age for each year
            player_age = features_df.get("player_age", pd.Series([30])).iloc[0] if not features_df.empty else 30
            current_year = datetime.now().year
            ages = years - current_year + player_age
            
            # Plot career trends
            ax.plot(ages, df['avg_pts'], 'o-', label='Points', linewidth=2, markersize=6)
            ax.plot(ages, df['avg_reb'], 's-', label='Rebounds', linewidth=2, markersize=6)
            ax.plot(ages, df['avg_ast'], '^-', label='Assists', linewidth=2, markersize=6)
            
            # Highlight current age
            current_age = player_age
            ax.axvline(x=current_age, color='red', linestyle='--', alpha=0.7, 
                      label=f'Current Age ({current_age:.1f})')
            
            # Show age decline zones
            if current_age >= 35:
                ax.axvspan(35, 50, alpha=0.1, color='orange', label='Veteran Zone (35+)')
            if current_age >= 40:
                ax.axvspan(40, 50, alpha=0.1, color='red', label='Elite Longevity (40+)')
            
            ax.set_title("🎂 Career Performance by Age", fontweight='bold')
            ax.set_xlabel("Player Age")
            ax.set_ylabel("Season Averages")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            logger.error(f"Error plotting age impact: {e}")

    def _plot_confidence_factors(self, ax, predictions_df: pd.DataFrame, features_df: pd.DataFrame):
        """Plot factors affecting prediction confidence."""
        try:
            # Get confidence scores
            stats = ['pts', 'reb', 'ast', 'stl', 'blk']
            stat_labels = ['Points', 'Rebounds', 'Assists', 'Steals', 'Blocks']
            confidences = []
            
            for stat in stats:
                conf_col = f"confidence_{stat}"
                if conf_col in predictions_df.columns:
                    confidences.append(predictions_df[conf_col].iloc[0] * 100)
                else:
                    confidences.append(50)  # Default
            
            # Create confidence bar chart
            colors = ['green' if c >= 70 else 'orange' if c >= 50 else 'red' for c in confidences]
            bars = ax.barh(stat_labels, confidences, color=colors, alpha=0.7)
            
            # Add confidence level lines
            ax.axvline(x=70, color='green', linestyle='--', alpha=0.5, label='High Confidence (70%+)')
            ax.axvline(x=50, color='orange', linestyle='--', alpha=0.5, label='Medium Confidence (50%+)')
            ax.axvline(x=30, color='red', linestyle='--', alpha=0.5, label='Low Confidence (<50%)')
            
            # Add value labels
            for i, (bar, conf) in enumerate(zip(bars, confidences)):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                       f'{conf:.0f}%', va='center', fontsize=10)
            
            ax.set_title("🎯 Prediction Confidence Levels", fontweight='bold')
            ax.set_xlabel("Confidence (%)")
            ax.set_xlim(0, 100)
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add confidence explanation
            player_age = features_df.get("player_age", pd.Series([30])).iloc[0] if not features_df.empty else 30
            if player_age >= 35:
                explanation = f"Lower confidence due to age {player_age:.1f} (higher variance)"
                ax.text(0.02, 0.98, explanation, transform=ax.transAxes, va='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
                       
        except Exception as e:
            logger.error(f"Error plotting confidence factors: {e}")

    def _plot_statistical_context(self, ax, player_id: int, player_name: str, 
                                predictions_df: pd.DataFrame, recent_stats: Dict):
        """Plot statistical context and key insights."""
        try:
            # Create text-based insights
            insights = []
            
            # Age insights
            player_age = 30  # Default
            for col in predictions_df.columns:
                if 'player_age' in str(col).lower():
                    player_age = predictions_df[col].iloc[0] if not predictions_df[col].empty else 30
                    break
            
            if player_age >= 40:
                insights.append(f"🔸 At {player_age:.1f} years old, {player_name} is in elite longevity territory")
                insights.append("🔸 Predictions heavily favor recent form over career averages")
                insights.append("🔸 Performance may be more variable game-to-game")
            elif player_age >= 35:
                insights.append(f"🔹 At {player_age:.1f} years old, {player_name} is a veteran player")
                insights.append("🔹 Age-related adjustments applied to account for typical decline")
                insights.append("🔹 Recent performance weighted more heavily")
            
            # Performance insights
            recent_pts = recent_stats.get('pts_avg', 0)
            pred_pts = predictions_df.get('predicted_pts', pd.Series([0])).iloc[0] if 'predicted_pts' in predictions_df.columns else 0
            
            if abs(recent_pts - pred_pts) < 2:
                insights.append(f"✅ Prediction closely matches recent form ({recent_pts:.1f} avg)")
            elif pred_pts < recent_pts:
                insights.append(f"📉 Prediction below recent average (regression expected)")
            else:
                insights.append(f"📈 Prediction above recent average (positive outlook)")
            
            # Confidence insights
            avg_confidence = 0
            conf_count = 0
            for stat in ['pts', 'reb', 'ast', 'stl', 'blk']:
                conf_col = f"confidence_{stat}"
                if conf_col in predictions_df.columns:
                    avg_confidence += predictions_df[conf_col].iloc[0]
                    conf_count += 1
            
            if conf_count > 0:
                avg_confidence = (avg_confidence / conf_count) * 100
                if avg_confidence >= 70:
                    insights.append(f"🟢 High confidence predictions ({avg_confidence:.0f}% average)")
                elif avg_confidence >= 50:
                    insights.append(f"🟡 Moderate confidence predictions ({avg_confidence:.0f}% average)")
                else:
                    insights.append(f"🔴 Lower confidence due to aging player variability ({avg_confidence:.0f}% average)")
            
            # Display insights
            ax.axis('off')
            
            # Title
            ax.text(0.5, 0.95, "📊 Key Prediction Insights", 
                   ha='center', va='top', fontsize=14, fontweight='bold',
                   transform=ax.transAxes)
            
            # Insights
            y_pos = 0.85
            for insight in insights[:6]:  # Limit to 6 insights
                ax.text(0.05, y_pos, insight, 
                       ha='left', va='top', fontsize=11,
                       transform=ax.transAxes)
                y_pos -= 0.12
                
            # Add methodology note
            methodology = ("🔬 Methodology: Age-adjusted predictions combine recent performance "
                          "with historical data, applying decline curves for aging players.")
            ax.text(0.05, 0.15, methodology, 
                   ha='left', va='top', fontsize=10, style='italic',
                   transform=ax.transAxes, wrap=True)
                   
        except Exception as e:
            logger.error(f"Error plotting statistical context: {e}")

    def show_prediction_rationale(
        self,
        player_id: int,
        player_name: str,
        predictions_df: pd.DataFrame,
        features_df: pd.DataFrame,
        recent_stats: Dict,
        opponent_name: str = ""
    ):
        """Show prediction rationale visualization."""
        try:
            # Create the chart
            chart_path = self.create_prediction_rationale_chart(
                player_id, player_name, predictions_df, features_df, 
                recent_stats, opponent_name
            )
            
            # Display the chart
            plt.show()
            
            print(f"\n📊 Visualization saved to: {chart_path}")
            print("\n🔍 Prediction Rationale Summary:")
            print("=" * 50)
            
            # Print key insights
            player_age = features_df.get("player_age", pd.Series([30])).iloc[0] if not features_df.empty else 30
            
            if player_age >= 40:
                print(f"🔸 Age Factor: {player_age:.1f} years (Elite longevity territory)")
                print("🔸 Weighting: 80% recent form, 20% historical averages")
                print("🔸 Confidence: Reduced due to higher variance in aging players")
            elif player_age >= 35:
                print(f"🔹 Age Factor: {player_age:.1f} years (Veteran adjustments applied)")
                print("🔹 Weighting: 60% recent form, 40% historical averages")
                print("🔹 Confidence: Moderate reduction for age-related uncertainty")
            else:
                print(f"✅ Age Factor: {player_age:.1f} years (Prime/Standard predictions)")
                print("✅ Weighting: Standard model predictions")
                
            return chart_path
            
        except Exception as e:
            logger.error(f"Error showing prediction rationale: {e}")
            print(f"❌ Error creating visualization: {e}")
            return None
