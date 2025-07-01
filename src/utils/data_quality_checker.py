"""
Data Quality Checker - Identifies and fixes data quality issues in the NBA database.
"""

import logging
import sqlite3
from typing import Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class DataQualityChecker:
    """Checks and fixes data quality issues in the NBA database."""

    def __init__(self, db_path: str = "data/nba_data.db"):
        """Initialize the data quality checker."""
        self.db_path = db_path

    def run_comprehensive_check(self) -> Dict:
        """Run comprehensive data quality checks."""
        print("🔍 Running Comprehensive Data Quality Check...")
        print("=" * 60)

        results = {
            "duplicates": self.check_duplicates(),
            "impossible_values": self.check_impossible_values(),
            "missing_data": self.check_missing_data(),
            "consistency": self.check_data_consistency(),
            "outliers": self.detect_outliers(),
            "recommendations": []
        }

        # Generate recommendations
        if results["duplicates"]["count"] > 0:
            results["recommendations"].append("Remove duplicate game entries")
        
        if results["impossible_values"]["count"] > 0:
            results["recommendations"].append("Fix or remove impossible statistical values")
            
        if results["missing_data"]["critical_missing"] > 0:
            results["recommendations"].append("Fill or remove games with missing critical stats")

        if results["outliers"]["extreme_count"] > 0:
            results["recommendations"].append("Review extreme statistical outliers")

        self._display_summary(results)
        return results

    def check_duplicates(self) -> Dict:
        """Check for duplicate game entries."""
        print("\n1. 🔍 Checking for duplicate games...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Check for exact duplicates
        duplicate_query = """
            SELECT game_id, player_id, COUNT(*) as count
            FROM player_games
            GROUP BY game_id, player_id
            HAVING count > 1
        """
        
        duplicates = pd.read_sql_query(duplicate_query, conn)
        conn.close()
        
        print(f"   Found {len(duplicates)} duplicate game entries")
        
        return {
            "count": len(duplicates),
            "duplicates": duplicates.to_dict('records') if len(duplicates) > 0 else []
        }

    def check_impossible_values(self) -> Dict:
        """Check for statistically impossible values."""
        print("\n2. 🚫 Checking for impossible statistical values...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Define impossible value ranges
        impossible_conditions = [
            "pts > 100 OR pts < 0",
            "reb > 50 OR reb < 0", 
            "ast > 30 OR ast < 0",
            "stl > 15 OR stl < 0",
            "blk > 20 OR blk < 0",
            "tov > 15 OR tov < 0",
            "fg_pct > 1.0 OR fg_pct < 0",
            "fgm > fga",  # Made shots can't exceed attempts
            "fg3m > fg3a",
            "ftm > fta",
            "min > 60 OR min < 0"  # Minutes can't exceed 60 (including OT)
        ]
        
        impossible_games = []
        for condition in impossible_conditions:
            query = f"""
                SELECT game_id, player_id, game_date, pts, reb, ast, stl, blk, 
                       fgm, fga, fg_pct, min, '{condition}' as violation
                FROM player_games
                WHERE {condition}
                LIMIT 50
            """
            
            violations = pd.read_sql_query(query, conn)
            if len(violations) > 0:
                impossible_games.extend(violations.to_dict('records'))
        
        conn.close()
        
        print(f"   Found {len(impossible_games)} games with impossible values")
        
        return {
            "count": len(impossible_games),
            "violations": impossible_games[:20]  # Limit to first 20 for display
        }

    def check_missing_data(self) -> Dict:
        """Check for missing critical data."""
        print("\n3. ❓ Checking for missing data...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Check for missing critical stats
        missing_query = """
            SELECT 
                COUNT(*) as total_games,
                SUM(CASE WHEN pts IS NULL THEN 1 ELSE 0 END) as missing_pts,
                SUM(CASE WHEN reb IS NULL THEN 1 ELSE 0 END) as missing_reb,
                SUM(CASE WHEN ast IS NULL THEN 1 ELSE 0 END) as missing_ast,
                SUM(CASE WHEN game_date IS NULL OR game_date = '' THEN 1 ELSE 0 END) as missing_date,
                SUM(CASE WHEN player_id IS NULL THEN 1 ELSE 0 END) as missing_player_id
            FROM player_games
        """
        
        missing_stats = pd.read_sql_query(missing_query, conn)
        
        # Games with critical missing data
        critical_missing_query = """
            SELECT COUNT(*) as critical_missing
            FROM player_games
            WHERE pts IS NULL OR reb IS NULL OR ast IS NULL 
               OR game_date IS NULL OR game_date = ''
               OR player_id IS NULL
        """
        
        critical_missing = pd.read_sql_query(critical_missing_query, conn)
        conn.close()
        
        missing_info = missing_stats.iloc[0].to_dict()
        missing_info["critical_missing"] = critical_missing.iloc[0]["critical_missing"]
        
        print(f"   Total games: {missing_info['total_games']}")
        print(f"   Games with missing critical data: {missing_info['critical_missing']}")
        
        return missing_info

    def check_data_consistency(self) -> Dict:
        """Check for data consistency issues."""
        print("\n4. 🔗 Checking data consistency...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Check shooting percentage consistency
        consistency_query = """
            SELECT COUNT(*) as inconsistent_fg_pct
            FROM player_games
            WHERE fga > 0 AND fgm > 0 
              AND ABS(CAST(fgm AS FLOAT) / fga - fg_pct) > 0.01
        """
        
        consistency_results = pd.read_sql_query(consistency_query, conn)
        
        # Check for games with 0 minutes but stats
        zero_min_query = """
            SELECT COUNT(*) as zero_min_with_stats
            FROM player_games
            WHERE (min = 0 OR min IS NULL) 
              AND (pts > 0 OR reb > 0 OR ast > 0)
        """
        
        zero_min_results = pd.read_sql_query(zero_min_query, conn)
        conn.close()
        
        consistency_info = {
            "inconsistent_fg_pct": consistency_results.iloc[0]["inconsistent_fg_pct"],
            "zero_min_with_stats": zero_min_results.iloc[0]["zero_min_with_stats"]
        }
        
        print(f"   Inconsistent FG%: {consistency_info['inconsistent_fg_pct']}")
        print(f"   Games with 0 minutes but stats: {consistency_info['zero_min_with_stats']}")
        
        return consistency_info

    def detect_outliers(self) -> Dict:
        """Detect extreme statistical outliers."""
        print("\n5. 📊 Detecting extreme outliers...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Define outlier thresholds (99.9th percentile)
        outlier_query = """
            SELECT 
                COUNT(CASE WHEN pts > 60 THEN 1 END) as extreme_scoring,
                COUNT(CASE WHEN reb > 25 THEN 1 END) as extreme_rebounding,
                COUNT(CASE WHEN ast > 20 THEN 1 END) as extreme_assists,
                COUNT(CASE WHEN stl > 8 THEN 1 END) as extreme_steals,
                COUNT(CASE WHEN blk > 10 THEN 1 END) as extreme_blocks
            FROM player_games
        """
        
        outliers = pd.read_sql_query(outlier_query, conn)
        conn.close()
        
        outlier_info = outliers.iloc[0].to_dict()
        outlier_info["extreme_count"] = sum(outlier_info.values())
        
        print(f"   Extreme performances found: {outlier_info['extreme_count']}")
        
        return outlier_info

    def fix_data_issues(self, auto_fix: bool = False) -> Dict:
        """Fix identified data quality issues."""
        print("\n🔧 Fixing Data Quality Issues...")
        
        if not auto_fix:
            print("Manual fix mode - recommendations only")
            return {"message": "Manual mode - use auto_fix=True to apply fixes"}
        
        fixes_applied = {
            "duplicates_removed": 0,
            "impossible_values_fixed": 0,
            "missing_data_filled": 0
        }
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Remove exact duplicates
        cursor.execute("""
            DELETE FROM player_games
            WHERE rowid NOT IN (
                SELECT MIN(rowid)
                FROM player_games
                GROUP BY game_id, player_id
            )
        """)
        fixes_applied["duplicates_removed"] = cursor.rowcount
        
        # Fix impossible percentage values (> 1.0)
        cursor.execute("""
            UPDATE player_games 
            SET fg_pct = CAST(fgm AS FLOAT) / fga
            WHERE fga > 0 AND fg_pct > 1.0
        """)
        fixes_applied["impossible_values_fixed"] += cursor.rowcount
        
        cursor.execute("""
            UPDATE player_games 
            SET fg3_pct = CAST(fg3m AS FLOAT) / fg3a
            WHERE fg3a > 0 AND fg3_pct > 1.0
        """)
        fixes_applied["impossible_values_fixed"] += cursor.rowcount
        
        conn.commit()
        conn.close()
        
        print(f"✅ Applied {sum(fixes_applied.values())} fixes")
        return fixes_applied

    def _display_summary(self, results: Dict):
        """Display summary of data quality check results."""
        print("\n" + "=" * 60)
        print("📋 DATA QUALITY SUMMARY")
        print("=" * 60)
        
        total_issues = (
            results["duplicates"]["count"] +
            results["impossible_values"]["count"] +
            results["missing_data"]["critical_missing"] +
            results["outliers"]["extreme_count"]
        )
        
        if total_issues == 0:
            print("✅ No significant data quality issues found!")
        else:
            print(f"⚠️  Found {total_issues} data quality issues:")
            
            if results["duplicates"]["count"] > 0:
                print(f"   • {results['duplicates']['count']} duplicate entries")
                
            if results["impossible_values"]["count"] > 0:
                print(f"   • {results['impossible_values']['count']} impossible values")
                
            if results["missing_data"]["critical_missing"] > 0:
                print(f"   • {results['missing_data']['critical_missing']} games with critical missing data")
                
            if results["outliers"]["extreme_count"] > 0:
                print(f"   • {results['outliers']['extreme_count']} extreme outliers")
        
        if results["recommendations"]:
            print("\n💡 Recommendations:")
            for i, rec in enumerate(results["recommendations"], 1):
                print(f"   {i}. {rec}")
        
        print("=" * 60) 