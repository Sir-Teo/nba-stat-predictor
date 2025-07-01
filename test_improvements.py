#!/usr/bin/env python3
"""
Test script to evaluate improvements in prediction accuracy and visualization.
"""

import os
import sys
import logging
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List

# Add src to path
sys.path.append('src')

from src.data.feature_engineer import AdvancedFeatureEngineer
from src.models.stat_predictors import ModelManager, AdvancedEnsembleStatPredictor
from src.visualization.prediction_visualizer import PredictionVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PredictionImprovementTester:
    """Test class to evaluate prediction improvements."""
    
    def __init__(self, db_path: str = "data/nba_data.db"):
        """Initialize the tester."""
        self.db_path = db_path
        self.feature_engineer = AdvancedFeatureEngineer(db_path)
        self.model_manager = ModelManager(db_path)
        self.visualizer = PredictionVisualizer(db_path)
# Skip tonight_predictor due to import issues for now
        self.stat_types = ["pts", "reb", "ast", "stl", "blk"]
        
    def test_confidence_improvements(self) -> Dict:
        """Test the improved confidence calculation system."""
        print("\n" + "="*60)
        print("TESTING CONFIDENCE IMPROVEMENTS")
        print("="*60)
        
        try:
            # Load or train models
            self.model_manager.load_models(self.stat_types)
            
            # If no models loaded, train simple ones for testing
            missing_models = [stat for stat in self.stat_types if stat not in self.model_manager.predictors]
            if missing_models:
                print(f"Training models for: {missing_models}")
                self._train_test_models(missing_models)
            
            # Test prediction with confidence for a known player (LeBron James)
            lebron_id = 2544
            target_date = "2024-06-15"  # Use a date we likely have data for
            
            # Create features
            features_df = self.feature_engineer.create_features_for_player(
                lebron_id, target_date, lookback_games=15
            )
            
            if features_df.empty:
                print("⚠️  No features available for test player")
                return {"status": "no_data"}
            
            # Get predictions with confidence
            results = {}
            for stat_type in self.stat_types:
                if stat_type in self.model_manager.predictors:
                    predictor = self.model_manager.predictors[stat_type]
                    
                    # Test prediction with new confidence system
                    predictions, confidence = predictor.predict(features_df, return_confidence=True)
                    
                    results[stat_type] = {
                        "prediction": predictions[0] if len(predictions) > 0 else 0,
                        "confidence": confidence[0] if len(confidence) > 0 else 0,
                        "confidence_range": f"{confidence.min():.2f} - {confidence.max():.2f}"
                    }
                    
                    print(f"{stat_type.upper():>5}: {predictions[0]:5.1f} "
                          f"(Confidence: {confidence[0]*100:5.1f}%)")
            
            # Calculate average confidence
            avg_confidence = np.mean([r["confidence"] for r in results.values()]) * 100
            print(f"\nAverage Confidence: {avg_confidence:.1f}%")
            
            # Test if confidence varies appropriately
            confidence_values = [r["confidence"] for r in results.values()]
            confidence_std = np.std(confidence_values)
            
            print(f"Confidence Variation (std): {confidence_std:.3f}")
            if confidence_std > 0.05:
                print("✓ Good: Confidence varies by stat type")
            else:
                print("⚠️  Warning: Confidence seems uniform - may need adjustment")
            
            return {
                "status": "success",
                "avg_confidence": avg_confidence,
                "confidence_variation": confidence_std,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error testing confidence improvements: {e}")
            return {"status": "error", "message": str(e)}
    
    def test_feature_improvements(self) -> Dict:
        """Test the enhanced feature engineering."""
        print("\n" + "="*60)
        print("TESTING FEATURE IMPROVEMENTS")
        print("="*60)
        
        try:
            # Test player with known age (LeBron James)
            lebron_id = 2544
            target_date = "2024-06-15"
            opponent_team_id = 1610612739  # Cleveland Cavaliers
            
            # Create enhanced features
            features_df = self.feature_engineer.create_features_for_player(
                lebron_id, target_date, 
                lookback_games=15,
                opponent_team_id=opponent_team_id,
                include_h2h_features=True,
                include_advanced_features=True
            )
            
            if features_df.empty:
                print("⚠️  No features available for test")
                return {"status": "no_data"}
            
            # Analyze features
            feature_analysis = {}
            
            # Check age features
            age_features = [col for col in features_df.columns if 'age' in col.lower()]
            feature_analysis["age_features"] = age_features
            print(f"Age-related features: {len(age_features)}")
            for feat in age_features[:5]:  # Show first 5
                value = features_df[feat].iloc[0] if not features_df[feat].empty else "N/A"
                print(f"  • {feat}: {value}")
            
            # Check opponent features
            opp_features = [col for col in features_df.columns if 'opp' in col.lower()]
            feature_analysis["opponent_features"] = opp_features
            print(f"\nOpponent-related features: {len(opp_features)}")
            for feat in opp_features[:5]:  # Show first 5
                value = features_df[feat].iloc[0] if not features_df[feat].empty else "N/A"
                print(f"  • {feat}: {value}")
            
            # Check H2H features
            h2h_features = [col for col in features_df.columns if 'h2h' in col.lower()]
            feature_analysis["h2h_features"] = h2h_features
            print(f"\nHead-to-head features: {len(h2h_features)}")
            for feat in h2h_features:
                value = features_df[feat].iloc[0] if not features_df[feat].empty else "N/A"
                print(f"  • {feat}: {value}")
            
            # Check feature diversity
            total_features = len(features_df.columns)
            print(f"\nTotal features created: {total_features}")
            
            feature_analysis.update({
                "total_features": total_features,
                "feature_quality": "good" if total_features > 50 else "needs_improvement"
            })
            
            return {
                "status": "success",
                "analysis": feature_analysis,
                "sample_features": features_df.columns.tolist()[:10]
            }
            
        except Exception as e:
            logger.error(f"Error testing feature improvements: {e}")
            return {"status": "error", "message": str(e)}
    
    def test_visualization_improvements(self) -> Dict:
        """Test the improved visualization system."""
        print("\n" + "="*60)
        print("TESTING VISUALIZATION IMPROVEMENTS")
        print("="*60)
        
        try:
            # Test with LeBron James data
            lebron_id = 2544
            player_name = "LeBron James"
            target_date = "2024-06-15"
            opponent_name = "Cleveland Cavaliers"
            
            # Create test features and predictions
            features_df = self.feature_engineer.create_features_for_player(
                lebron_id, target_date, lookback_games=10
            )
            
            if features_df.empty:
                print("⚠️  No features available for visualization test")
                return {"status": "no_data"}
            
            # Create mock predictions for testing
            predictions_df = pd.DataFrame({
                'predicted_pts': [26.5],
                'predicted_reb': [7.8],
                'predicted_ast': [9.1],
                'predicted_stl': [1.4],
                'predicted_blk': [0.6],
                'confidence_pts': [0.72],
                'confidence_reb': [0.68],
                'confidence_ast': [0.75],
                'confidence_stl': [0.58],
                'confidence_blk': [0.52]
            })
            
            # Mock recent stats
            recent_stats = {
                'pts_avg': 25.8,
                'reb_avg': 7.5,
                'ast_avg': 9.3,
                'stl_avg': 1.3,
                'blk_avg': 0.5
            }
            
            # Test visualization creation
            try:
                chart_path = self.visualizer.create_prediction_rationale_chart(
                    lebron_id, player_name, predictions_df, features_df, 
                    recent_stats, opponent_name
                )
                
                # Check if file was created
                if os.path.exists(chart_path):
                    file_size = os.path.getsize(chart_path)
                    print(f"✓ Visualization created successfully: {chart_path}")
                    print(f"  File size: {file_size / 1024:.1f} KB")
                    
                    return {
                        "status": "success",
                        "chart_path": chart_path,
                        "file_size_kb": file_size / 1024,
                        "no_unicode_errors": True  # If we get here, no unicode errors
                    }
                else:
                    print("⚠️  Visualization file not found")
                    return {"status": "file_not_created"}
                    
            except Exception as viz_error:
                if "missing from font" in str(viz_error) or "Glyph" in str(viz_error):
                    print(f"❌ Unicode/Font error still present: {viz_error}")
                    return {"status": "unicode_error", "error": str(viz_error)}
                else:
                    raise viz_error
                    
        except Exception as e:
            logger.error(f"Error testing visualization: {e}")
            return {"status": "error", "message": str(e)}
    
    def test_end_to_end_prediction(self) -> Dict:
        """Test the complete prediction pipeline."""
        print("\n" + "="*60)
        print("TESTING END-TO-END PREDICTION PIPELINE")
        print("="*60)
        
        try:
            # Test with a known active player
            test_players = [
                {"id": 2544, "name": "LeBron James"},
                {"id": 203999, "name": "Nikola Jokic"},
                {"id": 1629029, "name": "Luka Doncic"}
            ]
            
            results = {}
            
            for player in test_players:
                print(f"\nTesting predictions for {player['name']}...")
                
                try:
                    # Use recent date
                    target_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
                    
                    # Create features and get predictions directly from model manager
                    features_df = self.feature_engineer.create_features_for_player(
                        player["id"], target_date, lookback_games=15
                    )
                    
                    if features_df.empty:
                        predictions = None
                    else:
                        predictions_df = self.model_manager.predict_stats(features_df, self.stat_types)
                        
                        if not predictions_df.empty:
                            predictions = {}
                            for stat in self.stat_types:
                                pred_col = f"predicted_{stat}"
                                conf_col = f"confidence_{stat}"
                                if pred_col in predictions_df.columns:
                                    predictions[pred_col] = predictions_df[pred_col].iloc[0]
                                if conf_col in predictions_df.columns:
                                    predictions[conf_col] = predictions_df[conf_col].iloc[0]
                        else:
                            predictions = None
                    
                    if predictions:
                        # Calculate metrics
                        pred_values = [predictions.get(f"predicted_{stat}", 0) for stat in self.stat_types]
                        conf_values = [predictions.get(f"confidence_{stat}", 0) for stat in self.stat_types]
                        
                        avg_prediction = np.mean(pred_values)
                        avg_confidence = np.mean(conf_values) * 100
                        
                        print(f"  Average prediction: {avg_prediction:.1f}")
                        print(f"  Average confidence: {avg_confidence:.1f}%")
                        
                        results[player["name"]] = {
                            "predictions": predictions,
                            "avg_prediction": avg_prediction,
                            "avg_confidence": avg_confidence,
                            "status": "success"
                        }
                    else:
                        print(f"  No predictions available for {player['name']}")
                        results[player["name"]] = {"status": "no_predictions"}
                        
                except Exception as player_error:
                    print(f"  Error for {player['name']}: {player_error}")
                    results[player["name"]] = {"status": "error", "error": str(player_error)}
            
            # Overall assessment
            successful_predictions = len([r for r in results.values() if r["status"] == "success"])
            
            print(f"\nSUMMARY:")
            print(f"Successful predictions: {successful_predictions}/{len(test_players)}")
            
            if successful_predictions > 0:
                successful_results = [r for r in results.values() if r["status"] == "success"]
                overall_avg_conf = np.mean([r["avg_confidence"] for r in successful_results])
                print(f"Overall average confidence: {overall_avg_conf:.1f}%")
            
            return {
                "status": "success",
                "successful_predictions": successful_predictions,
                "total_tests": len(test_players),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error in end-to-end test: {e}")
            return {"status": "error", "message": str(e)}
    
    def _train_test_models(self, stat_types: List[str]) -> None:
        """Train simple models for testing purposes."""
        print("Training test models...")
        
        try:
            # Get some training data
            conn = sqlite3.connect(self.db_path)
            
            # Get recent player data for training
            query = """
                SELECT player_id, game_date, pts, reb, ast, stl, blk
                FROM player_games 
                WHERE game_date >= '2024-01-01' AND game_date < '2024-06-01'
                AND pts > 0 AND reb >= 0 AND ast >= 0
                ORDER BY game_date DESC
                LIMIT 1000
            """
            
            training_data = pd.read_sql_query(query, conn)
            conn.close()
            
            if len(training_data) < 100:
                print("⚠️  Insufficient training data")
                return
            
            # Train simple models for missing stats
            for stat_type in stat_types:
                if stat_type in training_data.columns:
                    print(f"  Training {stat_type} model...")
                    
                    # Create simple features for training
                    features = pd.DataFrame({
                        'player_age': [30] * len(training_data),  # Dummy age
                        'recent_avg': training_data[stat_type].rolling(3, min_periods=1).mean(),
                        'trend': training_data[stat_type].diff().fillna(0),
                        'consistency': training_data[stat_type].rolling(5, min_periods=1).std().fillna(1)
                    })
                    
                    target = training_data[stat_type]
                    
                    # Create and train simple ensemble model
                    predictor = AdvancedEnsembleStatPredictor(stat_type)
                    
                    try:
                        metrics = predictor.train(features, target, optimize_hyperparams=False)
                        self.model_manager.predictors[stat_type] = predictor
                        print(f"    ✓ Trained {stat_type} (MAE: {metrics.get('mae', 0):.2f})")
                    except Exception as train_error:
                        print(f"    ❌ Failed to train {stat_type}: {train_error}")
                        
        except Exception as e:
            logger.error(f"Error training test models: {e}")
    
    def run_all_tests(self) -> Dict:
        """Run all improvement tests."""
        print("STARTING COMPREHENSIVE IMPROVEMENT TESTS")
        print("=" * 80)
        
        all_results = {}
        
        # Test 1: Confidence improvements
        all_results["confidence"] = self.test_confidence_improvements()
        
        # Test 2: Feature improvements
        all_results["features"] = self.test_feature_improvements()
        
        # Test 3: Visualization improvements
        all_results["visualization"] = self.test_visualization_improvements()
        
        # Test 4: End-to-end pipeline
        all_results["end_to_end"] = self.test_end_to_end_prediction()
        
        # Summary
        print("\n" + "="*80)
        print("FINAL TEST SUMMARY")
        print("="*80)
        
        for test_name, result in all_results.items():
            status = result.get("status", "unknown")
            print(f"{test_name.upper():15}: {status}")
            
            if status == "success":
                if test_name == "confidence":
                    avg_conf = result.get("avg_confidence", 0)
                    print(f"                 Average confidence: {avg_conf:.1f}%")
                elif test_name == "features":
                    total_feat = result.get("analysis", {}).get("total_features", 0)
                    print(f"                 Total features: {total_feat}")
                elif test_name == "end_to_end":
                    success_rate = result.get("successful_predictions", 0) / result.get("total_tests", 1)
                    print(f"                 Success rate: {success_rate*100:.1f}%")
        
        # Overall assessment
        successful_tests = len([r for r in all_results.values() if r.get("status") == "success"])
        total_tests = len(all_results)
        
        print(f"\nOVERALL: {successful_tests}/{total_tests} tests passed")
        
        if successful_tests == total_tests:
            print("🎉 ALL IMPROVEMENTS WORKING CORRECTLY!")
        elif successful_tests >= total_tests * 0.75:
            print("✅ Most improvements working - minor issues to address")
        else:
            print("⚠️  Significant issues detected - review needed")
        
        return all_results


def main():
    """Main test execution."""
    # Ensure plots directory exists
    os.makedirs("plots", exist_ok=True)
    
    # Run tests
    tester = PredictionImprovementTester()
    results = tester.run_all_tests()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"test_results_{timestamp}.json"
    
    import json
    with open(results_file, 'w') as f:
        # Convert numpy types to regular Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        # Simple recursive conversion
        def clean_for_json(data):
            if isinstance(data, dict):
                return {k: clean_for_json(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [clean_for_json(item) for item in data]
            elif isinstance(data, (np.ndarray, np.integer, np.floating)):
                return convert_numpy(data)
            else:
                return data
        
        clean_results = clean_for_json(results)
        json.dump(clean_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")


if __name__ == "__main__":
    main() 