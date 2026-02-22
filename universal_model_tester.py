# universal_model_tester.py
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import time
import os
from datetime import datetime

class UniversalModelTester:
    def __init__(self):
        self.results = {}
        self.dataset_path = None
        self.X_test = None
        self.y_test = None
        self.feature_names = None
        
    def load_dataset(self, dataset_path=None):
        """Load and prepare the universal test dataset WITH INTERACTION FEATURES"""
        if dataset_path is None:
            # Find the latest dataset
            dataset_files = [f for f in os.listdir('.') 
                           if f.startswith(('kaggle_training_dataset_', 'synthetic_large_', 'enhanced_medical_dataset')) 
                           and f.endswith('.csv')]
            if not dataset_files:
                raise FileNotFoundError("No dataset files found!")
            dataset_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            dataset_path = dataset_files[0]
        
        self.dataset_path = dataset_path
        print(f"📊 Loading dataset: {dataset_path}")
        
        # Load with proper dtype handling
        df = pd.read_csv(dataset_path, low_memory=False)
        print(f"✅ Loaded {len(df):,} records")
        
        # Prepare features WITH INTERACTIONS (same as training)
        X, y, feature_names = self._prepare_features_with_interactions(df)
        self.feature_names = feature_names
        
        # Split data (use same random state for consistency)
        _, self.X_test, _, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=pd.cut(y, bins=10)
        )
        
        print(f"🧪 Test set: {self.X_test.shape[0]:,} samples")
        print(f"🎯 Features: {len(feature_names)}")
        print(f"📋 Feature names: {feature_names}")
        
        return self.X_test, self.y_test
    
    def _prepare_features_with_interactions(self, df):
        """Prepare features WITH INTERACTION FEATURES (same as training pipeline)"""
        # Define feature columns
        pollution_features = ['PM2.5', 'PM10', 'O3', 'CO', 'NO2', 'SO2']
        demographic_features = ['age', 'bmi']
        lifestyle_features = ['smoker', 'former_smoker', 'outdoor_worker', 'regular_exercise', 'urban_resident']
        medical_conditions = [
            'asthma', 'copd', 'heart_disease', 'hypertension', 'diabetes',
            'pregnancy', 'child_under_5', 'child_5_12', 'elderly_65_75', 'elderly_over_75'
        ]
        
        # Check which columns exist
        available_features = []
        for feature_group in [pollution_features, demographic_features, lifestyle_features, medical_conditions]:
            for feature in feature_group:
                if feature in df.columns:
                    available_features.append(feature)
        
        # Create clean dataframe
        df_clean = df[available_features + ['medical_risk_score']].copy()
        
        # Convert all to numeric (robust handling)
        for col in available_features + ['medical_risk_score']:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
        
        # Remove any remaining non-numeric values
        for col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
        
        X = df_clean[available_features]
        y = df_clean['medical_risk_score']
        
        # 🎯 CRITICAL FIX: CREATE INTERACTION FEATURES (same as training)
        X = self._create_interaction_features(X)
        
        return X, y, list(X.columns)
    
    def _create_interaction_features(self, X):
        """Create the same interaction features as during training"""
        print("🔗 Creating interaction features...")
        
        X_enhanced = X.copy()
        
        # Ensure all columns are numeric
        for col in X_enhanced.columns:
            X_enhanced[col] = pd.to_numeric(X_enhanced[col], errors='coerce').fillna(0)
        
        # Pollution-health condition interactions (SAME AS TRAINING)
        if 'PM2.5' in X_enhanced.columns and 'asthma' in X_enhanced.columns:
            X_enhanced['pm25_asthma'] = X_enhanced['PM2.5'] * X_enhanced['asthma']
            print("✅ Created pm25_asthma interaction")
        
        if 'O3' in X_enhanced.columns and 'child_under_5' in X_enhanced.columns:
            X_enhanced['o3_child'] = X_enhanced['O3'] * X_enhanced['child_under_5']
            print("✅ Created o3_child interaction")
        
        if 'PM2.5' in X_enhanced.columns and 'elderly_over_75' in X_enhanced.columns:
            X_enhanced['pm25_elderly'] = X_enhanced['PM2.5'] * X_enhanced['elderly_over_75']
            print("✅ Created pm25_elderly interaction")
        
        if 'CO' in X_enhanced.columns and 'heart_disease' in X_enhanced.columns:
            X_enhanced['co_heart'] = X_enhanced['CO'] * X_enhanced['heart_disease']
            print("✅ Created co_heart interaction")
        
        print(f"📈 Total features after interactions: {X_enhanced.shape[1]}")
        return X_enhanced
    
    def test_model(self, model_path, model_name):
        """Test a single model and return performance metrics"""
        print(f"\n{'='*50}")
        print(f"🧪 TESTING: {model_name}")
        print(f"{'='*50}")
        
        try:
            # Load model
            print(f"📁 Loading model from: {model_path}")
            model_data = joblib.load(model_path)
            model = model_data['model']
            scaler = model_data.get('scaler', None)
            feature_columns = model_data.get('feature_columns', None)
            
            print(f"🔧 Model type: {type(model).__name__}")
            print(f"📋 Expected features: {len(feature_columns) if feature_columns else 'Not specified'}")
            
            # Prepare test data
            if feature_columns:
                # Ensure we have all required features
                missing_features = set(feature_columns) - set(self.X_test.columns)
                if missing_features:
                    print(f"⚠️ Missing features: {missing_features}")
                    # Add missing features as zeros
                    for feature in missing_features:
                        self.X_test[feature] = 0
                
                X_test = self.X_test[feature_columns]
            else:
                X_test = self.X_test
            
            print(f"📊 Test data shape: {X_test.shape}")
            
            # Scale if scaler exists
            if scaler:
                X_test_scaled = scaler.transform(X_test)
                print("⚖️ Features scaled")
            else:
                X_test_scaled = X_test
                print("⚖️ No scaling applied")
            
            # Predict
            start_time = time.time()
            y_pred = model.predict(X_test_scaled)
            prediction_time = time.time() - start_time
            
            # Calculate metrics
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            
            # Ensure predictions are in 0-10 range
            y_pred_clipped = np.clip(y_pred, 0, 10)
            
            # Additional metrics for clipped predictions
            rmse_clipped = np.sqrt(mean_squared_error(self.y_test, y_pred_clipped))
            r2_clipped = r2_score(self.y_test, y_pred_clipped)
            
            # Store results
            results = {
                'model_name': model_name,
                'model_path': model_path,
                'r2_score': r2,
                'r2_score_clipped': r2_clipped,
                'rmse': rmse,
                'rmse_clipped': rmse_clipped,
                'mae': mae,
                'mse': mse,
                'prediction_time': prediction_time,
                'test_samples': len(self.y_test),
                'timestamp': datetime.now().isoformat()
            }
            
            # Print results
            print(f"✅ R² Score: {r2:.4f}")
            print(f"✅ R² Score (Clipped): {r2_clipped:.4f}")
            print(f"✅ RMSE: {rmse:.4f}")
            print(f"✅ RMSE (Clipped): {rmse_clipped:.4f}")
            print(f"✅ MAE: {mae:.4f}")
            print(f"⏱️ Prediction Time: {prediction_time:.4f}s")
            print(f"📊 Test Samples: {len(self.y_test):,}")
            
            self.results[model_name] = results
            return results
            
        except Exception as e:
            print(f"❌ Error testing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def compare_all_models(self, models_dict):
        """Test multiple models and compare results"""
        print(f"\n🎯 COMPARATIVE ANALYSIS")
        print(f"{'='*60}")
        
        # Ensure test data is loaded
        if self.X_test is None:
            self.load_dataset()
        
        # Test each model
        for model_name, model_path in models_dict.items():
            if os.path.exists(model_path):
                self.test_model(model_path, model_name)
            else:
                print(f"❌ Model file not found: {model_path}")
        
        # Generate comparison report
        if self.results:
            self.generate_comparison_report()
        else:
            print("❌ No models were successfully tested!")
        
        return self.results
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        if not self.results:
            print("❌ No results to compare!")
            return
        
        print(f"\n{'='*60}")
        print(f"📊 COMPARATIVE PERFORMANCE REPORT")
        print(f"{'='*60}")
        
        # Create results dataframe
        results_df = pd.DataFrame(self.results.values())
        results_df = results_df.sort_values('r2_score', ascending=False)
        
        # Print comparison table
        print("\n🏆 MODEL PERFORMANCE RANKING:")
        print("-" * 80)
        print(f"{'Model':<25} {'R² Score':<10} {'RMSE':<10} {'MAE':<10} {'Time(s)':<10}")
        print("-" * 80)
        
        for _, row in results_df.iterrows():
            print(f"{row['model_name']:<25} {row['r2_score']:<10.4f} {row['rmse']:<10.4f} {row['mae']:<10.4f} {row['prediction_time']:<10.4f}")
        
        # Performance interpretation
        best_model = results_df.iloc[0]
        print(f"\n🏆 BEST MODEL: {best_model['model_name']}")
        print(f"🎯 R² Score: {best_model['r2_score']:.4f}")
        
        if best_model['r2_score'] > 0.9:
            print("💪 EXCELLENT performance! World-class accuracy!")
        elif best_model['r2_score'] > 0.8:
            print("👍 VERY GOOD performance! Highly reliable!")
        elif best_model['r2_score'] > 0.7:
            print("⚠️ GOOD performance! Suitable for production!")
        else:
            print("❌ Performance needs improvement!")
        
        # Create visualizations
        self._create_comparison_plots(results_df)
        
        # Save results
        results_df.to_csv('model_comparison_results.csv', index=False)
        print(f"\n💾 Results saved to: model_comparison_results.csv")
        
        return results_df
    
    def _create_comparison_plots(self, results_df):
        """Create comparison visualizations"""
        try:
            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Medical AI Model Performance Comparison', fontsize=16, fontweight='bold')
            
            # R² Score comparison
            colors = ['#FF6B6B' if x == results_df['r2_score'].max() else '#4ECDC4' for x in results_df['r2_score']]
            axes[0,0].barh(results_df['model_name'], results_df['r2_score'], color=colors)
            axes[0,0].set_xlabel('R² Score')
            axes[0,0].set_title('R² Score Comparison (Higher is Better)')
            axes[0,0].axvline(x=0.9, color='red', linestyle='--', alpha=0.7, label='Excellent (0.9)')
            axes[0,0].axvline(x=0.8, color='orange', linestyle='--', alpha=0.7, label='Good (0.8)')
            axes[0,0].legend()
            
            # RMSE comparison
            colors = ['#FF6B6B' if x == results_df['rmse'].min() else '#4ECDC4' for x in results_df['rmse']]
            axes[0,1].barh(results_df['model_name'], results_df['rmse'], color=colors)
            axes[0,1].set_xlabel('RMSE')
            axes[0,1].set_title('RMSE Comparison (Lower is Better)')
            
            # MAE comparison
            colors = ['#FF6B6B' if x == results_df['mae'].min() else '#4ECDC4' for x in results_df['mae']]
            axes[1,0].barh(results_df['model_name'], results_df['mae'], color=colors)
            axes[1,0].set_xlabel('MAE')
            axes[1,0].set_title('MAE Comparison (Lower is Better)')
            
            # Prediction time comparison
            colors = ['#FF6B6B' if x == results_df['prediction_time'].min() else '#4ECDC4' for x in results_df['prediction_time']]
            axes[1,1].barh(results_df['model_name'], results_df['prediction_time'], color=colors)
            axes[1,1].set_xlabel('Prediction Time (seconds)')
            axes[1,1].set_title('Prediction Speed Comparison')
            
            plt.tight_layout()
            plt.savefig('model_comparison_plots.png', dpi=300, bbox_inches='tight')
            print("📊 Comparison plots saved as: model_comparison_plots.png")
            plt.show()
            
        except Exception as e:
            print(f"❌ Error creating plots: {e}")

# Quick testing function
def quick_test_model(model_path, model_name):
    """Quick test for a single model"""
    tester = UniversalModelTester()
    tester.load_dataset()
    results = tester.test_model(model_path, model_name)
    return results

if __name__ == "__main__":
    # Example usage
    tester = UniversalModelTester()
    
    # Load test data
    tester.load_dataset()
    
    # Define models to test
    models_to_test = {
        "Random Forest": "models/medical_ai_model.pkl",
        # We'll add other models as we create them
         "XGBoost": "models/gradient_boosting_model.pkl",
         "SVR": "models/fast_svr_model.pkl", 
         "Neural Network": "models/neural_network_model.pkl",
         "Decision Tree": "models/decision_tree_model.pkl",
         "Linear Regression": "models/linear_regression_model.pkl"
    }
    tester.compare_all_models(models_to_test)
    # Test current Random Forest model first
    # if os.path.exists("models/medical_ai_model.pkl"):
    #     tester.test_model("models/medical_ai_model.pkl", "Random Forest")
    # else:
    #     print("❌ Random Forest model not found at: models/medical_ai_model.pkl")
    
   