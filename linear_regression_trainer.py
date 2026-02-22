# linear_regression_trainer.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class LinearRegressionTrainer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def find_latest_dataset(self):
        """Find the most recent dataset file"""
        dataset_files = []
        
        for file in os.listdir('.'):
            if file.startswith(('kaggle_training_dataset_', 'synthetic_large_', 'enhanced_medical_dataset')) and file.endswith('.csv'):
                dataset_files.append(file)
        
        if not dataset_files:
            print("❌ No dataset files found.")
            return None
        
        dataset_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return dataset_files[0]
    
    def load_and_prepare_data(self, dataset_path=None):
        """Load and prepare data with interaction features"""
        if dataset_path is None:
            dataset_path = self.find_latest_dataset()
            if dataset_path is None:
                return None, None, None
        
        print(f"📥 Loading dataset: {dataset_path}")
        try:
            df = pd.read_csv(dataset_path, low_memory=False)
            print(f"✅ Loaded {len(df):,} records")
            return self.prepare_features_with_interactions(df)
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None, None, None
    
    def prepare_features_with_interactions(self, df):
        """Prepare features with interaction features"""
        print("⚙️ Preparing features with interactions...")
        
        pollution_features = ['PM2.5', 'PM10', 'O3', 'CO', 'NO2', 'SO2']
        demographic_features = ['age', 'bmi']
        lifestyle_features = ['smoker', 'former_smoker', 'outdoor_worker', 'regular_exercise', 'urban_resident']
        medical_conditions = [
            'asthma', 'copd', 'heart_disease', 'hypertension', 'diabetes',
            'pregnancy', 'child_under_5', 'child_5_12', 'elderly_65_75', 'elderly_over_75'
        ]
        
        available_features = []
        for feature_group in [pollution_features, demographic_features, lifestyle_features, medical_conditions]:
            for feature in feature_group:
                if feature in df.columns:
                    available_features.append(feature)
        
        print(f"📊 Using {len(available_features)} base features")
        
        df_clean = df[available_features + ['medical_risk_score']].copy()
        
        for col in available_features + ['medical_risk_score']:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
        
        X = df_clean[available_features]
        y = df_clean['medical_risk_score']
        
        X = self.create_interaction_features(X)
        self.feature_columns = list(X.columns)
        
        print(f"✅ Final features: {len(self.feature_columns)}")
        return X, y, self.feature_columns
    
    def create_interaction_features(self, X):
        """Create interaction features"""
        X_enhanced = X.copy()
        
        for col in X_enhanced.columns:
            X_enhanced[col] = pd.to_numeric(X_enhanced[col], errors='coerce').fillna(0)
        
        if 'PM2.5' in X_enhanced.columns and 'asthma' in X_enhanced.columns:
            X_enhanced['pm25_asthma'] = X_enhanced['PM2.5'] * X_enhanced['asthma']
        if 'O3' in X_enhanced.columns and 'child_under_5' in X_enhanced.columns:
            X_enhanced['o3_child'] = X_enhanced['O3'] * X_enhanced['child_under_5']
        if 'PM2.5' in X_enhanced.columns and 'elderly_over_75' in X_enhanced.columns:
            X_enhanced['pm25_elderly'] = X_enhanced['PM2.5'] * X_enhanced['elderly_over_75']
        if 'CO' in X_enhanced.columns and 'heart_disease' in X_enhanced.columns:
            X_enhanced['co_heart'] = X_enhanced['CO'] * X_enhanced['heart_disease']
        
        print(f"📈 Total features: {X_enhanced.shape[1]}")
        return X_enhanced
    
    def train_linear_regression(self, dataset_path=None):
        """Train Linear Regression model"""
        print("\n" + "="*60)
        print("🚀 TRAINING LINEAR REGRESSION (Baseline)")
        print("="*60)
        
        X, y, feature_names = self.load_and_prepare_data(dataset_path)
        if X is None:
            return None
        
        print(f"📊 Dataset shape: {X.shape}")
        print(f"🎯 Target range: {y.min():.2f} to {y.max():.2f}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=pd.cut(y, bins=10)
        )
        
        print(f"📚 Training samples: {X_train.shape[0]:,}")
        print(f"🧪 Testing samples: {X_test.shape[0]:,}")
        
        print("⚖️ Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Linear Regression (baseline model)
        print("📈 Training Linear Regression...")
        self.model = LinearRegression()
        
        print("⏳ Training started (should be instant)...")
        self.model.fit(X_train_scaled, y_train)
        
        print("\n📈 Evaluating Linear Regression...")
        self.evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test, feature_names)
        
        self.save_model()
        return self.model
    
    def evaluate_model(self, X_train_scaled, X_test_scaled, y_train, y_test, feature_names):
        """Real evaluation with actual metrics"""
        print("⏳ Making predictions...")
        
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        y_train_pred = np.clip(y_train_pred, 0, 10)
        y_test_pred = np.clip(y_test_pred, 0, 10)
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = np.mean(np.abs(y_train - y_train_pred))
        test_mae = np.mean(np.abs(y_test - y_test_pred))
        
        print("\n" + "="*50)
        print("🎯 LINEAR REGRESSION RESULTS")
        print("="*50)
        print(f"📚 Training R²:   {train_r2:.4f}  # From {len(y_train):,} samples")
        print(f"🧪 Testing R²:    {test_r2:.4f}  # From {len(y_test):,} samples")
        print(f"📚 Training RMSE: {train_rmse:.4f}")
        print(f"🧪 Testing RMSE:  {test_rmse:.4f}")
        print(f"📚 Training MAE:  {train_mae:.4f}")
        print(f"🧪 Testing MAE:   {test_mae:.4f}")
        
        # Performance comparison (baseline perspective)
        current_best_r2 = 0.9615  # Neural Network
        performance_gap = current_best_r2 - test_r2
        
        print(f"\n📊 BASELINE PERFORMANCE ANALYSIS:")
        print(f"   • Linear Regression explains {test_r2*100:.1f}% of variance")
        print(f"   • Best model (Neural Network) explains {current_best_r2*100:.1f}%")
        print(f"   • Performance gap: {performance_gap*100:.1f}% variance")
        print(f"   • This shows importance of non-linear modeling")
        
        # Show coefficients
        coef_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': self.model.coef_
        }).sort_values('coefficient', key=abs, ascending=False)
        
        print(f"\n🔝 Top 5 Most Influential Features:")
        for i, row in coef_df.head(5).iterrows():
            print(f"   {i+1:2d}. {row['feature']:20} {row['coefficient']:+.4f}")
    
    def save_model(self):
        """Save the trained Linear Regression model"""
        os.makedirs('models', exist_ok=True)
        
        model_package = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'training_date': datetime.now().isoformat(),
            'model_type': 'LinearRegression'
        }
        
        model_path = 'models/linear_regression_model.pkl'
        joblib.dump(model_package, model_path)
        print(f"\n💾 Linear Regression model saved to: {model_path}")
        print("🎉 Ready for comparative testing!")

if __name__ == "__main__":
    print("🚀 Training Linear Regression")
    print("📈 Baseline model to show non-linearity importance")
    print("📊 Demonstrates how much advanced models improve over linear")
    
    trainer = LinearRegressionTrainer()
    model = trainer.train_linear_regression()
    
    if model:
        print("\n✅ Linear Regression training completed!")
        print("📁 Run: python universal_model_tester.py for 6-model comparison")