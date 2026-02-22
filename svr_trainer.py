# fast_svr_trainer.py
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FastSVR_Trainer:
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
    
    def train_fast_svr(self, dataset_path=None):
        """Train SVR quickly with optimized settings"""
        print("\n" + "="*60)
        print("🚀 TRAINING FAST SUPPORT VECTOR REGRESSION (SVR)")
        print("="*60)
        
        X, y, feature_names = self.load_and_prepare_data(dataset_path)
        if X is None:
            return None
        
        print(f"📊 Dataset shape: {X.shape}")
        print(f"🎯 Target range: {y.min():.2f} to {y.max():.2f}")
        
        # Use even smaller subset for very fast training
        print("📉 Using 10,000 sample subset for very fast SVR training...")
        X_sample, _, y_sample, _ = train_test_split(X, y, train_size=10000, random_state=42, stratify=pd.cut(y, bins=10))
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_sample, y_sample, test_size=0.2, random_state=42, stratify=pd.cut(y_sample, bins=10)
        )
        
        print(f"📚 Training samples: {X_train.shape[0]:,}")
        print(f"🧪 Testing samples: {X_test.shape[0]:,}")
        
        print("⚖️ Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Fast SVR with linear kernel (much faster than RBF)
        print("🤖 Training SVR with linear kernel (fastest option)...")
        self.model = SVR(
            kernel='linear',  # Linear is much faster than RBF
            C=1.0,
            epsilon=0.1,
            cache_size=1000,
            verbose=True
        )
        
        print("⏳ Training started (should take 1-5 minutes)...")
        self.model.fit(X_train_scaled, y_train)
        
        print("\n📈 Evaluating Fast SVR...")
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
        print("🎯 FAST SVR RESULTS (Linear Kernel)")
        print("="*50)
        print(f"📚 Training R²:   {train_r2:.4f}  # From {len(y_train):,} samples")
        print(f"🧪 Testing R²:    {test_r2:.4f}  # From {len(y_test):,} samples")
        print(f"📚 Training RMSE: {train_rmse:.4f}")
        print(f"🧪 Testing RMSE:  {test_rmse:.4f}")
        print(f"📚 Training MAE:  {train_mae:.4f}")
        print(f"🧪 Testing MAE:   {test_mae:.4f}")
        
        # Performance comparison
        current_best_r2 = 0.9593  # Gradient Boosting
        if test_r2 > current_best_r2:
            print(f"🎉 SVR OUTPERFORMS Gradient Boosting!")
        elif test_r2 > 0.9:
            print(f"📊 SVR is competitive (R²: {test_r2:.4f})")
        else:
            print(f"📉 SVR performance: {test_r2:.4f}")
        
        # Show SVR coefficients for linear kernel
        if hasattr(self.model, 'coef_'):
            print(f"\n🔍 SVR Feature Coefficients (Linear Kernel):")
            coef_df = pd.DataFrame({
                'feature': feature_names,
                'coefficient': self.model.coef_[0]
            }).sort_values('coefficient', key=abs, ascending=False)
            
            for i, row in coef_df.head(5).iterrows():
                print(f"   {row['feature']:20} {row['coefficient']:+.4f}")
    
    def save_model(self):
        """Save the trained SVR model"""
        os.makedirs('models', exist_ok=True)
        
        model_package = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'training_date': datetime.now().isoformat(),
            'model_type': 'FastSVR_Linear'
        }
        
        model_path = 'models/fast_svr_model.pkl'
        joblib.dump(model_package, model_path)
        print(f"\n💾 Fast SVR model saved to: {model_path}")
        print("🎉 Ready for comparative testing!")

if __name__ == "__main__":
    print("🚀 Training FAST Support Vector Regression")
    print("⚡ Using linear kernel for speed (1-5 minutes)")
    print("📊 Will show if linear SVR can be competitive")
    
    trainer = FastSVR_Trainer()
    model = trainer.train_fast_svr()
    
    if model:
        print("\n✅ Fast SVR training completed!")
        print("📁 Run: python universal_model_tester.py for comparison")
        print("💡 Even if SVR is worse, it's valuable for algorithm diversity")