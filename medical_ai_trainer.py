import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MedicalAITrainer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        
    def find_latest_dataset(self):
        """Find the most recent dataset file"""
        dataset_files = []
        
        for file in os.listdir('.'):
            if file.startswith(('kaggle_training_dataset_', 'synthetic_large_', 'enhanced_medical_dataset')) and file.endswith('.csv'):
                dataset_files.append(file)
        
        if not dataset_files:
            print("❌ No dataset files found. Please run the data generator first.")
            return None
        
        dataset_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        latest_file = dataset_files[0]
        
        print(f"📁 Found dataset: {latest_file}")
        print(f"📅 Modified: {datetime.fromtimestamp(os.path.getmtime(latest_file))}")
        
        return latest_file
    
    def load_and_prepare_data(self, dataset_path=None):
        """Load and prepare the training data with robust error handling"""
        if dataset_path is None:
            dataset_path = self.find_latest_dataset()
            if dataset_path is None:
                return None, None, None
        
        print(f"📥 Loading dataset: {dataset_path}")
        try:
            df = pd.read_csv(dataset_path)
            print(f"✅ Loaded {len(df):,} records")
            
            print("🔍 Dataset overview:")
            print(f"   Columns: {list(df.columns)}")
            print(f"   Shape: {df.shape}")
            
            return self.prepare_features_robust(df)
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None, None, None
    
    def prepare_features_robust(self, df):
        """Robust feature preparation that handles mixed data types"""
        print("⚙️ Preparing features with robust type handling...")
        
        # Define feature columns
        pollution_features = ['PM2.5', 'PM10', 'O3', 'CO', 'NO2', 'SO2']
        demographic_features = ['age', 'bmi']
        lifestyle_features = ['smoker', 'former_smoker', 'outdoor_worker', 'regular_exercise', 'urban_resident']
        
        medical_conditions = [
            'asthma', 'copd', 'heart_disease', 'hypertension', 'diabetes',
            'pregnancy', 'child_under_5', 'child_5_12', 'elderly_65_75', 'elderly_over_75'
        ]
        
        # Check which columns actually exist
        available_features = []
        for feature_group in [pollution_features, demographic_features, lifestyle_features, medical_conditions]:
            for feature in feature_group:
                if feature in df.columns:
                    available_features.append(feature)
        
        print(f"📊 Using {len(available_features)} features")
        
        # Create a clean copy of the dataframe
        df_clean = df[available_features + ['medical_risk_score']].copy()
        
        # Handle each feature type carefully
        for feature in available_features:
            if feature in pollution_features:
                # Pollution data - ensure numeric
                df_clean[feature] = pd.to_numeric(df_clean[feature], errors='coerce').fillna(0)
                
            elif feature in demographic_features:
                # Demographic data - ensure numeric
                df_clean[feature] = pd.to_numeric(df_clean[feature], errors='coerce')
                df_clean[feature] = df_clean[feature].fillna(df_clean[feature].median())
                
            elif feature in lifestyle_features + medical_conditions:
                # Boolean-like features - handle mixed types robustly
                print(f"   Processing {feature}...")
                
                # First, try to convert to numeric to handle decimal strings like '0.25'
                numeric_vals = pd.to_numeric(df_clean[feature], errors='coerce')
                
                # If conversion worked and we have values between 0 and 1, threshold to 0/1
                if not numeric_vals.isna().all():
                    # Convert to boolean (threshold at 0.5) then to integer
                    df_clean[feature] = (numeric_vals > 0.5).astype(int)
                else:
                    # Try string conversion for actual boolean strings
                    str_vals = df_clean[feature].astype(str).str.lower()
                    df_clean[feature] = str_vals.isin(['true', '1', 'yes', 't']).astype(int)
                
                df_clean[feature] = df_clean[feature].fillna(0)
        
        # Handle target variable
        df_clean['medical_risk_score'] = pd.to_numeric(df_clean['medical_risk_score'], errors='coerce')
        df_clean = df_clean.dropna(subset=['medical_risk_score'])
        
        # Remove any remaining non-numeric values
        for col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
        
        X = df_clean[available_features]
        y = df_clean['medical_risk_score']
        
        print(f"✅ Features prepared: {X.shape[1]} features, {X.shape[0]} samples")
        print(f"✅ Data types: {X.dtypes.value_counts().to_dict()}")
        
        # Verify all data is numeric
        if not all(X.dtypes.isin([np.int64, np.float64])):
            print("❌ Warning: Some features are still not numeric")
            print(f"   Non-numeric dtypes: {X.dtypes[X.dtypes.apply(lambda x: not np.issubdtype(x, np.number))]}")
        
        return X, y, available_features
    
    def create_interaction_features(self, X):
        """Create medically meaningful interaction features"""
        print("🔗 Creating interaction features...")
        
        X_enhanced = X.copy()
        
        # Ensure all columns are numeric
        for col in X_enhanced.columns:
            X_enhanced[col] = pd.to_numeric(X_enhanced[col], errors='coerce').fillna(0)
        
        # Pollution-health condition interactions
        interaction_features = []
        
        if 'PM2.5' in X_enhanced.columns and 'asthma' in X_enhanced.columns:
            X_enhanced['pm25_asthma'] = X_enhanced['PM2.5'] * X_enhanced['asthma']
            interaction_features.append('pm25_asthma')
        
        if 'O3' in X_enhanced.columns and 'child_under_5' in X_enhanced.columns:
            X_enhanced['o3_child'] = X_enhanced['O3'] * X_enhanced['child_under_5']
            interaction_features.append('o3_child')
        
        if 'PM2.5' in X_enhanced.columns and 'elderly_over_75' in X_enhanced.columns:
            X_enhanced['pm25_elderly'] = X_enhanced['PM2.5'] * X_enhanced['elderly_over_75']
            interaction_features.append('pm25_elderly')
        
        if 'CO' in X_enhanced.columns and 'heart_disease' in X_enhanced.columns:
            X_enhanced['co_heart'] = X_enhanced['CO'] * X_enhanced['heart_disease']
            interaction_features.append('co_heart')
        
        print(f"📈 Added {len(interaction_features)} interaction features")
        print(f"📈 Total features: {X_enhanced.shape[1]}")
        
        return X_enhanced
    
    def prepare_data_for_training(self, X, y):
        """Prepare data for training with proper scaling"""
        print("📊 Splitting data...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"📊 Training set: {X_train.shape[0]:,} samples")
        print(f"📊 Test set: {X_test.shape[0]:,} samples")
        
        # Convert to numpy arrays to avoid pandas issues
        X_train_np = X_train.astype(float).values
        X_test_np = X_test.astype(float).values
        y_train_np = y_train.astype(float).values
        y_test_np = y_test.astype(float).values
        
        print("⚖️ Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train_np)
        X_test_scaled = self.scaler.transform(X_test_np)
        
        return X_train_scaled, X_test_scaled, y_train_np, y_test_np
    
    def train_medical_ai(self, dataset_path=None):
        """Train the medical AI model"""
        print("🏥 Starting Medical AI Training...")
        
        # Load and prepare data
        X, y, feature_names = self.load_and_prepare_data(dataset_path)
        if X is None:
            print("❌ Failed to load training data")
            return None
        
        # Create interaction features
        X_enhanced = self.create_interaction_features(X)
        feature_names = list(X_enhanced.columns)
        
        # Prepare for training
        X_train_scaled, X_test_scaled, y_train, y_test = self.prepare_data_for_training(X_enhanced, y)
        
        # Train Random Forest model
        print("🤖 Training Random Forest model...")
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        print("⏳ Training in progress... (this may take 5-15 minutes)")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        print("📈 Evaluating model performance...")
        self.evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test, feature_names)
        
        # Save model
        self.save_model_and_analysis(X_enhanced.columns, X_train_scaled, y_train, X_test_scaled, y_test)
        
        return self.model
    
    def evaluate_model(self, X_train_scaled, X_test_scaled, y_train, y_test, feature_names):
        """Comprehensive model evaluation"""
        # Predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        print("\n" + "="*50)
        print("🎯 MEDICAL AI TRAINING RESULTS")
        print("="*50)
        print(f"✅ Training R²:   {train_r2:.4f}")
        print(f"✅ Testing R²:    {test_r2:.4f}")
        print(f"✅ Training RMSE: {train_rmse:.4f}")
        print(f"✅ Testing RMSE:  {test_rmse:.4f}")
        
        # Performance interpretation
        if test_r2 > 0.9:
            print("🏆 Excellent performance! Model will provide accurate risk assessments.")
        elif test_r2 > 0.8:
            print("👍 Very good performance! Model will provide reliable risk assessments.")
        elif test_r2 > 0.7:
            print("⚠️ Good performance. Model should work well for most cases.")
        else:
            print("❌ Performance needs improvement. Consider more data or feature engineering.")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔝 Top 10 Most Important Features:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"   {i+1:2d}. {row['feature']:20} {row['importance']:.4f}")
    
    def save_model_and_analysis(self, feature_columns, X_train, y_train, X_test, y_test):
        """Save the trained model and analysis"""
        os.makedirs('models', exist_ok=True)
        
        model_package = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': list(feature_columns),
            'training_date': datetime.now().isoformat(),
            'model_type': 'MedicalRiskRandomForest',
            'version': '1.0',
            'dataset_size': X_train.shape[0] + X_test.shape[0]
        }
        
        model_path = 'models/medical_ai_model.pkl'
        joblib.dump(model_package, model_path)
        
        print(f"\n💾 Model saved to: {model_path}")
        print(f"📊 Model trained on: {model_package['dataset_size']:,} records")
        print(f"🎯 Features used: {len(feature_columns)}")
        
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importance.to_csv('models/feature_importance.csv', index=False)
        print("📈 Feature importance analysis saved")
        
        print("\n🎉 MEDICAL AI TRAINING COMPLETE!")
        print("🚀 Your model is ready to provide personalized risk assessments!")

# Quick diagnostic function to check data issues
def diagnose_data_issues():
    """Diagnose what's wrong with the dataset"""
    print("🔍 Diagnosing data issues...")
    
    # Find latest dataset
    dataset_files = [f for f in os.listdir('.') if f.startswith('kaggle_training_dataset_') and f.endswith('.csv')]
    if not dataset_files:
        print("❌ No dataset found")
        return
    
    dataset_files.sort(reverse=True)
    latest_file = dataset_files[0]
    print(f"📁 Checking: {latest_file}")
    
    df = pd.read_csv(latest_file)
    
    # Check problematic columns
    problematic_columns = ['smoker', 'former_smoker', 'asthma', 'copd', 'heart_disease']
    
    for col in problematic_columns:
        if col in df.columns:
            unique_vals = df[col].dropna().unique()
            print(f"   {col}: {unique_vals[:10]}... (total unique: {len(unique_vals)})")
    
    # Show sample of problematic data
    print("\n🔍 Sample of problematic columns:")
    sample = df[problematic_columns].head(10)
    print(sample)
def predict_risk(self, pollution_data, user_profile):
    """Predict medical risk using our trained AI model"""
    try:
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Prepare features for prediction
        features = self._prepare_prediction_features(pollution_data, user_profile)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict using our AI model
        risk_score = self.model.predict(features_scaled)[0]
        
        return max(0, min(10, risk_score))  # Ensure score is between 0-10
        
    except Exception as e:
        print(f"AI Prediction error: {e}")
        raise

def _prepare_prediction_features(self, pollution_data, user_profile):
    """Prepare features for AI model prediction"""
    # Create feature vector in the same format as training
    feature_dict = {}
    
    # Pollution features
    pollutants = ['PM2.5', 'PM10', 'O3', 'CO', 'NO2', 'SO2']
    for pollutant in pollutants:
        feature_dict[pollutant] = pollution_data.get(pollutant, 0.0)
    
    # User profile features
    profile_features = [
        'age', 'smoker', 'former_smoker', 'outdoor_worker', 
        'regular_exercise', 'urban_resident', 'asthma', 'copd', 
        'heart_disease', 'hypertension', 'diabetes', 'pregnancy',
        'child_under_5', 'child_5_12', 'elderly_65_75', 'elderly_over_75'
    ]
    
    for feature in profile_features:
        feature_dict[feature] = user_profile.get(feature, 0)
    
    # Convert to DataFrame with same column order as training
    import pandas as pd
    features_df = pd.DataFrame([feature_dict])
    
    # Ensure all training features are present
    if hasattr(self, 'feature_columns'):
        for col in self.feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0
    
    return features_df[self.feature_columns] if hasattr(self, 'feature_columns') else features_df
if __name__ == "__main__":
    # First diagnose any data issues
    diagnose_data_issues()
    
    print("\n" + "="*50)
    print("🚀 Starting training with robust data handling...")
    print("="*50)
    
    trainer = MedicalAITrainer()
    model = trainer.train_medical_ai()
    
    if model is not None:
        print("\n✅ Training completed successfully!")
        print("📁 Next step: Run 'python app.py' to start the application")
    else:
        print("\n❌ Training failed.")