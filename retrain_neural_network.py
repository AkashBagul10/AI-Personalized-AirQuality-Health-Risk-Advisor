# retrain_neural_network.py
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ImprovedNeuralNetworkTrainer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def load_and_prepare_data(self):
        """Load and prepare data properly"""
        dataset_files = [f for f in os.listdir('.') 
                        if f.startswith(('kaggle_training_dataset_', 'synthetic_large_')) 
                        and f.endswith('.csv')]
        
        if not dataset_files:
            raise FileNotFoundError("No dataset found!")
        
        dataset_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        dataset_path = dataset_files[0]
        
        print(f"📥 Loading dataset: {dataset_path}")
        df = pd.read_csv(dataset_path, low_memory=False)
        print(f"✅ Loaded {len(df):,} records")
        
        return self.prepare_features_with_interactions(df)
    
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
        
        # Robust data cleaning
        for col in available_features + ['medical_risk_score']:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
        
        # Remove extreme outliers from target variable
        Q1 = df_clean['medical_risk_score'].quantile(0.01)
        Q3 = df_clean['medical_risk_score'].quantile(0.99)
        df_clean = df_clean[(df_clean['medical_risk_score'] >= Q1) & (df_clean['medical_risk_score'] <= Q3)]
        
        X = df_clean[available_features]
        y = df_clean['medical_risk_score']
        
        # Create interaction features
        X = self.create_interaction_features(X)
        self.feature_columns = list(X.columns)
        
        print(f"✅ Final features: {len(self.feature_columns)}")
        print(f"🎯 Target range after cleaning: {y.min():.2f} to {y.max():.2f}")
        
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
    
    def train_improved_neural_network(self):
        """Train improved Neural Network with better parameters"""
        print("\n" + "="*60)
        print("🚀 TRAINING IMPROVED NEURAL NETWORK")
        print("="*60)
        
        X, y, feature_names = self.load_and_prepare_data()
        
        print(f"📊 Dataset shape: {X.shape}")
        print(f"🎯 Target statistics - Mean: {y.mean():.2f}, Std: {y.std():.2f}")
        
        # Better train-test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=pd.cut(y, bins=5)
        )
        
        print(f"📚 Training samples: {X_train.shape[0]:,}")
        print(f"🧪 Testing samples: {X_test.shape[0]:,}")
        
        print("⚖️ Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # IMPROVED NEURAL NETWORK ARCHITECTURE
        print("🧠 Building IMPROVED Neural Network...")
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),  # SMALLER network to prevent overfitting
            activation='relu',
            solver='adam',
            alpha=0.01,           # STRONGER regularization
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=1000,        # More iterations
            early_stopping=True,
            validation_fraction=0.15,  # Larger validation set
            n_iter_no_change=25,  # More patience
            random_state=42,
            verbose=True
        )
        
        print("🤖 Training Improved Neural Network...")
        print("⏳ This will take 5-15 minutes...")
        
        self.model.fit(X_train_scaled, y_train)
        
        print("\n📈 Evaluating Improved Neural Network...")
        self.evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test, feature_names)
        
        self.save_model()
        return self.model
    
    def evaluate_model(self, X_train_scaled, X_test_scaled, y_train, y_test, feature_names):
        """Comprehensive evaluation"""
        print("⏳ Making predictions...")
        
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Ensure predictions are in realistic range
        y_train_pred = np.clip(y_train_pred, 0, 10)
        y_test_pred = np.clip(y_test_pred, 0, 10)
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        print("\n" + "="*50)
        print("🎯 IMPROVED NEURAL NETWORK RESULTS")
        print("="*50)
        print(f"🏗️ Architecture: {self.model.hidden_layer_sizes}")
        print(f"📚 Training iterations: {self.model.n_iter_}")
        print(f"📚 Final loss: {self.model.loss_:.4f}")
        print(f"📚 Training R²:   {train_r2:.4f}")
        print(f"🧪 Testing R²:    {test_r2:.4f}")
        print(f"📚 Training RMSE: {train_rmse:.4f}")
        print(f"🧪 Testing RMSE:  {test_rmse:.4f}")
        
        # Analyze prediction distribution
        print(f"\n📊 Prediction Distribution Analysis:")
        print(f"   Training predictions - Min: {y_train_pred.min():.2f}, Max: {y_train_pred.max():.2f}, Mean: {y_train_pred.mean():.2f}")
        print(f"   Test predictions - Min: {y_test_pred.min():.2f}, Max: {y_test_pred.max():.2f}, Mean: {y_test_pred.mean():.2f}")
        print(f"   Actual values - Min: {y_test.min():.2f}, Max: {y_test.max():.2f}, Mean: {y_test.mean():.2f}")
        
        # Performance interpretation
        if test_r2 > 0.94:
            print("🏆 EXCELLENT performance with realistic scores!")
        elif test_r2 > 0.90:
            print("💪 VERY GOOD performance!")
        else:
            print("⚠️ Performance needs improvement.")
    
    def save_model(self):
        """Save the trained model"""
        os.makedirs('models', exist_ok=True)
        
        model_package = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'training_date': datetime.now().isoformat(),
            'model_type': 'Improved_NeuralNetwork_MLP',
            'version': '2.0'
        }
        
        model_path = 'models/improved_neural_network.pkl'
        joblib.dump(model_package, model_path)
        print(f"\n💾 Improved Neural Network saved to: {model_path}")
        
        # Also save as the main neural network model
        main_path = 'models/neural_network_model.pkl'
        joblib.dump(model_package, main_path)
        print(f"💾 Also saved as main model: {main_path}")
        
        print("🎉 Ready for testing!")

def quick_test_model():
    """Quick test to verify the model works"""
    print("\n🧪 QUICK MODEL VERIFICATION")
    print("="*30)
    
    # Load the newly trained model
    model_data = joblib.load('models/improved_neural_network.pkl')
    model = model_data['model']
    scaler = model_data['scaler']
    
    # Create a test case: Healthy person in moderate pollution
    test_features = {
        'PM2.5': 35, 'PM10': 60, 'O3': 30, 'CO': 500, 'NO2': 20, 'SO2': 10,
        'age': 30, 'bmi': 24,
        'smoker': 0, 'former_smoker': 0, 'outdoor_worker': 0, 'regular_exercise': 1, 'urban_resident': 1,
        'asthma': 0, 'copd': 0, 'heart_disease': 0, 'hypertension': 0, 'diabetes': 0,
        'pregnancy': 0, 'child_under_5': 0, 'child_5_12': 0, 'elderly_65_75': 0, 'elderly_over_75': 0
    }
    
    # Add interaction features
    test_features['pm25_asthma'] = test_features['PM2.5'] * test_features['asthma']
    test_features['o3_child'] = test_features['O3'] * test_features['child_under_5']
    test_features['pm25_elderly'] = test_features['PM2.5'] * test_features['elderly_over_75']
    test_features['co_heart'] = test_features['CO'] * test_features['heart_disease']
    
    # Convert to array and scale
    feature_array = np.array([list(test_features.values())])
    scaled_features = scaler.transform(feature_array)
    
    # Predict
    prediction = model.predict(scaled_features)[0]
    prediction = max(0, min(10, prediction))
    
    print(f"✅ Test prediction: {prediction:.2f}/10")
    print("   (Healthy person, moderate pollution - should be 3-5/10)")
    
    return prediction

if __name__ == "__main__":
    print("🚀 Retraining Neural Network with Improved Configuration")
    print("🎯 Goal: Realistic risk scores without over-prediction")
    
    trainer = ImprovedNeuralNetworkTrainer()
    model = trainer.train_improved_neural_network()
    
    if model:
        print("\n✅ Improved Neural Network training completed!")
        
        # Quick test
        test_score = quick_test_model()
        
        if 3 <= test_score <= 5:
            print("🎉 SUCCESS: Model producing realistic scores!")
        else:
            print("⚠️ WARNING: Scores may still need adjustment")
        
        print("\n📁 Next steps:")
        print("1. Run: python app.py to test the new model")
        print("2. Test various cities and health profiles")
        print("3. Verify scores are realistic (2-8 range typically)")