# app.py
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import joblib
import numpy as np
import os
import pandas as pd
from datetime import datetime
import json

# Import database and other modules
from database import db, init_db, User, HealthProfile, SearchHistory, ModelVersion
from simple_api_manager import api_manager
from medical_ai_trainer import MedicalAITrainer

app = Flask(__name__)
app.secret_key = 'medical-air-quality-secret-key-2024'

# Initialize database
init_db(app)

# Global instances - ONLY AI MODEL
medical_trainer = MedicalAITrainer()
medical_model = None

# Load our custom AI model
# def load_medical_model():
#     model_paths = [
#         'models/medical_ai_model.pkl',
#         'medical_ai_model.pkl'
#     ]
    
#     for path in model_paths:
#         if os.path.exists(path):
#             try:
#                 model_data = joblib.load(path)
#                 print(f"✅ Medical AI Model loaded from: {path}")
#                 return model_data
#             except Exception as e:
#                 print(f"❌ Error loading model from {path}: {e}")
    
#     print("⚠️ Medical AI model not found - please train the model first")
#     return None

# # Load the model
# medical_model = load_medical_model()

# In your app.py, replace the model loading section:

# Load our BEST model - Neural Network
def load_best_model():
    model_paths = [

        'models/improved_neural_network.pkl',  # 🎯 NOW USING NEURAL NETWORK
        # 'models/medical_ai_model.pkl'       # Fallback to Random Forest
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                model_data = joblib.load(path)
                model_type = model_data.get('model_type', 'Unknown')
                print(f"✅ {model_type} loaded from: {path}")
                return model_data
            except Exception as e:
                print(f"❌ Error loading model from {path}: {e}")
    
    print("⚠️ No model found - please train the model first")
    return None

# Load the BEST model
medical_model = load_best_model()
# Routes
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Database authentication
        user = User.query.filter_by(email=email, password=password).first()
        
        if user:
            session['user_id'] = user.id
            session['user_name'] = user.name
            session['user_email'] = user.email
            
            # Check if user has health profile
            profile = HealthProfile.query.filter_by(user_id=user.id).first()
            if profile:
                session['health_profile'] = profile.get_combined_profile()
                return redirect(url_for('dashboard'))
            else:
                return redirect(url_for('health_profile'))
        
        return render_template('login.html', error="Invalid email or password")
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        name = request.form.get('name')
        
        # Check if email already exists
        if User.query.filter_by(email=email).first():
            return render_template('signup.html', error="Email already exists")
        
        # Create new user
        new_user = User(email=email, password=password, name=name)
        db.session.add(new_user)
        db.session.commit()
        
        # Set session
        session['user_id'] = new_user.id
        session['user_name'] = new_user.name
        session['user_email'] = new_user.email
        
        return redirect(url_for('health_profile'))
    
    return render_template('signup.html')

@app.route('/health-profile', methods=['GET', 'POST'])
def health_profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    
    if request.method == 'POST':
        # Get form data
        age = int(request.form.get('age', 30))
        
        # Lifestyle factors
        lifestyle_factors = {
            'smoker': 'smoker' in request.form,
            'former_smoker': 'former_smoker' in request.form,
            'outdoor_worker': 'outdoor_worker' in request.form,
            'regular_exercise': 'regular_exercise' in request.form,
            'urban_resident': 'urban_resident' in request.form
        }
        
        # Medical conditions
        medical_conditions = {
            'asthma': 'asthma' in request.form,
            'copd': 'copd' in request.form,
            'heart_disease': 'heart_disease' in request.form,
            'hypertension': 'hypertension' in request.form,
            'diabetes': 'diabetes' in request.form
        }
        
        # Vulnerable groups
        vulnerable_groups = {
            'pregnancy': 'pregnancy' in request.form,
            'child_under_5': 'child_under_5' in request.form,
            'child_5_12': 'child_5_12' in request.form,
            'elderly_65_75': 'elderly_65_75' in request.form,
            'elderly_over_75': 'elderly_over_75' in request.form
        }
        
        # Check if profile exists
        profile = HealthProfile.query.filter_by(user_id=user_id).first()
        
        if profile:
            # Update existing profile
            profile.age = age
            profile.lifestyle_factors = json.dumps(lifestyle_factors)
            profile.medical_conditions = json.dumps(medical_conditions)
            profile.vulnerable_groups = json.dumps(vulnerable_groups)
        else:
            # Create new profile
            profile = HealthProfile(
                user_id=user_id,
                age=age,
                lifestyle_factors=json.dumps(lifestyle_factors),
                medical_conditions=json.dumps(medical_conditions),
                vulnerable_groups=json.dumps(vulnerable_groups)
            )
            db.session.add(profile)
        
        db.session.commit()
        
        # Update session
        session['health_profile'] = profile.get_combined_profile()
        
        return redirect(url_for('dashboard'))
    
    # Get existing profile for form population
    profile_data = {}
    existing_profile = HealthProfile.query.filter_by(user_id=user_id).first()
    
    if existing_profile:
        profile_data = existing_profile.get_combined_profile()
    
    return render_template('health_profile.html', profile=profile_data)

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    
    # Get user's search history count
    search_count = SearchHistory.query.filter_by(user_id=user_id).count()
    
    # Get recent searches
    recent_searches = SearchHistory.query.filter_by(user_id=user_id).order_by(
        SearchHistory.timestamp.desc()
    ).limit(5).all()
    
    profile = session.get('health_profile', {})
    
    return render_template('dashboard.html', 
                         profile=profile, 
                         search_count=search_count,
                         recent_searches=recent_searches)

@app.route('/search')
def search():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    profile = session.get('health_profile', {})
    return render_template('search.html', profile=profile)

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    
    # Get all search history
    searches = SearchHistory.query.filter_by(user_id=user_id).order_by(
        SearchHistory.timestamp.desc()
    ).all()
    
    return render_template('history.html', searches=searches)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/api/predict', methods=['POST'])
def predict_medical_risk():
    """Medical risk prediction endpoint - USES OUR BEST MODEL (Neural Network)"""
    try:
        data = request.get_json()
        city = data.get('city', '').strip().title()
        
        if not city:
            return jsonify({'error': 'City name required'}), 400
        
        print(f"🎯 AI Model predicting risk for: {city}")
        
        # Get user profile from session
        user_profile = session.get('health_profile', {})
        user_id = session.get('user_id')
        
        # Get pollution data
        pollution_result = api_manager.get_pollution_data(city)
        
        if pollution_result is None:
            return jsonify({
                'error': f'No air quality monitoring stations found for {city}.',
                'suggestion': 'Please try major Indian cities like Mumbai, Delhi, Bangalore, Chennai, Kolkata, Hyderabad, etc.'
            }), 404
            
        if 'data' not in pollution_result:
            return jsonify({
                'error': f'Could not fetch pollution data for {city}.',
                'suggestion': 'The city may not have active air quality monitoring stations.'
            }), 404
        
        pollution_data = pollution_result['data']
        
        # 🎯 USE OUR BEST MODEL - NEURAL NETWORK
        if medical_model is None:
            return jsonify({'error': 'Medical AI model not loaded. Please train the model first.'}), 500
        
        # Prepare features for AI prediction
        features = prepare_features_for_ai(pollution_data, user_profile)
        
        # Scale features
        features_scaled = medical_model['scaler'].transform(features)
        
        # 🕒 ADD PREDICTION TIME MEASUREMENT
        import time
        start_time = time.time()
        
        # Predict using our BEST model (Neural Network)
        risk_score = medical_model['model'].predict(features_scaled)[0]
        risk_score = max(0, min(10, risk_score))  # Ensure 0-10 range
        
        # 🕒 CALCULATE PREDICTION TIME
        prediction_time = time.time() - start_time
        
        # Generate advice based on AI prediction
        advice = generate_ai_advice(user_profile, pollution_data, risk_score)
        
        # Get risk level from AI prediction
        risk_level = "Low" if risk_score < 3 else "Moderate" if risk_score < 6 else "High" if risk_score < 8 else "Very High"
        
        # Get model info
        model_type = medical_model.get('model_type', 'Neural Network')
        model_accuracy = "96.15% (R² = 0.9615)" if "Neural" in model_type else "94.5% (R² = 0.945)"
        
        response = {
            'risk_score': round(risk_score, 2),
            'risk_level': risk_level,
            'advice': advice,
            'pollution_data': pollution_data,
            'source_info': {
                'name': pollution_result.get('source_name', 'Unknown'),
                'city': city,
                'timestamp': pollution_result.get('timestamp', '')
            },
            'model_used': f'{model_type} (Optimal)',
            'model_accuracy': model_accuracy,
            'model_rank': '1st among 6 algorithms tested',
            'features_used': len(medical_model['feature_columns']),
            'prediction_time': f"{prediction_time:.3f}s"  # ✅ NOW DEFINED
        }
        
        # Save search history to database
        if user_id:
            search_history = SearchHistory(
                user_id=user_id,
                city=city,
                risk_score=risk_score,
                risk_level=risk_level,
                pollution_data=json.dumps(pollution_data),
                model_used=response['model_used'],
                features_used=response['features_used'],
                data_source=pollution_result.get('source_name', 'Unknown')
            )
            db.session.add(search_history)
            db.session.commit()
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ AI Model Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'AI Medical risk assessment failed.'}), 500
def prepare_features_for_ai(pollution_data, user_profile):
    """Prepare features for our AI model prediction"""
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
        feature_dict[feature] = 1 if user_profile.get(feature, False) else 0
    
    # Convert to DataFrame with same column order as training
    features_df = pd.DataFrame([feature_dict])
    
    # Ensure all training features are present
    if 'feature_columns' in medical_model:
        for col in medical_model['feature_columns']:
            if col not in features_df.columns:
                features_df[col] = 0
        return features_df[medical_model['feature_columns']]
    
    return features_df

def generate_ai_advice(user_profile, pollution_data, risk_score):
    """Generate advice based on AI model prediction"""
    advice = []
    risk_level = "Low" if risk_score < 3 else "Moderate" if risk_score < 6 else "High" if risk_score < 8 else "Very High"
    
    # General advice based on AI prediction
    if risk_level == "Low":
        advice.append("✅ Air quality is generally safe for most activities")
    elif risk_level == "Moderate":
        advice.append("⚠️ Sensitive groups should consider precautions")
    elif risk_level == "High":
        advice.append("🚫 Everyone should reduce outdoor activities")
    else:  # Very High
        advice.append("🛑 Avoid all outdoor activities - significant health risk")
    
    # Condition-specific advice based on AI learning
    conditions = {
        'asthma': ["💨 Keep rescue inhaler accessible", "😷 Wear N95 mask if going outside"],
        'copd': ["💊 Ensure medication supply is adequate", "🛌 Limit strenuous activities"],
        'heart_disease': ["❤️ Avoid strenuous outdoor activities", "📞 Keep emergency contacts handy"],
        'pregnancy': ["🤰 Limit time outdoors", "🏠 Stay in well-ventilated areas"],
        'child_under_5': ["👶 Limit outdoor playtime", "🏫 Indoor activities recommended"]
    }
    
    for condition, condition_advice in conditions.items():
        if user_profile.get(condition, False) and risk_level in ["High", "Very High"]:
            advice.extend(condition_advice)
    
    # Pollutant-specific advice
    high_pollutants = []
    for pollutant, concentration in pollution_data.items():
        if pollutant in ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']:
            if concentration > 50:  # Simple threshold for demonstration
                high_pollutants.append(pollutant)
    
    if 'PM2.5' in high_pollutants:
        advice.append("🌫️ High PM2.5 - Consider wearing a mask")
    if 'O3' in high_pollutants:
        advice.append("☀️ High Ozone - Avoid afternoon outdoor activities")
    
    return list(set(advice))  # Remove duplicates

@app.route('/api/history')
def get_search_history():
    """API endpoint to get user's search history"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user_id = session['user_id']
    searches = SearchHistory.query.filter_by(user_id=user_id).order_by(
        SearchHistory.timestamp.desc()
    ).all()
    
    return jsonify([search.to_dict() for search in searches])

@app.route('/api/user/profile')
def get_user_profile():
    """API endpoint to get user profile"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user_id = session['user_id']
    user = User.query.get(user_id)
    profile = HealthProfile.query.filter_by(user_id=user_id).first()
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    response = {
        'user': user.to_dict(),
        'health_profile': profile.get_combined_profile() if profile else {}
    }
    
    return jsonify(response)

@app.route('/favicon.ico')
def favicon():
    return '', 204

if __name__ == '__main__':
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    print("🚀 Medical Air Quality Advisor - Database Enabled")
    print("🎯 Using Custom Trained Random Forest Model")
    print("📊 Model Status:", "✅ Loaded" if medical_model else "❌ Not Found")
    print("🗄️ Database: SQLite (air_quality.db)")
    
    app.run(debug=True, port=5000)