# 🎯 **Project Overview**

A **production-ready, AI-powered web application** that provides **personalized health risk assessments** based on real-time air quality data for Indian cities. The system combines **medical science, environmental data, and machine learning** to deliver individualized health recommendations.

### **✨ Key Features**
- **🧠 Custom AI Model** - 94.5% accurate Random Forest trained on 221,275 medical records
- **📊 Real-time Pollution Data** - Multi-API integration (CPCB India, WAQI, OpenWeather)
- **🩺 Personalized Risk Assessment** - Based on age, medical conditions, and lifestyle
- **🎨 Professional UI/UX** - Modern dark mode, responsive design, medical-grade interface
- **🗄️ Database Integration** - SQLite with user profiles, health data, and search history
- **🔍 Search History** - Track and visualize all past assessments
🏗️ **System Architecture**
┌─────────────────┐ ┌──────────────────┐ ┌─────────────────┐
│ Frontend │ │ Backend │ │ Data Sources │
│ (HTML/CSS/JS) │◄──►│ Flask + SQL │◄──►│ • CPCB India │
│ Dark Mode UI │ │ SQLAlchemy │ │ • WAQI │
└─────────────────┘ └────────┬─────────┘ │ • OpenWeather │
│ └─────────────────┘
▼
┌──────────────────┐
│ AI Engine │
│ Random Forest │
│ 94.5% Accuracy │
│ 27 Features │
└──────────────────┘

---

## 🧠 **AI Model Details**

### **Model Architecture**
- **Algorithm:** Random Forest Regressor (100 decision trees)
- **Training Data:** 221,275 medical records
- **Features:** 27 medical & pollution factors
- **Accuracy:** 94.5% (R² = 0.9450)

### **Top Risk Factors Discovered by AI**
| Rank | Feature | Importance | Medical Significance |
|------|---------|------------|---------------------|
| 1 | PM2.5 | 30.67% | Fine particles penetrate deep into lungs |
| 2 | PM10 | 14.66% | Respiratory and cardiovascular issues |
| 3 | Smoking Status | 11.33% | Major lifestyle risk factor |
| 4 | PM2.5 × Asthma | 7.10% | AI-discovered medical interaction |
| 5 | NO2 | 5.56% | Traffic-related respiratory impact |

### **Risk Scale (0-10)**
- **0-3:** Low Risk - Generally safe for most activities
- **3-6:** Moderate Risk - Sensitive groups take precautions
- **6-8:** High Risk - Everyone reduce outdoor activities
- **8-10:** Very High Risk - Significant health threat

---

## 🚀 **Live Demo**

**Coming Soon** - Deployment in progress

---

## 📦 **Tech Stack**

### **Backend**
- **Python 3.8+** - Core programming language
- **Flask** - Web framework with RESTful API
- **SQLAlchemy** - ORM for database management
- **SQLite** - Production database

### **Machine Learning**
- **Scikit-learn** - Random Forest implementation
- **Pandas/NumPy** - Data processing
- **Joblib** - Model serialization

### **Frontend**
- **HTML5/CSS3** - Responsive dark mode UI
- **JavaScript** - Dynamic content and API calls
- **Font Awesome** - Medical-grade icons

### **APIs & Data Sources**
- **CPCB India** - Official government pollution data
- **WAQI** - World Air Quality Index
- **OpenWeather** - Global weather data

---

## 📊 **Database Schema**

```sql
Users (id, email, name, password, created_at)
HealthProfiles (user_id, age, lifestyle_factors, medical_conditions, vulnerable_groups)
SearchHistory (user_id, city, risk_score, pollution_data, timestamp)
ModelVersions (version, accuracy, features_count, training_date, is_active)

Installation
Clone the repository

bash
cd AirQuality-Health-Risk-Advisor

Create virtual environment

bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

Install dependencies

bash
pip install -r requirements.txt
Initialize database

bash
python migrate_database.py
Run the application

bash
python app.py

Open browser

text
http://127.0.0.1:5000

# Run database verification
python check_database.py

# Test database operations
python test_database_functionality.py

# Monitor database in real-time
python monitor_database_fixed.py
