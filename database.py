# database.py
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from datetime import datetime
import json

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    health_profile = db.relationship('HealthProfile', backref='user', uselist=False, cascade='all, delete-orphan')
    search_history = db.relationship('SearchHistory', backref='user', cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'name': self.name,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class HealthProfile(db.Model):
    __tablename__ = 'health_profiles'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Basic Information
    age = db.Column(db.Integer, nullable=False)
    
    # Lifestyle Factors (store as JSON for flexibility)
    lifestyle_factors = db.Column(db.Text, default='{}')  # JSON string
    
    # Medical Conditions (store as JSON for flexibility)
    medical_conditions = db.Column(db.Text, default='{}')  # JSON string
    
    # Vulnerable Groups (store as JSON for flexibility)
    vulnerable_groups = db.Column(db.Text, default='{}')  # JSON string
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'age': self.age,
            'lifestyle_factors': json.loads(self.lifestyle_factors),
            'medical_conditions': json.loads(self.medical_conditions),
            'vulnerable_groups': json.loads(self.vulnerable_groups),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    def get_combined_profile(self):
        """Combine all profile data into a single dictionary for AI model"""
        profile = {'age': self.age}
        
        # Add lifestyle factors
        lifestyle = json.loads(self.lifestyle_factors)
        profile.update(lifestyle)
        
        # Add medical conditions
        medical = json.loads(self.medical_conditions)
        profile.update(medical)
        
        # Add vulnerable groups
        vulnerable = json.loads(self.vulnerable_groups)
        profile.update(vulnerable)
        
        return profile

class SearchHistory(db.Model):
    __tablename__ = 'search_history'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Search details
    city = db.Column(db.String(100), nullable=False)
    risk_score = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(20), nullable=False)
    
    # Pollution data (store as JSON)
    pollution_data = db.Column(db.Text, nullable=False)
    
    # AI model info
    model_used = db.Column(db.String(100))
    features_used = db.Column(db.Integer)
    
    # Source info
    data_source = db.Column(db.String(100))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'city': self.city,
            'risk_score': self.risk_score,
            'risk_level': self.risk_level,
            'pollution_data': json.loads(self.pollution_data),
            'model_used': self.model_used,
            'features_used': self.features_used,
            'data_source': self.data_source,
            'timestamp': self.timestamp.isoformat()
        }

class ModelVersion(db.Model):
    __tablename__ = 'model_versions'
    
    id = db.Column(db.Integer, primary_key=True)
    version = db.Column(db.String(50), nullable=False)
    model_path = db.Column(db.String(200), nullable=False)
    accuracy = db.Column(db.Float)
    features_count = db.Column(db.Integer)
    training_date = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    description = db.Column(db.Text)
    
    def to_dict(self):
        return {
            'id': self.id,
            'version': self.version,
            'model_path': self.model_path,
            'accuracy': self.accuracy,
            'features_count': self.features_count,
            'training_date': self.training_date.isoformat(),
            'is_active': self.is_active,
            'description': self.description
        }

def init_db(app):
    """Initialize the database with the Flask app"""
    # Configure SQLite database
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///air_quality.db"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    
    # Initialize the database
    db.init_app(app)
    
    # Create tables
    with app.app_context():
        db.create_all()
        print("✅ Database tables created successfully!")
        
        # Add default model version if none exists
        if not ModelVersion.query.first():
            default_model = ModelVersion(
                version="1.0",
                model_path="models/medical_ai_model.pkl",
                accuracy=0.945,
                features_count=27,
                description="Initial Random Forest model trained on 221,275 records"
            )
            db.session.add(default_model)
            db.session.commit()
            print("✅ Default model version added!")