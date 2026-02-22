# check_database.py
from app import app, db
from database import User, HealthProfile, SearchHistory, ModelVersion
from datetime import datetime
import json

def check_database():
    with app.app_context():
        print("🔍 DATABASE VERIFICATION")
        print("=" * 50)
        
        # Check if database file exists
        import os
        if os.path.exists('air_quality.db'):
            print("✅ Database file: air_quality.db (FOUND)")
            file_size = os.path.getsize('air_quality.db')
            print(f"📊 Database size: {file_size} bytes")
        else:
            print("❌ Database file: air_quality.db (NOT FOUND)")
            return
        
        # Count records in each table
        try:
            users_count = User.query.count()
            profiles_count = HealthProfile.query.count()
            history_count = SearchHistory.query.count()
            models_count = ModelVersion.query.count()
            
            print(f"👥 Users: {users_count}")
            print(f"🏥 Health Profiles: {profiles_count}")
            print(f"📊 Search History: {history_count}")
            print(f"🤖 Model Versions: {models_count}")
            
            # Show sample data
            print("\n📋 SAMPLE DATA:")
            print("-" * 30)
            
            # Show users
            users = User.query.all()
            for user in users[:3]:  # Show first 3 users
                print(f"User: {user.name} ({user.email})")
                
                # Show their profile
                profile = HealthProfile.query.filter_by(user_id=user.id).first()
                if profile:
                    print(f"  Profile: Age {profile.age}")
                    lifestyle = json.loads(profile.lifestyle_factors)
                    medical = json.loads(profile.medical_conditions)
                    print(f"  Lifestyle: {lifestyle}")
                    print(f"  Medical: {medical}")
                
                # Show their search history
                searches = SearchHistory.query.filter_by(user_id=user.id).all()
                print(f"  Searches: {len(searches)} records")
                for search in searches[:2]:  # Show first 2 searches
                    print(f"    - {search.city}: {search.risk_score}/10 ({search.risk_level})")
            
            print("\n✅ Database is working correctly!")
            
        except Exception as e:
            print(f"❌ Database error: {e}")

if __name__ == "__main__":
    check_database()