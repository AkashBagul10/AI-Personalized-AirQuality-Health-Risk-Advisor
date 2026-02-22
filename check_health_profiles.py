# check_health_profiles.py
from app import app, db
from database import User, HealthProfile
import json

def check_all_health_profiles():
    with app.app_context():
        print("🏥 ALL HEALTH PROFILES IN DATABASE")
        print("=" * 50)
        
        profiles = HealthProfile.query.all()
        
        if not profiles:
            print("❌ No health profiles found in database")
            return
        
        for profile in profiles:
            user = User.query.get(profile.user_id)
            print(f"\n👤 User: {user.name} ({user.email})")
            print(f"📅 Age: {profile.age}")
            
            # Lifestyle factors
            lifestyle = json.loads(profile.lifestyle_factors)
            active_lifestyle = [k for k, v in lifestyle.items() if v]
            if active_lifestyle:
                print(f"💪 Lifestyle: {', '.join(active_lifestyle)}")
            else:
                print("💪 Lifestyle: None selected")
            
            # Medical conditions
            medical = json.loads(profile.medical_conditions)
            active_medical = [k for k, v in medical.items() if v]
            if active_medical:
                print(f"❤️  Medical: {', '.join(active_medical)}")
            else:
                print("❤️  Medical: None selected")
            
            # Vulnerable groups
            vulnerable = json.loads(profile.vulnerable_groups)
            active_vulnerable = [k for k, v in vulnerable.items() if v]
            if active_vulnerable:
                print(f"👥 Vulnerable: {', '.join(active_vulnerable)}")
            else:
                print("👥 Vulnerable: None selected")
            
            print(f"🕒 Last Updated: {profile.updated_at}")

if __name__ == "__main__":
    check_all_health_profiles()