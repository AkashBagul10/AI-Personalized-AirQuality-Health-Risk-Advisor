# monitor_database_fixed.py
import time
import sqlite3
from datetime import datetime
from app import app, db
from database import User, HealthProfile, SearchHistory

def monitor_database_live():
    print("📊 LIVE DATABASE MONITOR (FIXED)")
    print("=" * 50)
    print("Monitoring database changes in real-time...")
    print("Press Ctrl+C to stop monitoring")
    print("-" * 50)
    
    try:
        with app.app_context():
            while True:
                # Get counts using SQLAlchemy (proper way)
                users_count = User.query.count()
                profiles_count = HealthProfile.query.count()
                history_count = SearchHistory.query.count()
                
                # Get latest search
                latest_search = SearchHistory.query.order_by(SearchHistory.timestamp.desc()).first()
                
                # Clear screen and display
                print(f"\r🕒 {datetime.now().strftime('%H:%M:%S')} | 👥 Users: {users_count} | 🏥 Profiles: {profiles_count} | 📊 Searches: {history_count} | ", end="")
                
                if latest_search:
                    print(f"Latest: {latest_search.city} ({latest_search.risk_score:.1f}/10)", end="")
                else:
                    print("Latest: No searches yet", end="")
                
                time.sleep(2)  # Update every 2 seconds
                
    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring stopped")

if __name__ == "__main__":
    monitor_database_live()