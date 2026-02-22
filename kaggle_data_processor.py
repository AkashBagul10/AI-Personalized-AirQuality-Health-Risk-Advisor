import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class KaggleDataProcessor:
    def __init__(self):
        self.large_datasets = {
            'india_air_quality': 'city_day.csv',  # Your downloaded file
        }
    
    def load_and_process_kaggle_data(self):
        """Process the large Kaggle dataset"""
        print("📥 Loading Kaggle Indian Air Quality Dataset...")
        
        try:
            # Load the main dataset
            df = pd.read_csv(self.large_datasets['india_air_quality'])
            print(f"✅ Loaded {len(df):,} records from Kaggle dataset")
            
            # Basic cleaning
            df_clean = self.clean_kaggle_data(df)
            
            # Create training dataset with health profiles
            training_data = self.create_training_dataset(df_clean)
            
            return training_data
            
        except FileNotFoundError:
            print("❌ Kaggle dataset not found. Please ensure 'city_day.csv' is in the project folder.")
            return self.create_large_synthetic_dataset(50000)
    
    def clean_kaggle_data(self, df):
        """Clean and prepare the Kaggle data"""
        print("🧹 Cleaning Kaggle data...")
        
        # Convert date
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Remove invalid dates
        df = df[df['Date'].notna()]
        
        # Handle missing values
        pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
        
        for pollutant in pollutants:
            if pollutant in df.columns:
                # Fill missing values with city averages
                city_means = df.groupby('City')[pollutant].transform('mean')
                df[pollutant] = df[pollutant].fillna(city_means)
        
        # Remove cities with too much missing data
        city_counts = df['City'].value_counts()
        valid_cities = city_counts[city_counts > 50].index
        df = df[df['City'].isin(valid_cities)]
        
        print(f"✅ Cleaned data: {len(df):,} records, {df['City'].nunique()} cities")
        return df
    
    def create_training_dataset(self, pollution_df):
        """Create training dataset by combining real pollution with synthetic health profiles"""
        print("🏥 Creating training dataset with health profiles...")
        
        training_records = []
        
        # For each pollution record, create multiple health profiles
        for idx, row in pollution_df.iterrows():
            if idx % 1000 == 0:
                print(f"📊 Processing record {idx}/{len(pollution_df)}...")
            
            # Create 5-10 health profiles for this pollution scenario
            num_profiles = np.random.randint(5, 11)
            
            for _ in range(num_profiles):
                health_profile = self.generate_health_profile(row['City'])
                
                # Combine pollution and health data
                record = {
                    'city': row['City'].lower(),
                    'date': row['Date'],
                    'PM2.5': row.get('PM2.5', 0),
                    'PM10': row.get('PM10', 0),
                    'NO2': row.get('NO2', 0),
                    'SO2': row.get('SO2', 0),
                    'CO': row.get('CO', 0),
                    'O3': row.get('O3', 0),
                    **health_profile
                }
                
                # Calculate risk score
                from medical_risk_calculator import medical_risk_calculator
                pollution_data = {
                    'PM2.5': record['PM2.5'],
                    'PM10': record['PM10'], 
                    'NO2': record['NO2'],
                    'SO2': record['SO2'],
                    'CO': record['CO'],
                    'O3': record['O3']
                }
                
                risk_score = medical_risk_calculator.calculate_individual_risk(
                    health_profile, pollution_data
                )
                record['medical_risk_score'] = risk_score
                
                training_records.append(record)
        
        df = pd.DataFrame(training_records)
        print(f"✅ Created training dataset with {len(df):,} records")
        return df
    
    def generate_health_profile(self, city_name):
        """Generate realistic health profile based on city"""
        from enhanced_data_generator import EnhancedDataGenerator
        generator = EnhancedDataGenerator()
        
        # Map city to region for regional health patterns
        region_map = {
            'Delhi': 'north', 'Mumbai': 'west', 'Bengaluru': 'south', 
            'Kolkata': 'east', 'Chennai': 'south', 'Hyderabad': 'south',
            'Ahmedabad': 'west', 'Pune': 'west', 'Jaipur': 'north',
            'Lucknow': 'north', 'Patna': 'east', 'Surat': 'west',
            'Kochi': 'south', 'Chandigarh': 'north', 'Bhopal': 'central'
        }
        
        region = region_map.get(city_name, 'north')
        return generator.generate_indian_health_profile(region)
    
    def create_large_synthetic_dataset(self, num_records=50000):
        """Create a large synthetic dataset"""
        print(f"🔄 Creating large synthetic dataset ({num_records:,} records)...")
        
        # Indian cities with realistic pollution profiles
        indian_cities = {
            'delhi': {'PM2.5': (80, 300), 'PM10': (120, 400), 'region': 'north'},
            'mumbai': {'PM2.5': (60, 180), 'PM10': (90, 250), 'region': 'west'},
            'bangalore': {'PM2.5': (30, 100), 'PM10': (50, 150), 'region': 'south'},
            'kolkata': {'PM2.5': (70, 220), 'PM10': (100, 300), 'region': 'east'},
            'chennai': {'PM2.5': (40, 120), 'PM10': (70, 180), 'region': 'south'},
            'hyderabad': {'PM2.5': (45, 130), 'PM10': (75, 190), 'region': 'south'},
            'ahmedabad': {'PM2.5': (65, 200), 'PM10': (95, 270), 'region': 'west'},
            'pune': {'PM2.5': (35, 110), 'PM10': (60, 160), 'region': 'west'},
            'jaipur': {'PM2.5': (55, 170), 'PM10': (85, 230), 'region': 'north'},
            'lucknow': {'PM2.5': (75, 250), 'PM10': (110, 320), 'region': 'north'}
        }
        
        records = []
        for i in range(num_records):
            if i % 10000 == 0:
                print(f"📊 Generated {i:,} records...")
            
            # Select random city
            city_name = np.random.choice(list(indian_cities.keys()))
            city_profile = indian_cities[city_name]
            
            # Generate realistic pollution data
            pollution = self.generate_realistic_pollution(city_profile)
            
            # Generate health profile
            health_profile = self.generate_health_profile(city_name.title())
            
            # Combine into record
            record = {
                'city': city_name,
                'region': city_profile['region'],
                'timestamp': self.generate_timestamp(),
                **pollution,
                **health_profile
            }
            
            # Calculate risk score
            from medical_risk_calculator import medical_risk_calculator
            risk_score = medical_risk_calculator.calculate_individual_risk(
                health_profile, pollution
            )
            record['medical_risk_score'] = risk_score
            
            records.append(record)
        
        df = pd.DataFrame(records)
        print(f"✅ Created synthetic dataset with {len(df):,} records")
        return df
    
    def generate_realistic_pollution(self, city_profile):
        """Generate realistic pollution data for a city"""
        pollution = {}
        
        # Base levels with seasonal variation
        base_pm25 = np.random.normal(
            (city_profile['PM2.5'][0] + city_profile['PM2.5'][1]) / 2,
            (city_profile['PM2.5'][1] - city_profile['PM2.5'][0]) / 4
        )
        
        # Ensure within realistic bounds
        pollution['PM2.5'] = max(city_profile['PM2.5'][0], 
                               min(city_profile['PM2.5'][1], base_pm25))
        
        # Other pollutants correlated with PM2.5
        pollution['PM10'] = pollution['PM2.5'] * 1.5 + np.random.normal(0, 10)
        pollution['NO2'] = pollution['PM2.5'] * 0.6 + np.random.normal(0, 5)
        pollution['SO2'] = pollution['PM2.5'] * 0.3 + np.random.normal(0, 3)
        pollution['O3'] = np.random.normal(30, 15)
        pollution['CO'] = pollution['PM2.5'] * 10 + np.random.normal(0, 50)
        
        # Round to 1 decimal place
        for key in pollution:
            pollution[key] = round(max(0, pollution[key]), 1)
        
        return pollution
    
    def generate_timestamp(self):
        """Generate random timestamp within last 3 years"""
        current_time = datetime.now()
        random_days = np.random.randint(0, 3 * 365)
        random_seconds = np.random.randint(0, 24 * 60 * 60)
        
        random_date = current_time - timedelta(days=random_days, seconds=random_seconds)
        return random_date.isoformat()

# Fixed main execution
if __name__ == "__main__":
    processor = KaggleDataProcessor()
    
    print("🚀 Large Dataset Creator")
    print("1. Use Kaggle dataset (if downloaded)")
    print("2. Create large synthetic dataset (50K records)")
    print("3. Create very large synthetic dataset (100K records)")
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        data = processor.load_and_process_kaggle_data()
        filename = f"kaggle_training_dataset_{datetime.now().strftime('%Y%m%d')}.csv"
    elif choice == "2":
        data = processor.create_large_synthetic_dataset(50000)
        filename = f"synthetic_large_50k_{datetime.now().strftime('%Y%m%d')}.csv"
    elif choice == "3":
        data = processor.create_large_synthetic_dataset(100000)
        filename = f"synthetic_large_100k_{datetime.now().strftime('%Y%m%d')}.csv"
    else:
        print("❌ Invalid choice")
        exit()
    
    if data is not None:
        data.to_csv(filename, index=False)
        print(f"💾 Saved {len(data):,} records to {filename}")
        print("📊 Dataset Summary:")
        print(f"   Records: {len(data):,}")
        print(f"   Cities: {data['city'].nunique()}")
        print(f"   Risk Score Range: {data['medical_risk_score'].min():.1f} - {data['medical_risk_score'].max():.1f}")
        print(f"   Average Risk: {data['medical_risk_score'].mean():.2f}")
        
        # Show sample of the data
        print("\n🔍 Sample of the dataset:")
        print(data[['city', 'PM2.5', 'age', 'asthma', 'medical_risk_score']].head())