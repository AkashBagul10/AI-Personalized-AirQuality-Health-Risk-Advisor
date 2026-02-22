import requests
import json
from datetime import datetime

class SimpleAPIManager:
    def __init__(self):
        self.sources = {
            'openweather': {
                'name': 'OpenWeather Map',
                'priority': 1,
                'active': True,
                'api_key': ''
            },
            'waqi': {
                'name': 'World Air Quality Index',
                'priority': 2, 
                'active': True,
                'api_key': ''
            },
            'cpcb': {
                'name': 'CPCB India',
                'priority': 3,
                'active': True,
                'api_key': ''
            }
        }
    def _validate_waqi_data_quality(self, data, city):
        """Simple WAQI data quality validation"""
        # Basic sanity checks
        if data.get('PM2.5', 0) > 500:  # PM2.5 over 500 is garbage
            return False
        if data.get('PM10', 0) > 1000:  # PM10 over 1000 is garbage  
            return False
        # Check for all zeros
        if all(v == 0 for v in [data.get('PM2.5'), data.get('PM10'), data.get('O3')]):
            return False
        return True
    def get_pollution_data(self, city):
        """Get pollution data - NO FALLBACKS, fail properly if no data"""
        print(f"🌍 Fetching pollution data for: {city}")
        
        # Try all available sources
        for source_name in ['cpcb', 'waqi', 'openweather']:
            source_info = self.sources[source_name]
            if not source_info['active']:
                continue
                
            try:
                print(f"🔄 Trying {source_name}...")
                data = getattr(self, f'_get_{source_name}_data')(city)
                if data and self._validate_pollution_data(data):
                    print(f"✅ Using {source_name} data for {city}")
                    return {
                        'data': data,
                        'source': source_name,
                        'source_name': source_info['name'],
                        'timestamp': datetime.now().isoformat(),
                        'units': 'μg/m³',
                        'api_used': source_name
                    }
            except Exception as e:
                print(f"❌ {source_name} failed: {e}")
                continue
        
        # 🚨 CRITICAL FIX: NO FALLBACK - Return None if no data
        print(f"❌ No pollution data available for {city} from any API")
        return None

    def _get_waqi_data(self, city):
        """Get WAQI data - WITH COMPREHENSIVE QUALITY CHECKS"""
        try:
            api_key = self.sources['waqi']['api_key']
            
            # Search for station
            search_url = "https://api.waqi.info/search/"
            search_params = {
                "token": api_key,
                "keyword": f"{city} india"
            }
            
            search_response = requests.get(search_url, params=search_params, timeout=10)
            search_data = search_response.json()
            
            if search_data.get('status') != 'ok' or not search_data.get('data'):
                print(f"❌ WAQI: No stations found for {city}")
                return None
            
            stations = search_data['data']
            best_station = None
            
            # 🎯 STRICT CITY MATCHING
            city_lower = city.lower()
            for station in stations:
                station_name = station.get('station', {}).get('name', '').lower()
                
                # Check if city name appears in station name
                if city_lower in station_name:
                    best_station = station
                    print(f"📍 WAQI City Match: {station_name}")
                    break
            
            # 🚨 REJECT if no proper city match
            if not best_station:
                print(f"❌ WAQI: No station properly matches {city}")
                return None
            
            station_uid = best_station['uid']
            
            # Get station feed
            feed_url = f"https://api.waqi.info/feed/@{station_uid}/"
            feed_params = {"token": api_key}
            
            feed_response = requests.get(feed_url, params=feed_params, timeout=10)
            feed_data = feed_response.json()
            
            if feed_data.get('status') != 'ok':
                return None
            
            iaqi = feed_data.get('data', {}).get('iaqi', {})
            
            pollution_data = {
                'PM2.5': iaqi.get('pm25', {}).get('v', 0),
                'PM10': iaqi.get('pm10', {}).get('v', 0),
                'O3': iaqi.get('o3', {}).get('v', 0),
                'NO2': iaqi.get('no2', {}).get('v', 0),
                'SO2': iaqi.get('so2', {}).get('v', 0),
                'CO': iaqi.get('co', {}).get('v', 0),
                'AQI': feed_data.get('data', {}).get('aqi', 0)
            }
            
            print(f"📊 WAQI Raw: {pollution_data}")
            
            # 🎯 CRITICAL: ADD WAQI-SPECIFIC QUALITY CHECKS
            if not self._validate_waqi_data_quality(pollution_data, city):
                print(f"❌ WAQI: Data quality check failed for {city}")
                return None
            
            return pollution_data
            
        except Exception as e:
            print(f"WAQI API error: {e}")
            return None

        
    def _get_openweather_data(self, city):
        """Get OpenWeather data with better error handling"""
        try:
            api_key = self.sources['openweather']['api_key']
            
            # Geocode city
            geocode_url = "http://api.openweathermap.org/geo/1.0/direct"
            geocode_params = {
                "q": f"{city},IN",
                "limit": 1,
                "appid": api_key
            }
            
            geocode_response = requests.get(geocode_url, params=geocode_params, timeout=10)
            geocode_data = geocode_response.json()
            
            if not geocode_data:
                print(f"❌ OpenWeather: City {city} not found")
                return None
                
            lat, lon = geocode_data[0]['lat'], geocode_data[0]['lon']
            print(f"📍 OpenWeather Coordinates: {lat}, {lon}")
            
            # Get pollution data
            pollution_url = "http://api.openweathermap.org/data/2.5/air_pollution"
            pollution_params = {
                "lat": lat,
                "lon": lon, 
                "appid": api_key
            }
            
            pollution_response = requests.get(pollution_url, params=pollution_params, timeout=10)
            pollution_data = pollution_response.json()
            
            if 'list' not in pollution_data or not pollution_data['list']:
                print("❌ OpenWeather: No pollution data")
                return None
            
            # Extract components
            components = pollution_data['list'][0].get('components', {})
            
            pollution_data = {
                'PM2.5': components.get('pm2_5', 0),
                'PM10': components.get('pm10', 0),
                'O3': components.get('o3', 0),
                'NO2': components.get('no2', 0),
                'SO2': components.get('so2', 0),
                'CO': components.get('co', 0),  # μg/m³
                'AQI': pollution_data['list'][0].get('main', {}).get('aqi', 0)
            }
            
            print(f"📊 OpenWeather Raw: {pollution_data}")
            return pollution_data
            
        except Exception as e:
            print(f"OpenWeather API error: {e}")
            return None

    def _get_cpcb_data(self, city):
        """Get CPCB data - return None if no stations found"""
        try:
            api_key = self.sources['cpcb']['api_key']
            url = "https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
            
            params = {
                "api-key": api_key,
                "format": "json",
                "filters[city]": city.title(),
                "limit": 100
            }
            
            response = requests.get(url, params=params, timeout=15)
            data = response.json()
            
            if not data or 'records' not in data or not data['records']:
                print(f"❌ CPCB: No monitoring stations found for {city}")
                return None
        
            records = data['records']
            print(f"📍 CPCB found {len(records)} stations for {city}")
            pollutant_readings = {"PM2.5": [], "PM10": [], "O3": [], "CO": [], "NO2": [], "SO2": []}
            
            for record in records:
                pollutant_id = record.get('pollutant_id')
                if pollutant_id == "OZONE":
                    pollutant_id = "O3"
                
                if pollutant_id in pollutant_readings:
                    value_str = record.get('avg_value')
                    try:
                        value = float(value_str)
                        if 0 < value < 1000:  # Reasonable range check
                            pollutant_readings[pollutant_id].append(value)
                    except (ValueError, TypeError):
                        continue
            
            # Calculate averages
            avg_data = {}
            for key, values in pollutant_readings.items():
                avg_data[key] = round(sum(values) / len(values), 2) if values else 0
            
            avg_data["AQI"] = 0
            print(f"📊 CPCB Processed: {avg_data}")
            return avg_data
            
        except Exception as e:
            print(f"CPCB API error: {e}")
            return None

    def _get_realistic_city_data(self, city):
        """Fallback to realistic city data based on averages"""
        print(f"⚠️ Using realistic data for {city}")
        
        # City-specific realistic averages based on typical Indian data
        city_profiles = {
            'mumbai': {'PM2.5': 27.0, 'PM10': 36.0, 'O3': 30.0, 'NO2': 20.0, 'SO2': 10.0, 'CO': 18.0},
            'delhi': {'PM2.5': 95.0, 'PM10': 180.0, 'O3': 45.0, 'NO2': 60.0, 'SO2': 25.0, 'CO': 1500},
            'bangalore': {'PM2.5': 35.0, 'PM10': 65.0, 'O3': 25.0, 'NO2': 30.0, 'SO2': 12.0, 'CO': 600},
            'chennai': {'PM2.5': 45.0, 'PM10': 75.0, 'O3': 28.0, 'NO2': 35.0, 'SO2': 15.0, 'CO': 700},
            'kolkata': {'PM2.5': 65.0, 'PM10': 110.0, 'O3': 35.0, 'NO2': 45.0, 'SO2': 18.0, 'CO': 900}
        }
        
        city_lower = city.lower()
        for city_key, data in city_profiles.items():
            if city_key in city_lower:
                return {
                    'data': data,
                    'source': 'realistic_fallback',
                    'source_name': f'Realistic {city.title()} Average',
                    'timestamp': datetime.now().isoformat(),
                    'units': 'μg/m³',
                    'api_used': 'fallback'
                }
        
        # Default Indian average
        default_data = {'PM2.5': 50.0, 'PM10': 85.0, 'O3': 32.0, 'NO2': 35.0, 'SO2': 15.0, 'CO': 800}
        return {
            'data': default_data,
            'source': 'realistic_fallback', 
            'source_name': 'Realistic Indian Average',
            'timestamp': datetime.now().isoformat(),
            'units': 'μg/m³',
            'api_used': 'fallback'
        }

    def _validate_pollution_data(self, data):
        """Better validation to catch garbage data"""
        required_pollutants = ['PM2.5', 'PM10', 'O3', 'CO', 'NO2', 'SO2']
        
        for pollutant in required_pollutants:
            if pollutant not in data:
                print(f"❌ Missing pollutant: {pollutant}")
                return False
            
            value = data[pollutant]
            if not isinstance(value, (int, float)):
                print(f"❌ Invalid type for {pollutant}: {type(value)}")
                return False
            
            # Reasonable range checks
            reasonable_ranges = {
                'PM2.5': (0, 500),
                'PM10': (0, 600),
                'O3': (0, 200),
                'NO2': (0, 200),
                'SO2': (0, 200),
                'CO': (0, 5000)
            }
            
            if pollutant in reasonable_ranges:
                min_val, max_val = reasonable_ranges[pollutant]
                if value < min_val or value > max_val:
                    print(f"❌ Unreasonable {pollutant}: {value}")
                    return False
        
        print("✅ Data validation passed")
        return True

# Global instance
api_manager = SimpleAPIManager()
