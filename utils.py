import numpy as np
import random
import math

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    R = 6371  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def generate_wind_data():
    """
    Generates a random wind vector (speed in km/h, direction in degrees).
    """
    speed = random.uniform(5, 40) # 5 to 40 km/h
    direction = random.uniform(0, 360) # 0 to 360 degrees
    return speed, direction

def predict_drift(center_lat, center_lon, wind_speed, wind_direction, hours=24):
    """
    Predicts the future location of the spill center based on wind.
    Simple rule of thumb: Oil moves at ~3% of wind speed.
    """
    drift_speed = wind_speed * 0.03 # 3% wind factor
    distance_km = drift_speed * hours
    
    # Calculate new coordinates (simplified flat earth for small distances)
    # 1 deg lat ~ 111 km
    d_lat = (distance_km * math.cos(math.radians(wind_direction))) / 111
    d_lon = (distance_km * math.sin(math.radians(wind_direction))) / (111 * math.cos(math.radians(center_lat)))
    
    return center_lat + d_lat, center_lon + d_lon

def get_nearby_ais_ships(center_lat, center_lon, radius_km=50):
    """
    Generates mock AIS ship data near the spill.
    Returns a list of dicts: {'id', 'lat', 'lon', 'name', 'type'}
    """
    ships = []
    num_ships = random.randint(1, 5)
    
    types = ['Tanker', 'Cargo', 'Fishing', 'Passenger']
    
    for i in range(num_ships):
        # Random position within radius
        angle = random.uniform(0, 2*math.pi)
        dist = random.uniform(0, radius_km)
        
        d_lat = (dist * math.cos(angle)) / 111
        d_lon = (dist * math.sin(angle)) / (111 * math.cos(math.radians(center_lat)))
        
        ships.append({
            'id': f"MMSI-{random.randint(200000000, 700000000)}",
            'name': f"Vessel-{chr(65+i)}{random.randint(10,99)}",
            'type': random.choice(types),
            'lat': center_lat + d_lat,
            'lon': center_lon + d_lon
        })
    return ships

def check_environmental_vulnerability(center_lat, center_lon):
    """
    Checks if the spill is near any (mock) sensitive areas.
    """
    # Mock database of sensitive areas
    sensitive_areas = [
        {'name': 'Coral Reef Sanctuary', 'lat': center_lat + 0.05, 'lon': center_lon - 0.05, 'radius': 10},
        {'name': 'Mangrove Forest', 'lat': center_lat - 0.08, 'lon': center_lon + 0.02, 'radius': 15},
        {'name': 'fictional_beach', 'lat': center_lat + 0.1, 'lon': center_lon + 0.1, 'radius': 5}
    ]
    
    alerts = []
    for area in sensitive_areas:
        dist = haversine_distance(center_lat, center_lon, area['lat'], area['lon'])
        if dist < area['radius'] + 5: # +5 buffer
            alerts.append(f"CLOSE to {area['name']} ({dist:.1f} km)")
            
    return alerts
