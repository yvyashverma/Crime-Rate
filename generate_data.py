import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_crime_data(num_records=10000):
    """
    Generate synthetic crime data for EDA
    """
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Define crime categories and their typical frequencies
    crime_categories = {
        'Theft': 0.25,
        'Assault': 0.15,
        'Burglary': 0.12,
        'Vandalism': 0.10,
        'Robbery': 0.08,
        'Drug Offense': 0.07,
        'Fraud': 0.06,
        'Vehicle Theft': 0.05,
        'Homicide': 0.02,
        'Sexual Assault': 0.03,
        'Domestic Violence': 0.04,
        'Arson': 0.03
    }
    
    # Define cities with population and crime rate factors
    cities = {
        'New York': {'population': 8400000, 'crime_factor': 0.9},
        'Los Angeles': {'population': 3800000, 'crime_factor': 1.1},
        'Chicago': {'population': 2700000, 'crime_factor': 1.3},
        'Houston': {'population': 2300000, 'crime_factor': 1.0},
        'Phoenix': {'population': 1600000, 'crime_factor': 0.8},
        'Philadelphia': {'population': 1580000, 'crime_factor': 1.2},
        'San Antonio': {'population': 1500000, 'crime_factor': 0.7},
        'San Diego': {'population': 1400000, 'crime_factor': 0.6},
        'Dallas': {'population': 1300000, 'crime_factor': 1.1},
        'San Jose': {'population': 1000000, 'crime_factor': 0.5}
    }
    
    # Define neighborhoods for each city
    neighborhoods = {}
    for city in cities.keys():
        num_neighborhoods = random.randint(5, 10)
        neighborhoods[city] = [f"{city} - Neighborhood {i+1}" for i in range(num_neighborhoods)]
    
    # Define time periods
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    # Generate data
    data = []
    
    for _ in range(num_records):
        # Select city based on population and crime factor
        city = random.choices(
            list(cities.keys()),
            weights=[cities[c]['population'] * cities[c]['crime_factor'] for c in cities.keys()]
        )[0]
        
        # Select neighborhood
        neighborhood = random.choice(neighborhoods[city])
        
        # Select crime type based on weights
        crime_type = random.choices(
            list(crime_categories.keys()),
            weights=list(crime_categories.values())
        )[0]
        
        # Generate date
        random_days = random.randint(0, (end_date - start_date).days)
        date = start_date + timedelta(days=random_days)
        
        # Generate time of day (weighted towards evening/night)
        hour_weights = [0.02, 0.02, 0.02, 0.02, 0.03, 0.04, 0.05, 0.06,
                       0.07, 0.06, 0.05, 0.04, 0.04, 0.04, 0.05, 0.06,
                       0.07, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02]
        hour = random.choices(range(24), weights=hour_weights)[0]
        minute = random.randint(0, 59)
        time_of_day = f"{hour:02d}:{minute:02d}"
        
        # Generate location coordinates (approximate within city bounds)
        # This is just for visualization, not real coordinates
        base_lat = {
            'New York': 40.71,
            'Los Angeles': 34.05,
            'Chicago': 41.88,
            'Houston': 29.76,
            'Phoenix': 33.45,
            'Philadelphia': 39.95,
            'San Antonio': 29.42,
            'San Diego': 32.72,
            'Dallas': 32.78,
            'San Jose': 37.34
        }[city]
        
        base_lon = {
            'New York': -74.01,
            'Los Angeles': -118.24,
            'Chicago': -87.63,
            'Houston': -95.37,
            'Phoenix': -112.07,
            'Philadelphia': -75.17,
            'San Antonio': -98.49,
            'San Diego': -117.16,
            'Dallas': -96.80,
            'San Jose': -121.89
        }[city]
        
        # Add some random variation
        latitude = base_lat + random.uniform(-0.1, 0.1)
        longitude = base_lon + random.uniform(-0.1, 0.1)
        
        # Generate additional attributes
        is_arrested = random.choices([True, False], weights=[0.3, 0.7])[0]
        domestic_violence = crime_type == 'Domestic Violence' or random.random() < 0.1
        
        # Generate victim demographics (simplified)
        victim_age = random.randint(0, 90) if random.random() > 0.2 else None
        victim_gender = random.choice(['Male', 'Female', 'Unknown']) if random.random() > 0.2 else None
        
        # Generate case status
        case_status = random.choices(
            ['Open', 'Closed', 'Pending', 'Archived'],
            weights=[0.2, 0.5, 0.2, 0.1]
        )[0]
        
        # Generate severity level (1-5)
        severity_map = {
            'Homicide': 5,
            'Sexual Assault': 5,
            'Robbery': 4,
            'Assault': 4,
            'Burglary': 3,
            'Vehicle Theft': 3,
            'Arson': 4,
            'Domestic Violence': 4,
            'Theft': 2,
            'Fraud': 2,
            'Vandalism': 2,
            'Drug Offense': 2
        }
        severity = severity_map.get(crime_type, random.randint(1, 3))
        
        # Generate description
        descriptions = [
            f"Report of {crime_type.lower()} at {time_of_day}",
            f"Suspect involved in {crime_type.lower()} incident",
            f"Victim reported {crime_type.lower()}",
            f"Police responded to {crime_type.lower()} call"
        ]
        description = random.choice(descriptions)
        
        # Create record
        record = {
            'crime_id': f"CR{str(_+1).zfill(6)}",
            'date': date.strftime('%Y-%m-%d'),
            'year': date.year,
            'month': date.month,
            'day': date.day,
            'day_of_week': date.strftime('%A'),
            'time': time_of_day,
            'hour': hour,
            'city': city,
            'neighborhood': neighborhood,
            'crime_type': crime_type,
            'severity': severity,
            'latitude': round(latitude, 4),
            'longitude': round(longitude, 4),
            'victim_age': victim_age,
            'victim_gender': victim_gender,
            'arrested': is_arrested,
            'domestic_violence': domestic_violence,
            'case_status': case_status,
            'description': description
        }
        
        data.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add derived columns
    df['month_name'] = pd.to_datetime(df['date']).dt.month_name()
    df['quarter'] = pd.to_datetime(df['date']).dt.quarter
    df['hour_category'] = pd.cut(df['hour'], 
                                 bins=[-1, 5, 11, 17, 23], 
                                 labels=['Late Night', 'Morning', 'Afternoon', 'Evening'])
    
    # Add season
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    df['season'] = df['month'].apply(get_season)
    
    return df

def save_data(df, filename='crime_data.csv'):
    """
    Save the generated data to CSV file
    """
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Crime types: {df['crime_type'].nunique()}")
    print(f"Cities: {df['city'].nunique()}")
    return filename

if __name__ == "__main__":
    # Generate and save data
    print("Generating synthetic crime data...")
    crime_df = generate_crime_data(10000)
    save_data(crime_df)
    
    # Display sample
    print("\nSample data:")
    print(crime_df.head())
    print("\nData Info:")
    print(crime_df.info())