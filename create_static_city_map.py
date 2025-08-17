import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import os
from geopy.geocoders import Nominatim
import time
import json
from geopy.distance import geodesic

# Read data
# df = pd.read_excel('travel_data_722.xlsx')
df = pd.read_excel('travel_data_3836.xlsx')
df_2024 = df[df['Year'] == 2024]

# Calculate passenger count by city
city_stats = df_2024.groupby('Origin')['No. Pax'].sum().sort_values(ascending=False)
print(f"2024 data: {len(city_stats)} cities, total {city_stats.sum()} passengers")
print("\nTop 20 cities:")
print(city_stats.head(20))

# Show country distribution
countries = df_2024['Origin Country'].value_counts()
print(f"\nCountry distribution ({len(countries)} countries):")
print(countries.head(10))

# Check for unmapped cities
all_cities = set(city_stats.index)
print(f"\nAll {len(all_cities)} cities in data")

# City coordinates mapping
city_coordinates = {
    # Chinese cities
    'Beijing': [116.4074, 39.9042],
    'Shanghai': [121.4737, 31.2304],
    'Hangzhou': [120.1614, 30.2936], 
    'Hong Kong': [114.1694, 22.3193],
    'Shenzhen': [114.0579, 22.5431],
    'Guangzhou': [113.2644, 23.1291],
    'Nanjing': [118.7969, 32.0603],
    'Chengdu': [104.0648, 30.5728],
    'Xi\'an': [108.9540, 34.2658],
    'Wuhan': [114.2919, 30.5844],
    'Tianjin': [117.3616, 39.1189],
    'Changsha': [112.9388, 28.2282],
    'Harbin': [126.5363, 45.8038],
    'Hefei': [117.2461, 31.8206],
    'Changchun': [125.3245, 43.8868],
    'Jinan': [117.0009, 36.6758],
    'Xiamen': [118.1689, 24.4797],
    'Macau': [113.5491, 22.1987],
    
    # US cities
    'Mountain View': [-122.0838, 37.3861],
    'San Francisco': [-122.4194, 37.7749],
    'Los Angeles': [-122.2711, 34.0522],
    'Seattle': [-122.3321, 47.6062],
    'New York': [-74.0060, 40.7128],
    'Boston': [-71.0589, 42.3601],
    'Chicago': [-87.6298, 41.8781],
    'Austin': [-97.7431, 30.2672],
    'San Diego': [-117.1611, 32.7157],
    'Stanford': [-122.1660, 37.4419],
    'Berkeley': [-122.2585, 37.8719],
    'Cambridge': [-71.1097, 42.3736],
    'Cambridge, MA': [-71.1097, 42.3736],
    'Princeton': [-74.6672, 40.3573],
    'Pasadena': [-118.1445, 34.1478],
    'Pittsburgh': [-79.9959, 40.4406],
    'Philadelphia': [-75.1652, 39.9526],
    'Atlanta': [-84.3880, 33.7490],
    'Houston': [-95.3698, 29.7604],
    'Ann Arbor': [-83.7430, 42.2808],
    'Madison': [-89.4012, 43.0731],
    'Irvine': [-117.8265, 33.6846],
    'Davis': [-121.7405, 38.5449],
    'Santa Barbara': [-119.6982, 34.4208],
    'Santa Clara': [-121.9552, 37.3541],
    'Santa Cruz': [-122.0308, 36.9741],
    'Cupertino': [-122.0322, 37.3230],
    'Menlo Park': [-122.1817, 37.4530],
    'San Jose': [-121.8863, 37.3382],
    'Durham': [-78.8986, 35.9940],
    'Chapel Hill': [-79.0558, 35.9132],
    'Raleigh': [-78.6382, 35.7796],
    'Minneapolis': [-93.2650, 44.9778],
    'Orlando': [-81.3792, 28.5383],
    'Baltimore': [-76.6122, 39.2904],
    'Providence': [-71.4128, 41.8240],
    'New Haven': [-72.9279, 41.3083],
    'Buffalo': [-78.8784, 42.8864],
    'Rochester': [-77.6088, 43.1566],
    'Columbus': [-82.9988, 39.9612],
    'Ithaca': [-76.5019, 42.4430],
    'College Park': [-76.9378, 38.9897],
    'College Station': [-96.3344, 30.6280],
    'State College': [-77.8600, 40.7934],
    'West Lafayette': [-87.0073, 40.4259],
    'Evanston': [-87.6900, 42.0451],
    'St. Louis': [-90.1994, 38.6270],
    'Charlottesville': [-78.4767, 38.0293],
    'Notre Dame': [-86.2379, 41.7001],
    'East Lansing': [-84.4839, 42.3370],
    'Blacksburg': [-80.4139, 37.2296],
    'Stony Brook': [-73.1408, 40.9247],
    'Troy': [-73.6918, 42.7284],
    'Piscataway': [-74.4049, 40.5554],
    'Champaign': [-88.2434, 40.1164],
    'Amherst': [-72.5201, 42.3732],
    'Armonk': [-73.7137, 41.1287],
    'Redmond': [-122.1215, 47.6740],
    'Merced': [-120.4829, 37.3022],
    'Tempe': [-111.9401, 33.4255],
    
    # Canadian cities
    'Toronto': [-79.3832, 43.6532],
    'Montreal': [-73.5673, 45.5017],
    'Vancouver': [-123.1207, 49.2827],
    'Edmonton': [-113.4909, 53.5461],
    'Waterloo': [-80.5164, 43.4643],
    'Quebec City': [-71.2080, 46.8139],
    'Surrey': [-122.8447, 49.1913],
    
    # UK cities
    'London': [-0.1276, 51.5074],
    'Oxford': [-1.2577, 51.7520],
    'Cambridge': [0.1218, 52.2053],
    'Edinburgh': [-3.1883, 55.9533],
    'Bristol': [-2.5879, 51.4545],
    
    # Other European cities
    'Amsterdam': [4.9041, 52.3676],
    'Stockholm': [18.0686, 59.3293],
    'Copenhagen': [12.5683, 55.6761],
    'Zurich': [8.5417, 47.3769],
    'Munich': [11.5820, 48.1351],
    'Darmstadt': [8.6511, 49.8728],
    'Tübingen': [9.0576, 48.5216],
    'Freiburg': [7.8421, 47.9990],
    'Lausanne': [6.6323, 46.5197],
    'Delft': [4.3571, 52.0116],
    'Eindhoven': [5.4697, 51.4416],
    'Athens': [23.7275, 37.9755],
    'Palaiseau': [2.2472, 48.7140],
    'Sophia Antipolis': [7.0682, 43.6177],
    'Espoo': [24.6559, 60.2055],
    
    # Asian cities
    'Tokyo': [139.6917, 35.6895],
    'Saitama': [139.6566, 35.8617],
    'Seoul': [126.9780, 37.5665],
    'Daejeon': [127.3849, 36.3504],
    'Pohang': [129.3656, 36.0190],
    'Taipei': [121.5654, 25.0330],
    'Hsinchu': [120.9647, 24.8138],
    'Singapore': [103.8198, 1.3521],
    
    # Australian cities
    'Sydney': [151.2093, -33.8688],
    'Melbourne': [144.9631, -37.8136],
    'Canberra': [149.1300, -35.2809],
    'Adelaide': [138.6007, -34.9285],
    
    # Middle East cities
    'Tel Aviv': [34.7818, 32.0853],
    'Jerusalem': [35.2137, 31.7683],
    'Haifa': [34.9896, 32.7940],
    'Abu Dhabi': [54.3773, 24.2992],
    'Thuwal': [39.1025, 22.2783],
    
    # Russian cities
    'Moscow': [37.6173, 55.7558],
    'Dolgoprudny': [37.5114, 55.9344],
    'Skolkovo': [37.3606, 55.6983],
    
    # Special case for unknown location (use Singapore as placeholder)
    'Unknown': [103.8198, 1.3521]
}

# Persistent cache for city coordinates
cache_file = 'city_coordinates_cache.json'
if os.path.exists(cache_file):
    with open(cache_file, 'r') as f:
        city_coordinates.update(json.load(f))

# Check for unmapped cities
mapped_cities = set(city_coordinates.keys())
unmapped_cities = all_cities - mapped_cities
if unmapped_cities:
    print(f"Warning: {len(unmapped_cities)} cities without coordinates: {list(unmapped_cities)}")
else:
    print("All cities have coordinate mappings!")

# Automatically fill missing city coordinates
geolocator = Nominatim(user_agent="city_mapper")
def get_city_coord(city_name):
    try:
        location = geolocator.geocode(city_name)
        if location:
            return [location.longitude, location.latitude]
    except Exception as e:
        print(f"Geocoding error for {city_name}: {e}")
    # fallback: use Singapore's coordinates
    return city_coordinates['Singapore']

new_coords = False
for city in unmapped_cities:
    if city not in city_coordinates:
        coord = get_city_coord(city)
        city_coordinates[city] = coord
        print(f"Auto-mapped {city}: {coord}")
        time.sleep(0.5)  # avoid hitting geocoding rate limits
        new_coords = True

# Save updated city_coordinates to cache
if new_coords:
    with open(cache_file, 'w') as f:
        json.dump(city_coordinates, f, indent=2)

# Prepare data for plotting
cities_to_plot = []
passengers_to_plot = []
coordinates_to_plot = []

for city, passengers in city_stats.items():
    # if city in city_coordinates and city != 'Singapore':
    cities_to_plot.append(city)
    passengers_to_plot.append(passengers)
    coordinates_to_plot.append(city_coordinates[city])

print(f"\nMapped {len(cities_to_plot)} city coordinates")


shp_path = 'ne_10m_admin_0_countries.shp'
world = gpd.read_file(shp_path)


fig, ax = plt.subplots(figsize=(20, 12))

world.plot(ax=ax, color='white', edgecolor='#CCCCCC', linewidth=0.5, zorder=1)

# Create a custom colormap from blue to white
# colors = ['#1E90FF', '#87CEEB', '#E1FFFF']
colors = ['#E1FFFF', '#87CEEB', '#1E90FF']
n_bins = 100

cmap = LinearSegmentedColormap.from_list('custom_blue', colors, N=n_bins)

# Plot cities as scatter points
x_coords = [coord[0] for coord in coordinates_to_plot]
y_coords = [coord[1] for coord in coordinates_to_plot]

import math
print([i for i in passengers_to_plot])
passengers_to_plot = [math.log(i,1.2) if i > 0 else 0 for i in passengers_to_plot]
# Normalize passenger counts for size (min 50, max 400) - increased size range
min_passengers = min(passengers_to_plot)
max_passengers = max(passengers_to_plot)
adjusted_max = max_passengers * 0.5
sizes = [50 + (p - min_passengers) / (max_passengers - min_passengers) * 350 for p in passengers_to_plot]

# Create scatter plot with size representing passenger count
scatter = ax.scatter(x_coords, y_coords, 
                    s=sizes, 
                    c=passengers_to_plot, 
                    cmap=cmap, 
                    alpha=0.9, 
                    edgecolors='darkblue', 
                    linewidth=2,
                    zorder=3)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=30)
cbar.set_label('Number of Passengers', color='black', fontsize=12)
cbar.ax.tick_params(colors='black')
cbar.ax.yaxis.label.set_color('black')

# Add title
plt.title('Neurips 2024 City Distribution\nCity size represents number of passengers', 
          color='black', fontsize=20, fontweight='bold', pad=20)

# Add subtitle with statistics
# subtitle = f"{len(cities_to_plot)} cities from {len(countries)} countries, total {city_stats.sum():,} passengers"
# plt.figtext(0.5, 0.92, subtitle, ha='center', color='#4682B4', fontsize=14)

# City labels removed per user request

ax.set_facecolor('white')
plt.gcf().set_facecolor('white')
ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()
plt.savefig('static_city_map_2024.png', 
            dpi=300, 
            bbox_inches='tight', 
            facecolor='white',
            edgecolor='none')

print(f"\nStatic map saved as static_city_map_2024.png")
# print(f"Displayed {len(cities_to_plot)} cities from {len(countries)} countries, total {city_stats.sum():,} passengers")

# Vancouver Convention Centre coordinates
vancouver_coord = (-123.1150, 49.2880)
emission_factor = 0.158  # kg CO2 per person per km

total_co2 = 0
for city, passengers in city_stats.items():
    city_coord = city_coordinates[city]
    city_latlon = (city_coord[1], city_coord[0])
    distance_km = geodesic(city_latlon, (vancouver_coord[1], vancouver_coord[0])).km
    co2 = passengers * distance_km * emission_factor * 2 
    total_co2 += co2

# Vancouver 2021 annual GHG emissions: 2,500,000 tCO2e (City of Vancouver official report)
vancouver_monthly = 208_333  # tCO2e per month

total_co2_tonnes = total_co2 / 1000  # convert kg to tCO2e
day_vancouver = total_co2_tonnes * 30 / vancouver_monthly

print(f"Total CO2 emissions for all attendees flying to Vancouver: {total_co2_tonnes:,.0f} tCO2e")
print(f"This is equivalent to {day_vancouver:.2f} days of Vancouver's total CO2 emissions.")

# Suzhou monthly emission
suzhou_monthly = 416_666_667  # kg
day_suzhou = total_co2 * 30 / suzhou_monthly
print(f"This is equivalent to {day_suzhou:.2f} days of Suzhou's total CO2 emissions.")

plt.show() 
