import streamlit as st
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from math import sqrt
import matplotlib.pyplot as plt
import math
from streamlit_folium import folium_static
import folium
from folium.plugins import MarkerCluster

# Define color and shape mappings for power plant types
source_colors = {
    'nuclear': ('blue', 'o'),
    'natural gas': ('green', 's'),
    'coal': ('black', 'D'),
    'hydroelectric': ('cyan', '^'),
    'petroleum': ('magenta', 'v'),
    'pumped storage': ('orange', '^'),
    'biomass': ('brown', 'P'),
    'other': ('gray', '*'),
    'solar': ('yellow', 'h'),
    'batteries': ('purple', 'o')
}

@st.cache_data
def load_geojson_data(path):
    """Load and cache GeoJSON data."""
    return gpd.read_file(path)

@st.cache_data
def filter_population_data(population_gdf, georgia_union, min_pop):
    """Filter population data based on minimum population."""
    filtered = population_gdf[population_gdf.intersects(georgia_union)].copy()
    high_pop_areas = filtered[filtered['POP2010'] > min_pop].copy()
    high_pop_areas['longitude'] = high_pop_areas['geometry'].centroid.x
    high_pop_areas['latitude'] = high_pop_areas['geometry'].centroid.y
    high_pop_areas['diameter'] = high_pop_areas['SQMI'].apply(lambda sqmi: sqrt(sqmi * 2.58999 / math.pi) * 2)
    return high_pop_areas

@st.cache_data
def filter_power_lines(power_lines_gdf, min_voltage):
    """Filter power lines based on voltage."""
    return power_lines_gdf[power_lines_gdf['VOLTAGE'] >= min_voltage]

@st.cache_data
def load_power_plants(power_plants_gdf, georgia_union):
    """Filter and load power plants within Georgia."""
    return power_plants_gdf[power_plants_gdf.geometry.intersects(georgia_union)]

@st.cache_data
def calculate_power_requirements(filtered_power_plants, max_estimated_consumption, scaling_factor):
    """Calculate the power requirements and distribute them to power plants."""
    Power_needed = max_estimated_consumption * scaling_factor * 1e6
    sorted_plants = filtered_power_plants.sort_values(by='Install_MW', ascending=False)
    remaining_power_needed = Power_needed / 1e6
    allocated_power_plants = {}

    for _, row in sorted_plants.iterrows():
        if remaining_power_needed > 0:
            plant_capacity = row['Install_MW']
            prim_source = row['PrimSource'].lower()
            if prim_source not in allocated_power_plants:
                allocated_power_plants[prim_source] = []
            allocated_power_plants[prim_source].append(row)
            remaining_power_needed -= plant_capacity

    return allocated_power_plants

# Streamlit UI and operations
st.sidebar.header("Filters")
min_population = st.sidebar.slider("Minimum Urban Area Population", 1000, 100000, 50000)
min_voltage = st.sidebar.slider("Minimum Voltage (kV)", 0, 500, 100)
scaling_factor = st.sidebar.slider("Power Installed Scaling Factor", 1.0, 5.0, 2.5, step=0.1)

# Paths to data
population_path = 'data/USA_Urban_Areas_(1%3A500k-1.5M).geojson'
georgia_boundary_path = 'data/georgia-counties.json'
power_lines_path = 'data/Transmission_Lines_all.geojson'
power_plants_path = 'data/Power_Plants_georgia.geojson'

# Load data
population_gdf = load_geojson_data(population_path)
georgia = load_geojson_data(georgia_boundary_path)
georgia_union = georgia.geometry.unary_union

high_pop_areas = filter_population_data(population_gdf, georgia_union, min_population)

power_lines_gdf = load_geojson_data(power_lines_path)
filtered_power_lines = filter_power_lines(power_lines_gdf, min_voltage)

power_plants_gdf = load_geojson_data(power_plants_path)
filtered_power_plants = load_power_plants(power_plants_gdf, georgia_union)

# Calculate power metrics
max_estimated_consumption = high_pop_areas['POP2010'].sum() * scaling_factor * 1e6
filtered_power_plant_data = calculate_power_requirements(filtered_power_plants, max_estimated_consumption, scaling_factor)

# Visualize data
st.title("Georgia Energy Infrastructure")

# Create Folium map
m = folium.Map(location=[32.8407, -83.6324], zoom_start=7)

# Add population areas
for _, area in high_pop_areas.iterrows():
    folium.Circle(
        location=(area['latitude'], area['longitude']),
        radius=area['diameter'] * 500,
        color='orange',
        fill=True,
        fill_opacity=0.4,
        tooltip=f"Urban Area: {area['POP2010']} people"
    ).add_to(m)

# Add power lines
for _, line in filtered_power_lines.iterrows():
    coords = list(line.geometry.coords)
    folium.PolyLine(coords, color='red', weight=1).add_to(m)

# Add power plants
for source, plants in filtered_power_plant_data.items():
    color, marker = source_colors.get(source, ('gray', 'o'))
    for plant in plants:
        folium.CircleMarker(
            location=(plant['Latitude'], plant['Longitude']),
            radius=5,
            color=color,
            fill=True,
            fill_opacity=0.8,
            tooltip=f"{plant['Plant_Name']} ({plant['Install_MW']} MW, {source.capitalize()})"
        ).add_to(m)

# Display map
folium_static(m)
