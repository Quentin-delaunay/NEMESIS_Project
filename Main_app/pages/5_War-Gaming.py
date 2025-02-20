import streamlit as st
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from shapely.geometry import Point
from math import sqrt
import os
import pandas as pd
import networkx as nx
import matplotlib.lines as mlines
import folium
from folium.plugins import HeatMap
from shapely.geometry import Point, LineString
import math
from streamlit_folium import folium_static, st_folium
from utils import create_network_from_filtered_data
import networkx as nx
import pandapower as pp
import plotly.graph_objects as go
import matplotlib.cm as cm
import matplotlib.colors as mcolors

st.set_page_config(page_title="War-Gaming", layout="wide")
st.title("War-Gaming")

BASELINE = {
    'MIN_POPULATION': 50000, #This makes the filter take look at all the urban areas
    'MAX_POWER_PEAK': 15636, # MW Georgia Power IRP prediction for 2024
    'POWER_INSTALLED': 37786, # MW Generated in Georgia in total
    'MIN_VOLTAGE': 250# looks at only major transmission lines
}
electricity_imports= 2806.17899 #MW/hr or 24,582,128 MW a year
electricity_exports = 0 #MW/hr or 0MW a year

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

# Function to calculate distance between two points (Haversine formula)
def haversine(lon1, lat1, lon2, lat2):
    R = 6371  # Earth radius in kilometers
    dlon = math.radians(lon2 - lon1)
    dlat = math.radians(lon2 - lat1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

GA_energy_consumption = {
    'Coal': 180.9,
    'NaturalGas': 812.4,
    'Electricity_in': 355.4 + 10.8 + 250.3 + 25.7,
    'Electricity_out': 262.2
}  # in BTU
total_consumption = sum(GA_energy_consumption.values())
Power_installed = BASELINE['POWER_INSTALLED']
max_peak_power = BASELINE['MAX_POWER_PEAK']
Elec_per_CP_2024 = Power_installed * max_peak_power * 1e6 / 11e6  # MW

# Function to create the baseline map
def create_baseline_map():
    # Load population data
    population_path = r'data/USA_Urban_Areas_(1%3A500k-1.5M).geojson'
    population_gdf = gpd.read_file(population_path)
    population_gdf['geometry'] = population_gdf['geometry'].buffer(0)

    # Load Georgia boundary data
    georgia_boundary_path = 'data/georgia-counties.json'
    georgia = gpd.read_file(georgia_boundary_path)
    MAX_POP = 10e6
    georgia['geometry'] = georgia['geometry'].buffer(0)
    georgia_union = georgia.geometry.unary_union

    # Filter urban areas
    urban_in_georgia = population_gdf[population_gdf.intersects(georgia_union)].copy()
    high_pop_areas = urban_in_georgia[urban_in_georgia['POP2010'] > BASELINE['MIN_POPULATION']].copy()
    high_pop_areas['longitude'] = high_pop_areas['geometry'].centroid.x
    high_pop_areas['latitude'] = high_pop_areas['geometry'].centroid.y
    high_pop_areas['diameter'] = high_pop_areas['SQMI'].apply(lambda sqmi: sqrt(sqmi * 2.58999 / 3.14159) * 2)

    # Total filtered population
    total_population = high_pop_areas['POP2010'].sum()

    # Calculate maximum estimated power consumption based on filtered population
    max_estimated_consumption = (max_peak_power/MAX_POP) * (total_population)  # in MW

    # Load pipelines, power lines, and plants
    gas_pipeline_path = r'data/NaturalGas_InterIntrastate_Pipelines_US_georgia.geojson'
    gas_pipeline = gpd.read_file(gas_pipeline_path)
    gas_pipeline = gas_pipeline[gas_pipeline.geometry.intersects(georgia_union)]

    power_lines_path = r'data/Transmission_Lines_all.geojson'
    power_lines = gpd.read_file(power_lines_path)
    power_lines = power_lines[power_lines['VOLTAGE'] >= BASELINE['MIN_VOLTAGE']]

    power_plants_path = r'data/Power_Plants_georgia.geojson'
    power_plants = gpd.read_file(power_plants_path)

    # Calculate required power
    Power_needed = max_estimated_consumption * Power_installed * 1e6
    sorted_plants = power_plants.sort_values(by='Install_MW', ascending=False)
    remaining_power_needed = Power_needed / 1e6
    filtered_power_plants = {}
    for _, row in sorted_plants.iterrows():
        if remaining_power_needed > 0:
            plant_capacity = row['Install_MW']
            prim_source = row['PrimSource'].lower()
            if prim_source not in filtered_power_plants:
                filtered_power_plants[prim_source] = []
            filtered_power_plants[prim_source].append(row)
            remaining_power_needed -= plant_capacity

    # Installed power for filtered power plants
    installed_power = sum(
        sum(plant['Install_MW'] for plant in plants)
        for plants in filtered_power_plants.values()
    )

    # Extract substations and create edges
    substations = []
    for _, line in power_lines.iterrows():
        coords = list(line['geometry'].coords)
        substations.append(Point(coords[0]))
        substations.append(Point(coords[-1]))
    substations_gdf = gpd.GeoDataFrame(geometry=substations, crs=power_lines.crs)
    substations_gdf = substations_gdf.drop_duplicates(subset=['geometry'], keep='first').reset_index(drop=True)

    edges = []
    for _, line in power_lines.iterrows():
        coords = list(line['geometry'].coords)
        start_point = Point(coords[0])
        end_point = Point(coords[-1])
        start_substation = substations_gdf.distance(start_point).idxmin()
        end_substation = substations_gdf.distance(end_point).idxmin()
        if start_substation != end_substation:
            edges.append((start_substation, end_substation, line['VOLTAGE']))

    # Create baseline visualization
    fig_baseline, ax_baseline = plt.subplots(figsize=(12, 8))
    gpd.GeoSeries([georgia_union]).plot(ax=ax_baseline, color='lightgray', edgecolor='black', zorder=0)

    # Process baseline data
    baseline_pop_areas = urban_in_georgia[urban_in_georgia['POP2010'] > BASELINE['MIN_POPULATION']].copy()
    baseline_pop_areas['longitude'] = baseline_pop_areas['geometry'].centroid.x
    baseline_pop_areas['latitude'] = baseline_pop_areas['geometry'].centroid.y
    baseline_pop_areas['diameter'] = baseline_pop_areas['SQMI'].apply(lambda sqmi: sqrt(sqmi * 2.58999 / 3.14159) * 2)

    # Plot baseline population areas
    pop_plot = baseline_pop_areas.plot(
        column='POP2010',
        cmap='inferno',
        legend=False,
        ax=ax_baseline,
        edgecolor='black',
        linewidth=0.5,
        norm=LogNorm(vmin=baseline_pop_areas['POP2010'].min(), vmax=baseline_pop_areas['POP2010'].max())
    )

    # Add colorbar
    sm = plt.cm.ScalarMappable(
        cmap='inferno',
        norm=LogNorm(vmin=baseline_pop_areas['POP2010'].min(), vmax=baseline_pop_areas['POP2010'].max())
    )
    cbar = fig_baseline.colorbar(sm, ax=ax_baseline, orientation='vertical', pad=0.1, location='left')
    cbar.set_label('Population in Urban Areas')

    # Filter and plot baseline infrastructure
    baseline_power_lines = power_lines[power_lines['VOLTAGE'] >= BASELINE['MIN_VOLTAGE']]
    baseline_power_lines.plot(ax=ax_baseline, color='red', linewidth=0.5, label='Power Lines', zorder=2)

    # Calculate baseline substations
    baseline_substations = []
    for _, line in baseline_power_lines.iterrows():
        coords = list(line['geometry'].coords)
        baseline_substations.append(Point(coords[0]))
        baseline_substations.append(Point(coords[-1]))
    baseline_substations_gdf = gpd.GeoDataFrame(geometry=baseline_substations, crs=baseline_power_lines.crs)
    baseline_substations_gdf = baseline_substations_gdf.drop_duplicates(subset=['geometry'])
    baseline_substations_gdf.plot(ax=ax_baseline, color='darkblue', marker='o', markersize=5, label='Substations', zorder=3)

    # Calculate and plot baseline power plants
    baseline_power_needed = (BASELINE['MAX_POWER_PEAK']/MAX_POP) * baseline_pop_areas['POP2010'].sum() * BASELINE['POWER_INSTALLED'] * 1e6
    remaining_power = baseline_power_needed / 1e6
    baseline_power_plants = {}

    for _, row in power_plants.sort_values(by='Install_MW', ascending=False).iterrows():
        if remaining_power > 0:
            source = row['PrimSource'].lower()
            if source not in baseline_power_plants:
                baseline_power_plants[source] = []
            baseline_power_plants[source].append(row)
            remaining_power -= row['Install_MW']

    for source, plants in baseline_power_plants.items():
        color, marker = source_colors.get(source, ('gray', 'o'))
        for plant in plants:
            size = (plant['Install_MW'] / power_plants['Install_MW'].max()) * 300
            ax_baseline.scatter(
                plant['Longitude'],
                plant['Latitude'],
                color=color,
                label=source.capitalize(),
                s=size,
                alpha=0.8,
                edgecolor='black',
                marker=marker,
                zorder=3
            )

    # Add legend
    handles = [
        mlines.Line2D([], [], color='darkblue', marker='o', markersize=5, label='Substations', linestyle='None'),
        mlines.Line2D([], [], color='red', label='Power Lines')
    ] + [
        mlines.Line2D([], [], color=color, marker=marker, markersize=10, label=source.capitalize(), linestyle='None')
        for source, (color, marker) in source_colors.items()
        if source in baseline_power_plants
    ]

    ax_baseline.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.axis('off')
    st.pyplot(fig_baseline)

    # Display baseline metrics
    baseline_total_population = baseline_pop_areas['POP2010'].sum()
    baseline_installed_power = sum(
        sum(plant['Install_MW'] for plant in plants)
        for plants in baseline_power_plants.values()
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Population", f"{baseline_total_population:,.0f}")
    col2.metric("Installed Power", f"{baseline_installed_power:.2f} MW")
    col3.metric("Min Population", f"{BASELINE['MIN_POPULATION']:,}")
    col4.metric("Min Voltage", f"{BASELINE['MIN_VOLTAGE']} kV")

    # Create a Folium map centered on Georgia with no background tiles
    m = folium.Map(location=[32.1656, -82.9001], zoom_start=7, tiles=None)

    # Add the transmission lines with loading percentages to the Folium map
    for _, line in power_lines.iterrows():
        coords = list(line['geometry'].coords)
        loading = line['loading_percent'] if 'loading_percent' in line else 0
        color = cm.Blues(loading / 100)
        folium.PolyLine(
            locations=[(coords[0][1], coords[0][0]), (coords[-1][1], coords[-1][0])],
            color=mcolors.to_hex(color),
            weight=5,
            opacity=0.7,
            tooltip=f"Loading: {loading:.1f}%"
        ).add_to(m)

    return fig_baseline, m

# Create the baseline map and heatmap
fig_baseline, folium_map = create_baseline_map()

#display heatmap
col1 = st.columns(1)
col1 = folium_static(folium_map)

# Button to create and save the Pandapower network from filtered data
if st.button("Create Pandapower Network from Filtered Data"):
    try:
        created_net = create_network_from_filtered_data()  # This function returns the created network
        st.success("Pandapower network created and saved successfully!")
    except Exception as e:
        st.error(f"Error creating network: {e}")

def small_modular_reactors_effect():
    st.write('Effect for Small Modular Reactors')

def cyber_attack_effect():
    power_plants_path = r'data/Power_Plants_georgia.geojson'
    power_plants = gpd.read_file(power_plants_path)
    # Function to set a random top 5 plant's production to 0
    # Sort plants by Total_MW and select the top 5
    top5_plants = power_plants.nlargest(5, 'Total_MW')
    
    # Randomly select one of the top 5 plants
    selected_plant = top5_plants.sample(1).iloc[0]
    
    # Set the selected plant's production to 0
    power_plants.loc[power_plants['Plant_Name'] == selected_plant['Plant_Name'], 'Total_MW'] = 0
    
    st.write(f"{selected_plant['Plant_Name']} taken offline")
    st.write('Effect for Cyber Attack')

def fossil_fuel_outage_effect():
    power_plants_path = r'data/Power_Plants_georgia.geojson'
    power_plants = gpd.read_file(power_plants_path)
    
    # List of fossil fuel sources
    #34% of petroleum is domestic
    #95% of natural gas is domestic
    #90% of coal is domestic
    fossil_fuel_sources = ['coal', 'petroleum', 'natural gas']
    
    # Set the production of all fossil fuel plants to half of their original Total_MW
    for source in fossil_fuel_sources:
        power_plants.loc[power_plants['PrimSource'] == source, 'Total_MW'] /= 2
    
    st.write('Production of all coal, petroleum, and natural gas plants set to half of their original Total_MW')
    st.write(power_plants[['Plant_Name', 'PrimSource', 'Total_MW']])
    st.write('Effect for Fossil Fuel Shortage')

if st.button('Small Modular Reactors'):
    small_modular_reactors_effect()

if st.button('Cyber Attack'):
    cyber_attack_effect()

if st.button('Fossil Fuel Shortage'):
    fossil_fuel_outage_effect()