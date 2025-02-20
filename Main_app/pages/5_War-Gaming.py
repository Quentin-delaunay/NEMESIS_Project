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
from folium.plugins import MarkerCluster
from shapely.geometry import Point, LineString
import math
from streamlit_folium import folium_static, st_folium
from utils import create_network_from_filtered_data
import networkx as nx
import pandapower as pp

st.set_page_config(page_title="War-Gaming", layout="wide")
st.title("War-Gaming")

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
    dlat = math.radians(lat2 - lat1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# Georgia filter parameters

filter_pop = 100000 # only looking at most populated region
max_peak_power = 12431  # Maximum power peak ever recorded in Georgia
Power_installed = 3.242

min_voltage = 500 #only looking at major tranmission lines


# Load population data
population_path = r'data/USA_Urban_Areas_(1%3A500k-1.5M).geojson'
population_gdf = gpd.read_file(population_path)
population_gdf['geometry'] = population_gdf['geometry'].buffer(0)

# Load Georgia boundary data
georgia_boundary_path = 'data/georgia-counties.json'
georgia = gpd.read_file(georgia_boundary_path)
MAX_POP = 11269572 #TODO: use a list of historical data or prediction
georgia['geometry'] = georgia['geometry'].buffer(0)
georgia_union = georgia.geometry.unary_union

# Filter urban areas
urban_in_georgia = population_gdf[population_gdf.intersects(georgia_union)].copy()
high_pop_areas = urban_in_georgia[urban_in_georgia['POP2010'] > 100000].copy()
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
power_lines = power_lines[power_lines['VOLTAGE'] >= min_voltage]

power_plants_path = r'data/Power_Plants_georgia.geojson'
power_plants = gpd.read_file(power_plants_path)

# Calculate required power
Power_needed = max_estimated_consumption * Power_installed *1e6
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


# Display updated metrics


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
    # Extend the graph with generators and urban areas
    extended_edges = edges.copy()

    # Connect filtered generators to substations
    for source, plants in filtered_power_plants.items():
        for generator in plants:
            generator_point = Point(generator['Longitude'], generator['Latitude'])
            generator_id = f"gen_{generator['Plant_Name']}"
            connected_substations = []

            # Check for substations within a defined radius (e.g., 0.5 km)
            for idx, substation in substations_gdf.iterrows():
                if haversine(generator['Longitude'], generator['Latitude'], substation.geometry.x, substation.geometry.y) < 0.5:
                    connected_substations.append(idx)
                    extended_edges.append((generator_id, idx, "generator"))

            # If no substation is within the radius, connect to the nearest substation
            if not connected_substations:
                nearest_substation = substations_gdf.distance(generator_point).idxmin()
                extended_edges.append((generator_id, nearest_substation, "generator"))

    # Connect urban areas to substations
    for _, area in high_pop_areas.iterrows():
        area_point = Point(area['longitude'], area['latitude'])
        area_radius = area['diameter'] / 2  # Diameter to radius
        area_id = f"urban_{area['POP2010']}"
        connected_substations = []

        # Check for substations within the urban area's radius
        for idx, substation in substations_gdf.iterrows():
            if haversine(area['longitude'], area['latitude'], substation.geometry.x, substation.geometry.y) < area_radius:
                connected_substations.append(idx)
                extended_edges.append((area_id, idx, "urban"))

        # If no substation is within the radius, connect the nearest substation
        if not connected_substations:
            nearest_substation = substations_gdf.distance(area_point).idxmin()
            extended_edges.append((area_id, nearest_substation, "urban"))

    # Create Folium map with larger dimensions
    m = folium.Map(location=[32.8407, -83.6324], zoom_start=7, width='100%', height='750px')



    # Add edges (connections between substations, generators, and urban areas)
    for edge in extended_edges:
        if edge[2] == "generator":
            color = "green"  # Same color for generator connections
        elif edge[2] == "urban":
            color = "orange"  # Urban area connections
        else:
            color = "red"  # Default for power lines

        # Determine start and end coordinates for edges
        if isinstance(edge[0], str) and edge[0].startswith("gen"):
            start_point = power_plants.loc[power_plants['Plant_Name'] == edge[0][4:], ['Longitude', 'Latitude']].iloc[0]
            start_coords = (start_point['Latitude'], start_point['Longitude'])
        elif isinstance(edge[0], str) and edge[0].startswith("urban"):
            start_point = high_pop_areas.loc[high_pop_areas['POP2010'] == int(edge[0][6:]), ['longitude', 'latitude']].iloc[0]
            start_coords = (start_point['latitude'], start_point['longitude'])
        else:
            start_coords = (substations_gdf.loc[edge[0]].geometry.y, substations_gdf.loc[edge[0]].geometry.x)

        end_coords = (substations_gdf.loc[edge[1]].geometry.y, substations_gdf.loc[edge[1]].geometry.x)
        folium.PolyLine([start_coords, end_coords], color=color, weight=2).add_to(m)
    
    
    # Add substations to the map
    for idx, substation in substations_gdf.iterrows():
        folium.CircleMarker(
            location=(substation.geometry.y, substation.geometry.x),
            radius=1,
            color='darkblue',
            fill=True,
            fill_color='blue',
            tooltip=f"Substation {idx}"
        ).add_to(m)



    # Add urban areas to the map
    for _, area in high_pop_areas.iterrows():
        folium.Circle(
            location=(area['latitude'], area['longitude']),
            radius=area['diameter'] * 1000 / 2,  # Convert km to meters
            color='orange',
            fill=True,
            fill_opacity=0.1,
            tooltip=f"Urban Area: {area['POP2010']} people"
        ).add_to(m)
        
        
    source_icons = {
        'nuclear': ('blue', 'radiation'),  # Example icon for nuclear
        'natural gas': ('green', 'fire'),
        'coal': ('black', 'cloud'),
        'hydroelectric': ('cyan', 'water'),
        'petroleum': ('magenta', 'oil'),
        'pumped storage': ('orange', 'plug'),
        'biomass': ('brown', 'leaf'),
        'other': ('gray', 'question'),
        'solar': ('yellow', 'sun'),
        'batteries': ('purple', 'battery-full')
    }

    # Add generators to the map     
    for source, plants in filtered_power_plants.items():
        color, icon_name = source_icons.get(source, ('gray', 'question'))  # Same color and icon as before
        for generator in plants:
            # Calculate scaling for size
            size_scale = generator['Install_MW'] / power_plants['Install_MW'].max() * 20  # Adjust max size as needed

            # Add a circle underneath to represent size
            folium.Circle(
                location=(generator['Latitude'], generator['Longitude']),
                radius=size_scale * 500,  # Adjust radius scaling for visualization
                color=color,
                fill=True,
                fill_opacity=0.8,  # Transparent circle
            ).add_to(m)

            # Add the regular marker with Icon
            folium.Marker(
                location=(generator['Latitude'], generator['Longitude']),
                icon=folium.Icon(color=color, icon=icon_name, prefix='fa'),
                tooltip=f"Generator: {generator['Plant_Name']} ({source.capitalize()}, {generator['Install_MW']} MW)"
            ).add_to(m)

        



    # Add legend for generator types
    legend_html = """
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 250px; height: auto; 
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; border-radius:10px; padding: 10px;">
        <strong>Generator Types</strong><br>
    """
    for source, (color, _) in source_colors.items():
        legend_html += f"<i style='background:{color}; width:10px; height:10px; float:left; margin-right:5px;'></i>{source.capitalize()}<br>"
    legend_html += "</div>"
    from folium import MacroElement
    from jinja2 import Template
    legend = MacroElement()
    legend._template = Template(legend_html)
    m.get_root().add_child(legend)

    # Display Folium map
    folium_static(m, width=1000, height= 1000)

    # Create a NetworkX graph and add nodes with positions
    G = nx.Graph()

    # Add substations with positions
    for idx, substation in substations_gdf.iterrows():
        G.add_node(idx, pos=(substation.geometry.x, substation.geometry.y))

    # Add generator nodes with positions
    for source, plants in filtered_power_plants.items():
        for generator in plants:
            generator_id = f"gen_{generator['Plant_Name']}"
            # Add generator node with position
            G.add_node(generator_id, pos=(generator['Longitude'], generator['Latitude']))

    # Add urban areas with positions
    for _, area in high_pop_areas.iterrows():
        area_id = f"urban_{area['POP2010']}"
        # Add urban area node with position
        G.add_node(area_id, pos=(area['longitude'], area['latitude']))

    # Add edges (substations, generators, and urban areas connections)
    for edge in extended_edges:
        G.add_edge(edge[0], edge[1], type=edge[2])

    if st.button("Save Extended Graph"):
        output_folder = "output/"
        os.makedirs(output_folder, exist_ok=True)

        # Convert 'pos' attribute to a string "x,y" for compatibility with GEXF
        for node, data in G.nodes(data=True):
            if "pos" in data and isinstance(data["pos"], tuple):
                data["pos"] = f"{data['pos'][0]},{data['pos'][1]}"

        # Save graph in GEXF format
        nx.write_gexf(G, f"{output_folder}graph.gexf", encoding="utf-8", prettyprint=True)
        st.success("Graph saved successfully!")

if "net" in st.session_state:
    net = st.session_state["net"]
    

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
    st.write('Effect for Fossil Fuel Outage')

if st.button('Small Modular Reactors'):
    small_modular_reactors_effect()

if st.button('Cyber Attack'):
    cyber_attack_effect()

if st.button('Fossil Fuel Outage'):
    fossil_fuel_outage_effect()