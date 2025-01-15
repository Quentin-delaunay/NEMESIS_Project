import streamlit as st
import geopandas as gpd
import matplotlib.pyplot as plt
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
import networkx as nx


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

# Add a selection for the view
view = st.sidebar.radio("Select View", ["Filtered Data Visualization", "Mathematical Graph Visualization"])

# Sidebar parameters
st.sidebar.header("Filters")
filter_pop = st.sidebar.slider("Minimum Urban Area Population", 1000, 100000, 50000, key="pop_filter")
st.sidebar.caption("Adjust the minimum population required for an urban area to be displayed on the map.")
max_peak_power = st.sidebar.number_input("Maximum Power Peak (MW)", min_value=5000, max_value=20000, value=11000, step=500, key="peak_power")
st.sidebar.caption("Set the maximum power peak ever recorded for the region.")
Power_installed = st.sidebar.slider(
    "Power Installed Scaling Factor",
    min_value=1.0,  # Minimum value
    max_value=5.0,  # Maximum value
    value=2.5,      # Default value
    step=0.1,       # Step size for higher precision
    key="power_filter"
)
st.sidebar.caption("Set the scaling factor for installed power to visualize power plants proportionately.")
min_voltage = st.sidebar.slider("Minimum Voltage (kV)", 0, 500, 100, key="voltage_filter")
st.sidebar.caption("Define the minimum voltage threshold for displaying power transmission lines.")

# Constants
GA_energy_consumption = {
    'Coal': 180.9,
    'NaturalGas': 812.4,
    'Electricity_in': 355.4 + 10.8 + 250.3 + 25.7,
    'Electricity_out': 262.2
}  # in BTU
total_consumption = sum(GA_energy_consumption.values())
Elec_per_CP_2024 = Power_installed * max_peak_power * 1e6 / 11e6  # MW

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
high_pop_areas = urban_in_georgia[urban_in_georgia['POP2010'] > filter_pop].copy()
high_pop_areas['longitude'] = high_pop_areas['geometry'].centroid.x
high_pop_areas['latitude'] = high_pop_areas['geometry'].centroid.y
high_pop_areas['diameter'] = high_pop_areas['SQMI'].apply(lambda sqmi: sqrt(sqmi * 2.58999 / 3.14159) * 2)

# Total filtered population
total_population = high_pop_areas['POP2010'].sum()

# Calculate maximum estimated power consumption based on filtered population
max_estimated_consumption = (max_peak_power/MAX_POP  ) * ( total_population)  # in MW

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

# Metrics
installed_power = sum(
    sum(plant['Install_MW'] for plant in plants)
    for plants in filtered_power_plants.values()
)



# Display updated metrics
st.sidebar.subheader("Metrics")
st.sidebar.metric(label="Installed Power (MW)", value=f"{installed_power:.2f}")
st.sidebar.metric(label="Total Population", value=f"{total_population:,}")
st.sidebar.metric(label="Max Population", value=f"{MAX_POP:,}")
st.sidebar.metric(label="Max Estimated Power Consumption (MW)", value=f"{max_estimated_consumption:.2f}")


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

# Visualizations
if view == "Filtered Data Visualization":
    st.title("Filtered Data Visualization")
    fig, ax = plt.subplots(figsize=(12, 8))
    gpd.GeoSeries([georgia_union]).plot(ax=ax, color='lightgray', edgecolor='black', zorder=0)
    # Plot high population areas
    pop_plot = high_pop_areas.plot(column='POP2010', cmap='inferno', legend=False, ax=ax, edgecolor='black', linewidth=0.5)

    # Add colorbar for heatmap
    sm = plt.cm.ScalarMappable(cmap='inferno', norm=plt.Normalize(vmin=high_pop_areas['POP2010'].min(), vmax=high_pop_areas['POP2010'].max()))
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.1, location='left')

    cbar.set_label('Population in Urban Areas')
    power_lines.plot(ax=ax, color='red', linewidth=0.5, label='Power Lines', zorder=2)
    substations_gdf.plot(ax=ax, color='blue', markersize=10, label='Substations', zorder=3)

    # Plot power plants
    source_colors = {
        'nuclear': ('blue', 'o'), 'natural gas': ('green', 's'), 'coal': ('black', 'D'),
        'hydroelectric': ('cyan', '^'), 'petroleum': ('magenta', 'v'), 'pumped storage': ('orange', '^'),
        'biomass': ('brown', 'P'), 'other': ('gray', '*'), 'solar': ('yellow', 'h'), 'batteries': ('purple', 'o')
    }
    for source, plants in filtered_power_plants.items():
        color, marker = source_colors.get(source, ('gray', 'o'))
        for plant in plants:
            size = (plant['Install_MW'] / power_plants['Install_MW'].max()) * 300
            ax.scatter(plant['Longitude'], plant['Latitude'], color=color, label=source.capitalize(),
                       s=size, alpha=0.8, edgecolor='black', marker=marker, zorder=3)

    handles = [
        mlines.Line2D([], [], color='blue', label='Substations'),
        mlines.Line2D([], [], color='red', label='Power Lines')
    ] + [
        mlines.Line2D([], [], color=color, marker=marker, markersize=10, label=source.capitalize())
        for source, (color, marker) in source_colors.items()
    ]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.05, 1), title='Legend')
    plt.title("Georgia Energy Infrastructure")
    plt.axis('off')
    st.pyplot(fig)

    if st.button("Save Data for Pandapower Network with Urban Areas"):
        output_folder = "output_pandapower/"
        os.makedirs(output_folder, exist_ok=True)

        # Structures to store data
        buses = []  # List of buses
        lines = []  # List of lines
        transformers = []  # List of transformers
        existing_buses = {}  # Track buses by location

        def create_or_find_bus(location, voltage, line_id):
            """
            Finds an existing bus or creates a new one at the given location with the specified voltage.
            Associates the bus with the given line ID. If a bus exists at the location but has a different
            voltage, creates a new bus slightly offset for visualization clarity.
            """
            # Exact location match
            location_key = (location[0], location[1])
            bus_id_key = f"{location_key[0]}_{location_key[1]}_{voltage}"

            # Check if the bus already exists
            if bus_id_key in existing_buses:
                # Append the line ID to the associated lines for the bus
                existing_buses[bus_id_key]['lines'].append(line_id)
                return existing_buses[bus_id_key]['bus_id']

            # Create a new bus if it doesn't exist
            new_bus_id = f"bus_{bus_id_key}"  # Unique ID with voltage included
            buses.append({
                "bus_id": new_bus_id,
                "x": location_key[0],
                "y": location_key[1],
                "type": "b",  # Regular bus
                "vn_kv": voltage,  # Voltage
                "lines": [line_id]  # Initialize with the current line ID
            })
            existing_buses[bus_id_key] = {
                "bus_id": new_bus_id,
                "voltage": voltage,
                "x": location_key[0],
                "y": location_key[1],
                "lines": [line_id]  # Track associated lines
            }
            return new_bus_id

        # Process power lines to create buses and record line IDs
        for idx, line in power_lines.iterrows():
            try:
                coords = list(line.geometry.coords)
            except Exception as e:
                st.error(f"Error reading line geometry: {e}")
                continue

            start_location = (coords[0][0], coords[0][1])
            end_location = (coords[-1][0], coords[-1][1])
            line_id = line['ID']  # Extract the line ID
            line_voltage = line['VOLTAGE']

            # Create or find buses at the start and end of the line
            start_bus = create_or_find_bus(start_location, line_voltage, line_id)
            end_bus = create_or_find_bus(end_location, line_voltage, line_id)
            
                    # Ensure the bus entry in 'buses' reflects all associated line IDs
            for bus in buses:
                if bus['bus_id'] == start_bus:
                    bus['lines'].append(line_id)
                if bus['bus_id'] == end_bus:
                    bus['lines'].append(line_id)

        # Save data to CSV files
        pd.DataFrame(buses).to_csv(f"{output_folder}buses.csv", index=False)

        # File path to the saved buses CSV
        buses_file_path = f"{output_folder}buses.csv"

        # Load the buses CSV
        buses_df = pd.read_csv(buses_file_path)

        # Group by location (x, y) and voltage (vn_kv), and merge line IDs
        merged_buses = (
            buses_df.groupby(['x', 'y', 'vn_kv'], as_index=False)
            .agg({
                'bus_id': 'first',  # Keep the first bus_id for each group
                'lines': lambda line_lists: list(set(sum([eval(lines) if isinstance(lines, str) else lines for lines in line_lists], [])))
            })
        )

        # Save the consolidated buses back to the CSV
        merged_buses['lines'] = merged_buses['lines'].apply(lambda lines: str(lines))  # Convert lines list to string
        merged_buses.to_csv(buses_file_path, index=False)

        print(f"Consolidated buses saved to {buses_file_path}")
        
        # Create transformers by identifying buses with the same location but different voltages
        for idx, bus in merged_buses.iterrows():
            for idx2, bus2 in merged_buses.iterrows():
                if idx < idx2:  # Avoid redundant comparisons
                    if bus['x'] == bus2['x'] and bus['y'] == bus2['y'] and bus['vn_kv'] != bus2['vn_kv']:
                        # Create a transformer entry
                        
                        transformers.append({
                            "from_bus": bus['bus_id'],
                            "to_bus": bus2['bus_id'],
                            "hv_voltage_kv": max(bus['vn_kv'], bus2['vn_kv']),  # Higher voltage
                            "lv_voltage_kv": min(bus['vn_kv'], bus2['vn_kv']),  # Lower voltage
                            "std_type": "Standard Transformer"  # Example transformer type
                        })
        # Define the output file path for transformers
        transformers_file_path = f"{output_folder}transformers.csv"

        # Convert the transformers list to a DataFrame
        transformers_df = pd.DataFrame(transformers)

        # Save the DataFrame to a CSV file
        transformers_df.to_csv(transformers_file_path, index=False)
    
        pd.DataFrame(lines).to_csv(f"{output_folder}lines.csv", index=False)
        pd.DataFrame(transformers).to_csv(f"{output_folder}transformers.csv", index=False)

        # Save Urban Areas as Pandapower Loads
        loads = []
        for _, area in high_pop_areas.iterrows():
            area_geometry = area['geometry'].buffer(0.001)  # Add buffer for precision if needed

            # Find buses in merged_buses that intersect the urban area
            merged_buses['intersects'] = merged_buses.apply(
                lambda row: Point(row['x'], row['y']).intersects(area_geometry), axis=1
            )
            intersecting_buses = merged_buses[merged_buses['intersects']]

            if not intersecting_buses.empty:
                # Filter for buses with the lowest voltage
                min_voltage = intersecting_buses['vn_kv'].min()
                low_voltage_buses = intersecting_buses[intersecting_buses['vn_kv'] == min_voltage]

                # Calculate the load per bus
                estimated_max_load = (area['POP2010'] / total_population) * max_estimated_consumption  # MW
                load_per_bus = (estimated_max_load * 1000) / len(low_voltage_buses)  # kW per bus

                # Distribute the load across the low voltage buses
                for _, bus in low_voltage_buses.iterrows():
                    loads.append({
                        "bus": bus['bus_id'],
                        "p_kw": load_per_bus,
                        "type": "urban",
                        "name": f"{area['NAME']} (Bus {bus['bus_id']})"
                    })
            else:
                # Fallback to nearest bus if no intersections found
                merged_buses['distance'] = merged_buses.apply(
                    lambda row: area_geometry.distance(Point(row['x'], row['y'])), axis=1
                )
                nearest_bus_idx = merged_buses['distance'].idxmin()
                nearest_bus = merged_buses.iloc[nearest_bus_idx]

                # Estimate load for this urban area
                estimated_max_load = (area['POP2010'] / total_population) * max_estimated_consumption  # MW
                loads.append({
                    "bus": nearest_bus['bus_id'],
                    "p_kw": estimated_max_load * 1000,  # Convert MW to kW
                    "type": "urban",
                    "name": f"{area['NAME']} (Nearest Bus {nearest_bus['bus_id']})"
                })

        # Save loads to CSV
        loads_df = pd.DataFrame(loads)
        loads_df.to_csv(f"{output_folder}loads.csv", index=False)
        # Add generators
        generators = []

        for source, plants in filtered_power_plants.items():
            for generator in plants:
                # Record the generator's position
                generator_point = Point(generator['Longitude'], generator['Latitude'])

                # Find the nearest bus in merged_buses
                merged_buses['distance'] = merged_buses.apply(
                    lambda row: generator_point.distance(Point(row['x'], row['y'])), axis=1
                )
                nearest_bus_idx = merged_buses['distance'].idxmin()
                nearest_bus = merged_buses.iloc[nearest_bus_idx]

                # Add the generator to the output with its associated bus
                generators.append({
                    "generator_name": generator['Plant_Name'],  # Name of the generator
                    "longitude": generator['Longitude'],  # Longitude of the generator
                    "latitude": generator['Latitude'],  # Latitude of the generator
                    "type": source,  # Primary energy source
                    "p_mw": generator['Install_MW'],  # Power output in MW
                    "vm_pu": 1.0,  # Voltage magnitude in per unit
                    "sn_mva": generator['Install_MW'],  # Apparent power in MVA
                    "bus_id": nearest_bus['bus_id'],  # ID of the nearest bus
                    "bus_x": nearest_bus['x'],  # X coordinate of the bus
                    "bus_y": nearest_bus['y']  # Y coordinate of the bus
                })

        # Save all information into a single CSV file
        generators_df = pd.DataFrame(generators)
        generators_df.to_csv(f"{output_folder}generators.csv", index=False)

        st.success("Generator data with associated bus information saved successfully!")



        st.success("Data saved successfully for Pandapower network creation, prioritizing lower voltage substations!")





elif view == "Mathematical Graph Visualization":
    st.title("Mathematical Graph Visualization")

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
            color='blue',
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


