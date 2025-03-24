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
import math
from utils import create_network_from_filtered_data, plot_network_heatmaps, load_network
import pandapower as pp
import plotly.graph_objects as go
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pandapower.topology import create_nxgraph

st.set_page_config(page_title="War-Gaming", layout="wide")
st.title("War-Gaming")

military_bases = [
    {"name": "Moody Air Force Base", "latitude": 30.968611, "longitude": -83.193056, 'type': 'Air Force Base'},
    {"name": "Robins Air Force Base", "latitude": 32.64, "longitude": -83.591667, 'type': 'Air Force Base'},
    {"name": "Dobbins Air Reserve Base", "latitude": 33.915278, "longitude": -84.516389, 'type': 'Air Force Base'},
    {"name": "Fort Benning Army Base", "latitude": 32.366111, "longitude": -84.969167, 'type': 'Army Base'},
    {"name": "Fort Gillem", "latitude": 33.6202, "longitude": -84.3289, 'type': 'Army Base'},
    {"name": "Fort Gordon", "latitude": 33.413333, "longitude": -82.135278, 'type': 'Army Base'},
    {"name": "Fort McPherson", "latitude": 33.706206, "longitude": -84.433279, 'type': 'Army Base'},
    {"name": "Fort Stewart", "latitude": 31.88, "longitude": -81.6075, 'type': 'Army Base'},
    {"name": "Hunter Army Airfield", "latitude": 32.01, "longitude": -81.145556, 'type': 'Army Base'},
    {"name": "Marine Corps Logistics Base Albany", "latitude": 31.55, "longitude": -84.054167, 'type': 'Marine Base'},
    {"name": "Naval Submarine Base Kings Bay", "latitude": 30.781667, "longitude": -81.535, 'type': 'Navy Base'},
    {"name": "Fort Eisenhower", "latitude": 33.413333, "longitude": -82.135278, 'type': 'Army Base'},
    {"name": "Fort Stewart", "latitude": 31.88, "longitude": -81.6075, 'type': 'Army Base'},
    {"name": "Camp Frank D Merrill", "latitude": 34.628293, "longitude": -84.103033, 'type': 'Army Base'},
    {"name": "General Lucius D. Clay National Guard Center", "latitude": 33.915278, "longitude": -84.516389, 'type': 'National Guard Base'},
]

BASELINE = {
    'MIN_POPULATION': 50000,  # This makes the filter take look at all the urban areas
    'MAX_POWER_PEAK': 15636,  # MW Georgia Power IRP prediction for 2024
    'POWER_INSTALLED': 37786,  # MW Generated in Georgia in total
    'MIN_VOLTAGE': 300  # looks at only major transmission lines
}
years_pred = [2024,2025,2026,2027,2028,2029,2030,2031,2032,2033,2034,2035,2036,2037,2038,2039,2040,2041,2042,2043,2044]
Max_power_peak_pred = [15636,16300,17300,18300,20250,22200,23350,24500,24850,25200,25450,25700,25850,26000,26300,26600,26950,27300,27700,28100,28500]
electricity_imports = 2806.17899  # MW/hr or 24,582,128 MW a year
electricity_exports = 0  # MW/hr or 0MW a year
selected_year = st.slider('Select Year', min_value=2024, max_value=2044, value=2024, step=1)


# Update BASELINE MAX_POWER_PEAK based on selected year
BASELINE['MAX_POWER_PEAK'] = Max_power_peak_pred[years_pred.index(selected_year)]

# Calculate the power installed with an annual growth rate of 2.4%
initial_power_installed = 37786  # MW
growth_rate = 0.024
years_since_2024 = selected_year - 2024
BASELINE['POWER_INSTALLED'] = initial_power_installed * ((1 + growth_rate) ** years_since_2024)

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
}

# Function to calculate distance between two points (Haversine formula)
def haversine(lat1, lon1, lat2, lon2):
    R = 3959.87433  # Earth radius in miles
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    a = math.sin(dLat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dLon / 2) ** 2
    c = 2 * math.asin(sqrt(a))
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
    max_estimated_consumption = (BASELINE['MAX_POWER_PEAK'] / MAX_POP) * (total_population)  # in MW

    # Load pipelines, power lines, and plants
    power_lines_path = r'data/Transmission_Lines_all.geojson'
    power_lines = gpd.read_file(power_lines_path)
    power_lines = power_lines[power_lines['VOLTAGE'] >= BASELINE['MIN_VOLTAGE']]

    power_plants_path = r'data/Power_Plants_georgia.geojson'
    power_plants = gpd.read_file(power_plants_path)

    # Calculate required power
    Power_needed = max_estimated_consumption * BASELINE['POWER_INSTALLED'] * 1e6
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

    # Create baseline visualization using same method as filtering file
    fig_baseline, ax_baseline = plt.subplots(figsize=(12, 8))
    gpd.GeoSeries([georgia_union]).plot(ax=ax_baseline, color='lightgray', edgecolor='black', zorder=0)

    # Plot population areas - using same styling as filtering file
    high_pop_areas.plot(
        column='POP2010',
        cmap='inferno',
        legend=False,
        ax=ax_baseline,
        edgecolor='black',
        linewidth=0.5,
        norm=LogNorm(vmin=high_pop_areas['POP2010'].min(), vmax=high_pop_areas['POP2010'].max()),
        zorder=1
    )

    # Add colorbar
    sm = plt.cm.ScalarMappable(
        cmap='inferno',
        norm=LogNorm(vmin=high_pop_areas['POP2010'].min(), vmax=high_pop_areas['POP2010'].max())
    )
    cbar = fig_baseline.colorbar(sm, ax=ax_baseline, orientation='vertical', pad=0.1, location='left')
    cbar.set_label('Population in Urban Areas')

    # Plot power lines
    power_lines.plot(ax=ax_baseline, color='red', linewidth=0.5, label='Power Lines', zorder=2)

    # Plot substations
    substations_gdf.plot(ax=ax_baseline, color='darkblue', marker='o', markersize=5, zorder=3)

    # Get baseline power plants
    baseline_power_needed = (BASELINE['MAX_POWER_PEAK'] / MAX_POP) * high_pop_areas['POP2010'].sum() * BASELINE['POWER_INSTALLED'] * 1e6
    remaining_power = baseline_power_needed / 1e6
    baseline_power_plants = {}

    for _, row in power_plants.sort_values(by='Install_MW', ascending=False).iterrows():
        if remaining_power > 0:
            source = row['PrimSource'].lower()
            if source not in baseline_power_plants:
                baseline_power_plants[source] = []
            baseline_power_plants[source].append(row)
            remaining_power -= row['Install_MW']

    # Plot power plants using the same style as filtering file
    for source, plants in baseline_power_plants.items():
        color, marker = source_colors.get(source, ('gray', 'o'))
        for plant in plants:
            size = (plant['Install_MW'] / power_plants['Install_MW'].max()) * 300
            ax_baseline.scatter(
                plant['Longitude'],
                plant['Latitude'],
                color=color,
                s=size,
                alpha=0.8,
                edgecolor='black',
                marker=marker,
                zorder=3
            )

    # Remove duplicate military bases
    unique_military_bases = []
    seen_bases = set()
    for base in military_bases:
        if (base["name"], base["latitude"], base["longitude"]) not in seen_bases:
            unique_military_bases.append(base)
            seen_bases.add((base["name"], base["latitude"], base["longitude"]))

    # Define base colors and markers as in filtering file
    base_colors = {
        'Air Force Base': 'red',
        'Army Base': 'green',
        'Marine Base': 'blue',
        'Navy Base': 'purple',
        'National Guard Base': 'orange'
    }
    base_markers = {
        'Air Force Base': 'v',  # tri-down
        'Army Base': 'h',       # hexagon
        'Marine Base': 'X',     # filled X
        'Navy Base': 's',       # square
        'National Guard Base': 'd'  # diamond
    }

    # Plot military bases without name annotations
    for base in unique_military_bases:
        ax_baseline.scatter(
            base["longitude"],
            base["latitude"],
            color=base_colors[base['type']],
            marker=base_markers[base['type']],
            s=100,
            edgecolor='black',
            zorder=4
        )

    # Add legend with all elements
    handles = [
        mlines.Line2D([], [], color='darkblue', marker='o', markersize=5, label='Substations', linestyle='None'),
        mlines.Line2D([], [], color='red', label='Power Lines')
    ] + [
        mlines.Line2D([], [], color=color, marker=marker, markersize=10, label=source.capitalize(), linestyle='None')
        for source, (color, marker) in source_colors.items()
        if source in baseline_power_plants
    ] + [
        mlines.Line2D([], [], color=color, marker=base_markers[base_type], markersize=10, label=base_type, linestyle='None')
        for base_type, color in base_colors.items()
    ]

    ax_baseline.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.title("Georgia Energy Infrastructure")
    plt.axis('off')
    st.pyplot(fig_baseline)

    return substations_gdf, high_pop_areas, edges, fig_baseline, baseline_power_plants

# Create the baseline map and get the data
with st.spinner("Creating baseline map..."):
    substations_gdf, high_pop_areas, edges, fig_baseline, baseline_power_plants = create_baseline_map()

def create_and_save_baseline_network(substations_gdf, high_pop_areas, edges, baseline_power_plants):
    """
    Creates a pandapower network from the baseline map data and saves it.
    """
    # Create an empty pandapower network
    net = pp.create_empty_network()
    
    # Create buses from substations
    bus_index_map = {}
    for idx, substation in substations_gdf.iterrows():
        x, y = substation.geometry.x, substation.geometry.y
        bus_idx = pp.create_bus(net, vn_kv=500, geodata=(x, y), name=f"Bus_{idx}")
        bus_index_map[idx] = bus_idx
    
    # Create lines between substations
    for start, end, voltage in edges:
        if start in bus_index_map and end in bus_index_map:
            from_bus = bus_index_map[start]
            to_bus = bus_index_map[end]
            
            # Set parameters based on voltage level
            if voltage >= 500:
                pp.create_line_from_parameters(
                    net,
                    from_bus=from_bus,
                    to_bus=to_bus,
                    length_km=haversine(
                        substations_gdf.iloc[start].geometry.y,
                        substations_gdf.iloc[start].geometry.x,
                        substations_gdf.iloc[end].geometry.y,
                        substations_gdf.iloc[end].geometry.x
                    ),
                    r_ohm_per_km=0.01,
                    x_ohm_per_km=0.25,
                    c_nf_per_km=12,
                    max_i_ka=9,  # guessing values here to approximate real-world values
                    parallel=2      # Added parallel lines for high voltage
                )
            else:
                pp.create_line_from_parameters(
                    net,
                    from_bus=from_bus,
                    to_bus=to_bus,
                    length_km=haversine(
                        substations_gdf.iloc[start].geometry.y,
                        substations_gdf.iloc[start].geometry.x,
                        substations_gdf.iloc[end].geometry.y,
                        substations_gdf.iloc[end].geometry.x
                    ),
                    r_ohm_per_km=0.1,
                    x_ohm_per_km=0.4,
                    c_nf_per_km=9,
                    max_i_ka=9,  # guess value to achieve overload in 2030
                    parallel=1      # Single line for lower voltage
                )
    
    # Add generators based on power plants
    slack_assigned = False
    for source, plants in baseline_power_plants.items():
        for plant in plants:
            # Find nearest bus to plant
            plant_point = Point(plant['Longitude'], plant['Latitude'])
            nearest_bus = substations_gdf.distance(plant_point).idxmin()
            bus_idx = bus_index_map[nearest_bus]
            
            # Create generator
            is_slack = not slack_assigned and source == 'natural gas'
            pp.create_gen(
                net,
                bus=bus_idx,
                p_mw=plant['Total_MW'],
                vm_pu=1.0,
                min_p_mw=0,
                max_p_mw=plant['Install_MW'],
                min_q_mvar=-plant['Install_MW'] / 0.85,
                max_q_mvar=plant['Install_MW'] / 0.85,
                name=f"{plant['Plant_Name']} ({source})",
                slack=is_slack,
                controllable=True
            )
            if is_slack:
                slack_assigned = True
    
    # If no slack generator was assigned, assign the first bus as slack
    if not slack_assigned and len(net.bus) > 0:
        pp.create_ext_grid(net, bus=0, vm_pu=1.0, va_degree=0.0)
    
    # Add loads to high population areas
    total_population = high_pop_areas['POP2010'].sum()
    for _, area in high_pop_areas.iterrows():
        area_point = Point(area['longitude'], area['latitude'])
        nearest_bus = substations_gdf.distance(area_point).idxmin()
        bus_idx = bus_index_map[nearest_bus]
        
        # Calculate load based on population proportion
        load_mw = (area['POP2010'] / total_population) * BASELINE['MAX_POWER_PEAK']
        
        # Use NAME key for the area name
        area_name = area['NAME'] if 'NAME' in area else f"Area_{area['POP2010']}"
        
        pp.create_load(
            net,
            bus=bus_idx,
            p_mw=load_mw,
            q_mvar=load_mw * 0.3,  # Assuming power factor of about 0.96
            name=f"Load_{area_name}"
        )
    
    # Create folder if it doesn't exist
    os.makedirs("output_pandapower", exist_ok=True)
    
    # Save the network
    pp.to_pickle(net, "output_pandapower/baseline_network.p")
    st.success("Baseline network saved to output_pandapower/baseline_network.p")
    
    return net

# Automatically create and save the baseline network
with st.spinner("Creating and saving network..."):
    net = create_and_save_baseline_network(substations_gdf, high_pop_areas, edges, baseline_power_plants)

def calculate_line_loading(net):
    pp.runpp(net)
    line_loading = net.res_line[['loading_percent']]
    return line_loading

# Calculate line loading
with st.spinner("Calculating line loading..."):
    line_loading = calculate_line_loading(net)

# Create the network graph using NetworkX
G = create_nxgraph(net, respect_switches=True)

# Use geographical coordinates if available, otherwise use spring layout
if hasattr(net, "bus_geodata") and not net.bus_geodata.empty:
    pos = {bus: (net.bus_geodata.at[bus, "x"], net.bus_geodata.at[bus, "y"]) for bus in net.bus_geodata.index}
else:
    pos = nx.spring_layout(G, seed=42)

# Prepare node trace for bus overloads
node_x, node_y, node_color, node_text = [], [], [], []
for bus in G.nodes():
    x, y = pos[bus]
    node_x.append(x)
    node_y.append(y)
    overload = G.nodes[bus].get("overload", 0)
    node_color.append(overload)
    label = f"Bus {bus}<br>Overload: {overload:.1f}%" if overload > 0 else ""
    node_text.append(label)

node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode="markers+text",
    text=node_text,
    textposition="top center",
    hoverinfo="text",
    marker=dict(
        size=15,
        color=node_color,
        colorscale="Reds",
        colorbar=dict(title="Bus Overload (%)", x=0.0),
        cmin=0,
        cmax=max(node_color) if node_color else 1,
        line=dict(width=2)
    )
)

# Prepare edge traces for line loadings
edge_traces = []
edge_annotations = []
for idx, line in net.line.iterrows():
    u = line["from_bus"]
    v = line["to_bus"]
    if u not in pos or v not in pos:
        continue
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    loading = net.res_line.at[idx, "loading_percent"] if idx in net.res_line.index else 0
    cmap = cm.get_cmap("Blues")
    norm = mcolors.Normalize(vmin=0, vmax=100)
    rgba = cmap(norm(loading))
    hex_color = mcolors.to_hex(rgba)
    edge_trace = go.Scatter(
        x=[x0, x1],
        y=[y0, y1],
        mode="lines",
        line=dict(color=hex_color, width=3),
        hoverinfo="text",
        text=f"Line {idx}<br>Loading: {loading:.1f}%"
    )
    edge_traces.append(edge_trace)
    if loading > 100:
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        edge_annotations.append(dict(
            x=mid_x,
            y=mid_y,
            text=f"{loading:.1f}%",
            showarrow=False,
            font=dict(color="red", size=15)
        ))

fig_network = go.Figure(
    data=edge_traces + [node_trace],
    layout=go.Layout(
        title="Network Graph: Bus Overloads & Line Loadings",
        showlegend=False,
        hovermode="closest",
        annotations=edge_annotations,
        xaxis=dict(
            scaleanchor="y", scaleratio=1,
            showgrid=False, zeroline=False, showticklabels=False
        ),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=750,
        height=750,
        margin=dict(l=20, r=20, t=40, b=20)
    )
)

# Add a dummy trace for the line loading colorbar
dummy_trace = go.Scatter(
    x=[None],
    y=[None],
    mode="markers",
    marker=dict(
        colorscale="Blues",
        showscale=True,
        cmin=0,
        cmax=100,
        colorbar=dict(title="Line Loading (%)", x=1.0)
    ),
    hoverinfo="none"
)
fig_network.add_trace(dummy_trace)

st.plotly_chart(fig_network, use_container_width=True, key="network_chart")

def find_nearest_military_bases_to_lines(line_positions, military_bases, top_n=3):
    nearest_bases = []
    for line, (line_x, line_y) in line_positions.items():
        distances = []
        for base in military_bases:
            distance = haversine(line_y, line_x, base["latitude"], base["longitude"])
            distances.append((base["name"], distance))
        distances.sort(key=lambda x: x[1])
        nearest_bases.append((line, distances[:top_n]))
    return nearest_bases

# Find the three highest loaded lines
highest_loaded_lines = net.res_line.nlargest(3, 'loading_percent').index

# Calculate the midpoint of each line
line_positions = {}
for idx in highest_loaded_lines:
    line = net.line.loc[idx]
    u = line["from_bus"]
    v = line["to_bus"]
    if u in pos and v in pos:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        line_positions[idx] = (mid_x, mid_y)

# Find the nearest military bases to the highest loaded lines
nearest_bases_to_lines = find_nearest_military_bases_to_lines(line_positions, military_bases)

# Display the results on the dashboard
st.subheader("Nearest Military Bases to Highly Loaded Lines")
for line, bases in nearest_bases_to_lines:
    line_loading = net.res_line.at[line, 'loading_percent']
    st.write(f"Line {line} (Loading: {line_loading:.1f}%) is closest to the following military bases:")
    table_data = []
    for base, distance in bases:
        table_data.append({"Base": base, "Distance (km)": distance})
    st.table(pd.DataFrame(table_data))

# Define effect functions before using them
def small_modular_reactors_effect():
    st.write('Effect for Small Modular Reactors')
    # Add implementation here

def cyber_attack_effect():
    power_plants_path = r'data/Power_Plants_georgia.geojson'
    power_plants = gpd.read_file(power_plants_path)
    # Function to set a random top 5 plant's production to their max installed MW
    # Sort plants by Install_MW and select the top 5
    top5_plants = power_plants.nlargest(5, 'Install_MW')
    
    # Randomly select one of the top 5 plants
    selected_plant = top5_plants.sample(1).iloc[0]
    
    # Set the selected plant's production to their max installed MW
    power_plants.loc[power_plants['Plant_Name'] == selected_plant['Plant_Name'], 'Total_MW'] = selected_plant['Install_MW']
    
    st.write(f"{selected_plant['Plant_Name']} production set to max installed MW ({selected_plant['Install_MW']} MW)")
    st.write('Effect for Cyber Attack')

def fossil_fuel_outage_effect():
    power_plants_path = r'data/Power_Plants_georgia.geojson'
    power_plants = gpd.read_file(power_plants_path)
    
    # List of fossil fuel sources and their domestic production percentages
    fossil_fuel_sources = {
        'coal': 0.90,        # 90% of coal is domestic
        'petroleum': 0.34,   # 34% of petroleum is domestic
        'natural gas': 0.95  # 95% of natural gas is domestic
    }
    
    # Set the production of all fossil fuel plants to match their domestic production percentages
    for source, percentage in fossil_fuel_sources.items():
        power_plants.loc[power_plants['PrimSource'] == source, 'Total_MW'] *= percentage
    
    st.write('Production of all coal, petroleum, and natural gas plants adjusted to match their domestic production percentages')
    st.write(power_plants[['Plant_Name', 'PrimSource', 'Total_MW']])
    st.write('Effect for Fossil Fuel Shortage')

# Organize buttons into two sections
st.header("Mitigation Actions")
if st.button('Small Modular Reactors'):
    small_modular_reactors_effect()

st.header("Shock Instances")
if st.button('Cyber Attack'):
    cyber_attack_effect()

if st.button('Fossil Fuel Shortage'):
    fossil_fuel_outage_effect()
