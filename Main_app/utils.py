import pandapower as pp
import folium
import streamlit as st
import os

def load_network(network_name="original_network.pkl"):
    """
    Loads a pandapower network from a pickle file with the specified name.
    The file is assumed to be located in the "Main_app" folder.
    
    Parameters:
        network_name (str): Name of the pickle file (default "original_network.pkl").
        
    Returns:
        net: The loaded pandapower network.
    """
    file_path = os.path.join("Main_app", network_name)
    net = pp.from_pickle(file_path)
    return net

def folium_plot(net):
    """
    Creates a Folium map for the pandapower network.
    
    - Each bus (using net.bus_geodata) is represented by a marker with detailed popup info.
    - Network lines are drawn between buses with a label at the midpoint showing the line ID.
    
    If net.bus_geodata is missing or empty, an error is shown.
    """
    if not hasattr(net, "bus_geodata") or net.bus_geodata.empty:
        st.error("The network does not have bus_geodata with x and y coordinates.")
        return None

    mean_lat = net.bus_geodata["y"].mean()
    mean_lon = net.bus_geodata["x"].mean()
    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=6)

    bus_details = {}
    for bus in net.bus_geodata.index:
        details = f"<b>Bus {bus}</b><br>"
        if "name" in net.bus.columns:
            details += f"Name: {net.bus.at[bus, 'name']}<br>"
        if "vn_kv" in net.bus.columns:
            details += f"Voltage: {net.bus.at[bus, 'vn_kv']} kV<br>"
        # Connected lines for this bus
        connected_lines = net.line[(net.line["from_bus"] == bus) | (net.line["to_bus"] == bus)]
        if not connected_lines.empty:
            details += "<br><b>Connected Lines:</b><br>"
            for idx, line in connected_lines.iterrows():
                details += f"Line {idx}: {line['from_bus']} ↔ {line['to_bus']}<br>"
        # Generators at this bus
        gens = net.gen[net.gen["bus"] == bus]
        if not gens.empty:
            details += "<br><b>Generators:</b><br>"
            for idx, gen in gens.iterrows():
                p_mw = gen.get("p_mw", "N/A")
                details += f"Generator {idx}: max P = {gen['max_p_mw']} MW, P = {p_mw} MW<br>"
        # Loads at this bus
        loads = net.load[net.load["bus"] == bus]
        if not loads.empty:
            details += "<br><b>Loads:</b><br>"
            for idx, load in loads.iterrows():
                details += f"Load {idx}: P = {load['p_mw']} MW<br>"

        bus_details[bus] = details

    # Add bus markers with popups
    for bus in net.bus_geodata.index:
        lat = net.bus_geodata.at[bus, "y"]
        lon = net.bus_geodata.at[bus, "x"]
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(bus_details[bus], max_width=300),
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)

    # Draw network lines with mid-point labels
    for idx, line in net.line.iterrows():
        from_bus = line["from_bus"]
        to_bus = line["to_bus"]
        latlon_from = [net.bus_geodata.at[from_bus, "y"], net.bus_geodata.at[from_bus, "x"]]
        latlon_to = [net.bus_geodata.at[to_bus, "y"], net.bus_geodata.at[to_bus, "x"]]
        folium.PolyLine(
            locations=[latlon_from, latlon_to],
            color="black",
            weight=2,
            popup=f"Line {idx}"
        ).add_to(m)
        mid_lat = (latlon_from[0] + latlon_to[0]) / 2
        mid_lon = (latlon_from[1] + latlon_to[1]) / 2
        folium.map.Marker(
            [mid_lat, mid_lon],
            icon=folium.DivIcon(
                html=f'<div style="font-size: 12pt; color: red;">{idx}</div>'
            )
        ).add_to(m)
    return m


import streamlit as st
import pandapower as pp
import pandas as pd
import numpy as np
import networkx as nx
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def plot_network_heatmaps(net):
    """
    Creates an interactive Plotly figure with two heatmap‐like views:
      - Left: Bus Overload Heatmap – each bus is plotted using its (x, y) coordinates (from net.bus_geodata)
        and colored by its overload percentage computed from the voltage (vm_pu).
      - Right: Line Loading Heatmap – for each line, the midpoint is computed and a marker is plotted
        colored by the line loading (loading_percent).
    
    If bus_geodata is not available, a spring layout is computed.
    """
    # Get bus coordinates (preferably from net.bus_geodata)
    if hasattr(net, "bus_geodata") and not net.bus_geodata.empty:
        bus_coords = {bus: (net.bus_geodata.at[bus, "x"], net.bus_geodata.at[bus, "y"])
                      for bus in net.bus_geodata.index}
    else:
        # Fallback to a spring layout (using only the bus indices)
        G = nx.from_pandas_edgelist(net.line, source="from_bus", target="to_bus")
        pos = nx.spring_layout(G, seed=42)
        bus_coords = pos

    # Compute bus overload from voltage data (using net.res_bus if available)
    bus_overloads = {}
    if hasattr(net, "res_bus") and not net.res_bus.empty:
        for bus in net.bus.index:
            vm = net.res_bus.at[bus, "vm_pu"] if bus in net.res_bus.index else 1.0
            if vm < 0.98:
                overload = (0.98 - vm) * 100
            elif vm > 1.02:
                overload = (vm - 1.02) * 100
            else:
                overload = 0
            bus_overloads[bus] = overload
    else:
        # Default: no overload
        for bus in net.bus.index:
            bus_overloads[bus] = 0

    # For line loading, use net.res_line if available
    line_midpoints_x = []
    line_midpoints_y = []
    line_loadings = []
    if hasattr(net, "res_line") and not net.res_line.empty:
        for idx, line in net.line.iterrows():
            loading = net.res_line.at[idx, "loading_percent"] if idx in net.res_line.index else 0
            from_bus = line["from_bus"]
            to_bus = line["to_bus"]
            if from_bus in bus_coords and to_bus in bus_coords:
                x0, y0 = bus_coords[from_bus]
                x1, y1 = bus_coords[to_bus]
                mid_x = (x0 + x1) / 2
                mid_y = (y0 + y1) / 2
                line_midpoints_x.append(mid_x)
                line_midpoints_y.append(mid_y)
                line_loadings.append(loading)
    else:
        # If no line results available, return empty lists
        line_midpoints_x = []
        line_midpoints_y = []
        line_loadings = []

    # Create two subplots: one for bus overload, one for line loading.
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Bus Overload Heatmap", "Line Loading Heatmap"))

    # Left subplot: Bus Overload Heatmap
    bus_x = []
    bus_y = []
    bus_colors = []
    bus_text = []
    for bus, (x, y) in bus_coords.items():
        bus_x.append(x)
        bus_y.append(y)
        overload = bus_overloads.get(bus, 0)
        bus_colors.append(overload)
        bus_text.append(f"Bus {bus}<br>Overload: {overload:.1f}%")
    
    trace_buses = go.Scatter(
        x=bus_x,
        y=bus_y,
        mode="markers",
        marker=dict(
            size=15,
            color=bus_colors,
            colorscale="Reds",
            colorbar=dict(title="Overload (%)"),
            cmin=0,
            cmax=max(bus_colors) if bus_colors else 1
        ),
        text=bus_text,
        hoverinfo="text"
    )
    fig.add_trace(trace_buses, row=1, col=1)

    # Right subplot: Line Loading Heatmap
    trace_lines = go.Scatter(
        x=line_midpoints_x,
        y=line_midpoints_y,
        mode="markers",
        marker=dict(
            size=15,
            color=line_loadings,
            colorscale="Blues",
            colorbar=dict(title="Line Loading (%)"),
            cmin=0,
            cmax=max(line_loadings) if line_loadings else 1
        ),
        text=[f"Line {idx}<br>Loading: {loading:.1f}%" 
              for idx, loading in zip(net.line.index, line_loadings)],
        hoverinfo="text"
    )
    fig.add_trace(trace_lines, row=1, col=2)

    fig.update_layout(title_text="Network Heatmaps", showlegend=False)
    return fig

# Example usage in your Streamlit page:
# st.title("Network Heatmaps")
# net = load_network()  # Make sure the network has been updated (OPF results available)
# fig_heatmaps = plot_network_heatmaps(net)
# st.plotly_chart(fig_heatmaps, use_container_width=True)


import os
import math
import pandas as pd
import pandapower as pp
import pandapower.plotting as plot
import pandapower.plotting.plotly as pp_plotly
from pandapower.plotting.plotly import simple_plotly
import matplotlib.pyplot as plt
import plotly.io as pio
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pandapower.topology import create_nxgraph
import pickle

# Transformer parameters dictionary (example)
transformer_types = {
    (500, 230): {
        "sn_mva": 100,
        "vk_percent": 12.0,
        "vkr_percent": 0.5,
        "pfe_kw": 150,
        "i0_percent": 0.15,
    },
    (230, 115): {
        "sn_mva": 400,
        "vk_percent": 10.0,
        "vkr_percent": 0.4,
        "pfe_kw": 50,
        "i0_percent": 0.07,
    },
    (115, 20): {
        "sn_mva": 100,
        "vk_percent": 8.0,
        "vkr_percent": 0.5,
        "pfe_kw": 50,
        "i0_percent": 0.1,
    },
    (33, 20): {
        "sn_mva": 20,
        "vk_percent": 6.0,
        "vkr_percent": 0.7,
        "pfe_kw": 20,
        "i0_percent": 0.2,
    },
    (20, 0.4): {
        "sn_mva": 2,
        "vk_percent": 4.0,
        "vkr_percent": 1.0,
        "pfe_kw": 10,
        "i0_percent": 0.3,
    },
}

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance (in km) between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

def create_network_from_filtered_data(output_folder=r"Main_app\output_pandapower", filename_prefix="original_network"):
    """
    Creates a Pandapower network from filtered data and saves it as a pickle file.
    Also saves a text file listing each load (by index) and its type (Fractional/Fixed).
    """
    def load_filtered_data(file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        # Check if file is not empty
        if os.stat(file_path).st_size == 0:
            raise ValueError(f"File is empty: {file_path}")
        return pd.read_csv(file_path)

    buses_path = os.path.join("Main_app\output_pandapower", "buses.csv")
    loads_path = os.path.join("Main_app\output_pandapower", "loads.csv")
    generators_path = os.path.join("Main_app\output_pandapower", "generators.csv")

    buses_df = load_filtered_data(buses_path)
    loads_df = load_filtered_data(loads_path)
    generators_df = load_filtered_data(generators_path)

    
    # Create an empty Pandapower network
    net = pp.create_empty_network()

    # Create buses (and store mapping from bus_id to internal index)
    bus_index_map = {}
    unique_positions = {}
    pos_bus = {}
    offset = 0.0001  # small offset to avoid overlap

    for i, bus in buses_df.iterrows():
        position = (bus['x'], bus['y'])
        # Check for duplicates and apply an offset if necessary
        if position in unique_positions:
            bus['x'] += offset
            bus['y'] += offset
            position = (bus['x'], bus['y'])
        unique_positions[position] = bus['bus_id']
        pos_bus[bus['bus_id']] = position

        bus_index = pp.create_bus(net, name=bus['bus_id'], vn_kv=bus['vn_kv'], geodata=position, type='b')
        bus_index_map[bus['bus_id']] = bus_index

    # Add transformers if transformers.csv exists
    transformers_path = os.path.join(output_folder, "transformers.csv")
    if os.path.exists(transformers_path):
        print("Adding transformers...")
        try:
            transformers_df = pd.read_csv(transformers_path)
        except pd.errors.EmptyDataError:
            transformers_df = pd.DataFrame()  # File is empty, so use an empty DataFrame

        if transformers_df.empty:
            print("Transformers file is empty. Skipping transformer addition.")
        else:
            for _, transformer in transformers_df.iterrows():
                try:
                    if transformer['from_bus'] not in net.bus.name.values:
                        raise ValueError(f"from_bus {transformer['from_bus']} does not exist in the network.")
                    if transformer['to_bus'] not in net.bus.name.values:
                        raise ValueError(f"to_bus {transformer['to_bus']} does not exist in the network.")

                    max_kV = max(net.bus.loc[net.bus.name == transformer['from_bus'], 'vn_kv'].values[0],
                                net.bus.loc[net.bus.name == transformer['to_bus'], 'vn_kv'].values[0])
                    min_kV = min(net.bus.loc[net.bus.name == transformer['from_bus'], 'vn_kv'].values[0],
                                net.bus.loc[net.bus.name == transformer['to_bus'], 'vn_kv'].values[0])
                    voltage_jump = (max_kV, min_kV)
                    if voltage_jump in transformer_types:
                        params = transformer_types[voltage_jump]
                        pp.create_transformer_from_parameters(
                            net,
                            hv_bus=net.bus.loc[net.bus.name == transformer['from_bus']].index[0],
                            lv_bus=net.bus.loc[net.bus.name == transformer['to_bus']].index[0],
                            sn_mva=params['sn_mva'],
                            vn_hv_kv=max_kV,
                            vn_lv_kv=min_kV,
                            vkr_percent=params['vkr_percent'],
                            vk_percent=params['vk_percent'],
                            pfe_kw=params['pfe_kw'],
                            i0_percent=params['i0_percent'],
                            name=f"Transformer {transformer['from_bus']}->{transformer['to_bus']}",
                            max_loading_percent=200
                        )
                    else:
                        print(f"No transformer type defined for voltage jump: {voltage_jump}")
                except Exception as e:
                    print(f"Error adding transformer: {transformer.to_dict()}")
                    raise e
    else:
        print("Transformers file does not exist. Skipping transformer addition.")

    # Create lines based on shared line IDs between buses
    added_lines = set()
    for _, bus1 in buses_df.iterrows():
        for _, bus2 in buses_df.iterrows():
            if bus1['bus_id'] != bus2['bus_id']:
                common_lines = set(eval(bus1['lines'])) & set(eval(bus2['lines']))
                for line_id in common_lines:
                    if (bus1['bus_id'], bus2['bus_id'], line_id) not in added_lines and \
                       (bus2['bus_id'], bus1['bus_id'], line_id) not in added_lines:
                        try:
                            if bus1['vn_kv'] == 500:
                                pp.create_line_from_parameters(
                                    net,
                                    from_bus=bus_index_map[bus1['bus_id']],
                                    to_bus=bus_index_map[bus2['bus_id']],
                                    length_km=haversine(bus1['x'], bus1['y'], bus2['x'], bus2['y']),
                                    r_ohm_per_km=0.01,
                                    x_ohm_per_km=0.25,
                                    c_nf_per_km=12,
                                    max_i_ka=2.0,
                                    name=f"Line {line_id}"
                                )
                            else:
                                pp.create_line_from_parameters(
                                    net,
                                    from_bus=bus_index_map[bus1['bus_id']],
                                    to_bus=bus_index_map[bus2['bus_id']],
                                    length_km=haversine(bus1['x'], bus1['y'], bus2['x'], bus2['y']),
                                    r_ohm_per_km=0.1,
                                    x_ohm_per_km=0.4,
                                    c_nf_per_km=9,
                                    max_i_ka=1.5,
                                    name="230kV Line"
                                )
                            added_lines.add((bus1['bus_id'], bus2['bus_id'], line_id))
                        except Exception as e:
                            print(f"Error creating line between {bus1['bus_id']} and {bus2['bus_id']} for Line {line_id}")
                            raise e

    # Add loads from filtered data
    for _, load in loads_df.iterrows():
        try:
            departure_bus = bus_index_map[load['bus']]
            pp.create_load(
                net,
                bus=departure_bus,
                p_mw=load['p_kw'] * 0.001,  # Convert kW to MW
                max_p_mw=load['p_kw'] * 0.001,
                min_p_mw=0,
                name=load['name'],
                max_q_mvar=load['p_kw'] * 0.001 / 0.9,
                min_q_mvar=0,
                controllable=False
            )
        except Exception as e:
            print(f"Error creating load: {load}")
            raise e

    # Add generators from filtered data
    slack_assigned = False
    for _, gen in generators_df.iterrows():
        try:
            ref_bus_id = gen['bus_id']
            if ref_bus_id not in pos_bus:
                print(f"Position not found for bus {ref_bus_id}; using original bus.")
                ref_pos = None
            else:
                ref_pos = pos_bus[ref_bus_id]
            nearby_bus_ids = []
            if ref_pos is not None:
                for bus_id, pos in pos_bus.items():
                    if haversine(ref_pos[0], ref_pos[1], pos[0], pos[1]) <= 1:
                        nearby_bus_ids.append(bus_id)
            else:
                nearby_bus_ids = [ref_bus_id]
            if len(nearby_bus_ids) > 1:
                generation_share = gen['p_mw'] / len(nearby_bus_ids)
                for bus in nearby_bus_ids:
                    bus_index = bus_index_map[bus]
                    pp.create_gen(
                        net,
                        bus=bus_index,
                        p_mw=1,
                        vm_pu=1.0,
                        min_p_mw=0,
                        max_p_mw=generation_share,
                        min_q_mvar=-generation_share / 0.85,
                        max_q_mvar=generation_share / 0.85,
                        name=f"{gen['generator_name']} ({gen['type']}) - Part on bus {bus}",
                        slack=(not slack_assigned and gen['type'] == 'natural gas'),
                        controllable=True
                    )
                    if not slack_assigned and gen['type'] == 'natural gas':
                        slack_assigned = True
            else:
                bus = nearby_bus_ids[0]
                bus_index = bus_index_map[bus]
                pp.create_gen(
                    net,
                    bus=bus_index,
                    p_mw=1,
                    vm_pu=1.0,
                    min_p_mw=0,
                    max_p_mw=gen['p_mw'],
                    min_q_mvar=-gen['p_mw'] / 0.85,
                    max_q_mvar=gen['p_mw'] / 0.85,
                    name=f"{gen['generator_name']} ({gen['type']})",
                    slack=(not slack_assigned and gen['type'] == 'natural gas'),
                    controllable=True
                )
                if not slack_assigned and gen['type'] == 'natural gas':
                    slack_assigned = True
        except Exception as e:
            print(f"Error creating generator {gen['generator_name']}: {e}")
            raise e

    # Add external grid if no slack generator assigned
    if not slack_assigned:
        pp.create_ext_grid(net, bus=0, vm_pu=1.0, va_degree=0)

    # Remove isolated buses and inactive elements
    isolated_buses = net.bus.index.difference(pd.concat([net.line["from_bus"], net.line["to_bus"], net.trafo["hv_bus"], net.trafo["lv_bus"]]))
    net.bus.drop(isolated_buses, inplace=True)
    for component in ['line', 'trafo', 'gen', 'load']:
        invalid_entries = net[component][net[component]['in_service'] == False].index
        net[component].drop(invalid_entries, inplace=True)

    # Set generator voltage constraints
    net.gen['min_vm_pu'] = 0.98
    net.gen['max_vm_pu'] = 1.02

    # Save the network to a pickle file
    output_pickle = os.path.join(output_folder, f"{filename_prefix}.p")
    pp.to_pickle(net, output_pickle)
    print(f"Network saved as pickle file: {output_pickle}")

    # Save load types to a text file (one line per load)
    output_txt = os.path.join(output_folder, f"{filename_prefix}.txt")
    with open(output_txt, "w") as f:
        for idx in net.load.index:
            frac = net.load.at[idx, "fraction"] if "fraction" in net.load.columns else True
            load_type = "Fractional" if frac else "Fixed"
            f.write(f"Load {idx}: {load_type}\n")
    print(f"Load types saved as text file: {output_txt}")

    return net  # Optionally, return the created network

# Run the function to create and save the network
if __name__ == "__main__":
    created_net = create_network_from_filtered_data()
