import streamlit as st
import os
import pandapower as pp
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from plotly.subplots import make_subplots
from pandapower.topology import create_nxgraph
from utils import load_network  # Ensure that this module is available
import copy  # To perform a deep copy of the network
import re
import pickle

###############################################################################
# Loading the Reference Network
###############################################################################
original_network = pp.from_pickle("Main_app/original_network.p")

###############################################################################
# Initializing the Streamlit Session
###############################################################################
if "sim_history" not in st.session_state:
    st.session_state["sim_history"] = []
if "error_log" not in st.session_state:
    st.session_state["error_log"] = []

st.set_page_config(page_title="Simulation & Network Visualization", layout="wide")
st.title("Simulation & Network Visualization")

###############################################################################
# Simulation Section
###############################################################################
st.header("Simulation Operation")

# Upload a network pickle file via the sidebar
uploaded_file = st.sidebar.file_uploader("Choose a network pickle file", type=["p", "pkl"])

if uploaded_file is not None:
    st.sidebar.write(f"Selected file: {uploaded_file.name}")
    if st.sidebar.button("Load Network", key="load_net"):
        try:
            # Load the network directly from the uploaded file object
            net = pp.from_pickle(uploaded_file)
            st.session_state["net"] = net  # Store the network in the session
            st.sidebar.success("Network loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading network: {e}")
else:
    st.sidebar.info("Please upload a network file to load.")

if "net" in st.session_state and uploaded_file is not None:
    net = st.session_state["net"]

    # Extract base name from the uploaded file to load load-type info
    if "net" in st.session_state:
        uploaded_name = os.path.splitext(uploaded_file.name)[0]
        txt_filepath = os.path.join(r"Main_app\modified_network", f"{uploaded_name}.txt")
        if os.path.exists(txt_filepath):
            st.sidebar.info(f"Loading load type info from {txt_filepath}...")
            with open(txt_filepath, "r") as f:
                for line in f:
                    # Line format: "Load 44: Fractional" or "Load 44: Fixed"
                    m = re.match(r"Load\s+(\d+):\s*(\w+)", line)
                    if m:
                        load_idx = int(m.group(1))
                        load_type = m.group(2).strip().lower()
                        if load_idx in net.load.index:
                            net.load.at[load_idx, "fraction"] = True if load_type == "fractional" else False
            st.sidebar.success("Load type info loaded successfully!")
        else:
            st.sidebar.warning(f"No load info file found at {txt_filepath}. All loads remain as set.")

    # Load the forecast CSV file
    forecast_csv_path = "Main_app/saved_forecast/load_forecast.csv"
    try:
        forecast_csv = pd.read_csv(forecast_csv_path)
        forecast_csv["ds"] = pd.to_datetime(forecast_csv["ds"])
    except Exception as e:
        st.error(f"Error loading forecast CSV file: {e}")
        st.stop()
    
    # Run the initial OPF if no simulation is recorded
    if "net" in st.session_state and len(st.session_state["sim_history"]) == 0:
        try:
            pp.runopp(net)
            net_snapshot = copy.deepcopy(net)
            total_demand = net.load["p_mw"].sum()
            if hasattr(net, "res_gen") and not net.res_gen.empty:
                total_generation = net.res_gen["p_mw"].sum()
                gen_distribution = net.res_gen["p_mw"].to_dict()
                gen_names = net.gen["name"].to_dict()
            else:
                total_generation = 0
                gen_distribution = {}
                gen_names = {}
            if hasattr(net, "res_line") and not net.res_line.empty:
                avg_line_loading = net.res_line["loading_percent"].mean()
            else:
                avg_line_loading = 0
            avg_bus_load = net.load["p_mw"].mean() if not net.load.empty else 0
    
            init_time = forecast_csv.iloc[0]["ds"] if not forecast_csv.empty else pd.Timestamp.now()
    
            sim_state = {
                "time": init_time,
                "demand": total_demand,
                "generation": total_generation,
                "gen_distribution": gen_distribution,
                "gen_names": gen_names,
                "avg_line_loading": avg_line_loading,
                "avg_bus_load": avg_bus_load,
                "net_snapshot": net_snapshot,
            }
            st.session_state["sim_history"].append(sim_state)
            st.sidebar.success("Initial OPF run completed and simulation snapshot recorded.")
        except Exception as e:
            st.sidebar.error(f"Error running initial OPF: {e}")
    
    # Save the initial load values if not already stored
    if "original_loads" not in st.session_state:
        st.session_state["original_loads"] = net.load["p_mw"].to_dict()
    original_loads = st.session_state["original_loads"]
    
    # Simulation Controls in the Sidebar
    st.sidebar.subheader("Simulation Controls")
    step_size = st.sidebar.number_input("Time Steps to Advance", min_value=1, value=1, step=1)
    advance_btn = st.sidebar.button("Advance Simulation by Step Size", key="advance_btn")
    run_full_btn = st.sidebar.button("Run Full Simulation", key="run_full_btn")
    reset_btn = st.sidebar.button("Reset Simulation", key="reset_btn")
    
    if reset_btn:
        st.session_state["sim_history"] = []
        st.session_state["error_log"] = []
        st.sidebar.success("Simulation reset successfully.")
        
    current_step = len(st.session_state["sim_history"])
    total_steps = len(forecast_csv)
    
    def run_simulation_steps(start, end):
        for t in range(start, end):
            # Re-enable all generators before each timestep
            net.gen["in_service"] = True
    
            # Get the forecasted load (in MW) for timestep t
            load_forecast = forecast_csv.iloc[t]["yhat"]
            total_orig = sum(original_loads.values())
            for idx, orig_val in original_loads.items():
                try:
                    if "fraction" in net.load.columns and net.load.at[idx, "fraction"]:
                        new_load = load_forecast * (orig_val / total_orig)
                        net.load.at[idx, "p_mw"] = new_load
                except KeyError:
                    continue
                
            # Execute the OPF; if it fails, log the error and skip this timestep.
            try:
                pp.runopp(net)
            except Exception as e:
                st.session_state["error_log"].append(f"OPF did not converge at timestep {t}; skipping this timestep.")
                continue
            
            net_snapshot = copy.deepcopy(net)
    
            total_demand = net.load["p_mw"].sum()
            if hasattr(net, "res_gen") and not net.res_gen.empty:
                total_generation = net.res_gen["p_mw"].sum()
                gen_distribution = net.res_gen["p_mw"].to_dict()
                gen_names = net.gen["name"].to_dict()
            else:
                total_generation = 0
                gen_distribution = {}
                gen_names = {}
            if hasattr(net, "res_line") and not net.res_line.empty:
                avg_line_loading = net.res_line["loading_percent"].mean()
            else:
                avg_line_loading = 0
            avg_bus_load = net.load["p_mw"].mean() if not net.load.empty else 0
    
            sim_state = {
                "time": forecast_csv.iloc[t]["ds"],
                "demand": total_demand,
                "generation": total_generation,
                "gen_distribution": gen_distribution,
                "gen_names": gen_names,
                "avg_line_loading": avg_line_loading,
                "avg_bus_load": avg_bus_load,
                "net_snapshot": net_snapshot,
            }
            st.session_state["sim_history"].append(sim_state)
    
    if advance_btn:
        new_step = min(current_step + step_size, total_steps)
        run_simulation_steps(current_step, new_step)
        st.sidebar.success(f"Simulation advanced to step {len(st.session_state['sim_history'])}")
    
    if run_full_btn:
        run_simulation_steps(current_step, total_steps)
        st.sidebar.success("Full simulation run completed.")
    
    ###############################################################################
    # Displaying Simulation History
    ###############################################################################
    st.subheader("Simulation History")
    if st.session_state["sim_history"]:
        max_step = len(st.session_state["sim_history"])
        if max_step > 1:
            selected_step = st.sidebar.slider("Select Simulation Timestep", 1, max_step, max_step, key="sim_slider")
        else:
            selected_step = 1
    
        history = st.session_state["sim_history"][:selected_step]
        times = [entry["time"] for entry in history]
        demands = [entry["demand"] for entry in history]
        generations = [entry["generation"] for entry in history]
    
        fig_sim = go.Figure(data=[
            go.Bar(name="Demand (MW)", x=times, y=demands, marker_color="red"),
            go.Bar(name="Generation (MW)", x=times, y=generations, marker_color="blue")
        ])
        fig_sim.update_layout(barmode="group",
                              title="Demand & Generation Over Time",
                              xaxis_title="Time",
                              yaxis_title="MW",
                              template="plotly_white")
        st.plotly_chart(fig_sim, use_container_width=True, key="sim_chart")
    else:
        st.info("No simulation data available yet. Please advance the simulation.")
    
    # Saving the Simulation
    sim_prefix = st.sidebar.text_input("Enter simulation file name", value=uploaded_file.name.split(".")[0])
    if st.sidebar.button("Save Entire Simulation", key="save_full_sim"):
        save_folder = os.path.join("Main_app", "simulations", sim_prefix)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            
        sim_history = st.session_state.get("sim_history", [])
        if not sim_history:
            st.sidebar.error("No simulation data available to save.")
        else:
            for i, step in enumerate(sim_history, start=1):
                net_snapshot = step.get("net_snapshot", None)
                if net_snapshot is not None:
                    filename = os.path.join(save_folder, f"{sim_prefix}_timestep_{i}.pkl")
                    try:
                        with open(filename, "wb") as f:
                            pickle.dump(net_snapshot, f)
                    except Exception as e:
                        st.sidebar.error(f"Error saving timestep {i}: {e}")
                else:
                    st.sidebar.warning(f"No network snapshot for timestep {i}.")
            st.sidebar.success("Simulation saved successfully!")
    
    filename_ts = st.sidebar.text_input("Enter filename for selected timestep", value="selected_timestep.pkl", key="ts_filename")
    if st.sidebar.button("Save Selected Timestep", key="save_ts"):
        selected_data = st.session_state["sim_history"][selected_step - 1]
        with open(os.path.join(r"Main_app\simulations\individual_timesteps", filename_ts), "wb") as f:
            pickle.dump(selected_data, f)
        st.sidebar.success("Selected timestep saved successfully!")
    
    ###############################################################################
    # Network Visualization (Standard Graph with Heatmaps)
    ###############################################################################
    st.header("Network Visualization with Heatmaps")
    
    # Use the snapshot from the selected timestep to visualize the network
    if st.session_state["sim_history"]:
        sim_snapshot = st.session_state["sim_history"][selected_step - 1]
        net_snapshot = sim_snapshot.get("net_snapshot", None)
    else:
        net_snapshot = None
    
    if net_snapshot is not None:
        G = create_nxgraph(net_snapshot, respect_switches=True)
    else:
        G = create_nxgraph(net, respect_switches=True)
    
    if hasattr(net, "bus_geodata") and not net.bus_geodata.empty:
        pos = {bus: (net.bus_geodata.at[bus, "x"], net.bus_geodata.at[bus, "y"])
               for bus in net.bus_geodata.index}
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Use the snapshot for bus voltages if available.
    if net_snapshot is not None and hasattr(net_snapshot, "res_bus") and not net_snapshot.res_bus.empty:
        res_bus_snapshot = net_snapshot.res_bus
    else:
        res_bus_snapshot = net.res_bus if hasattr(net, "res_bus") else pd.DataFrame()
    
    if res_bus_snapshot is not None and not res_bus_snapshot.empty:
        for bus in G.nodes():
            if bus in res_bus_snapshot.index:
                vm = res_bus_snapshot.at[bus, "vm_pu"]
            else:
                vm = 1.0
            G.nodes[bus]["vm_pu"] = vm
    else:
        for bus in G.nodes():
            G.nodes[bus]["vm_pu"] = 1.0
    
    # Calculate the overload for each bus
    for bus in G.nodes():
        vm = G.nodes[bus].get("vm_pu", 1.0)
        if vm < 0.98:
            overload = (0.98 - vm) * 100
        elif vm > 1.02:
            overload = (vm - 1.02) * 100
        else:
            overload = 0
        G.nodes[bus]["overload"] = overload
    
    node_x, node_y, node_color, node_text = [], [], [], []
    for bus in G.nodes():
        x, y = pos[bus]
        node_x.append(x)
        node_y.append(y)
        overload = G.nodes[bus]["overload"]
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
            colorbar=dict(title="Bus Overload (%)", x=0.0),  # Colormap on the left
            cmin=-10,
            cmax=10
        )
    )
    
    edge_traces = []
    edge_annotations = []
    if net_snapshot is not None and hasattr(net_snapshot, "res_line") and not net_snapshot.res_line.empty:
        res_line_snapshot = net_snapshot.res_line
    else:
        res_line_snapshot = net.res_line if hasattr(net, "res_line") else pd.DataFrame()
    
    if res_line_snapshot is not None and not res_line_snapshot.empty:
        for idx, line in net.line.iterrows():
            u = line["from_bus"]
            v = line["to_bus"]
            if u not in pos or v not in pos:
                continue
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            loading = res_line_snapshot.at[idx, "loading_percent"] if idx in res_line_snapshot.index else 0
            cmap = cm.get_cmap("viridis")  # Use viridis for line loading colors
            norm = mcolors.Normalize(vmin=0, vmax=100)
            rgba = cmap(norm(loading))
            hex_color = mcolors.to_hex(rgba)
            edge_trace = go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color=hex_color, width=3),
                hoverinfo="text",
                text=f"Line {idx}<br>Loading: {loading:.1f}%",
                showlegend=False
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
    else:
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_trace = go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color="grey", width=2),
                hoverinfo="none",
                showlegend=False
            )
            edge_traces.append(edge_trace)
    
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
    # Dummy trace for the line loading colorbar in viridis
    dummy_trace = go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(
            colorscale="viridis",
            showscale=True,
            cmin=0,
            cmax=100,
            colorbar=dict(title="Line Loading (%)", x=1.0)  # Colorbar for line loading on the right
        ),
        hoverinfo="none",
        showlegend=False
    )
    fig_network.add_trace(dummy_trace)
    st.plotly_chart(fig_network, use_container_width=True, key="network_chart")
    
    # Display pie charts for generation data
    if "sim_history" in st.session_state and st.session_state["sim_history"]:
        current_state = st.session_state["sim_history"][0]
        total_gen = current_state["generation"]
        if total_gen > 0:
            gen_dist = current_state["gen_distribution"]
            gen_names = current_state["gen_names"]
        else:
            gen_dist = {}
            gen_names = {}
        if gen_dist:
            labels = [gen_names[gen_id] for gen_id in gen_dist.keys()]
            values = list(gen_dist.values())
            fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
            fig_pie.update_layout(title="Generation Distribution by Generator")
            st.plotly_chart(fig_pie, use_container_width=True, key="pie_chart")
            
            types = [re.findall(r'\((.*?)\)', gen_names[gen_id])[0] for gen_id in gen_dist.keys()]
            unique_types = list(set(types))
            type_values = [0 for _ in range(len(unique_types))]
            for i, label in enumerate(labels):
                gen_type = types[i]
                type_idx = unique_types.index(gen_type)
                type_values[type_idx] += values[i]
            fig_pie2 = go.Figure(data=[go.Pie(labels=unique_types, values=type_values, hole=0.3)])
            fig_pie2.update_layout(title="Generation Distribution by Type")
            st.plotly_chart(fig_pie2, use_container_width=True, key="pie_chart2")
        else:
            st.info("No generation data available for the first timestep.")
    else:
        st.info("No simulation data available yet.")
    
    # Display error messages
    if st.session_state["error_log"]:
        st.subheader("Simulation Messages")
        for msg in st.session_state["error_log"]:
            st.error(msg)
    
    ###############################################################################
    # Network Overlaid on the Map of Georgia (using Scattermapbox)
    ###############################################################################
    st.header("Heat map")
    
    # Check that geographical coordinates are available in net.bus_geodata
    if hasattr(net, "bus_geodata") and not net.bus_geodata.empty:
        # Here, the 'x' and 'y' columns correspond to longitude and latitude respectively
        pos = {bus: (net.bus_geodata.at[bus, "x"], net.bus_geodata.at[bus, "y"])
               for bus in net.bus_geodata.index}
    else:
        st.error("Geographic data are not available to overlay the network on the map.")
        pos = nx.spring_layout(G, seed=42)
    
    # Prepare node data for Scattermapbox
    node_lon, node_lat, node_color, node_text = [], [], [], []
    for bus in G.nodes():
        x, y = pos[bus]
        node_lon.append(x)
        node_lat.append(y)
        overload = G.nodes[bus].get("overload", 0)
        node_color.append(overload)
        label = f"Bus {bus}<br>Overload: {overload:.1f}%" if overload > 0 else f"Bus {bus}"
        node_text.append(label)
    
    node_trace_map = go.Scattermapbox(
        lon=node_lon,
        lat=node_lat,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hoverinfo="text",
        marker=dict(
            size=15,
            color=node_color,
            colorscale="Reds",
            colorbar=dict(
                title="Overload (%)",
                x=-0.08,           # Colorbar on the left
                xanchor="left",
                y=0.5,
                yanchor="middle",
                len=0.8
            ),
            cmin=-10,
            cmax=10
        ),
        showlegend=False
    )
    
    # Prepare traces for lines on the map
    edge_traces_map = []
    if res_line_snapshot is not None and not res_line_snapshot.empty:
        for idx, line in net.line.iterrows():
            u = line["from_bus"]
            v = line["to_bus"]
            if u not in pos or v not in pos:
                continue
            lon0, lat0 = pos[u]
            lon1, lat1 = pos[v]
            loading = res_line_snapshot.at[idx, "loading_percent"] if idx in res_line_snapshot.index else 0
            cmap = cm.get_cmap("viridis")
            norm = mcolors.Normalize(vmin=0, vmax=100)
            rgba = cmap(norm(loading))
            hex_color = mcolors.to_hex(rgba)
            edge_trace = go.Scattermapbox(
                lon=[lon0, lon1],
                lat=[lat0, lat1],
                mode="lines",
                line=dict(color=hex_color, width=3),
                hoverinfo="text",
                text=f"Line {idx}<br>Loading: {loading:.1f}%",
                showlegend=False
            )
            edge_traces_map.append(edge_trace)
            if loading > 100:
                mid_lon = (lon0 + lon1) / 2
                mid_lat = (lat0 + lat1) / 2
                annotation = go.Scattermapbox(
                    lon=[mid_lon],
                    lat=[mid_lat],
                    mode="text",
                    text=[f"{loading:.1f}%"],
                    textfont=dict(color="red", size=15),
                    hoverinfo="none",
                    showlegend=False
                )
                edge_traces_map.append(annotation)
    else:
        for u, v in G.edges():
            if u not in pos or v not in pos:
                continue
            lon0, lat0 = pos[u]
            lon1, lat1 = pos[v]
            edge_trace = go.Scattermapbox(
                lon=[lon0, lon1],
                lat=[lat0, lat1],
                mode="lines",
                line=dict(color="grey", width=2),
                hoverinfo="none",
                showlegend=False
            )
            edge_traces_map.append(edge_trace)
    
    # Create the Scattermapbox figure
    fig_network_map = go.Figure(data=edge_traces_map + [node_trace_map])
    
    if node_lon and node_lat:
        center_lon = sum(node_lon) / len(node_lon)
        center_lat = sum(node_lat) / len(node_lat)
    else:
        center_lon, center_lat = -82.9001, 32.1656
    
    # Dummy trace for the line loading colorbar in viridis.
    dummy_trace_map = go.Scattermapbox(
        lon=[None],
        lat=[None],
        mode="markers",
        marker=dict(
            colorscale="viridis",
            showscale=True,
            cmin=0,
            cmax=100,
            colorbar=dict(
                title="Line Loading (%)",
                x=1.0,         # Colorbar on the right
                xanchor="left",
                y=0.5,
                yanchor="middle",
                len=0.8
            )
        ),
        hoverinfo="none",
        showlegend=False
    )
    fig_network_map.add_trace(dummy_trace_map)
    
    # Update the layout for the map (with similar dimensions and margins as the first graph)
    fig_network_map.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=7
        ),
        width=750,
        height=750,
        margin=dict(l=20, r=20, t=40, b=20),
        title="Network Graph Overlaid on the Map of Georgia",
        showlegend=False
    )
    
    st.plotly_chart(
        fig_network_map,
        use_container_width=True,
        key="network_map",
        config={
            "scrollZoom": True,  # Enable scroll zoom
            "displayModeBar": True,
            "modeBarButtonsToAdd": [
                "zoomInMapbox",
                "zoomOutMapbox",
                "resetViewMapbox"
            ]
        }
    )
