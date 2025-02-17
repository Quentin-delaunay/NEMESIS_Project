import streamlit as st
import os
import re
import pickle
import pandapower as pp
import networkx as nx
import plotly.graph_objects as go
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pandapower.topology import create_nxgraph
from streamlit_folium import st_folium

st.set_page_config(page_title="Compare Simulations", layout="wide")
st.title("Compare Simulation Timesteps")

# Sidebar: Enter folder paths for the two simulations
st.sidebar.header("Load Simulation Histories")
folder_a = st.sidebar.text_input("Enter Simulation A folder path", value="Main_app/simulations/simulationA")
folder_b = st.sidebar.text_input("Enter Simulation B folder path", value="Main_app/simulations/simulationB")

def load_simulation_files(folder):
    """List and sort all .pkl files in the folder by extracted timestep number."""
    if not os.path.exists(folder):
        st.sidebar.error(f"Folder {folder} does not exist.")
        return []
    file_list = [f for f in os.listdir(folder) if f.endswith('.pkl')]
    if not file_list:
        st.sidebar.info(f"No .pkl files found in {folder}.")
        return []
    # Extract timestep using a regex that expects a pattern like "timestep_<number>" in the filename.
    def extract_timestep(filename):
        m = re.search(r"timestep_(\d+)", filename)
        return int(m.group(1)) if m else 0
    return sorted(file_list, key=extract_timestep)

files_a = load_simulation_files(folder_a)
files_b = load_simulation_files(folder_b)

if files_a and files_b:
    # Determine the number of timesteps available (use the minimum count)
    num_steps = min(len(files_a), len(files_b))
    selected_step = st.sidebar.slider("Select Simulation Timestep", 1, num_steps, 1, key="sim_compare_slider")
    
    # Load the corresponding snapshot files from each folder
    file_a = files_a[selected_step - 1]
    file_b = files_b[selected_step - 1]
    try:
        with open(os.path.join(folder_a, file_a), "rb") as fp:
            snapshot_a = pickle.load(fp)
    except Exception as e:
        st.sidebar.error(f"Error loading {file_a} from Simulation A: {e}")
        snapshot_a = None
    try:
        with open(os.path.join(folder_b, file_b), "rb") as fp:
            snapshot_b = pickle.load(fp)
    except Exception as e:
        st.sidebar.error(f"Error loading {file_b} from Simulation B: {e}")
        snapshot_b = None

    # Define a function to create the network figure from a snapshot.
    def create_network_figure(net_snapshot):
        if net_snapshot is None:
            st.error("No network snapshot provided.")
            return None
        # Create a NetworkX graph from the snapshot
        G = create_nxgraph(net_snapshot, respect_switches=True)
        # Use geographic coordinates if available; otherwise, use spring layout.
        if hasattr(net_snapshot, "bus_geodata") and not net_snapshot.bus_geodata.empty:
            pos = {bus: (net_snapshot.bus_geodata.at[bus, "x"], net_snapshot.bus_geodata.at[bus, "y"])
                   for bus in net_snapshot.bus_geodata.index}
        else:
            pos = nx.spring_layout(G, seed=42)
        # Use res_bus if available
        if hasattr(net_snapshot, "res_bus") and not net_snapshot.res_bus.empty:
            res_bus = net_snapshot.res_bus
        else:
            res_bus = pd.DataFrame()
        for bus in G.nodes():
            vm = res_bus.at[bus, "vm_pu"] if bus in res_bus.index else 1.0
            G.nodes[bus]["vm_pu"] = vm
        # Calculate bus overload
        for bus in G.nodes():
            vm = G.nodes[bus].get("vm_pu", 1.0)
            if vm < 0.98:
                overload = (0.98 - vm) * 100
            elif vm > 1.02:
                overload = (vm - 1.02) * 100
            else:
                overload = 0
            G.nodes[bus]["overload"] = overload
        # Prepare node trace (display text only if overloaded)
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
                colorbar=dict(title="Bus Overload (%)", x=-0.1),
                cmin=0,
                cmax=max(node_color) if node_color else 1,
                line=dict(width=2)
            )
        )
        # Prepare edge traces for line loading
        edge_traces = []
        edge_annotations = []
        if hasattr(net_snapshot, "res_line") and not net_snapshot.res_line.empty:
            res_line = net_snapshot.res_line
        else:
            res_line = pd.DataFrame()
        if res_line is not None and not res_line.empty:
            for idx, line in net_snapshot.line.iterrows():
                u = line["from_bus"]
                v = line["to_bus"]
                if u not in pos or v not in pos:
                    continue
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                loading = res_line.at[idx, "loading_percent"] if idx in res_line.index else 0
                # Use a distinct color if overloaded
                if loading > 100:
                    hex_color = "#8B0000"  # Dark red
                else:
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
                        font=dict(color="white", size=12)
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
                    hoverinfo="none"
                )
                edge_traces.append(edge_trace)
        fig = go.Figure(
            data=edge_traces + [node_trace],
            layout=go.Layout(
                title="Network Graph",
                showlegend=False,
                hovermode="closest",
                annotations=edge_annotations,
                xaxis=dict(scaleanchor="y", scaleratio=1, showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                width=750,
                height=750,
                margin=dict(l=5, r=25, t=40, b=20)
            )
        )
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
        fig.add_trace(dummy_trace)
        return fig

    # Display the two network graphs side by side
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Simulation A")
        fig_a = create_network_figure(snapshot_a)
        if fig_a:
            st.plotly_chart(fig_a, use_container_width=True, key="net_fig_a")
    with col2:
        st.subheader("Simulation B")
        fig_b = create_network_figure(snapshot_b)
        if fig_b:
            st.plotly_chart(fig_b, use_container_width=True, key="net_fig_b")
else:
    st.info("Please load simulation histories in the sidebar.")
