import pandapower as pp
import folium
import streamlit as st

def load_network():
    """
    Loads the prepared pandapower network.
    Replace this with your own network file if needed.
    """
    net = pp.from_pickle('Main_app/original_network.p')
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
