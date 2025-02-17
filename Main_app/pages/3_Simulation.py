import streamlit as st
import os
import pandapower as pp
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from plotly.subplots import make_subplots
from pandapower.topology import create_nxgraph
from utils import load_network
import copy  # Pour réaliser une copie profonde du réseau
import re
import pickle

# Load the reference (original) network from file
original_network = pp.from_pickle("Main_app/original_network.p")

# Initialize session state keys
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

# File uploader in the sidebar
uploaded_file = st.sidebar.file_uploader("Choose a network pickle file", type=["p", "pkl"])

if uploaded_file is not None:
    st.sidebar.write(f"Selected file: {uploaded_file.name}")
    if st.sidebar.button("Load Network", key="load_net"):
        try:
            # Charger directement le réseau à partir de l'objet uploaded_file
            net = pp.from_pickle(uploaded_file)
            st.session_state["net"] = net  # Stocker le réseau dans session_state
            st.sidebar.success("Network loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading network: {e}")
else:
    st.sidebar.info("Please upload a network file to load.")

if "net" in st.session_state:
    net = st.session_state["net"]




# Determine the base name from the uploaded file name or a user input (if you want to choose it)
if "net" in st.session_state:
    # For example, use the base name of the uploaded file (without extension)
    uploaded_name = os.path.splitext(uploaded_file.name)[0]
    txt_filepath = os.path.join(r"Main_app\modified_network", f"{uploaded_name}.txt")
    if os.path.exists(txt_filepath):
        st.sidebar.info(f"Loading load type info from {txt_filepath}...")
        with open(txt_filepath, "r") as f:
            for line in f:
                # Expect lines of the form: "Load 44: Fractional" or "Load 44: Fixed"
                m = re.match(r"Load\s+(\d+):\s*(\w+)", line)
                if m:
                    load_idx = int(m.group(1))
                    load_type = m.group(2).strip().lower()
                    # Update the network's load table if the load exists
                    if load_idx in net.load.index:
                        net.load.at[load_idx, "fraction"] = True if load_type == "fractional" else False
        st.sidebar.success("Load type info loaded successfully!")
    else:
        st.sidebar.warning(f"No load info file found at {txt_filepath}. All loads remain as set.")


forecast_csv_path = "Main_app/saved_forecast/load_forecast.csv"
try:
    forecast_csv = pd.read_csv(forecast_csv_path)
    forecast_csv["ds"] = pd.to_datetime(forecast_csv["ds"])
except Exception as e:
    st.error(f"Error loading forecast CSV file: {e}")
    st.stop()


# Run initial OPF if simulation history is empty
if "net" in st.session_state and len(st.session_state["sim_history"]) == 0:
    try:
        pp.runopp(net)
        # Create a deep copy (snapshot) of the network after OPF
        net_snapshot = copy.deepcopy(net)
        # Collect simulation metrics
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

        # Use the forecast CSV's first timestamp if available, else current time
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

# Sauvegarder les proportions de charge initiales si non déjà stockées
if "original_loads" not in st.session_state:
    st.session_state["original_loads"] = net.load["p_mw"].to_dict()
original_loads = st.session_state["original_loads"]


# --- Simulation Controls in Sidebar ---
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
        # Réactiver tous les générateurs avant chaque timestep
        net.gen["in_service"] = True

        # Obtenir la charge forecastée (en MW) pour le timestep t
        load_forecast = forecast_csv.iloc[t]["yhat"]
        total_orig = sum(original_loads.values())
        for idx, orig_val in original_loads.items():
            # Update only if the load is fractional
            if "fraction" in net.load.columns and net.load.at[idx, "fraction"]:
                new_load = load_forecast * (orig_val / total_orig)
                net.load.at[idx, "p_mw"] = new_load


        # Exécuter l'OPF ; si ça échoue, on logge l'erreur et on saute ce timestep.
        try:
            pp.runopp(net)  # ou utiliser pp.runpp(net) selon vos préférences
        except Exception as e:
            st.session_state["error_log"].append(f"OPF did not converge at timestep {t}; skipping this timestep.")
            continue

        # Enregistrer une snapshot du réseau (copie profonde)
        net_snapshot = copy.deepcopy(net)

        # Collecter les métriques de simulation.
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
            "net_snapshot": net_snapshot,  # Enregistrer la snapshot du réseau
        }
        st.session_state["sim_history"].append(sim_state)

if advance_btn:
    new_step = min(current_step + step_size, total_steps)
    run_simulation_steps(current_step, new_step)
    st.sidebar.success(f"Simulation advanced to step {len(st.session_state['sim_history'])}")

if run_full_btn:
    run_simulation_steps(current_step, total_steps)
    st.sidebar.success("Full simulation run completed.")

# --- Display Simulation History ---
st.subheader("Simulation History")
if st.session_state["sim_history"]:
    max_step = len(st.session_state["sim_history"])
    # Slider dans la sidebar (si un seul timestep, on fixe à 1)
    if max_step > 1:
        selected_step = st.sidebar.slider("Select Simulation Timestep", 1, max_step, max_step, key="sim_slider")
    else:
        selected_step = 1

    history = st.session_state["sim_history"][:selected_step]
    times = [entry["time"] for entry in history]
    demands = [entry["demand"] for entry in history]
    generations = [entry["generation"] for entry in history]

    # Affichage en deux colonnes : gauche pour le graphique à barres groupées, droite pour le camembert.

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
    
    
import pickle
# In the sidebar, let the user enter a base filename (prefix)
sim_prefix = st.sidebar.text_input("Enter simulation file name", value="simulation")

if st.sidebar.button("Save Entire Simulation", key="save_full_sim"):
    # Create a folder for saving simulations if it doesn't exist
    save_folder = "Main_app/simulations/" + sim_prefix
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    sim_history = st.session_state.get("sim_history", [])
    if not sim_history:
        st.sidebar.error("No simulation data available to save.")
    else:
        # Save each timestep's network snapshot with a filename based on the prefix and timestep number.
        for i, step in enumerate(sim_history, start=1):
            # We assume that each simulation state contains the network snapshot in "net_snapshot"
            net_snapshot = step.get("net_snapshot", None)
            if net_snapshot is not None:
                filename = os.path.join(save_folder, f"{sim_prefix}_timestep_{i}.pkl")
                try:
                    with open(filename, "wb") as f:
                        pickle.dump(net_snapshot, f)
                    #st.sidebar.write(f"Saved: {filename}")
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
# Network Visualization Section
###############################################################################
st.header("Network Visualization with Heatmaps")

# Utiliser la snapshot du timestep sélectionné pour la visualisation du réseau.
if st.session_state["sim_history"]:
    sim_snapshot = st.session_state["sim_history"][selected_step - 1]
    # On récupère la snapshot du réseau enregistrée
    net_snapshot = sim_snapshot.get("net_snapshot", None)
else:
    net_snapshot = None

# Si une snapshot est disponible, on l'utilise pour générer le graphe.
if net_snapshot is not None:
    G = create_nxgraph(net_snapshot, respect_switches=True)
else:
    G = create_nxgraph(net, respect_switches=True)

# Utiliser les coordonnées géographiques si disponibles; sinon, utiliser un layout spring.
if hasattr(net, "bus_geodata") and not net.bus_geodata.empty:
    pos = {bus: (net.bus_geodata.at[bus, "x"], net.bus_geodata.at[bus, "y"])
           for bus in net.bus_geodata.index}
else:
    pos = nx.spring_layout(G, seed=42)

# Utiliser la snapshot pour les tensions si disponible.
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

# Calcul de la surcharge pour chaque bus.
for bus in G.nodes():
    vm = G.nodes[bus].get("vm_pu", 1.0)
    if vm < 0.98:
        overload = (0.98 - vm) * 100
    elif vm > 1.02:
        overload = (vm - 1.02) * 100
    else:
        overload = 0
    G.nodes[bus]["overload"] = overload

# Préparation de la trace des nœuds (afficher le texte uniquement si le bus est overload).
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
        colorbar=dict(title="Bus Overload (%)", x=0.0),
        cmin=0,
        cmax=max(node_color) if node_color else 1,
        line=dict(width=2)
    )
)

# Préparation des traces d'arêtes pour le loading des lignes.
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

# Ajout d'une trace fictive pour la colorbar des lignes, positionnée à droite.
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

current_step = len(st.session_state["sim_history"])
total_steps = len(forecast_csv)


if "sim_history" in st.session_state and st.session_state["sim_history"]:
    # Use the first simulation step instead of the last one
    current_state = st.session_state["sim_history"][0]
    total_gen = current_state["generation"]
    if total_gen > 0:
        gen_dist = current_state["gen_distribution"]
        gen_names = current_state["gen_names"]
    else:
        gen_dist = {}
        gen_names = {}
    if gen_dist:
        # Replace generator IDs with their names
        labels = [gen_names[gen_id] for gen_id in gen_dist.keys()]
        values = list(gen_dist.values())
        fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
        fig_pie.update_layout(title="Generation Distribution by Generator")
        st.plotly_chart(fig_pie, use_container_width=True, key="pie_chart")
    else:
        st.info("No generation data available for the first timestep.")
else:
    st.info("No simulation data available yet.")



# Display error messages at the bottom.
if st.session_state["error_log"]:
    st.subheader("Simulation Messages")
    for msg in st.session_state["error_log"]:
        st.error(msg)
