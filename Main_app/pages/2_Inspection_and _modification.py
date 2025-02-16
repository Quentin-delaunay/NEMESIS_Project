import streamlit as st
from streamlit_folium import st_folium
from utils import load_network, folium_plot
import pandapower as pp
import os
import pickle
import re


TYPE_SYNONYMS = {
    "nuclear": "nuclear",
    "coal": "coal",
    "natural gas": "natural gas",
    "pumped storage": "pumped storage",
    "petroleum": "petroleum",
    "wind": "wind",
    "solar": "solar"
}

# Page configuration
st.set_page_config(page_title="Network Inspection & Modification", layout="wide")
st.title("Inspection and modification of a given network.")

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
    


# Sidebar: Choose the type of element to modify
selected_type = st.sidebar.radio(
    "Select type to modify", 
    ("Loads", "Lines", "Buses", "Generators")
)

# If the 'fraction' column does not exist in net.load, add it and mark existing loads as fractional by default.
if "fraction" not in net.load.columns:
    net.load["fraction"] = True

# Create two columns: left for network visualization, right for management interface.
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Loaded Electric Network")
    folium_map = folium_plot(net)
    if folium_map:
        st_folium(folium_map, width=700, height=700)

with col_right:
    if selected_type == "Loads":
        st.subheader("Loads Management")
        load_option = st.selectbox("Choose an action", 
                                   ("Modify existing loads", "Add a new load", "Delete a load"))
        if load_option == "Modify existing loads":
            st.subheader("Modify Existing Load")
            load_ids = net.load.index.tolist()
            if load_ids:
                selected_load = st.selectbox("Select a load to modify", load_ids)
                current_power = net.load.at[selected_load, "p_mw"]
                current_type = "Fractional" if net.load.at[selected_load, "fraction"] else "Fixed"
                st.write(f"Current active power: {current_power} MW")
                st.write(f"Current load type: {current_type}")
                new_power = st.number_input("New active power (MW)", min_value=0.0, value=current_power)
                new_type = st.selectbox("Select load type", ("Fractional", "Fixed"), index=0 if current_type=="Fractional" else 1)
                if st.button("Update Load"):
                    net.load.at[selected_load, "p_mw"] = new_power
                    net.load.at[selected_load, "fraction"] = True if new_type=="Fractional" else False
                    st.success(f"Load {selected_load} updated to {new_power} MW as a {new_type} load.")
            else:
                st.info("No loads available in the network.")
        
        elif load_option == "Add a new load":
            st.subheader("Add a New Load")
            with st.form("add_load_form"):
                departure_bus = st.number_input("Departure Bus", min_value=0, step=1)
                p_mw = st.number_input("Active Power (MW)", min_value=0.0, value=1.0)
                name = st.text_input("Load Name", value="New Load")
                load_type = st.selectbox("Load Type", ("Fractional", "Fixed"))
                controllable = st.checkbox("Controllable", value=False)
                submitted = st.form_submit_button("Add Load")
            if submitted:
                if load_type == "Fixed":
                    new_load_id = pp.create_load(
                        net,
                        bus=departure_bus,
                        p_mw=p_mw,  # Already in MW
                        max_p_mw=p_mw,
                        min_p_mw=p_mw,
                        name=name,
                        max_q_mvar=p_mw / 0.9,  # Example formula
                        min_q_mvar=0,
                        controllable=controllable
                    )
                    net.load.at[new_load_id, "fraction"] = False
                    st.success(f"Fixed load '{name}' added on bus {departure_bus} with {p_mw} MW.")
                else:
                    new_load_id = pp.create_load(
                        net,
                        bus=departure_bus,
                        p_mw=p_mw,
                        max_p_mw=p_mw,
                        min_p_mw=0,  # Fractional load can be modulated
                        name=name,
                        max_q_mvar=p_mw / 0.9,
                        min_q_mvar=0,
                        controllable=controllable
                    )
                    net.load.at[new_load_id, "fraction"] = True
                    st.success(f"Fractional load '{name}' added on bus {departure_bus} with {p_mw} MW.")
        
        elif load_option == "Delete a load":
            st.subheader("Delete a Load")
            load_ids = net.load.index.tolist()
            if load_ids:
                selected_load = st.selectbox("Select a load to delete", load_ids)
                if st.button("Delete Load"):
                    try:
                        pp.drop_elements_simple(net, "load", selected_load)
                        st.success(f"Load {selected_load} has been deleted.")
                    except Exception as e:
                        st.error(f"Error deleting load {selected_load}: {e}")
            else:
                st.info("No loads available in the network.")
    
    elif selected_type == "Lines":
        st.subheader("Lines Management")
        line_option = st.selectbox("Choose an action", 
                                ("Modify existing lines", "Add a new line", "Delete a line"))
        if line_option == "Add a new line":
            st.subheader("Add a New Line")
            # Récupérer la liste des bus disponibles
            bus_ids = net.bus.index.tolist()
            bus1_id = st.selectbox("Select From Bus", bus_ids, key="from_bus")
            bus2_id = st.selectbox("Select To Bus", bus_ids, key="to_bus")
            
            # Si des coordonnées géographiques sont disponibles, on les utilisera
            if hasattr(net, "bus_geodata") and not net.bus_geodata.empty:
                # On peut extraire les coordonnées à partir de net.bus_geodata pour les bus sélectionnés
                x1, y1 = net.bus_geodata.loc[bus1_id, ["x", "y"]]
                x2, y2 = net.bus_geodata.loc[bus2_id, ["x", "y"]]
                st.write(f"Coordinates from bus {bus1_id}: ({x1}, {y1})")
                st.write(f"Coordinates from bus {bus2_id}: ({x2}, {y2})")
            else:
                st.markdown("#### Enter bus coordinates manually (if not available in network data)")
                x1 = st.number_input("From Bus X (longitude)", value=0.0, key="x_from")
                y1 = st.number_input("From Bus Y (latitude)", value=0.0, key="y_from")
                x2 = st.number_input("To Bus X (longitude)", value=0.0, key="x_to")
                y2 = st.number_input("To Bus Y (latitude)", value=0.0, key="y_to")
            
            # Haversine function to compute distance in km.
            def haversine(lat1, lon1, lat2, lon2):
                import math
                R = 6371  # Earth radius in kilometers
                dlat = math.radians(lat2 - lat1)
                dlon = math.radians(lon2 - lon1)
                a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
                return R * c

            # Utiliser les coordonnées disponibles pour calculer la longueur de la ligne
            length_km = haversine(y1, x1, y2, x2)
            st.write(f"Calculated line length: {length_km:.2f} km")
            
            line_id = st.text_input("Line Name/ID", value="New Line")
            
            if st.button("Add Line"):
                try:
                    pp.create_line_from_parameters(
                        net,
                        from_bus=bus1_id,
                        to_bus=bus2_id,
                        length_km=length_km,
                        r_ohm_per_km=0.01,  # Example parameter
                        x_ohm_per_km=0.25,  # Example parameter
                        c_nf_per_km=12,     # Example parameter
                        max_i_ka=2.0,       # Example parameter
                        name=line_id
                    )
                    st.success(f"Line '{line_id}' added connecting bus {bus1_id} to bus {bus2_id}.")
                except Exception as e:
                    st.error(f"Error adding line: {e}")
        
        
        
        elif line_option == "Modify existing lines":
            st.write("Interface to modify existing lines.")
            line_ids = net.line.index.tolist()
            if line_ids:
                selected_line = st.selectbox("Select a line to modify", line_ids, key="modify_line")
                # Retrieve current properties; use defaults if not present.
                current_max_loading = net.line.at[selected_line, "max_loading_percent"] if "max_loading_percent" in net.line.columns else 100.0
                current_parallel = net.line.at[selected_line, "parallel"] if "parallel" in net.line.columns else 1
                
                st.write(f"Current max loading percent: {current_max_loading}")
                st.write(f"Current number of parallel lines: {current_parallel}")
                
                new_max_loading = st.number_input("New Maximum Loading Percent", value=current_max_loading)
                new_parallel = st.number_input("Number of Parallel Lines", min_value=1, value=current_parallel, step=1)
                
                if st.button("Update Line", key="update_line"):
                    try:
                        net.line.at[selected_line, "max_loading_percent"] = new_max_loading
                        net.line.at[selected_line, "parallel"] = new_parallel
                        st.success(f"Line {selected_line} updated: max loading percent set to {new_max_loading}, parallel lines set to {new_parallel}.")
                    except Exception as e:
                        st.error(f"Error updating line {selected_line}: {e}")
            else:
                st.info("No lines available in the network.")
                
        
        elif line_option == "Delete a line":
            st.subheader("Delete a Line")
            line_ids = net.line.index.tolist()
            if line_ids:
                selected_line = st.selectbox("Select a line to delete", line_ids, key="delete_line")
                if st.button("Delete Line"):
                    try:
                        pp.drop_elements_simple(net, "line", selected_line)
                        st.success(f"Line {selected_line} has been deleted.")
                    except Exception as e:
                        st.error(f"Error deleting line {selected_line}: {e}")
            else:
                st.info("No lines available in the network.")
                
                

    elif selected_type == "Buses":
        st.subheader("Buses Management")
        bus_option = st.selectbox("Choose an action", 
                                ("Modify existing buses", "Add a new bus", "Delete a bus"))
        if bus_option == "Modify existing buses":
            st.subheader("Modify Existing Bus")
            bus_ids = net.bus.index.tolist()
            if bus_ids:
                selected_bus = st.selectbox("Select a bus to modify", bus_ids, key="modify_bus")
                current_vn = net.bus.at[selected_bus, "vn_kv"]
                # If geodata is available, try to fetch the coordinates.
                if hasattr(net, "bus_geodata") and not net.bus_geodata.empty and selected_bus in net.bus_geodata.index:
                    current_x = net.bus_geodata.at[selected_bus, "x"]
                    current_y = net.bus_geodata.at[selected_bus, "y"]
                else:
                    current_x, current_y = 0.0, 0.0

                new_vn = st.number_input("New nominal voltage (kV)", min_value=0.0, value=current_vn)
                new_x = st.number_input("New bus X coordinate", value=current_x)
                new_y = st.number_input("New bus Y coordinate", value=current_y)
                st.write(f"Current bus type: {net.bus.at[selected_bus, 'type'] if 'type' in net.bus.columns else 'N/A'}")
                if st.button("Update Bus", key="update_bus"):
                    try:
                        net.bus.at[selected_bus, "vn_kv"] = new_vn
                        # Mise à jour des coordonnées géographiques si disponibles.
                        if hasattr(net, "bus_geodata") and not net.bus_geodata.empty:
                            if selected_bus in net.bus_geodata.index:
                                net.bus_geodata.at[selected_bus, "x"] = new_x
                                net.bus_geodata.at[selected_bus, "y"] = new_y
                            else:
                                st.warning("No geodata available for the selected bus.")
                        st.success(f"Bus {selected_bus} updated to {new_vn} kV and coordinates ({new_x}, {new_y}).")
                    except Exception as e:
                        st.error(f"Error updating bus {selected_bus}: {e}")
            else:
                st.info("No buses available in the network.")
        
        elif bus_option == "Add a new bus":
            st.subheader("Add a New Bus")
            with st.form("add_bus_form"):
                bus_id = st.text_input("Bus ID", value="NewBus")
                vn_kv = st.number_input("Nominal Voltage (kV)", min_value=0.0, value=500.0)
                x_coord = st.number_input("Bus X coordinate", value=0.0)
                y_coord = st.number_input("Bus Y coordinate", value=0.0)
                submitted = st.form_submit_button("Add Bus")
            if submitted:
                try:
                    new_bus_index = pp.create_bus(
                        net, 
                        name=bus_id, 
                        vn_kv=vn_kv, 
                        geodata=(x_coord, y_coord), 
                        type="b"
                    )
                    st.success(f"Bus '{bus_id}' added with index {new_bus_index}.")
                except Exception as e:
                    st.error(f"Error adding bus: {e}")
        
        elif bus_option == "Delete a bus":
            st.subheader("Delete a Bus")
            bus_ids = net.bus.index.tolist()
            if bus_ids:
                selected_bus = st.selectbox("Select a bus to delete", bus_ids, key="delete_bus")
                if st.button("Delete Bus", key="del_bus"):
                    try:
                        pp.drop_elements_simple(net, "bus", selected_bus)
                        st.success(f"Bus {selected_bus} has been deleted.")
                    except Exception as e:
                        st.error(f"Error deleting bus {selected_bus}: {e}")
            else:
                st.info("No buses available in the network.")

    
   
    elif selected_type == "Generators":
        
        MIN_LOAD_FRACTIONS = {
        "nuclear": 0.0,
        "coal": 0.0,
        "natural gas": 0.0,
        "pumped storage": 0.0,
        "petroleum": 0.0,
        "wind": 0.0,
        "solar": 0.0
        }
        MARGINAL_COSTS = { 
            "nuclear": 77.16,
            "coal": 89.33,
            "natural gas": 42.72,
            "pumped storage": 57.12,
            "petroleum": 128.82,
            "wind": 31,
            "solar": 23
        }
        
        st.subheader("Generators Management")
        gen_option = st.selectbox("Choose an action", 
                                ("Modify existing generators", "Add a new generator", "Delete a generator", "Modify global parameters by type"))
        
        if gen_option == "Modify existing generators":
            st.subheader("Modify Existing Generator")
            gen_ids = net.gen.index.tolist()
            if gen_ids:
                # Format the options to display the generator name along with its ID
                def format_gen(gen_id):
                    return f"{gen_id}: {net.gen.at[gen_id, 'name']}"
                selected_gen = st.selectbox("Select a generator to modify", gen_ids, format_func=format_gen, key="modify_gen")
                current_p = net.gen.at[selected_gen, "p_mw"]
                current_vm = net.gen.at[selected_gen, "vm_pu"]
                current_name = net.gen.at[selected_gen, "name"]
                st.write(f"Current active power: {current_p} MW")
                st.write(f"Current voltage setpoint: {current_vm} p.u.")
                st.write(f"Current generator name: {current_name}")
                new_p = st.number_input("New active power (MW)", min_value=0.0, value=current_p)
                new_vm = st.number_input("New voltage setpoint (p.u.)", min_value=0.0, value=current_vm)
                new_name = st.text_input("New Generator Name (include type in parentheses)", value=current_name)
                controllable = st.checkbox("Controllable", value=True)
                if st.button("Update Generator", key="update_gen"):
                    try:
                        net.gen.at[selected_gen, "p_mw"] = new_p
                        net.gen.at[selected_gen, "vm_pu"] = new_vm
                        net.gen.at[selected_gen, "name"] = new_name
                        net.gen.at[selected_gen, "controllable"] = controllable
                        st.success(f"Generator {selected_gen} updated: p_mw={new_p} MW, vm_pu={new_vm}, name='{new_name}'.")
                    except Exception as e:
                        st.error(f"Error updating generator {selected_gen}: {e}")
            else:
                st.info("No generators available in the network.")
        
        elif gen_option == "Add a new generator":
            st.subheader("Add a New Generator")
            with st.form("add_gen_form"):
                bus_index = st.number_input("Bus index", min_value=0, step=1)
                p_mw = st.number_input("Active Power (MW)", min_value=0.0, value=1.0)
                vm_pu = st.number_input("Voltage setpoint (p.u.)", min_value=0.0, value=1.0)
                max_p_mw = st.number_input("Maximum Active Power (MW)", min_value=0.0, value=p_mw)
                gen_name = st.text_input("Generator Name", value="New Generator")
                gen_type = st.selectbox("Generator Type", ("natural gas", "nuclear", "coal", "wind", "solar"))
                controllable = st.checkbox("Controllable", value=True)
                # Determine slack: if no slack is already assigned and type is "natural gas"
                slack_assigned = net.gen["slack"].any() if "slack" in net.gen.columns else False
                slack = (not slack_assigned) and (gen_type == "natural gas")
                submitted = st.form_submit_button("Add Generator")
            if submitted:
                try:
                    pp.create_gen(
                        net,
                        bus=bus_index,
                        p_mw=p_mw,
                        vm_pu=vm_pu,
                        min_p_mw=0,
                        max_p_mw=max_p_mw,
                        min_q_mvar=-p_mw / 0.85,
                        max_q_mvar=p_mw / 0.85,
                        name=f"{gen_name} ({gen_type})",
                        slack=slack,
                        controllable=controllable
                    )
                    st.success(f"Generator '{gen_name}' added on bus {bus_index} with active power {p_mw} MW.")
                except Exception as e:
                    st.error(f"Error adding generator: {e}")
        
        elif gen_option == "Delete a generator":
            st.subheader("Delete a Generator")
            gen_ids = net.gen.index.tolist()
            if gen_ids:
                def format_gen(gen_id):
                    return f"{gen_id}: {net.gen.at[gen_id, 'name']}"
                selected_gen = st.selectbox("Select a generator to delete", gen_ids, format_func=format_gen, key="delete_gen")
                if st.button("Delete Generator", key="del_gen"):
                    try:
                        pp.drop_elements_simple(net, "gen", selected_gen)
                        st.success(f"Generator {selected_gen} has been deleted.")
                    except Exception as e:
                        st.error(f"Error deleting generator {selected_gen}: {e}")
            else:
                st.info("No generators available in the network.")
                
        elif gen_option == "Modify global parameters by type":
            st.subheader("Modify Global Parameters by Generator Type")
            available_types = list(MARGINAL_COSTS.keys())
            selected_gen_type = st.selectbox("Select generator type", available_types)
            new_cost_global = st.number_input("New Marginal Cost (USD/MW)", min_value=0.0, value=MARGINAL_COSTS[selected_gen_type])
            new_util_pct = st.number_input("New minimum utilization (%)", min_value=0.0, max_value=100.0, value=MIN_LOAD_FRACTIONS[selected_gen_type]*100)
            if st.button("Update Global Parameters"):
                updated_count = 0
                for idx in net.gen.index:
                    gen_name = net.gen.at[idx, "name"]
                    match = re.search(r"\(([^)]+)\)", gen_name)
                    if match:
                        raw_type = match.group(1).strip().lower()
                        current_type = TYPE_SYNONYMS.get(raw_type, raw_type)
                    else:
                        current_type = "other"
                    if current_type == selected_gen_type:
                        # Update minimum production: new_min = (new_util_pct/100)*max_p_mw
                        new_min_val = (new_util_pct/100.0) * net.gen.at[idx, "max_p_mw"]
                        net.gen.at[idx, "min_p_mw"] = new_min_val
                        # Update marginal cost (create a new poly cost)
                        pp.create_poly_cost(
                            net,
                            element=idx,
                            et="gen",
                            cp1_eur_per_mw=new_cost_global,
                            cp0_eur=0.0,
                            cq1_eur_per_mvar=0.0,
                            cq0_eur=0.0,
                            cp2_eur_per_mw2=0.0,
                            cq2_eur_per_mvar2=0.0,
                            check=False
                        )
                        updated_count += 1
                st.success(f"Updated {updated_count} generators of type {selected_gen_type}.")





st.sidebar.subheader("Save Modified Network")
network_name = st.sidebar.text_input("Enter network name", value="modified_network")
if st.sidebar.button("Save Network", key="save_network"):
    # Créer le dossier s'il n'existe pas
    save_folder = "Main_app/modified_network/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    filename = os.path.join(save_folder, f"{network_name}.pkl")
    try:
        with open(filename, "wb") as f:
            pickle.dump(net, f)
        st.sidebar.success(f"Network saved as {filename}")
    except Exception as e:
        st.sidebar.error(f"Error saving network: {e}")