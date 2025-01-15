import pandapower as pp
import pandapower.plotting as plot
import pandas as pd
import matplotlib.pyplot as plt
import pandapower.plotting.plotly as pp_plotly
from pandapower.plotting.plotly import simple_plotly
import plotly.io as pio
import math


transformer_types = {
    (500, 230): {
        "sn_mva": 1000,  # Transformateurs THT : typiquement 1000 MVA
        "vk_percent": 12.0,
        "vkr_percent": 0.3,
        "pfe_kw": 50,
        "i0_percent": 0.05,
    },
    (230, 115): {
        "sn_mva": 400,  # Sous-transmission : typiquement 400 MVA
        "vk_percent": 10.0,
        "vkr_percent": 0.4,
        "pfe_kw": 50,
        "i0_percent": 0.07,
    },
    (115, 20): {
        "sn_mva": 100,  # Distribution régionale : typiquement 100 MVA
        "vk_percent": 8.0,
        "vkr_percent": 0.5,
        "pfe_kw": 50,
        "i0_percent": 0.1,
    },
    (33, 20): {
        "sn_mva": 20,  # Distribution locale : typiquement 20 MVA
        "vk_percent": 6.0,
        "vkr_percent": 0.7,
        "pfe_kw": 20,
        "i0_percent": 0.2,
    },
    (20, 0.4): {
        "sn_mva": 2,  # Distribution finale : typiquement 2 MVA
        "vk_percent": 4.0,
        "vkr_percent": 1.0,
        "pfe_kw": 10,
        "i0_percent": 0.3,
    },
}


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r

# Configure Plotly to display in the browser
pio.renderers.default = 'browser'

# Load filtered data
buses_df = pd.read_csv(r"output_pandapower\buses.csv")
loads_df = pd.read_csv(r"output_pandapower\loads.csv")
generators_df = pd.read_csv(r"output_pandapower\generators.csv")

# Create a new Pandapower network
net = pp.create_empty_network()

# Add buses to the network
bus_index_map = {}
# Adjust bus positions to avoid overlapping
offset = 0.0001  # Small offset in degrees (~10 meters)
unique_positions = {}
pos_bus ={}

for i, bus in buses_df.iterrows():
    position = (bus['x'], bus['y'])
    # Check if position is already used
    if position in unique_positions:
        # Apply a unique offset
        bus['x'] += offset
        bus['y'] += offset
        position = (bus['x'], bus['y'])
    unique_positions[position] = bus['bus_id']
    pos_bus[bus['bus_id']] = position
    
    # Create bus
    bus_index = pp.create_bus(
        net, 
        name=bus['bus_id'], 
        vn_kv=bus['vn_kv'], 
        type='b', 
        geodata=(bus['x'], bus['y'])
    )
    bus_index_map[bus['bus_id']] = bus_index

# Load transformers data
try:
    transformers_df = pd.read_csv(r"output_pandapower\transformers.csv")

    # Check if the file is empty
    if transformers_df.empty:
        print("The transformers.csv file is empty. Skipping transformer addition.")
    else:
        # Add transformers to the network
        for _, transformer in transformers_df.iterrows():
            try:
                # Ensure bus IDs exist in the network
                if transformer['from_bus'] not in net.bus.name.values:
                    raise ValueError(f"from_bus {transformer['from_bus']} does not exist in the network.")
                if transformer['to_bus'] not in net.bus.name.values:
                    raise ValueError(f"to_bus {transformer['to_bus']} does not exist in the network.")


                max_kV = max(net.bus.loc[net.bus.name == transformer['from_bus'],
                            'vn_kv'].values[0],net.bus.loc[net.bus.name == transformer['to_bus'], 'vn_kv'].values[0])

                min_kV = min(net.bus.loc[net.bus.name == transformer['from_bus'],
                            'vn_kv'].values[0],net.bus.loc[net.bus.name == transformer['to_bus'], 'vn_kv'].values[0])
                
                # Retrieve bus voltages
                vn_hv_kv = max_kV
                vn_lv_kv = min_kV

                # Create a key for the transformer type dictionary
                voltage_jump = (vn_hv_kv, vn_lv_kv)

                # Check if the transformer type exists in the dictionary
                if voltage_jump in transformer_types:
                    params = transformer_types[voltage_jump]

                    # Add the transformer using parameters from the dictionary
                    pp.create_transformer_from_parameters(
                        net,
                        hv_bus=net.bus.loc[net.bus.name == transformer['from_bus']].index[0],
                        lv_bus=net.bus.loc[net.bus.name == transformer['to_bus']].index[0],
                        sn_mva=params['sn_mva'],
                        vn_hv_kv=net.bus.loc[net.bus.name == transformer['from_bus'],
                            'vn_kv'].values[0],
                        vn_lv_kv=net.bus.loc[net.bus.name == transformer['to_bus'], 'vn_kv'].values[0],
                        vkr_percent=params['vkr_percent'],
                        vk_percent=params['vk_percent'],
                        pfe_kw=params['pfe_kw'],
                        i0_percent=params['i0_percent'],
                        name=f"Transformer from {transformer['from_bus']} to {transformer['to_bus']}"
                    )
                else:
                    print(f"No transformer type defined for voltage jump: {voltage_jump}")
            except Exception as e:
                print(f"Error adding transformer with data: {transformer.to_dict()}")
                raise e

except FileNotFoundError:
    print("The transformers.csv file does not exist. Skipping transformer addition.")
except pd.errors.EmptyDataError:
    print("The transformers.csv file is empty. Skipping transformer addition.")




# Add lines based on shared line IDs
added_lines = set()
for _, bus1 in buses_df.iterrows():
    for _, bus2 in buses_df.iterrows():
        if bus1['bus_id'] != bus2['bus_id']:
            common_lines = set(eval(bus1['lines'])) & set(eval(bus2['lines']))
            for line_id in common_lines:
                if (bus1['bus_id'], bus2['bus_id'], line_id) not in added_lines and (bus2['bus_id'], bus1['bus_id'], line_id) not in added_lines:
                    try:
                        pp.create_line_from_parameters(
                            net,
                            from_bus=bus_index_map[bus1['bus_id']],
                            to_bus=bus_index_map[bus2['bus_id']],
                            length_km=haversine(bus1['x'], bus1['y'], bus2['x'], bus2['y']),  # Default length; adjust if specific lengths are available
                            r_ohm_per_km=0.01,
                            x_ohm_per_km=0.25,
                            c_nf_per_km=12,
                            max_i_ka=2.0,
                            name=f"Line {line_id}"
                        )
                        added_lines.add((bus1['bus_id'], bus2['bus_id'], line_id))
                    except Exception as e:
                        print(f"Error creating line between Bus {bus1['bus_id']} and Bus {bus2['bus_id']} for Line {line_id}")
                        raise e



for _, load in loads_df.iterrows():
    try:
        # Original bus (depart bus)
        depart_bus = bus_index_map[load['bus']]
        
        # pos = pos_bus[load['bus']]
        # # Create a new bus for the load with an offset position
        # load_bus = pp.create_bus(
        #     net,
        #     vn_kv=20,  # Voltage level of distribution (adjust as needed)
        #     type='b',  # Typically, loads are connected to 'b' type buses
        #     geodata=(pos[0] + 0.0001, pos[1] + 0.0001),  # Offset the position slightly
        #     name=f"{load['name']} Bus"
        # )
        
        # # Create a transformer connecting the depart bus to the load bus
        # pp.create_transformer_from_parameters(
        #     net,
        #     hv_bus=depart_bus,
        #     lv_bus=load_bus,
        #     sn_mva=load['p_kw'] * 0.001 /0.9,  # Example transformer size, adjust as needed
        #     vn_hv_kv=net.bus.loc[depart_bus, 'vn_kv'],  # Voltage of the depart bus
        #     vn_lv_kv=20,  # Voltage level of the new load bus
        #     vkr_percent = 1,  # Example short-circuit voltage, adjust as needed
        #     vk_percent=10,  # Example voltage drop under rated current
        #     pfe_kw=50,  # Example iron losses
        #     i0_percent=0.1,  # Example no-load current
        #     name=f"Transformer {load['name']}"
        # )
        
        # Create the load on the new bus
        pp.create_load(
            net,
            bus=depart_bus,
            p_mw=load['p_kw'] * 0.001 *0.98,  # Convert kW to MW
            max_p_mw=load['p_kw'] * 0.001,
            miin_p_mw=0.1*load['p_kw'] * 0.001,  # Default to 0.0 if not provided
            name=load['name'],
            max_q_mvar = load['p_kw'] * 0.001 / 0.9,
            min_q_mvar = 0.1*load['p_kw'] * 0.001 / 0.9,
        )
        
    except Exception as e:
        print(f"Error creating load: {load}")
        raise e

# Create specific buses for generators and connect them to the network
slack_assigned = False

for _, gen in generators_df.iterrows():
    try:
        # # Create a bus specifically for the generator
        # gen_bus = pp.create_bus(
        #     net, 
        #     vn_kv= 20,  # Generator voltage `vn_kv` approximation
        #     name=f'Bus for {gen["generator_name"]} ({gen["type"]})',
        #     geodata=(gen['longitude'], gen['latitude'])
        # )
        # # Add the bus to the bus_index_map
        # bus_index_map[gen["generator_name"]] = gen_bus

        # Connect the generator to its specific bus
        pp.create_gen(
            net,
            bus=bus_index_map[gen['bus_id']],  # Use the original bus from the buses
            p_mw = 1,
            min_p_mw=0,  # Default to 0.0 if not provided
            max_p_mw=gen.get('max_p_mw', gen['p_mw']),  # Default to p_mw if not provided
            name=f'{gen["generator_name"]} ({gen["type"]})',
            slack=(not slack_assigned),  # Assign slack to the first generator
        )
        slack_assigned = True

        # pp.create_transformer_from_parameters(
        #     net,
        #     hv_bus=bus_index_map[gen['bus_id']],  # Bus haute tension
        #     lv_bus=gen_bus,  # Bus basse tension (20 kV)
        #     sn_mva=gen['p_mw']/0.9,  # Puissance nominale (MVA)
        #     vn_hv_kv=net.bus.loc[bus_index_map[gen["bus_id"]], 'vn_kv'],  # Tension nominale haute tension (kV)
        #     vn_lv_kv=20,  # Tension nominale basse tension (kV)
        #     vkr_percent=0.2,  # Pertes résistives (%)
        #     vk_percent=12.0,  # Impédance de court-circuit (%)
        #     pfe_kw=50,  # Pertes à vide (kW)
        #     i0_percent=0.1,  # Courant à vide (%)
        #     name="20/500 kV Transformer"
        # )
        
        
    except Exception as e:
        print(f"Error creating generator and connection for: {gen['generator_name']}")
        raise e


# Add an external grid if no slack generator is assigned
if not slack_assigned:
    pp.create_ext_grid(net, bus=0, vm_pu=1.0, va_degree=0)

# Vérifier les buses isolés
isolated_buses = net.bus.index.difference(pd.concat([net.line["from_bus"], net.line["to_bus"], net.trafo["hv_bus"], net.trafo["lv_bus"]]))
if not isolated_buses.empty:
    print("Buses isolés :", isolated_buses)

# Ajouter des géodonnées par défaut
if net.bus_geodata.empty:
    for idx in net.bus.index:
        net.bus_geodata.loc[idx] = {"x": idx * 0.1, "y": idx * 0.1}

def run_power_optimization_with_debug(net, max_iterations=5):
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")

        try:

            # Lancer le flux optimal
            print("Running Optimal Power Flow...")
            pp.runopp(net, delta=1e-6, debug=True)
            print("Optimal Power Flow simulation succeeded.")

            print("Optimization completed successfully.")
            break  # Si tout fonctionne, sortir de la boucle

        except (pp.LoadflowNotConverged, pp.optimal_powerflow.OPFNotConverged):
            print("Simulation did not converge. Running diagnostics for corrections...")
            diagnostic_report = pp.diagnostic(net, report_style='detailed')

            # 1. Corriger les buses et lignes déconnectées
            if "disconnected_elements" in diagnostic_report:
                for section in diagnostic_report["disconnected_elements"]:
                    buses_to_remove = section.get("buses", [])
                    lines_to_remove = section.get("lines", [])
                    if buses_to_remove:
                        print(f"Removing disconnected buses: {buses_to_remove}")
                        net.bus.drop(buses_to_remove, inplace=True)
                    if lines_to_remove:
                        print(f"Removing disconnected lines: {lines_to_remove}")
                        net.line.drop(lines_to_remove, inplace=True)

            # 2. Corriger les impédances proches de zéro
            if "impedance_values_close_to_zero" in diagnostic_report:
                for item in diagnostic_report["impedance_values_close_to_zero"]:
                    line_idx = item.get("line")
                    if line_idx:
                        print(f"Fixing impedance for line {line_idx}")
                        net.line.loc[line_idx, "r_ohm_per_km"] = net.line.loc[line_idx, "r_ohm_per_km"].apply(lambda x: max(x, 0.031))
                        net.line.loc[line_idx, "x_ohm_per_km"] = net.line.loc[line_idx, "x_ohm_per_km"].apply(lambda x: max(x, 0.35))

            print("Corrections applied. Retrying simulation...")

        except Exception as e:
            print(f"Unexpected error during simulation: {e}")
            break  # Sortir si une erreur imprévue survient

    else:
        print("Max iterations reached. The optimization could not be completed.")

# Exécuter la fonction avec le réseau Pandapower
run_power_optimization_with_debug(net)
fig = simple_plotly(net)
plt.show()



# Afficher les réglages des générateurs
print("Réglages des générateurs après OPF :")
print(net.res_gen[['p_mw', 'vm_pu']])



 # Sauvegarder le réseau dans un fichier pickle
pp.to_pickle(net, "my_network.p")

print("Réseau sauvegardé sous forme de fichier pickle : my_network.p")
