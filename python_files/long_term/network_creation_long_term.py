# ------------------------------------------------------
# Author: Quentin Delaunay
# Date: January 20, 2025
# Description: Script for creating a Pandapower network from filtered data.
# File: network_creation.py
# ------------------------------------------------------
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
        "sn_mva": 100,  # High voltage transformers: typically 100 MVA
        "vk_percent": 12.0,
        "vkr_percent": 0.5,
        "pfe_kw": 150,
        "i0_percent": 0.15,
    },
    (230, 115): {
        "sn_mva": 400,  # Subtransmission: typically 400 MVA
        "vk_percent": 10.0,
        "vkr_percent": 0.4,
        "pfe_kw": 50,
        "i0_percent": 0.07,
    },
    (115, 20): {
        "sn_mva": 100,  # Regional distribution: typically 100 MVA
        "vk_percent": 8.0,
        "vkr_percent": 0.5,
        "pfe_kw": 50,
        "i0_percent": 0.1,
    },
    (33, 20): {
        "sn_mva": 20,  # Local distribution: typically 20 MVA
        "vk_percent": 6.0,
        "vkr_percent": 0.7,
        "pfe_kw": 20,
        "i0_percent": 0.2,
    },
    (20, 0.4): {
        "sn_mva": 2,  # Final distribution: typically 2 MVA
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
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
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
pos_bus = {}

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
        geodata=(bus['x'], bus['y']),
        type='b',
        

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
                            'vn_kv'].values[0], net.bus.loc[net.bus.name == transformer['to_bus'], 'vn_kv'].values[0])

                min_kV = min(net.bus.loc[net.bus.name == transformer['from_bus'],
                            'vn_kv'].values[0], net.bus.loc[net.bus.name == transformer['to_bus'], 'vn_kv'].values[0])
                
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
                        name=f"Transformer from {transformer['from_bus']} to {transformer['to_bus']}",
                        max_loading_percent=200
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
                        if bus1['vn_kv'] == 500:
                            pp.create_line_from_parameters(
                                net,
                                from_bus=bus_index_map[bus1['bus_id']],  # Map the 'from' bus ID to its index in the network
                                to_bus=bus_index_map[bus2['bus_id']],    # Map the 'to' bus ID to its index in the network
                                length_km=haversine(bus1['x'], bus1['y'], bus2['x'], bus2['y']),  # Calculate the line length using the haversine formula
                                r_ohm_per_km=0.01,  # Resistance per kilometer in ohms
                                x_ohm_per_km=0.25,  # Reactance per kilometer in ohms
                                c_nf_per_km=12,  # Line capacitance per kilometer in nanofarads
                                max_i_ka=2.0,  # Maximum current capacity in kiloamperes
                                name=f"Line {line_id}",
                                #max_loading_percent=150,  # Maximum loading percentage
                                  # Assign a name to the line based on its unique ID,
                                #max_loading_percent=200,
                                #parallel= 3
                                #aximum loading percentage
                                
                            )

                        else:
                            # Line for 230 kV
                            pp.create_line_from_parameters(
                                net,
                                from_bus=bus_index_map[bus1['bus_id']],
                                to_bus=bus_index_map[bus2['bus_id']],
                                length_km=haversine(bus1['x'], bus1['y'], bus2['x'], bus2['y']),  # Default length; adjust if specific lengths are available
                                r_ohm_per_km=0.1,   # Resistance per km
                                x_ohm_per_km=0.4,   # Reactance per km
                                c_nf_per_km=9,      # Capacitance per km
                                max_i_ka=1.5,       # Maximum current in kA
                                name="230kV Line",
                                #max_loading_percent=150,  # Maximum loading percentage
                                #parallel= 3
                                
                            )
                        added_lines.add((bus1['bus_id'], bus2['bus_id'], line_id))
                    except Exception as e:
                        print(f"Error creating line between Bus {bus1['bus_id']} and Bus {bus2['bus_id']} for Line {line_id}")
                        raise e


for _, load in loads_df.iterrows():
    try:
        # Original bus (departure bus)
        departure_bus = bus_index_map[load['bus']]
        
        # Uncomment and modify the following lines if you need to create a new bus for the load with an offset position
        # pos = pos_bus[load['bus']]
        # # Create a new bus for the load with an offset position
        # load_bus = pp.create_bus(
        #     net,
        #     vn_kv=20,  # Voltage level of distribution (adjust as needed)
        #     type='b',  # Typically, loads are connected to 'b' type buses
        #     geodata=(pos[0] + 0.0001, pos[1] + 0.0001),  # Offset the position slightly
        #     name=f"{load['name']} Bus"
        # )
        
        # # Create a transformer connecting the departure bus to the load bus
        # pp.create_transformer_from_parameters(
        #     net,
        #     hv_bus=departure_bus,
        #     lv_bus=load_bus,
        #     sn_mva=load['p_kw'] * 0.001 /0.9,  # Example transformer size, adjust as needed
        #     vn_hv_kv=net.bus.loc[departure_bus, 'vn_kv'],  # Voltage of the departure bus
        #     vn_lv_kv=20,  # Voltage level of the new load bus
        #     vkr_percent = 1,  # Example short-circuit voltage, adjust as needed
        #     vk_percent=10,  # Example voltage drop under rated current
        #     pfe_kw=50,  # Example iron losses
        #     i0_percent=0.1,  # Example no-load current
        #     name=f"Transformer {load['name']}"
        # )

        # Create the load on the departure bus
        pp.create_load(
            net,
            bus=departure_bus,
            p_mw=load['p_kw'] * 0.001,  # Convert kW to MW
            max_p_mw=load['p_kw'] * 0.001,
            min_p_mw=0,  # Default to 0.0 if not provided
            name=load['name'],
            max_q_mvar=load['p_kw'] * 0.001 / 0.9,
            min_q_mvar=0,
            controllable=False
        )
        
    except Exception as e:
        print(f"Error creating load: {load}")
        raise e

# Ensure at least one Slack Bus is assigned
slack_assigned = False
# Création des générateurs en répartissant la génération si plusieurs bus sont proches (<500 m)
for _, gen in generators_df.iterrows():
    try:
        # Récupérer le bus de référence et sa position
        ref_bus_id = gen['bus_id']
        if ref_bus_id not in pos_bus:
            print(f"Position non trouvée pour le bus {ref_bus_id} du générateur {gen['generator_name']}. Utilisation du bus original.")
            ref_pos = None
        else:
            ref_pos = pos_bus[ref_bus_id]

        # Identifier tous les bus dans un rayon de 0.5 km autour du bus de référence
        nearby_bus_ids = []
        if ref_pos is not None:
            for bus_id, pos in pos_bus.items():
                distance = haversine(ref_pos[0], ref_pos[1], pos[0], pos[1])
                if distance <= 1:  # 0.5 km = 500 m
                    nearby_bus_ids.append(bus_id)
        else:
            nearby_bus_ids = [ref_bus_id]

        # Si plusieurs bus sont trouvés, répartir la génération
        if len(nearby_bus_ids) > 1:
            generation_share = gen['p_mw'] / len(nearby_bus_ids)
            print(f"Répartition du générateur '{gen['generator_name']}' sur les bus {nearby_bus_ids} avec {generation_share} MW chacun.")
            for bus in nearby_bus_ids:
                bus_index = bus_index_map[bus]
                pp.create_gen(
                    net,
                    bus=bus_index,
                    p_mw=1,  # Puissance active initiale (peut être ajustée)
                    vm_pu=1.0,
                    min_p_mw=0,
                    max_p_mw=generation_share,
                    min_q_mvar=-generation_share / 0.85,
                    max_q_mvar=generation_share / 0.85,
                    name=f"{gen['generator_name']} ({gen['type']}) - Part sur bus {bus}",
                    slack=(not slack_assigned and gen['type'] == 'natural gas'),
                    controllable=True
                )
                # Assigner le slack au premier générateur éligible
                if not slack_assigned and gen['type'] == 'natural gas':
                    slack_assigned = True
        else:
            # Aucun bus supplémentaire trouvé, on utilise le bus de référence (ou le plus proche)
            bus = nearby_bus_ids[0]
            bus_index = bus_index_map[bus]
            print(f"Création du générateur '{gen['generator_name']}' sur le bus {bus}.")
            pp.create_gen(
                net,
                bus=bus_index,
                p_mw=1,  # Puissance active initiale
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
        print(f"Erreur lors de la création du générateur {gen['generator_name']}: {e}")
        raise e


# Add an external grid if no slack generator is assigned
if not slack_assigned:
    pp.create_ext_grid(net, bus=0, vm_pu=1.0, va_degree=0)

# Check for isolated buses
isolated_buses = net.bus.index.difference(pd.concat([net.line["from_bus"], net.line["to_bus"], net.trafo["hv_bus"], net.trafo["lv_bus"]]))
if not isolated_buses.empty:
    print("Isolated buses:", isolated_buses)

# Add default geodata if missing
if net.bus_geodata.empty:
    for idx in net.bus.index:
        net.bus_geodata.loc[idx] = {"x": idx * 0.1, "y": idx * 0.1}
        
net.gen['min_vm_pu'] = 0.98
net.gen['max_vm_pu'] = 1.02



def run_power_optimization_with_debug(net, max_iterations=5, voltage_threshold=0.97, reactive_gen_step=1.0, max_generators=1):
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")

        try:
            # Exécuter l'Optimal Power Flow (OPF)
            print("Running Optimal Power Flow...")
            pp.runopp(net, delta=1e-6, debug=True, numba=True)
            print("Optimal Power Flow simulation succeeded.")
            break

            # # Vérification des tensions après optimisation
            # low_voltage_buses = net.res_bus[net.res_bus.vm_pu < voltage_threshold].sort_values(by='vm_pu')
            # if low_voltage_buses.empty:
            #     print("All bus voltages are within acceptable limits.")
            #     break  # Si tout est OK, on sort de la boucle
            # else:
            #     print(f"Low voltage detected at buses:\n{low_voltage_buses[['vm_pu']]}")
            #     print(f"Adding reactive generators to the {max_generators} most critical buses...")

            #     # Cibler les N bus les plus critiques
            #     critical_buses = low_voltage_buses.head(max_generators)
                
            #     for bus_index in critical_buses.index:
            #         if bus_index not in net.gen.bus.values:
            #             print(f"Adding reactive generator at bus {bus_index} with {300} MVAr range.")
            #             # pp.create_gen(net, 
            #             #         bus=bus_index, 
            #             #         p_mw=0,  # Pas de production active
            #             #         vm_pu=1,  # Tension cible réaliste
            #             #         min_q_mvar=-300,  # Plage de puissance réactive
            #             #         max_q_mvar=300,
            #             #         min_p_mw=0,  # Bornes actives pour éviter les comportements inattendus
            #             #         max_p_mw=0,
            #             #         controllable=True,  # Autoriser l'OPF à ajuster la réactivité
            #             #         min_vm_pu=0.98,  # Limites de tension pour éviter les surcharges
            #             #         max_vm_pu=1.02,
            #             #         name=f"Reactive Generator at Bus {bus_index}")
            #         else:
            #             print(f"Reactive generator already exists at bus {bus_index}.")

        except (pp.LoadflowNotConverged, pp.optimal_powerflow.OPFNotConverged):
            pp.diagnostic(net, report_style='detailed')
            print("Simulation did not converge.")

        except Exception as e:
            print(f"Unexpected error during simulation: {e}")
            break  # Sortie en cas d'erreur inattendue

    else:
        print("Max iterations reached. The optimization could not be completed.")

        
isolated_buses = set(net.bus.index) - set(net.line[['from_bus', 'to_bus']].values.flatten())
net.bus.drop(isolated_buses, inplace=True)
for component in ['line', 'trafo', 'gen', 'load']:
    invalid_entries = net[component][net[component]['in_service'] == False].index
    net[component].drop(invalid_entries, inplace=True)


run_power_optimization_with_debug(net)
# fig = simple_plotly(net)
# plt.show()

# Display generator settings after OPF
print("Generator settings after OPF:")
print(net.res_gen[['p_mw', 'vm_pu']])

total_generation = sum(net.res_gen['p_mw'])
total_demand = sum(net.res_load['p_mw'])
print(f"Generation: {total_generation} MW, Demand: {total_demand} MW")

# Save the network to a pickle file
pp.to_pickle(net, "my_network.p")
print("Network saved as pickle file: my_network.p")

# Vérification des contraintes de tension et de surcharge
out_of_bounds = net.res_bus[(net.res_bus.vm_pu < 0.98) | (net.res_bus.vm_pu > 1.02)]
overloaded_lines = net.res_line[net.res_line.loading_percent > 100].sort_values(by='loading_percent', ascending=False)
overloaded_transformers = net.res_trafo[net.res_trafo.loading_percent > 100]

# print("Overloaded buses:\n", out_of_bounds[:5])
# print("Overloaded lines:\n", overloaded_lines[:5])
# print("Overloaded transformers:\n", overloaded_transformers[:5])

low_voltage_buses = net.res_bus[net.res_bus.vm_pu < 0.95].sort_values(by='vm_pu')
