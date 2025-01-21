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
                                name=f"Line {line_id}"  # Assign a name to the line based on its unique ID
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
                                name="230kV Line"
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

for _, gen in generators_df.iterrows():
    try:
        # Step 1: Connect generator to its specific bus
        gen_bus = bus_index_map.get(gen['bus_id'])  # Get the mapped bus index
        if gen_bus is None:
            print(f"Warning: Bus {gen['bus_id']} not found for generator {gen['generator_name']}. Skipping.")
            continue
        
        # Assign Slack Bus to the first natural gas generator
        is_slack = False
        if gen['type'] == 'natural gas' and not slack_assigned:
            is_slack = True
            slack_assigned = True

        # Create generator
        pp.create_gen(
            net,
            bus=gen_bus,  # Link generator to the bus
            p_mw=1000,  # Active power in MW
            vm_pu=1.0,  # Voltage magnitude in per unit (typically 1.0)
            min_p_mw=0,  # Minimum active power generation
            max_p_mw=gen['p_mw'],  # Maximum active power generation
            min_q_mvar=-gen['p_mw']/0.9,  # Minimum reactive power generation (set as needed)
            max_q_mvar=gen['p_mw']/0.9,  # Maximum reactive power generation (set as needed)
            name=f"{gen['generator_name']} ({gen['type']})",
            slack=is_slack,  # Define if it's the slack bus
            controllable=True  # Set generator as controllable for OPF
        )

        # Optional Step 2: Add a transformer if generator voltage differs
        # Uncomment this if the generator is connected via a transformer
        # pp.create_transformer_from_parameters(
        #     net,
        #     hv_bus=gen_bus,  # High voltage bus
        #     lv_bus=gen_bus,  # Low voltage bus (or a new low voltage bus if needed)
        #     sn_mva=gen['p_mw'] / 0.9,  # Rated power in MVA (90% efficiency assumption)
        #     vn_hv_kv=net.bus.loc[gen_bus, 'vn_kv'],  # High voltage nominal voltage
        #     vn_lv_kv=20,  # Low voltage nominal voltage
        #     vkr_percent=0.2,  # Resistive losses (%)
        #     vk_percent=12.0,  # Short-circuit impedance (%)
        #     pfe_kw=50,  # Iron losses (kW)
        #     i0_percent=0.1,  # No-load current (%)
        #     name="20 kV Transformer"
        # )


    except Exception as e:
        print(f"Error creating generator and connection for: {gen['generator_name']}")
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


def run_power_optimization_with_debug(net, max_iterations=5):
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")

        try:

            # Optimal Power Flow
            print("Running Optimal Power Flow...")
            pp.runopp(net, delta=1e-6, debug=True)
            print("Optimal Power Flow simulation succeeded.")

            print("Optimization completed successfully.")
            break  # If everything works, exit the loop

        except (pp.LoadflowNotConverged, pp.optimal_powerflow.OPFNotConverged):
            print("Simulation did not converge. Running diagnostics for corrections...")
            diagnostic_report = pp.diagnostic(net, report_style='detailed')
            print(diagnostic_report)
            print("Corrections applied. Retrying simulation...")

        except Exception as e:
            print(f"Unexpected error during simulation: {e}")
            break  # Exit if an unexpected error occurs

    else:
        print("Max iterations reached. The optimization could not be completed.")
        
isolated_buses = set(net.bus.index) - set(net.line[['from_bus', 'to_bus']].values.flatten())
net.bus.drop(isolated_buses, inplace=True)
for component in ['line', 'trafo', 'gen', 'load']:
    invalid_entries = net[component][net[component]['in_service'] == False].index
    net[component].drop(invalid_entries, inplace=True)


run_power_optimization_with_debug(net)
fig = simple_plotly(net)
plt.show()

# Display generator settings after OPF
print("Generator settings after OPF:")
print(net.res_gen[['p_mw', 'vm_pu']])

total_generation = sum(net.res_gen['p_mw'])
total_demand = sum(net.res_load['p_mw'])
print(f"Generation: {total_generation} MW, Demand: {total_demand} MW")






# Save the network to a pickle file
pp.to_pickle(net, "my_network.p")

print("Network saved as pickle file: my_network.p")
