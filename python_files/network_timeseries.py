# ------------------------------------------------------
# Author: Quentin Delaunay
# Date: January 20, 2025
# Description: Script for loading a power panda network and running it with a time series
# File: network_timeseries.py
# ------------------------------------------------------
import pandapower as pp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# -------------------------------------------------------------------
# 1) Load the Pandapower network from a pickle file
# -------------------------------------------------------------------
net = pp.from_pickle("my_network.p")

# -------------------------------------------------------------------
# 2) Adjust generator voltage limits (sometimes aids convergence)
# -------------------------------------------------------------------
net.gen["min_vm_pu"] = 0.98
net.gen["max_vm_pu"] = 1.02

# -------------------------------------------------------------------
# 3) Ensure we have a Slack/External Grid
#    If none exists, create an external grid at the first bus
# -------------------------------------------------------------------
if "slack" not in net.gen.columns or not net.gen["slack"].any():
    print("No Slack generator found. Adding an External Grid at the first bus.")
    slack_bus = net.bus.index[0]  # Use the first bus as reference
    pp.create_ext_grid(net, bus=slack_bus, vm_pu=1.0, va_degree=0.0, name="External Grid")
    print(f"Bus {slack_bus} is configured as Slack (External Grid).")

print("Total installed generation capacity (MW):", net.gen["max_p_mw"].sum())
print("Total load demand (MW):", net.load["p_mw"].sum())

# -------------------------------------------------------------------
# 4) Clean up the network
#    - Remove isolated buses
#    - Remove out-of-service lines/gens/loads
# -------------------------------------------------------------------
isolated_buses = set(net.bus.index) - set(net.line[["from_bus", "to_bus"]].values.flatten())
net.bus.drop(isolated_buses, inplace=True)

for component in ["line", "trafo", "gen", "load"]:
    invalid_entries = net[component][net[component]["in_service"] == False].index
    net[component].drop(invalid_entries, inplace=True)

# -------------------------------------------------------------------
# 5) Ensure generators are controllable for OPF
# -------------------------------------------------------------------
if "controllable" not in net.gen.columns:
    net.gen["controllable"] = True

# -------------------------------------------------------------------
# 6) Define typical min-load fractions & marginal costs
# -------------------------------------------------------------------
min_load_fractions = {
    "nuclear": 0.9,       # e.g., 90% minimum load for nuclear
    "coal": 0.0,          # e.g., 0% min load for coal
    "natural gas": 0.0,   # e.g., 0% min load for combined-cycle gas
    "pumped storage": 0.00,
    "petroleum": 0.0,
    "wind": 0.00,
    "solar": 0.00
}

marginal_costs = {
    "nuclear": 10.0,
    "coal": 25.0,
    "natural gas": 35.0,
    "pumped storage": 50.0,
    "petroleum": 60.0,
    "wind": 5.0,
    "solar": 3.0
}

# (Optional) synonyms if some generator names differ
type_synonyms = {
    # Example: "gaz" -> "natural gas"
}

# -------------------------------------------------------------------
# 7) Create / Reset the poly_cost table if needed
#    (Here, we clear it to start from scratch)
# -------------------------------------------------------------------
expected_columns = ["object", "element", "et", "c0", "c1", "c2"]
if not hasattr(net, "poly_cost"):
    net["poly_cost"] = pd.DataFrame(columns=expected_columns)
else:
    net.poly_cost = net.poly_cost.reindex(columns=expected_columns)
net.poly_cost.drop(net.poly_cost.index, inplace=True)

# -------------------------------------------------------------------
# 8) (Optional) Avoid cp2/cq2 errors by creating columns at 0 if missing
# -------------------------------------------------------------------
for col in ["cp2_eur_per_mw2", "cq2_eur_per_mvar2"]:
    if col not in net.gen.columns:
        net.gen[col] = 0.0

# -------------------------------------------------------------------
# 9) Iterate over generators: set min_p_mw and create cost entries
#    with create_poly_cost(...)
# -------------------------------------------------------------------
for gen_idx in net.gen.index:
    gen_name = net.gen.at[gen_idx, "name"]

    # Extract the type in parentheses, e.g. "Plant Scherer (coal)" => "coal"
    match = re.search(r"\(([^)]+)\)", gen_name)
    if match:
        raw_type = match.group(1).strip().lower()
        # Check synonyms if relevant
        if raw_type in type_synonyms:
            gen_type = type_synonyms[raw_type]
        else:
            gen_type = raw_type
    else:
        gen_type = "other"

    # 9a) Determine the min_p_mw fraction
    if gen_type in min_load_fractions:
        frac = min_load_fractions[gen_type]
        net.gen.at[gen_idx, "min_p_mw"] = frac * net.gen.at[gen_idx, "max_p_mw"]
    else:
        net.gen.at[gen_idx, "min_p_mw"] = 0.0

    # 9b) Determine the linear marginal cost
    if gen_type in marginal_costs:
        cp1_val = marginal_costs[gen_type]
    else:
        cp1_val = 9999.0  # Very high cost if unknown type

    # 9c) Create a polynomial cost entry via Pandapower
    pp.create_poly_cost(
        net,
        element=gen_idx,     # generator index
        et="gen",            # "gen" for a generator
        cp1_eur_per_mw=cp1_val,  # linear cost
        cp0_eur=0.0,             # active power offset cost
        cq1_eur_per_mvar=0.0,    # reactive cost = 0
        cq0_eur=0.0,
        cp2_eur_per_mw2=0.0,     # no quadratic cost
        cq2_eur_per_mvar2=0.0,   # no quadratic reactive cost
        check=False              # avoid warnings if an entry already exists
    )

# -------------------------------------------------------------------
# 10) Time-Stepped OPF Simulation
#     Simulate 1 day (24 hrs) with sinusoidal load variation
# -------------------------------------------------------------------
n_timesteps = 24
scaling_factors = 0.8 + 0.2 * np.sin(2 * np.pi * np.arange(n_timesteps) / 24)

# Prepare to store results
demand_history = []
gen_history = []
generator_output_history = pd.DataFrame(index=np.arange(n_timesteps), columns=net.gen.index)

def run_time_series_opf(net, scaling_list):
    for t, scale in enumerate(scaling_list):
        print(f"\n-- Time step {t+1}/{len(scaling_list)} -- Scaling factor = {scale:.2f}")

        # Apply the scaling factor to all loads
        net.load["scaling"] = scale

        # Ensure we have an active Slack generator
        if not net.gen.loc[net.gen["slack"] & net.gen["in_service"]].any(axis=None):
            print("No active Slack generator. Attempting to reassign Slack...")
            slack_candidates = net.gen.loc[net.gen["in_service"]].index
            if len(slack_candidates) > 0:
                net.gen.at[slack_candidates[0], "slack"] = True
            else:
                # Fallback: create an external grid on the first bus
                fallback_bus = net.bus.index[0]
                pp.create_ext_grid(net, bus=fallback_bus, vm_pu=1.0, name="External Grid")

        # Run the OPF
        try:
            pp.runopp(net)

            # Collect results
            total_load = net.res_load["p_mw"].sum()
            total_gen = net.res_gen["p_mw"].sum()
            demand_history.append(total_load)
            gen_history.append(total_gen)

            generator_output_history.loc[t] = net.res_gen["p_mw"]
            print(f"   Demand = {total_load:.2f} MW, Generation = {total_gen:.2f} MW")

        except Exception as e:
            print(f"Error at timestep {t+1}: {e}")
            pp.diagnostic(net, report_style='detailed')
            demand_history.append(None)
            gen_history.append(None)
            generator_output_history.loc[t] = None

# -------------------------------------------------------------------
# 11) Execute the time-stepped OPF simulation
# -------------------------------------------------------------------
run_time_series_opf(net, scaling_factors)

# Filter valid time steps
valid_steps = [i for i, val in enumerate(demand_history) if val is not None]

# -------------------------------------------------------------------
# 12) Plot - Evolution of Demand vs. Generation
# -------------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(
    valid_steps,
    [demand_history[i] for i in valid_steps],
    label="Total Demand (MW)", marker="o"
)
plt.plot(
    valid_steps,
    [gen_history[i] for i in valid_steps],
    label="Total Generation (MW)", marker="s"
)
plt.title("Demand and Generation Over Time")
plt.xlabel("Time Steps (hours)")
plt.ylabel("Power (MW)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# 13) Plot - Individual Generator Outputs
# -------------------------------------------------------------------
plt.figure(figsize=(12, 8))
for gen_idx in net.gen.index:
    gen_name = net.gen.at[gen_idx, "name"]
    series_values = [
        generator_output_history.loc[t, gen_idx]
        for t in valid_steps
        if pd.notnull(generator_output_history.loc[t, gen_idx])
    ]
    plt.plot(valid_steps, series_values, label=gen_name, marker=".")

plt.title("Generator Outputs Over Time")
plt.xlabel("Time Steps (hours)")
plt.ylabel("Power (MW)")
plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
plt.grid(True)
plt.tight_layout()
plt.show()
