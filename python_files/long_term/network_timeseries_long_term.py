import pandapower as pp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
import networkx as nx
from matplotlib.colors import LogNorm
from datetime import datetime, timedelta

# Load the Pandapower network from a pickle file
net = pp.from_pickle("my_network.p")

# Ensure generators have appropriate voltage limits
net.gen["min_vm_pu"] = 0.98
net.gen["max_vm_pu"] = 1.02

# Ensure we have a Slack/External Grid
if "slack" not in net.gen.columns or not net.gen["slack"].any():
    print("No Slack generator found. Adding an External Grid at the first bus.")
    slack_bus = net.bus.index[0]  # Use the first bus as reference
    pp.create_ext_grid(net, bus=slack_bus, vm_pu=1.0, va_degree=0.0, name="External Grid")

print("Total installed generation capacity (MW):", net.gen["max_p_mw"].sum())
print("Total load demand (MW):", net.load["p_mw"].sum())

# Define long-term simulation parameters
years = 10  # Simulation duration in years
timesteps_per_year = 12  # Monthly resolution
total_timesteps = years * timesteps_per_year
start_date = datetime.now() #TODO: change to 2025-01-01

demand_growth_rate = 0.02  # 2% per year

def compute_scaling_factor(step):
    """Compute the demand scaling factor for a given timestep."""
    demand_2025 = 16.3 # MW
    year_demand_prediction = [16.3, 17.3, 18.3, 20.25, 22.2, 23.35, 24.5, 24.85, 25.2, 25.45, 25.7] # MW from 2025 to 2035 from winter peak 2025 IRP prediction
    year_index = step // timesteps_per_year
    month_index = step % timesteps_per_year
    if year_index >= len(year_demand_prediction) - 1:
        monthly_step_demand = year_demand_prediction[-1]
    else:
        start_demand = year_demand_prediction[year_index]
        end_demand = year_demand_prediction[year_index + 1]
        monthly_step_demand = start_demand + (end_demand - start_demand) * (month_index / timesteps_per_year)
    return monthly_step_demand/demand_2025

demand_history = []
gen_history = []
generator_output_history = pd.DataFrame(index=np.arange(total_timesteps), columns=net.gen.index)

images = []

def run_long_term_simulation(net, total_timesteps):
    for t in range(total_timesteps):
        scale_factor = compute_scaling_factor(t)
        current_date = start_date + timedelta(days=30 * t)
        print(f"\n-- Time step {t+1}/{total_timesteps} ({current_date.strftime('%Y-%m')}) -- Scaling factor = {scale_factor:.4f}")

        net.load["scaling"] = scale_factor

        if not net.gen.loc[net.gen["slack"] & net.gen["in_service"]].any(axis=None):
            print("No active Slack generator. Attempting to reassign Slack...")
            slack_candidates = net.gen.loc[net.gen["in_service"].index]
            if not slack_candidates.empty:
                net.gen.at[slack_candidates.index[0], "slack"] = True
            else:
                fallback_bus = net.bus.index[0]
                pp.create_ext_grid(net, bus=fallback_bus, vm_pu=1.0, name="External Grid")

        try:
            pp.runopp(net, numba=True)
            total_load = net.res_load["p_mw"].sum()
            total_gen = net.res_gen["p_mw"].sum()
            demand_history.append(total_load)
            gen_history.append(total_gen)
            generator_output_history.loc[t] = net.res_gen["p_mw"]
            print(f"   Demand = {total_load:.2f} MW, Generation = {total_gen:.2f} MW")
            
            if (t + 1) % timesteps_per_year == 0:
                year = (t + 1) // timesteps_per_year
                
                graph = pp.topology.create_nxgraph(net, include_lines=True, include_trafos=False)
                line_loading = net.res_line["loading_percent"]
                norm = LogNorm(vmin=max(line_loading.min(), 1e-3), vmax=line_loading.max())
                edge_colors = [plt.cm.viridis(norm(val)) for val in line_loading]
                
                if "x" in net.bus_geodata.columns and "y" in net.bus_geodata.columns:
                    pos = {bus: (net.bus_geodata.at[bus, "x"], net.bus_geodata.at[bus, "y"]) for bus in net.bus.index}
                else:
                    print("Geographic positions not available. Using spring layout.")
                    pos = nx.spring_layout(graph, seed=42)
                
                fig, ax = plt.subplots(figsize=(12, 8))
                nx.draw_networkx_nodes(graph, pos, node_size=50, node_color="black", alpha=0.9, ax=ax)
                nx.draw_networkx_edges(graph, pos, edge_color=edge_colors, width=2, ax=ax)
                
                sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax, orientation="vertical")
                cbar.set_label("Line Loading (%)")
                
                ax.set_title(f"Heatmap of Line Loading in Year {year}")
                ax.axis("off")
                plt.tight_layout()
                plt.savefig(f"line_loading_year_{year}.png")
                images.append(f"line_loading_year_{year}.png")
                plt.close()
                
        except Exception as e:
            print(f"Error at timestep {t+1}: {e}")
            demand_history.append(None)
            gen_history.append(None)
            generator_output_history.loc[t] = None

run_long_term_simulation(net, total_timesteps)

valid_steps = [i for i, val in enumerate(demand_history) if val is not None]

plt.figure(figsize=(14, 8))
plt.plot(valid_steps, [demand_history[i] for i in valid_steps], label="Total Demand (MW)", marker="o", linewidth=2)
plt.plot(valid_steps, [gen_history[i] for i in valid_steps], label="Total Generation (MW)", marker="s", linewidth=2)
plt.xlabel("Time Steps (Months)")
plt.ylabel("Power (MW)")
plt.title("Long-Term Demand and Generation Evolution")
plt.legend()
plt.grid()
plt.show()

# Create a GIF from the saved plots
imageio.mimsave("line_loading_evolution.gif", [imageio.imread(img) for img in images], duration=1.5)
