import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import shape
import json

# Global parameters
filter_pop = 50000 # only the urban areas with filter_pop ppl or more will be displayed / used

power_intensity = 1

GA_energy_consumption = { 'Coal':180.9, 'NaturalGas':812.4, 'Electricity_in': 355.4 + 10.8 + 250.3 + 25.7, 'Electricity_out': 262.2 } #in BTU
total_consumption = sum(GA_energy_consumption.values())

consumption_factor = 3
Elec_per_CP_2024 = consumption_factor * 11000e6/11e6 #MW
#print(6516+2189+4289+146+1316)

###########################################################
# Load data
###########################################################

# Load population data
population_path = 'data/USA_Urban_Areas_(1%3A500k-1.5M).geojson'
population_gdf = gpd.read_file(population_path) 
population_gdf['geometry'] = population_gdf['geometry'].buffer(0) # Fix invalid geometries

# Load region boundary data
georgia_boundary_path = 'data/georgia-counties.json'
georgia = gpd.read_file(georgia_boundary_path)
georgia['geometry'] = georgia['geometry'].buffer(0) # Fix invalid geometries
georgia_union = georgia.geometry.union_all() # Merge geometries
 
# Filter urban areas 
urban_in_georgia = population_gdf[population_gdf.intersects(georgia_union)].copy() # Intersect population data with region of interest
high_pop_areas = urban_in_georgia[urban_in_georgia['POP2010'] > filter_pop ].copy() # Only keep zones of more than filter_pop people

# Compute ratios of areas population to total population
total_population = high_pop_areas['POP2010'].sum()
high_pop_areas['POP_PERCENT'] = (high_pop_areas['POP2010'] / total_population) * 100

# Load gas pipeline data
pipeline_gas_path = 'data/NaturalGas_InterIntrastate_Pipelines_US_georgia.geojson'
pipeline_gas = gpd.read_file(pipeline_gas_path)
pipeline_gas = pipeline_gas[pipeline_gas.geometry.intersects(georgia_union)] # Filter gas pipelines that cross Georgia

# Load power lines data
power_lines_path = 'data/Transmission_Lines (2).geojson'
power_lines = gpd.read_file(power_lines_path)

# Load power plants data
power_plants_path = 'data/Power_Plants_georgia.geojson'  # Update this path
data = gpd.read_file(power_plants_path)

###########################################################
# Scaling the filter for the power plants
###########################################################

# Compute region consumption
GA_consumption = total_consumption*1E9 * 1.055 /1E6 #MWh for one year. Let's say that 0.35 of this energy is used in summer. 
GA_consumption_PC = GA_consumption/total_population

ratio_gas_elec = 373647/738986

# GA_consumption_Elec_PC = total_population * Elec_per_CP_2024 #MWh for one year 
# tot_power_installed = sum(data['Install_MW'].values())
# Energy_one_year = tot_power_installed * 8760 

# Compute needed energy per year
Power_needed = total_population * Elec_per_CP_2024  # yearly MWh

# Sort plants in descending order of installed capacity
sorted_plants = data.sort_values(by='Install_MW', ascending=False)

# Filter the power plants needed to meet energy demand
remaining_power_needed = Power_needed/1e6
        
filtered_power_plants = {}
for _, row in sorted_plants.iterrows():
    if remaining_power_needed > 0:
        plant_capacity = row['Install_MW']
        # Convert capacity in yearly MWh
        plant_annual_output = plant_capacity   # MWh/year
        

        if plant_annual_output <= remaining_power_needed:
           # If the plant can cover the remaining demand
            prim_source = row['PrimSource'].lower()  # Energy source of the plant
            if prim_source not in filtered_power_plants:
                filtered_power_plants[prim_source] = []
            filtered_power_plants[prim_source].append(row)
            remaining_power_needed -= plant_annual_output
        else:
            # Partially add the power plant (if necessary)
            prim_source = row['PrimSource'].lower()  # Energy source of the plant
            if prim_source not in filtered_power_plants:
                filtered_power_plants[prim_source] = []
            filtered_power_plants[prim_source].append(row)
            break
        
# Print result
c = 0
for source, plants in filtered_power_plants.items():
    for plant in plants:
        c+= plant['Install_MW']

print("Installed Power:", c)
print(f"Max demand times {consumption_factor}: {Power_needed}" )



###########################################################
# Plot
###########################################################

# Define colors and shapes for each energy source
source_colors = {
    'nuclear': ('b', 'o'), 'natural gas': ('g', 's'), 'coal': ('k', 'D'),
    'hydroelectric': ('c', '^'), 'petroleum': ('m', 'v'), 'pumped storage': ('orange', '^'),
    'biomass': ('brown', 'P'), 'other': ('gray', '*'), 'solar': ('y', 'h'), 'batteries': ('purple', 'o')
}

# Initialize figure and axes for display
fig, ax = plt.subplots(figsize=(10, 10))

# Plot Georgia map
gpd.GeoSeries([georgia_union]).plot(ax=ax, color='lightgray', edgecolor='black')

# Display high-population areas with a heat map
high_pop_areas.plot(column='POP2010', cmap='inferno', legend=False, ax=ax, edgecolor='black', linewidth=0.5)



# Add a unique colorbar for population percentage
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=high_pop_areas['POP2010'].min(), vmax=high_pop_areas['POP2010'].max()))
sm._A = []
# Adjusted pad value for spacing
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.02)
cbar.set_label('Population')
cbar.ax.yaxis.set_label_position('left')
cbar.ax.yaxis.set_ticks_position('right')

# Show gas pipelines in blue
pipeline_gas.plot(ax=ax, color='blue', linewidth=1, alpha=0.9, label='Natural Gas Pipeline', zorder=1)

# Display electrical transmission lines in red
power_lines.plot(ax=ax, color='red', linewidth=1, alpha=0.9, label='Electric Power Line >100 kV', zorder=2)


# Add power plants on the map with higher order
max_install_mw = max(data['Install_MW'])
for source, plants in filtered_power_plants.items():
    color, marker = source_colors.get(source, ('gray', 'o'))
    for plant in plants:
        size = (plant['Install_MW'] / max_install_mw) * 500
        ax.scatter(plant['Longitude'], plant['Latitude'], color=color, label=source, s=size, alpha=0.9, edgecolor='black', marker=marker, zorder=3)


# Legend for each energy source
handles = [plt.Line2D([0], [0], marker=marker, color='w', label=source, markerfacecolor=color, markersize=10) for source, (color, marker) in source_colors.items()]
handles.append(plt.Line2D([0], [0], color='blue', linewidth=1, label='Natural Gas Pipeline'))
handles.append(plt.Line2D([0], [0], color='red', linewidth=1, label='Electric Power Line'))
plt.legend(handles=handles, title='Energy Source', bbox_to_anchor=(-0.05, 1), loc='upper right')

# Configure title and axes
plt.title('Power Plants > 100 MW, Gas Pipelines, Power Transmission Lines, and Population in Georgia')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid()

# Show the map
plt.tight_layout()
plt.show()
