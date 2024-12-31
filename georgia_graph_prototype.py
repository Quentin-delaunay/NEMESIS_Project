import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
# plt.style.use('classic')
COLOR = 'black'
mpl.rcParams['text.color'] = COLOR
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR
mpl.rcParams['legend.frameon'] = 'False'
plt.rcParams.update({'font.size': 12})
import matplotlib.image as mpimg
import json
import geopandas as gpd
import numpy as np
from Generator import CoalPlant, GasPlant, NuclearPlant, Renewable
from Sink import Sink
from Source import Source
from Transmission import Transmission

# Create a graph
G = nx.Graph()

def load_data():
    # Global parameters
    filter_pop = 50000 # only the urban areas with filter_pop ppl or more will be displayed / used
    
    GA_energy_consumption = { 'Coal':180.9, 'NaturalGas':812.4, 'Electricity_in': 355.4 + 10.8 + 250.3 + 25.7, 'Electricity_out': 262.2 } #in BTU
    total_consumption = sum(GA_energy_consumption.values())

    consumption_factor = 3
    Elec_per_CP_2024 = consumption_factor * 11000e6/11e6 #MW

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

    # Load power lines data
    power_lines_path = 'data/Transmission_Lines (2).geojson'
    power_lines = gpd.read_file(power_lines_path)

    # Load power plants data
    power_plants_path = 'data/Power_Plants_georgia.geojson'  # Update this path
    data = gpd.read_file(power_plants_path)

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

    return high_pop_areas, power_lines, filtered_power_plants
    
def main():
    high_pop_areas, power_lines, filtered_power_plants  = load_data()

    city_names = np.array(high_pop_areas.NAME)
    city_pop = np.array(high_pop_areas.POP2010)
    city_geo= np.array(high_pop_areas.geometry)
    city_loc = []
    for multipoly in city_geo:
        lats = []
        lons = []
        for poly in multipoly.geoms:
            points = poly.exterior.coords[:-1]
            for p in points:
                lats.append(p[0])
                lons.append(p[1])
        lats = np.array(lats)
        lons = np.array(lons)
        lat, lon = np.mean(lats), np.mean(lons)
        city_loc.append((lat, lon))

    for i in range(len(city_loc)):
        city_sink = Sink(city_names[i], city_loc[i], demand={'gas': 0, 'fuel': 0, 'electricity': city_pop[i]})
        print("X : "+city_sink.name.split(",")[0])
        G.add_node(city_sink.name.split(",")[0], pos = city_sink.location, node_shape='s', node_color = 'red')
    
    for source, plants in filtered_power_plants.items():
        for plant in plants:
            if source == "coal":
                node_plant = CoalPlant(plant["Plant_Name"], (plant["Longitude"], plant["Latitude"] ), plant['Install_MW'], plant['Install_MW'])
                G.add_node(node_plant.name, pos=node_plant.location, node_color="black", node_shape='d')
            elif source == "natural gas":
                node_plant = GasPlant(plant["Plant_Name"], (plant["Longitude"], plant["Latitude"] ), plant['Install_MW'], plant['Install_MW'])
                G.add_node(node_plant.name, pos= node_plant.location, node_color="green", node_shape='d')
            elif source == "nuclear":
                node_plant = NuclearPlant(plant["Plant_Name"], (plant["Longitude"], plant["Latitude"] ), plant['Install_MW'], plant['Install_MW'])
                G.add_node(node_plant.name, pos= node_plant.location, node_color="blue", node_shape='d')
            else:
                break
            
    G.add_edge("Edwin I Hatch", 'Warner Robins, GA'.split(",")[0], style = '--')
    G.add_edge("Edwin I Hatch", 'Brunswick, GA'.split(",")[0], style = '-')
    G.add_edge("Edwin I Hatch", 'Albany, GA'.split(",")[0], style = '--')
    G.add_edge("Edwin I Hatch", 'Columbus, GA--AL'.split(",")[0], style = '--')
    G.add_edge("Vogtle", 'Augusta-Richmond County, GA--SC'.split(",")[0], style = '-')
    G.add_edge("Vogtle", 'Macon, GA'.split(",")[0], style='--')
    G.add_edge("Vogtle", 'McIntosh Combined Cycle Facility', style = '-')
    G.add_edge('McIntosh Combined Cycle Facility', 'Savannah, GA'.split(",")[0], style = '-')
    G.add_edge("Scherer", 'Augusta-Richmond County, GA--SC'.split(",")[0], style = '--')
    G.add_edge("Scherer", 'Warner Robins, GA'.split(",")[0], style = '-')
    G.add_edge("Scherer", 'Atlanta, GA'.split(",")[0], style = '-')
    G.add_edge("Dahlberg", 'Athens-Clarke County, GA'.split(",")[0], style = '-')
    G.add_edge("Bowen", 'Atlanta, GA'.split(",")[0], style = '-')
    G.add_edge("Bowen", 'Cartersville, GA'.split(",")[0], style = '-')
    G.add_edge("Bowen", 'Rome, GA'.split(",")[0], style = '-')
    G.add_edge('Thomas A Smith Energy Facility', "Bowen", style = '-')
    G.add_edge("Tenaska Georgia Generation Facility", 'Atlanta, GA'.split(",")[0], style = '-')
    G.add_edge("Wansley Combined Cycle", 'Atlanta, GA'.split(",")[0], style = '-')
    G.add_edge('Gainesville, GA'.split(",")[0], 'Atlanta, GA'.split(",")[0], style = '-')
    G.add_edge('Thomas A Smith Energy Facility', 'Chattanooga, TN--GA'.split(",")[0], style = '-')
    G.add_edge('Thomas A Smith Energy Facility', 'Dalton, GA'.split(",")[0], style = '-')


    # Add transmission lines (edges)
    # transmission_line_1 = Transmission("Electric Line 1", "electric", max_capacity=500, current_flux=300)
    # transmission_line_2 = Transmission("Gas Pipeline", "pipeline", max_capacity=400, current_flux=200)
    # add_edge(G, coal_plant.name, city_sink.name, transmission_line_1)
    # add_edge(G, gas_plant.name, external_sink.name, transmission_line_2)
    # add_edge(G, nuclear_plant.name, industrial_sink.name, transmission_line_1)

    background_image = mpimg.imread("data/georgia_background.png")
    fig, ax = plt.subplots()
    ax.imshow(background_image, extent=[-85.6,-80.75, 30.34, 35], aspect='auto')
    
    pos = nx.get_node_attributes(G, "pos")

    source_colors = {
    'nuclear': ('b', 'o'), 'natural gas': ('g', 's'), 'coal': ('k', 'D'),
    'hydroelectric': ('c', '^'), 'petroleum': ('m', 'v'), 'pumped storage': ('orange', '^'),
    'biomass': ('brown', 'P'), 'other': ('gray', '*'), 'solar': ('y', 'h'), 'batteries': ('purple', 'o')
}   
    for node in G.nodes(data=True):
        print(node[0],node[1])
        nx.draw_networkx_nodes(G, pos, nodelist = [node[0]], node_size=250, node_color=node[1]["node_color"], node_shape=node[1]["node_shape"], edgecolors="black")
    for edge in G.edges(data=True):
        nx.draw_networkx_edges(G, pos, edgelist=[(edge[0],edge[1])], arrows=True, connectionstyle='arc3, rad = 0.1', style=edge[2]["style"])
  
    # nx.draw_networkx_labels(G, pos, font_size=6, font_color="white")


    # nx.draw(G, pos, ax=ax, with_labels=True, node_color="skyblue", edge_color='grey', font_size=5)
    # nx.draw_networkx_nodes(G, pos, node_color="white", node_size=500, edgecolors="black")#, font_size=11)
    # nx.draw_networkx_labels(G, pos)
    plt.xlim(-87,-80)
    plt.ylim(30,35.5)
    plt.tight_layout()
    plt.axis('off')
    plt.subplots_adjust(left=0.2, right=1, top=0.9, bottom=0)
    plt.savefig("prototype_graph_hurricane.png", bbox_inches='tight', transparent=False)
    plt.show()

if __name__ == "__main__":
    main()