import pygame
import networkx as nx
import matplotlib.pyplot as plt
from Generator import CoalPlant, GasPlant, NuclearPlant, Renewable
from Sink import Sink
from Source import Source
from Transmission import Transmission

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
FONT = pygame.font.SysFont("Arial", 16)

# Setup Pygame screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Energy Infrastructure Simulation")

# Create a graph
G = nx.Graph()

# Function to draw the graph
def draw_graph(graph, screen):
    pos = nx.spring_layout(graph)  # Topology layout
    for edge in graph.edges(data=True):
        pygame.draw.line(screen, BLUE, pos[edge[0]], pos[edge[1]], 2)

    for node in graph.nodes(data=True):
        pygame.draw.circle(screen, RED if node[1]['type'] == 'generator' else GREEN, pos[node[0]], 10)
        label = FONT.render(node[0], True, BLACK)
        screen.blit(label, (pos[node[0]][0], pos[node[0]][1]))

# Function to add a node
def add_node(graph, node_type, name, location, **kwargs):
    print(location)
    graph.add_node(name, type=node_type, **kwargs)

# Function to add an edge
def add_edge(graph, node1, node2, transmission):
    graph.add_edge(node1, node2, transmission=transmission)

# Main simulation loop
def main():
    running = True
    clock = pygame.time.Clock()

    # Predefined nodes (For demonstration purposes)
    coal_plant = CoalPlant("Coal Plant A", (200, 200), 500, 250)
    gas_plant = GasPlant("Gas Plant B", (400, 200), 300, 250)
    nuclear_plant = NuclearPlant("Nuclear Plant C", (600, 200), 1200, 250, is_on=True)
    solar_plant = Renewable("Solar Plant D", (300, 400), 100, "Solar", 80)
    
    city_sink = Sink("City A", (200, 600), demand={'gas': 100, 'fuel': 200, 'electricity': 400})
    industrial_sink = Sink("Factory B", (600, 600), demand={'electricity': 500})
    external_sink = Sink("External", (700, 300), is_external=True)

    # Add nodes to the graph
    add_node(G, 'generator', coal_plant.name, coal_plant.location, capacity=coal_plant.capacity, current_production=coal_plant.current_production)
    add_node(G, 'generator', gas_plant.name, gas_plant.location, capacity=gas_plant.capacity, current_production=gas_plant.current_production)
    add_node(G, 'generator', nuclear_plant.name, nuclear_plant.location, capacity=nuclear_plant.capacity, current_production=nuclear_plant.current_production)
    add_node(G, 'generator', solar_plant.name, solar_plant.location, capacity=solar_plant.capacity, current_production=solar_plant.current_production)
    add_node(G, 'sink', city_sink.name, city_sink.location, demand=city_sink.demand)
    add_node(G, 'sink', industrial_sink.name, industrial_sink.location, demand=industrial_sink.demand)
    add_node(G, 'sink', external_sink.name, external_sink.location, demand=external_sink.demand, is_external=True)

    # Add transmission lines (edges)
    transmission_line_1 = Transmission("Electric Line 1", "electric", max_capacity=500, current_flux=300)
    transmission_line_2 = Transmission("Gas Pipeline", "pipeline", max_capacity=400, current_flux=200)
    add_edge(G, coal_plant.name, city_sink.name, transmission_line_1)
    add_edge(G, gas_plant.name, external_sink.name, transmission_line_2)
    add_edge(G, nuclear_plant.name, industrial_sink.name, transmission_line_1)
    add_edge(G, solar_plant.name, industrial_sink.name, transmission_line_1)

    while running:
        screen.fill(WHITE)

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Draw the graph (topology)
        draw_graph(G, screen)

        # Update the display
        pygame.display.flip()

        # Control the frame rate
        clock.tick(60)

    pygame.quit()

def main2():
    
    # Predefined nodes (For demonstration purposes)
    coal_plant = CoalPlant("Coal Plant A", (200, 200), 500, 250)
    gas_plant = GasPlant("Gas Plant B", (400, 200), 300, 250)
    nuclear_plant = NuclearPlant("Nuclear Plant C", (600, 200), 1200, 250, is_on=True)
    solar_plant = Renewable("Solar Plant D", (300, 400), 100, "Solar", 80)
    
    city_sink = Sink("City A", (200, 600), demand={'gas': 100, 'fuel': 200, 'electricity': 400})
    industrial_sink = Sink("Factory B", (600, 600), demand={'electricity': 500})
    external_sink = Sink("External", (700, 300), is_external=True)

    # Add nodes to the graph
    add_node(G, 'generator', coal_plant.name, (200, 200))
    add_node(G, 'generator', gas_plant.name, gas_plant.location, capacity=gas_plant.capacity, current_production=gas_plant.current_production)
    add_node(G, 'generator', nuclear_plant.name, nuclear_plant.location, capacity=nuclear_plant.capacity, current_production=nuclear_plant.current_production)
    add_node(G, 'generator', solar_plant.name, solar_plant.location, capacity=solar_plant.capacity, current_production=solar_plant.current_production)
    add_node(G, 'sink', city_sink.name, city_sink.location, demand=city_sink.demand)
    add_node(G, 'sink', industrial_sink.name, industrial_sink.location, demand=industrial_sink.demand)
    add_node(G, 'sink', external_sink.name, external_sink.location, demand=external_sink.demand, is_external=True)

    # Add transmission lines (edges)
    transmission_line_1 = Transmission("Electric Line 1", "electric", max_capacity=500, current_flux=300)
    transmission_line_2 = Transmission("Gas Pipeline", "pipeline", max_capacity=400, current_flux=200)
    add_edge(G, coal_plant.name, city_sink.name, transmission_line_1)
    add_edge(G, gas_plant.name, external_sink.name, transmission_line_2)
    add_edge(G, nuclear_plant.name, industrial_sink.name, transmission_line_1)
    add_edge(G, solar_plant.name, industrial_sink.name, transmission_line_1)

    plt.figure(figsize=(12,12))
    pos = nx.get_node_attributes(G, "pos")

    nx.draw(G)
    # nx.draw_networkx_nodes(G, pos, node_color="white", node_size=500, edgecolors="black")#, font_size=11)
    # nx.draw_networkx_labels(G, pos)
    plt.tight_layout()
    plt.axis('off')
    plt.subplots_adjust(left=0.2, right=1, top=0.9, bottom=0)
    # plt.savefig("first_graph.pdf", bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main2()
