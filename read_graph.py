import networkx as nx

import matplotlib.pyplot as plt

# Read the graph from a GEXF file
G = nx.read_gexf('output/graph.gexf')

# Draw the graph
plt.figure(figsize=(12, 8))
nx.draw(G, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_color='black', edge_color='gray')
plt.title('Graph Visualization')
plt.show()