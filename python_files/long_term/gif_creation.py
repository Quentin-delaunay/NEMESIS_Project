import imageio.v2 as imageio
import os
import glob
import pickle
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import LogNorm

def generate_images_from_saved_states(
    saved_folder="saved_states",
    output_folder="frames",
    cmap_name="Reds",
    figsize=(14, 10),
    dpi=200
):
    """
    Loads the saved graphs and generates images for each of them with improved styling.
    
    For each node (bus), the overload is computed based on its voltage deviation:
      - If vm_pu < 0.98, overload = (0.98 - vm_pu)*100
      - If vm_pu > 1.02, overload = (vm_pu - 1.02)*100
    The overload value (as an integer percentage) is displayed centered in the node
    only if it is greater than 0.
    
    Edge annotations (loading %) are shown at the midpoint of each edge if loading > 100%.
    """
    # Retrieve the list of saved graph files
    graph_files = sorted(glob.glob(os.path.join(saved_folder, "graph_state_*.pkl")))

    # Compute global min and max loading across all graphs
    global_min = float("inf")
    global_max = float("-inf")
    for gf in graph_files:
        with open(gf, "rb") as f:
            G = pickle.load(f)
        for _, _, data in G.edges(data=True):
            if data.get("loading_percent") is not None:
                val = data["loading_percent"]
                global_min = min(global_min, val)
                global_max = max(global_max, val)
    if global_min == float("inf"):
        global_min = 1e-3
    if global_max == float("-inf"):
        global_max = 100

    print(f"Global simulation bounds: min = {global_min}, max = {global_max}")

    # Prepare colormap and normalization (log scale)
    cmap = plt.cm.get_cmap(cmap_name)
    norm = LogNorm(vmin=global_min, vmax=global_max)

    os.makedirs(output_folder, exist_ok=True)

    for gf in graph_files:
        timestep = int(os.path.basename(gf).split("_")[-1].split(".")[0])
        with open(gf, "rb") as f:
            G = pickle.load(f)

        # Determine node positions (use geo coordinates if available)
        if all("x" in G.nodes[n] and "y" in G.nodes[n] for n in G.nodes):
            pos = {n: (G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes}
        else:
            pos = nx.spring_layout(G, seed=42)

        # Determine edge colors based on loading values
        edge_colors = []
        for _, _, data in G.edges(data=True):
            loading = data.get("loading_percent")
            if loading is not None:
                edge_colors.append(cmap(norm(loading)))
            else:
                edge_colors.append("grey")

        # Pour chaque bus, calculer la surcharge en fonction de la tension (vm_pu)
        node_labels = {}
        node_sizes = []
        for n in G.nodes:
            vm = G.nodes[n].get("vm_pu")
            overload = 0
            if vm is not None:
                if vm < 0.98:
                    overload = (0.98 - vm) * 100
                elif vm > 1.02:
                    overload = (vm - 1.02) * 100
            # Afficher le pourcentage d'overload en entier uniquement s'il est > 0
            node_labels[n] = f"{overload:.0f}%" if overload > 0 else ""
            node_sizes.append(300 if overload > 0 else 200)

        # Récupérer la couleur des noeuds
        node_colors = [G.nodes[n].get("color", "black") for n in G.nodes]

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Dessiner les arêtes et les noeuds
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2, ax=ax)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, ax=ax)

        # Afficher les labels directement centrés dans le noeud
        nx.draw_networkx_labels(
            G,
            pos,
            labels=node_labels,
            font_color="white",
            font_size=12,
            horizontalalignment="center",
            verticalalignment="center",
            ax=ax
        )

        # Annoter chaque arête avec le loading si > 100%
        for (u, v, data) in G.edges(data=True):
            loading = data.get("loading_percent")
            if loading is not None and loading > 100:
                mid_x = (pos[u][0] + pos[v][0]) / 2
                mid_y = (pos[u][1] + pos[v][1]) / 2
                ax.text(
                    mid_x,
                    mid_y,
                    f"{loading:.1f}%",
                    fontsize=9,
                    color="blue",
                    ha="center",
                    va="center",
                    bbox=None
                )

        # Création de la colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation="vertical", format="%.2f")
        mid_val = (global_min * global_max) ** 0.5
        cbar.set_ticks([global_min, mid_val, global_max])
        cbar.set_ticklabels([f"{global_min:.2f}", f"{mid_val:.2f}", f"{global_max:.2f}"])
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label("Line Loading (%)", fontsize=12)

        ax.set_title(f"Network at Time Step {timestep}", fontsize=16, pad=15)
        ax.axis("off")

        filename = os.path.join(output_folder, f"frame_{timestep:04d}.png")
        plt.savefig(filename, bbox_inches="tight")
        plt.close(fig)
        print(f"Image saved: {filename}")


def create_gif_from_frames(output_gif="network_simulation.gif", frame_folder="frames", duration=1.0):
    """Assembles the generated images into an animated GIF with a slower frame rate."""
    images = []
    filenames = sorted(os.listdir(frame_folder))
    for filename in filenames:
        if filename.endswith(".png"):
            images.append(imageio.imread(os.path.join(frame_folder, filename)))
    if images:
        imageio.mimsave(output_gif, images, duration=duration)
        print(f"GIF created successfully: {output_gif}")
    else:
        print("No images found in the folder.")


# Example usage:
generate_images_from_saved_states(saved_folder="saved_states", output_folder="frames", cmap_name="Reds")
create_gif_from_frames(output_gif="network_simulation.gif", frame_folder="frames", duration=1.2)
