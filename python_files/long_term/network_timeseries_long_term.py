import pandapower as pp
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime, timedelta
import re
import pickle
import funct_emissions as fe
import compute_emissions as ce

# Constantes globales
DEFAULT_MIN_VM_PU = 0.98
DEFAULT_MAX_VM_PU = 1.02

MIN_LOAD_FRACTIONS = {
    "nuclear": 0,
    "coal": 0.0,
    "natural gas": 0.0,
    "pumped storage": 0.0,
    "petroleum": 0.0,
    "wind": 0.0,
    "solar": 0.0
}

MARGINAL_COSTS = {
    "nuclear": 77.16,
    "coal": 89.33,
    "natural gas": 42.72,
    "pumped storage": 57.12,
    "petroleum": 128.82,
    "wind": 31,
    "solar": 23
}

# Si vous avez des synonymes pour le type de générateur, ajoutez-les ici
TYPE_SYNONYMS = {
    # Exemple : "gaz" -> "natural gas"
}


def load_and_prepare_network(filename: str):
    """
    Charge le réseau pandapower depuis un fichier pickle et le prépare :
      - Ajustement des limites de tension des générateurs
      - Ajout d'une grille externe (slack) si nécessaire
      - Nettoyage du réseau (buses isolés, éléments hors service)
      - Configuration des coûts polynomiaux pour l'OPF
    """
    net = pp.from_pickle(filename)
    
    # 1) Ajuster les limites de tension
    net.gen["min_vm_pu"] = DEFAULT_MIN_VM_PU
    net.gen["max_vm_pu"] = DEFAULT_MAX_VM_PU

    # 2) S'assurer qu'une grille externe (slack) existe
    if "slack" not in net.gen.columns or not net.gen["slack"].any():
        print("Aucun générateur Slack trouvé. Ajout d'une grille externe au premier bus.")
        slack_bus = net.bus.index[0]
        pp.create_ext_grid(net, bus=slack_bus, vm_pu=1.0, va_degree=0.0, name="External Grid")

    print("Capacité installée totale (MW):", net.gen["max_p_mw"].sum())
    print("Demande totale (MW):", net.load["p_mw"].sum())

    # 3) Nettoyage du réseau
    isolated_buses = set(net.bus.index) - set(net.line[["from_bus", "to_bus"]].values.flatten())
    net.bus.drop(isolated_buses, inplace=True)
    for comp in ["line", "trafo", "gen", "load"]:
        invalid_entries = net[comp][net[comp]["in_service"] == False].index
        net[comp].drop(invalid_entries, inplace=True)
    
    # 4) S'assurer que les générateurs sont contrôlables
    if "controllable" not in net.gen.columns:
        net.gen["controllable"] = True

    # 5) Création/réinitialisation de la table poly_cost
    expected_columns = ["object", "element", "et", "c0", "c1", "c2"]
    if not hasattr(net, "poly_cost"):
        net["poly_cost"] = pd.DataFrame(columns=expected_columns)
    else:
        net.poly_cost = net.poly_cost.reindex(columns=expected_columns)
    net.poly_cost.drop(net.poly_cost.index, inplace=True)

    # 6) Éviter des erreurs sur cp2/cq2 en créant ces colonnes à 0 si elles manquent
    for col in ["cp2_eur_per_mw2", "cq2_eur_per_mvar2"]:
        if col not in net.gen.columns:
            net.gen[col] = 0.0

    # 7) Itérer sur les générateurs pour définir min_p_mw et créer les coûts polynomiaux
    for gen_idx in net.gen.index:
        gen_name = net.gen.at[gen_idx, "name"]
        match = re.search(r"\(([^)]+)\)", gen_name)
        if match:
            raw_type = match.group(1).strip().lower()
            gen_type = TYPE_SYNONYMS.get(raw_type, raw_type)
        else:
            gen_type = "other"

        # Définir la puissance minimale
        if gen_type in MIN_LOAD_FRACTIONS:
            frac = MIN_LOAD_FRACTIONS[gen_type]
            net.gen.at[gen_idx, "min_p_mw"] = frac * net.gen.at[gen_idx, "max_p_mw"]
        else:
            net.gen.at[gen_idx, "min_p_mw"] = 0.0

        # Définir le coût marginal linéaire
        cp1_val = MARGINAL_COSTS.get(gen_type, 9999.0)

        # Créer l'entrée de coût polynomial
        pp.create_poly_cost(
            net,
            element=gen_idx,
            et="gen",
            cp1_eur_per_mw=cp1_val,
            cp0_eur=0.0,
            cq1_eur_per_mvar=0.0,
            cq0_eur=0.0,
            cp2_eur_per_mw2=0.0,
            cq2_eur_per_mvar2=0.0,
            check=False
        )
    return net


def compute_scaling_factor(year: int, month: int) -> float:
    """
    Calcule le facteur d'échelle de la demande pour une année et un mois donnés.
    """
    baseline_demand = 16.3  # MW pour 2025
    year_demand_prediction = [16.3, 17.3, 18.3, 20.25, 22.2, 23.35, 24.5, 24.85, 25.2, 25.45, 25.7]
    seasonal_variation = np.sin((month / 12) * 2 * np.pi) * 0.1
    if year >= len(year_demand_prediction) - 1:
        monthly_demand = year_demand_prediction[-1]
    else:
        start_demand = year_demand_prediction[year]
        end_demand = year_demand_prediction[year + 1]
        monthly_demand = start_demand + (end_demand - start_demand) * (month / 12)
    adjusted_demand = monthly_demand * (1 + seasonal_variation)
    return adjusted_demand / baseline_demand


def save_network_state(net, timestep: int, output_folder="saved_states"):
    """
    Sauvegarde l'état du réseau pandapower et du graphe NetworkX associé pour un pas de temps donné.
    Le graphe inclut les attributs des lignes (loading_percent) et des bus (tension, couleur, géo si disponible).
    """
    # Création du graphe NetworkX à partir du réseau pandapower
    graph = pp.topology.create_nxgraph(net, include_lines=True, include_trafos=False)
    
    # Ajouter l'information de chargement à chaque arête
    for i, (u, v, data) in enumerate(graph.edges(data=True)):
        try:
            loading = net.res_line.iloc[i]["loading_percent"]
        except IndexError:
            loading = None
        data["loading_percent"] = loading
    
    # Vérifier si les données géographiques sont disponibles
    has_geo = ("bus_geodata" in net and 
               "x" in net.bus_geodata.columns and 
               "y" in net.bus_geodata.columns)
    
    # Ajouter les attributs aux bus
    for bus in net.bus.index:
        try:
            voltage = net.res_bus.at[bus, "vm_pu"]
        except KeyError:
            voltage = None
        color = "red" if voltage is not None and (voltage < 0.98 or voltage > 1.02) else "black"
        if bus in graph.nodes:
            graph.nodes[bus]["vm_pu"] = voltage
            graph.nodes[bus]["color"] = color
            if has_geo:
                try:
                    graph.nodes[bus]["x"] = net.bus_geodata.at[bus, "x"]
                    graph.nodes[bus]["y"] = net.bus_geodata.at[bus, "y"]
                except KeyError:
                    pass
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Sauvegarder le réseau pandapower
    net_filename = os.path.join(output_folder, f"net_state_{timestep:04d}.pkl")
    with open(net_filename, "wb") as f:
        pickle.dump(net, f)
    
    # Sauvegarder le graphe NetworkX
    graph_filename = os.path.join(output_folder, f"graph_state_{timestep:04d}.pkl")
    with open(graph_filename, "wb") as f:
        pickle.dump(graph, f)
    
    print(f"Pas de temps {timestep} sauvegardé dans le dossier '{output_folder}'.")


def run_long_term_simulation(net, years=10, timesteps_per_year=12):
    """
    Exécute une simulation par pas de temps (OPF avec contraintes CO2) et retourne les historiques
    de demande, de production, des sorties et des coûts des générateurs, ainsi que l'historique des émissions CO2.
    """
    total_timesteps = years * timesteps_per_year
    # Calculer la situation de base en émissions et coûts
    total_emissions_kg, total_cost_usd, generator_details = ce.calculate_emissions_and_cost(net)
    co2_targets = fe.generate_monthly_targets(total_emissions_kg, total_timesteps, final_reduction_percent=25)
    
    # Initialisation des historiques
    demand_history = []
    gen_history = []
    co2_emissions_history = []
    
    generator_cost_history = pd.DataFrame(index=np.arange(total_timesteps), columns=net.gen["name"])
    generator_output_history = pd.DataFrame(index=np.arange(total_timesteps), columns=net.gen["name"])
    
    for t in range(total_timesteps):
        failed = False
        # Exemple : mettre temporairement un générateur hors service au pas de temps 5*12 et le rétablir au pas de temps 6*12
        if t == 5 * timesteps_per_year:
            net.gen.at[0, "in_service"] = False
        if t == 6 * timesteps_per_year:
            net.gen.at[0, "in_service"] = True
        
        year_index = t // timesteps_per_year
        month_index = t % timesteps_per_year
        scale_factor = compute_scaling_factor(year_index, month_index)
        print(f"\n-- Pas de temps {t+1}/{total_timesteps} ({2025+year_index}-{month_index}) -- Facteur d'échelle = {scale_factor:.4f}")
        
        net.load["scaling"] = scale_factor
        
        try:
            pp.runopp(net, numba=True)
            current_emissions_kg, total_cost_usd, generator_details = ce.calculate_emissions_and_cost(net)
            print("Optimisation réussie.")
            print(f"Émissions CO2: {current_emissions_kg/1000:.2f} tonnes, Coût total: {total_cost_usd:.2f} USD")
            
            out_of_bounds = net.res_bus[(net.res_bus.vm_pu < 0.98) | (net.res_bus.vm_pu > 1.02)]
            overloaded_lines = net.res_line[net.res_line.loading_percent > 100].sort_values(by='loading_percent', ascending=False)
            overloaded_transformers = net.res_trafo[net.res_trafo.loading_percent > 100]
            print("Buses hors tolérance:\n", out_of_bounds.head())
            print("Lignes surchargées:\n", overloaded_lines.head())
            print("Transformateurs surchargés:\n", overloaded_transformers.head())
        except pp.optimal_powerflow.OPFNotConverged:
            print(f"L'OPF n'a pas convergé au pas de temps {t}.")
            failed = True
            pp.diagnostic(net)
            out_of_bounds = net.res_bus[(net.res_bus.vm_pu < 0.95) | (net.res_bus.vm_pu > 1.05)]
            overloaded_lines = net.res_line[net.res_line.loading_percent > 100].sort_values(by='loading_percent', ascending=False)
            overloaded_transformers = net.res_trafo[net.res_trafo.loading_percent > 100]
            print("Buses hors tolérance:\n", out_of_bounds.head())
            print("Lignes surchargées:\n", overloaded_lines.head())
            print("Transformateurs surchargés:\n", overloaded_transformers.head())
            current_emissions_kg = 0.0
            total_cost_usd = 0.0
        
        total_load = net.res_load["p_mw"].sum() if "res_load" in net else 0.0
        total_gen = net.res_gen["p_mw"].sum() if "res_gen" in net else 0.0
        demand_history.append(total_load)
        gen_history.append(total_gen)
        co2_emissions_history.append(current_emissions_kg)
        
        # Mise à jour des historiques pour chaque générateur
        for gen_idx in net.gen.index:
            gen_name = net.gen.at[gen_idx, "name"]
            if gen_name in generator_details:
                generator_cost_history.at[t, gen_name] = generator_details[gen_name].get('Cost (USD)', 0.0)
            else:
                generator_cost_history.at[t, gen_name] = generator_cost_history.at[t-1, gen_name] if t > 0 else 0.0
            
            if not failed:
                generator_output_history.at[t, gen_name] = net.res_gen.at[gen_idx, "p_mw"]
            else:
                generator_output_history.at[t, gen_name] = 0
        
        # Sauvegarder l'état du réseau et du graphe
        save_network_state(net, t)
    
    return (demand_history, gen_history, generator_output_history, 
            generator_cost_history, co2_emissions_history, co2_targets)


def plot_demand_and_generation(demand_history, gen_history, generator_output_history, net):
    """
    Affiche l'évolution de la demande totale, de la génération totale et des contributions de certains générateurs.
    """
    valid_steps = [i for i in range(len(demand_history)) if demand_history[i] > 0 and gen_history[i] > 0]
    plt.figure(figsize=(14, 8))
    plt.plot(valid_steps, [demand_history[i] for i in valid_steps], label="Demande totale (MW)", linewidth=2)
    plt.plot(valid_steps, [gen_history[i] for i in valid_steps], label="Génération totale (MW)", linewidth=2)
    
    for gen_idx in net.gen.index:
        gen_name = net.gen.at[gen_idx, "name"]
        series_values = [generator_output_history.at[t, gen_name] for t in valid_steps 
                         if pd.notnull(generator_output_history.at[t, gen_name])]
        if max(series_values, default=0) > 100:
            plt.plot(valid_steps, series_values, label=f"{gen_name}", linestyle="--")
    
    plt.title("Évolution de la demande, de la génération et des contributions par centrale")
    plt.xlabel("Pas de temps (mois)")
    plt.ylabel("Puissance (MW)")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_cost_and_emissions(generator_cost_history, co2_emissions_history, net):
    """
    Affiche les coûts opérationnels par type de générateur et les émissions totales de CO2.
    """
    # Conversion en valeurs numériques
    generator_cost_history = generator_cost_history.astype(float)
    
    # Regroupement par type
    net.gen['type'] = net.gen['name'].str.extract(r'\((.*?)\)', expand=False).fillna('Unknown')
    cost_by_type = generator_cost_history.sum().groupby(net.gen.set_index('name')['type']).sum()
    cost_by_type = cost_by_type[cost_by_type > 0]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    if not cost_by_type.empty:
        cost_by_type.plot(kind='bar', ax=axes[0], color='skyblue')
        axes[0].set_title('Coût opérationnel par type de générateur')
        axes[0].set_ylabel('Coût (USD)')
        axes[0].grid(axis='y')
    else:
        axes[0].set_visible(False)
    
    # Affichage des émissions totales (non ventilées par type dans ce cas)
    emissions_series = pd.Series(co2_emissions_history, name='Emissions')
    emissions_series.plot(kind='bar', ax=axes[1], color='salmon')
    axes[1].set_title('Émissions totales de CO2 au cours du temps')
    axes[1].set_ylabel('Émissions (kg CO2)')
    axes[1].grid(axis='y')
    
    plt.tight_layout()
    plt.show()


def plot_co2_emissions(co2_emissions_history):
    """
    Affiche l'évolution des émissions de CO2 au cours de la simulation.
    """
    valid_steps = [i for i in range(len(co2_emissions_history)) if co2_emissions_history[i] > 0]
    plt.figure(figsize=(14, 8))
    plt.plot(valid_steps, [co2_emissions_history[i] / 1000 for i in valid_steps],
             label="Émissions CO2 (tonnes)", marker="o", linewidth=2, color="red")
    plt.title("Émissions CO2 au cours du temps")
    plt.xlabel("Pas de temps (mois)")
    plt.ylabel("Émissions CO2 (tonnes)")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    # Charger et préparer le réseau
    net = load_and_prepare_network("my_network.p")
    
    # Lancer la simulation
    (demand_history, gen_history, generator_output_history, 
     generator_cost_history, co2_emissions_history, co2_targets) = run_long_term_simulation(net)
    
    # Afficher les résultats
    plot_demand_and_generation(demand_history, gen_history, generator_output_history, net)
    plot_cost_and_emissions(generator_cost_history, co2_emissions_history, net)
    plot_co2_emissions(co2_emissions_history)


if __name__ == "__main__":
    main()
