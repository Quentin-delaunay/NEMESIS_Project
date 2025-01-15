import pandapower as pp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Charger le réseau depuis un fichier pickle
net = pp.from_pickle("my_network.p")

# Vérifier ou ajouter une source externe (Slack Bus)
if "slack" not in net.gen.columns or not net.gen["slack"].any():
    print("Aucun générateur Slack trouvé. Ajout d'un bus Slack.")
    slack_bus = net.bus.index[0]  # Utiliser le premier bus comme référence
    pp.create_ext_grid(net, bus=slack_bus, vm_pu=1.0, va_degree=0.0, name="External Grid")
    print(f"Bus {slack_bus} configuré comme Slack via une source externe.")

# Ajouter des coûts aux générateurs pour permettre l'optimisation OPF
if "cost_per_mw" not in net.gen.columns:
    net.gen["cost_per_mw"] = np.linspace(10, 50, len(net.gen))  # Coût croissant entre 10 et 50 €/MW

# S'assurer que les générateurs sont contrôlables pour l'OPF
if "controllable" not in net.gen.columns:
    net.gen["controllable"] = True

# Ajouter des limites aux générateurs
if "min_p_mw" not in net.gen.columns:
    net.gen["min_p_mw"] = 0  # Production minimale
if "max_p_mw" not in net.gen.columns:
    net.gen["max_p_mw"] = net.gen["p_mw"] * 1.5  # Production maximale (150 % des valeurs initiales)

# Nombre de pas de temps (une semaine, 24 heures par jour)
n_timesteps = 7 * 24

# Générer une série temporelle simple pour les oscillations jour/nuit
scaling_factors = 1 + 0.5 * np.sin(2 * np.pi * np.arange(n_timesteps) / 24)  # Oscillation journalière

# Initialiser les historiques pour la demande, la production totale et le contrôle des générateurs
demand_history = []
gen_history = []
generator_output_history = pd.DataFrame(index=np.arange(n_timesteps), columns=net.gen.index)

# Définir des incidents aléatoires sur les générateurs
np.random.seed(42)  # Fixer la graine pour des résultats reproductibles
incident_probability = 0.1  # Probabilité qu'un générateur tombe en panne à un moment donné
generator_incidents = {
    gen_idx: np.random.choice([True, False], size=n_timesteps, p=[incident_probability, 1 - incident_probability])
    for gen_idx in net.gen.index
}

# Fonction pour exécuter l'OPF avec des charges oscillantes et des incidents
def run_opf_with_oscillating_load_and_incidents(net, scaling_factors):
    for t, scaling in enumerate(scaling_factors):
        print(f"Simulation de l'étape temporelle {t + 1}/{len(scaling_factors)}...")

        # Appliquer le facteur de scaling à toutes les charges
        net.load["scaling"] = scaling

        # Gérer les incidents sur les générateurs
        for gen_idx, incidents in generator_incidents.items():
            net.gen.at[gen_idx, "in_service"] = not incidents[t]  # Activer/Désactiver le générateur

        # Vérifier s'il reste un bus Slack
        if not net.gen.loc[net.gen["slack"] & net.gen["in_service"]].any(axis=None):
            print("Aucun générateur Slack actif. Réassignation d'un bus Slack.")
            slack_candidates = net.gen.loc[net.gen["in_service"]].index
            if len(slack_candidates) > 0:
                net.gen.at[slack_candidates[0], "slack"] = True
                print(f"Générateur {slack_candidates[0]} configuré comme Slack.")
            else:
                slack_bus = net.bus.index[0]  # Par défaut, le premier bus
                pp.create_ext_grid(net, bus=slack_bus, vm_pu=1.0, va_degree=0.0, name="External Grid")
                print(f"Bus {slack_bus} configuré comme Slack via une source externe.")

        # Tenter d'exécuter l'OPF avec gestion des erreurs
        try:
            pp.runopp(net)
            # Collecter les résultats en cas de succès
            total_demand = net.res_load["p_mw"].sum()
            total_generation = net.res_gen["p_mw"].sum()
            demand_history.append(total_demand)
            gen_history.append(total_generation)
            generator_output_history.loc[t] = net.res_gen["p_mw"]

        except Exception as e:
            print(f"Erreur lors de l'optimisation à l'étape {t + 1}: {e}")
            # Ajouter des valeurs par défaut en cas d'échec
            demand_history.append(None)
            gen_history.append(None)
            generator_output_history.loc[t] = None
            continue

# Lancer la simulation
run_opf_with_oscillating_load_and_incidents(net, scaling_factors)

# Supprimer les étapes avec des résultats manquants
valid_steps = [i for i, val in enumerate(demand_history) if val is not None]

# Graphique 1 : Evolution de la demande et de la production
plt.figure(figsize=(10, 6))
plt.plot(
    [time_steps for time_steps in valid_steps],
    [demand_history[i] for i in valid_steps],
    label="Total Demand (MW)", linestyle='-', marker='o'
)
plt.plot(
    [time_steps for time_steps in valid_steps],
    [gen_history[i] for i in valid_steps],
    label="Total Generation (MW)", linestyle='--', marker='s'
)
plt.title("Evolution of Demand and Generation Over a Week")
plt.xlabel("Time Steps (hours)")
plt.ylabel("Power (MW)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Graphique 2 : Evolution de la production des générateurs
plt.figure(figsize=(12, 8))
for gen_idx in net.gen.index:
    plt.plot(
        [time_steps for time_steps in valid_steps],
        [generator_output_history.loc[i, gen_idx] for i in valid_steps if generator_output_history.loc[i, gen_idx] is not None],
        label=f"Generator {gen_idx}", linestyle='-', marker='.'
    )
plt.title("Generator Output Over a Week")
plt.xlabel("Time Steps (hours)")
plt.ylabel("Generator Power Output (MW)")
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
plt.grid(True)
plt.tight_layout()
plt.show()
