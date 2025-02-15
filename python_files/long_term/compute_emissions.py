import pandapower as pp
import pandas as pd
import re

# Load the existing network
net = pp.from_pickle("my_network.p")

# Emission factors in kg CO2/MWh
emission_factors = {
    "coal": 403.2,          # kg CO2/MWh
    "natural gas": 201.96,
    "petroleum": 265,
    "nuclear": 0,
    "solar": 0,
    "wind": 0,
    "pumped storage": 0
}


marginal_costs = {
    "nuclear": 77.16,
    "coal": 89.33,
    "natural gas": 42.72,
    "pumped storage": 57.12,
    "petroleum": 128.82,
    "wind": 31,
    "solar": 23
}

# Since the timestep is 1 month, convert costs to USD per MW per month
hours_per_month = 30 * 24  # Approximate number of hours in a month
marginal_costs_per_month = {k: v * hours_per_month for k, v in marginal_costs.items()}

# Synonyms for generator types
type_synonyms = {
    "gaz": "natural gas"
}

# Apply marginal costs to the optimization function
if not hasattr(net, "poly_cost"):
    net["poly_cost"] = pd.DataFrame(columns=["object", "element", "et", "c0", "c1", "c2"])
else:
    net.poly_cost = net.poly_cost.reindex(columns=["object", "element", "et", "c0", "c1", "c2"])
net.poly_cost.drop(net.poly_cost.index, inplace=True)

# Assign marginal costs to each generator
for gen_idx in net.gen.index:
    gen_name = net.gen.at[gen_idx, "name"]

    # Identify the generator type
    match = re.search(r"\(([^)]+)\)", gen_name)
    if match:
        raw_type = match.group(1).strip().lower()
        gen_type = type_synonyms.get(raw_type, raw_type)
    else:
        gen_type = "other"

    # Apply marginal cost if the type is known
    if gen_type in marginal_costs_per_month:
        cost_per_mw_month = marginal_costs_per_month[gen_type]
        pp.create_poly_cost(
            net,
            element=gen_idx,
            et="gen",
            cp1_eur_per_mw=cost_per_mw_month,  # Apply monthly cost per MW to optimization
            cp0_eur=0.0,
            cq1_eur_per_mvar=0.0,
            cq0_eur=0.0,
            cp2_eur_per_mw2=0.0,
            cq2_eur_per_mvar2=0.0,
            check=False
        )

# Run the optimal power flow optimization
pp.runopp(net, numba=True)

# Function to calculate CO2 emissions and operational cost
def calculate_emissions_and_cost(net):
    total_emissions_kg = 0
    total_cost_usd = 0
    generator_details = {}

    for gen_idx in net.gen.index:
        gen_name = net.gen.at[gen_idx, "name"]

        # Identify the generator type
        match = re.search(r"\(([^)]+)\)", gen_name)
        if match:
            raw_type = match.group(1).strip().lower()
            gen_type = type_synonyms.get(raw_type, raw_type)
        else:
            gen_type = "other"

        # Calculate emissions and cost if the type is known
        if gen_type in emission_factors and gen_type in marginal_costs_per_month:
            emission_factor = emission_factors[gen_type]
            cost_per_mw_month = marginal_costs_per_month[gen_type]
            power_mw = net.res_gen.at[gen_idx, "p_mw"]  # Power output in MW

            emissions_kg = emission_factor * power_mw * hours_per_month   # Monthly emissions in kg CO2
            cost_usd = cost_per_mw_month * power_mw     # Monthly cost in USD

            total_emissions_kg += emissions_kg
            total_cost_usd += cost_usd

            generator_details[gen_name] = {
                'Emissions (kg CO2)': emissions_kg,
                'Cost (USD)': cost_usd
            }

    return total_emissions_kg, total_cost_usd, generator_details

# Calculate CO2 emissions and operational cost
total_emissions_kg, total_cost_usd, generator_details = calculate_emissions_and_cost(net)

# Convert emissions to tonnes of CO2
total_emissions_tonnes = total_emissions_kg / 1000

# Display results
print(f"Total CO2 Emissions: {total_emissions_tonnes:.2f} tonnes")
print(f"Total Operational Cost: ${total_cost_usd:.2f}")

# Detailed emissions and cost per generator
details_df = pd.DataFrame.from_dict(generator_details, orient='index')
print(details_df)

# Optional: Save results to a CSV file
details_df.to_csv("generator_emissions_and_cost.csv")
