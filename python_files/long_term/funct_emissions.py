import pandapower as pp
import re
import pandas as pd
import compute_emissions as ce

# Function to evaluate CO2 reduction at each timestep
def co2_reduction_objective(initial_co2, reduction_target_percent, current_co2):
    """
    Calculate if the current CO2 emissions meet the reduction target.

    Parameters:
    initial_co2 (float): The initial CO2 emissions (in kg).
    reduction_target_percent (float): Desired CO2 reduction percentage.
    current_co2 (float): Current CO2 emissions (in kg).

    Returns:
    bool: True if current emissions meet the target, False otherwise.
    float: The target CO2 emission level (in kg).
    """
    target_co2 = initial_co2 * (1 - reduction_target_percent / 100)
    return current_co2 <= target_co2, target_co2

# Function to generate monthly CO2 reduction targets
def generate_monthly_targets(initial_co2, total_months, final_reduction_percent):
    """
    Generate a list of monthly CO2 reduction targets linearly decreasing from the initial CO2.

    Parameters:
    initial_co2 (float): The initial CO2 emissions (in kg).
    total_months (int): The total number of months over which the reduction occurs.
    final_reduction_percent (float): The final desired reduction percentage.

    Returns:
    list: Monthly CO2 targets in kg.
    """
    monthly_targets = []
    for month in range(total_months):
        reduction_fraction = (month + 1) / total_months
        target_co2 = initial_co2 * (1 - reduction_fraction * (final_reduction_percent / 100))
        monthly_targets.append(target_co2)
    return monthly_targets

# Function to adjust CO2 cost with penalty for high-emission generators
def adjust_co2_cost(net, co2_cost_increment=0.0):
    """
    Increase the cost associated with CO2 emissions in the optimization, applying higher penalties to high-emission generators.

    Parameters:
    net (pandapowerNet): The pandapower network.
    co2_cost_increment (float): Base amount to increase CO2 cost per MW.
    """
    if "poly_cost" not in net or net.poly_cost.empty:
        raise ValueError("Poly cost table is not initialized.")

    emission_factors = {
        "coal": 1001,
        "natural gas": 469,
        "petroleum": 840,
        "nuclear": 0,
        "solar": 0,
        "wind": 0,
        "pumped storage": 0
    }

    type_synonyms = {
        "gaz": "natural gas"
    }

    # Adjust cost with penalty based on emission factors
    for gen_idx in net.gen.index:
        gen_name = net.gen.at[gen_idx, "name"]
        match = re.search(r"\(([^)]+)\)", gen_name)
        if match:
            raw_type = match.group(1).strip().lower()
            gen_type = type_synonyms.get(raw_type, raw_type)
        else:
            gen_type = "other"

        if gen_type in emission_factors:
            emission_penalty = emission_factors[gen_type] * 0.1  # Penalty factor proportional to emissions
            total_increment = co2_cost_increment + emission_penalty
            net.poly_cost.loc[(net.poly_cost['et'] == 'gen') & (net.poly_cost['element'] == gen_idx), 'cp1_eur_per_mw'] += total_increment

# Function to run the optimization with CO2 reduction targets
def run_optimized_with_co2_reduction(net, total_months, final_reduction_percent, max_iterations=10):
    """
    Run the optimization iteratively until CO2 emissions meet the reduction target for each month.

    Parameters:
    net (pandapowerNet): The pandapower network.
    total_months (int): Total number of months to simulate.
    final_reduction_percent (float): Final desired CO2 reduction percentage.
    max_iterations (int): Maximum number of iterations per month to achieve the target.

    Returns:
    list: Final CO2 emissions and total costs for each month.
    """
    # Initial run to get baseline CO2 emissions
    pp.runopp(net, numba=True)
    initial_emissions_kg, _, _ = ce.calculate_emissions_and_cost(net)

    # Generate monthly reduction targets
    monthly_targets = generate_monthly_targets(initial_emissions_kg, total_months, final_reduction_percent)

    monthly_results = []

    for month in range(total_months):
        target_co2 = monthly_targets[month]
        print(f"\n--- Month {month + 1}: Target CO2 = {target_co2 / 1000:.2f} tonnes ---")

        for iteration in range(max_iterations):
            pp.runopp(net, numba=True)
            current_emissions_kg, total_cost_usd, _ = ce.calculate_emissions_and_cost(net)

            if current_emissions_kg <= target_co2:
                print(f"Target met for Month {month + 1} after {iteration + 1} iterations.")
                break

            adjust_co2_cost(net)
            print(f"Iteration {iteration + 1}: CO2 = {current_emissions_kg / 1000:.2f} tonnes, Target = {target_co2 / 1000:.2f} tonnes")
        else:
            print(f"Max iterations reached for Month {month + 1} without meeting target.")

        monthly_results.append({
            'Month': month + 1,
            'CO2 Emissions (tonnes)': current_emissions_kg / 1000,
            'Operational Cost (USD)': total_cost_usd,
            'Iterations': iteration + 1
        })

    return monthly_results

# Example usage
if __name__ == "__main__":
    net = pp.from_pickle("my_network.p")
    total_months = 12  # Example: 1-year simulation
    final_reduction_percent = 20  # Example: 20% CO2 reduction over the year

    results = run_optimized_with_co2_reduction(net, total_months, final_reduction_percent)

    # Display results
    for result in results:
        print(f"Month {result['Month']}: CO2 = {result['CO2 Emissions (tonnes)']:.2f} tonnes, Cost = ${result['Operational Cost (USD)']:.2f}, Iterations = {result['Iterations']}")

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("monthly_co2_reduction_results.csv", index=False)
