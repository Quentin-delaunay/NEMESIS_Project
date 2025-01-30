# National Energy Modeling for Evaluating Strategic Infrastructure Scenarios #

Power network simulation project, designed for creating, analyzing, and simulating a power network using `pandapower`. The project allows for network creation, time-series analysis, visualization, and data filtering using `streamlit`.

## Repository Structure
- `app.py`: A `streamlit` application for filtering data and visualizing the power network.
- `network_creation.py`: Creates a power network from filtered data and saves it for further analysis.
- `network_timeseries.py`: Loads a power network and runs it with a time series simulation.

## Installation
### Prerequisites
Ensure you have Python 3.8 or higher installed, along with the necessary dependencies.

### Required Packages
Install the required Python packages using:
```sh
pip install -r requirements.txt
```

## Usage
### 1 Filtering and visualizing input data
To filter data and visualize it using `streamlit`, run:
```sh
streamlit run app.py
```
The application allows users to interact with power network data, filter power plants, transmission lines, and gas pipelines, and visualize them on a map.

### 2. Creating a Power Network
To create a power network from filtered data, run:
```sh
python network_creation.py
```
This script loads filtered data from CSV files and generates a power network using `pandapower`.

### 3. Running a Time-Series Simulation
To run a time-series optimal power flow (OPF) simulation, execute:
```sh
python network_timeseries.py
```
This will simulate 12 time steps with sinusoidal load variation and generate plots for demand, generation, and network heatmaps.


## Data Sources
This project utilizes:
- Power plant data from `Power_Plants_georgia.geojson`
- Transmission line data from `Transmission_Lines_all.geojson`
- Urban area data from `USA_Urban_Areas.geojson`
- County boundaries from `georgia-counties.json`
- Gas pipeline data from `NaturalGas_InterIntrastate_Pipelines_US_georgia.geojson`

## Outputs
The scripts generate:
- Processed power network models saved as `.p` files.
- Time-series simulation results plotted for visualization.
- Streamlit-based interactive data filtering and visualization.

## Contributors
- Quentin Delaunay

