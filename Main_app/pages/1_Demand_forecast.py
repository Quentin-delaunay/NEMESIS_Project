import streamlit as st
import pickle
import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Generate demand forecast", layout="wide")

st.title("Phase 0: Setting up the Simulation")
st.write("This phase loads the daily demand model and allows you to configure simulation parameters.")
st.write("The daily demand model was trained on the data from 2019 to 2024. It is recomended to stay in this range.")
st.write("It was generated using the southwest dataset from the EIA. It was rescaled to correpond to the Georgia power consumption, but is not perfect.")
st.write("Therefore, there is several parameters that can be adjusted to change the forecast accuracy.")
         

# Load the Prophet model
try:
    with open('daily_model_smooth.pkl', 'rb') as f:
        model = pickle.load(f)
    st.success("Daily demand model loaded successfully.")
except Exception as e:
    st.error(f"Error loading daily_model_smooth.pkl: {e}")
    st.stop()

# Load error info silently and parse mu and sigma
try:
    with open('daily_error_smooth.txt', 'r') as f:
        error_lines = f.readlines()
    mu = None
    sigma = None
    for line in error_lines:
        if "Mean of residuals" in line or "Moyenne" in line:
            mu = float(line.split(":")[1].strip())
        if "Standard deviation of residuals" in line or "Ecart-type" in line:
            sigma = float(line.split(":")[1].strip())
    if mu is None or sigma is None:
        st.warning("Could not parse residual error values. Using 0.0 for both.")
        mu, sigma = 0.0, 0.0
except Exception as e:
    st.warning(f"Error loading daily_error_smooth.txt: {e}")
    mu, sigma = 0.0, 0.0

st.subheader("Daily Demand Forecast")
if 'prediction' in st.session_state:
    prediction = st.session_state['prediction']
    fig = go.Figure(
        go.Bar(
            x=prediction['ds'],
            y=prediction['yhat'],
            name='Forecasted Demand',
            marker_color='lightsalmon'
        )
    )
    fig.update_layout(
        title="Daily Demand Forecast",
        xaxis_title="Date",
        yaxis_title="Demand (MW)",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No forecast generated yet. Adjust the options below and generate a forecast.")

# Options below the graph in two columns
col_left, col_right = st.columns(2)
with col_left:
    st.subheader("Simulation Period & Adjustments")
    start_date = st.date_input("Start Date", value=datetime.date(2023, 1, 1))
    end_date = st.date_input("End Date", value=datetime.date(2023, 12, 31))
    if start_date >= end_date:
        st.error("End date must be after start date.")
        st.stop()
    offset = st.number_input("Offset (MW) to subtract from forecast", value=0.0, step=0.1)
    add_noise = st.checkbox("Activate Gaussian noise", value=False)
with col_right:
    st.subheader("Peak Limits")
    min_peak = st.number_input("Minimum Peak Limit (MW)", value=0.0)
    max_peak = st.number_input("Maximum Peak Limit (MW)", value=20000.0)
    if min_peak > max_peak:
        st.error("Minimum peak limit cannot be greater than maximum peak limit.")
        st.stop()

if st.button("Generate Forecast"):
    future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    future_df = pd.DataFrame({'ds': future_dates})
    forecast = model.predict(future_df)
    prediction = forecast[['ds', 'yhat']].copy()
    # Apply the offset (ensure it is in MW)
    prediction['yhat'] = prediction['yhat'] - float(offset)
    if add_noise:
        noise = np.random.normal(mu, sigma, size=len(prediction))
        prediction['yhat'] = prediction['yhat'] + abs(noise)
    prediction['yhat'] = prediction['yhat'].clip(lower=min_peak, upper=max_peak)
    st.session_state['prediction'] = prediction
    st.success("Forecast generated successfully.")
    
if st.button("Save Forecast"):
    output_folder = "Main_app/saved_forecast/"
    os.makedirs(output_folder, exist_ok=True)
    prediction.to_csv(f"{output_folder}load_forecast.csv", index=False)
    st.success("Forecast saved successfully.")