import streamlit as st
import pandapower as pp
from utils import load_network

st.set_page_config(page_title="Advanced Operations", layout="wide")

st.title("Phase 3: Advanced Operations")

net = load_network()
st.subheader("Simulation: Add a Load")

bus_list = net.bus.index.tolist()
selected_bus = st.selectbox("Select a bus", bus_list)
new_load = st.number_input("Load value to add (in MW)", value=10.0, min_value=0.0)

if st.button("Add the Load"):
    try:
        pp.create_load(net, bus=selected_bus, p_mw=new_load)
        st.write("Load added successfully!")
        st.subheader("Load Details")
        st.write(net.load)
        pp.runpp(net)
        st.subheader("Bus Results after adding the load")
        st.write(net.res_bus)
    except Exception as e:
        st.error(f"Error adding the load or running the calculation: {e}")
