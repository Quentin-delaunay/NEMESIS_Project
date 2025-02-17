import streamlit as st
import os
import re
import pickle
import pandapower as pp
import networkx as nx
import plotly.graph_objects as go
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pandapower.topology import create_nxgraph
from streamlit_folium import st_folium

st.set_page_config(page_title="War-Gaming", layout="wide")
st.title("War-Gaming")
