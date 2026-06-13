import streamlit as st
import pandas as pd
import numpy as np
import pickle
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap

# 1. Page Configuration
st.set_page_config(page_title="Miri Environmental Twin", layout="wide")

st.title("🏙️ Miri Microclimate Adaptation Framework")
st.markdown("An AI-driven predictive simulation tool for municipality thermal optimization.")

# 2. Load the AI Model (The Brain Trained on Day 1)
@st.cache_resource
def load_model():
    with open('miri_heat_model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    ai_model = load_model()
except FileNotFoundError:
    st.error("⚠️ 'miri_heat_model.pkl' not found in this folder. Please upload it to your project directory.")
    st.stop()

# 3. Sidebar Control Dashboard
st.sidebar.header("🛠️ Simulation Control Panel")
st.sidebar.markdown("Modify the city's surface conditions below to watch the microclimate shift.")

# Simulation Sliders
sim_concrete = st.sidebar.slider("Global Concrete Density (NDBI)", min_value=-0.10, max_value=0.50, value=0.25, step=0.05)
sim_canopy = st.sidebar.slider("Global Tree Canopy Density (NDVI)", min_value=0.00, max_value=0.60, value=0.20, step=0.05)

# Calculate interaction feature for the AI model
sim_ratio = sim_concrete / (sim_canopy + 1e-5)

# 4. Generate the Geographical Spatial Matrix for Miri
# This creates a 30x30 bounding box matrix over actual Miri city limits
@st.cache_data
def generate_miri_grid():
    lat_range = np.linspace(4.35, 4.45, 30) # From Marina up past Pujut
    lon_range = np.linspace(113.95, 114.03, 30)
    grid = []
    for lat in lat_range:
        for lon in lon_range:
            grid.append([lon, lat])
    return np.array(grid)

miri_coordinates = generate_miri_grid()

# 5. Run the Real-Time AI Inference Pass
# Format: ['NDVI', 'NDBI', 'Concrete_to_Green_Ratio', 'Longitude', 'Latitude']
features_matrix = np.zeros((len(miri_coordinates), 5))
features_matrix[:, 0] = sim_canopy   # NDVI
features_matrix[:, 1] = sim_concrete # NDBI
features_matrix[:, 2] = sim_ratio    # Ratio
features_matrix[:, 3] = miri_coordinates[:, 0] # Longitude
features_matrix[:, 4] = miri_coordinates[:, 1] # Latitude

# Predict temperatures for all points across Miri simultaneously
predicted_temperatures = ai_model.predict(features_matrix)

# Combine into a dataframe for mapping: [Latitude, Longitude, Weight/Temperature]
heatmap_data = np.zeros((len(miri_coordinates), 3))
heatmap_data[:, 0] = miri_coordinates[:, 1] # Lat
heatmap_data[:, 1] = miri_coordinates[:, 0] # Lon
heatmap_data[:, 2] = predicted_temperatures # Heat Weight

avg_temp = np.mean(predicted_temperatures)

# 6. Build the Interactive UI Layout (Two Equal Columns)
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🌍 Interactive Predictive Thermal Map")
    
    # Initialize a baseline geographic Folium Map centered on Miri City Centre
    m = folium.Map(location=[4.39, 113.99], zoom_start=13, tiles="cartodbpositron")
    
    # Inject our AI's predicted values directly into an interactive HeatMap layer
    # We subtract a baseline to accentuate the hot/cool variations visually
    heat_layer_data = [[row[0], row[1], float(row[2] - 20)] for row in heatmap_data]
    
    HeatMap(
        heat_layer_data,
        radius=25,
        blur=15,
        max_zoom=13,
        gradient={0.4: 'blue', 0.65: 'yellow', 1: 'red'}
    ).add_to(m)
    
    # Display map inside the Streamlit frame
    st_folium(m, width=800, height=500, returned_objects=[])

with col2:
    st.subheader("📊 Dynamic Metrics")
    st.metric(label="Predicted Average Urban Temperature", value=f"{avg_temp:.2f} °C")
    
    st.subheader("💡 Strategic Adaptation Protocols")
    # Provide smart contextual evaluation based on the AI's math
    if avg_temp > 35.0:
        st.error("🚨 *CRITICAL RISK STATE\n\nImpact:* High heat-island retention over built-up areas.\n\n*Action Item:* Mandatory cool-roof reflection mandates and green vegetative buffers required along highway zones.")
    elif avg_temp > 31.0:
        st.warning("⚠️ *MODERATE THERMAL ACCUMULATION\n\nImpact:* Heightened resident discomfort and grid cooling draw.\n\n*Action Item:* Intersect urban corridors with decentralized pocket parks to break up continuous concrete absorption.")
    else:
        st.success("✅ *STABLE CLIMATE RESILIENT MATRIX\n\nImpact:* Thermal balance within target parameters.\n\n*Action Item:* Preserve current urban canopy footprint.")