import streamlit as st
import pandas as pd
import numpy as np
import pickle
import folium
import io
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
from folium.plugins import Draw

st.set_page_config(page_title="Miri Environmental Twin", layout="wide")
st.title("🏙️ Miri Spatial SimCity Twin Engine")
st.markdown("Advanced Geographic Interpolation Engine for Real-Time Thermal Spectrum Simulations.")

# 1. Securely load the AI Brain Model
@st.cache_resource
def load_model():
    with open('miri_heat_model.pkl', 'rb') as f:
        return pickle.load(f)

ai_model = load_model()

# 2. Sidebar Configuration Setup
st.sidebar.header("🕹️ Object Blueprint Settings")
blueprint_type = st.sidebar.selectbox(
    "Select Structural Placement Material:",
    ["Dense Concrete Skyscraper Grid (High Heat Retention)", "Urban Green Canopy Park (Cooling Infrastructure)"]
)

# 3. Establish High-Resolution Grid Tracking Limits (Miri Boundaries)
LAT_MIN, LAT_MAX = 4.32, 4.46
LON_MIN, LON_MAX = 113.93, 114.05
GRID_SIZE = 30 # 30x30 resolution mesh matrix 

# Initialize persistent matrix layers directly inside state memory
if 'ndvi_layer' not in st.session_state:
    st.session_state.ndvi_layer = np.full((GRID_SIZE, GRID_SIZE), 0.25)
if 'ndbi_layer' not in st.session_state:
    st.session_state.ndbi_layer = np.full((GRID_SIZE, GRID_SIZE), 0.25)

import base64

# 4. Extract target materials based on sidebar selection
if blueprint_type == "Dense Concrete Skyscraper Grid (High Heat Retention)":
    target_ndvi, target_ndbi = 0.02, 0.55
else:
    target_ndvi, target_ndbi = 0.65, -0.05

# 5. Build Dynamic High-Resolution Arrays for the AI
lat_axis = np.linspace(LAT_MIN, LAT_MAX, GRID_SIZE)
lon_axis = np.linspace(LON_MIN, LON_MAX, GRID_SIZE)
LON, LAT = np.meshgrid(lon_axis, lat_axis)

# Calculate the feature ratio matrix
ratio_layer = st.session_state.ndbi_layer / (st.session_state.ndvi_layer + 1e-5)

# Flatten the spatial layers to feed into your 66.88% accurate AI Model
features_matrix = np.stack([
    st.session_state.ndvi_layer.flatten(),
    st.session_state.ndbi_layer.flatten(),
    ratio_layer.flatten(),
    LON.flatten(),
    LAT.flatten()
], axis=1)

# Generate predictions across the whole city footprint simultaneously
predicted_flat = ai_model.predict(features_matrix)
predicted_grid = predicted_flat.reshape((GRID_SIZE, GRID_SIZE))

mean_temp = np.mean(predicted_grid)
min_grid_temp = np.min(predicted_grid)
max_grid_temp = np.max(predicted_grid)

# 6. Advanced GIS Image Generation (Bicubic Thermal Smoothing Filter)
fig, ax = plt.subplots(figsize=(8, 8))
ax.axis('off')
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

# DYNAMIC VISUAL SPECTRUM: Lock the bounds strictly to the active range
# This forces the color spectrum to show deep transitions even for fractional degree shifts
im = ax.imshow(
    predicted_grid, 
    cmap='turbo', 
    interpolation='bicubic', 
    origin='lower',
    vmin=min_grid_temp, 
    vmax=max_grid_temp
)

# Render the layout directly into an in-memory transparent PNG byte buffer
buf = io.BytesIO()
fig.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
buf.seek(0)
image_base64 = base64.b64encode(buf.read()).decode('utf-8')
image_url = f"data:image/png;base64,{image_base64}"
plt.close(fig)

# 7. Initialize Base Leaflet Map Canvas
m = folium.Map(location=[4.393, 113.993], zoom_start=13, tiles="CartoDB positron")

# Project our continuous bicubic thermal spectrum layer perfectly over Miri's geographic bounds
folium.raster_layers.ImageOverlay(
    image=image_url,
    bounds=[[LAT_MIN, LON_MIN], [LAT_MAX, LON_MAX]],
    opacity=0.55,
    interactive=False,
    cross_origin=False
).add_to(m)

# Attach drawing capabilities onto the live map layer
draw_control = Draw(
    export=False,
    position='topleft',
    draw_options={'polyline': False, 'circle': False, 'marker': False, 'circlemarker': False}
)
draw_control.add_to(m)

# 8. Render Dual Panel Screen Viewport
col1, col2 = st.columns([3, 1])

with col1:
    output_map = st_folium(m, width=900, height=600, key="miri_raster_overlay_engine")

# 9. Map Coordinate Bounding Box Interceptor Loop
if output_map and output_map.get("last_active_drawing"):
    geometry = output_map["last_active_drawing"]["geometry"]
    
    if geometry["type"] in ["Polygon", "Rectangle"]:
        coords = geometry["coordinates"][0]
        df_bounds = pd.DataFrame(coords, columns=['Longitude', 'Latitude'])
        
        min_lon, max_lon = df_bounds['Longitude'].min(), df_bounds['Longitude'].max()
        min_lat, max_lat = df_bounds['Latitude'].min(), df_bounds['Latitude'].max()
        
        # Translate geographic coordinates directly into matrix index positions (0 to GRID_SIZE)
        lon_idx_min = int(np.clip((min_lon - LON_MIN) / (LON_MAX - LON_MIN) * GRID_SIZE, 0, GRID_SIZE - 1))
        lon_idx_max = int(np.clip((max_lon - LON_MIN) / (LON_MAX - LON_MIN) * GRID_SIZE, 0, GRID_SIZE - 1))
        lat_idx_min = int(np.clip((min_lat - LAT_MIN) / (LAT_MAX - LAT_MIN) * GRID_SIZE, 0, GRID_SIZE - 1))
        lat_idx_max = int(np.clip((max_lat - LAT_MIN) / (LAT_MAX - LAT_MIN) * GRID_SIZE, 0, GRID_SIZE - 1))
        
        # Re-verify boundaries are distinct before applying changes to state to prevent reload loops
        if (lon_idx_max >= lon_idx_min) and (lat_idx_max >= lat_idx_min):
            st.session_state.ndvi_layer[lat_idx_min:lat_idx_max + 1, lon_idx_min:lon_idx_max + 1] = target_ndvi
            st.session_state.ndbi_layer[lat_idx_min:lat_idx_max + 1, lon_idx_min:lon_idx_max + 1] = target_ndbi
            st.rerun()

# 10. Dashboard Analytics Display Module
with col2:
    st.subheader("📊 Spectrum Analytics")
    st.metric(label="Miri Spatial Mean Temperature", value=f"{mean_temp:.2f} °C")
    
    # Real-time tracking display of model fluctuations
    st.markdown("---")
    st.markdown("*Microclimate Dynamic Frame:*")
    st.info(f"🔥 Max Peak Intensity: {max_grid_temp:.2f} °C\n\n❄️ Min Valley Intensity: {min_grid_temp:.2f} °C")
    
    st.markdown("---")
    st.markdown("*Infrastructure Adaptation Status:*")
    if max_grid_temp - min_grid_temp < 0.1:
        st.write("Current footprint reflects baseline environmental parameters. Draw shapes to simulate modifications.")
    elif mean_temp > 39.40:
        st.error("🚨 *CRITICAL RETENTION ERROR*\n\nBuilt infrastructure footprint triggers localized thermal clustering. Canopy introduction recommended.")
    else:
        st.success("🌲 *THERMAL MITIGATION DETECTED*\n\nVegetation matrix successfully breaks radiation bounds, producing microclimate cooling pockets.")
        
    st.markdown("---")
    if st.button("Reset Entire Urban Matrix", use_container_width=True):
        if 'ndvi_layer' in st.session_state:
            del st.session_state['ndvi_layer']
        if 'ndbi_layer' in st.session_state:
            del st.session_state['ndbi_layer']
        st.rerun()
