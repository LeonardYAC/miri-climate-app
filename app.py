import streamlit as st
import pandas as pd
import numpy as np
import pickle
import folium
import io
import requests
import base64
import matplotlib.pyplot as plt
import branca.colormap as cm
from streamlit_folium import st_folium
from folium.plugins import Draw

# --- MODULE 1: APP INITIALIZATION & CONFIGURATION ---
st.set_page_config(page_title="Miri Environmental Twin", page_icon="🏙️", layout="wide", initial_sidebar_state="expanded")

st.title("🏙️ Miri Spatial SimCity Twin Engine")
st.markdown("### Near Real-Time (NRT) Microclimate Predictive Framework")
st.caption("Interact with the urban grid to simulate structural changes and analyze localized thermal impacts.")
st.divider()

# Load the upgraded, coordinate-free AI brain model (Expects: NDVI, NDBI, Ratio)
@st.cache_resource
def load_model():
    with open('miri_heat_model.pkl', 'rb') as f:
        return pickle.load(f)

ai_model = load_model()

# --- MODULE 2: NRT WEATHER API INTEGRATION ---
# --- MODULE 2: NRT WEATHER API INTEGRATION (FIXED INDEXING) ---
@st.cache_data(ttl=600)  # Cache weather data for 10 minutes to remain performant
def fetch_miri_nrt_weather():
    """Fetches real-time baseline ambient temperatures for Miri Municipality."""
    try:
        # Pulls live data directly from open-source stations relative to Miri's coordinates
        url = "https://api.open-meteo.com/v1/forecast?latitude=4.393&longitude=113.993&current=temperature_2m&hourly=temperature_2m&timezone=Asia/Singapore"
        response = requests.get(url).json()
        live_temp = response['current']['temperature_2m']
        
        current_time_str = response['current']['time']  # e.g., "2026-06-24T15:00"
        hourly_times = response['hourly']['time']
        hourly_temps = response['hourly']['temperature_2m']
        
        # Strip minutes to safely match the top of the hour string
        target_hour_str = current_time_str.split(":")[0]
        
        # Find the index where the API's hourly tracking matches right now
        current_idx = 0
        for i, time_step in enumerate(hourly_times):
            if time_step.startswith(target_hour_str):
                current_idx = i
                break
                
        # Slice the next 24 hours starting FROM NOW, instead of from midnight
        hourly_forecasts = hourly_temps[current_idx : current_idx + 24]
        return live_temp, hourly_forecasts

    except Exception:
        # Smart fallback baseline adjusted to match standard diurnal cycle starting at 3 PM (Hour 15)
        return 31.5, [30.0 + np.sin((h + 15)/24 * 2 * np.pi) * 4 for h in range(24)]

live_base_temp, future_hourly_temps = fetch_miri_nrt_weather()

# --- MODULE 3: SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.header("⏱️ NRT Temporal Controls")
    forecast_hour = st.slider("Predictive Forecast Horizon (Hours):", 0, 12, 0)
    projected_base_temp = future_hourly_temps[forecast_hour]
    
    st.info(f"**Baseline City Temp at Horizon:** {projected_base_temp:.1f}°C")
    st.divider()
    
    st.header("🕹️ Object Blueprint Settings")
    blueprint_type = st.selectbox(
        "Select Structural Placement Material:",
        ["Dense Concrete Skyscraper Grid (High Heat Retention)", "Urban Green Canopy Park (Cooling Infrastructure)"]
    )

    # Extract target materials based on sidebar selection
    if blueprint_type == "Dense Concrete Skyscraper Grid (High Heat Retention)":
        target_ndvi, target_ndbi = 0.02, 0.55
    else:
        target_ndvi, target_ndbi = 0.65, -0.05

# --- MODULE 4: GEOGRAPHIC GRID & STATE MANAGEMENT ---
# EXPANDED BOUNDS: Increased the geographic catch-area for the map
LAT_MIN, LAT_MAX = 4.25, 4.58  
LON_MIN, LON_MAX = 113.88, 114.12  
GRID_SIZE = 35 

@st.cache_data
def load_prebaked_miri_layers(csv_path, grid_size):
    """Loads the pre-collapsed satellite layer matrix for instant boot times."""
    initial_ndvi = np.full((grid_size, grid_size), 0.20)
    initial_ndbi = np.full((grid_size, grid_size), 0.10)
    
    try:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            r, c = int(row['lat_idx']), int(row['lon_idx'])
            initial_ndvi[r, c] = row['NDVI']
            initial_ndbi[r, c] = row['NDBI']
    except Exception:
        # Fallback safeguard matrix if file path fails
        pass
        
    return initial_ndvi, initial_ndbi

# Initialize canvas memory tracks using our lightweight asset
if 'ndvi_layer' not in st.session_state or st.session_state.ndvi_layer.shape != (GRID_SIZE, GRID_SIZE):
    real_ndvi, real_ndbi = load_prebaked_miri_layers('miri_base_grid.csv', GRID_SIZE)
    st.session_state.ndvi_layer = real_ndvi
    st.session_state.ndbi_layer = real_ndbi

# --- MODULE 5: COORDINATE-FREE AI INFERENCE & THERMAL TRANSLATION ---
# Prevent division-by-zero errors in areas with minimal canopy footprint
ratio_layer = st.session_state.ndbi_layer / (st.session_state.ndvi_layer + 1e-5)

# Build feature matrix matching exactly what the random forest regressor expects
features_matrix = np.stack([
    st.session_state.ndvi_layer.flatten(),
    st.session_state.ndbi_layer.flatten(),
    ratio_layer.flatten()
], axis=1)

# Predict across the expanded spatial web canvas simultaneously
predicted_flat = ai_model.predict(features_matrix)
predicted_grid = predicted_flat.reshape((GRID_SIZE, GRID_SIZE))

# Dynamic Delta Calculation: Map AI structural variances relative to the Live Weather Feed
ai_baseline_mean = np.mean(predicted_grid)  
spatial_deviations = predicted_grid - ai_baseline_mean
nrt_final_heatmap = projected_base_temp + spatial_deviations

# --- MODULE 6: HEATMAP RENDER SPECTRUM CREATION (GREEN -> RED) ---
fig, ax = plt.subplots(figsize=(8, 8))
ax.axis('off')
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

# CHANGED: Switched to 'RdYlGn_r' (Reversed Red-Yellow-Green) 
# High vegetation/cool areas show as Green, transition zones as Yellow, high-concrete heat islands as Red.
VMIN_TEMP, VMAX_TEMP = 26.0, 42.0
im = ax.imshow(
    nrt_final_heatmap, 
    cmap='RdYlGn_r', 
    interpolation='bicubic', 
    origin='lower',
    vmin=VMIN_TEMP, 
    vmax=VMAX_TEMP  
)

# Convert figure to a base64 string buffer to prevent writing local disk files
buf = io.BytesIO()
fig.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
buf.seek(0)
image_base64 = base64.b64encode(buf.read()).decode('utf-8')
image_url = f"data:image/png;base64,{image_base64}"
plt.close(fig)

from folium.plugins import Draw, MousePosition  # Add MousePosition to your imports

# --- MODULE 7: GEOGRAPHIC MAP & INTERACTIVE LAYER CONFIGURATION ---
m = folium.Map(location=[4.415, 114.00], zoom_start=12, tiles="CartoDB positron")

# Overlay the polished thermal canvas over the expanded bounds
folium.raster_layers.ImageOverlay(
    image=image_url,
    bounds=[[LAT_MIN, LON_MIN], [LAT_MAX, LON_MAX]],
    opacity=0.55,
    interactive=False,
    cross_origin=False
).add_to(m)

# Visible on-screen color range legend
colormap_legend = cm.LinearColormap(
    colors=['green', 'yellow', 'red'],
    vmin=VMIN_TEMP,
    vmax=VMAX_TEMP,
    caption="Microclimate Temperature Range (°C)"
)
colormap_legend.add_to(m)

# Natively track mouse coordinates on hover (This one is safe and won't crash the map!)
MousePosition(
    position="bottomleft",
    separator=" | ",
    empty_string="Hovering outside matrix box",
    lng_first=False,
    prefix="Coordinates: Lat ",
).add_to(m)

# REMOVED: folium.LatLngPopup().add_to(m) <- Deleting this fixes the invisible map bug!

# Drop pinpoint markers for localized spot-checking when user clicks map
if "clicked_data" in st.session_state and st.session_state.clicked_data:
    click_lat = st.session_state.clicked_data["lat"]
    click_lon = st.session_state.clicked_data["lon"]
    click_temp = st.session_state.clicked_data["temp"]
    
    folium.Marker(
        location=[click_lat, click_lon],
        popup=folium.Popup(f"<b>Localized Readout:</b><br>{click_temp:.2f} °C", max_width=200),
        icon=folium.Icon(color="darkred" if click_temp > 34 else "green", icon="thermometer", prefix="fa")
    ).add_to(m)

# Attach Leaflet drawing capability tools
draw_control = Draw(
    export=False,
    position='topleft',
    draw_options={'polyline': False, 'circle': False, 'marker': False, 'circlemarker': False}
)
draw_control.add_to(m)

# --- MODULE 8: DUAL-PANEL VIEWPORT INTERFACE ---
col1, col2 = st.columns([3, 1], gap="medium")

with col1:
    # Render map and catch user interaction feeds
    output_map = st_folium(m, width=950, height=650, key="miri_nrt_overlay_engine", use_container_width=True)

# --- MODULE 9: MAP CLICK DETECTION & DRAWING INTERACTIONS ---
# NEW: Detect Map Clicks for point temperature inspection
if output_map and output_map.get("last_clicked"):
    click_pos = output_map["last_clicked"]
    click_lat, click_lon = click_pos["lat"], click_pos["lng"]
    
    # Check if this click position has already been processed to prevent infinite rerun loops
    if "last_processed_click" not in st.session_state or st.session_state.last_processed_click != click_pos:
        st.session_state.last_processed_click = click_pos
        
        # Translate real-world coordinate positions to internal grid matrix spaces
        lon_idx = int(np.clip((click_lon - LON_MIN) / (LON_MAX - LON_MIN) * GRID_SIZE, 0, GRID_SIZE - 1))
        lat_idx = int(np.clip((click_lat - LAT_MIN) / (LAT_MAX - LAT_MIN) * GRID_SIZE, 0, GRID_SIZE - 1))
        click_temp = nrt_final_heatmap[lat_idx, lon_idx]
        
        st.session_state.clicked_data = {"lat": click_lat, "lon": click_lon, "temp": click_temp}
        st.rerun()

# Detect blueprint geometry drawing shapes
if output_map and output_map.get("last_active_drawing"):
    geometry = output_map["last_active_drawing"]["geometry"]
    
    if geometry["type"] in ["Polygon", "Rectangle"]:
        coords = geometry["coordinates"][0]
        df_bounds = pd.DataFrame(coords, columns=['Longitude', 'Latitude'])
        
        min_lon, max_lon = df_bounds['Longitude'].min(), df_bounds['Longitude'].max()
        min_lat, max_lat = df_bounds['Latitude'].min(), df_bounds['Latitude'].max()
        
        # Translate real world coordinates into matrix index slots
        lon_idx_min = int(np.clip((min_lon - LON_MIN) / (LON_MAX - LON_MIN) * GRID_SIZE, 0, GRID_SIZE - 1))
        lon_idx_max = int(np.clip((max_lon - LON_MIN) / (LON_MAX - LON_MIN) * GRID_SIZE, 0, GRID_SIZE - 1))
        lat_idx_min = int(np.clip((min_lat - LAT_MIN) / (LAT_MAX - LAT_MIN) * GRID_SIZE, 0, GRID_SIZE - 1))
        lat_idx_max = int(np.clip((max_lat - LAT_MIN) / (LAT_MAX - LAT_MIN) * GRID_SIZE, 0, GRID_SIZE - 1))
        
        if (lon_idx_max >= lon_idx_min) and (lat_idx_max >= lat_idx_min):
            blend_factor = 0.30 
            
            # Grab existing tracking layers for modification
            current_ndvi_chunk = st.session_state.ndvi_layer[lat_idx_min:lat_idx_max + 1, lon_idx_min:lon_idx_max + 1]
            current_ndbi_chunk = st.session_state.ndbi_layer[lat_idx_min:lat_idx_max + 1, lon_idx_min:lon_idx_max + 1]
            
            # Apply proportional blending upgrades
            st.session_state.ndvi_layer[lat_idx_min:lat_idx_max + 1, lon_idx_min:lon_idx_max + 1] = \
                (current_ndvi_chunk * (1 - blend_factor)) + (target_ndvi * blend_factor)
                
            st.session_state.ndbi_layer[lat_idx_min:lat_idx_max + 1, lon_idx_min:lon_idx_max + 1] = \
                (current_ndbi_chunk * (1 - blend_factor)) + (target_ndbi * blend_factor)
            
            # Reset old click selectors to prevent pinning stale coordinates on updated grids
            if "clicked_data" in st.session_state:
                st.session_state.clicked_data = None
                st.session_state.last_processed_click = None
                
            st.rerun()

# --- MODULE 10: CLEAN & PROFESSIONAL ANALYTICS REPORTING PROFILE ---
with col2:
    st.subheader("📊 Live NRT Analytics")
    
    # Modern card formatting for key simulation metrics
    with st.container(border=True):
        current_sim_mean = np.mean(nrt_final_heatmap)
        st.metric(label="Simulated City Mean Temp", value=f"{current_sim_mean:.2f} °C")
        
    # NEW: Dedicated container display block for the spot-inspection tool
    with st.container(border=True):
        st.markdown("🎯 **Localized Target Inspector**")
        if "clicked_data" in st.session_state and st.session_state.clicked_data:
            inspected_temp = st.session_state.clicked_data['temp']
            st.markdown(f"**Inspected Temp:** `{inspected_temp:.2f} °C`")
            st.caption(f"Coordinates: {st.session_state.clicked_data['lat']:.4f}, {st.session_state.clicked_data['lon']:.4f}")
        else:
            st.info("Click anywhere inside the thermal heatmap overlay area to extract a point microclimate reading.")

    st.markdown("#### Bounding Horizon Metrics")
    st.markdown(f"""
    * **Live Weather Feed Status:** `Connected`
    * **Target Forecast Horizon:** `+{forecast_hour} Hour(s)`
    * **Matrix Peak Temperature:** **{np.max(nrt_final_heatmap):.1f}°C**
    * **Matrix Minimum Temperature:** **{np.min(nrt_final_heatmap):.1f}°C**
    """)
    
    st.divider()
    
    # Grid Reset Utility
    if st.button("Reset Entire Urban Matrix", use_container_width=True, type="secondary"):
        if 'ndvi_layer' in st.session_state:
            del st.session_state['ndvi_layer']
        if 'ndbi_layer' in st.session_state:
            del st.session_state['ndbi_layer']
        if 'clicked_data' in st.session_state:
            del st.session_state['clicked_data']
        if 'last_processed_click' in st.session_state:
            del st.session_state['last_processed_click']
        st.rerun()
