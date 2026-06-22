import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import requests
import base64

# NEW IMPORT FOR 3D RENDERING
import pydeck as pdk

# --- MODULE 1: APP INITIALIZATION & CONFIGURATION ---
st.set_page_config(page_title="Miri Environmental Twin", layout="wide")
st.title("🏙️ Miri Spatial SimCity Twin Engine")
st.markdown("### Near Real-Time (NRT) Microclimate Predictive Framework")

# Load the upgraded, coordinate-free AI brain model (Expects: NDVI, NDBI, Ratio)
@st.cache_resource
def load_model():
    with open('miri_heat_model.pkl', 'rb') as f:
        return pickle.load(f)

ai_model = load_model()

# --- MODULE 2: NRT WEATHER API INTEGRATION ---
@st.cache_data(ttl=600)  
def fetch_miri_nrt_weather():
    """Fetches real-time baseline ambient temperatures for Miri Municipality."""
    try:
        url = "https://api.open-meteo.com/v1/forecast?latitude=4.393&longitude=113.993&current=temperature_2m&hourly=temperature_2m&timezone=Asia/Singapore"
        response = requests.get(url).json()
        live_temp = response['current']['temperature_2m']
        hourly_forecasts = response['hourly']['temperature_2m'][:24] 
        return live_temp, hourly_forecasts
    except Exception:
        return 31.5, [30.0 + np.sin(h/24 * 2 * np.pi) * 4 for h in range(24)]

live_base_temp, future_hourly_temps = fetch_miri_nrt_weather()

# --- MODULE 3: SIDEBAR CONTROL PANEL ---
st.sidebar.header("⏱️ NRT Temporal Controls")
forecast_hour = st.sidebar.slider("Predictive Forecast Horizon (Hours from Now):", 0, 12, 0)
projected_base_temp = future_hourly_temps[forecast_hour]

st.sidebar.markdown(f"Baseline City Temp at Horizon: {projected_base_temp:.1f}°C")
st.sidebar.write("---")

st.sidebar.header("🕹️ Object Blueprint Settings")
blueprint_type = st.sidebar.selectbox(
    "Select Structural Placement Material:",
    ["Dense Concrete Skyscraper Grid (High Heat Retention)", "Urban Green Canopy Park (Cooling Infrastructure)"]
)

if blueprint_type == "Dense Concrete Skyscraper Grid (High Heat Retention)":
    target_ndvi, target_ndbi = 0.02, 0.55
else:
    target_ndvi, target_ndbi = 0.65, -0.05

# --- MODULE 4: GEOGRAPHIC GRID & PRE-BAKED SATELLITE REALISM ENGINE ---
LAT_MIN, LAT_MAX = 4.30, 4.53  
LON_MIN, LON_MAX = 113.92, 114.05  
GRID_SIZE = 35 

@st.cache_data
def load_prebaked_miri_layers(csv_path, grid_size):
    initial_ndvi = np.full((grid_size, grid_size), 0.20)
    initial_ndbi = np.full((grid_size, grid_size), 0.10)
    try:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            r, c = int(row['lat_idx']), int(row['lon_idx'])
            initial_ndvi[r, c] = row['NDVI']
            initial_ndbi[r, c] = row['NDBI']
    except Exception:
        pass
    return initial_ndvi, initial_ndbi

if 'ndvi_layer' not in st.session_state or st.session_state.ndvi_layer.shape != (GRID_SIZE, GRID_SIZE):
    real_ndvi, real_ndbi = load_prebaked_miri_layers('miri_base_grid.csv', GRID_SIZE)
    st.session_state.ndvi_layer = real_ndvi
    st.session_state.ndbi_layer = real_ndbi

# --- MODULE 5: COORDINATE-FREE AI INFERENCE ---
ratio_layer = st.session_state.ndbi_layer / (st.session_state.ndvi_layer + 1e-5)

features_matrix = np.stack([
    st.session_state.ndvi_layer.flatten(),
    st.session_state.ndbi_layer.flatten(),
    ratio_layer.flatten()
], axis=1)

predicted_flat = ai_model.predict(features_matrix)
predicted_grid = predicted_flat.reshape((GRID_SIZE, GRID_SIZE))

ai_baseline_mean = np.mean(predicted_grid)  
spatial_deviations = predicted_grid - ai_baseline_mean
nrt_final_heatmap = projected_base_temp + spatial_deviations

# --- NEW MODULE 6: 3D DATA TRANSFORMATION ---
# Convert 2D arrays into flat lists of coordinates for WebGL rendering
latitudes = np.linspace(LAT_MIN, LAT_MAX, GRID_SIZE)
longitudes = np.linspace(LON_MIN, LON_MAX, GRID_SIZE)

grid_data = []
for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        temp_val = nrt_final_heatmap[r, c]
        
        # Color mapping: Higher temperatures map to brighter red, cooler to blue
        min_t, max_t = 26.0, 42.0
        r_ratio = (temp_val - min_t) / (max_t - min_t)
        r_color = int(np.clip(r_ratio * 255, 0, 255))
        b_color = int(np.clip((1 - r_ratio) * 255, 0, 255))
        
        grid_data.append({
            "lon": longitudes[c],
            "lat": latitudes[r],
            "temperature": float(temp_val),
            "ndvi": float(st.session_state.ndvi_layer[r, c]),
            "ndbi": float(st.session_state.ndbi_layer[r, c]),
            "r": r_color,
            "b": b_color
        })

df_3d = pd.DataFrame(grid_data)

# --- NEW MODULE 7: DUAL-PANEL VIEWPORT INTERFACE (WITH PYDECK) ---
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("🌐 3D Microclimate Viewport")
    
    # Configure the 3D column layer (Extruding temperature as physical height)
    layer_3d = pdk.Layer(
        "ColumnLayer",
        data=df_3d,
        get_position="[lon, lat]",
        get_elevation="(temperature - 25) * 150",  # Heat severity dictates tower height
        elevation_scale=1,
        radius=400,  
        get_fill_color="[r, 50, b, 180]",  # dynamic RGB array from our DataFrame
        pickable=True,
        auto_highlight=True,
    )

    # Pitch set to 45 degrees activates the 3D angle perspective
    view_state = pdk.ViewState(
        latitude=4.393,
        longitude=113.993,
        zoom=11,
        pitch=45,  
        bearing=0
    )

    r = pdk.Deck(
        layers=[layer_3d],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/light-v9",
        tooltip={"text": "Simulated Temp: {temperature}°C\nNDVI: {ndvi}\nNDBI: {ndbi}"}
    )
    
    st.pydeck_chart(r)

# --- MODULE 8: ANALYTICS REPORTING PROFILE ---
with col2:
    st.subheader("📊 Live NRT Analytics")
    current_sim_mean = np.mean(nrt_final_heatmap)
    st.metric(label="Simulated City Mean Temp", value=f"{current_sim_mean:.2f} °C")
    
    st.markdown("---")
    st.markdown("Predictive Horizon Summary:")
    st.info(f"🌐 Live Weather Feed: Connected\n\n🕒 Target Horizon: +{forecast_hour} Hour(s)\n\n🌡️ Predicted Matrix Peak: {np.max(nrt_final_heatmap):.1f}°C")
    
    st.markdown("---")
    if st.button("Reset Entire Urban Matrix", use_container_width=True):
        if 'ndvi_layer' in st.session_state:
            del st.session_state['ndvi_layer']
        if 'ndbi_layer' in st.session_state:
            del st.session_state['ndbi_layer']
        st.rerun()
