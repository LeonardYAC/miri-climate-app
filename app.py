import streamlit as st
import pandas as pd
import numpy as np
import pickle
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw, HeatMap

st.set_page_config(page_title="Miri Environmental Twin", layout="wide")
st.title("🏙️ Miri Spatial SimCity Twin Engine")
st.markdown("Draw infrastructure blueprints onto the map. The AI will instantly overlay the predictive thermal microclimate layout over Miri's baseline footprint.")

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

# 3. High-Performance Macro Baseline Generator (The Whole Map of Miri)
@st.cache_data
def generate_base_miri_heatmap():
    # Create a dense mesh of geographic points over the entirety of Miri
    lat_points = np.linspace(4.32, 4.46, 35)
    lon_points = np.linspace(113.93, 114.05, 35)
    
    matrix_rows = []
    # Moderate baseline urban metrics: 25% concrete, 25% green canopy
    base_ndvi, base_ndbi = 0.25, 0.25
    base_ratio = base_ndbi / (base_ndvi + 1e-5)
    
    for lat in lat_points:
        for lon in lon_points:
            matrix_rows.append([base_ndvi, base_ndbi, base_ratio, lon, lat])
            
    # Run rapid parallel predictions through the Random Forest model
    predicted_temps = ai_model.predict(np.array(matrix_rows))
    
    # Construct a lightweight, fast pandas dataframe for the mapping layer
    raw_data = []
    idx = 0
    for lat in lat_points:
        for lon in lon_points:
            raw_data.append({
                'lat': lat,
                'lon': lon,
                'temp': float(predicted_temps[idx])
            })
            idx += 1
            
    return pd.DataFrame(raw_data)

# Initialize global state variables to store calculations across interface refreshes
if 'miri_global_df' not in st.session_state:
    st.session_state.miri_global_df = generate_base_miri_heatmap()

if 'active_modifications' not in st.session_state:
    st.session_state.active_modifications = []

# 4. Set Feature Mapping variables using the exact mathematical order your AI expects
# Correct AI training feature order: [NDVI, NDBI, Ratio, Longitude, Latitude]
if blueprint_type == "Dense Concrete Skyscraper Grid (High Heat Retention)":
    current_ndvi = 0.02  # Low foliage
    current_ndbi = 0.55  # High concrete index
else:
    current_ndvi = 0.65  # Rich vegetation canopy
    current_ndbi = -0.05 # Low concrete score

# Calculate interaction feature for the AI model
current_ratio = current_ndbi / (current_ndvi + 1e-5)

# 5. Build a clean Base Map centered over Miri City Centre
m = folium.Map(location=[4.393, 113.993], zoom_start=13, tiles="CartoDB positron")

# Inject Toolbar controls
draw_control = Draw(
    export=False,
    position='topleft',
    draw_options={'polyline': False, 'circle': False, 'marker': False, 'circlemarker': False}
)
draw_control.add_to(m)

# 6. Deep-Copy Baseline DataFrame to inject real-time changes
working_df = st.session_state.miri_global_df.copy()

# If the user has drawn custom modifications, override those coordinate values in our tracking dataframe
if st.session_state.active_modifications:
    for mod in st.session_state.active_modifications:
        min_lon, max_lon, min_lat, max_lat = mod['bounds']
        
        # Identify all points in the city dataframe that fall inside the user's custom drawn blueprint boundary
        mask = (
            (working_df['lon'] >= min_lon) & (working_df['lon'] <= max_lon) &
            (working_df['lat'] >= min_lat) & (working_df['lat'] <= max_lat)
        )
        
        # If any matching points are found, run the new conditions through the AI model
        if working_df[mask].shape[0] > 0:
            override_rows = []
            for _, row in working_df[mask].iterrows():
                # Correct feature lineup: [NDVI, NDBI, Ratio, Longitude, Latitude]
                override_rows.append([current_ndvi, current_ndbi, current_ratio, row['lon'], row['lat']])
            
            # Predict the modified temperature values instantly
            new_predictions = ai_model.predict(np.array(override_rows))
            working_df.loc[mask, 'temp'] = new_predictions

# 7. HIGH-PERFORMANCE MAP RENDER HOOK: Render a smooth, un-grid-like geographical heatmap
# We scale the intensity value (temp - 22) so that heat signatures pop vividly on the screen
heat_data = [[row['lat'], row['lon'], float(row['temp'] - 22)] for _, row in working_df.iterrows()]

HeatMap(
    heat_data,
    radius=30,
    blur=18,
    max_zoom=13,
    gradient={0.3: '#1a9850', 0.55: '#fee090', 0.75: '#fdae61', 1.0: '#d73027'} # Smooth Green -> Yellow -> Red gradient
).add_to(m)

# 8. Render App Layout Windows Split
col1, col2 = st.columns([3, 1])

with col1:
    output_map = st_folium(m, width=900, height=600, key="miri_highres_sim_map")

# 9. Catch drawn blueprint coordinates and feed them to the system live
if output_map and output_map.get("last_active_drawing"):
    geometry = output_map["last_active_drawing"]["geometry"]
    
    if geometry["type"] in ["Polygon", "Rectangle"]:
        coords = geometry["coordinates"][0]
        df_bounds = pd.DataFrame(coords, columns=['Longitude', 'Latitude'])
        
        # Define the box parameters
        drawn_bounds = [
            df_bounds['Longitude'].min(),
            df_bounds['Longitude'].max(),
            df_bounds['Latitude'].min(),
            df_bounds['Latitude'].max()
        ]
        
        # Add this footprint area modification into state history memory
        st.session_state.active_modifications.append({'bounds': drawn_bounds})
        st.rerun()

# 10. Render Sidebar Statistics Metrics Panel Module
with col2:
    st.subheader("📊 Climate Analytics")
    city_avg_temp = working_df['temp'].mean()
    
    st.metric(label="Miri Municipality Mean Temp", value=f"{city_avg_temp:.2f} °C")
    
    st.markdown("---")
    st.markdown("*Infrastructure Framework Advisory:*")
    if city_avg_temp > 35.5:
        st.error("🚨 *CRITICAL OVERHEAT ZONE*\n\nConcrete levels have pushed regional grids past thermal safety bounds. Plant trees immediately to stabilize microclimate pockets.")
    elif city_avg_temp > 32.5:
        st.warning("⚠️ *WARMING GRADIENT OBSERVED*\n\nModerate thermal tracking profile. Recommend introducing reflective surfaces or green balconies.")
    else:
        st.success("🌲 *CLIMATE RESILIENT MATRIX*\n\nVegetation distribution index successfully absorbs incoming solar radiation, keeping Miri safe.")
        
    # Quick clear button to completely reset modifications
    if st.button("Reset City Blueprint Landscape"):
        st.session_state.active_modifications = []
        st.rerun()
