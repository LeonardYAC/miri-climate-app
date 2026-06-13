import streamlit as st
import pandas as pd
import numpy as np
import pickle
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw

st.set_page_config(page_title="Miri Environmental Twin", layout="wide")
st.title("🏙️ Miri Spatial SimCity Twin Engine")
st.markdown("Professional GIS Raster Simulation Framework for Municipality Thermal Optimization.")

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

# 3. Custom Microclimate Color Profiler Engine
def get_thermal_color(temp):
    """Maps custom temperature ranges directly into smooth hexadecimal visual gradients."""
    if temp < 28.0:
        return '#0571b0'  # Deep Blue (Cooling/Hydrological baseline)
    elif temp < 33.0:
        return '#4dac26'  # Emerald Green (Target Climate Resilient Zone)
    elif temp < 36.0:
        return '#fdb863'  # Amber Yellow/Orange (Moderate Built-up Heat)
    else:
        return '#e66101'  # Crimson Red (Critical Thermal Accumulation)

# 4. High-Resolution GIS Raster Grid Generator
@st.cache_data
def generate_miri_raster_matrix():
    # Defining exact coordinate bounds for the Miri municipality frame
    lat_min, lat_max = 4.32, 4.46
    lon_min, max_lon = 113.93, 114.05
    
    # Increase step density to create a seamless pixel-like matrix surface
    RESOLUTION_STEPS = 40
    lat_edges = np.linspace(lat_min, lat_max, RESOLUTION_STEPS)
    lon_edges = np.linspace(lon_min, max_lon, RESOLUTION_STEPS)
    
    matrix_cells = []
    inference_rows = []
    
    # Establish realistic baseline conditions: 25% concrete, 25% tree cover
    base_ndvi, base_ndbi = 0.25, 0.25
    base_ratio = base_ndbi / (base_ndvi + 1e-5)
    
    # Build continuous raster tile structures bounding coordinates together
    for i in range(len(lat_edges)-1):
        for j in range(len(lon_edges)-1):
            c_lat = (lat_edges[i] + lat_edges[i+1]) / 2.0
            c_lon = (lon_edges[j] + lon_edges[j+1]) / 2.0
            
            inference_rows.append([base_ndvi, base_ndbi, base_ratio, c_lon, c_lat])
            matrix_cells.append({
                'bounds': [[lat_edges[i], lon_edges[j]], [lat_edges[i+1], lon_edges[j+1]]],
                'lon': c_lon,
                'lat': c_lat,
                'is_modified': False
            })
            
    # Run rapid parallel predictions through your 66.88% model
    predicted_temps = ai_model.predict(np.array(inference_rows))
    for idx, temp in enumerate(predicted_temps):
        matrix_cells[idx]['temp'] = float(temp)
        
    return matrix_cells

# Initialize Miri's global continuous raster state into memory
if 'miri_raster_layer' not in st.session_state:
    st.session_state.miri_raster_layer = generate_miri_raster_matrix()

if 'active_modifications' not in st.session_state:
    st.session_state.active_modifications = []

# 5. Map Input Vector Assignment
# Strict alignment with your 66.88% AI model training order: [NDVI, NDBI]
if blueprint_type == "Dense Concrete Skyscraper Grid (High Heat Retention)":
    current_ndvi = 0.02   # Bare minimum foliage
    current_ndbi = 0.55   # Dense structural concrete score
else:
    current_ndvi = 0.65   # Dense cooling forest canopy
    current_ndbi = -0.05  # Negligible built footprint

current_ratio = current_ndbi / (current_ndvi + 1e-5)

# 6. Initialize Base Map Canvas
# Using a clean, minimal map layout style so your thermal colors pop beautifully
m = folium.Map(location=[4.393, 113.993], zoom_start=13, tiles="CartoDB positron")

# Attach the lightweight drawing controller to the map UI
draw_control = Draw(
    export=False,
    position='topleft',
    draw_options={'polyline': False, 'circle': False, 'marker': False, 'circlemarker': False}
)
draw_control.add_to(m)

# 7. Spatial Blueprint Intersection Engine
# If the user draws a custom shape, identify matching coordinates and override them live
if st.session_state.active_modifications:
    override_rows = []
    override_indices = []
    
    for idx, cell in enumerate(st.session_state.miri_raster_layer):
        cell_lon = cell['lon']
        cell_lat = cell['lat']
        
        for mod in st.session_state.active_modifications:
            min_lon, max_lon, min_lat, max_lat = mod['bounds']
            
            # Check if this specific raster pixel falls within any drawn blueprint box
            if (cell_lon >= min_lon) and (cell_lon <= max_lon) and (cell_lat >= min_lat) and (cell_lat <= max_lat):
                override_rows.append([current_ndvi, current_ndbi, current_ratio, cell_lon, cell_lat])
                override_indices.append(idx)
                break # Avoid double modifying if boxes happen to overlap
                
    # Run batch predictions simultaneously to maintain lightning-fast response speeds
    if override_rows:
        new_predictions = ai_model.predict(np.array(override_rows))
        for i, p_temp in enumerate(new_predictions):
            target_idx = override_indices[i]
            st.session_state.miri_raster_layer[target_idx]['temp'] = float(p_temp)
            st.session_state.miri_raster_layer[target_idx]['is_modified'] = True

# 8. RENDER MASTER LAYER: Build the seamless visual raster map surface
for cell in st.session_state.miri_raster_layer:
    cell_bounds = cell['bounds']
    cell_temp = cell['temp']
    cell_color = get_thermal_color(cell_temp)
    
    # Setting weight=0 is the magic key—it removes borders completely.
    # The adjacent rectangles merge seamlessly to look like a continuous weather map layer.
    folium.Rectangle(
        bounds=cell_bounds,
        color=cell_color,
        fill=True,
        fill_color=cell_color,
        fill_opacity=0.50 if not cell['is_modified'] else 0.80, # Bold opacity highlight for user modifications
        weight=0, 
        popup=f"Simulated Temp: {cell_temp:.1f}°C"
    ).add_to(m)

# 9. Render App Layout Split Frame Windows
col1, col2 = st.columns([3, 1])

with col1:
    # Display the integrated live map frame onto the screen
    output_map = st_folium(m, width=900, height=600, key="miri_highres_raster_map")

# 10. Intercept Drawing Inputs from the Frontend Map Interface
if output_map and output_map.get("last_active_drawing"):
    geometry = output_map["last_active_drawing"]["geometry"]
    
    if geometry["type"] in ["Polygon", "Rectangle"]:
        coords = geometry["coordinates"][0]
        df_bounds = pd.DataFrame(coords, columns=['Longitude', 'Latitude'])
        
        # Extract geographic bounding coordinates of the drawn footprint
        drawn_bounds = [
            float(df_bounds['Longitude'].min()),
            float(df_bounds['Longitude'].max()),
            float(df_bounds['Latitude'].min()),
            float(df_bounds['Latitude'].max())
        ]
        
        # Verify it's a new drawing bounds block to prevent infinite calculation rerun loops
        if not st.session_state.active_modifications or st.session_state.active_modifications[-1]['bounds'] != drawn_bounds:
            st.session_state.active_modifications.append({'bounds': drawn_bounds})
            st.rerun()

# 11. Render Sidebar Statistics Metrics Panel Module (Analytics Dashboard)
with col2:
    st.subheader("📊 Climate Analytics")
    
    # Calculate current real-time city averages across the modified raster matrix
    all_current_temps = [cell['temp'] for cell in st.session_state.miri_raster_layer]
    city_avg_temp = np.mean(all_current_temps)
    
    st.metric(label="Miri Municipality Mean Temp", value=f"{city_avg_temp:.2f} °C")
    
    st.markdown("---")
    st.markdown("*Infrastructure Framework Advisory:*")
    
    if city_avg_temp > 35.5:
        st.error("🚨 *CRITICAL OVERHEAT ZONE*\n\nConcrete layers have pushed regional grids past thermal safety bounds. Plant trees immediately to stabilize microclimate pockets.")
    elif city_avg_temp > 32.5:
        st.warning("⚠️ *WARMING GRADIENT OBSERVED*\n\nModerate thermal tracking profile. Recommend introducing reflective surfaces or green balconies.")
    else:
        st.success("🌲 *CLIMATE RESILIENT MATRIX*\n\nVegetation distribution index successfully absorbs incoming solar radiation, keeping Miri safe.")
        
    st.markdown("---")
    # Quick clear action button to completely restore modifications back to baseline state
    if st.button("Reset City Blueprint Landscape", use_container_width=True):
        st.session_state.active_modifications = []
        if 'miri_raster_layer' in st.session_state:
            del st.session_state['miri_raster_layer'] # Drops the modified layer from cache memory
        st.rerun()
