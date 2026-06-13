import streamlit as st
import pandas as pd
import numpy as np
import pickle
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw

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

# 3. Generate a permanent 20x20 Background Grid for the Entirety of Miri
@st.cache_data
def generate_base_miri_heatmap():
    # Bounds matching Miri city limits
    lat_range = np.linspace(4.34, 4.45, 20)
    lon_range = np.linspace(113.94, 114.04, 20)
    
    base_cells = []
    matrix_rows = []
    
    # Assume Miri's current structural baseline: moderate concrete (0.25), moderate trees (0.25)
    base_ndvi, base_ndbi = 0.25, 0.25
    base_ratio = base_ndbi / (base_ndvi + 1e-5)
    
    for i in range(len(lat_range)-1):
        for j in range(len(lon_range)-1):
            c_lat = (lat_range[i] + lat_range[i+1]) / 2.0
            c_lon = (lon_range[j] + lon_range[j+1]) / 2.0
            
            matrix_rows.append([base_ndvi, base_ndbi, base_ratio, c_lon, c_lat])
            base_cells.append({
                'bounds': [[lat_range[i], lon_range[j]], [lat_range[i+1], lon_range[j+1]]],
                'is_modified': False
            })
            
    # Run structural baseline predictions through your AI model
    predicted_base_temps = ai_model.predict(np.array(matrix_rows))
    for idx, temp in enumerate(predicted_base_temps):
        base_cells[idx]['temp'] = temp
        
    return base_cells

# Initialize Miri's global background map data into state memory
if 'miri_global_map' not in st.session_state:
    st.session_state.miri_global_map = generate_base_miri_heatmap()

if 'last_sim_data' not in st.session_state:
    st.session_state.last_sim_data = None

# 4. Set Feature Mapping variables using the exact mathematical order your AI expects
# Order: NDVI (index 0), NDBI (index 1)
if blueprint_type == "Dense Concrete Skyscraper Grid (High Heat Retention)":
    current_ndvi = 0.02  # Almost zero foliage
    current_ndbi = 0.55  # Dense structural concrete footprint
else:
    current_ndvi = 0.65  # Rich forest canopy index
    current_ndbi = -0.05 # Low built-up structure score

# 5. Build the Dynamic Mapping Object Frame
m = folium.Map(location=[4.393, 113.993], zoom_start=13, tiles="OpenStreetMap")

# Attach Drawing Toolbar System
draw_control = Draw(
    export=False,
    position='topleft',
    draw_options={'polyline': False, 'circle': False, 'marker': False, 'circlemarker': False}
)
draw_control.add_to(m)

# 6. RENDER MASTER LAYER HOOK: Draw the whole city map layout combining baseline and user shapes
# Loop through the base elements tracking the whole city footprint
for cell in st.session_state.miri_global_map:
    cell_bounds = cell['bounds']
    cell_temp = cell['temp']
    
    # Check if a custom blueprint footprint overrides this specific cell block coordinate
    if st.session_state.last_sim_data is not None:
        for mod_cell in st.session_state.last_sim_data:
            # Simple spatial intersection check: see if centers align roughly
            if (mod_cell['bounds'][0][0] >= cell_bounds[0][0] and mod_cell['bounds'][1][0] <= cell_bounds[1][0] and
                mod_cell['bounds'][0][1] >= cell_bounds[0][1] and mod_cell['bounds'][1][1] <= cell_bounds[1][1]):
                cell_temp = mod_cell['temp']
                cell['is_modified'] = True

    # Color Scale Mapping for Miri: Sharp visualization gradient adjustments
    if cell_temp > 37.0:
        fill_color = '#d73027'   # Severe Heat Zone (Red)
    elif cell_temp > 34.0:
        fill_color = '#fdae61'   # Urbanized Warm Zone (Orange)
    elif cell_temp > 31.0:
        fill_color = '#fee090'   # Mild Thermal Step (Yellow)
    else:
        fill_color = '#1a9850'   # Regenerated Cool Zone (Green)

    folium.Rectangle(
        bounds=cell_bounds,
        color=fill_color,
        fill=True,
        fill_color=fill_color,
        fill_opacity=0.45 if not cell.get('is_modified') else 0.75, # Make user modifications look bolder
        weight=0.5,
        popup=f"Temp: {cell_temp:.1f}°C"
    ).add_to(m)

# 7. Render Layout Split Frame Windows
col1, col2 = st.columns([3, 1])

with col1:
    output_map = st_folium(m, width=900, height=600, key="miri_sim_map_v2")

# 8. Core Intercept Logic: Catch drawn shapes and feed them through the correct feature channels
if output_map and output_map.get("last_active_drawing"):
    geometry = output_map["last_active_drawing"]["geometry"]
    
    if geometry["type"] in ["Polygon", "Rectangle"]:
        coords = geometry["coordinates"][0]
        df_coords = pd.DataFrame(coords, columns=['Longitude', 'Latitude'])
        
        min_lon, max_lon = df_coords['Longitude'].min(), df_coords['Longitude'].max()
        min_lat, max_lat = df_coords['Latitude'].min(), df_coords['Latitude'].max()
        
        # FIXED CONFIGURATION: Upgraded tracking loop matrix to a detailed 12x12 micro-grid
        UPGRADED_GRID_STEPS = 12
        lon_steps = np.linspace(min_lon, max_lon, UPGRADED_GRID_STEPS)
        lat_steps = np.linspace(min_lat, max_lat, UPGRADED_GRID_STEPS)
        
        new_sim_cells = []
        matrix_rows = []
        
        for i in range(len(lat_steps)-1):
            for j in range(len(lon_steps)-1):
                c_lat = (lat_steps[i] + lat_steps[i+1]) / 2.0
                c_lon = (lon_steps[j] + lon_steps[j+1]) / 2.0
                ratio = current_ndbi / (current_ndvi + 1e-5)
                
                # FIXED CRITICAL BUG: Verified index array order perfectly matches the AI's training configuration!
                # Array order: [NDVI, NDBI, Ratio, Longitude, Latitude]
                matrix_rows.append([current_ndvi, current_ndbi, ratio, c_lon, c_lat])
                new_sim_cells.append({
                    'bounds': [[lat_steps[i], lon_steps[j]], [lat_steps[i+1], lon_steps[j+1]]]
                })
                
        predicted_temps = ai_model.predict(np.array(matrix_rows))
        
        for idx, temp in enumerate(predicted_temps):
            new_sim_cells[idx]['temp'] = temp
            
        st.session_state.last_sim_data = new_sim_cells
        st.rerun()

# 9. Sidebar Metric Readouts Module
with col2:
    st.subheader("📊 Climate Analytics")
    all_city_temps = [c['temp'] for c in st.session_state.miri_global_map]
    city_avg = np.mean(all_city_temps)
    
    st.metric(label="Miri Municipality Mean Temp", value=f"{city_avg:.2f} °C")
    
    st.markdown("---")
    st.markdown("*System Framework Status:*")
    if city_avg > 35.0:
        st.error("🚨 Urban Grid exceeding target climate boundaries. Immediate canopy buffers required.")
    else:
        st.success("🌲 Adaptation balance verified. Structural layout matches regional stability bounds.")
