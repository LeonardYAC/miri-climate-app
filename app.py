import streamlit as st
import pandas as pd
import numpy as np
import pickle
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw

st.set_page_config(page_title="Miri Environmental Twin", layout="wide")
st.title("🏙️ Miri Spatial SimCity Twin Engine")
st.markdown("Draw infrastructure blueprints onto the map. The AI will instantly overlay the predictive thermal microclimate layout.")

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

if blueprint_type == "Dense Concrete Skyscraper Grid (High Heat Retention)":
    chosen_ndvi, chosen_ndbi = 0.02, 0.55
else:
    chosen_ndvi, chosen_ndbi = 0.65, -0.10

# Initialize session state to save calculations between screen refreshes
if 'last_sim_data' not in st.session_state:
    st.session_state.last_sim_data = None

# 3. Create the Base Folium Map Object
m = folium.Map(location=[4.393, 113.993], zoom_start=14, tiles="OpenStreetMap")

# Attach drawing capabilities to the map
draw_control = Draw(
    export=False,
    position='topleft',
    draw_options={
        'polyline': False, 'circle': False, 'marker': False, 'circlemarker': False
    }
)
draw_control.add_to(m)

# 4. If a calculation was previously made, render the colored thermal overlay BEFORE displaying the map
if st.session_state.last_sim_data is not None:
    sim_cells = st.session_state.last_sim_data
    for cell in sim_cells:
        temp = cell['temp']
        
        # Select appropriate visualization color mapping based on model thermal temperature
        if temp > 36.0:
            cell_color = '#d73027' # Hot Crimson Red
        elif temp > 32.0:
            cell_color = '#fdae61' # Warn Orange
        else:
            cell_color = '#1a9850' # Cool Dense Green
            
        # Draw a solid, semi-transparent colored micro-square block directly over that specific coordinate point
        folium.Rectangle(
            bounds=cell['bounds'],
            color=cell_color,
            fill=True,
            fill_color=cell_color,
            fill_opacity=0.6,
            weight=1,
            popup=f"AI Prediction: {temp:.2f}°C"
        ).add_to(m)

# 5. Display Layout Windows Split
col1, col2 = st.columns([3, 1])

with col1:
    # Render the interactive map workspace onto the browser page
    output_map = st_folium(m, width=900, height=600, key="miri_sim_map")

# 6. Intercept user drawing movements live
if output_map and output_map.get("last_active_drawing"):
    geometry = output_map["last_active_drawing"]["geometry"]
    
    if geometry["type"] in ["Polygon", "Rectangle"]:
        coords = geometry["coordinates"][0]
        df_coords = pd.DataFrame(coords, columns=['Longitude', 'Latitude'])
        
        min_lon, max_lon = df_coords['Longitude'].min(), df_coords['Longitude'].max()
        min_lat, max_lat = df_coords['Latitude'].min(), df_coords['Latitude'].max()
        
        # Formulate a granular 6x6 predictive patch grid inside the user's custom shape boundaries
        GRID_STEPS = 6
        lon_steps = np.linspace(min_lon, max_lon, GRID_STEPS)
        lat_steps = np.linspace(min_lat, max_lat, GRID_STEPS)
        
        new_sim_cells = []
        matrix_rows = []
        
        # Calculate bounding squares for each tiny pixel block inside the drawn shape
        for i in range(len(lat_steps)-1):
            for j in range(len(lon_steps)-1):
                c_lat = (lat_steps[i] + lat_steps[i+1]) / 2.0
                c_lon = (lon_steps[j] + lon_steps[j+1]) / 2.0
                ratio = chosen_ndbi / (chosen_ndvi + 1e-5)
                
                matrix_rows.append([chosen_ndvi, chosen_ndbi, ratio, c_lon, c_lat])
                new_sim_cells.append({
                    'bounds': [[lat_steps[i], lon_steps[j]], [lat_steps[i+1], lon_steps[j+1]]]
                })
                
        # Fire array predictions directly into your 66.88% Random Forest AI
        predicted_temps = ai_model.predict(np.array(matrix_rows))
        
        # Match temperatures to their respective geographic grid blocks
        for idx, temp in enumerate(predicted_temps):
            new_sim_cells[idx]['temp'] = temp
            
        # Store state and force page refresh to trigger step 4 layer injection mapping
        st.session_state.last_sim_data = new_sim_cells
        st.rerun()

# 7. Render Sidebar Metric readouts
with col2:
    st.subheader("📊 Simulation Readout")
    if st.session_state.last_sim_data is not None:
        all_temps = [c['temp'] for c in st.session_state.last_sim_data]
        local_avg_temp = np.mean(all_temps)
        
        st.metric(label="Simulated Micro-Zone Heat", value=f"{local_avg_temp:.2f} °C")
        if local_avg_temp > 35.5:
            st.error("🚨 *CRITICAL HEAT ISLAND RISK*\n\nMaterial placement creates high heat retention. Localized temperatures spike.")
        else:
            st.success("🌲 *THERMAL ADAPTATION ACHIEVED*\n\nVegetation structure successfully lowers baseline surface radiation absorption.")
    else:
        st.write("Awaiting design coordinates... Use the map drawing toolbar on the left to draw a shape over Miri.")
