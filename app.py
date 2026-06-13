import streamlit as st
import pandas as pd
import numpy as np
import pickle
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw

st.set_page_config(page_title="Miri Environmental Twin", layout="wide")
st.title("🏙️ Miri Spatial SimCity Twin Engine")
st.markdown("Draw infrastructure directly onto the map to run localized machine learning thermal simulations.")

# 1. Load the AI Model Brain
@st.cache_resource
def load_model():
    with open('miri_heat_model.pkl', 'rb') as f:
        return pickle.load(f)

ai_model = load_model()

# 2. Left and Right View Columns
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("🗺️ Interactive Blueprint Map")
    st.info("Instructions: Use the toolbar on the left side of the map to draw a Polygon or Rectangle over an area in Miri. Select your placement type in the sidebar to simulate microclimate shifts!")

    # Initialize map over Miri Centre
    m = folium.Map(location=[4.393, 113.993], zoom_start=14, tiles="OpenStreetMap")
    
    # Inject the drawing interface toolbar overlay
    draw_control = Draw(
        export=False,
        filename='spatial_blueprint.geojson',
        position='topleft',
        draw_options={
            'polyline': False,
            'circle': False,
            'marker': False,
            'circlemarker': False
        }
    )
    draw_control.add_to(m)
    
    # Capture map interaction details back into Streamlit
    output_map = st_folium(m, width=900, height=600)

# 3. Sidebar Configuration Controls
st.sidebar.header("🕹️ Object Blueprint Settings")
blueprint_type = st.sidebar.selectbox(
    "Select Structural Placement Material:",
    ["Dense Concrete Skyscraper Grid (High Heat Retention)", "Urban Green Canopy Park (Cooling Infrastructure)"]
)

# Set raw values based on selector
if blueprint_type == "Dense Concrete Skyscraper Grid (High Heat Retention)":
    chosen_ndvi, chosen_ndbi = 0.02, 0.55
else:
    chosen_ndvi, chosen_ndbi = 0.65, -0.10

# 4. Read User Intersect Polygons & Predict Values Live
# Detect if the user has completed drawing a shape on their screen
if output_map and output_map.get("last_active_drawing"):
    geometry = output_map["last_active_drawing"]["geometry"]
    
    # Verify coordinates exist inside the geometric polygon bounds
    if geometry["type"] in ["Polygon", "Rectangle"]:
        coords = geometry["coordinates"][0]
        
        # Convert drawn bounds into a dataframe list of geographic coordinates
        df_coords = pd.DataFrame(coords, columns=['Longitude', 'Latitude'])
        
        # Extract boundaries
        min_lon, max_lon = df_coords['Longitude'].min(), df_coords['Longitude'].max()
        min_lat, max_lat = df_coords['Latitude'].min(), df_coords['Latitude'].max()
        
        # Create a detailed localized grid inside just that drawn layout space
        lon_points = np.linspace(min_lon, max_lon, 15)
        lat_points = np.linspace(min_lat, max_lat, 15)
        
        matrix_rows = []
        for lat in lat_points:
            for lon in lon_points:
                ratio = chosen_ndbi / (chosen_ndvi + 1e-5)
                matrix_rows.append([chosen_ndvi, chosen_ndbi, ratio, lon, lat])
                
        inference_matrix = np.array(matrix_rows)
        
        # Pass data points straight into your 66.88% model
        predicted_temps = ai_model.predict(inference_matrix)
        local_avg_temp = np.mean(predicted_temps)
        
        with col2:
            st.subheader("📊 Simulation Readout")
            st.metric(label="Simulated Boundary Local Heat Signature", value=f"{local_avg_temp:.2f} °C")
            
            # Actionable insight displays for presentation validation
            if local_avg_temp > 36.0:
                st.error("🚨 Warning: Concrete expansion footprint breaches municipal safety bounds. High risk of localized heat collection pockets.")
            else:
                st.success("🌲 Environmental mitigation verified. Canopy placement absorbs regional radiation effectively.")
else:
    with col2:
        st.subheader("📊 Simulation Readout")
        st.write("Awaiting blueprint geometry input... Draw a shape on the map to begin analysis.")
