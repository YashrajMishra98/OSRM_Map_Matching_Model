# app.py

# --- Section 1: Import Libraries ---
import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import osmnx as ox
import matplotlib.pyplot as plt
import joblib 
import os
import random
import math
from shapely.geometry import Point, LineString
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score
from streamlit_js_eval import get_geolocation # Library for Live GPS

# --- Section 2: Configuration & Directories ---

# Define directories
base_data_dir = "data"
model_path = os.path.join(base_data_dir, "model", "road_type_classifier_model.joblib")

# Ensure directories exist (optional, mostly for training script, but good practice)
os.makedirs(os.path.join(base_data_dir, "model"), exist_ok=True)

# --- Section 3: Helper Functions ---

@st.cache_resource
def load_ml_model(model_file_path):
    """Loads the pre-trained ML model."""
    try:
        model = joblib.load(model_file_path)
        return model
    except Exception as e:
        return None

@st.cache_resource
def load_map_data(place_name):
    """Downloads map data for the specific user-selected place."""
    try:
        # Download graph
        G_all = ox.graph_from_place(place_name, network_type="all")
        edges = ox.graph_to_gdfs(G_all, nodes=False, edges=True)

        # Filter Highways
        highway_types = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 
                         'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link',
                         'road', 'unclassified', 'residential']
        gdf_highways = edges[edges['highway'].apply(lambda x: any(ht in str(x) for ht in highway_types))]
        gdf_highways_geo = gdf_highways.to_crs(epsg=4326)

        # Filter Service Roads
        service_road_types = ['service', 'residential', 'unclassified', 'track', 'path', 'pedestrian', 'cycleway']
        gdf_service_roads = edges[edges['highway'].apply(lambda x: any(srt in str(x) for srt in service_road_types))]
        gdf_service_roads_geo = gdf_service_roads.to_crs(epsg=4326)

        return G_all, gdf_highways_geo, gdf_service_roads_geo
    except Exception as e:
        st.error(f"Error finding place '{place_name}'. Check spelling or internet connection.")
        return None, None, None

def generate_noisy_point(lat, lon, noise_meters=10):
    """Adds random noise to lat/lon coordinates."""
    lat_noise = (random.uniform(-1, 1) * noise_meters) / 111320
    lon_noise = (random.uniform(-1, 1) * noise_meters) / (111320 * np.cos(np.radians(lat)))
    return lat + lat_noise, lon + lon_noise

def create_simulated_trajectory(num_points, noise_meters, road_label, graph):
    """Generates a simulated trajectory along a random road segment."""
    if not graph or not graph.nodes:
        return pd.DataFrame()

    # Define target types based on label
    if road_label == "Highway":
        targets = ['trunk', 'primary', 'secondary', 'tertiary', 'trunk_link']
    else: # Service Road
        targets = ['residential', 'service', 'track', 'path', 'unclassified']

    selected_segment_geometry = None
    
    # Retry loop to find a valid edge
    for _ in range(100):
        try:
            u, v, key = random.choice(list(graph.edges(keys=True)))
            data = graph.edges[u, v, key]
            
            # Check if edge matches requested type
            if 'highway' in data:
                h_tags = data['highway']
                if isinstance(h_tags, list):
                    match = any(t in targets for t in h_tags)
                else:
                    match = h_tags in targets
                
                if match:
                    if 'geometry' in data:
                        selected_segment_geometry = data['geometry']
                    else:
                        selected_segment_geometry = LineString([
                            (graph.nodes[u]['x'], graph.nodes[u]['y']),
                            (graph.nodes[v]['x'], graph.nodes[v]['y'])
                        ])
                    break
        except:
            continue

    if selected_segment_geometry is None:
        return pd.DataFrame()

    # Interpolate points
    points_on_segment = []
    for i in range(num_points):
        fraction = i / (num_points - 1)
        point = selected_segment_geometry.interpolate(fraction, normalized=True)
        points_on_segment.append((point.y, point.x))

    # Add noise and kinematics
    trajectory_data = []
    start_time = datetime.now()
    
    for i, (lat, lon) in enumerate(points_on_segment):
        noisy_lat, noisy_lon = generate_noisy_point(lat, lon, noise_meters)
        timestamp = start_time + timedelta(seconds=i * 2)
        
        # Fake speed based on type
        speed = random.uniform(60, 100) if road_label == "Highway" else random.uniform(20, 40)
        
        # Calculate heading
        heading = 0
        if i < len(points_on_segment) - 1:
            next_lat, next_lon = points_on_segment[i+1]
            y = np.sin(np.radians(next_lon - lon)) * np.cos(np.radians(next_lat))
            x = np.cos(np.radians(lat)) * np.sin(np.radians(next_lat)) - \
                np.sin(np.radians(lat)) * np.cos(np.radians(next_lat)) * np.cos(np.radians(next_lon - lon))
            heading = (np.degrees(np.arctan2(y, x)) + 360) % 360

        trajectory_data.append({
            'timestamp': timestamp,
            'latitude': noisy_lat,
            'longitude': noisy_lon,
            'speed_kph': speed,
            'heading': heading,
            'road_type_label': road_label
        })
        
    return pd.DataFrame(trajectory_data)

def apply_feature_engineering(raw_df, gdf_highways, gdf_services):
    """Calculates features and distances using Projected CRS (Meters)."""
    
    df = raw_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 1. Kinematic Features
    df['time_diff'] = df.groupby('trajectory_id')['timestamp'].diff().dt.total_seconds().fillna(0)
    df['speed_mps'] = df['speed_kph'] * 1000 / 3600
    df['acceleration_mps2'] = df.groupby('trajectory_id')['speed_mps'].diff().fillna(0) / df['time_diff'].replace(0, 1)
    
    df['heading_diff'] = df.groupby('trajectory_id')['heading'].diff().fillna(0)
    df['heading_diff'] = df['heading_diff'].apply(lambda x: (x + 180) % 360 - 180)
    df['angular_velocity_deg_per_sec'] = df['heading_diff'] / df['time_diff'].replace(0, 1)
    
    # Clean up Infs
    df = df.replace([np.inf, -np.inf], 0).fillna(0)

    # 2. Geometric Features (Distance)
    # Convert points to GeoDataFrame (Lat/Lon)
    gdf_points = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326"
    )

    # PROJECT TO METERS (EPSG:3857)
    gdf_points_proj = gdf_points.to_crs(epsg=3857)
    gdf_highways_proj = gdf_highways.to_crs(epsg=3857)
    gdf_services_proj = gdf_services.to_crs(epsg=3857)

    max_dist = 250 # meters

    # Distance to Highway
    if not gdf_highways_proj.empty:
        sjoin = gpd.sjoin_nearest(gdf_points_proj, gdf_highways_proj, how='left', max_distance=max_dist, distance_col='dist_temp')
        # Handle duplicates if multiple segments are equidistant
        min_dists = sjoin.groupby(sjoin.index)['dist_temp'].min()
        gdf_points['dist_to_nearest_highway'] = gdf_points.index.map(min_dists)
    else:
        gdf_points['dist_to_nearest_highway'] = np.nan

    # Distance to Service Road
    if not gdf_services_proj.empty:
        sjoin = gpd.sjoin_nearest(gdf_points_proj, gdf_services_proj, how='left', max_distance=max_dist, distance_col='dist_temp')
        min_dists = sjoin.groupby(sjoin.index)['dist_temp'].min()
        gdf_points['dist_to_nearest_service_road'] = gdf_points.index.map(min_dists)
    else:
        gdf_points['dist_to_nearest_service_road'] = np.nan

    # Fill NaNs with large distance
    gdf_points = gdf_points.fillna(9999)

    # 3. Ratio Feature
    gdf_points['dist_ratio_service_highway'] = gdf_points['dist_to_nearest_service_road'] / (gdf_points['dist_to_nearest_highway'] + 1e-6)
    gdf_points['dist_ratio_service_highway'] = gdf_points['dist_ratio_service_highway'].replace([np.inf, -np.inf], 1e6)

    return gdf_points

# --- Section 4: Streamlit App Logic ---

st.set_page_config(layout="wide", page_title="Road Classifier")
st.title("🛣️ Road Type Classifier & Map-Matching")

# Initialize Session State
if 'map_loaded' not in st.session_state: st.session_state['map_loaded'] = False
if 'G_all' not in st.session_state: st.session_state['G_all'] = None
if 'highways' not in st.session_state: st.session_state['highways'] = None
if 'service_roads' not in st.session_state: st.session_state['service_roads'] = None
if 'raw_trajectory_df' not in st.session_state: st.session_state['raw_trajectory_df'] = None

# --- STEP 1: SELECT AREA ---
st.sidebar.header("1. Setup Area")
place_input = st.sidebar.text_input("Enter Location:", value="Deoria, Uttar Pradesh, India")

if st.sidebar.button("Load Map Data"):
    with st.spinner(f"Downloading map for {place_input}..."):
        G, highways, services = load_map_data(place_input)
        if G is not None:
            st.session_state['G_all'] = G
            st.session_state['highways'] = highways
            st.session_state['service_roads'] = services
            st.session_state['map_loaded'] = True
            st.session_state['place_name'] = place_input
            st.sidebar.success("Map Loaded!")
        else:
            st.session_state['map_loaded'] = False

# --- STEP 2: INPUT DATA (Only if map loaded) ---
if st.session_state['map_loaded']:
    st.info(f"Monitoring Area: **{st.session_state['place_name']}**")
    
    st.subheader("2. Get Vehicle Position")
    c1, c2, c3 = st.columns(3)
    
    # Option A: Simulation
    with c1:
        st.markdown("### A. Simulation")
        r_type = st.selectbox("Type:", ["Highway", "Service Road"])
        noise = st.slider("GPS Noise (m):", 0, 50, 10)
        if st.button("Simulate Car"):
            df = create_simulated_trajectory(150, noise, r_type, st.session_state['G_all'])
            if not df.empty:
                df['trajectory_id'] = "sim_1"
                st.session_state['raw_trajectory_df'] = df
                st.success("Simulation created!")

    # Option B: Upload
    with c2:
        st.markdown("### B. Upload File")
        f = st.file_uploader("Upload CSV", type='csv')
        if f:
            st.session_state['raw_trajectory_df'] = pd.read_csv(f)
            st.success("File uploaded!")
            
    # Option C: Live GPS
    with c3:
        st.markdown("### C. Live GPS")
        if st.checkbox("📍 Get My Location"):
            loc = get_geolocation()
            if loc:
                lat = loc['coords']['latitude']
                lon = loc['coords']['longitude']
                st.write(f"Found: {lat:.5f}, {lon:.5f}")
                
                # Create 1-row dataframe
                live_df = pd.DataFrame([{
                    'timestamp': datetime.now(),
                    'latitude': lat,
                    'longitude': lon,
                    'speed_kph': 40, # Dummy speed
                    'heading': 0,
                    'trajectory_id': 'live'
                }])
                st.session_state['raw_trajectory_df'] = live_df
            else:
                st.warning("Click 'Allow' in browser popup.")

    # --- STEP 3: ANALYZE ---
    if st.session_state['raw_trajectory_df'] is not None:
        st.divider()
        st.subheader("3. AI Classification")
        
        if st.button("Analyze Road Type"):
            model = load_ml_model(model_path)
            if not model:
                st.error("Model not found! Run main_project.py first.")
            else:
                with st.spinner("Classifying..."):
                    # Process features
                    processed_gdf = apply_feature_engineering(
                        st.session_state['raw_trajectory_df'],
                        st.session_state['highways'],
                        st.session_state['service_roads']
                    )
                    
                    # Predict
                    features = ['speed_kph', 'heading', 'acceleration_mps2', 'angular_velocity_deg_per_sec',
                                'dist_to_nearest_highway', 'dist_to_nearest_service_road', 'dist_ratio_service_highway']
                    
                    # Ensure columns exist
                    X = processed_gdf[features].fillna(0)
                    preds = model.predict(X)
                    processed_gdf['predicted_road_type'] = preds
                    
                    # --- VISUALIZATION ---
                    st.success("Analysis Complete!")
                    
                    # Counts
                    st.write("Prediction Counts:", processed_gdf['predicted_road_type'].value_counts())

                    # Map
                    fig, ax = plt.subplots(figsize=(10, 10))
                    
                    # Background Map
                    st.session_state['highways'].plot(ax=ax, color='gray', linewidth=2, label='Highway (Map)', alpha=0.5)
                    st.session_state['service_roads'].plot(ax=ax, color='lightgray', linewidth=1, label='Service (Map)', alpha=0.5)
                    
                    # Plot Predictions
                    # Highway = Blue, Service = Red
                    highway_pts = processed_gdf[processed_gdf['predicted_road_type'] == 'highway']
                    service_pts = processed_gdf[processed_gdf['predicted_road_type'] == 'service_road']
                    
                    if not highway_pts.empty:
                        ax.plot(highway_pts['longitude'], highway_pts['latitude'], 'o', color='blue', markersize=5, label='Pred: Highway')
                    if not service_pts.empty:
                        ax.plot(service_pts['longitude'], service_pts['latitude'], 'o', color='red', markersize=5, label='Pred: Service')
                        
                    ax.legend()
                    ax.set_title(f"Classification Result in {st.session_state['place_name']}")
                    st.pyplot(fig)

else:
    # Landing Page
    st.info("👈 **Start by entering a location in the sidebar and clicking 'Load Map Data'.**")