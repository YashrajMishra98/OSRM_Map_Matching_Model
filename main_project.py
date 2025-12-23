# main_project.py

# --- Section 1: Import Libraries ---
import os
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

print("--- Starting Map Data Acquisition ---")

# --- Section 2: Define Target Region and Road Filters ---

# Define your place of interest.
# We'll use Lucknow, Uttar Pradesh, India as discussed.
place_name = "Deoria, Uttar Pradesh, India"

# Define what constitutes a "highway" for your project using OSM tags.
# These are generally major roads for faster travel.
# You can customize these tags if your definition changes.
custom_filter_highway = {
    "highway": [
        "motorway",     # Highest class, limited access
        "trunk",        # Major inter-city roads
        "primary",      # Important roads connecting major towns/cities
        "secondary"     # Less important than primary, but still significant
    ]
}

# Define what constitutes a "service road" for your project using OSM tags.
# These are typically access roads, parallel to main roads, or leading to properties.
# 'unclassified' can sometimes function as service roads, but check if it fits your definition.
custom_filter_service = {
    "highway": [
        "service",      # Explicitly defined service roads
        "unclassified"  # General-purpose roads, often minor local roads or access
                        # Consider if 'unclassified' truly represents a 'service road' in your area.
                        # You might remove it if it pulls too many unrelated roads.
    ]
}

# --- Section 3: Download Road Network Data ---

print(f"Downloading road network for: {place_name}")
# Download the entire road network graph for the specified place.
# network_type="all" gets all road types, we will filter them later.
# simplify=True reduces nodes for cleaner graph.
# retain_all=True keeps all original OSM attributes (tags) like speed limits, lanes, etc.
G_all_roads = ox.graph_from_place(place_name, network_type="all", simplify=True, retain_all=True)
print("Road network downloaded.")

# Convert the NetworkX graph edges into a GeoDataFrame.
# GeoDataFrames are excellent for working with spatial data in Python.
gdf_edges = ox.graph_to_gdfs(G_all_roads, nodes=False, edges=True)

# --- Section 4: Filter Road Segments into Highways and Service Roads ---

# Filter for highway segments using the defined custom_filter_highway.
# The .apply(lambda x: ...) part handles cases where 'highway' tag might be a list.
gdf_highways = gdf_edges[gdf_edges['highway'].apply(lambda x: any(tag in x for tag in custom_filter_highway['highway']) if isinstance(x, list) else x in custom_filter_highway['highway'])]

# Filter for service road segments using the defined custom_filter_service.
gdf_service_roads = gdf_edges[gdf_edges['highway'].apply(lambda x: any(tag in x for tag in custom_filter_service['highway']) if isinstance(x, list) else x in custom_filter_service['highway'])]

print(f"Number of highway segments found: {len(gdf_highways)}")
print(f"Number of service road segments found: {len(gdf_service_roads)}")

# --- Section 5: Create Output Directories and Save Map Data ---

# Define the base directory for storing all data
base_data_dir = "data"
# Define the specific directory for map data within the base data directory
map_data_output_dir = os.path.join(base_data_dir, "map_data")
# Define the specific directory for trajectory data within the base data directory
trajectory_output_dir = os.path.join(base_data_dir, "trajectory_data")
# Define the specific directory for feature engineered data within the base data directory
feature_output_dir = os.path.join(base_data_dir, "features")


# Create these directories if they don't already exist.
# exist_ok=True prevents an error if the directory is already there.
os.makedirs(map_data_output_dir, exist_ok=True)
os.makedirs(trajectory_output_dir, exist_ok=True)
os.makedirs(feature_output_dir, exist_ok=True)
print(f"Ensured data directories exist: {base_data_dir}, {map_data_output_dir}, {trajectory_output_dir}, {feature_output_dir}")

# Save the GeoDataFrames to files. Shapefiles (.shp) are common geospatial formats.
# We also save to GeoJSON (.geojson) as it's a more modern and single-file format.
try:
    # Save Highways data
    gdf_highways.to_file(os.path.join(map_data_output_dir, "lucknow_highways.shp"))
    gdf_highways.to_file(os.path.join(map_data_output_dir, "lucknow_highways.geojson"), driver="GeoJSON")

    # Save Service Roads data
    gdf_service_roads.to_file(os.path.join(map_data_output_dir, "lucknow_service_roads.shp"))
    gdf_service_roads.to_file(os.path.join(map_data_output_dir, "lucknow_service_roads.geojson"), driver="GeoJSON")

    print(f"Map data saved successfully to {map_data_output_dir}.")
except Exception as e:
    print(f"Error saving map data: {e}")
    print("Please check file permissions or that the output directory path is valid.")

# --- Section 6: Optional: Visualize the Map Data ---

print("Generating visualization of downloaded map data...")
# Create a plot to visualize the highways and service roads
fig, ax = plt.subplots(figsize=(12, 12)) # Set up a figure and axes for plotting

# Plot highways in blue
gdf_highways.plot(ax=ax, color='blue', linewidth=2, label='Highways')

# Plot service roads in red
gdf_service_roads.plot(ax=ax, color='red', linewidth=1, label='Service Roads')

# Add titles and labels for clarity
ax.set_title(f"Roads in {place_name}: Highways vs. Service Roads")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend() # Show the legend with our labels

plt.show() # Display the plot
print("Map data visualization complete.")

# --- End of Map Data Acquisition Section ---
# --- Section 7: GPS Trajectory Data Acquisition - SIMULATED DATA ---

print("\n--- Generating Simulated GPS Trajectories ---")

# --- Helper Functions for Simulation ---

def generate_noisy_point(lat, lon, noise_meters=5):
    """
    Generates a noisy GPS point around a given latitude and longitude.
    noise_meters: The radius of noise in meters.
    """
    # Convert meters to degrees (approximate, for small distances)
    # 1 degree latitude ~ 111,000 meters
    # 1 degree longitude ~ cos(latitude) * 111,000 meters
    lat_noise_deg = (random.uniform(-noise_meters, noise_meters)) / 111000
    lon_noise_deg = (random.uniform(-noise_meters, noise_meters)) / (111000 * np.cos(np.radians(lat)))

    noisy_lat = lat + lat_noise_deg
    noisy_lon = lon + lon_noise_deg # Fix: Should be lon_noise_deg
    return noisy_lat, noisy_lon

def create_simulated_trajectory(gdf_road_type, num_points=100, noise_meters=10, road_label="unknown"):
    """
    Creates a simulated GPS trajectory along a randomly selected road segment
    from the given GeoDataFrame.
    """
    if gdf_road_type.empty:
        print(f"Warning: No segments available for {road_label} simulation.")
        return pd.DataFrame() # Return an empty DataFrame

    # Randomly select one road segment to simulate movement along.
    # We choose the longest segment to ensure enough points can be generated.
    # Using .iloc[0] after sort_values to get the actual row.
    selected_segment = gdf_road_type.sort_values(by='length', ascending=False).iloc[0]

    # Get the coordinates (Lat/Lon) of the segment's geometry (LineString).
    coords = selected_segment.geometry.coords

    # Interpolate points along the segment to get 'num_points' evenly spaced points.
    # This creates a smooth path before adding noise.
    points_on_segment = []
    # If the segment is too short for the desired number of points, adjust num_points
    if selected_segment.geometry.length < 0.001 and num_points > 1: # Very short line, avoid interpolation error
        num_points = 2 # At least start and end point
    
    # Generate points along the line
    for i in range(num_points):
        fraction = i / (num_points - 1) if num_points > 1 else 0
        point = selected_segment.geometry.interpolate(fraction, normalized=True)
        points_on_segment.append((point.y, point.x)) # (lat, lon)

    trajectory_data = []
    start_time = datetime.now()

    for i, (lat, lon) in enumerate(points_on_segment):
        # Add noise to simulate GPS inaccuracy.
        # Note: We're setting higher noise for service roads as they can be in "urban canyons."
        current_noise = noise_meters if road_label == "highway" else noise_meters * 1.5 # 1.5x noise for service roads
        noisy_lat, noisy_lon = generate_noisy_point(lat, lon, noise_meters=current_noise)

        timestamp = start_time + timedelta(seconds=i * random.uniform(2, 5)) # Simulate varying time intervals.
                                                                           # Points every 2-5 seconds.

        # Simulate speed (adjust based on road type).
        # Highways typically have higher average speed, service roads lower.
        if road_label == "highway":
            speed_kph = random.uniform(60, 100) # km/h
        elif road_label == "service_road":
            speed_kph = random.uniform(20, 40) # km/h
        else:
            speed_kph = random.uniform(30, 70) # Default for unknown
            
        # Simulate heading (direction of travel).
        # For simplicity, calculate heading based on current and next point, if available.
        heading = 0 # Default heading
        if i < len(points_on_segment) - 1:
            next_lat, next_lon = points_on_segment[i+1]
            # Calculate bearing between two points
            delta_lon = next_lon - lon
            y = np.sin(np.radians(delta_lon)) * np.cos(np.radians(next_lat))
            x = np.cos(np.radians(lat)) * np.sin(np.radians(next_lat)) - \
                np.sin(np.radians(lat)) * np.cos(np.radians(next_lat)) * np.cos(np.radians(delta_lon))
            heading = np.degrees(np.arctan2(y, x))
            heading = (heading + 360) % 360 # Normalize to 0-360 degrees
        elif trajectory_data: # If this is the last point, use the previous heading
            heading = trajectory_data[-1]['heading']

        trajectory_data.append({
            'timestamp': timestamp,
            'latitude': noisy_lat,
            'longitude': noisy_lon,
            'speed_kph': speed_kph,
            'heading': heading,
            'road_type_label': road_label # This is our ground truth!
        })

    return pd.DataFrame(trajectory_data)

# --- Generate Multiple Trajectories ---

# Define how many trajectories of each type you want to simulate.
# Start with a small number for testing. You can increase this later for more training data.
num_simulated_trajectories_per_type = 5

all_simulated_trajectories = []

print(f"Generating {num_simulated_trajectories_per_type} highway trajectories...")
for i in range(num_simulated_trajectories_per_type):
    # Ensure there are highway segments available before trying to simulate.
    if not gdf_highways.empty:
        # Generate highway trajectory: 150 points, less noise.
        highway_traj = create_simulated_trajectory(gdf_highways, num_points=150, noise_meters=8, road_label="highway")
        if not highway_traj.empty: # Make sure the simulation actually generated data
            highway_traj['trajectory_id'] = f"highway_traj_{i+1}" # Assign a unique ID
            all_simulated_trajectories.append(highway_traj)
    else:
        print("Skipping highway trajectory generation: No highway segments found in map data.")
        break # No point continuing if no highways

print(f"Generating {num_simulated_trajectories_per_type} service road trajectories...")
for i in range(num_simulated_trajectories_per_type):
    # Ensure there are service road segments available before trying to simulate.
    if not gdf_service_roads.empty:
        # Generate service road trajectory: 100 points, more noise.
        service_road_traj = create_simulated_trajectory(gdf_service_roads, num_points=100, noise_meters=15, road_label="service_road")
        if not service_road_traj.empty: # Make sure the simulation actually generated data
            service_road_traj['trajectory_id'] = f"service_road_traj_{i+1}" # Assign a unique ID
            all_simulated_trajectories.append(service_road_traj)
    else:
        print("Skipping service road trajectory generation: No service road segments found in map data.")
        break # No point continuing if no service roads

# Combine all generated trajectories into one large DataFrame.
if all_simulated_trajectories:
    full_trajectory_df = pd.concat(all_simulated_trajectories, ignore_index=True)
    print(f"Total simulated GPS points generated: {len(full_trajectory_df)}")
    print("First 5 rows of simulated data:")
    print(full_trajectory_df.head())

    # --- Section 8: Save Simulated Trajectory Data ---

    # The 'trajectory_output_dir' variable was defined earlier in Section 5.
    csv_path = os.path.join(trajectory_output_dir, "simulated_gps_trajectories.csv")
    full_trajectory_df.to_csv(csv_path, index=False) # Save to CSV without the Pandas index
    print(f"Simulated GPS trajectories saved to: {csv_path}")

    # --- Section 9: Optional: Visualize Simulated Trajectories ---

    print("\nVisualizing simulated trajectories on the map...")
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot highways and service roads from your map data as a background
    gdf_highways.plot(ax=ax, color='lightgray', linewidth=2, label='Highways (Map)')
    gdf_service_roads.plot(ax=ax, color='darkgray', linewidth=1, label='Service Roads (Map)')

    # Plot the simulated trajectories.
    # We use unique colors for each trajectory for better distinction.
    unique_traj_ids = full_trajectory_df['trajectory_id'].unique()
    # Get a colormap to pick distinct colors
    colors = plt.cm.get_cmap('viridis', len(unique_traj_ids))

    for i, traj_id in enumerate(unique_traj_ids):
        traj_df = full_trajectory_df[full_trajectory_df['trajectory_id'] == traj_id]
        ax.plot(traj_df['longitude'], traj_df['latitude'], marker='o', linestyle='-', markersize=3,
                color=colors(i), alpha=0.7, # alpha for transparency
                label=f'Simulated {traj_df["road_type_label"].iloc[0]} Trajectory {i+1}')

    ax.set_title("Simulated GPS Trajectories with Noise on Map")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc='lower left', bbox_to_anchor=(1, 0)) # Place legend outside the plot area
    plt.tight_layout() # Adjust plot parameters for a tight layout
    plt.show() # Display the plot
    print("Simulated trajectory visualization complete.")

else:
    print("No simulated trajectories were generated (check if map data was empty).")

# --- End of Simulated Trajectory Data Section ---
# --- Section 10: Feature Engineering (CORRECTED) ---

print("\n--- Starting Feature Engineering ---")

# ... (Previous timestamp code is fine) ...

# 1. Kinematic Features (Keep as is)
full_trajectory_df['time_diff_seconds'] = full_trajectory_df.groupby('trajectory_id')['timestamp'].diff().dt.total_seconds().fillna(0)
full_trajectory_df['speed_mps'] = full_trajectory_df['speed_kph'] * 1000 / 3600
full_trajectory_df['acceleration_mps2'] = full_trajectory_df.groupby('trajectory_id')['speed_mps'].diff().fillna(0) / full_trajectory_df['time_diff_seconds']
full_trajectory_df['acceleration_mps2'] = full_trajectory_df['acceleration_mps2'].replace([np.inf, -np.inf], 0).fillna(0)
full_trajectory_df['heading_diff'] = full_trajectory_df.groupby('trajectory_id')['heading'].diff().fillna(0)
full_trajectory_df['heading_diff'] = full_trajectory_df['heading_diff'].apply(lambda x: (x + 180) % 360 - 180)
full_trajectory_df['angular_velocity_deg_per_sec'] = full_trajectory_df['heading_diff'] / full_trajectory_df['time_diff_seconds']
full_trajectory_df['angular_velocity_deg_per_sec'] = full_trajectory_df['angular_velocity_deg_per_sec'].replace([np.inf, -np.inf], 0).fillna(0)

# 2. Geometric Features (CORRECTED TO USE METERS/EPSG:3857)
print("Extracting geometric features using Projected CRS (Meters)...")

# Convert Trajectories to GeoDataFrame
gdf_trajectories = gpd.GeoDataFrame(
    full_trajectory_df,
    geometry=gpd.points_from_xy(full_trajectory_df.longitude, full_trajectory_df.latitude),
    crs="EPSG:4326"
)

# --- CRITICAL FIX: Reproject everything to EPSG:3857 (Meters) ---
gdf_trajectories_proj = gdf_trajectories.to_crs(epsg=3857)
gdf_highways_proj = gdf_highways.to_crs(epsg=3857)
gdf_service_roads_proj = gdf_service_roads.to_crs(epsg=3857)

max_distance_meters = 250 # Match app.py

# Calculate Highway Distance
if not gdf_highways_proj.empty and not gdf_trajectories_proj.empty:
    sjoined_highway = gpd.sjoin_nearest(
        gdf_trajectories_proj, 
        gdf_highways_proj, 
        how="left",
        max_distance=max_distance_meters, 
        distance_col="dist_temp"
    )
    # Get min distance per point (handle duplicates from sjoin)
    min_dists = sjoined_highway.groupby(sjoined_highway.index)['dist_temp'].min()
    gdf_trajectories['dist_to_nearest_highway'] = gdf_trajectories.index.map(min_dists)
else:
    gdf_trajectories['dist_to_nearest_highway'] = np.nan

# Calculate Service Road Distance
if not gdf_service_roads_proj.empty and not gdf_trajectories_proj.empty:
    sjoined_service = gpd.sjoin_nearest(
        gdf_trajectories_proj, 
        gdf_service_roads_proj, 
        how="left",
        max_distance=max_distance_meters, 
        distance_col="dist_temp"
    )
    min_dists = sjoined_service.groupby(sjoined_service.index)['dist_temp'].min()
    gdf_trajectories['dist_to_nearest_service_road'] = gdf_trajectories.index.map(min_dists)
else:
    gdf_trajectories['dist_to_nearest_service_road'] = np.nan

# Fill NaNs
gdf_trajectories['dist_to_nearest_highway'] = gdf_trajectories['dist_to_nearest_highway'].fillna(9999)
gdf_trajectories['dist_to_nearest_service_road'] = gdf_trajectories['dist_to_nearest_service_road'].fillna(9999)

# 3. Ratio Features
gdf_trajectories['dist_ratio_service_highway'] = (
    gdf_trajectories['dist_to_nearest_service_road'] /
    (gdf_trajectories['dist_to_nearest_highway'] + 1e-6)
)
gdf_trajectories['dist_ratio_service_highway'] = gdf_trajectories['dist_ratio_service_highway'].replace([np.inf, -np.inf], 1e6)

# --- Section 11: Prepare Final DataFrame for Model Training ---

# Select only the columns that will be used as features for the ML model,
# plus the 'road_type_label' which is our target variable (what we want to predict).
# Drop intermediate calculation columns like 'geometry', 'time_diff_seconds', etc.
final_features_df = gdf_trajectories[[
    'trajectory_id',
    'timestamp',
    'latitude',
    'longitude',
    'speed_kph',
    'heading',
    'acceleration_mps2',
    'angular_velocity_deg_per_sec',
    'dist_to_nearest_highway',
    'dist_to_nearest_service_road',
    'dist_ratio_service_highway',
    'road_type_label' # This is our ground truth (what the model learns to predict)
]].copy() # Use .copy() to avoid SettingWithCopyWarning

# --- Section 12: Save the Feature-Engineered Data ---

# The 'feature_output_dir' variable was defined earlier in Section 5.
features_csv_path = os.path.join(feature_output_dir, "engineered_features.csv")
final_features_df.to_csv(features_csv_path, index=False) # Save to CSV without the Pandas index

print(f"Feature engineering complete. Data saved to: {features_csv_path}")
print("\nFirst 5 rows of the feature-engineered data:")
print(final_features_df.head())
print("\nData types of the feature-engineered data:")
print(final_features_df.info())

print("\n--- End of Feature Engineering ---")

# --- Section 13: Model Development and Training ---

print("\n--- Starting Model Development and Training ---")

# Import necessary libraries for machine learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib # For saving the model

# Load the feature-engineered data
# The 'features_csv_path' variable was defined earlier in Section 12.
try:
    data = pd.read_csv(features_csv_path)
    print(f"Successfully loaded feature-engineered data from: {features_csv_path}")
except FileNotFoundError:
    print(f"Error: {features_csv_path} not found. Please ensure feature engineering ran successfully.")
    exit() # Exit if the data isn't found

# --- Prepare the Data for the Model ---

# Define the features (X) and the target variable (y)
# Features are all columns except 'trajectory_id', 'timestamp', 'latitude', 'longitude', and 'road_type_label'
features = [
    'speed_kph',
    'heading',
    'acceleration_mps2',
    'angular_velocity_deg_per_sec',
    'dist_to_nearest_highway',
    'dist_to_nearest_service_road',
    'dist_ratio_service_highway'
]

X = data[features] # Our input features
y = data['road_type_label'] # Our target (what we want to predict: 'highway' or 'service_road')

print(f"Features selected: {features}")
print(f"Target variable: {y.name}")
print(f"Shape of X (features): {X.shape}")
print(f"Shape of y (target): {y.shape}")

# --- Split Data into Training and Testing Sets ---
# We split the data to train the model on one part and test its performance on unseen data.
# test_size=0.2 means 20% of the data will be used for testing, 80% for training.
# random_state ensures reproducibility (you'll get the same split every time).
print("Splitting data into training and testing sets (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# stratify=y ensures that the proportion of 'highway' and 'service_road' labels
# is roughly the same in both training and testing sets.

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train value counts:\n{y_train.value_counts()}")
print(f"y_test value counts:\n{y_test.value_counts()}")


# --- Train the Machine Learning Model (Random Forest Classifier) ---
print("Training RandomForestClassifier model...")
# RandomForestClassifier is a powerful ensemble model, good for classification tasks.
# n_estimators: number of trees in the forest. More trees usually mean better performance but slower training.
# random_state: for reproducibility.
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1 uses all available CPU cores
model.fit(X_train, y_train)
print("Model training complete.")

# --- Evaluate the Model ---
print("\n--- Evaluating Model Performance ---")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate Accuracy: The proportion of correctly classified samples.
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Set: {accuracy:.4f}")

# Generate a Classification Report: Provides precision, recall, f1-score for each class.
# Precision: Out of all predicted as 'X', how many were actually 'X'?
# Recall: Out of all actual 'X', how many were correctly predicted as 'X'?
# F1-score: Harmonic mean of precision and recall.
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Generate a Confusion Matrix: Shows counts of true positives, true negatives, false positives, false negatives.
# Rows are actual classes, columns are predicted classes.
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --- Feature Importance (Optional but helpful) ---
# See which features the model considered most important for its decisions.
print("\nFeature Importances (Top 5):")
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(feature_importances.head(5))

# --- Save the Trained Model ---
# Saving the model allows you to load it later without retraining.
model_output_dir = os.path.join(base_data_dir, "model") # Create a 'model' directory
os.makedirs(model_output_dir, exist_ok=True)
model_path = os.path.join(model_output_dir, "road_type_classifier_model.joblib")
joblib.dump(model, model_path)
print(f"\nTrained model saved to: {model_path}")

print("\n--- End of Model Development and Training ---")

# --- Section 14: Map-Matching Implementation ---

print("\n--- Starting Map-Matching Implementation ---")

# Load the trained model
try:
    loaded_model = joblib.load(model_path)
    print(f"Successfully loaded trained model from: {model_path}")
except FileNotFoundError:
    print(f"Error: Trained model not found at {model_path}. Please ensure model training ran successfully.")
    exit()

# --- Simulate a NEW Trajectory for Map-Matching ---
# We'll simulate a single new trajectory to test the map-matching.
# It's good to pick a segment that wasn't necessarily one of the "longest" ones used for training.

print("Simulating a new trajectory for map-matching demonstration...")

# Randomly choose whether this new trajectory is on a highway or service road
simulate_on_highway = random.choice([True, False])

if simulate_on_highway:
    road_type_gdf = gdf_highways # Use highway segments
    road_label_sim = "highway"
    # Try to pick a random segment, not necessarily the longest one
    if not road_type_gdf.empty:
        # Select a random segment index
        random_idx = random.randint(0, len(road_type_gdf) - 1)
        # Ensure 'length' column exists for sorting, or handle if not
        if 'length' in road_type_gdf.columns:
            selected_segment_for_matching = road_type_gdf.sample(n=1).iloc[0] # Pick a random one directly
        else:
            selected_segment_for_matching = road_type_gdf.iloc[random_idx]
    else:
        print("Warning: No highway segments to simulate a new trajectory.")
        selected_segment_for_matching = None
else:
    road_type_gdf = gdf_service_roads # Use service road segments
    road_label_sim = "service_road"
    if not road_type_gdf.empty:
        random_idx = random.randint(0, len(road_type_gdf) - 1)
        if 'length' in road_type_gdf.columns:
            selected_segment_for_matching = road_type_gdf.sample(n=1).iloc[0] # Pick a random one directly
        else:
            selected_segment_for_matching = road_type_gdf.iloc[random_idx]
    else:
        print("Warning: No service road segments to simulate a new trajectory.")
        selected_segment_for_matching = None

if selected_segment_for_matching is None:
    print("Cannot simulate new trajectory as no suitable road segments were found. Exiting map-matching.")
else:
    # Use the helper function to create a single simulated trajectory
    # More points for a more continuous demo, less noise for clarity in matching
    new_sim_trajectory_df = create_simulated_trajectory(
        gpd.GeoDataFrame([selected_segment_for_matching], crs=road_type_gdf.crs),
        num_points=200, # More points for the demo
        noise_meters=10, # Moderate noise
        road_label=road_label_sim # The true label for this demo trajectory
    )
    new_sim_trajectory_df['trajectory_id'] = "new_demo_trajectory_1"
    print(f"New demo trajectory simulated on a {road_label_sim} segment with {len(new_sim_trajectory_df)} points.")
    print("First 5 rows of new simulated data:")
    print(new_sim_trajectory_df.head())

    # --- Apply Feature Engineering to the NEW Trajectory ---
    print("\nApplying feature engineering to the new trajectory...")

    # Ensure 'timestamp' is datetime
    if not pd.api.types.is_datetime64_any_dtype(new_sim_trajectory_df['timestamp']):
        new_sim_trajectory_df['timestamp'] = pd.to_datetime(new_sim_trajectory_df['timestamp'])

    # Kinematic Features (re-calculate using the same logic)
    new_sim_trajectory_df['time_diff_seconds'] = new_sim_trajectory_df.groupby('trajectory_id')['timestamp'].diff().dt.total_seconds().fillna(0)
    new_sim_trajectory_df['speed_mps'] = new_sim_trajectory_df['speed_kph'] * 1000 / 3600
    new_sim_trajectory_df['acceleration_mps2'] = new_sim_trajectory_df.groupby('trajectory_id')['speed_mps'].diff().fillna(0) / new_sim_trajectory_df['time_diff_seconds']
    new_sim_trajectory_df['acceleration_mps2'] = new_sim_trajectory_df['acceleration_mps2'].replace([np.inf, -np.inf], 0).fillna(0)
    new_sim_trajectory_df['heading_diff'] = new_sim_trajectory_df.groupby('trajectory_id')['heading'].diff().fillna(0)
    new_sim_trajectory_df['heading_diff'] = new_sim_trajectory_df['heading_diff'].apply(lambda x: (x + 180) % 360 - 180)
    new_sim_trajectory_df['angular_velocity_deg_per_sec'] = new_sim_trajectory_df['heading_diff'] / new_sim_trajectory_df['time_diff_seconds']
    new_sim_trajectory_df['angular_velocity_deg_per_sec'] = new_sim_trajectory_df['angular_velocity_deg_per_sec'].replace([np.inf, -np.inf], 0).fillna(0)

    # Geometric Features (re-calculate using the same logic)
    gdf_new_trajectory_points = gpd.GeoDataFrame(
        new_sim_trajectory_df,
        geometry=gpd.points_from_xy(new_sim_trajectory_df.longitude, new_sim_trajectory_df.latitude),
        crs="EPSG:4326"
    )
    gdf_new_trajectory_points['dist_to_nearest_highway'] = np.nan
    gdf_new_trajectory_points['dist_to_nearest_service_road'] = np.nan

    if not gdf_highways.empty and not gdf_new_trajectory_points.empty:
        sjoined_highway_new = gpd.sjoin_nearest(gdf_new_trajectory_points, gdf_highways, how="left",
                                                max_distance=0.1, distance_col="distance_sjoin_highway_deg")
        if 'index_right' in sjoined_highway_new.columns and not sjoined_highway_new.empty:
            for idx, row in sjoined_highway_new.iterrows():
                if pd.notna(row['index_right']):
                    point_geom = row.geometry
                    line_geom = gdf_highways.loc[row['index_right']].geometry
                    closest_point_on_line = line_geom.interpolate(line_geom.project(point_geom))
                    gdf_new_trajectory_points.loc[idx, 'dist_to_nearest_highway'] = point_geom.distance(closest_point_on_line) * 111000
        else:
            print("INFO: No nearest highway segments found for new trajectory points.")
    else:
        print("WARNING: Skipping highway distance calculation for new trajectory. Either gdf_highways or gdf_new_trajectory_points is empty.")

    if not gdf_service_roads.empty and not gdf_new_trajectory_points.empty:
        sjoined_service_new = gpd.sjoin_nearest(gdf_new_trajectory_points, gdf_service_roads, how="left",
                                                max_distance=0.1, distance_col="distance_sjoin_service_deg")
        if 'index_right' in sjoined_service_new.columns and not sjoined_service_new.empty:
            for idx, row in sjoined_service_new.iterrows():
                if pd.notna(row['index_right']):
                    point_geom = row.geometry
                    line_geom = gdf_service_roads.loc[row['index_right']].geometry
                    closest_point_on_line = line_geom.interpolate(line_geom.project(point_geom))
                    gdf_new_trajectory_points.loc[idx, 'dist_to_nearest_service_road'] = point_geom.distance(closest_point_on_line) * 111000
        else:
            print("INFO: No nearest service road segments found for new trajectory points.")
    else:
        print("WARNING: Skipping service road distance calculation for new trajectory. Either gdf_service_roads or gdf_new_trajectory_points is empty.")

    gdf_new_trajectory_points['dist_to_nearest_highway'].fillna(9999, inplace=True)
    gdf_new_trajectory_points['dist_to_nearest_service_road'].fillna(9999, inplace=True)

    # Combined/Ratio Features
    gdf_new_trajectory_points['dist_ratio_service_highway'] = (
        gdf_new_trajectory_points['dist_to_nearest_service_road'] /
        (gdf_new_trajectory_points['dist_to_nearest_highway'] + 1e-6)
    )
    gdf_new_trajectory_points['dist_ratio_service_highway'].replace([np.inf, -np.inf], 1e6, inplace=True)

    print("Feature engineering applied to new trajectory.")

    # --- Make Predictions using the Loaded Model ---

    # Select the same features used during training
    features_for_prediction = [
        'speed_kph',
        'heading',
        'acceleration_mps2',
        'angular_velocity_deg_per_sec',
        'dist_to_nearest_highway',
        'dist_to_nearest_service_road',
        'dist_ratio_service_highway'
    ]

    # Ensure all required columns exist in the new trajectory data
    if not all(f in gdf_new_trajectory_points.columns for f in features_for_prediction):
        print("Error: Not all required features present in new trajectory data for prediction.")
        print(f"Missing features: {set(features_for_prediction) - set(gdf_new_trajectory_points.columns)}")
    else:
        X_new = gdf_new_trajectory_points[features_for_prediction]
        predictions = loaded_model.predict(X_new)
        prediction_probabilities = loaded_model.predict_proba(X_new) # Get probabilities

        # Add predictions and probabilities to the new trajectory DataFrame
        gdf_new_trajectory_points['predicted_road_type'] = predictions
        # Get the probability for the predicted class
        class_labels = loaded_model.classes_ # ['highway', 'service_road'] or vice-versa
        prob_df = pd.DataFrame(prediction_probabilities, columns=class_labels)
        gdf_new_trajectory_points['predicted_road_type_proba'] = prob_df.max(axis=1) # Max probability of predicted class

        print("Predictions made for the new trajectory.")
        print("First 5 points of predicted data:")
        print(gdf_new_trajectory_points[['timestamp', 'latitude', 'longitude', 'predicted_road_type', 'predicted_road_type_proba', 'road_type_label']].head())


    # --- Visualize Map-Matched Trajectory ---

    print("\nVisualizing the map-matched trajectory...")

    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot map data as background
    gdf_highways.plot(ax=ax, color='lightgray', linewidth=2, label='Highways (Map)')
    gdf_service_roads.plot(ax=ax, color='darkgray', linewidth=1, label='Service Roads (Map)')

    # Plot the original simulated trajectory
    ax.plot(new_sim_trajectory_df['longitude'], new_sim_trajectory_df['latitude'],
            color='purple', linestyle='--', marker='.', markersize=4, alpha=0.6,
            label=f'Simulated Trajectory (True: {road_label_sim})')

    # Plot the predicted segments, colored by predicted road type
    # Group by predicted_road_type to plot segments in different colors
    for pred_type, group in gdf_new_trajectory_points.groupby('predicted_road_type'):
        color = 'blue' if pred_type == 'highway' else 'red'
        label = f'Predicted: {pred_type}'
        ax.plot(group['longitude'], group['latitude'],
                color=color, linestyle='-', marker='o', markersize=3, alpha=0.8,
                label=label if label not in [l.get_label() for l in ax.get_lines()] else "") # Avoid duplicate labels

    # Add legend, title, labels
    ax.set_title(f"Map-Matched Trajectory (Simulated True: {road_label_sim})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc='upper left', bbox_to_anchor=(1, 0)) # Legend outside
    plt.tight_layout()
    plt.show()

    print("Map-matching visualization complete.")

print("\n--- End of Map-Matching Implementation ---")