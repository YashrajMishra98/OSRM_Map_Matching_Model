🛣️ Road Type Classifier & Map-Matching System
A Machine Learning-powered application that intelligently distinguishes whether a vehicle is traveling on a Main Highway or a parallel Service Road using GPS data.

📖 Overview
Standard GPS navigation often struggles to distinguish between a highway and a service road running immediately next to it (the "Blue Dot Drift" problem).

This project solves that by using Context-Aware Machine Learning. Instead of relying solely on raw coordinates, it analyzes:

Geometric Features: Distance to nearest highway vs. service road.

Kinematic Features: Speed, acceleration, and heading changes.

It visualizes the result on a map, color-coding the trajectory to show the classified road type.

✨ Key Features
🌍 Dynamic Map Loading: Fetches live road networks for any city using OpenStreetMap (OSMnx).

🤖 Machine Learning Core: Uses a Random Forest Classifier trained on kinematic and spatial features.

📡 Multi-Source Input:

Simulation: Generates noisy GPS data to test the model.

CSV Upload: Analyze historical trip logs.

Live GPS: Get real-time location from your device (requires HTTPS/Localhost).

📊 Advanced Visualization: Plots the classification results directly over the road network.

🛠️ Installation & Setup
1. Prerequisites
Ensure you have Python installed (version 3.8 or higher).

2. Install Dependencies
Run the following command in your terminal to install the required libraries:

Bash

pip install streamlit pandas numpy geopandas osmnx matplotlib networkx shapely scikit-learn joblib streamlit-js-eval
3. Project Structure
Ensure your folder looks like this:

/Your-Project-Folder
│
├── main_project.py    # Script to train the ML model
├── app.py             # The Streamlit Dashboard (Frontend)
├── README.md          # This file
└── data/              # Created automatically (stores maps and models)
🚀 How to Run
Step 1: Train the Model (Do this once)
Before using the app, you need to generate the Machine Learning model. Run the training script:

Bash

python main_project.py
This will download map data, simulate training trajectories, train the Random Forest model, and save it to data/model/road_type_classifier_model.joblib.

Step 2: Launch the App
Start the web interface:

Bash

streamlit run app.py
Step 3: Use the Dashboard
Sidebar: Enter a location (e.g., "Lucknow, India") and click Load Map Data.

Main Screen: Choose your input method:

Simulation: Create a fake car path.

Live GPS: Use your current location.

Analyze: Click Analyze Road Type.

Result: View the map. Blue dots = Highway, Red dots = Service Road.

🧠 How It Works (The Logic)
Data Acquisition: The system downloads the road graph for the selected area and separates edges into "Highway" and "Service" categories based on OSM tags.

Feature Engineering: For every GPS point, it calculates:

dist_to_nearest_highway (Meters)

dist_to_nearest_service_road (Meters)

speed_kph & acceleration

angular_velocity (Turning rate)

Classification: The Random Forest model predicts the class based on these features. For example, high speed + close to highway = Highway. Low speed + close to service road = Service Road.

⚠️ Troubleshooting
"Model not found": You likely skipped Step 1. Run python main_project.py first.

"Map data not loading": Check your internet connection or try a different city spelling (e.g., "Delhi, India" instead of just "Delhi").

"Live GPS not working": Live GPS requires browser permissions. It works on localhost or secure https domains. It may not work if you host it on an insecure http server.

"Dots coincide on the map": This is expected. The system classifies existing points; it does not move them. The color change indicates the classification.

📜 License
This project uses OpenStreetMap data (ODbL). Feel free to use and modify the code for educational purposes.