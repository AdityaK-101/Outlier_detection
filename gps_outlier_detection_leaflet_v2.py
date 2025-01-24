import base64
import asammdf
import pandas as pd
import numpy as np
from dash import Dash, html, dcc, Input, Output, State
import dash_leaflet as dl
from geopy.distance import geodesic
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from sklearn.linear_model import LogisticRegression
import geopandas as gpd
from shapely.geometry import Point, LineString
from math import sqrt
from math import radians, degrees, sin, cos, atan2
from sklearn.cluster import DBSCAN

# Function to load land boundary data
def load_land_data():
    shapefile_path = "ne_110m_admin_0_countries.shp"
    try:
        world = gpd.read_file(shapefile_path)
        print(f"Shapefile read successfully. Found {len(world)} countries.")
    except Exception as e:
        print(f"Error reading shapefile: {e}")
    return world

# Function to check if a point is in land
def is_in_land(lat, lon, land_data):
    point = Point(lon, lat)
    return land_data.contains(point).any()

# Function to calculate the angle between consecutive points
def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    d_lon = lon2 - lon1
    x = cos(lat2) * sin(d_lon)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(d_lon)
    bearing = atan2(x, y)
    return (degrees(bearing) + 360) % 360

# Function to calculate distance between consecutive GPS points
def calculate_distance(lat1, lon1, lat2, lon2):
    # Using geodesic to calculate distance in meters
    return geodesic((lat1, lon1), (lat2, lon2)).meters


# Function to enhance outlier detection with additional features
def enhanced_outlier_detection(df):
    # Shift LATITUDE and LONGITUDE to calculate bearing difference
    df['LATITUDE_shifted'] = df['LATITUDE'].shift()
    df['LONGITUDE_shifted'] = df['LONGITUDE'].shift()

    # Apply bearing calculation with shifted values
    df['bearing_diff'] = df.apply(lambda row: calculate_bearing(row['LATITUDE'], row['LONGITUDE'], row['LATITUDE_shifted'], row['LONGITUDE_shifted']), axis=1)

    # Calculate the distance between consecutive points
    df['distance'] = df.apply(
        lambda row: calculate_distance(row['LATITUDE'], row['LONGITUDE'], row['LATITUDE_shifted'], row['LONGITUDE_shifted'])
        if pd.notna(row['LATITUDE']) and pd.notna(row['LONGITUDE']) and pd.notna(row['LATITUDE_shifted']) and pd.notna(row['LONGITUDE_shifted'])
        else 0,
        axis=1
    )
    
    # Drop the shifted columns after use
    df.drop(['LATITUDE_shifted', 'LONGITUDE_shifted'], axis=1, inplace=True)

    # Z-score based outlier detection
    zscore_features = ['distance', 'bearing_diff']
    for feature in zscore_features:
        df[f'{feature}_zscore'] = zscore(df[feature].fillna(0))
    
    # Detect Z-score outliers with different thresholds
    thresholds = {
        'distance': 8.5,
        'bearing_diff': 3.5
    }
    
    df['zscore_outliers'] = False
    for feature, threshold in thresholds.items():
        df['zscore_outliers'] |= abs(df[f'{feature}_zscore']) > threshold
    
    # Enhanced Isolation Forest
    features = df[['LATITUDE', 'LONGITUDE', 'SPEED_OVER_GROUND', 'distance', 'bearing_diff']]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features.fillna(0))
    
    isolation_model = IsolationForest(n_estimators=200, contamination=0.0095, random_state=42)
    df['isolation_outliers'] = isolation_model.fit_predict(features_scaled) == -1
    
    def calculate_angle(lat1, lon1, lat2, lon2, lat3, lon3):
        angle = abs(calculate_bearing(lat1, lon1, lat2, lon2) - calculate_bearing(lat2, lon2, lat3, lon3))
        return angle if angle <= 180 else 360 - angle

    # Adjusted outlier detection logic
    for i in range(1, len(df) - 1):
        angle = calculate_angle(df.iloc[i-1]['LATITUDE'], df.iloc[i-1]['LONGITUDE'],
                                df.iloc[i]['LATITUDE'], df.iloc[i]['LONGITUDE'],
                                df.iloc[i+1]['LATITUDE'], df.iloc[i+1]['LONGITUDE'])
        if angle <= 2:
            df.at[i, 'isolation_outliers'] = False  # Do not mark as outlier if angle is small

    # Apply logical rules for inliers (minimal distance and bearing_diff)
    df['logical_inliers'] = (df['distance'] < 10) & (df['bearing_diff'] < 5)  # Apply custom thresholds
    
    # Combine logical inliers with Isolation Forest outliers
    df['final_outliers'] = df['isolation_outliers'] & ~df['logical_inliers']  # Mark as outliers unless logical rule inverts
    
    # Define the threshold for a U-turn
    u_turn_threshold = 170  # Degrees

    # Initialize the u_turn column
    df['u_turn'] = False

    # Mark U-turns based on bearing_diff
    for i in range(1, len(df) - 1):
        if df['bearing_diff'].iloc[i] >= u_turn_threshold:
            df.at[i, 'u_turn'] = True

    # Define the threshold for a sharp turn
    sharp_turn_threshold = 150  # Degrees

    # Initialize the zigzag_outlier column
    df['zigzag_outlier'] = False

    # Mark zigzag outliers based on angles
    for i in range(1, len(df) - 1):
        angle = calculate_angle(df.iloc[i-1]['LATITUDE'], df.iloc[i-1]['LONGITUDE'],
                                df.iloc[i]['LATITUDE'], df.iloc[i]['LONGITUDE'],
                                df.iloc[i+1]['LATITUDE'], df.iloc[i+1]['LONGITUDE'])
        if angle >= sharp_turn_threshold:
            df.at[i, 'zigzag_outlier'] = True
            df.at[i-1, 'zigzag_outlier'] = True  # Also mark the previous point as part of the zigzag

    return df

# Function to adaptively smooth the path and interpolate gaps
def adaptive_smooth_path(df):
    smoothed_df = df.copy()
    long_gap_threshold = 50  # Adjust this value as needed

    for idx in smoothed_df[smoothed_df['stacked_outliers']].index:
        prev_index = smoothed_df.loc[:idx - 1][~smoothed_df.loc[:idx - 1]['stacked_outliers']].last_valid_index()
        next_index = smoothed_df.loc[idx + 1:][~smoothed_df.loc[idx + 1:]['stacked_outliers']].first_valid_index()

        if prev_index is not None and next_index is not None:
            distance = calculate_distance(smoothed_df.loc[prev_index, 'LATITUDE'], smoothed_df.loc[prev_index, 'LONGITUDE'],
                                          smoothed_df.loc[next_index, 'LATITUDE'], smoothed_df.loc[next_index, 'LONGITUDE'])
            if distance > long_gap_threshold:
                # Skip interpolation for this segment
                continue
            else:
                # Perform interpolation if the gap is acceptable
                smoothed_df.loc[idx, 'LATITUDE'] = np.interp(idx, [prev_index, next_index], 
                                                             [smoothed_df.loc[prev_index, 'LATITUDE'], smoothed_df.loc[next_index, 'LATITUDE']])
                smoothed_df.loc[idx, 'LONGITUDE'] = np.interp(idx, [prev_index, next_index], 
                                                              [smoothed_df.loc[prev_index, 'LONGITUDE'], smoothed_df.loc[next_index, 'LONGITUDE']])

    return smoothed_df

# Function to improve stacking method for outlier detection
def improved_stack_outlier_detection(df):
    # Apply enhanced outlier detection
    df = enhanced_outlier_detection(df)
    
    # Combine the outlier results
    df['combined_outliers'] = df['zscore_outliers'] | df['final_outliers']
    
    # Stack the outlier results
    stacked_features = df[['zscore_outliers', 'final_outliers']].values
    stacked_labels = df['combined_outliers'].values
    
    # Use Random Forest for stacking
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,  # Limit tree depth
        class_weight='balanced',  # Handle imbalanced classes
        random_state=42
    )
    model.fit(stacked_features, stacked_labels)
    
    # Predict outliers and combine with original outliers
    rf_predictions = model.predict(stacked_features)
    df['stacked_outliers'] = rf_predictions | df['combined_outliers']
    
    return df

# Function to handle clustering and marking start/end points as outliers
def mark_start_end_clusters(df):
    start_end_distance_threshold = 100  # meters
    start_end_bearing_threshold = 45  # degrees

    # Check for clusters near the start and end points
    for idx in range(len(df)):
        # Start cluster: First few points
        if idx < 10:  # First 10 points
            df.loc[idx, 'stacked_outliers'] = True
        # End cluster: Last few points
        elif idx > len(df) - 10:  # Last 10 points
            df.loc[idx, 'stacked_outliers'] = True

    return df

# Function to identify concentrated clusters
def identify_concentrated_clusters(gps_data, eps=0.1, min_samples=10):
    """
    Identify concentrated clusters in GPS data using DBSCAN.
    
    Parameters:
    gps_data (array-like): The input GPS data points.
    eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
    min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
    
    Returns:
    labels (array): Cluster labels for concentrated areas. Noise points are labeled as -1.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(gps_data)
    return labels

# Function to resample and clean data
def resample_and_clean_data(df, land_data):
    df = df.reset_index(drop=True)

    # Apply improved stacked outlier detection
    df = improved_stack_outlier_detection(df)

    # Mark start and end clusters as outliers
    df = mark_start_end_clusters(df)

    # Identify concentrated clusters
    gps_data = df[['LATITUDE', 'LONGITUDE']].values
    concentrated_labels = identify_concentrated_clusters(gps_data, eps=0.1, min_samples=10)
    df['concentrated_outliers'] = concentrated_labels == -1

    # Apply adaptive smoothing
    df = adaptive_smooth_path(df)

    # Check if the point is in land
    df['in_land'] = df.apply(lambda row: is_in_land(row['LATITUDE'], row['LONGITUDE'], land_data), axis=1)

    # Filter out points that are in the ocean
    df = df[df['in_land']]  # Keep only points that are in land
    
    return df

# Initialize Dash app
app = Dash(__name__)

# Load land boundary data (this is done once when the app starts)
land_data = load_land_data()

# Layout of the app
app.layout = html.Div([
    html.H1("MF4 Data Processing and Outlier Removal", style={'textAlign': 'center'}),
    dcc.Upload(
        id='upload-mf4',
        children=html.Button('Upload MF4 File'),
        multiple=False
    ),
    html.Div(id='file-info', style={'textAlign': 'center'}),
    dl.Map(
        id='map',
        style={'width': '100%', 'height': '80vh'},
        center=[20, 0],
        zoom=2,
        maxZoom=22,
        scrollWheelZoom=0.3,  # Reduce scroll wheel zoom sensitivity
        children=[
            dl.TileLayer(
                url='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
                maxZoom=22  
            ),
            dl.LayerGroup(id='inliers'),
            dl.LayerGroup(id='outliers'),
            dl.LayerGroup(id='start-end'),
            dl.LayerGroup(id='ocean'),
            dl.LayerGroup(id='concentrated-outliers'),
            html.Div(
            [
                html.Div([
                    html.Div(style={'backgroundColor': 'blue', 'width': '20px', 'height': '20px', 'display': 'inline-block', 'marginRight': '5px', 'borderRadius': '50%'}),
                    html.Span("Inliers")
                ], style={'marginBottom': '5px'}),
                html.Div([
                    html.Div(style={'backgroundColor': 'red', 'width': '20px', 'height': '20px', 'display': 'inline-block', 'marginRight': '5px', 'borderRadius': '50%'}),
                    html.Span("Outliers")
                ], style={'marginBottom': '5px'}),
                html.Div([
                    html.Div(style={'backgroundColor': 'cyan', 'width': '20px', 'height': '20px', 'display': 'inline-block', 'marginRight': '5px', 'borderRadius': '50%'}),
                    html.Span("Ocean Points")
                ], style={'marginBottom': '5px'}),
                html.Div([
                    html.Div(style={'backgroundColor': 'green', 'width': '20px', 'height': '20px', 'display': 'inline-block', 'marginRight': '5px', 'borderRadius': '50%'}),
                    html.Span("Start Point")
                ], style={'marginBottom': '5px'}),
                html.Div([
                    html.Div(style={'backgroundColor': 'orange', 'width': '20px', 'height': '20px', 'display': 'inline-block', 'marginRight': '5px', 'borderRadius': '50%'}),
                    html.Span("End Point")
                ], style={'marginBottom': '5px'}),
                # html.Div([
                #     html.Div(style={'backgroundColor': 'purple', 'width': '20px', 'height': '20px', 'display': 'inline-block', 'marginRight': '5px', 'borderRadius': '50%'}),
                #     html.Span("Concentrated Outliers")
                # ], style={'marginBottom': '5px'})
            ],
            style={
                'position': 'absolute',
                'top': '10px',
                'right': '10px',
                'backgroundColor': 'white',
                'padding': '10px',
                'borderRadius': '5px',
                'boxShadow': '0 0 10px rgba(0,0,0,0.2)',
                'zIndex': 1000
            }
        )
        ]
    ),
    dcc.Loading(
        id='loading',
        type='circle',
        children=html.Div(id="loading-text", children="Processing file..."),
        style={
            'position': 'absolute',
            'top': 0,
            'left': 0,
            'width': '100%',
            'height': '100%',
            'backgroundColor': 'rgba(255, 255, 255, 0.5)',
            'display': 'flex',
            'justifyContent': 'center',
            'alignItems': 'center',
            'zIndex': 9999
        }
    )
])

# Callback to handle file upload, process data, and render map
@app.callback(
    [Output('file-info', 'children'),
     Output('inliers', 'children'),
     Output('outliers', 'children'),
     Output('start-end', 'children'),
     Output('ocean', 'children'),
     Output('concentrated-outliers', 'children'),
     Output('map', 'center'),  
     Output('map', 'zoom'),    
     Output('loading-text', 'children')],
    [Input('upload-mf4', 'contents')],
    [State('upload-mf4', 'filename')]
)
def update_output(file_contents, filename):
    if file_contents is None:
        return [], [], [], [], [], [], [20, 0], 2, "Upload a file to start."

    # Show loading circle while processing the file
    loading_text = "Processing file..."

    # Read and process the MF4 file
    content_type, content_string = file_contents.split(',')
    decoded = base64.b64decode(content_string)  # Decode the file content
    temp_filename = f"/tmp/{filename}"
    
    with open(temp_filename, 'wb') as temp_file:
        temp_file.write(decoded)

    # Load MF4 data
    mdf = asammdf.MDF(temp_filename)
    resampled = mdf.resample(raster=0.1)  # Resample data to 0.2 seconds
    df = resampled.to_dataframe()
    print(df.columns)

    # Clean and process data with stacking method for outlier detection
    df = resample_and_clean_data(df, land_data)

    # Map visualization
    start_point = None
    for i in range(len(df)):
        if not df.iloc[i]['stacked_outliers']:
            start_point = df.iloc[i]
            break

    end_point = None
    for i in range(len(df)-1, -1, -1):
        if not df.iloc[i]['stacked_outliers']:
            end_point = df.iloc[i]
            break

    # Define a distance threshold for long gaps (in meters)
    long_gap_threshold = 200  # Adjust this value as needed

    # Initialize lists to store inliers and outliers
    inliers_path = []
    inliers_markers = []
    outliers_markers = []
    start_marker = None
    end_marker = None
    ocean_markers = []
    concentrated_outliers_markers = []

    # Iterate over the points and draw lines between inliers that are not separated by a large gap
    for i in range(len(df) - 1):
        if not df.iloc[i]['stacked_outliers'] and not df.iloc[i + 1]['stacked_outliers']:
            # Calculate the distance to the next inlier
            distance = calculate_distance(df.iloc[i]['LATITUDE'], df.iloc[i]['LONGITUDE'],
                                          df.iloc[i + 1]['LATITUDE'], df.iloc[i + 1]['LONGITUDE'])
            if distance <= long_gap_threshold:
                # Draw a line between the two inliers
                inliers_path.append(dl.Polyline(
                    positions=[[df.iloc[i]['LATITUDE'], df.iloc[i]['LONGITUDE']], [df.iloc[i + 1]['LATITUDE'], df.iloc[i + 1]['LONGITUDE']]],
                    color='blue',
                    weight=1  
                ))
        if not df.iloc[i]['stacked_outliers']:
            inliers_markers.append(dl.CircleMarker(center=(df.iloc[i]['LATITUDE'], df.iloc[i]['LONGITUDE']), radius=1, color='blue', fillOpacity=0.7))
        if df.iloc[i]['stacked_outliers']:
            outliers_markers.append(dl.CircleMarker(center=(df.iloc[i]['LATITUDE'], df.iloc[i]['LONGITUDE']), radius=1, color='red', fillOpacity=0.7))
        if df.iloc[i]['in_land'] == False:
            ocean_markers.append(dl.CircleMarker(center=(df.iloc[i]['LATITUDE'], df.iloc[i]['LONGITUDE']), radius=2, color='cyan', fillOpacity=1.0))
        if df.iloc[i]['concentrated_outliers']:
            concentrated_outliers_markers.append(dl.CircleMarker(center=(df.iloc[i]['LATITUDE'], df.iloc[i]['LONGITUDE']), radius=3, color='purple', fillOpacity=0.7))
#################long gap logic end

    # Add start point
    start_marker = dl.CircleMarker(
        center=(start_point['LATITUDE'], start_point['LONGITUDE']),
        radius=3,  
        color='green',
        fillOpacity=1.0
    )

    # Add end point
    end_marker = dl.CircleMarker(
        center=(end_point['LATITUDE'], end_point['LONGITUDE']),
        radius=3,  
        color='orange',
        fillOpacity=1.0
    )

    return (
        f"File '{filename}' processed successfully.",
        inliers_path + inliers_markers,
        outliers_markers,
        [start_marker, end_marker],
        ocean_markers,
        concentrated_outliers_markers,
        [df['LATITUDE'].mean(), df['LONGITUDE'].mean()],
        5,
        ""
    )

if __name__ == '__main__':
    app.run_server(debug=True, port=8055)
