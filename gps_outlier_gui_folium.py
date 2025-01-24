import base64
import asammdf
import pandas as pd
import numpy as np
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
import folium
from folium import plugins
import tempfile
import os
import argparse
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, ttk
from tkinter import messagebox
import sys
import logging
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial import cKDTree

# Function to load land boundary data
def load_land_data():
    shapefile_path = "ne_110m_admin_0_countries.shp"
    try:
        world = gpd.read_file(shapefile_path)
        print(f"Shapefile read successfully. Found {len(world)} countries.")
    except Exception as e:
        print(f"Error reading shapefile: {e}")
    return world

# Function to check if a point is in land (optimized version)
def is_in_land(lat, lon, land_data):
    point = Point(lon, lat)
    return any(geom.contains(point) for geom in land_data.geometry)

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
    long_gap_threshold = 500  # Adjust this value as needed

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
    logging.info(f"Starting data processing with {len(df)} points")
    df = df.reset_index(drop=True)

    # Create numpy arrays for faster processing
    coords = df[['LATITUDE', 'LONGITUDE']].values
    
    # Batch process land check
    logging.info("Checking land points")
    in_land = np.array([is_in_land(lat, lon, land_data) for lat, lon in coords])
    df = df[in_land]
    logging.info(f"Remaining points after land check: {len(df)}")

    # Apply improved stacked outlier detection
    logging.info("Applying outlier detection")
    df = improved_stack_outlier_detection(df)

    # Mark start and end clusters as outliers
    df = mark_start_end_clusters(df)

    # Identify concentrated clusters with optimized parameters
    gps_data = df[['LATITUDE', 'LONGITUDE']].values
    concentrated_labels = identify_concentrated_clusters(
        gps_data,
        eps=0.001,  # Adjusted for better clustering
        min_samples=5  # Reduced for faster processing
    )
    df['concentrated_outliers'] = concentrated_labels == -1

    # Apply adaptive smoothing
    logging.info("Applying path smoothing")
    df = adaptive_smooth_path(df)
    
    logging.info(f"Processing completed. Final points: {len(df)}")
    return df

def chunk_processor(chunk_data, land_data):
    """Process a chunk of GPS data"""
    # Convert to numpy for faster processing
    coords = chunk_data[['LATITUDE', 'LONGITUDE']].values
    
    # Use KDTree for faster spatial queries
    land_coords = np.array([(geom.centroid.y, geom.centroid.x) for geom in land_data.geometry])
    tree = cKDTree(land_coords)
    
    # Find nearest land points
    distances, _ = tree.query(coords)
    in_land = distances < 0.1  # Threshold distance to land
    
    return chunk_data[in_land]

def process_mf4_file(file_path, land_data, chunk_size=10000):
    """Process a single MF4 file with chunked processing"""
    logging.info(f"Processing file: {file_path}")
    
    try:
        # Load MF4 file with specific channels only
        mdf = asammdf.MDF(file_path)
        required_cols = ['LATITUDE', 'LONGITUDE', 'SPEED_OVER_GROUND', 
                        'COURSE_OVER_GROUND', 'ALTITUDE']
        
        # Convert to dataframe with only required columns
        df = mdf.to_dataframe(channels=required_cols)
        del mdf  # Free up memory
        
        # Remove NaN values early
        df = df.dropna()
        logging.info(f"Initial data points: {len(df)}")
        
        # Process in chunks using parallel processing
        chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
        logging.info(f"Processing {len(chunks)} chunks")
        
        # Use ThreadPoolExecutor for I/O-bound operations
        processed_chunks = []
        with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
            futures = [executor.submit(chunk_processor, chunk, land_data) 
                      for chunk in chunks]
            for future in futures:
                result = future.result()
                if len(result) > 0:
                    processed_chunks.append(result)
        
        # Combine processed chunks
        if processed_chunks:
            df = pd.concat(processed_chunks, ignore_index=True)
        else:
            return pd.DataFrame()  # Return empty df if no valid points
        
        logging.info(f"Points after initial processing: {len(df)}")
        
        # Optimize outlier detection parameters
        if len(df) > 1000:
            sample_size = min(1000, len(df))
            sample_df = df.sample(n=sample_size)
            eps = estimate_eps(sample_df[['LATITUDE', 'LONGITUDE']].values)
        else:
            eps = 0.001
        
        # Final processing with optimized parameters
        gps_data = df[['LATITUDE', 'LONGITUDE']].values
        concentrated_labels = identify_concentrated_clusters(
            gps_data,
            eps=eps,
            min_samples=3,  # Reduced for faster processing
            algorithm='ball_tree'  # More efficient for geographic coordinates
        )
        df['concentrated_outliers'] = concentrated_labels == -1
        
        # Quick smoothing for large datasets
        if len(df) > 10000:
            df = df.rolling(window=3, center=True).mean()
        else:
            df = adaptive_smooth_path(df)
        
        logging.info(f"Final processed points: {len(df)}")
        return df
        
    except Exception as e:
        logging.error(f"Error processing file: {str(e)}")
        return pd.DataFrame()

def estimate_eps(points):
    """Estimate optimal eps parameter for DBSCAN"""
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=2)  # Find distance to nearest neighbor
    return np.percentile(distances[:, 1], 95) * 1.5  # Use 95th percentile with scaling

def create_map_visualization(file_paths, output_dir="maps"):
    """Create an interactive map visualization for multiple MF4 files"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load land data
    land_data = load_land_data()
    
    # Initialize lists to store all coordinates
    all_latitudes = []
    all_longitudes = []
    
    # Create a map
    m = folium.Map(location=[20, 0], zoom_start=2, max_zoom=30)
    
    # Create feature groups for different types of points
    inliers_group = folium.FeatureGroup(name='Inliers')
    outliers_group = folium.FeatureGroup(name='Outliers')
    ocean_group = folium.FeatureGroup(name='Ocean Points')
    start_end_group = folium.FeatureGroup(name='Start/End Points')
    
    # Process each file
    for file_path in file_paths:
        try:
            print(f"Processing {os.path.basename(file_path)}...")
            df = process_mf4_file(file_path, land_data)
            
            # Store coordinates for centering
            valid_points = df[~df['stacked_outliers']]
            all_latitudes.extend(valid_points['LATITUDE'].tolist())
            all_longitudes.extend(valid_points['LONGITUDE'].tolist())
            
            # Create line for inlier points
            current_line = []
            last_inlier = None
            
            for idx, row in df.iterrows():
                if not row['stacked_outliers']:
                    current_point = [row['LATITUDE'], row['LONGITUDE']]
                    
                    # If we have a previous inlier point
                    if last_inlier is not None:
                        # Calculate distance to last inlier
                        distance = calculate_distance(
                            last_inlier[0], last_inlier[1],
                            current_point[0], current_point[1]
                        )
                        
                        # If distance is within threshold, connect the points
                        if distance <= 100:  # Adjust threshold as needed
                            if not current_line:  # If starting new line, add last_inlier first
                                current_line.append(last_inlier)
                            current_line.append(current_point)
                        else:
                            # If we have a line and distance exceeds threshold, add it to map
                            if len(current_line) > 1:
                                folium.PolyLine(
                                    current_line,
                                    color='blue',
                                    weight=2,
                                    opacity=0.8
                                ).add_to(inliers_group)
                            # Start new line with current point
                            current_line = [current_point]
                    else:
                        # First inlier point
                        current_line = [current_point]
                    
                    # Update last_inlier
                    last_inlier = current_point
                    
                    # Add inlier marker
                    folium.CircleMarker(
                        location=current_point,
                        radius=1,
                        color='blue',
                        fill=True,
                        fillOpacity=0.7
                    ).add_to(inliers_group)
                else:
                    # Add outlier marker
                    folium.CircleMarker(
                        location=[row['LATITUDE'], row['LONGITUDE']],
                        radius=1,
                        color='red',
                        fill=True,
                        fillOpacity=0.7
                    ).add_to(outliers_group)
                
                # Add ocean points
                if not row['in_land']:
                    folium.CircleMarker(
                        location=[row['LATITUDE'], row['LONGITUDE']],
                        radius=3,
                        color='cyan',
                        fill=True,
                        fillOpacity=1.0
                    ).add_to(ocean_group)
            
            # Add any remaining line
            if len(current_line) > 1:
                folium.PolyLine(
                    current_line,
                    color='blue',
                    weight=2,
                    opacity=0.8
                ).add_to(inliers_group)
            
            # Add start and end points
            start_point = None
            for idx, row in df.iterrows():
                if not row['stacked_outliers']:
                    start_point = row
                    break
            
            end_point = None
            for idx, row in df.iloc[::-1].iterrows():
                if not row['stacked_outliers']:
                    end_point = row
                    break
            
            if start_point is not None:
                folium.CircleMarker(
                    location=[start_point['LATITUDE'], start_point['LONGITUDE']],
                    radius=5,
                    color='green',
                    fill=True,
                    fillOpacity=1.0,
                    popup=f"Start - {os.path.basename(file_path)}"
                ).add_to(start_end_group)
            
            if end_point is not None:
                folium.CircleMarker(
                    location=[end_point['LATITUDE'], end_point['LONGITUDE']],
                    radius=5,
                    color='orange',
                    fill=True,
                    fillOpacity=1.0,
                    popup=f"End - {os.path.basename(file_path)}"
                ).add_to(start_end_group)
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
    
    # Add all feature groups to the map
    inliers_group.add_to(m)
    outliers_group.add_to(m)
    ocean_group.add_to(m)
    start_end_group.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Center the map if we have coordinates
    if all_latitudes and all_longitudes:
        m.fit_bounds([[min(all_latitudes), min(all_longitudes)], 
                     [max(all_latitudes), max(all_longitudes)]])
    
    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"gps_visualization_{timestamp}.html")
    
    # Save the map
    m.save(output_file)
    print(f"Map saved to: {output_file}")
    return output_file

class FileSelectionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("GPS Data Visualization")
        self.root.geometry("1000x700")
        
        # Set style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Create left panel for file list
        left_panel = ttk.Frame(main_frame)
        left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        left_panel.columnconfigure(0, weight=1)
        left_panel.rowconfigure(1, weight=1)
        
        # File list label
        ttk.Label(left_panel, text="Selected Files:", font=('TkDefaultFont', 10, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        # Create listbox with scrollbar for selected files
        self.file_frame = ttk.Frame(left_panel)
        self.file_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.file_frame.columnconfigure(0, weight=1)
        self.file_frame.rowconfigure(0, weight=1)
        
        self.file_listbox = tk.Listbox(self.file_frame, selectmode=tk.MULTIPLE, height=15, font=('TkDefaultFont', 9))
        self.file_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.file_listbox.bind('<<ListboxSelect>>', self.on_select_file)
        
        scrollbar = ttk.Scrollbar(self.file_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.file_listbox.configure(yscrollcommand=scrollbar.set)
        
        # Create right panel for file details
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_panel.columnconfigure(0, weight=1)
        
        # File details frame
        details_frame = ttk.LabelFrame(right_panel, text="File Details", padding="5")
        details_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        details_frame.columnconfigure(1, weight=1)
        
        # File details labels
        self.file_name_var = tk.StringVar(value="")
        self.file_size_var = tk.StringVar(value="")
        self.file_date_var = tk.StringVar(value="")
        
        ttk.Label(details_frame, text="Name:", width=10).grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Label(details_frame, textvariable=self.file_name_var).grid(row=0, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(details_frame, text="Size:", width=10).grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Label(details_frame, textvariable=self.file_size_var).grid(row=1, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(details_frame, text="Modified:", width=10).grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Label(details_frame, textvariable=self.file_date_var).grid(row=2, column=1, sticky=tk.W, pady=2)
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        # Add buttons with icons (using Unicode characters as simple icons)
        ttk.Button(button_frame, text=" Add Files", command=self.add_files).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text=" Remove Selected", command=self.remove_selected).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text=" Clear All", command=self.clear_all).grid(row=0, column=2, padx=5)
        
        # Output directory selection
        dir_frame = ttk.LabelFrame(main_frame, text="Output Settings", padding="5")
        dir_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        dir_frame.columnconfigure(1, weight=1)
        
        ttk.Label(dir_frame, text="Save to:").grid(row=0, column=0, padx=5)
        self.output_dir = tk.StringVar(value="maps")
        self.dir_entry = ttk.Entry(dir_frame, textvariable=self.output_dir)
        self.dir_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(dir_frame, text=" Browse", command=self.select_output_dir).grid(row=0, column=2, padx=5)
        
        # Process button with progress info
        process_frame = ttk.Frame(main_frame)
        process_frame.grid(row=3, column=0, columnspan=2, pady=20, sticky=(tk.W, tk.E))
        process_frame.columnconfigure(0, weight=1)
        
        self.process_button = ttk.Button(process_frame, text=" Process Files", command=self.process_files, style='Accent.TButton')
        self.process_button.grid(row=0, column=0, pady=(0, 10))
        
        # Progress frame
        progress_frame = ttk.Frame(process_frame)
        progress_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        progress_frame.columnconfigure(0, weight=1)
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(progress_frame, textvariable=self.status_var)
        self.status_label.grid(row=0, column=0, pady=(0, 5))
        
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(progress_frame, length=300, mode='determinate', variable=self.progress_var)
        self.progress.grid(row=1, column=0)
        
        # Store selected files
        self.selected_files = []
        
        # Create accent style for process button
        style.configure('Accent.TButton', font=('TkDefaultFont', 10, 'bold'))
        
    def on_select_file(self, event):
        selection = self.file_listbox.curselection()
        if selection:
            index = selection[0]
            file_path = self.selected_files[index]
            
            # Update file details
            file_stat = os.stat(file_path)
            self.file_name_var.set(os.path.basename(file_path))
            self.file_size_var.set(f"{file_stat.st_size / (1024*1024):.2f} MB")
            self.file_date_var.set(datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'))
        else:
            self.file_name_var.set("")
            self.file_size_var.set("")
            self.file_date_var.set("")
    
    def add_files(self):
        files = filedialog.askopenfilenames(
            title="Select MF4 Files",
            filetypes=[("MF4 files", "*.mf4"), ("All files", "*.*")]
        )
        for file in files:
            if file not in self.selected_files:
                self.selected_files.append(file)
                self.file_listbox.insert(tk.END, os.path.basename(file))
        
        # Select first file if this is the first addition
        if self.selected_files and not self.file_listbox.curselection():
            self.file_listbox.selection_set(0)
            self.on_select_file(None)
    
    def remove_selected(self):
        selected_indices = self.file_listbox.curselection()
        for index in reversed(selected_indices):
            self.file_listbox.delete(index)
            self.selected_files.pop(index)
        
        # Clear file details if no files are selected
        if not self.selected_files:
            self.file_name_var.set("")
            self.file_size_var.set("")
            self.file_date_var.set("")
    
    def clear_all(self):
        self.file_listbox.delete(0, tk.END)
        self.selected_files.clear()
        self.file_name_var.set("")
        self.file_size_var.set("")
        self.file_date_var.set("")
    
    def select_output_dir(self):
        directory = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=self.output_dir.get()
        )
        if directory:
            self.output_dir.set(directory)
    
    def process_files(self):
        if not self.selected_files:
            messagebox.showerror("Error", "Please select at least one file to process.")
            return
        
        # Disable buttons during processing
        self.process_button.state(['disabled'])
        self.progress_var.set(0)
        self.status_var.set("Processing...")
        self.root.update()  # Refresh GUI immediately
        
        try:
            total_files = len(self.selected_files)
            for i, file in enumerate(self.selected_files, 1):
                self.status_var.set(f"Processing {os.path.basename(file)}...")
                self.progress_var.set((i / total_files) * 100)
                self.root.update()  # Refresh GUI during processing
                
                # Create visualization
                output_file = create_map_visualization(self.selected_files, self.output_dir.get())
            
            self.status_var.set("Processing complete!")
            self.progress_var.set(100)
            
            # Show success message and ask to open the file
            if messagebox.askyesno("Success", 
                                 f"Map has been generated successfully!\n\nWould you like to open it now?"):
                import webbrowser
                webbrowser.open(f"file://{os.path.abspath(output_file)}")
        
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while processing files:\n{str(e)}")
            self.status_var.set("Error occurred during processing")
        
        finally:
            # Re-enable buttons
            self.process_button.state(['!disabled'])
    
    def run(self):
        # Center the window on screen
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        
        self.root.mainloop()

if __name__ == '__main__':
    # Check if command line arguments are provided
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description='Process MF4 files and create GPS visualization')
        parser.add_argument('files', nargs='+', help='MF4 files to process')
        parser.add_argument('--output-dir', default='maps', help='Output directory for the generated maps')
        
        args = parser.parse_args()
        
        # Create the visualization
        output_file = create_map_visualization(args.files, args.output_dir)
        
        # Try to open the map in the default browser
        try:
            import webbrowser
            webbrowser.open(f"file://{os.path.abspath(output_file)}")
        except Exception as e:
            print(f"Could not automatically open the map: {str(e)}")
            print(f"Please open {output_file} in your web browser manually.")
    else:
        # Launch GUI if no command line arguments
        app = FileSelectionGUI()
        app.run()
