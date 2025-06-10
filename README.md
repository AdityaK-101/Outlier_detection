# GPS Outlier Detection and Visualization

## Overview

This project provides tools for processing GPS track data from MF4 files. It aims to identify and remove outliers from the GPS data, clean the path, and visualize the results. The process involves several stages of data filtering, feature engineering, outlier detection using statistical methods and machine learning, and finally, path smoothing and visualization on a map.

Two main Python scripts are provided:
*   A Dash-based web application (`gps_outlier_detection_leaflet_v2.py`) for interactive processing and visualization.
*   A Tkinter-based GUI application (`gps_outlier_gui_folium.py`), which also supports command-line operations, for batch processing and generating HTML map reports.

## Features

*   **MF4 File Processing:** Directly ingests GPS data from `.mf4` log files.
*   **Comprehensive Outlier Detection:**
    *   Uses statistical methods like Z-score on distance and bearing.
    *   Employs machine learning models like Isolation Forest.
    *   Applies rule-based heuristics for U-turns, zigzag patterns, and excessive angle changes.
    *   Identifies and flags concentrated static points using DBSCAN.
*   **Ocean Filtering:** Removes GPS points located in oceans/seas using geographical boundary data.
*   **Stacking Ensemble Model:** Combines results from various outlier detection techniques using a Random Forest classifier for more robust decision-making.
*   **Path Smoothing & Interpolation:** Applies adaptive smoothing to the cleaned path and interpolates short gaps.
*   **Interactive Visualization (Dash/Leaflet):**
    *   Upload MF4 files through a web interface.
    *   View inliers, outliers, start/end points, and points identified in the ocean on an interactive map.
*   **GUI & Batch Processing (Tkinter/Folium):**
    *   User-friendly GUI for selecting multiple MF4 files.
    *   Command-line interface for automated batch processing.
    *   Generates an HTML file with a Folium map displaying processed tracks.
*   **Data Resampling:** Resamples time-series data to a consistent frequency.

## Algorithm Explanation

The core of this project lies in its multi-stage approach to cleaning GPS data. Here's a breakdown of the key algorithms and techniques:

### 1. Data Preprocessing & Initial Filtering
*   **MF4 Data Loading:** GPS signals (primarily Latitude, Longitude, Speed Over Ground) are extracted from the input MF4 files.
*   **Resampling:** The raw GPS data is resampled to a fixed frequency (e.g., 0.1 seconds) to ensure consistent time intervals between data points. This is crucial for accurate calculation of speed, acceleration, and bearing.
*   **Ocean Filtering:** Before intensive outlier detection, points are checked against a land boundary dataset (`ne_110m_admin_0_countries.shp`). Any points falling into oceans or large bodies of water are typically flagged or removed early in the process, as these often represent significant GPS errors or periods where the tracked asset is on a vessel.

### 2. Feature Engineering
Based on the resampled GPS coordinates and timestamps, several key features are calculated:
*   **Distance:** The geodesic distance between consecutive GPS points. Abnormally large distances can indicate jumps or signal loss.
*   **Speed:** Calculated from distance and time difference. Unrealistic speeds are strong indicators of outliers.
*   **Bearing (Course):** The direction of travel between consecutive points. Sudden, erratic changes in bearing can highlight outliers.
*   **Bearing Difference:** The change in bearing from one segment to the next. This helps identify sharp turns or erratic movements.
*   **Angle Calculation:** The angle formed by three consecutive points is used to detect sharp zigzags or unrealistic turns.

### 3. Outlier Detection Methods
A combination of methods is used to identify potential outliers:

*   **Z-score Analysis:**
    *   Applied to features like `distance` and `bearing_diff`.
    *   Points with Z-scores exceeding predefined thresholds (e.g., > 3.5 for bearing difference, > 8.5 for distance) are flagged as potential outliers. This method is effective for catching extreme deviations from the mean.

*   **Isolation Forest:**
    *   An unsupervised machine learning algorithm particularly effective for anomaly detection.
    *   It works by randomly partitioning the data and isolating instances. Outliers, being "few and different," are typically isolated in fewer partitions.
    *   Features like Latitude, Longitude, Speed, calculated distance, and bearing difference are scaled and fed into the Isolation Forest model.
    *   A `contamination` parameter is set to estimate the proportion of outliers in the dataset.

*   **Rule-Based Heuristics:**
    *   **U-Turn Detection:** Identifies sequences of points where the bearing changes by a large amount (e.g., > 170 degrees) over a short interval,czych indicating an invalid sharp turn or GPS reflection.
    *   **Zigzag/Sharp Turn Detection:** Flags points where the angle formed by three consecutive points is very acute (e.g., > 150 degrees), suggesting rapid, unrealistic side-to-side movements.
    *   **Small Angle Filtering for Isolation Forest:** Points involved in very slight turns (angle <= 2 degrees) are less likely to be outliers, and this rule can override an initial Isolation Forest outlier flag for such points.
    *   **Logical Inliers:** Points with very small travel distances and minimal bearing changes are explicitly marked as inliers to prevent them from being incorrectly flagged by more sensitive methods.

*   **Concentrated Cluster Detection (DBSCAN):**
    *   The DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm is used to find areas where GPS points are highly concentrated (e.g., when a vehicle is stationary for an extended period).
    *   While not always outliers in the traditional sense, these clusters (especially at the start/end of trips or unexpected locations) can sometimes be treated as non-representative of the main travel path or indicative of GPS drift while stationary. The scripts mark points not belonging to dense clusters as `concentrated_outliers`.
    *   The `eps` (maximum distance between samples for one to be considered as in the neighborhood of the other) and `min_samples` (number of samples in a neighborhood for a point to be considered as a core point) parameters are crucial for DBSCAN's performance. The `gps_outlier_gui_folium.py` script includes logic to estimate `eps`.

*   **Start/End Point Handling:**
    *   A fixed number of points at the beginning and end of a track (e.g., first 10 and last 10 points) are often marked as outliers. This is a heuristic to handle potential GPS inaccuracies or irrelevant data that can occur when a recording device is turned on/off or is acquiring an initial fix.

### 4. Stacking Ensemble Model
To improve the robustness and accuracy of outlier detection, a stacking ensemble method is employed:
*   The predictions from the Z-score method and the (rule-adjusted) Isolation Forest (`zscore_outliers`, `final_outliers`) are used as input features for a meta-classifier.
*   A **Random Forest Classifier** is typically used as this meta-classifier. It learns from the outputs of the base detectors to make a final decision on whether a point is an outlier.
*   The `class_weight='balanced'` parameter is often used to handle the typically imbalanced nature of outlier vs. inlier data.
*   The final prediction from this stacking model (`stacked_outliers`) provides a more nuanced classification.

### 5. Path Smoothing and Interpolation
After outliers identified by the stacking model are flagged:
*   **Adaptive Smoothing/Interpolation:** For points marked as `stacked_outliers`, an attempt is made to interpolate their position based on the nearest valid preceding and succeeding inlier points.
*   **Gap Handling:** If the gap between valid points (after removing an outlier) is too large (e.g., > 50-500 meters, configurable), interpolation might be skipped for that segment to avoid creating unrealistic straight lines over long distances. The `adaptive_smooth_path` function handles this. The smoothed coordinates replace the original outlier coordinates.

This layered approach, combining statistical measures, machine learning, and domain-specific rules, aims to provide a flexible and effective way to clean GPS trajectory data.

## Scripts Description

*   **`gps_outlier_detection_leaflet_v2.py`**:
    *   A web application built using Dash and Plotly.
    *   Provides an interactive interface where users can upload an MF4 file.
    *   Displays the processed GPS track on a Leaflet map, color-coding inliers, outliers, start/end points, and points detected in the ocean.
    *   Suitable for detailed inspection of individual files.

*   **`gps_outlier_gui_folium.py`**:
    *   A desktop application built with Tkinter for the GUI.
    *   Also supports command-line operations for batch processing.
    *   Allows users to select one or more MF4 files.
    *   Processes the files and generates an HTML report with an interactive Folium map for each, showing similar information to the Dash application (inliers, outliers, etc.).
    *   Optimized for processing multiple files and includes features like chunked processing for larger datasets.

## Dependencies

To run these scripts, you will need Python 3 and the following libraries:

*   `asammdf`: For reading MF4 files.
*   `pandas`: For data manipulation and analysis.
*   `numpy`: For numerical operations.
*   `dash`: For the web application framework.
*   `dash_leaflet`: For Leaflet map integration in Dash.
*   `geopy`: For geodesic distance calculations.
*   `scikit-learn`: For machine learning algorithms (IsolationForest, RandomForestClassifier, StandardScaler, DBSCAN).
*   `geopandas`: For working with geospatial data (reading shapefiles, spatial operations).
*   `shapely`: For geometric objects and operations.
*   `folium`: For creating HTML map visualizations.
*   `tkinter`: (Usually part of standard Python) For the GUI in `gps_outlier_gui_folium.py`.

You will also need a shapefile for land boundary data:
*   **`ne_110m_admin_0_countries.shp`** (and its associated files like `.dbf`, `.shx`, etc.). This can be downloaded from [Natural Earth](https://www.naturalearthdata.com/downloads/110m-cultural-vectors/110m-admin-0-countries/).

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Install Python Dependencies:**
    It is recommended to use a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
    Install the required libraries using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download Geographic Data:**
    *   Download the "Admin 0 – Countries" shapefile (110m scale) from [Natural Earth](https://www.naturalearthdata.com/downloads/110m-cultural-vectors/110m-admin-0-countries/).
    *   Extract the contents of the downloaded zip file.
    *   Ensure that `ne_110m_admin_0_countries.shp` and its related files (e.g., `ne_110m_admin_0_countries.dbf`, `ne_110m_admin_0_countries.shx`, `ne_110m_admin_0_countries.prj`) are placed in the root directory of this project.

## Usage Instructions

### `gps_outlier_detection_leaflet_v2.py` (Dash Web Application)

1.  Navigate to the project directory.
2.  Run the script:
    ```bash
    python gps_outlier_detection_leaflet_v2.py
    ```
3.  Open your web browser and go to the address displayed in the terminal (usually `http://127.0.0.1:8055/`).
4.  Click the "Upload MF4 File" button to select and upload your `.mf4` file.
5.  The map will display the processed GPS data with inliers, outliers, and other identified points. A legend is provided on the map.

### `gps_outlier_gui_folium.py` (Tkinter GUI and CLI)

#### GUI Mode:

1.  Navigate to the project directory.
2.  Run the script without any command-line arguments:
    ```bash
    python gps_outlier_gui_folium.py
    ```
3.  The File Selection GUI will open.
    *   Click "Add Files" to select one or more `.mf4` files.
    *   You can view details of selected files, remove files, or clear the list.
    *   Specify an "Output Directory" (default is `maps`).
    *   Click "Process Files".
4.  An HTML map file will be generated in the specified output directory for the processed data. You'll be prompted to open it.

#### Command-Line Interface (CLI) Mode:

1.  Navigate to the project directory.
2.  Run the script with file paths as arguments:
    ```bash
    python gps_outlier_gui_folium.py path/to/your/file1.mf4 path/to/your/file2.mf4
    ```
3.  You can also specify an output directory:
    ```bash
    python gps_outlier_gui_folium.py your_file.mf4 --output-dir custom_maps_directory
    ```
4.  HTML map files will be generated in the specified output directory (or `maps` by default).

## Input Data

*   **MF4 Files:** The primary input is `.mf4` files, which are standard log files in the automotive industry. These files should contain GPS channels, specifically:
    *   `LATITUDE`
    *   `LONGITUDE`
    *   `SPEED_OVER_GROUND`
    *   (Optionally `COURSE_OVER_GROUND`, `ALTITUDE` if available and used by specific configurations)
*   **CSV/Readme:** The `CSV/` directory contains a `Readme` file mentioning "SCV XENON files". This might refer to a specific type or source of MF4 files that were used for testing or development.

## Output

*   **`gps_outlier_detection_leaflet_v2.py`**:
    *   An interactive map displayed in the web browser, showing the GPS track with:
        *   Blue lines/markers for inlier points.
        *   Red markers for outlier points.
        *   Green marker for the start point of the cleaned track.
        *   Orange marker for the end point of the cleaned track.
        *   Cyan markers for points identified as being in the ocean.
        *   (Potentially other categories like concentrated outliers, depending on exact script version).

*   **`gps_outlier_gui_folium.py`**:
    *   One or more HTML files (e.g., `gps_visualization_YYYYMMDD_HHMMSS.html`) saved in the specified output directory (default: `maps/`).
    *   Each HTML file contains an interactive Folium map with a similar visualization style to the Dash application (inliers, outliers, start/end points, ocean points).

## Customization/Configuration

While the scripts are designed to work well with default parameters for many common GPS datasets, you might find the need to adjust certain settings for optimal performance on your specific data:

*   **Outlier Detection Thresholds:**
    *   In both scripts (e.g., in `enhanced_outlier_detection` function):
        *   Z-score thresholds for `distance` and `bearing_diff`.
        *   `IsolationForest` parameters: `n_estimators`, `contamination`.
        *   Angle thresholds for U-turn and zigzag detection.
        *   Distance and bearing thresholds for `logical_inliers`.
*   **DBSCAN Parameters:**
    *   In `identify_concentrated_clusters` and `process_mf4_file` (in `gps_outlier_gui_folium.py`):
        *   `eps` (epsilon): The maximum distance between two samples for one to be considered as in the neighborhood of the other. The GUI script has a function `estimate_eps` to help with this.
        *   `min_samples`: The number of samples in a neighborhood for a point to be considered as a core point.
*   **Path Smoothing & Interpolation:**
    *   In `adaptive_smooth_path`:
        *   `long_gap_threshold`: The maximum distance (in meters) for an outlier point to be interpolated. Gaps larger than this will not be filled.
*   **Start/End Cluster Marking:**
    *   In `mark_start_end_clusters`: The number of points at the beginning and end of the track to be marked as outliers (e.g., `idx < 10` or `idx > len(df) - 10`).
*   **Map Settings:**
    *   Default zoom, center, tile layers in both Dash and Folium map initializations.
    *   Scroll wheel zoom sensitivity in the Dash/Leaflet map.

## Troubleshooting

*   **Shapefile Not Found Error:**
    *   **Error Message:** Typically `FileNotFoundError` or an error from `geopandas.read_file()` indicating `ne_110m_admin_0_countries.shp` (or related files like `.dbf`, `.shx`) cannot be found.
    *   **Solution:** Ensure you have downloaded the "Admin 0 – Countries" shapefile (110m scale) from [Natural Earth](https://www.naturalearthdata.com/downloads/110m-cultural-vectors/110m-admin-0-countries/). Extract all files from the zip archive and place them in the root directory of the project. The required files include `.shp`, `.shx`, `.dbf`, and `.prj`.

*   **MF4 File Issues:**
    *   **Error Message:** Errors from `asammdf.MDF()` or when accessing specific channels.
    *   **Solution:**
        *   Verify the MF4 file is not corrupted and is a valid file.
        *   Ensure the file contains the expected GPS channels (e.g., `LATITUDE`, `LONGITUDE`, `SPEED_OVER_GROUND`). Channel names are case-sensitive.
        *   Some MF4 files might have variations in channel naming or structure. You may need to inspect the file with a dedicated MF4 tool to confirm channel names.

*   **Performance Issues with Large Files (especially `gps_outlier_gui_folium.py`):**
    *   **Symptom:** The script runs very slowly or consumes excessive memory.
    *   **Solution:**
        *   The `gps_outlier_gui_folium.py` script includes chunked processing and other optimizations. However, for extremely large files, consider:
            *   Increasing the `chunk_size` in `process_mf4_file` if memory allows, or decreasing it if memory is constrained.
            *   Ensure you have sufficient RAM.
            *   Processing files one by one if batch processing is too demanding.

*   **Dependency Installation Errors:**
    *   **Symptom:** `pip install -r requirements.txt` fails for one or more packages.
    *   **Solution:**
        *   Ensure you have Python 3 and pip installed correctly.
        *   Some packages (especially `geopandas` and its dependencies like `fiona`, `pyproj`, `rtree`) can have complex system-level dependencies (like GEOS, PROJ, GDAL). If you encounter issues, it's often helpful to consult the official installation guides for these packages or try installing them via a system package manager or a distribution like Conda which can manage these binary dependencies more easily. For example: `conda install geopandas`.

## Contributing

(Placeholder: Add guidelines for contributing to the project, if applicable. E.g., fork the repository, create a new branch, submit a pull request.)

## License

(Placeholder: Add license information. E.g., MIT License, Apache 2.0. If no license is chosen, you might state "All rights reserved" or similar.)
