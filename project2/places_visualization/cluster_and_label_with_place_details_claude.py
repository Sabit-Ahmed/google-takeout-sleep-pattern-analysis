import json
import pandas as pd
import numpy as np
import requests
import folium
from folium.plugins import MarkerCluster
from branca.element import Template, MacroElement
import re
from datetime import datetime, timedelta, timezone
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure requests with a session for better connection handling
session = requests.Session()
retries = Retry(
    total=5,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)
session.mount("https://", HTTPAdapter(max_retries=retries))
session.mount("http://", HTTPAdapter(max_retries=retries))

GOOGLE_API_KEY = ""
DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"
NEARBY_SEARCH_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

CATEGORY_MAP = {
    "Residential": ["locality", "political", "neighborhood", "sublocality", "premise"],
    "Restaurant": ["restaurant", "cafe", "food"],
    "Shopping & Grocery": ["store", "shopping_mall", "department_store", "clothing_store",
                           "grocery_or_supermarket", "convenience_store", "supermarket"],
    "Academic": ["university", "school", "library"],
    "Travel": ["airport", "train_station", "transit_station", "lodging", "farm", "street_address"],
    "Spiritual": ["church", "mosque", "synagogue", "hindu_temple", "place_of_worship"],
    "Government Offices": ["local_government_office", "post_office", "courthouse", "city_hall"],
    "Commute": ["bus_station", "subway_station", "parking"],
    "Historical": ["museum", "cemetery", "tourist_attraction"],
    "Financial": ["bank", "atm", "insurance_agency", "finance"],
    "Auto & Gas": ["car_repair", "car_dealer", "car_wash", "gas_station"],
    "Medical": ["hospital", "doctor", "pharmacy", "dentist"],
    "Park & Entertainment": ["park", "natural_feature", "campground", "movie_theater", "night_club", "bar",
                             "amusement_park", "zoo"]
}

CATEGORY_COLORS = {
    "Residential": "gray",
    "Restaurant": "red",
    "Shopping & Grocery": "orange",
    "Academic": "green",
    "Travel": "blue",
    "Spiritual": "purple",
    "Government Offices": "cadetblue",
    "Commute": "darkred",
    "Historical": "lightgray",
    "Financial": "black",
    "Auto & Gas": "brown",
    "Medical": "lightblue",
    "Park & Entertainment": "gold",
    "Work": "teal",
    "Home": "darkblue",
    "Other": "lightgray"
}


def map_place_types_to_category(types):
    if not types:
        return "Other"
    for t in types:
        for category, keywords in CATEGORY_MAP.items():
            if t in keywords:
                return category
    for t in types:
        for category, keywords in CATEGORY_MAP.items():
            if any(kw in t for kw in keywords):
                return category
    return "Other"


def extract_coords(geo_str):
    match = re.match(r"geo:([-0-9.]+),([-0-9.]+)", geo_str)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None


def calculate_duration_minutes(start_time, end_time):
    """Calculate duration in minutes between start and end time."""
    if not start_time or not end_time:
        return 0

    try:
        # Handle various timestamp formats
        try:
            # Try standard ISO format with timezone offset (like in the sample)
            start = datetime.fromisoformat(start_time)
        except:
            # Try with Z (UTC) format
            start = datetime.fromisoformat(start_time.replace("Z", "+00:00"))

        try:
            # Try standard ISO format with timezone offset (like in the sample)
            end = datetime.fromisoformat(end_time)
        except:
            # Try with Z (UTC) format
            end = datetime.fromisoformat(end_time.replace("Z", "+00:00"))

        # Convert both to UTC for consistent comparison
        if start.tzinfo:
            start = start.astimezone(timezone.utc)
        if end.tzinfo:
            end = end.astimezone(timezone.utc)

        duration = (end - start).total_seconds() / 60  # Convert to minutes

        # Cap duration at 24 hours (1440 minutes) to handle unrealistic durations
        # This prevents multi-day visits from skewing the averages
        return min(1440, max(0, duration))  # Ensure between 0 and 24 hours
    except Exception as e:
        print(f"Error calculating duration: {e}")
        return 0


def load_points_with_duration(filepath, start_days_ago, end_days_ago):
    """Load points with place IDs and duration information."""
    with open(filepath, "r") as f:
        data = json.load(f)

    start = datetime.now(timezone.utc) - timedelta(days=start_days_ago)
    end = datetime.now(timezone.utc) - timedelta(days=end_days_ago)

    points = []
    for entry in data:
        start_time = entry.get("startTime")
        end_time = entry.get("endTime")

        if not start_time or not end_time:
            continue

        try:
            # Handle various timestamp formats
            try:
                # Try standard ISO format with timezone offset (like in the sample)
                timestamp = datetime.fromisoformat(start_time)
            except:
                # Try with Z (UTC) format
                timestamp = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        except Exception as e:
            print(f"Skipping entry due to timestamp parsing error: {e}")
            continue

        if not (start <= timestamp <= end):
            continue

        duration = calculate_duration_minutes(start_time, end_time)

        if "visit" in entry:
            top = entry["visit"].get("topCandidate", {})
            loc_str = top.get("placeLocation", "")
            lat, lon = extract_coords(loc_str)
            place_id = top.get("placeID")
            if lat and lon:
                points.append({
                    "timestamp": timestamp,
                    "latitude": lat,
                    "longitude": lon,
                    "place_id": place_id,
                    "duration_minutes": duration
                })
        elif "activity" in entry:
            for key in ["start", "end"]:
                loc_str = entry["activity"].get(key, "")
                lat, lon = extract_coords(loc_str)
                if lat and lon:
                    points.append({
                        "timestamp": timestamp,
                        "latitude": lat,
                        "longitude": lon,
                        "place_id": None,
                        "duration_minutes": duration / 2  # Split duration between start and end points
                    })

    # Create DataFrame
    df = pd.DataFrame(points)

    # Handle timezone-aware datetime objects
    if not df.empty and "timestamp" in df.columns:
        # Convert timezone-aware datetime objects to timezone-naive by converting to UTC
        df["timestamp"] = df["timestamp"].apply(lambda x: x.astimezone(timezone.utc).replace(tzinfo=None))

    return df


def cluster_locations(df, distance_meters=250, min_samples=2):
    """Cluster locations using DBSCAN algorithm."""
    coords = df[["latitude", "longitude"]].to_numpy()
    coords_rad = np.radians(coords)
    kms_per_radian = 6371.0088
    epsilon = distance_meters / 1000 / kms_per_radian

    db = DBSCAN(eps=epsilon, min_samples=min_samples, algorithm='ball_tree', metric='haversine')
    labels = db.fit_predict(coords_rad)
    df["cluster"] = labels
    return df


def analyze_clusters(df):
    """Analyze clusters to extract visit counts and durations."""
    cluster_analysis = []

    # Check if timestamp column exists and is in datetime format
    has_timestamp = "timestamp" in df.columns
    if has_timestamp:
        # Check if timestamp is already a datetime object
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            # Convert to datetime if it's not already
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                print("Converted timestamp column to datetime")
            except Exception as e:
                print(f"Warning: Could not convert timestamp to datetime: {e}")
                has_timestamp = False

    for cluster_id in df[df["cluster"] != -1]["cluster"].unique():
        cluster_df = df[df["cluster"] == cluster_id]

        # Get the most common place_id in this cluster
        place_ids = cluster_df["place_id"].dropna().tolist()
        if place_ids:
            place_id = max(set(place_ids), key=place_ids.count)
        else:
            place_id = None

        # Calculate metrics
        visit_count = len(cluster_df)
        total_duration = cluster_df["duration_minutes"].sum()
        avg_duration_per_visit = total_duration / visit_count if visit_count > 0 else 0

        # Calculate day-based metrics if timestamp is available
        if has_timestamp:
            try:
                # Get unique days visited
                unique_days = cluster_df["timestamp"].dt.date.unique()
                days_visited = len(unique_days)

                # For more accurate daily duration, calculate day by day
                daily_durations = []
                for day in unique_days:
                    # Get visits for this day
                    day_visits = cluster_df[cluster_df["timestamp"].dt.date == day]

                    # Sum durations for this day, but cap at 24 hours max per day
                    day_duration = min(1440, day_visits["duration_minutes"].sum())
                    daily_durations.append(day_duration)

                # Calculate average daily duration from the per-day totals
                avg_duration_per_day = sum(daily_durations) / days_visited if days_visited > 0 else 0

                # Calculate visits per day
                visits_per_day = visit_count / days_visited if days_visited > 0 else 0

            except Exception as e:
                print(f"Warning: Could not calculate day-based metrics: {e}")
                days_visited = 1
                avg_duration_per_day = min(1440, total_duration)  # Cap at 24h per day
                visits_per_day = visit_count
        else:
            # If no timestamp, use defaults
            days_visited = 1
            avg_duration_per_day = min(1440, total_duration)  # Cap at 24h per day
            visits_per_day = visit_count

        # Add centroid coordinates
        lat = cluster_df["latitude"].mean()
        lon = cluster_df["longitude"].mean()

        cluster_analysis.append({
            "cluster_id": int(cluster_id),
            "latitude": lat,
            "longitude": lon,
            "place_id": place_id,
            "visit_count": visit_count,
            "total_duration_minutes": total_duration,
            "avg_duration_per_visit": avg_duration_per_visit,
            "days_visited": days_visited,
            "avg_duration_per_day": avg_duration_per_day,
            "visits_per_day": visits_per_day
        })

    return pd.DataFrame(cluster_analysis)


def get_place_details(place_id, max_retries=3, retry_delay=2):
    """Get place details from Google Maps API with retry logic."""
    params = {
        "place_id": place_id,
        "fields": "name,types,formatted_address",
        "key": GOOGLE_API_KEY
    }

    for attempt in range(max_retries):
        try:
            res = requests.get(DETAILS_URL, params=params, timeout=10)
            if res.status_code == 200:
                result = res.json().get("result", {})
                return {
                    "name": result.get("name", "Unknown"),
                    "address": result.get("formatted_address", "Unknown"),
                    "types": result.get("types", [])
                }
            elif res.status_code == 429:  # Too Many Requests
                print(f"Rate limited by Google API. Waiting before retry {attempt + 1}/{max_retries}...")
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            else:
                print(f"API error (status code {res.status_code}). Retry {attempt + 1}/{max_retries}...")
                time.sleep(retry_delay)
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}. Retry {attempt + 1}/{max_retries}...")
            time.sleep(retry_delay)

    # If all retries failed, return default values
    print(f"❌ Failed to get place details for place_id: {place_id} after {max_retries} attempts")
    return {
        "name": f"Unknown Location (ID: {place_id[:8]}...)",
        "address": "Address unavailable",
        "types": []
    }


def get_place_id(lat, lon, max_retries=3, retry_delay=2):
    """Get place ID from coordinates using Google Maps API with retry logic."""
    params = {
        "location": f"{lat},{lon}",
        "radius": 100,
        "key": GOOGLE_API_KEY
    }

    for attempt in range(max_retries):
        try:
            res = requests.get(NEARBY_SEARCH_URL, params=params, timeout=10)
            if res.status_code == 200:
                results = res.json().get("results", [])
                if results:
                    return results[0].get("place_id")
                return None
            elif res.status_code == 429:  # Too Many Requests
                print(f"Rate limited by Google API. Waiting before retry {attempt + 1}/{max_retries}...")
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            else:
                print(f"API error (status code {res.status_code}). Retry {attempt + 1}/{max_retries}...")
                time.sleep(retry_delay)
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}. Retry {attempt + 1}/{max_retries}...")
            time.sleep(retry_delay)

    # If all retries failed
    print(f"❌ Failed to get place ID for coordinates ({lat}, {lon}) after {max_retries} attempts")
    return None


def label_centroids_with_places(cluster_df):
    """Label cluster centroids with place information."""
    results = []
    total_clusters = len(cluster_df)

    for idx, row in cluster_df.iterrows():
        place_id = row["place_id"]
        print(f"Processing cluster {idx + 1}/{total_clusters} (ID: {row['cluster_id']})...")

        try:
            if not place_id:
                place_id = get_place_id(row["latitude"], row["longitude"])

            place_info = {}
            if place_id:
                details = get_place_details(place_id)
                types = details.get("types", [])
                category = map_place_types_to_category(types)
                place_info = {
                    "name": details.get("name", "Unknown Location"),
                    "address": details.get("address", "Unknown Address"),
                    "types": ", ".join(types[:2]) if types else "unknown",
                    "category": category
                }
                print(f"[{row['cluster_id']}] {details.get('name')} → {types} → category: {category}")
            else:
                # If no place ID could be found, use generic info
                place_info = {
                    "name": f"Location Cluster {row['cluster_id']}",
                    "address": f"Lat: {row['latitude']:.6f}, Lon: {row['longitude']:.6f}",
                    "types": "unlabeled",
                    "category": "Other"
                }
                print(f"[{row['cluster_id']}] Could not identify location → category: Other")

            # Add all information to results
            results.append({
                "cluster_id": row["cluster_id"],
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "place_id": place_id,
                "name": place_info.get("name"),
                "address": place_info.get("address"),
                "types": place_info.get("types"),
                "category": place_info.get("category"),
                "visit_count": row["visit_count"],
                "total_duration_minutes": row["total_duration_minutes"],
                "avg_duration_per_visit": row["avg_duration_per_visit"],
                "days_visited": row["days_visited"],
                "avg_duration_per_day": row["avg_duration_per_day"],
                "visits_per_day": row["visits_per_day"]
            })

            # Save intermediate results every 10 clusters processed
            if (idx + 1) % 10 == 0 or idx == total_clusters - 1:
                temp_df = pd.DataFrame(results)
                temp_df.to_csv(f"results/intermediate_labeled_clusters_{idx + 1}.csv", index=False)
                print(f"✓ Saved intermediate results ({idx + 1}/{total_clusters} clusters processed)")

        except Exception as e:
            print(f"Error processing cluster {row['cluster_id']}: {e}")
            # Add with minimal information to avoid losing the cluster
            results.append({
                "cluster_id": row["cluster_id"],
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "place_id": place_id,
                "name": f"Error Processing Location {row['cluster_id']}",
                "address": f"Lat: {row['latitude']:.6f}, Lon: {row['longitude']:.6f}",
                "types": "error",
                "category": "Other",
                "visit_count": row["visit_count"],
                "total_duration_minutes": row["total_duration_minutes"],
                "avg_duration_per_visit": row["avg_duration_per_visit"],
                "days_visited": row["days_visited"],
                "avg_duration_per_day": row["avg_duration_per_day"],
                "visits_per_day": row["visits_per_day"]
            })

    return pd.DataFrame(results)


def apply_custom_label_adjustments(labeled_df):
    """Apply custom label adjustments for common places like Home and Work."""
    # Copy to avoid modifying the original
    df = labeled_df.copy()

    # Handle Residential → mark most visited as Home
    residential = df[df["category"] == "Residential"].copy()  # Create a proper copy
    if not residential.empty:
        # Consider both duration and visit count for Home determination
        residential.loc[:, "importance"] = (residential["total_duration_minutes"] * 0.7) + (
                    residential["visit_count"] * 0.3)
        max_row = residential.loc[residential["importance"].idxmax()]
        df.loc[df["cluster_id"] == max_row["cluster_id"], "category"] = "Home"

    # Handle Work location - frequent visits during weekdays with significant duration
    potential_work = df[(df["category"].isin(["Academic", "Government Offices", "Financial"]))
                        | (df["types"].str.contains(
        "office|workplace|company|business"))].copy()  # Create a proper copy

    if not potential_work.empty:
        potential_work.loc[:, "importance"] = (potential_work["total_duration_minutes"] * 0.6) + (
                    potential_work["visit_count"] * 0.4)
        max_row = potential_work.loc[potential_work["importance"].idxmax()]
        df.loc[df["cluster_id"] == max_row["cluster_id"], "category"] = "Work"

    return df


def normalize_metrics(df):
    """Normalize metrics for visualization purposes."""
    # Add columns with normalized values (0-1 scale)
    if not df.empty:
        # Normalize visit count
        df["norm_visits"] = (df["visit_count"] - df["visit_count"].min()) / (
                    df["visit_count"].max() - df["visit_count"].min()) if df["visit_count"].max() > df[
            "visit_count"].min() else 0.5

        # Normalize total duration
        df["norm_duration"] = (df["total_duration_minutes"] - df["total_duration_minutes"].min()) / (
                    df["total_duration_minutes"].max() - df["total_duration_minutes"].min()) if df[
                                                                                                    "total_duration_minutes"].max() > \
                                                                                                df[
                                                                                                    "total_duration_minutes"].min() else 0.5

        # Normalize daily duration
        df["norm_duration_per_day"] = (df["avg_duration_per_day"] - df["avg_duration_per_day"].min()) / (
                    df["avg_duration_per_day"].max() - df["avg_duration_per_day"].min()) if df[
                                                                                                "avg_duration_per_day"].max() > \
                                                                                            df[
                                                                                                "avg_duration_per_day"].min() else 0.5

        # Ensure no NaN values
        df.fillna(0, inplace=True)

    return df


def save_interactive_map(df, filename="results/duration_visit_map.html", use_daily_metrics=True):
    """Save an interactive map with markers sized by visits and colored by duration."""
    m = folium.Map(location=[df["latitude"].mean(), df["longitude"].mean()], zoom_start=11)

    # Use daily metrics or absolute metrics based on parameter
    duration_col = "avg_duration_per_day" if use_daily_metrics else "total_duration_minutes"
    visits_col = "visits_per_day" if use_daily_metrics else "visit_count"

    # Add base map layers
    folium.TileLayer('cartodbpositron', name='Light Map').add_to(m)
    folium.TileLayer('cartodbdark_matter', name='Dark Map').add_to(m)

    # Create feature groups for each category
    category_layers = {}
    for category in CATEGORY_COLORS.keys():
        category_layers[category] = folium.FeatureGroup(name=category)

    for _, row in df.iterrows():
        category = row.get("category", "Other")
        base_color = CATEGORY_COLORS.get(category, "gray")
        cluster_id = int(row["cluster_id"])

        # Calculate marker size based on visit count (normalized between 5-25)
        visit_value = row[visits_col]
        size_min, size_max = 5, 25
        visit_scale = row.get("norm_visits", 0.5)  # Default to 0.5 if missing
        radius = size_min + ((size_max - size_min) * visit_scale)

        # Calculate opacity based on duration (normalized between 0.3-0.9)
        duration_value = row[duration_col]
        opacity_min, opacity_max = 0.3, 0.9
        duration_scale = row.get("norm_duration_per_day" if use_daily_metrics else "norm_duration", 0.5)
        opacity = opacity_min + ((opacity_max - opacity_min) * duration_scale)

        # Create detailed popup
        total_hours = int(row["total_duration_minutes"] // 60)
        total_minutes = int(row["total_duration_minutes"] % 60)

        avg_visit_hours = int(row["avg_duration_per_visit"] // 60)
        avg_visit_minutes = int(row["avg_duration_per_visit"] % 60)

        daily_hours = int(row["avg_duration_per_day"] // 60)
        daily_minutes = int(row["avg_duration_per_day"] % 60)

        popup = (
            f"<b>{row.get('name', 'Unknown')}</b><br>"
            f"<i>Address:</i> {row.get('address', 'Unknown')}<br>"
            f"<i>Category:</i> {category}<br>"
            f"<i>Types:</i> {row.get('types', 'Unknown')}<br><br>"
            f"<i>Visit Count:</i> {int(row.get('visit_count', 0))}<br>"
            f"<i>Days Visited:</i> {int(row.get('days_visited', 0))}<br>"
            f"<i>Visits per Day:</i> {row.get('visits_per_day', 0):.1f}<br><br>"
            f"<i>Total Duration:</i> {total_hours}h {total_minutes}m<br>"
            f"<i>Avg Duration per Visit:</i> {avg_visit_hours}h {avg_visit_minutes}m<br>"
            f"<i>Avg Duration per Day:</i> {daily_hours}h {daily_minutes}m<br>"
            f"<i>Cluster ID:</i> {cluster_id}"
        )

        # Create circle marker
        circle = folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=radius,
            popup=folium.Popup(popup, max_width=300),
            tooltip=f"{row.get('name', 'Unknown')} ({daily_hours}h {daily_minutes}m / {int(row.get('visit_count', 0))} visits)",
            color=base_color,
            weight=2,
            fill=True,
            fill_color=base_color,
            fill_opacity=opacity
        )

        # Add to appropriate category layer
        category_layers[category].add_to(m)
        circle.add_to(category_layers[category])

    # Add all category layers to the map
    for category, layer in category_layers.items():
        layer.add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    def save_interactive_map(df, filename="results/duration_visit_map.html", use_daily_metrics=True):
        # ... existing code ...

        # Create enhanced legend that includes category colors
        legend_html = """
        <div style="position: fixed; bottom: 50px; right: 50px; width: 220px; 
                 border:2px solid grey; z-index:9999; font-size:14px;
                 background-color: white; padding: 10px; border-radius: 5px;">
            <div style="text-align: center; margin-bottom: 5px;"><b>Duration & Visits</b></div>
            <div style="display: flex; flex-direction: column; height: 80px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 20px; opacity: 0.9;">●</span> High duration
                    <span style="font-size: 20px;">⬤</span> Many visits
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 5px;">
                    <span style="font-size: 20px; opacity: 0.3;">●</span> Low duration
                    <span style="font-size: 10px;">⬤</span> Few visits
                </div>
            </div>
            <hr style="margin: 10px 0;">
            <div style="text-align: center; margin-bottom: 5px;"><b>Categories</b></div>
            <div style="display: flex; flex-direction: column; max-height: 200px; overflow-y: auto;">
        """

        # Add category colors to legend
        for category, color in CATEGORY_COLORS.items():
            legend_html += f"""
                <div style="display: flex; align-items: center; margin-bottom: 4px;">
                    <span style="color:{color}; font-size:18px; margin-right: 8px;">●</span>
                    <span>{category}</span>
                </div>
            """

        legend_html += """
            </div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

    # Add title and information about metrics
    metric_type = "Daily Average Metrics" if use_daily_metrics else "Absolute Total Metrics"
    title_html = f"""
    <div style="position: fixed; top: 10px; left: 50px; z-index:9999; font-size:14px;
             background-color: white; padding: 10px; border-radius: 5px; border:2px solid grey;">
        <h3 style="margin: 0;">Location Visit Patterns</h3>
        <div><b>{metric_type}</b></div>
        <div style="font-size: 12px; margin-top: 5px;">
            • Circle size: Visit frequency<br>
            • Color opacity: Time spent<br>
            • Click markers for details
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # Save the map
    m.save(filename)
    print(f"✅ Interactive map saved to {filename}")

    # Save a daily normalized version if we're not already using daily metrics
    if not use_daily_metrics:
        save_interactive_map(df, filename.replace('.html', '_daily.html'), use_daily_metrics=True)


def save_metric_distributions(df, filename_prefix="results/metrics_distribution"):
    """Save visualizations of visit and duration distributions."""
    plt.figure(figsize=(12, 10))

    # Scatter plot of visit count vs. total duration
    plt.subplot(2, 2, 1)
    scatter = plt.scatter(df["visit_count"], df["total_duration_minutes"] / 60,
                          c=df["category"].map(lambda x: CATEGORY_COLORS.get(x, "gray")),
                          alpha=0.7, s=50)
    plt.xlabel("Number of Visits")
    plt.ylabel("Total Duration (hours)")
    plt.title("Visits vs. Duration by Category")

    # Create categories legend
    categories = df["category"].unique()
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=CATEGORY_COLORS.get(cat, "gray"),
                          markersize=8, label=cat) for cat in categories]
    plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1))

    # Histogram of visit counts
    plt.subplot(2, 2, 3)
    plt.hist(df["visit_count"], bins=20, color="skyblue", edgecolor="black")
    plt.xlabel("Number of Visits")
    plt.ylabel("Number of Places")
    plt.title("Distribution of Visit Counts")

    # Histogram of durations
    plt.subplot(2, 2, 4)
    plt.hist(df["total_duration_minutes"] / 60, bins=20, color="lightgreen", edgecolor="black")
    plt.xlabel("Total Duration (hours)")
    plt.ylabel("Number of Places")
    plt.title("Distribution of Total Durations")

    # Scatter of daily metrics
    plt.subplot(2, 2, 2)

    # Check for column name compatibility
    y_col = "avg_duration_per_day" if "avg_duration_per_day" in df.columns else "duration_per_day"

    plt.scatter(df["visits_per_day"], df[y_col] / 60,
                c=df["category"].map(lambda x: CATEGORY_COLORS.get(x, "gray")),
                alpha=0.7, s=50)
    plt.xlabel("Average Visits per Day")
    plt.ylabel("Average Hours per Day")
    plt.title("Daily Visits vs. Daily Duration")

    plt.tight_layout()
    plt.savefig(f"{filename_prefix}.png", dpi=300, bbox_inches="tight")
    print(f"✅ Metrics distribution saved to {filename_prefix}.png")


def save_basic_distributions(df, filename_prefix="results/metrics_distribution_basic"):
    """Create basic visualizations with dual pie charts."""
    # Create a figure with 2 rows - top row has 3 plots, bottom row has 3 pie charts
    fig = plt.figure(figsize=(18, 14))

    # Set up a more complex grid to give proper space to each plot
    gs = fig.add_gridspec(2, 3)

    # Scatter plot - maintain square aspect ratio
    ax1 = fig.add_subplot(gs[0, 0])
    scatter = ax1.scatter(df["visit_count"], df["total_duration_minutes"] / 60,
                          c=df["category"].map(lambda x: CATEGORY_COLORS.get(x, "gray")),
                          alpha=0.7, s=50)
    ax1.set_xlabel("Number of Visits")
    ax1.set_ylabel("Total Duration (hours)")
    ax1.set_title("Visits vs. Duration by Category")

    # Create categories legend
    categories = df["category"].unique()
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=CATEGORY_COLORS.get(cat, "gray"),
                          markersize=8, label=cat) for cat in categories]
    ax1.legend(handles=handles, loc='upper right')

    # Histogram of visit counts
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(df["visit_count"], bins=20, color="skyblue", edgecolor="black")
    ax2.set_xlabel("Number of Visits")
    ax2.set_ylabel("Number of Places")
    ax2.set_title("Distribution of Visit Counts")

    # Histogram of durations
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(df["total_duration_minutes"] / 60, bins=20, color="lightgreen", edgecolor="black")
    ax3.set_xlabel("Total Duration (hours)")
    ax3.set_ylabel("Number of Places")
    ax3.set_title("Distribution of Total Durations")

    # Define shared pie chart function for consistent styling
    def create_pie_chart(ax, data, title):
        # For the pie chart, only show labels for categories with significant percentage
        threshold = 3.0  # Only show labels for categories with >= 3% of the total

        # Create pie wedge labels based on threshold
        def label_func(pct, allvals):
            absolute = int(pct / 100. * sum(allvals))
            if pct >= threshold:
                return f"{pct:.1f}%"
            return ""

        wedges, texts, autotexts = ax.pie(
            data,
            labels=None,
            colors=[CATEGORY_COLORS.get(cat, "gray") for cat in data.index],
            autopct=lambda pct: label_func(pct, data),
            startangle=90,
            pctdistance=0.85
        )

        # Add a legend outside the pie chart
        ax.legend(data.index, loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_title(title)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

    # PIE CHART 1: Unique locations by category (excluding Home and Work)
    ax4 = fig.add_subplot(gs[1, 0])
    # Make a copy to avoid modifying the original data
    df_for_unique = df.copy()
    # Remove Home and Work categories
    df_for_unique = df_for_unique[~df_for_unique["category"].isin(["Home", "Work"])]
    # Count unique locations by category
    category_counts = df_for_unique["category"].value_counts()
    create_pie_chart(ax4, category_counts, "Unique Locations by Category\n(excluding Home & Work)")

    # PIE CHART 2: Visits by category (including Home and Work)
    ax5 = fig.add_subplot(gs[1, 1])
    # Aggregate visit counts by category
    visit_by_category = df.groupby("category")["visit_count"].sum()
    create_pie_chart(ax5, visit_by_category, "Total Visits by Category\n(including Home & Work)")

    # PIE CHART 3: Duration by category
    ax6 = fig.add_subplot(gs[1, 2])
    # Aggregate duration by category (in hours)
    duration_by_category = df.groupby("category")["total_duration_minutes"].sum() / 60
    create_pie_chart(ax6, duration_by_category, "Total Duration by Category\n(hours)")

    plt.tight_layout()
    plt.savefig(f"{filename_prefix}.png", dpi=300, bbox_inches="tight")
    print(f"✅ Enhanced distribution charts saved to {filename_prefix}.png")


def main(input_file, output_prefix="results/location_duration", start_days_ago=180, end_days_ago=0,
         skip_api_calls=False):
    """Main function to process location history and create visualizations."""
    print(f"Processing location history from {start_days_ago} to {end_days_ago} days ago...")

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # 1. Load data with duration
    df = load_points_with_duration(input_file, start_days_ago, end_days_ago)
    print(f"Loaded {len(df)} location points with duration information")

    # 2. Cluster locations
    clustered_df = cluster_locations(df)
    cluster_count = len(clustered_df[clustered_df["cluster"] != -1]["cluster"].unique())
    print(f"Identified {cluster_count} location clusters")

    # 3. Analyze clusters for frequency and duration
    cluster_analysis = analyze_clusters(clustered_df)
    print(f"Analyzed metrics for {len(cluster_analysis)} clusters")

    # Save intermediate results before API calls
    cluster_analysis.to_csv(f"{output_prefix}_clusters_raw.csv", index=False)
    print(f"✅ Saved raw cluster metrics to {output_prefix}_clusters_raw.csv")

    # 4. Label with place information (or skip if API issues)
    if skip_api_calls:
        print("Skipping API calls for place details as requested...")
        # Create a basic labeled dataframe without API calls
        labeled_df = cluster_analysis.copy()
        labeled_df["name"] = labeled_df.apply(lambda x: f"Location Cluster {x['cluster_id']}", axis=1)
        labeled_df["address"] = labeled_df.apply(lambda x: f"Lat: {x['latitude']:.6f}, Lon: {x['longitude']:.6f}",
                                                 axis=1)
        labeled_df["types"] = "unlabeled"
        labeled_df["category"] = "Other"
    else:
        # Try to load cached labeled data first
        cached_file = f"{output_prefix}_labeled_clusters.csv"
        if os.path.exists(cached_file):
            print(f"Found cached labeled data at {cached_file}")
            try:
                labeled_df = pd.read_csv(cached_file)
                print(f"Loaded {len(labeled_df)} labeled clusters from cache")

                # Check for old column names and update them
                if 'duration_per_day' in labeled_df.columns and 'avg_duration_per_day' not in labeled_df.columns:
                    print("Converting old column names to new format...")
                    labeled_df = labeled_df.rename(columns={
                        'duration_per_day': 'avg_duration_per_day',
                        'avg_duration_minutes': 'avg_duration_per_visit'
                    })
            except Exception as e:
                print(f"Error loading cached data: {e}. Will process from scratch.")
                labeled_df = label_centroids_with_places(cluster_analysis)
        else:
            # Process from scratch
            labeled_df = label_centroids_with_places(cluster_analysis)

        # Save labeled results
        labeled_df.to_csv(f"{output_prefix}_labeled_clusters.csv", index=False)
        print(f"✅ Labeled {len(labeled_df)} clusters with place information")

    # 5. Apply custom adjustments for Home/Work
    adjusted_df = apply_custom_label_adjustments(labeled_df)

    # 6. Normalize metrics for visualization
    final_df = normalize_metrics(adjusted_df)

    # 7. Save results
    final_df.to_csv(f"{output_prefix}_metrics.csv", index=False)
    print(f"✅ Saved metrics to {output_prefix}_metrics.csv")

    # 8. Create interactive map
    save_interactive_map(final_df, f"{output_prefix}_map.html", use_daily_metrics=False)

    # 9. Create metric distribution visualizations
    # Create metric distribution visualizations
    # try:
    #     save_metric_distributions(final_df, f"{output_prefix}_distributions")
    # except KeyError as e:
    #     print(f"Warning: Could not create all distribution charts due to missing column: {e}")
    #     # Fallback version of save_metric_distributions that uses available columns
    #     save_basic_distributions(final_df, f"{output_prefix}_basic_distributions")
    # Always create both types of visualizations
    save_metric_distributions(final_df, f"{output_prefix}_distributions")
    save_basic_distributions(final_df, f"{output_prefix}_basic_distributions")

    return final_df


if __name__ == "__main__":
    input_file = "../data/location-history.json"
    try:
        # Try running with API calls
        results = main(input_file, output_prefix="results/location_duration_6months", start_days_ago=180,
                       end_days_ago=0)
        print("✅ Analysis complete with API integration!")
    except requests.exceptions.ConnectionError as e:
        print(f"\n⚠️ Connection error occurred: {e}")
        print("Retrying without API calls to Google Maps...\n")

        # Retry without API calls
        results = main(input_file, output_prefix="results/location_duration_6months_no_api",
                       start_days_ago=180, end_days_ago=0, skip_api_calls=True)
        print("✅ Analysis complete without API integration")
        print("Note: To add place names later, fix the API connection and run with the cached data")