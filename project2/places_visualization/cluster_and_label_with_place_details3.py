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
    "Park & Entertainment": ["park", "natural_feature", "campground", "movie_theater", "night_club", "bar", "amusement_park", "zoo"]
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

def load_recent_points_with_place_ids(filename, days=365):
    with open(filename, "r") as f:
        data = json.load(f)

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    points = []

    for entry in data:
        timestamp_str = entry.get("startTime") or entry.get("endTime")
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except:
            continue

        if timestamp < cutoff:
            continue

        if "visit" in entry:
            top = entry["visit"].get("topCandidate", {})
            loc_str = top.get("placeLocation", "")
            lat, lon = extract_coords(loc_str)
            place_id = top.get("placeID")
            if lat and lon:
                points.append({"latitude": lat, "longitude": lon, "place_id": place_id})
        elif "activity" in entry:
            for key in ["start", "end"]:
                loc_str = entry["activity"].get(key, "")
                lat, lon = extract_coords(loc_str)
                if lat and lon:
                    points.append({"latitude": lat, "longitude": lon, "place_id": None})

    return pd.DataFrame(points)

def load_recent_points_with_place_ids_in_range(filepath, start_days_ago, end_days_ago):
    with open(filepath, "r") as f:
        data = json.load(f)

    start = datetime.now(timezone.utc) - timedelta(days=start_days_ago)
    end = datetime.now(timezone.utc) - timedelta(days=end_days_ago)

    points = []
    for entry in data:
        timestamp_str = entry.get("startTime") or entry.get("endTime")
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except:
            continue

        if not (start <= timestamp <= end):
            continue

        if "visit" in entry:
            top = entry["visit"].get("topCandidate", {})
            loc_str = top.get("placeLocation", "")
            lat, lon = extract_coords(loc_str)
            place_id = top.get("placeID")
            if lat and lon:
                points.append({"timestamp": timestamp, "latitude": lat, "longitude": lon, "place_id": place_id})
        elif "activity" in entry:
            for key in ["start", "end"]:
                loc_str = entry["activity"].get(key, "")
                lat, lon = extract_coords(loc_str)
                if lat and lon:
                    points.append({"timestamp": timestamp, "latitude": lat, "longitude": lon, "place_id": None})

    return pd.DataFrame(points)

def cluster_locations(df, distance_meters=250, min_samples=2):
    coords = df[["latitude", "longitude"]].to_numpy()
    coords_rad = np.radians(coords)
    kms_per_radian = 6371.0088
    epsilon = distance_meters / 1000 / kms_per_radian

    db = DBSCAN(eps=epsilon, min_samples=min_samples, algorithm='ball_tree', metric='haversine')
    labels = db.fit_predict(coords_rad)
    df["cluster"] = labels
    return df

def get_cluster_centroids(df):
    centroids = {}
    for cluster_id in df["cluster"].unique():
        if cluster_id == -1:
            continue
        group = df[df["cluster"] == cluster_id]
        timestamp = group["timestamp"]
        lat = group["latitude"].mean()
        lon = group["longitude"].mean()
        place_ids = group["place_id"].dropna().tolist()
        centroids[str(int(cluster_id))] = (timestamp, lat, lon, place_ids[0] if place_ids else None)
    return centroids

def get_place_details(place_id):
    params = {
        "place_id": place_id,
        "fields": "name,types,formatted_address",
        "key": GOOGLE_API_KEY
    }
    res = requests.get(DETAILS_URL, params=params)
    if res.status_code != 200:
        return {}
    result = res.json().get("result", {})
    return {
        "name": result.get("name", "Unknown"),
        "address": result.get("formatted_address", "Unknown"),
        "types": result.get("types", [])
    }

def get_place_id(lat, lon):
    params = {
        "location": f"{lat},{lon}",
        "radius": 100,
        "key": GOOGLE_API_KEY
    }
    res = requests.get(NEARBY_SEARCH_URL, params=params)
    if res.status_code != 200:
        return None
    results = res.json().get("results", [])
    if results:
        return results[0].get("place_id")
    return None

def label_centroids_with_places(centroids):
    results = []
    for cid, (timestamp, lat, lon, place_id) in centroids.items():
        if not place_id:
            place_id = get_place_id(lat, lon)
        if place_id:
            details = get_place_details(place_id)
            types = details.get("types", [])
            category = map_place_types_to_category(types)
            print(f"[{cid}] {details.get('name')} → {types} → category: {category}")
            results.append({
                "timestamp": timestamp,
                "cluster_id": cid,
                "latitude": lat,
                "longitude": lon,
                "place_id": place_id,
                "name": details.get("name"),
                "address": details.get("address"),
                "types": ", ".join(types[:2]),
                "category": category
            })
    return pd.DataFrame(results)

def apply_custom_label_adjustments(labeled_df, freq_df):
    # Ensure cluster_id is the same dtype for merge
    labeled_df["cluster_id"] = labeled_df["cluster_id"].astype(int)
    freq_df["cluster_id"] = freq_df["cluster_id"].astype(int)

    # Merge frequencies for easier access
    labeled_df = labeled_df.merge(freq_df, how="left", on="cluster_id")

    # Handle Residential → mark most visited as Home
    residential = labeled_df[labeled_df["category"] == "Residential"]
    if not residential.empty:
        max_row = residential.loc[residential["visits"].idxmax()]
        labeled_df.loc[labeled_df["cluster_id"] == max_row["cluster_id"], "category"] = "Home"

    # Handle Academic → mark most visited university as Work
    academic = labeled_df[labeled_df["category"] == "Academic"]
    if not academic.empty:
        max_row = academic.loc[academic["visits"].idxmax()]
        labeled_df.loc[labeled_df["cluster_id"] == max_row["cluster_id"], "category"] = "Work"

    return labeled_df

def save_map(df, freq_df, filename="results/clustered_labeled_map_3_6months.html"):
    m = folium.Map(location=[df["latitude"].mean(), df["longitude"].mean()], zoom_start=6)
    cluster_layer = MarkerCluster().add_to(m)

    for _, row in df.iterrows():
        category = row.get("category", "Other")
        color = CATEGORY_COLORS.get(category, "gray")
        cluster_id = int(row["cluster_id"])
        visits = freq_df.loc[freq_df["cluster_id"] == cluster_id, "visits"].values
        visit_count = int(visits[0]) if len(visits) > 0 else 1
        radius = 5 + visit_count

        popup = (
            f"<b>{row['name']}</b><br>{row['address']}<br>"
            f"<i>Top Types:</i> {row['types']}<br>"
            f"<i>Category:</i> {category}<br>"
            f"<i>Visits:</i> {visit_count}<br>"
            f"<i>Cluster ID:</i> {row['cluster_id']}"
        )

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=min(radius, 20),
            popup=popup,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8
        ).add_to(cluster_layer)

    legend_items = "".join(
        f"<div style='margin-bottom:4px;'><span style='color:{color}; font-size:18px;'>●</span> {category}</div>"
        for category, color in CATEGORY_COLORS.items()
    )

    legend_html = f"""
        <div style='position: fixed; bottom: 30px; left: 30px; background-color: white;
                    border: 2px solid grey; z-index:9999; font-size:14px; padding: 10px; max-height: 400px; overflow-y: auto;'>
            <b>Legend (Categories)</b><br>
            {legend_items}
        </div>
        """
    legend = MacroElement()
    legend._template = Template(legend_html)
    m.get_root().html.add_child(legend)

    m.save(filename)
    print(f"✅ Map saved to {filename}")

def get_cluster_centroids_from_coords(df):
    centroids = {}
    for cluster_id in df["cluster"].unique():
        if cluster_id == -1:
            continue
        group = df[df["cluster"] == cluster_id]
        lat = group["latitude"].mean()
        lon = group["longitude"].mean()
        centroids[str(int(cluster_id))] = (lat, lon)
    return centroids


if __name__ == "__main__":
    input_file = "../data/location-history.json"

    # df = load_recent_points_with_place_ids(input_file, days=365)
    df = load_recent_points_with_place_ids_in_range(input_file, start_days_ago=180, end_days_ago=0)
    label = '6months'
    df = cluster_locations(df)
    centroids = get_cluster_centroids(df)

    freq = df[df["cluster"] != -1]["cluster"].value_counts().sort_values(ascending=False).reset_index()
    freq.columns = ["cluster_id", "visits"]
    freq.to_csv(f"results/cluster_visit_counts_{label}.csv", index=False)

    labeled = label_centroids_with_places(centroids)
    labeled = apply_custom_label_adjustments(labeled, freq)
    labeled.to_csv(f"results/cluster_centroids_labeled_{label}.csv", index=False)
    save_map(labeled, freq)
