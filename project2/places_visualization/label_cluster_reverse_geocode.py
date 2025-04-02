import json
import requests
import pandas as pd
import time

GOOGLE_API_KEY = ""
GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"

def reverse_geocode(lat, lon):
    params = {
        "latlng": f"{lat},{lon}",
        "key": GOOGLE_API_KEY
    }

    try:
        res = requests.get(GEOCODE_URL, params=params, timeout=5)
        res.raise_for_status()
        results = res.json().get("results", [])
        if results:
            address = results[0].get("formatted_address", "Unknown")
            return address
    except Exception as e:
        print(f"Error reverse geocoding ({lat}, {lon}): {e}")
    return "Unknown"

def label_clusters(centroids_file):
    with open(centroids_file, "r") as f:
        centroids = json.load(f)

    labeled = []
    for cluster_id, (lat, lon) in centroids.items():
        address = reverse_geocode(lat, lon)
        if address == "Unknown":
            print(f"[Cluster {cluster_id}] Skipped (Unknown)")
            continue
        print(f"[Cluster {cluster_id}] {address}")
        labeled.append({
            "cluster_id": cluster_id,
            "latitude": lat,
            "longitude": lon,
            "address": address
        })
        time.sleep(0.1)  # Be kind to the API

    return pd.DataFrame(labeled)

if __name__ == "__main__":
    labeled_df = label_clusters("results/cluster_centroids.json")
    labeled_df.to_csv("results/labeled_clusters_geocoded.csv", index=False)
    print("✅ Labeled addresses saved to results/labeled_clusters_geocoded.csv")
