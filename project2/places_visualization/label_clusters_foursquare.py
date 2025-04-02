import json

import requests
import pandas as pd

FSQ_API_KEY = ""

def label_cluster_with_foursquare(lat, lon, radius=100, limit=1):
    url = "https://api.foursquare.com/v3/places/search"
    headers = {
        "Authorization": FSQ_API_KEY
    }
    params = {
        "ll": f"{lat},{lon}",
        "radius": radius,
        "limit": limit,
        "sort": "RELEVANCE"
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        response.raise_for_status()
        results = response.json().get("results", [])
        if results:
            top = results[0]
            return {
                "name": top.get("name", "Unknown"),
                "category": top["categories"][0]["name"] if top.get("categories") else "Unknown"
            }
    except Exception as e:
        print(f"Error fetching info for ({lat}, {lon}): {e}")
    return {"name": "Unknown", "category": "Unknown"}

def enrich_centroids_with_labels(centroids_file):
    with open(centroids_file, "r") as f:
        centroids = json.load(f)

    labeled = []
    for cluster_id, (lat, lon) in centroids.items():
        label = label_cluster_with_foursquare(lat, lon)
        if label["name"] == "Unknown" or label["category"] == "Unknown":
            print(f"[Cluster {cluster_id}] Skipped (Unknown place)")
            continue
        print(f"[Cluster {cluster_id}] {label['name']} — {label['category']}")
        labeled.append({
            "cluster_id": cluster_id,
            "latitude": lat,
            "longitude": lon,
            "name": label["name"],
            "category": label["category"]
        })

    return pd.DataFrame(labeled)

# Usage
if __name__ == "__main__":
    input_file = "results/cluster_centroids.json"
    df_labeled = enrich_centroids_with_labels(input_file)
    df_labeled.to_csv("results/labeled_clusters.csv", index=False)
    print("✅ Labeled clusters saved to results/labeled_clusters.csv")
