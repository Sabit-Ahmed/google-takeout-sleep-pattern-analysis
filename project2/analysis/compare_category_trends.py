import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CATEGORY_ORDER = [
    "Home", "Residential", "Restaurant", "Shopping & Grocery", "Academic", "Work", "Government Offices",
    "Travel", "Spiritual", "Commute", "Historical",
    "Financial", "Auto & Gas", "Medical", "Park & Entertainment", "Other"
]

# Extract categories from labeled centroids CSV
def extract_categories(csv_path):
    df = pd.read_csv(csv_path)
    if "category" not in df.columns or df.empty:
        print(f"⚠️ No 'category' column or empty data in {csv_path}")
        return pd.Series(dtype=float)
    return df.groupby("category")["visits"].sum().sort_values(ascending=False)

# Normalize for comparison
def normalize_counts(counts):
    total = counts.sum()
    return (counts / total).reindex(CATEGORY_ORDER, fill_value=0)

# Plot comparison chart
def plot_category_differences(counts_this, counts_last):
    # Fill missing categories with 0 for alignment
    all_categories = sorted(set(counts_this.index).union(set(counts_last.index)))
    counts_this = counts_this.reindex(all_categories, fill_value=0)
    counts_last = counts_last.reindex(all_categories, fill_value=0)
    if counts_this.sum() == 0 or counts_last.sum() == 0:
        print("⚠️ One of the datasets is empty or has no valid categories.")
        return

    diff = counts_this - counts_last
    print("This year counts:\n", counts_this)
    print("Last year counts:\n", counts_last)
    print("Difference:\n", diff)

    diff_sorted = diff.sort_values()
    colors = np.where(diff_sorted < 0, "red", "green")

    plt.figure(figsize=(12, 6))
    bars = diff_sorted.plot(kind="barh", color=colors)

    plt.title("Change in Visit Frequency by Category (This Year vs Last Year)")
    plt.xlabel("Proportional Change")
    plt.grid(True, axis="x", linestyle=":")

    # Custom legend
    red_patch = mpatches.Patch(color='red', label='Decreased Visits')
    green_patch = mpatches.Patch(color='green', label='Increased Visits')
    plt.legend(handles=[green_patch, red_patch], loc="lower right")

    plt.tight_layout()
    plt.savefig("results/category_diff_bar_chart.png")
    plt.show()

# Load CSVs
last_year_csv = "results/cluster_centroids_labeled_last_year.csv"
this_year_csv = "results/cluster_centroids_labeled_this_year.csv"

# last_counts = extract_categories(last_year_csv)
# this_counts = extract_categories(this_year_csv)

last_counts = normalize_counts(extract_categories(last_year_csv))
this_counts = normalize_counts(extract_categories(this_year_csv))

plot_category_differences(this_counts, last_counts)
print("✅ Category difference chart saved to results/category_diff_bar_chart.png")

from scipy.stats import chi2_contingency

# Build contingency table
contingency = pd.concat([last_counts, this_counts], axis=1).fillna(0)
contingency.columns = ["last_year", "this_year"]

# Convert proportions back to raw counts for test
total_last = 1000  # scale factors (can be any)
total_this = 1000

observed = pd.DataFrame({
    "last_year": (contingency["last_year"] * total_last).astype(int),
    "this_year": (contingency["this_year"] * total_this).astype(int)
})

# Remove categories with zero counts in both years
observed = observed[(observed["last_year"] > 0) | (observed["this_year"] > 0)]
chi2, p, dof, expected = chi2_contingency(observed.T)
print(f"Chi-squared test statistic: {chi2:.2f}, p-value: {p:.4f}")
