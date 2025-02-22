import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def create_topic_heatmap(csv_file):
    # Read and process data
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp']
                                .str.replace('EST', '')  # Remove EST timezone indicator
                                .str.strip(),            # Remove any extra whitespace
                                format='mixed',          # Allow mixed formats
                                utc=False)

    df['hour'] = df['timestamp'].dt.hour
    df['topic'] = df['topic_compressed']
    # Get significant topics
    topic_counts = df.groupby('topic').size()
    significant_topics = topic_counts[topic_counts >= 20].index
    print(topic_counts[topic_counts >= 20].sort_values(ascending=False))

    # Create pivot table with all hours (0-23)
    df_significant = df[df['topic'].isin(significant_topics)]
    pivot_table = pd.crosstab(
        df_significant['topic'],
        df_significant['hour'],
        normalize='index'
    )

    # Ensure all hours are present (0-23)
    all_hours = range(24)
    for hour in all_hours:
        if hour not in pivot_table.columns:
            pivot_table[hour] = 0

    # Sort columns (hours) and rows (topics)
    pivot_table = pivot_table.reindex(columns=sorted(pivot_table.columns))
    pivot_table = pivot_table.reindex(df_significant['topic'].value_counts().index)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(15, 8))

    # Create heatmap manually
    im = ax.imshow(pivot_table, aspect='auto', cmap='Reds')

    # Add text annotations
    # for i in range(len(pivot_table.index)):
    #     for j in range(24):
    #         value = pivot_table.iloc[i, j]
    #         text = f'{value:.1%}'
    #         ax.text(j, i, text, ha='center', va='center', size=7)

    # Set ticks
    ax.set_xticks(range(24))
    ax.set_yticks(range(len(pivot_table.index)))

    # Set tick labels
    ax.set_xticklabels([f'{i:02d}:00' for i in range(24)], rotation=45, ha='right')
    ax.set_yticklabels(pivot_table.index)

    # Add gridlines
    ax.set_xticks(np.arange(-.5, 24, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(pivot_table.index), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.5)

    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Activity Percentage')

    # Add labels and title
    plt.title('Topic Activity Distribution by Hour', pad=20)
    plt.xlabel('Hour of Day', labelpad=10)
    plt.ylabel('Topic')

    # Adjust layout
    plt.tight_layout()

    return fig, pivot_table


def analyze_peak_hours(pivot_table):
    """Analyze peak activity hours for each topic"""
    peak_stats = []

    for topic in pivot_table.index:
        hour_dist = pivot_table.loc[topic]
        peak_hour = hour_dist.idxmax()
        peak_percentage = hour_dist[peak_hour] * 100

        weighted_hours = sum(hour * pct for hour, pct in hour_dist.items())
        total_activity = hour_dist.sum()

        peak_stats.append({
            'Topic': topic,
            'Peak Hour': f"{peak_hour:02d}:00",
            'Peak Activity': f"{peak_percentage:.1f}%",
            'Avg Activity Time': f"{weighted_hours / total_activity:.1f}"
        })

    return pd.DataFrame(peak_stats)


def main():
    fig, pivot_data = create_topic_heatmap('../data/categorized_output_compressed.csv')
    peak_stats = analyze_peak_hours(pivot_data)

    print("\nActivity Statistics by Topic:")
    print("-" * 80)
    print(peak_stats.to_string(index=False))

    plt.savefig('topic_heatmap.png', dpi=300, bbox_inches='tight')
    print("\nHeatmap saved as 'topic_heatmap.png'")
    plt.show()


if __name__ == "__main__":
    main()