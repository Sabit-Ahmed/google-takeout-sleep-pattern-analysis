import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from scipy.stats import norm


def parse_sleep_time(time_range):
    """Parse sleep time range like '1.30 AM - 9.00 AM' and return duration in hours"""
    start, end = time_range.split(' - ')

    def convert_to_24hr(time_str):
        # Split into time and AM/PM
        time_parts = time_str.split(' ')
        hour_min = time_parts[0].split('.')
        period = time_parts[1]

        hour = int(hour_min[0])
        minute = int(hour_min[1]) if len(hour_min) > 1 else 0

        # Convert to 24-hour format
        if period == 'PM' and hour != 12:
            hour += 12
        elif period == 'AM' and hour == 12:
            hour = 0

        return hour + minute / 60

    start_time = convert_to_24hr(start)
    end_time = convert_to_24hr(end)

    # Calculate duration
    if end_time > start_time:
        duration = end_time - start_time
    else:
        duration = (24 - start_time) + end_time

    return duration


def calculate_ci(r, n, conf_level=0.95):
    """Calculate confidence interval using Fisher's Z-transformation"""
    # Fisher's Z-transformation
    z = 0.5 * np.log((1 + r) / (1 - r))

    # Standard error of Z
    se_z = 1 / np.sqrt(n - 3)

    # Critical value
    critical_value = abs(norm.ppf((1 - conf_level) / 2))

    # Confidence interval for Z
    z_lower = z - critical_value * se_z
    z_upper = z + critical_value * se_z

    # Transform back to r
    r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
    r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)

    return r_lower, r_upper

def calculate_correlations(sleep_file, activity_file, min_days=1):
    # Read sleep data
    sleep_df = pd.read_csv(sleep_file)
    sleep_df['sleep_duration'] = sleep_df['sleep_time'].apply(parse_sleep_time)

    # Read activity data
    activity_df = pd.read_csv(activity_file)
    activity_df['timestamp'] = pd.to_datetime(activity_df['timestamp']
                                 .str.replace('EST', '')  # Remove EST timezone indicator
                                 .str.strip(),  # Remove any extra whitespace
                                 format='mixed',  # Allow mixed formats
                                 utc=False)
    activity_df['date'] = activity_df['timestamp'].dt.strftime('%-m/%-d/25')
    activity_df['topic'] = activity_df['topic_compressed']

    # Create daily topic proportions and counts
    daily_topics = []
    daily_counts = []  # Add this to track daily counts

    for date in sleep_df['date']:
        day_activities = activity_df[activity_df['date'] == date]
        if len(day_activities) > 0:
            topic_counts = day_activities['topic'].value_counts()
            topic_props = topic_counts / len(day_activities)
            daily_topics.append(pd.Series(topic_props, name=date))
            daily_counts.append(pd.Series(topic_counts, name=date))

    # Create topic proportion and count matrices
    topic_matrix = pd.DataFrame(daily_topics).fillna(0)
    count_matrix = pd.DataFrame(daily_counts).fillna(0)

    # Calculate correlations for each topic
    correlations = []
    sleep_durations = sleep_df['sleep_duration'].values

    for topic in topic_matrix.columns:
        topic_props = topic_matrix[topic].values
        topic_counts = count_matrix[topic].values
        days_present = np.sum(topic_props > 0)

        if days_present >= min_days:
            correlation, p_value = stats.pearsonr(topic_props, sleep_durations)
            mean_sleep_with = np.mean(sleep_durations[topic_props > 0])
            mean_sleep_without = np.mean(sleep_durations[topic_props == 0])
            avg_freq = np.mean(topic_counts[topic_counts > 0])  # Average frequency when topic appears
            confidence_interval = calculate_ci(correlation, days_present)

            correlations.append({
                'topic': topic,
                'correlation': correlation,
                'p_value': p_value,
                'days_present': days_present,
                'mean_sleep_with': mean_sleep_with,
                'mean_sleep_without': mean_sleep_without,
                'sleep_diff': mean_sleep_with - mean_sleep_without,
                'avg_freq': avg_freq,  # Add average frequency
                'confidence_interval': tuple(round(x, 3) for x in confidence_interval)
            })

    # Convert to DataFrame and sort by absolute correlation
    corr_df = pd.DataFrame(correlations)
    corr_df['abs_correlation'] = abs(corr_df['correlation'])
    corr_df = corr_df.sort_values('abs_correlation', ascending=False)

    return corr_df


def plot_correlations(corr_df, output_file='topic_correlations.png', min_days=3):
    # Filter by minimum days
    plot_df = corr_df[corr_df['days_present'] >= min_days].copy()

    # Sort by absolute correlation
    plot_df = plot_df.sort_values('abs_correlation', ascending=True)

    # Create figure
    plt.figure(figsize=(12, max(8, len(plot_df) * 0.3)))

    # Create horizontal bar plot
    bars = plt.barh(range(len(plot_df)), plot_df['correlation'])

    # Color bars based on correlation direction and significance
    for i, (_, row) in enumerate(plot_df.iterrows()):
        color = 'red' if row['correlation'] < 0 else 'blue'
        alpha = 1.0 if row['p_value'] < 0.05 else 0.5
        bars[i].set_color(color)
        bars[i].set_alpha(alpha)

    # Add topic labels with days
    plt.yticks(range(len(plot_df)),
               [f"{topic} (n={days})" for topic, days in zip(plot_df['topic'], plot_df['days_present'])])

    # Add vertical line at x=0
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

    # Add correlation values
    for i, row in enumerate(plot_df.itertuples()):
        # Add correlation value
        plt.text(row.correlation + (0.002 if row.correlation >= 0 else -0.002),
                 i,
                 f'{row.correlation:.3f}',
                 fontsize=8,
                 va='center',
                 ha='left' if row.correlation >= 0 else 'right')

    # Customize plot
    plt.xlabel('Correlation with Sleep Duration')
    plt.title(f'Topic-Sleep Duration Correlations (min {min_days} days)')
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=1.0, label='Positive correlation (p < 0.05)'),
        Patch(facecolor='blue', alpha=0.5, label='Positive correlation (p ≥ 0.05)'),
        Patch(facecolor='red', alpha=1.0, label='Negative correlation (p < 0.05)'),
        Patch(facecolor='red', alpha=0.5, label='Negative correlation (p ≥ 0.05)')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')


def main():
    # Calculate correlations
    correlations = calculate_correlations('../data/sleep_ground_truths.csv', '../data/categorized_output_compressed.csv', 20)

    # Print detailed results
    pd.set_option('display.max_rows', None)
    print("\nAll Topic-Sleep Correlations (sorted by absolute correlation):")
    print("=" * 100)
    results_df = correlations[['topic', 'correlation', 'p_value', 'days_present',
                               'mean_sleep_with', 'mean_sleep_without', 'sleep_diff', 'avg_freq', 'confidence_interval']]
    results_df = results_df.round(3)
    print(results_df.to_string(index=False))

    # Plot correlations for topics with at least 3 days
    plot_correlations(correlations, min_days=3)
    print("\nPlot saved as 'topic_correlations.png'")

    # Show strong and significant correlations
    strong_corr = correlations[
        (abs(correlations['correlation']) > 0.2) &
        (correlations['days_present'] >= 3)
        ]
    if len(strong_corr) > 0:
        print("\nStrong Correlations (|r| > 0.2, min 3 days):")
        print("=" * 100)
        strong_results = strong_corr[['topic', 'correlation', 'p_value', 'days_present',
                                      'mean_sleep_with', 'mean_sleep_without', 'sleep_diff', 'avg_freq', 'confidence_interval']]
        strong_results = strong_results.round(3)
        print(strong_results.to_string(index=False))


if __name__ == "__main__":
    main()