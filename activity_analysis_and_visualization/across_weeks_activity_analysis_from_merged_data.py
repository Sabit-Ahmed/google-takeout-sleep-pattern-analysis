import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple


def analyze_consistent_sleep_hours(weeks_data: List[Dict]) -> Dict:
    """
    Analyze which hours are consistently inactive across weeks.

    Args:
        weeks_data: List of dictionaries containing weekly sleep data

    Returns:
        Dictionary with analysis of consistent sleep hours
    """
    # Stack all weeks' inactivity percentages
    all_percentages = np.array([week['inactivePercentages'] for week in weeks_data])

    # Calculate mean and std of inactivity for each hour
    hourly_means = np.mean(all_percentages, axis=0)
    hourly_stds = np.std(all_percentages, axis=0)

    # Find highly inactive hours (mean > 90%)
    consistent_hours = []
    for hour, (mean, std) in enumerate(zip(hourly_means, hourly_stds)):
        if mean >= 90:
            consistent_hours.append({
                'hour': hour,
                'mean_inactivity': mean,
                'std_inactivity': std
            })

    # Find consecutive ranges
    ranges = []
    current_range = [consistent_hours[0]['hour']]

    for i in range(1, len(consistent_hours)):
        if consistent_hours[i]['hour'] == consistent_hours[i - 1]['hour'] + 1:
            current_range.append(consistent_hours[i]['hour'])
        else:
            ranges.append(current_range)
            current_range = [consistent_hours[i]['hour']]
    ranges.append(current_range)

    return {
        'consistent_hours': consistent_hours,
        'ranges': ranges,
        'hourly_means': hourly_means.tolist(),
        'hourly_stds': hourly_stds.tolist()
    }


def analyze_weekly_variations(weeks_data: List[Dict]) -> Dict:
    """
    Analyze how sleep patterns vary week to week.

    Args:
        weeks_data: List of dictionaries containing weekly sleep data

    Returns:
        Dictionary with analysis of weekly variations
    """
    weekly_stats = []

    for week in weeks_data:
        inactive_percentages = np.array(week['inactivePercentages'])
        avg_durations = np.array(week['avgDurations'])

        stats = {
            'week_start': week['startDate'],
            'week_end': week['endDate'],
            'mean_inactivity': np.mean(inactive_percentages),
            'std_inactivity': np.std(inactive_percentages),
            'max_duration': np.max(avg_durations),
            'mean_duration': np.mean(avg_durations[avg_durations > 0]),
            'consistency_score': 100 - np.std(inactive_percentages)  # Higher score = more consistent
        }
        weekly_stats.append(stats)

    return {
        'weekly_stats': weekly_stats,
        'most_consistent_week': max(weekly_stats, key=lambda x: x['consistency_score']),
        'least_consistent_week': min(weekly_stats, key=lambda x: x['consistency_score'])
    }


def analyze_long_inactive_periods(weeks_data: List[Dict]) -> Dict:
    """
    Analyze patterns in long inactive periods.

    Args:
        weeks_data: List of dictionaries containing weekly sleep data

    Returns:
        Dictionary with analysis of long inactive periods
    """
    all_durations = {hour: [] for hour in range(24)}

    # Collect all durations by start hour
    for week in weeks_data:
        for hour, duration in enumerate(week['avgDurations']):
            if duration > 0:
                all_durations[hour].append(duration)

    # Calculate statistics for each hour
    hourly_stats = {}
    for hour, durations in all_durations.items():
        if durations:
            hourly_stats[hour] = {
                'mean_duration': np.mean(durations),
                'max_duration': np.max(durations),
                'frequency': len(durations),
                'std_duration': np.std(durations) if len(durations) > 1 else 0
            }

    # Find hours with consistently long durations
    long_duration_hours = {
        hour: stats for hour, stats in hourly_stats.items()
        if stats['mean_duration'] >= 6  # Consider periods over 6 hours as long
    }

    return {
        'hourly_stats': hourly_stats,
        'long_duration_hours': long_duration_hours
    }


def generate_weekly_summary(csv_file: str) -> str:
    """
    Generate a comprehensive weekly summary of sleep patterns.

    Args:
        csv_file: Path to the CSV file containing activity data

    Returns:
        String containing the formatted analysis summary
    """
    # Read and process data
    df = pd.read_csv(csv_file)
    # Remove EST timezone indicator
    df['timestamp'] = pd.to_datetime(df['timestamp'].str.replace('EST', '')
                                     .str.strip(), format='mixed', utc=False)

    # Group by week and analyze
    weeks_data = []
    min_date = df['timestamp'].min()
    max_date = df['timestamp'].max()

    current_date = min_date - pd.Timedelta(days=min_date.dayofweek)

    while current_date < max_date:
        week_end = current_date + pd.Timedelta(days=7)
        week_data = df[(df['timestamp'] >= current_date) & (df['timestamp'] < week_end)]

        if not week_data.empty:
            # Calculate hourly inactivity
            hourly_inactivity = []
            hourly_durations = []

            for hour in range(24):
                hour_mask = week_data['timestamp'].dt.hour == hour
                days_with_activity = week_data[hour_mask]['timestamp'].dt.date.nunique()
                inactivity_percentage = ((7 - days_with_activity) / 7) * 100
                hourly_inactivity.append(inactivity_percentage)

                # Calculate durations
                hour_data = week_data[hour_mask].copy()
                if not hour_data.empty:
                    hour_data = hour_data.sort_values('timestamp')
                    gaps = hour_data['timestamp'].diff().dt.total_seconds() / 3600
                    avg_duration = gaps[gaps >= 2].mean() if any(gaps >= 2) else 0
                    hourly_durations.append(avg_duration)
                else:
                    hourly_durations.append(0)

            weeks_data.append({
                'startDate': current_date.strftime('%Y-%m-%d'),
                'endDate': week_end.strftime('%Y-%m-%d'),
                'inactivePercentages': hourly_inactivity,
                'avgDurations': hourly_durations
            })

        current_date = week_end

    # Run analyses
    consistent_sleep = analyze_consistent_sleep_hours(weeks_data)
    weekly_var = analyze_weekly_variations(weeks_data)
    long_periods = analyze_long_inactive_periods(weeks_data)

    # Generate summary
    summary = []
    summary.append("=== Sleep Pattern Analysis Summary ===\n")

    # Consistent Sleep Hours
    summary.append("1. Consistent Sleep Hours:")
    for range_hours in consistent_sleep['ranges']:
        start_hour = f"{range_hours[0]:02d}:00"
        end_hour = f"{range_hours[-1]:02d}:59"
        summary.append(f"   * Consistent inactivity {start_hour}-{end_hour}")

    # Weekly Variations
    summary.append("\n2. Weekly Variations:")
    most_consistent = weekly_var['most_consistent_week']
    summary.append(f"   * Most consistent week: {most_consistent['week_start']} to {most_consistent['week_end']}")
    summary.append(f"   * Consistency score: {most_consistent['consistency_score']:.1f}/100")

    # Long Inactive Periods
    summary.append("\n3. Long Inactive Periods:")
    for hour, stats in sorted(
            long_periods['long_duration_hours'].items(),
            key=lambda x: x[1]['mean_duration'],
            reverse=True
    )[:3]:
        summary.append(
            f"   * Starting at {hour:02d}:00: "
            f"Average {stats['mean_duration']:.1f} hours "
            f"(occurred {stats['frequency']} times)"
        )

    return "\n".join(summary)


def visualize_hourly_inactivity(csv_file: str):
    """
    Create a heatmap visualization of hourly inactivity patterns.

    Args:
        csv_file: Path to the CSV file containing activity data
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    # Read and process data
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'].str.replace('EST', '')
                                     .str.strip(), format='mixed', utc=False)

    # Group by week and analyze
    weeks_data = []
    min_date = df['timestamp'].min()
    max_date = df['timestamp'].max()

    current_date = min_date - pd.Timedelta(days=min_date.dayofweek)

    while current_date < max_date:
        week_end = current_date + pd.Timedelta(days=7)
        week_data = df[(df['timestamp'] >= current_date) & (df['timestamp'] < week_end)]

        if not week_data.empty:
            # Calculate hourly inactivity
            hourly_inactivity = []

            for hour in range(24):
                hour_mask = week_data['timestamp'].dt.hour == hour
                days_with_activity = week_data[hour_mask]['timestamp'].dt.date.nunique()
                inactivity_percentage = ((days_with_activity) / 7) * 100
                hourly_inactivity.append(inactivity_percentage)

            weeks_data.append({
                'week': current_date.strftime('%Y-%m-%d'),
                'inactivity': hourly_inactivity
            })

        current_date = week_end

    # Create DataFrame for heatmap
    heatmap_data = pd.DataFrame([week['inactivity'] for week in weeks_data],
                                index=[week['week'] for week in weeks_data],
                                columns=[f"{h:02d}:00" for h in range(24)])

    # Set up the matplotlib figure
    plt.figure(figsize=(15, 8))

    # Create custom colormap (white to blue)
    colors = ['#ffffff', '#d4e6f1', '#a9cce3', '#7fb3d5', '#5499c7', '#2980b9', '#1b4f72']
    cmap = LinearSegmentedColormap.from_list('custom_blues', colors)

    # Create heatmap
    sns.heatmap(heatmap_data,
                cmap=cmap,
                annot=True,  # Show values in cells
                fmt='.0f',  # Format as integer
                cbar_kws={'label': 'Activity %'},
                vmin=0,  # Minimum value for color scale
                vmax=100)  # Maximum value for color scale

    # Customize the plot
    plt.title('Hourly Activity Patterns Across Weeks')
    plt.xlabel('Hour of Day')
    plt.ylabel('Week Starting')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Show plot
    plt.show()


if __name__ == "__main__":
    # Example usage
    summary = generate_weekly_summary('../data/categorized_output.csv')
    heatmap_data = visualize_hourly_inactivity('../data/categorized_output.csv')
    print(summary)