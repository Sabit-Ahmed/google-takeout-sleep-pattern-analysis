import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns


def analyze_daily_screen_time(df, date=None):
    """
    Analyze screen time for a specific date.
    If date is None, uses the earliest date in the dataset.

    Args:
        df: DataFrame with screen time data
        date: str in format 'YYYY-MM-DD' or datetime object
    """
    # Convert timestamp columns to datetime
    timestamp_cols = ['start_time', 'end_time', 'created_at']
    for col in timestamp_cols:
        df[col] = pd.to_datetime(df[col], unit='s')

    # Add date column for easier filtering
    df['date'] = df['start_time'].dt.date

    # If no date specified, use the earliest date
    if date is None:
        date = df['date'].min()
    elif isinstance(date, str):
        date = pd.to_datetime(date).date()

    # Filter for the specific date
    daily_df = df[df['date'] == date]

    if len(daily_df) == 0:
        print(f"No data found for date: {date}")
        print("Available dates:", sorted(df['date'].unique()))
        return None

    print(f"\nAnalyzing screen time for: {date}")

    # Calculate statistics
    app_usage = daily_df.groupby('app').agg({
        'usage': ['sum', 'mean', 'count']
    }).round(2)
    app_usage.columns = ['Total Minutes', 'Average Minutes per Session', 'Number of Sessions']

    # Calculate time range for the day
    day_start = pd.to_datetime(datetime.combine(date, datetime.min.time()))
    day_end = pd.to_datetime(datetime.combine(date, datetime.max.time()))

    # Create visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Set a color palette
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99CCFF']

    # App Usage Pie Chart
    app_labels = [app.split('.')[-1] for app in app_usage.index]
    ax1.pie(app_usage['Total Minutes'],
            labels=app_labels,
            autopct='%1.1f%%',
            colors=colors[:len(app_usage)],
            startangle=90)
    ax1.set_title(f'App Usage Distribution - {date}')

    # Timeline Chart (24-hour view)
    apps = daily_df['app'].unique()
    for i, app in enumerate(apps):
        app_data = daily_df[daily_df['app'] == app]
        for _, row in app_data.iterrows():
            start_minutes = (row['start_time'] - day_start).total_seconds() / 60
            duration = row['usage']
            ax2.barh(
                y=i,
                width=duration,
                left=start_minutes,
                label=app.split('.')[-1] if start_minutes == 0 else "",
                alpha=0.7,
                color=colors[i % len(colors)]
            )

    ax2.set_yticks(range(len(apps)))
    ax2.set_yticklabels([app.split('.')[-1] for app in apps])
    ax2.set_xlabel('Minutes since midnight')
    ax2.set_title(f'App Usage Timeline - {date}')

    # Set x-axis to show hours
    ax2.set_xlim(0, 24 * 60)  # 24 hours in minutes
    hour_ticks = np.arange(0, 24 * 60 + 1, 60)
    ax2.set_xticks(hour_ticks)
    ax2.set_xticklabels([f'{int(x / 60):02d}:00' for x in hour_ticks], rotation=45)

    # Add legend without duplicates
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.tight_layout()

    # Print statistics
    print("\nApp Usage Statistics:")
    print(app_usage)

    total_usage = daily_df['usage'].sum()
    print(f"\nTotal app usage: {total_usage:.1f} minutes ({total_usage / 60:.1f} hours)")
    print(f"Number of app switches: {len(daily_df)}")

    # Calculate and print active hours
    active_hours = daily_df['start_time'].dt.hour.unique()
    print(f"Active hours: {sorted(active_hours)}")

    return fig


def is_likely_active_usage(row, min_duration=3):
    """
    Determine if an app session is likely active usage based on heuristics:
    - Duration is above minimum threshold (default 3 seconds)
    - Not a background-heavy app (like music players, download managers)

    Args:
        row: DataFrame row with app usage data
        min_duration: Minimum duration in seconds to consider as active usage
    """
    # List of apps that commonly run in background
    background_apps = [
        'com.apple.audiod',
        'com.apple.backupd',
        'com.apple.syncdefaultsd',
        'com.apple.CloudKit',
        'com.apple.security',
        'com.apple.preferences',
        'com.apple.Safari',
        'com.apple.systempreferences',
        'com.jetbrains.pycharm'
        # Add more as needed
    ]

    # Check if it's not a background app and meets minimum duration
    return (row['app'] not in background_apps and
            row['usage'] >= min_duration)


def analyze_active_screen_time(df, date=None, min_duration_seconds=3):
    """
    Analyze active screen time for a specific date, filtering out likely background usage.

    Args:
        df: DataFrame with screen time data
        date: str in format 'YYYY-MM-DD' or datetime object
        min_duration_seconds: Minimum duration to consider as active usage
    """
    # Convert timestamp columns to datetime
    timestamp_cols = ['start_time', 'end_time', 'created_at']
    for col in timestamp_cols:
        df[col] = pd.to_datetime(df[col], unit='s')

    # Add date column
    df['date'] = df['start_time'].dt.date

    # If no date specified, use earliest date
    if date is None:
        date = df['date'].min()
    elif isinstance(date, str):
        date = pd.to_datetime(date).date()

    # Filter for specific date
    daily_df = df[df['date'] == date].copy()

    if len(daily_df) == 0:
        print(f"No data found for date: {date}")
        print("Available dates:", sorted(df['date'].unique()))
        return None

    # Add active usage flag
    daily_df['is_active'] = daily_df.apply(
        lambda row: is_likely_active_usage(row, min_duration_seconds),
        axis=1
    )

    # Separate active and background usage
    active_df = daily_df[daily_df['is_active']]
    background_df = daily_df[~daily_df['is_active']]

    print(f"\nAnalyzing screen time for: {date}")
    print(f"Total sessions: {len(daily_df)}")
    print(f"Active sessions: {len(active_df)}")
    print(f"Background sessions: {len(background_df)}")

    # Calculate statistics for active usage
    app_usage = active_df.groupby('app').agg({
        'usage': ['sum', 'mean', 'count']
    }).round(2)
    app_usage.columns = ['Total Minutes', 'Average Minutes per Session', 'Number of Sessions']

    # Create visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Color palette
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99CCFF']

    # 1. Active Usage Pie Chart
    if len(active_df) > 0:
        app_labels = [app.split('.')[-1] for app in app_usage.index]
        ax1.pie(app_usage['Total Minutes'],
                labels=app_labels,
                autopct='%1.1f%%',
                colors=colors[:len(app_usage)],
                startangle=90)
        ax1.set_title(f'Active App Usage Distribution - {date}')

    # 2. Timeline of Active Usage
    day_start = pd.to_datetime(datetime.combine(date, datetime.min.time()))
    apps = active_df['app'].unique()

    for i, app in enumerate(apps):
        app_data = active_df[active_df['app'] == app]
        for _, row in app_data.iterrows():
            start_minutes = (row['start_time'] - day_start).total_seconds() / 60
            duration = row['usage']
            ax2.barh(
                y=i,
                width=duration,
                left=start_minutes,
                label=app.split('.')[-1] if start_minutes == 0 else "",
                alpha=0.7,
                color=colors[i % len(colors)]
            )

    ax2.set_yticks(range(len(apps)))
    ax2.set_yticklabels([app.split('.')[-1] for app in apps])
    ax2.set_xlabel('Minutes since midnight')
    ax2.set_title(f'Active App Usage Timeline - {date}')

    # Set x-axis to show hours
    ax2.set_xlim(0, 24 * 60)
    hour_ticks = np.arange(0, 24 * 60 + 1, 60)
    ax2.set_xticks(hour_ticks)
    ax2.set_xticklabels([f'{int(x / 60):02d}:00' for x in hour_ticks], rotation=45)

    # 3. Session Duration Distribution
    ax3.hist(active_df['usage'], bins=30, color='skyblue', alpha=0.7)
    ax3.set_xlabel('Session Duration (minutes)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Active Session Duration Distribution')

    # 4. Hourly Usage Pattern
    hourly_usage = active_df.groupby(active_df['start_time'].dt.hour)['usage'].sum()
    ax4.bar(hourly_usage.index, hourly_usage.values, color='skyblue', alpha=0.7)
    ax4.set_xlabel('Hour of Day')
    ax4.set_ylabel('Total Usage (minutes)')
    ax4.set_title('Hourly Active Usage Pattern')
    ax4.set_xticks(range(24))
    ax4.set_xticklabels([f'{i:02d}:00' for i in range(24)], rotation=45)

    plt.tight_layout()

    # Print statistics
    print("\nActive App Usage Statistics:")
    print(app_usage)

    total_active = active_df['usage'].sum()
    total_background = background_df['usage'].sum()
    print(f"\nTotal active usage: {total_active:.1f} minutes ({total_active / 60:.1f} hours)")
    print(f"Total background usage: {total_background:.1f} minutes ({total_background / 60:.1f} hours)")

    # Calculate and print active hours
    active_hours = active_df['start_time'].dt.hour.unique()
    print(f"Active hours: {sorted(active_hours)}")

    return fig


# Example usage:
if __name__ == "__main__":

    # Sample data
    data = pd.read_csv('../data/screen_time_data.csv')
    # First, check your DataFrame columns
    print("DataFrame columns:", data.columns)
    # fig = analyze_daily_screen_time(data, '2025-02-12')
    fig = analyze_active_screen_time(data, '2025-02-11')
    plt.show()  # To display the plots