import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns

sns.set_style("whitegrid")  # or another style like "darkgrid", "white", "dark", "ticks"

def analyze_sleep_patterns(csv_file, start_date, end_date):
    """
    Analyze sleep patterns from activity data between given dates.
    """
    # Read the CSV file
    print(f"Reading file: {csv_file}")
    df = pd.read_csv(csv_file)

    # Convert timestamps to datetime, handling potential format issues
    try:
        # First, try parsing with dateutil parser (flexible parsing)
        df['timestamp'] = pd.to_datetime(df['timestamp']
        .str.replace('EST', '')  # Remove EST timezone indicator
        .str.strip(),            # Remove any extra whitespace
        format='mixed',          # Allow mixed formats
        utc=False)
    except Exception as e:
        print(f"Error in first parsing attempt: {e}")
        try:
            # If that fails, try forcing timezone naive datetime
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        except Exception as e:
            print(f"Error in second parsing attempt: {e}")
            return None, None

    # Drop any rows with NULL timestamps
    original_len = len(df)
    df = df.dropna(subset=['timestamp'])
    print(f"Rows after dropping NaN: {len(df)} (dropped {original_len - len(df)} rows)")

    # Filter for the specified week
    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date)
    mask = (df['timestamp'] >= start_datetime) & (df['timestamp'] < end_datetime)
    df_week = df.loc[mask].copy()

    print(f"\nRows in specified week: {len(df_week)}")

    if len(df_week) == 0:
        raise ValueError(f"No data found between {start_date} and {end_date}")

    # Sort by timestamp
    df_week = df_week.sort_values('timestamp')

    # Initialize data structures for analysis
    hours_inactive = {i: 0 for i in range(24)}
    inactive_durations = {i: [] for i in range(24)}

    # Analyze each day in the week
    current_date = start_datetime
    days_analyzed = 0

    while current_date < end_datetime:
        next_date = current_date + timedelta(days=1)

        # Get activities for the current day
        day_activities = df_week[
            (df_week['timestamp'] >= current_date) &
            (df_week['timestamp'] < next_date)
            ]

        # Check each hour for activity
        for hour in range(24):
            hour_start = current_date + timedelta(hours=hour)
            hour_end = hour_start + timedelta(hours=1)

            hour_activities = day_activities[
                (day_activities['timestamp'] >= hour_start) &
                (day_activities['timestamp'] < hour_end)
                ]

            if len(hour_activities) == 0:
                hours_inactive[hour] += 1

        current_date = next_date
        days_analyzed += 1

    # Calculate inactive periods
    df_week = df_week.reset_index(drop=True)
    print("\nCalculating inactive periods...")

    for i in range(len(df_week) - 1):
        try:
            current_time = pd.to_datetime(df_week.iloc[i]['timestamp'])
            next_time = pd.to_datetime(df_week.iloc[i + 1]['timestamp'])

            if pd.isna(current_time) or pd.isna(next_time):
                continue

            gap = (next_time - current_time).total_seconds() / 3600

            if gap >= 2:  # Consider gaps of 2+ hours
                start_hour = current_time.hour
                inactive_durations[start_hour].append(gap)
        except Exception as e:
            print(f"Warning: Error calculating gap at index {i}: {e}")
            continue

    # Calculate percentages and average durations
    inactive_percentages = {
        hour: (count / days_analyzed) * 100
        for hour, count in hours_inactive.items()
    }

    avg_durations = {
        hour: sum(durations) / len(durations) if durations else 0
        for hour, durations in inactive_durations.items()
    }

    # Create visualization
    fig, ax1 = plt.subplots(figsize=(15, 8))
    ax2 = ax1.twinx()

    x = range(24)
    percentages = [inactive_percentages[i] for i in x]
    durations = [avg_durations[i] for i in x]

    width = 0.35
    bars1 = ax1.bar([i - width / 2 for i in x], percentages, width, alpha=0.7,
                    color='royalblue', label='% Days Inactive')
    bars2 = ax2.bar([i + width / 2 for i in x], durations, width, alpha=0.7,
                    color='seagreen', label='Avg Duration When Inactive (hours)')

    ax1.set_xlabel('Hour of Day', fontsize=12)
    ax1.set_ylabel('% Days Inactive', color='royalblue', fontsize=12)
    ax2.set_ylabel('Average Hours When Inactive', color='seagreen', fontsize=12)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{i:02d}:00' for i in x], rotation=45)

    ax1.tick_params(axis='y', labelcolor='royalblue')
    ax2.tick_params(axis='y', labelcolor='seagreen')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title(f'Sleep Pattern Analysis ({start_date} to {end_date})',
              fontsize=14, pad=20)

    plt.tight_layout()

    results = {
        'inactive_percentages': inactive_percentages,
        'avg_durations': avg_durations,
        'days_analyzed': days_analyzed
    }

    return results, plt.gcf()


def print_analysis_summary(results):
    """Print a summary of the sleep pattern analysis."""
    if results is None:
        print("No results to summarize")
        return

    print("\nSleep Pattern Analysis Summary:")
    print("-" * 50)

    inactive_hours = [
        hour for hour, pct in results['inactive_percentages'].items()
        if pct >= 90
    ]
    if inactive_hours:
        consecutive_ranges = []
        current_range = [inactive_hours[0]]

        for i in range(1, len(inactive_hours)):
            if inactive_hours[i] == inactive_hours[i - 1] + 1:
                current_range.append(inactive_hours[i])
            else:
                consecutive_ranges.append(current_range)
                current_range = [inactive_hours[i]]
        consecutive_ranges.append(current_range)

        print("\nHighly Inactive Hours (>= 90%):")
        for range_hours in consecutive_ranges:
            start = f"{range_hours[0]:02d}:00"
            end = f"{range_hours[-1] + 1:02d}:00"
            print(f"  {start} - {end}")

    long_periods = {
        hour: duration for hour, duration in results['avg_durations'].items()
        if duration > 4
    }
    if long_periods:
        print("\nLongest Inactive Periods:")
        for hour, duration in sorted(
                long_periods.items(),
                key=lambda x: x[1],
                reverse=True
        )[:3]:
            print(f"  Starting at {hour:02d}:00: {duration:.1f} hours")


if __name__ == "__main__":
    try:
        results, fig = analyze_sleep_patterns(
            '../data/categorized_output.csv',
            '2025-01-06',
            '2025-01-12'
        )
        if results:
            print_analysis_summary(results)
            plt.show()
    except Exception as e:
        print(f"Error: {e}")