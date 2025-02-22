import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta


def load_and_preprocess_data(file_path):
    """
    Load and preprocess the web activity data
    """
    # Read CSV file
    df = pd.read_csv(file_path)

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp']
                                              .str.replace('EST', '')  # Remove EST timezone indicator
                                              .str.strip(),  # Remove any extra whitespace
                                              format='mixed',  # Allow mixed formats
                                              utc=False)

    # Sort by timestamp
    df = df.sort_values('timestamp')

    # Add date column
    df['date'] = df['timestamp'].dt.date

    return df


def analyze_daily_patterns(df):
    """
    Analyze daily activity patterns and sleep duration
    """
    # Group by date
    daily_stats = []

    for date, group in df.groupby('date'):
        # Get first and last activity of the day
        first_activity = group['timestamp'].min()
        last_activity = group['timestamp'].max()

        # Get topic distribution for the day
        topic_counts = group['topic'].value_counts()
        main_topics = topic_counts.head(3)

        daily_stats.append({
            'date': date,
            'first_activity': first_activity,
            'last_activity': last_activity,
            'num_activities': len(group),
            'topics': dict(topic_counts),
            'main_topics': dict(main_topics)
        })

    # Convert to DataFrame
    daily_df = pd.DataFrame(daily_stats)

    # Calculate sleep duration between days
    sleep_patterns = []

    for i in range(len(daily_df) - 1):
        current_day = daily_df.iloc[i]
        next_day = daily_df.iloc[i + 1]

        sleep_start = current_day['last_activity']
        sleep_end = next_day['first_activity']
        sleep_duration = (sleep_end - sleep_start).total_seconds() / 3600  # Convert to hours

        if 3 <= sleep_duration <= 12:  # Filter reasonable sleep durations
            sleep_patterns.append({
                'date': current_day['date'],
                'sleep_start': sleep_start,
                'sleep_end': sleep_end,
                'sleep_duration': sleep_duration,
                'prev_day_activities': current_day['num_activities'],
                'prev_day_topics': current_day['topics'],
                'main_topics': current_day['main_topics']
            })

    return pd.DataFrame(sleep_patterns)


def analyze_topic_sleep_relations(sleep_df):
    """
    Analyze relationships between topics and sleep patterns
    """
    # Create dictionary to store sleep durations by topic
    topic_sleep = {}

    for _, row in sleep_df.iterrows():
        for topic, count in row['prev_day_topics'].items():
            if topic not in topic_sleep:
                topic_sleep[topic] = []
            topic_sleep[topic].append(row['sleep_duration'])

    # Filter topics with minimum occurrences
    min_occurrences = 3
    topic_stats = {}

    for topic, durations in topic_sleep.items():
        if len(durations) >= min_occurrences:
            topic_stats[topic] = {
                'count': len(durations),
                'mean': np.mean(durations),
                'std': np.std(durations),
                'min': np.min(durations),
                'max': np.max(durations)
            }

    return pd.DataFrame(topic_stats).T


def analyze_sleep_timing(sleep_df):
    """
    Analyze sleep timing patterns
    """
    sleep_df['sleep_start_hour'] = sleep_df['sleep_start'].dt.hour + sleep_df['sleep_start'].dt.minute / 60
    sleep_df['sleep_end_hour'] = sleep_df['sleep_end'].dt.hour + sleep_df['sleep_end'].dt.minute / 60

    timing_stats = {
        'mean_start': sleep_df['sleep_start_hour'].mean(),
        'std_start': sleep_df['sleep_start_hour'].std(),
        'mean_end': sleep_df['sleep_end_hour'].mean(),
        'std_end': sleep_df['sleep_end_hour'].std()
    }

    return timing_stats


def create_visualizations(sleep_df, topic_stats):
    """
    Create visualizations of sleep patterns
    """
    # Sleep duration distribution
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    sns.histplot(data=sleep_df, x='sleep_duration', bins=15)
    plt.title('Distribution of Sleep Duration')
    plt.xlabel('Sleep Duration (hours)')

    # Sleep start time distribution
    plt.subplot(2, 2, 2)
    sns.histplot(data=sleep_df, x='sleep_start_hour', bins=12)
    plt.title('Distribution of Sleep Start Times')
    plt.xlabel('Hour of Day (24-hour format)')

    # Activities vs Sleep Duration
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=sleep_df, x='prev_day_activities', y='sleep_duration')
    plt.title('Number of Activities vs Sleep Duration')
    plt.xlabel('Number of Activities')
    plt.ylabel('Sleep Duration (hours)')

    # Average sleep by top topics
    plt.subplot(2, 2, 4)
    topic_means = topic_stats.sort_values('mean', ascending=False).head(10)
    sns.barplot(x=topic_means.index, y='mean', data=topic_means)
    plt.xticks(rotation=45, ha='right')
    plt.title('Average Sleep Duration by Top Topics')
    plt.xlabel('Topic')
    plt.ylabel('Average Sleep Duration (hours)')

    plt.tight_layout()
    plt.show()


def main():
    # Load and process data
    df = load_and_preprocess_data('../data/categorized_output.csv')

    # Analyze daily patterns
    sleep_df = analyze_daily_patterns(df)

    # Analyze topic relationships
    topic_stats = analyze_topic_sleep_relations(sleep_df)

    # Analyze timing patterns
    timing_stats = analyze_sleep_timing(sleep_df)

    # Print summary statistics
    print("Sleep Pattern Analysis Results")
    print("=============================")
    print(f"\nTotal nights analyzed: {len(sleep_df)}")
    print(
        f"Average sleep duration: {sleep_df['sleep_duration'].mean():.2f} ± {sleep_df['sleep_duration'].std():.2f} hours")

    print("\nSleep Timing Statistics")
    print(f"Average sleep start: {timing_stats['mean_start']:.2f} ± {timing_stats['std_start']:.2f} (24-hour format)")
    print(f"Average sleep end: {timing_stats['mean_end']:.2f} ± {timing_stats['std_end']:.2f} (24-hour format)")

    print("\nTop Topics by Average Sleep Duration (min 3 occurrences):")
    print(topic_stats.sort_values('mean', ascending=False).head(10)[['count', 'mean', 'std']])

    # Calculate correlation between activities and sleep duration
    correlation, p_value = stats.pearsonr(sleep_df['prev_day_activities'], sleep_df['sleep_duration'])
    print(f"\nActivity-Sleep Duration Correlation:")
    print(f"r = {correlation:.3f}, p = {p_value:.3f}")

    # Create visualizations
    create_visualizations(sleep_df, topic_stats)


if __name__ == "__main__":
    main()