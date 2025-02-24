import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from pandas import unique
from scipy import stats


def analyze_sleep_patterns(csv_file):
    """
    Analyze sleep patterns from activity timestamps.

    Args:
        csv_file (str): Path to CSV file containing activity data

    Returns:
        dict: Analysis results including statistics and hypothesis test
    """
    # Read CSV file
    df = pd.read_csv(csv_file)

    # Convert timestamp strings to datetime objects
    df['timestamp'] = pd.to_datetime(
        df['timestamp'].astype(str)  # Convert to string first
        .str.replace('EST', '')  # Remove EST timezone indicator
        .str.strip(),  # Remove any extra whitespace
        format='mixed',  # Allow mixed formats
        utc=False
    )

    # Sort by timestamp
    df = df.sort_values('timestamp')
    print(len(df['timestamp'].dt.date.unique()))

    def is_night_time(time):
        """Check if time is between 10 PM and 10 AM"""
        hour = time.hour
        return hour >= 22 or hour <= 10

    def is_weekend(date):
        """Check if date is weekend (0 = Monday, 6 = Sunday)"""
        return date.weekday() >= 5

    # Find gaps in activity that might represent sleep
    sleep_periods = []

    for i in range(1, len(df)):
        last_activity = df['timestamp'].iloc[i - 1]
        current_activity = df['timestamp'].iloc[i]

        time_diff = current_activity - last_activity
        hours_diff = time_diff.total_seconds() / 3600

        # Only consider gaps > 2 hours that START during night hours (10 PM - 10 AM)
        # AND END during night hours
        if (hours_diff > 2 and
                is_night_time(last_activity) and
                is_night_time(current_activity)):

            sleep_duration = hours_diff

            # Only consider reasonable sleep durations (3-12 hours)
            if 3 <= sleep_duration <= 12:
                sleep_periods.append({
                    'start': last_activity,
                    'end': current_activity,
                    'duration': sleep_duration,
                    'is_weekend': is_weekend(last_activity)
                })

    # Convert to DataFrame for easier analysis
    sleep_df = pd.DataFrame(sleep_periods)

    # If no sleep periods found, return empty results
    if len(sleep_periods) == 0:
        return {
            'sleep_periods': pd.DataFrame(),
            'statistics': {
                'weekday': {'count': 0, 'mean': 0, 'std': 0, 'median': 0},
                'weekend': {'count': 0, 'mean': 0, 'std': 0, 'median': 0}
            },
            'test_results': {
                't_statistic': 0,
                'p_value': 1,
                'degrees_of_freedom': 0
            },
            'distribution': {
                'min': 0, 'q1': 0, 'median': 0, 'q3': 0, 'max': 0
            }
        }

    # Separate weekend and weekday sleep
    weekday_sleep = sleep_df[~sleep_df['is_weekend']]['duration']
    weekend_sleep = sleep_df[sleep_df['is_weekend']]['duration']

    # Calculate basic statistics
    stats_dict = {
        'weekday': {
            'count': len(weekday_sleep),
            'mean': weekday_sleep.mean() if len(weekday_sleep) > 0 else 0,
            'std': weekday_sleep.std() if len(weekday_sleep) > 0 else 0,
            'median': weekday_sleep.median() if len(weekday_sleep) > 0 else 0
        },
        'weekend': {
            'count': len(weekend_sleep),
            'mean': weekend_sleep.mean() if len(weekend_sleep) > 0 else 0,
            'std': weekend_sleep.std() if len(weekend_sleep) > 0 else 0,
            'median': weekend_sleep.median() if len(weekend_sleep) > 0 else 0
        }
    }

    # Perform Welch's t-test only if both weekday and weekend data exist
    if len(weekday_sleep) > 0 and len(weekend_sleep) > 0:
        t_stat, p_value = stats.ttest_ind(weekday_sleep, weekend_sleep, equal_var=False)

        # Calculate degrees of freedom (Welch–Satterthwaite equation)
        n1, n2 = len(weekday_sleep), len(weekend_sleep)
        var1, var2 = weekday_sleep.var(), weekend_sleep.var()
        df_value = ((var1 / n1 + var2 / n2) ** 2) / \
                   ((var1 ** 2) / (n1 ** 2 * (n1 - 1)) + (var2 ** 2) / (n2 ** 2 * (n2 - 1)))
    else:
        t_stat, p_value, df_value = 0, 1, 0

    # Calculate overall distribution statistics
    all_sleep = pd.concat([weekday_sleep, weekend_sleep])
    distribution = {
        'min': all_sleep.min(),
        'q1': all_sleep.quantile(0.25),
        'median': all_sleep.median(),
        'q3': all_sleep.quantile(0.75),
        'max': all_sleep.max()
    }

    return {
        'sleep_periods': sleep_df,
        'statistics': stats_dict,
        'test_results': {
            't_statistic': t_stat,
            'p_value': p_value,
            'degrees_of_freedom': df_value
        },
        'distribution': distribution
    }


def print_analysis_results(results):
    """Print formatted analysis results"""
    print("\nSleep Pattern Analysis Results")
    print("=" * 30)

    if len(results['sleep_periods']) == 0:
        print("\nNo valid sleep periods found in the data.")
        return

    print("\nDescriptive Statistics:")
    print("\nWeekday Sleep:")
    print(f"Number of periods: {results['statistics']['weekday']['count']}")
    print(f"Mean duration: {results['statistics']['weekday']['mean']:.2f} hours")
    print(f"Standard deviation: {results['statistics']['weekday']['std']:.2f} hours")
    print(f"Median duration: {results['statistics']['weekday']['median']:.2f} hours")

    print("\nWeekend Sleep:")
    print(f"Number of periods: {results['statistics']['weekend']['count']}")
    print(f"Mean duration: {results['statistics']['weekend']['mean']:.2f} hours")
    print(f"Standard deviation: {results['statistics']['weekend']['std']:.2f} hours")
    print(f"Median duration: {results['statistics']['weekend']['median']:.2f} hours")

    if results['test_results']['t_statistic'] != 0:
        print("\nHypothesis Test Results:")
        print(f"t-statistic: {results['test_results']['t_statistic']:.4f}")
        print(f"p-value: {results['test_results']['p_value']:.4f}")
        print(f"Degrees of freedom: {results['test_results']['degrees_of_freedom']:.1f}")

    print("\nOverall Sleep Distribution:")
    print(f"Minimum: {results['distribution']['min']:.2f} hours")
    print(f"25th percentile: {results['distribution']['q1']:.2f} hours")
    print(f"Median: {results['distribution']['median']:.2f} hours")
    print(f"75th percentile: {results['distribution']['q3']:.2f} hours")
    print(f"Maximum: {results['distribution']['max']:.2f} hours")


def create_sleep_boxplots(results):
    """
    Create a single boxplot comparing overall, weekday, and weekend sleep durations.

    Args:
        results: Dictionary containing sleep analysis results
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import f_oneway

    # Get the sleep DataFrame
    sleep_df = results['sleep_periods']

    # Prepare data for box plots
    # Create a new column for plotting that distinguishes overall, weekday, and weekend
    plot_df = pd.DataFrame({
        'Duration': pd.concat([
            sleep_df['duration'],  # Overall
            sleep_df[~sleep_df['is_weekend']]['duration'],  # Weekday
            sleep_df[sleep_df['is_weekend']]['duration']  # Weekend
        ]),
        'Category': ['Overall'] * len(sleep_df) +
                    ['Weekday'] * sum(~sleep_df['is_weekend']) +
                    ['Weekend'] * sum(sleep_df['is_weekend'])
    })

    # Set style and create figure
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # Create boxplot
    sns.boxplot(x='Category', y='Duration', data=plot_df,
                hue='Category', palette=['orange', 'red', 'lightgreen'],
                width=0.5, legend=False)

    plt.title('Sleep Duration Distribution')
    plt.xlabel('Category')
    plt.ylabel('Hours of Sleep')

    # Add statistics as text
    stats = results['statistics']
    all_stats = f"Overall (n={len(sleep_df)})\nMean: {sleep_df['duration'].mean():.2f}\nMedian: {sleep_df['duration'].median():.2f}"
    weekday_stats = f"Weekday (n={stats['weekday']['count']})\nMean: {stats['weekday']['mean']:.2f}\nMedian: {stats['weekday']['median']:.2f}"
    weekend_stats = f"Weekend (n={stats['weekend']['count']})\nMean: {stats['weekend']['mean']:.2f}\nMedian: {stats['weekend']['median']:.2f}"

    plt.text(2.1, plt.ylim()[0], all_stats + '\n\n' + weekday_stats + '\n\n' + weekend_stats,
             verticalalignment='bottom')

    # Perform one-way ANOVA
    weekday_sleep = sleep_df[~sleep_df['is_weekend']]['duration']
    weekend_sleep = sleep_df[sleep_df['is_weekend']]['duration']
    f_stat, anova_p = f_oneway(weekday_sleep, weekend_sleep)

    # Add ANOVA results to the plot
    # anova_text = f"One-way ANOVA:\nF-statistic: {f_stat:.4f}\np-value: {anova_p:.4f}"
    # plt.text(1.8, plt.ylim()[1], anova_text,
    #          verticalalignment='bottom')

    plt.tight_layout()
    plt.show()

    return f_stat, anova_p


# Update the main code to include ANOVA results
if __name__ == "__main__":
    # Analyze the data
    results = analyze_sleep_patterns('../data/categorized_output.csv')

    # Create boxplot and get ANOVA results
    f_stat, anova_p = create_sleep_boxplots(results)

    # Print results
    print_analysis_results(results)

    # Statistical conclusions
    if len(results['sleep_periods']) > 0:
        print("\nStatistical Analysis:")
        print("\n1. T-test Results:")
        alpha = 0.05
        if results['test_results']['p_value'] < alpha:
            print("Reject the null hypothesis.")
            print("There is significant evidence of a difference in sleep patterns between weekdays and weekends.")
            print(f"(t-test p-value = {results['test_results']['p_value']:.4f} < {alpha})")
        else:
            print("Fail to reject the null hypothesis.")
            print(
                "There is insufficient evidence to conclude a difference in sleep patterns between weekdays and weekends.")
            print(f"(t-test p-value = {results['test_results']['p_value']:.4f} > {alpha})")

        print("\n2. ANOVA Results:")
        if anova_p < alpha:
            print("Reject the null hypothesis.")
            print("There is significant evidence of a difference in sleep patterns between weekdays and weekends.")
            print(f"(ANOVA p-value = {anova_p:.4f} < {alpha})")
        else:
            print("Fail to reject the null hypothesis.")
            print(
                "There is insufficient evidence to conclude a difference in sleep patterns between weekdays and weekends.")
            print(f"(ANOVA p-value = {anova_p:.4f} > {alpha})")

        # Add coverage information
        total_days = (results['sleep_periods']['start'].max() - results['sleep_periods']['start'].min()).days + 1
        print(f"\nCoverage Analysis:")
        print(f"Total days in data: {total_days}")
        print(f"Days with detected sleep: {len(results['sleep_periods'])}")
        print(f"Coverage rate: {(len(results['sleep_periods']) / total_days) * 100:.1f}%")
    else:
        print("\nNo statistical tests could be performed due to insufficient data.")