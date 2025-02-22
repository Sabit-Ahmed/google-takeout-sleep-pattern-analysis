import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import circmean, circvar
import astropy.stats as astrostats
from typing import Dict
import statsmodels.api as sm


def analyze_temporal_patterns(csv_file: str) -> Dict:
    """
    Analyze temporal patterns using appropriate statistical methods for cyclical time data.
    """
    # Read data
    df = pd.read_csv(csv_file)

    # First, let's look at the timestamp format
    print("\nSample of original timestamps:")
    print(df['timestamp'].head())

    try:
        # Clean and convert timestamps with better error handling
        df['timestamp'] = pd.to_datetime(
            df['timestamp'].astype(str)  # Convert to string first
            .str.replace('EST', '')  # Remove EST timezone indicator
            .str.strip(),  # Remove any extra whitespace
            format='mixed',  # Allow mixed formats
            utc=False
        )

        # Sort by timestamp
        df = df.sort_values('timestamp')

        # Handle duplicate timestamps by adding milliseconds
        if df['timestamp'].duplicated().any():
            df['tmp_idx'] = range(len(df))
            df['timestamp'] = df['timestamp'] + pd.to_timedelta(df['tmp_idx'], unit='ms')
            df = df.drop('tmp_idx', axis=1)

        print(f"\nDate range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Total activities: {len(df)}")

        # Extract hour and convert to radians for circular statistics
        df['hour'] = df['timestamp'].dt.hour
        hours_rad = df['hour'] * 2 * np.pi / 24

        # 1. Circular Statistics
        try:
            circular_stats = {
                'mean_hour': (circmean(hours_rad, high=2 * np.pi, low=0) * 24 / (2 * np.pi)) % 24,
                'variance': circvar(hours_rad, high=2 * np.pi, low=0),
            }
        except Exception as e:
            print(f"Warning: Error calculating circular statistics: {e}")
            circular_stats = {
                'mean_hour': np.nan,
                'variance': np.nan
            }

        # 2. Rayleigh Test for circular uniformity
        try:
            rayleigh_p = astrostats.rayleightest(hours_rad)
        except Exception as e:
            print(f"Warning: Error in Rayleigh test: {e}")
            rayleigh_p = np.nan

        # 3. Time Series Analysis
        # Count activities per hour using crosstab
        activity_counts = pd.crosstab(
            df['timestamp'].dt.date,
            df['timestamp'].dt.hour
        ).fillna(0)

        # Calculate autocorrelation
        try:
            mean_daily_pattern = activity_counts.mean()
            acf = sm.tsa.acf(mean_daily_pattern, nlags=23)
            significant_lags = [i for i, corr in enumerate(acf) if abs(corr) > 2 / np.sqrt(len(mean_daily_pattern))]
        except Exception as e:
            print(f"Warning: Error calculating autocorrelation: {e}")
            acf = []
            significant_lags = []

        # 4. Daily Pattern Analysis
        daily_stats = df.groupby(df['timestamp'].dt.date).size()
        daily_pattern = {
            'mean_daily_activities': daily_stats.mean(),
            'std_daily_activities': daily_stats.std(),
            'cv_daily_activities': daily_stats.std() / daily_stats.mean() if daily_stats.mean() > 0 else np.nan,
            'min_daily_activities': daily_stats.min(),
            'max_daily_activities': daily_stats.max()
        }

        # 5. Gap Analysis
        gaps = df['timestamp'].diff().dt.total_seconds() / 3600

        gap_threshold = gaps.mean() + 2 * gaps.std()
        significant_gaps = gaps[gaps > gap_threshold]

        gap_start_hours = []
        if not significant_gaps.empty:
            gap_indices = significant_gaps.index
            gap_start_hours = [df.iloc[i - 1]['hour'] for i in gap_indices if i > 0]

        # Day of week patterns
        df['day_of_week'] = df['timestamp'].dt.day_name()
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_stats = df.groupby('day_of_week').size().reindex(days_order)

        # Hourly patterns
        hourly_stats = df.groupby('hour').size()

        results = {
            'circular_statistics': {
                'mean_activity_hour': circular_stats['mean_hour'],
                'circular_variance': circular_stats['variance'],
                'is_uniform': rayleigh_p > 0.05 if not np.isnan(rayleigh_p) else None,
                'rayleigh_p_value': rayleigh_p
            },
            'temporal_patterns': {
                'significant_periodicities': significant_lags,
                'autocorrelation': acf.tolist() if len(acf) > 0 else [],
                'daily_pattern': daily_pattern
            },
            'gap_analysis': {
                'mean_gap': gaps.mean(),
                'median_gap': gaps.median(),
                'gap_threshold': gap_threshold,
                'significant_gaps_count': len(significant_gaps),
                'significant_gap_hours': gap_start_hours,
                'max_gap': gaps.max(),
                'min_gap': gaps.min()
            },
            'hourly_patterns': {
                'average_pattern': hourly_stats.to_dict(),
                'peak_hours': hourly_stats.nlargest(3).index.tolist(),
                'quiet_hours': hourly_stats.nsmallest(3).index.tolist()
            },
            'day_of_week_patterns': dow_stats.to_dict()
        }

        return results

    except Exception as e:
        print(f"Error processing timestamps: {e}")
        print("Sample of problematic timestamps:")
        print(df['timestamp'].head())
        raise


def print_temporal_analysis(analysis: Dict) -> str:
    """Format temporal analysis results as readable text."""
    summary = []
    summary.append("\n=== Temporal Pattern Analysis ===\n")

    # Circular Statistics
    circ_stats = analysis['circular_statistics']
    summary.append("Circular Statistics:")
    if not np.isnan(circ_stats['mean_activity_hour']):
        summary.append(f"  Mean activity hour: {circ_stats['mean_activity_hour']:.1f}")

    if circ_stats['is_uniform'] is not None:
        summary.append(f"  Activity distribution is {'uniform' if circ_stats['is_uniform'] else 'non-uniform'} " +
                       f"(Rayleigh test p={circ_stats['rayleigh_p_value']:.3f})")

    # Daily Patterns
    daily = analysis['temporal_patterns']['daily_pattern']
    summary.append("\nDaily Patterns:")
    summary.append(f"  Average daily activities: {daily['mean_daily_activities']:.1f}")
    summary.append(f"  Range: {daily['min_daily_activities']:.0f} to {daily['max_daily_activities']:.0f} activities")
    if not np.isnan(daily['std_daily_activities']):
        summary.append(f"  Day-to-day variability: {daily['std_daily_activities']:.2f} (variation)")

    # Day of Week Patterns
    dow_stats = analysis['day_of_week_patterns']
    summary.append("\nDay of Week Patterns:")
    for day, count in dow_stats.items():
        summary.append(f"  {day}: {count:.0f} activities")

    # Hourly Patterns
    hourly = analysis['hourly_patterns']
    summary.append("\nHourly Patterns:")
    summary.append(f"  Peak activity hours: {', '.join(f'{h:02d}:00' for h in hourly['peak_hours'])}")
    summary.append(f"  Quietest hours: {', '.join(f'{h:02d}:00' for h in hourly['quiet_hours'])}")

    # Gap Analysis
    gaps = analysis['gap_analysis']
    summary.append("\nActivity Gaps:")
    summary.append(f"  Average gap: {gaps['mean_gap']:.1f} hours")
    summary.append(f"  Median gap: {gaps['median_gap']:.1f} hours")
    summary.append(f"  Range: {gaps['min_gap']:.1f} to {gaps['max_gap']:.1f} hours")
    summary.append(f"  Significant gap threshold: {gaps['gap_threshold']:.1f} hours")
    summary.append(f"  Number of significant gaps: {gaps['significant_gaps_count']}")

    if gaps['significant_gap_hours']:
        hour_counts = pd.Series(gaps['significant_gap_hours']).value_counts()
        top_hours = hour_counts.nlargest(3)
        summary.append("  Most common hours for significant gaps:")
        for hour, count in top_hours.items():
            summary.append(f"    {hour:02d}:00 ({count} occurrences)")

    return "\n".join(summary)


if __name__ == "__main__":
    try:
        temporal_analysis = analyze_temporal_patterns('../data/categorized_output.csv')
        print(print_temporal_analysis(temporal_analysis))
    except Exception as e:
        print(f"Error: {e}")