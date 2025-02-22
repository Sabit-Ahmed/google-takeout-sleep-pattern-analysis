import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime


def parse_sleep_time(time_range):
    """Parse sleep time range like '1.30 AM - 9.00 AM' and return duration in hours"""
    start, end = time_range.split(' - ')

    def convert_to_24hr(time_str):
        time_parts = time_str.split(' ')
        hour_min = time_parts[0].split('.')
        period = time_parts[1]

        hour = int(hour_min[0])
        minute = int(hour_min[1]) if len(hour_min) > 1 else 0

        if period == 'PM' and hour != 12:
            hour += 12
        elif period == 'AM' and hour == 12:
            hour = 0

        return hour + minute / 60

    start_time = convert_to_24hr(start)
    end_time = convert_to_24hr(end)

    if end_time > start_time:
        duration = end_time - start_time
    else:
        duration = (24 - start_time) + end_time

    return duration


def prepare_data(sleep_file, activity_file):
    # Read sleep data
    sleep_df = pd.read_csv(sleep_file)
    sleep_df['sleep_duration'] = sleep_df['sleep_time'].apply(parse_sleep_time)

    # Read activity data
    activity_df = pd.read_csv(activity_file)
    activity_df['timestamp'] = pd.to_datetime(activity_df['timestamp']
                                              .str.replace('EST', '')
                                              .str.strip(),
                                              format='mixed',
                                              utc=False)
    activity_df['date'] = activity_df['timestamp'].dt.strftime('%-m/%-d/25')
    activity_df['hour'] = activity_df['timestamp'].dt.hour
    activity_df['topic'] = activity_df['topic_compressed']

    # Create daily features
    daily_data = []
    for date in sleep_df['date']:
        day_activities = activity_df[activity_df['date'] == date]
        if len(day_activities) > 0:
            # Basic activity features
            total_activities = len(day_activities)
            evening_acts = len(day_activities[day_activities['hour'] >= 18])
            night_acts = len(day_activities[day_activities['hour'] >= 22])
            morning_acts = len(day_activities[day_activities['hour'] < 6])
            afternoon_acts = len(day_activities[(day_activities['hour'] >= 12) & (day_activities['hour'] < 18)])

            # Activity timing
            last_hour = day_activities['hour'].max()
            first_hour = day_activities['hour'].min()
            activity_span = last_hour - first_hour

            # Activity patterns
            hourly_counts = day_activities['hour'].value_counts()
            max_hourly = hourly_counts.max()
            peak_hour = hourly_counts.idxmax()

            # Topic features
            topic_counts = day_activities['topic'].value_counts()
            topic_props = topic_counts / total_activities

            features = {
                'date': date,
                'total_activities': total_activities,
                'evening_activities': evening_acts,
                'night_activities': night_acts,
                'morning_activities': morning_acts,
                'afternoon_activities': afternoon_acts,
                'last_activity_hour': last_hour,
                'first_activity_hour': first_hour,
                'activity_span': activity_span,
                'max_hourly_activities': max_hourly,
                'peak_activity_hour': peak_hour
            }

            # Add topic proportions
            for topic in activity_df['topic'].unique():
                features[f'topic_{topic}'] = topic_props.get(topic, 0)

            daily_data.append(features)

    X_df = pd.DataFrame(daily_data)
    y = sleep_df['sleep_duration']

    # Select features that appear in at least 3 days
    topic_cols = [col for col in X_df.columns if col.startswith('topic_')]
    frequent_topics = []
    for col in topic_cols:
        if sum(X_df[col] > 0) >= 3:
            frequent_topics.append(col)

    # Final feature selection
    feature_cols = ['total_activities', 'evening_activities', 'night_activities',
                    'morning_activities', 'afternoon_activities', 'last_activity_hour',
                    'first_activity_hour', 'activity_span', 'max_hourly_activities',
                    'peak_activity_hour'] + frequent_topics

    X = X_df[feature_cols].fillna(0)

    return X, y, feature_cols


def train_tree_model(X, y, feature_cols):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model with different max_depths
    results = []
    for depth in range(2, 7):
        tree = DecisionTreeRegressor(max_depth=depth, random_state=42)
        tree.fit(X_train, y_train)

        # Get predictions
        train_pred = tree.predict(X_train)
        test_pred = tree.predict(X_test)

        # Calculate metrics
        results.append({
            'depth': depth,
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'cv_scores': cross_val_score(tree, X, y, cv=5, scoring='r2').mean()
        })

    # Find best depth
    best_depth = max(results, key=lambda x: x['cv_scores'])['depth']

    # Train final model
    final_tree = DecisionTreeRegressor(max_depth=best_depth, random_state=42)
    final_tree.fit(X_train, y_train)
    final_pred = final_tree.predict(X_test)

    # Get feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': final_tree.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nDecision Tree Results:")
    print(f"Best max_depth: {best_depth}")
    print(f"R² Score: {r2_score(y_test, final_pred):.3f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, final_pred)):.3f} hours")
    print(f"5-fold CV R² Score: {cross_val_score(final_tree, X, y, cv=5, scoring='r2').mean():.3f}")

    print("\nFeature Importance:")
    print(importance.head(10).to_string(index=False))

    # Create visualizations
    plt.figure(figsize=(15, 10))

    # Plot actual vs predicted
    plt.subplot(2, 2, 1)
    plt.scatter(y_test, final_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Actual Sleep Duration')
    plt.ylabel('Predicted Sleep Duration')
    plt.title('Actual vs Predicted Sleep Duration')

    # Plot feature importance
    plt.subplot(2, 2, 2)
    importance_plot = importance.head(10)
    plt.barh(range(len(importance_plot)), importance_plot['importance'])
    plt.yticks(range(len(importance_plot)), importance_plot['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Most Important Features')

    # Plot tree structure
    plt.subplot(2, 1, 2)
    plot_tree(final_tree, feature_names=feature_cols, filled=True, rounded=True, fontsize=8)
    plt.title('Decision Tree Structure')

    plt.tight_layout()
    plt.savefig('sleep_tree_analysis.png', dpi=300, bbox_inches='tight')
    print("\nAnalysis plots saved as 'sleep_tree_analysis.png'")

    return final_tree, importance


def main():
    # Prepare data
    X, y, feature_cols = prepare_data('../data/sleep_ground_truths.csv', '../data/categorized_output_compressed.csv')

    print("Dataset Info:")
    print(f"Number of samples: {len(X)}")
    print(f"Number of features: {len(feature_cols)}")

    # Train and evaluate model
    tree_model, importance = train_tree_model(X, y, feature_cols)


if __name__ == "__main__":
    main()