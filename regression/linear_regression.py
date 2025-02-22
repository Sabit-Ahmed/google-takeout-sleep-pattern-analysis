import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
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


def prepare_data(sleep_file, activity_file, min_topic_days=3):
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

    # Create features
    daily_data = []

    for date in sleep_df['date']:
        day_activities = activity_df[activity_df['date'] == date]
        if len(day_activities) > 0:
            # Topic proportions
            topic_counts = day_activities['topic'].value_counts()
            total_activities = len(day_activities)
            topic_props = topic_counts / total_activities

            # Time-based features
            evening_acts = len(day_activities[day_activities['hour'] >= 18])
            night_acts = len(day_activities[day_activities['hour'] >= 22])
            morning_acts = len(day_activities[day_activities['hour'] < 6])
            mid_day_acts = len(day_activities[(day_activities['hour'] > 12) & (day_activities['hour'] < 15)])

            # Last activity hour
            last_hour = day_activities['hour'].max()
            first_hour = day_activities['hour'].min()

            features = {
                'date': date,
                'total_activities': total_activities,
                'evening_activities': evening_acts,
                'night_activities': night_acts,
                'mid_day_activities': mid_day_acts,
                'morning_activities': morning_acts,
                'last_activity_hour': last_hour,
                'first_activity_hour': first_hour,
                'activity_span': last_hour - first_hour
            }

            # Add topic proportions with 0s for missing topics
            for topic in activity_df['topic'].unique():
                features[f'topic_{topic}'] = topic_props.get(topic, 0)

            daily_data.append(features)

    # Create DataFrame
    X_df = pd.DataFrame(daily_data)

    # Get topic columns that appear in at least min_topic_days days
    topic_cols = [col for col in X_df.columns if col.startswith('topic_')]
    frequent_topics = []
    for col in topic_cols:
        if sum(X_df[col] > 0) >= min_topic_days:
            frequent_topics.append(col)

    # Select features
    feature_cols = ['total_activities', 'evening_activities', 'night_activities', 'mid_day_activities',
                    'morning_activities', 'last_activity_hour', 'first_activity_hour',
                    'activity_span'] + frequent_topics

    X = X_df[feature_cols]
    y = sleep_df['sleep_duration']

    # Handle any remaining NaN values
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    return X, y, feature_cols

def plot_regression_relationships_all(X_train, y_train, X_test, y_test, lr_model, feature_cols, save_file_name):
    """
    Plot regression lines for both actual relationship and model predictions
    """
    # Get the most important feature
    coefficients = pd.DataFrame({
        'feature': feature_cols,
        'coef': abs(lr_model.coef_)
    })
    # most_important_feature = coefficients.nlargest(1, 'coef')['feature'].iloc[0]
    n_features = len(feature_cols)
    n_cols = 3  # Number of columns in the subplot grid
    n_rows = (n_features + n_cols - 1) // n_cols  # Ceiling division to get number of rows needed

    plt.figure(figsize=(15, 5 * n_rows))

    for idx, feature in enumerate(feature_cols):
        plt.subplot(n_rows, n_cols, idx + 1)

        # Training data
        x_train = X_train[feature]
        y_pred_train = lr_model.predict(X_train)

        # Plot data points
        plt.scatter(x_train, y_train, color='blue', alpha=0.5)

        # Line for actual data
        z_actual = np.polyfit(x_train, y_train, 1)
        p_actual = np.poly1d(z_actual)
        plt.plot(x_train, p_actual(x_train), 'b-', label='Actual', linewidth=2)

        # Line from model predictions
        z_pred = np.polyfit(x_train, y_pred_train, 1)
        p_pred = np.poly1d(z_pred)
        plt.plot(x_train, p_pred(x_train), 'r-', label='Predicted', linewidth=2)

        plt.xlabel(feature)
        plt.ylabel('Sleep Duration (hours)')
        plt.title(f'Feature: {feature}')
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_file_name)
    print(f"\nRegression lines comparison saved as {save_file_name}")

def plot_regression_relationships(X_train, y_train, X_test, y_test, lr_model, feature_cols, save_file_name):
    """
    Plot regression lines for both actual relationship and model predictions
    """
    # Get the most important feature
    coefficients = pd.DataFrame({
        'feature': feature_cols,
        'coef': abs(lr_model.coef_)
    })
    most_important_feature = coefficients.nlargest(1, 'coef')['feature'].iloc[0]

    plt.figure(figsize=(12, 5))

    # Training plot
    plt.subplot(1, 2, 1)
    x_train = X_train[most_important_feature]
    y_pred_train = lr_model.predict(X_train)

    # Plot data points
    plt.scatter(x_train, y_train, color='blue', alpha=0.5)

    # Line for actual data
    z_actual = np.polyfit(x_train, y_train, 1)
    p_actual = np.poly1d(z_actual)
    plt.plot(x_train, p_actual(x_train), 'b-', label='Actual relationship', linewidth=2)

    # Line from model predictions
    z_pred = np.polyfit(x_train, y_pred_train, 1)
    p_pred = np.poly1d(z_pred)
    plt.plot(x_train, p_pred(x_train), 'r-', label='Model predictions', linewidth=2)

    plt.xlabel(most_important_feature)
    plt.ylabel('Sleep Duration (hours)')
    plt.title('Training Data')
    plt.legend()

    # Test plot
    plt.subplot(1, 2, 2)
    x_test = X_test[most_important_feature]
    y_pred_test = lr_model.predict(X_test)

    # Plot data points
    plt.scatter(x_test, y_test, color='blue', alpha=0.5)

    # Line for actual data
    z_actual = np.polyfit(x_test, y_test, 1)
    p_actual = np.poly1d(z_actual)
    plt.plot(x_test, p_actual(x_test), 'b-', label='Actual relationship', linewidth=2)

    # Line from model predictions
    z_pred = np.polyfit(x_test, y_pred_test, 1)
    p_pred = np.poly1d(z_pred)
    plt.plot(x_test, p_pred(x_test), 'r-', label='Model predictions', linewidth=2)

    plt.xlabel(most_important_feature)
    plt.ylabel('Sleep Duration (hours)')
    plt.title('Test Data')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_file_name)
    print(f"\nRegression lines comparison saved as {save_file_name}")

def train_and_evaluate_model(X, y, feature_cols):
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train models
    # 1. Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # 2. Lasso with CV for feature selection
    lasso_model = LassoCV(cv=5, random_state=42)
    lasso_model.fit(X_train, y_train)

    plot_regression_relationships(X_train, y_train, X_test, y_test, lr_model, feature_cols,'lr_relationship_most_imp_feature.png')
    plot_regression_relationships(X_train, y_train, X_test, y_test, lasso_model, feature_cols,'lasso_relationship_most_imp_feature.png')

    plot_regression_relationships_all(X_train, y_train, X_test, y_test, lr_model, feature_cols,
                                  'lr_relationship_all_feature.png')
    plot_regression_relationships_all(X_train, y_train, X_test, y_test, lasso_model, feature_cols,
                                  'lasso_relationship_all_feature.png')

    # Evaluate models
    print("\nModel Evaluation:")
    print("=" * 50)

    # Linear Regression Results
    lr_pred = lr_model.predict(X_test)
    lr_r2 = r2_score(y_test, lr_pred)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))

    print("\nLinear Regression:")
    print(f"R² Score: {lr_r2:.3f}")
    print(f"RMSE: {lr_rmse:.3f} hours")

    # Cross-validation scores
    cv_scores = cross_val_score(lr_model, X_scaled, y, cv=5, scoring='r2')
    print(f"5-fold CV R² Scores: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Feature Importance
    print("\nFeature Importance (Linear Regression):")
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': lr_model.coef_
    })
    importance_df['abs_coef'] = abs(importance_df['coefficient'])
    importance_df = importance_df.sort_values('abs_coef', ascending=False)

    print("\nTop 10 Most Important Features:")
    for _, row in importance_df.head(10).iterrows():
        print(f"{row['feature']}: {row['coefficient']:.3f}")

    # Lasso Results
    lasso_pred = lasso_model.predict(X_test)
    lasso_r2 = r2_score(y_test, lasso_pred)
    lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_pred))

    print("\nLasso Regression:")
    print(f"R² Score: {lasso_r2:.3f}")
    print(f"RMSE: {lasso_rmse:.3f} hours")

    # Selected features by Lasso
    lasso_importance = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': lasso_model.coef_
    })
    lasso_importance = lasso_importance[lasso_importance['coefficient'] != 0]
    lasso_importance = lasso_importance.sort_values('coefficient', key=abs, ascending=False)

    print("\nFeatures Selected by Lasso:")
    for _, row in lasso_importance.iterrows():
        print(f"{row['feature']}: {row['coefficient']:.3f}")

    # Visualize actual vs predicted
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(y_test, lr_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Sleep Duration')
    plt.ylabel('Predicted Sleep Duration')
    plt.title('Linear Regression\nActual vs Predicted')

    plt.subplot(1, 2, 2)
    plt.scatter(y_test, lasso_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Sleep Duration')
    plt.ylabel('Predicted Sleep Duration')
    plt.title('Lasso Regression\nActual vs Predicted')

    plt.tight_layout()
    plt.savefig('sleep_predictions.png')
    print("\nPrediction plots saved as 'sleep_predictions.png'")

    # Feature importance plot
    plt.figure(figsize=(12, 6))
    importance_plot_df = importance_df.head(10)
    plt.barh(range(len(importance_plot_df)), importance_plot_df['coefficient'])
    plt.yticks(range(len(importance_plot_df)), importance_plot_df['feature'])
    plt.xlabel('Coefficient Value')
    plt.title('Top 10 Most Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Feature importance plot saved as 'feature_importance.png'")

    return lr_model, lasso_model, importance_df, lasso_importance


def main():
    # Prepare data
    X, y, feature_cols = prepare_data('../data/sleep_ground_truths.csv', '../data/categorized_output_compressed.csv')

    print("Dataset Info:")
    print(f"Number of samples: {len(X)}")
    print(f"Number of features: {len(feature_cols)}")
    print("\nFeatures used:")
    for feat in feature_cols:
        print(f"- {feat}")

    # Train and evaluate models
    lr_model, lasso_model, importance_df, lasso_importance = train_and_evaluate_model(X, y, feature_cols)


if __name__ == "__main__":
    main()