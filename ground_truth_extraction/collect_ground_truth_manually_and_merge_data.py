import pandas as pd

df = pd.read_csv('../data/categorized_output.csv')

df['timestamp'] = pd.to_datetime(df['timestamp']
                                 .str.replace('EST', '')  # Remove EST timezone indicator
                                 .str.strip(),  # Remove any extra whitespace
                                 format='mixed',  # Allow mixed formats
                                 utc=False)

df['date'] = df['timestamp'].dt.date
df['hour'] = df['timestamp'].dt.hour

# Group by date
dates = sorted(df['date'].unique())
date_df = pd.DataFrame(dates, columns=['date'])
date_df.to_csv('data/sleep_ground_truths.csv', index=None)