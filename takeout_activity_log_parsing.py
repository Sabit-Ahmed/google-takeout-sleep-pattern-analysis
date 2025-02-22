import pandas as pd
from datetime import datetime
import pytz

# Define the input data
df = pd.read_csv('data/Activities - A list of Google services accessed by.csv')

# Convert timestamps from UTC to EST
utc_tz = pytz.utc
est_tz = pytz.timezone("US/Eastern")

def convert_to_est(timestamp):
    dt_utc = datetime.strptime(timestamp.replace(" UTC", ""), "%Y-%m-%d %H:%M:%S")
    dt_utc = utc_tz.localize(dt_utc)
    dt_est = dt_utc.astimezone(est_tz)
    return dt_est.strftime("%Y-%m-%d %H:%M:%S EST")

df["Activity Timestamp"] = df["Activity Timestamp"].apply(convert_to_est)

# Save to CSV
csv_filename = "data/Activities_Converted_EST.csv"
df.to_csv(csv_filename, index=False)

print(f"File saved as {csv_filename}")
