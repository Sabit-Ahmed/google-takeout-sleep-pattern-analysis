import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.extras import unique

df = pd.read_csv('../data/Activities - A list of Google services accessed by.csv')


df["Activity Timestamp"] = pd.to_datetime(df["Activity Timestamp"], errors='coerce')
df = df[df["User Agent String"] != "App : APPLE_NATIVE_APP. Os : MAC_OS."]
df = df[df["User Agent String"] != "'App : OTHER_APP. Os : MAC_OS. Os Version : 10.15. Device Type : PC.'"]
df = df[~(df["User Agent String"].str.contains("safari", case=False) &
                            df["User Agent String"].str.contains("PC", case=False))]

# df = df[df["User Agent String"] == "App : GMAIL_APP. App Version : 6.0.250119. Os : IOS_OS. Os Version : 18.3. Device Type : MOBILE."]

# Drop any rows where the conversion failed (NaT values)
df = df.dropna(subset=["Activity Timestamp"])

print(df["User Agent String"].unique())

# Extract the date and hour from the timestamp
df["Date"] = df["Activity Timestamp"].dt.date
df["Hour"] = df["Activity Timestamp"].dt.hour

# Compute sleep duration for each day
sleep_durations = []

# Iterate through each unique date
for date in df["Date"].unique():
    df_day = df[df["Date"] == date]

    # Create a 24-hour activity map
    activity_map = np.zeros(24)  # Default: no activity (assume sleep)

    # Mark active hours
    for hour in df_day["Hour"].unique():
        activity_map[hour] = 1  # Mark hours where activity was recorded

    # Compute total sleep duration (hours with no activity)
    sleep_hours = 24 - activity_map.sum()
    sleep_durations.append((date, sleep_hours))

# Convert to DataFrame
df_sleep = pd.DataFrame(sleep_durations, columns=["Date", "Sleep Duration (hours)"])

# Sort by date
df_sleep = df_sleep.sort_values("Date")

# Pick a sample week for visualization
sample_week = df_sleep.iloc[:7]  # Selecting the first 7 days for weekly analysis

# Plot sleep duration over the selected week
plt.figure(figsize=(10, 5))
plt.plot(sample_week["Date"], sample_week["Sleep Duration (hours)"], marker="o", linestyle="-", color="blue")

plt.xlabel("Date")
plt.ylabel("Sleep Duration (hours)")
plt.title("Weekly Sleep Duration Trend")
plt.xticks(rotation=45)
plt.grid(True)

plt.show()
