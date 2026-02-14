# ============================================================
# OPTIONAL: Time series + visualization (Broadway traffic)
# ============================================================
#
# In this optional section, we treat time as an ordering variable.
# We are NOT predicting future values.
# We are only visualizing how traffic volume changes over time.

import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Load the dataset
# ------------------------------------------------------------

# this data is from:
# https://data.cityofnewyork.us/Transportation/Automated-Traffic-Volume-Counts/7ym2-wayt/about_data
# To keep the size manageable, I only downloaded Manhattan traffic.

traffic = pd.read_csv("data/manhattan_traffic_volume_counts.csv")

# ------------------------------------------------------------
# Build a timestamp column from date/time parts
# ------------------------------------------------------------

traffic["timestamp"] = pd.to_datetime(
    dict(
        year=traffic["Yr"],
        month=traffic["M"],
        day=traffic["D"],
        hour=traffic["HH"],
        minute=traffic["MM"],
    ),
    errors="coerce",
)

# Keep only rows with valid timestamps and volume counts
traffic = traffic.dropna(subset=["timestamp", "Vol"])

# Sort by time (important for time series work)
traffic = traffic.sort_values("timestamp")

# ------------------------------------------------------------
# Focus on ONE real-world location: Broadway
# ------------------------------------------------------------

# Filter to rows where the street name contains "BROADWAY"
# (Using .str.contains makes this robust to capitalization or spacing.)
broadway = traffic[traffic["street"].str.contains("BROADWAY", case=False, na=False)].copy()

print("Number of Broadway observations:", len(broadway))
print(broadway.head())

# ------------------------------------------------------------
# 1) Daily traffic volume over time
# ------------------------------------------------------------

# Aggregate to daily totals (sum of volumes within each day)
daily = (
    broadway
    .set_index("timestamp")["Vol"]
    .resample("D")
    .sum()
)

"""
Alt text — Line plot (Daily traffic volume on Broadway)
This line plot shows daily total traffic volume recorded on Broadway in Manhattan.
The x-axis represents calendar date.
The y-axis represents the total traffic volume for that day, calculated by summing
all recorded counts throughout the day.
The line rises and falls over time, showing day-to-day variation in traffic levels.
This chart is useful for seeing overall trends and fluctuations across days.
"""

plt.figure()
plt.plot(daily.index, daily.values)
plt.title("Daily Traffic Volume on Broadway (Total per Day)")
plt.xlabel("Date")
plt.ylabel("Total Traffic Volume")
plt.show()

# ------------------------------------------------------------
# 2) Smoothing: 7-day moving average
# ------------------------------------------------------------

daily_ma7 = daily.rolling(window=7).mean()

"""
Alt text — Line plot (Daily traffic volume with 7-day moving average)
This chart shows daily traffic volume on Broadway along with a 7-day moving average.
The x-axis represents calendar date.
The y-axis represents total daily traffic volume.
The moving-average line smooths short-term day-to-day variation,
making longer-term patterns easier to see.
This chart is useful for comparing raw daily fluctuations with a smoothed trend.
"""

plt.figure()
plt.plot(daily.index, daily.values, label="Daily total")
plt.plot(daily_ma7.index, daily_ma7.values, label="7-day moving average")
plt.title("Daily Traffic Volume on Broadway with 7-Day Moving Average")
plt.xlabel("Date")
plt.ylabel("Total Traffic Volume")
plt.legend()
plt.show()

# ------------------------------------------------------------
# 3) Average traffic volume by weekday
# ------------------------------------------------------------

weekday_df = daily.to_frame("daily_volume")
weekday_df["weekday"] = weekday_df.index.day_name()

weekday_order = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

weekday_means = (
    weekday_df
    .groupby("weekday")["daily_volume"]
    .mean()
    .reindex(weekday_order)
)

"""
Alt text — Bar plot (Average daily traffic by weekday on Broadway)
This bar plot shows the average daily traffic volume on Broadway for each day of the week.
The x-axis lists weekdays from Monday through Sunday.
The y-axis shows the mean daily traffic volume.
Differences in bar heights indicate that traffic levels vary by weekday,
often with higher volumes on weekdays and different patterns on weekends.
This chart is useful for comparing typical traffic levels across days of the week.
"""

plt.figure()
plt.bar(weekday_means.index, weekday_means.values)
plt.title("Average Daily Traffic Volume on Broadway by Weekday")
plt.xlabel("Weekday")
plt.ylabel("Average Daily Traffic Volume")
plt.xticks(rotation=30, ha="right")
plt.show()

# ------------------------------------------------------------
# 4) Average traffic volume by hour of day
# ------------------------------------------------------------

broadway["hour"] = broadway["timestamp"].dt.hour
hourly_means = broadway.groupby("hour")["Vol"].mean()

"""
Alt text — Line plot (Average traffic volume by hour of day on Broadway)
This line plot shows the average traffic volume on Broadway by hour of day.
The x-axis represents hour of day from 0 to 23.
The y-axis represents the mean traffic volume recorded at that hour.
Traffic volume typically increases during morning hours, peaks around commute times,
and decreases overnight, reflecting a daily cycle.
This chart is useful for understanding typical within-day traffic patterns.
"""

plt.figure()
plt.plot(hourly_means.index, hourly_means.values, marker="o")
plt.title("Average Traffic Volume on Broadway by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Average Traffic Volume")
plt.xticks(range(0, 24))
plt.show()