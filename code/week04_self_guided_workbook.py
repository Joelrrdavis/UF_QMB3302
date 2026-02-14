"""
week04_self_guided_workbook.py

Week 4 — Self-Guided Practice Workbook (NO solutions)

What this is:
- A self-guided practice file for simulation + visualization using matplotlib.
- You will modify small pieces of code (the "TRY THIS" prompts),
  re-run sections, and compare what you observe to the "CHECK YOURSELF" notes.

How you know you’re “right”:
- You are NOT trying to match an exact number.
- You are looking for patterns described in the CHECK YOURSELF notes.
- If what you see matches the described patterns, you’re on track.

What you need:
- Python, pandas, matplotlib (and optionally seaborn)
- data/titanic.csv (required section uses this)
- data/manhattan_traffic_volume_counts.csv (optional time-series extension)

Important:
- We are NOT forecasting or predicting in this workbook.
- We are practicing visualization and interpretation.

------------------------------------------------------------
Running this file:
- In VS Code terminal:
    python week04_self_guided_workbook.py
------------------------------------------------------------
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# OPTIONAL: seaborn (only used in one optional section)
try:
    import seaborn as sns  # type: ignore
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False


# ============================================================
# 0) Setup: reproducibility (optional)
# ============================================================
#
# Randomness is useful for simulation, but it can be confusing when results change every run.
# If you set a seed, you’ll get the same “random” numbers each time.
#
# TRY THIS:
# - Comment out the seed line and rerun. What changes? What stays the same?
#
random.seed(42)


# ============================================================
# 1) Distributions with a histogram (simulated data)
# ============================================================

# We simulate values between 0 and 1 using random.random().
n = 200
values = [random.random() for _ in range(n)]

"""
Alt text — Histogram (Simulated Values)
This histogram shows the distribution of simulated values between 0 and 1.
The x-axis represents the value, from 0 to 1.
The y-axis represents how many observations fall into each range (frequency).
Because the data was generated uniformly at random, the bars should be
roughly similar in height, with small variations due to randomness.
This chart is useful for seeing the overall shape of a distribution.
"""

plt.figure()
plt.hist(values, bins=20, edgecolor="black")
plt.title("Histogram: Distribution of Simulated Values")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

# TRY THIS:
# 1) Change n to 50, then 500. Re-run the histogram.
# 2) Change bins to 10, then 40. Re-run the histogram.
#
# CHECK YOURSELF:
# - With smaller n, bars often look more uneven (more randomness).
# - With larger n, the histogram looks smoother (randomness averages out).
# - Changing bins changes the resolution (bar width), not the underlying data.


# ============================================================
# 2) Sequence view with a line plot (same values, different question)
# ============================================================
#
# A line plot treats data as ordered. Here, the “order” is just generation order.
# In real time series, the order is time.
#
"""
Alt text — Line plot (Simulated values in generation order)
This line plot shows the simulated values in the order they were generated.
The x-axis represents the observation index (the order of each value).
The y-axis represents the value between 0 and 1.
There is no meaningful trend expected; fluctuations reflect randomness.
This chart is useful for seeing variability over a sequence.
"""

plt.figure()
plt.plot(values)
plt.title("Line Plot: Simulated Values in Generation Order")
plt.xlabel("Observation Index")
plt.ylabel("Value")
plt.show()

# TRY THIS:
# 1) Increase n to 1000 and rerun. What changes about the “look” of the line?
# 2) Comment out random.seed(42) at the top and rerun. What changes?


# ============================================================
# 3) Relationships with a scatter plot (simulated related variables)
# ============================================================
#
# We create a second variable by adding a small amount of noise to the first.
# That creates a relationship that should be visible in a scatter plot.
#
noise = [random.gauss(0, 0.05) for _ in range(n)]
related_values = [v + e for v, e in zip(values, noise)]

"""
Alt text — Scatter plot (Related simulated variables)
This scatter plot shows the relationship between two simulated numeric variables.
The x-axis represents original values generated between 0 and 1.
The y-axis represents those same values after adding a small amount of random noise.
Each point corresponds to one observation.
The points form a diagonal cloud from lower left to upper right, indicating a positive relationship.
The vertical spread around the diagonal reflects the added noise.
This chart is useful for seeing how two related numeric variables vary together.
"""

plt.figure()
plt.scatter(values, related_values, alpha=0.6, s=20)
plt.title("Scatter Plot: Relationship Between Simulated Variables")
plt.xlabel("Original Value")
plt.ylabel("Value with Noise")
plt.show()

# TRY THIS:
# 1) Increase the noise level to 0.2 and rerun (more noise).
# 2) Decrease the noise level to 0.01 and rerun (less noise).
#
# CHECK YOURSELF:
# - More noise makes the diagonal pattern harder to see (more spread).
# - Less noise makes points cluster tightly along a diagonal.
# - If you see no points, check that related_values exists and has the same length as values.


# ============================================================
# 4) Categorical data with a bar plot (simulated categories)
# ============================================================
#
# Bar plots are typically used for counts (how many in each category).
#
categories = ["A", "B", "C"]
assigned_categories = [random.choice(categories) for _ in range(n)]
counts = {c: assigned_categories.count(c) for c in categories}

"""
Alt text — Bar plot (Simulated category counts)
This bar plot shows the number of observations assigned to each category.
The x-axis lists the categories A, B, and C.
The y-axis represents the count in each category.
Each bar corresponds to one category, and its height reflects how often that category appears.
Because categories were assigned randomly, the bars are expected to be similar in height
but small differences appear due to randomness.
This chart is useful for comparing counts across categories.
"""

plt.figure()
plt.bar(counts.keys(), counts.values())
plt.title("Bar Plot: Counts by Simulated Category")
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()

# TRY THIS:
# 1) Increase n to 1000 and rerun. What happens to how similar the bars look?
# 2) Create an intentionally biased assignment (more 'A' than 'B' or 'C').
#
# Example bias idea (replace assigned_categories line above):
# assigned_categories = [random.choices(categories, weights=[0.7, 0.2, 0.1])[0] for _ in range(n)]
#
# CHECK YOURSELF:
# - Larger n makes random proportions stabilize (bars become more similar when truly random).
# - With bias, one bar becomes noticeably taller.


# ============================================================
# 5) REQUIRED PRACTICE: Real-world visualization (Titanic dataset)
# ============================================================
#
# This section applies the same visualization tools to real, messy data.
# Real data often includes missing values and multiple variable types.
#
# Expected file location: data/titanic.csv
# If your Titanic file is elsewhere, update TITANIC_PATH.
#
TITANIC_PATH = Path("data") / "titanic.csv"
if not TITANIC_PATH.exists():
    # fallback: same folder as this script (useful for quick experiments)
    script_dir = Path(__file__).resolve().parent
    fallback = script_dir / "titanic.csv"
    if fallback.exists():
        TITANIC_PATH = fallback

if not TITANIC_PATH.exists():
    print("\n[NOTE] Titanic dataset not found.")
    print("Expected: data/titanic.csv (relative to where you run the script).")
    print("Add the file and rerun to complete the required practice section.")
else:
    titanic = pd.read_csv(TITANIC_PATH)

    # --- Histogram: Age distribution ---
    """
    Alt text — Histogram (Titanic passenger ages)
    This histogram shows the distribution of passenger ages on the Titanic.
    The x-axis represents age in years.
    The y-axis represents the number of passengers within each age range.
    Most passengers are adults, with many ages clustered between roughly 20 and 40.
    There are fewer children and fewer elderly passengers.
    This chart is useful for understanding the age composition of the passengers.
    """
    plt.figure()
    plt.hist(titanic["Age"].dropna(), bins=20, edgecolor="black")
    plt.title("Titanic: Distribution of Passenger Ages")
    plt.xlabel("Age")
    plt.ylabel("Number of Passengers")
    plt.show()

    # TRY THIS:
    # 1) Change bins to 10 and 40. What changes? What stays the same?
    # 2) Replace Age with Fare. What looks different about the distribution?
    #
    # CHECK YOURSELF:
    # - Fare is usually more skewed (many low fares, a few very high fares).

    # --- Bar plot: Survival counts ---
    """
    Alt text — Bar plot (Titanic survival counts)
    This bar plot shows how many passengers survived versus did not survive.
    The x-axis shows survival status: 0 = did not survive, 1 = survived.
    The y-axis shows the number of passengers in each group.
    The non-survivor bar is typically taller, indicating more passengers did not survive.
    This chart is useful for comparing counts across categories.
    """
    survival_counts = titanic["Survived"].value_counts().sort_index()
    plt.figure()
    plt.bar(survival_counts.index, survival_counts.values)
    plt.title("Titanic: Survival Counts")
    plt.xlabel("Survived (0 = No, 1 = Yes)")
    plt.ylabel("Number of Passengers")
    plt.xticks([0, 1])
    plt.show()

    # TRY THIS:
    # 1) Make a bar plot for Sex instead of Survived.
    # 2) Make a bar plot for Pclass instead of Survived.

    # --- Scatter plot: Age vs Fare ---
    """
    Alt text — Scatter plot (Titanic Age vs Fare)
    This scatter plot shows the relationship between passenger age and ticket fare.
    The x-axis represents age in years.
    The y-axis represents fare paid for the ticket.
    Each point represents one passenger.
    Many passengers paid lower fares across a wide range of ages.
    A small number of passengers paid very high fares, appearing as points high on the y-axis.
    This chart is useful for seeing relationships and outliers in numeric data.
    """
    plt.figure()
    plt.scatter(titanic["Age"], titanic["Fare"], alpha=0.5, s=20)
    plt.title("Titanic: Age vs Fare")
    plt.xlabel("Age")
    plt.ylabel("Fare")
    plt.show()

    # TRY THIS:
    # 1) Add .dropna() to remove missing Age values before plotting.
    # 2) Reduce alpha to 0.2. Does it help you see density?
    #
    # Example:
    # clean = titanic[["Age", "Fare"]].dropna()
    # plt.scatter(clean["Age"], clean["Fare"], alpha=0.2)

    # --- Box plot: Fare by passenger class ---
    """
    Alt text — Box plot (Titanic Fare by class)
    This box plot compares the distribution of fares across passenger classes.
    The x-axis lists passenger classes (1st, 2nd, 3rd).
    The y-axis represents fare paid.
    First-class fares are generally higher than second- and third-class fares.
    The spread is often wider in first class, with high-fare outliers.
    This chart is useful for comparing distributions across groups.
    """
    fares_by_class = [
        titanic[titanic["Pclass"] == 1]["Fare"].dropna(),
        titanic[titanic["Pclass"] == 2]["Fare"].dropna(),
        titanic[titanic["Pclass"] == 3]["Fare"].dropna(),
    ]
    plt.figure()
    plt.boxplot(fares_by_class, labels=["1st", "2nd", "3rd"])
    plt.title("Titanic: Fare by Passenger Class")
    plt.xlabel("Passenger Class")
    plt.ylabel("Fare")
    plt.show()

    # TRY THIS:
    # 1) Make a box plot of Age by Pclass instead of Fare by Pclass.
    # 2) Interpret: which class tends to have older passengers?


# ============================================================
# 6) OPTIONAL EXTENSION: Time series visualization (Broadway traffic)
# ============================================================
#
# This section introduces a new idea: time as an ordering variable.
# It is OPTIONAL and not required for Week 4.
#
# Expected file location: data/manhattan_traffic_volume_counts.csv
#
TRAFFIC_PATH = Path("data") / "manhattan_traffic_volume_counts.csv"
if not TRAFFIC_PATH.exists():
    script_dir = Path(__file__).resolve().parent
    fallback = script_dir / "manhattan_traffic_volume_counts.csv"
    if fallback.exists():
        TRAFFIC_PATH = fallback

if not TRAFFIC_PATH.exists():
    print("\n[NOTE] Traffic dataset not found.")
    print("Expected: data/manhattan_traffic_volume_counts.csv")
    print("Add the file and rerun to explore the optional time-series section.")
else:
    traffic = pd.read_csv(TRAFFIC_PATH)

    # Build a single datetime column (timestamp)
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

    traffic = traffic.dropna(subset=["timestamp", "Vol"])
    traffic = traffic.sort_values("timestamp")

    # Focus on Broadway
    broadway = traffic[traffic["street"].astype(str).str.contains("BROADWAY", case=False, na=False)].copy()

    # If the filter is too broad (many segments), visuals can be noisy.
    # A good extension is to pick ONE SegmentID after you inspect options:
    #
    # TRY THIS:
    # print(broadway["SegmentID"].value_counts().head(10))
    # Then choose a specific SegmentID and filter:
    # broadway = broadway[broadway["SegmentID"] == YOUR_CHOICE].copy()

    # Daily totals
    daily = broadway.set_index("timestamp")["Vol"].resample("D").sum()

    """
    Alt text — Line plot (Broadway daily traffic volume)
    This line plot shows daily total traffic volume recorded on Broadway.
    The x-axis represents calendar date.
    The y-axis represents total traffic volume per day, computed by summing counts across the day.
    The line rises and falls over time, reflecting day-to-day variation in traffic levels.
    This chart is useful for seeing overall trends and fluctuations across days.
    """
    plt.figure()
    plt.plot(daily.index, daily.values)
    plt.title("Broadway Traffic: Daily Total Volume")
    plt.xlabel("Date")
    plt.ylabel("Total Volume (per day)")
    plt.show()

    # 7-day moving average (smoothing)
    daily_ma7 = daily.rolling(window=7).mean()

    """
    Alt text — Line plot (Broadway daily volume with 7-day moving average)
    This chart shows daily traffic volume along with a 7-day moving average.
    The moving-average line is smoother than the raw daily line, helping highlight longer-term patterns
    by reducing day-to-day variation.
    """
    plt.figure()
    plt.plot(daily.index, daily.values, label="Daily total")
    plt.plot(daily_ma7.index, daily_ma7.values, label="7-day moving average")
    plt.title("Broadway Traffic: Daily Volume with 7-Day Moving Average")
    plt.xlabel("Date")
    plt.ylabel("Total Volume (per day)")
    plt.legend()
    plt.show()

    # Weekday pattern
    weekday_df = daily.to_frame("daily_volume")
    weekday_df["weekday"] = weekday_df.index.day_name()
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_means = weekday_df.groupby("weekday")["daily_volume"].mean().reindex(weekday_order)

    """
    Alt text — Bar plot (Broadway average daily traffic by weekday)
    This bar plot shows average daily traffic volume on Broadway for each weekday.
    The x-axis lists weekdays from Monday through Sunday.
    The y-axis shows mean daily traffic volume.
    Differences in bar heights indicate that typical traffic levels vary by day of week.
    """
    plt.figure()
    plt.bar(weekday_means.index, weekday_means.values)
    plt.title("Broadway Traffic: Average Daily Volume by Weekday")
    plt.xlabel("Weekday")
    plt.ylabel("Average Daily Volume")
    plt.xticks(rotation=30, ha="right")
    plt.show()

    # Hour-of-day profile
    broadway["hour"] = broadway["timestamp"].dt.hour
    hourly_means = broadway.groupby("hour")["Vol"].mean()

    """
    Alt text — Line plot (Broadway average traffic by hour of day)
    This line plot shows average traffic volume by hour of day on Broadway.
    The x-axis represents hour of day from 0 to 23.
    The y-axis shows mean volume at that hour across the dataset.
    The line often rises during daytime hours and falls overnight, reflecting a daily cycle.
    """
    plt.figure()
    plt.plot(hourly_means.index, hourly_means.values, marker="o")
    plt.title("Broadway Traffic: Average Volume by Hour of Day")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Volume")
    plt.xticks(range(0, 24))
    plt.show()

    # OPTIONAL: Seaborn distribution (only if seaborn is installed)
    if HAS_SEABORN:
        """
        Alt text — Seaborn histogram (Broadway daily traffic totals)
        This histogram shows the distribution of daily total traffic volumes on Broadway.
        The x-axis represents daily total volume.
        The y-axis shows the count of days in each range.
        This chart is useful for seeing how daily totals vary and whether there are extreme days.
        """
        plt.figure()
        sns.histplot(daily.dropna(), bins=20)
        plt.title("Broadway Traffic: Distribution of Daily Totals (Seaborn)")
        plt.xlabel("Daily Total Volume")
        plt.ylabel("Count of Days")
        plt.show()


print("\nEnd of Week 4 self-guided workbook.\n")
