"""
week04_main.py

Simulation and Visualization

- Synthetic data generation using random processes
- Visualization for reasoning and exploration
- An end-to-end workflow: simulate → summarize → visualize → interpret


"""

import random
import math

import matplotlib.pyplot as plt
import seaborn as sns

#if you run the above, and get "ModuleNotFoundError: No module named 'package_name_whatever'"
# it means you need to install the right package
# if you are using uv (check week 0 and week 1 setup)
# type: uv add package name whatever. Check you are in the terminal!

# ------------------------------------------------------------
# 1) Simulation: Randomness and reproducibility
# ------------------------------------------------------------


# Set a seed so results are reproducible
random.seed(1955)

# Simulate numeric data: uniform random values between 0 and 1
n = 200
values = [random.random() for _ in range(n)]

values

# ------------------------------------------------------------
# 2) Summarize before visualizing
# ------------------------------------------------------------

#Summary Statistics

min_value = min(values)
max_value = max(values)
mean_value = sum(values) / len(values)

# Are these values what we expect?
print("Summary of simulated values:")
print("Minimum:", min_value)
print("Maximum:", max_value)
print("Mean:", mean_value)


# ------------------------------------------------------------
# 3) Visualization foundations (matplotlib)
# ------------------------------------------------------------

# ---------- Histogram ----------

"""
Alt text — Histogram
This histogram shows the distribution of 200 simulated values between 0 and 1.
The x-axis represents the value, from 0 to 1.
The y-axis represents how many observations fall into each range (frequency).
Because the data was generated uniformly at random, the bars should be
roughly similar in height, with small variations due to randomness.
This chart is useful for seeing the overall shape of the distribution.
"""

plt.figure()
plt.hist(values, bins=20, edgecolor="black")
plt.title("Distribution of Simulated Values")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()


# ---------- Line plot ----------

"""
Alt text — Line plot
This line plot shows the simulated values in the order they were generated.
The x-axis represents the observation index (the order of each value).
The y-axis represents the value between 0 and 1.
There is no meaningful trend expected; fluctuations reflect randomness.
This chart is useful for seeing variability over a sequence.
"""

plt.figure()
plt.plot(values)
plt.title("Simulated Values in Generation Order")
plt.xlabel("Observation Index")
plt.ylabel("Value")
plt.show()


# ---------- Scatter plot ----------

"""
Alt text — Scatter plot
This scatter plot shows the relationship between two simulated numeric variables.
The x-axis represents original values generated between 0 and 1.
The y-axis represents those same values after adding a small amount of random noise.
Each point corresponds to one observation.
The points form a diagonal looking cloud from lower left to upper right, indicating a positive relationship.
The vertical spread around the diagonal is because of the added noise.
"""

# lets change up the dataset

n = 400
# Base variable: values between 0 and 1
values = [random.random() for _ in range(n)]
# Create a related variable by adding small noise
noise = [random.gauss(0, 0.05) for _ in range(n)]
related_values = [v + e for v, e in zip(values, noise)]


plt.figure()
plt.scatter(values, related_values, s=60, alpha=1.0, marker="o")
plt.title("Relationship Between Simulated Variables")
plt.xlabel("Original Value")
plt.ylabel("Value with Noise")
plt.show()



# ---------- Bar plot ----------

"""
Alt text — Bar plot
This bar plot shows the number of observations assigned to each category.
The x-axis lists the categories (A, B, C).
The y-axis represents the count in each category.
Because categories were assigned randomly, the bars should be similar in height
but not exactly equal.
This chart is useful for comparing counts across categories.
"""


# Number of observations
n = 300
categories = ["A", "B", "C"]
# Randomly assign a category to each observation
assigned_categories = [random.choice(categories) for _ in range(n)]
# Count how many times each category appears
counts = {
    category: assigned_categories.count(category)
    for category in categories
}

plt.figure()
plt.bar(counts.keys(), counts.values())
plt.title("Counts by Simulated Category")
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()

# ---------- Seaborn histogram (optional) ----------

"""
Alt text — Seaborn histogram
This plot shows the distribution of simulated values using seaborn.
The x-axis represents the value from 0 to 1.
The y-axis represents the count in each bin.
The distribution should appear approximately uniform across the range 0 to 1.
This chart is useful for quickly visualizing distribution shape.
"""

plt.figure()
sns.histplot(values, bins=20)
plt.title("Distribution of Simulated Values (Seaborn)")
plt.xlabel("Value")
plt.ylabel("Count")
plt.show()

# ------------------------------------------------------------
# Seaborn (optional): Statistical visualization
# ------------------------------------------------------------


print(
    "\n[Alt text : Seaborn histogram]\n"
    "This plot shows the distribution of simulated values using seaborn.\n"
    "It emphasizes the shape of the distribution rather than exact counts.\n"
    "The distribution should appear approximately uniform across the range 0 to 1."
)

plt.figure()
sns.histplot(values, bins=20)
plt.title("Distribution of Simulated Values (Seaborn)")
plt.xlabel("Value")
plt.ylabel("Count")
plt.show()


# ------------------------------------------------------------
# mini-lab workflow- Titanic
# ------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Titanic dataset: visualizing real-world data
# ============================================================

# Load the Titanic dataset
titanic = pd.read_csv("data/titanic.csv") # load this however you prefer!

titanic.head()
titanic.info()
titanic.describe()

# Remember this!
# Each row represents one observation (one passenger).
# Each column represents one variable about that observation.

# Some variables are numeric (Age, Fare).
# Others are categorical (Sex, Pclass).
# Different chart types and different modeling steps (later) 
# make sense for each.

# ---------- Histogram ----------

"""
Alt text — Histogram (Age)
This histogram shows the distribution of passenger ages on the Titanic.
The x-axis represents age in years.
The y-axis represents the number of passengers within each age range.
Most passengers are adults, with a large concentration between roughly
20 and 40 years old.
There are fewer children and fewer elderly passengers.
This chart is useful for understanding the age composition of the passengers.
"""

print(titanic.columns)
print(titanic["Age"].head())
print(titanic["age"].notna().sum())


plt.figure()
plt.hist(titanic["age"].dropna(), bins=20, edgecolor="black")
plt.title("Distribution of Passenger Ages")
plt.xlabel("age")
plt.ylabel("Number of Passengers")
plt.show()

# ---------- Bar plot ----------

"""
Alt text — Bar plot (Survival)
This bar plot shows how many passengers survived versus did not survive.
The x-axis shows survival status: 0 represents passengers who did not survive,
and 1 represents passengers who survived.
The y-axis represents the number of passengers in each group.
The bar for non-survivors is taller, indicating that more passengers
did not survive than survived.
This chart is useful for comparing counts across categories.
"""

survival_counts = titanic["survived"].value_counts().sort_index()

plt.figure()
plt.bar(survival_counts.index, survival_counts.values)
plt.title("Survival Counts on the Titanic")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.show()

# ---------- Bar plot (grouped conceptually) ----------

"""
Alt text — Bar plot (Survival by Sex)
This bar plot shows the number of survivors and non-survivors by sex.
The x-axis represents sex (male and female).
The y-axis represents the number of passengers.
Female passengers show a higher count of survivors compared to non-survivors,
while male passengers show a higher count of non-survivors.
This chart is useful for comparing outcomes across groups.
"""

survival_by_sex = titanic.groupby("sex")["survived"].sum()
total_by_sex = titanic["sex"].value_counts()

plt.figure()
plt.bar(survival_by_sex.index, survival_by_sex.values, label="survived")
plt.bar(
    total_by_sex.index,
    total_by_sex.values - survival_by_sex.values,
    bottom=survival_by_sex.values,
    label="Did Not Survive"
)
plt.title("Survival by Sex")
plt.xlabel("Sex")
plt.ylabel("Number of Passengers")
plt.legend()
plt.show()


# Fix?
#Matplotlib stacks by position, not by label.
#Pandas aligns by label—but only if you let it.


survival_table = pd.crosstab(titanic["sex"], titanic["survived"])

plt.figure()
plt.bar(
    survival_table.index,
    survival_table[1],
    label="Survived"
)
plt.bar(
    survival_table.index,
    survival_table[0],
    bottom=survival_table[1],
    label="Did Not Survive"
)
plt.title("Survival by Sex")
plt.xlabel("Sex")
plt.ylabel("Number of Passengers")
plt.legend()
plt.show()


# ---------- Scatter plot ----------

"""
Alt text — Scatter plot (Age vs Fare)
This scatter plot shows the relationship between passenger age and ticket fare.
The x-axis represents age in years.
The y-axis represents fare paid for the ticket.
Each point represents one passenger.
Most passengers paid lower fares, regardless of age.
A small number of passengers paid very high fares, appearing as points high
on the y-axis.
This chart is useful for seeing relationships and outliers in numeric data.
"""

plt.figure()
plt.scatter(
    titanic["age"],
    titanic["fare"],
    alpha=0.5
)
plt.title("Passenger Age vs Ticket Fare")
plt.xlabel("age")
plt.ylabel("fare")
plt.show()

# Notice that Fare values are on a very different scale than Age.
# This matters later when we fit models!


# ---------- Box plot ----------

"""
Alt text — Box plot (Fare by Passenger Class)
This box plot shows the distribution of ticket fares for each passenger class.
The x-axis represents passenger class (1st, 2nd, and 3rd class).
The y-axis represents fare paid.
First-class passengers generally paid much higher fares than second- and
third-class passengers.
The spread of fares is widest for first-class passengers, with several
high-value outliers.
This chart is useful for comparing distributions across groups.
"""

plt.figure()
plt.boxplot(
    [
        titanic[titanic["Pclass"] == 1]["fare"].dropna(),
        titanic[titanic["Pclass"] == 2]["fare"].dropna(),
        titanic[titanic["Pclass"] == 3]["fare"].dropna(),
    ],
    labels=["1st Class", "2nd Class", "3rd Class"]
)
plt.title("Ticket Fare by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")
plt.show()

# if the above didn't run... see if you can fix it. 

# In modeling, we usually distinguish between:
# - features (inputs): variables we use to explain or predict
# - target (output): the variable we care about

# Example (Titanic):
# Features: Age, Sex, Pclass, Fare
# Target: Survived


