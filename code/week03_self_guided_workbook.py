"""
week03_self_guided_workbook.py

Week 3 — Self-Guided Practice Workbook (NO solutions)

Focus:
- Loading a real CSV into pandas
- Inspecting rows/columns/types
- Selecting, filtering, sorting
- Missing values (detecting and simple handling)
- Group summaries (value_counts, groupby)
- Saving a cleaned file

Dataset used:
- seattle_pet_licenses.csv

How you know you’re “right”:
- You are NOT trying to match an exact number.
- You are looking for patterns described in the CHECK YOURSELF notes.
- If your outputs match those patterns (shapes, columns, visible changes), you’re on track.

------------------------------------------------------------
Running this file:
- Put the CSV in: data/seattle_pet_licenses.csv

- In VS Code terminal:
    python week03_self_guided_workbook.py
------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# ============================================================
# 0) Load the dataset
# ============================================================

# Expected file location (relative to where you run the script)
data_path = Path("data") / "seattle_pet_licenses.csv"


pets = pd.read_csv(data_path)

print("\nLoaded Seattle Pet Licenses dataset.")
print(pets.head())


# ============================================================
# 1) Inspect the dataset (shape, columns, dtypes)
# ============================================================

# The goal: build the habit of checking structure immediately after loading.
print("\nShape (rows, columns):", pets.shape)
print("Columns:", list(pets.columns))

print("\nData types (dtypes):")
print(pets.dtypes)

# TRY THIS:
# - Scroll up and look at pets.head(). Do the column names look reasonable?
# - Which columns look categorical (text) vs numeric?
#
# CHECK YOURSELF:
# - Real datasets often have many categorical columns (breed, species, etc.).
# - It is normal for CSVs to load text columns as "object" dtype.


# ============================================================
# 2) Basic row and column selection
# ============================================================

# Selecting a single column produces a Series
# (Replace 'Species' with a column you actually have, if needed.)
#
# TIP: use the printed list of columns above.
#
# TRY THIS:
# - Pick one column name and print the first 10 values.
#
# Example (edit the column name):
print(pets["Species"].head(10))

# Selecting multiple columns produces a DataFrame (double brackets)
# TRY THIS:
# - Pick 2–3 columns and print the first 10 rows.
#
# Example (edit column names):
print(pets[["Species", "Primary Breed"]].head(10))


# ============================================================
# 3) Value counts (fast summaries for categories)
# ============================================================

# value_counts() answers: "How many rows are in each category?"
#
# TRY THIS:
# - Choose a categorical column and run value_counts().
# - Then add .head(10) to show only the top 10 categories.
#
# Example (edit the column name):
print(pets["Species"].value_counts().head(10))

# CHECK YOURSELF:
# - You should see counts sorted from most common to least common.
# - If you choose a high-cardinality column, you may see many unique values.


# ============================================================
# 4) Filtering rows
# ============================================================

# Filtering answers questions like:
# - "Show me only dogs"
# - "Show me only a specific breed"
#
# TRY THIS:
# 1) Choose a column and a value you see in the dataset.
# 2) Filter the dataset to those rows.
#
# Example pattern:
dogs = pets[pets["Species"] == "DOG"]
print(dogs.head())
print("Filtered shape:", dogs.shape)

# CHECK YOURSELF:
# - The filtered shape should have fewer rows than the original.
# - If you get 0 rows, your filter value probably does not match exactly
#   (case, spacing, spelling).


# ============================================================
# 5) Sorting (make patterns easier to see)
# ============================================================

# Sorting helps you quickly inspect extremes:
# - largest values
# - earliest/latest dates (if you have dates)
#
# TRY THIS:
# - Identify a numeric column (if any) and sort descending.
#
# Example pattern:
#pets_sorted = pets.sort_values("SomeNumericColumn", ascending=False)
#print(pets_sorted.head(10))
# IF you get an error...
# you probably did not add in a real numeric column instead of "SomeNumericColumn"

# CHECK YOURSELF:
# - The top rows should show larger values in the sorted column.


# ============================================================
# 6) Missing values (detect + simple handling)
# ============================================================

# Missing values are common in real data.
# Before you "fix" them, you first measure them.
print("\nMissing values per column:")
print(pets.isna().sum())

# TRY THIS:
# - Identify a column with missing values.
# - Decide whether missing values are acceptable for your question.
#
# Two common strategies:
# A) Drop rows with missing values in a specific column
# B) Fill missing values with a placeholder for categorical columns

# Example A (drop missing for one column):
# clean_a = pets.dropna(subset=["Breed"])
# print("Rows before:", len(pets), "Rows after:", len(clean_a))
# Check the column names if you get an error!

# Example B (fill missing for one column):
# clean_b = pets.copy()
# clean_b["Primary Breed"] = clean_b["Primary Breed"].fillna("Unknown")
# print(clean_b["Primary Breed"].isna().sum())

# CHECK YOURSELF:
# - After dropping, row count decreases.
# - After filling, missing count for that column becomes 0.


# ============================================================
# 7) Simple cleaning: standardize text
# ============================================================

# Real text fields often contain inconsistencies:
# - extra spaces
# - different capitalization
#
# TRY THIS:
# - Pick a column with text values (Species, Breed, etc.).
# - Standardize it by stripping spaces and uppercasing.
#
# Example (edit column name):
# pets_clean = pets.copy()
# pets_clean["Species"] = pets_clean["Species"].astype(str).str.strip().str.upper()
# print(pets_clean["Species"].value_counts().head(10))

# CHECK YOURSELF:
# - You may see categories merge (e.g., "Dog" and "DOG" become one).
# - If you had inconsistent spacing, strip() can reduce duplicates.


# ============================================================
# 8) Group summaries (the key move before machine learning)
# ============================================================

# groupby lets you summarize numeric values by category.
# Many real datasets have mostly categorical data, so you may need to find
# a numeric column first.
#
# TRY THIS:
# - Find a numeric column (look at dtypes printed earlier).
# - Choose a category column (e.g., Species).
# - Compute the mean of the numeric column by category.
#
# Example pattern:
# pets.groupby("Species")["SomeNumericColumn"].mean()

# If you do not have a numeric column:
# - You can still group and count rows per category using size():
#
# Example:
# print(pets.groupby("Species").size().sort_values(ascending=False).head(10))

# CHECK YOURSELF:
# - size() returns counts of rows per group.
# - mean() only makes sense for numeric columns.


# ============================================================
# 9) Save a cleaned copy (so your work is reusable)
# ============================================================

# Saving a cleaned dataset is a common “end of workflow” step.
#
# TRY THIS:
# - Create a cleaned version (e.g., fill missing values, standardize a text column).
# - Save it to data/seattle_pet_licenses_clean.csv
#
# Example (edit columns):
# pets_clean = pets.copy()
# pets_clean["Species"] = pets_clean["Species"].astype(str).str.strip().str.upper()
# out_path = Path("data") / "seattle_pet_licenses_clean.csv"
# out_path.parent.mkdir(parents=True, exist_ok=True)
# pets_clean.to_csv(out_path, index=False)
# print("Saved:", out_path)

# CHECK YOURSELF:
# - You should see the new CSV file in your data/ folder.
# - When you load it again, your cleaning changes should persist.

