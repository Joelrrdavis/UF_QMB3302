"""
week03_main.py

Week 3: Working with Data: Pandas, Files, and APIs

This file is intentionally more detailed than the textbook. Focused on:
- Pandas DataFrames (load, inspect, clean, transform, summarize)
- Files + paths (relative paths, working directory, common errors)
- JSON + APIs (requests, response validation, parsing, flattening)

"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests


# ============================================================
# 0) Sanity checks: where am I running from?
# ============================================================

# Relative paths (like "data/sample.csv") are resolved from the
# current working directory (CWD).

cwd = os.getcwd()
print(cwd)

# A relative path is interpreted starting from the CWD.
# This shows what Python will actually try to access.
example_path = Path("data") / "sample.csv"
print("\nRelative path example:")
print(example_path)
print("Resolved absolute path:")
print(example_path.resolve())

# Listing files helps confirm you're in the right folder.
# limiting output here so so it stays readable.
print("\nFiles and folders in the CWD (first 20):")
try:
    print(sorted(os.listdir(cwd))[:20])
except Exception as e:
    print("Could not list directory contents:", e)

# This is often the fastest check when a FileNotFoundError occurs.
# Does 'data/sample.csv' exist from this CWD? Probably not... 
print(example_path.exists())

# ============================================================
# 1) What is a dataset?
# ============================================================

# A dataset is often a table:
# - rows = observations (cases, records)
# - columns = variables (features)

# Remember this! Rows are individual observations; columns are variables

df_small = pd.DataFrame(
    {
        "student": ["Alex", "Jordan", "Casey"],
        "score": [90, 85, 93],
        "major": ["Finance", "Marketing", "IS"],
    }
)

print(df_small)

# Basic structure of a DataFrame
print("Shape (rows, columns):", df_small.shape)
print("Column names:", list(df_small.columns))
print("Data types:")
print(df_small.dtypes)

# Selecting a single column returns a different object type
scores = df_small["score"]
print(scores)
print("Type of df_small['score']:", type(scores))

# ============================================================
# 2) Files + paths: use a RELATIVE path and ensure the CSV exists
# ============================================================

# We will load data from "data/example1.csv".
# This path is RELATIVE to the current working directory (CWD).
test_scores = Path("data") / "student_test_results.csv"
print(test_scores)

# If the file does not exist, create it! 
# This is the structure I will use for the rest of the course (a data folder inside your code folder)
# This statement (the if statement) is WAY more than you need here. Just get the idea. 

if not test_scores.exists():

    # Create the parent folder ("data/") if needed.
    test_scores.parent.mkdir(parents=True, exist_ok=True)

    # This dataset is intentionally messy:
    # - has missing values
    # - a column name with spaces (bad)
    # - a column we will later drop
    df_make = pd.DataFrame(
        {
            "Student Name": ["Alex", "Jordan", "Casey", "Riley", "Morgan"],
            "score": [90, 85, None, 72, 93],
            "major": ["Finance", "Marketing", "IS", None, "IS"],
            "unused_column": ["x", "x", "x", "x", "x"],
        }
    )

    df_make.to_csv(test_scores, index=False)
    print("CSV not found — created a sample dataset.")

else:
    print("CSV already exists — using the existing file.")


# ============================================================
# 3) Load CSV files into DataFrames (and verify each one)
# ============================================================

# Pandas can read CSVs from either:
# - a string path (e.g., "../data/file.csv")
# - a Path object (recommended in modern Python)
#
# We'll do BOTH styles, and we'll load TWO different CSVs.

# ---- Load #1 (Path object) ----
test_scores = Path("data") / "student_test_results.csv"
df = pd.read_csv(test_scores)

# remember what this is doing. test_scores is the path to the file, like directions
# print(test_scores)

print("Shape (rows, cols):", df.shape)
print("Columns:", list(df.columns))
print(df.head())

# ---- Load #2 (Loading, from the data folder) ----
# You might get stuck here... take your time. 

# First, we need a csv file to load. In Canvas, under files, is the csv called 
# segments.csv. Put that file in your 'data' folder

import pandas as pd
path= 'segments.csv'
segments_file = pd.read_csv(path)

# so where is it?
# two ways to create a path, and two simple steps

# 1. create a path (point at the file)
dir_path = "data/segments.csv"
# 2. load the data
segments_dir = pd.read_csv(dir_path)
#3. check that the data loaded
print(segments_dir.head())

#####
# OR
#####

# 1. create a path (in general this is the way we will prefer to load)
path = Path("data") / "segments.csv"
#2. load the data
segments = pd.read_csv(path)
#3. Check that the data loaded
print(segments.head())

# ============================================================
# 4) Inspect + summarize (structure, types, descriptive stats)
# ============================================================

# We will use the segments data from the prior section. If you need to reload it, 
# uncomment the below

# 1. create a path
# path = Path("data") / "segments.csv"
# 2. load the data using read_csv
# segments = pd.read_csv(path)
# 3. Check that the data loaded
# print(segments.head())

segments.info()

# you can slice dataframes
print(segments[0:3])

# ------------------------------------------------------------
# One-line summary statistics on a numeric column
# ------------------------------------------------------------

# Mean: average value
segments["age"].mean()

# Median: middle value (less sensitive to extreme values)
segments["age"].median()

# Count: number of non-missing values
segments["age"].count()

# Minimum and maximum values
segments["age"].min()
segments["age"].max()

# ------------------------------------------------------------
# One-line summaries for categorical data
# ------------------------------------------------------------

# Count how many rows fall into each category
segments["Segment"].value_counts()

# ------------------------------------------------------------
# Bundle many statistics at once
# ------------------------------------------------------------

# Summary statistics for a single column
segments["age"].describe()

# Summary statistics for all numeric columns
segments.describe()

# Getting data from a URL. Many public datasets are available as CSV files online.
# A common workflow is:
# 1) download into your /data folder
# 2) read it with pandas

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
wine_path = Path("data") / "winequality-red.csv"
wine_path.parent.mkdir(parents=True, exist_ok=True)

# Download only if the file isn't already there.
if not wine_path.exists():
    urllib.request.urlretrieve(url, wine_path)

# This dataset uses ';' as the separator (not a comma).
web_data = pd.read_csv(wine_path, sep=";")

print("\nWine data: first 10 rows")
print(web_data.head(10))

print("\nWine data: describe numeric columns")
print(web_data.describe())


# ============================================================
# 5) Missing values: detect them, then try two strategies
# ============================================================

# Don't forget to load the dataframe (df) if needed

# Missing counts per column (df.isna().sum()):")
print(df.isna().sum())

# Strategy A: drop rows with any missing values
# drop rows with missing values (df.dropna())
df_drop = df.dropna()
print("Before:", df.shape, "| After:", df_drop.shape)
df_drop
# Strategy B: fill missing values (simple choices, not always "correct")
# fill missing values"
df_fill = df.copy()

# Fill missing numeric values with the column mean
if "score" in df_fill.columns:
    mean_score = df_fill["score"].mean()
    df_fill["score"] = df_fill["score"].fillna(mean_score)

# Fill missing text values with a placeholder
if "major" in df_fill.columns:
    df_fill["major"] = df_fill["major"].fillna("Unknown")
df_fill
print("After filling, missing counts:")
print(df_fill.isna().sum())

# For the rest of this script, we continue using df_fill.
df = df_fill


# ============================================================
# 6) Rename columns + standardize names + drop an unused column
# ============================================================

# Columns BEFORE renaming/standardizing:
print(list(df.columns))

# Rename one known messy column explicitly
df = df.rename(columns={"Student Name": "student_name"})

# Standardize: trim, lowercase, spaces -> underscores
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# Columns AFTER renaming/standardizing:
print(list(df.columns))

# Drop an unused column if it exists
if "unused_column" in df.columns:
    df = df.drop(columns=["unused_column"])

df
# Preview after cleaning:
print(df.head())


# ============================================================
# 7) Selecting columns + filtering rows
# ============================================================

# Select one column (Series)
if "score" in df.columns:
    print(df["score"].head())

# Select multiple columns (DataFrame) — note the DOUBLE brackets
wanted_cols = [c for c in ["student_name", "score"] if c in df.columns]
if wanted_cols:
    print(df[wanted_cols].head())

# Filter rows with a condition where score > 80
if "score" in df.columns:
    df_high = df[df["score"] > 80]
    print(df_high)

    print("\n[7] Mean score for score > 80:")
    print(df_high["score"].mean())

# .loc = label-based selection (rows and columns by label)
if wanted_cols:
    print("\n[7] .loc example: df.loc[:, wanted_cols].head()")
    print(df.loc[:, wanted_cols].head())

# .iloc = position-based selection (rows and columns by integer position)
if df.shape[0] >= 3 and df.shape[1] >= 2:
    print("\n[7] .iloc example: first 3 rows, first 2 columns")
    print(df.iloc[0:3, 0:2])

# Common mistake example:
# df["student_name", "score"]  # KeyError (because this is a tuple, not a list)


# ============================================================
# 8) Descriptive statistics + group-level summary (intro concepts only!)
# ============================================================

if "score" in df.columns:
    print("\n[8] Basic descriptive stats on score:")
    print("Mean:", df["score"].mean())
    print("Median:", df["score"].median())
    print("Min:", df["score"].min())
    print("Max:", df["score"].max())

print("\n[8] Describe numeric columns again (post-cleaning):")
# print(df.describe(numeric_only=True))
print(df.describe())

# Group summary: mean score by major
if "major" in df.columns and "score" in df.columns:
    print("\n[8] Group summary: mean score by major")
    print(df.groupby("major")["score"].mean())


# ============================================================
# 9) APIs + JSON: request, validate, parse, and flatten
# ============================================================

# API call + JSON parsing (public demo endpoint)
# This section is just for demo purposes

url = "https://jsonplaceholder.typicode.com/todos/1"
print("[9] Requesting:", url)

try:
    response = requests.get(url, timeout=10)
    print("HTTP status:", response.status_code)

    # If status indicates an error (4xx/5xx), this raises an exception
    response.raise_for_status()

    data = response.json()  # JSON -> Python dict/list
    print("Parsed JSON (dict):")
    print(data)

    print("Keys:")
    print(list(data.keys()))

    # Convert a single dict to a one-row DataFrame
    print("JSON -> one-row DataFrame:")
    df_json = pd.DataFrame([data])
    print(df_json)

    # If JSON is nested, json_normalize is common:
    # df_flat = pd.json_normalize(data)
    # print(df_flat)

except requests.RequestException as e:
    print("API request failed (network restrictions happen):")
    print(e)
    print("Just carry on if this happens, we won't make great use of API requests")

# It printed already, but you may have missed it..
print(data)

# ============================================================
# 10) Mini-lab (no functions): a compact “import → clean → insight”
# ============================================================


# Load (already done?) → verify!
print("Rows/Cols:", df.shape)
print("Columns:", list(df.columns))

# Inspect
print(df.head())

# Summarize
if "score" in df.columns:
    print("Overall mean score:", df["score"].mean())

# Simple insight question: Who scored above 80?
if "score" in df.columns:
    above_80 = df[df["score"] > 80]
    print(above_80)


# What changed when we filled missing values instead of dropping rows?
# What assumptions did we make when we filled missing 'score' with the mean?
# Why do standardized column names help once code gets longer?


