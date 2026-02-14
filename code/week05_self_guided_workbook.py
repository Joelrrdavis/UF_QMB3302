#!/usr/bin/env python3
"""
week05_self_guided_workbook.py
Week 5: Self-Guided Workbook — Regression & the Supervised ML Workflow

What this file is
- An OPTIONAL STUDENT PRACTICE WORKBOOK that accompanies week05.py.
- You are expected to READ week05.py first.
- This file reinforces concepts through guided prompts and light coding.

This file is NOT
- Not graded via code submission.
- Not a full re-implementation of week05.py.
- Not a place to optimize performance or add advanced models.

How to use this workbook
1) Run each section top-to-bottom.
2) Fill in TODO blocks where indicated.
3) Use print statements to check your understanding.
4) If something breaks, that is part of the learning process.

Note:

- Focus on *why* each step exists, not just *how* to write it.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# -----------------------------------------------------
# 1) Setup
# -----------------------------------------------------

DATA_PATH = Path("data") / "ames.csv"
TARGET = "SalePrice"
RANDOM_STATE = 1955

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}.\n"
        f"Make sure you have the same data folder used in week05.py."
    )


# -----------------------------------------------------
# 2) Load the data
# -----------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded.")
print("Shape:", df.shape)
print("\nPreview:")
print(df.head(3))

# TODO:
# - Verify that TARGET exists in the dataset.
# - Print the first few values of the target column.
#
# Think about this...:
# Why do we always define the target before thinking about models?


# -----------------------------------------------------
# 3) Define features (X) and target (y)
# -----------------------------------------------------

# TODO:
# - Create X by dropping the TARGET column.
# - Create y using the TARGET column.

# X = ...
# y = ...

# TODO:
# - Print the shapes of X and y.
#
# QUESTION:
# Why must X be two-dimensional and y one-dimensional?


# -----------------------------------------------------
# 4) Train / validation / test split
# -----------------------------------------------------

# We will use a simple two-step split, as in week05.py.

# TODO:
# Step 1: split into train+val and test (80% / 20%)
# Step 2: split train+val into train and val (75% / 25%)

# X_train, X_val, X_test = ...
# y_train, y_val, y_test = ...

# TODO:
# - Print the shapes of each split.
#
# QUESTION:
# What is the role of the validation set compared to the test set?


# -----------------------------------------------------
# 5) Identify numeric vs categorical features
# -----------------------------------------------------

# TODO:
# - Identify numeric columns using pandas dtypes.
# - Identify categorical columns as "everything else".

# numeric_features = ...
# categorical_features = ...

# TODO:
# - Print how many numeric and categorical features you found.
#
# QUESTION:
# Why do we treat numeric and categorical features differently?


# -----------------------------------------------------
# 6) Build a preprocessing pipeline
# -----------------------------------------------------

# TODO:
# - Create a numeric preprocessing pipeline:
#     * median imputation
#     * standard scaling
#
# - Create a categorical preprocessing pipeline:
#     * most-frequent imputation
#     * one-hot encoding
#
# - Combine them using ColumnTransformer.

# numeric_transformer = ...
# categorical_transformer = ...
# preprocess = ...

# QUESTION:
# What problem does one-hot encoding solve?


# -----------------------------------------------------
# 7) Baseline linear regression model
# -----------------------------------------------------

# TODO:
# - Create a Pipeline that includes:
#     * preprocess
#     * LinearRegression
#
# - Fit the model on the training data.
# - Generate predictions for:
#     * training set
#     * validation set

# model = ...
# model.fit(...)
# y_train_pred = ...
# y_val_pred = ...

# TODO:
# - Compute RMSE, MAE, and R^2 for the validation set.
#
# QUESTION:
# Why do we usually report validation performance instead of training performance?


# -----------------------------------------------------
# 8) Interpreting coefficients (conceptual)
# -----------------------------------------------------

# TODO:
# - Inspect the model coefficients if possible.
# - Identify at least one feature with a large positive or negative coefficient.
#
# QUESTION:
# Why should we be cautious when interpreting regression coefficients?


# -----------------------------------------------------
# 9) Ridge regression (bias–variance tradeoff)
# -----------------------------------------------------

# Ridge adds a penalty that shrinks coefficients.

# TODO:
# - Fit two Ridge models:
#     * one with a small alpha (e.g., 0.1)
#     * one with a large alpha (e.g., 50)
#
# - Compare their validation RMSE.

# ridge_small = ...
# ridge_large = ...

# QUESTION:
# How does increasing alpha affect bias and variance?


# -----------------------------------------------------
# 10) Reflection
# -----------------------------------------------------

print("- Defining a supervised learning problem")
print("- Creating train/validation/test splits")
print("- Building preprocessing + modeling pipelines")
print("- Evaluating regression models with proper metrics")

# FINAL QUESTIONS (answer for yourself, not in code):
# 1) Why is the workflow more important than the specific algorithm?
# 2) Why is regularization a *design choice*, not just a tuning trick?
# 3) What mistakes would pipelines help prevent in real projects?

