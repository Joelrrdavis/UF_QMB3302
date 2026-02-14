#!/usr/bin/env python3
"""
week06_applied_demo.py
End-to-End Supervised Learning on Real Housing Data (Instructor Walkthrough)

Intent here:
- This is a WALKTHROUGH / DEMO, not a student lab.
- The goal is workflow discipline + modeling judgment on messy real data.
- demonstrate: regression, logistic regression, decision trees, random forests.
- This is required, but the intuition/concepts, NOT the code. This code gets crazy...

Data
- homesadj.csv (This is *portion* of a dataset I used for a real world client.)
- Place it in your course data folder and set DATA_PATH below accordingly.

WRINKLE: Repeat sales (the same home appears multiple times)
This dataset contains homes that sold more than once. That creates two issues:

1) Evaluation leakage via dependence
   If the same home appears in both train and test, performance can look artificially good,
   because the model "recognizes" the home through stable attributes (sqft, neighborhood, etc.).

2) Modeling perspective: what is a "row"?
   For many real housing prediction questions, the MOST RECENT sale is the best proxy for the
   home's current latent state (market conditions + renovations + wear + neighborhood change).
   Earlier sales are increasingly stale.... but.... there is signal here. 

Our modeling choice for this demo (it is not the perfect solution, but it is a defensible approach):
- We will FILTER TO THE MOST RECENT SALE per home.
- Optionally, if I feel like it,  we will add a turnover feature (e.g., number of sales observed for that home),
  because high turnover may itself carry signal (e.g., investor activity, issues, flips, etc.) (note, there is
  a signal here... can we capture it? Let's see...)

To Users:
- This may look chaotic, crazy and messy. It is. 
- This is not harder ML; but it is for sure more realistic ML.
- Hard parts here are data decisions, not fancy algorithms. But I like fancy algorithms so we
well use a lot of them. 
"""

from __future__ import annotations

# ----------------------------
# 0) Imports + global settings
# ----------------------------

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd

# Metrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

# Model selection / evaluation
from sklearn.model_selection import train_test_split

# Preprocessing + pipelines
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Models (regression + classification)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


# ----------------------------
# 1) Configuration
# ----------------------------

# TODO: adjust to your repo layout (e.g., data/homesadj.csv)
DATA_PATH = Path("data") / "homesadj.csv"

RANDOM_STATE = 42

# Repeat-sale handling configuration
# We need a stable "home id" (grouping key) + a sale date column to pick "most recent".
# Preferred grouping key: folio (parcel id). Backup: site_address.
HOME_ID_COL_PRIMARY = "folio"
HOME_ID_COL_FALLBACK = "site_address"
SALE_DATE_COL = "sale_date"

# Optional turnover feature
ADD_TURNOVER_FEATURE = True
TURNOVER_FEATURE_NAME = "num_sales_observed"  # count of sales rows for that home in the dataset

# Teaching note:
# Keep thresholds and hyperparameters intentionally simple.
# This is not a tuning/competition script.


# ----------------------------
# 2) Helper functions (minimal)
# ----------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def summarize_regression(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> None:
    """Print a compact regression metric summary."""
    print(f"\n--- {label} (Regression metrics) ---")
    print(f"RMSE: {rmse(y_true, y_pred):,.2f}")
    print(f"MAE : {mean_absolute_error(y_true, y_pred):,.2f}")
    print(f"R^2 : {r2_score(y_true, y_pred):.3f}")


def summarize_classification(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None, label: str) -> None:
    """Print a compact classification metric summary."""
    print(f"\n--- {label} (Classification metrics) ---")
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=3))
    if y_proba is not None:
        try:
            print(f"AUC: {roc_auc_score(y_true, y_proba):.3f}")
        except Exception:
            # AUC can fail if only one class appears in y_true.
            print("AUC: (not available for this split)")


# ----------------------------
# 3) Load + orient the dataset
# ----------------------------

print("\n=== Week 6 Applied Demo: Real Housing Data ===")

# CHECKPOINT: framing
# - "This is not harder ML; this is more realistic ML."
# - "Today we model decisions, not just code."

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Could not find {DATA_PATH}. "
        "Place homesadj.csv in the expected folder or update DATA_PATH."
    )

df_raw = pd.read_csv(DATA_PATH)

# CHECKPOINT: dataset orientation
print("\nDataset preview (raw):")
print(df_raw.head(3))
print("\nRaw shape:", df_raw.shape)
print("\nRaw dtypes:")
print(df_raw.dtypes)

missing = df_raw.isna().mean().sort_values(ascending=False)
print("\nMissingness rate (top 10):")
print(missing.head(10))


# ----------------------------
# 4) Repeat-sales handling: define the unit of prediction
# ----------------------------

# CHECKPOINT: "What is a row?"
# Homes may appear multiple times (repeat sales). If we treat each sale row as independent
# and split randomly, train/test leakage can occur: the same home could land in both sets.
#
# Modeling stance for this demo:
# 1) Filter to the MOST RECENT sale per home (a clean, defensible default).
# 2) Optionally include a turnover feature: how many sales rows we observed for that home.
#
# Key line to say out loud:
# "The last sale is usually more predictive than the three sales before it,
#  but repeated turnover can itself be a meaningful signal."

# Choose a grouping key
home_id_col = None
if HOME_ID_COL_PRIMARY in df_raw.columns:
    home_id_col = HOME_ID_COL_PRIMARY
elif HOME_ID_COL_FALLBACK in df_raw.columns:
    home_id_col = HOME_ID_COL_FALLBACK
else:
    raise ValueError(
        f"Could not find a home id column. Expected '{HOME_ID_COL_PRIMARY}' "
        f"or '{HOME_ID_COL_FALLBACK}' to exist."
    )

# Parse sale_date if present
# NOTE: If SALE_DATE_COL is missing, we cannot reliably choose "most recent".
if SALE_DATE_COL not in df_raw.columns:
    raise ValueError(
        f"Expected '{SALE_DATE_COL}' to filter to most recent sale per home, but it was not found."
    )

df = df_raw.copy()
df[SALE_DATE_COL] = pd.to_datetime(df[SALE_DATE_COL], errors="coerce")

# Inspect repeat sales
sales_per_home = df.groupby(home_id_col).size().sort_values(ascending=False)
num_repeat_homes = int((sales_per_home > 1).sum())

print(f"\nRepeat-sales scan using home id = '{home_id_col}':")
print(f"- Homes with >1 observed sale: {num_repeat_homes:,}")
print(f"- Max sales for a single home: {int(sales_per_home.max()):,}")

# Optional turnover feature (computed BEFORE filtering to most recent sale)
if ADD_TURNOVER_FEATURE:
    # Turnover count is a *summary* of history, not direct leakage from past prices.
    # It captures the idea: "frequent turnover may indicate something else going on."
    df[TURNOVER_FEATURE_NAME] = df.groupby(home_id_col)[home_id_col].transform("size")

# Filter to most recent sale per home
# Strategy:
# - Sort by date ascending, take the last row per home
# - Drop rows with missing sale_date only if necessary
df_sorted = df.sort_values(by=[home_id_col, SALE_DATE_COL])
df_most_recent = df_sorted.groupby(home_id_col, as_index=False).tail(1).copy()

print("\nAfter filtering to most recent sale per home:")
print("Shape:", df_most_recent.shape)

# From here on, we model using df_most_recent
df = df_most_recent


# ----------------------------
# 5) Define the prediction target
# ----------------------------

TARGET_REG = "sale_price"

# CHECKPOINT: target discipline
# - Define y first; everything else is downstream.
# - Ask: "Is this target stable, meaningful, and measured consistently?"

if TARGET_REG not in df.columns:
    raise ValueError(f"Expected target column '{TARGET_REG}' not found in dataset.")

print("\nTarget summary (sale_price) on most-recent sales:")
print(df[TARGET_REG].describe())


# ----------------------------
# 6) Leakage triage + feature selection policy
# ----------------------------

# CHECKPOINT: leakage and identifiers (the core lesson)
# Always exclude:
# - Unique identifiers
# - Direct identifiers (addresses)
# Discuss carefully:
# - Dates (time leakage, drift)
# - Geography (signal vs proxy vs fairness)
# - Post-hoc flags (if any)

ALWAYS_EXCLUDE = [
    "folio",          # used for grouping; not a feature
    "site_address",   # identifier
    SALE_DATE_COL,    # for this demo: exclude to avoid time leakage; keep as "discussion point"
    # NOTE: if you want a "time-aware" advanced variant later, revisit this.
]

DISCUSS_CAREFULLY = [
    "neighborhood",   # strong signal, but proxy + high-cardinality
    "SiteCity",       # geography proxy
    "qualified",      # ambiguous: what does it mean? when assigned?
]

INCLUDE_DISCUSS = {
    "neighborhood": True,
    "SiteCity": True,
    "qualified": True,
}

# Build candidate feature list
candidate_cols = [c for c in df.columns if c != TARGET_REG]
present_excludes = [c for c in ALWAYS_EXCLUDE if c in df.columns]

final_feature_cols = []
for c in candidate_cols:
    if c in present_excludes:
        continue
    if c in DISCUSS_CAREFULLY:
        if c in df.columns and INCLUDE_DISCUSS.get(c, False):
            final_feature_cols.append(c)
        else:
            continue
    else:
        final_feature_cols.append(c)

print("\nSelected feature columns (count):", len(final_feature_cols))
print(final_feature_cols)

X = df[final_feature_cols].copy()
y_reg = df[TARGET_REG].copy()

print("\nFeature dtypes snapshot:")
print(X.dtypes.value_counts())

if ADD_TURNOVER_FEATURE and TURNOVER_FEATURE_NAME in X.columns:
    print(f"\nTurnover feature '{TURNOVER_FEATURE_NAME}' included.")
    print("Teaching note: This captures 'turnover frequency' without directly feeding past prices to the model.")


# ----------------------------
# 7) Train/test split (row-independent after most-recent filtering)
# ----------------------------

# CHECKPOINT: evaluation discipline
# Now that we reduced to 1 row per home, a standard train/test split is reasonable.

X_train, X_test, y_train, y_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=RANDOM_STATE
)

print("\nSplit sizes:")
print("Train:", X_train.shape, "Test:", X_test.shape)


# ----------------------------
# 8) Baseline regression (numeric-only)
# ----------------------------

# CHECKPOINT: baseline philosophy
# - Start with a restricted model on purpose to build intuition.
# - We will:
#   1) select numeric columns
#   2) impute missing values
#   3) scale
#   4) fit LinearRegression

numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

print("\nNumeric columns:", len(numeric_cols))
print("Categorical columns:", len(categorical_cols))

numeric_preprocess = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

baseline_reg_model = Pipeline(
    steps=[
        ("preprocess", numeric_preprocess),
        ("model", LinearRegression()),
    ]
)

baseline_reg_model.fit(X_train[numeric_cols], y_train)
y_pred_lr = baseline_reg_model.predict(X_test[numeric_cols])
summarize_regression(y_test.to_numpy(), y_pred_lr, label="Baseline Linear Regression (numeric-only)")

# CHECKPOINT: interpretation
# - Coefficients interpretation is tricky because:
#   * correlated predictors
#   * omitted variable bias (we omitted categoricals)
#   * scaling changes coefficient magnitude meaning


# ----------------------------
# 9) Full-feature regression pipeline (mixed types)
# ----------------------------

# CHECKPOINT: pipelines prevent common real-data failures:
# - categorical strings cause model errors
# - leakage from preprocessing fit on full data
# - inconsistent transforms train vs test

full_preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_cols),
        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_cols),
    ],
    remainder="drop",
)

full_lr = Pipeline(
    steps=[
        ("preprocess", full_preprocess),
        ("model", LinearRegression()),
    ]
)

full_lr.fit(X_train, y_train)
y_pred_full_lr = full_lr.predict(X_test)
summarize_regression(y_test.to_numpy(), y_pred_full_lr, label="Linear Regression (numeric + categorical via pipeline)")


# ----------------------------
# 10) Logistic regression (reframe as classification)
# ----------------------------

# CHECKPOINT: problem formulation is a decision
# Example: classify "above median price"
median_price = y_train.median()
y_train_cls = (y_train >= median_price).astype(int)
y_test_cls = (y_test >= median_price).astype(int)

print("\nClassification target: above-median sale price")
print("Median threshold (train):", float(median_price))

log_reg = Pipeline(
    steps=[
        ("preprocess", full_preprocess),
        ("model", LogisticRegression(max_iter=200)),
    ]
)

log_reg.fit(X_train, y_train_cls)

y_pred_cls = log_reg.predict(X_test)
y_proba_cls = None
try:
    y_proba_cls = log_reg.predict_proba(X_test)[:, 1]
except Exception:
    pass

summarize_classification(y_test_cls.to_numpy(), y_pred_cls, y_proba_cls, label="Logistic Regression (above-median classifier)")

# CHECKPOINT: interpretation shift
# - Regression predicts numeric outcome (dollars).
# - Logistic regression predicts probability of class membership.
# - Metrics change; confusion matrix becomes central.


# ----------------------------
# 11) Decision trees (regression + classification)
# ----------------------------

# CHECKPOINT: trees
# - no scaling required
# - handle nonlinearity
# - can overfit dramatically without constraints

tree_reg = Pipeline(
    steps=[
        ("preprocess", full_preprocess),
        ("model", DecisionTreeRegressor(random_state=RANDOM_STATE)),
    ]
)

tree_reg.fit(X_train, y_train)
y_pred_tree_reg = tree_reg.predict(X_test)
summarize_regression(y_test.to_numpy(), y_pred_tree_reg, label="Decision Tree Regressor")

tree_cls = Pipeline(
    steps=[
        ("preprocess", full_preprocess),
        ("model", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ]
)
tree_cls.fit(X_train, y_train_cls)
y_pred_tree_cls = tree_cls.predict(X_test)
summarize_classification(y_test_cls.to_numpy(), y_pred_tree_cls, None, label="Decision Tree Classifier")

# TODO (optional teaching enhancement):
# - Show explicit overfitting by comparing train vs test, then constrain max_depth and compare again.


# ----------------------------
# 12) Random forests (regression + classification)
# ----------------------------

# CHECKPOINT: forests
# - reduce variance by averaging many trees
# - often strong out-of-the-box performance
# - interpretability: feature importance is useful but imperfect

rf_reg = Pipeline(
    steps=[
        ("preprocess", full_preprocess),
        ("model", RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ]
)

rf_reg.fit(X_train, y_train)
y_pred_rf_reg = rf_reg.predict(X_test)
summarize_regression(y_test.to_numpy(), y_pred_rf_reg, label="Random Forest Regressor")

rf_cls = Pipeline(
    steps=[
        ("preprocess", full_preprocess),
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ]
)

rf_cls.fit(X_train, y_train_cls)
y_pred_rf_cls = rf_cls.predict(X_test)

y_proba_rf_cls = None
try:
    y_proba_rf_cls = rf_cls.predict_proba(X_test)[:, 1]
except Exception:
    pass

summarize_classification(y_test_cls.to_numpy(), y_pred_rf_cls, y_proba_rf_cls, label="Random Forest Classifier")


# ----------------------------
# 13) Feature importance (critical interpretation)
# ----------------------------

# CHECKPOINT: feature importance is NOT causality
# - correlated predictors split importance
# - high-cardinality categoricals can dominate
# - proxies can appear "important" for the wrong reasons
#
# Implementation note:
# - With one-hot encoding, feature names expand.
# - Extracting them cleanly takes a few steps.
# - For this skeleton, we leave a placeholder.

# TODO: implement feature name extraction from ColumnTransformer + OneHotEncoder
# TODO: extract rf_reg feature_importances_ and print top 15 features
# TODO: repeat for rf_cls if desired


# ----------------------------
# 14) Cross-model comparison summary (synthesis)
# ----------------------------

# CHECKPOINT: tradeoffs
# - Linear regression: interpretable baseline, may underfit.
# - Logistic regression: clean decisions, different goal.
# - Trees: flexible but unstable.
# - Forests: strong performance, weaker interpretability.
#
# TRANSITION: why unsupervised learning exists
# - Supervised learning assumes labels exist and are trustworthy.
# - Next question: what if we don't have a target label (or can't agree on one)?

print("\n=== Synthesis ===")
print("Key lesson: Real-world ML is dominated by data + decisions, not algorithms.")
print("Next transition: What if we *don't* have a target label? → Unsupervised learning.")


# ----------------------------
# 15) Optional: “Model card” style recap
# ----------------------------

# TODO: print a short recap block:
# - data used (most recent sale per home)
# - target definition
# - excluded columns + why
# - included 'discuss carefully' columns + why
# - whether turnover feature was included + why
# - best-performing model (on chosen metric)
# - top limitations / caveats

print("\nDone.")
