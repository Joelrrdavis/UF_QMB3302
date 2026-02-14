#!/usr/bin/env python3
"""
week05.py
Week 5: Supervised Learning with Regression (Ames Housing)

What this script is
- A guided, end-to-end walkthrough of the supervised ML workflow for REGRESSION.
- Focus: train/validation/test splits, regression metrics, scaling, linear regression,
  coefficient interpretation, regularization (Ridge/Lasso), and pipelines.
- Not  optimization. These models won't win any competitions. 

If your file uses different names:
- Update DATA_PATH and/or TARGET below.

Notes:
- focus on the *workflow* and the *reasoning*.
- read comments, watch the checkpoints, go slow but don't memorize code here
"""

"""
MENTAL MODEL FOR THIS WEEK

You are learning ONE idea:

    A model is a rule learned from past data
    that makes predictions on new data.

Everything else in this file exists to:
- make that rule fair
- make that rule testable
- make that rule interpretable

You are NOT trying to:
- get the best score
- tune endlessly
- memorize sklearn syntax
"""


from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

#you may need to add sklearn. 
#In your terminal, if you use uv...  run 'uv add scikit-learn'
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ----------------------------
# 1) Configuration
# ----------------------------

## LLM helpers... might show up in my code, and not yours. 

DATA_PATH = Path("data") / "ames.csv"    # adjust to your file path if different
TARGET = "SalePrice"                    # adjust if needed, check your targets
RANDOM_STATE = 1955

# OPTIONAL: A light regularization grid ... safe to ignore until optional content
RIDGE_ALPHAS = [0.0, 0.1, 1.0, 10.0, 50.0, 100.0]  # 0.0 ~ "no penalty" (approx)
LASSO_ALPHAS = [0.0005, 0.001, 0.01, 0.05, 0.1]

# Optional: target transformation (feature engineering on the target) Not relevant until optional section
USE_LOG_TARGET = False  # set True if you want a "log1p(SalePrice)" 


# ----------------------------
# 2) Metrics + utilities
# ----------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def metric_block(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }

def print_metrics(title: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    m = metric_block(y_true, y_pred)
    print(f"\n--- {title} ---")
    print(f"RMSE: {m['RMSE']:.2f}")
    print(f"MAE : {m['MAE']:.2f}")
    print(f"R^2 : {m['R2']:.3f}")

#the below might be good engineering, but it is hostile code. 
def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find dataset at: {path}\n"
            f"Expected a local CSV.\n"
            f"Fix: place your Ames CSV at {path} or update DATA_PATH in your code.py."
        )
    return pd.read_csv(path)

def train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size_within_train: float = 0.25,
    random_state: int = 1955
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Two-step split:
    1) Train+Val vs Test (test_size)
    2) Train vs Val (val_size_within_train, applied to the remaining data)

    Example: test_size=0.2 and val_size_within_train=0.25 yields:
    - Train: 60%
    - Val  : 20%
    - Test : 20%
    """
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size_within_train, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ----------------------------
# 3) Load + orient the dataset
# ----------------------------

# Load the dataset
import os
os.getcwd()

DATA_PATH = Path("data") / "ames.csv"    # adjust to your file path if different

if not os.path.exists(DATA_PATH):
    print("ERROR: Could not find the dataset.")
    print("Expected file:", DATA_PATH)
    print("Fix: Place the CSV in this folder or update DATA_PATH.")
else:
    df = pd.read_csv(DATA_PATH)

#or use this if you prefer the function from earlier
#df = safe_read_csv(DATA_PATH)

# CHECKPOINT: sanity check... lets look at the dataset
print(df.head(3))
print("\nShape:", df.shape)

# We want to seperate our Y (our Target variable) from our
# x's for modeling

TARGET = "SalePrice"

# Separate features/target
X = df.drop(columns=[TARGET]).copy()
y_raw = df[TARGET].copy()

X.head()
y_raw.head()

# Optional target feature engineering: log transform
# - Log transform can stabilize variance and make errors more proportional.
# - BUT: metrics are in log-units... that's harder to think about. 
#if USE_LOG_TARGET:
#    y = np.log1p(y_raw)
#    print("\nUsing log1p(SalePrice) as the modeling target (USE_LOG_TARGET=True).")
#else:
#    y = y_raw


print("\nTarget summary:")
print(y_raw.describe())

y = y_raw
y

# ----------------------------
# 4) Split: Train / Validation / Test
# ----------------------------

# CHECKPOINT: evaluation discipline
# - Train: fit model
# - Validation: choose hyperparameters / compare models
# - Test: final unbiased estimate 

X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
    X, y, test_size=0.2, val_size_within_train=0.25, random_state=RANDOM_STATE
)

print("\nSplit sizes:")
print("Train:", X_train.shape)
print("Val  :", X_val.shape)
print("Test :", X_test.shape)


# ----------------------------
# 5) Preprocessing: numeric vs categorical
# ----------------------------

# CHECKPOINT: why pipelines exist
# - Real datasets contain missing values and categories (strings).
# - Models like linear regression require numeric feature matrices.
# - Pipelines keep preprocessing consistent and prevent leakage.


#what are the different feature types?

df.info()
df

#group the numeric and categorical features
numeric_features: List[str] = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_features: List[str] = [c for c in X_train.columns if c not in numeric_features]

numeric_features
categorical_features

print("\nFeature types:")
print(f"- Numeric: {len(numeric_features)}")
print(f"- Categorical: {len(categorical_features)}")

#Now we can do something to all fo the different types, all at once.
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

# This encoding step converts categories into numeric indicators.
# We will study this transformation explicitly in a later lesson.
# For now, treat it as a required translation step.

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# HOW TO READ THIS (ColumnTransformer):
# - We have two groups of columns: numeric_features and categorical_features.
# - For each group, we define a mini-recipe (a Pipeline).
# - ColumnTransformer applies each recipe to the right columns.
# - preprocess.fit(X_train) learns:
#     - median values (numeric imputer)
#     - most common categories (categorical imputer)
#     - scaling parameters (mean/std)
#     - the one-hot categories to create
# - preprocess.transform(X_valid/test) applies the SAME learned rules (no relearning).

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop",
)

# SANITY CHECK: what comes out of preprocessing?
X_train_prepared = preprocess.fit_transform(X_train)
X_test_prepared = preprocess.transform(X_test)

print("Raw X_train shape:", X_train.shape)
print("Prepared X_train shape:", X_train_prepared.shape)
print("Prepared data type:", type(X_train_prepared))



# ----------------------------
# 6) Model 1: Linear Regression (baseline)
# ----------------------------

# CHECKPOINT: baseline first
# - A baseline model gives you a reference point.
# - If you cannot beat the baseline, "complexity" is not the solution!!

linreg = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", LinearRegression()),
    ]
)


linreg.fit(X_train, y_train)

pred_train = linreg.predict(X_train)
pred_val = linreg.predict(X_val)
pred_test = linreg.predict(X_test)

print_metrics("Linear Regression (Train)", y_train.to_numpy(), pred_train)
print_metrics("Linear Regression (Val)", y_val.to_numpy(), pred_val)
print_metrics("Linear Regression (Test)", y_test.to_numpy(), pred_test)

# CHECKPOINT: coefficient interpretation 
# Coefficients are easiest to interpret when:
# - features are scaled (we did StandardScaler for numeric)
# - categories are one-hot encoded (we did OneHot)
# Caveats:
# - correlated predictors can make coefficients unstable
# - one-hot encoding produces many related coefficients (one per category)

# Optional: top coefficients 


# ----------------------------
# 7) Scaling demonstration (why scaling matters for coefficients)
# ----------------------------

# CHECKPOINT: scaling vs prediction
# - Scaling may not dramatically change predictions for basic linear regression,
#   but it matters a lot for:
#   * coefficient comparability
#   * regularization (Ridge/Lasso), because penalties depend on coefficient sizes

numeric_transformer_no_scale = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        # no scaler
    ]
)

preprocess_no_scale = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer_no_scale, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop",
)

linreg_no_scale = Pipeline(
    steps=[
        ("preprocess", preprocess_no_scale),
        ("model", LinearRegression()),
    ]
)

linreg_no_scale.fit(X_train, y_train)
pred_val_no_scale = linreg_no_scale.predict(X_val)

print_metrics("Linear Regression (Val) - NO scaling", y_val.to_numpy(), pred_val_no_scale)
print_metrics("Linear Regression (Val) - WITH scaling", y_val.to_numpy(), pred_val)

print("\nCHECKPOINT:")
print("- Prediction performance may be similar.")
print("- Coefficients become more comparable when numeric features are standardized.")
print("- Regularization depends on scaling (next).")

