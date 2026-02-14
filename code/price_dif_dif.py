"""
price_dif_dif.py

Generates synthetic weekly coffee transaction data with:
- Price change (week 17)
- Competitor price variation
- Weekly trend
- AR(1) demand persistence
- Revenue and profit calculation

Outputs:
    synthetic_coffee_price_experiment.csv
to the same directory as this script.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# -------------------------
# Reproducibility
# -------------------------
np.random.seed(42)

# -------------------------
# PARAMETERS
# -------------------------
weeks = 32
price_pre = 2.89
price_post = 2.99
unit_cost = 1.40

beta_0 = 5.5                 # baseline log demand level
beta_own = -0.8              # own-price elasticity
beta_cross = 0.4             # cross-price elasticity
beta_trend = 0.0015          # weekly growth (~0.15%)
rho = 0.5                    # AR(1) persistence
sigma = 0.03                 # random noise std

# -------------------------
# CREATE BASE DATAFRAME
# -------------------------
data = pd.DataFrame({
    "week": np.arange(1, weeks + 1)
})

data["period"] = np.where(data["week"] <= 16, "pre", "post")
data["price"] = np.where(data["period"] == "pre", price_pre, price_post)

# Competitor price fluctuates slightly
comp_base = 2.95
data["competitor_price"] = comp_base + np.random.normal(0, 0.03, weeks)

# -------------------------
# GENERATE DEMAND
# -------------------------
log_Q = np.zeros(weeks)

for t in range(weeks):

    trend_component = beta_trend * data.loc[t, "week"]
    price_component = beta_own * np.log(data.loc[t, "price"])
    cross_component = beta_cross * np.log(data.loc[t, "competitor_price"])
    noise = np.random.normal(0, sigma)

    if t == 0:
        log_Q[t] = beta_0 + price_component + cross_component + trend_component + noise
    else:
        log_Q[t] = (
            beta_0
            + price_component
            + cross_component
            + trend_component
            + rho * log_Q[t-1]
            + noise
        )

data["transactions"] = np.round(np.exp(log_Q)).astype(int)

# -------------------------
# ECONOMICS
# -------------------------
data["revenue"] = data["price"] * data["transactions"]
data["cost"] = unit_cost * data["transactions"]
data["profit"] = data["revenue"] - data["cost"]

# -------------------------
# EXPORT CSV
# -------------------------
output_path = "synthetic_coffee_price_experiment.csv"
data.to_csv(output_path, index=False)

print("Synthetic dataset created:")
print(output_path)
print("\nPreview:")
print(data.head())
