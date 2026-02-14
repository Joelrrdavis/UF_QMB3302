# one_hot_encoding_step_by_step.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import pairwise_distances

# -------------------------------------------------
# Step 0) tiny dataset 
# -------------------------------------------------
df = pd.DataFrame({
    "color": ["Red", "Blue", "Green", "Blue"]
})

print("\nSTEP 0 — Original data (categories are labels, not measurements)")
print(df)

# -------------------------------------------------
# Step 1) The tempting (but wrong) shortcut: assign numbers
# -------------------------------------------------
# This creates fake order/distance: Blue=2 is "greater than" Red=1, etc.
mapping = {"Red": 1, "Blue": 2, "Green": 3}
df["color_code_wrong"] = df["color"].map(mapping)

print("\nSTEP 1 — Wrong shortcut: categories turned into numbers")
print(df[["color", "color_code_wrong"]])

# problem: distance now has meaning that isn't real
# (distance-based methods will treat 1,2,3 like a line with spacing)
X_wrong = df[["color_code_wrong"]].to_numpy()
dist_wrong = pairwise_distances(X_wrong)

print("\nSTEP 1A — Distances created by the wrong encoding (fake geometry)")
print(dist_wrong.astype(int))

print("\nInterpretation:")
print("- Red(1) to Blue(2) distance = 1")
print("- Red(1) to Green(3) distance = 2  (suggests 'Green is twice as far' — nonsense for labels)")

# -------------------------------------------------
# Step 2) The core idea: one-hot encoding (yes/no questions)
# -------------------------------------------------
# Each category becomes its own column: Is color == Red? Is color == Blue? ...
enc = OneHotEncoder(sparse_output=False)  # dense array so it's easy to print
X_onehot = enc.fit_transform(df[["color"]])
categories = enc.categories_[0]  # e.g., ['Blue', 'Green', 'Red'] (alphabetical)

onehot_df = pd.DataFrame(
    X_onehot.astype(int),
    columns=[f"is_{c.lower()}" for c in categories]
)

print("\nSTEP 2 — One-hot encoding (each column is a yes/no question)")
print(pd.concat([df[["color"]], onehot_df], axis=1))

# -------------------------------------------------
# Step 3) What geometry looks like after one-hot encoding
# -------------------------------------------------
dist_onehot = pairwise_distances(X_onehot)

print("\nSTEP 3 — Distances after one-hot encoding (no fake ordering)")
print(dist_onehot.round(1))

print("\nInterpretation:")
print("- Different colors are equally 'different' (same distance),")
print("  rather than pretending Green is farther from Red than Blue is.")

# -------------------------------------------------
# Step 4) Show the dataset shape change (one column becomes many)
# -------------------------------------------------
print("\nSTEP 4 — Shape change")
print("Original features:", df[['color']].shape[1], "column")
print("One-hot features:", onehot_df.shape[1], "columns")

print("\nFinal takeaway:")
print("- One-hot encoding translates labels into numbers WITHOUT inventing order or distance.")
print("- It preserves meaning, but changes the dataset's shape.")
