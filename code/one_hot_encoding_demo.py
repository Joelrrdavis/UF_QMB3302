"""
One-Hot Encoding — Standalone Coding Demo (first-time modelers)

GOAL
-----
Build an intuitive understanding of how categorical variables are converted into numbers
WITHOUT implying numeric order.

This file is intentionally linear and print-heavy.
It is NOT about pipelines, model performance, or hyperparameter tuning.
It is about representation.

Suggested pacing (live):
- Sections 1–2: core idea (10–15 min)
- Sections 3–4: sklearn + unknown categories (10–15 min)
- Section 5: bridge back to pipelines (2–5 min)
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def banner(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ----------------------------
# 1) The core problem: categories are labels, not quantities
# ----------------------------
banner("1) The problem: models require numbers, but real data has categories")

df = pd.DataFrame(
    {
        "Color": ["Red", "Blue", "Green", "Blue"],
        "Size": ["S", "M", "S", "L"],
        "Price": [10, 12, 11, 13],
    }
)

print("Toy dataset:\n", df)

print("\nKey question:")
print("- If we want to use 'Color' or 'Size' in a model, how do we represent them as numbers?")


# ----------------------------
# 2) A common mistake: mapping categories to integers (creates fake order)
# ----------------------------
banner("2) A common mistake: mapping categories to integers creates fake order")

df_bad = df.copy()
df_bad["Color_bad"] = df_bad["Color"].map({"Red": 1, "Blue": 2, "Green": 3})
df_bad["Size_bad"] = df_bad["Size"].map({"S": 1, "M": 2, "L": 3})

print(df_bad[["Color", "Color_bad", "Size", "Size_bad"]])

print("\nWhy this is a problem:")
print("- 'Blue' (2) is not 'bigger' than 'Red' (1)")
print("- The distance between categories is not meaningful")
print("- A model may treat 3 as 'more' than 1, which is not what we mean")


# ----------------------------
# 3) The one-hot idea: turn each category into a simple yes/no indicator
# ----------------------------
banner("3) The one-hot idea: ask simple yes/no questions (indicators)")

print("One-hot encoding creates columns like:")
print("- Is Color == Blue?")
print("- Is Color == Green?")
print("- Is Color == Red?")
print("\nEach row gets 1 for 'yes' and 0 for 'no'.")


# First: pandas convenience method (very readable)
color_dummies = pd.get_dummies(df["Color"], prefix="Color")
size_dummies = pd.get_dummies(df["Size"], prefix="Size")

print("\nPandas get_dummies for Color:\n", color_dummies)
print("\nPandas get_dummies for Size:\n", size_dummies)

df_onehot_pandas = pd.concat([df, color_dummies, size_dummies], axis=1)
print("\nCombined dataset (original + one-hot columns):\n", df_onehot_pandas)

print("\nInterpretation:")
print("- No category is 'larger' than another")
print("- Categories are represented as independent indicators")
print("- This is a representation that many models can use")


# ----------------------------
# 4) Doing the same idea with sklearn (what you'll use inside pipelines)
# ----------------------------
banner("4) One-hot encoding with sklearn (pipeline-friendly)")

# Use sparse_output=False so the result is a regular dense array for easy viewing.
# (For large real datasets, sklearn may prefer sparse matrices for efficiency.)
encoder = OneHotEncoder(sparse_output=False)

# sklearn expects a 2D input for features (DataFrame with columns)
X_cat = df[["Color", "Size"]]
encoded = encoder.fit_transform(X_cat)

encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(["Color", "Size"]))

print("Categorical input features:\n", X_cat)
print("\nEncoded output (sklearn):\n", encoded_df)

print("\nWhat was 'learned' during fit?")
print("- The set of categories in each column (e.g., Color has {Blue, Green, Red})")
print("- The order of the output columns")


# ----------------------------
# 5) What happens with NEW categories? (handle_unknown)
# ----------------------------
banner("5) New categories at prediction time (why handle_unknown matters)")

new_df = pd.DataFrame({"Color": ["Red", "Yellow"], "Size": ["M", "XL"]})
print("New data (includes categories not seen in training):\n", new_df)

print("\nAttempting transform with the original encoder (no handle_unknown):")
try:
    encoder.transform(new_df)
except Exception as e:
    print("Transform failed (as expected). Error message:\n", e)

print("\nNow we use handle_unknown='ignore':")
encoder_safe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoder_safe.fit(X_cat)

encoded_new = encoder_safe.transform(new_df)
encoded_new_df = pd.DataFrame(encoded_new, columns=encoder_safe.get_feature_names_out(["Color", "Size"]))

print("Encoded new data (unknown categories become all-zeros for that feature group):\n", encoded_new_df)

print("\nKey idea:")
print("- A model can only handle categories it has a numeric representation for.")
print("- handle_unknown='ignore' avoids crashing by mapping unseen categories to 'no known category'.")


# ----------------------------
# 6) Bridge back to pipelines (conceptual, not code-heavy)
# ----------------------------
banner("6) Bridge back to pipelines (the reason we automate this)")

print("Takeaway:")
print("- One-hot encoding is a translation step: categories -> numeric indicators")
print("- In real workflows, we combine this with imputing missing values")
print("- Pipelines are used to ensure:")
print("  (1) preprocessing is learned ONLY from training data")
print("  (2) the same preprocessing is applied consistently to test/validation data")
print("  (3) we avoid data leakage and reduce mistakes")


if __name__ == "__main__":
    # This file is meant to be run top-to-bottom.
    # Keeping an empty main block so students see the convention without needing it.
    pass
