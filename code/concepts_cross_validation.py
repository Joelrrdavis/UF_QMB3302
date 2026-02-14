# cross_validation_is_variability.py

import numpy as np

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score

# ---------------------------
# 1) Make a simple dataset
# ---------------------------
X, y = make_classification(
    n_samples=400,
    n_features=6,
    n_informative=3,
    n_redundant=1,
    n_clusters_per_class=2,
    flip_y=0.06,          # a little label noise (realistic)
    class_sep=1.0,
    random_state=1955
)

model = LogisticRegression(max_iter=2000)

# ---------------------------
# 2) A single train-test split (one "judge")
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1, stratify=y
)

model.fit(X_train, y_train)
single_split_score = model.score(X_test, y_test)

print("\nSINGLE SPLIT RESULT")
print("-------------------")
print(f"Accuracy (random_state=1): {single_split_score:.3f}")

# Show how a different split can change the story
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X, y, test_size=0.25, random_state=99, stratify=y
)

model.fit(X_train2, y_train2)
single_split_score_2 = model.score(X_test2, y_test2)

print(f"Accuracy (random_state=99): {single_split_score_2:.3f}")
print("Note: same data + same model, different split -> different score.\n")

# ---------------------------
# 3) Cross-validation (a "panel of judges")
# ---------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=1955)

cv_scores = cross_val_score(model, X, y, cv=kf, scoring="accuracy")

print("5-FOLD CROSS-VALIDATION RESULTS")
print("-------------------------------")
print("Fold accuracies:", "  ".join(f"{s:.3f}" for s in cv_scores))

print("\nSUMMARY (THIS IS THE POINT)")
print("---------------------------")
print(f"Mean accuracy: {cv_scores.mean():.3f}")
print(f"Std (variation across folds): {cv_scores.std():.3f}")
print(f"Min / Max fold accuracy: {cv_scores.min():.3f} / {cv_scores.max():.3f}")

print("\nInterpretation:")
print("- The single-split score is one outcome from one particular split.")
print("- Cross-validation shows the *range* and *stability* of performance across splits.")
