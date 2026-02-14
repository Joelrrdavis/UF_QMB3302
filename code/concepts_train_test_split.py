import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# -------------------------------------------------
# Create simple data with noise
# -------------------------------------------------
# Perfect prediction is impossible on purpose
rng = np.random.RandomState(1955)

X = rng.uniform(0, 10, size=(100, 1))
y = 3 * X[:, 0] + rng.normal(0, 5, size=100)

# -------------------------------------------------
# FIRST split: one particular judgment
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

model = LinearRegression()
model.fit(X_train, y_train)

# Error on data the model learned from (too optimistic!)
train_mse = mean_squared_error(y_train, model.predict(X_train))

# Error on data the model never saw (a more honest judgment?)
test_mse = mean_squared_error(y_test, model.predict(X_test))

# Baseline: always predict the mean of training targets
baseline_pred = np.full_like(y_test, y_train.mean(), dtype=float)
baseline_mse = mean_squared_error(y_test, baseline_pred)

print("\nFIRST SPLIT (random_state = 1)")
print("-----------------------------")
print("Training MSE:", round(train_mse, 1))
print("Test MSE:    ", round(test_mse, 1))
print("Baseline MSE:", round(baseline_mse, 1))

# -------------------------------------------------
# SECOND split: same data, same model, different judgment
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=99
)

model.fit(X_train, y_train)

train_mse = mean_squared_error(y_train, model.predict(X_train))
test_mse = mean_squared_error(y_test, model.predict(X_test))

baseline_pred = np.full_like(y_test, y_train.mean(), dtype=float)
baseline_mse = mean_squared_error(y_test, baseline_pred)

print("\nSECOND SPLIT (random_state = 99)")
print("-------------------------------")
print("Training MSE:", round(train_mse, 1))
print("Test MSE:    ", round(test_mse, 1))
print("Baseline MSE:", round(baseline_mse, 1))

print("\nCheck in:")
print("- Training error reflects learning, not judgment.")
print("- Test error estimates performance on unseen data.")
print("- Changing the split changes the judgment.")
print("- Cross-validation exists because one split might not be enough.")

