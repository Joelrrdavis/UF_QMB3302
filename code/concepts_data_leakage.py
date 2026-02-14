import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Simple data with noise
rng = np.random.RandomState(0)
X = rng.normal(size=(100, 1))
y = 5 * X[:, 0] + rng.normal(0, 1, size=100)

# ---- LEAKAGE HAPPENS HERE ----
# We "peek" at the full dataset to compute a statistic
global_mean = X.mean()
X_leaky = X - global_mean  # uses information from future test data

# Now we split (too late)
X_train, X_test, y_train, y_test = train_test_split(
    X_leaky, y, test_size=0.3, random_state=0
)

model = LinearRegression()
model.fit(X_train, y_train)

leaky_test_error = mean_squared_error(y_test, model.predict(X_test))
print("Test error with leakage:", round(leaky_test_error, 3))

# Correct approach: compute statistics using training data only
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

train_mean = X_train_raw.mean()
X_train_clean = X_train_raw - train_mean
X_test_clean = X_test_raw - train_mean

model.fit(X_train_clean, y_train)
clean_test_error = mean_squared_error(y_test, model.predict(X_test_clean))

print("Test error without leakage:", round(clean_test_error, 3))
