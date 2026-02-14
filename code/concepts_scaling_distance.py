import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

# -------------------------------------------------
# Three people described by two features
# -------------------------------------------------
# age    -> measured in years (small scale)
# income -> measured in dollars (large scale)
X = np.array([
    [25,  40000],
    [30,  42000],
    [40, 120000]
])

print("\nRaw feature values (age, income):")
print(X)

# -------------------------------------------------
# Distances when features are NOT scaled
# -------------------------------------------------
# Income dominates because its numbers are much larger
raw_distances = pairwise_distances(X)

print("\nDistances without scaling:")
print(raw_distances.round(1))

# -------------------------------------------------
# Scale features so 'one unit' means 'one standard deviation'
# -------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nScaled feature values:")
print(X_scaled.round(2))

# -------------------------------------------------
# Distances AFTER scaling
# -------------------------------------------------
# Age and income now contribute comparably
scaled_distances = pairwise_distances(X_scaled)

print("\nDistances with scaling:")
print(scaled_distances.round(2))

print("\nKey takeaway:")
print("- Distance-based models assume features are comparable.")
print("- Without scaling, large-unit features dominate closeness.")
print("- Scaling changes *geometry*, not data meaning.")
