#!/usr/bin/env python3
"""
Week 6B Walkthrough (Step-by-step): Unsupervised Learning with RFM Segmentation

As before with other scripts, this is written as a LINEAR walkthrough (minimal use of functions),
this is a little clumsier than you would normally see in production, but easier to consume for first time users.

Expected dataset:
- File: rfmfile.csv (in Canvas files)


What you’ll do:
1) Load + sanity-check RFM data
2) Why scaling matters (distance-based methods)
3) K-Means:
   - Elbow plot (inertia)
   - Silhouette score vs k
   - Fit final model and interpret segments (cluster profiles)
4) Hierarchical clustering (Agglomerative / Ward) as a contrast
5) (Optional) DBSCAN as a contrast: density + noise points
6) (Optional) Isolation Forest anomaly detection (behavioral outliers)

Outputs: (take note... this part might be new!)
- Figures + CSV tables saved to ./outputs/

To run the whole file in the terminal:
  python week06b_main.py
or (uv):
  uv run python week06b_main.py

  (sometimes you need to add "3" after python. Like "python3")
"""

2+2
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest


# -----------------------------------------------------------------------------
# 0) Setup
# -----------------------------------------------------------------------------
DATA_PATH = Path("rfmfile.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 88)
print("Week 6B — Unsupervised Learning Walkthrough (RFM: Recency–Frequency–Monetary)")
print("=" * 88)


# -----------------------------------------------------------------------------
# 1) Load data + sanity checks
# -----------------------------------------------------------------------------
print("\n" + "-" * 88)
print("1) Load the RFM dataset")
print("-" * 88)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Could not find {DATA_PATH.resolve()}\n"
        "Put rfmfile.csv in the same folder as this script."
    )

df = pd.read_csv(DATA_PATH)
print(f"Loaded: {DATA_PATH}  |  shape = {df.shape}")
print("Columns:", list(df.columns))

# Expect these columns exactly (simple + explicit for teaching)
RFM_COLS = ["recency", "frequency", "monetary_value"]
missing = [c for c in RFM_COLS if c not in df.columns]
if missing:
    raise ValueError(
        f"Missing expected columns: {missing}\n"
        f"Found columns: {list(df.columns)}\n"
        "Rename your columns or edit RFM_COLS at the top of this script."
    )

# Basic preview
print("\nHead (first 5 rows):")
print(df[RFM_COLS].head())

# Convert to numeric just in case CSV stored numbers as strings
df[RFM_COLS] = df[RFM_COLS].apply(pd.to_numeric, errors="coerce")

# Drop rows with missing values in RFM
before = len(df)
df = df.dropna(subset=RFM_COLS).copy()
after = len(df)
if after < before:
    print(f"\n[WARN] Dropped {before - after} rows with missing RFM values. Remaining = {after}")

# Summary stats (this is a good place to pause and interpret typical ranges)
print("\nDescribe (RFM):")
print(df[RFM_COLS].describe().round(3))

# OPTIONAL: If your RFM values are extremely skewed, you could log-transform.
# For this walkthrough, we keep it simple and stick to scaling.
# df["frequency"] = np.log1p(df["frequency"])
# df["monetary_value"] = np.log1p(df["monetary_value"])


# -----------------------------------------------------------------------------
# 2) Build feature matrix + scaling (critical for distance-based clustering)
# -----------------------------------------------------------------------------
print("\n" + "-" * 88)
print("2) Build X matrix and SCALE it (distance needs fair units)")
print("-" * 88)

X = df[RFM_COLS].to_numpy(dtype=float)

print("Raw example row (unscaled):", X[0])

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

print("Scaled example row:", Xs[0])
print("Scaled feature means (≈0):", Xs.mean(axis=0).round(4))
print("Scaled feature stds   (≈1):", Xs.std(axis=0).round(4))

# Teaching note:
# - If you skip scaling, "monetary_value" often dominates the distance calculation.
# - K-Means, Hierarchical (Ward), and DBSCAN are all distance-based in practice.


# -----------------------------------------------------------------------------
# 3) K-Means: Elbow + Silhouette to choose k
# -----------------------------------------------------------------------------
print("\n" + "-" * 88)
print("3) K-Means: Choose k with Elbow + Silhouette")
print("-" * 88)

K_RANGE = range(2, 11)

inertias = []
silhouettes = []

for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = km.fit_predict(Xs)

    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(Xs, labels))

print("\nElbow data (k, inertia):")
for k, inertia in zip(K_RANGE, inertias):
    print(f"  k={k:2d}  inertia={inertia:,.2f}")

print("\nSilhouette data (k, silhouette):")
for k, sil in zip(K_RANGE, silhouettes):
    print(f"  k={k:2d}  silhouette={sil:.3f}")

# Plot: Elbow
plt.figure()
plt.plot(list(K_RANGE), inertias, marker="o")
plt.xlabel("k (number of clusters)")
plt.ylabel("Inertia (within-cluster sum of squares)")
plt.title("K-Means Elbow Plot")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "kmeans_elbow.png", dpi=180)
plt.close()
print(f"Saved: {OUTPUT_DIR / 'kmeans_elbow.png'}")

# Plot: Silhouette vs k
plt.figure()
plt.plot(list(K_RANGE), silhouettes, marker="o")
plt.xlabel("k (number of clusters)")
plt.ylabel("Silhouette score")
plt.title("K-Means Silhouette vs k")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "kmeans_silhouette.png", dpi=180)
plt.close()
print(f"Saved: {OUTPUT_DIR / 'kmeans_silhouette.png'}")

# Pick a final k for the walkthrough (you can change this live in class)
FINAL_K = 4
print(f"\nFor the walkthrough, we'll use FINAL_K = {FINAL_K}.")


# -----------------------------------------------------------------------------
# 4) Fit K-Means with FINAL_K and interpret clusters
# -----------------------------------------------------------------------------
print("\n" + "-" * 88)
print("4) Fit K-Means (final) + interpret segments using cluster profiles")
print("-" * 88)

kmeans_final = KMeans(n_clusters=FINAL_K, n_init=30, random_state=42)
df["cluster_kmeans"] = kmeans_final.fit_predict(Xs)

print("Cluster counts:")
print(df["cluster_kmeans"].value_counts().sort_index())

# Cluster profile table: mean + median + count
profile_km = (
    df.groupby("cluster_kmeans")[RFM_COLS]
      .agg(["mean", "median"])
      .round(3)
)
profile_km.columns = [f"{a}_{b}" for a, b in profile_km.columns]
profile_km["count"] = df.groupby("cluster_kmeans").size()
profile_km = profile_km.sort_index()

print("\nCluster profiles (K-Means):")
print(profile_km)

profile_km.to_csv(OUTPUT_DIR / "cluster_profiles_kmeans.csv")
print(f"Saved: {OUTPUT_DIR / 'cluster_profiles_kmeans.csv'}")

# Visualize in 2D using PCA (not for modeling—just for plotting)
pca = PCA(n_components=2, random_state=42)
Z = pca.fit_transform(Xs)

plt.figure()
plt.scatter(Z[:, 0], Z[:, 1], c=df["cluster_kmeans"].to_numpy(), s=18, alpha=0.85)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title(f"K-Means clusters (k={FINAL_K}) in PCA space")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "kmeans_clusters_pca.png", dpi=180)
plt.close()
print(f"Saved: {OUTPUT_DIR / 'kmeans_clusters_pca.png'}")


# -----------------------------------------------------------------------------
# 5) Hierarchical clustering (Agglomerative / Ward) as a contrast
# -----------------------------------------------------------------------------
print("\n" + "-" * 88)
print("5) Hierarchical clustering (Agglomerative / Ward) as a contrast")
print("-" * 88)

hier = AgglomerativeClustering(n_clusters=FINAL_K, linkage="ward")
df["cluster_hier"] = hier.fit_predict(Xs)

print("Cluster counts (hierarchical):")
print(df["cluster_hier"].value_counts().sort_index())

profile_h = (
    df.groupby("cluster_hier")[RFM_COLS]
      .agg(["mean", "median"])
      .round(3)
)
profile_h.columns = [f"{a}_{b}" for a, b in profile_h.columns]
profile_h["count"] = df.groupby("cluster_hier").size()
profile_h = profile_h.sort_index()

print("\nCluster profiles (Hierarchical):")
print(profile_h)

profile_h.to_csv(OUTPUT_DIR / "cluster_profiles_hierarchical.csv")
print(f"Saved: {OUTPUT_DIR / 'cluster_profiles_hierarchical.csv'}")

plt.figure()
plt.scatter(Z[:, 0], Z[:, 1], c=df["cluster_hier"].to_numpy(), s=18, alpha=0.85)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title(f"Hierarchical clusters (k={FINAL_K}, ward) in PCA space")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hierarchical_clusters_pca.png", dpi=180)
plt.close()
print(f"Saved: {OUTPUT_DIR / 'hierarchical_clusters_pca.png'}")


# -----------------------------------------------------------------------------
# 6) OPTIONAL: DBSCAN (density-based) as a contrast
# -----------------------------------------------------------------------------
DO_DBSCAN = True  # flip to False if you want to skip for time

if DO_DBSCAN:
    print("\n" + "-" * 88)
    print("6) (Optional) DBSCAN: density-based clustering + noise points")
    print("-" * 88)

    # Heuristic: k-distance plot to estimate eps
    MIN_SAMPLES = 10
    nbrs = NearestNeighbors(n_neighbors=MIN_SAMPLES).fit(Xs)
    distances, _ = nbrs.kneighbors(Xs)
    k_dist = np.sort(distances[:, -1])

    plt.figure()
    plt.plot(k_dist)
    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"Distance to {MIN_SAMPLES}th nearest neighbor")
    plt.title("DBSCAN eps heuristic: k-distance plot (look for the 'knee')")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dbscan_k_distance.png", dpi=180)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'dbscan_k_distance.png'}")

    # Simple eps selection for a classroom demo:
    # Use a high percentile of k-distances as a starting point.
    eps = float(np.percentile(k_dist, 92))
    print(f"Using eps ≈ 92nd percentile of k-distances: eps={eps:.3f} (adjust live if needed)")

    db = DBSCAN(eps=eps, min_samples=MIN_SAMPLES)
    df["cluster_dbscan"] = db.fit_predict(Xs)  # -1 = noise

    n_noise = int((df["cluster_dbscan"] == -1).sum())
    n_clusters = len(set(df["cluster_dbscan"])) - (1 if -1 in set(df["cluster_dbscan"]) else 0)
    print(f"DBSCAN clusters={n_clusters}, noise points={n_noise}")

    plt.figure()
    plt.scatter(Z[:, 0], Z[:, 1], c=df["cluster_dbscan"].to_numpy(), s=18, alpha=0.85)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(f"DBSCAN clusters (eps={eps:.3f}, min_samples={MIN_SAMPLES}) in PCA space (-1=noise)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dbscan_clusters_pca.png", dpi=180)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'dbscan_clusters_pca.png'}")

    # Profiles excluding noise
    non_noise = df[df["cluster_dbscan"] != -1].copy()
    if len(non_noise) > 0 and n_clusters > 0:
        prof_db = (
            non_noise.groupby("cluster_dbscan")[RFM_COLS]
                     .agg(["mean", "median"])
                     .round(3)
        )
        prof_db.columns = [f"{a}_{b}" for a, b in prof_db.columns]
        prof_db["count"] = non_noise.groupby("cluster_dbscan").size()
        prof_db = prof_db.sort_index()

        print("\nCluster profiles (DBSCAN, excluding noise):")
        print(prof_db)

        prof_db.to_csv(OUTPUT_DIR / "cluster_profiles_dbscan.csv")
        print(f"Saved: {OUTPUT_DIR / 'cluster_profiles_dbscan.csv'}")
    else:
        print("[WARN] DBSCAN produced no non-noise clusters to profile.")


# -----------------------------------------------------------------------------
# 7) OPTIONAL: Isolation Forest anomaly detection (behavioral outliers)
# -----------------------------------------------------------------------------
DO_IFOREST = True  # flip to False if you want to skip for time

if DO_IFOREST:
    print("\n" + "-" * 88)
    print("7) (Optional) Isolation Forest: anomaly detection on RFM behavior")
    print("-" * 88)

    contamination = 0.03  # % flagged as anomalies (tweak for your data)
    iso = IsolationForest(n_estimators=300, contamination=contamination, random_state=42)
    iso_pred = iso.fit_predict(Xs)          # -1 anomaly, +1 normal
    df["is_anomaly"] = (iso_pred == -1).astype(int)

    n_anom = int(df["is_anomaly"].sum())
    print(f"Flagged anomalies: {n_anom} / {len(df)}  (contamination={contamination})")

    # Plot anomalies in PCA space (X markers)
    normal_mask = df["is_anomaly"].to_numpy() == 0
    anom_mask = ~normal_mask

    plt.figure()
    plt.scatter(Z[normal_mask, 0], Z[normal_mask, 1], s=18, alpha=0.6)
    plt.scatter(Z[anom_mask, 0], Z[anom_mask, 1], s=30, alpha=0.95, marker="x")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("Isolation Forest anomalies in PCA space (X = flagged)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "iforest_anomalies_pca.png", dpi=180)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'iforest_anomalies_pca.png'}")

    # Show top anomalies by score (lower score = more anomalous)
    scores = iso.score_samples(Xs)
    df["iforest_score"] = scores

    top = df.sort_values("iforest_score").head(min(10, len(df)))
    print("\nTop anomalous customers (lowest Isolation Forest scores):")
    print(top[RFM_COLS + ["iforest_score", "is_anomaly"]].round(4))

    top[RFM_COLS + ["iforest_score", "is_anomaly"]].to_csv(
        OUTPUT_DIR / "top_anomalies_iforest.csv", index=False
    )
    print(f"Saved: {OUTPUT_DIR / 'top_anomalies_iforest.csv'}")


# -----------------------------------------------------------------------------
# 8) Closing takeaway (evaluation mindset)
# -----------------------------------------------------------------------------
print("\n" + "-" * 88)
print("8) Takeaway: Evaluating unsupervised learning")
print("-" * 88)

print(
    "Unsupervised learning has no single 'accuracy' score.\n"
    "Use multiple lenses:\n"
    "  1) Structure: do you see coherent, stable groups? (silhouette, stability checks)\n"
    "  2) Interpretation: do segments make business sense? (RFM cluster profiles)\n"
    "  3) Impact: can you act on them? (targeting, retention, risk monitoring)\n"
)

print("\nDone. Open the outputs/ folder for figures and tables.")
