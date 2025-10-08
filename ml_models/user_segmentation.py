# =========================================================
# USER SEGMENTATION (pure NumPy Mini-Batch KMeans)
# =========================================================

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# -----------------------------
# Config
# -----------------------------
PG_URL = "postgresql+psycopg2://admin:admin@localhost:5433/ads_db"
TABLE_IN = "mrt_user_features"
TABLE_OUT_USERS = "user_segments"
TABLE_OUT_SUMMARY = "user_cluster_summary"

N_CLUSTERS = 4
BATCH_SIZE = 512
N_ITERS = 80
RANDOM_SEED = 42

# -----------------------------
# Helpers
# -----------------------------
def standardize_np(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std, mean, std

def kmeanspp_init(X, k, rng):
    """k-means++ initializer."""
    n, d = X.shape
    centers = np.empty((k, d), dtype=X.dtype)
    # pick first center
    idx = rng.integers(0, n)
    centers[0] = X[idx]
    # pick remaining
    closest_sq = np.full(n, np.inf)
    for c in range(1, k):
        # update distances
        dist_sq = np.sum((X - centers[c-1])**2, axis=1)
        closest_sq = np.minimum(closest_sq, dist_sq)
        probs = closest_sq / closest_sq.sum()
        idx = rng.choice(n, p=probs)
        centers[c] = X[idx]
    return centers

def minibatch_kmeans(X, k, batch_size, n_iters, rng):
    """Streaming mini-batch k-means with per-center learning rate."""
    n, d = X.shape
    centers = kmeanspp_init(X, k, rng)
    counts = np.zeros(k, dtype=np.int64)

    for it in range(n_iters):
        # sample a batch
        batch_idx = rng.integers(0, n, size=min(batch_size, n))
        B = X[batch_idx]

        # assign to nearest centers
        # (B - centers_j)^2 -> compute using broadcasting
        # distances shape: (batch, k)
        dists = np.sum((B[:, None, :] - centers[None, :, :])**2, axis=2)
        labels = np.argmin(dists, axis=1)

        # update centers with learning-rate 1/(count+1)
        for j in range(k):
            mask = (labels == j)
            if not np.any(mask):
                continue
            pts = B[mask]
            for x in pts:
                counts[j] += 1
                eta = 1.0 / counts[j]
                centers[j] = (1 - eta) * centers[j] + eta * x

    # final full assignment
    full_dists = np.sum((X[:, None, :] - centers[None, :, :])**2, axis=2)
    final_labels = np.argmin(full_dists, axis=1)
    inertia = np.sum(np.min(full_dists, axis=1))
    return centers, final_labels, inertia

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(RANDOM_SEED)

    engine = create_engine(PG_URL)
    df = pd.read_sql(f"SELECT * FROM {TABLE_IN};", engine)
    print(f"✅ Loaded {len(df)} user records for clustering")

    # features for clustering
    feat_cols = [
        "impressions",
        "clicks",
        "conversions",
        "total_revenue",
        "avg_ctr_score",
        "days_since_signup",
    ]
    X = df[feat_cols].fillna(0).to_numpy(dtype=np.float64)

    # standardize (NumPy)
    Xs, mu, sigma = standardize_np(X)

    # run mini-batch k-means
    centers, labels, inertia = minibatch_kmeans(
        Xs, N_CLUSTERS, BATCH_SIZE, N_ITERS, rng
    )
    df["user_cluster"] = labels.astype(int)
    print(f"🎯 Clustering done. Inertia: {inertia:,.2f}")

    # cluster summary
    summary = (
        df.groupby("user_cluster")[feat_cols]
          .mean()
          .round(2)
          .reset_index()
    )

    # write back to Postgres
    cols_keep = ["user_id", "region", "device", "subscription_tier"] + feat_cols + ["user_cluster"]
    (df[cols_keep]).to_sql(TABLE_OUT_USERS, engine, if_exists="replace", index=False)
    summary.to_sql(TABLE_OUT_SUMMARY, engine, if_exists="replace", index=False)

    print(f"🚀 Wrote user segments → {TABLE_OUT_USERS}")
    print(f"🚀 Wrote cluster summary → {TABLE_OUT_SUMMARY}")
    print("✅ Done.")