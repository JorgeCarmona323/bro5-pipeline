"""
sweep_min_cluster_size.py
=========================
Sweep HDBSCAN min_cluster_size over a range using the already-computed
nD UMAP embeddings (no UMAP recompute). For each value records:
  - n_clusters
  - noise_fraction
  - silhouette score  (sklearn, Euclidean on nD embedding)
  - DBCV             (hdbscan validity_index, patched for overflow)

Produces a combined plot and a CSV of results for each branch.

Usage:
    python 03_umap_analysis/sweep_min_cluster_size.py
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import hdbscan
from hdbscan import validity as hdbscan_validity
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
ANALYSIS_DIR  = Path("outputs/analysis/2026-04-06")
OUTPUT_DIR    = Path("outputs/tuning/2026-04-06")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MCS_RANGE     = list(range(50, 201, 10))   # 50 → 200 in steps of 10
MIN_SAMPLES   = 10                          # fixed throughout

ALIGNED_CSV   = ANALYSIS_DIR / "aligned_metadata.csv"

BRANCHES = {
    "2d": {
        "csv":     ALIGNED_CSV,
        "nd_cols": ["umap1_2d", "umap2_2d"],                  # 2D viz coords
        "color":   "#4CAF50",
    },
    "mordred": {
        "csv":     ANALYSIS_DIR / "mordred_umap.csv",
        "nd_cols": [f"UMAP_{i}_nd" for i in range(1, 11)],   # 10D
        "color":   "#2196F3",
    },
    "mapchiral": {
        "csv":     ANALYSIS_DIR / "mapchiral_umap.csv",
        "nd_cols": [f"UMAP_{i}_nd" for i in range(1, 6)],    # 5D
        "color":   "#E91E63",
    },
}

SILHOUETTE_SAMPLE = 3000   # subsample for silhouette speed (full 8k is slow)

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def run_hdbscan(X: np.ndarray, mcs: int) -> np.ndarray:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=mcs,
        min_samples=MIN_SAMPLES,
        metric="euclidean",
    )
    return clusterer.fit_predict(X)


def compute_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    n_clusters   = len(set(labels)) - (1 if -1 in labels else 0)
    noise_frac   = (labels == -1).mean()

    # silhouette — skip if fewer than 2 clusters or all noise
    if n_clusters >= 2 and noise_frac < 1.0:
        mask = labels != -1
        X_cl = X[mask]
        L_cl = labels[mask]
        if len(X_cl) > SILHOUETTE_SAMPLE:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X_cl), SILHOUETTE_SAMPLE, replace=False)
            sil = silhouette_score(X_cl[idx], L_cl[idx], metric="euclidean")
        else:
            sil = silhouette_score(X_cl, L_cl, metric="euclidean")
    else:
        sil = np.nan

    # DBCV — skip if fewer than 2 clusters
    if n_clusters >= 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                dbcv = hdbscan_validity.validity_index(
                    X.astype(np.float64), labels
                )
            except Exception:
                dbcv = np.nan
    else:
        dbcv = np.nan

    return {
        "n_clusters":  n_clusters,
        "noise_frac":  noise_frac,
        "silhouette":  sil,
        "dbcv":        dbcv,
    }


def find_optimal(df: pd.DataFrame) -> dict:
    """
    Pick the min_cluster_size that jointly optimises silhouette and DBCV.
    Strategy: min-max normalise both, sum them, pick the peak.
    Falls back to silhouette peak if DBCV is all-NaN.
    """
    out = {}

    sil = df["silhouette"].values.copy()
    dbcv = df["dbcv"].values.copy()

    def norm(v):
        lo, hi = np.nanmin(v), np.nanmax(v)
        if hi == lo:
            return np.zeros_like(v, dtype=float)
        return (v - lo) / (hi - lo)

    sil_n = norm(sil)
    dbcv_n = norm(dbcv)

    if not np.all(np.isnan(dbcv)):
        composite = (sil_n + dbcv_n) / 2
        label = "silhouette + DBCV (composite)"
    else:
        composite = sil_n
        label = "silhouette only (DBCV unavailable)"

    best_idx = np.nanargmax(composite)
    out["best_mcs"]   = int(df["min_cluster_size"].iloc[best_idx])
    out["best_score"] = float(composite[best_idx])
    out["method"]     = label
    out["composite"]  = composite
    return out


# ---------------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------------

def make_plot(df: pd.DataFrame, branch: str, color: str,
              optimal: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        f"{branch.upper()} — min_cluster_size sweep\n"
        f"Optimal: mcs={optimal['best_mcs']}  "
        f"({optimal['method']})",
        fontsize=11, fontweight="bold"
    )

    mcs = df["min_cluster_size"].values

    panels = [
        (axes[0, 0], "n_clusters",  "Cluster count",    False),
        (axes[0, 1], "noise_frac",  "Noise fraction",   False),
        (axes[1, 0], "silhouette",  "Silhouette score", True),
        (axes[1, 1], "dbcv",        "DBCV",             True),
    ]

    for ax, col, ylabel, mark_opt in panels:
        vals = df[col].values
        ax.plot(mcs, vals, color=color, lw=2, marker="o", ms=5)
        ax.set_xlabel("min_cluster_size")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.grid(True, alpha=0.3)

        if mark_opt:
            best_val = df.loc[df["min_cluster_size"] == optimal["best_mcs"], col].values
            if len(best_val) and not np.isnan(best_val[0]):
                ax.axvline(optimal["best_mcs"], color="red", lw=1.2,
                           linestyle="--", alpha=0.7)
                ax.scatter([optimal["best_mcs"]], [best_val[0]],
                           color="red", zorder=5, s=60)

        if col == "noise_frac":
            ax.axhline(0.05, color="grey", lw=0.8, linestyle=":")
            ax.axhline(0.40, color="grey", lw=0.8, linestyle=":")
            ax.text(mcs[-1], 0.05, " min", va="bottom", fontsize=7, color="grey")
            ax.text(mcs[-1], 0.40, " max", va="bottom", fontsize=7, color="grey")

        if col == "silhouette":
            for thresh, label in [(0.25, "weak"), (0.50, "ok"), (0.70, "good")]:
                ax.axhline(thresh, color="grey", lw=0.7, linestyle=":")
                ax.text(mcs[0], thresh, f" {label}", va="bottom",
                        fontsize=7, color="grey")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def sweep_branch(branch: str, cfg: dict) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"  Branch: {branch.upper()}")
    print(f"{'='*60}")

    df_raw = pd.read_csv(cfg["csv"])
    X = df_raw[cfg["nd_cols"]].values.astype(np.float64)
    print(f"  Loaded: {X.shape[0]:,} × {X.shape[1]}D embedding")

    rows = []
    for mcs in MCS_RANGE:
        t0     = time.time()
        labels = run_hdbscan(X, mcs)
        m      = compute_metrics(X, labels)
        elapsed = time.time() - t0
        print(
            f"  mcs={mcs:>4}  →  "
            f"clusters={m['n_clusters']:>3}  "
            f"noise={m['noise_frac']:.3f}  "
            f"sil={m['silhouette']:+.3f}  "
            f"dbcv={m['dbcv']:+.3f}  "
            f"({elapsed:.1f}s)"
        )
        rows.append({"min_cluster_size": mcs, **m})

    return pd.DataFrame(rows)


def main():
    print("\n" + "="*60)
    print("  min_cluster_size SWEEP")
    print(f"  Range: {MCS_RANGE[0]} → {MCS_RANGE[-1]}, step {MCS_RANGE[1]-MCS_RANGE[0]}")
    print(f"  Metrics: n_clusters, noise_frac, silhouette, DBCV")
    print("="*60)

    for branch, cfg in BRANCHES.items():
        df = sweep_branch(branch, cfg)

        # Save CSV
        csv_path = OUTPUT_DIR / f"{branch}_mcs_sweep.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path.name}")

        # Find optimal
        opt = find_optimal(df)
        print(f"\n  OPTIMAL  mcs={opt['best_mcs']}  "
              f"(method: {opt['method']})")

        # Plot
        plot_path = OUTPUT_DIR / f"figures/{branch}_mcs_sweep.svg"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        make_plot(df, branch, cfg["color"], opt, plot_path)

        # Print summary table
        print(f"\n  {'mcs':>5}  {'clusters':>8}  {'noise':>7}  "
              f"{'sil':>7}  {'dbcv':>7}")
        print("  " + "-"*45)
        for _, row in df.iterrows():
            marker = " ←" if row["min_cluster_size"] == opt["best_mcs"] else ""
            print(
                f"  {int(row['min_cluster_size']):>5}  "
                f"{int(row['n_clusters']):>8}  "
                f"{row['noise_frac']:>7.3f}  "
                f"{row['silhouette']:>7.3f}  "
                f"{row['dbcv']:>7.3f}"
                f"{marker}"
            )

    print("\n[✓] Sweep complete.")


if __name__ == "__main__":
    main()
