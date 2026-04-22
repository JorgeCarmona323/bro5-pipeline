#!/usr/bin/env python3
"""
tune_viz2d.py — 2D UMAP Visualization Parameter Sweep
=======================================================

Separate from clustering tuning. This sweep optimizes UMAP 2D layout quality
for figure presentation across all three descriptor branches.

Objective: connected, continuous topology — minimize fragmented islands while
preserving meaningful separation between hit and literature regions.
Cluster labels from analyse_chemical_space.py are NOT changed. Only the
2D x/y coordinates used for visualization are updated.

Branches
  2D physicochemical  — euclidean,         6 features (scaled)
  Mordred             — cosine,           335 features (scaled)
  MAPchiral           — minhash_distance, 2048 features

Sweep grid (12 combinations per branch)
  n_neighbors : [20, 30, 40]
  min_dist    : [0.1, 0.2, 0.3, 0.4]

Visualization quality metrics (all normalised min-max, higher = better)
  trustworthiness   40%  — nn in embedding are also nn in original space
  continuity        30%  — nn in original space also appear in embedding
  connected_score   30%  — fraction of points in the largest 2D kNN component
                           penalises fragmented, island-heavy layouts

Outputs
  outputs/tuning/{RUN_TAG}/viz2d_sweep_results.csv
  outputs/tuning/{RUN_TAG}/viz2d_top_candidates.csv
  outputs/tuning/{RUN_TAG}/figures/viz2d_{branch}_grid.svg     — all 12 combos
  outputs/tuning/{RUN_TAG}/figures/viz2d_{branch}_best.svg     — winner annotated
  outputs/analysis/{RUN_TAG}/aligned_metadata.csv              — updated in-place
"""

import time
import warnings
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from numba import njit
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.manifold import trustworthiness as sklearn_trustworthiness
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import umap

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ===========================================================================
# CONFIG
# ===========================================================================

_REPO_ROOT = Path(__file__).resolve().parent.parent

RUN_TAG     = "2026-04-06"
OUTPUT_DIR  = _REPO_ROOT / "outputs" / "tuning"  / RUN_TAG
CACHE_DIR   = OUTPUT_DIR / "cache"
ANALYSIS_DIR= _REPO_ROOT / "outputs" / "analysis" / RUN_TAG
ALIGNED_CSV = ANALYSIS_DIR / "aligned_metadata.csv"

MORDRED_SCALED_CSV = _REPO_ROOT / "outputs" / "mordred"   / RUN_TAG / "mordred_filtered_scaled.csv"
MAPC_FPS_NPY       = _REPO_ROOT / "outputs" / "mapchiral" / RUN_TAG / "mapchiral_fingerprints.npy"
MAPC_META_CSV      = _REPO_ROOT / "outputs" / "mapchiral" / RUN_TAG / "mapchiral_metadata.csv"
INPUT_2D_CSV       = (_REPO_ROOT / "data" / "libraries" / "2026-01-29"
                      / "canonicalized_master_macrocycles_2D_Descriptors_FINAL_20260129.csv")

SMILES_COL = "Smiles"
SOURCE_COL = "Source"
DESC_2D_COLS = [
    "Total Molweight", "cLogP", "H-Acceptors",
    "H-Donors", "Polar Surface Area", "Rotatable Bonds",
]
IQR_MULTIPLIER_2D   = 1.5
CONDITION_D_SOURCES = {"Literature", "34_Hits", "Hit"}

RANDOM_STATE   = 42
N_TRUST_SAMPLE = 2000
N_KNN_CONNECT  = 15    # kNN graph k for connected-component score

SWEEP_N_NEIGHBORS = [20, 30, 40]
SWEEP_MIN_DIST    = [0.1, 0.2, 0.3, 0.4]

# Composite weights
W_TRUST   = 0.40
W_CONT    = 0.30
W_CONNECT = 0.30

# Figure: source color scheme
DRAW_ORDER   = ["Literature", "34_Hits", "Hit"]
SOURCE_STYLE = {
    "Literature": ("#D0D0D0", 3, 0.25),
    "34_Hits":    ("#FF00FF", 25, 0.90),
    "Hit":        ("#FFD700", 40, 1.00),
}
CLUSTER_CMAP = plt.get_cmap("tab20")


# ===========================================================================
# MinHash distance
# ===========================================================================

@njit
def minhash_distance(fp_a: np.ndarray, fp_b: np.ndarray) -> float:
    m = fp_a.shape[0]
    eq = 0
    for k in range(m):
        if fp_a[k] == fp_b[k]:
            eq += 1
    return 1.0 - (eq / m)


# ===========================================================================
# Helpers
# ===========================================================================

def _elapsed(t0): return f"{time.time() - t0:.1f}s"

def _md_str(md): return str(md).replace(".", "p")

def _cache_path(branch, nn, md):
    return CACHE_DIR / f"{branch}_viz2d_nc2_nn{nn}_md{_md_str(md)}_emb.npy"

def _minmax_norm(arr):
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if hi == lo:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)

def iqr_clip(df, multiplier):
    q1, q3 = df.quantile(0.25), df.quantile(0.75)
    return df.clip(lower=q1 - multiplier * (q3 - q1),
                   upper=q3 + multiplier * (q3 - q1), axis=1)


# ===========================================================================
# Data loading
# ===========================================================================

def load_2d():
    print(f"\n[Load] 2D physicochemical")
    df   = pd.read_csv(INPUT_2D_CSV)
    df   = df[df[SOURCE_COL].isin(CONDITION_D_SOURCES)].reset_index(drop=True)
    df   = df.dropna(subset=DESC_2D_COLS).reset_index(drop=True)
    meta = df[[SMILES_COL, SOURCE_COL]].copy()
    X    = iqr_clip(df[DESC_2D_COLS].astype(float), IQR_MULTIPLIER_2D)
    X_sc = pd.DataFrame(StandardScaler().fit_transform(X.values),
                        columns=DESC_2D_COLS, index=meta.index)
    print(f"   {X_sc.shape[0]:,} × {X_sc.shape[1]}")
    return meta, X_sc


def load_mordred():
    print(f"\n[Load] Mordred")
    df   = pd.read_csv(MORDRED_SCALED_CSV, low_memory=False)
    meta = df[[c for c in [SMILES_COL, SOURCE_COL] if c in df.columns]].copy()
    X    = df[df.select_dtypes(include="number").columns].astype(float)
    print(f"   {X.shape[0]:,} × {X.shape[1]}")
    return meta, X


def load_mapchiral():
    print(f"\n[Load] MAPchiral")
    fps  = np.load(MAPC_FPS_NPY)
    meta = pd.read_csv(MAPC_META_CSV)
    print(f"   {fps.shape[0]:,} × {fps.shape[1]}")
    return meta, fps


def _branch_metric(branch):
    if branch == "mapchiral": return minhash_distance
    if branch == "2d":        return "euclidean"
    return "cosine"


# ===========================================================================
# UMAP (with cache)
# ===========================================================================

def get_embedding(X, branch, nn, md, idx, total):
    path = _cache_path(branch, nn, md)
    if path.exists():
        emb = np.load(path)
        print(f"   [{idx:>2}/{total}] nn={nn} md={md}  → (cache)")
        return emb
    t0     = time.time()
    metric = _branch_metric(branch)
    params = dict(n_components=2, n_neighbors=nn, min_dist=md,
                  random_state=RANDOM_STATE, metric=metric)
    emb = umap.UMAP(**params).fit_transform(
        X if isinstance(X, np.ndarray) else X.values
    )
    np.save(path, emb)
    print(f"   [{idx:>2}/{total}] nn={nn} md={md}  → {_elapsed(t0)}")
    return emb


# ===========================================================================
# Visualization quality metrics
# ===========================================================================

def compute_trustworthiness(X, emb, branch):
    rng = np.random.default_rng(RANDOM_STATE)
    n   = len(emb)
    idx = rng.choice(n, min(N_TRUST_SAMPLE, n), replace=False)
    emb_s = emb[idx]
    X_s   = X[idx] if isinstance(X, np.ndarray) else X.iloc[idx].values
    metric = _branch_metric(branch)
    try:
        D = pairwise_distances(X_s, metric=metric)
        return float(sklearn_trustworthiness(D, emb_s, n_neighbors=10,
                                             metric="precomputed"))
    except Exception:
        return float("nan")


def compute_continuity(X, emb, branch):
    """Continuity: how well original nns appear in the embedding."""
    rng = np.random.default_rng(RANDOM_STATE)
    n   = len(emb)
    idx = rng.choice(n, min(N_TRUST_SAMPLE, n), replace=False)
    emb_s = emb[idx]
    X_s   = X[idx] if isinstance(X, np.ndarray) else X.iloc[idx].values
    metric = _branch_metric(branch)
    try:
        D = pairwise_distances(X_s, metric=metric)
        # Swap X and emb so we measure the other direction
        D_emb = pairwise_distances(emb_s, metric="euclidean")
        return float(sklearn_trustworthiness(D_emb, X_s if metric == "euclidean"
                     else D, n_neighbors=10, metric="precomputed"))
    except Exception:
        return float("nan")


def compute_connected_score(emb, k=N_KNN_CONNECT):
    """Fraction of points in the largest connected component of the 2D kNN graph."""
    n   = len(emb)
    nn  = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(emb)
    idx = nn.kneighbors(emb, return_distance=False)[:, 1:]   # exclude self
    rows = np.repeat(np.arange(n), k)
    cols = idx.ravel()
    adj  = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    adj  = adj + adj.T   # make symmetric
    n_components, labels = connected_components(adj, directed=False)
    counts = np.bincount(labels)
    return float(counts.max() / n)


# ===========================================================================
# Sweep
# ===========================================================================

def run_branch(branch, X, meta):
    combos = list(product(SWEEP_N_NEIGHBORS, SWEEP_MIN_DIST))
    total  = len(combos)
    print(f"\n{'='*60}")
    print(f"  Viz sweep: {branch.upper()}  ({total} combos)")
    print(f"{'='*60}")

    rows = []
    for i, (nn, md) in enumerate(combos, 1):
        emb = get_embedding(X, branch, nn, md, i, total)

        t0    = time.time()
        trust = compute_trustworthiness(X, emb, branch)
        cont  = compute_continuity(X, emb, branch)
        conn  = compute_connected_score(emb)
        print(f"         trust={trust:.3f}  cont={cont:.3f}  "
              f"connected={conn:.3f}  metrics: {_elapsed(t0)}")

        rows.append({"branch": branch, "n_neighbors": nn, "min_dist": md,
                     "trustworthiness": round(trust, 4),
                     "continuity": round(cont, 4),
                     "connected_score": round(conn, 4),
                     "_emb": emb})

    # Score
    trust_n = _minmax_norm(np.array([r["trustworthiness"] for r in rows]))
    cont_n  = _minmax_norm(np.array([r["continuity"]      for r in rows]))
    conn_n  = _minmax_norm(np.array([r["connected_score"] for r in rows]))

    for i, r in enumerate(rows):
        r["composite_score"] = round(
            W_TRUST   * (trust_n[i] if not np.isnan(trust_n[i]) else 0)
          + W_CONT    * (cont_n[i]  if not np.isnan(cont_n[i])  else 0)
          + W_CONNECT * conn_n[i], 4)

    rows.sort(key=lambda r: r["composite_score"], reverse=True)
    for rank, r in enumerate(rows, 1):
        r["rank"] = rank

    return rows


# ===========================================================================
# Figures
# ===========================================================================

def _load_cluster_labels(branch):
    """Read current cluster labels from aligned_metadata.csv."""
    df = pd.read_csv(ALIGNED_CSV)
    col_map = {
        "2d":       "hdbscan_2d",
        "mordred":  "ward_mordred",
        "mapchiral":"kmedoids_mapchiral",
    }
    return df[col_map[branch]].values, df[SOURCE_COL].values


def _scatter_source(ax, emb, sources):
    handles = []
    for src in DRAW_ORDER:
        idx = np.where(np.array(sources) == src)[0]
        if len(idx) == 0:
            continue
        c, s, a = SOURCE_STYLE.get(src, ("#999999", 3, 0.3))
        ax.scatter(emb[idx, 0], emb[idx, 1], s=s, c=c, alpha=a,
                   linewidths=0, rasterized=True, zorder=3 if src != "Literature" else 1)
        handles.append(mpatches.Patch(color=c, label=f"{src} (n={len(idx):,})"))
    ax.legend(handles=handles, fontsize=6, loc="lower left", framealpha=0.8)


def _scatter_clusters(ax, emb, labels):
    for lbl in sorted(set(labels)):
        idx   = np.where(np.array(labels) == lbl)[0]
        color = "#CCCCCC" if lbl == -1 else CLUSTER_CMAP(lbl % 20)
        alpha = 0.15 if lbl == -1 else 0.45
        ax.scatter(emb[idx, 0], emb[idx, 1], s=3, c=[color],
                   alpha=alpha, linewidths=0, rasterized=True)


def generate_grid_figure(branch, rows, fig_dir):
    """3×4 grid showing all 12 combos colored by source."""
    nn_vals = sorted(set(r["n_neighbors"] for r in rows))
    md_vals = sorted(set(r["min_dist"]    for r in rows))
    n_nn, n_md = len(nn_vals), len(md_vals)

    cluster_labels, sources = _load_cluster_labels(branch)
    fig, axes = plt.subplots(n_nn, n_md, figsize=(4 * n_md, 3.5 * n_nn))
    fig.suptitle(f"{branch.upper()} — 2D Visualization Sweep  "
                 f"(source coloring, hits=magenta/gold)", fontsize=9)

    row_map = {nn: i for i, nn in enumerate(nn_vals)}
    col_map = {md: j for j, md in enumerate(md_vals)}

    for r in rows:
        ax  = axes[row_map[r["n_neighbors"]], col_map[r["min_dist"]]]
        emb = r["_emb"]
        _scatter_source(ax, emb, sources)
        star = "★ " if r["rank"] == 1 else ""
        ax.set_title(
            f"{star}nn={r['n_neighbors']} md={r['min_dist']}  "
            f"[rank {r['rank']}]\n"
            f"trust={r['trustworthiness']:.3f}  "
            f"conn={r['connected_score']:.3f}  "
            f"score={r['composite_score']:.3f}",
            fontsize=6)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("#F5F5F5")

    fig.tight_layout()
    path = fig_dir / f"viz2d_{branch}_grid.svg"
    fig.savefig(path, format="svg", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"   Grid figure: {path.name}")


def generate_best_figure(branch, best_row, fig_dir):
    """Side-by-side: source coloring + cluster coloring for the best combo."""
    cluster_labels, sources = _load_cluster_labels(branch)
    emb = best_row["_emb"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"{branch.upper()} — Best 2D Viz  "
        f"nn={best_row['n_neighbors']}  md={best_row['min_dist']}  "
        f"| trust={best_row['trustworthiness']:.3f}  "
        f"conn={best_row['connected_score']:.3f}  "
        f"score={best_row['composite_score']:.3f}",
        fontsize=9)

    ax = axes[0]
    ax.set_facecolor("#F5F5F5")
    _scatter_source(ax, emb, sources)
    ax.set_title("Source coloring", fontsize=9)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.grid(True, alpha=0.2, linewidth=0.5)

    ax = axes[1]
    ax.set_facecolor("#F5F5F5")
    _scatter_clusters(ax, emb, cluster_labels)
    ax.set_title("Existing cluster labels", fontsize=9)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.grid(True, alpha=0.2, linewidth=0.5)

    fig.tight_layout()
    path = fig_dir / f"viz2d_{branch}_best.svg"
    fig.savefig(path, format="svg", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Best figure: {path.name}")


# ===========================================================================
# Update aligned_metadata and regenerate combined figures
# ===========================================================================

def update_aligned_metadata(best_per_branch):
    df = pd.read_csv(ALIGNED_CSV)

    col_map = {
        "2d":       ("umap1_2d",       "umap2_2d"),
        "mordred":  ("umap1_mordred",  "umap2_mordred"),
        "mapchiral":("umap1_mapchiral","umap2_mapchiral"),
    }

    for branch, row in best_per_branch.items():
        emb = row["_emb"]
        c1, c2 = col_map[branch]
        df[c1] = emb[:, 0]
        df[c2] = emb[:, 1]
        print(f"   Updated {c1}, {c2}  (nn={row['n_neighbors']} md={row['min_dist']})")

    df.to_csv(ALIGNED_CSV, index=False)
    print(f"   Saved: {ALIGNED_CSV.name}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = OUTPUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    combos_per = len(SWEEP_N_NEIGHBORS) * len(SWEEP_MIN_DIST)
    print("=" * 60)
    print("2D UMAP VISUALIZATION SWEEP")
    print(f"  Branches     : 2D (euclidean), Mordred (cosine), MAPchiral (minhash)")
    print(f"  Combos/branch: {combos_per}  ({combos_per * 3} total)")
    print(f"  n_neighbors  : {SWEEP_N_NEIGHBORS}")
    print(f"  min_dist     : {SWEEP_MIN_DIST}")
    print(f"  Metrics      : trustworthiness ({W_TRUST:.0%})  "
          f"continuity ({W_CONT:.0%})  connected ({W_CONNECT:.0%})")
    print("=" * 60)

    t_total = time.time()

    branch_loaders = [
        ("2d",        load_2d),
        ("mordred",   load_mordred),
        ("mapchiral", load_mapchiral),
    ]

    all_rows = []
    best_per_branch = {}

    for branch, load_fn in branch_loaders:
        meta, X = load_fn()
        rows     = run_branch(branch, X, meta)
        all_rows.extend(rows)
        best_per_branch[branch] = rows[0]

        print(f"\n  [Figures] {branch} ...")
        generate_grid_figure(branch, rows, fig_dir)
        generate_best_figure(branch, rows[0], fig_dir)

    # Save results CSV (drop internal _emb column)
    csv_rows = [{k: v for k, v in r.items() if k != "_emb"} for r in all_rows]
    results_df = pd.DataFrame(csv_rows).sort_values(
        ["branch", "rank"]).reset_index(drop=True)
    results_df.to_csv(OUTPUT_DIR / "viz2d_sweep_results.csv", index=False)

    top_rows = [r for r in all_rows if r["rank"] == 1]
    top_csv  = [{k: v for k, v in r.items() if k != "_emb"} for r in top_rows]
    pd.DataFrame(top_csv).to_csv(OUTPUT_DIR / "viz2d_top_candidates.csv", index=False)

    # Print summary
    print(f"\n{'='*60}")
    print("BEST PARAMETERS PER BRANCH")
    print(f"{'='*60}")
    print(f"  {'Branch':<12} {'nn':>4} {'md':>5}  {'trust':>6}  {'cont':>6}  {'conn':>6}  {'score':>6}")
    print(f"  {'-'*52}")
    for branch, row in best_per_branch.items():
        print(f"  {branch:<12} {row['n_neighbors']:>4} {row['min_dist']:>5}  "
              f"{row['trustworthiness']:>6.3f}  {row['continuity']:>6.3f}  "
              f"{row['connected_score']:>6.3f}  {row['composite_score']:>6.3f}")

    # Update aligned_metadata.csv with best coords
    print(f"\n[Update] Writing best 2D coords to aligned_metadata.csv ...")
    update_aligned_metadata(best_per_branch)

    # Regenerate combined figures
    print(f"\n[Figures] Regenerating combined figures ...")
    import subprocess
    result = subprocess.run(
        ["python3", str(Path(__file__).parent / "make_combined_figures.py")],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("ERROR:", result.stderr)

    print(f"\n[✓] Done — {_elapsed(t_total)}")
    print(f"    Grid figures : {fig_dir}/viz2d_{{branch}}_grid.svg")
    print(f"    Best figures : {fig_dir}/viz2d_{{branch}}_best.svg")
    print(f"    Combined figs: {ANALYSIS_DIR}/figures/")


if __name__ == "__main__":
    main()
