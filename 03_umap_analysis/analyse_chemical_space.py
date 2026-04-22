"""
analyse_chemical_space.py
Chemical Space Analysis — UMAP + HDBSCAN + Ward / K-Medoids
============================================================

Overview
--------
Loads pre-computed feature matrices from three branches and runs a unified
analysis pipeline:

  1. UMAP (2D) — for visualization only, all branches
  2. Dimensionality sweep (Mordred + MAPchiral only):
       - UMAP at n_components = [5, 10, 15]
       - HDBSCAN at each dimensionality
       - Select smallest n_components where cluster structure is stable
  3. Final UMAP (nD, selected dimensionality) — Mordred + MAPchiral
  4. HDBSCAN on nD embeddings (all branches)
  5. Secondary clustering per branch:
       - 2D:        HDBSCAN only (6-feature panel, no secondary)
       - Mordred:   Ward hierarchical (k selected by silhouette sweep)
       - MAPchiral: K-Medoids k=15 (confirmed by Gap Statistic, prior run)

Why separate 2D from clustering dimensionality
----------------------------------------------
2D UMAP is a strong nonlinear compression and is excellent for visualization,
but the embedding may merge or split clusters that are distinct in higher
dimensions. Clustering on a 5–15 dimensional embedding better preserves local
structure before HDBSCAN density estimation.

Branches
--------
  2D descriptors  — loaded from master CSV, preprocessed inline
                    IQR clip → StandardScaler → UMAP (euclidean)
                    No dimensionality sweep: 8 input features → 2D is reasonable
  Mordred         — from compute_mordred_descriptors.py (already scaled)
                    UMAP metric: cosine
                    Dimensionality sweep: yes
  MAPchiral       — from compute_mapchiral_fingerprints.py (uint32 fps)
                    UMAP metric: minhash_distance
                    Dimensionality sweep: yes

Stability criterion for dimensionality selection
-------------------------------------------------
Given sweep results at [n1, n2, n3]:
  For each consecutive pair (n_k, n_{k+1}):
    stable if:
      |n_clusters(n_k) - n_clusters(n_{k+1})| <= CLUSTER_DELTA
      AND |noise_frac(n_k) - noise_frac(n_{k+1})| <= NOISE_DELTA
  Select the smallest n_k where the pair (n_k, n_{k+1}) is stable.
  If no pair is stable, fall back to N_COMPONENTS_DEFAULT.

Clustering note
---------------
HDBSCAN and K-Medoids run on the nD UMAP embeddings (Euclidean distance).
Clustering in embedding space avoids O(n²) pairwise MinHash distances and
keeps all branches comparable. 2D UMAP coordinates are stored separately
and used only for figures.

Outputs
-------
  aligned_metadata.csv
  {branch}_umap_2d.csv                  2D coords for visualization
  {branch}_umap_nd.csv                  nD coords used for clustering
  {branch}_sweep_report.txt             dimensionality sweep summary
  figures/
    {branch}_umap_source.svg
    {branch}_umap_hdbscan.svg
    {branch}_umap_kmedoids.svg
  analysis_report.txt
"""

import warnings
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from numba import njit
import hdbscan as hdbscan_pkg
from hdbscan import validity as hdbscan_validity
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, cophenet
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import umap

try:
    from sklearn_extra.cluster import KMedoids
    HAS_KMEDOIDS = True
except ImportError:
    HAS_KMEDOIDS = False
    print("WARNING: scikit-learn-extra not installed — K-Medoids will be skipped.\n"
          "  pip install scikit-learn-extra")

warnings.filterwarnings("ignore", category=FutureWarning)


# ===========================================================================
# CONFIG
# ===========================================================================

_REPO_ROOT = Path(__file__).parent.parent

RUN_TAG    = "2026-04-06"   # update to today's date before each new run
OUTPUT_DIR = _REPO_ROOT / "outputs" / "analysis" / RUN_TAG

# Must match the DATA_CONDITION used in both compute scripts
DATA_CONDITION = "D"

CONDITION_SOURCES = {
    "A":   {"Literature", "Hit"},
    "B":   {"Library", "Hit"},
    "C":   {"Literature", "Library", "Hit"},
    "ALL": {"Literature", "Library", "34_Hits", "Hit"},
    "D":   {"Literature", "34_Hits", "Hit"},
    "E":   {"Library", "34_Hits", "Hit"},
}

# ------------------------------------------------------------------
# Input paths — outputs from the two compute scripts
# ------------------------------------------------------------------
MORDRED_SCALED_CSV = _REPO_ROOT / "outputs" / "mordred"   / RUN_TAG / "mordred_filtered_scaled.csv"
MAPC_FPS_NPY       = _REPO_ROOT / "outputs" / "mapchiral" / RUN_TAG / "mapchiral_fingerprints.npy"
MAPC_META_CSV      = _REPO_ROOT / "outputs" / "mapchiral" / RUN_TAG / "mapchiral_metadata.csv"

INPUT_CSV = (
    _REPO_ROOT
    / "data" / "libraries" / "2026-01-29"
    / "canonicalized_master_macrocycles_2D_Descriptors_FINAL_20260129.csv"
)

SMILES_COL    = "Smiles"
SOURCE_COL    = "Source"
HIT_ID_COL    = "Hit_ID"
HIGHLIGHT_COL = "Highlight_ID"

# Curated 2D descriptor panel — preserved as-is, no filtering
# Note: cLogS and Aromatic Rings are not present in the source CSV
DESC_2D_COLS = [
    "Total Molweight",
    "cLogP",
    "H-Acceptors",
    "H-Donors",
    "Polar Surface Area",
    "Rotatable Bonds",
]

IQR_MULTIPLIER_2D = 1.5

# ------------------------------------------------------------------
# Dimensionality sweep (Mordred + MAPchiral only)
# ------------------------------------------------------------------
SWEEP_COMPONENTS     = [5, 10, 15]   # n_components values to test
N_COMPONENTS_DEFAULT = 10            # fallback if no stable pair found
CLUSTER_DELTA        = 2             # max cluster count change for "stable"
NOISE_DELTA          = 0.05          # max noise fraction change for "stable"

# ------------------------------------------------------------------
# UMAP base parameters per branch
# n_components is set per-run (sweep or final); not specified here
# ------------------------------------------------------------------
RANDOM_STATE = 42

UMAP_BASE_PARAMS = {
    "2d": dict(
        n_neighbors  = 15,
        min_dist     = 0.15,
        metric       = "euclidean",
        random_state = RANDOM_STATE,
    ),
    "mordred": dict(
        n_neighbors  = 60,
        min_dist     = 0.1,
        metric       = "cosine",
        random_state = RANDOM_STATE,
    ),
    # MAPchiral: metric set to minhash_distance callable at runtime
    "mapchiral": dict(
        n_neighbors  = 40,
        min_dist     = 0.1,
        random_state = RANDOM_STATE,
    ),
}

# ------------------------------------------------------------------
# HDBSCAN (on nD UMAP embeddings, Euclidean)
# min_cluster_size per branch — selected from mcs sweep
# ------------------------------------------------------------------
HDBSCAN_MIN_CLUSTER_SIZE = {
    "2d":        50,    # 2D branch: mcs=50 retained (low DBCV space, enrichment preferred)
    "mordred":   80,    # sweep-selected: 35 clusters, DBCV=+0.511, sil=+0.768
    "mapchiral": 180,   # sweep-selected: 16 clusters, DBCV=+0.463 (k-medoids k=15 used for figures)
}
HDBSCAN_MIN_SAMPLES = 10

# ------------------------------------------------------------------
# Ward hierarchical clustering (Mordred branch only)
# k selected by silhouette sweep; dendrogram saved for inspection
# ------------------------------------------------------------------
WARD_K_MAX  = 100  # upper bound of silhouette sweep; true peak confirmed at k=64

# ------------------------------------------------------------------
# K-Medoids (MAPchiral branch only)
# k=15 confirmed by Gap Statistic in prior run — no re-sweep needed
# ------------------------------------------------------------------
K_MEDOIDS_K_FIXED = 15             # confirmed optimal k for MAPchiral
K_MEDOIDS_N_INIT  = 20             # multiple initializations at fixed k
K_MEDOIDS_INIT    = "k-medoids++"

# ------------------------------------------------------------------
# Figure colors
# ------------------------------------------------------------------
COLOR_LITERATURE = "#606060"
COLOR_LIBRARY    = "#1F77B4"
COLOR_34HITS     = "#E41A1C"
COLOR_HIT        = "#FF7F00"

DRAW_ORDER = ["Literature", "Library", "34_Hits", "Hit"]
SOURCE_STYLE = {
    "Literature": (COLOR_LITERATURE, 4,  0.45),
    "Library":    (COLOR_LIBRARY,    4,  0.15),
    "34_Hits":    (COLOR_34HITS,     30, 0.90),
    "Hit":        (COLOR_HIT,        40, 1.00),
}


# ===========================================================================
# MinHash distance metric
# Must be at module level for numba JIT compilation.
#
# Verification:
#   MinHash theorem: P(h_i(A) == h_i(B)) = Jaccard(shingles_A, shingles_B)
#   => fraction of agreeing positions ≈ Jaccard similarity
#   => 1 - fraction = Jaccard distance (MinHash-consistent)
#
#   Equivalent to: 1 - mapchiral.jaccard_similarity(fp_A, fp_B)
#   Ref: doi.org/10.1186/s13321-024-00849-6
# ===========================================================================

@njit
def minhash_distance(fp_a: np.ndarray, fp_b: np.ndarray) -> float:
    m  = fp_a.shape[0]
    eq = 0
    for k in range(m):
        if fp_a[k] == fp_b[k]:
            eq += 1
    return 1.0 - (eq / m)


# ===========================================================================
# HELPERS
# ===========================================================================

def _elapsed(start: float) -> str:
    return f"{time.time() - start:.1f}s"


def iqr_clip(df: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    q1  = df.quantile(0.25)
    q3  = df.quantile(0.75)
    iqr = q3 - q1
    return df.clip(lower=q1 - multiplier * iqr,
                   upper=q3 + multiplier * iqr, axis=1)


def _hdbscan_metrics(labels: np.ndarray) -> dict:
    n_clusters    = len(set(labels)) - (1 if -1 in labels else 0)
    noise_frac    = float((labels == -1).sum()) / len(labels)
    cluster_sizes = sorted(
        [int((labels == c).sum()) for c in set(labels) if c != -1],
        reverse=True,
    )
    return {
        "n_clusters":    n_clusters,
        "noise_frac":    noise_frac,
        "cluster_sizes": cluster_sizes,
    }


# ===========================================================================
# LOAD BRANCHES
# ===========================================================================

def load_2d_descriptors(condition: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load curated 2D descriptor panel from master CSV.
    Pipeline: condition filter → IQR clip → StandardScaler.
    No variance or correlation filtering — panel is small and intentionally curated.
    """
    print("\n[2D] Loading 2D descriptors ...")
    df = pd.read_csv(INPUT_CSV)

    for col in [SMILES_COL, SOURCE_COL, HIT_ID_COL, HIGHLIGHT_COL]:
        if col not in df.columns:
            df[col] = ""

    missing = [c for c in DESC_2D_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing 2D descriptor columns: {missing}")

    allowed = CONDITION_SOURCES[condition]
    df = df[df[SOURCE_COL].isin(allowed)].reset_index(drop=True)
    print(f"   Condition '{condition}' → {len(df):,} molecules")

    n_before = len(df)
    df = df.dropna(subset=DESC_2D_COLS).reset_index(drop=True)
    if len(df) < n_before:
        print(f"   Dropped {n_before - len(df)} rows with NaN in descriptor cols")

    meta = df[[SMILES_COL, SOURCE_COL, HIT_ID_COL, HIGHLIGHT_COL]].copy()
    X    = df[DESC_2D_COLS].astype(float)
    X    = iqr_clip(X, IQR_MULTIPLIER_2D)
    X_sc = pd.DataFrame(
        StandardScaler().fit_transform(X.values),
        columns=DESC_2D_COLS,
        index=meta.index,
    )
    print(f"   Ready: {X_sc.shape[0]:,} × {X_sc.shape[1]}")
    return meta, X_sc


def load_mordred(scaled_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load pre-scaled Mordred matrix from compute_mordred_descriptors.py."""
    print("\n[Mordred] Loading scaled matrix ...")
    df = pd.read_csv(scaled_csv)
    meta_cols = [c for c in [SMILES_COL, SOURCE_COL, HIT_ID_COL, HIGHLIGHT_COL]
                 if c in df.columns]
    meta = df[meta_cols].copy()
    X    = df[[c for c in df.columns if c not in meta_cols]].astype(float)
    print(f"   Ready: {X.shape[0]:,} × {X.shape[1]}")
    return meta, X


def load_mapchiral(fps_npy: Path, meta_csv: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Load MAPchiral fingerprints. No preprocessing applied."""
    print("\n[MAPchiral] Loading fingerprints ...")
    fps  = np.load(fps_npy)
    meta = pd.read_csv(meta_csv)
    if fps.shape[0] != len(meta):
        raise ValueError(
            f"Shape mismatch: {fps.shape[0]} fps vs {len(meta)} metadata rows."
        )
    print(f"   Ready: {fps.shape[0]:,} × {fps.shape[1]}, dtype {fps.dtype}")
    return meta, fps


# ===========================================================================
# ALIGNMENT
# ===========================================================================

def align_branches(
    meta_2d: pd.DataFrame,
    meta_mordred: pd.DataFrame,
    meta_mapc: pd.DataFrame,
) -> np.ndarray:
    """Inner join on SMILES; preserve 2D branch ordering."""
    common = (set(meta_2d[SMILES_COL])
              & set(meta_mordred[SMILES_COL])
              & set(meta_mapc[SMILES_COL]))
    ordered = meta_2d[meta_2d[SMILES_COL].isin(common)][SMILES_COL].values

    print(f"\n[Align]  2D: {len(meta_2d):,}  |  "
          f"Mordred: {len(meta_mordred):,}  |  "
          f"MAPchiral: {len(meta_mapc):,}  →  common: {len(ordered):,}")
    if len(ordered) < len(meta_2d):
        print(f"   {len(meta_2d) - len(ordered)} molecules absent from ≥1 branch — excluded")
    return ordered


def reindex_to_common(
    meta: pd.DataFrame,
    X,
    common_smiles: np.ndarray,
) -> tuple[pd.DataFrame, any]:
    idx_map = {s: i for i, s in enumerate(meta[SMILES_COL])}
    idx     = [idx_map[s] for s in common_smiles]
    meta_a  = meta.iloc[idx].reset_index(drop=True)
    X_a     = X.iloc[idx].reset_index(drop=True) if isinstance(X, pd.DataFrame) else X[idx]
    return meta_a, X_a


# ===========================================================================
# UMAP
# ===========================================================================

def _build_umap_params(base: dict, n_components: int, branch: str) -> dict:
    p = dict(base, n_components=n_components)
    if branch == "mapchiral":
        p["metric"] = minhash_distance
    return p


def run_umap(X, n_components: int, base_params: dict, branch: str) -> np.ndarray:
    params = _build_umap_params(base_params, n_components, branch)
    metric_label = "minhash_distance" if branch == "mapchiral" else params["metric"]
    print(f"   UMAP n_components={n_components}  "
          f"n_neighbors={params['n_neighbors']}  "
          f"min_dist={params['min_dist']}  "
          f"metric={metric_label}  ...", end=" ", flush=True)
    t0        = time.time()
    reducer   = umap.UMAP(**params)
    embedding = reducer.fit_transform(X if isinstance(X, np.ndarray) else X.values)
    print(f"{_elapsed(t0)}")
    return embedding


# ===========================================================================
# DIMENSIONALITY SWEEP
# ===========================================================================

def sweep_dimensionality(
    X,
    base_params: dict,
    branch: str,
    output_dir: Path,
) -> tuple[int, dict]:
    """
    Sweep UMAP n_components over SWEEP_COMPONENTS, run HDBSCAN at each,
    record cluster metrics, and select the smallest n_components where
    cluster structure is stable relative to the next step.

    Stability criterion (per consecutive pair n_k, n_{k+1}):
      |n_clusters(n_k) - n_clusters(n_{k+1})| <= CLUSTER_DELTA
      AND |noise_frac(n_k) - noise_frac(n_{k+1})| <= NOISE_DELTA

    Returns (selected_n_components, sweep_results_dict).
    """
    print(f"\n[Sweep:{branch}] n_components = {SWEEP_COMPONENTS}")

    sweep = {}
    for nc in SWEEP_COMPONENTS:
        emb    = run_umap(X, nc, base_params, branch)
        labels = _run_hdbscan_internal(emb, branch)
        m      = _hdbscan_metrics(labels)
        sweep[nc] = {"embedding": emb, "labels": labels, "metrics": m}
        print(f"   n_components={nc:>2}  →  "
              f"clusters={m['n_clusters']:>3}  "
              f"noise={m['noise_frac']:.3f}  "
              f"sizes={m['cluster_sizes'][:5]}{'...' if len(m['cluster_sizes']) > 5 else ''}")

    # Select smallest stable n_components
    selected = _select_n_components(sweep)
    print(f"   → Selected n_components = {selected}")

    # Write sweep report
    _write_sweep_report(branch, sweep, selected, output_dir)

    return selected, sweep


def _run_hdbscan_internal(embedding: np.ndarray, branch: str) -> np.ndarray:
    """HDBSCAN on an embedding — used internally during dimensionality sweep."""
    return hdbscan_pkg.HDBSCAN(
        min_cluster_size = HDBSCAN_MIN_CLUSTER_SIZE[branch],
        min_samples      = HDBSCAN_MIN_SAMPLES,
        metric           = "euclidean",
    ).fit_predict(embedding)


def _select_n_components(sweep: dict) -> int:
    """
    Return the smallest n_components where the (n_k, n_{k+1}) pair is stable.
    Falls back to N_COMPONENTS_DEFAULT if no stable pair found.
    """
    components = sorted(sweep.keys())
    for i in range(len(components) - 1):
        n_k   = components[i]
        n_k1  = components[i + 1]
        m_k   = sweep[n_k]["metrics"]
        m_k1  = sweep[n_k1]["metrics"]
        cluster_stable = abs(m_k["n_clusters"] - m_k1["n_clusters"]) <= CLUSTER_DELTA
        noise_stable   = abs(m_k["noise_frac"]  - m_k1["noise_frac"])  <= NOISE_DELTA
        if cluster_stable and noise_stable:
            return n_k
    return N_COMPONENTS_DEFAULT


def _write_sweep_report(
    branch: str,
    sweep: dict,
    selected: int,
    output_dir: Path,
) -> None:
    lines = [
        "=" * 62,
        f"DIMENSIONALITY SWEEP REPORT — {branch.upper()}",
        "=" * 62,
        f"HDBSCAN min_cluster_size : {HDBSCAN_MIN_CLUSTER_SIZE[branch]}",
        f"HDBSCAN min_samples      : {HDBSCAN_MIN_SAMPLES}",
        f"Stability criterion      : |Δclusters| <= {CLUSTER_DELTA}  "
        f"AND  |Δnoise_frac| <= {NOISE_DELTA}",
        f"Default fallback         : n_components = {N_COMPONENTS_DEFAULT}",
        "",
        f"{'n_components':<14} {'n_clusters':<12} {'noise_frac':<12} {'top_5_sizes'}",
        "-" * 62,
    ]
    for nc, res in sorted(sweep.items()):
        m    = res["metrics"]
        top5 = str(m["cluster_sizes"][:5])
        marker = "  ← SELECTED" if nc == selected else ""
        lines.append(
            f"{nc:<14} {m['n_clusters']:<12} {m['noise_frac']:<12.3f} {top5}{marker}"
        )
    lines += [
        "",
        f"Selected n_components : {selected}",
        "=" * 62,
    ]
    report_text = "\n".join(lines)
    print("\n" + report_text)
    path = output_dir / f"{branch}_sweep_report.txt"
    with open(path, "w") as fh:
        fh.write(report_text + "\n")
    print(f"   Saved: {path.name}")


# ===========================================================================
# HDBSCAN (final, on nD embedding)
# ===========================================================================

def run_hdbscan(
    embedding: np.ndarray,
    branch: str,
    fig_dir: Path,
) -> tuple[np.ndarray, dict]:
    """
    Fit HDBSCAN, compute DBCV, extract stability scores and membership
    probabilities, and save condensed tree + stability bar chart.

    Returns (labels, validation_dict).
    Validation dict keys:
      n_clusters, noise_frac, dbcv,
      persistence (per-cluster lambda-based stability, bigger = better),
      probabilities (per-sample membership probability, 1.0 = core member)
    """
    mcs = HDBSCAN_MIN_CLUSTER_SIZE[branch]
    t0  = time.time()

    clusterer = hdbscan_pkg.HDBSCAN(
        min_cluster_size = mcs,
        min_samples      = HDBSCAN_MIN_SAMPLES,
        metric           = "euclidean",
    )
    clusterer.fit(embedding)
    labels      = clusterer.labels_
    probs       = clusterer.probabilities_
    persistence = clusterer.cluster_persistence_   # lambda-based, bigger = better

    m = _hdbscan_metrics(labels)

    # DBCV
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            dbcv = hdbscan_validity.validity_index(
                embedding.astype(np.float64), labels
            )
        except Exception:
            dbcv = np.nan

    print(f"   HDBSCAN → clusters: {m['n_clusters']}  noise: {m['noise_frac']:.3f}"
          f"  DBCV: {dbcv:+.3f}  ({_elapsed(t0)})")

    # ── Condensed tree ────────────────────────────────────────────────
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        clusterer.condensed_tree_.plot(
            select_clusters=True,
            selection_palette=plt.get_cmap("tab20").colors,
            axis=ax,
        )
        ax.set_title(
            f"{branch.upper()} — HDBSCAN Condensed Tree\n"
            f"mcs={mcs}  clusters={m['n_clusters']}  DBCV={dbcv:+.3f}",
            fontsize=9,
        )
        fig.tight_layout()
        path = fig_dir / f"{branch}_hdbscan_condensed_tree.svg"
        fig.savefig(path, format="svg", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"   Saved: {path.name}")
    except Exception as e:
        print(f"   Condensed tree plot failed: {e}")

    # ── Stability bar chart ───────────────────────────────────────────
    if len(persistence) > 0:
        fig, ax = plt.subplots(figsize=(max(6, len(persistence) * 0.4), 4))
        cluster_ids = list(range(len(persistence)))
        bars = ax.bar(cluster_ids, persistence, color="#2196F3", alpha=0.8)
        ax.set_xlabel("Cluster ID")
        ax.set_ylabel("Stability (λ-based, bigger = more robust)")
        ax.set_title(
            f"{branch.upper()} — HDBSCAN Per-Cluster Stability\n"
            f"Relative comparison: taller bars = more trustworthy clusters",
            fontsize=9,
        )
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        path = fig_dir / f"{branch}_hdbscan_stability.svg"
        fig.savefig(path, format="svg", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"   Saved: {path.name}")

    val = {
        "n_clusters":   m["n_clusters"],
        "noise_frac":   m["noise_frac"],
        "dbcv":         dbcv,
        "persistence":  persistence,
        "probabilities": probs,
    }
    return labels, val


# ===========================================================================
# WARD HIERARCHICAL CLUSTERING (Mordred branch)
# ===========================================================================

def run_ward(
    embedding: np.ndarray,
    branch: str,
    fig_dir: Path,
) -> tuple[np.ndarray, dict]:
    """
    Ward hierarchical clustering on the nD UMAP embedding.

    Cut level selected by silhouette sweep over k=2..WARD_K_MAX.
    Saves: dendrogram SVG + silhouette sweep SVG.

    Returns (labels, validation_dict).
    Validation dict keys: k, silhouette, cophenetic_r
    """
    t0 = time.time()
    print(f"   [Ward] Computing linkage on {embedding.shape[0]:,} × {embedding.shape[1]}D ...")

    Z = linkage(embedding, method="ward", metric="euclidean")

    # Cophenetic correlation — measures how well the dendrogram preserves distances
    c, _ = cophenet(Z, pdist(embedding, metric="euclidean"))
    print(f"   Cophenetic r = {c:.3f}")

    # ── Silhouette sweep to select cut level ──────────────────────────
    k_max   = min(WARD_K_MAX, len(embedding) - 1)
    k_range = range(2, k_max + 1)
    sil_scores = []
    for k in k_range:
        lbls = fcluster(Z, k, criterion="maxclust") - 1   # 0-indexed
        try:
            s = silhouette_score(embedding, lbls, metric="euclidean")
        except Exception:
            s = np.nan
        sil_scores.append(s)

    best_k = int(list(k_range)[int(np.nanargmax(sil_scores))])
    labels = fcluster(Z, best_k, criterion="maxclust") - 1

    sil = float(np.nanmax(sil_scores))
    print(f"   Ward → k={best_k}  silhouette={sil:.3f}  ({_elapsed(t0)})")

    # ── Figure 1: Dendrogram (truncated to top 40 merges) ─────────────
    # Color threshold = midpoint of the merge that creates best_k clusters
    thresh = float((Z[-(best_k), 2] + Z[-(best_k - 1), 2]) / 2) if best_k > 1 else Z[-1, 2]
    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(Z, truncate_mode="lastp", p=40, ax=ax,
               color_threshold=thresh, above_threshold_color="#AAAAAA")
    ax.axhline(thresh, color="red", lw=1.2, linestyle="--", alpha=0.7,
               label=f"Cut → k={best_k}")
    ax.set_xlabel("Sample / merged cluster (count in brackets)")
    ax.set_ylabel("Ward merge distance")
    ax.set_title(
        f"{branch.upper()} — Ward Dendrogram (top 40 merges)\n"
        f"k={best_k} (silhouette-optimal)  |  cophenetic r={c:.3f}",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = fig_dir / f"{branch}_ward_dendrogram.svg"
    fig.savefig(path, format="svg", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {path.name}")

    # ── Figure 2: Silhouette sweep ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(list(k_range), sil_scores, "o-", color="#2196F3", lw=2, ms=4)
    ax.axvline(best_k, color="red", lw=1.2, linestyle="--",
               label=f"Selected k={best_k} (silhouette peak)")
    for thresh_val, lbl in [(0.25, "weak"), (0.50, "ok"), (0.70, "good")]:
        ax.axhline(thresh_val, color="grey", lw=0.7, linestyle=":")
        ax.text(list(k_range)[0], thresh_val, f" {lbl}", va="bottom",
                fontsize=7, color="grey")
    ax.set_xlabel("Number of Ward clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title(
        f"{branch.upper()} — Ward cut level selection\n"
        f"Silhouette sweep k=2..{k_max}",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = fig_dir / f"{branch}_ward_silhouette.svg"
    fig.savefig(path, format="svg", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {path.name}")

    val = {
        "k":            best_k,
        "silhouette":   sil,
        "cophenetic_r": c,
    }
    return labels, val


# ===========================================================================
# K-MEDOIDS (MAPchiral branch — fixed k=15, confirmed by prior Gap Statistic)
# ===========================================================================

def run_kmedoids(
    embedding: np.ndarray,
    branch: str,
    fig_dir: Path,
) -> tuple[np.ndarray, dict]:
    """
    Fit K-Medoids at K_MEDOIDS_K_FIXED with multiple initializations.
    k=15 was selected by Gap Statistic (1-SE rule) in the prior analysis run
    and is hardcoded here to avoid redundant computation.

    Returns (labels, validation_dict).
    Validation dict keys: k, silhouette, dbi, total_cost
    """
    if not HAS_KMEDOIDS:
        print("   K-Medoids skipped — scikit-learn-extra not installed")
        return np.full(len(embedding), -1, dtype=int), {}

    k   = K_MEDOIDS_K_FIXED
    t0  = time.time()
    rng = np.random.default_rng(RANDOM_STATE)
    print(f"   Fitting K-Medoids k={k} × {K_MEDOIDS_N_INIT} inits ...")

    best_cost = np.inf
    best_km   = None
    for _ in range(K_MEDOIDS_N_INIT):
        km = KMedoids(
            n_clusters   = k,
            metric       = "euclidean",
            init         = K_MEDOIDS_INIT,
            random_state = int(rng.integers(1_000_000)),
        )
        km.fit(embedding)
        if km.inertia_ < best_cost:
            best_cost = km.inertia_
            best_km   = km

    labels = best_km.labels_

    try:
        sil = silhouette_score(embedding, labels, metric="euclidean")
        dbi = davies_bouldin_score(embedding, labels)
    except Exception:
        sil, dbi = np.nan, np.nan

    print(f"   K-Medoids (k={k}) → cost={best_cost:.1f}  "
          f"silhouette={sil:.3f}  DBI={dbi:.3f}  ({_elapsed(t0)})")

    val = {
        "k":          k,
        "silhouette": sil,
        "dbi":        dbi,
        "total_cost": best_cost,
    }
    return labels, val


# ===========================================================================
# FIGURES (use 2D embedding for all visualizations)
# ===========================================================================

def _scatter_source(ax, emb2d: np.ndarray, sources: pd.Series) -> list:
    handles = []
    for src in DRAW_ORDER:
        idx = np.where(sources == src)[0]
        if len(idx) == 0:
            continue
        color, size, alpha = SOURCE_STYLE.get(src, ("#999999", 4, 0.3))
        ax.scatter(emb2d[idx, 0], emb2d[idx, 1],
                   s=size, c=color, alpha=alpha, linewidths=0, rasterized=True)
        handles.append(mpatches.Patch(color=color, label=f"{src} (n={len(idx):,})"))
    return handles


def _scatter_labels(ax, emb2d: np.ndarray, labels: np.ndarray) -> list:
    unique  = sorted(set(labels))
    cmap    = plt.get_cmap("tab20")
    handles = []
    for lbl in unique:
        idx   = np.where(labels == lbl)[0]
        color = "#CCCCCC" if lbl == -1 else cmap(lbl % 20)
        name  = "Noise" if lbl == -1 else f"Cluster {lbl}"
        ax.scatter(emb2d[idx, 0], emb2d[idx, 1],
                   s=4, c=[color], alpha=0.4, linewidths=0, rasterized=True)
        handles.append(mpatches.Patch(color=color, label=f"{name} (n={len(idx):,})"))
    return handles


def save_figure(
    emb2d: np.ndarray,
    color_data,
    mode: str,
    branch: str,
    fig_dir: Path,
    n_components_cluster: int,
    k: int = None,
) -> None:
    """
    All figures use the 2D UMAP embedding for x/y axes.
    Cluster labels come from the nD embedding (noted in title).
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_facecolor("#F5F5F5")

    if mode == "source":
        handles      = _scatter_source(ax, emb2d, color_data)
        title_suffix = "Source"
        note         = ""
    else:
        handles      = _scatter_labels(ax, emb2d, color_data)
        if mode == "hdbscan":
            method = "HDBSCAN"
        elif mode == "ward":
            method = f"Ward (k={k})"
        else:
            method = f"K-Medoids (k={k})"
        title_suffix = method
        note         = f" [labels from {n_components_cluster}D embedding]"

    base   = UMAP_BASE_PARAMS[branch]
    metric = "minhash_distance" if branch == "mapchiral" else base.get("metric", "euclidean")
    ax.set_xlabel("UMAP 1 (2D)")
    ax.set_ylabel("UMAP 2 (2D)")
    ax.set_title(
        f"{branch.upper()} — {title_suffix}{note}\n"
        f"n_neighbors={base['n_neighbors']}  "
        f"min_dist={base['min_dist']}  "
        f"metric={metric}",
        fontsize=9,
    )
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.8,
              ncol=max(1, len(handles) // 12))
    ax.grid(True, alpha=0.3, linewidth=0.5)
    fig.tight_layout()

    path = fig_dir / f"{branch}_umap_{mode}.svg"
    fig.savefig(path, format="svg", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {path.name}")


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    print("=" * 62)
    print("CHEMICAL SPACE ANALYSIS")
    print(f"  Condition : {DATA_CONDITION}")
    print(f"  Output    : {OUTPUT_DIR}")
    print("=" * 62)

    t_total = time.time()
    fig_dir = OUTPUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load all three branches
    # ------------------------------------------------------------------
    meta_2d,      X_2d      = load_2d_descriptors(DATA_CONDITION)
    meta_mordred, X_mordred  = load_mordred(MORDRED_SCALED_CSV)
    meta_mapc,    fps_mapc   = load_mapchiral(MAPC_FPS_NPY, MAPC_META_CSV)

    # ------------------------------------------------------------------
    # Align to common SMILES
    # ------------------------------------------------------------------
    common_smiles             = align_branches(meta_2d, meta_mordred, meta_mapc)
    meta_2d,      X_2d       = reindex_to_common(meta_2d,      X_2d,      common_smiles)
    meta_mordred, X_mordred   = reindex_to_common(meta_mordred, X_mordred, common_smiles)
    meta_mapc,    fps_mapc    = reindex_to_common(meta_mapc,    fps_mapc,  common_smiles)

    aligned_meta = meta_2d.copy()
    sources      = aligned_meta[SOURCE_COL]

    # ------------------------------------------------------------------
    # 2D branch — no dimensionality sweep (8D input, curated panel)
    # ------------------------------------------------------------------
    print(f"\n{'='*62}\n  Branch: 2D DESCRIPTORS\n{'='*62}")

    print("\n  [2D] UMAP 2D (visualization + clustering):")
    emb_2d_2d = run_umap(X_2d, 2, UMAP_BASE_PARAMS["2d"], "2d")

    print("\n  [2D] HDBSCAN:")
    hdb_2d, hdb_2d_val = run_hdbscan(emb_2d_2d, "2d", fig_dir)

    n_components_2d = 2   # 2D is sufficient given 6-feature panel; no secondary clustering

    # ------------------------------------------------------------------
    # Mordred branch — dimensionality sweep then final run
    # ------------------------------------------------------------------
    print(f"\n{'='*62}\n  Branch: MORDRED\n{'='*62}")

    print("\n  [Mordred] UMAP 2D (visualization only):")
    emb_mordred_2d = run_umap(X_mordred, 2, UMAP_BASE_PARAMS["mordred"], "mordred")

    print("\n  [Mordred] Dimensionality sweep (for clustering):")
    n_mordred, sweep_mordred = sweep_dimensionality(
        X_mordred, UMAP_BASE_PARAMS["mordred"], "mordred", OUTPUT_DIR
    )

    # Use embedding from sweep if already computed, else re-run
    emb_mordred_nd = sweep_mordred[n_mordred]["embedding"]
    hdb_labels_from_sweep = sweep_mordred[n_mordred]["labels"]

    print(f"\n  [Mordred] Final HDBSCAN on {n_mordred}D embedding:")
    hdb_mordred, hdb_mordred_val = run_hdbscan(emb_mordred_nd, "mordred", fig_dir)

    print(f"\n  [Mordred] Ward hierarchical on {n_mordred}D embedding:")
    ward_mordred, ward_mordred_val = run_ward(emb_mordred_nd, "mordred", fig_dir)

    # ------------------------------------------------------------------
    # MAPchiral branch — dimensionality sweep then final run
    # ------------------------------------------------------------------
    print(f"\n{'='*62}\n  Branch: MAPCHIRAL\n{'='*62}")

    print("\n  [MAPchiral] UMAP 2D (visualization only):")
    emb_mapc_2d = run_umap(fps_mapc, 2, UMAP_BASE_PARAMS["mapchiral"], "mapchiral")

    print("\n  [MAPchiral] Dimensionality sweep (for clustering):")
    n_mapc, sweep_mapc = sweep_dimensionality(
        fps_mapc, UMAP_BASE_PARAMS["mapchiral"], "mapchiral", OUTPUT_DIR
    )

    emb_mapc_nd = sweep_mapc[n_mapc]["embedding"]

    print(f"\n  [MAPchiral] Final HDBSCAN on {n_mapc}D embedding:")
    hdb_mapc, hdb_mapc_val = run_hdbscan(emb_mapc_nd, "mapchiral", fig_dir)

    print(f"\n  [MAPchiral] K-Medoids on {n_mapc}D embedding:")
    km_mapc, km_mapc_val = run_kmedoids(emb_mapc_nd, "mapchiral", fig_dir)

    # ------------------------------------------------------------------
    # Save per-branch CSVs
    # ------------------------------------------------------------------
    print(f"\n[Save] Writing CSVs ...")

    def _save_branch_csv(name, emb2d, embnd, hdb, hdb_val,
                         secondary=None, secondary_col=None):
        df = aligned_meta.copy()
        df["UMAP_1_2d"] = emb2d[:, 0]
        df["UMAP_2_2d"] = emb2d[:, 1]
        for i in range(embnd.shape[1]):
            df[f"UMAP_{i+1}_nd"] = embnd[:, i]
        df["cluster_hdbscan"]         = hdb
        df["hdbscan_membership_prob"] = hdb_val.get("probabilities", np.nan)
        if secondary is not None and secondary_col is not None:
            df[secondary_col] = secondary
        path = OUTPUT_DIR / f"{name}_umap.csv"
        df.to_csv(path, index=False)
        print(f"   {path.name}  ({len(df):,} rows)")

    _save_branch_csv("2d",        emb_2d_2d,      emb_2d_2d,      hdb_2d,      hdb_2d_val)
    _save_branch_csv("mordred",   emb_mordred_2d, emb_mordred_nd, hdb_mordred, hdb_mordred_val,
                     secondary=ward_mordred,  secondary_col="cluster_ward")
    _save_branch_csv("mapchiral", emb_mapc_2d,    emb_mapc_nd,    hdb_mapc,    hdb_mapc_val,
                     secondary=km_mapc,       secondary_col="cluster_kmedoids")

    # ------------------------------------------------------------------
    # Figures (all use 2D embedding for x/y)
    # ------------------------------------------------------------------
    print(f"\n[Figures] ...")

    # 2D — HDBSCAN only
    save_figure(emb_2d_2d,     sources,      "source",  "2d",        fig_dir, n_components_2d)
    save_figure(emb_2d_2d,     hdb_2d,       "hdbscan", "2d",        fig_dir, n_components_2d)

    # Mordred — HDBSCAN + Ward
    save_figure(emb_mordred_2d, sources,     "source",  "mordred",   fig_dir, n_mordred)
    save_figure(emb_mordred_2d, hdb_mordred, "hdbscan", "mordred",   fig_dir, n_mordred)
    save_figure(emb_mordred_2d, ward_mordred, "ward",   "mordred",   fig_dir, n_mordred,
                k=ward_mordred_val.get("k"))

    # MAPchiral — HDBSCAN + K-Medoids
    save_figure(emb_mapc_2d,   sources,      "source",  "mapchiral", fig_dir, n_mapc)
    save_figure(emb_mapc_2d,   hdb_mapc,     "hdbscan", "mapchiral", fig_dir, n_mapc)
    save_figure(emb_mapc_2d,   km_mapc,      "kmedoids","mapchiral", fig_dir, n_mapc,
                k=km_mapc_val.get("k"))

    # ------------------------------------------------------------------
    # Combined aligned metadata
    # ------------------------------------------------------------------
    combined = aligned_meta.copy()
    for col, vals in [
        ("umap1_2d",                emb_2d_2d[:, 0]),
        ("umap2_2d",                emb_2d_2d[:, 1]),
        ("hdbscan_2d",              hdb_2d),
        ("hdbscan_2d_prob",         hdb_2d_val.get("probabilities", np.nan)),
        ("umap1_mordred",           emb_mordred_2d[:, 0]),
        ("umap2_mordred",           emb_mordred_2d[:, 1]),
        ("hdbscan_mordred",         hdb_mordred),
        ("hdbscan_mordred_prob",    hdb_mordred_val.get("probabilities", np.nan)),
        ("ward_mordred",            ward_mordred),
        ("umap1_mapchiral",         emb_mapc_2d[:, 0]),
        ("umap2_mapchiral",         emb_mapc_2d[:, 1]),
        ("hdbscan_mapchiral",       hdb_mapc),
        ("hdbscan_mapchiral_prob",  hdb_mapc_val.get("probabilities", np.nan)),
        ("kmedoids_mapchiral",      km_mapc),
    ]:
        combined[col] = vals
    combined.to_csv(OUTPUT_DIR / "aligned_metadata.csv", index=False)
    print(f"\n   aligned_metadata.csv  ({len(combined):,} rows)")

    # ------------------------------------------------------------------
    # Analysis report
    # ------------------------------------------------------------------
    def _branch_summary(name, hdb_val, secondary_val, nc):
        lines = [
            f"  {name.upper()}:",
            f"    Clustering n_components : {nc}",
            f"    HDBSCAN clusters        : {hdb_val.get('n_clusters', '?')}",
            f"    HDBSCAN noise fraction  : {hdb_val.get('noise_frac', float('nan')):.3f}",
            f"    HDBSCAN DBCV            : {hdb_val.get('dbcv', float('nan')):+.3f}",
            f"    HDBSCAN stable clusters : {len(hdb_val.get('persistence', []))}  "
            f"(λ-based, bigger = more robust)",
        ]
        if name == "mordred":
            lines += [
                f"    Ward k (sil peak)       : {secondary_val.get('k', '?')}",
                f"    Ward silhouette         : {secondary_val.get('silhouette', float('nan')):.3f}",
                f"    Ward cophenetic r       : {secondary_val.get('cophenetic_r', float('nan')):.3f}",
            ]
        elif name == "mapchiral":
            lines += [
                f"    K-Medoids k (fixed)     : {secondary_val.get('k', '?')}",
                f"    K-Medoids silhouette    : {secondary_val.get('silhouette', float('nan')):.3f}",
                f"    K-Medoids DBI           : {secondary_val.get('dbi', float('nan')):.3f}",
                f"    K-Medoids total cost    : {secondary_val.get('total_cost', float('nan')):.1f}",
            ]
        else:
            lines.append("    Secondary clustering    : none (6-feature panel)")
        lines.append("")
        return lines

    lines = [
        "=" * 62,
        "CHEMICAL SPACE ANALYSIS REPORT",
        "=" * 62,
        f"Run date          : {RUN_TAG}",
        f"Data condition    : {DATA_CONDITION}",
        f"Aligned molecules : {len(common_smiles):,}",
        "",
        "--- UMAP Metrics ---",
        "  2D UMAP : visualization only (all branches)",
        "  nD UMAP : clustering (Mordred + MAPchiral, sweep-selected)",
        "",
        "--- Branch Results ---",
    ]
    lines += _branch_summary("2d",        hdb_2d_val,      {},              n_components_2d)
    lines += _branch_summary("mordred",   hdb_mordred_val, ward_mordred_val, n_mordred)
    lines += _branch_summary("mapchiral", hdb_mapc_val,    km_mapc_val,     n_mapc)
    lines += [
        "--- Clustering ---",
        "  2D:        HDBSCAN only",
        "  Mordred:   HDBSCAN + Ward hierarchical (k by silhouette sweep)",
        "  MAPchiral: HDBSCAN + K-Medoids (k=15, confirmed by prior Gap Statistic)",
        "  All clustering on nD UMAP embeddings (Euclidean).",
        "  Figures show 2D UMAP coords; cluster labels from nD embedding.",
        "",
        "--- Output Files ---",
        "  2d_umap.csv  |  mordred_umap.csv  |  mapchiral_umap.csv",
        "  aligned_metadata.csv",
        "  mordred_sweep_report.txt  |  mapchiral_sweep_report.txt",
        "  figures/  (2D: source+hdbscan; Mordred: +ward; MAPchiral: +kmedoids)",
        "=" * 62,
    ]
    report_text = "\n".join(lines)
    print("\n" + report_text)
    with open(OUTPUT_DIR / "analysis_report.txt", "w") as fh:
        fh.write(report_text + "\n")

    print(f"\n[✓] Analysis complete — {_elapsed(t_total)}")


if __name__ == "__main__":
    main()
