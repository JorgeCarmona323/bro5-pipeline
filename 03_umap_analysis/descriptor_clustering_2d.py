"""
Descriptor-Based Chemical Space Analysis: UMAP + HDBSCAN + K-Medoids
=====================================================================

Pipeline (in order):
  1. Load CSV and filter by condition
  2. Optional RDKit augmentation (Aromatic Rings)
  3. Outlier handling — IQR-based clipping per descriptor
  4. Scaling — StandardScaler (default) or MinMaxScaler
  5. UMAP — projects nD descriptor space → 2D
  6. HDBSCAN — density-based clustering on UMAP embeddings
  7. K-Medoids — partition clustering on UMAP embeddings (k configurable)
  8. Figures:
        Fig A: UMAP colored by source / hit type (+ zoom inset)
        Fig B: UMAP colored by HDBSCAN cluster labels
        Fig C: UMAP colored by K-Medoids cluster labels
        Fig D: Descriptor correlation heatmap (preprocessing diagnostic)
        Fig E: K-selection silhouette scores (run once, then fix K_MEDOIDS)
  9. CSV exports:
        embeddings + cluster labels for all compounds
        HDBSCAN cluster summary
        K-Medoids cluster summary + medoid compound info

Scientific notes
----------------
- With only 6 Lipinski-like descriptors, UMAP largely reveals MW/polarity
  gradients, not fine structural differences. This is consistent with the
  Panel 3 "weak clustering" finding from 2026-01-30. Mordred descriptors or
  3D descriptors will substantially improve cluster resolution.
- HDBSCAN on UMAP embeddings is the approach recommended by McInnes et al.
  (UMAP authors). The cluster boundaries are UMAP-parameter-dependent, so
  fix random_state and use consistent UMAP params.
- K-Medoids picks actual molecules as cluster centers (vs abstract centroids),
  enabling direct chemical interpretation of each cluster representative.
- Euclidean distance on StandardScaled descriptors is appropriate here.
  For non-Euclidean metrics (Cosine, Manhattan), toggle METRIC below and a
  precomputed distance matrix will be passed to UMAP instead.

Dependencies
------------
  conda env: bro5 (see environment.yml)
  Extra (pip install if missing):
    scikit-learn-extra   # K-Medoids
    seaborn              # correlation heatmap

Usage
-----
  python descriptor_clustering_2d.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
import umap

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("WARNING: seaborn not installed — descriptor heatmap will be skipped.")

try:
    from sklearn_extra.cluster import KMedoids
    HAS_KMEDOIDS = True
except ImportError:
    HAS_KMEDOIDS = False
    print(
        "WARNING: scikit-learn-extra not found — K-Medoids will be skipped.\n"
        "  Install with:  pip install scikit-learn-extra"
    )

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdMolDescriptors
RDLogger.DisableLog("rdApp.*")

warnings.filterwarnings("ignore", category=FutureWarning)


# ==========================================================
# USER CONFIG — edit this section
# ==========================================================

INPUT_CSV = (
    "/home/j4carmon/projects/Macrocycle/data/libraries/2026-01-29/"
    "canonicalized_master_macrocycles_2D_Descriptors_FINAL_20260129.csv"
)

# Output directories (created automatically)
RUN_TAG = "2026-04-01"
OUTPUT_DIR = f"/home/j4carmon/projects/Macrocycle/outputs/descriptor_clustering/{RUN_TAG}/csv"
FIG_DIR    = f"/home/j4carmon/projects/Macrocycle/outputs/descriptor_clustering/{RUN_TAG}/figures"

# Column names
SMILES_COL    = "Smiles"
SOURCE_COL    = "Source"
HIT_ID_COL    = "Hit_ID"
HIGHLIGHT_COL = "Highlight_ID"

# Descriptor columns present in the CSV.
# NOTE: cLogS and Aromatic Rings are NOT in the FINAL_20260129 CSV.
#   - Aromatic Rings will be computed from RDKit if AUGMENT_RDKIT = True.
#   - cLogS would require Mordred or ESOL; set AUGMENT_RDKIT=True to skip it
#     (it just adds Aromatic Rings for now).
DESC_COLS = [
    "Total Molweight",
    "cLogP",
    "H-Acceptors",
    "H-Donors",
    "Polar Surface Area",
    "Rotatable Bonds",
]

# Set True to add RDKit-computed Aromatic Rings as a 7th descriptor
AUGMENT_RDKIT = True

# Data condition to analyze:
#   "ALL"  — all sources (Library + Literature + 34_Hits + Hit)  ← recommended
#   "A"    — Literature + Hit
#   "B"    — Library + Hit
#   "C"    — Literature + Library + Hit
#   "D"    — Literature + 34_Hits + Hit
#   "E"    — Library + 34_Hits + Hit
DATA_CONDITION = "ALL"

# Preprocessing
SCALER        = "standard"   # "standard" (zero-mean unit-var) or "minmax" ([0,1])
IQR_CLIP      = True         # clip outliers to [Q1 - 3×IQR, Q3 + 3×IQR] before scaling
IQR_WHISKER   = 3.0          # multiplier for IQR clipping fence

# Distance metric for UMAP.
# "euclidean" uses the scaled descriptor matrix directly.
# Any other metric (e.g. "cosine", "manhattan") triggers precomputation of a
# full distance matrix, which is memory-intensive for 30k compounds.
METRIC = "euclidean"

# UMAP parameters (descriptor space)
UMAP_PARAMS = dict(
    n_neighbors  = 50,
    min_dist     = 0.10,
    n_components = 2,
    metric       = METRIC,       # overridden by precomputed matrix if METRIC != euclidean
    random_state = 42,
    init         = "random",     # more stable than "spectral" for large datasets
)

# HDBSCAN parameters
HDBSCAN_PARAMS = dict(
    min_cluster_size = 80,   # ~0.3% of 30k; increase for fewer, coarser clusters
    min_samples      = 20,   # lower = more points assigned, fewer noise
    metric           = "euclidean",  # always euclidean on UMAP 2D embeddings
    cluster_selection_method = "eom",  # "eom" (default) or "leaf" (more clusters)
    store_centers    = "medoid",  # store medoid for each cluster
)

# K-Medoids parameters
K_MEDOIDS      = 12    # number of clusters; run Fig E first to pick a good k
KMEDOIDS_INIT  = "k-medoids++"   # or "random"
KMEDOIDS_METHOD = "alternate"    # "alternate" (fast, O(n*k)) or "pam" (slow, O(n^2))
KMEDOIDS_MAX_ITER = 300

# K-selection diagnostic: range to scan (Fig E)
K_SCAN_RANGE = range(4, 25)

RANDOM_STATE = 42


# ==========================================================
# Condition filter
# ==========================================================
CONDITION_SOURCES = {
    "A":   {"Literature", "Hit"},
    "B":   {"Library", "Hit"},
    "C":   {"Literature", "Library", "Hit"},
    "ALL": {"Literature", "Library", "34_Hits", "Hit"},
    "D":   {"Literature", "34_Hits", "Hit"},
    "E":   {"Library", "34_Hits", "Hit"},
}


# ==========================================================
# Color / marker scheme  (matches v3 conventions)
# ==========================================================
COLOR_LITERATURE     = "#D0D0D0"   # light gray
COLOR_LIBRARY        = "#1F77B4"   # muted blue
COLOR_34HITS         = "#E41A1C"   # red
COLOR_BRAIN_6_4_4_13 = "#984EA3"   # purple
COLOR_NOISE          = "#CCCCCC"   # HDBSCAN noise points

COLOR_HITS = [
    "#FF7F00",   # orange
    "#377EB8",   # blue
    "#4DAF4A",   # green
    "#A65628",   # brown
    "#F781BF",   # pink
    "#FFFF33",   # yellow
    "#66C2A5",   # teal
    "#FC8D62",   # coral
]

HIGHLIGHT_MARKERS = {
    "Cyclosporin A":   ("*", "black"),
    "Hexapeptide":     ("s", "black"),
    "N-Me Hexapeptide": ("^", "black"),
}

HIGHLIGHT_MARKER_SIZE = 200
HIT_MARKER_SIZE       = 150
HITS34_MARKER_SIZE    = 60
BG_ALPHA_LIT          = 0.20
BG_ALPHA_LIB          = 0.15
BG_MARKER_SIZE        = 6


# ==========================================================
# Helpers
# ==========================================================
def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)


def norm_str(x) -> str:
    return "" if pd.isna(x) else str(x).strip()


def banner(step: int, title: str):
    bar = "=" * 70
    print(f"\n{bar}\nSTEP {step} — {title}\n{bar}")


# ==========================================================
# Step 1: Load data and apply condition filter
# ==========================================================
def load_and_filter(csv_path: str, condition: str) -> pd.DataFrame:
    banner(1, f"Load data  |  Condition: {condition}")

    df = pd.read_csv(csv_path, index_col=0)
    print(f"  Loaded:  {len(df):,} rows × {df.shape[1]} columns")

    # Normalise string columns
    df[SOURCE_COL]    = df[SOURCE_COL].apply(norm_str)
    df[HIT_ID_COL]    = df[HIT_ID_COL].apply(norm_str)
    df[HIGHLIGHT_COL] = df[HIGHLIGHT_COL].apply(norm_str)

    # Add boolean flags used throughout
    df["_is_hit"]      = df[HIT_ID_COL] != ""
    df["_is_34hit"]    = df[SOURCE_COL] == "34_Hits"
    df["_is_highlight"] = df[HIGHLIGHT_COL] != ""

    allowed = CONDITION_SOURCES[condition]
    df_filt = df[df[SOURCE_COL].isin(allowed)].copy().reset_index(drop=True)

    print(f"  After filter '{condition}':  {len(df_filt):,} rows")
    print(f"  Source breakdown:\n{df_filt[SOURCE_COL].value_counts().to_string()}")
    print(f"  Hits (Hit_ID):     {df_filt['_is_hit'].sum()}")
    print(f"  34_Hits:           {df_filt['_is_34hit'].sum()}")
    print(f"  Highlights:        {df_filt['_is_highlight'].sum()}")
    return df_filt


# ==========================================================
# Step 2: RDKit augmentation (Aromatic Rings)
# ==========================================================
def augment_with_rdkit(df: pd.DataFrame) -> pd.DataFrame:
    banner(2, "RDKit augmentation — Aromatic Rings")
    col = "Aromatic Rings"
    vals = []
    failed = 0
    for smi in df[SMILES_COL]:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError("parse failed")
            vals.append(rdMolDescriptors.CalcNumAromaticRings(mol))
        except Exception:
            vals.append(np.nan)
            failed += 1
    df[col] = vals
    print(f"  Added '{col}' — {failed} RDKit parse failures (set to NaN)")
    return df


# ==========================================================
# Step 3: Preprocess descriptors
# ==========================================================
def preprocess_descriptors(
    df: pd.DataFrame,
    desc_cols: list[str],
    iqr_clip: bool,
    iqr_whisker: float,
    scaler_type: str,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """
    Returns
    -------
    X_scaled  : (n, d) float array ready for UMAP
    df_clean  : DataFrame with rows that passed the NaN filter
    desc_cols : final list of descriptor columns used
    """
    banner(3, "Preprocess descriptors")

    # Check all columns exist
    missing = [c for c in desc_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Descriptor columns missing from DataFrame: {missing}")

    # Drop rows with any NaN in descriptor columns
    n_before = len(df)
    df_clean = df.dropna(subset=desc_cols).copy().reset_index(drop=True)
    n_dropped = n_before - len(df_clean)
    print(f"  Rows with NaN in descriptors dropped: {n_dropped:,}")

    # Sanity check: hits and highlights must survive
    for label, flag in [("Hits", "_is_hit"), ("34_Hits", "_is_34hit"), ("Highlights", "_is_highlight")]:
        n_before_flag = df[flag].sum()
        n_after_flag  = df_clean[flag].sum()
        if n_after_flag < n_before_flag:
            print(f"  WARNING: {label} lost {n_before_flag - n_after_flag} rows due to NaN!")

    X = df_clean[desc_cols].values.astype(float)

    # --- IQR clipping (per-feature) ---
    if iqr_clip:
        Q1  = np.percentile(X, 25, axis=0)
        Q3  = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        lo  = Q1 - iqr_whisker * IQR
        hi  = Q3 + iqr_whisker * IQR
        X_clipped = np.clip(X, lo, hi)
        n_clipped = np.sum((X < lo) | (X > hi))
        print(f"  IQR clipping (whisker={iqr_whisker}×IQR): {n_clipped:,} values clipped")
        # Print per-feature clip stats
        for i, col in enumerate(desc_cols):
            n_lo = np.sum(X[:, i] < lo[i])
            n_hi = np.sum(X[:, i] > hi[i])
            if n_lo + n_hi > 0:
                print(f"    {col}: {n_lo} below fence, {n_hi} above fence")
        X = X_clipped

    # --- Scaling ---
    if scaler_type == "standard":
        scaler = StandardScaler()
        label  = "StandardScaler (zero-mean, unit-var)"
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
        label  = "MinMaxScaler [0, 1]"
    else:
        raise ValueError(f"Unknown scaler: {scaler_type!r}. Use 'standard' or 'minmax'.")

    X_scaled = scaler.fit_transform(X)
    print(f"  Scaler: {label}")
    print(f"  Final descriptor matrix: {X_scaled.shape[0]:,} compounds × {X_scaled.shape[1]} features")

    # Report descriptor statistics (pre-scale)
    stats = df_clean[desc_cols].describe().T[["mean", "std", "min", "max"]]
    print("\n  Descriptor statistics (before scaling):")
    print(stats.to_string())

    return X_scaled, df_clean, desc_cols


# ==========================================================
# Step 4: Optional distance matrix (non-Euclidean metric)
# ==========================================================
def maybe_precompute_distance_matrix(
    X_scaled: np.ndarray, metric: str
) -> tuple[np.ndarray | None, str]:
    """
    If metric != "euclidean", compute a full pairwise distance matrix and return
    it with metric="precomputed" for UMAP.
    NOTE: For 30k compounds this allocates ~7 GB (float64). Use euclidean for
    large datasets unless you have sufficient RAM.
    """
    if metric == "euclidean":
        return X_scaled, "euclidean"

    banner(4, f"Precompute {metric} distance matrix")
    n = X_scaled.shape[0]
    estimated_gb = n * n * 8 / 1e9
    print(f"  Compounds: {n:,} → matrix size: {estimated_gb:.1f} GB")
    if estimated_gb > 8:
        print(f"  WARNING: {estimated_gb:.1f} GB may exceed available RAM.")
        print("  Consider using metric='euclidean' for large datasets.")

    D = squareform(pdist(X_scaled, metric=metric))
    print(f"  Distance matrix computed: {D.shape}")
    return D, "precomputed"


# ==========================================================
# Step 5: UMAP
# ==========================================================
def run_umap(
    X_or_D: np.ndarray,
    umap_metric: str,
    umap_params: dict,
) -> np.ndarray:
    banner(5, "UMAP dimensionality reduction")

    params = {**umap_params, "metric": umap_metric}
    print(f"  Parameters: {params}")

    reducer = umap.UMAP(**params)
    embedding = reducer.fit_transform(X_or_D)

    print(f"  Embedding shape: {embedding.shape}")
    return embedding


# ==========================================================
# Step 6: HDBSCAN clustering on UMAP embeddings
# ==========================================================
def run_hdbscan(
    embedding: np.ndarray,
    hdbscan_params: dict,
) -> np.ndarray:
    banner(6, "HDBSCAN clustering on UMAP embeddings")
    print(f"  Parameters: {hdbscan_params}")

    clusterer = HDBSCAN(**hdbscan_params)
    labels    = clusterer.fit_predict(embedding)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = np.sum(labels == -1)
    print(f"  Clusters found:  {n_clusters}")
    print(f"  Noise points:    {n_noise:,} ({100*n_noise/len(labels):.1f}%)")
    for lbl in sorted(set(labels)):
        if lbl == -1:
            continue
        print(f"    Cluster {lbl:3d}: {np.sum(labels==lbl):,} compounds")

    return labels


# ==========================================================
# Step 7: K-Medoids clustering on UMAP embeddings
# ==========================================================
def run_kmedoids(
    embedding: np.ndarray,
    k: int,
    init: str,
    method: str,
    max_iter: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (labels, medoid_indices)."""
    banner(7, f"K-Medoids clustering on UMAP embeddings  (k={k})")

    if not HAS_KMEDOIDS:
        print("  SKIPPED — scikit-learn-extra not installed.")
        print("  Run:  pip install scikit-learn-extra")
        return np.full(len(embedding), -1), np.array([])

    km = KMedoids(
        n_clusters   = k,
        init         = init,
        method       = method,
        max_iter     = max_iter,
        random_state = random_state,
        metric       = "euclidean",
    )
    labels = km.fit_predict(embedding)
    medoid_indices = km.medoid_indices_
    inertia = km.inertia_

    print(f"  Inertia (sum of distances to medoid): {inertia:.4f}")
    print(f"  Medoid indices (row positions in df_clean):")
    for i, idx in enumerate(medoid_indices):
        print(f"    Cluster {i:3d}  →  row {idx}")

    return labels, medoid_indices


def k_selection_scan(
    embedding: np.ndarray,
    k_range,
    method: str,
    init: str,
    max_iter: int,
    random_state: int,
) -> dict:
    """Compute silhouette scores for a range of k values."""
    banner("E", "K-selection scan (silhouette scores)")

    if not HAS_KMEDOIDS:
        print("  SKIPPED — scikit-learn-extra not installed.")
        return {}

    results = {}
    for k in k_range:
        km = KMedoids(
            n_clusters   = k,
            init         = init,
            method       = method,
            max_iter     = max_iter,
            random_state = random_state,
            metric       = "euclidean",
        )
        lbl = km.fit_predict(embedding)
        sil = silhouette_score(embedding, lbl, metric="euclidean", sample_size=5000,
                               random_state=random_state)
        results[k] = sil
        print(f"  k={k:3d}  silhouette={sil:.4f}")

    best_k = max(results, key=results.get)
    print(f"\n  Best k by silhouette: {best_k}  (score={results[best_k]:.4f})")
    print(f"  NOTE: silhouette favors compact clusters. Inspect Fig E visually too.")
    return results


# ==========================================================
# Plotting helpers
# ==========================================================
def _build_hit_color_map(df: pd.DataFrame) -> dict:
    """Map Hit_ID → hex color string. Brain 6-4-4-13 gets purple."""
    color_map = {}
    hits = df[df["_is_hit"]][HIT_ID_COL].unique()
    regular = []
    for h in hits:
        h = norm_str(h)
        if not h:
            continue
        if "brain 6-4-4-13" in h.lower():
            color_map[h] = COLOR_BRAIN_6_4_4_13
        else:
            regular.append(h)
    for i, h in enumerate(regular):
        color_map[h] = COLOR_HITS[i % len(COLOR_HITS)]
    return color_map


def _scatter_background(ax, df, xcol, ycol):
    """Draw Literature (gray) and Library (blue) background points."""
    lit = df[df[SOURCE_COL] == "Literature"]
    lib = df[df[SOURCE_COL] == "Library"]
    if not lit.empty:
        ax.scatter(lit[xcol], lit[ycol], s=BG_MARKER_SIZE, alpha=BG_ALPHA_LIT,
                   c=COLOR_LITERATURE, rasterized=True, label="Literature")
    if not lib.empty:
        ax.scatter(lib[xcol], lib[ycol], s=BG_MARKER_SIZE, alpha=BG_ALPHA_LIB,
                   c=COLOR_LIBRARY, rasterized=True, label="Library")


def _scatter_foreground(ax, df, xcol, ycol, hit_color_map):
    """Draw 34_Hits, library hits, and reference highlights on top."""
    # 34_Hits — red dots
    hits34 = df[df["_is_34hit"] & ~df["_is_hit"]]
    if not hits34.empty:
        ax.scatter(hits34[xcol], hits34[ycol], s=HITS34_MARKER_SIZE, alpha=0.85,
                   c=COLOR_34HITS, edgecolors="darkred", linewidths=0.6,
                   zorder=6, label="34_Hits")

    # Library hits — X markers, colored per Hit_ID
    for hid, grp in df[df["_is_hit"]].groupby(HIT_ID_COL):
        hid = norm_str(hid)
        if not hid:
            continue
        color = hit_color_map.get(hid, "#000000")
        ax.scatter(grp[xcol], grp[ycol], s=HIT_MARKER_SIZE, marker="X",
                   alpha=0.95, c=color, edgecolors="black", linewidths=1.2,
                   zorder=7, label=f"Hit: {hid}")

    # Highlights — distinct black shapes
    for hlid, grp in df[df["_is_highlight"]].groupby(HIGHLIGHT_COL):
        hlid = norm_str(hlid)
        if not hlid:
            continue
        marker, color = HIGHLIGHT_MARKERS.get(hlid, ("D", "black"))
        ax.scatter(grp[xcol], grp[ycol], s=HIGHLIGHT_MARKER_SIZE, marker=marker,
                   alpha=0.90, c=color, edgecolors="black", linewidths=1.5,
                   zorder=8, label=f"Highlight: {hlid}")


def _style_ax(ax, title, xcol="UMAP-1", ycol="UMAP-2"):
    ax.set_xlabel(f"{xcol} →", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"{ycol} →", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.grid(True, alpha=0.18, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92, fancybox=True,
              shadow=True)


def _add_zoom_inset(ax, df, xcol, ycol, hit_color_map):
    """Inset showing the region around hits + 34_Hits."""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.patches import Rectangle

    interest = df[df["_is_hit"] | df["_is_34hit"] | df["_is_highlight"]]
    if interest.empty:
        return

    xm, xx = interest[xcol].min(), interest[xcol].max()
    ym, yx = interest[ycol].min(), interest[ycol].max()
    dx = max((xx - xm) * 0.35, 0.8)
    dy = max((yx - ym) * 0.35, 0.8)
    xm -= dx; xx += dx
    ym -= dy; yx += dy

    axins = inset_axes(ax, width="35%", height="35%", loc="upper left",
                       borderpad=2.5)
    sub = df[(df[xcol].between(xm, xx)) & (df[ycol].between(ym, yx))]
    axins.scatter(
        sub.loc[sub[SOURCE_COL] == "Literature", xcol],
        sub.loc[sub[SOURCE_COL] == "Literature", ycol],
        s=20, alpha=0.30, c=COLOR_LITERATURE, rasterized=True,
    )
    axins.scatter(
        sub.loc[sub[SOURCE_COL] == "Library", xcol],
        sub.loc[sub[SOURCE_COL] == "Library", ycol],
        s=20, alpha=0.25, c=COLOR_LIBRARY, rasterized=True,
    )
    h34 = sub[sub["_is_34hit"] & ~sub["_is_hit"]]
    if not h34.empty:
        axins.scatter(h34[xcol], h34[ycol], s=80, alpha=0.9, c=COLOR_34HITS,
                      edgecolors="darkred", linewidths=0.8, zorder=6)
    for hid, grp in sub[sub["_is_hit"]].groupby(HIT_ID_COL):
        hid = norm_str(hid)
        axins.scatter(grp[xcol], grp[ycol], s=100, marker="X",
                      c=hit_color_map.get(hid, "#000000"),
                      edgecolors="black", linewidths=1.5, zorder=7)
    for hlid, grp in sub[sub["_is_highlight"]].groupby(HIGHLIGHT_COL):
        hlid = norm_str(hlid)
        marker, color = HIGHLIGHT_MARKERS.get(hlid, ("D", "black"))
        axins.scatter(grp[xcol], grp[ycol], s=140, marker=marker,
                      c=color, edgecolors="black", linewidths=1.5, zorder=8)

    axins.set_xlim(xm, xx); axins.set_ylim(ym, yx)
    axins.tick_params(labelsize=7)
    axins.grid(True, alpha=0.25, linestyle=":", linewidth=0.5)
    axins.text(0.95, 0.05, "ZOOM", transform=axins.transAxes, fontsize=9,
               fontweight="bold", ha="right", va="bottom",
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                         edgecolor="black", alpha=0.8))
    ax.add_patch(Rectangle((xm, ym), xx - xm, yx - ym,
                            fill=False, edgecolor="black", linewidth=1.5,
                            linestyle="--", zorder=10))


# ==========================================================
# Step 8A: Figure A — UMAP colored by source / hit type
# ==========================================================
def plot_umap_source(df: pd.DataFrame, hit_color_map: dict, condition: str,
                     desc_cols_used: list[str]):
    fig, ax = plt.subplots(figsize=(10, 8))

    _scatter_background(ax, df, "UMAP-1", "UMAP-2")
    _scatter_foreground(ax, df, "UMAP-1", "UMAP-2", hit_color_map)
    _add_zoom_inset(ax, df, "UMAP-1", "UMAP-2", hit_color_map)
    _style_ax(ax,
              title=(f"UMAP — 2D Descriptor Space  [{condition}]\n"
                     f"Features: {', '.join(desc_cols_used)}"))

    path = os.path.join(FIG_DIR, f"FigA_UMAP_source_{condition}.svg")
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ==========================================================
# Step 8B: Figure B — UMAP colored by HDBSCAN clusters
# ==========================================================
def plot_hdbscan_clusters(df: pd.DataFrame, hit_color_map: dict, condition: str):
    labels    = df["hdbscan_label"].values
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # Build cluster color palette
    cmap = plt.get_cmap("tab20", max(n_clusters, 1))
    clust_colors = {lbl: cmap(lbl) for lbl in range(n_clusters)}
    clust_colors[-1] = matplotlib.colors.to_rgba(COLOR_NOISE)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Background: cluster-colored points
    for lbl in sorted(set(labels)):
        mask = labels == lbl
        alpha = 0.10 if lbl == -1 else 0.25
        size  = BG_MARKER_SIZE if lbl >= 0 else 4
        lname = "Noise" if lbl == -1 else f"Cluster {lbl}"
        ax.scatter(df.loc[mask, "UMAP-1"], df.loc[mask, "UMAP-2"],
                   s=size, alpha=alpha, c=[clust_colors[lbl]],
                   rasterized=True, label=lname)

    # Overlay hits / highlights on top (same style as Fig A)
    _scatter_foreground(ax, df, "UMAP-1", "UMAP-2", hit_color_map)

    _style_ax(ax, title=(f"HDBSCAN Clusters on UMAP Embeddings  [{condition}]\n"
                         f"{n_clusters} clusters  |  "
                         f"min_cluster_size={HDBSCAN_PARAMS['min_cluster_size']}  "
                         f"min_samples={HDBSCAN_PARAMS['min_samples']}"))

    path = os.path.join(FIG_DIR, f"FigB_HDBSCAN_{condition}.svg")
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ==========================================================
# Step 8C: Figure C — UMAP colored by K-Medoids clusters
# ==========================================================
def plot_kmedoids_clusters(df: pd.DataFrame, hit_color_map: dict, condition: str,
                           medoid_indices: np.ndarray):
    if not HAS_KMEDOIDS:
        return
    labels    = df["kmedoids_label"].values
    n_clusters = K_MEDOIDS

    cmap = plt.get_cmap("tab20", max(n_clusters, 1))

    fig, ax = plt.subplots(figsize=(10, 8))

    for lbl in range(n_clusters):
        mask = labels == lbl
        ax.scatter(df.loc[mask, "UMAP-1"], df.loc[mask, "UMAP-2"],
                   s=BG_MARKER_SIZE, alpha=0.25, c=[cmap(lbl)],
                   rasterized=True, label=f"Cluster {lbl}")

    # Medoid markers — diamond outline
    if len(medoid_indices):
        med_x = df.iloc[medoid_indices]["UMAP-1"].values
        med_y = df.iloc[medoid_indices]["UMAP-2"].values
        ax.scatter(med_x, med_y, s=180, marker="D", c="white",
                   edgecolors="black", linewidths=1.8, zorder=9,
                   label="K-Medoid center")

    _scatter_foreground(ax, df, "UMAP-1", "UMAP-2", hit_color_map)

    _style_ax(ax, title=(f"K-Medoids (k={n_clusters}) on UMAP Embeddings  [{condition}]"))

    path = os.path.join(FIG_DIR, f"FigC_KMedoids_k{n_clusters}_{condition}.svg")
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ==========================================================
# Step 8D: Figure D — Descriptor correlation heatmap
# ==========================================================
def plot_descriptor_heatmap(df: pd.DataFrame, desc_cols: list[str], condition: str):
    if not HAS_SEABORN:
        return
    banner("D", "Descriptor correlation heatmap")

    corr = df[desc_cols].corr()
    fig, ax = plt.subplots(figsize=(len(desc_cols) + 1, len(desc_cols)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title(f"Pearson Correlation — Descriptors  [{condition}]",
                 fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"FigD_DescriptorCorrelation_{condition}.svg")
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ==========================================================
# Step 8E: Figure E — K-selection silhouette plot
# ==========================================================
def plot_k_selection(sil_results: dict, condition: str):
    if not sil_results:
        return
    k_vals = sorted(sil_results)
    sil_vals = [sil_results[k] for k in k_vals]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(k_vals, sil_vals, "o-", color="#1F77B4", linewidth=2, markersize=6)
    ax.axvline(K_MEDOIDS, color="red", linestyle="--", linewidth=1.5,
               label=f"K_MEDOIDS={K_MEDOIDS} (current)")
    ax.set_xlabel("k (number of clusters)", fontsize=11)
    ax.set_ylabel("Silhouette score", fontsize=11)
    ax.set_title(f"K-Medoids: K selection  [{condition}]", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"FigE_KSelection_{condition}.svg")
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ==========================================================
# Step 9: CSV exports
# ==========================================================
def export_csvs(df: pd.DataFrame, desc_cols: list[str], medoid_indices: np.ndarray,
                condition: str):
    banner(9, "Export CSVs")

    # 9A — full embedding + cluster labels
    export_cols = [SMILES_COL, SOURCE_COL, HIT_ID_COL, HIGHLIGHT_COL,
                   "UMAP-1", "UMAP-2", "hdbscan_label", "kmedoids_label"] + desc_cols
    export_cols = [c for c in export_cols if c in df.columns]
    path_emb = os.path.join(OUTPUT_DIR, f"embeddings_{condition}.csv")
    df[export_cols].to_csv(path_emb, index=False)
    print(f"  Embeddings CSV: {path_emb}")

    # 9B — HDBSCAN cluster summary
    rows = []
    for lbl in sorted(df["hdbscan_label"].unique()):
        sub = df[df["hdbscan_label"] == lbl]
        rows.append({
            "cluster":      lbl,
            "n_compounds":  len(sub),
            "pct_library":  100 * (sub[SOURCE_COL] == "Library").mean(),
            "pct_lit":      100 * (sub[SOURCE_COL] == "Literature").mean(),
            "n_34hits":     sub["_is_34hit"].sum(),
            "n_hits":       sub["_is_hit"].sum(),
            "n_highlights": sub["_is_highlight"].sum(),
            **{f"mean_{c}": sub[c].mean() for c in desc_cols if c in sub.columns},
        })
    path_hdb = os.path.join(OUTPUT_DIR, f"hdbscan_summary_{condition}.csv")
    pd.DataFrame(rows).to_csv(path_hdb, index=False)
    print(f"  HDBSCAN summary: {path_hdb}")

    # 9C — K-Medoids cluster summary + medoid info
    if HAS_KMEDOIDS and len(medoid_indices):
        rows_km = []
        for lbl in sorted(df["kmedoids_label"].unique()):
            sub    = df[df["kmedoids_label"] == lbl]
            med_df = df.iloc[medoid_indices[lbl]] if lbl < len(medoid_indices) else None
            rows_km.append({
                "cluster":        lbl,
                "n_compounds":    len(sub),
                "pct_library":    100 * (sub[SOURCE_COL] == "Library").mean(),
                "pct_lit":        100 * (sub[SOURCE_COL] == "Literature").mean(),
                "n_34hits":       sub["_is_34hit"].sum(),
                "n_hits":         sub["_is_hit"].sum(),
                "n_highlights":   sub["_is_highlight"].sum(),
                "medoid_smiles":  med_df[SMILES_COL] if med_df is not None else "",
                "medoid_source":  med_df[SOURCE_COL] if med_df is not None else "",
                **{f"mean_{c}": sub[c].mean() for c in desc_cols if c in sub.columns},
            })
        path_km = os.path.join(OUTPUT_DIR, f"kmedoids_summary_k{K_MEDOIDS}_{condition}.csv")
        pd.DataFrame(rows_km).to_csv(path_km, index=False)
        print(f"  K-Medoids summary: {path_km}")


# ==========================================================
# Main
# ==========================================================
def main():
    ensure_dirs()

    # ── 1. Load & filter ────────────────────────────────────
    df = load_and_filter(INPUT_CSV, DATA_CONDITION)

    # ── 2. RDKit augmentation ───────────────────────────────
    if AUGMENT_RDKIT:
        df = augment_with_rdkit(df)
        desc_cols_used = DESC_COLS + ["Aromatic Rings"]
    else:
        desc_cols_used = list(DESC_COLS)

    # ── 3. Preprocess ───────────────────────────────────────
    X_scaled, df_clean, desc_cols_used = preprocess_descriptors(
        df, desc_cols_used, IQR_CLIP, IQR_WHISKER, SCALER
    )

    # ── 4. Descriptor correlation heatmap (before UMAP) ─────
    plot_descriptor_heatmap(df_clean, desc_cols_used, DATA_CONDITION)

    # ── 5. Distance matrix (if non-Euclidean metric) ────────
    X_or_D, umap_metric = maybe_precompute_distance_matrix(X_scaled, METRIC)

    # ── 6. UMAP ─────────────────────────────────────────────
    embedding = run_umap(X_or_D, umap_metric, UMAP_PARAMS)
    df_clean["UMAP-1"] = embedding[:, 0]
    df_clean["UMAP-2"] = embedding[:, 1]

    # ── 7. HDBSCAN ──────────────────────────────────────────
    hdb_labels = run_hdbscan(embedding, HDBSCAN_PARAMS)
    df_clean["hdbscan_label"] = hdb_labels

    # ── 8. K-Medoids ────────────────────────────────────────
    km_labels, medoid_indices = run_kmedoids(
        embedding, K_MEDOIDS, KMEDOIDS_INIT, KMEDOIDS_METHOD,
        KMEDOIDS_MAX_ITER, RANDOM_STATE,
    )
    df_clean["kmedoids_label"] = km_labels

    # ── K-selection diagnostic (optional — comment out after picking k) ──
    sil_results = k_selection_scan(
        embedding, K_SCAN_RANGE, KMEDOIDS_METHOD, KMEDOIDS_INIT,
        KMEDOIDS_MAX_ITER, RANDOM_STATE,
    )
    plot_k_selection(sil_results, DATA_CONDITION)

    # ── 9. Figures ──────────────────────────────────────────
    banner(8, "Generate figures")
    hit_color_map = _build_hit_color_map(df_clean)
    print(f"  Hit color map: {hit_color_map}")

    plot_umap_source(df_clean, hit_color_map, DATA_CONDITION, desc_cols_used)
    plot_hdbscan_clusters(df_clean, hit_color_map, DATA_CONDITION)
    plot_kmedoids_clusters(df_clean, hit_color_map, DATA_CONDITION, medoid_indices)

    # ── 10. Export ──────────────────────────────────────────
    export_csvs(df_clean, desc_cols_used, medoid_indices, DATA_CONDITION)

    print("\n" + "=" * 70)
    print("DONE")
    print(f"  Figures: {FIG_DIR}")
    print(f"  CSVs:    {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
