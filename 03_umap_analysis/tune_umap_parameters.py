"""
tune_umap_parameters.py
UMAP/HDBSCAN Parameter Selection Report
========================================

Produces a reproducible, evidence-based ranking of UMAP parameter sets for
the Mordred and MAPchiral branches.  The goal is to identify the smallest
embedding dimensionality and simplest parameter set that yields stable,
high-quality clustering — not to maximise any single metric blindly.

Sweep grid (27 combinations per branch)
  n_components : [5, 10, 15]
  n_neighbors  : [20, 40, 60]
  min_dist     : [0.0, 0.1, 0.2]
Fixed: metric (cosine / minhash_distance), random_state

Metrics recorded
----------------
  DBCV                  density-based clustering validity (primary)
                        computed on the UMAP embedding (clustering space)
  avg_persistence       mean HDBSCAN cluster persistence (primary)
  noise_frac            fraction of noise points (penalised at extremes)
  n_clusters            number of non-noise clusters
  cluster_sizes         min / median / max cluster size
  mean_neighbor_ari     stability: mean ARI against settings differing
                        by exactly one parameter (primary)
  trustworthiness       nearest-neighbour preservation in original space
                        (secondary; computed on a random sample)

Composite score (per run, all metrics normalised min-max to [0,1])
  35%  DBCV
  25%  avg_persistence
  20%  mean_neighbor_ari
  10%  trustworthiness
  10%  sanity_score  (noise + cluster-structure penalties)

Caching
-------
Each UMAP embedding + HDBSCAN label array is written to CACHE_DIR.
Subsequent runs reload from cache, so the expensive sweep runs only once.
Delete the cache directory to force a full re-run.

Dependencies
------------
  Required  : umap-learn, scikit-learn, numba, pandas, matplotlib
  Required  : hdbscan  (pip install hdbscan)  — for DBCV
  Optional  : if hdbscan package unavailable DBCV falls back to NaN
              and the composite score is renormalised without it

Outputs
-------
  {branch}_sweep_results.csv
  {branch}_top_candidates.csv
  {branch}_selection_report.txt
  figures/{branch}_top{rank}_nc{n}_nn{n}_md{m}.svg   (top 3 only)
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
from sklearn.cluster import HDBSCAN
from sklearn.manifold import trustworthiness as sklearn_trustworthiness
from sklearn.metrics import adjusted_rand_score, pairwise_distances
import umap

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# DBCV from the hdbscan package (preferred); falls back gracefully if absent
try:
    from hdbscan.validity import validity_index as _hdbscan_validity
    HAS_DBCV = True
except ImportError:
    HAS_DBCV = False
    print("WARNING: 'hdbscan' package not found — DBCV will be NaN.\n"
          "  pip install hdbscan")


# ===========================================================================
# CONFIG
# ===========================================================================

_REPO_ROOT = Path(__file__).parent.parent

RUN_TAG    = "2026-04-06"
OUTPUT_DIR = _REPO_ROOT / "outputs" / "tuning" / RUN_TAG
CACHE_DIR  = OUTPUT_DIR / "cache"

MORDRED_SCALED_CSV = _REPO_ROOT / "outputs" / "mordred"   / RUN_TAG / "mordred_filtered_scaled.csv"
MAPC_FPS_NPY       = _REPO_ROOT / "outputs" / "mapchiral" / RUN_TAG / "mapchiral_fingerprints.npy"
MAPC_META_CSV      = _REPO_ROOT / "outputs" / "mapchiral" / RUN_TAG / "mapchiral_metadata.csv"

SMILES_COL = "Smiles"
SOURCE_COL = "Source"

# ------------------------------------------------------------------
# Sweep grid
# ------------------------------------------------------------------
SWEEP_N_COMPONENTS = [5, 10, 15]
SWEEP_N_NEIGHBORS  = [20, 40, 60]
SWEEP_MIN_DIST     = [0.0, 0.1, 0.2]

RANDOM_STATE = 42

# ------------------------------------------------------------------
# HDBSCAN (fixed across all runs)
# ------------------------------------------------------------------
HDBSCAN_MIN_CLUSTER_SIZE = 50
HDBSCAN_MIN_SAMPLES      = 10

# ------------------------------------------------------------------
# Trustworthiness: computed on a random sample to keep it tractable
# ------------------------------------------------------------------
N_TRUST_SAMPLE  = 2000
N_TRUST_NEIGHBORS = 10

# ------------------------------------------------------------------
# Composite score weights (must sum to 1.0)
# ------------------------------------------------------------------
W_DBCV        = 0.35
W_PERSISTENCE = 0.25
W_ARI         = 0.20
W_TRUST       = 0.10
W_SANITY      = 0.10

# ------------------------------------------------------------------
# Sanity / noise penalty bands
# ------------------------------------------------------------------
NOISE_IDEAL_LO = 0.05   # below this: suspiciously low noise
NOISE_IDEAL_HI = 0.40   # above this: elevated noise penalty
NOISE_HARD_HI  = 0.80   # above this: severe penalty

CLUSTER_MIN    = 3
CLUSTER_MAX    = 80

# ------------------------------------------------------------------
# Figure colors
# ------------------------------------------------------------------
DRAW_ORDER = ["Literature", "Library", "34_Hits", "Hit"]
SOURCE_STYLE = {
    "Literature": ("#D0D0D0", 4,  0.25),
    "Library":    ("#1F77B4", 4,  0.15),
    "34_Hits":    ("#E41A1C", 30, 0.90),
    "Hit":        ("#FF7F00", 40, 1.00),
}


# ===========================================================================
# MinHash distance
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


def _cache_key(branch: str, nc: int, nn: int, md: float, suffix: str = "") -> str:
    md_str = str(md).replace(".", "p")
    return f"{branch}{suffix}_nc{nc}_nn{nn}_md{md_str}"


def _cache_paths(key: str) -> tuple[Path, Path]:
    return CACHE_DIR / f"{key}_emb.npy", CACHE_DIR / f"{key}_lbl.npy"


def _minmax_norm(values: np.ndarray, invert: bool = False) -> np.ndarray:
    """Min-max normalise to [0,1]; invert=True for metrics where lower is better."""
    lo, hi = np.nanmin(values), np.nanmax(values)
    if hi == lo:
        return np.zeros_like(values, dtype=float)
    normed = (values - lo) / (hi - lo)
    return 1.0 - normed if invert else normed


# ===========================================================================
# LOAD DATA
# ===========================================================================

def load_mordred() -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"\n[Load] Mordred: {MORDRED_SCALED_CSV.name}")
    df   = pd.read_csv(MORDRED_SCALED_CSV, low_memory=False)
    meta = df[[c for c in [SMILES_COL, SOURCE_COL] if c in df.columns]].copy()
    # select only numeric columns — Mordred emits some string/bool cols that survive to CSV
    num_cols = df.select_dtypes(include="number").columns
    X    = df[num_cols].astype(float)
    print(f"   {X.shape[0]:,} × {X.shape[1]}")
    return meta, X


def load_mapchiral() -> tuple[pd.DataFrame, np.ndarray]:
    print(f"\n[Load] MAPchiral: {MAPC_FPS_NPY.name}")
    fps  = np.load(MAPC_FPS_NPY)
    meta = pd.read_csv(MAPC_META_CSV)
    if fps.shape[0] != len(meta):
        raise ValueError(f"Shape mismatch: {fps.shape[0]} fps vs {len(meta)} meta rows")
    print(f"   {fps.shape[0]:,} × {fps.shape[1]}  dtype={fps.dtype}")
    return meta, fps


# ===========================================================================
# SINGLE UMAP + HDBSCAN (with caching)
# ===========================================================================

def run_combination(
    X,
    branch: str,
    nc: int,
    nn: int,
    md: float,
    idx: int,
    total: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run UMAP (nc components) + HDBSCAN for one parameter set.
    Returns (embedding, labels, cluster_persistence).
    Results are cached to CACHE_DIR and reloaded on subsequent calls.
    """
    key              = _cache_key(branch, nc, nn, md)
    emb_path, lbl_path = _cache_paths(key)
    pers_path          = CACHE_DIR / f"{key}_pers.npy"

    if emb_path.exists() and lbl_path.exists():
        emb         = np.load(emb_path)
        labels      = np.load(lbl_path)
        persistence = np.load(pers_path) if pers_path.exists() else np.array([])
        n_cl        = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"   [{idx:>2}/{total}] nc={nc} nn={nn} md={md}  "
              f"→ (cache) clusters={n_cl}  noise={float((labels==-1).mean()):.3f}")
        return emb, labels, persistence

    params = dict(n_components=nc, n_neighbors=nn, min_dist=md, random_state=RANDOM_STATE)
    if branch == "mapchiral":
        params["metric"] = minhash_distance
    else:
        params["metric"] = "cosine"

    t0  = time.time()
    emb = umap.UMAP(**params).fit_transform(
        X if isinstance(X, np.ndarray) else X.values
    )

    clusterer = HDBSCAN(
        min_cluster_size = HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples      = HDBSCAN_MIN_SAMPLES,
        metric           = "euclidean",
    )
    clusterer.fit(emb)
    labels      = clusterer.labels_
    persistence = getattr(clusterer, "cluster_persistence_", np.array([]))

    np.save(emb_path, emb)
    np.save(lbl_path, labels)
    np.save(pers_path, persistence)

    n_cl = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"   [{idx:>2}/{total}] nc={nc} nn={nn} md={md}  "
          f"→ {_elapsed(t0)}  clusters={n_cl}  noise={float((labels==-1).mean()):.3f}")
    return emb, labels, persistence


# ===========================================================================
# DBCV
# ===========================================================================

def compute_dbcv(embedding: np.ndarray, labels: np.ndarray) -> float:
    """
    DBCV (Density-Based Clustering Validation) on the UMAP embedding.
    Computed in the clustering space (Euclidean on embedding), not original space.
    Requires the 'hdbscan' package.  Returns NaN if unavailable.
    """
    if not HAS_DBCV:
        return float("nan")

    # validity_index requires at least 2 clusters and some non-noise points
    non_noise = labels != -1
    if non_noise.sum() < 10 or len(set(labels[non_noise])) < 2:
        return float("nan")

    try:
        return float(_hdbscan_validity(embedding, labels, metric="euclidean"))
    except Exception:
        return float("nan")


# ===========================================================================
# TRUSTWORTHINESS (sampled)
# ===========================================================================

def compute_trustworthiness(
    X,
    embedding: np.ndarray,
    branch: str,
    nc: int,
    nn: int,
    md: float,
) -> float:
    """
    Nearest-neighbour preservation between original space and UMAP embedding.
    Computed on a random sample of N_TRUST_SAMPLE molecules for efficiency.

    Original-space distances:
      Mordred   — cosine (on the scaled descriptor matrix)
      MAPchiral — minhash_distance (pairwise on the sampled fingerprints)

    The pairwise distance matrix is passed to sklearn's trustworthiness as
    'precomputed' so any metric is supported.
    """
    key      = _cache_key(branch + "_trust", nc, nn, md)
    t_path   = CACHE_DIR / f"{key}.npy"

    if t_path.exists():
        return float(np.load(t_path))

    n       = len(embedding)
    rng     = np.random.default_rng(RANDOM_STATE)
    idx     = rng.choice(n, min(N_TRUST_SAMPLE, n), replace=False)
    emb_s   = embedding[idx]

    if isinstance(X, np.ndarray):
        X_s = X[idx]
    else:
        X_s = X.iloc[idx].values

    try:
        metric = minhash_distance if branch == "mapchiral" else "cosine"
        D_orig = pairwise_distances(X_s, metric=metric)
        trust  = float(sklearn_trustworthiness(
            D_orig, emb_s,
            n_neighbors = N_TRUST_NEIGHBORS,
            metric      = "precomputed",
        ))
    except Exception:
        trust = float("nan")

    np.save(t_path, np.array([trust]))
    return trust


# ===========================================================================
# SANITY SCORE (noise + cluster structure penalties)
# ===========================================================================

def compute_sanity_score(
    noise_frac: float,
    n_clusters: int,
    min_size: int,
    max_size: int,
) -> float:
    """
    Returns a score in [0, 1] penalising pathological clustering solutions.

    Penalties applied:
      Noise fraction:
        < NOISE_IDEAL_LO  (-0.30)  near-zero noise: suspicious over-clustering
        > NOISE_HARD_HI   (-0.50)  most points are noise
        > NOISE_IDEAL_HI  (-0.20)  elevated noise (partial)
      Cluster count:
        < CLUSTER_MIN     (-0.30)  too few clusters
        > CLUSTER_MAX     (-0.20)  excessive fragmentation
      Size imbalance (max/min ratio):
        > 100             (-0.20)  extreme imbalance: one giant + tiny fragments
        > 50              (-0.10)  moderate imbalance
    """
    score = 1.0

    if noise_frac < NOISE_IDEAL_LO:
        score -= 0.30
    elif noise_frac > NOISE_HARD_HI:
        score -= 0.50
    elif noise_frac > NOISE_IDEAL_HI:
        score -= 0.20

    if n_clusters < CLUSTER_MIN:
        score -= 0.30
    elif n_clusters > CLUSTER_MAX:
        score -= 0.20

    if n_clusters > 0 and min_size > 0:
        ratio = max_size / max(min_size, 1)
        if ratio > 100:
            score -= 0.20
        elif ratio > 50:
            score -= 0.10

    return float(max(0.0, score))


# ===========================================================================
# FULL SWEEP
# ===========================================================================

def run_sweep(X, branch: str) -> dict:
    """Run all 27 combinations for one branch. Returns dict keyed by (nc,nn,md)."""
    combos = list(product(SWEEP_N_COMPONENTS, SWEEP_N_NEIGHBORS, SWEEP_MIN_DIST))
    total  = len(combos)

    print(f"\n{'='*62}")
    print(f"  Sweep: {branch.upper()}  ({total} combinations)")
    metric_label = "minhash_distance" if branch == "mapchiral" else "cosine"
    print(f"  Metric: {metric_label}  |  HDBSCAN min_cluster_size={HDBSCAN_MIN_CLUSTER_SIZE}")
    print(f"{'='*62}")

    results = {}
    for i, (nc, nn, md) in enumerate(combos, 1):
        emb, labels, persistence = run_combination(X, branch, nc, nn, md, i, total)

        non_noise  = labels[labels != -1]
        sizes      = sorted([int((labels == c).sum()) for c in set(labels) if c != -1])
        n_cl       = len(sizes)
        nf         = float((labels == -1).sum() / len(labels))
        avg_sz     = float(np.mean(sizes)) if sizes else 0.0
        med_sz     = float(np.median(sizes)) if sizes else 0.0
        min_sz     = int(sizes[0]) if sizes else 0
        max_sz     = int(sizes[-1]) if sizes else 0
        avg_pers   = float(persistence.mean()) if len(persistence) > 0 else 0.0

        var_per_dim = emb.var(axis=0)
        var_ratio   = float(var_per_dim[0] / var_per_dim.sum()) if var_per_dim.sum() > 0 else float("nan")

        results[(nc, nn, md)] = {
            "embedding":    emb,
            "labels":       labels,
            "n_clusters":   n_cl,
            "noise_frac":   nf,
            "avg_size":     avg_sz,
            "med_size":     med_sz,
            "min_size":     min_sz,
            "max_size":     max_sz,
            "avg_persistence": avg_pers,
            "var_ratio":    var_ratio,
        }

    # DBCV + trustworthiness pass (logged separately — can be slow)
    print(f"\n  Computing DBCV and trustworthiness for {total} runs ...")
    for i, (nc, nn, md) in enumerate(combos, 1):
        r   = results[(nc, nn, md)]
        emb = r["embedding"]

        dbcv  = compute_dbcv(emb, r["labels"])
        trust = compute_trustworthiness(X, emb, branch, nc, nn, md)

        r["dbcv"]            = dbcv
        r["trustworthiness"] = trust
        print(f"   [{i:>2}/{total}] nc={nc} nn={nn} md={md}  "
              f"DBCV={dbcv:+.3f}  trust={trust:.3f}")

    return results


# ===========================================================================
# STABILITY (neighbor ARI)
# ===========================================================================

def compute_stability(results: dict) -> dict:
    """Mean ARI against all settings differing in exactly one parameter."""
    keys = list(results.keys())
    for k0 in keys:
        nc0, nn0, md0 = k0
        aris = [
            adjusted_rand_score(results[k0]["labels"], results[k1]["labels"])
            for k1 in keys
            if sum([nc0 != k1[0], nn0 != k1[1], md0 != k1[2]]) == 1
        ]
        results[k0]["mean_neighbor_ari"] = float(np.mean(aris)) if aris else float("nan")
    return results


# ===========================================================================
# COMPOSITE SCORING
# ===========================================================================

def score_all(results: dict) -> dict:
    """
    Normalise each metric min-max across all 27 runs, then apply weights.
    Metrics where lower is better are inverted before normalisation.
    The sanity score is computed independently (already in [0,1]).
    """
    keys = sorted(results.keys())

    def _arr(field):
        return np.array([results[k].get(field, float("nan")) for k in keys])

    dbcv_arr   = _arr("dbcv")
    pers_arr   = _arr("avg_persistence")
    ari_arr    = _arr("mean_neighbor_ari")
    trust_arr  = _arr("trustworthiness")

    # Min-max normalise (higher raw value → higher normalised score)
    dbcv_n  = _minmax_norm(dbcv_arr)
    pers_n  = _minmax_norm(pers_arr)
    ari_n   = _minmax_norm(ari_arr)
    trust_n = _minmax_norm(trust_arr)

    # Adjust weights if DBCV unavailable
    all_dbcv_nan = np.all(np.isnan(dbcv_arr))
    if all_dbcv_nan:
        w_dbcv, w_pers, w_ari, w_trust, w_san = 0.00, 0.40, 0.30, 0.15, 0.15
        dbcv_n = np.zeros(len(keys))
    else:
        w_dbcv, w_pers, w_ari, w_trust, w_san = W_DBCV, W_PERSISTENCE, W_ARI, W_TRUST, W_SANITY

    for i, k in enumerate(keys):
        r     = results[k]
        san   = compute_sanity_score(
            r["noise_frac"], r["n_clusters"], r["min_size"], r["max_size"]
        )
        score = (
            w_dbcv  * (0.0 if np.isnan(dbcv_n[i])  else dbcv_n[i])
            + w_pers  * (0.0 if np.isnan(pers_n[i])  else pers_n[i])
            + w_ari   * (0.0 if np.isnan(ari_n[i])   else ari_n[i])
            + w_trust * (0.0 if np.isnan(trust_n[i]) else trust_n[i])
            + w_san   * san
        )
        results[k]["sanity_score"]    = san
        results[k]["composite_score"] = score
        results[k]["norm_dbcv"]       = float(dbcv_n[i])
        results[k]["norm_persistence"]= float(pers_n[i])
        results[k]["norm_ari"]        = float(ari_n[i])
        results[k]["norm_trust"]      = float(trust_n[i])

    return results


# ===========================================================================
# RANKING
# ===========================================================================

def build_summary_df(results: dict, branch: str) -> pd.DataFrame:
    metric_label = "minhash_distance" if branch == "mapchiral" else "cosine"
    rows = []
    for (nc, nn, md), r in results.items():
        rows.append({
            "branch":            branch,
            "n_components":      nc,
            "n_neighbors":       nn,
            "min_dist":          md,
            "metric":            metric_label,
            "n_clusters":        r["n_clusters"],
            "noise_frac":        round(r["noise_frac"], 4),
            "avg_cluster_size":  round(r["avg_size"], 1),
            "med_cluster_size":  round(r["med_size"], 1),
            "min_cluster_size":  r["min_size"],
            "max_cluster_size":  r["max_size"],
            "dbcv":              round(r.get("dbcv", float("nan")), 4),
            "avg_persistence":   round(r.get("avg_persistence", 0.0), 4),
            "mean_neighbor_ari": round(r.get("mean_neighbor_ari", float("nan")), 4),
            "trustworthiness":   round(r.get("trustworthiness", float("nan")), 4),
            "sanity_score":      round(r.get("sanity_score", 0.0), 4),
            "composite_score":   round(r.get("composite_score", 0.0), 4),
        })

    df = (
        pd.DataFrame(rows)
        .sort_values("composite_score", ascending=False)
        .reset_index(drop=True)
    )
    df.insert(0, "rank", df.index + 1)
    return df


def select_top3(df: pd.DataFrame) -> list[dict]:
    """
    From the ranked table, return top 3.
    Among candidates within 0.03 composite score of the top, prefer lower n_components.
    """
    best_score = df["composite_score"].iloc[0]
    close      = df[df["composite_score"] >= best_score - 0.03].copy()
    close      = close.sort_values(
        ["composite_score", "n_components"], ascending=[False, True]
    ).reset_index(drop=True)

    top3 = []
    seen_nc = set()
    for _, row in close.iterrows():
        if len(top3) >= 3:
            break
        top3.append(row.to_dict())

    # Fill to 3 from remaining if needed
    if len(top3) < 3:
        for _, row in df.iterrows():
            if len(top3) >= 3:
                break
            if row["rank"] not in [r["rank"] for r in top3]:
                top3.append(row.to_dict())

    return top3[:3]


def _explain_winner(candidate: dict, df: pd.DataFrame) -> str:
    parts = []
    nc    = candidate["n_components"]

    parts.append(
        f"composite score {candidate['composite_score']:.3f} "
        f"(rank {int(candidate['rank'])} of {len(df)})"
    )

    dbcv = candidate["dbcv"]
    if not np.isnan(dbcv):
        parts.append(
            f"DBCV {dbcv:+.3f} ({'above' if dbcv > 0 else 'below'} zero — "
            + ("clusters well-separated from noise" if dbcv > 0.1
               else "clusters marginally separated" if dbcv > 0
               else "clusters poorly separated in embedding") + ")"
        )

    ari = candidate["mean_neighbor_ari"]
    if not np.isnan(ari):
        parts.append(
            f"ARI stability {ari:.3f} — "
            + ("robust across nearby settings" if ari > 0.6
               else "moderate sensitivity to parameters" if ari > 0.3
               else "sensitive to parameter changes")
        )

    nf = candidate["noise_frac"]
    parts.append(
        f"noise fraction {nf:.3f} — "
        + ("healthy range" if NOISE_IDEAL_LO <= nf <= NOISE_IDEAL_HI
           else "slightly elevated" if nf <= NOISE_HARD_HI
           else "high — review HDBSCAN min_cluster_size")
    )

    parts.append(
        f"n_components={nc} selected as smallest dimensionality with these qualities"
    )
    return "; ".join(parts)


# ===========================================================================
# 2D UMAP FIGURES FOR TOP 3
# ===========================================================================

def _get_2d_embedding(X, branch: str, nn: int, md: float) -> np.ndarray:
    key              = _cache_key(branch + "_viz2d", 2, nn, md)
    emb_path, _      = _cache_paths(key)
    if emb_path.exists():
        return np.load(emb_path)
    params = dict(n_components=2, n_neighbors=nn, min_dist=md, random_state=RANDOM_STATE)
    params["metric"] = minhash_distance if branch == "mapchiral" else "cosine"
    t0  = time.time()
    emb = umap.UMAP(**params).fit_transform(
        X if isinstance(X, np.ndarray) else X.values
    )
    np.save(emb_path, emb)
    print(f"   2D viz UMAP (nn={nn}, md={md}): {_elapsed(t0)}")
    return emb


def generate_figures(
    X,
    meta: pd.DataFrame,
    results: dict,
    top3: list[dict],
    branch: str,
    fig_dir: Path,
) -> None:
    cmap    = plt.get_cmap("tab20")
    sources = meta[SOURCE_COL] if SOURCE_COL in meta.columns else None

    for i, cand in enumerate(top3, 1):
        nc, nn, md = int(cand["n_components"]), int(cand["n_neighbors"]), float(cand["min_dist"])
        labels     = results[(nc, nn, md)]["labels"]
        emb2d      = _get_2d_embedding(X, branch, nn, md)

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(
            f"{branch.upper()}  —  Rank {int(cand['rank'])}  "
            f"|  nc={nc}  nn={nn}  md={md}\n"
            f"clusters={cand['n_clusters']}  "
            f"noise={cand['noise_frac']:.3f}  "
            f"DBCV={cand['dbcv']:+.3f}  "
            f"ARI={cand['mean_neighbor_ari']:.3f}  "
            f"composite={cand['composite_score']:.3f}",
            fontsize=9,
        )

        # Left panel: source
        ax = axes[0]
        ax.set_facecolor("#F5F5F5")
        handles = []
        if sources is not None:
            for src in DRAW_ORDER:
                idx_s = np.where(sources == src)[0]
                if len(idx_s) == 0:
                    continue
                color, size, alpha = SOURCE_STYLE.get(src, ("#999999", 4, 0.3))
                ax.scatter(emb2d[idx_s, 0], emb2d[idx_s, 1],
                           s=size, c=color, alpha=alpha, linewidths=0, rasterized=True)
                handles.append(mpatches.Patch(color=color, label=f"{src} (n={len(idx_s):,})"))
        ax.legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.8)
        ax.set_title("Colored by source  (2D UMAP)", fontsize=9)
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.grid(True, alpha=0.3, linewidth=0.5)

        # Right panel: HDBSCAN clusters (labels from nD embedding)
        ax = axes[1]
        ax.set_facecolor("#F5F5F5")
        handles = []
        for lbl in sorted(set(labels)):
            idx_l = np.where(labels == lbl)[0]
            color = "#CCCCCC" if lbl == -1 else cmap(lbl % 20)
            name  = "Noise" if lbl == -1 else f"C{lbl}"
            ax.scatter(emb2d[idx_l, 0], emb2d[idx_l, 1],
                       s=4, c=[color], alpha=0.4, linewidths=0, rasterized=True)
            if lbl == -1 or lbl < 18:
                handles.append(mpatches.Patch(color=color, label=f"{name} ({len(idx_l):,})"))
        ax.legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.8,
                  ncol=max(1, len(handles) // 10))
        ax.set_title(f"HDBSCAN labels from {nc}D embedding  (2D viz)", fontsize=9)
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.grid(True, alpha=0.3, linewidth=0.5)

        fig.tight_layout()
        fname = f"{branch}_rank{i}_nc{nc}_nn{nn}_md{str(md).replace('.','p')}.svg"
        path  = fig_dir / fname
        fig.savefig(path, format="svg", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"   Saved: {path.name}")


# ===========================================================================
# OUTPUT: CSV
# ===========================================================================

def save_csv(df: pd.DataFrame, top3: list[dict], branch: str) -> None:
    df.to_csv(OUTPUT_DIR / f"{branch}_sweep_results.csv", index=False)
    pd.DataFrame(top3).to_csv(OUTPUT_DIR / f"{branch}_top_candidates.csv", index=False)
    print(f"   {branch}_sweep_results.csv  +  {branch}_top_candidates.csv")


# ===========================================================================
# OUTPUT: HUMAN-READABLE REPORT
# ===========================================================================

def write_report(
    branch: str,
    df: pd.DataFrame,
    top3: list[dict],
    results: dict,
    all_nan_dbcv: bool,
) -> None:
    selected = top3[0]
    metric   = "minhash_distance" if branch == "mapchiral" else "cosine"

    W = dict(dbcv=W_DBCV, persistence=W_PERSISTENCE, ari=W_ARI,
             trust=W_TRUST, sanity=W_SANITY)
    if all_nan_dbcv:
        W = dict(dbcv=0.0, persistence=0.40, ari=0.30, trust=0.15, sanity=0.15)

    lines = [
        "=" * 72,
        f"UMAP/HDBSCAN PARAMETER SELECTION REPORT — {branch.upper()}",
        "=" * 72,
        f"Run tag        : {RUN_TAG}",
        f"UMAP metric    : {metric}",
        f"Sweep grid     : n_components={SWEEP_N_COMPONENTS}",
        f"                 n_neighbors={SWEEP_N_NEIGHBORS}",
        f"                 min_dist={SWEEP_MIN_DIST}",
        f"HDBSCAN        : min_cluster_size={HDBSCAN_MIN_CLUSTER_SIZE}  "
        f"min_samples={HDBSCAN_MIN_SAMPLES}",
        "",
        "--- Scoring weights ---",
        f"  DBCV                : {W['dbcv']:.0%}"
        + ("  [NaN — hdbscan pkg missing; weight redistributed]" if all_nan_dbcv else ""),
        f"  Avg persistence     : {W['persistence']:.0%}",
        f"  Mean neighbor ARI   : {W['ari']:.0%}",
        f"  Trustworthiness     : {W['trust']:.0%}  (sampled, n={N_TRUST_SAMPLE})",
        f"  Sanity score        : {W['sanity']:.0%}  (noise + cluster structure penalties)",
        f"  All metrics min-max normalised across 27 runs before weighting",
        "",
        "=" * 72,
        "SELECTED PARAMETER SET",
        "=" * 72,
        f"  n_components : {int(selected['n_components'])}",
        f"  n_neighbors  : {int(selected['n_neighbors'])}",
        f"  min_dist     : {selected['min_dist']}",
        f"  metric       : {metric}",
        "",
        f"  Why this set won:",
        f"  {_explain_winner(selected, df)}",
        "",
        "=" * 72,
        "TOP 3 CANDIDATES",
        "=" * 72,
        f"{'Rank':>4} {'nc':>4} {'nn':>4} {'md':>5}  {'score':>6}  {'DBCV':>6}  "
        f"{'pers':>5}  {'ARI':>5}  {'trust':>5}  {'noise':>6}  {'n_cl':>5}",
        "-" * 72,
    ]
    for c in top3:
        lines.append(
            f"{int(c['rank']):>4} {int(c['n_components']):>4} {int(c['n_neighbors']):>4} "
            f"{c['min_dist']:>5.1f}  {c['composite_score']:>6.3f}  "
            f"{c.get('dbcv', float('nan')):>+6.3f}  "
            f"{c.get('avg_persistence', 0):>5.3f}  "
            f"{c.get('mean_neighbor_ari', float('nan')):>5.3f}  "
            f"{c.get('trustworthiness', float('nan')):>5.3f}  "
            f"{c['noise_frac']:>6.3f}  {int(c['n_clusters']):>5}"
        )

    lines += [
        "",
        "  Tradeoffs across top 3:",
    ]
    for c in top3:
        lines.append(f"    Rank {int(c['rank'])}: {_explain_winner(c, df)}")

    lines += [
        "",
        "=" * 72,
        "FULL SWEEP RESULTS (ranked by composite score)",
        "=" * 72,
        f"{'Rk':>3} {'nc':>4} {'nn':>4} {'md':>5}  {'score':>6}  {'DBCV':>6}  "
        f"{'pers':>5}  {'ARI':>5}  {'trust':>5}  {'noise':>6}  {'n_cl':>5}  "
        f"{'avg_sz':>7}",
        "-" * 72,
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{int(row['rank']):>3} {int(row['n_components']):>4} "
            f"{int(row['n_neighbors']):>4} {row['min_dist']:>5.1f}  "
            f"{row['composite_score']:>6.3f}  "
            f"{row.get('dbcv', float('nan')):>+6.3f}  "
            f"{row.get('avg_persistence', 0):>5.3f}  "
            f"{row.get('mean_neighbor_ari', float('nan')):>5.3f}  "
            f"{row.get('trustworthiness', float('nan')):>5.3f}  "
            f"{row['noise_frac']:>6.3f}  {int(row['n_clusters']):>5}  "
            f"{row['avg_cluster_size']:>7.1f}"
        )

    # Parameter sensitivity analysis
    lines += [
        "",
        "=" * 72,
        "PARAMETER SENSITIVITY ANALYSIS",
        "=" * 72,
        "",
        "--- Effect of n_components on composite score (mean across nn/md) ---",
    ]
    for nc in SWEEP_N_COMPONENTS:
        subset    = df[df["n_components"] == nc]
        mean_sc   = subset["composite_score"].mean()
        mean_ari  = subset["mean_neighbor_ari"].mean()
        mean_ncl  = subset["n_clusters"].mean()
        lines.append(
            f"  nc={nc:>2}:  composite={mean_sc:.3f}  "
            f"ARI={mean_ari:.3f}  avg_clusters={mean_ncl:.1f}"
        )

    lines += ["", "--- Effect of n_neighbors (mean across nc/md) ---"]
    for nn in SWEEP_N_NEIGHBORS:
        subset   = df[df["n_neighbors"] == nn]
        mean_sc  = subset["composite_score"].mean()
        mean_ncl = subset["n_clusters"].mean()
        mean_nf  = subset["noise_frac"].mean()
        lines.append(
            f"  nn={nn:>2}:  composite={mean_sc:.3f}  "
            f"avg_clusters={mean_ncl:.1f}  avg_noise={mean_nf:.3f}"
        )

    lines += ["", "--- Effect of min_dist (mean across nc/nn) ---"]
    for md in SWEEP_MIN_DIST:
        subset   = df[df["min_dist"] == md]
        mean_sc  = subset["composite_score"].mean()
        mean_ncl = subset["n_clusters"].mean()
        mean_nf  = subset["noise_frac"].mean()
        lines.append(
            f"  md={md}:  composite={mean_sc:.3f}  "
            f"avg_clusters={mean_ncl:.1f}  avg_noise={mean_nf:.3f}"
        )

    lines += [
        "",
        "=" * 72,
        "INTERPRETATION NOTES",
        "=" * 72,
        "",
        "  Clustering metrics (DBCV, persistence, ARI) are the primary drivers.",
        "  Trustworthiness is secondary: it measures embedding fidelity, not",
        "  clustering quality.  A beautiful embedding with poor DBCV is not",
        "  a good clustering result.",
        "",
        "  n_components selection principle:",
        "    Use the smallest n_components where clustering is stable (ARI stable)",
        "    and DBCV does not improve meaningfully with higher dimensionality.",
        "    2D UMAP is kept separately as a visualization tool only.",
        "",
        "  Noise fraction guidance:",
        f"    Ideal range    : [{NOISE_IDEAL_LO}, {NOISE_IDEAL_HI}]",
        f"    Above {NOISE_HARD_HI:.0%}     : most points are noise — "
        "consider larger min_cluster_size",
        f"    Below {NOISE_IDEAL_LO:.0%}    : suspicious — consider smaller min_cluster_size",
        "",
        "  ARI stability guidance:",
        "    > 0.6 : robust — cluster structure does not change with small param shifts",
        "    0.3–0.6 : moderate sensitivity — inspect neighboring runs",
        "    < 0.3 : sensitive — results may be unstable",
        "",
        "  DBCV interpretation:",
        "    > 0   : clusters are denser than the noise/inter-cluster space (good)",
        "    < 0   : cluster boundaries poorly defined in the embedding",
        "    NaN   : too few clusters or all-noise result",
        "",
        "=" * 72,
        "RECOMMENDED SETTINGS FOR analyse_chemical_space.py",
        "=" * 72,
        f"  Update UMAP_BASE_PARAMS['{branch}'] and the dimensionality sweep",
        f"  configuration with the selected settings above.",
        f"  Selected: nc={int(selected['n_components'])}  "
        f"nn={int(selected['n_neighbors'])}  md={selected['min_dist']}",
        "=" * 72,
    ]

    report_text = "\n".join(lines)
    print("\n" + report_text)
    path = OUTPUT_DIR / f"{branch}_selection_report.txt"
    with open(path, "w") as fh:
        fh.write(report_text + "\n")
    print(f"\n   Saved: {path.name}")


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = OUTPUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    n_combos = len(SWEEP_N_COMPONENTS) * len(SWEEP_N_NEIGHBORS) * len(SWEEP_MIN_DIST)

    print("=" * 72)
    print("UMAP/HDBSCAN PARAMETER SELECTION")
    print(f"  Branches    : Mordred (cosine), MAPchiral (minhash_distance)")
    print(f"  Combinations: {n_combos} per branch  ({2 * n_combos} total)")
    print(f"  DBCV        : {'available (hdbscan pkg)' if HAS_DBCV else 'UNAVAILABLE — pip install hdbscan'}")
    print(f"  Trust sample: {N_TRUST_SAMPLE} molecules")
    print(f"  Cache       : {CACHE_DIR}")
    print(f"  Output      : {OUTPUT_DIR}")
    print("=" * 72)

    t_total = time.time()

    for branch, load_fn in [("mordred", load_mordred), ("mapchiral", load_mapchiral)]:
        meta, X = load_fn()

        results = run_sweep(X, branch)
        results = compute_stability(results)
        results = score_all(results)

        df   = build_summary_df(results, branch)
        top3 = select_top3(df)

        print(f"\n[Figures] {branch} top 3 ...")
        generate_figures(X, meta, results, top3, branch, fig_dir)

        save_csv(df, top3, branch)
        all_nan_dbcv = df["dbcv"].isna().all()
        write_report(branch, df, top3, results, all_nan_dbcv)

    print(f"\n{'='*72}")
    print("FINAL RECOMMENDATIONS")
    print(f"{'='*72}")
    for branch in ["mordred", "mapchiral"]:
        df   = pd.read_csv(OUTPUT_DIR / f"{branch}_top_candidates.csv")
        best = df.iloc[0]
        print(f"  {branch.upper():12}: "
              f"nc={int(best['n_components'])}  "
              f"nn={int(best['n_neighbors'])}  "
              f"md={best['min_dist']}  "
              f"→ score={best['composite_score']:.3f}  "
              f"clusters={int(best['n_clusters'])}  "
              f"noise={best['noise_frac']:.3f}")

    print(f"\n[✓] Done — {_elapsed(t_total)}")


if __name__ == "__main__":
    main()
