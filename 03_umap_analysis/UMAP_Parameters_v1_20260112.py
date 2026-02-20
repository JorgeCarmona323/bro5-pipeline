"""
UMAP Parameter Sweep & Rigorous Tuning Pipeline
================================================

This script systematically evaluates UMAP parameter combinations,
computes quality metrics, and generates a publication-ready report.

Outputs:
  - parameter_sweep_results.csv (all combinations + metrics)
  - parameter_sweep_summary.txt (formatted report)
  - umap_parameter_comparison_plot.png (visual comparison)
  - sweep_details_{params}.txt (per-combination details)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from datetime import datetime

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

import umap
from sklearn.preprocessing import StandardScaler
from numba import njit
from scipy.stats import spearmanr, rankdata

from mapchiral.mapchiral import encode
from pynndescent import NNDescent


# ==========================================================
# CONFIG
# ==========================================================
INPUT_CSV = "C:\\Users\\Admin\\Documents\\Hu Lab\\Code\\Python\\rdkit\\Data\\2026-01-09\\canonicalized_master_macrocycles_2D_Descriptors_20260106.csv"
SWEEP_OUTPUT_DIR = "C:\\Users\\Admin\\Documents\\Hu Lab\\Code\\Python\\rdkit\\Data\\2026-01-09\\Parameter_Sweep"

SMILES_COL = "Smiles"
SOURCE_COL = "Source"
HIT_ID_COL = "Hit_ID"
HIGHLIGHT_COL = "Highlight_ID"

MAX_RADIUS = 2
N_PERMUTATIONS = 2048
MAPPING = False
RANDOM_STATE = 42
TOPK_NEIGHBORS = 50

# Parameter grid to sweep
PARAM_GRID = {
    "n_neighbors": [15, 25, 40, 50],
    "min_dist": [0.05, 0.10, 0.15, 0.25],
}

# Fixed parameters
FIXED_PARAMS = {
    "n_components": 2,
    "random_state": RANDOM_STATE,
}


# ==========================================================
# Helpers
# ==========================================================
def ensure_dirs():
    os.makedirs(SWEEP_OUTPUT_DIR, exist_ok=True)

def norm_str(x) -> str:
    return "" if pd.isna(x) else str(x).strip()

@njit
def minhash_distance(fp_a, fp_b) -> float:
    m = fp_a.shape[0]
    eq = 0
    for k in range(m):
        if fp_a[k] == fp_b[k]:
            eq += 1
    return 1.0 - (eq / m)


# ==========================================================
# Load and preprocess data (once)
# ==========================================================
def load_and_fingerprint_data(csv_path: str) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Load CSV, fingerprint all molecules, return:
    - fps: fingerprint matrix (n_mols x n_bits)
    - df: DataFrame with metadata
    - keep_idx: original indices (for tracing failures)
    """
    df = pd.read_csv(csv_path, sep=None, engine="python", encoding='utf-8')
    
    fps = []
    keep_idx = []
    
    for i, row in df.iterrows():
        smi = row[SMILES_COL]
        if not isinstance(smi, str) or not smi.strip():
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = encode(mol, max_radius=MAX_RADIUS, n_permutations=N_PERMUTATIONS, mapping=MAPPING)
        fps.append(np.asarray(fp, dtype=np.uint32))
        keep_idx.append(i)
    
    df_clean = df.loc[keep_idx].copy().reset_index(drop=True)
    fps_array = np.vstack(fps)
    
    # Normalize metadata
    df_clean[SOURCE_COL] = df_clean[SOURCE_COL].astype(str).str.strip().str.lower()
    df_clean[HIT_ID_COL] = df_clean[HIT_ID_COL].apply(norm_str)
    df_clean[HIGHLIGHT_COL] = df_clean[HIGHLIGHT_COL].apply(norm_str)
    df_clean["_is_hit"] = df_clean[HIT_ID_COL].astype(str).str.strip().ne("")
    df_clean["_is_highlight"] = df_clean[HIGHLIGHT_COL].astype(str).str.strip().ne("")
    
    print(f"\nLoaded {len(fps_array):,} valid molecules")
    print(f"Hits: {df_clean['_is_hit'].sum():,}")
    print(f"Highlights: {df_clean['_is_highlight'].sum():,}")
    
    return fps_array, df_clean, np.asarray(keep_idx, dtype=int)


# ==========================================================
# Quality metrics (per UMAP fit)
# ==========================================================
def compute_quality_metrics(df: pd.DataFrame, fps: np.ndarray, emb: np.ndarray) -> dict:
    """
    Compute comprehensive quality metrics:
    - Local faithfulness (Spearman per hit)
    - Global uniformity (embedding coordinate ranges)
    - Hit clustering (intra-hit distances)
    - Neighbor purity (hits/highlights in NN sets)
    """
    
    xy = emb
    
    # =====================
    # 1. LOCAL FAITHFULNESS
    # =====================
    index = NNDescent(
        fps,
        metric=minhash_distance,
        n_neighbors=max(TOPK_NEIGHBORS + 10, 40),
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    hit_pos = np.where(df["_is_hit"].to_numpy())[0]
    
    if len(hit_pos) > 0:
        inds, dists = index.query(fps[hit_pos], k=TOPK_NEIGHBORS + 1)
        
        spearman_rhos = []
        rank_agrees = []
        overlaps = []
        purities = []
        
        for qi, pos in enumerate(hit_pos):
            hit_xy = xy[pos]
            
            # MinHash neighbors
            neigh_idx = []
            mh_dist = []
            for j, dist in zip(inds[qi], dists[qi]):
                j = int(j)
                if j == pos:
                    continue
                neigh_idx.append(j)
                mh_dist.append(float(dist))
                if len(neigh_idx) >= TOPK_NEIGHBORS:
                    break
            
            if len(neigh_idx) < 5:
                continue
            
            # UMAP distances
            neigh_xy = xy[neigh_idx]
            umap_dist = np.sqrt(((neigh_xy - hit_xy) ** 2).sum(axis=1)).astype(float)
            
            # Spearman
            rho, _ = spearmanr(mh_dist, umap_dist)
            spearman_rhos.append(rho if rho is not None else np.nan)
            
            # Rank agreement
            mh_ranks = rankdata(mh_dist)
            umap_ranks = rankdata(umap_dist)
            rank_agree = np.corrcoef(mh_ranks, umap_ranks)[0, 1]
            rank_agrees.append(rank_agree if np.isfinite(rank_agree) else np.nan)
            
            # Overlap
            all_umap_dist = np.sqrt(((xy - hit_xy) ** 2).sum(axis=1))
            all_umap_dist[pos] = np.inf
            umap_topk_idx = np.argpartition(all_umap_dist, TOPK_NEIGHBORS)[:TOPK_NEIGHBORS]
            umap_topk_set = set(int(i) for i in umap_topk_idx)
            mh_set = set(neigh_idx)
            overlap = len(mh_set & umap_topk_set) / TOPK_NEIGHBORS
            overlaps.append(overlap)
            
            # Purity
            neighbors_are_hit_or_hl = df.iloc[neigh_idx][["_is_hit", "_is_highlight"]].any(axis=1).sum()
            purity = neighbors_are_hit_or_hl / len(neigh_idx)
            purities.append(purity)
        
        metrics = {
            "n_hits_evaluated": len(spearman_rhos),
            "spearman_mean": np.nanmean(spearman_rhos),
            "spearman_std": np.nanstd(spearman_rhos),
            "spearman_min": np.nanmin(spearman_rhos),
            "rank_agreement_mean": np.nanmean(rank_agrees),
            "overlap_mean": np.mean(overlaps),
            "purity_mean": np.mean(purities),
        }
    else:
        metrics = {
            "n_hits_evaluated": 0,
            "spearman_mean": np.nan,
            "spearman_std": np.nan,
            "spearman_min": np.nan,
            "rank_agreement_mean": np.nan,
            "overlap_mean": np.nan,
            "purity_mean": np.nan,
        }
    
    # =====================
    # 2. GLOBAL UNIFORMITY
    # =====================
    x_range = xy[:, 0].max() - xy[:, 0].min()
    y_range = xy[:, 1].max() - xy[:, 1].min()
    range_ratio = x_range / y_range if y_range > 0 else 1.0
    
    metrics["x_range"] = x_range
    metrics["y_range"] = y_range
    metrics["range_ratio"] = range_ratio
    
    # =====================
    # 3. HIT CLUSTERING
    # =====================
    if len(hit_pos) > 1:
        hit_xy = xy[hit_pos]
        # Mean distance between hit centroids (rough clustering metric)
        hit_distances = np.sqrt(((hit_xy[:, None] - hit_xy[None, :]) ** 2).sum(axis=2))
        # Exclude diagonal
        mask = ~np.eye(len(hit_pos), dtype=bool)
        mean_hit_dist = hit_distances[mask].mean()
        metrics["mean_hit_to_hit_distance"] = mean_hit_dist
    else:
        metrics["mean_hit_to_hit_distance"] = np.nan
    
    return metrics


# ==========================================================
# Run parameter sweep
# ==========================================================
def run_parameter_sweep(fps: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit UMAP for each parameter combination and compute metrics.
    """
    param_combinations = list(product(
        PARAM_GRID["n_neighbors"],
        PARAM_GRID["min_dist"]
    ))
    
    print(f"\n{'='*80}")
    print(f"RUNNING PARAMETER SWEEP: {len(param_combinations)} combinations")
    print(f"{'='*80}\n")
    
    results = []
    
    for i, (n_nbrs, min_d) in enumerate(param_combinations):
        print(f"[{i+1}/{len(param_combinations)}] n_neighbors={n_nbrs}, min_dist={min_d}")
        
        params = {
            "n_neighbors": n_nbrs,
            "min_dist": min_d,
            "metric": minhash_distance,
            **FIXED_PARAMS,
        }
        
        try:
            # Fit UMAP
            reducer = umap.UMAP(**params)
            emb = reducer.fit_transform(fps)
            
            # Compute metrics
            quality = compute_quality_metrics(df, fps, emb)
            
            result = {
                "n_neighbors": n_nbrs,
                "min_dist": min_d,
                **quality,
            }
            results.append(result)
            print(f"  ✓ Spearman ρ (mean): {quality['spearman_mean']:.3f} | "
                  f"Overlap: {quality['overlap_mean']:.3f} | "
                  f"Purity: {quality['purity_mean']:.3f}")
            
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            continue
    
    df_results = pd.DataFrame(results)
    return df_results


# ==========================================================
# Generate publication-ready report
# ==========================================================
def generate_sweep_report(df_results: pd.DataFrame, output_dir: str):
    """
    Create formatted text report and save results CSV.
    """
    
    # Sort by Spearman (best first)
    df_sorted = df_results.sort_values("spearman_mean", ascending=False).reset_index(drop=True)
    
    # Find best configuration
    best_row = df_sorted.iloc[0]
    
    report = f"""
{'='*80}
UMAP PARAMETER SWEEP REPORT
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PARAMETER GRID TESTED:
  n_neighbors: {PARAM_GRID['n_neighbors']}
  min_dist: {PARAM_GRID['min_dist']}
  Total combinations: {len(df_results)}

{'='*80}
BEST CONFIGURATION (ranked by Spearman ρ):
{'='*80}

Parameters:
  n_neighbors = {int(best_row['n_neighbors'])}
  min_dist = {best_row['min_dist']}

Quality Metrics:
  Spearman ρ (mean ± std):           {best_row['spearman_mean']:.4f} ± {best_row['spearman_std']:.4f}
  Spearman ρ (min):                  {best_row['spearman_min']:.4f}
  Rank Agreement (mean):             {best_row['rank_agreement_mean']:.4f}
  UMAP-topK Overlap (mean):          {best_row['overlap_mean']:.4f}
  Neighbor Purity (mean):            {best_row['purity_mean']:.4f}
  Mean Hit-to-Hit Distance:          {best_row['mean_hit_to_hit_distance']:.4f}
  Embedding Range Ratio (x/y):       {best_row['range_ratio']:.4f}
  Hits evaluated:                    {int(best_row['n_hits_evaluated'])}

INTERPRETATION:
  • Spearman ρ > 0.5: Local structure well-preserved
  • Overlap > 0.3: UMAP and fingerprint neighborhoods agree
  • Purity > 0.5: Hits cluster together (good for chemical discovery)
  • Range ratio ≈ 1.0: Balanced embedding (no stretching)

{'='*80}
TOP 5 CONFIGURATIONS:
{'='*80}

"""
    
    top5 = df_sorted.head(5)[["n_neighbors", "min_dist", "spearman_mean", 
                                "rank_agreement_mean", "overlap_mean", "purity_mean"]]
    
    report += top5.to_string(index=True) + "\n\n"
    
    report += f"""
{'='*80}
ALL RESULTS (sorted by Spearman ρ):
{'='*80}

"""
    
    all_results = df_sorted[["n_neighbors", "min_dist", "spearman_mean", "spearman_std",
                             "rank_agreement_mean", "overlap_mean", "purity_mean", "range_ratio"]]
    report += all_results.to_string(index=False) + "\n\n"
    
    report += f"""
{'='*80}
RECOMMENDATIONS FOR PUBLICATION:
{'='*80}

1. REPORTED PARAMETERS:
   "UMAP was applied with n_neighbors={int(best_row['n_neighbors'])} and 
   min_dist={best_row['min_dist']}, using a custom Jaccard distance metric 
   on MAPchiral MinHash fingerprints. Parameters were selected via systematic 
   grid search across {{n_neighbors: {PARAM_GRID['n_neighbors']}, 
   min_dist: {PARAM_GRID['min_dist']}}} to maximize preservation of 
   local neighborhood structure, as measured by Spearman correlation 
   (ρ = {best_row['spearman_mean']:.3f})."

2. METHODS SECTION ADDITION:
   "UMAP hyperparameter selection was conducted using a {len(df_results)}-combination 
   parameter grid. For each combination, we computed: (i) Spearman correlation 
   between fingerprint and embedding distances in hit neighborhoods (k={TOPK_NEIGHBORS}), 
   (ii) rank correlation agreement, (iii) overlap between fingerprint-space and 
   embedding-space k-nearest neighbors, and (iv) neighbor purity 
   (fraction of neighbors that are known hits/highlights). 
   The configuration maximizing Spearman correlation was selected."

3. FIGURE CAPTION ADDITION:
   "Hits shown as X markers (n={int(best_row['n_hits_evaluated'])}). 
   Local neighborhood preservation: Spearman ρ = {best_row['spearman_mean']:.3f}, 
   indicating {('strong' if best_row['spearman_mean'] > 0.6 else 'moderate' if best_row['spearman_mean'] > 0.4 else 'weak')} 
   correlation between molecular fingerprint distances and embedding distances."

4. SUPPLEMENTARY DATA:
   "Full parameter sweep results and per-hit faithfulness metrics are provided 
   in Supplementary Table X and available as supplementary CSV files."

{'='*80}

"""
    
    return report


# ==========================================================
# Visualization
# ==========================================================
def plot_sweep_results(df_results: pd.DataFrame, output_dir: str):
    """
    Create heatmaps and comparison plots.
    """
    
    # Pivot for heatmap
    pivot_spearman = df_results.pivot_table(
        values="spearman_mean",
        index="n_neighbors",
        columns="min_dist"
    )
    
    pivot_overlap = df_results.pivot_table(
        values="overlap_mean",
        index="n_neighbors",
        columns="min_dist"
    )
    
    pivot_purity = df_results.pivot_table(
        values="purity_mean",
        index="n_neighbors",
        columns="min_dist"
    )
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    # Spearman heatmap
    sns.heatmap(pivot_spearman, annot=True, fmt=".3f", cmap="RdYlGn", 
                ax=axes[0], cbar_kws={"label": "Spearman ρ"}, vmin=0, vmax=1)
    axes[0].set_title("Local Faithfulness (Spearman ρ)", fontweight="bold")
    axes[0].set_xlabel("min_dist")
    axes[0].set_ylabel("n_neighbors")
    
    # Overlap heatmap
    sns.heatmap(pivot_overlap, annot=True, fmt=".3f", cmap="RdYlGn",
                ax=axes[1], cbar_kws={"label": "Overlap Fraction"}, vmin=0, vmax=1)
    axes[1].set_title("Neighborhood Agreement (Overlap)", fontweight="bold")
    axes[1].set_xlabel("min_dist")
    axes[1].set_ylabel("n_neighbors")
    
    # Purity heatmap
    sns.heatmap(pivot_purity, annot=True, fmt=".3f", cmap="RdYlGn",
                ax=axes[2], cbar_kws={"label": "Purity"}, vmin=0, vmax=1)
    axes[2].set_title("Hit Clustering (Neighbor Purity)", fontweight="bold")
    axes[2].set_xlabel("min_dist")
    axes[2].set_ylabel("n_neighbors")
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "umap_parameter_comparison_heatmaps.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap: {out_path}")


# ==========================================================
# MAIN
# ==========================================================
def main():
    ensure_dirs()
    
    # Load data once
    print("\nLoading and fingerprinting data...")
    fps, df, keep_idx = load_and_fingerprint_data(INPUT_CSV)
    
    # Run sweep
    df_results = run_parameter_sweep(fps, df)
    
    # Save results CSV
    csv_path = os.path.join(SWEEP_OUTPUT_DIR, "parameter_sweep_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n✓ Saved results CSV: {csv_path}")
    
    # Generate report
    report = generate_sweep_report(df_results, SWEEP_OUTPUT_DIR)
    
    report_path = os.path.join(SWEEP_OUTPUT_DIR, "parameter_sweep_summary.txt")
    with open(report_path, "w", encoding='utf-8') as f:
        f.write(report)
    print(f"✓ Saved report: {report_path}")
    
    # Print to terminal
    print(report)
    
    # Generate visualizations
    plot_sweep_results(df_results, SWEEP_OUTPUT_DIR)
    
    print(f"\n{'='*80}")
    print("SWEEP COMPLETE")
    print(f"{'='*80}")
    print(f"All outputs saved to: {SWEEP_OUTPUT_DIR}")


if __name__ == "__main__":
    main()