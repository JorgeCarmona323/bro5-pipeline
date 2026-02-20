"""
COMPREHENSIVE UMAP Parameter Sweep for Conditions A, B, ALL × 3 Panel Types
===========================================================================

Systematically evaluates UMAP parameters for:
- Condition A (Literature + Hits)
- Condition B (Library + Hits)  
- Condition ALL (Literature + Library + Hits)

For each condition, tests:
- Panel 1: Structural (MAPchiral fingerprints)
- Panel 2: FPM-normalized (size-corrected complexity)
- Panel 3: Descriptor-based (physicochemical properties)

Total: 9 independent parameter sweeps

Outputs organized by: {condition}/{panel_type}/
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
SWEEP_OUTPUT_DIR = "C:\\Users\\Admin\\Documents\\Hu Lab\\Code\\Python\\rdkit\\Scripts\\UMAP_scripts\\runs\\2026-01-29\\Sweep_Results"

SMILES_COL = "Smiles"
SOURCE_COL = "Source"
HIT_ID_COL = "Hit_ID"
HIGHLIGHT_COL = "Highlight_ID"

DESC_COLS = [
    "Total Molweight",
    "cLogP",
    "H-Acceptors",
    "H-Donors",
    "Polar Surface Area",
    "Rotatable Bonds",
]

MAX_RADIUS = 2
N_PERMUTATIONS = 2048
MAPPING = False
RANDOM_STATE = 42
TOPK_NEIGHBORS = 50

# Conditions to test
CONDITIONS = ["A", "B", "ALL"]  # A=Lit+Hits, B=Lib+Hits, ALL=all data

# Parameter grid (broader sweep)
PARAM_GRID = {
    "n_neighbors": [10, 15, 20, 30, 40, 50],
    "min_dist": [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
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
    for cond in CONDITIONS:
        for panel in ["Panel1_Structural", "Panel2_FPM", "Panel3_Descriptors"]:
            os.makedirs(os.path.join(SWEEP_OUTPUT_DIR, cond, panel), exist_ok=True)

def is_sweep_completed(output_dir: str) -> bool:
    """Check if a sweep was already completed by looking for results CSV"""
    csv_path = os.path.join(output_dir, "sweep_results.csv")
    return os.path.exists(csv_path)

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
# Data loading and filtering
# ==========================================================
def load_and_filter_data(csv_path: str, condition: str) -> pd.DataFrame:
    """Load CSV and filter by condition"""
    df = pd.read_csv(csv_path, sep=None, engine="python", encoding='utf-8')
    
    # Normalize metadata
    df[SOURCE_COL] = df[SOURCE_COL].astype(str).str.strip().str.lower()
    df[HIT_ID_COL] = df[HIT_ID_COL].apply(norm_str)
    df[HIGHLIGHT_COL] = df[HIGHLIGHT_COL].apply(norm_str)
    df["_is_hit"] = df[HIT_ID_COL].astype(str).str.strip().ne("")
    df["_is_highlight"] = df[HIGHLIGHT_COL].astype(str).str.strip().ne("")
    
    n_before = len(df)
    
    # Apply filtering
    if condition == "A":
        df = df[df[SOURCE_COL].isin(["literature", "hit"])].copy()
        cond_label = "Condition A (Literature + Hits)"
    elif condition == "B":
        df = df[df[SOURCE_COL].isin(["library", "hit"])].copy()
        cond_label = "Condition B (Library + Hits)"
    else:
        cond_label = "Condition ALL (All Data)"
    
    n_after = len(df)
    
    print(f"\n{cond_label}")
    print(f"  Filtered: {n_before:,} → {n_after:,} compounds")
    print(f"  Hits: {df['_is_hit'].sum():,}")
    print(f"  Highlights: {df['_is_highlight'].sum():,}")
    
    return df


def fingerprint_data(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Fingerprint molecules with MAPchiral"""
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
    
    print(f"  Fingerprinted: {len(fps_array):,} molecules")
    
    return fps_array, df_clean


def normalize_fpm(df: pd.DataFrame, fps: np.ndarray) -> np.ndarray:
    """Normalize fingerprints by molecular weight (FPM)"""
    mw = df["Total Molweight"].astype(float).values
    fps_normalized = fps / mw[:, np.newaxis]
    
    on_bits = np.count_nonzero(fps, axis=1)
    fpm = on_bits / mw
    
    print(f"  FPM range: {fpm.min():.6f} - {fpm.max():.6f}")
    
    return fps_normalized


# ==========================================================
# Quality metrics
# ==========================================================
def compute_quality_metrics(df: pd.DataFrame, fps: np.ndarray, emb: np.ndarray, metric_spec) -> dict:
    """Compute quality metrics for UMAP embedding
    
    Args:
        metric_spec: Either a numba-compiled function (for fingerprints) or a string like "euclidean" (for descriptors)
    """
    
    xy = emb
    
    # Build NN index
    index = NNDescent(
        fps,
        metric=metric_spec,
        n_neighbors=max(TOPK_NEIGHBORS + 10, 40),
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    hit_pos = np.where(df["_is_hit"].to_numpy())[0]
    
    if len(hit_pos) > 0:
        inds, dists = index.query(fps[hit_pos], k=TOPK_NEIGHBORS + 1)
        
        spearman_rhos = []
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
            "overlap_mean": np.mean(overlaps),
            "purity_mean": np.mean(purities),
        }
    else:
        metrics = {
            "n_hits_evaluated": 0,
            "spearman_mean": np.nan,
            "spearman_std": np.nan,
            "spearman_min": np.nan,
            "overlap_mean": np.nan,
            "purity_mean": np.nan,
        }
    
    # Global uniformity
    x_range = xy[:, 0].max() - xy[:, 0].min()
    y_range = xy[:, 1].max() - xy[:, 1].min()
    range_ratio = x_range / y_range if y_range > 0 else 1.0
    
    metrics["x_range"] = x_range
    metrics["y_range"] = y_range
    metrics["range_ratio"] = range_ratio
    
    # Hit clustering
    if len(hit_pos) > 1:
        hit_xy = xy[hit_pos]
        hit_distances = np.sqrt(((hit_xy[:, None] - hit_xy[None, :]) ** 2).sum(axis=2))
        mask = ~np.eye(len(hit_pos), dtype=bool)
        mean_hit_dist = hit_distances[mask].mean()
        metrics["mean_hit_to_hit_distance"] = mean_hit_dist
    else:
        metrics["mean_hit_to_hit_distance"] = np.nan
    
    return metrics


# ==========================================================
# Run sweep for specific panel
# ==========================================================
def run_panel_sweep(condition: str, panel_name: str, features: np.ndarray, df: pd.DataFrame, 
                     metric_spec, umap_params_base: dict) -> pd.DataFrame:
    """Run parameter sweep for a specific panel type
    
    Args:
        metric_spec: Either a numba-compiled function or string metric name
    """
    
    param_combinations = list(product(
        PARAM_GRID["n_neighbors"],
        PARAM_GRID["min_dist"]
    ))
    
    print(f"\n{'='*80}")
    print(f"SWEEP: Condition {condition} - {panel_name}")
    print(f"{'='*80}")
    print(f"Parameter combinations: {len(param_combinations)}")
    print(f"Feature shape: {features.shape}")
    
    results = []
    
    for i, (n_nbrs, min_d) in enumerate(param_combinations):
        print(f"  [{i+1}/{len(param_combinations)}] n_neighbors={n_nbrs}, min_dist={min_d}", end="")
        
        params = {
            "n_neighbors": n_nbrs,
            "min_dist": min_d,
            **umap_params_base,
        }
        
        try:
            reducer = umap.UMAP(**params)
            emb = reducer.fit_transform(features)
            
            quality = compute_quality_metrics(df, features, emb, metric_spec)
            
            result = {
                "condition": condition,
                "panel": panel_name,
                "n_neighbors": n_nbrs,
                "min_dist": min_d,
                **quality,
            }
            results.append(result)
            
            print(f" → ρ={quality['spearman_mean']:.3f}, overlap={quality['overlap_mean']:.3f}")
            
        except Exception as e:
            print(f" → ERROR: {e}")
            continue
    
    return pd.DataFrame(results)


# ==========================================================
# Generate reports
# ==========================================================
def generate_panel_report(df_results: pd.DataFrame, condition: str, panel_name: str, output_dir: str):
    """Generate report for specific condition × panel combination"""
    
    if df_results.empty:
        return
    
    df_sorted = df_results.sort_values("spearman_mean", ascending=False).reset_index(drop=True)
    best_row = df_sorted.iloc[0]
    
    report = f"""
{'='*80}
UMAP PARAMETER SWEEP REPORT
{'='*80}
Condition: {condition}
Panel: {panel_name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PARAMETER GRID:
  n_neighbors: {PARAM_GRID['n_neighbors']}
  min_dist: {PARAM_GRID['min_dist']}
  Total tested: {len(df_results)}

{'='*80}
BEST CONFIGURATION (by Spearman ρ):
{'='*80}

Parameters:
  n_neighbors = {int(best_row['n_neighbors'])}
  min_dist = {best_row['min_dist']}

Quality Metrics:
  Spearman ρ (mean ± std):           {best_row['spearman_mean']:.4f} ± {best_row['spearman_std']:.4f}
  Spearman ρ (min):                  {best_row['spearman_min']:.4f}
  UMAP-topK Overlap (mean):          {best_row['overlap_mean']:.4f}
  Neighbor Purity (mean):            {best_row['purity_mean']:.4f}
  Mean Hit-to-Hit Distance:          {best_row['mean_hit_to_hit_distance']:.4f}
  Embedding Range Ratio (x/y):       {best_row['range_ratio']:.4f}
  Hits evaluated:                    {int(best_row['n_hits_evaluated'])}

{'='*80}
TOP 5 CONFIGURATIONS:
{'='*80}

{df_sorted.head(5)[["n_neighbors", "min_dist", "spearman_mean", "overlap_mean", "purity_mean"]].to_string(index=True)}

{'='*80}
ALL RESULTS (sorted by Spearman ρ):
{'='*80}

{df_sorted[["n_neighbors", "min_dist", "spearman_mean", "spearman_std", "overlap_mean", "purity_mean", "range_ratio"]].to_string(index=False)}

{'='*80}
"""
    
    # Save report
    report_path = os.path.join(output_dir, "sweep_summary.txt")
    with open(report_path, "w", encoding='utf-8') as f:
        f.write(report)
    
    # Save CSV
    csv_path = os.path.join(output_dir, "sweep_results.csv")
    df_results.to_csv(csv_path, index=False)
    
    print(f"\n  ✓ Saved: {report_path}")
    print(f"  ✓ Saved: {csv_path}")
    
    return report


def plot_panel_heatmaps(df_results: pd.DataFrame, condition: str, panel_name: str, output_dir: str):
    """Generate heatmaps for panel results"""
    
    if df_results.empty or len(df_results) < 4:
        return
    
    pivot_spearman = df_results.pivot_table(values="spearman_mean", index="n_neighbors", columns="min_dist")
    pivot_overlap = df_results.pivot_table(values="overlap_mean", index="n_neighbors", columns="min_dist")
    pivot_purity = df_results.pivot_table(values="purity_mean", index="n_neighbors", columns="min_dist")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.heatmap(pivot_spearman, annot=True, fmt=".3f", cmap="RdYlGn", 
                ax=axes[0], cbar_kws={"label": "Spearman ρ"}, vmin=0, vmax=1)
    axes[0].set_title(f"{condition} - {panel_name}\nSpearman ρ", fontweight="bold")
    
    sns.heatmap(pivot_overlap, annot=True, fmt=".3f", cmap="RdYlGn",
                ax=axes[1], cbar_kws={"label": "Overlap"}, vmin=0, vmax=1)
    axes[1].set_title("Neighborhood Overlap", fontweight="bold")
    
    sns.heatmap(pivot_purity, annot=True, fmt=".3f", cmap="RdYlGn",
                ax=axes[2], cbar_kws={"label": "Purity"}, vmin=0, vmax=1)
    axes[2].set_title("Neighbor Purity", fontweight="bold")
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "parameter_heatmaps.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"  ✓ Saved: {out_path}")


# ==========================================================
# MAIN
# ==========================================================
def main():
    ensure_dirs()
    
    all_results = []
    total_sweeps = len(CONDITIONS) * 3
    completed_count = 0
    skipped_count = 0
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE UMAP PARAMETER SWEEP")
    print(f"{'='*80}")
    print(f"Total scenarios: {total_sweeps} ({len(CONDITIONS)} conditions × 3 panels)")
    print(f"Output: {SWEEP_OUTPUT_DIR}")
    print(f"{'='*80}\n")
    
    for condition in CONDITIONS:
        print(f"\n\n{'#'*80}")
        print(f"# PROCESSING CONDITION: {condition}")
        print(f"{'#'*80}")
        
        # Load and filter data
        df = load_and_filter_data(INPUT_CSV, condition)
        
        # Fingerprint
        fps, df_clean = fingerprint_data(df)
        
        # # Panel 1: Structural (standard fingerprints) - COMMENTED OUT FOR SPEED
        # print(f"\n>>> Panel 1: Structural (MAPchiral)")
        # panel1_dir = os.path.join(SWEEP_OUTPUT_DIR, condition, "Panel1_Structural")
        
        # if is_sweep_completed(panel1_dir):
        #     print(f"  ✓ ALREADY COMPLETED - Loading existing results")
        #     panel1_results = pd.read_csv(os.path.join(panel1_dir, "sweep_results.csv"))
        #     skipped_count += 1
        # else:
        #     panel1_results = run_panel_sweep(
        #         condition, "Panel1_Structural", fps, df_clean, 
        #         minhash_distance, {**FIXED_PARAMS, "metric": minhash_distance}
        #     )
        #     generate_panel_report(panel1_results, condition, "Panel1_Structural", panel1_dir)
        #     plot_panel_heatmaps(panel1_results, condition, "Panel1_Structural", panel1_dir)
        #     completed_count += 1
        
        # all_results.append(panel1_results)
        
        # # Panel 2: FPM-normalized - COMMENTED OUT FOR SPEED
        # print(f"\n>>> Panel 2: FPM-Normalized")
        # fps_fpm = normalize_fpm(df_clean, fps)
        # panel2_dir = os.path.join(SWEEP_OUTPUT_DIR, condition, "Panel2_FPM")
        
        # if is_sweep_completed(panel2_dir):
        #     print(f"  ✓ ALREADY COMPLETED - Loading existing results")
        #     panel2_results = pd.read_csv(os.path.join(panel2_dir, "sweep_results.csv"))
        #     skipped_count += 1
        # else:
        #     panel2_results = run_panel_sweep(
        #         condition, "Panel2_FPM", fps_fpm, df_clean,
        #         minhash_distance, {**FIXED_PARAMS, "metric": minhash_distance}
        #     )
        #     generate_panel_report(panel2_results, condition, "Panel2_FPM", panel2_dir)
        #     plot_panel_heatmaps(panel2_results, condition, "Panel2_FPM", panel2_dir)
        #     completed_count += 1
        
        # all_results.append(panel2_results)
        
        # Panel 3: Descriptors
        print(f"\n>>> Panel 3: Descriptors")
        desc_data = df_clean[DESC_COLS].astype(float).values
        desc_scaled = StandardScaler().fit_transform(desc_data)
        panel3_dir = os.path.join(SWEEP_OUTPUT_DIR, condition, "Panel3_Descriptors")
        
        if is_sweep_completed(panel3_dir):
            print(f"  ✓ ALREADY COMPLETED - Loading existing results")
            panel3_results = pd.read_csv(os.path.join(panel3_dir, "sweep_results.csv"))
            skipped_count += 1
        else:
            panel3_results = run_panel_sweep(
                condition, "Panel3_Descriptors", desc_scaled, df_clean,
                "euclidean",  # Use string metric for descriptors (compatible with NNDescent)
                {**FIXED_PARAMS, "metric": "euclidean", "init": "random"}
            )
            generate_panel_report(panel3_results, condition, "Panel3_Descriptors", panel3_dir)
            plot_panel_heatmaps(panel3_results, condition, "Panel3_Descriptors", panel3_dir)
            completed_count += 1
        
        all_results.append(panel3_results)
    
    # Combine all results
    df_all = pd.concat(all_results, ignore_index=True)
    combined_csv = os.path.join(SWEEP_OUTPUT_DIR, "all_sweeps_combined.csv")
    df_all.to_csv(combined_csv, index=False)
    
    print(f"\n\n{'='*80}")
    print(f"COMPREHENSIVE SWEEP COMPLETE")
    print(f"{'='*80}")
    print(f"Total scenarios: {total_sweeps}")
    print(f"Newly completed: {completed_count}")
    print(f"Previously completed (skipped): {skipped_count}")
    print(f"Total parameter combinations tested: {len(df_all)}")
    print(f"All results saved to: {SWEEP_OUTPUT_DIR}")
    print(f"Combined results: {combined_csv}")
    
    # Summary of best configurations
    print(f"\n{'='*80}")
    print("BEST CONFIGURATIONS SUMMARY:")
    print(f"{'='*80}\n")
    
    for condition in CONDITIONS:
        for panel in ["Panel1_Structural", "Panel2_FPM", "Panel3_Descriptors"]:
            subset = df_all[(df_all["condition"] == condition) & (df_all["panel"] == panel)]
            if not subset.empty:
                best = subset.sort_values("spearman_mean", ascending=False).iloc[0]
                print(f"{condition} - {panel:20s}: n_neighbors={int(best['n_neighbors']):2d}, min_dist={best['min_dist']:.2f}, ρ={best['spearman_mean']:.3f}")


if __name__ == "__main__":
    main()
