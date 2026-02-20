"""
Panel 3 Validation: Descriptor Space Dimensionality Reduction Comparison

Validates whether the lack of clear clustering in Panel 3 is:
1. Real biological signal (2D descriptors don't predict permeability)
2. UMAP parameter artifact
3. Insufficient descriptors

Compares: PCA, t-SNE, UMAP (default), UMAP (optimized)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform
import umap

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


# ==========================================================
# USER CONFIG
# ==========================================================
INPUT_CSV = "C:\\Users\\Admin\\Documents\\Hu Lab\\Code\\Python\\rdkit\\Data\\2026-01-29\\canonicalized_master_macrocycles_2D_Descriptors_FINAL_20260129.csv"
OUTPUT_DIR = "C:\\Users\\Admin\\Documents\\Hu Lab\\Code\\Python\\rdkit\\Scripts\\UMAP_scripts\\UMAP_runs\\2026-02-03_Panel3_Validation"

SMILES_COL = "Smiles"
SOURCE_COL = "Source"
HIT_ID_COL = "Hit_ID"

# 8 descriptors (same as Panel 3)
DESC_COLS = [
    "Total Molweight",
    "cLogP",
    "cLogS",
    "H-Acceptors",
    "H-Donors",
    "Polar Surface Area",
    "Rotatable Bonds",
    "Aromatic Rings",
]

# Condition to analyze
DATA_CONDITION = "D"  # Literature + 34_Hits + Hits

# Optimized UMAP parameters (from your sweep)
UMAP_OPTIMIZED = {
    "A": dict(n_neighbors=15, min_dist=0.15),
    "B": dict(n_neighbors=10, min_dist=0.05),
    "C": dict(n_neighbors=50, min_dist=0.10),
    "ALL": dict(n_neighbors=50, min_dist=0.10),
    "D": dict(n_neighbors=15, min_dist=0.15),
    "E": dict(n_neighbors=10, min_dist=0.05),
}

RANDOM_STATE = 42

# Color scheme
COLOR_LITERATURE = "#D0D0D0"
COLOR_LIBRARY = "#1F77B4"
COLOR_34HITS = "#E41A1C"
COLOR_BRAIN_6_4_4_13 = "#984EA3"


# ==========================================================
# Helper Functions
# ==========================================================
def norm_str(x) -> str:
    return "" if pd.isna(x) else str(x).strip()


def compute_embedding_faithfulness(X_high: np.ndarray, X_low: np.ndarray) -> dict:
    """
    Compare high-dim vs low-dim distances to assess embedding quality.
    
    Returns dict with:
    - spearman_rho: Correlation of distance ranks
    - pearson_r: Correlation of actual distances
    - variance_explained: For linear methods
    """
    # Pairwise distances
    dist_high = pdist(X_high, metric='euclidean')
    dist_low = pdist(X_low, metric='euclidean')
    
    # Correlations
    spearman_rho, _ = spearmanr(dist_high, dist_low)
    pearson_r, _ = pearsonr(dist_high, dist_low)
    
    return {
        "spearman_rho": float(spearman_rho),
        "pearson_r": float(pearson_r),
    }


def plot_comparison(
    embeddings: dict,
    df: pd.DataFrame,
    output_path: str,
    title: str = "Descriptor Space: Method Comparison"
):
    """
    Create 2x2 comparison plot of different dimensionality reduction methods.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    methods = list(embeddings.keys())
    
    for idx, method_name in enumerate(methods):
        ax = axes[idx]
        emb = embeddings[method_name]["embedding"]
        metrics = embeddings[method_name]["metrics"]
        
        # Get data
        src = df[SOURCE_COL].astype(str).str.strip().str.lower()
        
        # Plot background
        lit = df[src == "literature"]
        lib = df[src == "library"]
        hits_34 = df[df["_is_34hit"]]
        hits = df[df["_is_hit"]]
        
        if not lit.empty:
            lit_emb = emb[lit.index]
            ax.scatter(lit_emb[:, 0], lit_emb[:, 1], s=15, alpha=0.2, 
                      c=COLOR_LITERATURE, label="Literature", rasterized=True)
        
        if not lib.empty:
            lib_emb = emb[lib.index]
            ax.scatter(lib_emb[:, 0], lib_emb[:, 1], s=15, alpha=0.15,
                      c=COLOR_LIBRARY, label="Library", rasterized=True)
        
        if not hits_34.empty:
            hits34_emb = emb[hits_34.index]
            ax.scatter(hits34_emb[:, 0], hits34_emb[:, 1], s=80, alpha=0.85,
                      c=COLOR_34HITS, label="34_Hits", zorder=3,
                      edgecolors='darkred', linewidths=0.5)
        
        if not hits.empty:
            hits_emb = emb[hits.index]
            ax.scatter(hits_emb[:, 0], hits_emb[:, 1], s=100, marker="X",
                      alpha=0.95, c="black", label="Hits", zorder=7,
                      edgecolors="white", linewidths=1.5)
        
        # Add metrics to title
        rho = metrics.get("spearman_rho", np.nan)
        var_exp = metrics.get("variance_explained", None)
        
        subtitle = f"{method_name}\nSpearman ρ = {rho:.3f}"
        if var_exp is not None:
            subtitle += f"\nVariance = {var_exp:.1f}%"
        
        ax.set_title(subtitle, fontsize=11, fontweight='bold')
        ax.set_xlabel("Component 1", fontsize=9)
        ax.set_ylabel("Component 2", fontsize=9)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison plot: {output_path}")


def analyze_descriptor_correlations(X: np.ndarray, desc_names: list, output_dir: str):
    """
    Analyze correlations between descriptors to understand structure.
    """
    corr_matrix = np.corrcoef(X.T)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        xticklabels=desc_names,
        yticklabels=desc_names,
        square=True,
        cbar_kws={'label': 'Pearson Correlation'},
        ax=ax
    )
    ax.set_title("Descriptor Correlation Matrix (Raw Values)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "descriptor_correlations.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved correlation matrix: {plot_path}")
    
    # Print high correlations
    print("\nHighly correlated descriptor pairs (|r| > 0.7):")
    for i in range(len(desc_names)):
        for j in range(i+1, len(desc_names)):
            if abs(corr_matrix[i, j]) > 0.7:
                print(f"  {desc_names[i]} <-> {desc_names[j]}: r = {corr_matrix[i, j]:.3f}")


def compute_group_separability(emb: np.ndarray, df: pd.DataFrame) -> dict:
    """
    Quantify how well different groups are separated in embedding.
    """
    from sklearn.metrics import silhouette_score
    
    # Create group labels
    labels = []
    for idx in range(len(df)):
        if df.iloc[idx]["_is_hit"]:
            labels.append(2)  # Hits
        elif df.iloc[idx]["_is_34hit"]:
            labels.append(1)  # 34_hits
        else:
            labels.append(0)  # Background
    
    labels = np.array(labels)
    
    # Silhouette score (higher = better separated)
    # Only compute if we have multiple groups
    if len(np.unique(labels)) > 1:
        sil_score = silhouette_score(emb, labels, metric='euclidean')
    else:
        sil_score = np.nan
    
    # Mean distances within vs between groups
    hits_idx = df["_is_hit"].values
    hits34_idx = df["_is_34hit"].values
    bg_idx = ~(hits_idx | hits34_idx)
    
    def mean_dist(emb, mask1, mask2):
        if mask1.sum() == 0 or mask2.sum() == 0:
            return np.nan
        emb1 = emb[mask1]
        emb2 = emb[mask2]
        dists = np.linalg.norm(emb1[:, None] - emb2[None, :], axis=2)
        return float(dists.mean())
    
    metrics = {
        "silhouette_score": float(sil_score),
        "hits_to_bg_dist": mean_dist(emb, hits_idx, bg_idx),
        "34hits_to_bg_dist": mean_dist(emb, hits34_idx, bg_idx),
        "hits_to_34hits_dist": mean_dist(emb, hits_idx, hits34_idx),
    }
    
    return metrics


# ==========================================================
# MAIN ANALYSIS
# ==========================================================
def main():
    print("\n" + "=" * 78)
    print("Panel 3 Validation: Descriptor Space Analysis")
    print("=" * 78)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    print(f"\nLoading data from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    
    # Apply condition filter
    condition_map = {
        "A": ["literature", "hit"],
        "B": ["library", "hit"],
        "C": ["literature", "library", "hit"],
        "ALL": None,
        "D": ["literature", "hit", "34_hits"],
        "E": ["library", "hit", "34_hits"],
    }
    
    sources = condition_map[DATA_CONDITION]
    if sources is not None:
        df = df[df[SOURCE_COL].str.lower().isin(sources)].copy()
    
    print(f"Analyzing Condition {DATA_CONDITION}: {len(df)} compounds")
    
    # Mark groups
    df["_is_hit"] = df[HIT_ID_COL].notna() & (df[HIT_ID_COL].astype(str).str.strip() != "")
    df["_is_34hit"] = df[SOURCE_COL].str.lower() == "34_hits"
    
    print(f"  Hits: {df['_is_hit'].sum()}")
    print(f"  34_Hits: {df['_is_34hit'].sum()}")
    print(f"  Literature: {(df[SOURCE_COL].str.lower() == 'literature').sum()}")
    
    # Extract and scale descriptors
    print(f"\nProcessing {len(DESC_COLS)} descriptors...")
    X = df[DESC_COLS].astype(float).values
    
    # Analyze raw descriptor correlations
    analyze_descriptor_correlations(X, DESC_COLS, OUTPUT_DIR)
    
    # Standardize (same as Panel 3)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("\nDescriptor statistics (scaled):")
    print(f"  Mean: {X_scaled.mean(axis=0).round(2)}")
    print(f"  Std:  {X_scaled.std(axis=0).round(2)}")
    
    # Store embeddings and metrics
    embeddings = {}
    
    # ========== METHOD 1: PCA (Linear Baseline) ==========
    print("\n1. Computing PCA (linear baseline)...")
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    emb_pca = pca.fit_transform(X_scaled)
    
    metrics_pca = compute_embedding_faithfulness(X_scaled, emb_pca)
    metrics_pca["variance_explained"] = pca.explained_variance_ratio_.sum() * 100
    metrics_pca.update(compute_group_separability(emb_pca, df))
    
    embeddings["PCA (Linear)"] = {
        "embedding": emb_pca,
        "metrics": metrics_pca
    }
    
    print(f"   Variance explained: {metrics_pca['variance_explained']:.1f}%")
    print(f"   Spearman ρ: {metrics_pca['spearman_rho']:.3f}")
    
    # ========== METHOD 2: t-SNE (Nonlinear, perplexity=30) ==========
    print("\n2. Computing t-SNE (nonlinear, perplexity=30)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_STATE, n_jobs=-1)
    emb_tsne = tsne.fit_transform(X_scaled)
    
    metrics_tsne = compute_embedding_faithfulness(X_scaled, emb_tsne)
    metrics_tsne.update(compute_group_separability(emb_tsne, df))
    
    embeddings["t-SNE (perplexity=30)"] = {
        "embedding": emb_tsne,
        "metrics": metrics_tsne
    }
    
    print(f"   Spearman ρ: {metrics_tsne['spearman_rho']:.3f}")
    
    # ========== METHOD 3: UMAP Default ==========
    print("\n3. Computing UMAP (default: n=15, min_dist=0.1)...")
    umap_default = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric='euclidean',
        random_state=RANDOM_STATE
    )
    emb_umap_default = umap_default.fit_transform(X_scaled)
    
    metrics_umap_default = compute_embedding_faithfulness(X_scaled, emb_umap_default)
    metrics_umap_default.update(compute_group_separability(emb_umap_default, df))
    
    embeddings["UMAP (Default)"] = {
        "embedding": emb_umap_default,
        "metrics": metrics_umap_default
    }
    
    print(f"   Spearman ρ: {metrics_umap_default['spearman_rho']:.3f}")
    
    # ========== METHOD 4: UMAP Optimized ==========
    opt_params = UMAP_OPTIMIZED[DATA_CONDITION]
    print(f"\n4. Computing UMAP (optimized: n={opt_params['n_neighbors']}, min_dist={opt_params['min_dist']})...")
    umap_opt = umap.UMAP(
        n_neighbors=opt_params['n_neighbors'],
        min_dist=opt_params['min_dist'],
        n_components=2,
        metric='euclidean',
        random_state=RANDOM_STATE,
        init='random'
    )
    emb_umap_opt = umap_opt.fit_transform(X_scaled)
    
    metrics_umap_opt = compute_embedding_faithfulness(X_scaled, emb_umap_opt)
    metrics_umap_opt.update(compute_group_separability(emb_umap_opt, df))
    
    embeddings["UMAP (Optimized)"] = {
        "embedding": emb_umap_opt,
        "metrics": metrics_umap_opt
    }
    
    print(f"   Spearman ρ: {metrics_umap_opt['spearman_rho']:.3f}")
    
    # ========== COMPARISON PLOT ==========
    print("\n5. Generating comparison plot...")
    plot_path = os.path.join(OUTPUT_DIR, f"panel3_comparison_condition_{DATA_CONDITION}.png")
    plot_comparison(
        embeddings, df, plot_path,
        title=f"Panel 3 Validation: Condition {DATA_CONDITION} - Method Comparison"
    )
    
    # ========== SUMMARY REPORT ==========
    print("\n" + "=" * 78)
    print("SUMMARY REPORT")
    print("=" * 78)
    
    summary_data = []
    for method_name, data in embeddings.items():
        metrics = data["metrics"]
        summary_data.append({
            "Method": method_name,
            "Spearman_rho": metrics["spearman_rho"],
            "Pearson_r": metrics["pearson_r"],
            "Variance_Explained": metrics.get("variance_explained", np.nan),
            "Silhouette_Score": metrics["silhouette_score"],
            "Hits_to_BG_Dist": metrics["hits_to_bg_dist"],
            "34Hits_to_BG_Dist": metrics["34hits_to_bg_dist"],
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    print("\nFaithfulness & Separability Metrics:")
    print(summary_df.to_string(index=False))
    
    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, f"panel3_validation_summary_condition_{DATA_CONDITION}.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}")
    
    # ========== INTERPRETATION ==========
    print("\n" + "=" * 78)
    print("INTERPRETATION GUIDE")
    print("=" * 78)
    
    # Check if methods agree
    rhos = [data["metrics"]["spearman_rho"] for data in embeddings.values()]
    rho_range = max(rhos) - min(rhos)
    
    print(f"\nFaithfulness (Spearman ρ) range across methods: {rho_range:.3f}")
    
    if rho_range < 0.1:
        print("✓ All methods show similar faithfulness → Result is ROBUST")
        print("  The descriptor space structure is consistent across methods")
    else:
        print("⚠ Methods show different faithfulness → Result is METHOD-DEPENDENT")
        print("  Choice of dimensionality reduction method matters")
    
    # Check separability
    sil_scores = [data["metrics"]["silhouette_score"] for data in embeddings.values()]
    avg_sil = np.mean(sil_scores)
    
    print(f"\nAverage Silhouette Score: {avg_sil:.3f}")
    if avg_sil < 0.25:
        print("✓ LOW separability → Groups are NOT well-separated in descriptor space")
        print("  This suggests 2D descriptors do NOT strongly predict your categories")
        print("  → Supports hypothesis that 3D descriptors may be needed")
    elif avg_sil > 0.5:
        print("✓ HIGH separability → Groups ARE well-separated in descriptor space")
        print("  2D descriptors capture meaningful differences")
    else:
        print("~ MODERATE separability → Some structure exists but not strong")
    
    # Check PCA variance
    pca_var = embeddings["PCA (Linear)"]["metrics"]["variance_explained"]
    print(f"\nPCA Variance Explained: {pca_var:.1f}%")
    if pca_var < 50:
        print("  → Most variance in higher dimensions (8D space is complex)")
        print("  → Nonlinear methods (UMAP) may capture additional structure")
    else:
        print("  → Most variance in first 2 PCs (space is relatively simple)")
        print("  → Linear projection (PCA) captures main structure")
    
    print("\n" + "=" * 78)
    print("✅ Panel 3 validation complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 78)


if __name__ == "__main__":
    main()
