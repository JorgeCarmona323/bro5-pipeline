"""
ONE SCRIPT TO RULE THEM ALL (30k–50k scale) — ENHANCED VERSION 3

Version 3 improvements:
  - 6 conditions support (A, B, C, ALL, D, E)
  - 34_Hits integration with RED color
  - Brain 6-4-4-13 with PURPLE color
  - 8 descriptors (added cLogS, Aromatic Rings)
  - Condition-specific optimized UMAP parameters
"""

import os
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

import umap
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from numba import njit

from mapchiral.mapchiral import encode

from pynndescent import NNDescent
from scipy.stats import spearmanr, rankdata


# ==========================================================
# USER CONFIG (EDIT THIS SECTION)
# ==========================================================
INPUT_CSV = "C:\\Users\\Admin\\Documents\\Hu Lab\\Code\\Python\\rdkit\\Data\\2026-01-29\\canonicalized_master_macrocycles_2D_Descriptors_FINAL_20260129.csv"
OUTPUT_DIR = "C:\\Users\\Admin\\Documents\\Hu Lab\\Code\\Python\\rdkit\\Scripts\\UMAP_scripts\\runs\\2026-01-30\\Output"
FIG_DIR = "C:\\Users\\Admin\\Documents\\Hu Lab\\Code\\Python\\rdkit\\Scripts\\UMAP_scripts\\runs\\2026-01-30\\Figures"

SMILES_COL = "Smiles"
SOURCE_COL = "Source"
HIT_ID_COL = "Hit_ID"
HITS_34_COL = "34_Hits"  # New column for 34_Hits
HIGHLIGHT_COL = "Highlight_ID"

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

MAX_RADIUS = 2
N_PERMUTATIONS = 2048
MAPPING = False

RANDOM_STATE = 42

# OPTIMIZED PARAMETERS FROM PARAMETER SWEEP (2026-01-30)
# Condition-specific parameters for Panel 1 (Structural MAPchiral)
UMAP_STRUCTURAL_PARAMS = {
    "A": dict(n_neighbors=30, min_dist=0.20, n_components=2, random_state=RANDOM_STATE),
    "B": dict(n_neighbors=10, min_dist=0.01, n_components=2, random_state=RANDOM_STATE),
    "C": dict(n_neighbors=20, min_dist=0.20, n_components=2, random_state=RANDOM_STATE),
    "ALL": dict(n_neighbors=20, min_dist=0.20, n_components=2, random_state=RANDOM_STATE),
    "D": dict(n_neighbors=30, min_dist=0.20, n_components=2, random_state=RANDOM_STATE),
    "E": dict(n_neighbors=10, min_dist=0.01, n_components=2, random_state=RANDOM_STATE),
}

# Condition-specific parameters for Panel 2 (FPM-normalized)
UMAP_FPM_PARAMS = {
    "A": dict(n_neighbors=20, min_dist=0.10, n_components=2, random_state=RANDOM_STATE),
    "B": dict(n_neighbors=15, min_dist=0.15, n_components=2, random_state=RANDOM_STATE),
    "C": dict(n_neighbors=10, min_dist=0.30, n_components=2, random_state=RANDOM_STATE),
    "ALL": dict(n_neighbors=10, min_dist=0.30, n_components=2, random_state=RANDOM_STATE),
    "D": dict(n_neighbors=20, min_dist=0.10, n_components=2, random_state=RANDOM_STATE),
    "E": dict(n_neighbors=15, min_dist=0.15, n_components=2, random_state=RANDOM_STATE),
}

# Condition-specific parameters for Panel 3 (Descriptors)
# Optimized from parameter sweep (2026-01-30)
UMAP_DESCRIPTOR_PARAMS = {
    "A": dict(n_neighbors=15, min_dist=0.15, n_components=2, metric="euclidean", random_state=RANDOM_STATE, init="random"),
    "B": dict(n_neighbors=10, min_dist=0.05, n_components=2, metric="euclidean", random_state=RANDOM_STATE, init="random"),
    "C": dict(n_neighbors=50, min_dist=0.10, n_components=2, metric="euclidean", random_state=RANDOM_STATE, init="random"),
    "ALL": dict(n_neighbors=50, min_dist=0.10, n_components=2, metric="euclidean", random_state=RANDOM_STATE, init="random"),
    "D": dict(n_neighbors=15, min_dist=0.15, n_components=2, metric="euclidean", random_state=RANDOM_STATE, init="random"),
    "E": dict(n_neighbors=10, min_dist=0.05, n_components=2, metric="euclidean", random_state=RANDOM_STATE, init="random"),
}

DEDUPLICATE_FPS = True
REPORT_DESCRIPTOR_DUPLICATES = True
DESC_DUP_DECIMALS = 6

# FPM normalization (Fingerprints per Molecular weight)
ENABLE_FPM_NORMALIZATION = True  # Panel 2: complexity-density analysis

# Data filtering conditions
DATA_CONDITION = "D"  # Options: "A" (Lit+Hits), "B" (Lib+Hits), "C" (Lit+Lib+Hits), "ALL" (Lit+Lib+34Hits+Hits), "D" (Lit+34Hits+Hits), "E" (Lib+34Hits+Hits)

TOPK_NEIGHBORS = 50

# Enhanced color schemes
COLOR_LITERATURE = "#D0D0D0"  # light gray
COLOR_LIBRARY = "#1F77B4"     # muted blue
COLOR_34HITS = "#E41A1C"      # RED for 34_Hits
COLOR_BRAIN_6_4_4_13 = "#984EA3"  # PURPLE for Brain 6-4-4-13

# High-contrast hit colors (excluding red for 34_Hits and purple for Brain)
COLOR_HITS = [
    "#377EB8",  # blue
    "#FFFF33",  # yellow
    "#E41A1C",  # red
    "#4DAF4A",  # green
    "#A65628",  # brown
    "#F781BF",  # pink
    "#999999",  # gray
    "#66C2A5",  # teal
    "#FC8D62",  # coral
    "#FFFF33",  # yellow
    "#8DD3C7",  # cyan
]

# Highlight marker shapes and colors
HIGHLIGHT_MARKERS = {
    "Cyclosporin A": ("*", "black"),
    "Hexapeptide": ("s", "black"),
    "N-Me Hexapeptide": ("^", "black"),
}

HIGHLIGHT_MARKER_SIZE = 160
HIT_MARKER_SIZE = 120


# ==========================================================
# Helpers
# ==========================================================
def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

def norm_str(x) -> str:
    return "" if pd.isna(x) else str(x).strip()

def require_cols(df: pd.DataFrame, cols: list[str], where: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {where}: {missing}")


# ==========================================================
# MAPchiral fingerprinting + terminal summary ONLY
# ==========================================================
def compute_mapchiral_fps_with_terminal_summary(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    fps = []
    keep_idx = []
    failed_rows = []

    bad_smiles = 0
    bad_parse = 0

    for row in df.itertuples():
        i = row.Index
        smi = getattr(row, SMILES_COL)
        src = norm_str(getattr(row, SOURCE_COL))
        hid = norm_str(getattr(row, HIT_ID_COL))
        hlid = norm_str(getattr(row, HIGHLIGHT_COL))

        if not isinstance(smi, str) or not smi.strip():
            bad_smiles += 1
            failed_rows.append((i, src, hid, hlid, "empty/invalid_smiles", smi))
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            bad_parse += 1
            failed_rows.append((i, src, hid, hlid, "rdkit_parse_failed", smi))
            continue

        fp = encode(mol, max_radius=MAX_RADIUS, n_permutations=N_PERMUTATIONS, mapping=MAPPING)
        fps.append(np.asarray(fp, dtype=np.uint32))
        keep_idx.append(i)

    total = len(df)
    good = len(keep_idx)
    failed = bad_smiles + bad_parse

    print("\n" + "=" * 78)
    print("STEP 4 — MAPchiral Fingerprinting Summary (terminal only)")
    print("=" * 78)
    print(f"Total compounds in input:        {total:,}")
    print(f"Successfully fingerprinted:      {good:,} ({(100*good/total if total else 0):.1f}%)")
    print(f"Failed to fingerprint:           {failed:,} ({(100*failed/total if total else 0):.1f}%)")
    print(f"  - Empty/invalid SMILES:        {bad_smiles:,}")
    print(f"  - RDKit parsing failed:        {bad_parse:,}")

    if failed_rows:
        fail_df = pd.DataFrame(
            failed_rows,
            columns=["row_index", SOURCE_COL, HIT_ID_COL, HIGHLIGHT_COL, "reason", SMILES_COL],
        )
        print("\nFailures by Source:")
        print(fail_df[SOURCE_COL].value_counts(dropna=False).to_string())
        print("\nFailures by reason:")
        print(fail_df["reason"].value_counts(dropna=False).to_string())

        hit_fail = fail_df[fail_df[HIT_ID_COL].astype(str).str.strip() != ""]
        hl_fail = fail_df[fail_df[HIGHLIGHT_COL].astype(str).str.strip() != ""]
        if not hit_fail.empty:
            print("\n" + "!" * 78)
            print("ERROR: One or more HITS failed MAPchiral fingerprinting (would be excluded).")
            print("Fix these rows and re-run.")
            print("!" * 78)
            print(hit_fail.to_string(index=False))
            raise ValueError("Hit fingerprinting failure: at least one hit would be dropped.")
        if not hl_fail.empty:
            print("\n" + "!" * 78)
            print("ERROR: One or more HIGHLIGHTS failed MAPchiral fingerprinting (would be excluded).")
            print("Fix these rows and re-run.")
            print("!" * 78)
            print(hl_fail.to_string(index=False))
            raise ValueError("Highlight fingerprinting failure: at least one highlight would be dropped.")
    else:
        print("\nNo fingerprint failures detected 🎉")

    print("=" * 78 + "\n")

    if not fps:
        raise ValueError("No valid molecules fingerprinted. Check SMILES and input.")

    return np.vstack(fps), np.asarray(keep_idx, dtype=int)


# ==========================================================
# SAFE fingerprint dedup (preserve hits + highlights)
# ==========================================================
def deduplicate_fingerprints_safe(df_fp: pd.DataFrame, fps: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, int]:
    n_before = len(fps)

    df_fp = df_fp.copy()
    df_fp["_priority_hit"] = df_fp["_is_hit"].astype(int)
    df_fp["_priority_34hit"] = df_fp["_is_34hit"].astype(int)
    df_fp["_priority_high"] = df_fp["_is_highlight"].astype(int)
    df_fp["_orig_pos"] = np.arange(len(df_fp))

    # lexsort: last key is PRIMARY sort (most significant)
    # Priority: highlights > hits > 34_hits > original position
    order = np.lexsort((
        df_fp["_orig_pos"].values,           # quaternary (least significant)
        -df_fp["_priority_34hit"].values,    # tertiary
        -df_fp["_priority_hit"].values,      # secondary
        -df_fp["_priority_high"].values,     # primary (most significant)
    ))

    df_sorted = df_fp.iloc[order].reset_index(drop=True)
    fps_sorted = fps[order]

    fps_unique, unique_positions = np.unique(fps_sorted, axis=0, return_index=True)
    df_unique = df_sorted.iloc[unique_positions].copy().reset_index(drop=True)

    n_after = len(fps_unique)
    n_dups = n_before - n_after

    df_unique.drop(columns=["_priority_hit", "_priority_34hit", "_priority_high", "_orig_pos"], inplace=True, errors="ignore")

    hits_before = set(df_fp.loc[df_fp["_is_hit"], HIT_ID_COL].astype(str))
    hits_after = set(df_unique.loc[df_unique["_is_hit"], HIT_ID_COL].astype(str))
    if hits_before - hits_after:
        raise ValueError(f"Dedup would remove hit(s): {sorted(hits_before - hits_after)}")

    hits34_before = df_fp["_is_34hit"].sum()
    hits34_after = df_unique["_is_34hit"].sum()
    if hits34_before != hits34_after:
        print(f"WARNING: Dedup removed {hits34_before - hits34_after} 34_hits (duplicates with literature/library)")

    highs_before = set(df_fp.loc[df_fp["_is_highlight"], HIGHLIGHT_COL].astype(str))
    highs_after = set(df_unique.loc[df_unique["_is_highlight"], HIGHLIGHT_COL].astype(str))
    if highs_before - highs_after:
        raise ValueError(f"Dedup would remove highlight(s): {sorted(highs_before - highs_after)}")

    print("\n" + "=" * 78)
    print("Fingerprint Deduplication (SAFE):")
    print("=" * 78)
    print(f"Fingerprints before dedup:  {n_before:,}")
    print(f"Fingerprints after dedup:   {n_after:,}")
    print(f"Duplicate fingerprints:     {n_dups:,}")
    print("=" * 78 + "\n")

    return df_unique, fps_unique, n_dups


# ==========================================================
# FPM Normalization (Fingerprints per Molecular weight)
# ==========================================================
def normalize_fingerprints_by_fpm(df: pd.DataFrame, fps: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize fingerprints by molecular weight to get complexity density.
    
    Returns:
        fps_normalized: FPM-normalized fingerprints
        on_bits: Count of non-zero elements per fingerprint
        fpm: Fingerprints per molecular weight values
    """
    # Get molecular weights
    if "Total Molweight" not in df.columns:
        raise ValueError("Total Molweight column not found in dataframe")
    
    mw = df["Total Molweight"].astype(float).values
    
    # Count non-zero elements (structural features present)
    on_bits = np.count_nonzero(fps, axis=1)
    
    # Calculate FPM (features per unit mass)
    fpm = on_bits / mw
    
    # Normalize each fingerprint by its molecular weight
    # This makes fingerprints comparable based on structural density, not absolute size
    fps_normalized = fps / mw[:, np.newaxis]
    
    print("\n" + "=" * 78)
    print("FPM Normalization Summary")
    print("=" * 78)
    print(f"Molecular weight range:  {mw.min():.1f} - {mw.max():.1f} Da")
    print(f"On-bits range:           {on_bits.min()} - {on_bits.max()}")
    print(f"FPM range:               {fpm.min():.6f} - {fpm.max():.6f}")
    print(f"FPM mean ± std:          {fpm.mean():.6f} ± {fpm.std():.6f}")
    print("=" * 78 + "\n")
    
    return fps_normalized, on_bits, fpm


# ==========================================================
# Custom MinHash distance (Numba-jitted)
# ==========================================================
@njit
def minhash_distance(fp_a, fp_b) -> float:
    m = fp_a.shape[0]
    eq = 0
    for k in range(m):
        if fp_a[k] == fp_b[k]:
            eq += 1
    return 1.0 - (eq / m)


# ==========================================================
# Descriptor duplicates (terminal total only)
# ==========================================================
def print_total_descriptor_duplicates(df: pd.DataFrame, decimals: int = 6):
    X = df[DESC_COLS].astype(float).values
    Xr = np.round(X, decimals=decimals)
    tuples = [tuple(row) for row in Xr]
    counts = pd.Series(tuples).value_counts()
    dup_extra_rows = int((counts[counts > 1].sum()) - (counts > 1).sum())

    print("\n" + "=" * 78)
    print("Descriptor duplicate check (terminal total only)")
    print("=" * 78)
    print(f"Total duplicate rows (extra), rounded @ {decimals} dp: {dup_extra_rows:,}")
    print("=" * 78 + "\n")


# ==========================================================
# Enhanced plotting: publication-ready with proper legend + zoom inset
# ==========================================================
def plot_umap_enhanced(
    df_umap: pd.DataFrame,
    xcol: str,
    ycol: str,
    title: str,
    outpath_svg: str,
    hit_color_map: dict,
    condition_code: str = "ALL",
    add_zoom_inset: bool = False,
):
    """
    Enhanced UMAP plot with 34_Hits (RED) and Brain 6-4-4-13 (PURPLE)
    Optionally adds a zoomed inset panel for crowded regions
    """
    fig, ax = plt.subplots(figsize=(12, 10), dpi=100)
    
    src = df_umap[SOURCE_COL].astype(str).str.strip().str.lower()

    # Background: literature + library
    lit = df_umap[src == "literature"]
    lib = df_umap[src == "library"]
    hits_34 = df_umap[df_umap["_is_34hit"]]
    
    if not lit.empty:
        ax.scatter(lit[xcol], lit[ycol], s=20, alpha=0.2, c=COLOR_LITERATURE, 
                  label="Literature", rasterized=True)
    if not lib.empty:
        ax.scatter(lib[xcol], lib[ycol], s=20, alpha=0.15, c=COLOR_LIBRARY, 
                  label="Library", rasterized=True)
    if not hits_34.empty:
        ax.scatter(hits_34[xcol], hits_34[ycol], s=80, alpha=0.85, c=COLOR_34HITS, 
                  label="34_Hits", rasterized=True, zorder=3, edgecolors='darkred', linewidths=0.5)

    # Highlights: distinct black shapes
    high = df_umap[df_umap["_is_highlight"]]
    if not high.empty:
        for hlid, sub in high.groupby(HIGHLIGHT_COL):
            hlid = norm_str(hlid) or "Highlight"
            marker, color = HIGHLIGHT_MARKERS.get(hlid, ("o", "black"))
            ax.scatter(
                sub[xcol], sub[ycol],
                s=HIGHLIGHT_MARKER_SIZE, marker=marker, alpha=0.85,
                c=color, edgecolors="black", linewidths=1.5,
                label=f"Highlight: {hlid}",
                zorder=5
            )

    # Hits: X markers with high-contrast colors (including Brain in PURPLE)
    hits = df_umap[df_umap["_is_hit"]]
    if not hits.empty:
        for hid, sub in hits.groupby(HIT_ID_COL):
            hid = norm_str(hid) or "Hit"
            color = hit_color_map.get(hid, "#000000")
            ax.scatter(
                sub[xcol], sub[ycol],
                s=HIT_MARKER_SIZE, marker="X", alpha=0.95,
                c=color, edgecolors="black", linewidths=1.2,
                label=f"Hit: {hid}",
                zorder=7
            )

    ax.set_xlabel(f"{xcol} →", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"{ycol} →", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    
    ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add zoomed inset if requested (for crowded regions)
    if add_zoom_inset and not hits_34.empty and not hits.empty:
        # Calculate bounding box for 34_hits + hits
        combined = pd.concat([hits_34, hits])
        x_min, x_max = combined[xcol].min(), combined[xcol].max()
        y_min, y_max = combined[ycol].min(), combined[ycol].max()
        
        # Add margin (20% of range)
        x_range = x_max - x_min
        y_range = y_max - y_min
        margin_x = x_range * 0.2 if x_range > 0 else 0.5
        margin_y = y_range * 0.2 if y_range > 0 else 0.5
        
        zoom_x_min = x_min - margin_x
        zoom_x_max = x_max + margin_x
        zoom_y_min = y_min - margin_y
        zoom_y_max = y_max + margin_y
        
        # Create inset axes in upper left corner
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        axins = inset_axes(ax, width="35%", height="35%", loc='upper left', 
                          borderpad=2.5)
        
        # Plot same data in inset with larger markers
        lit_zoom = lit[(lit[xcol] >= zoom_x_min) & (lit[xcol] <= zoom_x_max) & 
                       (lit[ycol] >= zoom_y_min) & (lit[ycol] <= zoom_y_max)]
        lib_zoom = lib[(lib[xcol] >= zoom_x_min) & (lib[xcol] <= zoom_x_max) & 
                       (lib[ycol] >= zoom_y_min) & (lib[ycol] <= zoom_y_max)]
        
        if not lit_zoom.empty:
            axins.scatter(lit_zoom[xcol], lit_zoom[ycol], s=30, alpha=0.3, 
                         c=COLOR_LITERATURE, rasterized=True)
        if not lib_zoom.empty:
            axins.scatter(lib_zoom[xcol], lib_zoom[ycol], s=30, alpha=0.25, 
                         c=COLOR_LIBRARY, rasterized=True)
        
        # 34_hits larger in zoom
        axins.scatter(hits_34[xcol], hits_34[ycol], s=120, alpha=0.9, 
                     c=COLOR_34HITS, zorder=8, edgecolors='darkred', linewidths=1,
                     label="34_Hits (zoomed)")
        
        # Hits in zoom
        for hid, sub in hits.groupby(HIT_ID_COL):
            hid = norm_str(hid) or "Hit"
            color = hit_color_map.get(hid, "#000000")
            axins.scatter(sub[xcol], sub[ycol], s=130, marker="X", alpha=0.95,
                         c=color, edgecolors="black", linewidths=1.5, zorder=7)
        
        # Set zoom limits
        axins.set_xlim(zoom_x_min, zoom_x_max)
        axins.set_ylim(zoom_y_min, zoom_y_max)
        axins.set_xlabel("", fontsize=8)
        axins.set_ylabel("", fontsize=8)
        axins.tick_params(labelsize=7)
        axins.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
        
        # Add zoom indicator box on main plot
        from matplotlib.patches import Rectangle
        zoom_box = Rectangle((zoom_x_min, zoom_y_min), 
                            zoom_x_max - zoom_x_min, 
                            zoom_y_max - zoom_y_min,
                            fill=False, edgecolor='black', linewidth=1.5, 
                            linestyle='--', zorder=10)
        ax.add_patch(zoom_box)
        
        # Add "ZOOM" label
        axins.text(0.95, 0.05, 'ZOOM', transform=axins.transAxes,
                  fontsize=9, fontweight='bold', ha='right', va='bottom',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                           edgecolor='black', alpha=0.8))
    
    # Legend positioned to not obscure data
    ax.legend(loc="upper right", fontsize=9, frameon=True, fancybox=True, 
             shadow=True, framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(outpath_svg, format="svg", bbox_inches="tight")
    plt.close(fig)


# ==========================================================
# Build hit color map (high-contrast palette)
# ==========================================================
def build_hit_color_map(df: pd.DataFrame) -> dict:
    """Build color mapping for hits and Brain 6-4-4-13"""
    color_map = {}
    
    # Handle Brain 6-4-4-13 specifically (PURPLE)
    brain_hits = df[df[HIT_ID_COL].astype(str).str.contains("Brain 6-4-4-13", case=False, na=False)]
    for hit_id in brain_hits[HIT_ID_COL].unique():
        if hit_id and str(hit_id).strip():
            color_map[str(hit_id).strip()] = COLOR_BRAIN_6_4_4_13
    
    # Handle other regular hits (multi-color palette)
    df_hits = df[df["_is_hit"]].copy()
    regular_hits = []
    for hit_id in df_hits[HIT_ID_COL].unique():
        if hit_id and str(hit_id).strip():
            hit_str = str(hit_id).strip()
            # Skip if already assigned (34_hits or Brain)
            if hit_str not in color_map:
                regular_hits.append(hit_str)
    
    # Assign colors to regular hits
    for i, hit_id in enumerate(regular_hits):
        color_map[hit_id] = COLOR_HITS[i % len(COLOR_HITS)]
    
    print(f"\nHit Color Assignments:")
    print(f"  Brain 6-4-4-13: {sum(1 for c in color_map.values() if c == COLOR_BRAIN_6_4_4_13)} (PURPLE)")
    print(f"  Other hits: {len(regular_hits)}")
    
    return color_map


# ==========================================================
# Fingerprint-space NN per hit (MinHash NNDescent)
# ==========================================================
def fingerprint_space_neighbors_per_hit(df_fp: pd.DataFrame, fps: np.ndarray, topk: int) -> pd.DataFrame:
    index = NNDescent(
        fps,
        metric=minhash_distance,
        n_neighbors=max(topk + 10, 40),
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    hit_pos = np.where(df_fp["_is_hit"].to_numpy())[0]
    if len(hit_pos) == 0:
        return pd.DataFrame()

    inds, dists = index.query(fps[hit_pos], k=topk + 1)
    rows = []

    for qi, pos in enumerate(hit_pos):
        hit_row = df_fp.iloc[pos]
        hit_id = norm_str(hit_row[HIT_ID_COL]) or "Hit"

        rank = 0
        for j, dist in zip(inds[qi], dists[qi]):
            j = int(j)
            if j == pos:
                continue
            rank += 1
            nb = df_fp.iloc[j]
            rows.append({
                "hit_id": hit_id,
                "hit_smiles": hit_row[SMILES_COL],
                "hit_source": hit_row[SOURCE_COL],
                "rank": rank,
                "neighbor_smiles": nb[SMILES_COL],
                "neighbor_source": nb[SOURCE_COL],
                "neighbor_hit_id": nb[HIT_ID_COL],
                "neighbor_highlight_id": nb[HIGHLIGHT_COL],
                "minhash_distance": float(dist),
            })
            if rank >= topk:
                break

    return pd.DataFrame(rows)


# ==========================================================
# Enhanced faithfulness: Spearman + rank correlation agreement + overlap
# ==========================================================
def local_map_faithfulness_per_hit(
    df_fp: pd.DataFrame,
    fps: np.ndarray,
    topk: int
) -> pd.DataFrame:
    """
    Compute local faithfulness with:
    - Spearman ρ (correlation of distances)
    - Rank correlation agreement (Kendall-tau proxy via rank consistency)
    - UMAP-topK overlap fraction
    - Neighbor purity (fraction of neighbors that are hits/highlights)
    """
    index = NNDescent(
        fps,
        metric=minhash_distance,
        n_neighbors=max(topk + 10, 40),
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    hit_pos = np.where(df_fp["_is_hit"].to_numpy())[0]
    if len(hit_pos) == 0:
        return pd.DataFrame()

    inds, dists = index.query(fps[hit_pos], k=topk + 1)
    xy = df_fp[["UMAP_MAPCHIRAL_1", "UMAP_MAPCHIRAL_2"]].to_numpy(dtype=float)

    rows = []
    for qi, pos in enumerate(hit_pos):
        hit_row = df_fp.iloc[pos]
        hit_id = norm_str(hit_row[HIT_ID_COL]) or "Hit"
        hit_xy = xy[pos]

        # MinHash neighbor set (topk)
        neigh_idx = []
        mh_dist = []
        for j, dist in zip(inds[qi], dists[qi]):
            j = int(j)
            if j == pos:
                continue
            neigh_idx.append(j)
            mh_dist.append(float(dist))
            if len(neigh_idx) >= topk:
                break

        n_used = len(neigh_idx)
        if n_used < 5:
            rows.append({
                "hit_id": hit_id,
                "hit_smiles": hit_row[SMILES_COL],
                "n_neighbors_used": n_used,
                "spearman_rho": np.nan,
                "spearman_p": np.nan,
                "rank_agreement": np.nan,
                "umap_topk_overlap_fraction": np.nan,
                "neighbor_purity": np.nan,
            })
            continue

        # UMAP distances to those MinHash neighbors
        neigh_xy = xy[neigh_idx]
        umap_dist_to_mh = np.sqrt(((neigh_xy - hit_xy) ** 2).sum(axis=1)).astype(float)

        rho, p = spearmanr(mh_dist, umap_dist_to_mh)

        # Rank agreement: correlation between rank orders
        mh_ranks = rankdata(mh_dist)
        umap_ranks = rankdata(umap_dist_to_mh)
        rank_agree = np.corrcoef(mh_ranks, umap_ranks)[0, 1]

        # UMAP-distance neighbors (topk) over all points (exclude self)
        all_umap_dist = np.sqrt(((xy - hit_xy) ** 2).sum(axis=1))
        all_umap_dist[pos] = np.inf
        umap_topk_idx = np.argpartition(all_umap_dist, topk)[:topk]
        umap_topk_set = set(int(i) for i in umap_topk_idx)

        mh_set = set(neigh_idx)
        overlap = len(mh_set & umap_topk_set)
        overlap_frac = overlap / topk

        # Neighbor purity (fraction of neighbors that are hits or highlights)
        neighbors_are_hit_or_hl = df_fp.iloc[neigh_idx][["_is_hit", "_is_highlight"]].any(axis=1).sum()
        purity = neighbors_are_hit_or_hl / n_used

        rows.append({
            "hit_id": hit_id,
            "hit_smiles": hit_row[SMILES_COL],
            "n_neighbors_used": n_used,
            "spearman_rho": float(rho) if rho is not None else np.nan,
            "spearman_p": float(p) if p is not None else np.nan,
            "rank_agreement": float(rank_agree) if np.isfinite(rank_agree) else np.nan,
            "umap_topk_overlap_fraction": float(overlap_frac),
            "neighbor_purity": float(purity),
        })

    return pd.DataFrame(rows).sort_values("spearman_rho", ascending=False)


# ==========================================================
# MAIN
# ==========================================================
def main():
    ensure_dirs()

    df = pd.read_csv(INPUT_CSV, sep=None, engine="python")

    require_cols(df, [SMILES_COL, SOURCE_COL, HIT_ID_COL, HIGHLIGHT_COL], "master CSV")
    require_cols(df, DESC_COLS, "master CSV (descriptors)")

    df[SOURCE_COL] = df[SOURCE_COL].astype(str).str.strip().str.lower()
    df[HIT_ID_COL] = df[HIT_ID_COL].apply(norm_str)
    df[HIGHLIGHT_COL] = df[HIGHLIGHT_COL].apply(norm_str)

    df["_is_hit"] = df[HIT_ID_COL].astype(str).str.strip().ne("")
    df["_is_34hit"] = df[SOURCE_COL] == "34_hits"
    df["_is_highlight"] = df[HIGHLIGHT_COL].astype(str).str.strip().ne("")

    # Apply data filtering based on condition
    condition_map = {
        "A": (["literature", "hit"], "Literature + Hits"),
        "B": (["library", "hit"], "Library + Hits"),
        "C": (["literature", "library", "hit"], "Literature + Library + Hits"),
        "ALL": (None, "Everything (Literature + Library + 34_Hits + Hits)"),
        "D": (["literature", "hit", "34_hits"], "Literature + 34_Hits + Hits"),
        "E": (["library", "hit", "34_hits"], "Library + 34_Hits + Hits"),
    }
    
    if DATA_CONDITION not in condition_map:
        raise ValueError(f"Unknown DATA_CONDITION: {DATA_CONDITION}")
    
    sources, condition_desc = condition_map[DATA_CONDITION]
    n_before_filter = len(df)
    
    if sources is not None:
        df = df[df[SOURCE_COL].isin(sources)].copy()
    
    n_after_filter = len(df)
    
    print(f"\nCondition {DATA_CONDITION}: {condition_desc}")
    if DATA_CONDITION != "ALL":
        print(f"Filtered: {n_before_filter:,} → {n_after_filter:,} compounds ({n_before_filter - n_after_filter:,} removed)")

    print(f"Hits detected:       {df['_is_hit'].sum():,}")
    print(f"34_Hits detected:    {df['_is_34hit'].sum():,}")
    print(f"Highlights detected: {df['_is_highlight'].sum():,}")

    # Build high-contrast hit color map
    hit_color_map = build_hit_color_map(df)

    # STEP 4 — MAPchiral fingerprinting
    fps, keep_idx = compute_mapchiral_fps_with_terminal_summary(df)
    df_fp = df.loc[keep_idx].copy().reset_index(drop=True)

    if DEDUPLICATE_FPS:
        df_fp, fps, _ = deduplicate_fingerprints_safe(df_fp, fps)

    # STEP 5 — Structural UMAP (Panel 1)
    structural_params = UMAP_STRUCTURAL_PARAMS[DATA_CONDITION]
    print(f"STEP 5 — Fitting UMAP (fingerprints; custom MinHash metric; no NxN)")
    print(f"Using optimized parameters for condition {DATA_CONDITION}: n_neighbors={structural_params['n_neighbors']}, min_dist={structural_params['min_dist']:.2f}")
    reducer_fp = umap.UMAP(metric=minhash_distance, **structural_params)
    emb_fp = reducer_fp.fit_transform(fps)

    # Embedding uniformity summary
    x_range = emb_fp[:, 0].max() - emb_fp[:, 0].min()
    y_range = emb_fp[:, 1].max() - emb_fp[:, 1].min()
    range_ratio = x_range / y_range if y_range != 0 else float("inf")
    print(f"X range: {x_range:.2f}")
    print(f"Y range: {y_range:.2f}")
    print(f"Range ratio: {range_ratio:.4f}")

    df_fp["UMAP_MAPCHIRAL_1"] = emb_fp[:, 0]
    df_fp["UMAP_MAPCHIRAL_2"] = emb_fp[:, 1]

    out_csv_fp = os.path.join(OUTPUT_DIR, f"structural_umap_mapchiral_{DATA_CONDITION}.csv")
    df_fp.to_csv(out_csv_fp, index=False)
    print(f"Saved structural UMAP CSV: {out_csv_fp}")

    out_svg_fp = os.path.join(FIG_DIR, f"structural_umap_mapchiral_{DATA_CONDITION}_hits_highlights.svg")
    plot_umap_enhanced(
        df_fp, "UMAP_MAPCHIRAL_1", "UMAP_MAPCHIRAL_2",
        f"MAPChiral Fingerprint Space - Condition {DATA_CONDITION} ({condition_desc})",
        out_svg_fp, hit_color_map, DATA_CONDITION
    )
    print(f"Saved structural UMAP SVG: {out_svg_fp}")

    # STEP 5B — FPM-normalized UMAP (Panel 2: complexity density analysis)
    if ENABLE_FPM_NORMALIZATION:
        fpm_params = UMAP_FPM_PARAMS[DATA_CONDITION]
        print("STEP 5B — Fitting UMAP (FPM-normalized fingerprints; complexity density)")
        print(f"Using optimized parameters for condition {DATA_CONDITION}: n_neighbors={fpm_params['n_neighbors']}, min_dist={fpm_params['min_dist']:.2f}")
        fps_fpm, on_bits, fpm = normalize_fingerprints_by_fpm(df_fp, fps)
        
        # Store FPM values for analysis
        df_fp["FP_on_bits"] = on_bits
        df_fp["FPM"] = fpm
        
        # Fit UMAP on FPM-normalized fingerprints
        reducer_fpm = umap.UMAP(metric=minhash_distance, **fpm_params)
        emb_fpm = reducer_fpm.fit_transform(fps_fpm)
        
        # Embedding uniformity summary
        x_range_fpm = emb_fpm[:, 0].max() - emb_fpm[:, 0].min()
        y_range_fpm = emb_fpm[:, 1].max() - emb_fpm[:, 1].min()
        range_ratio_fpm = x_range_fpm / y_range_fpm if y_range_fpm != 0 else float("inf")
        print(f"X range (FPM): {x_range_fpm:.2f}")
        print(f"Y range (FPM): {y_range_fpm:.2f}")
        print(f"Range ratio (FPM): {range_ratio_fpm:.4f}")
        
        df_fp["UMAP_FPM_1"] = emb_fpm[:, 0]
        df_fp["UMAP_FPM_2"] = emb_fpm[:, 1]
        
        out_csv_fpm = os.path.join(OUTPUT_DIR, f"structural_umap_fpm_normalized_{DATA_CONDITION}.csv")
        df_fp.to_csv(out_csv_fpm, index=False)
        print(f"Saved FPM-normalized UMAP CSV: {out_csv_fpm}")
        
        out_svg_fpm = os.path.join(FIG_DIR, f"structural_umap_fpm_normalized_{DATA_CONDITION}_hits_highlights.svg")
        plot_umap_enhanced(
            df_fp, "UMAP_FPM_1", "UMAP_FPM_2",
            f"FPM Normalized Fingerprint Space - Condition {DATA_CONDITION} ({condition_desc})",
            out_svg_fpm, hit_color_map, DATA_CONDITION
        )
        print(f"Saved FPM-normalized UMAP SVG: {out_svg_fpm}")

    # STEP 6 — Descriptor duplicates + Descriptor UMAP
    if REPORT_DESCRIPTOR_DUPLICATES:
        print_total_descriptor_duplicates(df, decimals=DESC_DUP_DECIMALS)

    descriptor_params = UMAP_DESCRIPTOR_PARAMS[DATA_CONDITION]
    print(f"STEP 6B — Fitting UMAP (descriptors)")
    print(f"Using optimized parameters for condition {DATA_CONDITION}: n_neighbors={descriptor_params['n_neighbors']}, min_dist={descriptor_params['min_dist']:.2f}")
    df_desc = df.copy()
    
    # Debug: Check 34_hits presence in descriptor space
    print(f"Panel 3 data: {len(df_desc)} compounds, {df_desc['_is_34hit'].sum()} are 34_hits")
    
    X = df_desc[DESC_COLS].astype(float).values
    Xs = StandardScaler().fit_transform(X)

    reducer_desc = umap.UMAP(**descriptor_params)
    emb_desc = reducer_desc.fit_transform(Xs)

    df_desc["UMAP_DESC_1"] = emb_desc[:, 0]
    df_desc["UMAP_DESC_2"] = emb_desc[:, 1]

    out_csv_desc = os.path.join(OUTPUT_DIR, f"descriptor_umap_8desc_{DATA_CONDITION}.csv")
    df_desc.to_csv(out_csv_desc, index=False)
    print(f"Saved descriptor UMAP CSV: {out_csv_desc}")

    out_svg_desc = os.path.join(FIG_DIR, f"descriptor_umap_8desc_{DATA_CONDITION}_hits_highlights.svg")
    # Enable zoom inset for Panel 3 if 34_hits are present
    enable_zoom = df_desc["_is_34hit"].sum() > 0
    plot_umap_enhanced(
        df_desc, "UMAP_DESC_1", "UMAP_DESC_2",
        f"Property Space - Condition {DATA_CONDITION} ({condition_desc})",
        out_svg_desc, hit_color_map, DATA_CONDITION,
        add_zoom_inset=enable_zoom
    )
    print(f"Saved descriptor UMAP SVG: {out_svg_desc}")

    # STEP 7 — Fingerprint-space neighbors per hit
    print("STEP 7 — Fingerprint-space neighbors per hit (MinHash NN)")
    nn_fp = fingerprint_space_neighbors_per_hit(df_fp, fps, topk=TOPK_NEIGHBORS)

    if not nn_fp.empty:
        nn_fp_path = os.path.join(OUTPUT_DIR, "hit_neighbors_fingerprint_space.csv")
        nn_fp.to_csv(nn_fp_path, index=False)
        print(f"Saved hit neighbors CSV: {nn_fp_path}")
    else:
        print("No hits detected; no neighbors CSV written.")

    # STEP 8 — Enhanced local map faithfulness
    print("STEP 8 — Local map faithfulness per hit (enhanced metrics)")
    faith = local_map_faithfulness_per_hit(df_fp, fps, topk=TOPK_NEIGHBORS)

    if not faith.empty:
        faith_path = os.path.join(OUTPUT_DIR, "hit_local_map_faithfulness.csv")
        faith.to_csv(faith_path, index=False)
        print(f"Saved faithfulness CSV: {faith_path}")

        print("\nFaithfulness summary (Spearman ρ):")
        print(faith["spearman_rho"].describe().to_string())
        
        print("\nRank agreement summary (Pearson correlation of ranks):")
        print(faith["rank_agreement"].describe().to_string())
        
        print("\nUMAP-topK overlap summary:")
        print(faith["umap_topk_overlap_fraction"].describe().to_string())
        
        print("\nNeighbor purity summary:")
        print(faith["neighbor_purity"].describe().to_string())
        
        low_faith = (faith["spearman_rho"] < 0.4).sum()
        print(f"\nHits with low faithfulness (<0.4): {low_faith}")
        
        print("\nLowest-rho hits (most locally distorted):")
        print(faith.nsmallest(min(5, len(faith)), "spearman_rho")[
            ["hit_id", "spearman_rho", "rank_agreement", "umap_topk_overlap_fraction", "neighbor_purity"]
        ].to_string(index=False))
    else:
        print("No hits detected; no faithfulness CSV written.")

    print("\nDONE ✅")


if __name__ == "__main__":
    main()