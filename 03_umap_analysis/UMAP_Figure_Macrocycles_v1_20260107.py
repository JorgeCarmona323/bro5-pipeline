"""
ONE SCRIPT TO RULE THEM ALL (30k–50k scale) — ENHANCED VERSION

Key improvements:
  - Better UMAP plots: size scaling, grid, legend positioning
  - Local density metrics: isolation score + neighborhood purity
  - Spearman correlation + rank correlation agreement
  - Publication-ready SVG with improved aesthetics
"""

import os
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

import umap
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch

from sklearn.preprocessing import StandardScaler
from numba import njit

from mapchiral.mapchiral import encode

from pynndescent import NNDescent
from scipy.stats import spearmanr, rankdata


# ==========================================================
# USER CONFIG (EDIT THIS SECTION)
# ==========================================================
INPUT_CSV = "C:\\Users\\Admin\\Documents\\Hu Lab\\Code\\Python\\rdkit\\Data\\2026-01-09\\canonicalized_master_macrocycles_2D_Descriptors_20260106.csv"
OUTPUT_DIR = "C:\\Users\\Admin\\Documents\\Hu Lab\\Code\\Python\\rdkit\\Data\\2026-01-09\\Output"
FIG_DIR = "C:\\Users\\Admin\\Documents\\Hu Lab\\Code\\Python\\rdkit\\Data\\2026-01-09\\Figures"

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

UMAP_FP_PARAMS = dict(
    n_neighbors=15,   # configuration optimized for better local structure. JOC 20260112 J
    min_dist=0.05,   
    n_components=2,
    random_state=RANDOM_STATE,
    # metric set later (custom)
)

UMAP_DESC_PARAMS = dict(
    n_neighbors=40,
    min_dist=0.30,
    n_components=2,
    metric="euclidean",
    random_state=RANDOM_STATE,
    init="random",
)

DEDUPLICATE_FPS = True
REPORT_DESCRIPTOR_DUPLICATES = True
DESC_DUP_DECIMALS = 6

TOPK_NEIGHBORS = 50

# Enhanced color schemes
COLOR_LITERATURE =  "#D0D0D0"   # light gray
COLOR_LIBRARY = "#1f77b4" 
# High-contrast hit colors (perceptually distinct, publication-ready)
COLOR_HITS = [
    "#E41A1C",  # red
    "#377EB8",  # blue
    "#4DAF4A",  # green
    "#984EA3",  # purple
    "#FF7F00",  # orange
    "#A65628",  # brown
    "#F781BF",  # pink
    "#999999",  # gray
    "#66C2A5",  # teal
    "#FC8D62",  # coral
]

# Highlight marker shapes and colors
HIGHLIGHT_MARKERS = {
    "cyclosporin A": ("*", "black"),
    "hexapeptide": ("s", "black"),
    "n-me hexapeptide": ("^", "black"),
}

HIGHLIGHT_MARKER_SIZE = 300
HIT_MARKER_SIZE = 160


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

    for i, row in df.iterrows():
        smi = row[SMILES_COL]
        src = norm_str(row[SOURCE_COL])
        hid = norm_str(row[HIT_ID_COL])
        hlid = norm_str(row[HIGHLIGHT_COL])

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
    df_fp["_priority_high"] = df_fp["_is_highlight"].astype(int)
    df_fp["_orig_pos"] = np.arange(len(df_fp))

    order = np.lexsort((
        df_fp["_orig_pos"].values,
        -df_fp["_priority_high"].values,
        -df_fp["_priority_hit"].values,
    ))

    df_sorted = df_fp.iloc[order].reset_index(drop=True)
    fps_sorted = fps[order]

    fps_unique, unique_positions = np.unique(fps_sorted, axis=0, return_index=True)
    df_unique = df_sorted.iloc[unique_positions].copy().reset_index(drop=True)

    n_after = len(fps_unique)
    n_dups = n_before - n_after

    df_unique.drop(columns=["_priority_hit", "_priority_high", "_orig_pos"], inplace=True, errors="ignore")

    hits_before = set(df_fp.loc[df_fp["_is_hit"], HIT_ID_COL].astype(str))
    hits_after = set(df_unique.loc[df_unique["_is_hit"], HIT_ID_COL].astype(str))
    if hits_before - hits_after:
        raise ValueError(f"Dedup would remove hit(s): {sorted(hits_before - hits_after)}")

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
# Enhanced plotting: publication-ready with proper legend
# ==========================================================
def plot_umap_enhanced(
    df_umap: pd.DataFrame,
    xcol: str,
    ycol: str,
    title: str,
    outpath_svg: str,
    hit_color_map: dict,
):
    """
    Enhanced UMAP plot with:
    - Better background contrast
    - Clearly visible hit markers (X) with high-contrast colors
    - Distinct highlight shapes (*, s, ^) in black
    - Proper legend and grid
    - Publication-ready aesthetics
    """
    fig, ax = plt.subplots(figsize=(12, 10), dpi=100)
    
    src = df_umap[SOURCE_COL].astype(str).str.strip().str.lower()

    # Background: literature + library
    lit = df_umap[src == "literature"]
    lib = df_umap[src == "library"]
    
    if not lit.empty:
        ax.scatter(lit[xcol], lit[ycol], s=20, alpha=0.2, c=COLOR_LITERATURE, 
                  label="Literature", rasterized=True)
    if not lib.empty:
        ax.scatter(lib[xcol], lib[ycol], s=20, alpha=0.15, c=COLOR_LIBRARY, 
                  label="Library", rasterized=True)

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
                zorder=4
            )

    # Hits: X markers with high-contrast colors
    hits = df_umap[df_umap["_is_hit"]]
    if not hits.empty:
        for hid, sub in hits.groupby(HIT_ID_COL):
            hid = norm_str(hid) or "Hit"
            color = hit_color_map.get(hid, "#000000")
            ax.scatter(
                sub[xcol], sub[ycol],
                s=HIT_MARKER_SIZE, marker="X", alpha=0.95,
                c=color, edgecolors="black", linewidths=0.8,
                label=f"Hit: {hid}",
                zorder=5
            )

    ax.set_xlabel(f"{xcol} →", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"{ycol} →", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    
    ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Legend positioned to not obscure data
    ax.legend(loc="upper right", fontsize=9, frameon=True, fancybox=True, 
             shadow=True, framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(outpath_svg, format="svg", bbox_inches="tight")
    plt.close()


# ==========================================================
# Build hit color map (high-contrast palette)
# ==========================================================
def build_hit_color_map(df: pd.DataFrame) -> dict:
    hit_ids = sorted({h for h in df.loc[df["_is_hit"], HIT_ID_COL].unique() if h})
    hit_color_map = {hid: COLOR_HITS[i % len(COLOR_HITS)] for i, hid in enumerate(hit_ids)}
    
    print("\nHit color assignments:")
    for hid, color in hit_color_map.items():
        print(f"  {hid}: {color}")
    
    return hit_color_map


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
    df["_is_highlight"] = df[HIGHLIGHT_COL].astype(str).str.strip().ne("")

    print(f"Hits detected:       {df['_is_hit'].sum():,}")
    print(f"Highlights detected: {df['_is_highlight'].sum():,}")

    # Build high-contrast hit color map
    hit_color_map = build_hit_color_map(df)

    # STEP 4 — MAPchiral fingerprinting
    fps, keep_idx = compute_mapchiral_fps_with_terminal_summary(df)
    df_fp = df.loc[keep_idx].copy().reset_index(drop=True)

    if DEDUPLICATE_FPS:
        df_fp, fps, _ = deduplicate_fingerprints_safe(df_fp, fps)

    # STEP 5 — Structural UMAP
    print("STEP 5 — Fitting UMAP (fingerprints; custom MinHash metric; no NxN)")
    reducer_fp = umap.UMAP(metric=minhash_distance, **UMAP_FP_PARAMS)
    emb_fp = reducer_fp.fit_transform(fps)

    df_fp["UMAP_MAPCHIRAL_1"] = emb_fp[:, 0]
    df_fp["UMAP_MAPCHIRAL_2"] = emb_fp[:, 1]

    out_csv_fp = os.path.join(OUTPUT_DIR, "structural_umap_mapchiral.csv")
    df_fp.to_csv(out_csv_fp, index=False)
    print(f"Saved structural UMAP CSV: {out_csv_fp}")

    out_svg_fp = os.path.join(FIG_DIR, "structural_umap_mapchiral_hits_highlights.svg")
    plot_umap_enhanced(
        df_fp, "UMAP_MAPCHIRAL_1", "UMAP_MAPCHIRAL_2",
        "Structural Space (MAPchiral + UMAP) with Hits and Highlights",
        out_svg_fp, hit_color_map
    )
    print(f"Saved structural UMAP SVG: {out_svg_fp}")

    # STEP 6 — Descriptor duplicates + Descriptor UMAP
    if REPORT_DESCRIPTOR_DUPLICATES:
        print_total_descriptor_duplicates(df, decimals=DESC_DUP_DECIMALS)

    print("STEP 6B — Fitting UMAP (descriptors)")
    df_desc = df.copy()
    X = df_desc[DESC_COLS].astype(float).values
    Xs = StandardScaler().fit_transform(X)

    reducer_desc = umap.UMAP(**UMAP_DESC_PARAMS)
    emb_desc = reducer_desc.fit_transform(Xs)

    df_desc["UMAP_DESC_1"] = emb_desc[:, 0]
    df_desc["UMAP_DESC_2"] = emb_desc[:, 1]

    out_csv_desc = os.path.join(OUTPUT_DIR, "descriptor_umap_6desc.csv")
    df_desc.to_csv(out_csv_desc, index=False)
    print(f"Saved descriptor UMAP CSV: {out_csv_desc}")

    out_svg_desc = os.path.join(FIG_DIR, "descriptor_umap_6desc_hits_highlights.svg")
    plot_umap_enhanced(
        df_desc, "UMAP_DESC_1", "UMAP_DESC_2",
        "Property Space (6 descriptors + UMAP) with Hits and Highlights",
        out_svg_desc, hit_color_map
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