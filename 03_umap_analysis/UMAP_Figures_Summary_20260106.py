import os
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

import umap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib as mpl

from numba import njit
from mapchiral.mapchiral import encode  # Reymond-group mapchiral


# ==========================================================
# USER CONFIG
# ==========================================================
INPUT_CSV = "C:\\Users\\Admin\\Documents\\Hu Lab\\Code\\Python\\rdkit\\Data\\2026-01-06\\canonicalized_master_macrocycles_2D_Descriptors_20260106.csv"
OUTPUT_DIR = "C:\\Users\\Admin\\Documents\\Hu Lab\\Code\\Python\\rdkit\\Data\\2026-01-07"
FIG_DIR = "C:\\Users\\Admin\\Documents\\Hu Lab\\Code\\Python\\rdkit\\Data\\2026-01-07\\Figures"

# Column headers (your file)
SMILES_COL = "Smiles"
SOURCE_COL = "Source"
HIT_ID_COL = "Hit_ID"

# 6 uniform descriptors (your file)
DESC_COLS = [
    "Total Molweight",
    "cLogP",
    "H-Acceptors",
    "H-Donors",
    "Polar Surface Area",
    "Rotatable Bonds",
]

# MAPchiral settings (MAP4C-style when max_radius=2)
MAX_RADIUS = 2
N_PERMUTATIONS = 2048
MAPPING = False

# UMAP settings
RANDOM_STATE = 42
UMAP_FP_PARAMS = dict(
    n_neighbors=20,
    min_dist=0.10,
    metric="precomputed",
    random_state=RANDOM_STATE,
)
UMAP_DESC_PARAMS = dict(
    n_neighbors=40,
    min_dist=0.30,
    metric="euclidean",
    random_state=RANDOM_STATE,
)

TOPK_NEIGHBORS = 30


# ==========================================================
# UTILS
# ==========================================================
def require_cols(df: pd.DataFrame, cols: list[str], where: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {where}: {missing}")

def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

def norm_str(x) -> str:
    return "" if pd.isna(x) else str(x).strip()

def source_norm_series(df: pd.DataFrame) -> pd.Series:
    return df[SOURCE_COL].astype(str).str.strip().str.lower()

def build_hit_color_map(hit_ids: list[str]):
    """Stable mapping Hit_ID -> RGBA color"""
    unique = sorted({h for h in hit_ids if h})
    # tab10 supports up to 10 distinct colors; tab20 up to 20
    cmap = mpl.colormaps.get_cmap("tab10" if len(unique) <= 10 else "tab20")
    return {hid: cmap(i % cmap.N) for i, hid in enumerate(unique)}


# ==========================================================
# MAPCHIRAL FINGERPRINTING + REPORTING
# ==========================================================
def compute_mapchiral_fps_with_report(df: pd.DataFrame):
    """
    Compute MAPchiral MinHash fingerprints.
    Returns: fps (n_good, N_PERMUTATIONS), keep_idx (original row indices retained)
    Writes failure log CSV and raises if any hit fails.
    """
    fps = []
    keep_idx = []

    bad_smiles = 0
    bad_parse = 0
    failed_rows = []  # (row_index, Source, Hit_ID, reason, Smiles)

    for i, row in df.iterrows():
        smi = row[SMILES_COL]
        src = norm_str(row[SOURCE_COL])
        hid = norm_str(row[HIT_ID_COL])

        if not isinstance(smi, str) or not smi.strip():
            bad_smiles += 1
            failed_rows.append((i, src, hid, "empty/invalid_smiles", smi))
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            bad_parse += 1
            failed_rows.append((i, src, hid, "rdkit_parse_failed", smi))
            continue

        fp = encode(mol, max_radius=MAX_RADIUS, n_permutations=N_PERMUTATIONS, mapping=MAPPING)
        fps.append(np.asarray(fp, dtype=np.uint32))
        keep_idx.append(i)

    total = len(df)
    good = len(keep_idx)
    failed = bad_smiles + bad_parse

    print("\n" + "=" * 70)
    print("MAPchiral Fingerprinting Summary")
    print("=" * 70)
    print(f"Total compounds in input:        {total:,}")
    print(f"Successfully fingerprinted:      {good:,} ({100*good/total:.1f}%)")
    print(f"Failed to fingerprint:           {failed:,} ({100*failed/total:.1f}%)")
    print(f"  - Empty/invalid SMILES:        {bad_smiles:,}")
    print(f"  - RDKit parsing failed:        {bad_parse:,}")
    print("=" * 70)

    if failed_rows:
        fail_df = pd.DataFrame(failed_rows, columns=["row_index", SOURCE_COL, HIT_ID_COL, "reason", SMILES_COL])
        fail_path = os.path.join(OUTPUT_DIR, "mapchiral_fingerprint_failures.csv")
        fail_df.to_csv(fail_path, index=False)

        print("\nFailures by Source:")
        print(fail_df[SOURCE_COL].value_counts(dropna=False).to_string())

        print("\nFailures by reason:")
        print(fail_df["reason"].value_counts().to_string())

        # Critical: hit failures
        hit_fail_df = fail_df[fail_df[HIT_ID_COL].astype(str).str.strip() != ""].copy()
        if not hit_fail_df.empty:
            print("\n" + "!" * 70)
            print("ERROR: One or more HITS failed MAPchiral fingerprinting and would be excluded.")
            print("Fix these Hit_ID rows (SMILES/canonicalization) and re-run.")
            print("!" * 70)
            print(hit_fail_df.to_string(index=False))
            raise ValueError("Hit fingerprinting failure: at least one hit would be dropped.")

        print(f"\nSaved failure log: {fail_path}")
    else:
        print("\nNo failures detected 🎉")

    print("=" * 70 + "\n")
    return np.vstack(fps), np.asarray(keep_idx, dtype=int)


# ==========================================================
# MINHASH DISTANCE MATRIX FOR UMAP (precomputed)
# distance = 1 - fraction_equal
# ==========================================================
@njit
def _dist_row(fp_i, fps):
    n = fps.shape[0]
    d = np.empty(n, dtype=np.float32)
    m = fp_i.shape[0]
    for j in range(n):
        eq = 0
        for k in range(m):
            if fp_i[k] == fps[j, k]:
                eq += 1
        d[j] = 1.0 - (eq / m)
    return d

def compute_distance_matrix(fps: np.ndarray) -> np.ndarray:
    n = fps.shape[0]
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        D[i, :] = _dist_row(fps[i], fps)
    return D


# ==========================================================
# PLOTTING (hits colored by Hit_ID)
# ==========================================================
def plot_umap(df: pd.DataFrame, xcol: str, ycol: str, title: str, outpath: str, hit_color_map: dict):
    plt.figure(figsize=(10, 8))

    s_norm = source_norm_series(df)

    # Base layers
    lit = df[s_norm == "literature"]
    lib = df[s_norm == "library"]

    if not lit.empty:
        plt.scatter(lit[xcol], lit[ycol], s=10, alpha=0.35, c="tab:blue", label="Literature")
    if not lib.empty:
        plt.scatter(lib[xcol], lib[ycol], s=10, alpha=0.25, c="tab:gray", label="Library")

    # Hits: one color per Hit_ID
    hits = df[df["_is_hit"]]
    if not hits.empty:
        for hid, sub in hits.groupby(HIT_ID_COL):
            hid = norm_str(hid)
            if not hid:
                hid = "Hit"
            plt.scatter(
                sub[xcol], sub[ycol],
                s=180, alpha=0.95,
                c=[hit_color_map.get(hid, (1, 0, 0, 1))],
                marker="X",
                edgecolors="black",
                linewidths=1.0,
                label=hid
            )

    plt.title(title)
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


# ==========================================================
# HIT NEIGHBORS (UMAP SPACE)
# ==========================================================
def extract_hit_neighbors_umap(df: pd.DataFrame, xcol: str, ycol: str, topk: int) -> pd.DataFrame:
    hits = df[df["_is_hit"]].copy()
    if hits.empty:
        print("No hits detected. Skipping neighbor extraction.")
        return pd.DataFrame()

    nonhits = df[~df["_is_hit"]].copy()
    rows = []

    for _, h in hits.iterrows():
        hx, hy = h[xcol], h[ycol]
        if pd.isna(hx) or pd.isna(hy):
            continue

        d = np.sqrt((nonhits[xcol] - hx) ** 2 + (nonhits[ycol] - hy) ** 2)
        nn = nonhits.assign(dist_to_hit=d).nsmallest(topk, "dist_to_hit").copy()

        nn.insert(0, "hit_id", h[HIT_ID_COL])
        nn.insert(1, "hit_smiles", h[SMILES_COL])
        nn.insert(2, "hit_source", h[SOURCE_COL])
        rows.append(nn)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# ==========================================================
# HIT-CENTRIC DUAL-SPACE REPORT (SMILES-keyed overlap)
# ==========================================================
def generate_hit_centric_report(df_fp: pd.DataFrame, df_desc: pd.DataFrame, topk: int) -> pd.DataFrame:
    hits_fp = df_fp[df_fp["_is_hit"]].copy()
    if hits_fp.empty:
        print("No hits detected. Skipping hit-centric report generation.")
        return pd.DataFrame()

    reports = []
    desc_lookup = df_desc.set_index(SMILES_COL, drop=False)

    nonhits_fp = df_fp[~df_fp["_is_hit"]].copy()
    nonhits_desc = df_desc[~df_desc["_is_hit"]].copy()

    for _, hit_row in hits_fp.iterrows():
        hit_id = norm_str(hit_row[HIT_ID_COL]) or "Hit"
        hit_smiles = hit_row[SMILES_COL]
        hit_source = hit_row[SOURCE_COL]

        hit_fp_x = hit_row["UMAP_MAPCHIRAL_1"]
        hit_fp_y = hit_row["UMAP_MAPCHIRAL_2"]

        if hit_smiles in desc_lookup.index:
            hit_desc_x = desc_lookup.loc[hit_smiles, "UMAP_DESC_1"]
            hit_desc_y = desc_lookup.loc[hit_smiles, "UMAP_DESC_2"]
        else:
            hit_desc_x, hit_desc_y = np.nan, np.nan

        # FP neighbors
        dist_fp = np.sqrt(
            (nonhits_fp["UMAP_MAPCHIRAL_1"] - hit_fp_x) ** 2 +
            (nonhits_fp["UMAP_MAPCHIRAL_2"] - hit_fp_y) ** 2
        )
        neighbors_fp = nonhits_fp.assign(distance=dist_fp).nsmallest(topk, "distance").copy()
        neighbors_fp["neighbor_space"] = "fp"
        fp_set = set(neighbors_fp[SMILES_COL].astype(str))

        # Desc neighbors
        if np.isnan(hit_desc_x) or np.isnan(hit_desc_y):
            neighbors_desc = nonhits_desc.head(0).copy()
            neighbors_desc["distance"] = []
        else:
            dist_desc = np.sqrt(
                (nonhits_desc["UMAP_DESC_1"] - hit_desc_x) ** 2 +
                (nonhits_desc["UMAP_DESC_2"] - hit_desc_y) ** 2
            )
            neighbors_desc = nonhits_desc.assign(distance=dist_desc).nsmallest(topk, "distance").copy()
        neighbors_desc["neighbor_space"] = "desc"
        desc_set = set(neighbors_desc[SMILES_COL].astype(str))

        both_set = fp_set & desc_set

        combined = pd.concat([neighbors_fp, neighbors_desc], ignore_index=True)
        combined["neighbor_type"] = np.where(
            combined[SMILES_COL].astype(str).isin(both_set),
            "both_spaces",
            np.where(combined["neighbor_space"] == "fp", "fp_only", "desc_only")
        )

        # add descriptors for interpretation
        for col in DESC_COLS:
            combined[col] = combined[SMILES_COL].map(df_desc.set_index(SMILES_COL)[col])

        out = combined[[SMILES_COL, SOURCE_COL, "neighbor_space", "neighbor_type", "distance"] + DESC_COLS].copy()
        out.insert(0, "hit_id", hit_id)
        out.insert(1, "hit_smiles", hit_smiles)
        out.insert(2, "hit_source", hit_source)
        out.insert(3, "hit_fp_umap1", hit_fp_x)
        out.insert(4, "hit_fp_umap2", hit_fp_y)
        out.insert(5, "hit_desc_umap1", hit_desc_x)
        out.insert(6, "hit_desc_umap2", hit_desc_y)

        reports.append(out)

        n_fp_only = (out["neighbor_type"] == "fp_only").sum()
        n_desc_only = (out["neighbor_type"] == "desc_only").sum()
        n_both = (out["neighbor_type"] == "both_spaces").sum()
        print(f"\n{hit_id}: {n_fp_only} fp_only | {n_desc_only} desc_only | {n_both} both_spaces")

    return pd.concat(reports, ignore_index=True)


# ==========================================================
# MAIN
# ==========================================================
def main():
    ensure_dirs()

    # Auto-detect delimiter (csv/tsv). If it fails, set sep="\t" manually.
    df = pd.read_csv(INPUT_CSV, sep=None, engine="python")

    require_cols(df, [SMILES_COL, SOURCE_COL, HIT_ID_COL], "master file")
    require_cols(df, DESC_COLS, "master file")

    # Normalize string columns
    df[SOURCE_COL] = df[SOURCE_COL].apply(norm_str)
    df[HIT_ID_COL] = df[HIT_ID_COL].apply(norm_str)

    # Define hits robustly
    df["_is_hit"] = df[HIT_ID_COL] != ""

    # Build per-hit colors (stable across both plots)
    hit_color_map = build_hit_color_map(df.loc[df["_is_hit"], HIT_ID_COL].tolist())

    # --- Step 4: MAPchiral UMAP ---
    print("STEP 4: Fingerprint UMAP (MAPchiral)")
    fps, keep_idx = compute_mapchiral_fps_with_report(df)

    # Keep only rows successfully fingerprinted
    df_fp = df.loc[keep_idx].copy().reset_index(drop=True)

    # Safety: ensure all hits made it into df_fp
    hits_master = set(df.loc[df["_is_hit"], HIT_ID_COL])
    hits_fp = set(df_fp.loc[df_fp["_is_hit"], HIT_ID_COL])
    missing_hits = sorted(hits_master - hits_fp)
    if missing_hits:
        raise ValueError(f"ERROR: These hits are missing from fingerprint UMAP input: {missing_hits}")

    print("Computing MinHash distance matrix...")
    D = compute_distance_matrix(fps)

    reducer_fp = umap.UMAP(**UMAP_FP_PARAMS)
    emb_fp = reducer_fp.fit_transform(D)
    df_fp["UMAP_MAPCHIRAL_1"] = emb_fp[:, 0]
    df_fp["UMAP_MAPCHIRAL_2"] = emb_fp[:, 1]

    fp_csv = os.path.join(OUTPUT_DIR, "master_with_umap_mapchiral.csv")
    df_fp.to_csv(fp_csv, index=False)
    print(f"Saved: {fp_csv}")

    fp_fig = os.path.join(FIG_DIR, "umap_mapchiral_hits_colored.png")
    plot_umap(df_fp, "UMAP_MAPCHIRAL_1", "UMAP_MAPCHIRAL_2",
              "Structural Space (MAPchiral + UMAP)", fp_fig, hit_color_map)
    print(f"Saved: {fp_fig}")

    # Neighbors around each hit in FP UMAP space
    nn_fp = extract_hit_neighbors_umap(df_fp, "UMAP_MAPCHIRAL_1", "UMAP_MAPCHIRAL_2", TOPK_NEIGHBORS)
    if not nn_fp.empty:
        nn_fp_path = os.path.join(OUTPUT_DIR, "hit_neighbors_mapchiral_umap.csv")
        nn_fp.to_csv(nn_fp_path, index=False)
        print(f"Saved: {nn_fp_path}")

    # --- Step 5: Descriptor UMAP ---
    print("\nSTEP 5: Descriptor UMAP")
    df_desc = df.copy()

    X = df_desc[DESC_COLS].astype(float).values
    X = StandardScaler().fit_transform(X)

    reducer_desc = umap.UMAP(**UMAP_DESC_PARAMS)
    emb_desc = reducer_desc.fit_transform(X)
    df_desc["UMAP_DESC_1"] = emb_desc[:, 0]
    df_desc["UMAP_DESC_2"] = emb_desc[:, 1]

    desc_csv = os.path.join(OUTPUT_DIR, "master_with_umap_descriptors.csv")
    df_desc.to_csv(desc_csv, index=False)
    print(f"Saved: {desc_csv}")

    desc_fig = os.path.join(FIG_DIR, "umap_descriptors_hits_colored.png")
    plot_umap(df_desc, "UMAP_DESC_1", "UMAP_DESC_2",
              "Property Space (6 Descriptors + UMAP)", desc_fig, hit_color_map)
    print(f"Saved: {desc_fig}")

    # --- Step 6: Dual-space hit-centric report ---
    print("\nSTEP 6: Hit-centric dual-space report")
    report = generate_hit_centric_report(df_fp=df_fp, df_desc=df_desc, topk=TOPK_NEIGHBORS)
    if not report.empty:
        report_path = os.path.join(OUTPUT_DIR, "hit_centric_report_dual_space.csv")
        report.to_csv(report_path, index=False)
        print(f"Saved: {report_path}")

    print("\nDONE ✅ Figures in /figures and tables in /outputs")


if __name__ == "__main__":
    main()
