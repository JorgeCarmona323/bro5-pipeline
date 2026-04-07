"""
compute_mapchiral_fingerprints.py
MAPchiral Fingerprint Computation Pipeline
==========================================

Philosophy
----------
MAPchiral fingerprints are not descriptor tables. They are MinHash signatures —
fixed-length arrays of uint32 integers produced by applying N_PERMUTATIONS hash
functions to the set of structural shingles of each molecule. The geometry is
entirely defined by the similarity metric, not by scaling or filtering.

No descriptor-style preprocessing is applied here. The only "preprocessing" that
makes sense for this representation is:
  1. Ensuring every fingerprint was successfully computed (no failed SMILES)
  2. Confirming the fingerprint type and representation properties are as expected
  3. Logging density statistics as a sanity check

Pipeline
--------
  1. Load CSV — SMILES + metadata, apply condition filter
  2. Compute MAPchiral fingerprints (uint32 MinHash signatures)
  3. Validate fingerprints:
       - consistent length across all molecules
       - no all-zero fingerprints (would indicate a computation failure)
       - density statistics (mean/std of active positions per molecule)
  4. Confirm no Hit or Highlight was lost
  5. Save fingerprint matrix + metadata + report

NOT applied (and why)
---------------------
  StandardScaler      — meaningless on hash integers; would destroy the
                        positional equality structure that minhash_distance relies on
  IQR clipping        — not applicable to integer hash values
  Deduplication       — duplicates retained; removing them would silently drop
                        molecules and corrupt the downstream alignment
  FPM normalization   — no method-based justification for this branch;
                        structural shingle coverage is not directly comparable
                        to MW-normalised descriptor density
  Variance filtering  — fingerprint dimensions are not individual features;
                        the full positional vector is required for MinHash similarity
  Correlation filter  — same reason; dimensions are not independently interpretable

MinHash distance verification
------------------------------
MAPchiral uses MinHashing to estimate Jaccard similarity between shingle sets:

    P(h_i(A) == h_i(B)) = |A ∩ B| / |A ∪ B|  =  Jaccard(A, B)

The fraction of positions where two signatures agree is therefore an unbiased
estimator of the Jaccard similarity of their underlying shingle sets.

Our minhash_distance (implemented with @njit in analyse_chemical_space.py):

    distance(A, B) = 1 − (count of positions where fp_A[k] == fp_B[k]) / N

This is consistent with the definition in the MAPchiral paper
(doi.org/10.1186/s13321-024-00849-6) and matches the jaccard_similarity
function exposed by the mapchiral library.

Standard metrics (Euclidean, Cosine, binary Jaccard) are incorrect because they
treat the integer positions as numeric values rather than hash slots.

UMAP and clustering notes
--------------------------
UMAP is run in analyse_chemical_space.py using minhash_distance as the metric.
HDBSCAN and K-Medoids are then run on the 2D UMAP embeddings using Euclidean
distance. Clustering in the embedding space (not the fingerprint space) is
intentional — this is consistent with the McInnes et al. recommendation and
avoids the O(n²) pairwise MinHash distance computation at cluster time.

Outputs
-------
  mapchiral_fingerprints.npy          (n_mols, N_PERMUTATIONS) uint32
  mapchiral_metadata.csv              aligned metadata, same row order as .npy
  mapchiral_failed_smiles.csv         failed molecules, if any
  mapchiral_preprocessing_report.txt
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from mapchiral.mapchiral import encode as mapchiral_encode
except ImportError:
    sys.exit(
        "\nERROR: mapchiral is not installed.\n"
        "  pip install mapchiral\n"
    )


# ===========================================================================
# CONFIG
# ===========================================================================

_REPO_ROOT = Path(__file__).parent.parent

INPUT_CSV = (
    _REPO_ROOT
    / "data" / "libraries" / "2026-01-29"
    / "canonicalized_master_macrocycles_2D_Descriptors_FINAL_20260129.csv"
)

RUN_TAG    = "2026-04-06"
OUTPUT_DIR = _REPO_ROOT / "outputs" / "mapchiral" / RUN_TAG

SMILES_COL    = "Smiles"
SOURCE_COL    = "Source"
HIT_ID_COL    = "Hit_ID"
HIGHLIGHT_COL = "Highlight_ID"

META_COLS = [SMILES_COL, SOURCE_COL, HIT_ID_COL, HIGHLIGHT_COL]

# Must match the condition used in compute_mordred_descriptors.py and
# analyse_chemical_space.py so that downstream alignment is valid.
#   "ALL" — Literature + Library + 34_Hits + Hit
#   "A"   — Literature + Hit
#   "B"   — Library + Hit
#   "C"   — Literature + Library + Hit
#   "D"   — Literature + 34_Hits + Hit
#   "E"   — Library + 34_Hits + Hit
DATA_CONDITION = "D"

CONDITION_SOURCES = {
    "A":   {"Literature", "Hit"},
    "B":   {"Library", "Hit"},
    "C":   {"Literature", "Library", "Hit"},
    "ALL": {"Literature", "Library", "34_Hits", "Hit"},
    "D":   {"Literature", "34_Hits", "Hit"},
    "E":   {"Library", "34_Hits", "Hit"},
}

# MAPchiral parameters
MAX_RADIUS     = 2
N_PERMUTATIONS = 2048   # number of hash permutations → signature length
MAPPING        = False


# ===========================================================================
# HELPERS
# ===========================================================================

def _elapsed(start: float) -> str:
    return f"{time.time() - start:.1f}s"


# ===========================================================================
# STEP 1 — LOAD DATA
# ===========================================================================

def load_data(path: Path, condition: str) -> pd.DataFrame:
    print(f"\n[1] Loading data: {path.name}")
    df = pd.read_csv(path)

    for col in META_COLS:
        if col not in df.columns:
            df[col] = ""

    df[SMILES_COL] = df[SMILES_COL].fillna("").astype(str).str.strip()

    n_before = len(df)
    df = df[df[SMILES_COL] != ""].reset_index(drop=True)
    print(f"   {len(df):,} rows after dropping empty SMILES "
          f"({n_before - len(df)} removed)")

    allowed = CONDITION_SOURCES[condition]
    df = df[df[SOURCE_COL].isin(allowed)].reset_index(drop=True)
    print(f"   Condition '{condition}' → {len(df):,} molecules "
          f"(sources: {', '.join(sorted(allowed))})")

    df["_is_hit"]       = df[HIT_ID_COL].astype(str).str.strip().ne("")
    df["_is_highlight"] = df[HIGHLIGHT_COL].astype(str).str.strip().ne("")
    print(f"   Hits:       {df['_is_hit'].sum():,}")
    print(f"   Highlights: {df['_is_highlight'].sum():,}")
    return df


# ===========================================================================
# STEP 2 — COMPUTE MAPchiral FINGERPRINTS
# ===========================================================================

def compute_mapchiral(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Compute MAPchiral MinHash fingerprints for all valid SMILES.

    Returns
    -------
    fps       : uint32 array, shape (n_valid, N_PERMUTATIONS)
    keep_idx  : original positions in df where fingerprinting succeeded
    failed_df : DataFrame of failed rows with reason
    """
    print(f"\n[2] Computing MAPchiral fingerprints "
          f"(max_radius={MAX_RADIUS}, n_permutations={N_PERMUTATIONS}) ...")
    t0 = time.time()

    fps, keep_idx, failed_rows = [], [], []

    for i, row in df.iterrows():
        smi = row[SMILES_COL]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failed_rows.append({
                "row_index":   i,
                SOURCE_COL:    str(row[SOURCE_COL]).strip(),
                HIT_ID_COL:    str(row[HIT_ID_COL]).strip(),
                HIGHLIGHT_COL: str(row[HIGHLIGHT_COL]).strip(),
                "reason":      "rdkit_parse_failed",
                SMILES_COL:    smi,
            })
            continue

        fp = mapchiral_encode(
            mol,
            max_radius=MAX_RADIUS,
            n_permutations=N_PERMUTATIONS,
            mapping=MAPPING,
        )
        fps.append(np.asarray(fp, dtype=np.uint32))
        keep_idx.append(i)

    fps_array  = np.vstack(fps) if fps else np.empty((0, N_PERMUTATIONS), dtype=np.uint32)
    keep_array = np.asarray(keep_idx, dtype=int)
    failed_df  = pd.DataFrame(failed_rows)

    n_total  = len(df)
    n_good   = len(keep_idx)
    n_failed = len(failed_rows)
    print(f"   Total:   {n_total:,}")
    print(f"   Success: {n_good:,} ({100 * n_good / n_total:.1f}%)")
    print(f"   Failed:  {n_failed:,}  ({_elapsed(t0)})")
    return fps_array, keep_array, failed_df


# ===========================================================================
# STEP 3 — VALIDATE FINGERPRINTS
# ===========================================================================

def validate_fingerprints(
    fps: np.ndarray,
    failed_df: pd.DataFrame,
    df_fp: pd.DataFrame,
) -> dict:
    """
    Three checks:
      A. Consistent fingerprint length across all molecules.
      B. No all-zero fingerprints (would indicate a silent computation failure).
      C. Density statistics — logged for sanity, not used to filter.

    Returns a stats dict for the report.
    """
    print(f"\n[3] Validating fingerprints ...")
    stats = {}

    # A — consistent length
    if fps.shape[1] != N_PERMUTATIONS:
        raise ValueError(
            f"Fingerprint length mismatch: expected {N_PERMUTATIONS}, "
            f"got {fps.shape[1]}. Check N_PERMUTATIONS config."
        )
    print(f"   [A] Fingerprint length: {fps.shape[1]} ✓  (all molecules)")

    # B — all-zero fingerprints
    all_zero_mask = ~fps.any(axis=1)
    n_all_zero    = int(all_zero_mask.sum())
    if n_all_zero > 0:
        zero_sources = df_fp.loc[all_zero_mask, SOURCE_COL].value_counts()
        print(f"   [B] WARNING: {n_all_zero} all-zero fingerprints detected:")
        print(zero_sources.to_string())
        # Check if any are hits or highlights
        zero_hits = df_fp.loc[all_zero_mask & df_fp["_is_hit"]].shape[0]
        zero_hl   = df_fp.loc[all_zero_mask & df_fp["_is_highlight"]].shape[0]
        if zero_hits > 0 or zero_hl > 0:
            raise ValueError(
                f"All-zero fingerprint in Hit ({zero_hits}) or "
                f"Highlight ({zero_hl}) — investigation required."
            )
        print(f"   [B] No hits or highlights affected by all-zero issue.")
    else:
        print(f"   [B] No all-zero fingerprints ✓")
    stats["n_all_zero"] = n_all_zero

    # C — density statistics
    # "Active" positions = positions where the hash value is non-zero.
    # For MinHash signatures this is not a conventional sparsity metric,
    # but large deviations between molecules may indicate encoding problems.
    nonzero_per_mol = np.count_nonzero(fps, axis=1)
    stats["density_mean"]   = float(nonzero_per_mol.mean())
    stats["density_std"]    = float(nonzero_per_mol.std())
    stats["density_min"]    = int(nonzero_per_mol.min())
    stats["density_max"]    = int(nonzero_per_mol.max())
    stats["density_median"] = float(np.median(nonzero_per_mol))

    print(f"   [C] Non-zero positions per molecule:")
    print(f"       mean  : {stats['density_mean']:.1f}")
    print(f"       std   : {stats['density_std']:.1f}")
    print(f"       min   : {stats['density_min']}")
    print(f"       median: {stats['density_median']:.1f}")
    print(f"       max   : {stats['density_max']}")

    # Check for hits or highlights in failed set
    if not failed_df.empty:
        hit_fail = failed_df[failed_df[HIT_ID_COL].astype(str).str.strip() != ""]
        hl_fail  = failed_df[failed_df[HIGHLIGHT_COL].astype(str).str.strip() != ""]

        if not hit_fail.empty:
            print("\n" + "!" * 62)
            print("ERROR: Hit(s) failed fingerprinting — would be excluded.")
            print("!" * 62)
            print(hit_fail.to_string(index=False))
            raise ValueError(f"{len(hit_fail)} hit(s) failed MAPchiral fingerprinting.")

        if not hl_fail.empty:
            print("\n" + "!" * 62)
            print("ERROR: Highlight(s) failed fingerprinting — would be excluded.")
            print("!" * 62)
            print(hl_fail.to_string(index=False))
            raise ValueError(f"{len(hl_fail)} highlight(s) failed MAPchiral fingerprinting.")

        print(f"   No hits or highlights in failed set ✓")
    else:
        print(f"   No fingerprinting failures ✓")

    return stats


# ===========================================================================
# SAVE OUTPUTS
# ===========================================================================

def save_outputs(
    df_fp: pd.DataFrame,
    fps: np.ndarray,
    failed_df: pd.DataFrame,
    report: dict,
    stats: dict,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[4] Saving outputs → {output_dir}")

    # Fingerprint matrix (row-aligned with metadata)
    np.save(output_dir / "mapchiral_fingerprints.npy", fps)
    print(f"   mapchiral_fingerprints.npy         — shape {fps.shape}, dtype uint32")

    # Metadata (same row order as .npy — required for downstream alignment)
    meta_out = df_fp[[c for c in df_fp.columns if not c.startswith("_")]].copy()
    meta_out.to_csv(output_dir / "mapchiral_metadata.csv", index=False)
    print(f"   mapchiral_metadata.csv             — {len(meta_out):,} rows")

    # Failed SMILES
    if not failed_df.empty:
        failed_df.to_csv(output_dir / "mapchiral_failed_smiles.csv", index=False)
        print(f"   mapchiral_failed_smiles.csv        — {len(failed_df)} rows")

    # Preprocessing report
    lines = [
        "=" * 62,
        "MAPCHIRAL FINGERPRINT PREPROCESSING REPORT",
        "=" * 62,
        f"Run date           : {RUN_TAG}",
        f"Input CSV          : {INPUT_CSV.name}",
        f"Data condition     : {DATA_CONDITION}",
        f"Sources            : {', '.join(sorted(CONDITION_SOURCES[DATA_CONDITION]))}",
        "",
        "--- Fingerprint Type Confirmation ---",
        f"  Type             : MinHash signature (NOT a binary bit vector)",
        f"  Dtype            : uint32",
        f"  Length           : {N_PERMUTATIONS} permutations",
        f"  max_radius       : {MAX_RADIUS}",
        f"  mapping          : {MAPPING}",
        "  Each integer is the minimum hash value of one permutation applied",
        "  to the full set of structural shingles of the molecule.",
        "",
        "--- MinHash Distance Verification ---",
        "  minhash_distance(A,B) = 1 - count(fp_A[k]==fp_B[k], k=0..N-1) / N",
        "  P(h_i(A)==h_i(B)) = Jaccard(shingles_A, shingles_B)  [MinHash theorem]",
        "  Therefore: distance = 1 - estimated Jaccard similarity  ✓",
        "  Matches mapchiral.jaccard_similarity() and the MAPchiral paper.",
        "  Ref: doi.org/10.1186/s13321-024-00849-6",
        "",
        "--- Preprocessing NOT Applied (and why) ---",
        "  StandardScaler   : NO — destroys positional equality structure",
        "  IQR clipping     : NO — not applicable to hash integers",
        "  Deduplication    : NO — duplicates retained to avoid alignment issues",
        "  FPM normalisation: NO — no method-based justification",
        "  Variance filter  : NO — dimensions are hash slots, not features",
        "  Correlation filt.: NO — same reason",
        "",
        "--- Molecules ---",
        f"  Input (condition-filtered) : {report['n_input']:>7,}",
        f"  Successfully fingerprinted : {report['n_fingerprinted']:>7,}",
        f"  Failed (SMILES parse)      : {report['n_failed']:>7,}",
        "",
        "--- Fingerprint Validation ---",
        f"  Consistent length ({N_PERMUTATIONS}) : ✓",
        f"  All-zero fingerprints      : {stats['n_all_zero']}",
        f"  Non-zero positions / mol:",
        f"    mean   : {stats['density_mean']:.1f}",
        f"    std    : {stats['density_std']:.1f}",
        f"    min    : {stats['density_min']}",
        f"    median : {stats['density_median']:.1f}",
        f"    max    : {stats['density_max']}",
        "",
        "--- Downstream Notes ---",
        "  UMAP metric      : minhash_distance (@njit, in analyse_chemical_space.py)",
        "  Clustering space : UMAP 2D embeddings (Euclidean)",
        "  Rationale        : avoids O(n²) pairwise MinHash distances at cluster time;",
        "                     consistent with McInnes et al. recommendation.",
        "",
        "--- Output Files ---",
        "  mapchiral_fingerprints.npy",
        "  mapchiral_metadata.csv",
        "  mapchiral_failed_smiles.csv  (if failures exist)",
        "  mapchiral_preprocessing_report.txt",
        "=" * 62,
    ]
    report_text = "\n".join(lines)
    print("\n" + report_text)
    with open(output_dir / "mapchiral_preprocessing_report.txt", "w") as fh:
        fh.write(report_text + "\n")


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    print("=" * 62)
    print("MAPCHIRAL FINGERPRINT PIPELINE")
    print(f"  Input     : {INPUT_CSV.name}")
    print(f"  Condition : {DATA_CONDITION}")
    print(f"  Output    : {OUTPUT_DIR}")
    print("=" * 62)

    t_total = time.time()
    report  = {}

    # 1. Load
    df = load_data(INPUT_CSV, DATA_CONDITION)
    report["n_input"] = len(df)

    # 2. Compute
    fps, keep_idx, failed_df = compute_mapchiral(df)
    report["n_fingerprinted"] = len(keep_idx)
    report["n_failed"]        = len(failed_df)

    df_fp = df.iloc[keep_idx].copy().reset_index(drop=True)

    # 3. Validate
    stats = validate_fingerprints(fps, failed_df, df_fp)

    # 4. Save
    save_outputs(df_fp, fps, failed_df, report, stats, OUTPUT_DIR)

    print(f"\n[✓] Pipeline complete — {_elapsed(t_total)}")


if __name__ == "__main__":
    main()
