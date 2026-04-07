"""
compute_mordred_descriptors.py
Mordred 2D Descriptor Computation and Preprocessing Pipeline
=============================================================

Pipeline (in order):
  1.  Load CSV — SMILES + metadata
  2.  Compute Mordred 2D descriptors for all valid molecules
  3.  Drop descriptor columns with >NAN_COL_THRESHOLD missing values
        Rationale: high-NaN columns are likely structurally inapplicable
        to this scaffold class, not worth rescuing
  4.  Median-impute residual missing values
        Rationale: low-NaN columns are valid descriptors with occasional
        computation failures; median imputation is appropriate
  5.  Drop near-zero variance features
  6.  Drop highly correlated features (Spearman |r| >= CORR_THRESHOLD)
        Tie-breaking: lower NaN rate > higher variance > alphabetical
  7.  IQR-based outlier clipping per column [Q1 - k*IQR, Q3 + k*IQR]
  8.  StandardScaler (zero-mean, unit-variance)

  UMAP and clustering are handled separately in analyse_chemical_space.py.
  This script stops at the scaled feature matrix.

Outputs (all written to OUTPUT_DIR)
------------------------------------
  mordred_raw.csv                   full Mordred matrix, no filtering
  mordred_filtered_unscaled.csv     filtered + imputed, raw values (interpretable)
  mordred_filtered_scaled.csv       filtered + IQR-clipped + scaled (ready for downstream)
  mordred_retained_descriptors.txt  sorted list of kept descriptors
  mordred_dropped_descriptors.csv   dropped descriptors with reason
  mordred_preprocessing_report.txt  full preprocessing audit log

Installation
------------
  Python 3.12+: pip install mordredcommunity
  Python ≤3.10: pip install mordred
  (both packages expose the same Calculator / descriptors API)

  conda: rdkit umap-learn scikit-learn scipy pandas numpy matplotlib
"""

import re
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Mordred import — try original, fall back to community fork
# ---------------------------------------------------------------------------
try:
    from mordred import Calculator, descriptors as mordred_descriptors
    _MORDRED_PKG = "mordred"
except ImportError:
    try:
        from mordredcommunity import Calculator, descriptors as mordred_descriptors
        _MORDRED_PKG = "mordredcommunity"
    except ImportError:
        sys.exit(
            "\nERROR: Mordred is not installed.\n"
            "  Python 3.12+:  pip install mordredcommunity\n"
            "  Python ≤3.10:  pip install mordred\n"
        )


# ===========================================================================
# CONFIG — edit this block
# ===========================================================================

_REPO_ROOT = Path(__file__).parent.parent

INPUT_CSV = (
    _REPO_ROOT
    / "data" / "libraries" / "2026-01-29"
    / "canonicalized_master_macrocycles_2D_Descriptors_FINAL_20260129.csv"
)

RUN_TAG    = "2026-04-06"
OUTPUT_DIR = _REPO_ROOT / "outputs" / "mordred" / RUN_TAG

# Metadata columns to carry through (must exist in INPUT_CSV)
SMILES_COL = "Smiles"
META_COLS  = ["Smiles", "Source", "Hit_ID", "Highlight_ID"]

# Data condition filter (applied before descriptor computation)
#   "ALL"  — all sources
#   "A"    — Literature + Hit
#   "B"    — Library + Hit
#   "C"    — Literature + Library + Hit
#   "D"    — Literature + 34_Hits + Hit
#   "E"    — Library + 34_Hits + Hit
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
# Preprocessing thresholds
# ------------------------------------------------------------------
NAN_COL_THRESHOLD  = 0.10   # drop columns with > 10% NaN
CORR_THRESHOLD     = 0.90   # Spearman |r| cutoff for correlation filter
IQR_MULTIPLIER     = 1.5    # clip fence: [Q1 - k*IQR, Q3 + k*IQR]
VARIANCE_THRESHOLD = 1e-6   # near-zero variance cutoff (on raw values)

# ------------------------------------------------------------------
# UMAP parameters
# ------------------------------------------------------------------


# ===========================================================================
# HELPERS
# ===========================================================================

def require_cols(df: pd.DataFrame, cols: list, context: str = "") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{context} — missing columns: {missing}")


def descriptor_family(name: str) -> str:
    """Extract Mordred descriptor family prefix: 'ATSC0c' → 'ATSC'."""
    m = re.match(r"^([A-Za-z]+)", name)
    return m.group(1) if m else name


def _elapsed(start: float) -> str:
    return f"{time.time() - start:.1f}s"


# ===========================================================================
# STEP 1 — LOAD DATA
# ===========================================================================

def load_data(path: Path, condition: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load CSV, validate columns, apply condition filter.
    Returns (df_full_meta, df_filtered) where df_filtered has valid SMILES.
    """
    print(f"\n[1] Loading data: {path}")
    df = pd.read_csv(path)
    require_cols(df, [SMILES_COL], context="load_data")

    # Ensure metadata columns present
    for col in META_COLS:
        if col not in df.columns:
            df[col] = ""

    # Normalise SMILES
    df[SMILES_COL] = df[SMILES_COL].fillna("").astype(str).str.strip()

    # Drop empty SMILES
    n_before = len(df)
    df = df[df[SMILES_COL] != ""].reset_index(drop=True)
    print(f"   {len(df):,} rows after dropping empty SMILES "
          f"({n_before - len(df)} removed)")

    # Apply condition filter
    allowed = CONDITION_SOURCES[condition]
    df_cond = df[df["Source"].isin(allowed)].reset_index(drop=True)
    print(f"   Condition '{condition}' → {len(df_cond):,} molecules "
          f"(sources: {', '.join(sorted(allowed))})")

    return df_cond


# ===========================================================================
# STEP 2 — COMPUTE MORDRED DESCRIPTORS
# ===========================================================================

def compute_mordred(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[int]]:
    """
    Compute all Mordred 2D descriptors.

    Returns
    -------
    desc_df   : DataFrame, shape (n_valid, n_descriptors)
                index = original row positions in df
    valid_idx : list of row positions where SMILES parsed successfully
    """
    print(f"\n[2] Computing Mordred 2D descriptors "
          f"(package: {_MORDRED_PKG}) ...")
    t0 = time.time()

    calc = Calculator(mordred_descriptors, ignore_3D=True)

    mols, valid_idx, n_failed = [], [], 0
    for i, smi in enumerate(df[SMILES_COL]):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_failed += 1
        else:
            mols.append(mol)
            valid_idx.append(i)

    if n_failed:
        print(f"   WARNING: {n_failed} SMILES failed RDKit parsing — excluded")

    print(f"   Running Calculator on {len(mols):,} molecules ...")
    desc_df = calc.pandas(mols)

    # Convert Mordred Error objects and non-numeric entries to NaN
    desc_df = desc_df.apply(pd.to_numeric, errors="coerce")

    # Convert ±inf to NaN (some Mordred descriptors can produce these)
    desc_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    desc_df.index = valid_idx

    print(f"   {desc_df.shape[1]} descriptors × {desc_df.shape[0]:,} molecules "
          f"({_elapsed(t0)})")
    return desc_df, valid_idx


# ===========================================================================
# STEP 3 — DROP HIGH-NaN COLUMNS
# ===========================================================================

def drop_high_nan_cols(
    desc_df: pd.DataFrame,
    threshold: float,
    report: dict,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Drop columns where fraction missing > threshold.

    Also logs the top descriptor families that were dropped, so we can
    spot entire Mordred classes that are unstable on this scaffold.

    Returns (filtered_df, nan_rates_original, dropped_column_names)
    nan_rates_original is saved for the tie-breaking logic in Step 6.
    """
    nan_rates = desc_df.isna().mean()
    to_drop   = nan_rates[nan_rates > threshold].index.tolist()

    # Family breakdown of dropped columns
    families   = Counter(descriptor_family(c) for c in to_drop)
    top_fam    = families.most_common(10)

    df_out = desc_df.drop(columns=to_drop)

    report["n_high_nan_dropped"]   = len(to_drop)
    report["high_nan_top_families"] = top_fam

    print(f"\n[3] Drop high-NaN columns (>{threshold*100:.0f}%):")
    print(f"   Dropped  : {len(to_drop):>5,}  |  Remaining: {df_out.shape[1]:,}")
    if top_fam:
        fam_str = ", ".join(f"{f}({n})" for f, n in top_fam[:6])
        print(f"   Top families in dropped set → {fam_str}")

    return df_out, nan_rates, to_drop


# ===========================================================================
# STEP 4 — MEDIAN IMPUTE RESIDUAL NaN
# ===========================================================================

def median_impute(desc_df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """
    Replace any remaining NaN with per-column median.
    After Step 3, these are rare computation failures, not structural absence.
    """
    n_nan  = int(desc_df.isna().sum().sum())
    n_cols = int((desc_df.isna().sum() > 0).sum())

    medians = desc_df.median()
    df_out  = desc_df.fillna(medians)

    report["n_values_imputed"] = n_nan
    report["n_cols_imputed"]   = n_cols

    print(f"\n[4] Median imputation:")
    print(f"   Imputed {n_nan:,} values across {n_cols} columns")
    return df_out


# ===========================================================================
# STEP 5 — NEAR-ZERO VARIANCE FILTER
# ===========================================================================

def drop_near_zero_variance(
    desc_df: pd.DataFrame,
    threshold: float,
    report: dict,
) -> tuple[pd.DataFrame, list[str]]:
    variances = desc_df.var()
    to_drop   = variances[variances < threshold].index.tolist()
    df_out    = desc_df.drop(columns=to_drop)

    report["n_low_variance_dropped"] = len(to_drop)

    print(f"\n[5] Near-zero variance filter (var < {threshold}):")
    print(f"   Dropped  : {len(to_drop):>5,}  |  Remaining: {df_out.shape[1]:,}")
    return df_out, to_drop


# ===========================================================================
# STEP 6 — SPEARMAN CORRELATION FILTER
# ===========================================================================

def drop_correlated_spearman(
    desc_df: pd.DataFrame,
    nan_rates_orig: pd.Series,
    threshold: float,
    report: dict,
) -> tuple[pd.DataFrame, list[str], dict]:
    """
    Greedy Spearman correlation filter.

    For each correlated pair (|r| >= threshold), the descriptor to DROP is
    chosen by priority:
      1. Higher original NaN rate  (keep the one with fewer failures)
      2. Lower variance            (keep the more informative one)
      3. Alphabetical tie-break    (deterministic)

    Pairs are processed in descending |r| order so the strongest redundancies
    are resolved first.

    Returns (filtered_df, dropped_col_names, drop_reason_dict)
    """
    print(f"\n[6] Spearman correlation filter (|r| >= {threshold}):")
    t0 = time.time()

    cols = list(desc_df.columns)
    n    = len(cols)

    # Compute Spearman correlation via rank transformation
    # Using rankdata + corrcoef is faster than pandas .corr('spearman')
    # for large feature sets
    print(f"   Ranking {n} descriptors × {len(desc_df):,} molecules ...")
    X     = desc_df.values.astype(np.float64)
    ranks = np.apply_along_axis(rankdata, 0, X)        # (n_mols, n_feats)
    corr  = np.abs(np.corrcoef(ranks.T))               # (n_feats, n_feats)
    np.fill_diagonal(corr, 0.0)                        # ignore self-correlation
    print(f"   Correlation matrix done ({_elapsed(t0)})")

    # Collect all pairs above threshold (upper triangle)
    row_idx, col_idx = np.where(
        np.triu(corr >= threshold, k=1)
    )
    pairs = sorted(
        zip(corr[row_idx, col_idx], row_idx, col_idx),
        reverse=True,
    )
    print(f"   {len(pairs):,} correlated pairs found above threshold")

    variances   = desc_df.var()
    dropped     = set()
    drop_reasons = {}

    for r_val, i, j in pairs:
        col_a, col_b = cols[i], cols[j]
        if col_a in dropped or col_b in dropped:
            continue

        nan_a = float(nan_rates_orig.get(col_a, 0.0))
        nan_b = float(nan_rates_orig.get(col_b, 0.0))
        var_a = float(variances.get(col_a, 0.0))
        var_b = float(variances.get(col_b, 0.0))

        if nan_a != nan_b:
            drop   = col_a if nan_a > nan_b else col_b
            reason = "higher_nan_rate"
        elif var_a != var_b:
            drop   = col_a if var_a < var_b else col_b
            reason = "lower_variance"
        else:
            drop   = max(col_a, col_b)   # alphabetical, deterministic
            reason = "alphabetical_tiebreak"

        keep   = col_b if drop == col_a else col_a
        dropped.add(drop)
        drop_reasons[drop] = (reason, f"|r|={r_val:.3f} with {keep}")

    df_out = desc_df.drop(columns=list(dropped))
    report["n_correlated_dropped"] = len(dropped)
    report["corr_drop_details"]    = drop_reasons

    print(f"   Dropped  : {len(dropped):>5,}  |  Remaining: {df_out.shape[1]:,} "
          f"({_elapsed(t0)})")
    return df_out, list(dropped), drop_reasons


# ===========================================================================
# STEP 7 — IQR CLIPPING
# ===========================================================================

def iqr_clip(desc_df: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    """Clip each column to [Q1 - k*IQR, Q3 + k*IQR]."""
    desc_df = desc_df.astype(float)   # guard: Mordred emits boolean cols for some descriptors
    q1    = desc_df.quantile(0.25)
    q3    = desc_df.quantile(0.75)
    iqr   = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    df_out = desc_df.clip(lower=lower, upper=upper, axis=1)
    print(f"\n[7] IQR clipping (multiplier={multiplier}) applied")
    return df_out


# ===========================================================================
# STEP 8 — STANDARD SCALER
# ===========================================================================

def standard_scale(
    desc_df: pd.DataFrame,
) -> tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(desc_df.values)
    df_out = pd.DataFrame(scaled, index=desc_df.index, columns=desc_df.columns)
    print(f"\n[8] StandardScaler applied — final matrix: "
          f"{df_out.shape[0]:,} molecules × {df_out.shape[1]} features")
    return df_out, scaler


# ===========================================================================
# VALIDATION ASSERTIONS
# ===========================================================================

def validate_final_matrix(df: pd.DataFrame) -> None:
    assert df.shape[1] >= 1, \
        "Final descriptor matrix has 0 features — check preprocessing thresholds."
    assert not df.isnull().any().any(), \
        "Final descriptor matrix contains NaN — imputation may have failed."
    assert not np.isinf(df.values).any(), \
        "Final descriptor matrix contains inf — check IQR clipping."
    print(f"\n[✓] Validation passed: no NaN, no inf, {df.shape[1]} features retained")


# ===========================================================================
# SAVE OUTPUTS
# ===========================================================================

def save_outputs(
    df_meta: pd.DataFrame,
    desc_raw: pd.DataFrame,
    desc_filtered_unscaled: pd.DataFrame,
    desc_scaled: pd.DataFrame,
    valid_idx: list[int],
    nan_rates_orig: pd.Series,
    dropped_high_nan: list[str],
    dropped_low_var: list[str],
    dropped_corr: list[str],
    corr_drop_reasons: dict,
    report: dict,
    output_dir: Path,
) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)
    meta_valid = df_meta.iloc[valid_idx].reset_index(drop=True)

    # Raw descriptors -------------------------------------------------------
    raw_path = output_dir / "mordred_raw.csv"
    df_meta.iloc[valid_idx].reset_index(drop=True).join(
        desc_raw.reset_index(drop=True)
    ).to_csv(raw_path, index=False)
    print(f"\n   mordred_raw.csv                — {desc_raw.shape[1]} descriptors")

    # Filtered unscaled (human-interpretable) --------------------------------
    unscaled_path = output_dir / "mordred_filtered_unscaled.csv"
    meta_valid.join(desc_filtered_unscaled.reset_index(drop=True)).to_csv(
        unscaled_path, index=False
    )
    print(f"   mordred_filtered_unscaled.csv  — {desc_filtered_unscaled.shape[1]} descriptors")

    # Filtered + IQR-clipped + scaled (ready for downstream clustering) -----
    scaled_path = output_dir / "mordred_filtered_scaled.csv"
    meta_valid.join(desc_scaled.reset_index(drop=True)).to_csv(
        scaled_path, index=False
    )
    print(f"   mordred_filtered_scaled.csv    — {desc_scaled.shape[1]} descriptors")

    # Retained descriptor list -----------------------------------------------
    retained = sorted(desc_scaled.columns.tolist())
    with open(output_dir / "mordred_retained_descriptors.txt", "w") as fh:
        fh.write(f"# {len(retained)} retained descriptors\n")
        fh.write("\n".join(retained) + "\n")
    print(f"   mordred_retained_descriptors.txt")

    # Dropped descriptors with reason ----------------------------------------
    rows = []
    for c in dropped_high_nan:
        rows.append({
            "descriptor": c,
            "reason":     "high_nan",
            "detail":     f"nan_rate={nan_rates_orig.get(c, 0.0):.3f}",
        })
    for c in dropped_low_var:
        rows.append({
            "descriptor": c,
            "reason":     "near_zero_variance",
            "detail":     "",
        })
    for c in dropped_corr:
        r, detail = corr_drop_reasons.get(c, ("", ""))
        rows.append({
            "descriptor": c,
            "reason":     f"high_correlation ({r})",
            "detail":     detail,
        })
    pd.DataFrame(rows).to_csv(output_dir / "mordred_dropped_descriptors.csv", index=False)
    print(f"   mordred_dropped_descriptors.csv")

    # Preprocessing report ---------------------------------------------------
    retained_count = len(desc_scaled.columns)
    lines = [
        "=" * 62,
        "MORDRED DESCRIPTOR PREPROCESSING REPORT",
        "=" * 62,
        f"Run date         : {RUN_TAG}",
        f"Input CSV        : {INPUT_CSV.name}",
        f"Mordred package  : {_MORDRED_PKG}",
        f"Data condition   : {DATA_CONDITION}",
        "",
        "--- Molecules ---",
        f"Input rows                 : {len(df_meta):>7,}",
        f"Valid SMILES               : {len(valid_idx):>7,}",
        f"Failed SMILES              : {len(df_meta) - len(valid_idx):>7,}",
        "",
        "--- Descriptor Filtering ---",
        f"Original descriptors       : {report['n_original']:>7,}",
        f"Dropped >10% NaN           : {report['n_high_nan_dropped']:>7,}",
        f"After NaN filter           : {report['n_original'] - report['n_high_nan_dropped']:>7,}",
        f"Values median-imputed      : {report['n_values_imputed']:>7,}  "
        f"({report['n_cols_imputed']} columns)",
        f"Dropped near-zero variance : {report['n_low_variance_dropped']:>7,}",
        f"Dropped high correlation   : {report['n_correlated_dropped']:>7,}",
        f"Final descriptor count     : {retained_count:>7,}",
        "",
        "--- Preprocessing ---",
        f"IQR multiplier             : {IQR_MULTIPLIER}",
        f"Scaler                     : StandardScaler",
        f"Correlation threshold      : |r| >= {CORR_THRESHOLD} (Spearman)",
        f"Tie-breaking               : lower NaN rate > higher variance > alphabetical",
        "",
    ]

    if report.get("high_nan_top_families"):
        lines.append("Top Mordred families in high-NaN dropped set:")
        for fam, cnt in report["high_nan_top_families"]:
            lines.append(f"  {fam:<22}: {cnt}")
        lines.append("")

    # Feature count warnings
    if retained_count < 50:
        lines.append(
            "WARNING: Final feature count is unusually LOW (<50).\n"
            "         Consider relaxing NAN_COL_THRESHOLD or CORR_THRESHOLD."
        )
    if retained_count > 800:
        lines.append(
            "WARNING: Final feature count is unusually HIGH (>800).\n"
            "         Consider tightening CORR_THRESHOLD."
        )

    lines += [
        "--- Downstream ---",
        "  UMAP metric    : cosine (applied in analyse_chemical_space.py)",
        "  Clustering     : HDBSCAN + K-Medoids on UMAP embeddings",
        "",
        "--- Output files ---",
        "  mordred_raw.csv",
        "  mordred_filtered_unscaled.csv",
        "  mordred_filtered_scaled.csv",
        "  mordred_retained_descriptors.txt",
        "  mordred_dropped_descriptors.csv",
        "  mordred_preprocessing_report.txt",
        "=" * 62,
    ]

    report_text = "\n".join(lines)
    print("\n" + report_text)
    with open(output_dir / "mordred_preprocessing_report.txt", "w") as fh:
        fh.write(report_text + "\n")


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    print("=" * 62)
    print("MORDRED DESCRIPTOR COMPUTATION PIPELINE")
    print(f"  Input  : {INPUT_CSV.name}")
    print(f"  Output : {OUTPUT_DIR}")
    print(f"  Package: {_MORDRED_PKG}")
    print("=" * 62)

    t_total = time.time()
    report  = {}

    # 1. Load
    df = load_data(INPUT_CSV, DATA_CONDITION)
    df_meta = df[[c for c in META_COLS if c in df.columns]].copy()

    # 2. Compute Mordred
    desc_raw, valid_idx = compute_mordred(df)
    report["n_original"] = desc_raw.shape[1]

    # Save original NaN rates before any filtering (for tie-breaking in Step 6)
    nan_rates_orig = desc_raw.isna().mean()

    # 3. Drop high-NaN columns
    desc, _, dropped_high_nan = drop_high_nan_cols(
        desc_raw, NAN_COL_THRESHOLD, report
    )

    # 4. Median impute residual NaN
    desc = median_impute(desc, report)

    # 5. Near-zero variance
    desc, dropped_low_var = drop_near_zero_variance(desc, VARIANCE_THRESHOLD, report)

    # 6. Spearman correlation filter
    desc, dropped_corr, corr_drop_reasons = drop_correlated_spearman(
        desc, nan_rates_orig, CORR_THRESHOLD, report
    )

    # Snapshot before IQR/scaling — saved as unscaled output for inspection
    desc_filtered_unscaled = desc.copy()

    # 7. IQR clip
    desc = iqr_clip(desc, IQR_MULTIPLIER)

    # 8. Scale
    desc_scaled, _ = standard_scale(desc)

    # Validate
    validate_final_matrix(desc_scaled)

    # Save outputs (UMAP runs in analyse_chemical_space.py)
    print("\n[9] Saving outputs ...")
    save_outputs(
        df_meta, desc_raw, desc_filtered_unscaled, desc_scaled,
        valid_idx, nan_rates_orig,
        dropped_high_nan, dropped_low_var, dropped_corr, corr_drop_reasons,
        report, OUTPUT_DIR,
    )

    print(f"\n[✓] Pipeline complete — total time: {_elapsed(t_total)}")


if __name__ == "__main__":
    main()
