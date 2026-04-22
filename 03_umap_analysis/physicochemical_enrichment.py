"""
physicochemical_enrichment.py
==============================
Enrichment analysis for the 2D physicochemical branch.

Instead of UMAP (unreliable on 6 features), this script quantifies how
hits occupy the physicochemical property space of the full DEL library.

Outputs
-------
  figures/2d_enrichment_kde.svg       — 2×3 KDE panel per property
  figures/2d_enrichment_scatter.svg   — pairwise scatter grid (hits overlaid)
  figures/2d_enrichment_radar.svg     — radar chart: hit median vs library median
  2d_enrichment_stats.csv             — Cohen's d + Mann-Whitney U per property

Comparison groups
-----------------
  Library  : Source == "Literature"  (DEL background, n≈8468)
  Hits     : Source == "34_Hits" or "Hit"  (confirmed actives, n≈39)

Cohen's d interpretation: 0.2=small, 0.5=medium, 0.8=large
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu

ROOT         = Path(__file__).resolve().parent.parent
MASTER_CSV   = (
    ROOT / "data/libraries/2026-01-29"
    / "canonicalized_master_macrocycles_2D_Descriptors_FINAL_20260129.csv"
)
ALIGNED_CSV  = ROOT / "outputs/analysis/2026-04-06/aligned_metadata.csv"
FIG_DIR      = ROOT / "outputs/analysis/2026-04-21/figures"
STATS_OUT    = ROOT / "outputs/analysis/2026-04-21/2d_enrichment_stats.csv"

PROPERTIES = [
    "Total Molweight",
    "cLogP",
    "Polar Surface Area",
    "H-Acceptors",
    "H-Donors",
    "Rotatable Bonds",
]
PROP_LABELS = {
    "Total Molweight":  "Molecular Weight (Da)",
    "cLogP":            "cLogP",
    "Polar Surface Area": "TPSA (Å²)",
    "H-Acceptors":      "H-Bond Acceptors",
    "H-Donors":         "H-Bond Donors",
    "Rotatable Bonds":  "Rotatable Bonds",
}

# Colour scheme
C_LIB  = "#1F77B4"   # library — blue
C_HITS = "#E41A1C"   # hits — red


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    aligned = pd.read_csv(ALIGNED_CSV)[["Smiles", "Source"]]
    master  = pd.read_csv(MASTER_CSV)

    df = aligned.merge(master[["Smiles"] + PROPERTIES], on="Smiles", how="left")
    # master has 517 duplicate SMILES (enumerated stereoisomers sharing canonical form)
    # dedup to avoid inflating counts — keep first match per SMILES
    n_before = len(df)
    df = df.drop_duplicates(subset="Smiles", keep="first").reset_index(drop=True)
    if len(df) < n_before:
        print(f"  Dropped {n_before - len(df)} duplicate-SMILES rows after merge")

    missing = df[PROPERTIES].isna().sum().sum()
    if missing:
        print(f"  Warning: {missing} NaN values in properties — dropping rows")
        df = df.dropna(subset=PROPERTIES).reset_index(drop=True)

    lib  = df[df["Source"] == "Literature"][PROPERTIES].copy()
    hits = df[df["Source"].isin(["34_Hits", "Hit"])][PROPERTIES].copy()

    print(f"  Library : {len(lib):,} molecules")
    print(f"  Hits    : {len(hits):,} molecules")
    return lib, hits


# ── Statistics ────────────────────────────────────────────────────────────────

def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled_std = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    return (b.mean() - a.mean()) / pooled_std if pooled_std > 0 else 0.0


def compute_stats(lib: pd.DataFrame, hits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for prop in PROPERTIES:
        a = lib[prop].values
        b = hits[prop].values
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        n    = len(a) * len(b)
        rbc  = 1 - (2 * u) / n   # rank-biserial correlation
        cd   = cohen_d(a, b)
        rows.append({
            "property":        prop,
            "lib_median":      float(np.median(a)),
            "lib_iqr":         float(np.percentile(a, 75) - np.percentile(a, 25)),
            "hit_median":      float(np.median(b)),
            "hit_iqr":         float(np.percentile(b, 75) - np.percentile(b, 25)),
            "cohen_d":         round(cd, 3),
            "mwu_p":           float(p),
            "rank_biserial_r": round(rbc, 3),
        })
    return pd.DataFrame(rows)


# ── Figure 1: KDE panel ───────────────────────────────────────────────────────

def plot_kde_panel(lib: pd.DataFrame, hits: pd.DataFrame,
                   stats_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(
        "Physicochemical Enrichment — Hit vs Library (DEL Condition D)\n"
        "Shaded = library  |  Line = hits  |  Dashed = medians",
        fontsize=10, fontweight="bold"
    )

    for ax, prop in zip(axes.flat, PROPERTIES):
        a    = lib[prop].values
        b    = hits[prop].values
        row  = stats_df[stats_df["property"] == prop].iloc[0]

        # KDE — library
        xmin = min(a.min(), b.min())
        xmax = max(a.max(), b.max())
        pad  = (xmax - xmin) * 0.08
        xs   = np.linspace(xmin - pad, xmax + pad, 300)

        kde_lib  = stats.gaussian_kde(a, bw_method="scott")
        kde_hits = stats.gaussian_kde(b, bw_method="scott")

        ax.fill_between(xs, kde_lib(xs),  alpha=0.35, color=C_LIB,  label="Library")
        ax.plot(xs,        kde_lib(xs),   lw=1.2,     color=C_LIB)
        ax.plot(xs,        kde_hits(xs),  lw=2.0,     color=C_HITS, label="Hits")

        # Medians
        ax.axvline(row["lib_median"],  color=C_LIB,  lw=1.0, linestyle="--", alpha=0.7)
        ax.axvline(row["hit_median"],  color=C_HITS, lw=1.5, linestyle="--", alpha=0.9)

        p_label = (f"p<0.001" if row["mwu_p"] < 0.001
                   else f"p={row['mwu_p']:.3f}")
        d_label = f"d={row['cohen_d']:+.2f}"

        ax.set_xlabel(PROP_LABELS[prop], fontsize=9)
        ax.set_ylabel("Density", fontsize=8)
        ax.set_title(f"{PROP_LABELS[prop]}\n{p_label}  {d_label}", fontsize=8.5)
        ax.legend(fontsize=7, framealpha=0.7)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=7)

    fig.tight_layout()
    path = FIG_DIR / "2d_enrichment_kde.svg"
    fig.savefig(path, format="svg", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Figure 2: Pairwise scatter panel ─────────────────────────────────────────

PAIRS = [
    ("Total Molweight", "Polar Surface Area"),
    ("Total Molweight", "cLogP"),
    ("cLogP",           "H-Acceptors"),
    ("Polar Surface Area", "H-Donors"),
    ("Rotatable Bonds", "Total Molweight"),
    ("H-Donors",        "H-Acceptors"),
]


def plot_scatter_panel(lib: pd.DataFrame, hits: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(
        "Physicochemical Property Pairs — Hit vs Library",
        fontsize=10, fontweight="bold"
    )

    for ax, (px, py) in zip(axes.flat, PAIRS):
        ax.scatter(lib[px],  lib[py],  s=2,  c=C_LIB,  alpha=0.15,
                   linewidths=0, label="Library", rasterized=True)
        ax.scatter(hits[px], hits[py], s=40, c=C_HITS, alpha=0.85,
                   linewidths=0.6, edgecolors="black", zorder=5, label="Hits")
        ax.set_xlabel(PROP_LABELS[px], fontsize=8)
        ax.set_ylabel(PROP_LABELS[py], fontsize=8)
        ax.legend(fontsize=7, framealpha=0.7,
                  handles=[
                      mpatches.Patch(color=C_LIB,  label=f"Library (n={len(lib):,})"),
                      mpatches.Patch(color=C_HITS, label=f"Hits (n={len(hits)})"),
                  ])
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=7)

    fig.tight_layout()
    path = FIG_DIR / "2d_enrichment_scatter.svg"
    fig.savefig(path, format="svg", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Figure 3: Radar chart ─────────────────────────────────────────────────────

def plot_radar(lib: pd.DataFrame, hits: pd.DataFrame,
               stats_df: pd.DataFrame) -> None:
    """
    Normalised radar: each property scaled so the library median = 0.5.
    Shows where hit medians fall relative to the library centre.
    """
    labels = [PROP_LABELS[p] for p in PROPERTIES]
    N      = len(PROPERTIES)

    # Normalize: lib median → 0.5;  use IQR for scale
    lib_med  = stats_df["lib_median"].values
    lib_iqr  = stats_df["lib_iqr"].values.clip(min=1e-6)
    hit_med  = stats_df["hit_median"].values

    lib_norm  = np.full(N, 0.5)
    hit_norm  = 0.5 + (hit_med - lib_med) / (2 * lib_iqr)
    hit_norm  = np.clip(hit_norm, 0.0, 1.0)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    # close the polygon
    angles  += [angles[0]]
    lib_norm  = np.append(lib_norm,  lib_norm[0])
    hit_norm  = np.append(hit_norm,  hit_norm[0])

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["−IQR", "library\nmedian", "+IQR", "+2IQR"], fontsize=6)
    ax.set_ylim(0, 1)
    ax.set_title(
        "Hit physicochemical profile vs library\n"
        "(centre = library median; ±IQR scale)",
        fontsize=9, pad=20
    )

    ax.fill(angles, lib_norm, color=C_LIB,  alpha=0.25)
    ax.plot(angles, lib_norm, color=C_LIB,  lw=2, label="Library median")
    ax.fill(angles, hit_norm, color=C_HITS, alpha=0.30)
    ax.plot(angles, hit_norm, color=C_HITS, lw=2, label="Hit median")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.8)

    path = FIG_DIR / "2d_enrichment_radar.svg"
    fig.savefig(path, format="svg", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  PHYSICOCHEMICAL ENRICHMENT ANALYSIS")
    print("=" * 60)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    lib, hits = load_data()

    print("\n[Stats] Computing Cohen's d and Mann-Whitney U ...")
    stats_df = compute_stats(lib, hits)
    stats_df.to_csv(STATS_OUT, index=False)
    print(f"  Saved: {STATS_OUT.name}")

    print("\n  Property enrichment summary:")
    print(f"  {'Property':<24} {'Lib med':>8} {'Hit med':>8} "
          f"{'Cohen d':>8} {'MWU p':>10} {'|r|':>6} {'direction'}")
    print("  " + "-" * 80)
    for _, row in stats_df.iterrows():
        direction = "HIGHER" if row["hit_median"] > row["lib_median"] else "lower"
        print(
            f"  {row['property']:<24} "
            f"{row['lib_median']:>8.2f} "
            f"{row['hit_median']:>8.2f} "
            f"{row['cohen_d']:>+8.3f} "
            f"{row['mwu_p']:>10.4f} "
            f"{abs(row['rank_biserial_r']):>6.3f}  "
            f"{direction}"
        )

    print("\n[Figures]")
    plot_kde_panel(lib, hits, stats_df)
    plot_scatter_panel(lib, hits)
    plot_radar(lib, hits, stats_df)

    print("\n[✓] Done.")
    print(f"  Figures: {FIG_DIR}/2d_enrichment_*.svg")
    print(f"  Stats  : {STATS_OUT}")


if __name__ == "__main__":
    main()
