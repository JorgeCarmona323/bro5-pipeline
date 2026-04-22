# Combined UMAP Figures — Cluster Coloring + Hit/Reference Overlay
**Date:** 2026-04-21 | **Status:** Complete | **Script:** `03_umap_analysis/make_combined_figures.py`
**Related:** `docs/2026-04-16_condition_D_chemical_space_findings.md`

---

## What These Figures Show

Each figure layers two pieces of information in one plot:

1. **Background:** all 8,507 molecules colored by cluster assignment (tab20/tab20b palette, cycling for k > 20)
2. **Overlay:** hit compounds and reference molecules drawn on top with distinct colors and marker shapes

This makes it possible to see simultaneously *where* the hits land in chemical space and *which clusters* they are near — supporting the key finding that hits are physicochemically familiar (sit near literature clusters) but scaffoldally novel (isolated island in MAPchiral space).

---

## Reference Molecule Marker Guide

| Molecule | Color | Marker | Why included |
|----------|-------|--------|--------------|
| 34_Hits (n=31) | Magenta `#FF00FF` | Circle | DEL screen hits |
| Hit (n=8) | Gold `#FFD700` | Circle | Named confirmed hits |
| Hexapeptide | Dark grey `#333333` | Square | Linear peptide reference |
| N-Me Hexapeptide | Dark grey `#333333` | Star | N-methylated reference |
| Cyclosporin A | Dark grey `#333333` | Triangle | Canonical macrocycle reference |

> Colors chosen to avoid clash with tab20/tab20b cluster palette (which contains reds and oranges).

---

## Figures

### Figure 1 — 2D Physicochemical UMAP · HDBSCAN
**File:** `outputs/analysis/2026-04-06/figures/2d_combined_hdbscan.svg`

- **Input features:** MW, cLogP, PSA, HBD, HBA, RotBonds (6 features)
- **UMAP params:** n_neighbors=30, min_dist=0.25
- **Clustering:** HDBSCAN · 55 clusters · 7.7% noise
- **Story:** hits clump in literature-dense regions → bulk properties are familiar

---

### Figure 2 — Mordred UMAP · HDBSCAN
**File:** `outputs/analysis/2026-04-06/figures/mordred_combined_hdbscan.svg`

- **Input features:** 335 Mordred 2D descriptors (cosine distance, 10D UMAP for clustering)
- **UMAP params:** n_neighbors=60, min_dist=0.1 (clustering); min_dist=0.2 (viz)
- **Clustering:** HDBSCAN · 60 clusters · 13.7% noise · DBCV = +0.531
- **Story:** hits still co-locate with literature clusters in fine-grained chemical space

---

### Figure 3 — Mordred UMAP · Ward k=64 *(algorithm convergence support)*
**File:** `outputs/analysis/2026-04-06/figures/mordred_combined_ward.svg`

- Same UMAP coordinates as Figure 2, different cluster coloring
- **Clustering:** Ward hierarchical · k=64 · silhouette = 0.794 · cophenetic r = 0.539
- **Why this matters:** HDBSCAN (60 clusters) and Ward (k=64) independently agree → strong validation that ~60–64 groups is the true structure of Mordred space

---

### Figure 4 — MAPchiral UMAP · K-Medoids k=15
**File:** `outputs/analysis/2026-04-06/figures/mapchiral_combined_kmedoids.svg`

- **Input features:** MAPchiral 2048-dim MinHash fingerprints (minhash_distance, 5D UMAP for clustering)
- **UMAP params:** n_neighbors=40, min_dist=0.1
- **Clustering:** K-Medoids · k=15 · silhouette = 0.624 · DBI = 0.549 (Gap Statistic confirmed k=15)
- **Story:** hits form an isolated island separate from all literature clusters → scaffolds are genuinely novel

---

## QC Summary

| Figure | Branch | Method | Clusters | Noise | Best score |
|--------|--------|--------|----------|-------|------------|
| Fig 1 | 2D Physicochemical | HDBSCAN | 55 | 7.7% | — |
| Fig 2 | Mordred | HDBSCAN | 60 | 13.7% | DBCV = +0.531 |
| Fig 3 | Mordred | Ward | 64 | — | Silhouette = 0.794 |
| Fig 4 | MAPchiral | K-Medoids | 15 | — | Silhouette = 0.624 |

---

## How Figures Were Generated

`03_umap_analysis/make_combined_figures.py`

```python
#!/usr/bin/env python3
"""Generate combined cluster-coloured + hit/reference overlay figures."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT      = Path(__file__).resolve().parent.parent
ALIGNED   = ROOT / "outputs/analysis/2026-04-06/aligned_metadata.csv"
MAPC_META = ROOT / "outputs/mapchiral/2026-04-06/mapchiral_metadata.csv"
FIG_DIR   = ROOT / "outputs/analysis/2026-04-06/figures"

COLOR_REF  = "#333333"   # dark grey  — references
COLOR_HITS = "#FF00FF"   # magenta    — 34_Hits (outside tab20/tab20b palette)
COLOR_HIT  = "#FFD700"   # gold       — Hit named (outside tab20/tab20b palette)

OVERLAY_STYLES = {
    "34_Hits":          (COLOR_HITS, "o",  40, 0.90),
    "Hit":              (COLOR_HIT,  "o",  60, 1.00),
    "Hexapeptide":      (COLOR_REF,  "s", 120, 1.00),
    "N-Me Hexapeptide": (COLOR_REF,  "*", 200, 1.00),
    "Cyclosporin A":    (COLOR_REF,  "^", 120, 1.00),
}
DRAW_ORDER = ["34_Hits", "Hit", "Hexapeptide", "N-Me Hexapeptide", "Cyclosporin A"]

_CMAPS = [plt.get_cmap("tab20"), plt.get_cmap("tab20b")]

def cluster_color(lbl):
    if lbl == -1:
        return "#CCCCCC"
    bank = lbl // 20
    return _CMAPS[bank % 2](lbl % 20)

def patch_highlight_id(df, meta):
    """Restore Highlight_ID, keeping exactly one point per reference name."""
    refs = meta[meta["Highlight_ID"].notna() & (meta["Highlight_ID"] != "")]
    ref_map = refs.set_index("Smiles")["Highlight_ID"]
    df = df.copy()
    df["Highlight_ID"] = df["Smiles"].map(ref_map)
    ref_rows = df[df["Highlight_ID"].notna()]
    keep_idx = ref_rows.drop_duplicates(subset="Highlight_ID", keep="first").index
    drop_idx = ref_rows.index.difference(keep_idx)
    df.loc[drop_idx, "Highlight_ID"] = float("nan")
    return df

def make_combined(df, emb_cols, cluster_col, branch, method_label, fig_name):
    x      = df[emb_cols[0]].values
    y      = df[emb_cols[1]].values
    labels = df[cluster_col].values
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor("#F5F5F5")
    cluster_handles = []
    for lbl in sorted(set(labels)):
        idx   = np.where(labels == lbl)[0]
        color = cluster_color(lbl)
        alpha = 0.15 if lbl == -1 else 0.45
        ax.scatter(x[idx], y[idx], s=4, c=[color], alpha=alpha,
                   linewidths=0, rasterized=True)
        name = "Noise" if lbl == -1 else f"Cluster {lbl} (n={len(idx)})"
        cluster_handles.append(
            plt.scatter([], [], s=18, c=[color], label=name, linewidths=0)
        )
    overlay_handles = []
    for role in DRAW_ORDER:
        if role == "34_Hits":
            mask = df["Source"] == "34_Hits"
        elif role == "Hit":
            mask = df["Source"] == "Hit"
        else:
            mask = df["Highlight_ID"] == role
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        color, marker, size, alpha = OVERLAY_STYLES[role]
        label = f"{role} (n={len(idx)})"
        ax.scatter(x[idx], y[idx], s=size, c=color, marker=marker,
                   alpha=alpha, linewidths=0.8, edgecolors="black", zorder=5)
        overlay_handles.append(
            plt.scatter([], [], s=size * 0.8, c=color, marker=marker,
                        edgecolors="black", linewidths=0.8, label=label)
        )
    cluster_leg = ax.legend(
        handles=cluster_handles,
        bbox_to_anchor=(1.01, 1), loc="upper left",
        fontsize=5, framealpha=0.7, ncol=3,
        handleheight=0.6, handlelength=0.8,
        borderpad=0.4, labelspacing=0.15, columnspacing=0.5,
        title="Clusters", title_fontsize=6,
    )
    ax.add_artist(cluster_leg)
    ax.legend(handles=overlay_handles,
              loc="lower left", fontsize=8,
              framealpha=0.85, borderpad=0.6,
              handletextpad=0.4, labelspacing=0.3)
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.set_title(f"{branch.upper()} — {method_label}", fontsize=9)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    path = FIG_DIR / fig_name
    fig.savefig(path, format="svg", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")

def main():
    df   = pd.read_csv(ALIGNED)
    meta = pd.read_csv(MAPC_META)
    df   = patch_highlight_id(df, meta)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    make_combined(df, ["umap1_2d","umap2_2d"], "hdbscan_2d",
        "2D Physicochemical", "HDBSCAN  55 clusters · noise=7.7%",
        "2d_combined_hdbscan.svg")

    make_combined(df, ["umap1_mordred","umap2_mordred"], "hdbscan_mordred",
        "Mordred", "HDBSCAN  60 clusters · DBCV = +0.531",
        "mordred_combined_hdbscan.svg")

    make_combined(df, ["umap1_mordred","umap2_mordred"], "ward_mordred",
        "Mordred", "Ward  k=64 · silhouette = 0.794",
        "mordred_combined_ward.svg")

    make_combined(df, ["umap1_mapchiral","umap2_mapchiral"], "kmedoids_mapchiral",
        "MAPchiral", "K-Medoids  k=15 · silhouette = 0.624",
        "mapchiral_combined_kmedoids.svg")

if __name__ == "__main__":
    main()
```

To regenerate all figures:
```bash
cd /path/to/Macrocycle
python3 03_umap_analysis/make_combined_figures.py
```

---

## Design Notes

**Why magenta and gold for hits?**
tab20 and tab20b (used for cluster colors) contain reds and oranges. Magenta (`#FF00FF`) and gold (`#FFD700`) are outside both palettes, ensuring hits are never visually confused with a cluster color.

**Why deduplicate reference molecules?**
The same SMILES for Cyclosporin A, Hexapeptide, and N-Me Hexapeptide each appear multiple times in the dataset (stereoisomeric variants). `patch_highlight_id()` maps by SMILES then keeps only the first occurrence per reference name so each appears exactly once as n=1 in the legend.

**Why a separate cluster legend outside the axes?**
With 55–64 clusters, an inside legend would occlude the data. The compact outside legend (fontsize=5, 3 columns) keeps all cluster labels visible without covering the UMAP.

---

## Related Documents

- `docs/2026-04-16_condition_D_chemical_space_findings.md` — full findings, QC scores, and pipeline summary for Condition D
- `docs/2026-04-21_tpsa_sandp_experiment.md` — PSA correction experiment (note: PSA values in the 2D branch used DataWarrior values; if recomputing from SMILES use `includeSandP=True`)
- `PIPELINE_STATUS.md` — run log and design decisions
