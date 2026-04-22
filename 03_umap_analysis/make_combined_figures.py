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
FIG_DIR   = ROOT / "outputs/analysis/2026-04-21/figures"

# ── Overlay marker styles ──────────────────────────────────────────────────
COLOR_REF  = "#333333"   # dark grey  — references
COLOR_HITS = "#FF00FF"   # magenta    — 34_Hits (outside tab20/tab20b palette)
COLOR_HIT  = "#FFD700"   # gold       — Hit named (outside tab20/tab20b palette)

# (color, marker, size, alpha)
OVERLAY_STYLES = {
    "34_Hits":         (COLOR_HITS, "o",  40, 0.90),
    "Hit":             (COLOR_HIT,  "o",  60, 1.00),
    "Hexapeptide":     (COLOR_REF,  "s", 120, 1.00),   # square
    "N-Me Hexapeptide":(COLOR_REF,  "*", 200, 1.00),   # star
    "Cyclosporin A":   (COLOR_REF,  "^", 120, 1.00),   # triangle
}
DRAW_ORDER = ["34_Hits", "Hit", "Hexapeptide", "N-Me Hexapeptide", "Cyclosporin A"]


# ── Cluster colourmap: tab20 + tab20b, cycling for any k ──────────────────
_CMAPS = [plt.get_cmap("tab20"), plt.get_cmap("tab20b")]

def cluster_color(lbl):
    if lbl == -1:
        return "#CCCCCC"
    bank = lbl // 20
    return _CMAPS[bank % 2](lbl % 20)


def patch_highlight_id(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Restore Highlight_ID, keeping exactly one point per reference name."""
    refs = meta[meta["Highlight_ID"].notna() & (meta["Highlight_ID"] != "")]
    ref_map = refs.set_index("Smiles")["Highlight_ID"]
    df = df.copy()
    df["Highlight_ID"] = df["Smiles"].map(ref_map)
    # Deduplicate: keep only first row per reference name in df
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

    # ── Layer 1: cluster-coloured background ──────────────────────────────
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

    # ── Layer 2: hit / reference overlays ─────────────────────────────────
    overlay_handles = []
    for role in DRAW_ORDER:
        if role == "34_Hits":
            mask = df["Source"] == "34_Hits"
        elif role == "Hit":
            mask = df["Source"] == "Hit"
        else:  # individual reference names
            mask = df["Highlight_ID"] == role

        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue

        color, marker, size, alpha = OVERLAY_STYLES[role]
        label = f"{role} (n={len(idx)})"
        ax.scatter(x[idx], y[idx],
                   s=size, c=color, marker=marker,
                   alpha=alpha, linewidths=0.8, edgecolors="black",
                   zorder=5)
        overlay_handles.append(
            plt.scatter([], [], s=size * 0.8, c=color, marker=marker,
                        edgecolors="black", linewidths=0.8, label=label)
        )

    # ── Cluster legend: inside upper right, compact, ncol scales with k ───
    n_unique = len(set(labels))
    n_cols   = 4 if n_unique > 30 else 3 if n_unique > 15 else 2
    fs       = 4 if n_unique > 30 else 5
    cluster_leg = ax.legend(
        handles=cluster_handles,
        loc="upper right",
        fontsize=fs, framealpha=0.7,
        ncol=n_cols,
        handleheight=0.6, handlelength=0.8,
        borderpad=0.4, labelspacing=0.15, columnspacing=0.5,
        title="Clusters", title_fontsize=6,
    )
    ax.add_artist(cluster_leg)

    # ── Overlay legend: lower left, always visible ────────────────────────
    if overlay_handles:
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

    n_refs = int(df["Highlight_ID"].notna().sum())
    print(f"Reference molecules matched: {n_refs} "
          f"({df[df['Highlight_ID'].notna()]['Highlight_ID'].value_counts().to_dict()})")

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("\nGenerating combined figures...")

    make_combined(df,
        emb_cols    = ["umap1_2d", "umap2_2d"],
        cluster_col = "hdbscan_2d",
        branch      = "2D Physicochemical",
        method_label= "HDBSCAN  55 clusters · noise=7.7%",
        fig_name    = "2d_combined_hdbscan.svg")

    make_combined(df,
        emb_cols    = ["umap1_mordred", "umap2_mordred"],
        cluster_col = "hdbscan_mordred",
        branch      = "Mordred",
        method_label= "HDBSCAN  35 clusters · DBCV = +0.511",
        fig_name    = "mordred_combined_hdbscan.svg")

    make_combined(df,
        emb_cols    = ["umap1_mordred", "umap2_mordred"],
        cluster_col = "ward_mordred",
        branch      = "Mordred",
        method_label= "Ward  k=64 · silhouette = 0.794",
        fig_name    = "mordred_combined_ward.svg")

    make_combined(df,
        emb_cols    = ["umap1_mapchiral", "umap2_mapchiral"],
        cluster_col = "kmedoids_mapchiral",
        branch      = "MAPchiral",
        method_label= "K-Medoids  k=15 · silhouette = 0.624",
        fig_name    = "mapchiral_combined_kmedoids.svg")

    print("\nDone. 4 figures written to:", FIG_DIR)


if __name__ == "__main__":
    main()
