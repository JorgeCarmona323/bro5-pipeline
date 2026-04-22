# Condition D Chemical Space Analysis
### UMAP-Based Exploration of 8,507 Macrocycles
**Date:** 2026-04-16 | **Dataset:** Literature + 34_Hits + Hit (Condition D) | **Pipeline:** `03_umap_analysis/`

---

## Dataset

| Source | n | % of total |
|--------|---|------------|
| Literature | 8,468 | 99.5% |
| 34_Hits | 31 | 0.4% |
| Hit | 8 | 0.1% |
| **Total** | **8,507** | |

> **Note:** 39 hits total (0.5% hit rate). Sufficient for visual exploration and cluster-level inspection; too sparse for per-cluster statistical enrichment (most clusters have 0–1 hits).

---

## Key Finding

> **Hits are physicochemically familiar but scaffoldally novel.**

| Space | Hit distribution | Interpretation |
|-------|-----------------|----------------|
| 2D physicochemical | Hits clumped within literature-dense regions | Hits share bulk properties (MW, logP, PSA) with known compounds |
| Mordred descriptors | Hits clumped within literature-dense regions | Hits share fine-grained 2D chemistry with known compounds |
| MAPchiral scaffolds | Hits isolated on a separate island | Hits have ring systems and scaffolds not represented in the literature |

This pattern is ideal for a DEL screen result: the hits are not physicochemically anomalous (they look like drug-like compounds), but they represent genuinely unexplored structural territory not covered by the existing literature.

---

## Pipeline Summary

| Step | Method | Parameters |
|------|--------|------------|
| Descriptor computation | Mordred (335 features after filtering) | `compute_mordred_descriptors.py` |
| Fingerprint computation | MAPchiral (2048-dim MinHash) | `compute_mapchiral_fingerprints.py` |
| UMAP parameter tuning | 54-combination sweep (n_neighbors × min_dist × n_components) | `tune_umap_parameters.py` |
| Cluster size sweep | HDBSCAN min_cluster_size per branch | `sweep_min_cluster_size.py` |
| Full analysis | UMAP + HDBSCAN + Ward + K-Medoids | `analyse_chemical_space.py` |
| Combined figures | Cluster coloring + hit/reference overlay | `make_combined_figures.py` |

---

## Branch 1 — 2D Physicochemical (MW, cLogP, PSA, HBD, HBA, RotBonds)

**UMAP parameters:** n_neighbors=30, min_dist=0.25, n_components=2

### QC Scores

| Method | Clusters | Noise | DBCV |
|--------|----------|-------|------|
| HDBSCAN | 55 | 7.7% | — |

> No secondary clustering method applied — 6-feature panel is too coarse for meaningful flat partitioning beyond HDBSCAN micro-clusters.

### Figures

**Figure 1.** 2D Physicochemical UMAP — HDBSCAN clusters + hit/reference overlay
`figures/2d_combined_hdbscan.svg`

**Figure 2.** 2D Physicochemical UMAP — source coloring only
`figures/2d_umap_source.svg`

**Figure 3.** HDBSCAN condensed tree
`figures/2d_hdbscan_condensed_tree.svg`

**Figure 4.** HDBSCAN per-cluster stability
`figures/2d_hdbscan_stability.svg`

---

## Branch 2 — Mordred Descriptors (335 2D features)

**UMAP parameters:** n_neighbors=60, min_dist=0.1 (clustering, 10D); min_dist=0.2 (visualization, 2D)
**UMAP metric:** Cosine distance

### QC Scores

| Method | Clusters | Noise | DBCV | Silhouette | Cophenetic r |
|--------|----------|-------|------|------------|--------------|
| HDBSCAN | 60 | 13.7% | +0.531 | — | — |
| Ward hierarchical | 64 | — | — | 0.794 | 0.539 |

> **Convergence:** HDBSCAN (60 clusters) and Ward (k=64 silhouette peak) independently agree on ~60–64 groups — strong validation of the chemical space structure.

### Figures

**Figure 5.** Mordred UMAP — HDBSCAN clusters + hit/reference overlay *(primary figure)*
`figures/mordred_combined_hdbscan.svg`

**Figure 6.** Mordred UMAP — Ward k=64 clusters + hit/reference overlay *(supporting: algorithm convergence)*
`figures/mordred_combined_ward.svg`

**Figure 7.** Mordred UMAP — source coloring only
`figures/mordred_umap_source.svg`

**Figure 8.** Ward dendrogram (top 40 merges)
`figures/mordred_ward_dendrogram.svg`

**Figure 9.** Ward silhouette sweep k=2..100 — peak at k=64 (silhouette=0.794)
`figures/mordred_ward_silhouette_extended.svg`

**Figure 10.** HDBSCAN condensed tree
`figures/mordred_hdbscan_condensed_tree.svg`

**Figure 11.** HDBSCAN per-cluster stability
`figures/mordred_hdbscan_stability.svg`

---

## Branch 3 — MAPchiral Fingerprints (scaffold topology, 2048-dim MinHash)

**UMAP parameters:** n_neighbors=40, min_dist=0.1 (clustering, 5D); min_dist=0.1 (visualization, 2D)
**UMAP metric:** minhash_distance

### QC Scores

| Method | Clusters | Noise | DBCV | Silhouette | DBI |
|--------|----------|-------|------|------------|-----|
| HDBSCAN | 9 | 16.8% | +0.456 | — | — |
| K-Medoids | 15 | — | — | 0.624 | 0.549 |

> **Gap Statistic confirmed k=15** — scaffold space partitions cleanly into ~15 macro-families. Gap Statistic returned k=2 for Mordred (no valid flat structure), which is why Ward was used there instead.

### Figures

**Figure 12.** MAPchiral UMAP — K-Medoids k=15 clusters + hit/reference overlay *(primary figure)*
`figures/mapchiral_combined_kmedoids.svg`

**Figure 13.** MAPchiral UMAP — source coloring only
`figures/mapchiral_umap_source.svg`

**Figure 14.** MAPchiral UMAP — HDBSCAN clusters
`figures/mapchiral_umap_hdbscan.svg`

**Figure 15.** HDBSCAN condensed tree
`figures/mapchiral_hdbscan_condensed_tree.svg`

**Figure 16.** HDBSCAN per-cluster stability
`figures/mapchiral_hdbscan_stability.svg`

---

## QC Summary Table

| Branch | Method | Clusters | Noise | DBCV | Silhouette | DBI | Cophenetic r |
|--------|--------|----------|-------|------|------------|-----|--------------|
| 2D Physicochemical | HDBSCAN | 55 | 7.7% | — | — | — | — |
| Mordred | HDBSCAN | 60 | 13.7% | **+0.531** | — | — | — |
| Mordred | Ward k=64 | 64 | — | — | **0.794** | — | 0.539 |
| MAPchiral | HDBSCAN | 9 | 16.8% | **+0.456** | — | — | — |
| MAPchiral | K-Medoids k=15 | 15 | — | — | **0.624** | **0.549** | — |

> **DBCV > 0** = valid density-based cluster structure. **Silhouette → 1** = compact, well-separated clusters. **DBI → 0** = low within-cluster scatter relative to between-cluster distance.

---

## Design Decisions

**Why Ward for Mordred, not K-Medoids?**
Gap Statistic returned k=2 with monotonically rising gap — no valid flat cluster structure. Ward hierarchical respects the multi-level density structure; silhouette peak at k=64 independently confirmed by HDBSCAN (60 clusters).

**Why K-Medoids for MAPchiral?**
Gap Statistic confirmed k=15 with a genuine peak — scaffold space partitions cleanly into ~15 macro-families.

**Why no secondary clustering for 2D?**
6-feature panel is too coarse for meaningful flat partitioning beyond HDBSCAN micro-clusters.

**Why separate 2D UMAP embeddings for visualization vs. clustering?**
Clustering runs on nD embeddings (Mordred: 10D, MAPchiral: 5D) for structure fidelity. 2D embeddings use a looser min_dist for visual continuity. Cluster labels come from nD; x/y axes come from 2D.

---

## Output Files

All outputs in `outputs/analysis/2026-04-06/`:

```
aligned_metadata.csv          — all 8,507 molecules with cluster labels from all branches
2d_umap.csv                   — 2D branch: UMAP coords + HDBSCAN labels
mordred_umap.csv              — Mordred branch: UMAP coords + HDBSCAN + Ward labels
mapchiral_umap.csv            — MAPchiral branch: UMAP coords + HDBSCAN + K-Medoids labels

figures/
  2d_combined_hdbscan.svg         — Fig 1:  2D HDBSCAN + overlays (primary)
  mordred_combined_hdbscan.svg    — Fig 5:  Mordred HDBSCAN + overlays (primary)
  mordred_combined_ward.svg       — Fig 6:  Mordred Ward + overlays (supporting)
  mapchiral_combined_kmedoids.svg — Fig 12: MAPchiral K-Medoids + overlays (primary)
  mordred_ward_silhouette_extended.svg — silhouette sweep k=2..100
  mordred_ward_dendrogram.svg          — Ward dendrogram (top 40 merges)
  {branch}_hdbscan_condensed_tree.svg  — HDBSCAN condensed tree per branch
  {branch}_hdbscan_stability.svg       — per-cluster stability bar chart per branch
```

---

## Next Steps

1. **Visual cluster exploration** — identify which specific literature clusters the hits sit near in Mordred and 2D space
2. **Hit enrichment figure** — annotated overlay showing hit-adjacent literature clusters with molecule counts
3. **Feature enrichment** (future) — Mann-Whitney U on Mordred descriptors per cluster (n=39 hits limits per-cluster power)
4. **Chameleon_Predictor benchmark** — results will determine whether to add ETKDG 3D Mordred descriptors; see `docs/2026-04-09_feature_benchmark_implications.md`
