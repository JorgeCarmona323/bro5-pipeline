# Pipeline Status — updated 2026-04-16

## Completed

| Script | Status | Date | Notes |
|--------|--------|------|-------|
| `compute_mordred_descriptors.py` | Done | 2026-04-06 | 8,507 × 335 descriptors, 0 failures |
| `compute_mapchiral_fingerprints.py` | Done | 2026-04-06 | 8,507 × 2,048 MinHash fps, 0 failures |
| `tune_umap_parameters.py` | Done | 2026-04-06 | 54 combos swept; mordred: nn=60 md=0.1; mapchiral: nn=40 md=0.1 |
| `sweep_min_cluster_size.py` | Done | 2026-04-06 | mordred mcs=50, mapchiral mcs=350 |
| `analyse_chemical_space.py` | Done | 2026-04-16 | Full validated run; see findings doc |

## Current outputs

- `outputs/analysis/2026-04-06/` — cluster labels, UMAP coords, figures for all 3 branches
- Key findings: `docs/2026-04-16_condition_D_chemical_space_findings.md`

## Design decisions

| Branch | UMAP metric | Clustering | n_components |
|--------|-------------|------------|--------------|
| 2D (6 features) | Euclidean | HDBSCAN only | 2 |
| Mordred (335 features) | Cosine | HDBSCAN + Ward (k=64, sil=0.794) | 10 |
| MAPchiral (2048-dim) | minhash_distance | HDBSCAN + K-Medoids (k=15) | 5 |

**Why Ward for Mordred, not K-Medoids:** Gap Statistic returned k=2 with monotonically rising
gap — no valid flat cluster structure. Ward hierarchical respects the multi-level density
structure; silhouette peak at k=64 independently confirmed by HDBSCAN (60 clusters).

**Why K-Medoids for MAPchiral:** Gap Statistic confirmed k=15 with genuine peak — scaffold
space partitions cleanly into ~15 macro-families.

**Why no secondary clustering for 2D:** 6-feature panel is too coarse for meaningful flat
partitioning beyond HDBSCAN micro-clusters.

## Next steps

1. Visual exploration — identify which literature clusters the hits sit near (in progress)
2. Hit enrichment overlay figure
3. Feature enrichment (Mordred descriptors per cluster) — after visual exploration confirms signal
4. Await Chameleon_Predictor feature benchmark before adding 3D Mordred descriptors
