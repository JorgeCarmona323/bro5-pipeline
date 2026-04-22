# Recent Experiments — 2026-04-21
**Project:** Macrocycle DEL Chemical Space Analysis | **Branch:** main

---

## 1. TPSA S/P Inclusion Experiment

**Script:** `03_umap_analysis/tpsa_sandp_experiment.py`
**Full write-up:** `docs/2026-04-21_tpsa_sandp_experiment.md`

### Finding
RDKit's `CalcTPSA()` excludes S and P atoms by default. DataWarrior uses Ertl's 2000 algorithm which includes them. For Cys-based macrocycles with two thioether bridges, the default RDKit TPSA undercounts by **50–79 Å²**.

| Compound | RDKit default | RDKit +S/P | DataWarrior | Δ fixed |
|----------|--------------|-----------|-------------|---------|
| Cmp1 DOPC 2-9-9-8 | 213.25 | 263.85 | 263.85 | ✓ |
| Cmp2 DOPC 3-12-8-12 | 249.36 | 328.20 | 328.20 | ✓ |
| Cmp3 DOPC 1-6-4-7 | 283.50 | 342.48 | 342.48 | ✓ |
| Cmp4 Brain 6-4-4-13 | 229.13 | 279.73 | 279.73 | ✓ |

**Fix:** `rdMolDescriptors.CalcTPSA(mol, includeSandP=True)` everywhere in the pipeline.

---

## 2. Combined UMAP Figures — Legend and Label Fixes

**Script:** `03_umap_analysis/make_combined_figures.py`
**Full write-up:** `docs/2026-04-21_combined_figures_notion.md`

### Changes made
- Cluster legend moved **inside upper right** for all 4 figures (was outside-axes right)
- `ncol` now scales dynamically: 4 columns for >30 clusters, 2 columns for ≤15
- Figure 1 (2D physicochemical) method label corrected: `"55 clusters · noise=7.7%"` (was stale at 76 clusters)
- Overlay legend (`if overlay_handles:` guard) — ensures hit/reference legend always renders when data is present

### Figures produced
| File | Branch | Clustering |
|------|--------|-----------|
| `2d_combined_hdbscan.svg` | 2D Physicochemical | HDBSCAN 55 clusters |
| `mordred_combined_hdbscan.svg` | Mordred | HDBSCAN 60 clusters · DBCV=+0.531 |
| `mordred_combined_ward.svg` | Mordred | Ward k=64 · silhouette=0.794 |
| `mapchiral_combined_kmedoids.svg` | MAPchiral | K-Medoids k=15 · silhouette=0.624 |

---

## 3. 2D UMAP Parameter Tuning — Clustering (`tune_umap_parameters.py`)

**Script:** `03_umap_analysis/tune_umap_parameters.py`

### Change: added 2D physicochemical branch
Previously the parameter sweep only covered Mordred and MAPchiral. The 2D physicochemical branch UMAP was never formally optimized — parameters were set manually.

Added to the sweep:
- `load_2d()` — loads 6-feature panel, applies IQR clip + StandardScaler (matches `analyse_chemical_space.py`)
- Euclidean metric routing across `run_combination`, `compute_trustworthiness`, `_get_2d_embedding`
- Branch-specific grid: `n_components=[2,3,4]`, `n_neighbors=[10,20,30]` (smaller range appropriate for 6 features)
- Helper functions `_branch_sweep_grid()` and `_branch_metric_label()` to keep branch logic clean

---

## 4. 2D UMAP Visualization Sweep (`tune_viz2d.py`)

**Script:** `03_umap_analysis/tune_viz2d.py`
**Results:** `outputs/tuning/2026-04-06/viz2d_sweep_results.csv`

### Motivation
Clustering uses nD UMAP embeddings (Mordred: 10D, MAPchiral: 5D). The 2D x/y coordinates are only for visualization. Optimizing for clustering quality (DBCV, ARI) does not guarantee a visually connected layout. A dedicated sweep targeting visual continuity was missing.

### Metrics (all higher = better)
| Metric | Weight | What it measures |
|--------|--------|-----------------|
| Trustworthiness | 40% | nn in embedding are also nn in original space |
| Continuity | 30% | nn in original space appear in embedding |
| Connected score | 30% | fraction of points in the largest 2D kNN component — penalises fragmented island layouts |

### Sweep grid
- `n_neighbors`: [20, 30, 40]
- `min_dist`: [0.1, 0.2, 0.3, 0.4]
- 12 combinations × 3 branches = 36 UMAP runs

### Results

| Branch | Best nn | Best md | Trust | Continuity | Connected | Score |
|--------|---------|---------|-------|------------|-----------|-------|
| 2D physicochemical | 40 | 0.4 | 0.980 | 0.918 | 0.402 | 0.909 |
| Mordred | 40 | 0.3 | 0.971 | 0.922 | 0.224 | 0.958 |
| MAPchiral | 40 | 0.3 | 0.983 | 0.981 | 0.210 | 0.961 |

All three branches converged on `n_neighbors=40`. Higher min_dist (0.3–0.4 vs previous 0.1–0.25) reduces the fragmented island appearance without distorting local structure.

The 2D physicochemical branch has the lowest connected score (0.40) even at best parameters — reflecting that 6 physicochemical features genuinely produce a more fragmented manifold than 335 Mordred descriptors or 2048 MAPchiral fingerprints.

### What was updated
- `outputs/analysis/2026-04-06/aligned_metadata.csv` — 2D viz coords for all three branches replaced with best sweep results
- All 4 combined figures regenerated with new coordinates
- Grid comparison figures: `outputs/tuning/2026-04-06/figures/viz2d_{branch}_grid.svg`

---

## Output File Index

| File | Description |
|------|-------------|
| `03_umap_analysis/tpsa_sandp_experiment.py` | TPSA S/P comparison script |
| `03_umap_analysis/make_combined_figures.py` | Combined cluster + overlay figures |
| `03_umap_analysis/tune_umap_parameters.py` | Clustering UMAP sweep (now includes 2D branch) |
| `03_umap_analysis/tune_viz2d.py` | Visualization UMAP sweep (new) |
| `outputs/analysis/2026-04-06/aligned_metadata.csv` | Updated with best viz2d coords |
| `outputs/analysis/2026-04-06/figures/` | All combined figures (regenerated) |
| `outputs/tuning/2026-04-06/viz2d_sweep_results.csv` | Full sweep rankings |
| `outputs/tuning/2026-04-06/figures/viz2d_*_grid.svg` | 12-combo comparison grids |
