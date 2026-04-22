# Chemical Space Analysis — Update 2026-04-22
**Project:** Macrocycle DEL Chemical Space Analysis | **Condition:** D | **Branch:** main

> This document supersedes `2026-04-21_recent_experiments.md` and `2026-04-16_condition_D_chemical_space_findings.md` for all clustering parameters, figure descriptions, and 2D branch interpretation.

---

## Current Analysis State

### Dataset
- **8,507 aligned molecules** across all three branches (inner join on canonical SMILES)
- Sources: Literature 8,468 · 34_Hits 31 · Hit 8
- Condition D: Literature + 34_Hits + Hit (Library excluded)

---

## Branch 1 — 2D Physicochemical

### Decision: UMAP visualization replaced by enrichment analysis

The 2D physicochemical UMAP (6 features: MW, cLogP, TPSA, HBA, HBD, RotBonds) produced a fragmented layout with DBCV going negative at mcs=170 (10 clusters). Six features lack the discriminating power to form density-based clusters comparable to Mordred or MAPchiral. A UMAP figure for this branch is not appropriate for publication.

**Replaced with:** Physicochemical enrichment analysis (`physicochemical_enrichment.py`)

### Enrichment Results (corrected — n=36 hits vs n=7,990 library)

> Note: An earlier run inflated counts to 84 hits / 9,620 library due to 517 duplicate canonical SMILES in the master CSV. Fixed by deduplication post-merge.

| Property | Library median | Hit median | Cohen's d | MWU p | Direction |
|----------|---------------|-----------|----------|-------|-----------|
| Molecular Weight | 856 Da | 823 Da | −0.68 | 0.028 | lower |
| cLogP | 2.84 | −0.71 | **−2.73** | <0.001 | **MUCH lower** |
| TPSA | 180 Å² | 288 Å² | **+1.69** | <0.001 | **HIGHER** |
| H-Bond Acceptors | 14 | 16 | −0.47 | 0.757 | not significant |
| H-Bond Donors | 4 | 7 | **+2.01** | <0.001 | **HIGHER** |
| Rotatable Bonds | 10 | 7 | −1.52 | <0.001 | lower |

**TPSA interpretation note:** All hits share two thioether bridges (−CSC−) contributing +50.60 Å² baseline to DataWarrior TPSA (Ertl 2000, includes S/P). Thioethers are non-H-bonding and lipophilic — this contribution is a descriptor artifact for permeability purposes. Permeability-relevant TPSA uses RDKit default (excludes S). See Cmp1-4 section.

### Figures (outputs/analysis/2026-04-21/figures/)
- `2d_enrichment_kde.svg` — 6-panel KDE, library vs hits per property, annotated with Cohen's d and MWU p-value
- `2d_enrichment_scatter.svg` — 6 property-pair scatter panels, hits overlaid
- `2d_enrichment_radar.svg` — radar chart, hit median vs library median normalised to library IQR

---

## Branch 2 — Mordred (335 descriptors, 10D UMAP)

### Clustering — Updated mcs=80

Previous mcs=50 was the statistical peak (60 clusters, DBCV=+0.534) but exceeded the 10–30 cluster interpretability range. Fine-grained mcs sweep (steps of 10, range 50–200) identified mcs=80 as the principled choice: above the DBCV=+0.45 quality threshold with 35 clusters.

| mcs | Clusters | Noise | Silhouette | DBCV | Selected |
|-----|----------|-------|-----------|------|----------|
| 50 | 60 | 13.7% | +0.747 | +0.534 | — statistical peak |
| **80** | **35** | **18.4%** | **+0.768** | **+0.511** | ← **current** |
| 90 | 28 | 19.5% | +0.735 | +0.410 | — below 0.45 threshold |

### Secondary clustering — Ward k=64
Silhouette peak confirmed at k=64 (silhouette=0.794, cophenetic r high). Unchanged.

### Figures
- `mordred_combined_hdbscan.svg` — HDBSCAN 35 clusters · DBCV=+0.511 *(updated)*
- `mordred_combined_ward.svg` — Ward k=64 · silhouette=0.794 *(unchanged)*

---

## Branch 3 — MAPchiral (2048-bit fingerprints, 5D UMAP)

### Clustering — K-Medoids k=15 retained

Fine-grained mcs sweep showed MAPchiral DBCV improves monotonically with larger mcs (mcs=200: 14 clusters, DBCV=+0.472). K-Medoids k=15 (Gap Statistic confirmed, silhouette=0.624) remains the primary figure — comparable cluster count with independent validation method.

HDBSCAN mcs updated to 180 in `analyse_chemical_space.py` for alignment, but k-medoids figure is the publication figure.

### Figure
- `mapchiral_combined_kmedoids.svg` — K-Medoids k=15 · silhouette=0.624 *(unchanged)*

---

## Cmp1-4 Physicochemical Profile

Compounds 1–4 are the four named hits used in the TPSA S/P experiment (2026-04-21). All share the same macrocycle scaffold with two thioether bridges.

### TPSA breakdown (DataWarrior vs permeability-relevant)

| Compound | DataWarrior TPSA | Thioether +S | Extra S source | Permeability TPSA (noSP) | cLogP (RDKit) |
|----------|-----------------|-------------|---------------|--------------------------|---------------|
| Cmp1 DOPC 2-9-9-8 | 264 Å² | +50.60 | — | **213 Å²** | +1.32 |
| Cmp2 DOPC 3-12-8-12 | 328 Å² | +50.60 | +28.24 (thiophene) | **249 Å²** | −1.49 |
| Cmp3 DOPC 1-6-4-7 | 342 Å² | +50.60 | +8.38 (sulfonyl S) | **284 Å²** | −2.23 |
| Cmp4 Brain 6-4-4-13 | 280 Å² | +50.60 | — | **229 Å²** | +1.80 |

Permeability-relevant TPSA = RDKit default (S/P excluded). Thioether and thiophene sulfur do not contribute to H-bonding or aqueous solvation costs at the membrane. Sulfonyl oxygens in Cmp3 are included in the noSP value — the 284 Å² reflects genuine polarity.

### Permeability assessment

| Compound | Permeability TPSA | cLogP | Assessment |
|----------|------------------|-------|-----------|
| **Cmp1** | 213 Å² | +1.32 | CsA territory (ref: 203 Å²). Best candidate. |
| **Cmp2** | 249 Å² | −1.49 | Borderline. Thiophene adds lipophilicity. Optimisable. |
| **Cmp3** | 284 Å² | −2.23 | Sulfonic acid sidechain (pKa ~1) fully ionised at pH 7.4. **See note.** |
| **Cmp4** | 229 Å² | +1.80 | Strong profile. Best cLogP. Challenging for CNS (<90 Å²) but viable for other routes. |

> **Cmp3 anomaly (pending clarification):** A permanently charged compound (−1 at any physiological pH) passing a permeability screen is inconsistent with passive transcellular permeability at 825 Da. Possible explanations: active transporter substrate (cell-based assay), assay artifact (aggregation/fluorescence), or compound instability. Follow-up with Dr. Hu on assay format (PAMPA vs Caco-2/MDCK) and re-confirmation status before interpreting as a validated permeable hit. If confirmed, the sulfonyl position is a priority SAR vector — neutral bioisostere series (sulfonamide, hydroxymethyl, Ala) would deconvolute charge vs geometry contribution.

### cLogP note
RDKit (Crippen) and DataWarrior cLogP differ by ~1.7–2 units across all four compounds (DataWarrior consistently more negative). Neither method is well-validated for macrocyclic peptides. LogD at pH 7.4 from a pKa-aware method is the appropriate descriptor for permeability prediction — not cLogP.

---

## Audit Notes

Three issues identified and resolved:

| Issue | Status |
|-------|--------|
| `analyse_chemical_space.py` RUN_TAG stale ("2026-04-16") — outputs in 2026-04-06 | ✅ Fixed → "2026-04-06"; comment added to update before each run |
| `HDBSCAN_MIN_CLUSTER_SIZE` not reflecting sweep results (mordred=50, mapchiral=350) | ✅ Fixed → mordred=80, mapchiral=180 |
| `aligned_metadata.csv` 2D branch: cluster labels (nn=15/md=0.15) mismatched with updated viz coords (nn=40/md=0.4) | ✅ Resolved by decision to drop 2D UMAP figure; enrichment analysis used instead |
| `physicochemical_enrichment.py` duplicate SMILES inflating counts (84 hits, 9620 lib) | ✅ Fixed → drop_duplicates post-merge; corrected to 36 hits, 7990 lib |

---

## Output File Index

| File | Description |
|------|-------------|
| `outputs/analysis/2026-04-21/figures/mordred_combined_hdbscan.svg` | Mordred HDBSCAN 35 clusters, mcs=80 |
| `outputs/analysis/2026-04-21/figures/mordred_combined_ward.svg` | Mordred Ward k=64 |
| `outputs/analysis/2026-04-21/figures/mapchiral_combined_kmedoids.svg` | MAPchiral k-medoids k=15 |
| `outputs/analysis/2026-04-21/figures/2d_enrichment_kde.svg` | 2D branch KDE enrichment panel |
| `outputs/analysis/2026-04-21/figures/2d_enrichment_scatter.svg` | 2D branch property-pair scatters |
| `outputs/analysis/2026-04-21/figures/2d_enrichment_radar.svg` | 2D branch radar chart |
| `outputs/analysis/2026-04-21/2d_enrichment_stats.csv` | Cohen's d + MWU per property |
| `outputs/analysis/2026-04-06/aligned_metadata.csv` | Master aligned dataset (source of truth) |
| `outputs/tuning/2026-04-06/mordred_mcs_sweep.csv` | Fine-grained mcs sweep results |
| `03_umap_analysis/physicochemical_enrichment.py` | Enrichment analysis script |
| `03_umap_analysis/update_mordred_mcs80.py` | Mordred mcs=80 re-cluster script |
| `03_umap_analysis/sweep_min_cluster_size.py` | Fine-grained mcs sweep (all 3 branches) |
