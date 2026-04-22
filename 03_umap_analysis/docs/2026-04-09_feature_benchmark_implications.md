# 2026-04-09 — Feature Benchmark Implications for Macrocycle Project

## Context

A decision-critical feature benchmark was designed in the **Chameleon_Predictor** project
(see `~/projects/Chameleon_Predictor/docs/experiments/2026-04-09_feature_benchmark_design.md`)
motivated by PROTAC-TS (Murakami et al., *JACS Au* 2026), which achieved R²=0.710 predicting
Caco-2 permeability on 43 PROTACs using only 500-dim count-based Morgan + TabPFN — no conformer
ensemble, no 3D simulation.

The results of that benchmark directly determine the feature strategy for this project too.
**Do not make new feature or pipeline decisions until the Chameleon_Predictor benchmark runs.**

---

## What the Benchmark Tests

8 feature sets × 3 models on the CREMP permeability subset (n=3,258):

| Feature | Relevance to Macrocycle |
|---|---|
| Morgan bit-based (r=2, 2048-dim) | Currently used in `Morgan_Bit_Introspection_Analysis.py` |
| Morgan count-based (r=2, 500-dim) | PROTAC-TS best performer — untested here |
| **MAPC / MAPchiral** | **Current UMAP fingerprint** (`compute_mapchiral_fingerprints.py`) |
| Mordred 2D only | **Current descriptor pipeline** (`compute_mordred_descriptors.py`) |
| Mordred 2D+3D (single ETKDG conformer) | Not yet in Macrocycle pipeline |
| CREST CHCl3 ensemble descriptors | Relevant if 3D adds signal |
| CREST aqueous ensemble descriptors | Pending compute |

---

## Decision Tree — What the Results Mean Here

### If Morgan count-based >> MAPC for permeability prediction:
- **UMAP:** MAPchiral may still be the right choice for *chemical space visualization* — it
  excels at capturing diversity and chirality-aware topology. Don't conflate regression
  performance with embedding quality. These are different tasks.
- **ML models (future):** Switch to count-based Morgan (r=2, 500-dim) as the fingerprint
  for any permeability prediction models built on this library.
- **Morgan_Bit_Introspection_Analysis.py:** Update to count-based if building ML on top.

### If MAPC >= Morgan for permeability:
- Current MAPchiral choice is validated for both UMAP and any downstream ML.
- No changes needed to fingerprint strategy.

### If Mordred 2D+3D (ETKDG) >> Mordred 2D only:
- Add a conformer generation step to `compute_mordred_descriptors.py`:
  - One ETKDG call per molecule (RDKit, fast, ~seconds for 8,507 molecules)
  - Enable 3D descriptor computation in mordred (`mordred.Calculator(descs, ignore_3D=False)`)
- This is a small addition — the filtering/scaling pipeline downstream is unchanged.

### If CREST 3D >> ETKDG 3D:
- Consider running CREST on the Macrocycle library (8,507 molecules is large — ~3.9M CPU-hrs
  for CREMP's 36K molecules implies ~1M CPU-hrs for 8.5K, still substantial).
- Only worthwhile if the delta between CREST and ETKDG 3D is large.
- **Hold this decision until the benchmark result.**

### If ETKDG 3D ≈ CREST 3D:
- Do not invest compute in CREST for this library. ETKDG + Mordred 3D is sufficient.

---

## Current State of This Project

- `compute_mapchiral_fingerprints.py` — Done (2026-04-06), 8,507 molecules, 0 failures
- `compute_mordred_descriptors.py` — Done (2026-04-06), 2D only, 8,507 × 335 features
- `tune_umap_parameters.py` — **Ready to run tonight.** Fix already committed (fb1992b).
  All data loads cleanly. Run from `03_umap_analysis/`:
  ```bash
  python tune_umap_parameters.py
  ```
  Outputs go to `outputs/tuning/2026-04-06/`. Sweep is 27 combinations × 2 branches = 54 UMAP runs.

---

## Recommended Order of Operations

1. **Tonight:** Run `tune_umap_parameters.py` — gets optimal UMAP parameters for both branches
2. **After Chameleon_Predictor benchmark (F1–F7):** Decide whether to add ETKDG 3D to Mordred pipeline
3. **After CREST aqueous runs complete:** Run full benchmark (F1–F8), finalize feature strategy
4. **Then:** Update `analyse_chemical_space.py` with optimal parameters from step 1 and any new features from step 2

---

## Why Both Projects Are Coupled

The Chameleon_Predictor is running CREST on macrocyclic peptides and building permeability
models. This project is visualizing the chemical space of macrocyclic DEL hits and will
eventually need permeability models for hit prioritization. The feature benchmark answers
the same question for both: **what descriptors actually carry permeability signal for macrocycles?**

One experiment, two projects.
