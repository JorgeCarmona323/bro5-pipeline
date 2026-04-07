# Pipeline Status — 2026-04-06

## What was completed today (lab session)

### Scripts written / modified
| Script | Status | Notes |
|--------|--------|-------|
| `compute_mordred_descriptors.py` | Done | DATA_CONDITION="D"; boolean-col cast fix applied |
| `compute_mapchiral_fingerprints.py` | Done | DATA_CONDITION="D"; 8507/8507 mols, 0 failures |
| `analyse_chemical_space.py` | Done | Not yet run |
| `tune_umap_parameters.py` | Done | Not yet run successfully |

### Runs completed
- `compute_mapchiral_fingerprints.py` — **SUCCESS** (425 s). Outputs in `outputs/mapchiral/2026-04-06/`
- `compute_mordred_descriptors.py` — **SUCCESS** (1064 s). Outputs in `outputs/mordred/2026-04-06/`

---

## What to do tonight (home)

### Step 1 — Fix `tune_umap_parameters.py` loader (2 min)

The tuning script crashes because `mordred_filtered_scaled.csv` contains non-numeric
columns beyond `Smiles` and `Source`. Need to identify them and add to the meta exclusion list.

Run this to find the offending columns:
```bash
python -c "
import pandas as pd
df = pd.read_csv('outputs/mordred/2026-04-06/mordred_filtered_scaled.csv', low_memory=False)
for c in df.columns:
    try:
        pd.to_numeric(df[c], errors='raise')
    except:
        print('Non-numeric col:', c, '| sample:', df[c].iloc[0])
"
```

Then in `tune_umap_parameters.py` around line 204, expand the meta columns list:
```python
# Current:
META_COLS = [SMILES_COL, SOURCE_COL]
# Likely fix — add any non-numeric ID columns found above, e.g.:
META_COLS = [SMILES_COL, SOURCE_COL, "Compound_ID"]   # adjust name as needed
```

Or simply change the loader to drop non-numeric columns automatically:
```python
def load_mordred() -> tuple[pd.DataFrame, pd.DataFrame]:
    df   = pd.read_csv(MORDRED_SCALED_CSV, low_memory=False)
    meta = df[[c for c in [SMILES_COL, SOURCE_COL] if c in df.columns]].copy()
    num_cols = df.select_dtypes(include="number").columns
    X    = df[num_cols].astype(float)
    print(f"   {X.shape[0]:,} × {X.shape[1]}")
    return meta, X
```

### Step 2 — Run the tuning sweep (long, leave overnight)
```bash
python 03_umap_analysis/tune_umap_parameters.py
```
This runs 54 UMAP/HDBSCAN combinations (27 Mordred + 27 MAPchiral).
Results are cached — safe to interrupt and resume.
Outputs go to `outputs/tuning/2026-04-06/`.

### Step 3 — After tuning finishes: update params & run analysis
1. Review `outputs/tuning/2026-04-06/{mordred,mapchiral}_selection_report.txt`
2. Update `UMAP_BASE_PARAMS` in `analyse_chemical_space.py` with the winning settings
3. Run:
```bash
python 03_umap_analysis/analyse_chemical_space.py
```

---

## Key design decisions (for reference)

| Branch | UMAP metric | Preprocessing |
|--------|-------------|---------------|
| 2D descriptors | Euclidean | IQR clip + StandardScaler (8 features, no variance pruning) |
| Mordred | Cosine | drop >10% NaN cols → median impute → NZV → Spearman corr filter → IQR clip → StandardScaler |
| MAPchiral | `minhash_distance` (@njit) | **none** — uint32 MinHash, positional equality must be preserved |

- Clustering always on UMAP embeddings (Euclidean), NOT raw feature space
- DATA_CONDITION = "D" → 8,507 mols (34_Hits + Hit + Literature)
- Dimensionality sweep: n_components=[5,10,15]; stability = |Δclusters|≤2 AND |Δnoise_frac|≤0.05
