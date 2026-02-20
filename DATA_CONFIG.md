# Data Configuration

Data files are **not committed to git** (too large, contains raw research data).
Drop files into the appropriate folders below after copying from Windows.

## Folder Map

### `data/building_blocks/`
Master lookup table for enumeration.

| File | What it is |
|------|-----------|
| `Master_Building_Blocks_Cleaned.csv` | Monomer name → cleaned SMILES lookup. Columns: `monomer`, `Smiles_Cleaned` |

### `data/libraries/YYYY-MM-DD/`
DEL hit lists and peptide libraries, organized by date collected/generated.

| File | What it is |
|------|-----------|
| `2026-01-22/34_Hit_values_extracted.csv` | DEL screening hits with X1–X4 building block numbers. Columns: `Reactant_Value_1` through `Reactant_Value_4` |
| `2026-01-29/Smiles.smi` | Flat file of peptide SMILES for Dibromo cyclization notebook (one SMILES per line) |
| `YYYY-MM-DD/your_file.csv` | Add new dated batches as subdirectories |

### `data/lcms/uv/`
Chromeleon UV-VIS exports (.txt). One file per purified sample.

| Format | What it is |
|--------|-----------|
| `SampleName_YYYYMMDD.txt` | Tab-separated retention time + intensity (mAU) from UV detector |

### `data/lcms/ms/`
Converted mass spectrometry files (.mzML). One file per run.

| Format | What it is |
|--------|-----------|
| `SampleName_RAW.mzML` | Scan-by-scan m/z and intensity values across the LC run |

---

## File Registry
Tracks known data files, their origin, and current status. Update this as new files are added.

### Building Blocks
| File | Date Generated | Origin | Status |
|------|---------------|--------|--------|
| `Master_Building_Blocks_Cleaned.csv` | 2026-01-29 | Cleaned from X1–X4 raw libraries via `02_data_cleaning/` scripts | Pending copy from Windows |

### Libraries
| File | Date Generated | Origin | Status |
|------|---------------|--------|--------|
| `2026-01-22/34_Hit_values_extracted.csv` | 2026-01-22 | Extracted from DEL screening hit data via `extract_reactant_values.py` | Pending copy from Windows |
| `2026-01-29/Smiles.smi` | 2026-01-29 | Peptide SMILES library used for Dibromo cyclization | Pending copy from Windows |

### LCMS
| File | Date Generated | Origin | Status |
|------|---------------|--------|--------|
| `uv/Brain-Ala-4-4-13_PURE_20251114.txt` | 2025-11-14 | Chromeleon UV-VIS export, purified sample | Pending copy from Windows |
| `ms/X1_MS_RAW.mzML` | — | Converted mass spec file for X1 sample | Pending copy from Windows |

> Add new rows here as you generate or collect new data files.

---

## Copying from Windows to WSL2

```bash
# Building blocks
cp /mnt/c/Users/Admin/path/to/Master_Building_Blocks_Cleaned.csv ~/projects/Macrocycle/data/building_blocks/

# DEL hit lists — organize by date
cp /mnt/c/Users/Admin/path/to/34_Hit_values_extracted.csv ~/projects/Macrocycle/data/libraries/2026-01-22/
cp /mnt/c/Users/Admin/path/to/Smiles.smi ~/projects/Macrocycle/data/libraries/2026-01-29/

# LCMS UV data (.txt)
cp /mnt/c/Users/Admin/path/to/*.txt ~/projects/Macrocycle/data/lcms/uv/

# LCMS mass spec data (.mzML)
cp /mnt/c/Users/Admin/path/to/*.mzML ~/projects/Macrocycle/data/lcms/ms/
```

---

## Script Path Configuration

| Script | Variable to update |
|--------|--------------------|
| `01_enumeration/enumerate_macrocycles.py` | `BUILDING_BLOCKS_CSV`, `INPUT_INSTRUCTIONS_CSV`, `OUTPUT_DIR` |
| `03_umap_analysis/UMAP_Figure_Macrocycles_v2_*.py` | `INPUT_CSV`, `OUTPUT_DIR`, `FIG_DIR` |
| `04_validation/LCMS_Analysis_and_Plots.ipynb` | `DATA_ROOT`, `OUTPUT_ROOT` in cell 1 |
| `04_validation/Dibromo_Linker_Reaction_v2.ipynb` | `DATA_ROOT`, `OUTPUT_ROOT` in cell 2 |
