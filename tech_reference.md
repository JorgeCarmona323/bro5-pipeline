# Macrocycle Project

## Overview
Doctoral research project focused on membrane permeable macrocycles (BRo5 Project).
Monorepo consolidating multiple repos related to computational chemistry and cheminformatics.

---

## Repos

### 1. Macrocycle (BRo5 Project)
- **GitHub**: https://github.com/JorgeCarmona323/Macrocycle
- **Purpose**: Confirm macrocycle cyclization reactions and generate LCMS plots (UV + Mass spectra)
- **Notebooks**:
  - `LCMS_Analysis_and_Plots.ipynb` — Parses mzML files, detects peaks, plots UV chromatogram and mass spectra
  - `Dibromo_Linker_Reaction_v2.ipynb` — Builds macrocycle structures from SMILES using RDKit
  - `macrocycle_linker_reaction_v1.ipynb` — Earlier version of linker reaction notebook
- **Runtime**: Jupyter Notebook (Google Colab or local terminal)
- **Language**: Python 3
- **Key Dependencies**:
  - `rdkit` — Cheminformatics, SMILES parsing, molecule drawing
  - `pymzml` — Parsing mzML mass spectrometry data files
  - `pandas`, `numpy` — Data manipulation
  - `matplotlib` — Plotting
  - `scipy.signal` — Peak width calculations
  - `peakutils` — Peak detection
- **Data Tools**: DataWarrior (external, for cyclization confirmation)
- **Notes**:
  - Notebooks use `google.colab.drive` for file access — needs adaptation for local Linux runs
  - No `requirements.txt` yet; dependencies are installed inline via `!pip install`

### 2. Data-Cleaning-and-other-cool-beans
- **GitHub**: https://github.com/JorgeCarmona323/Data-Cleaning-and-other-cool-beans
- **Purpose**: Stash of standalone data-cleaning utilities written during grad school for
  processing molecular data, SMILES canonicalization, and dataset merging/matching.
- **Scripts**:
  | Script | What it does |
  |--------|-------------|
  | `canonicalize_amino_acids.py` | Canonicalizes SMILES strings for amino acid structures |
  | `canonicalize_macrocycles.py` | Canonicalizes cyclic SMILES; CLI: `--input` / `--output` |
  | `clean_csv_amino_acids.py` | Cleans/validates amino acid molecular structures |
  | `extract_reactant_values.py` | Extracts numeric suffixes from reactant identifiers |
  | `matching_hits_to_library.py` | Matches hit compounds against a reference library |
  | `merge_batch_to_master.py` | Merges batch enumeration CSVs into master macrocycles file |
  | `smiles_cleaner.py` | Internal module — `clean_smiles()` used by other scripts |
- **Language**: Python 3
- **Key Dependencies**:
  - `rdkit` (`Chem`, `RDLogger`, `Chem.MolStandardize`) — molecular parsing, canonicalization, standardization
  - `pandas` — CSV I/O and data manipulation (used in all scripts)
  - `argparse` — CLI argument handling (3 of 6 scripts)
  - `os` — file path operations
- **Install**: `conda install -c conda-forge rdkit pandas`
- **Patterns**:
  - All scripts are standalone (`if __name__ == "__main__"`)
  - Module-level constants for default file paths (`INPUT_FILE`, `OUTPUT_FILE`, `SMILES_COLUMN`)
  - CSV in → transform → CSV out + failure report
  - Try-except fallback for `MolStandardize` (optional dependency)
- **Notes**:
  - No shared entrypoint or package structure — flat collection of scripts
  - Intended monorepo location: `scripts/data-cleaning/`

### 3. UMAP (Chemical Space Analysis)
- **GitHub**: https://github.com/JorgeCarmona323/UMAP
- **Purpose**: UMAP-based visualization and analysis of macrocycle chemical libraries across
  three complementary chemical spaces (structural, complexity density, physicochemical)
- **Scripts**:
  | Script | What it does |
  |--------|-------------|
  | `UMAP_Figure_Macrocycles_v2_20260112.py` | Main analysis script with FPM normalization (latest) |
  | `UMAP_Figure_Macrocycles_v1_20260107.py` | Original version without FPM normalization |
  | `UMAP_Figures_Summary_20260106.py` | Summary figure generation |
  | `UMAP_Parameters_20260112.py` | Parameter configuration utilities |
- **Language**: Python 3
- **Key Dependencies**:
  - `rdkit` — SMILES parsing and molecular descriptors
  - `mapchiral` — MAPchiral fingerprinting with chirality encoding
  - `umap-learn` — UMAP dimensionality reduction
  - `numpy`, `pandas` — Data handling
  - `matplotlib` — Publication-ready SVG visualizations
  - `scikit-learn` — Preprocessing and metrics
  - `scipy` — Spearman/rank correlation for map faithfulness metrics
  - `numba`, `pynndescent` — UMAP performance dependencies
- **Install**: `pip install rdkit umap-learn numpy pandas matplotlib scikit-learn numba mapchiral pynndescent scipy`
- **Key Concepts**:
  - **Three-panel analysis**: Structural space (MAPchiral) | Complexity density (FPM-normalized) | Property space (6 descriptors)
  - **FPM normalization**: Fingerprints-per-Molecular-weight — normalizes structural complexity by MW to remove size bias; important for libraries with wide MW ranges (600–1400 Da)
  - **Hit/highlight tracking**: Distinct markers for hit compounds vs highlighted compounds
  - **Map faithfulness metrics**: Spearman, rank correlation, neighbor purity per hit
- **Config constants** (set at top of script):
  - `INPUT_CSV`, `OUTPUT_DIR`, `FIG_DIR` — file paths
  - `ENABLE_FPM_NORMALIZATION` — toggle FPM mode
- **Required CSV columns**: `Smiles`, `Source`, `Hit_ID`, `Highlight_ID`, `Total Molweight`,
  `cLogP`, `H-Acceptors`, `H-Donors`, `Polar Surface Area`, `Rotatable Bonds`
- **Outputs**:
  - CSVs: UMAP coordinates for each panel, hit neighbor lists, faithfulness metrics
  - SVGs: One publication-ready figure per panel
- **Notes**:
  - Repo is private (not publicly accessible)
  - Intended monorepo location: `analysis/umap/`
  - Script filenames include dates — versioning is date-based, not git tags

### 4. Macrocycle_Enumeration (DEL Hit Enumeration Pipeline)
- **GitHub**: https://github.com/JorgeCarmona323/Macrocycle_Enumeration
- **Purpose**: Enumerate macrocyclic peptide structures from DNA-Encoded Library (DEL) screening
  hits using building block identifiers (X1–X4) and sequential peptide coupling reactions
- **Status**: Active development — core framework exists, reaction SMARTS and coupling logic in progress
- **Scripts**:
  | File | What it does |
  |------|-------------|
  | `enumerate_macrocycles.py` | Main enumeration script — configure paths here |
  | `RESEARCH_NOTES.md` | Project overview, implementation planning, open questions |
  | `LITERATURE.md` | Literature database organized by topic with reading status |
  | `research_notes/YYYY-MM-DD.md` | Daily research logs (started 2026-01-22) |
- **Language**: Python 3
- **Key Dependencies**:
  - `rdkit` — Reaction SMARTS execution, SMILES/idcode handling, stereochemistry
  - `pandas` — Building block library loading (CSV), output writing
  - `numpy` — Numerical support
- **Install**: `pip install pandas numpy rdkit-pypi`
- **Enumeration Strategy** (sequential peptide coupling → macrocyclization):
  1. X1-COOH + X2-NH2 → X1-X2 (amide bond)
  2. X1-X2-COOH + X3-NH2 → X1-X2-X3
  3. X1-X2-X3-COOH + X4-NH2 → X1-X2-X3-X4 (linear tetrapeptide)
  4. Cyclize: X4-COOH + X1-NH2 → Macrocycle
- **Reaction SMARTS**:
  - Standard: `[C:1](=[O:2])[OH].[N:3][H]>>[C:1](=[O:2])[N:3]`
  - N-methyl: `[C:1](=[O:2])[OH].[N:3]([H])[CH3]>>[C:1](=[O:2])[N:3][CH3]`
- **Required input**: CSVs for X1–X4 building block libraries with `ID`/`Number` and `SMILES`/`Structure` columns
- **Config**: File paths set as hardcoded constants in `enumerate_macrocycles.py` (Windows-style paths — needs updating for Linux)
- **Outputs**: `enumerated_macrocycles.csv`, `enumeration_log.txt`
- **Key Features**: N-methylation support, chirality preservation, DataWarrior idcode compatibility, validation against known hits
- **Notes**:
  - Repo is private
  - Intended monorepo location: `pipelines/enumeration/`
  - Hardcoded Windows paths (`r"path\to\..."`) need converting to Linux paths on migration
  - Research notes and literature DB are part of the repo — valuable domain context

---

## Environment
- OS: Linux (WSL2)
- Primary working directory: ~/projects/Macrocycle
