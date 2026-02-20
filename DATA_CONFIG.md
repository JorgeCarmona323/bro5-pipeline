# Data Configuration

Data files are **not committed to git** (too large, contains raw research data).
Drop files into the appropriate folders below after copying from Windows.

## Folder Map

| Folder | What goes here | File types |
|--------|---------------|------------|
| `data/building_blocks/` | X1, X2, X3, X4 building block libraries | CSV |
| `data/lcms/` | Raw LCMS instrument output | mzML |
| `data/libraries/` | Master macrocycle library CSVs | CSV |

## Copying from Windows to WSL2

From your Windows filesystem, copy files into WSL2:

```bash
# Example — adjust paths to match your Windows file locations
cp /mnt/c/Users/YourName/path/to/X1_building_blocks.csv data/building_blocks/
cp /mnt/c/Users/YourName/path/to/X2_building_blocks.csv data/building_blocks/
cp /mnt/c/Users/YourName/path/to/X3_building_blocks.csv data/building_blocks/
cp /mnt/c/Users/YourName/path/to/X4_building_blocks.csv data/building_blocks/

cp /mnt/c/Users/YourName/path/to/*.mzML data/lcms/
cp /mnt/c/Users/YourName/path/to/master_library.csv data/libraries/
```

## Script Path Configuration

Each script uses a `DATA_ROOT` or hardcoded path — update these when running locally:

| Script | Variable to update |
|--------|--------------------|
| `01_enumeration/enumerate_macrocycles.py` | `x1_library`, `x2_library`, `x3_library`, `x4_library` |
| `03_umap_analysis/UMAP_Figure_Macrocycles_v2_*.py` | `INPUT_CSV`, `OUTPUT_DIR`, `FIG_DIR` |
| `04_validation/LCMS_Analysis_and_Plots.ipynb` | First cell path variables |
