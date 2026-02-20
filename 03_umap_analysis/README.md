# UMAP Analysis for Macrocycle Chemical Space

A comprehensive UMAP-based analysis toolkit for visualizing and analyzing macrocycle chemical libraries with support for:
- **Structural similarity** (MAPchiral fingerprints)
- **Complexity density analysis** (FPM-normalized fingerprints)
- **Physicochemical property space** (descriptor-based)

## Features

### Three-Panel Analysis
1. **Structural Space (Panel 1)**: Standard MAPchiral fingerprint UMAP showing structural similarity
2. **Complexity Density Space (Panel 2)**: FPM-normalized fingerprint UMAP accounting for molecular weight diversity
3. **Property Space (Panel 3)**: Descriptor-based UMAP for physicochemical similarity

### Key Capabilities
- MAPchiral fingerprinting with chirality encoding
- FPM (Fingerprints per Molecular weight) normalization for size-independent complexity analysis
- Safe deduplication preserving hits and highlights
- Local map faithfulness metrics (Spearman, rank correlation, neighbor purity)
- Publication-ready SVG visualizations
- Hit and highlight tracking with distinct markers

## Scripts

- `UMAP_Figure_Macrocycles_v2_20260112.py` - Main analysis script with FPM normalization (latest)
- `UMAP_Figure_Macrocycles_v1_20260107.py` - Original version without FPM
- `UMAP_Figures_Summary_20260106.py` - Summary figure generation
- `UMAP_Parameters_20260112.py` - Parameter configuration utilities

## Requirements

```python
rdkit
umap-learn
numpy
pandas
matplotlib
scikit-learn
numba
mapchiral
pynndescent
scipy
```

## Usage

```python
# Edit configuration section in script
INPUT_CSV = "path/to/your/data.csv"
OUTPUT_DIR = "path/to/output"
FIG_DIR = "path/to/figures"

# Enable/disable FPM normalization
ENABLE_FPM_NORMALIZATION = True

# Run analysis
python UMAP_Figure_Macrocycles_v2_20260112.py
```

## Input Data Format

CSV file with required columns:
- `Smiles`: SMILES strings
- `Source`: Data source (literature/library)
- `Hit_ID`: Hit identifier (empty for non-hits)
- `Highlight_ID`: Highlight identifier (empty for non-highlights)
- `Total Molweight`: Molecular weight
- `cLogP`, `H-Acceptors`, `H-Donors`, `Polar Surface Area`, `Rotatable Bonds`: Descriptors

## Outputs

### CSV Files
- `structural_umap_mapchiral.csv` - Standard fingerprint UMAP coordinates
- `structural_umap_fpm_normalized.csv` - FPM-normalized UMAP coordinates (includes FPM values)
- `descriptor_umap_6desc.csv` - Descriptor UMAP coordinates
- `hit_neighbors_fingerprint_space.csv` - Nearest neighbors for each hit
- `hit_local_map_faithfulness.csv` - Faithfulness metrics per hit

### Visualizations (SVG)
- `structural_umap_mapchiral_hits_highlights.svg` - Panel 1
- `structural_umap_fpm_normalized_hits_highlights.svg` - Panel 2
- `descriptor_umap_6desc_hits_highlights.svg` - Panel 3

## FPM Normalization

FPM (Fingerprints per Molecular weight) normalization addresses the inherent bias where larger molecules naturally have more structural features. By normalizing by molecular weight, we can compare structural complexity density across molecules with diverse sizes.

**Key insight**: Two molecules with similar MW but different FPM values indicate different structural complexity:
- Higher FPM = more structurally dense/complex (branched, cyclic, heteroatom-rich)
- Lower FPM = simpler structure (linear chains, fewer functional groups)

This is particularly important for libraries with wide MW ranges (e.g., 600-1400 Da macrocycles).

## Citation

If you use this code, please cite the relevant papers for:
- MAPchiral: [relevant citation]
- UMAP: McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv:1802.03426

## License

[Specify your license]

## Contact

[Your contact information]
