# 05 Enumeration Analysis (R)

R scripts for statistical analysis and visualization of enumerated macrocycle libraries across different linker chemistries.

## Scripts

| Script | Purpose |
|--------|---------|
| `01_data_prep.R` | Reads per-linker descriptor CSVs, tags each by library name, and combines into a single `all_libs` data frame |
| `02_analysis.R` | Univariate summaries, pairwise statistical tests (t-test, Wilcoxon, KS), and Cohen's d effect sizes |
| `03_plots.R` | Histograms, raincloud plots, and PCA scatter for all descriptors across libraries |
| `04_compare_all_effects.R` | Aggregates effect-size tables from multiple comparisons into a grouped bar chart |
| `05_compare_original.R` | Monomer vs. Dipeptide effect-size comparison and Mahalanobis distance density plot |

## Descriptors Analyzed

- Total Molecular Weight
- cLogP / aLogP
- cLogS
- H-bond Acceptors / Donors
- Rotatable Bonds
- Polar Surface Area (TPSA)

## Dependencies

- `tidyverse`, `broom`, `purrr`
- `ggthemes`, `patchwork`, `cowplot`, `RColorBrewer`

## Usage

Scripts are numbered and intended to run sequentially. `01_data_prep.R` must run first to produce the combined CSV consumed by downstream scripts.

> **Note:** File paths in these scripts reference local data directories and may need to be updated for your environment.
