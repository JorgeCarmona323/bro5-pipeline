# Data-Cleaning-and-other-cool-beans

This repo is a stash of a bunch of one offs I made while manipulating data throughout my time in grad school.

## Data Cleaning Scripts

Collection of Python scripts for cleaning and processing molecular data, SMILES canonicalization, and data merging.

## Scripts

### canonicalize_amino_acids.py
Canonicalizes SMILES strings for amino acid structures.

### canonicalize_macrocycles.py
Canonicalizes cyclic SMILES strings for macrocycle structures. Processes the `Cyclic_SMILES` column in CSV files and outputs standardized canonical isomeric SMILES.

**Usage:**
```bash
python canonicalize_macrocycles.py
# Or with custom paths:
python canonicalize_macrocycles.py --input "path/to/input.csv" --output "path/to/output.csv"
```

### clean_csv_amino_acids.py
Cleans and processes CSV files containing amino acid data.

### extract_reactant_values.py
Extracts reactant values from molecular data files.

### matching_hits_to_library.py
Matches hit compounds to a reference library.

### merge_batch_to_master.py
Merges batch enumeration data into a master macrocycles file. Maps enumeration columns to master file structure, creates compound names from X-position values (e.g., "1-8-9-2"), and adds Source identifier.

**Usage:**
```bash
python merge_batch_to_master.py
# Or with custom paths:
python merge_batch_to_master.py --master "path/to/master.csv" --batch "path/to/batch.csv" --output "path/to/output.csv"
```

## Requirements

- Python 3.x
- pandas
- RDKit
- rdkit.Chem.MolStandardize (optional, for advanced standardization)

## Installation

```bash
conda install -c conda-forge rdkit pandas
```

## Notes

- All scripts support command-line arguments for custom file paths
- Default paths are configured in each script but can be overridden
- SMILES canonicalization uses RDKit's canonical isomeric SMILES representation
