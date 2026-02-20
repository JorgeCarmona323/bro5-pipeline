# Macrocycle Enumeration Pipeline

A computational chemistry workflow for enumerating macrocyclic peptide structures from building block identifiers.

## Project Overview

This project enumerates macrocyclic peptides from DNA-Encoded Library (DEL) screening hits by:
1. Taking building block identifiers (X1, X2, X3, X4)
2. Looking up structures from building block libraries
3. Performing sequential peptide coupling reactions
4. Generating final macrocycle structures

## Key Features

- **Reaction SMARTS Support**: Implements Daylight Reaction SMARTS for peptide coupling
- **N-Methylation Handling**: Supports N-methyl amino acids
- **Stereochemistry Preservation**: Maintains chirality through coupling steps
- **DataWarrior Compatible**: Handles both idcode and SMILES formats
- **Validation**: Compares enumerated structures to original hits

## Requirements

```python
pandas
numpy
rdkit
```

## Installation

```bash
cd enumeration
pip install pandas numpy rdkit-pypi
```

## Usage

### 1. Prepare Building Block Libraries

Ensure you have CSV files for X1, X2, X3, X4 with columns:
- `ID` or `Number`: Building block identifier (1, 2, 3, ...)
- `SMILES` or `Structure`: SMILES string of the building block

### 2. Configure Paths

Edit `enumerate_macrocycles.py`:
```python
# Update these paths
x1_library = r"path\to\X1_building_blocks.csv"
x2_library = r"path\to\X2_building_blocks.csv"
x3_library = r"path\to\X3_building_blocks.csv"
x4_library = r"path\to\X4_building_blocks.csv"
```

### 3. Run Enumeration

```bash
python enumerate_macrocycles.py
```

Output:
- `enumerated_macrocycles.csv`: Enumerated structures
- `enumeration_log.txt`: Detailed log

## Enumeration Strategy

### Sequential Peptide Coupling
```
X1-COOH + X2-NH2 → X1-X2 (amide bond)
X1-X2-COOH + X3-NH2 → X1-X2-X3
X1-X2-X3-COOH + X4-NH2 → X1-X2-X3-X4 (linear)
Cyclize: X4-COOH + X1-NH2 → Macrocycle
```

### Reaction SMARTS

Standard amide coupling:
```
[C:1](=[O:2])[OH].[N:3][H]>>[C:1](=[O:2])[N:3]
```

N-methyl amide coupling:
```
[C:1](=[O:2])[OH].[N:3]([H])[CH3]>>[C:1](=[O:2])[N:3][CH3]
```

## Development Status

**Current Phase:** 🚧 Active Development

### Completed
- [x] Project structure
- [x] Input/output handling
- [x] Basic enumeration framework

### In Progress
- [ ] Implement Reaction SMARTS from DataWarrior
- [ ] Building block library loading
- [ ] Sequential coupling logic
- [ ] Macrocyclization
- [ ] Validation against known structures

### To Do
- [ ] Unit tests
- [ ] Error handling
- [ ] Performance optimization
- [ ] Documentation

## Project Structure

```
Macrocycle_Enumeration/
├── enumerate_macrocycles.py    # Main enumeration script
├── README.md                    # This file
├── RESEARCH_NOTES.md           # Project overview & technical notes
├── LITERATURE.md               # Literature database
├── research_notes/             # Daily research logs
│   ├── 2026-01-22.md
│   ├── 2026-01-23.md
│   └── ...
└── .gitignore                  # Git ignore rules
```

## Research Notes

See [RESEARCH_NOTES.md](RESEARCH_NOTES.md) for:
- Project overview and technical details
- Implementation planning
- Open questions and decisions

### Daily Research Logs
- [2026-01-22](research_notes/2026-01-22.md) - Project initialization
- [2026-01-23](research_notes/2026-01-23.md) - Literature review

### Literature Database
See [LITERATURE.md](LITERATURE.md) for the complete list of papers, organized by topic with reading status and notes.

## Contributing

This is an active research project. For questions or collaboration, contact the Hu Lab.

## License

[To be determined]

## Authors

Hu Lab - Bioanalytical Chemistry Group

## Acknowledgments

- RDKit for reaction handling
- DataWarrior for building block libraries
- DEL technology platform

---

**Last Updated:** January 22, 2026
