# TPSA Discrepancy: RDKit vs DataWarrior — S/P Inclusion Experiment
**Date:** 2026-04-21 | **Status:** Resolved | **Script:** `03_umap_analysis/tpsa_sandp_experiment.py`

---

## Background

During review of hit compound physicochemical properties, PSA values calculated in **DataWarrior** were consistently higher than those computed by **RDKit** for the same SMILES. The discrepancy was traced to a difference in algorithm scope:

- **DataWarrior** uses Ertl's 2000 algorithm, which includes contributions from **S and P atoms**
- **RDKit `CalcTPSA()`** excludes S and P by default (N and O only)
- **Fix:** pass `includeSandP=True` to `rdMolDescriptors.CalcTPSA()`

---

## Test Set

Four hit macrocycles from the DEL screen, chosen because they share the same Cys-based scaffold (two thioether bridges) but differ in their variable positions:

| Compound | X1–X4 | Key S features |
|----------|--------|----------------|
| Cmp1 | DOPC 2-9-9-8 | Thioether bridges only |
| Cmp2 | DOPC 3-12-8-12 | Thiophene ring + thioether bridges |
| Cmp3 | DOPC 1-6-4-7 | Sulfonyloxy (–SO₃H) + thioether bridges |
| Cmp4 | Brain 6-4-4-13 | Thioether bridges only |

---

## Results

| Compound | RDKit default (Å²) | RDKit +S/P (Å²) | DataWarrior (Å²) | Δ default vs DW | Δ +S/P vs DW |
|----------|--------------------|-----------------|------------------|-----------------|--------------|
| Cmp1 (2-9-9-8) | 213.25 | 263.85 | 263.85 | −50.60 | **0.00** |
| Cmp2 (3-12-8-12) | 249.36 | 328.20 | 328.20 | −78.84 | **0.00** |
| Cmp3 (1-6-4-7) | 283.50 | 342.48 | 342.48 | −58.98 | **0.00** |
| Cmp4 (Brain 6-4-4-13) | 229.13 | 279.73 | 279.73 | −50.60 | **0.00** |

`includeSandP=True` achieves **exact parity with DataWarrior for all 4 compounds.**

---

## S/P Contribution Breakdown

| Compound | S contribution (Å²) | Source |
|----------|---------------------|--------|
| Cmp1 | +50.60 | 2 thioether bridges |
| Cmp2 | +78.84 | 2 thioether bridges + thiophene ring (+28.24) |
| Cmp3 | +58.98 | 2 thioether bridges + sulfonyloxy (+8.38) |
| Cmp4 | +50.60 | 2 thioether bridges |

> The shared macrocycle scaffold contributes a baseline **+50.60 Å²** from the two `–CSC–` thioether bridges present in every compound. Additional S-containing substituents add on top of this.

---

## Interpretation

The discrepancy is **systematic and significant** — up to ~79 Å² underestimation with the default RDKit setting. For macrocycles with sulfur-rich scaffolds like these Cys-based DEL compounds, the default RDKit TPSA would incorrectly classify them as more permeable than they are, potentially misleading permeability predictions or rule-of-thumb filters (e.g., Veber's rule: PSA < 140 Å²).

---

## Fix

Wherever TPSA is computed in the pipeline, replace:

```python
# Before (incorrect for S/P-containing compounds)
tpsa = rdMolDescriptors.CalcTPSA(mol)

# After (matches DataWarrior / Ertl 2000)
tpsa = rdMolDescriptors.CalcTPSA(mol, includeSandP=True)
```

---

## Full Experiment Code

`03_umap_analysis/tpsa_sandp_experiment.py`

```python
#!/usr/bin/env python3
"""
Mini-experiment: compare TPSA calculations across methods for 4 hit compounds.

Background: DataWarrior uses Ertl's 2000 algorithm which includes S and P atom
contributions to PSA. RDKit's CalcTPSA() excludes S and P by default; passing
includeSandP=True restores parity with DataWarrior for compounds containing
sulfur (thioether bridges, sulfonyl groups, thiophene rings).
"""

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import pandas as pd

COMPOUNDS = [
    {
        "name": "Cmp1 (DOPC 2-9-9-8)",
        "smiles": "C#CCCC(=O)N[C@H]1CSCc2ccccc2CSC[C@@H](C(N)=O)NC(=O)C[C@H](c2ccco2)NC(=O)[C@H](CC(C)C)N(C)C(=O)CNC(=O)CN(C)C1=O",
        "dw_psa": 263.85,
        "s_features": "thioether bridges only",
    },
    {
        "name": "Cmp2 (DOPC 3-12-8-12)",
        "smiles": "C#CCCC(=O)N[C@H]1CSCc2ccccc2CSC[C@@H](C(N)=O)NC(=O)[C@H](c2cccs2)NC(=O)[C@H](CO)NC(=O)[C@@H]2CCN2C(=O)[C@H](CO)NC1=O",
        "dw_psa": 328.20,
        "s_features": "thiophene ring + thioether bridges",
    },
    {
        "name": "Cmp3 (DOPC 1-6-4-7)",
        "smiles": "C#CCCC(=O)N[C@H]1CSCc2ccccc2CSC[C@@H](C(N)=O)NC(=O)[C@H](CS(=O)(=O)O)NC(=O)[C@@H]([C@H](C)O)NC(=O)[C@@H]2CCCCN2C(=O)CNC1=O",
        "dw_psa": 342.48,
        "s_features": "sulfonyloxy (-SO3H) + thioether bridges",
    },
    {
        "name": "Cmp4 (Brain 6-4-4-13)",
        "smiles": "C#CCCC(=O)N[C@H]1CSCc2ccccc2CSC[C@@H](C(N)=O)NC(=O)C[C@@H](c2cccc3ccccc23)NC(=O)[C@H](C)NC(=O)[C@@H]2CCCCN2C(=O)[C@@H](CO)NC1=O",
        "dw_psa": 279.73,
        "s_features": "thioether bridges only",
    },
]


def calc_tpsa(smiles, include_sp=False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return rdMolDescriptors.CalcTPSA(mol, includeSandP=include_sp)


def main():
    rows = []
    for cmp in COMPOUNDS:
        tpsa_default  = calc_tpsa(cmp["smiles"], include_sp=False)
        tpsa_with_sp  = calc_tpsa(cmp["smiles"], include_sp=True)
        dw            = cmp["dw_psa"]

        delta_default = tpsa_default - dw
        delta_with_sp = tpsa_with_sp - dw

        rows.append({
            "Compound":              cmp["name"],
            "S features":            cmp["s_features"],
            "RDKit default (Å²)":    round(tpsa_default, 2),
            "RDKit +S/P (Å²)":       round(tpsa_with_sp, 2),
            "DataWarrior (Å²)":      dw,
            "Δ default vs DW":       round(delta_default, 2),
            "Δ +S/P vs DW":          round(delta_with_sp, 2),
        })

    df = pd.DataFrame(rows)

    print("=" * 90)
    print("TPSA Comparison: RDKit default vs includeSandP=True vs DataWarrior")
    print("=" * 90)
    print(df.to_string(index=False))
    print()
    print("S/P contribution per compound (Å²):")
    for row in rows:
        sp_contrib = row["RDKit +S/P (Å²)"] - row["RDKit default (Å²)"]
        print(f"  {row['Compound']}: +{sp_contrib:.2f}  ({row['S features']})")

    print()
    print("Conclusion:")
    close_default = [r for r in rows if abs(r["Δ default vs DW"]) < 5]
    close_sp      = [r for r in rows if abs(r["Δ +S/P vs DW"]) < 5]
    print(f"  RDKit default within 5 Å² of DataWarrior: {len(close_default)}/4 compounds")
    print(f"  RDKit +S/P  within 5 Å² of DataWarrior: {len(close_sp)}/4 compounds")


if __name__ == "__main__":
    main()
```

---

## Scope of Impact

| Pipeline step | Uses TPSA? | Action needed |
|---------------|-----------|---------------|
| 2D Physicochemical UMAP branch | Yes — PSA is one of 6 input features | Recompute with `includeSandP=True` if re-running |
| Mordred descriptors (`compute_mordred_descriptors.py`) | Mordred has its own TPSA descriptor (`TopoPSA`) | Check Mordred's default — separate investigation |
| DataWarrior `.dwar` files | Reference values already correct | No action |

---

## References

- Ertl, P. et al. *Fast calculation of molecular polar surface area as a sum of fragment-based contributions and its application to the prediction of drug transport properties.* J. Med. Chem. 2000, 43, 3714–3717.
- RDKit docs: [`rdMolDescriptors.CalcTPSA`](https://www.rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html)
