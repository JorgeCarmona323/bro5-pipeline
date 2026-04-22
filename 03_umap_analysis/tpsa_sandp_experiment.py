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
