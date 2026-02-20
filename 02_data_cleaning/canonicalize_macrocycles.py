## to use this code type : python "...\\canonicalize_hits.py" --input "...\\inputfile.csv" --output "...\\out.csv"

import argparse
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
try:
    from rdkit.Chem.MolStandardize import rdMolStandardize
except Exception:
    rdMolStandardize = None

RDLogger.DisableLog('rdApp.*')

INPUT_FILE = r"C:\Users\Admin\Documents\Hu Lab\Code\Python\rdkit\Scripts\Macrocycle_Enumeration\outputs\batch_enumeration_20260129_212919.csv"  # defaults; can be overridden via CLI
OUTPUT_FILE = r"C:\Users\Admin\Documents\Hu Lab\Code\Python\rdkit\Scripts\Macrocycle_Enumeration\outputs\batch_enumeration_20260129_212919_canonicalized.csv"
SMILES_COL = "Cyclic_SMILES"  # fixed column name


def canonicalize_smiles(smiles):
    """Parse, optionally standardize, and return canonical isomeric SMILES."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            Chem.SanitizeMol(mol)
    except Exception:
        return None

    if rdMolStandardize:
        try:
            lfc = rdMolStandardize.LargestFragmentChooser()
            mol = lfc.choose(mol)
            mol = rdMolStandardize.Normalizer().normalize(mol)
            mol = rdMolStandardize.Reionizer().reionize(mol)
        except Exception:
            pass

    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def canonicalize_file(input_file: str, output_file: str):
    """Load CSV, canonicalize the Cyclic_SMILES column in place, then write CSV."""
    df = pd.read_csv(input_file)

    for i, smi in df[SMILES_COL].items():
        iso = canonicalize_smiles(smi)
        df.at[i, SMILES_COL] = iso

    df.to_csv(output_file, index=False)
    print(f"Canonicalized {len(df)} rows -> {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Canonicalize the 'Cyclic_SMILES' column in a CSV file")
    parser.add_argument("--input", "-i", default=INPUT_FILE, help="Input CSV path (default: hits_raw.csv)")
    parser.add_argument("--output", "-o", default=OUTPUT_FILE, help="Output CSV path (default: hits_canonicalized.csv)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    canonicalize_file(args.input, args.output)

