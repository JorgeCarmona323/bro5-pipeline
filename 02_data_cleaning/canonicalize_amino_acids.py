# smiles_cleaner.py
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import MolStandardize

# Silence RDKit warnings
RDLogger.DisableLog('rdApp.*')

# -------------------------------------------------------
# Strict RDKit parse
# -------------------------------------------------------
def rdkit_clean(smiles):
    """Strict RDKit parse → canonical isomeric SMILES."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    except Exception:
        pass
    return None

# -------------------------------------------------------
# Permissive RDKit parse
# -------------------------------------------------------
def rdkit_clean_fallback(smiles):
    """Permissive RDKit parse (sanitize=False) + sanitize."""
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol:
            Chem.SanitizeMol(mol)
            return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    except Exception:
        pass
    return None

# -------------------------------------------------------
# Standardization (fragment → normalize → reionize)
# -------------------------------------------------------
def normalize_molecule(mol):
    """Standardize molecule components using Cleanup."""
    if mol is None:
        return None
    try:
        # Use Cleanup() which does fragment removal, normalization, and reionization
        return MolStandardize.Cleanup(mol)
    except Exception:
        return mol

# -------------------------------------------------------
# Neutralization (simple amino acid zwitterions)
# -------------------------------------------------------
def neutralize(smiles):
    """Simple zwitterion neutralizer."""
    return smiles.replace("[NH3+]", "NH").replace("[O-]", "O")

# -------------------------------------------------------
# Full cleaning pipeline
# -------------------------------------------------------
def clean_smiles(smiles):
    """Neutralize → strict parse → fallback parse → normalize → canonicalize."""
    smiles = neutralize(smiles)

    for fixer in (rdkit_clean, rdkit_clean_fallback):
        fixed = fixer(smiles)
        if fixed:
            mol = Chem.MolFromSmiles(fixed)
            mol = normalize_molecule(mol)
            if mol:
                return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            return fixed

    return None


# -------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------
if __name__ == "__main__":
    # Input file path
    INPUT_FILE = r"C:\Users\Admin\Documents\Hu Lab\Code\Python\rdkit\Scripts\Macrocycle_Enumeration\Macrocycle_Enumeration_Inputs\Enumeration_Deprotected_Data_20260129\Master_Building_Blocks.csv"
    OUTPUT_FILE = r"C:\Users\Admin\Documents\Hu Lab\Code\Python\rdkit\Scripts\Macrocycle_Enumeration\Macrocycle_Enumeration_Inputs\Enumeration_Deprotected_Data_20260129\Master_Building_Blocks_Cleaned.csv"
    FAILED_FILE = r"C:\Users\Admin\Documents\Hu Lab\Code\Python\rdkit\Scripts\Macrocycle_Enumeration\Macrocycle_Enumeration_Inputs\Enumeration_Deprotected_Data_20260129\failed_smiles.txt"
    
    print("Reading input file...")
    df = pd.read_csv(INPUT_FILE)
    
    print(f"Processing {len(df)} rows...")
    
    # Clean the Smiles_Deprotected column
    cleaned_smiles = []
    failed_smiles = []
    
    for idx, row in df.iterrows():
        original_smiles = row['Smiles_Deprotected']
        
        # Skip if NaN or empty
        if pd.isna(original_smiles) or str(original_smiles).strip() == '':
            cleaned_smiles.append(None)
            failed_smiles.append(f"Row {idx}: Empty or NaN")
            continue
        
        cleaned = clean_smiles(str(original_smiles))
        
        if cleaned is None:
            failed_smiles.append(f"Row {idx}: {original_smiles}")
            cleaned_smiles.append(None)
        else:
            cleaned_smiles.append(cleaned)
    
    # Add cleaned SMILES as new column
    df['Smiles_Cleaned'] = cleaned_smiles
    
    # Save results
    df.to_csv(OUTPUT_FILE, index=False)
    
    # Save failed SMILES
    if failed_smiles:
        with open(FAILED_FILE, 'w') as f:
            f.write('\n'.join(failed_smiles))
    
    print(f"\nCleaning complete!")
    print(f"Total rows: {len(df)}")
    print(f"Successfully cleaned: {sum(1 for x in cleaned_smiles if x is not None)}")
    print(f"Failed: {len(failed_smiles)}")
    print(f"\nOutput saved to: {OUTPUT_FILE}")
    if failed_smiles:
        print(f"Failed SMILES saved to: {FAILED_FILE}")

