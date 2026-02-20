"""
Macrocycle Enumeration Script

This script enumerates macrocyclic peptides through:
1. Sequential amide coupling of building blocks (Amino-Cys → X1 → X2 → X3 → X4 → 4-PA-Cysteine)
2. Macrocyclization via o-dibromoxylene linker connecting cysteine thiols

Author: Jorge Carmona
Date: January 29, 2026
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import os
from pathlib import Path

# =============================================================================
# USER CONFIGURATION
# =============================================================================

# Resolve paths relative to this script's location
_REPO_ROOT = Path(__file__).parent.parent

# Cleaned master building blocks CSV (drop file in data/building_blocks/)
BUILDING_BLOCKS_CSV = _REPO_ROOT / "data" / "building_blocks" / "Master_Building_Blocks_Cleaned.csv"

# o-Dibromoxylene linker SMILES
LINKER_SMILES = "C1=CC=C(C(=C1)CBr)CBr"

# Output directory
OUTPUT_DIR = _REPO_ROOT / "outputs" / "csv"

# Input instruction CSV (drop file in data/libraries/)
INPUT_INSTRUCTIONS_CSV = _REPO_ROOT / "data" / "libraries" / "34_Hit_values_extracted.csv"


# =============================================================================
# INITIALIZE REACTION
# =============================================================================

# Schotten-Baumann amide formation (N-acylation to amide)
# Reference: ACS J. Chem. Inf. Model. (ci200379p_si_003.xls)
AMIDE_COUPLING_SMIRKS = "[C;$(C=O):1][OH1].[N;$(N[#6]);!$(N=*);!$([N-]);!$(N#*);!$([ND3]);!$([ND4]);!$(N[O,N]);!$(N[C,S]=[S,O,N]):2]>>[C:1][N+0:2]"

# Initialize reaction object
amide_rxn = AllChem.ReactionFromSmarts(AMIDE_COUPLING_SMIRKS)

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def load_building_blocks(csv_path):
    """
    Load building blocks from cleaned CSV file.
    
    Parameters:
    -----------
    csv_path : str - Path to Master_Building_Blocks_Cleaned.csv
    
    Returns:
    --------
    df : pandas.DataFrame - Building blocks with columns: monomer, Source_File, Smiles_Cleaned
    """
    df = pd.read_csv(csv_path)
    # Strip whitespace from monomer names
    df['monomer'] = df['monomer'].str.strip()
    return df


def get_building_block(df, monomer_name):
    """
    Retrieve SMILES for a building block by monomer name.
    
    Parameters:
    -----------
    df : pandas.DataFrame - Building blocks dataframe
    monomer_name : str - Name in the 'monomer' column
    
    Returns:
    --------
    smiles : str or None - SMILES from Smiles_Cleaned column
    """
    monomer_name = monomer_name.strip()
    matching = df[df['monomer'] == monomer_name]
    if len(matching) == 0:
        print(f"  ❌ No building block found: {monomer_name}")
        return None
    elif len(matching) > 1:
        print(f"  ⚠ Multiple matches for {monomer_name}, using first")
    return matching.iloc[0]['Smiles_Cleaned']


def couple_amide(carboxylic_acid_smiles, amine_smiles):
    """
    Couple a carboxylic acid with an amine to form an amide bond.
    
    Parameters:
    -----------
    carboxylic_acid_smiles : str - SMILES of carboxylic acid
    amine_smiles : str - SMILES of amine
    
    Returns:
    --------
    product_smiles : str or None - SMILES of amide product
    product_mol : RDKit Mol or None - Molecule object
    """
    try:
        acid_mol = Chem.MolFromSmiles(carboxylic_acid_smiles)
        amine_mol = Chem.MolFromSmiles(amine_smiles)
        
        if acid_mol is None or amine_mol is None:
            return None, None
        
        products = amide_rxn.RunReactants((acid_mol, amine_mol))
        
        if not products:
            return None, None
        
        product_mol = products[0][0]
        Chem.SanitizeMol(product_mol)
        product_smiles = Chem.MolToSmiles(product_mol)
        
        return product_smiles, product_mol
        
    except Exception as e:
        print(f"  ❌ Coupling failed: {e}")
        return None, None


def macrocyclize(pep_smi, linker_smiles):
    """
    Cyclize two peptide thiols onto a dibromo linker.
    
    Based on: https://github.com/JorgeCarmona323/Macrocycle/blob/main/Dibromo_Linker_Reaction_v2.ipynb
    
    Parameters:
    -----------
    pep_smi : str - SMILES of linear peptide with at least 2 free thiols (-SH)
    linker_smiles : str - SMILES of dibromo linker
    
    Returns:
    --------
    mol : RDKit Mol object of cyclic peptide or None if failed
    """
    pep = Chem.MolFromSmiles(pep_smi)
    linker = Chem.MolFromSmiles(linker_smiles)
    if pep is None or linker is None:
        return None
    
    # Combine peptide and linker
    combined = Chem.CombineMols(pep, linker)
    rw = Chem.RWMol(combined)
    pep_atom_count = pep.GetNumAtoms()
    
    # Find linker bromines and their neighboring carbons
    HALOGENS = {35}  # Br
    linker_carbons, halogens_to_remove = [], []
    
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() in HALOGENS and atom.GetIdx() >= pep_atom_count:
            for nb in atom.GetNeighbors():
                if nb.GetAtomicNum() == 6 and nb.GetIdx() >= pep_atom_count:
                    linker_carbons.append(nb.GetIdx())
                    halogens_to_remove.append(atom.GetIdx())
    
    linker_carbons = list(dict.fromkeys(linker_carbons))
    if len(linker_carbons) != 2:
        return None
    c1, c2 = linker_carbons
    
    # Find two peptide thiols (S with one H)
    s_idxs = [a.GetIdx() for a in rw.GetAtoms()
              if a.GetAtomicNum() == 16 and a.GetIdx() < pep_atom_count and a.GetTotalNumHs() == 1]
    if len(s_idxs) < 2:
        return None
    s1, s2 = s_idxs[:2]
    
    # Add S–C bonds and remove Br
    rw.AddBond(s1, c1, Chem.BondType.SINGLE)
    rw.AddBond(s2, c2, Chem.BondType.SINGLE)
    for idx in sorted(set(halogens_to_remove), reverse=True):
        rw.RemoveAtom(idx)
    
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol



# =============================================================================
# ENUMERATION FUNCTION
# =============================================================================

def enumerate_macrocycle(building_block_names, bb_df, linker_smiles, verbose=True):
    """
    Enumerate a macrocycle from building block names.
    
    Sequence: Amino-Cys → X1 → X2 → X3 → X4 → 4-PA-Cysteine → Cyclization
    
    Parameters:
    -----------
    building_block_names : list - List of 6 monomer names in order
    bb_df : pandas.DataFrame - Building blocks dataframe
    linker_smiles : str - Dibromo linker SMILES for cyclization
    verbose : bool - Print progress messages
    
    Returns:
    --------
    dict with keys:
        'linear_smiles' : str or None - Linear peptide SMILES
        'cyclic_smiles' : str or None - Cyclic macrocycle SMILES
        'success' : bool - Whether enumeration succeeded
        'error' : str or None - Error message if failed
    """
    result = {
        'linear_smiles': None,
        'cyclic_smiles': None,
        'success': False,
        'error': None
    }
    
    try:
        if verbose:
            print(f"\n  Building blocks: {' → '.join(building_block_names)}")
        
        # Look up all SMILES
        smiles_list = []
        for bb_name in building_block_names:
            smi = get_building_block(bb_df, bb_name)
            if smi is None:
                result['error'] = f"Building block not found: {bb_name}"
                return result
            smiles_list.append(smi)
        
        # Step 1: Couple Amino-Cys (amine) to X1 (acid)
        if verbose:
            print(f"  Step 1: {building_block_names[0]} + {building_block_names[1]}")
        current_smiles, current_mol = couple_amide(smiles_list[1], smiles_list[0])
        
        if current_smiles is None:
            result['error'] = "Step 1 coupling failed"
            return result
        
        # Steps 2-5: Add remaining building blocks
        for i in range(2, len(building_block_names)):
            if verbose:
                print(f"  Step {i}: Adding {building_block_names[i]}")
            
            current_smiles, current_mol = couple_amide(smiles_list[i], current_smiles)
            
            if current_smiles is None:
                result['error'] = f"Step {i} coupling failed"
                return result
        
        result['linear_smiles'] = current_smiles
        
        # Macrocyclization
        if verbose:
            print("  Step 6: Macrocyclization with linker")
        
        cyclic_mol = macrocyclize(current_smiles, linker_smiles)
        
        if cyclic_mol:
            result['cyclic_smiles'] = Chem.MolToSmiles(cyclic_mol, True)
            result['success'] = True
            if verbose:
                print("  ✓ Enumeration complete!")
        else:
            result['error'] = "Macrocyclization failed"
            if verbose:
                print("  ⚠ Macrocyclization failed")
        
    except Exception as e:
        result['error'] = str(e)
        if verbose:
            print(f"  ❌ Error: {e}")
    
    return result


def batch_enumerate_from_csv(input_csv, bb_df, linker_smiles, output_dir):
    """
    Batch enumerate macrocycles from input instruction CSV.
    
    Expected CSV columns:
    - Reactant_Value_1: X1 building block number (e.g., 1 for X1_1)
    - Reactant_Value_2: X2 building block number
    - Reactant_Value_3: X3 building block number
    - Reactant_Value_4: X4 building block number
    
    Parameters:
    -----------
    input_csv : str - Path to input instructions CSV
    bb_df : pandas.DataFrame - Building blocks dataframe
    linker_smiles : str - Dibromo linker SMILES
    output_dir : str - Output directory path
    
    Returns:
    --------
    results_df : pandas.DataFrame - Results with all enumerated macrocycles
    """
    print(f"\nLoading enumeration instructions from:")
    print(f"  {input_csv}")
    
    # Load input instructions
    instructions_df = pd.read_csv(input_csv)
    print(f"✓ Loaded {len(instructions_df)} enumeration tasks")
    
    # Check required columns
    required_cols = ['Reactant_Value_1', 'Reactant_Value_2', 'Reactant_Value_3', 'Reactant_Value_4']
    missing_cols = [col for col in required_cols if col not in instructions_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Enumerate all compounds
    results = []
    print(f"\nEnumerating {len(instructions_df)} macrocycles...")
    print("=" * 80)
    
    for idx, row in instructions_df.iterrows():
        print(f"\n[{idx+1}/{len(instructions_df)}] Hit {idx}")
        
        # Extract reactant values
        x1_val = int(row['Reactant_Value_1'])
        x2_val = int(row['Reactant_Value_2'])
        x3_val = int(row['Reactant_Value_3'])
        x4_val = int(row['Reactant_Value_4'])
        
        # Build sequence: Amino-Cys, X1_n, X2_n, X3_n, X4_n, 4-PA-Cysteine
        sequence = [
            'Amino-Cys',
            f'X1_{x1_val}',
            f'X2_{x2_val}',
            f'X3_{x3_val}',
            f'X4_{x4_val}',
            '4-PA-Cysteine'
        ]
        
        # Enumerate
        result = enumerate_macrocycle(sequence, bb_df, linker_smiles, verbose=True)
        
        # Store result
        results.append({
            'Hit_Index': idx,
            'X1_Value': x1_val,
            'X2_Value': x2_val,
            'X3_Value': x3_val,
            'X4_Value': x4_val,
            'Building_Block_Sequence': ' → '.join(sequence),
            'Linear_SMILES': result['linear_smiles'],
            'Cyclic_SMILES': result['cyclic_smiles'],
            'Success': result['success'],
            'Error': result['error']
        })
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Print summary
    print("\n" + "=" * 80)
    print("BATCH ENUMERATION SUMMARY")
    print("=" * 80)
    successful = results_df['Success'].sum()
    failed = len(results_df) - successful
    print(f"\n✓ Successful: {successful}/{len(results_df)}")
    if failed > 0:
        print(f"❌ Failed: {failed}/{len(results_df)}")
        print("\nFailed enumerations:")
        for idx, row in results_df[~results_df['Success']].iterrows():
            print(f"  Hit {row['Hit_Index']}: {row['Error']}")
    
    return results_df


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    
    print("=" * 80)
    print("MACROCYCLE ENUMERATION")
    print("=" * 80)
    
    # Load building blocks
    print(f"\nLoading building blocks from:")
    print(f"  {BUILDING_BLOCKS_CSV}")
    bb_df = load_building_blocks(BUILDING_BLOCKS_CSV)
    print(f"✓ Loaded {len(bb_df)} building blocks")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Run example enumeration first
    print("\n" + "=" * 80)
    print("EXAMPLE ENUMERATION")
    print("=" * 80)
    
    example_sequence = [
        'Amino-Cys',
        'X1_1',
        'X2_1',
        'X3_1',
        'X4_1',
        '4-PA-Cysteine'
    ]
    
    result = enumerate_macrocycle(example_sequence, bb_df, LINKER_SMILES, verbose=True)
    
    if result['success']:
        print("\n✓ Example enumeration successful!")
        print(f"\nLinear SMILES:\n{result['linear_smiles']}")
        print(f"\nCyclic SMILES:\n{result['cyclic_smiles']}")
    else:
        print(f"\n❌ Example enumeration failed: {result['error']}")
    
    # Run batch enumeration
    print("\n" + "=" * 80)
    print("BATCH ENUMERATION")
    print("=" * 80)
    
    results_df = batch_enumerate_from_csv(
        INPUT_INSTRUCTIONS_CSV, 
        bb_df, 
        LINKER_SMILES, 
        OUTPUT_DIR
    )
    
    # Save results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"batch_enumeration_{timestamp}.csv")
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

