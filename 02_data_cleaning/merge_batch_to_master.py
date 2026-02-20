"""
Merge batch enumeration data into the master macrocycles file.
Maps batch enumeration columns to master file structure and adds Source = "34_Hits"
"""

import argparse
import pandas as pd

# Default file paths
MASTER_FILE = r"C:\Users\Admin\Documents\Hu Lab\Code\Python\rdkit\Data\2026-01-09\canonicalized_master_macrocycles_2D_Descriptors_20260106.csv"
BATCH_FILE = r"C:\Users\Admin\Documents\Hu Lab\Code\Python\rdkit\Scripts\Macrocycle_Enumeration\outputs\batch_enumeration_20260129_212919_canonicalized.csv"
OUTPUT_FILE = r"C:\Users\Admin\Documents\Hu Lab\Code\Python\rdkit\Data\2026-01-29\canonicalized_master_macrocycles_2D_Descriptors_20260129.csv"


def merge_files(master_file, batch_file, output_file):
    """Merge batch enumeration into master file."""
    
    # Load both files
    print(f"Loading master file: {master_file}")
    master_df = pd.read_csv(master_file)
    print(f"  Master file has {len(master_df)} rows")
    
    print(f"Loading batch file: {batch_file}")
    batch_df = pd.read_csv(batch_file)
    print(f"  Batch file has {len(batch_df)} rows")
    
    # Filter only successful entries from batch
    batch_df = batch_df[batch_df['Success'] == True].copy()
    print(f"  Filtered to {len(batch_df)} successful entries")
    
    # Create new rows with master file structure
    new_rows = []
    
    for idx, row in batch_df.iterrows():
        smiles = row['Cyclic_SMILES']
        
        # Create name from X1-X4 values
        name = f"{int(row['X1_Value'])}-{int(row['X2_Value'])}-{int(row['X3_Value'])}-{int(row['X4_Value'])}"
        
        new_row = {
            'Hit_ID': name,
            'Smiles': smiles,
            'Source': '34_Hits',
            'Highlight_ID': row['Hit_Index'],
            'Structure of Smiles [idcode]': None,
            'cLogP': None,
            'cLogS': None,
            'H-Acceptors': None,
            'H-Donors': None,
            'Polar Surface Area': None,
            'Rotatable Bonds': None,
            'Total Molweight': None
        }
        
        new_rows.append(new_row)
    
    # Convert to DataFrame
    new_df = pd.DataFrame(new_rows)
    print(f"  Created {len(new_df)} new rows")
    
    # Append to master
    merged_df = pd.concat([master_df, new_df], ignore_index=True)
    print(f"  Merged file has {len(merged_df)} total rows")
    
    # Save output
    merged_df.to_csv(output_file, index=False)
    print(f"\nSaved merged file to: {output_file}")
    print(f"  Added {len(new_df)} new entries from 34_Hits")


def parse_args():
    parser = argparse.ArgumentParser(description="Merge batch enumeration into master macrocycles file")
    parser.add_argument("--master", "-m", default=MASTER_FILE, help="Master CSV file path")
    parser.add_argument("--batch", "-b", default=BATCH_FILE, help="Batch enumeration CSV file path")
    parser.add_argument("--output", "-o", default=OUTPUT_FILE, help="Output CSV file path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge_files(args.master, args.batch, args.output)
