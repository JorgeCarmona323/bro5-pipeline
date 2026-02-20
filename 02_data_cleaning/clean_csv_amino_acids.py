# clean_amino_acids.py
import pandas as pd
from smiles_cleaner import clean_smiles

# -------------------------------------------------------
# USER CONFIGURATION
# -------------------------------------------------------
INPUT_FILE = "input.txt"      # The file you want to clean
SMILES_COL = "smiles"         # Column containing SMILES
NAME_COL = "name"             # Column with amino acid name/3-letter code
OUTPUT_FILE = "clean_output.csv"
FAILED_FILE = "failed_smiles.txt"
# -------------------------------------------------------

def clean_file(df):
    fixed = []
    failed = []

    for i, row in df.iterrows():
        raw = row[SMILES_COL]
        name = row[NAME_COL]

        cleaned = clean_smiles(raw)

        if cleaned is None:
            failed.append(raw)
            continue

        fixed.append([name, cleaned])

    # Build output dataframe
    out = pd.DataFrame(fixed, columns=["name", "smiles"])

    # Add standard fields
    out["source"] = "cleaned_import"
    out["type"] = "amino_acid"
    out["three_letter"] = out["name"]
    out["sidechain_class"] = ""
    out["pKa_sidechain"] = ""
    out["pKa_N"] = ""
    out["pKa_C"] = ""

    # Standardized output column order
    out = out[[
        "name",
        "smiles",
        "source",
        "type",
        "three_letter",
        "sidechain_class",
        "pKa_sidechain",
        "pKa_N",
        "pKa_C"
    ]]

    return out, failed


# -------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------
if __name__ == "__main__":
    # Flexible CSV/TXT reader
    df = pd.read_csv(INPUT_FILE, sep=None, engine="python")

    clean_df, failed = clean_file(df)

    clean_df.to_csv(OUTPUT_FILE, index=False)

    with open(FAILED_FILE, "w") as f:
        for s in failed:
            f.write(s + "\n")

    print("Cleaning complete.")
    print(f"Cleaned: {len(clean_df)}")
    print(f"Failed: {len(failed)}")

