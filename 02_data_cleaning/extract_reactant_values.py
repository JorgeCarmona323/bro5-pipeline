import pandas as pd
import os

# Load your CSV file
input_file = r"C:\Users\Admin\Documents\Hu Lab\DataWarrior\Cyclized_Hits_DEL18\DEL18_Cyclized Hits.csv"
df = pd.read_csv(input_file)

print(f"Processing {len(df)} hits from the input file...")

# Extract the numeric value after underscore for each Reactant-ID column
for i in range(1, 5):
    col_name = f"Reactant-ID {i}"
    new_col_name = f"Reactant_Value_{i}"
    
    if col_name in df.columns:
        # Extract the number after the underscore (e.g., X1_1 -> 1, X2_8 -> 8)
        df[new_col_name] = df[col_name].str.split('_').str[-1]

# Save to new CSV with the extracted values
output_file = r"C:\Users\Admin\Documents\Hu Lab\Code\Python\rdkit\Data\2026-01-22\34_Hit_values_extracted.csv"

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(output_file), exist_ok=True)

df.to_csv(output_file, index=False)
print(f"Extraction complete! Processed all {len(df)} hits.")
print(f"Saved to: {output_file}")

# Display all rows of extracted values
print("\nAll extracted reactant values:")
print(df[["Reactant-ID 1", "Reactant_Value_1", 
         "Reactant-ID 2", "Reactant_Value_2",
         "Reactant-ID 3", "Reactant_Value_3",
         "Reactant-ID 4", "Reactant_Value_4"]].to_string())

# Save summary to log file
with open("extraction_log.txt", "w") as log:
    log.write("Reactant Value Extraction Summary\n")
    log.write("=" * 50 + "\n\n")
    log.write(f"Input file: {input_file}\n")
    log.write(f"Output file: {output_file}\n")
    log.write(f"Total rows processed: {len(df)}\n\n")
    log.write("ALL EXTRACTED DATA:\n")
    log.write("=" * 50 + "\n")
    log.write(df[["Reactant-ID 1", "Reactant_Value_1", 
                  "Reactant-ID 2", "Reactant_Value_2",
                  "Reactant-ID 3", "Reactant_Value_3",
                  "Reactant-ID 4", "Reactant_Value_4"]].to_string())

print("\nLog saved to: extraction_log.txt")
