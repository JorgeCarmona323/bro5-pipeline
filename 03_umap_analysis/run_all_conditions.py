"""
Runner script to execute UMAP analysis for all 6 conditions
"""
import subprocess
import sys

conditions = ["A", "B", "C", "ALL", "D", "E"]

print("=" * 80)
print("Running UMAP analysis for all 6 conditions")
print("=" * 80)

for condition in conditions:
    print(f"\n{'=' * 80}")
    print(f"Starting Condition {condition}")
    print(f"{'=' * 80}\n")
    
    # Run the script with modified DATA_CONDITION
    result = subprocess.run(
        [sys.executable, "-c", f"""
import sys
sys.path.insert(0, r'C:\\Users\\Admin\\Documents\\Hu Lab\\Code\\Python\\rdkit\\Scripts\\UMAP_scripts')

# Temporarily override DATA_CONDITION
import UMAP_Figure_Macrocycles_v3_20260130 as script
script.DATA_CONDITION = "{condition}"
script.main()
"""],
        capture_output=False
    )
    
    if result.returncode != 0:
        print(f"\n⚠️  Condition {condition} failed with error code {result.returncode}")
    else:
        print(f"\n✅ Condition {condition} completed successfully")

print("\n" + "=" * 80)
print("All conditions completed!")
print("=" * 80)
