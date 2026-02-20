"""
check_data.py
Validates that all expected data files are present in the data/ directory.
Run this after copying files from Windows to confirm everything is in place.

Usage:
    python check_data.py
"""

from pathlib import Path

REPO_ROOT = Path(__file__).parent

EXPECTED_FILES = {
    "Building Blocks": [
        "data/building_blocks/Master_Building_Blocks_Cleaned.csv",
    ],
    "Libraries": [
        "data/libraries/2026-01-22/34_Hit_values_extracted.csv",
        "data/libraries/2026-01-29/Smiles.smi",
    ],
    "LCMS UV (.txt)": [
        "data/lcms/uv/Brain-Ala-4-4-13_PURE_20251114.txt",
    ],
    "LCMS Mass Spec (.mzML)": [
        "data/lcms/ms/X1_MS_RAW.mzML",
    ],
}


def check():
    print("\n" + "=" * 60)
    print("  BRo5 PIPELINE — DATA FILE CHECK")
    print("=" * 60)

    total = 0
    found = 0

    for section, files in EXPECTED_FILES.items():
        print(f"\n{section}")
        print("-" * 40)
        for f in files:
            path = REPO_ROOT / f
            total += 1
            if path.exists():
                found += 1
                print(f"  ✅ {f}")
            else:
                print(f"  ❌ {f}  ← MISSING")

    print("\n" + "=" * 60)
    print(f"  {found}/{total} files present")
    if found == total:
        print("  All data files accounted for. Pipeline is ready to run.")
    else:
        missing = total - found
        print(f"  {missing} file(s) still need to be copied from Windows.")
        print("  See DATA_CONFIG.md for copy instructions.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    check()
