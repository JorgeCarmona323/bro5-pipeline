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
        "data/building_blocks/Master_Building_Blocks.csv",
        "data/building_blocks/X1_Deprotected.csv",
        "data/building_blocks/X2_Deprotected.csv",
        "data/building_blocks/X3_Deprotected.csv",
        "data/building_blocks/X4_Deprotected.csv",
        "data/building_blocks/Cys_Start.csv",
        "data/building_blocks/Cys_PA_End.csv",
    ],
    "Libraries": [
        "data/libraries/2026-01-22/34_Hit_values_extracted.csv",
        "data/libraries/2026-01-29/canonicalized_master_macrocycles_2D_Descriptors_FINAL_20260129.csv",
        "data/libraries/2026-01-29/canonicalized_master_macrocycles_2D_Descriptors_20260129.csv",
        "data/libraries/2026-01-06/canonicalized_master_macrocycles_20260106.csv",
        "data/libraries/2026-01-06/canonicalized_master_macrocycles_2D_Descriptors_20260106.csv",
        "data/libraries/2025-12-18/canonicalized_6mer_library_20251218.csv",
        "data/libraries/2025-12-18/Hit_Compounds_canonicalized_20251218.csv",
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
