"""
Morgan Fingerprint Bit Introspection Analysis

Provides chemical interpretability for UMAP embeddings by:
- Identifying frequent Morgan fingerprint bits in different groups
- Visualizing characteristic substructures
- Comparing hits vs background molecular features
- Analyzing 34_hits structural motifs

Run this AFTER the main UMAP script to analyze existing output CSVs.
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================================
# USER CONFIG
# ==========================================================
# Path to UMAP output CSV (choose the condition you want to analyze)
INPUT_CSV = "C:\\Users\\Admin\\Documents\\Hu Lab\\Code\\Python\\rdkit\\Scripts\\UMAP_scripts\\UMAP_runs\\2026-01-30\\Output\\structural_umap_mapchiral_D.csv"

OUTPUT_DIR = "C:\\Users\\Admin\\Documents\\Hu Lab\\Code\\Python\\rdkit\\Scripts\\UMAP_scripts\\Bit_Analysis_runs\\2026-02-03"

SMILES_COL = "Smiles"
SOURCE_COL = "Source"
HIT_ID_COL = "Hit_ID"

# Morgan fingerprint parameters
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048

# Analysis parameters
TOP_N_BITS = 20  # Number of most frequent bits to analyze per group
MIN_BIT_FREQUENCY = 0.05  # Minimum fraction of molecules that must have a bit (5%)


# ==========================================================
# Bit Analysis Functions
# ==========================================================
def generate_morgan_fingerprints_with_bitinfo(
    smiles_list: List[str]
) -> Tuple[List[np.ndarray], List[Dict], List[Chem.Mol]]:
    """
    Generate Morgan fingerprints with bit information for all molecules.
    
    Returns:
        fps: List of numpy arrays (fingerprints)
        bit_infos: List of dictionaries mapping bit → [(atom_id, radius), ...]
        mols: List of RDKit molecules
    """
    fps = []
    bit_infos = []
    mols = []
    
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(None)
            bit_infos.append(None)
            mols.append(None)
            continue
        
        bit_info = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, MORGAN_RADIUS, nBits=MORGAN_NBITS, bitInfo=bit_info
        )
        
        fps.append(np.array(fp))
        bit_infos.append(bit_info)
        mols.append(mol)
    
    return fps, bit_infos, mols


def get_bit_frequencies(fps: List[np.ndarray], valid_indices: np.ndarray) -> Dict[int, float]:
    """Calculate frequency of each bit across valid fingerprints."""
    bit_counts = defaultdict(int)
    n_valid = len(valid_indices)
    
    for idx in valid_indices:
        fp = fps[idx]
        if fp is not None:
            on_bits = np.where(fp == 1)[0]
            for bit in on_bits:
                bit_counts[bit] += 1
    
    # Convert to frequencies
    bit_frequencies = {bit: count / n_valid for bit, count in bit_counts.items()}
    return bit_frequencies


def get_discriminative_bits(
    fps: List[np.ndarray],
    group_indices: np.ndarray,
    background_indices: np.ndarray,
    top_n: int = 20
) -> List[Tuple[int, float, float, float]]:
    """
    Find bits that discriminate group from background.
    
    Returns:
        List of (bit, group_freq, background_freq, enrichment) tuples
        Enrichment = group_freq / background_freq
    """
    group_freqs = get_bit_frequencies(fps, group_indices)
    bg_freqs = get_bit_frequencies(fps, background_indices)
    
    discriminative = []
    for bit in group_freqs.keys():
        group_freq = group_freqs[bit]
        bg_freq = bg_freqs.get(bit, 0.0001)  # Avoid division by zero
        
        # Only consider bits that appear frequently in group
        if group_freq >= MIN_BIT_FREQUENCY:
            enrichment = group_freq / bg_freq
            discriminative.append((bit, group_freq, bg_freq, enrichment))
    
    # Sort by enrichment (descending)
    discriminative.sort(key=lambda x: x[3], reverse=True)
    
    return discriminative[:top_n]


def visualize_bit_substructure(
    mol: Chem.Mol,
    bit_info: Dict,
    bit: int,
    radius: int = MORGAN_RADIUS,
    img_size: Tuple[int, int] = (300, 300)
):
    """
    Visualize the substructure corresponding to a Morgan bit.
    
    Returns:
        PIL Image
    """
    if bit not in bit_info:
        return None
    
    # Get atom environment for this bit (use first occurrence)
    atom_id, bit_radius = bit_info[bit][0]
    
    # Get atom environment
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, bit_radius, atom_id)
    
    # Get atoms in this environment
    atoms_to_use = set()
    for bond_id in env:
        bond = mol.GetBondWithIdx(bond_id)
        atoms_to_use.add(bond.GetBeginAtomIdx())
        atoms_to_use.add(bond.GetEndAtomIdx())
    atoms_to_use.add(atom_id)
    
    # Draw molecule with highlighted substructure
    img = Draw.MolToImage(
        mol,
        size=img_size,
        highlightAtoms=list(atoms_to_use),
        highlightBonds=env
    )
    
    return img


# ==========================================================
# Analysis & Reporting
# ==========================================================
def analyze_group_bits(
    df: pd.DataFrame,
    fps: List[np.ndarray],
    bit_infos: List[Dict],
    mols: List[Chem.Mol],
    group_name: str,
    group_mask: np.ndarray,
    output_dir: str
):
    """
    Analyze and visualize characteristic bits for a group.
    """
    print(f"\n{'=' * 78}")
    print(f"Analyzing {group_name}")
    print(f"{'=' * 78}")
    
    group_indices = np.where(group_mask & (df[SMILES_COL].notna()))[0]
    background_indices = np.where(~group_mask & (df[SMILES_COL].notna()))[0]
    
    if len(group_indices) == 0:
        print(f"No molecules in {group_name} group. Skipping.")
        return
    
    print(f"{group_name} molecules: {len(group_indices)}")
    print(f"Background molecules: {len(background_indices)}")
    
    # Get discriminative bits
    disc_bits = get_discriminative_bits(fps, group_indices, background_indices, TOP_N_BITS)
    
    print(f"\nTop {len(disc_bits)} discriminative bits:")
    print(f"{'Bit':>6} {'Group Freq':>12} {'BG Freq':>12} {'Enrichment':>12}")
    print("-" * 48)
    for bit, gf, bf, enrich in disc_bits[:10]:
        print(f"{bit:>6} {gf:>12.3f} {bf:>12.3f} {enrich:>12.2f}x")
    
    # Visualize substructures
    group_dir = os.path.join(output_dir, group_name.replace(" ", "_"))
    os.makedirs(group_dir, exist_ok=True)
    
    print(f"\nVisualizing substructures...")
    
    # Find representative molecules for each bit
    for bit, gf, bf, enrich in disc_bits[:10]:
        # Find a molecule in the group that has this bit
        for idx in group_indices:
            if fps[idx] is not None and fps[idx][bit] == 1:
                mol = mols[idx]
                bit_info = bit_infos[idx]
                
                if mol is not None and bit in bit_info:
                    img = visualize_bit_substructure(mol, bit_info, bit)
                    if img is not None:
                        img_path = os.path.join(
                            group_dir,
                            f"bit_{bit}_enrich_{enrich:.1f}x.png"
                        )
                        img.save(img_path)
                        break
    
    # Save bit frequency data
    bit_data = []
    for bit, gf, bf, enrich in disc_bits:
        bit_data.append({
            "bit": bit,
            "group_frequency": gf,
            "background_frequency": bf,
            "enrichment": enrich
        })
    
    bit_df = pd.DataFrame(bit_data)
    csv_path = os.path.join(group_dir, f"{group_name.replace(' ', '_')}_discriminative_bits.csv")
    bit_df.to_csv(csv_path, index=False)
    print(f"Saved bit data: {csv_path}")
    
    # Plot enrichment
    fig, ax = plt.subplots(figsize=(10, 6))
    top_10 = bit_df.head(10)
    ax.barh(range(len(top_10)), top_10["enrichment"], color='steelblue')
    ax.set_yticks(range(len(top_10)))
    ax.set_yticklabels([f"Bit {b}" for b in top_10["bit"]])
    ax.set_xlabel("Enrichment (Group Freq / Background Freq)", fontsize=11)
    ax.set_title(f"Top 10 Discriminative Bits - {group_name}", fontsize=13, fontweight='bold')
    ax.axvline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='No enrichment')
    ax.legend()
    plt.tight_layout()
    
    plot_path = os.path.join(group_dir, f"{group_name.replace(' ', '_')}_enrichment.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved enrichment plot: {plot_path}")


def compare_bit_profiles(
    df: pd.DataFrame,
    fps: List[np.ndarray],
    groups: Dict[str, np.ndarray],
    output_dir: str
):
    """
    Compare bit frequency profiles across multiple groups.
    """
    print(f"\n{'=' * 78}")
    print("Comparing Bit Profiles Across Groups")
    print(f"{'=' * 78}")
    
    # Get top bits for each group
    all_top_bits = set()
    group_bit_freqs = {}
    
    for group_name, group_mask in groups.items():
        group_indices = np.where(group_mask & (df[SMILES_COL].notna()))[0]
        if len(group_indices) == 0:
            continue
        
        bit_freqs = get_bit_frequencies(fps, group_indices)
        group_bit_freqs[group_name] = bit_freqs
        
        # Get top 50 bits by frequency
        top_bits = sorted(bit_freqs.items(), key=lambda x: x[1], reverse=True)[:50]
        all_top_bits.update([bit for bit, _ in top_bits])
    
    print(f"Analyzing {len(all_top_bits)} unique top bits across groups")
    
    # Build comparison matrix
    bits_list = sorted(all_top_bits)
    comparison_data = []
    
    for bit in bits_list:
        row = {"bit": bit}
        for group_name in groups.keys():
            row[group_name] = group_bit_freqs.get(group_name, {}).get(bit, 0.0)
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save comparison
    csv_path = os.path.join(output_dir, "bit_frequency_comparison.csv")
    comparison_df.to_csv(csv_path, index=False)
    print(f"Saved comparison: {csv_path}")
    
    # Plot heatmap (top 30 bits)
    top_30_bits = comparison_df.nlargest(30, groups.keys()[0] if groups else "bit")
    
    fig, ax = plt.subplots(figsize=(10, 12))
    heatmap_data = top_30_bits.set_index("bit")[list(groups.keys())]
    
    sns.heatmap(
        heatmap_data,
        cmap="YlOrRd",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Bit Frequency"},
        ax=ax
    )
    ax.set_ylabel("Morgan Bit", fontsize=11, fontweight='bold')
    ax.set_xlabel("Group", fontsize=11, fontweight='bold')
    ax.set_title("Top 30 Bits - Frequency Comparison", fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "bit_frequency_heatmap.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved heatmap: {plot_path}")


# ==========================================================
# MAIN
# ==========================================================
def main():
    print("\n" + "=" * 78)
    print("Morgan Fingerprint Bit Introspection Analysis")
    print("=" * 78)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    print(f"\nLoading data from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} compounds")
    
    # Check required columns
    required_cols = [SMILES_COL, SOURCE_COL, HIT_ID_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    # Generate Morgan fingerprints
    print(f"\nGenerating Morgan fingerprints (radius={MORGAN_RADIUS}, nBits={MORGAN_NBITS})...")
    fps, bit_infos, mols = generate_morgan_fingerprints_with_bitinfo(df[SMILES_COL].tolist())
    
    valid_count = sum(1 for fp in fps if fp is not None)
    print(f"Successfully generated {valid_count} fingerprints")
    
    # Define groups for analysis
    df["_is_hit"] = df[HIT_ID_COL].notna() & (df[HIT_ID_COL].astype(str).str.strip() != "")
    df["_is_34hit"] = df[SOURCE_COL].str.lower() == "34_hits"
    df["_is_literature"] = df[SOURCE_COL].str.lower() == "literature"
    df["_is_library"] = df[SOURCE_COL].str.lower() == "library"
    
    # Analyze each group
    groups_to_analyze = {
        "Hits": df["_is_hit"].values,
        "34_Hits": df["_is_34hit"].values,
        "Literature": df["_is_literature"].values,
        "Library": df["_is_library"].values,
    }
    
    for group_name, group_mask in groups_to_analyze.items():
        analyze_group_bits(df, fps, bit_infos, mols, group_name, group_mask, OUTPUT_DIR)
    
    # Compare profiles across groups
    compare_bit_profiles(df, fps, groups_to_analyze, OUTPUT_DIR)
    
    print("\n" + "=" * 78)
    print("✅ Bit introspection analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 78)


if __name__ == "__main__":
    main()
