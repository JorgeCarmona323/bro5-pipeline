"""
update_mordred_mcs80.py
=======================
Re-run HDBSCAN on the existing 10D Mordred UMAP embedding with mcs=80
(selected from fine-grained sweep: 35 clusters, DBCV=+0.511, sil=+0.768).

Updates hdbscan_mordred and hdbscan_mordred_prob in aligned_metadata.csv,
then regenerates the combined figures.
"""

import subprocess
import warnings
from pathlib import Path

import hdbscan as hdbscan_pkg
from hdbscan import validity as hdbscan_validity
import numpy as np
import pandas as pd

ROOT          = Path(__file__).resolve().parent.parent
MORDRED_CSV   = ROOT / "outputs/analysis/2026-04-06/mordred_umap.csv"
ALIGNED_CSV   = ROOT / "outputs/analysis/2026-04-06/aligned_metadata.csv"
MAKE_FIGS     = ROOT / "03_umap_analysis/make_combined_figures.py"

MCS           = 80
MIN_SAMPLES   = 10
ND_COLS       = [f"UMAP_{i}_nd" for i in range(1, 11)]


def main():
    print(f"[Load] {MORDRED_CSV.name}")
    mordred = pd.read_csv(MORDRED_CSV)
    X = mordred[ND_COLS].values.astype(np.float64)
    print(f"  {X.shape[0]:,} × {X.shape[1]}D embedding")

    print(f"\n[HDBSCAN] mcs={MCS}  min_samples={MIN_SAMPLES}")
    clusterer = hdbscan_pkg.HDBSCAN(
        min_cluster_size = MCS,
        min_samples      = MIN_SAMPLES,
        metric           = "euclidean",
    )
    clusterer.fit(X)
    labels = clusterer.labels_
    probs  = clusterer.probabilities_

    n_clusters  = len(set(labels)) - (1 if -1 in labels else 0)
    noise_frac  = (labels == -1).mean()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            dbcv = hdbscan_validity.validity_index(X, labels)
        except Exception:
            dbcv = float("nan")

    print(f"  clusters={n_clusters}  noise={noise_frac:.3f}  DBCV={dbcv:+.3f}")

    print(f"\n[Update] {ALIGNED_CSV.name}")
    aligned = pd.read_csv(ALIGNED_CSV)

    # Align by row order — both CSVs were built from the same common_smiles ordering
    if len(aligned) != len(labels):
        raise ValueError(
            f"Row count mismatch: aligned_metadata={len(aligned)}, "
            f"new labels={len(labels)}"
        )

    aligned["hdbscan_mordred"]      = labels
    aligned["hdbscan_mordred_prob"] = probs
    aligned.to_csv(ALIGNED_CSV, index=False)
    print(f"  Saved hdbscan_mordred (mcs={MCS})")

    print("\n[Figures] Regenerating combined figures ...")
    result = subprocess.run(
        ["python", str(MAKE_FIGS)],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        raise RuntimeError("make_combined_figures.py failed")

    print(f"\n[✓] Done — mordred HDBSCAN updated to mcs={MCS}")
    print(f"     clusters={n_clusters}  noise={noise_frac:.3f}  DBCV={dbcv:+.3f}")


if __name__ == "__main__":
    main()
