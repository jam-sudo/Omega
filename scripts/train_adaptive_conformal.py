#!/usr/bin/env python3
"""Train adaptive conformal UQ model from expanded_cmax.csv.

For each drug in the calibration set, runs OmegaPipeline.simulate() to get
Cmax_pred, then stores the nonconformity score |log10(Cmax_obs / Cmax_pred)|
alongside the Morgan fingerprint.

Usage:
    python scripts/train_adaptive_conformal.py
    python scripts/train_adaptive_conformal.py --k 10 --csv data/ml/clinical/expanded_cmax.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest  # noqa: E402


def load_calibration_data(csv_path: Path) -> list[dict]:
    """Load drug SMILES, dose, and observed Cmax from expanded_cmax.csv."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles = row["smiles"].strip()
            dose_mg = float(row["dose_mg"])
            cmax_obs = float(row["cmax_mg_L"])
            drug = row["drug"].strip()

            if cmax_obs <= 0 or dose_mg <= 0:
                continue

            rows.append(
                {
                    "drug": drug,
                    "smiles": smiles,
                    "dose_mg": dose_mg,
                    "cmax_obs": cmax_obs,
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Train adaptive conformal UQ model")
    parser.add_argument(
        "--csv",
        type=str,
        default=str(repo_root / "data" / "ml" / "clinical" / "expanded_cmax.csv"),
        help="Path to calibration CSV",
    )
    parser.add_argument("--k", type=int, default=10, help="Number of nearest neighbors")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(repo_root / "models" / "corrections" / "conformal"),
        help="Output directory for calibration data",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)

    print(f"Loading calibration data from {csv_path}")
    calibration_data = load_calibration_data(csv_path)
    print(f"  Found {len(calibration_data)} drugs")

    # Initialize pipeline
    print("Initializing OmegaPipeline...")
    pipeline = OmegaPipeline()

    # Run predictions
    smiles_list = []
    cmax_obs_list = []
    cmax_pred_list = []
    failed = []

    print(f"\nRunning predictions for {len(calibration_data)} drugs...")
    print("=" * 70)

    t_start = time.perf_counter()

    for i, entry in enumerate(calibration_data):
        drug = entry["drug"]
        smiles = entry["smiles"]
        dose_mg = entry["dose_mg"]
        cmax_obs = entry["cmax_obs"]

        try:
            result = pipeline.simulate(
                SimulationRequest(
                    smiles=smiles,
                    dose_mg=dose_mg,
                    route="oral",
                    duration_h=24.0,
                )
            )
            cmax_pred = result.cmax_mg_L

            if cmax_pred <= 0:
                print(f"  [{i + 1:3d}] {drug:30s}  SKIP (pred=0)")
                failed.append(drug)
                continue

            fold_error = max(cmax_obs / cmax_pred, cmax_pred / cmax_obs)
            smiles_list.append(smiles)
            cmax_obs_list.append(cmax_obs)
            cmax_pred_list.append(cmax_pred)

            print(
                f"  [{i + 1:3d}] {drug:30s}  obs={cmax_obs:10.4f}  pred={cmax_pred:10.4f}"
                f"  FE={fold_error:6.2f}x"
            )

        except Exception as exc:
            print(f"  [{i + 1:3d}] {drug:30s}  FAILED: {exc}")
            failed.append(drug)

    elapsed = time.perf_counter() - t_start

    print("=" * 70)
    print(f"\nSuccessful: {len(smiles_list)} / {len(calibration_data)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print(f"  Failed drugs: {', '.join(failed)}")
    print(f"Total time: {elapsed:.1f}s ({elapsed / len(calibration_data) * 1000:.0f}ms/drug)")

    # Compute summary statistics
    if len(smiles_list) > 0:
        log_fe = np.abs(np.log10(np.array(cmax_obs_list) / np.array(cmax_pred_list)))
        aafe = 10 ** np.mean(log_fe)
        print(f"\nCalibration set AAFE: {aafe:.3f}")
        print(f"Mean nonconformity score: {log_fe.mean():.3f}")
        print(f"Median nonconformity score: {np.median(log_fe):.3f}")
        print(f"Max nonconformity score: {log_fe.max():.3f}")

    # Fit and save
    print(f"\nFitting AdaptiveConformal with k={args.k}...")
    from omega_pbpk.ml.corrections.adaptive_conformal import AdaptiveConformal

    ac = AdaptiveConformal(model_dir=output_dir, k=args.k)
    ac.fit(smiles_list, cmax_obs_list, cmax_pred_list)
    print(f"Saved to {output_dir}")

    # Verify round-trip
    print("\nVerifying round-trip load...")
    ac2 = AdaptiveConformal(model_dir=output_dir, k=args.k)
    assert ac2.is_fitted, "Round-trip load failed!"
    print(f"  Loaded {len(ac2._fps)} calibration molecules")

    # Demo: predict interval for a known drug (caffeine)
    caffeine_smiles = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
    result = ac2.predict_interval(caffeine_smiles, cmax_pred=1.5)
    if result:
        print("\nDemo: caffeine (pred=1.5 mg/L)")
        print(f"  90% CI: [{result['lower']:.4f}, {result['upper']:.4f}]")
        print(f"  Width: {result['width']:.2f}x")
        print(f"  Mean similarity to neighbors: {result['similarity']:.3f}")
        print(f"  Quantile q: {result['q']:.3f}")

    # Demo: predict interval for a novel molecule
    novel_smiles = "C1=CC=C(C=C1)C(=O)NCCNC(=O)C2=CC=CC=C2"
    result2 = ac2.predict_interval(novel_smiles, cmax_pred=1.0)
    if result2:
        print("\nDemo: novel molecule (pred=1.0 mg/L)")
        print(f"  90% CI: [{result2['lower']:.4f}, {result2['upper']:.4f}]")
        print(f"  Width: {result2['width']:.2f}x")
        print(f"  Mean similarity to neighbors: {result2['similarity']:.3f}")
        print(f"  Quantile q: {result2['q']:.3f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
