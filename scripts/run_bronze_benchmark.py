#!/usr/bin/env python
"""Bronze-tier ADME property validation on 153 reference compounds.

Loads ground-truth ADME data from data/adme_reference.csv, predicts properties
using EnsembleADMEPredictor(admet_ai=False), and reports per-property AAFE,
%within-2-fold, and sample size. Results are saved to outputs/bronze_tier_results.json.
"""

import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from omega_pbpk.ml.models.adme.ensemble import EnsembleADMEPredictor  # noqa: E402

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_fold_error(pred: float, obs: float) -> float:
    if abs(pred) < 1e-12 or abs(obs) < 1e-12:
        return float("nan")
    ratio = pred / obs
    return max(ratio, 1.0 / ratio)


def compute_aafe(fold_errors: list[float]) -> float:
    valid = [fe for fe in fold_errors if not math.isnan(fe) and fe > 0]
    if not valid:
        return float("nan")
    return float(10.0 ** np.mean(np.log10(valid)))


def pct_within_2fold(fold_errors: list[float]) -> float:
    valid = [fe for fe in fold_errors if not math.isnan(fe) and fe > 0]
    if not valid:
        return float("nan")
    return 100.0 * sum(1 for fe in valid if fe <= 2.0) / len(valid)


# ---------------------------------------------------------------------------
# Property mapping: (csv_column, predictor_attribute)
# ---------------------------------------------------------------------------

PROPERTIES = [
    ("logP", "logP"),
    ("fup", "fup"),
    ("rbp", "rbp"),
    ("clint_3a4_uL_min_pmol", "clint_3a4"),
    ("peff_cm_s", "peff"),
]


def main() -> None:
    csv_path = PROJECT_ROOT / "data" / "adme_reference.csv"
    output_path = PROJECT_ROOT / "outputs" / "bronze_tier_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load reference data
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loaded {len(rows)} compounds from {csv_path.name}")

    # Initialise predictor
    predictor = EnsembleADMEPredictor(admet_ai=False)

    # Collect fold-errors per property
    fold_errors: dict[str, list[float]] = {col: [] for col, _ in PROPERTIES}
    n_success = 0
    n_fail = 0

    t0 = time.time()
    for i, row in enumerate(rows):
        smiles = row.get("smiles", "").strip()
        name = row.get("name", f"compound_{i}")
        if not smiles:
            n_fail += 1
            continue

        try:
            pred = predictor.predict(smiles)
        except Exception as e:
            print(f"  [{i+1}/{len(rows)}] FAIL {name}: {e}")
            n_fail += 1
            continue

        n_success += 1

        for csv_col, attr in PROPERTIES:
            obs_str = row.get(csv_col, "").strip()
            if not obs_str:
                continue
            try:
                obs = float(obs_str)
            except ValueError:
                continue
            if obs <= 0:
                continue

            pred_val = getattr(pred, attr, None)
            if pred_val is None or pred_val <= 0:
                continue

            fe = compute_fold_error(pred_val, obs)
            fold_errors[csv_col].append(fe)

        if (i + 1) % 25 == 0 or (i + 1) == len(rows):
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(rows)}] {elapsed:.1f}s elapsed")

    elapsed = time.time() - t0
    print(f"\nPredictions complete: {n_success} ok, {n_fail} failed, {elapsed:.1f}s total\n")

    # Compute summary metrics
    results: dict[str, dict] = {}
    print(f"{'Property':<28} {'AAFE':>6} {'%2-fold':>8} {'n':>5}")
    print("-" * 52)

    for csv_col, attr in PROPERTIES:
        fes = fold_errors[csv_col]
        aafe = compute_aafe(fes)
        pct2 = pct_within_2fold(fes)
        n = len([fe for fe in fes if not math.isnan(fe) and fe > 0])

        results[attr] = {
            "csv_column": csv_col,
            "aafe": round(aafe, 3) if not math.isnan(aafe) else None,
            "pct_within_2fold": round(pct2, 1) if not math.isnan(pct2) else None,
            "n": n,
        }

        aafe_s = f"{aafe:.3f}" if not math.isnan(aafe) else "N/A"
        pct2_s = f"{pct2:.1f}%" if not math.isnan(pct2) else "N/A"
        print(f"{csv_col:<28} {aafe_s:>6} {pct2_s:>8} {n:>5}")

    # Save
    output = {
        "tier": "bronze",
        "description": "ADME property prediction accuracy on 153 reference compounds",
        "predictor": "EnsembleADMEPredictor(admet_ai=False)",
        "n_compounds": len(rows),
        "n_success": n_success,
        "n_fail": n_fail,
        "elapsed_s": round(elapsed, 1),
        "properties": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
