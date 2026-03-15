#!/usr/bin/env python3
"""Silver-tier validation: compare predicted t_half against OpenFDA values."""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("silver_benchmark")
logger.setLevel(logging.INFO)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

CSV_PATH = REPO_ROOT / "data" / "ml" / "clinical" / "openfda_validation.csv"
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_PATH = OUTPUT_DIR / "silver_tier_results.json"


def load_drugs(csv_path: Path) -> list[dict]:
    """Load drugs with t_half data and valid SMILES from the CSV."""
    drugs = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            has_thalf = row.get("has_thalf", "").strip()
            smiles = row.get("smiles", "").strip()
            if has_thalf not in ("1", "True", "true"):
                continue
            if not smiles:
                continue
            thalf_h = row.get("thalf_h", "").strip()
            if not thalf_h:
                continue
            drugs.append(
                {
                    "drug_name": row["drug_name"].strip(),
                    "smiles": smiles,
                    "obs_thalf_h": float(thalf_h),
                }
            )
    return drugs


def main() -> None:
    from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

    drugs = load_drugs(CSV_PATH)
    logger.info("Loaded %d drugs with t_half data from %s", len(drugs), CSV_PATH)

    pipeline = OmegaPipeline()

    results = []
    fold_errors = []
    t_start_total = time.time()

    for i, drug in enumerate(drugs):
        name = drug["drug_name"]
        smiles = drug["smiles"]
        obs = drug["obs_thalf_h"]

        # Simulate long enough to capture full elimination
        duration_h = max(obs * 5.0, 24.0)
        n_timepoints = max(241, int(duration_h * 10) + 1)

        request = SimulationRequest(
            smiles=smiles,
            dose_mg=100.0,
            route="oral",
            duration_h=duration_h,
            n_timepoints=n_timepoints,
        )

        t0 = time.time()
        try:
            sim = pipeline.simulate(request)
            pred = sim.t_half_h
            elapsed = time.time() - t0

            fe = max(pred / obs, obs / pred) if pred > 0 and obs > 0 else float("inf")
            within_2fold = fe <= 2.0

            fold_errors.append(fe)
            results.append(
                {
                    "drug_name": name,
                    "smiles": smiles,
                    "obs_thalf_h": obs,
                    "pred_thalf_h": round(pred, 3),
                    "fold_error": round(fe, 3),
                    "within_2fold": within_2fold,
                    "elapsed_s": round(elapsed, 2),
                }
            )

            status = "OK" if within_2fold else "MISS"
            logger.info(
                "[%2d/%d] %-20s obs=%6.1fh pred=%6.1fh FE=%5.2f %s (%.1fs)",
                i + 1,
                len(drugs),
                name,
                obs,
                pred,
                fe,
                status,
                elapsed,
            )
        except Exception as exc:
            elapsed = time.time() - t0
            logger.warning(
                "[%2d/%d] %-20s FAILED: %s (%.1fs)", i + 1, len(drugs), name, exc, elapsed
            )
            results.append(
                {
                    "drug_name": name,
                    "smiles": smiles,
                    "obs_thalf_h": obs,
                    "pred_thalf_h": None,
                    "fold_error": None,
                    "within_2fold": False,
                    "elapsed_s": round(elapsed, 2),
                    "error": str(exc),
                }
            )

    total_time = time.time() - t_start_total

    # Compute aggregate metrics
    valid_fe = [fe for fe in fold_errors if fe != float("inf")]
    n_valid = len(valid_fe)
    n_within_2fold = sum(1 for fe in valid_fe if fe <= 2.0)

    if valid_fe:
        aafe = float(10 ** np.mean(np.log10(valid_fe)))
        pct_2fold = 100.0 * n_within_2fold / n_valid
    else:
        aafe = float("nan")
        pct_2fold = 0.0

    summary = {
        "n_drugs_total": len(drugs),
        "n_drugs_valid": n_valid,
        "n_within_2fold": n_within_2fold,
        "pct_within_2fold": round(pct_2fold, 1),
        "AAFE": round(aafe, 3),
        "total_time_s": round(total_time, 1),
    }

    output = {
        "benchmark": "silver_tier_thalf",
        "dataset": str(CSV_PATH),
        "summary": summary,
        "per_drug": results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Silver-Tier t_half Benchmark Results")
    logger.info("=" * 60)
    logger.info("  Drugs tested:    %d / %d valid", n_valid, len(drugs))
    logger.info("  AAFE:            %.3f", aafe)
    logger.info("  %%within 2-fold:  %.1f%% (%d/%d)", pct_2fold, n_within_2fold, n_valid)
    logger.info("  Total time:      %.1fs", total_time)
    logger.info("  Results saved:   %s", OUTPUT_PATH)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
