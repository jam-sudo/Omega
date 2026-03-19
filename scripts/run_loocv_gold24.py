#!/usr/bin/env python3
"""Anchor contamination analysis on Gold-24 benchmark.

Runs the production pipeline on all 25 BENCHMARK_DRUGS and stratifies
results into ANCHORED (14 drugs with CLint reference anchors) vs
CLEAN (11 drugs without anchors). The CLEAN subset AAFE is the honest
in-sample generalization estimate.

Note: this is NOT true LOOCV (which would require retraining per fold).
It's a contamination stratification analysis.

Usage:
    python scripts/run_loocv_gold24.py
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from run_l1_benchmarks import compute_fold_error, load_observed_pk  # noqa: E402

from omega_pbpk.data.drug_registry import BENCHMARK_DRUGS  # noqa: E402
from omega_pbpk.ml.models.adme.xgboost_clint import _ANCHOR_DRUG_NAMES  # noqa: E402
from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest  # noqa: E402


def aafe_with_ci(fes, n_boot=10000, seed=42):
    """Compute AAFE with bootstrap 95% CI."""
    if len(fes) < 2:
        return {"aafe": None, "ci_lo": None, "ci_hi": None, "n": len(fes)}
    log_fe = np.log10(fes)
    aafe = float(10 ** np.mean(np.abs(log_fe)))
    rng = np.random.default_rng(seed)
    n = len(log_fe)
    boots = [float(10 ** np.mean(np.abs(log_fe[rng.integers(0, n, n)]))) for _ in range(n_boot)]
    return {
        "aafe": round(aafe, 4),
        "ci_lo": round(float(np.percentile(boots, 2.5)), 4),
        "ci_hi": round(float(np.percentile(boots, 97.5)), 4),
        "n": n,
        "pct_2fold": round(sum(1 for fe in fes if fe <= 2.0) / n * 100, 1),
    }


def main():
    anchor_drugs = set(_ANCHOR_DRUG_NAMES.values())

    results = []
    fold_errors_all = []
    fold_errors_anchored = []
    fold_errors_clean = []

    pipeline = OmegaPipeline()

    for drug_name, info in BENCHMARK_DRUGS.items():
        smiles = info["smiles"]
        dose_mg = info["dose_mg"]
        has_anchor = drug_name in anchor_drugs

        observed = load_observed_pk(drug_name)
        if not observed or observed.get("cmax", 0) <= 0:
            print(f"SKIP {drug_name}: no observed Cmax")
            continue

        obs_cmax = observed["cmax"]

        # Predict (full pipeline — anchor included if present)
        try:
            sim = pipeline.simulate(SimulationRequest(smiles=smiles, dose_mg=dose_mg, route="oral"))
            pred_cmax = sim.cmax_mg_L
            fe = compute_fold_error(pred_cmax, obs_cmax)
        except Exception as e:
            print(f"FAIL {drug_name}: {e}")
            continue

        entry = {
            "drug": drug_name,
            "has_anchor": has_anchor,
            "pred_cmax": round(pred_cmax, 6),
            "obs_cmax": round(obs_cmax, 6),
            "fold_error": round(fe, 4),
        }
        results.append(entry)
        fold_errors_all.append(fe)
        if has_anchor:
            fold_errors_anchored.append(fe)
        else:
            fold_errors_clean.append(fe)

        status = "ANCHORED" if has_anchor else "CLEAN"
        print(
            f"{drug_name:20s} [{status:8s}] FE={fe:.2f}x  pred={pred_cmax:.4f} obs={obs_cmax:.4f}"
        )

    # Aggregate
    summary = {
        "all_drugs": aafe_with_ci(fold_errors_all),
        "anchored_drugs": aafe_with_ci(fold_errors_anchored),
        "clean_drugs": aafe_with_ci(fold_errors_clean),
        "per_drug": results,
    }

    print("\n" + "=" * 60)
    print(f"ALL ({len(fold_errors_all)} drugs):      AAFE = {summary['all_drugs']['aafe']}")
    print(
        f"ANCHORED ({len(fold_errors_anchored)} drugs): AAFE = {summary['anchored_drugs']['aafe']}"
    )
    print(f"CLEAN ({len(fold_errors_clean)} drugs):    AAFE = {summary['clean_drugs']['aafe']}")
    print(
        f"\nContamination delta: ANCHORED - CLEAN = {summary['anchored_drugs']['aafe'] - summary['clean_drugs']['aafe']:.3f}"
        if summary["anchored_drugs"]["aafe"] and summary["clean_drugs"]["aafe"]
        else ""
    )
    print("=" * 60)

    out_path = REPO / "outputs" / "loocv_gold24.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
