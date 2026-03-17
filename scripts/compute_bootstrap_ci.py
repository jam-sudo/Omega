#!/usr/bin/env python3
"""Compute bootstrap confidence intervals for AAFE from benchmark fold errors.

Loads the latest benchmark results, extracts per-drug Cmax fold errors,
computes 10,000 bootstrap resamples of AAFE, and reports 95% CI.
Also tests whether Bayer's reported mfce of 1.87 falls inside the CI.

Output: outputs/diagnostic_report_bootstrap_ci.json
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent


def load_latest_benchmark() -> dict:
    """Load the most recent benchmark JSON from outputs/."""
    outputs_dir = repo_root / "outputs"
    # Prefer date-stamped benchmarks (benchmark_YYYY-MM-DD.json)
    candidates = sorted(outputs_dir.glob("benchmark_20*.json"), reverse=True)
    if not candidates:
        # Fall back to any benchmark file
        candidates = sorted(outputs_dir.glob("benchmark_*.json"), reverse=True)
    if not candidates:
        raise FileNotFoundError("No benchmark_*.json files found in outputs/")
    latest = candidates[0]
    print(f"Loading benchmark: {latest.name}")
    with open(latest) as f:
        return json.load(f)


def extract_cmax_fold_errors(benchmark: dict) -> list[float]:
    """Extract valid Cmax fold errors from benchmark results."""
    fold_errors = []
    for entry in benchmark.get("per_drug", []):
        fe = entry.get("fe_cmax")
        if fe is not None and not np.isnan(fe) and fe > 0:
            fold_errors.append(fe)
    return fold_errors


def bootstrap_aafe_ci(
    fold_errors: list[float],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Compute bootstrap CI for AAFE.

    AAFE = 10^mean(|log10(FE)|)

    Returns dict with point estimate, 95% CI bounds, and Bayer comparison.
    """
    log_fe = np.log10(np.array(fold_errors))
    rng = np.random.default_rng(seed)

    boot_aafes = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(log_fe), size=len(log_fe))
        boot_aafes.append(10 ** np.mean(np.abs(log_fe[idx])))

    boot_aafes = np.array(boot_aafes)
    ci_lo = float(np.percentile(boot_aafes, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_aafes, 100 * (1 - alpha / 2)))

    return {
        "aafe": float(10 ** np.mean(np.abs(log_fe))),
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "n_drugs": len(fold_errors),
        "n_bootstrap": n_boot,
        "bayer_1.87_inside_ci": bool(ci_lo <= 1.87 <= ci_hi),
    }


def main():
    # Load benchmark
    benchmark = load_latest_benchmark()
    fold_errors = extract_cmax_fold_errors(benchmark)

    if len(fold_errors) == 0:
        print("ERROR: No valid Cmax fold errors found in benchmark.")
        sys.exit(1)

    print(f"Found {len(fold_errors)} drugs with Cmax fold errors")
    print(f"Fold errors: {[round(fe, 3) for fe in fold_errors]}")

    # Compute bootstrap CI
    result = bootstrap_aafe_ci(fold_errors, n_boot=10000)

    # Also compute per-drug summary
    per_drug = []
    for entry in benchmark.get("per_drug", []):
        fe = entry.get("fe_cmax")
        if fe is not None and not np.isnan(fe) and fe > 0:
            per_drug.append(
                {
                    "drug": entry["drug"],
                    "fe_cmax": round(fe, 4),
                }
            )

    # Compute % within 2-fold with bootstrap CI
    np.log10(np.array(fold_errors))
    rng = np.random.default_rng(42)
    boot_pct2fold = []
    for _ in range(10000):
        idx = rng.integers(0, len(fold_errors), size=len(fold_errors))
        sample = np.array(fold_errors)[idx]
        pct = np.sum(sample <= 2.0) / len(sample) * 100
        boot_pct2fold.append(pct)

    pct2fold_ci_lo = float(np.percentile(boot_pct2fold, 2.5))
    pct2fold_ci_hi = float(np.percentile(boot_pct2fold, 97.5))
    pct2fold_point = sum(1 for fe in fold_errors if fe <= 2.0) / len(fold_errors) * 100

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark_source": "outputs/benchmark_2026-03-17.json",
        "cmax_aafe_bootstrap": result,
        "cmax_pct_2fold": {
            "point_estimate": round(pct2fold_point, 2),
            "ci95_lo": round(pct2fold_ci_lo, 2),
            "ci95_hi": round(pct2fold_ci_hi, 2),
        },
        "interpretation": {
            "omega_vs_bayer": (
                "Bayer mfce 1.87 is INSIDE Omega 95% CI"
                if result["bayer_1.87_inside_ci"]
                else "Bayer mfce 1.87 is OUTSIDE Omega 95% CI (Omega is significantly better)"
            ),
            "note": (
                "AAFE and mfce are both geometric mean fold error metrics. "
                "Bayer uses median fold change error (mfce=1.87) on their dataset. "
                "This CI is for Omega's AAFE on its 24-drug Gold benchmark."
            ),
        },
        "per_drug_fold_errors": per_drug,
    }

    # Print summary
    print(f"\n{'=' * 60}")
    print("BOOTSTRAP CI RESULTS (10,000 resamples)")
    print(f"{'=' * 60}")
    print(f"Cmax AAFE:  {result['aafe']:.3f}  [{result['ci95_lo']:.3f}, {result['ci95_hi']:.3f}]")
    print(f"Cmax %2-fold: {pct2fold_point:.1f}%  [{pct2fold_ci_lo:.1f}%, {pct2fold_ci_hi:.1f}%]")
    print(f"Bayer 1.87 inside CI: {result['bayer_1.87_inside_ci']}")
    print(f"N drugs: {result['n_drugs']}")

    # Save
    out_path = repo_root / "outputs" / "diagnostic_report_bootstrap_ci.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
