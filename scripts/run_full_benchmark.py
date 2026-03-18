#!/usr/bin/env python3
"""Automated benchmark runner with regression detection.

Runs all BENCHMARK_DRUGS through OmegaPipeline.simulate(), measures latency,
outputs structured JSON results, and optionally compares against a previous
benchmark to detect regressions.

Usage:
    python scripts/run_full_benchmark.py
    python scripts/run_full_benchmark.py --previous outputs/benchmark_2026-03-14.json
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "scripts"))

from run_l1_benchmarks import (  # noqa: E402
    BENCHMARK_DRUGS,
    compute_aafe,
    compute_fold_error,
    load_observed_pk,
)

from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest  # noqa: E402


def bootstrap_aafe_ci(
    fold_errors: list[float], n_boot: int = 10000, seed: int = 42
) -> tuple[float, float]:
    """Compute 95% bootstrap confidence interval for AAFE.

    Resamples the log10 fold-errors with replacement and computes the
    AAFE (10^mean(|log10(FE)|)) for each bootstrap replicate.

    Returns:
        (ci95_lo, ci95_hi) as floats.
    """
    log_fe = np.log10(np.array(fold_errors))
    rng = np.random.default_rng(seed)
    n = len(log_fe)
    boot_aafes = [
        float(10 ** np.mean(np.abs(log_fe[rng.integers(0, n, n)]))) for _ in range(n_boot)
    ]
    return float(np.percentile(boot_aafes, 2.5)), float(np.percentile(boot_aafes, 97.5))


def run_benchmark(ml_corrections: bool = False) -> dict:
    """Run the full benchmark and return structured results."""
    pipeline = OmegaPipeline()
    if ml_corrections:
        pipeline.use_ml_corrections = True

    per_drug_results = []
    cmax_fold_errors = []
    auc_fold_errors = []
    n_success = 0

    print("=" * 80)
    print("FULL BENCHMARK: SMILES -> OmegaPipeline.simulate() -> PK Metrics")
    print("=" * 80)

    for drug_name, info in BENCHMARK_DRUGS.items():
        smiles = info["smiles"]
        dose_mg = info["dose_mg"]
        print(f"\n--- {drug_name} (dose={dose_mg}mg) ---")

        observed = load_observed_pk(drug_name)
        if not observed:
            print(f"  WARNING: No observed data found for {drug_name}")

        entry = {
            "drug": drug_name,
            "smiles": smiles,
            "dose_mg": dose_mg,
            "success": False,
        }

        try:
            t0 = time.perf_counter()
            sim_result = pipeline.simulate(
                SimulationRequest(
                    smiles=smiles,
                    dose_mg=dose_mg,
                    route="oral",
                    duration_h=24.0,
                )
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0

            pred_cmax = sim_result.cmax_mg_L
            pred_auc = sim_result.auc0t_mg_h_L
            pred_thalf = sim_result.t_half_h

            entry.update(
                {
                    "success": True,
                    "pred_cmax": pred_cmax,
                    "pred_auc": pred_auc,
                    "pred_thalf": pred_thalf,
                    "latency_ms": round(latency_ms, 1),
                }
            )

            print(
                f"  Predicted: Cmax={pred_cmax:.4f} mg/L, AUC={pred_auc:.4f} mg*h/L, "
                f"t_half={pred_thalf:.2f}h  ({latency_ms:.0f}ms)"
            )

            if observed:
                obs_cmax = observed["cmax"]
                obs_auc = observed["auc"]
                fe_cmax = compute_fold_error(pred_cmax, obs_cmax)
                fe_auc = compute_fold_error(pred_auc, obs_auc)
                cmax_fold_errors.append(fe_cmax)
                auc_fold_errors.append(fe_auc)
                entry.update(
                    {
                        "obs_cmax": obs_cmax,
                        "obs_auc": obs_auc,
                        "fe_cmax": round(fe_cmax, 4) if not np.isnan(fe_cmax) else None,
                        "fe_auc": round(fe_auc, 4) if not np.isnan(fe_auc) else None,
                    }
                )
                print(f"  Observed:  Cmax={obs_cmax:.4f} mg/L, AUC={obs_auc:.4f} mg*h/L")
                print(f"  Fold-error: Cmax={fe_cmax:.2f}x, AUC={fe_auc:.2f}x")

            n_success += 1

        except Exception as e:
            print(f"  FAIL: {e}")
            entry["error"] = str(e)

        per_drug_results.append(entry)

    # Aggregate metrics
    valid_cmax = [fe for fe in cmax_fold_errors if not np.isnan(fe)]
    valid_auc = [fe for fe in auc_fold_errors if not np.isnan(fe)]
    aafe_cmax = compute_aafe(cmax_fold_errors)
    aafe_auc = compute_aafe(auc_fold_errors)
    pct_2fold_cmax = sum(1 for fe in valid_cmax if fe <= 2.0) / max(len(valid_cmax), 1) * 100
    pct_2fold_auc = sum(1 for fe in valid_auc if fe <= 2.0) / max(len(valid_auc), 1) * 100

    # Bootstrap 95% CI for AAFE
    ci_cmax = bootstrap_aafe_ci(valid_cmax) if len(valid_cmax) >= 3 else (None, None)
    ci_auc = bootstrap_aafe_ci(valid_auc) if len(valid_auc) >= 3 else (None, None)

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_drugs": len(BENCHMARK_DRUGS),
        "n_success": n_success,
        "aafe_cmax": round(aafe_cmax, 4) if not np.isnan(aafe_cmax) else None,
        "aafe_auc": round(aafe_auc, 4) if not np.isnan(aafe_auc) else None,
        "aafe_cmax_ci95_lo": round(ci_cmax[0], 4) if ci_cmax[0] is not None else None,
        "aafe_cmax_ci95_hi": round(ci_cmax[1], 4) if ci_cmax[1] is not None else None,
        "aafe_auc_ci95_lo": round(ci_auc[0], 4) if ci_auc[0] is not None else None,
        "aafe_auc_ci95_hi": round(ci_auc[1], 4) if ci_auc[1] is not None else None,
        "pct_2fold_cmax": round(pct_2fold_cmax, 2),
        "pct_2fold_auc": round(pct_2fold_auc, 2),
        "per_drug": per_drug_results,
    }

    return result


def check_regressions(current: dict, previous: dict) -> list[str]:
    """Compare current vs previous benchmark and return regression messages."""
    regressions = []

    # Aggregate AAFE regression (>20% increase)
    for metric in ("aafe_cmax", "aafe_auc"):
        cur_val = current.get(metric)
        prev_val = previous.get(metric)
        if cur_val is not None and prev_val is not None and prev_val > 0:
            pct_change = (cur_val - prev_val) / prev_val * 100
            if pct_change > 20:
                regressions.append(
                    f"REGRESSION: {metric} increased {pct_change:.1f}% "
                    f"({prev_val:.3f} -> {cur_val:.3f})"
                )

    # Per-drug fold-error regression (>50% increase)
    prev_by_drug = {d["drug"]: d for d in previous.get("per_drug", [])}
    for entry in current.get("per_drug", []):
        drug = entry.get("drug")
        prev_entry = prev_by_drug.get(drug)
        if not prev_entry:
            continue
        for fe_key in ("fe_cmax", "fe_auc"):
            cur_fe = entry.get(fe_key)
            prev_fe = prev_entry.get(fe_key)
            if cur_fe is not None and prev_fe is not None and prev_fe > 0:
                pct_change = (cur_fe - prev_fe) / prev_fe * 100
                if pct_change > 50:
                    regressions.append(
                        f"REGRESSION: {drug} {fe_key} increased {pct_change:.1f}% "
                        f"({prev_fe:.3f} -> {cur_fe:.3f})"
                    )

    return regressions


def print_summary(result: dict, regressions: list[str] | None = None) -> None:
    """Print human-readable summary."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Drugs: {result['n_success']}/{result['n_drugs']} succeeded")
    cmax_ci_lo = result.get("aafe_cmax_ci95_lo")
    cmax_ci_hi = result.get("aafe_cmax_ci95_hi")
    auc_ci_lo = result.get("aafe_auc_ci95_lo")
    auc_ci_hi = result.get("aafe_auc_ci95_hi")
    cmax_ci_str = f"  [{cmax_ci_lo:.2f}, {cmax_ci_hi:.2f}]" if cmax_ci_lo else ""
    auc_ci_str = f"  [{auc_ci_lo:.2f}, {auc_ci_hi:.2f}]" if auc_ci_lo else ""
    print(f"Cmax AAFE: {result['aafe_cmax']}{cmax_ci_str}")
    print(f"AUC  AAFE: {result['aafe_auc']}{auc_ci_str}")
    print(f"Cmax %2-fold: {result['pct_2fold_cmax']:.0f}%")
    print(f"AUC  %2-fold: {result['pct_2fold_auc']:.0f}%")

    # Count >3-fold Cmax errors
    gt3_cmax = sum(
        1 for d in result["per_drug"] if d.get("fe_cmax") is not None and d["fe_cmax"] > 3.0
    )
    print(f">3-fold Cmax errors: {gt3_cmax}")

    if regressions:
        print("\n*** REGRESSIONS DETECTED ***")
        for r in regressions:
            print(f"  {r}")
    elif regressions is not None:
        print("\nNo regressions detected vs previous benchmark.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run full benchmark with optional regression detection"
    )
    parser.add_argument(
        "--previous",
        type=str,
        default=None,
        help="Path to previous benchmark JSON for regression comparison",
    )
    parser.add_argument(
        "--ml-corrections",
        action="store_true",
        help="Enable ML correction layers (Pre-ODE + Post-ODE)",
    )
    args = parser.parse_args()

    # Run benchmark
    result = run_benchmark(ml_corrections=args.ml_corrections)

    # Save output
    today = datetime.now().strftime("%Y-%m-%d")
    out_path = repo_root / "outputs" / f"benchmark_{today}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Regression check
    regressions = None
    if args.previous:
        prev_path = Path(args.previous)
        if not prev_path.is_absolute():
            prev_path = repo_root / prev_path
        if prev_path.exists():
            with open(prev_path) as f:
                previous = json.load(f)
            regressions = check_regressions(result, previous)
        else:
            print(f"\nWARNING: Previous benchmark file not found: {prev_path}")
            regressions = []
    else:
        regressions = []

    print_summary(result, regressions)

    if regressions:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
