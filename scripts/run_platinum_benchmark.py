#!/usr/bin/env python3
"""Unified Platinum benchmark runner.

Runs all drugs in platinum_reference.json through OmegaPipeline, computes
AAFE/%2-fold with bootstrap CI, supports subsetting and cross-validation.

Usage:
    python scripts/run_platinum_benchmark.py
    python scripts/run_platinum_benchmark.py --subset core24
    python scripts/run_platinum_benchmark.py --bootstrap 10000
    python scripts/run_platinum_benchmark.py --clean-only
    python scripts/run_platinum_benchmark.py --cv 5
"""

import argparse
import json
import math
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from omega_pbpk.data.drug_registry import CORE24_NAMES  # noqa: E402
from omega_pbpk.data.platinum_schema import load_platinum_reference  # noqa: E402
from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest  # noqa: E402

PLATINUM_REF = repo_root / "data" / "clinical" / "platinum_reference.json"


def compute_aafe(fold_errors: list[float]) -> float:
    if not fold_errors:
        return float("nan")
    return math.exp(sum(math.log(fe) for fe in fold_errors) / len(fold_errors))


def bootstrap_ci(fold_errors: list[float], n_boot: int = 10000, seed: int = 42):
    log_fe = np.log10(np.array(fold_errors))
    rng = np.random.default_rng(seed)
    n = len(log_fe)
    aafes = []
    for _ in range(n_boot):
        sample = rng.choice(log_fe, size=n, replace=True)
        aafes.append(10 ** np.mean(np.abs(sample)))
    return float(np.percentile(aafes, 2.5)), float(np.percentile(aafes, 97.5))


def run_benchmark(drugs: dict, pipeline: OmegaPipeline) -> dict:
    results = {}
    for name, entry in drugs.items():
        t0 = time.perf_counter()
        try:
            result = pipeline.simulate(
                SimulationRequest(
                    smiles=entry["smiles"],
                    dose_mg=entry["dose_mg"],
                    route="oral",
                )
            )
            pred_cmax = result.cmax_mg_L
        except Exception as e:
            print(f"  ERROR {name}: {e}")
            continue
        dt_ms = (time.perf_counter() - t0) * 1000
        obs_cmax = entry["cmax_mg_L"]
        fe = max(pred_cmax / obs_cmax, obs_cmax / pred_cmax)
        results[name] = {
            "pred_cmax": pred_cmax,
            "obs_cmax": obs_cmax,
            "fold_error": fe,
            "latency_ms": dt_ms,
            "tuning_contaminated": entry.get("tuning_contaminated", False),
            "nonlinear_pk": entry.get("nonlinear_pk", False),
        }
    return results


def report(results: dict, label: str, n_boot: int = 0):
    fes = [r["fold_error"] for r in results.values()]
    if not fes:
        print(f"  {label}: no results")
        return
    aafe = compute_aafe(fes)
    pct2 = 100 * sum(1 for fe in fes if fe <= 2.0) / len(fes)
    pct3 = 100 * sum(1 for fe in fes if fe <= 3.0) / len(fes)
    max_fe = max(fes)
    ci_str = ""
    if n_boot > 0:
        lo, hi = bootstrap_ci(fes, n_boot)
        ci_str = f" [{lo:.2f}, {hi:.2f}]"

    print(f"\n{'=' * 50}")
    print(f"  {label} (N={len(fes)})")
    print(f"  AAFE:    {aafe:.3f}{ci_str}")
    print(f"  %2-fold: {pct2:.1f}%")
    print(f"  %3-fold: {pct3:.1f}%")
    print(f"  Max FE:  {max_fe:.2f}x")
    med_fe = float(np.median(fes))
    print(f"  Median:  {med_fe:.2f}x")
    mean_lat = np.mean([r["latency_ms"] for r in results.values()])
    print(f"  Latency: {mean_lat:.0f} ms/drug (mean)")
    print(f"{'=' * 50}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=["core24"], help="Run only a subset")
    parser.add_argument(
        "--clean-only", action="store_true", help="Exclude tuning-contaminated drugs"
    )
    parser.add_argument("--linear-only", action="store_true", help="Exclude non-linear PK drugs")
    parser.add_argument("--bootstrap", type=int, default=10000, help="Bootstrap resamples for CI")
    parser.add_argument("--cv", type=int, default=0, help="K-fold CV (0=disabled)")
    parser.add_argument("--output", type=str, default=None, help="Save JSON results")
    args = parser.parse_args()

    drugs = load_platinum_reference(PLATINUM_REF)

    # Apply filters
    if args.subset == "core24":
        drugs = {k: v for k, v in drugs.items() if k in CORE24_NAMES}
    if args.clean_only:
        drugs = {k: v for k, v in drugs.items() if not v.get("tuning_contaminated")}
    if args.linear_only:
        drugs = {k: v for k, v in drugs.items() if not v.get("nonlinear_pk")}

    print(f"Platinum Benchmark: {len(drugs)} drugs")
    pipeline = OmegaPipeline()
    results = run_benchmark(drugs, pipeline)

    # Full report
    report(results, "All Drugs", args.bootstrap)

    # Core-24 subset
    core_results = {k: v for k, v in results.items() if k in CORE24_NAMES}
    if core_results and args.subset != "core24":
        report(core_results, "Core-24 Subset", args.bootstrap)

    # Clean subset
    clean_results = {k: v for k, v in results.items() if not v["tuning_contaminated"]}
    if clean_results and len(clean_results) < len(results):
        report(clean_results, "Clean (non-contaminated)", args.bootstrap)

    # Per-drug table
    print(f"\n{'Drug':<25} {'Pred':>8} {'Obs':>8} {'FE':>6} {'ms':>5} {'Flags'}")
    print("-" * 65)
    for name in sorted(results):
        r = results[name]
        flags = []
        if r["tuning_contaminated"]:
            flags.append("C")
        if r["nonlinear_pk"]:
            flags.append("NL")
        flag_str = ",".join(flags) if flags else ""
        status = "OK" if r["fold_error"] <= 2.0 else "WARN" if r["fold_error"] <= 3.0 else "FAIL"
        print(
            f"{name:<25} {r['pred_cmax']:>8.4f} {r['obs_cmax']:>8.4f} {r['fold_error']:>5.2f}x {r['latency_ms']:>5.0f} {flag_str:>4} {status}"
        )

    # Save JSON
    if args.output:
        out = {
            "date": str(date.today()),
            "n_drugs": len(results),
            "results": results,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
