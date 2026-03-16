#!/usr/bin/env python
"""Expanded benchmark runner for all 285 drugs in the Omega reference database.

Runs OmegaPipeline predictions for each drug with a valid SMILES string,
computes fold errors against observed Cmax, AUC, and t_half, and reports
per-tier AAFE metrics. Results are saved to outputs/expanded_benchmark_YYYY-MM-DD.json.

Usage:
    python scripts/run_expanded_benchmark.py
    python scripts/run_expanded_benchmark.py --tiers platinum,gold
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest  # noqa: E402

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def fold_error(pred: float, obs: float) -> float:
    """Fold error = max(pred/obs, obs/pred). Returns NaN for invalid inputs."""
    if abs(pred) < 1e-12 or abs(obs) < 1e-12:
        return float("nan")
    ratio = pred / obs
    return max(ratio, 1.0 / ratio)


def aafe(fold_errors: list[float]) -> float:
    """Geometric mean of fold errors (AAFE)."""
    valid = [fe for fe in fold_errors if not math.isnan(fe) and fe > 0]
    if not valid:
        return float("nan")
    return float(10.0 ** np.mean(np.log10(valid)))


def pct_within_2fold(fold_errors: list[float]) -> float:
    """Percentage of fold errors <= 2.0."""
    valid = [fe for fe in fold_errors if not math.isnan(fe) and fe > 0]
    if not valid:
        return float("nan")
    return 100.0 * sum(1 for fe in valid if fe <= 2.0) / len(valid)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_reference_database(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def run_benchmark(tiers_filter: list[str] | None = None) -> dict:
    """Run the expanded benchmark and return results dict."""
    db_path = repo_root / "data" / "clinical" / "reference_database.json"
    db = load_reference_database(db_path)
    drugs = db["drugs"]

    pipeline = OmegaPipeline()

    per_drug_results: list[dict] = []
    # Collect fold errors per tier per metric
    tier_fold_errors: dict[str, dict[str, list[float]]] = {}

    n_total = 0
    n_success = 0

    for drug_name, drug_data in sorted(drugs.items()):
        tier = drug_data.get("tier", "unknown")

        # Filter by tier if requested
        if tiers_filter and tier not in tiers_filter:
            continue

        smiles = drug_data.get("smiles")
        if not smiles:
            continue

        n_total += 1
        dose_mg = drug_data.get("dose_mg", 100.0)
        route = drug_data.get("route", "oral")
        pk_params = drug_data.get("pk_params", {})

        obs_cmax = pk_params.get("cmax_mg_L")
        obs_auc = pk_params.get("auc_mg_h_L")
        obs_thalf = pk_params.get("thalf_h")

        # Run prediction
        t_start = time.perf_counter()
        try:
            req = SimulationRequest(smiles=smiles, dose_mg=dose_mg, route=route, duration_h=24.0)
            result = pipeline.simulate(req)
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            n_success += 1
        except Exception as e:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            print(f"  FAIL {drug_name}: {e}")
            per_drug_results.append(
                {
                    "drug": drug_name,
                    "tier": tier,
                    "smiles": smiles,
                    "dose_mg": dose_mg,
                    "latency_ms": round(latency_ms, 1),
                    "error": str(e),
                }
            )
            continue

        pred_cmax = result.cmax_mg_L
        pred_auc = result.auc0t_mg_h_L
        pred_thalf = result.t_half_h

        # Compute fold errors (only when observed value exists)
        fe_cmax = fold_error(pred_cmax, obs_cmax) if obs_cmax is not None else None
        fe_auc = fold_error(pred_auc, obs_auc) if obs_auc is not None else None
        fe_thalf = fold_error(pred_thalf, obs_thalf) if obs_thalf is not None else None

        # Accumulate per-tier fold errors
        if tier not in tier_fold_errors:
            tier_fold_errors[tier] = {"cmax": [], "auc": [], "thalf": []}
        if fe_cmax is not None and not math.isnan(fe_cmax):
            tier_fold_errors[tier]["cmax"].append(fe_cmax)
        if fe_auc is not None and not math.isnan(fe_auc):
            tier_fold_errors[tier]["auc"].append(fe_auc)
        if fe_thalf is not None and not math.isnan(fe_thalf):
            tier_fold_errors[tier]["thalf"].append(fe_thalf)

        per_drug_results.append(
            {
                "drug": drug_name,
                "tier": tier,
                "smiles": smiles,
                "dose_mg": dose_mg,
                "latency_ms": round(latency_ms, 1),
                "pred_cmax": pred_cmax,
                "pred_auc": pred_auc,
                "pred_thalf": pred_thalf,
                "obs_cmax": obs_cmax,
                "obs_auc": obs_auc,
                "obs_thalf": obs_thalf,
                "fe_cmax": round(fe_cmax, 3) if fe_cmax is not None else None,
                "fe_auc": round(fe_auc, 3) if fe_auc is not None else None,
                "fe_thalf": round(fe_thalf, 3) if fe_thalf is not None else None,
            }
        )

        status = "OK"
        if fe_cmax is not None and fe_cmax > 5.0:
            status = f"HIGH fe_cmax={fe_cmax:.1f}"
        print(f"  {drug_name:30s}  tier={tier:10s}  {latency_ms:6.0f}ms  {status}")

    # Build tier_metrics summary
    tier_metrics: dict[str, dict] = {}
    for tier, fes in sorted(tier_fold_errors.items()):
        tier_metrics[tier] = {
            "n_drugs": sum(
                1 for d in per_drug_results if d.get("tier") == tier and "error" not in d
            ),
            "cmax_aafe": round(aafe(fes["cmax"]), 3) if fes["cmax"] else None,
            "cmax_n": len(fes["cmax"]),
            "cmax_pct_2fold": (round(pct_within_2fold(fes["cmax"]), 1) if fes["cmax"] else None),
            "auc_aafe": round(aafe(fes["auc"]), 3) if fes["auc"] else None,
            "auc_n": len(fes["auc"]),
            "auc_pct_2fold": (round(pct_within_2fold(fes["auc"]), 1) if fes["auc"] else None),
            "thalf_aafe": round(aafe(fes["thalf"]), 3) if fes["thalf"] else None,
            "thalf_n": len(fes["thalf"]),
            "thalf_pct_2fold": (round(pct_within_2fold(fes["thalf"]), 1) if fes["thalf"] else None),
        }

    output = {
        "timestamp": datetime.now().isoformat(),
        "n_drugs_total": n_total,
        "n_success": n_success,
        "tier_metrics": tier_metrics,
        "per_drug": per_drug_results,
    }

    return output


def print_summary(results: dict) -> None:
    """Print a summary table to stdout."""
    print("\n" + "=" * 80)
    print(f"EXPANDED BENCHMARK RESULTS  ({results['timestamp']})")
    print(f"Total drugs: {results['n_drugs_total']}  |  Success: {results['n_success']}")
    print("=" * 80)

    header = (
        f"{'Tier':12s}  {'N':>4s}  "
        f"{'Cmax AAFE':>10s} {'(n)':>4s} {'%2f':>5s}  "
        f"{'AUC AAFE':>10s} {'(n)':>4s} {'%2f':>5s}  "
        f"{'t1/2 AAFE':>10s} {'(n)':>4s} {'%2f':>5s}"
    )
    print(header)
    print("-" * len(header))

    for tier, m in sorted(results["tier_metrics"].items()):
        cmax_str = f"{m['cmax_aafe']:.2f}" if m["cmax_aafe"] is not None else "—"
        cmax_pct = f"{m['cmax_pct_2fold']:.0f}" if m["cmax_pct_2fold"] is not None else "—"
        auc_str = f"{m['auc_aafe']:.2f}" if m["auc_aafe"] is not None else "—"
        auc_pct = f"{m['auc_pct_2fold']:.0f}" if m["auc_pct_2fold"] is not None else "—"
        thalf_str = f"{m['thalf_aafe']:.2f}" if m["thalf_aafe"] is not None else "—"
        thalf_pct = f"{m['thalf_pct_2fold']:.0f}" if m["thalf_pct_2fold"] is not None else "—"

        print(
            f"{tier:12s}  {m['n_drugs']:4d}  "
            f"{cmax_str:>10s} {m['cmax_n']:4d} {cmax_pct:>5s}  "
            f"{auc_str:>10s} {m['auc_n']:4d} {auc_pct:>5s}  "
            f"{thalf_str:>10s} {m['thalf_n']:4d} {thalf_pct:>5s}"
        )

    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run expanded PK benchmark on Omega reference database"
    )
    parser.add_argument(
        "--tiers",
        type=str,
        default=None,
        help="Comma-separated list of tiers to include (e.g. platinum,gold)",
    )
    args = parser.parse_args()

    tiers_filter = None
    if args.tiers:
        tiers_filter = [t.strip() for t in args.tiers.split(",")]
        print(f"Filtering to tiers: {tiers_filter}")

    print("Running expanded benchmark...")
    results = run_benchmark(tiers_filter=tiers_filter)

    # Save output
    out_dir = repo_root / "outputs"
    out_dir.mkdir(exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    out_path = out_dir / f"expanded_benchmark_{date_str}.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print_summary(results)


if __name__ == "__main__":
    main()
