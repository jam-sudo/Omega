#!/usr/bin/env python3
"""Run pipeline on permanent holdout set and record baseline metrics.

This script is the GROUND TRUTH for measuring generalization improvement.
Never modify to improve results — only measure honestly.

Usage:
    python scripts/run_holdout_benchmark.py
"""

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest  # noqa: E402

SPLIT_PATH = REPO / "data" / "clinical" / "holdout_split.json"
PLATINUM_PATH = REPO / "data" / "clinical" / "platinum_reference.json"


def bootstrap_aafe_ci(fold_errors, n_boot=10000, seed=42):
    log_fe = np.log10(np.array(fold_errors))
    rng = np.random.default_rng(seed)
    n = len(log_fe)
    boots = [float(10 ** np.mean(np.abs(log_fe[rng.integers(0, n, n)]))) for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def compute_fold_error(pred, obs):
    if pred <= 0 or obs <= 0:
        return float("nan")
    return max(pred / obs, obs / pred)


def main():
    # Load split
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    holdout_drugs = set(split["holdout"])

    # Load platinum
    with open(PLATINUM_PATH) as f:
        plat = json.load(f)

    from omega_pbpk.pipeline import _check_applicability_domain  # noqa: E402

    pipeline = OmegaPipeline()
    results = []
    fold_errors = []
    in_domain_fe = []  # excludes prodrugs, DDI-boosted, AD-flagged
    strat_errors = defaultdict(list)  # stratification key -> [fold_errors]

    print(f"Running holdout benchmark on {len(holdout_drugs)} drugs")
    print("=" * 70)

    for drug_name in sorted(holdout_drugs):
        entry = plat["drugs"].get(drug_name)
        if entry is None:
            print(f"WARNING: {drug_name} not in platinum — skipping")
            continue

        smiles = entry["smiles"]
        dose_mg = entry["dose_mg"]
        obs_cmax = entry["cmax_mg_L"]
        data_quality = entry.get("data_quality", "unknown")
        ddi_boosted = entry.get("ddi_boosted", False)

        # Check applicability domain
        in_ad, ad_flags = _check_applicability_domain(smiles)

        try:
            sim = pipeline.simulate(SimulationRequest(smiles=smiles, dose_mg=dose_mg, route="oral"))
            pred_cmax = sim.cmax_mg_L
            fe = compute_fold_error(pred_cmax, obs_cmax)
        except Exception as e:
            print(f"  FAIL {drug_name}: {e}")
            results.append(
                {
                    "drug": drug_name,
                    "success": False,
                    "error": str(e),
                }
            )
            continue

        fold_errors.append(fe)
        strat_errors[data_quality].append(fe)

        # In-domain: not DDI-boosted and passes AD filter
        is_in_domain = in_ad and not ddi_boosted
        if is_in_domain:
            in_domain_fe.append(fe)

        exclusion_reason = []
        if ddi_boosted:
            exclusion_reason.append("DDI_BOOSTED")
        if not in_ad:
            exclusion_reason.extend(ad_flags)

        results.append(
            {
                "drug": drug_name,
                "success": True,
                "smiles": smiles,
                "dose_mg": dose_mg,
                "pred_cmax": round(pred_cmax, 6),
                "obs_cmax": round(obs_cmax, 6),
                "fold_error": round(fe, 4),
                "data_quality": data_quality,
                "source_type": entry.get("source_type"),
                "in_domain": is_in_domain,
                "exclusion_reason": exclusion_reason if exclusion_reason else None,
            }
        )

        flag_str = f" [{','.join(exclusion_reason)}]" if exclusion_reason else ""
        symbol = "v" if fe <= 2.0 else ("~" if fe <= 3.0 else "x")
        print(
            f"  {symbol} {drug_name:25s} FE={fe:6.2f}x  pred={pred_cmax:.4f}  obs={obs_cmax:.4f}  [{data_quality}]{flag_str}"
        )

    # Aggregate
    valid_fe = [fe for fe in fold_errors if not np.isnan(fe)]
    log_fe = np.log10(valid_fe)
    aafe = float(10 ** np.mean(np.abs(log_fe)))
    pct_2fold = sum(1 for fe in valid_fe if fe <= 2.0) / len(valid_fe) * 100
    pct_3fold = sum(1 for fe in valid_fe if fe <= 3.0) / len(valid_fe) * 100
    ci_lo, ci_hi = bootstrap_aafe_ci(valid_fe)

    # Stratified
    strat_summary = {}
    for key, fes in strat_errors.items():
        if len(fes) >= 3:
            s_log = np.log10(fes)
            s_ci = bootstrap_aafe_ci(fes)
            strat_summary[key] = {
                "n": len(fes),
                "aafe": round(float(10 ** np.mean(np.abs(s_log))), 4),
                "ci_lo": round(s_ci[0], 4),
                "ci_hi": round(s_ci[1], 4),
                "pct_2fold": round(sum(1 for f in fes if f <= 2.0) / len(fes) * 100, 1),
            }

    # Top errors
    sorted_results = sorted(
        [r for r in results if r.get("success")],
        key=lambda r: r["fold_error"],
        reverse=True,
    )

    # In-domain metrics
    in_domain_summary = {}
    if in_domain_fe:
        id_valid = [fe for fe in in_domain_fe if not np.isnan(fe)]
        id_log = np.log10(id_valid)
        id_aafe = float(10 ** np.mean(np.abs(id_log)))
        id_2f = sum(1 for fe in id_valid if fe <= 2.0) / len(id_valid) * 100
        id_3f = sum(1 for fe in id_valid if fe <= 3.0) / len(id_valid) * 100
        id_ci = bootstrap_aafe_ci(id_valid)
        in_domain_summary = {
            "n": len(id_valid),
            "aafe": round(id_aafe, 4),
            "ci95_lo": round(id_ci[0], 4),
            "ci95_hi": round(id_ci[1], 4),
            "pct_2fold": round(id_2f, 1),
            "pct_3fold": round(id_3f, 1),
        }

    # Excluded drugs summary
    excluded = [r for r in results if r.get("success") and not r.get("in_domain")]

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_holdout": len(holdout_drugs),
        "n_success": len(valid_fe),
        "aafe": round(aafe, 4),
        "ci95_lo": round(ci_lo, 4),
        "ci95_hi": round(ci_hi, 4),
        "pct_2fold": round(pct_2fold, 1),
        "pct_3fold": round(pct_3fold, 1),
        "in_domain": in_domain_summary,
        "n_excluded": len(excluded),
        "excluded_drugs": [
            {"drug": r["drug"], "reason": r.get("exclusion_reason")} for r in excluded
        ],
        "stratified_by_quality": strat_summary,
        "top_10_errors": [
            {"drug": r["drug"], "fold_error": r["fold_error"]} for r in sorted_results[:10]
        ],
        "per_drug": results,
    }

    print("\n" + "=" * 70)
    print(f"HOLDOUT ALL ({len(valid_fe)} drugs)")
    print(f"  AAFE:    {aafe:.3f}  [95% CI: {ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"  %2-fold: {pct_2fold:.1f}%")
    print(f"  %3-fold: {pct_3fold:.1f}%")
    if in_domain_summary:
        print(f"\nHOLDOUT IN-DOMAIN ({in_domain_summary['n']} drugs)")
        print(
            f"  AAFE:    {in_domain_summary['aafe']:.3f}  [95% CI: {in_domain_summary['ci95_lo']:.3f}, {in_domain_summary['ci95_hi']:.3f}]"
        )
        print(f"  %2-fold: {in_domain_summary['pct_2fold']:.1f}%")
        print(f"  %3-fold: {in_domain_summary['pct_3fold']:.1f}%")
        print(f"  Excluded: {len(excluded)} drugs")
        for r in excluded:
            print(f"    - {r['drug']}: {r.get('exclusion_reason')}")
    print("\nStratified by quality:")
    for key, s in sorted(strat_summary.items()):
        print(
            f"  {key:30s} N={s['n']:3d}  AAFE={s['aafe']:.3f} [{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]  %2f={s['pct_2fold']:.0f}%"
        )
    print("\nTop 5 errors:")
    for r in sorted_results[:5]:
        print(f"  {r['drug']:25s} FE={r['fold_error']:.2f}x")

    out_path = REPO / "outputs" / "holdout_baseline.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
