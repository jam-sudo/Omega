#!/usr/bin/env python3
"""Temporal holdout validation runner.

Runs OmegaPipeline on drugs in data/ml/clinical/temporal_holdout.csv that have
observed Cmax data, then computes AAFE and %2-fold with bootstrap 95% CI.

These drugs were NOT in the benchmark or ADME training set, and represent
post-2020 FDA approvals or rarely-studied compounds. They serve as a
true out-of-sample (temporal) holdout.

Usage:
    python scripts/run_temporal_validation.py
    python scripts/run_temporal_validation.py --csv path/to/custom.csv
    python scripts/run_temporal_validation.py --verbose
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from omega_pbpk.ml.applicability import check_applicability  # noqa: E402
from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest  # noqa: E402


def bootstrap_aafe_ci(
    fold_errors: list[float], n_boot: int = 10000, seed: int = 42
) -> tuple[float, float]:
    """Compute 95% bootstrap confidence interval for AAFE.

    Resamples log10 fold-errors with replacement and computes
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


def compute_fold_error(pred: float, obs: float) -> float:
    """Return fold-error = max(pred/obs, obs/pred). Always >= 1.0."""
    if pred <= 0 or obs <= 0:
        return float("nan")
    ratio = pred / obs
    return max(ratio, 1.0 / ratio)


def compute_aafe(fold_errors: list[float]) -> float:
    """Average absolute fold error (geometric mean of fold-errors)."""
    valid = [fe for fe in fold_errors if not np.isnan(fe) and fe > 0]
    if not valid:
        return float("nan")
    return float(10 ** np.mean(np.log10(valid)))


def load_temporal_drugs(csv_path: Path) -> list[dict]:
    """Load drugs from temporal holdout CSV.

    Returns list of dicts with keys: drug_name, smiles, dose_mg, route,
    obs_cmax_mg_L, obs_auc_mg_h_L, obs_thalf_h, source.
    Numeric fields are converted to float (None if empty).
    """
    drugs = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:

            def to_float(val: str) -> float | None:
                v = val.strip()
                return float(v) if v else None

            drugs.append(
                {
                    "drug_name": row["drug_name"].strip(),
                    "smiles": row["smiles"].strip(),
                    "dose_mg": to_float(row["dose_mg"]),
                    "route": row.get("route", "oral").strip(),
                    "obs_cmax_mg_L": to_float(row.get("obs_cmax_mg_L", "")),
                    "obs_auc_mg_h_L": to_float(row.get("obs_auc_mg_h_L", "")),
                    "obs_thalf_h": to_float(row.get("obs_thalf_h", "")),
                    "source": row.get("source", "").strip(),
                }
            )
    return drugs


def run_temporal_validation(csv_path: Path, verbose: bool = False) -> dict:
    """Run pipeline on all temporal holdout drugs with Cmax data.

    Returns a summary dict with per-drug results and aggregate metrics.
    """
    drugs = load_temporal_drugs(csv_path)
    pipeline = OmegaPipeline()

    print("=" * 80)
    print("TEMPORAL HOLDOUT VALIDATION")
    print(f"  CSV: {csv_path}")
    print(f"  N drugs in file: {len(drugs)}")
    cmax_drugs = [d for d in drugs if d["obs_cmax_mg_L"] is not None]
    auc_drugs = [d for d in drugs if d["obs_auc_mg_h_L"] is not None]
    print(f"  N with obs Cmax: {len(cmax_drugs)}")
    print(f"  N with obs AUC:  {len(auc_drugs)}")
    print("=" * 80)

    per_drug_results = []
    cmax_fold_errors: list[float] = []
    auc_fold_errors: list[float] = []

    for drug in drugs:
        name = drug["drug_name"]
        smiles = drug["smiles"]
        dose_mg = drug["dose_mg"]
        obs_cmax = drug["obs_cmax_mg_L"]
        obs_auc = drug["obs_auc_mg_h_L"]
        obs_thalf = drug["obs_thalf_h"]

        if obs_cmax is None and obs_auc is None:
            if verbose:
                print(f"\n--- {name} [SKIPPED — no observed PK data] ---")
            per_drug_results.append(
                {
                    "drug": name,
                    "skipped": True,
                    "reason": "no observed Cmax or AUC",
                }
            )
            continue

        # --- Applicability domain check ---
        ad_result = check_applicability(smiles=smiles, drug_name=name)
        ad_status = "in-scope" if ad_result.tractable else "out-of-scope"

        print(
            f"\n--- {name} (dose={dose_mg}mg) [AD: {ad_status} | flags: {', '.join(ad_result.flags) or 'none'}] ---"
        )

        entry: dict = {
            "drug": name,
            "smiles": smiles,
            "dose_mg": dose_mg,
            "obs_cmax": obs_cmax,
            "obs_auc": obs_auc,
            "obs_thalf": obs_thalf,
            "success": False,
            "ad_tractable": ad_result.tractable,
            "ad_flags": list(ad_result.flags),
        }

        try:
            # Set simulation duration based on known t_half (if available)
            # Use 5x t_half, capped at 168h (1 week), minimum 48h
            if obs_thalf and obs_thalf > 0:
                sim_duration = min(168.0, max(48.0, 5.0 * obs_thalf))
            else:
                sim_duration = 48.0

            t0 = time.perf_counter()
            result = pipeline.simulate(
                SimulationRequest(
                    smiles=smiles,
                    dose_mg=dose_mg,
                    route="oral",
                    duration_h=sim_duration,
                )
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0

            pred_cmax = result.cmax_mg_L
            pred_auc = result.auc0t_mg_h_L
            pred_thalf = result.t_half_h

            entry["success"] = True
            entry["pred_cmax"] = pred_cmax
            entry["pred_auc"] = pred_auc
            entry["pred_thalf"] = pred_thalf
            entry["latency_ms"] = round(latency_ms, 1)

            print(
                f"  Predicted: Cmax={pred_cmax:.4f} mg/L, AUC={pred_auc:.4f} mg*h/L, "
                f"t_half={pred_thalf:.2f}h  ({latency_ms:.0f}ms)"
            )

            if obs_cmax is not None:
                fe_cmax = compute_fold_error(pred_cmax, obs_cmax)
                entry["fe_cmax"] = round(fe_cmax, 4) if not np.isnan(fe_cmax) else None
                cmax_fold_errors.append(fe_cmax)
                print(f"  Obs Cmax={obs_cmax:.4f} mg/L  ->  Cmax FE={fe_cmax:.2f}x")

            if obs_auc is not None:
                fe_auc = compute_fold_error(pred_auc, obs_auc)
                entry["fe_auc"] = round(fe_auc, 4) if not np.isnan(fe_auc) else None
                auc_fold_errors.append(fe_auc)
                print(f"  Obs AUC ={obs_auc:.4f} mg*h/L  ->  AUC  FE={fe_auc:.2f}x")

            if obs_thalf is not None:
                print(f"  Obs t_half={obs_thalf:.1f}h (reference, not scored)")

            if result.warnings:
                for w in result.warnings:
                    print(f"  WARN: {w}")

        except Exception as exc:
            print(f"  FAIL: {exc}")
            entry["error"] = str(exc)

        per_drug_results.append(entry)

    # Aggregate metrics — full set
    valid_cmax = [fe for fe in cmax_fold_errors if not np.isnan(fe)]
    valid_auc = [fe for fe in auc_fold_errors if not np.isnan(fe)]

    aafe_cmax = compute_aafe(valid_cmax)
    aafe_auc = compute_aafe(valid_auc)
    pct_2fold_cmax = sum(1 for fe in valid_cmax if fe <= 2.0) / max(len(valid_cmax), 1) * 100
    pct_2fold_auc = sum(1 for fe in valid_auc if fe <= 2.0) / max(len(valid_auc), 1) * 100

    ci_cmax = bootstrap_aafe_ci(valid_cmax) if len(valid_cmax) >= 3 else (None, None)
    ci_auc = bootstrap_aafe_ci(valid_auc) if len(valid_auc) >= 3 else (None, None)

    # In-scope metrics (AD tractable=True only)
    cmax_entries_all = [
        e for e in per_drug_results if e.get("success") and e.get("fe_cmax") is not None
    ]
    inscope_cmax_fes = [e["fe_cmax"] for e in cmax_entries_all if e.get("ad_tractable", True)]
    outscope_drugs_cmax = [e["drug"] for e in cmax_entries_all if not e.get("ad_tractable", True)]

    aafe_cmax_inscope = compute_aafe(inscope_cmax_fes) if inscope_cmax_fes else float("nan")
    pct_2fold_cmax_inscope = (
        sum(1 for fe in inscope_cmax_fes if fe <= 2.0) / max(len(inscope_cmax_fes), 1) * 100
    )
    ci_cmax_inscope = (
        bootstrap_aafe_ci(inscope_cmax_fes) if len(inscope_cmax_fes) >= 3 else (None, None)
    )

    # Summary table — per-drug (sorted by fold error)
    print("\n" + "=" * 80)
    print("TEMPORAL HOLDOUT SUMMARY")
    print("=" * 80)
    print()
    print("  Per-drug Cmax fold-errors (sorted by fold-error):")
    print(f"  {'Drug':<22s}  {'FE':>6s}  {'Pred':>8s}  {'Obs':>8s}  AD")
    print("  " + "-" * 60)
    for e in sorted(cmax_entries_all, key=lambda x: x["fe_cmax"]):
        within = "OK  " if e["fe_cmax"] <= 2.0 else "MISS"
        scope = "in-scope " if e.get("ad_tractable", True) else "OUT-SCOPE"
        flags_str = ", ".join(e.get("ad_flags", [])) or "none"
        print(
            f"  [{within}] {e['drug']:<20s}  {e['fe_cmax']:>6.2f}x  "
            f"{e['pred_cmax']:>8.3f}  {e['obs_cmax']:>8.3f}  {scope} [{flags_str}]"
        )

    print()
    print("=" * 80)
    print(f"=== FULL temporal holdout (N={len(valid_cmax)}) ===")
    print("=" * 80)
    if not np.isnan(aafe_cmax):
        ci_str = f" [95% CI: {ci_cmax[0]:.2f}, {ci_cmax[1]:.2f}]" if ci_cmax[0] is not None else ""
        print(f"  Cmax AAFE   : {aafe_cmax:.2f}{ci_str}")
        print(f"  Cmax %2-fold: {pct_2fold_cmax:.1f}%")
    if not np.isnan(aafe_auc):
        ci_str = f" [95% CI: {ci_auc[0]:.2f}, {ci_auc[1]:.2f}]" if ci_auc[0] is not None else ""
        print(f"  AUC AAFE    : {aafe_auc:.2f}{ci_str}  (N={len(valid_auc)})")
        print(f"  AUC %2-fold : {pct_2fold_auc:.1f}%")

    print()
    excl_str = ", ".join(outscope_drugs_cmax) if outscope_drugs_cmax else "none"
    print(f"=== IN-SCOPE temporal holdout (N={len(inscope_cmax_fes)}, excl. {excl_str}) ===")
    print("=" * 80)
    print(
        "  Exclusion criterion: tractable=False from applicability domain module "
        "(known_untractable, prodrug ester/phosphonate/carbamate, or invalid SMILES)."
    )
    if not np.isnan(aafe_cmax_inscope):
        ci_str = (
            f" [95% CI: {ci_cmax_inscope[0]:.2f}, {ci_cmax_inscope[1]:.2f}]"
            if ci_cmax_inscope[0] is not None
            else ""
        )
        print(f"  Cmax AAFE   : {aafe_cmax_inscope:.2f}{ci_str}")
        print(f"  Cmax %2-fold: {pct_2fold_cmax_inscope:.1f}%")

    # Sensitivity analysis: exclude sonidegib + atazanavir regardless of AD status
    # (atazanavir flagged by structural SMARTS as prodrug_ester/carbamate but the
    # primary reason for exclusion is P-gp substrate + pH-dependent BCS II solubility)
    _BORDERLINE = {"sonidegib", "atazanavir"}
    borderline_excl = [
        e["fe_cmax"] for e in cmax_entries_all if e["drug"].lower() not in _BORDERLINE
    ]
    aafe_borderline = compute_aafe(borderline_excl) if borderline_excl else float("nan")
    pct_2fold_borderline = (
        sum(1 for fe in borderline_excl if fe <= 2.0) / max(len(borderline_excl), 1) * 100
    )
    ci_borderline = (
        bootstrap_aafe_ci(borderline_excl) if len(borderline_excl) >= 3 else (None, None)
    )
    print()
    print(f"=== SENSITIVITY: excl. sonidegib + atazanavir (N={len(borderline_excl)}) ===")
    print("=" * 80)
    print(
        "  Note: atazanavir is a P-gp substrate with pH-dependent BCS II solubility — "
        "borderline exclusion."
    )
    if not np.isnan(aafe_borderline):
        ci_str = (
            f" [95% CI: {ci_borderline[0]:.2f}, {ci_borderline[1]:.2f}]"
            if ci_borderline[0] is not None
            else ""
        )
        print(f"  Cmax AAFE   : {aafe_borderline:.2f}{ci_str}")
        print(f"  Cmax %2-fold: {pct_2fold_borderline:.1f}%")

    summary = {
        "n_cmax_scored": len(valid_cmax),
        "n_auc_scored": len(valid_auc),
        "aafe_cmax": round(aafe_cmax, 4) if not np.isnan(aafe_cmax) else None,
        "aafe_auc": round(aafe_auc, 4) if not np.isnan(aafe_auc) else None,
        "aafe_cmax_ci95_lo": round(ci_cmax[0], 4) if ci_cmax[0] is not None else None,
        "aafe_cmax_ci95_hi": round(ci_cmax[1], 4) if ci_cmax[1] is not None else None,
        "aafe_auc_ci95_lo": round(ci_auc[0], 4) if ci_auc[0] is not None else None,
        "aafe_auc_ci95_hi": round(ci_auc[1], 4) if ci_auc[1] is not None else None,
        "pct_2fold_cmax": round(pct_2fold_cmax, 2),
        "pct_2fold_auc": round(pct_2fold_auc, 2),
        # In-scope
        "n_cmax_inscope": len(inscope_cmax_fes),
        "aafe_cmax_inscope": round(aafe_cmax_inscope, 4)
        if not np.isnan(aafe_cmax_inscope)
        else None,
        "aafe_cmax_inscope_ci95_lo": round(ci_cmax_inscope[0], 4)
        if ci_cmax_inscope[0] is not None
        else None,
        "aafe_cmax_inscope_ci95_hi": round(ci_cmax_inscope[1], 4)
        if ci_cmax_inscope[1] is not None
        else None,
        "pct_2fold_cmax_inscope": round(pct_2fold_cmax_inscope, 2),
        "excluded_drugs": outscope_drugs_cmax,
        # Sensitivity (excl. sonidegib + atazanavir)
        "n_cmax_sensitivity": len(borderline_excl),
        "aafe_cmax_sensitivity": round(aafe_borderline, 4)
        if not np.isnan(aafe_borderline)
        else None,
        "pct_2fold_cmax_sensitivity": round(pct_2fold_borderline, 2),
        "per_drug": per_drug_results,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run temporal holdout validation.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=repo_root / "data" / "ml" / "clinical" / "temporal_holdout.csv",
        help="Path to temporal holdout CSV (default: data/ml/clinical/temporal_holdout.csv)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print skipped drugs as well",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"ERROR: CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    run_temporal_validation(args.csv, verbose=args.verbose)


if __name__ == "__main__":
    main()
