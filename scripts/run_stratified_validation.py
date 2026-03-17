#!/usr/bin/env python3
"""ER-stratified validation: classify benchmark drugs by hepatic extraction
ratio and report AAFE / %2-fold per stratum.

Saves results to outputs/diagnostic_report_er_stratified.json.
"""

import json
import sys
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

Q_LIVER = 90.0  # L/h
IVIVE_ALPHA = 0.3
IVIVE_BETA = 0.9


def classify_er(er: float) -> str:
    if er < 0.3:
        return "low"
    elif er < 0.7:
        return "mid"
    else:
        return "high"


def compute_er(smiles: str, clint_predictor) -> tuple[float, float]:
    """Compute hepatic extraction ratio using XGBoost CLint.

    The pipeline uses: CLh_target = 0.3 * CLint_hep^0.9
    Then ER = CLh_target / Q_liver.

    Returns (er, clint_hepatocyte).
    """
    clint_hep = 0.0
    if clint_predictor is not None:
        try:
            clint_hep = clint_predictor.predict_clint(smiles)
        except Exception:
            pass

    if clint_hep > 0:
        clh_target = min(IVIVE_ALPHA * clint_hep**IVIVE_BETA, 0.95 * Q_LIVER)
        er = clh_target / Q_LIVER
    else:
        er = 0.0

    return er, clint_hep


def main() -> int:
    pipeline = OmegaPipeline()
    pipeline._ensure_initialized()

    drug_records = []

    print("=" * 80)
    print("ER-STRATIFIED VALIDATION")
    print("=" * 80)

    for drug_name, info in BENCHMARK_DRUGS.items():
        smiles = info["smiles"]
        dose_mg = info["dose_mg"]
        print(f"\n--- {drug_name} ---")

        observed = load_observed_pk(drug_name)
        if not observed:
            print("  No observed Cmax data, skipping.")
            continue

        try:
            sim = pipeline.simulate(
                SimulationRequest(
                    smiles=smiles,
                    dose_mg=dose_mg,
                    route="oral",
                    duration_h=24.0,
                )
            )
        except Exception as e:
            print(f"  Simulation failed: {e}")
            continue

        adme = sim.adme_properties
        fup = adme.get("fup", 0.01)

        # Compute ER using the same XGBoost CLint + power-law IVIVE
        # that the pipeline uses internally
        er, clint_hep = compute_er(smiles, pipeline._clint_predictor)
        category = classify_er(er)

        pred_cmax = sim.cmax_mg_L
        obs_cmax = observed["cmax"]
        fe_cmax = compute_fold_error(pred_cmax, obs_cmax)

        clh_target = (
            min(IVIVE_ALPHA * clint_hep**IVIVE_BETA, 0.95 * Q_LIVER) if clint_hep > 0 else 0.0
        )
        print(
            f"  fup={fup:.4f}, CLint_hep={clint_hep:.1f} uL/min/1e6, "
            f"CLh={clh_target:.2f} L/h, ER={er:.3f} ({category}), FE_Cmax={fe_cmax:.2f}x"
        )

        drug_records.append(
            {
                "name": drug_name,
                "smiles": smiles,
                "er": round(er, 4),
                "category": category,
                "fe_cmax": round(fe_cmax, 4) if not np.isnan(fe_cmax) else None,
                "fup": round(fup, 4),
                "clint_hep_uL_min": round(clint_hep, 2),
                "clh_target_L_per_h": round(clh_target, 4),
                "pred_cmax": round(pred_cmax, 6),
                "obs_cmax": round(obs_cmax, 6),
            }
        )

    # Stratify
    stratified = {}
    for cat in ("low", "mid", "high"):
        drugs_in_cat = [d for d in drug_records if d["category"] == cat]
        fes = [d["fe_cmax"] for d in drugs_in_cat if d["fe_cmax"] is not None]
        aafe = compute_aafe(fes) if fes else None
        pct_2fold = sum(1 for fe in fes if fe <= 2.0) / len(fes) * 100 if fes else None
        stratified[cat] = {
            "n": len(drugs_in_cat),
            "drugs": [d["name"] for d in drugs_in_cat],
            "aafe": round(aafe, 4) if aafe and not np.isnan(aafe) else None,
            "pct_2fold": round(pct_2fold, 2) if pct_2fold is not None else None,
        }
        print(f"\n{cat.upper()} ER (n={len(drugs_in_cat)}): AAFE={aafe}, %2-fold={pct_2fold}")
        for d in drugs_in_cat:
            print(f"  {d['name']:20s} ER={d['er']:.3f}  FE={d['fe_cmax']}")

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "drugs": drug_records,
        "stratified": stratified,
    }

    out_path = repo_root / "outputs" / "diagnostic_report_er_stratified.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
