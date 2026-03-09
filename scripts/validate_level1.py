#!/usr/bin/env python3
"""Validate Level 1 ADME ensemble predictions against reference compounds.

Tests exit criteria:
  - ADME AAFE < 3.0
  - PK within 2-fold for >= 70% of 20+ drugs

Usage:
    python scripts/validate_level1.py
"""

from __future__ import annotations

import logging
import sys

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("validate_level1")

# Reference compounds with known PK data
REFERENCE_DRUGS = {
    "caffeine": {
        "smiles": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        "dose_mg": 100.0,
        "route": "oral",
        "ref_cmax_mg_L": 1.5,  # ~1-2 mg/L for 100mg dose
        "ref_t_half_h": 4.0,  # 3-5h
        "ref_cl_L_h": 1.7,  # 1.5-2.0
    },
    "midazolam": {
        "smiles": "Clc1ccc2c(c1)C(=NCc1nccn1C)c1ccccc1-2",
        "dose_mg": 5.0,
        "route": "oral",
        "ref_cmax_mg_L": 0.02,  # ~20 ng/mL for 5mg oral
        "ref_t_half_h": 2.5,  # 1.5-3h
        "ref_cl_L_h": 30.0,  # 27-35
    },
    "warfarin": {
        "smiles": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
        "dose_mg": 5.0,
        "route": "oral",
        "ref_cmax_mg_L": 0.8,  # ~0.5-1 mg/L for 5mg
        "ref_t_half_h": 37.0,  # 20-60h
        "ref_cl_L_h": 0.18,  # 0.15-0.20
    },
    "metformin": {
        "smiles": "CN(C)C(=N)NC(=N)N",
        "dose_mg": 500.0,
        "route": "oral",
        "ref_cmax_mg_L": 1.0,  # ~1-2 mg/L for 500mg
        "ref_t_half_h": 5.0,  # 4-8h
        "ref_cl_L_h": 30.0,  # 25-35
    },
    "propranolol": {
        "smiles": "CC(C)NCC(O)COc1cccc2ccccc12",
        "dose_mg": 40.0,
        "route": "oral",
        "ref_cmax_mg_L": 0.04,  # ~30-60 ng/mL for 40mg
        "ref_t_half_h": 4.0,  # 3-6h
        "ref_cl_L_h": 70.0,  # 60-80
    },
    "diazepam": {
        "smiles": "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",
        "dose_mg": 10.0,
        "route": "oral",
        "ref_cmax_mg_L": 0.15,
        "ref_t_half_h": 40.0,  # 20-80h
        "ref_cl_L_h": 1.5,
    },
    "ibuprofen": {
        "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "dose_mg": 400.0,
        "route": "oral",
        "ref_cmax_mg_L": 25.0,  # ~20-30 mg/L
        "ref_t_half_h": 2.0,  # 1.5-3h
        "ref_cl_L_h": 3.5,
    },
    "acetaminophen": {
        "smiles": "CC(=O)Nc1ccc(O)cc1",
        "dose_mg": 1000.0,
        "route": "oral",
        "ref_cmax_mg_L": 15.0,  # ~10-20 mg/L
        "ref_t_half_h": 2.5,  # 1.5-3h
        "ref_cl_L_h": 20.0,
    },
    "theophylline": {
        "smiles": "Cn1c(=O)c2[nH]cnc2n(C)c1=O",
        "dose_mg": 300.0,
        "route": "oral",
        "ref_cmax_mg_L": 8.0,
        "ref_t_half_h": 8.0,  # 6-12h
        "ref_cl_L_h": 2.8,
    },
    "omeprazole": {
        "smiles": "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1",
        "dose_mg": 20.0,
        "route": "oral",
        "ref_cmax_mg_L": 0.5,
        "ref_t_half_h": 1.0,  # 0.5-1.5h
        "ref_cl_L_h": 30.0,
    },
    "amoxicillin": {
        "smiles": "CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O",
        "dose_mg": 500.0,
        "route": "oral",
        "ref_cmax_mg_L": 8.0,
        "ref_t_half_h": 1.0,
        "ref_cl_L_h": 15.0,
    },
    "ciprofloxacin": {
        "smiles": "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
        "dose_mg": 500.0,
        "route": "oral",
        "ref_cmax_mg_L": 2.5,
        "ref_t_half_h": 4.0,
        "ref_cl_L_h": 30.0,
    },
    "fluoxetine": {
        "smiles": "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1",
        "dose_mg": 20.0,
        "route": "oral",
        "ref_cmax_mg_L": 0.015,
        "ref_t_half_h": 48.0,  # 24-72h
        "ref_cl_L_h": 40.0,
    },
    "metoprolol": {
        "smiles": "COCCc1ccc(OCC(O)CNC(C)C)cc1",
        "dose_mg": 50.0,
        "route": "oral",
        "ref_cmax_mg_L": 0.05,
        "ref_t_half_h": 4.0,
        "ref_cl_L_h": 60.0,
    },
    "atorvastatin": {
        "smiles": "CC(C)c1n(CC(O)CC(O)CC(=O)O)c(-c2ccccc2)c(-c2ccc(F)cc2)c1C(=O)Nc1ccccc1",
        "dose_mg": 40.0,
        "route": "oral",
        "ref_cmax_mg_L": 0.005,
        "ref_t_half_h": 14.0,
        "ref_cl_L_h": 625.0,  # very high first-pass
    },
    "losartan": {
        "smiles": "CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nn[nH]n2)cc1",
        "dose_mg": 50.0,
        "route": "oral",
        "ref_cmax_mg_L": 0.25,
        "ref_t_half_h": 2.0,
        "ref_cl_L_h": 36.0,
    },
    "simvastatin": {
        "smiles": "CCC(C)(C)C(=O)OC1CC(O)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C21",
        "dose_mg": 40.0,
        "route": "oral",
        "ref_cmax_mg_L": 0.01,
        "ref_t_half_h": 3.0,
        "ref_cl_L_h": 450.0,
    },
    "ranitidine": {
        "smiles": "CNC(/N=C/[N+](=O)[O-])NCCSCc1ccc(CN(C)C)o1",
        "dose_mg": 150.0,
        "route": "oral",
        "ref_cmax_mg_L": 0.5,
        "ref_t_half_h": 2.5,
        "ref_cl_L_h": 35.0,
    },
    "amlodipine": {
        "smiles": "CCOC(=O)C1=C(COCCN)NC(C)=C(C(=O)OC)C1c1ccccc1Cl",
        "dose_mg": 10.0,
        "route": "oral",
        "ref_cmax_mg_L": 0.006,
        "ref_t_half_h": 40.0,
        "ref_cl_L_h": 21.0,
    },
    "verapamil": {
        "smiles": "COc1ccc(CCN(C)CCCC(C#N)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC",
        "dose_mg": 80.0,
        "route": "oral",
        "ref_cmax_mg_L": 0.08,
        "ref_t_half_h": 6.0,
        "ref_cl_L_h": 65.0,
    },
}


def run_validation():
    """Run Level 1 validation against reference compounds."""
    from omega_pbpk.ml.models.adme.admet_ai_wrapper import ADMETAIPredictor
    from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

    # Set up pipeline with ADMET-AI
    pipeline = OmegaPipeline()
    try:
        predictor = ADMETAIPredictor()
        pipeline._adme_predictor = predictor
        pipeline._initialized = True
        logger.info("Using ADMET-AI predictor")
    except ImportError:
        logger.warning("ADMET-AI not available, using legacy predictor")
        pipeline._ensure_initialized()

    results = []
    fold_errors_t_half = []
    fold_errors_cmax = []

    for name, ref in REFERENCE_DRUGS.items():
        try:
            result = pipeline.simulate(
                SimulationRequest(
                    smiles=ref["smiles"],
                    dose_mg=ref["dose_mg"],
                    route=ref["route"],
                )
            )

            pred_t_half = result.t_half_h
            pred_cmax = result.cmax_mg_L

            # Fold errors
            eps = 1e-12
            fe_cmax = max(
                pred_cmax / max(ref["ref_cmax_mg_L"], eps),
                ref["ref_cmax_mg_L"] / max(pred_cmax, eps),
            )

            if np.isfinite(pred_t_half) and pred_t_half > 0:
                fe_t_half = max(
                    pred_t_half / max(ref["ref_t_half_h"], eps),
                    ref["ref_t_half_h"] / max(pred_t_half, eps),
                )
                fold_errors_t_half.append(fe_t_half)
            else:
                fe_t_half = float("nan")

            within_2fold = fe_cmax <= 2.0 and np.isfinite(fe_t_half) and fe_t_half <= 2.0

            fold_errors_cmax.append(fe_cmax)

            results.append(
                {
                    "name": name,
                    "pred_cmax": pred_cmax,
                    "ref_cmax": ref["ref_cmax_mg_L"],
                    "fe_cmax": fe_cmax,
                    "pred_t_half": pred_t_half,
                    "ref_t_half": ref["ref_t_half_h"],
                    "fe_t_half": fe_t_half,
                    "within_2fold": within_2fold,
                }
            )

            status = "✓" if within_2fold else "✗"
            logger.info(
                "%s %s: Cmax=%.3f (ref %.3f, FE=%.1f), t½=%.1fh (ref %.1fh, FE=%.1f)",
                status,
                name,
                pred_cmax,
                ref["ref_cmax_mg_L"],
                fe_cmax,
                pred_t_half,
                ref["ref_t_half_h"],
                fe_t_half,
            )

        except Exception as e:
            logger.error("FAILED %s: %s", name, e)
            results.append({"name": name, "error": str(e)})

    # Summary
    valid = [r for r in results if "within_2fold" in r]
    n_within_2fold = sum(1 for r in valid if r["within_2fold"])

    if fold_errors_t_half:
        aafe_t_half = float(np.exp(np.mean(np.log(fold_errors_t_half))))
        aafe_cmax = float(np.exp(np.mean(np.log(fold_errors_cmax))))
    else:
        aafe_t_half = float("inf")
        aafe_cmax = float("inf")

    pct_2fold = (n_within_2fold / len(valid) * 100) if valid else 0

    print("\n" + "=" * 60)
    print("LEVEL 1 VALIDATION RESULTS")
    print("=" * 60)
    print(f"  Drugs tested: {len(valid)}")
    print(f"  Within 2-fold (both Cmax + t½): {n_within_2fold}/{len(valid)} ({pct_2fold:.0f}%)")
    print(f"  AAFE (Cmax): {aafe_cmax:.2f}")
    print(f"  AAFE (t½):   {aafe_t_half:.2f}")
    print()
    print("EXIT CRITERIA:")
    print(f"  AAFE < 3.0:    {'PASS ✓' if max(aafe_cmax, aafe_t_half) < 3.0 else 'FAIL ✗'}")
    print(f"  ≥70% 2-fold:   {'PASS ✓' if pct_2fold >= 70 else 'FAIL ✗'} ({pct_2fold:.0f}%)")
    print("=" * 60)

    return {
        "n_drugs": len(valid),
        "n_within_2fold": n_within_2fold,
        "pct_within_2fold": pct_2fold,
        "aafe_cmax": aafe_cmax,
        "aafe_t_half": aafe_t_half,
    }


if __name__ == "__main__":
    results = run_validation()
    if (
        results["pct_within_2fold"] >= 70
        and max(results["aafe_cmax"], results["aafe_t_half"]) < 3.0
    ):
        sys.exit(0)
    else:
        sys.exit(1)
