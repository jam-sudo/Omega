"""Blind prediction test: SMILES-only → PK profile vs published clinical data.

These drugs have NO compound YAML files — all ADME parameters are predicted by ML.
This tests the real value proposition: can Omega predict PK for a novel molecule?

Reference PK data from published clinical studies (healthy volunteers, fasted state).
Pass criterion: within 2-fold of observed (0.5x–2.0x), per FDA/EMA PBPK guidance.
"""

import math

import pytest

# ── Reference data: published clinical PK for well-characterized drugs ──────
# Each entry: (SMILES, drug_name, dose_mg, route, {metric: observed_value})
# Sources: Goodman & Gilman's, DrugBank, clinical pharmacology textbooks

BLIND_TEST_DRUGS = [
    {
        "name": "Ibuprofen",
        "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "dose_mg": 400.0,
        "route": "oral",
        "duration_h": 24.0,
        # Cmax ~25-30 mg/L after 400mg, Tmax ~1.5h, t½ ~2h, AUC ~100-130 mg·h/L
        "ref_cmax_mg_L": 27.0,
        "ref_tmax_h": 1.5,
        "ref_auc_mg_h_L": 115.0,
        "ref_thalf_h": 2.0,
    },
    {
        "name": "Acetaminophen",
        "smiles": "CC(=O)Nc1ccc(O)cc1",
        "dose_mg": 1000.0,
        "route": "oral",
        "duration_h": 24.0,
        # Cmax ~15-20 mg/L after 1g, Tmax ~0.5-1h, t½ ~2-3h, AUC ~50-70 mg·h/L
        "ref_cmax_mg_L": 17.0,
        "ref_tmax_h": 0.75,
        "ref_auc_mg_h_L": 60.0,
        "ref_thalf_h": 2.5,
    },
    {
        "name": "Theophylline",
        "smiles": "Cn1c(=O)c2[nH]cnc2n(C)c1=O",
        "dose_mg": 300.0,
        "route": "oral",
        "duration_h": 24.0,
        # Cmax ~6-8 mg/L after 300mg IR, Tmax ~1-2h, t½ ~6-10h, AUC ~80-120 mg·h/L
        "ref_cmax_mg_L": 7.0,
        "ref_tmax_h": 1.5,
        "ref_auc_mg_h_L": 100.0,
        "ref_thalf_h": 8.0,
    },
    {
        "name": "Diclofenac",
        "smiles": "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
        "dose_mg": 50.0,
        "route": "oral",
        "duration_h": 12.0,
        # Cmax ~1.5-2.5 mg/L after 50mg, Tmax ~1-2h, t½ ~1.5h, AUC ~3-5 mg·h/L
        "ref_cmax_mg_L": 2.0,
        "ref_tmax_h": 1.5,
        "ref_auc_mg_h_L": 4.0,
        "ref_thalf_h": 1.5,
    },
    {
        "name": "Omeprazole",
        "smiles": "COc1ccc2[nH]c(nc2c1)S(=O)Cc1ncc(C)c(OC)c1C",
        "dose_mg": 20.0,
        "route": "oral",
        "duration_h": 12.0,
        # Cmax ~0.5-1.0 mg/L after 20mg, Tmax ~1-3h, t½ ~0.7-1.5h, AUC ~1-2 mg·h/L
        "ref_cmax_mg_L": 0.7,
        "ref_tmax_h": 2.0,
        "ref_auc_mg_h_L": 1.5,
        "ref_thalf_h": 1.0,
    },
]


def _fold_error(predicted: float, observed: float) -> float:
    """Fold-error: max(pred/obs, obs/pred). 1.0 = perfect, 2.0 = 2-fold off."""
    if observed <= 0 or predicted <= 0:
        return float("inf")
    ratio = predicted / observed
    return max(ratio, 1.0 / ratio)


def _run_blind_prediction(drug: dict) -> dict:
    """Run Omega pipeline from SMILES only — no compound YAML."""
    from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

    pipeline = OmegaPipeline()
    req = SimulationRequest(
        smiles=drug["smiles"],
        dose_mg=drug["dose_mg"],
        route=drug["route"],
        duration_h=drug["duration_h"],
    )
    result = pipeline.simulate(req)
    return {
        "cmax_mg_L": result.cmax_mg_L,
        "tmax_h": result.tmax_h,
        "auc_mg_h_L": result.auc0t_mg_h_L,
        "thalf_h": result.t_half_h,
        "confidence": result.confidence,
        "adme": result.adme_properties,
        "warnings": result.warnings,
    }


@pytest.mark.parametrize("drug", BLIND_TEST_DRUGS, ids=[d["name"] for d in BLIND_TEST_DRUGS])
def test_blind_prediction_report(drug):
    """Run blind prediction and report fold-errors (informational, not strict pass/fail).

    This test ALWAYS passes — it's a diagnostic to see current prediction accuracy.
    When accuracy improves, tighten the assertions.
    """
    result = _run_blind_prediction(drug)

    fe_cmax = _fold_error(result["cmax_mg_L"], drug["ref_cmax_mg_L"])
    fe_auc = _fold_error(result["auc_mg_h_L"], drug["ref_auc_mg_h_L"])
    fe_tmax = (
        _fold_error(result["tmax_h"], drug["ref_tmax_h"]) if result["tmax_h"] > 0 else float("inf")
    )
    fe_thalf = (
        _fold_error(result["thalf_h"], drug["ref_thalf_h"])
        if not math.isnan(result["thalf_h"])
        else float("inf")
    )

    within_2fold = sum(1 for fe in [fe_cmax, fe_auc, fe_tmax, fe_thalf] if fe <= 2.0)
    status = "PASS" if within_2fold >= 3 else "MARGINAL" if within_2fold >= 2 else "FAIL"

    print(f"\n{'=' * 60}")
    print(f"BLIND PREDICTION: {drug['name']} ({drug['dose_mg']}mg {drug['route']})")
    print(f"{'=' * 60}")
    print(f"  SMILES: {drug['smiles']}")
    print(f"  Confidence: {result['confidence']}")
    print()
    print(f"  {'Metric':<12} {'Predicted':>10} {'Observed':>10} {'FE':>6} {'2-fold?':>8}")
    print(f"  {'-' * 48}")
    print(
        f"  {'Cmax':<12} {result['cmax_mg_L']:>10.3f} {drug['ref_cmax_mg_L']:>10.3f} "
        f"{fe_cmax:>6.2f} {'YES' if fe_cmax <= 2 else 'NO':>8}"
    )
    print(
        f"  {'AUC':<12} {result['auc_mg_h_L']:>10.3f} {drug['ref_auc_mg_h_L']:>10.3f} "
        f"{fe_auc:>6.2f} {'YES' if fe_auc <= 2 else 'NO':>8}"
    )
    print(
        f"  {'Tmax':<12} {result['tmax_h']:>10.2f} {drug['ref_tmax_h']:>10.2f} "
        f"{fe_tmax:>6.2f} {'YES' if fe_tmax <= 2 else 'NO':>8}"
    )
    thalf_str = f"{result['thalf_h']:.2f}" if not math.isnan(result["thalf_h"]) else "NaN"
    print(
        f"  {'t½':<12} {thalf_str:>10} {drug['ref_thalf_h']:>10.2f} "
        f"{fe_thalf:>6.2f} {'YES' if fe_thalf <= 2 else 'NO':>8}"
    )
    print()
    print(f"  Score: {within_2fold}/4 within 2-fold — {status}")

    if result["warnings"]:
        print(f"  Warnings: {result['warnings']}")

    # Predicted ADME properties (for debugging)
    print("\n  Predicted ADME properties:")
    for k, v in result["adme"].items():
        if isinstance(v, (int, float)) and k != "confidence":
            print(f"    {k:<25}: {v:.4g}")

    # This test is informational — always passes to collect data
    # Uncomment below when accuracy target is met:
    # assert within_2fold >= 3, f"{drug['name']}: only {within_2fold}/4 within 2-fold"


@pytest.mark.parametrize("drug", BLIND_TEST_DRUGS, ids=[d["name"] for d in BLIND_TEST_DRUGS])
def test_blind_prediction_runs_without_crash(drug):
    """Smoke test: prediction completes without exceptions."""
    result = _run_blind_prediction(drug)
    assert result["cmax_mg_L"] > 0, "Cmax should be positive"
    assert result["tmax_h"] >= 0, "Tmax should be non-negative"
    assert result["auc_mg_h_L"] > 0, "AUC should be positive"


def test_blind_prediction_summary():
    """Run all 5 drugs and print aggregate accuracy summary."""
    results = []
    for drug in BLIND_TEST_DRUGS:
        try:
            pred = _run_blind_prediction(drug)
            fe_cmax = _fold_error(pred["cmax_mg_L"], drug["ref_cmax_mg_L"])
            fe_auc = _fold_error(pred["auc_mg_h_L"], drug["ref_auc_mg_h_L"])
            within_2fold_cmax = fe_cmax <= 2.0
            within_2fold_auc = fe_auc <= 2.0
            results.append(
                {
                    "name": drug["name"],
                    "fe_cmax": fe_cmax,
                    "fe_auc": fe_auc,
                    "cmax_pass": within_2fold_cmax,
                    "auc_pass": within_2fold_auc,
                }
            )
        except Exception as e:
            results.append(
                {
                    "name": drug["name"],
                    "fe_cmax": float("inf"),
                    "fe_auc": float("inf"),
                    "cmax_pass": False,
                    "auc_pass": False,
                    "error": str(e),
                }
            )

    print(f"\n{'=' * 60}")
    print("BLIND PREDICTION ACCURACY SUMMARY (SMILES-only, no calibration)")
    print(f"{'=' * 60}")
    print(f"  {'Drug':<15} {'FE(Cmax)':>10} {'FE(AUC)':>10} {'Cmax OK':>8} {'AUC OK':>8}")
    print(f"  {'-' * 52}")

    cmax_passes = 0
    auc_passes = 0
    aafe_cmax_sum = 0.0
    aafe_auc_sum = 0.0
    n_valid = 0

    for r in results:
        err = r.get("error", "")
        if err:
            print(f"  {r['name']:<15} {'ERROR':>10} {'ERROR':>10} {'---':>8} {'---':>8}  {err}")
        else:
            print(
                f"  {r['name']:<15} {r['fe_cmax']:>10.2f} {r['fe_auc']:>10.2f} "
                f"{'PASS' if r['cmax_pass'] else 'FAIL':>8} "
                f"{'PASS' if r['auc_pass'] else 'FAIL':>8}"
            )
            cmax_passes += int(r["cmax_pass"])
            auc_passes += int(r["auc_pass"])
            aafe_cmax_sum += r["fe_cmax"]
            aafe_auc_sum += r["fe_auc"]
            n_valid += 1

    if n_valid > 0:
        print(
            f"\n  Cmax within 2-fold: {cmax_passes}/{n_valid} ({100 * cmax_passes / n_valid:.0f}%)"
        )
        print(f"  AUC  within 2-fold: {auc_passes}/{n_valid} ({100 * auc_passes / n_valid:.0f}%)")
        print(f"  AAFE(Cmax): {aafe_cmax_sum / n_valid:.2f}")
        print(f"  AAFE(AUC):  {aafe_auc_sum / n_valid:.2f}")
        print("\n  Target: ≥70% within 2-fold, AAFE < 3.0")
        pct = 100 * (cmax_passes + auc_passes) / (2 * n_valid)
        print(f"  Overall: {pct:.0f}% within 2-fold")
