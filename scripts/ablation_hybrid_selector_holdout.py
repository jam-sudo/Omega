#!/usr/bin/env python3
"""Ablation: Hybrid Cmax selector on holdout set.

Runs holdout benchmark TWICE:
  1. WITH hybrid selector (baseline — current pipeline)
  2. WITHOUT hybrid selector (ODE Cmax only)

Compares AAFE, %2-fold, %3-fold to determine if the selector
generalizes beyond gold-24 or is N=24 overfitting.
"""

import json
import sys
import unittest.mock as mock
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


def run_holdout(pipeline, holdout_drugs, plat, label=""):
    fold_errors = []
    per_drug = []

    for drug_name in sorted(holdout_drugs):
        entry = plat["drugs"].get(drug_name)
        if entry is None:
            continue

        smiles = entry["smiles"]
        dose_mg = entry["dose_mg"]
        obs_cmax = entry["cmax_mg_L"]

        try:
            sim = pipeline.simulate(SimulationRequest(smiles=smiles, dose_mg=dose_mg, route="oral"))
            pred_cmax = sim.cmax_mg_L
            fe = compute_fold_error(pred_cmax, obs_cmax)
        except Exception as e:
            per_drug.append({"drug": drug_name, "fe": float("nan"), "error": str(e)})
            continue

        fold_errors.append(fe)
        per_drug.append(
            {
                "drug": drug_name,
                "fe": round(fe, 4),
                "pred": round(pred_cmax, 6),
                "obs": round(obs_cmax, 6),
            }
        )

    valid_fe = [fe for fe in fold_errors if not np.isnan(fe)]
    log_fe = np.log10(valid_fe)
    aafe = float(10 ** np.mean(np.abs(log_fe)))
    pct_2fold = sum(1 for fe in valid_fe if fe <= 2.0) / len(valid_fe) * 100
    pct_3fold = sum(1 for fe in valid_fe if fe <= 3.0) / len(valid_fe) * 100
    ci_lo, ci_hi = bootstrap_aafe_ci(valid_fe)

    return {
        "label": label,
        "n": len(valid_fe),
        "aafe": round(aafe, 4),
        "ci95": [round(ci_lo, 4), round(ci_hi, 4)],
        "pct_2fold": round(pct_2fold, 1),
        "pct_3fold": round(pct_3fold, 1),
        "per_drug": per_drug,
    }


def disable_hybrid_selector(pipeline):
    """Monkey-patch: make the hybrid selector block a no-op.

    The selector lives in simulate() lines 489-627. It modifies `cmax`
    after the ODE solve. We patch by setting _vdss_predictor to None
    AND wrapping simulate to skip the try block.

    Simplest approach: patch predict_bioavailability to raise, which
    triggers the except clause at line 626-627 and leaves ODE cmax intact.
    """
    import omega_pbpk.prediction.bioavailability_prediction as bio_mod

    original_predict = bio_mod.predict_bioavailability
    bio_mod.predict_bioavailability = mock.Mock(
        side_effect=RuntimeError("ablation: hybrid selector disabled")
    )
    # Also disable VDss predictor to prevent any VDss correction path
    original_vdss = pipeline._vdss_predictor
    pipeline._vdss_predictor = None
    return original_predict, original_vdss


def restore_hybrid_selector(pipeline, original_predict, original_vdss):
    import omega_pbpk.prediction.bioavailability_prediction as bio_mod

    bio_mod.predict_bioavailability = original_predict
    pipeline._vdss_predictor = original_vdss


def main():
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    holdout_drugs = set(split["holdout"])

    with open(PLATINUM_PATH) as f:
        plat = json.load(f)

    pipeline = OmegaPipeline()

    # Run 1: WITH hybrid selector (baseline)
    print("=" * 70)
    print("RUN 1: WITH hybrid selector (baseline)")
    print("=" * 70)
    result_with = run_holdout(pipeline, holdout_drugs, plat, label="WITH_SELECTOR")

    print(
        f"  N={result_with['n']}  AAFE={result_with['aafe']:.3f} "
        f"[{result_with['ci95'][0]:.3f}, {result_with['ci95'][1]:.3f}]  "
        f"%2f={result_with['pct_2fold']:.1f}%  %3f={result_with['pct_3fold']:.1f}%"
    )

    # Run 2: WITHOUT hybrid selector
    print("\n" + "=" * 70)
    print("RUN 2: WITHOUT hybrid selector (ODE Cmax only)")
    print("=" * 70)
    orig_predict, orig_vdss = disable_hybrid_selector(pipeline)
    result_without = run_holdout(pipeline, holdout_drugs, plat, label="WITHOUT_SELECTOR")
    restore_hybrid_selector(pipeline, orig_predict, orig_vdss)

    print(
        f"  N={result_without['n']}  AAFE={result_without['aafe']:.3f} "
        f"[{result_without['ci95'][0]:.3f}, {result_without['ci95'][1]:.3f}]  "
        f"%2f={result_without['pct_2fold']:.1f}%  %3f={result_without['pct_3fold']:.1f}%"
    )

    # Comparison
    delta_aafe = result_without["aafe"] - result_with["aafe"]
    delta_2fold = result_without["pct_2fold"] - result_with["pct_2fold"]

    print("\n" + "=" * 70)
    print("ABLATION COMPARISON")
    print("=" * 70)
    print(f"  {'Metric':<15} {'WITH':>12} {'WITHOUT':>12} {'Delta':>12}")
    print(f"  {'─' * 15} {'─' * 12} {'─' * 12} {'─' * 12}")
    print(
        f"  {'AAFE':<15} {result_with['aafe']:>12.3f} {result_without['aafe']:>12.3f} {delta_aafe:>+12.3f}"
    )
    print(
        f"  {'%2-fold':<15} {result_with['pct_2fold']:>11.1f}% {result_without['pct_2fold']:>11.1f}% {delta_2fold:>+11.1f}%"
    )
    print(
        f"  {'%3-fold':<15} {result_with['pct_3fold']:>11.1f}% {result_without['pct_3fold']:>11.1f}% {result_without['pct_3fold'] - result_with['pct_3fold']:>+11.1f}%"
    )

    # Per-drug delta (biggest movers)
    drug_deltas = []
    with_map = {
        d["drug"]: d
        for d in result_with["per_drug"]
        if "fe" in d and not np.isnan(d.get("fe", float("nan")))
    }
    without_map = {
        d["drug"]: d
        for d in result_without["per_drug"]
        if "fe" in d and not np.isnan(d.get("fe", float("nan")))
    }
    for drug in with_map:
        if drug in without_map:
            d_fe = without_map[drug]["fe"] - with_map[drug]["fe"]
            drug_deltas.append((drug, with_map[drug]["fe"], without_map[drug]["fe"], d_fe))

    drug_deltas.sort(key=lambda x: abs(x[3]), reverse=True)

    print("\nTop 10 biggest movers (|ΔFE|):")
    print(f"  {'Drug':<25} {'FE_with':>10} {'FE_without':>10} {'ΔFE':>10}")
    for drug, fe_w, fe_wo, d_fe in drug_deltas[:10]:
        direction = "worse" if d_fe > 0 else "better"
        print(f"  {drug:<25} {fe_w:>10.2f} {fe_wo:>10.2f} {d_fe:>+10.2f} ({direction})")

    # Interpretation
    print(f"\n{'=' * 70}")
    if delta_aafe > 0.15:
        print("INTERPRETATION: Selector generalizes to holdout (Δ > 0.15).")
        print("Concept is valid. Consider re-tuning thresholds on holdout-inclusive data.")
    elif delta_aafe > 0.05:
        print("INTERPRETATION: Selector has moderate holdout effect (0.05 < Δ < 0.15).")
        print("Some generalization, but thresholds likely overfit to gold-24.")
    else:
        print("INTERPRETATION: Selector has MINIMAL holdout effect (Δ < 0.05).")
        print("Core-24 performance is likely N=24 overfitting. Re-evaluate selector necessity.")
    print("=" * 70)

    # Save
    output = {
        "with_selector": result_with,
        "without_selector": result_without,
        "delta_aafe": round(delta_aafe, 4),
        "delta_pct_2fold": round(delta_2fold, 1),
        "top_movers": [
            {"drug": d, "fe_with": fw, "fe_without": fwo, "delta": round(dfe, 4)}
            for d, fw, fwo, dfe in drug_deltas[:20]
        ],
    }
    out_path = REPO / "outputs" / "ablation_hybrid_selector_holdout.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
