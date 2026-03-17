#!/usr/bin/env python3
"""Measured-ADME ablation: compare predicted vs measured ADME parameters.

For drugs present in BOTH the gold-tier reference database (with observed Cmax)
and the ADME reference CSV (with measured ADME values), this script runs two
simulations:
  A: pipeline with measured ADME values injected (bypassing ML prediction)
  B: pipeline with predicted ADME (normal pipeline)

If AAFE_A > AAFE_B, it suggests error cancellation in the predicted pipeline --
i.e., ML ADME errors are compensating for ODE/IVIVE structural errors.

Output: outputs/diagnostic_report_measured_ablation.json
"""

import csv
import json
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "scripts"))

from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest  # noqa: E402


def compute_fold_error(pred, obs):
    if abs(pred) < 1e-12 or abs(obs) < 1e-12:
        return float("nan")
    ratio = pred / obs
    return max(ratio, 1.0 / ratio)


def compute_aafe(fold_errors):
    valid = [fe for fe in fold_errors if not np.isnan(fe) and fe > 0]
    if not valid:
        return float("nan")
    return float(10.0 ** np.mean(np.log10(valid)))


def load_adme_reference():
    """Load measured ADME values from data/adme_reference.csv."""
    adme_path = repo_root / "data" / "adme_reference.csv"
    adme = {}
    with open(adme_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"].strip().lower()
            # Skip rows with missing smiles
            if not row.get("smiles", "").strip():
                continue
            adme[name] = {
                "smiles": row["smiles"].strip(),
                "mw": float(row["mw"]) if row.get("mw") else None,
                "logP": float(row["logP"]) if row.get("logP") else None,
                "fup": float(row["fup"]) if row.get("fup") else None,
                "rbp": float(row["rbp"]) if row.get("rbp") else None,
                "clint_3a4_uL_min_pmol": float(row["clint_3a4_uL_min_pmol"])
                if row.get("clint_3a4_uL_min_pmol")
                else None,
                "peff_cm_s": float(row["peff_cm_s"]) if row.get("peff_cm_s") else None,
            }
    return adme


def load_gold_drugs():
    """Load gold/platinum tier drugs with Cmax from reference database."""
    ref_path = repo_root / "data" / "clinical" / "reference_database.json"
    with open(ref_path) as f:
        db = json.load(f)
    gold = {}
    for name, d in db["drugs"].items():
        tier = d.get("tier", "")
        pk = d.get("pk_params", {})
        dose = d.get("dose_mg")
        if tier in ("gold", "platinum") and "cmax_mg_L" in pk and dose is not None:
            smiles = d.get("smiles", "")
            if smiles:
                gold[name] = {
                    "smiles": smiles,
                    "dose_mg": float(dose),
                    "route": d.get("route", "oral"),
                    "obs_cmax": pk["cmax_mg_L"],
                    "obs_auc": pk.get("auc_mg_h_L"),
                }
    return gold


def make_measured_pipeline(measured_adme_dict):
    """Create a pipeline that injects measured ADME values for known drugs.

    measured_adme_dict: {smiles_lower: {mw, logP, fup, rbp, clint_3a4, peff}}
    """
    p = OmegaPipeline()
    p._ensure_initialized()

    original_predict_adme = p._predict_adme.__func__

    def predict_adme_with_override(self, smiles, warnings_list):
        """Try measured ADME first, fall back to ML prediction."""
        key = smiles.strip()
        measured = measured_adme_dict.get(key)
        if measured is not None:
            warnings_list.append("Using MEASURED ADME values")
            return measured
        # Fall back to normal prediction
        return original_predict_adme(self, smiles, warnings_list)

    p._predict_adme = types.MethodType(predict_adme_with_override, p)
    return p


def main():
    print("=" * 80)
    print("MEASURED-ADME ABLATION STUDY")
    print("=" * 80)

    # Load data sources
    adme_ref = load_adme_reference()
    gold_drugs = load_gold_drugs()

    print(f"ADME reference: {len(adme_ref)} drugs")
    print(f"Gold/Platinum with Cmax: {len(gold_drugs)} drugs")

    # Find overlapping drugs
    overlap = []
    for name, gd in gold_drugs.items():
        name_lower = name.lower().replace(" ", "_").replace("-", "_")
        if name_lower in adme_ref:
            overlap.append((name, name_lower, gd, adme_ref[name_lower]))

    print(f"Overlapping drugs: {len(overlap)}")
    if not overlap:
        print("ERROR: No overlapping drugs found!")
        sys.exit(1)

    # Build measured ADME lookup keyed by SMILES (from gold_drugs)
    measured_by_smiles = {}
    for _name, _name_lower, gd, meas in overlap:
        smiles = gd["smiles"]
        # Convert measured ADME to the format expected by _predict_adme
        # ADME reference peff is in cm/s; pipeline expects 1e-4 cm/s units
        # e.g., 2.8e-4 cm/s -> 2.8 in pipeline units
        peff_1e4 = meas["peff_cm_s"] * 1e4 if meas["peff_cm_s"] is not None else 1.0
        logP_val = meas["logP"] if meas["logP"] is not None else 2.0
        adme_dict = {
            "mw": meas["mw"] or 300.0,
            "logP": logP_val,
            "logS": 0.5 - logP_val,  # GSE estimate: logS ~ 0.5 - logP
            "fup": meas["fup"] if meas["fup"] is not None else 0.1,
            "rbp": meas["rbp"] if meas["rbp"] is not None else 0.55,
            "clint_3a4": meas["clint_3a4_uL_min_pmol"]
            if meas["clint_3a4_uL_min_pmol"] is not None
            else 5.0,
            "peff": peff_1e4,
            "herg_ic50_uM": 100.0,  # Not in reference; use safe default
            "confidence": "high",
        }
        # Also provide hepatocyte CLint for the IVIVE path
        # Convert clint_3a4 (uL/min/pmol CYP) to approximate hepatocyte CLint
        # using typical CYP3A4 abundance: ~130 pmol/mg protein, 45 mg/g liver
        # CLint_hep (uL/min/10^6) ~ clint_3a4 * 130 * 45 / 10^6 * scaling
        # Simplified: use direct mapping, let pipeline handle it
        if meas["clint_3a4_uL_min_pmol"] is not None:
            # The pipeline uses XGBoost CLint (hepatocyte) when available.
            # For measured ADME, provide the clint_3a4 value and let the
            # legacy CYP-attributed IVIVE path handle it.
            # We set clint_hepatocyte_uL_min to 0 to force the legacy path.
            adme_dict["clint_hepatocyte_uL_min"] = 0.0
        measured_by_smiles[smiles] = adme_dict

    # Create pipelines
    pipeline_predicted = OmegaPipeline()
    pipeline_measured = make_measured_pipeline(measured_by_smiles)

    # Run both pipelines
    per_drug_comparison = []

    fe_pred_list = []
    fe_meas_list = []

    for name, _name_lower, gd, meas in overlap:
        smiles = gd["smiles"]
        dose_mg = gd["dose_mg"]
        obs_cmax = gd["obs_cmax"]

        print(f"\n--- {name} (dose={dose_mg}mg, obs_cmax={obs_cmax:.4f} mg/L) ---")

        # A: Predicted ADME (normal pipeline)
        try:
            t0 = time.time()
            res_pred = pipeline_predicted.simulate(
                SimulationRequest(smiles=smiles, dose_mg=dose_mg, route="oral", duration_h=24.0)
            )
            dt_pred = time.time() - t0
            pred_cmax = res_pred.cmax_mg_L
            fe_pred = compute_fold_error(pred_cmax, obs_cmax)
            print(f"  Predicted ADME: Cmax={pred_cmax:.4f}, FE={fe_pred:.2f}x ({dt_pred:.1f}s)")
            print(
                f"    ADME: logP={res_pred.adme_properties.get('logP', 0):.2f}, "
                f"fup={res_pred.adme_properties.get('fup', 0):.4f}, "
                f"clint={res_pred.adme_properties.get('clint_3a4', 0):.3f}, "
                f"peff={res_pred.adme_properties.get('peff', 0):.3f}"
            )
        except Exception as e:
            print(f"  Predicted FAIL: {e}")
            pred_cmax = None
            fe_pred = float("nan")

        # B: Measured ADME
        try:
            t0 = time.time()
            res_meas = pipeline_measured.simulate(
                SimulationRequest(smiles=smiles, dose_mg=dose_mg, route="oral", duration_h=24.0)
            )
            dt_meas = time.time() - t0
            meas_cmax = res_meas.cmax_mg_L
            fe_meas = compute_fold_error(meas_cmax, obs_cmax)
            print(f"  Measured ADME:  Cmax={meas_cmax:.4f}, FE={fe_meas:.2f}x ({dt_meas:.1f}s)")
            print(
                f"    ADME: logP={meas.get('logP', '?')}, "
                f"fup={meas.get('fup', '?')}, "
                f"clint={meas.get('clint_3a4_uL_min_pmol', '?')}, "
                f"peff={meas.get('peff_cm_s', '?')}"
            )
        except Exception as e:
            print(f"  Measured FAIL: {e}")
            meas_cmax = None
            fe_meas = float("nan")

        if not np.isnan(fe_pred) and fe_pred > 0:
            fe_pred_list.append(fe_pred)
        if not np.isnan(fe_meas) and fe_meas > 0:
            fe_meas_list.append(fe_meas)

        # Determine which is better
        if not np.isnan(fe_pred) and not np.isnan(fe_meas):
            if fe_meas < fe_pred:
                winner = "MEASURED"
            elif fe_pred < fe_meas:
                winner = "PREDICTED"
            else:
                winner = "TIE"
        else:
            winner = "N/A"

        entry = {
            "drug": name,
            "obs_cmax": obs_cmax,
            "pred_cmax": round(pred_cmax, 6) if pred_cmax else None,
            "meas_cmax": round(meas_cmax, 6) if meas_cmax else None,
            "fe_predicted": round(fe_pred, 4) if not np.isnan(fe_pred) else None,
            "fe_measured": round(fe_meas, 4) if not np.isnan(fe_meas) else None,
            "winner": winner,
            "measured_adme": {
                "logP": meas.get("logP"),
                "fup": meas.get("fup"),
                "clint_3a4": meas.get("clint_3a4_uL_min_pmol"),
                "peff_cm_s": meas.get("peff_cm_s"),
                "rbp": meas.get("rbp"),
            },
        }
        per_drug_comparison.append(entry)

    # Aggregate metrics
    aafe_pred = compute_aafe(fe_pred_list)
    aafe_meas = compute_aafe(fe_meas_list)
    pct_2fold_pred = (
        100 * sum(1 for f in fe_pred_list if f <= 2.0) / len(fe_pred_list) if fe_pred_list else 0
    )
    pct_2fold_meas = (
        100 * sum(1 for f in fe_meas_list if f <= 2.0) / len(fe_meas_list) if fe_meas_list else 0
    )

    n_meas_wins = sum(1 for d in per_drug_comparison if d["winner"] == "MEASURED")
    n_pred_wins = sum(1 for d in per_drug_comparison if d["winner"] == "PREDICTED")
    n_tie = sum(1 for d in per_drug_comparison if d["winner"] == "TIE")

    # Error cancellation assessment
    if aafe_pred < aafe_meas:
        cancellation_verdict = "ERROR_CANCELLATION_CONFIRMED"
        cancellation_detail = (
            f"Predicted ADME (AAFE={aafe_pred:.3f}) beats measured ADME (AAFE={aafe_meas:.3f}). "
            f"ML prediction errors are compensating for ODE/IVIVE structural biases."
        )
    elif aafe_meas < aafe_pred:
        cancellation_verdict = "NO_ERROR_CANCELLATION"
        cancellation_detail = (
            f"Measured ADME (AAFE={aafe_meas:.3f}) beats predicted ADME (AAFE={aafe_pred:.3f}). "
            f"ML ADME errors are genuinely degrading predictions -- improving ADME accuracy "
            f"would improve PK predictions."
        )
    else:
        cancellation_verdict = "EQUIVALENT"
        cancellation_detail = "Measured and predicted ADME give equivalent PK accuracy."

    # Build report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "description": "Measured vs predicted ADME ablation study",
        "n_drugs": len(overlap),
        "summary": {
            "predicted_adme": {
                "aafe_cmax": round(aafe_pred, 4),
                "pct_2fold_cmax": round(pct_2fold_pred, 1),
                "n_valid": len(fe_pred_list),
            },
            "measured_adme": {
                "aafe_cmax": round(aafe_meas, 4),
                "pct_2fold_cmax": round(pct_2fold_meas, 1),
                "n_valid": len(fe_meas_list),
            },
            "measured_wins": n_meas_wins,
            "predicted_wins": n_pred_wins,
            "ties": n_tie,
        },
        "error_cancellation": {
            "verdict": cancellation_verdict,
            "detail": cancellation_detail,
        },
        "per_drug": per_drug_comparison,
    }

    # Save
    out_path = repo_root / "outputs" / "diagnostic_report_measured_ablation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("MEASURED-ADME ABLATION SUMMARY")
    print("=" * 80)
    print(f"Drugs compared: {len(overlap)}")
    print(f"\n{'Method':20s} {'AAFE Cmax':>10s} {'%2-fold':>8s}")
    print("-" * 40)
    print(f"{'Predicted ADME':20s} {aafe_pred:10.3f} {pct_2fold_pred:7.0f}%")
    print(f"{'Measured ADME':20s} {aafe_meas:10.3f} {pct_2fold_meas:7.0f}%")
    print(f"\nMeasured wins: {n_meas_wins}  |  Predicted wins: {n_pred_wins}  |  Ties: {n_tie}")
    print(f"\nVerdict: {cancellation_verdict}")
    print(f"  {cancellation_detail}")

    # Per-drug table
    print("\n" + "=" * 100)
    print(f"{'Drug':20s} {'Obs Cmax':>10s} {'Pred FE':>10s} {'Meas FE':>10s} {'Winner':>12s}")
    print("-" * 100)
    for d in sorted(
        per_drug_comparison,
        key=lambda x: abs((x["fe_predicted"] or 0) - (x["fe_measured"] or 0)),
        reverse=True,
    ):
        fe_p = f"{d['fe_predicted']:.2f}x" if d["fe_predicted"] else "FAIL"
        fe_m = f"{d['fe_measured']:.2f}x" if d["fe_measured"] else "FAIL"
        print(f"{d['drug']:20s} {d['obs_cmax']:10.4f} {fe_p:>10s} {fe_m:>10s} {d['winner']:>12s}")


if __name__ == "__main__":
    main()
