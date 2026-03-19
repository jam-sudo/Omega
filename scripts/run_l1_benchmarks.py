#!/usr/bin/env python3
"""Run L1 benchmarks on all 25 benchmark drugs using ML evaluation pipeline.

Uses SMILES → EnsembleADMEPredictor → Drug → WholeBodyPBPK → PK metrics.
Reports AAFE for Cmax/AUC, %2-fold accuracy per drug.
Compares against L1 exit criteria: AAFE<3.0, ≤2-fold for ≥70% of 20+ drugs.
Reports 20-drug dev set and 5-drug validation set AAFEs separately.
"""

import json
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from omega_pbpk.data.drug_registry import BENCHMARK_DRUGS  # noqa: E402


def _load_csv_observed_pk(drug_name: str) -> dict:
    """Load observed PK from synthetic benchmark CSV (fallback)."""
    datasets_dir = repo_root / "benchmarks" / "datasets"
    for csv_file in datasets_dir.glob("*.csv"):
        if drug_name.replace("_", "") in csv_file.stem.replace("_", "").lower():
            data = np.genfromtxt(csv_file, delimiter=",", skip_header=1)
            time_h = data[:, 0]
            conc = data[:, 1]
            cmax = float(conc.max())
            auc = float(np.trapz(conc, time_h))
            tmax_idx = int(np.argmax(conc))
            tmax = float(time_h[tmax_idx])
            return {
                "cmax": cmax,
                "auc": auc,
                "tmax": tmax,
                "file": str(csv_file.name),
                "data_source": "synthetic",
                "data_quality": "synthetic",
            }
    return {}


def load_observed_pk(drug_name: str) -> dict:
    """Load observed Cmax and AUC.

    Priority:
      1. Real clinical reference (data/clinical/gold24_reference_cmax.json)
      2. Synthetic benchmark CSV (benchmarks/datasets/*.csv)

    The clinical reference contains PK-DB / FDA-label Cmax values where available.
    Drugs without a clinical reference entry fall back to the 1-cpt synthetic CSV.
    """
    clinical_ref_file = repo_root / "data" / "clinical" / "gold24_reference_cmax.json"
    if clinical_ref_file.exists():
        with open(clinical_ref_file) as _f:
            clinical_refs = json.load(_f)
        if drug_name in clinical_refs and not drug_name.startswith("_"):
            ref = clinical_refs[drug_name]
            cmax_ref = ref.get("cmax_mg_L")
            if cmax_ref is not None:
                # Use CSV for AUC/tmax (secondary metric, OK to keep synthetic for now)
                csv_data = _load_csv_observed_pk(drug_name)
                result = csv_data.copy()
                result["cmax"] = cmax_ref
                result["data_source"] = ref.get("source", "clinical")
                result["data_quality"] = ref.get("data_quality", "clinical")
                # Override AUC if reference provides it
                if ref.get("auc_mg_h_L") is not None:
                    result["auc"] = ref["auc_mg_h_L"]
                return result

    return _load_csv_observed_pk(drug_name)


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


def main():
    from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

    pipeline = OmegaPipeline()

    all_results = []
    cmax_fold_errors = []
    auc_fold_errors = []
    dev_cmax_fe = []
    dev_auc_fe = []
    val_cmax_fe = []
    val_auc_fe = []

    print("=" * 80)
    print("L1 BENCHMARK: SMILES → OmegaPipeline.simulate() → PK Metrics")
    print("=" * 80)

    for drug_name, info in BENCHMARK_DRUGS.items():
        smiles = info["smiles"]
        dose_mg = info["dose_mg"]

        print(f"\n--- {drug_name} (dose={dose_mg}mg) ---")

        # Load observed PK
        observed = load_observed_pk(drug_name)
        if not observed:
            print(f"  WARNING: No observed data found for {drug_name}")

        try:
            # Use the full OmegaPipeline.simulate() which includes:
            # - ADME prediction (ensemble)
            # - IVIVE (power-law CLh = 0.3 × CLint^0.9)
            # - Berezhkovskiy Kp + RDKit logP
            # - Renal clearance estimation
            # - ODE simulation
            # - Hybrid Cmax selector (geometric mean of ODE + analytical)
            # - Hybrid t½ selector (curve-fit vs analytical)
            # - VDss correction (XGBoost vs Berezhkovskiy)
            sim_result = pipeline.simulate(
                SimulationRequest(
                    smiles=smiles,
                    dose_mg=dose_mg,
                    route="oral",
                    duration_h=24.0,
                )
            )

            adme = sim_result.adme_properties
            print(
                f"  ADME: mw={adme.get('mw', 0):.1f}, logP={adme.get('logP', 0):.2f}, "
                f"fup={adme.get('fup', 0):.4f}, rbp={adme.get('rbp', 0):.3f}, "
                f"clint_3a4={adme.get('clint_3a4', 0):.3f}, peff={adme.get('peff', 0):.3f}"
            )

            pred_cmax = sim_result.cmax_mg_L
            pred_auc = sim_result.auc0t_mg_h_L
            pred_thalf = sim_result.t_half_h

            print(
                f"  Predicted: Cmax={pred_cmax:.4f} mg/L, AUC={pred_auc:.4f} mg*h/L, "
                f"t_half={pred_thalf:.2f}h"
            )

            entry = {
                "drug": drug_name,
                "smiles": smiles,
                "dose_mg": dose_mg,
                "predicted_cmax": pred_cmax,
                "predicted_auc": pred_auc,
                "predicted_thalf": pred_thalf,
                "adme": {
                    "mw": adme.get("mw", 0),
                    "logP": adme.get("logP", 0),
                    "fup": adme.get("fup", 0),
                    "rbp": adme.get("rbp", 0),
                    "clint_3a4": adme.get("clint_3a4", 0),
                    "peff": adme.get("peff", 0),
                },
                "success": True,
            }

            if observed:
                obs_cmax = observed["cmax"]
                obs_auc = observed["auc"]
                fe_cmax = compute_fold_error(pred_cmax, obs_cmax)
                fe_auc = compute_fold_error(pred_auc, obs_auc)
                cmax_fold_errors.append(fe_cmax)
                auc_fold_errors.append(fe_auc)
                is_val = info.get("set") == "validation"
                if is_val:
                    val_cmax_fe.append(fe_cmax)
                    val_auc_fe.append(fe_auc)
                else:
                    dev_cmax_fe.append(fe_cmax)
                    dev_auc_fe.append(fe_auc)
                print(f"  Observed:  Cmax={obs_cmax:.4f} mg/L, AUC={obs_auc:.4f} mg*h/L")
                print(f"  Fold-error: Cmax={fe_cmax:.2f}x, AUC={fe_auc:.2f}x")
                within_2fold = fe_cmax <= 2.0 and fe_auc <= 2.0
                print(f"  Within 2-fold: {'YES' if within_2fold else 'NO'}")
                entry["observed_cmax"] = obs_cmax
                entry["observed_auc"] = obs_auc
                entry["fold_error_cmax"] = fe_cmax
                entry["fold_error_auc"] = fe_auc
                entry["within_2fold"] = within_2fold

            all_results.append(entry)

        except Exception as e:
            print(f"  FAIL: {e}")
            all_results.append(
                {
                    "drug": drug_name,
                    "smiles": smiles,
                    "success": False,
                    "error": str(e),
                }
            )

    # --- Summary ---
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)

    n_success = sum(1 for r in all_results if r.get("success"))
    n_total = len(all_results)
    print(f"Drugs: {n_success}/{n_total} succeeded")

    aafe_cmax = compute_aafe(cmax_fold_errors)
    aafe_auc = compute_aafe(auc_fold_errors)
    valid_cmax = [fe for fe in cmax_fold_errors if not np.isnan(fe)]
    valid_auc = [fe for fe in auc_fold_errors if not np.isnan(fe)]
    pct_2fold_cmax = sum(1 for fe in valid_cmax if fe <= 2.0) / max(len(valid_cmax), 1) * 100
    pct_2fold_auc = sum(1 for fe in valid_auc if fe <= 2.0) / max(len(valid_auc), 1) * 100

    print(
        f"\nCmax: AAFE={aafe_cmax:.2f}, within 2-fold={pct_2fold_cmax:.0f}% ({sum(1 for fe in valid_cmax if fe <= 2.0)}/{len(valid_cmax)})"
    )
    print(
        f"AUC:  AAFE={aafe_auc:.2f}, within 2-fold={pct_2fold_auc:.0f}% ({sum(1 for fe in valid_auc if fe <= 2.0)}/{len(valid_auc)})"
    )

    # Dev vs Validation split
    aafe_dev_cmax = compute_aafe(dev_cmax_fe)
    aafe_dev_auc = compute_aafe(dev_auc_fe)
    aafe_val_cmax = compute_aafe(val_cmax_fe)
    aafe_val_auc = compute_aafe(val_auc_fe)

    print("\n--- DEV SET (20 drugs) ---")
    print(f"  Cmax AAFE: {aafe_dev_cmax:.2f}, AUC AAFE: {aafe_dev_auc:.2f}")
    print("\n--- VALIDATION SET (5 drugs) ---")
    print(f"  Cmax AAFE: {aafe_val_cmax:.2f}, AUC AAFE: {aafe_val_auc:.2f}")

    print("\n--- EXIT CRITERIA CHECK (all 25 drugs) ---")
    print(
        f"  AAFE < 3.0:  Cmax={'PASS' if aafe_cmax < 3.0 else 'FAIL'} ({aafe_cmax:.2f}), AUC={'PASS' if aafe_auc < 3.0 else 'FAIL'} ({aafe_auc:.2f})"
    )
    print(
        f"  ≤2-fold ≥70%: Cmax={'PASS' if pct_2fold_cmax >= 70 else 'FAIL'} ({pct_2fold_cmax:.0f}%), AUC={'PASS' if pct_2fold_auc >= 70 else 'FAIL'} ({pct_2fold_auc:.0f}%)"
    )
    print(f"  20+ drugs:   {'PASS' if len(valid_cmax) >= 20 else 'FAIL'} ({len(valid_cmax)} drugs)")

    # Save
    summary = {
        "n_drugs": n_total,
        "n_success": n_success,
        "n_with_observed": len(valid_cmax),
        "aafe_cmax": aafe_cmax,
        "aafe_auc": aafe_auc,
        "pct_2fold_cmax": pct_2fold_cmax,
        "pct_2fold_auc": pct_2fold_auc,
        "exit_criteria": {
            "aafe_lt_3": aafe_cmax < 3.0 and aafe_auc < 3.0,
            "pct_2fold_gte_70": pct_2fold_cmax >= 70 and pct_2fold_auc >= 70,
            "n_drugs_gte_20": len(valid_cmax) >= 20,
        },
        "per_drug": all_results,
    }

    out_path = repo_root / "outputs" / "l1_benchmark_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
