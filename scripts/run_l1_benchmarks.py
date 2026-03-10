#!/usr/bin/env python3
"""Run L1 benchmarks on all 20 benchmark drugs using ML evaluation pipeline.

Uses SMILES → EnsembleADMEPredictor → Drug → WholeBodyPBPK → PK metrics.
Reports AAFE for Cmax/AUC, %2-fold accuracy per drug.
Compares against L1 exit criteria: AAFE<3.0, ≤2-fold for ≥70% of 20+ drugs.
"""

import sys
import csv
import json
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))


# All 20 benchmark drugs with SMILES and observed PK from CSV datasets
BENCHMARK_DRUGS = {
    "caffeine": {"smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O", "dose_mg": 100},
    "metoprolol": {"smiles": "COCCc1ccc(OCC(O)CNC(C)C)cc1", "dose_mg": 100},
    "midazolam": {"smiles": "Clc1ccc2c(c1)C(=NCc1nccn1C)c1ccccc1N2", "dose_mg": 2},
    "propranolol": {"smiles": "CC(C)NCC(O)COc1cccc2ccccc12", "dose_mg": 80},
    "warfarin": {"smiles": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O", "dose_mg": 5},
    "d_amphetamine": {"smiles": "C[C@@H](N)Cc1ccccc1", "dose_mg": 20},
    "methanol": {"smiles": "CO", "dose_mg": 100},
    "ibuprofen": {"smiles": "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "dose_mg": 400},
    "acetaminophen": {"smiles": "CC(=O)Nc1ccc(O)cc1", "dose_mg": 1000},
    "amoxicillin": {"smiles": "CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O", "dose_mg": 500},
    "atorvastatin": {
        "smiles": "CC(C)c1n(CC[C@@H](O)C[C@@H](O)CC(=O)O)c(-c2ccccc2)c(-c2ccc(F)cc2)c1C(=O)Nc1ccccc1",
        "dose_mg": 40,
    },
    "carbamazepine": {"smiles": "NC(=O)N1c2ccccc2C=Cc2ccccc21", "dose_mg": 200},
    "diazepam": {"smiles": "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21", "dose_mg": 10},
    "digoxin": {
        "smiles": "C[C@@H]1O[C@@H](O[C@@H]2C[C@H](O)[C@@H](O[C@@H]3C[C@H](O)[C@@H](O[C@@H]4C[C@H](O)[C@@H](OC5CC(CO)=CC(=O)O5)C(C)O4)C(C)O3)C(C)O2)C[C@H](O)[C@H]1O",
        "dose_mg": 0.5,
    },
    "fluoxetine": {"smiles": "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1", "dose_mg": 20},
    "nifedipine": {"smiles": "COC(=O)C1=C(C)NC(C)=C(C(=O)OC)C1c1ccccc1[N+](=O)[O-]", "dose_mg": 10},
    "omeprazole": {"smiles": "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1", "dose_mg": 20},
    "phenytoin": {"smiles": "O=C1NC(=O)C(c2ccccc2)(c2ccccc2)N1", "dose_mg": 300},
    "theophylline": {"smiles": "Cn1c(=O)c2[nH]cnc2n(C)c1=O", "dose_mg": 300},
    "verapamil": {"smiles": "COc1ccc(CCN(C)CCCC(C#N)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC", "dose_mg": 80},
}


def load_observed_pk(drug_name: str) -> dict:
    """Load observed Cmax and AUC from benchmark CSV dataset."""
    datasets_dir = repo_root / "benchmarks" / "datasets"
    # Find matching CSV
    for csv_file in datasets_dir.glob("*.csv"):
        if drug_name.replace("_", "") in csv_file.stem.replace("_", "").lower():
            data = np.genfromtxt(csv_file, delimiter=",", skip_header=1)
            time_h = data[:, 0]
            conc = data[:, 1]
            cmax = float(conc.max())
            auc = float(np.trapz(conc, time_h))
            tmax_idx = int(np.argmax(conc))
            tmax = float(time_h[tmax_idx])
            return {"cmax": cmax, "auc": auc, "tmax": tmax, "file": str(csv_file.name)}
    return {}


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
    from omega_pbpk.ml.models.adme.ensemble import EnsembleADMEPredictor
    from omega_pbpk.core.body import WholeBodyPBPK
    from omega_pbpk.drugs.drug import Drug

    predictor = EnsembleADMEPredictor()

    all_results = []
    cmax_fold_errors = []
    auc_fold_errors = []

    print("=" * 80)
    print("L1 BENCHMARK: SMILES → EnsembleADMEPredictor → WholeBodyPBPK → PK Metrics")
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
            # Predict ADME
            adme = predictor.predict(smiles)
            print(
                f"  ADME: mw={adme.mw:.1f}, logP={adme.logP:.2f}, fup={adme.fup:.4f}, "
                f"rbp={adme.rbp:.3f}, clint_3a4={adme.clint_3a4:.3f}, peff={adme.peff:.3f}"
            )

            # Build Drug - use hepatocyte CLint directly if available (ADMET-AI),
            # otherwise use CYP3A4 pmol CLint with standard IVIVE
            clint_hep_raw = getattr(adme, "clint_hepatocyte_uL_min", 0.0)
            if clint_hep_raw and clint_hep_raw > 0.01:
                # Direct hepatocyte IVIVE: µL/min/10^6 cells × hepatocellularity × liver_wt
                # = cl_hep × 120e6 cells/g × 1800g / 1e6 µL/mL / 60 min/h / 1000 mL/L
                clint_L_per_h = clint_hep_raw * 120.0 * 1800.0 / 1e6 / 60.0
                ivive_source = "hepatocyte"
            else:
                # CYP3A4 pmol IVIVE: µL/min/pmol × MPPGL(40) × mg/g(45) × liver(1800g)
                clint_L_per_h = adme.clint_3a4 * 40.0 * 45.0 * 1800.0 / 1e6 / 60.0
                ivive_source = "pmol_cyp"

            print(
                f"  IVIVE: clint_hepatocyte={clint_hep_raw:.3f}, clint_L_per_h={clint_L_per_h:.4f} ({ivive_source})"
            )

            drug = Drug(
                name=drug_name,
                mw=adme.mw,
                logP=adme.logP,
                fup=max(adme.fup, 0.001),
                rbp=adme.rbp,
                clint_hepatic_L_per_h=clint_L_per_h,
                peff=adme.peff,
            )

            # Simulate
            model = WholeBodyPBPK(drug=drug, body_weight=70.0)
            model.setup_oral(dose_mg)
            result = model.simulate(t_end_h=24.0)
            pk = result.pk_summary()

            pred_cmax = pk.get("Cmax_mg_L", float("nan"))
            pred_auc = pk.get("AUC_mg_h_L", float("nan"))
            pred_thalf = pk.get("half_life_h", float("nan"))

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
                    "mw": adme.mw,
                    "logP": adme.logP,
                    "fup": adme.fup,
                    "rbp": adme.rbp,
                    "clint_3a4": adme.clint_3a4,
                    "peff": adme.peff,
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

    print(f"\n--- EXIT CRITERIA CHECK ---")
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
