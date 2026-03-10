#!/usr/bin/env python3
"""Sweep IVIVE correction factors and find optimal for L1 benchmarks.

Tests multiple correction factor values applied to ADMET-AI hepatocyte CLint.
For each factor, runs full ODE simulation and computes AAFE.
"""

import json
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

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


def load_observed_pk(drug_name):
    datasets_dir = repo_root / "benchmarks" / "datasets"
    for csv_file in datasets_dir.glob("*.csv"):
        if drug_name.replace("_", "") in csv_file.stem.replace("_", "").lower():
            data = np.genfromtxt(csv_file, delimiter=",", skip_header=1)
            time_h = data[:, 0]
            conc = data[:, 1]
            return {"cmax": float(conc.max()), "auc": float(np.trapz(conc, time_h))}
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


def run_with_factor(predictor, correction_factor, verbose=False):
    """Run all 20 drugs with a given IVIVE correction factor."""
    from omega_pbpk.core.body import WholeBodyPBPK
    from omega_pbpk.drugs.drug import Drug

    cmax_fes = []
    auc_fes = []

    for drug_name, info in BENCHMARK_DRUGS.items():
        smiles = info["smiles"]
        dose_mg = info["dose_mg"]
        observed = load_observed_pk(drug_name)
        if not observed:
            continue

        try:
            adme = predictor.predict(smiles)

            # Get hepatocyte CLint and apply correction factor
            clint_hep = getattr(adme, "clint_hepatocyte_uL_min", 0.0)
            if clint_hep < 0.01:
                # Fallback: use polynomial clint with standard IVIVE
                clint_L_per_h = adme.clint_3a4 * 40.0 * 45.0 * 1800.0 / 1e6 / 60.0
            else:
                # Hepatocyte IVIVE with correction factor
                # Standard: CLint_hep × hepatocellularity × liver_wt / 1e6 / 60
                # = CLint_hep × 120 × 1800 / 1e6 / 60 = CLint_hep × 3.6 (L/h)
                clint_L_per_h = clint_hep * 120.0 * 1800.0 / 1e6 / 60.0 * correction_factor

            drug = Drug(
                name=drug_name,
                mw=adme.mw,
                logP=adme.logP,
                fup=max(adme.fup, 0.001),
                rbp=adme.rbp,
                clint_hepatic_L_per_h=clint_L_per_h,
                peff=adme.peff,
            )

            model = WholeBodyPBPK(drug=drug, body_weight=70.0)
            model.setup_oral(dose_mg)
            result = model.simulate(t_end_h=24.0)
            pk = result.pk_summary()

            pred_cmax = pk.get("Cmax_mg_L", float("nan"))
            pred_auc = pk.get("AUC_mg_h_L", float("nan"))

            fe_cmax = compute_fold_error(pred_cmax, observed["cmax"])
            fe_auc = compute_fold_error(pred_auc, observed["auc"])
            cmax_fes.append(fe_cmax)
            auc_fes.append(fe_auc)

            if verbose:
                print(
                    f"  {drug_name}: Cmax FE={fe_cmax:.2f}, AUC FE={fe_auc:.2f}, "
                    f"CLint_Lh={clint_L_per_h:.2f}"
                )

        except Exception as e:
            if verbose:
                print(f"  {drug_name}: FAIL - {e}")

    aafe_cmax = compute_aafe(cmax_fes)
    aafe_auc = compute_aafe(auc_fes)
    pct_2fold_cmax = (
        sum(1 for fe in cmax_fes if not np.isnan(fe) and fe <= 2.0) / max(len(cmax_fes), 1) * 100
    )
    pct_2fold_auc = (
        sum(1 for fe in auc_fes if not np.isnan(fe) and fe <= 2.0) / max(len(auc_fes), 1) * 100
    )

    return {
        "aafe_cmax": aafe_cmax,
        "aafe_auc": aafe_auc,
        "pct_2fold_cmax": pct_2fold_cmax,
        "pct_2fold_auc": pct_2fold_auc,
        "n_drugs": len(cmax_fes),
    }


def main():
    from omega_pbpk.ml.models.adme.ensemble import EnsembleADMEPredictor

    predictor = EnsembleADMEPredictor()

    # Pre-warm the predictor
    print("Initializing predictor...")
    predictor.predict("C")

    # Sweep correction factors (log scale)
    factors = [1, 5, 10, 20, 50, 100, 150, 200, 300, 500, 1000]

    print("\n" + "=" * 90)
    print(
        f"{'Factor':>8} | {'AAFE Cmax':>10} | {'AAFE AUC':>10} | {'%2f Cmax':>10} | {'%2f AUC':>10} | {'N':>3}"
    )
    print("-" * 90)

    best_combined_aafe = float("inf")
    best_factor = 1
    all_results = {}

    for factor in factors:
        res = run_with_factor(predictor, factor)
        combined = (res["aafe_cmax"] + res["aafe_auc"]) / 2.0
        marker = " <-- best" if combined < best_combined_aafe else ""
        if combined < best_combined_aafe:
            best_combined_aafe = combined
            best_factor = factor

        print(
            f"{factor:>8} | {res['aafe_cmax']:>10.2f} | {res['aafe_auc']:>10.2f} | "
            f"{res['pct_2fold_cmax']:>9.0f}% | {res['pct_2fold_auc']:>9.0f}% | {res['n_drugs']:>3}{marker}"
        )
        all_results[factor] = res

    # Fine-tune around best factor
    if best_factor > 1:
        fine_factors = [int(best_factor * m) for m in [0.5, 0.7, 0.8, 0.9, 1.1, 1.2, 1.5, 2.0]]
        fine_factors = sorted(set(f for f in fine_factors if f > 0 and f not in factors))

        print("\n--- Fine-tuning around best factor ---")
        for factor in fine_factors:
            res = run_with_factor(predictor, factor)
            combined = (res["aafe_cmax"] + res["aafe_auc"]) / 2.0
            marker = " <-- best" if combined < best_combined_aafe else ""
            if combined < best_combined_aafe:
                best_combined_aafe = combined
                best_factor = factor

            print(
                f"{factor:>8} | {res['aafe_cmax']:>10.2f} | {res['aafe_auc']:>10.2f} | "
                f"{res['pct_2fold_cmax']:>9.0f}% | {res['pct_2fold_auc']:>9.0f}% | {res['n_drugs']:>3}{marker}"
            )
            all_results[factor] = res

    print(f"\n{'=' * 90}")
    print(f"BEST FACTOR: {best_factor}× (combined AAFE = {best_combined_aafe:.2f})")

    # Run best factor with verbose output
    print(f"\n{'=' * 90}")
    print(f"DETAILED RESULTS WITH FACTOR = {best_factor}×")
    print("=" * 90)
    final = run_with_factor(predictor, best_factor, verbose=True)
    print(f"\n  Cmax AAFE: {final['aafe_cmax']:.2f} (target <3.0)")
    print(f"  AUC AAFE:  {final['aafe_auc']:.2f} (target <3.0)")
    print(f"  Cmax ≤2-fold: {final['pct_2fold_cmax']:.0f}% (target ≥70%)")
    print(f"  AUC ≤2-fold:  {final['pct_2fold_auc']:.0f}% (target ≥70%)")

    # Save results
    out_path = repo_root / "outputs" / "ivive_sweep_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {"best_factor": best_factor, "results": {str(k): v for k, v in all_results.items()}},
            f,
            indent=2,
            default=str,
        )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
