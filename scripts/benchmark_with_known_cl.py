#!/usr/bin/env python3
"""L1 benchmark using known in-vivo hepatic CL for each drug.

This tests the ODE engine + ADME pipeline when clearance is correct,
establishing the performance ceiling for L1.
"""

import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

# 20 benchmark drugs with known in-vivo hepatic clearance (L/h)
BENCHMARK_DRUGS = {
    "caffeine": {"smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O", "dose_mg": 100, "cl_h": 6.0},
    "metoprolol": {"smiles": "COCCc1ccc(OCC(O)CNC(C)C)cc1", "dose_mg": 100, "cl_h": 63.0},
    "midazolam": {"smiles": "Clc1ccc2c(c1)C(=NCc1nccn1C)c1ccccc1N2", "dose_mg": 2, "cl_h": 27.0},
    "propranolol": {"smiles": "CC(C)NCC(O)COc1cccc2ccccc12", "dose_mg": 80, "cl_h": 60.0},
    "warfarin": {"smiles": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O", "dose_mg": 5, "cl_h": 0.2},
    "d_amphetamine": {"smiles": "C[C@@H](N)Cc1ccccc1", "dose_mg": 20, "cl_h": 20.0},
    "methanol": {"smiles": "CO", "dose_mg": 100, "cl_h": 5.0},
    "ibuprofen": {"smiles": "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "dose_mg": 400, "cl_h": 3.6},
    "acetaminophen": {"smiles": "CC(=O)Nc1ccc(O)cc1", "dose_mg": 1000, "cl_h": 21.0},
    "amoxicillin": {
        "smiles": "CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O",
        "dose_mg": 500,
        "cl_h": 10.0,
    },
    "atorvastatin": {
        "smiles": "CC(C)c1n(CC[C@@H](O)C[C@@H](O)CC(=O)O)c(-c2ccccc2)c(-c2ccc(F)cc2)c1C(=O)Nc1ccccc1",
        "dose_mg": 40,
        "cl_h": 38.0,
    },
    "carbamazepine": {"smiles": "NC(=O)N1c2ccccc2C=Cc2ccccc21", "dose_mg": 200, "cl_h": 5.0},
    "diazepam": {"smiles": "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21", "dose_mg": 10, "cl_h": 1.6},
    "digoxin": {
        "smiles": "C[C@@H]1O[C@@H](O[C@@H]2C[C@H](O)[C@@H](O[C@@H]3C[C@H](O)[C@@H](O[C@@H]4C[C@H](O)[C@@H](OC5CC(CO)=CC(=O)O5)C(C)O4)C(C)O3)C(C)O2)C[C@H](O)[C@H]1O",
        "dose_mg": 0.5,
        "cl_h": 7.0,
    },
    "fluoxetine": {"smiles": "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1", "dose_mg": 20, "cl_h": 40.0},
    "nifedipine": {
        "smiles": "COC(=O)C1=C(C)NC(C)=C(C(=O)OC)C1c1ccccc1[N+](=O)[O-]",
        "dose_mg": 10,
        "cl_h": 30.0,
    },
    "omeprazole": {
        "smiles": "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1",
        "dose_mg": 20,
        "cl_h": 30.0,
    },
    "phenytoin": {"smiles": "O=C1NC(=O)C(c2ccccc2)(c2ccccc2)N1", "dose_mg": 300, "cl_h": 1.5},
    "theophylline": {"smiles": "Cn1c(=O)c2[nH]cnc2n(C)c1=O", "dose_mg": 300, "cl_h": 3.0},
    "verapamil": {
        "smiles": "COc1ccc(CCN(C)CCCC(C#N)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC",
        "dose_mg": 80,
        "cl_h": 63.0,
    },
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


def main():
    from omega_pbpk.core.body import WholeBodyPBPK
    from omega_pbpk.drugs.drug import Drug
    from omega_pbpk.ml.models.adme.ensemble import EnsembleADMEPredictor

    predictor = EnsembleADMEPredictor()

    print("=" * 90)
    print("L1 BENCHMARK WITH KNOWN IN-VIVO CL (performance ceiling test)")
    print("=" * 90)

    # For CLh as a direct value, we need to back-calculate the CLint_in_vivo
    # that the well-stirred model would need:
    # CLh = Q × fup × CLint / (Q + fup × CLint)
    # CLint = Q × CLh / (fup × (Q - CLh))
    Q_H = 90.0  # hepatic blood flow

    cmax_fes = []
    auc_fes = []

    for drug_name, info in BENCHMARK_DRUGS.items():
        smiles = info["smiles"]
        dose_mg = info["dose_mg"]
        cl_h = info["cl_h"]

        observed = load_observed_pk(drug_name)
        if not observed:
            print(f"\n{drug_name}: No observed data")
            continue

        try:
            adme = predictor.predict(smiles)
            fup = max(adme.fup, 0.001)

            # Back-calculate CLint from CLh using well-stirred model
            if cl_h >= Q_H:
                cl_h = Q_H * 0.99  # cap at hepatic blood flow
            clint_in_vivo = Q_H * cl_h / (fup * (Q_H - cl_h))

            drug = Drug(
                name=drug_name,
                mw=adme.mw,
                logP=adme.logP,
                fup=fup,
                rbp=adme.rbp,
                clint_hepatic_L_per_h=clint_in_vivo,
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

            within = "YES" if fe_cmax <= 2.0 and fe_auc <= 2.0 else "NO"
            print(f"\n{drug_name} (dose={dose_mg}mg, CLh={cl_h}L/h):")
            print(f"  CLint_in_vivo={clint_in_vivo:.1f} L/h, fup={fup:.4f}")
            print(f"  Predicted: Cmax={pred_cmax:.4f}, AUC={pred_auc:.4f}")
            print(f"  Observed:  Cmax={observed['cmax']:.4f}, AUC={observed['auc']:.4f}")
            print(f"  FE: Cmax={fe_cmax:.2f}x, AUC={fe_auc:.2f}x, 2-fold: {within}")

        except Exception as e:
            print(f"\n{drug_name}: FAIL - {e}")

    # Summary
    aafe_cmax = compute_aafe(cmax_fes)
    aafe_auc = compute_aafe(auc_fes)
    pct_cmax = (
        sum(1 for fe in cmax_fes if not np.isnan(fe) and fe <= 2.0) / max(len(cmax_fes), 1) * 100
    )
    pct_auc = (
        sum(1 for fe in auc_fes if not np.isnan(fe) and fe <= 2.0) / max(len(auc_fes), 1) * 100
    )

    print(f"\n{'=' * 90}")
    print("RESULTS WITH KNOWN IN-VIVO CL")
    print(f"{'=' * 90}")
    print(f"  Cmax AAFE: {aafe_cmax:.2f} (target <3.0) {'PASS' if aafe_cmax < 3.0 else 'FAIL'}")
    print(f"  AUC AAFE:  {aafe_auc:.2f} (target <3.0) {'PASS' if aafe_auc < 3.0 else 'FAIL'}")
    print(f"  Cmax ≤2-fold: {pct_cmax:.0f}% (target ≥70%) {'PASS' if pct_cmax >= 70 else 'FAIL'}")
    print(f"  AUC ≤2-fold:  {pct_auc:.0f}% (target ≥70%) {'PASS' if pct_auc >= 70 else 'FAIL'}")
    print(f"  N drugs: {len(cmax_fes)}")


if __name__ == "__main__":
    main()
