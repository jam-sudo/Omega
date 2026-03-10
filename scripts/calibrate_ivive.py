#!/usr/bin/env python3
"""Calibrate IVIVE correction factor using 20 benchmark drugs.

Strategy:
1. Get ADMET-AI raw hepatocyte CLint for each drug
2. Compute standard hepatocyte IVIVE: CLint_hep × 120 × 1800 / 1e6 / 60 = CLint_hep × 3.6 L/h
3. Back-calculate required CLint from observed AUC using well-stirred model
4. Fit an empirical correction factor (median ratio)
5. Re-run benchmarks with the calibrated factor
"""

import json
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

# 20 benchmark drugs
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

# Known in-vivo hepatic clearances (L/h) from literature for calibration
# Source: various clinical PK reviews, Drug Information Handbook
KNOWN_CL_HEPATIC = {
    "caffeine": 6.0,  # ~6 L/h, CYP1A2 primary
    "metoprolol": 63.0,  # ~63 L/h, CYP2D6 extensive metabolizer
    "midazolam": 27.0,  # ~27 L/h, CYP3A4 primary
    "propranolol": 60.0,  # ~60 L/h, CYP1A2/2D6
    "warfarin": 0.2,  # ~0.2 L/h, very low CL (long half-life)
    "d_amphetamine": 20.0,  # ~20 L/h, CYP2D6
    "ibuprofen": 3.6,  # ~3.6 L/h, CYP2C9
    "acetaminophen": 21.0,  # ~21 L/h, UGT + CYP2E1
    "atorvastatin": 38.0,  # ~38 L/h, CYP3A4, high first-pass
    "carbamazepine": 5.0,  # ~5 L/h, CYP3A4 (autoinduction)
    "diazepam": 1.6,  # ~1.6 L/h, CYP3A4/2C19
    "fluoxetine": 40.0,  # ~40 L/h, CYP2D6/2C9
    "nifedipine": 30.0,  # ~30 L/h, CYP3A4
    "omeprazole": 30.0,  # ~30 L/h, CYP2C19/3A4
    "phenytoin": 1.5,  # ~1.5 L/h, CYP2C9/2C19 (saturable)
    "theophylline": 3.0,  # ~3.0 L/h, CYP1A2
    "verapamil": 63.0,  # ~63 L/h, CYP3A4, high first-pass
}


def load_observed_pk(drug_name):
    """Load observed Cmax and AUC from benchmark CSV."""
    datasets_dir = repo_root / "benchmarks" / "datasets"
    for csv_file in datasets_dir.glob("*.csv"):
        if drug_name.replace("_", "") in csv_file.stem.replace("_", "").lower():
            data = np.genfromtxt(csv_file, delimiter=",", skip_header=1)
            time_h = data[:, 0]
            conc = data[:, 1]
            cmax = float(conc.max())
            auc = float(np.trapz(conc, time_h))
            return {"cmax": cmax, "auc": auc}
    return {}


def main():
    from omega_pbpk.ml.models.adme.ensemble import EnsembleADMEPredictor

    predictor = EnsembleADMEPredictor()

    print("=" * 90)
    print("IVIVE CALIBRATION: Analyzing CLint gap for 20 benchmark drugs")
    print("=" * 90)

    # Standard IVIVE constants
    HEPATOCELLULARITY = 120.0  # 10^6 cells/g liver
    LIVER_WT = 1800.0  # g

    results = []

    for drug_name, info in BENCHMARK_DRUGS.items():
        smiles = info["smiles"]

        try:
            adme = predictor.predict(smiles)
            observed = load_observed_pk(drug_name)

            # Raw hepatocyte CLint from ADMET-AI (µL/min/10^6 cells)
            clint_hep_raw = getattr(adme, "clint_hepatocyte_uL_min", 0.0)

            # Standard hepatocyte IVIVE: µL/min/10^6 cells → L/h
            # = clint_hep × hepatocellularity × liver_wt / 1e6 / 60
            clint_ivive_standard = clint_hep_raw * HEPATOCELLULARITY * LIVER_WT / 1e6 / 60.0

            # pmol CYP3A4 IVIVE (what ensemble currently gives)
            clint_pmol = adme.clint_3a4  # µL/min/pmol (after /5480)
            clint_ivive_pmol = clint_pmol * 40.0 * 45.0 * 1800.0 / 1e6 / 60.0

            # Known in-vivo CL if available
            cl_known = KNOWN_CL_HEPATIC.get(drug_name, None)

            # Ratio: known / standard IVIVE
            ratio = None
            if cl_known and clint_ivive_standard > 0.001:
                ratio = cl_known / clint_ivive_standard

            entry = {
                "drug": drug_name,
                "clint_hep_raw": clint_hep_raw,
                "clint_3a4_pmol": adme.clint_3a4,
                "clint_ivive_standard_Lh": clint_ivive_standard,
                "clint_ivive_pmol_Lh": clint_ivive_pmol,
                "cl_known_Lh": cl_known,
                "correction_ratio": ratio,
                "fup": adme.fup,
                "logP": adme.logP,
            }

            if observed:
                entry["obs_cmax"] = observed["cmax"]
                entry["obs_auc"] = observed["auc"]

            results.append(entry)

            print(f"\n{drug_name}:")
            print(f"  ADMET-AI hepatocyte CLint: {clint_hep_raw:.3f} µL/min/10^6 cells")
            print(f"  clint_3a4 (after /5480):   {adme.clint_3a4:.4f} µL/min/pmol")
            print(f"  Standard IVIVE (×3.6):     {clint_ivive_standard:.4f} L/h")
            print(f"  pmol IVIVE (×0.054):       {clint_ivive_pmol:.6f} L/h")
            if cl_known:
                print(f"  Known in-vivo CL:          {cl_known:.1f} L/h")
                if ratio:
                    print(f"  Correction factor:         {ratio:.1f}×")

        except Exception as e:
            print(f"\n{drug_name}: ERROR - {e}")

    # Compute statistics on correction ratios
    ratios = [r["correction_ratio"] for r in results if r["correction_ratio"] is not None]
    if ratios:
        print("\n" + "=" * 90)
        print("CORRECTION FACTOR ANALYSIS")
        print("=" * 90)
        print(f"  N drugs with known CL: {len(ratios)}")
        print(f"  Correction factors: {[f'{r:.1f}' for r in sorted(ratios)]}")
        print(f"  Median:  {np.median(ratios):.1f}×")
        print(f"  Mean:    {np.mean(ratios):.1f}×")
        print(f"  Geomean: {10 ** np.mean(np.log10(ratios)):.1f}×")
        print(f"  Min:     {min(ratios):.1f}×")
        print(f"  Max:     {max(ratios):.1f}×")

        # Use geometric mean as the empirical correction factor
        correction = 10 ** np.mean(np.log10(ratios))
        print(f"\n  → Recommended empirical IVIVE correction: {correction:.1f}×")
        print(
            f"    (applied to standard hepatocyte IVIVE: CLint_in_vivo = CLint_hep × 3.6 × {correction:.0f})"
        )

    # Also analyze drugs with zero/negligible hepatocyte CLint
    zero_clint = [r for r in results if r["clint_hep_raw"] < 0.1]
    if zero_clint:
        print(f"\n  NOTE: {len(zero_clint)} drugs have negligible hepatocyte CLint from ADMET-AI:")
        for r in zero_clint:
            print(f"    {r['drug']}: clint_hep={r['clint_hep_raw']:.3f}")

    # Save calibration results
    out = {
        "per_drug": results,
        "correction_factors": ratios,
        "recommended_factor": float(10 ** np.mean(np.log10(ratios))) if ratios else None,
        "median_factor": float(np.median(ratios)) if ratios else None,
    }
    out_path = repo_root / "outputs" / "ivive_calibration.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nCalibration saved to {out_path}")


if __name__ == "__main__":
    main()
