#!/usr/bin/env python3
"""Validate dose normalization assumption in MMPK.

Checks whether Cmax/dose is consistent across dose levels for drugs
appearing in both MMPK and platinum at different doses.

Usage:
    python scripts/audit_mmpk_dose_linearity.py
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

MMPK_PATH = REPO / "data" / "ml" / "clinical" / "mmpk_clean.csv"
PLATINUM_PATH = REPO / "data" / "clinical" / "platinum_reference.json"


def canonical_smiles(smi):
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol else smi


def main():
    # Load both datasets
    mmpk = {}
    with open(MMPK_PATH) as f:
        for row in csv.DictReader(f):
            can_smi = canonical_smiles(row["smiles"])
            mmpk[can_smi] = {
                "name": row["name"],
                "dose": float(row["dose_mg"]),
                "cmax": float(row["cmax_mg_L"]),
            }

    with open(PLATINUM_PATH) as f:
        plat = json.load(f)
    platinum = {}
    for name, entry in plat["drugs"].items():
        can_smi = canonical_smiles(entry["smiles"])
        platinum[can_smi] = {
            "name": name,
            "dose": entry["dose_mg"],
            "cmax": entry["cmax_mg_L"],
        }

    # Find overlapping drugs with DIFFERENT doses
    different_dose = []
    overlap = set(mmpk.keys()) & set(platinum.keys())
    for smi in overlap:
        m, p = mmpk[smi], platinum[smi]
        dose_ratio = max(m["dose"] / p["dose"], p["dose"] / m["dose"])
        if dose_ratio > 1.2:  # at least 20% dose difference
            cpd_m = m["cmax"] / m["dose"]
            cpd_p = p["cmax"] / p["dose"]
            linearity_ratio = (
                max(cpd_m / cpd_p, cpd_p / cpd_m) if cpd_m > 0 and cpd_p > 0 else float("nan")
            )
            different_dose.append(
                {
                    "name": m["name"],
                    "mmpk_dose": m["dose"],
                    "platinum_dose": p["dose"],
                    "mmpk_cpd": round(cpd_m, 8),
                    "platinum_cpd": round(cpd_p, 8),
                    "linearity_ratio": round(linearity_ratio, 3),
                    "dose_ratio": round(dose_ratio, 2),
                }
            )

    # High-dose vs low-dose analysis across all MMPK drugs
    all_cpd = [
        (d["cmax"] / d["dose"], d["dose"]) for d in mmpk.values() if d["dose"] > 0 and d["cmax"] > 0
    ]
    low_dose = [cpd for cpd, dose in all_cpd if dose <= 100]
    high_dose = [cpd for cpd, dose in all_cpd if dose > 500]

    print(f"Overlap drugs with different doses: {len(different_dose)}")
    for d in sorted(different_dose, key=lambda x: -x["linearity_ratio"]):
        flag = "NONLINEAR" if d["linearity_ratio"] > 2.0 else "OK"
        print(
            f"  {flag:10s} {d['name']:25s} MMPK={d['mmpk_dose']}mg PLAT={d['platinum_dose']}mg  ratio={d['linearity_ratio']:.2f}x"
        )

    n_nonlinear = sum(
        1
        for d in different_dose
        if not np.isnan(d["linearity_ratio"]) and d["linearity_ratio"] > 2.0
    )
    print(f"\nNonlinear drugs (Cmax/dose ratio > 2x): {n_nonlinear}/{len(different_dose)}")

    if low_dose and high_dose:
        print("\nDose range analysis:")
        print(
            f"  Low dose (<=100mg):  median Cmax/dose = {np.median(low_dose):.6f} (n={len(low_dose)})"
        )
        print(
            f"  High dose (>500mg): median Cmax/dose = {np.median(high_dose):.6f} (n={len(high_dose)})"
        )

    output = {
        "n_different_dose": len(different_dose),
        "n_nonlinear": n_nonlinear,
        "pct_nonlinear": round(n_nonlinear / max(len(different_dose), 1) * 100, 1),
        "different_dose_comparisons": different_dose,
    }

    if low_dose and high_dose:
        output["dose_range_analysis"] = {
            "n_low_dose": len(low_dose),
            "n_high_dose": len(high_dose),
            "median_cpd_low": round(float(np.median(low_dose)), 6),
            "median_cpd_high": round(float(np.median(high_dose)), 6),
        }

    out_path = REPO / "outputs" / "mmpk_dose_linearity.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
