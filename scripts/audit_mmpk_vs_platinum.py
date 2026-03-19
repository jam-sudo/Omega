#!/usr/bin/env python3
"""Cross-reference MMPK Cmax vs platinum Cmax for overlapping drugs.

For each drug present in both datasets:
- Compare Cmax values (dose-normalized if doses differ)
- Flag disagreements > 2-fold as potential data quality issues
- Compute overall agreement statistics

Usage:
    python scripts/audit_mmpk_vs_platinum.py
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


def canonical_smiles(smi: str) -> str:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol else smi


def main():
    # Load MMPK
    mmpk_drugs = {}
    with open(MMPK_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            can_smi = canonical_smiles(row["smiles"])
            mmpk_drugs[can_smi] = {
                "name": row["name"],
                "smiles": row["smiles"],
                "dose_mg": float(row["dose_mg"]),
                "cmax_mg_L": float(row["cmax_mg_L"]),
                "n_studies": int(row["n_studies"]),
            }
    print(f"MMPK: {len(mmpk_drugs)} drugs (canonical SMILES)")

    # Load platinum
    with open(PLATINUM_PATH) as f:
        plat = json.load(f)
    plat_drugs = {}
    for name, entry in plat["drugs"].items():
        can_smi = canonical_smiles(entry["smiles"])
        plat_drugs[can_smi] = {
            "name": name,
            "dose_mg": entry["dose_mg"],
            "cmax_mg_L": entry["cmax_mg_L"],
            "source_type": entry.get("source_type"),
            "data_quality": entry.get("data_quality"),
        }
    print(f"Platinum: {len(plat_drugs)} drugs (canonical SMILES)")

    # Cross-reference
    overlap_smiles = set(mmpk_drugs.keys()) & set(plat_drugs.keys())
    print(f"Overlap: {len(overlap_smiles)} drugs")

    comparisons = []
    fold_ratios = []

    for can_smi in sorted(overlap_smiles):
        m = mmpk_drugs[can_smi]
        p = plat_drugs[can_smi]

        # Dose-normalized comparison
        m_cpd = m["cmax_mg_L"] / m["dose_mg"]  # Cmax per mg dose
        p_cpd = p["cmax_mg_L"] / p["dose_mg"]

        if m_cpd > 0 and p_cpd > 0:
            fold_ratio = max(m_cpd / p_cpd, p_cpd / m_cpd)
        else:
            fold_ratio = float("nan")

        fold_ratios.append(fold_ratio)
        flag = "DISAGREE" if fold_ratio > 2.0 else ("WARN" if fold_ratio > 1.5 else "OK")

        comp = {
            "mmpk_name": m["name"],
            "platinum_name": p["name"],
            "mmpk_dose": m["dose_mg"],
            "platinum_dose": p["dose_mg"],
            "mmpk_cmax": round(m["cmax_mg_L"], 6),
            "platinum_cmax": round(p["cmax_mg_L"], 6),
            "mmpk_cpd": round(m_cpd, 8),
            "platinum_cpd": round(p_cpd, 8),
            "fold_ratio": round(fold_ratio, 3),
            "n_studies": m["n_studies"],
            "data_quality": p["data_quality"],
            "flag": flag,
        }
        comparisons.append(comp)

        if flag != "OK":
            print(
                f"  {flag:8s} {m['name']:25s} MMPK={m['cmax_mg_L']:.4f}@{m['dose_mg']}mg  "
                f"PLAT={p['cmax_mg_L']:.4f}@{p['dose_mg']}mg  ratio={fold_ratio:.2f}x"
            )

    # Summary
    valid_ratios = [r for r in fold_ratios if not np.isnan(r)]
    n_ok = sum(1 for c in comparisons if c["flag"] == "OK")
    n_warn = sum(1 for c in comparisons if c["flag"] == "WARN")
    n_disagree = sum(1 for c in comparisons if c["flag"] == "DISAGREE")

    summary = {
        "n_overlap": len(overlap_smiles),
        "n_ok": n_ok,
        "n_warn": n_warn,
        "n_disagree": n_disagree,
        "pct_within_2fold": round((n_ok + n_warn) / max(len(valid_ratios), 1) * 100, 1),
        "median_fold_ratio": round(float(np.median(valid_ratios)), 3),
        "mean_fold_ratio": round(float(np.mean(valid_ratios)), 3),
        "comparisons": comparisons,
    }

    print("\nSummary:")
    print(f"  OK (< 1.5x):      {n_ok} ({n_ok / len(comparisons) * 100:.0f}%)")
    print(f"  WARN (1.5-2.0x):   {n_warn} ({n_warn / len(comparisons) * 100:.0f}%)")
    print(f"  DISAGREE (> 2.0x): {n_disagree} ({n_disagree / len(comparisons) * 100:.0f}%)")
    print(
        f"  Within 2-fold:     {n_ok + n_warn}/{len(comparisons)} ({summary['pct_within_2fold']}%)"
    )
    print(f"  Median fold ratio: {np.median(valid_ratios):.3f}x")

    # Gate check
    pct_disagree = n_disagree / max(len(comparisons), 1) * 100
    if pct_disagree > 30:
        print(f"\n⚠️  GATE FAILURE: {pct_disagree:.0f}% DISAGREE (>30% threshold)")
        print("    Investigate MMPK data sources before proceeding.")
    else:
        print(f"\n✓  GATE PASS: {pct_disagree:.0f}% DISAGREE (≤30% threshold)")

    out_path = REPO / "outputs" / "mmpk_platinum_crossref.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
