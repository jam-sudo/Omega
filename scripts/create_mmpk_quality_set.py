#!/usr/bin/env python3
"""Create quality-scored MMPK training set.

Integrates findings from B1-B4 audits into per-drug quality scores.
Output: data/ml/clinical/mmpk_quality_scored.csv

Usage:
    python scripts/create_mmpk_quality_set.py
"""

import csv
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

MMPK_PATH = REPO / "data" / "ml" / "clinical" / "mmpk_clean.csv"
FEATURES_PATH = REPO / "data" / "ml" / "clinical" / "mmpk_pbpk_features.csv"
CROSSREF_PATH = REPO / "outputs" / "mmpk_platinum_crossref.json"
OUTLIER_PATH = REPO / "outputs" / "mmpk_outlier_report.json"
RELIABILITY_PATH = REPO / "outputs" / "mmpk_reliability.json"
LINEARITY_PATH = REPO / "outputs" / "mmpk_dose_linearity.json"
OUTPUT_PATH = REPO / "data" / "ml" / "clinical" / "mmpk_quality_scored.csv"


def main():
    # Load MMPK clean data
    drugs = {}
    with open(MMPK_PATH) as f:
        for row in csv.DictReader(f):
            drugs[row["name"]] = {
                "name": row["name"],
                "smiles": row["smiles"],
                "dose_mg": float(row["dose_mg"]),
                "cmax_mg_L": float(row["cmax_mg_L"]),
                "log_cmax_per_dose": float(row["log_cmax_per_dose"]),
                "n_studies": int(row["n_studies"]),
                "in_platinum": str(row["in_platinum"]).strip() in ("True", "1.0", "1"),
            }

    # Load PBPK fold errors
    pbpk_fe = {}
    with open(FEATURES_PATH) as f:
        for row in csv.DictReader(f):
            obs = float(row["cmax_obs"])
            pred = float(row["cmax_pbpk"])
            if obs > 0 and pred > 0:
                pbpk_fe[row["name"]] = max(obs / pred, pred / obs)

    # Load cross-reference results (B1)
    crossref_flags = {}
    if CROSSREF_PATH.exists():
        with open(CROSSREF_PATH) as f:
            xref = json.load(f)
        for comp in xref.get("comparisons", []):
            crossref_flags[comp["mmpk_name"]] = comp["flag"]

    # Load outlier list (B2)
    outlier_names = set()
    if OUTLIER_PATH.exists():
        with open(OUTLIER_PATH) as f:
            outliers = json.load(f)
        outlier_names = {d["name"] for d in outliers.get("outliers", [])}

    # Load linearity results (B4)
    nonlinear_names = set()
    if LINEARITY_PATH.exists():
        with open(LINEARITY_PATH) as f:
            lin = json.load(f)
        for comp in lin.get("different_dose_comparisons", []):
            if comp.get("linearity_ratio", 0) > 2.0:
                nonlinear_names.add(comp["name"])

    # Compute quality scores
    rows = []
    for name, d in drugs.items():
        fe = pbpk_fe.get(name, float("nan"))

        # Component 1: reproducibility (n_studies)
        # n=1 drugs get 0.5 cap per team-lead guidance (B3 Mann-Whitney p<0.0001)
        w_studies = min(d["n_studies"] / 3.0, 1.0)
        if d["n_studies"] == 1:
            w_studies *= 0.5

        # Component 2: platinum cross-reference
        xref_flag = crossref_flags.get(name)
        if xref_flag == "OK":
            w_platinum = 1.0
        elif xref_flag == "WARN":
            w_platinum = 0.7
        elif xref_flag == "DISAGREE":
            w_platinum = 0.3
        elif d["in_platinum"]:
            w_platinum = 0.8  # in platinum but no specific cross-ref entry
        else:
            w_platinum = 0.5  # not in platinum — unknown external validity

        # Component 3: PBPK agreement (outlier check)
        if name in outlier_names or (not math.isnan(fe) and fe > 10.0):
            w_outlier = 0.0
        elif not math.isnan(fe) and fe > 5.0:
            w_outlier = 0.5
        else:
            w_outlier = 1.0

        # Component 4: dose linearity
        w_linearity = 0.5 if name in nonlinear_names else 1.0

        # Final score
        quality = (w_studies + w_platinum + w_outlier + w_linearity) / 4.0
        include = quality >= 0.25

        rows.append(
            {
                "name": name,
                "smiles": d["smiles"],
                "dose_mg": d["dose_mg"],
                "cmax_mg_L": d["cmax_mg_L"],
                "log_cmax_per_dose": d["log_cmax_per_dose"],
                "n_studies": d["n_studies"],
                "in_platinum": d["in_platinum"],
                "pbpk_fold_error": round(fe, 3) if not math.isnan(fe) else "",
                "w_studies": round(w_studies, 3),
                "w_platinum": round(w_platinum, 3),
                "w_outlier": round(w_outlier, 3),
                "w_linearity": round(w_linearity, 3),
                "quality_score": round(quality, 4),
                "include": include,
            }
        )

    # Write
    fieldnames = list(rows[0].keys())
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: -r["quality_score"]))

    n_include = sum(1 for r in rows if r["include"])
    n_exclude = len(rows) - n_include
    scores = [r["quality_score"] for r in rows]

    print(f"Quality-scored MMPK dataset: {len(rows)} drugs")
    print(f"  Include: {n_include} ({n_include / len(rows) * 100:.1f}%)")
    print(f"  Exclude: {n_exclude} ({n_exclude / len(rows) * 100:.1f}%)")
    print(f"  Median quality: {sorted(scores)[len(scores) // 2]:.3f}")
    print(f"  Min quality: {min(scores):.3f}")
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
