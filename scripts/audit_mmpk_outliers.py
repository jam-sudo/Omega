#!/usr/bin/env python3
"""Detect outliers in MMPK dataset using PBPK fold errors.

Uses existing cmax_pbpk predictions from mmpk_pbpk_features.csv.
Flags drugs with PBPK fold error > 10x as potential data errors
or applicability domain violations.

Usage:
    python scripts/audit_mmpk_outliers.py
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

FEATURES_PATH = REPO / "data" / "ml" / "clinical" / "mmpk_pbpk_features.csv"


def main():
    drugs = []
    with open(FEATURES_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cmax_obs = float(row["cmax_obs"])
            cmax_pbpk = float(row["cmax_pbpk"])
            if cmax_obs > 0 and cmax_pbpk > 0:
                fe = max(cmax_obs / cmax_pbpk, cmax_pbpk / cmax_obs)
            else:
                fe = float("nan")

            drugs.append(
                {
                    "name": row["name"],
                    "smiles": row["smiles"],
                    "dose_mg": float(row["dose_mg"]),
                    "cmax_obs": round(cmax_obs, 6),
                    "cmax_pbpk": round(cmax_pbpk, 6),
                    "fold_error": round(fe, 3),
                    "fup": float(row["fup"]),
                    "clint": float(row["clint"]),
                    "logP": float(row["logP"]),
                    "is_acid": float(row["is_acid"]) > 0.5,
                    "is_base": float(row["is_base"]) > 0.5,
                    "in_platinum": str(row["in_platinum"]).strip() in ("True", "1.0", "1"),
                }
            )

    # Classify
    ok = [d for d in drugs if not np.isnan(d["fold_error"]) and d["fold_error"] <= 3.0]
    warning = [d for d in drugs if not np.isnan(d["fold_error"]) and 3.0 < d["fold_error"] <= 10.0]
    outlier = [d for d in drugs if not np.isnan(d["fold_error"]) and d["fold_error"] > 10.0]
    invalid = [d for d in drugs if np.isnan(d["fold_error"])]

    print(f"Total: {len(drugs)} drugs")
    print(f"  OK (<=3x):      {len(ok)} ({len(ok) / len(drugs) * 100:.1f}%)")
    print(f"  WARNING (3-10x): {len(warning)} ({len(warning) / len(drugs) * 100:.1f}%)")
    print(f"  OUTLIER (>10x):  {len(outlier)} ({len(outlier) / len(drugs) * 100:.1f}%)")
    if invalid:
        print(f"  INVALID (NaN):   {len(invalid)}")

    # Analyze outliers
    print(f"\n{'=' * 70}")
    print(f"OUTLIER DRUGS (FE > 10x): {len(outlier)}")
    print(f"{'=' * 70}")

    # Check for common patterns
    outlier_patterns = {
        "very_low_fup": sum(1 for d in outlier if d["fup"] < 0.01),
        "very_high_logP": sum(1 for d in outlier if d["logP"] > 5.0),
        "very_low_clint": sum(1 for d in outlier if d["clint"] < 0.1),
        "very_high_clint": sum(1 for d in outlier if d["clint"] > 3.0),
        "acids": sum(1 for d in outlier if d["is_acid"]),
        "bases": sum(1 for d in outlier if d["is_base"]),
    }

    for d in sorted(outlier, key=lambda x: -x["fold_error"])[:30]:
        direction = "OVER" if d["cmax_pbpk"] > d["cmax_obs"] else "UNDER"
        print(
            f"  {d['name']:30s} FE={d['fold_error']:7.1f}x {direction:5s}  "
            f"fup={d['fup']:.4f} clint={d['clint']:.2f} logP={d['logP']:.1f}"
        )

    valid_fes = [d["fold_error"] for d in drugs if not np.isnan(d["fold_error"])]

    output = {
        "n_total": len(drugs),
        "n_ok": len(ok),
        "n_warning": len(warning),
        "n_outlier": len(outlier),
        "outlier_patterns": outlier_patterns,
        "outliers": sorted(
            [
                {
                    "name": d["name"],
                    "fold_error": d["fold_error"],
                    "smiles": d["smiles"],
                    "fup": d["fup"],
                    "clint": d["clint"],
                    "logP": d["logP"],
                }
                for d in outlier
            ],
            key=lambda x: -x["fold_error"],
        ),
        "fold_error_distribution": {
            "p25": round(float(np.percentile(valid_fes, 25)), 3),
            "p50": round(float(np.percentile(valid_fes, 50)), 3),
            "p75": round(float(np.percentile(valid_fes, 75)), 3),
            "p90": round(float(np.percentile(valid_fes, 90)), 3),
        },
    }

    out_path = REPO / "outputs" / "mmpk_outlier_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(f"\nOutlier patterns: {outlier_patterns}")


if __name__ == "__main__":
    main()
