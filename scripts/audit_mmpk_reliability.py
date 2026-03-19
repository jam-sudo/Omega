#!/usr/bin/env python3
"""Assess MMPK reliability by study count.

Compares PBPK prediction accuracy between:
- n=1 study drugs (54% of MMPK)
- n>=2 study drugs (46% of MMPK)

If n=1 drugs have systematically worse PBPK agreement,
they should be downweighted in UDE training.

Usage:
    python scripts/audit_mmpk_reliability.py
"""

import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
FEATURES_PATH = REPO / "data" / "ml" / "clinical" / "mmpk_pbpk_features.csv"
CLEAN_PATH = REPO / "data" / "ml" / "clinical" / "mmpk_clean.csv"


def main():
    # Load n_studies from clean, features from pbpk_features
    n_studies_map = {}
    with open(CLEAN_PATH) as f:
        for row in csv.DictReader(f):
            n_studies_map[row["name"]] = int(row["n_studies"])

    single_study_fe = []
    multi_study_fe = []

    with open(FEATURES_PATH) as f:
        for row in csv.DictReader(f):
            name = row["name"]
            cmax_obs = float(row["cmax_obs"])
            cmax_pbpk = float(row["cmax_pbpk"])
            if cmax_obs <= 0 or cmax_pbpk <= 0:
                continue
            fe = max(cmax_obs / cmax_pbpk, cmax_pbpk / cmax_obs)
            n = n_studies_map.get(name, 1)
            if n == 1:
                single_study_fe.append(fe)
            else:
                multi_study_fe.append(fe)

    def stats(fes, label):
        log_fe = np.log10(fes)
        aafe = float(10 ** np.mean(np.abs(log_fe)))
        p2f = sum(1 for fe in fes if fe <= 2.0) / len(fes) * 100
        median = float(np.median(fes))
        return {
            "label": label,
            "n": len(fes),
            "aafe": round(aafe, 3),
            "median_fe": round(median, 3),
            "pct_2fold": round(p2f, 1),
            "pct_gt10x": round(sum(1 for fe in fes if fe > 10) / len(fes) * 100, 1),
        }

    s1 = stats(single_study_fe, "n=1 study")
    sm = stats(multi_study_fe, "n>=2 studies")

    print(f"{'Metric':20s} {'n=1':>12s} {'n>=2':>12s}")
    print("-" * 46)
    print(f"{'N drugs':20s} {s1['n']:12d} {sm['n']:12d}")
    print(f"{'AAFE (PBPK)':20s} {s1['aafe']:12.3f} {sm['aafe']:12.3f}")
    print(f"{'Median FE':20s} {s1['median_fe']:12.3f} {sm['median_fe']:12.3f}")
    print(f"{'%2-fold':20s} {s1['pct_2fold']:11.1f}% {sm['pct_2fold']:11.1f}%")
    print(f"{'%>10x':20s} {s1['pct_gt10x']:11.1f}% {sm['pct_gt10x']:11.1f}%")

    # Mann-Whitney U test
    from scipy.stats import mannwhitneyu

    stat, pval = mannwhitneyu(single_study_fe, multi_study_fe, alternative="greater")

    print(f"\nMann-Whitney U test (n=1 > n>=2): p = {pval:.4f}")
    if pval < 0.05:
        print("-> Single-study drugs have SIGNIFICANTLY higher PBPK errors")
        print("-> RECOMMENDATION: downweight n=1 drugs in UDE training (weight=0.5)")
    else:
        print("-> No significant difference -- n=1 drugs are OK to use at full weight")

    output = {
        "single_study": s1,
        "multi_study": sm,
        "mann_whitney_p": round(pval, 6),
        "recommendation": "downweight_n1" if pval < 0.05 else "equal_weight",
    }

    out_path = REPO / "outputs" / "mmpk_reliability.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
