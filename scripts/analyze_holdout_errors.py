#!/usr/bin/env python3
"""Decompose holdout prediction errors by PK mechanism.

For each drug with FE > 2x:
- Identifies whether over/under prediction
- Gets predicted ADME values
- Classifies failure mode (CLint, fup, Vd, F, absorption)

Usage:
    python scripts/analyze_holdout_errors.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from omega_pbpk.pipeline import OmegaPipeline  # noqa: E402


def main():
    baseline_path = REPO / "outputs" / "holdout_baseline.json"
    plat_path = REPO / "data" / "clinical" / "platinum_reference.json"

    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(plat_path) as f:
        plat = json.load(f)

    pipeline = OmegaPipeline()

    # Analyze drugs with FE > 2x
    high_error_drugs = [
        r for r in baseline["per_drug"] if r.get("success") and r.get("fold_error", 0) > 2.0
    ]

    print(f"Analyzing {len(high_error_drugs)} drugs with FE > 2.0x")
    print("=" * 80)
    analyses = []

    for drug_result in sorted(high_error_drugs, key=lambda r: -r["fold_error"]):
        drug_name = drug_result["drug"]
        entry = plat["drugs"][drug_name]
        smiles = entry["smiles"]
        pred_cmax = drug_result["pred_cmax"]
        obs_cmax = drug_result["obs_cmax"]

        direction = "OVER" if pred_cmax > obs_cmax else "UNDER"

        # Get baseline ADME predictions
        try:
            warnings_list = []
            adme = pipeline._predict_adme(smiles, warnings_list)
        except Exception:
            analyses.append(
                {
                    "drug": drug_name,
                    "fold_error": drug_result["fold_error"],
                    "direction": direction,
                    "failure_mode": "ADME_PREDICTION_FAILED",
                }
            )
            continue

        # Extract ADME values (dict access)
        fup = adme.get("fup")
        clint = adme.get("clint_3a4")
        clint_hep = adme.get("clint_hepatocyte_uL_min")
        logp = adme.get("logP")
        peff = adme.get("peff")
        mw = adme.get("mw")

        # Classify failure mode based on direction and drug properties
        if direction == "UNDER" and drug_result["fold_error"] > 50:
            # Extreme underprediction: likely prodrug or missing absorption mechanism
            failure_mode = "PRODRUG_OR_MISSING_MECHANISM"
        elif direction == "OVER" and drug_result["fold_error"] > 50:
            failure_mode = "EXTREME_OVERPREDICTION"
        elif direction == "UNDER" and clint and clint > 100:
            failure_mode = "CLint_OVERPREDICTED"
        elif direction == "UNDER" and fup and fup < 0.01:
            failure_mode = "fup_TOO_LOW"
        elif direction == "UNDER" and peff and peff < 1.0:
            failure_mode = "ABSORPTION_LIMITED"
        elif direction == "UNDER":
            failure_mode = "F_UNDERPREDICTED"
        elif direction == "OVER" and clint and clint < 5.0:
            failure_mode = "CLint_UNDERPREDICTED"
        elif direction == "OVER" and fup and fup > 0.5:
            failure_mode = "Vd_TOO_LOW"
        elif direction == "OVER":
            failure_mode = "F_OVERPREDICTED"
        else:
            failure_mode = "MIXED"

        analyses.append(
            {
                "drug": drug_name,
                "fold_error": drug_result["fold_error"],
                "direction": direction,
                "failure_mode": failure_mode,
                "pred_cmax": round(pred_cmax, 6),
                "obs_cmax": round(obs_cmax, 6),
                "adme": {
                    "fup": round(fup, 6) if fup else None,
                    "clint_3a4": round(clint, 4) if clint else None,
                    "clint_hepatocyte": round(clint_hep, 4) if clint_hep else None,
                    "logP": round(logp, 4) if logp else None,
                    "peff": round(peff, 4) if peff else None,
                    "mw": round(mw, 2) if mw else None,
                },
            }
        )

        print(
            f"  {drug_name:25s} FE={drug_result['fold_error']:8.2f}x {direction:5s} -> {failure_mode}"
        )

    # Aggregate failure modes
    mode_counts = Counter(a["failure_mode"] for a in analyses)

    # Direction summary
    over_count = sum(1 for a in analyses if a["direction"] == "OVER")
    under_count = sum(1 for a in analyses if a["direction"] == "UNDER")

    # Severity bins
    severe = [a for a in analyses if a["fold_error"] > 10]
    moderate = [a for a in analyses if 3 < a["fold_error"] <= 10]
    mild = [a for a in analyses if 2 < a["fold_error"] <= 3]

    output = {
        "n_analyzed": len(analyses),
        "n_over": over_count,
        "n_under": under_count,
        "severity": {
            "severe_gt10x": len(severe),
            "moderate_3_10x": len(moderate),
            "mild_2_3x": len(mild),
        },
        "failure_mode_distribution": dict(mode_counts.most_common()),
        "severe_drugs": [
            {"drug": a["drug"], "fold_error": a["fold_error"], "mode": a["failure_mode"]}
            for a in severe
        ],
        "per_drug": analyses,
    }

    print(f"\n{'=' * 80}")
    print(f"ERROR DECOMPOSITION SUMMARY ({len(analyses)} drugs with FE > 2x)")
    print(f"  Direction: {over_count} OVER / {under_count} UNDER")
    print(
        f"  Severity:  {len(severe)} severe (>10x), {len(moderate)} moderate (3-10x), {len(mild)} mild (2-3x)"
    )
    print("\nFailure mode distribution:")
    for mode, count in mode_counts.most_common():
        print(f"  {mode:35s} {count:3d} ({count / len(analyses) * 100:4.0f}%)")

    out_path = REPO / "outputs" / "holdout_error_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
