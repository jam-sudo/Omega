#!/usr/bin/env python3
"""Classify benchmark failures (>3-fold Cmax error) by pharmacological mechanism."""

from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

# Curated mechanism lookup from pharmacology literature
KNOWN_MECHANISMS: dict[str, dict[str, str]] = {
    "verapamil": {
        "category": "transporter",
        "detail": "P-gp substrate, high first-pass via CYP3A4+P-gp efflux",
    },
    "ibuprofen": {
        "category": "protein_binding",
        "detail": (
            "99% protein bound (fup~0.01), small fup errors cause large PK errors"
        ),
    },
    "phenytoin": {
        "category": "metabolism",
        "detail": "Saturable (Michaelis-Menten) CYP2C9 metabolism, nonlinear PK",
    },
    "digoxin": {
        "category": "transporter",
        "detail": "P-gp substrate, renal elimination, narrow therapeutic index",
    },
    "atorvastatin": {
        "category": "transporter",
        "detail": "OATP1B1 substrate, hepatic uptake transporter",
    },
    "nifedipine": {
        "category": "metabolism",
        "detail": "High CYP3A4 first-pass, bioavailability ~50%",
    },
    "midazolam": {
        "category": "metabolism",
        "detail": "CYP3A4 substrate, high first-pass extraction",
    },
    "fluoxetine": {
        "category": "solubility",
        "detail": "Lipophilic, solubility-limited absorption (fixed by GSE floor)",
    },
    "carbamazepine": {
        "category": "metabolism",
        "detail": "CYP3A4 autoinduction, complex PK at steady state",
    },
}

FOLD_ERROR_THRESHOLD = 3.0


def find_latest_benchmark() -> Path:
    """Find the most recent benchmark JSON file in outputs/."""
    outputs_dir = Path(__file__).resolve().parent.parent / "outputs"
    files = sorted(glob.glob(str(outputs_dir / "benchmark_*.json")))
    if not files:
        print("ERROR: No benchmark_*.json files found in outputs/", file=sys.stderr)
        sys.exit(1)
    return Path(files[-1])


def classify_failures(benchmark_path: Path) -> dict:
    """Load benchmark results and classify >3-fold Cmax failures."""
    with open(benchmark_path) as f:
        data = json.load(f)

    failures = []
    for entry in data["per_drug"]:
        fe_cmax = entry.get("fe_cmax", 0)
        if fe_cmax > FOLD_ERROR_THRESHOLD:
            drug = entry["drug"]
            mechanism = KNOWN_MECHANISMS.get(
                drug, {"category": "unknown", "detail": "Needs investigation"}
            )
            failures.append(
                {
                    "drug": drug,
                    "fe_cmax": fe_cmax,
                    "category": mechanism["category"],
                    "detail": mechanism["detail"],
                }
            )

    # Group by category
    by_category: dict[str, list[str]] = defaultdict(list)
    for f in failures:
        by_category[f["category"]].append(f["drug"])

    return {
        "source": str(benchmark_path),
        "n_failures": len(failures),
        "by_category": dict(by_category),
        "details": failures,
    }


def main() -> None:
    benchmark_path = find_latest_benchmark()
    print(f"Analyzing: {benchmark_path.name}")

    result = classify_failures(benchmark_path)

    # Print summary
    print(f"\nDrugs with >{FOLD_ERROR_THRESHOLD}-fold Cmax error: {result['n_failures']}")
    if result["n_failures"] == 0:
        print("No failures to classify.")
    else:
        print("\nBy category:")
        for category, drugs in sorted(result["by_category"].items()):
            print(f"  {category}: {', '.join(drugs)}")
        print("\nDetails:")
        for d in result["details"]:
            print(f"  {d['drug']:20s} FE={d['fe_cmax']:.2f}  [{d['category']}] {d['detail']}")

    # Save output
    out_path = benchmark_path.parent / "failure_analysis.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
