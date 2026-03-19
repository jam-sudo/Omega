#!/usr/bin/env python3
"""Migrate gold-24 reference data to platinum format.

Reads gold24_reference_cmax.json + BENCHMARK_DRUGS, produces
platinum_reference.json with all required fields.
"""

import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from omega_pbpk.data.drug_registry import BENCHMARK_DRUGS, CORE24_NAMES  # noqa: E402
from omega_pbpk.data.platinum_schema import save_platinum_reference, validate_entry  # noqa: E402

GOLD_REF = repo_root / "data" / "clinical" / "gold24_reference_cmax.json"
OUTPUT = repo_root / "data" / "clinical" / "platinum_reference.json"

# Drugs with CLint/VDss anchors or hand-tuned pipeline parameters
TUNING_CONTAMINATED = {
    "caffeine",
    "metoprolol",
    "midazolam",
    "propranolol",
    "warfarin",
    "ibuprofen",
    "acetaminophen",
    "atorvastatin",
    "diazepam",
    "fluoxetine",
    "nifedipine",
    "omeprazole",
    "verapamil",
    "fluconazole",
}

NONLINEAR_PK = {"omeprazole", "phenytoin"}


def main():
    gold_ref = json.loads(GOLD_REF.read_text())
    gold_ref.pop("_metadata", None)

    drugs = {}
    for name in sorted(CORE24_NAMES):
        if name not in BENCHMARK_DRUGS:
            continue
        bm = BENCHMARK_DRUGS[name]
        ref = gold_ref.get(name, {})

        cmax = ref.get("cmax_mg_L")
        if cmax is None or cmax <= 0:
            print(f"SKIP {name}: no valid Cmax in gold reference")
            continue

        entry = {
            "smiles": bm["smiles"],
            "dose_mg": bm["dose_mg"],
            "cmax_mg_L": cmax,
            "auc_mg_h_L": ref.get("auc_mg_h_L"),
            "tmax_h": ref.get("tmax_h"),
            "source_type": "fda_label"
            if "FDA" in ref.get("source", "") or "NDA" in ref.get("source", "")
            else "literature",
            "source_id": ref.get("source", "unknown"),
            "fasted_confidence": "confirmed_fasted",
            "formulation": "IR",
            "route": "oral",
            "population": "healthy",
            "single_dose": True,
            "tuning_contaminated": name in TUNING_CONTAMINATED,
            "nonlinear_pk": name in NONLINEAR_PK,
            "data_quality": ref.get("data_quality", "clinical_exact"),
            "notes": ref.get("note", ""),
        }

        try:
            validate_entry(name, entry)
            drugs[name] = entry
            tag = " [contaminated]" if entry["tuning_contaminated"] else ""
            print(f"  OK  {name}: Cmax={cmax:.4f} mg/L{tag}")
        except Exception as e:
            print(f"FAIL {name}: {e}")

    save_platinum_reference(drugs, OUTPUT)
    print(f"\nWrote {len(drugs)} drugs to {OUTPUT}")
    n_clean = sum(1 for d in drugs.values() if not d["tuning_contaminated"])
    print(f"  Clean (non-contaminated): {n_clean}")
    print(f"  Contaminated: {len(drugs) - n_clean}")


if __name__ == "__main__":
    main()
