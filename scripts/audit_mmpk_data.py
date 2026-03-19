#!/usr/bin/env python3
"""Audit and clean the MMPK Cmax training dataset.

Reads data/external/mmpk/mmpk_cmax_training.csv, validates each drug,
detects overlap with platinum benchmark, and writes clean data to
data/ml/clinical/mmpk_clean.csv.

Validation filters:
  - SMILES must be RDKit-parseable
  - Dose: 0.1–5000 mg
  - Cmax: > 0
  - Cmax/dose ratio: < 1.0 mg/L per mg (sanity: higher implies >100% F at Vd=1L)
"""

import csv
import json
import math
import sys
from pathlib import Path

from rdkit import Chem, RDLogger

# Suppress RDKit warnings for unparseable SMILES
RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
MMPK_INPUT = ROOT / "data" / "external" / "mmpk" / "mmpk_cmax_training.csv"
PLATINUM_PATH = ROOT / "data" / "clinical" / "platinum_reference.json"
OUTPUT_PATH = ROOT / "data" / "ml" / "clinical" / "mmpk_clean.csv"


def load_platinum_smiles(path: Path) -> set[str]:
    """Load canonical SMILES from platinum reference JSON."""
    with open(path) as f:
        data = json.load(f)
    canonical = set()
    for drug_info in data["drugs"].values():
        smi = drug_info.get("smiles", "")
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            canonical.add(Chem.MolToSmiles(mol))
    return canonical


def audit_and_clean():
    """Main audit pipeline."""
    platinum_smiles = load_platinum_smiles(PLATINUM_PATH)
    print(f"Platinum benchmark: {len(platinum_smiles)} drugs with valid SMILES\n")

    # Counters for removal reasons
    removal_reasons = {
        "invalid_smiles": 0,
        "dose_out_of_range": 0,
        "cmax_non_positive": 0,
        "cmax_dose_ratio_too_high": 0,
        "duplicate_canonical_smiles": 0,
    }
    removed_examples = {k: [] for k in removal_reasons}

    clean_rows = []
    seen_canonical = {}  # canonical SMILES -> row index (keep first = highest n_studies)
    total_input = 0

    with open(MMPK_INPUT, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_input += 1
            name = row.get("name", "").strip()
            raw_smiles = row.get("canon_smiles", "").strip()
            try:
                dose_mg = float(row["dose_mg"])
            except (ValueError, KeyError):
                removal_reasons["dose_out_of_range"] += 1
                removed_examples["dose_out_of_range"].append(name)
                continue
            try:
                cmax_mg_L = float(row["cmax_mg_L"])
            except (ValueError, KeyError):
                removal_reasons["cmax_non_positive"] += 1
                removed_examples["cmax_non_positive"].append(name)
                continue
            try:
                log_cpd = float(row["log_cmax_per_dose"])
            except (ValueError, KeyError):
                log_cpd = math.log10(cmax_mg_L / dose_mg) if dose_mg > 0 else float("nan")
            n_studies = row.get("n_studies", "")
            try:
                n_studies = int(n_studies)
            except (ValueError, TypeError):
                n_studies = 1

            # --- Validation ---

            # 1. SMILES validity
            mol = Chem.MolFromSmiles(raw_smiles)
            if mol is None:
                removal_reasons["invalid_smiles"] += 1
                removed_examples["invalid_smiles"].append(f"{name} [{raw_smiles[:40]}]")
                continue
            canon_smi = Chem.MolToSmiles(mol)

            # 2. Dose range
            if not (0.1 <= dose_mg <= 5000.0):
                removal_reasons["dose_out_of_range"] += 1
                removed_examples["dose_out_of_range"].append(f"{name} (dose={dose_mg})")
                continue

            # 3. Cmax positive
            if cmax_mg_L <= 0:
                removal_reasons["cmax_non_positive"] += 1
                removed_examples["cmax_non_positive"].append(f"{name} (cmax={cmax_mg_L})")
                continue

            # 4. Cmax/dose ratio sanity check
            ratio = cmax_mg_L / dose_mg
            if ratio >= 1.0:
                removal_reasons["cmax_dose_ratio_too_high"] += 1
                removed_examples["cmax_dose_ratio_too_high"].append(
                    f"{name} (ratio={ratio:.3f}, cmax={cmax_mg_L}, dose={dose_mg})"
                )
                continue

            # 5. Duplicate canonical SMILES (keep first occurrence = highest n_studies since input is sorted)
            if canon_smi in seen_canonical:
                removal_reasons["duplicate_canonical_smiles"] += 1
                removed_examples["duplicate_canonical_smiles"].append(
                    f"{name} (dup of row {seen_canonical[canon_smi]})"
                )
                continue

            # Passed all checks
            in_platinum = canon_smi in platinum_smiles
            seen_canonical[canon_smi] = total_input

            clean_rows.append(
                {
                    "name": name,
                    "smiles": canon_smi,
                    "dose_mg": dose_mg,
                    "cmax_mg_L": cmax_mg_L,
                    "log_cmax_per_dose": log_cpd,
                    "n_studies": n_studies,
                    "in_platinum": in_platinum,
                }
            )

    # --- Write output ---
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "smiles",
        "dose_mg",
        "cmax_mg_L",
        "log_cmax_per_dose",
        "n_studies",
        "in_platinum",
    ]
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clean_rows)

    # --- Audit report ---
    total_removed = sum(removal_reasons.values())
    overlap_count = sum(1 for r in clean_rows if r["in_platinum"])
    non_overlap = len(clean_rows) - overlap_count

    print("=" * 60)
    print("MMPK DATA AUDIT REPORT")
    print("=" * 60)
    print(f"\nInput:   {total_input} drugs")
    print(f"Kept:    {len(clean_rows)} drugs ({len(clean_rows) / total_input * 100:.1f}%)")
    print(f"Removed: {total_removed} drugs ({total_removed / total_input * 100:.1f}%)")
    print()

    print("--- Removal reasons ---")
    for reason, count in removal_reasons.items():
        if count > 0:
            print(f"  {reason}: {count}")
            examples = removed_examples[reason][:5]
            for ex in examples:
                print(f"    - {ex}")
            if len(removed_examples[reason]) > 5:
                print(f"    ... and {len(removed_examples[reason]) - 5} more")
    print()

    print("--- Platinum overlap ---")
    print(f"  In platinum:     {overlap_count} drugs")
    print(f"  NOT in platinum: {non_overlap} drugs (novel training data)")
    print()

    # Distribution stats
    doses = [r["dose_mg"] for r in clean_rows]
    cmaxes = [r["cmax_mg_L"] for r in clean_rows]
    studies = [r["n_studies"] for r in clean_rows]
    print("--- Distribution stats ---")
    print(
        f"  Dose (mg):    min={min(doses):.1f}, median={sorted(doses)[len(doses) // 2]:.1f}, max={max(doses):.1f}"
    )
    print(
        f"  Cmax (mg/L):  min={min(cmaxes):.4f}, median={sorted(cmaxes)[len(cmaxes) // 2]:.4f}, max={max(cmaxes):.4f}"
    )
    print(
        f"  N studies:    min={min(studies)}, median={sorted(studies)[len(studies) // 2]}, max={max(studies)}"
    )
    print()

    print(f"Output written to: {OUTPUT_PATH}")
    print(f"  {len(clean_rows)} rows x {len(fieldnames)} columns")

    return len(clean_rows), total_removed, overlap_count


if __name__ == "__main__":
    n_kept, n_removed, n_overlap = audit_and_clean()
    # Exit with error if fewer than 900 drugs kept
    if n_kept < 900:
        print(f"\nWARNING: Only {n_kept} drugs kept (expected >900)")
        sys.exit(1)
