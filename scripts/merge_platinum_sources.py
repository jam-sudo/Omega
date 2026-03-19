#!/usr/bin/env python3
"""Merge all Cmax sources into unified platinum_reference.json.

Priority order:
  1. Gold-24 (existing platinum_reference.json) — highest priority
  2. DailyMed FDA labels (table extraction)
  3. DailyMed FDA labels (text extraction)
  4. PK-DB timecourses
  5. Expanded Cmax CSV (existing, gap-fill)

Deduplication by normalized drug name. Each entry validated against
platinum_schema before inclusion.

Usage:
    python scripts/merge_platinum_sources.py
    python scripts/merge_platinum_sources.py --verbose
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from omega_pbpk.data.platinum_schema import (  # noqa: E402
    ValidationError,
    load_platinum_reference,
    save_platinum_reference,
    validate_entry,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PLATINUM_REF = repo_root / "data" / "clinical" / "platinum_reference.json"
DAILYMED_CSV = repo_root / "data" / "ml" / "clinical" / "dailymed_extracted.csv"
PKDB_CSV = repo_root / "data" / "ml" / "clinical" / "pkdb_extracted.csv"
EXPANDED_CSV = repo_root / "data" / "ml" / "clinical" / "expanded_cmax.csv"
OUTPUT = PLATINUM_REF  # overwrite in place

# Drugs known to have non-linear PK
NONLINEAR_PK_DRUGS = {
    "omeprazole",
    "phenytoin",
    "diazepam",
    "carbamazepine",
    "theophylline",
    "valproic acid",
}

# Drugs where pipeline has been tuned (CLint anchors, VDss anchors, etc.)
# This only applies to the original gold-24 — new drugs are clean
GOLD24_CONTAMINATED = {
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


def normalize_name(name: str) -> str:
    """Normalize drug name for deduplication."""
    name = name.strip().lower()
    # Remove common suffixes
    for suffix in [
        " hydrochloride",
        " hcl",
        " sulfate",
        " sodium",
        " potassium",
        " mesylate",
        " maleate",
        " fumarate",
        " tartrate",
        " citrate",
        " acetate",
        " phosphate",
        " bisulfate",
        " succinate",
        " besylate",
        " calcium",
        " dipivoxil",
        " proxetil",
    ]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    # Normalize spaces and hyphens
    name = re.sub(r"[\s_-]+", "_", name)
    return name


def load_dailymed(path: Path) -> dict[str, dict]:
    """Load DailyMed extracted CSV as {normalized_name: entry}."""
    if not path.exists():
        return {}
    entries = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            name = normalize_name(row["drug"])
            cmax = float(row["cmax_mg_L"])
            dose = float(row["dose_mg"]) if row.get("dose_mg") and row["dose_mg"] != "" else None
            if cmax <= 0:
                continue
            entries[name] = {
                "cmax_mg_L": cmax,
                "dose_mg": dose,
                "extraction_method": row.get("extraction_method", "text"),
                "fasted_confidence": row.get("fasted_confidence", "assumed_fasted"),
                "source": f"DailyMed SPL ({row.get('extraction_method', 'text')})",
            }
    return entries


def load_pkdb(path: Path) -> dict[str, dict]:
    """Load PK-DB extracted CSV."""
    if not path.exists():
        return {}
    entries = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            name = normalize_name(row["drug"])
            cmax = float(row["obs_cmax"])
            dose = float(row["dose_mg"]) if row.get("dose_mg") and row["dose_mg"] != "" else None
            smiles = row.get("smiles", "")
            if cmax <= 0:
                continue
            entries[name] = {
                "cmax_mg_L": cmax,
                "dose_mg": dose,
                "smiles": smiles,
                "source_study": row.get("source_study", ""),
                "source": f"PK-DB timecourse ({row.get('source_study', '')})",
            }
    return entries


def load_expanded(path: Path) -> dict[str, dict]:
    """Load expanded Cmax CSV."""
    if not path.exists():
        return {}
    entries = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            name = normalize_name(row["drug"])
            try:
                cmax = float(row["cmax_mg_L"])
            except (ValueError, KeyError):
                continue
            if cmax <= 0:
                continue
            dose = float(row["dose_mg"]) if row.get("dose_mg") else None
            smiles = row.get("smiles", "")
            tier = row.get("tier", "expanded")
            source = row.get("source", f"expanded_cmax ({tier})")
            entries[name] = {
                "cmax_mg_L": cmax,
                "dose_mg": dose,
                "smiles": smiles,
                "tier": tier,
                "source": source,
            }
    return entries


def fetch_smiles_pubchem(drug_name: str) -> str | None:
    """Look up SMILES from PubChem REST API."""
    import requests

    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/property/CanonicalSMILES/JSON"
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        props = resp.json().get("PropertyTable", {}).get("Properties", [])
        if props:
            return props[0].get("CanonicalSMILES")
    except Exception:
        return None
    return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Merge all Cmax sources into platinum reference")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--no-api", action="store_true", help="Skip PubChem lookups for missing SMILES"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Platinum Reference Merge")
    print("=" * 60)

    # 1. Load existing platinum (gold-24 base)
    existing = load_platinum_reference(PLATINUM_REF)
    print(f"  Existing platinum: {len(existing)} drugs")

    # 2. Load new sources
    dailymed = load_dailymed(DAILYMED_CSV)
    print(f"  DailyMed extracted: {len(dailymed)} drugs")

    pkdb = load_pkdb(PKDB_CSV)
    print(f"  PK-DB timecourses: {len(pkdb)} drugs")

    expanded = load_expanded(EXPANDED_CSV)
    print(f"  Expanded CSV: {len(expanded)} drugs")

    # 3. Collect all drug names (normalized)
    # Priority: platinum > dailymed_table > dailymed_text > pkdb > expanded
    existing_normalized = {normalize_name(k): k for k in existing}

    # Track what we add
    added = {"dailymed_table": 0, "dailymed_text": 0, "pkdb": 0, "expanded": 0}
    skipped = {"no_smiles": [], "no_dose": [], "validation_fail": [], "already_in": []}
    merged = dict(existing)  # start with gold-24

    # Build SMILES lookup from all sources (expanded has best coverage)
    smiles_lookup = {}
    for name, entry in expanded.items():
        s = entry.get("smiles", "")
        if s and len(s) >= 3:
            smiles_lookup[name] = s
    for name, entry in pkdb.items():
        s = entry.get("smiles", "")
        if s and len(s) >= 3:
            smiles_lookup[name] = s

    # 4. Process each source in priority order
    all_candidates = {}

    # DailyMed (table > text)
    for name, entry in dailymed.items():
        if name not in all_candidates:
            # Enrich DailyMed entries with SMILES from other sources
            if not entry.get("smiles") or len(entry.get("smiles", "")) < 3:
                entry["smiles"] = smiles_lookup.get(name, "")
            all_candidates[name] = ("dailymed", entry)

    # PK-DB
    for name, entry in pkdb.items():
        if name not in all_candidates:
            all_candidates[name] = ("pkdb", entry)

    # Expanded
    for name, entry in expanded.items():
        if name not in all_candidates:
            all_candidates[name] = ("expanded", entry)

    print(f"\n  Total unique candidates: {len(all_candidates)}")

    for norm_name, (source, entry) in sorted(all_candidates.items()):
        # Skip if already in platinum (by normalized name)
        if norm_name in existing_normalized:
            skipped["already_in"].append(norm_name)
            continue

        # Also check original names
        original_name = norm_name.replace("_", " ")
        if original_name in merged:
            skipped["already_in"].append(norm_name)
            continue

        cmax_mg_L = entry["cmax_mg_L"]
        dose_mg = entry.get("dose_mg")
        smiles = entry.get("smiles", "")

        # Need dose for validation
        if dose_mg is None or dose_mg <= 0:
            skipped["no_dose"].append(norm_name)
            if args.verbose:
                print(f"  SKIP {norm_name}: no dose")
            continue

        # Need SMILES — try lookup table, then PubChem
        if not smiles or len(smiles) < 3:
            smiles = smiles_lookup.get(norm_name, "")
        if not smiles or len(smiles) < 3:
            if not args.no_api:
                smiles = fetch_smiles_pubchem(original_name)
            if not smiles or len(smiles) < 3:
                skipped["no_smiles"].append(norm_name)
                if args.verbose:
                    print(f"  SKIP {norm_name}: no SMILES")
                continue

        # Build platinum entry
        fasted = entry.get("fasted_confidence", "assumed_fasted")
        if source == "dailymed":
            source_type = "fda_label"
            method = entry.get("extraction_method", "text")
            data_quality = "fda_label_exact" if method == "table" else "fda_label_dose_normalized"
            source_id = entry.get("source", f"DailyMed ({method})")
        elif source == "pkdb":
            source_type = "pkdb"
            data_quality = "clinical_exact"
            source_id = entry.get("source_study", "PK-DB")
        else:
            # Expanded CSV
            tier = entry.get("tier", "expanded")
            if "fda" in tier.lower():
                source_type = "fda_label"
                data_quality = "fda_label_exact"
            elif "gold" in tier.lower():
                source_type = "literature"
                data_quality = "clinical_exact"
            else:
                source_type = "literature"
                data_quality = "clinical_exact"
            source_id = entry.get("source", f"expanded_cmax ({tier})")

        plat_entry = {
            "smiles": smiles,
            "dose_mg": dose_mg,
            "cmax_mg_L": cmax_mg_L,
            "source_type": source_type,
            "source_id": source_id,
            "fasted_confidence": fasted,
            "formulation": "IR",
            "route": "oral",
            "population": "healthy",
            "single_dose": True,
            "tuning_contaminated": False,  # new drugs are clean
            "nonlinear_pk": original_name.lower() in NONLINEAR_PK_DRUGS,
            "data_quality": data_quality,
        }

        # Validate
        try:
            validate_entry(original_name, plat_entry)
        except ValidationError as e:
            skipped["validation_fail"].append((norm_name, str(e)))
            if args.verbose:
                print(f"  FAIL {norm_name}: {e}")
            continue

        merged[original_name] = plat_entry
        src_label = (
            "dailymed_table"
            if source == "dailymed" and entry.get("extraction_method") == "table"
            else ("dailymed_text" if source == "dailymed" else source)
        )
        added[src_label] = added.get(src_label, 0) + 1
        if args.verbose:
            print(f"  ADD  {original_name}: Cmax={cmax_mg_L:.4f} mg/L [{src_label}]")

    # 5. Save
    save_platinum_reference(merged, OUTPUT, version="2.0")

    # 6. Report
    print(f"\n{'=' * 60}")
    print("MERGE SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Gold-24 base:         {len(existing)}")
    print(f"  Added from DailyMed (table): {added.get('dailymed_table', 0)}")
    print(f"  Added from DailyMed (text):  {added.get('dailymed_text', 0)}")
    print(f"  Added from PK-DB:            {added.get('pkdb', 0)}")
    print(f"  Added from expanded CSV:     {added.get('expanded', 0)}")
    print("  ---")
    print(f"  Total platinum drugs: {len(merged)}")
    print(f"  Skipped (already in):     {len(skipped['already_in'])}")
    print(f"  Skipped (no dose):        {len(skipped['no_dose'])}")
    print(f"  Skipped (no SMILES):      {len(skipped['no_smiles'])}")
    print(f"  Skipped (validation):     {len(skipped['validation_fail'])}")

    # Quality breakdown
    n_clean = sum(1 for d in merged.values() if not d.get("tuning_contaminated"))
    n_contaminated = len(merged) - n_clean
    n_fasted = sum(1 for d in merged.values() if d.get("fasted_confidence") == "confirmed_fasted")
    n_nonlinear = sum(1 for d in merged.values() if d.get("nonlinear_pk"))
    print(f"\n  Clean (non-contaminated): {n_clean}")
    print(f"  Tuning contaminated:     {n_contaminated}")
    print(f"  Confirmed fasted:        {n_fasted}")
    print(f"  Non-linear PK:           {n_nonlinear}")

    if skipped["validation_fail"] and args.verbose:
        print("\n  Validation failures:")
        for name, err in skipped["validation_fail"][:10]:
            print(f"    {name}: {err}")

    print(f"\n  Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
