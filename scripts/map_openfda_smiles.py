#!/usr/bin/env python3
"""Map OpenFDA-extracted drugs to SMILES for multi-metric validation.

Reads openfda_pk_extracted.csv (drug_name, parameter, value, unit, context)
and adme_reference.csv (name, smiles, ...), maps each drug to a SMILES string
via case-insensitive name matching or PubChem API fallback, then outputs
openfda_validation.csv with per-drug PK availability flags.
"""

import csv
import json
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OPENFDA_CSV = ROOT / "data" / "ml" / "clinical" / "openfda_pk_extracted.csv"
ADME_REF_CSV = ROOT / "data" / "adme_reference.csv"
OUTPUT_CSV = ROOT / "data" / "ml" / "clinical" / "openfda_validation.csv"

PUBCHEM_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/CanonicalSMILES/JSON"
)
PUBCHEM_DELAY = 0.3  # seconds between requests

# Hardcoded SMILES for common drugs not in adme_reference.csv
# (fallback when PubChem API is unavailable)
KNOWN_SMILES: dict[str, str] = {
    "ezetimibe": "O=C1[C@@H](CC[C@@H](c2ccc(F)cc2)c2ccc(O)cc2)N1c1ccc(F)cc1",
    "hydroxychloroquine": "CCN(CCO)CCCC(C)Nc1ccnc2cc(Cl)ccc12",
    "lorazepam": "OC1N=C(c2ccccc2Cl)c2cc(Cl)ccc2NC1=O",
    "oxazepam": "OC1N=C(c2ccccc2)c2cc(Cl)ccc2NC1=O",
}


def load_adme_reference(path: Path) -> dict[str, str]:
    """Load adme_reference.csv and return {lowercase_name: smiles}."""
    mapping = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"].strip().lower()
            smiles = row["smiles"].strip()
            if smiles:
                mapping[name] = smiles
    return mapping


def load_openfda_params(path: Path) -> dict[str, dict[str, str]]:
    """Load openfda_pk_extracted.csv → {drug: {parameter: value}}."""
    params: dict[str, dict[str, str]] = defaultdict(dict)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            drug = row["drug_name"].strip().lower()
            param = row["parameter"].strip().lower()
            value = row["value"].strip()
            params[drug][param] = value
    return dict(params)


def lookup_pubchem(drug_name: str) -> str | None:
    """Look up canonical SMILES from PubChem REST API. Returns None on failure."""
    # Try original name, then with underscores replaced by spaces
    variants = [drug_name]
    if "_" in drug_name:
        variants.append(drug_name.replace("_", " "))

    for name in variants:
        url = PUBCHEM_URL.format(urllib.request.quote(name))
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                props = data.get("PropertyTable", {}).get("Properties", [])
                if props:
                    smiles = props[0].get("CanonicalSMILES", "")
                    if smiles:
                        return smiles
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError,
                TimeoutError, OSError) as e:
            print(f"  PubChem lookup failed for '{name}': {e}")
            continue
    return None


def main():
    # Load reference data
    adme_ref = load_adme_reference(ADME_REF_CSV)
    print(f"Loaded {len(adme_ref)} compounds from adme_reference.csv")

    # Load OpenFDA parameters
    drug_params = load_openfda_params(OPENFDA_CSV)
    drugs = sorted(drug_params.keys())
    print(f"Found {len(drugs)} unique drugs in openfda_pk_extracted.csv\n")

    # Map drugs to SMILES
    results = []
    mapped = 0
    pubchem_needed = []

    for drug in drugs:
        # Try adme_reference first (case-insensitive, handle underscores)
        smiles = adme_ref.get(drug)
        source = "adme_reference" if smiles else None

        # Also try with underscores replaced by spaces and vice versa
        if not smiles:
            alt = drug.replace("_", " ")
            smiles = adme_ref.get(alt)
            source = "adme_reference" if smiles else None
        if not smiles:
            alt = drug.replace(" ", "_")
            smiles = adme_ref.get(alt)
            source = "adme_reference" if smiles else None

        # Try hardcoded fallback
        if not smiles:
            smiles = KNOWN_SMILES.get(drug)
            source = "known_smiles" if smiles else None

        if not smiles:
            pubchem_needed.append(drug)

        results.append({"drug": drug, "smiles": smiles, "source": source})

    # PubChem fallback for unmatched drugs
    if pubchem_needed:
        print(f"Looking up {len(pubchem_needed)} drugs via PubChem API...")
        for drug in pubchem_needed:
            time.sleep(PUBCHEM_DELAY)
            smiles = lookup_pubchem(drug)
            # Update the result entry
            for r in results:
                if r["drug"] == drug:
                    if smiles:
                        r["smiles"] = smiles
                        r["source"] = "pubchem"
                        print(f"  Found: {drug}")
                    else:
                        print(f"  NOT FOUND: {drug}")
                    break

    # Count mappings
    for r in results:
        if r["smiles"]:
            mapped += 1

    print(f"\nMapped {mapped}/{len(drugs)} drugs to SMILES")

    # Build output rows
    output_rows = []
    for r in results:
        drug = r["drug"]
        params = drug_params[drug]

        has_cmax = "cmax" in params
        has_auc = "auc" in params
        has_thalf = "t_half" in params
        has_f = "bioavailability" in params

        row = {
            "drug_name": drug,
            "smiles": r["smiles"] or "",
            "smiles_source": r["source"] or "",
            "has_cmax": int(has_cmax),
            "has_auc": int(has_auc),
            "has_thalf": int(has_thalf),
            "has_F": int(has_f),
            "cmax_raw": params.get("cmax", ""),
            "auc_raw": params.get("auc", ""),
            "thalf_h": params.get("t_half", ""),
            "bioavailability": params.get("bioavailability", ""),
        }
        output_rows.append(row)

    # Write output CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "drug_name", "smiles", "smiles_source",
        "has_cmax", "has_auc", "has_thalf", "has_F",
        "cmax_raw", "auc_raw", "thalf_h", "bioavailability",
    ]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} rows to {OUTPUT_CSV}\n")

    # Summary
    gold = sum(1 for r in output_rows if r["smiles"] and r["has_cmax"] and r["has_auc"])
    silver = sum(1 for r in output_rows if r["smiles"] and r["has_thalf"] and not (r["has_cmax"] and r["has_auc"]))
    has_smiles = sum(1 for r in output_rows if r["smiles"])

    print("=== Summary ===")
    print(f"Total drugs:        {len(output_rows)}")
    print(f"With SMILES:        {has_smiles}")
    print(f"Gold (cmax+auc):    {gold}")
    print(f"Silver (t_half):    {silver}")
    print(f"No SMILES:          {len(output_rows) - has_smiles}")


if __name__ == "__main__":
    main()
