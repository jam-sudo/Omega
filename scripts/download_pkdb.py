#!/usr/bin/env python3
"""
PK-DB and OpenFDA data downloader for Omega PBPK clinical data pipeline (Branch D).

Usage:
    python3 download_pkdb.py

Outputs:
    data/ml/pkdb/raw/<drug>_pkdb_studies.json    -- PK-DB study metadata per drug
    data/ml/pkdb/processed/<drug>_pkdb.csv       -- processed PK params from PK-DB
    data/ml/fda/raw/<drug>_fda.json              -- OpenFDA label PK section
    data/ml/fda/processed/<drug>_fda_params.csv  -- extracted PK params
    data/ml/pkdb/manifest.json                   -- combined manifest
"""

import json
import re
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_PKDB = "https://pk-db.com/api/v1"
BASE_FDA = "https://api.fda.gov/drug/label.json"
RATE_LIMIT = 1.2  # seconds between requests

RAW_DIR = Path("raw")
PROC_DIR = Path("processed")
FDA_RAW = Path("../fda/raw")
FDA_PROC = Path("../fda/processed")

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)
FDA_RAW.mkdir(parents=True, exist_ok=True)
FDA_PROC.mkdir(parents=True, exist_ok=True)

# 25 benchmark drugs (from benchmarks/datasets/)
BENCHMARK_DRUGS = [
    "acetaminophen",
    "amoxicillin",
    "atenolol",
    "atorvastatin",
    "caffeine",
    "carbamazepine",
    "amphetamine",  # d_amphetamine
    "diazepam",
    "digoxin",
    "fluconazole",
    "fluoxetine",
    "furosemide",
    "gabapentin",
    "ibuprofen",
    "metformin",
    "metoprolol",
    "midazolam",
    "nifedipine",
    "omeprazole",
    "phenytoin",
    "propranolol",
    "theophylline",
    "verapamil",
    "warfarin",
]

# Additional drugs to expand coverage to ~50
ADDITIONAL_DRUGS = [
    "simvastatin",
    "lisinopril",
    "amlodipine",
    "losartan",
    "sertraline",
    "alprazolam",
    "lorazepam",
    "clonazepam",
    "cyclosporine",
    "tacrolimus",
    "clarithromycin",
    "erythromycin",
    "ciprofloxacin",
    "levofloxacin",
    "rifampicin",
    "ketoconazole",
    "itraconazole",
    "fluconazole",
    "hydroxychloroquine",
    "chloroquine",
    "lamotrigine",
    "valproic acid",
    "lithium",
    "haloperidol",
    "risperidone",
    "olanzapine",
]

_last_req_time = 0.0


def _get(url: str, params: dict | None = None, timeout: int = 30) -> dict:
    global _last_req_time
    elapsed = time.monotonic() - _last_req_time
    if elapsed < RATE_LIMIT:
        time.sleep(RATE_LIMIT - elapsed)
    resp = requests.get(url, params=params, timeout=timeout)
    _last_req_time = time.monotonic()
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# PK-DB helpers
# ---------------------------------------------------------------------------


def pkdb_get_studies_for_drug(drug_name: str) -> dict:
    """Get study metadata for a drug from PK-DB."""
    try:
        data = _get(f"{BASE_PKDB}/studies/", {"substance": drug_name, "page_size": 200})
        studies = data.get("data", {}).get("data", [])
        total = data.get("data", {}).get("count", len(studies))
        tc_studies = [s for s in studies if s.get("timecourse_count", 0) > 0]
        output_studies = [s for s in studies if s.get("output_count", 0) > 0]
        return {
            "drug": drug_name,
            "total_studies": total,
            "studies_with_timecourses": len(tc_studies),
            "total_timecourses": sum(s.get("timecourse_count", 0) for s in tc_studies),
            "studies_with_outputs": len(output_studies),
            "total_outputs": sum(s.get("output_count", 0) for s in output_studies),
            "study_ids": [s["sid"] for s in studies[:20]],
            "timecourse_studies": [
                {"sid": s["sid"], "name": s["name"], "tc_count": s["timecourse_count"]}
                for s in tc_studies[:10]
            ],
            "licences": list({s.get("licence", "unknown") for s in studies}),
            "raw_studies": studies[:5],  # first 5 for inspection
        }
    except requests.RequestException as exc:
        return {"drug": drug_name, "error": str(exc)}


# ---------------------------------------------------------------------------
# PK-DB timecourse access test
# ---------------------------------------------------------------------------


def pkdb_test_timecourse_access() -> dict:
    """Test whether the timecourses endpoint is accessible without auth."""
    result = {}

    # Try with no filters
    try:
        data = _get(f"{BASE_PKDB}/pkdata/timecourses/", {"page_size": 5})
        count = data.get("data", {}).get("count", 0)
        result["timecourses_no_filter"] = count
    except Exception as e:
        result["timecourses_no_filter_error"] = str(e)

    # Try outputs
    try:
        data = _get(f"{BASE_PKDB}/outputs/", {"page_size": 5})
        count = data.get("data", {}).get("count", 0)
        result["outputs_no_filter"] = count
    except Exception as e:
        result["outputs_no_filter_error"] = str(e)

    # Try pkdata/groups (known to work)
    try:
        data = _get(f"{BASE_PKDB}/pkdata/groups/", {"page_size": 1})
        count = data.get("data", {}).get("count", 0)
        result["pkdata_groups"] = count
    except Exception as e:
        result["pkdata_groups_error"] = str(e)

    # Try pkdata/individuals (known to work)
    try:
        data = _get(f"{BASE_PKDB}/pkdata/individuals/", {"page_size": 1})
        count = data.get("data", {}).get("count", 0)
        result["pkdata_individuals"] = count
    except Exception as e:
        result["pkdata_individuals_error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# OpenFDA helpers
# ---------------------------------------------------------------------------

_PK_PATTERNS = {
    "cmax_ng_mL": re.compile(
        r"[Cc][\s_]?max\s*(?:was|of|=|:)?\s*(\d+\.?\d*)\s*(ng/mL|mcg/mL|µg/mL|mg/L)"
    ),
    "auc_ng_h_mL": re.compile(
        r"AUC\s*(?:0-∞|0-inf|infinity|0-t)?\s*(?:was|of|=|:)?\s*(\d+\.?\d*)\s*"
        r"(ng·h/mL|ng\*h/mL|mcg·h/mL|h·ng/mL|ng/mL·h)"
    ),
    "t_half_h": re.compile(
        r"(?:t½|t1/2|half.?life)\s*(?:was|of|=|:)?\s*(\d+\.?\d*)\s*(h(?:ours?)?|hr)"
    ),
    "bioavailability_pct": re.compile(
        r"(?:bioavailability|absolute bioavailability|F)\s*(?:was|of|is|=|:)?\s*"
        r"(\d+\.?\d*)\s*%"
    ),
    "vd_L_kg": re.compile(
        r"(?:volume of distribution|Vd|Vss|Vdss)\s*(?:was|of|=|:)?\s*(\d+\.?\d*)\s*(L/kg)"
    ),
    "clearance_L_h": re.compile(
        r"(?:clearance|CL|CL/F)\s*(?:was|of|=|:)?\s*(\d+\.?\d*)\s*(L/h(?:r)?(?:/kg)?)"
    ),
    "protein_binding_pct": re.compile(
        r"(?:protein binding|plasma protein binding|bound to plasma protein)\s*"
        r"(?:is|was|of|=|:)?\s*(?:approximately\s*)?(\d+\.?\d*)\s*%"
    ),
}


def fda_get_pk_text(drug_name: str) -> dict:
    """Download PK text from OpenFDA drug labels."""
    out = {"drug": drug_name, "found": False, "pk_text": "", "params": {}}
    try:
        data = _get(
            BASE_FDA,
            {
                "search": f"openfda.generic_name:{drug_name}",
                "limit": 3,
            },
        )
        if not data.get("results"):
            # Try brand name search
            data = _get(
                BASE_FDA,
                {
                    "search": f"openfda.brand_name:{drug_name.upper()}",
                    "limit": 3,
                },
            )
        if not data.get("results"):
            out["error"] = "No FDA label found"
            return out

        # Pick the result with pharmacokinetics section
        pk_text = ""
        for result in data["results"]:
            pk_section = result.get("pharmacokinetics", [])
            clin_pharm = result.get("clinical_pharmacology", [])
            combined = " ".join(pk_section + clin_pharm)
            if combined:
                pk_text = combined
                break

        if not pk_text:
            out["error"] = "No PK section in label"
            return out

        out["found"] = True
        out["pk_text"] = pk_text[:5000]

        # Extract PK parameters
        params = {}
        for param, pattern in _PK_PATTERNS.items():
            matches = pattern.findall(pk_text)
            if matches:
                try:
                    values = [float(m[0]) for m in matches]
                    params[param] = {
                        "values": values,
                        "mean": sum(values) / len(values),
                        "unit": matches[0][1] if isinstance(matches[0], tuple) else "",
                    }
                except (ValueError, IndexError):
                    pass

        out["params"] = params

    except requests.RequestException as exc:
        out["error"] = str(exc)
    return out


# ---------------------------------------------------------------------------
# Main download loop
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("PK-DB + OpenFDA Clinical Data Downloader")
    print("=" * 60)

    # Step 1: Test PK-DB access
    print("\n[Step 1] Testing PK-DB API access...")
    access_test = pkdb_test_timecourse_access()
    print(f"  pkdata/timecourses (no filter): {access_test.get('timecourses_no_filter', 'ERROR')}")
    print(f"  outputs (no filter):            {access_test.get('outputs_no_filter', 'ERROR')}")
    print(f"  pkdata/groups (demographics):   {access_test.get('pkdata_groups', 'ERROR')}")
    print(f"  pkdata/individuals:             {access_test.get('pkdata_individuals', 'ERROR')}")

    # Step 2: PK-DB study inventory for benchmark drugs
    print("\n[Step 2] Inventorying PK-DB for benchmark drugs...")
    all_drugs = BENCHMARK_DRUGS + [d for d in ADDITIONAL_DRUGS if d not in BENCHMARK_DRUGS]
    pkdb_results = {}

    for drug in all_drugs:
        print(f"  Checking: {drug}")
        result = pkdb_get_studies_for_drug(drug)
        pkdb_results[drug] = result
        raw_path = RAW_DIR / f"{drug.replace(' ', '_')}_pkdb_studies.json"
        raw_path.write_text(json.dumps(result, indent=2))

    # Step 3: OpenFDA PK parameters
    print("\n[Step 3] Downloading FDA label PK parameters...")
    fda_results = {}

    for drug in all_drugs:
        print(f"  FDA: {drug}")
        result = fda_get_pk_text(drug)
        fda_results[drug] = result
        raw_path = FDA_RAW / f"{drug.replace(' ', '_')}_fda.json"
        raw_path.write_text(json.dumps(result, indent=2))

    # Step 4: Build manifest
    print("\n[Step 4] Building manifest...")
    manifest = {
        "generated": time.strftime("%Y-%m-%d"),
        "api_access_test": access_test,
        "timecourse_data_accessible": access_test.get("timecourses_no_filter", 0) > 0,
        "authentication_required_for_timecourses": access_test.get("timecourses_no_filter", 0) == 0,
        "drugs": {},
    }

    for drug in all_drugs:
        pkdb = pkdb_results.get(drug, {})
        fda = fda_results.get(drug, {})
        manifest["drugs"][drug] = {
            "pkdb_total_studies": pkdb.get("total_studies", 0),
            "pkdb_studies_with_timecourses": pkdb.get("studies_with_timecourses", 0),
            "pkdb_total_timecourses": pkdb.get("total_timecourses", 0),
            "pkdb_studies_with_outputs": pkdb.get("studies_with_outputs", 0),
            "pkdb_total_outputs": pkdb.get("total_outputs", 0),
            "pkdb_licences": pkdb.get("licences", []),
            "fda_label_found": fda.get("found", False),
            "fda_params_extracted": list(fda.get("params", {}).keys()),
            "pkdb_error": pkdb.get("error"),
            "fda_error": fda.get("error"),
        }

    manifest_path = Path("manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print("  Saved manifest.json")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    with_tc = [d for d, m in manifest["drugs"].items() if m["pkdb_studies_with_timecourses"] > 0]
    with_outputs = [d for d, m in manifest["drugs"].items() if m["pkdb_studies_with_outputs"] > 0]
    fda_found = [d for d, m in manifest["drugs"].items() if m["fda_label_found"]]

    print(f"Drugs with PK-DB timecourse studies (metadata only): {len(with_tc)}/{len(all_drugs)}")
    print(
        f"Drugs with PK-DB output studies (metadata only):     {len(with_outputs)}/{len(all_drugs)}"
    )
    print(f"Drugs with OpenFDA PK section:                       {len(fda_found)}/{len(all_drugs)}")
    print(
        f"\nAuthentication required for PK-DB data: {manifest['authentication_required_for_timecourses']}"
    )
    print("\nTop drugs by PK-DB timecourse count:")
    ranked = sorted(
        manifest["drugs"].items(), key=lambda x: x[1]["pkdb_total_timecourses"], reverse=True
    )
    for d, m in ranked[:10]:
        print(
            f"  {d}: {m['pkdb_total_timecourses']} timecourses in {m['pkdb_studies_with_timecourses']} studies"
        )

    return manifest


if __name__ == "__main__":
    import os

    os.chdir(Path(__file__).parent)
    manifest = main()
