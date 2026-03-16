#!/usr/bin/env python3
"""Fetch real clinical C(t) timecourse data from PK-DB for Omega benchmark drugs.

PK-DB API: https://pk-db.com/api/v1/
TSV files: https://pk-db.com/media/data/.{StudyName}_{FileLabel}.tsv

Pipeline:
  1. Paginate through all studies with timecourse_count > 0
  2. For each study, fetch detail to get file URLs
  3. Download TSV files, filter for plasma/blood/serum concentration data
  4. Match substances to our target drugs
  5. Extract dose/route from intervention strings + interventions API
  6. Save to data/clinical/pkdb_timecourses.json

Usage:
  python scripts/fetch_pkdb_clinical.py [--no-cache] [--verbose]
"""

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://pk-db.com/api/v1"
MEDIA_URL = "https://pk-db.com/media/data"
CACHE_DIR = Path("data/ml/pkdb_cache")
OUTPUT_FILE = Path("data/clinical/pkdb_timecourses.json")
RATE_LIMIT_S = 1.0  # seconds between requests
PAGE_SIZE = 200

# Target drugs and their aliases (PK-DB may use different names)
TARGET_DRUGS = {
    "caffeine": ["caffeine"],
    "midazolam": ["midazolam"],
    "warfarin": ["warfarin", "s-warfarin", "r-warfarin"],
    "metoprolol": ["metoprolol"],
    "verapamil": ["verapamil", "s-verapamil", "r-verapamil"],
    "diazepam": ["diazepam"],
    "simvastatin": ["simvastatin"],
    "morphine": ["morphine"],
    "codeine": ["codeine"],
    "ibuprofen": ["ibuprofen", "s-ibuprofen", "r-ibuprofen"],
    "theophylline": ["theophylline"],
    "acetaminophen": ["paracetamol", "acetaminophen"],
    "propranolol": ["propranolol"],
    "phenytoin": ["phenytoin"],
    "carbamazepine": ["carbamazepine"],
    "atenolol": ["atenolol"],
    "fluconazole": ["fluconazole"],
    "furosemide": ["furosemide"],
    "gabapentin": ["gabapentin"],
    "metformin": ["metformin"],
    "omeprazole": ["omeprazole", "s-omeprazole"],
    "amoxicillin": ["amoxicillin"],
    "digoxin": ["digoxin"],
    "nifedipine": ["nifedipine"],
    "fluoxetine": ["fluoxetine"],
}

# Build reverse lookup: alias -> canonical name
ALIAS_TO_DRUG = {}
for drug, aliases in TARGET_DRUGS.items():
    for alias in aliases:
        ALIAS_TO_DRUG[alias.lower()] = drug

# Approximate molecular weights for unit conversion (g/mol)
MW_TABLE = {
    "caffeine": 194.19,
    "midazolam": 325.77,
    "warfarin": 308.33,
    "metoprolol": 267.36,
    "verapamil": 454.60,
    "diazepam": 284.74,
    "simvastatin": 418.57,
    "morphine": 285.34,
    "codeine": 299.36,
    "ibuprofen": 206.28,
    "theophylline": 180.16,
    "acetaminophen": 151.16,
    "propranolol": 259.34,
    "phenytoin": 252.27,
    "carbamazepine": 236.27,
    "atenolol": 266.34,
    "fluconazole": 306.27,
    "furosemide": 330.74,
    "gabapentin": 171.24,
    "metformin": 129.16,
    "omeprazole": 345.42,
    "amoxicillin": 365.40,
    "digoxin": 780.94,
    "nifedipine": 346.33,
    "fluoxetine": 309.33,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VERBOSE = False


def log(msg: str) -> None:
    if VERBOSE:
        print(f"  [DEBUG] {msg}", file=sys.stderr)


def cache_path(url: str) -> Path:
    """Deterministic cache file for a URL."""
    h = hashlib.sha256(url.encode()).hexdigest()[:16]
    return CACHE_DIR / f"{h}.json"


def fetch_url(url: str, use_cache: bool = True, is_tsv: bool = False) -> str | None:
    """Fetch URL with caching and rate limiting. Returns text or None on error."""
    cp = cache_path(url)
    if use_cache and cp.exists():
        data = json.loads(cp.read_text())
        return data.get("content")

    log(f"GET {url}")
    time.sleep(RATE_LIMIT_S)

    try:
        resp = requests.get(
            url, timeout=30, headers={"Accept": "text/plain" if is_tsv else "application/json"}
        )
        if resp.status_code != 200:
            log(f"  HTTP {resp.status_code} for {url}")
            return None
        content = resp.text
    except Exception as e:
        log(f"  Error fetching {url}: {e}")
        return None

    # Cache the response
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps({"url": url, "content": content}))
    return content


def fetch_json(url: str, use_cache: bool = True) -> dict | None:
    """Fetch JSON from URL."""
    content = fetch_url(url, use_cache=use_cache)
    if content is None:
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        log(f"  JSON decode error for {url}")
        return None


# ---------------------------------------------------------------------------
# Drug name matching
# ---------------------------------------------------------------------------


def match_drug(substance: str) -> str | None:
    """Match a substance name to a target drug. Returns canonical name or None.

    Must match the parent drug exactly — reject metabolites like
    '1-hydroxymidazolam', 'paracetamol glucuronide', etc.
    """
    s = substance.lower().strip()

    # Direct alias match
    if s in ALIAS_TO_DRUG:
        return ALIAS_TO_DRUG[s]

    # Handle caffeine with code like "caffeine (137X)"
    for alias, drug in ALIAS_TO_DRUG.items():
        if s.startswith(alias + " (") or s == alias:
            return drug

    return None


# ---------------------------------------------------------------------------
# Route / dose extraction from intervention strings in TSV
# ---------------------------------------------------------------------------


def parse_intervention_string(intervention: str) -> dict:
    """Parse intervention strings like 'iv, po4', 'oral 200mg', 'po RIF', etc.

    Returns dict with 'route' and 'dose_mg' (if parseable).
    """
    s = intervention.lower().strip()
    info = {"route": "unknown", "dose_mg": None}

    # Route detection
    if any(x in s for x in ["oral", " po", "po ", "po,", "po4", "po6", "po8"]):
        info["route"] = "oral"
    elif any(x in s for x in [" iv", "iv ", "iv,", "intravenous"]):
        info["route"] = "iv"
    elif "im" in s.split():
        info["route"] = "im"

    # Dose extraction: look for patterns like "200mg", "200 mg", "0.5mg"
    dose_match = re.search(r"(\d+\.?\d*)\s*mg", s)
    if dose_match:
        info["dose_mg"] = float(dose_match.group(1))

    return info


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------


def convert_to_mg_per_L(value: float, unit: str, drug: str) -> float | None:
    """Convert concentration value to mg/L.

    Common units in PK-DB:
      mg/L, µg/L (ug/L), ng/mL, ng/L, µg/mL, µmol/L (umol/L), nmol/L
    """
    if value is None:
        return None

    unit_lower = unit.lower().strip()

    # Already mg/L
    if unit_lower in ("mg/l", "mg/liter"):
        return value

    # µg/L = mg/L * 0.001
    if unit_lower in ("µg/l", "ug/l", "microgram/l", "microgram/liter"):
        return value / 1000.0

    # ng/mL = µg/L = mg/L * 0.001
    if unit_lower in ("ng/ml", "ng/milliliter"):
        return value / 1000.0

    # µg/mL = mg/L
    if unit_lower in ("µg/ml", "ug/ml", "microgram/ml"):
        return value

    # ng/L = mg/L * 1e-6
    if unit_lower in ("ng/l", "ng/liter"):
        return value / 1e6

    # µmol/L -> mg/L = µmol/L * MW / 1000
    if unit_lower in ("µmol/l", "umol/l", "micromol/l", "µmol/liter"):
        mw = MW_TABLE.get(drug)
        if mw:
            return value * mw / 1000.0
        return None

    # nmol/L -> mg/L = nmol/L * MW / 1e6
    if unit_lower in ("nmol/l", "nmol/liter"):
        mw = MW_TABLE.get(drug)
        if mw:
            return value * mw / 1e6
        return None

    # pmol/L -> mg/L = pmol/L * MW / 1e9
    if unit_lower in ("pmol/l", "pmol/liter"):
        mw = MW_TABLE.get(drug)
        if mw:
            return value * mw / 1e9
        return None

    log(f"  Unknown unit: {unit}")
    return None


# ---------------------------------------------------------------------------
# TSV parsing
# ---------------------------------------------------------------------------


def parse_tsv(content: str, study_name: str) -> list[dict]:
    """Parse a PK-DB TSV file and extract matching drug timecourse data.

    Returns list of dicts with keys:
      drug, study, substance, tissue, measurement, intervention,
      n_subjects, unit, time_h, mean, sd, se
    """
    results = []

    try:
        reader = csv.DictReader(io.StringIO(content), delimiter="\t")
    except Exception as e:
        log(f"  TSV parse error in {study_name}: {e}")
        return results

    for row in reader:
        try:
            # Get fields (handle different column name variants)
            measurement = (
                (row.get("measurement") or row.get("measurement_type") or "").strip().lower()
            )
            tissue = (row.get("tissue") or "").strip().lower()
            substance = (row.get("substance") or "").strip()
            unit = (row.get("unit") or "").strip()
            time_str = (row.get("time") or "").strip()
            mean_str = (row.get("mean") or row.get("value") or "").strip()
            sd_str = (row.get("mean_sd") or row.get("sd") or "").strip()
            se_str = (row.get("mean_se") or row.get("se") or "").strip()
            n_str = (row.get("n") or row.get("count") or "").strip()
            intervention = (row.get("intervention") or "").strip()
            time_unit = (row.get("time_unit") or "hr").strip().lower()

            # Filter: concentration in plasma/blood/serum
            if measurement != "concentration":
                continue
            if tissue not in ("plasma", "blood", "serum"):
                continue

            # Match drug
            drug = match_drug(substance)
            if drug is None:
                continue

            # Parse numeric values
            if not time_str or not mean_str:
                continue
            if time_str.lower() in ("na", "nan", "none", ""):
                continue
            if mean_str.lower() in ("na", "nan", "none", ""):
                continue

            time_val = float(time_str)
            mean_val = float(mean_str)

            # Convert time to hours
            if time_unit in ("min", "minute", "minutes"):
                time_val /= 60.0
            elif time_unit in ("day", "days", "d"):
                time_val *= 24.0

            # Parse SD and SE
            sd_val = (
                float(sd_str)
                if sd_str and sd_str.lower() not in ("nan", "na", "none", "")
                else None
            )
            se_val = (
                float(se_str)
                if se_str and se_str.lower() not in ("nan", "na", "none", "")
                else None
            )

            # Parse n
            n_val = None
            if n_str:
                try:
                    n_val = int(float(n_str))
                except (ValueError, TypeError):
                    pass

            results.append(
                {
                    "drug": drug,
                    "study": study_name,
                    "substance": substance,
                    "tissue": tissue,
                    "measurement": measurement,
                    "intervention": intervention,
                    "n_subjects": n_val,
                    "unit": unit,
                    "time_h": time_val,
                    "mean_raw": mean_val,
                    "sd_raw": sd_val,
                    "se_raw": se_val,
                    "original_unit": unit,
                }
            )

        except (ValueError, TypeError, KeyError) as e:
            log(f"  Row parse error in {study_name}: {e}")
            continue

    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def get_all_studies_with_timecourses(use_cache: bool = True) -> list[dict]:
    """Paginate through studies API, return those with timecourse_count > 0."""
    all_studies = []
    page = 1

    while True:
        url = f"{BASE_URL}/studies/?format=json&page_size={PAGE_SIZE}&page={page}"
        data = fetch_json(url, use_cache=use_cache)
        if data is None:
            print(f"  WARNING: Failed to fetch studies page {page}", file=sys.stderr)
            break

        # Navigate nested structure
        studies_data = data.get("data", data)
        if isinstance(studies_data, dict):
            studies_list = studies_data.get("data", studies_data.get("results", []))
        else:
            studies_list = studies_data

        if not studies_list:
            break

        for study in studies_list:
            tc = study.get("timecourse_count", 0)
            if tc and tc > 0:
                all_studies.append(study)

        # Check pagination
        last_page = data.get("last_page", 1)
        if page >= last_page:
            break
        page += 1

    return all_studies


def get_study_files(sid: str, use_cache: bool = True) -> list[str]:
    """Fetch study detail and return list of TSV file URLs."""
    url = f"{BASE_URL}/_studies/{sid}/?format=json"
    data = fetch_json(url, use_cache=use_cache)
    if data is None:
        return []

    # Navigate to files list
    study_data = data.get("data", data)
    if isinstance(study_data, dict) and "data" in study_data:
        study_data = study_data["data"]

    files = study_data.get("files", [])
    tsv_urls = []

    for f in files:
        # File can be a string URL or a dict with 'file' key
        if isinstance(f, str):
            file_url = f
        elif isinstance(f, dict):
            file_url = f.get("file", f.get("url", ""))
        else:
            continue

        if file_url.endswith(".tsv"):
            # Ensure full URL
            if not file_url.startswith("http"):
                file_url = f"https://pk-db.com{file_url}"
            tsv_urls.append(file_url)

    return tsv_urls


def get_study_interventions(study_name: str, use_cache: bool = True) -> dict:
    """Fetch interventions for a study. Returns dict keyed by intervention label."""
    url = f"{BASE_URL}/interventions/?format=json&page_size=100&study_name={study_name}"
    data = fetch_json(url, use_cache=use_cache)
    if data is None:
        return {}

    interventions = {}
    results = data.get("data", data)
    if isinstance(results, dict):
        results = results.get("data", results.get("results", []))

    if not isinstance(results, list):
        return {}

    for intv in results:
        if not isinstance(intv, dict):
            continue

        # Fields may be strings or dicts with "name" key
        def _get_str(field, record=intv):
            v = record.get(field, "")
            if isinstance(v, dict):
                return (v.get("name") or v.get("sid") or "").lower()
            return (v or "").lower()

        substance = _get_str("substance")
        route = _get_str("route")
        value = intv.get("value")
        unit = (
            (intv.get("unit") or "").lower()
            if isinstance(intv.get("unit"), str)
            else (
                intv.get("unit", {}).get("name", "") if isinstance(intv.get("unit"), dict) else ""
            ).lower()
        )
        application = _get_str("application")
        normed = intv.get("normed", False)

        # Skip normed (duplicate) entries
        if normed:
            continue

        # Convert dose to mg
        dose_mg = None
        if value is not None:
            try:
                dose_mg = float(value)
                if "gram" in unit or unit == "g":
                    dose_mg *= 1000
                elif "microgram" in unit or unit == "µg" or unit == "ug":
                    dose_mg /= 1000
            except (ValueError, TypeError):
                pass

        drug = match_drug(substance)
        if drug:
            key = drug
            if key not in interventions:
                interventions[key] = []
            interventions[key].append(
                {
                    "route": route if route else "unknown",
                    "dose_mg": dose_mg,
                    "application": application,
                }
            )

    return interventions


def build_timecourses(
    raw_points: list[dict],
    study_interventions: dict,
    pmid: str | None,
) -> list[dict]:
    """Group raw data points into timecourse records.

    Groups by (drug, study, intervention) and builds the output structure.
    """
    # Group points by (drug, study, intervention)
    groups = defaultdict(list)
    for pt in raw_points:
        key = (pt["drug"], pt["study"], pt["intervention"])
        groups[key].append(pt)

    timecourses = []
    for (drug, study, intervention_str), points in groups.items():
        # Sort by time
        points.sort(key=lambda p: p["time_h"])

        # Need at least 3 timepoints for a useful curve
        if len(points) < 3:
            log(f"  Skipping {drug}/{study}/{intervention_str}: only {len(points)} points")
            continue

        # Parse route/dose from intervention string
        intv_info = parse_intervention_string(intervention_str)

        # Try to get better dose/route from interventions API
        if drug in study_interventions:
            api_intvs = study_interventions[drug]
            if api_intvs:
                # Prefer oral, single-dose
                best = api_intvs[0]
                for ai in api_intvs:
                    if "oral" in ai.get("route", "") and "single" in ai.get("application", ""):
                        best = ai
                        break
                if best.get("route") and best["route"] != "unknown":
                    intv_info["route"] = best["route"]
                if best.get("dose_mg") is not None:
                    intv_info["dose_mg"] = best["dose_mg"]

        # Get original unit and convert
        original_unit = points[0]["original_unit"]
        timepoints = []
        for pt in points:
            mean_mg_L = convert_to_mg_per_L(pt["mean_raw"], pt["original_unit"], drug)
            if mean_mg_L is None:
                continue

            sd_mg_L = None
            if pt["sd_raw"] is not None:
                sd_mg_L = convert_to_mg_per_L(pt["sd_raw"], pt["original_unit"], drug)

            tp = {"time_h": round(pt["time_h"], 4), "mean": round(mean_mg_L, 6)}
            if sd_mg_L is not None:
                tp["sd"] = round(sd_mg_L, 6)
            timepoints.append(tp)

        if len(timepoints) < 3:
            continue

        # N subjects: take max from points (most consistent)
        n_subjects = None
        ns = [p["n_subjects"] for p in points if p["n_subjects"]]
        if ns:
            n_subjects = max(ns)

        record = {
            "study": study,
            "pmid": pmid,
            "route": intv_info["route"],
            "dose_mg": intv_info["dose_mg"],
            "n_subjects": n_subjects,
            "unit": "mg/L",
            "original_unit": original_unit,
            "intervention_raw": intervention_str,
            "timepoints": timepoints,
        }
        timecourses.append(record)

    return timecourses


def main():
    global VERBOSE

    parser = argparse.ArgumentParser(description="Fetch clinical C(t) data from PK-DB")
    parser.add_argument("--no-cache", action="store_true", help="Bypass cache")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    VERBOSE = args.verbose
    use_cache = not args.no_cache

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PK-DB Clinical Timecourse Extraction")
    print("=" * 60)

    # Step 1: Get all studies with timecourses
    print("\n[1/4] Fetching studies with timecourse data...")
    studies = get_all_studies_with_timecourses(use_cache=use_cache)
    print(f"  Found {len(studies)} studies with timecourse data")

    # Step 2: Check which studies contain our target drugs (by substance list)
    print("\n[2/4] Filtering studies for target drugs...")
    relevant_studies = []
    for study in studies:
        substances = study.get("substances", [])
        substance_names = []
        for s in substances:
            if isinstance(s, str):
                substance_names.append(s.lower())
            elif isinstance(s, dict):
                name = s.get("name", s.get("substance", ""))
                substance_names.append(name.lower())

        # Check if any substance matches our targets
        has_match = False
        for sname in substance_names:
            if match_drug(sname) is not None:
                has_match = True
                break

        if has_match:
            relevant_studies.append(study)

    print(f"  {len(relevant_studies)} studies contain target drugs")

    # Step 3: Download TSV files and extract data
    print("\n[3/4] Downloading and parsing TSV files...")
    all_timecourses = defaultdict(list)
    studies_processed = 0
    studies_with_data = 0

    for i, study in enumerate(relevant_studies):
        sid = study.get("sid", "")
        name = study.get("name", "unknown")
        ref = study.get("reference", {})
        pmid = None
        if isinstance(ref, dict):
            pmid = ref.get("pmid")
            if pmid:
                pmid = str(pmid)

        print(
            f"  [{i + 1}/{len(relevant_studies)}] {name} (sid={sid}, tc={study.get('timecourse_count', 0)})"
        )
        studies_processed += 1

        # Get TSV file URLs from study detail
        tsv_urls = get_study_files(sid, use_cache=use_cache)
        if not tsv_urls:
            log(f"  No TSV files for {name}")
            continue

        log(f"  Found {len(tsv_urls)} TSV files")

        # Get interventions
        study_interventions = get_study_interventions(name, use_cache=use_cache)

        # Download and parse each TSV
        study_points = []
        for tsv_url in tsv_urls:
            content = fetch_url(tsv_url, use_cache=use_cache, is_tsv=True)
            if content is None:
                continue

            points = parse_tsv(content, name)
            study_points.extend(points)

        if not study_points:
            continue

        studies_with_data += 1

        # Build timecourse records
        timecourses = build_timecourses(study_points, study_interventions, pmid)
        for tc in timecourses:
            drug = None
            # Find which drug this timecourse is for
            for pt in study_points:
                if pt["study"] == tc["study"] and pt["intervention"] == tc["intervention_raw"]:
                    drug = pt["drug"]
                    break
            if drug is None:
                # Fallback: parse from first matching point
                for pt in study_points:
                    if pt["study"] == tc["study"]:
                        drug = pt["drug"]
                        break
            if drug:
                # Clean up: remove intervention_raw from output
                tc_clean = {k: v for k, v in tc.items() if k != "intervention_raw"}
                all_timecourses[drug].append(tc_clean)

    # Step 4: Save results
    print(f"\n[4/4] Saving results to {OUTPUT_FILE}...")

    # Sort timecourses: prefer oral, single-dose, more subjects, more timepoints
    for drug in all_timecourses:
        all_timecourses[drug].sort(
            key=lambda tc: (
                0 if tc.get("route") == "oral" else 1,
                -(tc.get("n_subjects") or 0),
                -len(tc.get("timepoints", [])),
            )
        )

    # Convert to regular dict for JSON
    output = dict(all_timecourses)

    OUTPUT_FILE.write_text(json.dumps(output, indent=2))

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Studies processed:    {studies_processed}")
    print(f"Studies with data:    {studies_with_data}")
    print(f"Drugs found:          {len(output)}")
    print(f"Total timecourses:    {sum(len(v) for v in output.values())}")
    print()

    print(f"{'Drug':<20} {'Timecourses':>12} {'Best Route':<10} {'Best N':>8} {'Timepoints':>12}")
    print("-" * 65)
    for drug in sorted(output.keys()):
        tcs = output[drug]
        best = tcs[0] if tcs else {}
        print(
            f"{drug:<20} {len(tcs):>12} "
            f"{best.get('route', '?'):<10} "
            f"{best.get('n_subjects', '?'):>8} "
            f"{len(best.get('timepoints', [])):>12}"
        )

    # List drugs NOT found
    missing = sorted(set(TARGET_DRUGS.keys()) - set(output.keys()))
    if missing:
        print(f"\nDrugs NOT found in PK-DB: {', '.join(missing)}")

    print(f"\nOutput saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
