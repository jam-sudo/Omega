#!/usr/bin/env python3
"""Expand Cmax reference dataset by querying PK-DB for additional oral, single-dose,
healthy-adult PK studies.

Strategy:
  1. Query PK-DB studies endpoint for each candidate drug
  2. For studies with output_count > 0, download TSV files (Tab* files contain
     reported PK params like Cmax, while Fig* files contain timecourses)
  3. Parse TSV rows with measurement_type='cmax' and tissue='plasma'/'serum'/'blood'
  4. Filter for oral route, single-dose, and normalise units to mg/L
  5. Look up SMILES from PubChem and validate with RDKit
  6. Save to data/ml/clinical/pkdb_expanded_cmax.csv

Note: PK-DB's /outputs/ endpoint returns 0 records (auth-gated), so we use the
study->TSV file approach which is publicly accessible.

Usage:
    python scripts/expand_pkdb_cmax.py                # full run
    python scripts/expand_pkdb_cmax.py --dry-run      # no network calls, mock data
    python scripts/expand_pkdb_cmax.py --max-drugs 10 # limit to 10 new drugs
"""

import argparse
import csv
import io
import math
import re
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests", file=sys.stderr)
    sys.exit(1)

try:
    from rdkit import Chem
except ImportError:
    Chem = None
    print("WARNING: RDKit not available, SMILES validation will be skipped", file=sys.stderr)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_PKDB = "https://pk-db.com/api/v1"
PUBCHEM_REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name"
RATE_LIMIT = 1.2  # seconds between API requests

TRAINING_CSV = Path("data/ml/clinical/cmax_training_set.csv")
OUTPUT_CSV = Path("data/ml/clinical/pkdb_expanded_cmax.csv")

# Approximate molecular weights for molar->mass unit conversion (g/mol)
MW_TABLE = {
    "simvastatin": 418.57,
    "amlodipine": 408.88,
    "alprazolam": 308.76,
    "lorazepam": 321.16,
    "clonazepam": 315.72,
    "erythromycin": 733.93,
    "levofloxacin": 361.37,
    "ketoconazole": 531.43,
    "lamotrigine": 256.09,
    "valproic acid": 144.21,
    "haloperidol": 375.86,
    "risperidone": 410.49,
    "olanzapine": 312.43,
    "ranitidine": 314.41,
    "famotidine": 337.45,
    "dexamethasone": 392.46,
    "prednisone": 358.43,
    "prednisolone": 360.44,
    "metoclopramide": 299.80,
    "ondansetron": 293.37,
    "sildenafil": 474.58,
    "tadalafil": 389.40,
    "montelukast": 586.18,
    "cetirizine": 388.89,
    "fexofenadine": 501.66,
    "loratadine": 382.88,
    "bupropion": 239.74,
    "paroxetine": 329.37,
    "citalopram": 324.39,
    "escitalopram": 324.39,
    "duloxetine": 297.42,
    "venlafaxine": 277.40,
    "mirtazapine": 265.35,
    "aripiprazole": 448.39,
    "quetiapine": 383.51,
    "buspirone": 385.50,
    "modafinil": 273.35,
    "methylphenidate": 233.31,
    "codeine": 299.36,
    "naproxen": 230.26,
    "celecoxib": 381.37,
    "meloxicam": 351.40,
    "piroxicam": 331.35,
    "indomethacin": 357.79,
    "methotrexate": 454.44,
    "tacrolimus": 804.02,
    "cyclosporine": 1202.61,
    "rosuvastatin": 481.54,
    "pravastatin": 424.53,
    "fluvastatin": 411.47,
    "gemfibrozil": 250.33,
    "fenofibrate": 360.83,
    "acyclovir": 225.20,
    "valacyclovir": 324.34,
    "ribavirin": 244.20,
    "rifampicin": 822.94,
    "isoniazid": 137.14,
    "ethambutol": 204.31,
    "pyrazinamide": 123.11,
    "dapsone": 248.30,
    "chloroquine": 319.87,
    "mefloquine": 378.31,
    "morphine": 285.34,
    "oxycodone": 315.36,
    "sumatriptan": 295.40,
    "amitriptyline": 277.40,
    "nortriptyline": 263.38,
    "imipramine": 280.41,
    "pregabalin": 159.23,
    "topiramate": 339.36,
    "levetiracetam": 170.21,
    "oxcarbazepine": 252.27,
    "phenobarbital": 232.24,
}

# Candidate drugs to search PK-DB for (common oral drugs with known PK).
# These are drugs likely in PK-DB that are NOT in the current 66-drug training set.
CANDIDATE_DRUGS = [
    "codeine",
    "morphine",
    "simvastatin",
    "naproxen",
    "ranitidine",
    "dexamethasone",
    "sildenafil",
    "bupropion",
    "venlafaxine",
    "paroxetine",
    "citalopram",
    "risperidone",
    "olanzapine",
    "quetiapine",
    "aripiprazole",
    "haloperidol",
    "lamotrigine",
    "valproic acid",
    "phenobarbital",
    "levetiracetam",
    "pregabalin",
    "topiramate",
    "oxcarbazepine",
    "amlodipine",
    "alprazolam",
    "lorazepam",
    "clonazepam",
    "buspirone",
    "modafinil",
    "methylphenidate",
    "erythromycin",
    "levofloxacin",
    "ketoconazole",
    "rifampicin",
    "isoniazid",
    "ethambutol",
    "pyrazinamide",
    "dapsone",
    "chloroquine",
    "mefloquine",
    "acyclovir",
    "valacyclovir",
    "ribavirin",
    "celecoxib",
    "meloxicam",
    "piroxicam",
    "indomethacin",
    "rosuvastatin",
    "pravastatin",
    "fluvastatin",
    "gemfibrozil",
    "fenofibrate",
    "tacrolimus",
    "cyclosporine",
    "methotrexate",
    "montelukast",
    "cetirizine",
    "fexofenadine",
    "loratadine",
    "famotidine",
    "ondansetron",
    "tadalafil",
    "escitalopram",
    "duloxetine",
    "mirtazapine",
    "oxycodone",
    "sumatriptan",
    "amitriptyline",
    "nortriptyline",
    "imipramine",
    "prednisolone",
    "prednisone",
    "metoclopramide",
]

# Drug name aliases: PK-DB may use different names
DRUG_ALIASES = {
    "acetaminophen": ["paracetamol", "acetaminophen"],
    "codeine": ["codeine"],
    "morphine": ["morphine"],
    "simvastatin": ["simvastatin"],
    "naproxen": ["naproxen"],
    "valproic acid": ["valproic acid", "valproate"],
    "rifampicin": ["rifampicin", "rifampin"],
    "methylphenidate": ["methylphenidate"],
    "dexamethasone": ["dexamethasone"],
    "prednisolone": ["prednisolone"],
    "prednisone": ["prednisone"],
}

# ---------------------------------------------------------------------------
# PK-DB timecourse-based Cmax extraction
# ---------------------------------------------------------------------------

TIMECOURSE_PATH = Path("data/clinical/pkdb_timecourses.json")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_last_req_time = 0.0


def _rate_limited_get(
    url: str, params: dict | None = None, timeout: int = 30, headers: dict | None = None
) -> requests.Response:
    """GET with rate limiting, matching download_pkdb.py pattern."""
    global _last_req_time
    elapsed = time.monotonic() - _last_req_time
    if elapsed < RATE_LIMIT:
        time.sleep(RATE_LIMIT - elapsed)
    resp = requests.get(url, params=params, timeout=timeout, headers=headers)
    _last_req_time = time.monotonic()
    return resp


def load_existing_drugs(csv_path: Path) -> set[str]:
    """Load drug names already in the training set (case-insensitive)."""
    drugs = set()
    if not csv_path.exists():
        return drugs
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("drug", "").strip().lower()
            if name:
                drugs.add(name)
    return drugs


def normalize_unit_to_mg_per_L(value: float, unit: str, drug: str | None = None) -> float | None:
    """Convert concentration value to mg/L.

    Handles mass/volume and molar units (requires MW for molar).
    """
    if value is None or not math.isfinite(value):
        return None

    u = unit.lower().strip()

    # Mass/volume conversions
    if u in ("mg/l", "mg/liter"):
        return value
    if u in ("µg/ml", "ug/ml", "mcg/ml", "microgram/ml"):
        return value  # µg/mL == mg/L
    if u in ("ng/ml", "ng/milliliter"):
        return value * 0.001
    if u in ("µg/l", "ug/l", "mcg/l", "microgram/l", "microgram/liter"):
        return value * 0.001
    if u in ("ng/l", "ng/liter"):
        return value * 1e-6
    if u in ("pg/ml", "pg/milliliter"):
        return value * 1e-6

    # Molar conversions (need MW)
    mw = MW_TABLE.get(drug) if drug else None
    if mw:
        if u in ("µmol/l", "umol/l", "micromol/l"):
            return value * mw / 1000.0
        if u in ("nmol/l", "nmol/liter"):
            return value * mw / 1e6
        if u in ("pmol/l", "pmol/liter", "pmol/ml"):
            # pmol/ml = pmol/mL = nmol/L (1 mL = 0.001 L, so pmol/mL = 1000 pmol/L = 1 nmol/L)
            # Actually: pmol/mL = 1e-12 mol / 1e-3 L = 1e-9 mol/L = nmol/L
            # mg/L = nmol/L * MW / 1e6 ... wait:
            # pmol/mL = 1e-12 mol / 1e-3 L = 1e-9 mol/L
            # mg/L = mol/L * MW * 1000 = 1e-9 * MW * 1000 = MW * 1e-6
            return value * mw * 1e-6

    return None  # unknown unit


def fetch_smiles_pubchem(drug_name: str) -> str | None:
    """Look up canonical SMILES from PubChem REST API.

    PubChem returns different key names depending on the property requested:
    - CanonicalSMILES -> returned as 'ConnectivitySMILES'
    - IsomericSMILES -> returned as 'SMILES'
    We request both and prefer ConnectivitySMILES (canonical, no stereochemistry).
    """
    try:
        url = f"{PUBCHEM_REST}/{drug_name}/property/CanonicalSMILES,IsomericSMILES/JSON"
        resp = _rate_limited_get(url, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        props = data.get("PropertyTable", {}).get("Properties", [])
        if props:
            p = props[0]
            # PubChem returns ConnectivitySMILES for canonical, SMILES for isomeric
            return (
                p.get("CanonicalSMILES")
                or p.get("ConnectivitySMILES")
                or p.get("SMILES")
                or p.get("IsomericSMILES")
            )
    except Exception:
        return None
    return None


def validate_smiles(smiles: str) -> bool:
    """Validate SMILES with RDKit. Returns True if valid."""
    if Chem is None:
        return True
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


# ---------------------------------------------------------------------------
# PK-DB study + TSV fetching
# ---------------------------------------------------------------------------


def get_studies_for_drug(drug_name: str) -> list[dict]:
    """Query PK-DB studies endpoint for a drug. Returns list of study dicts."""
    try:
        resp = _rate_limited_get(
            f"{BASE_PKDB}/studies/",
            params={"substance": drug_name, "page_size": 200},
            timeout=30,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        studies = data.get("data", {}).get("data", [])
        return studies
    except requests.RequestException:
        return []


def get_study_tsv_urls(sid: str) -> list[str]:
    """Get TSV file URLs from study detail endpoint."""
    try:
        resp = _rate_limited_get(f"{BASE_PKDB}/_studies/{sid}/", timeout=30)
        if resp.status_code != 200:
            return []
        data = resp.json()
        study_data = data.get("data", data)
        if isinstance(study_data, dict) and "data" in study_data:
            study_data = study_data["data"]
        files = study_data.get("files", [])
        tsv_urls = []
        for f in files:
            if isinstance(f, str):
                url = f
            elif isinstance(f, dict):
                url = f.get("file", f.get("url", ""))
            else:
                continue
            if url.endswith(".tsv"):
                if not url.startswith("http"):
                    url = f"https://pk-db.com{url}"
                tsv_urls.append(url)
        return tsv_urls
    except requests.RequestException:
        return []


def download_tsv(url: str) -> str | None:
    """Download a TSV file. Returns content as string or None."""
    try:
        resp = _rate_limited_get(url, timeout=15, headers={"Accept": "text/plain"})
        if resp.status_code == 200:
            return resp.text
    except requests.RequestException:
        pass
    return None


def get_interventions_for_study(study_name: str, study_sid: str = "") -> dict:
    """Fetch interventions API and filter for the given study.

    PK-DB's interventions endpoint does not filter by study_name reliably,
    so we paginate and filter client-side by study.name.

    Returns {substance_name: [{route, dose_mg, application}]}.
    """
    try:
        # Paginate (the endpoint may return unrelated studies too)
        all_records = []
        page = 1
        while page <= 5:  # limit pages to avoid excessive fetching
            resp = _rate_limited_get(
                f"{BASE_PKDB}/interventions/",
                params={"page_size": 200, "page": page},
                timeout=30,
            )
            if resp.status_code != 200:
                break
            data = resp.json()
            results_data = data.get("data", data)
            if isinstance(results_data, dict):
                records = results_data.get("data", results_data.get("results", []))
            else:
                records = results_data if isinstance(results_data, list) else []
            if not records:
                break
            all_records.extend(records)
            last_page = data.get("last_page", 1)
            if page >= last_page:
                break
            page += 1

        interventions = {}
        for rec in all_records:
            if not isinstance(rec, dict):
                continue

            # Filter by study name
            rec_study = rec.get("study", {})
            if isinstance(rec_study, dict):
                rec_study_name = rec_study.get("name", "")
                rec_study_sid = rec_study.get("sid", "")
            else:
                rec_study_name = str(rec_study)
                rec_study_sid = ""

            if rec_study_name != study_name and rec_study_sid != study_sid:
                continue

            substance = rec.get("substance", "")
            if isinstance(substance, dict):
                substance = substance.get("name", "")
            substance = str(substance).lower().strip()

            route = rec.get("route", "")
            if isinstance(route, dict):
                route = route.get("name", "")
            route = str(route).lower().strip()

            application = rec.get("application", "")
            if isinstance(application, dict):
                application = application.get("name", "")
            application = str(application).lower().strip()

            # Skip normalised entries (duplicate in different units)
            if rec.get("normed", False):
                continue

            dose_mg = None
            value = rec.get("value")
            unit = rec.get("unit", "")
            if isinstance(unit, dict):
                unit = unit.get("name", "")
            unit = str(unit).lower().strip()

            if value is not None:
                try:
                    dose_mg = float(value)
                    if unit in ("g", "gram"):
                        dose_mg *= 1000
                    elif unit in ("µg", "ug", "microgram"):
                        dose_mg /= 1000
                except (ValueError, TypeError):
                    dose_mg = None

            if substance not in interventions:
                interventions[substance] = []
            interventions[substance].append(
                {"route": route, "dose_mg": dose_mg, "application": application}
            )

        return interventions
    except requests.RequestException:
        return {}


# Common standard doses for PK studies (mg, oral) — used as fallback
STANDARD_DOSES = {
    "codeine": 60.0,
    "morphine": 30.0,
    "simvastatin": 40.0,
    "naproxen": 500.0,
    "ranitidine": 150.0,
    "dexamethasone": 8.0,
    "sildenafil": 50.0,
    "bupropion": 150.0,
    "venlafaxine": 75.0,
    "paroxetine": 20.0,
    "citalopram": 20.0,
    "risperidone": 2.0,
    "olanzapine": 10.0,
    "quetiapine": 200.0,
    "aripiprazole": 15.0,
    "haloperidol": 5.0,
    "lamotrigine": 100.0,
    "valproic acid": 500.0,
    "phenobarbital": 100.0,
    "levetiracetam": 500.0,
    "pregabalin": 75.0,
    "topiramate": 100.0,
    "oxcarbazepine": 300.0,
    "amlodipine": 5.0,
    "alprazolam": 1.0,
    "lorazepam": 2.0,
    "clonazepam": 1.0,
    "buspirone": 10.0,
    "modafinil": 200.0,
    "methylphenidate": 20.0,
    "erythromycin": 500.0,
    "levofloxacin": 500.0,
    "ketoconazole": 200.0,
    "rifampicin": 600.0,
    "isoniazid": 300.0,
    "ethambutol": 800.0,
    "pyrazinamide": 1000.0,
    "dapsone": 100.0,
    "chloroquine": 600.0,
    "mefloquine": 250.0,
    "acyclovir": 200.0,
    "valacyclovir": 1000.0,
    "ribavirin": 600.0,
    "celecoxib": 200.0,
    "meloxicam": 15.0,
    "piroxicam": 20.0,
    "indomethacin": 50.0,
    "rosuvastatin": 20.0,
    "pravastatin": 40.0,
    "fluvastatin": 40.0,
    "gemfibrozil": 600.0,
    "fenofibrate": 200.0,
    "tacrolimus": 5.0,
    "cyclosporine": 200.0,
    "methotrexate": 15.0,
    "montelukast": 10.0,
    "cetirizine": 10.0,
    "fexofenadine": 60.0,
    "loratadine": 10.0,
    "famotidine": 20.0,
    "ondansetron": 8.0,
    "tadalafil": 10.0,
    "escitalopram": 10.0,
    "duloxetine": 60.0,
    "mirtazapine": 15.0,
    "oxycodone": 10.0,
    "sumatriptan": 100.0,
    "amitriptyline": 50.0,
    "nortriptyline": 25.0,
    "imipramine": 50.0,
    "prednisolone": 20.0,
    "prednisone": 20.0,
    "metoclopramide": 10.0,
}


def parse_cmax_from_tsv(content: str, drug_name: str, study_name: str) -> list[dict]:
    """Parse TSV for Cmax rows matching the drug.

    Looks for measurement_type='cmax' (reported PK param) and also
    extracts max concentration from timecourse data as fallback.
    """
    results = []
    drug_lower = drug_name.lower()
    aliases = DRUG_ALIASES.get(drug_lower, [drug_lower])

    try:
        reader = csv.DictReader(io.StringIO(content), delimiter="\t")
    except Exception:
        return results

    for row in reader:
        try:
            measurement_type = (row.get("measurement_type", "") or "").strip().lower()
            substance = (row.get("substance", "") or "").strip().lower()
            tissue = (row.get("tissue", "") or "").strip().lower()
            intervention = (row.get("intervention", "") or "").strip().lower()

            # Match substance to our target drug
            if substance not in aliases and drug_lower not in substance:
                continue

            # Skip metabolites (contains "glucuronide", "nor-", "hydroxy-" etc.)
            if any(
                met in substance
                for met in (
                    "glucuronide",
                    "sulfate",
                    "oxide",
                    "nor",
                    "hydroxy",
                    "desmethyl",
                    "metabolite",
                )
            ):
                # But allow the parent drug if it contains these as part of name
                if substance not in aliases:
                    continue

            # Filter for Cmax specifically (reported PK param)
            if measurement_type != "cmax":
                continue

            # Tissue: plasma, blood, serum
            if tissue and tissue not in ("plasma", "blood", "serum"):
                continue

            # Get value
            mean_str = (row.get("mean") or row.get("value") or "").strip()
            if not mean_str or mean_str.lower() in ("na", "nan", "none", ""):
                continue

            value = float(mean_str)
            if value <= 0 or not math.isfinite(value):
                continue

            unit = (row.get("unit") or "").strip()

            # Normalise to mg/L
            cmax_mg_L = normalize_unit_to_mg_per_L(value, unit, drug_lower)
            if cmax_mg_L is None:
                continue

            # Check for IV routes in intervention string (skip them)
            if any(kw in intervention for kw in ("iv", "intravenous")):
                continue

            results.append(
                {
                    "drug": drug_name,
                    "cmax_mg_L": cmax_mg_L,
                    "value_original": value,
                    "unit_original": unit,
                    "study": study_name,
                    "tissue": tissue or "plasma",
                    "intervention": intervention,
                }
            )

        except (ValueError, TypeError, KeyError):
            continue

    return results


def extract_cmax_for_drug(drug_name: str, max_studies: int = 10) -> list[dict]:
    """Full pipeline: studies -> TSV files -> Cmax extraction for one drug."""
    all_cmax = []

    # Also try aliases
    names_to_try = DRUG_ALIASES.get(drug_name.lower(), [drug_name])

    studies_checked = 0
    for query_name in names_to_try:
        studies = get_studies_for_drug(query_name)
        if not studies:
            continue

        # Focus on studies with output_count > 0 (reported PK params)
        studies_with_outputs = [s for s in studies if s.get("output_count", 0) > 0]
        if not studies_with_outputs:
            # Fall back to studies with timecourses
            studies_with_outputs = [s for s in studies if s.get("timecourse_count", 0) > 0]

        for study in studies_with_outputs[:max_studies]:
            sid = study.get("sid", "")
            name = study.get("name", "unknown")
            studies_checked += 1

            # Get TSV files
            tsv_urls = get_study_tsv_urls(sid)
            if not tsv_urls:
                continue

            # Prioritise Tab* files (contain reported PK params like Cmax)
            tab_urls = [u for u in tsv_urls if "_Tab" in u]
            fig_urls = [u for u in tsv_urls if "_Fig" in u]
            ordered_urls = tab_urls + fig_urls  # Tab files first

            for url in ordered_urls:
                content = download_tsv(url)
                if content is None:
                    continue

                cmax_records = parse_cmax_from_tsv(content, drug_name, name)
                all_cmax.extend(cmax_records)

            # Early termination: if we found enough records, stop querying more studies
            if len(all_cmax) >= 5:
                break

        if len(all_cmax) >= 5:
            break

    # Try to extract dose from intervention strings in TSV records
    dose_pattern = re.compile(r"(\d+\.?\d*)\s*mg", re.IGNORECASE)
    for rec in all_cmax:
        if rec.get("dose_mg") is not None:
            continue
        intv = rec.get("intervention", "")
        m = dose_pattern.search(intv)
        if m:
            rec["dose_mg"] = float(m.group(1))
            rec["route"] = "oral"  # assume oral if dose is in intervention name

    # Fallback: use standard dose from lookup table
    std_dose = STANDARD_DOSES.get(drug_name.lower())
    for rec in all_cmax:
        if rec.get("dose_mg") is None and std_dose is not None:
            rec["dose_mg"] = std_dose
            rec["dose_source"] = "standard_dose_table"

    return all_cmax


def select_best_cmax(records: list[dict]) -> dict | None:
    """Select the best Cmax record: prefer plasma, with dose, oral route."""
    if not records:
        return None

    def score(r):
        s = 0
        if r.get("dose_mg") is not None and r.get("dose_mg", 0) > 0:
            s += 100
        if r.get("tissue") == "plasma":
            s += 10
        elif r.get("tissue") == "serum":
            s += 5
        route = str(r.get("route", "")).lower()
        intv = str(r.get("intervention", "")).lower()
        if "oral" in route or "po" in route or "oral" in intv:
            s += 50
        if any(kw in intv for kw in ("iv", "intravenous")):
            s -= 200
        return s

    return max(records, key=score)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def cmax_from_timecourse(studies: list[dict], drug_name: str) -> dict | None:
    """Extract best Cmax from PK-DB timecourse records for a drug.

    Filters for oral route, reasonable dose, and selects the study with the
    most plausible Cmax (median across studies to avoid outliers).
    """
    oral_cmax_records = []
    std_dose = STANDARD_DOSES.get(drug_name.lower())

    for s in studies:
        route = (s.get("route") or "").lower()
        # Accept oral and unknown (many PK-DB studies don't specify route)
        if route not in ("oral", "unknown", "po", ""):
            continue

        tps = s.get("timepoints", [])
        if not tps:
            continue

        concs = [
            p.get("mean", 0) for p in tps if p.get("mean") is not None and p.get("mean", 0) > 0
        ]
        if not concs:
            continue

        cmax = max(concs)
        dose = s.get("dose_mg")
        unit = s.get("unit", "mg/L")

        # Sanity check: skip obviously wrong values
        # Cmax/dose ratio should be < 1.0 mg/L per mg (most drugs)
        if dose and dose > 0 and cmax / dose > 1.0:
            continue  # likely wrong route or unit

        oral_cmax_records.append(
            {
                "cmax_mg_L": cmax,
                "dose_mg": dose or std_dose,
                "study": s.get("study", "unknown"),
                "pmid": s.get("pmid", ""),
                "unit": unit,
                "n_subjects": s.get("n_subjects"),
            }
        )

    if not oral_cmax_records:
        return None

    # If multiple studies, take the median Cmax (more robust than max)
    cmax_values = [r["cmax_mg_L"] for r in oral_cmax_records]
    if len(cmax_values) >= 3:
        import statistics

        median_cmax = statistics.median(cmax_values)
        # Pick the record closest to the median
        best = min(oral_cmax_records, key=lambda r: abs(r["cmax_mg_L"] - median_cmax))
    else:
        # With few studies, prefer one with known dose
        with_dose = [r for r in oral_cmax_records if r.get("dose_mg")]
        best = with_dose[0] if with_dose else oral_cmax_records[0]

    return best


def run_from_timecourses() -> list[dict]:
    """Extract Cmax from local PK-DB timecourse data (no API calls needed)."""
    import json

    if not TIMECOURSE_PATH.exists():
        print(f"  ERROR: {TIMECOURSE_PATH} not found")
        return []

    with open(TIMECOURSE_PATH) as f:
        timecourse_data = json.load(f)

    print(f"  PK-DB timecourse drugs: {len(timecourse_data)}")

    results = []
    for drug_name, studies in sorted(timecourse_data.items()):
        best = cmax_from_timecourse(studies, drug_name)
        if best is None:
            print(f"  SKIP {drug_name}: no valid oral Cmax in timecourses")
            continue

        cmax_mg_L = best["cmax_mg_L"]
        dose_mg = best.get("dose_mg")
        study = best.get("study", "unknown")
        pmid = best.get("pmid", "")

        # Look up SMILES
        smiles = fetch_smiles_pubchem(drug_name)
        if smiles is None:
            print(f"  SKIP {drug_name}: no PubChem SMILES")
            continue
        if not validate_smiles(smiles):
            print(f"  SKIP {drug_name}: invalid SMILES")
            continue

        log_obs = (
            math.log(cmax_mg_L / dose_mg) if dose_mg and dose_mg > 0 and cmax_mg_L > 0 else None
        )

        results.append(
            {
                "drug": drug_name,
                "smiles": smiles,
                "dose_mg": dose_mg,
                "obs_cmax": cmax_mg_L,
                "pred_cmax_pbpk": "",
                "log_obs_cmax_per_mg": log_obs if log_obs is not None else "",
                "tier": "pkdb_timecourse",
                "source_study": f"{study} (PMID:{pmid})" if pmid else study,
            }
        )
        print(f"  ADDED {drug_name}: Cmax={cmax_mg_L:.4f} mg/L, dose={dose_mg} mg, study={study}")

    return results


def run_real(max_drugs: int | None = None) -> list[dict]:
    """Run real PK-DB + PubChem queries."""
    existing = load_existing_drugs(TRAINING_CSV)
    print(f"Existing drugs in training set: {len(existing)}")

    # Filter candidates
    candidates = [d for d in CANDIDATE_DRUGS if d.lower() not in existing]
    if max_drugs is not None:
        candidates = candidates[:max_drugs]

    print(f"Candidate drugs to query: {len(candidates)}")

    results = []
    failed_drugs = []
    skipped_drugs = []

    for i, drug in enumerate(candidates):
        print(f"\n[{i + 1}/{len(candidates)}] Querying PK-DB for: {drug}")

        # Extract Cmax from study TSV files
        cmax_records = extract_cmax_for_drug(drug, max_studies=5)
        if not cmax_records:
            print("  No Cmax records found")
            skipped_drugs.append((drug, "no Cmax in PK-DB TSV files"))
            continue

        print(f"  Found {len(cmax_records)} Cmax records")

        # Select best
        best = select_best_cmax(cmax_records)
        if best is None:
            skipped_drugs.append((drug, "no valid record"))
            continue

        cmax_mg_L = best["cmax_mg_L"]
        dose_mg = best.get("dose_mg")

        print(
            f"  Best: Cmax={cmax_mg_L:.6f} mg/L"
            f" ({best.get('value_original'):.4g} {best.get('unit_original', '?')})"
            f", dose={dose_mg} mg"
            f", study={best.get('study', '?')}"
        )

        # Look up SMILES from PubChem
        print("  Looking up SMILES from PubChem...")
        smiles = fetch_smiles_pubchem(drug)
        if smiles is None:
            print(f"  WARNING: No SMILES found on PubChem for {drug}")
            failed_drugs.append((drug, "no PubChem SMILES"))
            continue

        if not validate_smiles(smiles):
            print(f"  WARNING: Invalid SMILES from PubChem for {drug}: {smiles}")
            failed_drugs.append((drug, "invalid SMILES"))
            continue

        print(f"  SMILES: {smiles[:60]}{'...' if len(smiles) > 60 else ''}")

        # Compute log_obs_cmax_per_mg
        log_obs = None
        if dose_mg and dose_mg > 0 and cmax_mg_L > 0:
            log_obs = math.log(cmax_mg_L / dose_mg)

        results.append(
            {
                "drug": drug,
                "smiles": smiles,
                "dose_mg": dose_mg,
                "obs_cmax": cmax_mg_L,
                "pred_cmax_pbpk": "",
                "log_obs_cmax_per_mg": log_obs if log_obs is not None else "",
                "tier": "pkdb_expanded",
                "source_study": best.get("study", ""),
            }
        )
        print("  -> ADDED to expansion set")

    # Summary
    print("\n" + "=" * 60)
    print("EXPANSION SUMMARY")
    print("=" * 60)
    print(f"Candidates queried:  {len(candidates)}")
    print(f"New drugs added:     {len(results)}")
    print(f"Skipped (no data):   {len(skipped_drugs)}")
    print(f"Failed (SMILES):     {len(failed_drugs)}")

    if results:
        print("\nNew drugs:")
        for r in results:
            print(
                f"  {r['drug']}: Cmax={r['obs_cmax']:.6f} mg/L"
                f", dose={r['dose_mg']} mg"
                f", study={r['source_study']}"
            )

    if skipped_drugs:
        print("\nSkipped drugs:")
        for drug, reason in skipped_drugs:
            print(f"  {drug}: {reason}")

    if failed_drugs:
        print("\nFailed drugs:")
        for drug, reason in failed_drugs:
            print(f"  {drug}: {reason}")

    return results


def save_results(results: list[dict], output_path: Path) -> None:
    """Save expansion results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "drug",
        "smiles",
        "dose_mg",
        "obs_cmax",
        "pred_cmax_pbpk",
        "log_obs_cmax_per_mg",
        "tier",
        "source_study",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\nSaved {len(results)} records to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Expand Cmax reference dataset from PK-DB")
    parser.add_argument(
        "--timecourses-only",
        action="store_true",
        help="Extract Cmax from local timecourse data only (no API calls)",
    )
    parser.add_argument(
        "--max-drugs",
        type=int,
        default=None,
        help="Maximum number of new drugs to query via API",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (overrides default)",
    )
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else OUTPUT_CSV

    print("=" * 60)
    print("PK-DB Cmax Expansion Pipeline")
    print("=" * 60)

    # Step 1: Extract from local timecourses (always)
    print("\n--- Step 1: Local timecourse extraction ---")
    tc_results = run_from_timecourses()
    print(f"  Timecourse results: {len(tc_results)} drugs")

    # Step 2: Query API for additional drugs (unless timecourses-only)
    api_results = []
    if not args.timecourses_only:
        print("\n--- Step 2: PK-DB API queries ---")
        # Exclude drugs already found in timecourses
        tc_drugs = {r["drug"].lower() for r in tc_results}
        api_results = run_real(max_drugs=args.max_drugs)
        # Deduplicate
        api_results = [r for r in api_results if r["drug"].lower() not in tc_drugs]
        print(f"  API results (new): {len(api_results)} drugs")

    # Combine
    all_results = tc_results + api_results
    print(f"\n  Total: {len(all_results)} drugs")

    if all_results:
        save_results(all_results, output_path)
    else:
        print("\nNo new drugs found.")
        save_results([], output_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
