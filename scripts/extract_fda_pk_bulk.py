#!/usr/bin/env python3
"""Bulk-extract PK parameters from 200+ FDA drug labels via OpenFDA API.

Queries the OpenFDA drug label API for oral drugs with clinical_pharmacology
sections, extracts PK parameters (Cmax, AUC, t_half, bioavailability, protein
binding, clearance, Vd) using regex, fetches SMILES from PubChem, and saves
a comprehensive reference database.

Output: data/clinical/fda_pk_reference.json

Usage:
    python scripts/extract_fda_pk_bulk.py [--max-pages 50] [--no-pubchem]
"""

import argparse
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OPENFDA_BASE = "https://api.fda.gov/drug/label.json"
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name"
CACHE_DIR = Path("data/ml/fda_cache")
OUTPUT_PATH = Path("data/clinical/fda_pk_reference.json")
OPENFDA_DELAY = 0.35  # seconds between OpenFDA requests
PUBCHEM_DELAY = 0.3  # seconds between PubChem requests
PAGE_SIZE = 100
MAX_SKIP = 5000  # OpenFDA hard limit

# ---------------------------------------------------------------------------
# Salt / formulation suffixes to strip from drug names
# ---------------------------------------------------------------------------
SALT_SUFFIXES = [
    " hydrochloride",
    " hcl",
    " sulfate",
    " sodium",
    " potassium",
    " besylate",
    " calcium",
    " tartrate",
    " succinate",
    " mesylate",
    " phosphate",
    " acetate",
    " fumarate",
    " maleate",
    " citrate",
    " saccharate",
    " aspartate",
    " monohydrate",
    " dihydrate",
    " trihydrate",
    " hydrobromide",
    " nitrate",
    " chloride",
    " bromide",
    " gluconate",
    " lactate",
    " oxalate",
    " benzoate",
    " stearate",
    " valerate",
    " propionate",
    " butyrate",
    " pamoate",
    " tosylate",
    " ethanolamine",
    " tromethamine",
    " dimesylate",
    " dihydrochloride",
    " disodium",
    " dipotassium",
    " hemifumarate",
    " xinafoate",
    " napsylate",
    " malonate",
    " edisylate",
    " esylate",
    " embonate",
    " ophthalmic solution",
    " extended-release capsules",
    " extended-release tablets",
    " delayed-release capsules",
    " er tablets",
    " tablets",
    " capsules",
    " film coated",
    " anhydrous",
    " oral suspension",
    " oral solution",
    " chewable tablets",
    " orally disintegrating tablets",
    " injection",
    " for injection",
]


def normalize_drug_name(name: str) -> str:
    """Lowercase, strip salts/formulations, take first drug from combos."""
    name = name.lower().strip()
    for suffix in SALT_SUFFIXES:
        name = name.replace(suffix, "")
    # Combo drugs: keep first component
    if " and " in name:
        name = name.split(" and ")[0]
    if "/" in name:
        name = name.split("/")[0]
    if ", " in name:
        name = name.split(", ")[0]
    return name.strip()


# ---------------------------------------------------------------------------
# PK extraction regex patterns (reused from existing script, expanded)
# ---------------------------------------------------------------------------
_CONC_UNITS = r"(ng/[mM][lL]|[uμµ]g/[mM][lL]|mcg/[mM][lL]|mg/[lL]|mg/dL)"
_AUC_UNITS = (
    r"((?:ng|µg|mcg|μg|ug|mg)[·•*×x\s]h(?:r|our)?s?/[mM][lL]|"
    r"(?:ng|µg|mcg|μg|ug|mg)/[mM][lL][·•*×x\s]h(?:r|our)?s?|"
    r"h(?:r|our)?s?[·•*×x\s](?:ng|µg|mcg|μg|ug|mg)/(?:mL|ml)|"
    r"(?:ng|µg|mcg|μg|ug|mg)\*h(?:r|our)?s?/(?:mL|ml)|"
    r"hr?\*(?:ng|µg|mcg|μg|ug|mg)/(?:mL|ml))"
)
_TIME_UNITS = r"(hours?|hrs?|h\b|days?|minutes?|min)"
_VERB = r"(?:was|of|is|approximately|about|~|=|:|averaged?|averaging|ranges?\s+from|occurring|estimated|reported|reached)"
_NUM = r"(\d{1,6}(?:\.\d+)?)"
_OPT_APPROX = r"(?:approximately\s+|about\s+|~\s*|around\s+)?"
_OPT_PLUSMINUS = r"(?:\s*±\s*\d+\.?\d*)?"

# Each entry: (param, compiled_regex, value_group_index, unit_group_index_or_None)
_RAW_PATTERNS = [
    # --- Cmax ---
    (
        "cmax",
        r"(?:C\s*max|peak\s+(?:plasma\s+)?concentration)[^\d]{0,60}?"
        r"(?:" + _VERB + r"\s+)" + _OPT_APPROX + _NUM + r"\s*" + _CONC_UNITS,
        1,
        2,
    ),
    ("cmax", r"C\s*max(?:\s*,?\s*ss)?\s*\(\s*" + _CONC_UNITS + r"\s*\)\s*" + _NUM, 2, 1),
    ("cmax", r"C\s*max\s*(?:of|was|is|=|:)\s*" + _OPT_APPROX + _NUM + r"\s*" + _CONC_UNITS, 1, 2),
    (
        "cmax",
        r"[Mm]ean\s+C\s*max\s+(?:value\s+)?(?:was|of|is)\s+"
        + _OPT_APPROX
        + _NUM
        + r"\s*"
        + _CONC_UNITS,
        1,
        2,
    ),
    # --- AUC ---
    (
        "auc",
        r"(?:AUC(?:\s*0?\s*[-–]\s*(?:∞|inf|infinity|last|tau|24\s*h?r?|12\s*h?r?|t))?|"
        r"area\s+under\s+the\s+(?:plasma\s+)?(?:concentration[- ]*time\s+)?curve)"
        r"[^\d]{0,80}?(?:"
        + _VERB
        + r"\s+)"
        + _OPT_APPROX
        + _NUM
        + _OPT_PLUSMINUS
        + r"\s*"
        + _AUC_UNITS,
        1,
        2,
    ),
    (
        "auc",
        r"AUC(?:\s*0?\s*[-–]\s*(?:∞|inf|infinity|last|tau))?\s*\(\s*"
        + _AUC_UNITS
        + r"\s*\)\s*"
        + _NUM,
        2,
        1,
    ),
    (
        "auc",
        r"AUC\s*(?:of|was|is|=|:)\s*" + _OPT_APPROX + _NUM + _OPT_PLUSMINUS + r"\s*" + _AUC_UNITS,
        1,
        2,
    ),
    # --- Half-life ---
    (
        "t_half",
        r"(?:half[\-\s]*life|t\s*1\s*/?\s*2|t½)[^\d]{0,80}?"
        r"(?:" + _VERB + r"\s+)" + _OPT_APPROX + _NUM + r"\s*(?:to\s+\d+\.?\d*\s*)?" + _TIME_UNITS,
        1,
        2,
    ),
    (
        "t_half",
        r"(?:elimination|terminal|plasma|mean|apparent|effective)\s+(?:half[\-\s]*life|t\s*1\s*/?\s*2|t½)\s+"
        r"(?:is\s+|was\s+)?" + _OPT_APPROX + _NUM + r"\s*" + _TIME_UNITS,
        1,
        2,
    ),
    (
        "t_half",
        r"(?:half[\-\s]*life|t\s*1\s*/?\s*2|t½)\s*(?:of|was|is|=|:)\s*"
        + _OPT_APPROX
        + _NUM
        + r"\s*"
        + _TIME_UNITS,
        1,
        2,
    ),
    (
        "t_half",
        r"(?:half[\-\s]*life|t\s*1\s*/?\s*2|t½)\s*\(\s*" + _TIME_UNITS + r"\s*\)\s*" + _NUM,
        2,
        1,
    ),
    # --- Bioavailability ---
    (
        "bioavailability",
        r"(?:(?:absolute|oral|mean|relative)\s+)?bioavailability[^\d]{0,60}?"
        r"(?:" + _VERB + r"\s+)" + _OPT_APPROX + _NUM + r"\s*%",
        1,
        None,
    ),
    (
        "bioavailability",
        r"(?:(?:absolute|oral)\s+)?bioavailability\s+(?:is\s+|was\s+)?"
        + _OPT_APPROX
        + _NUM
        + r"\s*%",
        1,
        None,
    ),
    # --- Protein binding ---
    (
        "protein_binding",
        r"(?:plasma\s+)?protein\s*(?:[-–]?\s*)?bind(?:ing|s)[^\d]{0,60}?"
        r"(?:" + _VERB + r"\s+)" + _OPT_APPROX + _NUM + r"\s*%",
        1,
        None,
    ),
    (
        "protein_binding",
        _NUM + r"\s*%\s*(?:is\s+)?(?:bound\s+to\s+(?:plasma\s+)?proteins?|protein[- ]bound)",
        1,
        None,
    ),
    (
        "protein_binding",
        r"(?:plasma\s+)?protein\s+binding\s+(?:is\s+|was\s+)?" + _OPT_APPROX + _NUM + r"\s*%",
        1,
        None,
    ),
    (
        "protein_binding",
        r"bound\s+to\s+(?:human\s+)?(?:plasma\s+)?(?:serum\s+)?proteins?\s*(?:to\s+the\s+extent\s+of\s+)?"
        + _OPT_APPROX
        + _NUM
        + r"\s*%",
        1,
        None,
    ),
    # --- Clearance ---
    (
        "clearance",
        r"(?:(?:oral|renal|total|apparent|systemic|plasma|body)\s+)?(?:clearance|CL(?:/F)?)[^\d]{0,60}?"
        r"(?:"
        + _VERB
        + r"\s+)"
        + _OPT_APPROX
        + _NUM
        + _OPT_PLUSMINUS
        + r"\s*(mL/min(?:/kg)?|L/h(?:r|our)?(?:/kg)?)",
        1,
        2,
    ),
    (
        "clearance",
        r"(?:clearance|CL(?:/F)?)\s*\(\s*(mL/min(?:/kg)?|L/h(?:r)?(?:/kg)?)\s*\)\s*" + _NUM,
        2,
        1,
    ),
    # --- Volume of distribution ---
    (
        "vd",
        r"(?:volume\s+of\s+distribution|V\s*d|V/F|Vd?ss)[^\d]{0,80}?"
        r"(?:" + _VERB + r"\s+)" + _OPT_APPROX + _NUM + _OPT_PLUSMINUS + r"\s*(liters?|L\b|L/kg)",
        1,
        2,
    ),
    (
        "vd",
        r"(?:volume\s+of\s+distribution|V\s*d|V/F|Vd?ss)\s*\(\s*(L(?:/kg)?|liters?)\s*\)\s*" + _NUM,
        2,
        1,
    ),
    # --- Dose (for context) ---
    ("dose", r"(?:single|oral)\s+(?:dose\s+of\s+|)" + _NUM + r"\s*(?:[-–]\s*\d+\s*)?mg", 1, None),
    ("dose", r"(?:administered|given|received)\s+(?:a\s+)?" + _NUM + r"\s*mg", 1, None),
]

# Compile patterns
PK_PATTERNS = []
for param, pat, vg, ug in _RAW_PATTERNS:
    PK_PATTERNS.append((param, re.compile(pat, re.IGNORECASE), vg, ug))


def convert_cmax_to_mg_L(value: float, unit: str) -> float | None:
    """Convert Cmax to mg/L."""
    u = unit.lower().replace(" ", "")
    if "ng/ml" in u:
        return value / 1000.0
    if any(x in u for x in ["ug/ml", "µg/ml", "μg/ml", "mcg/ml"]):
        return value  # ug/mL == mg/L
    if "mg/l" in u:
        return value
    if "mg/dl" in u:
        return value * 10.0
    return None


def convert_auc_to_mg_h_L(value: float, unit: str) -> float | None:
    """Convert AUC to mg*h/L."""
    u = unit.lower().replace(" ", "")
    if "ng" in u:
        return value / 1000.0
    if any(x in u for x in ["ug", "µg", "μg", "mcg"]):
        return value
    if "mg" in u:
        return value * 1000.0
    return None


def convert_thalf_to_hours(value: float, unit: str) -> float | None:
    """Convert half-life to hours."""
    u = unit.lower()
    if "day" in u:
        return value * 24.0
    if "min" in u:
        return value / 60.0
    return value  # hours


def extract_pk_from_text(text: str) -> dict:
    """Extract all PK parameters from clinical pharmacology text.

    Returns dict with keys: cmax_value, cmax_unit, auc_value, auc_unit,
    thalf_h, bioavailability_pct, protein_binding_pct, clearance_L_h,
    vd_L, dose_mg, plus raw values.
    """
    raw: dict[str, list[tuple[float, str]]] = {}

    for param_name, pattern, val_group, unit_group in PK_PATTERNS:
        for match in pattern.finditer(text):
            try:
                value = float(match.group(val_group))
            except (ValueError, IndexError):
                continue

            # Get unit
            if unit_group is not None:
                try:
                    unit = match.group(unit_group).strip()
                except (IndexError, AttributeError):
                    unit = ""
            else:
                if param_name == "bioavailability":
                    unit = "%"
                elif param_name == "protein_binding":
                    unit = "%"
                elif param_name == "t_half":
                    unit = "hours"
                elif param_name == "dose":
                    unit = "mg"
                else:
                    unit = ""

            # Sanity checks
            if param_name == "cmax" and (value < 0.01 or value > 1e6):
                continue
            if param_name == "auc" and (value < 0.1 or value > 1e7):
                continue
            if param_name == "t_half":
                h = convert_thalf_to_hours(value, unit)
                if h is None or h < 0.1 or h > 1000:
                    continue
            if param_name == "bioavailability" and (value < 1 or value > 100):
                continue
            if param_name == "protein_binding" and (value < 1 or value > 100):
                continue
            if param_name == "clearance" and (value < 0.01 or value > 1e5):
                continue
            if param_name == "vd" and (value < 0.01 or value > 1e5):
                continue
            if param_name == "dose" and (value < 0.5 or value > 5000):
                continue

            raw.setdefault(param_name, []).append((value, unit))

    # Build output — take first match for each parameter
    result: dict = {
        "cmax_value": None,
        "cmax_unit": None,
        "auc_value": None,
        "auc_unit": None,
        "thalf_h": None,
        "bioavailability_pct": None,
        "protein_binding_pct": None,
        "clearance_L_h": None,
        "vd_L": None,
        "dose_mg": None,
    }

    if "cmax" in raw:
        val, unit = raw["cmax"][0]
        mg_l = convert_cmax_to_mg_L(val, unit)
        if mg_l is not None:
            result["cmax_value"] = round(mg_l, 6)
            result["cmax_unit"] = "mg/L"
        else:
            result["cmax_value"] = round(val, 4)
            result["cmax_unit"] = unit

    if "auc" in raw:
        val, unit = raw["auc"][0]
        mg_h_l = convert_auc_to_mg_h_L(val, unit)
        if mg_h_l is not None:
            result["auc_value"] = round(mg_h_l, 6)
            result["auc_unit"] = "mg*h/L"
        else:
            result["auc_value"] = round(val, 4)
            result["auc_unit"] = unit

    if "t_half" in raw:
        val, unit = raw["t_half"][0]
        h = convert_thalf_to_hours(val, unit)
        if h is not None:
            result["thalf_h"] = round(h, 2)

    if "bioavailability" in raw:
        result["bioavailability_pct"] = round(raw["bioavailability"][0][0], 1)

    if "protein_binding" in raw:
        result["protein_binding_pct"] = round(raw["protein_binding"][0][0], 1)

    if "clearance" in raw:
        val, unit = raw["clearance"][0]
        u = unit.lower()
        if "ml/min" in u and "/kg" not in u:
            result["clearance_L_h"] = round(val * 0.06, 3)  # mL/min -> L/h
        elif "l/h" in u:
            result["clearance_L_h"] = round(val, 3)
        else:
            result["clearance_L_h"] = round(val, 3)

    if "vd" in raw:
        val, unit = raw["vd"][0]
        u = unit.lower()
        if "kg" in u:
            result["vd_L"] = round(val * 70.0, 1)  # assume 70kg
        else:
            result["vd_L"] = round(val, 1)

    if "dose" in raw:
        result["dose_mg"] = round(raw["dose"][0][0], 1)

    return result


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def fetch_openfda_page(skip: int, session: requests.Session) -> dict | None:
    """Fetch one page of OpenFDA drug labels (oral, with clinical_pharmacology)."""
    cache_file = CACHE_DIR / f"bulk_skip_{skip}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    params = {
        "search": 'openfda.route:"ORAL" AND _exists_:clinical_pharmacology',
        "limit": PAGE_SIZE,
        "skip": skip,
    }
    try:
        resp = session.get(OPENFDA_BASE, params=params, timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        # Cache
        with open(cache_file, "w") as f:
            json.dump(data, f)
        return data
    except Exception as e:
        print(f"  WARNING: OpenFDA request skip={skip} failed: {e}")
        return None


def fetch_smiles_pubchem(
    drug_name: str, session: requests.Session, smiles_cache: dict
) -> str | None:
    """Fetch canonical SMILES from PubChem by drug name."""
    key = drug_name.lower().strip()
    if key in smiles_cache:
        return smiles_cache[key]

    url = f"{PUBCHEM_BASE}/{requests.utils.quote(drug_name)}/property/CanonicalSMILES/JSON"
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            props = data.get("PropertyTable", {}).get("Properties", [])
            if props:
                smi = props[0].get("CanonicalSMILES") or props[0].get("ConnectivitySMILES")
                smiles_cache[key] = smi
                return smi
        smiles_cache[key] = None
        return None
    except Exception:
        smiles_cache[key] = None
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Bulk FDA PK extraction")
    parser.add_argument(
        "--max-pages", type=int, default=50, help="Max pages to fetch (100 labels each, max 50)"
    )
    parser.add_argument("--no-pubchem", action="store_true", help="Skip PubChem SMILES lookup")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = "OmegaPBPK/1.0 (research; PK extraction)"

    # --- Step 1: Fetch all labels ---
    print("=" * 70)
    print("Step 1: Fetching FDA drug labels from OpenFDA API")
    print("=" * 70)

    all_results = []
    max_pages = min(args.max_pages, MAX_SKIP // PAGE_SIZE)

    for page in range(max_pages):
        skip = page * PAGE_SIZE
        if skip > MAX_SKIP:
            break
        cached = (CACHE_DIR / f"bulk_skip_{skip}.json").exists()
        tag = "(cached)" if cached else "(fetching)"
        print(f"  Page {page + 1}/{max_pages}, skip={skip} {tag}")

        data = fetch_openfda_page(skip, session)
        if data is None:
            print(f"  No more results at skip={skip}")
            break

        results = data.get("results", [])
        if not results:
            print(f"  Empty results at skip={skip}")
            break

        all_results.extend(results)

        if not cached:
            time.sleep(OPENFDA_DELAY)

    print(f"\n  Total labels fetched: {len(all_results)}")

    # --- Step 2: Extract PK parameters ---
    print("\n" + "=" * 70)
    print("Step 2: Extracting PK parameters from label text")
    print("=" * 70)

    # drug_name -> {pk_params, source_text, nda_number, param_count}
    drug_data: dict[str, dict] = {}

    for result in all_results:
        openfda = result.get("openfda", {})

        # Get drug name (prefer generic)
        generic_names = openfda.get("generic_name", [])
        brand_names = openfda.get("brand_name", [])
        if generic_names:
            raw_name = generic_names[0] if isinstance(generic_names, list) else generic_names
        elif brand_names:
            raw_name = brand_names[0] if isinstance(brand_names, list) else brand_names
        else:
            continue

        name = normalize_drug_name(raw_name)
        if not name or len(name) < 2:
            continue

        # Collect PK text
        pk_parts = []
        for field in ["clinical_pharmacology", "clinical_pharmacology_table", "pharmacokinetics"]:
            raw = result.get(field)
            if raw:
                text = raw[0] if isinstance(raw, list) else raw
                if text and len(text) > 30:
                    pk_parts.append(text)

        pk_text = "\n".join(pk_parts)
        if len(pk_text) < 80:
            continue

        # Extract
        params = extract_pk_from_text(pk_text)

        # Count useful params (Cmax, AUC, t_half)
        count = sum(1 for k in ["cmax_value", "auc_value", "thalf_h"] if params.get(k) is not None)

        # NDA / application number
        app_numbers = openfda.get("application_number", [])
        nda = app_numbers[0] if app_numbers else None

        # Source excerpt (first 500 chars)
        source_excerpt = pk_text[:500].replace("\n", " ").strip()

        # Dedup: keep entry with most PK params
        if name in drug_data:
            old_count = drug_data[name]["param_count"]
            if count <= old_count:
                continue

        drug_data[name] = {
            "pk_params": params,
            "source_text": source_excerpt,
            "nda_number": nda,
            "param_count": count,
            "raw_name": raw_name,
        }

    # Filter: at least 2 of (Cmax, AUC, t_half)
    filtered = {name: data for name, data in drug_data.items() if data["param_count"] >= 2}

    print(f"  Drugs with any PK data: {len(drug_data)}")
    print(f"  Drugs with >=2 key params: {len(filtered)}")

    # Also keep drugs with 1 key param + protein_binding or bioavailability
    for name, data in drug_data.items():
        if name in filtered:
            continue
        pk = data["pk_params"]
        key_count = data["param_count"]
        extra_count = sum(
            1 for k in ["protein_binding_pct", "bioavailability_pct"] if pk.get(k) is not None
        )
        if key_count >= 1 and extra_count >= 1:
            filtered[name] = data

    print(f"  After relaxed filter (1 key + 1 extra): {len(filtered)}")

    # --- Step 3: Fetch SMILES from PubChem ---
    print("\n" + "=" * 70)
    print("Step 3: Fetching SMILES from PubChem")
    print("=" * 70)

    # Load SMILES cache
    smiles_cache_file = CACHE_DIR / "smiles_cache.json"
    smiles_cache: dict[str, str | None] = {}
    if smiles_cache_file.exists():
        with open(smiles_cache_file) as f:
            smiles_cache = json.load(f)

    if args.no_pubchem:
        print("  Skipping PubChem (--no-pubchem flag)")
    else:
        names_to_fetch = [n for n in filtered if n.lower() not in smiles_cache]
        print(f"  Need SMILES for {len(names_to_fetch)} drugs ({len(smiles_cache)} cached)")

        for i, name in enumerate(names_to_fetch):
            if (i + 1) % 20 == 0 or i == 0:
                print(f"  Fetching {i + 1}/{len(names_to_fetch)}: {name}")
            fetch_smiles_pubchem(name, session, smiles_cache)
            time.sleep(PUBCHEM_DELAY)

        # Save updated cache
        with open(smiles_cache_file, "w") as f:
            json.dump(smiles_cache, f, indent=2)

    # --- Step 4: Build output ---
    print("\n" + "=" * 70)
    print("Step 4: Building output JSON")
    print("=" * 70)

    drugs_list = []
    for name in sorted(filtered):
        data = filtered[name]
        pk = data["pk_params"]
        smiles = smiles_cache.get(name.lower())

        entry = {
            "name": name,
            "smiles": smiles,
            "smiles_source": "pubchem" if smiles else None,
            "route": "oral",
            "pk_params": {
                "cmax_value": pk.get("cmax_value"),
                "cmax_unit": pk.get("cmax_unit"),
                "auc_value": pk.get("auc_value"),
                "auc_unit": pk.get("auc_unit"),
                "thalf_h": pk.get("thalf_h"),
                "bioavailability_pct": pk.get("bioavailability_pct"),
                "protein_binding_pct": pk.get("protein_binding_pct"),
                "clearance_L_h": pk.get("clearance_L_h"),
                "vd_L": pk.get("vd_L"),
            },
            "dose_mg": pk.get("dose_mg"),
            "source_text": data["source_text"],
            "nda_number": data["nda_number"],
        }
        drugs_list.append(entry)

    output = {
        "metadata": {
            "source": "OpenFDA Drug Label API",
            "extraction_date": str(date.today()),
            "n_drugs": len(drugs_list),
            "n_with_smiles": sum(1 for d in drugs_list if d["smiles"]),
            "extraction_script": "scripts/extract_fda_pk_bulk.py",
        },
        "drugs": drugs_list,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print(f"  Total drugs: {len(drugs_list)}")
    print(f"  With SMILES: {sum(1 for d in drugs_list if d['smiles'])}")

    param_counts = {}
    for d in drugs_list:
        pk = d["pk_params"]
        for k, v in pk.items():
            if v is not None and not k.endswith("_unit"):
                param_counts[k] = param_counts.get(k, 0) + 1

    print("\n  Parameter coverage:")
    for k in [
        "cmax_value",
        "auc_value",
        "thalf_h",
        "bioavailability_pct",
        "protein_binding_pct",
        "clearance_L_h",
        "vd_L",
    ]:
        count = param_counts.get(k, 0)
        pct = 100 * count / len(drugs_list) if drugs_list else 0
        print(f"    {k:25s}: {count:4d} ({pct:5.1f}%)")

    print(f"\n  Output: {OUTPUT_PATH}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
