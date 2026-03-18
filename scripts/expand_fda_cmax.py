#!/usr/bin/env python3
"""Expand Cmax training data using FDA drug labels via OpenFDA API.

Queries cached (or live) OpenFDA drug label pages, extracts Cmax and dose
using regex patterns (reused from extract_fda_pk_bulk.py), normalizes units
to mg/L, fetches SMILES from PubChem, deduplicates against existing datasets,
and writes a CSV suitable for ML training.

Output: data/ml/clinical/fda_expanded_cmax.csv

Usage:
    python scripts/expand_fda_cmax.py [--max-pages 30] [--dry-run]
"""

import argparse
import csv
import json
import math
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "data" / "ml" / "fda_cache"
OUTPUT_PATH = REPO_ROOT / "data" / "ml" / "clinical" / "fda_expanded_cmax.csv"
EXISTING_TRAINING = REPO_ROOT / "data" / "ml" / "clinical" / "cmax_training_set.csv"
EXISTING_PKDB = REPO_ROOT / "data" / "ml" / "clinical" / "pkdb_expanded_cmax.csv"

OPENFDA_BASE = "https://api.fda.gov/drug/label.json"
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name"
OPENFDA_DELAY = 0.35  # seconds between live OpenFDA requests
PUBCHEM_DELAY = 0.3  # seconds between PubChem requests
PAGE_SIZE = 100
MAX_SKIP = 5000  # OpenFDA hard limit


# ---------------------------------------------------------------------------
# Salt / formulation suffixes to strip (from extract_fda_pk_bulk.py)
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
    if " and " in name:
        name = name.split(" and ")[0]
    if "/" in name:
        name = name.split("/")[0]
    if ", " in name:
        name = name.split(", ")[0]
    return name.strip()


# ---------------------------------------------------------------------------
# Regex patterns — reused from extract_fda_pk_bulk.py
# ---------------------------------------------------------------------------
_CONC_UNITS = r"(ng/[mM][lL]|[uμµ]g/[mM][lL]|mcg/[mM][lL]|mg/[lL])"
_VERB = (
    r"(?:was|of|is|approximately|about|~|=|:|averaged?|averaging|"
    r"ranges?\s+from|occurring|estimated|reported|reached)"
)
_NUM = r"(\d{1,6}(?:\.\d+)?)"
_OPT_APPROX = r"(?:approximately\s+|about\s+|~\s*|around\s+)?"

CMAX_PATTERNS = [
    # 0: "Cmax was 5.2 ng/mL", "peak plasma concentration of 3.4 ug/mL"
    re.compile(
        r"(?:C\s*max|peak\s+(?:plasma\s+)?concentration)[^\d]{0,60}?"
        r"(?:" + _VERB + r"\s+)" + _OPT_APPROX + _NUM + r"\s*" + _CONC_UNITS,
        re.IGNORECASE,
    ),
    # 1: Table form — "Cmax (ng/mL) 5.2"
    re.compile(
        r"C\s*max(?:\s*,?\s*ss)?\s*\(\s*" + _CONC_UNITS + r"\s*\)\s*" + _NUM,
        re.IGNORECASE,
    ),
    # 2: "Cmax of 5.2 ng/mL"
    re.compile(
        r"C\s*max\s*(?:of|was|is|=|:)\s*" + _OPT_APPROX + _NUM + r"\s*" + _CONC_UNITS,
        re.IGNORECASE,
    ),
    # 3: "Mean Cmax was 5.2 ng/mL"
    re.compile(
        r"[Mm]ean\s+C\s*max\s+(?:value\s+)?(?:was|of|is)\s+"
        + _OPT_APPROX
        + _NUM
        + r"\s*"
        + _CONC_UNITS,
        re.IGNORECASE,
    ),
    # 4: "Cmax) increased from 73 to 113 ng/mL" — pick first number
    re.compile(
        r"C\s*max\s*\)?\s*(?:increased|decreased|rose|ranged?)\s+(?:from\s+)?"
        + _OPT_APPROX
        + _NUM
        + r"\s*(?:to\s+\d+\.?\d*\s*)?"
        + _CONC_UNITS,
        re.IGNORECASE,
    ),
    # 5: "range from approximately 14 ± 4 to 37 ± 9 mcg/mL" near Cmax
    re.compile(
        r"C\s*max\s*\)?[^\d]{0,80}?(?:range[sd]?\s+(?:from\s+)?)"
        + _OPT_APPROX
        + _NUM
        + r"(?:\s*±\s*\d+\.?\d*)?\s*(?:to\s+\d+\.?\d*(?:\s*±\s*\d+\.?\d*)?\s*)?"
        + _CONC_UNITS,
        re.IGNORECASE,
    ),
    # 6: Table form with parenthetical CV — "Cmax (pg/mL) 984 (43)"
    re.compile(
        r"C\s*m\s*a\s*x\s*\(\s*" + _CONC_UNITS + r"\s*\)\s*" + _NUM,
        re.IGNORECASE,
    ),
    # 7: "Cmax ... N ng/mL" with up to 120 chars gap (looser match)
    re.compile(
        r"C\s*max[^\d]{0,120}?" + _NUM + r"\s*" + _CONC_UNITS,
        re.IGNORECASE,
    ),
]

# Pattern groups: (pattern, value_group_index, unit_group_index)
CMAX_PATTERN_GROUPS = [
    (CMAX_PATTERNS[0], 1, 2),  # verb-led
    (CMAX_PATTERNS[1], 2, 1),  # table form
    (CMAX_PATTERNS[2], 1, 2),  # of/was/is
    (CMAX_PATTERNS[3], 1, 2),  # mean Cmax
    (CMAX_PATTERNS[4], 1, 2),  # increased from
    (CMAX_PATTERNS[5], 1, 2),  # range from
    (CMAX_PATTERNS[6], 2, 1),  # table form with spaces
    (CMAX_PATTERNS[7], 1, 2),  # loose Cmax ... N unit
]

DOSE_PATTERNS = [
    # "following a 200 mg dose", "after a single oral dose of 500 mg"
    re.compile(
        r"(?:following|after|given|administered|received|with)\s+"
        r"(?:a\s+)?(?:single\s+)?(?:oral\s+)?(?:dose\s+of\s+)?"
        r"(\d{1,5})\s*(?:-?\s*mg|milligrams?)",
        re.IGNORECASE,
    ),
    # "single dose of 250 mg", "oral 400 mg"
    re.compile(
        r"(?:single|oral)\s+(?:dose\s+of\s+|)" + _NUM + r"\s*(?:[-\u2013]\s*\d+\s*)?mg",
        re.IGNORECASE,
    ),
    # "administered 100 mg"
    re.compile(
        r"(?:administered|given|received)\s+(?:a\s+)?" + _NUM + r"\s*mg",
        re.IGNORECASE,
    ),
    # "200 mg to [adult|healthy|human|normal] [subjects|volunteers|patients]"
    re.compile(
        r"(\d{1,5})\s*mg\s+(?:to\s+)?(?:adult|healthy|human|normal|male|female)?\s*"
        r"(?:adult\s+)?(?:subjects?|volunteers?|patients?|individuals?)",
        re.IGNORECASE,
    ),
    # "dose of 10 mg", "doses of 5 mg"
    re.compile(
        r"doses?\s+of\s+(\d{1,5})\s*mg",
        re.IGNORECASE,
    ),
    # "N mg tablets/capsules" — common in FDA labels
    re.compile(
        r"(\d{1,5})\s*mg\s+(?:tablets?|capsules?|dose)\s+(?:of\s+)?(?:\w+\s+){0,3}"
        r"(?:produced|resulted|gave|yielded|achieved)",
        re.IGNORECASE,
    ),
    # "at a dose of N mg" or "at the N mg dose"
    re.compile(
        r"at\s+(?:a\s+)?(?:the\s+)?(?:dose\s+of\s+)?(\d{1,5})\s*(?:-?\s*)?mg(?:\s+dose)?",
        re.IGNORECASE,
    ),
    # "N mg (two X mg tablets)" — indapamide pattern
    re.compile(
        r"(\d{1,5})\s*mg\s*\((?:two|three|four|one)\s+\d",
        re.IGNORECASE,
    ),
    # "<drug> N mg" — common pattern like "doxycycline 200 mg"
    # Exclude matches followed by /mL, /L, /dL (those are concentration units)
    re.compile(
        r"(?:^|\s)[a-z]+\s+(\d{1,5})\s*mg(?!\s*/|\s*\*|\s*\xb7)(?:\s+(?:oral|tablet|capsule|daily|dose|twice|once|per|bid|qd)|\b)",
        re.IGNORECASE,
    ),
]


# ---------------------------------------------------------------------------
# Unit normalization
# ---------------------------------------------------------------------------
def normalize_cmax_to_mg_L(value: float, unit: str) -> float | None:
    """Convert Cmax value to mg/L."""
    u = unit.lower().replace(" ", "")
    if "ng" in u:
        return value * 0.001  # ng/mL -> mg/L
    if any(x in u for x in ["ug", "\u00b5g", "\u03bcg", "mcg"]):
        return value  # ug/mL = mg/L
    if "mg" in u:
        return value  # mg/L already
    return None


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------
def extract_cmax(text: str) -> tuple[float | None, str | None, int | None]:
    """Extract first Cmax value + unit from clinical pharmacology text.

    Returns (value, unit, match_position) or (None, None, None).
    """
    for pattern, val_grp, unit_grp in CMAX_PATTERN_GROUPS:
        for m in pattern.finditer(text):
            try:
                value = float(m.group(val_grp))
                unit = m.group(unit_grp).strip()
            except (ValueError, IndexError, AttributeError):
                continue
            # Sanity: Cmax should be between 0.01 and 1e6 in raw units
            if value < 0.01 or value > 1e6:
                continue
            return value, unit, m.start()
    return None, None, None


def extract_dose(text: str) -> float | None:
    """Extract dose (mg) from clinical pharmacology text.

    Tries each pattern in order and returns the first plausible match.
    Rejects values that appear in daily-maximum-dose contexts.
    """
    for pattern in DOSE_PATTERNS:
        m = pattern.search(text)
        if m:
            try:
                dose = float(m.group(1))
                if not (0.5 <= dose <= 2000):
                    continue
                # Reject if it looks like a daily maximum, not a PK study dose
                context = text[max(0, m.start() - 30) : m.end() + 60].lower()
                if any(
                    kw in context
                    for kw in [
                        "per day",
                        "daily",
                        "24-hour",
                        "24 hour",
                        "maximum",
                        "not exceed",
                        "should not",
                    ]
                ):
                    continue
                return dose
            except (ValueError, IndexError):
                continue
    return None


# ---------------------------------------------------------------------------
# SMILES fetching and validation
# ---------------------------------------------------------------------------
def validate_smiles(smiles: str) -> bool:
    """Validate SMILES using RDKit."""
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except ImportError:
        # If RDKit not available, accept the SMILES
        return True


def fetch_smiles_pubchem(drug_name: str, session, smiles_cache: dict) -> str | None:
    """Fetch canonical SMILES from PubChem by drug name."""
    key = drug_name.lower().strip()
    if key in smiles_cache:
        return smiles_cache[key]

    try:
        import requests as req_lib
    except ImportError:
        return None

    url = f"{PUBCHEM_BASE}/{req_lib.utils.quote(drug_name)}/property/CanonicalSMILES/JSON"
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
# Load existing drug names for dedup
# ---------------------------------------------------------------------------
def load_existing_drugs() -> set[str]:
    """Load drug names from existing training CSV and PK-DB expansion."""
    existing = set()
    for path in [EXISTING_TRAINING, EXISTING_PKDB]:
        if path.exists():
            with open(path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.get("drug", "").lower().strip()
                    if name:
                        existing.add(name)
    return existing


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_extraction(max_pages: int = 30, dry_run: bool = False) -> list[dict]:
    """Run the full FDA Cmax extraction pipeline.

    Args:
        max_pages: Maximum number of OpenFDA pages to process.
        dry_run: If True, skip all network calls and return empty list.

    Returns:
        List of extracted drug records.
    """
    if dry_run:
        print(f"[DRY-RUN] Would process up to {max_pages} pages of FDA labels")
        print("[DRY-RUN] No network calls made.")
        return []

    try:
        import requests
    except ImportError:
        print("ERROR: requests not installed. Run: pip install requests")
        sys.exit(1)

    session = requests.Session()
    session.headers["User-Agent"] = "OmegaPBPK/1.0 (research; PK extraction)"

    # Load existing drugs for dedup
    existing_drugs = load_existing_drugs()
    print(f"Loaded {len(existing_drugs)} existing drugs for deduplication")

    # Load SMILES cache
    smiles_cache_file = CACHE_DIR / "smiles_cache.json"
    smiles_cache: dict[str, str | None] = {}
    if smiles_cache_file.exists():
        with open(smiles_cache_file) as f:
            smiles_cache = json.load(f)
    print(f"SMILES cache: {len(smiles_cache)} entries")

    # ---- Step 1: Fetch/read FDA label pages ----
    print("\n" + "=" * 70)
    print("Step 1: Reading FDA drug labels (cache + API)")
    print("=" * 70)

    all_results = []
    max_pages = min(max_pages, MAX_SKIP // PAGE_SIZE)

    for page in range(max_pages):
        skip = page * PAGE_SIZE
        if skip > MAX_SKIP:
            break

        cache_file = CACHE_DIR / f"bulk_skip_{skip}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                data = json.load(f)
            results = data.get("results", [])
            all_results.extend(results)
            if (page + 1) % 10 == 0:
                print(f"  Page {page + 1}/{max_pages} (cached): {len(results)} labels")
        else:
            # Live fetch
            params = {
                "search": 'openfda.route:"ORAL" AND _exists_:clinical_pharmacology',
                "limit": PAGE_SIZE,
                "skip": skip,
            }
            try:
                resp = session.get(OPENFDA_BASE, params=params, timeout=30)
                if resp.status_code == 404:
                    print(f"  No more results at skip={skip}")
                    break
                resp.raise_for_status()
                data = resp.json()

                # Cache the response
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                with open(cache_file, "w") as f:
                    json.dump(data, f)

                results = data.get("results", [])
                all_results.extend(results)
                print(f"  Page {page + 1}/{max_pages} (fetched): {len(results)} labels")
                time.sleep(OPENFDA_DELAY)
            except Exception as e:
                print(f"  WARNING: OpenFDA request skip={skip} failed: {e}")
                continue

    print(f"\n  Total labels loaded: {len(all_results)}")

    # ---- Step 2: Extract Cmax + dose from each label ----
    print("\n" + "=" * 70)
    print("Step 2: Extracting Cmax and dose from label text")
    print("=" * 70)

    # drug_name -> best record (highest info)
    drug_records: dict[str, dict] = {}
    n_with_cmax = 0
    n_with_dose = 0

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

        # Skip existing drugs
        if name in existing_drugs:
            continue

        # Collect PK text
        pk_parts = []
        for field in [
            "clinical_pharmacology",
            "clinical_pharmacology_table",
            "pharmacokinetics",
        ]:
            raw = result.get(field)
            if raw:
                text = raw[0] if isinstance(raw, list) else raw
                if text and len(text) > 30:
                    pk_parts.append(text)

        pk_text = "\n".join(pk_parts)
        if len(pk_text) < 80:
            continue

        # Extract Cmax
        cmax_raw, cmax_unit, cmax_pos = extract_cmax(pk_text)
        if cmax_raw is None or cmax_unit is None:
            continue

        cmax_mg_L = normalize_cmax_to_mg_L(cmax_raw, cmax_unit)
        if cmax_mg_L is None or cmax_mg_L <= 0:
            continue

        n_with_cmax += 1

        # Extract dose — try nearby Cmax text first, then full PK text,
        # then dosage_and_administration as fallback
        dose_mg = None

        # Strategy 1: search text near the Cmax match (±500 chars)
        if cmax_pos is not None:
            nearby_start = max(0, cmax_pos - 500)
            nearby_end = min(len(pk_text), cmax_pos + 500)
            dose_mg = extract_dose(pk_text[nearby_start:nearby_end])

        # Strategy 2: full PK text
        if dose_mg is None or dose_mg <= 0:
            dose_mg = extract_dose(pk_text)

        # Strategy 3: dosage_and_administration field
        if dose_mg is None or dose_mg <= 0:
            dose_text_parts = []
            for field in ["dosage_and_administration", "dosage_and_administration_table"]:
                raw = result.get(field)
                if raw:
                    text = raw[0] if isinstance(raw, list) else raw
                    if text and len(text) > 20:
                        dose_text_parts.append(text)
            if dose_text_parts:
                dose_mg = extract_dose("\n".join(dose_text_parts))
        if dose_mg is None or dose_mg < 1.0:
            continue

        n_with_dose += 1

        # Dedup: keep record with highest Cmax (most informative label)
        if name in drug_records:
            if cmax_mg_L <= drug_records[name]["cmax_mg_L"]:
                continue

        drug_records[name] = {
            "drug": name,
            "cmax_raw": cmax_raw,
            "cmax_unit": cmax_unit,
            "cmax_mg_L": cmax_mg_L,
            "dose_mg": dose_mg,
            "raw_name": raw_name,
        }

    print(f"  Labels with extractable Cmax: {n_with_cmax}")
    print(f"  Labels with Cmax + dose: {n_with_dose}")
    print(f"  Unique drugs (after dedup): {len(drug_records)}")

    # ---- Step 3: Fetch SMILES from PubChem ----
    print("\n" + "=" * 70)
    print("Step 3: Fetching SMILES from PubChem")
    print("=" * 70)

    names_needing_smiles = [n for n in drug_records if n.lower() not in smiles_cache]
    print(f"  Need SMILES for {len(names_needing_smiles)} drugs ({len(smiles_cache)} cached)")

    for i, name in enumerate(names_needing_smiles):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Fetching {i + 1}/{len(names_needing_smiles)}: {name}")
        fetch_smiles_pubchem(name, session, smiles_cache)
        time.sleep(PUBCHEM_DELAY)

    # Save updated SMILES cache
    smiles_cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(smiles_cache_file, "w") as f:
        json.dump(smiles_cache, f, indent=2)
    print(f"  SMILES cache updated: {len(smiles_cache)} total entries")

    # ---- Step 4: Validate SMILES and build final records ----
    print("\n" + "=" * 70)
    print("Step 4: Validating SMILES and building output")
    print("=" * 70)

    final_records = []
    seen_smiles: set[str] = set()  # dedup by canonical SMILES
    n_no_smiles = 0
    n_invalid_smiles = 0
    n_dup_smiles = 0
    n_quality_filtered = 0

    # Minimum molecular weight: skip tiny/fragment SMILES
    MIN_MW = 100.0

    for name in sorted(drug_records):
        rec = drug_records[name]
        smiles = smiles_cache.get(name.lower())

        if not smiles:
            n_no_smiles += 1
            continue

        if not validate_smiles(smiles):
            n_invalid_smiles += 1
            continue

        # Dedup by SMILES (e.g. "guanfacine" and "guanfacine er" share SMILES)
        # Use first component of multi-component SMILES for comparison
        canonical = smiles.split(".")[0] if "." in smiles else smiles
        if canonical in seen_smiles:
            n_dup_smiles += 1
            continue

        # Quality filter: MW > 100 (reject fragments like "CC" for "dimethyl")
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors

            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mw = Descriptors.MolWt(mol)
                if mw < MIN_MW:
                    print(f"    Skipping {name}: MW={mw:.1f} < {MIN_MW}")
                    n_quality_filtered += 1
                    continue
        except ImportError:
            pass

        # Quality filter: Cmax/dose plausibility
        # Cmax/dose should be between 1e-6 and 5 mg/L per mg
        # Most oral drugs have Cmax/dose in [1e-5, 1.0] range
        cmax_per_mg = rec["cmax_mg_L"] / rec["dose_mg"]
        if cmax_per_mg < 1e-6 or cmax_per_mg > 5:
            print(f"    Skipping {name}: Cmax/dose={cmax_per_mg:.2e} out of range")
            n_quality_filtered += 1
            continue

        seen_smiles.add(canonical)

        # Compute log(obs_cmax / dose) for the training set
        log_obs_cmax_per_mg = math.log(rec["cmax_mg_L"] / rec["dose_mg"])

        final_records.append(
            {
                "drug": name,
                "smiles": smiles,
                "dose_mg": rec["dose_mg"],
                "obs_cmax": rec["cmax_mg_L"],
                "pred_cmax_pbpk": "",  # to be filled by pipeline later
                "log_obs_cmax_per_mg": log_obs_cmax_per_mg,
                "tier": "fda_expanded",
            }
        )

    print(f"  No SMILES found: {n_no_smiles}")
    print(f"  Invalid SMILES: {n_invalid_smiles}")
    print(f"  Duplicate SMILES: {n_dup_smiles}")
    print(f"  Quality filtered: {n_quality_filtered}")
    print(f"  Final records: {len(final_records)}")

    return final_records


def save_csv(records: list[dict], path: Path) -> None:
    """Save records to CSV."""
    if not records:
        print("No records to save.")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "drug",
        "smiles",
        "dose_mg",
        "obs_cmax",
        "pred_cmax_pbpk",
        "log_obs_cmax_per_mg",
        "tier",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\nSaved {len(records)} records to {path}")


def main():
    parser = argparse.ArgumentParser(description="Expand Cmax training data from FDA drug labels")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=30,
        help="Max OpenFDA pages to process (100 labels each, max 50)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without making network calls",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("FDA Bulk Cmax Extraction for Omega PBPK")
    print("=" * 70)

    records = run_extraction(max_pages=args.max_pages, dry_run=args.dry_run)

    if args.dry_run:
        print("\n[DRY-RUN] Complete. No output file written.")
        return

    if records:
        save_csv(records, OUTPUT_PATH)

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        cmax_values = [r["obs_cmax"] for r in records]
        doses = [r["dose_mg"] for r in records]
        print(f"  Total drugs: {len(records)}")
        print(f"  Cmax range: {min(cmax_values):.4f} - {max(cmax_values):.2f} mg/L")
        print(f"  Dose range: {min(doses):.0f} - {max(doses):.0f} mg")

        # Show first 10
        print("\n  Sample (first 10):")
        for r in records[:10]:
            print(
                f"    {r['drug']:25s}  dose={r['dose_mg']:>7.1f} mg  "
                f"Cmax={r['obs_cmax']:>10.4f} mg/L"
            )
    else:
        print("\nNo records extracted.")


if __name__ == "__main__":
    main()
