#!/usr/bin/env python3
"""DailyMed FDA label PK extraction.

Phase 0: Tests 20 drugs to measure what fraction have Cmax in <table> elements
vs narrative text in their SPL XML labels.

Full mode: Extracts Cmax from all candidate drugs using table parsing + text fallback.

Usage:
    python scripts/fetch_dailymed_pk.py                  # Phase 0 (20 drugs)
    python scripts/fetch_dailymed_pk.py --full            # Full extraction
    python scripts/fetch_dailymed_pk.py --full --output X # Full extraction to CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

CACHE_DIR = Path("data/ml/dailymed_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DAILYMED_SEARCH = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"
DAILYMED_SPL = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}.xml"

# 20 test drugs for Phase 0
PHASE0_DRUGS = [
    "metformin",
    "atorvastatin",
    "omeprazole",
    "metoprolol",
    "warfarin",
    "ibuprofen",
    "caffeine",
    "midazolam",
    "fluconazole",
    "carbamazepine",
    "amlodipine",
    "lisinopril",
    "sertraline",
    "escitalopram",
    "rosuvastatin",
    "montelukast",
    "losartan",
    "clopidogrel",
    "duloxetine",
    "aripiprazole",
]

CMAX_PATTERN = re.compile(
    r"C\s*max|peak\s+(?:plasma\s+)?concentration",
    re.IGNORECASE,
)

CMAX_VALUE_PATTERN = re.compile(
    r"(\d+\.?\d*)\s*(ng/m[Ll]|[uµ]g/m[Ll]|mg/[Ll]|mcg/m[Ll])",
)

# Unit conversion factors to mg/L
UNIT_TO_MG_L = {
    "ng/ml": 0.001,
    "ng/mL": 0.001,
    "ug/ml": 1.0,
    "ug/mL": 1.0,
    "µg/ml": 1.0,
    "µg/mL": 1.0,
    "mcg/ml": 1.0,
    "mcg/mL": 1.0,
    "mg/l": 1.0,
    "mg/L": 1.0,
}

_last_req = 0.0


def _rate_limited_get(url: str, params: dict | None = None, timeout: int = 30) -> requests.Response:
    global _last_req
    elapsed = time.monotonic() - _last_req
    if elapsed < 0.5:
        time.sleep(0.5 - elapsed)
    resp = requests.get(url, params=params, timeout=timeout)
    _last_req = time.monotonic()
    return resp


def fetch_spl_xml(drug_name: str) -> str | None:
    """Fetch SPL XML for a drug from DailyMed. Returns XML string or None."""
    cache_path = CACHE_DIR / f"{drug_name.lower()}_spl.xml"
    if cache_path.exists():
        return cache_path.read_text()

    # Search for drug
    try:
        resp = _rate_limited_get(
            DAILYMED_SEARCH,
            params={"drug_name": drug_name, "page": 1, "pagesize": 5},
            timeout=30,
        )
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None

    results = resp.json().get("data", [])
    if not results:
        return None

    # Prefer oral dosage forms
    setid = None
    for r in results:
        title = (r.get("title") or "").lower()
        if any(w in title for w in ["tablet", "capsule", "oral"]):
            setid = r.get("setid")
            break
    if setid is None:
        setid = results[0].get("setid")
    if setid is None:
        return None

    # Fetch SPL XML
    try:
        xml_resp = _rate_limited_get(DAILYMED_SPL.format(setid=setid), timeout=30)
    except requests.RequestException:
        return None
    if xml_resp.status_code != 200:
        return None

    xml_text = xml_resp.text
    cache_path.write_text(xml_text)
    return xml_text


def _find_pk_sections(soup: BeautifulSoup) -> list:
    """Find all PK-related sections in SPL XML, including subsections."""
    pk_sections = []
    for section in soup.find_all("section"):
        title_el = section.find("title")
        if title_el and re.search(
            r"clinical\s+pharmacol|pharmacokinetic|absorption|12\.3|12 ",
            title_el.get_text(),
            re.IGNORECASE,
        ):
            pk_sections.append(section)
    return pk_sections


def _get_section_text(section) -> str:
    """Get ALL text from a section, including content/text/paragraph tags."""
    parts = []
    for el in section.find_all(["paragraph", "content", "text"]):
        parts.append(el.get_text())
    if not parts:
        parts.append(section.get_text())
    return "\n".join(parts)


# Broader Cmax value patterns — handles more text styles
CMAX_TEXT_PATTERNS = [
    # "Cmax of 123 ng/mL", "Cmax was 123 ng/mL", "Cmax is 123 ng/mL"
    re.compile(
        r"C\s*max\s*(?:of|was|is|=|:)\s*(?:approximately\s+)?(\d[\d,]*\.?\d*)\s*(ng/m[Ll]|[uµ]g/m[Ll]|mg/[Ll]|mcg/m[Ll])",
        re.IGNORECASE,
    ),
    # "mean Cmax 123 ng/mL" or "mean Cmax (123 ng/mL)"
    re.compile(
        r"(?:mean|average|median|geometric\s+mean)\s+C\s*max\s*[\(]?\s*(\d[\d,]*\.?\d*)\s*(ng/m[Ll]|[uµ]g/m[Ll]|mg/[Ll]|mcg/m[Ll])",
        re.IGNORECASE,
    ),
    # "Cmax (123 ng/mL)" or "Cmax = 123 ± 45 ng/mL"
    re.compile(
        r"C\s*max\s*[\(=]\s*(\d[\d,]*\.?\d*)\s*(?:[±\+\-]\s*\d[\d,]*\.?\d*)?\s*(ng/m[Ll]|[uµ]g/m[Ll]|mg/[Ll]|mcg/m[Ll])",
        re.IGNORECASE,
    ),
    # "peak plasma concentration of 123 ng/mL"
    re.compile(
        r"peak\s+(?:plasma\s+)?(?:concentration|level)s?\s+(?:of\s+|was\s+|is\s+|are\s+)?(?:approximately\s+)?(\d[\d,]*\.?\d*)\s*(ng/m[Ll]|[uµ]g/m[Ll]|mg/[Ll]|mcg/m[Ll])",
        re.IGNORECASE,
    ),
    # "peak levels ... of 1.8 mcg/mL" or "peak plasma levels ... 1.8 mcg/mL"
    re.compile(
        r"peak\s+(?:plasma\s+)?levels?\s+.*?(\d[\d,]*\.?\d*)\s*(ng/m[Ll]|[uµ]g/m[Ll]|mg/[Ll]|mcg/m[Ll])",
        re.IGNORECASE,
    ),
    # "Cmax\nof 6.72 mcg/mL" — handles whitespace/newlines between Cmax and value
    re.compile(
        r"C\s*max\s{0,5}(?:of|was|is)?\s*(\d[\d,]*\.?\d*)\s*(ng/m[Ll]|[uµ]g/m[Ll]|mg/[Ll]|mcg/m[Ll])",
        re.IGNORECASE,
    ),
]


def analyze_pk_tables(xml_text: str) -> dict:
    """Analyze an SPL XML for PK table presence and Cmax extractability."""
    soup = BeautifulSoup(xml_text, "lxml-xml")
    pk_sections = _find_pk_sections(soup)

    if not pk_sections:
        return {
            "status": "no_pk_section",
            "tables": 0,
            "cmax_in_table": False,
            "cmax_in_text": False,
        }

    tables_found = 0
    cmax_in_table = False
    cmax_in_text = False
    cmax_table_values = []
    cmax_text_values = []

    for sec in pk_sections:
        # Check tables
        for table in sec.find_all("table"):
            table_text = table.get_text()
            tables_found += 1
            if CMAX_PATTERN.search(table_text):
                matches = CMAX_VALUE_PATTERN.findall(table_text)
                if matches:
                    cmax_in_table = True
                    for val, unit in matches:
                        cmax_table_values.append({"value": float(val), "unit": unit})

        # Check ALL text (paragraphs, content, text elements)
        sec_text = _get_section_text(sec)
        for pat in CMAX_TEXT_PATTERNS:
            m = pat.search(sec_text)
            if m:
                val_str = m.group(1).replace(",", "")
                try:
                    val = float(val_str)
                    unit = m.group(2)
                    cmax_in_text = True
                    cmax_text_values.append({"value": val, "unit": unit})
                except ValueError:
                    pass

    return {
        "status": "found",
        "tables": tables_found,
        "cmax_in_table": cmax_in_table,
        "cmax_in_text": cmax_in_text,
        "cmax_table_values": cmax_table_values[:5],
        "cmax_text_values": cmax_text_values[:5],
    }


def extract_cmax_from_table(xml_text: str, drug_name: str) -> dict | None:
    """Extract Cmax from structured tables in SPL XML.

    Parses table headers to find Cmax column, extracts value and unit,
    converts to mg/L.
    """
    soup = BeautifulSoup(xml_text, "lxml-xml")
    pk_sections = _find_pk_sections(soup)

    for sec in pk_sections:
        for table in sec.find_all("table"):
            table_text = table.get_text()
            if not CMAX_PATTERN.search(table_text):
                continue

            # Try to parse table structure
            rows = table.find_all("tr")
            if not rows:
                continue

            # Find header row with Cmax
            cmax_col = -1
            dose_col = -1

            for row_idx, row in enumerate(rows):
                cells = row.find_all(["th", "td"])
                cell_texts = [c.get_text(strip=True) for c in cells]

                for i, text in enumerate(cell_texts):
                    if re.search(r"C\s*max", text, re.IGNORECASE):
                        cmax_col = i
                        # Extract unit from header
                        unit_match = re.search(
                            r"\((ng/m[Ll]|[uµ]g/m[Ll]|mg/[Ll]|mcg/m[Ll])\)", text
                        )
                        header_unit = unit_match.group(1) if unit_match else None
                    if re.search(r"dose|strength", text, re.IGNORECASE):
                        dose_col = i

                if cmax_col >= 0:
                    # Now parse data rows after header
                    for data_row in rows[row_idx + 1 :]:
                        data_cells = data_row.find_all(["th", "td"])
                        data_texts = [c.get_text(strip=True) for c in data_cells]
                        if len(data_texts) <= cmax_col:
                            continue

                        cmax_text = data_texts[cmax_col]
                        # Handle "mean ± SD" format
                        cmax_text = re.split(r"[±\+\-]", cmax_text)[0].strip()
                        # Remove commas
                        cmax_text = cmax_text.replace(",", "")

                        try:
                            cmax_val = float(cmax_text)
                        except ValueError:
                            # Try regex extraction
                            m = re.search(r"(\d+\.?\d*)", cmax_text)
                            if m:
                                cmax_val = float(m.group(1))
                            else:
                                continue

                        # Determine unit
                        unit = header_unit
                        if unit is None:
                            # Check cell text for unit
                            m = CMAX_VALUE_PATTERN.search(data_texts[cmax_col])
                            if m:
                                unit = m.group(2)

                        if unit is None:
                            continue

                        # Convert to mg/L
                        factor = UNIT_TO_MG_L.get(unit)
                        if factor is None:
                            continue
                        cmax_mg_L = cmax_val * factor

                        # Extract dose if available
                        dose_mg = None
                        if dose_col >= 0 and dose_col < len(data_texts):
                            dm = re.search(r"(\d+\.?\d*)\s*mg", data_texts[dose_col])
                            if dm:
                                dose_mg = float(dm.group(1))

                        if cmax_mg_L > 0:
                            return {
                                "cmax_mg_L": cmax_mg_L,
                                "cmax_raw": cmax_val,
                                "unit_raw": unit,
                                "dose_mg": dose_mg,
                                "extraction_method": "table",
                                "drug": drug_name,
                            }
                    break  # Found Cmax column, done with this table

    return None


def extract_cmax_from_text(xml_text: str, drug_name: str) -> dict | None:
    """Extract Cmax from narrative text as fallback.

    Looks for patterns like "Cmax of 123 ng/mL" or "peak concentration was 4.5 µg/mL".
    Uses broadened section search and multiple regex patterns.
    """
    soup = BeautifulSoup(xml_text, "lxml-xml")
    pk_sections = _find_pk_sections(soup)

    # Also look for dose in nearby text
    dose_patterns = [
        re.compile(r"(\d+\.?\d*)\s*mg\s+(?:oral|dose|single|tablet|capsule)", re.IGNORECASE),
        re.compile(r"(?:oral|single)\s+(?:dose\s+(?:of\s+)?)?(\d+\.?\d*)\s*mg", re.IGNORECASE),
        re.compile(r"(\d+\.?\d*)\s*mg\s+(?:once|twice|daily)", re.IGNORECASE),
    ]

    for sec in pk_sections:
        sec_text = _get_section_text(sec)
        for pat in CMAX_TEXT_PATTERNS:
            m = pat.search(sec_text)
            if m:
                val_str = m.group(1).replace(",", "")
                try:
                    cmax_val = float(val_str)
                except ValueError:
                    continue
                unit = m.group(2)
                factor = UNIT_TO_MG_L.get(unit)
                if factor is None:
                    continue
                cmax_mg_L = cmax_val * factor
                if cmax_mg_L <= 0:
                    continue

                # Try to find dose in same section text
                dose_mg = None
                for dp in dose_patterns:
                    dm = dp.search(sec_text)
                    if dm:
                        dose_mg = float(dm.group(1))
                        break

                return {
                    "cmax_mg_L": cmax_mg_L,
                    "cmax_raw": cmax_val,
                    "unit_raw": unit,
                    "dose_mg": dose_mg,
                    "extraction_method": "text",
                    "drug": drug_name,
                }

    return None


def detect_fasting_state(xml_text: str) -> str:
    """Detect fasting state from SPL XML.

    Returns: confirmed_fasted, assumed_fasted, or fed_only.
    """
    soup = BeautifulSoup(xml_text, "lxml-xml")
    full_text = soup.get_text().lower()

    fasted_kw = ["fasting", "fasted state", "empty stomach", "fasting conditions"]
    fed_kw = ["fed state", "with food", "high-fat meal", "after a meal"]

    has_fasted = any(kw in full_text for kw in fasted_kw)
    has_fed = any(kw in full_text for kw in fed_kw)

    if has_fasted:
        return "confirmed_fasted"
    if has_fed:
        return "fed_only"
    return "assumed_fasted"


# ---------------------------------------------------------------------------
# Phase 0: yield prototype
# ---------------------------------------------------------------------------


def run_phase0():
    """Run Phase 0 yield prototype on 20 drugs."""
    print("=" * 60)
    print("Phase 0: DailyMed Yield Prototype (20 drugs)")
    print("=" * 60)

    results = {}
    for drug in PHASE0_DRUGS:
        print(f"  Fetching {drug}...", end=" ", flush=True)
        xml = fetch_spl_xml(drug)
        if xml is None:
            results[drug] = {"status": "fetch_failed"}
            print("FETCH FAILED")
            continue
        analysis = analyze_pk_tables(xml)
        results[drug] = analysis
        status_parts = []
        if analysis["cmax_in_table"]:
            status_parts.append("TABLE")
        if analysis["cmax_in_text"]:
            status_parts.append("TEXT")
        if not status_parts:
            status_parts.append("NO_CMAX")
        print(f"{' + '.join(status_parts)} ({analysis['tables']} tables)")

    # Summary
    n_total = len(results)
    n_table = sum(1 for r in results.values() if r.get("cmax_in_table"))
    n_text = sum(1 for r in results.values() if r.get("cmax_in_text"))
    n_either = sum(1 for r in results.values() if r.get("cmax_in_table") or r.get("cmax_in_text"))
    n_failed = sum(1 for r in results.values() if r.get("status") == "fetch_failed")

    print(f"\n--- Phase 0 Results ({n_total} drugs) ---")
    print(f"  Cmax in <table>: {n_table}/{n_total} ({100 * n_table / n_total:.0f}%)")
    print(f"  Cmax in text:    {n_text}/{n_total} ({100 * n_text / n_total:.0f}%)")
    print(f"  Cmax anywhere:   {n_either}/{n_total} ({100 * n_either / n_total:.0f}%)")
    print(f"  Fetch failed:    {n_failed}/{n_total}")

    gate_pass = n_table >= 10
    print(
        f"\n  Decision gate: {'PASS (>=50% table) -> Path A (XML tables)' if gate_pass else 'FAIL (<50% table) -> Path B (improved regex)'}"
    )

    # Save results
    out_path = Path("outputs/phase0_dailymed_yield.json")
    out_path.parent.mkdir(exist_ok=True)

    summary = {
        "n_total": n_total,
        "n_table": n_table,
        "n_text": n_text,
        "n_either": n_either,
        "n_failed": n_failed,
        "gate_pass": gate_pass,
        "recommended_path": "A_xml_tables" if gate_pass else "B_improved_regex",
        "per_drug": {},
    }
    for drug, r in results.items():
        summary["per_drug"][drug] = {
            "status": r.get("status", "found"),
            "tables": r.get("tables", 0),
            "cmax_in_table": r.get("cmax_in_table", False),
            "cmax_in_text": r.get("cmax_in_text", False),
        }

    out_path.write_text(json.dumps(summary, indent=2))
    print(f"  Results saved to {out_path}")

    return summary


# ---------------------------------------------------------------------------
# Full extraction
# ---------------------------------------------------------------------------


def run_full(drug_list: list[str], output_path: str | None = None):
    """Run full Cmax extraction on a list of drugs."""
    print("=" * 60)
    print(f"Full DailyMed Extraction ({len(drug_list)} drugs)")
    print("=" * 60)

    results = []
    for i, drug in enumerate(drug_list):
        print(f"  [{i + 1}/{len(drug_list)}] {drug}...", end=" ", flush=True)
        xml = fetch_spl_xml(drug)
        if xml is None:
            print("FETCH FAILED")
            continue

        # Try table extraction first
        extracted = extract_cmax_from_table(xml, drug)
        if extracted is None:
            # Fallback to text
            extracted = extract_cmax_from_text(xml, drug)

        if extracted is None:
            print("NO CMAX")
            continue

        # Detect fasting state
        fasted = detect_fasting_state(xml)
        extracted["fasted_confidence"] = fasted

        results.append(extracted)
        print(
            f"{extracted['extraction_method'].upper()} "
            f"Cmax={extracted['cmax_mg_L']:.4f} mg/L "
            f"({extracted['cmax_raw']} {extracted['unit_raw']}) "
            f"[{fasted}]"
        )

    # Summary
    print("\n--- Full Extraction Results ---")
    print(f"  Total drugs queried:  {len(drug_list)}")
    print(f"  Cmax extracted:       {len(results)}")
    n_table = sum(1 for r in results if r["extraction_method"] == "table")
    n_text = sum(1 for r in results if r["extraction_method"] == "text")
    print(f"    From tables: {n_table}")
    print(f"    From text:   {n_text}")

    # Save CSV
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "drug",
            "cmax_mg_L",
            "cmax_raw",
            "unit_raw",
            "dose_mg",
            "extraction_method",
            "fasted_confidence",
        ]
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"  Saved to {out}")

    return results


# ---------------------------------------------------------------------------
# Extended drug list for full extraction
# ---------------------------------------------------------------------------

# Common oral drugs with known PK — candidates for platinum benchmark
FULL_DRUG_LIST = [
    # Gold-24 drugs
    "caffeine",
    "metoprolol",
    "midazolam",
    "propranolol",
    "warfarin",
    "amphetamine",
    "ibuprofen",
    "acetaminophen",
    "amoxicillin",
    "atorvastatin",
    "carbamazepine",
    "diazepam",
    "digoxin",
    "fluoxetine",
    "nifedipine",
    "omeprazole",
    "phenytoin",
    "theophylline",
    "verapamil",
    "tramadol",
    "atenolol",
    "fluconazole",
    "furosemide",
    "gabapentin",
    "metformin",
    # Additional common drugs
    "amlodipine",
    "lisinopril",
    "sertraline",
    "escitalopram",
    "rosuvastatin",
    "montelukast",
    "losartan",
    "clopidogrel",
    "duloxetine",
    "aripiprazole",
    "simvastatin",
    "naproxen",
    "celecoxib",
    "meloxicam",
    "piroxicam",
    "ranitidine",
    "famotidine",
    "sildenafil",
    "tadalafil",
    "bupropion",
    "venlafaxine",
    "paroxetine",
    "citalopram",
    "mirtazapine",
    "risperidone",
    "olanzapine",
    "quetiapine",
    "haloperidol",
    "lamotrigine",
    "levetiracetam",
    "pregabalin",
    "topiramate",
    "alprazolam",
    "lorazepam",
    "buspirone",
    "modafinil",
    "dexamethasone",
    "prednisone",
    "ondansetron",
    "metoclopramide",
    "erythromycin",
    "levofloxacin",
    "ciprofloxacin",
    "amoxicillin",
    "azithromycin",
    "doxycycline",
    "metronidazole",
    "trimethoprim",
    "cetirizine",
    "fexofenadine",
    "loratadine",
    "diphenhydramine",
    "ranitidine",
    "pantoprazole",
    "esomeprazole",
    "lansoprazole",
    "lisinopril",
    "ramipril",
    "enalapril",
    "captopril",
    "losartan",
    "valsartan",
    "irbesartan",
    "candesartan",
    "hydrochlorothiazide",
    "furosemide",
    "spironolactone",
    "metformin",
    "glipizide",
    "pioglitazone",
    "sitagliptin",
    "atorvastatin",
    "rosuvastatin",
    "simvastatin",
    "pravastatin",
    "amlodipine",
    "nifedipine",
    "diltiazem",
    "verapamil",
    "codeine",
    "morphine",
    "oxycodone",
    "hydrocodone",
    "sumatriptan",
    "rizatriptan",
    "zolmitriptan",
    "amitriptyline",
    "nortriptyline",
    "imipramine",
    "lithium",
    "valproic acid",
    "oxcarbazepine",
    "methylphenidate",
    "atomoxetine",
    "sildenafil",
    "tamsulosin",
    "finasteride",
    "allopurinol",
    "colchicine",
    "warfarin",
    "apixaban",
    "rivaroxaban",
    "dabigatran",
    "clopidogrel",
    "ticagrelor",
]


def main():
    parser = argparse.ArgumentParser(description="DailyMed FDA label PK extraction")
    parser.add_argument("--full", action="store_true", help="Run full extraction")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument("--drugs", type=str, nargs="*", default=None, help="Specific drugs")
    args = parser.parse_args()

    if args.full:
        drugs = args.drugs if args.drugs else list(dict.fromkeys(FULL_DRUG_LIST))
        output = args.output or "data/ml/clinical/dailymed_extracted.csv"
        run_full(drugs, output)
    else:
        run_phase0()


if __name__ == "__main__":
    main()
