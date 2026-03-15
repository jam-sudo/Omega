#!/usr/bin/env python3
"""Extract PK parameters from OpenFDA drug label pharmacokinetics text.

Extracts: Cmax, AUC, t½, Tmax, Vd, CL, bioavailability, protein binding
from cached OpenFDA label JSON files.

Output: data/ml/clinical/openfda_pk_extracted.csv
"""

import csv
import glob
import json
import os
import re
from dataclasses import dataclass


@dataclass
class PKExtraction:
    drug_name: str
    parameter: str
    value: float
    unit: str
    context: str  # snippet of surrounding text


def normalize_drug_name(name: str) -> str:
    """Normalize drug name: lowercase, strip salts/formulations."""
    name = name.lower().strip()
    # Remove common salt forms
    for suffix in [
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
        " ophthalmic solution",
        " extended-release capsules",
        " er tablets",
        " tablets",
        " capsules",
        " film coated",
        " anhydrous",
        ", benzalkonium chloride",
    ]:
        name = name.replace(suffix, "")
    # For combos like "X and Y", keep first drug only
    if " and " in name:
        name = name.split(" and ")[0]
    # For "X, Y, Z" combos, keep first
    if ", " in name:
        name = name.split(", ")[0]
    return name.strip()


# Unit pattern fragments (non-capturing for inline use, capturing versions for tabular)
_CONC_UNITS_NC = r"(?:ng/mL|µg/mL|mcg/mL|μg/mL|mg/L|µM|nM|ug/mL)"
_CONC_UNITS_C = r"(ng/mL|µg/mL|mcg/mL|μg/mL|mg/L|µM|nM|ug/mL)"
_AUC_UNITS_INNER = (
    r"(?:ng|µg|mcg|μg|ug|mg)[·•*×x\s]h(?:r|our)?s?/mL|"  # ng·hr/mL etc
    r"(?:ng|µg|mcg|μg|ug|mg)/mL[·•*×x\s]h(?:r|our)?s?|"  # ng/mL·hr etc
    r"h(?:r|our)?s?[·•*×x\s](?:ng|µg|mcg|μg|ug|mg)/(?:mL|ml)|"  # hr*ng/mL etc
    r"µM[·•*×x\s]h(?:r|our)?s?|"  # µM·hr
    r"hr?\*(?:ng|µg|mcg|μg|ug|mg)/(?:mL|ml)|"  # hr*mcg/ml (table format)
    r"(?:ng|µg|mcg|μg|ug|mg)\*h(?:r|our)?s?/(?:mL|ml)"  # mcg*hr/mL
)
_AUC_UNITS_NC = r"(?:" + _AUC_UNITS_INNER + r")"
_AUC_UNITS_C = r"(" + _AUC_UNITS_INNER + r")"
_TIME_UNITS = r"(?:hours?|hrs?|h\b|days?|minutes?|min)"
_VERB = r"(?:was|of|is|approximately|about|~|=|averaged?|averaging|ranges?\s+from|occurring|estimated|reported)"

# Patterns for PK parameter extraction
# Each pattern: (parameter_name, regex, value_group, unit_group)
PK_PATTERNS = [
    # --- Cmax ---
    # Pattern 1: Cmax was/of/= VALUE UNIT
    (
        "cmax",
        r"(?:C\s*max|peak\s+(?:plasma\s+)?concentration)[^\d]{0,60}?"
        r"(?:" + _VERB + r"\s*)"
        r"(\d+\.?\d*)\s*" + _CONC_UNITS_NC,
        1,
        2,
    ),
    # Pattern 2: Cmax (UNIT) VALUE  (tabular)
    (
        "cmax",
        r"C\s*max(?:\s*,?\s*ss)?\s*\(\s*" + _CONC_UNITS_C + r"\s*\)\s*(\d+\.?\d*)",
        2,
        1,
    ),
    # --- AUC ---
    # Pattern 1: AUC was/of VALUE UNIT
    (
        "auc",
        r"(?:AUC(?:\s*0?\s*[-–]\s*(?:∞|inf|infinity|last|tau|24\s*h?r?|12\s*h?r?))?|"
        r"area\s+under\s+the\s+(?:plasma\s+)?(?:concentration[- ]*time\s+)?curve)"
        r"[^\d]{0,80}?"
        r"(?:" + _VERB + r"\s*)"
        r"(\d+\.?\d*)\s*" + _AUC_UNITS_NC,
        1,
        2,
    ),
    # Pattern 2: AUC (UNIT) VALUE  (tabular)
    (
        "auc",
        r"AUC(?:\s*0?\s*[-–]\s*(?:∞|inf|infinity|last|tau))?\s*\(\s*"
        + _AUC_UNITS_C
        + r"\s*\)\s*(\d+\.?\d*)",
        2,
        1,
    ),
    # Pattern 3: AUC was VALUE UNIT (relaxed)
    (
        "auc",
        r"(?:AUC(?:\s*0?\s*[-–]\s*(?:∞|inf|last|tau))?)\s+(?:" + _VERB + r"\s+)"
        r"(\d+\.?\d*)\s+" + _AUC_UNITS_NC,
        1,
        2,
    ),
    # Pattern 4: AUC 0-∞ of VALUE (no unit — infer from context)
    (
        "auc",
        r"AUC(?:\s*0?\s*[-–]\s*(?:∞|inf|infinity|last|tau))\s+(?:" + _VERB + r"\s+)"
        r"(\d+\.?\d*)",
        1,
        None,
    ),
    # Pattern 5: AUC=VALUE (compact, e.g. "AUC=241.9")
    (
        "auc",
        r"AUC\s*=\s*(\d+\.?\d*)",
        1,
        None,
    ),
    # --- t½ (half-life) ---
    # Pattern 1: standard "half-life was X hours"
    (
        "t_half",
        r"(?:half[\-\s]*life|t\s*1\s*/?\s*2|t½)[^\d]{0,80}?"
        r"(?:" + _VERB + r"\s*)"
        r"(\d+\.?\d*)\s*(?:to\s+\d+\.?\d*\s*)?" + _TIME_UNITS,
        1,
        None,
    ),
    # Pattern 2: "elimination t½ X hours" (no verb)
    (
        "t_half",
        r"(?:elimination|terminal|plasma|mean)\s+(?:half[\-\s]*life|t\s*1\s*/?\s*2|t½)\s+"
        r"(\d+\.?\d*)\s*" + _TIME_UNITS,
        1,
        None,
    ),
    # --- Tmax ---
    (
        "tmax",
        r"(?:T\s*max|time\s+to\s+(?:peak|maximum)\s+(?:plasma\s+)?concentration)[^\d]{0,60}?"
        r"(?:" + _VERB + r"\s*)"
        r"(\d+\.?\d*)\s*(?:to\s+(\d+\.?\d*)\s*)?" + _TIME_UNITS,
        1,
        None,
    ),
    # --- Vd (volume of distribution) ---
    # Pattern 1: standard
    (
        "vd",
        r"(?:volume\s+of\s+distribution|V\s*d|V/F|Vd?ss)[^\d]{0,80}?"
        r"(?:" + _VERB + r"\s*)"
        r"(?:approximately\s+)?"
        r"(\d+\.?\d*)\s*(?:±\s*\d+\.?\d*\s*)?(?:liters?|L\b|L/kg)",
        1,
        None,
    ),
    # Pattern 2: "Vd (L) VALUE" or "Vd (L/kg) VALUE" (tabular)
    (
        "vd",
        r"(?:volume\s+of\s+distribution|V\s*d|V/F|Vd?ss)\s*\(\s*(L(?:/kg)?|liters?)\s*\)\s*(\d+\.?\d*)",
        2,
        None,
    ),
    # --- Clearance ---
    # Pattern 1: standard
    (
        "clearance",
        r"(?:(?:oral|renal|total|apparent|systemic|plasma|body)\s+)?(?:clearance|CL(?:/F)?)[^\d]{0,60}?"
        r"(?:" + _VERB + r"\s*)"
        r"(?:approximately\s+)?"
        r"(\d+\.?\d*)\s*(?:±\s*\d+\.?\d*\s*)?(?:mL/min(?:/kg)?|L/h(?:r|our)?(?:/kg)?)",
        1,
        None,
    ),
    # Pattern 2: "CL (mL/min) VALUE" (tabular)
    (
        "clearance",
        r"(?:clearance|CL(?:/F)?)\s*\(\s*(mL/min(?:/kg)?|L/h(?:r)?(?:/kg)?)\s*\)\s*(\d+\.?\d*)",
        2,
        None,
    ),
    # --- Bioavailability ---
    (
        "bioavailability",
        r"(?:(?:absolute|oral|mean)\s+)?bioavailability[^\d]{0,60}?"
        r"(?:" + _VERB + r"\s*)"
        r"(?:approximately\s+)?"
        r"(\d+\.?\d*)\s*%",
        1,
        None,
    ),
    # --- Protein binding ---
    # Pattern 1: "protein binding was X%"
    (
        "protein_binding",
        r"(?:protein\s+bind(?:ing|s)|bound\s+to\s+(?:plasma\s+)?proteins?)[^\d]{0,60}?"
        r"(?:" + _VERB + r"\s*)"
        r"(\d+\.?\d*)\s*%",
        1,
        None,
    ),
    # Pattern 2: "X% bound to (plasma) protein" (reverse pattern)
    (
        "protein_binding",
        r"(\d+\.?\d*)\s*%\s*(?:is\s+)?(?:bound\s+to\s+(?:plasma\s+)?proteins?|protein[- ]bound)",
        1,
        None,
    ),
    # Pattern 3: "plasma protein binding X%" (no verb)
    (
        "protein_binding",
        r"(?:plasma\s+)?protein\s+binding\s+(?:is\s+)?(\d+\.?\d*)\s*%",
        1,
        None,
    ),
]


def extract_pk_from_text(drug_name: str, text: str) -> list[PKExtraction]:
    """Extract PK parameters from pharmacokinetics text."""
    results = []
    clean_name = normalize_drug_name(drug_name)

    for param_name, pattern, val_group, unit_group in PK_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                value = float(match.group(val_group))
            except (ValueError, IndexError):
                continue

            # Get unit
            if unit_group is not None:
                try:
                    raw_unit = match.group(unit_group)
                    unit = raw_unit.strip() if raw_unit else ""
                except IndexError:
                    unit = ""
            else:
                # Infer unit from parameter
                if param_name == "auc":
                    # Try to find unit in nearby text
                    span_text = text[max(0, match.start() - 30) : match.end() + 30]
                    if re.search(r"mcg|µg|μg|ug", span_text, re.IGNORECASE):
                        unit = "mcg·hr/mL"
                    elif re.search(r"ng", span_text, re.IGNORECASE):
                        unit = "ng·hr/mL"
                    else:
                        unit = "unknown"
                elif param_name == "t_half":
                    span_text = text[match.start() : match.end() + 5]
                    unit = "days" if "day" in span_text.lower() else "hours"
                elif param_name == "tmax":
                    unit = "hours"
                elif param_name == "vd":
                    # Check if L or L/kg
                    span_text = text[match.start() : match.end() + 10]
                    unit = "L/kg" if "L/kg" in span_text else "L"
                elif param_name == "clearance":
                    span_text = text[match.start() : match.end() + 15]
                    if "mL/min/kg" in span_text:
                        unit = "mL/min/kg"
                    elif "L/h" in span_text or "L/hr" in span_text:
                        unit = "L/h"
                    else:
                        unit = "mL/min"
                elif param_name in ("bioavailability", "protein_binding"):
                    unit = "%"
                else:
                    unit = ""

            # Sanity checks
            if param_name == "auc" and value < 0.1:
                continue
            if param_name == "cmax" and value < 0.01:
                continue
            if param_name == "t_half" and (value < 0.1 or value > 1000):
                continue
            if param_name == "bioavailability" and (value < 1 or value > 100):
                continue
            if param_name == "protein_binding" and (value < 1 or value > 100):
                continue
            if param_name == "tmax" and (value < 0.1 or value > 72):
                continue
            if param_name == "vd" and value < 0.01:
                continue

            # Context snippet
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            context = text[start:end].replace("\n", " ").strip()

            results.append(
                PKExtraction(
                    drug_name=clean_name,
                    parameter=param_name,
                    value=value,
                    unit=unit,
                    context=context,
                )
            )

    return results


def main():
    cache_dir = "data/ml/openfda_cache/"
    files = glob.glob(os.path.join(cache_dir, "*.json"))

    all_extractions: list[PKExtraction] = []
    drugs_processed = set()

    for filepath in files:
        with open(filepath) as f:
            data = json.load(f)

        for result in data.get("results", []):
            # Get drug name
            openfda = result.get("openfda", {})
            generic_names = openfda.get("generic_name", [])
            if not generic_names:
                continue
            drug_name = generic_names[0] if isinstance(generic_names, list) else generic_names

            # Get PK text from multiple fields
            pk_parts = []
            for field in ["pharmacokinetics", "clinical_pharmacology"]:
                raw = result.get(field)
                if raw:
                    text = raw[0] if isinstance(raw, list) else raw
                    if text and len(text) > 50:
                        pk_parts.append(text)
            pk_text = "\n".join(pk_parts)

            if not pk_text or len(pk_text) < 100:
                continue

            normalized = normalize_drug_name(drug_name)
            drugs_processed.add(normalized)

            extractions = extract_pk_from_text(drug_name, pk_text)
            all_extractions.extend(extractions)

    # Deduplicate: keep first extraction per (drug, parameter)
    seen = set()
    unique = []
    for e in all_extractions:
        key = (e.drug_name, e.parameter)
        if key not in seen:
            seen.add(key)
            unique.append(e)

    # Write CSV
    out_path = "data/ml/clinical/openfda_pk_extracted.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["drug_name", "parameter", "value", "unit", "context"])
        for e in unique:
            writer.writerow([e.drug_name, e.parameter, e.value, e.unit, e.context])

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Extracted {len(unique)} PK parameters from {len(drugs_processed)} drugs")
    print(f"Output: {out_path}")
    print(f"{'=' * 60}\n")

    # Per-parameter counts
    from collections import Counter

    param_counts = Counter(e.parameter for e in unique)
    print("Parameters extracted:")
    for param, count in param_counts.most_common():
        print(f"  {param}: {count}")

    # Per-drug summary
    print("\nPer-drug breakdown:")
    drug_params = {}
    for e in unique:
        drug_params.setdefault(e.drug_name, []).append(e.parameter)
    for drug in sorted(drug_params):
        params = drug_params[drug]
        print(f"  {drug}: {', '.join(sorted(params))}")


if __name__ == "__main__":
    main()
