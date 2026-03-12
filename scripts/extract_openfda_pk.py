#!/usr/bin/env python3
"""Extract PK parameters from OpenFDA drug label pharmacokinetics text.

Extracts: Cmax, AUC, t½, Tmax, Vd, CL, bioavailability, protein binding
from cached OpenFDA label JSON files.

Output: data/ml/clinical/openfda_pk_extracted.csv
"""

import json
import glob
import os
import re
import csv
from dataclasses import dataclass, field
from typing import Optional


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
        " hydrochloride", " hcl", " sulfate", " sodium", " potassium",
        " besylate", " calcium", " tartrate", " succinate", " mesylate",
        " phosphate", " acetate", " fumarate", " maleate", " citrate",
        " ophthalmic solution", " extended-release capsules",
        " er tablets", " tablets", " capsules",
    ]:
        name = name.replace(suffix, "")
    return name.strip()


# Patterns for PK parameter extraction
# Each pattern: (parameter_name, regex, value_group, unit_group)
PK_PATTERNS = [
    # Cmax
    (
        "cmax",
        r"(?:C\s*max|peak\s+(?:plasma\s+)?concentration)[^\d]{0,60}?"
        r"(?:(?:was|of|is|approximately|about|~|=)\s*)"
        r"(\d+\.?\d*)\s*(ng/mL|µg/mL|mg/L|µM|nM|mcg/mL|μg/mL)",
        1, 2,
    ),
    # AUC
    (
        "auc",
        r"(?:AUC(?:\s*0\s*[-–]\s*(?:∞|inf|last|24h?r?|12h?r?))?|area\s+under\s+the\s+(?:plasma\s+)?(?:concentration|curve))[^\d]{0,80}?"
        r"(?:(?:was|of|is|approximately|about|~|=)\s*)"
        r"(\d+\.?\d*)\s*(ng[·•\*\s]h(?:r|our)?s?/mL|µg[·•\*\s]h(?:r|our)?s?/mL|mg[·•\*\s]h(?:r|our)?s?/L|µM[·•\*\s]h(?:r|our)?s?|mcg[·•\*\s]h(?:r|our)?s?/mL|h[·•\*\s](?:ng|µg|mcg)/mL|ng/mL[·•\*\s]h(?:r|our)?s?|µg/mL[·•\*\s]h(?:r|our)?s?)",
        1, 2,
    ),
    # t½ (half-life)
    (
        "t_half",
        r"(?:half[\-\s]*life|t\s*1\s*/?\s*2|t½)[^\d]{0,80}?"
        r"(?:(?:was|of|is|approximately|about|~|=|ranges?\s+from)\s*)"
        r"(\d+\.?\d*)\s*(?:to\s+\d+\.?\d*\s*)?(?:hours?|hrs?|h\b|days?)",
        1, None,
    ),
    # Tmax
    (
        "tmax",
        r"(?:T\s*max|time\s+to\s+(?:peak|maximum)\s+(?:plasma\s+)?concentration)[^\d]{0,60}?"
        r"(?:(?:was|of|is|approximately|about|~|=|occurring)\s*)"
        r"(\d+\.?\d*)\s*(?:to\s+(\d+\.?\d*)\s*)?(?:hours?|hrs?|h\b)",
        1, None,
    ),
    # Vd (volume of distribution)
    (
        "vd",
        r"(?:volume\s+of\s+distribution|V\s*d|V/F|Vd?ss)[^\d]{0,80}?"
        r"(?:(?:was|of|is|approximately|about|~|=|averaged)\s*)"
        r"(\d+\.?\d*)\s*(?:±\s*\d+\.?\d*\s*)?(?:liters?|L\b|L/kg)",
        1, None,
    ),
    # Clearance
    (
        "clearance",
        r"(?:(?:oral|renal|total|apparent|systemic)\s+)?(?:clearance|CL(?:/F)?)[^\d]{0,60}?"
        r"(?:(?:was|of|is|approximately|about|~|=)\s*)"
        r"(\d+\.?\d*)\s*(?:±\s*\d+\.?\d*\s*)?(?:mL/min|L/hr?|L/h|mL/min/kg|L/h/kg)",
        1, None,
    ),
    # Bioavailability
    (
        "bioavailability",
        r"(?:(?:absolute|oral)\s+)?bioavailability[^\d]{0,60}?"
        r"(?:(?:was|of|is|approximately|about|~|=)\s*)"
        r"(\d+\.?\d*)\s*%",
        1, None,
    ),
    # Protein binding
    (
        "protein_binding",
        r"(?:protein\s+bind(?:ing|s)|bound\s+to\s+(?:plasma\s+)?proteins?)[^\d]{0,60}?"
        r"(?:(?:was|of|is|approximately|about|~|=)\s*)"
        r"(\d+\.?\d*)\s*%",
        1, None,
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
                    unit = match.group(unit_group)
                except IndexError:
                    unit = ""
            else:
                # Infer unit from parameter
                if param_name == "t_half":
                    unit = "hours"
                elif param_name == "tmax":
                    unit = "hours"
                elif param_name == "vd":
                    # Check if L or L/kg
                    span_text = text[match.start():match.end() + 10]
                    unit = "L/kg" if "L/kg" in span_text else "L"
                elif param_name == "clearance":
                    span_text = text[match.start():match.end() + 15]
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

            results.append(PKExtraction(
                drug_name=clean_name,
                parameter=param_name,
                value=value,
                unit=unit,
                context=context,
            ))

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

            # Get PK text
            pk_text = result.get("pharmacokinetics", [""])[0] if isinstance(
                result.get("pharmacokinetics"), list
            ) else result.get("pharmacokinetics", "")

            if not pk_text or len(pk_text) < 100:
                continue

            normalized = normalize_drug_name(drug_name)
            if normalized in drugs_processed:
                continue
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
    print(f"\n{'='*60}")
    print(f"Extracted {len(unique)} PK parameters from {len(drugs_processed)} drugs")
    print(f"Output: {out_path}")
    print(f"{'='*60}\n")

    # Per-parameter counts
    from collections import Counter
    param_counts = Counter(e.parameter for e in unique)
    print("Parameters extracted:")
    for param, count in param_counts.most_common():
        print(f"  {param}: {count}")

    # Per-drug summary
    print(f"\nPer-drug breakdown:")
    drug_params = {}
    for e in unique:
        drug_params.setdefault(e.drug_name, []).append(e.parameter)
    for drug in sorted(drug_params):
        params = drug_params[drug]
        print(f"  {drug}: {', '.join(sorted(params))}")


if __name__ == "__main__":
    main()
