#!/usr/bin/env python
"""Build unified reference database from PK-DB timecourses + FDA PK parameters.

Combines two data sources into a single reference JSON:
  - PK-DB real C(t) data (platinum tier)
  - FDA PK parameters with analytical C(t) curves (gold/silver tier)

Output: data/clinical/reference_database.json
        frontend/src/data/referenceData.ts
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PKDB_PATH = ROOT / "data" / "clinical" / "pkdb_timecourses.json"
FDA_PATH = ROOT / "data" / "clinical" / "fda_pk_reference.json"
GOLD24_PATH = ROOT / "data" / "clinical" / "gold24_reference_cmax.json"
PLATINUM_PATH = ROOT / "data" / "clinical" / "platinum_reference.json"
OUTPUT_JSON = ROOT / "data" / "clinical" / "reference_database.json"
OUTPUT_TS = ROOT / "frontend" / "src" / "data" / "referenceData.ts"


# ---------------------------------------------------------------------------
# Analytical C(t) generation
# ---------------------------------------------------------------------------


def generate_ct_curve(
    cmax: float,
    thalf_h: float,
    dose_mg: float = 100.0,
    F: float = 0.7,
    n_points: int = 50,
    duration_h: float | None = None,
) -> tuple[list[float], list[float]]:
    """Generate analytical C(t) from PK parameters (1-cpt oral model)."""
    ke = 0.693 / thalf_h
    # Approximate Vd from Cmax: Vd = F * Dose / Cmax
    vd = F * dose_mg / cmax if cmax > 0 else 50.0
    # Assume ka >> ke for typical oral absorption
    ka = max(1.0, 3 * ke)
    if abs(ka - ke) < 1e-6:
        ka = ke * 1.5

    if duration_h is None:
        duration_h = min(5 * thalf_h, 120)

    t = np.linspace(0, duration_h, n_points)
    c = (F * dose_mg / vd) * (ka / (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))
    c = np.maximum(c, 0)

    return _round_list(t.tolist()), _round_list(c.tolist())


def generate_elimination_curve(
    thalf_h: float,
    c0: float = 1.0,
    n_points: int = 50,
    duration_h: float | None = None,
) -> tuple[list[float], list[float]]:
    """Generate elimination-only curve (silver tier) — C(t) = C0 * exp(-ke*t)."""
    ke = 0.693 / thalf_h
    if duration_h is None:
        duration_h = min(5 * thalf_h, 120)
    t = np.linspace(0, duration_h, n_points)
    c = c0 * np.exp(-ke * t)

    return _round_list(t.tolist()), _round_list(c.tolist())


def _round_list(vals: list[float], decimals: int = 6) -> list[float]:
    """Round list values for compact JSON."""
    return [round(v, decimals) for v in vals]


# ---------------------------------------------------------------------------
# PK-DB best curve selection
# ---------------------------------------------------------------------------


def select_best_pkdb_curve(
    studies: list[dict],
) -> dict | None:
    """Pick the best C(t) curve from PK-DB studies.

    Priority: oral route > largest n_subjects > most timepoints.
    """
    if not studies:
        return None

    def score(s: dict) -> tuple[int, int, int]:
        route_score = 2 if s.get("route") == "oral" else (0 if s.get("route") == "iv" else 1)
        n_subj = s.get("n_subjects") or 0
        n_pts = len(s.get("timepoints", []))
        return (route_score, n_subj, n_pts)

    best = max(studies, key=score)
    return best


def extract_pkdb_curve(study: dict) -> dict:
    """Extract time, concentration, and SD arrays from a PK-DB study."""
    timepoints = study.get("timepoints", [])
    time_h = [tp["time_h"] for tp in timepoints]
    conc_mg_l = [tp["mean"] for tp in timepoints]
    sd_mg_l = [tp.get("sd") for tp in timepoints]

    # Replace None SDs with 0
    sd_mg_l = [s if s is not None else 0.0 for s in sd_mg_l]

    return {
        "time_h": _round_list(time_h),
        "conc_mg_L": _round_list(conc_mg_l),
        "sd_mg_L": _round_list(sd_mg_l),
    }


# ---------------------------------------------------------------------------
# Unit conversion for FDA Cmax
# ---------------------------------------------------------------------------


def convert_cmax_to_mg_l(value: float, unit: str | None) -> float | None:
    """Convert Cmax value to mg/L."""
    if value is None or unit is None:
        return None
    unit_lower = unit.lower().strip()
    conversions = {
        "mg/l": 1.0,
        "ug/l": 1e-3,
        "ng/ml": 1e-3,
        "µg/l": 1e-3,
        "µg/ml": 1.0,
        "mg/ml": 1e3,
    }
    factor = conversions.get(unit_lower, 1.0)
    return value * factor


def convert_auc_to_mg_h_l(value: float, unit: str | None) -> float | None:
    """Convert AUC value to mg*h/L."""
    if value is None or unit is None:
        return None
    unit_lower = unit.lower().strip()
    conversions = {
        "mg*h/l": 1.0,
        "mg·h/l": 1.0,
        "ug*h/l": 1e-3,
        "ng*h/ml": 1e-3,
        "ng·h/ml": 1e-3,
        "µg*h/l": 1e-3,
        "µg·h/l": 1e-3,
        "µg*h/ml": 1.0,
        "h*mg/l": 1.0,
        "h*ug/l": 1e-3,
        "h*ng/ml": 1e-3,
    }
    factor = conversions.get(unit_lower, 1.0)
    return value * factor


# ---------------------------------------------------------------------------
# Build the database
# ---------------------------------------------------------------------------


def build_database() -> dict:
    """Build the unified reference database."""
    # Load sources
    with open(PKDB_PATH) as f:
        pkdb = json.load(f)

    with open(FDA_PATH) as f:
        fda_data = json.load(f)

    fda_drugs = fda_data["drugs"]

    # Index FDA drugs by lowercase name
    fda_by_name: dict[str, dict] = {}
    for drug in fda_drugs:
        name = drug["name"].lower().strip()
        fda_by_name[name] = drug

    drugs_output: dict[str, dict] = {}
    tier_counts = {"platinum": 0, "gold": 0, "silver": 0}

    # --- Pass 1: PK-DB drugs (platinum) ---
    for drug_name, studies in pkdb.items():
        name_lower = drug_name.lower().strip()
        best = select_best_pkdb_curve(studies)
        if best is None:
            continue

        ct_data = extract_pkdb_curve(best)

        # Get SMILES from FDA if available
        smiles = None
        pk_params: dict = {}
        fda_entry = fda_by_name.get(name_lower)
        if fda_entry:
            smiles = fda_entry.get("smiles")
            fda_pk = fda_entry.get("pk_params", {})
            cmax_val = convert_cmax_to_mg_l(fda_pk.get("cmax_value"), fda_pk.get("cmax_unit"))
            auc_val = convert_auc_to_mg_h_l(fda_pk.get("auc_value"), fda_pk.get("auc_unit"))
            pk_params = {
                "cmax_mg_L": cmax_val,
                "auc_mg_h_L": auc_val,
                "thalf_h": fda_pk.get("thalf_h"),
                "bioavailability_pct": fda_pk.get("bioavailability_pct"),
            }
            # Remove None values
            pk_params = {k: v for k, v in pk_params.items() if v is not None}

        # Build source string
        study_ref = best.get("study", "unknown")
        pmid = best.get("pmid", "")
        source = f"PK-DB ({study_ref}"
        if pmid:
            source += f", PMID: {pmid}"
        source += ")"

        entry = {
            "name": name_lower,
            "tier": "platinum",
            "source": source,
            "dose_mg": best.get("dose_mg"),
            "route": best.get("route", "unknown"),
            "pk_params": pk_params,
            "ct_curve": ct_data,
        }
        if smiles:
            entry["smiles"] = smiles

        drugs_output[name_lower] = entry
        tier_counts["platinum"] += 1

    # --- Pass 2: FDA drugs (gold / silver) ---
    for drug in fda_drugs:
        name_lower = drug["name"].lower().strip()

        # Skip if already in platinum
        if name_lower in drugs_output:
            continue

        pk = drug.get("pk_params", {})
        thalf = pk.get("thalf_h")
        cmax_raw = pk.get("cmax_value")
        cmax_unit = pk.get("cmax_unit")
        smiles = drug.get("smiles")
        dose_mg = drug.get("dose_mg")
        route = drug.get("route", "oral")

        cmax_mg_l = convert_cmax_to_mg_l(cmax_raw, cmax_unit)
        auc_mg_h_l = convert_auc_to_mg_h_l(pk.get("auc_value"), pk.get("auc_unit"))

        # Build pk_params dict
        pk_params_out: dict = {}
        if cmax_mg_l is not None:
            pk_params_out["cmax_mg_L"] = round(cmax_mg_l, 6)
        if auc_mg_h_l is not None:
            pk_params_out["auc_mg_h_L"] = round(auc_mg_h_l, 6)
        if thalf is not None:
            pk_params_out["thalf_h"] = thalf
        if pk.get("bioavailability_pct") is not None:
            pk_params_out["bioavailability_pct"] = pk["bioavailability_pct"]

        # Determine tier
        if thalf is not None and thalf > 0 and cmax_mg_l is not None and cmax_mg_l > 0:
            # Gold: can generate full C(t)
            tier = "gold"
            F = (pk.get("bioavailability_pct") or 70) / 100.0
            use_dose = dose_mg if dose_mg and dose_mg > 0 else 100.0
            time_h, conc_mg_l = generate_ct_curve(
                cmax=cmax_mg_l,
                thalf_h=thalf,
                dose_mg=use_dose,
                F=F,
            )
            ct_curve: dict | None = {
                "time_h": time_h,
                "conc_mg_L": conc_mg_l,
            }
        elif thalf is not None and thalf > 0:
            # Silver: elimination only
            tier = "silver"
            c0 = cmax_mg_l if (cmax_mg_l is not None and cmax_mg_l > 0) else 1.0
            time_h, conc_mg_l = generate_elimination_curve(thalf_h=thalf, c0=c0)
            ct_curve = {
                "time_h": time_h,
                "conc_mg_L": conc_mg_l,
            }
        else:
            # Skip — not enough data
            continue

        entry = {
            "name": name_lower,
            "tier": tier,
            "source": "FDA label (analytical model from PK params)",
            "dose_mg": dose_mg,
            "route": route,
            "pk_params": pk_params_out,
            "ct_curve": ct_curve,
        }
        if smiles:
            entry["smiles"] = smiles

        drugs_output[name_lower] = entry
        tier_counts[tier] += 1

    # --- Pass 3: Supplement dose_mg from gold24, platinum, and manual lookup ---
    dose_supplement: dict[str, float] = {}

    # Manual doses for FDA drugs where dose_mg is missing from all sources.
    # Values are the standard single-dose PK study dose from FDA labels.
    _MANUAL_DOSES: dict[str, float] = {
        "aspirin": 500.0,  # standard OTC single dose
        "benzhydrocodone": 4.08,  # Apadaz (4.08 mg benzhydrocodone component)
        "dimethyl": 240.0,  # dimethyl fumarate (Tecfidera 240 mg)
        "glycopyrrolate": 1.0,  # oral glycopyrrolate (Cuvposa 1 mg)
        "guanfacine er": 1.0,  # Intuniv 1 mg
        "lanthanum carbonate": 500.0,  # Fosrenol 500 mg
        "lofexidine": 0.18,  # Lucemyra 0.18 mg
        "naproxen oral": 500.0,  # standard single dose
        "paricalcitol": 0.004,  # Zemplar 4 mcg oral
        "pregabalin": 300.0,  # Lyrica 300 mg
        "ranolazine": 500.0,  # Ranexa 500 mg
    }
    dose_supplement.update(_MANUAL_DOSES)

    if GOLD24_PATH.exists():
        with open(GOLD24_PATH) as f:
            gold24 = json.load(f)
        for name, d in gold24.items():
            if name.startswith("_") or not isinstance(d, dict):
                continue
            if d.get("dose_mg"):
                dose_supplement[name.lower().strip()] = d["dose_mg"]
    if PLATINUM_PATH.exists():
        with open(PLATINUM_PATH) as f:
            plat = json.load(f)
        plat_drugs = plat.get("drugs", {})
        for name, d in plat_drugs.items():
            if isinstance(d, dict) and d.get("dose_mg"):
                dose_supplement[name.lower().strip()] = d["dose_mg"]

    for name, entry in drugs_output.items():
        if not entry.get("dose_mg") and name in dose_supplement:
            entry["dose_mg"] = dose_supplement[name]

    # --- Pass 4: Override/add gold-24 curated data ---
    # Gold-24 values are manually verified and take priority over auto-extracted
    # FDA/PK-DB values for Cmax, dose_mg, and other PK params.
    # Standard t½ from FDA labels for gold-24 drugs missing thalf everywhere.
    _GOLD24_THALF: dict[str, float] = {
        "warfarin": 40.0,
        "atenolol": 6.5,
        "diazepam": 43.0,
        "digoxin": 39.0,
        "furosemide": 1.5,
        "d_amphetamine": 10.0,
        "fluoxetine": 48.0,
        "nifedipine": 2.0,
        "phenytoin": 22.0,
    }

    if GOLD24_PATH.exists():
        # Get SMILES from platinum for drugs missing from FDA
        plat_smiles: dict[str, str] = {}
        if PLATINUM_PATH.exists():
            for pname, pd in plat_drugs.items():
                if isinstance(pd, dict) and pd.get("smiles"):
                    plat_smiles[pname.lower().strip()] = pd["smiles"]

        for gname, gdata in gold24.items():
            if gname.startswith("_") or not isinstance(gdata, dict):
                continue
            gname_lower = gname.lower().strip()
            gcmax = gdata.get("cmax_mg_L")
            gdose = gdata.get("dose_mg")
            gauc = gdata.get("auc_mg_h_L")
            gsource = gdata.get("source", "gold-24 curated")

            if gname_lower in drugs_output:
                entry = drugs_output[gname_lower]
                # Override dose_mg with gold-24 curated value
                if gdose:
                    entry["dose_mg"] = gdose
                # Override pk_params with curated values
                pk = entry.get("pk_params", {})
                if gcmax:
                    pk["cmax_mg_L"] = gcmax
                if gauc:
                    pk["auc_mg_h_L"] = gauc
                # Add thalf from lookup if missing
                if not pk.get("thalf_h") and gname_lower in _GOLD24_THALF:
                    pk["thalf_h"] = _GOLD24_THALF[gname_lower]
                entry["pk_params"] = pk
                # Add SMILES from platinum if missing
                if not entry.get("smiles") and gname_lower in plat_smiles:
                    entry["smiles"] = plat_smiles[gname_lower]
                # Regenerate C(t) curve if we now have Cmax + thalf
                thalf = pk.get("thalf_h")
                if gcmax and thalf and thalf > 0:
                    use_dose = gdose if gdose else 100.0
                    bio = (
                        pk.get("bioavailability_pct", 70)
                        if isinstance(pk.get("bioavailability_pct"), (int, float))
                        else 70
                    )
                    F = bio / 100.0
                    time_h, conc_mg_l = generate_ct_curve(
                        cmax=gcmax,
                        thalf_h=thalf,
                        dose_mg=use_dose,
                        F=F,
                    )
                    entry["ct_curve"] = {
                        "time_h": time_h,
                        "conc_mg_L": conc_mg_l,
                    }
                    if entry["tier"] == "silver":
                        entry["tier"] = "gold"
                        tier_counts["silver"] -= 1
                        tier_counts["gold"] += 1
            else:
                # Drug not in refdb at all — add it with platinum SMILES
                smiles = plat_smiles.get(gname_lower)
                if not smiles:
                    continue  # Can't add without SMILES

                pk_params_new: dict = {}
                if gcmax:
                    pk_params_new["cmax_mg_L"] = gcmax
                if gauc:
                    pk_params_new["auc_mg_h_L"] = gauc

                # Get thalf from lookup
                thalf_new = _GOLD24_THALF.get(gname_lower)
                if thalf_new:
                    pk_params_new["thalf_h"] = thalf_new

                # Generate C(t) curve
                ct_curve_new = None
                tier_new = "gold"
                if gcmax and thalf_new and thalf_new > 0:
                    use_dose = gdose if gdose else 100.0
                    time_h, conc_mg_l = generate_ct_curve(
                        cmax=gcmax,
                        thalf_h=thalf_new,
                        dose_mg=use_dose,
                    )
                    ct_curve_new = {"time_h": time_h, "conc_mg_L": conc_mg_l}

                new_entry: dict = {
                    "name": gname_lower,
                    "smiles": smiles,
                    "tier": tier_new,
                    "source": f"Gold-24 curated ({gsource})",
                    "dose_mg": gdose,
                    "route": "oral",
                    "pk_params": pk_params_new,
                }
                if ct_curve_new:
                    new_entry["ct_curve"] = ct_curve_new
                drugs_output[gname_lower] = new_entry
                tier_counts[tier_new] += 1

    # Sort by name
    drugs_sorted = dict(sorted(drugs_output.items()))

    # Build output
    output = {
        "metadata": {
            "created": str(date.today()),
            "n_drugs": len(drugs_sorted),
            "tiers": tier_counts,
        },
        "drugs": drugs_sorted,
    }

    return output


# ---------------------------------------------------------------------------
# TypeScript export
# ---------------------------------------------------------------------------


def _json_value(v: object) -> str:
    """Format a value for TypeScript/JSON output."""
    if v is None:
        return "undefined"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        # Escape single quotes and backslashes
        escaped = v.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return "undefined"
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, list):
        items = ", ".join(_json_value(x) for x in v)
        return f"[{items}]"
    if isinstance(v, dict):
        items = ", ".join(f"{k}: {_json_value(val)}" for k, val in v.items())
        return f"{{ {items} }}"
    return str(v)


def generate_typescript(db: dict) -> str:
    """Generate TypeScript file with reference database."""
    lines = [
        "// Auto-generated by scripts/build_reference_database.py",
        "// Do not edit manually.",
        "",
        "export interface ReferenceData {",
        "  name: string;",
        "  smiles: string;",
        "  tier: 'platinum' | 'gold' | 'silver';",
        "  source: string;",
        "  dose_mg?: number;",
        "  pk_params: {",
        "    cmax_mg_L?: number;",
        "    auc_mg_h_L?: number;",
        "    thalf_h?: number;",
        "  };",
        "  ct_curve?: {",
        "    time_h: number[];",
        "    conc_mg_L: number[];",
        "    sd_mg_L?: number[];",
        "  };",
        "}",
        "",
        "export const referenceDatabase: Record<string, ReferenceData> = {",
    ]

    drugs = db["drugs"]
    drug_names = sorted(drugs.keys())
    included = 0

    for name in drug_names:
        drug = drugs[name]
        smiles = drug.get("smiles")
        if not smiles:
            continue  # Skip drugs without SMILES for frontend

        tier = drug["tier"]
        source = drug["source"]
        pk_params = drug.get("pk_params", {})
        ct_curve = drug.get("ct_curve")

        # Build pk_params TS object
        pk_parts = []
        if "cmax_mg_L" in pk_params:
            pk_parts.append(f"cmax_mg_L: {pk_params['cmax_mg_L']}")
        if "auc_mg_h_L" in pk_params:
            pk_parts.append(f"auc_mg_h_L: {pk_params['auc_mg_h_L']}")
        if "thalf_h" in pk_params:
            pk_parts.append(f"thalf_h: {pk_params['thalf_h']}")
        pk_str = "{ " + ", ".join(pk_parts) + " }"

        # Build ct_curve TS object
        ct_str = "undefined"
        if ct_curve:
            ct_parts = [
                f"time_h: {json.dumps(ct_curve['time_h'])}",
                f"conc_mg_L: {json.dumps(ct_curve['conc_mg_L'])}",
            ]
            if "sd_mg_L" in ct_curve:
                ct_parts.append(f"sd_mg_L: {json.dumps(ct_curve['sd_mg_L'])}")
            ct_str = "{\n      " + ",\n      ".join(ct_parts) + ",\n    }"

        # Escape the key if it contains special chars
        key = json.dumps(name)

        dose_mg = drug.get("dose_mg")

        lines.append(f"  {key}: {{")
        lines.append(f"    name: {json.dumps(name)},")
        lines.append(f"    smiles: {json.dumps(smiles)},")
        lines.append(f"    tier: {json.dumps(tier)},")
        lines.append(f"    source: {json.dumps(source)},")
        if dose_mg is not None:
            lines.append(f"    dose_mg: {dose_mg},")
        lines.append(f"    pk_params: {pk_str},")
        if ct_str != "undefined":
            lines.append(f"    ct_curve: {ct_str},")
        lines.append("  },")

        included += 1

    lines.append("};")
    lines.append("")

    print(f"  TypeScript: {included} drugs with SMILES included")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Building unified reference database...")
    print(f"  PK-DB source: {PKDB_PATH}")
    print(f"  FDA source:   {FDA_PATH}")

    db = build_database()

    meta = db["metadata"]
    tiers = meta["tiers"]
    print("\nResults:")
    print(f"  Total drugs: {meta['n_drugs']}")
    print(f"  Platinum (real C(t)):     {tiers['platinum']}")
    print(f"  Gold (Cmax + t_half):     {tiers['gold']}")
    print(f"  Silver (t_half only):     {tiers['silver']}")

    # Write JSON
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(db, f, indent=2)
    print(f"\n  JSON output: {OUTPUT_JSON}")

    # Write TypeScript
    ts_content = generate_typescript(db)
    OUTPUT_TS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TS, "w") as f:
        f.write(ts_content)
    print(f"  TS output:   {OUTPUT_TS}")

    print("\nDone.")


if __name__ == "__main__":
    main()
