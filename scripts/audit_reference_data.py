#!/usr/bin/env python3
"""Phase 0 diagnostic: Audit reference data for leakage, steady-state,
pKa mismatches, and salt form issues.

Outputs JSON report to outputs/diagnostic_report_data_audit.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import csv  # noqa: E402


def load_gold_tier_drugs(ref_db_path: str) -> dict:
    """Load platinum + gold tier drugs from reference_database.json."""
    with open(ref_db_path) as f:
        db = json.load(f)
    gold = {}
    for name, info in db["drugs"].items():
        if info.get("tier") in ("platinum", "gold"):
            gold[name] = info
    return gold


def load_adme_training(adme_csv_path: str) -> list[dict]:
    """Load ADME training compounds from CSV."""
    rows = []
    with open(adme_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ── Audit 1: Data Leakage ──────────────────────────────────────────────


def audit_data_leakage(gold_drugs: dict, adme_rows: list[dict]) -> dict:
    """Check for overlap between gold tier drugs and ADME training data."""
    # Normalize gold drug names
    gold_names = {n.lower().strip().replace(" ", "").replace("-", "") for n in gold_drugs}
    # Also collect gold SMILES (non-empty)
    gold_smiles = set()
    for info in gold_drugs.values():
        s = info.get("smiles", "")
        if s and s != "N/A":
            gold_smiles.add(s.strip())

    # Normalize ADME names and SMILES
    adme_names = set()
    adme_smiles = set()
    for row in adme_rows:
        name = row.get("name", "").lower().strip().replace(" ", "").replace("-", "")
        if name:
            adme_names.add(name)
        s = row.get("smiles", "").strip()
        if s:
            adme_smiles.add(s)

    # Find overlaps by name
    name_overlap = gold_names & adme_names
    # Find overlaps by SMILES
    smiles_overlap_raw = gold_smiles & adme_smiles

    # Map back to original drug names for reporting
    overlapping_drugs = []
    seen = set()

    for orig_name, info in gold_drugs.items():
        norm = orig_name.lower().strip().replace(" ", "").replace("-", "")
        smiles = (info.get("smiles") or "").strip()
        matched_by = []
        if norm in name_overlap:
            matched_by.append("name")
        if smiles and smiles != "N/A" and smiles in smiles_overlap_raw:
            matched_by.append("smiles")
        if matched_by and norm not in seen:
            seen.add(norm)
            overlapping_drugs.append(
                {
                    "drug": orig_name,
                    "tier": info.get("tier"),
                    "matched_by": matched_by,
                }
            )

    return {
        "overlapping_drugs": sorted(overlapping_drugs, key=lambda x: x["drug"]),
        "n_overlap": len(overlapping_drugs),
        "n_gold": len(gold_drugs),
        "n_adme_training": len(adme_rows),
    }


# ── Audit 2: Single-Dose vs Steady-State ───────────────────────────────


# Known autoinducers / long-t½ drugs where steady-state != single-dose
_STEADY_STATE_RISK_DRUGS = {
    "carbamazepine": {
        "t_half_h": 36.0,
        "reason": "Autoinducer (CYP3A4); steady-state CL ~2x single-dose CL",
    },
    "warfarin": {
        "t_half_h": 40.0,
        "reason": "Very long t½; clinical PK often from multi-dose studies",
    },
    "diazepam": {
        "t_half_h": 43.0,
        "reason": "Very long t½; accumulation confounds single-dose comparison",
    },
    "phenytoin": {
        "t_half_h": 22.0,
        "reason": "Non-linear (Michaelis-Menten) kinetics; single-dose != steady-state",
    },
    "tamoxifen": {
        "t_half_h": 120.0,
        "reason": "Extremely long t½; steady-state takes weeks",
    },
    "hydroxychloroquine": {
        "t_half_h": 960.0,
        "reason": "Extremely long t½ (~40 days); FDA label Cmax likely from multi-dose",
    },
    "azithromycin": {
        "t_half_h": 68.0,
        "reason": "Long tissue t½; high Vd with tissue accumulation",
    },
    "sonidegib": {
        "t_half_h": 672.0,
        "reason": "Extremely long t½ (~28 days); PK params almost certainly steady-state",
    },
    "efavirenz": {
        "t_half_h": 52.0,
        "reason": "Long t½ + CYP2B6 autoinduction; steady-state CL differs from single-dose",
    },
    "vorasidenib": {
        "t_half_h": 240.0,
        "reason": "Very long t½ (~10 days); accumulation expected",
    },
    "entecavir": {
        "t_half_h": 128.0,
        "reason": "Long t½; reference data may be steady-state",
    },
    "isotretinoin": {
        "t_half_h": 90.0,
        "reason": "Long t½; variable absorption with food",
    },
    "valganciclovir": {
        "t_half_h": 73.0,
        "reason": "Long t½; prodrug with variable conversion",
    },
    "zonisamide": {
        "t_half_h": 63.0,
        "reason": "Long t½; extensive accumulation at steady-state",
    },
}


def audit_steady_state(gold_drugs: dict) -> list[dict]:
    """Flag drugs where reference PK data may be steady-state rather than single-dose."""
    results = []
    for name, info in sorted(gold_drugs.items()):
        pk = info.get("pk_params", {})
        thalf = pk.get("thalf_h")

        # Check explicit known risks
        norm_name = name.lower().strip()
        if norm_name in _STEADY_STATE_RISK_DRUGS:
            risk_info = _STEADY_STATE_RISK_DRUGS[norm_name]
            results.append(
                {
                    "drug": name,
                    "t_half_h": thalf if thalf else risk_info["t_half_h"],
                    "risk": "high",
                    "reason": risk_info["reason"],
                    "source": info.get("source", "unknown"),
                }
            )
        elif thalf and thalf > 24.0:
            # Any drug with t½ > 24h is at risk since sim is 24h
            results.append(
                {
                    "drug": name,
                    "t_half_h": thalf,
                    "risk": "medium",
                    "reason": f"t½ ({thalf}h) > 24h simulation window; incomplete elimination",
                    "source": info.get("source", "unknown"),
                }
            )

    return sorted(results, key=lambda x: -(x.get("t_half_h") or 0))


# ── Audit 3: pKa Mismatch ──────────────────────────────────────────────


def audit_pka_mismatches(gold_drugs: dict) -> list[dict]:
    """Predict pKa for gold tier drugs and flag mismatches vs pipeline default of 7.0."""
    from omega_pbpk.prediction.pka_predictor import predict_pka

    # Known molecule types for common drugs
    _KNOWN_TYPES = {
        # Bases (amines)
        "propranolol": "base",
        "metoprolol": "base",
        "verapamil": "base",
        "amlodipine": "base",
        "imipramine": "base",
        "clozapine": "base",
        "sertraline": "base",
        "tamoxifen": "base",
        "amphetamine": "base",
        "dextroamphetamine": "base",
        "tramadol": "base",
        "codeine": "base",
        "morphine": "base",
        "metformin": "base",
        "ciprofloxacin": "base",
        "clarithromycin": "base",
        "erythromycin": "base",
        "azithromycin": "base",
        "hydroxychloroquine": "base",
        "primaquine": "base",
        "fluconazole": "base",
        "ketoconazole": "base",
        "itraconazole": "base",
        "clopidogrel": "base",
        "methylphenidate": "base",
        "oxybutynin": "base",
        "terbinafine": "base",
        "desloratadine": "base",
        "cyclobenzaprine": "base",
        "lofexidine": "base",
        "glasdegib": "base",
        "ranolazine": "base",
        "vilazodone": "base",
        "metoclopramide": "base",
        "tamsulosin": "base",
        "dextromethorphan": "base",
        # Acids
        "warfarin": "acid",
        "ibuprofen": "acid",
        "naproxen": "acid",
        "diclofenac": "acid",
        "aspirin": "acid",
        "atorvastatin": "acid",
        "losartan": "acid",
        "gabapentin": "acid",
        "pregabalin": "acid",
        "valacyclovir": "acid",
        "amoxicillin": "acid",
        # Neutral
        "caffeine": "neutral",
        "midazolam": "neutral",
        "alprazolam": "neutral",
        "diazepam": "neutral",
        "nifedipine": "neutral",
        "carbamazepine": "neutral",
        "acetaminophen": "neutral",
        "theophylline": "neutral",
        "omeprazole": "neutral",
        "simvastatin": "neutral",
        "digoxin": "neutral",
        "celecoxib": "neutral",
        "lansoprazole": "neutral",
        "pantoprazole": "neutral",
        "zolpidem": "neutral",
    }

    results = []
    for name, info in sorted(gold_drugs.items()):
        smiles = info.get("smiles", "")
        if not smiles or smiles == "N/A":
            continue

        mol_type = _KNOWN_TYPES.get(name.lower().strip(), "neutral")

        try:
            pred = predict_pka(smiles, mol_type)
        except Exception as e:
            results.append(
                {
                    "drug": name,
                    "predicted_pka": None,
                    "default": 7.0,
                    "delta": None,
                    "molecule_type": mol_type,
                    "detected_group": None,
                    "error": str(e),
                }
            )
            continue

        if pred.pka_predicted is not None:
            delta = abs(pred.pka_predicted - 7.0)
            if delta > 1.0:
                results.append(
                    {
                        "drug": name,
                        "predicted_pka": round(pred.pka_predicted, 2),
                        "default": 7.0,
                        "delta": round(delta, 2),
                        "molecule_type": mol_type,
                        "detected_group": pred.detected_group,
                        "ionization_at_pH7": round(pred.ionization_at_pH7, 3),
                        "impact": (
                            "High — basic drug treated as neutral; Kp for lung/kidney affected"
                            if mol_type == "base" and pred.pka_predicted > 8.0
                            else "High — acidic drug treated as neutral; ionization-dependent Kp affected"
                            if mol_type == "acid" and pred.pka_predicted < 6.0
                            else "Moderate — pKa differs from default but impact on Kp may be limited"
                        ),
                    }
                )

    return sorted(results, key=lambda x: -(x.get("delta") or 0))


# ── Audit 4: Salt Form Identification ──────────────────────────────────


# Known salt forms for common drugs (from clinical practice)
_KNOWN_SALT_FORMS = {
    "amlodipine": "besylate",
    "sertraline": "HCl",
    "metformin": "HCl",
    "propranolol": "HCl",
    "verapamil": "HCl",
    "tramadol": "HCl",
    "ciprofloxacin": "HCl",
    "metoclopramide": "HCl",
    "tamsulosin": "HCl",
    "lofexidine": "HCl",
    "methylphenidate": "HCl",
    "oxybutynin": "HCl",
    "terbinafine": "HCl",
    "hydroxychloroquine": "sulfate",
    "morphine": "sulfate",
    "amphetamine": "sulfate",
    "dextroamphetamine": "sulfate",
    "codeine": "phosphate",
    "ibuprofen": "lysine (or free acid)",
    "naproxen": "sodium",
    "diclofenac": "sodium/potassium",
    "losartan": "potassium",
    "atorvastatin": "calcium",
    "clopidogrel": "bisulfate",
    "clopidogrel bisulfate": "bisulfate",
    "guanfacine er": "HCl",
    "glycopyrrolate": "bromide",
    "dasatinib": "monohydrate",
    "pazopanib": "HCl",
    "levocetirizine": "dihydrochloride",
    "valacyclovir": "HCl",
    "oseltamivir": "phosphate",
}


def audit_salt_forms(gold_drugs: dict) -> list[dict]:
    """Identify gold tier drugs commonly administered as salts."""
    results = []
    for name, info in sorted(gold_drugs.items()):
        norm = name.lower().strip()
        smiles = info.get("smiles", "")

        # Check known salt forms
        if norm in _KNOWN_SALT_FORMS:
            salt = _KNOWN_SALT_FORMS[norm]
            # Check if SMILES contains salt indicator (e.g., dot-disconnected)
            has_dot = "." in (smiles or "")
            results.append(
                {
                    "drug": name,
                    "likely_salt": salt,
                    "smiles_has_counterion": has_dot,
                    "impact": (
                        "Salt form affects solubility and potentially Cmax; "
                        "pipeline uses free-base/free-acid SMILES without salt correction"
                    ),
                }
            )
        elif smiles and "." in smiles:
            # SMILES has dot-disconnected components → likely salt/solvate
            results.append(
                {
                    "drug": name,
                    "likely_salt": "unknown (dot-disconnected SMILES)",
                    "smiles_has_counterion": True,
                    "impact": "SMILES contains counterion/solvate but salt correction not applied",
                }
            )

    return sorted(results, key=lambda x: x["drug"])


# ── Main ────────────────────────────────────────────────────────────────


def main():
    ref_db_path = PROJECT_ROOT / "data" / "clinical" / "reference_database.json"
    adme_csv_path = PROJECT_ROOT / "data" / "adme_reference.csv"
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "diagnostic_report_data_audit.json"

    print("=" * 60)
    print("Phase 0 Diagnostic: Reference Data Audit")
    print("=" * 60)

    # Load data
    gold_drugs = load_gold_tier_drugs(str(ref_db_path))
    adme_rows = load_adme_training(str(adme_csv_path))
    print(f"\nLoaded {len(gold_drugs)} gold/platinum tier drugs")
    print(f"Loaded {len(adme_rows)} ADME training compounds")

    # Audit 1: Data leakage
    print("\n── Audit 1: Data Leakage Check ──")
    leakage = audit_data_leakage(gold_drugs, adme_rows)
    print(
        f"  Overlap: {leakage['n_overlap']}/{leakage['n_gold']} gold tier drugs found in ADME training"
    )
    for d in leakage["overlapping_drugs"]:
        print(f"    - {d['drug']} ({d['tier']}) matched by: {', '.join(d['matched_by'])}")

    # Audit 2: Steady-state risk
    print("\n── Audit 2: Steady-State Risk ──")
    steady_state = audit_steady_state(gold_drugs)
    high_risk = [d for d in steady_state if d["risk"] == "high"]
    med_risk = [d for d in steady_state if d["risk"] == "medium"]
    print(f"  High risk: {len(high_risk)} drugs")
    for d in high_risk:
        print(f"    - {d['drug']} (t½={d['t_half_h']}h): {d['reason']}")
    print(f"  Medium risk: {len(med_risk)} drugs")
    for d in med_risk:
        print(f"    - {d['drug']} (t½={d['t_half_h']}h)")

    # Audit 3: pKa mismatches
    print("\n── Audit 3: pKa Mismatches (|predicted - 7.0| > 1.0) ──")
    pka_mismatches = audit_pka_mismatches(gold_drugs)
    print(f"  {len(pka_mismatches)} drugs with significant pKa mismatch")
    for d in pka_mismatches:
        if d.get("predicted_pka") is not None:
            print(
                f"    - {d['drug']}: pKa={d['predicted_pka']} (Δ={d['delta']}) "
                f"[{d['molecule_type']}, {d['detected_group']}]"
            )

    # Audit 4: Salt forms
    print("\n── Audit 4: Salt Form Identification ──")
    salt_forms = audit_salt_forms(gold_drugs)
    print(f"  {len(salt_forms)} drugs commonly administered as salts")
    for d in salt_forms:
        print(
            f"    - {d['drug']}: {d['likely_salt']} (counterion in SMILES: {d['smiles_has_counterion']})"
        )

    # Build report
    report = {
        "audit_date": "2026-03-17",
        "description": "Phase 0 diagnostic — reference data audit for leakage, steady-state, pKa, salt forms",
        "data_leakage": leakage,
        "steady_state_risk": steady_state,
        "pka_mismatches": pka_mismatches,
        "salt_form_drugs": salt_forms,
        "summary": {
            "n_gold_tier": leakage["n_gold"],
            "n_adme_training": leakage["n_adme_training"],
            "n_leakage_overlap": leakage["n_overlap"],
            "n_steady_state_high_risk": len(high_risk),
            "n_steady_state_medium_risk": len(med_risk),
            "n_pka_mismatches": len(pka_mismatches),
            "n_salt_form_drugs": len(salt_forms),
        },
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Report saved to: {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
