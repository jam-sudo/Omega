"""
Phase 588 — Predict optimal formulation type based on drug physicochemical properties.
"""

from __future__ import annotations

import math


def recommend_formulation(
    logP: float,
    pka: float,
    drug_type: str,
    dose_mg: float,
    solubility_mg_mL: float,
    t_half_h: float,
) -> dict:
    """
    Rule-based formulation recommendation.

    Parameters
    ----------
    logP              : octanol-water partition coefficient
    pka               : acid/base dissociation constant
    drug_type         : 'acid', 'base', or 'neutral'
    dose_mg           : dose in milligrams
    solubility_mg_mL  : aqueous solubility (mg/mL)
    t_half_h          : elimination half-life (h)

    Returns
    -------
    dict with keys: recommended_formulation, rationale, alternative_formulations,
                    bioavailability_concern, salt_form_suggested
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive.")
    if t_half_h <= 0:
        raise ValueError("t_half_h must be positive.")
    if drug_type not in ("acid", "base", "neutral"):
        raise ValueError("drug_type must be 'acid', 'base', or 'neutral'.")

    alternatives: list[str] = []

    # Priority rules (evaluated in order)
    if t_half_h > 12 and dose_mg < 100:
        recommended = "immediate_release"
        rationale = (
            f"Long half-life ({t_half_h:.1f} h) with low dose ({dose_mg:.0f} mg) "
            "means standard IR tablet is sufficient."
        )
        alternatives = ["extended_release"]
    elif t_half_h < 4 and dose_mg <= 200:
        recommended = "extended_release"
        rationale = (
            f"Short half-life ({t_half_h:.1f} h) requires extended release "
            "to maintain therapeutic concentrations."
        )
        alternatives = ["immediate_release", "lipid_based"]
    elif solubility_mg_mL < 0.1 and logP > 2:
        recommended = "lipid_based"
        rationale = (
            f"Poor aqueous solubility ({solubility_mg_mL:.3f} mg/mL) with "
            f"high logP ({logP:.1f}) indicates lipid-based system (SEDDS/SMEDDS)."
        )
        alternatives = ["nanoparticle", "immediate_release"]
    elif logP < 0 and solubility_mg_mL > 10:
        recommended = "solution"
        rationale = (
            f"Highly hydrophilic (logP={logP:.1f}) with good solubility "
            f"({solubility_mg_mL:.1f} mg/mL) — oral solution is optimal."
        )
        alternatives = ["immediate_release"]
    else:
        recommended = "immediate_release"
        rationale = "Standard immediate-release formulation is appropriate."
        alternatives = ["extended_release", "nanoparticle"]

    # Add lipid_based to alternatives if not already present and not recommended
    if (
        solubility_mg_mL < 1.0
        and recommended not in ("lipid_based",)
        and "lipid_based" not in alternatives
    ):
        alternatives.append("lipid_based")

    bioavailability_concern = solubility_mg_mL < 1.0

    # Salt form suggestion for ionizable drugs
    salt_form_suggested = drug_type in ("acid", "base")

    return {
        "recommended_formulation": recommended,
        "rationale": rationale,
        "alternative_formulations": alternatives,
        "bioavailability_concern": bioavailability_concern,
        "salt_form_suggested": salt_form_suggested,
    }


def _standard_fa(peff_cm_s: float) -> float:
    """Fraction absorbed from Peff using intestinal transit model."""
    # ka = 2 * Peff * 3600 / radius_cm (radius = 1.75 cm)
    ka = 2.0 * peff_cm_s * 3600.0 / 1.75
    # fa = 1 - exp(-ka * transit_time_h) using 3.5 h small intestinal transit
    transit_h = 3.5
    fa = 1.0 - math.exp(-ka * transit_h)
    return max(0.0, min(1.0, fa))


def formulation_bioavailability_estimate(
    formulation: str,
    logP: float,
    solubility_mg_mL: float,
    dose_mg: float,
    peff_cm_s: float,
) -> dict:
    """
    Estimate fraction absorbed (fa) for a given formulation.

    Parameters
    ----------
    formulation      : one of 'solution', 'immediate_release', 'lipid_based',
                       'extended_release', 'nanoparticle'
    logP             : octanol-water partition coefficient
    solubility_mg_mL : aqueous solubility (mg/mL)
    dose_mg          : dose (mg)
    peff_cm_s        : effective intestinal permeability (cm/s)

    Returns
    -------
    dict with keys: formulation, fa_predicted, note
    """
    valid_formulations = {
        "solution",
        "immediate_release",
        "lipid_based",
        "extended_release",
        "nanoparticle",
    }
    if formulation not in valid_formulations:
        raise ValueError(
            f"Invalid formulation '{formulation}'. Must be one of {sorted(valid_formulations)}."
        )

    # Volume of fluid for dissolution (250 mL)
    volume_mL = 250.0
    fa_std = _standard_fa(peff_cm_s)

    if formulation == "solution":
        # If dose can be dissolved, fa = 1.0; otherwise solubility-limited
        max_dissolved = volume_mL * solubility_mg_mL
        if dose_mg <= max_dissolved:
            fa = 1.0
            note = "Fully dissolved in GI fluid — complete absorption expected."
        else:
            fa = min(1.0, max_dissolved / dose_mg)
            note = "Dose exceeds solubility; absorption is solubility-limited."

    elif formulation == "immediate_release":
        # Dose number Do = dose / (solubility * 250 mL)
        dose_number = dose_mg / (solubility_mg_mL * volume_mL)
        if dose_number <= 1.0:
            fa = fa_std
        else:
            # High Do: solubility-limited absorption
            fa = fa_std / dose_number
        fa = max(0.0, min(1.0, fa))
        note = f"IR: Dose number Do={dose_number:.2f}."

    elif formulation == "lipid_based":
        if logP > 2:
            fa = min(1.0, 0.8 + logP * 0.04)
        else:
            fa = 0.6
        note = "Lipid-based formulation enhances solubilization of lipophilic drug."

    elif formulation == "extended_release":
        # Absorption window limitation: 70% of standard fa
        fa = 0.7 * fa_std
        fa = max(0.0, min(1.0, fa))
        note = "ER: absorption window limits total fa to ~70% of standard."

    else:  # nanoparticle
        fa = min(1.0, fa_std * 1.3)
        note = "Nanoparticle formulation increases surface area and dissolution rate."

    return {
        "formulation": formulation,
        "fa_predicted": fa,
        "note": note,
    }


def compare_formulations(
    drug_name: str,
    logP: float,
    solubility_mg_mL: float,
    dose_mg: float,
    peff_cm_s: float,
    t_half_h: float,
) -> list[dict]:
    """
    Evaluate multiple formulation strategies and rank by predicted fa.

    Parameters
    ----------
    drug_name        : name of the drug (used in output only)
    logP             : octanol-water partition coefficient
    solubility_mg_mL : aqueous solubility (mg/mL)
    dose_mg          : dose (mg)
    peff_cm_s        : effective intestinal permeability (cm/s)
    t_half_h         : elimination half-life (h)

    Returns
    -------
    List of dicts (sorted by fa_predicted descending), each with:
        formulation, fa_predicted, relative_advantage_vs_IR
    """
    formulations = [
        "solution",
        "immediate_release",
        "lipid_based",
        "extended_release",
        "nanoparticle",
    ]

    results = []
    for f in formulations:
        est = formulation_bioavailability_estimate(
            formulation=f,
            logP=logP,
            solubility_mg_mL=solubility_mg_mL,
            dose_mg=dose_mg,
            peff_cm_s=peff_cm_s,
        )
        results.append(est)

    # Compute IR fa for relative advantage
    ir_fa = next(r["fa_predicted"] for r in results if r["formulation"] == "immediate_release")

    output = []
    for r in results:
        fa = r["fa_predicted"]
        if ir_fa > 0:
            rel = (fa - ir_fa) / ir_fa
        else:
            rel = 0.0
        output.append(
            {
                "formulation": r["formulation"],
                "fa_predicted": fa,
                "relative_advantage_vs_IR": rel,
            }
        )

    output.sort(key=lambda x: x["fa_predicted"], reverse=True)
    return output
