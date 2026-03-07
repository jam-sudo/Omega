"""Nephrotoxicity (kidney toxicity) risk assessment based on physicochemical and mechanistic properties."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "NephrotoxicityResult",
    "assess_nephrotoxicity_risk",
    "screen_nephrotoxicity",
    "estimate_renal_excretion_fraction",
]


@dataclass(frozen=True)
class NephrotoxicityResult:
    """Result from nephrotoxicity risk assessment."""

    compound_name: str
    nephrotoxicity_score: float
    risk_class: str
    c_kidney_uM: float
    renal_safety_margin: float
    primary_mechanism: str
    requires_renal_monitoring: bool
    notes: str


def _classify_risk(score: float) -> str:
    """Classify nephrotoxicity risk from score (0-100)."""
    if score <= 10:
        return "minimal"
    elif score <= 30:
        return "low"
    elif score <= 60:
        return "moderate"
    elif score <= 80:
        return "high"
    else:
        return "very_high"


def _determine_primary_mechanism(
    is_nephrotoxin_class: bool,
    n_reactive_metabolite_alerts: int,
    cl_renal_L_per_h: float | None,
    pka_base: float | None,
    logP: float,
    fup: float,
) -> str:
    """Determine the most likely nephrotoxicity mechanism."""
    if is_nephrotoxin_class:
        return "direct_nephrotoxicity"
    if n_reactive_metabolite_alerts > 0:
        return "reactive_metabolite_toxicity"
    if cl_renal_L_per_h is not None and cl_renal_L_per_h > 10:
        return "tubular_secretion_accumulation"
    if pka_base is not None and pka_base > 9:
        return "tubular_ion_trapping"
    if logP > 4:
        return "tubular_lipid_accumulation"
    if fup < 0.01:
        return "concentration_after_filtration"
    return "unknown"


def assess_nephrotoxicity_risk(
    compound_name: str,
    logP: float,
    mw: float,
    pka_base: float | None = None,
    cl_renal_L_per_h: float | None = None,
    fup: float = 0.1,
    is_nephrotoxin_class: bool = False,
    n_reactive_metabolite_alerts: int = 0,
    plasma_conc_uM: float = 1.0,
) -> NephrotoxicityResult:
    """Assess nephrotoxicity risk based on physicochemical and mechanistic properties.

    Parameters
    ----------
    compound_name:
        Name of the compound.
    logP:
        Lipophilicity.
    mw:
        Molecular weight (g/mol).
    pka_base:
        Basic pKa (if applicable).
    cl_renal_L_per_h:
        Renal clearance in L/h (optional).
    fup:
        Fraction unbound in plasma (0-1).
    is_nephrotoxin_class:
        True if compound belongs to known nephrotoxin class (aminoglycosides, NSAIDs, contrast, cisplatin).
    n_reactive_metabolite_alerts:
        Number of reactive metabolite structural alerts.
    plasma_conc_uM:
        Plasma concentration in uM.

    Returns
    -------
    NephrotoxicityResult
    """
    if fup <= 0 or fup > 1:
        raise ValueError(f"fup must be in (0, 1], got {fup}")
    if plasma_conc_uM < 0:
        raise ValueError(f"plasma_conc_uM must be >= 0, got {plasma_conc_uM}")
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")

    # 1. Renal concentration factor
    concentration_factor = 300.0  # typical urine/plasma ratio
    c_kidney_uM = plasma_conc_uM * fup * concentration_factor

    # Basic drugs undergo additional ionic trapping in acidic urine
    if pka_base is not None and pka_base > 7:
        c_kidney_uM *= 2.0

    # 2. Tubular secretion risk factor
    secretion_risk_factor = 1.0
    if cl_renal_L_per_h is not None:
        secretion_risk_factor = min(3.0, cl_renal_L_per_h / 5.0)

    # 3. Build direct toxicity score
    score = 0.0

    if is_nephrotoxin_class:
        score += 40.0

    rm_contribution = min(30.0, n_reactive_metabolite_alerts * 15.0)
    score += rm_contribution

    if pka_base is not None and pka_base > 9:
        score += 10.0

    if logP > 4:
        score += 10.0

    if cl_renal_L_per_h is not None and cl_renal_L_per_h > 10:
        score += 15.0

    if fup < 0.01:
        score += 10.0

    # Apply secretion risk modulation (scale but don't exceed 100)
    score = score * secretion_risk_factor

    # Clamp to [0, 100]
    score = max(0.0, min(100.0, score))

    risk_class = _classify_risk(score)
    requires_renal_monitoring = score > 30

    # Renal safety margin: 100 / max(1, score)
    renal_safety_margin = 100.0 / max(1.0, score)

    primary_mechanism = _determine_primary_mechanism(
        is_nephrotoxin_class=is_nephrotoxin_class,
        n_reactive_metabolite_alerts=n_reactive_metabolite_alerts,
        cl_renal_L_per_h=cl_renal_L_per_h,
        pka_base=pka_base,
        logP=logP,
        fup=fup,
    )

    # Build notes
    note_parts = [
        f"Estimated tubular concentration: {c_kidney_uM:.2f} uM ({concentration_factor:.0f}x plasma)."
    ]
    if is_nephrotoxin_class:
        note_parts.append("Known nephrotoxin class: direct renal toxicity risk.")
    if n_reactive_metabolite_alerts > 0:
        note_parts.append(f"{n_reactive_metabolite_alerts} reactive metabolite alert(s) detected.")
    if pka_base is not None and pka_base > 7:
        note_parts.append(f"Basic drug (pKa={pka_base:.1f}): enhanced tubular ion trapping.")
    if logP > 4:
        note_parts.append(f"High logP ({logP:.1f}): risk of tubular lipid accumulation.")
    if cl_renal_L_per_h is not None and cl_renal_L_per_h > 5:
        note_parts.append(f"Active renal secretion (CLr={cl_renal_L_per_h:.1f} L/h).")
    if requires_renal_monitoring:
        note_parts.append("Renal function monitoring recommended.")
    notes = " ".join(note_parts)

    return NephrotoxicityResult(
        compound_name=compound_name,
        nephrotoxicity_score=round(score, 4),
        risk_class=risk_class,
        c_kidney_uM=round(c_kidney_uM, 6),
        renal_safety_margin=round(renal_safety_margin, 6),
        primary_mechanism=primary_mechanism,
        requires_renal_monitoring=requires_renal_monitoring,
        notes=notes,
    )


def screen_nephrotoxicity(compounds: list[dict]) -> list[NephrotoxicityResult]:
    """Screen multiple compounds for nephrotoxicity risk, sorted by score descending.

    Each compound dict must have keys: compound_name, logP, mw.
    Optional keys: pka_base, cl_renal_L_per_h, fup, is_nephrotoxin_class,
    n_reactive_metabolite_alerts, plasma_conc_uM.
    """
    results = []
    for cpd in compounds:
        result = assess_nephrotoxicity_risk(
            compound_name=cpd["compound_name"],
            logP=cpd["logP"],
            mw=cpd["mw"],
            pka_base=cpd.get("pka_base"),
            cl_renal_L_per_h=cpd.get("cl_renal_L_per_h"),
            fup=cpd.get("fup", 0.1),
            is_nephrotoxin_class=cpd.get("is_nephrotoxin_class", False),
            n_reactive_metabolite_alerts=cpd.get("n_reactive_metabolite_alerts", 0),
            plasma_conc_uM=cpd.get("plasma_conc_uM", 1.0),
        )
        results.append(result)

    results.sort(key=lambda r: r.nephrotoxicity_score, reverse=True)
    return results


def estimate_renal_excretion_fraction(cl_renal_L_per_h: float, cl_total_L_per_h: float) -> float:
    """Estimate renal excretion fraction as fe = cl_renal / cl_total.

    Parameters
    ----------
    cl_renal_L_per_h:
        Renal clearance (L/h).
    cl_total_L_per_h:
        Total clearance (L/h).

    Returns
    -------
    float
        Fraction of drug eliminated by the kidney (0-1).
    """
    if cl_total_L_per_h <= 0:
        raise ValueError(f"cl_total_L_per_h must be > 0, got {cl_total_L_per_h}")
    return cl_renal_L_per_h / cl_total_L_per_h


# Suppress unused import warning for math (used implicitly via type hints in docstrings)
_ = math.inf
