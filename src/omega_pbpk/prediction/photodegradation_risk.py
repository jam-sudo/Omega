"""Phase 763 — Drug Photodegradation Risk Prediction.

Predicts photodegradation risk from structural chromophore features.
ICH Q1B photostability conditions: 1.2 million lux-hours total exposure.
First-order photodegradation: C(t) = C0 * exp(-k_photo * I * t)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PhotodegradationResult:
    """Result of photodegradation risk prediction."""

    compound_name: str
    k_photo_per_h_per_klux: float
    fraction_remaining_24h_50klux: float
    fraction_remaining_1h_5klux: float
    t_half_50klux_h: float
    photostability_category: str  # "stable", "moderate", "labile"
    ich_q1b_compliant: bool
    packaging_recommendation: str
    notes: str


def _compute_k_photo(
    has_conjugated_system: bool,
    has_nitro_group: bool,
    has_carbonyl_extended: bool,
    has_aromatic_amine: bool,
    logp: float,
) -> float:
    """Compute photodegradation rate constant k_photo (per h per klux).

    Values are calibrated so that:
      - No chromophores → fraction_remaining_24h_50klux > 0.90 (stable)
      - 2+ chromophores → fraction_remaining_24h_50klux < 0.70 (labile)
    Total ICH Q1B exposure: 50 klux × 24 h = 1200 klux·h
    """
    base_k = 0.00005  # baseline: exp(-0.00005*1200) = 0.942 → stable
    if has_conjugated_system:
        base_k += 0.0003  # exp(-0.00035*1200)=0.657 alone → labile
    if has_nitro_group:
        base_k += 0.0002  # exp(-0.00025*1200)=0.741 alone → moderate
    if has_carbonyl_extended:
        base_k += 0.0001  # exp(-0.00015*1200)=0.835 alone → moderate
    if has_aromatic_amine:
        base_k += 0.00025  # exp(-0.00030*1200)=0.698 alone → labile
    if logp > 3.0:
        base_k *= 1.2  # lipophilic → faster solid-state degradation
    return base_k


def _photostability_category(fraction_remaining_24h_50klux: float) -> str:
    """Determine photostability category from fraction remaining at 24h/50klux."""
    if fraction_remaining_24h_50klux > 0.90:
        return "stable"
    elif fraction_remaining_24h_50klux >= 0.70:
        return "moderate"
    else:
        return "labile"


def _packaging_recommendation(category: str, ich_q1b_compliant: bool) -> str:
    """Recommend packaging based on photostability."""
    if category == "labile" or not ich_q1b_compliant:
        return "foil overwrap"
    elif category == "moderate":
        return "amber bottle"
    else:
        return "standard packaging"


def _build_notes(
    category: str,
    ich_q1b_compliant: bool,
    has_conjugated_system: bool,
    has_nitro_group: bool,
    has_carbonyl_extended: bool,
    has_aromatic_amine: bool,
    logp: float,
    k_photo: float,
) -> str:
    """Build human-readable notes string."""
    flags: list[str] = []
    if has_conjugated_system:
        flags.append("conjugated system")
    if has_nitro_group:
        flags.append("nitro group")
    if has_carbonyl_extended:
        flags.append("alpha-beta-unsaturated carbonyl")
    if has_aromatic_amine:
        flags.append("aromatic amine")
    if logp > 3.0:
        flags.append("high logP (>3)")

    chromophore_str = (
        "Chromophores detected: " + ", ".join(flags) + "."
        if flags
        else "No significant chromophores detected."
    )

    compliance_str = (
        "ICH Q1B compliant (minimal degradation)."
        if ich_q1b_compliant
        else "ICH Q1B non-compliant: significant degradation expected."
    )

    return (
        f"k_photo={k_photo:.5f} per h per klux. "
        f"Photostability: {category}. "
        f"{chromophore_str} "
        f"{compliance_str}"
    )


def predict_photodegradation(
    compound_name: str,
    has_conjugated_system: bool = False,
    has_nitro_group: bool = False,
    has_carbonyl_extended: bool = False,
    has_aromatic_amine: bool = False,
    logp: float = 2.0,
    dose_mg: float = 100.0,
) -> PhotodegradationResult:
    """Predict photodegradation risk for a drug compound.

    Parameters
    ----------
    compound_name:
        Name of the drug compound.
    has_conjugated_system:
        Whether the compound has a conjugated pi system.
    has_nitro_group:
        Whether the compound has a nitro group (-NO2).
    has_carbonyl_extended:
        Whether the compound has an alpha,beta-unsaturated carbonyl.
    has_aromatic_amine:
        Whether the compound has an aromatic amine.
    logp:
        Calculated logP (octanol-water partition coefficient).
    dose_mg:
        Dose in milligrams.

    Returns
    -------
    PhotodegradationResult
    """
    k_photo = _compute_k_photo(
        has_conjugated_system,
        has_nitro_group,
        has_carbonyl_extended,
        has_aromatic_amine,
        logp,
    )

    # Fraction remaining: A(t) = dose_mg * exp(-k_photo * illuminance_klux * t_h)
    # ICH Q1B: 50 klux for 24h
    illuminance_ich = 50.0  # klux
    t_ich = 24.0  # hours
    fraction_24h_50klux = math.exp(-k_photo * illuminance_ich * t_ich)

    # Indoor quick exposure: 5 klux for 1h
    illuminance_indoor = 5.0  # klux
    t_indoor = 1.0  # hour
    fraction_1h_5klux = math.exp(-k_photo * illuminance_indoor * t_indoor)

    # Half-life at 50 klux
    t_half = math.log(2.0) / (k_photo * illuminance_ich)

    category = _photostability_category(fraction_24h_50klux)
    ich_compliant = fraction_24h_50klux > 0.9
    packaging = _packaging_recommendation(category, ich_compliant)
    notes = _build_notes(
        category,
        ich_compliant,
        has_conjugated_system,
        has_nitro_group,
        has_carbonyl_extended,
        has_aromatic_amine,
        logp,
        k_photo,
    )

    return PhotodegradationResult(
        compound_name=compound_name,
        k_photo_per_h_per_klux=k_photo,
        fraction_remaining_24h_50klux=fraction_24h_50klux,
        fraction_remaining_1h_5klux=fraction_1h_5klux,
        t_half_50klux_h=t_half,
        photostability_category=category,
        ich_q1b_compliant=ich_compliant,
        packaging_recommendation=packaging,
        notes=notes,
    )


def screen_photodegradation(
    compounds: list[dict],
) -> list[PhotodegradationResult]:
    """Screen multiple compounds for photodegradation risk.

    Parameters
    ----------
    compounds:
        List of dicts with keys matching predict_photodegradation parameters.

    Returns
    -------
    List of PhotodegradationResult sorted by fraction_remaining_24h_50klux
    descending (most stable first).
    """
    results: list[PhotodegradationResult] = []
    for c in compounds:
        result = predict_photodegradation(**c)
        results.append(result)
    results.sort(key=lambda r: r.fraction_remaining_24h_50klux, reverse=True)
    return results
