"""Drug crystal nucleation kinetics prediction from supersaturated solutions.

Nucleation rate determines how quickly a supersaturated ASD reverts to
crystalline form. Critical for formulation stability prediction.
Uses Classical Nucleation Theory (CNT) approximations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CrystalNucleationResult:
    """Result of crystal nucleation kinetics prediction."""

    drug_name: str
    supersaturation_ratio: float
    logp: float
    nucleation_rate: float
    induction_time_min: float
    growth_rate_relative: float
    crystallization_tendency: str
    polymorphism_risk: bool
    glass_forming_ability: str
    recommended_stabilizer: str
    notes: str


def predict_crystal_nucleation(
    drug_name: str,
    supersaturation_ratio: float,
    logp: float,
    mw: float = 300.0,
    melting_point_celsius: float = 170.0,
) -> CrystalNucleationResult:
    """Predict crystal nucleation kinetics from a supersaturated solution.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    supersaturation_ratio:
        S = C / Cs0 — concentration divided by equilibrium solubility.
        Must be > 1.0.
    logp:
        Octanol-water partition coefficient (log scale).
    mw:
        Molecular weight in g/mol (default 300).
    melting_point_celsius:
        Melting point in degrees Celsius (default 170).

    Returns
    -------
    CrystalNucleationResult
    """
    # --- Validation ---
    if supersaturation_ratio <= 1.0:
        raise ValueError(f"supersaturation_ratio must be > 1.0, got {supersaturation_ratio}")
    if mw <= 0:
        raise ValueError(f"mw must be positive, got {mw}")
    if melting_point_celsius <= 0:
        raise ValueError(f"melting_point_celsius must be positive, got {melting_point_celsius}")

    # --- Nucleation barrier (simplified CNT) ---
    ln_s = math.log(supersaturation_ratio)
    delta_G_barrier = 2.0 / (ln_s**2)

    # Relative nucleation rate (0-100 arbitrary units)
    nucleation_rate = min(100.0, 50.0 / delta_G_barrier)

    # Induction time (minutes) — inverse of nucleation driving force
    induction_time_min = max(1.0, 60.0 * delta_G_barrier / supersaturation_ratio)

    # Crystal growth rate (power law)
    growth_rate_relative = min(100.0, (supersaturation_ratio - 1.0) ** 1.5)

    # --- Crystallization tendency ---
    if nucleation_rate > 50:
        crystallization_tendency = "high"
    elif nucleation_rate > 20:
        crystallization_tendency = "moderate"
    else:
        crystallization_tendency = "low"

    # --- Polymorphism risk ---
    polymorphism_risk = logp > 3.0

    # --- Glass forming ability (GFA) ---
    gfa_score = 10.0 / (mw / 300.0 * supersaturation_ratio)
    if gfa_score > 2.0:
        glass_forming_ability = "class_I"
    elif gfa_score >= 1.0:
        glass_forming_ability = "class_II"
    else:
        glass_forming_ability = "class_III"

    # --- Recommended stabilizer ---
    if nucleation_rate > 50:
        recommended_stabilizer = "PVP"
    elif nucleation_rate > 20:
        recommended_stabilizer = "HPMC"
    elif nucleation_rate >= 10:
        recommended_stabilizer = "HPC"
    else:
        recommended_stabilizer = "none"

    notes = (
        f"ln(S)={ln_s:.3f}, delta_G_barrier={delta_G_barrier:.3f}. "
        f"GFA score={gfa_score:.3f}. "
        f"Melting point={melting_point_celsius}C used for context. "
        f"Higher MW and lower supersaturation improve glass-forming ability."
    )

    return CrystalNucleationResult(
        drug_name=drug_name,
        supersaturation_ratio=supersaturation_ratio,
        logp=logp,
        nucleation_rate=nucleation_rate,
        induction_time_min=induction_time_min,
        growth_rate_relative=growth_rate_relative,
        crystallization_tendency=crystallization_tendency,
        polymorphism_risk=polymorphism_risk,
        glass_forming_ability=glass_forming_ability,
        recommended_stabilizer=recommended_stabilizer,
        notes=notes,
    )


def screen_nucleation_stability(compounds: list) -> list:
    """Screen a list of compound dicts and rank by stability (longest induction first).

    Each compound dict must contain keys accepted by predict_crystal_nucleation:
    - drug_name (str)
    - supersaturation_ratio (float)
    - logp (float)
    - mw (float, optional)
    - melting_point_celsius (float, optional)

    Parameters
    ----------
    compounds:
        List of dicts, each describing a compound.

    Returns
    -------
    list of CrystalNucleationResult sorted by induction_time_min descending
    (most stable = longest induction time first).
    """
    results = []
    for compound in compounds:
        result = predict_crystal_nucleation(
            drug_name=compound["drug_name"],
            supersaturation_ratio=compound["supersaturation_ratio"],
            logp=compound["logp"],
            mw=compound.get("mw", 300.0),
            melting_point_celsius=compound.get("melting_point_celsius", 170.0),
        )
        results.append(result)

    # Sort by induction_time_min descending (most stable first)
    results.sort(key=lambda r: r.induction_time_min, reverse=True)
    return results
