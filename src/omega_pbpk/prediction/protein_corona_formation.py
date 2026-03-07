"""Phase 1055 — Drug Protein Corona Formation.

Models the full protein corona layer composition on nanoparticles.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ProteinCoronaResult:
    particle_name: str
    particle_size_nm: float
    hard_corona_thickness_nm: float  # irreversibly bound proteins
    soft_corona_thickness_nm: float  # loosely bound proteins
    total_corona_thickness_nm: float
    effective_size_nm: float  # particle + total corona
    dominant_proteins: tuple  # top 3 proteins in corona (strings)
    protein_coverage_pct: float  # % surface covered by proteins
    corona_stability_score: float  # 0-100 (100 = very stable corona)
    cellular_uptake_modifier: float  # multiplier on baseline uptake (>1 = enhanced)
    stealth_efficiency: float  # 0-1 (PEG/coating reduces corona formation)
    notes: str


_VALID_COATINGS = {"none", "PEG_2k", "PEG_5k", "citrate", "amine", "carboxyl"}
_VALID_MATERIALS = {"gold", "silica", "PLGA", "iron_oxide", "lipid"}

_MATERIAL_AFFINITY = {
    "gold": 0.9,
    "silica": 0.8,
    "PLGA": 0.6,
    "iron_oxide": 0.85,
    "lipid": 0.5,
}

_COATING_SHIELD = {
    "none": 0.0,
    "PEG_2k": 0.5,
    "PEG_5k": 0.7,
    "citrate": 0.2,
    "amine": -0.1,  # enhances adsorption
    "carboxyl": 0.15,
}


def predict_protein_corona(
    particle_name: str,
    particle_size_nm: float,
    surface_coating: str = "none",
    zeta_potential_mV: float = -10.0,
    material: str = "gold",
    serum_protein_conc_mg_mL: float = 70.0,
    incubation_time_h: float = 1.0,
) -> ProteinCoronaResult:
    """Predict protein corona formation on nanoparticles.

    Args:
        particle_name: Name/identifier of the nanoparticle.
        particle_size_nm: Particle diameter (1-500 nm).
        surface_coating: Surface coating type.
        zeta_potential_mV: Zeta potential in mV.
        material: Particle material type.
        serum_protein_conc_mg_mL: Total serum protein concentration (mg/mL).
        incubation_time_h: Incubation time in hours.

    Returns:
        ProteinCoronaResult with corona composition details.
    """
    # Validation
    if particle_size_nm <= 0 or particle_size_nm > 500:
        raise ValueError(f"particle_size_nm must be in (0, 500], got {particle_size_nm}")
    if surface_coating not in _VALID_COATINGS:
        raise ValueError(
            f"surface_coating must be one of {_VALID_COATINGS}, got {surface_coating!r}"
        )
    if material not in _VALID_MATERIALS:
        raise ValueError(f"material must be one of {_VALID_MATERIALS}, got {material!r}")
    if serum_protein_conc_mg_mL <= 0:
        raise ValueError(f"serum_protein_conc_mg_mL must be > 0, got {serum_protein_conc_mg_mL}")
    if incubation_time_h <= 0:
        raise ValueError(f"incubation_time_h must be > 0, got {incubation_time_h}")

    # Surface area (relative, for scoring context)
    _area_nm2 = 4 * math.pi * (particle_size_nm / 2) ** 2

    # Material adsorption affinity
    affinity = _MATERIAL_AFFINITY[material]

    # Surface coating shield effect
    shield_value = _COATING_SHIELD[surface_coating]
    stealth_efficiency = max(0.0, min(1.0, shield_value if shield_value > 0 else 0.0))

    # Effective adsorption
    effective_adsorption = affinity * (1.0 - stealth_efficiency)

    # Hard corona thickness (nm)
    base = 5.0 + 0.01 * serum_protein_conc_mg_mL
    hard_corona = base * effective_adsorption * (1.0 + 0.1 * zeta_potential_mV / 10.0)
    hard_corona = max(0.5, min(20.0, abs(hard_corona)))
    # Grows with time, saturates at 1h
    hard_corona *= 1.0 + min(1.0, incubation_time_h) * 0.2

    # Soft corona thickness
    soft_corona = hard_corona * 0.3

    total_corona = hard_corona + soft_corona
    effective_size = particle_size_nm + total_corona

    # Dominant proteins based on zeta potential
    if zeta_potential_mV < -20.0:
        dominant_proteins: tuple = ("IgG", "fibronectin", "albumin")
    elif zeta_potential_mV <= 0.0:
        dominant_proteins = ("albumin", "IgG", "fibrinogen")
    else:
        dominant_proteins = ("vitronectin", "fibrinogen", "IgM")

    # Protein coverage percent
    protein_coverage_pct = (
        effective_adsorption
        * (1.0 - stealth_efficiency)
        * 100.0
        * (1.0 - math.exp(-incubation_time_h))
    )
    if surface_coating == "PEG_5k":
        protein_coverage_pct *= 0.7
    protein_coverage_pct = max(5.0, min(99.0, protein_coverage_pct))

    # Corona stability score (0-100)
    corona_stability_score = 50.0 + hard_corona * 2.0
    if surface_coating in ("PEG_2k", "PEG_5k"):
        corona_stability_score -= 20.0
    corona_stability_score = max(10.0, min(100.0, corona_stability_score))

    # Cellular uptake modifier
    cellular_uptake_modifier = 1.0
    if dominant_proteins[0] in ("IgG", "vitronectin", "fibronectin"):
        cellular_uptake_modifier *= 1.5
    if stealth_efficiency > 0.6:
        cellular_uptake_modifier *= 0.5
    if zeta_potential_mV > 10.0:
        cellular_uptake_modifier *= 1.3
    cellular_uptake_modifier = max(0.1, min(3.0, cellular_uptake_modifier))

    # Build notes
    notes_parts = [
        f"Material: {material}, coating: {surface_coating}.",
        f"Effective adsorption factor: {effective_adsorption:.3f}.",
        f"Dominant opsonin: {dominant_proteins[0]}.",
    ]
    if stealth_efficiency > 0.5:
        notes_parts.append("PEG coating provides significant stealth effect.")
    if cellular_uptake_modifier > 1.2:
        notes_parts.append("Enhanced cellular uptake expected (opsonization).")
    elif cellular_uptake_modifier < 0.8:
        notes_parts.append("Reduced cellular uptake due to stealth coating.")
    notes = " ".join(notes_parts)

    return ProteinCoronaResult(
        particle_name=particle_name,
        particle_size_nm=particle_size_nm,
        hard_corona_thickness_nm=hard_corona,
        soft_corona_thickness_nm=soft_corona,
        total_corona_thickness_nm=total_corona,
        effective_size_nm=effective_size,
        dominant_proteins=dominant_proteins,
        protein_coverage_pct=protein_coverage_pct,
        corona_stability_score=corona_stability_score,
        cellular_uptake_modifier=cellular_uptake_modifier,
        stealth_efficiency=stealth_efficiency,
        notes=notes,
    )


def compare_coatings_corona(
    particle_name: str,
    particle_size_nm: float,
    zeta_potential_mV: float = -10.0,
    material: str = "gold",
) -> list:
    """Compare all surface coatings for protein corona formation.

    Returns a list of ProteinCoronaResult for all 6 coatings,
    sorted by stealth_efficiency descending.
    """
    results = []
    for coating in _VALID_COATINGS:
        result = predict_protein_corona(
            particle_name=particle_name,
            particle_size_nm=particle_size_nm,
            surface_coating=coating,
            zeta_potential_mV=zeta_potential_mV,
            material=material,
        )
        results.append(result)

    results.sort(key=lambda r: r.stealth_efficiency, reverse=True)
    return results
