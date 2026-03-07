"""Phase 1056 — Drug Plasma Stability (extended species-aware model).

Predict drug stability in plasma — key for IV formulations and ex-vivo stability testing.
Species-specific esterase activity, peptidase degradation, and temperature correction.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PlasmaStabilityResult:
    drug_name: str
    species: str
    temperature_C: float
    initial_conc_uM: float
    fraction_remaining_at_30min: float  # 0-1
    fraction_remaining_at_60min: float  # 0-1
    fraction_remaining_at_120min: float  # 0-1
    t50_min: float  # half-life in plasma (minutes)
    primary_degradation_pathway: str  # "esterase", "peptidase", "oxidation", "hydrolysis", "stable"
    stability_class: (
        str  # "unstable" (<30min), "labile" (30-60), "moderate" (60-120), "stable" (>120)
    )
    esterase_susceptibility: bool
    notes: str


_VALID_SPECIES = {"human", "rat", "mouse", "dog"}

_SPECIES_FACTOR = {
    "human": 1.0,
    "rat": 2.0,
    "mouse": 2.5,
    "dog": 1.5,
}


def predict_plasma_stability(
    drug_name: str,
    mw_Da: float,
    logp: float,
    has_ester: bool = False,
    has_amide: bool = False,
    has_lactam: bool = False,
    has_peptide_bond: bool = False,
    species: str = "human",
    temperature_C: float = 37.0,
    initial_conc_uM: float = 1.0,
) -> PlasmaStabilityResult:
    """Predict plasma stability of a drug.

    Args:
        drug_name: Name of the drug.
        mw_Da: Molecular weight in Daltons (must be > 0).
        logp: Lipophilicity (log P).
        has_ester: Whether drug contains an ester group.
        has_amide: Whether drug contains an amide group.
        has_lactam: Whether drug contains a lactam ring.
        has_peptide_bond: Whether drug contains peptide bonds.
        species: Biological species ("human", "rat", "mouse", "dog").
        temperature_C: Incubation temperature in Celsius (4-50).
        initial_conc_uM: Initial drug concentration in uM (must be > 0).

    Returns:
        PlasmaStabilityResult with stability predictions.
    """
    # Validation
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if initial_conc_uM <= 0:
        raise ValueError(f"initial_conc_uM must be > 0, got {initial_conc_uM}")
    if species not in _VALID_SPECIES:
        raise ValueError(f"species must be one of {_VALID_SPECIES}, got {species!r}")
    if temperature_C < 4 or temperature_C > 50:
        raise ValueError(f"temperature_C must be in [4, 50], got {temperature_C}")

    species_factor = _SPECIES_FACTOR[species]

    # Degradation rate constants (/min)
    k_esterase = 0.0
    if has_ester:
        k_esterase = 0.05 * species_factor

    k_peptidase = 0.0
    if has_amide or has_peptide_bond:
        k_peptidase = 0.02 * (1.0 + 0.5 * (1 if has_peptide_bond else 0))
        k_peptidase *= species_factor

    k_lactam = 0.0
    if has_lactam:
        k_lactam = 0.01

    k_ox = 0.001 * max(0.0, logp)

    # Total rate with baseline
    k_total = k_esterase + k_peptidase + k_lactam + k_ox + 0.0001

    # Temperature correction (37°C reference)
    k_total *= math.exp(0.05 * (temperature_C - 37.0))

    # Fraction remaining at timepoints
    fraction_at_30 = math.exp(-k_total * 30.0)
    fraction_at_60 = math.exp(-k_total * 60.0)
    fraction_at_120 = math.exp(-k_total * 120.0)

    # t50 in minutes
    t50_min = math.log(2.0) / k_total

    # Primary degradation pathway
    pathway_rates = {
        "esterase": k_esterase,
        "peptidase": k_peptidase,
        "hydrolysis": k_lactam,
        "oxidation": k_ox,
    }
    max_rate = max(pathway_rates.values())
    if max_rate < 0.005:
        primary_degradation_pathway = "stable"
    else:
        primary_degradation_pathway = max(pathway_rates, key=lambda p: pathway_rates[p])

    # Stability class
    if t50_min < 30.0:
        stability_class = "unstable"
    elif t50_min < 60.0:
        stability_class = "labile"
    elif t50_min < 120.0:
        stability_class = "moderate"
    else:
        stability_class = "stable"

    esterase_susceptibility = has_ester

    # Build notes
    notes_parts = [
        f"Species: {species}, T={temperature_C}°C.",
        f"k_total={k_total:.5f}/min, t\u00bd={t50_min:.1f} min.",
        f"Primary pathway: {primary_degradation_pathway}.",
        f"Stability class: {stability_class}.",
    ]
    if has_ester:
        notes_parts.append(f"Ester hydrolysis by {species} plasma esterases.")
    if has_peptide_bond:
        notes_parts.append("Peptide bond susceptible to plasma peptidases.")
    if stability_class in ("unstable", "labile"):
        notes_parts.append("Consider prodrug strategy or formulation protection.")
    notes = " ".join(notes_parts)

    return PlasmaStabilityResult(
        drug_name=drug_name,
        species=species,
        temperature_C=temperature_C,
        initial_conc_uM=initial_conc_uM,
        fraction_remaining_at_30min=fraction_at_30,
        fraction_remaining_at_60min=fraction_at_60,
        fraction_remaining_at_120min=fraction_at_120,
        t50_min=t50_min,
        primary_degradation_pathway=primary_degradation_pathway,
        stability_class=stability_class,
        esterase_susceptibility=esterase_susceptibility,
        notes=notes,
    )


def compare_species_stability(
    drug_name: str,
    mw_Da: float,
    logp: float,
    has_ester: bool = False,
    has_amide: bool = False,
    has_lactam: bool = False,
    has_peptide_bond: bool = False,
) -> list:
    """Compare plasma stability across all 4 species.

    Returns a list of PlasmaStabilityResult for all species,
    sorted by t50_min descending (most stable first).
    """
    results = []
    for species in _VALID_SPECIES:
        result = predict_plasma_stability(
            drug_name=drug_name,
            mw_Da=mw_Da,
            logp=logp,
            has_ester=has_ester,
            has_amide=has_amide,
            has_lactam=has_lactam,
            has_peptide_bond=has_peptide_bond,
            species=species,
        )
        results.append(result)

    results.sort(key=lambda r: r.t50_min, reverse=True)
    return results
