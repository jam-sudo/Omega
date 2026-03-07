"""
Phase 251: Nanoemulsion Drug Delivery Formulation Designer

Design and screen nanoemulsion formulations for oral or topical drug delivery.
Uses physicochemical properties to predict droplet size, PDI, solubilization
efficiency, stability, and bioavailability enhancement.
"""

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Oil phase lookup table
# ---------------------------------------------------------------------------

_OIL_PHASES = {
    "MCT": {
        "solubilization_capacity": 5.0,
        "viscosity_cP": 30.0,
        "hlab_required": 9.0,
    },
    "soybean_oil": {
        "solubilization_capacity": 3.0,
        "viscosity_cP": 50.0,
        "hlab_required": 7.0,
    },
    "castor_oil": {
        "solubilization_capacity": 4.0,
        "viscosity_cP": 100.0,
        "hlab_required": 10.0,
    },
    "IPM": {
        "solubilization_capacity": 6.0,
        "viscosity_cP": 5.0,
        "hlab_required": 12.0,
    },
    "oleic_acid": {
        "solubilization_capacity": 7.0,
        "viscosity_cP": 25.0,
        "hlab_required": 10.0,
    },
}

VALID_OIL_PHASES = list(_OIL_PHASES.keys())


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NanoemulsionResult:
    """Result of a nanoemulsion formulation design calculation."""

    drug_name: str
    oil_phase: str
    drug_loading_pct: float
    droplet_size_nm: float
    polydispersity_index: float
    solubilization_efficiency_pct: float
    stability_score: float
    bioavailability_enhancement_factor: float
    emulsification_time_min: float
    formulation_class: str
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _formulation_class(stability_score: float) -> str:
    """Classify formulation quality from stability score."""
    if stability_score >= 80.0:
        return "excellent"
    if stability_score >= 60.0:
        return "good"
    if stability_score >= 40.0:
        return "acceptable"
    return "poor"


def _compute_nanoemulsion(
    drug_name: str,
    oil_phase: str,
    drug_loading_pct: float,
    sol_cap: float,
    viscosity_cP: float,
) -> NanoemulsionResult:
    """Core computation shared by design_nanoemulsion and screen_oil_phases."""

    # Droplet size
    raw_droplet = 100.0 + 50.0 * (viscosity_cP / 50.0) - 30.0 * math.log(sol_cap)
    droplet_size_nm = _clamp(raw_droplet, 50.0, 500.0)

    # PDI
    raw_pdi = 0.1 + 0.01 * drug_loading_pct
    pdi = _clamp(raw_pdi, 0.1, 0.5)

    # Solubilization efficiency
    sol_eff = min(100.0, sol_cap * (1.0 - 0.005 * drug_loading_pct) * 15.0)

    # Stability score
    stability_score = 100.0 * (1.0 - pdi) * (sol_eff / 100.0)

    # Bioavailability enhancement factor
    bioavailability_factor = 1.0 + (sol_eff / 100.0) * 3.0

    # Emulsification time
    emulsification_time_min = 5.0 + viscosity_cP / 20.0

    # Classification
    f_class = _formulation_class(stability_score)

    notes = (
        f"Oil: {oil_phase}; droplet={droplet_size_nm:.1f} nm; "
        f"PDI={pdi:.3f}; sol_eff={sol_eff:.1f}%; "
        f"stability={stability_score:.1f}; class={f_class}"
    )

    return NanoemulsionResult(
        drug_name=drug_name,
        oil_phase=oil_phase,
        drug_loading_pct=drug_loading_pct,
        droplet_size_nm=droplet_size_nm,
        polydispersity_index=pdi,
        solubilization_efficiency_pct=sol_eff,
        stability_score=stability_score,
        bioavailability_enhancement_factor=bioavailability_factor,
        emulsification_time_min=emulsification_time_min,
        formulation_class=f_class,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def design_nanoemulsion(
    drug_name: str,
    logp: float,
    drug_loading_pct: float,
    oil_phase: str = "MCT",
) -> NanoemulsionResult:
    """
    Design a nanoemulsion formulation for a given drug.

    Parameters
    ----------
    drug_name : str
        Name of the drug compound.
    logp : float
        Partition coefficient (log P) of the drug.
    drug_loading_pct : float
        Drug loading as a percentage of total formulation (0, 50].
    oil_phase : str
        One of: MCT, soybean_oil, castor_oil, IPM, oleic_acid.

    Returns
    -------
    NanoemulsionResult
    """
    if not (0.0 < drug_loading_pct <= 50.0):
        raise ValueError(f"drug_loading_pct must be in (0, 50], got {drug_loading_pct}")
    if oil_phase not in _OIL_PHASES:
        raise ValueError(f"oil_phase must be one of {VALID_OIL_PHASES}, got '{oil_phase}'")

    props = _OIL_PHASES[oil_phase]
    sol_cap = props["solubilization_capacity"]
    viscosity_cP = props["viscosity_cP"]

    # logP effect on solubilization capacity
    if logp < 2.0:
        sol_cap *= 0.5

    return _compute_nanoemulsion(
        drug_name=drug_name,
        oil_phase=oil_phase,
        drug_loading_pct=drug_loading_pct,
        sol_cap=sol_cap,
        viscosity_cP=viscosity_cP,
    )


def screen_oil_phases(
    drug_name: str,
    logp: float,
    drug_loading_pct: float = 10.0,
) -> list[NanoemulsionResult]:
    """
    Screen all 5 oil phase types for the given drug.

    Returns a list of NanoemulsionResult sorted by stability_score descending.

    Parameters
    ----------
    drug_name : str
        Name of the drug compound.
    logp : float
        Partition coefficient (log P) of the drug.
    drug_loading_pct : float
        Drug loading percentage (default 10.0).

    Returns
    -------
    list of NanoemulsionResult (5 items, sorted best → worst)
    """
    results = []
    for op in VALID_OIL_PHASES:
        result = design_nanoemulsion(
            drug_name=drug_name,
            logp=logp,
            drug_loading_pct=drug_loading_pct,
            oil_phase=op,
        )
        results.append(result)

    results.sort(key=lambda r: r.stability_score, reverse=True)
    return results
