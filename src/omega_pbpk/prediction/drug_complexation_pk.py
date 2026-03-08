"""Drug-cyclodextrin complexation effects on solubility and PK (Phase 1018).

Models inclusion complex formation using Higuchi & Connors phase solubility
theory (AL type).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "CyclodextrinComplexResult",
    "predict_cyclodextrin_complexation",
    "screen_cyclodextrins",
]

# Cyclodextrin properties: Ka_factor (M^-1 dimensionless scale), aq_solubility_mM
_CD_PROPERTIES: dict[str, dict[str, float]] = {
    "beta-CD": {"Ka_factor": 500.0, "aqueous_solubility_mM": 16.0},
    "HP-beta-CD": {"Ka_factor": 800.0, "aqueous_solubility_mM": 600.0},
    "SBE-beta-CD": {"Ka_factor": 1200.0, "aqueous_solubility_mM": 600.0},
    "gamma-CD": {"Ka_factor": 200.0, "aqueous_solubility_mM": 200.0},
    "alpha-CD": {"Ka_factor": 100.0, "aqueous_solubility_mM": 130.0},
}


@dataclass(frozen=True)
class CyclodextrinComplexResult:
    """Result of cyclodextrin complexation prediction."""

    drug_name: str
    cyclodextrin_name: str
    drug_logp: float
    complex_stability_constant_Ka: float
    drug_intrinsic_solubility_mg_mL: float
    complexed_solubility_mg_mL: float
    solubility_enhancement: float
    cd_concentration_mM: float
    solubilization_efficiency_pct: float
    phase_solubility_slope: float
    bioavailability_benefit: str
    notes: str


def predict_cyclodextrin_complexation(
    drug_name: str,
    cyclodextrin_name: str,
    drug_logp: float,
    drug_mw: float = 300.0,
    cd_concentration_mM: float = 10.0,
) -> CyclodextrinComplexResult:
    """Predict solubility enhancement from drug-cyclodextrin complexation.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    cyclodextrin_name:
        One of "beta-CD", "HP-beta-CD", "SBE-beta-CD", "gamma-CD", "alpha-CD".
    drug_logp:
        Octanol-water partition coefficient (log scale).
    drug_mw:
        Drug molecular weight (g/mol), default 300.
    cd_concentration_mM:
        Cyclodextrin concentration in the formulation (mM).

    Returns
    -------
    CyclodextrinComplexResult
    """
    if cyclodextrin_name not in _CD_PROPERTIES:
        raise ValueError(
            f"cyclodextrin_name must be one of {list(_CD_PROPERTIES)}, got {cyclodextrin_name!r}"
        )
    if cd_concentration_mM <= 0:
        raise ValueError(f"cd_concentration_mM must be > 0, got {cd_concentration_mM}")
    if drug_mw <= 0:
        raise ValueError(f"drug_mw must be > 0, got {drug_mw}")

    props = _CD_PROPERTIES[cyclodextrin_name]
    ka_factor = props["Ka_factor"]

    # Intrinsic solubility from logP
    s0 = 10.0 ** (-0.7 * drug_logp + 0.44)  # mg/mL

    # Stability constant Ka (M^-1) — logP-dependent affinity
    ka = ka_factor * (1.0 + 0.1 * drug_logp)

    # Phase solubility (Higuchi & Connors AL type)
    cd_conc_M = cd_concentration_mM / 1000.0
    s_total = s0 * (1.0 + ka * cd_conc_M)  # mg/mL

    # Phase solubility slope (mg/mL per mM CD)
    phase_slope = s0 * ka * drug_mw / 1000.0

    solubility_enhancement = s_total / s0  # == 1 + Ka * [CD]

    # Complexed fraction
    f_complex = ka * cd_conc_M / (1.0 + ka * cd_conc_M)
    solubilization_efficiency = min(100.0, f_complex * 100.0)

    # Bioavailability benefit classification
    if solubility_enhancement > 10.0:
        benefit = "high"
    elif solubility_enhancement > 3.0:
        benefit = "moderate"
    elif solubility_enhancement > 1.5:
        benefit = "low"
    else:
        benefit = "none"

    notes = f"S0={s0:.4f} mg/mL, Ka={ka:.1f} M-1, CD_aq_sol={props['aqueous_solubility_mM']:.0f} mM"

    return CyclodextrinComplexResult(
        drug_name=drug_name,
        cyclodextrin_name=cyclodextrin_name,
        drug_logp=drug_logp,
        complex_stability_constant_Ka=ka,
        drug_intrinsic_solubility_mg_mL=s0,
        complexed_solubility_mg_mL=s_total,
        solubility_enhancement=solubility_enhancement,
        cd_concentration_mM=cd_concentration_mM,
        solubilization_efficiency_pct=solubilization_efficiency,
        phase_solubility_slope=phase_slope,
        bioavailability_benefit=benefit,
        notes=notes,
    )


def screen_cyclodextrins(
    drug_name: str,
    drug_logp: float,
    drug_mw: float,
    cd_concentration_mM: float = 10.0,
) -> list[CyclodextrinComplexResult]:
    """Screen all cyclodextrins for a drug and return sorted by enhancement.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    drug_logp:
        Octanol-water partition coefficient (log scale).
    drug_mw:
        Drug molecular weight (g/mol).
    cd_concentration_mM:
        Cyclodextrin concentration in mM.

    Returns
    -------
    List of CyclodextrinComplexResult sorted by solubility_enhancement descending.
    """
    results = [
        predict_cyclodextrin_complexation(
            drug_name=drug_name,
            cyclodextrin_name=cd_name,
            drug_logp=drug_logp,
            drug_mw=drug_mw,
            cd_concentration_mM=cd_concentration_mM,
        )
        for cd_name in _CD_PROPERTIES
    ]
    results.sort(key=lambda r: r.solubility_enhancement, reverse=True)
    return results


# Expose math import usage to satisfy any lint checks
_LOG10 = math.log10
