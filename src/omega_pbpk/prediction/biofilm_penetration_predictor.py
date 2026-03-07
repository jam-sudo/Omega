"""
Phase 283 — Biofilm Penetration Predictor.

Predict drug penetration through bacterial biofilms using a reaction-diffusion model.
Estimates effective concentration at biofilm base for antibiotic PK/PD.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cosh, exp, sqrt

__all__ = [
    "BiofilmPenetrationResult",
    "predict_biofilm_penetration",
    "screen_antibiotics",
]

# D_water in cm2/s: 3.6e-6 cm2/s
_D_WATER_CM2_S = 3.6e-6


@dataclass(frozen=True)
class BiofilmPenetrationResult:
    """Result of biofilm penetration prediction (frozen dataclass)."""

    drug_name: str
    mw: float
    logp: float
    charge: float
    bulk_conc_mg_L: float
    biofilm_thickness_um: float
    diffusion_reduction_factor: float
    consumption_rate_per_h: float
    thiele_modulus: float
    penetration_fraction: float
    c_effective_mg_L: float
    mw_factor: float
    charge_factor: float
    penetration_class: str
    notes: str


def _validate_inputs(
    mw: float,
    bulk_conc_mg_L: float,
    biofilm_thickness_um: float,
    diffusion_reduction_factor: float,
    consumption_rate_per_h: float,
) -> None:
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if bulk_conc_mg_L <= 0:
        raise ValueError(f"bulk_conc_mg_L must be > 0, got {bulk_conc_mg_L}")
    if biofilm_thickness_um <= 0:
        raise ValueError(f"biofilm_thickness_um must be > 0, got {biofilm_thickness_um}")
    if not (0 < diffusion_reduction_factor <= 1):
        raise ValueError(
            f"diffusion_reduction_factor must be in (0, 1], got {diffusion_reduction_factor}"
        )
    if consumption_rate_per_h < 0:
        raise ValueError(f"consumption_rate_per_h must be >= 0, got {consumption_rate_per_h}")


def predict_biofilm_penetration(
    drug_name: str,
    mw: float,
    logp: float,
    charge: float,
    bulk_conc_mg_L: float,
    biofilm_thickness_um: float,
    diffusion_reduction_factor: float,
    consumption_rate_per_h: float,
) -> BiofilmPenetrationResult:
    """
    Predict drug penetration through a bacterial biofilm.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    mw : float
        Molecular weight (Da). Must be > 0.
    logp : float
        Octanol-water partition coefficient (log units).
    charge : float
        Net charge of the drug at physiological pH.
    bulk_conc_mg_L : float
        Drug concentration in bulk fluid (mg/L). Must be > 0.
    biofilm_thickness_um : float
        Biofilm thickness in micrometers. Must be > 0.
    diffusion_reduction_factor : float
        Ratio D_biofilm / D_water (0 < value <= 1).
    consumption_rate_per_h : float
        Drug consumption rate by bacteria (per hour). Must be >= 0.

    Returns
    -------
    BiofilmPenetrationResult
    """
    _validate_inputs(
        mw,
        bulk_conc_mg_L,
        biofilm_thickness_um,
        diffusion_reduction_factor,
        consumption_rate_per_h,
    )

    # Effective diffusion coefficient in cm2/s
    d_eff_cm2_s = _D_WATER_CM2_S * diffusion_reduction_factor

    # Biofilm thickness in cm
    l_cm = biofilm_thickness_um * 1e-4

    # Thiele modulus (dimensionless): phi = L * sqrt(k / D_eff)
    # consumption_rate_per_h converted to per second: / 3600
    k_per_s = consumption_rate_per_h / 3600.0
    if k_per_s > 0:
        phi = l_cm * sqrt(k_per_s / d_eff_cm2_s)
    else:
        phi = 0.0

    # Penetration fraction from reaction-diffusion: 1/cosh(phi)
    pf_reaction = 1.0 / cosh(phi)

    # MW penalty: larger drugs penetrate less through EPS matrix
    mw_factor = exp(-max(0.0, mw - 300.0) / 500.0)

    # Charge penalty: charged drugs are excluded by EPS
    charge_factor = 0.7 if abs(charge) > 1 else 1.0

    # Final penetration fraction (clamped to [0, 1])
    penetration_fraction = pf_reaction * mw_factor * charge_factor
    penetration_fraction = max(0.0, min(1.0, penetration_fraction))

    # Effective concentration at biofilm base
    c_effective_mg_L = bulk_conc_mg_L * penetration_fraction

    # Classify penetration
    if penetration_fraction > 0.5:
        penetration_class = "good"
    elif penetration_fraction >= 0.2:
        penetration_class = "moderate"
    else:
        penetration_class = "poor"

    notes_parts = []
    if phi > 2.0:
        notes_parts.append("high Thiele modulus: reaction-limited penetration")
    if mw > 500:
        notes_parts.append("large MW significantly reduces EPS penetration")
    if abs(charge) > 1:
        notes_parts.append("high charge reduces EPS penetration")
    if not notes_parts:
        notes_parts.append("standard penetration conditions")
    notes = "; ".join(notes_parts)

    return BiofilmPenetrationResult(
        drug_name=drug_name,
        mw=mw,
        logp=logp,
        charge=charge,
        bulk_conc_mg_L=bulk_conc_mg_L,
        biofilm_thickness_um=biofilm_thickness_um,
        diffusion_reduction_factor=diffusion_reduction_factor,
        consumption_rate_per_h=consumption_rate_per_h,
        thiele_modulus=phi,
        penetration_fraction=penetration_fraction,
        c_effective_mg_L=c_effective_mg_L,
        mw_factor=mw_factor,
        charge_factor=charge_factor,
        penetration_class=penetration_class,
        notes=notes,
    )


def screen_antibiotics(drug_list: list[dict]) -> list[BiofilmPenetrationResult]:
    """
    Screen a list of antibiotics for biofilm penetration.

    Parameters
    ----------
    drug_list : list of dict
        Each dict must have keys matching the parameters of
        ``predict_biofilm_penetration``.

    Returns
    -------
    list of BiofilmPenetrationResult
        Sorted descending by penetration_fraction.
    """
    results = []
    for drug in drug_list:
        result = predict_biofilm_penetration(
            drug_name=drug["drug_name"],
            mw=drug["mw"],
            logp=drug["logp"],
            charge=drug["charge"],
            bulk_conc_mg_L=drug["bulk_conc_mg_L"],
            biofilm_thickness_um=drug["biofilm_thickness_um"],
            diffusion_reduction_factor=drug["diffusion_reduction_factor"],
            consumption_rate_per_h=drug["consumption_rate_per_h"],
        )
        results.append(result)
    results.sort(key=lambda r: r.penetration_fraction, reverse=True)
    return results
