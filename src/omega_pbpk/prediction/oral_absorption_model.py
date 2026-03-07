"""Comprehensive oral absorption model combining dissolution, solubility, and permeability."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["OralAbsorptionModelResult", "model_oral_absorption", "screen_oral_absorption"]


@dataclass(frozen=True)
class OralAbsorptionModelResult:
    """Result of comprehensive oral absorption modelling."""

    compound_name: str
    logP: float
    mw: float
    dose_mg: float
    f_dissolved: float
    f_soluble: float
    fa_permeability: float
    fa_combined: float
    fh_hepatic: float
    f_oral_bioavailability: float
    papp_estimate_1e6: float
    dissolution_rate_limiting: bool
    solubility_rate_limiting: bool
    permeability_rate_limiting: bool
    limiting_factor: str
    notes: str


def _validate_inputs(
    dose_mg: float,
    solubility_mg_per_mL: float,
    mw: float,
    psa: float,
    particle_size_um: float,
) -> None:
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if solubility_mg_per_mL <= 0:
        raise ValueError("solubility_mg_per_mL must be > 0")
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if psa < 0:
        raise ValueError("psa must be >= 0")
    if particle_size_um <= 0:
        raise ValueError("particle_size_um must be > 0")


def model_oral_absorption(
    compound_name: str,
    logP: float,
    mw: float,
    psa: float,
    solubility_mg_per_mL: float,
    dose_mg: float,
    particle_size_um: float = 50.0,
    n_h_donors: int = 1,
    dissolution_time_min: float = 30.0,
) -> OralAbsorptionModelResult:
    """Model oral absorption combining dissolution, solubility, and permeability.

    Args:
        compound_name: Compound identifier.
        logP: Lipophilicity (log octanol/water partition coefficient).
        mw: Molecular weight (g/mol).
        psa: Polar surface area (Å²).
        solubility_mg_per_mL: Aqueous solubility (mg/mL).
        dose_mg: Oral dose (mg).
        particle_size_um: Particle diameter (µm). Default 50.
        n_h_donors: Number of H-bond donors. Default 1.
        dissolution_time_min: Dissolution assessment time (min). Default 30.

    Returns:
        OralAbsorptionModelResult with absorption fractions and limiting factor.
    """
    _validate_inputs(dose_mg, solubility_mg_per_mL, mw, psa, particle_size_um)

    # Step 1: Dissolution (Noyes-Whitney simplified)
    k_diss_per_h = 3.0 * (10.0 / particle_size_um) * (0.5 / mw) ** 0.3
    f_dissolved = min(1.0, 1.0 - math.exp(-k_diss_per_h * dissolution_time_min / 60.0))
    f_dissolved = max(0.0, f_dissolved)

    # Step 2: Solubility limit
    gastric_vol_mL = 250.0
    max_soluble_mg = solubility_mg_per_mL * gastric_vol_mL
    f_soluble = min(1.0, max_soluble_mg / dose_mg)

    f_dissolved_effective = f_dissolved * f_soluble

    # Step 3: Permeability (Papp from QSAR)
    papp_base = 10 ** (0.5 * logP - 1.0)
    psa_penalty = math.exp(-max(0.0, psa - 60.0) * 0.02)
    mw_penalty = math.exp(-max(0.0, mw - 300.0) * 0.003)
    donor_penalty = math.exp(-max(0.0, n_h_donors - 1) * 0.3)
    papp_1e6 = papp_base * psa_penalty * mw_penalty * donor_penalty
    papp_1e6 = max(0.01, min(300.0, papp_1e6))

    fa_permeability = min(0.99, papp_1e6 / (papp_1e6 + 3.0))

    # Step 4: Combined fa
    fa_combined = f_dissolved_effective * fa_permeability

    # Step 5: Hepatic first-pass (well-stirred model)
    effective_logP = max(0.1, logP)
    clint_base = 20.0 * effective_logP
    fu_mic = 1.0 / (1.0 + 0.072 * effective_logP)
    clint_u = clint_base / fu_mic
    fup = max(0.01, 1.0 / (1.0 + 0.1 * max(0.0, logP)))
    q_h = 90.0  # hepatic blood flow L/h
    clh = q_h * fup * clint_u / (q_h + fup * clint_u)
    fh_hepatic = 1.0 - clh / q_h

    # Step 6: Oral bioavailability
    f_oral = fa_combined * fh_hepatic

    # Identify limiting factor
    dissolution_rate_limiting = f_dissolved < fa_permeability
    solubility_rate_limiting = f_soluble < f_dissolved
    permeability_rate_limiting = fa_permeability < f_dissolved_effective

    # Determine primary limiting factor
    factors = []
    if dissolution_rate_limiting:
        factors.append("dissolution")
    if solubility_rate_limiting:
        factors.append("solubility")
    if permeability_rate_limiting:
        factors.append("permeability")

    if fh_hepatic < 0.5:
        factors.append("hepatic")

    if len(factors) == 0:
        limiting_factor = "permeability"  # default
    elif len(factors) == 1:
        limiting_factor = factors[0]
    else:
        limiting_factor = "multiple"

    notes = (
        f"k_diss={k_diss_per_h:.3f}/h; "
        f"Papp={papp_1e6:.2f}×10⁻⁶ cm/s; "
        f"fup={fup:.3f}; "
        f"fh={fh_hepatic:.3f}; "
        f"F={f_oral:.3f}. "
        f"Limiting: {limiting_factor}."
    )

    return OralAbsorptionModelResult(
        compound_name=compound_name,
        logP=logP,
        mw=mw,
        dose_mg=dose_mg,
        f_dissolved=f_dissolved,
        f_soluble=f_soluble,
        fa_permeability=fa_permeability,
        fa_combined=fa_combined,
        fh_hepatic=fh_hepatic,
        f_oral_bioavailability=f_oral,
        papp_estimate_1e6=papp_1e6,
        dissolution_rate_limiting=dissolution_rate_limiting,
        solubility_rate_limiting=solubility_rate_limiting,
        permeability_rate_limiting=permeability_rate_limiting,
        limiting_factor=limiting_factor,
        notes=notes,
    )


def screen_oral_absorption(compounds: list) -> list:
    """Screen multiple compounds for oral absorption.

    Args:
        compounds: List of dicts, each with keys matching model_oral_absorption parameters.

    Returns:
        List of OralAbsorptionModelResult sorted by f_oral_bioavailability descending.
    """
    results = []
    for c in compounds:
        result = model_oral_absorption(
            compound_name=c.get("compound_name", "Unknown"),
            logP=c["logP"],
            mw=c["mw"],
            psa=c["psa"],
            solubility_mg_per_mL=c["solubility_mg_per_mL"],
            dose_mg=c["dose_mg"],
            particle_size_um=c.get("particle_size_um", 50.0),
            n_h_donors=c.get("n_h_donors", 1),
            dissolution_time_min=c.get("dissolution_time_min", 30.0),
        )
        results.append(result)

    results.sort(key=lambda r: r.f_oral_bioavailability, reverse=True)
    return results
