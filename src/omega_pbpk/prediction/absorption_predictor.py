"""Oral fraction absorbed prediction from Peff, solubility, and dose."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AbsorptionResult:
    drug_name: str
    fa: float  # fraction absorbed (0–1)
    fa_dissolution: float  # fraction limited by dissolution
    fa_permeability: float  # fraction limited by permeability
    limiting_factor: str  # "dissolution" / "permeability" / "none"
    peff_cm_s: float
    solubility_mg_mL: float
    dose_number: float
    absorption_number: float  # An = Peff * 2 * t_transit / R
    bcs_like: str  # I, II, III, IV
    t_abs_h: float  # estimated time to reach fa
    confidence: float


# GI tract parameters
_RADIUS_CM = 1.75  # mean small intestine radius
_TRANSIT_TIME_H = 3.5  # mean transit time
_VOLUME_SIF_ML = 250.0  # volume of water for dose dissolution


def _absorption_number(peff_cm_s: float, transit_h: float = _TRANSIT_TIME_H) -> float:
    """An = Peff * 2 * t / R."""
    return peff_cm_s * 2.0 * transit_h * 3600.0 / _RADIUS_CM


def _dose_number(
    dose_mg: float, solubility_mg_mL: float, volume_mL: float = _VOLUME_SIF_ML
) -> float:
    """Do = Dose / (Cs * V250)."""
    return dose_mg / (solubility_mg_mL * volume_mL)


def _fa_permeability(an: float) -> float:
    """Permeability-limited fa = 1 - exp(-2 * An)  (cylindrical model)."""
    return float(1.0 - math.exp(-2.0 * an))


def _fa_dissolution(do: float) -> float:
    """Dissolution-limited fa = min(1, 1/Do).

    From Oh 1993: fa ≈ 1/Do when Do > 1.
    """
    if do <= 1.0:
        return 1.0
    return float(min(1.0 / do, 1.0))


def predict_absorption(
    drug_name: str,
    dose_mg: float,
    peff_cm_s: float,
    solubility_mg_mL: float,
    transit_h: float = _TRANSIT_TIME_H,
) -> AbsorptionResult:
    """Predict oral fraction absorbed (fa) using biopharmaceutical model.

    Combines permeability-limited and dissolution-limited absorption:
    fa = min(fa_permeability, fa_dissolution)

    Based on Amidon et al. 1995 BCS framework.

    Parameters
    ----------
    peff_cm_s : float
        Effective jejunal permeability (cm/s).
    solubility_mg_mL : float
        Equilibrium solubility in simulated intestinal fluid.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if peff_cm_s <= 0:
        raise ValueError("peff_cm_s must be > 0")
    if solubility_mg_mL <= 0:
        raise ValueError("solubility_mg_mL must be > 0")

    an = _absorption_number(peff_cm_s, transit_h)
    do = _dose_number(dose_mg, solubility_mg_mL)

    fa_perm = _fa_permeability(an)
    fa_diss = _fa_dissolution(do)

    fa = min(fa_perm, fa_diss)

    if fa_diss < fa_perm:
        limiting = "dissolution"
    elif fa_perm < fa_diss:
        limiting = "permeability"
    else:
        limiting = "none"

    # BCS-like classification
    high_perm = an >= 1.0  # fa_perm > 0.86
    high_sol = do <= 1.0  # fully dissolved at dose
    if high_perm and high_sol:
        bcs = "I"
    elif not high_perm and high_sol:
        bcs = "III"
    elif high_perm and not high_sol:
        bcs = "II"
    else:
        bcs = "IV"

    # Time to reach fa (simplified: t when 90% of fa achieved)
    if an > 0:
        t_abs = -math.log(1.0 - 0.9 * fa_perm) / (2.0 * peff_cm_s / _RADIUS_CM) / 3600.0
    else:
        t_abs = transit_h

    # Confidence
    confidence = (
        0.80 if (1e-6 <= peff_cm_s <= 1e-3) and (0.001 <= solubility_mg_mL <= 100.0) else 0.60
    )

    return AbsorptionResult(
        drug_name=drug_name,
        fa=fa,
        fa_dissolution=fa_diss,
        fa_permeability=fa_perm,
        limiting_factor=limiting,
        peff_cm_s=peff_cm_s,
        solubility_mg_mL=solubility_mg_mL,
        dose_number=do,
        absorption_number=an,
        bcs_like=bcs,
        t_abs_h=float(t_abs),
        confidence=confidence,
    )


def screen_absorption(compounds: list[dict]) -> list[AbsorptionResult]:
    """Screen multiple compounds for oral absorption.

    Each dict: {drug_name, dose_mg, peff_cm_s, solubility_mg_mL}
    """
    results = []
    for c in compounds:
        r = predict_absorption(
            drug_name=c.get("drug_name", "unknown"),
            dose_mg=c.get("dose_mg", 100.0),
            peff_cm_s=c.get("peff_cm_s", 1e-4),
            solubility_mg_mL=c.get("solubility_mg_mL", 1.0),
        )
        results.append(r)
    return results


__all__ = [
    "AbsorptionResult",
    "predict_absorption",
    "screen_absorption",
    "_absorption_number",
    "_dose_number",
]
