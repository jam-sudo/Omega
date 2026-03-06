"""GI bioaccessibility — pH-dependent solubility and absorption potential."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BioaccessibilityResult:
    drug_name: str
    pka: float | None
    drug_type: str  # 'acid', 'base', or 'neutral'
    dose_mg: float
    # Solubility at GI pH values
    solubility_stomach_mg_mL: float  # pH 1.5
    solubility_duodenum_mg_mL: float  # pH 6.3
    solubility_jejunum_mg_mL: float  # pH 6.8
    solubility_ileum_mg_mL: float  # pH 7.5
    solubility_colon_mg_mL: float  # pH 7.8
    # Precipitation risk
    precipitation_risk: str  # none / low / moderate / high
    # Dissolved fraction at each segment
    fa_stomach: float
    fa_intestine: float
    fa_total: float
    # Dose number
    dose_number: float  # Do = Dose / (Solubility_SIF * 250 mL); Do > 1 = absorption limited
    dose_limited: bool  # True if Do > 1


def _henderson_hasselbalch(
    intrinsic_sol: float,
    pH: float,
    pka: float | None,
    drug_type: str,
) -> float:
    """Compute pH-dependent solubility using Henderson-Hasselbalch.

    For acids: S(pH) = S0 * (1 + 10^(pH-pKa))
    For bases: S(pH) = S0 * (1 + 10^(pKa-pH))
    For neutral: S(pH) = S0
    """
    if pka is None or drug_type == "neutral":
        return float(intrinsic_sol)
    if drug_type == "acid":
        return float(intrinsic_sol * (1.0 + 10.0 ** (pH - pka)))
    elif drug_type == "base":
        return float(intrinsic_sol * (1.0 + 10.0 ** (pka - pH)))
    return float(intrinsic_sol)


def simulate_bioaccessibility(
    drug_name: str,
    dose_mg: float,
    intrinsic_solubility_mg_mL: float,
    pka: float | None = None,
    drug_type: str = "neutral",
    peff_cm_s: float = 1e-4,
    volume_stomach_mL: float = 50.0,
    volume_intestine_mL: float = 250.0,
) -> BioaccessibilityResult:
    """Predict GI bioaccessibility from solubility and permeability.

    Estimates dissolved drug at each GI segment using pH-solubility profile.

    Parameters
    ----------
    intrinsic_solubility_mg_mL : float
        Thermodynamic (intrinsic) solubility at pKa unit (uncharged form).
    pka : float or None
        Drug pKa (acid or base).
    drug_type : str
        'acid', 'base', or 'neutral'.
    peff_cm_s : float
        Effective intestinal permeability (cm/s).
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if intrinsic_solubility_mg_mL <= 0:
        raise ValueError("intrinsic_solubility_mg_mL must be > 0")

    S0 = intrinsic_solubility_mg_mL

    # GI segment pH values
    pH_stomach = 1.5
    pH_duodenum = 6.3
    pH_jejunum = 6.8
    pH_ileum = 7.5
    pH_colon = 7.8

    def sol(pH):
        return _henderson_hasselbalch(S0, pH, pka, drug_type)

    sol_stomach = sol(pH_stomach)
    sol_duodenum = sol(pH_duodenum)
    sol_jejunum = sol(pH_jejunum)
    sol_ileum = sol(pH_ileum)
    sol_colon = sol(pH_colon)

    # Dissolved fraction in stomach
    dissolved_stomach = min(dose_mg, sol_stomach * volume_stomach_mL)
    fa_stomach = dissolved_stomach / dose_mg

    # Dissolved fraction in intestine (use jejunum as representative)
    dissolved_intestine = min(dose_mg, sol_jejunum * volume_intestine_mL)
    fa_intestine = dissolved_intestine / dose_mg

    # Total bioaccessibility (absorption limited by dissolution)
    fa_total = min(1.0, fa_intestine)

    # Dose number (BCS)
    dose_number = dose_mg / (sol_jejunum * 250.0) if sol_jejunum > 0 else float("inf")
    dose_limited = dose_number > 1.0

    # Precipitation risk: acid drugs at high pH (small intestine), base drugs at low pH (stomach)
    if drug_type == "base" and pka is not None:
        # Base precipitates when moving from acidic stomach to higher pH intestine
        # If solubility drops >10x between stomach and jejunum
        sol_ratio = sol_stomach / max(sol_jejunum, 1e-9)
        if sol_ratio > 100:
            prec_risk = "high"
        elif sol_ratio > 10:
            prec_risk = "moderate"
        elif sol_ratio > 2:
            prec_risk = "low"
        else:
            prec_risk = "none"
    elif drug_type == "acid" and pka is not None:
        # Acid: solubility increases with pH, low risk of precipitation
        if sol_jejunum < S0 * 0.5:
            prec_risk = "moderate"
        else:
            prec_risk = "none"
    else:
        prec_risk = "none" if fa_total > 0.8 else "low"

    return BioaccessibilityResult(
        drug_name=drug_name,
        pka=pka,
        drug_type=drug_type,
        dose_mg=dose_mg,
        solubility_stomach_mg_mL=sol_stomach,
        solubility_duodenum_mg_mL=sol_duodenum,
        solubility_jejunum_mg_mL=sol_jejunum,
        solubility_ileum_mg_mL=sol_ileum,
        solubility_colon_mg_mL=sol_colon,
        precipitation_risk=prec_risk,
        fa_stomach=fa_stomach,
        fa_intestine=fa_intestine,
        fa_total=fa_total,
        dose_number=dose_number,
        dose_limited=dose_limited,
    )


def pH_solubility_profile(
    drug_name: str,
    intrinsic_solubility_mg_mL: float,
    pka: float | None = None,
    drug_type: str = "neutral",
    pH_range: tuple[float, float] = (1.0, 8.0),
    n_points: int = 20,
) -> dict:
    """Generate pH-solubility profile."""
    pHs = np.linspace(pH_range[0], pH_range[1], n_points).tolist()
    sols = [_henderson_hasselbalch(intrinsic_solubility_mg_mL, pH, pka, drug_type) for pH in pHs]
    return {
        "drug_name": drug_name,
        "pH": pHs,
        "solubility_mg_mL": sols,
        "logS": [math.log10(max(s, 1e-9)) for s in sols],
    }


__all__ = [
    "BioaccessibilityResult",
    "simulate_bioaccessibility",
    "pH_solubility_profile",
]
