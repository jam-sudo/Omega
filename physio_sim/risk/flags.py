from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskFlags:
    high_cmax_risk: bool
    long_half_life_risk: bool
    high_variability_risk: bool
    high_ddi_susceptibility_risk: bool


def compute_risk_flags(
    cmax_mg_per_l: float,
    half_life_h: float,
    exposure_cv: float,
    ddi_risk_score: float,
) -> RiskFlags:
    return RiskFlags(
        high_cmax_risk=cmax_mg_per_l >= 5.0,
        long_half_life_risk=half_life_h >= 24.0,
        high_variability_risk=exposure_cv >= 0.4,
        high_ddi_susceptibility_risk=ddi_risk_score >= 0.5,
    )
