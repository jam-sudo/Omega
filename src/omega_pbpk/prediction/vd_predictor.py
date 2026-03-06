"""Volume of distribution prediction from physicochemical descriptors."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "VdPredictionResult",
    "predict_vd",
    "vd_allometric_scale",
    "vd_population_range",
]


@dataclass
class VdPredictionResult:
    drug_name: str
    logP: float
    MW: float
    fup: float
    pka: float
    PSA: float
    vd_predicted_L: float
    vd_per_kg_L_per_kg: float
    vd_central_L: float
    vd_peripheral_L: float
    tissue_distribution_category: str
    confidence: float
    allometric_exponent: float


def _vd_from_descriptors(logP: float, MW: float, fup: float, PSA: float) -> float:
    """Estimate Vd (L/70 kg) from physicochemical descriptors."""
    base = 20.0 / fup
    logP_boost = 1.0 + 0.8 * max(logP - 1.0, 0.0)
    mw_factor = max(0.3, 1.0 - (MW - 300.0) / 800.0)
    PSA_factor = max(0.1, 1.0 - PSA / 250.0)
    vd = base * logP_boost * mw_factor * PSA_factor
    return max(5.0, min(5000.0, vd))


def predict_vd(
    drug_name: str,
    logP: float,
    MW: float,
    fup: float,
    pka: float = 7.0,
    PSA: float = 60.0,
) -> VdPredictionResult:
    """Predict volume of distribution for a 70 kg human.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    logP:
        Octanol-water partition coefficient (log scale).
    MW:
        Molecular weight (g/mol).
    fup:
        Fraction unbound in plasma (0 < fup <= 1).
    pka:
        Acid dissociation constant (default 7.0).
    PSA:
        Polar surface area in Å² (default 60.0).

    Returns
    -------
    VdPredictionResult

    Raises
    ------
    ValueError
        If fup <= 0, fup > 1, or MW <= 0.
    """
    if fup <= 0:
        raise ValueError("fup must be > 0")
    if fup > 1:
        raise ValueError("fup must be <= 1")
    if MW <= 0:
        raise ValueError("MW must be > 0")

    vd = _vd_from_descriptors(logP, MW, fup, PSA)

    if vd < 30.0:
        category = "low"
    elif vd < 150.0:
        category = "moderate"
    elif vd < 1000.0:
        category = "high"
    else:
        category = "very_high"

    vd_central = vd * 0.4
    vd_peripheral = vd * 0.6
    vd_per_kg = vd / 70.0

    confidence = 0.85 if (0.0 <= logP <= 5.0 and 100.0 <= MW <= 500.0) else 0.6

    return VdPredictionResult(
        drug_name=drug_name,
        logP=logP,
        MW=MW,
        fup=fup,
        pka=pka,
        PSA=PSA,
        vd_predicted_L=vd,
        vd_per_kg_L_per_kg=vd_per_kg,
        vd_central_L=vd_central,
        vd_peripheral_L=vd_peripheral,
        tissue_distribution_category=category,
        confidence=confidence,
        allometric_exponent=0.9,
    )


def vd_allometric_scale(
    vd_reference_L: float,
    reference_weight_kg: float,
    target_weight_kg: float,
    exponent: float = 0.9,
) -> float:
    """Scale Vd between body weights using allometric scaling.

    Parameters
    ----------
    vd_reference_L:
        Reference volume of distribution in litres.
    reference_weight_kg:
        Body weight (kg) corresponding to the reference Vd.
    target_weight_kg:
        Target body weight (kg) to scale to.
    exponent:
        Allometric exponent (default 0.9 for Vd).

    Returns
    -------
    float
        Scaled Vd in litres.
    """
    return vd_reference_L * (target_weight_kg / reference_weight_kg) ** exponent


def vd_population_range(
    drug_name: str,
    logP: float,
    MW: float,
    fup: float,
    cv_pct: float = 40.0,
) -> dict:
    """Compute population percentile range for Vd assuming lognormal variability.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    logP:
        Octanol-water partition coefficient.
    MW:
        Molecular weight (g/mol).
    fup:
        Fraction unbound in plasma.
    cv_pct:
        Coefficient of variation as a percentage (default 40 %).

    Returns
    -------
    dict with keys: drug_name, p50, p5, p95, cv_pct
    """
    result = predict_vd(drug_name, logP, MW, fup)
    p50 = result.vd_predicted_L

    # CV → sigma for lognormal: sigma² = ln(1 + CV²)
    cv = cv_pct / 100.0
    sigma = math.sqrt(math.log(1.0 + cv ** 2))
    mu = math.log(p50)

    # z-scores for 5th and 95th percentiles ≈ ±1.645
    z95 = 1.6448536269514729
    p5 = math.exp(mu - z95 * sigma)
    p95 = math.exp(mu + z95 * sigma)

    return {
        "drug_name": drug_name,
        "p5": p5,
        "p50": p50,
        "p95": p95,
        "cv_pct": cv_pct,
    }
