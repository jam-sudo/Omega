"""
ML-based clearance prediction from physicochemical descriptors.

Uses linear physicochemical relationships derived from literature to predict
hepatic clearance, renal clearance, volume of distribution, and half-life.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = [
    "ClearancePredictionResult",
    "predict_clearance",
    "batch_clearance_predict",
]


@dataclass(frozen=True)
class ClearancePredictionResult:
    """Result of a clearance prediction for a single compound."""

    drug_name: str
    logP: float
    MW: float
    pka: float
    fup: float
    PSA: float
    cl_predicted_L_per_h: float
    cl_renal_L_per_h: float
    cl_hepatic_L_per_h: float
    cl_total_L_per_h: float
    vd_predicted_L: float
    t_half_predicted_h: float
    elimination_route: str  # 'renal', 'hepatic', or 'mixed'
    confidence: float  # 0-1
    model_used: str


def _predict_cl_hepatic(logP: float, MW: float, fup: float, PSA: float) -> float:
    """Predict hepatic clearance (L/h) from physicochemical properties."""
    base_cl = 5.0 * fup
    logP_factor = 1 + 0.3 * max(logP - 2.0, 0)
    mw_factor = max(0.3, 1.0 - (MW - 300) / 1000)
    PSA_factor = max(0.3, 1.0 - PSA / 300)
    return max(0.01, min(150.0, base_cl * logP_factor * mw_factor * PSA_factor))


def _predict_cl_renal(
    MW: float, fup: float, pka: float, drug_type: str = "neutral"
) -> float:
    """Predict renal clearance (L/h) from physicochemical properties."""
    base_renal = 1.5 * fup
    mw_penalty = max(0.0, 1.0 - (MW - 300) / 500)
    return max(0.0, min(30.0, base_renal * mw_penalty))


def _predict_vd(logP: float, MW: float, fup: float, PSA: float) -> float:
    """Predict volume of distribution (L) from physicochemical properties."""
    base_vd = 30.0 / fup
    logP_boost = 1 + 0.5 * max(logP - 1.0, 0)
    PSA_penalty = max(0.2, 1.0 - PSA / 200)
    return max(1.0, min(2000.0, base_vd * logP_boost * PSA_penalty))


def predict_clearance(
    drug_name: str,
    logP: float,
    MW: float,
    fup: float,
    pka: float = 7.0,
    PSA: float = 60.0,
    drug_type: str = "neutral",
) -> ClearancePredictionResult:
    """
    Predict clearance parameters for a compound from physicochemical descriptors.

    Parameters
    ----------
    drug_name : str
        Name of the drug/compound.
    logP : float
        Lipophilicity (log partition coefficient octanol/water).
    MW : float
        Molecular weight (g/mol).
    fup : float
        Fraction unbound in plasma (0 < fup <= 1).
    pka : float, optional
        pKa value. Default 7.0.
    PSA : float, optional
        Polar surface area (Å²). Default 60.0.
    drug_type : str, optional
        Drug ionization type ('neutral', 'acid', 'base'). Default 'neutral'.

    Returns
    -------
    ClearancePredictionResult

    Raises
    ------
    ValueError
        If fup <= 0, fup > 1, or MW <= 0.
    """
    if fup <= 0 or fup > 1:
        raise ValueError(f"fup must be in (0, 1], got {fup}")
    if MW <= 0:
        raise ValueError(f"MW must be positive, got {MW}")

    cl_hepatic = _predict_cl_hepatic(logP, MW, fup, PSA)
    cl_renal = _predict_cl_renal(MW, fup, pka, drug_type)
    cl_total = cl_hepatic + cl_renal

    vd = _predict_vd(logP, MW, fup, PSA)
    t_half = 0.693 * vd / cl_total

    if cl_renal > 0.6 * cl_total:
        elimination_route = "renal"
    elif cl_hepatic > 0.6 * cl_total:
        elimination_route = "hepatic"
    else:
        elimination_route = "mixed"

    confidence = 0.9 if (100 <= MW <= 500 and 0 <= logP <= 5) else 0.6

    return ClearancePredictionResult(
        drug_name=drug_name,
        logP=logP,
        MW=MW,
        pka=pka,
        fup=fup,
        PSA=PSA,
        cl_predicted_L_per_h=cl_total,
        cl_renal_L_per_h=cl_renal,
        cl_hepatic_L_per_h=cl_hepatic,
        cl_total_L_per_h=cl_total,
        vd_predicted_L=vd,
        t_half_predicted_h=t_half,
        elimination_route=elimination_route,
        confidence=confidence,
        model_used="linear_physicochemical_v1",
    )


def batch_clearance_predict(
    compounds: list[dict],
) -> list[ClearancePredictionResult]:
    """
    Predict clearance for multiple compounds.

    Parameters
    ----------
    compounds : list[dict]
        Each dict must have keys: drug_name, logP, MW, fup.
        Optional keys: pka, PSA, drug_type.

    Returns
    -------
    list[ClearancePredictionResult]
        Results sorted by cl_total_L_per_h descending.
        Invalid compounds are skipped with a warning.
    """
    results: list[ClearancePredictionResult] = []

    for compound in compounds:
        try:
            result = predict_clearance(
                drug_name=compound["drug_name"],
                logP=compound["logP"],
                MW=compound["MW"],
                fup=compound["fup"],
                pka=compound.get("pka", 7.0),
                PSA=compound.get("PSA", 60.0),
                drug_type=compound.get("drug_type", "neutral"),
            )
            results.append(result)
        except (ValueError, KeyError) as exc:
            logger.warning(
                "Skipping compound %r due to error: %s",
                compound.get("drug_name", "<unknown>"),
                exc,
            )

    results.sort(key=lambda r: r.cl_total_L_per_h, reverse=True)
    return results
