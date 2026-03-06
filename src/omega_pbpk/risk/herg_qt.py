"""hERG channel inhibition and QT prolongation risk prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["HERGRiskResult", "predict_herg_risk", "qtc_prolongation_risk_table"]


@dataclass(frozen=True)
class HERGRiskResult:
    drug_name: str
    logP: float
    pka_basic: float
    MW: float
    risk_score: float
    herg_inhibition_pct: float
    delta_qtc_ms: float
    risk_category: str
    risk_factors: list[str]


def _herg_physicochemical_score(logP: float, pka_basic: float, MW: float) -> float:
    """Return 0-10 composite physicochemical hERG risk score.

    Based on Leeson & Springthorpe 2007 + Redfern 2003 rules.
    """
    basic_score = 3.0 if pka_basic > 7.0 else (1.5 if pka_basic > 5.0 else 0.0)
    logP_score = float(np.clip((logP - 1.0) * 1.0, 0.0, 3.0))
    mw_score = 1.0 if 300 < MW < 600 else 0.0
    total = float(np.clip(basic_score + logP_score + mw_score, 0.0, 10.0))
    return total


def predict_herg_risk(
    drug_name: str,
    logP: float,
    pka_basic: float,
    MW: float,
    cmax_free_mg_L: float | None = None,
) -> HERGRiskResult:
    """Predict hERG channel inhibition and QT prolongation risk."""
    risk_score = _herg_physicochemical_score(logP, pka_basic, MW)
    herg_inhibition_pct = float(np.clip(risk_score * 8.0, 0.0, 95.0))

    if cmax_free_mg_L is not None:
        conc_factor = math.log10(max(cmax_free_mg_L, 0.001) + 1) / math.log10(2)
        herg_inhibition_pct = float(np.clip(herg_inhibition_pct * conc_factor, 0.0, 95.0))

    delta_qtc_ms = herg_inhibition_pct * 0.5

    if delta_qtc_ms < 5:
        risk_category = "low"
    elif delta_qtc_ms < 20:
        risk_category = "moderate"
    elif delta_qtc_ms < 60:
        risk_category = "high"
    else:
        risk_category = "very_high"

    risk_factors: list[str] = []
    if pka_basic > 7.0:
        risk_factors.append(f"Basic nitrogen (pKa={pka_basic:.1f})")
    if logP > 3.0:
        risk_factors.append(f"High lipophilicity (logP={logP:.1f})")
    if 300 < MW < 600:
        risk_factors.append("Optimal hERG MW range")
    if cmax_free_mg_L is not None and cmax_free_mg_L > 1.0:
        risk_factors.append("High free drug exposure")

    return HERGRiskResult(
        drug_name=drug_name,
        logP=logP,
        pka_basic=pka_basic,
        MW=MW,
        risk_score=risk_score,
        herg_inhibition_pct=herg_inhibition_pct,
        delta_qtc_ms=delta_qtc_ms,
        risk_category=risk_category,
        risk_factors=risk_factors,
    )


def qtc_prolongation_risk_table(
    drug_name: str,
    logP: float,
    pka_basic: float,
    MW: float,
) -> list[dict]:
    """Return risk data for a range of Cmax_free values."""
    results = []
    for cmax in [0.01, 0.1, 1.0, 10.0]:
        r = predict_herg_risk(drug_name, logP, pka_basic, MW, cmax_free_mg_L=cmax)
        results.append(
            {
                "cmax_free_mg_L": cmax,
                "herg_inhibition_pct": r.herg_inhibition_pct,
                "delta_qtc_ms": r.delta_qtc_ms,
                "risk_category": r.risk_category,
            }
        )
    return results
