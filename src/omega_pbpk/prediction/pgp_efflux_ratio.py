"""Phase 552 - P-glycoprotein Efflux Ratio Model.

Predicts P-gp (MDR1) efflux ratio from physicochemical properties
and classifies CNS penetration impact.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PgpEffluxResult:
    drug_name: str
    logP: float
    mw_Da: float
    hbd: int
    psa_A2: float
    predicted_efflux_ratio: float
    pgp_substrate_probability: float
    net_flux_class: str
    cns_penetration_impact: str
    clinical_concern: bool
    notes: str


def _sigmoid(x: float) -> float:
    """Logistic sigmoid function."""
    if x < -500:
        return 0.0
    if x > 500:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def predict_pgp_efflux(
    drug_name: str,
    logP: float,
    mw_Da: float,
    hbd: int,
    psa_A2: float,
) -> PgpEffluxResult:
    """Predict P-gp efflux ratio from physicochemical properties."""
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")
    if hbd < 0:
        raise ValueError("hbd must be >= 0")
    if psa_A2 < 0:
        raise ValueError("psa_A2 must be >= 0")

    # P-gp substrate probability
    prob_raw = _sigmoid(0.003 * mw_Da + 0.01 * psa_A2 + 0.2 * hbd - 2.0)
    prob = max(0.0, min(1.0, prob_raw))

    # Efflux ratio
    er_raw = 1.0 + prob * 5.0 * (1.0 + 0.1 * max(0.0, logP - 2.0))
    er = max(1.0, min(20.0, er_raw))

    # Classification
    net_flux_class = "efflux" if er > 2.0 else "passive"

    if er > 6.0:
        cns_impact = "impaired"
    elif er > 3.0:
        cns_impact = "moderate"
    else:
        cns_impact = "good"

    clinical_concern = er > 4.0

    notes = f"P-gp prob={prob:.3f}, ER={er:.2f}, MW={mw_Da}, HBD={hbd}, PSA={psa_A2}"

    return PgpEffluxResult(
        drug_name=drug_name,
        logP=logP,
        mw_Da=mw_Da,
        hbd=hbd,
        psa_A2=psa_A2,
        predicted_efflux_ratio=er,
        pgp_substrate_probability=prob,
        net_flux_class=net_flux_class,
        cns_penetration_impact=cns_impact,
        clinical_concern=clinical_concern,
        notes=notes,
    )


def screen_pgp_efflux(compounds: list) -> list:
    """Screen multiple compounds for P-gp efflux, sorted by ER descending."""
    results = [predict_pgp_efflux(**comp) for comp in compounds]
    results.sort(key=lambda r: r.predicted_efflux_ratio, reverse=True)
    return results
