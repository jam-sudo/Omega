"""Plasma protein binding prediction from physicochemical descriptors."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PPBResult:
    drug_name: str
    fup: float              # fraction unbound in plasma (0–1)
    ppb_pct: float          # protein binding % (100*(1-fup))
    primary_protein: str    # HSA / AAG / mixed
    log_ka: float           # log10 association constant
    confidence: float
    classification: str     # low (<10%) / moderate (10-75%) / high (>75%)


def _predict_fup(
    logP: float,
    MW: float,
    PSA: float,
    pka_acid: float | None = None,
    pka_basic: float | None = None,
) -> tuple[float, str, float]:
    """Predict fup from descriptors.

    Empirical model based on Lobell & Sivarajah (2003) and Austin (2002).
    Higher logP → more HSA binding → lower fup.
    Basic drugs bind AAG; acidic drugs bind HSA.
    """
    # Base logit(fup) from logP (main driver)
    # logit(fup) = intercept - slope*logP
    intercept = 1.5
    slope_logP = 0.45
    slope_MW = 0.003
    slope_PSA = -0.008

    logit_fup = intercept - slope_logP * logP - slope_MW * MW + slope_PSA * PSA

    # Ionisation correction
    if pka_acid is not None and 3.0 <= pka_acid <= 9.0:
        logit_fup -= 0.3  # acidic drugs bind more to HSA
    if pka_basic is not None and 7.0 <= pka_basic <= 11.0:
        logit_fup += 0.2  # basic drugs bind more to AAG (lower apparent PPB)

    # Convert logit to probability (fup)
    fup = 1.0 / (1.0 + math.exp(-logit_fup))
    fup = float(min(max(fup, 0.001), 0.999))

    # Primary protein
    if pka_basic is not None and pka_basic > 7.0:
        protein = "AAG"
    elif pka_acid is not None and pka_acid < 7.0:
        protein = "HSA"
    else:
        protein = "HSA" if logP > 2.0 else "mixed"

    # log Ka (association constant, L/mol)
    # Typical range 3–6 for drugs; higher logP → higher Ka
    log_ka = 3.5 + 0.3 * max(logP, 0.0) - 0.001 * MW

    return fup, protein, float(log_ka)


def predict_ppb(
    drug_name: str,
    logP: float,
    MW: float,
    PSA: float = 60.0,
    pka_acid: float | None = None,
    pka_basic: float | None = None,
) -> PPBResult:
    """Predict plasma protein binding.

    Parameters
    ----------
    logP : float
        Lipophilicity.
    MW : float
        Molecular weight (g/mol).
    PSA : float
        Polar surface area (Å²).
    pka_acid : float or None
        Acidic pKa if applicable.
    pka_basic : float or None
        Basic pKa if applicable.
    """
    fup, protein, log_ka = _predict_fup(logP, MW, PSA, pka_acid, pka_basic)
    ppb_pct = 100.0 * (1.0 - fup)

    if ppb_pct < 10.0:
        classification = "low"
    elif ppb_pct < 75.0:
        classification = "moderate"
    else:
        classification = "high"

    # Confidence: higher when descriptor values in typical drug-like range
    drug_like = (200 <= MW <= 600) and (-1 <= logP <= 5) and (PSA <= 150)
    confidence = 0.75 if drug_like else 0.55

    return PPBResult(
        drug_name=drug_name,
        fup=fup,
        ppb_pct=ppb_pct,
        primary_protein=protein,
        log_ka=log_ka,
        confidence=confidence,
        classification=classification,
    )


def ppb_sensitivity(
    drug_name: str,
    logP: float,
    MW: float,
    PSA: float = 60.0,
    delta_logP: float = 1.0,
) -> dict:
    """Compute how fup changes with logP perturbation."""
    base = predict_ppb(drug_name, logP, MW, PSA)
    high = predict_ppb(drug_name, logP + delta_logP, MW, PSA)
    low = predict_ppb(drug_name, logP - delta_logP, MW, PSA)
    return {
        "base": base,
        "high_logP": high,
        "low_logP": low,
        "dfup_dlogP": (high.fup - low.fup) / (2 * delta_logP),
    }


__all__ = [
    "PPBResult",
    "predict_ppb",
    "ppb_sensitivity",
]
