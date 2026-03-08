"""Phase 897 — Salivary Drug Excretion.

Models drug excretion into saliva using Henderson-Hasselbalch ionization
equilibrium.  Useful for TDM, workplace drug testing, and paediatric
sampling where venepuncture is impractical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SalivaryExcretionResult:
    """Immutable result of a salivary excretion prediction."""

    drug_name: str
    pka: float
    logp: float
    ionization_type: str
    plasma_ph: float
    saliva_ph: float
    saliva_plasma_ratio: float
    predicted_saliva_conc_mg_L: float
    plasma_conc_input_mg_L: float
    fu_plasma_estimate: float
    tdm_suitable: bool
    saliva_sampling_recommendation: str
    notes: str


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

_PLASMA_PH = 7.4
_VALID_IONIZATION = {"acid", "base", "neutral"}


def _estimate_fu_plasma(logp: float) -> float:
    """Estimate free plasma fraction from logP (sigmoid approximation)."""
    return max(0.01, min(1.0, 1.0 / (1.0 + math.pow(10.0, logp - 2.5))))


def _sp_ratio(
    pka: float,
    ionization_type: str,
    fluid_ph: float,
    plasma_ph: float = _PLASMA_PH,
) -> float:
    """Henderson-Hasselbalch saliva/plasma (or sweat/plasma) ratio."""
    if ionization_type == "base":
        numerator = 1.0 + math.pow(10.0, pka - fluid_ph)
        denominator = 1.0 + math.pow(10.0, pka - plasma_ph)
    elif ionization_type == "acid":
        numerator = 1.0 + math.pow(10.0, fluid_ph - pka)
        denominator = 1.0 + math.pow(10.0, plasma_ph - pka)
    else:  # neutral
        return 1.0

    if denominator == 0.0:
        return 1.0
    return numerator / denominator


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_salivary_excretion(
    drug_name: str,
    pka: float,
    logp: float,
    ionization_type: str = "base",
    plasma_conc_mg_L: float = 1.0,
    saliva_ph: float = 6.8,
    fu_plasma: float | None = None,
) -> SalivaryExcretionResult:
    """Predict salivary drug concentration and TDM suitability.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    pka:
        Acid/base dissociation constant (0–14).
    logp:
        Octanol-water partition coefficient.
    ionization_type:
        One of "acid", "base", or "neutral".
    plasma_conc_mg_L:
        Total plasma concentration (mg/L).  Must be > 0.
    saliva_ph:
        Salivary pH (typically 6.8–7.4).
    fu_plasma:
        Free fraction in plasma.  Estimated from logP if not provided.

    Returns
    -------
    SalivaryExcretionResult
    """
    # --- Validation ---
    if not (0.0 <= pka <= 14.0):
        raise ValueError(f"pka must be in [0, 14], got {pka}")
    if plasma_conc_mg_L <= 0.0:
        raise ValueError(f"plasma_conc_mg_L must be > 0, got {plasma_conc_mg_L}")
    if ionization_type not in _VALID_IONIZATION:
        raise ValueError(
            f"ionization_type must be one of {_VALID_IONIZATION}, got {ionization_type!r}"
        )

    plasma_ph = _PLASMA_PH

    # --- Free fraction ---
    fu_plasma_estimate = fu_plasma if fu_plasma is not None else _estimate_fu_plasma(logp)

    # --- S/P ratio ---
    raw_ratio = _sp_ratio(pka, ionization_type, saliva_ph, plasma_ph)
    saliva_plasma_ratio = max(0.01, min(20.0, raw_ratio))

    # --- Predicted salivary concentration ---
    predicted_saliva_conc_mg_L = saliva_plasma_ratio * plasma_conc_mg_L * fu_plasma_estimate

    # --- TDM suitability ---
    tdm_suitable = 0.5 <= saliva_plasma_ratio <= 2.0

    # --- Recommendation ---
    if tdm_suitable:
        recommendation = "Saliva suitable for TDM — collect unstimulated saliva at steady state"
    elif saliva_plasma_ratio > 2.0:
        recommendation = (
            "High salivary accumulation — not ideal for TDM but useful for exposure monitoring"
        )
    else:
        recommendation = "Low salivary concentration — insufficient for reliable TDM"

    # --- Notes ---
    if ionization_type == "base" and pka > 7.0:
        notes = (
            "Basic drug with pKa>7: elevated salivary concentration due to "
            "ion trapping in acidic saliva"
        )
    elif ionization_type == "acid":
        notes = "Acidic drug: saliva conc lower than plasma"
    else:
        notes = "Neutral drug: saliva mirrors free plasma concentration"

    return SalivaryExcretionResult(
        drug_name=drug_name,
        pka=pka,
        logp=logp,
        ionization_type=ionization_type,
        plasma_ph=plasma_ph,
        saliva_ph=saliva_ph,
        saliva_plasma_ratio=saliva_plasma_ratio,
        predicted_saliva_conc_mg_L=predicted_saliva_conc_mg_L,
        plasma_conc_input_mg_L=plasma_conc_mg_L,
        fu_plasma_estimate=fu_plasma_estimate,
        tdm_suitable=tdm_suitable,
        saliva_sampling_recommendation=recommendation,
        notes=notes,
    )


def screen_salivary_tdm(candidates: list[dict]) -> list[SalivaryExcretionResult]:
    """Screen a list of drug candidates for salivary TDM suitability.

    Parameters
    ----------
    candidates:
        Each dict must contain "name", "pka", "logp".
        Optional key "ionization_type" defaults to "neutral".

    Returns
    -------
    List of :class:`SalivaryExcretionResult` sorted by *saliva_plasma_ratio*
    descending.
    """
    results: list[SalivaryExcretionResult] = []
    for c in candidates:
        result = predict_salivary_excretion(
            drug_name=c["name"],
            pka=c["pka"],
            logp=c["logp"],
            ionization_type=c.get("ionization_type", "neutral"),
        )
        results.append(result)
    return sorted(results, key=lambda r: r.saliva_plasma_ratio, reverse=True)
