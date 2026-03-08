"""Phase 898 — Sweat Drug Excretion.

Models drug excretion via sweat — relevant for wearable biosensors and
occupational drug-exposure monitoring.  Sweat pH (4.0–6.8) causes ion
trapping for basic drugs; eccrine glands preferentially excrete
water-soluble molecules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweatExcretionResult:
    """Immutable result of a sweat excretion prediction."""

    drug_name: str
    pka: float
    logp: float
    ionization_type: str
    sweat_ph: float
    sweat_plasma_ratio: float
    predicted_sweat_conc_ug_mL: float
    plasma_conc_input_mg_L: float
    sweat_rate_mL_h: float
    drug_flux_ug_per_h: float
    detection_window_h: float
    patch_test_suitable: bool
    notes: str


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

_PLASMA_PH = 7.4
_VALID_IONIZATION = {"acid", "base", "neutral"}


def _estimate_fu_plasma(logp: float) -> float:
    """Estimate free plasma fraction from logP (sigmoid approximation)."""
    return max(0.01, min(1.0, 1.0 / (1.0 + math.pow(10.0, logp - 2.5))))


def _sweat_plasma_ratio(
    pka: float,
    ionization_type: str,
    sweat_ph: float,
    plasma_ph: float = _PLASMA_PH,
) -> float:
    """Henderson-Hasselbalch sweat/plasma ratio."""
    if ionization_type == "base":
        numerator = 1.0 + math.pow(10.0, pka - sweat_ph)
        denominator = 1.0 + math.pow(10.0, pka - plasma_ph)
    elif ionization_type == "acid":
        numerator = 1.0 + math.pow(10.0, sweat_ph - pka)
        denominator = 1.0 + math.pow(10.0, plasma_ph - pka)
    else:  # neutral
        return 1.0

    if denominator == 0.0:
        return 1.0
    return numerator / denominator


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_sweat_excretion(
    drug_name: str,
    pka: float,
    logp: float,
    ionization_type: str = "base",
    plasma_conc_mg_L: float = 1.0,
    sweat_ph: float = 5.5,
    sweat_rate_mL_h: float = 0.5,
    fu_plasma: float | None = None,
    lod_ug_mL: float = 0.01,
) -> SweatExcretionResult:
    """Predict sweat drug concentration and patch-test suitability.

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
    sweat_ph:
        Sweat pH (typically 4.0–6.8; default 5.5).
    sweat_rate_mL_h:
        Eccrine sweat production rate (mL/h).  Must be > 0.
    fu_plasma:
        Free fraction in plasma.  Estimated from logP if not provided.
    lod_ug_mL:
        Limit of detection for sweat assay (µg/mL; default 0.01).

    Returns
    -------
    SweatExcretionResult
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
    if sweat_rate_mL_h <= 0.0:
        raise ValueError(f"sweat_rate_mL_h must be > 0, got {sweat_rate_mL_h}")

    plasma_ph = _PLASMA_PH

    # --- Free fraction ---
    fu_plasma_estimate = fu_plasma if fu_plasma is not None else _estimate_fu_plasma(logp)

    # --- Sweat/plasma ratio ---
    raw_ratio = _sweat_plasma_ratio(pka, ionization_type, sweat_ph, plasma_ph)
    ratio = max(0.01, min(100.0, raw_ratio))

    # --- Predicted sweat concentration (µg/mL) ---
    # plasma_conc_mg_L * 1000 → µg/mL equivalent; × ratio × fu
    predicted_sweat_conc_ug_mL = ratio * plasma_conc_mg_L * fu_plasma_estimate * 1000.0

    # --- Drug flux (µg/h) ---
    drug_flux_ug_per_h = predicted_sweat_conc_ug_mL * sweat_rate_mL_h

    # --- Detection window (h) ---
    raw_window = (plasma_conc_mg_L * 1000.0 * ratio) / lod_ug_mL / 2.0
    detection_window_h = max(0.0, min(720.0, raw_window))

    # --- Patch test suitability ---
    patch_test_suitable = drug_flux_ug_per_h >= 0.1

    # --- Notes ---
    if patch_test_suitable:
        notes = "Sweat patch suitable for monitoring. Consider 7-day patch collection."
    elif ratio > 5.0:
        notes = "High sweat/plasma ratio — sweat monitoring viable for exposure assessment"
    else:
        notes = "Low sweat drug concentration — sweat monitoring may be challenging"

    return SweatExcretionResult(
        drug_name=drug_name,
        pka=pka,
        logp=logp,
        ionization_type=ionization_type,
        sweat_ph=sweat_ph,
        sweat_plasma_ratio=ratio,
        predicted_sweat_conc_ug_mL=predicted_sweat_conc_ug_mL,
        plasma_conc_input_mg_L=plasma_conc_mg_L,
        sweat_rate_mL_h=sweat_rate_mL_h,
        drug_flux_ug_per_h=drug_flux_ug_per_h,
        detection_window_h=detection_window_h,
        patch_test_suitable=patch_test_suitable,
        notes=notes,
    )


def screen_sweat_excretion(candidates: list[dict]) -> list[SweatExcretionResult]:
    """Screen a list of drug candidates for sweat-based monitoring suitability.

    Parameters
    ----------
    candidates:
        Each dict must contain "name", "pka", "logp".
        Optional key "ionization_type" defaults to "neutral".

    Returns
    -------
    List of :class:`SweatExcretionResult` sorted by *sweat_plasma_ratio*
    descending.
    """
    results: list[SweatExcretionResult] = []
    for c in candidates:
        result = predict_sweat_excretion(
            drug_name=c["name"],
            pka=c["pka"],
            logp=c["logp"],
            ionization_type=c.get("ionization_type", "neutral"),
        )
        results.append(result)
    return sorted(results, key=lambda r: r.sweat_plasma_ratio, reverse=True)
