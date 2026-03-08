"""Sweat drug excretion model — Phase 550.

Models drug excretion through sweat glands for forensic/occupational
exposure assessment and therapeutic drug monitoring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SweatExcretionResult:
    """Result of sweat drug excretion prediction."""

    drug_name: str
    plasma_concentration_mg_L: float
    sweat_ph: float
    drug_type: str
    sweat_plasma_ratio: float
    sweat_concentration_mg_L: float
    daily_sweat_excretion_mg: float
    detection_window_h: float
    excretion_class: str
    notes: str


_VALID_DRUG_TYPES = {"acid", "base", "amphoteric", "neutral"}
_DETECTION_LIMIT = 0.001  # mg/L
_DEFAULT_HALF_LIFE = 12.0  # hours


def _ionized_fraction(pKa: float, drug_type: str, ph: float) -> float:
    """Compute ionized fraction using Henderson-Hasselbalch."""
    if drug_type == "acid":
        return 1.0 / (1.0 + 10.0 ** (pKa - ph))
    elif drug_type in ("base", "amphoteric"):
        return 1.0 / (1.0 + 10.0 ** (ph - pKa))
    else:
        # neutral
        return 0.0


def predict_sweat_excretion(
    drug_name: str,
    plasma_concentration_mg_L: float,
    pKa: float,
    drug_type: str,
    sweat_ph: float = 6.5,
    plasma_ph: float = 7.4,
    fup: float = 0.5,
    sweat_rate_L_per_day: float = 1.0,
) -> SweatExcretionResult:
    """Predict drug excretion through sweat.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    plasma_concentration_mg_L : float
        Plasma drug concentration in mg/L.
    pKa : float
        Acid dissociation constant.
    drug_type : str
        One of "acid", "base", "amphoteric", "neutral".
    sweat_ph : float
        Sweat pH (default 6.5).
    plasma_ph : float
        Plasma pH (default 7.4).
    fup : float
        Fraction unbound in plasma (0-1).
    sweat_rate_L_per_day : float
        Daily sweat production rate in L/day.

    Returns
    -------
    SweatExcretionResult
    """
    if drug_type not in _VALID_DRUG_TYPES:
        raise ValueError(
            f"Invalid drug_type '{drug_type}'. Must be one of {sorted(_VALID_DRUG_TYPES)}."
        )

    if plasma_concentration_mg_L < 0:
        raise ValueError("plasma_concentration_mg_L must be >= 0")

    if not 0.0 <= fup <= 1.0:
        raise ValueError("fup must be between 0 and 1")

    # Calculate S/P ratio
    if drug_type == "neutral":
        sp_ratio = fup * 1.0
    else:
        ion_frac_sweat = _ionized_fraction(pKa, drug_type, sweat_ph)
        ion_frac_plasma = _ionized_fraction(pKa, drug_type, plasma_ph)
        sp_ratio = fup * (1.0 + ion_frac_sweat) / (1.0 + ion_frac_plasma)

    # Clamp S/P ratio
    sp_ratio = max(0.01, min(10.0, sp_ratio))

    sweat_concentration = plasma_concentration_mg_L * sp_ratio
    daily_excretion = sweat_concentration * sweat_rate_L_per_day

    # Detection window assuming first-order decay
    if sweat_concentration > _DETECTION_LIMIT:
        detection_window = (_DEFAULT_HALF_LIFE / math.log(2)) * math.log(
            sweat_concentration / _DETECTION_LIMIT
        )
    else:
        detection_window = 0.0

    # Excretion class
    if sp_ratio >= 1.0:
        excretion_class = "high"
    elif sp_ratio >= 0.3:
        excretion_class = "moderate"
    else:
        excretion_class = "low"

    return SweatExcretionResult(
        drug_name=drug_name,
        plasma_concentration_mg_L=plasma_concentration_mg_L,
        sweat_ph=sweat_ph,
        drug_type=drug_type,
        sweat_plasma_ratio=sp_ratio,
        sweat_concentration_mg_L=sweat_concentration,
        daily_sweat_excretion_mg=daily_excretion,
        detection_window_h=detection_window,
        excretion_class=excretion_class,
        notes=f"Sweat excretion model for {drug_type} drug at sweat pH {sweat_ph}",
    )


def compare_sweat_excretion(
    compounds: list,  # type: ignore[type-arg]
) -> list:  # type: ignore[type-arg]
    """Compare sweat excretion across multiple compounds.

    Parameters
    ----------
    compounds : list[dict]
        Each dict must have keys matching predict_sweat_excretion parameters.

    Returns
    -------
    list[SweatExcretionResult]
        Sorted by sweat_plasma_ratio descending.
    """
    results = [predict_sweat_excretion(**c) for c in compounds]
    results.sort(key=lambda r: r.sweat_plasma_ratio, reverse=True)
    return results
