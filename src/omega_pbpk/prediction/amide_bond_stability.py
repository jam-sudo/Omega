"""Phase 213 — Amide bond stability prediction under physiological conditions.

Predicts chemical stability of amide bonds in drug candidates via pseudo-first-order
hydrolysis kinetics with pH, temperature, and flanking-group effects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_K_BASE = 1e-6  # /h, hydrolysis rate at pH 7.4, 37 °C
_PH_REF = 7.4
_T_REF_K = 310.15  # 37 °C in Kelvin
_EA_J_PER_MOL = 80_000.0  # Arrhenius activation energy (80 kJ/mol)
_R_J_PER_MOL_K = 8.314  # universal gas constant

_VALID_FLANKING = {"electron_withdrawing", "electron_donating", "none"}
_FLANKING_FACTOR = {
    "electron_withdrawing": 5.0,
    "electron_donating": 0.3,
    "none": 1.0,
}

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AmideStabilityResult:
    """Immutable result for a single amide stability prediction."""

    drug_name: str
    ph: float
    temperature_C: float
    flanking_groups: str
    k_hydrolysis_per_h: float
    t_half_h: float
    pct_remaining_24h: float
    pct_remaining_7d: float
    stability_class: str
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_inputs(temperature_C: float, flanking_groups: str) -> None:
    """Raise ValueError for invalid inputs."""
    temp_K = temperature_C + 273.15
    if temp_K <= 0:
        raise ValueError(
            f"Temperature {temperature_C} °C corresponds to a non-positive Kelvin value."
        )
    if flanking_groups not in _VALID_FLANKING:
        raise ValueError(
            f"Invalid flanking_groups '{flanking_groups}'. "
            f"Must be one of {sorted(_VALID_FLANKING)}."
        )


def _stability_class(t_half_h: float) -> str:
    if t_half_h > 10_000:
        return "highly_stable"
    if t_half_h >= 1_000:
        return "stable"
    if t_half_h >= 100:
        return "labile"
    return "very_labile"


def _compute_k(ph: float, temperature_C: float, flanking_groups: str) -> float:
    """Return pseudo-first-order hydrolysis rate constant (1/h)."""
    # pH effect: symmetric acid-base catalysis
    ph_factor = 10 ** (0.5 * abs(ph - _PH_REF))

    # Arrhenius temperature correction
    temp_K = temperature_C + 273.15
    arrhenius_factor = math.exp(-(_EA_J_PER_MOL / _R_J_PER_MOL_K) * (1.0 / temp_K - 1.0 / _T_REF_K))

    flanking_factor = _FLANKING_FACTOR[flanking_groups]

    return _K_BASE * ph_factor * arrhenius_factor * flanking_factor


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_amide_stability(
    drug_name: str,
    ph: float,
    temperature_C: float,
    flanking_groups: str = "none",
    notes: str = "",
) -> AmideStabilityResult:
    """Predict amide bond stability for a drug candidate.

    Parameters
    ----------
    drug_name:
        Identifier for the drug / compound.
    ph:
        Physiological or assay pH value.
    temperature_C:
        Temperature in degrees Celsius.
    flanking_groups:
        Electronic character of substituents adjacent to the amide bond.
        One of ``"electron_withdrawing"``, ``"electron_donating"``, or ``"none"``.
    notes:
        Optional free-text annotation.

    Returns
    -------
    AmideStabilityResult
    """
    _validate_inputs(temperature_C, flanking_groups)

    k = _compute_k(ph, temperature_C, flanking_groups)
    t_half = math.log(2) / k

    pct_24h = 100.0 * math.exp(-k * 24.0)
    pct_7d = 100.0 * math.exp(-k * 7 * 24.0)

    cls = _stability_class(t_half)

    return AmideStabilityResult(
        drug_name=drug_name,
        ph=ph,
        temperature_C=temperature_C,
        flanking_groups=flanking_groups,
        k_hydrolysis_per_h=k,
        t_half_h=t_half,
        pct_remaining_24h=pct_24h,
        pct_remaining_7d=pct_7d,
        stability_class=cls,
        notes=notes,
    )


def compare_ph_stability(
    drug_name: str,
    ph_values: list[float],
    temperature_C: float,
    flanking_groups: str = "none",
) -> list[AmideStabilityResult]:
    """Predict amide stability across multiple pH values.

    Parameters
    ----------
    drug_name:
        Identifier for the drug / compound.
    ph_values:
        List of pH values to evaluate.
    temperature_C:
        Temperature in degrees Celsius.
    flanking_groups:
        Electronic character of flanking substituents.

    Returns
    -------
    List of :class:`AmideStabilityResult` sorted by *t_half_h* descending
    (most stable first).
    """
    results = [
        predict_amide_stability(
            drug_name=drug_name,
            ph=ph,
            temperature_C=temperature_C,
            flanking_groups=flanking_groups,
        )
        for ph in ph_values
    ]
    results.sort(key=lambda r: r.t_half_h, reverse=True)
    return results
