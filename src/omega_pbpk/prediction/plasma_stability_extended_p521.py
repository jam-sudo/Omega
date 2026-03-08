"""Phase 521 — Extended plasma stability model.

Degradation pathways:
  1. Esterase hydrolysis (if ester_bond=True)
  2. Oxidative degradation (logP-dependent)
  3. Non-specific plasma degradation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class PlasmaStabilityExtendedResult:
    """Result from extended plasma stability prediction."""

    compound_name: str
    logP: float
    ester_bond: bool
    esterase_activity: float
    k_ester_per_h: float
    k_ox_per_h: float
    k_nonspec_per_h: float
    k_total_per_h: float
    times_min: list[float] = field(default_factory=list)
    fraction_remaining: list[float] = field(default_factory=list)
    t50_min: float = 0.0
    t90_min: float = 0.0
    stability_class: str = ""
    dominant_pathway: str = ""
    temperature_celsius: float = 37.0
    k_total_at_temp_per_h: float = 0.0
    notes: str = ""


def _stability_class(t50_min: float) -> str:
    if t50_min > 120.0:
        return "stable"
    if t50_min >= 30.0:
        return "moderate"
    return "unstable"


def _dominant_pathway(k_ester: float, k_ox: float, k_nonspec: float) -> str:
    if k_ester >= k_ox and k_ester >= k_nonspec:
        return "esterase"
    if k_ox >= k_nonspec:
        return "oxidative"
    return "nonspecific"


def predict_plasma_stability_extended(
    compound_name: str,
    logP: float,
    ester_bond: bool = False,
    esterase_activity: float = 1.0,
    temperature_celsius: float = 37.0,
) -> PlasmaStabilityExtendedResult:
    """Predict extended plasma stability for a compound.

    Parameters
    ----------
    compound_name:
        Identifier for the compound.
    logP:
        Lipophilicity (log partition coefficient).
    ester_bond:
        Whether the compound contains an ester bond susceptible to hydrolysis.
    esterase_activity:
        Relative esterase activity (1.0 = normal human plasma). Must be > 0.
    temperature_celsius:
        Incubation temperature in Celsius. Must be between 4 and 60.

    Returns
    -------
    PlasmaStabilityExtendedResult
    """
    if esterase_activity <= 0:
        raise ValueError("esterase_activity must be > 0")
    if not (4.0 <= temperature_celsius <= 60.0):
        raise ValueError("temperature_celsius must be between 4 and 60")

    # Pathway rate constants [1/h]
    k_ester = esterase_activity * 0.01 if ester_bond else 0.0
    k_ox = 0.001 * (1.0 + 0.1 * logP)
    k_nonspec = 0.002

    k_total = k_ester + k_ox + k_nonspec

    # Stability metrics
    t50_min = 0.693 / k_total * 60.0  # convert h to min
    t90_min = 2.303 / k_total * 60.0

    # Profile 0–120 min in steps of 10 min
    times_min: list[float] = [float(t) for t in range(0, 130, 10)]
    fraction_remaining: list[float] = [math.exp(-k_total * (t / 60.0)) for t in times_min]

    # Temperature correction (Q10 = 2)
    k_total_at_temp = k_total * (2.0 ** ((temperature_celsius - 37.0) / 10.0))

    notes = (
        f"k_total={k_total:.4f} /h; t50={t50_min:.1f} min; "
        f"T_effect factor={2.0 ** ((temperature_celsius - 37) / 10):.3f}"
    )

    return PlasmaStabilityExtendedResult(
        compound_name=compound_name,
        logP=logP,
        ester_bond=ester_bond,
        esterase_activity=esterase_activity,
        k_ester_per_h=k_ester,
        k_ox_per_h=k_ox,
        k_nonspec_per_h=k_nonspec,
        k_total_per_h=k_total,
        times_min=times_min,
        fraction_remaining=fraction_remaining,
        t50_min=t50_min,
        t90_min=t90_min,
        stability_class=_stability_class(t50_min),
        dominant_pathway=_dominant_pathway(k_ester, k_ox, k_nonspec),
        temperature_celsius=temperature_celsius,
        k_total_at_temp_per_h=k_total_at_temp,
        notes=notes,
    )


def compare_stability_conditions(
    compound_name: str,
    logP: float,
    ester_bond: bool,
    temperatures: list[float],
) -> list[PlasmaStabilityExtendedResult]:
    """Compare stability across multiple temperatures.

    Returns results sorted by t50_min descending (most stable first).
    """
    results = [
        predict_plasma_stability_extended(
            compound_name=compound_name,
            logP=logP,
            ester_bond=ester_bond,
            temperature_celsius=T,
        )
        for T in temperatures
    ]
    return sorted(results, key=lambda r: r.t50_min, reverse=True)
