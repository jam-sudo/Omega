"""Drug crystallization kinetics — Avrami nucleation-growth model (Phase 815).

Models crystallization from supersaturated solutions using the Avrami equation:
    fraction = 1 - exp(-(k * (t - t_ind))^n)  for t > t_ind
    fraction = 0                                 for t <= t_ind
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "CrystallizationKineticsResult",
    "simulate_crystallization",
    "screen_supersaturation",
]


@dataclass(frozen=True)
class CrystallizationKineticsResult:
    """Result of Avrami crystallization kinetics simulation."""

    compound_name: str
    avrami_n: float
    rate_constant_per_h: float
    induction_time_h: float
    t50_crystallization_h: float  # time to 50% crystallization; math.inf if not reached
    supersaturation_ratio: float
    crystallized_fraction_at_1h: float
    crystallized_fraction_at_6h: float
    crystallized_fraction_at_24h: float
    maximum_crystallization_rate_per_h: float
    nucleation_type: str  # "homogeneous" or "heterogeneous"
    notes: str


def _avrami_fraction(t: float, k: float, n: float, t_ind: float) -> float:
    """Compute Avrami crystallized fraction at time t."""
    if t <= t_ind:
        return 0.0
    dt = t - t_ind
    val = k * dt
    raw = 1.0 - math.exp(-(val**n))
    return max(0.0, min(1.0, raw))


def _max_rate(k: float, n: float, t_ind: float, t_end: float = 24.0, dt: float = 0.001) -> float:
    """Numerically find the maximum instantaneous crystallization rate (d(fraction)/dt)."""
    max_r = 0.0
    t = t_ind + dt
    prev_f = 0.0
    while t <= t_end:
        f = _avrami_fraction(t, k, n, t_ind)
        rate = (f - prev_f) / dt
        if rate > max_r:
            max_r = rate
        prev_f = f
        t += dt
    return max_r


def simulate_crystallization(
    compound_name: str,
    supersaturation_ratio: float,
    avrami_n: float,
    rate_constant_per_h: float,
    induction_time_h: float,
    t_end_h: float = 24.0,
    dt_h: float = 0.01,
) -> CrystallizationKineticsResult:
    """Simulate crystallization from supersaturated solution using the Avrami equation.

    Parameters
    ----------
    compound_name:
        Name of the compound.
    supersaturation_ratio:
        C / C_eq (>= 1.0 required, otherwise raises ValueError).
    avrami_n:
        Avrami exponent (must be in [1, 4]).
    rate_constant_per_h:
        Avrami rate constant k (h^-1).
    induction_time_h:
        Time before nucleation begins (h); fraction = 0 until this time.
    t_end_h:
        Simulation end time (h).
    dt_h:
        Time step (h).

    Returns
    -------
    CrystallizationKineticsResult
    """
    if supersaturation_ratio < 1.0:
        raise ValueError(f"supersaturation_ratio must be >= 1.0, got {supersaturation_ratio}")
    if not (1.0 <= avrami_n <= 4.0):
        raise ValueError(f"avrami_n must be in [1, 4], got {avrami_n}")
    if rate_constant_per_h < 0.0:
        raise ValueError(f"rate_constant_per_h must be >= 0, got {rate_constant_per_h}")
    if induction_time_h < 0.0:
        raise ValueError(f"induction_time_h must be >= 0, got {induction_time_h}")

    k = rate_constant_per_h
    n = avrami_n
    t_ind = induction_time_h

    # Evaluate fractions at reporting times
    f_1h = _avrami_fraction(1.0, k, n, t_ind)
    f_6h = _avrami_fraction(6.0, k, n, t_ind)
    f_24h = _avrami_fraction(24.0, k, n, t_ind)

    # Find t50 numerically (first time fraction >= 0.5)
    t50 = math.inf
    t = t_ind
    while t <= t_end_h:
        if _avrami_fraction(t, k, n, t_ind) >= 0.5:
            t50 = t
            break
        t += dt_h

    # Maximum crystallization rate
    max_rate = _max_rate(k, n, t_ind, t_end=max(t_end_h, 24.0), dt=dt_h)

    nucleation_type = "homogeneous" if n >= 3.0 else "heterogeneous"

    notes_parts = []
    if supersaturation_ratio >= 5.0:
        notes_parts.append("high supersaturation — rapid crystallization expected")
    if t50 == math.inf:
        notes_parts.append("50% crystallization not reached within simulation window")
    if nucleation_type == "homogeneous":
        notes_parts.append("3D nucleation-controlled growth (homogeneous)")
    else:
        notes_parts.append("surface/heterogeneous nucleation kinetics")
    notes = "; ".join(notes_parts) if notes_parts else "normal crystallization kinetics"

    return CrystallizationKineticsResult(
        compound_name=compound_name,
        avrami_n=n,
        rate_constant_per_h=k,
        induction_time_h=t_ind,
        t50_crystallization_h=t50,
        supersaturation_ratio=supersaturation_ratio,
        crystallized_fraction_at_1h=f_1h,
        crystallized_fraction_at_6h=f_6h,
        crystallized_fraction_at_24h=f_24h,
        maximum_crystallization_rate_per_h=max_rate,
        nucleation_type=nucleation_type,
        notes=notes,
    )


def screen_supersaturation(
    compound_name: str,
    supersaturation_ratios: list[float],
    avrami_n: float,
    rate_constant_per_h: float,
    induction_time_h: float,
) -> list[CrystallizationKineticsResult]:
    """Screen multiple supersaturation ratios and return sorted by t50 descending.

    Most stable (longest t50) first.
    """
    results = []
    for sr in supersaturation_ratios:
        result = simulate_crystallization(
            compound_name=compound_name,
            supersaturation_ratio=sr,
            avrami_n=avrami_n,
            rate_constant_per_h=rate_constant_per_h,
            induction_time_h=induction_time_h,
        )
        results.append(result)
    # Sort by t50 descending (most stable first); math.inf sorts last in ascending → first here
    results.sort(
        key=lambda r: (r.t50_crystallization_h == math.inf, r.t50_crystallization_h), reverse=True
    )
    return results
