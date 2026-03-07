"""Phase 217 — Tablet Coating PK

Models drug release kinetics from coated tablets:
- immediate release
- enteric coating
- sustained release
- film coating
"""

from __future__ import annotations

import math
from dataclasses import dataclass

VALID_COATINGS = {"immediate", "enteric", "sustained_release", "film"}


@dataclass
class CoatingReleaseResult:
    """Result of tablet coating release simulation."""

    drug_name: str
    dose_mg: float
    coating_type: str
    ph_dissolution: float
    times_h: list[float]
    pct_released: list[float]
    t50_h: float | None
    t80_h: float | None
    t_complete_h: float | None
    dissolution_efficiency: float
    coating_quality: str  # "fast" / "intermediate" / "slow"
    notes: str


def _trapz(xs: list[float], ys: list[float]) -> float:
    """Manual trapezoidal integration."""
    total = 0.0
    for i in range(1, len(xs)):
        total += 0.5 * (ys[i] + ys[i - 1]) * (xs[i] - xs[i - 1])
    return total


def _find_time_for_pct(
    times: list[float], pct_released: list[float], target: float
) -> float | None:
    """Linear interpolation to find time when pct_released crosses target."""
    for i in range(1, len(pct_released)):
        if pct_released[i - 1] <= target <= pct_released[i]:
            dt = times[i] - times[i - 1]
            dp = pct_released[i] - pct_released[i - 1]
            if dp == 0.0:
                return times[i - 1]
            return times[i - 1] + dt * (target - pct_released[i - 1]) / dp
        if pct_released[i - 1] == target:
            return times[i - 1]
    return None


def simulate_coating_release(
    drug_name: str,
    dose_mg: float,
    coating_type: str,
    coating_thickness_um: float = 100.0,
    pH_dissolution: float = 6.8,
) -> CoatingReleaseResult:
    """Simulate drug release from a coated tablet.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose in milligrams. Must be > 0.
    coating_type:
        One of "immediate", "enteric", "sustained_release", "film".
    coating_thickness_um:
        Coating thickness in micrometres (informational).
    pH_dissolution:
        pH of dissolution medium. Must be in [1, 9].

    Returns
    -------
    CoatingReleaseResult
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if coating_type not in VALID_COATINGS:
        raise ValueError(
            f"coating_type must be one of {sorted(VALID_COATINGS)}, got '{coating_type}'"
        )
    if not (1.0 <= pH_dissolution <= 9.0):
        raise ValueError(f"pH_dissolution must be in [1, 9], got {pH_dissolution}")

    # Determine release parameters
    t_lag = 0.0
    notes_parts = []

    if coating_type == "immediate":
        k_release = 0.5  # /h
        notes_parts.append("No lag; first-order release at k=0.5/h.")
    elif coating_type == "enteric":
        if pH_dissolution >= 5.5:
            t_lag = 0.0
            k_release = 0.3  # /h
            notes_parts.append(f"Enteric coating dissolved at pH {pH_dissolution:.1f}>=5.5.")
        else:
            t_lag = 1e6  # effectively no release in 24 h window
            k_release = 0.3
            notes_parts.append(
                f"Enteric coating intact at pH {pH_dissolution:.1f}<5.5; no release."
            )
    elif coating_type == "sustained_release":
        # Zero-order: 1/12 of dose per hour over 12 h
        # Model as fast first-order to approximate zero-order behaviour
        # but per spec: k_release = 1/12 /h of dose per hour
        k_release = None  # handled specially below
        notes_parts.append("Zero-order release over 12 h.")
    else:  # film
        k_release = 0.8  # /h
        t_lag = 0.0
        notes_parts.append("Thin film coating; fast first-order release at k=0.8/h.")

    # Simulate 0 to 24 h, step 0.5 h
    dt = 0.5
    times = [i * dt for i in range(int(24 / dt) + 1)]  # 0 .. 24 inclusive

    pct_released: list[float] = []

    for t in times:
        effective_t = max(0.0, t - t_lag)
        if coating_type == "sustained_release":
            # Zero-order: dose_mg/12 per hour, capped at dose_mg
            released_mg = min(dose_mg, (dose_mg / 12.0) * effective_t)
        else:
            released_mg = dose_mg * (1.0 - math.exp(-k_release * effective_t))
        pct = min(100.0, (released_mg / dose_mg) * 100.0)
        pct_released.append(pct)

    # Compute t50, t80, t_complete (>95%)
    t50 = _find_time_for_pct(times, pct_released, 50.0)
    t80 = _find_time_for_pct(times, pct_released, 80.0)
    t_complete = _find_time_for_pct(times, pct_released, 95.0)

    # Dissolution efficiency: AUC of release profile / theoretical max (100% * 24 h)
    auc_release = _trapz(times, pct_released)
    theoretical_max = 100.0 * 24.0
    dissolution_efficiency = auc_release / theoretical_max

    # Coating quality classification based on t50
    if t50 is None:
        coating_quality = "slow"
    elif t50 < 2.0:
        coating_quality = "fast"
    elif t50 < 8.0:
        coating_quality = "intermediate"
    else:
        coating_quality = "slow"

    notes = " ".join(notes_parts)

    return CoatingReleaseResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        coating_type=coating_type,
        ph_dissolution=pH_dissolution,
        times_h=times,
        pct_released=pct_released,
        t50_h=t50,
        t80_h=t80,
        t_complete_h=t_complete,
        dissolution_efficiency=dissolution_efficiency,
        coating_quality=coating_quality,
        notes=notes,
    )


def compare_coatings(
    drug_name: str,
    dose_mg: float,
    pH_dissolution: float = 6.8,
) -> list[CoatingReleaseResult]:
    """Simulate all 4 coating types and return sorted by t50 ascending (fastest first).

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose in milligrams.
    pH_dissolution:
        pH of dissolution medium.

    Returns
    -------
    List of 4 CoatingReleaseResult sorted by t50_h ascending (None treated as infinity).
    """
    results = [
        simulate_coating_release(drug_name, dose_mg, ct, pH_dissolution=pH_dissolution)
        for ct in sorted(VALID_COATINGS)
    ]

    def _sort_key(r: CoatingReleaseResult) -> float:
        return r.t50_h if r.t50_h is not None else float("inf")

    results.sort(key=_sort_key)
    return results
