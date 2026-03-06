"""Gastric emptying and GI transit simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class GastricEmptyingResult:
    state: str                    # fasted / fed
    t_lag_h: float                # gastric lag time
    t_emptying_h: float           # time for 50% gastric emptying
    ka_effective_per_h: float     # effective absorption rate (after transit)
    fraction_absorbed_1h: float
    fraction_absorbed_4h: float
    times_h: list[float]
    fraction_remaining_gut: list[float]
    fraction_absorbed: list[float]
    transit_time_h: float         # total small intestine transit time
    note: str


def _gastric_emptying_rate(state: str, dose_mg: float = 100.0) -> tuple[float, float]:
    """Return (ke_gastric_per_h, t_lag_h) for fasted/fed state.

    Fasted: rapid emptying (ke~2/h, lag~0.1h)
    Fed: slower emptying (ke~0.5/h, lag~0.5h) depending on fat content
    """
    if state == "fed":
        ke = 0.5
        t_lag = 0.5
    else:  # fasted
        ke = 2.0
        t_lag = 0.1
    return ke, t_lag


def simulate_gastric_emptying(
    state: str = "fasted",
    dose_mg: float = 100.0,
    ka_absorption_per_h: float = 1.0,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
    transit_time_h: float | None = None,
) -> GastricEmptyingResult:
    """Simulate gastric emptying and intestinal absorption.

    Two-phase model:
    1. Gastric phase: first-order emptying with lag
    2. Intestinal phase: first-order absorption of delivered drug

    Parameters
    ----------
    state : str
        'fasted' or 'fed'.
    ka_absorption_per_h : float
        Intestinal absorption rate constant (1/h).
    transit_time_h : float or None
        Small intestine transit time; defaults to 3h (fasted) or 5h (fed).
    """
    if state not in ("fasted", "fed"):
        raise ValueError("state must be 'fasted' or 'fed'")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if ka_absorption_per_h <= 0:
        raise ValueError("ka_absorption_per_h must be > 0")

    ke_gastric, t_lag = _gastric_emptying_rate(state, dose_mg)

    if transit_time_h is None:
        transit_time_h = 3.0 if state == "fasted" else 5.0

    n = max(int(t_end_h / dt_h), 1)
    t = np.linspace(0.0, t_end_h, n + 1)

    m_stomach = dose_mg  # drug in stomach
    m_intestine = 0.0    # drug in small intestine
    m_absorbed = 0.0     # drug absorbed

    frac_remaining = np.zeros(n + 1)
    frac_absorbed = np.zeros(n + 1)
    frac_remaining[0] = 1.0

    # Effective ka accounting for transit: if drug passes ileocecal valve, not absorbed
    # Simplified: ka_eff = ka_absorption if transit_time >> 1/ka else ka_eff = min(ka, 1/transit_time)
    ka_eff = min(ka_absorption_per_h, 1.0 / max(transit_time_h, 0.1) * 3.0)

    for i in range(n):
        current_t = t[i]
        if current_t < t_lag:
            # Lag phase — no emptying
            pass
        else:
            # Gastric emptying
            emptied = ke_gastric * m_stomach * dt_h
            emptied = min(emptied, m_stomach)
            m_stomach -= emptied
            m_stomach = max(m_stomach, 0.0)
            m_intestine += emptied

        # Intestinal absorption
        absorbed = ka_eff * m_intestine * dt_h
        absorbed = min(absorbed, m_intestine)
        m_intestine -= absorbed
        m_intestine = max(m_intestine, 0.0)
        m_absorbed += absorbed

        total_remaining = m_stomach + m_intestine
        frac_remaining[i + 1] = total_remaining / dose_mg
        frac_absorbed[i + 1] = m_absorbed / dose_mg

    # 50% gastric emptying time (from stomach only)
    t_empty_50 = math.log(2.0) / ke_gastric + t_lag

    f1h_idx = min(int(1.0 / dt_h), n)
    f4h_idx = min(int(4.0 / dt_h), n)

    return GastricEmptyingResult(
        state=state,
        t_lag_h=t_lag,
        t_emptying_h=t_empty_50,
        ka_effective_per_h=ka_eff,
        fraction_absorbed_1h=float(frac_absorbed[f1h_idx]),
        fraction_absorbed_4h=float(frac_absorbed[f4h_idx]),
        times_h=t.tolist(),
        fraction_remaining_gut=frac_remaining.tolist(),
        fraction_absorbed=frac_absorbed.tolist(),
        transit_time_h=transit_time_h,
        note=f"State: {state}; ke_gastric={ke_gastric}/h; lag={t_lag}h",
    )


def compare_food_states(
    dose_mg: float = 100.0,
    ka_absorption_per_h: float = 1.0,
    t_end_h: float = 12.0,
) -> dict[str, GastricEmptyingResult]:
    """Compare fasted vs fed gastric emptying."""
    return {
        "fasted": simulate_gastric_emptying("fasted", dose_mg, ka_absorption_per_h, t_end_h),
        "fed": simulate_gastric_emptying("fed", dose_mg, ka_absorption_per_h, t_end_h),
    }


import math  # noqa: E402 (used in function body)

__all__ = [
    "GastricEmptyingResult",
    "simulate_gastric_emptying",
    "compare_food_states",
]
