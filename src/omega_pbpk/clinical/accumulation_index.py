"""Drug accumulation index calculator for multiple dosing regimens."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class AccumulationResult:
    """Result of multiple-dose accumulation simulation."""

    times_h: list
    c_plasma_mg_L: list
    dose_times_h: list
    cmax_per_dose: list
    ctrough_per_dose: list
    accumulation_ratio_observed: float
    t_half_h: float
    interval_h: float
    n_doses: int
    notes: str


def accumulation_index(t_half_h: float, interval_h: float) -> float:
    """Calculate drug accumulation ratio at steady state.

    Rac = 1 / (1 - exp(-ke * interval_h))
    where ke = ln(2) / t_half_h

    Returns:
        Accumulation ratio Cmax_ss / Cmax_dose1
    """
    if t_half_h <= 0:
        raise ValueError("t_half_h must be positive")
    if interval_h <= 0:
        raise ValueError("interval_h must be positive")

    ke = math.log(2) / t_half_h
    rac = 1.0 / (1.0 - math.exp(-ke * interval_h))
    return rac


def time_to_steady_state(t_half_h: float, fraction: float = 0.90) -> float:
    """Calculate time to reach a given fraction of steady state.

    t_ss = -ln(1 - fraction) / ke

    Returns:
        Time in hours to reach the specified fraction of steady state
    """
    if t_half_h <= 0:
        raise ValueError("t_half_h must be positive")
    if not (0 < fraction < 1):
        raise ValueError("fraction must be between 0 and 1 (exclusive)")

    ke = math.log(2) / t_half_h
    t_ss = -math.log(1.0 - fraction) / ke
    return t_ss


def simulate_multiple_dose_accumulation(
    t_half_h: float,
    dose_mg: float,
    vd_L: float,
    interval_h: float,
    n_doses: int = 10,
    dt_h: float = 0.1,
) -> AccumulationResult:
    """Simulate plasma concentration during multiple IV bolus dosing.

    Uses forward Euler integration. At each dose time, dose/Vd is added
    instantaneously to plasma concentration.

    Returns:
        AccumulationResult with concentration-time profile and statistics
    """
    if t_half_h <= 0:
        raise ValueError("t_half_h must be positive")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if vd_L <= 0:
        raise ValueError("vd_L must be positive")
    if interval_h <= 0:
        raise ValueError("interval_h must be positive")
    if n_doses < 1:
        raise ValueError("n_doses must be at least 1")

    ke = math.log(2) / t_half_h
    t_end = n_doses * interval_h
    n_steps = int(round(t_end / dt_h))

    times_h: list[float] = []
    c_plasma: list[float] = []
    dose_times_h: list[float] = [i * interval_h for i in range(n_doses)]

    # Track Cmax and Ctrough for each dosing interval
    cmax_per_dose: list[float] = []
    ctrough_per_dose: list[float] = []

    c = 0.0
    dose_index = 0

    for step in range(n_steps + 1):
        t = step * dt_h

        # Apply dose at dose times
        while dose_index < n_doses and abs(t - dose_index * interval_h) < dt_h * 0.5:
            c += dose_mg / vd_L
            dose_index += 1

        times_h.append(t)
        c_plasma.append(c)

        # Forward Euler decay
        if step < n_steps:
            c = c - ke * c * dt_h
            if c < 0:
                c = 0.0

    # Compute Cmax and Ctrough for each dosing interval
    # Interval i covers [dose_time_i, dose_time_{i+1}) — exclusive upper boundary
    for i in range(n_doses):
        t_start = i * interval_h
        # Last interval runs to end of simulation; others end just before next dose
        if i < n_doses - 1:
            t_end_interval = (i + 1) * interval_h - dt_h * 0.5
        else:
            t_end_interval = n_doses * interval_h

        interval_concs = [
            c_plasma[j]
            for j, t in enumerate(times_h)
            if t_start - dt_h * 0.1 <= t <= t_end_interval
        ]
        if interval_concs:
            cmax_per_dose.append(max(interval_concs))
            ctrough_per_dose.append(interval_concs[-1])
        else:
            cmax_per_dose.append(0.0)
            ctrough_per_dose.append(0.0)

    # Observed accumulation ratio: Cmax at last dose / Cmax at first dose
    if cmax_per_dose[0] > 0:
        acc_ratio_obs = cmax_per_dose[-1] / cmax_per_dose[0]
    else:
        acc_ratio_obs = 1.0

    theoretical_rac = accumulation_index(t_half_h, interval_h)
    notes = (
        f"Theoretical Rac={theoretical_rac:.2f}, "
        f"Observed Rac={acc_ratio_obs:.2f} after {n_doses} doses"
    )

    return AccumulationResult(
        times_h=times_h,
        c_plasma_mg_L=c_plasma,
        dose_times_h=dose_times_h,
        cmax_per_dose=cmax_per_dose,
        ctrough_per_dose=ctrough_per_dose,
        accumulation_ratio_observed=acc_ratio_obs,
        t_half_h=t_half_h,
        interval_h=interval_h,
        n_doses=n_doses,
        notes=notes,
    )


def peak_trough_ratio(t_half_h: float, interval_h: float) -> float:
    """Calculate peak-to-trough ratio at steady state.

    PTR = 1 / exp(-ke * interval_h) = exp(ke * interval_h)

    Returns:
        Ratio of Cmax to Ctrough at steady state
    """
    if t_half_h <= 0:
        raise ValueError("t_half_h must be positive")
    if interval_h <= 0:
        raise ValueError("interval_h must be positive")

    ke = math.log(2) / t_half_h
    ptr = math.exp(ke * interval_h)
    return ptr


def optimal_dosing_interval(
    t_half_h: float,
    target_accumulation_ratio: float = 2.0,
) -> float:
    """Find the dosing interval that achieves a target accumulation ratio.

    Solves analytically: tau = -ln(1 - 1/Rac) / ke

    Args:
        t_half_h: Elimination half-life in hours
        target_accumulation_ratio: Desired Rac (must be > 1)

    Returns:
        Optimal dosing interval in hours
    """
    if t_half_h <= 0:
        raise ValueError("t_half_h must be positive")
    if target_accumulation_ratio <= 1:
        raise ValueError("target_accumulation_ratio must be greater than 1")

    ke = math.log(2) / t_half_h
    tau = -math.log(1.0 - 1.0 / target_accumulation_ratio) / ke
    return tau
