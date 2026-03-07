"""
Phase 933 — Burn Patient PK Model

PK model for burn injury patients with time-varying parameters due to
burn pathophysiology (resuscitation, hypermetabolic, and recovery phases).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["BurnPatientPKResult", "simulate_burn_pk", "compare_burn_severity"]


@dataclass
class BurnPatientPKResult:
    """Result of a burn patient PK simulation."""

    drug_name: str
    tbsa_pct: float
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    vd_at_24h_L: float
    cl_at_96h_L_per_h: float
    vd_fold_change: float
    cl_fold_change: float
    notes: str


def _get_phase_multipliers(t_h: float, tbsa_pct: float) -> tuple:
    """
    Return (vd_mult, cl_mult) for the given time point and TBSA.

    Burn phases:
      0-48h   (resuscitation):  Vd * 2.5, CL * 0.7
      48-168h (hypermetabolic): Vd * 1.8, CL * 1.5
      >168h   (recovery):       Vd * 1.3, CL * 1.2

    Severe burn (>40% TBSA) multiplies the *alteration* (delta from 1.0) by 1.5.
    """
    # Base phase multipliers
    if t_h < 48.0:
        vd_base = 2.5
        cl_base = 0.7
    elif t_h < 168.0:
        vd_base = 1.8
        cl_base = 1.5
    else:
        vd_base = 1.3
        cl_base = 1.2

    if tbsa_pct == 0.0:
        return 1.0, 1.0

    # Scale alteration (delta from 1.0) by 1.5 for severe burns
    if tbsa_pct > 40.0:
        severity = 1.5
    else:
        severity = 1.0

    # tbsa fraction (0-1) to scale alteration proportionally
    tbsa_fraction = min(tbsa_pct / 100.0, 1.0)

    # Alteration from 1.0, scaled by tbsa fraction and severity
    vd_delta = (vd_base - 1.0) * tbsa_fraction * severity
    cl_delta = (cl_base - 1.0) * tbsa_fraction * severity

    vd_mult = 1.0 + vd_delta
    cl_mult = 1.0 + cl_delta

    return vd_mult, cl_mult


def simulate_burn_pk(
    drug_name: str,
    dose_mg: float,
    tbsa_pct: float,
    cl_normal_L_per_h: float = 10.0,
    vd_normal_L: float = 50.0,
    route: str = "iv",
    dosing_interval_h: float = 24.0,
    n_doses: int = 1,
    t_end_h: float = 168.0,
    dt_h: float = 0.5,
) -> BurnPatientPKResult:
    """
    Simulate PK in a burn patient using Forward Euler with time-varying parameters.

    Parameters
    ----------
    drug_name : str
    dose_mg : float
    tbsa_pct : float   (0-100)
    cl_normal_L_per_h : float
    vd_normal_L : float
    route : str   ("iv" only for now)
    dosing_interval_h : float
    n_doses : int
    t_end_h : float
    dt_h : float

    Returns
    -------
    BurnPatientPKResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if not (0 <= tbsa_pct <= 100):
        raise ValueError("tbsa_pct must be in [0, 100]")
    if cl_normal_L_per_h <= 0:
        raise ValueError("cl_normal_L_per_h must be > 0")
    if vd_normal_L <= 0:
        raise ValueError("vd_normal_L must be > 0")
    if n_doses < 1:
        raise ValueError("n_doses must be >= 1")

    # Build dose times
    dose_times = [i * dosing_interval_h for i in range(n_doses)]

    # Forward Euler simulation
    n_steps = int(math.ceil(t_end_h / dt_h)) + 1
    times_h = [i * dt_h for i in range(n_steps)]
    # Clamp last time to t_end_h
    if times_h and times_h[-1] > t_end_h:
        times_h[-1] = t_end_h

    c_plasma = [0.0] * len(times_h)

    # Amount in central compartment (mg)
    amount = 0.0

    # Apply initial IV bolus doses that land at t=0
    for dt in dose_times:
        if dt == 0.0:
            amount += dose_mg

    for i in range(len(times_h) - 1):
        t = times_h[i]
        t_next = times_h[i + 1]
        step = t_next - t

        vd_mult, cl_mult = _get_phase_multipliers(t, tbsa_pct)
        vd_eff = vd_normal_L * vd_mult
        cl_eff = cl_normal_L_per_h * cl_mult
        ke = cl_eff / vd_eff

        # Current concentration
        c_plasma[i] = amount / vd_eff if vd_eff > 0 else 0.0

        # dA/dt = -ke * A   (1-cpt elimination)
        dA = -ke * amount * step
        amount = max(0.0, amount + dA)

        # Apply any dose at t_next
        for dt in dose_times:
            if abs(t_next - dt) < step * 0.5 and dt > 0.0:
                amount += dose_mg

    # Last concentration
    if times_h:
        t_last = times_h[-1]
        vd_mult_last, _ = _get_phase_multipliers(t_last, tbsa_pct)
        vd_eff_last = vd_normal_L * vd_mult_last
        c_plasma[-1] = amount / vd_eff_last if vd_eff_last > 0 else 0.0

    # Cmax and Tmax
    cmax = max(c_plasma)
    tmax = times_h[c_plasma.index(cmax)] if cmax > 0 else 0.0

    # AUC via trapezoidal rule
    auc = 0.0
    for i in range(len(times_h) - 1):
        dt_step = times_h[i + 1] - times_h[i]
        auc += 0.5 * (c_plasma[i] + c_plasma[i + 1]) * dt_step

    # Vd at 24h and CL at 96h
    vd_mult_24, _ = _get_phase_multipliers(24.0, tbsa_pct)
    _, cl_mult_96 = _get_phase_multipliers(96.0, tbsa_pct)
    vd_at_24h = vd_normal_L * vd_mult_24
    cl_at_96h = cl_normal_L_per_h * cl_mult_96

    vd_fold_change = vd_mult_24
    cl_fold_change = cl_mult_96

    # Notes
    if tbsa_pct == 0.0:
        severity_desc = "no burn"
    elif tbsa_pct <= 20.0:
        severity_desc = "mild burn"
    elif tbsa_pct <= 40.0:
        severity_desc = "moderate burn"
    else:
        severity_desc = "severe burn"

    notes = (
        f"Burn patient PK for {drug_name}: {severity_desc} ({tbsa_pct:.0f}% TBSA). "
        f"Vd fold-change at 24h: {vd_fold_change:.2f}x, "
        f"CL fold-change at 96h: {cl_fold_change:.2f}x. "
        f"Phases: resuscitation (0-48h), hypermetabolic (48-168h), recovery (>168h)."
    )

    return BurnPatientPKResult(
        drug_name=drug_name,
        tbsa_pct=tbsa_pct,
        dose_mg=dose_mg,
        times_h=times_h,
        c_plasma_mg_L=c_plasma,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        vd_at_24h_L=vd_at_24h,
        cl_at_96h_L_per_h=cl_at_96h,
        vd_fold_change=vd_fold_change,
        cl_fold_change=cl_fold_change,
        notes=notes,
    )


def compare_burn_severity(
    drug_name: str,
    dose_mg: float,
    tbsa_levels: list[float] | None = None,
    cl_normal_L_per_h: float = 10.0,
    vd_normal_L: float = 50.0,
    route: str = "iv",
    dosing_interval_h: float = 24.0,
    n_doses: int = 1,
    t_end_h: float = 168.0,
    dt_h: float = 0.5,
) -> list[BurnPatientPKResult]:
    """
    Run simulate_burn_pk for multiple TBSA levels and return results
    sorted by AUC descending.

    Parameters
    ----------
    tbsa_levels : list of floats, default [10.0, 20.0, 40.0, 60.0]

    Returns
    -------
    List[BurnPatientPKResult] sorted by auc_mg_h_per_L descending
    """
    if tbsa_levels is None:
        tbsa_levels = [10.0, 20.0, 40.0, 60.0]

    results = []
    for tbsa in tbsa_levels:
        result = simulate_burn_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            tbsa_pct=tbsa,
            cl_normal_L_per_h=cl_normal_L_per_h,
            vd_normal_L=vd_normal_L,
            route=route,
            dosing_interval_h=dosing_interval_h,
            n_doses=n_doses,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(result)

    results.sort(key=lambda r: r.auc_mg_h_per_L, reverse=True)
    return results
