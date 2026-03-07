"""
Phase 932 — PK model for hemodialysis patients.

2-compartment ODE with dialysis drug removal sessions, forward Euler integration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "DialysisPKResult",
    "simulate_dialysis_pk",
    "compare_dialysis_vs_normal",
]


@dataclass
class DialysisPKResult:
    """Result from dialysis PK simulation."""

    drug_name: str
    route: str
    dose_mg: float
    times_h: list
    c_central_mg_L: list
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    fraction_removed_by_dialysis: float
    t_half_interdialytic_h: float
    dialysis_sessions_simulated: int
    notes: str


def _is_dialysis_on(
    t: float,
    dialysis_start_h: float,
    dialysis_duration_h: float,
    dialysis_interval_h: float,
) -> bool:
    """Return True if dialysis is active at time t."""
    if t < dialysis_start_h:
        return False
    elapsed = t - dialysis_start_h
    phase = elapsed % dialysis_interval_h
    return phase < dialysis_duration_h


def _trapezoidal_auc(times: list, concentrations: list) -> float:
    """Compute AUC using trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (concentrations[i] + concentrations[i - 1]) * dt
    return auc


def simulate_dialysis_pk(
    drug_name: str,
    dose_mg: float,
    route: str = "iv",
    cl_hepatic_L_per_h: float = 5.0,
    cl_renal_residual_L_per_h: float = 0.0,
    vd_central_L: float = 20.0,
    vd_periph_L: float = 30.0,
    k12_per_h: float = 0.2,
    k21_per_h: float = 0.1,
    dialyzer_cl_L_per_h: float = 12.0,
    ka_per_h: float = 1.0,
    t_end_h: float = 168.0,
    dt_h: float = 0.1,
    dialysis_start_h: float = 0.0,
    dialysis_duration_h: float = 4.0,
    dialysis_interval_h: float = 48.0,
) -> DialysisPKResult:
    """Simulate drug PK in a hemodialysis patient.

    Parameters
    ----------
    drug_name : str
    dose_mg : float
        Dose in mg (must be > 0).
    route : str
        "iv" or "oral".
    cl_hepatic_L_per_h : float
        Hepatic clearance (L/h, >= 0).
    cl_renal_residual_L_per_h : float
        Residual renal clearance (L/h, default 0 for anuric).
    vd_central_L : float
        Central compartment volume (L, > 0).
    vd_periph_L : float
        Peripheral compartment volume (L, > 0).
    k12_per_h : float
        Distribution rate constant central→peripheral (1/h).
    k21_per_h : float
        Distribution rate constant peripheral→central (1/h).
    dialyzer_cl_L_per_h : float
        Dialyzer clearance (L/h, >= 0).
    ka_per_h : float
        Oral absorption rate constant (1/h).
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Integration step size (h).
    dialysis_start_h : float
        Start time of first dialysis session (h).
    dialysis_duration_h : float
        Duration of each dialysis session (h).
    dialysis_interval_h : float
        Interval between consecutive dialysis sessions (h).

    Returns
    -------
    DialysisPKResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_hepatic_L_per_h < 0:
        raise ValueError("cl_hepatic_L_per_h must be >= 0")
    if vd_central_L <= 0:
        raise ValueError("vd_central_L must be > 0")
    if vd_periph_L <= 0:
        raise ValueError("vd_periph_L must be > 0")
    if dialyzer_cl_L_per_h < 0:
        raise ValueError("dialyzer_cl_L_per_h must be >= 0")
    if route not in {"iv", "oral"}:
        raise ValueError(f"route must be 'iv' or 'oral', got '{route}'")

    # Elimination rate constant (hepatic + residual renal)
    cl_total_baseline = cl_hepatic_L_per_h + cl_renal_residual_L_per_h
    k_elim = cl_total_baseline / vd_central_L  # 1/h
    k_dial_rate = dialyzer_cl_L_per_h / vd_central_L  # 1/h when dialysis on

    # Initial conditions
    # State: [A_central (mg), A_periph (mg), A_gut (mg for oral)]
    if route == "iv":
        a_central = dose_mg
        a_periph = 0.0
        a_gut = 0.0
    else:  # oral
        a_central = 0.0
        a_periph = 0.0
        a_gut = dose_mg

    # Track dialysis removal
    total_removed_by_dialysis = 0.0  # mg

    # Track dialysis sessions
    session_count = 0
    in_session_prev = _is_dialysis_on(
        0.0, dialysis_start_h, dialysis_duration_h, dialysis_interval_h
    )
    if in_session_prev:
        session_count = 1

    # Results storage
    n_steps = int(round(t_end_h / dt_h)) + 1
    times = []
    c_central = []

    t = 0.0
    for step in range(n_steps):
        # Record current state
        times.append(t)
        c_central.append(a_central / vd_central_L)

        if step == n_steps - 1:
            break

        # Check dialysis status
        on = _is_dialysis_on(t, dialysis_start_h, dialysis_duration_h, dialysis_interval_h)

        # Detect new session start
        if on and not in_session_prev:
            session_count += 1
        in_session_prev = on

        dial_rate = k_dial_rate if on else 0.0

        # Compute derivatives
        # dA_central/dt = -(k_elim + dial_rate + k12) * A_central + k21 * A_periph + ka * A_gut
        # dA_periph/dt = k12 * A_central - k21 * A_periph
        # dA_gut/dt = -ka * A_gut  (oral only)

        da_central = (
            -(k_elim + dial_rate + k12_per_h) * a_central
            + k21_per_h * a_periph
            + (ka_per_h * a_gut if route == "oral" else 0.0)
        )
        da_periph = k12_per_h * a_central - k21_per_h * a_periph
        da_gut = -ka_per_h * a_gut if route == "oral" else 0.0

        # Amount removed by dialysis this step
        removed_step = dial_rate * a_central * dt_h
        total_removed_by_dialysis += removed_step

        # Forward Euler update
        a_central = max(0.0, a_central + da_central * dt_h)
        a_periph = max(0.0, a_periph + da_periph * dt_h)
        a_gut = max(0.0, a_gut + da_gut * dt_h)

        t = round(t + dt_h, 10)

    # PK metrics
    cmax = max(c_central)
    tmax_idx = c_central.index(cmax)
    tmax = times[tmax_idx]
    auc = _trapezoidal_auc(times, c_central)

    fraction_removed = min(1.0, total_removed_by_dialysis / dose_mg)

    # Interdialytic half-life: apparent half-life outside dialysis
    # Estimate from k_elim only (between sessions)
    k_app = k_elim + k12_per_h * k21_per_h / (k21_per_h + k_elim)  # effective elim
    if k_app > 0:
        t_half = math.log(2.0) / max(k_app, 1e-9)
    else:
        t_half = float("inf")

    notes = (
        f"Route={route}, dose={dose_mg}mg, "
        f"CL_hepatic={cl_hepatic_L_per_h:.1f} L/h, "
        f"CL_renal_residual={cl_renal_residual_L_per_h:.1f} L/h, "
        f"Dialyzer CL={dialyzer_cl_L_per_h:.1f} L/h; "
        f"Sessions simulated={session_count}; "
        f"Fraction removed by dialysis={fraction_removed:.3f}; "
        f"Interdialytic t½={t_half:.1f} h"
    )

    return DialysisPKResult(
        drug_name=drug_name,
        route=route,
        dose_mg=dose_mg,
        times_h=times,
        c_central_mg_L=c_central,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        fraction_removed_by_dialysis=fraction_removed,
        t_half_interdialytic_h=t_half,
        dialysis_sessions_simulated=session_count,
        notes=notes,
    )


def compare_dialysis_vs_normal(
    drug_name: str,
    dose_mg: float,
    route: str = "iv",
    cl_hepatic_L_per_h: float = 5.0,
    vd_central_L: float = 20.0,
    vd_periph_L: float = 30.0,
    k12_per_h: float = 0.2,
    k21_per_h: float = 0.1,
    dialyzer_cl_L_per_h: float = 12.0,
    ka_per_h: float = 1.0,
    t_end_h: float = 168.0,
    dt_h: float = 0.1,
    dialysis_start_h: float = 0.0,
    dialysis_duration_h: float = 4.0,
    dialysis_interval_h: float = 48.0,
    normal_renal_cl_L_per_h: float = 5.0,
) -> dict:
    """Compare PK in a dialysis patient vs. a patient with normal renal function.

    Parameters
    ----------
    normal_renal_cl_L_per_h : float
        Renal clearance for normal renal function (L/h).

    Returns
    -------
    dict with keys:
        "dialysis" : DialysisPKResult (anuric patient on HD)
        "normal_renal" : DialysisPKResult (normal renal, no dialysis)
        "auc_ratio" : float (auc_dialysis / auc_normal)
    """
    dialysis_result = simulate_dialysis_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        cl_hepatic_L_per_h=cl_hepatic_L_per_h,
        cl_renal_residual_L_per_h=0.0,
        vd_central_L=vd_central_L,
        vd_periph_L=vd_periph_L,
        k12_per_h=k12_per_h,
        k21_per_h=k21_per_h,
        dialyzer_cl_L_per_h=dialyzer_cl_L_per_h,
        ka_per_h=ka_per_h,
        t_end_h=t_end_h,
        dt_h=dt_h,
        dialysis_start_h=dialysis_start_h,
        dialysis_duration_h=dialysis_duration_h,
        dialysis_interval_h=dialysis_interval_h,
    )

    # Normal renal: no dialysis sessions (set dialyzer CL to 0), add renal CL
    normal_result = simulate_dialysis_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        cl_hepatic_L_per_h=cl_hepatic_L_per_h,
        cl_renal_residual_L_per_h=normal_renal_cl_L_per_h,
        vd_central_L=vd_central_L,
        vd_periph_L=vd_periph_L,
        k12_per_h=k12_per_h,
        k21_per_h=k21_per_h,
        dialyzer_cl_L_per_h=0.0,  # no dialyzer
        ka_per_h=ka_per_h,
        t_end_h=t_end_h,
        dt_h=dt_h,
        dialysis_start_h=dialysis_start_h,
        dialysis_duration_h=dialysis_duration_h,
        dialysis_interval_h=dialysis_interval_h,
    )

    auc_ratio = dialysis_result.auc_mg_h_per_L / max(normal_result.auc_mg_h_per_L, 1e-12)

    return {
        "dialysis": dialysis_result,
        "normal_renal": normal_result,
        "auc_ratio": auc_ratio,
    }
