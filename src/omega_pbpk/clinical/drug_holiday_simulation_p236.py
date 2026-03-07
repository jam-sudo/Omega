"""Phase 236 — Drug holiday simulation (treatment interruption PK).

Models drug concentration decay during a holiday (treatment interruption)
and provides tools to optimize holiday duration using a 1-compartment oral
PK model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _trapz(times: list, values: list) -> float:
    """Trapezoidal integration."""
    return sum(
        0.5 * (values[i] + values[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DrugHolidayResult:
    """Result of a drug holiday simulation.

    Attributes
    ----------
    drug_name : Drug identifier string.
    dose_mg : Dose per administration (mg).
    dosing_interval_h : Dosing interval (h).
    holiday_duration_h : Duration of treatment interruption (h).
    css_before_holiday_mg_L : Approximate steady-state concentration before holiday (mg/L).
    c_at_restart_mg_L : Plasma concentration at the time of treatment restart (mg/L).
    washout_fraction : Fraction of Css remaining at restart (c_at_restart / css_before).
    rebound_auc_mg_h_per_L : AUC during the holiday period (mg*h/L).
    time_below_mic_h : Time during holiday when concentration is below mic_threshold (h).
    restart_recommended : True when washout_fraction < 0.2 (full washout achieved).
    notes : Summary notes.
    """

    drug_name: str
    dose_mg: float
    dosing_interval_h: float
    holiday_duration_h: float
    css_before_holiday_mg_L: float
    c_at_restart_mg_L: float
    washout_fraction: float
    rebound_auc_mg_h_per_L: float
    time_below_mic_h: float
    restart_recommended: bool
    notes: str


# ---------------------------------------------------------------------------
# Core simulation function
# ---------------------------------------------------------------------------


def simulate_drug_holiday(
    drug_name: str,
    dose_mg: float,
    dosing_interval_h: float,
    holiday_duration_h: float,
    cl_L_per_h: float,
    vd_L: float,
    f_oral: float = 0.8,
    mic_threshold_mg_L: float = 0.5,
) -> DrugHolidayResult:
    """Simulate drug concentration during a treatment holiday.

    Uses a simple 1-compartment oral PK model:
    - Css_avg = F * dose / (CL * tau)
    - At holiday start, C0 = Css_avg (approximation)
    - During holiday: C(t) = C0 * exp(-ke * t)

    Parameters
    ----------
    drug_name : Drug identifier string.
    dose_mg : Dose per administration (mg). Must be > 0.
    dosing_interval_h : Dosing interval (h). Must be > 0.
    holiday_duration_h : Duration of treatment interruption (h). Must be > 0.
    cl_L_per_h : Systemic clearance (L/h). Must be > 0.
    vd_L : Volume of distribution (L). Must be > 0.
    f_oral : Oral bioavailability fraction (0, 1]. Default 0.8.
    mic_threshold_mg_L : Minimum effective concentration threshold (mg/L).

    Returns
    -------
    DrugHolidayResult

    Raises
    ------
    ValueError
        If any required parameter is out of valid range.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if dosing_interval_h <= 0:
        raise ValueError("dosing_interval_h must be > 0")
    if holiday_duration_h <= 0:
        raise ValueError("holiday_duration_h must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if not (0 < f_oral <= 1):
        raise ValueError("f_oral must be in (0, 1]")

    ke = cl_L_per_h / vd_L  # elimination rate constant (1/h)

    # Steady-state average concentration at holiday start
    css_avg = f_oral * dose_mg / (cl_L_per_h * dosing_interval_h)
    c0 = css_avg  # C at holiday start = Css_avg

    # Concentration at end of holiday
    c_at_restart = c0 * math.exp(-ke * holiday_duration_h)

    # Washout fraction
    washout_fraction = c_at_restart / c0 if c0 > 0 else 0.0

    # Numerical AUC during holiday (dt = 0.1 h)
    dt = 0.1
    n_steps = max(int(round(holiday_duration_h / dt)), 1)
    times: list = []
    concs: list = []
    for step in range(n_steps + 1):
        t = step * dt
        if t > holiday_duration_h:
            t = holiday_duration_h
        c = c0 * math.exp(-ke * t)
        times.append(t)
        concs.append(c)

    rebound_auc = _trapz(times, concs)

    # Time below MIC during holiday
    time_below_mic = 0.0
    for i in range(1, len(times)):
        c_prev = concs[i - 1]
        c_curr = concs[i]
        dt_step = times[i] - times[i - 1]
        below_prev = 1.0 if c_prev < mic_threshold_mg_L else 0.0
        below_curr = 1.0 if c_curr < mic_threshold_mg_L else 0.0
        time_below_mic += (below_prev + below_curr) / 2.0 * dt_step

    # Restart recommendation: full washout if < 20% of Css remains
    restart_recommended = washout_fraction < 0.2

    t_half = math.log(2.0) / ke if ke > 0 else float("inf")
    notes = (
        f"ke={ke:.4f}/h, t_half={t_half:.2f}h; "
        f"Css_avg={css_avg:.4f}mg/L; "
        f"washout={washout_fraction:.3f} after {holiday_duration_h:.1f}h"
    )

    return DrugHolidayResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        dosing_interval_h=dosing_interval_h,
        holiday_duration_h=holiday_duration_h,
        css_before_holiday_mg_L=css_avg,
        c_at_restart_mg_L=c_at_restart,
        washout_fraction=washout_fraction,
        rebound_auc_mg_h_per_L=rebound_auc,
        time_below_mic_h=time_below_mic,
        restart_recommended=restart_recommended,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Holiday duration optimizer
# ---------------------------------------------------------------------------


def optimize_holiday_duration(
    drug_name: str,
    dose_mg: float,
    dosing_interval_h: float,
    cl_L_per_h: float,
    vd_L: float,
    f_oral: float = 0.8,
    target_washout_fraction: float = 0.1,
    mic_threshold_mg_L: float = 0.5,
) -> dict:
    """Find the holiday duration to achieve a target washout fraction.

    Uses the analytical formula:
        holiday_h = -ln(target_washout_fraction) / ke

    Parameters
    ----------
    drug_name : Drug identifier string.
    dose_mg : Dose per administration (mg).
    dosing_interval_h : Dosing interval (h).
    cl_L_per_h : Systemic clearance (L/h).
    vd_L : Volume of distribution (L).
    f_oral : Oral bioavailability fraction (0, 1].
    target_washout_fraction : Desired fraction of Css remaining at restart (0, 1).
    mic_threshold_mg_L : MIC threshold (mg/L).

    Returns
    -------
    dict with keys:
        optimal_holiday_h : float
        expected_washout_fraction : float
        time_below_mic_h : float
        achievable : bool
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if dosing_interval_h <= 0:
        raise ValueError("dosing_interval_h must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if not (0 < f_oral <= 1):
        raise ValueError("f_oral must be in (0, 1]")
    if not (0 < target_washout_fraction < 1):
        raise ValueError("target_washout_fraction must be in (0, 1)")

    ke = cl_L_per_h / vd_L

    # Analytical solution
    if ke > 0 and target_washout_fraction > 0:
        optimal_holiday_h = -math.log(target_washout_fraction) / ke
    else:
        optimal_holiday_h = float("inf")

    achievable = math.isfinite(optimal_holiday_h) and optimal_holiday_h > 0

    if achievable:
        result = simulate_drug_holiday(
            drug_name=drug_name,
            dose_mg=dose_mg,
            dosing_interval_h=dosing_interval_h,
            holiday_duration_h=optimal_holiday_h,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            f_oral=f_oral,
            mic_threshold_mg_L=mic_threshold_mg_L,
        )
        expected_washout_fraction = result.washout_fraction
        time_below_mic_h = result.time_below_mic_h
    else:
        expected_washout_fraction = target_washout_fraction
        time_below_mic_h = 0.0

    return {
        "optimal_holiday_h": optimal_holiday_h,
        "expected_washout_fraction": expected_washout_fraction,
        "time_below_mic_h": time_below_mic_h,
        "achievable": achievable,
    }


__all__ = [
    "DrugHolidayResult",
    "simulate_drug_holiday",
    "optimize_holiday_duration",
]
