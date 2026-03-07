"""Phase 1124 — Drug holiday simulation using superposition.

Simulates three phases of plasma concentration:
  1. Pre-holiday: multiple oral doses accumulate to steady state.
  2. Holiday: mono-exponential decay from last pre-holiday concentration.
  3. Restart: multi-dose accumulation back to steady state.

Single-compartment, first-order absorption model:
    C_dose(t) = (F * D * ka) / (Vd * (ka - ke)) * (exp(-ke*t) - exp(-ka*t))

Superposition is used to sum contributions of each dose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class DrugHolidayResult:
    """Result of a Phase-1124 drug holiday simulation.

    Attributes
    ----------
    drug_name : Drug identifier.
    dose_mg : Dose per administration (mg).
    dosing_interval_h : Dosing interval (h).
    holiday_duration_h : Duration of treatment interruption (h).
    times_h : Time points (h).
    c_plasma_mg_L : Plasma concentration at each time point (mg/L).
    css_avg_mg_L : Pre-holiday steady-state average concentration (mg/L).
    c_at_holiday_end_mg_L : Concentration at end of holiday (mg/L).
    c_percent_remaining_pct : c_at_holiday_end / css_avg * 100 (%).
    doses_to_resteady_state : Doses after restart until within 90% of css_avg.
    t_half_h : Elimination half-life (h).
    notes : Summary note string.
    """

    drug_name: str
    dose_mg: float
    dosing_interval_h: float
    holiday_duration_h: float
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    css_avg_mg_L: float = 0.0
    c_at_holiday_end_mg_L: float = 0.0
    c_percent_remaining_pct: float = 0.0
    doses_to_resteady_state: int = 0
    t_half_h: float = 0.0
    notes: str = ""


def _single_dose_conc(
    t: float,
    dose_mg: float,
    ka: float,
    ke: float,
    vd_L: float,
    f: float,
) -> float:
    """Concentration (mg/L) from a single oral dose at time t after administration.

    Uses the standard 1-cpt oral formula:
        C(t) = (F * D * ka) / (Vd * (ka - ke)) * (exp(-ke*t) - exp(-ka*t))

    When ka ≈ ke, uses L'Hopital limit → C(t) = (F * D * ka / Vd) * t * exp(-ke*t).
    """
    if t < 0:
        return 0.0
    if abs(ka - ke) < 1e-9:
        # L'Hopital's rule limit
        return (f * dose_mg * ka / vd_L) * t * math.exp(-ke * t)
    return (f * dose_mg * ka) / (vd_L * (ka - ke)) * (math.exp(-ke * t) - math.exp(-ka * t))


def simulate_drug_holiday(
    drug_name: str,
    dose_mg: float,
    dosing_interval_h: float,
    cl_L_per_h: float,
    vd_L: float,
    holiday_duration_h: float,
    n_pre_holiday_doses: int = 10,
    n_post_holiday_doses: int = 10,
    ka_per_h: float = 1.0,
    f_bioavail: float = 1.0,
) -> DrugHolidayResult:
    """Simulate drug plasma concentration during a drug holiday.

    Parameters
    ----------
    drug_name : Drug identifier.
    dose_mg : Dose per administration (mg). Must be > 0.
    dosing_interval_h : Dosing interval (h). Must be > 0.
    cl_L_per_h : Systemic clearance (L/h). Must be > 0.
    vd_L : Volume of distribution (L). Must be > 0.
    holiday_duration_h : Duration of holiday (h). Must be > 0.
    n_pre_holiday_doses : Number of doses before holiday to establish SS.
    n_post_holiday_doses : Number of doses after restart.
    ka_per_h : Absorption rate constant (1/h).
    f_bioavail : Oral bioavailability fraction (0, 1].

    Returns
    -------
    DrugHolidayResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if holiday_duration_h <= 0:
        raise ValueError("holiday_duration_h must be > 0")
    if dosing_interval_h <= 0:
        raise ValueError("dosing_interval_h must be > 0")
    if not (0 < f_bioavail <= 1):
        raise ValueError("f_bioavail must be in (0, 1]")

    ke = cl_L_per_h / vd_L
    t_half_h = math.log(2) / ke

    # Time resolution: 0.1 h steps
    dt = 0.1

    # Phase boundaries
    holiday_start = n_pre_holiday_doses * dosing_interval_h
    holiday_end = holiday_start + holiday_duration_h
    sim_end = holiday_end + n_post_holiday_doses * dosing_interval_h

    n_steps = int(round(sim_end / dt)) + 1
    times_h: list[float] = [i * dt for i in range(n_steps)]

    # All dose times: pre-holiday doses + post-holiday doses
    pre_dose_times = [i * dosing_interval_h for i in range(n_pre_holiday_doses)]
    post_dose_times = [holiday_end + i * dosing_interval_h for i in range(n_post_holiday_doses)]
    all_dose_times = pre_dose_times + post_dose_times

    # Superposition: sum contributions from all doses
    c_plasma: list[float] = [0.0] * n_steps
    for dose_t in all_dose_times:
        for i, t in enumerate(times_h):
            tau = t - dose_t
            if tau < 0:
                continue
            c_plasma[i] += _single_dose_conc(tau, dose_mg, ka_per_h, ke, vd_L, f_bioavail)

    # Ensure no negative concentrations
    c_plasma = [max(c, 0.0) for c in c_plasma]

    # css_avg: average concentration over last dosing interval before holiday
    css_window_start = holiday_start - dosing_interval_h
    css_window_end = holiday_start
    css_indices = [i for i, t in enumerate(times_h) if css_window_start <= t <= css_window_end]
    if css_indices:
        css_avg = sum(c_plasma[i] for i in css_indices) / len(css_indices)
    else:
        css_avg = c_plasma[0] if c_plasma else 0.0

    # c_at_holiday_end: concentration at the end of the holiday
    holiday_end_idx = min(range(n_steps), key=lambda i: abs(times_h[i] - holiday_end))
    c_at_holiday_end = c_plasma[holiday_end_idx]

    # c_percent_remaining
    c_percent_remaining = (c_at_holiday_end / css_avg * 100.0) if css_avg > 0 else 0.0

    # doses_to_resteady_state: count post-holiday doses until within 90% Css
    target_90 = 0.9 * css_avg
    doses_to_resteady = n_post_holiday_doses  # pessimistic default
    for d_idx in range(n_post_holiday_doses):
        # check concentration at end of this dose interval
        interval_end = holiday_end + (d_idx + 1) * dosing_interval_h
        idx = min(range(n_steps), key=lambda i: abs(times_h[i] - interval_end))
        if c_plasma[idx] >= target_90:
            doses_to_resteady = d_idx + 1
            break

    notes = (
        f"ke={ke:.4f}/h, t_half={t_half_h:.2f}h; "
        f"holiday {holiday_start:.1f}-{holiday_end:.1f}h; "
        f"css_avg={css_avg:.4f}mg/L; "
        f"c_at_end={c_at_holiday_end:.4f}mg/L ({c_percent_remaining:.1f}% remaining)"
    )

    return DrugHolidayResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        dosing_interval_h=dosing_interval_h,
        holiday_duration_h=holiday_duration_h,
        times_h=times_h,
        c_plasma_mg_L=c_plasma,
        css_avg_mg_L=css_avg,
        c_at_holiday_end_mg_L=c_at_holiday_end,
        c_percent_remaining_pct=c_percent_remaining,
        doses_to_resteady_state=doses_to_resteady,
        t_half_h=t_half_h,
        notes=notes,
    )


__all__ = [
    "DrugHolidayResult",
    "simulate_drug_holiday",
]
