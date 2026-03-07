"""Phase 635 — Drug holiday simulation: PK during treatment interruptions and rebound.

Models drug concentration decay during a holiday (treatment interruption) and
re-accumulation after restart using a 1-compartment oral PK model (Forward Euler).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class DrugHolidayResult:
    """Result of a drug holiday simulation.

    Attributes
    ----------
    drug_name : Name of the drug.
    dose_mg : Dose per administration (mg).
    dosing_interval_h : Dosing interval (h).
    holiday_start_h : Time at which holiday begins (h).
    holiday_duration_h : Duration of treatment interruption (h).
    times_h : Simulation time points (h).
    c_plasma_mg_L : Plasma concentration at each time point (mg/L).
    css_before_mg_L : Steady-state concentration (average over last 3 dose intervals) before holiday.
    c_nadir_mg_L : Minimum concentration during the holiday.
    time_to_nadir_h : Time (from holiday start) at which nadir occurs.
    c_at_holiday_end_mg_L : Concentration at end of holiday (before first restart dose).
    rebound_cmax_mg_L : Peak concentration after treatment restart.
    time_in_subtherapeutic_h : Total time concentration is below MEC.
    therapeutic_window_restored_h : Time after restart to reach MEC again (0 if already above).
    notes : Summary notes.
    """

    drug_name: str
    dose_mg: float
    dosing_interval_h: float
    holiday_start_h: float
    holiday_duration_h: float
    times_h: list
    c_plasma_mg_L: list
    css_before_mg_L: float
    c_nadir_mg_L: float
    time_to_nadir_h: float
    c_at_holiday_end_mg_L: float
    rebound_cmax_mg_L: float
    time_in_subtherapeutic_h: float
    therapeutic_window_restored_h: float
    notes: str


def simulate_drug_holiday(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    ka_per_h: float = 1.0,
    f_oral: float = 0.8,
    dosing_interval_h: float = 12.0,
    n_doses_before: int = 10,
    holiday_duration_h: float = 48.0,
    n_doses_after: int = 5,
    mec_mg_L: float = 0.5,
    dt_h: float = 0.1,
) -> DrugHolidayResult:
    """Simulate PK during a drug holiday (treatment interruption) and restart.

    Parameters
    ----------
    drug_name : Drug identifier string.
    dose_mg : Oral dose per administration (mg).
    cl_L_per_h : Systemic clearance (L/h).
    vd_L : Volume of distribution (L).
    ka_per_h : Oral absorption rate constant (1/h).
    f_oral : Oral bioavailability fraction (0, 1].
    dosing_interval_h : Dosing interval (h).
    n_doses_before : Number of doses before holiday starts.
    holiday_duration_h : Duration of treatment interruption (h).
    n_doses_after : Number of doses after restart.
    mec_mg_L : Minimum effective concentration (mg/L).
    dt_h : Integration step size (h).

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
    if not (0 < f_oral <= 1):
        raise ValueError("f_oral must be in (0, 1]")
    if dosing_interval_h <= 0:
        raise ValueError("dosing_interval_h must be > 0")
    if holiday_duration_h <= 0:
        raise ValueError("holiday_duration_h must be > 0")
    if n_doses_before < 1:
        raise ValueError("n_doses_before must be >= 1")
    if n_doses_after < 1:
        raise ValueError("n_doses_after must be >= 1")

    ke = cl_L_per_h / vd_L  # elimination rate constant (1/h)

    # Total simulation time
    holiday_start_h = n_doses_before * dosing_interval_h
    restart_h = holiday_start_h + holiday_duration_h
    total_h = restart_h + n_doses_after * dosing_interval_h

    n_steps = max(int(round(total_h / dt_h)), 1)
    times_h: list[float] = []
    c_plasma: list[float] = []

    # State: central compartment amount (mg) and gut amount (mg)
    a_gut = 0.0
    a_central = 0.0

    # Schedule dosing events: times at which doses are administered
    dose_schedule: list[float] = []
    # Before holiday
    for i in range(n_doses_before):
        dose_schedule.append(i * dosing_interval_h)
    # After restart
    for i in range(n_doses_after):
        dose_schedule.append(restart_h + i * dosing_interval_h)

    # Forward Euler integration
    t = 0.0
    # Track dose administration — use index approach to handle floating-point
    next_dose_idx = 0
    sorted_doses = sorted(dose_schedule)

    for step in range(n_steps + 1):
        t = step * dt_h

        # Administer doses at scheduled times (within dt/2 tolerance)
        while next_dose_idx < len(sorted_doses):
            dose_t = sorted_doses[next_dose_idx]
            if t >= dose_t - dt_h * 0.5 and t < dose_t + dt_h * 0.5:
                a_gut += f_oral * dose_mg
                next_dose_idx += 1
            elif t < dose_t - dt_h * 0.5:
                break
            else:
                next_dose_idx += 1

        c = a_central / vd_L
        times_h.append(t)
        c_plasma.append(max(c, 0.0))

        # Forward Euler step (avoid computing past last step)
        if step < n_steps:
            d_gut = -ka_per_h * a_gut
            d_central = ka_per_h * a_gut - ke * a_central
            a_gut = max(a_gut + d_gut * dt_h, 0.0)
            a_central = max(a_central + d_central * dt_h, 0.0)

    # Compute css_before: average concentration over the last 3 dose intervals before holiday
    n_avg_intervals = min(3, n_doses_before)
    css_start_h = holiday_start_h - n_avg_intervals * dosing_interval_h
    css_indices = [i for i, t in enumerate(times_h) if css_start_h <= t <= holiday_start_h]
    if css_indices:
        css_before = sum(c_plasma[i] for i in css_indices) / len(css_indices)
    else:
        css_before = c_plasma[0] if c_plasma else 0.0

    # Find holiday indices (from holiday_start to restart)
    holiday_indices = [i for i, t in enumerate(times_h) if holiday_start_h <= t <= restart_h]

    # c_nadir during holiday
    if holiday_indices:
        holiday_concs = [c_plasma[i] for i in holiday_indices]
        c_nadir = min(holiday_concs)
        nadir_local_idx = holiday_concs.index(c_nadir)
        time_to_nadir = times_h[holiday_indices[nadir_local_idx]] - holiday_start_h
    else:
        c_nadir = 0.0
        time_to_nadir = 0.0

    # c at end of holiday (just before restart)
    restart_idx = min(range(len(times_h)), key=lambda i: abs(times_h[i] - restart_h))
    c_at_holiday_end = c_plasma[restart_idx]

    # Rebound Cmax: peak after restart
    post_restart_indices = [i for i, t in enumerate(times_h) if t >= restart_h]
    if post_restart_indices:
        rebound_cmax = max(c_plasma[i] for i in post_restart_indices)
    else:
        rebound_cmax = 0.0

    # Time in subtherapeutic (below MEC) — trapezoidal
    time_in_sub = 0.0
    if mec_mg_L > 0:
        for i in range(len(times_h) - 1):
            c0 = c_plasma[i]
            c1 = c_plasma[i + 1]
            dt = times_h[i + 1] - times_h[i]
            below0 = 1.0 if c0 < mec_mg_L else 0.0
            below1 = 1.0 if c1 < mec_mg_L else 0.0
            time_in_sub += (below0 + below1) / 2.0 * dt

    # Therapeutic window restored: time after restart to first reach MEC
    therapeutic_window_restored = 0.0
    if mec_mg_L > 0:
        found = False
        for i in post_restart_indices:
            if c_plasma[i] >= mec_mg_L:
                therapeutic_window_restored = times_h[i] - restart_h
                found = True
                break
        if not found:
            # Never restored
            therapeutic_window_restored = total_h - restart_h

    t_half = math.log(2) / ke if ke > 0 else float("inf")
    notes = (
        f"ke={ke:.3f}/h, t_half={t_half:.1f}h; "
        f"holiday {holiday_start_h:.1f}–{restart_h:.1f}h; "
        f"css_before={css_before:.3f}mg/L"
    )

    return DrugHolidayResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        dosing_interval_h=dosing_interval_h,
        holiday_start_h=holiday_start_h,
        holiday_duration_h=holiday_duration_h,
        times_h=times_h,
        c_plasma_mg_L=c_plasma,
        css_before_mg_L=css_before,
        c_nadir_mg_L=c_nadir,
        time_to_nadir_h=time_to_nadir,
        c_at_holiday_end_mg_L=c_at_holiday_end,
        rebound_cmax_mg_L=rebound_cmax,
        time_in_subtherapeutic_h=time_in_sub,
        therapeutic_window_restored_h=therapeutic_window_restored,
        notes=notes,
    )


__all__ = [
    "DrugHolidayResult",
    "simulate_drug_holiday",
]
