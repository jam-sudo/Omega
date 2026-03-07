"""Drug holiday / structured treatment interruption PK model — Phase 266.

Models concentration-time profiles during and after drug discontinuation,
including wash-out kinetics, rebound effects, and return to therapeutic levels.
Relevant for MS, HIV, oncology, and psychiatric medications.

References
----------
- Havlir DV et al., N Engl J Med. 2000;343:1261-8 (HIV STI)
- Berger JR et al., Ann Neurol. 2009;65(4):385-90 (natalizumab washout)
"""

from __future__ import annotations

from dataclasses import dataclass

from omega_pbpk._compat import np_trapz  # noqa: F401  (available for callers)

__all__ = ["DrugHolidayResult", "simulate_drug_holiday"]


@dataclass(frozen=True)
class DrugHolidayResult:
    """Result from drug holiday simulation."""

    drug_name: str
    dose_mg: float
    dosing_interval_h: float
    holiday_start_h: float
    holiday_duration_h: float
    times_h: list[float]
    c_plasma_mg_L: list[float]
    c_at_holiday_start: float
    c_at_holiday_end: float
    c_min_during_holiday: float
    time_below_mic_h: float  # Time drug < minimum effective concentration
    washout_90pct_h: float  # Time to 90% washout from holiday start
    return_to_ss_h: float  # Time to return to steady-state (after restart)
    n_missed_doses: int
    notes: str


def simulate_drug_holiday(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    dosing_interval_h: float = 12.0,
    n_doses_before: int = 10,
    holiday_duration_h: float = 48.0,
    n_doses_after: int = 5,
    ka_per_h: float = 1.0,
    f_oral: float = 1.0,
    min_effective_c_mg_L: float = 0.1,
    dt_h: float = 0.1,
) -> DrugHolidayResult:
    """Simulate drug concentration during and after a structured holiday.

    Parameters
    ----------
    drug_name : Drug name.
    dose_mg : Dose per administration (mg). Must be > 0.
    cl_L_per_h : Clearance (L/h). Must be > 0.
    vd_L : Volume of distribution (L). Must be > 0.
    dosing_interval_h : Dosing interval (h). Must be > 0.
    n_doses_before : Number of doses before holiday (for steady state). Must be >= 1.
    holiday_duration_h : Duration of drug holiday (h). Must be > 0.
    n_doses_after : Number of doses after holiday restart. Must be >= 1.
    ka_per_h : Oral absorption rate constant (h^-1). Must be > 0.
    f_oral : Oral bioavailability (0, 1]. Must be in (0, 1].
    min_effective_c_mg_L : Minimum effective concentration (mg/L) for time-below calc.
    dt_h : Time step (h).

    Returns
    -------
    DrugHolidayResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if dosing_interval_h <= 0:
        raise ValueError("dosing_interval_h must be > 0")
    if n_doses_before < 1:
        raise ValueError("n_doses_before must be >= 1")
    if holiday_duration_h <= 0:
        raise ValueError("holiday_duration_h must be > 0")
    if n_doses_after < 1:
        raise ValueError("n_doses_after must be >= 1")
    if ka_per_h <= 0:
        raise ValueError("ka_per_h must be > 0")
    if not (0 < f_oral <= 1):
        raise ValueError("f_oral must be in (0, 1]")

    ke = cl_L_per_h / vd_L
    holiday_start_h = n_doses_before * dosing_interval_h
    t_end_h = holiday_start_h + holiday_duration_h + n_doses_after * dosing_interval_h

    n_steps = int(t_end_h / dt_h) + 1
    times = [i * dt_h for i in range(n_steps)]
    c_plasma = [0.0] * n_steps

    # Dose schedule
    dose_times_steps: set[int] = set()
    for d in range(n_doses_before):
        step = round(d * dosing_interval_h / dt_h)
        dose_times_steps.add(step)
    restart_h = holiday_start_h + holiday_duration_h
    for d in range(n_doses_after):
        step = round((restart_h + d * dosing_interval_h) / dt_h)
        dose_times_steps.add(step)

    # Simulate
    a_gut = 0.0
    for i in range(n_steps):
        if i in dose_times_steps:
            a_gut += dose_mg * f_oral
        if i > 0:
            abs_rate = ka_per_h * a_gut
            elim = ke * c_plasma[i - 1] * vd_L
            dc = (abs_rate - elim) / vd_L * dt_h
            a_gut = max(0.0, a_gut - abs_rate * dt_h)
            c_plasma[i] = max(0.0, c_plasma[i - 1] + dc)

    # Metrics
    holiday_start_idx = round(holiday_start_h / dt_h)
    holiday_end_idx = round((holiday_start_h + holiday_duration_h) / dt_h)
    holiday_end_idx = min(holiday_end_idx, n_steps - 1)

    c_at_start = c_plasma[holiday_start_idx]
    c_at_end = c_plasma[holiday_end_idx]

    holiday_concs = c_plasma[holiday_start_idx : holiday_end_idx + 1]
    c_min_holiday = min(holiday_concs) if holiday_concs else 0.0

    # Time below MEC during holiday
    time_below = sum(dt_h for c in holiday_concs if c < min_effective_c_mg_L)

    # Washout 90%: time from holiday start for conc to drop to 10% of start value
    target_90pct = c_at_start * 0.1
    washout_90 = holiday_duration_h  # default
    for i in range(holiday_start_idx, holiday_end_idx + 1):
        if c_plasma[i] <= target_90pct:
            washout_90 = times[i] - holiday_start_h
            break

    # Return to SS: time after restart for conc to recover to 80% of pre-holiday trough
    # Pre-holiday trough = c at step before first dose
    pre_holiday_trough = c_plasma[max(0, holiday_start_idx - int(dosing_interval_h / dt_h))]
    target_ss = 0.8 * pre_holiday_trough if pre_holiday_trough > 0 else 0.0
    return_ss_h = n_doses_after * dosing_interval_h  # default
    restart_idx = holiday_end_idx
    for i in range(restart_idx, n_steps):
        if c_plasma[i] >= target_ss:
            return_ss_h = times[i] - (holiday_start_h + holiday_duration_h)
            break

    n_missed = round(holiday_duration_h / dosing_interval_h)

    notes = (
        f"{drug_name}: {n_doses_before} doses, then {holiday_duration_h:.0f}h holiday, "
        f"then {n_doses_after} doses. "
        f"C_start={c_at_start:.3f}, C_end={c_at_end:.4f} mg/L. "
        f"Time<MEC={time_below:.1f}h. Washout 90%={washout_90:.1f}h."
    )

    return DrugHolidayResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        dosing_interval_h=dosing_interval_h,
        holiday_start_h=holiday_start_h,
        holiday_duration_h=holiday_duration_h,
        times_h=times,
        c_plasma_mg_L=c_plasma,
        c_at_holiday_start=c_at_start,
        c_at_holiday_end=c_at_end,
        c_min_during_holiday=c_min_holiday,
        time_below_mic_h=time_below,
        washout_90pct_h=washout_90,
        return_to_ss_h=return_ss_h,
        n_missed_doses=n_missed,
        notes=notes,
    )
