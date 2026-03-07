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


def _oral_conc_pk(
    t_since_dose: float,
    dose_mg: float,
    ka: float,
    ke: float,
    vd: float,
    f: float,
) -> float:
    """Oral 1-compartment concentration at time t_since_dose >= 0."""
    if t_since_dose < 0.0:
        return 0.0
    if abs(ka - ke) < 1e-9:
        c = f * dose_mg / vd * ka * t_since_dose * math.exp(-ke * t_since_dose)
    else:
        c = (f * dose_mg * ka / (vd * (ka - ke))) * (
            math.exp(-ke * t_since_dose) - math.exp(-ka * t_since_dose)
        )
    return max(0.0, c)


@dataclass
class AccumulationIndexResult:
    """Result of multi-dose accumulation index calculation.

    Attributes
    ----------
    drug_name : str
    dose_mg : float
    tau_h : float          Dosing interval (hours).
    n_doses : int          Number of doses simulated.
    ke_per_h : float       Elimination rate constant (1/h).
    t_half_h : float       Elimination half-life (h).
    rac : float            Accumulation ratio (trough-based, exact formula).
    cmax_ss : float        Predicted steady-state peak concentration (mg/L).
    cmin_ss : float        Predicted steady-state trough concentration (mg/L).
    css_avg : float        Average steady-state concentration (mg/L).
    fluctuation_pct : float  (Cmax_ss - Cmin_ss) / Css_avg * 100.
    ptf_pct : float        Peak-trough fluctuation: (Cmax_ss - Cmin_ss) / Cmin_ss * 100.
    t_to_ss_h : float      Time to reach steady state (5 * t_half).
    time_course_h : list   Simulation time points (h).
    conc_course_mg_L : list Plasma concentration at each time point (mg/L).
    accumulation_risk : str "low" (Rac<1.5), "moderate" (1.5-3), "high" (>3).
    notes : str
    """

    drug_name: str
    dose_mg: float
    tau_h: float
    n_doses: int
    ke_per_h: float
    t_half_h: float
    rac: float
    cmax_ss: float
    cmin_ss: float
    css_avg: float
    fluctuation_pct: float
    ptf_pct: float
    t_to_ss_h: float
    time_course_h: list
    conc_course_mg_L: list
    accumulation_risk: str
    notes: str


def calculate_accumulation_index(
    drug_name: str,
    dose_mg: float,
    tau_h: float,
    ke_per_h: float = 0.1,
    ka_per_h: float = 1.0,
    vd_L: float = 50.0,
    f_oral: float = 1.0,
    n_doses: int = 10,
    dt_h: float = 0.1,
) -> AccumulationIndexResult:
    """Calculate multi-dose accumulation index.

    Parameters
    ----------
    drug_name : str           Name of the drug.
    dose_mg : float           Dose per administration (mg). Must be > 0.
    tau_h : float             Dosing interval (hours). Must be > 0.
    ke_per_h : float          Elimination rate constant (1/h). Must be > 0.
    ka_per_h : float          Absorption rate constant (1/h). Must be > 0.
    vd_L : float              Volume of distribution (L). Must be > 0.
    f_oral : float            Oral bioavailability fraction (0, 1].
    n_doses : int             Number of doses to simulate.
    dt_h : float              Forward-Euler time step (h).

    Returns
    -------
    AccumulationIndexResult

    Raises
    ------
    ValueError
        If any parameter is out of range.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if tau_h <= 0:
        raise ValueError("tau_h must be > 0")
    if ke_per_h <= 0:
        raise ValueError("ke_per_h must be > 0")
    if ka_per_h <= 0:
        raise ValueError("ka_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if f_oral <= 0 or f_oral > 1.0:
        raise ValueError("f_oral must be in (0, 1]")

    t_half_h = math.log(2.0) / ke_per_h

    # Exact analytical accumulation ratio (trough-based)
    rac = 1.0 / (1.0 - math.exp(-ke_per_h * tau_h))

    # Simulate N doses using superposition (analytical 1-cpt per dose)
    t_end = n_doses * tau_h
    n_steps = max(int(round(t_end / dt_h)), 1)
    times: list[float] = [i * dt_h for i in range(n_steps + 1)]
    conc: list[float] = [0.0] * (n_steps + 1)

    dose_times_sim = [i * tau_h for i in range(n_doses)]
    for i, t in enumerate(times):
        c_total = 0.0
        for t_dose in dose_times_sim:
            dt_since = t - t_dose
            if dt_since >= 0.0:
                c_total += _oral_conc_pk(dt_since, dose_mg, ka_per_h, ke_per_h, vd_L, f_oral)
        conc[i] = c_total

    # Steady-state metrics — use last dosing interval for Cmax_ss / Cmin_ss
    last_dose_step = max(int(round((n_doses - 1) * tau_h / dt_h)), 0)
    last_dose_step = min(last_dose_step, n_steps)
    ss_conc = conc[last_dose_step:]
    cmax_ss = max(ss_conc) if ss_conc else 0.0
    cmin_ss = min(ss_conc) if ss_conc else 0.0

    # Average steady-state concentration (analytical: F*D / (CL * tau))
    css_avg = f_oral * dose_mg / (vd_L * ke_per_h * tau_h)

    # Fluctuation metrics
    fluctuation_pct = (cmax_ss - cmin_ss) / max(css_avg, 1e-12) * 100.0
    ptf_pct = (cmax_ss - cmin_ss) / max(cmin_ss, 1e-12) * 100.0

    t_to_ss_h = 5.0 * t_half_h  # 97% of steady state

    # Accumulation risk classification
    if rac < 1.5:
        accumulation_risk = "low"
    elif rac <= 3.0:
        accumulation_risk = "moderate"
    else:
        accumulation_risk = "high"

    notes = (
        f"{drug_name}: ke={ke_per_h:.3f}/h, t½={t_half_h:.1f}h, "
        f"τ={tau_h}h → Rac={rac:.2f} ({accumulation_risk}). "
        f"Css_avg={css_avg:.3f} mg/L, Cmax_ss={cmax_ss:.3f} mg/L, "
        f"Cmin_ss={cmin_ss:.3f} mg/L. T_to_SS≈{t_to_ss_h:.1f}h."
    )

    return AccumulationIndexResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        tau_h=tau_h,
        n_doses=n_doses,
        ke_per_h=ke_per_h,
        t_half_h=t_half_h,
        rac=rac,
        cmax_ss=cmax_ss,
        cmin_ss=cmin_ss,
        css_avg=css_avg,
        fluctuation_pct=fluctuation_pct,
        ptf_pct=ptf_pct,
        t_to_ss_h=t_to_ss_h,
        time_course_h=times,
        conc_course_mg_L=conc,
        accumulation_risk=accumulation_risk,
        notes=notes,
    )


def compare_dose_intervals(
    drug_name: str,
    dose_mg: float,
    tau_list: list[float],
    **kwargs,
) -> list[AccumulationIndexResult]:
    """Compare accumulation across different dosing intervals.

    Parameters
    ----------
    drug_name : str         Drug name.
    dose_mg : float         Dose (mg).
    tau_list : list[float]  List of dosing intervals (h) to evaluate.
    **kwargs                Additional keyword arguments for calculate_accumulation_index.

    Returns
    -------
    list[AccumulationIndexResult]
        Results sorted by rac ascending (lowest accumulation first).
    """
    results = [calculate_accumulation_index(drug_name, dose_mg, tau, **kwargs) for tau in tau_list]
    return sorted(results, key=lambda r: r.rac)


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
