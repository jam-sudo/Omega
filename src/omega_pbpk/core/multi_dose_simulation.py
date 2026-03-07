"""Multi-dose simulation with superposition for any dosing regimen.

Uses forward Euler 1-compartment model with gut depot for oral dosing.
Superposition approach: at each dose time, drug is added to appropriate
compartment and ODE continues forward.

References:
  - Rowland & Tozer, Clinical Pharmacokinetics, 5th Ed.
  - FDA Guidance on Multiple Dose Pharmacokinetics
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class MultiDoseResult:
    """Result of multi-dose simulation."""

    drug_name: str
    dose_mg: float
    tau_h: float
    n_doses: int
    times_h: list
    c_plasma_mg_L: list
    css_max_mg_L: float  # steady-state Cmax (last dose)
    css_min_mg_L: float  # steady-state trough (last dose)
    css_avg_mg_L: float  # AUC_last / tau
    accumulation_ratio: float  # css_max / single_dose_cmax
    doses_to_ss: int  # number of doses until within 5% of css
    fluctuation_pct: float  # (css_max - css_min) / css_avg * 100
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _trapezoidal_auc(times: list, concs: list) -> float:
    """Compute AUC via trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (concs[i - 1] + concs[i]) * dt
    return auc


def _simulate_single_interval(
    c0_plasma: float,
    c0_gut: float,
    cl_L_per_h: float,
    vd_L: float,
    ka_per_h: float,
    route: str,
    dose_mg: float,
    f_oral: float,
    tau_h: float,
    dt_h: float,
) -> tuple[list, list, list]:
    """Simulate one dosing interval with forward Euler.

    Returns (times_relative, c_plasma_list, c_gut_list) for the interval.
    Time starts at 0 for this interval.
    """
    ke = cl_L_per_h / vd_L  # elimination rate constant

    # Apply dose at start of interval
    if route == "iv":
        c_plasma = c0_plasma + dose_mg / vd_L
        c_gut = c0_gut
    else:
        c_plasma = c0_plasma
        c_gut = c0_gut + dose_mg * f_oral

    times = [0.0]
    c_pl = [c_plasma]
    c_gt = [c_gut]

    n_steps = max(1, int(round(tau_h / dt_h)))
    t = 0.0
    for _ in range(n_steps):
        if route == "iv":
            dcp_dt = -ke * c_plasma
            dc_gut = 0.0
        else:
            dcp_dt = ka_per_h * c_gut / vd_L - ke * c_plasma
            dc_gut = -ka_per_h * c_gut

        c_plasma = max(0.0, c_plasma + dcp_dt * dt_h)
        c_gut = max(0.0, c_gut + dc_gut * dt_h)
        t = min(t + dt_h, tau_h)

        times.append(t)
        c_pl.append(c_plasma)
        c_gt.append(c_gut)

    return times, c_pl, c_gt


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_multi_dose(
    drug_name: str,
    dose_mg: float,
    tau_h: float,
    n_doses: int = 10,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    f_oral: float = 0.8,
    ka_per_h: float = 1.0,
    route: str = "oral",
    t_end_h: float | None = None,
    dt_h: float = 0.1,
) -> MultiDoseResult:
    """Simulate multi-dose PK via forward Euler 1-compartment model.

    Parameters
    ----------
    drug_name : str
        Drug identifier.
    dose_mg : float
        Dose per administration (mg).
    tau_h : float
        Dosing interval (h).
    n_doses : int
        Number of doses to administer.
    cl_L_per_h : float
        Total clearance (L/h).
    vd_L : float
        Volume of distribution (L).
    f_oral : float
        Oral bioavailability fraction (0-1); ignored for IV route.
    ka_per_h : float
        First-order absorption rate constant (1/h); ignored for IV.
    route : str
        "oral" or "iv".
    t_end_h : float or None
        Simulation end time. Defaults to n_doses * tau_h.
    dt_h : float
        Euler step size (h).

    Returns
    -------
    MultiDoseResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if tau_h <= 0:
        raise ValueError("tau_h must be > 0")
    if n_doses < 1:
        raise ValueError("n_doses must be >= 1")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if not 0.0 < f_oral <= 1.0:
        raise ValueError("f_oral must be in (0, 1]")
    if ka_per_h <= 0:
        raise ValueError("ka_per_h must be > 0")
    if route not in ("oral", "iv"):
        raise ValueError("route must be 'oral' or 'iv'")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")

    if t_end_h is None:
        t_end_h = n_doses * tau_h

    ke = cl_L_per_h / vd_L

    # --- Simulate each dosing interval ---
    all_times: list = []
    all_c_plasma: list = []

    c_plasma = 0.0
    c_gut = 0.0

    # Per-dose tracking
    dose_cmaxes: list = []

    for dose_idx in range(n_doses):
        dose_start_t = dose_idx * tau_h

        # Determine time span for this interval
        # (last interval may extend to t_end_h)
        interval_end = (dose_idx + 1) * tau_h
        if dose_idx == n_doses - 1:
            interval_end = t_end_h
        interval_len = interval_end - dose_start_t

        rel_times, c_pl, c_gt = _simulate_single_interval(
            c_plasma,
            c_gut,
            cl_L_per_h,
            vd_L,
            ka_per_h,
            route,
            dose_mg,
            f_oral,
            interval_len,
            dt_h,
        )

        # Offset times to absolute
        abs_times = [dose_start_t + rt for rt in rel_times]

        # Skip first point for all but first dose (avoid duplicates)
        if dose_idx == 0:
            all_times.extend(abs_times)
            all_c_plasma.extend(c_pl)
        else:
            all_times.extend(abs_times[1:])
            all_c_plasma.extend(c_pl[1:])

        # Cmax in this interval
        dose_cmaxes.append(max(c_pl))

        # Update state for next dose
        c_plasma = c_pl[-1]
        c_gut = c_gt[-1]

    # --- Steady-state metrics (last full interval) ---
    # Find data for last dose interval
    last_dose_start = (n_doses - 1) * tau_h
    last_dose_end = last_dose_start + tau_h

    # Collect points in last interval
    last_times = []
    last_concs = []
    for t, c in zip(all_times, all_c_plasma):
        if last_dose_start <= t <= min(last_dose_end, all_times[-1]):
            last_times.append(t)
            last_concs.append(c)

    if not last_times:
        last_times = [all_times[-1]]
        last_concs = [all_c_plasma[-1]]

    css_max = max(last_concs)
    css_min = min(last_concs)

    # AUC of last interval
    if len(last_times) >= 2:
        last_auc = _trapezoidal_auc(last_times, last_concs)
        actual_tau = last_times[-1] - last_times[0]
        css_avg = last_auc / actual_tau if actual_tau > 0 else css_max
    else:
        css_avg = css_max

    # Accumulation ratio: css_max / single-dose Cmax
    single_dose_cmax = dose_cmaxes[0] if dose_cmaxes else css_max
    if single_dose_cmax > 0:
        accumulation_ratio = css_max / single_dose_cmax
    else:
        accumulation_ratio = 1.0

    # Doses to steady state: when Cmax is within 5% of css_max
    doses_to_ss = n_doses  # default: never reached
    for i in range(1, len(dose_cmaxes)):
        if abs(dose_cmaxes[i] - css_max) / css_max < 0.05 if css_max > 0 else True:
            doses_to_ss = i + 1
            break

    # Fluctuation
    if css_avg > 0:
        fluctuation_pct = (css_max - css_min) / css_avg * 100.0
    else:
        fluctuation_pct = 0.0

    ke_t12 = math.log(2.0) / ke if ke > 0 else float("inf")
    notes = (
        f"ke={ke:.4f}/h, t1/2={ke_t12:.2f}h, route={route}, "
        f"F={f_oral:.2f} (oral only), {n_doses} doses at tau={tau_h}h"
    )

    return MultiDoseResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        tau_h=tau_h,
        n_doses=n_doses,
        times_h=all_times,
        c_plasma_mg_L=all_c_plasma,
        css_max_mg_L=float(css_max),
        css_min_mg_L=float(css_min),
        css_avg_mg_L=float(css_avg),
        accumulation_ratio=float(accumulation_ratio),
        doses_to_ss=int(doses_to_ss),
        fluctuation_pct=float(fluctuation_pct),
        notes=notes,
    )


def simulate_loading_maintenance(
    drug_name: str,
    loading_dose_mg: float,
    maintenance_dose_mg: float,
    tau_h: float,
    n_maintenance: int = 5,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    f_oral: float = 0.8,
    ka_per_h: float = 1.0,
    route: str = "oral",
    dt_h: float = 0.1,
) -> MultiDoseResult:
    """Simulate loading dose followed by maintenance dosing.

    Parameters
    ----------
    drug_name : str
        Drug identifier.
    loading_dose_mg : float
        Loading dose (mg); given at t=0.
    maintenance_dose_mg : float
        Maintenance dose (mg); repeated every tau_h.
    tau_h : float
        Dosing interval (h).
    n_maintenance : int
        Number of maintenance doses after loading dose.
    cl_L_per_h : float
        Total clearance (L/h).
    vd_L : float
        Volume of distribution (L).
    f_oral : float
        Oral bioavailability (0-1).
    ka_per_h : float
        Absorption rate constant (1/h).
    route : str
        "oral" or "iv".
    dt_h : float
        Euler step size (h).

    Returns
    -------
    MultiDoseResult
        Based on loading + maintenance combined regimen.
        n_doses reflects total doses (1 loading + n_maintenance).
    """
    if loading_dose_mg <= 0:
        raise ValueError("loading_dose_mg must be > 0")
    if maintenance_dose_mg <= 0:
        raise ValueError("maintenance_dose_mg must be > 0")
    if tau_h <= 0:
        raise ValueError("tau_h must be > 0")
    if n_maintenance < 1:
        raise ValueError("n_maintenance must be >= 1")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if not 0.0 < f_oral <= 1.0:
        raise ValueError("f_oral must be in (0, 1]")
    if ka_per_h <= 0:
        raise ValueError("ka_per_h must be > 0")
    if route not in ("oral", "iv"):
        raise ValueError("route must be 'oral' or 'iv'")

    n_total = 1 + n_maintenance
    ke = cl_L_per_h / vd_L

    all_times: list = []
    all_c_plasma: list = []
    dose_cmaxes: list = []

    c_plasma = 0.0
    c_gut = 0.0

    for dose_idx in range(n_total):
        current_dose = loading_dose_mg if dose_idx == 0 else maintenance_dose_mg
        dose_start_t = dose_idx * tau_h

        rel_times, c_pl, c_gt = _simulate_single_interval(
            c_plasma,
            c_gut,
            cl_L_per_h,
            vd_L,
            ka_per_h,
            route,
            current_dose,
            f_oral,
            tau_h,
            dt_h,
        )

        abs_times = [dose_start_t + rt for rt in rel_times]

        if dose_idx == 0:
            all_times.extend(abs_times)
            all_c_plasma.extend(c_pl)
        else:
            all_times.extend(abs_times[1:])
            all_c_plasma.extend(c_pl[1:])

        dose_cmaxes.append(max(c_pl))
        c_plasma = c_pl[-1]
        c_gut = c_gt[-1]

    # Steady-state metrics from last interval
    last_dose_start = (n_total - 1) * tau_h
    last_dose_end = last_dose_start + tau_h

    last_times = []
    last_concs = []
    for t, c in zip(all_times, all_c_plasma):
        if last_dose_start <= t <= last_dose_end:
            last_times.append(t)
            last_concs.append(c)

    if not last_times:
        last_times = [all_times[-1]]
        last_concs = [all_c_plasma[-1]]

    css_max = max(last_concs)
    css_min = min(last_concs)

    if len(last_times) >= 2:
        last_auc = _trapezoidal_auc(last_times, last_concs)
        actual_tau = last_times[-1] - last_times[0]
        css_avg = last_auc / actual_tau if actual_tau > 0 else css_max
    else:
        css_avg = css_max

    single_dose_cmax = dose_cmaxes[0] if dose_cmaxes else css_max
    accumulation_ratio = css_max / single_dose_cmax if single_dose_cmax > 0 else 1.0

    doses_to_ss = n_total
    for i in range(1, len(dose_cmaxes)):
        if css_max > 0 and abs(dose_cmaxes[i] - css_max) / css_max < 0.05:
            doses_to_ss = i + 1
            break

    fluctuation_pct = (css_max - css_min) / css_avg * 100.0 if css_avg > 0 else 0.0

    ke_t12 = math.log(2.0) / ke if ke > 0 else float("inf")
    notes = (
        f"Loading={loading_dose_mg}mg + {n_maintenance}x maintenance={maintenance_dose_mg}mg; "
        f"ke={ke:.4f}/h, t1/2={ke_t12:.2f}h, route={route}"
    )

    return MultiDoseResult(
        drug_name=drug_name,
        dose_mg=maintenance_dose_mg,
        tau_h=tau_h,
        n_doses=n_total,
        times_h=all_times,
        c_plasma_mg_L=all_c_plasma,
        css_max_mg_L=float(css_max),
        css_min_mg_L=float(css_min),
        css_avg_mg_L=float(css_avg),
        accumulation_ratio=float(accumulation_ratio),
        doses_to_ss=int(doses_to_ss),
        fluctuation_pct=float(fluctuation_pct),
        notes=notes,
    )


__all__ = [
    "MultiDoseResult",
    "simulate_multi_dose",
    "simulate_loading_maintenance",
]
