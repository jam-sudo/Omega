"""Enzyme saturation PK simulation with Michaelis-Menten elimination.

Simulates nonlinear PK where drug elimination follows Michaelis-Menten
kinetics, potentially combined with additional linear clearance.

ODE system (1-compartment):
  IV bolus:
    dC/dt = -Vmax * C / (Km + C) - (CL_linear / Vd) * C

  Oral:
    d(gut)/dt = -ka * gut
    dC/dt = ka * gut / Vd - Vmax * C / (Km + C) - (CL_linear / Vd) * C

At low C (C << Km): apparent CL = Vmax/Km (first-order)
At high C (C >> Km): elimination approaches zero-order rate Vmax

References
----------
- Gibaldi M, Perrier D. Pharmacokinetics. 2nd ed. 1982.
- Wagner JG. Fundamentals of Clinical Pharmacokinetics. 1975.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class EnzymeSaturationPKResult:
    """Results from enzyme saturation PK simulation.

    Attributes
    ----------
    drug_name : Name of the drug compound.
    dose_mg : Administered dose (mg).
    route : Administration route ('iv' or 'oral').
    vmax_mg_per_h : Maximum elimination rate (mg/h).
    km_mg_per_L : Michaelis constant (mg/L).
    times_h : Simulation time points (h).
    c_plasma_mg_L : Plasma concentration vs time (mg/L).
    cmax_mg_L : Peak plasma concentration (mg/L).
    tmax_h : Time of peak concentration (h).
    auc_mg_h_per_L : Area under concentration-time curve (mg*h/L).
    t_half_terminal_h : Terminal half-life estimated from last 20% of curve (h).
    dose_proportional : True if AUC scales linearly with dose (False for pure MM).
    saturation_fraction_at_cmax : Fraction of enzyme saturated at Cmax
        (= Cmax / (Km + Cmax)).
    notes : Summary of key findings.
    """

    drug_name: str
    dose_mg: float
    route: str
    vmax_mg_per_h: float
    km_mg_per_L: float
    times_h: list
    c_plasma_mg_L: list
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    t_half_terminal_h: float
    dose_proportional: bool
    saturation_fraction_at_cmax: float
    notes: str


def _trapezoidal_auc(times: list[float], concs: list[float]) -> float:
    """Compute AUC using manual trapezoidal integration."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (concs[i] + concs[i - 1]) * dt
    return auc


def _terminal_half_life(times: list[float], concs: list[float]) -> float:
    """Estimate terminal half-life from last 20% of the time course.

    Uses linear regression on log(C) vs t in the terminal phase.
    Returns t_half = ln(2) / ke_terminal.
    """
    n = len(times)
    if n < 4:
        return float("nan")

    start_idx = max(1, int(0.8 * n))
    t_seg = times[start_idx:]
    c_seg = concs[start_idx:]

    # Filter out zero or negative concentrations
    valid = [(t, c) for t, c in zip(t_seg, c_seg) if c > 0]
    if len(valid) < 2:
        return float("nan")

    t_vals = [v[0] for v in valid]
    log_c = [math.log(v[1]) for v in valid]

    # Linear regression: log(C) = log(C0) - ke * t
    n_pts = len(t_vals)
    mean_t = sum(t_vals) / n_pts
    mean_lc = sum(log_c) / n_pts

    ss_xy = sum((t_vals[i] - mean_t) * (log_c[i] - mean_lc) for i in range(n_pts))
    ss_xx = sum((t_vals[i] - mean_t) ** 2 for i in range(n_pts))

    if ss_xx < 1e-12:
        return float("nan")

    slope = ss_xy / ss_xx  # slope = -ke
    ke_terminal = -slope

    if ke_terminal <= 0:
        return float("nan")

    return math.log(2.0) / ke_terminal


def simulate_enzyme_saturation_pk(
    drug_name: str,
    dose_mg: float,
    vmax_mg_per_h: float,
    km_mg_per_L: float,
    route: str = "iv",
    cl_linear_L_per_h: float = 0.0,
    vd_L: float = 10.0,
    ka_per_h: float = 1.0,
    f_oral: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> EnzymeSaturationPKResult:
    """Simulate 1-compartment PK with Michaelis-Menten elimination.

    Parameters
    ----------
    drug_name : Name of the drug compound.
    dose_mg : Administered dose (mg).
    vmax_mg_per_h : Maximum elimination rate (mg/h).
    km_mg_per_L : Michaelis constant (mg/L).
    route : Administration route: 'iv' or 'oral'.
    cl_linear_L_per_h : Additional linear clearance (L/h), default 0.
    vd_L : Volume of distribution (L).
    ka_per_h : Oral absorption rate constant (1/h), used only for oral route.
    f_oral : Oral bioavailability fraction (0-1), used only for oral route.
    t_end_h : Simulation end time (h).
    dt_h : Forward Euler time step (h).

    Returns
    -------
    EnzymeSaturationPKResult with simulated PK profile and derived metrics.

    Raises
    ------
    ValueError : If dose_mg, vmax_mg_per_h, km_mg_per_L, or vd_L <= 0,
                 or route not in {'iv', 'oral'}.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if vmax_mg_per_h <= 0:
        raise ValueError("vmax_mg_per_h must be > 0")
    if km_mg_per_L <= 0:
        raise ValueError("km_mg_per_L must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if route not in {"iv", "oral"}:
        raise ValueError("route must be 'iv' or 'oral'")

    n_steps = max(int(t_end_h / dt_h), 1)
    times: list[float] = []
    concs: list[float] = []

    # Linear clearance rate constant
    ke_linear = cl_linear_L_per_h / vd_L  # 1/h

    if route == "iv":
        # Initial condition: IV bolus
        c = dose_mg / vd_L
        for i in range(n_steps + 1):
            t = i * dt_h
            times.append(t)
            concs.append(max(0.0, c))
            if i < n_steps:
                # dC/dt = -Vmax*C/(Km+C)/Vd - ke_linear*C
                mm_rate = vmax_mg_per_h * c / (km_mg_per_L + c)  # mg/h
                dc_dt = -mm_rate / vd_L - ke_linear * c
                c = max(0.0, c + dc_dt * dt_h)
    else:
        # Oral route
        gut = dose_mg * f_oral
        c = 0.0
        for i in range(n_steps + 1):
            t = i * dt_h
            times.append(t)
            concs.append(max(0.0, c))
            if i < n_steps:
                # d(gut)/dt = -ka * gut
                abs_rate = ka_per_h * gut  # mg/h
                # dC/dt = ka*gut/Vd - Vmax*C/(Km+C)/Vd - ke_linear*C
                mm_rate = vmax_mg_per_h * c / (km_mg_per_L + c)  # mg/h
                dc_dt = abs_rate / vd_L - mm_rate / vd_L - ke_linear * c
                gut = max(0.0, gut - abs_rate * dt_h)
                c = max(0.0, c + dc_dt * dt_h)

    auc = _trapezoidal_auc(times, concs)

    # Find Cmax and Tmax
    cmax = 0.0
    tmax_h = 0.0
    for t, c_val in zip(times, concs):
        if c_val > cmax:
            cmax = c_val
            tmax_h = t

    # Terminal half-life
    t_half = _terminal_half_life(times, concs)
    if math.isnan(t_half) or t_half <= 0:
        t_half = 0.0

    # Saturation fraction at Cmax
    sat_frac = cmax / (km_mg_per_L + cmax) if (km_mg_per_L + cmax) > 0 else 0.0

    # dose_proportional: always False for Michaelis-Menten nonlinear elimination
    dose_proportional = False

    notes = (
        f"Drug: {drug_name}, route={route}, dose={dose_mg} mg. "
        f"Vmax={vmax_mg_per_h} mg/h, Km={km_mg_per_L} mg/L. "
        f"Cmax={cmax:.4f} mg/L at t={tmax_h:.2f}h, AUC={auc:.4f} mg*h/L. "
        f"Enzyme saturation at Cmax: {sat_frac * 100:.1f}%. "
        f"Terminal t1/2={t_half:.2f}h. "
        f"Nonlinear (dose-disproportional) PK expected."
    )

    return EnzymeSaturationPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        vmax_mg_per_h=vmax_mg_per_h,
        km_mg_per_L=km_mg_per_L,
        times_h=times,
        c_plasma_mg_L=concs,
        cmax_mg_L=cmax,
        tmax_h=tmax_h,
        auc_mg_h_per_L=auc,
        t_half_terminal_h=t_half,
        dose_proportional=dose_proportional,
        saturation_fraction_at_cmax=sat_frac,
        notes=notes,
    )


def compare_doses_nonlinear(
    drug_name: str,
    doses_mg: list[float],
    vmax_mg_per_h: float,
    km_mg_per_L: float,
    vd_L: float = 10.0,
    route: str = "iv",
    cl_linear_L_per_h: float = 0.0,
    ka_per_h: float = 1.0,
    f_oral: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> list[EnzymeSaturationPKResult]:
    """Compare nonlinear PK across multiple doses.

    Parameters
    ----------
    drug_name : Name of the drug compound.
    doses_mg : List of doses to simulate (mg).
    vmax_mg_per_h : Maximum elimination rate (mg/h).
    km_mg_per_L : Michaelis constant (mg/L).
    vd_L : Volume of distribution (L).
    route : Administration route: 'iv' or 'oral'.
    cl_linear_L_per_h : Additional linear clearance (L/h).
    ka_per_h : Oral absorption rate constant (1/h).
    f_oral : Oral bioavailability fraction.
    t_end_h : Simulation end time (h).
    dt_h : Forward Euler time step (h).

    Returns
    -------
    List of EnzymeSaturationPKResult sorted by dose_mg ascending.
    """
    results: list[EnzymeSaturationPKResult] = []
    for dose in doses_mg:
        result = simulate_enzyme_saturation_pk(
            drug_name=drug_name,
            dose_mg=dose,
            vmax_mg_per_h=vmax_mg_per_h,
            km_mg_per_L=km_mg_per_L,
            route=route,
            cl_linear_L_per_h=cl_linear_L_per_h,
            vd_L=vd_L,
            ka_per_h=ka_per_h,
            f_oral=f_oral,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(result)

    results.sort(key=lambda r: r.dose_mg)
    return results


__all__ = [
    "EnzymeSaturationPKResult",
    "simulate_enzyme_saturation_pk",
    "compare_doses_nonlinear",
]
