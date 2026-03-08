"""Phase 862 — Circadian Dosing Optimizer.

Optimize dosing time based on circadian PK variation and PD target.
"""

from dataclasses import dataclass
from math import cos, pi


@dataclass(frozen=True)
class CircadianDosingResult:
    """Result of circadian dosing optimization."""

    drug_name: str
    dose_mg: float
    optimal_dosing_time_h: int
    optimal_cmax_mg_L: float
    optimal_trough_mg_L: float
    optimal_time_above_mec_h: float
    worst_dosing_time_h: int
    worst_cmax_mg_L: float
    cmax_variability_pct: float
    objective: str
    mec_mg_L: float
    notes: str


def _simulate_1cpt(
    dose_mg: float,
    cl_mean: float,
    vd_mean: float,
    a_cl: float,
    phi_cl_h: float,
    a_vd: float,
    phi_vd_h: float,
    dosing_time_h: float,
    mec_mg_L: float,
    t_end_h: float,
    dt_h: float,
) -> dict:
    """Simulate 1-compartment PK with circadian variation using Forward Euler.

    CL(t) = CL_mean * (1 + A_cl * cos(2*pi*(t - phi_cl)/24))
    Vd(t) = Vd_mean * (1 + A_vd * cos(2*pi*(t - phi_vd)/24))

    The dosing_time_h defines which clock hour the dose is given.
    We simulate over t_end_h hours using absolute clock time.

    Returns dict with cmax, trough, time_above_mec_h.
    """
    n_steps = int(round(t_end_h / dt_h))
    c = 0.0
    cmax = 0.0
    time_above_mec = 0.0

    for i in range(n_steps):
        t_local = i * dt_h  # time since dose administration
        t_clock = (dosing_time_h + t_local) % 24.0

        # Circadian CL and Vd
        cl_t = cl_mean * (1.0 + a_cl * cos(2.0 * pi * (t_clock - phi_cl_h) / 24.0))
        vd_t = vd_mean * (1.0 + a_vd * cos(2.0 * pi * (t_clock - phi_vd_h) / 24.0))

        # Bolus dose at t_local = 0
        if i == 0:
            c = dose_mg / vd_t  # mg / L (dose in mg, Vd in L)

        # Elimination: dC/dt = -CL(t)/Vd(t) * C
        ke_t = cl_t / vd_t
        dc_dt = -ke_t * c
        c = max(0.0, c + dc_dt * dt_h)

        if c > cmax:
            cmax = c
        if c >= mec_mg_L:
            time_above_mec += dt_h

    trough = c  # concentration at end of interval

    return {
        "cmax": cmax,
        "trough": trough,
        "time_above_mec_h": time_above_mec,
    }


def optimize_circadian_dosing(
    drug_name: str,
    dose_mg: float,
    cl_mean_L_per_h: float = 5.0,
    vd_mean_L: float = 50.0,
    a_cl: float = 0.2,
    phi_cl_h: float = 14.0,
    a_vd: float = 0.1,
    phi_vd_h: float = 6.0,
    mec_mg_L: float = 0.5,
    objective: str = "minimize_trough",
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> CircadianDosingResult:
    """Optimize dosing time based on circadian PK variation.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose in mg. Must be > 0.
    cl_mean_L_per_h:
        Mean clearance (L/h). Must be > 0.
    vd_mean_L:
        Mean volume of distribution (L). Must be > 0.
    a_cl:
        Amplitude of CL circadian variation (0-0.9).
    phi_cl_h:
        Acrophase of CL (hour of peak CL, 0-24).
    a_vd:
        Amplitude of Vd circadian variation (0-0.9).
    phi_vd_h:
        Acrophase of Vd (hour of peak Vd, 0-24).
    mec_mg_L:
        Minimum effective concentration (mg/L).
    objective:
        "minimize_trough" or "maximize_time_above_mec".
    t_end_h:
        Simulation duration (hours).
    dt_h:
        Forward Euler time step (hours).

    Returns
    -------
    CircadianDosingResult
    """
    if dose_mg <= 0.0:
        raise ValueError("dose_mg must be > 0")
    if cl_mean_L_per_h <= 0.0:
        raise ValueError("cl_mean_L_per_h must be > 0")
    if vd_mean_L <= 0.0:
        raise ValueError("vd_mean_L must be > 0")
    if not (0.0 <= a_cl <= 0.9):
        raise ValueError("a_cl must be in [0, 0.9]")
    if objective not in {"minimize_trough", "maximize_time_above_mec"}:
        raise ValueError("objective must be 'minimize_trough' or 'maximize_time_above_mec'")

    # Evaluate all 24 integer clock hours
    hours = list(range(24))
    results_by_hour = {}
    for h in hours:
        res = _simulate_1cpt(
            dose_mg=dose_mg,
            cl_mean=cl_mean_L_per_h,
            vd_mean=vd_mean_L,
            a_cl=a_cl,
            phi_cl_h=phi_cl_h,
            a_vd=a_vd,
            phi_vd_h=phi_vd_h,
            dosing_time_h=float(h),
            mec_mg_L=mec_mg_L,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results_by_hour[h] = res

    cmax_values = [results_by_hour[h]["cmax"] for h in hours]
    cmax_mean = sum(cmax_values) / len(cmax_values) if cmax_values else 1.0
    cmax_max = max(cmax_values)
    cmax_min = min(cmax_values)
    if cmax_mean > 0.0:
        cmax_variability_pct = 100.0 * (cmax_max - cmax_min) / cmax_mean
    else:
        cmax_variability_pct = 0.0

    # Find optimal and worst based on objective
    if objective == "minimize_trough":
        # optimal = hour with highest trough (we want trough to be as high as possible to avoid sub-MEC)
        optimal_h = max(hours, key=lambda h: results_by_hour[h]["trough"])
        worst_h = min(hours, key=lambda h: results_by_hour[h]["trough"])
    else:  # maximize_time_above_mec
        optimal_h = max(hours, key=lambda h: results_by_hour[h]["time_above_mec_h"])
        worst_h = min(hours, key=lambda h: results_by_hour[h]["time_above_mec_h"])

    opt_res = results_by_hour[optimal_h]
    worst_res = results_by_hour[worst_h]

    notes = (
        f"Evaluated 24 clock hours (0-23h); "
        f"CL amplitude={a_cl:.2f}, acrophase={phi_cl_h:.1f}h; "
        f"Vd amplitude={a_vd:.2f}, acrophase={phi_vd_h:.1f}h; "
        f"Cmax range: {cmax_min:.3f}-{cmax_max:.3f} mg/L; "
        f"Objective: {objective}"
    )

    return CircadianDosingResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        optimal_dosing_time_h=optimal_h,
        optimal_cmax_mg_L=opt_res["cmax"],
        optimal_trough_mg_L=opt_res["trough"],
        optimal_time_above_mec_h=opt_res["time_above_mec_h"],
        worst_dosing_time_h=worst_h,
        worst_cmax_mg_L=worst_res["cmax"],
        cmax_variability_pct=cmax_variability_pct,
        objective=objective,
        mec_mg_L=mec_mg_L,
        notes=notes,
    )


def compare_dosing_schedules(
    drug_name: str,
    dose_mg: float,
    schedules: list,
    cl_mean_L_per_h: float = 5.0,
    vd_mean_L: float = 50.0,
    a_cl: float = 0.2,
    phi_cl_h: float = 14.0,
    a_vd: float = 0.1,
    phi_vd_h: float = 6.0,
    mec_mg_L: float = 0.5,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> list:
    """Compare multiple dosing schedules.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose in mg.
    schedules:
        List of dicts, each with {"name": str, "dosing_time_h": float}.
    cl_mean_L_per_h, vd_mean_L, a_cl, phi_cl_h, a_vd, phi_vd_h, mec_mg_L, t_end_h, dt_h:
        PK parameters (same as optimize_circadian_dosing).

    Returns
    -------
    list[dict]
        List of dicts with "name", "dosing_time_h", "cmax", "trough", "time_above_mec_h",
        sorted by time_above_mec_h descending.
    """
    output = []
    for schedule in schedules:
        name = schedule["name"]
        dosing_time_h = float(schedule["dosing_time_h"])
        res = _simulate_1cpt(
            dose_mg=dose_mg,
            cl_mean=cl_mean_L_per_h,
            vd_mean=vd_mean_L,
            a_cl=a_cl,
            phi_cl_h=phi_cl_h,
            a_vd=a_vd,
            phi_vd_h=phi_vd_h,
            dosing_time_h=dosing_time_h,
            mec_mg_L=mec_mg_L,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        output.append(
            {
                "name": name,
                "dosing_time_h": dosing_time_h,
                "cmax": res["cmax"],
                "trough": res["trough"],
                "time_above_mec_h": res["time_above_mec_h"],
            }
        )

    # Sort by time_above_mec_h descending
    output.sort(key=lambda d: d["time_above_mec_h"], reverse=True)
    return output
