"""Phase 230: Intrathecal drug delivery PK model (3-compartment forward Euler ODE)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IntrathecalPKResult:
    """Result of intrathecal PK simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_lumbar_mg_mL: list
    c_cisternal_mg_mL: list
    c_systemic_mg_L: list
    cmax_lumbar_mg_mL: float
    tmax_cisternal_h: float
    auc_lumbar_mg_h_per_mL: float
    auc_cisternal_mg_h_per_mL: float
    cmax_systemic_mg_L: float
    t_half_lumbar_h: float
    notes: str


def _trapz(y: list, x: list) -> float:
    """Manual trapezoidal AUC."""
    auc = 0.0
    for i in range(1, len(x)):
        auc += 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1])
    return auc


def simulate_intrathecal_pk(
    drug_name: str,
    dose_mg: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    t_end_h: float = 48.0,
    dt_h: float = 0.01,
    v_lumbar_mL: float = 30.0,
    v_cisternal_mL: float = 80.0,
    k_bulk_flow: float = 0.1,
    k_arachnoid: float = 0.05,
    k_csf_turn: float = 0.02,
) -> IntrathecalPKResult:
    """Simulate intrathecal bolus PK.

    Parameters
    ----------
    drug_name:
        Name of drug.
    dose_mg:
        Intrathecal bolus dose (mg).
    cl_sys_L_per_h:
        Systemic clearance (L/h).
    vd_sys_L:
        Systemic volume of distribution (L).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler step size (h).
    v_lumbar_mL:
        Lumbar CSF volume (mL).
    v_cisternal_mL:
        Cisternal/ventricular CSF volume (mL).
    k_bulk_flow:
        CSF bulk flow rate constant (/h).
    k_arachnoid:
        Arachnoid absorption rate constant (/h).
    k_csf_turn:
        Cisternal turnover rate constant (/h).
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")

    # systemic elimination rate constant
    ke = cl_sys_L_per_h / vd_sys_L  # /h

    # initial conditions
    c_lumb = dose_mg / v_lumbar_mL  # mg/mL
    c_cist = 0.0  # mg/mL
    c_sys = 0.0  # mg/L

    times: list[float] = [0.0]
    c_lumb_arr: list[float] = [c_lumb]
    c_cist_arr: list[float] = [c_cist]
    c_sys_arr: list[float] = [c_sys]

    t = 0.0
    n_steps = int(t_end_h / dt_h)

    for _ in range(n_steps):
        d_lumb = (-k_bulk_flow * c_lumb - k_arachnoid * c_lumb) * dt_h
        d_cist = (k_bulk_flow * c_lumb * v_lumbar_mL / v_cisternal_mL - k_csf_turn * c_cist) * dt_h
        # rate_in: k * C [mg/mL] * V [mL] = mg/h; divide by Vd [L] → mg/L/h
        rate_in_sys_mg_per_h = (
            k_arachnoid * c_lumb * v_lumbar_mL + k_csf_turn * c_cist * v_cisternal_mL
        )
        d_sys_proper = (rate_in_sys_mg_per_h / vd_sys_L - ke * c_sys) * dt_h

        c_lumb = max(0.0, c_lumb + d_lumb)
        c_cist = max(0.0, c_cist + d_cist)
        c_sys = max(0.0, c_sys + d_sys_proper)
        t += dt_h

        times.append(round(t, 6))
        c_lumb_arr.append(c_lumb)
        c_cist_arr.append(c_cist)
        c_sys_arr.append(c_sys)

    # Derived metrics
    cmax_lumbar = max(c_lumb_arr)
    cmax_systemic = max(c_sys_arr)

    # tmax cisternal: time of peak
    max_cist = max(c_cist_arr)
    tmax_cist_h = times[c_cist_arr.index(max_cist)]

    auc_lumbar = _trapz(c_lumb_arr, times)
    auc_cist = _trapz(c_cist_arr, times)

    # t_half lumbar: time for lumbar concentration to fall to half of cmax
    half = cmax_lumbar / 2.0
    t_half_lumbar = float("nan")
    for i in range(1, len(c_lumb_arr)):
        if c_lumb_arr[i] <= half:
            # interpolate
            frac = (c_lumb_arr[i - 1] - half) / (c_lumb_arr[i - 1] - c_lumb_arr[i])
            t_half_lumbar = times[i - 1] + frac * (times[i] - times[i - 1])
            break

    if t_half_lumbar != t_half_lumbar:  # NaN fallback
        # estimate from rate constants
        k_tot = k_bulk_flow + k_arachnoid
        t_half_lumbar = 0.693 / k_tot if k_tot > 0 else float("nan")

    return IntrathecalPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_lumbar_mg_mL=c_lumb_arr,
        c_cisternal_mg_mL=c_cist_arr,
        c_systemic_mg_L=c_sys_arr,
        cmax_lumbar_mg_mL=round(cmax_lumbar, 6),
        tmax_cisternal_h=round(tmax_cist_h, 4),
        auc_lumbar_mg_h_per_mL=round(auc_lumbar, 4),
        auc_cisternal_mg_h_per_mL=round(auc_cist, 6),
        cmax_systemic_mg_L=round(cmax_systemic, 6),
        t_half_lumbar_h=round(t_half_lumbar, 4),
        notes=f"Forward Euler dt={dt_h}h; ke={ke:.4f}/h",
    )


def compare_it_doses(
    drug_name: str,
    doses_mg: list,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    t_end_h: float = 48.0,
) -> list[IntrathecalPKResult]:
    """Simulate multiple IT doses and sort by lumbar AUC descending."""
    results = [
        simulate_intrathecal_pk(
            drug_name=drug_name,
            dose_mg=d,
            cl_sys_L_per_h=cl_sys_L_per_h,
            vd_sys_L=vd_sys_L,
            t_end_h=t_end_h,
        )
        for d in doses_mg
    ]
    return sorted(results, key=lambda r: r.auc_lumbar_mg_h_per_mL, reverse=True)
