"""Non-compartmental analysis (NCA) — comprehensive PK parameter estimation."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["NCAResult", "nca_iv_bolus", "nca_oral", "fit_terminal_slope"]


@dataclass(frozen=True)
class NCAResult:
    """Results from non-compartmental analysis."""

    route: str
    cmax_mg_L: float
    tmax_h: float
    auc0_t: float
    auc0_inf: float
    aumc0_inf: float
    lambda_z: float
    t_half_h: float
    cl_L_per_h: float
    vd_L: float
    mrt_h: float
    r_squared_terminal: float
    notes: str


def _trapezoidal_auc(times: list[float], concs: list[float]) -> float:
    """Compute AUC using linear trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (concs[i - 1] + concs[i]) * dt
    return auc


def _trapezoidal_aumc(times: list[float], concs: list[float]) -> float:
    """Compute AUMC (integral of t*C) using linear trapezoidal rule."""
    aumc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        tc_left = times[i - 1] * concs[i - 1]
        tc_right = times[i] * concs[i]
        aumc += 0.5 * (tc_left + tc_right) * dt
    return aumc


def fit_terminal_slope(
    times_h: list[float], concs_mg_L: list[float], n_points: int = 3
) -> tuple[float, float, float]:
    """Fit terminal elimination slope using OLS on log-linear data.

    Returns (lambda_z, intercept, r_squared) from log(C) = intercept - lambda_z * t.
    Uses last n_points with positive concentrations.
    """
    # Filter positive concentrations and get last n_points
    pairs = [(t, c) for t, c in zip(times_h, concs_mg_L) if c > 0]
    if len(pairs) < 3:
        raise ValueError("At least 3 positive concentrations required for terminal slope fit.")

    terminal = pairs[-n_points:]
    n = len(terminal)

    ts = [p[0] for p in terminal]
    log_cs = [math.log(p[1]) for p in terminal]

    # OLS: log(C) = intercept - lambda_z * t
    sum_t = sum(ts)
    sum_logc = sum(log_cs)
    sum_t2 = sum(t * t for t in ts)
    sum_t_logc = sum(ts[i] * log_cs[i] for i in range(n))

    denom = n * sum_t2 - sum_t * sum_t
    if abs(denom) < 1e-12:
        raise ValueError("Cannot fit terminal slope: degenerate time points.")

    slope = (n * sum_t_logc - sum_t * sum_logc) / denom
    intercept = (sum_logc - slope * sum_t) / n

    # lambda_z is negative slope (elimination rate)
    lambda_z = -slope

    # R-squared
    mean_logc = sum_logc / n
    ss_tot = sum((lc - mean_logc) ** 2 for lc in log_cs)
    ss_res = sum((log_cs[i] - (intercept + slope * ts[i])) ** 2 for i in range(n))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-14 else 1.0

    return lambda_z, intercept, r_squared


def _validate_inputs(times_h: list[float], concs_mg_L: list[float]) -> None:
    """Validate NCA input data."""
    if len(times_h) != len(concs_mg_L):
        raise ValueError(
            f"times_h and concs_mg_L must have same length, "
            f"got {len(times_h)} and {len(concs_mg_L)}."
        )
    if len(times_h) < 3:
        raise ValueError("At least 3 data points required for NCA.")
    for t in times_h:
        if t < 0:
            raise ValueError(f"All times must be >= 0, got {t}.")


def nca_iv_bolus(
    times_h: list[float],
    concs_mg_L: list[float],
    dose_mg: float = 100.0,
) -> NCAResult:
    """Non-compartmental analysis for IV bolus data.

    Parameters
    ----------
    times_h:
        Time points in hours.
    concs_mg_L:
        Plasma concentrations in mg/L.
    dose_mg:
        Administered dose in mg.

    Returns
    -------
    NCAResult
    """
    _validate_inputs(times_h, concs_mg_L)

    # C0 = first concentration
    c0 = concs_mg_L[0]
    cmax = c0
    tmax = times_h[0]

    # AUC0_t (trapezoidal)
    auc0_t = _trapezoidal_auc(times_h, concs_mg_L)

    # Terminal slope
    lambda_z, intercept, r_sq = fit_terminal_slope(times_h, concs_mg_L)

    # AUC extrapolation to infinity
    c_last = concs_mg_L[-1]
    t_last = times_h[-1]
    auc_tlast_inf = c_last / lambda_z if lambda_z > 0 else 0.0
    auc0_inf = auc0_t + auc_tlast_inf

    # Half-life
    t_half = math.log(2.0) / lambda_z if lambda_z > 0 else float("inf")

    # Clearance
    cl = dose_mg / auc0_inf if auc0_inf > 0 else 0.0

    # Volume of distribution (Vd = dose / (lambda_z * AUC0_inf))
    vd = dose_mg / (lambda_z * auc0_inf) if (lambda_z > 0 and auc0_inf > 0) else 0.0

    # AUMC0_t (trapezoidal)
    aumc0_t = _trapezoidal_aumc(times_h, concs_mg_L)

    # AUMC extrapolation: C_last/lambda_z^2 + t_last*C_last/lambda_z
    if lambda_z > 0:
        aumc_tlast_inf = c_last / (lambda_z**2) + t_last * c_last / lambda_z
    else:
        aumc_tlast_inf = 0.0
    aumc0_inf = aumc0_t + aumc_tlast_inf

    # MRT
    mrt = aumc0_inf / auc0_inf if auc0_inf > 0 else 0.0

    notes = f"IV bolus NCA; C0={c0:.3f} mg/L; lambda_z={lambda_z:.4f} 1/h; R2_term={r_sq:.4f}"

    return NCAResult(
        route="iv",
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc0_t=auc0_t,
        auc0_inf=auc0_inf,
        aumc0_inf=aumc0_inf,
        lambda_z=lambda_z,
        t_half_h=t_half,
        cl_L_per_h=cl,
        vd_L=vd,
        mrt_h=mrt,
        r_squared_terminal=r_sq,
        notes=notes,
    )


def nca_oral(
    times_h: list[float],
    concs_mg_L: list[float],
    dose_mg: float,
    f_oral: float = 1.0,
) -> NCAResult:
    """Non-compartmental analysis for oral dose data.

    Parameters
    ----------
    times_h:
        Time points in hours.
    concs_mg_L:
        Plasma concentrations in mg/L.
    dose_mg:
        Administered dose in mg.
    f_oral:
        Oral bioavailability fraction (default 1.0).

    Returns
    -------
    NCAResult
    """
    _validate_inputs(times_h, concs_mg_L)

    # Cmax and Tmax
    cmax_idx = max(range(len(concs_mg_L)), key=lambda i: concs_mg_L[i])
    cmax = concs_mg_L[cmax_idx]
    tmax = times_h[cmax_idx]

    # AUC0_t (trapezoidal)
    auc0_t = _trapezoidal_auc(times_h, concs_mg_L)

    # Terminal slope
    lambda_z, intercept, r_sq = fit_terminal_slope(times_h, concs_mg_L)

    # AUC extrapolation
    c_last = concs_mg_L[-1]
    t_last = times_h[-1]
    auc_tlast_inf = c_last / lambda_z if lambda_z > 0 else 0.0
    auc0_inf = auc0_t + auc_tlast_inf

    # Half-life
    t_half = math.log(2.0) / lambda_z if lambda_z > 0 else float("inf")

    # CL/F and Vd/F
    cl_f = dose_mg * f_oral / auc0_inf if auc0_inf > 0 else 0.0
    vd_f = cl_f / lambda_z if lambda_z > 0 else 0.0

    # AUMC
    aumc0_t = _trapezoidal_aumc(times_h, concs_mg_L)
    if lambda_z > 0:
        aumc_tlast_inf = c_last / (lambda_z**2) + t_last * c_last / lambda_z
    else:
        aumc_tlast_inf = 0.0
    aumc0_inf = aumc0_t + aumc_tlast_inf

    # MRT
    mrt = aumc0_inf / auc0_inf if auc0_inf > 0 else 0.0

    notes = (
        f"Oral NCA; Cmax={cmax:.3f} mg/L at Tmax={tmax:.1f}h; "
        f"lambda_z={lambda_z:.4f} 1/h; CL/F={cl_f:.3f} L/h; R2_term={r_sq:.4f}"
    )

    return NCAResult(
        route="oral",
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc0_t=auc0_t,
        auc0_inf=auc0_inf,
        aumc0_inf=aumc0_inf,
        lambda_z=lambda_z,
        t_half_h=t_half,
        cl_L_per_h=cl_f,
        vd_L=vd_f,
        mrt_h=mrt,
        r_squared_terminal=r_sq,
        notes=notes,
    )
