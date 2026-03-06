"""Non-Compartmental Analysis (NCA) of pharmacokinetic data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "NCAResult",
    "calculate_nca",
    "compare_nca",
]


@dataclass
class NCAResult:
    drug_name: str
    route: str
    dose_mg: float
    times_h: list
    conc_mg_L: list
    cmax_mg_L: float
    tmax_h: float
    auc_0_t: float
    auc_0_inf: float
    auc_extrap_pct: float
    aumc_0_inf: float
    mrt_h: float
    t_half_h: float
    ke_per_h: float
    cl_L_per_h: float
    vd_ss_L: float
    n_points: int
    lambda_z_r2: float


def _terminal_slope(
    times: np.ndarray,
    conc: np.ndarray,
    n_terminal_pct: float = 0.3,
) -> tuple[float, float, float]:
    """Fit log-linear to terminal phase; return (ke, intercept, r2).

    Parameters
    ----------
    times:
        Time points (sorted ascending).
    conc:
        Corresponding concentrations (all > 0 required for log transform).
    n_terminal_pct:
        Fraction of points to use from the tail (minimum 3).

    Returns
    -------
    ke : float
        Terminal elimination rate constant (positive value, h⁻¹).
    intercept : float
        y-intercept of log-linear fit (ln-scale).
    r2 : float
        Coefficient of determination for the fit.
    """
    n = len(times)
    n_pts = max(3, int(np.ceil(n * n_terminal_pct)))
    n_pts = min(n_pts, n)

    t_term = times[-n_pts:]
    c_term = conc[-n_pts:]

    # Drop non-positive concentrations
    mask = c_term > 0
    if mask.sum() < 3:
        raise ValueError("Fewer than 3 positive concentration points available for terminal slope.")
    t_term = t_term[mask]
    ln_c = np.log(c_term[mask])

    # Linear regression: ln(C) = intercept - ke * t
    t_mean = t_term.mean()
    ln_c_mean = ln_c.mean()
    ss_tt = np.sum((t_term - t_mean) ** 2)
    ss_tc = np.sum((t_term - t_mean) * (ln_c - ln_c_mean))

    slope = ss_tc / ss_tt  # negative for elimination
    intercept = ln_c_mean - slope * t_mean
    ke = -slope  # positive

    # R²
    ln_c_pred = intercept + slope * t_term
    ss_res = np.sum((ln_c - ln_c_pred) ** 2)
    ss_tot = np.sum((ln_c - ln_c_mean) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return ke, intercept, r2


def calculate_nca(
    drug_name: str,
    times_h: list,
    conc_mg_L: list,
    dose_mg: float,
    route: str = "oral",
) -> NCAResult:
    """Compute NCA metrics from observed concentration-time data.

    Parameters
    ----------
    drug_name : str
        Identifier for the drug.
    times_h : list
        Observation times (hours), must be monotonically increasing.
    conc_mg_L : list
        Plasma concentrations (mg/L) corresponding to *times_h*.
    dose_mg : float
        Administered dose (mg).
    route : str
        Dosing route — ``'iv'`` or ``'oral'`` (default).

    Returns
    -------
    NCAResult
        Populated dataclass with all NCA parameters.

    Raises
    ------
    ValueError
        If fewer than 3 data points are supplied or *dose_mg* ≤ 0.
    """
    times_h = list(times_h)
    conc_mg_L = list(conc_mg_L)

    if len(times_h) < 3:
        raise ValueError("At least 3 data points are required for NCA.")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive.")

    t = np.asarray(times_h, dtype=float)
    c = np.asarray(conc_mg_L, dtype=float)

    n_points = len(t)

    # Cmax and Tmax
    idx_max = int(np.argmax(c))
    cmax = float(c[idx_max])
    tmax = float(t[idx_max])

    # AUC0-t via trapezoidal rule
    auc_0_t = float(np.trapz(c, t))

    # Terminal slope
    ke, _intercept, r2 = _terminal_slope(t, c)

    # AUC0-inf = AUC0-t + Clast / ke
    c_last = float(c[-1])
    auc_extrap = c_last / ke
    auc_0_inf = auc_0_t + auc_extrap
    auc_extrap_pct = 100.0 * auc_extrap / auc_0_inf

    # AUMC via trapezoidal rule on t*C(t)
    tc = t * c
    aumc_0_t = float(np.trapz(tc, t))
    # Extrapolation: (Clast * tlast) / ke + Clast / ke^2
    t_last = float(t[-1])
    aumc_extrap = (c_last * t_last) / ke + c_last / (ke**2)
    aumc_0_inf = aumc_0_t + aumc_extrap

    # MRT and half-life
    mrt_h = aumc_0_inf / auc_0_inf
    t_half_h = np.log(2) / ke

    # CL and Vdss (IV only)
    if route.lower() == "iv":
        cl = dose_mg / auc_0_inf
        vd_ss = cl * mrt_h
    else:
        cl = float("nan")
        vd_ss = float("nan")

    return NCAResult(
        drug_name=drug_name,
        route=route,
        dose_mg=dose_mg,
        times_h=times_h,
        conc_mg_L=conc_mg_L,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_0_t=auc_0_t,
        auc_0_inf=auc_0_inf,
        auc_extrap_pct=auc_extrap_pct,
        aumc_0_inf=aumc_0_inf,
        mrt_h=mrt_h,
        t_half_h=t_half_h,
        ke_per_h=ke,
        cl_L_per_h=cl,
        vd_ss_L=vd_ss,
        n_points=n_points,
        lambda_z_r2=r2,
    )


def compare_nca(results: list[NCAResult]) -> dict:
    """Build a comparison table from multiple NCAResult objects.

    Parameters
    ----------
    results : list[NCAResult]
        Collection of NCA results to compare.

    Returns
    -------
    dict
        Keys are drug names; values are dicts with keys:
        ``drug_name``, ``cmax``, ``tmax``, ``auc_0_inf``, ``t_half``, ``cl``.
        CL is ``None`` for oral routes (NaN stored as None for readability).
    """
    table: dict[str, dict] = {}
    for r in results:
        cl_val = None if (r.route.lower() != "iv" or np.isnan(r.cl_L_per_h)) else r.cl_L_per_h
        table[r.drug_name] = {
            "drug_name": r.drug_name,
            "cmax": r.cmax_mg_L,
            "tmax": r.tmax_h,
            "auc_0_inf": r.auc_0_inf,
            "t_half": r.t_half_h,
            "cl": cl_val,
        }
    return table
