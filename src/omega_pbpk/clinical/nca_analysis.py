"""Full Non-Compartmental Analysis (NCA) — Phase 172.

Extracts standard NCA parameters from concentration-time data
including Cmax, Tmax, AUC, lambda_z, t_half, CL, Vd, MRT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class NCAResult:
    """Full NCA parameter set."""

    drug_name: str
    route: str
    cmax: float
    tmax: float
    auc_0_t: float
    auc_0_inf: float
    lambda_z: float
    t_half_h: float
    cl_obs_L_per_h: float
    vd_obs_L: float
    mrt_h: float
    f_extrap_pct: float
    notes: list[str]


def perform_nca(
    times_h: list[float],
    concentrations_mg_L: list[float],
    dose_mg: float,
    drug_name: str = "Unknown",
    route: str = "oral",
    lambda_z_points: int = 3,
) -> NCAResult:
    """Perform non-compartmental analysis on concentration-time data.

    Parameters
    ----------
    times_h : list[float]
        Time points (h). Must be strictly increasing.
    concentrations_mg_L : list[float]
        Concentrations (mg/L) at each time point.
    dose_mg : float
        Administered dose (mg). Must be > 0.
    drug_name : str
        Name of the drug.
    route : str
        'iv' or 'oral'.
    lambda_z_points : int
        Number of terminal points for lambda_z estimation.

    Returns
    -------
    NCAResult
    """
    if len(times_h) != len(concentrations_mg_L):
        raise ValueError("times_h and concentrations_mg_L must have equal length")
    if len(times_h) < 3:
        raise ValueError("At least 3 data points are required")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if route not in ("iv", "oral"):
        raise ValueError("route must be 'iv' or 'oral'")

    t = list(times_h)
    for i in range(1, len(t)):
        if t[i] <= t[i - 1]:
            raise ValueError("times_h must be strictly increasing")

    c = list(concentrations_mg_L)

    # Cmax, Tmax
    cmax = max(c)
    tmax = t[c.index(cmax)]

    # AUC 0-t (linear trapezoidal)
    auc_0_t = float(np_trapz(c, t))

    # Lambda_z: log-linear regression on last lambda_z_points positive concentrations
    n_pts = min(lambda_z_points, len(t))
    # Use last n_pts points with positive concentration
    valid_indices = [i for i in range(len(c)) if c[i] > 0]
    if len(valid_indices) < 2:
        raise ValueError("Not enough positive concentrations for lambda_z estimation")

    term_indices = valid_indices[-n_pts:]
    t_term = [t[i] for i in term_indices]
    c_term = [c[i] for i in term_indices]

    ln_c = [math.log(ci) for ci in c_term]

    n = len(t_term)
    sum_t = sum(t_term)
    sum_lnc = sum(ln_c)
    sum_t2 = sum(ti * ti for ti in t_term)
    sum_t_lnc = sum(t_term[i] * ln_c[i] for i in range(n))

    denom = n * sum_t2 - sum_t * sum_t
    if abs(denom) < 1e-15:
        slope = -0.01
    else:
        slope = (n * sum_t_lnc - sum_t * sum_lnc) / denom

    lambda_z = -slope if slope < 0 else 0.01

    t_half = math.log(2) / lambda_z

    # AUC 0-inf
    c_last = c_term[-1]
    auc_extrap = c_last / lambda_z
    auc_0_inf = auc_0_t + auc_extrap

    # CL and Vd
    cl_obs = dose_mg / auc_0_inf
    vd_obs = cl_obs / lambda_z

    # MRT = AUMC / AUC
    tc = [t[i] * c[i] for i in range(len(t))]
    aumc = float(np_trapz(tc, t))
    aumc_extrap = c_last * t[-1] / lambda_z + c_last / (lambda_z**2)
    aumc_inf = aumc + aumc_extrap
    mrt = aumc_inf / auc_0_inf if auc_0_inf > 0 else 0.0

    # f_extrap_pct
    f_extrap_pct = ((auc_0_inf - auc_0_t) / auc_0_inf * 100.0) if auc_0_inf > 0 else 0.0

    notes: list[str] = []
    if f_extrap_pct > 20:
        notes.append(f"High extrapolation ({f_extrap_pct:.1f}%); consider longer sampling")
    if route == "oral":
        notes.append("CL and Vd are apparent values (CL/F, Vd/F)")

    return NCAResult(
        drug_name=drug_name,
        route=route,
        cmax=cmax,
        tmax=tmax,
        auc_0_t=auc_0_t,
        auc_0_inf=auc_0_inf,
        lambda_z=lambda_z,
        t_half_h=t_half,
        cl_obs_L_per_h=cl_obs,
        vd_obs_L=vd_obs,
        mrt_h=mrt,
        f_extrap_pct=f_extrap_pct,
        notes=notes,
    )


def compare_nca(profiles: list[dict]) -> list[NCAResult]:
    """Run NCA on multiple profiles for comparison.

    Parameters
    ----------
    profiles : list[dict]
        Each dict must contain 'times_h', 'concentrations_mg_L', 'dose_mg'.
        Optional: 'drug_name', 'route', 'lambda_z_points'.

    Returns
    -------
    list[NCAResult]
    """
    if not profiles:
        raise ValueError("profiles must not be empty")
    results: list[NCAResult] = []
    for p in profiles:
        r = perform_nca(
            times_h=p["times_h"],
            concentrations_mg_L=p["concentrations_mg_L"],
            dose_mg=p["dose_mg"],
            drug_name=p.get("drug_name", "Unknown"),
            route=p.get("route", "oral"),
            lambda_z_points=p.get("lambda_z_points", 3),
        )
        results.append(r)
    return results
