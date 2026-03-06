"""Michaelis-Menten (nonlinear) PK model — saturable elimination."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class MichaelisMentenResult:
    drug_name: str
    times_h: list[float]
    conc_mg_L: list[float]
    auc_0_t: float
    auc_0_inf: float
    cmax: float
    tmax_h: float
    dose_mg: float
    vmax_mg_h: float
    km_mg_L: float
    vd_L: float
    # Apparent half-life at Cmax (concentration-dependent)
    thalf_at_cmax_h: float
    # Linearity check: CL at Cmax vs CL at Cmin
    cl_at_cmax_L_per_h: float
    cl_at_cmin_L_per_h: float
    is_nonlinear: bool  # True if CL varies >20% across concentration range


def _mm_clearance(conc: float, vmax: float, km: float) -> float:
    """Michaelis-Menten clearance at concentration C: CL(C) = Vmax/(Km+C)."""
    return vmax / (km + conc) if conc > 0 else vmax / km


def simulate_michaelis_menten(
    drug_name: str,
    dose_mg: float,
    vd_L: float,
    vmax_mg_h: float,
    km_mg_L: float,
    route: str = "iv",
    ka_per_h: float = 1.0,
    f: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.01,
) -> MichaelisMentenResult:
    """Simulate 1-compartment PK with Michaelis-Menten elimination.

    dC/dt = -Vmax*C/(Km+C)/Vd  [IV]
    dA_gut/dt = -ka*A_gut; dC/dt = ka*A_gut*f/Vd - Vmax*C/(Km+C)/Vd  [oral]

    Parameters
    ----------
    vmax_mg_h : float
        Maximum elimination rate (mg/h).
    km_mg_L : float
        Michaelis constant (mg/L) — concentration at half-maximal rate.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if vmax_mg_h <= 0:
        raise ValueError("vmax_mg_h must be > 0")
    if km_mg_L <= 0:
        raise ValueError("km_mg_L must be > 0")

    n = max(int(t_end_h / dt_h), 1)
    t = np.linspace(0.0, t_end_h, n + 1)
    conc = np.zeros(n + 1)

    is_oral = route.lower() == "oral"

    if not is_oral:
        conc[0] = dose_mg / vd_L
        for i in range(n):
            c = conc[i]
            rate = vmax_mg_h * c / (km_mg_L + c)  # mg/h eliminated
            dC = -rate / vd_L
            conc[i + 1] = max(c + dC * dt_h, 0.0)
    else:
        a_gut = dose_mg * f
        for i in range(n):
            abs_rate = ka_per_h * a_gut
            a_gut = max(a_gut - abs_rate * dt_h, 0.0)
            c = conc[i]
            mm_rate = vmax_mg_h * c / (km_mg_L + c)
            dC = abs_rate / vd_L - mm_rate / vd_L
            conc[i + 1] = max(c + dC * dt_h, 0.0)

    auc_0_t = float(np_trapz(conc, t))
    cmax = float(np.max(conc))
    tmax_h = float(t[int(np.argmax(conc))])

    # AUC0-inf: extrapolate using apparent terminal rate
    last_c = float(conc[-1])
    last_cl = _mm_clearance(last_c, vmax_mg_h, km_mg_L) / vd_L  # apparent ke
    auc_0_inf = auc_0_t + last_c / last_cl if last_cl > 0 else auc_0_t

    # Half-life at Cmax: t½ = ln(2)*Vd*(Km+Cmax)/Vmax
    thalf_cmax = math.log(2) * vd_L * (km_mg_L + cmax) / vmax_mg_h

    # CL at Cmax and Cmin
    cl_cmax = _mm_clearance(cmax, vmax_mg_h, km_mg_L)
    cmin_nonzero = float(np.min(conc[conc > 0])) if np.any(conc > 0) else 1e-6
    cl_cmin = _mm_clearance(cmin_nonzero, vmax_mg_h, km_mg_L)

    is_nonlinear = abs(cl_cmax - cl_cmin) / max(cl_cmin, 1e-9) > 0.20

    return MichaelisMentenResult(
        drug_name=drug_name,
        times_h=t.tolist(),
        conc_mg_L=conc.tolist(),
        auc_0_t=auc_0_t,
        auc_0_inf=auc_0_inf,
        cmax=cmax,
        tmax_h=tmax_h,
        dose_mg=dose_mg,
        vmax_mg_h=vmax_mg_h,
        km_mg_L=km_mg_L,
        vd_L=vd_L,
        thalf_at_cmax_h=thalf_cmax,
        cl_at_cmax_L_per_h=cl_cmax,
        cl_at_cmin_L_per_h=cl_cmin,
        is_nonlinear=is_nonlinear,
    )


def dose_proportionality_index(
    drug_name: str,
    doses: list[float],
    vd_L: float,
    vmax_mg_h: float,
    km_mg_L: float,
) -> dict:
    """Compute dose-AUC proportionality across multiple doses.

    Returns dict with doses, aucs, and power law exponent (beta).
    Linear PK: beta ≈ 1.0; nonlinear: beta != 1.
    """
    aucs = []
    for d in doses:
        r = simulate_michaelis_menten(drug_name, d, vd_L, vmax_mg_h, km_mg_L)
        aucs.append(r.auc_0_t)

    log_doses = np.log(doses)
    log_aucs = np.log(np.maximum(aucs, 1e-12))
    coeffs = np.polyfit(log_doses, log_aucs, 1)
    beta = float(coeffs[0])

    return {
        "drug_name": drug_name,
        "doses": doses,
        "aucs": aucs,
        "beta": beta,
        "is_proportional": abs(beta - 1.0) < 0.15,
    }


__all__ = [
    "MichaelisMentenResult",
    "simulate_michaelis_menten",
    "dose_proportionality_index",
    "_mm_clearance",
]
