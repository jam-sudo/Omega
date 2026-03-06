"""Subcutaneous/IM depot absorption PK model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from omega_pbpk._compat import np_trapz


class DepotRoute(str, Enum):
    SUBCUTANEOUS = "sc"
    INTRAMUSCULAR = "im"


@dataclass(frozen=True)
class DepotPKResult:
    drug_name: str
    route: str
    dose_mg: float
    times_h: list[float]
    cp_central: list[float]
    a_depot: list[float]
    ka_eff_per_h: float
    tlag_h: float
    cmax: float
    tmax_h: float
    auc_0_t: float
    auc_0_inf: float
    thalf_abs_h: float
    thalf_elim_h: float
    f_abs: float
    flip_flop: bool
    bioavailability: float


def simulate_depot_pk(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    ka_per_h: float,
    route: DepotRoute = DepotRoute.SUBCUTANEOUS,
    bioavailability: float = 1.0,
    tlag_h: float = 0.0,
    t_end_h: float = 72.0,
    dt_h: float = 0.1,
) -> DepotPKResult:
    """Simulate first-order absorption from a subcutaneous or IM depot site."""
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if ka_per_h <= 0:
        raise ValueError("ka_per_h must be > 0")
    if not (0 < bioavailability <= 1):
        raise ValueError("bioavailability must be in (0, 1]")

    ke = cl_L_per_h / vd_L
    ka_eff = ka_per_h * 1.3 if route == DepotRoute.INTRAMUSCULAR else ka_per_h
    f = bioavailability

    times = np.arange(0.0, t_end_h + dt_h / 2, dt_h)
    n = len(times)

    a_depot_arr = np.zeros(n)
    a_central = np.zeros(n)
    cp = np.zeros(n)

    a_depot_arr[0] = dose_mg

    for i in range(1, n):
        t = times[i]
        if t < tlag_h:
            a_depot_arr[i] = a_depot_arr[i - 1]
            a_central[i] = a_central[i - 1] - ke * a_central[i - 1] * dt_h
        else:
            da_depot = -ka_eff * a_depot_arr[i - 1] * dt_h
            da_central = (ka_eff * a_depot_arr[i - 1] * f - ke * a_central[i - 1]) * dt_h
            a_depot_arr[i] = a_depot_arr[i - 1] + da_depot
            a_central[i] = a_central[i - 1] + da_central
        cp[i] = a_central[i] / vd_L

    cmax = float(np.max(cp))
    tmax_h = float(times[np.argmax(cp)])
    auc_0_t = float(np_trapz(cp, times))
    auc_0_inf = auc_0_t + float(cp[-1]) / ke if ke > 0 else auc_0_t
    thalf_abs = math.log(2) / ka_eff
    thalf_elim = math.log(2) / ke
    f_abs = (dose_mg - float(a_depot_arr[-1])) / dose_mg
    flip_flop = ka_eff < ke

    return DepotPKResult(
        drug_name=drug_name,
        route=route.value,
        dose_mg=dose_mg,
        times_h=times.tolist(),
        cp_central=cp.tolist(),
        a_depot=a_depot_arr.tolist(),
        ka_eff_per_h=ka_eff,
        tlag_h=tlag_h,
        cmax=cmax,
        tmax_h=tmax_h,
        auc_0_t=auc_0_t,
        auc_0_inf=auc_0_inf,
        thalf_abs_h=thalf_abs,
        thalf_elim_h=thalf_elim,
        f_abs=f_abs,
        flip_flop=flip_flop,
        bioavailability=bioavailability,
    )


def compare_routes(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    ka_per_h: float,
    **kwargs,
) -> dict[str, DepotPKResult]:
    """Compare SC vs IM for same drug."""
    return {
        "sc": simulate_depot_pk(drug_name, dose_mg, cl_L_per_h, vd_L, ka_per_h, DepotRoute.SUBCUTANEOUS, **kwargs),
        "im": simulate_depot_pk(drug_name, dose_mg, cl_L_per_h, vd_L, ka_per_h, DepotRoute.INTRAMUSCULAR, **kwargs),
    }


__all__ = ["DepotRoute", "DepotPKResult", "simulate_depot_pk", "compare_routes"]
