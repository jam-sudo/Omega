"""Tumor Microenvironment Drug Distribution model.

Models drug distribution within the tumor microenvironment across four
compartments: systemic plasma, tumor vascular (TV), tumor interstitium (TI),
and tumor intracellular (TC).

Uses forward-Euler ODE integration and manual trapezoidal AUC.
No numpy or scipy — pure Python stdlib only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TumorMicroenvPKResult:
    """Result of tumor microenvironment pharmacokinetic simulation."""

    drug_name: str
    dose_mg: float
    times_h: list[float]
    c_plasma_mg_L: list[float]
    c_tv_mg_L: list[float]
    c_ti_mg_L: list[float]
    c_tc_mg_L: list[float]
    cmax_plasma: float
    auc_plasma: float
    cmax_tv: float
    auc_tv: float
    cmax_ti: float
    auc_ti: float
    cmax_tc: float
    auc_tc: float
    tumor_plasma_ratio: float
    penetration_depth_score: float
    notes: str


def _trapz(y: list[float], x: list[float]) -> float:
    """Manual trapezoidal integration."""
    total = 0.0
    for i in range(1, len(x)):
        total += 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1])
    return total


def simulate_tumor_microenv_pk(
    drug_name: str,
    dose_mg: float,
    ps_per_h: float = 0.5,
    k_ifp_factor: float = 0.3,
    k_intracell_per_h: float = 0.2,
    k_efflux_per_h: float = 0.1,
    v_tumor_L: float = 0.01,
    cl_sys_L_per_h: float = 10.0,
    vd_sys_L: float = 50.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> TumorMicroenvPKResult:
    """Simulate tumor microenvironment PK via forward-Euler ODE.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        IV dose administered into systemic plasma (mg). Must be > 0.
    ps_per_h:
        Vascular permeability × surface area product (1/h).
    k_ifp_factor:
        Interstitial fluid pressure factor (0-1); reduces convection to TI.
    k_intracell_per_h:
        Rate of cellular uptake from TI to TC (1/h).
    k_efflux_per_h:
        Intracellular efflux rate from TC (1/h).
    v_tumor_L:
        Total tumor volume in litres.
    cl_sys_L_per_h:
        Systemic clearance (L/h).
    vd_sys_L:
        Systemic volume of distribution (L).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward-Euler time step (h).

    Returns
    -------
    TumorMicroenvPKResult
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if v_tumor_L <= 0:
        raise ValueError(f"v_tumor_L must be > 0, got {v_tumor_L}")
    if cl_sys_L_per_h <= 0:
        raise ValueError(f"cl_sys_L_per_h must be > 0, got {cl_sys_L_per_h}")

    v_tv_L = v_tumor_L * 0.05  # 5% vascular fraction
    v_ti_L = v_tumor_L * 0.50  # 50% interstitial fraction
    v_tc_L = v_tumor_L * 0.45  # 45% intracellular fraction

    ke_sys = cl_sys_L_per_h / vd_sys_L

    # Initial conditions: all drug in systemic plasma
    a_plasma = dose_mg
    a_tv = 0.0
    a_ti = 0.0
    a_tc = 0.0

    n_steps = int(math.ceil(t_end_h / dt_h))

    times: list[float] = []
    c_plasma_list: list[float] = []
    c_tv_list: list[float] = []
    c_ti_list: list[float] = []
    c_tc_list: list[float] = []

    t = 0.0

    for _step in range(n_steps + 1):
        c_plasma = a_plasma / vd_sys_L
        c_tv = a_tv / v_tv_L
        c_ti = a_ti / v_ti_L
        c_tc = a_tc / v_tc_L

        times.append(t)
        c_plasma_list.append(c_plasma)
        c_tv_list.append(c_tv)
        c_ti_list.append(c_ti)
        c_tc_list.append(c_tc)

        if _step == n_steps:
            break

        # ODEs (amounts):
        # Plasma: eliminated + transferred to TV
        d_plasma = -ke_sys * a_plasma - ps_per_h * a_plasma / vd_sys_L * v_tv_L + ps_per_h * a_tv

        # Tumor vascular: influx from plasma, drain via IFP pressure factor
        d_tv = ps_per_h * a_plasma / vd_sys_L * v_tv_L - (ps_per_h + k_ifp_factor) * a_tv

        # Tumor interstitium: influx from TV via IFP, drain via cellular uptake
        d_ti = k_ifp_factor * a_tv - k_intracell_per_h * a_ti

        # Tumor intracellular: uptake from TI, efflux
        d_tc = k_intracell_per_h * a_ti - k_efflux_per_h * a_tc

        a_plasma = max(0.0, a_plasma + d_plasma * dt_h)
        a_tv = max(0.0, a_tv + d_tv * dt_h)
        a_ti = max(0.0, a_ti + d_ti * dt_h)
        a_tc = max(0.0, a_tc + d_tc * dt_h)

        t = min(t + dt_h, t_end_h)

    # Derived metrics
    cmax_plasma = max(c_plasma_list)
    cmax_tv = max(c_tv_list)
    cmax_ti = max(c_ti_list)
    cmax_tc = max(c_tc_list)

    auc_plasma = _trapz(c_plasma_list, times)
    auc_tv = _trapz(c_tv_list, times)
    auc_ti = _trapz(c_ti_list, times)
    auc_tc = _trapz(c_tc_list, times)

    if auc_plasma > 0:
        tumor_plasma_ratio = auc_tc / auc_plasma
    else:
        tumor_plasma_ratio = 0.0

    penetration_depth_score = min(100.0, tumor_plasma_ratio * 100.0)

    if tumor_plasma_ratio > 1.0:
        notes = "Excellent tumor accumulation (EPR + active)"
    elif tumor_plasma_ratio > 0.3:
        notes = "Good tumor penetration"
    elif tumor_plasma_ratio > 0.1:
        notes = "Moderate tumor penetration"
    else:
        notes = "Poor tumor penetration"

    return TumorMicroenvPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_tv_mg_L=c_tv_list,
        c_ti_mg_L=c_ti_list,
        c_tc_mg_L=c_tc_list,
        cmax_plasma=cmax_plasma,
        auc_plasma=auc_plasma,
        cmax_tv=cmax_tv,
        auc_tv=auc_tv,
        cmax_ti=cmax_ti,
        auc_ti=auc_ti,
        cmax_tc=cmax_tc,
        auc_tc=auc_tc,
        tumor_plasma_ratio=tumor_plasma_ratio,
        penetration_depth_score=penetration_depth_score,
        notes=notes,
    )


def compare_tumor_targeting(
    drug_name: str,
    dose_mg: float,
    scenarios: list[dict],
) -> list[TumorMicroenvPKResult]:
    """Compare multiple tumor-targeting scenarios sorted by auc_tc descending.

    Parameters
    ----------
    drug_name:
        Drug name.
    dose_mg:
        Dose in mg.
    scenarios:
        List of dicts with optional keys: "ps_per_h", "k_ifp_factor", "name".
        The "name" key is used for labelling only (ignored by the simulation).

    Returns
    -------
    list[TumorMicroenvPKResult]
        Results sorted by ``auc_tc`` in descending order.
    """
    results: list[TumorMicroenvPKResult] = []
    for scenario in scenarios:
        kwargs = {k: v for k, v in scenario.items() if k != "name"}
        result = simulate_tumor_microenv_pk(drug_name, dose_mg, **kwargs)
        results.append(result)
    results.sort(key=lambda r: r.auc_tc, reverse=True)
    return results
