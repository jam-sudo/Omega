"""Cerebrospinal Fluid (CSF) PK model.

Models drug distribution for intrathecal, intracerebroventricular (ICV),
or systemic (passive BBB crossing) drug administration.

Uses forward-Euler ODE integration and manual trapezoidal AUC.
No numpy or scipy — pure Python stdlib only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CSFPKResult:
    """Result of CSF pharmacokinetic simulation."""

    drug_name: str
    dose_mg: float
    route: str
    times_h: list[float]
    c_csf_mg_L: list[float]
    c_brain_mg_L: list[float]
    c_plasma_mg_L: list[float]
    cmax_csf_mg_L: float
    tmax_csf_h: float
    auc_csf: float
    cmax_brain_mg_L: float
    auc_brain: float
    brain_to_csf_ratio: float
    csf_t_half_h: float
    notes: str


def _trapz(y: list[float], x: list[float]) -> float:
    """Manual trapezoidal integration."""
    total = 0.0
    for i in range(1, len(x)):
        total += 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1])
    return total


def simulate_csf_pk(
    drug_name: str,
    dose_mg: float,
    route: str = "intrathecal",
    v_csf_L: float = 0.14,
    k_csf_drain_per_h: float = 0.15,
    k_cp_per_h: float = 0.1,
    k_pc_per_h: float = 0.05,
    bbb_perm_per_h: float = 0.01,
    cl_sys_L_per_h: float = 10.0,
    vd_sys_L: float = 50.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> CSFPKResult:
    """Simulate CSF pharmacokinetics via forward-Euler ODE.

    Parameters
    ----------
    drug_name:
        Name of the drug being simulated.
    dose_mg:
        Administered dose in mg. Must be > 0.
    route:
        Dosing route: "intrathecal", "ICV", or "systemic".
    v_csf_L:
        CSF volume in litres (default 0.14 L = 140 mL).
    k_csf_drain_per_h:
        CSF bulk-flow turnover rate constant (1/h).
    k_cp_per_h:
        CSF-to-brain parenchyma transfer rate (1/h).
    k_pc_per_h:
        Brain-to-CSF back-transfer rate (1/h).
    bbb_perm_per_h:
        BBB permeability rate for systemic→CSF entry (1/h).
    cl_sys_L_per_h:
        Systemic clearance (L/h).
    vd_sys_L:
        Systemic volume of distribution (L).
    t_end_h:
        Simulation duration (h).
    dt_h:
        Forward-Euler time step (h).

    Returns
    -------
    CSFPKResult
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if v_csf_L <= 0:
        raise ValueError(f"v_csf_L must be > 0, got {v_csf_L}")
    valid_routes = {"intrathecal", "ICV", "systemic"}
    if route not in valid_routes:
        raise ValueError(f"route must be one of {valid_routes}, got {route!r}")

    v_brain_L = 1.3  # ~1300 mL brain volume
    ke_sys = cl_sys_L_per_h / vd_sys_L

    # Initial conditions depend on route
    if route in {"intrathecal", "ICV"}:
        a_csf = dose_mg
        a_brain = 0.0
        a_plasma = 0.0
    else:  # systemic
        a_csf = 0.0
        a_brain = 0.0
        a_plasma = dose_mg

    times: list[float] = []
    c_csf_list: list[float] = []
    c_brain_list: list[float] = []
    c_plasma_list: list[float] = []

    t = 0.0
    n_steps = int(math.ceil(t_end_h / dt_h))

    for _step in range(n_steps + 1):
        c_csf = a_csf / v_csf_L
        c_brain = a_brain / v_brain_L
        c_plasma = a_plasma / vd_sys_L

        times.append(t)
        c_csf_list.append(c_csf)
        c_brain_list.append(c_brain)
        c_plasma_list.append(c_plasma)

        if _step == n_steps:
            break

        # ODEs (amounts)
        d_csf = (
            -k_csf_drain_per_h * a_csf
            - k_cp_per_h * a_csf
            + k_pc_per_h * a_brain
            + bbb_perm_per_h * a_plasma
        )
        d_brain = k_cp_per_h * a_csf - k_pc_per_h * a_brain
        d_plasma = k_csf_drain_per_h * a_csf - ke_sys * a_plasma - bbb_perm_per_h * a_plasma

        a_csf = max(0.0, a_csf + d_csf * dt_h)
        a_brain = max(0.0, a_brain + d_brain * dt_h)
        a_plasma = max(0.0, a_plasma + d_plasma * dt_h)

        t = min(t + dt_h, t_end_h)

    # Derived metrics
    cmax_csf = max(c_csf_list)
    tmax_csf_idx = c_csf_list.index(cmax_csf)
    tmax_csf = times[tmax_csf_idx]

    cmax_brain = max(c_brain_list)
    auc_csf = _trapz(c_csf_list, times)
    auc_brain = _trapz(c_brain_list, times)

    csf_t_half = math.log(2.0) / k_csf_drain_per_h

    if auc_csf > 0:
        brain_to_csf_ratio = auc_brain / auc_csf
    else:
        brain_to_csf_ratio = 0.0

    if brain_to_csf_ratio > 1.0:
        notes = "High brain penetration from CSF"
    elif brain_to_csf_ratio > 0.3:
        notes = "Moderate brain penetration"
    else:
        notes = "Limited brain penetration from CSF"

    return CSFPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        times_h=times,
        c_csf_mg_L=c_csf_list,
        c_brain_mg_L=c_brain_list,
        c_plasma_mg_L=c_plasma_list,
        cmax_csf_mg_L=cmax_csf,
        tmax_csf_h=tmax_csf,
        auc_csf=auc_csf,
        cmax_brain_mg_L=cmax_brain,
        auc_brain=auc_brain,
        brain_to_csf_ratio=brain_to_csf_ratio,
        csf_t_half_h=csf_t_half,
        notes=notes,
    )


def compare_csf_routes(
    drug_name: str,
    dose_mg: float,
    **kwargs,
) -> list[CSFPKResult]:
    """Simulate all three CSF routes and return results sorted by auc_csf descending.

    Parameters
    ----------
    drug_name:
        Drug name.
    dose_mg:
        Dose in mg (same for all routes).
    **kwargs:
        Additional keyword arguments forwarded to :func:`simulate_csf_pk`.

    Returns
    -------
    list[CSFPKResult]
        Three results sorted by ``auc_csf`` in descending order.
    """
    routes = ["intrathecal", "ICV", "systemic"]
    results = [simulate_csf_pk(drug_name, dose_mg, route=r, **kwargs) for r in routes]
    results.sort(key=lambda r: r.auc_csf, reverse=True)
    return results
