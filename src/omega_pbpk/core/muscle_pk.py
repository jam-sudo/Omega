"""Phase 293 — Muscle Tissue PK Model.

2-compartment perfusion-limited model: plasma + skeletal muscle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class MusclePKResult:
    """Result of a muscle PK simulation."""

    drug_name: str
    dose_mg: float
    route: str
    kp_muscle: float
    times_h: list
    c_plasma_mg_L: list
    c_muscle_mg_L: list
    auc_plasma_mg_h_per_L: float
    cmax_plasma_mg_L: float
    auc_muscle_mg_h_per_L: float
    cmax_muscle_mg_L: float
    tmax_muscle_h: float
    muscle_to_plasma_ratio: float
    notes: str


def _trapezoidal_auc(x: list, y: list) -> float:
    """Manual trapezoidal AUC."""
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


def _validate_inputs(
    dose_mg: float,
    cl_sys_L_per_h: float,
    vd_plasma_L: float,
    kp_muscle: float,
    muscle_volume_L: float,
    muscle_blood_flow_L_per_h: float,
    t_end_h: float,
    route: str,
) -> None:
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_plasma_L <= 0:
        raise ValueError("vd_plasma_L must be > 0")
    if kp_muscle <= 0:
        raise ValueError("kp_muscle must be > 0")
    if muscle_volume_L <= 0:
        raise ValueError("muscle_volume_L must be > 0")
    if muscle_blood_flow_L_per_h <= 0:
        raise ValueError("muscle_blood_flow_L_per_h must be > 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if route not in {"iv_bolus", "oral"}:
        raise ValueError("route must be 'iv_bolus' or 'oral'")


def simulate_muscle_pk(
    drug_name: str,
    dose_mg: float,
    route: str,
    cl_sys_L_per_h: float,
    vd_plasma_L: float,
    kp_muscle: float,
    muscle_volume_L: float = 28.0,
    muscle_blood_flow_L_per_h: float = 75.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> MusclePKResult:
    """Simulate drug distribution into skeletal muscle tissue.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose in milligrams.
    route:
        'iv_bolus' or 'oral' (F=0.8, ka=1.0/h).
    cl_sys_L_per_h:
        Systemic clearance (L/h).
    vd_plasma_L:
        Plasma/central volume of distribution (L).
    kp_muscle:
        Muscle-to-plasma partition coefficient.
    muscle_volume_L:
        Volume of skeletal muscle (L). Default 28.0.
    muscle_blood_flow_L_per_h:
        Muscle blood flow (L/h). Default 75.0.
    t_end_h:
        Simulation end time (h). Default 24.0.
    dt_h:
        Time step (h). Default 0.05.

    Returns
    -------
    MusclePKResult
    """
    _validate_inputs(
        dose_mg,
        cl_sys_L_per_h,
        vd_plasma_L,
        kp_muscle,
        muscle_volume_L,
        muscle_blood_flow_L_per_h,
        t_end_h,
        route,
    )

    ka = 1.0  # h^-1 (oral absorption rate)
    f_oral = 0.8

    # Rate constants
    k_elim = cl_sys_L_per_h / vd_plasma_L  # h^-1
    k_in = muscle_blood_flow_L_per_h / vd_plasma_L  # plasma -> muscle (per h)
    k_out = muscle_blood_flow_L_per_h / (kp_muscle * muscle_volume_L)  # muscle -> plasma (per h)

    # Initial conditions
    if route == "iv_bolus":
        a_plasma = dose_mg
        a_muscle = 0.0
        a_gut = 0.0
    else:  # oral
        a_plasma = 0.0
        a_muscle = 0.0
        a_gut = dose_mg * f_oral

    times_h: list = []
    c_plasma_list: list = []
    c_muscle_list: list = []

    n_steps = int(math.ceil(t_end_h / dt_h))
    t = 0.0

    for step in range(n_steps + 1):
        c_plasma = a_plasma / vd_plasma_L
        c_muscle = a_muscle / muscle_volume_L

        times_h.append(t)
        c_plasma_list.append(c_plasma)
        c_muscle_list.append(c_muscle)

        if step == n_steps:
            break

        # Derivatives
        d_a_plasma = -k_elim * a_plasma - k_in * a_plasma + k_out * a_muscle
        d_a_muscle = k_in * a_plasma - k_out * a_muscle
        d_a_gut = 0.0

        if route == "oral":
            d_a_plasma += ka * a_gut
            d_a_gut = -ka * a_gut

        # Forward Euler
        a_plasma = max(0.0, a_plasma + dt_h * d_a_plasma)
        a_muscle = max(0.0, a_muscle + dt_h * d_a_muscle)
        a_gut = max(0.0, a_gut + dt_h * d_a_gut)
        t = round(t + dt_h, 10)

    auc_plasma = _trapezoidal_auc(times_h, c_plasma_list)
    auc_muscle = _trapezoidal_auc(times_h, c_muscle_list)
    cmax_plasma = max(c_plasma_list)
    cmax_muscle = max(c_muscle_list)
    tmax_muscle_h = times_h[c_muscle_list.index(cmax_muscle)]

    if cmax_plasma > 0:
        muscle_to_plasma_ratio = cmax_muscle / cmax_plasma
    else:
        muscle_to_plasma_ratio = 0.0

    notes = (
        f"Perfusion-limited 2-cpt model; kp_muscle={kp_muscle:.3f}; "
        f"route={route}; muscle_blood_flow={muscle_blood_flow_L_per_h} L/h"
    )

    return MusclePKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        kp_muscle=kp_muscle,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        c_muscle_mg_L=c_muscle_list,
        auc_plasma_mg_h_per_L=auc_plasma,
        cmax_plasma_mg_L=cmax_plasma,
        auc_muscle_mg_h_per_L=auc_muscle,
        cmax_muscle_mg_L=cmax_muscle,
        tmax_muscle_h=tmax_muscle_h,
        muscle_to_plasma_ratio=muscle_to_plasma_ratio,
        notes=notes,
    )


def compare_muscle_distribution(
    drug_name: str,
    dose_mg: float,
    kp_muscle_list: list,
    cl_sys_L_per_h: float,
    vd_plasma_L: float,
    muscle_volume_L: float = 28.0,
    muscle_blood_flow_L_per_h: float = 75.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
    route: str = "iv_bolus",
) -> list:
    """Compare muscle distribution for different kp_muscle values.

    Returns a list of MusclePKResult sorted by auc_muscle descending.
    """
    results = []
    for kp in kp_muscle_list:
        result = simulate_muscle_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            route=route,
            cl_sys_L_per_h=cl_sys_L_per_h,
            vd_plasma_L=vd_plasma_L,
            kp_muscle=kp,
            muscle_volume_L=muscle_volume_L,
            muscle_blood_flow_L_per_h=muscle_blood_flow_L_per_h,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(result)
    results.sort(key=lambda r: r.auc_muscle_mg_h_per_L, reverse=True)
    return results
