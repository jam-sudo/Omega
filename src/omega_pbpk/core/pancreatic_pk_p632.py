"""
Phase 632 — Drug Pancreatic Distribution
Models drug distribution in pancreatic tissue, relevant for diabetes drugs and pancreatitis.
"""

import math
from dataclasses import dataclass


@dataclass
class PancreaticPKResult:
    drug_name: str
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    c_pancreas_mg_g: list
    c_pancreatic_juice_mg_mL: list
    kp_pancreas: float
    cmax_plasma_mg_L: float
    cmax_pancreas_mg_g: float
    auc_plasma_mg_h_per_L: float
    auc_pancreas_mg_h_per_g: float
    pancreas_plasma_ratio: float
    secretion_fraction: float
    t_half_pancreas_h: float
    notes: str


def simulate_pancreatic_pk(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    pancreas_weight_g: float = 100.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> PancreaticPKResult:
    """Simulate pancreatic drug distribution via a compartmental PK model."""
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if pancreas_weight_g <= 0:
        raise ValueError("pancreas_weight_g must be > 0")

    # Pancreatic Kp: clamp [0.2, 12.0]
    kp = (logP + 1.0) * 0.6 / max(1.0, mw_Da / 250.0)
    kp = max(0.5, kp)
    kp = max(0.2, min(12.0, kp))

    # Pancreatic blood flow (~12 mL/min per 100g → 0.72 L/h per 100g ≈ 0.75 L/h)
    Qp = 0.75  # L/h

    # Pancreatic juice secretion rate constant
    k_juice = 0.02 * kp  # /h

    # Pancreatic volume of distribution
    Vd_pancreas_L = pancreas_weight_g * kp / 1000.0

    # Transfer rate constants
    k_in = Qp / vd_sys_L
    k_out = Qp / Vd_pancreas_L

    # Absorption
    ka = 1.2  # /h
    F = 0.85

    # Initial conditions
    A_gut = dose_mg * F  # mg
    C_plasma = 0.0  # mg/L
    C_pancreas = 0.0  # mg/L (in pancreatic compartment)
    C_juice = 0.0  # mg/L (pancreatic juice proxy)

    # Output lists
    times_h = []
    c_plasma_list = []
    c_pancreas_list = []
    c_juice_list = []

    n_steps = int(round(t_end_h / dt_h))

    for step in range(n_steps + 1):
        t = step * dt_h

        # Convert C_pancreas from mg/L in compartment to mg/g tissue
        c_panc_mg_g = C_pancreas * Vd_pancreas_L * 1000.0 / pancreas_weight_g
        c_juice_mg_mL = C_juice / 1000.0  # mg/L -> mg/mL

        times_h.append(t)
        c_plasma_list.append(C_plasma)
        c_pancreas_list.append(c_panc_mg_g)
        c_juice_list.append(c_juice_mg_mL)

        if step == n_steps:
            break

        # ODEs
        gut_absorption = ka * A_gut  # mg/h

        # dC_plasma/dt
        dC_plasma = (
            gut_absorption / vd_sys_L
            - (cl_sys_L_per_h / vd_sys_L) * C_plasma
            - k_in * C_plasma
            + k_out * C_pancreas * (Vd_pancreas_L / vd_sys_L)
        )

        # dC_pancreas/dt
        dC_pancreas = (
            k_in * C_plasma * (vd_sys_L / Vd_pancreas_L) - k_out * C_pancreas - k_juice * C_pancreas
        )

        # dC_juice/dt (juice concentration proxy)
        dC_juice = k_juice * C_pancreas - 0.3 * C_juice

        # dA_gut/dt
        dA_gut = -ka * A_gut

        # Forward Euler
        A_gut = A_gut + dA_gut * dt_h
        C_plasma = C_plasma + dC_plasma * dt_h
        C_pancreas = C_pancreas + dC_pancreas * dt_h
        C_juice = C_juice + dC_juice * dt_h

        # Clamp
        A_gut = max(0.0, A_gut)
        C_plasma = max(0.0, C_plasma)
        C_pancreas = max(0.0, C_pancreas)
        C_juice = max(0.0, C_juice)

    # Derived metrics
    cmax_plasma = max(c_plasma_list)
    cmax_pancreas = max(c_pancreas_list)

    # AUC via trapezoidal rule
    auc_plasma = sum(
        0.5 * (c_plasma_list[i] + c_plasma_list[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )
    auc_pancreas = sum(
        0.5 * (c_pancreas_list[i] + c_pancreas_list[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    pancreas_plasma_ratio = auc_pancreas / auc_plasma if auc_plasma > 0 else 0.0

    # Secretion fraction: fraction of pancreatic clearance via juice
    secretion_fraction = k_juice / (k_juice + k_out) if (k_juice + k_out) > 0 else 0.0

    # t_half_pancreas: ln(2) / (k_out + k_juice)
    t_half_pancreas_h = math.log(2.0) / (k_out + k_juice) if (k_out + k_juice) > 0 else float("inf")

    notes = (
        f"kp_pancreas={kp:.4f}, Vd_pancreas={Vd_pancreas_L:.4f}L, "
        f"k_in={k_in:.4f}/h, k_out={k_out:.4f}/h, k_juice={k_juice:.4f}/h"
    )

    return PancreaticPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        c_pancreas_mg_g=c_pancreas_list,
        c_pancreatic_juice_mg_mL=c_juice_list,
        kp_pancreas=kp,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_pancreas_mg_g=cmax_pancreas,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_pancreas_mg_h_per_g=auc_pancreas,
        pancreas_plasma_ratio=pancreas_plasma_ratio,
        secretion_fraction=secretion_fraction,
        t_half_pancreas_h=t_half_pancreas_h,
        notes=notes,
    )
