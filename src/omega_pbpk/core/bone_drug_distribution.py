"""Bone Drug Distribution PK Model.

2-compartment plasma/bone PK model for drugs with bone affinity
(bisphosphonates, tetracyclines, fluoride).

Forward Euler ODE integration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass (plain @dataclass — has list fields)
# ---------------------------------------------------------------------------


@dataclass
class BoneDrugDistributionResult:
    """Result of bone drug distribution PK simulation."""

    drug_name: str
    dose_mg: float
    logP: float
    times_h: list
    c_plasma_mg_L: list
    c_bone_mg_L: list
    cmax_plasma: float
    tmax_plasma_h: float
    auc_plasma: float
    cmax_bone: float
    tmax_bone_h: float
    auc_bone: float
    kp_bone: float
    bone_retention_pct: float
    t_half_plasma_h: float
    notes: str


# ---------------------------------------------------------------------------
# Simulation function
# ---------------------------------------------------------------------------


def simulate_bone_pk(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_plasma_L: float,
    k_bone_uptake: float,
    k_bone_release: float,
    v_bone_L: float = 2.0,
    logP: float = 0.0,
    t_end_h: float = 72.0,
    dt_h: float = 0.1,
) -> BoneDrugDistributionResult:
    """Simulate 2-compartment plasma/bone PK with Forward Euler.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        IV bolus dose in mg.
    cl_L_per_h:
        Total plasma clearance (L/h).
    vd_plasma_L:
        Plasma distribution volume (L).
    k_bone_uptake:
        Bone uptake rate constant (1/h).
    k_bone_release:
        Bone release rate constant (1/h); very slow for bisphosphonates.
    v_bone_L:
        Bone compartment volume (L). Default 2.0 L.
    logP:
        Log octanol-water partition coefficient (for notes/future use).
    t_end_h:
        Simulation end time (h). Default 72 h.
    dt_h:
        Forward Euler step size (h). Default 0.1 h.

    Returns
    -------
    BoneDrugDistributionResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_plasma_L <= 0:
        raise ValueError("vd_plasma_L must be > 0")
    if k_bone_uptake <= 0:
        raise ValueError("k_bone_uptake must be > 0")
    if k_bone_release <= 0:
        raise ValueError("k_bone_release must be > 0")
    if v_bone_L <= 0:
        raise ValueError("v_bone_L must be > 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")

    # Derived parameters
    k_el = cl_L_per_h / vd_plasma_L  # elimination rate (1/h)
    kp_bone = k_bone_uptake / k_bone_release  # equilibrium ratio

    # Initial conditions: IV bolus into plasma
    c_plasma = dose_mg / vd_plasma_L  # mg/L
    c_bone = 0.0  # mg/L

    n_steps = int(round(t_end_h / dt_h))

    times_h: list[float] = [0.0]
    c_plasma_list: list[float] = [c_plasma]
    c_bone_list: list[float] = [c_bone]

    for _ in range(n_steps):
        # dC_plasma/dt = -(k_bone_uptake + k_el)*C_plasma + k_bone_release*C_bone*V_bone/V_plasma
        dcp = -(k_bone_uptake + k_el) * c_plasma + k_bone_release * c_bone * v_bone_L / vd_plasma_L
        # dC_bone/dt = k_bone_uptake*C_plasma*V_plasma/V_bone - k_bone_release*C_bone
        dcb = k_bone_uptake * c_plasma * vd_plasma_L / v_bone_L - k_bone_release * c_bone

        c_plasma = max(0.0, c_plasma + dcp * dt_h)
        c_bone = max(0.0, c_bone + dcb * dt_h)

        times_h.append(times_h[-1] + dt_h)
        c_plasma_list.append(c_plasma)
        c_bone_list.append(c_bone)

    # --- Derived metrics ---
    # Plasma
    cmax_plasma = max(c_plasma_list)
    tmax_plasma_h = times_h[c_plasma_list.index(cmax_plasma)]
    auc_plasma = sum(
        0.5 * (c_plasma_list[i] + c_plasma_list[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    # Bone
    cmax_bone = max(c_bone_list)
    tmax_bone_h = times_h[c_bone_list.index(cmax_bone)]
    auc_bone = sum(
        0.5 * (c_bone_list[i] + c_bone_list[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    # Terminal half-life in plasma (from rate constant k_el + k_bone_uptake - feedback)
    # Approximate: use overall elimination rate
    k_total = k_el + k_bone_uptake
    t_half_plasma_h = math.log(2.0) / k_total if k_total > 0 else float("inf")

    # Bone retention: amount in bone at end as % of dose
    mass_bone_end = c_bone_list[-1] * v_bone_L
    bone_retention_pct = min(100.0, max(0.0, 100.0 * mass_bone_end / dose_mg))

    notes = (
        f"Kp_bone={kp_bone:.3f}; "
        f"k_el={k_el:.4f} 1/h; "
        f"approximate t_half_plasma={t_half_plasma_h:.2f} h; "
        f"simulation dt={dt_h} h over {t_end_h} h"
    )

    return BoneDrugDistributionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logP=logP,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        c_bone_mg_L=c_bone_list,
        cmax_plasma=cmax_plasma,
        tmax_plasma_h=tmax_plasma_h,
        auc_plasma=auc_plasma,
        cmax_bone=cmax_bone,
        tmax_bone_h=tmax_bone_h,
        auc_bone=auc_bone,
        kp_bone=kp_bone,
        bone_retention_pct=bone_retention_pct,
        t_half_plasma_h=t_half_plasma_h,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Comparison function
# ---------------------------------------------------------------------------


def compare_bone_affinity(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_plasma_L: float,
    uptake_rates: list,
    k_bone_release: float = 0.001,
    v_bone_L: float = 2.0,
    t_end_h: float = 72.0,
    dt_h: float = 0.1,
) -> list:
    """Compare bone distribution for different bone uptake rates.

    Returns list of BoneDrugDistributionResult sorted by auc_bone descending.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        IV bolus dose (mg).
    cl_L_per_h:
        Plasma clearance (L/h).
    vd_plasma_L:
        Plasma volume of distribution (L).
    uptake_rates:
        List of k_bone_uptake values (1/h) to compare.
    k_bone_release:
        Bone release rate constant (1/h). Default 0.001.
    v_bone_L:
        Bone volume (L). Default 2.0.
    t_end_h:
        Simulation end time (h). Default 72.
    dt_h:
        Time step (h). Default 0.1.
    """
    results = []
    for rate in uptake_rates:
        res = simulate_bone_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            cl_L_per_h=cl_L_per_h,
            vd_plasma_L=vd_plasma_L,
            k_bone_uptake=rate,
            k_bone_release=k_bone_release,
            v_bone_L=v_bone_L,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(res)

    results.sort(key=lambda r: r.auc_bone, reverse=True)
    return results
