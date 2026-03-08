"""Phase 627 — Drug Splenic Distribution.

Models drug distribution in the spleen, relevant for nanoparticles and biologics.
Uses a 2-compartment (plasma + spleen) model with optional macrophage uptake.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SplenicPKResult:
    """Result of splenic drug distribution simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    c_spleen_mg_g: list
    kp_spleen: float
    cmax_plasma_mg_L: float
    cmax_spleen_mg_g: float
    auc_plasma_mg_h_per_L: float
    auc_spleen_mg_h_per_g: float
    spleen_to_plasma_auc_ratio: float
    macrophage_uptake_fraction: float
    t_half_spleen_h: float
    notes: str


def simulate_splenic_pk(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    spleen_weight_g: float = 150.0,
    nanoparticle: bool = False,
    t_end_h: float = 72.0,
    dt_h: float = 0.1,
) -> SplenicPKResult:
    """Simulate drug distribution into the spleen.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Oral dose in milligrams.
    logP:
        Octanol-water partition coefficient (log units).
    mw_Da:
        Molecular weight in Daltons.
    cl_sys_L_per_h:
        Systemic clearance (L/h).
    vd_sys_L:
        Systemic volume of distribution (L).
    spleen_weight_g:
        Spleen weight in grams (default 150 g).
    nanoparticle:
        If True, applies high MPS uptake Kp (default False).
    t_end_h:
        Simulation end time in hours.
    dt_h:
        Forward Euler time step in hours.

    Returns
    -------
    SplenicPKResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if spleen_weight_g <= 0:
        raise ValueError("spleen_weight_g must be > 0")

    # Spleen partition coefficient
    if nanoparticle:
        kp_spleen = 20.0
    else:
        raw_kp = (logP + 1.5) * 0.8 / max(1.0, mw_Da / 400.0)
        kp_spleen = max(0.2, min(10.0, raw_kp))

    # Splenic blood flow ~1% of CO
    Qs = 3.6  # L/h

    # Splenic distribution volume
    Vd_spleen_L = spleen_weight_g * kp_spleen / 1000.0

    # Transfer rate constants
    k_in = Qs / vd_sys_L
    k_out = Qs / Vd_spleen_L

    # Macrophage irreversible uptake rate
    k_mac = 0.0 if (mw_Da < 1000 and not nanoparticle) else 0.05  # /h

    # Half-life in spleen
    t_half_spleen_h = math.log(2.0) / (k_out + k_mac) if (k_out + k_mac) > 0 else float("inf")

    # Initial conditions: oral dose
    f_oral = 0.85
    ka = 1.2  # /h
    A_gut = dose_mg * f_oral
    C_plasma = 0.0
    C_spleen = 0.0  # concentration in spleen (mg/L equivalent volume)

    n_steps = int(math.ceil(t_end_h / dt_h))

    times_h: list = []
    c_plasma_list: list = []
    c_spleen_mg_g_list: list = []

    A_mac_total = 0.0  # cumulative macrophage uptake (mg)

    for step in range(n_steps + 1):
        t = step * dt_h

        # Convert spleen concentration to mg/g
        c_spleen_mg_g_val = C_spleen * Vd_spleen_L * 1000.0 / spleen_weight_g

        times_h.append(t)
        c_plasma_list.append(max(0.0, C_plasma))
        c_spleen_mg_g_list.append(max(0.0, c_spleen_mg_g_val))

        if step == n_steps:
            break

        # Forward Euler
        dA_gut = -ka * A_gut

        dC_plasma = (
            ka * A_gut / vd_sys_L
            - (cl_sys_L_per_h / vd_sys_L) * C_plasma
            - k_in * C_plasma
            + k_out * C_spleen * (Vd_spleen_L / vd_sys_L)
        )

        dC_spleen = k_in * C_plasma * (vd_sys_L / Vd_spleen_L) - k_out * C_spleen - k_mac * C_spleen

        # Accumulate macrophage uptake
        A_mac_total += k_mac * C_spleen * Vd_spleen_L * dt_h

        A_gut = max(0.0, A_gut + dA_gut * dt_h)
        C_plasma = max(0.0, C_plasma + dC_plasma * dt_h)
        C_spleen = max(0.0, C_spleen + dC_spleen * dt_h)

    # Derived metrics
    cmax_plasma = max(c_plasma_list)
    cmax_spleen = max(c_spleen_mg_g_list)

    # AUC by manual trapezoidal rule
    auc_plasma = sum(
        0.5 * (c_plasma_list[i] + c_plasma_list[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )
    auc_spleen = sum(
        0.5 * (c_spleen_mg_g_list[i] + c_spleen_mg_g_list[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    spleen_to_plasma_auc_ratio = auc_spleen / auc_plasma if auc_plasma > 0 else 0.0

    # Macrophage uptake fraction
    macrophage_uptake_fraction = (
        min(1.0, A_mac_total / (dose_mg * f_oral)) if (dose_mg * f_oral) > 0 else 0.0
    )

    notes = (
        f"{drug_name}: dose={dose_mg} mg, logP={logP:.2f}, MW={mw_Da:.0f} Da, "
        f"kp_spleen={kp_spleen:.2f}, nanoparticle={nanoparticle}, "
        f"k_mac={k_mac:.3f}/h, t_half_spleen={t_half_spleen_h:.2f} h"
    )

    return SplenicPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        c_spleen_mg_g=c_spleen_mg_g_list,
        kp_spleen=kp_spleen,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_spleen_mg_g=cmax_spleen,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_spleen_mg_h_per_g=auc_spleen,
        spleen_to_plasma_auc_ratio=spleen_to_plasma_auc_ratio,
        macrophage_uptake_fraction=macrophage_uptake_fraction,
        t_half_spleen_h=t_half_spleen_h,
        notes=notes,
    )
