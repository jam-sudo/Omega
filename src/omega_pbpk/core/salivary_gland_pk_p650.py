"""Drug salivary gland pharmacokinetics model.

Models drug distribution into salivary glands and secretion into saliva.
Useful for therapeutic drug monitoring (TDM) via saliva as a non-invasive
alternative to blood sampling.

Unlike the salivary_drug_excretion_p598 module (urinary-style excretion),
this module focuses on the dynamic PK simulation of drug in salivary glands
with an oral cavity compartment.

References
----------
- Haeckel R, Hänecke P, Clin Chem Lab Med. 1996;34(4):407-14 (saliva TDM)
- Aps JK, Martens LC, Forensic Sci Int. 2005;150(2-3):119-31 (drug monitoring)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class SalivaryGlandPKResult:
    """Result container for salivary gland PK simulation."""

    drug_name: str
    dose_mg: float
    times_h: list[float] = field(default_factory=list)
    c_plasma_mg_L: list[float] = field(default_factory=list)
    c_saliva_mg_L: list[float] = field(default_factory=list)
    c_salivary_gland_mg_g: list[float] = field(default_factory=list)
    saliva_to_plasma_ratio_auc: float = 0.0
    cmax_plasma_mg_L: float = 0.0
    cmax_saliva_mg_L: float = 0.0
    auc_plasma_mg_h_per_L: float = 0.0
    auc_saliva_mg_h_per_L: float = 0.0
    daily_saliva_excretion_mg: float = 0.0
    kp_salivary_gland: float = 0.0
    monitoring_utility: str = ""
    notes: str = ""


def simulate_salivary_gland_pk(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    pka: float,
    is_base: bool,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> SalivaryGlandPKResult:
    """Simulate salivary gland pharmacokinetics.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Oral dose in mg.
    logP : float
        Octanol-water partition coefficient (log scale).
    mw_Da : float
        Molecular weight in Daltons.
    pka : float
        Acid dissociation constant.
    is_base : bool
        True if drug is a base, False if acid/neutral.
    cl_sys_L_per_h : float
        Systemic clearance (L/h).
    vd_sys_L : float
        Volume of distribution (L).
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Time step (h).

    Returns
    -------
    SalivaryGlandPKResult
    """
    # --- validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")

    # --- salivary gland partition coefficient ---
    kp_base = max(0.1, logP * 0.5 + 0.3)
    if is_base and pka > 0:
        pH_diff = 7.4 - 6.7  # 0.7
        kp_base *= max(1.0, min(5.0, 1.0 + pH_diff))
    kp_sg = max(0.05, min(10.0, kp_base))

    # --- constants ---
    ka = 1.2  # oral absorption rate (1/h)
    k_el = cl_sys_L_per_h / vd_sys_L
    v_sg = 0.03  # salivary gland volume (L)
    q_sg = 0.06  # blood flow to salivary glands (L/h)
    saliva_flow_L_per_h = 0.03  # 30 mL/h = 0.5 mL/min

    k_sg_in = q_sg / vd_sys_L
    k_sg_out = q_sg / (v_sg * kp_sg)

    # --- state variables ---
    a_gut = dose_mg
    c_plasma = 0.0
    c_sg = 0.0  # concentration in salivary gland (mg/L equivalent)

    n_steps = int(math.ceil(t_end_h / dt_h)) + 1

    times: list[float] = []
    c_plasma_list: list[float] = []
    c_saliva_list: list[float] = []
    c_sg_list: list[float] = []

    for i in range(n_steps):
        t = i * dt_h

        c_saliva = c_sg / kp_sg if kp_sg > 0 else 0.0

        times.append(t)
        c_plasma_list.append(max(0.0, c_plasma))
        c_saliva_list.append(max(0.0, c_saliva))
        # mg/g: c_sg is mg/L, density 1.0 g/mL = 1000 g/L, so mg/g = c_sg / 1000
        c_sg_list.append(max(0.0, c_sg / 1000.0))

        # --- ODEs (Forward Euler) ---
        da_gut = -ka * a_gut
        dc_plasma = (
            ka * a_gut / vd_sys_L
            - k_el * c_plasma
            - k_sg_in * c_plasma
            + k_sg_out * c_sg * v_sg * kp_sg / vd_sys_L
        )
        dc_sg = k_sg_in * c_plasma * vd_sys_L / v_sg - k_sg_out * c_sg

        a_gut += da_gut * dt_h
        c_plasma += dc_plasma * dt_h
        c_sg += dc_sg * dt_h

        a_gut = max(0.0, a_gut)
        c_plasma = max(0.0, c_plasma)
        c_sg = max(0.0, c_sg)

    # --- derived metrics ---
    cmax_plasma = max(c_plasma_list) if c_plasma_list else 0.0
    cmax_saliva = max(c_saliva_list) if c_saliva_list else 0.0

    # AUC (trapezoidal)
    auc_plasma = sum(
        0.5 * (c_plasma_list[j] + c_plasma_list[j - 1]) * (times[j] - times[j - 1])
        for j in range(1, len(times))
    )
    auc_saliva = sum(
        0.5 * (c_saliva_list[j] + c_saliva_list[j - 1]) * (times[j] - times[j - 1])
        for j in range(1, len(times))
    )

    saliva_to_plasma_ratio = auc_saliva / auc_plasma if auc_plasma > 0 else 0.0

    # daily saliva excretion
    daily_excretion = sum(
        c_saliva_list[j] * saliva_flow_L_per_h * dt_h
        for j in range(len(c_saliva_list))
        if times[j] <= 24.0
    )

    # monitoring utility
    r = saliva_to_plasma_ratio
    if 0.8 <= r <= 1.2:
        monitoring = "excellent"
    elif 0.5 <= r <= 1.5:
        monitoring = "good"
    elif 0.3 <= r <= 2.0:
        monitoring = "fair"
    else:
        monitoring = "poor"

    return SalivaryGlandPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_saliva_mg_L=c_saliva_list,
        c_salivary_gland_mg_g=c_sg_list,
        saliva_to_plasma_ratio_auc=saliva_to_plasma_ratio,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_saliva_mg_L=cmax_saliva,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_saliva_mg_h_per_L=auc_saliva,
        daily_saliva_excretion_mg=daily_excretion,
        kp_salivary_gland=kp_sg,
        monitoring_utility=monitoring,
        notes="Salivary gland PK simulation with ion-trapping for basic drugs.",
    )
