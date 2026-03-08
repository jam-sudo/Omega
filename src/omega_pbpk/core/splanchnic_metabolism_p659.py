"""Splanchnic first-pass drug metabolism model.

Models first-pass drug metabolism in the splanchnic region (gut wall +
portal-hepatic first pass).  Tracks drug through intestinal enterocytes,
portal vein, and liver using a 4-compartment forward Euler simulation.

References
----------
- Paine MF et al., Clin Pharmacol Ther. 2006;80(3):275-285 (gut wall CYP3A4)
- Yang J et al., Clin Pharmacokinet. 2007;46(8):681-696 (first-pass model)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class SplanchnicMetabolismResult:
    """Result container for splanchnic first-pass metabolism simulation."""

    drug_name: str
    dose_mg: float
    times_h: list[float] = field(default_factory=list)
    c_enterocyte_mg_g: list[float] = field(default_factory=list)
    c_portal_mg_L: list[float] = field(default_factory=list)
    c_hepatic_vein_mg_L: list[float] = field(default_factory=list)
    c_systemic_mg_L: list[float] = field(default_factory=list)
    fg_gut_wall: float = 0.0
    fh_hepatic: float = 0.0
    f_total: float = 0.0
    cmax_systemic_mg_L: float = 0.0
    auc_systemic_mg_h_per_L: float = 0.0
    gut_wall_extraction: float = 0.0
    hepatic_extraction: float = 0.0
    gut_to_portal_ratio_auc: float = 0.0
    notes: str = ""


def simulate_splanchnic_metabolism(
    drug_name: str,
    dose_mg: float,
    logP: float,
    cl_gut_L_per_h: float,
    cl_hepatic_L_per_h: float,
    vd_sys_L: float,
    cyp3a4_gut_activity: float = 1.0,
    cyp3a4_hepatic_activity: float = 1.0,
    t_end_h: float = 12.0,
    dt_h: float = 0.1,
) -> SplanchnicMetabolismResult:
    """Simulate first-pass splanchnic drug metabolism.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Oral dose in mg.
    logP : float
        Log partition coefficient (lipophilicity).
    cl_gut_L_per_h : float
        Gut wall intrinsic clearance (L/h).
    cl_hepatic_L_per_h : float
        Hepatic intrinsic clearance (L/h).
    vd_sys_L : float
        Systemic volume of distribution (L).
    cyp3a4_gut_activity : float
        Relative CYP3A4 activity in gut wall (1.0 = normal).
    cyp3a4_hepatic_activity : float
        Relative CYP3A4 activity in liver (1.0 = normal).
    t_end_h : float
        Simulation duration in hours.
    dt_h : float
        Time step for forward Euler integration.

    Returns
    -------
    SplanchnicMetabolismResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if cl_gut_L_per_h <= 0:
        raise ValueError("cl_gut_L_per_h must be positive")
    if cl_hepatic_L_per_h <= 0:
        raise ValueError("cl_hepatic_L_per_h must be positive")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be positive")
    if cyp3a4_gut_activity < 0:
        raise ValueError("cyp3a4_gut_activity must be non-negative")
    if cyp3a4_hepatic_activity < 0:
        raise ValueError("cyp3a4_hepatic_activity must be non-negative")

    # Physiological constants
    gut_blood_flow = 3.0  # L/h (portal blood flow fraction)
    hepatic_blood_flow = 90.0  # L/h

    # Extraction ratios
    cl_gut_eff = cl_gut_L_per_h * cyp3a4_gut_activity
    cl_hep_eff = cl_hepatic_L_per_h * cyp3a4_hepatic_activity

    gut_wall_extr = cl_gut_eff / (gut_blood_flow + cl_gut_eff)
    fg = 1.0 - gut_wall_extr

    hepatic_extr = cl_hep_eff / (hepatic_blood_flow + cl_hep_eff)
    fh = 1.0 - hepatic_extr

    f_total = fg * fh

    # Compartment parameters
    ka = 1.5  # /h absorption rate
    v_enterocyte = 0.2  # L (200 g tissue, density ~1)
    v_portal = 1.5  # L

    # Forward Euler simulation
    n_steps = int(math.ceil(t_end_h / dt_h))
    times: list[float] = []
    c_enterocyte_list: list[float] = []
    c_portal_list: list[float] = []
    c_hepatic_vein_list: list[float] = []
    c_systemic_list: list[float] = []

    # State variables
    a_gut = dose_mg  # mg in gut lumen
    c_enter = 0.0  # mg/L in enterocyte compartment
    c_portal = 0.0  # mg/L in portal vein
    c_sys = 0.0  # mg/L in systemic

    for i in range(n_steps + 1):
        t = i * dt_h

        # Hepatic vein concentration (approximate)
        c_hep_vein = c_portal * fh

        times.append(round(t, 6))
        c_enterocyte_list.append(max(c_enter, 0.0))  # mg/L ~ mg/g
        c_portal_list.append(max(c_portal, 0.0))
        c_hepatic_vein_list.append(max(c_hep_vein, 0.0))
        c_systemic_list.append(max(c_sys, 0.0))

        if i == n_steps:
            break

        # ODEs (forward Euler)
        da_gut = -ka * a_gut
        dc_enter = (
            ka * a_gut / v_enterocyte
            - (gut_blood_flow / v_enterocyte) * c_enter
            - (cl_gut_eff / v_enterocyte) * c_enter
        )
        dc_portal = (
            gut_blood_flow * c_enter * v_enterocyte / v_portal
            - (hepatic_blood_flow / v_portal) * c_portal
        )
        # Portal flow * fh enters systemic; systemic clearance
        dc_sys = hepatic_blood_flow * c_portal * fh / vd_sys_L - (cl_hep_eff / vd_sys_L) * c_sys

        a_gut += da_gut * dt_h
        c_enter += dc_enter * dt_h
        c_portal += dc_portal * dt_h
        c_sys += dc_sys * dt_h

        # Clamp negatives
        a_gut = max(a_gut, 0.0)
        c_enter = max(c_enter, 0.0)
        c_portal = max(c_portal, 0.0)
        c_sys = max(c_sys, 0.0)

    # Post-processing
    cmax_sys = max(c_systemic_list) if c_systemic_list else 0.0

    # Manual trapezoidal AUC for systemic
    auc_sys = sum(
        0.5 * (c_systemic_list[i] + c_systemic_list[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )

    # AUC for enterocyte and portal
    auc_enter = sum(
        0.5 * (c_enterocyte_list[i] + c_enterocyte_list[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )
    auc_portal = sum(
        0.5 * (c_portal_list[i] + c_portal_list[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )

    gut_to_portal = auc_enter / auc_portal if auc_portal > 0 else 0.0

    notes_parts = []
    if logP > 3.0:
        notes_parts.append("High lipophilicity may increase gut wall retention")
    if f_total < 0.2:
        notes_parts.append("Low overall oral bioavailability (<20%)")

    return SplanchnicMetabolismResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_enterocyte_mg_g=c_enterocyte_list,
        c_portal_mg_L=c_portal_list,
        c_hepatic_vein_mg_L=c_hepatic_vein_list,
        c_systemic_mg_L=c_systemic_list,
        fg_gut_wall=fg,
        fh_hepatic=fh,
        f_total=f_total,
        cmax_systemic_mg_L=cmax_sys,
        auc_systemic_mg_h_per_L=auc_sys,
        gut_wall_extraction=gut_wall_extr,
        hepatic_extraction=hepatic_extr,
        gut_to_portal_ratio_auc=gut_to_portal,
        notes="; ".join(notes_parts),
    )
