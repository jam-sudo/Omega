"""Phase 355 — Peritoneal Drug Absorption Model.

Models drug absorption from the peritoneal cavity (intraperitoneal injection)
to systemic circulation using a 3-compartment Forward Euler approach:
peritoneal → portal → systemic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PeritonealAbsorptionResult:
    """Result of peritoneal absorption simulation."""

    times_h: list
    c_peritoneal_mg_mL: list
    c_portal_mg_L: list
    c_systemic_mg_L: list
    auc_systemic_mg_h_per_L: float
    cmax_systemic_mg_L: float
    tmax_systemic_h: float
    bioavailability_pct: float
    first_pass_loss_pct: float
    notes: str


def simulate_peritoneal_absorption(
    drug_name: str,
    dose_mg: float,
    k_abs_peritoneal_per_h: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    f_hepatic: float = 0.0,
    k_portal_to_sys_per_h: float = 30.0,
    v_peritoneal_mL: float = 50.0,
    v_portal_mL: float = 100.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.01,
) -> PeritonealAbsorptionResult:
    """Simulate peritoneal drug absorption.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose administered intraperitoneally (mg).
    k_abs_peritoneal_per_h:
        First-order absorption rate from peritoneum to portal (1/h).
    cl_sys_L_per_h:
        Systemic clearance (L/h).
    vd_sys_L:
        Volume of distribution of systemic compartment (L).
    f_hepatic:
        Hepatic extraction fraction (0 to <1).
    k_portal_to_sys_per_h:
        Transfer rate from portal to systemic (1/h), default 30.0.
    v_peritoneal_mL:
        Volume of peritoneal cavity (mL), default 50.0.
    v_portal_mL:
        Volume of portal compartment (mL), default 100.0.
    t_end_h:
        Simulation end time (h).
    dt_h:
        Time step for Forward Euler (h).

    Returns
    -------
    PeritonealAbsorptionResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")
    if cl_sys_L_per_h <= 0:
        raise ValueError(f"cl_sys_L_per_h must be positive, got {cl_sys_L_per_h}")
    if vd_sys_L <= 0:
        raise ValueError(f"vd_sys_L must be positive, got {vd_sys_L}")
    if not (0.0 <= f_hepatic < 1.0):
        raise ValueError(f"f_hepatic must be in [0, 1), got {f_hepatic}")
    if k_abs_peritoneal_per_h <= 0:
        raise ValueError(f"k_abs_peritoneal_per_h must be positive, got {k_abs_peritoneal_per_h}")
    if v_peritoneal_mL <= 0:
        raise ValueError(f"v_peritoneal_mL must be positive, got {v_peritoneal_mL}")
    if v_portal_mL <= 0:
        raise ValueError(f"v_portal_mL must be positive, got {v_portal_mL}")

    ke_sys = cl_sys_L_per_h / vd_sys_L

    # State variables
    a_peri = dose_mg  # amount in peritoneal (mg)
    a_portal = 0.0  # amount in portal (mg)
    c_sys = 0.0  # concentration in systemic (mg/L)

    times_h: list = []
    c_peri_list: list = []
    c_portal_list: list = []
    c_sys_list: list = []

    t = 0.0
    n_steps = int(math.ceil(t_end_h / dt_h))

    for _ in range(n_steps + 1):
        # Record state
        times_h.append(t)
        c_peri_list.append(a_peri / v_peritoneal_mL)  # mg/mL
        c_portal_list.append(a_portal * 1000.0 / v_portal_mL)  # mg/L
        c_sys_list.append(c_sys)

        if t >= t_end_h:
            break

        # ODEs
        da_peri_dt = -k_abs_peritoneal_per_h * a_peri
        da_portal_dt = k_abs_peritoneal_per_h * a_peri - k_portal_to_sys_per_h * a_portal
        dc_sys_dt = k_portal_to_sys_per_h * a_portal * (1.0 - f_hepatic) / vd_sys_L - ke_sys * c_sys

        # Forward Euler update
        a_peri += da_peri_dt * dt_h
        a_portal += da_portal_dt * dt_h
        c_sys += dc_sys_dt * dt_h

        # Clamp to non-negative
        if a_peri < 0.0:
            a_peri = 0.0
        if a_portal < 0.0:
            a_portal = 0.0
        if c_sys < 0.0:
            c_sys = 0.0

        t = round(t + dt_h, 10)

    # AUC via trapezoidal rule
    auc_systemic = sum(
        0.5 * (c_sys_list[i] + c_sys_list[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    cmax_systemic = max(c_sys_list)
    tmax_idx = c_sys_list.index(cmax_systemic)
    tmax_systemic = times_h[tmax_idx]

    # Bioavailability: compare IP AUC to hypothetical IV AUC (no first-pass)
    # IV AUC = dose / cl_sys (complete bioavailability, no first-pass)
    # IP bioavailability accounts for first-pass hepatic extraction
    # AUC_iv_reference = dose / cl_sys
    auc_iv_ref = dose_mg / cl_sys_L_per_h
    bioavailability_pct = (auc_systemic / auc_iv_ref) * 100.0 if auc_iv_ref > 0 else 0.0

    first_pass_loss_pct = f_hepatic * 100.0

    notes = (
        f"3-compartment peritoneal absorption model for {drug_name}. "
        f"k_abs={k_abs_peritoneal_per_h:.3f}/h, "
        f"f_hepatic={f_hepatic:.2f}, "
        f"first-pass loss={first_pass_loss_pct:.1f}%."
    )

    return PeritonealAbsorptionResult(
        times_h=times_h,
        c_peritoneal_mg_mL=c_peri_list,
        c_portal_mg_L=c_portal_list,
        c_systemic_mg_L=c_sys_list,
        auc_systemic_mg_h_per_L=auc_systemic,
        cmax_systemic_mg_L=cmax_systemic,
        tmax_systemic_h=tmax_systemic,
        bioavailability_pct=bioavailability_pct,
        first_pass_loss_pct=first_pass_loss_pct,
        notes=notes,
    )
