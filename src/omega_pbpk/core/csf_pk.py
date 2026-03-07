"""Phase 1037 — Drug Cerebrospinal Fluid PK (core/csf_pk.py).

Models drug penetration into CSF relevant for CNS drugs and intrathecal therapy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def _trapz(times: list[float], values: list[float]) -> float:
    """Manual trapezoidal integration."""
    if len(times) < 2:
        return 0.0
    total = 0.0
    for i in range(len(times) - 1):
        total += 0.5 * (values[i] + values[i + 1]) * (times[i + 1] - times[i])
    return total


@dataclass
class CSFPKResult:
    """Result of CSF PK simulation."""

    drug_name: str
    dose_mg: float
    route: str
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c_csf_mg_L: list = field(default_factory=list)
    c_brain_mg_L: list = field(default_factory=list)
    auc_plasma: float = 0.0
    auc_csf: float = 0.0
    cmax_csf_mg_L: float = 0.0
    tmax_csf_h: float = 0.0
    csf_to_plasma_ratio: float = 0.0
    bbb_penetration: str = "low"
    t_half_csf_h: float = 0.0
    notes: str = ""


def simulate_csf_pk(
    drug_name: str,
    dose_mg: float,
    route: str = "systemic",
    mw_Da: float = 300.0,
    logp: float = 2.0,
    fu_plasma: float = 0.3,
    cl_sys_L_per_h: float = 5.0,
    vd_sys_L: float = 30.0,
    v_csf_mL: float = 140.0,
    csf_turnover_per_h: float = 0.25,
    kp_bbb: float = 0.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> CSFPKResult:
    """Simulate drug PK in cerebrospinal fluid (CSF).

    Parameters
    ----------
    drug_name: Name of the drug.
    dose_mg: Dose in mg.
    route: "systemic" or "intrathecal".
    mw_Da: Molecular weight in Daltons.
    logp: Octanol-water partition coefficient.
    fu_plasma: Unbound fraction in plasma (0-1].
    cl_sys_L_per_h: Systemic clearance in L/h.
    vd_sys_L: Volume of distribution (systemic) in L.
    v_csf_mL: CSF volume in mL (default ~140 mL).
    csf_turnover_per_h: CSF production/absorption rate constant (/h).
    kp_bbb: BBB partition coefficient override (0 = compute from logP/MW/fu).
    t_end_h: Simulation end time in hours.
    dt_h: Time step in hours.

    Returns
    -------
    CSFPKResult with plasma, CSF, and brain time-concentration profiles.
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if v_csf_mL <= 0:
        raise ValueError("v_csf_mL must be > 0")
    if route not in {"systemic", "intrathecal"}:
        raise ValueError(f"route must be 'systemic' or 'intrathecal', got {route!r}")
    if kp_bbb < 0:
        raise ValueError("kp_bbb must be >= 0")

    # Compute BBB Kp if not provided
    if kp_bbb == 0.0:
        kp_bbb = max(0.01, min(2.0, (10 ** (0.5 * logp - 0.01 * mw_Da)) * fu_plasma))

    v_csf_L = v_csf_mL / 1000.0

    # Rate constants
    ke = cl_sys_L_per_h / vd_sys_L  # plasma elimination rate (/h)

    # Plasma -> CSF transfer (slow passive, scaled by BBB Kp)
    k12 = (0.1 / vd_sys_L) * kp_bbb  # /h

    # CSF -> plasma drainage
    k21 = csf_turnover_per_h  # /h

    # Brain tissue transfer rates
    k_br_in = k12 * 2.0  # /h (faster exchange than CSF)
    kp_brain_blood = kp_bbb * 1.5
    k_br_out = k_br_in / max(kp_brain_blood, 1e-9)  # /h

    # Volume ratio for mass-conservation in CSF ODE
    vol_ratio = vd_sys_L / v_csf_L

    # Initial conditions
    if route == "systemic":
        c_plasma = dose_mg / vd_sys_L
        c_csf = 0.0
        c_brain = 0.0
    else:  # intrathecal
        c_plasma = 0.0
        c_csf = dose_mg / v_csf_L
        c_brain = 0.0

    times: list[float] = []
    cp_list: list[float] = []
    ccsf_list: list[float] = []
    cbrain_list: list[float] = []

    t = 0.0
    n_steps = max(1, round(t_end_h / dt_h))

    for _i in range(n_steps + 1):
        times.append(t)
        cp_list.append(c_plasma)
        ccsf_list.append(c_csf)
        cbrain_list.append(c_brain)

        if _i == n_steps:
            break

        if route == "systemic":
            # dC_plasma/dt = -ke*Cp - k12*Cp + k21*Ccsf
            # dC_csf/dt = k12*Cp*(Vd/Vcsf) - k21*Ccsf - csf_turnover*Ccsf
            # dC_brain/dt = k_br_in*Cp - k_br_out*Cbrain
            dcp = -ke * c_plasma - k12 * c_plasma + k21 * c_csf
            dccsf = k12 * c_plasma * vol_ratio - k21 * c_csf - csf_turnover_per_h * c_csf
            dcbrain = k_br_in * c_plasma - k_br_out * c_brain
        else:
            # Intrathecal: CSF drains to plasma
            # dC_csf/dt = -csf_turnover*Ccsf
            # dC_plasma/dt = csf_turnover*Ccsf*(Vcsf/Vd) - ke*Cp
            # dC_brain/dt = k_br_in*Cp - k_br_out*Cbrain
            dcp = csf_turnover_per_h * c_csf * (v_csf_L / vd_sys_L) - ke * c_plasma
            dccsf = -csf_turnover_per_h * c_csf
            dcbrain = k_br_in * c_plasma - k_br_out * c_brain

        c_plasma = max(0.0, c_plasma + dcp * dt_h)
        c_csf = max(0.0, c_csf + dccsf * dt_h)
        c_brain = max(0.0, c_brain + dcbrain * dt_h)
        t += dt_h

    auc_plasma = _trapz(times, cp_list)
    auc_csf = _trapz(times, ccsf_list)

    cmax_csf = max(ccsf_list)
    tmax_csf = times[ccsf_list.index(cmax_csf)]

    if auc_plasma > 0:
        csf_to_plasma_ratio = auc_csf / auc_plasma
    else:
        csf_to_plasma_ratio = 0.0

    # BBB penetration classification
    if csf_to_plasma_ratio > 0.3:
        bbb_penetration = "high"
    elif csf_to_plasma_ratio >= 0.1:
        bbb_penetration = "moderate"
    else:
        bbb_penetration = "low"

    # CSF half-life approximation
    t_half_csf = math.log(2) / max(csf_turnover_per_h, 1e-9)

    notes = (
        f"BBB Kp={kp_bbb:.3f}; ke={ke:.3f}/h; k12={k12:.4f}/h; "
        f"CSF turnover t1/2={t_half_csf:.1f}h; route={route}"
    )

    return CSFPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        times_h=times,
        c_plasma_mg_L=cp_list,
        c_csf_mg_L=ccsf_list,
        c_brain_mg_L=cbrain_list,
        auc_plasma=auc_plasma,
        auc_csf=auc_csf,
        cmax_csf_mg_L=cmax_csf,
        tmax_csf_h=tmax_csf,
        csf_to_plasma_ratio=csf_to_plasma_ratio,
        bbb_penetration=bbb_penetration,
        t_half_csf_h=t_half_csf,
        notes=notes,
    )
