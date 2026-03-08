"""Phase 860 — Intrathecal Drug Distribution.

Models intrathecal (spinal) drug administration and CSF distribution
with lumbar and cisternal compartments.
"""

import math
from dataclasses import dataclass


@dataclass
class IntrathecalPKResult:
    """Result of intrathecal PK simulation."""

    drug_name: str
    dose_mg: float
    injection_site: str
    times_h: list
    c_csf_lumbar_mg_L: list
    c_csf_cistern_mg_L: list
    c_plasma_mg_L: list
    cmax_lumbar_mg_L: float
    cmax_cistern_mg_L: float
    cmax_plasma_mg_L: float
    auc_lumbar_mg_h_per_L: float
    auc_cistern_mg_h_per_L: float
    t_half_csf_h: float
    systemic_exposure_pct: float
    notes: str


def _trapezoidal_auc(times, values):
    """Manual trapezoidal AUC."""
    return sum(
        0.5 * (values[i] + values[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )


def simulate_intrathecal_pk(
    drug_name,
    dose_mg,
    logP,
    mw_Da,
    cl_sys_L_per_h,
    vd_sys_L,
    injection_site="lumbar",
    csf_bulk_flow_mL_per_h=21.0,
    t_end_h=24.0,
    dt_h=0.05,
):
    """Simulate intrathecal drug distribution in CSF compartments.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Intrathecal dose in mg.
    logP : float
        Octanol-water partition coefficient.
    mw_Da : float
        Molecular weight in Daltons.
    cl_sys_L_per_h : float
        Systemic clearance (L/h).
    vd_sys_L : float
        Volume of distribution (L).
    injection_site : str
        "lumbar" or "cisternal".
    csf_bulk_flow_mL_per_h : float
        CSF bulk flow rate (mL/h).
    t_end_h : float
        Simulation duration (h).
    dt_h : float
        Output time step (h).

    Returns
    -------
    IntrathecalPKResult
    """
    valid_sites = {"lumbar", "cisternal"}
    if injection_site not in valid_sites:
        raise ValueError(f"injection_site must be one of {valid_sites}")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be positive")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be positive")
    if csf_bulk_flow_mL_per_h <= 0:
        raise ValueError("csf_bulk_flow_mL_per_h must be positive")

    # CSF compartment volumes
    v_lumbar = 0.03  # L (30 mL)
    v_cistern = 0.02  # L (20 mL)
    v_csf_total = 0.025  # L (25 mL combined for transfer calc)

    # Rate constants
    k_lc = 0.3  # lumbar -> cistern /h
    k_cl = 0.1  # cistern -> lumbar /h
    k_elim_csf = 0.05  # CSF elimination /h
    k_bbb = 0.02  # BBB back-transfer /h
    ke = cl_sys_L_per_h / vd_sys_L

    # Adaptive time step
    dt_int = min(dt_h, 0.4 / max(k_lc, k_cl, k_elim_csf, k_bbb, ke))

    # Initial conditions
    if injection_site == "lumbar":
        c_lumbar = dose_mg / v_lumbar
        c_cistern = 0.0
    else:
        c_lumbar = 0.0
        c_cistern = dose_mg / v_cistern
    c_plasma = 0.0

    # Half-life in CSF (lumbar reference)
    t_half_csf = math.log(2) / (k_lc + k_elim_csf)

    # Output storage
    times_h = []
    c_lumbar_list = []
    c_cistern_list = []
    c_plasma_list = []

    t = 0.0
    next_out = 0.0

    while t <= t_end_h + 1e-9:
        if t >= next_out - 1e-9:
            times_h.append(round(t, 10))
            c_lumbar_list.append(c_lumbar)
            c_cistern_list.append(c_cistern)
            c_plasma_list.append(c_plasma)
            next_out += dt_h

        # Forward Euler ODEs
        dc_lumbar = -k_lc * c_lumbar - k_elim_csf * c_lumbar + k_cl * c_cistern
        dc_cistern = k_lc * c_lumbar - k_cl * c_cistern - k_elim_csf * c_cistern
        dc_plasma = (
            k_bbb * (c_lumbar + c_cistern) * v_csf_total / vd_sys_L
            - ke * c_plasma
        )

        c_lumbar += dc_lumbar * dt_int
        c_cistern += dc_cistern * dt_int
        c_plasma += dc_plasma * dt_int

        # Clamp non-negative
        c_lumbar = max(0.0, c_lumbar)
        c_cistern = max(0.0, c_cistern)
        c_plasma = max(0.0, c_plasma)

        t += dt_int

    # Final point
    if len(times_h) == 0 or times_h[-1] < t_end_h - 1e-9:
        times_h.append(round(t_end_h, 10))
        c_lumbar_list.append(c_lumbar)
        c_cistern_list.append(c_cistern)
        c_plasma_list.append(c_plasma)

    cmax_lumbar = max(c_lumbar_list)
    cmax_cistern = max(c_cistern_list)
    cmax_plasma = max(c_plasma_list)
    auc_lumbar = _trapezoidal_auc(times_h, c_lumbar_list)
    auc_cistern = _trapezoidal_auc(times_h, c_cistern_list)
    auc_plasma = _trapezoidal_auc(times_h, c_plasma_list)

    # Systemic exposure percentage
    if cl_sys_L_per_h > 0:
        iv_auc = dose_mg / cl_sys_L_per_h
        systemic_exposure_pct = max(0.0, min(100.0, (auc_plasma / iv_auc) * 100))
    else:
        systemic_exposure_pct = 0.0

    return IntrathecalPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        injection_site=injection_site,
        times_h=times_h,
        c_csf_lumbar_mg_L=c_lumbar_list,
        c_csf_cistern_mg_L=c_cistern_list,
        c_plasma_mg_L=c_plasma_list,
        cmax_lumbar_mg_L=cmax_lumbar,
        cmax_cistern_mg_L=cmax_cistern,
        cmax_plasma_mg_L=cmax_plasma,
        auc_lumbar_mg_h_per_L=auc_lumbar,
        auc_cistern_mg_h_per_L=auc_cistern,
        t_half_csf_h=t_half_csf,
        systemic_exposure_pct=systemic_exposure_pct,
        notes=f"Intrathecal PK for {drug_name}, site={injection_site}",
    )
