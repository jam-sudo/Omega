"""Phase 597 — Intranasal CNS Drug Delivery

Models nose-to-brain drug delivery via olfactory and trigeminal pathways.
"""

from dataclasses import dataclass


@dataclass
class IntranasalCNSDeliveryResult:
    """Result of intranasal CNS delivery simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_nasal_mg_mL: list  # nasal cavity concentration
    c_olfactory_mg_mL: list  # olfactory epithelium concentration
    c_brain_mg_mL: list  # brain (CNS target) concentration
    c_systemic_mg_L: list  # systemic (blood) concentration
    cmax_brain_mg_mL: float
    tmax_brain_h: float
    auc_brain_mg_h_per_mL: float
    auc_systemic_mg_h_per_L: float
    nose_to_brain_ratio: float  # brain AUC / systemic AUC (higher = better CNS targeting)
    brain_bioavailability_pct: float
    direct_cns_fraction: float  # fraction reaching brain via direct (not systemic) route
    notes: str


def simulate_intranasal_cns_delivery(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    cl_brain_per_h: float = 0.1,
    t_end_h: float = 6.0,
    dt_h: float = 0.02,
) -> IntranasalCNSDeliveryResult:
    """Simulate intranasal CNS drug delivery via olfactory and systemic pathways.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Intranasal dose in mg.
    logP : float
        Lipophilicity (log partition coefficient).
    mw_Da : float
        Molecular weight in Daltons.
    cl_sys_L_per_h : float
        Systemic clearance in L/h.
    vd_sys_L : float
        Volume of distribution (systemic) in L.
    cl_brain_per_h : float
        Brain clearance rate constant (/h).
    t_end_h : float
        Simulation end time in hours.
    dt_h : float
        Time step for forward Euler in hours.
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_sys_L_per_h <= 0:
        raise ValueError(f"cl_sys_L_per_h must be > 0, got {cl_sys_L_per_h}")
    if vd_sys_L <= 0:
        raise ValueError(f"vd_sys_L must be > 0, got {vd_sys_L}")

    # --- Rate constants ---
    # Direct olfactory pathway rate: logP effect / MW effect
    logP_eff = max(0.1, logP)
    mw_eff = max(1.0, mw_Da / 200.0)
    k_olf = 0.2 * logP_eff / mw_eff  # /h
    k_olf = max(0.01, min(0.5, k_olf))

    k_sys_abs = 0.3  # /h  systemic absorption from nasal
    k_mcc = 0.5  # /h  mucociliary clearance

    # BBB transfer from systemic
    k_bbb = 0.01 * logP_eff / max(1.0, mw_Da / 300.0)
    k_bbb = max(0.001, min(0.1, k_bbb))

    # Volumes
    nasal_vol_mL = 0.5
    olf_vol_mL = 0.2
    brain_vol_mL = 1400.0
    # sys_vd in L

    # --- State variables ---
    # A_nasal: amount in nasal cavity (mg)
    # C_olf: concentration in olfactory epithelium (mg/mL)
    # C_brain: concentration in brain (mg/mL)
    # C_sys: concentration in systemic (mg/L)

    A_nasal = dose_mg
    C_olf = 0.0
    C_brain = 0.0
    C_sys = 0.0

    # Storage
    times = []
    a_nasal_list = []
    c_olf_list = []
    c_brain_list = []
    c_sys_list = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h)) + 1

    for _ in range(n_steps):
        times.append(t)
        a_nasal_list.append(max(0.0, A_nasal))
        c_olf_list.append(max(0.0, C_olf))
        c_brain_list.append(max(0.0, C_brain))
        c_sys_list.append(max(0.0, C_sys))

        # ODEs (forward Euler)
        dA_nasal = -(k_olf + k_sys_abs + k_mcc) * A_nasal

        dC_olf = k_olf * A_nasal / olf_vol_mL - 0.3 * C_olf

        # Brain: from olfactory + from systemic via BBB - clearance
        # C_sys in mg/L, need *0.001 to get mg/mL
        dC_brain = (
            0.3 * C_olf * (olf_vol_mL / brain_vol_mL)
            + k_bbb * C_sys * 0.001
            - cl_brain_per_h * C_brain
        )

        # Systemic: absorption from nasal (mg/h -> mg/L/h = /vd_sys_L)
        # - elimination: (cl_sys/vd_sys)*C_sys
        dC_sys = k_sys_abs * A_nasal / vd_sys_L - (cl_sys_L_per_h / vd_sys_L) * C_sys

        # Update
        A_nasal = max(0.0, A_nasal + dt_h * dA_nasal)
        C_olf = max(0.0, C_olf + dt_h * dC_olf)
        C_brain = max(0.0, C_brain + dt_h * dC_brain)
        C_sys = max(0.0, C_sys + dt_h * dC_sys)

        t += dt_h

    # --- Derived metrics ---
    # Nasal concentration = amount / volume
    c_nasal_mg_mL = [a / nasal_vol_mL for a in a_nasal_list]

    # Cmax and Tmax for brain
    cmax_brain = max(c_brain_list)
    tmax_brain_h = times[c_brain_list.index(cmax_brain)]

    # AUC via trapezoidal rule
    def trapz(x, y):
        return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))

    auc_brain = trapz(times, c_brain_list)
    auc_systemic = trapz(times, c_sys_list)

    # Nose-to-brain ratio: brain AUC (mg*h/mL) / systemic AUC converted to same units
    # auc_systemic is in mg*h/L, convert to mg*h/mL by /1000
    if auc_systemic > 0:
        nose_to_brain_ratio = auc_brain / (auc_systemic / 1000.0)
    else:
        nose_to_brain_ratio = 0.0

    # Brain bioavailability %
    brain_bioavailability_pct = min(
        100.0,
        auc_brain * cl_brain_per_h * brain_vol_mL / dose_mg * 100.0,
    )

    # Direct CNS fraction: olfactory route vs total input to nasal
    direct_cns_fraction = k_olf / (k_olf + k_sys_abs)

    notes = f"k_olf={k_olf:.4f}/h, k_bbb={k_bbb:.4f}/h, k_sys_abs={k_sys_abs}/h, k_mcc={k_mcc}/h"

    return IntranasalCNSDeliveryResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_nasal_mg_mL=c_nasal_mg_mL,
        c_olfactory_mg_mL=c_olf_list,
        c_brain_mg_mL=c_brain_list,
        c_systemic_mg_L=c_sys_list,
        cmax_brain_mg_mL=cmax_brain,
        tmax_brain_h=tmax_brain_h,
        auc_brain_mg_h_per_mL=auc_brain,
        auc_systemic_mg_h_per_L=auc_systemic,
        nose_to_brain_ratio=nose_to_brain_ratio,
        brain_bioavailability_pct=brain_bioavailability_pct,
        direct_cns_fraction=direct_cns_fraction,
        notes=notes,
    )
