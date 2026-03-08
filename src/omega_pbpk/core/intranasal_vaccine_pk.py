"""
Phase 512 — Intranasal Vaccine Delivery PK

PK model for intranasal vaccine/antigen delivery with mucosal and systemic
immune compartments. Forward Euler ODE integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IntranasalVaccinePKResult:
    """Result of intranasal vaccine PK simulation."""

    drug_name: str
    dose_mcg: float

    times_h: list = field(default_factory=list)
    c_nasal_mcg_mL: list = field(default_factory=list)
    c_mucosal_mcg_mL: list = field(default_factory=list)
    c_systemic_mcg_mL: list = field(default_factory=list)

    cmax_nasal: float = 0.0
    cmax_mucosal: float = 0.0
    cmax_systemic: float = 0.0

    tmax_nasal_h: float = 0.0
    tmax_mucosal_h: float = 0.0
    tmax_systemic_h: float = 0.0

    auc_mucosal: float = 0.0
    auc_systemic: float = 0.0

    mucosal_to_systemic_ratio: float = 0.0
    nasal_retention_pct: float = 0.0

    notes: str = ""


def simulate_intranasal_vaccine_pk(
    drug_name: str,
    dose_mcg: float,
    k_clear: float = 2.0,
    k_uptake: float = 0.5,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> IntranasalVaccinePKResult:
    """
    Simulate intranasal vaccine/antigen delivery PK.

    Compartments:
      - A_nasal (mcg): antigen in nasal mucosa depot
      - C_mucosal (mcg/mL): concentration in mucosal immune compartment
      - C_systemic (mcg/mL): concentration in systemic compartment

    ODEs (Forward Euler):
      dA_nasal/dt    = -(k_clear + k_uptake) * A_nasal
      dC_mucosal/dt  = k_uptake * A_nasal / V_mucosal - k_deg_m * C_mucosal
      dC_systemic/dt = k_spill * C_mucosal * V_mucosal / V_sys - k_deg_s * C_systemic

    Fixed parameters:
      V_nasal   = 1 mL   (for concentration reporting of nasal depot)
      V_mucosal = 0.005 L = 5 mL
      V_sys     = 5.0 L
      k_deg_m   = 0.1 /h
      k_spill   = 0.02 /h
      k_deg_s   = 0.05 /h

    Parameters
    ----------
    drug_name : str
        Name of vaccine/antigen.
    dose_mcg : float
        Initial dose deposited in nasal mucosa (mcg), must be > 0.
    k_clear : float
        Mucociliary clearance rate (1/h), default 2.0.
    k_uptake : float
        M-cell antigen uptake rate (1/h), default 0.5.
    t_end_h : float
        Simulation end time (h), default 48 h.
    dt_h : float
        Forward Euler step size (h), default 0.1 h.

    Returns
    -------
    IntranasalVaccinePKResult
    """
    if dose_mcg <= 0:
        raise ValueError(f"dose_mcg must be > 0, got {dose_mcg}")
    if k_clear <= 0:
        raise ValueError(f"k_clear must be > 0, got {k_clear}")
    if k_uptake <= 0:
        raise ValueError(f"k_uptake must be > 0, got {k_uptake}")

    # Fixed physiological parameters
    V_nasal_mL = 1.0  # mL (for nasal concentration reporting)
    V_mucosal_L = 0.005  # L  (5 mL)
    V_mucosal_mL = V_mucosal_L * 1000.0  # 5 mL
    V_sys_L = 5.0  # L
    k_deg_m = 0.1  # /h  mucosal degradation
    k_spill = 0.02  # /h  systemic spillover
    k_deg_s = 0.05  # /h  systemic degradation

    # Initial conditions
    A_nasal = dose_mcg  # mcg
    C_mucosal = 0.0  # mcg/mL
    C_systemic = 0.0  # mcg/mL

    times_h = []
    c_nasal_list = []
    c_mucosal_list = []
    c_systemic_list = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h)) + 1

    for _ in range(n_steps):
        # Record state
        times_h.append(t)
        c_nasal_list.append(A_nasal / V_nasal_mL)
        c_mucosal_list.append(C_mucosal)
        c_systemic_list.append(C_systemic)

        # Forward Euler derivatives
        dA_nasal = -(k_clear + k_uptake) * A_nasal
        dC_mucosal = (k_uptake * A_nasal / V_mucosal_mL) - k_deg_m * C_mucosal
        dC_systemic = (
            k_spill * C_mucosal * V_mucosal_mL / (V_sys_L * 1000.0)
        ) - k_deg_s * C_systemic

        # Update
        A_nasal += dA_nasal * dt_h
        C_mucosal += dC_mucosal * dt_h
        C_systemic += dC_systemic * dt_h

        # Clamp negatives (numerical safety)
        if A_nasal < 0.0:
            A_nasal = 0.0
        if C_mucosal < 0.0:
            C_mucosal = 0.0
        if C_systemic < 0.0:
            C_systemic = 0.0

        t = round(t + dt_h, 10)

    # Cmax and Tmax
    cmax_nasal = max(c_nasal_list)
    cmax_mucosal = max(c_mucosal_list)
    cmax_systemic = max(c_systemic_list)

    tmax_nasal_h = times_h[c_nasal_list.index(cmax_nasal)]
    tmax_mucosal_h = times_h[c_mucosal_list.index(cmax_mucosal)]
    tmax_systemic_h = times_h[c_systemic_list.index(cmax_systemic)]

    # Trapezoidal AUC
    def trap_auc(x, y):
        return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))

    auc_mucosal = trap_auc(times_h, c_mucosal_list)
    auc_systemic = trap_auc(times_h, c_systemic_list)

    mucosal_to_systemic_ratio = auc_mucosal / auc_systemic if auc_systemic > 0.0 else 0.0

    # Nasal retention at end: fraction of dose still in nasal depot
    nasal_retention_pct = (A_nasal / dose_mcg) * 100.0

    notes = (
        f"k_clear={k_clear}/h, k_uptake={k_uptake}/h, "
        f"AUC_mucosal={auc_mucosal:.4f}, AUC_systemic={auc_systemic:.6f}"
    )

    result = IntranasalVaccinePKResult(
        drug_name=drug_name,
        dose_mcg=dose_mcg,
        times_h=times_h,
        c_nasal_mcg_mL=c_nasal_list,
        c_mucosal_mcg_mL=c_mucosal_list,
        c_systemic_mcg_mL=c_systemic_list,
        cmax_nasal=cmax_nasal,
        cmax_mucosal=cmax_mucosal,
        cmax_systemic=cmax_systemic,
        tmax_nasal_h=tmax_nasal_h,
        tmax_mucosal_h=tmax_mucosal_h,
        tmax_systemic_h=tmax_systemic_h,
        auc_mucosal=auc_mucosal,
        auc_systemic=auc_systemic,
        mucosal_to_systemic_ratio=mucosal_to_systemic_ratio,
        nasal_retention_pct=nasal_retention_pct,
        notes=notes,
    )
    return result


def compare_dose_levels(
    drug_name: str,
    doses_mcg: list,
) -> list:
    """
    Simulate intranasal vaccine PK for multiple dose levels.

    Parameters
    ----------
    drug_name : str
        Drug/vaccine name.
    doses_mcg : list
        List of doses (mcg) to simulate.

    Returns
    -------
    list of IntranasalVaccinePKResult, sorted by auc_mucosal descending.
    """
    results = [simulate_intranasal_vaccine_pk(drug_name=drug_name, dose_mcg=d) for d in doses_mcg]
    return sorted(results, key=lambda r: r.auc_mucosal, reverse=True)
