"""
Phase 415 — Intranasal drug delivery PK model.

3-compartment model: nasal mucosa -> systemic -> (optional CNS)
- Nasal absorption: first-order ka_nasal (bypasses first-pass)
- Mucociliary clearance: k_clearance from nasal compartment
- Systemic 1-cpt: CL + Vd
"""

from dataclasses import dataclass


@dataclass
class IntranasalPKResult:
    """Result of intranasal PK simulation."""

    drug_name: str
    dose_mg: float
    route: str
    times_h: list
    c_nasal_mg_mL: list
    c_systemic_mg_L: list
    cmax_systemic_mg_L: float
    tmax_systemic_h: float
    auc_systemic_mg_h_per_L: float
    bioavailability_pct: float
    t_half_systemic_h: float
    notes: str


def simulate_intranasal_pk(
    drug_name,
    dose_mg,
    ka_nasal_per_h,
    k_clearance_per_h,
    cl_sys_L_per_h,
    vd_sys_L,
    t_end_h=12.0,
    dt_h=0.05,
    v_nasal_mL=0.4,
):
    """
    Simulate intranasal PK using a 3-compartment forward Euler model.

    Parameters
    ----------
    drug_name : str
    dose_mg : float  -- dose in mg applied to nasal mucosa
    ka_nasal_per_h : float  -- first-order nasal absorption rate constant (h^-1)
    k_clearance_per_h : float  -- mucociliary clearance rate constant (h^-1), >= 0
    cl_sys_L_per_h : float  -- systemic clearance (L/h)
    vd_sys_L : float  -- volume of distribution (L)
    t_end_h : float  -- simulation end time (h)
    dt_h : float  -- time step for forward Euler (h)
    v_nasal_mL : float  -- nasal mucosa volume (mL) for concentration display

    Returns
    -------
    IntranasalPKResult
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if ka_nasal_per_h <= 0:
        raise ValueError(f"ka_nasal_per_h must be > 0, got {ka_nasal_per_h}")
    if k_clearance_per_h < 0:
        raise ValueError(f"k_clearance_per_h must be >= 0, got {k_clearance_per_h}")
    if cl_sys_L_per_h <= 0:
        raise ValueError(f"cl_sys_L_per_h must be > 0, got {cl_sys_L_per_h}")
    if vd_sys_L <= 0:
        raise ValueError(f"vd_sys_L must be > 0, got {vd_sys_L}")

    n_steps = int(round(t_end_h / dt_h)) + 1

    times_h = []
    c_nasal_list = []
    c_sys_list = []

    # State: M_nasal (mg mass in nasal compartment), C_sys (mg/L systemic)
    M_nasal = float(dose_mg)
    C_sys = 0.0

    k_total_nasal = ka_nasal_per_h + k_clearance_per_h
    k_elim_sys = cl_sys_L_per_h / vd_sys_L

    for i in range(n_steps):
        t = i * dt_h
        times_h.append(t)
        c_nasal_list.append(M_nasal / v_nasal_mL)  # mg/mL
        c_sys_list.append(C_sys)

        dM_nasal_dt = -k_total_nasal * M_nasal
        dC_sys_dt = (ka_nasal_per_h * M_nasal) / vd_sys_L - k_elim_sys * C_sys

        M_nasal = M_nasal + dt_h * dM_nasal_dt
        if M_nasal < 0.0:
            M_nasal = 0.0
        C_sys = C_sys + dt_h * dC_sys_dt
        if C_sys < 0.0:
            C_sys = 0.0

    cmax_systemic_mg_L = max(c_sys_list)
    tmax_idx = c_sys_list.index(cmax_systemic_mg_L)
    tmax_systemic_h = times_h[tmax_idx]

    auc_systemic_mg_h_per_L = sum(
        0.5 * (c_sys_list[i] + c_sys_list[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    bioavailability_pct = (ka_nasal_per_h / k_total_nasal) * 100.0
    t_half_systemic_h = 0.693 * vd_sys_L / cl_sys_L_per_h

    notes = (
        f"Bioavailability ~{bioavailability_pct:.1f}% "
        f"(mucociliary clearance removed {100.0 - bioavailability_pct:.1f}%). "
        f"Systemic t1/2 = {t_half_systemic_h:.2f} h."
    )

    return IntranasalPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route="intranasal",
        times_h=times_h,
        c_nasal_mg_mL=c_nasal_list,
        c_systemic_mg_L=c_sys_list,
        cmax_systemic_mg_L=cmax_systemic_mg_L,
        tmax_systemic_h=tmax_systemic_h,
        auc_systemic_mg_h_per_L=auc_systemic_mg_h_per_L,
        bioavailability_pct=bioavailability_pct,
        t_half_systemic_h=t_half_systemic_h,
        notes=notes,
    )


def compare_intranasal_clearance(
    drug_name,
    dose_mg,
    ka_nasal_per_h,
    clearance_rates,
    cl_sys_L_per_h,
    vd_sys_L,
):
    """
    Compare intranasal PK across different mucociliary clearance rates.

    Parameters
    ----------
    drug_name : str
    dose_mg : float
    ka_nasal_per_h : float
    clearance_rates : list of float -- mucociliary clearance rates (h^-1)
    cl_sys_L_per_h : float
    vd_sys_L : float

    Returns
    -------
    list of IntranasalPKResult sorted by cmax_systemic_mg_L descending
    """
    results = []
    for kc in clearance_rates:
        result = simulate_intranasal_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            ka_nasal_per_h=ka_nasal_per_h,
            k_clearance_per_h=kc,
            cl_sys_L_per_h=cl_sys_L_per_h,
            vd_sys_L=vd_sys_L,
        )
        results.append(result)

    results.sort(key=lambda r: r.cmax_systemic_mg_L, reverse=True)
    return results
