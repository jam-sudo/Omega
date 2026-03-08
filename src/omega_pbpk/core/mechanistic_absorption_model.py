"""
Phase 399 — Mechanistic Absorption Model (MAM)

Simplified mechanistic oral absorption model integrating:
- Noyes-Whitney dissolution (shrinking-sphere approximation)
- Passive transcellular permeability
- Small intestinal transit
- 1-compartment systemic PK

Pure Python stdlib only (math, dataclasses).
"""

import math
from dataclasses import dataclass


@dataclass
class MAMResult:
    """Results from a Mechanistic Absorption Model simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    m_undissolved_mg: list  # undissolved drug in GI lumen (mg)
    m_dissolved_mg: list  # dissolved drug in GI lumen (mg)
    c_plasma_mg_L: list  # plasma concentration (mg/L)
    fa_predicted: float  # fraction absorbed (0-1)
    fa_dissolution_limited: bool
    fa_permeability_limited: bool
    auc_mg_h_per_L: float
    cmax_mg_L: float
    tmax_h: float
    dose_number: float  # Do = dose / (solubility * 250 mL)
    absorption_number: float  # An = Peff * SA / (R * Q_transit)
    dissolution_number: float  # Dn = t_transit / t_dissolution
    notes: str


def simulate_mam(
    drug_name: str,
    dose_mg: float,
    solubility_mg_mL: float,
    permeability_cm_s: float,
    particle_size_um: float = 50.0,
    drug_density_g_cm3: float = 1.2,
    cl_sys_L_per_h: float = 5.0,
    vd_sys_L: float = 50.0,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> MAMResult:
    """
    Simulate mechanistic oral absorption using a 3-compartment Forward Euler model.

    Compartments:
        M_undiss  -- undissolved drug in GI lumen (mg)
        M_diss    -- dissolved drug in GI lumen (mg)
        C_plasma  -- plasma drug concentration (mg/L)

    Parameters
    ----------
    drug_name : str
    dose_mg : float
    solubility_mg_mL : float
        Aqueous solubility (mg/mL).
    permeability_cm_s : float
        Effective intestinal permeability (cm/s).
    particle_size_um : float
        Mean particle diameter (micrometres). Default 50.
    drug_density_g_cm3 : float
        Particle density (g/cm3). Default 1.2.
    cl_sys_L_per_h : float
        Systemic clearance (L/h). Default 5.0.
    vd_sys_L : float
        Volume of distribution (L). Default 50.0.
    t_end_h : float
        Simulation end time (h). Default 12.
    dt_h : float
        Integration step size (h). Default 0.05.

    Returns
    -------
    MAMResult
    """
    # --- validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if solubility_mg_mL <= 0:
        raise ValueError("solubility_mg_mL must be positive")
    if permeability_cm_s <= 0:
        raise ValueError("permeability_cm_s must be positive")
    if particle_size_um <= 0:
        raise ValueError("particle_size_um must be positive")
    if drug_density_g_cm3 <= 0:
        raise ValueError("drug_density_g_cm3 must be positive")

    # --- physiological constants ---
    t_transit_h = 3.5  # intestinal transit time (h)
    SA_intestine = 100_000.0  # effective absorptive area (cm2)
    R_intestine = 1.75  # intestinal radius (cm)
    V_intestine = 400.0  # luminal volume (mL)
    Q_transit = V_intestine / t_transit_h  # mL/h

    # --- dimensionless BCS numbers ---
    Do = dose_mg / (solubility_mg_mL * 250.0)  # dose number
    An = (permeability_cm_s * 3600.0 * SA_intestine) / (
        R_intestine * Q_transit
    )  # absorption number

    # --- dissolution parameters (Noyes-Whitney) ---
    d_cm = particle_size_um * 1.0e-4  # particle diameter (cm)
    rho_mg_cm3 = drug_density_g_cm3 * 1000.0  # density (mg/cm3)
    D_eff = 7.4e-6 * 3600.0  # diffusivity cm2/s -> cm2/h
    h_diff = min(30.0e-4, d_cm / 2.0)  # diffusion layer thickness (cm)

    # Dissolution time estimate: t_diss = rho * d^2 / (6 * D * Cs / h)
    t_dissolution_h = (d_cm**2 * rho_mg_cm3) / (6.0 * D_eff * solubility_mg_mL / h_diff)
    t_dissolution_h = max(t_dissolution_h, 1.0e-9)  # guard against zero

    Dn = t_transit_h / t_dissolution_h  # dissolution number

    # --- rate constants ---
    # Dissolution rate constant (1/h) from Noyes-Whitney
    k_diss_raw = 6.0 * D_eff * solubility_mg_mL / (h_diff * d_cm * rho_mg_cm3)
    k_diss = min(k_diss_raw, 5.0)

    # Absorption rate constant (1/h) from permeability
    ka_raw = permeability_cm_s * 3600.0 * SA_intestine / V_intestine
    ka = min(ka_raw, 20.0)

    # Transit loss rate (1/h)
    k_transit = 1.0 / t_transit_h

    # Systemic elimination rate (1/h)
    ke_sys = cl_sys_L_per_h / vd_sys_L

    # Max dissolved drug (solubility * luminal volume)
    max_diss_mg = solubility_mg_mL * V_intestine

    # --- initial conditions ---
    M_undiss = dose_mg
    M_diss = 0.0
    C_plasma = 0.0

    # --- storage ---
    n_steps = int(math.ceil(t_end_h / dt_h)) + 1
    times_h: list[float] = []
    m_undissolved: list[float] = []
    m_dissolved: list[float] = []
    c_plasma_list: list[float] = []

    t = 0.0
    for _ in range(n_steps):
        times_h.append(t)
        m_undissolved.append(M_undiss)
        m_dissolved.append(M_diss)
        c_plasma_list.append(C_plasma)

        if t >= t_end_h:
            break

        actual_dt = min(dt_h, t_end_h - t)

        # --- dissolution flux ---
        if M_diss >= max_diss_mg:
            dissolve_flux = 0.0
        else:
            dissolve_flux = k_diss * M_undiss  # mg/h

        # --- ODE derivatives ---
        dM_undiss_dt = -dissolve_flux - k_transit * M_undiss
        dM_diss_dt = dissolve_flux - ka * M_diss - k_transit * M_diss
        dC_plasma_dt = ka * M_diss / vd_sys_L - ke_sys * C_plasma

        # --- forward Euler ---
        M_undiss_new = M_undiss + dM_undiss_dt * actual_dt
        M_diss_new = M_diss + dM_diss_dt * actual_dt
        C_plasma_new = C_plasma + dC_plasma_dt * actual_dt

        # non-negativity clamp
        M_undiss = max(0.0, M_undiss_new)
        M_diss = max(0.0, M_diss_new)
        # cap dissolved at solubility limit
        M_diss = min(M_diss, max_diss_mg)
        C_plasma = max(0.0, C_plasma_new)

        t += actual_dt

    # --- derived PK metrics ---
    # AUC by manual trapezoidal rule
    auc = sum(
        0.5 * (c_plasma_list[i] + c_plasma_list[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    cmax = max(c_plasma_list)
    tmax = times_h[c_plasma_list.index(cmax)]

    # Fraction absorbed from mass balance: fa = AUC * CL / dose
    fa_predicted = min(1.0, auc * cl_sys_L_per_h / dose_mg)
    fa_predicted = max(0.0, fa_predicted)

    fa_dissolution_limited = Do > 1.0 and Dn < 1.0
    fa_permeability_limited = An < 1.0

    notes_parts = []
    if fa_dissolution_limited:
        notes_parts.append("Dissolution-limited absorption (Do>1, Dn<1).")
    if fa_permeability_limited:
        notes_parts.append("Permeability-limited absorption (An<1).")
    if not notes_parts:
        notes_parts.append("Well-absorbed drug (dissolution and permeability adequate).")
    notes = " ".join(notes_parts)

    return MAMResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        m_undissolved_mg=m_undissolved,
        m_dissolved_mg=m_dissolved,
        c_plasma_mg_L=c_plasma_list,
        fa_predicted=fa_predicted,
        fa_dissolution_limited=fa_dissolution_limited,
        fa_permeability_limited=fa_permeability_limited,
        auc_mg_h_per_L=auc,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        dose_number=Do,
        absorption_number=An,
        dissolution_number=Dn,
        notes=notes,
    )


def sensitivity_analysis_mam(
    drug_name: str,
    dose_mg: float,
    permeability_cm_s: float,
    solubilities: list = None,
) -> list:
    """
    Run MAM simulations across a range of solubilities and return results
    sorted by fa_predicted (descending).

    Parameters
    ----------
    drug_name : str
    dose_mg : float
    permeability_cm_s : float
    solubilities : list of float, optional
        Solubility values (mg/mL). Default [0.001, 0.01, 0.1, 1.0, 10.0].

    Returns
    -------
    list of MAMResult, sorted by fa_predicted descending
    """
    if solubilities is None:
        solubilities = [0.001, 0.01, 0.1, 1.0, 10.0]

    results = []
    for sol in solubilities:
        r = simulate_mam(
            drug_name=drug_name,
            dose_mg=dose_mg,
            solubility_mg_mL=sol,
            permeability_cm_s=permeability_cm_s,
        )
        results.append(r)

    results.sort(key=lambda x: x.fa_predicted, reverse=True)
    return results
