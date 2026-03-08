"""Phase 435 — Epidural Drug PK Model.

3-compartment ODE model (forward Euler):
  - Epidural space (with fat sequestration)
  - CSF
  - Systemic plasma
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EpiduralPKResult:
    """Result of an epidural PK simulation (Phase 435)."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_epidural_mg_mL: list
    c_csf_mg_mL: list
    c_systemic_mg_L: list
    cmax_epidural_mg_mL: float
    cmax_csf_mg_mL: float
    tmax_csf_h: float
    auc_epidural_mg_h_per_mL: float
    auc_csf_mg_h_per_mL: float
    auc_systemic_mg_h_per_L: float
    csf_to_epidural_ratio: float
    systemic_absorption_pct: float
    t_half_epidural_h: float
    notes: str


def _trapz(y: list, x: list) -> float:
    """Manual trapezoidal integration."""
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


def simulate_epidural_pk(
    drug_name: str,
    dose_mg: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float = 20.0,
    v_epidural_mL: float = 1.0,
    k_epi_to_csf: float = 0.3,
    k_epi_to_sys: float = 0.5,
    k_epi_fat: float = 0.2,
    k_csf_elim: float = 0.4,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> EpiduralPKResult:
    """Simulate epidural drug PK using a 3-compartment ODE model.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Epidural bolus dose in mg.
    cl_sys_L_per_h:
        Systemic clearance in L/h.
    vd_sys_L:
        Systemic volume of distribution in L (default 20 L).
    v_epidural_mL:
        Epidural space volume in mL (default 1.0 mL).
    k_epi_to_csf:
        Transfer rate epidural -> CSF in /h (default 0.3).
    k_epi_to_sys:
        Absorption rate epidural -> systemic in /h (default 0.5).
    k_epi_fat:
        Sequestration rate into epidural fat in /h (default 0.2).
    k_csf_elim:
        CSF elimination rate in /h (default 0.4).
    t_end_h:
        Simulation end time in hours (default 12 h).
    dt_h:
        Forward Euler time step in hours (default 0.05 h).

    Returns
    -------
    EpiduralPKResult with time-course profiles and summary PK metrics.
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_sys_L_per_h <= 0:
        raise ValueError(f"cl_sys_L_per_h must be > 0, got {cl_sys_L_per_h}")
    if vd_sys_L <= 0:
        raise ValueError(f"vd_sys_L must be > 0, got {vd_sys_L}")

    # CSF volume in L for unit consistency
    v_csf_L = 0.15  # L

    # Derived elimination rate constant for systemic compartment
    ke_sys = cl_sys_L_per_h / vd_sys_L  # /h

    # Total epidural loss rate
    k_epi_total = k_epi_to_csf + k_epi_to_sys + k_epi_fat

    # Initial conditions
    # M_epidural in mg (mass in epidural space)
    m_epi = dose_mg
    c_csf = 0.0  # mg/L (converted below: V_csf in L)
    c_sys = 0.0  # mg/L

    times: list = []
    epi_conc_list: list = []
    csf_list: list = []
    sys_list: list = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h))

    for step in range(n_steps + 1):
        # Epidural concentration in mg/mL
        c_epi_mL = m_epi / v_epidural_mL

        times.append(t)
        epi_conc_list.append(c_epi_mL)
        csf_list.append(c_csf)
        sys_list.append(c_sys)

        if step == n_steps:
            break

        # ODEs — forward Euler
        # dM_epidural/dt = -k_total * M_epidural  (mg/h)
        dm_epi = -k_epi_total * m_epi

        # dC_csf/dt = (k_epi_to_csf * M_epidural) / V_csf - k_csf_elim * C_csf
        # Units: (/h * mg) / L = mg/(L*h)  => C_csf in mg/L
        dc_csf = (k_epi_to_csf * m_epi) / v_csf_L - k_csf_elim * c_csf

        # dC_sys/dt = (k_epi_to_sys * M_epidural) / Vd_sys - (CL/Vd_sys) * C_sys
        # Units: (/h * mg) / L = mg/(L*h)  => C_sys in mg/L
        dc_sys = (k_epi_to_sys * m_epi) / vd_sys_L - ke_sys * c_sys

        m_epi = max(0.0, m_epi + dm_epi * dt_h)
        c_csf = max(0.0, c_csf + dc_csf * dt_h)
        c_sys = max(0.0, c_sys + dc_sys * dt_h)

        t += dt_h

    # Convert CSF concentrations from mg/L to mg/mL for output
    # (keep systemic in mg/L as specified)
    # Note: spec says c_csf_mg_mL — convert mg/L -> mg/mL (* 0.001)
    # BUT that would give tiny values; the spec says V_csf = 0.15 L
    # Let's keep CSF in mg/mL (divide by 1000)
    csf_mg_mL_list = [v / 1000.0 for v in csf_list]

    # Summary metrics
    cmax_epidural = max(epi_conc_list)
    cmax_csf = max(csf_mg_mL_list)
    tmax_csf_h = times[csf_mg_mL_list.index(cmax_csf)]

    auc_epidural = _trapz(epi_conc_list, times)
    auc_csf = _trapz(csf_mg_mL_list, times)
    auc_systemic = _trapz(sys_list, times)  # mg*h/L

    csf_to_epidural_ratio = auc_csf / max(auc_epidural, 1e-12)

    # Systemic absorption fraction (analytical, based on rate constants)
    systemic_absorption_pct = (k_epi_to_sys / k_epi_total) * 100.0

    # Epidural half-life
    t_half_epidural_h = 0.693 / k_epi_total

    notes = (
        f"Epidural bolus {dose_mg} mg, CL_sys={cl_sys_L_per_h} L/h, "
        f"Vd={vd_sys_L} L. k_epi_to_csf={k_epi_to_csf}/h, "
        f"k_epi_to_sys={k_epi_to_sys}/h, k_epi_fat={k_epi_fat}/h. "
        f"t_half_epi={t_half_epidural_h:.3f} h. "
        f"Systemic absorption={systemic_absorption_pct:.1f}%."
    )

    return EpiduralPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_epidural_mg_mL=epi_conc_list,
        c_csf_mg_mL=csf_mg_mL_list,
        c_systemic_mg_L=sys_list,
        cmax_epidural_mg_mL=cmax_epidural,
        cmax_csf_mg_mL=cmax_csf,
        tmax_csf_h=tmax_csf_h,
        auc_epidural_mg_h_per_mL=auc_epidural,
        auc_csf_mg_h_per_mL=auc_csf,
        auc_systemic_mg_h_per_L=auc_systemic,
        csf_to_epidural_ratio=csf_to_epidural_ratio,
        systemic_absorption_pct=systemic_absorption_pct,
        t_half_epidural_h=t_half_epidural_h,
        notes=notes,
    )


def compare_epidural_drugs(drugs_list: list) -> list:
    """Simulate epidural PK for multiple drug parameter sets.

    Parameters
    ----------
    drugs_list:
        List of dicts, each with keyword arguments for simulate_epidural_pk.

    Returns
    -------
    List of EpiduralPKResult sorted by csf_to_epidural_ratio descending.
    """
    results = [simulate_epidural_pk(**params) for params in drugs_list]
    return sorted(results, key=lambda r: r.csf_to_epidural_ratio, reverse=True)
