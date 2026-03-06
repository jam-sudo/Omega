"""Lung surfactant interaction model — Phase 231.

Models drug interaction with pulmonary surfactant (DPPC/lipid monolayer).
Relevant for inhaled drugs — surfactant binding affects free drug concentration
available for absorption from alveolar lining fluid (ALF).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class SurfactantPKResult:
    """Result of inhaled drug surfactant interaction PK simulation."""

    drug_name: str
    dose_mg: float
    logP: float
    mw: float
    kp_surf: float
    fu_surf: float
    times_h: list[float]
    m_aq_mg: list[float]
    c_plasma_mg_L: list[float]
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    f_absorbed: float
    effective_ka_per_h: float
    notes: list[str] = field(default_factory=list)


def surfactant_partition(logP: float, mw: float) -> float:
    """Compute surfactant partition coefficient Kp_surf.

    Empirical model: Kp_surf = 10^(0.8*logP - 0.003*mw + 0.5)
    Lipophilic (high logP) drugs bind more strongly to DPPC surfactant.

    Parameters
    ----------
    logP : float
        Octanol-water partition coefficient (log units).
    mw : float
        Molecular weight (g/mol).

    Returns
    -------
    float
        Surfactant partition coefficient (dimensionless, C_surf / C_aq).
    """
    exponent = 0.8 * logP - 0.003 * mw + 0.5
    return 10.0**exponent


def free_fraction_in_alf(
    kp_surf: float,
    v_surf_mL: float = 0.5,
    v_aq_mL: float = 10.0,
) -> float:
    """Compute free fraction of drug in alveolar lining fluid (ALF).

    fu = 1 / (1 + Kp_surf * V_surf / V_aq)

    Parameters
    ----------
    kp_surf : float
        Surfactant partition coefficient (Kp_surf >= 0).
    v_surf_mL : float
        Volume of surfactant phase in ALF (mL). Default 0.5 mL.
    v_aq_mL : float
        Volume of aqueous phase in ALF (mL). Default 10.0 mL.

    Returns
    -------
    float
        Free fraction in ALF (0, 1].
    """
    if v_aq_mL <= 0:
        raise ValueError("v_aq_mL must be > 0")
    if kp_surf < 0:
        raise ValueError("kp_surf must be >= 0")
    return 1.0 / (1.0 + kp_surf * v_surf_mL / v_aq_mL)


def simulate_inhaled_surfactant_pk(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw: float,
    ka_aq_per_h: float = 2.0,
    cl_L_per_h: float = 10.0,
    vd_L: float = 50.0,
    v_surf_mL: float = 0.5,
    v_aq_mL: float = 10.0,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> SurfactantPKResult:
    """Simulate inhaled drug PK with surfactant partitioning.

    Drug deposits in lungs and partitions between:
    - Surfactant phase (bound, not absorbed)
    - Aqueous lining fluid (free, available for absorption)

    Only free drug is absorbed (first-order from aqueous compartment).
    Systemic PK: 1-compartment model with CL/Vd.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Inhaled dose (mg). Must be > 0.
    logP : float
        Octanol-water partition coefficient.
    mw : float
        Molecular weight (g/mol).
    ka_aq_per_h : float
        Aqueous-phase absorption rate constant (1/h). Must be > 0.
    cl_L_per_h : float
        Systemic clearance (L/h). Must be > 0.
    vd_L : float
        Volume of distribution (L). Must be > 0.
    v_surf_mL : float
        Surfactant volume (mL).
    v_aq_mL : float
        Aqueous lining fluid volume (mL).
    t_end_h : float
        Simulation duration (h).
    dt_h : float
        Time step (h).

    Returns
    -------
    SurfactantPKResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if ka_aq_per_h <= 0:
        raise ValueError("ka_aq_per_h must be > 0")

    kp_surf = surfactant_partition(logP, mw)
    fu = free_fraction_in_alf(kp_surf, v_surf_mL, v_aq_mL)
    eff_ka = ka_aq_per_h * fu

    # Initial lung deposit split between aqueous and surfactant
    # At t=0, drug equilibrates instantly:
    # m_aq / (m_surf / (v_surf_mL/v_aq_mL * kp_surf)) = 1
    # Fraction in aqueous: 1 / (1 + kp_surf * v_surf_mL / v_aq_mL) = fu_surf
    m_aq = dose_mg * fu
    m_surf = dose_mg * (1.0 - fu)
    c_sys = 0.0  # mg/L systemic

    n_steps = int(math.ceil(t_end_h / dt_h))
    times: list[float] = [0.0]
    m_aq_list: list[float] = [m_aq]
    c_sys_list: list[float] = [c_sys]

    for _ in range(n_steps):
        # Surfactant and aqueous compartments stay in equilibrium:
        # drug absorbed from aqueous; surfactant re-equilibrates
        # Combined lung mass: m_lung = m_aq + m_surf
        m_lung = m_aq + m_surf

        # Effective rate on total lung mass:
        # dm_lung/dt = -eff_ka * m_lung
        # (because fu is the instantaneous free fraction)
        dm_lung = -eff_ka * m_lung * dt_h
        m_lung_new = max(0.0, m_lung + dm_lung)

        absorbed_mg = max(0.0, m_lung - m_lung_new)

        # Re-partition new lung mass
        m_aq = m_lung_new * fu
        m_surf = m_lung_new * (1.0 - fu)

        # Systemic 1-cpt
        dc_sys = (absorbed_mg / vd_L - c_sys * cl_L_per_h / vd_L * dt_h) / dt_h
        c_sys = max(0.0, c_sys + dc_sys * dt_h)

        t_new = times[-1] + dt_h
        times.append(t_new)
        m_aq_list.append(m_aq)
        c_sys_list.append(c_sys)

    times_arr = np.array(times)
    c_sys_arr = np.array(c_sys_list)

    cmax = float(np.max(c_sys_arr))
    tmax_idx = int(np.argmax(c_sys_arr))
    tmax = float(times_arr[tmax_idx])
    auc = float(np_trapz(c_sys_arr, times_arr))

    # Fraction absorbed = drug that reached systemic = total absorbed / dose
    # Mass in systemic at any time = Vd * C; total eliminated + remaining
    # Simpler: 1 - remaining lung mass / dose
    m_lung_final = m_aq + m_surf
    f_absorbed = max(0.0, min(1.0, (dose_mg - m_lung_final) / dose_mg))

    notes: list[str] = []
    notes.append(f"Kp_surf={kp_surf:.3f}, fu_surf={fu:.4f}")
    notes.append(f"Effective ka = {eff_ka:.4f} /h (= {ka_aq_per_h} * {fu:.4f})")
    if logP > 3:
        notes.append("High logP: significant surfactant trapping expected")

    return SurfactantPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logP=logP,
        mw=mw,
        kp_surf=kp_surf,
        fu_surf=fu,
        times_h=times,
        m_aq_mg=m_aq_list,
        c_plasma_mg_L=c_sys_list,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        f_absorbed=f_absorbed,
        effective_ka_per_h=eff_ka,
        notes=notes,
    )


def compare_logP_effect(
    drug_name: str,
    dose_mg: float,
    mw: float,
    logP_values: list[float],
    **kwargs: float,
) -> list[SurfactantPKResult]:
    """Compare inhaled surfactant PK across different logP values.

    Simulates the drug at each logP and returns results sorted by
    f_absorbed in descending order (highest absorption first).

    Parameters
    ----------
    drug_name : str
        Base drug name (logP value appended to each result).
    dose_mg : float
        Inhaled dose (mg).
    mw : float
        Molecular weight (g/mol).
    logP_values : list[float]
        List of logP values to evaluate.
    **kwargs : float
        Additional keyword arguments passed to simulate_inhaled_surfactant_pk.

    Returns
    -------
    list[SurfactantPKResult]
        Results sorted by f_absorbed descending.
    """
    results: list[SurfactantPKResult] = []
    for logP in logP_values:
        name = f"{drug_name}_logP{logP:.1f}"
        result = simulate_inhaled_surfactant_pk(
            drug_name=name,
            dose_mg=dose_mg,
            logP=logP,
            mw=mw,
            **kwargs,
        )
        results.append(result)

    results.sort(key=lambda r: r.f_absorbed, reverse=True)
    return results
