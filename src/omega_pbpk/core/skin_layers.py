"""Drug permeation through discrete skin layers — Phase 232.

Extended skin permeation model with three sequential layers:
- Stratum corneum (SC): topical application site, lipophilic barrier
- Viable epidermis (VE): metabolically active layer (CYP1A2)
- Dermis: vascular layer connecting to systemic circulation

Drug must permeate sequentially through SC → VE → Dermis → Systemic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class SkinLayerResult:
    """Result of multi-layer skin permeation simulation."""

    drug_name: str
    dose_mg_cm2: float
    application_area_cm2: float
    logP: float
    mw: float
    p_sc_cm_h: float
    lag_time_h: float
    times_h: list[float]
    m_sc_mg: list[float]
    m_ve_mg: list[float]
    m_dermis_mg: list[float]
    c_plasma_mg_L: list[float]
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    f_absorbed_systemic: float
    f_metabolized_skin: float
    notes: list[str] = field(default_factory=list)


def potts_guy_psc(logP: float, mw: float) -> float:
    """Compute stratum corneum permeability using Potts-Guy equation.

    log(P_sc) = -6.3 + 0.71*logP - 0.0061*MW

    Parameters
    ----------
    logP : float
        Octanol-water partition coefficient.
    mw : float
        Molecular weight (g/mol).

    Returns
    -------
    float
        Stratum corneum permeability (cm/h).
    """
    log_psc = -6.3 + 0.71 * logP - 0.0061 * mw
    return 10.0**log_psc


def lag_time(p_sc_cm_h: float, h_sc_cm: float = 0.001) -> float:
    """Compute lag time for drug diffusion through stratum corneum.

    Simplified: lag = h_sc^2 / (6 * P_sc)

    Parameters
    ----------
    p_sc_cm_h : float
        SC permeability (cm/h). Must be > 0.
    h_sc_cm : float
        SC thickness (cm). Default 0.001 cm (10 µm).

    Returns
    -------
    float
        Lag time (h).
    """
    if p_sc_cm_h <= 0:
        raise ValueError("p_sc_cm_h must be > 0")
    return (h_sc_cm**2) / (6.0 * p_sc_cm_h)


def simulate_skin_layers(
    drug_name: str,
    dose_mg_cm2: float,
    application_area_cm2: float,
    logP: float,
    mw: float,
    cl_systemic_L_per_h: float,
    vd_L: float,
    cl_skin_per_h: float = 0.01,
    q_dermal_L_per_h: float = 0.05,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> SkinLayerResult:
    """Simulate drug permeation through skin layers (SC → VE → Dermis → Systemic).

    3-layer forward Euler simulation:
    - SC: drug applied topically; permeates via Potts-Guy P_sc
    - VE: receives from SC; metabolized by CYP1A2 (cl_skin_per_h); passes to dermis
    - Dermis: receives from VE; enters systemic via dermal blood flow
    - Systemic: 1-compartment with CL/Vd

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg_cm2 : float
        Applied dose per unit area (mg/cm²). Must be > 0.
    application_area_cm2 : float
        Skin application area (cm²). Must be > 0.
    logP : float
        Octanol-water partition coefficient.
    mw : float
        Molecular weight (g/mol).
    cl_systemic_L_per_h : float
        Systemic clearance (L/h). Must be > 0.
    vd_L : float
        Volume of distribution (L). Must be > 0.
    cl_skin_per_h : float
        First-order skin metabolism rate constant (1/h). Default 0.01.
    q_dermal_L_per_h : float
        Dermal blood flow (L/h). Default 0.05 L/h.
    t_end_h : float
        Simulation duration (h). Default 24 h.
    dt_h : float
        Time step (h). Default 0.1 h.

    Returns
    -------
    SkinLayerResult
    """
    if dose_mg_cm2 <= 0:
        raise ValueError("dose_mg_cm2 must be > 0")
    if application_area_cm2 <= 0:
        raise ValueError("application_area_cm2 must be > 0")
    if cl_systemic_L_per_h <= 0:
        raise ValueError("cl_systemic_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")

    # Permeability and lag time
    p_sc = potts_guy_psc(logP, mw)
    p_ve = 100.0 * p_sc  # viable epidermis is ~100x more permeable than SC
    lag_t = lag_time(p_sc)

    # Convert permeability to first-order rate constants
    # Flux = P_sc * area * (C_sc - C_ve) where C = m / V_layer
    # Layer volumes (empirical, proportional to area)
    # SC: ~10 µm thick = 0.001 cm; VE: ~100 µm = 0.01 cm; Dermis: ~2 mm = 0.2 cm
    h_sc = 0.001  # cm
    h_ve = 0.01  # cm
    h_dermis = 0.2  # cm

    v_sc_L = h_sc * application_area_cm2 * 1e-3  # cm³ → L
    v_ve_L = h_ve * application_area_cm2 * 1e-3
    v_dermis_L = h_dermis * application_area_cm2 * 1e-3

    # Transfer rates (1/h) from layer i to layer j
    # k_sc_ve = P_sc * area / V_sc (permeation out of SC)
    _k_sc_ve = p_sc * application_area_cm2 / (v_sc_L * 1000.0)  # unused
    # Actually use mass-based: dm/dt = P * A * (C1 - C2) * (1/cm_to_L_factor)
    # C = m/V_L → m = C * V_L; dm_sc/dt = -P_sc * A * (m_sc/v_sc - m_ve/v_ve) * conv
    # conv: P [cm/h] * A [cm²] → cm³/h = mL/h; divide by V [mL] to get 1/h
    v_sc_mL = v_sc_L * 1000.0
    v_ve_mL = v_ve_L * 1000.0
    v_dermis_mL = v_dermis_L * 1000.0

    # Transfer conductance (mL/h)
    cond_sc_ve = p_sc * application_area_cm2  # cm/h * cm² = cm³/h = mL/h
    cond_ve_dermis = p_ve * application_area_cm2  # mL/h

    # Systemic: c_sys in mg/L, elimination: k_el = CL/Vd
    k_el = cl_systemic_L_per_h / vd_L

    # Dermal transfer to systemic: q_dermal removes drug from dermis compartment
    # Rate = q_dermal_L_per_h * c_dermis_mg_L = q * m_dermis / v_dermis_L
    q_to_sys_rate = q_dermal_L_per_h  # L/h, applied to conc in dermis

    # Initial conditions
    total_dose_mg = dose_mg_cm2 * application_area_cm2
    m_sc = total_dose_mg  # all drug starts in SC
    m_ve = 0.0
    m_dermis = 0.0
    c_sys = 0.0  # mg/L

    n_steps = int(math.ceil(t_end_h / dt_h))

    times: list[float] = [0.0]
    m_sc_list: list[float] = [m_sc]
    m_ve_list: list[float] = [m_ve]
    m_dermis_list: list[float] = [m_dermis]
    c_sys_list: list[float] = [c_sys]

    m_metabolized = 0.0  # cumulative skin metabolized mass

    for _ in range(n_steps):
        # Concentrations in each layer
        c_sc = m_sc / v_sc_mL  # mg/mL
        c_ve = m_ve / v_ve_mL
        c_dermis = m_dermis / v_dermis_mL

        # Fluxes (mg/h)
        flux_sc_to_ve = cond_sc_ve * (c_sc - c_ve)  # mL/h * mg/mL = mg/h
        flux_ve_to_dermis = cond_ve_dermis * (c_ve - c_dermis)
        # Skin metabolism in VE (CYP1A2)
        metab_ve = cl_skin_per_h * m_ve
        # Dermal → systemic transfer
        flux_dermis_to_sys = q_to_sys_rate * c_dermis  # L/h * mg/mL * 1000 = mg/h
        # Wait — c_dermis is mg/mL and q is L/h → flux = q[L/h] * c[mg/L]
        # so convert: c_dermis_mg_L = c_dermis * 1000 mg/L
        flux_dermis_to_sys = q_to_sys_rate * (c_dermis * 1000.0)  # mg/h

        # ODE forward Euler
        dm_sc = -flux_sc_to_ve
        dm_ve = flux_sc_to_ve - flux_ve_to_dermis - metab_ve
        dm_dermis = flux_ve_to_dermis - flux_dermis_to_sys

        # Systemic (c in mg/L)
        # d(Vd*c_sys)/dt = flux_dermis_to_sys / 1000 [L/h] - CL*c_sys [mg/h]
        # Wait: flux_dermis_to_sys is mg/h; Vd in L; c_sys in mg/L
        dc_sys = (flux_dermis_to_sys / vd_L) - k_el * c_sys  # mg/L/h... wait
        # flux / vd gives mg/h / L = mg/(L*h), and k_el*c_sys is 1/h * mg/L = mg/(L*h). Correct.
        # But flux_dermis_to_sys = q[L/h] * c_dermis[mg/L] = mg/h ✓

        m_sc = max(0.0, m_sc + dm_sc * dt_h)
        m_ve = max(0.0, m_ve + dm_ve * dt_h)
        m_dermis = max(0.0, m_dermis + dm_dermis * dt_h)
        c_sys = max(0.0, c_sys + dc_sys * dt_h)
        m_metabolized += metab_ve * dt_h

        times.append(times[-1] + dt_h)
        m_sc_list.append(m_sc)
        m_ve_list.append(m_ve)
        m_dermis_list.append(m_dermis)
        c_sys_list.append(c_sys)

    times_arr = np.array(times)
    c_sys_arr = np.array(c_sys_list)

    cmax = float(np.max(c_sys_arr))
    tmax = float(times_arr[int(np.argmax(c_sys_arr))])
    auc = float(np_trapz(c_sys_arr, times_arr))

    # Mass balance
    m_remaining = m_sc + m_ve + m_dermis + c_sys * vd_L
    _m_absorbed_sys = max(0.0, total_dose_mg - m_remaining - m_metabolized)

    f_absorbed = max(0.0, min(1.0, (total_dose_mg - m_sc - m_ve - m_dermis) / total_dose_mg))
    # f_absorbed_systemic = what actually reached systemic (not skin-metabolized)
    f_metabolized = max(0.0, min(1.0, m_metabolized / total_dose_mg))
    f_absorbed_systemic = max(0.0, min(1.0, f_absorbed - f_metabolized))

    notes: list[str] = []
    notes.append(f"P_sc={p_sc:.3e} cm/h, P_ve={p_ve:.3e} cm/h")
    notes.append(f"Lag time={lag_t:.3f} h")
    notes.append(f"Total dose={total_dose_mg:.3f} mg, Area={application_area_cm2} cm²")
    if logP < 0:
        notes.append("Negative logP: poor SC penetration expected")
    elif logP > 4:
        notes.append("High logP: good SC penetration but may accumulate in SC depot")

    return SkinLayerResult(
        drug_name=drug_name,
        dose_mg_cm2=dose_mg_cm2,
        application_area_cm2=application_area_cm2,
        logP=logP,
        mw=mw,
        p_sc_cm_h=p_sc,
        lag_time_h=lag_t,
        times_h=times,
        m_sc_mg=m_sc_list,
        m_ve_mg=m_ve_list,
        m_dermis_mg=m_dermis_list,
        c_plasma_mg_L=c_sys_list,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        f_absorbed_systemic=f_absorbed_systemic,
        f_metabolized_skin=f_metabolized,
        notes=notes,
    )
