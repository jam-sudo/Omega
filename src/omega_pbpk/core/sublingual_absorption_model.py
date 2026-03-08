"""
Phase 433 — Sublingual and buccal drug absorption model.

Simulates pH-dependent ionization across mucosal membrane with salivary loss.
Pure Python, stdlib only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SublingualAbsorptionResult:
    """Result of sublingual/buccal absorption simulation."""

    drug_name: str
    dose_mg: float
    route: str  # "sublingual" or "buccal"
    times_min: list
    m_absorbed_mg: list  # cumulative absorbed mass
    m_remaining_mg: list  # remaining in oral cavity
    c_systemic_mg_L: list  # systemic concentration
    f_absorbed_total: float  # total fraction absorbed (bypasses first-pass)
    tmax_systemic_min: float
    cmax_systemic_mg_L: float
    auc_systemic_mg_h_per_L: float
    t_half_absorption_min: float  # time for 50% absorption
    salivary_loss_pct: float  # % swallowed with saliva
    notes: str


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def simulate_sublingual_absorption(
    drug_name: str,
    dose_mg: float,
    route: str,
    drug_pka: float,
    logp: float,
    ionization_type: str,
    cl_sys_L_per_h: float,
    vd_sys_L: float = 10.0,
    t_end_min: float = 30.0,
    dt_min: float = 0.1,
) -> SublingualAbsorptionResult:
    """
    Simulate sublingual or buccal drug absorption.

    Parameters
    ----------
    drug_name : str
    dose_mg : float
    route : str  -- "sublingual" or "buccal"
    drug_pka : float
    logp : float
    ionization_type : str  -- "acid", "base", or "neutral"
    cl_sys_L_per_h : float  -- systemic clearance (L/h)
    vd_sys_L : float  -- volume of distribution (L)
    t_end_min : float  -- simulation end time (min)
    dt_min : float  -- ODE step size (min)

    Returns
    -------
    SublingualAbsorptionResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    valid_routes = {"sublingual", "buccal"}
    if route not in valid_routes:
        raise ValueError(f"route must be one of {valid_routes}, got '{route}'")
    valid_ionization = {"acid", "base", "neutral"}
    if ionization_type not in valid_ionization:
        raise ValueError(
            f"ionization_type must be one of {valid_ionization}, got '{ionization_type}'"
        )

    # --- pH and ionization ---
    sublingual_ph = 6.5
    if ionization_type == "acid":
        fraction_unionized = 1.0 / (1.0 + 10.0 ** (sublingual_ph - drug_pka))
    elif ionization_type == "base":
        fraction_unionized = 1.0 / (1.0 + 10.0 ** (drug_pka - sublingual_ph))
    else:  # neutral
        fraction_unionized = 1.0

    # --- Mucosal permeability ---
    peff_raw = 1e-4 * (10.0 ** (logp * 0.3 - 0.5)) * fraction_unionized
    peff_cm_per_s = _clamp(peff_raw, 1e-7, 1e-3)

    # --- Area ---
    if route == "sublingual":
        area_cm2 = 1.0
    else:  # buccal
        area_cm2 = 2.5

    # --- Volume of oral cavity ---
    v_oral_mL = 0.5

    # --- Absorption rate constant (per min) ---
    # ka = Peff [cm/s] * 60 [s/min] * Area [cm2] / V_oral [mL]
    ka_mucosal = peff_cm_per_s * 60.0 * area_cm2 / v_oral_mL

    # --- Salivary washout ---
    k_sal = 0.1  # per min

    # --- Combined loss from oral cavity ---
    k_total = ka_mucosal + k_sal

    # --- Systemic PK (convert to per-min) ---
    cl_sys_L_per_min = cl_sys_L_per_h / 60.0
    ke_sys = cl_sys_L_per_min / vd_sys_L  # per min

    # --- Forward Euler ODE ---
    n_steps = int(math.ceil(t_end_min / dt_min))
    dt = t_end_min / n_steps

    times_min = []
    m_remaining_mg = []
    m_absorbed_mg = []
    c_systemic_mg_L = []

    m_rem = dose_mg
    m_abs = 0.0
    c_sys = 0.0
    t = 0.0

    for _ in range(n_steps + 1):
        times_min.append(t)
        m_remaining_mg.append(m_rem)
        m_absorbed_mg.append(m_abs)
        c_systemic_mg_L.append(c_sys)

        if _ < n_steps:
            # d/dt of remaining in oral cavity
            dm_rem = -k_total * m_rem
            # d/dt absorbed
            dm_abs = ka_mucosal * m_rem
            # d/dt systemic concentration (mg/L)
            dc_sys = (ka_mucosal * m_rem) / vd_sys_L - ke_sys * c_sys

            m_rem = max(0.0, m_rem + dm_rem * dt)
            m_abs = m_abs + dm_abs * dt
            c_sys = max(0.0, c_sys + dc_sys * dt)
            t += dt

    # --- Summary metrics ---
    f_absorbed_total = m_abs / dose_mg if dose_mg > 0 else 0.0
    f_absorbed_total = _clamp(f_absorbed_total, 0.0, 1.0)

    # Salivary loss percentage (fraction that goes to saliva vs total lost)
    # = k_sal / (ka_mucosal + k_sal) * (1 - exp(-k_total * t_end_min)) * 100
    if k_total > 0:
        salivary_loss_pct = (k_sal / k_total) * (1.0 - math.exp(-k_total * t_end_min)) * 100.0
    else:
        salivary_loss_pct = 0.0

    # Tmax and Cmax of systemic concentration
    cmax_systemic_mg_L = max(c_systemic_mg_L)
    tmax_idx = c_systemic_mg_L.index(cmax_systemic_mg_L)
    tmax_systemic_min = times_min[tmax_idx]

    # AUC by manual trapezoidal (in mg*h/L, convert minutes to hours)
    times_h = [t / 60.0 for t in times_min]
    n = len(times_h)
    auc_systemic_mg_h_per_L = sum(
        0.5 * (c_systemic_mg_L[i] + c_systemic_mg_L[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, n)
    )

    # t_half_absorption: time for 50% of dose to be absorbed
    target = 0.5 * dose_mg
    t_half_absorption_min = t_end_min  # default if never reached
    for i, m in enumerate(m_absorbed_mg):
        if m >= target:
            t_half_absorption_min = times_min[i]
            break

    notes = (
        f"Route: {route}. Peff={peff_cm_per_s:.2e} cm/s, fraction_unionized={fraction_unionized:.3f}, "
        f"ka_mucosal={ka_mucosal:.4f}/min, k_sal={k_sal}/min. "
        f"f_abs={f_absorbed_total:.3f}, salivary_loss={salivary_loss_pct:.1f}%."
    )

    return SublingualAbsorptionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        times_min=times_min,
        m_absorbed_mg=m_absorbed_mg,
        m_remaining_mg=m_remaining_mg,
        c_systemic_mg_L=c_systemic_mg_L,
        f_absorbed_total=f_absorbed_total,
        tmax_systemic_min=tmax_systemic_min,
        cmax_systemic_mg_L=cmax_systemic_mg_L,
        auc_systemic_mg_h_per_L=auc_systemic_mg_h_per_L,
        t_half_absorption_min=t_half_absorption_min,
        salivary_loss_pct=salivary_loss_pct,
        notes=notes,
    )


def compare_routes_sublingual(
    drug_name: str,
    dose_mg: float,
    drug_pka: float,
    logp: float,
    ionization_type: str,
    cl_sys_L_per_h: float,
    vd_sys_L: float = 10.0,
) -> list[SublingualAbsorptionResult]:
    """
    Compare sublingual vs buccal routes.

    Returns list of SublingualAbsorptionResult for both routes,
    sorted by cmax_systemic_mg_L descending.
    """
    results = []
    for route in ["sublingual", "buccal"]:
        r = simulate_sublingual_absorption(
            drug_name=drug_name,
            dose_mg=dose_mg,
            route=route,
            drug_pka=drug_pka,
            logp=logp,
            ionization_type=ionization_type,
            cl_sys_L_per_h=cl_sys_L_per_h,
            vd_sys_L=vd_sys_L,
        )
        results.append(r)

    results.sort(key=lambda x: x.cmax_systemic_mg_L, reverse=True)
    return results
