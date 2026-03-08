"""
Phase 397 — Multi-Compartment Lymphatic PK

Simulates drug distribution through the lymphatic system with cervical,
thoracic duct, and systemic lymph compartments. Relevant for SC injections
and lymph-targeting nanoparticles.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LymphaticPKResult:
    """Result of a lymphatic PK simulation (plain dataclass — has list fields)."""

    drug_name: str
    route: str
    dose_mg: float
    times_h: list
    c_injection_site_mg_mL: list
    c_lymph_mg_L: list
    c_systemic_mg_L: list
    auc_systemic_mg_h_per_L: float
    auc_lymph_mg_h_per_L: float
    lymphatic_uptake_pct: float
    tmax_systemic_h: float
    cmax_systemic_mg_L: float
    bioavailability_pct: float
    notes: str


_VALID_ROUTES = {"subcutaneous", "intramuscular", "oral"}

_V_INJ_SITE_mL = 1.0  # injection depot volume in mL
_V_LYMPH_L = 2.0  # lymph node volume in L
_K_LYMPH_TO_BLOOD = 0.5  # lymphatic drainage rate (per h)


def _calc_f_lymph(route: str, mw_kDa: float, logp: float) -> float:
    """Compute fraction of dose absorbed via lymphatics."""
    if route == "subcutaneous":
        return 0.1 + 0.05 * min(1.0, mw_kDa / 10.0)
    elif route == "intramuscular":
        return 0.05
    else:  # oral
        raw = 0.01 + 0.02 * max(0.0, logp - 2.0)
        return min(0.1, raw)


def _trapezoidal_auc(x: list, y: list) -> float:
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


def simulate_lymphatic_pk(
    drug_name: str,
    dose_mg: float,
    route: str,
    mw_kDa: float,
    logp: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> LymphaticPKResult:
    """
    Simulate multi-compartment lymphatic PK.

    Parameters
    ----------
    drug_name : str
    dose_mg : float
        Total dose (mg).
    route : str
        One of "subcutaneous", "intramuscular", "oral".
    mw_kDa : float
        Molecular weight in kDa (affects lymphatic uptake for SC).
    logp : float
        Lipophilicity (affects lymphatic uptake for oral).
    cl_sys_L_per_h : float
        Systemic clearance (L/h).
    vd_sys_L : float
        Volume of distribution (L).
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Forward-Euler time step (h).

    Returns
    -------
    LymphaticPKResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if mw_kDa <= 0:
        raise ValueError("mw_kDa must be > 0")
    if route not in _VALID_ROUTES:
        raise ValueError(f"route must be one of {_VALID_ROUTES}, got '{route}'")

    f_lymph = _calc_f_lymph(route, mw_kDa, logp)

    # Rate constants
    k_inj_to_lymph = 0.1 * f_lymph
    k_inj_to_blood = 0.3 * (1.0 - f_lymph)
    k_total_out = k_inj_to_lymph + k_inj_to_blood
    k_elim = cl_sys_L_per_h / vd_sys_L

    # Initial state
    M_site = dose_mg  # mg at injection/dosing site
    C_lymph = 0.0  # mg/L
    C_sys = 0.0  # mg/L

    # Storage
    times_h: list = []
    c_site_list: list = []
    c_lymph_list: list = []
    c_sys_list: list = []

    n_steps = int(round(t_end_h / dt_h))
    t = 0.0
    for _ in range(n_steps + 1):
        times_h.append(t)
        c_site_list.append(M_site / _V_INJ_SITE_mL)  # mg/mL
        c_lymph_list.append(C_lymph)
        c_sys_list.append(C_sys)

        # Forward Euler derivatives
        dM_site = -k_total_out * M_site
        dC_lymph = k_inj_to_lymph * M_site / _V_LYMPH_L - _K_LYMPH_TO_BLOOD * C_lymph
        dC_sys = (
            k_inj_to_blood * M_site / vd_sys_L
            + _K_LYMPH_TO_BLOOD * C_lymph * _V_LYMPH_L / vd_sys_L
            - k_elim * C_sys
        )

        M_site += dM_site * dt_h
        C_lymph += dC_lymph * dt_h
        C_sys += dC_sys * dt_h

        # Clamp to avoid negative floating-point artefacts
        if M_site < 0.0:
            M_site = 0.0
        if C_lymph < 0.0:
            C_lymph = 0.0
        if C_sys < 0.0:
            C_sys = 0.0

        t += dt_h

    auc_systemic = _trapezoidal_auc(times_h, c_sys_list)
    auc_lymph = _trapezoidal_auc(times_h, c_lymph_list)

    cmax_sys = max(c_sys_list)
    tmax_sys = times_h[c_sys_list.index(cmax_sys)]

    # IV reference AUC = dose / CL
    iv_auc_ref = dose_mg / cl_sys_L_per_h
    bioavailability_pct = (auc_systemic / iv_auc_ref) * 100.0 if iv_auc_ref > 0 else 0.0

    notes = (
        f"f_lymph={f_lymph:.3f}, k_inj_to_lymph={k_inj_to_lymph:.4f}/h, "
        f"k_inj_to_blood={k_inj_to_blood:.4f}/h"
    )

    return LymphaticPKResult(
        drug_name=drug_name,
        route=route,
        dose_mg=dose_mg,
        times_h=times_h,
        c_injection_site_mg_mL=c_site_list,
        c_lymph_mg_L=c_lymph_list,
        c_systemic_mg_L=c_sys_list,
        auc_systemic_mg_h_per_L=auc_systemic,
        auc_lymph_mg_h_per_L=auc_lymph,
        lymphatic_uptake_pct=f_lymph * 100.0,
        tmax_systemic_h=tmax_sys,
        cmax_systemic_mg_L=cmax_sys,
        bioavailability_pct=bioavailability_pct,
        notes=notes,
    )


def compare_routes_lymph(
    drug_name: str,
    dose_mg: float,
    mw_kDa: float,
    logp: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
) -> list:
    """
    Compare all three routes ("subcutaneous", "intramuscular", "oral").

    Returns a list of LymphaticPKResult sorted by auc_systemic_mg_h_per_L
    descending (highest systemic exposure first).
    """
    results = []
    for route in _VALID_ROUTES:
        res = simulate_lymphatic_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            route=route,
            mw_kDa=mw_kDa,
            logp=logp,
            cl_sys_L_per_h=cl_sys_L_per_h,
            vd_sys_L=vd_sys_L,
        )
        results.append(res)

    results.sort(key=lambda r: r.auc_systemic_mg_h_per_L, reverse=True)
    return results
