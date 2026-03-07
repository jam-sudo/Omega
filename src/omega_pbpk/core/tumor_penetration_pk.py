"""
Phase 803 — Tumor Penetration PK Model

Simulates drug transport through tumor vasculature and interstitium using a
3-compartment model: plasma, tumor vasculature, tumor interstitium.
"""

import math
from dataclasses import dataclass


@dataclass
class TumorPenetrationResult:
    drug_name: str
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    c_tumor_vascular_mg_L: list
    c_tumor_interstitium_mg_L: list
    cmax_plasma: float
    cmax_tumor_vasculature: float
    cmax_tumor_interstitium: float
    auc_plasma: float
    auc_tumor: float
    tumor_plasma_kp: float
    interstitium_plasma_ratio: float
    transcapillary_perm_limited: bool
    tumor_penetration_score: float
    notes: str


def _trapezoidal_auc(times: list, concs: list) -> float:
    """Compute AUC using trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (concs[i] + concs[i - 1]) * dt
    return auc


def simulate_tumor_penetration(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float = 5.0,
    vd_plasma_L: float = 5.0,
    ps_tumor_L_per_h: float = 0.5,
    v_tumor_vascular_L: float = 0.05,
    v_tumor_interstitium_L: float = 0.2,
    cl_tumor_L_per_h: float = 0.1,
    ifp_factor: float = 1.0,
    mw_kDa: float = 0.5,
    route: str = "iv",
    f_oral: float = 0.8,
    ka_per_h: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> TumorPenetrationResult:
    """
    Simulate tumor penetration PK using a 3-compartment ODE model.

    Compartments:
      - Plasma (1-cpt with oral or IV input)
      - Tumor vasculature (exchange with plasma via tumor blood flow Q_tumor)
      - Tumor interstitium (exchange with vasculature via PS product, IFP-modulated)

    Parameters
    ----------
    drug_name : str
    dose_mg : float
    cl_L_per_h : float
        Systemic clearance from plasma
    vd_plasma_L : float
        Plasma volume of distribution
    ps_tumor_L_per_h : float
        Permeability-surface area product
    v_tumor_vascular_L : float
        Vascular volume of tumor
    v_tumor_interstitium_L : float
        Interstitial volume of tumor
    cl_tumor_L_per_h : float
        Drug clearance from tumor (e.g., metabolism)
    ifp_factor : float
        Interstitial fluid pressure modifier (1.0=normal, 0.2=high IFP)
    mw_kDa : float
        Molecular weight in kDa; large MW reduces PS
    route : str
        "iv" or "oral"
    f_oral : float
        Oral bioavailability fraction
    ka_per_h : float
        Oral absorption rate constant
    t_end_h : float
        Simulation end time in hours
    dt_h : float
        Time step in hours

    Returns
    -------
    TumorPenetrationResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if ps_tumor_L_per_h < 0:
        raise ValueError("ps_tumor_L_per_h must be non-negative")
    if vd_plasma_L <= 0:
        raise ValueError("vd_plasma_L must be positive")
    if cl_L_per_h < 0:
        raise ValueError("cl_L_per_h must be non-negative")
    if v_tumor_vascular_L <= 0:
        raise ValueError("v_tumor_vascular_L must be positive")
    if v_tumor_interstitium_L <= 0:
        raise ValueError("v_tumor_interstitium_L must be positive")
    if mw_kDa < 0:
        raise ValueError("mw_kDa must be non-negative")
    if ifp_factor <= 0:
        raise ValueError("ifp_factor must be positive")

    # MW-dependent effective PS: larger molecules have reduced PS
    ps_eff = ps_tumor_L_per_h * math.exp(-0.1 * mw_kDa)

    # Transcapillary permeability-limited: PS * transit_time < 1
    # Approximate transit time ~ Vvascular / (cl_L_per_h * 0.01)
    # Use a simplified criterion: PS_eff < 0.1 → limited
    # For practical model: use half-life in vascular compartment
    # PS * (v_tumor_vascular / (ps_eff + cl_tumor)) < 1 approximation
    if ps_eff > 0:
        t_transit_approx = v_tumor_vascular_L / (ps_eff * ifp_factor + 1e-9)
        transcapillary_perm_limited = (ps_eff * t_transit_approx) < 1.0
    else:
        transcapillary_perm_limited = True

    # --- Initial conditions ---
    # For IV: entire dose goes directly to plasma
    # For oral: dose is in GI compartment (absorbed with ka)
    c_plasma = 0.0
    c_vasc = 0.0
    c_inter = 0.0
    a_gut = 0.0  # oral absorption depot (mg)

    if route == "iv":
        c_plasma = dose_mg / vd_plasma_L  # mg/L
    else:
        a_gut = dose_mg * f_oral  # mg in gut (bioavailability applied to absorbed fraction)

    n_steps = int(t_end_h / dt_h) + 1
    times = []
    cp_list = []
    cv_list = []
    ci_list = []

    for step in range(n_steps):
        t = step * dt_h
        times.append(t)
        cp_list.append(c_plasma)
        cv_list.append(c_vasc)
        ci_list.append(c_inter)

        if step == n_steps - 1:
            break

        # --- Oral absorption into plasma ---
        if route != "iv":
            d_gut = -ka_per_h * a_gut
            rate_abs = ka_per_h * a_gut  # mg/h into plasma
        else:
            d_gut = 0.0
            rate_abs = 0.0

        # --- Tumor blood flow rate (simplified): Q_tumor
        # Q_tumor modulates delivery from plasma to tumor vasculature
        # Use Q_tumor = cl_L_per_h * 0.05 (5% of systemic flow goes to tumor)
        q_tumor = cl_L_per_h * 0.05

        # --- ODEs ---
        # dC_plasma/dt = (rate_abs - CL*C_p - Q_tumor*(C_p - C_v)) / Vd_plasma
        dcp = (
            rate_abs / vd_plasma_L
            - (cl_L_per_h / vd_plasma_L) * c_plasma
            - (q_tumor / vd_plasma_L) * (c_plasma - c_vasc)
        )

        # dC_vasc/dt = (Q_tumor*(C_p - C_v) - PS_eff*ifp_factor*(C_v - C_i)) / V_vasc
        dcv = (
            q_tumor * (c_plasma - c_vasc) - ps_eff * ifp_factor * (c_vasc - c_inter)
        ) / v_tumor_vascular_L

        # dC_inter/dt = (PS_eff*ifp_factor*(C_v - C_i) - CL_tumor*C_i) / V_inter
        dci = (
            ps_eff * ifp_factor * (c_vasc - c_inter) - cl_tumor_L_per_h * c_inter
        ) / v_tumor_interstitium_L

        # Forward Euler update
        c_plasma = max(0.0, c_plasma + dcp * dt_h)
        c_vasc = max(0.0, c_vasc + dcv * dt_h)
        c_inter = max(0.0, c_inter + dci * dt_h)

        if route != "iv":
            a_gut = max(0.0, a_gut + d_gut * dt_h)

    # --- Summary statistics ---
    cmax_plasma = max(cp_list)
    cmax_vasc = max(cv_list)
    cmax_inter = max(ci_list)

    auc_plasma = _trapezoidal_auc(times, cp_list)
    auc_vasc = _trapezoidal_auc(times, cv_list)
    auc_inter = _trapezoidal_auc(times, ci_list)
    auc_tumor = auc_vasc + auc_inter

    tumor_plasma_kp = auc_tumor / auc_plasma if auc_plasma > 0 else 0.0
    interstitium_plasma_ratio = cmax_inter / cmax_plasma if cmax_plasma > 0 else 0.0

    # Tumor penetration score 0-100
    # Based on: interstitium/plasma ratio (main driver) + kp_tumor
    # Score = 50 * min(interstitium_plasma_ratio, 1.0) + 50 * min(tumor_plasma_kp / 2.0, 1.0)
    score_ipr = min(interstitium_plasma_ratio * 100.0, 50.0)
    score_kp = min(tumor_plasma_kp / 2.0 * 50.0, 50.0)
    tumor_penetration_score = score_ipr + score_kp

    # Build notes
    notes_parts = []
    if ps_eff < 0.1:
        notes_parts.append("Very low PS: permeability-limited tumor penetration.")
    if ifp_factor < 0.5:
        notes_parts.append("High interstitial fluid pressure (IFP) reduces drug delivery.")
    if mw_kDa > 10:
        notes_parts.append(f"Large molecule (MW={mw_kDa} kDa): PS reduced by MW effect.")
    if tumor_penetration_score >= 70:
        notes_parts.append("Good tumor penetration expected.")
    elif tumor_penetration_score >= 40:
        notes_parts.append("Moderate tumor penetration.")
    else:
        notes_parts.append("Poor tumor penetration predicted.")
    notes = " ".join(notes_parts) if notes_parts else "Normal tumor penetration."

    return TumorPenetrationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_plasma_mg_L=cp_list,
        c_tumor_vascular_mg_L=cv_list,
        c_tumor_interstitium_mg_L=ci_list,
        cmax_plasma=cmax_plasma,
        cmax_tumor_vasculature=cmax_vasc,
        cmax_tumor_interstitium=cmax_inter,
        auc_plasma=auc_plasma,
        auc_tumor=auc_tumor,
        tumor_plasma_kp=tumor_plasma_kp,
        interstitium_plasma_ratio=interstitium_plasma_ratio,
        transcapillary_perm_limited=transcapillary_perm_limited,
        tumor_penetration_score=tumor_penetration_score,
        notes=notes,
    )
