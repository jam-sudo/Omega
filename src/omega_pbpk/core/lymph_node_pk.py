"""Lymph node drug distribution — Phase 734.

3-compartment model for SC/IM injection → lymphatic vessels → lymph node + systemic.
Relevant for immunology drugs (vaccines, immunosuppressants, nanoparticles).

Compartments:
  A_inj (mg)    : injection site
  A_lymph (mg)  : lymphatic vessel (in transit)
  A_node (mg)   : draining lymph node
  C_plasma (mg/L): systemic plasma
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LymphNodePKResult:
    """Result of lymph node drug distribution simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    a_node_mg: list
    a_inj_mg: list
    cmax_plasma: float
    tmax_plasma_h: float
    auc_plasma: float
    cmax_node_mg: float
    tmax_node_h: float
    lymph_to_plasma_exposure_ratio: float
    cumulative_node_exposure_mg_h: float
    f_via_lymphatics: float
    notes: str


def simulate_lymph_node_pk(
    drug_name: str,
    dose_mg: float,
    f_lymph: float = 0.1,
    k_rel_per_h: float = 0.3,
    k_lv_per_h: float = 1.0,
    k_out_node_per_h: float = 0.05,
    k_met_node_per_h: float = 0.02,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    t_end_h: float = 72.0,
    dt_h: float = 0.1,
) -> LymphNodePKResult:
    """Simulate lymph node drug distribution after SC/IM injection.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Injection dose (mg). Must be > 0.
    f_lymph : float
        Fraction absorbed via lymphatics (0-1). Default 0.10.
    k_rel_per_h : float
        Injection site release rate (1/h).
    k_lv_per_h : float
        Lymphatic vessel transit rate (1/h).
    k_out_node_per_h : float
        Lymph node outflow to systemic rate (1/h).
    k_met_node_per_h : float
        Lymph node metabolism/retention rate (1/h).
    cl_L_per_h : float
        Systemic clearance (L/h). Must be > 0.
    vd_L : float
        Volume of distribution (L). Must be > 0.
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Forward Euler time step (h).

    Returns
    -------
    LymphNodePKResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if f_lymph < 0 or f_lymph > 1:
        raise ValueError("f_lymph must be between 0 and 1")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")

    ke = cl_L_per_h / vd_L

    # Initial conditions
    a_inj = dose_mg
    a_lymph = 0.0
    a_node = 0.0
    c_plasma = 0.0

    times_h: list = []
    c_plasma_list: list = []
    a_node_list: list = []
    a_inj_list: list = []

    n_steps = max(1, int(round(t_end_h / dt_h)))
    t = 0.0

    for i in range(n_steps + 1):
        times_h.append(t)
        c_plasma_list.append(c_plasma)
        a_node_list.append(a_node)
        a_inj_list.append(a_inj)

        if i < n_steps:
            # ODEs (Forward Euler)
            da_inj = -k_rel_per_h * a_inj
            da_lymph = f_lymph * k_rel_per_h * a_inj - k_lv_per_h * a_lymph
            da_node = k_lv_per_h * a_lymph - (k_out_node_per_h + k_met_node_per_h) * a_node
            dc_plasma = (
                (1.0 - f_lymph) * k_rel_per_h * a_inj / vd_L
                + k_out_node_per_h * a_node / vd_L
                - ke * c_plasma
            )

            a_inj = max(0.0, a_inj + da_inj * dt_h)
            a_lymph = max(0.0, a_lymph + da_lymph * dt_h)
            a_node = max(0.0, a_node + da_node * dt_h)
            c_plasma = max(0.0, c_plasma + dc_plasma * dt_h)
            t += dt_h

    # PK metrics — plasma
    cmax_plasma = max(c_plasma_list)
    tmax_plasma_idx = c_plasma_list.index(cmax_plasma)
    tmax_plasma_h = times_h[tmax_plasma_idx]

    # PK metrics — lymph node
    cmax_node_mg = max(a_node_list)
    tmax_node_idx = a_node_list.index(cmax_node_mg)
    tmax_node_h = times_h[tmax_node_idx]

    # Trapezoidal AUC
    auc_plasma = 0.0
    for i in range(len(times_h) - 1):
        auc_plasma += (
            0.5 * (c_plasma_list[i] + c_plasma_list[i + 1]) * (times_h[i + 1] - times_h[i])
        )

    cumulative_node_exposure = 0.0
    for i in range(len(times_h) - 1):
        cumulative_node_exposure += (
            0.5 * (a_node_list[i] + a_node_list[i + 1]) * (times_h[i + 1] - times_h[i])
        )

    # Lymph-to-plasma exposure ratio: normalize by Vd for dimensional consistency
    auc_node_conc_equiv = cumulative_node_exposure / vd_L  # convert mg*h to mg/L*h
    lymph_to_plasma_ratio = auc_node_conc_equiv / auc_plasma if auc_plasma > 0 else 0.0

    # Notes
    enrichment_ratio = cumulative_node_exposure / (dose_mg * 1.0) if dose_mg > 0 else 0.0
    if enrichment_ratio > 0.5:
        notes_str = "Good lymph node targeting — high node exposure relative to dose."
    elif enrichment_ratio > 0.1:
        notes_str = "Moderate lymph node accumulation observed."
    else:
        notes_str = "Poor lymph node targeting — most drug bypasses lymph nodes."

    return LymphNodePKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        a_node_mg=a_node_list,
        a_inj_mg=a_inj_list,
        cmax_plasma=cmax_plasma,
        tmax_plasma_h=tmax_plasma_h,
        auc_plasma=auc_plasma,
        cmax_node_mg=cmax_node_mg,
        tmax_node_h=tmax_node_h,
        lymph_to_plasma_exposure_ratio=lymph_to_plasma_ratio,
        cumulative_node_exposure_mg_h=cumulative_node_exposure,
        f_via_lymphatics=f_lymph,
        notes=notes_str,
    )


def assess_lymph_targeting(result: LymphNodePKResult) -> dict:
    """Assess lymph node targeting quality from simulation result.

    Parameters
    ----------
    result : LymphNodePKResult
        Result from simulate_lymph_node_pk.

    Returns
    -------
    dict with keys:
        - lymph_enrichment: "poor" / "moderate" / "good"
        - cmax_node_mg: float
        - recommendation: str
    """
    enrichment_ratio = result.cumulative_node_exposure_mg_h / (result.dose_mg * 1.0)

    if enrichment_ratio < 0.1:
        lymph_enrichment = "poor"
        recommendation = (
            "Consider increasing lymphatic uptake fraction (f_lymph) or using "
            "lymph-targeting formulations (e.g., lipid nanoparticles)."
        )
    elif enrichment_ratio <= 0.5:
        lymph_enrichment = "moderate"
        recommendation = (
            "Moderate lymph node exposure. May be sufficient for local immunomodulation. "
            "Consider slow-release depot formulations to enhance node retention."
        )
    else:
        lymph_enrichment = "good"
        recommendation = (
            "Good lymph node targeting achieved. Suitable for lymph node-directed therapies "
            "such as vaccines or immunosuppressants."
        )

    return {
        "lymph_enrichment": lymph_enrichment,
        "cmax_node_mg": result.cmax_node_mg,
        "recommendation": recommendation,
    }


__all__ = [
    "LymphNodePKResult",
    "simulate_lymph_node_pk",
    "assess_lymph_targeting",
]
