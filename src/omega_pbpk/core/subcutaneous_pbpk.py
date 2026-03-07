"""Subcutaneous (SC) injection PBPK — Phase 672.

Models SC injection pharmacokinetics with depot absorption, lymphatic transport,
and interstitial drug distribution using a 3-compartment Forward Euler model:
  1. Depot: drug at injection site
  2. Interstitial: drug in interstitial fluid surrounding capillaries
  3. Plasma: systemic circulation

Science references:
  Supersaxo A et al. (1990) Pharm Res 7:167-169.
  Kagan L et al. (2007) J Pharm Sci 96:2541-2556.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "SubcutaneousPKResult",
    "simulate_sc_pk",
    "compare_sc_to_iv",
    # Phase 855
    "SubcutaneousPKResult855",
    "simulate_sc_absorption",
    "compare_sc_iv",
]


@dataclass
class SubcutaneousPKResult:
    """Result of a detailed subcutaneous PBPK simulation."""

    drug_name: str
    dose_mg: float
    route: str  # "sc"
    times_h: list
    a_depot_mg: list  # amount remaining in depot (mg)
    c_interstitial_mg_L: list  # interstitial fluid concentration (mg/L)
    c_plasma_mg_L: list  # plasma concentration (mg/L)
    cmax_plasma: float
    tmax_plasma_h: float
    auc_plasma: float
    t_half_plasma_h: float
    f_absorbed: float  # fraction absorbed from depot (not degraded)
    f_lymphatic: float  # fraction absorbed via lymphatic route
    ka_effective_per_h: float  # effective overall absorption rate
    lag_time_h: float  # estimated lag time
    notes: str


def _trapz(times: list, concs: list) -> float:
    """Manual trapezoidal AUC integration."""
    auc = 0.0
    for i in range(len(times) - 1):
        auc += (concs[i] + concs[i + 1]) / 2.0 * (times[i + 1] - times[i])
    return auc


def simulate_sc_pk(
    drug_name: str,
    dose_mg: float,
    mw_kDa: float = 0.5,
    ka_capillary_per_h: float = 0.5,
    ka_lymphatic_per_h: float = 0.05,
    k_degradation_per_h: float = 0.01,
    cl_plasma_L_per_h: float = 5.0,
    vd_plasma_L: float = 50.0,
    v_interstitial_L: float = 12.0,
    k_interstitial_to_plasma: float = 2.0,
    lag_time_h: float = 0.25,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> SubcutaneousPKResult:
    """Simulate subcutaneous injection pharmacokinetics.

    3-compartment model (Forward Euler):
      Depot → Interstitial → Plasma
           ↘ (lymphatic)  → Plasma
           ↘ (degradation)

    Before lag_time_h: no absorption from depot.
    After lag_time_h:
      dA_depot/dt = -(ka_cap + ka_lymph + k_degrad) * A_depot
      dC_inter/dt = ka_cap * A_depot / V_inter - k_inter_to_plasma * C_inter
      dC_plasma/dt = k_inter_to_plasma * C_inter * V_inter / V_plasma
                   + ka_lymph * A_depot / V_plasma
                   - (CL / V_plasma) * C_plasma

    Parameters
    ----------
    drug_name : str
        Drug identifier.
    dose_mg : float
        SC dose in mg (must be > 0).
    mw_kDa : float
        Molecular weight in kDa (must be > 0). Informational; does not modify
        default rates (caller should set ka_lymphatic_per_h for biologics).
    ka_capillary_per_h : float
        Capillary absorption rate constant (1/h, must be > 0).
    ka_lymphatic_per_h : float
        Lymphatic absorption rate constant (1/h).
    k_degradation_per_h : float
        Enzymatic degradation rate at injection site (1/h).
    cl_plasma_L_per_h : float
        Plasma clearance (L/h, must be > 0).
    vd_plasma_L : float
        Plasma volume of distribution (L, must be > 0).
    v_interstitial_L : float
        Interstitial fluid volume (L, must be > 0).
    k_interstitial_to_plasma : float
        Transfer rate from interstitial to plasma (1/h).
    lag_time_h : float
        Lag time before absorption starts (h, >= 0).
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Forward Euler time step (h).

    Returns
    -------
    SubcutaneousPKResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if mw_kDa <= 0:
        raise ValueError(f"mw_kDa must be > 0, got {mw_kDa}")
    if ka_capillary_per_h <= 0:
        raise ValueError(f"ka_capillary_per_h must be > 0, got {ka_capillary_per_h}")
    if cl_plasma_L_per_h <= 0:
        raise ValueError(f"cl_plasma_L_per_h must be > 0, got {cl_plasma_L_per_h}")
    if vd_plasma_L <= 0:
        raise ValueError(f"vd_plasma_L must be > 0, got {vd_plasma_L}")
    if v_interstitial_L <= 0:
        raise ValueError(f"v_interstitial_L must be > 0, got {v_interstitial_L}")
    if lag_time_h < 0:
        raise ValueError(f"lag_time_h must be >= 0, got {lag_time_h}")

    # State variables
    a_depot = dose_mg  # mg remaining in depot
    c_inter = 0.0  # mg/L in interstitial
    c_plasma = 0.0  # mg/L in plasma

    ke_plasma = cl_plasma_L_per_h / vd_plasma_L  # elimination rate constant

    # Output arrays
    n_steps = max(int(round(t_end_h / dt_h)), 1)
    dt_actual = t_end_h / n_steps

    times: list[float] = []
    a_depot_arr: list[float] = []
    c_inter_arr: list[float] = []
    c_plasma_arr: list[float] = []

    # Track degraded amount
    a_degraded = 0.0

    for i in range(n_steps + 1):
        t = i * dt_actual
        times.append(t)
        a_depot_arr.append(a_depot)
        c_inter_arr.append(c_inter)
        c_plasma_arr.append(c_plasma)

        if i < n_steps:
            absorbing = t >= lag_time_h

            if absorbing:
                k_total = ka_capillary_per_h + ka_lymphatic_per_h + k_degradation_per_h
                da_depot = -k_total * a_depot

                # Capillary flux: depot → interstitial
                flux_cap_to_inter = ka_capillary_per_h * a_depot  # mg/h

                # Lymphatic flux: depot → plasma directly
                flux_lymph_to_plasma = ka_lymphatic_per_h * a_depot  # mg/h

                # Degradation
                flux_degrad = k_degradation_per_h * a_depot  # mg/h

                # Interstitial dynamics
                dc_inter = flux_cap_to_inter / v_interstitial_L - k_interstitial_to_plasma * c_inter

                # Plasma dynamics
                dc_plasma = (
                    k_interstitial_to_plasma * c_inter * v_interstitial_L / vd_plasma_L
                    + flux_lymph_to_plasma / vd_plasma_L
                    - ke_plasma * c_plasma
                )

                a_depot = max(0.0, a_depot + da_depot * dt_actual)
                a_degraded += flux_degrad * dt_actual
                c_inter = max(0.0, c_inter + dc_inter * dt_actual)
                c_plasma = max(0.0, c_plasma + dc_plasma * dt_actual)
            else:
                # Before lag: only plasma elimination, no absorption
                dc_plasma = -ke_plasma * c_plasma
                c_plasma = max(0.0, c_plasma + dc_plasma * dt_actual)

    # PK metrics
    cmax_plasma = max(c_plasma_arr)
    tmax_plasma_h = times[c_plasma_arr.index(cmax_plasma)]
    auc_plasma = _trapz(times, c_plasma_arr)

    # t_half from plasma elimination rate
    t_half_plasma_h = math.log(2.0) / ke_plasma

    # f_absorbed: fraction not degraded
    f_absorbed = 1.0 - a_degraded / dose_mg if dose_mg > 0 else 0.0
    f_absorbed = min(1.0, max(0.0, f_absorbed))

    # f_lymphatic: fraction of total absorption rate that is lymphatic
    k_abs_total = ka_capillary_per_h + ka_lymphatic_per_h + k_degradation_per_h
    f_lymphatic = ka_lymphatic_per_h / k_abs_total if k_abs_total > 0 else 0.0

    # Effective absorption rate = capillary + lymphatic (not degradation)
    ka_effective_per_h = ka_capillary_per_h + ka_lymphatic_per_h

    notes = (
        f"SC PBPK for '{drug_name}': dose={dose_mg} mg, MW={mw_kDa} kDa. "
        f"ka_cap={ka_capillary_per_h}/h, ka_lymph={ka_lymphatic_per_h}/h, "
        f"k_degrad={k_degradation_per_h}/h, lag={lag_time_h} h. "
        f"CL={cl_plasma_L_per_h} L/h, Vd={vd_plasma_L} L, "
        f"V_inter={v_interstitial_L} L. "
        f"Cmax={cmax_plasma:.4g} mg/L at {tmax_plasma_h:.2f} h. "
        f"AUC={auc_plasma:.4g} mg*h/L. "
        f"f_absorbed={f_absorbed:.3f}, f_lymphatic={f_lymphatic:.3f}."
    )

    return SubcutaneousPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route="sc",
        times_h=times,
        a_depot_mg=a_depot_arr,
        c_interstitial_mg_L=c_inter_arr,
        c_plasma_mg_L=c_plasma_arr,
        cmax_plasma=cmax_plasma,
        tmax_plasma_h=tmax_plasma_h,
        auc_plasma=auc_plasma,
        t_half_plasma_h=t_half_plasma_h,
        f_absorbed=f_absorbed,
        f_lymphatic=f_lymphatic,
        ka_effective_per_h=ka_effective_per_h,
        lag_time_h=lag_time_h,
        notes=notes,
    )


def compare_sc_to_iv(
    drug_name: str,
    dose_mg: float,
    mw_kDa: float = 0.5,
    ka_capillary_per_h: float = 0.5,
    cl_plasma_L_per_h: float = 5.0,
    vd_plasma_L: float = 50.0,
) -> dict:
    """Compare SC vs IV pharmacokinetics for the same drug.

    IV reference: instantaneous bolus, analytical 1-compartment solution.
    SC: simulate_sc_pk with default parameters.

    Parameters
    ----------
    drug_name : str
        Drug name.
    dose_mg : float
        SC/IV dose (mg, must be > 0).
    mw_kDa : float
        Molecular weight (kDa, must be > 0).
    ka_capillary_per_h : float
        SC capillary absorption rate constant (1/h).
    cl_plasma_L_per_h : float
        Total body clearance (L/h, must be > 0).
    vd_plasma_L : float
        Volume of distribution (L, must be > 0).

    Returns
    -------
    dict
        Keys: "sc_result", "iv_result" (dict), "cmax_ratio", "auc_ratio", "tmax_ratio".
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_plasma_L_per_h <= 0:
        raise ValueError(f"cl_plasma_L_per_h must be > 0, got {cl_plasma_L_per_h}")
    if vd_plasma_L <= 0:
        raise ValueError(f"vd_plasma_L must be > 0, got {vd_plasma_L}")

    t_end = max(48.0, 5.0 * vd_plasma_L / cl_plasma_L_per_h)

    sc_result = simulate_sc_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        mw_kDa=mw_kDa,
        ka_capillary_per_h=ka_capillary_per_h,
        cl_plasma_L_per_h=cl_plasma_L_per_h,
        vd_plasma_L=vd_plasma_L,
        t_end_h=t_end,
    )

    # IV analytical: C(t) = (dose/Vd) * exp(-ke * t)
    ke = cl_plasma_L_per_h / vd_plasma_L
    cmax_iv = dose_mg / vd_plasma_L  # C at t=0
    auc_iv = dose_mg / cl_plasma_L_per_h  # analytical AUC_inf
    t_half_iv = math.log(2.0) / ke

    iv_result = {
        "drug_name": drug_name,
        "route": "iv",
        "cmax_plasma": cmax_iv,
        "tmax_plasma_h": 0.0,
        "auc_plasma": auc_iv,
        "t_half_plasma_h": t_half_iv,
        "notes": f"IV analytical bolus: Cmax={cmax_iv:.4g} mg/L, AUC={auc_iv:.4g} mg*h/L",
    }

    cmax_ratio = sc_result.cmax_plasma / cmax_iv if cmax_iv > 0 else float("nan")
    auc_ratio = sc_result.auc_plasma / auc_iv if auc_iv > 0 else float("nan")
    tmax_ratio = sc_result.tmax_plasma_h / 0.001 if sc_result.tmax_plasma_h > 0 else float("nan")
    # tmax_ratio: SC Tmax relative to IV (IV tmax=0, so just return SC tmax)
    tmax_ratio = sc_result.tmax_plasma_h  # SC Tmax in hours (IV is at t=0)

    return {
        "sc_result": sc_result,
        "iv_result": iv_result,
        "cmax_ratio": cmax_ratio,
        "auc_ratio": auc_ratio,
        "tmax_ratio": tmax_ratio,
    }


# ---------------------------------------------------------------------------
# Phase 855 — SC absorption PBPK with lymphatic pathway
# ---------------------------------------------------------------------------


@dataclass
class SubcutaneousPKResult855:
    """Phase 855 result of SC absorption PBPK simulation (plain dataclass, list fields)."""

    drug_name: str
    dose_mg: float
    mw_kDa: float
    route: str

    times_h: list
    c_plasma_mg_L: list
    a_sc_depot_mg: list
    a_lymph_mg: list

    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float

    f_via_lymph: float
    t_lag_h: float
    notes: str


def _apply_mw_effect_855(
    mw_kDa: float,
    ka_direct_per_h: float,
    ka_lymph_per_h: float,
) -> tuple:
    """Adjust absorption rate constants based on molecular weight (Phase 855)."""
    ka_direct = ka_direct_per_h
    ka_lymph = ka_lymph_per_h

    if mw_kDa > 5.0:
        scale = (mw_kDa / 5.0) * 0.5
        ka_lymph = ka_lymph_per_h * scale
        default_ka_lymph_cap = 3.0 * ka_lymph_per_h
        if ka_lymph > default_ka_lymph_cap:
            ka_lymph = default_ka_lymph_cap

    if mw_kDa > 50.0:
        ka_direct = ka_direct_per_h * 0.1

    return ka_direct, ka_lymph


def simulate_sc_absorption(
    drug_name: str,
    dose_mg: float,
    mw_kDa: float = 1.0,
    ka_direct_per_h: float = 0.3,
    ka_lymph_per_h: float = 0.05,
    ka_lym_to_plasma_per_h: float = 0.1,
    vd_L: float = 10.0,
    cl_L_per_h: float = 1.0,
    f_bioavail: float = 0.8,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> SubcutaneousPKResult855:
    """Simulate subcutaneous drug absorption via 3-compartment Forward Euler model.

    Phase 855 implementation: SC depot -> lymph -> plasma.

    Compartments:
      SC depot (mg): injection site
      Lymph (mg): lymphatic absorption route
      Plasma (mg/L): systemic

    ODEs:
      d(sc_depot)/dt = -(ka_direct + ka_lymph) * sc_depot
      d(lymph)/dt = ka_lymph * sc_depot - ka_lym_to_plasma * lymph
      d(C_plasma)/dt = ka_direct * sc_depot / Vd + ka_lym_to_plasma * lymph / Vd - ke * C_plasma
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if mw_kDa <= 0:
        raise ValueError("mw_kDa must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if not (0 < f_bioavail <= 1.0):
        raise ValueError("f_bioavail must be in (0, 1]")

    ka_direct, ka_lymph = _apply_mw_effect_855(mw_kDa, ka_direct_per_h, ka_lymph_per_h)
    ke = cl_L_per_h / vd_L

    sc_depot = dose_mg * f_bioavail
    a_lymph = 0.0
    c_plasma = 0.0

    times_h_out: list = []
    c_plasma_list: list = []
    a_sc_depot_list: list = []
    a_lymph_list: list = []

    t = 0.0
    n_steps = int(math.ceil(t_end_h / dt_h)) + 1

    total_via_direct = 0.0
    total_via_lymph = 0.0

    for _ in range(n_steps):
        times_h_out.append(t)
        c_plasma_list.append(c_plasma)
        a_sc_depot_list.append(sc_depot)
        a_lymph_list.append(a_lymph)

        if t >= t_end_h:
            break

        d_sc_depot = -(ka_direct + ka_lymph) * sc_depot
        d_lymph = ka_lymph * sc_depot - ka_lym_to_plasma_per_h * a_lymph
        d_c_plasma = (
            ka_direct * sc_depot / vd_L + ka_lym_to_plasma_per_h * a_lymph / vd_L - ke * c_plasma
        )

        total_via_direct += ka_direct * sc_depot * dt_h
        total_via_lymph += ka_lym_to_plasma_per_h * a_lymph * dt_h

        sc_depot = sc_depot + d_sc_depot * dt_h
        a_lymph = a_lymph + d_lymph * dt_h
        c_plasma = c_plasma + d_c_plasma * dt_h

        if sc_depot < 0.0:
            sc_depot = 0.0
        if a_lymph < 0.0:
            a_lymph = 0.0
        if c_plasma < 0.0:
            c_plasma = 0.0

        t = round(t + dt_h, 10)

    cmax_mg_L = max(c_plasma_list)
    tmax_h = times_h_out[c_plasma_list.index(cmax_mg_L)]

    auc = 0.0
    for i in range(1, len(times_h_out)):
        auc += (
            0.5 * (c_plasma_list[i] + c_plasma_list[i - 1]) * (times_h_out[i] - times_h_out[i - 1])
        )

    total_absorbed = total_via_direct + total_via_lymph
    if total_absorbed > 0:
        f_via_lymph = total_via_lymph / total_absorbed
    else:
        f_via_lymph = 0.0
    f_via_lymph = max(0.0, min(1.0, f_via_lymph))

    threshold = 0.1 * cmax_mg_L
    t_lag_h = 0.0
    for i, c in enumerate(c_plasma_list):
        if c >= threshold:
            t_lag_h = times_h_out[i]
            break

    notes_parts = [
        f"MW={mw_kDa:.1f} kDa",
        f"ka_direct={ka_direct:.4f}/h",
        f"ka_lymph={ka_lymph:.4f}/h",
        f"ke={ke:.4f}/h",
        f"F={f_bioavail:.2f}",
    ]
    if mw_kDa > 50:
        notes_parts.append("large protein: primarily lymphatic absorption")
    elif mw_kDa > 5:
        notes_parts.append("moderate MW: enhanced lymphatic fraction")

    notes = "; ".join(notes_parts)

    return SubcutaneousPKResult855(
        drug_name=drug_name,
        dose_mg=dose_mg,
        mw_kDa=mw_kDa,
        route="subcutaneous",
        times_h=times_h_out,
        c_plasma_mg_L=c_plasma_list,
        a_sc_depot_mg=a_sc_depot_list,
        a_lymph_mg=a_lymph_list,
        cmax_mg_L=cmax_mg_L,
        tmax_h=tmax_h,
        auc_mg_h_per_L=auc,
        f_via_lymph=f_via_lymph,
        t_lag_h=t_lag_h,
        notes=notes,
    )


def compare_sc_iv(
    drug_name: str,
    dose_mg: float,
    mw_kDa: float = 1.0,
    vd_L: float = 10.0,
    cl_L_per_h: float = 1.0,
    t_end_h: float = 24.0,
) -> dict:
    """Compare SC and IV administration for the same drug and dose (Phase 855).

    For IV: ka_direct=100/h, ka_lymph=0, f_bioavail=1.0.

    Returns
    -------
    dict with keys "sc" and "iv", each a SubcutaneousPKResult855.
    """
    sc_result = simulate_sc_absorption(
        drug_name=drug_name,
        dose_mg=dose_mg,
        mw_kDa=mw_kDa,
        ka_direct_per_h=0.3,
        ka_lymph_per_h=0.05,
        ka_lym_to_plasma_per_h=0.1,
        vd_L=vd_L,
        cl_L_per_h=cl_L_per_h,
        f_bioavail=0.8,
        t_end_h=t_end_h,
    )

    iv_result = simulate_sc_absorption(
        drug_name=drug_name,
        dose_mg=dose_mg,
        mw_kDa=mw_kDa,
        ka_direct_per_h=100.0,
        ka_lymph_per_h=0.0,
        ka_lym_to_plasma_per_h=0.1,
        vd_L=vd_L,
        cl_L_per_h=cl_L_per_h,
        f_bioavail=1.0,
        t_end_h=t_end_h,
    )

    return {"sc": sc_result, "iv": iv_result}
