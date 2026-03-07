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
