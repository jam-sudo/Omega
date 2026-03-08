"""Drug permeation through skin layers (stratum corneum, viable epidermis, dermis).

Models transdermal delivery as a three-layer diffusion system coupled to a
one-compartment systemic model.  All ordinary differential equations are
integrated with the forward-Euler method; no external numerical libraries are
required.

Layer cascade
-------------
Stratum corneum (SC) → Viable epidermis (Epi) → Dermis → Systemic circulation

References
----------
- Kalia YN & Guy RH. Adv Drug Deliv Rev. 2001
- Benson HAE. Curr Drug Deliv. 2005
- Naegel A et al., Adv Drug Deliv Rev. 2013
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SkinLayerPermeationResult:
    """Results from a transdermal permeation simulation."""

    drug_name: str
    dose_mg: float
    patch_area_cm2: float
    times_h: list
    c_sc_mg_cm3: list
    c_epidermis_mg_cm3: list
    c_dermis_mg_cm3: list
    c_systemic_mg_L: list
    cmax_systemic_mg_L: float
    tmax_systemic_h: float
    auc_systemic_mg_h_per_L: float
    lag_time_h: float
    steady_state_flux_mg_cm2_h: float
    total_absorbed_pct: float
    notes: str


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def simulate_skin_permeation(
    drug_name: str,
    dose_mg: float,
    patch_area_cm2: float,
    logP: float,
    mw_Da: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    t_end_h: float = 72.0,
    dt_h: float = 0.1,
) -> SkinLayerPermeationResult:
    """Simulate transdermal drug permeation through three skin layers.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        Total applied dose in the patch (mg).
    patch_area_cm2:
        Skin contact area of the patch (cm²).
    logP:
        Octanol-water partition coefficient (log scale).
    mw_Da:
        Molecular weight (Da).
    cl_sys_L_per_h:
        Systemic clearance (L/h).
    vd_sys_L:
        Volume of distribution (L).
    t_end_h:
        Simulation end time (h).  Default 72 h.
    dt_h:
        Forward-Euler step size (h).  Default 0.1 h.

    Returns
    -------
    SkinLayerPermeationResult
    """
    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if patch_area_cm2 <= 0:
        raise ValueError("patch_area_cm2 must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")

    # ------------------------------------------------------------------
    # Permeability coefficients (cm/h)
    # ------------------------------------------------------------------
    P_sc_raw = 0.001 * max(0.01, logP) / max(1.0, mw_Da / 200.0)
    P_sc = _clamp(P_sc_raw, 1e-5, 0.05)

    P_epi = 0.1 / max(1.0, mw_Da / 100.0)
    P_derm = 0.5 / max(1.0, mw_Da / 100.0)

    # ------------------------------------------------------------------
    # Layer geometry
    # ------------------------------------------------------------------
    h_sc = 0.001  # cm
    h_epi = 0.01  # cm
    h_derm = 0.2  # cm

    V_sc = h_sc * patch_area_cm2  # cm³
    V_epi = h_epi * patch_area_cm2  # cm³
    V_derm = h_derm * patch_area_cm2  # cm³

    # ------------------------------------------------------------------
    # Initial conditions
    # ------------------------------------------------------------------
    C_sc = dose_mg / V_sc  # mg/cm³
    C_epi = 0.0
    C_derm = 0.0
    C_sys = 0.0  # mg/L

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    times_h: list = []
    c_sc_list: list = []
    c_epi_list: list = []
    c_derm_list: list = []
    c_sys_list: list = []
    flux_derm_to_sys_list: list = []

    t = 0.0
    n_steps = int(math.ceil(t_end_h / dt_h))

    for step in range(n_steps + 1):
        times_h.append(t)
        c_sc_list.append(C_sc)
        c_epi_list.append(C_epi)
        c_derm_list.append(C_derm)
        c_sys_list.append(C_sys)

        # Fluxes (mg/h)
        flux_sc_to_epi = P_sc * patch_area_cm2 * (C_sc - C_epi)
        flux_epi_to_derm = P_epi * patch_area_cm2 * (C_epi - C_derm)
        flux_derm_to_sys = P_derm * patch_area_cm2 * C_derm  # one-directional sink

        flux_derm_to_sys_list.append(flux_derm_to_sys)

        if step == n_steps:
            break

        # Derivatives
        dC_sc = -flux_sc_to_epi / V_sc
        dC_epi = (flux_sc_to_epi - flux_epi_to_derm) / V_epi
        dC_derm = (flux_epi_to_derm - flux_derm_to_sys) / V_derm
        # dC_sys [mg/L/h] = flux [mg/h] / Vd [L] - (CL/Vd)*C_sys
        dC_sys = flux_derm_to_sys / vd_sys_L - (cl_sys_L_per_h / vd_sys_L) * C_sys

        # Forward Euler update
        C_sc = max(0.0, C_sc + dC_sc * dt_h)
        C_epi = max(0.0, C_epi + dC_epi * dt_h)
        C_derm = max(0.0, C_derm + dC_derm * dt_h)
        C_sys = max(0.0, C_sys + dC_sys * dt_h)

        t = round(t + dt_h, 10)

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------
    cmax_systemic_mg_L = max(c_sys_list)
    tmax_idx = c_sys_list.index(cmax_systemic_mg_L)
    tmax_systemic_h = times_h[tmax_idx]

    # Trapezoidal AUC
    auc_systemic = sum(
        0.5 * (c_sys_list[i] + c_sys_list[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    # Lag time: first time systemic concentration >= 0.001 mg/L
    lag_time_h = t_end_h  # default: never exceeded
    for i, c in enumerate(c_sys_list):
        if c >= 0.001:
            lag_time_h = times_h[i]
            break

    # Steady-state flux: mean flux_derm_to_sys over last quarter of simulation
    quarter_idx = max(1, len(flux_derm_to_sys_list) * 3 // 4)
    last_quarter_fluxes = flux_derm_to_sys_list[quarter_idx:]
    if last_quarter_fluxes:
        mean_flux_mg_h = sum(last_quarter_fluxes) / len(last_quarter_fluxes)
    else:
        mean_flux_mg_h = flux_derm_to_sys_list[-1]
    steady_state_flux_mg_cm2_h = mean_flux_mg_h / patch_area_cm2

    # Total absorbed from SC (fraction lost from SC reservoir)
    final_sc_mass = c_sc_list[-1] * V_sc
    total_absorbed_pct = (dose_mg - final_sc_mass) / dose_mg * 100.0

    notes = (
        f"P_sc={P_sc:.2e} cm/h, P_epi={P_epi:.4f} cm/h, P_derm={P_derm:.4f} cm/h; "
        f"logP={logP}, MW={mw_Da} Da"
    )

    return SkinLayerPermeationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        patch_area_cm2=patch_area_cm2,
        times_h=times_h,
        c_sc_mg_cm3=c_sc_list,
        c_epidermis_mg_cm3=c_epi_list,
        c_dermis_mg_cm3=c_derm_list,
        c_systemic_mg_L=c_sys_list,
        cmax_systemic_mg_L=cmax_systemic_mg_L,
        tmax_systemic_h=tmax_systemic_h,
        auc_systemic_mg_h_per_L=auc_systemic,
        lag_time_h=lag_time_h,
        steady_state_flux_mg_cm2_h=steady_state_flux_mg_cm2_h,
        total_absorbed_pct=total_absorbed_pct,
        notes=notes,
    )
