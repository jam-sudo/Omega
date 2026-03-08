"""Drug skin penetration with chemical penetration enhancers and iontophoresis.

Extends basic skin-layer permeation (phase 603) with chemical penetration
enhancers (CPEs) such as oleic acid, ethanol, DMSO, Azone, and propylene
glycol, as well as iontophoresis.  The three-layer skin model (stratum
corneum, viable epidermis, dermis) plus systemic compartment is solved with
forward Euler.  No external libraries are used.

References
----------
- Potts RO & Guy RH.  Pharm Res. 1992 (permeability equation)
- Williams AC & Barry BW.  Adv Drug Deliv Rev. 2004 (enhancers)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

VALID_ENHANCERS = {"none", "oleic_acid", "ethanol", "DMSO", "azone", "PG"}

ENHANCEMENT_FACTORS: dict[str, float] = {
    "none": 1.0,
    "oleic_acid": 5.0,
    "ethanol": 2.0,
    "DMSO": 8.0,
    "azone": 10.0,
    "PG": 1.5,
}


@dataclass
class SkinPenetrationEnhancedResult:
    """Results from an enhanced skin-penetration simulation."""

    drug_name: str
    dose_mg: float
    enhancer: str
    times_h: list[float] = field(default_factory=list)
    c_stratum_corneum_mg_g: list[float] = field(default_factory=list)
    c_viable_epidermis_mg_g: list[float] = field(default_factory=list)
    c_dermis_mg_g: list[float] = field(default_factory=list)
    c_systemic_mg_L: list[float] = field(default_factory=list)
    flux_mg_cm2_h: list[float] = field(default_factory=list)
    enhancement_ratio: float = 0.0
    cmax_systemic_mg_L: float = 0.0
    auc_systemic_mg_h_per_L: float = 0.0
    lag_time_h: float = 0.0
    steady_state_flux: float = 0.0
    total_absorbed_mg: float = 0.0
    bioavailability_pct: float = 0.0
    notes: str = ""


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def simulate_skin_penetration_enhanced(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    enhancer: str = "none",
    application_area_cm2: float = 10.0,
    vehicle_conc_mg_mL: float = 10.0,
    iontophoresis: bool = False,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> SkinPenetrationEnhancedResult:
    """Simulate drug penetration through skin with optional enhancer / iontophoresis."""

    # --- validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if application_area_cm2 <= 0:
        raise ValueError("application_area_cm2 must be > 0")
    if enhancer not in VALID_ENHANCERS:
        raise ValueError(f"enhancer must be one of {sorted(VALID_ENHANCERS)}, got '{enhancer}'")

    # --- base permeability (Potts-Guy) ---
    kp_raw = 10.0 ** (-2.72 + 0.71 * logP - 0.0061 * mw_Da)
    kp = _clamp(kp_raw, 1e-5, 0.1)

    # --- enhancement ---
    enh_factor = ENHANCEMENT_FACTORS[enhancer]
    ionto_factor = 3.0 if iontophoresis else 1.0
    kp_enh = _clamp(kp * enh_factor * ionto_factor, 1e-5, 0.5)
    enhancement_ratio = enh_factor * ionto_factor

    # --- volumes (L) ---
    V_SC_L = application_area_cm2 * 0.001 / 1000.0
    V_VE_L = application_area_cm2 * 0.01 / 1000.0
    V_D_L = application_area_cm2 * 0.1 / 1000.0

    # --- rate constants ---
    k1 = _clamp(kp_enh * 10.0, 0.001, 100.0)
    k2 = _clamp(kp_enh * 5.0, 0.001, 100.0)
    k3 = _clamp(kp_enh * 2.0, 0.001, 100.0)
    k4 = _clamp(kp_enh * 1.0, 0.001, 100.0)
    k_el = cl_sys_L_per_h / vd_sys_L

    # --- initial conditions ---
    A_veh = vehicle_conc_mg_mL * application_area_cm2 * 0.1  # mg
    C_SC = 0.0  # mg/L
    C_VE = 0.0
    C_D = 0.0
    C_sys = 0.0

    times: list[float] = []
    c_sc_list: list[float] = []
    c_ve_list: list[float] = []
    c_d_list: list[float] = []
    c_sys_list: list[float] = []
    flux_list: list[float] = []

    n_steps = int(math.ceil(t_end_h / dt_h)) + 1

    for i in range(n_steps):
        t = i * dt_h

        flux_val = k4 * C_D * V_D_L / application_area_cm2  # mg/cm2/h

        times.append(round(t, 6))
        # Store concentrations converted to mg/g (≈ mg/mL for tissue density ~1)
        # C values are in mg/L, so mg/g ≈ C / 1000
        c_sc_list.append(C_SC / 1000.0)
        c_ve_list.append(C_VE / 1000.0)
        c_d_list.append(C_D / 1000.0)
        c_sys_list.append(C_sys)
        flux_list.append(flux_val)

        # --- forward Euler ---
        dA_veh = -k1 * A_veh
        dC_SC = k1 * A_veh / V_SC_L - k2 * C_SC
        dC_VE = k2 * C_SC * V_SC_L / V_VE_L - k3 * C_VE
        dC_D = k3 * C_VE * V_VE_L / V_D_L - k4 * C_D
        dC_sys = k4 * C_D * V_D_L / vd_sys_L - k_el * C_sys

        A_veh += dA_veh * dt_h
        C_SC += dC_SC * dt_h
        C_VE += dC_VE * dt_h
        C_D += dC_D * dt_h
        C_sys += dC_sys * dt_h

        # Prevent negatives
        A_veh = max(0.0, A_veh)
        C_SC = max(0.0, C_SC)
        C_VE = max(0.0, C_VE)
        C_D = max(0.0, C_D)
        C_sys = max(0.0, C_sys)

    # --- derived quantities ---
    cmax_sys = max(c_sys_list) if c_sys_list else 0.0

    # AUC (trapezoidal)
    auc_sys = sum(
        0.5 * (c_sys_list[i] + c_sys_list[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )

    # lag time: first time C_sys > 1% of Cmax
    threshold = 0.01 * cmax_sys if cmax_sys > 0 else 0.0
    lag_time = 0.0
    for j, c in enumerate(c_sys_list):
        if c > threshold:
            lag_time = times[j]
            break

    ss_flux = flux_list[-1] if flux_list else 0.0

    total_absorbed = auc_sys * cl_sys_L_per_h
    bioavailability = _clamp(total_absorbed / dose_mg * 100.0, 0.0, 100.0)

    return SkinPenetrationEnhancedResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        enhancer=enhancer,
        times_h=times,
        c_stratum_corneum_mg_g=c_sc_list,
        c_viable_epidermis_mg_g=c_ve_list,
        c_dermis_mg_g=c_d_list,
        c_systemic_mg_L=c_sys_list,
        flux_mg_cm2_h=flux_list,
        enhancement_ratio=enhancement_ratio,
        cmax_systemic_mg_L=cmax_sys,
        auc_systemic_mg_h_per_L=auc_sys,
        lag_time_h=lag_time,
        steady_state_flux=ss_flux,
        total_absorbed_mg=total_absorbed,
        bioavailability_pct=bioavailability,
        notes=f"enhancer={enhancer}, iontophoresis={iontophoresis}",
    )
