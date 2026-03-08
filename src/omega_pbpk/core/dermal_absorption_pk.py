"""Phase 974 — Transdermal (passive) drug absorption PK model (3-compartment)."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class DermalAbsorptionResult:
    """Result of transdermal absorption simulation."""

    drug_name: str
    logp: float
    mw: float
    times_h: list
    c_skin_barrier_mg_cm2: list
    c_dermis_mg_L: list
    cmax_dermis_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    f_absorbed: float
    lag_time_h: float
    flux_max_mg_cm2_per_h: float
    notes: str


def simulate_dermal_absorption(
    drug_name: str,
    logp: float,
    mw: float,
    dose_mg_per_cm2: float = 1.0,
    area_cm2: float = 100.0,
    vd_L: float = 50.0,
    cl_L_per_h: float = 5.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> DermalAbsorptionResult:
    """Simulate transdermal (passive diffusion) drug absorption.

    Uses a 3-compartment Forward Euler model:
      - Donor: drug applied on skin surface (mg)
      - Skin barrier (stratum corneum): transit compartment (mg)
      - Dermis/plasma: systemic compartment (mg/L)

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    logp:
        Octanol-water partition coefficient.
    mw:
        Molecular weight (g/mol). Must be > 0.
    dose_mg_per_cm2:
        Applied dose per unit area (mg/cm²).
    area_cm2:
        Application area (cm²). Must be > 0.
    vd_L:
        Volume of distribution (L). Must be > 0.
    cl_L_per_h:
        Systemic clearance (L/h). Must be > 0.
    t_end_h:
        Simulation end time (h).
    dt_h:
        Integration time step (h).

    Returns
    -------
    DermalAbsorptionResult
    """
    # --- Validation ----------------------------------------------------------
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if area_cm2 <= 0:
        raise ValueError(f"area_cm2 must be > 0, got {area_cm2}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")

    # --- Derived parameters --------------------------------------------------
    # k_sc: stratum corneum permeation rate (1/h)
    # Base formula: k_sc = 0.001 * 10^(0.71 * logP) / sqrt(MW)
    # Hydrophilic (logP < 0): scale down by 10^logP
    mw_sqrt = math.sqrt(mw)
    k_sc_base = 0.001 * (10.0 ** (0.71 * logp)) / mw_sqrt
    if logp < 0.0:
        k_sc_base = k_sc_base * (10.0**logp)
    k_sc = min(k_sc_base, 0.5)  # cap at 0.5/h

    # Lag time: MW-dependent, capped at 6 h
    lag_time_h = min(0.1 * mw_sqrt, 6.0)

    # Dermis-to-plasma clearance rate (1/h)
    k_derm = cl_L_per_h / vd_L

    # Evaporation rate from donor (1/h)
    k_evap = 0.05

    # --- Initial conditions --------------------------------------------------
    a_donor = dose_mg_per_cm2 * area_cm2  # total applied amount (mg)
    a_sc = 0.0  # amount in stratum corneum (mg)
    c_plasma = 0.0  # plasma concentration (mg/L)

    # --- Time grid -----------------------------------------------------------
    n_steps = max(1, int(round(t_end_h / dt_h)))
    times: list[float] = []
    c_sc_list: list[float] = []  # amount in SC per unit area (mg/cm²)
    c_pl_list: list[float] = []  # plasma concentration (mg/L)

    flux_max = 0.0
    total_absorbed_mg = 0.0  # cumulative amount entering plasma

    for step in range(n_steps + 1):
        t = step * dt_h
        times.append(t)
        c_sc_list.append(a_sc / area_cm2)
        c_pl_list.append(c_plasma)

        if step == n_steps:
            break

        # Flux from donor → SC: only after lag time
        if t >= lag_time_h:
            flux_donor_to_sc = k_sc * a_donor  # mg/h
            flux = flux_donor_to_sc / area_cm2  # mg/cm²/h for tracking
            if flux > flux_max:
                flux_max = flux
        else:
            flux_donor_to_sc = 0.0

        # SC → plasma transfer
        flux_sc_to_plasma = k_derm * a_sc  # mg/h

        # Evaporation from donor
        evap = k_evap * a_donor  # mg/h

        # Forward Euler update
        da_donor = -(flux_donor_to_sc + evap) * dt_h
        da_sc = (flux_donor_to_sc - flux_sc_to_plasma) * dt_h
        dc_plasma = (flux_sc_to_plasma / vd_L - (cl_L_per_h / vd_L) * c_plasma) * dt_h

        total_absorbed_mg += flux_sc_to_plasma * dt_h

        a_donor = max(0.0, a_donor + da_donor)
        a_sc = max(0.0, a_sc + da_sc)
        c_plasma = max(0.0, c_plasma + dc_plasma)

    # --- Summary statistics --------------------------------------------------
    cmax = max(c_pl_list) if c_pl_list else 0.0
    tmax = times[c_pl_list.index(cmax)] if c_pl_list else 0.0

    # AUC by trapezoidal rule
    auc = 0.0
    for i in range(len(times) - 1):
        auc += 0.5 * (c_pl_list[i] + c_pl_list[i + 1]) * (times[i + 1] - times[i])

    # Fraction absorbed = total that entered plasma / initial applied dose
    total_dose_mg = dose_mg_per_cm2 * area_cm2
    f_absorbed = min(1.0, total_absorbed_mg / total_dose_mg) if total_dose_mg > 0 else 0.0

    notes = (
        f"k_sc={k_sc:.4f}/h, lag={lag_time_h:.2f}h, "
        f"k_derm={k_derm:.4f}/h, k_evap={k_evap:.3f}/h, "
        f"total_dose={total_dose_mg:.1f}mg"
    )

    return DermalAbsorptionResult(
        drug_name=drug_name,
        logp=logp,
        mw=mw,
        times_h=times,
        c_skin_barrier_mg_cm2=c_sc_list,
        c_dermis_mg_L=c_pl_list,
        cmax_dermis_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        f_absorbed=f_absorbed,
        lag_time_h=lag_time_h,
        flux_max_mg_cm2_per_h=flux_max,
        notes=notes,
    )
