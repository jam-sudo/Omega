"""Drug permeation across skin layers (Phase 560).

Models transdermal drug delivery through stratum corneum, epidermis,
dermis into systemic circulation using Potts-Guy permeability and
Forward Euler ODE integration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class TransdermalPermeationResult:
    """Result of transdermal permeation simulation."""

    drug_name: str
    dose_mg_per_cm2: float
    application_area_cm2: float
    logP: float
    times_h: list = field(default_factory=list)
    flux_mg_per_cm2_per_h: list = field(default_factory=list)
    c_systemic_mg_L: list = field(default_factory=list)
    cmax_systemic_mg_L: float = 0.0
    tmax_systemic_h: float = 0.0
    auc_systemic_mg_h_per_L: float = 0.0
    steady_state_flux: float = 0.0
    lag_time_h: float = 0.0
    total_absorbed_mg: float = 0.0
    bioavailability_pct: float = 0.0
    notes: str = ""


def simulate_transdermal_permeation(
    drug_name: str,
    dose_mg_per_cm2: float,
    application_area_cm2: float,
    logP: float,
    mw_Da: float,
    cl_L_per_h: float,
    vd_L: float,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> TransdermalPermeationResult:
    """Simulate transdermal drug permeation through skin layers.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg_per_cm2 : float
        Applied dose per unit area (mg/cm2), must be > 0.
    application_area_cm2 : float
        Patch area in cm2, must be > 0.
    logP : float
        Octanol-water partition coefficient (log scale).
    mw_Da : float
        Molecular weight in Daltons, must be > 0.
    cl_L_per_h : float
        Systemic clearance in L/h, must be > 0.
    vd_L : float
        Volume of distribution in L, must be > 0.
    t_end_h : float
        Simulation end time in hours.
    dt_h : float
        Time step in hours.

    Returns
    -------
    TransdermalPermeationResult
    """
    if dose_mg_per_cm2 <= 0:
        raise ValueError("dose_mg_per_cm2 must be > 0")
    if application_area_cm2 <= 0:
        raise ValueError("application_area_cm2 must be > 0")
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")

    # Potts-Guy permeability model
    log_kp = 0.71 * logP - 0.0061 * mw_Da - 6.3
    kp_cm_per_h = 10.0**log_kp
    # Clamp to [1e-6, 0.1] cm/h
    kp_cm_per_h = max(1e-6, min(0.1, kp_cm_per_h))

    # Lag time (simplified)
    lag_time_h = 0.1 / kp_cm_per_h
    lag_time_h = max(0.01, min(24.0, lag_time_h))

    # Total dose applied
    dose_total_mg = dose_mg_per_cm2 * application_area_cm2

    # State variables
    m_sc = dose_total_mg  # drug in stratum corneum (mg)
    c_sys = 0.0  # systemic concentration (mg/L)

    # Output arrays
    n_steps = int(math.ceil(t_end_h / dt_h)) + 1
    times_h = []
    flux_list = []  # flux per cm2
    c_sys_list = []
    total_absorbed = 0.0
    cmax = 0.0
    tmax = 0.0

    for i in range(n_steps):
        t = i * dt_h
        times_h.append(t)

        # Compute flux
        if t < lag_time_h:
            flux_rate_mg_per_h = 0.0
        else:
            flux_rate_mg_per_h = kp_cm_per_h * m_sc  # first-order depletion

        flux_per_cm2 = flux_rate_mg_per_h / application_area_cm2
        flux_list.append(flux_per_cm2)
        c_sys_list.append(c_sys)

        if c_sys > cmax:
            cmax = c_sys
            tmax = t

        # Forward Euler update
        if i < n_steps - 1:
            dc_dt = flux_rate_mg_per_h / vd_L - (cl_L_per_h / vd_L) * c_sys
            c_sys = c_sys + dc_dt * dt_h
            if c_sys < 0:
                c_sys = 0.0

            m_sc = m_sc - flux_rate_mg_per_h * dt_h
            if m_sc < 0:
                m_sc = 0.0

            total_absorbed += flux_rate_mg_per_h * dt_h

    # Steady-state flux: last flux value
    steady_state_flux = flux_list[-1] if flux_list else 0.0

    # AUC via manual trapezoidal rule
    auc = sum(
        0.5 * (c_sys_list[i] + c_sys_list[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    # Bioavailability
    bioavailability_pct = (total_absorbed / dose_total_mg) * 100.0 if dose_total_mg > 0 else 0.0
    if bioavailability_pct > 100.0:
        bioavailability_pct = 100.0

    notes = (
        f"Potts-Guy Kp={kp_cm_per_h:.2e} cm/h, lag={lag_time_h:.2f} h. "
        f"Total absorbed={total_absorbed:.4f} mg of {dose_total_mg:.2f} mg applied."
    )

    return TransdermalPermeationResult(
        drug_name=drug_name,
        dose_mg_per_cm2=dose_mg_per_cm2,
        application_area_cm2=application_area_cm2,
        logP=logP,
        times_h=times_h,
        flux_mg_per_cm2_per_h=flux_list,
        c_systemic_mg_L=c_sys_list,
        cmax_systemic_mg_L=cmax,
        tmax_systemic_h=tmax,
        auc_systemic_mg_h_per_L=auc,
        steady_state_flux=steady_state_flux,
        lag_time_h=lag_time_h,
        total_absorbed_mg=total_absorbed,
        bioavailability_pct=bioavailability_pct,
        notes=notes,
    )
