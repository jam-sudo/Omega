"""
Phase 406 — Drug Permeation Across Skin Layers

Detailed model of drug permeation through skin layers (stratum corneum,
viable epidermis, dermis) with Fick's diffusion approximation.
"""

import math
from dataclasses import dataclass

_FORMULATION_MODS: dict = {
    "solution": 1.0,
    "cream": 0.6,
    "gel": 0.8,
    "ointment": 0.4,
    "patch": 1.2,
}


@dataclass
class SkinPermeationResult:
    drug_name: str
    formulation_type: str
    times_h: list
    flux_ug_per_cm2_per_h: list
    cumulative_permeated_ug_per_cm2: list
    c_sc_mg_g: list  # stratum corneum concentration (mg/g tissue)
    c_dermis_mg_L: list  # dermis concentration
    steady_state_flux: float  # ug/cm2/h at steady state
    lag_time_h: float  # time to reach 10% of SS flux
    t_max_flux_h: float  # time to maximum flux
    permeability_coefficient_cm_h: float  # Kp total
    bioavailability_transdermal_pct: float  # absorbed / applied * 100
    notes: str


def _potts_guy_kp(logp: float, mw_da: float) -> float:
    """Potts-Guy equation: logKp (cm/h)."""
    log_kp = -2.7 + 0.71 * logp - 0.0061 * mw_da
    kp = 10**log_kp
    return max(1e-6, min(0.1, kp))


def _build_skin_notes(
    formulation_type: str,
    logp: float,
    mw_da: float,
    bioavailability_transdermal_pct: float,
    steady_state_flux: float,
) -> str:
    parts = []
    if logp > 3 and mw_da < 500:
        parts.append("Favorable physicochemical properties for transdermal delivery.")
    elif mw_da > 500:
        parts.append("High MW may limit skin penetration; consider nanoparticle formulations.")
    if formulation_type == "patch":
        parts.append("Occlusive patch enhances hydration and permeation.")
    elif formulation_type == "ointment":
        parts.append("Ointment provides occlusion but slower release of drug.")
    if bioavailability_transdermal_pct < 5:
        parts.append("Low transdermal bioavailability; dose adjustment may be needed.")
    elif bioavailability_transdermal_pct > 50:
        parts.append("High transdermal absorption; monitor for systemic effects.")
    if not parts:
        parts.append("Moderate transdermal permeation expected.")
    return " ".join(parts)


def simulate_skin_permeation(
    drug_name: str,
    applied_dose_ug_per_cm2: float,
    logp: float,
    mw_da: float,
    formulation_type: str = "solution",
    application_area_cm2: float = 10.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> SkinPermeationResult:
    """
    Simulate drug permeation through skin layers.

    Parameters
    ----------
    drug_name : str
    applied_dose_ug_per_cm2 : float  ug per cm2 applied to skin
    logp : float
    mw_da : float  molecular weight in Da
    formulation_type : str  one of solution/cream/gel/ointment/patch
    application_area_cm2 : float
    t_end_h : float  simulation end time (h)
    dt_h : float  time step (h)

    Returns
    -------
    SkinPermeationResult
    """
    # Validation
    if applied_dose_ug_per_cm2 <= 0:
        raise ValueError(f"applied_dose_ug_per_cm2 must be > 0, got {applied_dose_ug_per_cm2}")
    if formulation_type not in _FORMULATION_MODS:
        raise ValueError(
            f"formulation_type must be one of {list(_FORMULATION_MODS)}, got '{formulation_type}'"
        )
    if mw_da <= 0:
        raise ValueError(f"mw_da must be > 0, got {mw_da}")
    if application_area_cm2 <= 0:
        raise ValueError(f"application_area_cm2 must be > 0, got {application_area_cm2}")

    # Permeability coefficient (Potts-Guy)
    kp_cm_h = _potts_guy_kp(logp, mw_da)
    effective_kp = kp_cm_h * _FORMULATION_MODS[formulation_type]

    # Surface concentration (ug/cm3): thin SC layer ~0.01 cm thick
    c_surface_ug_per_cm3 = applied_dose_ug_per_cm2 * 100.0  # ug/cm2 / 0.01 cm

    # Steady-state flux (ug/cm2/h)
    ss_flux = effective_kp * c_surface_ug_per_cm3

    # Lag time and rise constant
    t_lag = max(0.1, 0.1 / effective_kp)
    tau_rise = max(0.01, 0.5 / effective_kp)

    # ODE state
    M_surface = applied_dose_ug_per_cm2  # ug/cm2 remaining on surface
    M_permeated = 0.0  # ug/cm2 total permeated

    times_h = []
    flux_list = []
    cumulative_list = []
    c_sc_list = []
    c_dermis_list = []

    t = 0.0
    t_max_flux_h = 0.0
    max_flux_seen = 0.0

    while t <= t_end_h + 1e-9:
        # Compute instantaneous flux
        if t < t_lag:
            flux_t = 0.0
        else:
            flux_t = ss_flux * (1.0 - math.exp(-(t - t_lag) / tau_rise))

        # Deplete surface reservoir
        if M_surface <= 0:
            flux_t = 0.0
            M_surface = 0.0

        # Track peak flux
        if flux_t > max_flux_seen:
            max_flux_seen = flux_t
            t_max_flux_h = t

        # Record
        times_h.append(round(t, 10))
        flux_list.append(flux_t)
        cumulative_list.append(M_permeated)

        # SC concentration proxy (mg/g)
        c_sc_mg_g = M_surface * 100.0  # proxy: ug/cm2 * 100 → mg/g rough

        # Dermis concentration proxy (mg/L)
        c_dermis_mg_L = M_permeated * 0.01  # small fraction distributes deep

        c_sc_list.append(c_sc_mg_g)
        c_dermis_list.append(c_dermis_mg_L)

        # Forward Euler update
        M_permeated += flux_t * dt_h
        M_surface -= flux_t * dt_h
        if M_surface < 0:
            M_surface = 0.0

        t += dt_h

    # Bioavailability
    total_applied = applied_dose_ug_per_cm2
    bioavailability_pct = (
        min(100.0, M_permeated / total_applied * 100.0) if total_applied > 0 else 0.0
    )

    # Lag time: time to reach 10% of SS flux
    lag_time_out = t_lag  # analytical lag time

    notes = _build_skin_notes(formulation_type, logp, mw_da, bioavailability_pct, ss_flux)

    return SkinPermeationResult(
        drug_name=drug_name,
        formulation_type=formulation_type,
        times_h=times_h,
        flux_ug_per_cm2_per_h=flux_list,
        cumulative_permeated_ug_per_cm2=cumulative_list,
        c_sc_mg_g=c_sc_list,
        c_dermis_mg_L=c_dermis_list,
        steady_state_flux=ss_flux,
        lag_time_h=lag_time_out,
        t_max_flux_h=t_max_flux_h,
        permeability_coefficient_cm_h=effective_kp,
        bioavailability_transdermal_pct=bioavailability_pct,
        notes=notes,
    )


def compare_formulations_skin(
    drug_name: str,
    applied_dose_ug_per_cm2: float,
    logp: float,
    mw_da: float,
) -> list:
    """
    Compare all 5 formulations for a given drug.

    Returns list of SkinPermeationResult sorted by steady_state_flux descending.
    """
    results = []
    for form_type in _FORMULATION_MODS:
        result = simulate_skin_permeation(
            drug_name=drug_name,
            applied_dose_ug_per_cm2=applied_dose_ug_per_cm2,
            logp=logp,
            mw_da=mw_da,
            formulation_type=form_type,
        )
        results.append(result)

    results.sort(key=lambda r: r.steady_state_flux, reverse=True)
    return results
