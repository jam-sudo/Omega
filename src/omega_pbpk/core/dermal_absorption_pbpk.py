"""Phase 878 — Dermal Absorption PBPK model for transdermal/topical drug delivery.

Compartments:
  - Stratum corneum (SC): main barrier
  - Viable epidermis (VE): intermediate layer
  - Dermis (D): vascularized layer
  - Systemic plasma: absorbed drug

Vehicles supported: solution, cream, gel, patch
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DermalAbsorptionResult:
    """Result of a dermal absorption PBPK simulation."""

    drug_name: str
    dose_mg: float
    vehicle: str
    application_area_cm2: float
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    a_sc_mg: list = field(default_factory=list)
    a_dermis_mg: list = field(default_factory=list)
    cmax_plasma_mg_L: float = 0.0
    tmax_plasma_h: float = 0.0
    auc_plasma_mg_h_per_L: float = 0.0
    flux_steady_state_ug_cm2_h: float = 0.0
    lag_time_h: float = 0.0
    f_absorbed: float = 0.0
    notes: str = ""


# Vehicle multipliers for kp
_VEHICLE_MULTIPLIERS = {
    "solution": 1.0,
    "cream": 0.6,
    "gel": 0.8,
    "patch": 1.5,
}


def simulate_dermal_absorption(
    drug_name: str,
    dose_mg: float,
    application_area_cm2: float = 100.0,
    kp_cm_h: float = 0.001,
    vehicle: str = "solution",
    cl_L_per_h: float = 10.0,
    vd_L: float = 50.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> DermalAbsorptionResult:
    """Simulate transdermal/topical drug absorption using a 4-compartment PBPK model.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        Dose applied to skin surface in mg.
    application_area_cm2:
        Area of skin application in cm².
    kp_cm_h:
        Skin permeability coefficient (cm/h) — base value, modified by vehicle.
    vehicle:
        Formulation vehicle: ``"solution"``, ``"cream"``, ``"gel"``, or ``"patch"``.
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_L:
        Volume of distribution (L).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler step size (h).

    Returns
    -------
    DermalAbsorptionResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if application_area_cm2 <= 0:
        raise ValueError("application_area_cm2 must be > 0")
    if kp_cm_h <= 0:
        raise ValueError("kp_cm_h must be > 0")
    if vehicle not in _VEHICLE_MULTIPLIERS:
        valid = set(_VEHICLE_MULTIPLIERS.keys())
        raise ValueError(f"vehicle must be one of {valid}, got '{vehicle}'")

    # Apply vehicle multiplier to kp
    kp_eff = kp_cm_h * _VEHICLE_MULTIPLIERS[vehicle]

    # Derived rate constants
    k_ve = kp_eff * 2.0  # VE clearance
    k_sys = 0.5  # dermis → systemic (1/h)
    ke = cl_L_per_h / vd_L

    # Lag time estimate (simplified)
    lag_time_h = 1.0 / max(0.001, kp_eff * 10)

    # Steady-state flux (µg/cm²/h)
    # concentration in vehicle = dose_mg / area cm² → multiply by 1000 for µg
    flux_ss = kp_eff * (dose_mg / application_area_cm2) * 1000.0

    # Initial conditions
    a_sc = dose_mg
    a_ve = 0.0
    a_derm = 0.0
    c_plasma = 0.0

    # Storage
    times: list = []
    c_plasma_arr: list = []
    a_sc_arr: list = []
    a_dermis_arr: list = []

    t = 0.0
    n_steps = max(1, round(t_end_h / dt_h))

    for _i in range(n_steps + 1):
        times.append(t)
        c_plasma_arr.append(c_plasma)
        a_sc_arr.append(a_sc)
        a_dermis_arr.append(a_derm)

        if _i == n_steps:
            break

        # ODEs
        da_sc = -kp_eff * a_sc
        da_ve = kp_eff * a_sc - k_ve * a_ve
        da_derm = k_ve * a_ve - k_sys * a_derm
        dc_plasma = k_sys * a_derm / vd_L - ke * c_plasma

        # Forward Euler
        a_sc = max(0.0, a_sc + da_sc * dt_h)
        a_ve = max(0.0, a_ve + da_ve * dt_h)
        a_derm = max(0.0, a_derm + da_derm * dt_h)
        c_plasma = max(0.0, c_plasma + dc_plasma * dt_h)
        t += dt_h

    # Derived metrics
    cmax = max(c_plasma_arr)
    tmax = times[c_plasma_arr.index(cmax)]

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (c_plasma_arr[i - 1] + c_plasma_arr[i]) * (times[i] - times[i - 1])

    f_absorbed = auc * cl_L_per_h / dose_mg

    # Notes
    notes_parts = []
    if vehicle == "patch":
        notes_parts.append("Patch vehicle provides highest absorption")
    if flux_ss > 10.0:
        notes_parts.append("High flux — suitable for systemic delivery")
    if not notes_parts:
        notes_parts.append(f"Vehicle: {vehicle}")
    notes = "; ".join(notes_parts)

    return DermalAbsorptionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        vehicle=vehicle,
        application_area_cm2=application_area_cm2,
        times_h=times,
        c_plasma_mg_L=c_plasma_arr,
        a_sc_mg=a_sc_arr,
        a_dermis_mg=a_dermis_arr,
        cmax_plasma_mg_L=cmax,
        tmax_plasma_h=tmax,
        auc_plasma_mg_h_per_L=auc,
        flux_steady_state_ug_cm2_h=flux_ss,
        lag_time_h=lag_time_h,
        f_absorbed=f_absorbed,
        notes=notes,
    )


def compare_vehicles(
    drug_name: str,
    dose_mg: float,
    application_area_cm2: float = 100.0,
    kp_cm_h: float = 0.001,
    cl_L_per_h: float = 10.0,
    vd_L: float = 50.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> list:
    """Simulate dermal absorption for all four vehicles, sorted by AUC descending.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        Dose applied to skin surface in mg.

    Other parameters are passed through to :func:`simulate_dermal_absorption`.

    Returns
    -------
    list[DermalAbsorptionResult]
        Four results (solution, cream, gel, patch) sorted by ``auc_plasma_mg_h_per_L``
        in descending order.
    """
    results = []
    for vehicle in ("solution", "cream", "gel", "patch"):
        res = simulate_dermal_absorption(
            drug_name=drug_name,
            dose_mg=dose_mg,
            application_area_cm2=application_area_cm2,
            kp_cm_h=kp_cm_h,
            vehicle=vehicle,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(res)

    results.sort(key=lambda r: r.auc_plasma_mg_h_per_L, reverse=True)
    return results
