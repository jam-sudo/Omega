"""Skin permeation model for topical/transdermal drug delivery.

Implements the Potts-Guy equation for skin permeability and related
calculations for steady-state flux, lag time, and absorbed dose.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["SkinPermeationResult", "calculate_skin_permeation", "compare_vehicles"]

# Vehicle enhancement factors relative to cream
VEHICLE_EF: dict[str, float] = {
    "cream": 1.0,
    "gel": 0.8,
    "ointment": 1.5,
    "patch": 1.2,
    "solution": 0.9,
}

# Skin condition modifiers
CONDITION_MOD: dict[str, float] = {
    "normal": 1.0,
    "damaged": 3.0,
    "diseased": 2.0,
}


@dataclass(frozen=True)
class SkinPermeationResult:
    """Result of skin permeation calculation."""

    drug_name: str
    logP: float
    mw: float
    kp_cm_per_h: float
    lag_time_h: float
    steady_state_flux_ug_cm2_h: float
    penetration_class: str
    vehicle: str
    vehicle_enhancement_factor: float
    absorbed_dose_pct: float
    notes: str


def calculate_skin_permeation(
    drug_name: str,
    logP: float,
    mw: float,
    concentration_mg_mL: float = 10.0,
    application_area_cm2: float = 100.0,
    application_time_h: float = 24.0,
    vehicle: str = "cream",
    skin_condition: str = "normal",
) -> SkinPermeationResult:
    """Calculate skin permeation parameters using the Potts-Guy equation.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    logP:
        Octanol-water partition coefficient (log scale).
    mw:
        Molecular weight (g/mol).
    concentration_mg_mL:
        Drug concentration in the formulation (mg/mL).
    application_area_cm2:
        Skin application area (cm²).
    application_time_h:
        Duration of application (h).
    vehicle:
        Formulation vehicle: "cream", "gel", "ointment", "patch", "solution".
    skin_condition:
        Skin barrier condition: "normal", "damaged", "diseased".

    Returns
    -------
    SkinPermeationResult
    """
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if concentration_mg_mL <= 0:
        raise ValueError(f"concentration_mg_mL must be > 0, got {concentration_mg_mL}")
    if vehicle not in VEHICLE_EF:
        raise ValueError(
            f"Invalid vehicle '{vehicle}'. Must be one of: {sorted(VEHICLE_EF)}"
        )
    if skin_condition not in CONDITION_MOD:
        raise ValueError(
            f"Invalid skin_condition '{skin_condition}'. Must be one of: {sorted(CONDITION_MOD)}"
        )

    # Potts-Guy equation: log(Kp) = -2.72 + 0.71*logP - 0.0061*MW  (Kp in cm/s)
    log_kp_cm_s = -2.72 + 0.71 * logP - 0.0061 * mw
    kp_cm_s = 10.0 ** log_kp_cm_s
    kp_cm_h = kp_cm_s * 3600.0

    # Apply vehicle and condition modifiers
    vehicle_ef = VEHICLE_EF[vehicle]
    condition_mod = CONDITION_MOD[skin_condition]
    kp_eff = kp_cm_h * vehicle_ef * condition_mod

    # Lag time through stratum corneum (~0.01 cm thick)
    h_sc = 0.01  # cm
    d_sc = kp_eff * h_sc / 6.0  # approximate diffusivity from Kp
    lag_time_h = max(0.1, h_sc**2 / (6.0 * max(d_sc, 1e-9)))
    lag_time_h = min(lag_time_h, 12.0)

    # Steady-state flux (µg/cm²/h): J = Kp * C_donor
    # concentration_mg_mL = mg/mL = µg/µL = mg/cm³ → *1000 to get µg/cm³
    ss_flux = kp_eff * concentration_mg_mL * 1000.0  # µg/cm²/h

    # Absorbed dose in 24h accounting for lag time
    effective_time = max(0.0, application_time_h - lag_time_h)
    absorbed_ug_cm2 = ss_flux * effective_time
    absorbed_dose_pct = min(100.0, absorbed_ug_cm2 / (concentration_mg_mL * 10000.0) * 100.0)

    # Penetration class
    if kp_eff < 0.001:
        penetration_class = "low"
    elif kp_eff < 0.1:
        penetration_class = "moderate"
    else:
        penetration_class = "high"

    notes = (
        f"Potts-Guy Kp={kp_cm_h:.4e} cm/h (base); "
        f"effective Kp={kp_eff:.4e} cm/h after "
        f"vehicle ({vehicle}, EF={vehicle_ef}) and "
        f"condition ({skin_condition}, mod={condition_mod}x)."
    )

    return SkinPermeationResult(
        drug_name=drug_name,
        logP=logP,
        mw=mw,
        kp_cm_per_h=kp_eff,
        lag_time_h=lag_time_h,
        steady_state_flux_ug_cm2_h=ss_flux,
        penetration_class=penetration_class,
        vehicle=vehicle,
        vehicle_enhancement_factor=vehicle_ef,
        absorbed_dose_pct=absorbed_dose_pct,
        notes=notes,
    )


def compare_vehicles(
    drug_name: str,
    logP: float,
    mw: float,
    **kwargs,
) -> list[SkinPermeationResult]:
    """Compare all 5 vehicles for a drug.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    logP:
        Octanol-water partition coefficient (log scale).
    mw:
        Molecular weight (g/mol).
    **kwargs:
        Additional keyword arguments forwarded to `calculate_skin_permeation`
        (except `vehicle`).

    Returns
    -------
    list[SkinPermeationResult]
        Results for each vehicle, sorted by kp_cm_per_h descending.
    """
    results = [
        calculate_skin_permeation(drug_name, logP, mw, vehicle=v, **kwargs)
        for v in VEHICLE_EF
    ]
    return sorted(results, key=lambda r: r.kp_cm_per_h, reverse=True)
