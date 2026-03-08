"""Phase 642 — Drug Dermal Absorption from Topical Formulation.

Predicts systemic absorption from topical dermal application based on
formulation type and drug physicochemistry, using the Potts-Guy model
for skin permeability.
"""

from __future__ import annotations

from dataclasses import dataclass

_VALID_FORMULATIONS = frozenset({"cream", "gel", "ointment", "lotion", "patch"})

_VEHICLE_EFFECT_FACTOR: dict[str, float] = {
    "cream": 1.5,
    "gel": 1.2,
    "ointment": 3.0,
    "lotion": 1.0,
    "patch": 2.0,
}

_KP_MIN_CM_H = 1e-6
_KP_MAX_CM_H = 0.1


@dataclass(frozen=True)
class DermalAbsorptionResult:
    """Result from dermal absorption prediction."""

    drug_name: str
    formulation_type: str
    application_area_cm2: float
    dose_applied_mg: float
    logP: float
    mw_Da: float
    kp_skin_cm_h: float
    steady_state_flux_mg_cm2_h: float
    daily_absorbed_mg: float
    systemic_exposure_pct: float
    vehicle_effect_factor: float
    occlusion_factor: float
    time_to_steady_state_h: float
    risk_of_systemic_effect: str
    notes: str


def _validate_inputs(
    dose_applied_mg: float,
    application_area_cm2: float,
    mw_Da: float,
    formulation_type: str,
    vehicle_conc_mg_mL: float,
) -> None:
    if dose_applied_mg <= 0.0:
        raise ValueError(f"dose_applied_mg must be > 0, got {dose_applied_mg}")
    if application_area_cm2 <= 0.0:
        raise ValueError(f"application_area_cm2 must be > 0, got {application_area_cm2}")
    if mw_Da <= 0.0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if formulation_type not in _VALID_FORMULATIONS:
        raise ValueError(
            f"formulation_type must be one of {sorted(_VALID_FORMULATIONS)}, "
            f"got '{formulation_type}'"
        )
    if vehicle_conc_mg_mL <= 0.0:
        raise ValueError(f"vehicle_conc_mg_mL must be > 0, got {vehicle_conc_mg_mL}")


def _potts_guy_kp(logP: float, mw_Da: float) -> float:
    """Potts-Guy model: log(kp) = -2.72 + 0.71*logP - 0.0061*mw_Da [cm/h]."""
    log_kp = -2.72 + 0.71 * logP - 0.0061 * mw_Da
    kp = 10.0**log_kp
    return max(_KP_MIN_CM_H, min(_KP_MAX_CM_H, kp))


def _risk_from_exposure(systemic_exposure_pct: float) -> str:
    if systemic_exposure_pct < 5.0:
        return "negligible"
    if systemic_exposure_pct < 20.0:
        return "low"
    if systemic_exposure_pct < 50.0:
        return "moderate"
    return "high"


def predict_dermal_absorption(
    drug_name: str,
    dose_applied_mg: float,
    application_area_cm2: float,
    logP: float,
    mw_Da: float,
    formulation_type: str,
    occluded: bool = False,
    vehicle_conc_mg_mL: float = 10.0,
    t_exposure_h: float = 8.0,
) -> DermalAbsorptionResult:
    """Predict systemic absorption from topical dermal application.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_applied_mg:
        Total dose applied to the skin (mg).
    application_area_cm2:
        Skin application area (cm²).
    logP:
        Octanol-water partition coefficient (log scale).
    mw_Da:
        Molecular weight (Da).
    formulation_type:
        One of "cream", "gel", "ointment", "lotion", "patch".
    occluded:
        Whether the application site is occluded (covered).
    vehicle_conc_mg_mL:
        Drug concentration in the vehicle/formulation (mg/mL).
    t_exposure_h:
        Duration of skin contact / exposure time (h).

    Returns
    -------
    DermalAbsorptionResult
    """
    _validate_inputs(
        dose_applied_mg, application_area_cm2, mw_Da, formulation_type, vehicle_conc_mg_mL
    )

    # Potts-Guy skin permeability (unenhanced, cm/h)
    kp_skin = _potts_guy_kp(logP, mw_Da)

    # Vehicle enhancement
    vehicle_factor = _VEHICLE_EFFECT_FACTOR.get(formulation_type, 1.0)

    # Occlusion factor
    occlusion_factor = 3.0 if occluded else 1.0

    kp_effective = kp_skin * vehicle_factor
    kp_final = kp_effective * occlusion_factor

    # Steady-state flux: J_ss = kp_final * vehicle_conc  [mg/cm²/h]
    j_ss = kp_final * vehicle_conc_mg_mL

    # Daily absorbed: use min(24, t_exposure_h)
    effective_h = min(24.0, t_exposure_h)
    daily_absorbed_mg = j_ss * application_area_cm2 * effective_h

    # Systemic exposure percent — clamp to 100
    systemic_exposure_pct = min(100.0, daily_absorbed_mg / dose_applied_mg * 100.0)

    # Lag time: combination of MW and permeability
    t_lag = mw_Da * 0.001 + 1.0 / (kp_skin + 0.01)
    t_lag = max(0.1, min(24.0, t_lag))
    time_to_steady_state_h = t_lag * 3.0

    risk = _risk_from_exposure(systemic_exposure_pct)

    notes = (
        f"Potts-Guy log(kp)={-2.72 + 0.71 * logP - 0.0061 * mw_Da:.3f}; "
        f"kp_skin={kp_skin:.2e} cm/h; vehicle_factor={vehicle_factor:.1f}; "
        f"occlusion_factor={occlusion_factor:.1f}; J_ss={j_ss:.4f} mg/cm²/h"
    )

    return DermalAbsorptionResult(
        drug_name=drug_name,
        formulation_type=formulation_type,
        application_area_cm2=application_area_cm2,
        dose_applied_mg=dose_applied_mg,
        logP=logP,
        mw_Da=mw_Da,
        kp_skin_cm_h=kp_skin,
        steady_state_flux_mg_cm2_h=j_ss,
        daily_absorbed_mg=daily_absorbed_mg,
        systemic_exposure_pct=systemic_exposure_pct,
        vehicle_effect_factor=vehicle_factor,
        occlusion_factor=occlusion_factor,
        time_to_steady_state_h=time_to_steady_state_h,
        risk_of_systemic_effect=risk,
        notes=notes,
    )


def compare_formulations(
    drug_name: str,
    dose_applied_mg: float,
    area_cm2: float,
    logP: float,
    mw_Da: float,
    vehicle_conc_mg_mL: float = 10.0,
) -> list:
    """Compare all 5 formulation types for the same drug and conditions.

    Returns a list of DermalAbsorptionResult objects sorted by
    daily_absorbed_mg in descending order.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_applied_mg:
        Total dose applied to the skin (mg).
    area_cm2:
        Skin application area (cm²).
    logP:
        Octanol-water partition coefficient.
    mw_Da:
        Molecular weight (Da).
    vehicle_conc_mg_mL:
        Drug concentration in vehicle (mg/mL).

    Returns
    -------
    list of DermalAbsorptionResult
    """
    results = [
        predict_dermal_absorption(
            drug_name=drug_name,
            dose_applied_mg=dose_applied_mg,
            application_area_cm2=area_cm2,
            logP=logP,
            mw_Da=mw_Da,
            formulation_type=ft,
            vehicle_conc_mg_mL=vehicle_conc_mg_mL,
        )
        for ft in sorted(_VALID_FORMULATIONS)
    ]
    results.sort(key=lambda r: r.daily_absorbed_mg, reverse=True)
    return results
