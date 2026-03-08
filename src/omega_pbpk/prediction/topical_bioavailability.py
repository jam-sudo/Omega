"""
Phase 1130 — Topical (dermal) bioavailability prediction.

Uses the extended Potts-Guy model with formulation-specific penetration
modifiers to predict percutaneous absorption from creams, gels, ointments,
lotions, and patches.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_FORMULATION_MODIFIERS: dict = {
    "cream": 0.7,
    "gel": 0.9,
    "ointment": 1.2,
    "lotion": 0.6,
    "patch": 1.5,
}

_VALID_FORMULATIONS = frozenset(_FORMULATION_MODIFIERS.keys())


@dataclass(frozen=True)
class TopicalBioavailabilityResult:
    """Result of a topical bioavailability prediction."""

    drug_name: str
    mw_Da: float
    logP: float
    dose_mg: float
    formulation_type: str
    application_area_cm2: float
    application_duration_h: float
    kp_base_cm_per_h: float
    kp_effective_cm_per_h: float
    flux_mg_per_h: float
    total_absorbed_mg: float
    systemic_bioavailability_pct: float
    local_tissue_concentration_class: str
    systemic_risk: str
    notes: str


def _kp_base(logP: float, mw_Da: float) -> float:
    """Potts-Guy Kp (cm/h): log(Kp) = -2.72 + 0.71*logP - 0.0061*MW."""
    log_kp = -2.72 + 0.71 * logP - 0.0061 * mw_Da
    return 10.0**log_kp


def _local_class(f_pct: float) -> str:
    if f_pct > 50.0:
        return "high"
    if f_pct >= 20.0:
        return "moderate"
    return "low"


def _systemic_risk(f_pct: float) -> str:
    if f_pct > 20.0:
        return "high"
    if f_pct >= 5.0:
        return "moderate"
    return "low"


def predict_topical_bioavailability(
    drug_name: str,
    mw_Da: float,
    logP: float,
    dose_mg: float,
    application_area_cm2: float,
    formulation_type: str,
    application_duration_h: float,
    vehicle_solubility_mg_mL: float,
) -> TopicalBioavailabilityResult:
    """Predict systemic bioavailability after topical application.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    mw_Da:
        Molecular weight (Da).
    logP:
        Octanol-water partition coefficient.
    dose_mg:
        Applied dose (mg).
    application_area_cm2:
        Skin surface area covered (cm²).
    formulation_type:
        One of {"cream", "gel", "ointment", "lotion", "patch"}.
    application_duration_h:
        Duration of skin contact (h).
    vehicle_solubility_mg_mL:
        Drug solubility in the formulation vehicle (mg/mL).

    Returns
    -------
    TopicalBioavailabilityResult
    """
    # Validation
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if application_area_cm2 <= 0:
        raise ValueError("application_area_cm2 must be > 0")
    if application_duration_h <= 0:
        raise ValueError("application_duration_h must be > 0")
    if vehicle_solubility_mg_mL <= 0:
        raise ValueError("vehicle_solubility_mg_mL must be > 0")
    if not (-5.0 <= logP <= 10.0):
        raise ValueError("logP must be in [-5, 10]")
    if formulation_type not in _VALID_FORMULATIONS:
        raise ValueError(f"formulation_type must be one of {sorted(_VALID_FORMULATIONS)}")

    modifier = _FORMULATION_MODIFIERS[formulation_type]
    kp_base = _kp_base(logP, mw_Da)
    kp_eff = kp_base * modifier

    # Drug available in the vehicle film (0.01 cm thick layer)
    film_thickness_cm = 0.01
    vehicle_volume_mL = application_area_cm2 * film_thickness_cm
    drug_in_vehicle_mg = min(dose_mg, vehicle_solubility_mg_mL * vehicle_volume_mL)

    # Flux through skin (mg/h)
    # Kp [cm/h] × concentration [mg/cm³] × area [cm²]
    # concentration = drug_in_vehicle_mg / (area * film_thickness) in mg/cm³
    concentration_mg_per_cm3 = drug_in_vehicle_mg / (application_area_cm2 * film_thickness_cm)
    flux_mg_per_cm2_per_h = kp_eff * concentration_mg_per_cm3
    flux_mg_per_h = flux_mg_per_cm2_per_h * application_area_cm2

    total_absorbed_mg = flux_mg_per_h * application_duration_h
    total_absorbed_mg = min(total_absorbed_mg, dose_mg)

    f_sys_pct = min(100.0, (total_absorbed_mg / dose_mg) * 100.0)

    notes = (
        f"Potts-Guy Kp_base={kp_base:.4e} cm/h; "
        f"modifier={modifier}; "
        f"drug in vehicle={drug_in_vehicle_mg:.3f} mg; "
        f"flux={flux_mg_per_h:.4f} mg/h"
    )

    return TopicalBioavailabilityResult(
        drug_name=drug_name,
        mw_Da=mw_Da,
        logP=logP,
        dose_mg=dose_mg,
        formulation_type=formulation_type,
        application_area_cm2=application_area_cm2,
        application_duration_h=application_duration_h,
        kp_base_cm_per_h=kp_base,
        kp_effective_cm_per_h=kp_eff,
        flux_mg_per_h=flux_mg_per_h,
        total_absorbed_mg=total_absorbed_mg,
        systemic_bioavailability_pct=f_sys_pct,
        local_tissue_concentration_class=_local_class(f_sys_pct),
        systemic_risk=_systemic_risk(f_sys_pct),
        notes=notes,
    )


def compare_formulations(
    drug_name: str,
    mw_Da: float,
    logP: float,
    dose_mg: float,
    application_area_cm2: float,
    application_duration_h: float,
    vehicle_solubility_mg_mL: float,
) -> list:
    """Compare all formulation types and return sorted by bioavailability (desc).

    Parameters
    ----------
    drug_name:
        Name of the drug.
    mw_Da:
        Molecular weight (Da).
    logP:
        Octanol-water partition coefficient.
    dose_mg:
        Applied dose (mg).
    application_area_cm2:
        Skin surface area covered (cm²).
    application_duration_h:
        Duration of skin contact (h).
    vehicle_solubility_mg_mL:
        Drug solubility in the formulation vehicle (mg/mL).

    Returns
    -------
    list[TopicalBioavailabilityResult]
        Five results sorted by systemic_bioavailability_pct descending.
    """
    results = []
    for formulation in _VALID_FORMULATIONS:
        result = predict_topical_bioavailability(
            drug_name=drug_name,
            mw_Da=mw_Da,
            logP=logP,
            dose_mg=dose_mg,
            application_area_cm2=application_area_cm2,
            formulation_type=formulation,
            application_duration_h=application_duration_h,
            vehicle_solubility_mg_mL=vehicle_solubility_mg_mL,
        )
        results.append(result)

    results.sort(key=lambda r: r.systemic_bioavailability_pct, reverse=True)
    return results


# Module-level math import used in _kp_base via math.log10 equivalent (10**)
_ = math.log  # keep math imported (used for clarity; 10** used directly)
