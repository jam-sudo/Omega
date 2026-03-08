"""Transdermal skin penetration prediction (Phase 1122).

Uses the Potts-Guy equation to predict the stratum-corneum permeability
coefficient (Kp) from logP and molecular weight, then computes flux and
total absorbed amount for a given donor concentration, application area,
and duration.

Reference: Potts & Guy (1992) Pharmaceutical Research 9(5):663-669.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SkinPenetrationResult:
    """Result of a transdermal skin penetration prediction."""

    drug_name: str
    mw_Da: float
    logP: float
    donor_conc_mg_mL: float
    application_area_cm2: float
    duration_h: float
    kp_cm_per_h: float
    flux_µg_per_cm2_per_h: float
    lag_time_h: float
    total_absorbed_µg: float
    total_absorbed_mg: float
    penetration_class: str
    transdermal_feasibility: str
    notes: str


def predict_skin_penetration(
    drug_name: str,
    mw_Da: float,
    logP: float,
    donor_conc_mg_mL: float,
    application_area_cm2: float,
    duration_h: float,
) -> SkinPenetrationResult:
    """Predict transdermal skin penetration using the Potts-Guy equation.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    mw_Da:
        Molecular weight in Daltons.
    logP:
        Octanol-water partition coefficient (log scale).
    donor_conc_mg_mL:
        Donor concentration in the vehicle (mg/mL).
    application_area_cm2:
        Skin application area in cm².
    duration_h:
        Duration of application in hours.

    Returns
    -------
    SkinPenetrationResult
    """
    # Validation
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if logP < -5 or logP > 10:
        raise ValueError(f"logP must be in [-5, 10], got {logP}")
    if donor_conc_mg_mL <= 0:
        raise ValueError(f"donor_conc_mg_mL must be > 0, got {donor_conc_mg_mL}")
    if application_area_cm2 <= 0:
        raise ValueError(f"application_area_cm2 must be > 0, got {application_area_cm2}")
    if duration_h <= 0:
        raise ValueError(f"duration_h must be > 0, got {duration_h}")

    # Potts-Guy equation: log(Kp) = -2.72 + 0.71*logP - 0.0061*MW  (Kp in cm/h)
    log_kp = -2.72 + 0.71 * logP - 0.0061 * mw_Da
    kp_cm_per_h = 10.0**log_kp

    # Donor concentration in µg/mL (= µg/cm³)
    donor_conc_µg_per_mL = donor_conc_mg_mL * 1000.0

    # Flux (µg/cm²/h)
    flux_µg_per_cm2_per_h = kp_cm_per_h * donor_conc_µg_per_mL

    # Lag time estimation
    # D from MW: D = 1e-3 * exp(-MW/100) cm²/h
    diffusivity_cm2_per_h = 1e-3 * math.exp(-mw_Da / 100.0)
    membrane_thickness_cm = 0.001  # stratum corneum ~10 µm
    lag_time_h = 0.5 * membrane_thickness_cm**2 / diffusivity_cm2_per_h

    # Total absorbed (zero if duration <= lag_time)
    effective_time_h = max(0.0, duration_h - lag_time_h)
    total_absorbed_µg = flux_µg_per_cm2_per_h * application_area_cm2 * effective_time_h
    total_absorbed_mg = total_absorbed_µg / 1000.0

    # Penetration class based on Kp
    if kp_cm_per_h < 1e-4:
        penetration_class = "poor"
    elif kp_cm_per_h < 1e-2:
        penetration_class = "moderate"
    else:
        penetration_class = "good"

    # Transdermal feasibility
    # "feasible": moderate or good, MW <= 500; "borderline": poor but MW <= 500; "not_feasible": MW > 500 or very poor
    if mw_Da > 500:
        transdermal_feasibility = "not_feasible"
    elif penetration_class == "good":
        transdermal_feasibility = "feasible"
    elif penetration_class == "moderate":
        transdermal_feasibility = "feasible"
    else:
        transdermal_feasibility = "borderline"

    # Notes
    note_parts = [f"Potts-Guy: log(Kp)={log_kp:.3f}"]
    if mw_Da > 500:
        note_parts.append("MW > 500 Da: transdermal unlikely for macromolecules")
    if logP < -2 or logP > 6:
        note_parts.append(f"logP={logP} outside optimal range [-2, 6]: penetration reduced")
    if duration_h <= lag_time_h:
        note_parts.append(
            f"duration ({duration_h:.1f}h) <= lag time ({lag_time_h:.2f}h): no net absorption"
        )
    notes = "; ".join(note_parts)

    return SkinPenetrationResult(
        drug_name=drug_name,
        mw_Da=mw_Da,
        logP=logP,
        donor_conc_mg_mL=donor_conc_mg_mL,
        application_area_cm2=application_area_cm2,
        duration_h=duration_h,
        kp_cm_per_h=kp_cm_per_h,
        flux_µg_per_cm2_per_h=flux_µg_per_cm2_per_h,
        lag_time_h=lag_time_h,
        total_absorbed_µg=total_absorbed_µg,
        total_absorbed_mg=total_absorbed_mg,
        penetration_class=penetration_class,
        transdermal_feasibility=transdermal_feasibility,
        notes=notes,
    )


def screen_transdermal_candidates(
    candidates: list[dict],
) -> list[SkinPenetrationResult]:
    """Screen a list of candidate compounds for transdermal delivery.

    Each candidate dict must contain keys: ``drug_name``, ``mw_Da``, ``logP``,
    ``donor_conc_mg_mL``, ``application_area_cm2``, ``duration_h``.

    Returns results sorted by ``kp_cm_per_h`` descending (best candidates first).

    Parameters
    ----------
    candidates:
        List of dicts with drug physicochemical parameters.

    Returns
    -------
    list[SkinPenetrationResult]
    """
    results = []
    for c in candidates:
        r = predict_skin_penetration(
            drug_name=c["drug_name"],
            mw_Da=c["mw_Da"],
            logP=c["logP"],
            donor_conc_mg_mL=c["donor_conc_mg_mL"],
            application_area_cm2=c["application_area_cm2"],
            duration_h=c["duration_h"],
        )
        results.append(r)
    results.sort(key=lambda x: x.kp_cm_per_h, reverse=True)
    return results
