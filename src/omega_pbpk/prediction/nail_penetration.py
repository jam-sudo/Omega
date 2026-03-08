"""Nail penetration prediction for topical drug delivery.

Predicts drug penetration into nail plate — relevant for onychomycosis
treatments (terbinafine, amorolfine, ciclopirox).
"""

from __future__ import annotations

from dataclasses import dataclass

_VALID_FORMULATIONS = {"lacquer", "solution", "patch", "cream"}
_FORMULATION_MODIFIERS = {
    "lacquer": 1.5,
    "patch": 2.0,
    "solution": 1.0,
    "cream": 0.5,
}
_NAIL_THICKNESS_CM = 0.05  # 0.5 mm


@dataclass(frozen=True)
class NailPenetrationResult:
    """Result of nail penetration prediction."""

    drug_name: str
    logp: float
    mw_Da: float
    formulation: str
    nail_permeability_coefficient_cm_h: float
    predicted_nail_conc_ug_g: float
    penetration_depth_fraction: float
    antifungal_target_achievable: bool
    estimated_treatment_duration_weeks: int
    nail_retention_score: float
    notes: str


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _pow10(x: float) -> float:
    """10**x using stdlib math."""
    import math

    return math.pow(10.0, x)


def predict_nail_penetration(
    drug_name: str,
    logp: float = 4.0,
    mw_Da: float = 300.0,
    formulation: str = "lacquer",
    application_dose_mg_cm2: float = 1.0,
) -> NailPenetrationResult:
    """Predict drug penetration into nail plate.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    logp:
        Octanol-water partition coefficient (log scale).
    mw_Da:
        Molecular weight in Daltons.
    formulation:
        Formulation type: "lacquer", "solution", "patch", or "cream".
    application_dose_mg_cm2:
        Applied dose in mg per cm² of nail surface.

    Returns
    -------
    NailPenetrationResult
    """
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")
    if formulation not in _VALID_FORMULATIONS:
        raise ValueError(
            f"formulation must be one of {sorted(_VALID_FORMULATIONS)}, got '{formulation}'"
        )
    if application_dose_mg_cm2 <= 0:
        raise ValueError("application_dose_mg_cm2 must be > 0")

    # Base permeability coefficient
    log_kp = -5.0 + logp * 0.3 - mw_Da / 1000.0
    modifier = _FORMULATION_MODIFIERS[formulation]
    raw_kp = _pow10(log_kp) * modifier
    nail_kp = _clamp(raw_kp, 1e-7, 1e-2)

    # Flux through nail (mg/cm²/h)
    flux_mg_cm2_h = nail_kp * application_dose_mg_cm2

    # Predicted nail concentration after 7 days (168 h)
    # flux × 168 h × 1000 µg/mg / 0.5 g/cm³ (density volume factor simplified)
    raw_nail_conc = flux_mg_cm2_h * 168.0 * 1000.0
    predicted_nail_conc_ug_g = _clamp(raw_nail_conc, 0.001, 1e5)

    # Penetration depth fraction in 7 days
    raw_depth = nail_kp * 168.0 / _NAIL_THICKNESS_CM
    penetration_depth_fraction = min(1.0, raw_depth)

    # Antifungal target
    antifungal_target_achievable = predicted_nail_conc_ug_g >= 0.5

    # Estimated treatment duration (weeks)
    safe_depth = max(0.01, penetration_depth_fraction)
    estimated_treatment_duration_weeks = max(4, int(12.0 / safe_depth))

    # Nail retention score (0-100)
    retention_bonus = 50 if formulation in {"lacquer", "patch"} else 20
    raw_retention = logp * 10.0 + retention_bonus
    nail_retention_score = _clamp(raw_retention, 0.0, 100.0)

    # Notes
    if antifungal_target_achievable:
        notes = "Therapeutic nail concentrations achieved — effective for onychomycosis"
    elif formulation in {"lacquer", "patch"}:
        notes = "Occlusive formulation enhances nail penetration"
    else:
        notes = "Standard nail penetration — consider lacquer formulation for better efficacy"

    return NailPenetrationResult(
        drug_name=drug_name,
        logp=logp,
        mw_Da=mw_Da,
        formulation=formulation,
        nail_permeability_coefficient_cm_h=nail_kp,
        predicted_nail_conc_ug_g=predicted_nail_conc_ug_g,
        penetration_depth_fraction=penetration_depth_fraction,
        antifungal_target_achievable=antifungal_target_achievable,
        estimated_treatment_duration_weeks=estimated_treatment_duration_weeks,
        nail_retention_score=nail_retention_score,
        notes=notes,
    )


def compare_nail_formulations(
    drug_name: str,
    logp: float = 4.0,
    mw_Da: float = 300.0,
) -> list[NailPenetrationResult]:
    """Compare all four formulations for a drug.

    Returns list of NailPenetrationResult for all 4 formulations,
    sorted by predicted_nail_conc_ug_g descending.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    logp:
        Octanol-water partition coefficient (log scale).
    mw_Da:
        Molecular weight in Daltons.
    """
    results = []
    for form in _VALID_FORMULATIONS:
        result = predict_nail_penetration(
            drug_name=drug_name,
            logp=logp,
            mw_Da=mw_Da,
            formulation=form,
        )
        results.append(result)

    results.sort(key=lambda r: r.predicted_nail_conc_ug_g, reverse=True)
    return results
