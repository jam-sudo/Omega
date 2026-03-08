"""
Phase 442 — Predict bioavailability differences between dosage forms.

BCS class-adjusted bioavailability estimates for 9 common dose forms.
"""

from dataclasses import dataclass

DOSE_FORMS = {
    "solution": {
        "f_factor": 1.00,
        "tmax_h": 0.5,
        "variability": "low",
        "food": "none",
        "ped": True,
        "eld": True,
        "diss": "rapid",
        "mfg": "low",
    },
    "suspension": {
        "f_factor": 0.95,
        "tmax_h": 0.8,
        "variability": "moderate",
        "food": "variable",
        "ped": True,
        "eld": True,
        "diss": "rapid",
        "mfg": "low",
    },
    "IR_tablet": {
        "f_factor": 0.90,
        "tmax_h": 1.5,
        "variability": "moderate",
        "food": "variable",
        "ped": False,
        "eld": True,
        "diss": "intermediate",
        "mfg": "moderate",
    },
    "ER_tablet": {
        "f_factor": 0.85,
        "tmax_h": 4.0,
        "variability": "moderate",
        "food": "positive",
        "ped": False,
        "eld": True,
        "diss": "slow",
        "mfg": "high",
    },
    "enteric_coated": {
        "f_factor": 0.88,
        "tmax_h": 3.0,
        "variability": "high",
        "food": "positive",
        "ped": False,
        "eld": True,
        "diss": "slow",
        "mfg": "high",
    },
    "capsule_hard": {
        "f_factor": 0.90,
        "tmax_h": 1.5,
        "variability": "moderate",
        "food": "variable",
        "ped": False,
        "eld": True,
        "diss": "intermediate",
        "mfg": "moderate",
    },
    "capsule_soft": {
        "f_factor": 0.95,
        "tmax_h": 1.0,
        "variability": "low",
        "food": "positive",
        "ped": False,
        "eld": True,
        "diss": "rapid",
        "mfg": "high",
    },
    "chewable": {
        "f_factor": 0.92,
        "tmax_h": 1.2,
        "variability": "low",
        "food": "none",
        "ped": True,
        "eld": True,
        "diss": "intermediate",
        "mfg": "moderate",
    },
    "ODT": {
        "f_factor": 0.93,
        "tmax_h": 0.8,
        "variability": "low",
        "food": "none",
        "ped": True,
        "eld": True,
        "diss": "rapid",
        "mfg": "high",
    },
}

# BCS class adjustments
BCS_ADJUSTMENTS = {
    1: {"default": 1.0, "slow_diss": 1.0},  # High sol, high perm — form has little effect
    2: {"default": 1.0, "slow_diss": 0.85},  # Low sol, high perm — dissolution rate matters
    3: {"default": 0.9, "slow_diss": 0.9},  # High sol, low perm — permeability limited
    4: {"default": 0.7, "slow_diss": 0.7},  # Low sol, low perm — poor F regardless
}

SLOW_DISS_FORMS = {"ER_tablet", "enteric_coated"}


@dataclass(frozen=True)
class DoseFormBioavailabilityResult:
    drug_name: str
    dose_form: str
    bcs_class: int
    f_oral_estimated: float
    f_relative_to_solution: float
    tmax_h: float
    variability_class: str
    food_effect: str
    suitable_for_ped: bool
    suitable_for_elderly: bool
    dissolution_rate_class: str
    manufacturing_complexity: str
    notes: str


def _get_bcs_adjustment(bcs_class: int, dose_form: str) -> float:
    adj = BCS_ADJUSTMENTS[bcs_class]
    if dose_form in SLOW_DISS_FORMS:
        return adj["slow_diss"]
    return adj["default"]


def predict_dose_form_bioavailability(
    drug_name: str,
    dose_form: str,
    bcs_class: int,
    intrinsic_f_oral: float = 0.8,
) -> DoseFormBioavailabilityResult:
    """
    Predict bioavailability for a drug in a given dosage form.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_form : str
        One of the 9 supported dose forms.
    bcs_class : int
        BCS class (1, 2, 3, or 4).
    intrinsic_f_oral : float
        Drug's intrinsic oral bioavailability (0 < value <= 1, default 0.8).

    Returns
    -------
    DoseFormBioavailabilityResult
    """
    if dose_form not in DOSE_FORMS:
        raise ValueError(
            f"Unknown dose_form '{dose_form}'. Must be one of {list(DOSE_FORMS.keys())}."
        )
    if bcs_class not in {1, 2, 3, 4}:
        raise ValueError(f"bcs_class must be 1, 2, 3, or 4, got {bcs_class}.")
    if not (0.0 < intrinsic_f_oral <= 1.0):
        raise ValueError(f"intrinsic_f_oral must be in (0, 1], got {intrinsic_f_oral}.")

    df_info = DOSE_FORMS[dose_form]
    f_factor = df_info["f_factor"]
    f_adjustment = _get_bcs_adjustment(bcs_class, dose_form)

    f_oral_estimated = intrinsic_f_oral * f_factor * f_adjustment
    f_oral_estimated = max(0.0, min(1.0, f_oral_estimated))

    # Reference: solution for same intrinsic F
    f_solution = (
        intrinsic_f_oral
        * DOSE_FORMS["solution"]["f_factor"]
        * _get_bcs_adjustment(bcs_class, "solution")
    )
    f_solution = max(0.0, min(1.0, f_solution))

    if f_solution > 0.0:
        f_relative = f_oral_estimated / f_solution
    else:
        f_relative = 0.0

    notes = (
        f"Dose form: {dose_form}. "
        f"Estimated F: {f_oral_estimated:.3f} (BCS {bcs_class} adjustment: {f_adjustment:.2f}). "
        f"Relative to solution: {f_relative:.3f}. "
        f"Tmax: {df_info['tmax_h']} h."
    )

    return DoseFormBioavailabilityResult(
        drug_name=drug_name,
        dose_form=dose_form,
        bcs_class=bcs_class,
        f_oral_estimated=f_oral_estimated,
        f_relative_to_solution=f_relative,
        tmax_h=df_info["tmax_h"],
        variability_class=df_info["variability"],
        food_effect=df_info["food"],
        suitable_for_ped=df_info["ped"],
        suitable_for_elderly=df_info["eld"],
        dissolution_rate_class=df_info["diss"],
        manufacturing_complexity=df_info["mfg"],
        notes=notes,
    )


def compare_dose_forms(
    drug_name: str,
    bcs_class: int,
    intrinsic_f_oral: float = 0.8,
) -> list:
    """
    Compare bioavailability across all 9 dosage forms.

    Parameters
    ----------
    drug_name : str
    bcs_class : int
    intrinsic_f_oral : float

    Returns
    -------
    list of DoseFormBioavailabilityResult sorted by f_oral_estimated descending.
    """
    results = []
    for df in DOSE_FORMS:
        result = predict_dose_form_bioavailability(
            drug_name=drug_name,
            dose_form=df,
            bcs_class=bcs_class,
            intrinsic_f_oral=intrinsic_f_oral,
        )
        results.append(result)

    results.sort(key=lambda r: r.f_oral_estimated, reverse=True)
    return results
