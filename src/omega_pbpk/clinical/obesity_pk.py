"""PK parameter scaling for obese patients (Phase 419 + Phase 625)."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "ObesityPKResult",
    "ObesityPKScalingResult",
    "compute_lean_body_weight",
    "obesity_pk_scaling",
    "adjust_for_obesity",
    "compute_body_metrics",
    "obesity_class_from_bmi",
]

_VALID_SEXES = {"male", "female"}
_VALID_DISTRIBUTION_TYPES = {"hydrophilic", "lipophilic", "mixed"}


# ---------------------------------------------------------------------------
# Phase 625 dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObesityPKResult:
    """Result of obesity-adjusted PK computation (Phase 625)."""

    drug_name: str
    total_body_weight_kg: float
    ideal_body_weight_kg: float
    lean_body_mass_kg: float
    adjusted_body_weight_kg: float
    bmi: float
    obesity_class: str  # "normal","overweight","class1","class2","class3"
    vd_scaling_factor: float
    cl_scaling_factor: float
    adjusted_vd_L: float
    adjusted_cl_L_per_h: float
    adjusted_t_half_h: float
    recommended_dose_basis: str  # "tbw","ibw","abw","lbm","flat"
    recommended_dose_mg: float
    notes: str


# ---------------------------------------------------------------------------
# Phase 419 dataclass (renamed for backward compat)
# ---------------------------------------------------------------------------


@dataclass
class ObesityPKScalingResult:
    """Result of obesity-adjusted PK parameter computation (Phase 419)."""

    drug_name: str
    bmi: float
    bmi_category: str
    lbw_kg: float
    tbw_kg: float
    distribution_type: str
    cl_adjusted_L_per_h: float
    vd_adjusted_L: float
    t_half_adjusted_h: float
    dose_recommendation: str
    notes: str


# ---------------------------------------------------------------------------
# Phase 625 functions
# ---------------------------------------------------------------------------


def compute_body_metrics(
    weight_kg: float,
    height_cm: float,
    sex: str,
) -> dict:
    """Compute IBW, LBM, ABW, BMI.

    IBW (Devine): male = 50 + 2.3*(height_in - 60); female = 45.5 + 2.3*(height_in - 60)
    LBM (Janmahasatian): male = 9270*weight/(6680 + 216*BMI); female = 9270*weight/(8780+244*BMI)
    ABW = IBW + 0.4*(TBW - IBW)
    BMI = weight/(height_m)^2
    Returns dict: ibw, lbm, abw, bmi
    """
    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m * height_m)

    height_in = height_cm / 2.54
    if sex == "male":
        ibw = 50.0 + 2.3 * (height_in - 60.0)
        lbm = 9270.0 * weight_kg / (6680.0 + 216.0 * bmi)
    else:
        ibw = 45.5 + 2.3 * (height_in - 60.0)
        lbm = 9270.0 * weight_kg / (8780.0 + 244.0 * bmi)

    ibw = max(ibw, 20.0)
    lbm = min(lbm, weight_kg)

    abw = ibw + 0.4 * (weight_kg - ibw)

    return {"ibw": ibw, "lbm": lbm, "abw": abw, "bmi": bmi}


def obesity_class_from_bmi(bmi: float) -> str:
    """< 25 -> normal; < 30 -> overweight; < 35 -> class1; < 40 -> class2; else -> class3."""
    if bmi < 25.0:
        return "normal"
    elif bmi < 30.0:
        return "overweight"
    elif bmi < 35.0:
        return "class1"
    elif bmi < 40.0:
        return "class2"
    else:
        return "class3"


def adjust_for_obesity(
    drug_name: str,
    total_body_weight_kg: float,
    height_cm: float,
    sex: str,  # "male","female"
    baseline_cl_L_per_h: float,
    baseline_vd_L: float,
    baseline_dose_mg: float,
    drug_type: str = "hydrophilic",  # "hydrophilic","lipophilic","intermediate"
    fe_renal: float = 0.3,
    baseline_weight_kg: float = 70.0,
    normal_gfr_mL_per_min: float = 100.0,
    actual_gfr_mL_per_min: float | None = None,
) -> ObesityPKResult:
    """Compute obesity-adjusted PK parameters and dosing recommendations."""
    if total_body_weight_kg <= 0:
        raise ValueError("total_body_weight_kg must be positive")
    if height_cm <= 0:
        raise ValueError("height_cm must be positive")
    if sex not in {"male", "female"}:
        raise ValueError(f"sex must be 'male' or 'female', got {sex!r}")
    if drug_type not in {"hydrophilic", "lipophilic", "intermediate"}:
        raise ValueError(
            f"drug_type must be hydrophilic/lipophilic/intermediate, got {drug_type!r}"
        )
    if not (0.0 <= fe_renal <= 1.0):
        raise ValueError("fe_renal must be in [0, 1]")

    metrics = compute_body_metrics(total_body_weight_kg, height_cm, sex)
    ibw = metrics["ibw"]
    lbm = metrics["lbm"]
    abw = metrics["abw"]
    bmi = metrics["bmi"]
    tbw = total_body_weight_kg

    obesity_class = obesity_class_from_bmi(bmi)

    if actual_gfr_mL_per_min is not None:
        obese_gfr = actual_gfr_mL_per_min
    else:
        obese_gfr = normal_gfr_mL_per_min * min(tbw / baseline_weight_kg, 1.5)

    renal_factor = obese_gfr / normal_gfr_mL_per_min

    if drug_type == "hydrophilic":
        vd_factor = lbm / 70.0
        hepatic_factor = 1.0 + 0.1 * (abw - baseline_weight_kg) / baseline_weight_kg
        hepatic_factor = max(0.5, min(hepatic_factor, 1.5))
        cl_factor = hepatic_factor * (1.0 - fe_renal) + renal_factor * fe_renal
        dose_basis = "ibw"
        recommended_dose = baseline_dose_mg * lbm / 70.0

    elif drug_type == "lipophilic":
        vd_factor = tbw / baseline_weight_kg
        hepatic_factor = 1.0 + 0.3 * (tbw - baseline_weight_kg) / baseline_weight_kg
        hepatic_factor = max(0.5, min(hepatic_factor, 1.5))
        cl_factor = hepatic_factor * (1.0 - fe_renal) + renal_factor * fe_renal
        dose_basis = "tbw"
        recommended_dose = baseline_dose_mg * tbw / baseline_weight_kg

    else:  # intermediate
        vd_factor = abw / baseline_weight_kg
        hepatic_factor = 1.0 + 0.2 * (abw - baseline_weight_kg) / baseline_weight_kg
        hepatic_factor = max(0.5, min(hepatic_factor, 1.5))
        cl_factor = hepatic_factor * (1.0 - fe_renal) + renal_factor * fe_renal
        dose_basis = "abw"
        recommended_dose = baseline_dose_mg * abw / baseline_weight_kg

    max_dose = baseline_dose_mg * 3.0
    recommended_dose = min(recommended_dose, max_dose)
    recommended_dose = max(recommended_dose, baseline_dose_mg * 0.1)

    vd_factor = max(vd_factor, 0.1)
    cl_factor = max(cl_factor, 0.1)

    adjusted_vd = baseline_vd_L * vd_factor
    adjusted_cl = baseline_cl_L_per_h * cl_factor
    adjusted_t_half = 0.6931 * adjusted_vd / adjusted_cl if adjusted_cl > 0 else 0.0

    notes_parts = [
        f"BMI={bmi:.1f} ({obesity_class})",
        f"IBW={ibw:.1f} kg, LBM={lbm:.1f} kg, ABW={abw:.1f} kg",
        f"Vd factor={vd_factor:.2f}, CL factor={cl_factor:.2f}",
        f"Drug type={drug_type}, dose basis={dose_basis}",
    ]
    notes = "; ".join(notes_parts)

    return ObesityPKResult(
        drug_name=drug_name,
        total_body_weight_kg=tbw,
        ideal_body_weight_kg=ibw,
        lean_body_mass_kg=lbm,
        adjusted_body_weight_kg=abw,
        bmi=bmi,
        obesity_class=obesity_class,
        vd_scaling_factor=vd_factor,
        cl_scaling_factor=cl_factor,
        adjusted_vd_L=adjusted_vd,
        adjusted_cl_L_per_h=adjusted_cl,
        adjusted_t_half_h=adjusted_t_half,
        recommended_dose_basis=dose_basis,
        recommended_dose_mg=recommended_dose,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Phase 419 functions (kept for backward compatibility)
# ---------------------------------------------------------------------------


def compute_lean_body_weight(height_cm: float, weight_kg: float, sex: str) -> float:
    """Compute lean body weight using the Devine formula.

    Parameters
    ----------
    height_cm:
        Patient height in centimetres. Must be > 0.
    weight_kg:
        Patient total body weight in kilograms. Must be > 0.
    sex:
        Biological sex: ``"male"`` or ``"female"``.

    Returns
    -------
    float
        Lean body weight in kilograms (clamped to minimum 30 kg).

    Raises
    ------
    ValueError
        If any input is out of range or ``sex`` is not recognised.
    """
    if height_cm <= 0:
        raise ValueError(f"height_cm must be > 0, got {height_cm}")
    if weight_kg <= 0:
        raise ValueError(f"weight_kg must be > 0, got {weight_kg}")
    sex_lower = sex.strip().lower()
    if sex_lower not in _VALID_SEXES:
        raise ValueError(f"sex must be one of {_VALID_SEXES}, got '{sex}'")

    height_in = height_cm / 2.54
    inches_over_60 = height_in - 60.0

    if sex_lower == "male":
        lbw = 50.0 + 2.3 * inches_over_60
    else:
        lbw = 45.5 + 2.3 * inches_over_60

    return max(30.0, lbw)


def _bmi_category(bmi: float) -> str:
    """Return BMI category string."""
    if bmi < 25.0:
        return "normal"
    elif bmi < 30.0:
        return "overweight"
    elif bmi <= 40.0:
        return "obese"
    else:
        return "morbidly_obese"


def obesity_pk_scaling(
    drug_name: str,
    normal_cl_L_per_h: float,
    normal_vd_L: float,
    bmi: float,
    height_cm: float,
    weight_kg: float,
    sex: str,
    distribution_type: str,
) -> ObesityPKScalingResult:
    """Compute obesity-adjusted PK parameters and dosing recommendation (Phase 419)."""
    if not drug_name or not drug_name.strip():
        raise ValueError("drug_name must be a non-empty string")
    if normal_cl_L_per_h <= 0:
        raise ValueError(f"normal_cl_L_per_h must be > 0, got {normal_cl_L_per_h}")
    if normal_vd_L <= 0:
        raise ValueError(f"normal_vd_L must be > 0, got {normal_vd_L}")
    if bmi <= 0:
        raise ValueError(f"bmi must be > 0, got {bmi}")
    if height_cm <= 0:
        raise ValueError(f"height_cm must be > 0, got {height_cm}")
    if weight_kg <= 0:
        raise ValueError(f"weight_kg must be > 0, got {weight_kg}")

    dist_lower = distribution_type.strip().lower()
    if dist_lower not in _VALID_DISTRIBUTION_TYPES:
        raise ValueError(
            f"distribution_type must be one of {_VALID_DISTRIBUTION_TYPES}, "
            f"got '{distribution_type}'"
        )

    lbw_kg = compute_lean_body_weight(height_cm, weight_kg, sex)
    tbw_kg = weight_kg
    category = _bmi_category(bmi)

    cl_adjusted = normal_cl_L_per_h * (lbw_kg / 70.0)

    if dist_lower == "hydrophilic":
        effective_weight = lbw_kg
    elif dist_lower == "lipophilic":
        effective_weight = tbw_kg
    else:  # mixed
        excess_fat = tbw_kg - lbw_kg
        effective_weight = lbw_kg + 0.3 * excess_fat

    vd_adjusted = normal_vd_L * (effective_weight / 70.0)

    t_half_h = (math.log(2.0) * vd_adjusted) / cl_adjusted if cl_adjusted > 0 else float("nan")

    cl_ratio = cl_adjusted / normal_cl_L_per_h
    vd_ratio = vd_adjusted / normal_vd_L

    notes_parts: list[str] = [
        f"BMI={bmi:.1f} kg/m² ({category}); LBW={lbw_kg:.1f} kg; TBW={tbw_kg:.1f} kg.",
        f"CL scaled by LBW/70={lbw_kg / 70.0:.3f} (ratio={cl_ratio:.2f}).",
        f"Vd scaled for {dist_lower} distribution (ratio={vd_ratio:.2f}).",
    ]

    if category == "normal":
        rec = (
            f"Use standard dose for {drug_name}. "
            "BMI in normal range; no obesity-based adjustment required."
        )
    elif category == "overweight":
        rec = (
            f"Consider LBW-based dose for {drug_name}. "
            f"CL adjustment factor {cl_ratio:.2f}; monitor response."
        )
    elif category == "obese":
        rec = (
            f"Use LBW-based CL dose adjustment for {drug_name} "
            f"(factor {cl_ratio:.2f}). "
            f"Vd adjustment factor {vd_ratio:.2f}; therapeutic drug monitoring recommended."
        )
    else:  # morbidly_obese
        rec = (
            f"Morbid obesity: use LBW-based CL dosing for {drug_name} "
            f"(factor {cl_ratio:.2f}). "
            f"Vd adjustment factor {vd_ratio:.2f}. "
            "Close TDM and clinical monitoring are essential."
        )

    return ObesityPKScalingResult(
        drug_name=drug_name,
        bmi=bmi,
        bmi_category=category,
        lbw_kg=lbw_kg,
        tbw_kg=tbw_kg,
        distribution_type=dist_lower,
        cl_adjusted_L_per_h=cl_adjusted,
        vd_adjusted_L=vd_adjusted,
        t_half_adjusted_h=t_half_h,
        dose_recommendation=rec,
        notes="; ".join(notes_parts),
    )
