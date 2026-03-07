"""PK parameter scaling for obese patients using BMI-based LBW/TBW adjustment (Phase 419)."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "ObesityPKResult",
    "compute_lean_body_weight",
    "obesity_pk_scaling",
]

_VALID_SEXES = {"male", "female"}
_VALID_DISTRIBUTION_TYPES = {"hydrophilic", "lipophilic", "mixed"}


@dataclass
class ObesityPKResult:
    """Result of obesity-adjusted PK parameter computation."""

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
) -> ObesityPKResult:
    """Compute obesity-adjusted PK parameters and dosing recommendation.

    Clearance is always scaled by LBW/70 (hepatic and renal function best
    correlate with lean body weight).  Volume of distribution is scaled by
    LBW, TBW, or a blended weight depending on ``distribution_type``.

    Parameters
    ----------
    drug_name:
        Identifier for the drug compound.
    normal_cl_L_per_h:
        Population-typical clearance in L/h (70 kg reference subject). Must be > 0.
    normal_vd_L:
        Population-typical volume of distribution in L (70 kg reference). Must be > 0.
    bmi:
        Patient body-mass index (kg/m²). Must be > 0.
    height_cm:
        Patient height in centimetres. Must be > 0.
    weight_kg:
        Patient total body weight in kilograms. Must be > 0.
    sex:
        Biological sex: ``"male"`` or ``"female"``.
    distribution_type:
        Drug distribution characteristic: ``"hydrophilic"``, ``"lipophilic"``,
        or ``"mixed"``.

    Returns
    -------
    ObesityPKResult
        Adjusted CL, Vd, t½, dosing recommendation and notes.

    Raises
    ------
    ValueError
        If any input is invalid.
    """
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

    # CL always scales with LBW
    cl_adjusted = normal_cl_L_per_h * (lbw_kg / 70.0)

    # Vd scaling depends on distribution type
    if dist_lower == "hydrophilic":
        effective_weight = lbw_kg
    elif dist_lower == "lipophilic":
        effective_weight = tbw_kg
    else:  # mixed
        excess_fat = tbw_kg - lbw_kg
        effective_weight = lbw_kg + 0.3 * excess_fat

    vd_adjusted = normal_vd_L * (effective_weight / 70.0)

    # Terminal half-life: t½ = 0.693 * Vd / CL
    t_half_h = (math.log(2.0) * vd_adjusted) / cl_adjusted if cl_adjusted > 0 else float("nan")

    # Dosing recommendation
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

    return ObesityPKResult(
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
