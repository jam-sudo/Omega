"""Phase 581 - Drug Distribution in Obesity.

Models how obesity affects drug distribution, clearance, and dosing
for lipophilic and hydrophilic drugs.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ObesityPKResult:
    drug_name: str
    total_weight_kg: float
    ibw_kg: float
    lbw_kg: float
    bmi: float
    obesity_class: str
    dosing_weight_kg: float
    vd_adjusted_L: float
    cl_adjusted_L_per_h: float
    t_half_adjusted_h: float
    dose_adjustment_factor: float
    dosing_strategy: str
    notes: str


def _validate_inputs(total_weight_kg: float, height_cm: float, sex: str) -> None:
    if total_weight_kg <= 0:
        raise ValueError("total_weight_kg must be > 0")
    if height_cm <= 0:
        raise ValueError("height_cm must be > 0")
    if sex not in ("male", "female"):
        raise ValueError("sex must be 'male' or 'female'")


def _calc_ibw(height_cm: float, sex: str) -> float:
    height_in = height_cm / 2.54
    if sex == "male":
        ibw = 50.0 + 2.3 * (height_in - 60.0)
    else:
        ibw = 45.5 + 2.3 * (height_in - 60.0)
    return max(40.0, min(120.0, ibw))


def _calc_bmi(weight_kg: float, height_cm: float) -> float:
    height_m = height_cm / 100.0
    return weight_kg / (height_m * height_m)


def _calc_lbw(weight_kg: float, bmi: float, sex: str, ibw: float) -> float:
    if sex == "male":
        lbw = 9270.0 * weight_kg / (6680.0 + 216.0 * bmi)
    else:
        lbw = 9270.0 * weight_kg / (8780.0 + 244.0 * bmi)
    return max(0.5 * ibw, min(ibw, lbw))


def _calc_abw(ibw: float, total_weight_kg: float) -> float:
    abw = ibw + 0.4 * (total_weight_kg - ibw)
    return max(ibw, min(total_weight_kg, abw))


def _classify_obesity(bmi: float) -> str:
    if bmi < 25.0:
        return "normal"
    elif bmi < 30.0:
        return "overweight"
    elif bmi < 35.0:
        return "obese_I"
    elif bmi < 40.0:
        return "obese_II"
    else:
        return "obese_III"


def assess_obesity_pk(
    drug_name: str,
    total_weight_kg: float,
    height_cm: float,
    sex: str,
    logP: float,
    cl_normal_L_per_h: float,
    vd_normal_L_per_kg_ibw: float = 0.5,
) -> ObesityPKResult:
    _validate_inputs(total_weight_kg, height_cm, sex)

    ibw = _calc_ibw(height_cm, sex)
    bmi = _calc_bmi(total_weight_kg, height_cm)
    lbw = _calc_lbw(total_weight_kg, bmi, sex, ibw)
    abw = _calc_abw(ibw, total_weight_kg)
    obesity_class = _classify_obesity(bmi)

    # Volume of distribution
    if logP > 2.0:
        vd_adjusted = vd_normal_L_per_kg_ibw * ibw + max(0.0, logP - 2.0) * 0.3 * (
            total_weight_kg - ibw
        )
    else:
        vd_adjusted = vd_normal_L_per_kg_ibw * ibw

    # Clearance adjusted by lean body weight
    cl_adjusted = cl_normal_L_per_h * (lbw / ibw)

    # Dosing strategy
    if logP > 2.0:
        dosing_strategy = "adjusted_body_weight"
        dosing_weight = abw
    else:
        dosing_strategy = "lean_body_weight"
        dosing_weight = lbw

    dose_adjustment_factor = dosing_weight / ibw

    # Half-life
    t_half = 0.693 * vd_adjusted / cl_adjusted if cl_adjusted > 0 else float("inf")

    notes = (
        f"BMI {bmi:.1f} ({obesity_class}); dosing by {dosing_strategy} at {dosing_weight:.1f} kg"
    )

    return ObesityPKResult(
        drug_name=drug_name,
        total_weight_kg=total_weight_kg,
        ibw_kg=ibw,
        lbw_kg=lbw,
        bmi=bmi,
        obesity_class=obesity_class,
        dosing_weight_kg=dosing_weight,
        vd_adjusted_L=vd_adjusted,
        cl_adjusted_L_per_h=cl_adjusted,
        t_half_adjusted_h=t_half,
        dose_adjustment_factor=dose_adjustment_factor,
        dosing_strategy=dosing_strategy,
        notes=notes,
    )


def compare_obesity_levels(
    drug_name: str,
    height_cm: float,
    sex: str,
    logP: float,
    cl_normal: float,
    vd_normal_per_kg: float,
    weights: list = None,
) -> list:
    if weights is None:
        weights = [70, 90, 110, 130, 150]

    results = []
    for w in weights:
        result = assess_obesity_pk(
            drug_name=drug_name,
            total_weight_kg=w,
            height_cm=height_cm,
            sex=sex,
            logP=logP,
            cl_normal_L_per_h=cl_normal,
            vd_normal_L_per_kg_ibw=vd_normal_per_kg,
        )
        results.append(result)

    results.sort(key=lambda r: r.bmi)
    return results
