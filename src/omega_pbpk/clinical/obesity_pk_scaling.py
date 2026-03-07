"""Phase 180 — PK parameter scaling in obese patients.

Uses lean body weight and fat mass adjustments for volume of distribution,
hepatic clearance, and renal function.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ObesityPKResult:
    drug_name: str
    sex: str
    actual_weight_kg: float
    lean_body_weight_kg: float
    fat_mass_kg: float
    bmi: float
    height_cm: float
    dosing_weight_kg: float
    vd_normal_L: float
    vd_obese_L: float
    cl_hepatic_normal_L_per_h: float
    cl_hepatic_obese_L_per_h: float
    t_half_normal_h: float
    t_half_obese_h: float
    dose_recommendation: str  # "use_lbw"/"use_adjbw"/"use_abw" based on logP
    obesity_class: (
        str  # "normal"(BMI<25), "overweight"(25-30), "obese"(30-40), "morbidly_obese"(>40)
    )
    notes: str


def calculate_lean_body_weight(sex: str, weight_kg: float, height_cm: float) -> float:
    """Calculate lean body weight using Hume equations.

    Args:
        sex: "male" or "female"
        weight_kg: Total body weight in kg.
        height_cm: Height in centimeters.

    Returns:
        Lean body weight in kg.
    """
    if sex.lower() == "male":
        lbw = 0.407 * weight_kg + 0.267 * height_cm - 19.2
    elif sex.lower() == "female":
        lbw = 0.252 * weight_kg + 0.473 * height_cm - 48.3
    else:
        raise ValueError("sex must be 'male' or 'female'")

    # Clamp: LBW must be between 10% of actual weight and actual weight
    lbw_min = 0.1 * weight_kg
    lbw_clamped = max(lbw_min, min(weight_kg, lbw))
    return lbw_clamped


def _classify_obesity(bmi: float) -> str:
    if bmi < 25.0:
        return "normal"
    elif bmi < 30.0:
        return "overweight"
    elif bmi <= 40.0:
        return "obese"
    else:
        return "morbidly_obese"


def _dose_recommendation_from_logp(logP: float) -> str:
    """Determine dosing weight recommendation based on lipophilicity."""
    if logP >= 2.0:
        # Lipophilic: distributes into fat — use adjusted body weight
        return "use_adjbw"
    elif logP <= 0.0:
        # Hydrophilic: does not distribute into fat — use lean body weight
        return "use_lbw"
    else:
        # Intermediate lipophilicity
        return "use_adjbw"


def scale_pk_for_obesity(
    drug_name: str,
    sex: str,
    weight_kg: float,
    height_cm: float,
    logP: float,
    cl_hepatic_L_per_h: float,
    vd_normal_L: float,
    fu_plasma: float,
    dosing_weight_method: str = "actual_body_weight",
) -> ObesityPKResult:
    """Scale PK parameters for an obese patient.

    Args:
        drug_name: Drug name.
        sex: "male" or "female".
        weight_kg: Actual body weight in kg.
        height_cm: Height in cm.
        logP: Octanol-water partition coefficient.
        cl_hepatic_L_per_h: Hepatic clearance in normal patient (L/h).
        vd_normal_L: Volume of distribution in normal patient (L).
        fu_plasma: Fraction unbound in plasma (0-1).
        dosing_weight_method: "actual_body_weight", "lean_body_weight", or
            "adjusted_body_weight".

    Returns:
        ObesityPKResult with scaled PK parameters.
    """
    # Validation
    if weight_kg <= 0:
        raise ValueError("weight_kg must be > 0")
    if height_cm <= 0:
        raise ValueError("height_cm must be > 0")
    if sex.lower() not in {"male", "female"}:
        raise ValueError("sex must be 'male' or 'female'")
    if cl_hepatic_L_per_h <= 0:
        raise ValueError("cl_hepatic_L_per_h must be > 0")
    if vd_normal_L <= 0:
        raise ValueError("vd_normal_L must be > 0")
    if not (-5.0 <= logP <= 10.0):
        raise ValueError("logP must be in [-5, 10]")
    if dosing_weight_method not in {
        "actual_body_weight",
        "lean_body_weight",
        "adjusted_body_weight",
    }:
        raise ValueError(
            "dosing_weight_method must be 'actual_body_weight', "
            "'lean_body_weight', or 'adjusted_body_weight'"
        )

    # Derived parameters
    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m * height_m)
    lbw = calculate_lean_body_weight(sex.lower(), weight_kg, height_cm)
    fat_mass = max(0.0, weight_kg - lbw)

    # Adjusted body weight
    adj_bw = lbw + 0.4 * fat_mass

    # Dosing weight selection
    if dosing_weight_method == "actual_body_weight":
        dosing_weight_kg = weight_kg
    elif dosing_weight_method == "lean_body_weight":
        dosing_weight_kg = lbw
    else:  # adjusted_body_weight
        dosing_weight_kg = adj_bw

    # Vd scaling
    # Vd_obese = Vd_normal * (LBW/70 + logP_factor * fat_mass/70)
    logP_factor = max(0.0, logP / 5.0)
    vd_obese = vd_normal_L * (lbw / 70.0 + logP_factor * fat_mass / 70.0)
    # Ensure Vd_obese >= Vd_normal * (LBW/70) — cannot be less than lean contribution
    vd_min = vd_normal_L * (lbw / 70.0)
    if vd_obese < vd_min:
        vd_obese = vd_min

    # CL_hepatic scaling
    # CL_factor = 1 + 0.1 * min(1, (BMI-30)/20) if BMI > 30 else 1.0
    if bmi > 30.0:
        cl_factor = 1.0 + 0.1 * min(1.0, (bmi - 30.0) / 20.0)
    else:
        cl_factor = 1.0
    cl_hepatic_obese = cl_hepatic_L_per_h * cl_factor

    # Half-life: t½ = 0.693 * Vd / CL
    t_half_normal = 0.693 * vd_normal_L / cl_hepatic_L_per_h
    t_half_obese = 0.693 * vd_obese / cl_hepatic_obese

    obesity_class = _classify_obesity(bmi)
    dose_recommendation = _dose_recommendation_from_logp(logP)

    # Build notes
    notes_parts = []
    ratio = vd_obese / vd_normal_L if vd_normal_L > 0 else 1.0
    notes_parts.append(f"BMI={bmi:.1f} ({obesity_class}).")
    notes_parts.append(f"LBW={lbw:.1f} kg, fat mass={fat_mass:.1f} kg.")
    notes_parts.append(f"Vd ratio (obese/normal)={ratio:.2f} (logP={logP:.1f}).")
    if logP >= 2.0:
        notes_parts.append(
            "Lipophilic drug: significant fat distribution, use adjusted body weight for dosing."
        )
    elif logP <= 0.0:
        notes_parts.append(
            "Hydrophilic drug: minimal fat distribution, use lean body weight for dosing."
        )
    else:
        notes_parts.append("Intermediate lipophilicity: use adjusted body weight.")
    notes = " ".join(notes_parts)

    return ObesityPKResult(
        drug_name=drug_name,
        sex=sex.lower(),
        actual_weight_kg=weight_kg,
        lean_body_weight_kg=lbw,
        fat_mass_kg=fat_mass,
        bmi=bmi,
        height_cm=height_cm,
        dosing_weight_kg=dosing_weight_kg,
        vd_normal_L=vd_normal_L,
        vd_obese_L=vd_obese,
        cl_hepatic_normal_L_per_h=cl_hepatic_L_per_h,
        cl_hepatic_obese_L_per_h=cl_hepatic_obese,
        t_half_normal_h=t_half_normal,
        t_half_obese_h=t_half_obese,
        dose_recommendation=dose_recommendation,
        obesity_class=obesity_class,
        notes=notes,
    )
