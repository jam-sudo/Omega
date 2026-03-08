"""Phase 564 - Allometric Scaling for Pediatric Dosing.

Predicts pediatric PK parameters from adult values using allometric scaling
with optional maturation functions (Anderson-Holford model).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AllometricPediatricResult:
    """Result of allometric pediatric dose scaling."""

    drug_name: str
    age_years: float
    weight_kg: float
    cl_adult_L_per_h: float
    vd_adult_L: float
    cl_pediatric_L_per_h: float
    vd_pediatric_L: float
    t_half_h: float
    dose_per_kg_mg: float
    dose_total_mg: float
    scaling_method: str
    maturation_factor: float
    notes: str


def _maturation_factor(age_years: float) -> float:
    """Anderson-Holford maturation function.

    Returns maturation factor (0-1) based on post-menstrual age.
    """
    if age_years >= 2.0:
        return 1.0
    pma_weeks = age_years * 52.0 + 40.0
    mf = pma_weeks**3.4 / (pma_weeks**3.4 + 55.4**3.4)
    return mf


def scale_pediatric_dose(
    drug_name: str,
    age_years: float,
    weight_kg: float,
    cl_adult_L_per_h: float,
    vd_adult_L: float,
    target_auc_mg_h_per_L: float,
    adult_weight_kg: float = 70.0,
    use_maturation: bool = True,
) -> AllometricPediatricResult:
    """Scale adult PK parameters to pediatric using allometric scaling.

    Args:
        drug_name: Name of the drug.
        age_years: Patient age in years (>= 0).
        weight_kg: Patient weight in kg (> 0).
        cl_adult_L_per_h: Adult clearance in L/h (> 0).
        vd_adult_L: Adult volume of distribution in L (> 0).
        target_auc_mg_h_per_L: Target AUC in mg*h/L (> 0).
        adult_weight_kg: Reference adult weight in kg.
        use_maturation: Whether to apply maturation function.

    Returns:
        AllometricPediatricResult with scaled parameters.
    """
    if weight_kg <= 0:
        raise ValueError(f"weight_kg must be > 0, got {weight_kg}")
    if age_years < 0:
        raise ValueError(f"age_years must be >= 0, got {age_years}")
    if cl_adult_L_per_h <= 0:
        raise ValueError(f"cl_adult_L_per_h must be > 0, got {cl_adult_L_per_h}")
    if vd_adult_L <= 0:
        raise ValueError(f"vd_adult_L must be > 0, got {vd_adult_L}")
    if target_auc_mg_h_per_L <= 0:
        raise ValueError(f"target_auc_mg_h_per_L must be > 0, got {target_auc_mg_h_per_L}")

    # Simple allometric scaling
    weight_ratio = weight_kg / adult_weight_kg
    cl_ped = cl_adult_L_per_h * (weight_ratio**0.75)
    vd_ped = vd_adult_L * (weight_ratio**1.0)

    if use_maturation:
        mf = _maturation_factor(age_years)
        cl_ped_final = cl_ped * mf
        scaling_method = "with_maturation"
        notes = f"Allometric scaling with Anderson-Holford maturation (MF={mf:.3f})"
    else:
        mf = 1.0
        cl_ped_final = cl_ped
        scaling_method = "simple_allometric"
        notes = "Simple allometric scaling without maturation adjustment"

    vd_ped_final = vd_ped  # Vd not affected by maturation

    t_half = 0.693 * vd_ped_final / cl_ped_final
    dose_total = target_auc_mg_h_per_L * cl_ped_final
    dose_per_kg = dose_total / weight_kg

    return AllometricPediatricResult(
        drug_name=drug_name,
        age_years=age_years,
        weight_kg=weight_kg,
        cl_adult_L_per_h=cl_adult_L_per_h,
        vd_adult_L=vd_adult_L,
        cl_pediatric_L_per_h=cl_ped_final,
        vd_pediatric_L=vd_ped_final,
        t_half_h=t_half,
        dose_per_kg_mg=dose_per_kg,
        dose_total_mg=dose_total,
        scaling_method=scaling_method,
        maturation_factor=mf,
        notes=notes,
    )


def screen_age_groups(
    drug_name: str,
    cl_adult: float,
    vd_adult: float,
    target_auc: float,
) -> list:
    """Screen standard pediatric age groups.

    Args:
        drug_name: Name of the drug.
        cl_adult: Adult clearance in L/h.
        vd_adult: Adult volume of distribution in L.
        target_auc: Target AUC in mg*h/L.

    Returns:
        List of AllometricPediatricResult sorted by age_years ascending.
    """
    age_weight_groups = [
        (0.25, 5.0),
        (1.0, 10.0),
        (5.0, 20.0),
        (10.0, 35.0),
        (15.0, 60.0),
    ]
    results = []
    for age, weight in age_weight_groups:
        result = scale_pediatric_dose(
            drug_name=drug_name,
            age_years=age,
            weight_kg=weight,
            cl_adult_L_per_h=cl_adult,
            vd_adult_L=vd_adult,
            target_auc_mg_h_per_L=target_auc,
        )
        results.append(result)
    results.sort(key=lambda r: r.age_years)
    return results
