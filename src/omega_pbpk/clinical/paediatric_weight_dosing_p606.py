"""
Phase 606 — Paediatric Weight-Based Dosing
Calculate and optimize weight-based dosing for pediatric patients across age groups.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PaediatricWeightDosingResult:
    drug_name: str
    age_years: float
    weight_kg: float
    age_group: str  # "neonate", "infant", "child", "adolescent"
    mg_per_kg: float  # calculated dose per kg
    total_dose_mg: float  # mg_per_kg * weight_kg
    max_dose_mg: float  # capped at adult dose
    recommended_dose_mg: float  # min(total_dose_mg, max_dose_mg)
    cl_scaling_factor: float  # CL relative to adult
    vd_scaling_factor: float  # Vd relative to adult
    t_half_paediatric_h: float  # adjusted half-life
    dosing_interval_h: float  # recommended interval
    notes: str


def _classify_age_group(age_years: float) -> str:
    if age_years < 0.08:
        return "neonate"
    elif age_years < 2.0:
        return "infant"
    elif age_years < 12.0:
        return "child"
    else:
        return "adolescent"


def _cl_factor_for_group(age_group: str) -> float:
    factors = {
        "neonate": 0.25,
        "infant": 1.5,
        "child": 1.2,
        "adolescent": 1.0,
    }
    return factors[age_group]


def _vd_factor_for_group(age_group: str) -> float:
    factors = {
        "neonate": 1.4,
        "infant": 1.2,
        "child": 1.1,
        "adolescent": 1.0,
    }
    return factors[age_group]


def calculate_paediatric_dose(
    drug_name: str,
    adult_dose_mg: float,
    adult_weight_kg: float,
    age_years: float,
    weight_kg: float,
    t_half_adult_h: float,
    mg_per_kg_max: float = None,
) -> PaediatricWeightDosingResult:
    """
    Calculate weight-based paediatric dose.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    adult_dose_mg : float
        Standard adult dose in mg.
    adult_weight_kg : float
        Reference adult weight in kg.
    age_years : float
        Patient age in years.
    weight_kg : float
        Patient weight in kg.
    t_half_adult_h : float
        Adult half-life in hours.
    mg_per_kg_max : float, optional
        Maximum allowed dose per kg. If provided, mg_per_kg is capped.

    Returns
    -------
    PaediatricWeightDosingResult
    """
    # Validation
    if age_years < 0:
        raise ValueError("age_years must be >= 0")
    if weight_kg <= 0:
        raise ValueError("weight_kg must be > 0")
    if adult_dose_mg <= 0:
        raise ValueError("adult_dose_mg must be > 0")
    if adult_weight_kg <= 0:
        raise ValueError("adult_weight_kg must be > 0")
    if t_half_adult_h <= 0:
        raise ValueError("t_half_adult_h must be > 0")

    age_group = _classify_age_group(age_years)
    cl_factor = _cl_factor_for_group(age_group)
    vd_factor = _vd_factor_for_group(age_group)

    adult_mg_per_kg = adult_dose_mg / adult_weight_kg
    mg_per_kg = adult_mg_per_kg * cl_factor

    if mg_per_kg_max is not None:
        mg_per_kg = min(mg_per_kg, mg_per_kg_max)

    total_dose_mg = mg_per_kg * weight_kg
    max_dose_mg = adult_dose_mg
    recommended_dose_mg = min(total_dose_mg, max_dose_mg)

    t_half_paediatric_h = t_half_adult_h * vd_factor / cl_factor
    dosing_interval_h = max(4.0, t_half_paediatric_h * 1.5)

    notes = (
        f"Age group: {age_group}. "
        f"CL scaling factor: {cl_factor:.2f}, Vd scaling factor: {vd_factor:.2f}. "
        f"Recommended dose: {recommended_dose_mg:.2f} mg every {dosing_interval_h:.1f} h."
    )

    return PaediatricWeightDosingResult(
        drug_name=drug_name,
        age_years=age_years,
        weight_kg=weight_kg,
        age_group=age_group,
        mg_per_kg=mg_per_kg,
        total_dose_mg=total_dose_mg,
        max_dose_mg=max_dose_mg,
        recommended_dose_mg=recommended_dose_mg,
        cl_scaling_factor=cl_factor,
        vd_scaling_factor=vd_factor,
        t_half_paediatric_h=t_half_paediatric_h,
        dosing_interval_h=dosing_interval_h,
        notes=notes,
    )


# Representative ages and standard weights for dose_across_ages
_REPRESENTATIVE_AGES_YEARS = [0.03, 0.5, 5.0, 15.0]
_REPRESENTATIVE_WEIGHTS_KG = [2.0, 8.0, 18.0, 55.0]


def dose_across_ages(
    drug_name: str,
    adult_dose_mg: float,
    adult_weight_kg: float,
    t_half_adult_h: float,
) -> list:
    """
    Calculate paediatric doses across representative age groups.

    Uses standard weight-for-age: ages [0.03, 0.5, 5.0, 15.0] years
    with weights [2.0, 8.0, 18.0, 55.0] kg.

    Returns a list of PaediatricWeightDosingResult.
    """
    results = []
    for age, weight in zip(_REPRESENTATIVE_AGES_YEARS, _REPRESENTATIVE_WEIGHTS_KG):
        result = calculate_paediatric_dose(
            drug_name=drug_name,
            adult_dose_mg=adult_dose_mg,
            adult_weight_kg=adult_weight_kg,
            age_years=age,
            weight_kg=weight,
            t_half_adult_h=t_half_adult_h,
        )
        results.append(result)
    return results
