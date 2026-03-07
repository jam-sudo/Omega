"""Pediatric renal maturation model for age-dependent renal clearance.

Models age-dependent maturation of renal drug clearance in pediatric patients
using postmenstrual age (PMA)-based sigmoidal Hill maturation function
(Rhodin 2009) combined with allometric body weight scaling.

References:
    Rhodin MM et al., Pediatr Nephrol 24:67-76, 2009 (GFR Hill maturation)
    Anderson BJ & Holford NHG, Annu Rev Pharmacol Toxicol, 2008
"""

from __future__ import annotations

__all__ = [
    "PediatricRenalCLResult",
    "calculate_pediatric_renal_cl",
    "compare_age_groups",
]

from dataclasses import dataclass


@dataclass(frozen=True)
class PediatricRenalCLResult:
    """Result of pediatric renal clearance calculation.

    Attributes:
        drug_name: Name of the drug.
        age_years: Patient age in years.
        weight_kg: Patient weight in kg.
        sex: Biological sex ("male" or "female").
        allometric_cl_L_per_h: Allometrically scaled renal CL (L/h).
        maturation_factor: GFR maturation fraction (0, 1].
        maturation_pct: Maturation expressed as percentage (0-100).
        cl_renal_L_per_h: Final pediatric renal CL after maturation + sex correction (L/h).
        cl_renal_per_kg_L_per_h_per_kg: Renal CL normalized per body weight (L/h/kg).
        age_class: Age classification string.
        dose_adjustment_factor: cl_renal_pediatric / adult_cl_renal.
        notes: Descriptive notes.
    """

    drug_name: str
    age_years: float
    weight_kg: float
    sex: str
    allometric_cl_L_per_h: float
    maturation_factor: float
    maturation_pct: float
    cl_renal_L_per_h: float
    cl_renal_per_kg_L_per_h_per_kg: float
    age_class: str
    dose_adjustment_factor: float
    notes: str


def _pma_weeks_from_age_years(age_years: float) -> float:
    """Estimate postmenstrual age in weeks from age in years.

    Assumes term birth (gestational age = 40 weeks).

    Parameters
    ----------
    age_years:
        Postnatal age in years.

    Returns
    -------
    float
        PMA in weeks.
    """
    return age_years * 52.0 + 40.0


def _mf_gfr(pma_weeks: float) -> float:
    """Sigmoid Hill maturation function for GFR (Rhodin 2009).

    MF_GFR = PMA^3.4 / (PMA^3.4 + 47.7^3.4)

    Parameters
    ----------
    pma_weeks:
        Postmenstrual age in weeks.

    Returns
    -------
    float
        Maturation factor in (0, 1].
    """
    hill_pma = pma_weeks**3.4
    hill_50 = 47.7**3.4
    mf = hill_pma / (hill_pma + hill_50)
    return min(mf, 1.0)


def _age_class(age_years: float) -> str:
    """Classify age group.

    Parameters
    ----------
    age_years:
        Age in years.

    Returns
    -------
    str
        One of "neonate", "infant", "toddler", "child", "adolescent".
    """
    if age_years < 1.0 / 12.0:  # < 1 month
        return "neonate"
    if age_years < 1.0:  # 1-12 months
        return "infant"
    if age_years <= 3.0:  # 1-3 years
        return "toddler"
    if age_years <= 12.0:  # 3-12 years
        return "child"
    return "adolescent"


def calculate_pediatric_renal_cl(
    drug_name: str,
    adult_cl_renal_L_per_h: float,
    age_years: float,
    weight_kg: float,
    adult_weight_kg: float,
    sex: str,
) -> PediatricRenalCLResult:
    """Calculate pediatric renal clearance with maturation scaling.

    Combines allometric body weight scaling with PMA-based sigmoidal
    Hill maturation function to predict renal CL in pediatric patients.
    Applies sex correction for female adolescents (age > 12 years).

    Parameters
    ----------
    drug_name:
        Name of the drug.
    adult_cl_renal_L_per_h:
        Adult renal clearance in L/h (> 0).
    age_years:
        Patient age in years (>= 0).
    weight_kg:
        Patient body weight in kg (> 0).
    adult_weight_kg:
        Reference adult body weight in kg (> 0).
    sex:
        Biological sex: "male" or "female".

    Returns
    -------
    PediatricRenalCLResult
    """
    # --- Validation ---
    if adult_cl_renal_L_per_h <= 0.0:
        raise ValueError(f"adult_cl_renal_L_per_h must be > 0, got {adult_cl_renal_L_per_h}")
    if age_years < 0.0:
        raise ValueError(f"age_years must be >= 0, got {age_years}")
    if weight_kg <= 0.0:
        raise ValueError(f"weight_kg must be > 0, got {weight_kg}")
    if adult_weight_kg <= 0.0:
        raise ValueError(f"adult_weight_kg must be > 0, got {adult_weight_kg}")
    if sex not in ("male", "female"):
        raise ValueError(f"sex must be 'male' or 'female', got {sex!r}")

    # Allometric scaling: CL_allo = adult_CL * (weight/adult_weight)^0.75
    allometric_cl = adult_cl_renal_L_per_h * (weight_kg / adult_weight_kg) ** 0.75

    # PMA-based maturation function (Rhodin 2009)
    pma_weeks = _pma_weeks_from_age_years(age_years)
    mf = _mf_gfr(pma_weeks)

    # Sex correction: female adolescents have ~15% lower renal CL
    if sex == "female" and age_years > 12.0:
        sex_factor = 0.85
    else:
        sex_factor = 1.0

    # Final pediatric renal CL
    cl_renal_pediatric = allometric_cl * mf * sex_factor

    # Normalized per kg
    cl_renal_per_kg = cl_renal_pediatric / weight_kg

    # Age class
    ac = _age_class(age_years)

    # Dose adjustment factor
    dose_adj = cl_renal_pediatric / adult_cl_renal_L_per_h

    notes = (
        f"drug={drug_name}; age={age_years:.3f}y; PMA={pma_weeks:.1f}w; "
        f"MF_GFR={mf:.4f}; allo_cl={allometric_cl:.4f}L/h; "
        f"sex_factor={sex_factor:.2f}; cl_renal={cl_renal_pediatric:.4f}L/h; "
        f"age_class={ac}"
    )

    return PediatricRenalCLResult(
        drug_name=drug_name,
        age_years=age_years,
        weight_kg=weight_kg,
        sex=sex,
        allometric_cl_L_per_h=allometric_cl,
        maturation_factor=mf,
        maturation_pct=mf * 100.0,
        cl_renal_L_per_h=cl_renal_pediatric,
        cl_renal_per_kg_L_per_h_per_kg=cl_renal_per_kg,
        age_class=ac,
        dose_adjustment_factor=dose_adj,
        notes=notes,
    )


def compare_age_groups(
    drug_name: str,
    adult_cl_renal_L_per_h: float,
    adult_weight_kg: float,
    sex: str,
    age_years_list: list[float],
    weight_kg_list: list[float],
) -> list[PediatricRenalCLResult]:
    """Compare renal CL across age groups.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    adult_cl_renal_L_per_h:
        Adult renal clearance in L/h (> 0).
    adult_weight_kg:
        Reference adult body weight in kg (> 0).
    sex:
        Biological sex: "male" or "female".
    age_years_list:
        List of ages in years.
    weight_kg_list:
        List of body weights in kg (must match length of age_years_list).

    Returns
    -------
    list[PediatricRenalCLResult]
        Results sorted by cl_renal_L_per_h ascending (lowest first).
    """
    if len(age_years_list) != len(weight_kg_list):
        raise ValueError(
            f"age_years_list ({len(age_years_list)}) and weight_kg_list "
            f"({len(weight_kg_list)}) must have the same length."
        )

    results = [
        calculate_pediatric_renal_cl(
            drug_name=drug_name,
            adult_cl_renal_L_per_h=adult_cl_renal_L_per_h,
            age_years=age,
            weight_kg=wt,
            adult_weight_kg=adult_weight_kg,
            sex=sex,
        )
        for age, wt in zip(age_years_list, weight_kg_list)
    ]

    return sorted(results, key=lambda r: r.cl_renal_L_per_h)
