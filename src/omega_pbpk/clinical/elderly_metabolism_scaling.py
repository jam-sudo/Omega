"""Phase 567 - Drug Metabolism Age Scaling (Adult to Elderly).

Models age-dependent decline in drug metabolism and elimination
from adult (30y) to elderly (80y+).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ElderlyMetabolismResult:
    drug_name: str
    age_years: int
    weight_kg: float
    sex: str
    cl_adult_L_per_h: float
    cl_elderly_L_per_h: float
    vd_adult_L: float
    vd_elderly_L: float
    t_half_adult_h: float
    t_half_elderly_h: float
    dose_adjustment_factor: float
    gfr_elderly_mL_per_min: float
    cyp3a4_activity_factor: float
    aging_class: str
    notes: str


def _classify_aging(age_years: int) -> str:
    if age_years < 40:
        return "adult"
    elif age_years < 65:
        return "middle_aged"
    elif age_years < 75:
        return "young_old"
    elif age_years < 85:
        return "old"
    else:
        return "oldest_old"


def _compute_gfr(age_years: int) -> float:
    """CKD-EPI simplified GFR decline."""
    if age_years <= 30:
        return 120.0
    raw = 120.0 * max(0.3, 1.0 - 0.01 * (age_years - 30))
    return max(10.0, min(120.0, raw))


def _compute_cyp3a4_factor(age_years: int) -> float:
    """CYP3A4 activity factor: declines ~30% from 30 to 80."""
    if age_years <= 30:
        return 1.0
    return max(0.5, 1.0 - 0.007 * (age_years - 30))


def scale_elderly_metabolism(
    drug_name: str,
    age_years: int,
    weight_kg: float,
    sex: str,
    cl_adult_L_per_h: float,
    vd_adult_L: float,
    f_renal: float = 0.3,
    f_hepatic: float = 0.7,
) -> ElderlyMetabolismResult:
    if age_years <= 0:
        raise ValueError("age_years must be > 0")
    if weight_kg <= 0:
        raise ValueError("weight_kg must be > 0")
    if cl_adult_L_per_h <= 0:
        raise ValueError("cl_adult_L_per_h must be > 0")
    if vd_adult_L <= 0:
        raise ValueError("vd_adult_L must be > 0")
    if f_renal + f_hepatic <= 0 or f_renal + f_hepatic > 1.0:
        raise ValueError("f_renal + f_hepatic must be in (0, 1]")
    if f_renal < 0 or f_hepatic < 0:
        raise ValueError("f_renal and f_hepatic must be >= 0")
    if sex not in ("male", "female"):
        raise ValueError("sex must be 'male' or 'female'")

    gfr = _compute_gfr(age_years)
    gfr_ratio = gfr / 120.0

    cyp3a4_factor = _compute_cyp3a4_factor(age_years)

    cl_hepatic = cl_adult_L_per_h * f_hepatic * cyp3a4_factor
    cl_renal = cl_adult_L_per_h * f_renal * gfr_ratio
    cl_elderly = cl_hepatic + cl_renal

    # Vd scaling
    if age_years <= 30:
        vd_elderly = vd_adult_L
    else:
        vd_scale = max(0.8, 1.0 + 0.005 * (age_years - 30))
        vd_elderly = vd_adult_L * vd_scale
        vd_elderly = max(vd_adult_L * 0.8, min(vd_adult_L * 1.5, vd_elderly))

    t_half_adult = 0.693 * vd_adult_L / cl_adult_L_per_h
    t_half_elderly = 0.693 * vd_elderly / cl_elderly

    dose_adj = cl_elderly / cl_adult_L_per_h

    aging_class = _classify_aging(age_years)

    notes_parts = []
    if dose_adj < 0.5:
        notes_parts.append("significant dose reduction recommended")
    elif dose_adj < 0.75:
        notes_parts.append("moderate dose reduction recommended")
    if gfr < 30:
        notes_parts.append("severe renal impairment")
    elif gfr < 60:
        notes_parts.append("moderate renal impairment")
    if cyp3a4_factor < 0.7:
        notes_parts.append("reduced hepatic metabolism")
    notes = "; ".join(notes_parts) if notes_parts else "no special considerations"

    return ElderlyMetabolismResult(
        drug_name=drug_name,
        age_years=age_years,
        weight_kg=weight_kg,
        sex=sex,
        cl_adult_L_per_h=cl_adult_L_per_h,
        cl_elderly_L_per_h=cl_elderly,
        vd_adult_L=vd_adult_L,
        vd_elderly_L=vd_elderly,
        t_half_adult_h=t_half_adult,
        t_half_elderly_h=t_half_elderly,
        dose_adjustment_factor=dose_adj,
        gfr_elderly_mL_per_min=gfr,
        cyp3a4_activity_factor=cyp3a4_factor,
        aging_class=aging_class,
        notes=notes,
    )


def screen_age_effects(
    drug_name: str,
    cl_adult: float,
    vd_adult: float,
    ages: list = None,
) -> list:
    if ages is None:
        ages = [40, 55, 65, 75, 85]
    results = []
    for age in ages:
        r = scale_elderly_metabolism(
            drug_name=drug_name,
            age_years=age,
            weight_kg=70.0,
            sex="male",
            cl_adult_L_per_h=cl_adult,
            vd_adult_L=vd_adult,
        )
        results.append(r)
    results.sort(key=lambda x: x.age_years)
    return results
