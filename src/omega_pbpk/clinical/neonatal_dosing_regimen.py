"""Phase 562 — Neonatal Drug Dosing Regimen.

Models PK and dosing recommendations for neonates (0-28 days postnatal age)
with immature organ function.
"""

import math
from dataclasses import dataclass

STANDARD_INTERVALS = [4, 6, 8, 12, 24, 36, 48]


@dataclass(frozen=True)
class NeonatalDosingResult:
    drug_name: str
    postnatal_age_days: int
    weight_kg: float
    gestational_age_weeks: int
    cl_mL_per_min_per_kg: float
    vd_L_per_kg: float
    t_half_h: float
    recommended_dose_mg_per_kg: float
    dosing_interval_h: float
    css_avg_mg_L: float
    dose_class: str
    maturation_stage: str
    notes: str


def _classify_maturation(ga_weeks: int) -> str:
    if ga_weeks < 28:
        return "extremely_premature"
    elif ga_weeks <= 36:
        return "premature"
    elif ga_weeks <= 41:
        return "term"
    return "post_term"


def _nearest_standard_interval(value: float) -> float:
    best = STANDARD_INTERVALS[0]
    best_diff = abs(value - best)
    for iv in STANDARD_INTERVALS[1:]:
        diff = abs(value - iv)
        if diff < best_diff:
            best_diff = diff
            best = iv
    return float(best)


def calculate_neonatal_dosing(
    drug_name: str,
    postnatal_age_days: int,
    weight_kg: float,
    gestational_age_weeks: int,
    target_css_mg_L: float,
    baseline_adult_cl_mL_per_min_per_kg: float = 1.0,
    baseline_adult_vd_L_per_kg: float = 0.5,
) -> NeonatalDosingResult:
    """Calculate neonatal dosing regimen based on maturation-adjusted PK."""
    if postnatal_age_days < 0:
        raise ValueError("postnatal_age_days must be >= 0")
    if weight_kg <= 0:
        raise ValueError("weight_kg must be > 0")
    if target_css_mg_L <= 0:
        raise ValueError("target_css_mg_L must be > 0")
    if gestational_age_weeks < 20:
        raise ValueError("gestational_age_weeks must be >= 20")

    # Post-menstrual age
    pma_weeks = gestational_age_weeks + postnatal_age_days / 7.0

    # CL maturation (Hill equation)
    cl_maturation = pma_weeks**3.4 / (pma_weeks**3.4 + 55.4**3.4)
    cl_mL_per_min_per_kg = baseline_adult_cl_mL_per_min_per_kg * cl_maturation

    # Vd maturation (higher in preterm due to higher TBW)
    vd_factor = 1.0 + max(0.0, (37 - gestational_age_weeks)) * 0.05
    vd_L_per_kg = baseline_adult_vd_L_per_kg * vd_factor
    vd_L_per_kg = max(baseline_adult_vd_L_per_kg, min(vd_L_per_kg, 3.0))

    # Half-life: 0.693 * Vd(L/kg) / (CL(mL/min/kg) * 60/1000)
    cl_L_per_h_per_kg = cl_mL_per_min_per_kg * 60.0 / 1000.0
    if cl_L_per_h_per_kg > 0:
        t_half_h = 0.693 * vd_L_per_kg / cl_L_per_h_per_kg
    else:
        t_half_h = float("inf")

    # Dosing interval: approximate t_half / ln(2), nearest standard
    if t_half_h < float("inf") and t_half_h > 0:
        ideal_interval = t_half_h / math.log(2)
        tau = _nearest_standard_interval(ideal_interval)
    else:
        tau = 48.0

    # Dose: D = target_css * CL(L/h/kg) * tau (fup=1.0)
    dose_mg_per_kg = target_css_mg_L * cl_L_per_h_per_kg * tau

    # Css_avg verification
    css_avg = dose_mg_per_kg / (cl_L_per_h_per_kg * tau) if cl_L_per_h_per_kg > 0 else 0.0

    dose_class = "loading_dose_needed" if t_half_h > 24.0 else "maintenance_only"
    maturation_stage = _classify_maturation(gestational_age_weeks)

    notes_parts = []
    if maturation_stage == "extremely_premature":
        notes_parts.append("Extremely premature: significantly reduced clearance")
    elif maturation_stage == "premature":
        notes_parts.append("Premature neonate: reduced clearance expected")
    if dose_class == "loading_dose_needed":
        notes_parts.append("Consider loading dose due to long half-life")
    notes = "; ".join(notes_parts) if notes_parts else "Standard neonatal dosing"

    return NeonatalDosingResult(
        drug_name=drug_name,
        postnatal_age_days=postnatal_age_days,
        weight_kg=weight_kg,
        gestational_age_weeks=gestational_age_weeks,
        cl_mL_per_min_per_kg=cl_mL_per_min_per_kg,
        vd_L_per_kg=vd_L_per_kg,
        t_half_h=t_half_h,
        recommended_dose_mg_per_kg=dose_mg_per_kg,
        dosing_interval_h=tau,
        css_avg_mg_L=css_avg,
        dose_class=dose_class,
        maturation_stage=maturation_stage,
        notes=notes,
    )
