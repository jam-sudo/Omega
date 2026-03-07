"""Pediatric dose scaling using allometric and maturation functions.

Phase 873: Scale adult doses to pediatric patients.
"""

from dataclasses import dataclass
from math import pow

# CYP/pathway maturation parameters (TM50 weeks, Hill coefficient)
_MATURATION_PARAMS = {
    "GFR": (47.7, 3.4),
    "CYP3A4": (38.5, 8.0),
    "CYP2D6": (38.7, 12.0),
    "other": (47.7, 3.4),  # default to GFR-like
}

# Adult PMA assumed: 40 weeks gestation + 18 years * 52 weeks/year
_ADULT_PMA_WEEKS = 40.0 + 18.0 * 52.0


def _maturation_fraction(pma_weeks: float, tm50: float, hill: float) -> float:
    """Rhodin 2009 sigmoid maturation function."""
    pma_hill = pow(pma_weeks, hill)
    tm50_hill = pow(tm50, hill)
    return pma_hill / (tm50_hill + pma_hill)


def _assign_age_group(pma_weeks: float) -> str:
    """Assign pediatric age group from PMA in weeks."""
    # PMA 24-44 weeks → neonate (preterm to term, inclusive)
    # 44 weeks - 2 years (44 + 2*52 = 148) → infant
    # 2 years (148) - 12 years (40 + 12*52 = 664) → child
    # 12-18 years (40 + 18*52 = 976) → adolescent
    if pma_weeks <= 44.0:
        return "neonate"
    elif pma_weeks <= 148.0:  # 44 + 104 weeks (2 years)
        return "infant"
    elif pma_weeks <= 664.0:  # 40 + 12*52
        return "child"
    else:
        return "adolescent"


@dataclass(frozen=True)
class PediatricDoseScalingResult:
    """Result from pediatric dose scaling calculation."""

    drug_name: str
    cyp_enzyme: str
    child_bw_kg: float
    child_pma_weeks: float
    maturation_fraction: float
    cl_allometric_L_per_h: float
    cl_mature_L_per_h: float
    vd_scaled_L: float
    recommended_dose_mg: float
    dose_per_kg_mg_per_kg: float
    age_group: str
    cl_ratio_ped_to_adult: float
    notes: str


def scale_pediatric_dose(
    drug_name: str,
    adult_cl_L_per_h: float,
    adult_vd_L: float,
    adult_dose_mg: float,
    child_bw_kg: float,
    child_pma_weeks: float,
    adult_bw_kg: float = 70.0,
    cyp_enzyme: str = "CYP3A4",
) -> PediatricDoseScalingResult:
    """Scale adult dose to a pediatric patient.

    Uses allometric scaling (BW^0.75 for CL, BW^1.0 for Vd) combined
    with a Rhodin 2009 sigmoid maturation function.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    adult_cl_L_per_h : float
        Adult clearance in L/h.
    adult_vd_L : float
        Adult volume of distribution in L.
    adult_dose_mg : float
        Adult dose in mg.
    child_bw_kg : float
        Child body weight in kg.
    child_pma_weeks : float
        Child post-menstrual age in weeks.
    adult_bw_kg : float
        Adult body weight in kg (default 70.0).
    cyp_enzyme : str
        CYP enzyme determining maturation parameters.

    Returns
    -------
    PediatricDoseScalingResult
    """
    if child_bw_kg <= 0:
        raise ValueError("child_bw_kg must be > 0")
    if child_pma_weeks <= 0:
        raise ValueError("child_pma_weeks must be > 0")
    if adult_cl_L_per_h <= 0:
        raise ValueError("adult_cl_L_per_h must be > 0")
    if adult_dose_mg <= 0:
        raise ValueError("adult_dose_mg must be > 0")

    cyp_key = cyp_enzyme if cyp_enzyme in _MATURATION_PARAMS else "other"
    tm50, hill = _MATURATION_PARAMS[cyp_key]

    # Allometric CL scaling (BW^0.75)
    bw_ratio = child_bw_kg / adult_bw_kg
    cl_allometric = adult_cl_L_per_h * pow(bw_ratio, 0.75)

    # Maturation fraction at child PMA
    mf_child = _maturation_fraction(child_pma_weeks, tm50, hill)

    # Maturation fraction at adult PMA (should be ~1.0 for adults)
    mf_adult = _maturation_fraction(_ADULT_PMA_WEEKS, tm50, hill)

    # Mature CL: allometric * (MF_child / MF_adult)
    cl_mature = cl_allometric * mf_child / mf_adult

    # Allometric Vd scaling (linear, BW^1.0)
    vd_scaled = adult_vd_L * bw_ratio

    # CL ratio
    cl_ratio = cl_mature / adult_cl_L_per_h

    # Recommended dose: scale adult dose by CL ratio (target same AUC)
    recommended_dose = round(adult_dose_mg * cl_ratio, 1)
    if recommended_dose <= 0:
        recommended_dose = 0.1  # floor to avoid zero

    dose_per_kg = recommended_dose / child_bw_kg

    age_group = _assign_age_group(child_pma_weeks)

    notes_parts = [
        "Allometric exponent 0.75 for CL, 1.0 for Vd.",
        f"Maturation model: {cyp_key} (TM50={tm50} wk, Hill={hill}).",
        f"MF(child)={mf_child:.3f}, MF(adult)={mf_adult:.3f}.",
        f"Target: same AUC as adult dose {adult_dose_mg} mg.",
    ]
    notes = " ".join(notes_parts)

    return PediatricDoseScalingResult(
        drug_name=drug_name,
        cyp_enzyme=cyp_enzyme,
        child_bw_kg=child_bw_kg,
        child_pma_weeks=child_pma_weeks,
        maturation_fraction=mf_child,
        cl_allometric_L_per_h=cl_allometric,
        cl_mature_L_per_h=cl_mature,
        vd_scaled_L=vd_scaled,
        recommended_dose_mg=recommended_dose,
        dose_per_kg_mg_per_kg=dose_per_kg,
        age_group=age_group,
        cl_ratio_ped_to_adult=cl_ratio,
        notes=notes,
    )


def dose_across_ages(
    drug_name: str,
    adult_cl_L_per_h: float,
    adult_dose_mg: float,
    weight_age_pairs: list,
) -> list:
    """Calculate dose across multiple pediatric age groups.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    adult_cl_L_per_h : float
        Adult clearance in L/h.
    adult_dose_mg : float
        Adult dose in mg.
    weight_age_pairs : list of tuple
        Each tuple: (bw_kg, pma_weeks).

    Returns
    -------
    list of PediatricDoseScalingResult, sorted by child_pma_weeks ascending.
    """
    results = []
    for bw_kg, pma_weeks in weight_age_pairs:
        # Use a nominal Vd (BW-scaled from 50 L in 70 kg adult)
        adult_vd_L = 50.0
        result = scale_pediatric_dose(
            drug_name=drug_name,
            adult_cl_L_per_h=adult_cl_L_per_h,
            adult_vd_L=adult_vd_L,
            adult_dose_mg=adult_dose_mg,
            child_bw_kg=bw_kg,
            child_pma_weeks=pma_weeks,
        )
        results.append(result)

    results.sort(key=lambda r: r.child_pma_weeks)
    return results
