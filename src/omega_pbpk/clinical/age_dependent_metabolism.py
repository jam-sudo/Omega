"""Age-dependent drug metabolism model — Phase 313.

Models how drug metabolism and renal elimination change across the entire human
lifespan from neonate to elderly.

References
----------
- Alcorn J & McNamara PJ. Pharmacokinetics in the newborn. Adv Drug Deliv Rev. 2003.
- Kearns GL et al. Developmental pharmacology — drug disposition, action, and
  therapy in infants and children. N Engl J Med. 2003;349:1157-67.
- Turnheim K. When drug therapy gets old: pharmacokinetics and pharmacodynamics
  in the elderly. Exp Gerontol. 2003;38(8):843-53.
- Klotz U. Pharmacokinetics and drug metabolism in the elderly. Drug Metab Rev.
  2009;41(2):67-76.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Standard age reference points
_NEONATE_END_Y = 0.08  # ~28 days in years (1/12.5)
_MATURATION_END_Y = 5.0
_PEAK_END_Y = 30.0
_DECLINE_END_Y = 80.0

# CYP enzyme parameters: (neonatal_factor, decline_factor_at_80)
# decline_factor_at_80 = fraction of peak at 80 years
_CYP_PARAMS: dict[str, tuple[float, float]] = {
    "CYP3A4": (0.20, 0.70),
    "CYP2D6": (0.05, 0.80),
    "CYP2C9": (0.10, 0.80),
    "CYP2C19": (0.05, 0.75),
    "CYP1A2": (0.05, 0.65),
    "CYP2E1": (0.40, 0.85),
}

# Standard weight-for-age (kg) — approximate midpoints
_WEIGHT_BY_AGE: list[tuple[float, float]] = [
    (0.0, 3.3),
    (0.083, 3.5),  # 1 month
    (0.25, 5.5),   # 3 months
    (0.5, 7.0),    # 6 months
    (1.0, 10.0),
    (2.0, 12.5),
    (5.0, 18.0),
    (10.0, 32.0),
    (18.0, 68.0),
    (30.0, 75.0),
    (50.0, 78.0),
    (65.0, 76.0),
    (75.0, 73.0),
    (80.0, 70.0),
]

# Ages used in lifespan dose table
_LIFESPAN_AGES = [0.05, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 18.0, 30.0, 50.0, 65.0, 75.0]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LifespanPKResult:
    """Pharmacokinetic parameters scaled across the lifespan."""

    drug_name: str
    age_years: float
    weight_kg: float
    sex: str
    cl_scaled_L_per_h: float
    vd_scaled_L: float
    t_half_h: float
    recommended_dose_mg: float
    dose_reduction_pct: float
    notes: str


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_age(age_years: float) -> None:
    if age_years < 0:
        raise ValueError(f"age_years must be >= 0, got {age_years}")


def _validate_weight(weight_kg: float) -> None:
    if weight_kg <= 0:
        raise ValueError(f"weight_kg must be > 0, got {weight_kg}")


def _validate_sex(sex: str) -> str:
    sex_lower = sex.lower()
    if sex_lower not in ("male", "female"):
        raise ValueError(f"sex must be 'male' or 'female', got {sex!r}")
    return sex_lower


def _validate_cl(cl: float) -> None:
    if cl <= 0:
        raise ValueError(f"CL must be > 0, got {cl}")


def _validate_vd(vd: float) -> None:
    if vd <= 0:
        raise ValueError(f"Vd must be > 0, got {vd}")


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def cyp_activity_by_age(enzyme: str, age_years: float) -> float:
    """Return CYP enzyme activity scaling factor (0–1) relative to young adult peak.

    The lifespan trajectory has four phases:

    1. Neonatal (0–28 days, 0–0.08 y): enzyme-specific low activity
    2. Maturation (0.08–5 y): sigmoid rise to 1.0
    3. Peak (5–30 y): plateau at 1.0
    4. Age-related decline (30–80 y): linear decline; capped at >80 y

    Parameters
    ----------
    enzyme : str
        One of CYP3A4, CYP2D6, CYP2C9, CYP2C19, CYP1A2, CYP2E1.
    age_years : float
        Age in years (>= 0).

    Returns
    -------
    float
        Activity factor in [0, 1].
    """
    _validate_age(age_years)

    enzyme_upper = enzyme.upper()
    if enzyme_upper not in _CYP_PARAMS:
        raise ValueError(
            f"Unsupported enzyme {enzyme!r}. Supported: {sorted(_CYP_PARAMS)}"
        )

    neonatal_factor, decline_at_80 = _CYP_PARAMS[enzyme_upper]

    if age_years <= _NEONATE_END_Y:
        return neonatal_factor

    if age_years <= _MATURATION_END_Y:
        # Sigmoid maturation from neonatal_factor to 1.0
        # Normalise age within (0.08, 5) to (0, 1)
        t = (age_years - _NEONATE_END_Y) / (_MATURATION_END_Y - _NEONATE_END_Y)
        # Sigmoid: S(t) = 1 / (1 + exp(-10*(t - 0.5)))
        sigmoid = 1.0 / (1.0 + math.exp(-10.0 * (t - 0.5)))
        # Map sigmoid (≈0→1) back to (neonatal_factor → 1.0)
        return neonatal_factor + (1.0 - neonatal_factor) * sigmoid

    if age_years <= _PEAK_END_Y:
        return 1.0

    # Linear decline from 1.0 at 30 y to decline_at_80 at 80 y
    slope = (decline_at_80 - 1.0) / (_DECLINE_END_Y - _PEAK_END_Y)
    activity = 1.0 + slope * (age_years - _PEAK_END_Y)
    # For ages beyond 80, continue the same slope but floor at 0
    return max(0.0, activity)


def gfr_by_age(age_years: float, sex: str = "male") -> float:
    """Return estimated GFR in mL/min/1.73 m² for a given age and sex.

    Developmental trajectory:
    - Neonates (~10 mL/min/1.73 m²)
    - 1 y: ~80
    - 5 y: ~110
    - Peak at 20–30 y: 120 (male) / 110 (female)
    - Decline after 40 y: ~1 mL/min/1.73 m² per year

    Parameters
    ----------
    age_years : float
        Age in years (>= 0).
    sex : str
        'male' or 'female'.

    Returns
    -------
    float
        GFR in mL/min/1.73 m².
    """
    _validate_age(age_years)
    sex_lower = _validate_sex(sex)

    peak_gfr = 120.0 if sex_lower == "male" else 110.0

    # Neonatal period (0–1 month, ~0.083 y): start at 10
    if age_years <= 0.083:
        return 10.0

    # Rapid postnatal maturation (0.083–1 y): 10 → 80
    if age_years <= 1.0:
        t = (age_years - 0.083) / (1.0 - 0.083)
        return 10.0 + (80.0 - 10.0) * t

    # Childhood maturation (1–5 y): 80 → 110
    if age_years <= 5.0:
        t = (age_years - 1.0) / (5.0 - 1.0)
        return 80.0 + (110.0 - 80.0) * t

    # Adolescent rise (5–20 y): 110 → peak_gfr
    if age_years <= 20.0:
        t = (age_years - 5.0) / (20.0 - 5.0)
        return 110.0 + (peak_gfr - 110.0) * t

    # Peak plateau (20–40 y)
    if age_years <= 40.0:
        return peak_gfr

    # Decline after 40 y: 1 mL/min/1.73 m² per year
    gfr = peak_gfr - 1.0 * (age_years - 40.0)
    return max(5.0, gfr)


def _weight_by_age(age_years: float) -> float:
    """Interpolate standard weight (kg) from age in years."""
    ages = [a for a, _ in _WEIGHT_BY_AGE]
    weights = [w for _, w in _WEIGHT_BY_AGE]
    if age_years <= ages[0]:
        return weights[0]
    if age_years >= ages[-1]:
        return weights[-1]
    return float(np.interp(age_years, ages, weights))


def _body_fat_fraction(age_years: float, sex: str) -> float:
    """Approximate body fat fraction by age and sex."""
    # Neonates ~15%, infants peak ~25%, children ~18%, adults M~20% F~28%, elderly M~28% F~38%
    if age_years < 0.5:
        return 0.15
    if age_years < 2.0:
        return 0.25
    if age_years < 10.0:
        return 0.18
    if sex == "male":
        if age_years < 30.0:
            return 0.18
        if age_years < 60.0:
            return 0.22
        return 0.28
    else:  # female
        if age_years < 30.0:
            return 0.28
        if age_years < 60.0:
            return 0.32
        return 0.38


def lifespan_pk_scaling(
    drug_name: str,
    age_years: float,
    weight_kg: float,
    sex: str,
    cl_adult_L_per_h: float,
    vd_adult_L: float,
    fm_cyp3a4: float = 0.5,
    fm_cyp2d6: float = 0.1,
    fm_cyp2c9: float = 0.2,
    fm_cyp2c19: float = 0.1,
    f_renal: float = 0.1,
) -> LifespanPKResult:
    """Compute CL and Vd scaled to the given age, weight, and sex.

    The metabolic component of CL is scaled by the weighted sum of CYP activity
    factors.  The renal component is scaled by GFR relative to adult reference
    (120 mL/min/1.73 m² for males).  The non-CYP metabolic fraction (1 - sum fm -
    f_renal) is scaled allometrically.

    Vd is scaled by weight and adjusted for body composition (total body water
    fraction is higher in neonates/infants → higher weight-normalised Vd for
    hydrophilic drugs).

    Parameters
    ----------
    drug_name : str
        Identifier for the drug.
    age_years : float
        Patient age in years (>= 0).
    weight_kg : float
        Patient body weight in kg (> 0).
    sex : str
        'male' or 'female'.
    cl_adult_L_per_h : float
        Clearance in a reference 70 kg adult (L/h).
    vd_adult_L : float
        Volume of distribution in a reference 70 kg adult (L).
    fm_cyp3a4, fm_cyp2d6, fm_cyp2c9, fm_cyp2c19 : float
        Fraction metabolised by each CYP enzyme (0–1).
    f_renal : float
        Fraction of total CL attributable to renal elimination (0–1).

    Returns
    -------
    LifespanPKResult
    """
    _validate_age(age_years)
    _validate_weight(weight_kg)
    sex_lower = _validate_sex(sex)
    _validate_cl(cl_adult_L_per_h)
    _validate_vd(vd_adult_L)

    adult_weight = 70.0
    adult_gfr = 120.0 if sex_lower == "male" else 110.0

    # ----- Metabolic CL scaling -----
    cyp_fractions = {
        "CYP3A4": fm_cyp3a4,
        "CYP2D6": fm_cyp2d6,
        "CYP2C9": fm_cyp2c9,
        "CYP2C19": fm_cyp2c19,
    }
    total_fm = sum(cyp_fractions.values())
    fm_non_cyp = max(0.0, 1.0 - total_fm - f_renal)

    # Weighted CYP activity
    cyp_activity_weighted = sum(
        fm * cyp_activity_by_age(enz, age_years) for enz, fm in cyp_fractions.items()
    )

    # Renal CL scaling: proportional to GFR ratio
    gfr = gfr_by_age(age_years, sex_lower)
    renal_scale = gfr / adult_gfr

    # Non-CYP metabolic: allometric scaling (weight^0.75 / 70^0.75)
    allometric_scale = (weight_kg / adult_weight) ** 0.75

    # Total CL
    cl_scaled = cl_adult_L_per_h * (
        cyp_activity_weighted + f_renal * renal_scale + fm_non_cyp * allometric_scale
    )
    cl_scaled = max(cl_scaled, 1e-6)

    # ----- Vd scaling -----
    # Base weight scaling (linear)
    vd_weight_scale = weight_kg / adult_weight

    # Body composition adjustment: total body water fraction
    # Neonates ~80% TBW, adults ~60%. Hydrophilic drugs scale with TBW.
    # We model a TBW-based uplift factor.
    body_fat = _body_fat_fraction(age_years, sex_lower)
    adult_body_fat = 0.22 if sex_lower == "male" else 0.30
    # TBW fraction ≈ 1 - fat fraction
    tbw_fraction = 1.0 - body_fat
    adult_tbw_fraction = 1.0 - adult_body_fat
    tbw_scale = tbw_fraction / adult_tbw_fraction  # > 1 for neonates/children

    # Apply a partial TBW correction (weight 30% TBW, 70% weight)
    composition_factor = 0.7 + 0.3 * tbw_scale
    vd_scaled = vd_adult_L * vd_weight_scale * composition_factor
    vd_scaled = max(vd_scaled, 1e-6)

    # ----- Derived PK parameters -----
    t_half_h = math.log(2.0) * vd_scaled / cl_scaled

    # Reference adult dose (on a per-kg basis scaled by AUC equivalence)
    # recommended dose ∝ CL_scaled / CL_adult * adult_dose  (assume adult dose ~ cl * AUC)
    # We cannot know adult dose here, so return as fractional of CL ratio
    cl_ratio = cl_scaled / cl_adult_L_per_h
    # recommended_dose_mg placeholder — will be computed in lifespan_dose_table
    recommended_dose_mg = float("nan")

    dose_reduction_pct = max(0.0, (1.0 - cl_ratio) * 100.0)

    notes_parts = []
    if age_years < 0.083:
        notes_parts.append("neonatal — immature CYP enzymes and GFR")
    elif age_years < 2.0:
        notes_parts.append("infant — rapidly maturing CYP and renal function")
    elif age_years < 12.0:
        notes_parts.append("pediatric — CYP approaching adult levels")
    elif age_years < 18.0:
        notes_parts.append("adolescent")
    elif age_years >= 65.0:
        notes_parts.append("elderly — reduced CYP and renal function")

    notes = "; ".join(notes_parts) if notes_parts else "adult reference range"

    return LifespanPKResult(
        drug_name=drug_name,
        age_years=age_years,
        weight_kg=weight_kg,
        sex=sex_lower,
        cl_scaled_L_per_h=round(cl_scaled, 4),
        vd_scaled_L=round(vd_scaled, 4),
        t_half_h=round(t_half_h, 3),
        recommended_dose_mg=recommended_dose_mg,
        dose_reduction_pct=round(dose_reduction_pct, 1),
        notes=notes,
    )


def lifespan_dose_table(
    drug_name: str,
    adult_dose_mg: float,
    cl_adult_L_per_h: float,
    vd_adult_L: float,
    sex: str = "male",
    fm_cyp3a4: float = 0.5,
    fm_cyp2d6: float = 0.1,
    fm_cyp2c9: float = 0.2,
    fm_cyp2c19: float = 0.1,
    f_renal: float = 0.1,
) -> list[LifespanPKResult]:
    """Generate lifespan PK and dose table across standard age checkpoints.

    Ages sampled: 0.05, 0.25, 0.5, 1, 2, 5, 10, 18, 30, 50, 65, 75 years.
    Weight is taken from standard pediatric/adult growth charts.
    Recommended dose is scaled proportionally to CL (AUC-matched dosing).

    Parameters
    ----------
    drug_name : str
        Name or identifier for the drug.
    adult_dose_mg : float
        Reference dose for a young adult (mg).
    cl_adult_L_per_h : float
        Adult clearance (L/h).
    vd_adult_L : float
        Adult volume of distribution (L).
    sex : str
        'male' or 'female'.
    fm_cyp3a4, fm_cyp2d6, fm_cyp2c9, fm_cyp2c19 : float
        Fraction metabolised by each CYP enzyme.
    f_renal : float
        Fraction of CL attributable to renal elimination.

    Returns
    -------
    list[LifespanPKResult]
    """
    if adult_dose_mg <= 0:
        raise ValueError(f"adult_dose_mg must be > 0, got {adult_dose_mg}")
    _validate_cl(cl_adult_L_per_h)
    _validate_vd(vd_adult_L)
    sex_lower = _validate_sex(sex)

    results: list[LifespanPKResult] = []
    for age in _LIFESPAN_AGES:
        weight = _weight_by_age(age)
        base = lifespan_pk_scaling(
            drug_name=drug_name,
            age_years=age,
            weight_kg=weight,
            sex=sex_lower,
            cl_adult_L_per_h=cl_adult_L_per_h,
            vd_adult_L=vd_adult_L,
            fm_cyp3a4=fm_cyp3a4,
            fm_cyp2d6=fm_cyp2d6,
            fm_cyp2c9=fm_cyp2c9,
            fm_cyp2c19=fm_cyp2c19,
            f_renal=f_renal,
        )
        cl_ratio = base.cl_scaled_L_per_h / cl_adult_L_per_h
        rec_dose = round(adult_dose_mg * cl_ratio, 2)
        dose_reduction_pct = max(0.0, (1.0 - cl_ratio) * 100.0)

        results.append(
            LifespanPKResult(
                drug_name=base.drug_name,
                age_years=base.age_years,
                weight_kg=base.weight_kg,
                sex=base.sex,
                cl_scaled_L_per_h=base.cl_scaled_L_per_h,
                vd_scaled_L=base.vd_scaled_L,
                t_half_h=base.t_half_h,
                recommended_dose_mg=rec_dose,
                dose_reduction_pct=round(dose_reduction_pct, 1),
                notes=base.notes,
            )
        )
    return results
