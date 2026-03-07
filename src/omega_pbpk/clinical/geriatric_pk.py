"""Comprehensive geriatric PK model for patients aged 65+.

Models multi-system physiological decline that affects drug pharmacokinetics:
- GFR decline (renal function)
- CYP enzyme activity reduction (hepatic metabolism)
- Cardiac output decrease (hepatic blood flow)
- Body composition changes (fat/muscle redistribution)
- Albumin reduction (protein binding)

References:
    - Turnheim K. Clin Pharmacokinet. 2003;42(12):1103-15
    - Klotz U. Eur J Clin Pharmacol. 2009;65(5):437-44
    - McLachlan AJ & Pont LG. Basic Clin Pharmacol Toxicol. 2012;111(4):224-9
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GeriatricPKResult:
    """Result of geriatric PK scaling analysis.

    Attributes
    ----------
    drug_name : Drug identifier.
    age_years : Patient age in years.
    sex : Patient sex ('male' or 'female').
    weight_kg : Patient body weight in kg.
    cl_geriatric_L_per_h : Scaled clearance (L/h) for geriatric patient.
    vd_geriatric_L : Scaled volume of distribution (L) for geriatric patient.
    t_half_h : Terminal half-life (h) = 0.693 * Vd / CL.
    gfr_estimated_mL_per_min : Estimated GFR (mL/min) based on age.
    cyp_factor : CYP enzyme activity factor (0-1).
    co_factor : Cardiac output factor (0-1).
    dose_adjustment_factor : Recommended dose adjustment relative to normal adult.
    dose_recommendation : Textual dosing recommendation.
    notes : Additional clinical notes.
    """

    drug_name: str
    age_years: float
    sex: str
    weight_kg: float
    cl_geriatric_L_per_h: float
    vd_geriatric_L: float
    t_half_h: float
    gfr_estimated_mL_per_min: float
    cyp_factor: float
    co_factor: float
    dose_adjustment_factor: float
    dose_recommendation: str
    notes: list[str] = field(default_factory=list)


_VALID_SEXES = {"male", "female"}
_VALID_DISTRIBUTION_TYPES = {"hydrophilic", "lipophilic", "neutral"}


def _calc_gfr(age_years: float) -> float:
    """Estimate GFR based on age.

    GFR starts declining after age 40 at approximately 0.8 mL/min/year.
    Normal GFR is ~120 mL/min; clamped to [15, 120].
    """
    decline = max(0.0, age_years - 40) * 0.008
    gfr = 120.0 * (1.0 - decline)
    return max(15.0, min(120.0, gfr))


def _calc_cyp_factor(age_years: float) -> float:
    """CYP enzyme activity factor — declines 1%/year after 60."""
    decline = max(0.0, age_years - 60) * 0.01
    return max(0.4, 1.0 - decline)


def _calc_co_factor(age_years: float) -> float:
    """Cardiac output factor — declines 1%/year after 60."""
    decline = max(0.0, age_years - 60) * 0.01
    return max(0.5, 1.0 - decline)


def _calc_albumin_factor(age_years: float) -> float:
    """Albumin factor — declines 0.5%/year after 60."""
    decline = max(0.0, age_years - 60) * 0.005
    return max(0.6, 1.0 - decline)


def _calc_fat_fraction(age_years: float) -> float:
    """Body fat fraction — increases with age after 40."""
    increase = max(0.0, age_years - 40) * 0.005
    return min(0.45, 0.15 + increase)


def _calc_muscle_fraction(age_years: float) -> float:
    """Muscle fraction — decreases with age after 40 (sarcopenia)."""
    decrease = max(0.0, age_years - 40) * 0.005
    return max(0.20, 0.40 - decrease)


def geriatric_pk_scaling(
    drug_name: str,
    age_years: float,
    weight_kg: float,
    sex: str,
    scr_mg_dL: float,
    cl_normal_L_per_h: float,
    vd_normal_L: float,
    distribution_type: str = "neutral",
) -> GeriatricPKResult:
    """Compute geriatric-adjusted PK parameters for a drug.

    Accounts for age-related changes in renal function, hepatic metabolism,
    cardiac output, and body composition.

    Parameters
    ----------
    drug_name:
        Name or identifier of the drug.
    age_years:
        Patient age in years. Must be >= 18.
    weight_kg:
        Patient body weight in kg. Must be > 0.
    sex:
        Patient sex: 'male' or 'female'.
    scr_mg_dL:
        Serum creatinine in mg/dL. Must be > 0.
    cl_normal_L_per_h:
        Normal adult total clearance (L/h). Must be > 0.
    vd_normal_L:
        Normal adult volume of distribution (L). Must be > 0.
    distribution_type:
        Tissue distribution type: 'hydrophilic', 'lipophilic', or 'neutral'.

    Returns
    -------
    GeriatricPKResult
        Dataclass with adjusted PK parameters and dosing recommendation.

    Raises
    ------
    ValueError
        If any input is out of physiological range or invalid.
    """
    # --- Input validation ---
    if not drug_name or not drug_name.strip():
        raise ValueError("drug_name must be a non-empty string")
    if age_years < 18:
        raise ValueError(f"age_years must be >= 18, got {age_years}")
    if weight_kg <= 0:
        raise ValueError(f"weight_kg must be > 0, got {weight_kg}")
    sex_lower = sex.lower().strip()
    if sex_lower not in _VALID_SEXES:
        raise ValueError(f"sex must be one of {_VALID_SEXES}, got '{sex}'")
    if scr_mg_dL <= 0:
        raise ValueError(f"scr_mg_dL must be > 0, got {scr_mg_dL}")
    if cl_normal_L_per_h <= 0:
        raise ValueError(f"cl_normal_L_per_h must be > 0, got {cl_normal_L_per_h}")
    if vd_normal_L <= 0:
        raise ValueError(f"vd_normal_L must be > 0, got {vd_normal_L}")
    dist_lower = distribution_type.lower().strip()
    if dist_lower not in _VALID_DISTRIBUTION_TYPES:
        raise ValueError(
            f"distribution_type must be one of {_VALID_DISTRIBUTION_TYPES}, "
            f"got '{distribution_type}'"
        )

    # --- Physiological factors ---
    gfr = _calc_gfr(age_years)
    cyp_factor = _calc_cyp_factor(age_years)
    co_factor = _calc_co_factor(age_years)
    alb_factor = _calc_albumin_factor(age_years)
    fat_frac = _calc_fat_fraction(age_years)
    muscle_frac = _calc_muscle_fraction(age_years)

    # --- Clearance scaling ---
    # Renal portion (50% of CL) scaled by GFR/120
    cl_renal = cl_normal_L_per_h * 0.5 * (gfr / 120.0)
    # Hepatic portion (50% of CL) scaled by CYP activity and cardiac output
    cl_hep = cl_normal_L_per_h * 0.5 * cyp_factor * co_factor
    cl_geriatric = cl_renal + cl_hep

    # --- Vd scaling by distribution type ---
    if dist_lower == "hydrophilic":
        # Hydrophilic drugs distribute into muscle/water — reduced with sarcopenia
        vd_factor = muscle_frac / 0.40
        vd_geriatric = vd_normal_L * vd_factor
    elif dist_lower == "lipophilic":
        # Lipophilic drugs distribute into fat — increased with adiposity
        vd_factor = fat_frac / 0.15
        vd_geriatric = vd_normal_L * vd_factor
    else:
        # Neutral: modest adjustment via albumin (protein binding changes)
        vd_factor = alb_factor
        vd_geriatric = vd_normal_L * vd_factor

    # Ensure Vd is physiologically plausible (at least 1 L)
    vd_geriatric = max(1.0, vd_geriatric)

    # --- Half-life ---
    t_half = 0.693 * vd_geriatric / cl_geriatric

    # --- Dose adjustment factor ---
    dose_adj = cl_geriatric / cl_normal_L_per_h

    # --- Clinical notes ---
    notes: list[str] = []
    if age_years >= 80:
        notes.append("Very elderly (>=80 y): consider further dose reduction and close monitoring")
    if gfr < 30:
        notes.append(f"Severely reduced renal function (GFR={gfr:.1f} mL/min): check renal dosing")
    elif gfr < 60:
        notes.append(f"Moderately reduced renal function (GFR={gfr:.1f} mL/min)")
    if cyp_factor < 0.7:
        notes.append(f"Significant CYP activity decline (factor={cyp_factor:.2f}): review DDIs")
    if co_factor < 0.7:
        notes.append(f"Reduced cardiac output (factor={co_factor:.2f}): hepatic flow affected")
    if alb_factor < 0.8:
        notes.append(
            f"Low albumin (factor={alb_factor:.2f}): protein binding reduced, higher free fraction"
        )
    if dist_lower == "lipophilic":
        notes.append(
            f"Lipophilic drug: Vd increased by fat fraction "
            f"({fat_frac:.2f} vs 0.15 normal)"
        )
    elif dist_lower == "hydrophilic":
        notes.append(
            f"Hydrophilic drug: Vd reduced by muscle fraction "
            f"({muscle_frac:.2f} vs 0.40 normal)"
        )
    if scr_mg_dL > 1.5:
        notes.append(
            f"Elevated serum creatinine ({scr_mg_dL:.2f} mg/dL): GFR may be further reduced"
        )

    # --- Dose recommendation ---
    if dose_adj >= 0.9:
        dose_recommendation = "Standard adult dose may be appropriate; monitor for toxicity"
    elif dose_adj >= 0.75:
        dose_recommendation = (
            f"Reduce dose by ~{(1-dose_adj)*100:.0f}% "
            f"(dose adjustment factor: {dose_adj:.2f})"
        )
    elif dose_adj >= 0.5:
        dose_recommendation = (
            f"Significant dose reduction required (~{(1-dose_adj)*100:.0f}%); "
            f"consider extended dosing interval"
        )
    else:
        dose_recommendation = (
            f"Major dose reduction required ({dose_adj:.2f}x normal); "
            f"specialist review recommended"
        )

    # Add age group note
    if age_years >= 65:
        age_group = "elderly"
        if age_years >= 75:
            age_group = "old-old"
        if age_years >= 85:
            age_group = "very old-old"
        notes.insert(0, f"Patient age group: {age_group} ({age_years:.0f} y)")

    return GeriatricPKResult(
        drug_name=drug_name,
        age_years=age_years,
        sex=sex_lower,
        weight_kg=weight_kg,
        cl_geriatric_L_per_h=cl_geriatric,
        vd_geriatric_L=vd_geriatric,
        t_half_h=t_half,
        gfr_estimated_mL_per_min=gfr,
        cyp_factor=cyp_factor,
        co_factor=co_factor,
        dose_adjustment_factor=dose_adj,
        dose_recommendation=dose_recommendation,
        notes=notes,
    )
