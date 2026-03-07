"""Pediatric renal function scaling for drug clearance.

Implements GFR maturation using the Rhodin 2009 sigmoid model and
allometric/maturation-based scaling of renal and hepatic clearance
for pediatric patients.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "PediatricRenalResult",
    "gfr_maturation",
    "calculate_pediatric_bsa",
    "simulate_pediatric_renal_pk",
    "pediatric_dose_schedule",
]

# Adult reference values
_ADULT_GFR_NORM = 120.0  # mL/min/1.73 m²
_ADULT_BSA = 1.73  # m²
_ADULT_WEIGHT = 70.0  # kg

# Rhodin 2009 sigmoid maturation model parameters
_RHODIN_TM50_WEEKS = 47.7  # postmenstrual age at 50% GFR maturation
_RHODIN_HILL = 3.4  # Hill coefficient

# Standard dosing intervals (hours)
_STANDARD_INTERVALS = [6.0, 8.0, 12.0, 24.0]


@dataclass(frozen=True)
class PediatricRenalResult:
    """Result of pediatric renal PK simulation."""

    drug_name: str
    age_months: float  # 0 = newborn, 12 = 1 year, etc.
    weight_kg: float

    gfr_mL_per_min_per_1_73m2: float  # normalised GFR
    gfr_mL_per_min: float  # absolute GFR for this child
    gfr_fraction_adult: float  # GFR as fraction of adult (1.0 = adult)
    bsa_m2: float  # body surface area (weight-only Mosteller approximation)

    # PK scaling
    cl_renal_scaled_L_per_h: float  # scaled renal CL
    cl_total_scaled_L_per_h: float  # total CL (renal + hepatic)
    vd_scaled_L: float  # scaled volume of distribution
    t_half_h: float  # resulting half-life

    # Dosing
    recommended_dose_mg: float  # AUC-matched scaled dose
    dose_mg_per_kg: float  # weight-normalised dose
    dosing_interval_h: float  # recommended interval (nearest standard)

    notes: str


def gfr_maturation(age_months: float) -> float:
    """GFR maturation fraction relative to adult (120 mL/min/1.73 m²).

    Uses the Rhodin 2009 sigmoid model parameterised to postmenstrual age
    (PMA). Postnatal age in months is converted assuming term birth at 40
    gestational weeks.

    Parameters
    ----------
    age_months : float
        Postnatal age in months (0 = term newborn).

    Returns
    -------
    float
        Maturation fraction in [0.01, 1.0].
    """
    if age_months < 0:
        raise ValueError(f"age_months must be >= 0, got {age_months}")

    # Convert postnatal months to PMA weeks (40 wk term + postnatal weeks)
    pma_weeks = 40.0 + age_months * 4.33

    # Rhodin sigmoid: maturation = PMA^Hill / (TM50^Hill + PMA^Hill)
    pma_h = pma_weeks ** _RHODIN_HILL
    tm50_h = _RHODIN_TM50_WEEKS ** _RHODIN_HILL
    maturation = pma_h / (tm50_h + pma_h)

    return float(min(max(maturation, 0.01), 1.0))


def calculate_pediatric_bsa(weight_kg: float, height_cm: float | None = None) -> float:
    """Calculate body surface area (BSA) in m².

    Uses DuBois formula when height is available, otherwise a
    weight-only power-law approximation.

    Parameters
    ----------
    weight_kg : float
        Body weight (kg).
    height_cm : float or None
        Height (cm). If None, a weight-only estimate is used.

    Returns
    -------
    float
        BSA in m².
    """
    if weight_kg <= 0:
        raise ValueError(f"weight_kg must be positive, got {weight_kg}")

    if height_cm is not None and height_cm > 0:
        # DuBois formula: BSA = 0.007184 * W^0.425 * H^0.725
        bsa = 0.007184 * (weight_kg ** 0.425) * (height_cm ** 0.725)
    else:
        # Weight-only approximation
        bsa = 0.1 * (weight_kg ** 0.67)

    return float(bsa)


def _round_to_standard_interval(t_half_h: float) -> float:
    """Round half-life to the nearest standard dosing interval."""
    # A rule of thumb: interval ~ 1–2× t½ to maintain trough above MEC
    target = t_half_h
    # Choose the standard interval closest to one half-life
    best = min(_STANDARD_INTERVALS, key=lambda iv: abs(iv - target))
    return best


def simulate_pediatric_renal_pk(
    drug_name: str,
    age_months: float,
    weight_kg: float,
    adult_cl_renal_L_per_h: float,
    adult_cl_hepatic_L_per_h: float,
    adult_vd_L: float,
    adult_dose_mg: float,
    fe_renal: float = 0.5,
    height_cm: float | None = None,
) -> PediatricRenalResult:
    """Simulate renal PK for a pediatric patient.

    Scales adult reference clearance and volume values to the child using
    GFR maturation (Rhodin 2009) and allometric scaling.

    Parameters
    ----------
    drug_name : str
        Drug identifier.
    age_months : float
        Postnatal age in months (0 = term newborn).
    weight_kg : float
        Body weight (kg).
    adult_cl_renal_L_per_h : float
        Adult reference renal clearance (L/h).
    adult_cl_hepatic_L_per_h : float
        Adult reference hepatic clearance (L/h).
    adult_vd_L : float
        Adult reference volume of distribution (L).
    adult_dose_mg : float
        Adult reference dose (mg).
    fe_renal : float
        Fraction eliminated renally (0–1). Informational.
    height_cm : float or None
        Height (cm) for BSA calculation. If None, weight-only BSA used.

    Returns
    -------
    PediatricRenalResult
    """
    # --- Input validation ---
    if age_months < 0:
        raise ValueError(f"age_months must be >= 0, got {age_months}")
    if weight_kg <= 0:
        raise ValueError(f"weight_kg must be positive, got {weight_kg}")
    if adult_cl_renal_L_per_h <= 0:
        raise ValueError(f"adult_cl_renal_L_per_h must be positive, got {adult_cl_renal_L_per_h}")
    if adult_cl_hepatic_L_per_h <= 0:
        raise ValueError(
            f"adult_cl_hepatic_L_per_h must be positive, got {adult_cl_hepatic_L_per_h}"
        )
    if adult_vd_L <= 0:
        raise ValueError(f"adult_vd_L must be positive, got {adult_vd_L}")
    if adult_dose_mg <= 0:
        raise ValueError(f"adult_dose_mg must be positive, got {adult_dose_mg}")
    if not (0.0 <= fe_renal <= 1.0):
        raise ValueError(f"fe_renal must be in [0, 1], got {fe_renal}")

    # --- GFR maturation ---
    gfr_frac = gfr_maturation(age_months)

    # --- BSA ---
    bsa = calculate_pediatric_bsa(weight_kg, height_cm)

    # --- Normalised and absolute GFR ---
    gfr_norm = _ADULT_GFR_NORM * gfr_frac  # mL/min/1.73 m²
    gfr_abs = gfr_norm * bsa / _ADULT_BSA  # mL/min

    # --- Renal CL scaling: GFR maturation × BSA fraction ---
    cl_renal_scaled = adult_cl_renal_L_per_h * gfr_frac * (bsa / _ADULT_BSA)

    # --- Hepatic CL scaling: allometric (0.75 exponent) × maturation proxy ---
    # GFR maturation used as simplified proxy for hepatic enzyme maturation
    weight_factor_hepatic = (weight_kg / _ADULT_WEIGHT) ** 0.75
    hepatic_mat = gfr_frac ** 0.5
    cl_hepatic_scaled = adult_cl_hepatic_L_per_h * weight_factor_hepatic * hepatic_mat

    # --- Vd scaling: linear allometric ---
    vd_scaled = adult_vd_L * (weight_kg / _ADULT_WEIGHT)

    # --- Derived PK ---
    cl_total = cl_renal_scaled + cl_hepatic_scaled
    t_half = 0.693 * vd_scaled / cl_total if cl_total > 0 else math.nan

    # --- Dose scaling: AUC-matched ---
    adult_cl_total = adult_cl_renal_L_per_h + adult_cl_hepatic_L_per_h
    recommended_dose = adult_dose_mg * (cl_total / adult_cl_total)
    dose_mg_per_kg = recommended_dose / weight_kg

    # --- Dosing interval ---
    dosing_interval = _round_to_standard_interval(t_half) if not math.isnan(t_half) else 24.0

    # --- Notes ---
    notes_parts = [
        f"GFR maturation fraction: {gfr_frac:.3f}",
        f"BSA: {bsa:.3f} m²",
        f"Scaled CL_renal: {cl_renal_scaled:.3f} L/h, "
        f"CL_hepatic: {cl_hepatic_scaled:.3f} L/h",
        f"Recommended dose: {recommended_dose:.2f} mg "
        f"({dose_mg_per_kg:.2f} mg/kg) q{dosing_interval:.0f}h",
    ]
    if age_months < 1:
        notes_parts.append("Neonatal patient: extreme caution — GFR very low at birth")
    elif age_months < 12:
        notes_parts.append("Infant: GFR still maturing — monitor closely")

    return PediatricRenalResult(
        drug_name=drug_name,
        age_months=age_months,
        weight_kg=weight_kg,
        gfr_mL_per_min_per_1_73m2=round(gfr_norm, 4),
        gfr_mL_per_min=round(gfr_abs, 4),
        gfr_fraction_adult=round(gfr_frac, 6),
        bsa_m2=round(bsa, 6),
        cl_renal_scaled_L_per_h=round(cl_renal_scaled, 6),
        cl_total_scaled_L_per_h=round(cl_total, 6),
        vd_scaled_L=round(vd_scaled, 4),
        t_half_h=round(t_half, 4) if not math.isnan(t_half) else math.nan,
        recommended_dose_mg=round(recommended_dose, 4),
        dose_mg_per_kg=round(dose_mg_per_kg, 4),
        dosing_interval_h=dosing_interval,
        notes="; ".join(notes_parts),
    )


def pediatric_dose_schedule(
    drug_name: str,
    age_months_list: list[float],
    weight_kg_list: list[float],
    adult_cl_renal_L_per_h: float,
    adult_cl_hepatic_L_per_h: float,
    adult_vd_L: float,
    adult_dose_mg: float,
    fe_renal: float = 0.5,
    height_cm: float | None = None,
) -> list[PediatricRenalResult]:
    """Compute pediatric dosing for a series of age/weight pairs.

    Parameters
    ----------
    drug_name : str
        Drug identifier.
    age_months_list : list of float
        Postnatal ages in months.
    weight_kg_list : list of float
        Corresponding body weights (kg).
    adult_cl_renal_L_per_h : float
        Adult reference renal clearance (L/h).
    adult_cl_hepatic_L_per_h : float
        Adult reference hepatic clearance (L/h).
    adult_vd_L : float
        Adult reference volume of distribution (L).
    adult_dose_mg : float
        Adult reference dose (mg).
    fe_renal : float
        Fraction eliminated renally (0–1).
    height_cm : float or None
        Optional height for BSA (applied uniformly).

    Returns
    -------
    list[PediatricRenalResult]
        Sorted by age_months ascending.
    """
    if len(age_months_list) != len(weight_kg_list):
        raise ValueError("age_months_list and weight_kg_list must have the same length")

    pairs = sorted(zip(age_months_list, weight_kg_list), key=lambda x: x[0])

    results: list[PediatricRenalResult] = []
    for age, weight in pairs:
        result = simulate_pediatric_renal_pk(
            drug_name=drug_name,
            age_months=age,
            weight_kg=weight,
            adult_cl_renal_L_per_h=adult_cl_renal_L_per_h,
            adult_cl_hepatic_L_per_h=adult_cl_hepatic_L_per_h,
            adult_vd_L=adult_vd_L,
            adult_dose_mg=adult_dose_mg,
            fe_renal=fe_renal,
            height_cm=height_cm,
        )
        results.append(result)

    return results
