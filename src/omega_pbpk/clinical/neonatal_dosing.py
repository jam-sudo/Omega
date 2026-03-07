"""Neonatal drug dosing adjustments for immature organ function.

Neonates (0-28 days postnatal age) have dramatically different PK from adults:
- GFR very low at birth (~10 mL/min/1.73m² at term), matures by ~1 year
- CYP3A4/2D6/1A2/2C9 activity very low at birth (1-10% of adult)
- Lower albumin binding → higher free fraction
- Higher total body water → larger Vd for water-soluble drugs

References:
- Kearns GL et al. NEJM 2003
- Anderson BJ, Holford NH. Annu Rev Pharmacol Toxicol 2008
- Allegaert K, van den Anker J. Brit J Clin Pharmacol 2015
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_VALID_ENZYMES = {"CYP3A4", "CYP2D6", "CYP1A2", "CYP2C9"}

# Adult reference GFR in mL/min/1.73m²
_ADULT_GFR = 100.0


@dataclass(frozen=True)
class NeonatalDoseResult:
    """Result of neonatal dose recommendation."""

    drug_name: str
    weight_kg: float
    gestational_age_weeks: float
    postnatal_age_days: float
    adult_dose_mg_per_kg: float
    neonatal_dose_mg_per_kg: float
    neonatal_dose_mg: float
    cl_renal_factor: float
    cl_hepatic_factor: float
    dosing_interval_h: float
    monitoring_recommendation: str
    notes: str


def neonatal_gfr(gestational_age_weeks: float, postnatal_age_days: float) -> float:
    """Estimate GFR in mL/min/1.73m² for neonates.

    Args:
        gestational_age_weeks: Gestational age in weeks (typically 24-42).
        postnatal_age_days: Postnatal age in days (0-28 for neonates).

    Returns:
        Estimated GFR in mL/min/1.73m², clamped to [2, 100].
    """
    if gestational_age_weeks <= 0:
        raise ValueError(f"gestational_age_weeks must be > 0, got {gestational_age_weeks}")
    if postnatal_age_days < 0:
        raise ValueError(f"postnatal_age_days must be >= 0, got {postnatal_age_days}")

    ga = gestational_age_weeks
    pna = postnatal_age_days

    # GFR = 10 * (GA/40)^2 at birth, matures linearly over ~1 year (365 days)
    gfr = 10.0 * (ga / 40.0) ** 2 * (1.0 + pna / 365.0)

    # Clamp to physiological range
    return max(2.0, min(100.0, gfr))


def neonatal_cyp_activity(
    enzyme: str,
    gestational_age_weeks: float,
    postnatal_age_days: float,
) -> float:
    """Estimate fraction of adult CYP enzyme activity in neonates.

    Args:
        enzyme: One of 'CYP3A4', 'CYP2D6', 'CYP1A2', 'CYP2C9'.
        gestational_age_weeks: Gestational age in weeks.
        postnatal_age_days: Postnatal age in days.

    Returns:
        Fraction of adult activity in [0.0, 1.0].
    """
    if enzyme not in _VALID_ENZYMES:
        raise ValueError(f"enzyme must be one of {sorted(_VALID_ENZYMES)}, got '{enzyme}'")
    if gestational_age_weeks <= 0:
        raise ValueError(f"gestational_age_weeks must be > 0, got {gestational_age_weeks}")
    if postnatal_age_days < 0:
        raise ValueError(f"postnatal_age_days must be >= 0, got {postnatal_age_days}")

    ga = gestational_age_weeks
    pna = postnatal_age_days

    if enzyme == "CYP3A4":
        # Very low at birth, rises with GA maturation and postnatal age
        f = 0.05 + 0.95 * (1.0 - math.exp(-0.01 * (pna + (ga - 28) * 7)))
    elif enzyme == "CYP2D6":
        # Almost absent at birth, rapid postnatal maturation
        f = 0.01 + 0.99 * (1.0 - math.exp(-0.008 * pna))
    elif enzyme == "CYP1A2":
        # Slowest to mature
        f = 0.01 + 0.99 * (1.0 - math.exp(-0.005 * pna))
    else:  # CYP2C9
        f = 0.1 + 0.9 * (1.0 - math.exp(-0.007 * pna))

    return max(0.0, min(1.0, f))


def neonatal_cl_adjustment(
    cl_adult_L_per_h_per_kg: float,
    weight_kg: float,
    gestational_age_weeks: float,
    postnatal_age_days: float,
    fraction_renal: float = 0.5,
    fraction_hepatic: float = 0.5,
    primary_cyp: str = "CYP3A4",
) -> dict:
    """Adjust clearance for neonatal organ immaturity.

    Args:
        cl_adult_L_per_h_per_kg: Adult clearance in L/h/kg.
        weight_kg: Neonate weight in kg.
        gestational_age_weeks: Gestational age in weeks.
        postnatal_age_days: Postnatal age in days.
        fraction_renal: Fraction of clearance via renal elimination.
        fraction_hepatic: Fraction of clearance via hepatic metabolism.
        primary_cyp: Primary CYP enzyme for hepatic clearance.

    Returns:
        dict with keys: cl_adult_per_kg, cl_adjusted_L_per_h, renal_factor,
        hepatic_factor.
    """
    if weight_kg <= 0:
        raise ValueError(f"weight_kg must be > 0, got {weight_kg}")
    if gestational_age_weeks <= 0:
        raise ValueError(f"gestational_age_weeks must be > 0, got {gestational_age_weeks}")
    if postnatal_age_days < 0:
        raise ValueError(f"postnatal_age_days must be >= 0, got {postnatal_age_days}")
    if primary_cyp not in _VALID_ENZYMES:
        raise ValueError(
            f"primary_cyp must be one of {sorted(_VALID_ENZYMES)}, got '{primary_cyp}'"
        )

    renal_factor = neonatal_gfr(gestational_age_weeks, postnatal_age_days) / _ADULT_GFR
    hepatic_factor = neonatal_cyp_activity(primary_cyp, gestational_age_weeks, postnatal_age_days)

    combined_factor = fraction_renal * renal_factor + fraction_hepatic * hepatic_factor
    cl_adjusted = cl_adult_L_per_h_per_kg * weight_kg * combined_factor

    return {
        "cl_adult_per_kg": cl_adult_L_per_h_per_kg,
        "cl_adjusted_L_per_h": cl_adjusted,
        "renal_factor": renal_factor,
        "hepatic_factor": hepatic_factor,
    }


def neonatal_dose_recommendation(
    drug_name: str,
    adult_dose_mg_per_kg: float,
    weight_kg: float,
    gestational_age_weeks: float,
    postnatal_age_days: float,
    fraction_renal: float = 0.5,
    fraction_hepatic: float = 0.5,
    primary_cyp: str = "CYP3A4",
    dosing_interval_h: float = 12.0,
) -> NeonatalDoseResult:
    """Recommend neonatal dose based on CL adjustment for organ immaturity.

    The neonatal dose per kg is scaled by the combined CL reduction factor
    relative to the adult, preserving exposure (AUC) equivalence.

    Args:
        drug_name: Name of the drug.
        adult_dose_mg_per_kg: Standard adult dose in mg/kg.
        weight_kg: Neonate body weight in kg.
        gestational_age_weeks: Gestational age at birth in weeks.
        postnatal_age_days: Postnatal age in days.
        fraction_renal: Fraction of adult clearance via renal route.
        fraction_hepatic: Fraction of adult clearance via hepatic route.
        primary_cyp: Primary CYP enzyme for hepatic metabolism.
        dosing_interval_h: Dosing interval in hours.

    Returns:
        NeonatalDoseResult with recommended dose and clinical guidance.
    """
    if weight_kg <= 0:
        raise ValueError(f"weight_kg must be > 0, got {weight_kg}")
    if gestational_age_weeks <= 0:
        raise ValueError(f"gestational_age_weeks must be > 0, got {gestational_age_weeks}")
    if postnatal_age_days < 0:
        raise ValueError(f"postnatal_age_days must be >= 0, got {postnatal_age_days}")

    cl_info = neonatal_cl_adjustment(
        cl_adult_L_per_h_per_kg=1.0,  # normalised — we use factor only
        weight_kg=weight_kg,
        gestational_age_weeks=gestational_age_weeks,
        postnatal_age_days=postnatal_age_days,
        fraction_renal=fraction_renal,
        fraction_hepatic=fraction_hepatic,
        primary_cyp=primary_cyp,
    )

    renal_factor = cl_info["renal_factor"]
    hepatic_factor = cl_info["hepatic_factor"]
    combined_factor = fraction_renal * renal_factor + fraction_hepatic * hepatic_factor

    # Neonatal dose proportional to CL reduction (AUC-equivalent approach)
    neonatal_dose_per_kg = adult_dose_mg_per_kg * combined_factor
    neonatal_dose_total = neonatal_dose_per_kg * weight_kg

    # Clinical monitoring guidance
    ga = gestational_age_weeks
    pna = postnatal_age_days

    monitoring_parts = ["TDM recommended"]
    if ga < 32:
        monitoring_parts.append("extreme prematurity — frequent monitoring")
    elif ga < 37:
        monitoring_parts.append("preterm — close monitoring")
    if renal_factor < 0.2:
        monitoring_parts.append("very low renal function — check serum creatinine")
    if hepatic_factor < 0.15:
        monitoring_parts.append("minimal hepatic CYP activity")

    monitoring_recommendation = "; ".join(monitoring_parts)

    notes_parts = [
        f"GA {ga:.0f}wk, PNA {pna:.0f}d",
        f"Renal maturation {renal_factor:.1%}",
        f"{primary_cyp} activity {hepatic_factor:.1%}",
        f"Combined CL factor {combined_factor:.2f}",
    ]
    notes = " | ".join(notes_parts)

    return NeonatalDoseResult(
        drug_name=drug_name,
        weight_kg=weight_kg,
        gestational_age_weeks=ga,
        postnatal_age_days=float(pna),
        adult_dose_mg_per_kg=adult_dose_mg_per_kg,
        neonatal_dose_mg_per_kg=neonatal_dose_per_kg,
        neonatal_dose_mg=neonatal_dose_total,
        cl_renal_factor=renal_factor,
        cl_hepatic_factor=hepatic_factor,
        dosing_interval_h=dosing_interval_h,
        monitoring_recommendation=monitoring_recommendation,
        notes=notes,
    )
