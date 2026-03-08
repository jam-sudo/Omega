"""Neonatal drug dosing prediction.

Predicts drug PK parameters and appropriate doses for neonates (0-28 days)
and young infants (29 days - 12 months), accounting for physiological
differences from adults using maturation-based scaling.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NeonatalDosingResult:
    """Result of neonatal dosing prediction."""

    drug_name: str
    age_days: int
    weight_kg: float
    gestational_age_weeks: float
    cl_adult_L_per_h: float
    vd_adult_L: float
    cl_neonate_L_per_h: float
    vd_neonate_L: float
    dose_adult_mg_kg: float
    dose_neonate_mg_kg: float
    dose_neonate_total_mg: float
    t_half_neonate_h: float
    renal_maturation_factor: float
    hepatic_maturation_factor: float
    total_cl_scaling_factor: float
    dosing_interval_h: float
    safety_flag: str
    notes: str


def predict_neonatal_dosing(
    drug_name: str,
    cl_adult_L_per_h: float,
    vd_adult_L: float,
    dose_adult_mg_per_kg: float,
    age_days: int,
    weight_kg: float,
    gestational_age_weeks: float = 40.0,
    renal_fraction: float = 0.5,
    hepatic_fraction: float = 0.5,
) -> NeonatalDosingResult:
    """Predict neonatal dosing based on maturation scaling.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    cl_adult_L_per_h : float
        Adult clearance (L/h). Must be > 0.
    vd_adult_L : float
        Adult volume of distribution (L). Must be > 0.
    dose_adult_mg_per_kg : float
        Adult dose (mg/kg).
    age_days : int
        Postnatal age in days. Must be >= 0.
    weight_kg : float
        Neonatal weight (kg). Must be > 0.
    gestational_age_weeks : float
        Gestational age at birth (weeks). Must be in [22, 44].
    renal_fraction : float
        Fraction of clearance via renal route.
    hepatic_fraction : float
        Fraction of clearance via hepatic route.

    Returns
    -------
    NeonatalDosingResult
    """
    if cl_adult_L_per_h <= 0:
        raise ValueError(f"cl_adult_L_per_h must be > 0, got {cl_adult_L_per_h}")
    if vd_adult_L <= 0:
        raise ValueError(f"vd_adult_L must be > 0, got {vd_adult_L}")
    if age_days < 0:
        raise ValueError(f"age_days must be >= 0, got {age_days}")
    if weight_kg <= 0:
        raise ValueError(f"weight_kg must be > 0, got {weight_kg}")
    if not (22 <= gestational_age_weeks <= 44):
        raise ValueError(f"gestational_age_weeks must be in [22, 44], got {gestational_age_weeks}")

    # Postmenstrual age in weeks
    pna_weeks = age_days / 7.0
    pma = gestational_age_weeks + pna_weeks

    # Renal maturation (GFR-based, Anderson & Holford)
    tm50_renal = 47.7
    hill_renal = 3.4
    renal_mat = pma**hill_renal / (tm50_renal**hill_renal + pma**hill_renal)

    # Hepatic maturation (CYP3A4-like)
    tm50_hepatic = 42.0
    hill_hepatic = 2.0
    hepatic_mat = pma**hill_hepatic / (tm50_hepatic**hill_hepatic + pma**hill_hepatic)

    # Total CL scaling
    scaling = renal_fraction * renal_mat + hepatic_fraction * hepatic_mat
    scaling = max(0.01, min(1.0, scaling))

    # Allometric weight scaling
    weight_scalar = (weight_kg / 70.0) ** 0.75

    # Neonatal PK parameters
    cl_neonate = cl_adult_L_per_h * scaling * weight_scalar
    vd_neonate = vd_adult_L * (weight_kg / 70.0)

    # Dose adjustment
    dose_neonate_mg_kg = dose_adult_mg_per_kg * scaling
    dose_neonate_total = dose_neonate_mg_kg * weight_kg

    # Half-life
    t_half = 0.693 * vd_neonate / cl_neonate

    # Dosing interval
    if t_half < 8.0:
        interval = 6.0
    elif t_half < 16.0:
        interval = 12.0
    else:
        interval = 24.0

    # Safety flag
    if scaling < 0.3 or weight_kg < 1.0:
        safety = "caution"
    else:
        safety = "standard"

    notes_parts = [f"PMA = {pma:.1f} weeks"]
    if scaling < 0.3:
        notes_parts.append("immature metabolism")
    if weight_kg < 1.0:
        notes_parts.append("very low birth weight")

    return NeonatalDosingResult(
        drug_name=drug_name,
        age_days=age_days,
        weight_kg=weight_kg,
        gestational_age_weeks=gestational_age_weeks,
        cl_adult_L_per_h=cl_adult_L_per_h,
        vd_adult_L=vd_adult_L,
        cl_neonate_L_per_h=cl_neonate,
        vd_neonate_L=vd_neonate,
        dose_adult_mg_kg=dose_adult_mg_per_kg,
        dose_neonate_mg_kg=dose_neonate_mg_kg,
        dose_neonate_total_mg=dose_neonate_total,
        t_half_neonate_h=t_half,
        renal_maturation_factor=renal_mat,
        hepatic_maturation_factor=hepatic_mat,
        total_cl_scaling_factor=scaling,
        dosing_interval_h=interval,
        safety_flag=safety,
        notes="; ".join(notes_parts),
    )
