"""Comprehensive organ impairment PK adjustment module."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "OrganImpairmentResult",
    "adjust_for_organ_impairment",
]


@dataclass
class OrganImpairmentResult:
    drug_name: str
    renal_function: str
    hepatic_function: str
    cardiac_function: str
    cl_adjustment_factor: float
    vd_adjustment_factor: float
    t_half_adjustment_factor: float
    adjusted_dose_mg: float
    adjusted_interval_h: float
    dose_recommendation: str
    pk_parameters_adjusted: dict = field(default_factory=dict)


def _renal_cl_factor(crcl_mL_per_min: float, fraction_renal: float = 0.5) -> float:
    """CL scaling factor based on creatinine clearance.

    Returns 1 - fraction_renal * (1 - crcl/100), clamped to [0.1, 1.0].
    """
    factor = 1.0 - fraction_renal * (1.0 - crcl_mL_per_min / 100.0)
    return float(np.clip(factor, 0.1, 1.0))


def _hepatic_cl_factor(child_pugh_score: int) -> float:
    """CL scaling factor based on Child-Pugh score."""
    if child_pugh_score <= 6:
        return 1.0
    elif child_pugh_score <= 9:
        return 0.75
    elif child_pugh_score <= 12:
        return 0.50
    else:
        return 0.25


def _cardiac_cl_factor(nyha_class: int) -> float:
    """CL scaling factor based on NYHA heart failure class."""
    mapping = {1: 1.0, 2: 0.85, 3: 0.70, 4: 0.50}
    return mapping.get(nyha_class, 1.0)


def _renal_function_label(crcl: float) -> str:
    if crcl >= 90:
        return "normal"
    elif crcl >= 60:
        return "mild"
    elif crcl >= 30:
        return "moderate"
    elif crcl > 0:
        return "severe"
    else:
        return "ESRD"


def _hepatic_function_label(score: int) -> str:
    if score <= 6:
        return "normal"
    elif score <= 9:
        return "child_pugh_A"
    elif score <= 12:
        return "child_pugh_B"
    else:
        return "child_pugh_C"


def _cardiac_function_label(nyha: int) -> str:
    mapping = {1: "normal", 2: "mild", 3: "moderate", 4: "severe"}
    return mapping.get(nyha, "normal")


def _build_dose_recommendation(
    drug_name: str,
    cl_factor: float,
    renal_label: str,
    hepatic_label: str,
    cardiac_label: str,
    adjusted_dose_mg: float,
    adjusted_interval_h: float,
    original_dose_mg: float,
    original_interval_h: float,
) -> str:
    parts = [f"Drug: {drug_name}"]

    impairments = []
    if renal_label != "normal":
        impairments.append(f"renal ({renal_label})")
    if hepatic_label not in ("normal", "child_pugh_A"):
        impairments.append(f"hepatic ({hepatic_label})")
    if cardiac_label != "normal":
        impairments.append(f"cardiac ({cardiac_label})")

    if not impairments:
        parts.append("No significant organ impairment — standard dosing applies.")
    else:
        parts.append(f"Organ impairment: {', '.join(impairments)}.")
        if abs(adjusted_dose_mg - original_dose_mg) > 0.01:
            parts.append(
                f"Reduce dose from {original_dose_mg:.1f} mg to {adjusted_dose_mg:.1f} mg "
                f"(CL factor: {cl_factor:.2f})."
            )
        if abs(adjusted_interval_h - original_interval_h) > 0.01:
            parts.append(
                f"Extend dosing interval from {original_interval_h:.1f} h to "
                f"{adjusted_interval_h:.1f} h."
            )
        if cl_factor < 0.25:
            parts.append("Consider therapeutic drug monitoring.")

    return " ".join(parts)


def adjust_for_organ_impairment(
    drug_name: str,
    dose_mg: float,
    dosing_interval_h: float,
    cl_L_per_h: float,
    vd_L: float,
    crcl_mL_per_min: float = 100.0,
    child_pugh_score: int = 5,
    nyha_class: int = 1,
    fraction_renal: float = 0.3,
    fraction_hepatic: float = 0.6,
) -> OrganImpairmentResult:
    """Adjust PK parameters and dosing for organ impairment.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Standard dose in mg.
    dosing_interval_h:
        Standard dosing interval in hours.
    cl_L_per_h:
        Clearance in L/h for a normal patient.
    vd_L:
        Volume of distribution in L for a normal patient.
    crcl_mL_per_min:
        Creatinine clearance in mL/min (default 100 = normal).
    child_pugh_score:
        Child-Pugh score (5-15; <=6 = normal).
    nyha_class:
        NYHA heart failure class (1-4; 1 = normal).
    fraction_renal:
        Fraction of drug eliminated renally (0-1).
    fraction_hepatic:
        Fraction of drug eliminated hepatically (0-1).

    Returns
    -------
    OrganImpairmentResult
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")

    renal_factor = _renal_cl_factor(crcl_mL_per_min, fraction_renal)
    hepatic_factor = _hepatic_cl_factor(child_pugh_score)
    # Scale hepatic factor by fraction_hepatic: full effect only if drug is
    # entirely hepatically cleared.
    hepatic_factor = 1.0 - fraction_hepatic * (1.0 - hepatic_factor)
    cardiac_factor = _cardiac_cl_factor(nyha_class)

    cl_factor = renal_factor * hepatic_factor * cardiac_factor
    cl_factor = float(np.clip(cl_factor, 0.1, 1.0))

    # Cardiac edema increases Vd
    nyha_factor = cardiac_factor  # same as _cardiac_cl_factor result
    vd_factor = 1.0 + 0.2 * (1.0 - nyha_factor)

    t_half_factor = vd_factor / cl_factor

    adjusted_dose_mg = dose_mg * cl_factor
    adjusted_interval_h = dosing_interval_h if cl_factor >= 0.5 else dosing_interval_h * 2.0

    adjusted_cl = cl_L_per_h * cl_factor
    adjusted_vd = vd_L * vd_factor
    adjusted_t_half = (adjusted_vd * np.log(2)) / adjusted_cl

    renal_label = _renal_function_label(crcl_mL_per_min)
    hepatic_label = _hepatic_function_label(child_pugh_score)
    cardiac_label = _cardiac_function_label(nyha_class)

    recommendation = _build_dose_recommendation(
        drug_name=drug_name,
        cl_factor=cl_factor,
        renal_label=renal_label,
        hepatic_label=hepatic_label,
        cardiac_label=cardiac_label,
        adjusted_dose_mg=adjusted_dose_mg,
        adjusted_interval_h=adjusted_interval_h,
        original_dose_mg=dose_mg,
        original_interval_h=dosing_interval_h,
    )

    return OrganImpairmentResult(
        drug_name=drug_name,
        renal_function=renal_label,
        hepatic_function=hepatic_label,
        cardiac_function=cardiac_label,
        cl_adjustment_factor=cl_factor,
        vd_adjustment_factor=vd_factor,
        t_half_adjustment_factor=t_half_factor,
        adjusted_dose_mg=adjusted_dose_mg,
        adjusted_interval_h=adjusted_interval_h,
        dose_recommendation=recommendation,
        pk_parameters_adjusted={
            "CL_L_per_h": adjusted_cl,
            "Vd_L": adjusted_vd,
            "t_half_h": float(adjusted_t_half),
        },
    )
