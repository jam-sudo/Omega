"""Body weight-based PK parameter scaling for dose adjustment.

Implements allometric scaling of CL and Vd from a reference body weight to
a patient body weight, and recommends a proportional dose adjustment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["BWScalingResult", "scale_pk_by_weight", "dose_by_weight_band"]


@dataclass(frozen=True)
class BWScalingResult:
    """Result of body-weight allometric scaling."""

    drug_name: str
    reference_weight_kg: float
    patient_weight_kg: float
    reference_cl_L_per_h: float
    scaled_cl_L_per_h: float
    reference_vd_L: float
    scaled_vd_L: float
    cl_scaling_factor: float
    vd_scaling_factor: float
    scaled_t_half_h: float
    recommended_dose_mg: float
    reference_dose_mg: float
    dose_capping: bool
    notes: str


def scale_pk_by_weight(
    drug_name: str,
    reference_weight_kg: float,
    patient_weight_kg: float,
    reference_cl_L_per_h: float,
    reference_vd_L: float,
    reference_dose_mg: float,
    cl_exponent: float = 0.75,
    vd_exponent: float = 1.0,
    max_dose_mg: float | None = None,
    min_dose_mg: float | None = None,
) -> BWScalingResult:
    """Scale PK parameters and dose by body weight using allometric scaling.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    reference_weight_kg : float
        Reference (typical) body weight (kg). Must be > 0.
    patient_weight_kg : float
        Patient body weight (kg). Must be > 0.
    reference_cl_L_per_h : float
        Reference clearance (L/h). Must be > 0.
    reference_vd_L : float
        Reference volume of distribution (L). Must be > 0.
    reference_dose_mg : float
        Reference dose (mg).
    cl_exponent : float
        Allometric exponent for CL scaling (default 0.75). Must be > 0.
    vd_exponent : float
        Allometric exponent for Vd scaling (default 1.0). Must be > 0.
    max_dose_mg : float or None
        Maximum dose cap (mg). Applied after scaling if set.
    min_dose_mg : float or None
        Minimum dose floor (mg). Applied after scaling if set.

    Returns
    -------
    BWScalingResult
    """
    # --- Validation ---
    if reference_weight_kg <= 0:
        raise ValueError("reference_weight_kg must be > 0")
    if patient_weight_kg <= 0:
        raise ValueError("patient_weight_kg must be > 0")
    if reference_cl_L_per_h <= 0:
        raise ValueError("reference_cl_L_per_h must be > 0")
    if reference_vd_L <= 0:
        raise ValueError("reference_vd_L must be > 0")
    if cl_exponent <= 0:
        raise ValueError("cl_exponent must be > 0")
    if vd_exponent <= 0:
        raise ValueError("vd_exponent must be > 0")

    weight_ratio = patient_weight_kg / reference_weight_kg

    cl_factor = weight_ratio**cl_exponent
    vd_factor = weight_ratio**vd_exponent

    scaled_cl = reference_cl_L_per_h * cl_factor
    scaled_vd = reference_vd_L * vd_factor

    ke = scaled_cl / scaled_vd
    scaled_t_half = math.log(2) / ke if ke > 0 else math.inf

    # Recommended dose is proportional to CL (maintains same Css at steady state)
    recommended_dose = reference_dose_mg * cl_factor

    dose_capping = False
    notes_parts: list[str] = []

    if max_dose_mg is not None and recommended_dose > max_dose_mg:
        recommended_dose = max_dose_mg
        dose_capping = True
        notes_parts.append(f"dose capped at max {max_dose_mg:.1f} mg")
    elif min_dose_mg is not None and recommended_dose < min_dose_mg:
        recommended_dose = min_dose_mg
        dose_capping = True
        notes_parts.append(f"dose floored at min {min_dose_mg:.1f} mg")

    notes_parts.append(
        f"CL factor {cl_factor:.3f} (exp={cl_exponent}); "
        f"Vd factor {vd_factor:.3f} (exp={vd_exponent}); "
        f"scaled t½ {scaled_t_half:.2f} h"
    )
    notes = "; ".join(notes_parts)

    return BWScalingResult(
        drug_name=drug_name,
        reference_weight_kg=reference_weight_kg,
        patient_weight_kg=patient_weight_kg,
        reference_cl_L_per_h=reference_cl_L_per_h,
        scaled_cl_L_per_h=scaled_cl,
        reference_vd_L=reference_vd_L,
        scaled_vd_L=scaled_vd,
        cl_scaling_factor=cl_factor,
        vd_scaling_factor=vd_factor,
        scaled_t_half_h=scaled_t_half,
        recommended_dose_mg=recommended_dose,
        reference_dose_mg=reference_dose_mg,
        dose_capping=dose_capping,
        notes=notes,
    )


def dose_by_weight_band(
    drug_name: str,
    weight_bands_kg: list[tuple[float, float]],
    reference_weight_kg: float,
    reference_dose_mg: float,
    reference_cl_L_per_h: float,
    reference_vd_L: float,
    **kwargs,
) -> list[BWScalingResult]:
    """Calculate recommended dose for the midpoint of each weight band.

    Parameters
    ----------
    drug_name : str
        Drug name.
    weight_bands_kg : list of (min_kg, max_kg) tuples
        Weight bands. Midpoint = (min_kg + max_kg) / 2.
    reference_weight_kg : float
        Reference body weight (kg).
    reference_dose_mg : float
        Reference dose (mg).
    reference_cl_L_per_h : float
        Reference clearance (L/h).
    reference_vd_L : float
        Reference volume of distribution (L).
    **kwargs
        Additional keyword arguments forwarded to :func:`scale_pk_by_weight`.

    Returns
    -------
    list[BWScalingResult]
        One result per weight band.
    """
    results: list[BWScalingResult] = []
    for lo, hi in weight_bands_kg:
        midpoint = (lo + hi) / 2.0
        result = scale_pk_by_weight(
            drug_name=drug_name,
            reference_weight_kg=reference_weight_kg,
            patient_weight_kg=midpoint,
            reference_cl_L_per_h=reference_cl_L_per_h,
            reference_vd_L=reference_vd_L,
            reference_dose_mg=reference_dose_mg,
            **kwargs,
        )
        results.append(result)
    return results
