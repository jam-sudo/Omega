"""Volume of Distribution Scaling (Phase 622).

Predict volume of distribution from physicochemical properties and scale
across species using allometric principles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class VdScalingResult:
    """Result of volume of distribution prediction and scaling."""

    drug_name: str
    species: str
    body_weight_kg: float
    logP: float
    fu_plasma: float
    pka: float
    drug_type: str  # "acid", "base", "neutral"
    vd_predicted_L_per_kg: float
    vd_predicted_L: float  # vd_L_per_kg * body_weight_kg
    vd_class: str  # "low", "moderate", "high", "very_high"
    main_distribution_tissue: str
    t_half_predicted_h: float  # ln(2)*vd_L/cl_L_per_h if cl>0, else 0.0
    fraction_in_plasma: float  # plasma_vol_L / vd_predicted_L, clamped [0, 1]
    notes: str


# Species parameters: (body_weight_kg, allometric_scale_factor)
_SPECIES_DATA: dict[str, tuple[float, float]] = {
    "human": (70.0, 1.00),
    "rat": (0.25, 0.85),
    "mouse": (0.02, 0.80),
    "dog": (10.0, 0.95),
    "monkey": (5.0, 0.95),
}

_PLASMA_VOLUME_FRACTION = 0.045  # L/kg (4.5% of body weight)


def _classify_vd(vd_L_per_kg: float) -> tuple[str, str]:
    """Return (vd_class, main_distribution_tissue)."""
    if vd_L_per_kg < 0.3:
        return "low", "plasma"
    elif vd_L_per_kg < 1.0:
        return "moderate", "well-perfused tissues"
    else:
        return "high" if vd_L_per_kg < 10.0 else "very_high", "extensive tissue binding"


def predict_vd_scaling(
    drug_name: str,
    logP: float,
    fu_plasma: float,
    body_weight_kg: float,
    pka: float = 7.0,
    drug_type: str = "neutral",
    species: str = "human",
    cl_L_per_h: float = 0.0,
) -> VdScalingResult:
    """Predict volume of distribution and scale across species.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    logP:
        Octanol-water partition coefficient.
    fu_plasma:
        Fraction unbound in plasma (0 < fu_plasma <= 1).
    body_weight_kg:
        Body weight in kg.
    pka:
        Acid dissociation constant.
    drug_type:
        One of "acid", "base", "neutral".
    species:
        Target species for scaling.
    cl_L_per_h:
        Total clearance in L/h (for t_half calculation). 0 disables.

    Returns
    -------
    VdScalingResult
    """
    # --- Validation ---
    if body_weight_kg <= 0:
        raise ValueError(f"body_weight_kg must be > 0, got {body_weight_kg}")
    if not (0.0 < fu_plasma <= 1.0):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if drug_type not in ("acid", "base", "neutral"):
        raise ValueError(f"drug_type must be 'acid', 'base', or 'neutral', got {drug_type!r}")

    # --- Base Vd prediction (Oie-Tozer simplified) ---
    if drug_type == "neutral":
        vd_base = 0.3 + logP * 0.2
        vd_base = max(0.1, min(5.0, vd_base))
    elif drug_type == "acid":
        vd_base = 0.1 + fu_plasma * 0.2 + logP * 0.1
        vd_base = max(0.1, min(5.0, vd_base))
    else:  # base
        vd_base = 0.5 + logP * 0.5 + (1.0 - fu_plasma) * 2.0
        vd_base = max(0.1, min(5.0, vd_base))

    # Unbound drugs distribute more
    vd_per_kg = max(0.05, vd_base) / fu_plasma
    vd_per_kg = max(0.05, min(50.0, vd_per_kg))

    # --- Species allometric scaling ---
    if species in _SPECIES_DATA:
        _, scale = _SPECIES_DATA[species]
    else:
        scale = 1.0

    vd_predicted_L_per_kg = vd_per_kg * scale
    vd_predicted_L = vd_predicted_L_per_kg * body_weight_kg

    # --- Classification ---
    vd_class, main_tissue = _classify_vd(vd_predicted_L_per_kg)

    # --- t_half ---
    if cl_L_per_h > 0.0:
        t_half_h = math.log(2.0) * vd_predicted_L / cl_L_per_h
    else:
        t_half_h = 0.0

    # --- Fraction in plasma ---
    plasma_vol_L = body_weight_kg * _PLASMA_VOLUME_FRACTION
    if vd_predicted_L > 0.0:
        fraction_in_plasma = min(1.0, max(0.0, plasma_vol_L / vd_predicted_L))
    else:
        fraction_in_plasma = 1.0

    notes = (
        f"Oie-Tozer simplified Vd prediction for {drug_type} drug. "
        f"Species: {species} ({body_weight_kg} kg), allometric scale={scale}. "
        f"fu_plasma={fu_plasma:.3f}, logP={logP:.2f}."
    )

    return VdScalingResult(
        drug_name=drug_name,
        species=species,
        body_weight_kg=body_weight_kg,
        logP=logP,
        fu_plasma=fu_plasma,
        pka=pka,
        drug_type=drug_type,
        vd_predicted_L_per_kg=vd_predicted_L_per_kg,
        vd_predicted_L=vd_predicted_L,
        vd_class=vd_class,
        main_distribution_tissue=main_tissue,
        t_half_predicted_h=t_half_h,
        fraction_in_plasma=fraction_in_plasma,
        notes=notes,
    )


def scale_across_species(
    drug_name: str,
    logP: float,
    fu_plasma: float,
    pka: float = 7.0,
    drug_type: str = "neutral",
    cl_L_per_h: float = 0.0,
) -> list:
    """Predict Vd across standard species sorted by vd_predicted_L_per_kg.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    logP:
        Octanol-water partition coefficient.
    fu_plasma:
        Fraction unbound in plasma.
    pka:
        Acid dissociation constant.
    drug_type:
        One of "acid", "base", "neutral".
    cl_L_per_h:
        Clearance in L/h (per species weight, used as-is for t_half). 0 disables.

    Returns
    -------
    list of VdScalingResult sorted ascending by vd_predicted_L_per_kg.
    """
    standard_species = ["human", "rat", "mouse", "dog", "monkey"]
    results = []
    for sp in standard_species:
        bw = _SPECIES_DATA[sp][0]
        result = predict_vd_scaling(
            drug_name=drug_name,
            logP=logP,
            fu_plasma=fu_plasma,
            body_weight_kg=bw,
            pka=pka,
            drug_type=drug_type,
            species=sp,
            cl_L_per_h=cl_L_per_h,
        )
        results.append(result)

    results.sort(key=lambda r: r.vd_predicted_L_per_kg)
    return results
