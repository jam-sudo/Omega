"""Physiological blood flow and organ volume reference values for PBPK model setup."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PhysiologicalParameters:
    species: str
    body_weight_kg: float
    cardiac_output_L_per_h: float
    organ_flows: dict  # fractions of CO
    organ_volumes_L: dict  # absolute volumes in L
    plasma_volume_L: float
    total_blood_volume_L: float
    hematocrit: float
    notes: str


# Human reference values at 70 kg
_HUMAN_CO_L_PER_H = 6.5 * 60  # 390 L/h
_HUMAN_BW_KG = 70.0
_HUMAN_HEMATOCRIT = 0.45
_HUMAN_PLASMA_VOLUME_L = 3.0

_HUMAN_ORGAN_FLOWS: dict[str, float] = {
    "liver": 0.215,
    "kidney": 0.194,
    "muscle": 0.170,
    "fat": 0.052,
    "brain": 0.114,
    "gut": 0.146,
    "lung": 1.0,
    "heart": 0.040,
    "skin": 0.069,
}

_HUMAN_ORGAN_VOLUMES_L: dict[str, float] = {
    "liver": 1.69,
    "kidney": 0.280,
    "muscle": 28.0,
    "fat": 14.5,
    "brain": 1.45,
    "gut": 1.15,
    "lung": 0.530,
    "heart": 0.310,
    "plasma": 3.0,
}


def get_human_parameters(body_weight_kg: float = 70.0) -> PhysiologicalParameters:
    """Return allometric-scaled human physiological parameters."""
    if body_weight_kg <= 0:
        raise ValueError(f"body_weight_kg must be > 0, got {body_weight_kg}")

    ratio = body_weight_kg / _HUMAN_BW_KG
    co = _HUMAN_CO_L_PER_H * (ratio**0.75)
    volumes = {organ: vol * ratio for organ, vol in _HUMAN_ORGAN_VOLUMES_L.items()}
    plasma_vol = _HUMAN_PLASMA_VOLUME_L * ratio
    total_blood = plasma_vol / (1.0 - _HUMAN_HEMATOCRIT)

    return PhysiologicalParameters(
        species="human",
        body_weight_kg=body_weight_kg,
        cardiac_output_L_per_h=co,
        organ_flows=dict(_HUMAN_ORGAN_FLOWS),
        organ_volumes_L=volumes,
        plasma_volume_L=plasma_vol,
        total_blood_volume_L=total_blood,
        hematocrit=_HUMAN_HEMATOCRIT,
        notes="ICRP 2002 / Davies & Morris 1993 reference values, allometrically scaled",
    )


def get_rat_parameters(body_weight_kg: float = 0.25) -> PhysiologicalParameters:
    """Return allometric-scaled rat physiological parameters."""
    if body_weight_kg <= 0:
        raise ValueError(f"body_weight_kg must be > 0, got {body_weight_kg}")

    ratio = body_weight_kg / _HUMAN_BW_KG
    co = _HUMAN_CO_L_PER_H * (ratio**0.75)
    volumes = {organ: vol * ratio for organ, vol in _HUMAN_ORGAN_VOLUMES_L.items()}
    plasma_vol = _HUMAN_PLASMA_VOLUME_L * ratio
    total_blood = plasma_vol / (1.0 - _HUMAN_HEMATOCRIT)

    return PhysiologicalParameters(
        species="rat",
        body_weight_kg=body_weight_kg,
        cardiac_output_L_per_h=co,
        organ_flows=dict(_HUMAN_ORGAN_FLOWS),
        organ_volumes_L=volumes,
        plasma_volume_L=plasma_vol,
        total_blood_volume_L=total_blood,
        hematocrit=_HUMAN_HEMATOCRIT,
        notes="Rat parameters scaled from human reference using allometry (bw/70)^exponent",
    )


def get_mouse_parameters(body_weight_kg: float = 0.025) -> PhysiologicalParameters:
    """Return allometric-scaled mouse physiological parameters."""
    if body_weight_kg <= 0:
        raise ValueError(f"body_weight_kg must be > 0, got {body_weight_kg}")

    ratio = body_weight_kg / _HUMAN_BW_KG
    co = _HUMAN_CO_L_PER_H * (ratio**0.75)
    volumes = {organ: vol * ratio for organ, vol in _HUMAN_ORGAN_VOLUMES_L.items()}
    plasma_vol = _HUMAN_PLASMA_VOLUME_L * ratio
    total_blood = plasma_vol / (1.0 - _HUMAN_HEMATOCRIT)

    return PhysiologicalParameters(
        species="mouse",
        body_weight_kg=body_weight_kg,
        cardiac_output_L_per_h=co,
        organ_flows=dict(_HUMAN_ORGAN_FLOWS),
        organ_volumes_L=volumes,
        plasma_volume_L=plasma_vol,
        total_blood_volume_L=total_blood,
        hematocrit=_HUMAN_HEMATOCRIT,
        notes="Mouse parameters scaled from human reference using allometry (bw/70)^exponent",
    )


def scale_to_body_weight(
    base_params: PhysiologicalParameters,
    new_weight_kg: float,
    flow_exponent: float = 0.75,
    volume_exponent: float = 1.0,
) -> PhysiologicalParameters:
    """Allometrically scale parameters to a new body weight."""
    if new_weight_kg <= 0:
        raise ValueError(f"new_weight_kg must be > 0, got {new_weight_kg}")

    ratio = new_weight_kg / base_params.body_weight_kg
    new_co = base_params.cardiac_output_L_per_h * (ratio**flow_exponent)

    # Flow fractions are preserved; absolute flows scale with CO
    new_organ_flows = dict(base_params.organ_flows)

    new_volumes = {
        organ: vol * (ratio**volume_exponent) for organ, vol in base_params.organ_volumes_L.items()
    }
    new_plasma = base_params.plasma_volume_L * (ratio**volume_exponent)
    new_total_blood = new_plasma / (1.0 - base_params.hematocrit)

    return PhysiologicalParameters(
        species=base_params.species,
        body_weight_kg=new_weight_kg,
        cardiac_output_L_per_h=new_co,
        organ_flows=new_organ_flows,
        organ_volumes_L=new_volumes,
        plasma_volume_L=new_plasma,
        total_blood_volume_L=new_total_blood,
        hematocrit=base_params.hematocrit,
        notes=base_params.notes,
    )


def compute_organ_clearance(
    organ: str,
    intrinsic_cl_mL_per_min: float,
    fu: float,
    blood_flow_L_per_h: float,
    model: str = "well_stirred",
) -> float:
    """Compute organ clearance via well-stirred or parallel-tube model (mL/min).

    Parameters
    ----------
    organ:
        Name of the organ (informational; used for validation).
    intrinsic_cl_mL_per_min:
        Intrinsic clearance in mL/min.
    fu:
        Unbound fraction in plasma (0 < fu <= 1).
    blood_flow_L_per_h:
        Organ blood flow in L/h.
    model:
        Either "well_stirred" or "parallel_tube".

    Returns
    -------
    float
        Organ clearance in mL/min.
    """
    if fu <= 0 or fu > 1:
        raise ValueError(f"fu must be in (0, 1], got {fu}")
    if model not in {"well_stirred", "parallel_tube"}:
        raise ValueError(f"model must be 'well_stirred' or 'parallel_tube', got '{model}'")
    if blood_flow_L_per_h <= 0:
        raise ValueError(f"blood_flow_L_per_h must be > 0, got {blood_flow_L_per_h}")
    if intrinsic_cl_mL_per_min < 0:
        raise ValueError(f"intrinsic_cl_mL_per_min must be >= 0, got {intrinsic_cl_mL_per_min}")

    # Convert blood flow from L/h to mL/min
    q_mL_per_min = blood_flow_L_per_h * 1000.0 / 60.0

    if model == "well_stirred":
        denom = q_mL_per_min + fu * intrinsic_cl_mL_per_min
        if denom == 0:
            return 0.0
        cl = q_mL_per_min * fu * intrinsic_cl_mL_per_min / denom
    else:  # parallel_tube
        exponent = fu * intrinsic_cl_mL_per_min / q_mL_per_min if q_mL_per_min > 0 else 0.0
        cl = q_mL_per_min * (1.0 - math.exp(-exponent))

    return cl
