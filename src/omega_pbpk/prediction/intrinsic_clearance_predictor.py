"""
Phase 603 — Intrinsic Clearance Predictor

Predicts hepatic intrinsic clearance (CLint) from physicochemical descriptors
using an empirical QSAR model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class IntrinsicClearanceResult:
    drug_name: str
    logP: float
    mw: float
    psa: float
    clint_mL_per_min_per_mg: float  # intrinsic clearance (microsomal)
    clint_scaled_mL_per_min: float  # scaled to 70 kg human (MPPGL × liver weight)
    hepatic_cl_L_per_h: float  # well-stirred hepatic CL
    fu_mic: float  # fraction unbound in microsome
    extraction_ratio: float  # hepatic ER (0-1)
    predicted_t_half_h: float  # based on hepatic CL + Vd estimate
    high_clearance: bool  # ER > 0.7
    metabolism_class: str  # "low","moderate","high","very_high"
    primary_enzyme: str  # predicted primary CYP
    notes: str


def _validate_inputs(
    mw: float,
    psa: float,
    n_aromatic_rings: int,
    n_hbd: int,
    fu_plasma: float,
    vd_L: float,
) -> None:
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if psa < 0:
        raise ValueError(f"psa must be >= 0, got {psa}")
    if n_aromatic_rings < 0:
        raise ValueError(f"n_aromatic_rings must be >= 0, got {n_aromatic_rings}")
    if n_hbd < 0:
        raise ValueError(f"n_hbd must be >= 0, got {n_hbd}")
    if not (0 < fu_plasma <= 1):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")


def scale_microsomal_to_in_vivo(
    clint_mL_per_min_per_mg_protein: float,
    mppgl: float = 45.0,
    liver_weight_g: float = 1500.0,
) -> float:
    """Scale CLint from microsomal to whole-liver (mL/min)."""
    return clint_mL_per_min_per_mg_protein * mppgl * liver_weight_g / 1000.0


def predict_intrinsic_clearance(
    drug_name: str,
    logP: float,
    mw: float,
    psa: float,
    n_aromatic_rings: int = 0,
    n_hbd: int = 0,
    pka_base: float | None = None,
    fu_plasma: float = 0.5,
    vd_L: float = 50.0,
) -> IntrinsicClearanceResult:
    """Predict hepatic intrinsic clearance from physicochemical descriptors."""
    _validate_inputs(mw, psa, n_aromatic_rings, n_hbd, fu_plasma, vd_L)

    # QSAR model: base log(CLint)
    log_clint = -0.5 * logP + 0.002 * mw - 0.01 * psa + 0.3 * n_aromatic_rings - 0.2 * n_hbd + 1.0

    # Basic drug bonus
    if pka_base is not None and pka_base > 7:
        log_clint += 0.5

    # CLint in mL/min/mg protein, clamped
    clint_mL_per_min_per_mg = max(0.01, min(500.0, 10**log_clint))

    # fu_mic estimation (Austin et al.)
    if logP > 0:
        fu_mic = 1.0 / (1.0 + 0.072 * logP)
    else:
        fu_mic = 1.0
    fu_mic = max(0.01, min(1.0, fu_mic))

    # Scale to in vivo (mL/min)
    clint_scaled_mL_per_min = scale_microsomal_to_in_vivo(clint_mL_per_min_per_mg)

    # Well-stirred hepatic CL
    # QH = 90.0 L/h = 1500 mL/min
    QH_mL_per_min = 1500.0
    fu_clint = fu_plasma * clint_scaled_mL_per_min
    cl_hepatic_mL_per_min = QH_mL_per_min * fu_clint / (QH_mL_per_min + fu_clint)

    # Convert to L/h
    hepatic_cl_L_per_h = cl_hepatic_mL_per_min * 60.0 / 1000.0

    # Extraction ratio (dimensionless)
    extraction_ratio = cl_hepatic_mL_per_min / QH_mL_per_min
    extraction_ratio = max(0.0, min(1.0, extraction_ratio))

    # Half-life
    if hepatic_cl_L_per_h > 0:
        predicted_t_half_h = math.log(2) * vd_L / hepatic_cl_L_per_h
    else:
        predicted_t_half_h = float("inf")

    # Classification
    if extraction_ratio < 0.25:
        metabolism_class = "low"
    elif extraction_ratio < 0.5:
        metabolism_class = "moderate"
    elif extraction_ratio < 0.7:
        metabolism_class = "high"
    else:
        metabolism_class = "very_high"

    high_clearance = extraction_ratio > 0.7

    # Primary enzyme prediction
    if logP > 3 and mw > 300:
        primary_enzyme = "CYP3A4"
    elif pka_base is not None and pka_base > 7:
        primary_enzyme = "CYP2D6"
    elif psa < 50 and mw < 300:
        primary_enzyme = "CYP2C19"
    else:
        primary_enzyme = "CYP3A4"

    notes = (
        f"CLint={clint_mL_per_min_per_mg:.3f} mL/min/mg; "
        f"ER={extraction_ratio:.3f}; "
        f"primary CYP={primary_enzyme}; "
        f"metabolism class={metabolism_class}"
    )

    return IntrinsicClearanceResult(
        drug_name=drug_name,
        logP=logP,
        mw=mw,
        psa=psa,
        clint_mL_per_min_per_mg=clint_mL_per_min_per_mg,
        clint_scaled_mL_per_min=clint_scaled_mL_per_min,
        hepatic_cl_L_per_h=hepatic_cl_L_per_h,
        fu_mic=fu_mic,
        extraction_ratio=extraction_ratio,
        predicted_t_half_h=predicted_t_half_h,
        high_clearance=high_clearance,
        metabolism_class=metabolism_class,
        primary_enzyme=primary_enzyme,
        notes=notes,
    )
