"""Phase 292 — Hepatic Extraction Ratio Calculator.

Calculate hepatic extraction ratio (EH) and first-pass effect using the
well-stirred, parallel-tube, and dispersion models. Compare models and
assess high/intermediate/low extraction.
"""

import math
from dataclasses import dataclass

_VALID_MODELS = {"well_stirred", "parallel_tube", "dispersion"}


@dataclass(frozen=True)
class HepaticExtractionResult:
    """Result of hepatic extraction ratio calculation."""

    drug_name: str
    model: str
    cl_int_mL_per_min_per_g: float
    fu_plasma: float
    q_hepatic_L_per_h: float
    extraction_ratio: float
    f_oral: float
    cl_hepatic_L_per_h: float
    t_half_estimate_h: float
    extraction_class: str
    dosing_implication: str
    notes: str


def _validate_inputs(
    cl_int_mL_per_min_per_g: float,
    fu_plasma: float,
    q_hepatic_L_per_h: float,
    liver_weight_g: float,
    model: str,
) -> None:
    """Raise ValueError if any input is invalid."""
    if cl_int_mL_per_min_per_g <= 0:
        raise ValueError(f"cl_int_mL_per_min_per_g must be > 0, got {cl_int_mL_per_min_per_g}")
    if not (0 < fu_plasma <= 1):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if q_hepatic_L_per_h <= 0:
        raise ValueError(f"q_hepatic_L_per_h must be > 0, got {q_hepatic_L_per_h}")
    if liver_weight_g <= 0:
        raise ValueError(f"liver_weight_g must be > 0, got {liver_weight_g}")
    if model not in _VALID_MODELS:
        raise ValueError(f"model must be one of {_VALID_MODELS}, got '{model}'")


def _compute_eh(fu_cl_int: float, q: float, model: str) -> float:
    """Compute hepatic extraction ratio for a given model.

    Parameters
    ----------
    fu_cl_int:
        fu_plasma * CL_int_total (L/h)
    q:
        Hepatic blood flow (L/h)
    model:
        One of "well_stirred", "parallel_tube", "dispersion"
    """
    if model == "well_stirred":
        eh = fu_cl_int / (q + fu_cl_int)
    elif model == "parallel_tube":
        eh = 1.0 - math.exp(-fu_cl_int / q)
    else:  # dispersion
        dn = fu_cl_int / q
        eh = 1.0 - math.exp(-dn / (1.0 + 0.5 * dn))
    # Clamp to [0, 1] for numerical safety
    return max(0.0, min(1.0, eh))


def calculate_hepatic_extraction(
    drug_name: str,
    cl_int_mL_per_min_per_g: float,
    fu_plasma: float,
    q_hepatic_L_per_h: float,
    model: str = "well_stirred",
    liver_weight_g: float = 1500.0,
) -> HepaticExtractionResult:
    """Calculate hepatic extraction ratio and first-pass effect.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    cl_int_mL_per_min_per_g:
        Intrinsic clearance per gram of liver tissue (mL/min/g). Must be > 0.
    fu_plasma:
        Unbound fraction in plasma (0, 1].
    q_hepatic_L_per_h:
        Hepatic blood flow (L/h). Must be > 0.
    model:
        Hepatic model: "well_stirred", "parallel_tube", or "dispersion".
    liver_weight_g:
        Total liver weight in grams (default 1500 g).

    Returns
    -------
    HepaticExtractionResult
    """
    _validate_inputs(cl_int_mL_per_min_per_g, fu_plasma, q_hepatic_L_per_h, liver_weight_g, model)

    # Convert intrinsic clearance to L/h
    # cl_int (mL/min/g) * liver_weight (g) * 60 (min/h) / 1000 (mL/L)
    cl_int_total_L_per_h = cl_int_mL_per_min_per_g * liver_weight_g * 60.0 / 1000.0

    fu_cl_int = fu_plasma * cl_int_total_L_per_h
    q = q_hepatic_L_per_h

    eh = _compute_eh(fu_cl_int, q, model)
    f_oral = 1.0 - eh
    cl_hepatic = eh * q  # L/h

    # t½ assuming Vd = 50 L
    if cl_hepatic > 0:
        t_half_estimate_h = 0.693 * 50.0 / cl_hepatic
    else:
        t_half_estimate_h = float("inf")

    if eh > 0.7:
        extraction_class = "high"
        dosing_implication = "high first-pass: IV preferred"
    elif eh > 0.3:
        extraction_class = "intermediate"
        dosing_implication = "moderate first-pass: consider dose adjustment"
    else:
        extraction_class = "low"
        dosing_implication = "low first-pass: oral dosing suitable"

    notes = (
        f"Drug '{drug_name}' | Model: {model} | "
        f"CL_int_total={cl_int_total_L_per_h:.2f} L/h | "
        f"fu*CL_int={fu_cl_int:.2f} L/h | Q={q:.1f} L/h | "
        f"EH={eh:.3f} ({extraction_class}) | "
        f"F_oral={f_oral:.3f} | CL_hepatic={cl_hepatic:.2f} L/h"
    )

    return HepaticExtractionResult(
        drug_name=drug_name,
        model=model,
        cl_int_mL_per_min_per_g=cl_int_mL_per_min_per_g,
        fu_plasma=fu_plasma,
        q_hepatic_L_per_h=q_hepatic_L_per_h,
        extraction_ratio=eh,
        f_oral=f_oral,
        cl_hepatic_L_per_h=cl_hepatic,
        t_half_estimate_h=t_half_estimate_h,
        extraction_class=extraction_class,
        dosing_implication=dosing_implication,
        notes=notes,
    )


def compare_models(
    drug_name: str,
    cl_int_mL_per_min_per_g: float,
    fu_plasma: float,
    q_hepatic_L_per_h: float,
) -> list:
    """Compare all three hepatic models for a given drug.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    cl_int_mL_per_min_per_g:
        Intrinsic clearance per gram of liver tissue (mL/min/g).
    fu_plasma:
        Unbound fraction in plasma.
    q_hepatic_L_per_h:
        Hepatic blood flow (L/h).

    Returns
    -------
    list of HepaticExtractionResult for all 3 models, sorted by
    extraction_ratio descending.
    """
    results = [
        calculate_hepatic_extraction(
            drug_name, cl_int_mL_per_min_per_g, fu_plasma, q_hepatic_L_per_h, model=m
        )
        for m in ("well_stirred", "parallel_tube", "dispersion")
    ]
    results.sort(key=lambda r: r.extraction_ratio, reverse=True)
    return results
