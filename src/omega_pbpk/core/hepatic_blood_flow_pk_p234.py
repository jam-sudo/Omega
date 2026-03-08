"""
Phase 234: Hepatic Blood Flow PK (Well-Stirred Model)

Models hepatic first-pass extraction based on the well-stirred
(venous equilibrium) model.  Pure-Python, stdlib only (math, dataclasses).

NOTE: A more advanced three-model hepatic clearance module already exists at
      core/hepatic_blood_flow_pk.py; this module provides a simpler, focused
      well-stirred model with sensitivity analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HepaticExtractionResult:
    drug_name: str
    hepatic_blood_flow_L_per_h: float
    cl_int_L_per_h: float
    fu_b: float
    extraction_ratio: float
    cl_hepatic_L_per_h: float
    f_oral_hepatic: float
    cl_classification: str  # "low", "intermediate", or "high"
    notes: str


# ---------------------------------------------------------------------------
# Core calculation function
# ---------------------------------------------------------------------------


def calculate_hepatic_extraction(
    drug_name: str,
    cl_int_L_per_h: float,
    fu_b: float,
    hepatic_blood_flow_L_per_h: float = 90.0,
) -> HepaticExtractionResult:
    """
    Calculate hepatic first-pass extraction using the well-stirred model.

    Well-stirred model:
        E = fu_b * CLint / (Qh + fu_b * CLint)
        CLhepatic = Qh * E
        F_oral_hepatic = 1 - E

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    cl_int_L_per_h : float
        Intrinsic clearance (L/h). Must be > 0.
    fu_b : float
        Fraction unbound in blood (0 < fu_b <= 1).
    hepatic_blood_flow_L_per_h : float
        Hepatic blood flow (L/h). Default 90.0 L/h.

    Returns
    -------
    HepaticExtractionResult
    """
    # --- Validation ---
    if cl_int_L_per_h <= 0:
        raise ValueError(f"cl_int_L_per_h must be > 0, got {cl_int_L_per_h}")
    if fu_b <= 0 or fu_b > 1.0:
        raise ValueError(f"fu_b must be in (0, 1], got {fu_b}")
    if hepatic_blood_flow_L_per_h <= 0:
        raise ValueError(
            f"hepatic_blood_flow_L_per_h must be > 0, got {hepatic_blood_flow_L_per_h}"
        )

    qh = hepatic_blood_flow_L_per_h
    fu_clint = fu_b * cl_int_L_per_h

    # Well-stirred model extraction ratio
    extraction_ratio = fu_clint / (qh + fu_clint)
    cl_hepatic = qh * extraction_ratio
    f_oral_hepatic = 1.0 - extraction_ratio

    # Classification
    if extraction_ratio < 0.3:
        cl_classification = "low"
    elif extraction_ratio <= 0.7:
        cl_classification = "intermediate"
    else:
        cl_classification = "high"

    notes = (
        f"Well-stirred model: E={extraction_ratio:.3f} ({cl_classification} extraction). "
        f"Qh={qh:.1f} L/h, fu_b*CLint={fu_clint:.2f} L/h."
    )

    return HepaticExtractionResult(
        drug_name=drug_name,
        hepatic_blood_flow_L_per_h=qh,
        cl_int_L_per_h=cl_int_L_per_h,
        fu_b=fu_b,
        extraction_ratio=extraction_ratio,
        cl_hepatic_L_per_h=cl_hepatic,
        f_oral_hepatic=f_oral_hepatic,
        cl_classification=cl_classification,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

_DEFAULT_FLOW_RANGE = [30.0, 60.0, 90.0, 120.0, 150.0]


def sensitivity_to_blood_flow(
    drug_name: str,
    cl_int_L_per_h: float,
    fu_b: float,
    flow_range_L_per_h: list[float] | None = None,
) -> list[HepaticExtractionResult]:
    """
    Evaluate hepatic extraction across a range of hepatic blood flow values.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    cl_int_L_per_h : float
        Intrinsic clearance (L/h). Must be > 0.
    fu_b : float
        Fraction unbound in blood (0 < fu_b <= 1).
    flow_range_L_per_h : list[float] | None
        Blood flow values to evaluate. Defaults to
        [30.0, 60.0, 90.0, 120.0, 150.0].

    Returns
    -------
    list[HepaticExtractionResult]
        Results sorted by extraction_ratio ascending (lowest ER first).
    """
    if flow_range_L_per_h is None:
        flow_range_L_per_h = _DEFAULT_FLOW_RANGE

    results = []
    for qh in flow_range_L_per_h:
        r = calculate_hepatic_extraction(
            drug_name=drug_name,
            cl_int_L_per_h=cl_int_L_per_h,
            fu_b=fu_b,
            hepatic_blood_flow_L_per_h=qh,
        )
        results.append(r)

    results.sort(key=lambda r: r.extraction_ratio)
    return results
