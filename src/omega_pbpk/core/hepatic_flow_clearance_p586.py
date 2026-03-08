"""Phase 586 - Hepatic Blood Flow Effect on Drug Clearance.

Simple sensitivity analysis of how hepatic blood flow changes affect drug
clearance for high/low extraction drugs using the well-stirred model.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HepaticFlowClearanceResult:
    """Result of hepatic flow-clearance sensitivity analysis."""

    drug_name: str
    q_hepatic_L_per_h: float
    cl_intrinsic_L_per_h: float
    fup: float
    cl_hepatic_L_per_h: float
    extraction_ratio: float
    flow_sensitivity_index: float
    cl_class: str
    dose_dependency_class: str
    notes: str


def predict_hepatic_flow_cl(
    drug_name: str,
    q_hepatic_L_per_h: float,
    cl_intrinsic_L_per_h: float,
    fup: float,
) -> HepaticFlowClearanceResult:
    """Predict hepatic clearance and flow sensitivity.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    q_hepatic_L_per_h : float
        Hepatic blood flow in L/h (must be > 0).
    cl_intrinsic_L_per_h : float
        Intrinsic clearance in L/h (must be > 0).
    fup : float
        Fraction unbound in plasma, in (0, 1].

    Returns
    -------
    HepaticFlowClearanceResult
    """
    if q_hepatic_L_per_h <= 0:
        raise ValueError("q_hepatic_L_per_h must be > 0")
    if cl_intrinsic_L_per_h <= 0:
        raise ValueError("cl_intrinsic_L_per_h must be > 0")
    if fup <= 0 or fup > 1:
        raise ValueError("fup must be in (0, 1]")

    q = q_hepatic_L_per_h
    fu_clint = fup * cl_intrinsic_L_per_h

    # Well-stirred model
    cl_hepatic = q * fu_clint / (q + fu_clint)

    # Extraction ratio
    extraction_ratio = cl_hepatic / q
    extraction_ratio = max(0.0, min(1.0, extraction_ratio))

    # Flow sensitivity index = E
    flow_sensitivity_index = extraction_ratio

    # Classification
    if extraction_ratio > 0.7:
        cl_class = "flow_limited"
    elif extraction_ratio < 0.3:
        cl_class = "capacity_limited"
    else:
        cl_class = "intermediate"

    if extraction_ratio > 0.7:
        dose_dependency_class = "highly_variable_with_flow"
    elif extraction_ratio > 0.3:
        dose_dependency_class = "moderately_variable"
    else:
        dose_dependency_class = "flow_independent"

    notes = (
        f"Q={q:.1f} L/h, fu*CLint={fu_clint:.2f} L/h, "
        f"E={extraction_ratio:.4f}, CL_h={cl_hepatic:.4f} L/h"
    )

    return HepaticFlowClearanceResult(
        drug_name=drug_name,
        q_hepatic_L_per_h=q,
        cl_intrinsic_L_per_h=cl_intrinsic_L_per_h,
        fup=fup,
        cl_hepatic_L_per_h=cl_hepatic,
        extraction_ratio=extraction_ratio,
        flow_sensitivity_index=flow_sensitivity_index,
        cl_class=cl_class,
        dose_dependency_class=dose_dependency_class,
        notes=notes,
    )


def sweep_flow_effects(
    drug_name: str,
    cl_intrinsic_L_per_h: float,
    fup: float,
    q_values: list[float] | None = None,
) -> list[HepaticFlowClearanceResult]:
    """Sweep hepatic blood flow values and return results sorted ascending by Q.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    cl_intrinsic_L_per_h : float
        Intrinsic clearance in L/h.
    fup : float
        Fraction unbound in plasma.
    q_values : list[float] or None
        Flow values to sweep. Defaults to [30, 60, 90, 120, 150, 180].

    Returns
    -------
    list[HepaticFlowClearanceResult]
        Results sorted by q_hepatic_L_per_h ascending.
    """
    if q_values is None:
        q_values = [30.0, 60.0, 90.0, 120.0, 150.0, 180.0]

    results = []
    for q in q_values:
        r = predict_hepatic_flow_cl(
            drug_name=drug_name,
            q_hepatic_L_per_h=q,
            cl_intrinsic_L_per_h=cl_intrinsic_L_per_h,
            fup=fup,
        )
        results.append(r)
    results.sort(key=lambda r: r.q_hepatic_L_per_h)
    return results
