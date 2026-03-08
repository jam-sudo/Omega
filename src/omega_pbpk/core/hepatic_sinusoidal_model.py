"""
Phase 511 — Hepatic Sinusoidal Model (Well-Stirred vs Parallel-Tube)

Compare well-stirred and parallel-tube hepatic extraction models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class HepaticSinusoidalResult:
    """Result of hepatic sinusoidal model comparison."""

    drug_name: str
    fup: float
    cl_int_L_per_h: float
    q_hepatic_L_per_h: float

    eh_well_stirred: float
    fh_well_stirred: float
    clh_well_stirred_L_per_h: float

    eh_parallel_tube: float
    fh_parallel_tube: float
    clh_parallel_tube_L_per_h: float

    model_discrepancy_pct: float
    dominant_model: str
    notes: str


def compare_hepatic_models(
    drug_name: str,
    fup: float,
    cl_int_L_per_h: float,
    q_hepatic_L_per_h: float = 90.0,
) -> HepaticSinusoidalResult:
    """
    Compare well-stirred and parallel-tube hepatic extraction models.

    Well-stirred:    EH = (fup * CLint) / (Q + fup * CLint)
    Parallel-tube:   EH = 1 - exp(-fup * CLint / Q)

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    fup : float
        Fraction unbound in plasma (0 < fup <= 1).
    cl_int_L_per_h : float
        Intrinsic clearance (L/h), must be > 0.
    q_hepatic_L_per_h : float
        Hepatic blood flow (L/h), default 90 L/h.

    Returns
    -------
    HepaticSinusoidalResult
    """
    if not (0 < fup <= 1):
        raise ValueError(f"fup must be in (0, 1], got {fup}")
    if cl_int_L_per_h <= 0:
        raise ValueError(f"cl_int_L_per_h must be > 0, got {cl_int_L_per_h}")
    if q_hepatic_L_per_h <= 0:
        raise ValueError(f"q_hepatic_L_per_h must be > 0, got {q_hepatic_L_per_h}")

    Q = q_hepatic_L_per_h
    fup_cl = fup * cl_int_L_per_h

    # Well-stirred model
    eh_ws = fup_cl / (Q + fup_cl)
    fh_ws = 1.0 - eh_ws
    clh_ws = Q * eh_ws

    # Parallel-tube model
    eh_pt = 1.0 - math.exp(-fup_cl / Q)
    fh_pt = 1.0 - eh_pt
    clh_pt = Q * eh_pt

    discrepancy_pct = abs(eh_ws - eh_pt) * 100.0

    if discrepancy_pct < 5.0:
        dominant_model = "equivalent"
    elif eh_ws < eh_pt:
        dominant_model = "well_stirred"
    else:
        dominant_model = "parallel_tube"

    notes = (
        f"Well-stirred EH={eh_ws:.4f}, Parallel-tube EH={eh_pt:.4f}; "
        f"discrepancy={discrepancy_pct:.2f}%"
    )

    return HepaticSinusoidalResult(
        drug_name=drug_name,
        fup=fup,
        cl_int_L_per_h=cl_int_L_per_h,
        q_hepatic_L_per_h=q_hepatic_L_per_h,
        eh_well_stirred=eh_ws,
        fh_well_stirred=fh_ws,
        clh_well_stirred_L_per_h=clh_ws,
        eh_parallel_tube=eh_pt,
        fh_parallel_tube=fh_pt,
        clh_parallel_tube_L_per_h=clh_pt,
        model_discrepancy_pct=discrepancy_pct,
        dominant_model=dominant_model,
        notes=notes,
    )


def hepatic_clearance_sensitivity(
    drug_name: str,
    fup: float,
    q_hepatic_L_per_h: float,
    cl_int_values: list,
) -> list:
    """
    Run hepatic model comparison across a range of CLint values.

    Parameters
    ----------
    drug_name : str
        Drug name.
    fup : float
        Fraction unbound in plasma.
    q_hepatic_L_per_h : float
        Hepatic blood flow (L/h).
    cl_int_values : list
        List of intrinsic clearance values (L/h) to evaluate.

    Returns
    -------
    list of HepaticSinusoidalResult, sorted by eh_well_stirred ascending.
    """
    results = [
        compare_hepatic_models(
            drug_name=drug_name,
            fup=fup,
            cl_int_L_per_h=cl_int,
            q_hepatic_L_per_h=q_hepatic_L_per_h,
        )
        for cl_int in cl_int_values
    ]
    return sorted(results, key=lambda r: r.eh_well_stirred)
