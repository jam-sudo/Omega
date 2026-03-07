"""
Hepatic blood flow-limited vs. enzyme-limited clearance model.
Implements the Rowland-Benet framework with three hepatic extraction models:
  - Well-stirred (venous equilibrium)
  - Parallel-tube (sinusoidal perfusion)
  - Dispersion (D_N model)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class HepaticClearanceResult:
    drug_name: str
    cl_int_L_per_h: float
    fup: float
    q_hepatic_L_per_h: float
    er_well_stirred: float  # extraction ratio (well-stirred model)
    er_parallel_tube: float  # parallel-tube model ER
    er_dispersion: float  # dispersion model ER (D_n=0.17)
    cl_hepatic_ws: float  # well-stirred hepatic CL
    cl_hepatic_pt: float  # parallel-tube hepatic CL
    cl_hepatic_disp: float  # dispersion model hepatic CL
    fh_well_stirred: float  # hepatic bioavailability (1-ER)
    fh_parallel_tube: float
    fh_dispersion: float
    clearance_model_comparison: str  # which model gives highest/lowest CL
    cl_sensitivity_to_flow: str  # "flow-limited", "enzyme-limited", "intermediate"
    notes: str


def calculate_hepatic_clearance(
    drug_name: str,
    cl_int_L_per_h: float,
    fup: float,
    q_hepatic_L_per_h: float = 97.0,
    dn: float = 0.17,
) -> HepaticClearanceResult:
    """
    Calculate hepatic clearance using three models.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    cl_int_L_per_h : float
        Intrinsic clearance (L/h). Must be > 0.
    fup : float
        Fraction unbound in plasma (0 < fup <= 1).
    q_hepatic_L_per_h : float
        Hepatic blood flow (L/h). Default 97.0 L/h (typical human liver).
    dn : float
        Dispersion number for the D_N model. Default 0.17.

    Returns
    -------
    HepaticClearanceResult
    """
    # --- Validation ---
    if cl_int_L_per_h <= 0:
        raise ValueError(f"cl_int_L_per_h must be > 0, got {cl_int_L_per_h}")
    if fup <= 0 or fup > 1:
        raise ValueError(f"fup must be in (0, 1], got {fup}")
    if q_hepatic_L_per_h <= 0:
        raise ValueError(f"q_hepatic_L_per_h must be > 0, got {q_hepatic_L_per_h}")
    if dn <= 0:
        raise ValueError(f"dn must be > 0, got {dn}")

    fu_clint = fup * cl_int_L_per_h
    qh = q_hepatic_L_per_h

    # --- Well-stirred model ---
    er_ws = fu_clint / (qh + fu_clint)
    cl_ws = qh * er_ws

    # --- Parallel-tube model ---
    er_pt = 1.0 - math.exp(-fu_clint / qh)
    cl_pt = qh * er_pt

    # --- Dispersion model (D_N model) ---
    # a = sqrt(1 + 4*dn*fup*CLint/Qh)
    a = math.sqrt(1.0 + 4.0 * dn * fu_clint / qh)
    # ER = 1 - 4a*exp(0.5/dn) / ((1+a)^2*exp(a/(2*dn)) - (1-a)^2*exp(-a/(2*dn)))
    numerator = 4.0 * a * math.exp(0.5 / dn)
    denom = (1.0 + a) ** 2 * math.exp(a / (2.0 * dn)) - (1.0 - a) ** 2 * math.exp(-a / (2.0 * dn))
    if abs(denom) < 1e-300:
        # Avoid division by zero in extreme cases — fallback to well-stirred
        er_disp = er_ws
    else:
        er_disp = 1.0 - numerator / denom
    # Clamp to [0, 1) for numerical safety
    er_disp = max(0.0, min(er_disp, 1.0 - 1e-12))
    cl_disp = qh * er_disp

    # --- Hepatic bioavailability (first-pass) ---
    fh_ws = 1.0 - er_ws
    fh_pt = 1.0 - er_pt
    fh_disp = 1.0 - er_disp

    # --- Sensitivity to blood flow ---
    if fu_clint > 3.0 * qh:
        sensitivity = "flow-limited"
    elif fu_clint < 0.1 * qh:
        sensitivity = "enzyme-limited"
    else:
        sensitivity = "intermediate"

    # --- Model comparison string ---
    models = {
        "well_stirred": cl_ws,
        "parallel_tube": cl_pt,
        "dispersion": cl_disp,
    }
    highest = max(models, key=lambda k: models[k])
    lowest = min(models, key=lambda k: models[k])
    if highest == lowest:
        comparison = "all models predict similar clearance"
    else:
        comparison = f"highest: {highest}, lowest: {lowest}"

    # --- Notes ---
    notes_parts = []
    if er_ws > 0.7:
        notes_parts.append("high-extraction drug (ER>0.7)")
    elif er_ws < 0.3:
        notes_parts.append("low-extraction drug (ER<0.3)")
    else:
        notes_parts.append("intermediate-extraction drug")
    notes_parts.append(f"sensitivity={sensitivity}")
    notes = "; ".join(notes_parts)

    return HepaticClearanceResult(
        drug_name=drug_name,
        cl_int_L_per_h=cl_int_L_per_h,
        fup=fup,
        q_hepatic_L_per_h=qh,
        er_well_stirred=er_ws,
        er_parallel_tube=er_pt,
        er_dispersion=er_disp,
        cl_hepatic_ws=cl_ws,
        cl_hepatic_pt=cl_pt,
        cl_hepatic_disp=cl_disp,
        fh_well_stirred=fh_ws,
        fh_parallel_tube=fh_pt,
        fh_dispersion=fh_disp,
        clearance_model_comparison=comparison,
        cl_sensitivity_to_flow=sensitivity,
        notes=notes,
    )
