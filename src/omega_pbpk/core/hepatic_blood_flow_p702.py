"""Phase 702 — Hepatic Blood Flow Extraction model."""

from dataclasses import dataclass
from math import exp


@dataclass
class HepaticBloodFlowResult:
    drug_name: str
    cl_int_L_per_h: float
    fu_plasma: float
    qh_L_per_h: float
    cl_well_stirred_L_per_h: float
    cl_parallel_tube_L_per_h: float
    cl_dispersion_L_per_h: float
    er_well_stirred: float
    er_parallel_tube: float
    er_dispersion: float
    f_oral_well_stirred: float
    f_oral_parallel_tube: float
    f_oral_dispersion: float
    classification: str
    notes: str


def simulate_hepatic_blood_flow(
    drug_name: str,
    cl_int_L_per_h: float,
    fu_plasma: float,
    qh_L_per_h: float = 90.0,
    dispersion_number: float = 0.5,
) -> HepaticBloodFlowResult:
    if cl_int_L_per_h <= 0:
        raise ValueError("cl_int_L_per_h must be positive")
    if fu_plasma <= 0 or fu_plasma > 1.0:
        raise ValueError("fu_plasma must be in (0, 1]")
    if qh_L_per_h <= 0:
        raise ValueError("qh_L_per_h must be positive")
    if dispersion_number <= 0:
        raise ValueError("dispersion_number must be positive")

    Qh = qh_L_per_h
    fu = fu_plasma
    CLint = cl_int_L_per_h

    # Well-stirred model
    cl_ws = Qh * fu * CLint / (Qh + fu * CLint)
    er_ws = cl_ws / Qh

    # Parallel tube model
    er_pt = 1.0 - exp(-fu * CLint / Qh)
    cl_pt = Qh * er_pt

    # Dispersion model (weighted average approximation)
    er_disp = er_pt * 0.7 + er_ws * 0.3
    cl_disp = Qh * er_disp

    # Oral bioavailability (first-pass)
    f_oral_ws = 1.0 - er_ws
    f_oral_pt = 1.0 - er_pt
    f_oral_disp = 1.0 - er_disp

    # Classification based on well-stirred ER
    if er_ws < 0.3:
        classification = "low extraction"
    elif er_ws < 0.7:
        classification = "intermediate extraction"
    else:
        classification = "high extraction"

    return HepaticBloodFlowResult(
        drug_name=drug_name,
        cl_int_L_per_h=cl_int_L_per_h,
        fu_plasma=fu_plasma,
        qh_L_per_h=Qh,
        cl_well_stirred_L_per_h=cl_ws,
        cl_parallel_tube_L_per_h=cl_pt,
        cl_dispersion_L_per_h=cl_disp,
        er_well_stirred=er_ws,
        er_parallel_tube=er_pt,
        er_dispersion=er_disp,
        f_oral_well_stirred=f_oral_ws,
        f_oral_parallel_tube=f_oral_pt,
        f_oral_dispersion=f_oral_disp,
        classification=classification,
        notes="Well-stirred, parallel tube, and dispersion hepatic extraction models",
    )
