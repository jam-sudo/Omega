"""Phase 580 - Hepatic Clearance Scaling from Microsomes.

Scale hepatic clearance from in vitro microsomal intrinsic clearance
to in vivo human hepatic clearance using well-stirred and parallel-tube models.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MicrosomeCLScalingResult:
    """Result of microsomal clearance scaling."""

    drug_name: str
    clint_microsomal_mL_per_min_per_mg_protein: float
    cl_intrinsic_liver_mL_per_min: float
    cl_hepatic_L_per_h: float
    cl_hepatic_pt_L_per_h: float
    extraction_ratio_ws: float
    extraction_ratio_pt: float
    f_oral_ws: float
    f_oral_pt: float
    model_discrepancy_pct: float
    scaling_class: str
    notes: str


Q_H = 90.0  # L/h hepatic blood flow


def scale_microsomal_cl(
    drug_name: str,
    clint_microsomal_mL_per_min_per_mg_protein: float,
    fup: float = 0.1,
    scaling_factor: float = 1.0,
    liver_weight_g: float = 1500.0,
    microsomal_protein_content_mg_per_g: float = 45.0,
) -> MicrosomeCLScalingResult:
    """Scale in vitro microsomal CLint to in vivo hepatic clearance."""
    if clint_microsomal_mL_per_min_per_mg_protein <= 0:
        raise ValueError("clint_microsomal must be > 0")
    if not (0.0 < fup <= 1.0):
        raise ValueError("fup must be in (0, 1]")
    if liver_weight_g <= 0:
        raise ValueError("liver_weight_g must be > 0")

    clint_liver = (
        clint_microsomal_mL_per_min_per_mg_protein
        * microsomal_protein_content_mg_per_g
        * liver_weight_g
        * scaling_factor
    )  # mL/min

    clint_L_per_h = clint_liver * 60.0 / 1000.0

    # Well-stirred model
    fu_clint = fup * clint_L_per_h
    cl_ws = Q_H * fu_clint / (Q_H + fu_clint)
    e_ws = cl_ws / Q_H

    # Parallel-tube model
    e_pt = 1.0 - math.exp(-fu_clint / Q_H)
    cl_pt = Q_H * e_pt

    f_oral_ws = 1.0 - e_ws
    f_oral_pt = 1.0 - e_pt

    if cl_ws > 0:
        model_discrepancy = abs(cl_ws - cl_pt) / cl_ws * 100.0
    else:
        model_discrepancy = 0.0

    if e_ws < 0.3:
        scaling_class = "low_e"
    elif e_ws <= 0.7:
        scaling_class = "intermediate_e"
    else:
        scaling_class = "high_e"

    notes = f"CLint_liver {clint_liver:.2f} mL/min; CLint {clint_L_per_h:.2f} L/h; Q_h {Q_H} L/h"

    return MicrosomeCLScalingResult(
        drug_name=drug_name,
        clint_microsomal_mL_per_min_per_mg_protein=clint_microsomal_mL_per_min_per_mg_protein,
        cl_intrinsic_liver_mL_per_min=clint_liver,
        cl_hepatic_L_per_h=cl_ws,
        cl_hepatic_pt_L_per_h=cl_pt,
        extraction_ratio_ws=e_ws,
        extraction_ratio_pt=e_pt,
        f_oral_ws=f_oral_ws,
        f_oral_pt=f_oral_pt,
        model_discrepancy_pct=model_discrepancy,
        scaling_class=scaling_class,
        notes=notes,
    )


def screen_microsomal_cl(compounds: list) -> list:
    """Screen multiple compounds, sorted by cl_hepatic_L_per_h descending."""
    results = []
    for comp in compounds:
        r = scale_microsomal_cl(
            drug_name=comp["drug_name"],
            clint_microsomal_mL_per_min_per_mg_protein=comp["clint_microsomal"],
            fup=comp.get("fup", 0.1),
        )
        results.append(r)
    results.sort(key=lambda x: x.cl_hepatic_L_per_h, reverse=True)
    return results
