"""Systematic dose adjustment for patients with renal impairment (CKD stages 1-5).

Uses the Giusti-Hayton method to adjust clearance based on GFR, then derives
dose or interval adjustments to maintain target Css_avg.
"""

from __future__ import annotations

__all__ = [
    "CKDStage",
    "RenalImpairmentDoseResult",
    "adjust_dose_for_ckd",
    "ckd_dose_table",
]

from dataclasses import dataclass
from enum import Enum


class CKDStage(Enum):
    STAGE_1 = 1  # GFR >= 90 mL/min/1.73m²
    STAGE_2 = 2  # GFR 60-89
    STAGE_3A = 3  # GFR 45-59
    STAGE_3B = 4  # GFR 30-44
    STAGE_4 = 5  # GFR 15-29
    STAGE_5 = 6  # GFR < 15 (ESRD)


@dataclass(frozen=True)
class RenalImpairmentDoseResult:
    drug_name: str
    gfr_mL_per_min: float
    ckd_stage: str
    fe_renal: float
    cl_normal_L_per_h: float
    cl_adjusted_L_per_h: float
    dose_normal_mg: float
    dose_adjusted_mg: float
    interval_normal_h: float
    interval_adjusted_h: float
    dose_reduction_pct: float
    t_half_normal_h: float
    t_half_adjusted_h: float
    accumulation_risk: str  # "low", "moderate", "high"
    monitoring: str
    notes: str


def _gfr_to_ckd_stage(gfr: float) -> str:
    if gfr >= 90.0:
        return "Stage 1"
    elif gfr >= 60.0:
        return "Stage 2"
    elif gfr >= 45.0:
        return "Stage 3A"
    elif gfr >= 30.0:
        return "Stage 3B"
    elif gfr >= 15.0:
        return "Stage 4"
    else:
        return "Stage 5 (ESRD)"


def _monitoring(gfr: float) -> str:
    if gfr < 30.0:
        return "TDM recommended; close renal function monitoring"
    elif gfr < 60.0:
        return "Monitor renal function and drug levels"
    else:
        return "Routine monitoring"


def adjust_dose_for_ckd(
    drug_name: str,
    gfr_mL_per_min: float,
    fe_renal: float,
    cl_normal_L_per_h: float,
    vd_L: float,
    dose_normal_mg: float,
    interval_normal_h: float = 24.0,
    dose_strategy: str = "dose_reduction",
) -> RenalImpairmentDoseResult:
    """Adjust dose for CKD using the Giusti-Hayton method.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    gfr_mL_per_min:
        Patient GFR in mL/min (>= 0).
    fe_renal:
        Fraction eliminated renally (0-1).
    cl_normal_L_per_h:
        Normal (healthy) total clearance in L/h (> 0).
    vd_L:
        Volume of distribution in L (> 0).
    dose_normal_mg:
        Usual dose in mg (> 0).
    interval_normal_h:
        Usual dosing interval in hours.
    dose_strategy:
        "dose_reduction" keeps interval fixed and reduces dose;
        "interval_extension" keeps dose fixed and extends interval.

    Returns
    -------
    RenalImpairmentDoseResult
    """
    if gfr_mL_per_min < 0.0:
        raise ValueError(f"gfr_mL_per_min must be >= 0, got {gfr_mL_per_min}")
    if not (0.0 <= fe_renal <= 1.0):
        raise ValueError(f"fe_renal must be in [0, 1], got {fe_renal}")
    if cl_normal_L_per_h <= 0.0:
        raise ValueError(f"cl_normal_L_per_h must be > 0, got {cl_normal_L_per_h}")
    if vd_L <= 0.0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if dose_normal_mg <= 0.0:
        raise ValueError(f"dose_normal_mg must be > 0, got {dose_normal_mg}")

    # Giusti-Hayton: CL in CKD
    cl_ckd = cl_normal_L_per_h * (1.0 - fe_renal * (1.0 - gfr_mL_per_min / 120.0))
    cl_ckd = max(cl_ckd, cl_normal_L_per_h * 0.05)

    dose_adj_factor = cl_ckd / cl_normal_L_per_h

    if dose_strategy == "dose_reduction":
        dose_adj = dose_normal_mg * dose_adj_factor
        interval_adj = interval_normal_h
    else:  # interval_extension
        dose_adj = dose_normal_mg
        interval_adj = (
            interval_normal_h / dose_adj_factor if dose_adj_factor > 0.0 else float("inf")
        )

    ckd_stage = _gfr_to_ckd_stage(gfr_mL_per_min)

    ke_normal = cl_normal_L_per_h / vd_L
    t_half_normal = 0.693 / ke_normal

    ke_ckd = cl_ckd / vd_L
    t_half_ckd = 0.693 / ke_ckd if ke_ckd > 0.0 else float("inf")

    dose_reduction_pct = (1.0 - dose_adj_factor) * 100.0

    # Accumulation risk based on adjusted half-life vs adjusted interval
    if t_half_ckd > 3.0 * interval_adj:
        accumulation_risk = "high"
    elif t_half_ckd > interval_adj:
        accumulation_risk = "moderate"
    else:
        accumulation_risk = "low"

    monitoring = _monitoring(gfr_mL_per_min)

    notes = (
        f"Giusti-Hayton method; fe_renal={fe_renal:.2f}; "
        f"CL adjusted from {cl_normal_L_per_h:.2f} to {cl_ckd:.2f} L/h"
    )

    return RenalImpairmentDoseResult(
        drug_name=drug_name,
        gfr_mL_per_min=gfr_mL_per_min,
        ckd_stage=ckd_stage,
        fe_renal=fe_renal,
        cl_normal_L_per_h=cl_normal_L_per_h,
        cl_adjusted_L_per_h=cl_ckd,
        dose_normal_mg=dose_normal_mg,
        dose_adjusted_mg=dose_adj,
        interval_normal_h=interval_normal_h,
        interval_adjusted_h=interval_adj,
        dose_reduction_pct=dose_reduction_pct,
        t_half_normal_h=t_half_normal,
        t_half_adjusted_h=t_half_ckd,
        accumulation_risk=accumulation_risk,
        monitoring=monitoring,
        notes=notes,
    )


# Representative GFR values for each CKD stage
_STAGE_GFR_VALUES = [95.0, 75.0, 52.0, 37.0, 22.0, 10.0]


def ckd_dose_table(
    drug_name: str,
    fe_renal: float,
    cl_normal_L_per_h: float,
    vd_L: float,
    dose_normal_mg: float,
    **kwargs,
) -> list[RenalImpairmentDoseResult]:
    """Return one RenalImpairmentDoseResult per CKD stage (6 entries).

    GFR representative values used: [95, 75, 52, 37, 22, 10] mL/min.

    Extra keyword arguments are forwarded to :func:`adjust_dose_for_ckd`.
    """
    return [
        adjust_dose_for_ckd(
            drug_name=drug_name,
            gfr_mL_per_min=gfr,
            fe_renal=fe_renal,
            cl_normal_L_per_h=cl_normal_L_per_h,
            vd_L=vd_L,
            dose_normal_mg=dose_normal_mg,
            **kwargs,
        )
        for gfr in _STAGE_GFR_VALUES
    ]
