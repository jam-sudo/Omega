"""Phase 190 — Renal dose adjustment for renally-cleared drugs based on eGFR.

Implements eGFR-based CL scaling with FDA guidance flags for dose adjustment.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "RenalDoseResult",
    "adjust_renal_dose",
    "screen_renal_impairment",
]

_NORMAL_EGFR = 90.0  # mL/min reference (normal)


def _egfr_stage(egfr_mL_min: float) -> str:
    """Return KDIGO eGFR stage string."""
    if egfr_mL_min >= 90.0:
        return "G1"
    if egfr_mL_min >= 60.0:
        return "G2"
    if egfr_mL_min >= 45.0:
        return "G3a"
    if egfr_mL_min >= 30.0:
        return "G3b"
    if egfr_mL_min >= 15.0:
        return "G4"
    return "G5"


@dataclass(frozen=True)
class RenalDoseResult:
    """Result of renal dose adjustment calculation."""

    drug_name: str
    egfr_mL_min: float
    egfr_stage: str
    fe_renal: float
    normal_dose_mg: float
    adjusted_dose_mg: float
    adjusted_interval_h: float
    cl_ratio: float
    dose_adjustment_pct: float
    dose_reduction_recommended: bool
    monitoring_required: bool
    notes: str


def adjust_renal_dose(
    drug_name: str,
    normal_dose_mg: float,
    fe_renal: float,
    egfr_mL_min: float,
    dosing_interval_h: float = 24.0,
) -> RenalDoseResult:
    """Calculate dose adjustment for renal impairment.

    eGFR-based CL scaling:
        CL_patient = CL_normal * (fe_renal * egfr / 90 + (1 - fe_renal))
    Adjusted dose (dose reduction method):
        D_adj = D_normal * (CL_patient / CL_normal)
    Extended interval method:
        tau_adj = tau_normal * (CL_normal / CL_patient)

    FDA guidance: dose_reduction_recommended if fe_renal > 0.5 and eGFR < 60.

    Args:
        drug_name: Name of the drug.
        normal_dose_mg: Normal dose in mg. Must be > 0.
        fe_renal: Fraction eliminated renally (0 to 1).
        egfr_mL_min: Patient eGFR in mL/min. Must be >= 0.
        dosing_interval_h: Normal dosing interval in hours (default 24).

    Returns:
        RenalDoseResult dataclass.

    Raises:
        ValueError: On invalid inputs.
    """
    if normal_dose_mg <= 0:
        raise ValueError(f"normal_dose_mg must be > 0, got {normal_dose_mg}")
    if not (0.0 <= fe_renal <= 1.0):
        raise ValueError(f"fe_renal must be in [0, 1], got {fe_renal}")
    if egfr_mL_min < 0:
        raise ValueError(f"egfr_mL_min must be >= 0, got {egfr_mL_min}")

    stage = _egfr_stage(egfr_mL_min)

    # CL ratio: patient CL relative to normal CL
    # CL_patient / CL_normal = fe_renal * (egfr / 90) + (1 - fe_renal)
    egfr_ratio = egfr_mL_min / _NORMAL_EGFR
    cl_ratio = fe_renal * egfr_ratio + (1.0 - fe_renal)

    # Adjusted dose (dose reduction)
    adjusted_dose_mg = normal_dose_mg * cl_ratio

    # Adjusted interval (interval extension)
    if cl_ratio > 0:
        adjusted_interval_h = dosing_interval_h / cl_ratio
    else:
        adjusted_interval_h = dosing_interval_h * 100.0  # extreme extension

    # Dose adjustment percentage (reduction from normal)
    dose_adjustment_pct = (1.0 - cl_ratio) * 100.0

    # FDA guidance flag
    dose_reduction_recommended = fe_renal > 0.5 and egfr_mL_min < 60.0

    # Monitoring required: G3b or worse, or dose reduction recommended
    monitoring_required = egfr_mL_min < 45.0 or dose_reduction_recommended

    notes_parts = [
        f"eGFR={egfr_mL_min:.1f} mL/min (stage {stage})",
        f"fe_renal={fe_renal:.2f}",
        f"cl_ratio={cl_ratio:.3f}",
        f"adjusted_dose={adjusted_dose_mg:.2f} mg",
        f"adjusted_interval={adjusted_interval_h:.1f} h",
    ]
    if dose_reduction_recommended:
        notes_parts.append("FDA: dose reduction recommended")
    if monitoring_required:
        notes_parts.append("enhanced monitoring required")
    notes = "; ".join(notes_parts)

    return RenalDoseResult(
        drug_name=drug_name,
        egfr_mL_min=egfr_mL_min,
        egfr_stage=stage,
        fe_renal=fe_renal,
        normal_dose_mg=normal_dose_mg,
        adjusted_dose_mg=adjusted_dose_mg,
        adjusted_interval_h=adjusted_interval_h,
        cl_ratio=cl_ratio,
        dose_adjustment_pct=dose_adjustment_pct,
        dose_reduction_recommended=dose_reduction_recommended,
        monitoring_required=monitoring_required,
        notes=notes,
    )


def screen_renal_impairment(
    drug_name: str,
    normal_dose_mg: float,
    fe_renal: float,
    dosing_interval_h: float,
    egfr_values: list[float],
) -> list[RenalDoseResult]:
    """Calculate dose adjustments for a list of eGFR values.

    Args:
        drug_name: Name of the drug.
        normal_dose_mg: Normal dose in mg.
        fe_renal: Fraction eliminated renally (0 to 1).
        dosing_interval_h: Normal dosing interval in hours.
        egfr_values: List of eGFR values in mL/min to screen.

    Returns:
        List of RenalDoseResult, one for each eGFR value (preserving input order).
    """
    results: list[RenalDoseResult] = []
    for egfr in egfr_values:
        result = adjust_renal_dose(
            drug_name=drug_name,
            normal_dose_mg=normal_dose_mg,
            fe_renal=fe_renal,
            egfr_mL_min=egfr,
            dosing_interval_h=dosing_interval_h,
        )
        results.append(result)
    return results
