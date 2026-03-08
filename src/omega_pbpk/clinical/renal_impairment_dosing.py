"""Phase 385: Renal Impairment Dosing Calculator.

Calculate dose adjustments for renally-impaired patients based on CKD stage
and fraction excreted unchanged (fe).

References:
    KDIGO 2012 CKD Classification
    FDA Guidance for Industry: Pharmacokinetics in Patients with Impaired
    Renal Function (2010)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenalDosingResult:
    """Result of renal impairment dose calculation."""

    drug_name: str
    egfr_mL_per_min: float
    ckd_stage: str  # "G1"(>=90), "G2"(60-89), "G3a"(45-59), "G3b"(30-44), "G4"(15-29), "G5"(<15)
    fe_unchanged: float  # fraction excreted unchanged (0-1)
    baseline_dose_mg: float
    adjusted_dose_mg: float
    dose_reduction_pct: float  # positive means dose reduction
    dosing_interval_adjustment: str  # "standard", "extend_interval", "reduce_dose", "avoid"
    auc_ratio_impaired_to_normal: float  # predicted AUC change
    monitoring_required: bool
    dialysis_supplement_needed: bool
    clinical_guidance: str
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _classify_ckd_stage(egfr: float) -> str:
    """Classify CKD stage per KDIGO 2012."""
    if egfr >= 90.0:
        return "G1"
    elif egfr >= 60.0:
        return "G2"
    elif egfr >= 45.0:
        return "G3a"
    elif egfr >= 30.0:
        return "G3b"
    elif egfr >= 15.0:
        return "G4"
    else:
        return "G5"


def _dosing_interval_adjustment(total_cl_fraction: float, fe_unchanged: float) -> str:
    """Determine dosing interval adjustment strategy."""
    if total_cl_fraction > 0.75:
        return "standard"
    elif total_cl_fraction > 0.5:
        return "reduce_dose"
    elif total_cl_fraction > 0.25:
        return "extend_interval"
    else:
        if fe_unchanged > 0.5:
            return "avoid"
        return "reduce_dose"


def _clinical_guidance(ckd_stage: str, fe_unchanged: float) -> str:
    """Generate clinical guidance string based on CKD stage and fe."""
    if ckd_stage in ("G1", "G2") or fe_unchanged < 0.2:
        return "No dose adjustment required"
    elif ckd_stage in ("G3a", "G3b") and fe_unchanged > 0.3:
        return "Moderate dose reduction recommended"
    elif ckd_stage in ("G4", "G5"):
        return "Significant dose reduction; consider alternative drug"
    else:
        return "Monitor closely; dose adjustment may be warranted"


def _build_notes(
    ckd_stage: str,
    total_cl_fraction: float,
    renal_cl_fraction: float,
    dosing_interval: str,
    monitoring_required: bool,
    dialysis_supplement_needed: bool,
) -> str:
    """Build notes string summarising the calculation."""
    parts = [
        f"CKD Stage {ckd_stage}",
        f"Renal CL fraction: {renal_cl_fraction:.2f}",
        f"Total CL fraction: {total_cl_fraction:.3f}",
        f"Strategy: {dosing_interval}",
    ]
    if monitoring_required:
        parts.append("Renal function monitoring required")
    if dialysis_supplement_needed:
        parts.append("Supplemental dose after dialysis may be required")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calculate_renal_dose(
    drug_name: str,
    baseline_dose_mg: float,
    egfr_mL_per_min: float,
    fe_unchanged: float,
    normal_egfr: float = 100.0,
) -> RenalDosingResult:
    """Calculate dose adjustment for a renally-impaired patient.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    baseline_dose_mg:
        Standard dose in a patient with normal renal function (mg).
    egfr_mL_per_min:
        Patient's estimated GFR (mL/min).
    fe_unchanged:
        Fraction of dose excreted unchanged in urine (0-1).
    normal_egfr:
        Reference eGFR for a patient with normal renal function
        (default 100 mL/min).

    Returns
    -------
    RenalDosingResult
    """
    # Validation
    if egfr_mL_per_min < 0:
        raise ValueError(f"egfr_mL_per_min must be >= 0, got {egfr_mL_per_min}")
    if not (0.0 <= fe_unchanged <= 1.0):
        raise ValueError(f"fe_unchanged must be in [0, 1], got {fe_unchanged}")
    if baseline_dose_mg <= 0:
        raise ValueError(f"baseline_dose_mg must be > 0, got {baseline_dose_mg}")
    if normal_egfr <= 0:
        raise ValueError(f"normal_egfr must be > 0, got {normal_egfr}")

    # CKD classification
    ckd_stage = _classify_ckd_stage(egfr_mL_per_min)

    # Dose adjustment model
    renal_cl_fraction = min(egfr_mL_per_min / normal_egfr, 1.0)
    # Non-renal CL is unchanged; renal CL scales with GFR
    total_cl_fraction = (1.0 - fe_unchanged) + fe_unchanged * renal_cl_fraction

    # Adjusted dose (to maintain same AUC as a normal patient)
    adjusted_dose_mg = baseline_dose_mg * total_cl_fraction
    dose_reduction_pct = (1.0 - total_cl_fraction) * 100.0

    # AUC ratio if dose NOT adjusted (reflects drug accumulation risk)
    if total_cl_fraction > 0.0:
        auc_ratio = 1.0 / total_cl_fraction
    else:
        auc_ratio = float("inf")

    # Dosing strategy
    dosing_interval = _dosing_interval_adjustment(total_cl_fraction, fe_unchanged)

    # Safety flags
    monitoring_required = (egfr_mL_per_min < 60.0 and fe_unchanged > 0.3) or ckd_stage in (
        "G4",
        "G5",
    )
    dialysis_supplement_needed = egfr_mL_per_min < 15.0 and fe_unchanged > 0.3

    # Guidance
    guidance = _clinical_guidance(ckd_stage, fe_unchanged)

    # Notes
    notes = _build_notes(
        ckd_stage,
        total_cl_fraction,
        renal_cl_fraction,
        dosing_interval,
        monitoring_required,
        dialysis_supplement_needed,
    )

    return RenalDosingResult(
        drug_name=drug_name,
        egfr_mL_per_min=egfr_mL_per_min,
        ckd_stage=ckd_stage,
        fe_unchanged=fe_unchanged,
        baseline_dose_mg=baseline_dose_mg,
        adjusted_dose_mg=adjusted_dose_mg,
        dose_reduction_pct=dose_reduction_pct,
        dosing_interval_adjustment=dosing_interval,
        auc_ratio_impaired_to_normal=auc_ratio,
        monitoring_required=monitoring_required,
        dialysis_supplement_needed=dialysis_supplement_needed,
        clinical_guidance=guidance,
        notes=notes,
    )


def screen_ckd_stages(
    drug_name: str,
    baseline_dose_mg: float,
    fe_unchanged: float,
) -> list[RenalDosingResult]:
    """Screen dose adjustments across representative CKD eGFR values.

    Screens eGFRs: [100, 75, 52, 37, 22, 10] mL/min (covering G1-G5).

    Parameters
    ----------
    drug_name:
        Name of the drug.
    baseline_dose_mg:
        Standard dose for a patient with normal renal function (mg).
    fe_unchanged:
        Fraction of dose excreted unchanged (0-1).

    Returns
    -------
    list[RenalDosingResult]
        Sorted by eGFR descending (G1 first, G5 last).
    """
    representative_egfrs = [100.0, 75.0, 52.0, 37.0, 22.0, 10.0]
    results = [
        calculate_renal_dose(drug_name, baseline_dose_mg, egfr, fe_unchanged)
        for egfr in representative_egfrs
    ]
    results.sort(key=lambda r: r.egfr_mL_per_min, reverse=True)
    return results


__all__ = [
    "RenalDosingResult",
    "calculate_renal_dose",
    "screen_ckd_stages",
    # Legacy Phase 205 API (kept for backward compatibility)
    "CKDStage",
    "RenalImpairmentDoseResult",
    "adjust_dose_for_ckd",
    "ckd_dose_table",
]


# ===========================================================================
# Legacy Phase 205 API — Giusti-Hayton method
# ===========================================================================


class CKDStage(Enum):
    STAGE_1 = 1  # GFR >= 90 mL/min/1.73m2
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
    """Adjust dose for CKD using the Giusti-Hayton method."""
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

    cl_ckd = cl_normal_L_per_h * (1.0 - fe_renal * (1.0 - gfr_mL_per_min / 120.0))
    cl_ckd = max(cl_ckd, cl_normal_L_per_h * 0.05)

    dose_adj_factor = cl_ckd / cl_normal_L_per_h

    if dose_strategy == "dose_reduction":
        dose_adj = dose_normal_mg * dose_adj_factor
        interval_adj = interval_normal_h
    else:
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


_STAGE_GFR_VALUES = [95.0, 75.0, 52.0, 37.0, 22.0, 10.0]


def ckd_dose_table(
    drug_name: str,
    fe_renal: float,
    cl_normal_L_per_h: float,
    vd_L: float,
    dose_normal_mg: float,
    **kwargs,
) -> list[RenalImpairmentDoseResult]:
    """Return one RenalImpairmentDoseResult per CKD stage (6 entries)."""
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
