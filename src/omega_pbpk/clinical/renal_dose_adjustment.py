"""
Phase 784 — Renal dose adjustment based on GFR and drug characteristics.

Logic:
- CKD stage from eGFR
- Drug factor df = 1 - fe_urine * (1 - egfr/normal_egfr), clamped [0.1, 1.0]
- Three methods: dose_reduction, interval_extension, combined
- Dialysis supplement flag: egfr < 15 and fe_urine > 0.5
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RenalDoseAdjustmentResult:
    drug_name: str
    egfr_ml_per_min: float
    ckd_stage: str  # G1, G2, G3a, G3b, G4, G5
    fe_urine: float  # fraction excreted unchanged in urine
    normal_dose_mg: float
    normal_interval_h: float
    adjusted_dose_mg: float
    adjusted_interval_h: float
    dose_fraction: float  # adjusted/normal
    dose_reduction_pct: float
    method: str  # "dose_reduction", "interval_extension", "combined"
    requires_dialysis_supplement: bool
    notes: str


def ckd_stage_from_egfr(egfr_ml_per_min: float) -> str:
    """Return KDIGO CKD stage string from eGFR in mL/min."""
    if egfr_ml_per_min >= 90.0:
        return "G1"
    elif egfr_ml_per_min >= 60.0:
        return "G2"
    elif egfr_ml_per_min >= 45.0:
        return "G3a"
    elif egfr_ml_per_min >= 30.0:
        return "G3b"
    elif egfr_ml_per_min >= 15.0:
        return "G4"
    else:
        return "G5"


def adjust_dose_for_renal_impairment(
    drug_name: str,
    normal_dose_mg: float,
    normal_interval_h: float,
    fe_urine: float,
    egfr_ml_per_min: float,
    normal_egfr_ml_per_min: float = 100.0,
    method: str = "dose_reduction",
) -> RenalDoseAdjustmentResult:
    """Calculate dose adjustment for renal impairment."""

    # Input validation
    if not (0.0 <= fe_urine <= 1.0):
        raise ValueError(f"fe_urine must be in [0, 1], got {fe_urine}")
    if egfr_ml_per_min < 0.0:
        raise ValueError(f"egfr_ml_per_min must be >= 0, got {egfr_ml_per_min}")

    ckd_stage = ckd_stage_from_egfr(egfr_ml_per_min)

    # Drug factor
    egfr_ratio = egfr_ml_per_min / normal_egfr_ml_per_min
    df_raw = 1.0 - fe_urine * (1.0 - egfr_ratio)
    df = max(0.1, min(1.0, df_raw))

    # Method-specific adjustments
    valid_methods = {"dose_reduction", "interval_extension", "combined"}
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got {method!r}")

    if method == "dose_reduction":
        adjusted_dose = normal_dose_mg * df
        adjusted_interval = normal_interval_h
    elif method == "interval_extension":
        adjusted_dose = normal_dose_mg
        if df > 0.0:
            adjusted_interval = normal_interval_h / df
        else:
            adjusted_interval = normal_interval_h * 10.0  # extreme extension
    else:  # combined
        sqrt_df = math.sqrt(df)
        adjusted_dose = normal_dose_mg * sqrt_df
        if sqrt_df > 0.0:
            adjusted_interval = normal_interval_h / sqrt_df
        else:
            adjusted_interval = normal_interval_h * 10.0

    # Dose fraction relative to normal total daily dose
    # dose_fraction = (adjusted_dose / adjusted_interval) / (normal_dose_mg / normal_interval_h)
    if normal_interval_h > 0.0 and adjusted_interval > 0.0:
        normal_daily_dose = normal_dose_mg / normal_interval_h
        adjusted_daily_dose = adjusted_dose / adjusted_interval
        dose_fraction = adjusted_daily_dose / normal_daily_dose if normal_daily_dose > 0.0 else 1.0
    else:
        dose_fraction = df

    dose_fraction = max(0.0, min(1.0, dose_fraction))
    dose_reduction_pct = (1.0 - dose_fraction) * 100.0

    requires_dialysis_supplement = egfr_ml_per_min < 15.0 and fe_urine > 0.5

    # Notes
    notes_parts = [
        f"CKD stage {ckd_stage} (eGFR={egfr_ml_per_min:.1f} mL/min)",
        f"df={df:.3f} (fe_urine={fe_urine:.2f})",
        f"method={method}",
    ]
    if requires_dialysis_supplement:
        notes_parts.append("Dialysis supplemental dose may be required")
    notes = "; ".join(notes_parts)

    return RenalDoseAdjustmentResult(
        drug_name=drug_name,
        egfr_ml_per_min=egfr_ml_per_min,
        ckd_stage=ckd_stage,
        fe_urine=fe_urine,
        normal_dose_mg=normal_dose_mg,
        normal_interval_h=normal_interval_h,
        adjusted_dose_mg=round(adjusted_dose, 4),
        adjusted_interval_h=round(adjusted_interval, 4),
        dose_fraction=round(dose_fraction, 6),
        dose_reduction_pct=round(dose_reduction_pct, 4),
        method=method,
        requires_dialysis_supplement=requires_dialysis_supplement,
        notes=notes,
    )


def recommend_dose_adjustment(
    drug_name: str,
    normal_dose_mg: float,
    normal_interval_h: float,
    fe_urine: float,
    egfr_ml_per_min: float,
) -> str:
    """Return a human-readable recommendation string for dose adjustment."""
    result = adjust_dose_for_renal_impairment(
        drug_name=drug_name,
        normal_dose_mg=normal_dose_mg,
        normal_interval_h=normal_interval_h,
        fe_urine=fe_urine,
        egfr_ml_per_min=egfr_ml_per_min,
        method="dose_reduction",
    )

    lines = [
        f"Drug: {drug_name}",
        f"CKD Stage: {result.ckd_stage} (eGFR {egfr_ml_per_min:.1f} mL/min)",
        f"Renal elimination fraction: {fe_urine * 100:.0f}%",
        f"Recommended dose: {result.adjusted_dose_mg:.1f} mg every {result.normal_interval_h:.0f} h",
        f"Dose reduction: {result.dose_reduction_pct:.1f}%",
    ]
    if result.requires_dialysis_supplement:
        lines.append("NOTE: Supplemental dose after dialysis may be required.")
    return "\n".join(lines)
