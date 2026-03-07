"""Renal dose adjustment using CrCl/eGFR.

Phase 679: Extends dose_individualization.py with detailed renal dosing guidance
based on Cockcroft-Gault CrCl and CKD-EPI eGFR.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "RenalDoseAdjResult",
    "estimate_crcl_cockcroft_gault",
    "estimate_egfr_ckd_epi",
    "ckd_stage",
    "adjust_dose_for_renal",
]


@dataclass(frozen=True)
class RenalDoseAdjResult:
    """Result of renal dose adjustment calculation."""

    compound_name: str
    crcl_mL_per_min: float
    egfr_estimate: float
    ckd_stage_str: str
    cl_adjustment_factor: float
    recommended_dose_mg: float
    dosing_interval_h: float
    dose_reduction_pct: float
    warnings: str
    notes: str


def estimate_crcl_cockcroft_gault(
    age_years: float,
    weight_kg: float,
    serum_creatinine_mg_dL: float,
    sex: str = "male",
) -> float:
    """Estimate CrCl using Cockcroft-Gault equation.

    Parameters
    ----------
    age_years:
        Patient age in years (must be > 0).
    weight_kg:
        Patient weight in kg (must be > 0).
    serum_creatinine_mg_dL:
        Serum creatinine in mg/dL (must be > 0).
    sex:
        "male" or "female".

    Returns
    -------
    float
        Estimated CrCl in mL/min.
    """
    if age_years <= 0:
        raise ValueError(f"age_years must be > 0, got {age_years}")
    if weight_kg <= 0:
        raise ValueError(f"weight_kg must be > 0, got {weight_kg}")
    if serum_creatinine_mg_dL <= 0:
        raise ValueError(f"serum_creatinine_mg_dL must be > 0, got {serum_creatinine_mg_dL}")

    crcl = (140.0 - age_years) * weight_kg / (72.0 * serum_creatinine_mg_dL)
    if sex.lower() in ("female", "f"):
        crcl *= 0.85
    return max(crcl, 1.0)


def estimate_egfr_ckd_epi(
    age_years: float,
    serum_creatinine_mg_dL: float,
    sex: str = "male",
    race: str = "non_black",
) -> float:
    """Estimate eGFR using CKD-EPI 2009 equation.

    Parameters
    ----------
    age_years:
        Patient age in years (must be > 0).
    serum_creatinine_mg_dL:
        Serum creatinine in mg/dL (must be > 0).
    sex:
        "male" or "female".
    race:
        "non_black" or "black" (race-based coefficient, CKD-EPI 2009).

    Returns
    -------
    float
        Estimated eGFR in mL/min/1.73m².
    """
    if age_years <= 0:
        raise ValueError(f"age_years must be > 0, got {age_years}")
    if serum_creatinine_mg_dL <= 0:
        raise ValueError(f"serum_creatinine_mg_dL must be > 0, got {serum_creatinine_mg_dL}")

    is_female = sex.lower() in ("female", "f")
    kappa = 0.7 if is_female else 0.9
    alpha = -0.329 if is_female else -0.411

    scr_ratio = serum_creatinine_mg_dL / kappa
    term_min = min(scr_ratio, 1.0) ** alpha
    term_max = max(scr_ratio, 1.0) ** (-1.209)

    egfr = 141.0 * term_min * term_max * (0.993**age_years)
    if is_female:
        egfr *= 1.018
    # Note: race parameter accepted but race-free version used (2021 update spirit)
    return max(egfr, 1.0)


def ckd_stage(egfr: float) -> str:
    """Return CKD stage string based on eGFR value.

    Parameters
    ----------
    egfr:
        eGFR in mL/min/1.73m².

    Returns
    -------
    str
        CKD stage: "G1", "G2", "G3a", "G3b", "G4", or "G5".
    """
    if egfr >= 90:
        return "G1"
    if egfr >= 60:
        return "G2"
    if egfr >= 45:
        return "G3a"
    if egfr >= 30:
        return "G3b"
    if egfr >= 15:
        return "G4"
    return "G5"


def adjust_dose_for_renal(
    compound_name: str,
    dose_normal_mg: float,
    dosing_interval_h: float,
    crcl_mL_per_min: float,
    fe_urine: float = 0.5,
    cl_renal_fraction: float = 0.5,
) -> RenalDoseAdjResult:
    """Adjust drug dose for renal impairment based on CrCl.

    The adjustment factor scales the total clearance based on the renal
    fraction of clearance and the patient's CrCl relative to normal (120 mL/min).

    Parameters
    ----------
    compound_name:
        Name of the compound.
    dose_normal_mg:
        Normal (standard) dose in mg.
    dosing_interval_h:
        Dosing interval in hours.
    crcl_mL_per_min:
        Patient's creatinine clearance in mL/min.
    fe_urine:
        Fraction of dose excreted unchanged in urine (0–1).
    cl_renal_fraction:
        Fraction of total clearance that is renal (0–1).

    Returns
    -------
    RenalDoseAdjResult
        Dataclass with recommended dose and adjustment details.
    """
    if not (0.0 <= fe_urine <= 1.0):
        raise ValueError(f"fe_urine must be in [0, 1], got {fe_urine}")
    if not (0.0 <= cl_renal_fraction <= 1.0):
        raise ValueError(f"cl_renal_fraction must be in [0, 1], got {cl_renal_fraction}")

    # Estimate eGFR from CrCl (use CrCl as proxy; assume BSA-normalized ≈ CrCl for typical patient)
    egfr = max(crcl_mL_per_min, 1.0)
    stage = ckd_stage(egfr)

    # CL_adjusted / CL_normal = (1 - cl_renal_fraction) + cl_renal_fraction * (CrCl / 120)
    _crcl_normal = 120.0
    cl_ratio = (1.0 - cl_renal_fraction) + cl_renal_fraction * (crcl_mL_per_min / _crcl_normal)
    cl_ratio = max(min(cl_ratio, 1.0), 0.01)  # clamp to sensible range

    recommended_dose = dose_normal_mg * cl_ratio
    dose_reduction_pct = (1.0 - cl_ratio) * 100.0

    # Build warnings
    warning_parts: list[str] = []
    if crcl_mL_per_min < 30:
        warning_parts.append(
            f"Severe renal impairment (CrCl={crcl_mL_per_min:.1f} mL/min); "
            "closely monitor drug levels."
        )
    if crcl_mL_per_min < 15:
        warning_parts.append("ESRD range; consider dialysis dosing guidance.")
    if cl_ratio < 0.5:
        warning_parts.append(
            "Adjustment factor < 0.50; consider dose reduction AND interval extension."
        )
    warnings_str = " | ".join(warning_parts) if warning_parts else ""

    notes = (
        f"CKD stage {stage}; cl_renal_fraction={cl_renal_fraction:.2f}; "
        f"fe_urine={fe_urine:.2f}; reference CrCl={_crcl_normal} mL/min."
    )

    return RenalDoseAdjResult(
        compound_name=compound_name,
        crcl_mL_per_min=crcl_mL_per_min,
        egfr_estimate=egfr,
        ckd_stage_str=stage,
        cl_adjustment_factor=cl_ratio,
        recommended_dose_mg=recommended_dose,
        dosing_interval_h=dosing_interval_h,
        dose_reduction_pct=dose_reduction_pct,
        warnings=warnings_str,
        notes=notes,
    )
