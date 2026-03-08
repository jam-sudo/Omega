"""Drug Hepatic Reserve Assessment — Phase 629.

Assesses drug dosing safety in patients with reduced hepatic reserve
(cirrhosis, NASH, acute liver failure) using Child-Pugh scoring and
MELD score to guide dose adjustments.

References:
    - Pugh RNH et al. Br J Surg 1973.
    - Kamath PS et al. Hepatology 2001 (MELD).
    - FDA Guidance: Pharmacokinetics in Patients with Impaired Hepatic Function (2003).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class HepaticReserveResult:
    """Result of hepatic reserve assessment for drug dosing."""

    drug_name: str
    child_pugh_score: int
    child_pugh_class: str
    meld_score: float
    cl_hepatic_reduced: float
    cl_reduction_factor: float
    recommended_dose_factor: float
    contraindicated: bool
    exposure_increase_fold: float
    monitoring_required: str
    notes: str


def _child_pugh_score(
    bilirubin_umol_L: float,
    albumin_g_dL: float,
    inr: float,
    ascites: str,
    encephalopathy_grade: int,
) -> int:
    """Calculate Child-Pugh score (5-15 points)."""
    score = 0

    # Bilirubin
    if bilirubin_umol_L < 34.0:
        score += 1
    elif bilirubin_umol_L <= 50.0:
        score += 2
    else:
        score += 3

    # Albumin
    if albumin_g_dL > 35.0:
        score += 1
    elif albumin_g_dL >= 28.0:
        score += 2
    else:
        score += 3

    # INR
    if inr < 1.7:
        score += 1
    elif inr <= 2.3:
        score += 2
    else:
        score += 3

    # Ascites
    if ascites == "none":
        score += 1
    elif ascites == "mild":
        score += 2
    else:  # severe
        score += 3

    # Encephalopathy grade (0=none, 1=grade1/2, 2=grade3/4)
    if encephalopathy_grade == 0:
        score += 1
    elif encephalopathy_grade == 1:
        score += 2
    else:  # grade 2 → grade 3/4
        score += 3

    return score


def _child_pugh_class(score: int) -> str:
    """Return Child-Pugh class A, B, or C from score."""
    if score <= 6:
        return "A"
    elif score <= 9:
        return "B"
    else:
        return "C"


def _cl_reduction_factor(cp_class: str) -> float:
    """CL reduction factor by Child-Pugh class."""
    if cp_class == "A":
        return 0.75
    elif cp_class == "B":
        return 0.50
    else:
        return 0.25


def _meld_score(
    bilirubin_umol_L: float,
    inr: float,
    creatinine_umol_L: float,
) -> float:
    """Calculate MELD score, clamped to [6, 40]."""
    bilirubin_mg_dL = bilirubin_umol_L / 17.1
    creatinine_mg_dL = creatinine_umol_L / 88.4

    # Clamp values to avoid log(0)
    bili_val = max(bilirubin_mg_dL, 1.0)
    inr_val = max(inr, 1.0)
    cr_val = max(creatinine_mg_dL, 1.0)
    # Cap creatinine at 4.0 per UNOS MELD policy
    cr_val = min(cr_val, 4.0)

    meld = 3.78 * math.log(bili_val) + 11.2 * math.log(inr_val) + 9.57 * math.log(cr_val) + 6.43

    return max(6.0, min(40.0, meld))


def _validate_inputs(
    bilirubin_umol_L: float,
    albumin_g_dL: float,
    inr: float,
    ascites: str,
    encephalopathy_grade: int,
) -> None:
    if bilirubin_umol_L <= 0:
        raise ValueError(f"bilirubin_umol_L must be > 0, got {bilirubin_umol_L}")
    if albumin_g_dL <= 0:
        raise ValueError(f"albumin_g_dL must be > 0, got {albumin_g_dL}")
    if inr <= 0:
        raise ValueError(f"inr must be > 0, got {inr}")
    if ascites not in ("none", "mild", "severe"):
        raise ValueError(f"ascites must be 'none', 'mild', or 'severe', got '{ascites}'")
    if encephalopathy_grade not in (0, 1, 2):
        raise ValueError(f"encephalopathy_grade must be 0, 1, or 2, got {encephalopathy_grade}")


def assess_hepatic_reserve(
    drug_name: str,
    cl_normal_L_per_h: float,
    extraction_ratio: float,
    bilirubin_umol_L: float,
    albumin_g_dL: float,
    inr: float,
    ascites: str,
    encephalopathy_grade: int,
    creatinine_umol_L_for_meld: float = 88.0,
) -> HepaticReserveResult:
    """Assess hepatic reserve and provide drug dosing recommendations.

    Parameters
    ----------
    drug_name : str
        Name of the drug being assessed.
    cl_normal_L_per_h : float
        Normal (healthy) hepatic clearance in L/h.
    extraction_ratio : float
        Hepatic extraction ratio (0.0 to 1.0).
    bilirubin_umol_L : float
        Serum bilirubin in µmol/L.
    albumin_g_dL : float
        Serum albumin in g/dL.
    inr : float
        International Normalized Ratio.
    ascites : str
        Ascites severity: "none", "mild", or "severe".
    encephalopathy_grade : int
        Hepatic encephalopathy grade: 0 (none), 1 (grade 1/2), 2 (grade 3/4).
    creatinine_umol_L_for_meld : float, optional
        Serum creatinine in µmol/L for MELD calculation. Default 88.0.

    Returns
    -------
    HepaticReserveResult
        Frozen dataclass with assessment results.
    """
    _validate_inputs(bilirubin_umol_L, albumin_g_dL, inr, ascites, encephalopathy_grade)

    cp_score = _child_pugh_score(bilirubin_umol_L, albumin_g_dL, inr, ascites, encephalopathy_grade)
    cp_class = _child_pugh_class(cp_score)
    meld = _meld_score(bilirubin_umol_L, inr, creatinine_umol_L_for_meld)

    reduction_factor = _cl_reduction_factor(cp_class)
    cl_reduced = cl_normal_L_per_h * reduction_factor
    recommended_dose_factor = cl_reduced / cl_normal_L_per_h  # = reduction_factor
    exposure_increase_fold = cl_normal_L_per_h / cl_reduced

    contraindicated = cp_class == "C" and extraction_ratio > 0.6

    if cp_class == "A":
        monitoring = "standard"
    elif cp_class == "B":
        monitoring = "intensive"
    else:
        monitoring = "avoid" if contraindicated else "intensive"

    # Build notes
    parts = [
        f"Child-Pugh {cp_class} (score {cp_score}), MELD {meld:.1f}.",
        f"CL reduced to {reduction_factor * 100:.0f}% of normal.",
        f"Dose factor: {recommended_dose_factor:.2f}.",
    ]
    if contraindicated:
        parts.append("HIGH EXTRACTION RATIO drug in Child-Pugh C: avoid use.")
    notes = " ".join(parts)

    return HepaticReserveResult(
        drug_name=drug_name,
        child_pugh_score=cp_score,
        child_pugh_class=cp_class,
        meld_score=round(meld, 2),
        cl_hepatic_reduced=round(cl_reduced, 4),
        cl_reduction_factor=round(reduction_factor, 4),
        recommended_dose_factor=round(recommended_dose_factor, 4),
        contraindicated=contraindicated,
        exposure_increase_fold=round(exposure_increase_fold, 4),
        monitoring_required=monitoring,
        notes=notes,
    )


def dose_recommendation_by_cp_class(
    drug_name: str,
    cl_normal_L_per_h: float,
    extraction_ratio: float,
) -> dict:
    """Return dose adjustment factors for all Child-Pugh classes.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    cl_normal_L_per_h : float
        Normal hepatic clearance in L/h.
    extraction_ratio : float
        Hepatic extraction ratio (0.0 to 1.0).

    Returns
    -------
    dict
        {"A": float, "B": float, "C": float} dose adjustment factors.
    """
    result = {}
    for cp_class in ("A", "B", "C"):
        factor = _cl_reduction_factor(cp_class)
        result[cp_class] = round(factor, 4)
    return result
