"""Dose adjustment for hepatic impairment using Child-Pugh scoring.

Provides FDA guidance-aligned dose recommendations based on Child-Pugh grade,
fraction eliminated hepatically, and protein-binding status.
"""

from __future__ import annotations

__all__ = [
    "ChildPughScore",
    "HepaticImpairmentResult",
    "child_pugh_score",
    "adjust_dose_hepatic",
]

from dataclasses import dataclass

_VALID_ASCITES = {"none", "mild", "moderate_severe"}
_VALID_ENCEPHALOPATHY = {"none", "grade_1_2", "grade_3_4"}

# CL retained fraction by Child-Pugh grade (FDA guidance)
_CL_FACTORS: dict[str, float] = {"A": 0.75, "B": 0.5, "C": 0.25}


@dataclass(frozen=True)
class ChildPughScore:
    """Child-Pugh scoring result for hepatic function assessment."""

    bilirubin_mg_dL: float
    albumin_g_dL: float
    pt_seconds_prolonged: float  # seconds above normal (INR proxy)
    ascites: str  # "none", "mild", "moderate_severe"
    encephalopathy: str  # "none", "grade_1_2", "grade_3_4"
    score: int  # 5-15
    grade: str  # "A" (5-6), "B" (7-9), "C" (10-15)


@dataclass(frozen=True)
class HepaticImpairmentResult:
    """Dose adjustment result for hepatically impaired patients."""

    drug_name: str
    child_pugh_grade: str  # "A", "B", "C"
    child_pugh_score: int
    fe_hepatic: float  # fraction eliminated hepatically
    cl_normal_L_per_h: float
    cl_adjusted_L_per_h: float
    dose_normal_mg: float
    dose_adjusted_mg: float
    dose_reduction_pct: float
    t_half_normal_h: float
    t_half_adjusted_h: float
    fup_adjustment_factor: float  # albumin-dependent PPB change
    recommendation: str
    contraindicated: bool
    notes: str


def child_pugh_score(
    bilirubin_mg_dL: float,
    albumin_g_dL: float,
    pt_seconds_prolonged: float,
    ascites: str = "none",
    encephalopathy: str = "none",
) -> ChildPughScore:
    """Calculate Child-Pugh score and grade from clinical parameters.

    Parameters
    ----------
    bilirubin_mg_dL:
        Serum bilirubin in mg/dL (> 0).
    albumin_g_dL:
        Serum albumin in g/dL (> 0).
    pt_seconds_prolonged:
        Prothrombin time seconds above normal (>= 0).
    ascites:
        Ascites severity: "none", "mild", or "moderate_severe".
    encephalopathy:
        Encephalopathy grade: "none", "grade_1_2", or "grade_3_4".

    Returns
    -------
    ChildPughScore
    """
    if bilirubin_mg_dL <= 0.0:
        raise ValueError(f"bilirubin_mg_dL must be > 0, got {bilirubin_mg_dL}")
    if albumin_g_dL <= 0.0:
        raise ValueError(f"albumin_g_dL must be > 0, got {albumin_g_dL}")
    if pt_seconds_prolonged < 0.0:
        raise ValueError(f"pt_seconds_prolonged must be >= 0, got {pt_seconds_prolonged}")
    if ascites not in _VALID_ASCITES:
        raise ValueError(f"ascites must be one of {_VALID_ASCITES}, got {ascites!r}")
    if encephalopathy not in _VALID_ENCEPHALOPATHY:
        raise ValueError(
            f"encephalopathy must be one of {_VALID_ENCEPHALOPATHY}, got {encephalopathy!r}"
        )

    score = 0

    # Bilirubin scoring
    if bilirubin_mg_dL < 2.0:
        score += 1
    elif bilirubin_mg_dL <= 3.0:
        score += 2
    else:
        score += 3

    # Albumin scoring
    if albumin_g_dL > 3.5:
        score += 1
    elif albumin_g_dL >= 2.8:
        score += 2
    else:
        score += 3

    # PT prolongation scoring
    if pt_seconds_prolonged < 4.0:
        score += 1
    elif pt_seconds_prolonged <= 6.0:
        score += 2
    else:
        score += 3

    # Ascites scoring
    ascites_map = {"none": 1, "mild": 2, "moderate_severe": 3}
    score += ascites_map[ascites]

    # Encephalopathy scoring
    enc_map = {"none": 1, "grade_1_2": 2, "grade_3_4": 3}
    score += enc_map[encephalopathy]

    if score <= 6:
        grade = "A"
    elif score <= 9:
        grade = "B"
    else:
        grade = "C"

    return ChildPughScore(
        bilirubin_mg_dL=bilirubin_mg_dL,
        albumin_g_dL=albumin_g_dL,
        pt_seconds_prolonged=pt_seconds_prolonged,
        ascites=ascites,
        encephalopathy=encephalopathy,
        score=score,
        grade=grade,
    )


def adjust_dose_hepatic(
    drug_name: str,
    child_pugh: ChildPughScore,
    fe_hepatic: float,
    cl_normal_L_per_h: float,
    vd_L: float,
    dose_normal_mg: float,
    protein_binding: float = 0.9,
) -> HepaticImpairmentResult:
    """Adjust dose for hepatic impairment using Child-Pugh grade.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    child_pugh:
        Child-Pugh score object (from :func:`child_pugh_score`).
    fe_hepatic:
        Fraction of drug eliminated hepatically (0 to 1).
    cl_normal_L_per_h:
        Normal (healthy) total clearance in L/h (> 0).
    vd_L:
        Volume of distribution in L (> 0).
    dose_normal_mg:
        Usual dose in mg (> 0).
    protein_binding:
        Fraction bound to plasma protein in normal patients (0 to 1).

    Returns
    -------
    HepaticImpairmentResult
    """
    if not (0.0 <= fe_hepatic <= 1.0):
        raise ValueError(f"fe_hepatic must be in [0, 1], got {fe_hepatic}")
    if cl_normal_L_per_h <= 0.0:
        raise ValueError(f"cl_normal_L_per_h must be > 0, got {cl_normal_L_per_h}")
    if vd_L <= 0.0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if dose_normal_mg <= 0.0:
        raise ValueError(f"dose_normal_mg must be > 0, got {dose_normal_mg}")

    grade = child_pugh.grade

    # CL adjustment: fraction of normal CL retained based on grade and fe_hepatic
    cl_retained = _CL_FACTORS[grade]
    cl_factor = 1.0 - fe_hepatic * (1.0 - cl_retained)
    cl_adj = cl_normal_L_per_h * cl_factor

    # fup adjustment: reduced albumin → higher free fraction for highly bound drugs
    alb_normal = 4.0  # g/dL
    alb_factor = min(1.5, alb_normal / max(child_pugh.albumin_g_dL, 0.5))
    if protein_binding > 0.8:
        fup_adj_factor = alb_factor
    else:
        fup_adj_factor = 1.0

    # Dose adjustment proportional to CL change
    dose_adj = dose_normal_mg * cl_factor
    dose_reduction_pct = (1.0 - cl_factor) * 100.0

    # Half-life calculations
    t_half_normal = 0.693 * vd_L / cl_normal_L_per_h
    t_half_adj = 0.693 * vd_L / max(cl_adj, 1e-6)

    # Contraindication: Child-Pugh C for drugs primarily eliminated hepatically
    contraindicated = (grade == "C") and (fe_hepatic > 0.5)

    # Clinical recommendation
    if grade == "A":
        recommendation = "Use with caution; monitor liver function"
    elif grade == "B":
        recommendation = "Reduce dose by ~50%; monitor LFTs"
    else:  # grade == "C"
        recommendation = (
            "Avoid if possible; significant dose reduction required; frequent monitoring"
        )

    notes = (
        f"FDA guidance Child-Pugh method; fe_hepatic={fe_hepatic:.2f}; "
        f"CL adjusted from {cl_normal_L_per_h:.2f} to {cl_adj:.2f} L/h; "
        f"albumin factor={alb_factor:.2f}"
    )

    return HepaticImpairmentResult(
        drug_name=drug_name,
        child_pugh_grade=grade,
        child_pugh_score=child_pugh.score,
        fe_hepatic=fe_hepatic,
        cl_normal_L_per_h=cl_normal_L_per_h,
        cl_adjusted_L_per_h=cl_adj,
        dose_normal_mg=dose_normal_mg,
        dose_adjusted_mg=dose_adj,
        dose_reduction_pct=dose_reduction_pct,
        t_half_normal_h=t_half_normal,
        t_half_adjusted_h=t_half_adj,
        fup_adjustment_factor=fup_adj_factor,
        recommendation=recommendation,
        contraindicated=contraindicated,
        notes=notes,
    )
