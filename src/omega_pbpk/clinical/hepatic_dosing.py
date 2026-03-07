"""Hepatic dose adjustment using Child-Pugh scoring."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChildPughResult:
    """Result of Child-Pugh scoring."""

    bilirubin_pts: int
    albumin_pts: int
    pt_pts: int
    ascites_pts: int
    encephalopathy_pts: int
    total_score: int
    child_pugh_class: str  # "A", "B", or "C"
    notes: str


@dataclass
class HepaticDosingResult:
    """Result of hepatic dose adjustment."""

    drug_name: str
    normal_dose_mg: float
    child_pugh_class: str
    dose_factor: float
    adjusted_dose_mg: float
    cl_adjusted_L_per_h: float
    t_half_adjusted_h: float
    notes: list[str] = field(default_factory=list)


def child_pugh_score(
    bilirubin_mg_dL: float,
    albumin_g_dL: float,
    pt_seconds_prolonged: float,
    ascites: str,
    encephalopathy: str,
) -> ChildPughResult:
    """Compute Child-Pugh score and class for hepatic function assessment.

    Parameters
    ----------
    bilirubin_mg_dL:
        Total serum bilirubin in mg/dL.
    albumin_g_dL:
        Serum albumin in g/dL.
    pt_seconds_prolonged:
        Prothrombin time prolongation in seconds above control.
    ascites:
        Ascites status: "none", "mild", or "severe".
    encephalopathy:
        Hepatic encephalopathy grade: "none", "grade_1_2", or "grade_3_4".

    Returns
    -------
    ChildPughResult
    """
    if bilirubin_mg_dL < 0:
        raise ValueError("bilirubin_mg_dL must be >= 0")
    if albumin_g_dL <= 0:
        raise ValueError("albumin_g_dL must be > 0")
    if pt_seconds_prolonged < 0:
        raise ValueError("pt_seconds_prolonged must be >= 0")
    valid_ascites = {"none", "mild", "severe"}
    if ascites not in valid_ascites:
        raise ValueError(f"ascites must be one of {valid_ascites}, got '{ascites}'")
    valid_enc = {"none", "grade_1_2", "grade_3_4"}
    if encephalopathy not in valid_enc:
        raise ValueError(f"encephalopathy must be one of {valid_enc}, got '{encephalopathy}'")

    # Bilirubin points
    if bilirubin_mg_dL < 2.0:
        bilirubin_pts = 1
    elif bilirubin_mg_dL <= 3.0:
        bilirubin_pts = 2
    else:
        bilirubin_pts = 3

    # Albumin points
    if albumin_g_dL > 3.5:
        albumin_pts = 1
    elif albumin_g_dL >= 2.8:
        albumin_pts = 2
    else:
        albumin_pts = 3

    # PT prolongation points
    if pt_seconds_prolonged < 4.0:
        pt_pts = 1
    elif pt_seconds_prolonged <= 6.0:
        pt_pts = 2
    else:
        pt_pts = 3

    # Ascites points
    ascites_map = {"none": 1, "mild": 2, "severe": 3}
    ascites_pts = ascites_map[ascites]

    # Encephalopathy points
    enc_map = {"none": 1, "grade_1_2": 2, "grade_3_4": 3}
    encephalopathy_pts = enc_map[encephalopathy]

    total_score = bilirubin_pts + albumin_pts + pt_pts + ascites_pts + encephalopathy_pts

    if total_score <= 6:
        child_pugh_class = "A"
        notes = "Class A (5-6 pts): mild hepatic impairment"
    elif total_score <= 9:
        child_pugh_class = "B"
        notes = "Class B (7-9 pts): moderate hepatic impairment"
    else:
        child_pugh_class = "C"
        notes = "Class C (10-15 pts): severe hepatic impairment"

    return ChildPughResult(
        bilirubin_pts=bilirubin_pts,
        albumin_pts=albumin_pts,
        pt_pts=pt_pts,
        ascites_pts=ascites_pts,
        encephalopathy_pts=encephalopathy_pts,
        total_score=total_score,
        child_pugh_class=child_pugh_class,
        notes=notes,
    )


def adjust_dose_hepatic(
    drug_name: str,
    normal_dose_mg: float,
    child_pugh_class: str,
    f_hepatic_metabolism: float,
    cl_normal_L_per_h: float,
    vd_L: float,
) -> HepaticDosingResult:
    """Compute hepatically adjusted dose and PK parameters.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    normal_dose_mg:
        Standard dose in mg for normal hepatic function.
    child_pugh_class:
        "A", "B", or "C".
    f_hepatic_metabolism:
        Fraction of drug metabolized hepatically (0–1).
    cl_normal_L_per_h:
        Total clearance in normal subjects (L/h).
    vd_L:
        Volume of distribution (L).

    Returns
    -------
    HepaticDosingResult
    """
    if not drug_name or not drug_name.strip():
        raise ValueError("drug_name must be a non-empty string")
    if normal_dose_mg <= 0:
        raise ValueError("normal_dose_mg must be > 0")
    if child_pugh_class not in ("A", "B", "C"):
        raise ValueError("child_pugh_class must be 'A', 'B', or 'C'")
    if not 0.0 <= f_hepatic_metabolism <= 1.0:
        raise ValueError("f_hepatic_metabolism must be between 0 and 1")
    if cl_normal_L_per_h <= 0:
        raise ValueError("cl_normal_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")

    dose_factors = {"A": 1.0, "B": 0.75, "C": 0.50}
    dose_factor = dose_factors[child_pugh_class]

    adjusted_dose_mg = normal_dose_mg * dose_factor
    cl_adjusted_L_per_h = cl_normal_L_per_h * dose_factor
    t_half_adjusted_h = 0.693 * vd_L / cl_adjusted_L_per_h

    notes: list[str] = []
    if child_pugh_class == "A":
        notes.append("Class A: no dose adjustment required")
    elif child_pugh_class == "B":
        notes.append("Class B: reduce dose by 25%")
    else:
        notes.append("Class C: reduce dose by 50%; use with caution")

    if f_hepatic_metabolism > 0.7:
        notes.append(
            f"High hepatic metabolism fraction ({f_hepatic_metabolism:.0%}); "
            "dose adjustment is clinically significant"
        )
    elif f_hepatic_metabolism < 0.3:
        notes.append(
            f"Low hepatic metabolism fraction ({f_hepatic_metabolism:.0%}); "
            "dose adjustment may have limited impact"
        )

    return HepaticDosingResult(
        drug_name=drug_name,
        normal_dose_mg=normal_dose_mg,
        child_pugh_class=child_pugh_class,
        dose_factor=dose_factor,
        adjusted_dose_mg=adjusted_dose_mg,
        cl_adjusted_L_per_h=cl_adjusted_L_per_h,
        t_half_adjusted_h=t_half_adjusted_h,
        notes=notes,
    )
