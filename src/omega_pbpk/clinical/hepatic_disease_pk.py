"""Hepatic disease PK adjustment — Child-Pugh scoring and CL/Vd adjustments."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "HepaticDiseaseAdjResult",
    "child_pugh_score",
    "estimate_hepatic_pk_adjustment",
    "simulate_hepatic_disease_pk_profiles",
]


@dataclass(frozen=True)
class HepaticDiseaseAdjResult:
    child_pugh_class: str
    child_pugh_score_val: int
    cl_normal_L_per_h: float
    cl_adjusted_L_per_h: float
    vd_normal_L: float
    vd_adjusted_L: float
    t_half_normal_h: float
    t_half_adjusted_h: float
    dose_adjustment_factor: float
    recommended_dose_fraction: str
    fup_normal: float
    fup_adjusted: float
    notes: str


def child_pugh_score(
    bilirubin_mg_dL: float,
    albumin_g_dL: float,
    pt_seconds_prolonged: float,
    ascites: str,
    encephalopathy: str,
) -> tuple[int, str]:
    """Compute Child-Pugh score and class (A, B, or C)."""
    # Bilirubin points
    if bilirubin_mg_dL < 2.0:
        bil_pts = 1
    elif bilirubin_mg_dL <= 3.0:
        bil_pts = 2
    else:
        bil_pts = 3

    # Albumin points
    if albumin_g_dL > 3.5:
        alb_pts = 1
    elif albumin_g_dL >= 2.8:
        alb_pts = 2
    else:
        alb_pts = 3

    # PT prolonged points
    if pt_seconds_prolonged < 4.0:
        pt_pts = 1
    elif pt_seconds_prolonged <= 6.0:
        pt_pts = 2
    else:
        pt_pts = 3

    # Ascites points
    ascites_lower = ascites.lower()
    if ascites_lower == "none":
        asc_pts = 1
    elif ascites_lower == "mild":
        asc_pts = 2
    else:
        asc_pts = 3

    # Encephalopathy points
    enc_lower = encephalopathy.lower()
    if enc_lower == "none":
        enc_pts = 1
    elif enc_lower in ("grade 1-2", "grade 1", "grade 2", "mild", "moderate"):
        enc_pts = 2
    else:
        enc_pts = 3

    total = bil_pts + alb_pts + pt_pts + asc_pts + enc_pts

    if total <= 6:
        cp_class = "A"
    elif total <= 9:
        cp_class = "B"
    else:
        cp_class = "C"

    return (total, cp_class)


def estimate_hepatic_pk_adjustment(
    cl_adult_L_per_h: float,
    vd_L: float,
    child_pugh_class: str,
    hepatic_extraction_ratio: float = 0.5,
    fu_plasma: float = 0.1,
) -> HepaticDiseaseAdjResult:
    """Estimate PK adjustments for hepatic disease based on Child-Pugh class."""
    if child_pugh_class not in {"A", "B", "C"}:
        raise ValueError(f"child_pugh_class must be 'A', 'B', or 'C', got '{child_pugh_class}'")
    if not (0.0 <= hepatic_extraction_ratio <= 1.0):
        raise ValueError("hepatic_extraction_ratio must be between 0 and 1")
    if not (0.0 < fu_plasma <= 1.0):
        raise ValueError("fu_plasma must be between 0 and 1")

    # Base CL reduction factors per FDA 2003 guidance
    cl_reduction_map = {"A": 0.75, "B": 0.50, "C": 0.25}
    fup_multiplier_map = {"A": 1.1, "B": 1.3, "C": 1.5}
    cp_score_map = {"A": 6, "B": 8, "C": 12}  # Representative scores

    cl_factor = cl_reduction_map[child_pugh_class]

    # Additional reduction for high-extraction drugs (ER > 0.7)
    if hepatic_extraction_ratio > 0.7:
        cl_factor = cl_factor * 0.8

    cl_adjusted = cl_adult_L_per_h * cl_factor

    # Vd adjustment: mild increase due to fluid accumulation / reduced albumin
    vd_factor_map = {"A": 1.0, "B": 1.1, "C": 1.2}
    vd_adjusted = vd_L * vd_factor_map[child_pugh_class]

    # Half-life calculations
    t_half_normal = (math.log(2) * vd_L) / cl_adult_L_per_h
    t_half_adjusted = (math.log(2) * vd_adjusted) / cl_adjusted

    # Dose adjustment factor
    dose_adjustment_factor = cl_adjusted / cl_adult_L_per_h

    # Recommended dose fraction
    if dose_adjustment_factor >= 0.70:
        recommended_dose_fraction = "100%"
    elif dose_adjustment_factor >= 0.45:
        recommended_dose_fraction = "75%"
    elif dose_adjustment_factor >= 0.20:
        recommended_dose_fraction = "50%"
    else:
        recommended_dose_fraction = "25%"

    # fup adjustment (hypoalbuminemia increases free fraction)
    fup_adjusted = fu_plasma * fup_multiplier_map[child_pugh_class]
    fup_adjusted = min(fup_adjusted, 1.0)

    # Representative CP score
    cp_score_val = cp_score_map[child_pugh_class]

    notes = (
        f"Child-Pugh {child_pugh_class}: CL reduced to {cl_factor * 100:.0f}% of normal. "
        f"High extraction: {'yes' if hepatic_extraction_ratio > 0.7 else 'no'}. "
        f"t½ increased from {t_half_normal:.1f}h to {t_half_adjusted:.1f}h."
    )

    return HepaticDiseaseAdjResult(
        child_pugh_class=child_pugh_class,
        child_pugh_score_val=cp_score_val,
        cl_normal_L_per_h=cl_adult_L_per_h,
        cl_adjusted_L_per_h=cl_adjusted,
        vd_normal_L=vd_L,
        vd_adjusted_L=vd_adjusted,
        t_half_normal_h=t_half_normal,
        t_half_adjusted_h=t_half_adjusted,
        dose_adjustment_factor=dose_adjustment_factor,
        recommended_dose_fraction=recommended_dose_fraction,
        fup_normal=fu_plasma,
        fup_adjusted=fup_adjusted,
        notes=notes,
    )


def _simulate_1cpt_oral(
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    ka_per_h: float,
    f_oral: float,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> dict:
    """Simulate 1-compartment oral PK using Forward Euler. Returns times, concentrations, cmax, auc, t_half."""
    times = []
    concs = []

    # State: depot (A_depot), central (A_central)
    a_depot = dose_mg * f_oral
    a_central = 0.0

    t = 0.0
    cmax = 0.0
    auc = 0.0
    prev_c = 0.0

    while t <= t_end_h + 1e-9:
        c = a_central / vd_L if vd_L > 0 else 0.0
        times.append(t)
        concs.append(c)

        if c > cmax:
            cmax = c

        # Trapezoidal AUC
        if len(times) > 1:
            auc += 0.5 * (prev_c + c) * dt_h

        prev_c = c

        # Forward Euler derivatives
        d_depot = -ka_per_h * a_depot
        d_central = ka_per_h * a_depot - (cl_L_per_h / vd_L) * a_central

        a_depot += d_depot * dt_h
        a_central += d_central * dt_h
        t += dt_h

    t_half = (math.log(2) * vd_L) / cl_L_per_h if cl_L_per_h > 0 else float("inf")

    return {
        "times_h": times,
        "c_plasma_mg_L": concs,
        "cmax": cmax,
        "auc": auc,
        "t_half": t_half,
    }


def simulate_hepatic_disease_pk_profiles(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    ka_per_h: float = 1.0,
    f_oral: float = 0.8,
    child_pugh_classes: list[str] | None = None,
) -> dict:
    """Simulate 1-cpt oral PK profiles for normal and each Child-Pugh class.

    Returns dict with keys: "normal", "A", "B", "C" (or subset if child_pugh_classes specified).
    Each value is a dict with: times_h, c_plasma_mg_L, cmax, auc, t_half.
    """
    if child_pugh_classes is None:
        child_pugh_classes = ["A", "B", "C"]

    results = {}

    # Normal (no hepatic disease)
    results["normal"] = _simulate_1cpt_oral(
        dose_mg=dose_mg,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        ka_per_h=ka_per_h,
        f_oral=f_oral,
    )

    # Hepatic disease classes
    cl_factor_map = {"A": 0.75, "B": 0.50, "C": 0.25}
    vd_factor_map = {"A": 1.0, "B": 1.1, "C": 1.2}

    for cp_class in child_pugh_classes:
        if cp_class not in {"A", "B", "C"}:
            raise ValueError(f"Invalid Child-Pugh class: '{cp_class}'")
        cl_adj = cl_L_per_h * cl_factor_map[cp_class]
        vd_adj = vd_L * vd_factor_map[cp_class]
        results[cp_class] = _simulate_1cpt_oral(
            dose_mg=dose_mg,
            cl_L_per_h=cl_adj,
            vd_L=vd_adj,
            ka_per_h=ka_per_h,
            f_oral=f_oral,
        )

    return results
