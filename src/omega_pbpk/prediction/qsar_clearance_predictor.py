"""
Phase 563 — QSAR-based systemic clearance prediction.

Pure Python, no numpy/scipy.
"""

import math


def predict_cl_from_properties(
    logP: float,
    mw: float,
    psa: float,
    fu_plasma: float,
    n_hbd: int = 0,
    cyp3a4_substrate: bool = False,
) -> dict:
    """Empirical QSAR model for total systemic clearance in L/h (typical human).

    Parameters
    ----------
    logP : octanol-water partition coefficient
    mw : molecular weight (Da)
    psa : polar surface area (A^2)
    fu_plasma : unbound fraction in plasma (0-1)
    n_hbd : number of hydrogen bond donors
    cyp3a4_substrate : if True, apply 1.5x CYP3A4 metabolic correction

    Returns
    -------
    dict with keys: cl_predicted_L_per_h, cl_category, hepatic_fraction_estimate, notes
    """
    if fu_plasma < 0 or fu_plasma > 1:
        raise ValueError("fu_plasma must be between 0 and 1")
    if mw <= 0:
        raise ValueError("mw must be positive")

    # Base clearance driven by unbound fraction and PSA
    base_cl = 5.0 * (fu_plasma**0.7) * math.exp(-0.1 * psa / 100.0)

    # logP correction
    if logP > 3:
        base_cl *= 1.2
    elif logP < 0:
        base_cl *= 0.7

    # MW correction
    base_cl *= (500.0 / mw) ** 0.3

    # CYP3A4 substrate correction
    if cyp3a4_substrate:
        base_cl *= 1.5

    cl_predicted = base_cl

    # Category
    if cl_predicted < 5.0:
        cl_category = "low"
    elif cl_predicted <= 15.0:
        cl_category = "moderate"
    else:
        cl_category = "high"

    # Hepatic fraction estimate: higher logP -> more hepatic
    # Simple linear: fraction = 0.5 + 0.1*logP clamped to [0.3, 0.95]
    hepatic_fraction = 0.5 + 0.1 * logP
    hepatic_fraction = max(0.3, min(0.95, hepatic_fraction))

    notes = (
        f"Empirical QSAR model. logP={logP}, MW={mw}, PSA={psa}, "
        f"fu={fu_plasma}, CYP3A4_substrate={cyp3a4_substrate}."
    )

    return {
        "cl_predicted_L_per_h": cl_predicted,
        "cl_category": cl_category,
        "hepatic_fraction_estimate": hepatic_fraction,
        "notes": notes,
    }


def predict_renal_cl(
    fu_plasma: float,
    gfr_mL_min: float = 120.0,
    reabsorption_fraction: float = 0.5,
) -> dict:
    """Predict renal clearance.

    CL_renal = fu * GFR * (1 - reabsorption_fraction)
    GFR in mL/min converted to L/h by multiplying by 0.06.

    Returns
    -------
    dict with keys: cl_renal_L_per_h, fe_renal_fraction, notes
    """
    if fu_plasma < 0 or fu_plasma > 1:
        raise ValueError("fu_plasma must be between 0 and 1")
    if gfr_mL_min < 0:
        raise ValueError("gfr_mL_min must be non-negative")
    if reabsorption_fraction < 0 or reabsorption_fraction > 1:
        raise ValueError("reabsorption_fraction must be between 0 and 1")

    gfr_L_per_h = gfr_mL_min * 0.06
    cl_renal = fu_plasma * gfr_L_per_h * (1.0 - reabsorption_fraction)

    # fe_renal as fraction of filtered drug that is excreted
    fe_renal = 1.0 - reabsorption_fraction

    notes = (
        f"CL_renal = fu ({fu_plasma}) * GFR ({gfr_mL_min} mL/min = "
        f"{gfr_L_per_h:.3f} L/h) * (1 - reabsorption {reabsorption_fraction})."
    )

    return {
        "cl_renal_L_per_h": cl_renal,
        "fe_renal_fraction": fe_renal,
        "notes": notes,
    }


def predict_cl_ckd(
    cl_normal_L_per_h: float,
    egfr_mL_min: float,
    renal_fraction: float,
) -> dict:
    """Predict clearance adjusted for CKD.

    CL_CKD = cl_normal * (1 - renal_fraction)
              + cl_normal * renal_fraction * (eGFR / 120)

    Parameters
    ----------
    cl_normal_L_per_h : total CL in normal renal function
    egfr_mL_min : estimated GFR (mL/min/1.73m^2)
    renal_fraction : fraction of total CL that is renal

    Returns
    -------
    dict with keys: cl_ckd_L_per_h, cl_ratio_vs_normal, dose_adjustment_factor
    """
    if cl_normal_L_per_h <= 0:
        raise ValueError("cl_normal_L_per_h must be positive")
    if egfr_mL_min < 0:
        raise ValueError("egfr_mL_min must be non-negative")
    if renal_fraction < 0 or renal_fraction > 1:
        raise ValueError("renal_fraction must be between 0 and 1")

    cl_ckd = cl_normal_L_per_h * (1.0 - renal_fraction) + cl_normal_L_per_h * renal_fraction * (
        egfr_mL_min / 120.0
    )

    cl_ratio = cl_ckd / cl_normal_L_per_h

    # Dose adjustment: to achieve same exposure, reduce dose proportionally to CL ratio
    dose_adjustment_factor = cl_ratio  # lower CL -> lower dose needed

    return {
        "cl_ckd_L_per_h": cl_ckd,
        "cl_ratio_vs_normal": cl_ratio,
        "dose_adjustment_factor": dose_adjustment_factor,
    }


def vd_from_cl_thalf(cl_L_per_h: float, t_half_h: float) -> float:
    """Calculate volume of distribution from CL and half-life.

    Vd = CL * t_half / ln(2)

    Parameters
    ----------
    cl_L_per_h : clearance in L/h
    t_half_h : elimination half-life in hours

    Returns
    -------
    float : volume of distribution in L
    """
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be positive")
    if t_half_h <= 0:
        raise ValueError("t_half_h must be positive")

    return cl_L_per_h * t_half_h / math.log(2.0)


def clearance_sensitivity(
    fu_range: list[float],
    mw: float,
    psa: float,
    logP: float,
) -> list[dict]:
    """Sensitivity analysis: predict CL across a range of fu_plasma values.

    Parameters
    ----------
    fu_range : list of fu_plasma values to evaluate
    mw : molecular weight (Da)
    psa : polar surface area (A^2)
    logP : octanol-water partition coefficient

    Returns
    -------
    list of dicts, each with keys: fu_plasma, cl_predicted_L_per_h
    """
    results = []
    for fu in fu_range:
        pred = predict_cl_from_properties(logP=logP, mw=mw, psa=psa, fu_plasma=fu)
        results.append(
            {
                "fu_plasma": fu,
                "cl_predicted_L_per_h": pred["cl_predicted_L_per_h"],
            }
        )
    return results


__all__ = [
    "predict_cl_from_properties",
    "predict_renal_cl",
    "predict_cl_ckd",
    "vd_from_cl_thalf",
    "clearance_sensitivity",
]
