"""
Hepatic and renal extraction ratio calculations.

Uses well-stirred (venous equilibration) model for hepatic clearance and
simple ratio model for renal clearance.
"""


def _classify_er(er: float) -> str:
    """Classify extraction ratio as low, intermediate, or high."""
    if er < 0.3:
        return "low"
    elif er <= 0.7:
        return "intermediate"
    else:
        return "high"


def hepatic_extraction_ratio(
    cl_int_L_per_h: float,
    q_liver_L_per_h: float = 90.0,
    fu_plasma: float = 1.0,
) -> dict:
    """
    Compute hepatic extraction ratio using the well-stirred model.

    ER = (fu * CLint) / (Q + fu * CLint)
    CL_hepatic = Q * ER

    Parameters
    ----------
    cl_int_L_per_h : float
        Intrinsic hepatic clearance (L/h).
    q_liver_L_per_h : float
        Hepatic blood flow (L/h), default 90 L/h.
    fu_plasma : float
        Unbound fraction in plasma (0 < fu <= 1), default 1.0.

    Returns
    -------
    dict with keys:
        er, cl_hepatic, cl_intrinsic_scaled, hepatic_class, f_oral_estimate
    """
    if cl_int_L_per_h <= 0:
        raise ValueError(f"cl_int_L_per_h must be positive, got {cl_int_L_per_h}.")
    if q_liver_L_per_h <= 0:
        raise ValueError(f"q_liver_L_per_h must be positive, got {q_liver_L_per_h}.")
    if not (0 < fu_plasma <= 1.0):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}.")

    fu_cl_int = fu_plasma * cl_int_L_per_h
    er = fu_cl_int / (q_liver_L_per_h + fu_cl_int)
    cl_hepatic = q_liver_L_per_h * er

    return {
        "er": er,
        "cl_hepatic": cl_hepatic,
        "cl_intrinsic_scaled": cl_int_L_per_h,
        "hepatic_class": _classify_er(er),
        "f_oral_estimate": 1.0 - er,
    }


def renal_extraction_ratio(
    cl_renal_L_per_h: float,
    q_renal_L_per_h: float = 72.0,
) -> dict:
    """
    Compute renal extraction ratio.

    ER_renal = CL_renal / Q_renal, capped at 1.0.

    Parameters
    ----------
    cl_renal_L_per_h : float
        Renal clearance (L/h).
    q_renal_L_per_h : float
        Renal blood flow (L/h), default 72 L/h.

    Returns
    -------
    dict with keys: er, cl_renal, renal_class
    """
    if cl_renal_L_per_h < 0:
        raise ValueError(f"cl_renal_L_per_h must be non-negative, got {cl_renal_L_per_h}.")
    if q_renal_L_per_h <= 0:
        raise ValueError(f"q_renal_L_per_h must be positive, got {q_renal_L_per_h}.")

    er = min(cl_renal_L_per_h / q_renal_L_per_h, 1.0)

    return {
        "er": er,
        "cl_renal": cl_renal_L_per_h,
        "renal_class": _classify_er(er),
    }


def organ_clearance_budget(
    cl_hepatic: float,
    cl_renal: float,
    cl_other: float = 0.0,
) -> dict:
    """
    Compute the fractional contribution of each elimination route to total clearance.

    Parameters
    ----------
    cl_hepatic : float
        Hepatic clearance (L/h).
    cl_renal : float
        Renal clearance (L/h).
    cl_other : float
        Other clearance pathways (L/h), default 0.0.

    Returns
    -------
    dict with keys: cl_total, frac_hepatic, frac_renal, frac_other, dominant_organ
    """
    if cl_hepatic < 0:
        raise ValueError(f"cl_hepatic must be non-negative, got {cl_hepatic}.")
    if cl_renal < 0:
        raise ValueError(f"cl_renal must be non-negative, got {cl_renal}.")
    if cl_other < 0:
        raise ValueError(f"cl_other must be non-negative, got {cl_other}.")

    cl_total = cl_hepatic + cl_renal + cl_other

    if cl_total == 0:
        frac_hepatic = 0.0
        frac_renal = 0.0
        frac_other = 0.0
        dominant_organ = "none"
    else:
        frac_hepatic = cl_hepatic / cl_total
        frac_renal = cl_renal / cl_total
        frac_other = cl_other / cl_total

        fracs = {
            "hepatic": frac_hepatic,
            "renal": frac_renal,
            "other": frac_other,
        }
        dominant_organ = max(fracs, key=fracs.get)

    return {
        "cl_total": cl_total,
        "frac_hepatic": frac_hepatic,
        "frac_renal": frac_renal,
        "frac_other": frac_other,
        "dominant_organ": dominant_organ,
    }


def first_pass_oral_bioavailability(
    er_gut: float,
    er_hepatic: float,
) -> float:
    """
    Compute oral bioavailability accounting for gut and hepatic first-pass extraction.

    F = (1 - er_gut) * (1 - er_hepatic)

    Parameters
    ----------
    er_gut : float
        Gut wall extraction ratio (0 to 1).
    er_hepatic : float
        Hepatic extraction ratio (0 to 1).

    Returns
    -------
    float : bioavailability F in [0, 1]
    """
    if not (0.0 <= er_gut <= 1.0):
        raise ValueError(f"er_gut must be in [0, 1], got {er_gut}.")
    if not (0.0 <= er_hepatic <= 1.0):
        raise ValueError(f"er_hepatic must be in [0, 1], got {er_hepatic}.")

    return (1.0 - er_gut) * (1.0 - er_hepatic)


def sensitivity_to_blood_flow(
    cl_int_L_per_h: float,
    fu_plasma: float,
    q_range: list,
) -> list:
    """
    Compute hepatic ER and clearance across a range of blood flow values.

    Parameters
    ----------
    cl_int_L_per_h : float
        Intrinsic hepatic clearance (L/h).
    fu_plasma : float
        Unbound fraction in plasma (0 < fu <= 1).
    q_range : list of float
        List of hepatic blood flow values (L/h) to evaluate.

    Returns
    -------
    list of dicts, each with keys: q, er, cl_hepatic
    """
    results = []
    for q in q_range:
        info = hepatic_extraction_ratio(cl_int_L_per_h, q_liver_L_per_h=q, fu_plasma=fu_plasma)
        results.append(
            {
                "q": q,
                "er": info["er"],
                "cl_hepatic": info["cl_hepatic"],
            }
        )
    return results
