"""Hepatic clearance scaling from in vitro to in vivo using three hepatic models."""

from __future__ import annotations

import math


def _scale_clint(
    clint_uL_min_mg: float,
    microsomal_protein_g_liver: float = 39.0,
) -> float:
    """Scale CLint from µL/min/mg to whole-liver L/h."""
    # clint_uL_min_mg * 1e-6 L/µL * 60 min/h * microsomal_protein_g_liver * 1000 mg/g
    return clint_uL_min_mg * 1e-6 * 60.0 * microsomal_protein_g_liver * 1000.0


def _validate_inputs(fu_plasma: float, rbp: float) -> None:
    if not (0.0 < fu_plasma <= 1.0):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if rbp <= 0.0:
        raise ValueError(f"rbp must be > 0, got {rbp}")


def well_stirred_model(
    clint_uL_min_mg: float,
    qh_L_per_h: float = 90.0,
    fu_plasma: float = 1.0,
    rbp: float = 1.0,
    microsomal_protein_g_liver: float = 39.0,
) -> dict:
    """Scale CLint using the well-stirred (venous equilibrium) hepatic model.

    Args:
        clint_uL_min_mg: Intrinsic clearance in µL/min/mg microsomal protein
        qh_L_per_h: Hepatic blood flow in L/h (default 90 L/h for 70 kg human)
        fu_plasma: Fraction unbound in plasma (0, 1]
        rbp: Red blood cell to plasma concentration ratio (blood/plasma)
        microsomal_protein_g_liver: Microsomal protein content g/liver (default 39 g)

    Returns:
        dict with clint_whole_liver_L_per_h, clh_L_per_h, extraction_ratio,
        f_available, qh_L_per_h
    """
    _validate_inputs(fu_plasma, rbp)

    clint_total = _scale_clint(clint_uL_min_mg, microsomal_protein_g_liver)
    fu_b = fu_plasma / rbp  # fraction unbound in blood

    # Well-stirred model: CLh = Qh * fu_b * CLint / (Qh + fu_b * CLint)
    numerator = qh_L_per_h * fu_b * clint_total
    denominator = qh_L_per_h + fu_b * clint_total
    clh = numerator / denominator if denominator > 0 else 0.0

    extraction_ratio = clh / qh_L_per_h if qh_L_per_h > 0 else 0.0
    f_available = 1.0 - extraction_ratio

    return {
        "clint_whole_liver_L_per_h": clint_total,
        "clh_L_per_h": clh,
        "extraction_ratio": extraction_ratio,
        "f_available": f_available,
        "qh_L_per_h": qh_L_per_h,
    }


def parallel_tube_model(
    clint_uL_min_mg: float,
    qh_L_per_h: float = 90.0,
    fu_plasma: float = 1.0,
    rbp: float = 1.0,
    microsomal_protein_g_liver: float = 39.0,
) -> dict:
    """Scale CLint using the parallel-tube (sinusoidal perfusion) hepatic model.

    Args:
        clint_uL_min_mg: Intrinsic clearance in µL/min/mg microsomal protein
        qh_L_per_h: Hepatic blood flow in L/h
        fu_plasma: Fraction unbound in plasma (0, 1]
        rbp: Blood-to-plasma concentration ratio
        microsomal_protein_g_liver: Microsomal protein content g/liver

    Returns:
        dict with clint_whole_liver_L_per_h, clh_L_per_h, extraction_ratio,
        f_available, qh_L_per_h
    """
    _validate_inputs(fu_plasma, rbp)

    clint_total = _scale_clint(clint_uL_min_mg, microsomal_protein_g_liver)
    fu_b = fu_plasma / rbp

    # Parallel-tube: CLh = Qh * (1 - exp(-fu_b * CLint / Qh))
    if qh_L_per_h > 0:
        exponent = -fu_b * clint_total / qh_L_per_h
        # Cap exponent to avoid underflow
        exponent = max(exponent, -500.0)
        clh = qh_L_per_h * (1.0 - math.exp(exponent))
    else:
        clh = 0.0

    extraction_ratio = clh / qh_L_per_h if qh_L_per_h > 0 else 0.0
    f_available = 1.0 - extraction_ratio

    return {
        "clint_whole_liver_L_per_h": clint_total,
        "clh_L_per_h": clh,
        "extraction_ratio": extraction_ratio,
        "f_available": f_available,
        "qh_L_per_h": qh_L_per_h,
    }


def dispersion_model(
    clint_uL_min_mg: float,
    qh_L_per_h: float = 90.0,
    fu_plasma: float = 1.0,
    rbp: float = 1.0,
    microsomal_protein_g_liver: float = 39.0,
    dn: float = 0.17,
) -> dict:
    """Scale CLint using the dispersion hepatic model.

    Args:
        clint_uL_min_mg: Intrinsic clearance in µL/min/mg microsomal protein
        qh_L_per_h: Hepatic blood flow in L/h
        fu_plasma: Fraction unbound in plasma (0, 1]
        rbp: Blood-to-plasma concentration ratio
        microsomal_protein_g_liver: Microsomal protein content g/liver
        dn: Dispersion number (default 0.17 for liver)

    Returns:
        dict with clint_whole_liver_L_per_h, clh_L_per_h, extraction_ratio,
        f_available, qh_L_per_h
    """
    _validate_inputs(fu_plasma, rbp)

    clint_total = _scale_clint(clint_uL_min_mg, microsomal_protein_g_liver)
    fu_b = fu_plasma / rbp

    if qh_L_per_h <= 0 or dn <= 0:
        clh = 0.0
    else:
        r = fu_b * clint_total / qh_L_per_h  # dimensionless clearance
        a = math.sqrt(1.0 + 4.0 * r * dn)

        # Dispersion model extraction ratio:
        # E = 1 - 4*a*exp(1/(2*Dn)) / ((1+a)^2*exp(a/(2*Dn)) - (1-a)^2*exp(-a/(2*Dn)))
        # Cap exponents to avoid overflow
        exp_half_dn = math.exp(min(0.5 / dn, 500.0))
        exp_a_half_dn = math.exp(min(a / (2.0 * dn), 500.0))
        exp_neg_a_half_dn = math.exp(max(-a / (2.0 * dn), -500.0))

        numerator = 4.0 * a * exp_half_dn
        denominator = (1.0 + a) ** 2 * exp_a_half_dn - (1.0 - a) ** 2 * exp_neg_a_half_dn

        if abs(denominator) < 1e-15:
            # Linear approximation fallback for near-zero denominator
            extraction_ratio = min(r / (1.0 + r), 1.0)
        else:
            extraction_ratio = 1.0 - numerator / denominator
            # Clamp to physically valid range
            extraction_ratio = max(0.0, min(1.0, extraction_ratio))

        clh = qh_L_per_h * extraction_ratio

    extraction_ratio_final = clh / qh_L_per_h if qh_L_per_h > 0 else 0.0
    f_available = 1.0 - extraction_ratio_final

    return {
        "clint_whole_liver_L_per_h": clint_total,
        "clh_L_per_h": clh,
        "extraction_ratio": extraction_ratio_final,
        "f_available": f_available,
        "qh_L_per_h": qh_L_per_h,
    }


def compare_hepatic_models(
    clint_uL_min_mg: float,
    qh_L_per_h: float = 90.0,
    fu_plasma: float = 1.0,
    rbp: float = 1.0,
) -> dict:
    """Run all three hepatic models and compare CLh predictions.

    Args:
        clint_uL_min_mg: Intrinsic clearance in µL/min/mg microsomal protein
        qh_L_per_h: Hepatic blood flow in L/h
        fu_plasma: Fraction unbound in plasma (0, 1]
        rbp: Blood-to-plasma concentration ratio

    Returns:
        dict with 'well_stirred', 'parallel_tube', 'dispersion' results,
        and 'clh_range' (min, max) tuple across models
    """
    ws = well_stirred_model(clint_uL_min_mg, qh_L_per_h, fu_plasma, rbp)
    pt = parallel_tube_model(clint_uL_min_mg, qh_L_per_h, fu_plasma, rbp)
    disp = dispersion_model(clint_uL_min_mg, qh_L_per_h, fu_plasma, rbp)

    clh_values = [ws["clh_L_per_h"], pt["clh_L_per_h"], disp["clh_L_per_h"]]

    return {
        "well_stirred": ws,
        "parallel_tube": pt,
        "dispersion": disp,
        "clh_range": (min(clh_values), max(clh_values)),
    }


def child_pugh_hepatic_adjustment(
    clh_normal_L_per_h: float,
    child_pugh_score: int,
) -> dict:
    """Adjust hepatic clearance based on Child-Pugh severity score.

    Args:
        clh_normal_L_per_h: Normal hepatic clearance in L/h
        child_pugh_score: Child-Pugh score (5-15; A=5-6, B=7-9, C=10-15)

    Returns:
        dict with class_label, cl_adjusted_L_per_h, dose_factor, recommendation
    """
    if not (5 <= child_pugh_score <= 15):
        raise ValueError(f"child_pugh_score must be 5-15, got {child_pugh_score}")
    if clh_normal_L_per_h < 0:
        raise ValueError(f"clh_normal_L_per_h must be >= 0, got {clh_normal_L_per_h}")

    if child_pugh_score <= 6:
        class_label = "A"
        cl_factor = 0.8
        recommendation = (
            "Mild hepatic impairment: consider 20% dose reduction or standard dose with monitoring"
        )
    elif child_pugh_score <= 9:
        class_label = "B"
        cl_factor = 0.5
        recommendation = (
            "Moderate hepatic impairment: 50% dose reduction recommended, close monitoring required"
        )
    else:
        class_label = "C"
        cl_factor = 0.25
        recommendation = "Severe hepatic impairment: 75% dose reduction or contraindication, expert consultation required"

    cl_adjusted = clh_normal_L_per_h * cl_factor
    # dose_factor is the factor to apply to the normal dose (inversely related to CL reduction
    # for high ER drugs, but simplified here as the CL scaling factor)
    dose_factor = cl_factor

    return {
        "class_label": class_label,
        "cl_adjusted_L_per_h": cl_adjusted,
        "dose_factor": dose_factor,
        "recommendation": recommendation,
    }
