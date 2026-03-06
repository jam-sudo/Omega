"""Hepatic and gut-wall first-pass extraction module."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "FirstPassResult",
    "calculate_first_pass",
    "first_pass_sensitivity",
]


@dataclass
class FirstPassResult:
    drug_name: str
    route: str
    dose_mg: float
    fup: float
    cl_int_L_per_h: float
    f_gut: float
    f_hepatic: float
    f_oral: float
    e_hepatic: float
    q_hepatic_L_per_h: float
    well_stirred_model: str
    peak_portal_conc_mg_L: float
    hepatic_cl_L_per_h: float


def _hepatic_extraction(
    cl_int_L_per_h: float,
    q_liver_L_per_h: float,
    fup: float,
) -> float:
    """Well-stirred model: E = fup*CLint / (Q + fup*CLint), clamped [0, 0.999]."""
    numerator = fup * cl_int_L_per_h
    denominator = q_liver_L_per_h + fup * cl_int_L_per_h
    if denominator == 0.0:
        return 0.0
    e = numerator / denominator
    return max(0.0, min(0.999, e))


def _gut_availability(
    fa: float,
    peff_cm_s: float = 1e-4,
    logP: float = 2.0,
    fg_base: float = 0.8,
) -> float:
    """Gut-wall availability: lipophilic substrates have less gut extraction."""
    fg = fg_base * (1 - max(0.0, logP - 4) * 0.1)
    return max(0.1, min(1.0, fg))


def calculate_first_pass(
    drug_name: str,
    dose_mg: float,
    fup: float,
    cl_int_L_per_h: float,
    fa: float = 1.0,
    logP: float = 2.0,
    q_hepatic_L_per_h: float = 90.0,
    route: str = "oral",
) -> FirstPassResult:
    """Calculate hepatic and gut-wall first-pass extraction.

    Parameters
    ----------
    drug_name: name of the drug
    dose_mg: dose in mg (must be > 0)
    fup: fraction unbound in plasma (must be > 0)
    cl_int_L_per_h: intrinsic clearance in L/h (must be >= 0)
    fa: fraction absorbed from gut lumen (0-1)
    logP: octanol-water partition coefficient
    q_hepatic_L_per_h: hepatic blood flow in L/h (default 90 L/h)
    route: 'oral' or 'iv'

    Returns
    -------
    FirstPassResult dataclass

    Raises
    ------
    ValueError: if dose_mg <= 0, fup <= 0, or cl_int_L_per_h < 0
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if fup <= 0:
        raise ValueError(f"fup must be > 0, got {fup}")
    if cl_int_L_per_h < 0:
        raise ValueError(f"cl_int_L_per_h must be >= 0, got {cl_int_L_per_h}")

    e = _hepatic_extraction(cl_int_L_per_h, q_hepatic_L_per_h, fup)
    fh = 1.0 - e
    fg = _gut_availability(fa, logP=logP)

    if route == "oral":
        f_oral = fa * fg * fh
    else:
        f_oral = 1.0

    peak_portal_conc = dose_mg * fa / q_hepatic_L_per_h
    hepatic_cl = e * q_hepatic_L_per_h

    return FirstPassResult(
        drug_name=drug_name,
        route=route,
        dose_mg=dose_mg,
        fup=fup,
        cl_int_L_per_h=cl_int_L_per_h,
        f_gut=fg,
        f_hepatic=fh,
        f_oral=f_oral,
        e_hepatic=e,
        q_hepatic_L_per_h=q_hepatic_L_per_h,
        well_stirred_model="well_stirred",
        peak_portal_conc_mg_L=peak_portal_conc,
        hepatic_cl_L_per_h=hepatic_cl,
    )


def first_pass_sensitivity(
    drug_name: str,
    dose_mg: float,
    fup: float,
    cl_int_range: list[float],
    q_hepatic: float = 90.0,
) -> list[FirstPassResult]:
    """Evaluate first-pass extraction at each CLint value.

    Parameters
    ----------
    drug_name: name of the drug
    dose_mg: dose in mg
    fup: fraction unbound in plasma
    cl_int_range: list of intrinsic clearance values (L/h)
    q_hepatic: hepatic blood flow in L/h (default 90 L/h)

    Returns
    -------
    List of FirstPassResult, one per CLint value
    """
    return [
        calculate_first_pass(
            drug_name=drug_name,
            dose_mg=dose_mg,
            fup=fup,
            cl_int_L_per_h=cl_int,
            q_hepatic_L_per_h=q_hepatic,
        )
        for cl_int in cl_int_range
    ]
