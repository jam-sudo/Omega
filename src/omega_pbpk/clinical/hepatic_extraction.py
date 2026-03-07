"""Hepatic drug extraction and first-pass effect calculator — Phase 472."""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_QH_ML_PER_MIN = 1500.0  # hepatic blood flow (mL/min)
_DEFAULT_DN = 0.17  # dispersion number (Weiss 1999 typical value)

_HIGH_EXTRACTION_THRESHOLD = 0.70
_LOW_EXTRACTION_THRESHOLD = 0.30


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HepaticExtractionResult:
    """Comparison of hepatic extraction across three physiological models."""

    cl_int_mL_per_min: float
    fu_p: float
    qh_mL_per_min: float
    eh_well_stirred: float
    eh_parallel_tube: float
    eh_dispersion: float
    cl_h_well_stirred: float
    cl_h_parallel_tube: float
    cl_h_dispersion: float
    fg_well_stirred: float
    fg_parallel_tube: float
    fg_dispersion: float
    high_extraction: bool
    low_extraction: bool
    notes: str


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Individual model functions
# ---------------------------------------------------------------------------


def well_stirred_model(
    cl_int_mL_per_min: float,
    fu_p: float,
    qh_mL_per_min: float = _DEFAULT_QH_ML_PER_MIN,
    rb: float = 1.0,
) -> dict[str, float]:
    """Well-stirred (venous equilibrium) hepatic extraction model.

    CL_h = Qh * fu_p * CLint / (Qh + fu_p * CLint)
    Eh   = CL_h / Qh

    Parameters
    ----------
    cl_int_mL_per_min:
        Intrinsic clearance (mL/min), must be > 0.
    fu_p:
        Fraction unbound in plasma (0 < fu_p ≤ 1).
    qh_mL_per_min:
        Hepatic blood flow (mL/min), default 1500.
    rb:
        Blood-to-plasma concentration ratio (default 1.0).

    Returns
    -------
    dict with keys 'eh', 'cl_h_mL_per_min', 'fg'.
    """
    if cl_int_mL_per_min <= 0:
        raise ValueError("cl_int_mL_per_min must be positive")
    if not (0 < fu_p <= 1.0):
        raise ValueError("fu_p must be in (0, 1]")
    if qh_mL_per_min <= 0:
        raise ValueError("qh_mL_per_min must be positive")
    if rb <= 0:
        raise ValueError("rb must be positive")

    # Adjust CLint for blood-to-plasma ratio
    cl_int_b = cl_int_mL_per_min / rb

    numerator = qh_mL_per_min * fu_p * cl_int_b
    denominator = qh_mL_per_min + fu_p * cl_int_b
    cl_h = numerator / denominator
    eh = _clamp(cl_h / qh_mL_per_min)
    fg = first_pass_effect(eh)
    return {"eh": eh, "cl_h_mL_per_min": cl_h, "fg": fg}


def parallel_tube_model(
    cl_int_mL_per_min: float,
    fu_p: float,
    qh_mL_per_min: float = _DEFAULT_QH_ML_PER_MIN,
    rb: float = 1.0,
) -> dict[str, float]:
    """Parallel-tube (sinusoidal) hepatic extraction model.

    Eh = 1 - exp(-fu_p * CLint / Qh)

    Parameters
    ----------
    cl_int_mL_per_min:
        Intrinsic clearance (mL/min).
    fu_p:
        Fraction unbound in plasma.
    qh_mL_per_min:
        Hepatic blood flow (mL/min), default 1500.
    rb:
        Blood-to-plasma ratio.

    Returns
    -------
    dict with keys 'eh', 'cl_h_mL_per_min', 'fg'.
    """
    if cl_int_mL_per_min <= 0:
        raise ValueError("cl_int_mL_per_min must be positive")
    if not (0 < fu_p <= 1.0):
        raise ValueError("fu_p must be in (0, 1]")
    if qh_mL_per_min <= 0:
        raise ValueError("qh_mL_per_min must be positive")
    if rb <= 0:
        raise ValueError("rb must be positive")

    cl_int_b = cl_int_mL_per_min / rb
    exponent = -(fu_p * cl_int_b) / qh_mL_per_min
    eh = _clamp(1.0 - math.exp(exponent))
    cl_h = eh * qh_mL_per_min
    fg = first_pass_effect(eh)
    return {"eh": eh, "cl_h_mL_per_min": cl_h, "fg": fg}


def dispersion_model(
    cl_int_mL_per_min: float,
    fu_p: float,
    qh_mL_per_min: float = _DEFAULT_QH_ML_PER_MIN,
    rb: float = 1.0,
    dn: float = _DEFAULT_DN,
) -> dict[str, float]:
    """Dispersion model for hepatic extraction (Roberts & Rowland 1986).

    Uses a = sqrt(1 + 4 * Eh_ws * Dn), then:

    Eh = 4*a*exp(0.5/Dn) /
         [ (1+a)^2 * exp(a/(2*Dn)) - (1-a)^2 * exp(-a/(2*Dn)) ]

    where Eh_ws is the well-stirred extraction ratio.

    Parameters
    ----------
    cl_int_mL_per_min:
        Intrinsic clearance (mL/min).
    fu_p:
        Fraction unbound in plasma.
    qh_mL_per_min:
        Hepatic blood flow (mL/min).
    rb:
        Blood-to-plasma ratio.
    dn:
        Dispersion number (dimensionless), default 0.17.

    Returns
    -------
    dict with keys 'eh', 'cl_h_mL_per_min', 'fg'.
    """
    if cl_int_mL_per_min <= 0:
        raise ValueError("cl_int_mL_per_min must be positive")
    if not (0 < fu_p <= 1.0):
        raise ValueError("fu_p must be in (0, 1]")
    if qh_mL_per_min <= 0:
        raise ValueError("qh_mL_per_min must be positive")
    if rb <= 0:
        raise ValueError("rb must be positive")
    if dn <= 0:
        raise ValueError("dn must be positive")

    cl_int_b = cl_int_mL_per_min / rb

    # Roberts & Rowland (1986) dispersion model.
    # R = fu_p * CLint / Qh  (dimensionless removal number)
    # a = sqrt(1 + 4 * R * Dn)
    # The formula computes F_out/F_in (the surviving fraction, i.e. 1 - Eh):
    #   fg_disp = 4*a*exp(0.5/Dn) /
    #             [(1+a)^2*exp(a/(2*Dn)) - (1-a)^2*exp(-a/(2*Dn))]
    R = fu_p * cl_int_b / qh_mL_per_min
    a = math.sqrt(1.0 + 4.0 * R * dn)
    half_inv_dn = 0.5 / dn

    try:
        exp_center = math.exp(half_inv_dn)
        exp_pos = math.exp(a * half_inv_dn)
        exp_neg = math.exp(-a * half_inv_dn)

        numerator = 4.0 * a * exp_center
        denominator = (1.0 + a) ** 2 * exp_pos - (1.0 - a) ** 2 * exp_neg

        if denominator <= 0 or not math.isfinite(denominator):
            # For extreme inputs fall back to well-stirred
            ws_fallback = well_stirred_model(cl_int_mL_per_min, fu_p, qh_mL_per_min, rb)
            eh = ws_fallback["eh"]
        else:
            # numerator/denominator = F_out = 1 - Eh
            fg_disp = numerator / denominator
            eh = _clamp(1.0 - fg_disp)
    except OverflowError:
        # Large R → high extraction
        eh = _clamp(1.0 - math.exp(-R))

    cl_h = eh * qh_mL_per_min
    fg = first_pass_effect(eh)
    return {"eh": eh, "cl_h_mL_per_min": cl_h, "fg": fg}


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def first_pass_effect(eh: float) -> float:
    """Calculate oral availability fraction after hepatic first-pass (fg = 1 - Eh).

    Parameters
    ----------
    eh:
        Hepatic extraction ratio (0–1).

    Returns
    -------
    float
        fg: fraction of drug surviving first-pass (0–1).
    """
    if not (0.0 <= eh <= 1.0):
        raise ValueError("eh must be in [0, 1]")
    return 1.0 - eh


# ---------------------------------------------------------------------------
# Comparison function
# ---------------------------------------------------------------------------


def compare_models(
    cl_int_mL_per_min: float,
    fu_p: float,
    qh_mL_per_min: float = _DEFAULT_QH_ML_PER_MIN,
) -> HepaticExtractionResult:
    """Compare hepatic extraction across well-stirred, parallel-tube, and dispersion models.

    Parameters
    ----------
    cl_int_mL_per_min:
        Intrinsic clearance (mL/min).
    fu_p:
        Fraction unbound in plasma.
    qh_mL_per_min:
        Hepatic blood flow (mL/min), default 1500.

    Returns
    -------
    HepaticExtractionResult
    """
    if cl_int_mL_per_min <= 0:
        raise ValueError("cl_int_mL_per_min must be positive")
    if not (0 < fu_p <= 1.0):
        raise ValueError("fu_p must be in (0, 1]")
    if qh_mL_per_min <= 0:
        raise ValueError("qh_mL_per_min must be positive")

    ws = well_stirred_model(cl_int_mL_per_min, fu_p, qh_mL_per_min)
    pt = parallel_tube_model(cl_int_mL_per_min, fu_p, qh_mL_per_min)
    dm = dispersion_model(cl_int_mL_per_min, fu_p, qh_mL_per_min)

    eh_ws = ws["eh"]
    eh_pt = pt["eh"]
    eh_dm = dm["eh"]

    high_extraction = eh_ws > _HIGH_EXTRACTION_THRESHOLD
    low_extraction = eh_ws < _LOW_EXTRACTION_THRESHOLD

    notes_parts: list[str] = []
    if high_extraction:
        notes_parts.append(
            "High-extraction drug: oral bioavailability flow-limited; sensitive to Qh changes"
        )
    elif low_extraction:
        notes_parts.append(
            "Low-extraction drug: oral bioavailability capacity-limited; sensitive to CLint/fu_p"
        )
    else:
        notes_parts.append("Intermediate-extraction drug")

    # Model comparison note
    delta_ws_pt = abs(eh_ws - eh_pt)
    if delta_ws_pt > 0.10:
        notes_parts.append(
            f"Well-stirred and parallel-tube Eh differ by {delta_ws_pt:.2f}"
            " — model selection important"
        )

    notes = "; ".join(notes_parts)

    return HepaticExtractionResult(
        cl_int_mL_per_min=cl_int_mL_per_min,
        fu_p=fu_p,
        qh_mL_per_min=qh_mL_per_min,
        eh_well_stirred=eh_ws,
        eh_parallel_tube=eh_pt,
        eh_dispersion=eh_dm,
        cl_h_well_stirred=ws["cl_h_mL_per_min"],
        cl_h_parallel_tube=pt["cl_h_mL_per_min"],
        cl_h_dispersion=dm["cl_h_mL_per_min"],
        fg_well_stirred=ws["fg"],
        fg_parallel_tube=pt["fg"],
        fg_dispersion=dm["fg"],
        high_extraction=high_extraction,
        low_extraction=low_extraction,
        notes=notes,
    )
