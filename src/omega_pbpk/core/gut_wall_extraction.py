"""Gut Wall Extraction Model for oral bioavailability.

Models intestinal (gut wall) first-pass extraction using:
- Well-stirred gut model for metabolic extraction (EG)
- Sigmoid approximation for absorption fraction (fa) from effective permeability
- Combined fa*fg oral bioavailability contribution

Reference: Sun D et al., J Pharm Sci. 2004;93(6):1535-50.
           Yu LX, Amidon GL. Int J Pharm. 1999;186(2):119-25.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GutExtractionResult:
    """Result of gut wall extraction calculation."""

    peff_cm_s: float
    fg: float          # gut availability (1 - EG)
    eg: float          # gut extraction ratio
    fa: float          # absorption fraction (from permeability)
    f_combined: float  # fa * fg
    qgut_L_per_h: float
    notes: str


def gut_wall_extraction(
    peff_cm_s: float,
    fg_factor: float | None = None,
    clint_gut_L_per_h: float | None = None,
    qgut_L_per_h: float = 18.0,
    fu_gut: float = 1.0,
) -> GutExtractionResult:
    """Compute gut wall extraction and absorption fraction.

    Parameters
    ----------
    peff_cm_s : float
        Effective intestinal permeability (cm/s). Used for sigmoid fa estimation.
    fg_factor : float | None
        Gut availability (fg = 1 - EG) provided directly. If given, overrides
        CLint-based calculation.
    clint_gut_L_per_h : float | None
        Intrinsic gut wall clearance (L/h). Used in well-stirred model when
        fg_factor is None.
    qgut_L_per_h : float
        Gut blood flow rate (L/h). Default 18 L/h (typical human villous blood flow).
    fu_gut: float
        Fraction unbound in gut tissue. Default 1.0.

    Returns
    -------
    GutExtractionResult
        Extraction parameters including fg, eg, fa, and f_combined (fa*fg).

    Raises
    ------
    ValueError
        If any required parameter is out of valid range.
    """
    if peff_cm_s <= 0:
        raise ValueError("peff_cm_s must be > 0")
    if qgut_L_per_h <= 0:
        raise ValueError("qgut_L_per_h must be > 0")
    if not (0.0 < fu_gut <= 1.0):
        raise ValueError("fu_gut must be in (0, 1]")
    if clint_gut_L_per_h is not None and clint_gut_L_per_h < 0:
        raise ValueError("clint_gut_L_per_h must be >= 0")

    # --- Compute fa from effective permeability (sigmoid approximation) ---
    # Based on: fa = 1 / (1 + exp(-5 * (peff_cm_s*1e4 - 2)))
    # peff_cm_s * 1e4 converts to units of 10^-4 cm/s
    peff_scaled = peff_cm_s * 1.0e4  # convert to 10^-4 cm/s units
    fa = 1.0 / (1.0 + math.exp(-5.0 * (peff_scaled - 2.0)))
    fa = min(max(fa, 0.0), 1.0)

    # --- Compute fg (gut availability) ---
    notes_parts: list[str] = []

    if fg_factor is not None:
        if not (0.0 <= fg_factor <= 1.0):
            raise ValueError("fg_factor must be in [0, 1]")
        fg = float(fg_factor)
        eg = 1.0 - fg
        notes_parts.append("fg provided directly")
    elif clint_gut_L_per_h is not None:
        # Well-stirred gut model: EG = CLint_gut * fu_gut / (Qgut + CLint_gut * fu_gut)
        clint_eff = clint_gut_L_per_h * fu_gut
        denom = qgut_L_per_h + clint_eff
        eg = clint_eff / denom if denom > 0 else 0.0
        fg = 1.0 - eg
        notes_parts.append(
            f"Well-stirred gut model: CLint_gut={clint_gut_L_per_h:.4f} L/h, "
            f"fu_gut={fu_gut:.4f}, EG={eg:.4f}"
        )
    else:
        # No gut metabolism: fg = 1, EG = 0
        fg = 1.0
        eg = 0.0
        notes_parts.append("No gut metabolism (CLint_gut not provided)")

    f_combined = fa * fg

    peff_note = f"Peff={peff_cm_s:.2e} cm/s -> fa={fa:.4f} (sigmoid approx)"
    notes_parts.append(peff_note)
    notes_parts.append(f"f_combined (fa*fg)={f_combined:.4f}")

    return GutExtractionResult(
        peff_cm_s=peff_cm_s,
        fg=fg,
        eg=eg,
        fa=fa,
        f_combined=f_combined,
        qgut_L_per_h=qgut_L_per_h,
        notes="; ".join(notes_parts),
    )


def predict_oral_f(
    fa: float,
    fg: float,
    fh: float,
) -> dict[str, float | str]:
    """Compute oral bioavailability F = fa * fg * fh.

    Parameters
    ----------
    fa : float
        Fraction absorbed (0-1).
    fg : float
        Gut availability (0-1), fraction escaping gut wall extraction.
    fh : float
        Hepatic availability (0-1), fraction escaping hepatic first-pass.

    Returns
    -------
    dict
        Contains F, fa, fg, fh, and qualitative notes.
    """
    fa = float(min(max(fa, 0.0), 1.0))
    fg = float(min(max(fg, 0.0), 1.0))
    fh = float(min(max(fh, 0.0), 1.0))

    F = fa * fg * fh

    # Qualitative classification
    if F >= 0.80:
        bioavailability_class = "High (>=80%)"
    elif F >= 0.50:
        bioavailability_class = "Moderate (50-80%)"
    elif F >= 0.20:
        bioavailability_class = "Low (20-50%)"
    else:
        bioavailability_class = "Very low (<20%)"

    limiting_step = "none"
    steps = {"absorption (fa)": fa, "gut wall (fg)": fg, "hepatic (fh)": fh}
    min_step = min(steps, key=lambda k: steps[k])
    if steps[min_step] < 0.5:
        limiting_step = min_step

    return {
        "F": F,
        "fa": fa,
        "fg": fg,
        "fh": fh,
        "bioavailability_class": bioavailability_class,
        "limiting_step": limiting_step,
        "notes": (
            f"F = fa ({fa:.3f}) x fg ({fg:.3f}) x fh ({fh:.3f}) = {F:.4f} "
            f"({100*F:.1f}%); {bioavailability_class}"
        ),
    }


def sensitivity_analysis(
    peff_range: list[float],
    fg_range: list[float],
    fh_range: list[float],
) -> list[dict[str, float]]:
    """Compute a grid of oral bioavailability F values over parameter ranges.

    Parameters
    ----------
    peff_range : list[float]
        Effective permeability values (cm/s) for fa calculation.
    fg_range : list[float]
        Gut availability values (0-1) to evaluate.
    fh_range : list[float]
        Hepatic availability values (0-1) to evaluate.

    Returns
    -------
    list[dict]
        Each entry: peff_cm_s, fa, fg, fh, F
    """
    rows: list[dict[str, float]] = []
    for peff in peff_range:
        peff_scaled = peff * 1.0e4
        fa = 1.0 / (1.0 + math.exp(-5.0 * (peff_scaled - 2.0)))
        fa = min(max(fa, 0.0), 1.0)
        for fg in fg_range:
            fg_c = min(max(float(fg), 0.0), 1.0)
            for fh in fh_range:
                fh_c = min(max(float(fh), 0.0), 1.0)
                rows.append(
                    {
                        "peff_cm_s": float(peff),
                        "fa": fa,
                        "fg": fg_c,
                        "fh": fh_c,
                        "F": fa * fg_c * fh_c,
                    }
                )
    return rows


__all__ = [
    "GutExtractionResult",
    "gut_wall_extraction",
    "predict_oral_f",
    "sensitivity_analysis",
]
