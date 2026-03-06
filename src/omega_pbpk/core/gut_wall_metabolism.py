"""Gut wall CYP3A4 metabolism effect on oral bioavailability.

Model:
  Eg  = CLg_int * fu_gut / (Qg + CLg_int * fu_gut)   [gut extraction ratio]
  Fg  = 1 - Eg                                         [gut wall availability]
  Eh  = CLh / (Qh + CLh)                               [hepatic ER, well-stirred]
  Fh  = 1 - Eh
  F   = Fa * Fg * Fh                                   [overall oral bioavailability]

Reference:
  Paine M.F. et al., J Pharmacol Exp Ther 1997;
  Yang J. et al., Drug Metab Dispos 2007.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "GutWallResult",
    "calculate_gut_wall_extraction",
    "bioavailability_with_gut_wall",
]


@dataclass(frozen=True)
class GutWallResult:
    """Result of gut wall first-pass extraction calculation."""

    drug_name: str
    fa: float                    # fraction absorbed (0–1)
    fg: float                    # gut wall availability (1 − Eg)
    fh: float                    # hepatic availability (1 − Eh)
    f_oral: float                # fa × fg × fh
    eg: float                    # gut extraction ratio
    eh: float                    # hepatic extraction ratio
    clg_int_mL_per_min: float
    qg_mL_per_min: float
    notes: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_gut_wall_extraction(
    drug_name: str,
    clg_int_mL_per_min: float,
    fu_gut: float = 0.5,
    qg_mL_per_min: float = 250.0,
) -> tuple[float, float]:
    """Calculate gut wall extraction ratio and availability.

    Parameters
    ----------
    drug_name : str
        Compound identifier (used for error messages only).
    clg_int_mL_per_min : float
        Gut wall intrinsic clearance (mL/min).  Must be >= 0.
    fu_gut : float
        Unbound fraction in gut wall tissue (0–1).
    qg_mL_per_min : float
        Gut blood flow in mL/min (default 250 mL/min).

    Returns
    -------
    (Eg, Fg) : tuple[float, float]
        Eg — gut extraction ratio (0–1).
        Fg — gut wall availability = 1 − Eg.

    Raises
    ------
    ValueError
        If clg_int_mL_per_min < 0, fu_gut outside [0, 1], or
        qg_mL_per_min <= 0.
    """
    if clg_int_mL_per_min < 0:
        raise ValueError(
            f"clg_int_mL_per_min must be >= 0 for '{drug_name}', "
            f"got {clg_int_mL_per_min}"
        )
    if not 0.0 <= fu_gut <= 1.0:
        raise ValueError(
            f"fu_gut must be in [0, 1] for '{drug_name}', got {fu_gut}"
        )
    if qg_mL_per_min <= 0:
        raise ValueError(
            f"qg_mL_per_min must be positive for '{drug_name}', "
            f"got {qg_mL_per_min}"
        )

    effective_cl = clg_int_mL_per_min * fu_gut
    eg = effective_cl / (qg_mL_per_min + effective_cl)
    fg = 1.0 - eg
    return eg, fg


def bioavailability_with_gut_wall(
    drug_name: str,
    fa: float,
    cl_hepatic_L_per_h: float,
    vd_L: float,
    clg_int_mL_per_min: float,
    fu_gut: float = 0.5,
    fu_plasma: float = 0.1,
    qh_L_per_h: float = 80.0,
    qg_mL_per_min: float = 250.0,
) -> GutWallResult:
    """Calculate oral bioavailability accounting for gut wall metabolism.

    Parameters
    ----------
    drug_name : str
        Compound identifier.
    fa : float
        Fraction absorbed from GI lumen (0–1).
    cl_hepatic_L_per_h : float
        Total hepatic clearance (L/h).  Must be >= 0.
    vd_L : float
        Volume of distribution (L).  Contextual; not used in calculation.
    clg_int_mL_per_min : float
        Gut wall intrinsic clearance (mL/min).  Must be >= 0.
    fu_gut : float
        Unbound fraction in gut wall (default 0.5).
    fu_plasma : float
        Unbound fraction in plasma (not used directly; included for
        completeness / future hepatic binding adjustment).
    qh_L_per_h : float
        Hepatic blood flow (L/h, default 80 L/h).
    qg_mL_per_min : float
        Gut blood flow (mL/min, default 250 mL/min).

    Returns
    -------
    GutWallResult

    Raises
    ------
    ValueError
        If fa is outside [0, 1], clg_int_mL_per_min < 0, or
        cl_hepatic_L_per_h < 0.
    """
    if not 0.0 <= fa <= 1.0:
        raise ValueError(
            f"fa must be in [0, 1] for '{drug_name}', got {fa}"
        )
    if clg_int_mL_per_min < 0:
        raise ValueError(
            f"clg_int_mL_per_min must be >= 0 for '{drug_name}', "
            f"got {clg_int_mL_per_min}"
        )
    if cl_hepatic_L_per_h < 0:
        raise ValueError(
            f"cl_hepatic_L_per_h must be >= 0 for '{drug_name}', "
            f"got {cl_hepatic_L_per_h}"
        )

    # Gut wall extraction
    eg, fg = calculate_gut_wall_extraction(
        drug_name, clg_int_mL_per_min, fu_gut, qg_mL_per_min
    )

    # Hepatic extraction — well-stirred model (convert units to mL/min)
    qh_mL_per_min = qh_L_per_h * 1000.0 / 60.0
    cl_hepatic_mL_per_min = cl_hepatic_L_per_h * 1000.0 / 60.0
    eh = cl_hepatic_mL_per_min / (qh_mL_per_min + cl_hepatic_mL_per_min)
    fh = 1.0 - eh

    f_oral = fa * fg * fh

    notes = (
        f"Eg={eg:.3f}, Fg={fg:.3f}; "
        f"Eh={eh:.3f}, Fh={fh:.3f}; "
        f"F_oral={f_oral:.3f} (fa={fa}, fg={fg:.3f}, fh={fh:.3f})"
    )

    return GutWallResult(
        drug_name=drug_name,
        fa=fa,
        fg=fg,
        fh=fh,
        f_oral=f_oral,
        eg=eg,
        eh=eh,
        clg_int_mL_per_min=clg_int_mL_per_min,
        qg_mL_per_min=qg_mL_per_min,
        notes=notes,
    )
