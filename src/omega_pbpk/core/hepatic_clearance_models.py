"""Multiple hepatic clearance models — Phase 276.

Implements three classic hepatic clearance models:
1. Well-Stirred (venous equilibration) — most common
2. Parallel-Tube (sinusoidal perfusion)
3. Dispersion model — intermediate

References
----------
- Pang KS & Rowland M, J Pharmacokinet Biopharm. 1977;5(6):625-53
- Roberts MS & Rowland M, J Pharmacokinet Biopharm. 1986
- Weiss M, J Pharm Sci. 1997;86(6):711-15
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "HepaticCLResult",
    "well_stirred_model",
    "parallel_tube_model",
    "dispersion_model",
    "compare_hepatic_models",
]


@dataclass(frozen=True)
class HepaticCLResult:
    """Result from hepatic clearance calculation."""

    model: str
    clint_L_per_h: float  # Intrinsic clearance
    fu: float  # Unbound fraction
    qh_L_per_h: float  # Hepatic blood flow
    eh: float  # Hepatic extraction ratio (0-1)
    cl_hepatic_L_per_h: float  # Hepatic clearance
    bioavailability_fg: float  # First-pass hepatic bioavailability (1-EH)
    notes: str


def _clint_effective(clint: float, fu: float) -> float:
    """Effective intrinsic CL: CLint_eff = CLint * fu (unbound)."""
    return clint * fu


def well_stirred_model(
    clint_L_per_h: float,
    fu: float,
    qh_L_per_h: float = 90.0,
) -> HepaticCLResult:
    """Well-stirred (venous equilibration) hepatic clearance model.

    CL_H = Q_H * fu * CLint / (Q_H + fu * CLint)
    EH   = fu * CLint / (Q_H + fu * CLint)

    Parameters
    ----------
    clint_L_per_h : Intrinsic clearance (L/h). Must be > 0.
    fu : Unbound fraction in blood (0, 1]. Must be in (0, 1].
    qh_L_per_h : Hepatic blood flow (L/h). Must be > 0.

    Returns
    -------
    HepaticCLResult
    """
    if clint_L_per_h <= 0:
        raise ValueError("clint_L_per_h must be > 0")
    if not (0 < fu <= 1):
        raise ValueError("fu must be in (0, 1]")
    if qh_L_per_h <= 0:
        raise ValueError("qh_L_per_h must be > 0")

    clint_eff = _clint_effective(clint_L_per_h, fu)
    eh = clint_eff / (qh_L_per_h + clint_eff)
    cl_h = qh_L_per_h * eh
    fg = 1.0 - eh

    return HepaticCLResult(
        model="well_stirred",
        clint_L_per_h=clint_L_per_h,
        fu=fu,
        qh_L_per_h=qh_L_per_h,
        eh=eh,
        cl_hepatic_L_per_h=cl_h,
        bioavailability_fg=fg,
        notes=f"Well-stirred: EH={eh:.3f}, CL_H={cl_h:.2f} L/h, Fg={fg:.3f}.",
    )


def parallel_tube_model(
    clint_L_per_h: float,
    fu: float,
    qh_L_per_h: float = 90.0,
) -> HepaticCLResult:
    """Parallel-tube (sinusoidal) hepatic clearance model.

    EH = 1 - exp(-fu * CLint / Q_H)
    CL_H = Q_H * EH

    Parameters
    ----------
    clint_L_per_h : Intrinsic clearance (L/h). Must be > 0.
    fu : Unbound fraction (0, 1]. Must be in (0, 1].
    qh_L_per_h : Hepatic blood flow (L/h). Must be > 0.

    Returns
    -------
    HepaticCLResult
    """
    if clint_L_per_h <= 0:
        raise ValueError("clint_L_per_h must be > 0")
    if not (0 < fu <= 1):
        raise ValueError("fu must be in (0, 1]")
    if qh_L_per_h <= 0:
        raise ValueError("qh_L_per_h must be > 0")

    clint_eff = _clint_effective(clint_L_per_h, fu)
    eh = 1.0 - math.exp(-clint_eff / qh_L_per_h)
    cl_h = qh_L_per_h * eh
    fg = 1.0 - eh

    return HepaticCLResult(
        model="parallel_tube",
        clint_L_per_h=clint_L_per_h,
        fu=fu,
        qh_L_per_h=qh_L_per_h,
        eh=eh,
        cl_hepatic_L_per_h=cl_h,
        bioavailability_fg=fg,
        notes=f"Parallel-tube: EH={eh:.3f}, CL_H={cl_h:.2f} L/h, Fg={fg:.3f}.",
    )


def dispersion_model(
    clint_L_per_h: float,
    fu: float,
    qh_L_per_h: float = 90.0,
    dn: float = 0.17,
) -> HepaticCLResult:
    """Dispersion hepatic clearance model.

    Uses dispersion number Dn to model mixing between plug flow and well-stirred.
    Intermediate between parallel-tube (Dn->0) and well-stirred (Dn->inf).

    EH via numerical approximation using the dispersion equation.

    Parameters
    ----------
    clint_L_per_h : Intrinsic clearance (L/h). Must be > 0.
    fu : Unbound fraction (0, 1]. Must be in (0, 1].
    qh_L_per_h : Hepatic blood flow (L/h). Must be > 0.
    dn : Dispersion number (dimensionless, 0.01-1.0). Must be > 0.

    Returns
    -------
    HepaticCLResult
    """
    if clint_L_per_h <= 0:
        raise ValueError("clint_L_per_h must be > 0")
    if not (0 < fu <= 1):
        raise ValueError("fu must be in (0, 1]")
    if qh_L_per_h <= 0:
        raise ValueError("qh_L_per_h must be > 0")
    if dn <= 0:
        raise ValueError("dn must be > 0")

    clint_eff = _clint_effective(clint_L_per_h, fu)
    r = clint_eff / qh_L_per_h  # dimensionless CLint/Q

    # Dispersion model EH formula (Roberts & Rowland 1986, simplified)
    # EH = 1 - [4a * exp(1/(2Dn))] / [(1+a)^2 * exp(a/(2Dn)) - (1-a)^2 * exp(-a/(2Dn))]
    # where a = sqrt(1 + 4*r*Dn)
    a = math.sqrt(1.0 + 4.0 * r * dn)
    denom = (1.0 + a) ** 2 * math.exp(a / (2.0 * dn)) - (1.0 - a) ** 2 * math.exp(
        -a / (2.0 * dn)
    )
    if abs(denom) < 1e-10:
        # Fallback to well-stirred approximation
        eh = r / (1.0 + r)
    else:
        numerator = 4.0 * a * math.exp(1.0 / (2.0 * dn))
        eh = max(0.0, min(1.0, 1.0 - numerator / denom))

    cl_h = qh_L_per_h * eh
    fg = 1.0 - eh

    return HepaticCLResult(
        model="dispersion",
        clint_L_per_h=clint_L_per_h,
        fu=fu,
        qh_L_per_h=qh_L_per_h,
        eh=eh,
        cl_hepatic_L_per_h=cl_h,
        bioavailability_fg=fg,
        notes=f"Dispersion(Dn={dn:.2f}): EH={eh:.3f}, CL_H={cl_h:.2f} L/h, Fg={fg:.3f}.",
    )


def compare_hepatic_models(
    clint_L_per_h: float,
    fu: float,
    qh_L_per_h: float = 90.0,
    dn: float = 0.17,
) -> list[HepaticCLResult]:
    """Compare all three hepatic clearance models.

    Returns list: [well_stirred, parallel_tube, dispersion].
    """
    return [
        well_stirred_model(clint_L_per_h, fu, qh_L_per_h),
        parallel_tube_model(clint_L_per_h, fu, qh_L_per_h),
        dispersion_model(clint_L_per_h, fu, qh_L_per_h, dn),
    ]
