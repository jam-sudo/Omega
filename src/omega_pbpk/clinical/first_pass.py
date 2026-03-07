"""First-pass metabolism calculator.

Estimates oral bioavailability by modelling sequential extraction across
the gut wall, liver, and lungs:

    F = Fa x Fg x Fh x Fl

where Fa is fraction absorbed, Fg is gut-wall availability, Fh is hepatic
availability, and Fl accounts for pulmonary extraction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "FirstPassResult",
    "calculate_first_pass",
    "sensitivity_first_pass",
]


@dataclass(frozen=True)
class FirstPassResult:
    """Result of a first-pass metabolism calculation.

    Attributes
    ----------
    drug_name : Drug identifier.
    fa : Fraction absorbed from GI lumen.
    fg : Gut-wall availability (1 - Eg).
    fh : Hepatic availability (1 - Eh).
    fl : Lung extraction factor.
    bioavailability_f : Overall oral bioavailability (Fa * Fg * Fh * Fl).
    eg_gut : Gut extraction ratio.
    eh_liver : Hepatic extraction ratio.
    high_extraction : True when Eh > 0.7.
    first_pass_loss_pct : Percentage of dose lost to first-pass (0-100).
    notes : Additional information.
    """

    drug_name: str
    fa: float
    fg: float
    fh: float
    fl: float
    bioavailability_f: float
    eg_gut: float
    eh_liver: float
    high_extraction: bool
    first_pass_loss_pct: float
    notes: str


def _validate_inputs(
    fa: float,
    clg_L_per_h: float,
    qg_L_per_h: float,
    clh_intrinsic_L_per_h: float,
    qh_L_per_h: float,
    fup: float,
    fub: float,
    fl: float,
) -> None:
    """Raise ``ValueError`` for out-of-range parameters."""
    if fa < 0.0 or fa > 1.0:
        raise ValueError(f"fa must be in [0, 1], got {fa}")
    if clg_L_per_h <= 0.0:
        raise ValueError(f"clg_L_per_h must be > 0, got {clg_L_per_h}")
    if qg_L_per_h <= 0.0:
        raise ValueError(f"qg_L_per_h must be > 0, got {qg_L_per_h}")
    if clh_intrinsic_L_per_h <= 0.0:
        raise ValueError(f"clh_intrinsic_L_per_h must be > 0, got {clh_intrinsic_L_per_h}")
    if qh_L_per_h <= 0.0:
        raise ValueError(f"qh_L_per_h must be > 0, got {qh_L_per_h}")
    if fup <= 0.0 or fup > 1.0:
        raise ValueError(f"fup must be in (0, 1], got {fup}")
    if fub <= 0.0 or fub > 1.0:
        raise ValueError(f"fub must be in (0, 1], got {fub}")
    if fl < 0.0 or fl > 1.0:
        raise ValueError(f"fl must be in [0, 1], got {fl}")


def calculate_first_pass(
    drug_name: str,
    fa: float = 1.0,
    clg_L_per_h: float = 5.0,
    qg_L_per_h: float = 18.0,
    clh_intrinsic_L_per_h: float = 20.0,
    qh_L_per_h: float = 90.0,
    fup: float = 0.1,
    fub: float = 0.1,
    fl: float = 1.0,
    use_fu_correction: bool = True,
) -> FirstPassResult:
    """Calculate first-pass extraction and oral bioavailability.

    Parameters
    ----------
    drug_name:
        Drug identifier.
    fa:
        Fraction absorbed from GI lumen (0-1, default 1.0).
    clg_L_per_h:
        Gut intrinsic clearance (L/h, default 5.0).
    qg_L_per_h:
        Gut blood flow (L/h, default 18.0).
    clh_intrinsic_L_per_h:
        Hepatic intrinsic clearance (L/h, default 20.0).
    qh_L_per_h:
        Hepatic blood flow (L/h, default 90.0).
    fup:
        Fraction unbound in plasma (0-1], default 0.1).
    fub:
        Fraction unbound in blood (0-1], default 0.1).
    fl:
        Lung extraction factor (0-1, default 1.0).
    use_fu_correction:
        If True, correct hepatic clearance for protein binding.

    Returns
    -------
    FirstPassResult
    """
    _validate_inputs(
        fa,
        clg_L_per_h,
        qg_L_per_h,
        clh_intrinsic_L_per_h,
        qh_L_per_h,
        fup,
        fub,
        fl,
    )

    # Gut extraction
    eg = clg_L_per_h / (qg_L_per_h + clg_L_per_h)
    fg = 1.0 - eg

    # Hepatic extraction
    if use_fu_correction:
        clh_eff = clh_intrinsic_L_per_h * fup / fub
    else:
        clh_eff = clh_intrinsic_L_per_h

    eh = clh_eff / (qh_L_per_h + clh_eff)
    fh = 1.0 - eh

    # Overall bioavailability
    bioavailability_f = fa * fg * fh * fl
    first_pass_loss_pct = (1.0 - bioavailability_f) * 100.0

    high_extraction = eh > 0.7

    notes_parts: list[str] = []
    if high_extraction:
        notes_parts.append("High hepatic extraction (Eh > 0.7)")
    if fa < 1.0:
        notes_parts.append(f"Incomplete absorption (Fa={fa:.2f})")
    if fl < 1.0:
        notes_parts.append(f"Pulmonary extraction applied (Fl={fl:.2f})")
    notes = "; ".join(notes_parts) if notes_parts else "Standard first-pass calculation"

    return FirstPassResult(
        drug_name=drug_name,
        fa=fa,
        fg=fg,
        fh=fh,
        fl=fl,
        bioavailability_f=bioavailability_f,
        eg_gut=eg,
        eh_liver=eh,
        high_extraction=high_extraction,
        first_pass_loss_pct=first_pass_loss_pct,
        notes=notes,
    )


def sensitivity_first_pass(
    drug_name: str,
    fg_range: tuple[float, float, int],
    fh_range: tuple[float, float, int],
    fa: float = 1.0,
    qg_L_per_h: float = 18.0,
    qh_L_per_h: float = 90.0,
    fup: float = 0.1,
    fub: float = 0.1,
    fl: float = 1.0,
    use_fu_correction: bool = True,
) -> list[FirstPassResult]:
    """Run a sensitivity analysis over gut and hepatic clearance ranges.

    Parameters
    ----------
    drug_name:
        Drug identifier.
    fg_range:
        (min, max, steps) for gut clearance (L/h) variation.
    fh_range:
        (min, max, steps) for hepatic intrinsic clearance (L/h) variation.
    fa:
        Fraction absorbed (default 1.0).
    qg_L_per_h:
        Gut blood flow (default 18.0 L/h).
    qh_L_per_h:
        Hepatic blood flow (default 90.0 L/h).
    fup:
        Fraction unbound in plasma (default 0.1).
    fub:
        Fraction unbound in blood (default 0.1).
    fl:
        Lung extraction factor (default 1.0).
    use_fu_correction:
        Correct hepatic clearance for protein binding (default True).

    Returns
    -------
    list[FirstPassResult]
        One result per combination of gut and hepatic clearance values.
    """
    clg_values = np.linspace(fg_range[0], fg_range[1], fg_range[2])
    clh_values = np.linspace(fh_range[0], fh_range[1], fh_range[2])

    results: list[FirstPassResult] = []
    for clg in clg_values:
        for clh in clh_values:
            result = calculate_first_pass(
                drug_name=drug_name,
                fa=fa,
                clg_L_per_h=float(clg),
                qg_L_per_h=qg_L_per_h,
                clh_intrinsic_L_per_h=float(clh),
                qh_L_per_h=qh_L_per_h,
                fup=fup,
                fub=fub,
                fl=fl,
                use_fu_correction=use_fu_correction,
            )
            results.append(result)

    return results
