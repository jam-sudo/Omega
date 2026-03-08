"""Phase 558 - Oral Bioavailability Sensitivity Analysis.

Analytically compute sensitivity of oral bioavailability (F = fa * fg * fh)
to each PK parameter.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BioavailabilitySensitivityResult:
    """Result of bioavailability sensitivity analysis."""

    drug_name: str
    fa: float
    fg: float
    fh: float
    f_oral: float
    d_f_d_fa: float
    d_f_d_fg: float
    d_f_d_fh: float
    sensitivity_fa: float
    sensitivity_fg: float
    sensitivity_fh: float
    limiting_factor: str
    f_class: str
    notes: str


def analyze_bioavailability_sensitivity(
    drug_name: str,
    fa: float,
    fg: float,
    fh: float,
) -> BioavailabilitySensitivityResult:
    """Analyze sensitivity of oral bioavailability to fa, fg, fh.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    fa : float
        Fraction absorbed, must be in (0, 1].
    fg : float
        Gut availability, must be in (0, 1].
    fh : float
        Hepatic availability, must be in (0, 1].

    Returns
    -------
    BioavailabilitySensitivityResult
    """
    for name, val in [("fa", fa), ("fg", fg), ("fh", fh)]:
        if val <= 0 or val > 1:
            raise ValueError(f"{name} must be in (0, 1], got {val}")

    f_oral = fa * fg * fh

    # Partial derivatives
    d_f_d_fa = fg * fh
    d_f_d_fg = fa * fh
    d_f_d_fh = fa * fg

    # Normalized sensitivities: (x/F) * dF/dx = 1.0 for product model
    sensitivity_fa = 1.0
    sensitivity_fg = 1.0
    sensitivity_fh = 1.0

    # Limiting factor
    params = {"fa": fa, "fg": fg, "fh": fh}
    limiting_factor = min(params, key=params.get)

    # Classification
    if f_oral > 0.7:
        f_class = "high"
    elif f_oral >= 0.3:
        f_class = "moderate"
    else:
        f_class = "low"

    notes = f"F = fa*fg*fh = {f_oral:.4f}, limiting factor: {limiting_factor}"

    return BioavailabilitySensitivityResult(
        drug_name=drug_name,
        fa=fa,
        fg=fg,
        fh=fh,
        f_oral=f_oral,
        d_f_d_fa=d_f_d_fa,
        d_f_d_fg=d_f_d_fg,
        d_f_d_fh=d_f_d_fh,
        sensitivity_fa=sensitivity_fa,
        sensitivity_fg=sensitivity_fg,
        sensitivity_fh=sensitivity_fh,
        limiting_factor=limiting_factor,
        f_class=f_class,
        notes=notes,
    )


def sweep_fa_fg_fh(drug_name: str, n_points: int = 5) -> list:
    """Sweep fa, fg, fh and return all combinations.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    n_points : int
        Number of points per parameter (values from 0.1 to 1.0).

    Returns
    -------
    list of BioavailabilitySensitivityResult
        Sorted by f_oral descending.
    """
    if n_points < 2:
        raise ValueError("n_points must be >= 2")

    step = (1.0 - 0.1) / (n_points - 1)
    values = [0.1 + i * step for i in range(n_points)]

    results = []
    for fa_val in values:
        for fg_val in values:
            for fh_val in values:
                result = analyze_bioavailability_sensitivity(
                    drug_name=drug_name,
                    fa=fa_val,
                    fg=fg_val,
                    fh=fh_val,
                )
                results.append(result)

    results.sort(key=lambda r: r.f_oral, reverse=True)
    return results
