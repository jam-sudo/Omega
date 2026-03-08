"""Phase 587 - Drug Absorption Limited by Gut Wall Metabolism.

Models first-pass gut wall metabolism (primarily CYP3A4 in enterocytes),
computing effective oral bioavailability using well-stirred models.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GutWallMetabolismResult:
    """Result of gut wall metabolism prediction."""

    drug_name: str
    dose_mg: float
    fa: float
    fg: float
    fh: float
    f_oral: float
    e_gut: float
    e_hepatic: float
    cl_gut_total_L_per_h: float
    cl_hepatic_L_per_h: float
    cyp3a4_contribution_pct: float
    effect_of_grapefruit: float
    notes: str


def predict_gut_wall_metabolism(
    drug_name: str,
    dose_mg: float,
    fa: float,
    cl_gut_intrinsic_L_per_h: float,
    cl_hepatic_intrinsic_L_per_h: float,
    fup: float,
    cyp3a4_fraction_gut: float = 0.8,
    q_gut_L_per_h: float = 36.0,
    q_hepatic_L_per_h: float = 90.0,
) -> GutWallMetabolismResult:
    """Predict gut wall metabolism and oral bioavailability.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Oral dose in mg.
    fa : float
        Fraction absorbed from lumen (0, 1].
    cl_gut_intrinsic_L_per_h : float
        Gut intrinsic clearance in L/h.
    cl_hepatic_intrinsic_L_per_h : float
        Hepatic intrinsic clearance in L/h.
    fup : float
        Fraction unbound in plasma (0, 1].
    cyp3a4_fraction_gut : float
        Fraction of gut metabolism via CYP3A4 (default 0.8).
    q_gut_L_per_h : float
        Gut blood flow in L/h (default 36.0).
    q_hepatic_L_per_h : float
        Hepatic blood flow in L/h (default 90.0).

    Returns
    -------
    GutWallMetabolismResult
    """
    if not (0.0 < fa <= 1.0):
        raise ValueError("fa must be in (0, 1]")
    if not (0.0 < fup <= 1.0):
        raise ValueError("fup must be in (0, 1]")
    if cl_gut_intrinsic_L_per_h <= 0:
        raise ValueError("cl_gut_intrinsic_L_per_h must be > 0")
    if cl_hepatic_intrinsic_L_per_h <= 0:
        raise ValueError("cl_hepatic_intrinsic_L_per_h must be > 0")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")

    # Gut wall: well-stirred model
    fg_ws = q_gut_L_per_h / (q_gut_L_per_h + fup * cl_gut_intrinsic_L_per_h)
    e_gut = 1.0 - fg_ws
    cl_gut_total = q_gut_L_per_h * e_gut

    # Hepatic: well-stirred model
    fh_ws = q_hepatic_L_per_h / (q_hepatic_L_per_h + fup * cl_hepatic_intrinsic_L_per_h)
    e_hepatic = 1.0 - fh_ws
    cl_hepatic = q_hepatic_L_per_h * e_hepatic

    # Oral bioavailability
    f_oral = fa * fg_ws * fh_ws

    # CYP3A4 contribution
    cyp3a4_contribution_pct = cyp3a4_fraction_gut * 100.0

    # Grapefruit effect: inhibits gut CYP3A4 by ~70%
    cl_gut_with_gf = cl_gut_intrinsic_L_per_h * (1.0 - 0.7 * cyp3a4_fraction_gut)
    fg_gf = q_gut_L_per_h / (q_gut_L_per_h + fup * cl_gut_with_gf)
    f_oral_gf = fa * fg_gf * fh_ws

    notes = (
        f"Well-stirred gut wall model for {drug_name}. "
        f"CYP3A4 accounts for {cyp3a4_contribution_pct:.0f}% of gut metabolism. "
        f"Grapefruit juice increases F_oral from {f_oral:.4f} to {f_oral_gf:.4f}."
    )

    return GutWallMetabolismResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        fa=fa,
        fg=fg_ws,
        fh=fh_ws,
        f_oral=f_oral,
        e_gut=e_gut,
        e_hepatic=e_hepatic,
        cl_gut_total_L_per_h=cl_gut_total,
        cl_hepatic_L_per_h=cl_hepatic,
        cyp3a4_contribution_pct=cyp3a4_contribution_pct,
        effect_of_grapefruit=f_oral_gf,
        notes=notes,
    )


def compare_cyp3a4_fractions(
    drug_name: str,
    dose_mg: float,
    fa: float,
    cl_gut_intrinsic: float,
    cl_hepatic_intrinsic: float,
    fup: float,
    fractions: list = None,
) -> list:
    """Compare gut wall metabolism across different CYP3A4 fractions.

    Returns list of GutWallMetabolismResult sorted by f_oral ascending.
    """
    if fractions is None:
        fractions = [0.2, 0.5, 0.8, 1.0]

    results = []
    for frac in fractions:
        result = predict_gut_wall_metabolism(
            drug_name=drug_name,
            dose_mg=dose_mg,
            fa=fa,
            cl_gut_intrinsic_L_per_h=cl_gut_intrinsic,
            cl_hepatic_intrinsic_L_per_h=cl_hepatic_intrinsic,
            fup=fup,
            cyp3a4_fraction_gut=frac,
        )
        results.append(result)

    results.sort(key=lambda r: r.f_oral)
    return results
