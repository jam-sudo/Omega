"""Plasma protein binding competition model — Phase 262.

Models displacement interactions when two drugs compete for albumin
or alpha-1-acid glycoprotein (AAG) binding sites, using the
Scatchard/competitive binding model.

References
----------
- Rowland M & Tozer TN. Clinical Pharmacokinetics. 4th ed.
- MacKichan JJ, Pharmacokinet Biopharm. 1989 (protein binding competition)
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["PPBCompetitionResult", "assess_ppb_competition", "batch_ppb_screen"]


@dataclass(frozen=True)
class PPBCompetitionResult:
    """Result of protein binding competition assessment."""

    drug_a_name: str
    drug_b_name: str  # displacing drug ("perpetrator")
    protein_type: str  # "albumin" or "aag"
    fu_a_baseline: float  # Unbound fraction of drug A without drug B
    fu_a_displaced: float  # Unbound fraction of drug A with drug B present
    fu_ratio: float  # fu_displaced / fu_baseline (>1 = displacement)
    displacement_pct: float  # Percent increase in fu_A
    clinical_significance: str  # "negligible", "minor", "moderate", "major"
    notes: str


def _competitive_fu(
    fu_a: float,
    kd_a: float,
    c_protein: float,
    c_b_total: float,
    kd_b: float,
    fu_b: float,
    n_sites: float = 1.0,
) -> float:
    """Compute displaced fu_A in presence of competing drug B.

    Simplified competitive binding model: displacement proportional to
    occupancy of drug B on binding sites.

    Returns new fu_A after displacement.
    """
    # Bound fraction of B: fb_B = cb_bound / c_total_B
    # Approximate: fraction sites occupied by B = f_occ_B
    # fu_A_new ≈ fu_A / (1 - f_occ_B * (1 - fu_A))  [simplified]
    fb_b = 1.0 - fu_b
    f_occ_b = fb_b  # simplified: fraction of B bound ≈ bound fraction
    # Site occupancy reduction: if sites 50% occupied by B, fu_A increases
    occupancy_b = min(0.99, f_occ_b * c_b_total / (c_protein * n_sites))

    # New fu_A: if B occupies fraction theta of sites, remaining available sites for A reduced
    fu_a_new = fu_a / (1.0 - occupancy_b * (1.0 - fu_a))
    return min(1.0, fu_a_new)


def assess_ppb_competition(
    drug_a_name: str,
    fu_a: float,
    drug_b_name: str,
    fu_b: float,
    c_b_total_mg_L: float,
    protein_type: str = "albumin",
    c_protein_g_dL: float = 4.0,
) -> PPBCompetitionResult:
    """Assess plasma protein binding displacement of drug A by drug B.

    Parameters
    ----------
    drug_a_name : Name of the displaced drug.
    fu_a : Unbound fraction of drug A at baseline (0, 1].
    drug_b_name : Name of the displacing drug.
    fu_b : Unbound fraction of drug B (0, 1].
    c_b_total_mg_L : Total plasma concentration of drug B (mg/L). Must be > 0.
    protein_type : 'albumin' or 'aag'.
    c_protein_g_dL : Plasma protein concentration (g/dL). Must be > 0.

    Returns
    -------
    PPBCompetitionResult
    """
    if not (0 < fu_a <= 1):
        raise ValueError("fu_a must be in (0, 1]")
    if not (0 < fu_b <= 1):
        raise ValueError("fu_b must be in (0, 1]")
    if c_b_total_mg_L <= 0:
        raise ValueError("c_b_total_mg_L must be > 0")
    if c_protein_g_dL <= 0:
        raise ValueError("c_protein_g_dL must be > 0")
    if protein_type not in ("albumin", "aag"):
        raise ValueError("protein_type must be 'albumin' or 'aag'")

    # Convert protein to mg/L (albumin MW ~66,500 Da; AAG ~41,000 Da)
    # Keep protein in mg/L for consistent units
    c_protein_mg_L = c_protein_g_dL * 10.0  # g/dL -> g/L, multiply by 1000 gives mg/L

    # Kd estimates (approximate, mg/L): not strictly needed for simplified model
    kd_a = fu_a * c_protein_mg_L / max(1e-6, 1.0 - fu_a)
    kd_b = fu_b * c_protein_mg_L / max(1e-6, 1.0 - fu_b)

    fu_a_displaced = _competitive_fu(
        fu_a=fu_a,
        kd_a=kd_a,
        c_protein=c_protein_mg_L,
        c_b_total=c_b_total_mg_L,
        kd_b=kd_b,
        fu_b=fu_b,
        n_sites=1.0,
    )

    fu_ratio = fu_a_displaced / fu_a
    displacement_pct = (fu_ratio - 1.0) * 100.0

    if displacement_pct < 10:
        significance = "negligible"
    elif displacement_pct < 25:
        significance = "minor"
    elif displacement_pct < 50:
        significance = "moderate"
    else:
        significance = "major"

    notes = (
        f"{drug_a_name} fu: {fu_a:.3f} \u2192 {fu_a_displaced:.3f} "
        f"(+{displacement_pct:.1f}%) displaced by {drug_b_name}. "
        f"Clinical significance: {significance}."
    )

    return PPBCompetitionResult(
        drug_a_name=drug_a_name,
        drug_b_name=drug_b_name,
        protein_type=protein_type,
        fu_a_baseline=fu_a,
        fu_a_displaced=fu_a_displaced,
        fu_ratio=fu_ratio,
        displacement_pct=displacement_pct,
        clinical_significance=significance,
        notes=notes,
    )


def batch_ppb_screen(
    drug_a_name: str,
    fu_a: float,
    perpetrators: list[dict],
    protein_type: str = "albumin",
) -> list[PPBCompetitionResult]:
    """Screen multiple displacing drugs against a single victim drug.

    Parameters
    ----------
    perpetrators : List of dicts with keys: name, fu_b, c_b_total_mg_L.

    Returns sorted by displacement_pct descending.
    """
    results = [
        assess_ppb_competition(
            drug_a_name=drug_a_name,
            fu_a=fu_a,
            drug_b_name=p["name"],
            fu_b=p["fu_b"],
            c_b_total_mg_L=p["c_b_total_mg_L"],
            protein_type=protein_type,
        )
        for p in perpetrators
    ]
    return sorted(results, key=lambda r: r.displacement_pct, reverse=True)
