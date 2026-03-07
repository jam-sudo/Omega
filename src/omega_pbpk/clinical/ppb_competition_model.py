"""Plasma Protein Binding (PPB) Competition Model.

Models competition between two drugs for plasma protein binding sites
(primarily albumin, α1-acid glycoprotein).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FuShiftResult:
    """Result of a displacement/DDI fu shift assessment."""

    drug_name: str
    competitor_name: str
    fu_alone: float
    fu_displaced: float
    fu_change_pct: float
    clinical_relevance: str  # 'none', 'monitor', 'significant'
    notes: str


def fu_adjusted_for_competition(
    fu_alone: float,
    c_drug_mg_L: float,
    c_competitor_mg_L: float,
    ka_drug: float,
    ka_competitor: float,
    protein_conc_mg_L: float = 42000.0,
) -> float:
    """Compute fu adjusted for competitive binding with a second drug.

    Uses a simplified Scatchard competitive binding model with iterative
    solution (10 iterations) to find the free drug concentration.

    Parameters
    ----------
    fu_alone:
        Free fraction of the drug when no competitor is present (0, 1].
    c_drug_mg_L:
        Total drug concentration (mg/L).
    c_competitor_mg_L:
        Total competitor concentration (mg/L).
    ka_drug:
        Association constant of the drug for albumin (L/mg).
    ka_competitor:
        Association constant of the competitor for albumin (L/mg).
    protein_conc_mg_L:
        Total albumin/protein concentration (mg/L). Default ≈ 42 g/L.

    Returns
    -------
    float
        Adjusted free fraction fu (0, 1].
    """
    if not (0 < fu_alone <= 1):
        raise ValueError("fu_alone must be in (0, 1]")
    if c_drug_mg_L < 0:
        raise ValueError("c_drug_mg_L must be >= 0")
    if c_competitor_mg_L < 0:
        raise ValueError("c_competitor_mg_L must be >= 0")
    if protein_conc_mg_L <= 0:
        raise ValueError("protein_conc_mg_L must be > 0")

    # Edge cases
    if c_drug_mg_L == 0.0:
        return fu_alone
    if c_competitor_mg_L == 0.0:
        return fu_alone

    # Estimate available binding sites from fu_alone.
    # When no competitor: fu_alone = c_free / c_total
    # bound_drug = c_drug * (1 - fu_alone)
    # Scatchard: bound = n * P * Ka * c_free / (1 + Ka * c_free)
    # Approximate n*P (total binding capacity) from fu_alone at c_drug_mg_L
    c_free_drug_init = fu_alone * c_drug_mg_L
    c_bound_drug_init = c_drug_mg_L - c_free_drug_init
    # Avoid division by zero
    if c_free_drug_init <= 0:
        n_sites = protein_conc_mg_L
    else:
        # n_sites = bound * (1 + Ka*cf) / (Ka*cf * P) ... but we fold P into n_sites
        # simplified: total_capacity = bound / (Ka_drug * c_free / (1 + Ka_drug*c_free))
        denom = ka_drug * c_free_drug_init / (1.0 + ka_drug * c_free_drug_init)
        if denom > 0:
            n_sites = c_bound_drug_init / denom
        else:
            n_sites = protein_conc_mg_L

    # Iterative solution: start with c_free_drug = fu_alone * c_drug
    c_free_drug = fu_alone * c_drug_mg_L
    c_free_comp = fu_alone * c_competitor_mg_L  # rough initial guess

    for _ in range(10):
        # Competitive Scatchard:
        # bound_drug = n_sites * ka_drug * c_free_drug / (1 + ka_drug*c_free_drug + ka_competitor*c_free_comp)
        denom = 1.0 + ka_drug * c_free_drug + ka_competitor * c_free_comp
        bound_drug = n_sites * ka_drug * c_free_drug / denom
        bound_comp = n_sites * ka_competitor * c_free_comp / denom

        # Mass balance
        c_free_drug = c_drug_mg_L - bound_drug
        c_free_comp = c_competitor_mg_L - bound_comp

        # Clamp to physically valid range
        c_free_drug = max(c_free_drug, 1e-12)
        c_free_comp = max(c_free_comp, 1e-12)

    fu_adjusted = c_free_drug / c_drug_mg_L
    # Clamp to (0, 1]
    fu_adjusted = max(min(fu_adjusted, 1.0), 1e-9)
    return fu_adjusted


def assess_ddi_fu_shift(
    drug_name: str,
    competitor_name: str,
    fu_drug_alone: float,
    c_drug_mg_L: float,
    c_competitor_mg_L: float,
    ka_drug: float = 1e5,
    ka_competitor: float = 1e5,
) -> FuShiftResult:
    """Assess clinical DDI significance of plasma protein binding displacement.

    Parameters
    ----------
    drug_name:
        Name of the victim drug.
    competitor_name:
        Name of the perpetrator / displacing drug.
    fu_drug_alone:
        Free fraction of the victim drug alone (0, 1].
    c_drug_mg_L:
        Total victim drug concentration (mg/L).
    c_competitor_mg_L:
        Total competitor concentration (mg/L).
    ka_drug:
        Association constant of victim drug (L/mg). Default 1e5.
    ka_competitor:
        Association constant of competitor (L/mg). Default 1e5.

    Returns
    -------
    FuShiftResult
    """
    fu_displaced = fu_adjusted_for_competition(
        fu_alone=fu_drug_alone,
        c_drug_mg_L=c_drug_mg_L,
        c_competitor_mg_L=c_competitor_mg_L,
        ka_drug=ka_drug,
        ka_competitor=ka_competitor,
    )

    if fu_drug_alone > 0:
        fu_change_pct = (fu_displaced - fu_drug_alone) / fu_drug_alone * 100.0
    else:
        fu_change_pct = 0.0

    abs_change = abs(fu_change_pct)
    if abs_change < 20.0:
        relevance = "none"
        notes = (
            f"Fu change of {fu_change_pct:.1f}% is below the 20% monitoring threshold. "
            "No clinically meaningful PPB displacement expected."
        )
    elif abs_change <= 50.0:
        relevance = "monitor"
        notes = (
            f"Fu change of {fu_change_pct:.1f}% warrants monitoring. "
            "Increased free drug may elevate pharmacological effect transiently."
        )
    else:
        relevance = "significant"
        notes = (
            f"Fu change of {fu_change_pct:.1f}% is clinically significant. "
            "Substantial displacement may alter exposure and toxicity risk."
        )

    return FuShiftResult(
        drug_name=drug_name,
        competitor_name=competitor_name,
        fu_alone=fu_drug_alone,
        fu_displaced=fu_displaced,
        fu_change_pct=fu_change_pct,
        clinical_relevance=relevance,
        notes=notes,
    )


def protein_binding_sensitivity(
    drug_name: str,
    fu_baseline: float,
    cl_L_per_h: float,
    vd_L: float,
    fu_values: list[float] | None = None,
) -> list[dict]:
    """Sensitivity analysis of PK parameters to changes in fu.

    Uses restrictive clearance model: CL_effective = CL_baseline * fu_baseline / fu.
    Vd is assumed unchanged for simplicity.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    fu_baseline:
        Baseline free fraction.
    cl_L_per_h:
        Baseline total clearance (L/h).
    vd_L:
        Baseline volume of distribution (L).
    fu_values:
        fu values to evaluate. Defaults to [0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
        filtered to those >= fu_baseline / 2.

    Returns
    -------
    list[dict]
        Each entry: {fu, cl_effective, auc_ratio, t_half_ratio}
    """
    if fu_values is None:
        candidates = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        min_fu = fu_baseline / 2.0
        fu_values = [f for f in candidates if f >= min_fu]

    results: list[dict] = []
    # Baseline t_half
    if cl_L_per_h > 0:
        ke_baseline = cl_L_per_h / vd_L
        t_half_baseline = math.log(2) / ke_baseline if ke_baseline > 0 else float("inf")
    else:
        t_half_baseline = float("inf")

    for fu in fu_values:
        if fu <= 0 or fu > 1:
            continue
        # Restrictive CL model
        cl_effective = cl_L_per_h * fu_baseline / fu
        # AUC = Dose / CL; auc_ratio relative to baseline
        if cl_effective > 0 and cl_L_per_h > 0:
            auc_ratio = cl_L_per_h / cl_effective  # = fu / fu_baseline
        else:
            auc_ratio = float("inf")

        # t_half = ln2 / ke = ln2 * Vd / CL (Vd unchanged)
        ke_new = cl_effective / vd_L if vd_L > 0 else 0.0
        t_half_new = math.log(2) / ke_new if ke_new > 0 else float("inf")
        t_half_ratio = t_half_new / t_half_baseline if t_half_baseline > 0 else float("inf")

        results.append(
            {
                "fu": fu,
                "cl_effective": cl_effective,
                "auc_ratio": auc_ratio,
                "t_half_ratio": t_half_ratio,
            }
        )

    return results
