"""Phase 349 — Bioequivalence Prediction from Formulation Parameters.

Predicts BE outcome between test and reference formulations from PK parameters
using log-normal GMR approximation and 90% CI TOST framework.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BEFormulationResult:
    """Result of bioequivalence prediction from formulation PK parameters."""

    drug_name: str
    gmr_auc: float
    gmr_cmax: float
    ci90_auc_lower: float
    ci90_auc_upper: float
    ci90_cmax_lower: float
    ci90_cmax_upper: float
    be_auc: bool
    be_cmax: bool
    be_conclusion: str
    tmax_diff_h: float
    power_estimate_pct: float
    required_n_for_80pct_power: int
    notes: str


def predict_be_formulation(
    drug_name: str,
    auc_test_mg_h_per_L: float,
    auc_ref_mg_h_per_L: float,
    cmax_test_mg_L: float,
    cmax_ref_mg_L: float,
    tmax_test_h: float,
    tmax_ref_h: float,
    n_subjects: int = 24,
    cv_intra_pct: float = 20.0,
    alpha: float = 0.05,
    nti_drug: bool = False,
) -> BEFormulationResult:
    """Predict bioequivalence outcome between test and reference formulations.

    Parameters
    ----------
    drug_name:
        Name of the drug being evaluated.
    auc_test_mg_h_per_L:
        AUC of the test formulation (mg*h/L).
    auc_ref_mg_h_per_L:
        AUC of the reference formulation (mg*h/L).
    cmax_test_mg_L:
        Cmax of the test formulation (mg/L).
    cmax_ref_mg_L:
        Cmax of the reference formulation (mg/L).
    tmax_test_h:
        Tmax of the test formulation (h).
    tmax_ref_h:
        Tmax of the reference formulation (h).
    n_subjects:
        Number of subjects in the BE study (default 24).
    cv_intra_pct:
        Intrasubject CV% (default 20.0).
    alpha:
        Significance level (default 0.05).
    nti_drug:
        Whether the drug is a narrow therapeutic index drug.

    Returns
    -------
    BEFormulationResult
    """
    # Validation
    if auc_test_mg_h_per_L <= 0:
        raise ValueError("auc_test_mg_h_per_L must be > 0")
    if auc_ref_mg_h_per_L <= 0:
        raise ValueError("auc_ref_mg_h_per_L must be > 0")
    if cmax_test_mg_L <= 0:
        raise ValueError("cmax_test_mg_L must be > 0")
    if cmax_ref_mg_L <= 0:
        raise ValueError("cmax_ref_mg_L must be > 0")
    if n_subjects < 2:
        raise ValueError("n_subjects must be >= 2")
    if cv_intra_pct <= 0:
        raise ValueError("cv_intra_pct must be > 0")

    # Geometric mean ratios
    gmr_auc = auc_test_mg_h_per_L / auc_ref_mg_h_per_L
    gmr_cmax = cmax_test_mg_L / cmax_ref_mg_L

    # SE for log-normal approximation (2-period crossover)
    se_log = (cv_intra_pct / 100.0) / math.sqrt(n_subjects / 2.0)

    # 90% CI uses z = 1.645
    t_val = 1.645

    # AUC 90% CI
    ci90_auc_lower = gmr_auc * math.exp(-t_val * se_log)
    ci90_auc_upper = gmr_auc * math.exp(t_val * se_log)

    # Cmax 90% CI
    ci90_cmax_lower = gmr_cmax * math.exp(-t_val * se_log)
    ci90_cmax_upper = gmr_cmax * math.exp(t_val * se_log)

    # BE criteria: 80-125% bounds
    be_lower = 0.80
    be_upper = 1.25

    be_auc = be_lower <= ci90_auc_lower and ci90_auc_upper <= be_upper
    be_cmax = be_lower <= ci90_cmax_lower and ci90_cmax_upper <= be_upper

    be_conclusion = "bioequivalent" if (be_auc and be_cmax) else "not_bioequivalent"

    # Tmax difference
    tmax_diff_h = abs(tmax_test_h - tmax_ref_h)

    # Power estimate (simplified)
    power_raw = 100.0 * (1.0 - (cv_intra_pct / 100.0) * 2.0 / math.sqrt(n_subjects))
    power_estimate_pct = max(0.0, min(100.0, power_raw))

    # Required n for 80% power
    # n = ceil(2 * (z_alpha + z_beta)^2 * (cv/100)^2 / (log(1.25))^2)
    # z_alpha = 1.645 (90% CI), z_beta = 0.842 (80% power)
    z_sum = 1.645 + 0.842
    log_bound = math.log(1.25)
    n_raw = 2.0 * (z_sum**2) * ((cv_intra_pct / 100.0) ** 2) / (log_bound**2)
    required_n = math.ceil(n_raw)
    required_n = max(2, required_n)

    # Build notes
    notes_parts = []
    if be_conclusion == "bioequivalent":
        notes_parts.append("Both AUC and Cmax 90% CIs within 80-125% limits.")
    else:
        if not be_auc:
            notes_parts.append(
                f"AUC 90% CI [{ci90_auc_lower:.3f}, {ci90_auc_upper:.3f}] outside BE limits."
            )
        if not be_cmax:
            notes_parts.append(
                f"Cmax 90% CI [{ci90_cmax_lower:.3f}, {ci90_cmax_upper:.3f}] outside BE limits."
            )

    if nti_drug and abs(gmr_auc - 1.0) > 0.05:
        notes_parts.append("NTI drug: tighter acceptance criteria may apply.")

    if tmax_diff_h > 1.0:
        notes_parts.append(f"Tmax difference {tmax_diff_h:.1f}h may affect rate of absorption.")

    notes = " ".join(notes_parts) if notes_parts else "No additional notes."

    return BEFormulationResult(
        drug_name=drug_name,
        gmr_auc=gmr_auc,
        gmr_cmax=gmr_cmax,
        ci90_auc_lower=ci90_auc_lower,
        ci90_auc_upper=ci90_auc_upper,
        ci90_cmax_lower=ci90_cmax_lower,
        ci90_cmax_upper=ci90_cmax_upper,
        be_auc=bool(be_auc),
        be_cmax=bool(be_cmax),
        be_conclusion=be_conclusion,
        tmax_diff_h=tmax_diff_h,
        power_estimate_pct=power_estimate_pct,
        required_n_for_80pct_power=required_n,
        notes=notes,
    )
