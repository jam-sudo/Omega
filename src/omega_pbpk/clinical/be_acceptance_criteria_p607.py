"""Phase 607 — Bioequivalence Acceptance Criteria

Evaluate whether two formulations meet bioequivalence acceptance criteria
per FDA/EMA guidelines (80-125% for standard drugs, 90-111.11% for NTI drugs).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BE_LOWER_STD: float = 0.80
BE_UPPER_STD: float = 1.25
NTI_LOWER: float = 0.90
NTI_UPPER: float = 1.1111  # 100/90 rounded


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BEAcceptanceCriteriaResult:
    """Result of bioequivalence acceptance criteria evaluation."""

    drug_name: str
    test_formulation: str
    reference_formulation: str
    auc_ratio: float  # geometric mean ratio test/reference
    cmax_ratio: float  # geometric mean ratio test/reference
    auc_90ci_lower: float
    auc_90ci_upper: float
    cmax_90ci_lower: float
    cmax_90ci_upper: float
    auc_be_pass: bool  # True if 90CI within 80-125%
    cmax_be_pass: bool
    overall_be_pass: bool  # both pass
    nti_drug: bool  # narrow therapeutic index
    nti_be_pass: bool  # True if 90CI within 90-111.11% for NTI
    guideline: str  # "FDA" or "EMA"
    recommendation: str  # summary recommendation
    notes: str


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------


def evaluate_be_criteria(
    drug_name: str,
    test_form: str,
    ref_form: str,
    auc_gmr: float,
    cmax_gmr: float,
    n_subjects: int,
    within_subject_cv_auc: float,
    within_subject_cv_cmax: float,
    nti_drug: bool = False,
    guideline: str = "FDA",
) -> BEAcceptanceCriteriaResult:
    """Evaluate bioequivalence acceptance criteria.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    test_form:
        Name of the test formulation.
    ref_form:
        Name of the reference formulation.
    auc_gmr:
        Geometric mean ratio (test/reference) for AUC.
    cmax_gmr:
        Geometric mean ratio (test/reference) for Cmax.
    n_subjects:
        Number of subjects in the study.
    within_subject_cv_auc:
        Within-subject coefficient of variation for AUC (as a fraction, e.g. 0.20 for 20%).
    within_subject_cv_cmax:
        Within-subject coefficient of variation for Cmax (as a fraction, e.g. 0.25 for 25%).
    nti_drug:
        Whether the drug has a narrow therapeutic index.
    guideline:
        Regulatory guideline ("FDA" or "EMA").

    Returns
    -------
    BEAcceptanceCriteriaResult
    """
    # --- Validation ---
    if not (0 < auc_gmr <= 3):
        raise ValueError(f"auc_gmr must be in (0, 3], got {auc_gmr}")
    if not (0 < cmax_gmr <= 3):
        raise ValueError(f"cmax_gmr must be in (0, 3], got {cmax_gmr}")
    if n_subjects < 2:
        raise ValueError(f"n_subjects must be >= 2, got {n_subjects}")
    if within_subject_cv_auc <= 0:
        raise ValueError(f"within_subject_cv_auc must be > 0, got {within_subject_cv_auc}")
    if within_subject_cv_cmax <= 0:
        raise ValueError(f"within_subject_cv_cmax must be > 0, got {within_subject_cv_cmax}")
    if guideline not in ("FDA", "EMA"):
        raise ValueError(f"guideline must be 'FDA' or 'EMA', got {guideline!r}")

    # --- t-critical value selection ---
    if n_subjects <= 10:
        t_crit = 1.833
    elif n_subjects <= 20:
        t_crit = 1.729
    else:
        t_crit = 1.645

    # --- Standard error in log space ---
    se_auc = within_subject_cv_auc / math.sqrt(n_subjects)
    se_cmax = within_subject_cv_cmax / math.sqrt(n_subjects)

    # --- 90% CI in log space, then exponentiate ---
    log_auc = math.log(auc_gmr)
    auc_90ci_lower = math.exp(log_auc - t_crit * se_auc)
    auc_90ci_upper = math.exp(log_auc + t_crit * se_auc)

    log_cmax = math.log(cmax_gmr)
    cmax_90ci_lower = math.exp(log_cmax - t_crit * se_cmax)
    cmax_90ci_upper = math.exp(log_cmax + t_crit * se_cmax)

    # --- Standard BE pass (80-125%) ---
    auc_be_pass = (auc_90ci_lower >= BE_LOWER_STD) and (auc_90ci_upper <= BE_UPPER_STD)
    cmax_be_pass = (cmax_90ci_lower >= BE_LOWER_STD) and (cmax_90ci_upper <= BE_UPPER_STD)
    overall_be_pass = auc_be_pass and cmax_be_pass

    # --- NTI criteria (90-111.11%) ---
    if nti_drug:
        nti_be_pass = (
            auc_90ci_lower >= NTI_LOWER
            and auc_90ci_upper <= NTI_UPPER
            and cmax_90ci_lower >= NTI_LOWER
            and cmax_90ci_upper <= NTI_UPPER
        )
    else:
        nti_be_pass = False

    # --- Recommendation ---
    if overall_be_pass:
        recommendation = "Bioequivalent - approve"
    elif not auc_be_pass and not cmax_be_pass:
        recommendation = "Failed BE - both AUC and Cmax outside limits"
    elif not auc_be_pass:
        recommendation = "Failed BE - AUC outside limits"
    else:
        recommendation = "Failed BE - Cmax outside limits"

    # --- Notes ---
    notes_parts = [
        f"Guideline: {guideline}",
        f"Standard limits: {BE_LOWER_STD * 100:.0f}-{BE_UPPER_STD * 100:.0f}%",
    ]
    if nti_drug:
        notes_parts.append(f"NTI limits applied: {NTI_LOWER * 100:.0f}-{NTI_UPPER * 100:.4f}%")
    if nti_drug and not nti_be_pass:
        notes_parts.append("NTI criteria NOT met")
    elif nti_drug and nti_be_pass:
        notes_parts.append("NTI criteria met")
    notes = "; ".join(notes_parts)

    return BEAcceptanceCriteriaResult(
        drug_name=drug_name,
        test_formulation=test_form,
        reference_formulation=ref_form,
        auc_ratio=auc_gmr,
        cmax_ratio=cmax_gmr,
        auc_90ci_lower=auc_90ci_lower,
        auc_90ci_upper=auc_90ci_upper,
        cmax_90ci_lower=cmax_90ci_lower,
        cmax_90ci_upper=cmax_90ci_upper,
        auc_be_pass=auc_be_pass,
        cmax_be_pass=cmax_be_pass,
        overall_be_pass=overall_be_pass,
        nti_drug=nti_drug,
        nti_be_pass=nti_be_pass,
        guideline=guideline,
        recommendation=recommendation,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------


def batch_be_evaluation(studies: list) -> list:
    """Evaluate a list of BE studies and return sorted results.

    Each study dict should have the same keys as ``evaluate_be_criteria``
    parameters.

    Results are sorted by overall_be_pass=True first, then by proximity of
    auc_ratio to 1.0 (ascending |auc_ratio - 1|).

    Parameters
    ----------
    studies:
        List of dicts, each containing keys matching ``evaluate_be_criteria``
        parameters.

    Returns
    -------
    list of BEAcceptanceCriteriaResult
    """
    results = []
    for study in studies:
        result = evaluate_be_criteria(
            drug_name=study["drug_name"],
            test_form=study["test_form"],
            ref_form=study["ref_form"],
            auc_gmr=study["auc_gmr"],
            cmax_gmr=study["cmax_gmr"],
            n_subjects=study["n_subjects"],
            within_subject_cv_auc=study["within_subject_cv_auc"],
            within_subject_cv_cmax=study["within_subject_cv_cmax"],
            nti_drug=study.get("nti_drug", False),
            guideline=study.get("guideline", "FDA"),
        )
        results.append(result)

    # Sort: passing studies first, then by |auc_ratio - 1.0| ascending
    results.sort(key=lambda r: (not r.overall_be_pass, abs(r.auc_ratio - 1.0)))
    return results
