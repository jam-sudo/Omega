"""Phase 253: Bioequivalence outcome predictor for generic drug formulations.

Predicts 90% confidence intervals for AUC and Cmax ratios (test/reference)
using log-normal BE statistics, evaluates FDA 80-125% BE criterion.

Reference: FDA Guidance for Industry - Statistical Approaches to Establishing
Bioequivalence (2001); FDA BE Guidance (2014).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BioequivalenceResult:
    """Result of bioequivalence prediction for a generic formulation."""

    drug_name: str
    test_formulation: str
    reference_formulation: str
    auc_test: float
    auc_reference: float
    cmax_test: float
    cmax_reference: float
    auc_ratio: float
    cmax_ratio: float
    auc_ci_lower: float
    auc_ci_upper: float
    cmax_ci_lower: float
    cmax_ci_upper: float
    be_conclusion: str  # "BE" or "Not_BE"
    auc_within_limits: bool
    cmax_within_limits: bool
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_BE_LOWER = 0.80
_BE_UPPER = 1.25
_T_VALUE = 1.796  # t-value at 80% CI, df=11, giving 90% CI for n=12 crossover


def _compute_ci(ratio: float, cv_intra_frac: float, n: int) -> tuple[float, float]:
    """Compute 90% CI for a ratio using log-normal BE assumption.

    ci_lower = ratio * exp(-1.796 * CV_intra / sqrt(n))
    ci_upper = ratio * exp(+1.796 * CV_intra / sqrt(n))
    """
    half_width = _T_VALUE * cv_intra_frac / math.sqrt(n)
    ci_lower = ratio * math.exp(-half_width)
    ci_upper = ratio * math.exp(half_width)
    return ci_lower, ci_upper


def _within_be_limits(ci_lower: float, ci_upper: float) -> bool:
    """Return True if the 90% CI is fully within [0.80, 1.25]."""
    return ci_lower >= _BE_LOWER and ci_upper <= _BE_UPPER


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_bioequivalence(
    drug_name: str,
    test_formulation: str,
    reference_formulation: str,
    auc_test: float,
    auc_reference: float,
    cmax_test: float,
    cmax_reference: float,
    cv_intra_pct: float = 15.0,
    n_subjects: int = 12,
) -> BioequivalenceResult:
    """Predict bioequivalence outcome for a test versus reference formulation.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    test_formulation:
        Name/description of the test (generic) formulation.
    reference_formulation:
        Name/description of the reference (innovator) formulation.
    auc_test:
        AUC for the test formulation (any consistent units, e.g. mg*h/L).
    auc_reference:
        AUC for the reference formulation (same units).
    cmax_test:
        Cmax for the test formulation (any consistent units).
    cmax_reference:
        Cmax for the reference formulation (same units).
    cv_intra_pct:
        Intra-subject coefficient of variation in percent. Default 15.0.
    n_subjects:
        Number of subjects in the crossover BE study. Default 12.

    Returns
    -------
    BioequivalenceResult
    """
    # Input validation
    if auc_test <= 0:
        raise ValueError("auc_test must be > 0")
    if auc_reference <= 0:
        raise ValueError("auc_reference must be > 0")
    if cmax_test <= 0:
        raise ValueError("cmax_test must be > 0")
    if cmax_reference <= 0:
        raise ValueError("cmax_reference must be > 0")
    if not (0 < cv_intra_pct < 100):
        raise ValueError("cv_intra_pct must be in (0, 100)")
    if n_subjects < 6:
        raise ValueError("n_subjects must be >= 6")

    cv_intra_frac = cv_intra_pct / 100.0

    auc_ratio = auc_test / auc_reference
    cmax_ratio = cmax_test / cmax_reference

    auc_ci_lower, auc_ci_upper = _compute_ci(auc_ratio, cv_intra_frac, n_subjects)
    cmax_ci_lower, cmax_ci_upper = _compute_ci(cmax_ratio, cv_intra_frac, n_subjects)

    auc_within_limits = _within_be_limits(auc_ci_lower, auc_ci_upper)
    cmax_within_limits = _within_be_limits(cmax_ci_lower, cmax_ci_upper)

    be_conclusion = "BE" if (auc_within_limits and cmax_within_limits) else "Not_BE"

    notes_parts: list[str] = []
    if not auc_within_limits:
        notes_parts.append(
            f"AUC 90% CI [{auc_ci_lower:.3f}, {auc_ci_upper:.3f}] outside [0.80, 1.25]"
        )
    if not cmax_within_limits:
        notes_parts.append(
            f"Cmax 90% CI [{cmax_ci_lower:.3f}, {cmax_ci_upper:.3f}] outside [0.80, 1.25]"
        )
    if cv_intra_pct > 30:
        notes_parts.append(
            f"High intra-subject CV ({cv_intra_pct:.1f}%); consider reference-scaled approach"
        )
    if not notes_parts:
        notes_parts.append("Both AUC and Cmax 90% CIs within [0.80, 1.25]; BE demonstrated")

    return BioequivalenceResult(
        drug_name=drug_name,
        test_formulation=test_formulation,
        reference_formulation=reference_formulation,
        auc_test=auc_test,
        auc_reference=auc_reference,
        cmax_test=cmax_test,
        cmax_reference=cmax_reference,
        auc_ratio=auc_ratio,
        cmax_ratio=cmax_ratio,
        auc_ci_lower=auc_ci_lower,
        auc_ci_upper=auc_ci_upper,
        cmax_ci_lower=cmax_ci_lower,
        cmax_ci_upper=cmax_ci_upper,
        be_conclusion=be_conclusion,
        auc_within_limits=auc_within_limits,
        cmax_within_limits=cmax_within_limits,
        notes="; ".join(notes_parts),
    )


def screen_formulations(
    drug_name: str,
    reference_auc: float,
    reference_cmax: float,
    test_scenarios: list[dict],
) -> list[BioequivalenceResult]:
    """Screen multiple test formulations against a reference.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    reference_auc:
        Reference formulation AUC.
    reference_cmax:
        Reference formulation Cmax.
    test_scenarios:
        List of dicts with keys "formulation", "auc", "cmax".

    Returns
    -------
    list[BioequivalenceResult]
        Sorted: BE results first, then by closeness of auc_ratio to 1.0.
    """
    results: list[BioequivalenceResult] = []
    for scenario in test_scenarios:
        formulation = scenario["formulation"]
        auc = scenario["auc"]
        cmax = scenario["cmax"]
        result = predict_bioequivalence(
            drug_name=drug_name,
            test_formulation=formulation,
            reference_formulation="Reference",
            auc_test=auc,
            auc_reference=reference_auc,
            cmax_test=cmax,
            cmax_reference=reference_cmax,
        )
        results.append(result)

    def _sort_key(r: BioequivalenceResult) -> tuple[int, float]:
        be_priority = 0 if r.be_conclusion == "BE" else 1
        auc_closeness = abs(r.auc_ratio - 1.0)
        return (be_priority, auc_closeness)

    results.sort(key=_sort_key)
    return results


__all__ = [
    "BioequivalenceResult",
    "predict_bioequivalence",
    "screen_formulations",
]
