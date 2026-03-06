"""Bioequivalence assessment module (FDA 80-125% rule)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "BioequivalenceResult",
    "assess_bioequivalence",
    "simulate_be_study",
]


@dataclass
class BioequivalenceResult:
    """Results of a bioequivalence assessment."""

    drug_test: str
    drug_reference: str
    n_subjects: int
    auc_ratio_geometric_mean: float
    cmax_ratio_geometric_mean: float
    ci_90_auc: tuple[float, float]
    ci_90_cmax: tuple[float, float]
    be_auc: bool
    be_cmax: bool
    overall_be: bool
    study_design: str
    intra_subject_cv_auc_pct: float
    intra_subject_cv_cmax_pct: float


def _geometric_mean_ratio(
    test_values: list[float] | np.ndarray,
    ref_values: list[float] | np.ndarray,
) -> float:
    """Compute geometric mean ratio as exp(mean(log(test/ref)))."""
    test = np.asarray(test_values, dtype=float)
    ref = np.asarray(ref_values, dtype=float)
    log_ratios = np.log(test / ref)
    return float(np.exp(np.mean(log_ratios)))


def _90ci_ratio(
    test_values: list[float] | np.ndarray,
    ref_values: list[float] | np.ndarray,
    alpha: float = 0.1,
) -> tuple[float, float]:
    """Compute 90% CI for geometric mean ratio using log-transform.

    Uses normal approximation with z=1.645 for 90% CI (no scipy).
    """
    test = np.asarray(test_values, dtype=float)
    ref = np.asarray(ref_values, dtype=float)
    n = len(test)
    log_ratios = np.log(test / ref)
    mean_log = np.mean(log_ratios)
    se_log = np.std(log_ratios, ddof=1) / math.sqrt(n)
    # t-critical approximation for 90% CI (two-sided, alpha=0.10)
    # Using 1.645 as normal approximation per specification
    t_crit = 1.645
    lower = math.exp(mean_log - t_crit * se_log)
    upper = math.exp(mean_log + t_crit * se_log)
    return (lower, upper)


def _intra_cv(
    test_values: np.ndarray,
    ref_values: np.ndarray,
) -> float:
    """Compute intra-subject CV% from log-transformed ratios."""
    log_ratios = np.log(test_values / ref_values)
    cv = (math.exp(math.sqrt(float(np.var(log_ratios, ddof=1)))) - 1) * 100
    return cv


def assess_bioequivalence(
    drug_test: str,
    drug_reference: str,
    auc_test: list[float],
    auc_reference: list[float],
    cmax_test: Optional[list[float]] = None,
    cmax_reference: Optional[list[float]] = None,
) -> BioequivalenceResult:
    """Assess bioequivalence using FDA 80-125% rule.

    Parameters
    ----------
    drug_test:
        Name of the test drug formulation.
    drug_reference:
        Name of the reference drug formulation.
    auc_test:
        Per-subject AUC values for the test formulation.
    auc_reference:
        Per-subject AUC values for the reference formulation.
    cmax_test:
        Per-subject Cmax values for the test formulation. Defaults to auc_test
        if not provided (for testing purposes).
    cmax_reference:
        Per-subject Cmax values for the reference formulation. Defaults to
        auc_reference if not provided.

    Returns
    -------
    BioequivalenceResult

    Raises
    ------
    ValueError
        If fewer than 2 subjects or any value is <= 0.
    """
    n_auc = min(len(auc_test), len(auc_reference))
    if n_auc < 2:
        raise ValueError("At least 2 subjects required for bioequivalence assessment.")

    auc_t = np.asarray(auc_test[:n_auc], dtype=float)
    auc_r = np.asarray(auc_reference[:n_auc], dtype=float)

    if np.any(auc_t <= 0) or np.any(auc_r <= 0):
        raise ValueError("All AUC values must be positive (> 0).")

    if cmax_test is None:
        cmax_test = auc_test
    if cmax_reference is None:
        cmax_reference = auc_reference

    n_cmax = min(len(cmax_test), len(cmax_reference))
    if n_cmax < 2:
        raise ValueError("At least 2 subjects required for Cmax bioequivalence assessment.")

    cmax_t = np.asarray(cmax_test[:n_cmax], dtype=float)
    cmax_r = np.asarray(cmax_reference[:n_cmax], dtype=float)

    if np.any(cmax_t <= 0) or np.any(cmax_r <= 0):
        raise ValueError("All Cmax values must be positive (> 0).")

    n_subjects = min(n_auc, n_cmax)

    auc_gmr = _geometric_mean_ratio(auc_t, auc_r)
    cmax_gmr = _geometric_mean_ratio(cmax_t, cmax_r)

    ci_auc = _90ci_ratio(auc_t, auc_r)
    ci_cmax = _90ci_ratio(cmax_t, cmax_r)

    be_auc = 0.8 <= ci_auc[0] and ci_auc[1] <= 1.25
    be_cmax = 0.8 <= ci_cmax[0] and ci_cmax[1] <= 1.25
    overall_be = be_auc and be_cmax

    cv_auc = _intra_cv(auc_t, auc_r)
    cv_cmax = _intra_cv(cmax_t, cmax_r)

    return BioequivalenceResult(
        drug_test=drug_test,
        drug_reference=drug_reference,
        n_subjects=n_subjects,
        auc_ratio_geometric_mean=auc_gmr,
        cmax_ratio_geometric_mean=cmax_gmr,
        ci_90_auc=ci_auc,
        ci_90_cmax=ci_cmax,
        be_auc=be_auc,
        be_cmax=be_cmax,
        overall_be=overall_be,
        study_design="2x2_crossover",
        intra_subject_cv_auc_pct=cv_auc,
        intra_subject_cv_cmax_pct=cv_cmax,
    )


def simulate_be_study(
    drug_test: str,
    drug_reference: str,
    n_subjects: int = 24,
    true_ratio: float = 1.0,
    intra_cv: float = 0.2,
    seed: int = 42,
) -> BioequivalenceResult:
    """Simulate a 2x2 crossover bioequivalence study with lognormal AUC values.

    Parameters
    ----------
    drug_test:
        Name of the test drug formulation.
    drug_reference:
        Name of the reference drug formulation.
    n_subjects:
        Number of subjects in the study.
    true_ratio:
        True geometric mean ratio (test/reference) for AUC and Cmax.
    intra_cv:
        Intra-subject coefficient of variation (as a fraction, e.g. 0.2 = 20%).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    BioequivalenceResult
    """
    rng = np.random.default_rng(seed)

    # Log-scale parameters
    # CV = sqrt(exp(sigma^2) - 1) => sigma^2 = log(1 + CV^2)
    sigma = math.sqrt(math.log(1.0 + intra_cv**2))
    mu_ref = 0.0  # log-scale mean for reference (arbitrary baseline)
    mu_test = math.log(true_ratio)  # log-scale mean for test

    auc_reference = rng.lognormal(mean=mu_ref, sigma=sigma, size=n_subjects).tolist()
    auc_test = rng.lognormal(mean=mu_test, sigma=sigma, size=n_subjects).tolist()

    cmax_reference = rng.lognormal(mean=mu_ref, sigma=sigma, size=n_subjects).tolist()
    cmax_test = rng.lognormal(mean=mu_test, sigma=sigma, size=n_subjects).tolist()

    return assess_bioequivalence(
        drug_test=drug_test,
        drug_reference=drug_reference,
        auc_test=auc_test,
        auc_reference=auc_reference,
        cmax_test=cmax_test,
        cmax_reference=cmax_reference,
    )
