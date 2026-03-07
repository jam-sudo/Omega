"""
Phase 1064 — Drug Bioequivalence Statistical Simulator

Statistical simulation module for bioequivalence (BE) assessment using
2-period crossover TOST framework. Distinct from be_study_design.py (design).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BioequivalenceResult:
    """Result of a bioequivalence simulation."""

    reference_name: str
    test_name: str
    n_subjects: int
    gm_ratio_auc: float  # geometric mean ratio AUC (Test/Ref)
    gm_ratio_cmax: float  # geometric mean ratio Cmax
    ci_lower_auc: float  # 90% CI lower bound
    ci_upper_auc: float  # 90% CI upper bound
    ci_lower_cmax: float
    ci_upper_cmax: float
    be_conclusion_auc: bool  # True if CI within [0.80, 1.25]
    be_conclusion_cmax: bool
    overall_be: bool  # both AUC and Cmax must pass
    power_estimate: float  # 0-1 estimated power of the study
    intra_subject_cv_auc: float  # intra-subject CV (%)
    notes: str


def _sigma_w(cv_pct: float) -> float:
    """Log-normal intra-subject SD from CV%."""
    cv = cv_pct / 100.0
    return math.sqrt(2.0 * math.log(1.0 + cv * cv))


def _standard_error(sigma: float, n: int) -> float:
    """Standard error of log-ratio for 2-period crossover."""
    return sigma * math.sqrt(2.0 / n)


def _t_critical(df: int) -> float:
    """Approximate t critical value for 90% CI (alpha=0.05 each side)."""
    return 1.645 + 1.645 / math.sqrt(df)


def _compute_ci(gm_ratio: float, cv_pct: float, n: int) -> tuple[float, float]:
    """Compute 90% CI for a single PK metric."""
    sigma = _sigma_w(cv_pct)
    se = _standard_error(sigma, n)
    df = n - 1
    t_crit = _t_critical(df)
    log_gm = math.log(gm_ratio)
    log_lower = log_gm - t_crit * se
    log_upper = log_gm + t_crit * se
    return math.exp(log_lower), math.exp(log_upper)


def _power_metric(gm_ratio: float, intra_cv_pct: float, n: int) -> float:
    """Estimate power for a single PK metric."""
    if 0.90 <= gm_ratio <= 1.11 and n > 12:
        power = max(0.5, 1.0 - (intra_cv_pct / 100.0) ** 2 * 10.0 / n)
    elif n < 8:
        power = 0.2
    else:
        power = max(0.3, 0.8 - abs(math.log(gm_ratio)) * 5.0)
    return power


def simulate_bioequivalence(
    reference_name: str,
    test_name: str,
    n_subjects: int,
    gm_ratio_auc: float = 1.0,
    gm_ratio_cmax: float = 1.0,
    intra_cv_auc_pct: float = 20.0,
    intra_cv_cmax_pct: float = 25.0,
    be_limits: tuple = (0.80, 1.25),
) -> BioequivalenceResult:
    """Simulate a 2-period crossover bioequivalence study using TOST.

    Parameters
    ----------
    reference_name:
        Name of the reference (innovator) product.
    test_name:
        Name of the test (generic) product.
    n_subjects:
        Number of subjects enrolled (must be >= 2).
    gm_ratio_auc:
        True geometric mean Test/Ref AUC ratio (must be > 0).
    gm_ratio_cmax:
        True geometric mean Test/Ref Cmax ratio (must be > 0).
    intra_cv_auc_pct:
        Intra-subject coefficient of variation for AUC in % (must be > 0).
    intra_cv_cmax_pct:
        Intra-subject coefficient of variation for Cmax in % (must be > 0).
    be_limits:
        Tuple (lower, upper) for BE acceptance limits; default FDA/EMA (0.80, 1.25).

    Returns
    -------
    BioequivalenceResult
    """
    # Validation
    if n_subjects < 2:
        raise ValueError(f"n_subjects must be >= 2, got {n_subjects}")
    if gm_ratio_auc <= 0:
        raise ValueError(f"gm_ratio_auc must be > 0, got {gm_ratio_auc}")
    if gm_ratio_cmax <= 0:
        raise ValueError(f"gm_ratio_cmax must be > 0, got {gm_ratio_cmax}")
    if intra_cv_auc_pct <= 0:
        raise ValueError(f"intra_cv_auc_pct must be > 0, got {intra_cv_auc_pct}")
    if intra_cv_cmax_pct <= 0:
        raise ValueError(f"intra_cv_cmax_pct must be > 0, got {intra_cv_cmax_pct}")
    if len(be_limits) != 2:
        raise ValueError(f"be_limits must be a tuple of 2 values, got {be_limits}")
    lim_lo, lim_hi = float(be_limits[0]), float(be_limits[1])
    if lim_lo >= lim_hi:
        raise ValueError(f"be_limits lower ({lim_lo}) must be < upper ({lim_hi})")

    # Compute 90% CIs
    ci_lo_auc, ci_hi_auc = _compute_ci(gm_ratio_auc, intra_cv_auc_pct, n_subjects)
    ci_lo_cmax, ci_hi_cmax = _compute_ci(gm_ratio_cmax, intra_cv_cmax_pct, n_subjects)

    # BE conclusions
    be_auc = ci_lo_auc >= lim_lo and ci_hi_auc <= lim_hi
    be_cmax = ci_lo_cmax >= lim_lo and ci_hi_cmax <= lim_hi
    overall_be = be_auc and be_cmax

    # Power estimation
    power_auc = _power_metric(gm_ratio_auc, intra_cv_auc_pct, n_subjects)
    power_cmax = _power_metric(gm_ratio_cmax, intra_cv_cmax_pct, n_subjects)
    power_estimate = min(0.99, max(0.01, power_auc * power_cmax))

    # Notes
    note_parts: list[str] = []
    note_parts.append(f"2-period crossover TOST; n={n_subjects}, df={n_subjects - 1}")
    note_parts.append(
        f"AUC: GMR={gm_ratio_auc:.3f}, 90% CI=[{ci_lo_auc:.3f}, {ci_hi_auc:.3f}], "
        f"BE={'PASS' if be_auc else 'FAIL'}"
    )
    note_parts.append(
        f"Cmax: GMR={gm_ratio_cmax:.3f}, 90% CI=[{ci_lo_cmax:.3f}, {ci_hi_cmax:.3f}], "
        f"BE={'PASS' if be_cmax else 'FAIL'}"
    )
    note_parts.append(
        f"Overall BE: {'PASS' if overall_be else 'FAIL'}; Estimated power={power_estimate:.2f}"
    )
    note_parts.append(f"BE limits: [{lim_lo:.2f}, {lim_hi:.2f}]")
    if not overall_be:
        note_parts.append("Consider increasing n or reformulating to bring GMR closer to 1.0")
    notes = "; ".join(note_parts)

    return BioequivalenceResult(
        reference_name=reference_name,
        test_name=test_name,
        n_subjects=n_subjects,
        gm_ratio_auc=gm_ratio_auc,
        gm_ratio_cmax=gm_ratio_cmax,
        ci_lower_auc=ci_lo_auc,
        ci_upper_auc=ci_hi_auc,
        ci_lower_cmax=ci_lo_cmax,
        ci_upper_cmax=ci_hi_cmax,
        be_conclusion_auc=be_auc,
        be_conclusion_cmax=be_cmax,
        overall_be=overall_be,
        power_estimate=power_estimate,
        intra_subject_cv_auc=intra_cv_auc_pct,
        notes=notes,
    )
