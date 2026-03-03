"""Bioequivalence (BE) Study Design — sample size, power, and sensitivity analysis.

Supports 2x2 crossover, 3x3 Williams, and 4x2 replicate designs using
TOST (Two One-Sided Tests) power via the non-central t distribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import t as t_dist

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BE_LOWER: float = 0.80
BE_UPPER: float = 1.25
NTI_LOWER: float = 0.90
NTI_UPPER: float = 1.1111
DEFAULT_ALPHA: float = 0.05
DEFAULT_POWER: float = 0.80
DEFAULT_GMR: float = 0.95

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BEStudyDesign:
    """Result of BE sample-size calculation."""

    design: str
    n_subjects: int
    n_per_sequence: int
    power_achieved: float
    alpha: float
    be_limits: tuple[float, float]
    gmr_assumed: float
    cv_within: float
    cv_between: float
    endpoint: str
    dropout_adjusted_n: int
    dropout_rate: float
    power_curve: list[dict[str, float]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BEPowerResult:
    """Power for a fixed sample size across AUC and Cmax."""

    n_subjects: int
    power_auc: float
    power_cmax: float
    power_joint: float
    design: str
    alpha: float
    be_limits: tuple[float, float]
    gmr_auc: float
    gmr_cmax: float
    cv_auc: float
    cv_cmax: float


@dataclass(frozen=True)
class SensitivityResult:
    """Sample-size sensitivity grid over CV and GMR ranges."""

    cv_values: list[float]
    gmr_values: list[float]
    sample_size_matrix: list[list[int]]
    design: str
    target_power: float
    alpha: float
    be_limits: tuple[float, float]


# ---------------------------------------------------------------------------
# TOST power (non-central t)
# ---------------------------------------------------------------------------


def _tost_power(
    n_total: int,
    cv: float,
    gmr: float = DEFAULT_GMR,
    be_lower: float = BE_LOWER,
    be_upper: float = BE_UPPER,
    alpha: float = DEFAULT_ALPHA,
    design: str = "2x2",
) -> float:
    """Compute TOST power for a BE crossover study.

    Uses the central t-distribution interval probability approach:
    power = P(t_crit - delta_L < T < delta_U - t_crit) where T ~ t(df).
    """
    if n_total < 4 or cv <= 0:
        return 0.0

    sigma_w = math.sqrt(math.log(1 + cv**2))

    if design == "2x2":
        df = n_total - 2
        se = sigma_w * math.sqrt(2.0 / n_total)
    elif design == "3x3_williams":
        n_per_seq = n_total // 3
        if n_per_seq < 1:
            return 0.0
        df = 2 * n_total // 3 - 1
        se = sigma_w * math.sqrt(2.0 / n_total)
    elif design == "4x2_replicate":
        df = n_total - 2
        se = sigma_w * math.sqrt(1.0 / n_total)
    else:
        df = n_total - 2
        se = sigma_w * math.sqrt(2.0 / n_total)

    if df < 1 or se <= 0:
        return 0.0

    delta_l = (math.log(gmr) - math.log(be_lower)) / se
    delta_u = (math.log(be_upper) - math.log(gmr)) / se
    t_crit = float(t_dist.ppf(1 - alpha, df))

    # TOST power: probability that test statistic falls in the acceptance region
    # P(t_crit - delta_L < T < delta_U - t_crit) where T ~ t(df)
    upper_bound = delta_u - t_crit
    lower_bound = t_crit - delta_l

    if upper_bound <= lower_bound:
        return 0.0

    power = float(t_dist.cdf(upper_bound, df) - t_dist.cdf(lower_bound, df))

    return max(0.0, min(1.0, power))


# ---------------------------------------------------------------------------
# Sample size calculation
# ---------------------------------------------------------------------------


def calculate_sample_size(
    cv_within: float,
    gmr: float = DEFAULT_GMR,
    target_power: float = DEFAULT_POWER,
    alpha: float = DEFAULT_ALPHA,
    be_limits: tuple[float, float] = (BE_LOWER, BE_UPPER),
    design: str = "2x2",
    dropout_rate: float = 0.0,
    endpoint: str = "auc",
    cv_between: float = 0.0,
    max_n: int = 200,
) -> BEStudyDesign:
    """Find minimum sample size achieving target power for a BE study.

    Iterates from n=4 upward (step 2) until target power is achieved.
    """
    warnings: list[str] = []
    power_curve: list[dict[str, float]] = []
    best_n = max_n
    best_power = 0.0

    for n in range(4, max_n + 1, 2):
        pwr = _tost_power(n, cv_within, gmr, be_limits[0], be_limits[1], alpha, design)
        power_curve.append({"n": float(n), "power": round(pwr, 6)})
        if pwr >= target_power and best_n == max_n:
            best_n = n
            best_power = pwr
            break
        best_power = pwr

    if best_n == max_n and best_power < target_power:
        warnings.append(
            f"Target power {target_power:.2f} not achieved within max_n={max_n}; "
            f"best power={best_power:.4f} at n={max_n}"
        )

    if design == "3x3_williams":
        n_per_seq = best_n // 3
    else:
        n_per_seq = best_n // 2

    dropout_adjusted = math.ceil(best_n / (1 - dropout_rate)) if dropout_rate < 1.0 else best_n

    return BEStudyDesign(
        design=design,
        n_subjects=best_n,
        n_per_sequence=n_per_seq,
        power_achieved=round(best_power, 6),
        alpha=alpha,
        be_limits=be_limits,
        gmr_assumed=gmr,
        cv_within=cv_within,
        cv_between=cv_between,
        endpoint=endpoint,
        dropout_adjusted_n=dropout_adjusted,
        dropout_rate=dropout_rate,
        power_curve=power_curve,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Power for fixed N
# ---------------------------------------------------------------------------


def compute_power(
    n_subjects: int,
    cv_auc: float,
    cv_cmax: float,
    gmr_auc: float = DEFAULT_GMR,
    gmr_cmax: float = DEFAULT_GMR,
    alpha: float = DEFAULT_ALPHA,
    be_limits: tuple[float, float] = (BE_LOWER, BE_UPPER),
    design: str = "2x2",
) -> BEPowerResult:
    """Compute TOST power for AUC and Cmax at a fixed sample size."""
    power_auc = _tost_power(n_subjects, cv_auc, gmr_auc, be_limits[0], be_limits[1], alpha, design)
    power_cmax = _tost_power(
        n_subjects, cv_cmax, gmr_cmax, be_limits[0], be_limits[1], alpha, design
    )
    power_joint = min(power_auc, power_cmax)

    return BEPowerResult(
        n_subjects=n_subjects,
        power_auc=round(power_auc, 6),
        power_cmax=round(power_cmax, 6),
        power_joint=round(power_joint, 6),
        design=design,
        alpha=alpha,
        be_limits=be_limits,
        gmr_auc=gmr_auc,
        gmr_cmax=gmr_cmax,
        cv_auc=cv_auc,
        cv_cmax=cv_cmax,
    )


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------


def sensitivity_analysis(
    cv_range: tuple[float, float] = (0.10, 0.50),
    gmr_range: tuple[float, float] = (0.85, 1.00),
    n_cv_steps: int = 9,
    n_gmr_steps: int = 7,
    target_power: float = DEFAULT_POWER,
    alpha: float = DEFAULT_ALPHA,
    be_limits: tuple[float, float] = (BE_LOWER, BE_UPPER),
    design: str = "2x2",
) -> SensitivityResult:
    """Grid sensitivity: sample size over CV and GMR ranges."""
    cv_values = list(np.round(np.linspace(cv_range[0], cv_range[1], n_cv_steps), 4))
    gmr_values = list(np.round(np.linspace(gmr_range[0], gmr_range[1], n_gmr_steps), 4))

    matrix: list[list[int]] = []
    for cv in cv_values:
        row: list[int] = []
        for gmr in gmr_values:
            result = calculate_sample_size(
                cv_within=cv,
                gmr=gmr,
                target_power=target_power,
                alpha=alpha,
                be_limits=be_limits,
                design=design,
            )
            row.append(result.n_subjects)
        matrix.append(row)

    return SensitivityResult(
        cv_values=cv_values,
        gmr_values=gmr_values,
        sample_size_matrix=matrix,
        design=design,
        target_power=target_power,
        alpha=alpha,
        be_limits=be_limits,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_be_design(
    cv_auc: float,
    cv_cmax: float,
    gmr: float = DEFAULT_GMR,
    target_power: float = DEFAULT_POWER,
    alpha: float = DEFAULT_ALPHA,
    be_limits: tuple[float, float] = (BE_LOWER, BE_UPPER),
    design: str = "2x2",
    dropout_rate: float = 0.10,
    is_nti: bool = False,
) -> BEStudyDesign:
    """Main entry point for BE study design.

    Uses the larger of cv_auc and cv_cmax as the driving CV.
    If is_nti, narrows BE limits to 90.00%-111.11%.
    """
    if is_nti:
        be_limits = (NTI_LOWER, NTI_UPPER)

    driving_cv = max(cv_auc, cv_cmax)

    return calculate_sample_size(
        cv_within=driving_cv,
        gmr=gmr,
        target_power=target_power,
        alpha=alpha,
        be_limits=be_limits,
        design=design,
        dropout_rate=dropout_rate,
        endpoint="both",
    )
