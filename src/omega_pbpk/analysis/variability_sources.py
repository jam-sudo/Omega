"""Quantify contributions of different sources of PK variability.

Phase 391: Partition total PK variability into within-group and between-group components
using ANOVA-like decomposition. Provides bootstrap CI for CV%.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

__all__ = [
    "VariabilityResult",
    "partition_variability",
    "bootstrap_ci_cv",
]


@dataclass(frozen=True)
class VariabilityResult:
    """Result of PK variability partitioning.

    Attributes
    ----------
    param_name : str
        Name of the PK parameter analyzed.
    source_names : list[str]
        Names of variability sources (groups).
    n_groups : int
        Number of groups.
    within_cv_pct : float
        Pooled within-group CV% (sqrt of pooled within-group variance / grand mean * 100).
    between_cv_pct : float
        Between-group CV% (std of group means / grand mean * 100).
    total_cv_pct : float
        Total CV% across all observations.
    variance_fraction_within : float
        Fraction of total SS explained by within-group variation [0, 1].
    variance_fraction_between : float
        Fraction of total SS explained by between-group variation [0, 1].
    notes : str
        Descriptive notes.
    """

    param_name: str
    source_names: list[str]
    n_groups: int
    within_cv_pct: float
    between_cv_pct: float
    total_cv_pct: float
    variance_fraction_within: float
    variance_fraction_between: float
    notes: str


def _mean(values: list[float]) -> float:
    """Arithmetic mean of a list."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _var_ddof1(values: list[float]) -> float:
    """Sample variance (ddof=1)."""
    n = len(values)
    if n < 2:
        return 0.0
    m = _mean(values)
    return sum((x - m) ** 2 for x in values) / (n - 1)


def _std_ddof1(values: list[float]) -> float:
    """Sample standard deviation (ddof=1)."""
    return math.sqrt(_var_ddof1(values))


def _cv_pct(values: list[float]) -> float:
    """Coefficient of variation as percentage."""
    m = _mean(values)
    if m == 0.0 or len(values) < 2:
        return 0.0
    return _std_ddof1(values) / m * 100.0


def partition_variability(
    param_name: str,
    values_list: list[list[float]],
    source_names: list[str],
) -> VariabilityResult:
    """Partition total PK variability into within-group and between-group components.

    Uses ANOVA-like SS decomposition:
      - Total SS   = sum_ij (x_ij - grand_mean)^2
      - Between SS = sum_i  n_i * (group_mean_i - grand_mean)^2
      - Within SS  = Total SS - Between SS
      - variance_fraction_between = Between SS / Total SS
      - variance_fraction_within  = Within SS  / Total SS

    Parameters
    ----------
    param_name : str
        Name of the PK parameter (e.g., "CL", "Vd", "AUC").
    values_list : list[list[float]]
        One inner list per group; each inner list contains observed values.
    source_names : list[str]
        Names of the groups / variability sources (len must equal len(values_list)).

    Returns
    -------
    VariabilityResult
    """
    if not param_name or not param_name.strip():
        raise ValueError("param_name must be a non-empty string")
    if not values_list:
        raise ValueError("values_list must contain at least one group")
    if len(values_list) != len(source_names):
        raise ValueError(
            f"values_list has {len(values_list)} groups but source_names has "
            f"{len(source_names)} entries; they must match"
        )
    if not source_names:
        raise ValueError("source_names must be non-empty")

    for i, grp in enumerate(values_list):
        if not grp:
            raise ValueError(f"Group {i} ('{source_names[i]}') is empty; all groups must have "
                             "at least one value")
        for j, v in enumerate(grp):
            if not math.isfinite(v):
                raise ValueError(
                    f"Non-finite value {v} in group {i} ('{source_names[i]}'), index {j}"
                )

    # Flatten all values
    all_values: list[float] = []
    for grp in values_list:
        all_values.extend(grp)

    n_total = len(all_values)
    grand_mean = _mean(all_values)

    # Group means and sizes
    group_means = [_mean(grp) for grp in values_list]
    group_ns = [len(grp) for grp in values_list]

    # SS decomposition
    total_ss = sum((x - grand_mean) ** 2 for x in all_values)
    between_ss = sum(
        n_i * (gm - grand_mean) ** 2
        for n_i, gm in zip(group_ns, group_means, strict=True)
    )
    within_ss = max(0.0, total_ss - between_ss)

    if total_ss > 0.0:
        var_frac_between = between_ss / total_ss
        var_frac_within = within_ss / total_ss
    else:
        var_frac_between = 0.0
        var_frac_within = 1.0

    # Clamp to [0, 1] (numerical safety)
    var_frac_between = max(0.0, min(1.0, var_frac_between))
    var_frac_within = max(0.0, min(1.0, var_frac_within))

    # CV% metrics
    total_cv = _cv_pct(all_values)

    # Between-group CV: CV of group means (weighted by group size if unequal)
    # Simple: std of group means / grand_mean * 100
    between_cv: float
    if len(group_means) >= 2:
        between_cv = _cv_pct(group_means)
    else:
        between_cv = 0.0

    # Within-group CV: pooled
    # pooled_var = Within SS / (n_total - n_groups)
    n_groups = len(values_list)
    df_within = n_total - n_groups
    if df_within > 0 and grand_mean != 0.0:
        pooled_var = within_ss / df_within
        within_cv = math.sqrt(pooled_var) / abs(grand_mean) * 100.0
    else:
        within_cv = 0.0

    notes = (
        f"ANOVA decomposition for '{param_name}'. "
        f"N_total={n_total}, N_groups={n_groups}. "
        f"Grand mean={grand_mean:.4g}. "
        f"Total SS={total_ss:.4g}, Between SS={between_ss:.4g}, Within SS={within_ss:.4g}. "
        f"Between fraction={var_frac_between:.3f}, Within fraction={var_frac_within:.3f}."
    )

    return VariabilityResult(
        param_name=param_name,
        source_names=list(source_names),
        n_groups=n_groups,
        within_cv_pct=within_cv,
        between_cv_pct=between_cv,
        total_cv_pct=total_cv,
        variance_fraction_within=var_frac_within,
        variance_fraction_between=var_frac_between,
        notes=notes,
    )


def bootstrap_ci_cv(
    values: list[float],
    n_bootstrap: int = 500,
    ci_pct: float = 90.0,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the CV% of a sample.

    Uses simple resampling with replacement via Python's ``random`` module
    with a fixed seed of 42. No numpy.random is used.

    Parameters
    ----------
    values : list[float]
        Observed values. Must have at least 2 elements.
    n_bootstrap : int
        Number of bootstrap replications (default 500).
    ci_pct : float
        Confidence level percentage, e.g. 90.0 for 90% CI (default 90.0).

    Returns
    -------
    tuple[float, float]
        (lower_ci, upper_ci) in percent.
    """
    if len(values) < 2:
        raise ValueError("values must contain at least 2 elements for bootstrap CI")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be >= 1")
    if not (0.0 < ci_pct < 100.0):
        raise ValueError("ci_pct must be in (0, 100)")
    for i, v in enumerate(values):
        if not math.isfinite(v):
            raise ValueError(f"Non-finite value at index {i}: {v}")

    n = len(values)
    rng = random.Random(42)

    boot_cvs: list[float] = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(values) for _ in range(n)]
        boot_cvs.append(_cv_pct(sample))

    boot_cvs.sort()

    alpha = (100.0 - ci_pct) / 2.0 / 100.0
    lo_idx = max(0, int(math.floor(alpha * n_bootstrap)))
    hi_idx = min(n_bootstrap - 1, int(math.ceil((1.0 - alpha) * n_bootstrap)) - 1)

    return (boot_cvs[lo_idx], boot_cvs[hi_idx])
