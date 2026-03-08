"""Phase 436 — Bootstrap-based PK Parameter Uncertainty Quantification.

Provides confidence intervals for NCA-derived PK parameters (AUC, Cmax,
t_half, CL) via bootstrap resampling. Pure Python stdlib only.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

_VALID_PARAMETERS = {"AUC", "Cmax", "t_half", "CL"}


@dataclass(frozen=True)
class PKUncertaintyResult:
    """Result of bootstrap-based PK parameter uncertainty quantification."""

    drug_name: str
    parameter_name: str
    point_estimate: float
    ci_lower_95: float
    ci_upper_95: float
    cv_pct: float
    n_bootstrap: int
    n_samples: int
    uncertainty_class: str
    notes: str


def _bootstrap_ci(
    values: list,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple:
    """Compute bootstrap 95% CI and CV% for the mean of values.

    Parameters
    ----------
    values:
        Observed data points.
    n_bootstrap:
        Number of bootstrap resamples.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    Tuple of (lower_95, upper_95, cv_pct).
    """
    rng = random.Random(seed)
    n = len(values)

    bootstrap_means = []
    for _ in range(n_bootstrap):
        resample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        bootstrap_means.append(sum(resample) / n)

    bootstrap_means.sort()

    lower_idx = int(round(0.025 * n_bootstrap))
    upper_idx = int(round(0.975 * n_bootstrap))
    # Clamp indices
    lower_idx = max(0, min(lower_idx, n_bootstrap - 1))
    upper_idx = max(0, min(upper_idx, n_bootstrap - 1))

    lower_95 = bootstrap_means[lower_idx]
    upper_95 = bootstrap_means[upper_idx]

    # CV% = std(bootstrap_means) / mean(bootstrap_means) * 100
    mean_b = sum(bootstrap_means) / n_bootstrap
    var_b = sum((x - mean_b) ** 2 for x in bootstrap_means) / n_bootstrap
    std_b = var_b**0.5
    cv_pct = (std_b / abs(mean_b)) * 100.0 if mean_b != 0.0 else 0.0

    return lower_95, upper_95, cv_pct


def _classify_uncertainty(cv_pct: float) -> str:
    """Classify uncertainty based on CV%."""
    if cv_pct < 15.0:
        return "low"
    elif cv_pct <= 30.0:
        return "moderate"
    else:
        return "high"


def quantify_pk_uncertainty(
    drug_name: str,
    parameter_name: str,
    observed_values: list,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> PKUncertaintyResult:
    """Quantify PK parameter uncertainty via bootstrap.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    parameter_name:
        PK parameter name. Must be one of: AUC, Cmax, t_half, CL.
    observed_values:
        List of observed parameter values (e.g., from multiple subjects).
    n_bootstrap:
        Number of bootstrap resamples (default 1000).
    seed:
        Random seed for reproducibility (default 42).

    Returns
    -------
    PKUncertaintyResult with point estimate, 95% CI, and CV%.
    """
    if len(observed_values) < 3:
        raise ValueError(f"Need at least 3 observations, got {len(observed_values)}")
    if parameter_name not in _VALID_PARAMETERS:
        raise ValueError(
            f"parameter_name must be one of {sorted(_VALID_PARAMETERS)}, got '{parameter_name}'"
        )

    point_estimate = sum(observed_values) / len(observed_values)

    ci_lower, ci_upper, cv_pct = _bootstrap_ci(observed_values, n_bootstrap, seed)

    uncertainty_class = _classify_uncertainty(cv_pct)

    notes = (
        f"{parameter_name} uncertainty for {drug_name}: "
        f"n={len(observed_values)}, "
        f"point_estimate={point_estimate:.4g}, "
        f"95% CI [{ci_lower:.4g}, {ci_upper:.4g}], "
        f"CV={cv_pct:.1f}% ({uncertainty_class})."
    )

    return PKUncertaintyResult(
        drug_name=drug_name,
        parameter_name=parameter_name,
        point_estimate=point_estimate,
        ci_lower_95=ci_lower,
        ci_upper_95=ci_upper,
        cv_pct=cv_pct,
        n_bootstrap=n_bootstrap,
        n_samples=len(observed_values),
        uncertainty_class=uncertainty_class,
        notes=notes,
    )


def compare_pk_uncertainty(
    drug_name: str,
    pk_data_dict: dict,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list:
    """Quantify uncertainty for multiple PK parameters and sort by CV%.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    pk_data_dict:
        Dict mapping parameter_name -> list of observed values.
    n_bootstrap:
        Number of bootstrap resamples.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    List of PKUncertaintyResult sorted by cv_pct ascending.
    """
    results = []
    for param_name, values in pk_data_dict.items():
        result = quantify_pk_uncertainty(drug_name, param_name, values, n_bootstrap, seed)
        results.append(result)

    return sorted(results, key=lambda r: r.cv_pct)
