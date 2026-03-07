"""Dose proportionality testing using the power model (Phase 401).

Implements the Smith et al. (2000) power model approach for assessing whether
AUC increases proportionally with dose.

References
----------
- Smith BP et al. J Clin Pharmacol 2000;40:1057-1073.
- FDA Guidance: Bioavailability and Bioequivalence Studies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "DoseProportionalityResult",
    "power_model_test",
    "normalized_auc_test",
]


@dataclass(frozen=True)
class DoseProportionalityResult:
    """Results from dose proportionality power model analysis.

    Attributes
    ----------
    doses_mg : Dose levels tested (mg).
    auc_values : AUC measurements corresponding to each dose (mg*h/L or any unit).
    b_exponent : Estimated exponent b from power model AUC = a * Dose^b.
    b_ci_lower : Lower bound of confidence interval for b.
    b_ci_upper : Upper bound of confidence interval for b.
    a_coefficient : Estimated coefficient a from power model.
    r_squared : R-squared of log-log linear fit.
    is_proportional : True if CI for b includes 1.0.
    proportionality_verdict : Descriptive verdict string.
    cv_dn_auc_pct : CV% of dose-normalised AUC across groups.
    notes : Additional notes about the analysis.
    """

    doses_mg: tuple[float, ...]
    auc_values: tuple[float, ...]
    b_exponent: float
    b_ci_lower: float
    b_ci_upper: float
    a_coefficient: float
    r_squared: float
    is_proportional: bool
    proportionality_verdict: str
    cv_dn_auc_pct: float
    notes: str


def _t_value(df: int, confidence_level: float = 0.90) -> float:
    """Approximate t critical value for two-tailed CI.

    Uses the approximation: t ≈ 1.645 + 0.01*(20-df) for df>=5, else 2.0.
    Adjusts z_base for non-90% CI using a linear mapping.
    """
    if df < 2:
        return 2.0
    # Base z for 90% CI (two-tailed, alpha=0.10)
    alpha = 1.0 - confidence_level
    # Map alpha to z via normal approximation (rational approximation)
    p = 1.0 - alpha / 2.0
    # Rational approximation of normal quantile (Abramowitz & Stegun 26.2.17)
    t_val = p - 0.5
    if abs(t_val) < 1e-9:
        z = 0.0
    else:
        sign = 1.0 if t_val > 0 else -1.0
        u = math.log(1.0 / (p * (1.0 - p)))
        z = sign * math.sqrt(u - (0.131538 + 0.189269 * u + 0.001308 * u * u))

    if df < 5:
        return max(2.0, z)
    # t approximation based on specification
    t_approx = z + 0.01 * (20 - df)
    return max(z, t_approx)


def _ols_log_log(
    doses_mg: list[float], auc_values: list[float]
) -> tuple[float, float, float, float]:
    """Fit log(AUC) = log(a) + b*log(Dose) via OLS.

    Returns
    -------
    (a, b, r_squared, se_b) where a is the coefficient, b is the exponent,
    r_squared is R² of the log-log fit, and se_b is the standard error of b.
    """
    n = len(doses_mg)
    log_doses = [math.log(d) for d in doses_mg]
    log_aucs = [math.log(a) for a in auc_values]

    mean_x = sum(log_doses) / n
    mean_y = sum(log_aucs) / n

    sxx = sum((x - mean_x) ** 2 for x in log_doses)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_doses, log_aucs, strict=True))
    syy = sum((y - mean_y) ** 2 for y in log_aucs)

    if sxx < 1e-15:
        raise ValueError("All dose values are identical; cannot fit power model.")

    b = sxy / sxx
    log_a = mean_y - b * mean_x
    a = math.exp(log_a)

    # R²
    ss_res = sum((y - (log_a + b * x)) ** 2 for x, y in zip(log_doses, log_aucs, strict=True))
    r_squared = 1.0 - ss_res / syy if syy > 1e-15 else 1.0

    # SE of b
    if n > 2:
        s2 = ss_res / (n - 2)
        se_b = math.sqrt(s2 / sxx)
    else:
        se_b = 0.0

    return a, b, r_squared, se_b


def power_model_test(
    doses_mg: list[float],
    auc_values: list[float],
    confidence_level: float = 0.90,
) -> DoseProportionalityResult:
    """Test dose proportionality using the power model AUC = a * Dose^b.

    Fits log(AUC) = log(a) + b*log(Dose) via OLS and computes a confidence
    interval for b. Dose proportionality is concluded when the CI includes 1.0.

    Parameters
    ----------
    doses_mg : Dose levels (mg). Must have >=3 values, all positive.
    auc_values : AUC at each dose. Must match length of doses_mg, all positive.
    confidence_level : Confidence level for the CI on b (default 0.90).

    Returns
    -------
    DoseProportionalityResult with power model fit statistics and verdict.

    Raises
    ------
    ValueError : If inputs are invalid.
    """
    if len(doses_mg) < 3:
        raise ValueError("At least 3 dose-AUC pairs are required for power model fit.")
    if len(doses_mg) != len(auc_values):
        raise ValueError("doses_mg and auc_values must have the same length.")
    if any(d <= 0 for d in doses_mg):
        raise ValueError("All doses_mg values must be positive.")
    if any(a <= 0 for a in auc_values):
        raise ValueError("All auc_values must be positive.")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError("confidence_level must be between 0 and 1 (exclusive).")

    n = len(doses_mg)
    df = n - 2

    a, b, r_squared, se_b = _ols_log_log(doses_mg, auc_values)

    t_crit = _t_value(df, confidence_level)
    margin = t_crit * se_b
    b_ci_lower = b - margin
    b_ci_upper = b + margin

    is_proportional = b_ci_lower <= 1.0 <= b_ci_upper

    # CV% of dose-normalised AUC
    dn_aucs = [a_val / d for a_val, d in zip(auc_values, doses_mg, strict=True)]
    mean_dn = sum(dn_aucs) / n
    if mean_dn > 1e-15 and n > 1:
        std_dn = math.sqrt(sum((x - mean_dn) ** 2 for x in dn_aucs) / (n - 1))
        cv_pct = 100.0 * std_dn / mean_dn
    else:
        cv_pct = 0.0

    if is_proportional:
        verdict = "dose_proportional"
    elif b > 1.0:
        verdict = "supra_proportional (b > 1.0)"
    else:
        verdict = "sub_proportional (b < 1.0)"

    notes = (
        f"Power model: AUC = {a:.4g} * Dose^{b:.4f}; "
        f"{int(confidence_level * 100)}% CI for b: [{b_ci_lower:.4f}, {b_ci_upper:.4f}]; "
        f"n={n}, df={df}"
    )

    return DoseProportionalityResult(
        doses_mg=tuple(doses_mg),
        auc_values=tuple(auc_values),
        b_exponent=b,
        b_ci_lower=b_ci_lower,
        b_ci_upper=b_ci_upper,
        a_coefficient=a,
        r_squared=r_squared,
        is_proportional=is_proportional,
        proportionality_verdict=verdict,
        cv_dn_auc_pct=cv_pct,
        notes=notes,
    )


def normalized_auc_test(
    doses_mg: list[float],
    auc_values: list[float],
) -> dict[str, object]:
    """Test dose proportionality using dose-normalised AUC CV%.

    Parameters
    ----------
    doses_mg : Dose levels (mg). All must be positive.
    auc_values : AUC at each dose. Must match length of doses_mg, all positive.

    Returns
    -------
    dict with keys:
      - dose_normalized_aucs : list of AUC/dose values
      - cv_pct : CV% of dose-normalized AUCs
      - is_proportional : True if CV% < 25%
      - mean_dn_auc : Mean dose-normalized AUC

    Raises
    ------
    ValueError : If inputs are invalid.
    """
    if len(doses_mg) < 2:
        raise ValueError("At least 2 dose-AUC pairs are required.")
    if len(doses_mg) != len(auc_values):
        raise ValueError("doses_mg and auc_values must have the same length.")
    if any(d <= 0 for d in doses_mg):
        raise ValueError("All doses_mg values must be positive.")
    if any(a <= 0 for a in auc_values):
        raise ValueError("All auc_values must be positive.")

    n = len(doses_mg)
    dn_aucs = [a_val / d for a_val, d in zip(auc_values, doses_mg, strict=True)]
    mean_dn = sum(dn_aucs) / n

    if n > 1:
        std_dn = math.sqrt(sum((x - mean_dn) ** 2 for x in dn_aucs) / (n - 1))
        cv_pct = 100.0 * std_dn / mean_dn if mean_dn > 1e-15 else 0.0
    else:
        cv_pct = 0.0

    return {
        "dose_normalized_aucs": dn_aucs,
        "cv_pct": cv_pct,
        "is_proportional": cv_pct < 25.0,
        "mean_dn_auc": mean_dn,
    }
