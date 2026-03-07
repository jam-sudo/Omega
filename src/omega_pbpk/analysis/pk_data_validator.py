"""PK data quality validator — checks concentration-time data for common errors."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "PKDataValidationResult",
    "validate_pk_data",
    "flag_outliers_grubbs",
    "compute_lloq_adjusted_auc",
    "check_dose_proportionality_consistency",
]


@dataclass(frozen=True)
class PKDataValidationResult:
    """Result of PK data quality validation."""

    n_timepoints: int
    n_issues: int
    errors: str  # "|"-joined error messages
    warnings: str  # "|"-joined warning messages
    outlier_indices: str  # ","-joined indices or ""
    data_quality_score: float  # 0-100
    is_acceptable: bool  # score >= 60
    notes: str


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (n - 1))


def flag_outliers_grubbs(values: list[float]) -> list[int]:
    """Grubbs test for outliers (simplified sequential).

    Returns indices of outliers where G > G_crit.
    G_crit = 2.5 for n < 10, 3.0 for n >= 10.
    """
    if len(values) < 3:
        return []

    g_crit = 2.5 if len(values) < 10 else 3.0
    m = _mean(values)
    s = _std(values)
    if s == 0.0:
        return []

    outliers = []
    for i, v in enumerate(values):
        g = abs(v - m) / s
        if g > g_crit:
            outliers.append(i)
    return outliers


def compute_lloq_adjusted_auc(
    times_h: list[float], concs_mg_L: list[float], lloq: float = 0.001
) -> float:
    """Compute trapezoidal AUC, replacing BLQ values (conc < lloq) with lloq/2."""
    adjusted = [lloq / 2 if c < lloq else c for c in concs_mg_L]
    auc = 0.0
    for i in range(1, len(times_h)):
        dt = times_h[i] - times_h[i - 1]
        auc += 0.5 * (adjusted[i] + adjusted[i - 1]) * dt
    return auc


def check_dose_proportionality_consistency(
    doses_mg: list[float],
    aucs: list[float],
    expected_exponent: float = 1.0,
    tolerance: float = 0.3,
) -> dict:
    """Check if AUC-dose relationship is consistent with expected exponent.

    Fits power law via log-log OLS.
    Returns dict: fitted_exponent, is_consistent, deviation.
    """
    if len(doses_mg) < 2 or len(aucs) < 2:
        return {"fitted_exponent": float("nan"), "is_consistent": False, "deviation": float("nan")}

    # Filter valid (positive) pairs
    pairs = [(d, a) for d, a in zip(doses_mg, aucs) if d > 0 and a > 0]
    if len(pairs) < 2:
        return {"fitted_exponent": float("nan"), "is_consistent": False, "deviation": float("nan")}

    log_d = [math.log(d) for d, _ in pairs]
    log_a = [math.log(a) for _, a in pairs]

    # OLS: log_a = slope * log_d + intercept
    n = len(pairs)
    sum_x = sum(log_d)
    sum_y = sum(log_a)
    sum_xy = sum(x * y for x, y in zip(log_d, log_a))
    sum_xx = sum(x * x for x in log_d)

    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-12:
        fitted_exponent = 1.0
    else:
        fitted_exponent = (n * sum_xy - sum_x * sum_y) / denom

    deviation = abs(fitted_exponent - expected_exponent)
    is_consistent = deviation < tolerance

    return {
        "fitted_exponent": round(fitted_exponent, 4),
        "is_consistent": is_consistent,
        "deviation": round(deviation, 4),
    }


def validate_pk_data(
    times_h: list[float],
    concs_mg_L: list[float],
    expected_route: str = "oral",
    dose_mg: float | None = None,
) -> PKDataValidationResult:
    """Validate PK concentration-time data for quality issues.

    Args:
        times_h: Time points in hours.
        concs_mg_L: Concentrations in mg/L.
        expected_route: "oral" or "iv".
        dose_mg: Optional nominal dose for reference.

    Returns:
        PKDataValidationResult with errors, warnings, outlier indices, and quality score.
    """
    if len(times_h) != len(concs_mg_L):
        raise ValueError(
            f"times_h (len={len(times_h)}) and concs_mg_L (len={len(concs_mg_L)}) "
            "must have the same length."
        )
    if len(times_h) < 2:
        raise ValueError("At least 2 data points required for validation.")

    errors_list: list[str] = []
    warnings_list: list[str] = []
    notes_parts: list[str] = []

    n = len(times_h)

    # 1. Monotone time check
    for i in range(1, n):
        if times_h[i] <= times_h[i - 1]:
            warnings_list.append(
                f"non-monotone time: t[{i}]={times_h[i]} not > t[{i - 1}]={times_h[i - 1]}"
            )
            break

    # 2. Non-negative concentrations
    for i, c in enumerate(concs_mg_L):
        if c < 0:
            errors_list.append(f"negative concentration at index {i}: {c} mg/L")

    # 3. Sufficient data points
    if n < 6:
        warnings_list.append(f"insufficient data points ({n}) for reliable NCA; recommend >= 6")

    # 4. Pre-dose / initial concentration check
    positive_concs = [c for c in concs_mg_L if c > 0]
    cmax = max(positive_concs) if positive_concs else 0.0

    if expected_route == "iv":
        if times_h[0] < 0:
            warnings_list.append("pre-dose sample detected (t < 0) for IV route")
    else:  # oral
        if times_h[0] < 0:
            warnings_list.append("pre-dose sample detected (t < 0)")
        # Oral: very high t=0 conc is suspicious (>50% Cmax at t=0)
        if cmax > 0 and concs_mg_L[0] > 0.5 * cmax and times_h[0] >= 0:
            notes_parts.append(
                f"concentration at t[0]={times_h[0]} h is {concs_mg_L[0]:.3g} mg/L "
                f"({100 * concs_mg_L[0] / cmax:.0f}% of Cmax); verify pre-dose baseline"
            )

    # 5. Outlier detection (on log-concs of positive values)
    outlier_indices: list[int] = []
    pos_indices = [i for i, c in enumerate(concs_mg_L) if c > 0]
    if len(pos_indices) >= 3:
        log_concs = [math.log(concs_mg_L[i]) for i in pos_indices]
        grubbs_hits = flag_outliers_grubbs(log_concs)
        outlier_indices = [pos_indices[j] for j in grubbs_hits]
        if outlier_indices:
            notes_parts.append(
                f"potential outlier(s) at index(es) {outlier_indices} by Grubbs test on log-concs"
            )

    # 6. Missing tmax for oral route — detect if profile is monotone decreasing after t=0
    if expected_route == "oral" and n >= 3:
        # Check if concs are strictly decreasing from index 0
        is_always_decreasing = all(concs_mg_L[i] <= concs_mg_L[i - 1] for i in range(1, n))
        if is_always_decreasing and concs_mg_L[0] > 0:
            warnings_list.append(
                "concentration appears monotone decreasing from t=0; "
                "may indicate missed tmax (early absorption phase)"
            )

    # 7. Below LOQ flag — zeros among otherwise positive data
    n_zero = sum(1 for c in concs_mg_L if c == 0)
    n_positive = sum(1 for c in concs_mg_L if c > 0)
    if n_zero > 0 and n_positive > 0:
        warnings_list.append(
            f"{n_zero} zero concentration(s) detected with other positive values; "
            "potential BLQ imputation"
        )

    # 8. Truncated data — last conc > 20% of Cmax
    if cmax > 0 and concs_mg_L[-1] > 0.2 * cmax:
        warnings_list.append(
            f"last concentration ({concs_mg_L[-1]:.3g} mg/L) is "
            f"{100 * concs_mg_L[-1] / cmax:.0f}% of Cmax; profile may be truncated"
        )

    # Compute quality score
    n_errors = len(errors_list)
    n_warnings = len(warnings_list)
    score = 100 - 20 * n_errors - 10 * n_warnings
    score = max(0, min(100, score))

    n_issues = n_errors + n_warnings
    is_acceptable = score >= 60

    errors_str = "|".join(errors_list)
    warnings_str = "|".join(warnings_list)
    outlier_str = ",".join(str(i) for i in outlier_indices)

    if not notes_parts:
        notes_parts.append("Validation complete")
    if dose_mg is not None:
        notes_parts.append(f"nominal dose={dose_mg} mg")

    notes = "; ".join(notes_parts)

    return PKDataValidationResult(
        n_timepoints=n,
        n_issues=n_issues,
        errors=errors_str,
        warnings=warnings_str,
        outlier_indices=outlier_str,
        data_quality_score=float(score),
        is_acceptable=is_acceptable,
        notes=notes,
    )
