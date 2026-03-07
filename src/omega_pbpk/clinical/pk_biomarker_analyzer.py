"""PK/PD biomarker analysis — Phase 466.

Analyze relationships between PK parameters and PD biomarker responses
using pure-Python implementations (no numpy/scipy).

References
----------
- Danhof M et al., Pharm Res. 2007;24(12):2221-2232.
- Holford NH & Sheiner LB, Clin Pharmacokinet. 1981;6(6):429-453.
- FDA Guidance on Exposure-Response Relationships. 2003.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ExposureResponseResult:
    """Result of exposure-response analysis across PK bins."""

    n_subjects: int
    n_bins: int
    bin_edges: list[float]
    bin_median_exposure: list[float]
    bin_mean_response: list[float]
    bin_n: list[int]
    overall_correlation_r: float
    monotonic: bool
    notes: str


def _mean(values: list[float]) -> float:
    """Compute arithmetic mean of a list."""
    if not values:
        raise ValueError("Cannot compute mean of empty list")
    return sum(values) / len(values)


def _median(values: list[float]) -> float:
    """Compute median of a sorted or unsorted list."""
    if not values:
        raise ValueError("Cannot compute median of empty list")
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2.0
    return float(s[mid])


def _pearson_r(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient using sum-of-products formula.

    r = Σ[(xi - x̄)(yi - ȳ)] / sqrt[Σ(xi - x̄)² · Σ(yi - ȳ)²]
    """
    n = len(x)
    if n != len(y):
        raise ValueError("x and y must have the same length")
    if n < 2:
        raise ValueError("At least 2 data points required for Pearson r")

    x_mean = _mean(x)
    y_mean = _mean(y)

    cov = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y, strict=True))
    ss_x = sum((xi - x_mean) ** 2 for xi in x)
    ss_y = sum((yi - y_mean) ** 2 for yi in y)

    denom = math.sqrt(ss_x * ss_y)
    if denom == 0.0:
        return 0.0  # one or both variables are constant
    return cov / denom


def _t_to_p_approx(t_stat: float, df: int) -> float:
    """Approximate two-sided p-value from t-statistic using normal approximation.

    For large df, t ≈ z (standard normal).  For small df we use a simple
    approximation based on the relationship between the t-distribution CDF
    and the regularised incomplete beta function via:
        p ≈ 2 * Φ(-|t|)   (normal approximation for df >= 30)
    For df < 30, we apply a Student-t correction factor.

    Uses the error function (math.erf) for the normal CDF.
    """
    abs_t = abs(t_stat)

    if df <= 0:
        return 1.0

    if df >= 30:
        # Normal approximation
        p_one_tail = 0.5 * math.erfc(abs_t / math.sqrt(2.0))
    else:
        # Simple approximation: scale t by sqrt(df/(df-2)) to match variance
        # then use normal CDF — good enough for an approximation
        if df == 1:
            # Cauchy distribution approximation
            p_one_tail = 0.5 - math.atan(abs_t) / math.pi
        elif df == 2:
            p_one_tail = 0.5 / (1.0 + abs_t / math.sqrt(2.0))
        else:
            # Wilson-Hilferty approximation
            x = df / (df + abs_t**2)
            # Beta distribution approximation
            p_one_tail = 0.5 * (x ** (df / 2.0))
            p_one_tail = max(0.0, min(0.5, p_one_tail))

    return min(1.0, 2.0 * p_one_tail)


def correlate_pk_biomarker(
    pk_values: list[float],
    biomarker_values: list[float],
) -> dict[str, float]:
    """Compute Pearson correlation between PK exposure and biomarker response.

    Parameters
    ----------
    pk_values:
        List of PK parameter values (e.g., AUC, Cmax) for each subject.
    biomarker_values:
        List of biomarker response values (same order/length as pk_values).

    Returns
    -------
    dict
        Keys: 'pearson_r', 'r_squared', 'slope', 'intercept', 'p_value_approx'.
    """
    if not pk_values or not biomarker_values:
        raise ValueError("pk_values and biomarker_values must not be empty")
    if len(pk_values) != len(biomarker_values):
        raise ValueError(
            f"pk_values length ({len(pk_values)}) must equal "
            f"biomarker_values length ({len(biomarker_values)})"
        )
    if len(pk_values) < 2:
        raise ValueError("At least 2 data points required")

    r = _pearson_r(pk_values, biomarker_values)
    r_sq = r**2

    # Linear regression coefficients (OLS)
    x_mean = _mean(pk_values)
    y_mean = _mean(biomarker_values)
    ss_x = sum((xi - x_mean) ** 2 for xi in pk_values)
    cov = sum(
        (xi - x_mean) * (yi - y_mean)
        for xi, yi in zip(pk_values, biomarker_values, strict=True)
    )

    slope = cov / ss_x if ss_x != 0.0 else 0.0
    intercept = y_mean - slope * x_mean

    # p-value approximation via t-distribution
    n = len(pk_values)
    df = n - 2
    if abs(r) >= 1.0:
        p_val = 0.0
    else:
        t_stat = r * math.sqrt(df) / math.sqrt(1.0 - r**2)
        p_val = _t_to_p_approx(t_stat, df)

    return {
        "pearson_r": r,
        "r_squared": r_sq,
        "slope": slope,
        "intercept": intercept,
        "p_value_approx": p_val,
    }


def calculate_pkpd_index(
    auc_values: list[float],
    mic_values: list[float],
    index_type: str = "auc_mic",
) -> list[float]:
    """Calculate PK/PD index values for each subject/pathogen combination.

    Parameters
    ----------
    auc_values:
        AUC values (mg·h/L or appropriate units) per subject.
    mic_values:
        Minimum inhibitory concentration (MIC) values in matched units.
        If a single value provided it is broadcast to all subjects.
    index_type:
        'auc_mic' → AUC/MIC ratio.
        'cmax_mic' → Cmax/MIC ratio (auc_values treated as Cmax in this case).
        't_mic' → %T>MIC; not applicable here, treated as AUC/MIC fallback.

    Returns
    -------
    list[float]
        PK/PD index for each subject.
    """
    valid_types = {"auc_mic", "cmax_mic", "t_mic"}
    if index_type not in valid_types:
        raise ValueError(f"index_type must be one of {valid_types}, got '{index_type}'")
    if not auc_values:
        raise ValueError("auc_values must not be empty")
    if not mic_values:
        raise ValueError("mic_values must not be empty")

    # Broadcast single MIC to all subjects
    if len(mic_values) == 1:
        mic_broadcast = [mic_values[0]] * len(auc_values)
    else:
        if len(mic_values) != len(auc_values):
            raise ValueError(
                f"mic_values length ({len(mic_values)}) must be 1 or match "
                f"auc_values length ({len(auc_values)})"
            )
        mic_broadcast = mic_values

    results = []
    for pk_val, mic in zip(auc_values, mic_broadcast, strict=True):
        if mic <= 0:
            raise ValueError("All MIC values must be positive")
        if index_type in ("auc_mic", "t_mic"):
            results.append(pk_val / mic)
        elif index_type == "cmax_mic":
            results.append(pk_val / mic)
    return results


def classify_responders(
    biomarker_values: list[float],
    threshold: float,
    direction: str = "above",
) -> dict[str, float | int | str]:
    """Classify subjects as responders or non-responders based on biomarker threshold.

    Parameters
    ----------
    biomarker_values:
        Biomarker measurements for each subject.
    threshold:
        Cutoff value for response classification.
    direction:
        'above' → responder if biomarker > threshold.
        'below' → responder if biomarker < threshold.

    Returns
    -------
    dict
        Keys: 'n_responders', 'n_non_responders', 'response_rate_pct', 'threshold'.
    """
    if not biomarker_values:
        raise ValueError("biomarker_values must not be empty")
    valid_dirs = {"above", "below"}
    if direction not in valid_dirs:
        raise ValueError(f"direction must be one of {valid_dirs}, got '{direction}'")

    if direction == "above":
        n_resp = sum(1 for v in biomarker_values if v > threshold)
    else:
        n_resp = sum(1 for v in biomarker_values if v < threshold)

    n_total = len(biomarker_values)
    n_non = n_total - n_resp
    rate = 100.0 * n_resp / n_total

    return {
        "n_responders": n_resp,
        "n_non_responders": n_non,
        "response_rate_pct": rate,
        "threshold": threshold,
    }


def exposure_response_summary(
    pk_param_values: list[float],
    response_values: list[float],
    n_bins: int = 4,
) -> ExposureResponseResult:
    """Summarize exposure-response relationship across quantile bins.

    Subjects are sorted by PK exposure (pk_param_values) and divided into
    n_bins quantile groups.  For each bin, the median exposure and mean
    response are computed.

    Parameters
    ----------
    pk_param_values:
        PK exposure metric (AUC, Cmax, etc.) per subject.
    response_values:
        Corresponding PD response values per subject.
    n_bins:
        Number of quantile bins (default 4).

    Returns
    -------
    ExposureResponseResult
    """
    if not pk_param_values or not response_values:
        raise ValueError("pk_param_values and response_values must not be empty")
    if len(pk_param_values) != len(response_values):
        raise ValueError(
            f"pk_param_values length ({len(pk_param_values)}) must equal "
            f"response_values length ({len(response_values)})"
        )
    n = len(pk_param_values)
    if n < n_bins:
        raise ValueError(
            f"Number of subjects ({n}) must be >= n_bins ({n_bins})"
        )
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    # Sort by PK exposure
    paired = sorted(zip(pk_param_values, response_values, strict=True), key=lambda p: p[0])
    sorted_pk = [p[0] for p in paired]
    sorted_resp = [p[1] for p in paired]

    # Overall correlation
    overall_r = _pearson_r(sorted_pk, sorted_resp)

    # Build equal-size bins
    bin_size = n // n_bins
    remainder = n % n_bins

    bin_edges: list[float] = []
    bin_median_exposure: list[float] = []
    bin_mean_response: list[float] = []
    bin_n: list[int] = []

    start = 0
    bin_edges.append(sorted_pk[0])
    for b in range(n_bins):
        # Distribute remainder subjects into first bins
        extra = 1 if b < remainder else 0
        end = start + bin_size + extra
        pk_bin = sorted_pk[start:end]
        resp_bin = sorted_resp[start:end]

        bin_median_exposure.append(_median(pk_bin))
        bin_mean_response.append(_mean(resp_bin))
        bin_n.append(len(pk_bin))
        bin_edges.append(sorted_pk[end - 1])
        start = end

    # Check monotonicity of bin_mean_response
    monotonic = all(
        bin_mean_response[i] <= bin_mean_response[i + 1]
        for i in range(len(bin_mean_response) - 1)
    ) or all(
        bin_mean_response[i] >= bin_mean_response[i + 1]
        for i in range(len(bin_mean_response) - 1)
    )

    notes_parts = [f"n={n} subjects in {n_bins} bins"]
    notes_parts.append(f"Overall Pearson r={overall_r:.3f}")
    notes_parts.append("Monotonic trend" if monotonic else "Non-monotonic trend")

    return ExposureResponseResult(
        n_subjects=n,
        n_bins=n_bins,
        bin_edges=bin_edges,
        bin_median_exposure=bin_median_exposure,
        bin_mean_response=bin_mean_response,
        bin_n=bin_n,
        overall_correlation_r=overall_r,
        monotonic=monotonic,
        notes="; ".join(notes_parts),
    )
