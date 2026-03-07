"""Phase 566 — Statistical methods for population PK/PD analysis."""

import math
import random

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _mean(values: list) -> float:
    return sum(values) / len(values)


def _variance(values: list, ddof: int = 1) -> float:
    mu = _mean(values)
    n = len(values)
    return sum((v - mu) ** 2 for v in values) / (n - ddof)


def _std(values: list, ddof: int = 1) -> float:
    return math.sqrt(_variance(values, ddof=ddof))


def _geometric_mean(values: list) -> float:
    log_sum = sum(math.log(v) for v in values)
    return math.exp(log_sum / len(values))


def _cv(values: list) -> float:
    """Coefficient of variation (%)."""
    mu = _mean(values)
    if mu == 0:
        return 0.0
    return 100.0 * _std(values) / abs(mu)


def _rank_list(values: list) -> list:
    """Return ranks (1-based) for a list of values (average ranks for ties)."""
    n = len(values)
    # Create (value, original_index) pairs sorted by value
    sorted_pairs = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        # Find end of tie group
        while j < n - 1 and sorted_pairs[j + 1][1] == sorted_pairs[j][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[sorted_pairs[k][0]] = avg_rank
        i = j + 1
    return ranks


# ---------------------------------------------------------------------------
# chi-squared CDF approximation (Wilson-Hilferty)
# ---------------------------------------------------------------------------


def _chi2_sf(x: float, df: float) -> float:
    """Survival function of chi-squared distribution (approximate).

    Uses the Wilson-Hilferty cube-root transform to approximate the
    chi-squared CDF via the standard normal.
    """
    if x <= 0:
        return 1.0
    # Wilson-Hilferty approximation: (X/df)^(1/3) ~ N(1 - 2/(9*df), 2/(9*df))
    mu_wh = 1.0 - 2.0 / (9.0 * df)
    sigma_wh = math.sqrt(2.0 / (9.0 * df))
    z = ((x / df) ** (1.0 / 3.0) - mu_wh) / sigma_wh
    # Standard normal survival function approximation (Abramowitz & Stegun 26.2.17)
    return _normal_sf(z)


def _normal_cdf(z: float) -> float:
    """Standard normal CDF via math.erf."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _normal_sf(z: float) -> float:
    return 1.0 - _normal_cdf(z)


# ---------------------------------------------------------------------------
# t-distribution survival function approximation
# ---------------------------------------------------------------------------


def _t_sf_two_tailed(t: float, df: float) -> float:
    """Two-tailed p-value for t-statistic (approximate via normal for large df)."""
    abs_t = abs(t)
    if df >= 30:
        return 2.0 * _normal_sf(abs_t)
    # Use a simple rational approximation for small df
    # P(T > t) ≈ using regularized incomplete beta function
    # For simplicity, use the normal approximation with a correction
    # Cornish-Fisher approximation is used here as a lightweight substitute
    _x = abs_t / math.sqrt(df)
    z = abs_t * (1.0 - 1.0 / (4.0 * df))  # simple correction
    return 2.0 * _normal_sf(z)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: list,
    statistic: str = "mean",
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap confidence interval for a statistic.

    Parameters
    ----------
    values:
        Data values. Must have at least 2 elements.
    statistic:
        One of 'mean', 'median', 'geometric_mean', 'cv'.
    n_bootstrap:
        Number of bootstrap resamples.
    ci_level:
        Confidence level (e.g., 0.95).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict: estimate, lower_ci, upper_ci, ci_level, n_bootstrap
    """
    if len(values) < 2:
        raise ValueError("bootstrap_ci requires at least 2 values")

    stat_funcs = {
        "mean": _mean,
        "median": lambda v: (
            sorted(v)[len(v) // 2]
            if len(v) % 2 == 1
            else (sorted(v)[len(v) // 2 - 1] + sorted(v)[len(v) // 2]) / 2.0
        ),
        "geometric_mean": _geometric_mean,
        "cv": _cv,
    }
    if statistic not in stat_funcs:
        raise ValueError(f"Unknown statistic '{statistic}'. Supported: {sorted(stat_funcs)}")

    stat_fn = stat_funcs[statistic]
    estimate = stat_fn(values)

    rng = random.Random(seed)
    n = len(values)
    boot_stats = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(values) for _ in range(n)]
        boot_stats.append(stat_fn(sample))

    boot_stats.sort()
    alpha = 1.0 - ci_level
    lower_idx = int(math.floor(alpha / 2.0 * n_bootstrap))
    upper_idx = int(math.ceil((1.0 - alpha / 2.0) * n_bootstrap)) - 1
    lower_idx = max(0, min(lower_idx, n_bootstrap - 1))
    upper_idx = max(0, min(upper_idx, n_bootstrap - 1))

    return {
        "estimate": estimate,
        "lower_ci": boot_stats[lower_idx],
        "upper_ci": boot_stats[upper_idx],
        "ci_level": ci_level,
        "n_bootstrap": n_bootstrap,
    }


def nonparametric_anova(groups: list) -> dict:
    """Kruskal-Wallis H-test (rank-based ANOVA).

    Parameters
    ----------
    groups:
        List of groups, each a list of floats. Need at least 2 groups,
        each with at least 1 observation.

    Returns
    -------
    dict: H_statistic, df, p_value_approx, significant (p < 0.05)
    """
    if len(groups) < 2:
        raise ValueError("nonparametric_anova requires at least 2 groups")
    for i, g in enumerate(groups):
        if len(g) < 1:
            raise ValueError(f"Group {i} has no observations")

    # Pool all observations and compute ranks
    all_values = []
    group_sizes = []
    for g in groups:
        all_values.extend(g)
        group_sizes.append(len(g))

    N = len(all_values)
    ranks = _rank_list(all_values)

    # Compute rank sums per group
    idx = 0
    H = 0.0
    for nj in group_sizes:
        group_ranks = ranks[idx : idx + nj]
        Rj = sum(group_ranks)
        H += (Rj**2) / nj
        idx += nj

    H = (12.0 / (N * (N + 1))) * H - 3.0 * (N + 1)
    df = len(groups) - 1
    p_value = _chi2_sf(H, df)

    return {
        "H_statistic": H,
        "df": df,
        "p_value_approx": p_value,
        "significant": p_value < 0.05,
    }


def pearson_correlation(x: list, y: list) -> dict:
    """Pearson product-moment correlation.

    Parameters
    ----------
    x, y:
        Paired data lists. Must have at least 2 elements each.

    Returns
    -------
    dict: r, r_squared, interpretation ('strong', 'moderate', 'weak')
    """
    if len(x) < 2 or len(y) < 2:
        raise ValueError("pearson_correlation requires at least 2 values")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    n = len(x)
    mx = _mean(x)
    my = _mean(y)

    cov = sum((x[i] - mx) * (y[i] - my) for i in range(n)) / (n - 1)
    sx = _std(x, ddof=1)
    sy = _std(y, ddof=1)

    if sx == 0 or sy == 0:
        r = 0.0
    else:
        r = cov / (sx * sy)

    r = max(-1.0, min(1.0, r))
    abs_r = abs(r)

    if abs_r > 0.7:
        interpretation = "strong"
    elif abs_r >= 0.3:
        interpretation = "moderate"
    else:
        interpretation = "weak"

    return {
        "r": r,
        "r_squared": r**2,
        "interpretation": interpretation,
    }


def spearman_correlation(x: list, y: list) -> dict:
    """Spearman rank correlation.

    Parameters
    ----------
    x, y:
        Paired data lists. Must have at least 2 elements each.

    Returns
    -------
    dict: rho, p_value_approx
    """
    if len(x) < 2 or len(y) < 2:
        raise ValueError("spearman_correlation requires at least 2 values")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    n = len(x)
    rx = _rank_list(x)
    ry = _rank_list(y)

    # Pearson on ranks
    pearson_result = pearson_correlation(rx, ry)
    rho = pearson_result["r"]

    # t approximation: t = rho * sqrt((n-2) / (1 - rho^2))
    if abs(rho) >= 1.0:
        p_value = 0.0
    else:
        t_stat = rho * math.sqrt((n - 2) / (1.0 - rho**2))
        p_value = _t_sf_two_tailed(t_stat, df=n - 2)

    return {
        "rho": rho,
        "p_value_approx": p_value,
    }


def effect_size_cohens_d(group1: list, group2: list) -> dict:
    """Cohen's d effect size between two groups.

    d = (mean1 - mean2) / pooled_std
    pooled_std = sqrt(((n1-1)*s1^2 + (n2-1)*s2^2) / (n1+n2-2))

    Parameters
    ----------
    group1, group2:
        Data for each group. Must have at least 2 elements each.

    Returns
    -------
    dict: d, pooled_std, interpretation ('small', 'medium', 'large')
    """
    if len(group1) < 2 or len(group2) < 2:
        raise ValueError("effect_size_cohens_d requires at least 2 values per group")

    n1, n2 = len(group1), len(group2)
    m1, m2 = _mean(group1), _mean(group2)
    s1_sq = _variance(group1, ddof=1)
    s2_sq = _variance(group2, ddof=1)

    pooled_var = ((n1 - 1) * s1_sq + (n2 - 1) * s2_sq) / (n1 + n2 - 2)
    pooled_std = math.sqrt(pooled_var)

    if pooled_std == 0:
        d = 0.0
    else:
        d = (m1 - m2) / pooled_std

    abs_d = abs(d)
    if abs_d >= 0.8:
        interpretation = "large"
    elif abs_d >= 0.2:
        interpretation = "medium"
    else:
        interpretation = "small"

    return {
        "d": d,
        "pooled_std": pooled_std,
        "interpretation": interpretation,
    }
