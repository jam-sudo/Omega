"""
Phase 590 — Sample Size and Statistical Power Calculations for PK Studies

Implements power computation for two-sample t-tests, sample size determination,
bioequivalence sample size, minimum detectable difference, and crossover vs.
parallel design comparison.
"""

import math

# ---------------------------------------------------------------------------
# Standard normal CDF approximation via math.erf
# ---------------------------------------------------------------------------


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Power for two-sample t-test
# ---------------------------------------------------------------------------


def power_t_test(
    n: int,
    delta: float,
    sigma: float,
    alpha: float = 0.05,
) -> float:
    """
    Approximate power of a two-sample t-test.

    Parameters
    ----------
    n     : Sample size per group
    delta : Effect size (mean difference)
    sigma : Within-group standard deviation
    alpha : Type I error rate (two-sided)

    Returns
    -------
    Power in [0, 1]
    """
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be in (0, 1)")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if n < 1:
        raise ValueError("n must be >= 1")

    z_alpha = _norm_cdf_inv(1.0 - alpha / 2.0)
    effect_size = abs(delta) / sigma
    non_centrality = effect_size * math.sqrt(n / 2.0)
    power = _norm_cdf(non_centrality - z_alpha)
    return min(max(power, 0.0), 1.0)


def _norm_cdf_inv(p: float) -> float:
    """
    Approximate inverse normal CDF (quantile function) using rational
    approximation (Beasley-Springer-Moro algorithm subset).
    Valid for p in (0, 1).
    """
    if p <= 0 or p >= 1:
        raise ValueError("p must be in (0, 1)")
    # Common critical values shortcut
    if abs(p - 0.975) < 1e-9:
        return 1.959964
    if abs(p - 0.95) < 1e-9:
        return 1.644854

    # Rational approximation
    a = [
        0.0,
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        0.0,
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        0.0,
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        0.0,
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    p_low = 0.02425
    p_high = 1 - p_low

    if p_low <= p <= p_high:
        q = p - 0.5
        r = q * q
        num = (((((a[1] * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * r + a[6]) * q
        den = ((((b[1] * r + b[2]) * r + b[3]) * r + b[4]) * r + b[5]) * r + 1
        return num / den
    elif p < p_low:
        q = math.sqrt(-2 * math.log(p))
        num = ((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]
        den = (((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1
        return num / den
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        num = ((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]
        den = (((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1
        return -(num / den)


# ---------------------------------------------------------------------------
# Sample size for two-sample t-test
# ---------------------------------------------------------------------------


def sample_size_t_test(
    delta: float,
    sigma: float,
    power: float = 0.80,
    alpha: float = 0.05,
) -> int:
    """
    Find minimum n per group to achieve desired power.

    Searches n from 2 to 500. Raises ValueError if not achievable.

    Returns
    -------
    Minimum n per group (int)
    """
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be in (0, 1)")
    if power <= 0 or power >= 1:
        raise ValueError("power must be in (0, 1)")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    for n in range(2, 501):
        if power_t_test(n, delta, sigma, alpha) >= power:
            return n
    raise ValueError(
        f"Cannot achieve power={power} within n=500 subjects per group "
        f"(delta={delta}, sigma={sigma}, alpha={alpha})"
    )


# ---------------------------------------------------------------------------
# Bioequivalence sample size (Chow & Wang approximate method)
# ---------------------------------------------------------------------------


def be_sample_size(
    cv_intra: float,
    theta_1: float = 0.80,
    theta_2: float = 1.25,
    power: float = 0.80,
    alpha: float = 0.05,
) -> dict:
    """
    Approximate BE sample size per the Chow & Wang method.

    Parameters
    ----------
    cv_intra : Within-subject CV (e.g. 0.20 for 20%)
    theta_1  : Lower BE limit (default 0.80)
    theta_2  : Upper BE limit (default 1.25)
    power    : Desired power (default 0.80)
    alpha    : One-sided significance level for TOST (default 0.05)

    Returns
    -------
    dict with n_per_group, n_total, cv_intra, power, notes
    """
    if cv_intra <= 0:
        raise ValueError("cv_intra must be positive")

    sigma_ln = math.sqrt(math.log(1.0 + cv_intra**2))
    delta_ln = math.log(theta_2)  # symmetric BE limits → use upper margin

    # z_alpha for one-sided alpha=0.05 → 1.645
    # z_power for 80% power → 0.842
    z_alpha = 1.645
    z_power = 0.842

    n_raw = math.ceil(2.0 * ((z_alpha + z_power) / delta_ln * sigma_ln) ** 2)
    n_per_group = max(n_raw, 12)  # regulatory minimum

    return {
        "n_per_group": n_per_group,
        "n_total": n_per_group * 2,
        "cv_intra": cv_intra,
        "power": power,
        "notes": (
            f"BE sample size: CV={cv_intra * 100:.1f}%, "
            f"BE limits [{theta_1}-{theta_2}], alpha={alpha}"
        ),
    }


# ---------------------------------------------------------------------------
# Minimum detectable difference
# ---------------------------------------------------------------------------


def minimum_detectable_difference(
    n: int,
    sigma: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """
    Given n per group, compute the minimum detectable difference (MDD).

    delta = sigma * (z_alpha + z_power) * sqrt(2/n)

    Parameters
    ----------
    n     : Sample size per group
    sigma : Within-group standard deviation
    alpha : Two-sided type I error rate
    power : Desired power

    Returns
    -------
    Minimum detectable delta (float)
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    z_alpha = 1.96  # two-sided alpha=0.05
    z_power = 0.842  # 80% power

    return sigma * (z_alpha + z_power) * math.sqrt(2.0 / n)


# ---------------------------------------------------------------------------
# Crossover vs. parallel design comparison
# ---------------------------------------------------------------------------


def crossover_vs_parallel_n(
    cv_intra: float,
    cv_inter: float,
    delta: float,
    power: float = 0.80,
) -> dict:
    """
    Compare sample size requirements for crossover vs. parallel-group designs.

    For crossover: sigma driven by within-subject variability (cv_intra).
    For parallel:  sigma driven by total variability sqrt(cv_intra^2 + cv_inter^2).

    Parameters
    ----------
    cv_intra : Within-subject CV (e.g. 0.20)
    cv_inter : Between-subject CV (e.g. 0.30)
    delta    : Effect size (mean difference to detect)
    power    : Desired power (default 0.80)

    Returns
    -------
    dict with n_crossover, n_parallel, efficiency_gain
    """
    if cv_intra <= 0:
        raise ValueError("cv_intra must be positive")
    if cv_inter < 0:
        raise ValueError("cv_inter must be non-negative")

    sigma_crossover = cv_intra
    sigma_parallel = math.sqrt(cv_intra**2 + cv_inter**2)

    n_crossover = sample_size_t_test(delta, sigma_crossover, power=power)
    n_parallel = sample_size_t_test(delta, sigma_parallel, power=power)

    efficiency_gain = n_parallel / n_crossover if n_crossover > 0 else 1.0

    return {
        "n_crossover": n_crossover,
        "n_parallel": n_parallel,
        "efficiency_gain": efficiency_gain,
    }
