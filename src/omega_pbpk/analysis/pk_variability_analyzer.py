"""PK variability and between-subject variability (BSV) analysis — Phase 800.

Generates virtual populations using LCG-based random sampling with Box-Muller
transform. Computes percentiles, geometric mean, GSD, and variability class.

Pure Python only: stdlib math, dataclasses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "PKVariabilityResult",
    "analyze_pk_variability",
    "compare_pk_variability",
]

_LCG_A = 1664525
_LCG_C = 1013904223
_LCG_M = 0x100000000  # 2^32


def _lcg_next(state: int) -> tuple:
    """Advance LCG and return (new_state, uniform_0_1)."""
    state = (_LCG_A * state + _LCG_C) & 0xFFFFFFFF
    return state, state / _LCG_M


def _generate_normal_samples(n: int, seed: int = 12345) -> list:
    """Generate n standard normal samples using LCG + Box-Muller.

    Parameters
    ----------
    n : int
        Number of samples.
    seed : int
        Initial LCG seed.

    Returns
    -------
    list[float]
        Standard normal samples.
    """
    samples = []
    state = seed
    # Box-Muller generates pairs; generate enough pairs
    n_pairs = (n + 1) // 2
    for _ in range(n_pairs):
        state, u1 = _lcg_next(state)
        state, u2 = _lcg_next(state)
        # Avoid log(0)
        u1 = max(u1, 1e-15)
        r = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2
        z1 = r * math.cos(theta)
        z2 = r * math.sin(theta)
        samples.append(z1)
        samples.append(z2)
    return samples[:n]


def _percentile(sorted_vals: list, p: float) -> float:
    """Compute percentile from a sorted list using linear interpolation.

    Parameters
    ----------
    sorted_vals : list[float]
        Sorted list of values.
    p : float
        Percentile in [0, 100].

    Returns
    -------
    float
    """
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    # Linear interpolation
    idx = p / 100.0 * (n - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= n:
        return sorted_vals[-1]
    frac = idx - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


@dataclass(frozen=True)
class PKVariabilityResult:
    """Result of PK variability analysis.

    Fields
    ------
    parameter_name : str
        Name of the PK parameter (e.g., "CL", "Vd").
    typical_value : float
        Typical (population) value of the parameter.
    cv_pct : float
        Coefficient of variation (%).
    geometric_mean : float
        Geometric mean of virtual subjects.
    geometric_sd : float
        Geometric standard deviation (GSD).
    p5 : float
        5th percentile of virtual subjects.
    p25 : float
        25th percentile of virtual subjects.
    p50 : float
        Median (50th percentile) of virtual subjects.
    p75 : float
        75th percentile of virtual subjects.
    p95 : float
        95th percentile of virtual subjects.
    n_subjects : int
        Number of virtual subjects generated.
    distribution : str
        Distribution type: "log-normal" or "normal".
    variability_class : str
        "low" (<20%), "moderate" (20-50%), or "high" (>50%).
    notes : str
        Summary notes.
    """

    parameter_name: str
    typical_value: float
    cv_pct: float
    geometric_mean: float
    geometric_sd: float
    p5: float
    p25: float
    p50: float
    p75: float
    p95: float
    n_subjects: int
    distribution: str
    variability_class: str
    notes: str


def _variability_class(cv_pct: float) -> str:
    """Classify variability based on CV%.

    Returns
    -------
    str
        "low", "moderate", or "high".
    """
    if cv_pct < 20.0:
        return "low"
    elif cv_pct <= 50.0:
        return "moderate"
    else:
        return "high"


def analyze_pk_variability(
    parameter_name: str,
    typical_value: float,
    cv_pct: float,
    n_subjects: int = 1000,
    distribution: str = "log-normal",
) -> PKVariabilityResult:
    """Analyze PK variability for a single parameter.

    Generates virtual subjects and computes statistical summaries.

    Parameters
    ----------
    parameter_name : str
        Name of the PK parameter.
    typical_value : float
        Typical (population mean) value (must be > 0).
    cv_pct : float
        Coefficient of variation in % (must be >= 0).
    n_subjects : int
        Number of virtual subjects to generate (default 1000).
    distribution : str
        "log-normal" (default) or "normal".

    Returns
    -------
    PKVariabilityResult

    Raises
    ------
    ValueError
        If typical_value <= 0 or cv_pct < 0.
    """
    if typical_value <= 0.0:
        raise ValueError(f"typical_value must be > 0, got {typical_value}")
    if cv_pct < 0.0:
        raise ValueError(f"cv_pct must be >= 0, got {cv_pct}")
    if n_subjects < 1:
        raise ValueError(f"n_subjects must be >= 1, got {n_subjects}")

    cv = cv_pct / 100.0
    z_samples = _generate_normal_samples(n_subjects, seed=42)

    if distribution == "log-normal":
        # sigma = sqrt(ln(1 + cv^2))
        sigma = math.sqrt(math.log(1.0 + cv * cv))
        # subject_value = typical * exp(sigma * z)
        values = [typical_value * math.exp(sigma * z) for z in z_samples]
        # Geometric SD = exp(sigma)
        geometric_sd = math.exp(sigma)
    else:
        # Normal distribution: subject_value = typical + sigma_abs * z
        sigma_abs = typical_value * cv
        values = [typical_value + sigma_abs * z for z in z_samples]
        # GSD: not meaningful for normal, use approximate
        geometric_sd = math.exp(cv)  # approximate

    sorted_vals = sorted(values)

    # Geometric mean from log-normal: exp(mean(log(values)))
    # Use only positive values for log computation
    pos_vals = [v for v in values if v > 0.0]
    if pos_vals:
        log_sum = sum(math.log(v) for v in pos_vals)
        geometric_mean = math.exp(log_sum / len(pos_vals))
    else:
        geometric_mean = typical_value

    p5 = _percentile(sorted_vals, 5.0)
    p25 = _percentile(sorted_vals, 25.0)
    p50 = _percentile(sorted_vals, 50.0)
    p75 = _percentile(sorted_vals, 75.0)
    p95 = _percentile(sorted_vals, 95.0)

    var_class = _variability_class(cv_pct)

    notes = (
        f"{parameter_name}: typical={typical_value:.4g}, CV={cv_pct:.1f}% "
        f"({var_class}), {distribution}, n={n_subjects}; "
        f"p5={p5:.4g}, p50={p50:.4g}, p95={p95:.4g}, "
        f"geomean={geometric_mean:.4g}, GSD={geometric_sd:.3f}"
    )

    return PKVariabilityResult(
        parameter_name=parameter_name,
        typical_value=typical_value,
        cv_pct=cv_pct,
        geometric_mean=geometric_mean,
        geometric_sd=geometric_sd,
        p5=p5,
        p25=p25,
        p50=p50,
        p75=p75,
        p95=p95,
        n_subjects=n_subjects,
        distribution=distribution,
        variability_class=var_class,
        notes=notes,
    )


def compare_pk_variability(parameters: list) -> list:
    """Compare PK variability across multiple parameters.

    Parameters
    ----------
    parameters : list[dict]
        Each dict must have keys:
            - "parameter_name" (str)
            - "typical_value" (float)
            - "cv_pct" (float)
            - "n_subjects" (int, optional, default 1000)
            - "distribution" (str, optional, default "log-normal")

    Returns
    -------
    list[PKVariabilityResult]
        Results sorted by cv_pct descending (highest variability first).
    """
    results = []
    for p in parameters:
        result = analyze_pk_variability(
            parameter_name=p["parameter_name"],
            typical_value=p["typical_value"],
            cv_pct=p["cv_pct"],
            n_subjects=p.get("n_subjects", 1000),
            distribution=p.get("distribution", "log-normal"),
        )
        results.append(result)

    # Sort by cv_pct descending
    results.sort(key=lambda r: r.cv_pct, reverse=True)
    return results
