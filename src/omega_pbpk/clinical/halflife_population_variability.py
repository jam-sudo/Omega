"""Drug half-life population variability model (Phase 570).

Models inter-individual variability in drug half-life across a population,
generating percentile distributions using log-normal PK parameter sampling.
"""

import math
import random
from dataclasses import dataclass


@dataclass
class HalflifeVariabilityResult:
    """Result of a half-life population variability simulation."""

    drug_name: str
    n_subjects: int
    mean_thalf_h: float
    sd_thalf_h: float
    cv_pct: float
    p5_thalf_h: float
    p25_thalf_h: float
    p50_thalf_h: float
    p75_thalf_h: float
    p95_thalf_h: float
    thalf_values: list
    n_short_halflife: int
    n_long_halflife: int
    variability_class: str
    notes: str


def _percentile(sorted_values: list, p: float) -> float:
    """Compute the p-th percentile from a sorted list (0 <= p <= 100)."""
    n = len(sorted_values)
    idx = p / 100.0 * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


def simulate_halflife_variability(
    drug_name: str,
    typical_cl_L_per_h: float,
    typical_vd_L: float,
    omega_cl: float = 0.3,
    omega_vd: float = 0.2,
    n_subjects: int = 100,
    seed: int = 42,
) -> HalflifeVariabilityResult:
    """Simulate half-life variability across a population.

    Samples CL and Vd from log-normal distributions and computes
    individual half-lives: t_half = 0.693 * Vd / CL.
    """
    if n_subjects <= 0:
        raise ValueError("n_subjects must be > 0")
    if typical_cl_L_per_h <= 0:
        raise ValueError("typical_cl_L_per_h must be > 0")
    if typical_vd_L <= 0:
        raise ValueError("typical_vd_L must be > 0")
    if omega_cl <= 0:
        raise ValueError("omega_cl must be > 0")
    if omega_vd <= 0:
        raise ValueError("omega_vd must be > 0")

    rng = random.Random(seed)

    thalf_values = []
    for _ in range(n_subjects):
        cl_i = typical_cl_L_per_h * math.exp(rng.gauss(0, omega_cl))
        vd_i = typical_vd_L * math.exp(rng.gauss(0, omega_vd))
        t_half_i = 0.693 * vd_i / cl_i
        thalf_values.append(t_half_i)

    sorted_vals = sorted(thalf_values)

    # Percentiles
    p5 = _percentile(sorted_vals, 5)
    p25 = _percentile(sorted_vals, 25)
    p50 = _percentile(sorted_vals, 50)
    p75 = _percentile(sorted_vals, 75)
    p95 = _percentile(sorted_vals, 95)

    # Mean and SD
    mean_val = sum(thalf_values) / n_subjects
    variance = (
        sum((v - mean_val) ** 2 for v in thalf_values) / (n_subjects - 1) if n_subjects > 1 else 0.0
    )
    sd_val = math.sqrt(variance)
    cv_pct = (sd_val / mean_val * 100) if mean_val > 0 else 0.0

    # Counts
    n_short = sum(1 for v in thalf_values if v < p25)
    n_long = sum(1 for v in thalf_values if v > p75)

    # Variability classification
    if cv_pct < 30:
        variability_class = "low"
    elif cv_pct <= 60:
        variability_class = "moderate"
    else:
        variability_class = "high"

    return HalflifeVariabilityResult(
        drug_name=drug_name,
        n_subjects=n_subjects,
        mean_thalf_h=mean_val,
        sd_thalf_h=sd_val,
        cv_pct=cv_pct,
        p5_thalf_h=p5,
        p25_thalf_h=p25,
        p50_thalf_h=p50,
        p75_thalf_h=p75,
        p95_thalf_h=p95,
        thalf_values=thalf_values,
        n_short_halflife=n_short,
        n_long_halflife=n_long,
        variability_class=variability_class,
        notes=f"Log-normal sampling, omega_cl={omega_cl}, omega_vd={omega_vd}, seed={seed}",
    )


def compare_variability_scenarios(
    drug_name: str,
    typical_cl: float,
    typical_vd: float,
    omega_cl_values: list = None,
) -> list:
    """Compare variability across different omega_cl values.

    Returns results sorted by cv_pct ascending.
    """
    if omega_cl_values is None:
        omega_cl_values = [0.1, 0.3, 0.5, 0.8]

    results = [
        simulate_halflife_variability(
            drug_name=drug_name,
            typical_cl_L_per_h=typical_cl,
            typical_vd_L=typical_vd,
            omega_cl=ocl,
        )
        for ocl in omega_cl_values
    ]
    results.sort(key=lambda r: r.cv_pct)
    return results
