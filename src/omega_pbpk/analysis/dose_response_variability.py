"""Dose-response variability analysis across a virtual population (Phase 664).

Analyzes variability in dose-response relationships including EC50 variability,
Hill coefficient estimation, and responder/non-responder classification.

Uses a log-normal distribution for individual PD parameters (EC50, Emax)
because PK/PD parameters are inherently positive and right-skewed.

Key features
------------
- Population EC50 variability analysis (CV%, percentiles)
- Responder classification based on therapeutic concentration
- Hill Emax PD response simulation for virtual population
- LCG-based RNG for reproducibility without numpy

References
----------
- Sheiner LB, Steimer JL. Annu Rev Pharmacol Toxicol. 2000;40:67-95
- Holford NHG, Sheiner LB. Clin Pharmacokinet. 1981;6(6):429-53
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Pure-Python LCG random number generator
# ---------------------------------------------------------------------------


def _lcg(seed: int):
    """Linear congruential generator yielding uniform samples in [0, 1)."""
    a, c, m = 1664525, 1013904223, 2**32
    state = int(seed) % m
    while True:
        state = (a * state + c) % m
        yield state / m


def _lognormal_sample(mean: float, cv: float, rng_gen) -> float:
    """Draw one log-normal sample using Box-Muller transform.

    Parameters
    ----------
    mean : Population mean of the lognormal distribution.
    cv   : Coefficient of variation (decimal, e.g. 0.4 for 40%).
    rng_gen : Generator yielding uniform [0,1) values.
    """
    u = next(rng_gen)
    v = next(rng_gen)
    # Box-Muller
    z = math.sqrt(-2.0 * math.log(max(u, 1e-10))) * math.cos(2.0 * math.pi * v)
    sigma = math.sqrt(math.log(1.0 + cv**2))
    mu = math.log(mean) - sigma**2 / 2.0
    return math.exp(mu + sigma * z)


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Compute percentile p (0-100) from a pre-sorted list via linear interpolation."""
    n = len(sorted_vals)
    if n == 0:
        return float("nan")
    if n == 1:
        return sorted_vals[0]
    idx = p / 100.0 * (n - 1)
    lo = int(math.floor(idx))
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoseResponseVariabilityResult:
    """Results from population dose-response variability analysis."""

    n_subjects: int
    ec50_mean: float
    ec50_sd: float
    ec50_cv_pct: float
    ec50_p5: float
    ec50_p95: float
    hill_n_mean: float
    emax_mean: float
    responder_threshold_mg_L: float
    n_responders: int
    responder_fraction: float
    pd_variability_class: str  # "low" (<30%), "moderate" (30-60%), "high" (>60%)
    notes: str


# ---------------------------------------------------------------------------
# Hill Emax response
# ---------------------------------------------------------------------------


def _hill_response(conc: float, ec50: float, emax: float, n: float) -> float:
    """Hill Emax dose-response: E = Emax * C^n / (EC50^n + C^n)."""
    if conc <= 0.0:
        return 0.0
    cn = conc**n
    ec50n = ec50**n
    return emax * cn / (ec50n + cn)


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------


def analyze_dose_response_variability(
    ec50_values: list[float],
    emax_values: list[float] | None = None,
    hill_values: list[float] | None = None,
    standard_dose_conc_mg_L: float = 1.0,
    responder_threshold_fraction: float = 0.5,
) -> DoseResponseVariabilityResult:
    """Analyze variability in dose-response across a population.

    Parameters
    ----------
    ec50_values:
        Individual EC50 values (mg/L) from population. Must have >= 2 values,
        all > 0.
    emax_values:
        Individual Emax values. If None, defaults to 100.0 for all subjects.
    hill_values:
        Individual Hill n values. If None, defaults to 1.0 for all subjects.
    standard_dose_conc_mg_L:
        Typical therapeutic drug concentration (mg/L) used to classify responders.
    responder_threshold_fraction:
        A subject is a responder if their EC50 <= standard_dose_conc_mg_L *
        responder_threshold_fraction. Default 0.5 means responder if EC50 <=
        half the standard concentration (response > 50% Emax at standard dose).

    Returns
    -------
    DoseResponseVariabilityResult

    Raises
    ------
    ValueError
        If ec50_values is empty, has < 2 items, contains non-positive values.
    """
    if len(ec50_values) < 2:
        raise ValueError(f"ec50_values must have at least 2 values, got {len(ec50_values)}")
    if any(v <= 0 for v in ec50_values):
        raise ValueError("All ec50_values must be > 0")

    n = len(ec50_values)

    # Fill defaults
    emax_vals = list(emax_values) if emax_values is not None else [100.0] * n
    hill_vals = list(hill_values) if hill_values is not None else [1.0] * n

    # EC50 statistics
    ec50_mean = sum(ec50_values) / n
    ec50_var = sum((v - ec50_mean) ** 2 for v in ec50_values) / (n - 1) if n > 1 else 0.0
    ec50_sd = math.sqrt(ec50_var)
    ec50_cv_pct = (ec50_sd / ec50_mean * 100.0) if ec50_mean > 0 else 0.0

    sorted_ec50 = sorted(ec50_values)
    ec50_p5 = _percentile(sorted_ec50, 5.0)
    ec50_p95 = _percentile(sorted_ec50, 95.0)

    # Emax / Hill means
    hill_n_mean = sum(hill_vals) / len(hill_vals) if hill_vals else 1.0
    emax_mean = sum(emax_vals) / len(emax_vals) if emax_vals else 100.0

    # Responder classification
    threshold_ec50 = standard_dose_conc_mg_L * responder_threshold_fraction
    n_responders = sum(1 for v in ec50_values if v <= threshold_ec50)
    responder_fraction = n_responders / n

    # Variability class
    if ec50_cv_pct < 30.0:
        pd_variability_class = "low"
    elif ec50_cv_pct <= 60.0:
        pd_variability_class = "moderate"
    else:
        pd_variability_class = "high"

    notes = (
        f"Population EC50 variability: mean={ec50_mean:.3g} mg/L, "
        f"CV={ec50_cv_pct:.1f}% ({pd_variability_class}). "
        f"{n_responders}/{n} subjects respond at {standard_dose_conc_mg_L} mg/L "
        f"(threshold EC50 <= {threshold_ec50:.3g} mg/L). "
        f"Mean Hill n={hill_n_mean:.2f}, mean Emax={emax_mean:.1f}."
    )

    return DoseResponseVariabilityResult(
        n_subjects=n,
        ec50_mean=ec50_mean,
        ec50_sd=ec50_sd,
        ec50_cv_pct=ec50_cv_pct,
        ec50_p5=ec50_p5,
        ec50_p95=ec50_p95,
        hill_n_mean=hill_n_mean,
        emax_mean=emax_mean,
        responder_threshold_mg_L=threshold_ec50,
        n_responders=n_responders,
        responder_fraction=responder_fraction,
        pd_variability_class=pd_variability_class,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Population simulation
# ---------------------------------------------------------------------------


def simulate_population_dose_response(
    n_subjects: int = 100,
    ec50_mean: float = 1.0,
    ec50_cv_pct: float = 40.0,
    emax_mean: float = 100.0,
    emax_cv_pct: float = 20.0,
    hill_n: float = 1.5,
    dose_concentrations: list[float] | None = None,
    seed: int = 42,
) -> dict:
    """Generate a virtual population and compute dose-response curves.

    Uses a log-normal distribution for EC50 and Emax. Hill n is fixed across
    subjects (population mean assumed homogeneous for simplicity).

    Parameters
    ----------
    n_subjects:
        Number of virtual subjects to simulate.
    ec50_mean:
        Population mean EC50 (mg/L).
    ec50_cv_pct:
        CV% for EC50 between subjects (inter-individual variability).
    emax_mean:
        Population mean Emax (% or arbitrary units).
    emax_cv_pct:
        CV% for Emax between subjects.
    hill_n:
        Hill coefficient (fixed, same for all subjects).
    dose_concentrations:
        List of concentration values (mg/L) for dose-response curve.
        Defaults to [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0].
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        ec50_values         : list[float] - individual EC50s
        emax_values         : list[float] - individual Emaxs
        mean_response_curve : list[float] - population mean response at each concentration
        p5_response_curve   : list[float] - 5th percentile response
        p95_response_curve  : list[float] - 95th percentile response
        dose_concentrations : list[float] - concentration grid
        analysis            : DoseResponseVariabilityResult

    Raises
    ------
    ValueError
        If n_subjects < 2 or ec50_cv_pct < 0.
    """
    if n_subjects < 2:
        raise ValueError(f"n_subjects must be >= 2, got {n_subjects}")
    if ec50_cv_pct < 0:
        raise ValueError(f"ec50_cv_pct must be >= 0, got {ec50_cv_pct}")

    if dose_concentrations is None:
        dose_concentrations = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]

    rng = _lcg(seed)

    # Generate individual parameters
    ec50_cv = ec50_cv_pct / 100.0
    emax_cv = emax_cv_pct / 100.0

    ec50_values: list[float] = []
    emax_values: list[float] = []

    for _ in range(n_subjects):
        if ec50_cv > 0:
            ec50_i = _lognormal_sample(ec50_mean, ec50_cv, rng)
        else:
            ec50_i = ec50_mean
        if emax_cv > 0:
            emax_i = _lognormal_sample(emax_mean, emax_cv, rng)
        else:
            emax_i = emax_mean
        ec50_values.append(max(ec50_i, 1e-9))
        emax_values.append(max(emax_i, 0.0))

    # Compute response at each concentration for each subject
    n_conc = len(dose_concentrations)
    # responses[j] = list of n_subjects responses at concentration j
    responses_by_conc: list[list[float]] = [[] for _ in range(n_conc)]

    for i in range(n_subjects):
        for j, conc in enumerate(dose_concentrations):
            resp = _hill_response(conc, ec50_values[i], emax_values[i], hill_n)
            responses_by_conc[j].append(resp)

    # Compute mean, p5, p95 at each concentration
    mean_response_curve: list[float] = []
    p5_response_curve: list[float] = []
    p95_response_curve: list[float] = []

    for j in range(n_conc):
        resps = responses_by_conc[j]
        mean_r = sum(resps) / n_subjects
        sorted_resps = sorted(resps)
        p5 = _percentile(sorted_resps, 5.0)
        p95 = _percentile(sorted_resps, 95.0)
        mean_response_curve.append(mean_r)
        p5_response_curve.append(p5)
        p95_response_curve.append(p95)

    # Run variability analysis on generated population
    analysis = analyze_dose_response_variability(
        ec50_values=ec50_values,
        emax_values=emax_values,
        hill_values=[hill_n] * n_subjects,
        standard_dose_conc_mg_L=ec50_mean,
        responder_threshold_fraction=0.5,
    )

    return {
        "ec50_values": ec50_values,
        "emax_values": emax_values,
        "mean_response_curve": mean_response_curve,
        "p5_response_curve": p5_response_curve,
        "p95_response_curve": p95_response_curve,
        "dose_concentrations": list(dose_concentrations),
        "analysis": analysis,
    }


__all__ = [
    "DoseResponseVariabilityResult",
    "analyze_dose_response_variability",
    "simulate_population_dose_response",
]
