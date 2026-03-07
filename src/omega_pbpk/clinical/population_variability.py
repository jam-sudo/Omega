"""Population variability simulation for PK parameters (Monte Carlo).

Simulates inter-individual variability in CL, Vd, ka using a lognormal distribution
sampled via a linear congruential generator (LCG) for reproducibility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubjectPKResult:
    subject_id: int
    cl_L_per_h: float
    vd_L: float
    ka_per_h: float
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_L: float
    t_half_h: float


@dataclass(frozen=True)
class PopulationVariabilityResult:
    n_subjects: int
    dose_mg: float
    route: str
    mean_cmax: float
    cv_cmax_pct: float  # coefficient of variation (%)
    p5_cmax: float
    p95_cmax: float
    mean_auc: float
    cv_auc_pct: float
    p5_auc: float
    p95_auc: float
    mean_t_half: float
    subjects: tuple  # tuple of SubjectPKResult
    fraction_above_target: float  # fraction with AUC > target_auc
    fraction_below_mic: float  # fraction with Cmax < mic_mg_L
    notes: str


# ---------------------------------------------------------------------------
# LCG random number generator
# ---------------------------------------------------------------------------


def lcg_random(seed: int, n: int) -> list:
    """Simple Linear Congruential Generator for reproducible pseudo-random numbers.

    Returns n uniform random numbers in [0, 1).

    Parameters
    ----------
    seed : int
        Initial seed value.
    n : int
        Number of random numbers to generate.

    Returns
    -------
    list of float
        n uniform random numbers in [0, 1).
    """
    modulus = 2**32
    a = 1664525
    c = 1013904223

    x = seed & 0xFFFFFFFF  # ensure 32-bit
    result = []
    for _ in range(n):
        x = (a * x + c) % modulus
        result.append(x / modulus)
    return result


# ---------------------------------------------------------------------------
# Normal quantile approximation (Beasley-Springer-Moro)
# ---------------------------------------------------------------------------


def _normal_quantile(u: float) -> float:
    """Rational approximation to the normal quantile (inverse CDF).

    Uses the Beasley-Springer-Moro approximation:
        t = sqrt(-2 * ln(min(u, 1-u)))
        z = t - (a0 + a1*t + a2*t^2) / (1 + b1*t + b2*t^2 + b3*t^3)
        sign: negative if u <= 0.5, positive otherwise.

    Parameters
    ----------
    u : float
        Uniform random number in (0, 1).

    Returns
    -------
    float
        Approximate normal quantile.
    """
    # Clamp to avoid log(0)
    u = max(1e-10, min(1.0 - 1e-10, u))

    a0 = 2.515517
    a1 = 0.802853
    a2 = 0.010328
    b1 = 1.432788
    b2 = 0.189269
    b3 = 0.001308

    p = u if u <= 0.5 else 1.0 - u
    t = math.sqrt(-2.0 * math.log(p))
    numerator = a0 + a1 * t + a2 * t * t
    denominator = 1.0 + b1 * t + b2 * t * t + b3 * t * t * t
    rational = t - numerator / denominator

    # Sign: for u <= 0.5 the quantile is negative
    return -rational if u <= 0.5 else rational


# ---------------------------------------------------------------------------
# Lognormal sampler
# ---------------------------------------------------------------------------


def lognormal_sample(mu: float, omega: float, u: float) -> float:
    """Sample from lognormal distribution using inverse CDF.

    Parameters
    ----------
    mu : float
        Typical (median) value (geometric mean).
    omega : float
        CV parameter (lognormal standard deviation of log).
    u : float
        Uniform random number in [0, 1).

    Returns
    -------
    float
        Lognormal sample: mu * exp(omega * z) where z ~ N(0,1).
    """
    z = _normal_quantile(u)
    return mu * math.exp(omega * z)


# ---------------------------------------------------------------------------
# 1-compartment PK simulation (analytical)
# ---------------------------------------------------------------------------


def _simulate_1cpt(
    dose_mg: float,
    cl: float,
    vd: float,
    ka: float,
    route: str,
    t_end_h: float,
    dt_h: float,
) -> tuple:
    """Simulate 1-compartment PK, return (cmax, tmax, auc, times, concs).

    Uses analytical solution:
    - IV: C(t) = (dose/vd) * exp(-ke*t)
    - Oral: C(t) = dose * ka / (vd * (ka - ke)) * (exp(-ke*t) - exp(-ka*t))
      (degenerate case ka==ke handled separately)
    """
    ke = cl / vd if vd > 0 else 0.0
    n_steps = int(math.ceil(t_end_h / dt_h)) + 1
    times = [i * dt_h for i in range(n_steps)]

    if route == "iv":
        concs = [(dose_mg / vd) * math.exp(-ke * t) for t in times]
    else:
        # Oral 1-cpt analytical
        if abs(ka - ke) < 1e-8:
            # Degenerate: ka == ke → C(t) = dose * ka / vd * t * exp(-ka*t)
            concs = [dose_mg * ka / vd * t * math.exp(-ka * t) for t in times]
        else:
            factor = dose_mg * ka / (vd * (ka - ke))
            concs = [factor * (math.exp(-ke * t) - math.exp(-ka * t)) for t in times]
        concs = [max(c, 0.0) for c in concs]

    cmax = max(concs)
    tmax = times[concs.index(cmax)]

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, n_steps):
        auc += 0.5 * (concs[i - 1] + concs[i]) * dt_h

    return cmax, tmax, auc


# ---------------------------------------------------------------------------
# Population statistics helpers
# ---------------------------------------------------------------------------


def _mean(values: list) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list, mean_val: float) -> float:
    if len(values) < 2:
        return 0.0
    var = sum((v - mean_val) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def _percentile(sorted_vals: list, p: float) -> float:
    """Return the p-th percentile from a sorted list."""
    if not sorted_vals:
        return 0.0
    idx = int(p / 100.0 * len(sorted_vals))
    idx = max(0, min(idx, len(sorted_vals) - 1))
    return sorted_vals[idx]


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------


def simulate_population_pk(
    dose_mg: float,
    typical_cl_L_per_h: float,
    typical_vd_L: float,
    omega_cl: float = 0.3,
    omega_vd: float = 0.2,
    omega_ka: float = 0.4,
    typical_ka_per_h: float = 1.0,
    route: str = "oral",
    n_subjects: int = 100,
    seed: int = 42,
    t_end_h: float = 24.0,
    dt_h: float = 0.2,
    target_auc_mg_h_L: float = 50.0,
    mic_mg_L: float = 0.5,
) -> PopulationVariabilityResult:
    """Simulate population PK variability using Monte Carlo sampling.

    Samples individual CL, Vd, ka from lognormal distributions using an LCG
    for reproducibility. For each subject, simulates 1-compartment PK and
    computes Cmax, Tmax, AUC, t½.

    Parameters
    ----------
    dose_mg : float
        Dose in mg.
    typical_cl_L_per_h : float
        Typical (population median) clearance (L/h).
    typical_vd_L : float
        Typical volume of distribution (L).
    omega_cl : float
        Between-subject CV for CL (lognormal SD of log).
    omega_vd : float
        Between-subject CV for Vd.
    omega_ka : float
        Between-subject CV for ka.
    typical_ka_per_h : float
        Typical oral absorption rate constant (h^-1).
    route : str
        Dosing route ('oral' or 'iv').
    n_subjects : int
        Number of virtual subjects.
    seed : int
        LCG seed for reproducibility.
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Time step (h).
    target_auc_mg_h_L : float
        Target AUC threshold for fraction_above_target.
    mic_mg_L : float
        MIC threshold for fraction_below_mic.

    Returns
    -------
    PopulationVariabilityResult
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if typical_cl_L_per_h <= 0:
        raise ValueError(f"typical_cl_L_per_h must be > 0, got {typical_cl_L_per_h}")
    if typical_vd_L <= 0:
        raise ValueError(f"typical_vd_L must be > 0, got {typical_vd_L}")
    if n_subjects < 1:
        raise ValueError(f"n_subjects must be >= 1, got {n_subjects}")
    route = route.lower()
    if route not in {"oral", "iv"}:
        raise ValueError(f"route must be 'oral' or 'iv', got {route!r}")

    # Generate 3 * n_subjects random numbers: u_cl, u_vd, u_ka per subject
    randoms = lcg_random(seed, 3 * n_subjects)

    subject_results = []

    for i in range(n_subjects):
        u_cl = randoms[3 * i]
        u_vd = randoms[3 * i + 1]
        u_ka = randoms[3 * i + 2]

        cl_i = lognormal_sample(typical_cl_L_per_h, omega_cl, u_cl)
        vd_i = lognormal_sample(typical_vd_L, omega_vd, u_vd)
        ka_i = lognormal_sample(typical_ka_per_h, omega_ka, u_ka)

        cmax_i, tmax_i, auc_i = _simulate_1cpt(
            dose_mg=dose_mg,
            cl=cl_i,
            vd=vd_i,
            ka=ka_i,
            route=route,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )

        t_half_i = math.log(2.0) * vd_i / cl_i

        subject_results.append(
            SubjectPKResult(
                subject_id=i + 1,
                cl_L_per_h=cl_i,
                vd_L=vd_i,
                ka_per_h=ka_i,
                cmax_mg_L=cmax_i,
                tmax_h=tmax_i,
                auc_mg_h_L=auc_i,
                t_half_h=t_half_i,
            )
        )

    # Collect population statistics
    cmax_vals = [s.cmax_mg_L for s in subject_results]
    auc_vals = [s.auc_mg_h_L for s in subject_results]
    t_half_vals = [s.t_half_h for s in subject_results]

    mean_cmax = _mean(cmax_vals)
    std_cmax = _std(cmax_vals, mean_cmax)
    cv_cmax_pct = (std_cmax / mean_cmax * 100.0) if mean_cmax > 0 else 0.0

    mean_auc = _mean(auc_vals)
    std_auc = _std(auc_vals, mean_auc)
    cv_auc_pct = (std_auc / mean_auc * 100.0) if mean_auc > 0 else 0.0

    mean_t_half = _mean(t_half_vals)

    sorted_cmax = sorted(cmax_vals)
    sorted_auc = sorted(auc_vals)

    p5_cmax = _percentile(sorted_cmax, 5.0)
    p95_cmax = _percentile(sorted_cmax, 95.0)
    p5_auc = _percentile(sorted_auc, 5.0)
    p95_auc = _percentile(sorted_auc, 95.0)

    n_above_target = sum(1 for a in auc_vals if a > target_auc_mg_h_L)
    fraction_above_target = n_above_target / n_subjects

    n_below_mic = sum(1 for c in cmax_vals if c < mic_mg_L)
    fraction_below_mic = n_below_mic / n_subjects

    notes = (
        f"PopPK simulation | n={n_subjects} subjects | {route.upper()} {dose_mg} mg | "
        f"seed={seed} | CL_typ={typical_cl_L_per_h} L/h | Vd_typ={typical_vd_L} L | "
        f"omega_cl={omega_cl} omega_vd={omega_vd} omega_ka={omega_ka}"
    )

    return PopulationVariabilityResult(
        n_subjects=n_subjects,
        dose_mg=dose_mg,
        route=route,
        mean_cmax=mean_cmax,
        cv_cmax_pct=cv_cmax_pct,
        p5_cmax=p5_cmax,
        p95_cmax=p95_cmax,
        mean_auc=mean_auc,
        cv_auc_pct=cv_auc_pct,
        p5_auc=p5_auc,
        p95_auc=p95_auc,
        mean_t_half=mean_t_half,
        subjects=tuple(subject_results),
        fraction_above_target=fraction_above_target,
        fraction_below_mic=fraction_below_mic,
        notes=notes,
    )


__all__ = [
    "SubjectPKResult",
    "PopulationVariabilityResult",
    "lcg_random",
    "lognormal_sample",
    "simulate_population_pk",
]
