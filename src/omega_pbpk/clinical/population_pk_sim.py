"""Population PK simulation with between-subject variability — Phase 476.

Simulates PK profiles for a virtual patient population using log-normal
between-subject variability (BSV). Each subject's individual parameters
are sampled as:
    CL_i  = CL_typ  * exp(eta_CL),   eta_CL  ~ N(0, omega_CL²)
    Vd_i  = Vd_typ  * exp(eta_Vd),   eta_Vd  ~ N(0, omega_Vd²)
    ka_i  = ka_typ  * exp(eta_ka),   eta_ka  ~ N(0, omega_ka²)

Uses Box-Muller transform via Python's standard random module (no numpy).
Forward Euler ODE integration, trapezoidal AUC.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class PopulationPKResult:
    """Result of a population PK simulation.

    Attributes:
        n_subjects: Number of simulated subjects.
        cl_values: Individual CL values (L/h).
        vd_values: Individual Vd values (L).
        cmax_values: Individual Cmax values (mg/L).
        auc_values: Individual AUC(0-t) values (mg·h/L).
        times_h: Shared time vector (h).
        concentration_profiles: Concentration vs time per subject (n × T).
        cmax_p5: 5th percentile of Cmax across subjects.
        cmax_p50: Median Cmax.
        cmax_p95: 95th percentile of Cmax.
        auc_p5: 5th percentile of AUC.
        auc_p50: Median AUC.
        auc_p95: 95th percentile of AUC.
        notes: Simulation notes.
    """

    n_subjects: int
    cl_values: list[float] = field(default_factory=list)
    vd_values: list[float] = field(default_factory=list)
    cmax_values: list[float] = field(default_factory=list)
    auc_values: list[float] = field(default_factory=list)
    times_h: list[float] = field(default_factory=list)
    concentration_profiles: list[list[float]] = field(default_factory=list)
    cmax_p5: float = 0.0
    cmax_p50: float = 0.0
    cmax_p95: float = 0.0
    auc_p5: float = 0.0
    auc_p50: float = 0.0
    auc_p95: float = 0.0
    notes: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_positive(value: float, name: str) -> None:
    """Raise ValueError if value is not strictly positive."""
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}")


def _box_muller(rng: random.Random) -> float:
    """Generate one standard normal sample using the Box-Muller transform."""
    u1 = rng.random()
    u2 = rng.random()
    # Avoid log(0)
    while u1 == 0.0:
        u1 = rng.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _sample_lognormal(rng: random.Random, typical: float, omega: float) -> float:
    """Sample individual parameter from log-normal distribution.

    Parameters
    ----------
    rng:
        Random number generator.
    typical:
        Typical (population) parameter value.
    omega:
        Between-subject variability (standard deviation on log scale).

    Returns
    -------
    float
        Individual parameter value.
    """
    eta = _box_muller(rng) * omega
    return typical * math.exp(eta)


def _percentile(values: list[float], p: float) -> float:
    """Compute percentile p (0-100) from a list using linear interpolation."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    idx = (p / 100.0) * (n - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= n:
        return sorted_vals[-1]
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _trapezoidal_auc(times: list[float], concentrations: list[float]) -> float:
    """Compute AUC using the trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (concentrations[i - 1] + concentrations[i]) * dt
    return auc


def _simulate_subject_iv(
    dose_mg: float,
    cl: float,
    vd: float,
    times: list[float],
) -> list[float]:
    """Simulate IV bolus 1-compartment PK using Forward Euler.

    C(0) = dose/Vd, dC/dt = -(CL/Vd)*C
    """
    ke = cl / vd
    c = dose_mg / vd  # initial concentration (mg/L)
    profile: list[float] = []
    for i, _t in enumerate(times):
        profile.append(c)
        if i < len(times) - 1:
            dt = times[i + 1] - times[i]
            c = max(0.0, c - ke * c * dt)
    return profile


def _simulate_subject_oral(
    dose_mg: float,
    cl: float,
    vd: float,
    ka: float,
    times: list[float],
) -> list[float]:
    """Simulate oral 1-compartment PK using Forward Euler.

    dA_gut/dt = -ka * A_gut
    dC/dt     = ka * A_gut / Vd - (CL/Vd) * C
    """
    ke = cl / vd
    a_gut = dose_mg  # amount in gut (mg)
    c = 0.0          # plasma concentration (mg/L)
    profile: list[float] = []
    for i, _t in enumerate(times):
        profile.append(c)
        if i < len(times) - 1:
            dt = times[i + 1] - times[i]
            da_gut = -ka * a_gut * dt
            dc = (ka * a_gut / vd - ke * c) * dt
            a_gut = max(0.0, a_gut + da_gut)
            c = max(0.0, c + dc)
    return profile


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_population(
    n_subjects: int,
    cl_typical: float,
    vd_typical: float,
    dose_mg: float,
    omega_cl: float = 0.3,
    omega_vd: float = 0.2,
    route: str = "iv",
    ka_typical: float = 1.0,
    omega_ka: float = 0.3,
    t_end_h: float = 24.0,
    dt_h: float = 0.5,
    seed: int | None = None,
) -> PopulationPKResult:
    """Simulate PK profiles for a virtual patient population.

    Parameters
    ----------
    n_subjects:
        Number of subjects to simulate (≥ 2).
    cl_typical:
        Typical population clearance (L/h).
    vd_typical:
        Typical population volume of distribution (L).
    dose_mg:
        Dose administered (mg).
    omega_cl:
        BSV for CL (SD on log scale; CV% ≈ omega for small omega).
    omega_vd:
        BSV for Vd.
    route:
        Administration route: 'iv' or 'oral'.
    ka_typical:
        Typical absorption rate constant (1/h; oral only).
    omega_ka:
        BSV for ka (oral only).
    t_end_h:
        Simulation duration (h).
    dt_h:
        Time step (h).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    PopulationPKResult
    """
    if n_subjects < 2:
        raise ValueError(f"n_subjects must be >= 2, got {n_subjects}")
    _validate_positive(cl_typical, "cl_typical")
    _validate_positive(vd_typical, "vd_typical")
    _validate_positive(dose_mg, "dose_mg")
    _validate_positive(t_end_h, "t_end_h")
    _validate_positive(dt_h, "dt_h")

    if route not in ("iv", "oral"):
        raise ValueError(f"route must be 'iv' or 'oral', got '{route}'")

    rng = random.Random(seed)

    # Build shared time vector
    n_steps = max(1, round(t_end_h / dt_h))
    times = [i * dt_h for i in range(n_steps + 1)]

    cl_values: list[float] = []
    vd_values: list[float] = []
    cmax_values: list[float] = []
    auc_values: list[float] = []
    profiles: list[list[float]] = []

    for _ in range(n_subjects):
        cl_i = _sample_lognormal(rng, cl_typical, omega_cl)
        vd_i = _sample_lognormal(rng, vd_typical, omega_vd)

        if route == "iv":
            profile = _simulate_subject_iv(dose_mg, cl_i, vd_i, times)
        else:
            ka_i = _sample_lognormal(rng, ka_typical, omega_ka)
            profile = _simulate_subject_oral(dose_mg, cl_i, vd_i, ka_i, times)

        cmax_i = max(profile)
        auc_i = _trapezoidal_auc(times, profile)

        cl_values.append(cl_i)
        vd_values.append(vd_i)
        cmax_values.append(cmax_i)
        auc_values.append(auc_i)
        profiles.append(profile)

    # Compute percentiles
    cmax_p5 = _percentile(cmax_values, 5.0)
    cmax_p50 = _percentile(cmax_values, 50.0)
    cmax_p95 = _percentile(cmax_values, 95.0)
    auc_p5 = _percentile(auc_values, 5.0)
    auc_p50 = _percentile(auc_values, 50.0)
    auc_p95 = _percentile(auc_values, 95.0)

    notes = (
        f"Population PK simulation: {n_subjects} subjects, route={route}, "
        f"CL_typ={cl_typical} L/h, Vd_typ={vd_typical} L, dose={dose_mg} mg. "
        f"omega_CL={omega_cl}, omega_Vd={omega_vd}."
    )

    return PopulationPKResult(
        n_subjects=n_subjects,
        cl_values=cl_values,
        vd_values=vd_values,
        cmax_values=cmax_values,
        auc_values=auc_values,
        times_h=times,
        concentration_profiles=profiles,
        cmax_p5=cmax_p5,
        cmax_p50=cmax_p50,
        cmax_p95=cmax_p95,
        auc_p5=auc_p5,
        auc_p50=auc_p50,
        auc_p95=auc_p95,
        notes=notes,
    )


def summarize_population(result: PopulationPKResult) -> dict:
    """Summarize population PK result with Cmax and AUC percentiles.

    Parameters
    ----------
    result:
        A PopulationPKResult instance.

    Returns
    -------
    dict
        Keys: cmax_p5, cmax_p50, cmax_p95, auc_p5, auc_p50, auc_p95,
              n_subjects, mean_cmax, mean_auc.
    """
    n = result.n_subjects
    mean_cmax = sum(result.cmax_values) / n if n > 0 else 0.0
    mean_auc = sum(result.auc_values) / n if n > 0 else 0.0

    return {
        "n_subjects": n,
        "cmax_p5": result.cmax_p5,
        "cmax_p50": result.cmax_p50,
        "cmax_p95": result.cmax_p95,
        "auc_p5": result.auc_p5,
        "auc_p50": result.auc_p50,
        "auc_p95": result.auc_p95,
        "mean_cmax": mean_cmax,
        "mean_auc": mean_auc,
    }


def identify_outliers(
    result: PopulationPKResult,
    n_sd: float = 2.0,
) -> dict[str, list[int]]:
    """Identify subjects whose Cmax or AUC lie beyond n_sd standard deviations.

    Uses log-scale statistics (geometric mean/SD) consistent with log-normal
    BSV. Outliers are subjects whose log(Cmax) or log(AUC) deviate more than
    n_sd * log_SD from the log mean.

    Parameters
    ----------
    result:
        Population simulation result.
    n_sd:
        Number of standard deviations to use as the outlier threshold.

    Returns
    -------
    dict
        'cmax_outliers': list of subject indices (0-based) for Cmax.
        'auc_outliers': list of subject indices (0-based) for AUC.
    """
    if n_sd <= 0.0:
        raise ValueError(f"n_sd must be positive, got {n_sd}")

    def _log_stats(values: list[float]) -> tuple[float, float]:
        """Return (log_mean, log_sd) for a list of positive values."""
        n = len(values)
        if n < 2:
            return 0.0, 0.0
        log_vals = [math.log(v) for v in values if v > 0.0]
        log_mean = sum(log_vals) / len(log_vals)
        variance = sum((x - log_mean) ** 2 for x in log_vals) / (len(log_vals) - 1)
        log_sd = math.sqrt(variance)
        return log_mean, log_sd

    cmax_log_mean, cmax_log_sd = _log_stats(result.cmax_values)
    auc_log_mean, auc_log_sd = _log_stats(result.auc_values)

    cmax_outliers: list[int] = []
    auc_outliers: list[int] = []

    for i, (cmax, auc) in enumerate(zip(result.cmax_values, result.auc_values, strict=True)):
        if cmax > 0.0 and cmax_log_sd > 0.0:
            z_cmax = abs(math.log(cmax) - cmax_log_mean) / cmax_log_sd
            if z_cmax > n_sd:
                cmax_outliers.append(i)
        if auc > 0.0 and auc_log_sd > 0.0:
            z_auc = abs(math.log(auc) - auc_log_mean) / auc_log_sd
            if z_auc > n_sd:
                auc_outliers.append(i)

    return {
        "cmax_outliers": cmax_outliers,
        "auc_outliers": auc_outliers,
    }
