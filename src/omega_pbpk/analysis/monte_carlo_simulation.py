"""Monte Carlo PK simulation for uncertainty propagation — Phase 449.

Propagates uncertainty in CL and Vd through a 1-compartment multi-dose
forward Euler PK model to produce population percentile predictions.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field

__all__ = [
    "MonteCarloResult",
    "MonteCarloPKResult",
    "monte_carlo_pk",
    "run_monte_carlo_pk",
    "summarize_percentiles",
    "calculate_probability_target_attainment",
    "uncertainty_propagation",
]


# ---------------------------------------------------------------------------
# Phase 449 new API
# ---------------------------------------------------------------------------


@dataclass
class MonteCarloResult:
    """Results from run_monte_carlo_pk — Phase 449."""

    n_samples: int
    cl_samples: list[float]
    vd_samples: list[float]
    cmax_samples: list[float]
    auc_samples: list[float]
    cmax_p5: float
    cmax_p50: float
    cmax_p95: float
    auc_p5: float
    auc_p50: float
    auc_p95: float
    notes: str


def _lognormal_samples_bm(
    mean: float,
    cv: float,
    n: int,
    rng: random.Random,
) -> list[float]:
    """Sample n log-normal values using Box-Muller via stdlib random."""
    sigma_log = math.sqrt(math.log(1.0 + cv ** 2))
    mu_log = math.log(mean) - 0.5 * sigma_log ** 2
    samples: list[float] = []
    while len(samples) < n:
        u1 = rng.random()
        u2 = rng.random()
        if u1 <= 0.0:
            u1 = 1e-300
        z1 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        z2 = math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2)
        samples.append(math.exp(mu_log + sigma_log * z1))
        if len(samples) < n:
            samples.append(math.exp(mu_log + sigma_log * z2))
    return samples[:n]


def _simulate_1cpt_iv_new(
    dose_mg: float,
    cl: float,
    vd: float,
    t_end_h: float,
    dt_h: float,
) -> tuple[list[float], list[float]]:
    """1-cpt IV bolus forward Euler."""
    kel = cl / vd
    n_steps = max(int(round(t_end_h / dt_h)), 1)
    times = [0.0] * (n_steps + 1)
    concs = [0.0] * (n_steps + 1)
    c = dose_mg / vd
    concs[0] = c
    for i in range(n_steps):
        times[i] = i * dt_h
        concs[i] = max(c, 0.0)
        c = max(c - kel * c * dt_h, 0.0)
    times[n_steps] = n_steps * dt_h
    concs[n_steps] = max(c, 0.0)
    return times, concs


def _simulate_1cpt_oral_new(
    dose_mg: float,
    cl: float,
    vd: float,
    ka: float,
    t_end_h: float,
    dt_h: float,
) -> tuple[list[float], list[float]]:
    """1-cpt oral absorption forward Euler."""
    kel = cl / vd
    n_steps = max(int(round(t_end_h / dt_h)), 1)
    times = [0.0] * (n_steps + 1)
    concs = [0.0] * (n_steps + 1)
    c_gut = float(dose_mg)
    c_plasma = 0.0
    for i in range(n_steps):
        times[i] = i * dt_h
        concs[i] = max(c_plasma, 0.0)
        dc_gut = -ka * c_gut
        dc_plasma = (ka * c_gut / vd) - kel * c_plasma
        c_gut = max(c_gut + dc_gut * dt_h, 0.0)
        c_plasma = max(c_plasma + dc_plasma * dt_h, 0.0)
    times[n_steps] = n_steps * dt_h
    concs[n_steps] = max(c_plasma, 0.0)
    return times, concs


def _trapz_new(times: list[float], concs: list[float]) -> float:
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (concs[i] + concs[i - 1]) * (times[i] - times[i - 1])
    return auc


def _pctile_new(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    n = len(s)
    idx = (p / 100.0) * (n - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= n:
        return float(s[-1])
    return float(s[lo] * (1.0 - (idx - lo)) + s[hi] * (idx - lo))


def run_monte_carlo_pk(
    cl_mean: float,
    vd_mean: float,
    dose_mg: float,
    n_samples: int = 1000,
    cv_cl: float = 0.3,
    cv_vd: float = 0.3,
    route: str = "iv",
    ka_mean: float = 1.0,
    cv_ka: float = 0.3,
    t_end_h: float = 24.0,
    dt_h: float = 0.5,
    seed: int | None = None,
) -> MonteCarloResult:
    """Monte Carlo PK simulation with log-normal parameter uncertainty.

    Parameters
    ----------
    cl_mean:
        Population mean clearance (L/h). Must be > 0.
    vd_mean:
        Population mean volume of distribution (L). Must be > 0.
    dose_mg:
        Dose in mg. Must be > 0.
    n_samples:
        Number of Monte Carlo samples. Must be >= 10.
    cv_cl:
        Coefficient of variation for CL (fraction). Must be > 0.
    cv_vd:
        Coefficient of variation for Vd (fraction). Must be > 0.
    route:
        Dosing route: 'iv' or 'oral'.
    ka_mean:
        Mean absorption rate constant for oral route (h^-1). Must be > 0.
    cv_ka:
        CV for ka (fraction). Must be >= 0.
    t_end_h:
        Simulation end time (h). Must be > 0.
    dt_h:
        Forward Euler time step (h). Must be > 0.
    seed:
        Random seed for reproducibility. If None, unseeded.

    Returns
    -------
    MonteCarloResult
    """
    if cl_mean <= 0:
        raise ValueError("cl_mean must be > 0")
    if vd_mean <= 0:
        raise ValueError("vd_mean must be > 0")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if n_samples < 10:
        raise ValueError("n_samples must be >= 10")
    if cv_cl <= 0:
        raise ValueError("cv_cl must be > 0")
    if cv_vd <= 0:
        raise ValueError("cv_vd must be > 0")
    if ka_mean <= 0:
        raise ValueError("ka_mean must be > 0")
    if cv_ka < 0:
        raise ValueError("cv_ka must be >= 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")
    route_lower = route.lower()
    if route_lower not in ("iv", "oral"):
        raise ValueError("route must be 'iv' or 'oral'")

    rng = random.Random(seed)

    cl_samps = _lognormal_samples_bm(cl_mean, cv_cl, n_samples, rng)
    vd_samps = _lognormal_samples_bm(vd_mean, cv_vd, n_samples, rng)

    if route_lower == "oral" and cv_ka > 0:
        ka_samps = _lognormal_samples_bm(ka_mean, cv_ka, n_samples, rng)
    else:
        ka_samps = [ka_mean] * n_samples

    cmax_samps: list[float] = []
    auc_samps: list[float] = []

    for i in range(n_samples):
        cl_i = cl_samps[i]
        vd_i = vd_samps[i]
        if route_lower == "iv":
            times, concs = _simulate_1cpt_iv_new(dose_mg, cl_i, vd_i, t_end_h, dt_h)
        else:
            ka_i = ka_samps[i]
            times, concs = _simulate_1cpt_oral_new(dose_mg, cl_i, vd_i, ka_i, t_end_h, dt_h)
        cmax_samps.append(max(concs))
        auc_samps.append(_trapz_new(times, concs))

    notes_parts = []
    if route_lower == "oral":
        notes_parts.append("oral 1-cpt PK with ka uncertainty")
    else:
        notes_parts.append("IV 1-cpt PK")
    notes_parts.append(f"n_samples={n_samples}")

    return MonteCarloResult(
        n_samples=n_samples,
        cl_samples=cl_samps,
        vd_samples=vd_samps,
        cmax_samples=cmax_samps,
        auc_samples=auc_samps,
        cmax_p5=_pctile_new(cmax_samps, 5),
        cmax_p50=_pctile_new(cmax_samps, 50),
        cmax_p95=_pctile_new(cmax_samps, 95),
        auc_p5=_pctile_new(auc_samps, 5),
        auc_p50=_pctile_new(auc_samps, 50),
        auc_p95=_pctile_new(auc_samps, 95),
        notes="; ".join(notes_parts),
    )


def summarize_percentiles(
    values: list[float],
    percentiles: tuple[int, ...] = (5, 50, 95),
) -> dict[str, float]:
    """Compute specified percentiles of a list of values.

    Parameters
    ----------
    values:
        List of numeric values.
    percentiles:
        Tuple of percentile values (0-100).

    Returns
    -------
    dict mapping 'p{p}' -> value for each percentile in percentiles.
    """
    if not values:
        raise ValueError("values must not be empty")
    return {f"p{p}": _pctile_new(values, float(p)) for p in percentiles}


def calculate_probability_target_attainment(
    values: list[float],
    target: float,
) -> float:
    """Calculate fraction of values at or above target (PTA).

    Parameters
    ----------
    values:
        List of simulated PK metric values.
    target:
        Target threshold.

    Returns
    -------
    Fraction of values >= target (0.0 to 1.0).
    """
    if not values:
        raise ValueError("values must not be empty")
    return sum(1 for v in values if v >= target) / len(values)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class MonteCarloPKResult:
    """Results from a Monte Carlo PK uncertainty simulation."""

    drug_name: str
    dose_mg: float
    n_simulations: int
    cl_mean: float
    vd_mean: float
    cl_cv_pct: float
    vd_cv_pct: float
    # AUC percentiles (last dosing interval)
    auc_p5: float
    auc_p50: float
    auc_p95: float
    auc_cv_pct: float
    # Cmax percentiles
    cmax_p5: float
    cmax_p50: float
    cmax_p95: float
    cmax_cv_pct: float
    # Trough percentiles
    trough_p5: float
    trough_p50: float
    trough_p95: float
    # Individual distributions
    individual_aucs: list[float] = field(default_factory=list)
    individual_cmaxes: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _box_muller_lognormal(
    mu_log: float,
    sigma_log: float,
    n: int,
    seed: int,
) -> list[float]:
    """Sample n values from log-normal using Box-Muller transform.

    Parameters
    ----------
    mu_log:
        Mean of the underlying normal (log-space mean).
    sigma_log:
        SD of the underlying normal (log-space SD = log(1 + CV²)^0.5).
    n:
        Number of samples.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    list of n positive samples from log-normal(mu_log, sigma_log).
    """
    # Simple LCG for reproducible uniform variates without numpy
    # Using Box-Muller to convert pairs of uniforms → standard normals
    # Then shift/scale to log-normal.

    # LCG parameters (Numerical Recipes)
    a = 1664525
    c = 1013904223
    m = 2**32

    state = (seed & 0xFFFFFFFF)
    samples: list[float] = []

    i = 0
    while len(samples) < n:
        # Generate two uniform variates
        state = (a * state + c) % m
        u1 = (state + 1) / (m + 1)  # avoid exact 0
        state = (a * state + c) % m
        u2 = state / m

        # Box-Muller
        z1 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        z2 = math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2)

        samples.append(math.exp(mu_log + sigma_log * z1))
        if len(samples) < n:
            samples.append(math.exp(mu_log + sigma_log * z2))

        i += 1

    return samples[:n]


def _percentile(data: list[float], p: float) -> float:
    """Compute the p-th percentile of sorted or unsorted data (linear interpolation)."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    n = len(sorted_data)
    idx = (p / 100.0) * (n - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= n:
        return float(sorted_data[-1])
    frac = idx - lo
    return float(sorted_data[lo] * (1.0 - frac) + sorted_data[hi] * frac)


def _cv_pct(data: list[float]) -> float:
    """Coefficient of variation in percent."""
    if len(data) < 2:
        return 0.0
    mean = sum(data) / len(data)
    if mean == 0.0:
        return 0.0
    variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
    return math.sqrt(variance) / mean * 100.0


def _simulate_multidose_1cpt(
    dose_mg: float,
    cl_l_per_h: float,
    vd_l: float,
    ka_per_h: float,
    f_oral: float,
    interval_h: float,
    n_doses: int,
    t_end_h: float,
    dt_h: float,
) -> tuple[list[float], list[float]]:
    """Forward Euler 1-compartment multi-dose PK simulation.

    Returns (times_h, concentrations_mg_L).
    """
    kel = cl_l_per_h / vd_l

    n_steps = max(int(round(t_end_h / dt_h)), 1)
    times: list[float] = [0.0] * (n_steps + 1)
    concs: list[float] = [0.0] * (n_steps + 1)

    # State variables
    c_gut = 0.0   # mg in gut compartment
    c_plasma = 0.0  # mg/L in plasma

    # Pre-compute dose times
    dose_times: set[int] = set()
    for d in range(n_doses):
        t_dose = d * interval_h
        step_dose = int(round(t_dose / dt_h))
        if step_dose <= n_steps:
            dose_times.add(step_dose)

    for i in range(n_steps):
        times[i] = i * dt_h
        concs[i] = max(c_plasma, 0.0)

        if i in dose_times:
            c_gut += dose_mg * f_oral

        # Forward Euler
        dc_gut = -ka_per_h * c_gut
        dc_plasma = (ka_per_h * c_gut / vd_l) - kel * c_plasma

        c_gut = max(c_gut + dc_gut * dt_h, 0.0)
        c_plasma = max(c_plasma + dc_plasma * dt_h, 0.0)

    times[n_steps] = n_steps * dt_h
    concs[n_steps] = max(c_plasma, 0.0)

    return times, concs


def _auc_trapezoid(times: list[float], concs: list[float]) -> float:
    """Compute AUC using trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (concs[i] + concs[i - 1]) * dt
    return auc


def _last_interval_auc(
    times: list[float],
    concs: list[float],
    last_dose_time_h: float,
    t_end_h: float,
) -> float:
    """Extract AUC over the last dosing interval using trapezoidal rule."""
    t_start = last_dose_time_h
    t_end = t_end_h

    interval_times: list[float] = []
    interval_concs: list[float] = []

    for t, c in zip(times, concs, strict=True):
        if t_start <= t <= t_end:
            interval_times.append(t)
            interval_concs.append(c)

    if len(interval_times) < 2:
        return _auc_trapezoid(times, concs)

    return _auc_trapezoid(interval_times, interval_concs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def monte_carlo_pk(
    drug_name: str,
    dose_mg: float,
    cl_mean: float,
    cl_cv_pct: float,
    vd_mean: float,
    vd_cv_pct: float,
    n_simulations: int = 1000,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
    interval_h: float = 12.0,
    n_doses: int = 1,
    ka_per_h: float = 1.0,
    f_oral: float = 1.0,
    seed: int = 42,
) -> MonteCarloPKResult:
    """Monte Carlo PK simulation with log-normal CL and Vd uncertainty.

    Samples CL and Vd from log-normal distributions using Box-Muller,
    runs 1-compartment forward Euler multi-dose PK for each subject,
    and returns percentile statistics for AUC, Cmax, and trough.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        Single dose in mg.
    cl_mean:
        Population mean CL (L/h).
    cl_cv_pct:
        Coefficient of variation for CL (%).
    vd_mean:
        Population mean Vd (L).
    vd_cv_pct:
        Coefficient of variation for Vd (%).
    n_simulations:
        Number of Monte Carlo subjects.
    t_end_h:
        Total simulation time (h).
    dt_h:
        Forward Euler time step (h).
    interval_h:
        Dosing interval (h); used when n_doses > 1.
    n_doses:
        Number of doses to administer.
    ka_per_h:
        First-order absorption rate constant (h^-1).
    f_oral:
        Oral bioavailability fraction (0–1).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    MonteCarloPKResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_mean <= 0:
        raise ValueError("cl_mean must be > 0")
    if vd_mean <= 0:
        raise ValueError("vd_mean must be > 0")
    if cl_cv_pct < 0:
        raise ValueError("cl_cv_pct must be >= 0")
    if vd_cv_pct < 0:
        raise ValueError("vd_cv_pct must be >= 0")
    if n_simulations < 2:
        raise ValueError("n_simulations must be >= 2")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")
    if interval_h <= 0:
        raise ValueError("interval_h must be > 0")
    if n_doses < 1:
        raise ValueError("n_doses must be >= 1")
    if ka_per_h <= 0:
        raise ValueError("ka_per_h must be > 0")
    if not (0.0 < f_oral <= 1.0):
        raise ValueError("f_oral must be in (0, 1]")

    # Log-normal parameters: sigma_log = sqrt(log(1 + CV^2))
    cl_cv = cl_cv_pct / 100.0
    vd_cv = vd_cv_pct / 100.0

    cl_sigma_log = math.sqrt(math.log(1.0 + cl_cv**2))
    vd_sigma_log = math.sqrt(math.log(1.0 + vd_cv**2))

    # Log-normal mean correction so that E[X] = mean (not median)
    cl_mu_log = math.log(cl_mean) - 0.5 * cl_sigma_log**2
    vd_mu_log = math.log(vd_mean) - 0.5 * vd_sigma_log**2

    # Sample using Box-Muller
    cl_samples = _box_muller_lognormal(cl_mu_log, cl_sigma_log, n_simulations, seed)
    vd_samples = _box_muller_lognormal(vd_mu_log, vd_sigma_log, n_simulations, seed + 1)

    aucs: list[float] = []
    cmaxes: list[float] = []
    troughs: list[float] = []

    # Last dose time for AUC extraction
    last_dose_t = (n_doses - 1) * interval_h

    for sim_idx in range(n_simulations):
        cl_i = cl_samples[sim_idx]
        vd_i = vd_samples[sim_idx]

        times, concs = _simulate_multidose_1cpt(
            dose_mg=dose_mg,
            cl_l_per_h=cl_i,
            vd_l=vd_i,
            ka_per_h=ka_per_h,
            f_oral=f_oral,
            interval_h=interval_h,
            n_doses=n_doses,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )

        auc_i = _last_interval_auc(times, concs, last_dose_t, t_end_h)
        cmax_i = max(concs)
        trough_i = concs[-1]

        aucs.append(auc_i)
        cmaxes.append(cmax_i)
        troughs.append(trough_i)

    notes: list[str] = []
    auc_cv = _cv_pct(aucs)
    cmax_cv = _cv_pct(cmaxes)

    if auc_cv > 60.0:
        notes.append(f"High AUC variability: CV={auc_cv:.1f}%")
    if cmax_cv > 60.0:
        notes.append(f"High Cmax variability: CV={cmax_cv:.1f}%")
    if n_doses > 1:
        notes.append(f"Multi-dose simulation: {n_doses} doses at {interval_h} h intervals")

    return MonteCarloPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        n_simulations=n_simulations,
        cl_mean=cl_mean,
        vd_mean=vd_mean,
        cl_cv_pct=cl_cv_pct,
        vd_cv_pct=vd_cv_pct,
        auc_p5=_percentile(aucs, 5),
        auc_p50=_percentile(aucs, 50),
        auc_p95=_percentile(aucs, 95),
        auc_cv_pct=auc_cv,
        cmax_p5=_percentile(cmaxes, 5),
        cmax_p50=_percentile(cmaxes, 50),
        cmax_p95=_percentile(cmaxes, 95),
        cmax_cv_pct=cmax_cv,
        trough_p5=_percentile(troughs, 5),
        trough_p50=_percentile(troughs, 50),
        trough_p95=_percentile(troughs, 95),
        individual_aucs=aucs,
        individual_cmaxes=cmaxes,
        notes=notes,
    )


def uncertainty_propagation(
    param_means: dict[str, float],
    param_cvs: dict[str, float],
    pk_function: Callable[..., dict[str, float]],
    output_key: str,
    n_sim: int = 500,
    seed: int = 42,
) -> dict[str, float]:
    """Generic Monte Carlo uncertainty propagation.

    Samples each parameter independently from log-normal distributions
    using Box-Muller, calls pk_function(**sampled_params), and collects
    the scalar output identified by output_key.

    Parameters
    ----------
    param_means:
        Dict mapping parameter name to population mean value (must be > 0).
    param_cvs:
        Dict mapping parameter name to CV (fraction, e.g. 0.3 for 30%).
        Must have same keys as param_means.
    pk_function:
        Callable accepting keyword arguments matching param_means keys,
        returning a dict with at least the key output_key.
    output_key:
        Key in the dict returned by pk_function to collect.
    n_sim:
        Number of Monte Carlo samples.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict with keys: mean_output, sd_output, cv_pct, p5, p50, p95
    """
    if not param_means:
        raise ValueError("param_means must not be empty")
    if set(param_means.keys()) != set(param_cvs.keys()):
        raise ValueError("param_means and param_cvs must have identical keys")
    if any(v <= 0 for v in param_means.values()):
        raise ValueError("All param_means values must be > 0")
    if any(v < 0 for v in param_cvs.values()):
        raise ValueError("All param_cvs values must be >= 0")
    if n_sim < 2:
        raise ValueError("n_sim must be >= 2")
    if not callable(pk_function):
        raise ValueError("pk_function must be callable")

    # Sample all parameters
    param_names = list(param_means.keys())
    param_samples: dict[str, list[float]] = {}

    for i, name in enumerate(param_names):
        mean = param_means[name]
        cv = param_cvs[name]
        sigma_log = math.sqrt(math.log(1.0 + cv**2)) if cv > 0 else 0.0
        mu_log = math.log(mean) - 0.5 * sigma_log**2

        if sigma_log == 0.0:
            param_samples[name] = [mean] * n_sim
        else:
            param_samples[name] = _box_muller_lognormal(mu_log, sigma_log, n_sim, seed + i)

    outputs: list[float] = []
    for sim_idx in range(n_sim):
        kwargs = {name: param_samples[name][sim_idx] for name in param_names}
        try:
            result = pk_function(**kwargs)
            val = float(result[output_key])
            outputs.append(val)
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue

    if not outputs:
        raise ValueError(f"pk_function never produced output_key='{output_key}'")

    n = len(outputs)
    mean_out = sum(outputs) / n
    if n > 1:
        variance = sum((x - mean_out) ** 2 for x in outputs) / (n - 1)
        sd_out = math.sqrt(variance)
    else:
        sd_out = 0.0

    cv_pct_out = (sd_out / mean_out * 100.0) if mean_out != 0.0 else 0.0

    return {
        "mean_output": mean_out,
        "sd_output": sd_out,
        "cv_pct": cv_pct_out,
        "p5": _percentile(outputs, 5),
        "p50": _percentile(outputs, 50),
        "p95": _percentile(outputs, 95),
        "n_successful": float(n),
    }
