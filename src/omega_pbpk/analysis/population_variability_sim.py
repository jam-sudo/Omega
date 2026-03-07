"""Population PK variability simulation.

Simulates a cohort of virtual patients with log-normal variability in
clearance (CL) and volume of distribution (Vd). Uses Python's built-in
``random`` module with Box-Muller transform for reproducibility without
NumPy dependency in parameter sampling.
"""

from __future__ import annotations

import math
import random as _random
from dataclasses import dataclass, field

__all__ = [
    "PopulationVariabilityResult",
    "simulate_population_pk",
    "therapeutic_range_analysis",
]

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PopulationVariabilityResult:
    """Result of a population PK variability simulation."""

    n_subjects: int
    dose_mg: float
    cl_mean: float
    vd_mean: float
    cl_cv_pct: float
    vd_cv_pct: float
    individual_troughs: list[float] = field(default_factory=list)
    individual_cl: list[float] = field(default_factory=list)
    individual_vd: list[float] = field(default_factory=list)
    p5: float = 0.0
    p25: float = 0.0
    p50: float = 0.0
    p75: float = 0.0
    p95: float = 0.0
    cv_trough_pct: float = 0.0
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _box_muller(rng: _random.Random) -> float:
    """Generate a standard normal variate using Box-Muller transform."""
    u1 = rng.random()
    u2 = rng.random()
    # Avoid log(0)
    while u1 < 1e-15:
        u1 = rng.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _lognormal_sample(mean: float, cv_pct: float, rng: _random.Random) -> float:
    """Sample from log-normal distribution with given mean and CV (%).

    Uses: X = mean * exp(omega * z)
    where omega = sqrt(log(1 + (cv/100)^2)) and z ~ N(0, 1).
    """
    cv_frac = cv_pct / 100.0
    omega = math.sqrt(math.log(1.0 + cv_frac * cv_frac))
    z = _box_muller(rng)
    return mean * math.exp(omega * z)


def _simulate_subject_pk(
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    t_end_h: float,
    dt_h: float,
    interval_h: float,
    n_doses: int,
    ka_per_h: float,
    f_oral: float,
) -> float:
    """Run forward Euler 1-cpt multi-dose PK and return steady-state trough.

    For oral dosing: dA_gut/dt = -ka * A_gut;  dC/dt = ka*A_gut*F/Vd - kel*C
    For IV (ka=0 sentinel): dC/dt = dose*F/Vd at t=0, then -kel*C

    Returns the trough at the end of the last dose interval.
    """
    kel = cl_L_per_h / vd_L
    use_oral = ka_per_h > 0.0

    dose_schedule = [i * interval_h for i in range(n_doses)]
    n_steps = max(int(round(t_end_h / dt_h)), 1)

    c_central = 0.0  # mg/L
    c_gut = 0.0       # mg in gut

    dose_idx = 0
    trough = 0.0

    for step in range(n_steps + 1):
        t = step * dt_h

        # Administer dose
        while dose_idx < len(dose_schedule) and dose_schedule[dose_idx] <= t + 1e-9:
            dose_t = dose_schedule[dose_idx]
            if abs(t - dose_t) < dt_h * 0.5 + 1e-9:
                if use_oral:
                    c_gut += dose_mg * f_oral
                else:
                    c_central += dose_mg * f_oral / vd_L
            dose_idx += 1

        if step < n_steps:
            if use_oral:
                dc_gut_dt = -ka_per_h * c_gut
                dc_central_dt = (ka_per_h * c_gut) / vd_L - kel * c_central
                c_gut = max(c_gut + dc_gut_dt * dt_h, 0.0)
                c_central = max(c_central + dc_central_dt * dt_h, 0.0)
            else:
                dc_central_dt = -kel * c_central
                c_central = max(c_central + dc_central_dt * dt_h, 0.0)

    # Trough = concentration at end of last dose interval
    trough = max(c_central, 0.0)
    return trough


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Compute percentile from a sorted list using linear interpolation."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_values[0]
    idx = (pct / 100.0) * (n - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_values[lo]
    frac = idx - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_population_pk(
    n_subjects: int,
    dose_mg: float,
    cl_mean_L_per_h: float,
    vd_mean_L: float,
    cl_cv_pct: float = 30.0,
    vd_cv_pct: float = 30.0,
    t_end_h: float = 168.0,
    dt_h: float = 0.25,
    interval_h: float = 24.0,
    n_doses: int = 7,
    ka_per_h: float = 1.0,
    f_oral: float = 1.0,
    seed: int = 42,
) -> PopulationVariabilityResult:
    """Simulate population PK variability across a cohort of virtual patients.

    Individual CL and Vd are sampled from log-normal distributions:
        CL_i = cl_mean * exp(omega_CL * z_CL)
        Vd_i = vd_mean * exp(omega_Vd * z_Vd)
    where omega = sqrt(log(1 + (cv/100)^2)) and z ~ N(0,1) via Box-Muller.

    Parameters
    ----------
    n_subjects:
        Number of virtual patients to simulate.
    dose_mg:
        Dose per administration (mg).
    cl_mean_L_per_h:
        Population mean clearance (L/h).
    vd_mean_L:
        Population mean volume of distribution (L).
    cl_cv_pct:
        Between-subject CV% for CL.
    vd_cv_pct:
        Between-subject CV% for Vd.
    t_end_h:
        Total simulation time (h).
    dt_h:
        Forward Euler time step (h).
    interval_h:
        Dosing interval (h).
    n_doses:
        Number of doses.
    ka_per_h:
        Absorption rate constant (h^-1). Use 0.0 for IV bolus.
    f_oral:
        Oral bioavailability fraction (0–1).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    PopulationVariabilityResult
    """
    # --- Validation ---
    if n_subjects < 1:
        raise ValueError("n_subjects must be >= 1")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_mean_L_per_h <= 0:
        raise ValueError("cl_mean_L_per_h must be > 0")
    if vd_mean_L <= 0:
        raise ValueError("vd_mean_L must be > 0")
    if cl_cv_pct < 0:
        raise ValueError("cl_cv_pct must be >= 0")
    if vd_cv_pct < 0:
        raise ValueError("vd_cv_pct must be >= 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")
    if interval_h <= 0:
        raise ValueError("interval_h must be > 0")
    if n_doses < 1:
        raise ValueError("n_doses must be >= 1")
    if ka_per_h < 0:
        raise ValueError("ka_per_h must be >= 0")
    if not 0.0 < f_oral <= 1.0:
        raise ValueError("f_oral must be in (0, 1]")

    rng = _random.Random(seed)

    individual_cl: list[float] = []
    individual_vd: list[float] = []
    individual_troughs: list[float] = []

    for _ in range(n_subjects):
        cl_i = _lognormal_sample(cl_mean_L_per_h, cl_cv_pct, rng)
        vd_i = _lognormal_sample(vd_mean_L, vd_cv_pct, rng)

        # Clamp to reasonable physiological bounds (>0)
        cl_i = max(cl_i, 1e-6)
        vd_i = max(vd_i, 1e-6)

        trough = _simulate_subject_pk(
            dose_mg=dose_mg,
            cl_L_per_h=cl_i,
            vd_L=vd_i,
            t_end_h=t_end_h,
            dt_h=dt_h,
            interval_h=interval_h,
            n_doses=n_doses,
            ka_per_h=ka_per_h,
            f_oral=f_oral,
        )

        individual_cl.append(cl_i)
        individual_vd.append(vd_i)
        individual_troughs.append(trough)

    # --- Percentiles ---
    sorted_troughs = sorted(individual_troughs)
    p5 = _percentile(sorted_troughs, 5.0)
    p25 = _percentile(sorted_troughs, 25.0)
    p50 = _percentile(sorted_troughs, 50.0)
    p75 = _percentile(sorted_troughs, 75.0)
    p95 = _percentile(sorted_troughs, 95.0)

    # --- CV of trough ---
    n = len(individual_troughs)
    mean_trough = sum(individual_troughs) / n if n > 0 else 0.0
    if mean_trough > 1e-9 and n > 1:
        variance = sum((x - mean_trough) ** 2 for x in individual_troughs) / (n - 1)
        cv_trough_pct = 100.0 * math.sqrt(variance) / mean_trough
    else:
        cv_trough_pct = 0.0

    notes = [
        f"n={n_subjects} subjects, CL CV={cl_cv_pct:.1f}%, Vd CV={vd_cv_pct:.1f}%",
        f"Trough CV={cv_trough_pct:.1f}%, median={p50:.4f} mg/L",
    ]

    return PopulationVariabilityResult(
        n_subjects=n_subjects,
        dose_mg=dose_mg,
        cl_mean=cl_mean_L_per_h,
        vd_mean=vd_mean_L,
        cl_cv_pct=cl_cv_pct,
        vd_cv_pct=vd_cv_pct,
        individual_troughs=individual_troughs,
        individual_cl=individual_cl,
        individual_vd=individual_vd,
        p5=p5,
        p25=p25,
        p50=p50,
        p75=p75,
        p95=p95,
        cv_trough_pct=cv_trough_pct,
        notes=notes,
    )


def therapeutic_range_analysis(
    troughs: list[float],
    lower_mg_L: float,
    upper_mg_L: float,
) -> dict:
    """Classify trough concentrations relative to a therapeutic range.

    Parameters
    ----------
    troughs:
        List of individual trough concentrations (mg/L).
    lower_mg_L:
        Lower bound of therapeutic range (mg/L).
    upper_mg_L:
        Upper bound of therapeutic range (mg/L).

    Returns
    -------
    dict with keys ``pct_below``, ``pct_in_range``, ``pct_above``.
    """
    if not troughs:
        raise ValueError("troughs must be a non-empty list")
    if lower_mg_L < 0:
        raise ValueError("lower_mg_L must be >= 0")
    if upper_mg_L <= lower_mg_L:
        raise ValueError("upper_mg_L must be > lower_mg_L")

    n = len(troughs)
    n_below = sum(1 for t in troughs if t < lower_mg_L)
    n_above = sum(1 for t in troughs if t > upper_mg_L)
    n_in_range = n - n_below - n_above

    return {
        "pct_below": 100.0 * n_below / n,
        "pct_in_range": 100.0 * n_in_range / n,
        "pct_above": 100.0 * n_above / n,
    }
