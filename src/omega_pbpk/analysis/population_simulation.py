"""Population-level PK simulation using LCG-based inter-individual variability."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "PopulationPKResult",
    "simulate_population_pk",
    "compute_percentiles",
    "population_summary_stats",
]


class LCG:
    """Linear congruential generator for deterministic pseudo-random numbers."""

    def __init__(self, seed: int) -> None:
        self.state = seed

    def next_float(self) -> float:
        """Return a float in [0, 1)."""
        self.state = (1664525 * self.state + 1013904223) % (2**32)
        return self.state / (2**32)

    def normal(self) -> float:
        """Return a standard normal sample using Box-Muller transform."""
        u1 = max(1e-10, self.next_float())
        u2 = self.next_float()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


@dataclass
class PopulationPKResult:
    """Result of population PK simulation."""

    n_subjects: int
    dose_mg: float
    cl_mean: float
    vd_mean: float
    ka_mean: float
    subject_aucs: list = field(default_factory=list)
    subject_cmaxes: list = field(default_factory=list)
    subject_cls: list = field(default_factory=list)
    subject_vds: list = field(default_factory=list)
    auc_p5: float = 0.0
    auc_p50: float = 0.0
    auc_p95: float = 0.0
    cmax_p5: float = 0.0
    cmax_p50: float = 0.0
    cmax_p95: float = 0.0
    cl_cv_pct: float = 0.0
    vd_cv_pct: float = 0.0
    auc_cv_pct: float = 0.0
    fraction_above_mec: float = 0.0
    notes: str = ""


def compute_percentiles(values: list[float], percentiles: list[float]) -> list[float]:
    """Compute percentiles using linear interpolation on sorted data.

    Args:
        values: list of numeric values
        percentiles: list of percentile values (0-100)

    Returns:
        list of interpolated percentile values
    """
    if not values:
        return [0.0] * len(percentiles)

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    results = []

    for p in percentiles:
        # Linear interpolation
        idx = (p / 100.0) * (n - 1)
        lo = int(idx)
        hi = lo + 1
        frac = idx - lo

        if hi >= n:
            results.append(sorted_vals[-1])
        elif lo < 0:
            results.append(sorted_vals[0])
        else:
            val = sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])
            results.append(val)

    return results


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    variance = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def _simulate_subject_pk(
    dose_mg: float,
    cl: float,
    vd: float,
    ka: float,
    f_oral: float,
    t_end_h: float,
    dt_h: float,
) -> tuple[float, float]:
    """Simulate 1-compartment oral PK using Forward Euler. Returns (AUC, Cmax)."""
    # 1-cpt oral model:
    # dA_gut/dt = -ka * A_gut
    # dC_sys/dt = ka * A_gut * F / Vd - (CL/Vd) * C_sys
    a_gut = dose_mg * f_oral
    c_sys = 0.0

    ke = cl / vd  # elimination rate constant

    cmax = 0.0
    auc = 0.0
    prev_c = c_sys
    t = 0.0

    n_steps = int(t_end_h / dt_h) + 1
    for step in range(n_steps):
        if step > 0:
            # Trapezoidal AUC
            auc += 0.5 * (prev_c + c_sys) * dt_h

        if c_sys > cmax:
            cmax = c_sys

        prev_c = c_sys

        # Forward Euler
        da_gut = -ka * a_gut
        # Note: f_oral already baked into a_gut initial condition
        dc_sys = ka * a_gut / vd - ke * c_sys

        a_gut += da_gut * dt_h
        c_sys += dc_sys * dt_h

        if a_gut < 0.0:
            a_gut = 0.0
        if c_sys < 0.0:
            c_sys = 0.0

        t += dt_h

    return auc, cmax


def simulate_population_pk(
    n_subjects: int = 100,
    dose_mg: float = 100.0,
    cl_mean: float = 5.0,
    vd_mean: float = 50.0,
    ka_mean: float = 1.0,
    omega_cl: float = 0.4,
    omega_vd: float = 0.3,
    omega_ka: float = 0.5,
    f_oral: float = 0.8,
    t_end_h: float = 24.0,
    dt_h: float = 0.5,
    seed: int = 12345,
    mec: float | None = None,
) -> PopulationPKResult:
    """Simulate population PK with log-normal inter-individual variability using LCG RNG.

    Args:
        n_subjects: number of virtual subjects
        dose_mg: oral dose in mg
        cl_mean: typical population CL (L/h)
        vd_mean: typical population Vd (L)
        ka_mean: typical population ka (1/h)
        omega_cl: log-normal SD for CL variability
        omega_vd: log-normal SD for Vd variability
        omega_ka: log-normal SD for ka variability
        f_oral: oral bioavailability fraction
        t_end_h: simulation duration in hours
        dt_h: time step in hours
        seed: RNG seed for reproducibility
        mec: minimum effective concentration (mg/L) for fraction_above_mec calculation

    Returns:
        PopulationPKResult with individual and summary PK metrics
    """
    if n_subjects < 2:
        raise ValueError(f"n_subjects must be >= 2, got {n_subjects}")
    if cl_mean <= 0:
        raise ValueError(f"cl_mean must be > 0, got {cl_mean}")
    if vd_mean <= 0:
        raise ValueError(f"vd_mean must be > 0, got {vd_mean}")
    if ka_mean <= 0:
        raise ValueError(f"ka_mean must be > 0, got {ka_mean}")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if omega_cl < 0:
        raise ValueError(f"omega_cl must be >= 0, got {omega_cl}")
    if omega_vd < 0:
        raise ValueError(f"omega_vd must be >= 0, got {omega_vd}")
    if omega_ka < 0:
        raise ValueError(f"omega_ka must be >= 0, got {omega_ka}")

    lcg = LCG(seed)

    subject_aucs: list[float] = []
    subject_cmaxes: list[float] = []
    subject_cls: list[float] = []
    subject_vds: list[float] = []

    for _ in range(n_subjects):
        # Log-normal inter-individual variability
        eta_cl = omega_cl * lcg.normal()
        eta_vd = omega_vd * lcg.normal()
        eta_ka = omega_ka * lcg.normal()

        cl_i = cl_mean * math.exp(eta_cl)
        vd_i = vd_mean * math.exp(eta_vd)
        ka_i = ka_mean * math.exp(eta_ka)

        auc_i, cmax_i = _simulate_subject_pk(
            dose_mg=dose_mg,
            cl=cl_i,
            vd=vd_i,
            ka=ka_i,
            f_oral=f_oral,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )

        subject_aucs.append(auc_i)
        subject_cmaxes.append(cmax_i)
        subject_cls.append(cl_i)
        subject_vds.append(vd_i)

    # Compute percentiles
    auc_percentiles = compute_percentiles(subject_aucs, [5.0, 50.0, 95.0])
    cmax_percentiles = compute_percentiles(subject_cmaxes, [5.0, 50.0, 95.0])

    # CV% = (SD / mean) * 100
    cl_cv_pct = (_std(subject_cls) / _mean(subject_cls)) * 100.0 if _mean(subject_cls) > 0 else 0.0
    vd_cv_pct = (_std(subject_vds) / _mean(subject_vds)) * 100.0 if _mean(subject_vds) > 0 else 0.0
    auc_cv_pct = (
        (_std(subject_aucs) / _mean(subject_aucs)) * 100.0 if _mean(subject_aucs) > 0 else 0.0
    )

    # Fraction above MEC (based on Cmax)
    fraction_above_mec = 0.0
    if mec is not None:
        above = sum(1 for c in subject_cmaxes if c >= mec)
        fraction_above_mec = above / n_subjects

    notes = (
        f"Population PK: N={n_subjects}, CL_mean={cl_mean} L/h, Vd_mean={vd_mean} L, "
        f"ka_mean={ka_mean} /h, omega_CL={omega_cl}, omega_Vd={omega_vd}, "
        f"omega_ka={omega_ka}, F={f_oral}, seed={seed}"
    )

    return PopulationPKResult(
        n_subjects=n_subjects,
        dose_mg=dose_mg,
        cl_mean=cl_mean,
        vd_mean=vd_mean,
        ka_mean=ka_mean,
        subject_aucs=subject_aucs,
        subject_cmaxes=subject_cmaxes,
        subject_cls=subject_cls,
        subject_vds=subject_vds,
        auc_p5=auc_percentiles[0],
        auc_p50=auc_percentiles[1],
        auc_p95=auc_percentiles[2],
        cmax_p5=cmax_percentiles[0],
        cmax_p50=cmax_percentiles[1],
        cmax_p95=cmax_percentiles[2],
        cl_cv_pct=cl_cv_pct,
        vd_cv_pct=vd_cv_pct,
        auc_cv_pct=auc_cv_pct,
        fraction_above_mec=fraction_above_mec,
        notes=notes,
    )


def population_summary_stats(pop_result: PopulationPKResult) -> dict:
    """Compute summary statistics from a population PK result.

    Returns:
        dict with auc_mean, auc_median, auc_cv, cmax_mean, cmax_median, cmax_cv, n_subjects
    """
    auc_mean = _mean(pop_result.subject_aucs)
    auc_median = compute_percentiles(pop_result.subject_aucs, [50.0])[0]
    auc_cv = (_std(pop_result.subject_aucs) / auc_mean * 100.0) if auc_mean > 0 else 0.0

    cmax_mean = _mean(pop_result.subject_cmaxes)
    cmax_median = compute_percentiles(pop_result.subject_cmaxes, [50.0])[0]
    cmax_cv = (_std(pop_result.subject_cmaxes) / cmax_mean * 100.0) if cmax_mean > 0 else 0.0

    return {
        "auc_mean": auc_mean,
        "auc_median": auc_median,
        "auc_cv": auc_cv,
        "cmax_mean": cmax_mean,
        "cmax_median": cmax_median,
        "cmax_cv": cmax_cv,
        "n_subjects": pop_result.n_subjects,
    }
