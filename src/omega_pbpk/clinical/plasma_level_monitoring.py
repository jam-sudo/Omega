"""Plasma level monitoring for therapeutic drug monitoring (TDM) with
Bayesian dose adjustment recommendations."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PlasmaMonitoringResult:
    drug_name: str
    measured_conc_mg_L: float
    sample_time_h: float
    target_range_low_mg_L: float
    target_range_high_mg_L: float
    status: str  # "sub-therapeutic", "therapeutic", "supra-therapeutic"
    percent_deviation: float  # % away from nearest target boundary
    estimated_ke_per_h: float  # back-calculated from population t½
    estimated_cmax_mg_L: float  # extrapolated to t=0
    dose_adjustment_factor: float  # recommended dose multiplier
    recommended_dose_mg: float
    next_sample_time_h: float
    probability_therapeutic: float  # Bayesian P(target achieved)
    notes: str


def _lcg_random(seed: int, n: int) -> list[float]:
    """Linear congruential generator returning n uniform [0,1) values."""
    a = 1664525
    c = 1013904223
    m = 2**32
    vals: list[float] = []
    state = seed & (m - 1)
    for _ in range(n):
        state = (a * state + c) % m
        vals.append(state / m)
    return vals


def _lognormal_sample(mean: float, cv: float, u: float) -> float:
    """Sample from lognormal using Box-Muller-like approach via uniform u in (0,1)."""
    sigma2 = math.log(1 + cv**2)
    mu = math.log(mean) - 0.5 * sigma2
    sigma = math.sqrt(sigma2)
    # Use inverse-CDF approximation (rational approx for normal quantile)
    # Beasley-Springer-Moro approximation for normal quantile
    if u <= 0.0:
        u = 1e-10
    if u >= 1.0:
        u = 1 - 1e-10
    z = _normal_quantile(u)
    return math.exp(mu + sigma * z)


def _normal_quantile(p: float) -> float:
    """Rational approximation of the normal quantile (inverse CDF)."""
    # Abramowitz & Stegun approximation
    if p < 0.5:
        sign = -1.0
        q = p
    else:
        sign = 1.0
        q = 1.0 - p
    if q < 1e-15:
        return sign * 8.0
    t = math.sqrt(-2.0 * math.log(q))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    num = c0 + c1 * t + c2 * t**2
    den = 1 + d1 * t + d2 * t**2 + d3 * t**3
    return sign * (t - num / den)


def monitor_plasma_level(
    drug_name: str,
    measured_conc_mg_L: float,
    sample_time_h: float,
    current_dose_mg: float,
    dosing_interval_h: float = 12.0,
    target_trough_mg_L: float | None = None,
    target_range_low_mg_L: float = 1.0,
    target_range_high_mg_L: float = 5.0,
    population_t_half_h: float = 8.0,
    population_vd_L: float = 50.0,
    cv_pct: float = 30.0,
) -> PlasmaMonitoringResult:
    """Monitor plasma drug concentration and recommend dose adjustment."""
    # Validation
    if measured_conc_mg_L < 0:
        raise ValueError("measured_conc_mg_L must be >= 0")
    if sample_time_h <= 0:
        raise ValueError("sample_time_h must be > 0")
    if current_dose_mg <= 0:
        raise ValueError("current_dose_mg must be > 0")
    if target_range_low_mg_L >= target_range_high_mg_L:
        raise ValueError("target_range_low must be < target_range_high")
    if population_t_half_h <= 0:
        raise ValueError("population_t_half_h must be > 0")
    if not (0 < cv_pct <= 100):
        raise ValueError("cv_pct must be in (0, 100]")

    # Back-calculate ke from population t½
    ke = math.log(2) / population_t_half_h

    # Extrapolate to Cmax (t=0)
    estimated_cmax = measured_conc_mg_L * math.exp(ke * sample_time_h)

    # Determine status
    if measured_conc_mg_L < target_range_low_mg_L:
        status = "sub-therapeutic"
        # % below the low boundary
        percent_deviation = (
            (measured_conc_mg_L - target_range_low_mg_L) / target_range_low_mg_L * 100
        )
    elif measured_conc_mg_L > target_range_high_mg_L:
        status = "supra-therapeutic"
        # % above the high boundary
        percent_deviation = (
            (measured_conc_mg_L - target_range_high_mg_L) / target_range_high_mg_L * 100
        )
    else:
        status = "therapeutic"
        # within range: 0 deviation (use midpoint)
        mid = (target_range_low_mg_L + target_range_high_mg_L) / 2
        percent_deviation = (measured_conc_mg_L - mid) / mid * 100

    # Dose adjustment
    target_mid = (target_range_low_mg_L + target_range_high_mg_L) / 2
    # Use trough target if provided and sample is near trough
    if target_trough_mg_L is not None:
        target_conc_at_time = target_trough_mg_L
    else:
        target_conc_at_time = target_mid

    if measured_conc_mg_L > 0:
        dose_adjustment_factor = target_conc_at_time / measured_conc_mg_L
    else:
        dose_adjustment_factor = 4.0

    # Cap adjustment factor
    dose_adjustment_factor = max(0.25, min(4.0, dose_adjustment_factor))
    recommended_dose_mg = current_dose_mg * dose_adjustment_factor

    # Next sample time: 1-2 t½ after dosing for trough, or at 1 t½ for Cmax
    next_sample_time_h = max(population_t_half_h, dosing_interval_h * 0.8)

    # Bayesian probability: Monte Carlo with LCG (seed=42, 500 samples)
    n_samples = 500
    cv = cv_pct / 100.0
    uniforms = _lcg_random(42, n_samples)
    count_therapeutic = 0
    for u in uniforms:
        ke_i = _lognormal_sample(ke, cv, u)
        # Predicted concentration at sample_time using sampled ke
        c_pred = estimated_cmax * math.exp(-ke_i * sample_time_h)
        if target_range_low_mg_L <= c_pred <= target_range_high_mg_L:
            count_therapeutic += 1
    probability_therapeutic = count_therapeutic / n_samples

    # Build notes
    notes_parts = [
        f"Status: {status}.",
        f"Estimated ke={ke:.4f} h^-1, estimated Cmax={estimated_cmax:.2f} mg/L.",
        f"Dose adjustment factor: {dose_adjustment_factor:.2f}x.",
        f"Recommended dose: {recommended_dose_mg:.1f} mg.",
        f"P(therapeutic): {probability_therapeutic:.2f}.",
    ]
    notes = " ".join(notes_parts)

    return PlasmaMonitoringResult(
        drug_name=drug_name,
        measured_conc_mg_L=measured_conc_mg_L,
        sample_time_h=sample_time_h,
        target_range_low_mg_L=target_range_low_mg_L,
        target_range_high_mg_L=target_range_high_mg_L,
        status=status,
        percent_deviation=percent_deviation,
        estimated_ke_per_h=ke,
        estimated_cmax_mg_L=estimated_cmax,
        dose_adjustment_factor=dose_adjustment_factor,
        recommended_dose_mg=recommended_dose_mg,
        next_sample_time_h=next_sample_time_h,
        probability_therapeutic=probability_therapeutic,
        notes=notes,
    )


def design_monitoring_schedule(
    drug_name: str,
    t_half_h: float,
    dosing_interval_h: float,
    n_samples: int = 3,
) -> list[dict]:
    """Return optimal sampling times for PK characterization (D-optimal design)."""
    if n_samples <= 0:
        return []

    # D-optimal design for 1-compartment: sample around Tmax, mid, and trough
    # Tmax estimate (absorption phase): t_half / 4 to t_half / 2
    # Distribute across the dosing interval
    schedule: list[dict] = []

    if n_samples == 1:
        # Best single sample: at 1 t½ (captures decay)
        schedule.append(
            {
                "time_h": round(t_half_h, 2),
                "rationale": "Single-point estimate near t½",
                "drug_name": drug_name,
            }
        )
    elif n_samples == 2:
        # Early and late
        t1 = max(0.5, t_half_h * 0.5)
        t2 = min(dosing_interval_h * 0.9, t_half_h * 2.0)
        schedule.append(
            {
                "time_h": round(t1, 2),
                "rationale": "Early phase (absorption/distribution)",
                "drug_name": drug_name,
            }
        )
        schedule.append(
            {
                "time_h": round(t2, 2),
                "rationale": "Late phase (elimination)",
                "drug_name": drug_name,
            }
        )
    else:
        # n >= 3: early, mid, trough (D-optimal spread)
        t_early = max(0.5, t_half_h * 0.25)
        t_mid = t_half_h
        t_trough = min(dosing_interval_h * 0.95, t_half_h * 2.5)

        # Distribute remaining samples evenly if n_samples > 3
        base_times = [t_early, t_mid, t_trough]
        if n_samples > 3:
            # Add intermediate points
            step = (t_trough - t_early) / (n_samples - 1)
            base_times = [t_early + i * step for i in range(n_samples)]

        rationales = [
            "Early phase (absorption/distribution)",
            "Mid-phase (Cmax region)",
            "Late phase / trough (elimination)",
        ]
        for i, t in enumerate(base_times[:n_samples]):
            rationale = rationales[i] if i < len(rationales) else f"Intermediate sample {i + 1}"
            schedule.append(
                {
                    "time_h": round(t, 2),
                    "rationale": rationale,
                    "drug_name": drug_name,
                }
            )

    return schedule
