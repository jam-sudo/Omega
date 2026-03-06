"""Clinical trial outcome simulation for Omega PBPK."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = [
    "TrialSimResult",
    "_simulate_pk_response",
    "simulate_trial",
    "dose_response_simulation",
]


@dataclass
class TrialSimResult:
    n_subjects: int
    drug_name: str
    dose_mg: float
    primary_endpoint: str
    responder_rate: float
    mean_response: float
    p_value: float
    power: float
    confidence_interval_95: tuple[float, float]
    is_significant: bool
    trial_recommendation: str


def _simulate_pk_response(
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    ec50_mg_L: float,
    emax: float = 1.0,
    n_hill: float = 1.0,
    seed: int = 42,
    n_subjects: int = 100,
    omega_cl: float = 0.3,
    omega_vd: float = 0.2,
) -> np.ndarray:
    """Simulate lognormal population PK and compute Emax response for each subject.

    css = dose * 24 / cl  (steady-state concentration assuming once-daily dosing)
    effect = emax * css^n / (ec50^n + css^n)
    """
    rng = np.random.default_rng(seed)

    # Lognormal individual CL and Vd
    cl_ind = cl_L_per_h * np.exp(rng.normal(0, omega_cl, n_subjects))
    # vd_ind is sampled but not used in css calculation (css depends only on CL)
    _vd_ind = vd_L * np.exp(rng.normal(0, omega_vd, n_subjects))  # noqa: F841

    # Steady-state concentration (daily dose / CL)
    css = (dose_mg * 24.0) / cl_ind  # mg/L  (dose in mg, cl in L/h -> mg/L)

    # Emax model
    css_n = css ** n_hill
    ec50_n = ec50_mg_L ** n_hill
    effect = emax * css_n / (ec50_n + css_n)

    return effect


def simulate_trial(
    drug_name: str,
    dose_mg: float,
    n_subjects: int,
    cl_L_per_h: float,
    vd_L: float,
    ec50_mg_L: float,
    emax: float = 1.0,
    n_hill: float = 1.0,
    placebo_effect: float = 0.1,
    responder_threshold: float = 0.5,
    primary_endpoint: str = "response_rate",
    seed: int = 42,
) -> TrialSimResult:
    """Simulate a clinical trial and return outcomes."""
    effects = _simulate_pk_response(
        dose_mg=dose_mg,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        ec50_mg_L=ec50_mg_L,
        emax=emax,
        n_hill=n_hill,
        seed=seed,
        n_subjects=n_subjects,
    )

    mean_response = float(np.mean(effects))

    # Responder rate: fraction above threshold, net of placebo
    raw_responder_rate = float(np.mean(effects > responder_threshold))
    responder_rate = max(0.0, raw_responder_rate - placebo_effect)

    # One-proportion z-test vs placebo_effect (null: p0 = placebo_effect)
    p0 = placebo_effect
    p_hat = raw_responder_rate
    n = n_subjects

    # Standard error under null
    se_null = np.sqrt(p0 * (1.0 - p0) / n)
    if se_null > 0:
        z_stat = (p_hat - p0) / se_null
    else:
        z_stat = 0.0

    p_value = float(1.0 - stats.norm.cdf(z_stat))  # one-sided

    # 95% CI for observed proportion
    se_obs = np.sqrt(p_hat * (1.0 - p_hat) / n) if n > 0 else 0.0
    z_95 = 1.959964
    ci_lo = float(max(0.0, p_hat - z_95 * se_obs))
    ci_hi = float(min(1.0, p_hat + z_95 * se_obs))
    confidence_interval_95 = (ci_lo, ci_hi)

    # Power estimate via normal approximation
    # beta = P(fail to reject | true p = p_hat)
    alpha = 0.05
    z_alpha = stats.norm.ppf(1.0 - alpha)
    se_alt = np.sqrt(p_hat * (1.0 - p_hat) / n) if n > 0 else 1e-9
    if se_alt > 0:
        z_power = (p_hat - p0) / se_alt - z_alpha * (se_null / se_alt)
        power = float(stats.norm.cdf(z_power))
    else:
        power = 0.0
    power = float(np.clip(power, 0.0, 1.0))

    is_significant = p_value < 0.05

    # Trial recommendation
    if p_value < 0.01 and responder_rate > 0.7:
        trial_recommendation = "stop_efficacy"
    elif p_value < 0.05:
        trial_recommendation = "proceed"
    elif 0.1 < p_value < 0.3:
        trial_recommendation = "enrich"
    else:
        trial_recommendation = "stop_futility"

    return TrialSimResult(
        n_subjects=n_subjects,
        drug_name=drug_name,
        dose_mg=dose_mg,
        primary_endpoint=primary_endpoint,
        responder_rate=responder_rate,
        mean_response=mean_response,
        p_value=p_value,
        power=power,
        confidence_interval_95=confidence_interval_95,
        is_significant=is_significant,
        trial_recommendation=trial_recommendation,
    )


def dose_response_simulation(
    drug_name: str,
    doses: list[float],
    n_subjects: int = 50,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    ec50_mg_L: float = 0.5,
    seed: int = 42,
) -> list[TrialSimResult]:
    """Simulate a dose-response study across multiple doses.

    Returns a list of TrialSimResult sorted by dose (ascending).
    """
    results = []
    for i, dose in enumerate(doses):
        result = simulate_trial(
            drug_name=drug_name,
            dose_mg=dose,
            n_subjects=n_subjects,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            ec50_mg_L=ec50_mg_L,
            seed=seed + i,
        )
        results.append(result)

    results.sort(key=lambda r: r.dose_mg)
    return results
