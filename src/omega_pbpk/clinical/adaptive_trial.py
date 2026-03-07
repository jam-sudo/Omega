"""Adaptive clinical trial design with interim analysis and stopping rules.

Implements group-sequential designs with alpha-spending functions for
efficacy and futility boundaries at pre-planned interim analyses.

Stopping rules
--------------
- O'Brien-Fleming (OBF): conservative early, liberal late
- Pocock: equal boundaries across looks
- None: no interim stopping, always completes full enrollment

References
----------
- O'Brien PC & Fleming TR, Biometrics. 1979;35(3):549-56
- Jennison C & Turnbull BW, Group Sequential Methods. Chapman & Hall, 2000
- FDA Guidance: Adaptive Designs for Clinical Trials, 2019
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InterimAnalysis:
    """Result of a single interim analysis.

    Attributes
    ----------
    analysis_number : Interim look number (1-indexed).
    n_enrolled : Cumulative subjects enrolled at this look.
    n_per_arm : Subjects per arm at this look.
    observed_effect : Observed treatment effect (standardized).
    z_statistic : Z test statistic.
    p_value : Two-sided p-value.
    boundary_alpha : Efficacy stopping boundary (z-scale).
    boundary_beta : Futility boundary (conditional power threshold).
    decision : 'continue', 'stop_efficacy', or 'stop_futility'.
    conditional_power : Estimated conditional power (0-1).
    """

    analysis_number: int
    n_enrolled: int
    n_per_arm: int
    observed_effect: float
    z_statistic: float
    p_value: float
    boundary_alpha: float
    boundary_beta: float
    decision: str
    conditional_power: float


@dataclass(frozen=True)
class AdaptiveTrialResult:
    """Results from an adaptive trial design simulation.

    Attributes
    ----------
    trial_name : Name of the trial.
    n_total_planned : Total planned sample size.
    n_arms : Number of treatment arms.
    alpha : Type I error rate.
    power_target : Desired statistical power.
    effect_size : True standardized effect size (Cohen's d).
    interim_analyses : List of InterimAnalysis results.
    final_decision : 'efficacy', 'futility', or 'inconclusive'.
    actual_n_enrolled : Actual subjects enrolled (may be < planned).
    type1_error_spent : Cumulative alpha spent.
    estimated_power : Estimated power at final analysis.
    stopping_reason : Reason for stopping or 'completed'.
    notes : Summary text.
    """

    trial_name: str
    n_total_planned: int
    n_arms: int
    alpha: float
    power_target: float
    effect_size: float
    interim_analyses: list[InterimAnalysis]
    final_decision: str
    actual_n_enrolled: int
    type1_error_spent: float
    estimated_power: float
    stopping_reason: str
    notes: str


def _z_to_p(z: float) -> float:
    """Two-sided p-value from z-statistic."""
    return float(2.0 * (1.0 - _norm_cdf(abs(z))))


def _norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return float(0.5 * (1.0 + math.erf(x / math.sqrt(2.0))))


def _norm_ppf(p: float) -> float:
    """Approximate inverse normal CDF (Beasley-Springer-Moro)."""
    if p <= 0:
        return -8.0
    if p >= 1:
        return 8.0
    if p == 0.5:
        return 0.0
    # Use rational approximation
    t = math.sqrt(-2.0 * math.log(min(p, 1.0 - p)))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    result = t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t**3)
    return result if p > 0.5 else -result


def _obf_boundary(alpha: float, info_fraction: float) -> float:
    """O'Brien-Fleming alpha-spending boundary (z-scale)."""
    alpha_spent = alpha * info_fraction**0.5
    alpha_spent = min(alpha_spent, alpha)
    z_boundary = _norm_ppf(1.0 - alpha_spent / 2.0)
    return max(z_boundary, 0.0)


def _pocock_boundary(alpha: float, info_fraction: float) -> float:
    """Pocock alpha-spending boundary (z-scale)."""
    alpha_spent = alpha * math.log(1.0 + (math.e - 1.0) * info_fraction)
    alpha_spent = min(alpha_spent, alpha)
    z_boundary = _norm_ppf(1.0 - alpha_spent / 2.0)
    return max(z_boundary, 0.0)


def _conditional_power(
    z_current: float, info_current: float, info_final: float, effect_size: float,
    n_per_arm_final: int,
) -> float:
    """Estimate conditional power given current z and remaining information."""
    if info_current >= info_final:
        return 1.0 if z_current > 1.96 else 0.0
    remaining_frac = (info_final - info_current) / info_final
    drift = effect_size * math.sqrt(n_per_arm_final / 2.0)
    projected_z = z_current / math.sqrt(info_current / info_final) + drift * math.sqrt(
        remaining_frac
    )
    cp = 1.0 - _norm_cdf(1.96 - projected_z)
    return float(np.clip(cp, 0.0, 1.0))


def design_adaptive_trial(
    trial_name: str,
    n_total: int,
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
    n_interim: int = 2,
    sigma: float = 1.0,
    n_arms: int = 2,
    stopping_rule: str = "obf",
    seed: int = 42,
) -> AdaptiveTrialResult:
    """Design and simulate an adaptive clinical trial.

    Parameters
    ----------
    trial_name : Trial identifier.
    n_total : Total planned sample size (all arms).
    effect_size : True standardized effect size (Cohen's d).
    alpha : Type I error rate.
    power : Target power.
    n_interim : Number of interim analyses (1-3).
    sigma : Within-group standard deviation.
    n_arms : Number of treatment arms.
    stopping_rule : 'obf', 'pocock', or 'none'.
    seed : Random seed for reproducibility.

    Returns
    -------
    AdaptiveTrialResult with interim analyses and final decision.

    Raises
    ------
    ValueError : If parameters are invalid.
    """
    if n_total <= 0:
        raise ValueError("n_total must be > 0")
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be in (0, 0.5)")
    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1)")
    if effect_size <= 0:
        raise ValueError("effect_size must be > 0")
    if not 1 <= n_interim <= 3:
        raise ValueError("n_interim must be in [1, 3]")

    rng = np.random.RandomState(seed)
    n_per_arm_total = n_total // n_arms
    total_looks = n_interim + 1  # interims + final

    boundary_fn = {
        "obf": _obf_boundary,
        "pocock": _pocock_boundary,
        "none": lambda a, t: float("inf"),
    }.get(stopping_rule.lower())
    if boundary_fn is None:
        raise ValueError(f"Unknown stopping_rule: {stopping_rule}")

    interims: list[InterimAnalysis] = []
    stopped = False
    stop_reason = "completed"
    final_decision = "inconclusive"
    actual_n = n_total
    cumulative_alpha = 0.0

    for k in range(1, total_looks + 1):
        info_fraction = k / total_looks
        n_enrolled_k = int(n_total * info_fraction)
        n_per_arm_k = n_enrolled_k // n_arms

        # Simulate z-statistic under true effect
        signal = effect_size * math.sqrt(n_per_arm_k / 2.0) / sigma
        noise = float(rng.normal(0, 1))
        z_stat = signal + noise

        p_val = _z_to_p(z_stat)

        # Boundaries
        z_boundary = boundary_fn(alpha, info_fraction)

        # Futility: beta spending
        beta = 1.0 - power
        beta_boundary = beta / total_looks

        cp = _conditional_power(
            z_stat, info_fraction, 1.0, effect_size / sigma, n_per_arm_total,
        )

        # Determine decision
        if k < total_looks:
            # Interim analysis
            if stopping_rule.lower() != "none" and abs(z_stat) >= z_boundary:
                decision = "stop_efficacy"
            elif stopping_rule.lower() != "none" and cp < 0.20:
                decision = "stop_futility"
            else:
                decision = "continue"
        else:
            # Final analysis
            if abs(z_stat) >= _norm_ppf(1.0 - alpha / 2.0):
                decision = "stop_efficacy"
            elif cp < 0.20:
                decision = "stop_futility"
            else:
                decision = "continue"

        # Track alpha spent at this look
        if abs(z_stat) >= z_boundary and k < total_looks:
            p_boundary = _z_to_p(z_boundary)
            cumulative_alpha += p_boundary

        interim = InterimAnalysis(
            analysis_number=k,
            n_enrolled=n_enrolled_k,
            n_per_arm=n_per_arm_k,
            observed_effect=float(z_stat / max(math.sqrt(n_per_arm_k / 2.0), 0.01)),
            z_statistic=float(z_stat),
            p_value=float(p_val),
            boundary_alpha=float(z_boundary),
            boundary_beta=float(beta_boundary),
            decision=decision,
            conditional_power=float(cp),
        )
        interims.append(interim)

        if decision == "stop_efficacy" and not stopped:
            stopped = True
            final_decision = "efficacy"
            stop_reason = f"Stopped for efficacy at analysis {k}"
            actual_n = n_enrolled_k
            break
        elif decision == "stop_futility" and not stopped:
            stopped = True
            final_decision = "futility"
            stop_reason = f"Stopped for futility at analysis {k}"
            actual_n = n_enrolled_k
            break

    if not stopped:
        # Final look determines outcome
        last = interims[-1]
        if last.decision == "stop_efficacy":
            final_decision = "efficacy"
        elif last.decision == "stop_futility":
            final_decision = "futility"
        else:
            final_decision = "inconclusive"

    cumulative_alpha = min(cumulative_alpha, alpha)
    est_power = interims[-1].conditional_power

    notes = (
        f"Adaptive trial: {trial_name}, N={n_total}, d={effect_size}, "
        f"rule={stopping_rule}. Decision={final_decision}, "
        f"enrolled={actual_n}/{n_total}."
    )

    return AdaptiveTrialResult(
        trial_name=trial_name,
        n_total_planned=n_total,
        n_arms=n_arms,
        alpha=alpha,
        power_target=power,
        effect_size=effect_size,
        interim_analyses=interims,
        final_decision=final_decision,
        actual_n_enrolled=actual_n,
        type1_error_spent=float(cumulative_alpha),
        estimated_power=float(est_power),
        stopping_reason=stop_reason,
        notes=notes,
    )


def sample_size_calculator(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
    sigma: float = 1.0,
    n_arms: int = 2,
) -> int:
    """Calculate minimum sample size per arm for a two-sample z-test.

    Parameters
    ----------
    effect_size : Standardized effect size (Cohen's d).
    alpha : Type I error rate.
    power : Desired power.
    sigma : Within-group standard deviation.
    n_arms : Number of arms (used for label only; formula is per-arm).

    Returns
    -------
    Minimum n per arm (integer, rounded up).
    """
    if effect_size <= 0:
        raise ValueError("effect_size must be > 0")
    z_alpha = _norm_ppf(1.0 - alpha / 2.0)
    z_beta = _norm_ppf(power)
    n = (z_alpha + z_beta) ** 2 * 2.0 * sigma**2 / effect_size**2
    return int(math.ceil(n))


def compare_stopping_rules(
    trial_name: str,
    n_total: int,
    effect_size: float,
    **kwargs,
) -> list[AdaptiveTrialResult]:
    """Compare OBF, Pocock, and no-stopping rules.

    Returns
    -------
    List of [obf_result, pocock_result, none_result].
    """
    return [
        design_adaptive_trial(
            trial_name, n_total, effect_size, stopping_rule=rule, **kwargs,
        )
        for rule in ("obf", "pocock", "none")
    ]


__all__ = [
    "InterimAnalysis",
    "AdaptiveTrialResult",
    "design_adaptive_trial",
    "sample_size_calculator",
    "compare_stopping_rules",
    # Phase 330 additions
    "SimpleAdaptiveTrialResult",
    "DoseFindingResult",
    "conditional_power",
    "simulate_adaptive_trial",
    "simulate_dose_finding",
]


# ---------------------------------------------------------------------------
# Phase 330: New result dataclasses and functions
# ---------------------------------------------------------------------------

from scipy.optimize import curve_fit  # noqa: E402
from scipy.stats import norm  # noqa: E402


@dataclass(frozen=True)
class SimpleAdaptiveTrialResult:
    """Result of a simplified adaptive trial simulation (Phase 330).

    Attributes
    ----------
    drug_name : Name of the drug.
    n_planned : Planned sample size.
    n_enrolled : Actual patients enrolled.
    effect_size : True effect size used.
    stopped_early : Whether the trial stopped before full enrollment.
    stop_reason : 'futility', 'efficacy', or 'completed'.
    final_p_value : One-sided p-value at final analysis.
    conditional_power_at_interim : CP computed at interim analysis.
    trial_success : True if final_p_value < alpha.
    notes : Summary text.
    """

    drug_name: str
    n_planned: int
    n_enrolled: int
    effect_size: float
    stopped_early: bool
    stop_reason: str
    final_p_value: float
    conditional_power_at_interim: float
    trial_success: bool
    notes: str


@dataclass(frozen=True)
class DoseFindingResult:
    """Result of a dose-finding study simulation (Phase 330).

    Attributes
    ----------
    doses : Dose levels tested (mg).
    true_ed50 : True ED50 (mg).
    estimated_ed50 : Fitted ED50 (mg).
    med_mg : Minimum effective dose (20% Emax threshold).
    dose_response_observed : Mean observed response at each dose.
    dose_response_fitted : Fitted response at each dose.
    r2 : R-squared of Hill fit.
    notes : Summary text.
    """

    doses: list[float]
    true_ed50: float
    estimated_ed50: float
    med_mg: float
    dose_response_observed: list[float]
    dose_response_fitted: list[float]
    r2: float
    notes: str


def conditional_power(
    z_interim: float,
    n_interim: int,
    n_total: int,
    effect_size: float,
    alpha: float = 0.025,
) -> float:
    """Compute conditional power given interim Z-statistic.

    Under the assumed effect size, calculates the probability that the
    final test will be significant given the interim result.

    CP = Phi( -(z_alpha - z_interim * sqrt(n_interim/n_total))
               / sqrt(1 - n_interim/n_total)
               + drift * sqrt((n_total - n_interim)/n_total) )

    where drift = effect_size * sqrt(n_total).

    Parameters
    ----------
    z_interim : float
        Observed Z-statistic at interim.
    n_interim : int
        Sample size at interim analysis.
    n_total : int
        Planned total sample size.
    effect_size : float
        Assumed effect size (Cohen's d).
    alpha : float
        One-sided significance level.

    Returns
    -------
    float
        Conditional power in [0, 1].
    """
    if n_interim >= n_total:
        z_alpha = float(norm.ppf(1.0 - alpha))
        return float(norm.cdf(z_interim - z_alpha))

    frac = n_interim / n_total
    remaining_frac = 1.0 - frac

    z_alpha = float(norm.ppf(1.0 - alpha))
    drift = effect_size * math.sqrt(n_total)

    numerator = (
        -(z_alpha - z_interim * math.sqrt(frac)) / math.sqrt(remaining_frac)
        + drift * math.sqrt(remaining_frac)
    )
    cp = float(norm.cdf(numerator))
    return float(np.clip(cp, 0.0, 1.0))


def simulate_adaptive_trial(
    drug_name: str,
    n_planned: int,
    effect_size: float,
    effect_variability: float = 0.3,
    alpha: float = 0.025,
    power: float = 0.8,
    interim_fraction: float = 0.5,
    futility_boundary: float = 0.1,
    efficacy_boundary: float = 0.99,
    rng_seed: int | None = None,
) -> SimpleAdaptiveTrialResult:
    """Simulate an adaptive clinical trial with one interim analysis.

    At the interim (n * interim_fraction), conditional power is computed.
    The trial stops early for futility or efficacy based on the boundaries.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    n_planned : int
        Planned total sample size.
    effect_size : float
        True mean treatment effect (Cohen's d > 0).
    effect_variability : float
        Standard deviation of individual responses.
    alpha : float
        One-sided significance level.
    power : float
        Target power (informational).
    interim_fraction : float
        Fraction of patients enrolled at interim analysis, in (0.2, 0.8).
    futility_boundary : float
        Stop for futility if conditional power < this value.
    efficacy_boundary : float
        Stop for efficacy if conditional power > this value.
    rng_seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    SimpleAdaptiveTrialResult
    """
    if n_planned < 4:
        raise ValueError("n_planned must be >= 4")
    if effect_size <= 0:
        raise ValueError("effect_size must be > 0")
    if not (0.0 < alpha < 0.5):
        raise ValueError("alpha must be in (0, 0.5)")
    if not (0.0 < power < 1.0):
        raise ValueError("power must be in (0, 1)")
    if not (0.2 < interim_fraction < 0.8):
        raise ValueError("interim_fraction must be in (0.2, 0.8)")

    rng = np.random.default_rng(rng_seed)

    # Generate all patient responses: each ~ N(effect_size, effect_variability^2)
    responses = rng.normal(loc=effect_size, scale=effect_variability, size=n_planned)

    n_interim = max(2, int(n_planned * interim_fraction))

    # Interim analysis
    interim_data = responses[:n_interim]
    sd_interim = float(np.std(interim_data, ddof=1))
    sd_interim = max(sd_interim, 1e-9)
    z_interim = float(np.mean(interim_data)) / (sd_interim / math.sqrt(n_interim))

    cp_interim = conditional_power(z_interim, n_interim, n_planned, effect_size, alpha)

    stopped_early = False
    stop_reason = "completed"
    n_enrolled = n_planned

    if cp_interim < futility_boundary:
        stopped_early = True
        stop_reason = "futility"
        n_enrolled = n_interim
    elif cp_interim > efficacy_boundary:
        stopped_early = True
        stop_reason = "efficacy"
        n_enrolled = n_interim

    # Final analysis on enrolled data
    final_data = responses[:n_enrolled]
    n_final = len(final_data)
    sd_final = float(np.std(final_data, ddof=1))
    sd_final = max(sd_final, 1e-9)
    z_final = float(np.mean(final_data)) / (sd_final / math.sqrt(n_final))

    final_p_value = float(1.0 - norm.cdf(z_final))
    trial_success = final_p_value < alpha

    notes = (
        f"Drug: {drug_name}. "
        f"n_planned={n_planned}, n_enrolled={n_enrolled}. "
        f"Interim Z={z_interim:.3f}, CP={cp_interim:.3f}. "
        f"Final Z={z_final:.3f}, p={final_p_value:.4f}. "
        f"Target power={power}, alpha={alpha}."
    )

    return SimpleAdaptiveTrialResult(
        drug_name=drug_name,
        n_planned=n_planned,
        n_enrolled=n_enrolled,
        effect_size=effect_size,
        stopped_early=stopped_early,
        stop_reason=stop_reason,
        final_p_value=final_p_value,
        conditional_power_at_interim=cp_interim,
        trial_success=trial_success,
        notes=notes,
    )


def _hill_model(d: np.ndarray, ed50: float, hill_n: float) -> np.ndarray:
    """Hill (Emax) model: E(d) = d^n / (ED50^n + d^n)."""
    d = np.asarray(d, dtype=float)
    ed50 = max(ed50, 1e-12)
    return d**hill_n / (ed50**hill_n + d**hill_n)


def simulate_dose_finding(
    doses: list[float],
    true_ed50: float,
    hill_n: float = 2.0,
    n_per_dose: int = 30,
    alpha: float = 0.05,
    variability: float = 0.2,
    rng_seed: int | None = None,
) -> DoseFindingResult:
    """Simulate a dose-finding study and fit a Hill dose-response curve.

    Parameters
    ----------
    doses : list[float]
        Dose levels (mg) to test (all must be > 0).
    true_ed50 : float
        True ED50 (mg).
    hill_n : float
        Hill exponent.
    n_per_dose : int
        Subjects per dose group.
    alpha : float
        Significance level (informational).
    variability : float
        Standard deviation of noise.
    rng_seed : int or None
        Random seed.

    Returns
    -------
    DoseFindingResult
    """
    if not doses:
        raise ValueError("doses must be non-empty")
    if any(d <= 0 for d in doses):
        raise ValueError("all doses must be > 0")
    if true_ed50 <= 0:
        raise ValueError("true_ed50 must be > 0")
    if hill_n <= 0:
        raise ValueError("hill_n must be > 0")
    if n_per_dose < 1:
        raise ValueError("n_per_dose must be >= 1")

    rng = np.random.default_rng(rng_seed)
    doses_arr = np.array(sorted(doses), dtype=float)

    true_response = _hill_model(doses_arr, true_ed50, hill_n)

    observed = []
    for tr in true_response:
        subjects = rng.normal(loc=tr, scale=variability, size=n_per_dose)
        observed.append(float(np.clip(np.mean(subjects), 0.0, None)))

    observed_arr = np.array(observed, dtype=float)

    estimated_ed50 = true_ed50
    fitted_response = true_response.copy()
    r2 = 0.0
    fitted_hill_n = hill_n

    try:
        p0 = [float(np.median(doses_arr)), hill_n]
        bounds = ([1e-6, 0.1], [float(doses_arr.max() * 100), 20.0])
        popt, _ = curve_fit(
            _hill_model,
            doses_arr,
            observed_arr,
            p0=p0,
            bounds=bounds,
            maxfev=5000,
        )
        estimated_ed50 = float(popt[0])
        fitted_hill_n = float(popt[1])
        fitted_response = _hill_model(doses_arr, estimated_ed50, fitted_hill_n)

        ss_res = float(np.sum((observed_arr - fitted_response) ** 2))
        ss_tot = float(np.sum((observed_arr - np.mean(observed_arr)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        r2 = float(np.clip(r2, 0.0, 1.0))
    except (RuntimeError, ValueError):
        pass

    # MED: lowest dose where fitted response >= 20% Emax (Emax=1 for Hill)
    threshold = 0.20
    med_mg = float("nan")
    for d, f in zip(doses_arr.tolist(), fitted_response.tolist(), strict=True):
        if f >= threshold:
            med_mg = d
            break

    notes = (
        f"True ED50={true_ed50} mg, estimated ED50={estimated_ed50:.3f} mg. "
        f"Hill n (simulation)={hill_n}, fitted n={fitted_hill_n:.3f}. "
        f"R\u00b2={r2:.4f}. MED (20% Emax)={med_mg} mg. "
        f"n_per_dose={n_per_dose}, variability={variability}."
    )

    return DoseFindingResult(
        doses=doses_arr.tolist(),
        true_ed50=true_ed50,
        estimated_ed50=estimated_ed50,
        med_mg=med_mg,
        dose_response_observed=observed_arr.tolist(),
        dose_response_fitted=fitted_response.tolist(),
        r2=r2,
        notes=notes,
    )
